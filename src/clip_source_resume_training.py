"""Resumable execution of unchanged D1 numerical operations."""
from copy import deepcopy
from pathlib import Path
import json
import torch

from src import clip_source_regime as original
from src import clip_checkpoint_diagnostic as d
from src import clip_native_history_policy as native
from src import clip_policy_test as policy
from src import clip_geometry_audit as g
from src import clip_optimizer_reset as reset
from src.clip_source_resume_store import compact,expand,capture,restore,ResumeMismatch


def resumable_trunk(model,opt,sched,batch_epochs,config,p,*,seed,commit,store,interval=25,loader=None):
    key=f'seed_{seed}_trunk'
    saved=store.load(key)
    frozen=d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad})
    if saved is None:
        snapshots={};traces=[]
    else:
        snapshots={int(w):expand(v,model) for w,v in saved['snapshots'].items()};traces=saved['traces']
        # Reconstruct the deterministic loader cursor, including its own Generator.
        # No model forward/backward or optimizer update is used for skipped batches.
        if len(traces)<p['trunk_updates']:
            for trace in traces:
                epoch,batch=next(batch_epochs)
                identity=d.state_digest(dict(source_indices=batch['source_indices'],captions=batch['captions']))
                if epoch!=trace['epoch'] or identity!=trace['batch_identity_sha256']:
                    raise ResumeMismatch('Native loader replay identity differs')
            generator=getattr(loader,'generator',None)
            if generator is not None and saved['loader_rng'] is not None and not torch.equal(generator.get_state(),saved['loader_rng']):
                raise ResumeMismatch('Native loader generator state differs')
        restore(model,opt,sched,saved['live'])
        print('D1_RESUME_TRUNK:',seed,len(traces),'/',p['trunk_updates'],flush=True)
    if p['starting_state_history']!='native_only' or p['trunk_coefficient']!=0 or p['trunk_probes'] or p['trunk_gate_refresh_steps']:
        raise ResumeMismatch('Native preparation contract changed')
    commit('native_history_contract',dict(history='native_only',auxiliary_coefficient=0.,probe_calls=0,
        updates=p['trunk_updates'],checkpoint_steps=p['checkpoint_steps']))
    executed=0
    for completed in range(len(traces),p['trunk_updates']):
        epoch,batch=next(batch_epochs)
        trace=d.step_update(model,opt,sched,batch,config,0.);executed+=1
        store.events.append(dict(event='optimizer_update',phase=key,logical_step=completed+1))
        trace.update(epoch=epoch,completed_updates=completed+1,
            batch_identity_sha256=d.state_digest(dict(source_indices=batch['source_indices'],captions=batch['captions'])))
        traces.append(trace)
        if completed+1 in p['checkpoint_steps']:
            state=native.snapshot(model,opt,sched,epoch=epoch,step=completed+1,seed=seed)
            snapshots[completed+1]=state
            commit(f'start_{completed+1:06d}',dict(training_seed=seed,completed_updates=completed+1,epoch=epoch,
                arm='baseline',starting_state_history='native_only',prior_auxiliary_updates=0,
                state_sha256=d.live_digest(model,opt,sched),snapshot_sha256=d.state_digest(state)))
        if (completed+1)%50==0:commit('progress',dict(completed_updates=completed+1,auxiliary_updates=0))
        if (completed+1)%interval==0 or completed+1 in p['checkpoint_steps'] or completed+1==p['trunk_updates']:
            before=g.stamp(model,opt,sched)
            generator=getattr(loader,'generator',None)
            store.save(key,dict(live=capture(model,opt,sched),traces=traces,
                snapshots={str(w):compact(v,model) for w,v in snapshots.items()},
                loader_rng=None if generator is None else generator.get_state()))
            if g.stamp(model,opt,sched)!=before:raise ResumeMismatch('Checkpoint changed native state')
    if frozen!=d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad}):raise ResumeMismatch('Frozen native weights changed')
    store.events.append(dict(event='native_execution',seed=seed,optimizer_steps_this_attempt=executed,logical_steps=len(traces)))
    return snapshots,dict(actual_optimizer_steps=len(traces),traces=traces,decisions=[],auxiliary_updates=0,
        probe_calls=0,reporting_access=False,frozen_parameters_unchanged=True,history='native_only')

from src.clip_source_regime import (ARMS,branch_coefficients,step_update,optimizer_probe)
from src import clip_weight_lr as wl

def run_branches(model,optimizer,scheduler,states,batches,records,config,report,*,calibrations,p,
                 commit,save_point=lambda *_:None,progress=lambda _:None,native_reference=None,store=None,key=None,interval=10,legacy_folder=None):
    if len(batches)!=p['horizon'] or len(records)!=len(batches) or set(states)!=set(p['checkpoint_steps']):
        raise ValueError('Invalid D1 branch grid')
    source_digest=d.state_digest(states)
    population_digest=reset.digest(records)
    calibration_digest=reset.digest(calibrations)
    frozen=d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad})
    if store is None or key is None:raise ValueError('Resume store/key required')
    context=dict(source=source_digest,population=population_digest,calibration=calibration_digest,
                 protocol=reset.digest(p),config=reset.digest(config))
    saved=store.load(key)
    if saved is not None and saved['context']!=context:raise ResumeMismatch('Continuation checkpoint context differs')
    completed={} if saved is None else saved['branches']
    active=None if saved is None else saved.get('active')
    executed=0
    def checkpoint(branch_key,cell,arm,status):
        nonlocal active
        completed[branch_key]=dict(start=cell['starts'][arm],reset=cell['resets'][arm],
            points=deepcopy(cell['points'][arm]),traces=deepcopy(cell['traces'][arm]),
            pressure_probes=deepcopy(cell['pressure_probes'][arm]),status=status)
        if status=='complete':
            if active is not None and active['branch']==branch_key:active=None
        else:active=dict(branch=branch_key,live=capture(model,optimizer,scheduler))
        live=active
        before=g.stamp(model,optimizer,scheduler)
        store.save(key,dict(context=context,branches=completed,active=live))
        if g.stamp(model,optimizer,scheduler)!=before:raise ResumeMismatch('Checkpoint changed continuation state')
    cells={}; common_identity=None; expected_rng=None; expected_rates=None
    for w in p['checkpoint_steps']:
        coefficients=branch_coefficients(calibrations[str(w)],p)
        cell=dict(weight_step=w,lr_step=p['fixed_lr_checkpoint_step'],points={},traces={},pressure_probes={},
                  coefficients=coefficients,starts={},resets={})
        initial_report=None
        for arm in ARMS:
            cell['resets'][arm]=wl.prepare_branch(model,optimizer,scheduler,states[w],
                states[p['fixed_lr_checkpoint_step']],states[p['common_rng_source_step']])
            start=dict(state_sha256=d.live_digest(model,optimizer,scheduler),weights_sha256=d.state_digest(model.state_dict()),
                optimizer_sha256=d.state_digest(optimizer.state_dict()),scheduler_sha256=d.state_digest(scheduler.state_dict()),
                rng_sha256=d.state_digest(d.rng_state()),learning_rates=[g['lr'] for g in optimizer.param_groups])
            cell['starts'][arm]=start
            if start!=cell['starts']['native']:raise AssertionError('Unmatched branch start')
            identity={k:start[k] for k in ['optimizer_sha256','scheduler_sha256','rng_sha256','learning_rates']}
            common_identity=identity if common_identity is None else common_identity
            if identity!=common_identity:raise AssertionError('Unmatched optimization state across weights')
            commit(f'start_w{w}_{arm}',dict(start=start,reset=cell['resets'][arm],coefficient=coefficients[arm],
                population_sha256=population_digest,calibration_sha256=calibration_digest))
            branch_key=f'w{w}_{arm}'
            previous=completed.get(branch_key)
            if previous is None and legacy_folder is not None:
                prior=Path(legacy_folder)/f'branch_{branch_key}.json'
                if prior.exists():
                    previous=json.loads(prior.read_text())
                    previous.update(start=start,reset=cell['resets'][arm])
                    if previous.get('status')!='complete':raise ResumeMismatch('Incomplete legacy branch marker')
                    print('D1_REUSE_LEGACY_BRANCH:',branch_key,flush=True)
            if previous is not None:
                if previous['start']!=start or previous['reset']!=cell['resets'][arm]:raise ResumeMismatch('Branch starting identity differs')
                points=deepcopy(previous['points']);traces=deepcopy(previous['traces']);probes=deepcopy(previous['pressure_probes'])
                done=previous['status']=='complete'
                if done:
                    if len(traces)!=len(batches) or set(points)!={str(t) for t in p['observation_steps']} or set(probes)!={str(t) for t in p['pressure_probe_offsets']}:
                        raise ResumeMismatch('Incomplete supposedly completed branch')
                else:
                    if active is None or active['branch']!=branch_key:raise ResumeMismatch('Missing active branch state')
                    restore(model,optimizer,scheduler,active['live'])
                    if len(traces)>len(batches):raise ResumeMismatch('Invalid saved update offset')
                    if traces and (scheduler.last_epoch!=p['fixed_lr_checkpoint_step']+len(traces) or
                        any(float(v['step'])!=len(traces) for v in optimizer.state.values())):
                        raise ResumeMismatch('Checkpoint step counters differ')
                if '0' in points:
                    initial_report=points['0']['report'] if initial_report is None else initial_report
                    if points['0']['report']!=initial_report:raise ResumeMismatch('Saved initial report differs')
                print('D1_RESUME_BRANCH:',branch_key,len(traces),'/',len(batches),'complete' if done else 'active',flush=True)
            else:
                points={}; traces=[]; probes={};done=False
            cell['points'][arm]=points; cell['traces'][arm]=traces; cell['pressure_probes'][arm]=probes
            for t in ([] if done else range(len(traces),len(batches)+1)):
                if t in p['observation_steps']:
                    evaluated=policy.guarded(model,optimizer,scheduler,lambda:report.evaluate(model))
                    if t==0:
                        initial_report=evaluated if initial_report is None else initial_report
                        if evaluated!=initial_report:raise AssertionError('Unmatched initial reporting observation')
                    point=dict(report=evaluated,state_sha256=d.live_digest(model,optimizer,scheduler))
                    if native_reference is not None and arm=='native' and point!=native_reference[str(w)]['native']['points'][str(t)]:
                        raise AssertionError('Historical native point replay mismatch')
                    points[str(t)]=point
                    policy.guarded(model,optimizer,scheduler,lambda:save_point(f'w{w}',arm,t,point))
                    progress(f'w{w}/{arm}: {t}/{len(batches)}')
                if t==len(batches):break
                if t in p['pressure_probe_offsets']:
                    probes[str(t)]=optimizer_probe(model,optimizer,scheduler,batches[t],config,records[t],w,arm,coefficients[arm])
                    policy.guarded(model,optimizer,scheduler,lambda:commit(f'pressure_w{w}_{arm}_{t:03d}',probes[str(t)]))
                trace=step_update(model,optimizer,scheduler,batches[t],config,records[t],w,arm,coefficients[arm])
                if (not optimizer.state or any(float(s['step'])!=t+1 for s in optimizer.state.values())
                        or scheduler.last_epoch!=p['fixed_lr_checkpoint_step']+t+1):
                    raise AssertionError('Optimizer counters/schedule differ')
                executed+=1
                store.events.append(dict(event='optimizer_update',phase=key,branch=branch_key,logical_step=t+1))
                traces.append(trace)
                policy.guarded(model,optimizer,scheduler,lambda:commit(f'trace_w{w}_{arm}_{t:03d}',trace))
                if (t+1)%interval==0 or t+1==len(batches):checkpoint(branch_key,cell,arm,'running')
            rng=[v['rng_after_sha256'] for v in traces]; rates=[v['learning_rates'] for v in traces]
            expected_rng=rng if expected_rng is None else expected_rng
            expected_rates=rates if expected_rates is None else expected_rates
            if rng!=expected_rng or rates!=expected_rates:raise AssertionError('Unmatched branch RNG/LR trajectory')
            if native_reference is not None and arm=='native':
                from src.clip_difficulty import CORE_TRACE_KEYS
                core=[{key:trace[key] for key in CORE_TRACE_KEYS} for trace in traces]
                if reset.digest(core)!=native_reference[str(w)]['native']['traces_sha256']:
                    raise AssertionError('Historical native trace replay mismatch')
                if any(points[str(t)]!=native_reference[str(w)]['native']['points'][str(t)] for t in p['observation_steps']):
                    raise ResumeMismatch('Saved native point replay mismatch')
            policy.guarded(model,optimizer,scheduler,lambda:commit(f'branch_w{w}_{arm}',
                dict(points=points,traces=traces,pressure_probes=probes,status='complete')))
            checkpoint(branch_key,cell,arm,'complete')
        cells[f'w{w}']=cell
    if (d.state_digest(states)!=source_digest or reset.digest(records)!=population_digest
        or reset.digest(calibrations)!=calibration_digest or
        frozen!=d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad})):
        raise AssertionError('Frozen source, population, coefficients or parameters changed')
    store.events.append(dict(event='branch_execution',key=key,optimizer_steps_this_attempt=executed,logical_steps=len(states)*len(ARMS)*len(batches)))
    return dict(cells=cells,population_sha256=population_digest,calibration_sha256=calibration_digest,
        audit=dict(actual_optimizer_steps=len(states)*len(ARMS)*len(batches),source_immutable=True,
            frozen_parameters_unchanged=True,common_population_all_arms_weights=True,coefficients_fixed=True,
            matched_rng_and_lr_all_branches=True,fresh_adamw_all_branches=True,observations_preserve_state=True,
            native_exact_replay=native_reference is not None,reports_control_treatment=False))
