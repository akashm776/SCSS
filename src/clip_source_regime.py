"""D1: frozen common exposures and starting-state sources, evolving embeddings.

No selection, eligibility, coefficient or treatment decision uses reporting data.
"""
from copy import deepcopy
import json
import math
import statistics

import torch
import torch.nn.functional as F

from model.clip_training import native_clip_contrastive_loss, clip_relative_denominator_loss, tensor_gradient_norm
from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as geometry
from src import clip_train
from src import clip_weight_lr as wl
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy

ARMS = ['native'] + [f'{r}_{kind}' for r in 'HMCI' for kind in ['Select','Synth']]
FREEZE_SHA256 = 'c65026efd643e5be5bc8171ae2e2f1c6716b1063433caec82b19a08031d57dc9'


def load_protocol():
    path = d.ROOT/'configs/clip_source_regime_freeze_manifest.json'
    if d.checksum(path) != FREEZE_SHA256:
        raise ValueError('D1 freeze manifest changed')
    freeze = json.loads(path.read_text())
    for name, sha in freeze['files'].items():
        if d.checksum(d.ROOT/name) != sha:
            raise ValueError('Frozen D1 dependency changed: '+name)
    geometry.load_freeze()
    return json.loads((d.ROOT/'configs/clip_source_regime.json').read_text())


def validate_population(population, p, plan):
    grid = {(x['training_seed'],x['stream_seed'],x['batch_offset']):x for x in population['batches']}
    wanted = {(s,u,t) for s in p['training_seeds'] for u in p['stream_seeds'] for t in range(p['horizon'])}
    wanted |= {(s,None,t) for s in p['training_seeds'] for t in range(4)}
    if set(grid) != wanted or len(grid) != len(population['batches']) or population['k'] != 2:
        raise ValueError('Incomplete or duplicate frozen population')
    for (seed,stream,t), record in grid.items():
        batches = plan['calibration_batches'][str(seed)] if stream is None else plan['continuation_batches'][f'{seed}:{stream}']
        if [r['source_index'] for r in record['identities']] != batches[t]:
            raise ValueError('Frozen exposure membership differs')
        if set(record['states']) != {str(w) for w in p['checkpoint_steps']}:
            raise ValueError('Missing source state')
        eligible = [i for i in range(p['batch_size']) if all(record['states'][w]['eligible'][i] for w in map(str,p['checkpoint_steps']))]
        if eligible != record['eligible_queries']:
            raise ValueError('Eligibility is not the cross-state intersection')
        for w in p['checkpoint_steps']:
            validate_record(record, w, p['batch_size'])
    return grid


def validate_record(record, w, size):
    queries = record['eligible_queries']
    if not queries or queries != sorted(set(queries)) or any(i < 0 or i >= size for i in queries):
        raise ValueError('Empty or invalid frozen eligibility mask; no fallback')
    supports = record['states'][str(w)]['supports']
    if set(supports) != set('HMCI'):
        raise ValueError('Missing source regime')
    for regime, rows in supports.items():
        if len(rows) != len(queries):
            raise ValueError('Unmatched eligible support count')
        for i, ids in zip(queries, rows):
            if len(ids) != 2 or len(set(ids)) != 2 or i in ids or any(j < 0 or j >= size for j in ids):
                raise ValueError('Invalid two-source support')
    if any(set(c)&set(i) for c,i in zip(supports['C'],supports['I'])):
        raise ValueError('Frozen C/I supports overlap')


def validate_batch_identity(record, identities):
    actual = [{key:r[key] for key in ['source_index','caption_index','species_id']} for r in identities]
    if actual != record['identities']:
        raise ValueError('Processed D1 batch/caption differs from frozen D0 exposure')


def candidate(out, record, w, arm):
    """Fixed IDs; keep gradients through current features. No argmax/reselection."""
    if arm not in ARMS[1:]:
        raise ValueError('Expected intervention arm')
    validate_record(record,w,len(out.raw_similarity))
    regime,kind = arm.split('_')
    device = out.raw_similarity.device
    queries = torch.tensor(record['eligible_queries'],device=device)
    ids = torch.tensor(record['states'][str(w)]['supports'][regime],device=device)
    if kind == 'Select':
        return out.raw_similarity[queries,ids[:,0]], queries, ids, None
    # Identical uniform matrix-multiply construction family as D0 reference_mixed.
    weights = torch.zeros((len(queries),len(out.image_features)),dtype=out.raw_similarity.dtype,device=device)
    weights.scatter_(1,ids,0.5)
    mu = weights.detach().to(out.image_features.dtype) @ out.image_features
    norms = torch.linalg.vector_norm(mu.detach().float(),dim=-1)
    if not bool(torch.isfinite(mu).all()) or not bool((norms > 1e-6).all()):
        raise ValueError('Invalid/nonfinite near-zero mixture; no fallback')
    z = F.normalize(mu,dim=-1,eps=1e-12)
    return (out.text_features[queries]*z).sum(-1), queries, ids, z


def auxiliary(out, record, w, arm):
    similarity, queries, _, _ = candidate(out,record,w,arm)
    scale = out.logit_scale.detach()
    # Full B-column native denominator; only eligible rows participate in this mean.
    loss = clip_relative_denominator_loss(out.raw_similarity[queries]*scale,similarity*scale)[0]
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite masked auxiliary')
    return loss


@torch.no_grad()
def geometry_record(out, record, w, arm):
    if arm == 'native':
        return dict(eligible_count=len(record['eligible_queries']), frozen_exposure_sha256=reset.digest(record))
    sim, queries, ids, z = candidate(out,record,w,arm)
    regime = arm.split('_')[0]
    weights = torch.zeros((len(queries),len(out.image_features)),device=sim.device,dtype=out.raw_similarity.dtype)
    weights.scatter_(1,ids,0.5)
    mixed_mu = weights.to(out.image_features.dtype) @ out.image_features
    mixed_norms = torch.linalg.vector_norm(mixed_mu.float(),dim=-1)
    with geometry.scoring_precision(out.raw_similarity.device):
        q = geometry.normalized(out.text_features,1e-6,1e-5)
        v = geometry.normalized(out.image_features,1e-6,1e-5)
        positive = (q[queries]*v[queries]).sum(-1)
        source = positive[:,None]-(q[queries,None,:]*v[ids]).sum(-1)
        mu = v[ids].mean(1)
        norms = torch.linalg.vector_norm(mu,dim=-1)
        if not bool(torch.isfinite(mu).all()) or not bool((norms > 1e-6).all()):
            raise ValueError('Invalid ideal diagnostic mixture')
        ideal = F.normalize(mu,dim=-1)
        synthetic = positive-(q[queries]*ideal).sum(-1)
        actual = v[ids[:,0]] if z is None else geometry.normalized(z,1e-6,1e-5)
        actual_margin = positive-(q[queries]*actual).sum(-1)
    values = dict(eligible_queries=record['eligible_queries'],support_ids=ids.tolist(),
        frozen_exposure_sha256=reset.digest(record), eligible_count=len(queries),
        source_margins_fp32=source.tolist(), select_margins_fp32=source[:,0].tolist(),
        uniform_margins_fp32=synthetic.tolist(), ideal_mixture_norms=norms.tolist(),
        mixed_mixture_norms=mixed_norms.tolist(),
        applied_candidate_margins_fp32=actual_margin.tolist(),
        applied_candidate_margins_training=(out.raw_similarity.diagonal()[queries]-sim).float().tolist(),
        auxiliary_candidate_logits=(sim*out.logit_scale.detach()).float().tolist(),
        source_side_fraction=None if regime not in 'CI' else float(((source>0) if regime=='C' else (source<=0)).float().mean()))
    return values


def forward(model,batch,config):
    device=next(model.parameters()).device
    return model(batch['pixel_values'].to(device),clip_train._text_batch(batch,device))


def step_update(model,optimizer,scheduler,batch,config,record,w,arm,coefficient):
    if arm not in ARMS or not math.isfinite(coefficient) or (coefficient <= 0 if arm!='native' else coefficient != 0):
        raise ValueError('Invalid arm/coefficient')
    model.train(); optimizer.zero_grad(set_to_none=True)
    parameters=tuple(v for v in model.parameters() if v.requires_grad)
    with clip_train._autocast(next(model.parameters()).device,config['training']['mixed_precision']):
        out=forward(model,batch,config)
        n=native_clip_contrastive_loss(out.logits)
        a=None if arm=='native' else auxiliary(out,record,w,arm)
        loss=n if a is None else n+coefficient*a
        observed=geometry_record(out,record,w,arm)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,config['optimizer']['max_gradient_norm'],error_if_nonfinite=True)
    rates=[group['lr'] for group in optimizer.param_groups]
    optimizer.step()
    with torch.no_grad(): model.clip_model.logit_scale.clamp_(max=math.log(100.))
    scheduler.step()
    scale=min(1.,config['optimizer']['max_gradient_norm']/(float(norm)+1e-6))
    return dict(native_loss=float(n.detach()),auxiliary_loss=None if a is None else float(a.detach()),
        total_loss=float(loss.detach()),coefficient=coefficient,gradient_norm_before_clip=float(norm),
        clipping_scale=scale,clipping_activated=scale<1.,learning_rates=rates,
        rng_after_sha256=d.state_digest(d.rng_state()),geometry=observed)


def batch_norms(model,batch,config,record,w,coefficients):
    with geometry.preserved(model):
        model.train()
        with clip_train._autocast(next(model.parameters()).device,config['training']['mixed_precision']):
            out=forward(model,batch,config)
            losses={'native':native_clip_contrastive_loss(out.logits),
                    **{a:alpha*auxiliary(out,record,w,a) for a,alpha in coefficients.items()}}
        parameters=tuple(v for v in model.parameters() if v.requires_grad)
        norms={}
        for i,(arm,loss) in enumerate(losses.items()):
            grads=torch.autograd.grad(loss,parameters,retain_graph=i<len(losses)-1,allow_unused=True)
            norms[arm]=tensor_gradient_norm(grads)
            if not math.isfinite(norms[arm]) or norms[arm]<=1e-12:
                raise ValueError('Invalid calibration gradient norm')
            del grads
        return dict(native_norm=norms['native'],auxiliary_norms={a:n for a,n in norms.items() if a!='native'},
            ratios={a:n/norms['native'] for a,n in norms.items() if a!='native'})


def calibrate_states(model,optimizer,scheduler,states,batches,records,config,p,commit):
    if len(batches)!=4 or len(records)!=4: raise ValueError('Expected four calibration batches')
    result={}
    with geometry.preserved(model,optimizer,scheduler):
        for w in p['checkpoint_steps']:
            wl.prepare_branch(model,optimizer,scheduler,states[w],states[p['fixed_lr_checkpoint_step']],states[p['common_rng_source_step']])
            before=geometry.stamp(model,optimizer,scheduler)
            value=dict(status='running',unweighted=[],verification=[],coefficients={},optimizer_updates=0,
                       population_sha256=reset.digest(records),state_before_sha256=before)
            try:
                for batch,record in zip(batches,records):
                    value['unweighted'].append(batch_norms(model,batch,config,record,w,{a:1. for a in ARMS[1:]}))
                    commit(f'calibration_w{w}',value)
                value['mean_unweighted_ratios']={a:statistics.fmean(v['ratios'][a] for v in value['unweighted']) for a in ARMS[1:]}
                value['coefficients']={a:p['target_ratio']/ratio for a,ratio in value['mean_unweighted_ratios'].items()}
                for batch,record in zip(batches,records):
                    value['verification'].append(batch_norms(model,batch,config,record,w,value['coefficients']))
                    commit(f'calibration_w{w}',value)
                if [v['native_norm'] for v in value['unweighted']] != [v['native_norm'] for v in value['verification']]:
                    raise AssertionError('Calibration native gradient replay differs')
                value['verified_mean_ratios']={a:statistics.fmean(v['ratios'][a] for v in value['verification']) for a in ARMS[1:]}
                value['relative_errors']={a:abs(v/p['target_ratio']-1) for a,v in value['verified_mean_ratios'].items()}
                if any(not math.isfinite(v) or v>p['calibration']['relative_tolerance'] for v in value['relative_errors'].values()):
                    raise ValueError('Calibration target tolerance failed; no recalibration')
                value['state_after_sha256']=geometry.stamp(model,optimizer,scheduler)
                if value['state_after_sha256']!=before: raise AssertionError('Calibration changed state')
                value.update(status='complete',training_state_preserved=True)
                commit(f'calibration_w{w}',value)
                result[str(w)]=value
            except BaseException as error:
                value.update(status='failed',error=f'{type(error).__name__}: {error}')
                commit(f'calibration_w{w}',value)
                raise
    return result


def branch_coefficients(calibration,p):
    if calibration.get('status')!='complete' or calibration.get('training_state_preserved') is not True:
        raise ValueError('Incomplete calibration')
    if any(len(calibration[k])!=4 for k in ['unweighted','verification']):
        raise ValueError('Incomplete calibration batches')
    result={'native':0.}
    for arm in ARMS[1:]:
        ratio=statistics.fmean(v['ratios'][arm] for v in calibration['unweighted'])
        alpha=p['target_ratio']/ratio
        measured=statistics.fmean(v['ratios'][arm] for v in calibration['verification'])
        if (not math.isfinite(alpha) or alpha<=0 or alpha!=calibration['coefficients'][arm]
            or not math.isfinite(measured) or abs(measured/p['target_ratio']-1)>p['calibration']['relative_tolerance']):
            raise ValueError('Invalid persisted coefficient/verification')
        result[arm]=alpha
    return result


def optimizer_probe(model,optimizer,scheduler,batch,config,record,w,arm,coefficient):
    """Actual clipped AdamW virtual steps, evolved moments; restore all live state."""
    parameters=tuple(v for v in model.parameters() if v.requires_grad)
    with geometry.preserved(model,optimizer,scheduler) as audit:
        model.train()
        with clip_train._autocast(next(model.parameters()).device,config['training']['mixed_precision']):
            out=forward(model,batch,config)
            n=native_clip_contrastive_loss(out.logits)
            a=None if arm=='native' else coefficient*auxiliary(out,record,w,arm)
            total=n if a is None else n+a
        ng=torch.autograd.grad(n,parameters,retain_graph=True,allow_unused=True)
        ag=() if a is None else torch.autograd.grad(a,parameters,retain_graph=True,allow_unused=True)
        tg=torch.autograd.grad(total,parameters,allow_unused=True)
        nn,an,tn=[tensor_gradient_norm(v) for v in [ng,ag,tg]]
        dot=sum(float((x.float()*y.float()).sum()) for x,y in zip(ng,ag) if x is not None and y is not None)
        base=[v.detach().clone() for v in parameters]
        original_opt=deepcopy(optimizer.state_dict())
        displacements=[]
        for gradients in [ng,tg]:
            with torch.no_grad():
                for v,saved in zip(parameters,base):v.copy_(saved)
            optimizer.load_state_dict(deepcopy(original_opt))
            optimizer.zero_grad(set_to_none=True)
            for v,gradient in zip(parameters,gradients):v.grad=None if gradient is None else gradient.detach().clone()
            torch.nn.utils.clip_grad_norm_(parameters,config['optimizer']['max_gradient_norm'],error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():model.clip_model.logit_scale.clamp_(max=math.log(100.))
            displacements.append([(v.detach()-saved).float() for v,saved in zip(parameters,base)])
        nd=tensor_gradient_norm(displacements[0])
        difference=tensor_gradient_norm([a-n for a,n in zip(displacements[1],displacements[0])])
        clip_scale=min(1.,config['optimizer']['max_gradient_norm']/(tn+1e-6))
        result=dict(native_norm=nn,weighted_aux_norm=an,ratio=an/nn if nn else None,
            cosine=dot/(nn*an) if nn and an else None,total_norm_before_clip=tn,clipping_scale=clip_scale,
            clipping_activated=clip_scale<1.,native_update_norm=nd,intervention_update_difference_norm=difference,
            r_eff=difference/nd if nd else None,zero_native_displacement=nd==0,
            virtual_optimizer_step_calls=2,committed_optimizer_updates=0)
        if any(isinstance(v,float) and not math.isfinite(v) for v in result.values()):
            raise ValueError('Nonfinite optimizer diagnostic')
    result['state_preservation']=audit
    return result


def run_branches(model,optimizer,scheduler,states,batches,records,config,report,*,calibrations,p,
                 commit,save_point=lambda *_:None,progress=lambda _:None,native_reference=None):
    if len(batches)!=p['horizon'] or len(records)!=len(batches) or set(states)!=set(p['checkpoint_steps']):
        raise ValueError('Invalid D1 branch grid')
    source_digest=d.state_digest(states)
    population_digest=reset.digest(records)
    calibration_digest=reset.digest(calibrations)
    frozen=d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad})
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
            points={}; traces=[]; probes={}
            cell['points'][arm]=points; cell['traces'][arm]=traces; cell['pressure_probes'][arm]=probes
            for t in range(len(batches)+1):
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
                traces.append(trace)
                policy.guarded(model,optimizer,scheduler,lambda:commit(f'trace_w{w}_{arm}_{t:03d}',trace))
            rng=[v['rng_after_sha256'] for v in traces]; rates=[v['learning_rates'] for v in traces]
            expected_rng=rng if expected_rng is None else expected_rng
            expected_rates=rates if expected_rates is None else expected_rates
            if rng!=expected_rng or rates!=expected_rates:raise AssertionError('Unmatched branch RNG/LR trajectory')
            if native_reference is not None and arm=='native':
                from src.clip_difficulty import CORE_TRACE_KEYS
                core=[{key:trace[key] for key in CORE_TRACE_KEYS} for trace in traces]
                if reset.digest(core)!=native_reference[str(w)]['native']['traces_sha256']:
                    raise AssertionError('Historical native trace replay mismatch')
            policy.guarded(model,optimizer,scheduler,lambda:commit(f'branch_w{w}_{arm}',
                dict(points=points,traces=traces,pressure_probes=probes,status='complete')))
        cells[f'w{w}']=cell
    if (d.state_digest(states)!=source_digest or reset.digest(records)!=population_digest
        or reset.digest(calibrations)!=calibration_digest or
        frozen!=d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad})):
        raise AssertionError('Frozen source, population, coefficients or parameters changed')
    return dict(cells=cells,population_sha256=population_digest,calibration_sha256=calibration_digest,
        audit=dict(actual_optimizer_steps=len(states)*len(ARMS)*len(batches),source_immutable=True,
            frozen_parameters_unchanged=True,common_population_all_arms_weights=True,coefficients_fixed=True,
            matched_rng_and_lr_all_branches=True,fresh_adamw_all_branches=True,observations_preserve_state=True,
            native_exact_replay=native_reference is not None,reports_control_treatment=False))
