"""GPU D2-P execution using read-only completed D1 checkpoints, zero updates."""
import gc
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from model.clip_training import native_clip_contrastive_loss, tensor_gradient_norm, clip_relative_denominator_loss
from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src import clip_train
from src import clip_weight_lr as wl
from src.clip_geometry_run import authenticate_states
from src.clip_source_regime import load_protocol, validate_population, validate_batch_identity, forward
from src.clip_source_resume_store import safe_load, expand
from src.clip_placement_preflight import (ARMS, InvalidEvidence, ContractViolation, auxiliary_path,
    audit_placement, feature_probe, calibration_coefficients, require, finite)
from src.clip_placement_evidence import checksum, checked_path


@contextmanager
def capture_contract_evidence(result):
    try:
        yield
    except ContractViolation as error:
        error.evidence=result
        raise


def serial_tensor(x):
    return x.detach().float().cpu().tolist()


def describe(values,geom,delta):
    result={}
    for arm,v in values.items():
        if v['empty']:
            result[arm]=dict(empty=True,auxiliary_gradient_norm=0.,gradient_cosine=None);continue
        result[arm]={k:serial_tensor(v[k]) for k in ['p_sim','ell_plus','s_aux','e_aux','A','m_loss',
            'tau','r_train','g_logit','epsilon_m','epsilon_scale','P_plus','sigmoid_m']}
        result[arm].update(r_geom=serial_tensor(geom['margins'][arm]),
            delta=delta,target_margin=ARMS[arm]*delta,tau_delta=float(v['tau'])*delta,
            p_geom=serial_tensor(geom['p_geom']),orthogonal_norm=serial_tensor(geom['orthogonal_norm']),
            loss=float(v['loss'].detach()),candidate_dtype=str(v['z'].dtype))
    return result


def norm_record(model,batch,config,record,w,delta,p,coefficients):
    with clip_train._autocast(next(model.parameters()).device,config['training']['mixed_precision']):
        out=forward(model,batch,config)
        values,_=auxiliary_path(out,record,w,delta,p)
        audit_placement(values,delta,p)
        losses={'native':native_clip_contrastive_loss(out.logits),
                **{a:coefficients[a]*values[a]['loss'] for a in ARMS}}
    params=tuple(v for v in model.parameters() if v.requires_grad)
    norms={};native=None;cosines={}
    for i,(arm,loss) in enumerate(losses.items()):
        grads=torch.autograd.grad(loss,params,retain_graph=i<len(losses)-1,allow_unused=True)
        norms[arm]=tensor_gradient_norm(grads)
        if arm=='native':native=grads
        else:
            dot=sum(float((a.detach().double()*b.detach().double()).sum())
                    for a,b in zip(grads,native) if a is not None and b is not None)
            cosines[arm]=None if not norms[arm] or not norms['native'] else dot/(norms[arm]*norms['native'])
    return dict(native_norm=norms['native'],auxiliary_norms={a:norms[a] for a in ARMS},
                native_auxiliary_cosines=cosines,eligible=len(record['eligible_queries']))


def audit_batch(model,opt,sched,batch,config,record,w,common,delta,p):
    """Every record is observational and independently reproducible from its start."""
    result={}
    device=next(model.parameters()).device
    with g.preserved(model,opt,sched) as preservation, capture_contract_evidence(result):
        result["preservation"]=preservation
        g.restore_rng(common);model.train()
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        before=d.state_digest(model.state_dict())
        with torch.no_grad(),clip_train._autocast(device,config['training']['mixed_precision']):
            out=forward(model,batch,config)
            values,geom=auxiliary_path(out,record,w,delta,p)
            # Keep actual path values even when the acceptance assertion fails.
            result['placements']=describe(values,geom,delta)
            try:audit_placement(values,delta,p)
            except ContractViolation as error:
                error.evidence=result
                raise
            for a,v in values.items():
                if v['empty']:continue
                reference=clip_relative_denominator_loss(out.raw_similarity[v['queries']]*out.logit_scale.detach(),v['e_aux'])[0]
                if not torch.equal(reference,v['loss']):raise AssertionError('D1 auxiliary loss wiring differs')
        require(d.state_digest(model.state_dict())==before,'forward_state_unchanged',
                d.state_digest(model.state_dict()),before,'state')
        # Diagnostic on/off invariance at the exact live loss inputs and model state.
        diagnostic_before=g.stamp(model,opt,sched)
        with clip_train._autocast(device,config['training']['mixed_precision']):
            result['gradient_mixed']=feature_probe(out,record,w,delta,p)
        with g.scoring_precision(device):
            result['gradient_fp32']=feature_probe(out,record,w,delta,p,fp32=True)
        with torch.no_grad(),clip_train._autocast(device,config['training']['mixed_precision']):
            after,_=auxiliary_path(out,record,w,delta,p)
        if g.stamp(model,opt,sched)!=diagnostic_before:raise AssertionError('Diagnostics mutated training state')
        for arm in ARMS:
            for key in (['loss'] if values[arm]['empty'] else ['z','s_aux','e_aux','loss']):
                if not torch.equal(values[arm][key],after[arm][key]):raise AssertionError('Diagnostic on/off path differs')
        result['diagnostic_on_off_identical']=True
    result['preservation']=preservation
    return result


def calibration(model,opt,sched,batches,records,config,w,common,delta,p):
    rows=[];verification=[];restorations=[]
    for batch,record in zip(batches,records):
        with g.preserved(model,opt,sched) as audit:
            g.restore_rng(common)
            rows.append(norm_record(model,batch,config,record,w,delta,p,{a:1. for a in ARMS}))
        restorations.append(audit)
    coefficients=calibration_coefficients(rows,p)
    for batch,record in zip(batches,records):
        with g.preserved(model,opt,sched) as audit:
            g.restore_rng(common)
            verification.append(norm_record(model,batch,config,record,w,delta,p,{a:coefficients[a]['coefficient'] for a in ARMS}))
        restorations.append(audit)
    if [x['native_norm'] for x in rows]!=[x['native_norm'] for x in verification]:
        raise AssertionError('Weighted calibration replay changed native gradient')
    for arm in ARMS:
        ratios=[r['auxiliary_norms'][arm]/r['native_norm'] for r in verification]
        mean=statistics.fmean(ratios);error=abs(mean/p['target_ratio']-1)
        require(error<=p['calibration_relative_tolerance'],'weighted_pressure',mean,
                p['target_ratio'],arm+'/calibration')
        treated=[v for v,r in zip(ratios,verification) if r['eligible']]
        coefficients[arm].update(verified_scheduled_ratio=mean,relative_error=error,
                verified_treated_ratio=statistics.fmean(treated) if treated else None)
    return dict(coefficients=coefficients,unweighted=rows,verification=verification,
                preservation=restorations,optimizer_updates=0)


def run(d1, inputs, p, delta, journal):
    """Yield immutable check results; stop at first demonstrated contract violation.

    Remaining checks are explicitly unperformed. FAIL is sufficient to prohibit
    D2, so a known initial failure never spends GPU time calibrating invalid arms.
    """
    from transformers import AutoProcessor
    frozen=load_protocol()
    population=json.loads((d.ROOT/p['population']).read_text())
    if population!=inputs['frozen_population.json']:raise InvalidEvidence('D1 treatment population differs')
    plan=json.loads((d.ROOT/p['plan']).read_text())
    grid=validate_population(population,frozen,plan)
    if inputs['common_branch_plans.json']!=plan['continuation_batches']:raise InvalidEvidence('Continuation plans differ')
    for seed in p['training_seeds']:
        clip_train.set_global_seed(seed)
        cfg=inputs[f'seed_{seed}/training_config.json']
        # Only relocate the authenticated exclusion file; all numerical settings stay unchanged.
        config=json.loads(json.dumps(cfg))
        config['dataset']['diagnostic_holdout_indices']=str((Path(d1)/'training_exclusion_indices.json').resolve())
        if config['training']['mixed_precision']!='bf16':raise InvalidEvidence('Expected D1 bf16 numerical path')
        device,info=clip_train.resolve_device(config['runtime'])
        processor=AutoProcessor.from_pretrained(config['model']['checkpoint'])
        data=clip_train.build_clip_training_data(config,processor,pin_memory=False)
        if data.train_dataset.source_indices!=inputs['split_plan.json']['training']:raise InvalidEvidence('Dataset training identities changed')
        model=d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
        trainable=d.configure_clip_trainable_parameters(model,config['model']['trainable_policy'])
        if (trainable['trainable_parameter_count'],trainable['trainable_tensor_count'])!=(10895617,35):raise InvalidEvidence('Trainable policy differs')
        opt=d.build_clip_optimizer(model,config['optimizer'])
        sched=clip_train.build_scheduler(opt,warmup_steps=config['scheduler']['warmup_steps'],
                                        total_steps=inputs['runtime_protocol.json']['scheduler_horizon'])
        pointer=inputs[f'resume_state/seed_{seed}_trunk.json'];entry=pointer['generations'][0]
        blob=checked_path(d1,'resume_state/'+entry['file'])
        if checksum(blob)!=entry['sha256']:raise InvalidEvidence('Trunk changed after input authentication')
        saved=safe_load(blob)
        if saved['contract']!=pointer['contract'] or saved['key']!=f'seed_{seed}_trunk':raise InvalidEvidence('Decoded trunk identity differs')
        states={w:expand(saved['payload']['snapshots'][str(w)],model) for w in p['checkpoint_steps']}
        del saved
        try:authenticate_states(states,plan['seeds'][str(seed)],p['checkpoint_steps'],inputs[f'seed_{seed}/trunk_audit.json'])
        except AssertionError as error:raise InvalidEvidence('Native starting state authentication failed') from error
        common=states[100]
        groups=[(stream,plan['continuation_batches'][f'{seed}:{stream}']) for stream in p['stream_seeds']]
        groups.append((None,plan['calibration_batches'][str(seed)]))
        for stream,indices in groups:
            flat=[i for batch in indices for i in batch]
            with g.preserved(model,opt,sched):
                batches,identities=d.build_batches(data,flat,seed=seed,epoch=common['epoch'],canonical=False)
            if stream is None:
                expected=inputs[f'seed_{seed}/calibration_batches.json']
                identity=dict(records=identities,caption_epoch=common['epoch'],source_indices=flat,
                              disjoint_from_report_probes_and_continuation_by_image_key=True)
            else:
                expected=inputs[f'seed_{seed}/stream_{stream}/training_batches.json']
                identity=dict(records=identities,caption_epoch=common['epoch'])
            if identity!=expected:raise InvalidEvidence('Processed batch/caption identity differs')
            records=[grid[seed,stream,t] for t in range(len(batches))]
            for offset,record in enumerate(records):
                validate_batch_identity(record,identities[offset*64:(offset+1)*64])
            for w in p['checkpoint_steps']:
                wl.prepare_branch(model,opt,sched,states[w],common,common)
                for offset,(batch,record) in enumerate(zip(batches,records)):
                    key=f'geometry/{seed}/{stream}/{w}/{offset}'
                    context=dict(training_seed=seed,stream_seed=stream,weight_step=w,batch_offset=offset,
                                 population=record['population'],eligible_queries=record['eligible_queries'])
                    cached=journal.get(key)
                    if cached is None:
                        try:
                            result=dict(status='PASS',context=context,
                                evidence=audit_batch(model,opt,sched,batch,config,record,w,common,delta,p))
                        except ContractViolation as error:
                            result=dict(status='FAIL',context=context,violation=error.record,
                                evidence=getattr(error,'evidence',{}))
                        journal.put(key,result)
                    else:result=cached
                    yield key,result
                    if result['status']=='FAIL':return
                if stream is None:
                    key=f'calibration/{seed}/{w}';cached=journal.get(key)
                    if cached is None:
                        try:result=dict(status='PASS',context=dict(training_seed=seed,weight_step=w),
                            evidence=calibration(model,opt,sched,batches,records,config,w,common,delta,p))
                        except ContractViolation as error:result=dict(status='FAIL',context=dict(training_seed=seed,weight_step=w),violation=error.record)
                        journal.put(key,result)
                    else:result=cached
                    yield key,result
                    if result['status']=='FAIL':return
            del batches,identities
            gc.collect()
        del states,common,model,opt,sched,data
        gc.collect();torch.cuda.empty_cache()
