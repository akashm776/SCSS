from copy import deepcopy
import ast
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest
import torch
import torch.nn.functional as F

from src import clip_source_regime as s
from src import clip_source_analysis as analysis
from src import clip_geometry_audit as g
from src import clip_checkpoint_diagnostic as d
from src import clip_optimizer_reset as reset
from src import clip_weight_lr as wl
from src import clip_difficulty as b
from tests.test_clip_difficulty import setup


@pytest.fixture(autouse=True)
def isolation():
    rng=deepcopy(d.rng_state()); backend=g.backend_state()
    yield
    g.restore_rng(rng);g.restore_backend(backend)


def toy():
    torch.manual_seed(985)
    model,opt,sched,batch,cfg,states,report,ref=setup()
    records={'eligible_queries':[],'states':{}}
    all_supports={}
    for w in states:
        wl.prepare_branch(model,opt,sched,states[w],states[1],states[1])
        with torch.no_grad():
            out=s.forward(model,batch,cfg)
            q=F.normalize(out.text_features.float(),dim=-1);v=F.normalize(out.image_features.float(),dim=-1)
            supports=g.ordered_supports((q@v.T).numpy(),2)
        all_supports[w]=supports
    records['eligible_queries']=[i for i in range(32) if all(all_supports[w][r][i] is not None for w in states for r in 'CI')]
    for w in states:
        records['states'][str(w)]={'supports':{r:[all_supports[w][r][i] for i in records['eligible_queries']] for r in 'HMCI'}}
    p=deepcopy(s.load_protocol());p.update(checkpoint_steps=[1,4],fixed_lr_checkpoint_step=1,common_rng_source_step=1,
        horizon=4,observation_steps=[0,1,4],pressure_probe_offsets=[0,2],batch_size=32)
    return model,opt,sched,batch,cfg,states,report,records,p


def calibration(args):
    model,opt,sched,batch,cfg,states,report,record,p=args
    return s.calibrate_states(model,opt,sched,states,[batch]*4,[record]*4,cfg,p,lambda *_:None)


def test_frozen_population_complete_and_shared():
    p=s.load_protocol();population=json.loads((d.ROOT/p['population']).read_text())
    plan=json.loads((d.ROOT/p['plan']).read_text())
    grid=s.validate_population(population,p,plan)
    assert len(grid)==312 and population['eligible_exposures']=={'continuation':10667,'calibration_feasibility':416}
    assert population['minimum_eligible_per_batch']==26
    assert p['budget']['total_optimizer_steps']==6900
    one=deepcopy(next(iter(grid.values())))
    s.validate_batch_identity(one,one['identities'])
    one['identities'][0]['caption_index']+=1
    with pytest.raises(ValueError,match='caption'):s.validate_batch_identity(next(iter(grid.values())),one['identities'])


@pytest.mark.parametrize('mutation',['empty','diagonal','duplicate','overlap'])
def test_bad_frozen_supports_fail(mutation):
    *_,record,p=toy();record=deepcopy(record)
    if mutation=='empty':record['eligible_queries']=[]
    else:
        i=record['eligible_queries'][0];supports=record['states']['1']['supports']
        if mutation=='diagonal':supports['H'][0][0]=i
        if mutation=='duplicate':supports['H'][0][1]=supports['H'][0][0]
        if mutation=='overlap':supports['C'][0]=supports['I'][0][:]
    with pytest.raises(ValueError):s.validate_record(record,1,32)


def test_masked_loss_uses_full_partition_and_live_frozen_ids():
    model,opt,sched,batch,cfg,states,report,record,p=toy()
    out=s.forward(model,batch,cfg);q=torch.tensor(record['eligible_queries'])
    for arm in s.ARMS[1:]:
        sim,queries,ids,z=s.candidate(out,record,1,arm)
        expected=F.softplus(sim*out.logit_scale.detach()-torch.logsumexp(out.raw_similarity[q]*out.logit_scale.detach(),dim=1)).mean()
        assert torch.equal(expected,s.auxiliary(out,record,1,arm))
        if arm.endswith('Select'):assert torch.equal(sim,out.raw_similarity[q,ids[:,0]])
    # Auxiliary does not update scale; gradients reach feature parameters.
    grad=torch.autograd.grad(s.auxiliary(out,record,1,'C_Synth'),tuple(v for v in model.parameters() if v.requires_grad),allow_unused=True,retain_graph=True)
    assert grad[0] is None and any(v is not None and torch.count_nonzero(v) for v in grad[1:-1])
    altered=deepcopy(record)
    # Exposure mask is fixed metadata; ineligible rows still participate as denominator columns.
    subset=deepcopy(record);subset['eligible_queries']=record['eligible_queries'][:2]
    for state in subset['states'].values():state['supports']={r:v[:2] for r,v in state['supports'].items()}
    out.raw_similarity.retain_grad();s.auxiliary(out,subset,1,'C_Select').backward()
    outside=[i for i in range(32) if i not in subset['eligible_queries']]
    assert out.raw_similarity.grad[outside].count_nonzero()==0
    assert out.raw_similarity.grad[subset['eligible_queries']][:,outside].count_nonzero()>0
    assert record==altered


def test_select_does_not_reselect_when_ranking_changes():
    model,opt,sched,batch,cfg,states,report,record,p=toy()
    out=s.forward(model,batch,cfg);q=record['eligible_queries'][0]
    first,second=record['states']['1']['supports']['I'][0]
    out.raw_similarity=out.raw_similarity.detach().clone()
    out.raw_similarity[q,first]=-1.;out.raw_similarity[q,second]=1.
    actual,_,ids,_=s.candidate(out,record,1,'I_Select')
    assert ids[0,0]==first and actual[0]==-1.


def test_cancelling_mixture_fails_without_fallback():
    model,opt,sched,batch,cfg,states,report,record,p=toy()
    out=s.forward(model,batch,cfg)
    i,j=record['states']['1']['supports']['H'][0]
    out.image_features=out.image_features.detach().clone();out.image_features[j]=-out.image_features[i]
    with pytest.raises(ValueError,match='mixture'):s.candidate(out,record,1,'H_Synth')


@pytest.mark.parametrize('bf16',[False,True])
def test_calibration_frozen_and_state_preserving(monkeypatch,bf16):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args
    if bf16:monkeypatch.setattr(s.clip_train,'_autocast',lambda *_:torch.autocast('cpu',dtype=torch.bfloat16))
    # Accumulated gradients and heterogeneous module modes must survive.
    for v in model.parameters():v.grad=torch.ones_like(v)
    model.eval();model.clip_model.text_projection.train()
    before=g.stamp(model,opt,sched)
    cal=calibration(args)
    assert g.stamp(model,opt,sched)==before
    for value in cal.values():
        coeff=s.branch_coefficients(value,p)
        assert set(coeff)==set(s.ARMS) and max(value['relative_errors'].values())<=.05
        for arm in s.ARMS[1:]:assert coeff[arm]==.02/value['mean_unweighted_ratios'][arm]
    cal['1']['coefficients']['C_Select']*=2
    with pytest.raises(ValueError):s.branch_coefficients(cal['1'],p)


def test_calibration_failure_preserves_state_and_records_error(monkeypatch):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args
    before=g.stamp(model,opt,sched);saved=[]
    monkeypatch.setattr(s,'tensor_gradient_norm',lambda _:0.)
    with pytest.raises(ValueError,match='gradient norm'):
        s.calibrate_states(model,opt,sched,states,[batch]*4,[record]*4,cfg,p,lambda label,v:saved.append(deepcopy(v)))
    assert saved[-1]['status']=='failed' and g.stamp(model,opt,sched)==before


@pytest.mark.parametrize('w',[1,4])
def test_virtual_probe_matches_actual_paired_steps_and_restores(w):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args
    wl.prepare_branch(model,opt,sched,states[w],states[1],states[1])
    # Probe with evolved moments and active clipping, not just fresh AdamW.
    cfg['optimizer']['max_gradient_norm']=.001
    s.step_update(model,opt,sched,batch,cfg,record,w,'I_Synth',.03)
    before=g.stamp(model,opt,sched)
    probe=s.optimizer_probe(model,opt,sched,batch,cfg,record,w,'C_Select',.04)
    assert g.stamp(model,opt,sched)==before and probe['state_preservation']['preserved']
    actual=[];parameters=tuple(v for v in model.parameters() if v.requires_grad)
    baseline=[v.detach().clone() for v in parameters]
    for arm,alpha in [('native',0.),('C_Select',.04)]:
        with g.preserved(model,opt,sched):
            s.step_update(model,opt,sched,batch,cfg,record,w,arm,alpha)
            actual.append([(v.detach()-old).float() for v,old in zip(parameters,baseline)])
    numerator=s.tensor_gradient_norm([a-n for a,n in zip(actual[1],actual[0])])
    assert probe['r_eff']==pytest.approx(numerator/s.tensor_gradient_norm(actual[0]),rel=1e-5)
    assert probe['clipping_activated'] and probe['virtual_optimizer_step_calls']==2
    native=s.optimizer_probe(model,opt,sched,batch,cfg,record,w,'native',0.)
    assert native['r_eff']==0.


def test_failed_virtual_probe_restores_optimizer_and_everything(monkeypatch):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args
    before=g.stamp(model,opt,sched);original=opt.step
    def fail(*a,**kw):original(*a,**kw);raise RuntimeError('injected virtual failure')
    monkeypatch.setattr(opt,'step',fail)
    with pytest.raises(RuntimeError,match='injected'):s.optimizer_probe(model,opt,sched,batch,cfg,record,1,'H_Select',.02)
    assert g.stamp(model,opt,sched)==before


def test_all_branches_probes_do_not_change_training_or_native_replay():
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args
    cal=calibration(args);native_ref={}
    for w in states:
        wl.prepare_branch(model,opt,sched,states[w],states[1],states[1]);points={};traces=[]
        for t in range(5):
            if t in p['observation_steps']:
                points[str(t)]=dict(report=report.evaluate(model),state_sha256=d.live_digest(model,opt,sched))
            if t<4:traces.append(b.step_update(model,opt,sched,batch,cfg,'native',1.))
        native_ref[str(w)]={'native':dict(points=points,traces_sha256=reset.digest([{k:v[k] for k in b.CORE_TRACE_KEYS} for v in traces]))}
    kwargs=dict(calibrations=cal,p=p,commit=lambda *_:None,native_reference=native_ref)
    row=s.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs)
    sparse=deepcopy(p);sparse['pressure_probe_offsets']=[]
    other=s.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**{**kwargs,'p':sparse})
    for key,cell in row['cells'].items():
        assert cell['traces']==other['cells'][key]['traces'] and cell['points']==other['cells'][key]['points']
    assert row['audit']['actual_optimizer_steps']==72 and row['audit']['native_exact_replay']
    # Complete-grid summarizer, state interactions and nested seed averaging.
    p.update(training_seeds=[1,2,3],stream_seeds=[4,5]);p['budget'].update(branch_optimizer_steps=432,trunk_optimizer_steps=12)
    rows=[]
    for seed in p['training_seeds']:
        for stream in p['stream_seeds']:
            r=deepcopy(row);r.update(training_seed=seed,stream_seed=stream)
            # Known Q: +1 early, -2 late. Seed+stream offsets cancel in paired contrasts.
            for w in [1,4]:
                for t in p['observation_steps']:
                    for arm in s.ARMS:
                        report_value=r['cells'][f'w{w}']['points'][arm][str(t)]['report']
                        report_value['native_loss']=100+seed+stream+((1 if w==1 else -2) if arm=='C_Select' else 0)*t/4
                        report_value['pool_retrieval']={'mean_r1_percent':10.}
            rows.append(r)
    result=analysis.summarize(rows,p)
    assert result['primary_value']==-3. and result['timecourse']['4']['contrasts']['directional_loss_crossover']
    with pytest.raises(ValueError,match='grid'):analysis.summarize(rows[:-1],p)
    rows[1]['cells']['w1']['coefficients']['C_Select']*=2
    with pytest.raises(ValueError):analysis.summarize(rows,p)


def test_delivery_and_prior_run_guard(tmp_path):
    from colabs.run_clip_source_regime import require_no_previous_run
    require_no_previous_run(tmp_path,'clip_source_regime_v1')
    folder=tmp_path/'clip_source_regime_partial';folder.mkdir()
    (folder/'run_manifest.json').write_text(json.dumps({'experiment':'clip_source_regime_v1'}))
    with pytest.raises(RuntimeError,match='no automatic resume'):require_no_previous_run(tmp_path,'clip_source_regime_v1')
    path=d.ROOT/'colabs/clip_source_regime_drive_one_cell.py'
    if not path.exists():pytest.skip('Generated cell absent in source overlay')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==values['BUNDLE_SHA256']
    sources=json.loads(raw);manifest=json.loads(sources.pop('source_regime_source_manifest.json'))
    assert set(sources)==set(manifest)
    for name,value in sources.items():
        assert hashlib.sha256(value.encode()).hexdigest()==manifest[name]
        assert (d.ROOT/name).read_text()==value


def test_complete_d1_runner_persists_full_grid(tmp_path,monkeypatch):
    import transformers
    from src import clip_source_run as runner
    from src import clip_native_history_policy as native
    from src import clip_train
    from src import clip_policy_test as policy
    from src import clip_pressure as pressure
    from src.clip_fresh_split_policy import training_batches, FullPoolReport
    from tests.test_clip_fresh_split_policy import actual_split
    from tests.test_clip_paired_updates import TinyCLIP
    _,_,_,split=actual_split();seed=271828;stream=2026100101
    prep=deepcopy(b.load_protocol())
    prep.update(training_seeds=[seed],stream_seeds=[stream],checkpoint_steps=[1,4],trunk_updates=4,
        horizon=2,batch_size=32,expected_batches_per_epoch=2,common_stream_checkpoint_step=1)
    torch.manual_seed(410)
    batch=dict(pixel_values=torch.randn(32,6),input_ids=torch.randn(32,6),source_indices=list(range(32)),captions=['toy']*32)
    cfg=dict(experiment={},training={'seed':seed,'mixed_precision':'none'},runtime={},dataset={},
        model={'checkpoint':'toy','trainable_policy':'toy'},optimizer={'max_gradient_norm':1.},scheduler={'warmup_steps':2})
    dataset=SimpleNamespace(source_indices=split['training'],set_epoch=lambda _:None,
        grouped_split=SimpleNamespace(groups=[SimpleNamespace(image_key=f'image_{i}') for i in range(5994)]))
    data=SimpleNamespace(train_dataset=dataset,train_loader=[batch,batch])
    def build(data,indices,*,seed,epoch,canonical):
        records=[dict(source_index=i,image_key=f'image_{i}',caption='toy',caption_index=0,species_id=i%7) for i in indices]
        return [batch for _ in range(0,len(indices),32)],records
    def opt_builder(model,_):return torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001)
    def backend(_):return TinyCLIP()
    # Independently generate the tiny native reference before the audited run.
    clip_train.set_global_seed(seed)
    model=backend(None);opt=opt_builder(model,None)
    sched=clip_train.build_scheduler(opt,warmup_steps=2,total_steps=prep['scheduler_horizon'])
    starts={}
    states,trunk=native.trunk(model,opt,sched,training_batches(data),cfg,prep,seed=seed,
        commit=lambda name,v:starts.update({str(v['completed_updates']):deepcopy(v)}) if name.startswith('start_') else None)
    _,records=build(data,split['report'],seed=0,epoch=0,canonical=True)
    ids=[hashlib.sha256(r['image_key'].encode()).hexdigest() for r in records]
    roles=dict(report=records,train_source_indices=split['training'],row_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        image_key_sha256=ids,disjoint_by_image_key=True,historically_unused_images_claimed=False,original_validation_not_evaluated=True)
    plans,_=policy.make_plans(split['training'],prep)
    ref={n:reset.digest(v) for n,v in [('split_plan.json',split),('branch_plans.json',plans),('training_exclusion_indices.json',sorted(set(split['old_holdout'])|set(split['report'])))]}
    ref['seeds']={str(seed):{'starts':starts,'data_roles.json':reset.digest(roles),
        'training_batches.json':reset.digest({'probe_identity':'toy'}),'trunk_audit.json':reset.digest(trunk)}}
    common=wl.common_plans(split['training'],prep)
    used={i for batches in common.values() for bs in batches for i in bs}
    ci=sorted(set(split['training'])-used-set(split['frozen_probes']))[:128]
    _,cr=build(data,ci,seed=seed,epoch=states[1]['epoch'],canonical=False)
    cal_identity=dict(records=cr,caption_epoch=states[1]['epoch'],source_indices=ci,disjoint_from_report_probes_and_continuation_by_image_key=True)
    _,tr=build(data,[i for bs in common[f'{seed}:{stream}'] for i in bs],seed=seed,epoch=states[1]['epoch'],canonical=False)
    plan=dict(continuation_batches=common,calibration_batches={str(seed):[ci[i:i+32] for i in range(0,128,32)]},seeds={str(seed):dict(
        native_trunk_audit_canonical_sha256=reset.digest(trunk),
        states={str(w):dict(archived_start=starts[str(w)],weights_sha256=d.state_digest(states[w]['model'])) for w in [1,4]},
        streams={str(stream):dict(training_batch_records_sha256=reset.digest(dict(records=tr,caption_epoch=states[1]['epoch'])))},
        calibration_batch_records_canonical_sha256=reset.digest(cal_identity))})
    p=deepcopy(s.load_protocol())
    p.update(training_seeds=[seed],stream_seeds=[stream],checkpoint_steps=[1,4],horizon=2,batch_size=32,
        fixed_lr_checkpoint_step=1,common_rng_source_step=1,observation_steps=[0,1,2],pressure_probe_offsets=[0,1],
        trunk_updates=4,expected_batches_per_epoch=2)
    p['budget'].update(trunk_optimizer_steps=4,branch_optimizer_steps=36,total_optimizer_steps=40)
    # Use the production deterministic calibration-plan algorithm on toy membership.
    p['calibration']['batch_size']=32
    calplan=pressure.calibration_plan(split,common,p)
    calindices=[i for bs in calplan[str(seed)]['batches'] for i in bs]
    _,cr=build(data,calindices,seed=seed,epoch=states[1]['epoch'],canonical=False)
    plan['calibration_batches'][str(seed)]=calplan[str(seed)]['batches']
    plan['seeds'][str(seed)]['calibration_batch_records_canonical_sha256']=reset.digest(dict(records=cr,
        caption_epoch=states[1]['epoch'],source_indices=calindices,disjoint_from_report_probes_and_continuation_by_image_key=True))
    support_states={}
    for w in [1,4]:
        wl.prepare_branch(model,opt,sched,states[w],states[1],states[1])
        with torch.no_grad():
            out=s.forward(model,batch,cfg)
            supports=g.ordered_supports((F.normalize(out.text_features,dim=-1)@F.normalize(out.image_features,dim=-1).T).numpy(),2)
        support_states[str(w)]=dict(supports=supports,eligible=[supports['C'][i] is not None and supports['I'][i] is not None for i in range(32)])
    eq=[i for i in range(32) if all(state['eligible'][i] for state in support_states.values())]
    for state in support_states.values():state['supports']={r:[ids[i] for i in eq] for r,ids in state['supports'].items()}
    population=dict(k=2,batches=[])
    for u,batch_indices in [(stream,common[f'{seed}:{stream}']),(None,plan['calibration_batches'][str(seed)])]:
        for t,indices in enumerate(batch_indices):
            _,identities=build(data,indices,seed=seed,epoch=states[1]['epoch'],canonical=False)
            population['batches'].append(dict(training_seed=seed,stream_seed=u,batch_offset=t,
                eligible_queries=eq,states=deepcopy(support_states),
                identities=[{k:r[k] for k in ['source_index','caption_index','species_id']} for r in identities]))
    for name,value in [('population',population),('plan',plan),('calplan',dict(seeds=calplan))]:
        d.write_json(tmp_path/(name+'.json'),value)
    p.update(population=str(tmp_path/'population.json'),plan=str(tmp_path/'plan.json'))
    p['calibration']['plan']=str(tmp_path/'calplan.json')
    ref['pairs']={f'{seed}:1:{stream}':dict(training_batches_sha256=reset.digest(dict(records=tr,caption_epoch=states[1]['epoch'])))}
    report_batches,_=build(data,split['report'],seed=0,epoch=0,canonical=True)
    report=FullPoolReport(report_batches,512)
    continuation_ref={'pairs':{f'{seed}:{stream}':{'weights':{}}}}
    for w in [1,4]:
        wl.prepare_branch(model,opt,sched,states[w],states[1],states[1]);points={};traces=[]
        for t in range(3):
            points[str(t)]=dict(report=report.evaluate(model),state_sha256=d.live_digest(model,opt,sched))
            if t<2:traces.append(b.step_update(model,opt,sched,batch,cfg,'native',1.))
        continuation_ref['pairs'][f'{seed}:{stream}']['weights'][str(w)]={'native':dict(points=points,
            traces_sha256=reset.digest([{k:v[k] for k in b.CORE_TRACE_KEYS} for v in traces]))}
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *_:None)
    monkeypatch.setattr(clip_train,'load_training_config',lambda *_:deepcopy(cfg))
    monkeypatch.setattr(clip_train,'resolve_device',lambda *_:(torch.device('cpu'),{'device':'cpu'}))
    monkeypatch.setattr(clip_train,'build_clip_training_data',lambda *_args,**_kw:data)
    monkeypatch.setattr(d,'build_batches',build)
    monkeypatch.setattr(d,'CLIPEncoderBackend',SimpleNamespace(from_pretrained=backend))
    monkeypatch.setattr(d,'configure_clip_trainable_parameters',lambda *_:dict(trainable_parameter_count=10895617,trainable_tensor_count=35))
    monkeypatch.setattr(d,'build_clip_optimizer',opt_builder)
    monkeypatch.setattr(runner,'fixed_probe_batches',lambda _data,_p,folder:d.write_json(folder/'training_batches.json',{'probe_identity':'toy'}))
    original_step=s.step_update
    def checked_step(*args,**kwargs):
        assert (destination/f'seed_{seed}/calibration.json').exists()
        return original_step(*args,**kwargs)
    monkeypatch.setattr(s,'step_update',checked_step)
    destination=tmp_path/'d1';destination.mkdir();synced=[]
    summary=runner.run(destination,lambda:synced.append(1),p=p,native_reference=ref,reference=continuation_ref)
    done=json.loads((destination/'completion.json').read_text())
    assert summary['optimizer_steps']==40 and done['native_continuation_exact_replay']
    assert len(list(destination.rglob('features_*.npz')))==18
    assert len(list(destination.rglob('point_*.json')))==54
    assert len(list(destination.rglob('trace_*.json')))==36
    assert len(list(destination.rglob('pressure_w*.json')))==36
    assert not list(destination.rglob('*.pt')) and synced
    assert (destination/'REPORT.md').exists()
