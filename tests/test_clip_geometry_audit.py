from copy import deepcopy
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import clip_geometry_audit as g
from src import clip_geometry_run as runner
from src.clip_geometry_summary import Accumulator, Distribution, Summary
from src import clip_checkpoint_diagnostic as d
from src import clip_train
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy
from src import clip_weight_lr as wl
from src.clip_fresh_split_policy import training_batches
from tests.test_clip_paired_updates import TinyCLIP
from tests.test_clip_fresh_split_policy import actual_split


@pytest.fixture(autouse=True)
def isolated():
    before=deepcopy(d.rng_state());backend=g.backend_state()
    yield
    g.restore_rng(before);g.restore_backend(backend)


def fixture():
    torch.manual_seed(871)
    model=TinyCLIP();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    batch=dict(pixel_values=torch.randn(32,6),input_ids=torch.randn(32,6))
    ids=[dict(source_index=i,caption_index=i%3,species_id=i%4) for i in range(32)]
    context=dict(population='continuation',training_seed=1,stream_seed=2,batch_offset=0,weight_step=100)
    return model,opt,sched,batch,ids,context,{'training':{'mixed_precision':'none'}},g.load_protocol()


def rows_fixture():
    model,opt,sched,batch,ids,ctx,cfg,p=fixture()
    rows,audit=g.audit_batch(model,batch,ids,ctx,d.rng_state(),cfg,p,opt,sched)
    return rows,p


def test_frozen_contract_and_budget():
    p=g.load_protocol();assert p['budget']['total_batch_forwards']==624
    assert p['budget']['total_query_exposures']==39936
    plan=json.loads((d.ROOT/p['plan']).read_text())
    assert len(plan['continuation_batches'])==6
    assert all(len(b)==50 and all(len(x)==64 for x in b) for b in plan['continuation_batches'].values())


@pytest.mark.parametrize('name',['configs/clip_geometry_audit.json','docs/clip-geometry-audit.md','configs/clip_geometry_audit_plan.json'])
def test_freeze_rejects_mutation(tmp_path,monkeypatch,name):
    manifest=g.load_freeze()
    for f in ['configs/clip_geometry_audit_freeze_manifest.json',*manifest['files'],*manifest['parent_evidence_files'],*manifest['reference_source_files']]:
        dest=tmp_path/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((d.ROOT/f).read_bytes())
    path=tmp_path/name;path.write_bytes(path.read_bytes()+b'\n')
    monkeypatch.setattr(d,'ROOT',tmp_path)
    with pytest.raises(g.AuditFailure,match='changed'):g.load_protocol()


def test_strict_signs_ties_and_support_order():
    scores=np.zeros((32,32),dtype=np.float32)
    scores[0,0]=.5;scores[0,1:5]=.5;scores[0,5:10]=.6;scores[0,10:]=.4
    s=g.ordered_supports(scores,3)
    assert s['H'][0]==[5,6,7]
    assert s['I'][0]==[1,2,3] # exact ties count as nonpositive
    assert s['C'][0]==[10,11,12]
    assert s['M'][0]==[16,17,18]
    scores[0,0]=1.
    s=g.ordered_supports(scores,8)
    assert s['I'][0] is None and s['C'][0]==s['H'][0]


def test_geometry_invariants_missingness_and_json():
    rows,p=rows_fixture();json.dumps(rows,allow_nan=False)
    for r in rows:
        assert r['N_positive']+r['N_nonpositive']==31
        for k,e in r['sizes'].items():
            for name,s in e['regimes'].items():
                if not s['feasible']:
                    assert s['support'] is None and s['candidate'] is None
                    continue
                assert len(s['support'])==int(k) and r['query_index'] not in s['support']
                c=s['candidate']
                if c['ideal']['valid']:
                    if name=='I' and r['positive_cosine']>=0:
                        assert c['I_invariant']=='passed'
                        assert c['ideal']['margin']<=c['real']['ideal_margin']+1e-5
                    if k=='1':assert c['ideal']['margin']==pytest.approx(c['real']['ideal_margin'],abs=1e-5)
            assert e['overlaps']['CI'] is None or e['overlaps']['CI']['fraction']==0


def test_cancelling_mixture_retains_sources_and_real_candidate():
    images=torch.tensor([[1.,0.],[-1.,0.]]*16)
    text=torch.tensor([[0.,1.]]*32)
    out=SimpleNamespace(image_features=images,text_features=text,raw_similarity=text@images.T)
    ids=[dict(source_index=i,caption_index=0,species_id=i) for i in range(32)]
    rows,_=g.geometry(out,ids,{},g.load_protocol())
    selected=rows[2]['sizes']['2']['regimes']['H']
    assert selected['feasible'] and selected['support']==[0,1]
    assert selected['candidate']['real']['ideal_margin']==0
    assert not selected['candidate']['ideal']['valid']
    assert selected['candidate']['ideal']['margin'] is None
    assert rows[2]['sizes']['2']['overlaps']['HI'] is not None


def test_I_invariant_failure_and_negative_positive_exception():
    p=g.load_protocol()
    record=dict(k=2,regime='I',positive_cosine=.2,real={'ideal_margin':-.1},ideal={'valid':True,'margin':-.01})
    with pytest.raises(g.AuditFailure,match='I-Synth'):g.assert_candidate_invariants(record,p)
    record['positive_cosine']=-.2
    g.assert_candidate_invariants(record,p)
    assert record['I_invariant']=='not_applicable'


@pytest.mark.parametrize('exception',[False,True])
def test_complete_restoration_including_failure(exception):
    m,o,s,batch,ids,ctx,cfg,p=fixture()
    next(m.parameters()).grad=torch.ones_like(next(m.parameters()))
    m.eval();m.clip_model.visual_projection.train()
    before=g.stamp(m,o,s)
    def operation():
        with g.preserved(m,o,s) as audit:
            with torch.no_grad():next(m.parameters()).add_(2)
            next(m.parameters()).grad=None
            m.train();torch.rand(3);torch.set_float32_matmul_precision('medium')
            o.param_groups[0]['lr']=.9;s.last_epoch=17
            if exception:raise RuntimeError('deliberate failure')
        return audit
    if exception:
        with pytest.raises(RuntimeError):operation()
    else:assert operation()['preserved']
    assert g.stamp(m,o,s)==before


@pytest.mark.parametrize('bf16',[False,True])
def test_audit_forbids_training_and_is_repeatable(monkeypatch,bf16):
    m,o,s,batch,ids,ctx,cfg,p=fixture();rng=d.rng_state()
    def prohibited(*a,**k):pytest.fail('Audit invoked a prohibited operation')
    monkeypatch.setattr(o,'step',prohibited);monkeypatch.setattr(s,'step',prohibited)
    monkeypatch.setattr(torch.Tensor,'backward',prohibited)
    monkeypatch.setattr(torch.autograd,'grad',prohibited)
    if bf16:monkeypatch.setattr(clip_train,'_autocast',lambda device,*_:torch.autocast(device.type,dtype=torch.bfloat16))
    before=g.stamp(m,o,s)
    a,info=g.audit_batch(m,batch,ids,ctx,rng,cfg,p,o,s)
    b,_=g.audit_batch(m,batch,ids,ctx,rng,cfg,p,o,s)
    assert a==b and g.stamp(m,o,s)==before
    assert info['selection_dtype']=='torch.float32' and info['preserved']
    if bf16:assert info['raw_similarity_dtype']=='torch.bfloat16'


def test_forward_mutation_failure_restores_buffers_and_weights(monkeypatch):
    m,o,s,batch,ids,ctx,cfg,p=fixture();before=g.stamp(m,o,s);original=m.forward
    def bad(*args):
        next(m.parameters()).add_(1)
        return original(*args)
    monkeypatch.setattr(m,'forward',bad)
    with pytest.raises(g.AuditFailure,match='mutated'):g.audit_batch(m,batch,ids,ctx,d.rng_state(),cfg,p,o,s)
    assert g.stamp(m,o,s)==before


def test_nonfinite_forward_stops():
    m,o,s,batch,ids,ctx,cfg,p=fixture();batch['pixel_values'][0,0]=float('nan')
    before=g.stamp(m,o,s)
    with pytest.raises(g.AuditFailure,match='Invalid forward'):g.audit_batch(m,batch,ids,ctx,d.rng_state(),cfg,p,o,s)
    assert g.stamp(m,o,s)==before


def test_summary_missing_denominators_and_nested_mean():
    p=g.load_protocol();dist=Distribution()
    dist.add([1.,3.],(1,1));dist.add([None],(1,2));dist.add([10.],(2,1))
    result=dist.finish(p,[(1,1),(1,2),(2,1)])
    assert result['macro_mean'] is None and result['coverage']==.75
    assert result['pooled_mean']==pytest.approx(14/3)
    dist.add([6.],(1,2))
    result=dist.finish(p,[(1,1),(1,2),(2,1)])
    assert result['macro_mean']==7. # ((2+6)/2 + 10)/2
    empty=Distribution().finish(p,[(1,1)])
    assert empty['coverage'] is None and empty['pooled_mean'] is None


def test_paired_summary_and_primary_not_duplicated():
    rows,p=rows_fixture();p=deepcopy(p)
    p.update(training_seeds=[1],stream_seeds=[2],batch_size=32)
    p['budget'].update(total_batch_forwards=2,total_query_exposures=64)
    p['populations']['extended']['offsets']=[0]
    p['populations']['calibration_feasibility']['batch_offsets']=[]
    late=deepcopy(rows)
    for r in late:r['weight_step']=500
    summary=Summary(p);summary.add_batch_pair({100:rows,500:late})
    result=summary.finish()
    with pytest.raises(g.AuditFailure,match='Duplicate'):summary.add_batch_pair({100:rows,500:late})
    assert result['query_exposures']==64
    groups={(x['population'],x['weight'],x['seed'],x['stream']):x for x in result['groups']}
    assert groups['primary','100','all','all']['query_exposures']==32
    assert groups['extended','100','all','all']['query_exposures']==32
    paired=groups['extended','paired','all','all']
    for k in p['support_sizes']:
        assert paired['metrics'][f'k{k}/eligibility/early_only']['pooled_mean']==0
        assert paired['metrics'][f'k{k}/eligibility/late_only']['pooled_mean']==0


def test_delivery_and_run_guard(tmp_path):
    from colabs.run_clip_geometry_audit import require_no_previous_run
    require_no_previous_run(tmp_path,'clip_geometry_audit_v1')
    folder=tmp_path/'clip_geometry_audit_old';folder.mkdir()
    (folder/'run_manifest.json').write_text(json.dumps({'experiment':'clip_geometry_audit_v1'}))
    with pytest.raises(RuntimeError):require_no_previous_run(tmp_path,'clip_geometry_audit_v1')
    path=d.ROOT/'colabs/clip_geometry_audit_drive_one_cell.py'
    if not path.exists():pytest.skip('Delivery cell excluded from source overlay')
    vals={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(vals['PAYLOAD']));assert hashlib.sha256(raw).hexdigest()==vals['BUNDLE_SHA256']
    sources=json.loads(raw);pins=json.loads(sources.pop('geometry_source_manifest.json'))
    assert set(sources)==set(pins)
    for name,source in sources.items():
        assert source==(d.ROOT/name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==pins[name]


def test_complete_toy_run_preparation_loading_and_no_audit_updates(tmp_path,monkeypatch):
    import transformers
    from src import clip_difficulty as b
    from src.clip_fresh_split_policy import FullPoolReport
    _,_,_,split=actual_split();seed=271828;stream=2026100101
    prep=deepcopy(b.load_protocol())
    prep.update(training_seeds=[seed],stream_seeds=[stream],checkpoint_steps=[1,4],trunk_updates=4,
        horizon=2,batch_size=32,expected_batches_per_epoch=2,common_stream_checkpoint_step=1)
    p=deepcopy(g.load_protocol());p.update(training_seeds=[seed],stream_seeds=[stream],checkpoint_steps=[1,4],batch_size=32)
    p['populations']['extended']['offsets']=[0,1]
    p['populations']['calibration_feasibility']['batch_offsets']=[0]
    p['budget'].update(total_batch_forwards=6,total_query_exposures=192,native_preparation_updates_if_all_states_regenerated=4)
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
    def opt_builder(model,_):return torch.optim.AdamW(model.parameters(),lr=.001)
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
    ci=sorted(set(split['training'])-used-set(split['frozen_probes']))[:32]
    _,cr=build(data,ci,seed=seed,epoch=states[1]['epoch'],canonical=False)
    cal_identity=dict(records=cr,caption_epoch=states[1]['epoch'],source_indices=ci,disjoint_from_report_probes_and_continuation_by_image_key=True)
    _,tr=build(data,[i for bs in common[f'{seed}:{stream}'] for i in bs],seed=seed,epoch=states[1]['epoch'],canonical=False)
    plan=dict(continuation_batches=common,calibration_batches={str(seed):[ci]},seeds={str(seed):dict(
        native_trunk_audit_canonical_sha256=reset.digest(trunk),
        states={str(w):dict(archived_start=starts[str(w)],weights_sha256=d.state_digest(states[w]['model'])) for w in [1,4]},
        streams={str(stream):dict(training_batch_records_sha256=reset.digest(dict(records=tr,caption_epoch=states[1]['epoch'])))},
        calibration_batch_records_canonical_sha256=reset.digest(cal_identity))})
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *_:None)
    monkeypatch.setattr(clip_train,'load_training_config',lambda *_:deepcopy(cfg))
    monkeypatch.setattr(clip_train,'resolve_device',lambda *_:(torch.device('cpu'),{'device':'cpu'}))
    monkeypatch.setattr(clip_train,'build_clip_training_data',lambda *_args,**_kw:data)
    monkeypatch.setattr(d,'build_batches',build)
    monkeypatch.setattr(d,'CLIPEncoderBackend',SimpleNamespace(from_pretrained=backend))
    monkeypatch.setattr(d,'configure_clip_trainable_parameters',lambda *_:dict(trainable_parameter_count=10895617,trainable_tensor_count=35))
    monkeypatch.setattr(d,'build_clip_optimizer',opt_builder)
    monkeypatch.setattr(runner,'fixed_probe_batches',lambda _data,_p,folder:d.write_json(folder/'training_batches.json',{'probe_identity':'toy'}))
    monkeypatch.setattr(FullPoolReport,'evaluate',lambda *_:pytest.fail('D0 evaluated a reporting objective'))
    actual_steps=[];step=d.step_update
    def counted(*args,**kwargs):actual_steps.append(1);return step(*args,**kwargs)
    monkeypatch.setattr(d,'step_update',counted)
    out=tmp_path/'regenerated';out.mkdir();synced=[]
    result=runner.run(out,lambda:synced.append(1),p=p,plan=plan,preparation_protocol=prep,native_reference=ref)
    assert len(actual_steps)==4 and synced and result['query_exposures']==192
    done=json.loads((out/'completion.json').read_text())
    assert done['actual_native_preparation_updates']==4 and done['audit_optimizer_updates']==0
    shards=list(out.rglob('*.jsonl.gz'));assert len(shards)==6
    for path in shards:
        records=[json.loads(x) for x in gzip.open(path,'rt')]
        assert len(records)==32 and not any('caption' in r or 'captions' in r for r in records)
    assert not list(out.rglob('*.pt'))
    # Safe-load authenticated snapshots and run the same complete audit without training.
    saved=tmp_path/'snapshots'/f'seed_{seed}';saved.mkdir(parents=True)
    for w,state in states.items():torch.save(state,saved/f'step_{w:06d}.pt')
    d.write_json(saved/'trunk_audit.json',trunk)
    loaded=runner.load_snapshots(tmp_path/'snapshots',seed,[1,4],plan['seeds'][str(seed)])
    assert d.state_digest(loaded[0])==d.state_digest(states)
    actual_steps.clear();out2=tmp_path/'loaded';out2.mkdir()
    other=runner.run(out2,lambda:None,p=p,plan=plan,preparation_protocol=prep,native_reference=ref,snapshot_dir=tmp_path/'snapshots')
    assert other==result and not actual_steps
    assert json.loads((out2/'completion.json').read_text())['actual_native_preparation_updates']==0
    # An authentication failure is fatal and cannot silently regenerate.
    altered=deepcopy(states[1]);altered['model']['frozen'][0]+=1;torch.save(altered,saved/'step_000001.pt')
    with pytest.raises(g.AuditFailure,match='authentication'):
        runner.load_snapshots(tmp_path/'snapshots',seed,[1,4],plan['seeds'][str(seed)])
