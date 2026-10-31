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

from src import clip_optimizer_reset as reset
from src import clip_native_history_policy as native
from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src.clip_gradient_stages import observational_state
from tests.test_clip_native_history_policy import toy
from tests.test_clip_fresh_split_policy import actual_split
from tests.test_clip_paired_updates import TinyCLIP
from tests.test_clip_policy_test import Scores


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def setup(horizon=4):
    model,opt,sched,batch,config = toy()
    for _ in range(3):d.step_update(model,opt,sched,batch,config,0.)
    initial = native.snapshot(model,opt,sched,epoch=1,step=3,seed=1)
    report = d.ReportPool([batch],{'fixed':{'batches':[list(range(10))]}})
    prior = policy.run_pair(model,opt,sched,initial,[batch]*horizon,config,Scores([1.]),report,
        coefficient=.06,threshold=0.,refresh_offsets=[0],observation_steps=[0,horizon],
        permutation=list(range(horizon)),commit=lambda *_:None)
    reference = {'arms':{a:dict(points=prior['points'][a],traces_sha256=reset.digest(prior['traces'][a]))
                           for a in ['native','sustained']}}
    return model,opt,sched,batch,config,initial,report,reference


def test_frozen_contract_and_reference():
    p=reset.load_protocol();ref=reset.load_reference()
    assert p['optimizer_step_budget']==1500+3*2*2*4*50==3900
    assert p['observation_steps']==[0,1,5,10,25,50]
    assert not p['reset_scheduler'] and not p['reset_weights']
    assert p['reset_moments'] and p['reset_parameter_step_counters'] and p['retained_replay_required']
    assert not p['new_training_seeds'] and not p['fresh_reporting_split_for_this_run']
    assert p['primary_start']==500 and len(ref['pairs'])==12
    _,_,_,split=actual_split()
    assert policy.make_plans(split['training'],p)==policy.make_plans(split['training'],native.load_protocol())
    assert len({v['starts']['100']['state_sha256'] for v in ref['seeds'].values()})==3


@pytest.mark.parametrize('key,value',[('reset_scheduler',True),('reset_parameter_step_counters',False),
    ('primary_start',100),('observation_steps',[0,50]),('coefficient',.1),('optimizer_step_budget',3901)])
def test_changed_protocol_rejected(tmp_path,monkeypatch,key,value):
    p=reset.load_protocol();parent=native.load_protocol()
    p[key]=value;(tmp_path/'configs').mkdir()
    (tmp_path/'configs/clip_optimizer_reset.json').write_text(json.dumps(p))
    monkeypatch.setattr(d,'ROOT',tmp_path)
    monkeypatch.setattr(native,'load_protocol',lambda:parent)
    with pytest.raises(ValueError,match='Frozen'):reset.load_protocol()


def test_reset_changes_only_adamw_state_and_matches_fresh_optimizer():
    model,opt,sched,batch,config,initial,_,_=setup()
    d.restore(model,opt,sched,initial)
    before=d.state_digest(initial)
    lr=[g['lr'] for g in opt.param_groups];schedule=d.state_digest(sched.state_dict())
    audit=reset.reset_optimizer_state(model,opt,sched)
    assert audit['old_step_counts']==[3.] and not opt.state
    assert [g['lr'] for g in opt.param_groups]==lr and d.state_digest(sched.state_dict())==schedule
    for _ in range(4):d.step_update(model,opt,sched,batch,config,.06)
    actual=d.live_digest(model,opt,sched)
    fresh=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    new_sched=reset.clip_train.build_scheduler(fresh,warmup_steps=2,total_steps=3850)
    cold={**initial,'optimizer':{'state':{},'param_groups':deepcopy(initial['optimizer']['param_groups'])}}
    d.restore(model,fresh,new_sched,cold)
    for _ in range(4):d.step_update(model,fresh,new_sched,batch,config,.06)
    assert d.live_digest(model,fresh,new_sched)==actual
    assert all(float(s['step'])==4 for s in fresh.state.values())
    assert new_sched.last_epoch==7 and d.state_digest(initial)==before


def test_reject_empty_or_wrong_optimizer_state():
    model,opt,sched,_,_=toy()
    with pytest.raises(ValueError,match='populated'):reset.reset_optimizer_state(model,opt,sched)
    model,opt,sched,_,_,initial,_,_=setup();d.restore(model,opt,sched,initial)
    next(iter(opt.state.values()))['extra']=torch.tensor(1.)
    with pytest.raises(ValueError,match='layout'):reset.reset_optimizer_state(model,opt,sched)


def test_full_timecourse_replays_retained_and_matches_unobserved_reset():
    model,opt,sched,batch,config,initial,report,reference=setup(50)
    events=[];points=[]
    result=reset.run_pair(model,opt,sched,initial,[batch]*50,config,report,reference,
        coefficient=.06,observation_steps=[0,1,5,10,25,50],commit=lambda l,v:events.append(l),
        save_point=lambda a,t,v:points.append((a,t)))
    assert events==['start_'+a for a in reset.ARMS] and len(points)==24
    assert result['audit']['actual_optimizer_steps']==200 and result['audit']['retained_exact_replay']
    assert len({s['weights_sha256'] for s in result['starts'].values()})==1
    assert result['starts']['reset_native']==result['starts']['reset_sustained']
    assert result['starts']['retained_native']==result['starts']['retained_sustained']
    assert result['starts']['reset_native']['state_sha256']!=result['starts']['retained_native']['state_sha256']
    for arm in reset.ARMS:
        assert list(result['points'][arm])==['0','1','5','10','25','50']
        assert len(result['traces'][arm])==50
    for control in ['native','sustained']:
        assert result['points']['retained_'+control]['50']==reference['arms'][control]['points']['50']
    final=d.live_digest(model,opt,sched)
    cold={**initial,'optimizer':{'state':{},'param_groups':deepcopy(initial['optimizer']['param_groups'])}}
    d.restore(model,opt,sched,cold)
    for _ in range(50):d.step_update(model,opt,sched,batch,config,.06)
    assert final==d.live_digest(model,opt,sched)  # Added timecourse observations are inert.


@pytest.mark.parametrize('where',['start','endpoint','trace'])
def test_replay_mismatch_fails_before_any_reset_branch(where,monkeypatch):
    model,opt,sched,batch,config,initial,report,reference=setup()
    if where=='trace':reference['arms']['native']['traces_sha256']='wrong'
    else:reference['arms']['native']['points']['0' if where=='start' else '4']['state_sha256']='wrong'
    monkeypatch.setattr(reset,'reset_optimizer_state',lambda *_:pytest.fail('Reset started after replay mismatch'))
    with pytest.raises(AssertionError,match='replay|reproduce'):
        reset.run_pair(model,opt,sched,initial,[batch]*4,config,report,reference,
            coefficient=.06,observation_steps=[0,1,4],commit=lambda *_:None)


def test_persistence_failure_prevents_first_update(monkeypatch):
    model,opt,sched,batch,config,initial,report,reference=setup()
    monkeypatch.setattr(d,'step_update',lambda *_:pytest.fail('Training after failed commit'))
    def fail(*_):raise OSError('Drive unavailable')
    with pytest.raises(OSError):reset.run_pair(model,opt,sched,initial,[batch]*4,config,report,reference,
        coefficient=.06,observation_steps=[0,4],commit=fail)


def test_summary_interaction_and_all_timepoints():
    p=reset.load_protocol();rows=[]
    for s in p['training_seeds']:
        for k in p['checkpoint_steps']:
            for u in p['stream_seeds']:
                points={a:{str(t):{'report':dict(native_loss=10.+(t/50*(1 if a=='retained_sustained' else -1)
                    if a.endswith('sustained') else 0.),pool_retrieval={'mean_r1_percent':5.})}
                    for t in p['observation_steps']} for a in reset.ARMS}
                rows.append(dict(training_seed=s,checkpoint_step=k,stream_seed=u,points=points,
                    audit={'actual_optimizer_steps':200,'retained_exact_replay':True}))
    result=reset.summarize(rows,p)
    assert result['primary_value']==-2 and result['optimizer_steps']==3900
    assert result['by_starting_step']['100']['0']['interaction']['loss']==0
    assert result['by_starting_step']['500']['25']['interaction']['loss']==-1
    assert len(result['per_seed'])==3 and len(result['paired_timecourses'])==12
    with pytest.raises(ValueError,match='grid'):reset.summarize(rows[:-1],p)
    rows[0]['audit']['retained_exact_replay']=False
    with pytest.raises(ValueError,match='replay'):reset.summarize(rows,p)


def test_separate_run_guard(tmp_path):
    from colabs.run_clip_optimizer_reset import require_no_previous_run
    old=tmp_path/'clip_native_history_policy_today';old.mkdir()
    (old/'run_manifest.json').write_text(json.dumps({'experiment':'clip_native_history_policy_v1'}))
    require_no_previous_run(tmp_path,'clip_optimizer_reset_v1')
    new=tmp_path/'clip_optimizer_reset_today';new.mkdir()
    (new/'run_manifest.json').write_text(json.dumps({'experiment':'clip_optimizer_reset_v1'}))
    with pytest.raises(RuntimeError,match='no automatic resume'):require_no_previous_run(tmp_path,'clip_optimizer_reset_v1')


def test_delivery_checksums_and_preservation():
    path=d.ROOT/'colabs/clip_optimizer_reset_drive_one_cell.py'
    if not path.exists():pytest.skip('Delivery not part of source overlay')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==values['BUNDLE_SHA256']
    sources=json.loads(raw);pins=json.loads(sources.pop('optimizer_reset_source_manifest.json'))
    assert set(sources)==set(pins)
    for name,source in sources.items():
        assert (d.ROOT/name).read_text()==source and hashlib.sha256(source.encode()).hexdigest()==pins[name]
    assert 'clip_optimizer_reset_logs' in path.read_text() and 'PARTIAL_DRIVE_SYNC_COMPLETE' in path.read_text()


def test_end_to_end_regeneration_replay_reset_and_outputs(tmp_path,monkeypatch):
    import transformers
    old={**native.load_protocol(),'trunk_updates':4,'checkpoint_steps':[1,4],
        'horizon':4,'gate_refresh_offsets':[0,1,3],'observation_steps':[0,4],
        'trunk_optimizer_steps':12,'branch_optimizer_steps':192,'optimizer_step_budget':204,
        'expected_batches_per_epoch':2}
    p={**reset.load_protocol(),**{k:old[k] for k in ['trunk_updates','checkpoint_steps','horizon',
        'trunk_optimizer_steps','branch_optimizer_steps','optimizer_step_budget','expected_batches_per_epoch']},
        'primary_start':4,'observation_steps':[0,1,2,4]}
    _,_,_,split=actual_split()
    _,_,_,batch,config=toy()
    report_batch={**batch,'pixel_values':torch.randn(512,6),'input_ids':torch.randn(512,6),
        'species_ids':torch.arange(512)}
    config.update(experiment={},runtime={},dataset={},model={'checkpoint':'toy','trainable_policy':'toy'},
        scheduler={'warmup_steps':2})
    class Backend:
        @staticmethod
        def from_pretrained(_):return TinyCLIP()
    def data_builder(config,processor,**kwargs):
        exclusions=json.loads(Path(config['dataset']['diagnostic_holdout_indices']).read_text())
        training=sorted(set(range(5994))-set(exclusions));assert training==split['training']
        dataset=SimpleNamespace(source_indices=training,set_epoch=lambda _:None,
            grouped_split=SimpleNamespace(groups=[SimpleNamespace(image_key=f'image_{i}') for i in range(5994)]))
        return SimpleNamespace(train_dataset=dataset,train_loader=[batch,batch])
    def batches(data,indices,*,seed,epoch,canonical):
        records=[dict(source_index=i,image_key=f'image_{i}',caption='toy',caption_index=0,species_id=i) for i in indices]
        assert set(indices)<=set(split['report'] if canonical else split['training'])
        return ([report_batch] if canonical else [batch]*4),records
    def probe_batches(data,p,folder):
        d.write_json(folder/'training_batches.json',{'probe_identity':'toy'})
        return [batch]
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *_:None)
    monkeypatch.setattr(reset.clip_train,'load_training_config',lambda *_:deepcopy(config))
    monkeypatch.setattr(reset.clip_train,'resolve_device',lambda *_:('cpu',{'device':'cpu'}))
    monkeypatch.setattr(reset.clip_train,'build_clip_training_data',data_builder)
    monkeypatch.setattr(d,'build_batches',batches)
    monkeypatch.setattr(d,'CLIPEncoderBackend',Backend)
    monkeypatch.setattr(d,'configure_clip_trainable_parameters',lambda *_:{
        'trainable_parameter_count':10895617,'trainable_tensor_count':35})
    monkeypatch.setattr(d,'build_clip_optimizer',lambda model,_:torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad],lr=.001))
    monkeypatch.setattr(native,'fixed_probe_batches',probe_batches)
    monkeypatch.setattr(reset,'fixed_probe_batches',probe_batches)
    monkeypatch.setattr(policy,'AlignmentGate',lambda *_:Scores([1.,-1.,1.]*4))
    prior=tmp_path/'prior';prior.mkdir()
    native.run(prior,lambda:None,old)
    read=lambda f:json.loads(f.read_text())
    ref={'seeds':{},'pairs':{}}
    for name in ['split_plan.json','branch_plans.json','training_exclusion_indices.json']:
        ref[name]=reset.digest(read(prior/name))
    for s in p['training_seeds']:
        folder=prior/f'seed_{s}'
        ref['seeds'][str(s)]={name:reset.digest(read(folder/name)) for name in
            ['data_roles.json','training_batches.json','trunk_audit.json']}
        ref['seeds'][str(s)]['starts']={str(k):read(folder/f'start_{k:06d}.json') for k in p['checkpoint_steps']}
        for k in p['checkpoint_steps']:
            for u in p['stream_seeds']:
                pair=folder/f'step_{k:06d}'/f'stream_{u}'
                row=read(pair/'result.json')
                ref['pairs'][f'{s}:{k}:{u}']=dict(training_batches_sha256=reset.digest(read(pair/'training_batches.json')),
                    arms={a:dict(points=row['points'][a],traces_sha256=reset.digest(row['traces'][a]))
                          for a in ['native','sustained']})
    monkeypatch.setattr(policy,'AlignmentGate',lambda *_:pytest.fail('No gate in optimizer-reset experiment'))
    output=tmp_path/'reset';output.mkdir();synced=[]
    summary=reset.run(output,lambda:synced.append(1),p,ref)
    assert summary['optimizer_steps']==204 and summary['replication_units']==3
    assert len(list(output.glob('seed_*/step_*/stream_*/result.json')))==12
    assert len(list(output.rglob('features_*.npz')))==48
    assert len(list(output.rglob('point_*.json')))==192 and not list(output.rglob('*.pt'))
    assert read(output/'completion.json')['retained_exact_replay']
    assert read(output/'comparison.json')==summary
    for f in output.glob('seed_*/step_*/stream_*/result.json'):
        row=read(f)
        assert row['audit']['retained_exact_replay'] and row['audit']['reset_only_optimizer_state']
        for mode in ['native','sustained']:
            assert row['resets']['reset_'+mode]['old_step_counts']==[row['checkpoint_step']]
    assert synced
