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

from src import clip_weight_lr as wl
from src import clip_checkpoint_diagnostic as d
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
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
    states = {}
    for step in range(1,5):
        d.step_update(model,opt,sched,batch,config,0.)
        if step in [1,4]:
            states[step] = native.snapshot(model,opt,sched,epoch=step,step=step,seed=1)
    report = d.ReportPool([batch],{'fixed':{'batches':[list(range(10))]}})
    # Independently execute the existing reset path, without LR-crossing helpers.
    ref = {}
    for arm in wl.ARMS:
        d.restore(model,opt,sched,states[1])
        reset.reset_optimizer_state(model,opt,sched)
        points,traces = {},[]
        for offset in range(horizon+1):
            points[str(offset)] = dict(report=report.evaluate(model),state_sha256=d.live_digest(model,opt,sched))
            if offset < horizon:
                traces.append(d.step_update(model,opt,sched,batch,config,.06 if arm=='sustained' else 0.))
        ref[arm] = dict(points=points,traces_sha256=reset.digest(traces))
    return model,opt,sched,batch,config,states,report,ref


def run_toy(args, **kwargs):
    model,opt,sched,batch,config,states,report,ref = args
    options = dict(coefficient=.06,observation_steps=[0,1,4],weight_steps=[1,4],lr_steps=[1,4],
                   common_step=1,commit=lambda *_:None)
    options.update(kwargs)
    return wl.run_factorial(model,opt,sched,states,[batch]*4,config,report,ref,**options)


def test_frozen_protocol_and_common_plans():
    p = wl.load_protocol()
    assert p['branch_optimizer_steps']==2*2*2*3*2*50==2400
    assert p['optimizer_step_budget']==3900 and p['primary_contrasts']==wl.FACTORS
    assert p['observation_steps']==[0,1,5,10,25,50]
    _,_,_,split = actual_split()
    plans = wl.common_plans(split['training'],p)
    old,_ = policy.make_plans(split['training'],native.load_protocol())
    assert len(plans)==6
    for seed in p['training_seeds']:
        for stream in p['stream_seeds']:
            assert plans[f'{seed}:{stream}']==old[f'{seed}:100:{stream}']
    ref = wl.load_reference()
    assert len(ref['pairs'])==6
    archive = d.ROOT/'experiment_results/clip_optimizer_reset_2026-10'
    if archive.exists():
        for key,value in ref['pairs'].items():
            seed,stream = key.split(':')
            path = archive/f'seed_{seed}/step_000100/stream_{stream}/result.json'
            assert d.checksum(path)==value['source_result_sha256']
            row = json.loads(path.read_text())
            for arm in wl.ARMS:
                assert value['arms'][arm]['points']==row['points']['reset_'+arm]
                assert value['arms'][arm]['traces_sha256']==reset.digest(row['traces']['reset_'+arm])


@pytest.mark.parametrize('key,value', [('coefficient',.1),('lr_checkpoint_steps',[100,501]),
    ('common_stream_checkpoint_step',500),('reset_parameter_step_counters',False),
    ('observation_steps',[0,50]),('branch_optimizer_steps',2401)])
def test_changed_protocol_rejected(tmp_path,monkeypatch,key,value):
    expected = wl.expected_protocol()
    p = deepcopy(expected);p[key]=value
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/clip_weight_lr.json').write_text(json.dumps(p))
    monkeypatch.setattr(wl,'expected_protocol',lambda:expected)
    monkeypatch.setattr(d,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Frozen'):wl.load_protocol()


def test_crossed_branch_matches_fresh_adamw_with_donor_schedule():
    model,opt,sched,batch,config,states,_,_ = setup()
    before = d.state_digest(states)
    wl.prepare_branch(model,opt,sched,states[4],states[1],states[1])
    assert not opt.state and sched.last_epoch==1
    assert d.state_digest(model.state_dict())==d.state_digest(states[4]['model'])
    for _ in range(4):d.step_update(model,opt,sched,batch,config,.06)
    actual = d.live_digest(model,opt,sched)
    # Construct a genuinely new optimizer and scheduler as an independent oracle.
    model.load_state_dict(states[4]['model'])
    fresh = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001)
    new_sched = wl.clip_train.build_scheduler(fresh,warmup_steps=2,total_steps=3850)
    cold = deepcopy(states[4]);cold['optimizer'] = deepcopy(states[1]['optimizer'])
    cold['optimizer']['state'] = {};cold['scheduler'] = deepcopy(states[1]['scheduler'])
    for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']:cold[k] = states[1][k]
    d.restore(model,fresh,new_sched,cold)
    for _ in range(4):d.step_update(model,fresh,new_sched,batch,config,.06)
    assert d.live_digest(model,fresh,new_sched)==actual
    assert all(float(s['step'])==4 for s in fresh.state.values())
    assert new_sched.last_epoch==5 and d.state_digest(states)==before


def test_non_lr_group_change_rejected():
    model,opt,sched,_,_,states,_,_ = setup()
    donor = deepcopy(states[1]);donor['optimizer']['param_groups'][0]['weight_decay'] = .2
    with pytest.raises(ValueError,match='non-LR'):
        wl.prepare_branch(model,opt,sched,states[4],donor,states[1])


def test_all_cells_match_rng_lr_and_observations_are_inert():
    args = setup();saved=[]
    row = run_toy(args,save_point=lambda *x:saved.append(x[:3]))
    assert row['audit']['actual_optimizer_steps']==32 and len(saved)==24
    assert len(row['cells'])==4
    model,opt,sched,batch,config,states,_,_ = args
    for w in [1,4]:
        cells = [row['cells'][wl.cell_key(w,lr)] for lr in [1,4]]
        assert len({c['starts'][a]['weights_sha256'] for c in cells for a in wl.ARMS})==1
        assert cells[0]['starts']['native']['optimizer_sha256']!=cells[1]['starts']['native']['optimizer_sha256']
    assert len({c['starts'][a]['rng_sha256'] for c in row['cells'].values() for a in wl.ARMS})==1
    for lr in [1,4]:
        sequences = [[t['learning_rates'] for t in row['cells'][wl.cell_key(w,lr)]['traces'][a]]
                     for w in [1,4] for a in wl.ARMS]
        assert all(x==sequences[0] for x in sequences)
    # Removing intermediate reporting must leave all endpoints and traces identical.
    unobserved = run_toy(args,observation_steps=[0,4])
    for key,c in row['cells'].items():
        for a in wl.ARMS:
            assert c['points'][a]['4']==unobserved['cells'][key]['points'][a]['4']
            assert c['traces'][a]==unobserved['cells'][key]['traces'][a]


@pytest.mark.parametrize('where',['point','trace'])
def test_diagonal_failure_prevents_crossed_updates(where,monkeypatch):
    args = setup()
    if where=='point':args[-1]['native']['points']['0']['state_sha256']='wrong'
    else:args[-1]['native']['traces_sha256']='wrong'
    original = wl.prepare_branch
    def guarded(*values):
        assert values[-3]['completed_updates']==values[-2]['completed_updates']==1
        return original(*values)
    monkeypatch.setattr(wl,'prepare_branch',guarded)
    with pytest.raises(AssertionError,match='replay'):run_toy(args)


def test_failed_persistence_prevents_training(monkeypatch):
    args = setup()
    monkeypatch.setattr(d,'step_update',lambda *_:pytest.fail('Training after failed persistence'))
    def fail(*_):raise OSError('Drive unavailable')
    with pytest.raises(OSError):run_toy(args,commit=fail)


def test_factorial_arithmetic_and_nested_seed_averaging():
    p = wl.load_protocol();rows=[]
    audit = dict(actual_optimizer_steps=400,early_diagonal_exact_replay=True,source_immutable=True,
        frozen_parameters_unchanged=True,matched_rng_across_all_cells=True,matched_lr_within_lr_level=True,
        fresh_adamw_all_branches=True,observations_preserve_state=True,report_controls_treatment=False)
    for i,seed in enumerate(p['training_seeds']):
        for j,stream in enumerate(p['stream_seeds']):
            cells={}
            for w in [100,500]:
                for lr in [100,500]:
                    # D=1 + 2*w_late + 3*lr_late + 4*interaction + seed + 2*stream
                    effect=1+2*(w==500)+3*(lr==500)+4*(w==lr==500)+i+2*j
                    points={a:{str(t):{'report':dict(native_loss=100+w/10+lr/20+
                        (effect*t/50 if a=='sustained' else 0),pool_retrieval={'mean_r1_percent':5.})}
                        for t in p['observation_steps']} for a in wl.ARMS}
                    cells[wl.cell_key(w,lr)]=dict(weight_step=w,lr_step=lr,points=points,
                        traces={a:[{}]*50 for a in wl.ARMS})
            rows.append(dict(training_seed=seed,stream_seed=stream,cells=cells,audit=deepcopy(audit)))
    summary=wl.summarize(rows,p)
    assert summary['primary_values']==dict(weight_main_effect=4.,lr_main_effect=5.,weight_x_lr_interaction=4.)
    assert summary['timecourse']['50']['cell_effects']['w100_lr100']['loss']==3.
    assert summary['timecourse']['0']['contrasts']['loss']['weight_x_lr_interaction']==0
    assert summary['replication_units']==3 and summary['branches']==48 and summary['optimizer_steps']==3900
    with pytest.raises(ValueError,match='grid'):wl.summarize(rows[:-1],p)
    rows[0]['audit']['early_diagonal_exact_replay']=False
    with pytest.raises(ValueError,match='audit'):wl.summarize(rows,p)


def test_delivery_checksums():
    path=d.ROOT/'colabs/clip_weight_lr_drive_one_cell.py'
    if not path.exists():pytest.skip('Delivery not part of source overlay')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==values['BUNDLE_SHA256']
    sources=json.loads(raw);pins=json.loads(sources.pop('weight_lr_source_manifest.json'))
    assert set(sources)==set(pins)
    for name,source in sources.items():
        assert (d.ROOT/name).read_text()==source and hashlib.sha256(source.encode()).hexdigest()==pins[name]
    assert 'clip_weight_lr_logs' in path.read_text() and 'PARTIAL_DRIVE_SYNC_COMPLETE' in path.read_text()


def test_end_to_end_native_replay_crossing_and_artifacts(tmp_path,monkeypatch):
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
    reset_output=tmp_path/'reset';reset_output.mkdir()
    reset.run(reset_output,lambda:None,p,ref)
    diagonal={'pairs':{}}
    for seed in p['training_seeds']:
        for stream in p['stream_seeds']:
            row=read(reset_output/f'seed_{seed}/step_000001/stream_{stream}/result.json')
            diagonal['pairs'][f'{seed}:{stream}']={'arms':{a:dict(points=row['points']['reset_'+a],
                traces_sha256=reset.digest(row['traces']['reset_'+a])) for a in wl.ARMS}}
    wp={**wl.load_protocol(),**{k:p[k] for k in ['trunk_updates','checkpoint_steps','horizon',
        'trunk_optimizer_steps','branch_optimizer_steps','optimizer_step_budget','expected_batches_per_epoch',
        'observation_steps']},'lr_checkpoint_steps':[1,4],'common_stream_checkpoint_step':1}
    monkeypatch.setattr(wl,'fixed_probe_batches',probe_batches)
    output=tmp_path/'weight_lr';output.mkdir();synced=[]
    summary=wl.run(output,lambda:synced.append(1),wp,ref,diagonal)
    assert summary['optimizer_steps']==204 and summary['replication_units']==3
    assert len(list(output.glob('seed_*/stream_*/result.json')))==6
    assert len(list(output.rglob('features_*.npz')))==48
    assert len(list(output.rglob('point_*.json')))==192 and not list(output.rglob('*.pt'))
    assert read(output/'completion.json')['early_diagonal_exact_replay']
    assert read(output/'comparison.json')==summary
    plans=read(output/'common_branch_plans.json')
    assert len(plans)==6
    for f in output.glob('seed_*/stream_*/result.json'):
        row=read(f)
        assert row['common_caption_epoch']==ref['seeds'][str(row['training_seed'])]['starts']['1']['epoch']
        for cell in row['cells'].values():
            for arm in wl.ARMS:
                assert cell['resets'][arm]['old_step_counts']==[cell['weight_step']]
                assert len(cell['traces'][arm])==4
    assert synced


def test_separate_run_guard(tmp_path):
    from colabs.run_clip_weight_lr import require_no_previous_run
    old=tmp_path/'clip_optimizer_reset_old';old.mkdir()
    (old/'run_manifest.json').write_text(json.dumps({'experiment':'clip_optimizer_reset_v1'}))
    require_no_previous_run(tmp_path,'clip_weight_lr_v1')
    current=tmp_path/'clip_weight_lr_today';current.mkdir()
    (current/'run_manifest.json').write_text(json.dumps({'experiment':'clip_weight_lr_v1'}))
    with pytest.raises(RuntimeError,match='no automatic resume'):
        require_no_previous_run(tmp_path,'clip_weight_lr_v1')
