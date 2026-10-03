from copy import deepcopy
import ast
import base64
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest
import torch

from src import clip_native_history_policy as native
from src import clip_fresh_split_policy as parent
from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src.clip_gradient_stages import observational_state
from tests.test_clip_continuation_timecourse import make_pair
from tests.test_clip_paired_updates import TinyCLIP
from tests.test_clip_policy_test import Scores
from tests.test_clip_fresh_split_policy import actual_split


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def toy():
    _,_,_,batch,config,_,_,_ = make_pair()
    model = TinyCLIP()
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001)
    sched = native.clip_train.build_scheduler(opt,warmup_steps=2,total_steps=3850)
    batch = {**batch,'source_indices':torch.arange(10),'captions':['toy']*10}
    return model,opt,sched,batch,config


def test_frozen_design_only_changes_history_and_reporting_metadata():
    p,old = native.load_protocol(),parent.load_protocol()
    changed = {k for k in old if p[k] != old[k]}
    assert changed == {'experiment','trunk_gate_refresh_steps'}
    assert p['starting_state_history'] == 'native_only'
    assert p['trunk_coefficient'] == 0 and not p['trunk_probes']
    assert p['checkpoint_steps'] == [100,500]
    assert p['optimizer_step_budget'] == 3*500 + 3*2*2*4*50 == 3900
    _,holdout,probes,split = actual_split()
    assert native.split_indices(list(range(5994)),holdout,probes,p) == split
    assert policy.make_plans(split['training'],p) == policy.make_plans(split['training'],old)
    assert 'probe' not in inspect.signature(native.trunk).parameters
    assert 'report' not in inspect.signature(native.trunk).parameters


def test_parent_sources_and_delivery_unchanged():
    pins = json.loads((d.ROOT/'configs/clip_native_history_parent_pins.json').read_text())
    delivery_only = {'scripts/build_clip_fresh_split_policy_cell.py',
                     'colabs/clip_fresh_split_policy_drive_one_cell.py'}
    assert len(pins) == 46
    for relative,expected in pins.items():
        path = d.ROOT/relative
        if relative in delivery_only and not path.exists():
            continue  # These two are not included in a Colab source overlay.
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected,relative


@pytest.mark.parametrize('field,value', [('trunk_coefficient',.06),('trunk_probes',True),
    ('trunk_gate_refresh_steps',[100]),('starting_state_history','alignment_gated'),
    ('checkpoint_steps',[100,750]),('optimizer_step_budget',3901),('report_size',511),
    ('parent_bundle_id','wrong'),('primary','changed')])
def test_modified_contract_rejected(tmp_path,monkeypatch,field,value):
    p = native.load_protocol()
    p[field] = value
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/clip_native_history_policy.json').write_text(json.dumps(p))
    (tmp_path/'configs/clip_fresh_split_policy.json').write_text(json.dumps(parent.load_protocol()))
    monkeypatch.setattr(d,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Frozen native-history'):
        native.load_protocol()


def test_full_500_native_steps_exactly_match_manual_training(monkeypatch):
    model,opt,sched,batch,config = toy()
    p = native.load_protocol()
    initial = native.snapshot(model,opt,sched,epoch=0,step=0,seed=271828)
    committed = []
    def forbidden(*args,**kwargs):
        pytest.fail('Starting-state generation called auxiliary loss or a gate')
    monkeypatch.setattr(d,'auxiliary_losses',forbidden)
    monkeypatch.setattr(policy.AlignmentGate,'measure',forbidden)
    states,audit = native.trunk(model,opt,sched,iter([(1,batch)]*500),config,p,
        seed=271828,commit=lambda label,value:committed.append((label,deepcopy(value))))
    assert committed[0][0] == 'native_history_contract'
    assert set(states) == {100,500}
    assert len(audit['traces']) == audit['actual_optimizer_steps'] == 500
    assert audit['auxiliary_updates'] == audit['probe_calls'] == 0
    assert audit['decisions'] == [] and not audit['reporting_access']
    assert all(t['coefficient'] == 0 and t['native_loss'] == t['total_loss'] for t in audit['traces'])
    assert [t['completed_updates'] for t in audit['traces']] == list(range(1,501))
    digests = {step:d.state_digest(state) for step,state in states.items()}
    final = d.live_digest(model,opt,sched)
    for step,state in states.items():
        assert state['arm'] == 'baseline' and state['prior_auxiliary_updates'] == 0
        assert state['starting_state_history'] == 'native_only'
        assert state['scheduler']['last_epoch'] == state['completed_updates'] == step
    d.restore(model,opt,sched,initial)
    for step in range(1,501):
        d.step_update(model,opt,sched,batch,config,0.)
        if step in states:
            assert d.state_digest(model.state_dict()) == d.state_digest(states[step]['model'])
            assert d.state_digest(opt.state_dict()) == d.state_digest(states[step]['optimizer'])
            assert d.state_digest(sched.state_dict()) == d.state_digest(states[step]['scheduler'])
    assert d.live_digest(model,opt,sched) == final
    assert digests == {step:d.state_digest(state) for step,state in states.items()}


def test_parent_prefix_and_native_prefix_agree_before_first_treatment():
    model,opt,sched,batch,config = toy()
    p = {**native.load_protocol(),'trunk_updates':4,'checkpoint_steps':[1,4]}
    initial = native.snapshot(model,opt,sched,epoch=0,step=0,seed=1)
    states,_ = native.trunk(model,opt,sched,iter([(1,batch)]*4),config,p,seed=1,commit=lambda *_:None)
    d.restore(model,opt,sched,initial)
    old = {**parent.load_protocol(),'trunk_updates':4,'checkpoint_steps':[1,4],
           'trunk_gate_refresh_steps':[1,3],'alignment_threshold':0.}
    parent_states,_ = parent.trunk(model,opt,sched,iter([(1,batch)]*4),config,Scores([1.,1.]),
        old,seed=1,commit=lambda *_:None)
    for field in ['model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng']:
        assert d.state_digest(states[1][field]) == d.state_digest(parent_states[1][field])
    assert d.state_digest(states[4]['model']) != d.state_digest(parent_states[4]['model'])


@pytest.mark.parametrize('override', [{'trunk_coefficient':.1},{'trunk_probes':True},
    {'trunk_gate_refresh_steps':[1]},{'checkpoint_steps':[501]}])
def test_invalid_trunk_stops_before_update(override,monkeypatch):
    model,opt,sched,batch,config = toy()
    monkeypatch.setattr(d,'step_update',lambda *_:pytest.fail('Unexpected optimizer step'))
    with pytest.raises(ValueError,match='before updates'):
        native.trunk(model,opt,sched,iter([(1,batch)]*500),config,{**native.load_protocol(),**override},
            seed=1,commit=lambda *_:None)


def test_failed_initial_persistence_prevents_any_update(monkeypatch):
    model,opt,sched,batch,config = toy()
    monkeypatch.setattr(d,'step_update',lambda *_:pytest.fail('Unexpected optimizer step'))
    def failed(*_):
        raise OSError('Drive unavailable')
    with pytest.raises(OSError,match='Drive'):
        native.trunk(model,opt,sched,iter([(1,batch)]*500),config,native.load_protocol(),seed=1,commit=failed)


def synthetic_rows(p):
    rows = []
    for si,seed in enumerate(p['training_seeds']):
        for ki,step in enumerate(p['checkpoint_steps']):
            for ui,stream in enumerate(p['stream_seeds']):
                effect = (-2. if ki == 0 else 3.)+si*.1+ui*.2
                losses = dict(gated=9.,native=10.,sustained=10.+effect,random_matched=9.5)
                points = {a:{str(t):{'report':dict(native_loss=losses[a] if t else 10.,
                    pool_retrieval={'mean_r1_percent':effect if a=='sustained' and t else 0.})}
                    for t in p['observation_steps']} for a in policy.ARMS}
                differences = {str(t):{c:dict(
                    report_loss=points['gated'][str(t)]['report']['native_loss']-points[c][str(t)]['report']['native_loss'],
                    pool_r1_pp=points['gated'][str(t)]['report']['pool_retrieval']['mean_r1_percent']-
                        points[c][str(t)]['report']['pool_retrieval']['mean_r1_percent'])
                    for c in policy.ARMS[1:]} for t in p['observation_steps']}
                rows.append(dict(training_seed=seed,checkpoint_step=step,stream_seed=stream,
                    starting_state_history='native_only',prior_auxiliary_updates=0,
                    points=points,differences=differences,audit={'actual_optimizer_steps':200},
                    exposure={a:0 for a in policy.ARMS},random_schedule_equals_gate=True))
    return rows


def test_sustained_native_absolute_and_later_early_contrasts_not_conflated():
    p = native.load_protocol()
    rows = synthetic_rows(p)
    summary = native.summarize(rows,p)
    assert summary['primary_value'] == -.5
    assert summary['optimizer_steps'] == 3900
    contrast = summary['sustained_minus_native']
    assert contrast['per_starting_step']['100']['report_loss'] == pytest.approx(-1.8)
    assert contrast['per_starting_step']['500']['report_loss'] == pytest.approx(3.2)
    assert contrast['later_minus_early']['report_loss'] == pytest.approx(5.)
    assert contrast['later_minus_early']['pool_r1_pp'] == pytest.approx(5.)
    assert len(contrast['per_seed']) == 3 and len(summary['exposure']) == 12
    with pytest.raises(ValueError,match='Incomplete'):
        native.summarize(rows[:-1],p)
    rows[0]['prior_auxiliary_updates'] = 1
    with pytest.raises(AssertionError,match='Non-native'):
        native.summarize(rows,p)


def test_native_history_previous_run_guard_is_separate(tmp_path):
    from colabs.run_clip_native_history_policy import require_no_previous_run
    old = tmp_path/'clip_fresh_split_policy_today'
    old.mkdir()
    (old/'run_manifest.json').write_text(json.dumps({'experiment':parent.load_protocol()['experiment']}))
    require_no_previous_run(tmp_path,native.load_protocol()['experiment'])
    new = tmp_path/'clip_native_history_policy_today'
    new.mkdir()
    (new/'run_manifest.json').write_text(json.dumps({'experiment':native.load_protocol()['experiment'],'status':'failed'}))
    with pytest.raises(RuntimeError,match='no automatic resume'):
        require_no_previous_run(tmp_path,native.load_protocol()['experiment'])


def test_end_to_end_native_history_with_tiny_models(tmp_path,monkeypatch):
    import transformers
    p = {**native.load_protocol(),'trunk_updates':4,'checkpoint_steps':[1,4],
        'horizon':4,'gate_refresh_offsets':[0,1,3],'observation_steps':[0,4],
        'trunk_optimizer_steps':12,'branch_optimizer_steps':192,'optimizer_step_budget':204,
        'expected_batches_per_epoch':2}
    _,_,_,split = actual_split()
    _,_,_,batch,config = toy()
    report_batch = {**batch,'pixel_values':torch.randn(512,6),'input_ids':torch.randn(512,6),
        'species_ids':torch.arange(512)}
    config.update(experiment={},runtime={},dataset={},model={'checkpoint':'toy','trainable_policy':'toy'},
                  scheduler={'warmup_steps':2})
    models,probe_calls = [],[]
    class Backend:
        @staticmethod
        def from_pretrained(name):
            models.append(1)
            return TinyCLIP()
    def data_builder(config,processor,**kwargs):
        exclusions = json.loads(Path(config['dataset']['diagnostic_holdout_indices']).read_text())
        training = sorted(set(range(5994))-set(exclusions))
        assert training == split['training']
        dataset = SimpleNamespace(source_indices=training,set_epoch=lambda _:None,
            grouped_split=SimpleNamespace(groups=[SimpleNamespace(image_key=f'image_{i}') for i in range(5994)]))
        return SimpleNamespace(train_dataset=dataset,train_loader=[batch,batch])
    def batches(data,indices,*,seed,epoch,canonical):
        records = [dict(source_index=i,image_key=f'image_{i}',caption='toy',caption_index=0,species_id=i) for i in indices]
        assert set(indices) <= set(split['report'] if canonical else split['training'])
        return ([report_batch] if canonical else [batch]*4),records
    class BranchOnlyScores(Scores):
        def measure(self,model):
            # Every seed's complete native trunk must precede its first gate.
            assert (tmp_path/f'seed_{p["training_seeds"][len(models)-1]}'/'trunk_audit.json').exists()
            probe_calls.append(1)
            return super().measure(model)
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *_:None)
    monkeypatch.setattr(native.clip_train,'load_training_config',lambda *_:deepcopy(config))
    monkeypatch.setattr(native.clip_train,'resolve_device',lambda *_:('cpu',{'device':'cpu'}))
    monkeypatch.setattr(native.clip_train,'build_clip_training_data',data_builder)
    monkeypatch.setattr(d,'build_batches',batches)
    monkeypatch.setattr(d,'CLIPEncoderBackend',Backend)
    monkeypatch.setattr(d,'configure_clip_trainable_parameters',lambda *_:{
        'trainable_parameter_count':10895617,'trainable_tensor_count':35})
    monkeypatch.setattr(d,'build_clip_optimizer',lambda model,_:torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad],lr=.001))
    monkeypatch.setattr(native,'fixed_probe_batches',lambda *_:[batch])
    monkeypatch.setattr(policy,'AlignmentGate',lambda *_:BranchOnlyScores([1.,-1.,1.]*4))
    synced = []
    summary = native.run(tmp_path,lambda:synced.append(1),p)
    assert summary['optimizer_steps'] == 204 and len(models) == 3
    assert len(probe_calls) == 3*2*2*3 == 36  # branches only, no trunk probes
    assert len(list(tmp_path.glob('seed_*/step_*/stream_*/result.json'))) == 12
    assert len(list(tmp_path.rglob('features_*.npz'))) == 48 and not list(tmp_path.rglob('*.pt'))
    for audit in tmp_path.glob('seed_*/trunk_audit.json'):
        value = json.loads(audit.read_text())
        assert value['history'] == 'native_only' and value['auxiliary_updates'] == 0
        assert value['probe_calls'] == 0 and value['decisions'] == []
    for point in tmp_path.glob('seed_*/start_*.json'):
        value = json.loads(point.read_text())
        assert value['arm'] == 'baseline' and value['prior_auxiliary_updates'] == 0
    for path in tmp_path.glob('seed_*/step_*/stream_*/result.json'):
        value = json.loads(path.read_text())
        assert value['starting_state_history'] == 'native_only'
        assert value['audit']['matched_rng_and_lrs'] and value['audit']['source_immutable']
        assert len(set(value['audit']['branch_start_hashes'].values())) == 1
    complete = json.loads((tmp_path/'completion.json').read_text())
    assert complete['status'] == 'complete' and complete['trunk_auxiliary_updates'] == 0
    assert complete['trunk_probe_calls'] == 0 and complete['optimizer_steps'] == 204
    assert json.loads((tmp_path/'comparison.json').read_text()) == summary


def test_native_delivery_pins_and_separate_namespace():
    cell = d.ROOT/'colabs/clip_native_history_policy_drive_one_cell.py'
    if not cell.exists():
        pytest.skip('delivery cell is not part of runtime source overlay')
    assignments = {n.targets[0].id:ast.literal_eval(n.value)
        for n in ast.parse(cell.read_text()).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'BASE_COMMIT','BUNDLE_SHA256','PAYLOAD'}}
    raw = zlib.decompress(base64.b64decode(assignments['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == assignments['BUNDLE_SHA256']
    assert assignments['BUNDLE_SHA256'] != native.load_protocol()['parent_bundle_id']
    sources = json.loads(raw)
    pins = json.loads(sources.pop('native_history_policy_source_manifest.json'))
    assert set(pins) == set(sources)
    for relative,source in sources.items():
        assert hashlib.sha256(source.encode()).hexdigest() == pins[relative]
        assert (d.ROOT/relative).read_text() == source
    assert 'clip_native_history_policy_logs' in cell.read_text()
    assert 'run_clip_native_history_policy.py' in cell.read_text()
    assert 'run_clip_fresh_split_policy.py' not in cell.read_text()
    assert 'PARTIAL_DRIVE_SYNC_COMPLETE' in cell.read_text()
