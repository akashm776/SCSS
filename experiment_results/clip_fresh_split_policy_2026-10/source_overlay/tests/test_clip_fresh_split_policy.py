from copy import deepcopy
import ast
import base64
import hashlib
import json
from types import SimpleNamespace
import zlib

import pytest
import torch

from src import clip_fresh_split_policy as fresh
from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src.clip_gradient_stages import observational_state
from tests.test_clip_continuation_timecourse import make_pair
from tests.test_clip_policy_test import Scores


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def actual_split():
    p = fresh.load_protocol()
    old = json.loads((d.ROOT/'configs/cub200_clip_diagnostic_holdout_indices.json').read_text())
    layout = json.loads((d.ROOT/'experiment_results/clip_paired_updates_2026-09/training_source_indices.json').read_text())
    probes = [i for batch in layout for i in batch]
    split = fresh.split_indices(list(range(5994)),old,probes,p)
    return p,old,probes,split


def test_frozen_new_seed_budget_and_actual_image_split():
    p,old,probes,split = actual_split()
    assert p['optimizer_step_budget'] == 3*500 + 3*2*2*4*50 == 3900
    assert set(p['training_seeds']).isdisjoint([42,123,456,789,2026,31415])
    assert not p['independent_test_set'] and not p['historically_unused_images_claimed']
    assert len(split['training']) == 4458 and len(split['training'])//64 == 69
    assert len(split['report']) == 512 and len(set(probes)) == 1024
    assert set(probes) <= set(split['training'])
    assert not set(split['report']) & (set(old)|set(probes)|set(split['training']))
    assert split == fresh.split_indices(list(reversed(range(5994))),old[::-1],probes[::-1],p)
    plans,perms = policy.make_plans(split['training'],p)
    assert len(plans) == 12 and len({d.state_digest(v) for v in plans.values()}) == 12
    assert (plans,perms) == policy.make_plans(split['training'][::-1],p)
    for key,plan in plans.items():
        indices = [i for batch in plan for i in batch]
        assert len(plan) == 50 and all(len(batch) == 64 for batch in plan)
        assert len(set(indices)) == 3200 and set(indices) <= set(split['training'])
        assert sorted(perms[key]) == list(range(50))
    with pytest.raises(AssertionError,match='Repeated'):
        policy.make_plans(split['training'],{**p,'stream_seeds':[p['previous_stream_seeds'][0]]})


@pytest.mark.parametrize('population,old,probes,message', [
    ([0,0,1],[1],[],'Duplicate'), ([0,1],[1,1],[],'Duplicate'),
    ([0,1],[2],[],'Invalid'), ([0,1],[1],[1],'Invalid'),
    ([0,1],[1],[0],'Insufficient')])
def test_bad_split_rejected(population,old,probes,message):
    with pytest.raises(ValueError,match=message):
        fresh.split_indices(population,old,probes,{'report_size':1,'report_split_seed':17})


@pytest.mark.parametrize('field,value', [('coefficient',.2),('report_size',511),
    ('trunk_updates',501),('training_seeds',[789]),('report_controls_treatment',True),
    ('primary','other'),('new_external_dataset',True),('expected_batches_per_epoch',77)])
def test_changed_contract_rejected(tmp_path,monkeypatch,field,value):
    p = fresh.load_protocol()
    p[field] = value
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/clip_fresh_split_policy.json').write_text(json.dumps(p))
    monkeypatch.setattr(d,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Frozen'):
        fresh.load_protocol()


def toy_trunk():
    model,opt,sched,batch,config,initial,_,report = make_pair()
    batch = {**batch,'source_indices':torch.arange(10),'captions':['toy']*10}
    p = dict(trunk_updates=4,trunk_gate_refresh_steps=[1,3],checkpoint_steps=[1,4],
        alignment_threshold=0.,coefficient=.06)
    return model,opt,sched,batch,config,initial,report,p


def test_trunk_exact_exposure_commit_order_snapshots_and_matched_branches(monkeypatch):
    model,opt,sched,batch,config,initial,report,p = toy_trunk()
    saved = []
    original = d.step_update
    coefficients = []
    def counted(*args,**kwargs):
        if len(coefficients) == 1:
            assert 'gate_000001' in saved
        if len(coefficients) == 3:
            assert 'gate_000003' in saved
        coefficients.append(args[-1])
        return original(*args,**kwargs)
    monkeypatch.setattr(d,'step_update',counted)
    states,audit = fresh.trunk(model,opt,sched,iter([(1,batch)]*4),config,Scores([1.,0.]),p,
        seed=271828,commit=lambda label,value:saved.append(label))
    assert coefficients == [0.,.06,.06,0.]  # threshold equality is off
    assert audit['actual_optimizer_steps'] == 4 and not audit['reporting_access']
    assert set(states) == {1,4}
    assert states[1]['scheduler']['last_epoch']+3 == states[4]['scheduler']['last_epoch']
    assert states[1]['completed_updates'] == 1
    hashes = {k:d.state_digest(v) for k,v in states.items()}
    d.restore(model,opt,sched,states[1])
    for coefficient in [.06,.06,0.]:
        original(model,opt,sched,batch,config,coefficient)
    assert d.state_digest(model.state_dict()) == d.state_digest(states[4]['model'])
    assert d.state_digest(opt.state_dict()) == d.state_digest(states[4]['optimizer'])
    for state in states.values():
        result = policy.run_pair(model,opt,sched,state,[batch]*4,config,Scores([1.,-1.,1.]),report,
            coefficient=.06,threshold=0.,refresh_offsets=[0,1,3],observation_steps=[0,4],
            permutation=[1,2,3,0],commit=lambda *_:None)
        assert result['audit']['matched_rng_and_lrs']
        assert result['audit']['actual_optimizer_steps'] == 16
        assert len(set(result['audit']['branch_start_hashes'].values())) == 1
        assert result['exposure']['gated'] == result['exposure']['random_matched']
    assert {k:d.state_digest(v) for k,v in states.items()} == hashes


def test_trunk_probe_nonfinite_and_persistence_fail_before_affected_update(monkeypatch):
    model,opt,sched,batch,config,initial,report,p = toy_trunk()
    calls = []
    original = d.step_update
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(d,'step_update',counted)
    with pytest.raises(ValueError,match='Nonfinite'):
        fresh.trunk(model,opt,sched,iter([(1,batch)]*4),config,Scores([float('nan')]),p,
            seed=1,commit=lambda *_:None)
    assert len(calls) == 1
    d.restore(model,opt,sched,initial)
    calls.clear()
    def fail(label,value):
        if label.startswith('gate_'):
            raise OSError('Drive unavailable')
    with pytest.raises(OSError,match='Drive'):
        fresh.trunk(model,opt,sched,iter([(1,batch)]*4),config,Scores([1.]),p,seed=1,commit=fail)
    assert len(calls) == 1


def test_full_pool_report_observational_and_endpoint_cache(tmp_path):
    model,opt,sched,batch,config,initial,_,_ = make_pair()
    report = fresh.FullPoolReport([batch],10)
    before = policy.state_stamp(model,opt,sched)
    result = report.evaluate(model)
    assert policy.state_stamp(model,opt,sched) == before
    images,texts,scale = report.encoded
    logits = images.double() @ texts.double().T * scale
    target = torch.arange(10)
    expected = (torch.nn.functional.cross_entropy(logits,target)+
                torch.nn.functional.cross_entropy(logits.T,target))/2
    assert result['native_loss'] == pytest.approx(float(expected),abs=1e-14)
    assert result['partition_losses'] == {'full512':[result['native_loss']]}
    meta = cache.save_cache(tmp_path/'features.npz',report.encoded,'a'*64)
    assert meta['feature_sha256'] == result['features_sha256']
    restored = cache.load_cache(tmp_path/'features.npz',meta)
    assert d.feature_hash(restored) == result['features_sha256']
    assert policy.state_stamp(model,opt,sched) == before


def test_summary_seed_hierarchy_budget_and_no_double_count():
    p = fresh.load_protocol()
    rows = []
    for si,seed in enumerate(p['training_seeds']):
        for ki,step in enumerate(p['checkpoint_steps']):
            for ui,stream in enumerate(p['stream_seeds']):
                value = float(si*10+ki*2+ui)
                differences = {str(t):{c:dict(report_loss=value if t else 0.,pool_r1_pp=0.)
                    for c in policy.ARMS[1:]} for t in p['observation_steps']}
                rows.append(dict(training_seed=seed,checkpoint_step=step,stream_seed=stream,
                    differences=differences,audit={'actual_optimizer_steps':200},
                    exposure={a:0 for a in policy.ARMS},random_schedule_equals_gate=True))
    summary = fresh.summarize(rows,p)
    assert summary['primary_value'] == 11.5
    assert [s['primary'] for s in summary['per_seed']] == [1.5,11.5,21.5]
    assert summary['comparisons']['native']['report_loss'] == 11.5
    assert summary['optimizer_steps'] == 3900 and summary['branch_optimizer_steps'] == 2400
    assert len(summary['exposure']) == 12  # zero-exposure sets not dropped
    with pytest.raises(ValueError,match='Incomplete'):
        fresh.summarize(rows[:-1],p)
    bad = deepcopy(rows)
    bad[0]['audit']['actual_optimizer_steps'] += 1
    with pytest.raises(AssertionError,match='budget'):
        fresh.summarize(bad,p)


def test_previous_run_guard_only_blocks_this_experiment(tmp_path):
    from colabs.run_clip_fresh_split_policy import require_no_previous_run
    experiment = fresh.load_protocol()['experiment']
    old = tmp_path/'clip_policy_test_old'
    old.mkdir()
    (old/'run_manifest.json').write_text(json.dumps({'experiment':'clip_policy_test_v1'}))
    require_no_previous_run(tmp_path,experiment)
    new = tmp_path/'clip_fresh_split_policy_today'
    new.mkdir()
    (new/'run_manifest.json').write_text(json.dumps({'experiment':experiment,'status':'failed'}))
    with pytest.raises(RuntimeError,match='no automatic resume'):
        require_no_previous_run(tmp_path,experiment)


def test_end_to_end_orchestration_with_tiny_models(tmp_path,monkeypatch):
    """Exercise the actual run loop, persistence and caches without downloads/GPU."""
    import transformers
    from tests.test_clip_paired_updates import TinyCLIP
    p,old,probes,split = actual_split()
    p = {**p,'trunk_updates':4,'trunk_gate_refresh_steps':[1,3],
        'checkpoint_steps':[1,4],'horizon':4,'gate_refresh_offsets':[0,1,3],
        'observation_steps':[0,4],'trunk_optimizer_steps':12,'branch_optimizer_steps':192,
        'optimizer_step_budget':204,'expected_batches_per_epoch':2}
    _,_,_,batch,config,_,_,_ = make_pair()
    batch = {**batch,'source_indices':torch.arange(10),'captions':['toy']*10}
    report_batch = {**batch,'pixel_values':torch.randn(512,6),'input_ids':torch.randn(512,6),
        'species_ids':torch.arange(512)}
    config.update(experiment={},runtime={},dataset={},model={'checkpoint':'toy','trainable_policy':'toy'},
        scheduler={'warmup_steps':2})
    created = []
    class Backend:
        @staticmethod
        def from_pretrained(name):
            model = TinyCLIP()
            created.append(1)
            return model
    def data_builder(config,processor,**kwargs):
        excluded = json.loads(__import__('pathlib').Path(config['dataset']['diagnostic_holdout_indices']).read_text())
        training = sorted(set(range(5994))-set(excluded))
        assert training == split['training']
        assert config['training']['seed'] in p['training_seeds']
        dataset = SimpleNamespace(source_indices=training,set_epoch=lambda _:None,
            grouped_split=SimpleNamespace(groups=[SimpleNamespace(image_key=f'image_{i}') for i in range(5994)]))
        return SimpleNamespace(train_dataset=dataset,train_loader=[batch,batch])
    def batches(data,indices,*,seed,epoch,canonical):
        records = [dict(source_index=i,image_key=f'image_{i}',caption='toy',caption_index=0,species_id=i)
                   for i in indices]
        if canonical:
            assert indices == split['report']
            return [report_batch],records
        assert set(indices) <= set(split['training'])
        return [batch]*4,records
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *_:None)
    monkeypatch.setattr(fresh.clip_train,'load_training_config',lambda *_:deepcopy(config))
    monkeypatch.setattr(fresh.clip_train,'resolve_device',lambda *_:('cpu',{'device':'cpu'}))
    monkeypatch.setattr(fresh.clip_train,'build_clip_training_data',data_builder)
    monkeypatch.setattr(d,'build_batches',batches)
    monkeypatch.setattr(d,'CLIPEncoderBackend',Backend)
    monkeypatch.setattr(d,'configure_clip_trainable_parameters',lambda *_:{
        'trainable_parameter_count':10895617,'trainable_tensor_count':35})
    monkeypatch.setattr(d,'build_clip_optimizer',lambda m,_:torch.optim.AdamW(
        [v for v in m.parameters() if v.requires_grad],lr=.001))
    monkeypatch.setattr(fresh,'fixed_probe_batches',lambda *_:[batch])
    monkeypatch.setattr(policy,'AlignmentGate',lambda *_:Scores([1.,-1.,1.]*10))
    synced = []
    summary = fresh.run(tmp_path,lambda:synced.append(1),p)
    assert len(created) == 3 and summary['optimizer_steps'] == 204
    assert len(list(tmp_path.glob('seed_*/step_*/stream_*/result.json'))) == 12
    assert len(list(tmp_path.rglob('features_*.npz'))) == 48
    assert len(list(tmp_path.glob('seed_*/trunk_audit.json'))) == 3
    assert not list(tmp_path.rglob('*.pt')) and len(synced) > 48
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'complete'
    assert json.loads((tmp_path/'comparison.json').read_text()) == summary


def test_delivery_source_pins_and_no_old_inputs():
    cell = d.ROOT/'colabs/clip_fresh_split_policy_drive_one_cell.py'
    if not cell.exists():
        pytest.skip('delivery cell not included in runtime source overlay')
    assignments = {node.targets[0].id:ast.literal_eval(node.value)
        for node in ast.parse(cell.read_text()).body if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name)
        and node.targets[0].id in {'BASE_COMMIT','BUNDLE_SHA256','PAYLOAD'}}
    raw = zlib.decompress(base64.b64decode(assignments['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == assignments['BUNDLE_SHA256']
    sources = json.loads(raw)
    manifest = json.loads(sources.pop('fresh_policy_source_manifest.json'))
    assert set(manifest) == set(sources)
    for name,source in sources.items():
        assert hashlib.sha256(source.encode()).hexdigest() == manifest[name]
        assert (d.ROOT/name).read_text() == source
    runner = sources['colabs/run_clip_fresh_split_policy.py']
    assert 'source_preflight' not in runner and 'recover_inputs' not in runner
    assert "'failure.json'" in runner and '3900' in runner
    assert 'PARTIAL_DRIVE_SYNC_COMPLETE' in cell.read_text()
