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

from src import clip_difficulty as b
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


def setup():
    model, opt, sched, _, config = toy()
    batch = dict(pixel_values=torch.randn(32, 6), input_ids=torch.randn(32, 6), species_ids=torch.arange(32))
    states = {}
    for step in range(1, 5):
        d.step_update(model, opt, sched, batch, config, 0.)
        if step in [1, 4]:
            states[step] = native.snapshot(model, opt, sched, epoch=step, step=step, seed=1)
    report = d.ReportPool([batch], {'fixed': {'batches': [list(range(32))]}})
    ref = {}
    for w in [1, 4]:
        ref[str(w)] = {}
        for arm in ['native', 'synthetic_hard']:
            wl.prepare_branch(model, opt, sched, states[w], states[1], states[1])
            points, traces = {}, []
            for t in range(5):
                points[str(t)] = dict(report=report.evaluate(model), state_sha256=d.live_digest(model, opt, sched))
                if t < 4:
                    traces.append(d.step_update(model, opt, sched, batch, config, .06 if arm != 'native' else 0.))
            ref[str(w)][arm] = dict(points=points, traces_sha256=reset.digest(traces))
    return model, opt, sched, batch, config, states, report, ref


def run_toy(args, **kwargs):
    model, opt, sched, batch, config, states, report, ref = args
    options = dict(coefficient=.06, observation_steps=[0, 1, 4], pressure_offsets=[0, 2],
        weight_steps=[1, 4], common_step=1, fixed_lr_step=1, k=8, start_rank=16, commit=lambda *_: None)
    options.update(kwargs)
    return b.run_branches(model, opt, sched, states, [batch]*4, config, report, ref, **options)


def test_frozen_protocol_and_parent_reference():
    p = b.load_protocol()
    assert p['optimizer_step_budget'] == 4500 and p['branch_optimizer_steps'] == 3*2*2*5*50
    assert p['fixed_lr_checkpoint_step'] == 100 and p['semihard_start_rank'] == 16
    assert not p['negative_margin_filter'] and not p['pressure_matched']
    ref = b.load_reference()
    assert len(ref['pairs']) == 6 and ref['parent_bundle_id'] == p['parent_bundle_id']
    archive = d.ROOT/'experiment_results/clip_weight_lr_2026-10'
    if archive.exists():
        for key, value in ref['pairs'].items():
            seed, stream = key.split(':')
            path = archive/f'seed_{seed}/stream_{stream}/result.json'
            assert d.checksum(path) == value['source_result_sha256']
            row = json.loads(path.read_text())
            for w in [100, 500]:
                for arm, old_arm in [('native', 'native'), ('synthetic_hard', 'sustained')]:
                    expected = value['weights'][str(w)][arm]
                    assert expected['points'] == row['cells'][f'w{w}_lr100']['points'][old_arm]
                    assert expected['traces_sha256'] == reset.digest(row['cells'][f'w{w}_lr100']['traces'][old_arm])


@pytest.mark.parametrize('key,value', [('coefficient', .1), ('fixed_lr_checkpoint_step', 500),
    ('semihard_start_rank', 8), ('pressure_matched', True), ('pressure_probe_offsets', [0]), ('negative_margin_filter', True)])
def test_changed_protocol_rejected(tmp_path, monkeypatch, key, value):
    expected = b.expected_protocol()
    changed = deepcopy(expected); changed[key] = value
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/clip_difficulty.json').write_text(json.dumps(changed))
    monkeypatch.setattr(b, 'expected_protocol', lambda: expected)
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Frozen'):
        b.load_protocol()


def test_rank_bands_exact_disjoint_and_include_misranked_negatives():
    scores = torch.arange(64).float().repeat(64, 1)/64
    supports = b.select_supports(scores)
    for i in range(64):
        order = sorted([j for j in range(64) if i != j], key=lambda j: -float(scores[i,j]))
        assert set(supports['hard'][i].nonzero().flatten().tolist()) == set(order[:8])
        assert set(supports['semihard'][i].nonzero().flatten().tolist()) == set(order[15:23])
    assert not (supports['hard'] & supports['semihard']).any()
    assert not (supports['hard'] | supports['semihard']).diagonal().any()
    assert (scores.diagonal()[:, None]-scores)[supports['hard']].min() < 0


def test_ties_keep_legacy_hard_support_and_disjoint_deterministic_semihard():
    scores = torch.zeros(64, 64)
    diagonal = torch.eye(64, dtype=torch.bool)
    supports = b.select_supports(scores)
    assert torch.equal(supports['hard'], b._top_k_support(scores, diagonal, 8)[1])
    for i in range(64):
        remaining = [j for j in range(64) if j != i and not supports['hard'][i,j]]
        assert set(supports['semihard'][i].nonzero().flatten().tolist()) == set(remaining[7:15])


@pytest.mark.parametrize('scores,options', [(torch.zeros(23,23), {}),
    (torch.full((64,64), float('nan')), {}), (torch.zeros(64,63), {}),
    (torch.zeros(64,64), {'start_rank':8})])
def test_bad_candidates_fail_without_fallback(scores, options):
    with pytest.raises(ValueError):
        b.select_supports(scores, **options)


def test_auxiliary_matches_historical_top8_and_embeds_have_gradients():
    model, _, _, batch, config, _, _, _ = setup()
    out = model(batch['pixel_values'], {'input_ids': batch['input_ids']})
    supports = b.select_supports(out.raw_similarity)
    assert torch.equal(b.auxiliary(out, 'synthetic_hard', supports), d.auxiliary_losses(out)['uniform_top8'])
    for arm in b.ARMS[1:]:
        value = b.auxiliary(out, arm, supports)
        grads = torch.autograd.grad(value, [model.clip_model.visual_projection.weight,
            model.clip_model.text_projection.weight, model.clip_model.logit_scale], retain_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads[:2])
        assert grads[2] is None
    audit = b.selection_audit(out, supports)
    assert audit['overlap_count'] == audit['diagonal_count'] == 0
    assert audit['support_gap']['min'] >= 0
    assert set(audit['candidate_margins']) == set(b.ARMS[1:])


def test_pressure_probe_preserves_state_gradients_modes_and_matches_autograd():
    model, opt, sched, batch, config, _, _, _ = setup()
    model.eval()
    before = policy.state_stamp(model, opt, sched)
    probe = b.pressure_probe(model, batch, config, 'synthetic_semihard', .06, k=8, start_rank=16)
    assert policy.state_stamp(model, opt, sched) == before
    assert probe['native_norm'] > 0 and probe['weighted_aux_norm'] > 0
    assert probe['ratio'] == probe['weighted_aux_norm']/probe['native_norm']
    out = model(batch['pixel_values'], {'input_ids':batch['input_ids']})
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    ng = d.flat_grad(b.native_clip_contrastive_loss(out.logits), parameters)
    ag = d.flat_grad(.06*b.auxiliary(out, 'synthetic_semihard', b.select_supports(out.raw_similarity)), parameters)
    assert probe['native_norm'] == pytest.approx(float(ng.norm()), rel=1e-6)
    assert probe['weighted_aux_norm'] == pytest.approx(float(ag.norm()), rel=1e-6)
    assert probe['cosine'] == pytest.approx(float(torch.nn.functional.cosine_similarity(ng, ag, dim=0)), abs=1e-6)


def test_replay_and_all_branches_match_lr_rng_and_inert_observations():
    args = setup(); saved = []
    row = run_toy(args, save_point=lambda *v: saved.append(v[:3]))
    assert len(saved) == 30 and row['audit']['actual_optimizer_steps'] == 40
    sparse = run_toy(args, observation_steps=[0, 4], pressure_offsets=[])
    for key, cell in row['cells'].items():
        assert len(cell['points']) == 5
        assert all(v == cell['starts']['native'] for v in cell['starts'].values())
        for arm in b.ARMS:
            assert cell['points'][arm]['4'] == sparse['cells'][key]['points'][arm]['4']
            assert cell['traces'][arm] == sparse['cells'][key]['traces'][arm]
            assert cell['resets'][arm]['old_step_counts'] == [cell['weight_step']]
            assert set(cell['pressure_probes'][arm]) == {'0', '2'}
    sequences = [[t['rng_after_sha256'] for t in c['traces'][a]] for c in row['cells'].values() for a in b.ARMS]
    assert all(v == sequences[0] for v in sequences)


@pytest.mark.parametrize('arm', ['native', 'synthetic_hard'])
def test_bfloat16_update_replays_legacy_exactly(arm, monkeypatch):
    model, opt, sched, batch, config, states, _, _ = setup()
    # Exercise quantized selection and loss math without claiming CUDA replay.
    monkeypatch.setattr(b.clip_train, '_autocast', lambda *_: torch.autocast('cpu', dtype=torch.bfloat16))
    wl.prepare_branch(model, opt, sched, states[4], states[1], states[1])
    trace = d.step_update(model, opt, sched, batch, config, .06 if arm != 'native' else 0.)
    expected = d.live_digest(model, opt, sched)
    wl.prepare_branch(model, opt, sched, states[4], states[1], states[1])
    actual = b.step_update(model, opt, sched, batch, config, arm, .06)
    assert {k:actual[k] for k in b.CORE_TRACE_KEYS} == trace
    assert d.live_digest(model, opt, sched) == expected


@pytest.mark.parametrize('where', ['point', 'trace'])
def test_failed_replay_stops_before_novel_regimes(where, monkeypatch):
    args = setup()
    if where == 'point':
        args[-1]['1']['native']['points']['0']['state_sha256'] = 'wrong'
    else:
        args[-1]['1']['native']['traces_sha256'] = 'wrong'
    original = b.step_update
    def guarded(*values, **kwargs):
        assert values[5] == 'native'
        return original(*values, **kwargs)
    monkeypatch.setattr(b, 'step_update', guarded)
    with pytest.raises(AssertionError, match='replay'):
        run_toy(args)


def test_failed_persistence_stops_before_training(monkeypatch):
    args = setup()
    monkeypatch.setattr(b, 'step_update', lambda *_a, **_k: pytest.fail('Updated after failed persistence'))
    def fail(*_): raise OSError('Drive unavailable')
    with pytest.raises(OSError):
        run_toy(args, commit=fail)


def test_regime_interaction_and_seed_averaging():
    p = b.load_protocol(); rows = []
    audit = dict(actual_optimizer_steps=500, fixed_lr_exact_replay=True, source_immutable=True,
        frozen_parameters_unchanged=True, matched_rng_and_lr_all_branches=True, fresh_adamw_all_branches=True,
        observations_preserve_state=True, reports_control_treatment=False, candidate_supports_disjoint=True,
        pressure_controls_coefficient=False)
    for i, seed in enumerate(p['training_seeds']):
        for j, stream in enumerate(p['stream_seeds']):
            cells = {}
            for w in p['checkpoint_steps']:
                values = dict(native=0., synthetic_hard=2+i+j,
                    synthetic_semihard=(3 if w == 100 else 1)+i+j,
                    real_hard=3+i+j, real_semihard=(5 if w == 100 else 2)+i+j)
                points = {a: {str(t): dict(report=dict(native_loss=100+w+values[a]*t/50,
                    pool_retrieval={'mean_r1_percent':10.})) for t in p['observation_steps']} for a in b.ARMS}
                cells[f'w{w}'] = dict(weight_step=w, lr_step=100, points=points,
                    traces={a:[{}]*50 for a in b.ARMS},
                    pressure_probes={a:{str(t):{} for t in p['pressure_probe_offsets']} for a in b.ARMS})
            rows.append(dict(training_seed=seed, stream_seed=stream, cells=cells, audit=deepcopy(audit)))
    summary = b.summarize(rows,p)
    assert summary['primary_value'] == -2.
    last = summary['timecourse']['50']
    assert last['comparisons']['real']['late_minus_early']['loss'] == -3.
    assert last['effects']['w100']['synthetic_hard']['loss'] == 3.5
    # A crossover does not imply benefit: both synthetic arm effects are positive.
    assert last['effects']['w500']['synthetic_semihard']['loss'] > 0
    assert summary['branches'] == 60 and summary['optimizer_steps'] == 4500
    assert summary['replication_units'] == 3
    with pytest.raises(ValueError, match='grid'): b.summarize(rows[:-1],p)
    rows[0]['audit']['pressure_controls_coefficient'] = True
    with pytest.raises(ValueError, match='audit'): b.summarize(rows,p)


def test_delivery_checksums():
    path = d.ROOT/'colabs/clip_difficulty_drive_one_cell.py'
    if not path.exists(): pytest.skip('Delivery not part of source overlay')
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == values['BUNDLE_SHA256']
    sources = json.loads(raw); pins = json.loads(sources.pop('difficulty_source_manifest.json'))
    assert set(sources) == set(pins)
    for name, source in sources.items():
        assert (d.ROOT/name).read_text() == source and hashlib.sha256(source.encode()).hexdigest() == pins[name]


def test_separate_run_guard(tmp_path):
    from colabs.run_clip_difficulty import require_no_previous_run
    old = tmp_path/'clip_weight_lr_old'; old.mkdir()
    (old/'run_manifest.json').write_text(json.dumps({'experiment':'clip_weight_lr_v1'}))
    require_no_previous_run(tmp_path,'clip_difficulty_v1')
    current = tmp_path/'clip_difficulty_today'; current.mkdir()
    (current/'run_manifest.json').write_text(json.dumps({'experiment':'clip_difficulty_v1'}))
    with pytest.raises(RuntimeError, match='no automatic resume'):
        require_no_previous_run(tmp_path,'clip_difficulty_v1')


def test_end_to_end_difficulty_from_reproduced_parent(tmp_path,monkeypatch):
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
    batch={**batch,"pixel_values":torch.randn(64,6),"input_ids":torch.randn(64,6),"species_ids":torch.arange(64)}
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


    difficulty_ref = {'pairs': {}}
    for seed in p['training_seeds']:
        for stream in p['stream_seeds']:
            row = read(output/f'seed_{seed}/stream_{stream}/result.json')
            difficulty_ref['pairs'][f'{seed}:{stream}'] = {'weights': {str(w): {
                arm: dict(points=row['cells'][f'w{w}_lr1']['points'][old_arm],
                          traces_sha256=reset.digest(row['cells'][f'w{w}_lr1']['traces'][old_arm]))
                for arm, old_arm in [('native','native'),('synthetic_hard','sustained')]}
                for w in p['checkpoint_steps']}}
    bp = {**b.load_protocol(), **{k: wp[k] for k in ['trunk_updates','checkpoint_steps','horizon',
        'trunk_optimizer_steps','expected_batches_per_epoch','observation_steps','common_stream_checkpoint_step']},
        'fixed_lr_checkpoint_step':1, 'pressure_probe_offsets':[0,2],
        'branch_optimizer_steps':240, 'optimizer_step_budget':252}
    monkeypatch.setattr(b,'fixed_probe_batches',probe_batches)
    destination = tmp_path/'difficulty'; destination.mkdir()
    summary = b.run(destination,lambda:None,bp,ref,difficulty_ref)
    assert summary['optimizer_steps']==252 and summary['replication_units']==3
    assert len(list(destination.glob('seed_*/stream_*/result.json')))==6
    assert len(list(destination.rglob('features_*.npz')))==60
    assert len(list(destination.rglob('point_*.json')))==240
    assert len(list(destination.rglob('pressure_*.json')))==120
    assert not list(destination.rglob('*.pt'))
    assert read(destination/'comparison.json')==summary
    completion = read(destination/'completion.json')
    assert completion['fixed_lr_exact_replay'] and completion['candidate_supports_disjoint']
    assert completion['branches']==60 and completion['pressure_probe_records']==120
    for path in destination.glob('seed_*/stream_*/result.json'):
        row = read(path)
        for cell in row['cells'].values():
            for arm in b.ARMS:
                for trace in cell['traces'][arm]:
                    assert trace['selection']['overlap_count']==0
                    assert trace['selection']['diagonal_count']==0
