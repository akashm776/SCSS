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

from src import clip_pressure as c
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
from tests.test_clip_difficulty import setup as difficulty_setup


@pytest.fixture(autouse=True)
def isolate():
    with observational_state(torch.nn.Identity()):
        yield


def toy_protocol():
    p = deepcopy(c.load_protocol())
    p.update(checkpoint_steps=[1,4],fixed_lr_checkpoint_step=1,common_rng_source_step=1,
             horizon=4,observation_steps=[0,1,4],pressure_probe_offsets=[0,2])
    for a in p['arms']:
        if a['name'] in c.REPLAY_ARMS[1:]: a['coefficient'] = .06
    return p


def setup():
    model,opt,sched,batch,config,states,report,_ = difficulty_setup()
    p = toy_protocol()
    calibration_batches = [{**batch,'pixel_values':torch.randn_like(batch['pixel_values']),
                           'input_ids':torch.randn_like(batch['input_ids'])} for _ in range(4)]
    records = policy.guarded(model,opt,sched,lambda:c.calibrate_states(
        model,opt,sched,states,calibration_batches,config,p,commit=lambda *_:None))
    reference = {}
    for w in p['checkpoint_steps']:
        reference[str(w)] = {}
        for arm in c.REPLAY_ARMS:
            wl.prepare_branch(model,opt,sched,states[w],states[1],states[1])
            points,traces = {},[]
            for t in range(5):
                points[str(t)] = dict(report=report.evaluate(model),state_sha256=d.live_digest(model,opt,sched))
                if t < 4:
                    traces.append(b.step_update(model,opt,sched,batch,config,c.parent_arm(arm),.06))
            reference[str(w)][arm] = dict(points=points,traces_sha256=reset.digest(
                [{k:v[k] for k in b.CORE_TRACE_KEYS} for v in traces]))
    return model,opt,sched,batch,config,states,report,reference,records,p


def run_toy(args, **kwargs):
    model,opt,sched,batch,config,states,report,ref,records,p = args
    options = dict(calibrations=records,p=p,observation_steps=p['observation_steps'],pressure_offsets=p['pressure_probe_offsets'],
        weight_steps=[1,4],common_step=1,fixed_lr_step=1,k=8,start_rank=16,commit=lambda *_:None)
    options.update(kwargs)
    return c.run_branches(model,opt,sched,states,[batch]*4,config,report,ref,**options)


def test_frozen_protocol_membership_and_reference():
    p, ref = c.load_protocol(), c.load_reference()
    assert p['budget']['total_optimizer_steps'] == 5700 and p['budget']['branches'] == 84
    _,_,_,split = actual_split()
    plans = wl.common_plans(split['training'], b.load_protocol())
    frozen = json.loads((d.ROOT/p['calibration']['plan']).read_text())
    assert c.calibration_plan(split,plans,p) == frozen['seeds']
    assert [frozen['seeds'][str(s)]['eligible_count'] for s in p['training_seeds']] == [278,286,296]
    archive = d.ROOT/'experiment_results/clip_difficulty_2026-10'
    if archive.exists():
        for key,pair in ref['pairs'].items():
            seed,stream = key.split(':');path = archive/f'seed_{seed}/stream_{stream}/result.json'
            assert d.checksum(path) == pair['source_result_sha256']
            row = json.loads(path.read_text())
            for w in p['checkpoint_steps']:
                for a in c.REPLAY_ARMS:
                    old = c.parent_arm(a)
                    assert pair['weights'][str(w)][a]['points'] == row['cells'][f'w{w}']['points'][old]
                    assert pair['weights'][str(w)][a]['traces_sha256'] == reset.digest(
                        [{k:t[k] for k in b.CORE_TRACE_KEYS} for t in row['cells'][f'w{w}']['traces'][old]])


@pytest.mark.parametrize('name', ['configs/clip_pressure.json','configs/clip_pressure_calibration_plan.json',
                                 'docs/clip-pressure.md','configs/clip_pressure_freeze_manifest.json'])
def test_frozen_file_mutation_rejected(tmp_path,monkeypatch,name):
    frozen = c.load_freeze()
    for n in [*frozen['files'],'configs/clip_pressure_freeze_manifest.json']:
        dest=tmp_path/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((d.ROOT/n).read_bytes())
    path=tmp_path/name;path.write_bytes(path.read_bytes()+b'\n')
    monkeypatch.setattr(d,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='changed'): c.load_protocol()


def test_calibration_ratios_fixed_strengths_and_inert_state():
    model,opt,sched,batch,config,states,report,ref,records,p = setup()
    before = policy.state_stamp(model,opt,sched)
    source = d.state_digest(states)
    again = policy.guarded(model,opt,sched,lambda:c.calibrate_states(
        model,opt,sched,states,[batch]*4,config,p,commit=lambda *_:None))
    assert policy.state_stamp(model,opt,sched) == before and d.state_digest(states) == source
    for result in [*records.values(),*again.values()]:
        coefs = c.branch_coefficients(result,p)
        assert coefs['native'] == 0. and coefs['historical_hard'] == .06
        assert result['status'] == 'complete' and len(result['unweighted']) == len(result['verification']) == 4
        for a in c.CALIBRATED_ARMS:
            construction = 'semihard' if 'semihard' in a else 'hard'
            expected = result['targets'][a]/sum(v['ratios'][construction] for v in result['unweighted'])*4
            assert coefs[a] == pytest.approx(expected)
            assert result['mean_verified_ratios'][a] == pytest.approx(result['targets'][a],rel=.00001)
    # Altering a frozen coefficient is rejected, not silently accepted/recalibrated.
    bad = deepcopy(records['1']);bad['coefficients']['hard_low'] *= 2
    with pytest.raises(ValueError,match='coefficient'):c.branch_coefficients(bad,p)


def test_calibration_uses_mean_of_batch_ratios(monkeypatch):
    model,opt,sched,batch,config,_,_,_ = difficulty_setup();p=toy_protocol();calls=[]
    native_norms = [1.,2.,4.,8.]
    def measure(_model,_batch,_config,coefficients,_p,*,weighted=False):
        i=len(calls)%4;calls.append(weighted)
        ratios={name:alpha*((i+1) if geometry=='hard' else 2*(i+1)) for name,(geometry,alpha) in coefficients.items()}
        return dict(native_norm=native_norms[i],auxiliary_norms={n:r*native_norms[i] for n,r in ratios.items()},ratios=ratios)
    monkeypatch.setattr(c,'_batch_norms',measure)
    result=c.calibrate_state(model,opt,sched,[batch]*4,config,p,commit=lambda _:None)
    assert result['mean_unweighted_ratios']==dict(hard=2.5,semihard=5.)
    assert result['coefficients']['hard_low']==.02/2.5
    assert result['coefficients']['semihard_high']==.07/5.
    assert calls==[False]*4+[True]*4


def test_failed_verification_persists_failure_and_restores_live_state(monkeypatch):
    model,opt,sched,batch,config,states,_,_ = difficulty_setup();p=toy_protocol();saved=[]
    before=policy.state_stamp(model,opt,sched);original=c._batch_norms
    def wrong(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs.get('weighted'):
            result['ratios']['hard_low'] *= 2
        return result
    monkeypatch.setattr(c,'_batch_norms',wrong)
    with pytest.raises(ValueError,match='target tolerance'):
        c.calibrate_states(model,opt,sched,states,[batch]*4,config,p,commit=lambda label,v:saved.append((label,deepcopy(v))))
    assert saved[-1][1]['status']=='failed' and saved[-1][1]['verification']
    assert policy.state_stamp(model,opt,sched)==before


@pytest.mark.parametrize('badnorm',[0.,float('nan'),float('inf')])
def test_invalid_norm_fails_with_record(monkeypatch,badnorm):
    model,opt,sched,batch,config,_,_,_=difficulty_setup();saved=[]
    monkeypatch.setattr(c,'tensor_gradient_norm',lambda _:badnorm)
    with pytest.raises(ValueError,match='gradient norm'):
        c.calibrate_state(model,opt,sched,[batch]*4,config,toy_protocol(),commit=lambda v:saved.append(deepcopy(v)))
    assert saved[-1]['status']=='failed' and 'Invalid calibration gradient' in saved[-1]['error']


def test_branches_replay_match_rng_lr_and_preserve_coefficients():
    args=setup();row=run_toy(args)
    sparse=run_toy(args,pressure_offsets=[],observation_steps=[0,4])
    assert row['audit']['actual_optimizer_steps']==56
    for key,cell in row['cells'].items():
        for a in c.ARMS:
            assert cell['points'][a]['4']==sparse['cells'][key]['points'][a]['4']
            assert cell['traces'][a]==sparse['cells'][key]['traces'][a]
            assert all(v['coefficient']==cell['coefficients'][a] for v in cell['traces'][a])
            assert set(cell['pressure_probes'][a])=={'0','2'}
            for probe in cell['pressure_probes'][a].values():
                assert probe['clipping_activated']==(probe['clipping_scale']<1)
                assert probe['total_norm_before_clip']>0


@pytest.mark.parametrize('replay_arm',c.REPLAY_ARMS)
def test_any_historical_replay_failure_blocks_calibrated_arms(monkeypatch,replay_arm):
    args=setup();args[7]['1'][replay_arm]['traces_sha256']='wrong'
    original=c.step_update
    def guarded(*a,**kw):
        assert a[5] in c.REPLAY_ARMS
        return original(*a,**kw)
    monkeypatch.setattr(c,'step_update',guarded)
    with pytest.raises(AssertionError,match='replay'):run_toy(args)


def test_failed_calibration_persistence_blocks_continuation():
    model,opt,sched,batch,config,states,_,_=difficulty_setup();before=policy.state_stamp(model,opt,sched)
    def fail(*_):raise OSError('Drive unavailable')
    with pytest.raises(OSError):c.calibrate_states(model,opt,sched,states,[batch]*4,config,toy_protocol(),commit=fail)
    assert policy.state_stamp(model,opt,sched)==before


@pytest.mark.parametrize('arm',c.REPLAY_ARMS)
def test_bfloat16_replay_and_calibration(arm,monkeypatch):
    model,opt,sched,batch,config,states,_,_=difficulty_setup()
    monkeypatch.setattr(c.clip_train,'_autocast',lambda *_:torch.autocast('cpu',dtype=torch.bfloat16))
    wl.prepare_branch(model,opt,sched,states[4],states[1],states[1])
    trace=b.step_update(model,opt,sched,batch,config,c.parent_arm(arm),.06)
    expected=d.live_digest(model,opt,sched)
    wl.prepare_branch(model,opt,sched,states[4],states[1],states[1])
    actual=c.step_update(model,opt,sched,batch,config,arm,0. if arm=='native' else .06)
    assert {k:actual[k] for k in b.CORE_TRACE_KEYS}=={k:trace[k] for k in b.CORE_TRACE_KEYS}
    assert d.live_digest(model,opt,sched)==expected
    result=c.calibrate_state(model,opt,sched,[batch]*4,config,toy_protocol(),commit=lambda _:None)
    assert max(result['relative_errors'].values())<=.05


def test_delivery_checksums():
    path=d.ROOT/'colabs/clip_pressure_drive_one_cell.py'
    if not path.exists():pytest.skip('Delivery not part of source overlay')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==values['BUNDLE_SHA256']
    sources=json.loads(raw);pins=json.loads(sources.pop('pressure_source_manifest.json'))
    assert set(sources)==set(pins)
    for name,source in sources.items():
        assert (d.ROOT/name).read_text()==source and hashlib.sha256(source.encode()).hexdigest()==pins[name]


def test_separate_run_guard(tmp_path):
    from colabs.run_clip_pressure import require_no_previous_run
    previous=tmp_path/'clip_difficulty_old';previous.mkdir()
    (previous/'run_manifest.json').write_text(json.dumps({'experiment':'clip_difficulty_v1'}))
    require_no_previous_run(tmp_path,'clip_pressure_v1')
    current=tmp_path/'clip_pressure_today';current.mkdir()
    (current/'run_manifest.json').write_text(json.dumps({'experiment':'clip_pressure_v1'}))
    with pytest.raises(RuntimeError,match='no automatic resume'):require_no_previous_run(tmp_path,'clip_pressure_v1')


def test_primary_interaction_and_nested_averaging():
    p=c.runtime_protocol(c.load_protocol());rows=[]
    audit=dict(actual_optimizer_steps=700,historical_b_exact_replay=True,calibration_verified=True,
        coefficients_fixed=True,source_immutable=True,frozen_parameters_unchanged=True,
        matched_rng_and_lr_all_branches=True,fresh_adamw_all_branches=True,
        observations_preserve_state=True,reports_control_treatment=False,candidate_supports_disjoint=True,
        pressure_controls_coefficient=False)
    for i,seed in enumerate(p['training_seeds']):
        for j,stream in enumerate(p['stream_seeds']):
            cells={}
            for w in p['checkpoint_steps']:
                values=dict(native=0.,historical_hard=.5+i+j,historical_semihard=.25+i+j,
                    hard_low=1+i+j,hard_high=2+i+j,
                    semihard_low=1+i+j+(-2 if w==500 else 1),
                    semihard_high=2+i+j+(4 if w==500 else 2))
                points={a:{str(t):dict(report=dict(native_loss=100+w+v*t/50,
                    pool_retrieval={'mean_r1_percent':5.})) for t in p['observation_steps']} for a,v in values.items()}
                cells[f'w{w}']=dict(weight_step=w,lr_step=100,points=points,coefficients={a:0. if a=='native' else .1 for a in c.ARMS},
                    traces={a:[dict(coefficient=0. if a=='native' else .1)]*50 for a in c.ARMS},
                    pressure_probes={a:{str(t):{} for t in p['pressure_probe_offsets']} for a in c.ARMS})
            rows.append(dict(training_seed=seed,stream_seed=stream,cells=cells,audit=deepcopy(audit),calibration_sha256=str(seed)))
    result=c.summarize(rows,p);contrast=result['timecourse']['50']['contrasts']
    assert result['primary_value']==1.
    assert contrast['Q']['low']['w500']['loss']==-2.
    assert contrast['Q']['high']['w500']['loss']==4.
    assert contrast['late_construction_x_pressure']['loss']==6.
    assert contrast['mean_age_interaction']['loss']==-.5
    assert result['timecourse']['50']['effects']['w100']['hard_low']['loss']==2.5
    assert result['optimizer_steps']==5700 and result['branches']==84 and result['replication_units']==3
    with pytest.raises(ValueError,match='grid'):c.summarize(rows[:-1],p)
    rows[0]['cells']['w100']['coefficients']['hard_low']=.2
    with pytest.raises(ValueError,match='Coefficients differ'):c.summarize(rows,p)


def test_end_to_end_pressure_from_reproduced_parent(tmp_path,monkeypatch):
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

    pressure_ref={'pairs':{}}
    for seed in p['training_seeds']:
        for stream in p['stream_seeds']:
            row=read(destination/f'seed_{seed}/stream_{stream}/result.json')
            pressure_ref['pairs'][f'{seed}:{stream}']={'weights':{str(w):{
                a:dict(points=row['cells'][f'w{w}']['points'][c.parent_arm(a)],
                       traces_sha256=reset.digest([{k:t[k] for k in b.CORE_TRACE_KEYS}
                           for t in row['cells'][f'w{w}']['traces'][c.parent_arm(a)]]))
                for a in c.REPLAY_ARMS} for w in p['checkpoint_steps']}}
    cp=deepcopy(c.load_protocol())
    cp.update({k:bp[k] for k in ['trunk_updates','checkpoint_steps','horizon','expected_batches_per_epoch','observation_steps']})
    cp.update(fixed_lr_checkpoint_step=1,common_rng_source_step=1,pressure_probe_offsets=[0,2])
    cp['budget'].update(trunk_optimizer_steps=12,branch_optimizer_steps=336,total_optimizer_steps=348,
                        point_records=336,pressure_records=168)
    plan=c.calibration_plan(split,read(destination/'common_branch_plans.json'),cp)
    cal_path=tmp_path/'calibration_plan.json';d.write_json(cal_path,dict(seeds=plan))
    cp['calibration']['plan']=str(cal_path)
    monkeypatch.setattr(c,'fixed_probe_batches',probe_batches)
    output_c=tmp_path/'pressure';output_c.mkdir()
    original_step=c.step_update
    def checked_step(*args,**kwargs):
        # All seed-level calibrations must be durably recorded before updates.
        saved=list(output_c.glob('seed_*/calibration.json'))
        assert saved
        for path in saved:
            assert all(v['status']=='complete' for v in read(path)['weights'].values())
        return original_step(*args,**kwargs)
    monkeypatch.setattr(c,'step_update',checked_step)
    result=c.run(output_c,lambda:None,cp,ref,pressure_ref)
    assert result['optimizer_steps']==348 and result['branches']==84
    assert len(list(output_c.rglob('features_*.npz')))==84
    assert len(list(output_c.rglob('point_*.json')))==336
    assert len(list(output_c.rglob('pressure_*.json')))==169  # Includes replay-reference file.
    assert len(list(output_c.glob('seed_*/calibration_w*.json')))==6
    assert not list(output_c.rglob('*.pt'))
    assert read(output_c/'protocol.json')==cp
    assert read(output_c/'comparison.json')==result
    done=read(output_c/'completion.json')
    assert done['historical_b_exact_replay'] and done['initial_calibration_verified'] and done['coefficients_fixed']
    assert done['pressure_probe_records']==168 and done['calibration_optimizer_steps']==0
    for seed in cp['training_seeds']:
        cal=read(output_c/f'seed_{seed}/calibration.json')['weights']
        for stream in cp['stream_seeds']:
            row=read(output_c/f'seed_{seed}/stream_{stream}/result.json')
            for w in cp['checkpoint_steps']:
                assert row['cells'][f'w{w}']['coefficients']==c.branch_coefficients(cal[str(w)],cp)
