"""Weight-state x external LR trajectory, fresh AdamW, fixed synthetic treatment.

All eight branches within a seed/stream share processed batches and RNG. Historical
native trunks and the early-weight/early-LR reset branches are exact replay controls.
"""
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src.clip_fresh_split_policy import split_indices, training_batches, FullPoolReport
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol

ARMS = ['native', 'sustained']
METRICS = ['loss', 'r1_pp']
FACTORS = ['weight_main_effect', 'lr_main_effect', 'weight_x_lr_interaction']
REFERENCE_SHA256 = '3902f35ce40a36e34c5cb16b21e3dd31f2664c27114b10fe46ac08a6bb2fd857'


def expected_protocol():
    p = native.load_protocol()
    p.update(experiment='clip_weight_lr_v1', arms=ARMS,
        lr_checkpoint_steps=[100, 500], common_stream_checkpoint_step=100,
        observation_steps=[0, 1, 5, 10, 25, 50], gate_refresh_offsets=[],
        primary='seed_mean_factorial_contrasts_of_sustained_minus_native_full512_loss_at_50',
        primary_contrasts=FACTORS, secondary_contrast='factorial_timecourse_and_retrieval',
        parent_experiment='clip_optimizer_reset_v1',
        parent_bundle_id='03fb0a137afdcb615d8cdd9ae2c35b7e799acc85706e8cf04eada81fad9eacfc',
        new_training_seeds=False, fresh_reporting_split_for_this_run=False,
        reset_moments=True, reset_parameter_step_counters=True,
        external_scheduler_source='lr_checkpoint_step', restart_warmup=False,
        common_batches_captions_rng_across_weights_and_lrs=True,
        common_caption_epoch='checkpoint_100_epoch', common_rng='checkpoint_100_rng',
        native_trunk_exact_replay_required=True, early_diagonal_exact_replay_required=True,
        native_reference_sha256=reset.REFERENCE_SHA256, reference_sha256=REFERENCE_SHA256)
    return p


def load_protocol():
    p = json.loads((d.ROOT/'configs/clip_weight_lr.json').read_text())
    if p != expected_protocol():
        raise ValueError('Frozen weight-LR protocol changed')
    return p


def load_reference():
    path = d.ROOT/'configs/clip_weight_lr_reference.json'
    if d.checksum(path) != REFERENCE_SHA256:
        raise ValueError('Pinned weight-LR replay reference changed')
    return json.loads(path.read_text())


def cell_key(weight_step, lr_step):
    return f'w{weight_step}_lr{lr_step}'


def common_plans(training, p):
    """Reuse the historical early stream at BOTH weight ages, without new sampling."""
    historical, _ = policy.make_plans(training, p)
    return {f'{s}:{u}': historical[f'{s}:{p["common_stream_checkpoint_step"]}:{u}']
            for s in p['training_seeds'] for u in p['stream_seeds']}


def _without_lr(groups):
    return [{k: v for k, v in g.items() if k not in ('lr', 'initial_lr')} for g in groups]


def prepare_branch(model, optimizer, scheduler, weight_state, lr_state, rng_state):
    """Restore weights, clear AdamW state, transplant only scheduler/LR and RNG."""
    d.restore(model, optimizer, scheduler, weight_state)
    audit = reset.reset_optimizer_state(model, optimizer, scheduler)
    groups = optimizer.state_dict()['param_groups']
    donor = lr_state['optimizer']['param_groups']
    if _without_lr(groups) != _without_lr(donor):
        raise ValueError('LR donor changes non-LR optimizer group settings or parameter order')
    # Restoring a constructed snapshot also restores common RNG without consuming it.
    crossed = {**weight_state, 'optimizer': deepcopy(optimizer.state_dict()),
               'scheduler': deepcopy(lr_state['scheduler'])}
    for dst, src in zip(crossed['optimizer']['param_groups'], donor):
        for key in ('lr', 'initial_lr'):
            if key not in src or not math.isfinite(src[key]) or src[key] < 0:
                raise ValueError('Missing or invalid donor LR')
            dst[key] = src[key]
    for key in ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
        crossed[key] = rng_state[key]
    d.restore(model, optimizer, scheduler, crossed)
    if optimizer.state or scheduler.last_epoch != lr_state['completed_updates']:
        raise AssertionError('Fresh AdamW or external scheduler initialization failed')
    if [g['lr'] for g in optimizer.param_groups] != scheduler.get_last_lr():
        raise AssertionError('Optimizer LR and external scheduler disagree')
    if d.state_digest(model.state_dict()) != d.state_digest(weight_state['model']):
        raise AssertionError('LR crossing changed weights')
    return audit


def run_factorial(model, optimizer, scheduler, states, batches, config, report, reference, *,
                  coefficient, observation_steps, weight_steps, lr_steps, common_step,
                  commit, save_point=lambda *_: None, progress=lambda _: None):
    horizon = len(batches)
    if (not batches or len(weight_steps) != 2 or weight_steps != sorted(set(weight_steps))
            or lr_steps != weight_steps or set(states) != set(weight_steps)
            or common_step != weight_steps[0] or not math.isfinite(coefficient) or coefficient <= 0
            or observation_steps != sorted(set(observation_steps))
            or observation_steps[0] != 0 or observation_steps[-1] != horizon):
        raise ValueError('Invalid weight-LR factorial protocol')
    if set(reference) != set(ARMS):
        raise ValueError('Both early diagonal replay arms required')
    source_hash = d.state_digest(states)
    frozen = d.state_digest({n: v for n, v in model.named_parameters() if not v.requires_grad})
    cells, reports_at_start, reference_rng, lr_traces = {}, {}, None, {}
    for w in weight_steps:
        for lr in lr_steps:
            key = cell_key(w, lr)
            points, traces, starts, resets = {}, {}, {}, {}
            for arm in ARMS:
                resets[arm] = prepare_branch(model, optimizer, scheduler, states[w], states[lr], states[common_step])
                starts[arm] = dict(state_sha256=d.live_digest(model, optimizer, scheduler),
                    weights_sha256=d.state_digest(model.state_dict()),
                    optimizer_sha256=d.state_digest(optimizer.state_dict()),
                    scheduler_sha256=d.state_digest(scheduler.state_dict()),
                    rng_sha256=d.state_digest(d.rng_state()), learning_rates=[g['lr'] for g in optimizer.param_groups])
                if arm != ARMS[0] and starts[arm] != starts[ARMS[0]]:
                    raise AssertionError('Native and treated branches do not start identically')
                policy.guarded(model, optimizer, scheduler, lambda: commit(f'start_{key}_{arm}',
                    dict(start=starts[arm], reset=resets[arm], weight_step=w, lr_step=lr,
                         coefficient=coefficient if arm == 'sustained' else 0.)))
                points[arm], traces[arm] = {}, []
                for offset in range(horizon + 1):
                    if offset in observation_steps:
                        evaluated = policy.guarded(model, optimizer, scheduler, lambda: report.evaluate(model))
                        if offset == 0:
                            reports_at_start.setdefault(w, evaluated)
                            if evaluated != reports_at_start[w]:
                                raise AssertionError('LR/treatment changed no-update report')
                        point = dict(report=evaluated, state_sha256=d.live_digest(model, optimizer, scheduler))
                        if w == lr == common_step and point != reference[arm]['points'][str(offset)]:
                            raise AssertionError('Early diagonal reset replay mismatch')
                        points[arm][str(offset)] = point
                        policy.guarded(model, optimizer, scheduler, lambda: save_point(key, arm, offset, point))
                        policy.guarded(model, optimizer, scheduler, lambda: progress(f'{key}/{arm}: {offset}/{horizon}'))
                    if offset == horizon:
                        break
                    trace = d.step_update(model, optimizer, scheduler, batches[offset], config,
                                          coefficient if arm == 'sustained' else 0.)
                    if (not optimizer.state or any(float(s['step']) != offset + 1 for s in optimizer.state.values())
                            or scheduler.last_epoch != lr + offset + 1):
                        raise AssertionError('AdamW counters or scheduler position differ')
                    traces[arm].append(trace)
                rng_trace = [t['rng_after_sha256'] for t in traces[arm]]
                if reference_rng is None:
                    reference_rng = rng_trace
                if rng_trace != reference_rng:
                    raise AssertionError('Unmatched RNG across factorial branches')
                learning_rates = [t['learning_rates'] for t in traces[arm]]
                lr_traces.setdefault(lr, learning_rates)
                if learning_rates != lr_traces[lr]:
                    raise AssertionError('LR trajectory differs across weights/treatments')
                if w == lr == common_step and reset.digest(traces[arm]) != reference[arm]['traces_sha256']:
                    raise AssertionError('Early diagonal trace replay mismatch')
            cells[key] = dict(weight_step=w, lr_step=lr, points=points, traces=traces, starts=starts, resets=resets)
    if (d.state_digest(states) != source_hash or frozen != d.state_digest(
            {n: v for n, v in model.named_parameters() if not v.requires_grad})):
        raise AssertionError('Source states or frozen parameters changed')
    return dict(cells=cells, audit=dict(actual_optimizer_steps=8*horizon,
        early_diagonal_exact_replay=True, source_immutable=True, frozen_parameters_unchanged=True,
        matched_rng_across_all_cells=True, matched_lr_within_lr_level=True,
        fresh_adamw_all_branches=True, observations_preserve_state=True, report_controls_treatment=False))


def cell_effect(cell, offset):
    t, n = (cell['points'][a][str(offset)]['report'] for a in ['sustained', 'native'])
    values = dict(loss=t['native_loss'] - n['native_loss'],
                  r1_pp=t['pool_retrieval']['mean_r1_percent'] - n['pool_retrieval']['mean_r1_percent'])
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite factorial outcome')
    return values


def factorial(effects, steps):
    e, l = steps
    ee, el, le, ll = (effects[cell_key(w, r)] for w, r in [(e,e), (e,l), (l,e), (l,l)])
    return {metric: dict(
        weight_main_effect=((le[metric]-ee[metric])+(ll[metric]-el[metric]))/2,
        lr_main_effect=((el[metric]-ee[metric])+(ll[metric]-le[metric]))/2,
        weight_x_lr_interaction=(ll[metric]-le[metric])-(el[metric]-ee[metric]),
        weight_effect_at_early_lr=le[metric]-ee[metric],
        weight_effect_at_late_lr=ll[metric]-el[metric],
        lr_effect_at_early_weights=el[metric]-ee[metric],
        lr_effect_at_late_weights=ll[metric]-le[metric]) for metric in METRICS}


def summarize(rows, p):
    grid = {(r['training_seed'], r['stream_seed']): r for r in rows}
    expected = {(s,u) for s in p['training_seeds'] for u in p['stream_seeds']}
    keys = {cell_key(w,lr) for w in p['checkpoint_steps'] for lr in p['lr_checkpoint_steps']}
    if len(grid) != len(rows) or set(grid) != expected or any(set(r['cells']) != keys for r in rows):
        raise ValueError('Incomplete or duplicate factorial grid')
    for row in rows:
        audit = row['audit']
        for check in ['early_diagonal_exact_replay', 'source_immutable', 'frozen_parameters_unchanged',
                      'matched_rng_across_all_cells', 'matched_lr_within_lr_level', 'fresh_adamw_all_branches',
                      'observations_preserve_state']:
            if audit.get(check) is not True:
                raise ValueError('Factorial audit failed: '+check)
        if audit.get('report_controls_treatment') is not False:
            raise ValueError('Report controls treatment')
        if audit['actual_optimizer_steps'] != 8*p['horizon']:
            raise ValueError('Per-stream factorial budget mismatch')
        for key, cell in row['cells'].items():
            if cell_key(cell['weight_step'], cell['lr_step']) != key or set(cell['points']) != set(ARMS):
                raise ValueError('Invalid factorial cell identity/arms')
            for arm in ARMS:
                if set(cell['points'][arm]) != {str(t) for t in p['observation_steps']}:
                    raise ValueError('Incomplete factorial timecourse')
                if len(cell['traces'][arm]) != p['horizon']:
                    raise ValueError('Incomplete factorial trace')
    count = sum(r['audit']['actual_optimizer_steps'] for r in rows)
    if count != p['branch_optimizer_steps'] or count+p['trunk_optimizer_steps'] != p['optimizer_step_budget']:
        raise ValueError('Factorial optimizer budget mismatch')
    per_seed = []
    for seed in p['training_seeds']:
        times = {}
        for t in p['observation_steps']:
            effects = {key: {m: statistics.fmean(cell_effect(grid[seed,u]['cells'][key], t)[m]
                       for u in p['stream_seeds']) for m in METRICS} for key in sorted(keys)}
            times[str(t)] = dict(cell_effects=effects, contrasts=factorial(effects,p['checkpoint_steps']))
        per_seed.append(dict(training_seed=seed, timecourse=times))
    averages = {}
    for t in p['observation_steps']:
        effects = {key: {m: statistics.fmean(s['timecourse'][str(t)]['cell_effects'][key][m]
                   for s in per_seed) for m in METRICS} for key in sorted(keys)}
        averages[str(t)] = dict(cell_effects=effects, contrasts=factorial(effects,p['checkpoint_steps']))
    return dict(primary=p['primary'], primary_values={k: averages[str(p['horizon'])]['contrasts']['loss'][k] for k in FACTORS},
        per_seed=per_seed, timecourse=averages, matched_sets=len(rows), cells=4*len(rows), branches=8*len(rows),
        replication_units=len(per_seed), branch_optimizer_steps=count, trunk_optimizer_steps=p['trunk_optimizer_steps'],
        optimizer_steps=p['optimizer_step_budget'],
        interpretation='Effects are sustained minus native; negative loss helps, positive R@1 pp helps. '
        'Average streams within seed, then seeds equally. Three predeclared factorial loss contrasts at horizon 50; '
        'other horizons/retrieval secondary. No significance, equivalence, monotonicity or representation-causation claim. '
        'Weights also encode native history. Fixed coefficient is not matched effective pressure. '
        'Reused seeds, early continuation streams and reporting pool; exploratory, not independent confirmation.')


def run(directory,sync,p=None,native_reference=None,reference=None):
    from transformers import AutoProcessor
    p = load_protocol() if p is None else p
    native_reference = reset.load_reference() if native_reference is None else native_reference
    reference = load_reference() if reference is None else reference
    directory = Path(directory)
    old = json.loads((d.ROOT/'configs/cub200_clip_diagnostic_holdout_indices.json').read_text())
    probe_layout = json.loads((d.ROOT/'experiment_results/clip_paired_updates_2026-09/training_source_indices.json').read_text())
    probes = [i for b in probe_layout for i in b]
    if len(old) != p['old_holdout_size'] or len(set(probes)) != p['frozen_probe_size']:
        raise AssertionError('Historical identities changed')
    split = split_indices(list(range(p['original_train_size'])),old,probes,p)
    if len(split['training']) != p['new_training_size']:
        raise AssertionError('Unexpected training population')
    exclusion = directory/'training_exclusion_indices.json'
    d.write_json(exclusion,sorted(set(old)|set(split['report'])))
    d.write_json(directory/'protocol.json',p)
    d.write_json(directory/'split_plan.json',split)
    plans,permutations = policy.make_plans(split['training'],p)
    d.write_json(directory/'branch_plans.json',plans)
    d.write_json(directory/'timing_permutations.json',permutations)
    for name,value in [('split_plan.json',split),('branch_plans.json',plans),('training_exclusion_indices.json',sorted(set(old)|set(split['report'])))]:
        if reset.digest(value) != native_reference[name]:
            raise AssertionError('Replay plan differs: '+name)
    d.write_json(directory/'native_replay_reference.json',native_reference)
    d.write_json(directory/'early_diagonal_replay_reference.json',reference)
    plans = common_plans(split['training'],p)
    d.write_json(directory/'common_branch_plans.json',plans)
    sync()  # Protocol and replay reference durable before any model update.
    rows, trunk_steps = [],0
    for seed in p['training_seeds']:
        clip_train.set_global_seed(seed)
        folder = directory/f'seed_{seed}'
        folder.mkdir(exist_ok=False)
        config = clip_train.load_training_config(d.ROOT/'configs/hf_cub200_clip_vit_b32_baseline.yaml')
        config['experiment']['seed'] = config['training']['seed'] = seed
        config['runtime']['num_workers'] = 0
        config['dataset']['diagnostic_holdout_indices'] = str(exclusion.resolve())
        device,info = clip_train.resolve_device(config['runtime'])
        processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
        data = clip_train.build_clip_training_data(config,processor,pin_memory=False)
        if data.train_dataset.source_indices != split['training'] or len(data.train_loader) != p['expected_batches_per_epoch']:
            raise AssertionError('Training exclusions/batch count differ')
        report_batches,records = d.build_batches(data,split['report'],seed=0,epoch=0,canonical=True)
        keys = lambda indices: {data.train_dataset.grouped_split.groups[i].image_key for i in indices}
        train_keys,report_keys,old_keys,probe_keys = (keys(split[k]) for k in ['training','report','old_holdout','frozen_probes'])
        if len(report_keys) != p['report_size'] or report_keys & (train_keys|old_keys|probe_keys) or not probe_keys <= train_keys:
            raise AssertionError('Image-level report overlap')
        ids = [hashlib.sha256(r['image_key'].encode()).hexdigest() for r in records]
        row_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if rows and row_hash != common_row_hash:
            raise AssertionError('Reporting identities differ across seeds')
        common_row_hash = row_hash
        d.write_json(folder/'data_roles.json',dict(report=records,train_source_indices=split['training'],
            row_order_sha256=row_hash,image_key_sha256=ids,disjoint_by_image_key=True,
            historically_unused_images_claimed=False,original_validation_not_evaluated=True))
        d.write_json(folder/'training_config.json',config)
        d.write_json(folder/'device.json',info)
        fixed_probe_batches(data,gate_protocol(),folder)  # Reproduce archived data construction, no gradient probes.
        seed_reference = native_reference['seeds'][str(seed)]
        for name in ['data_roles.json','training_batches.json']:
            if reset.digest(json.loads((folder/name).read_text())) != seed_reference[name]:
                raise AssertionError('Replay identities differ: '+name)
        model = d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
        trainable = d.configure_clip_trainable_parameters(model,config['model']['trainable_policy'])
        if (trainable['trainable_parameter_count'],trainable['trainable_tensor_count']) != (10895617,35):
            raise AssertionError('Trainable policy changed')
        d.write_json(folder/'trainable_policy.json',trainable)
        optimizer = d.build_clip_optimizer(model,config['optimizer'])
        scheduler = clip_train.build_scheduler(optimizer,warmup_steps=config['scheduler']['warmup_steps'],total_steps=p['scheduler_horizon'])
        sync()
        def commit(label,value):
            if label.startswith('start_') and value != seed_reference['starts'][str(value['completed_updates'])]:
                raise AssertionError('Native starting-state replay mismatch')
            d.write_json(folder/(label+'.json'),value)
            sync()
            if label == 'progress':
                print('TRUNK:',seed,value['completed_updates'],flush=True)
        states,trunk_audit = native.trunk(model,optimizer,scheduler,training_batches(data),config,p,seed=seed,commit=commit)
        if (trunk_audit['auxiliary_updates'] != 0 or trunk_audit['decisions'] or trunk_audit['probe_calls'] != 0
                or any(t['coefficient'] != 0. for t in trunk_audit['traces'])
                or trunk_audit['actual_optimizer_steps'] != p['trunk_updates']):
            raise AssertionError('Starting states must have strictly native-only history')
        if reset.digest(trunk_audit) != seed_reference['trunk_audit.json']:
            raise AssertionError('Native trunk audit replay mismatch')
        trunk_steps += trunk_audit['actual_optimizer_steps']
        d.write_json(folder/'trunk_audit.json',trunk_audit)
        report = FullPoolReport(report_batches,p['report_size'])
        for step, initial in states.items():
            if (initial['arm'] != 'baseline' or initial['starting_state_history'] != 'native_only'
                    or initial['prior_auxiliary_updates'] != 0 or initial['completed_updates'] != step
                    or initial['scheduler']['last_epoch'] != step):
                raise AssertionError('Starting-state identity/scheduler/history differs')
        common = states[p['common_stream_checkpoint_step']]
        for stream in p['stream_seeds']:
            key = f'{seed}:{stream}'
            pair_dir = folder/f'stream_{stream}'
            pair_dir.mkdir(parents=True,exist_ok=False)
            batches,batch_records = d.build_batches(data,[i for b in plans[key] for i in b],
                seed=seed,epoch=common['epoch'],canonical=False)
            if len(batches) != p['horizon']:
                raise AssertionError('Continuation length differs')
            batch_identity = dict(records=batch_records,caption_epoch=common['epoch'])
            old_key = f'{seed}:{p["common_stream_checkpoint_step"]}:{stream}'
            if reset.digest(batch_identity) != native_reference['pairs'][old_key]['training_batches_sha256']:
                raise AssertionError('Common early continuation batch/caption replay mismatch')
            d.write_json(pair_dir/'training_batches.json',batch_identity)
            sync()
            def pair_commit(label,value):
                d.write_json(pair_dir/(label+'.json'),value)
                sync()
            def save_point(cell,arm,offset,value):
                d.write_json(pair_dir/f'point_{cell}_{arm}_{offset:03d}.json',value)
                if offset == p['horizon']:
                    meta = cache.save_cache(pair_dir/f'features_{cell}_{arm}.npz',report.encoded,row_hash)
                    if meta['feature_sha256'] != value['report']['features_sha256']:
                        raise AssertionError('Endpoint cache differs from evaluated state')
                sync()
            result = run_factorial(model,optimizer,scheduler,states,batches,config,report,reference['pairs'][key]['arms'],
                coefficient=p['coefficient'],observation_steps=p['observation_steps'],
                weight_steps=p['checkpoint_steps'],lr_steps=p['lr_checkpoint_steps'],
                common_step=p['common_stream_checkpoint_step'],commit=pair_commit,save_point=save_point,
                progress=lambda text:print('WEIGHT_LR_SET:',key,text,flush=True))
            result.update(training_seed=seed,stream_seed=stream,report_row_order_sha256=row_hash,
                starting_state_history='native_only',prior_auxiliary_updates=0,
                common_caption_epoch=common['epoch'],common_rng_source_step=p['common_stream_checkpoint_step'],
                source_states_sha256=d.state_digest(states),batch_records_sha256=reset.digest(batch_identity))
            d.write_json(pair_dir/'result.json',result)
            rows.append(result)
            sync()
            del batches,batch_records
            gc.collect()
        del states,initial,common,model,optimizer,scheduler,data,report,report_batches
        gc.collect()
        torch.cuda.empty_cache()
    if trunk_steps != p['trunk_optimizer_steps']:
        raise AssertionError('Trunk budget differs')
    summary = summarize(rows,p)
    d.write_json(directory/'comparison.json',summary)
    d.write_json(directory/'completion.json',dict(status='complete',matched_sets=summary['matched_sets'],
        cells=summary['cells'],branches=summary['branches'],endpoint_caches=summary['branches'],
        point_records=summary['branches']*len(p['observation_steps']),
        trunk_optimizer_steps=trunk_steps,branch_optimizer_steps=summary['branch_optimizer_steps'],
        optimizer_steps=summary['optimizer_steps'],rules_refitted=False,new_training_seeds=False,
        native_trunk_exact_replay=True,early_diagonal_exact_replay=True,
        independent_test_set=False,report_excluded_from_training_and_probes=True,
        starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        common_batches_captions_rng_across_weights_and_lrs=True,
        fresh_adamw_all_branches=True,external_scheduler_source='lr_checkpoint_step'))
    sync()
    return summary
