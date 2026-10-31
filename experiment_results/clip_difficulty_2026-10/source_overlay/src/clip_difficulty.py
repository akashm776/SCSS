"""Experiment B: fixed difficulty regimes at two native weight states.

Rank bands are defined on current non-diagonal training similarities, including
negative margins. Selection never consumes report data. Historical native/top-8
branches must replay Experiment A under the common early LR exactly.
"""
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from model.clip_training import (native_clip_contrastive_loss, clip_relative_denominator_loss,
                                 synthetic_barycentric_weights, tensor_gradient_norm)
from src import clip_checkpoint_diagnostic as d
from src import clip_weight_lr as wl
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src.clip_fresh_split_policy import split_indices, training_batches, FullPoolReport
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol
from src.clip_geometry_v2_metrics import _top_k_support

ARMS = ['native', 'synthetic_hard', 'real_hard', 'real_semihard', 'synthetic_semihard']
REFERENCE_SHA256 = 'bad4069efccf6624070899a7021ccc8e0e235ea0a131ebce5d9138159c7037d4'
CORE_TRACE_KEYS = ['native_loss', 'total_loss', 'gradient_norm_before_clip', 'learning_rates',
                   'coefficient', 'rng_after_sha256']


def expected_protocol():
    p = wl.load_protocol()
    for key in ['lr_checkpoint_steps', 'primary_contrasts', 'early_diagonal_exact_replay_required']:
        p.pop(key)
    p.update(experiment='clip_difficulty_v1', arms=ARMS,
        fixed_lr_checkpoint_step=100, support_size=8, semihard_start_rank=16,
        rank_population='all_non_diagonal_images_in_current_batch',
        negative_margin_filter=False, tie_rule='legacy_top8_then_stable_index_order_on_remainder',
        real_rule='highest_similarity_in_corresponding_support_lowest_index_tie',
        pressure_probe_offsets=[0, 10, 25], pressure_matched=False,
        primary='seed_mean_late_minus_early_synthetic_semihard_minus_hard_full512_loss_at_50',
        secondary_contrast='absolute_native_control_effects_real_regime_interaction_timecourse_retrieval',
        branch_optimizer_steps=3000, optimizer_step_budget=4500,
        parent_experiment='clip_weight_lr_v1',
        parent_bundle_id='9d1f48b37336cc190ad7be33ce77dbc325bc1f7422e0742ab86d26bba9bc07b0',
        fixed_lr_replay_required=True, reference_sha256=REFERENCE_SHA256,
        external_scheduler_source='fixed_checkpoint_100',
        reports_select_difficulty=False, pressure_controls_coefficient=False)
    return p


def load_protocol():
    p = json.loads((d.ROOT/'configs/clip_difficulty.json').read_text())
    if p != expected_protocol():
        raise ValueError('Frozen difficulty protocol changed')
    return p


def load_reference():
    path = d.ROOT/'configs/clip_difficulty_reference.json'
    if d.checksum(path) != REFERENCE_SHA256:
        raise ValueError('Pinned difficulty replay reference changed')
    return json.loads(path.read_text())


def select_supports(raw, *, k=8, start_rank=16):
    """Reserve historical top-k first; rank the remainder with deterministic ties.

At B64, k8/start16 skips seven remaining negatives and takes eight: ranks
16..23 in this explicitly defined total ordering. No margin or species filter.
"""
    if (raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not torch.isfinite(raw).all()
            or k < 1 or start_rank <= k or start_rank+k-1 > len(raw)-1):
        raise ValueError('Invalid scores or insufficient disjoint rank-band support')
    scores = raw.detach().float()
    diagonal = torch.eye(len(raw), dtype=torch.bool, device=raw.device)
    _, hard = _top_k_support(scores, diagonal, k)
    remaining = scores.masked_fill(diagonal | hard, -torch.inf)
    order = torch.argsort(remaining, dim=1, descending=True, stable=True)
    begin = start_rank-1-k
    semi = torch.zeros_like(hard).scatter_(1, order[:, begin:begin+k], True)
    if ((hard & semi).any() or (semi & diagonal).any()
            or not (hard.sum(1) == k).all() or not (semi.sum(1) == k).all()):
        raise AssertionError('Overlapping/invalid difficulty supports')
    return {'hard': hard, 'semihard': semi}


def candidate_similarity(out, support, kind):
    if kind == 'real':
        index = out.raw_similarity.detach().float().masked_fill(~support, -torch.inf).argmax(1)
        return out.raw_similarity[torch.arange(len(index), device=index.device), index]
    if kind != 'synthetic':
        raise ValueError('Unknown candidate kind')
    weights = synthetic_barycentric_weights(support.to(out.raw_similarity.dtype), support, 'uniform_topk').detach()
    mixed = F.normalize(weights.to(out.image_features.dtype) @ out.image_features, dim=-1)
    return (out.text_features*mixed).sum(-1)


def auxiliary(out, arm, supports):
    if arm not in ARMS or arm == 'native':
        raise ValueError('Expected an auxiliary arm')
    if arm == 'synthetic_hard':
        # Preserve the entire historical loss graph/precision for exact replay.
        return d.auxiliary_losses(out)['uniform_top8']
    kind, regime = arm.split('_')
    similarity = candidate_similarity(out, supports[regime], kind)
    scale = out.logit_scale.detach()
    return clip_relative_denominator_loss(out.raw_similarity*scale, similarity*scale)[0]


def describe(values):
    values = values.detach().float()
    return dict(mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                nonpositive_fraction=float((values <= 0).float().mean()))


@torch.no_grad()
def selection_audit(out, supports):
    raw = out.raw_similarity.detach().float()
    positive = raw.diagonal()
    margins = positive[:, None]-raw
    hard, semi = supports['hard'], supports['semihard']
    gap = raw.masked_fill(~hard, torch.inf).min(1).values-raw.masked_fill(~semi, -torch.inf).max(1).values
    return dict(overlap_count=int((hard & semi).sum()), diagonal_count=int((hard | semi).diagonal().sum()),
        supports_sha256=d.state_digest(supports), support_gap=describe(gap),
        support_margins={r: describe(margins[s]) for r, s in supports.items()},
        candidate_margins={a: describe(positive-candidate_similarity(out, supports[a.split('_')[1]], a.split('_')[0]).float())
                           for a in ARMS if a != 'native'})


def pressure_probe(model, batch, config, arm, coefficient, *, k, start_rank):
    """Training-batch observation only; restores RNG/buffers/gradients/modes.

These norms do not tune coefficient or select an arm. No optimizer step.
"""
    parameters = tuple(v for v in model.parameters() if v.requires_grad)
    device = next(model.parameters()).device
    with d.observe(model):
        model.train()
        with clip_train._autocast(device, config['training']['mixed_precision']):
            out = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
            n = native_clip_contrastive_loss(out.logits)
            supports = select_supports(out.raw_similarity, k=k, start_rank=start_rank)
            a = None if arm == 'native' else coefficient*auxiliary(out, arm, supports)
        ng = torch.autograd.grad(n, parameters, retain_graph=a is not None, allow_unused=True)
        ag = () if a is None else torch.autograd.grad(a, parameters, allow_unused=True)
        nn, an = tensor_gradient_norm(ng), tensor_gradient_norm(ag)
        dot = sum(float((x.detach().float()*y.detach().float()).sum())
                  for x, y in zip(ng, ag) if x is not None and y is not None)
        values = dict(native_norm=nn, weighted_aux_norm=an,
                      ratio=an/nn if nn else None, cosine=dot/(nn*an) if nn and an else None)
        if any(v is not None and not math.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite pressure probe')
        return values


def step_update(model, optimizer, scheduler, batch, config, arm, coefficient, *, k=8, start_rank=16):
    if arm not in ARMS or not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Invalid difficulty arm/coefficient')
    device = next(model.parameters()).device
    parameters = tuple(v for v in model.parameters() if v.requires_grad)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with clip_train._autocast(device, config['training']['mixed_precision']):
        out = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
        n = native_clip_contrastive_loss(out.logits)
        supports = select_supports(out.raw_similarity, k=k, start_rank=start_rank)
        a = None if arm == 'native' else auxiliary(out, arm, supports)
        loss = n if a is None else n+coefficient*a
        selection = selection_audit(out, supports)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, config['optimizer']['max_gradient_norm'], error_if_nonfinite=True)
    rates = [g['lr'] for g in optimizer.param_groups]
    optimizer.step()
    with torch.no_grad():
        model.clip_model.logit_scale.clamp_(max=math.log(100.))
    scheduler.step()
    return dict(native_loss=float(n.detach()), total_loss=float(loss.detach()),
        gradient_norm_before_clip=float(norm), learning_rates=rates,
        coefficient=coefficient if arm != 'native' else 0., rng_after_sha256=d.state_digest(d.rng_state()),
        auxiliary_loss=None if a is None else float(a.detach()), selection=selection)


def run_branches(model, optimizer, scheduler, states, batches, config, report, reference, *,
                 coefficient, observation_steps, pressure_offsets, weight_steps, common_step,
                 fixed_lr_step, k, start_rank, commit, save_point=lambda *_: None, progress=lambda _: None):
    horizon = len(batches)
    if (not batches or weight_steps != sorted(set(weight_steps)) or len(weight_steps) != 2
            or set(states) != set(weight_steps) or common_step != weight_steps[0]
            or fixed_lr_step != common_step or observation_steps != sorted(set(observation_steps))
            or observation_steps[0] != 0 or observation_steps[-1] != horizon
            or pressure_offsets != sorted(set(pressure_offsets))
            or any(t < 0 or t >= horizon for t in pressure_offsets)
            or set(reference) != {str(w) for w in weight_steps}):
        raise ValueError('Invalid difficulty branch protocol')
    source = d.state_digest(states)
    frozen = d.state_digest({n: v for n, v in model.named_parameters() if not v.requires_grad})
    cells, rngs, rates, common_start = {}, None, None, None
    for w in weight_steps:
        points, traces, starts, resets, probes = {}, {}, {}, {}, {}
        first_report, first_selection = None, None
        for arm in ARMS:
            resets[arm] = wl.prepare_branch(model, optimizer, scheduler, states[w], states[fixed_lr_step], states[common_step])
            start = dict(state_sha256=d.live_digest(model, optimizer, scheduler),
                         weights_sha256=d.state_digest(model.state_dict()),
                         optimizer_sha256=d.state_digest(optimizer.state_dict()),
                         scheduler_sha256=d.state_digest(scheduler.state_dict()),
                         rng_sha256=d.state_digest(d.rng_state()), learning_rates=[g['lr'] for g in optimizer.param_groups])
            starts[arm] = start
            if starts[arm] != starts['native']:
                raise AssertionError('Unmatched branch start')
            identity = {key: start[key] for key in ['optimizer_sha256', 'scheduler_sha256', 'rng_sha256', 'learning_rates']}
            common_start = identity if common_start is None else common_start
            if common_start != identity:
                raise AssertionError('Unmatched optimizer/schedule/RNG across weights')
            policy.guarded(model, optimizer, scheduler, lambda: commit(f'start_w{w}_{arm}',
                dict(start=start, reset=resets[arm], weight_step=w, lr_step=fixed_lr_step,
                     coefficient=0. if arm == 'native' else coefficient)))
            points[arm], traces[arm], probes[arm] = {}, [], {}
            for t in range(horizon+1):
                if t in observation_steps:
                    evaluated = policy.guarded(model, optimizer, scheduler, lambda: report.evaluate(model))
                    if t == 0:
                        first_report = evaluated if first_report is None else first_report
                        if evaluated != first_report:
                            raise AssertionError('Treatment changed initial report')
                    point = dict(report=evaluated, state_sha256=d.live_digest(model, optimizer, scheduler))
                    if arm in ['native', 'synthetic_hard'] and point != reference[str(w)][arm]['points'][str(t)]:
                        raise AssertionError('Fixed-LR historical point replay mismatch')
                    points[arm][str(t)] = point
                    policy.guarded(model, optimizer, scheduler, lambda: save_point(f'w{w}', arm, t, point))
                    policy.guarded(model, optimizer, scheduler, lambda: progress(f'w{w}/{arm}: {t}/{horizon}'))
                if t == horizon:
                    break
                if t in pressure_offsets:
                    probes[arm][str(t)] = policy.guarded(model, optimizer, scheduler,
                        lambda: pressure_probe(model, batches[t], config, arm, coefficient, k=k, start_rank=start_rank))
                    policy.guarded(model, optimizer, scheduler,
                        lambda: commit(f'pressure_w{w}_{arm}_{t:03d}', probes[arm][str(t)]))
                trace = step_update(model, optimizer, scheduler, batches[t], config, arm, coefficient, k=k, start_rank=start_rank)
                if t == 0:
                    first_selection = trace['selection'] if first_selection is None else first_selection
                    if trace['selection'] != first_selection:
                        raise AssertionError('Identical states selected different initial candidates')
                if (not optimizer.state or any(float(s['step']) != t+1 for s in optimizer.state.values())
                        or scheduler.last_epoch != fixed_lr_step+t+1):
                    raise AssertionError('AdamW counters or scheduler position differ')
                traces[arm].append(trace)
            branch_rng, branch_lr = ([x[key] for x in traces[arm]] for key in ['rng_after_sha256', 'learning_rates'])
            rngs, rates = branch_rng if rngs is None else rngs, branch_lr if rates is None else rates
            if branch_rng != rngs or branch_lr != rates:
                raise AssertionError('Unmatched LR/RNG across branches')
            if arm in ['native', 'synthetic_hard']:
                core = [{key: trace[key] for key in CORE_TRACE_KEYS} for trace in traces[arm]]
                if reset.digest(core) != reference[str(w)][arm]['traces_sha256']:
                    raise AssertionError('Fixed-LR historical trace replay mismatch')
            # Make each completed branch durable before starting the next one.
            policy.guarded(model, optimizer, scheduler, lambda: commit(f'branch_w{w}_{arm}',
                dict(points=points[arm], traces=traces[arm], pressure_probes=probes[arm])))
        cells[f'w{w}'] = dict(weight_step=w, lr_step=fixed_lr_step, points=points, traces=traces,
                             starts=starts, resets=resets, pressure_probes=probes)
    if (d.state_digest(states) != source or frozen != d.state_digest(
            {n: v for n, v in model.named_parameters() if not v.requires_grad})):
        raise AssertionError('Source states or frozen parameters changed')
    return dict(cells=cells, audit=dict(actual_optimizer_steps=len(weight_steps)*len(ARMS)*horizon,
        fixed_lr_exact_replay=True, source_immutable=True, frozen_parameters_unchanged=True,
        matched_rng_and_lr_all_branches=True, fresh_adamw_all_branches=True,
        observations_preserve_state=True, reports_control_treatment=False,
        candidate_supports_disjoint=True, pressure_controls_coefficient=False))


def effects(cell, t):
    reports = {a: cell['points'][a][str(t)]['report'] for a in ARMS}
    values = {a: dict(loss=reports[a]['native_loss']-reports['native']['native_loss'],
        r1_pp=reports[a]['pool_retrieval']['mean_r1_percent']-reports['native']['pool_retrieval']['mean_r1_percent']) for a in ARMS}
    if not all(math.isfinite(v) for row in values.values() for v in row.values()):
        raise ValueError('Nonfinite treatment effect')
    return values


def comparison(cells, weight_steps):
    early, late = [f'w{w}' for w in weight_steps]
    result = {}
    for kind in ['synthetic', 'real']:
        simple = {key: {m: cells[key][kind+'_semihard'][m]-cells[key][kind+'_hard'][m]
                        for m in wl.METRICS} for key in cells}
        result[kind] = dict(semihard_minus_hard=simple,
            late_minus_early={m: simple[late][m]-simple[early][m] for m in wl.METRICS})
    return result


def summarize(rows, p):
    grid = {(r['training_seed'], r['stream_seed']): r for r in rows}
    if len(grid) != len(rows) or set(grid) != {(s,u) for s in p['training_seeds'] for u in p['stream_seeds']}:
        raise ValueError('Incomplete or duplicate difficulty grid')
    keys = [f'w{w}' for w in p['checkpoint_steps']]
    for row in rows:
        audit = row['audit']
        if (set(row['cells']) != set(keys) or audit['actual_optimizer_steps'] != len(keys)*len(ARMS)*p['horizon']
                or any(audit[k] is not True for k in ['fixed_lr_exact_replay', 'source_immutable',
                    'frozen_parameters_unchanged', 'matched_rng_and_lr_all_branches', 'fresh_adamw_all_branches',
                    'observations_preserve_state', 'candidate_supports_disjoint'])
                or any(audit[k] is not False for k in ['reports_control_treatment', 'pressure_controls_coefficient'])):
            raise ValueError('Failed difficulty audit/grid')
        for key, cell in row['cells'].items():
            if key != f'w{cell["weight_step"]}' or cell['lr_step'] != p['fixed_lr_checkpoint_step'] or set(cell['points']) != set(ARMS):
                raise ValueError('Invalid difficulty cell')
            for arm in ARMS:
                if (set(cell['points'][arm]) != {str(t) for t in p['observation_steps']}
                        or len(cell['traces'][arm]) != p['horizon']
                        or set(cell['pressure_probes'][arm]) != {str(t) for t in p['pressure_probe_offsets']}):
                    raise ValueError('Incomplete observations/traces/probes')
    def aggregate(items):
        times = {}
        for t in p['observation_steps']:
            cells = {key: {arm: {m: statistics.fmean(effects(row['cells'][key], t)[arm][m] for row in items)
                                for m in wl.METRICS} for arm in ARMS} for key in keys}
            times[str(t)] = dict(effects=cells, comparisons=comparison(cells, p['checkpoint_steps']))
        return times
    per_seed = [dict(training_seed=s, timecourse=aggregate([grid[s,u] for u in p['stream_seeds']])) for s in p['training_seeds']]
    # Complete balanced grid: make the nested average explicit rather than treating streams as seeds.
    times = {}
    for t in p['observation_steps']:
        cells = {key: {arm: {m: statistics.fmean(s['timecourse'][str(t)]['effects'][key][arm][m] for s in per_seed)
                            for m in wl.METRICS} for arm in ARMS} for key in keys}
        times[str(t)] = dict(effects=cells, comparisons=comparison(cells, p['checkpoint_steps']))
    count = sum(r['audit']['actual_optimizer_steps'] for r in rows)
    if count != p['branch_optimizer_steps'] or count+p['trunk_optimizer_steps'] != p['optimizer_step_budget']:
        raise ValueError('Difficulty optimizer budget mismatch')
    return dict(primary=p['primary'], primary_value=times[str(p['horizon'])]['comparisons']['synthetic']['late_minus_early']['loss'],
        per_seed=per_seed, timecourse=times, matched_sets=len(rows), cells=len(rows)*len(keys),
        branches=len(rows)*len(keys)*len(ARMS), replication_units=len(per_seed),
        branch_optimizer_steps=count, trunk_optimizer_steps=p['trunk_optimizer_steps'], optimizer_steps=p['optimizer_step_budget'],
        interpretation='Negative loss helps; positive R1 pp helps. Primary negative interaction means a relative shift '
        'toward semi-hard, not necessarily a crossover or positive usefulness. Require early semi-minus-hard > 0 '
        'and late < 0 for the directional crossover. Always inspect each arm versus native. '
        'Three reused training seeds; streams nested within seed, reporting pool reused. '
        'Rank-defined semi-hard can have nonpositive margins; synthetic geometry can reorder difficulty. '
        'Fixed coefficient is not matched gradient pressure. Pressure probes are observational only; '
        'no significance, retrieval-win, general curriculum or independent-confirmation claim.')


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
    d.write_json(directory/'difficulty_replay_reference.json',reference)
    plans = wl.common_plans(split['training'],p)
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
            result = run_branches(model,optimizer,scheduler,states,batches,config,report,reference['pairs'][key]['weights'],
                coefficient=p['coefficient'],observation_steps=p['observation_steps'],
                pressure_offsets=p['pressure_probe_offsets'],weight_steps=p['checkpoint_steps'],
                common_step=p['common_stream_checkpoint_step'],fixed_lr_step=p['fixed_lr_checkpoint_step'],
                k=p['support_size'],start_rank=p['semihard_start_rank'],commit=pair_commit,save_point=save_point,
                progress=lambda text:print('DIFFICULTY_SET:',key,text,flush=True))
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
        native_trunk_exact_replay=True,fixed_lr_exact_replay=True,
        independent_test_set=False,report_excluded_from_training_and_probes=True,
        starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        common_batches_captions_rng_across_weights_and_lrs=True,
        fresh_adamw_all_branches=True,external_scheduler_source='fixed_checkpoint_100',
        candidate_supports_disjoint=True,pressure_matched=False,
        pressure_probe_records=summary['branches']*len(p['pressure_probe_offsets']),
        pressure_controls_coefficient=False))
    sync()
    return summary
