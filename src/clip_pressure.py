"""Experiment C: seven-arm continuations under a frozen calibration procedure.

The preparation-time protocol, data plan and interpretation remain byte-pinned.
Calibration has no reporting input and performs no optimizer steps.
"""
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from model.clip_training import native_clip_contrastive_loss, tensor_gradient_norm
from src import clip_difficulty as b
from src import clip_checkpoint_diagnostic as d
from src import clip_weight_lr as wl
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src.clip_fresh_split_policy import split_indices, training_batches, FullPoolReport
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol

FREEZE_SHA256 = '3d7cdf94e080c646197935c43147daafeb1953933b512ab3548246ce19e7a954'
REFERENCE_SHA256 = '93ff05aa37c51827b5619e715b93be89aa1242e8c66ecadba095410af8feada3'
ARMS = ['native', 'historical_hard', 'historical_semihard', 'hard_low', 'semihard_low', 'hard_high', 'semihard_high']
REPLAY_ARMS = ARMS[:3]
CALIBRATED_ARMS = ARMS[3:]


def load_freeze():
    path = d.ROOT/'configs/clip_pressure_freeze_manifest.json'
    if d.checksum(path) != FREEZE_SHA256:
        raise ValueError('Pressure freeze manifest changed')
    frozen = json.loads(path.read_text())
    for name, sha in frozen['files'].items():
        if d.checksum(d.ROOT/name) != sha:
            raise ValueError('Frozen pressure file changed: '+name)
    return frozen


def load_protocol():
    load_freeze()
    return json.loads((d.ROOT/'configs/clip_pressure.json').read_text())


def load_reference():
    path = d.ROOT/'configs/clip_pressure_reference.json'
    if d.checksum(path) != REFERENCE_SHA256:
        raise ValueError('Pinned pressure replay reference changed')
    ref = json.loads(path.read_text())
    freeze = load_freeze()
    if ref['freeze_manifest_sha256'] != FREEZE_SHA256 or ref['parent_files_sha256'] != freeze['parent_evidence_files']:
        raise ValueError('Replay reference provenance differs from frozen plan')
    return ref


def runtime_protocol(frozen):
    """Expose inherited data/trunk settings separately from the issued protocol."""
    return {**b.load_protocol(), **frozen,
        'common_stream_checkpoint_step': frozen['common_rng_source_step'],
        'primary': frozen['analysis']['primary'],
        'trunk_optimizer_steps': frozen['budget']['trunk_optimizer_steps'],
        'branch_optimizer_steps': frozen['budget']['branch_optimizer_steps'],
        'optimizer_step_budget': frozen['budget']['total_optimizer_steps']}


def calibration_plan(split, plans, p):
    """Recompute deterministic membership; compare it to the pinned plan at run time."""
    result = {}
    for seed in p['training_seeds']:
        used = {i for u in p['stream_seeds'] for batch in plans[f'{seed}:{u}'] for i in batch}
        eligible = set(split['training'])-used-set(split['frozen_probes'])
        count = p['calibration']['batches_per_seed']*p['calibration']['batch_size']
        if len(eligible) < count:
            raise ValueError('Insufficient disjoint calibration images')
        ordered = sorted(eligible, key=lambda i: (hashlib.sha256(
            f'scss-pressure-calibration-v1:{seed}:{i}'.encode()).hexdigest(), i))[:count]
        if set(ordered) & (set(split['report']) | set(split['old_holdout'])):
            raise ValueError('Calibration/report overlap')
        size = p['calibration']['batch_size']
        result[str(seed)] = dict(eligible_count=len(eligible), eligible_indices_sorted=sorted(eligible),
                                batches=[ordered[j:j+size] for j in range(0, count, size)])
    return result


def parent_arm(arm):
    if arm not in ARMS:
        raise ValueError('Unknown pressure arm')
    return 'native' if arm == 'native' else ('synthetic_semihard' if 'semihard' in arm else 'synthetic_hard')


def _batch_norms(model, batch, config, coefficients, p, *, weighted=False):
    """Native and every supplied auxiliary share one forward and RNG realization."""
    device = next(model.parameters()).device
    parameters = tuple(v for v in model.parameters() if v.requires_grad)
    with d.observe(model):
        model.train()
        with clip_train._autocast(device, config['training']['mixed_precision']):
            out = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
            n = native_clip_contrastive_loss(out.logits)
            supports = b.select_supports(out.raw_similarity, k=p['support_size'], start_rank=p['semihard_start_rank'])
            auxiliary = {a: b.auxiliary(out, 'synthetic_'+a, supports) for a in ['hard', 'semihard']}
            losses = {'native': n, **{name: alpha*auxiliary[construction]
                                      for name, (construction, alpha) in coefficients.items()}}
        norms = {}
        for i, (name, loss) in enumerate(losses.items()):
            grads = torch.autograd.grad(loss, parameters, retain_graph=i < len(losses)-1, allow_unused=True)
            norm = tensor_gradient_norm(grads)
            minimum = 0. if weighted and name != 'native' else p['calibration']['minimum_individual_gradient_norm_exclusive']
            if not math.isfinite(norm) or norm <= minimum:
                raise ValueError(f'Invalid calibration gradient norm for {name}: {norm!r}')
            norms[name] = norm
            del grads
        return dict(native_norm=norms['native'], auxiliary_norms={k:v for k,v in norms.items() if k != 'native'},
                    ratios={k:v/norms['native'] for k,v in norms.items() if k != 'native'})


def calibrate_state(model, optimizer, scheduler, batches, config, p, *, commit):
    if len(batches) != p['calibration']['batches_per_seed']:
        raise ValueError('Wrong calibration batch count')
    record = dict(status='running', unweighted=[], verification=[], coefficients={}, targets={},
                  optimizer_updates=0, state_before_sha256=d.live_digest(model, optimizer, scheduler))
    def persist():
        policy.guarded(model, optimizer, scheduler, lambda: commit(record))
    try:
        persist()
        for batch in batches:
            record['unweighted'].append(policy.guarded(model, optimizer, scheduler,
                lambda: _batch_norms(model, batch, config, {a:(a, 1.) for a in ['hard','semihard']}, p)))
            persist()
        means = {a: statistics.fmean(v['ratios'][a] for v in record['unweighted']) for a in ['hard','semihard']}
        record['mean_unweighted_ratios'] = means
        specs = {a['name']:a for a in p['arms']}
        for arm in CALIBRATED_ARMS:
            spec = specs[arm]
            alpha = spec['target_ratio']/means[spec['construction']]
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError('Invalid calibrated coefficient')
            record['coefficients'][arm] = alpha
            record['targets'][arm] = spec['target_ratio']
        persist()
        coefficients = {a:(specs[a]['construction'], record['coefficients'][a]) for a in CALIBRATED_ARMS}
        for batch in batches:
            record['verification'].append(policy.guarded(model, optimizer, scheduler,
                lambda: _batch_norms(model, batch, config, coefficients, p, weighted=True)))
            persist()
        if [v['native_norm'] for v in record['unweighted']] != [v['native_norm'] for v in record['verification']]:
            raise AssertionError('Calibration verification did not replay native gradients')
        record['mean_verified_ratios'] = {a: statistics.fmean(v['ratios'][a] for v in record['verification']) for a in CALIBRATED_ARMS}
        record['relative_errors'] = {a: abs(record['mean_verified_ratios'][a]/record['targets'][a]-1) for a in CALIBRATED_ARMS}
        if any(v > p['calibration']['weighted_gradient_verification_relative_tolerance'] for v in record['relative_errors'].values()):
            raise ValueError('Weighted calibration failed target tolerance; no fallback or recalibration')
        record['state_after_sha256'] = d.live_digest(model, optimizer, scheduler)
        if record['state_after_sha256'] != record['state_before_sha256']:
            raise AssertionError('Calibration changed training state')
        record.update(status='complete', training_state_preserved=True)
        persist()
        return record
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        persist()
        raise


def calibrate_states(model, optimizer, scheduler, states, batches, config, p, *, commit):
    """Intentionally visit two starting states, then restore the complete live state."""
    original = deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), **d.rng_state()))
    original_gradients = [(v, None if v.grad is None else v.grad.detach().clone()) for v in model.parameters()]
    source = d.state_digest(states)
    records = {}
    with d.observe(model):
        try:
            for w in p['checkpoint_steps']:
                wl.prepare_branch(model, optimizer, scheduler, states[w], states[p['fixed_lr_checkpoint_step']],
                                  states[p['common_rng_source_step']])
                records[str(w)] = calibrate_state(model, optimizer, scheduler, batches, config, p,
                    commit=lambda record: commit(f'calibration_w{w}', record))
        finally:
            d.restore(model, optimizer, scheduler, original)
            # restore() clears .grad; the generic observation context preserves
            # RNG/modes/buffers but does not restore parameter gradients.
            for parameter, gradient in original_gradients:
                parameter.grad = gradient
    if d.state_digest(states) != source:
        raise AssertionError('Calibration changed source snapshots')
    return records


def branch_coefficients(calibration, p):
    """Validate the persisted calibration contract before deriving branch strengths."""
    if calibration.get('status') != 'complete' or calibration.get('training_state_preserved') is not True:
        raise ValueError('Incomplete calibration blocks continuation')
    specs = {a['name']:a for a in p['arms']}
    if list(specs) != ARMS or len(p['arms']) != len(ARMS):
        raise ValueError('Pressure arm order/grid changed')
    n = p['calibration']['batches_per_seed']
    if len(calibration['unweighted']) != n or len(calibration['verification']) != n:
        raise ValueError('Missing individual calibration records')
    result = {a:specs[a]['coefficient'] for a in REPLAY_ARMS}
    for a in CALIBRATED_ARMS:
        c = statistics.fmean(v['ratios'][specs[a]['construction']] for v in calibration['unweighted'])
        alpha = specs[a]['target_ratio']/c
        measured = statistics.fmean(v['ratios'][a] for v in calibration['verification'])
        if (not math.isfinite(alpha) or alpha <= 0 or calibration['coefficients'][a] != alpha
                or calibration['targets'][a] != specs[a]['target_ratio']
                or not math.isfinite(measured) or
                abs(measured/specs[a]['target_ratio']-1) > p['calibration']['weighted_gradient_verification_relative_tolerance']):
            raise ValueError('Invalid coefficient or failed calibration verification')
        result[a] = alpha
    return result


def pressure_probe(model, batch, config, arm, coefficient, *, k, start_rank):
    device = next(model.parameters()).device
    parameters = tuple(v for v in model.parameters() if v.requires_grad)
    with d.observe(model):
        model.train()
        with clip_train._autocast(device, config['training']['mixed_precision']):
            out = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
            n = native_clip_contrastive_loss(out.logits)
            supports = b.select_supports(out.raw_similarity, k=k, start_rank=start_rank)
            a = None if arm == 'native' else coefficient*b.auxiliary(out, parent_arm(arm), supports)
            total = n if a is None else n+a
        ng = torch.autograd.grad(n, parameters, retain_graph=True, allow_unused=True)
        ag = () if a is None else torch.autograd.grad(a, parameters, retain_graph=True, allow_unused=True)
        tg = torch.autograd.grad(total, parameters, allow_unused=True)
        nn, an, tn = (tensor_gradient_norm(g) for g in [ng,ag,tg])
        dot = sum(float((x.detach().float()*y.detach().float()).sum())
                  for x,y in zip(ng,ag) if x is not None and y is not None)
        clip_scale = min(1., config['optimizer']['max_gradient_norm']/(tn+1e-6))
        record = dict(native_norm=nn, weighted_aux_norm=an, ratio=an/nn if nn else None,
                      cosine=dot/(nn*an) if nn and an else None, total_norm_before_clip=tn,
                      clipping_scale=clip_scale, clipping_activated=clip_scale < 1.)
        if any(v is not None and not math.isfinite(v) for v in record.values()):
            raise ValueError('Nonfinite pressure probe')
        return record


def step_update(model, optimizer, scheduler, batch, config, arm, coefficient, *, k=8, start_rank=16):
    # Native still uses the parent's exact native graph; its coefficient argument
    # is only a required positive API placeholder and the returned treatment is 0.
    trace = b.step_update(model, optimizer, scheduler, batch, config, parent_arm(arm),
                          coefficient if arm != 'native' else 1., k=k, start_rank=start_rank)
    trace['clipping_scale'] = min(1., config['optimizer']['max_gradient_norm']/(trace['gradient_norm_before_clip']+1e-6))
    trace['clipping_activated'] = trace['clipping_scale'] < 1.
    return trace


def run_branches(model, optimizer, scheduler, states, batches, config, report, reference, *,
                 calibrations, p, observation_steps, pressure_offsets, weight_steps, common_step,
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
    calibration_hash = reset.digest(calibrations)
    for w in weight_steps:
        coefficients = branch_coefficients(calibrations[str(w)], p)
        if set(reference[str(w)]) != set(REPLAY_ARMS):
            raise ValueError('Missing historical replay arms')
        points, traces, starts, resets, probes = {}, {}, {}, {}, {}
        first_report, first_selection = None, None
        for arm in ARMS:
            coefficient = coefficients[arm]
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
                    if arm in REPLAY_ARMS and point != reference[str(w)][arm]['points'][str(t)]:
                        raise AssertionError('Experiment B historical point replay mismatch')
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
            if arm in REPLAY_ARMS:
                core = [{key: trace[key] for key in b.CORE_TRACE_KEYS} for trace in traces[arm]]
                if reset.digest(core) != reference[str(w)][arm]['traces_sha256']:
                    raise AssertionError('Experiment B historical trace replay mismatch')
            # Make each completed branch durable before starting the next one.
            policy.guarded(model, optimizer, scheduler, lambda: commit(f'branch_w{w}_{arm}',
                dict(points=points[arm], traces=traces[arm], pressure_probes=probes[arm])))
        cells[f'w{w}'] = dict(weight_step=w, lr_step=fixed_lr_step, points=points, traces=traces,
                             starts=starts, resets=resets, pressure_probes=probes, coefficients=coefficients,
                             calibration_sha256=reset.digest(calibrations[str(w)]))
    if reset.digest(calibrations) != calibration_hash:
        raise AssertionError('Frozen coefficients changed during continuation')
    if (d.state_digest(states) != source or frozen != d.state_digest(
            {n: v for n, v in model.named_parameters() if not v.requires_grad})):
        raise AssertionError('Source states or frozen parameters changed')
    return dict(cells=cells, calibration_sha256=calibration_hash, audit=dict(actual_optimizer_steps=len(weight_steps)*len(ARMS)*horizon,
        historical_b_exact_replay=True, calibration_verified=True, coefficients_fixed=True, source_immutable=True, frozen_parameters_unchanged=True,
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
    q = {level: {key: {m: cells[key]['semihard_'+level][m]-cells[key]['hard_'+level][m]
                      for m in wl.METRICS} for key in cells} for level in ['low','high']}
    return dict(Q=q,
        late_geometry_main_effect={m: (q['low'][late][m]+q['high'][late][m])/2 for m in wl.METRICS},
        late_construction_x_pressure={m: q['high'][late][m]-q['low'][late][m] for m in wl.METRICS},
        age_interaction_by_target={level: {m:q[level][late][m]-q[level][early][m] for m in wl.METRICS}
                                   for level in ['low','high']},
        mean_age_interaction={m: statistics.fmean(q[level][late][m]-q[level][early][m] for level in ['low','high'])
                              for m in wl.METRICS})


def summarize(rows, p):
    grid = {(r['training_seed'],r['stream_seed']):r for r in rows}
    if len(grid) != len(rows) or set(grid) != {(s,u) for s in p['training_seeds'] for u in p['stream_seeds']}:
        raise ValueError('Incomplete or duplicate pressure grid')
    keys = [f'w{w}' for w in p['checkpoint_steps']]
    for seed in p['training_seeds']:
        seed_rows = [grid[seed,u] for u in p['stream_seeds']]
        if len({r['calibration_sha256'] for r in seed_rows}) != 1:
            raise ValueError('Calibration differs across streams')
        for key in keys:
            if any(r['cells'][key]['coefficients'] != seed_rows[0]['cells'][key]['coefficients'] for r in seed_rows):
                raise ValueError('Coefficients differ across streams')
    for row in rows:
        audit = row['audit']
        if (set(row['cells']) != set(keys) or audit['actual_optimizer_steps'] != len(keys)*len(ARMS)*p['horizon']
                or any(audit[k] is not True for k in ['historical_b_exact_replay','calibration_verified',
                    'coefficients_fixed','source_immutable','frozen_parameters_unchanged','matched_rng_and_lr_all_branches',
                    'fresh_adamw_all_branches','observations_preserve_state','candidate_supports_disjoint'])
                or audit['reports_control_treatment'] is not False or audit['pressure_controls_coefficient'] is not False):
            raise ValueError('Failed pressure audit/grid')
        for key, cell in row['cells'].items():
            if key != f'w{cell["weight_step"]}' or cell['lr_step'] != p['fixed_lr_checkpoint_step'] or set(cell['points']) != set(ARMS):
                raise ValueError('Invalid pressure cell')
            for arm in ARMS:
                if (set(cell['points'][arm]) != {str(t) for t in p['observation_steps']}
                        or len(cell['traces'][arm]) != p['horizon']
                        or set(cell['pressure_probes'][arm]) != {str(t) for t in p['pressure_probe_offsets']}
                        or any(t['coefficient'] != cell['coefficients'][arm] for t in cell['traces'][arm])):
                    raise ValueError('Incomplete records or changing coefficients')
    per_seed = []
    for seed in p['training_seeds']:
        times = {}
        for t in p['observation_steps']:
            cells = {key: {a: {m: statistics.fmean(effects(grid[seed,u]['cells'][key],t)[a][m] for u in p['stream_seeds'])
                               for m in wl.METRICS} for a in ARMS} for key in keys}
            times[str(t)] = dict(effects=cells, contrasts=comparison(cells,p['checkpoint_steps']))
        per_seed.append(dict(training_seed=seed,timecourse=times))
    times = {}
    for t in p['observation_steps']:
        cells = {key: {a: {m: statistics.fmean(s['timecourse'][str(t)]['effects'][key][a][m] for s in per_seed)
                           for m in wl.METRICS} for a in ARMS} for key in keys}
        times[str(t)] = dict(effects=cells, contrasts=comparison(cells,p['checkpoint_steps']))
    count = sum(r['audit']['actual_optimizer_steps'] for r in rows)
    if count != p['budget']['branch_optimizer_steps'] or count+p['budget']['trunk_optimizer_steps'] != p['budget']['total_optimizer_steps']:
        raise ValueError('Pressure optimizer budget mismatch')
    return dict(primary=p['analysis']['primary'],primary_value=times[str(p['horizon'])]['contrasts']['late_geometry_main_effect']['loss'],
        per_seed=per_seed,timecourse=times,matched_sets=len(rows),cells=len(rows)*len(keys),branches=len(rows)*len(keys)*len(ARMS),
        replication_units=len(per_seed),branch_optimizer_steps=count,trunk_optimizer_steps=p['budget']['trunk_optimizer_steps'],
        optimizer_steps=p['budget']['total_optimizer_steps'],
        interpretation='Negative loss helps; positive R1 pp helps. Primary averages the later semi-minus-hard '
        'effect across the two initial mean pressure targets. Report each target and construction x pressure '
        'interaction, then weight-state dependence, then absolute native comparisons separately. '
        'Fixed coefficients come from training-only calibration, not report outcomes. Calibration does not '
        'match individual batches, parameter groups, clipping, AdamW updates or future trajectories. '
        'Three reused seeds; streams nested within seed. Historical branches are replay controls. '
        'No significance/equivalence, isolated-geometry, retrieval-win or independent-confirmation claim.')



def run(directory,sync,p=None,native_reference=None,reference=None):
    from transformers import AutoProcessor
    frozen_protocol = load_protocol() if p is None else p
    p = runtime_protocol(frozen_protocol)
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
    d.write_json(directory/'protocol.json',frozen_protocol)
    d.write_json(directory/'runtime_protocol.json',p)
    d.write_json(directory/'freeze_manifest.json',load_freeze())
    d.write_json(directory/'split_plan.json',split)
    plans,permutations = policy.make_plans(split['training'],p)
    d.write_json(directory/'branch_plans.json',plans)
    d.write_json(directory/'timing_permutations.json',permutations)
    for name,value in [('split_plan.json',split),('branch_plans.json',plans),('training_exclusion_indices.json',sorted(set(old)|set(split['report'])))]:
        if reset.digest(value) != native_reference[name]:
            raise AssertionError('Replay plan differs: '+name)
    d.write_json(directory/'native_replay_reference.json',native_reference)
    d.write_json(directory/'pressure_replay_reference.json',reference)
    plans = wl.common_plans(split['training'],p)
    d.write_json(directory/'common_branch_plans.json',plans)
    frozen_calibration = json.loads((d.ROOT/p['calibration']['plan']).read_text())
    if calibration_plan(split,plans,p) != frozen_calibration['seeds']:
        raise AssertionError('Frozen calibration membership differs')
    d.write_json(directory/'calibration_plan.json',frozen_calibration)
    sync()  # All protocol/data/reference files durable before model updates.
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
        cal_indices = [i for batch in frozen_calibration['seeds'][str(seed)]['batches'] for i in batch]
        used_indices = [i for u in p['stream_seeds'] for batch in plans[f'{seed}:{u}'] for i in batch]
        cal_keys = keys(cal_indices)
        if len(cal_keys) != len(cal_indices) or cal_keys & (keys(used_indices)|report_keys|old_keys|probe_keys):
            raise AssertionError('Calibration image identity overlap')
        def build_calibration():
            with d.observe(model):
                return d.build_batches(data,cal_indices,seed=seed,epoch=common['epoch'],canonical=False)
        cal_batches,cal_records = policy.guarded(model,optimizer,scheduler,build_calibration)
        if len(cal_batches) != p['calibration']['batches_per_seed']:
            raise AssertionError('Calibration batch count differs')
        d.write_json(folder/'calibration_batches.json',dict(records=cal_records,caption_epoch=common['epoch'],
            source_indices=cal_indices,disjoint_from_report_probes_and_continuation_by_image_key=True))
        sync()
        def cal_commit(label,record):
            d.write_json(folder/(label+'.json'),record)
            sync()
        calibrations = policy.guarded(model,optimizer,scheduler,lambda: calibrate_states(
            model,optimizer,scheduler,states,cal_batches,config,p,commit=cal_commit))
        for w in p['checkpoint_steps']:
            branch_coefficients(calibrations[str(w)],p)
        d.write_json(folder/'calibration.json',dict(training_seed=seed,weights=calibrations,
            calibration_sha256=reset.digest(calibrations),optimizer_updates=0))
        sync()  # Both weight calibrations durable before ANY continuation for this seed.
        del cal_batches,cal_records
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
                calibrations=calibrations,p=p,observation_steps=p['observation_steps'],
                pressure_offsets=p['pressure_probe_offsets'],weight_steps=p['checkpoint_steps'],
                common_step=p['common_stream_checkpoint_step'],fixed_lr_step=p['fixed_lr_checkpoint_step'],
                k=p['support_size'],start_rank=p['semihard_start_rank'],commit=pair_commit,save_point=save_point,
                progress=lambda text:print('PRESSURE_SET:',key,text,flush=True))
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
        native_trunk_exact_replay=True,historical_b_exact_replay=True,initial_calibration_verified=True,
        independent_test_set=False,report_excluded_from_training_and_probes=True,
        starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        common_batches_captions_rng_across_weights_and_lrs=True,
        fresh_adamw_all_branches=True,external_scheduler_source='fixed_checkpoint_100',
        candidate_supports_disjoint=True,pressure_matched=False,
        pressure_probe_records=summary['branches']*len(p['pressure_probe_offsets']),
        pressure_controls_coefficient=False,coefficients_fixed=True,
        calibration_states=len(p['training_seeds'])*len(p['checkpoint_steps']),calibration_optimizer_steps=0))
    sync()
    return summary
