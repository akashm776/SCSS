"""Verify the completed difficulty experiment and export small, caption-free evidence.

CPU only. Full input authenticates every file and recomputes 60 cached endpoints.
Clone-only mode verifies retained bytes and recomputes difficulty arithmetic; it
cannot evaluate omitted caches or recreate the GPU training invariants.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.analyze_clip_image_influence import query_losses, unpack
from scripts.analyze_clip_policy_robustness import checked_json, close, require, verify_input
from scripts.archive_clip_followup_evidence import digest
from scripts.archive_clip_october_evidence import locate, retain, verify_export, write_json

ROOT = Path(__file__).resolve().parents[1]
NAME = 'clip_difficulty_2026-10'
SPEC = ('clip_difficulty_20261007T114311_800923Z',
        '08dd8cacde749ee230d5c2dcadc7472328c7f0eabdbc8a4be8f41b93d4926afa',
        '85abe879f99d6411f710186e35ef404a9708a5e488745e67710fb63615739700')
SOURCE_MANIFEST_SHA = 'ddd3aa68bcde7545e419ad8eb07e567a32ee4bab935f60c1e0762aaf6c1a85b1'
CELLS = ['w100', 'w500']
ARMS = ['native', 'synthetic_hard', 'real_hard', 'real_semihard', 'synthetic_semihard']
CORE_TRACE_KEYS = ['native_loss', 'total_loss', 'gradient_norm_before_clip', 'learning_rates', 'coefficient', 'rng_after_sha256']


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def verify_summary(rows, summary, p):
    """Recompute every effect and interaction, independently of the training code."""
    require(summary['primary'] == p['primary'] and
            [summary[k] for k in ['matched_sets', 'cells', 'branches', 'replication_units',
                                  'trunk_optimizer_steps', 'branch_optimizer_steps', 'optimizer_steps']]
            == [6, 12, 60, 3, 1500, 3000, 4500], 'Wrong summary identity/budget')
    seeds = {s['training_seed']: s for s in summary['per_seed']}
    require(len(seeds) == len(summary['per_seed']) == 3 and set(seeds) == set(p['training_seeds']),
            'Wrong summary seed grid')
    for t in p['observation_steps']:
        per_seed = {}
        for seed in p['training_seeds']:
            effects = {}
            for key in CELLS:
                effects[key] = {}
                for arm in ARMS:
                    values = {m: [] for m in ['loss', 'r1_pp']}
                    for stream in p['stream_seeds']:
                        points = rows[seed, stream]['cells'][key]['points']
                        a, n = [points[x][str(t)]['report'] for x in [arm, 'native']]
                        values['loss'].append(a['native_loss']-n['native_loss'])
                        values['r1_pp'].append(a['pool_retrieval']['mean_r1_percent']-n['pool_retrieval']['mean_r1_percent'])
                    effects[key][arm] = {m: fmean(v) for m, v in values.items()}
            per_seed[seed] = effects
        aggregates = [(seeds[s]['timecourse'][str(t)], e) for s, e in per_seed.items()]
        averages = {key: {a: {m: fmean(e[key][a][m] for e in per_seed.values()) for m in ['loss','r1_pp']}
                          for a in ARMS} for key in CELLS}
        aggregates.append((summary['timecourse'][str(t)], averages))
        for reported, effects in aggregates:
            for key in CELLS:
                for arm in ARMS:
                    for m, value in effects[key][arm].items():
                        close(value, reported['effects'][key][arm][m], 'Arm effect')
            for kind in ['synthetic', 'real']:
                for m in ['loss', 'r1_pp']:
                    differences = [effects[key][kind+'_semihard'][m]-effects[key][kind+'_hard'][m] for key in CELLS]
                    for key, value in zip(CELLS, differences):
                        close(value, reported['comparisons'][kind]['semihard_minus_hard'][key][m], 'Regime difference')
                    close(differences[1]-differences[0], reported['comparisons'][kind]['late_minus_early'][m], 'Interaction')
    close(summary['primary_value'], summary['timecourse']['50']['comparisons']['synthetic']['late_minus_early']['loss'], 'Primary')


def diagnostics(files):
    """Descriptive, post-result summaries of prespecified geometry/pressure records.

No fitting, trimming, new training, or independent observations are introduced.
Initial geometry uses one common-state record per seed/stream, not five copies.
"""
    read = lambda name: checked_json(files[name])
    rows = [read(n) for n in sorted(files) if n.endswith('/result.json')]
    p = read('protocol.json')
    def nested_mean(value):
        return fmean(fmean(value(r) for r in rows if r['training_seed'] == seed) for seed in p['training_seeds'])
    pressure, geometry, pairs = {}, {}, []
    for key in CELLS:
        geometry[key] = {kind: {arm: {m: nested_mean(lambda r: r['cells'][key]['traces']['native'][0]['selection'][kind][arm][m])
                                     for m in ['mean', 'nonpositive_fraction']}
                                for arm in rows[0]['cells'][key]['traces']['native'][0]['selection'][kind]}
                         for kind in ['support_margins', 'candidate_margins']}
        pressure[key] = {str(t): {arm: {m: nested_mean(lambda r: r['cells'][key]['pressure_probes'][arm][str(t)][m])
                                       for m in ['native_norm', 'weighted_aux_norm', 'ratio', 'cosine']}
                                 for arm in ARMS[1:]} for t in p['pressure_probe_offsets']}
    for row in rows:
        e = {key: {a: row['cells'][key]['points'][a]['50']['report']['native_loss']
                       -row['cells'][key]['points']['native']['50']['report']['native_loss'] for a in ARMS}
             for key in CELLS}
        q = {kind: {key: e[key][kind+'_semihard']-e[key][kind+'_hard'] for key in CELLS} for kind in ['synthetic','real']}
        pairs.append(dict(training_seed=row['training_seed'], stream_seed=row['stream_seed'], effects=e,
                          differences=q, interactions={kind: v['w500']-v['w100'] for kind, v in q.items()}))
    def signs(values):
        return dict(negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values), positive=sum(v > 0 for v in values))
    selections = [t['selection'] for r in rows for c in r['cells'].values() for traces in c['traces'].values() for t in traces]
    return dict(source_run=SPEC[0], source_comparison_sha256=digest(files['comparison.json'])['sha256'],
        analysis='Descriptive audit of predefined records after results; no fitting or independent replication.',
        pairs=pairs, initial_geometry=geometry, pressure=pressure,
        effect_signs={key: {a: signs([r['effects'][key][a] for r in pairs]) for a in ARMS} for key in CELLS},
        difference_signs={kind: {key: signs([r['differences'][kind][key] for r in pairs]) for key in CELLS} for kind in ['synthetic','real']},
        interaction_signs={kind: signs([r['interactions'][kind] for r in pairs]) for kind in ['synthetic','real']},
        selection_trace_records=len(selections), logged_query_selections=len(selections)*p['batch_size'],
        overlap_total=sum(x['overlap_count'] for x in selections), diagonal_total=sum(x['diagonal_count'] for x in selections),
        minimum_support_gap=min(x['support_gap']['min'] for x in selections),
        support_gap_tie_query_fraction=fmean(x['support_gap']['nonpositive_fraction'] for x in selections),
        note='Pressure ratios/cosines are means of per-probe values. Initial geometry deduplicates arms; '
             'all-step separation counts include replayed/correlated states. Training gradients and candidate '
             'features were not retained, so these diagnostic records cannot be recomputed from endpoint caches.')


def validate(files, *, full):
    read = lambda name: checked_json(files[name])
    run, done = read('run_manifest.json'), read('completion.json')
    require(run['run_id'] == SPEC[0] and run['bundle_id'] == SPEC[2]
            and run['status'] == 'complete_pending_drive_flush', 'Wrong run/bundle/status')
    require(done['status'] == 'complete' and
            [done[k] for k in ['matched_sets', 'cells', 'branches', 'endpoint_caches',
                              'point_records', 'trunk_optimizer_steps',
                              'branch_optimizer_steps', 'optimizer_steps']]
            == [6, 12, 60, 60, 360, 1500, 3000, 4500], 'Incomplete run')
    require(done['pressure_probe_records'] == 180 and done['pressure_matched'] is False
            and done['pressure_controls_coefficient'] is False and done['fixed_lr_exact_replay'] is True,
            'Invalid pressure/replay completion')
    require(digest(files['source_manifest.json'])['sha256'] == SOURCE_MANIFEST_SHA,
            'Wrong source manifest')
    for path, sha in read('source_manifest.json').items():
        require(digest(files['source_overlay/'+path])['sha256'] == sha, 'Changed source: '+path)
    p = read('source_overlay/configs/clip_difficulty.json')
    ref = read('source_overlay/configs/clip_difficulty_reference.json')
    native = read('source_overlay/configs/clip_optimizer_reset_reference.json')
    require(read('protocol.json') == p and read('native_replay_reference.json') == native
            and read('difficulty_replay_reference.json') == ref, 'Changed protocol/reference')
    for name in ['split_plan.json', 'branch_plans.json', 'training_exclusion_indices.json']:
        require(canonical_sha(read(name)) == native[name], 'Changed replay plan')
    common = {f'{s}:{u}': read('branch_plans.json')[f'{s}:100:{u}']
              for s in p['training_seeds'] for u in p['stream_seeds']}
    require(read('common_branch_plans.json') == common, 'Changed common batch plan')
    rows, expected_paths, row_hashes, cache_count = {}, set(), set(), 0
    for seed in p['training_seeds']:
        folder = f'seed_{seed}/'
        for name in ['trunk_audit.json'] + (['data_roles.json', 'training_batches.json'] if full else []):
            require(canonical_sha(read(folder+name)) == native['seeds'][str(seed)][name],
                    'Native trunk/data replay differs')
        for step in [100, 500]:
            require(read(folder+f'start_{step:06d}.json') == native['seeds'][str(seed)]['starts'][str(step)],
                    'Native start replay differs')
        for stream in p['stream_seeds']:
            prefix = folder+f'stream_{stream}/'
            expected_paths.add(prefix+'result.json')
            row = read(prefix+'result.json')
            require((row['training_seed'], row['stream_seed']) == (seed, stream)
                    and row['starting_state_history'] == 'native_only'
                    and row['prior_auxiliary_updates'] == 0 and row['common_rng_source_step'] == 100,
                    'Wrong continuation identity')
            row_hashes.add(row['report_row_order_sha256'])
            expected_batch = native['pairs'][f'{seed}:100:{stream}']['training_batches_sha256']
            require(row['batch_records_sha256'] == expected_batch, 'Wrong batch identity')
            if full:
                batches = read(prefix+'training_batches.json')
                require(canonical_sha(batches) == expected_batch
                        and batches['caption_epoch'] == row['common_caption_epoch'], 'Changed batches/captions')
            audit = row['audit']
            require(audit['actual_optimizer_steps'] == 500 and audit['reports_control_treatment'] is False
                    and audit['pressure_controls_coefficient'] is False
                    and all(audit[k] is True for k in ['fixed_lr_exact_replay', 'source_immutable', 'frozen_parameters_unchanged',
                        'matched_rng_and_lr_all_branches', 'fresh_adamw_all_branches',
                        'observations_preserve_state', 'candidate_supports_disjoint']),
                    'Failed runtime audit')
            require(set(row['cells']) == set(CELLS), 'Wrong cell grid')
            rngs, rates, weights, schedulers, initial_reports = None, {}, {}, {}, {}
            for key, cell in row['cells'].items():
                w, lr = cell['weight_step'], cell['lr_step']
                require(key == f'w{w}' and lr == 100 and set(cell['points']) == set(ARMS), 'Wrong cell identity')
                require(all(cell['starts'][a] == cell['starts']['native'] for a in ARMS), 'Unmatched treatment start')
                for arm in ARMS:
                    start, reset = cell['starts'][arm], cell['resets'][arm]
                    weights.setdefault(w, start['weights_sha256'])
                    schedulers.setdefault(lr, (start['scheduler_sha256'], start['optimizer_sha256'], start['learning_rates']))
                    require(start['weights_sha256'] == weights[w] and
                            (start['scheduler_sha256'], start['optimizer_sha256'], start['learning_rates']) == schedulers[lr]
                            and start['rng_sha256'] == row['cells'][CELLS[0]]['starts']['native']['rng_sha256'],
                            'Unmatched factorial starts')
                    coefficient = 0. if arm == 'native' else p['coefficient']
                    require(read(prefix+f'start_{key}_{arm}.json') == dict(start=start, reset=reset,
                            weight_step=w, lr_step=lr, coefficient=coefficient), 'Standalone start differs')
                    require(reset['cleared_parameter_states'] == 35 and reset['old_step_counts'] == [w]
                            and reset['new_parameter_states'] == 0
                            and reset['weights_scheduler_groups_rng_gradients_unchanged'] is True
                            and reset['lazy_initialization_next_update_step'] == 1, 'Invalid optimizer reset')
                    traces = cell['traces'][arm]
                    require(len(traces) == 50 and all(t['coefficient'] == coefficient for t in traces), 'Bad treatment trace')
                    branch_rng = [t['rng_after_sha256'] for t in traces]
                    branch_lr = [t['learning_rates'] for t in traces]
                    rngs = branch_rng if rngs is None else rngs
                    rates.setdefault(lr, branch_lr)
                    require(branch_rng == rngs and branch_lr == rates[lr]
                            and branch_lr[0] == start['learning_rates'], 'Unmatched RNG/LR traces')
                    require(set(cell['points'][arm]) == {str(t) for t in p['observation_steps']}, 'Wrong observations')
                    for t in p['observation_steps']:
                        require(read(prefix+f'point_{key}_{arm}_{t:03d}.json') == cell['points'][arm][str(t)],
                                'Standalone observation differs')
                    initial = cell['points'][arm]['0']['report']
                    initial_reports.setdefault(w, initial)
                    require(initial == initial_reports[w], 'LR/treatment changed initial report')
                    if arm in ['native', 'synthetic_hard']:
                        expected = ref['pairs'][f'{seed}:{stream}']['weights'][str(w)][arm]
                        core = [{k: trace[k] for k in CORE_TRACE_KEYS} for trace in traces]
                        require(cell['points'][arm] == expected['points'] and
                                canonical_sha(core) == expected['traces_sha256'], 'Fixed-LR replay differs')
                    require(read(prefix+f'branch_{key}_{arm}.json') == dict(points=cell['points'][arm],
                            traces=traces, pressure_probes=cell['pressure_probes'][arm]), 'Standalone branch differs')
                    for trace in traces:
                        selection = trace['selection']
                        require(selection['overlap_count'] == selection['diagonal_count'] == 0
                                and selection['support_gap']['min'] >= 0, 'Invalid support separation record')
                        for stats in [selection['support_gap'], *selection['support_margins'].values(),
                                      *selection['candidate_margins'].values()]:
                            require(stats['min'] <= stats['mean'] <= stats['max']
                                    and 0 <= stats['nonpositive_fraction'] <= 1, 'Invalid geometry statistics')
                    require(traces[0]['selection'] == cell['traces']['native'][0]['selection'], 'Initial candidates differ')
                    probes = cell['pressure_probes'][arm]
                    require(set(probes) == {str(t) for t in p['pressure_probe_offsets']}, 'Wrong pressure probe grid')
                    for t, probe in probes.items():
                        require(read(prefix+f'pressure_{key}_{arm}_{int(t):03d}.json') == probe, 'Standalone pressure differs')
                        require(probe['native_norm'] > 0 and probe['weighted_aux_norm'] >= 0, 'Invalid gradient norm')
                        close(probe['ratio'], probe['weighted_aux_norm']/probe['native_norm'], 'Pressure ratio')
                        require((arm == 'native' and probe['cosine'] is None and probe['weighted_aux_norm'] == 0)
                                or (arm != 'native' and -1.000001 <= probe['cosine'] <= 1.000001), 'Invalid gradient cosine')
                    close(probes['0']['native_norm'], cell['pressure_probes']['native']['0']['native_norm'],
                          'Initial native pressure')
                    if full:
                        meta = read(prefix+f'features_{key}_{arm}.json')
                        logits = unpack(files[prefix+f'features_{key}_{arm}.npz'], meta, row['report_row_order_sha256'])
                        report = cell['points'][arm]['50']['report']
                        require(meta['feature_sha256'] == report['features_sha256'], 'Report/cache identity differs')
                        close(float(query_losses(logits)[:, 2].mean()), report['native_loss'], 'Cached loss')
                        t2i, i2t = [100*float(np.mean(logits.argmax(axis) == np.arange(512))) for axis in [1, 0]]
                        for metric, value in [('text_to_image_r1_percent', t2i), ('image_to_text_r1_percent', i2t),
                                              ('mean_r1_percent', (t2i+i2t)/2)]:
                            close(value, report['pool_retrieval'][metric], 'Cached R1')
                        cache_count += 1
            rows[seed, stream] = row
    require({n for n in files if n.endswith('/result.json')} == expected_paths and len(row_hashes) == 1,
            'Unexpected result grid/report identity')
    verify_summary(rows, read('comparison.json'), p)
    require(cache_count == (60 if full else 0), 'Wrong cache count')
    return cache_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path.home()/'Downloads')
    parser.add_argument('--output-root', type=Path, default=ROOT/'experiment_results')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    target = args.output_root/NAME
    if args.verify_only:
        verify_export(target, SPEC)
        audit = checked_json((target/'export_audit.json').read_bytes())
        files = {n: (target/n).read_bytes() for n in audit['files']}
        validate(files, full=False)
        require(checked_json((target/'diagnostics.json').read_bytes()) == diagnostics(files), 'Derived diagnostics differ')
        print('Retained replay records and all seed/timecourse difficulty arithmetic verified; caches omitted.')
        return
    require(not target.exists(), 'Refusing to overwrite '+str(target))
    files, provenance = verify_input(locate(args.downloads, SPEC[0]),
                                     dict(source_run=SPEC[0], backup_manifest_sha256=SPEC[1]))
    count = validate(files, full=True)
    selected = {n: b for n, b in files.items() if retain(n, b)}
    audit = dict(**provenance, bundle_id=SPEC[2], status='small_evidence_verified', experiment_complete=True,
                 endpoint_caches_recomputed=count, omitted=sorted(set(files)-set(selected)),
                 files={n: digest(b) for n, b in selected.items()},
                 scope='Every full-input manifest member verified; 60 cached losses/R1 recomputed; native '
                       'and fixed-LR replay records, matching/reset audits and difficulty arithmetic checked. '
                       'Caption-bearing inputs and NPZ caches omitted. Clone-only mode cannot recompute caches '
                       'or reexecute training invariants. No proof of remote Drive flush or independent replication.')
    for name, data in selected.items():
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_json(target/'export_audit.json', audit)
    write_json(target/'diagnostics.json', diagnostics(files))
    verify_export(target, SPEC)
    print(f'{count} cached endpoints recomputed; replay records and difficulty contrasts verified.')


if __name__ == '__main__':
    main()
