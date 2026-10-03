"""CPU-only paired history interaction. Never trains, refits, or edits input runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from scripts.analyze_clip_policy_robustness import checked_json, close, require, verify_input
from scripts.analyze_clip_image_influence import query_losses, unpack

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT/'configs/clip_paired_history_comparison.json'


def load_spec():
    return checked_json(SPEC_PATH.read_bytes())


def read_run(path, condition, spec):
    path = Path(path)
    require(path.is_dir(), 'Supply an extracted complete run directory, not a ZIP')
    marker = path/'DRIVE_BACKUP_MANIFEST.json'
    manifest = checked_json(marker.read_bytes())
    expected = spec['conditions'][condition]
    run_id = manifest['run_id']
    require(run_id.startswith(expected['experiment'].removesuffix('_v1')+'_'), 'Wrong run family')
    if condition == 'native':
        require(run_id == expected['run_id'], 'Wrong pinned native run')
    digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    files, provenance = verify_input(path, dict(source_run=run_id,
        backup_manifest_sha256=expected.get('backup_manifest_sha256', digest)))
    read = lambda name: checked_json(files[name])
    run, done = read('run_manifest.json'), read('completion.json')
    require(run['run_id'] == run_id and run['experiment'] == expected['experiment']
            and run['bundle_id'] == expected['bundle_id']
            and run['status'] == 'complete_pending_drive_flush', 'Wrong bundle or incomplete run')
    require(done['status'] == 'complete' and [done[k] for k in
        ['pairs','branches','endpoint_caches','trunk_optimizer_steps','branch_optimizer_steps','optimizer_steps']]
        == [12,48,48,1500,2400,3900], 'Incomplete experiment budget/grid')
    require(read('protocol.json') == expected['protocol'], 'Protocol changed')
    require(read('source_manifest.json') == expected['source_files'], 'Source manifest changed')
    for name, sha in expected['source_files'].items():
        require(hashlib.sha256(files['source_overlay/'+name]).hexdigest() == sha, 'Source changed: '+name)
    rows = {}
    paths = set()
    for seed in spec['seeds']:
        audit = read(f'seed_{seed}/trunk_audit.json')
        validate_trunk(audit, condition, expected['protocol'])
        for step in [100,500]:
            for stream in spec['streams']:
                prefix = f'seed_{seed}/step_{step:06d}/stream_{stream}/'
                paths.add(prefix+'result.json')
                row = read(prefix+'result.json')
                require((row['training_seed'], row['checkpoint_step'], row['stream_seed']) ==
                        (seed,step,stream), 'Wrong pair identity')
                validate_pair(row, spec)
                for arm in spec['arms']:
                    meta = read(prefix+f'features_{arm}.json')
                    logits = unpack(files[prefix+f'features_{arm}.npz'], meta, row['report_row_order_sha256'])
                    report = row['points'][arm]['50']['report']
                    require(meta['feature_sha256'] == report['features_sha256'], 'Endpoint feature mismatch')
                    close(float(query_losses(logits)[:,2].mean()), report['native_loss'], 'cached full512 loss')
                    truth = np.arange(512)
                    t = 100*float(np.mean(logits.argmax(1) == truth))
                    i = 100*float(np.mean(logits.argmax(0) == truth))
                    for key, value in [('text_to_image_r1_percent',t),('image_to_text_r1_percent',i),
                                       ('mean_r1_percent',(t+i)/2)]:
                        close(value, report['pool_retrieval'][key], 'cached R1')
                rows[seed,step,stream] = row
    require({n for n in files if n.endswith('/result.json')} == paths, 'Extra or missing result grid')
    original = read('comparison.json')
    seed_means = {s:statistics.fmean(effect(rows[s,k,u], 'gated', 'random_matched')['loss']
        for k in [100,500] for u in spec['streams']) for s in spec['seeds']}
    close(statistics.fmean(seed_means.values()), original['primary_value'], 'original primary')
    require({x['training_seed'] for x in original['per_seed']} == set(spec['seeds']), 'Original seed grid')
    for x in original['per_seed']:
        close(seed_means[x['training_seed']], x['primary'], 'original seed mean')
    return dict(files=files, rows=rows, provenance=provenance, run=run)


def validate_trunk(audit, condition, p):
    traces = audit['traces']
    require(audit['actual_optimizer_steps'] == len(traces) == 500
        and audit['reporting_access'] is False, 'Invalid trunk audit')
    require([t['completed_updates'] for t in traces] == list(range(1,501)), 'Incomplete trunk order')
    decisions = audit['decisions']
    if condition == 'native':
        require(audit['history'] == 'native_only' and audit['auxiliary_updates'] == 0
            and audit['probe_calls'] == 0 and not decisions, 'Not native-only history')
    else:
        require([v['completed_updates'] for v in decisions] == [100,250], 'Wrong history gate refreshes')
        for v in decisions:
            require(v['active'] == (v['measurement']['mean_cosine'] > p['alignment_threshold']), 'History gate changed')
    active = False
    for offset,t in enumerate(traces):
        for v in decisions:
            if v['completed_updates'] == offset:
                active = v['active']
        require(t['coefficient'] == (p['coefficient'] if active else 0.), 'History coefficient mismatch')


def validate_pair(row, spec):
    a = row['audit']
    require(a['actual_optimizer_steps'] == 200 and all(a[k] is True for k in
        ['source_immutable','frozen_parameters_unchanged','observations_preserve_state',
         'exact_no_update_report_replay','matched_rng_and_lrs','exposure_matched'])
        and a['reports_control_treatment'] is False, 'Failed branch audit')
    require(set(a['branch_start_hashes']) == set(spec['arms'])
        and len(set(a['branch_start_hashes'].values())) == 1, 'Unmatched initial states')
    require(set(row['actions']) == set(row['traces']) == set(row['points']) == set(spec['arms']), 'Missing arms')
    require(row['actions']['native'] == [False]*50 and row['actions']['sustained'] == [True]*50, 'Wrong controls')
    require(sum(row['actions']['gated']) == sum(row['actions']['random_matched']), 'Exposure differs')
    for arm in spec['arms']:
        require(len(row['traces'][arm]) == len(row['actions'][arm]) == 50
            and all(type(x) is bool for x in row['actions'][arm]), 'Incomplete traces/actions')
        require(row['exposure'][arm] == sum(row['actions'][arm]), 'Exposure count wrong')
        require(set(row['points'][arm]) == {'0','50'}
                and row['points'][arm]['0'] == row['points']['native']['0'], 'Initial points differ')
        for j, trace in enumerate(row['traces'][arm]):
            require(trace['coefficient'] == (spec['conditions']['native']['protocol']['coefficient']
                if row['actions'][arm][j] else 0.), 'Action coefficient mismatch')
            for key in ['learning_rates','rng_after_sha256']:
                require(trace[key] == row['traces']['native'][j][key], 'Branch RNG/LR mismatch')


def effect(row, treated='sustained', control='native'):
    t,c = (row['points'][arm]['50']['report'] for arm in [treated,control])
    return dict(loss=t['native_loss']-c['native_loss'],
        r1_pp=t['pool_retrieval']['mean_r1_percent']-c['pool_retrieval']['mean_r1_percent'])


def verify_matching(native, gated, spec):
    read = lambda run,name: checked_json(run['files'][name])
    for key in ['packages','gpu','source_commit']:
        require(native['run'][key] == gated['run'][key], 'Environment mismatch: '+key)
    for name in ['split_plan.json','branch_plans.json','timing_permutations.json','training_exclusion_indices.json']:
        require(read(native,name) == read(gated,name), 'Unmatched '+name)
    for seed in spec['seeds']:
        prefix = f'seed_{seed}/'
        for name in ['data_roles.json','training_batches.json','trainable_policy.json']:
            require(read(native,prefix+name) == read(gated,prefix+name), 'Unmatched data/probes: '+name)
        configs = [read(run,prefix+'training_config.json') for run in [native,gated]]
        for cfg in configs:
            cfg['dataset'].pop('diagnostic_holdout_indices')  # Run-directory path only; indices checked above.
        require(configs[0] == configs[1], 'Training configuration mismatch')
        traces = [read(run,prefix+'trunk_audit.json')['traces'] for run in [native,gated]]
        for n,g in zip(*traces):
            for key in ['batch_identity_sha256','epoch','completed_updates','learning_rates','rng_after_sha256']:
                require(n[key] == g[key], 'Unmatched trunk stream/RNG/LR: '+key)
        for step in [100,500]:
            starts = [read(run,prefix+f'start_{step:06d}.json') for run in [native,gated]]
            require(starts[0]['epoch'] == starts[1]['epoch'], 'Starting epochs differ')
            if step == 100:
                require(starts[0]['state_sha256'] == starts[1]['state_sha256'], 'Early starting-state mismatch')
            for stream in spec['streams']:
                pair = prefix+f'step_{step:06d}/stream_{stream}/'
                require(read(native,pair+'training_batches.json') == read(gated,pair+'training_batches.json'),
                    'Unmatched continuation batches/captions')
                n,g = native['rows'][seed,step,stream],gated['rows'][seed,step,stream]
                require(n['report_row_order_sha256'] == g['report_row_order_sha256'], 'Report identities differ')
                for arm in spec['arms']:
                    for tn,tg in zip(n['traces'][arm],g['traces'][arm]):
                        require(all(tn[k] == tg[k] for k in ['learning_rates','rng_after_sha256']), 'Cross-history RNG/LR mismatch')
                if step == 100:
                    # Probe wall-clock times and snapshot metadata are intentionally excluded.
                    for key in ['points','actions','traces','exposure']:
                        require(n[key] == g[key], 'Early continuation mismatch: '+key)


def compare(native_rows, gated_rows, spec):
    expected = {(s,k,u) for s in spec['seeds'] for k in [100,500] for u in spec['streams']}
    require(set(native_rows) == set(gated_rows) == expected, 'Incomplete paired grid')
    pairs, seeds = [],[]
    for seed in spec['seeds']:
        by_step = {}
        for step in [100,500]:
            group = []
            for stream in spec['streams']:
                key = seed,step,stream
                n,g = effect(native_rows[key]),effect(gated_rows[key])
                item = dict(seed=seed,starting_step=step,stream=stream,native_history=n,gated_history=g,
                    interaction={m:g[m]-n[m] for m in n})
                group.append(item); pairs.append(item)
            by_step[str(step)] = {condition:{m:statistics.fmean(r[condition][m] for r in group)
                for m in ['loss','r1_pp']} for condition in ['native_history','gated_history','interaction']}
        seeds.append(dict(seed=seed,by_starting_step=by_step,
            early_later_interaction={m:by_step['500']['interaction'][m]-by_step['100']['interaction'][m]
                                    for m in ['loss','r1_pp']}))
    aggregates = {str(k):{c:{m:statistics.fmean(s['by_starting_step'][str(k)][c][m] for s in seeds)
        for m in ['loss','r1_pp']} for c in ['native_history','gated_history','interaction']} for k in [100,500]}
    return dict(experiment=spec['experiment'],primary=spec['primary'],
        primary_value=aggregates['500']['interaction']['loss'],by_starting_step=aggregates,
        per_seed=seeds,paired_sets=pairs,replication_units=3,analysis_optimizer_updates=0,
        interpretation='Positive interaction means gated history makes subsequent sustained treatment less helpful/more harmful. '
        'Show absolute sustained-minus-native effects too. Loss and R1 are separate. '
        'This compares a frozen gated-history policy with native history, not a fixed historical dose. '
        'It does not separate representation, optimizer, or learning-rate mechanisms. '
        'Chosen after native-only results, before gated-history results; exploratory, not independent confirmation.')


def main(native_path, gated_path=None, output=None):
    spec = load_spec()
    if output is not None:
        require(not Path(output).exists(), 'Refusing to overwrite output')
    native = read_run(native_path,'native',spec)
    if gated_path is None:
        print('NATIVE_INPUT_VERIFIED:',json.dumps(native['provenance']))
        print('Comparison pending: gated-history run is required. No comparison result written.')
        return
    gated = read_run(gated_path,'gated_history',spec)
    verify_matching(native,gated,spec)
    result = compare(native['rows'],gated['rows'],spec)
    result['provenance'] = dict(native=native['provenance'],gated_history=gated['provenance'],
        specification_sha256=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest())
    result['history_exposure_by_seed'] = {str(s):{c:sum(t['coefficient'] != 0 for t in
        checked_json(run['files'][f'seed_{s}/trunk_audit.json'])['traces'])
        for c,run in [('native',native),('gated_history',gated)]} for s in spec['seeds']}
    if output is not None:
        path = Path(output); path.mkdir(parents=True,exist_ok=False)
        (path/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (path/'protocol.json').write_bytes(SPEC_PATH.read_bytes())
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--native',required=True)
    parser.add_argument('--gated-history')
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.native,args.gated_history,args.output)
