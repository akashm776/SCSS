"""CPU-only, fixed endpoint image-influence sensitivity; no fitting or training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics

import numpy as np

from scripts.analyze_clip_policy_robustness import checked_json, verify_input

SPEC = dict(experiment='clip_image_influence_v1',
    source_run='clip_policy_pool_test_20260930T120952_487061Z',
    source_bundle='52ad1d1e45ae8cd8d39f623d028c890ed4ef73e57ad4f4301c1a5b929092bead',
    backup_manifest_sha256='880459ebeef76ab14521e76a2808f48f66faa20d4392aff6348d9c39de14c9b2',
    seeds=[789, 2026, 31415], checkpoints=[100, 500], streams=[2026093002, 2026093003],
    arms=['gated', 'native', 'sustained', 'random_matched'], rows=512,
    group_seed=2026093004, group_size=16, groups=32, concentration_counts=[1, 5, 10, 25],
    primary='gated_minus_random_matched_query_concentration_and_fixed_group_sensitivity',
    optimizer_updates=0, independent_test_set=False, rules_refitted=False,
    analysis_selected_after_aggregate_results=True, tolerance=1e-12)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def groups_for(ids, seed, size):
    require(len(ids) == len(set(ids)) and len(ids) % size == 0, 'Invalid row identity/groups')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'image-influence-v1:{seed}:{ids[i]}'.encode()).hexdigest())
    return [order[i:i+size] for i in range(0, len(order), size)]


def query_losses(logits):
    """Row i is text i; column i is image i. Each contributes one query per direction."""
    x = np.asarray(logits, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == x.shape[1] and len(x) >= 2 and np.isfinite(x).all(),
            'Invalid paired logits')
    def logsumexp(axis):
        maximum = x.max(axis=axis, keepdims=True)
        return np.log(np.exp(x-maximum).sum(axis=axis))+maximum.squeeze(axis)
    t2i = logsumexp(1)-np.diag(x)
    i2t = logsumexp(0)-np.diag(x)
    return np.stack([t2i, i2t, (t2i+i2t)/2], axis=1)


def deletion_losses(logits, groups):
    count = len(logits)
    require(sorted(i for g in groups for i in g) == list(range(count)), 'Groups must partition rows')
    values = []
    for group in groups:
        removed = set(group)
        keep = np.array([i for i in range(count) if i not in removed])
        require(len(keep) >= 2, 'Too few retained pairs')
        values.append(query_losses(logits[np.ix_(keep, keep)]).mean(axis=0))
    return np.asarray(values)


def unpack(data, metadata, row_hash):
    require(hashlib.sha256(data).hexdigest() == metadata['npz_sha256'], 'NPZ checksum differs')
    require(metadata['row_order_sha256'] == row_hash and metadata['rows'] == SPEC['rows']
            and metadata['dimension'] == 512 and metadata['dtype'] == 'float32', 'Wrong cache identity')
    with np.load(io.BytesIO(data), allow_pickle=False) as values:
        require(set(values.files) == {'images', 'texts', 'scale'}, 'Unexpected NPZ fields')
        images, texts, scale = values['images'], values['texts'], values['scale']
    require(images.shape == texts.shape == (512, 512) and images.dtype == texts.dtype == np.float32
            and scale.shape == () and np.isfinite(images).all() and np.isfinite(texts).all()
            and np.isfinite(scale) and scale > 0, 'Invalid feature arrays')
    scale = float(scale)
    digest = hashlib.sha256(images.tobytes()+texts.tobytes()+json.dumps(scale).encode()).hexdigest()
    require(digest == metadata['feature_sha256'], 'Feature hash differs')
    return texts.astype(np.float64) @ images.astype(np.float64).T * scale


def hierarchical(rows, arrays, seeds=None, steps=None, streams=None):
    seeds = SPEC['seeds'] if seeds is None else seeds
    steps = SPEC['checkpoints'] if steps is None else steps
    streams = SPEC['streams'] if streams is None else streams
    grid = {(r['training_seed'], r['checkpoint_step'], r['stream_seed']): a for r, a in zip(rows, arrays)}
    require(len(grid) == len(rows) == len(arrays), 'Duplicate/missing paired arrays')
    return np.mean([np.mean([np.mean([grid[s, k, u] for u in streams], axis=0)
                            for k in steps], axis=0) for s in seeds], axis=0)


def distribution(values):
    values = np.asarray(values)
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()),
                favorable=int((values < 0).sum()), tied=int((values == 0).sum()), unfavorable=int((values > 0).sum()))


def concentration(values, counts):
    """Outcome-ranked attribution only, not a new cleaned-data performance claim."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    require(values.ndim == 1 and all(0 < k < n for k in counts), 'Invalid concentration input')
    beneficial = np.maximum(-values, 0)
    gross = float(beneficial.sum())
    order = sorted(range(n), key=lambda i: (float(values[i]), i))
    top = []
    for k in counts:
        chosen = order[:k]
        remain = [i for i in range(n) if i not in set(chosen)]
        top.append(dict(k=k, rows=chosen,
            fraction_of_gross_benefit=float(beneficial[chosen].sum()/gross) if gross else None,
            query_only_mean_after_removing_most_favorable=float(values[remain].mean())))
    return dict(mean=float(values.mean()), **distribution(values), gross_benefit_sum=gross,
        gross_harm_sum=float(np.maximum(values, 0).sum()),
        leave_one_query_out=distribution((values.sum()-values)/(n-1)), top=top,
        warning='Top-k uses observed outcomes and retains all original negatives. Attribution stress test only; '
                'do not discard those images, report a cleaned score, or treat deletions as independent experiments.')


def analyze(files, groups):
    read = lambda n: checked_json(files[n])
    manifest, done, original = read('run_manifest.json'), read('completion.json'), read('comparison.json')
    require(manifest['bundle_id'] == SPEC['source_bundle'] and done['status'] == 'complete'
            and (done['pairs'], done['branches'], done['optimizer_steps'], done['endpoint_caches']) == (12,48,2400,48)
            and done['exact_policy_replay'] and done['original_features_exact'] and done['original_reports_exact']
            and not done['gate_decisions_recomputed'], 'Incomplete or wrong source experiment')
    row_info = read('report_row_order.json')
    ids = row_info['image_key_sha256']
    row_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    require(len(ids) == SPEC['rows'] and row_hash == row_info['row_order_sha256'], 'Row identity hash differs')
    require(groups == groups_for(ids, SPEC['group_seed'], SPEC['group_size']), 'Fixed group plan differs')
    rows, paired, deleted = [], {a: [] for a in SPEC['arms'][1:]}, {a: [] for a in SPEC['arms'][1:]}
    pair_outputs = []
    max_error = 0.
    for seed in SPEC['seeds']:
        for step in SPEC['checkpoints']:
            for stream in SPEC['streams']:
                folder = f'seed_{seed}/step_{step:06d}/stream_{stream}'
                row = read(folder+'/result.json')
                require((row['training_seed'],row['checkpoint_step'],row['stream_seed']) == (seed,step,stream), 'Pair identity differs')
                queries, removals = {}, {}
                for arm in SPEC['arms']:
                    meta = read(folder+f'/features_{arm}.json')
                    endpoint = row['endpoints'][arm]
                    require(meta == row['caches'][arm] and meta['feature_sha256'] == endpoint['features_sha256'], 'Cache metadata differs')
                    logits = unpack(files[folder+f'/features_{arm}.npz'], meta, row_hash)
                    q = query_losses(logits)
                    error = abs(float(q[:,2].mean())-endpoint['full_pool_loss'])
                    max_error = max(max_error, error)
                    require(error <= SPEC['tolerance'], 'Original full-pool loss failed reproduction')
                    truth = np.arange(len(ids))
                    r1 = endpoint['pool_retrieval']
                    require(100*float(np.mean(logits.argmax(1)==truth)) == r1['text_to_image_r1_percent']
                            and 100*float(np.mean(logits.argmax(0)==truth)) == r1['image_to_text_r1_percent'], 'Retrieval failed reproduction')
                    queries[arm], removals[arm] = q, deletion_losses(logits, groups)
                out = dict(training_seed=seed, checkpoint_step=step, stream_seed=stream, contrasts={})
                for control in paired:
                    delta, deletion = queries['gated']-queries[control], removals['gated']-removals[control]
                    paired[control].append(delta); deleted[control].append(deletion)
                    out['contrasts'][control] = dict(per_query=delta.tolist(), paired_pool_group_deletion=deletion.tolist())
                rows.append(row); pair_outputs.append(out)
                print(f'INFLUENCE: verified and evaluated {seed}:{step}:{stream}', flush=True)
    comparisons = {}
    for control, arrays in paired.items():
        aggregate = hierarchical(rows, arrays)
        pool_deletion = hierarchical(rows, deleted[control])
        query_deletion = np.asarray([np.delete(aggregate,g,axis=0).mean(axis=0) for g in groups])
        require(abs(float(aggregate[:,2].mean())-original['comparisons'][control]['full_pool_loss']) <= SPEC['tolerance'], 'Original contrast differs')
        comparisons[control] = dict(
            directions={name: concentration(aggregate[:,i],SPEC['concentration_counts'])
                        for i,name in enumerate(['text_to_image','image_to_text','symmetric'])},
            query_only_group_deletion=query_deletion.tolist(), paired_pool_group_deletion=pool_deletion.tolist(),
            query_only_group_summary=distribution(query_deletion[:,2]),
            paired_pool_group_summary=distribution(pool_deletion[:,2]),
            per_seed=[dict(seed=s, full_pool_loss=float(hierarchical(rows,arrays,seeds=[s])[:,2].mean()),
                paired_pool_group_summary=distribution(hierarchical(rows,deleted[control],seeds=[s])[:,2])) for s in SPEC['seeds']],
            per_starting_step=[dict(step=k, full_pool_loss=float(hierarchical(rows,arrays,steps=[k])[:,2].mean()),
                paired_pool_group_summary=distribution(hierarchical(rows,deleted[control],steps=[k])[:,2])) for k in SPEC['checkpoints']],
            per_stream=[dict(stream=u, full_pool_loss=float(hierarchical(rows,arrays,streams=[u])[:,2].mean()),
                paired_pool_group_summary=distribution(hierarchical(rows,deleted[control],streams=[u])[:,2])) for u in SPEC['streams']],
            per_query=aggregate.tolist())
    return dict(comparisons=comparisons, paired_results=pair_outputs, verified_endpoint_caches=48,
        max_original_loss_absolute_error=max_error, optimizer_updates=0,
        direction_order=['text_to_image','image_to_text','symmetric'],
        interpretation='Gated minus control. Negative helps. Same three seeds and 512 report pairs. '
            'Query-only deletion keeps 512 candidates; paired-pool deletion removes 16 matched pairs from '
            'both axes and re-evaluates 496-pair loss. These answer different sensitivity questions. '
            'Groups/images are not independent training replications, confidence intervals or significance tests.')


def report(result):
    c = result['comparisons']['random_matched']
    s = c['directions']['symmetric']
    return ('# Image-influence diagnostic\n\n'
        'Question: Is the fixed-endpoint timing advantage concentrated in a few report pairs?\n\n'
        'Hypothesis: A small number of influential queries may account for the net loss advantage.\n\n'
        'Experiment: Reuse all 48 verified caches; fixed 32 groups of 16 pairs; no updates or refitting.\n\n'
        f"Result: gated-minus-random loss {s['mean']:.12g}; favorable/harmful queries "
        f"{s['favorable']}/{s['unfavorable']}. Paired-pool group deletion: "
        f"{c['paired_pool_group_summary']['favorable']}/32 favorable; range "
        f"[{c['paired_pool_group_summary']['min']:.12g}, {c['paired_pool_group_summary']['max']:.12g}].\n\n"
        'Top-k attribution (outcome-ranked, query-only; original candidates retained):\n\n'+
        '\n'.join(f"- k={v['k']}: gross-benefit share={v['fraction_of_gross_benefit']}; "
                   f"remaining mean={v['query_only_mean_after_removing_most_favorable']:.12g}" for v in s['top'])+
        '\n\nStructural update: Read loss concentration and negative-pool sensitivity separately. '
        'Even a stable deletion sign establishes robustness only on these reused endpoints and images.\n\n'
        'Next question: Does any timing advantage replicate on genuinely new seeds and untouched reporting data?\n\n'+result['interpretation']+'\n')


def main(source, output):
    output = Path(output)
    require(not output.exists(), 'Output exists; refusing to overwrite or resume')
    files, provenance = verify_input(Path(source), SPEC)
    ids = checked_json(files['report_row_order.json'])['image_key_sha256']
    groups = groups_for(ids,SPEC['group_seed'],SPEC['group_size'])
    output.mkdir(parents=True)
    write(output/'protocol.json',SPEC)
    write(output/'groups.json',dict(groups=groups,image_key_sha256=ids))
    write(output/'source_provenance.json',provenance)
    try:
        result = analyze(files,groups)
        write(output/'analysis.json',result)
        (output/'REPORT.md').write_text(report(result))
        inventory = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        write(output/'completion.json',dict(status='complete',optimizer_updates=0,files=inventory,
            analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    except BaseException as error:
        write(output/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}'))
        raise
    print(report(result))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.source,args.output)
