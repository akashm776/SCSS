"""Derive D1's outcome-blind, k=2 common population from authenticated D0 shards."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'experiment_results/clip_geometry_audit_2026-10'


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(root):
    root = Path(root)
    for name in ['shards.json', 'completion.json', 'plan.json', 'source_manifest.json']:
        if checksum(root/name) != checksum(ARCHIVE/name):
            raise ValueError('Different D0 evidence: '+name)
    shards = json.loads((root/'shards.json').read_text())
    pairs = {}
    for shard in shards:
        path = root/shard['records']
        if checksum(path) != shard['sha256']:
            raise ValueError('D0 shard checksum mismatch')
        with gzip.open(path, 'rt') as f:
            rows = [json.loads(line) for line in f]
        key = f"{shard['training_seed']}:{shard['stream_seed']}:{shard['batch_offset']}"
        w = str(shard['weight_step'])
        pair = pairs.setdefault(key, dict(training_seed=shard['training_seed'],
            stream_seed=shard['stream_seed'], batch_offset=shard['batch_offset'],
            population=shard['population'], states={}))
        if w in pair['states'] or len(rows) != 64:
            raise ValueError('Duplicate or incomplete D0 batch')
        state = dict(shard=shard['records'], shard_sha256=shard['sha256'],
                     identities=[], eligible=[], supports={r:[] for r in 'HMCI'})
        for i,row in enumerate(rows):
            if row['query_index'] != i:
                raise ValueError('D0 query order changed')
            for name in ['training_seed','stream_seed','batch_offset','population','weight_step']:
                if row[name] != shard[name]:
                    raise ValueError('D0 context mismatch')
            state['identities'].append({k:row[k] for k in ['source_index','caption_index','species_id']})
            margins = dict(zip(row['candidate_indices'], row['margins']))
            hard = sorted(margins, key=lambda j:(margins[j], j))
            c = [j for j in hard if margins[j] > 0]
            inc = sorted((j for j in hard if margins[j] <= 0), key=lambda j:(abs(margins[j]), j))
            eligible = len(c) >= 2 and len(inc) >= 2
            if eligible != row['sizes']['2']['eligible']:
                raise ValueError('D0 eligibility mismatch')
            state['eligible'].append(eligible)
            for regime, support in zip('HMCI', [hard[:2],hard[15:17],c[:2],inc[:2]]):
                old = row['sizes']['2']['regimes'][regime]
                expected = support if len(support)==2 else None
                if old['support'] != expected:
                    raise ValueError('D0 support order mismatch')
                if eligible:
                    if not all(old['candidate'][path]['valid'] for path in ['ideal','mixed']):
                        raise ValueError('Invalid D0 mixture on eligible population')
                    if old['candidate']['real']['index'] != support[0]:
                        raise ValueError('D0 Select identity mismatch')
                state['supports'][regime].append(expected)
        pair['states'][w] = state
    if len(pairs) != 312:
        raise ValueError('Incomplete paired grid')
    for pair in pairs.values():
        a,b = (pair['states'][w] for w in ['100','500'])
        if a['identities'] != b['identities']:
            raise ValueError('Cross-state exposure identity mismatch')
        pair['identities'] = a['identities']
        pair['eligible_queries'] = [i for i in range(64) if a['eligible'][i] and b['eligible'][i]]
        if not pair['eligible_queries']:
            raise ValueError('Empty common batch: protocol amendment required')
        for state in pair['states'].values():
            del state['identities']
            state['supports'] = {r:[support[i] for i in pair['eligible_queries']]
                                 for r,support in state['supports'].items()}
    batches = sorted(pairs.values(),key=lambda b:(b['training_seed'], b['stream_seed'] or 0,b['batch_offset']))
    counts = {pop:sum(len(b['eligible_queries']) for b in batches if b['population']==pop)
              for pop in ['continuation','calibration_feasibility']}
    return dict(experiment='clip_source_regime_v1', d0_run=root.name, k=2,
        provenance={n:checksum(ARCHIVE/n) for n in ['shards.json','completion.json','plan.json','source_manifest.json']},
        selection='D0 FP32 ordered supports, common E2_100 intersect E2_500; no outcomes',
        eligible_exposures=counts, minimum_eligible_per_batch=min(len(b['eligible_queries']) for b in batches),
        batches=batches)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__); parser.add_argument('d0_directory')
    args=parser.parse_args()
    value=derive(args.d0_directory)
    target=ROOT/'configs/clip_source_population.json'
    target.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    print(target, value['eligible_exposures'], 'minimum per batch',value['minimum_eligible_per_batch'])
