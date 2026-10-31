"""Build immutable D2 gap input from authenticated D0 starting-state shards only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from scripts.build_clip_source_population import derive, checksum

ROOT=Path(__file__).resolve().parents[1]


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def build(root, output):
    root=Path(root);output=Path(output)
    p=json.loads((ROOT/'configs/clip_placement_preflight.json').read_text())
    frozen=json.loads((ROOT/p['population']).read_text())
    # Existing D1 derivation authenticates all shards, supports, identities and masks.
    if derive(root)!=frozen: raise ValueError('Authenticated D0 does not reproduce exact D1 population')
    plan=json.loads((ROOT/p['plan']).read_text())
    records=[];by_seed={s:[] for s in p['training_seeds']}
    for batch in frozen['batches']:
        if batch['population']!='continuation':continue
        for w in p['checkpoint_steps']:
            state=batch['states'][str(w)]
            with gzip.open(root/state['shard'],'rt') as f:rows=[json.loads(line) for line in f]
            for position,i in enumerate(batch['eligible_queries']):
                row=rows[i];margins=dict(zip(row['candidate_indices'],row['margins']))
                c=state['supports']['C'][position][0];inc=state['supports']['I'][position][0]
                rc,ri=margins[c],margins[inc]
                if not rc>0 or not ri<=0:raise ValueError('Incorrect C/I bracket')
                half=(rc-ri)/2
                seed=batch['training_seed'];by_seed[seed].append(half)
                records.append(dict(training_seed=seed,weight_step=w,stream_seed=batch['stream_seed'],
                    batch_offset=batch['batch_offset'],query_index=i,**batch['identities'][i],
                    C_index=c,I_index=inc,r_C=rc,r_I=ri,half_bracket=half,
                    shard=state['shard'],shard_sha256=state['shard_sha256']))
    medians={str(s):statistics.median(by_seed[s]) for s in p['training_seeds']}
    value=dict(schema=1,design_sha256=checksum(ROOT/'configs/clip_placement_preflight.json'),
        generator_sha256=checksum(Path(__file__)),population_sha256=checksum(ROOT/p['population']),
        plan_sha256=checksum(ROOT/p['plan']),d0_provenance=frozen['provenance'],
        checkpoints={str(s):plan['seeds'][str(s)]['states'] for s in p['training_seeds']},
        source_fields=['candidate_indices','margins','sizes.2.regimes.C.support','sizes.2.regimes.I.support'],
        scoring='D0 detached renormalized FP32, no autocast/TF32; exact margin ties by batch index',
        repeated_exposures='preserved',calibration_included=False,
        median_convention='arithmetic mean of two central sorted values for even counts',
        counts={str(s):len(by_seed[s]) for s in p['training_seeds']},seed_medians=medians,
        delta=statistics.median(medians.values()),records=records)
    data=encode(value)
    if output.exists() and output.read_bytes()!=data:
        raise FileExistsError('Refusing to replace a different gap manifest; version the protocol')
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists():output.write_bytes(data)
    print(json.dumps(dict(output=str(output),sha256=hashlib.sha256(data).hexdigest(),
        delta=value['delta'],counts=value['counts'],seed_medians=medians)))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('d0_directory');parser.add_argument('--output',default=str(ROOT/'configs/clip_placement_gap.json'))
    args=parser.parse_args();build(args.d0_directory,args.output)
