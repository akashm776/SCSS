"""Read-only authentication and summary of a v2b gradient observation."""
import argparse
import json
import math
from pathlib import Path
import shutil

from scripts.audit_clip_placement_v2a import read_input, sha, canonical, require

ROOT=Path(__file__).resolve().parents[1]
KEY='gradients/271828/2026100101/100/0'


def audit(path):
    read=read_input(path)
    raw={n:read(n) for n in ['acceptance.json','run_config.json','index.json']}
    receipt,config,index=(json.loads(raw[n]) for n in ['acceptance.json','run_config.json','index.json'])
    require(receipt['status']=='PASS','Receipt is not PASS')
    require(canonical(config)==receipt['contract_sha256'],'Run contract checksum mismatch')
    require(sha(raw['index.json'])==receipt['committed_check_index_sha256'],'Index checksum mismatch')
    require(set(index)=={KEY},'Unexpected check coverage')
    entry=index[KEY];raw[entry['file']]=read(entry['file'])
    require(sha(raw[entry['file']])==entry['sha256'],'Check checksum mismatch')
    record=json.loads(raw[entry['file']]);result=record['result'];evidence=result['evidence']
    require(record['key']==KEY and record['contract_sha256']==canonical(config),'Wrong check lineage')
    spec=json.loads((ROOT/'configs/clip_placement_v2b.json').read_text())
    require(config['specification']==spec,'Wrong frozen specification')
    manifest=json.loads((ROOT/'placement_v2b_source_manifest.json').read_text())
    require(config['implementation']==manifest,'Wrong issued implementation')
    for n,h in manifest.items():require(sha((ROOT/n).read_bytes())==h,'Issued source changed: '+n)
    for container in [receipt,result]:
        require(container['optimizer_updates']==0 and container['calibration_batches']==0,'Unexpected training/calibration')
        require(container['gradient_probes']==1,'Unexpected gradient observation count')
    require(not receipt['launch_eligible'] and not receipt['d2_launched'],'Unexpected training permission')
    require(result['status']=='PASS' and result['encoder_forwards']==1,'Observation did not pass')
    preservation=result['preservation']
    require(preservation['preserved'] and preservation['before_sha256']==preservation['after_sha256'],'State restoration mismatch')
    for gate in ['v1_reproduction','forward_reproduction']:
        r=evidence[gate]
        require(r['status']=='PASS' and r['differences']==[] and r['atol']==r['rtol']==0.,'Forward reproduction mismatch')
    parent_index=json.loads((ROOT/spec['forward_reference']['root']/'index.json').read_text())
    parent_entry=next(iter(parent_index.values()))
    parent=json.loads((ROOT/spec['forward_reference']['root']/parent_entry['file']).read_text())['result']
    require(result['context']==parent['context'],'Wrong exposure identity')
    gradients=evidence['gradients'];rows=gradients['per_query'];ids=result['context']['eligible_queries']
    require(set(rows)==set(map(str,ids)) and len(rows)==34,'Incomplete per-query coverage')
    policy=spec['gradient_contract'];summary={}
    for kind in ['source','carrier','query']:
        metrics=[]
        for q in ids:
            require(set(rows[str(q)])=={'Wrong','Boundary','Correct'},'Missing arm')
            for arm in ['Wrong','Boundary','Correct']:
                metric=rows[str(q)][arm][kind]
                require(all(math.isfinite(v) for v in metric.values()),'Nonfinite VJP metrics')
                require(metric['relative_vector_error']<=policy['vector_rtol'] and metric['cosine']>=policy['cosine_min'],
                        'Recorded VJP violates tolerance')
                require(metric['reference_norm']>policy['norm_min'] and metric['norm']>0,'Zero VJP')
                metrics.append(metric)
        summary[kind]=dict(count=len(metrics),max_relative_vector_error=max(x['relative_vector_error'] for x in metrics),
            min_cosine=min(x['cosine'] for x in metrics),min_norm=min(x['norm'] for x in metrics),max_norm=max(x['norm'] for x in metrics))
    scale=gradients['scale']
    for name in ['tau','logit_scale']:
        v=scale[name]
        error=abs(v['observed']-v['expected'])
        limit=policy['scale_atol']+policy['scale_rtol']*abs(v['expected'])
        require(math.isfinite(error) and error==v['error'] and limit==v['tolerance'] and error<=limit,
                'Native scale derivative check inconsistent')
    for arm in ['Wrong','Boundary','Correct']:
        require(scale[arm]=={k:[True,True] for k in ['candidate','partition','loss']},'Auxiliary scale is connected')
    full=gradients['full_gradients'];native=full['Native']['norm'];uniform=full['H-Uniform']['norm']
    require(math.isfinite(native) and native>0 and math.isfinite(uniform) and uniform>0,'Invalid reference norms')
    pressures={}
    for arm in ['H-Uniform','Wrong','Boundary','Correct']:
        v=full[arm]
        n=v['norm'] if arm=='H-Uniform' else v['vs_native']['norm']
        require(math.isfinite(n) and n>=0,'Invalid auxiliary norm')
        if arm!='H-Uniform':
            require(v['vs_native']['reference_norm']==native and v['vs_uniform']['reference_norm']==uniform
                    and v['vs_uniform']['norm']==n,'Gradient references differ')
        pressures[arm]=dict(norm=n,unweighted_ratio_to_native=n/native,
            cosine_to_native=None if arm=='H-Uniform' else v['vs_native']['cosine'],
            cosine_to_uniform=None if arm=='H-Uniform' else v['vs_uniform']['cosine'],nonzero_tensors=v['nonzero_tensors'])
    return dict(status='PASS',contract_sha256=canonical(config),input_sha256={n:sha(b) for n,b in raw.items()},
        context=result['context'],vjp_summary=summary,scale=scale,native_gradient_norm=native,
        native_nonzero_tensors=full['Native']['nonzero_tensors'],auxiliary_gradients=pressures,
        forward_reproduction='exact',state_preserved=True,optimizer_updates=0,calibration_batches=0,
        limitation='Audit of saved gradient metrics, not independent gradient recomputation. One uncalibrated starting batch only; no utility or full-population claim.')


def report(value):
    lines=['# Authenticated v2b gradient audit','',
        'PASS on 34 exposures × 3 placement arms. All 306 recorded source/carrier/query VJP checks satisfy their frozen tolerances.',
        'Both forward-reproduction gates were exact; complete state restoration was verified.',
        '', '| Arm | Raw gradient norm | Raw ratio to native | Cosine to native | Cosine to H-Uniform |',
        '|---|---:|---:|---:|---:|']
    for arm,row in value['auxiliary_gradients'].items():
        c=lambda x:'not recorded' if x is None else f'{x:.6f}'
        lines.append(f"| {arm} | {row['norm']:.6f} | {100*row['unweighted_ratio_to_native']:.3f}% | {c(row['cosine_to_native'])} | {c(row['cosine_to_uniform'])} |")
    lines += ['',f"Native gradient norm: {value['native_gradient_norm']:.9f}.",
        '', 'Source/carrier/query maximum recorded relative vector errors are all '+
        ', '.join(str(value['vjp_summary'][k]['max_relative_vector_error']) for k in ['source','carrier','query'])+'.',
        'Native scale derivatives passed; auxiliary candidate, partition and loss were disconnected from both scale tensors.',
        '', 'These are unweighted gradients on the first continuation batch, not calibrated 2% pressure.',
        'The smaller placed-candidate norms reinforce the need for separate frozen calibration; they do not rank training utility.',
        'Cosines against Native or H-Uniform are not pairwise cosines between the three placement arms.',
        '',value['limitation'],'',
        'Next stage: four-batch scheduled pressure calibration under the FP32 head, followed by the full-population audit. No training is authorized by this result.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('input');parser.add_argument('--output',required=True)
    args=parser.parse_args();value=audit(args.input)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if Path(args.input).is_file():
        value['source_zip_sha256']=sha(Path(args.input).read_bytes())
        shutil.copyfile(args.input,output/'source.zip')
    (output/'audit.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (output/'REPORT.md').write_text(report(value));print(report(value))
