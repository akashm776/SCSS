"""Authenticate saved v2c evidence and independently recompute pressure summaries."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

from scripts.audit_clip_placement_v2a import read_input, sha, canonical, require
from src.clip_placement_v2c import gradient_reference

ROOT=Path(__file__).resolve().parents[1]
ARMS=['H-Uniform','Wrong','Boundary','Correct']
SIGNS={'Wrong':-1,'Boundary':0,'Correct':1}


def audit(path):
    read=read_input(path)
    raw={n:read(n) for n in ['acceptance.json','run_config.json','index.json']}
    receipt,config,index=(json.loads(raw[n]) for n in raw)
    spec=json.loads((ROOT/'configs/clip_placement_v2c.json').read_text())
    manifest=json.loads((ROOT/'placement_v2c_source_manifest.json').read_text())
    require(receipt['status']=='PASS' and receipt['operational_error'] is None,'Not an uninterrupted completed PASS')
    require(config['specification']==spec and config['implementation']==manifest,'Issued specification/implementation differs')
    require(canonical(config)==receipt['contract_sha256'],'Contract hash differs')
    require(sha(raw['index.json'])==receipt['committed_check_index_sha256'],'Index hash differs')
    for n,h in manifest.items():require(sha((ROOT/n).read_bytes())==h,'Issued source changed: '+n)
    gradient_reference(ROOT,spec)
    p=config['original_design'];delta=config['delta'];rho=p['target_ratio']
    require(p==json.loads((ROOT/spec['original_design']).read_text()),'Original design differs')
    require(delta==json.loads((ROOT/p['gap_manifest']).read_text())['delta'],'Gap differs')
    require(set(index)=={f'calibration/{s}/{w}' for s in spec['training_seeds'] for w in spec['checkpoint_steps']},'Incomplete cell grid')
    require(receipt['completed_cells']==receipt['passing_cells']==receipt['expected_cells']==6,'Receipt coverage differs')
    require(receipt['calibration_batches']==receipt['verification_batches']==24,'Wrong batch counts')
    require(receipt['optimizer_updates']==0 and not receipt['d2_launched'] and not receipt['launch_eligible'],'Unexpected training permission')
    pop=json.loads((ROOT/p['population']).read_text())
    masks={(b['training_seed'],b['batch_offset']):b['eligible_queries'] for b in pop['batches'] if b['stream_seed'] is None}
    cells=[];exposures=0;empty=0;sim_errors=[];logit_errors=[];geom_errors=[];all_ratios=[]
    for key,entry in sorted(index.items()):
        raw[entry['file']]=read(entry['file'])
        require(sha(raw[entry['file']])==entry['sha256'],'Cell hash differs')
        envelope=json.loads(raw[entry['file']]);result=envelope['result'];e=result['evidence']
        seed,w=map(int,key.split('/')[1:])
        require(envelope['key']==key and envelope['contract_sha256']==canonical(config),'Wrong cell lineage')
        require(result['status']=='PASS' and result['context']==dict(training_seed=seed,weight_step=w),'Wrong cell identity/status')
        require(e['optimizer_updates']==0 and len(e['preservation'])==8,'Updates or missing restoration records')
        for state in e['preservation']:
            require(state['preserved'] and state['before_sha256']==state['after_sha256'],'State preservation failed')
        require(len(e['unweighted'])==len(e['verification'])==4,'Missing scheduled batches')
        for phase in ['unweighted','verification']:
            for offset,row in enumerate(e[phase]):
                ids=masks[seed,offset]
                require(row['batch_offset']==offset and row['eligible_queries']==ids and row['eligible']==len(ids),'Wrong eligibility')
                require(row['empty']==(not ids),'Wrong empty flag')
                if phase=='unweighted':exposures+=len(ids);empty+=not ids
                settings=row['fp32_settings']
                require(settings['matmul_precision']=='highest' and all(not settings[k] for k in ['autocast_enabled','matmul_tf32','cudnn_tf32','bf16_reduction','fp16_reduction']),'FP32 context differs')
                require(math.isfinite(row['native_norm']) and row['native_norm']>p['gradient_norm_min'],'Invalid native norm')
                require(set(row['auxiliary_norms'])==set(ARMS) and row['scale_disconnected']==dict.fromkeys(ARMS,True),'Missing arms or scale leakage')
                for a,n in row['auxiliary_norms'].items():
                    require(math.isfinite(n) and (n>p['gradient_norm_min'] if ids else n==0),'Invalid auxiliary norm')
                    cosine=row['native_auxiliary_cosines'][a]
                    require((cosine is None) if not ids else (math.isfinite(cosine) and abs(cosine)<=1+1e-12),'Invalid cosine')
                if not ids:continue
                require(row['placement_contract']['status']=='PASS','Placement not passed')
                for a,sign in SIGNS.items():
                    v=row['placements'][a];tau=v['tau'];target=sign*delta
                    require(math.isfinite(tau) and tau>0 and v['delta']==delta,'Invalid scale/gap')
                    require(all(len(v[k])==len(ids) for k in ['r_geom','r_train','g_logit']),'Incomplete exposure arrays')
                    for rg,rt,gl in zip(v['r_geom'],v['r_train'],v['g_logit']):
                        ge,se,le=abs(rg-target),abs(rt-target),abs(gl-tau*target)
                        require(ge<=max(p['fp32_margin_atol'],p['fp32_margin_rtol_delta']*delta),'Geometry margin failed')
                        require(se<=p['training_margin_rtol_delta']*delta and le<=p['training_logit_rtol_scaled_delta']*tau*delta,'Actual placement failed')
                        require(not sign or rt*sign>0 and gl*sign>0,'Wrong placement sign')
                        if phase=='unweighted':geom_errors.append(ge);sim_errors.append(se);logit_errors.append(le)
                for a,b in [('Wrong','Boundary'),('Boundary','Correct')]:
                    require(all(y-x>=p['minimum_adjacent_separation_delta']*delta for x,y in zip(row['placements'][a]['r_train'],row['placements'][b]['r_train'])),'Target separation failed')
        for u,v in zip(e['unweighted'],e['verification']):
            for field in ['native_norm','native_loss','eligible','placements','fp32_settings']:
                require(u[field]==v[field],'Weighted replay differs: '+field)
        coeff={}
        require(set(e['coefficients'])==set(ARMS),'Missing coefficients')
        for a in ARMS:
            r=[b['auxiliary_norms'][a]/b['native_norm'] for b in e['unweighted']]
            v=[b['auxiliary_norms'][a]/b['native_norm'] for b in e['verification']]
            c=statistics.fmean(r);mean=statistics.fmean(v);error=abs(mean/rho-1)
            require(c>p['gradient_norm_min'] and error<=p['calibration_relative_tolerance'],'Calibration pressure fails')
            fitted=e['coefficients'][a]
            expected=dict(coefficient=rho/c,unweighted_scheduled_ratio=c,batch_ratios=r,
                unweighted_treated_ratio=statistics.fmean([x for x,b in zip(r,e['unweighted']) if b['eligible']]),
                verified_batch_ratios=v,verified_scheduled_ratio=mean,relative_error=error,
                verified_treated_ratio=statistics.fmean([x for x,b in zip(v,e['verification']) if b['eligible']]))
            require(fitted==expected,'Coefficient or verification summary does not match saved norms')
            coeff[a]=expected;all_ratios.extend(v)
        cells.append(dict(seed=seed,weight_step=w,coefficients=coeff))
    means=[v['verified_scheduled_ratio'] for cell in cells for v in cell['coefficients'].values()]
    return dict(status='PASS',contract_sha256=canonical(config),input_sha256={n:sha(b) for n,b in raw.items()},
        cells=cells,coefficients=24,calibration_batches=24,verification_batches=24,eligible_exposures=exposures,
        placement_checks=3*exposures,empty_batches=empty,state_preserved=True,optimizer_updates=0,
        scheduled_pressure_range=[min(means),max(means)],individual_batch_pressure_range=[min(all_ratios),max(all_ratios)],
        max_relative_pressure_error=max(abs(x/rho-1) for x in means),
        max_geometry_error=max(geom_errors),max_similarity_error=max(sim_errors),max_logit_error=max(logit_errors),
        limitation='Authentication and recomputation from saved metrics; no independent GPU-gradient recomputation. Calibration batches only; no training utility or full-population acceptance.')


def report(v):
    lines=['# Authenticated v2c calibration audit','',
        'PASS: all six seed/state cells, 24 coefficients, 24 scheduled batches and 24 weighted-gradient replays.',
        f"Eligibility: {v['eligible_exposures']} state/exposure instances, {v['placement_checks']} arm/exposure checks; {v['empty_batches']} empty batches.",
        '',f"Verified scheduled pressure range: {100*v['scheduled_pressure_range'][0]:.6f}%–{100*v['scheduled_pressure_range'][1]:.6f}%.",
        f"Maximum relative error against the 2% target: {100*v['max_relative_pressure_error']:.6f}% (allowed 5%).",
        f"Individual batch pressures: {100*v['individual_batch_pressure_range'][0]:.4f}%–{100*v['individual_batch_pressure_range'][1]:.4f}%.",
        '', '| Seed | State | H-Uniform alpha | Wrong alpha | Boundary alpha | Correct alpha |','|---|---:|---:|---:|---:|---:|']
    for cell in v['cells']:
        lines.append(f"| {cell['seed']} | {cell['weight_step']} | "+' | '.join(f"{cell['coefficients'][a]['coefficient']:.8f}" for a in ARMS)+' |')
    lines+=['',f"Maximum errors: ideal margin {v['max_geometry_error']:.9g}; actual similarity margin {v['max_similarity_error']:.9g}; actual logit gap {v['max_logit_error']:.9g}.",
        'Weighted replay preserved native norms/losses and placement observables exactly. All 48 state-restoration records match. Auxiliary scale disconnection passed.',
        '', 'The 2% target is a four-scheduled-batch average, not a per-batch equality or an optimizer-displacement match. All batches were treated, so conditional and scheduled means coincide.',
        'Larger placement-arm coefficients compensate for smaller raw gradients; they do not establish greater training utility. Keep these coefficients frozen for the next stage.',
        '',v['limitation'],'','Next: full-population D2-P v2 audit. Zero optimizer updates; D2 training remains gated.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('input');parser.add_argument('--output',required=True)
    args=parser.parse_args();value=audit(args.input)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if Path(args.input).is_file():
        value['source_zip_sha256']=sha(Path(args.input).read_bytes());shutil.copyfile(args.input,output/'source.zip')
    (output/'audit.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (output/'REPORT.md').write_text(report(value));print(report(value))
