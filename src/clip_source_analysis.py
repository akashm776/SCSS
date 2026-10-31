"""Prespecified D1 matched contrasts; streams nested in equally weighted seeds."""
import math
import statistics
from src.clip_source_regime import ARMS

METRICS=['loss','r1_pp']


def effects(cell,t):
    reports={a:cell['points'][a][str(t)]['report'] for a in ARMS}
    result={a:dict(loss=r['native_loss']-reports['native']['native_loss'],
        r1_pp=r['pool_retrieval']['mean_r1_percent']-reports['native']['pool_retrieval']['mean_r1_percent']) for a,r in reports.items()}
    if any(not math.isfinite(v) for row in result.values() for v in row.values()):raise ValueError('Nonfinite effect')
    return result


def contrasts(cells,steps):
    early,late=[f'w{w}' for w in steps]
    Q={w:{m:cells[w]['C_Select'][m]-cells[w]['I_Select'][m] for m in METRICS} for w in cells}
    G={r:{w:{m:cells[w][r+'_Synth'][m]-cells[w][r+'_Select'][m] for m in METRICS} for w in cells} for r in 'HMCI'}
    return dict(Q=Q,Delta_side={m:Q[late][m]-Q[early][m] for m in METRICS},
        directional_loss_crossover=Q[early]['loss']>0 and Q[late]['loss']<0,
        G=G,synthesis_state_interactions={r:{m:G[r][late][m]-G[r][early][m] for m in METRICS} for r in 'HMCI'},
        H_minus_others={r:{kind:{w:{m:cells[w]['H_'+kind][m]-cells[w][r+'_'+kind][m] for m in METRICS}
             for w in cells} for kind in ['Select','Synth']} for r in 'MCI'})


def summarize(rows,p):
    grid={(r['training_seed'],r['stream_seed']):r for r in rows}
    if len(grid)!=len(rows) or set(grid)!={(s,u) for s in p['training_seeds'] for u in p['stream_seeds']}:
        raise ValueError('Incomplete/duplicate D1 grid; no survivor averaging')
    keys=[f'w{w}' for w in p['checkpoint_steps']]
    for row in rows:
        if row['audit']['reports_control_treatment'] is not False:raise ValueError('Outcome-controlled treatment')
        if set(row['cells'])!=set(keys):raise ValueError('Incomplete weights')
        for flag in ['source_immutable','frozen_parameters_unchanged','common_population_all_arms_weights',
                     'coefficients_fixed','matched_rng_and_lr_all_branches','fresh_adamw_all_branches','observations_preserve_state']:
            if row['audit'][flag] is not True:raise ValueError('Failed D1 audit')
        if row['audit']['actual_optimizer_steps']!=len(keys)*len(ARMS)*p['horizon']:raise ValueError('Wrong step count')
        for w,cell in row['cells'].items():
            if set(cell['points'])!=set(ARMS) or cell['lr_step']!=p['fixed_lr_checkpoint_step']:raise ValueError('Wrong arm/LR grid')
            for arm in ARMS:
                if (set(cell['points'][arm])!={str(t) for t in p['observation_steps']}
                    or len(cell['traces'][arm])!=p['horizon']
                    or set(cell['pressure_probes'][arm])!={str(t) for t in p['pressure_probe_offsets']}
                    or any(v['coefficient']!=cell['coefficients'][arm] for v in cell['traces'][arm])):
                    raise ValueError('Incomplete branch/changed coefficient')
            for t in range(p['horizon']):
                hashes={row['cells'][k]['traces'][a][t]['geometry']['frozen_exposure_sha256'] for k in keys for a in ARMS}
                if len(hashes)!=1:raise ValueError('Unmatched frozen population')
    for seed in p['training_seeds']:
        sr=[grid[seed,u] for u in p['stream_seeds']]
        if len({r['calibration_sha256'] for r in sr})!=1:raise ValueError('Calibration differs across streams')
        for w in keys:
            if any(r['cells'][w]['coefficients']!=sr[0]['cells'][w]['coefficients'] for r in sr):
                raise ValueError('Coefficients differ across streams')
    per_seed=[]
    for seed in p['training_seeds']:
        times={}
        for t in p['observation_steps']:
            individual=[{w:effects(grid[seed,u]['cells'][w],t) for w in keys} for u in p['stream_seeds']]
            avg={w:{a:{m:statistics.fmean(x[w][a][m] for x in individual) for m in METRICS} for a in ARMS} for w in keys}
            times[str(t)]=dict(effects=avg,contrasts=contrasts(avg,p['checkpoint_steps']))
        per_seed.append(dict(training_seed=seed,timecourse=times))
    times={}
    for t in p['observation_steps']:
        avg={w:{a:{m:statistics.fmean(s['timecourse'][str(t)]['effects'][w][a][m] for s in per_seed)
                      for m in METRICS} for a in ARMS} for w in keys}
        times[str(t)]=dict(effects=avg,contrasts=contrasts(avg,p['checkpoint_steps']))
    branch_steps=sum(r['audit']['actual_optimizer_steps'] for r in rows)
    if branch_steps!=p['budget']['branch_optimizer_steps']:raise ValueError('Branch budget mismatch')
    return dict(primary=p['analysis']['primary'],primary_value=times[str(p['horizon'])]['contrasts']['Delta_side']['loss'],
        per_seed=per_seed,timecourse=times,matched_sets=len(rows),branches=len(rows)*len(keys)*len(ARMS),cells=len(rows)*len(keys),
        branch_optimizer_steps=branch_steps,trunk_optimizer_steps=p['budget']['trunk_optimizer_steps'],
        optimizer_steps=branch_steps+p['budget']['trunk_optimizer_steps'],replication_units=len(per_seed),
        interpretation='Negative D loss helps versus native. Negative Q favors C-Select; negative Delta_side is a relative shift, '
        'not necessarily crossover or absolute benefit. Effects are on the full reporting pool of masked training treatments. '
        'G compares two-source uniform synthesis to one specified real representative. Initial norm calibration only; '
        'not matched future pressure or optimizer displacement. Reused development seeds/pool, no independent confirmation.')


def markdown_report(summary):
    t=summary['timecourse'][max(summary['timecourse'],key=int)]
    lines=['# D1: frozen source regimes on common eligible exposures','',
        f"Primary Delta_side (reporting loss): {summary['primary_value']:+.9f}",'',
        '| State | Arm | Loss minus native | R@1 change (pp) |','|---|---|---:|---:|']
    for w,arms in t['effects'].items():
        for arm,value in arms.items():
            lines.append(f"| {w} | {arm} | {value['loss']:+.9f} | {value['r1_pp']:+.5f} |")
    lines+=['',f"Q by state: {t['contrasts']['Q']}",'',
        f"Directional loss crossover: {t['contrasts']['directional_loss_crossover']}",'',summary['interpretation'],'']
    return '\n'.join(lines)
