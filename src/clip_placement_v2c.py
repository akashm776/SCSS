"""Four scheduled-batch FP32-head calibration. No optimizer updates."""
import math
import statistics
from pathlib import Path

import torch

from src import clip_geometry_audit as g
from src import clip_train
from src.clip_source_regime import forward
from src.clip_placement_preflight import ARMS, auxiliary_path, finite, require, InvalidEvidence
from src.clip_placement_v2a import fp32_head, placement_contract
from src.clip_placement_v2b import norm, vector_summary
from src.clip_placement_preflight_run import describe, capture_contract_evidence
from src.clip_placement_evidence import read_checked

ARMS_CAL = ('H-Uniform', 'Wrong', 'Boundary', 'Correct')


def gradient_reference(root, spec):
    from scripts.audit_clip_placement_v2b import audit
    ref = spec['gradient_reference']; folder = Path(root)/ref['root']
    try:
        for name, sha in ref['files'].items(): read_checked(folder, name, sha)
        result = audit(folder)
        if result['contract_sha256'] != ref['contract_sha256']:
            raise ValueError('Wrong v2b parent contract')
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise InvalidEvidence('Passing v2b parent authentication failed: '+str(error)) from error
    return result


def losses(out, record, w, delta, p, evidence):
    """Called inside ORIGINAL encoder/carrier autocast; FP32 head owns its context.

    Empty populations bypass fp32_head's nonempty candidate path (mean of an
    empty vector is undefined). Their differentiable zero excludes the scale.
    H-Uniform uses the very same original carrier as the placement family.
    """
    if not record['eligible_queries']:
        native, _, settings = fp32_head(out, {})
        zero = native['similarity'].sum()*0
        evidence.update(empty=True, placements={}, fp32_settings=settings)
        return native['loss_native'], {a: zero for a in ARMS_CAL}
    shared, geom = auxiliary_path(out, record, w, delta, p)
    uniform = dict(shared['Wrong'], z=shared['Wrong']['z0'])
    native, values, settings = fp32_head(out, dict(shared, **{'H-Uniform': uniform}))
    placed = {a: values[a] for a in ARMS}
    evidence.update(empty=False, placements=describe(placed, geom, delta), fp32_settings=settings)
    contract = placement_contract(placed, delta, p)
    evidence['placement_contract'] = contract
    require(contract['status']=='PASS', 'calibration_placement', contract, 'PASS', 'FP32_head')
    for a, value in values.items(): finite(value['loss'], a+'/loss')
    return native['loss_native'], {a: values[a]['loss'] for a in ARMS_CAL}


def observe(model, batch, config, record, w, delta, p, coefficients, evidence):
    device = next(model.parameters()).device
    with torch.enable_grad(), clip_train._autocast(device, config['training']['mixed_precision']):
        out = forward(model, batch, config)
        native_loss, aux = losses(out, record, w, delta, p, evidence)
    params = tuple(v for v in model.parameters() if v.requires_grad)
    # Backward outside autocast, as in the authenticated v2b implementation.
    with g.scoring_precision(device):
        native = torch.autograd.grad(native_loss, params, retain_graph=True, allow_unused=True)
        for i, v in enumerate(native):
            if v is not None: finite(v, f'native/gradient{i}')
        native_norm = norm(native)
        evidence.update(native_norm=native_norm, native_loss=float(native_loss.detach()),
                        eligible=len(record['eligible_queries']), auxiliary_norms={}, native_auxiliary_cosines={},
                        auxiliary_losses={}, scale_disconnected={})
        require(math.isfinite(native_norm) and native_norm>p['gradient_norm_min'],
                'native_gradient_norm', native_norm, p['gradient_norm_min'], 'calibration')
        for a in ARMS_CAL:
            loss = coefficients[a]*aux[a]
            scale_grad = torch.autograd.grad(loss, out.logit_scale, retain_graph=True, allow_unused=True)[0]
            evidence['scale_disconnected'][a] = scale_grad is None
            require(scale_grad is None, 'auxiliary_scale_disconnected', scale_grad is None, True, a)
            grads = torch.autograd.grad(loss, params, retain_graph=a!=ARMS_CAL[-1], allow_unused=True)
            for i, v in enumerate(grads):
                if v is not None: finite(v, f'{a}/gradient{i}')
            measured = vector_summary(grads, native)
            evidence['auxiliary_norms'][a] = measured['norm']
            evidence['native_auxiliary_cosines'][a] = measured['cosine']
            evidence['auxiliary_losses'][a] = float(loss.detach())
            valid = measured['norm']>p['gradient_norm_min'] if record['eligible_queries'] else measured['norm']==0
            require(valid, 'auxiliary_gradient_norm', measured['norm'],
                    p['gradient_norm_min'] if record['eligible_queries'] else 0., a)


def coefficients(rows, p):
    if len(rows)!=4: raise InvalidEvidence('Exactly four scheduled calibration batches required')
    result = {}
    for row in rows:
        n = row['native_norm']
        require(math.isfinite(n) and n>p['gradient_norm_min'], 'native_gradient_norm', n, p['gradient_norm_min'], 'calibration')
    for a in ARMS_CAL:
        ratios = []
        for row in rows:
            n = row['auxiliary_norms'][a]
            valid = math.isfinite(n) and (n>p['gradient_norm_min'] if row['eligible'] else n==0)
            require(valid, 'auxiliary_gradient_norm', n, p['gradient_norm_min'] if row['eligible'] else 0., a)
            ratios.append(n/row['native_norm'])
        c = statistics.fmean(ratios)
        require(math.isfinite(c) and c>p['gradient_norm_min'], 'calibration_mean_ratio', c, p['gradient_norm_min'], a)
        alpha = p['target_ratio']/c
        require(math.isfinite(alpha) and alpha>0, 'finite_positive_coefficient', alpha, '>0 finite', a)
        treated = [r for r, row in zip(ratios, rows) if row['eligible']]
        result[a] = dict(coefficient=alpha, unweighted_scheduled_ratio=c,
                         unweighted_treated_ratio=statistics.fmean(treated), batch_ratios=ratios)
    return result


def verify(rows, weighted, fitted, p):
    if len(rows)!=4 or len(weighted)!=4: raise InvalidEvidence('Incomplete calibration replay')
    for before, after in zip(rows, weighted):
        # A restored forward must be exactly identical, not merely close enough
        # to compensate for a different denominator in the pressure ratio.
        for name in ['native_norm', 'native_loss', 'eligible', 'placements', 'fp32_settings']:
            if before[name]!=after[name]: raise InvalidEvidence('Weighted replay changed '+name)
    for a in ARMS_CAL:
        ratios = [row['auxiliary_norms'][a]/row['native_norm'] for row in weighted]
        mean = statistics.fmean(ratios); error = abs(mean/p['target_ratio']-1)
        fitted[a].update(verified_batch_ratios=ratios, verified_scheduled_ratio=mean,
                        verified_treated_ratio=statistics.fmean([r for r,row in zip(ratios,weighted) if row['eligible']]),
                        relative_error=error)
        require(math.isfinite(error) and error<=p['calibration_relative_tolerance'],
                'weighted_scheduled_pressure', mean, p['target_ratio'], a)


def calibration(model, opt, sched, batches, records, config, w, common, delta, p):
    if len(batches)!=4 or len(records)!=4: raise InvalidEvidence('Expected four calibration batches')
    result = dict(unweighted=[], verification=[], coefficients={}, preservation=[], optimizer_updates=0)
    with capture_contract_evidence(result):
        for phase in ['unweighted', 'verification']:
            weights = {a:1. for a in ARMS_CAL} if phase=='unweighted' else {a:result['coefficients'][a]['coefficient'] for a in ARMS_CAL}
            for offset, (batch, record) in enumerate(zip(batches, records)):
                row = dict(batch_offset=offset, eligible_queries=record['eligible_queries'])
                result[phase].append(row)
                with g.preserved(model,opt,sched) as restoration:
                    result['preservation'].append(restoration)
                    g.restore_rng(common); model.train()
                    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
                    observe(model,batch,config,record,w,delta,p,weights,row)
            if phase=='unweighted': result['coefficients']=coefficients(result[phase],p)
        verify(result['unweighted'],result['verification'],result['coefficients'],p)
    return result
