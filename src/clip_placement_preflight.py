"""D2-P contracts. Observational only: this module never calls optimizer.step()."""
import math
import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src import clip_geometry_audit as geometry
from src.clip_source_regime import candidate

ARMS = {'Wrong': -1., 'Boundary': 0., 'Correct': 1.}


class InvalidEvidence(ValueError):
    """Provenance/configuration does not identify the prescribed experiment."""


class ContractViolation(ValueError):
    def __init__(self, requirement, observed, threshold, *, path, indices=None):
        def json_safe(value):
            if isinstance(value,float) and not math.isfinite(value):return str(value)
            if isinstance(value,list):return [json_safe(v) for v in value]
            if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
            return value
        self.record = dict(requirement=requirement, observed=json_safe(observed), threshold=json_safe(threshold),
                           path=path, indices=indices)
        super().__init__(str(self.record))


def require(condition, requirement, observed, threshold, path, indices=None):
    if not condition:
        raise ContractViolation(requirement, observed, threshold, path=path, indices=indices)


def finite(x, path):
    # Unexplained nonfinite tensors are numerical/implementation errors, not geometry.
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError('Nonfinite tensor: '+path)


def bounded(errors, limit, requirement, path):
    finite(errors, path)
    bad = errors > limit
    require(not bool(bad.any()), requirement, errors.detach().cpu().tolist(),
            float(limit), path, bad.nonzero().flatten().cpu().tolist())


def targets(q_live, positive_live, z0_live, delta, p, arms=None):
    """Detached geometry; original z0 remains the sole source backward carrier."""
    if not math.isfinite(delta):
        raise InvalidEvidence('Nonfinite global gap')
    require(delta >= p['delta_min'], 'minimum_gap', delta, p['delta_min'], 'geometry')
    arms = list(ARMS) if arms is None else arms
    with torch.no_grad(), geometry.scoring_precision(q_live.device):
        vectors = []
        for name, value in [('query',q_live),('positive',positive_live),('carrier',z0_live)]:
            value = value.detach().float()
            finite(value, name)
            norms = value.norm(dim=-1)
            require(bool((norms > 0).all()), 'nonzero_input_vector', norms.cpu().tolist(), 0., name)
            vectors.append(F.normalize(value, dim=-1))
        q, v, z0 = vectors
        positive = (q*v).sum(-1)
        h = z0 - (q*z0).sum(-1,keepdim=True)*q
        hnorm = h.norm(dim=-1)
        require(bool((hnorm > p['eps_u']).all()), 'orthogonal_norm', hnorm.cpu().tolist(), p['eps_u'], 'geometry')
        u = h / hnorm[:,None]
        result = {}
        for arm in arms:
            requested = ARMS[arm]*delta
            a = positive-requested
            finite(a, arm+'/target_cosine')
            require(bool((a.abs() <= 1.).all()), 'mathematical_feasibility', a.cpu().tolist(), 1., arm)
            require(bool((a.abs() <= 1.-p['eps_cos']).all()), 'cosine_headroom', a.cpu().tolist(), 1.-p['eps_cos'], arm)
            z = F.normalize(a[:,None]*q + torch.sqrt(1-a*a)[:,None]*u, dim=-1)
            margin = positive-(q*z).sum(-1)
            bounded((z.norm(dim=-1)-1).abs(),p['unit_norm_atol'],'unit_norm',arm+'/FP32')
            bounded((margin-requested).abs(), max(p['fp32_margin_atol'],p['fp32_margin_rtol_delta']*delta),
                    'target_margin',arm+'/FP32')
            result[arm] = z
    return result, dict(p_geom=positive, orthogonal_norm=hnorm,
                       margins={a:positive-(q*z).sum(-1) for a,z in result.items()})


def straight_through(z0, target):
    target = target.detach().to(z0.dtype)
    return target + (z0-z0.detach())


def auxiliary_path(out, record, w, delta, p, *, arms=None):
    """Actual loss path, retaining D1's detached auxiliary scale and live partition."""
    queries = torch.tensor(record['eligible_queries'],device=out.raw_similarity.device,dtype=torch.long)
    if not len(queries):
        zero = out.raw_similarity.sum()*0
        return {a:dict(loss=zero, empty=True) for a in (arms or ARMS)}, {}
    # Calling the existing function preserves its dense weight multiplication exactly.
    try:
        _, queries, ids, z0 = candidate(out, record, w, 'H_Synth')
    except ValueError as error:
        if str(error) != 'Invalid/nonfinite near-zero mixture; no fallback':
            raise InvalidEvidence('Invalid frozen D1 carrier inputs') from error
        # Diagnose the existing path's rejected carrier without replacing it.
        ids = torch.tensor(record['states'][str(w)]['supports']['H'], device=queries.device)
        weights = torch.zeros((len(queries),len(out.image_features)),
                              device=queries.device,dtype=out.raw_similarity.dtype)
        weights.scatter_(1,ids,.5)
        mu = weights.to(out.image_features.dtype) @ out.image_features
        finite(mu,'D1 carrier')
        norms = mu.detach().float().norm(dim=-1)
        raise ContractViolation('D1_carrier_norm',norms.cpu().tolist(),1e-6,path='carrier') from error
    target, geom = targets(out.text_features[queries], out.image_features[queries], z0,delta,p,arms)
    scale = out.logit_scale.detach()
    finite(scale, 'scale')
    if scale.numel()!=1 or not bool(scale>0):
        raise InvalidEvidence('A finite positive scalar CLIP scale is required')
    # D1's auxiliary partition is raw similarity multiplied by DETACHED scale.
    base_logits = out.raw_similarity[queries]*scale
    partition = torch.logsumexp(base_logits,dim=1)
    p_sim = out.raw_similarity.diagonal()[queries]
    ell_plus = out.logits.diagonal()[queries]
    values = {}
    for arm,zstar in target.items():
        z = straight_through(z0,zstar)
        sim = (out.text_features[queries]*z).sum(-1)
        extra = sim*scale
        m = extra-partition
        loss = F.softplus(m).mean()
        values[arm] = dict(loss=loss, empty=False, z=z, z0=z0, target=zstar,
            queries=queries, ids=ids, p_sim=p_sim, ell_plus=ell_plus, s_aux=sim,
            e_aux=extra, A=partition, m_loss=m, tau=scale,
            r_train=p_sim-sim, g_logit=ell_plus-extra,
            epsilon_m=m-(extra-partition), epsilon_scale=(ell_plus-extra)-scale*(p_sim-sim),
            P_plus=torch.exp(ell_plus-partition), sigmoid_m=torch.sigmoid(m))
    return values,geom


def audit_placement(values, delta, p):
    if all(v['empty'] for v in values.values()):
        return
    for arm,v in values.items():
        requested=ARMS[arm]*delta
        tau=float(v['tau'])
        for field in ['loss','r_train','g_logit','m_loss','epsilon_m','epsilon_scale']:
            finite(v[field],arm+'/'+field)
        # Subtract captured rounded tensors in FP32 for error comparisons.
        bounded((v['r_train'].float()-requested).abs(),p['training_margin_rtol_delta']*delta,
                'training_margin',arm+'/mixed')
        bounded((v['g_logit'].float()-tau*requested).abs(),p['training_logit_rtol_scaled_delta']*tau*delta,
                'training_logit_gap',arm+'/mixed')
        if requested:
            require(bool((v['g_logit']*ARMS[arm]>0).all()),'logit_sign',v['g_logit'].detach().cpu().tolist(),
                    int(ARMS[arm]),arm+'/mixed')
    if set(values)==set(ARMS):
        for left,right in [('Wrong','Boundary'),('Boundary','Correct')]:
            separation=values[right]['r_train'].float()-values[left]['r_train'].float()
            require(bool((separation>=p['minimum_adjacent_separation_delta']*delta).all()),
                    'adjacent_separation',separation.detach().cpu().tolist(),
                    p['minimum_adjacent_separation_delta']*delta,left+'/'+right)


def gradient_metrics(actual, expected, p, *, path, enforce=True):
    actual=actual.detach().double().flatten(); expected=expected.detach().double().flatten()
    finite(actual,path); finite(expected,path)
    n=float(expected.norm()); a=float(actual.norm())
    if enforce:
        require(n>p['gradient_norm_min'],'nonzero_reference_gradient',n,p['gradient_norm_min'],path)
    err=float((actual-expected).norm())/max(n,p['gradient_norm_min'])
    cosine=None if not n or not a else float(torch.dot(actual,expected)/(actual.norm()*expected.norm()))
    if enforce:
        require(err<=p['gradient_vector_rtol'],'gradient_vector_error',err,p['gradient_vector_rtol'],path)
        require(cosine is not None and cosine>=p['gradient_cosine_min'],'gradient_cosine',cosine,p['gradient_cosine_min'],path)
    return dict(relative_vector_error=err,norm=a,reference_norm=n,cosine=cosine)


def feature_probe(out, record, w, delta, p, *, fp32=False):
    """Real dense D1 carrier → source-feature/logit path; no encoder/optimizer update.

    Candidate-only VJPs isolate source carrier from the native-denominator path.
    Full auxiliary VJPs include that path and are descriptive, not equality tests.
    """
    image=out.image_features.detach().clone(); text=out.text_features.detach().clone()
    if fp32: image=image.float(); text=text.float()
    image.requires_grad_(); text.requires_grad_()
    raw=text@image.T
    scale=out.logit_scale.detach().clone()
    local=SimpleNamespace(image_features=image,text_features=text,raw_similarity=raw,
                          logit_scale=scale,logits=raw*scale)
    if not record['eligible_queries']: return dict(empty=True)
    uniform,queries,ids,z0=candidate(local,record,w,'H_Synth')
    vals,_=auxiliary_path(local,record,w,delta,p)
    ref,ref_query=torch.autograd.grad((uniform*scale).sum(),(image,text),retain_graph=True)
    ref_full=torch.autograd.grad(F.softplus(uniform*scale-torch.logsumexp(raw[queries]*scale,1)).mean(),
                                 (image,text),retain_graph=True)
    result={}
    for arm,v in vals.items():
        source,carrier,query=torch.autograd.grad(v['e_aux'].sum(),(image,v['z0'],text),retain_graph=True)
        expected_carrier=(text[queries]*scale).to(carrier.dtype)
        full=torch.autograd.grad(v['loss'],(image,text),retain_graph=True)
        result[arm]=dict(source=gradient_metrics(source,ref,p,path=arm+'/source'),
            carrier=gradient_metrics(carrier,expected_carrier,p,path=arm+'/carrier'),
            candidate_query_feature=gradient_metrics(query,ref_query,p,path=arm+'/candidate_query',enforce=False),
            full_source=gradient_metrics(full[0],ref_full[0],p,path=arm+'/full_source',enforce=False),
            full_query=gradient_metrics(full[1],ref_full[1],p,path=arm+'/full_query',enforce=False))
    return result


def calibration_coefficients(rows,p):
    if len(rows)!=p['calibration_batches']: raise InvalidEvidence('Incomplete calibration batch grid')
    result={}
    for index,row in enumerate(rows):
        n=row['native_norm']
        require(math.isfinite(n) and n>p['gradient_norm_min'],'native_gradient_norm',n,p['gradient_norm_min'],f'calibration/batch{index}')
    for arm in ARMS:
        ratios=[]; treated=[]
        for index,row in enumerate(rows):
            n=row['auxiliary_norms'][arm]
            valid=math.isfinite(n) and (n>p['gradient_norm_min'] if row['eligible'] else n==0)
            require(valid,'auxiliary_gradient_norm',n,p['gradient_norm_min'] if row['eligible'] else 0.,f'{arm}/calibration/batch{index}')
            ratio=n/row['native_norm'];ratios.append(ratio)
            if row['eligible']: treated.append(ratio)
        c=statistics.fmean(ratios)
        require(math.isfinite(c) and c>p['gradient_norm_min'],'calibration_mean',c,p['gradient_norm_min'],arm)
        alpha=p['target_ratio']/c
        require(math.isfinite(alpha) and alpha>0,'coefficient',alpha,'positive finite; uncapped',arm)
        result[arm]=dict(coefficient=alpha,mean_unweighted_ratio=c,
            scheduled_algebraic_ratio=alpha*c,
            treated_algebraic_ratio=None if not treated else alpha*statistics.fmean(treated))
    return result


def acceptance(*, valid, complete, violations, expected, completed, error=None):
    # A known violation remains visible even if later work is interrupted.
    status='INVALID' if not valid else 'FAIL' if violations else 'PASS' if complete and completed==expected else 'INCOMPLETE'
    return dict(status=status,provenance_valid=valid,execution_complete=complete,
                expected_checks=expected,completed_checks=completed,violations=violations,
                operational_error=error,launch_eligible=status=='PASS',d2_launched=False,optimizer_updates=0)
