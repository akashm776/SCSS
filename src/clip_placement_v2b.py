"""First-batch gradient contract for the unchanged v2a forward head."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src import clip_geometry_audit as g
from src.clip_placement_preflight import (ARMS, InvalidEvidence, ContractViolation,
    auxiliary_path, finite, gradient_metrics, require)
from src.clip_placement_v2a import fp32_head, reproduction, serialize_native
from src.clip_placement_preflight_run import describe
from src.clip_placement_evidence import read_checked
from scripts.audit_clip_placement_v2a import audit as audit_forward


def forward_reference(root, spec):
    ref = spec['forward_reference'];folder = Path(root)/ref['root']
    try:
        for name, sha in ref['files'].items(): read_checked(folder,name,sha)
        summary = audit_forward(folder)
    except (ValueError,KeyError,FileNotFoundError) as error:
        raise InvalidEvidence('Passing v2a parent authentication failed: '+str(error)) from error
    if summary['contract_sha256']!=ref['contract_sha256']:
        raise InvalidEvidence('Wrong passing forward parent')
    index=json.loads((folder/'index.json').read_text())
    entry=next(iter(index.values()))
    record=read_checked(folder,entry['file'],entry['sha256'])
    return record['result'],summary


def norm(grads):
    return sum(float(x.detach().double().square().sum()) for x in grads if x is not None)**.5


def vector_summary(actual, reference):
    a,b=norm(actual),norm(reference)
    dot=sum(float((x.detach().double()*y.detach().double()).sum())
            for x,y in zip(actual,reference) if x is not None and y is not None)
    return dict(norm=a,reference_norm=b,cosine=None if a==0 or b==0 else dot/(a*b))


def gradients(out, shared, native, values, params, log_scale, p, contract):
    """Actual live encoder outputs → original dense carrier → candidate logits.

    One-hot VJPs prevent cancellation between exposure gradients. Targets remain
    detached. All gradient reads use autograd.grad and never accumulate .grad.
    """
    checks=dict(p,gradient_vector_rtol=contract['vector_rtol'],
        gradient_cosine_min=contract['cosine_min'],gradient_norm_min=contract['norm_min'])
    first=shared['Wrong'];queries=first['queries'];z0=first['z0']
    require(out.logit_scale.requires_grad and log_scale.requires_grad,
            'native_scale_connected',out.logit_scale.requires_grad,True,'scale')
    with g.scoring_precision(out.text_features.device):
        q=out.text_features[queries].float();tau=out.logit_scale.detach().float()
        uniform=(q*z0.float()).sum(-1)*tau
        raw=out.text_features.float()@out.image_features.float().T
        uniform_loss=F.softplus(uniform-torch.logsumexp(raw[queries]*tau,1)).mean()
    result=dict(per_query={},scale={},full_gradients={})
    try:
        for arm in ARMS:
            require(not shared[arm]['target'].requires_grad,'target_detached',
                    shared[arm]['target'].requires_grad,False,arm)
        for i,query in enumerate(queries.tolist()):
            ref=torch.autograd.grad(uniform[i],out.image_features,retain_graph=True)[0]
            row={}
            for arm,v in values.items():
                source,carrier,query_grad=torch.autograd.grad(v['e_aux'][i],
                    (out.image_features,z0,out.text_features),retain_graph=True)
                expected_carrier=torch.zeros_like(z0)
                expected_carrier[i]=(tau*q[i]).to(z0.dtype)
                expected_query=torch.zeros_like(out.text_features)
                expected_query[query]=(tau*v['z'][i].float()).to(out.text_features.dtype)
                row[arm]=dict(
                    source=gradient_metrics(source,ref,checks,path=f'{arm}/query{query}/source'),
                    carrier=gradient_metrics(carrier,expected_carrier,checks,path=f'{arm}/query{query}/carrier'),
                    query=gradient_metrics(query_grad,expected_query,checks,path=f'{arm}/query{query}/query'))
            result['per_query'][str(query)]=row
        # Native scale connectivity is checked against the analytic symmetric CE
        # derivative, including exp(logit_scale)'s chain factor.
        d_tau,d_log=torch.autograd.grad(native['loss_native'],(out.logit_scale,log_scale),retain_graph=True)
        with torch.no_grad(),g.scoring_precision(out.text_features.device):
            scores=native['similarity'].float();logits=native['logits'].float()
            expected_tau=.5*((logits.softmax(1)*scores).sum(1).mean()+
                              (logits.softmax(0)*scores).sum(0).mean())-scores.diagonal().mean()
            expected_log=expected_tau*out.logit_scale.float()
        for name,observed,expected in [('tau',d_tau,expected_tau),('logit_scale',d_log,expected_log)]:
            finite(observed,'native_scale/'+name)
            error=abs(float(observed)-float(expected))
            limit=contract['scale_atol']+contract['scale_rtol']*abs(float(expected))
            result['scale'][name]=dict(observed=float(observed),expected=float(expected),error=error,tolerance=limit)
            require(error<=limit,'native_scale_derivative',error,limit,name)
        for arm,v in values.items():
            # Require disconnection separately at candidate, partition, and loss.
            parts={}
            for name,scalar in [('candidate',v['e_aux'].sum()),('partition',v['A'].sum()),('loss',v['loss'])]:
                grads=torch.autograd.grad(scalar,(out.logit_scale,log_scale),allow_unused=True,retain_graph=True)
                parts[name]=[x is None for x in grads]
                require(all(parts[name]),'auxiliary_scale_disconnected',parts[name],[True,True],arm+'/'+name)
            result['scale'][arm]=parts
        native_grad=torch.autograd.grad(native['loss_native'],params,allow_unused=True,retain_graph=True)
        uniform_grad=torch.autograd.grad(uniform_loss,params,allow_unused=True,retain_graph=True)
        for name,grads in [('Native',native_grad),('H-Uniform',uniform_grad)]:
            for i,x in enumerate(grads):
                if x is not None:finite(x,f'{name}/parameter{i}')
            result['full_gradients'][name]=dict(norm=norm(grads),nonzero_tensors=sum(x is not None and bool(x.count_nonzero()) for x in grads))
        for arm,v in values.items():
            full=torch.autograd.grad(v['loss'],params,allow_unused=True,retain_graph=True)
            for i,x in enumerate(full):
                if x is not None:finite(x,f'{arm}/parameter{i}')
            result['full_gradients'][arm]=dict(vs_native=vector_summary(full,native_grad),
                vs_uniform=vector_summary(full,uniform_grad),
                nonzero_tensors=sum(x is not None and bool(x.count_nonzero()) for x in full))
    except ContractViolation as error:
        error.evidence=result
        raise
    return result


def validate_live(out, record, w, delta, p, spec, archived_v1, parent, native_v1, model):
    """First authenticate grad-enabled forward values, then run gradient probes."""
    shared,geom=auxiliary_path(out,record,w,delta,p)
    old=describe(shared,geom,delta)
    gate=reproduction(old,archived_v1['evidence']['placements'],spec)
    evidence=dict(v1_reproduction=gate)
    if gate['status']!='PASS':return dict(status='INVALID',reason='GRAD_ENABLED_V1_REPRODUCTION_FAILED',evidence=evidence)
    native,values,settings=fp32_head(out,shared)
    current=dict(v1_native=serialize_native(native_v1),v2a_native=serialize_native(native),
                 placements=describe(values,geom,delta))
    expected=dict(v1_native=parent['evidence']['v1']['native'],v2a_native=parent['evidence']['v2a']['native'],
                  placements=parent['evidence']['v2a']['placements'])
    gate=reproduction(current,expected,spec)
    evidence.update(forward_reproduction=gate,fp32_settings=settings)
    if gate['status']!='PASS':return dict(status='INVALID',reason='GRAD_ENABLED_V2A_REPRODUCTION_FAILED',evidence=evidence)
    params=tuple(x for x in model.parameters() if x.requires_grad)
    try:
        with torch.autocast(device_type=out.text_features.device.type,enabled=False):
            evidence['gradients']=gradients(out,shared,native,values,params,model.clip_model.logit_scale,p,spec['gradient_contract'])
    except ContractViolation as error:
        evidence['gradients']=getattr(error,'evidence',{})
        return dict(status='FAIL',violation=error.record,evidence=evidence)
    return dict(status='PASS',evidence=evidence)
