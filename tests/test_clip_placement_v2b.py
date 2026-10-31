"""Small CPU tests of v2b; not the authenticated A100 gradient experiment."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src import clip_geometry_audit as g
from src import clip_placement_preflight as old
from src import clip_placement_v2a as head
from src import clip_placement_v2b as probe
from tests.test_clip_placement_preflight import features, preserve_test_environment

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def p():return json.loads((ROOT/'configs/clip_placement_preflight.json').read_text())


@pytest.fixture
def spec():return json.loads((ROOT/'configs/clip_placement_v2b.json').read_text())


def build(p):
    model,opt,sched,batch,cfg,record,out=features()
    shared,geom=old.auxiliary_path(out,record,1,.02,p)
    native,values,_=head.fp32_head(out,shared)
    return model,out,shared,native,values


def run_probe(model,out,shared,native,values,p,spec):
    return probe.gradients(out,shared,native,values,tuple(x for x in model.parameters() if x.requires_grad),
                           model.clip_model.logit_scale,p,spec['gradient_contract'])


def test_passing_forward_reference_is_authenticated(spec):
    parent,summary=probe.forward_reference(ROOT,spec)
    assert summary['status']=='PASS'
    assert len(parent['context']['eligible_queries'])==34
    assert summary['native_loss_change']==-.004682064056396484


def test_one_hot_vjps_scale_and_full_gradients_preserve_existing_grads(p,spec):
    model,out,shared,native,values=build(p)
    for x in model.parameters():x.grad=torch.ones_like(x)
    before=g.stamp(model)
    result=run_probe(model,out,shared,native,values,p,spec)
    assert g.stamp(model)==before
    assert len(result['per_query'])==len(shared['Wrong']['queries'])
    assert set(result['full_gradients'])=={'Native','H-Uniform','Wrong','Boundary','Correct'}
    for arm in old.ARMS:
        assert result['scale'][arm]=={k:[True,True] for k in ['candidate','partition','loss']}


def test_downstream_normalization_fails_carrier_contract(p,spec):
    model,out,shared,native,values=build(p)
    q=out.text_features[shared['Wrong']['queries']]
    values['Wrong']['e_aux']=(q*F.normalize(values['Wrong']['z'],dim=-1)).sum(-1)*out.logit_scale.detach()
    with pytest.raises(old.ContractViolation):run_probe(model,out,shared,native,values,p,spec)


def test_live_auxiliary_scale_leak_is_rejected(p,spec):
    model,out,shared,native,values=build(p)
    queries=shared['Wrong']['queries']
    raw=out.text_features@out.image_features.T
    # Candidate path is still correct; only the denominator has the wrong graph.
    values['Wrong']['A']=torch.logsumexp(raw[queries]*out.logit_scale,1)
    values['Wrong']['loss']=F.softplus(values['Wrong']['e_aux']-values['Wrong']['A']).mean()
    with pytest.raises(old.ContractViolation,match='auxiliary_scale_disconnected'):
        run_probe(model,out,shared,native,values,p,spec)


def test_forward_mismatch_blocks_all_gradient_probes(p,spec,monkeypatch):
    model,opt,sched,batch,cfg,record,out=features()
    shared,geom=old.auxiliary_path(out,record,1,.02,p)
    native_v1=head.native_record(out.raw_similarity,out.logits)
    native,values,_=head.fp32_head(out,shared)
    archived=dict(evidence=dict(placements=head.describe(shared,geom,.02)))
    parent=dict(evidence=dict(v1=dict(native=head.serialize_native(native_v1)),
        v2a=dict(native=head.serialize_native(native),placements=head.describe(values,geom,.02))))
    parent['evidence']['v2a']['placements']['Boundary']['r_train'][0]+=.01
    def forbidden(*args):raise AssertionError('Probe called before forward gate')
    monkeypatch.setattr(probe,'gradients',forbidden)
    result=probe.validate_live(out,record,1,.02,p,spec,archived,parent,native_v1,model)
    assert result['status']=='INVALID'
    assert result['reason']=='GRAD_ENABLED_V2A_REPRODUCTION_FAILED'


def test_mixed_carrier_with_fp32_head_vjps(p,spec):
    from src.clip_source_regime import forward
    model,opt,sched,batch,cfg,record,_=features()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=forward(model,batch,cfg)
        shared,_=old.auxiliary_path(out,record,1,.02,p)
        native,values,_=head.fp32_head(out,shared)
    result=run_probe(model,out,shared,native,values,p,spec)
    assert len(result['per_query'])==len(shared['Wrong']['queries'])
    assert result['scale']['Wrong']['partition']==[True,True]
