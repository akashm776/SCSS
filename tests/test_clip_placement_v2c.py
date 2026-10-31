"""CPU calibration contract tests; not an A100 acceptance result."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src.clip_placement_preflight import ContractViolation, InvalidEvidence
from src import clip_placement_v2c as c
from tests.test_clip_placement_preflight import features, preserve_test_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def p(): return json.loads((ROOT/'configs/clip_placement_preflight.json').read_text())


def row(eligible, ratio):
    return dict(native_norm=10., native_loss=1., eligible=eligible, placements={}, fp32_settings={},
                auxiliary_norms={a:10.*ratio for a in c.ARMS_CAL})


def test_passing_gradient_parent_authenticated():
    spec=json.loads((ROOT/'configs/clip_placement_v2c.json').read_text())
    result=c.gradient_reference(ROOT,spec)
    assert result['status']=='PASS'
    assert result['optimizer_updates']==0
    assert result['vjp_summary']['source']['count']==102


def test_scheduled_mean_includes_empty_steps_and_reports_conditional(p):
    rows=[row(4,.1),row(0,0),row(4,.3),row(0,0)]
    fitted=c.coefficients(rows,p)
    assert fitted['Wrong']['coefficient']==pytest.approx(.2)
    weighted=deepcopy(rows)
    for r in weighted:
        r['auxiliary_norms']={a:n*fitted[a]['coefficient'] for a,n in r['auxiliary_norms'].items()}
    c.verify(rows,weighted,fitted,p)
    for a in c.ARMS_CAL:
        assert fitted[a]['verified_scheduled_ratio']==pytest.approx(.02)
        assert fitted[a]['verified_treated_ratio']==pytest.approx(.04)


@pytest.mark.parametrize('rows',[
    [row(0,0)]*4, [row(1,0)]*4, [row(0,.01)]*4, [row(1,float('nan'))]*4])
def test_degenerate_calibration_fails_without_refitting_or_clipping(rows,p):
    with pytest.raises(ContractViolation):c.coefficients(rows,p)


def test_four_batches_required(p):
    with pytest.raises(InvalidEvidence):c.coefficients([row(1,.1)]*3,p)


def test_weighted_pressure_failure_is_not_auto_refit(p):
    rows=[row(1,.1)]*4
    fitted=c.coefficients(rows,p)
    weighted=[row(1,.025)]*4
    with pytest.raises(ContractViolation,match='weighted_scheduled_pressure'):
        c.verify(rows,weighted,fitted,p)
    assert fitted['H-Uniform']['coefficient']==pytest.approx(.2)


def test_replay_native_mismatch_invalidates_evidence(p):
    rows=[row(1,.1)]*4;weighted=[row(1,.02) for _ in range(4)]
    weighted[0]['native_loss']+=.000001
    with pytest.raises(InvalidEvidence,match='native_loss'):
        c.verify(rows,weighted,c.coefficients(rows,p),p)


def test_actual_model_calibration_restores_state_and_weighted_gradients(p):
    model,opt,sched,batch,cfg,record,_=features()
    for v in model.parameters():v.grad=torch.ones_like(v)
    before=g.stamp(model,opt,sched)
    result=c.calibration(model,opt,sched,[batch]*4,[record]*4,cfg,1,d.rng_state(),.02,p)
    assert g.stamp(model,opt,sched)==before
    assert len(result['preservation'])==8
    assert all(v['preserved'] for v in result['preservation'])
    for a in c.ARMS_CAL:
        assert result['coefficients'][a]['verified_scheduled_ratio']==pytest.approx(.02,rel=1e-5)
    assert all(all(r['scale_disconnected'].values()) for r in result['unweighted']+result['verification'])


def test_empty_actual_batch_is_zero_native_still_live(p):
    model,opt,sched,batch,cfg,record,out=features()
    record=deepcopy(record);record['eligible_queries']=[]
    native,aux=c.losses(out,record,1,.02,p,{})
    assert torch.isfinite(native) and native.requires_grad
    params=tuple(v for v in model.parameters() if v.requires_grad)
    for loss in aux.values():
        assert loss.item()==0
        assert torch.autograd.grad(loss,out.logit_scale,retain_graph=True,allow_unused=True)[0] is None
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        assert c.norm(grads)==0


def test_contract_failure_preserves_partial_evidence_and_state(p,monkeypatch):
    model,opt,sched,batch,cfg,record,_=features()
    before=g.stamp(model,opt,sched)
    def fail(*args):
        args[-1]['measured_before_failure']=True
        with torch.no_grad():next(model.parameters()).add_(1)
        raise ContractViolation('test',1,0,path='test')
    monkeypatch.setattr(c,'observe',fail)
    with pytest.raises(ContractViolation) as error:
        c.calibration(model,opt,sched,[batch]*4,[record]*4,cfg,1,d.rng_state(),.02,p)
    assert g.stamp(model,opt,sched)==before
    assert error.value.evidence['unweighted'][0]['measured_before_failure']
    assert error.value.evidence['preservation'][0]['preserved']
