"""CPU implementation tests; these do NOT substitute for the A100 D2-P audit."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src import clip_geometry_audit as g
from src import clip_checkpoint_diagnostic as d
from src import clip_source_regime as d1
from src import clip_placement_preflight as p2
from src.clip_placement_evidence import Journal, validate_gap
from tests.test_clip_source_regime import toy

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def p():return json.loads((ROOT/'configs/clip_placement_preflight.json').read_text())


@pytest.fixture(autouse=True)
def preserve_test_environment():
    rng=deepcopy(d.rng_state());backend=g.backend_state()
    yield
    g.restore_rng(rng);g.restore_backend(backend)


def features():
    model,opt,sched,batch,cfg,states,report,record,protocol=toy()
    return model,opt,sched,batch,cfg,record,d1.forward(model,batch,cfg)


def test_exact_forward_and_source_vjp_with_distinct_full_gradients(p):
    *_,record,out=features()
    values,geom=p2.auxiliary_path(out,record,1,.02,p)
    p2.audit_placement(values,.02,p)
    for a,v in values.items():
        assert torch.equal(v['z'],v['target'].to(v['z'].dtype))
        assert torch.allclose(v['r_train'],torch.full_like(v['r_train'],p2.ARMS[a]*.02),atol=1e-6)
    measured=p2.feature_probe(out,record,1,.02,p,fp32=True)
    assert all(x['source']['relative_vector_error']<1e-6 for x in measured.values())
    assert any(x['full_source']['relative_vector_error']>1e-3 for x in measured.values())


def test_detached_geometry_does_not_send_gradients_to_positive_or_target(p):
    q=F.normalize(torch.randn(3,12),dim=-1).requires_grad_()
    pos=F.normalize(torch.randn(3,12),dim=-1).requires_grad_()
    carrier=F.normalize(torch.randn(3,12),dim=-1).requires_grad_()
    target,_=p2.targets(q,pos,carrier,.02,p)
    z=p2.straight_through(carrier,target['Boundary'])
    dq,dp,dc=torch.autograd.grad((q*z).sum(),(q,pos,carrier),allow_unused=True)
    assert dp is None and torch.equal(dc,q) and torch.equal(dq,target['Boundary'])


def test_post_st_normalization_is_detected_by_vector_error(p):
    q=F.normalize(torch.randn(2,8),dim=-1)
    z0=F.normalize(torch.randn(2,8),dim=-1).requires_grad_()
    z=F.normalize(p2.straight_through(z0,q),dim=-1)
    bad=torch.autograd.grad((q*z).sum(),z0)[0]
    with pytest.raises(p2.ContractViolation,match='gradient_vector_error'):
        p2.gradient_metrics(bad,q,p,path='unwanted normalization')


def test_assigned_target_does_not_require_unused_wrong_target(p):
    q=torch.tensor([[1.,0.,0.]])
    pos=F.normalize(torch.tensor([[1.,.01,0.]]),dim=-1)
    z0=torch.tensor([[0.,1.,0.]])
    targets,_=p2.targets(q,pos,z0,.02,p,arms=['Correct'])
    assert list(targets)==['Correct']
    with pytest.raises(p2.ContractViolation,match='mathematical_feasibility'):
        p2.targets(q,pos,z0,.02,p)


def test_headroom_and_degenerate_geometry_are_distinct(p):
    q=torch.tensor([[1.,0.,0.]])
    with pytest.raises(p2.ContractViolation,match='orthogonal_norm'):
        p2.targets(q,torch.tensor([[0.,1.,0.]]),q,.02,p)
    # Mathematically legal a=1 is outside the frozen numerical headroom.
    with pytest.raises(p2.ContractViolation,match='cosine_headroom'):
        p2.targets(q,q,torch.tensor([[0.,1.,0.]]),.02,p,arms=['Boundary'])


def test_actual_logits_are_checked_even_when_similarity_passes(p):
    *_,record,out=features()
    values,_=p2.auxiliary_path(out,record,1,.02,p)
    values['Wrong']['g_logit']=values['Wrong']['g_logit'].abs()
    with pytest.raises(p2.ContractViolation,match='training_logit_gap'):
        p2.audit_placement(values,.02,p)


def test_nonfinite_is_not_scientific_geometry_failure(p):
    q=torch.tensor([[float('nan'),0.]])
    with pytest.raises(FloatingPointError):p2.targets(q,q,q,.02,p)


def test_invalid_calibration_evidence_is_json_serializable(p):
    rows=[dict(native_norm=float('nan'),auxiliary_norms={a:1. for a in p2.ARMS},eligible=1)]*4
    with pytest.raises(p2.ContractViolation) as error:p2.calibration_coefficients(rows,p)
    assert json.loads(json.dumps(error.value.record,allow_nan=False))['observed']=='nan'


def test_mixed_precision_source_vjp_and_carrier_failure(p):
    *_,record,out=features()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result=p2.feature_probe(out,record,1,.02,p)
    assert all(v['source']['relative_vector_error']<1e-6 for v in result.values())
    # A finite exactly cancelled source pair violates the carrier contract;
    # this is distinct from an unexplained NaN or device exception.
    broken=deepcopy(record)
    for state in broken['states'].values():
        state['supports']['H']=[[30,31] for _ in broken['eligible_queries']]
    broken['eligible_queries']=[i for i in broken['eligible_queries'] if i not in [30,31]]
    for state in broken['states'].values():
        # Preserve a valid, matching support grid after removing those queries.
        original=record['eligible_queries']
        for regime,rows in state['supports'].items():
            state['supports'][regime]=[row for i,row in zip(original,rows) if i not in [30,31]]
    image=out.image_features.detach().clone();image[31]=-image[30]
    local=SimpleNamespace(**vars(out));local.image_features=image
    with pytest.raises(p2.ContractViolation,match='D1_carrier_norm'):
        p2.auxiliary_path(local,broken,1,.02,p)


def test_empty_mask_zero_loss_and_conditional_calibration(p):
    *_,record,out=features();record=deepcopy(record);record['eligible_queries']=[]
    values,_=p2.auxiliary_path(out,record,1,.02,p)
    assert all(float(v['loss'].detach())==0 and v['empty'] for v in values.values())
    grad=torch.autograd.grad(values['Wrong']['loss'],out.raw_similarity)[0]
    assert not grad.count_nonzero()
    rows=[dict(native_norm=2.,auxiliary_norms={a:.2 if i<2 else 0. for a in p2.ARMS},eligible=3 if i<2 else 0) for i in range(4)]
    result=p2.calibration_coefficients(rows,p)
    assert result['Wrong']['coefficient']==pytest.approx(.4)
    assert result['Wrong']['scheduled_algebraic_ratio']==pytest.approx(.02)
    assert result['Wrong']['treated_algebraic_ratio']==pytest.approx(.04)
    rows[0]['auxiliary_norms']['Wrong']=0.
    with pytest.raises(p2.ContractViolation):p2.calibration_coefficients(rows,p)


def test_state_preservation_including_exception_and_diagnostics(p):
    model,opt,sched,batch,cfg,record,out=features()
    params=list(model.parameters());params[0].grad=torch.ones_like(params[0])
    before=g.stamp(model,opt,sched)
    with pytest.raises(RuntimeError,match='interruption'):
        with g.preserved(model,opt,sched):
            model.eval();torch.rand(10);params[0].grad=None
            with torch.no_grad():params[0].add_(1.)
            raise RuntimeError('interruption')
    assert before==g.stamp(model,opt,sched)
    from src.clip_placement_preflight_run import audit_batch
    result=audit_batch(model,opt,sched,batch,cfg,record,1,d.rng_state(),.02,p)
    assert result['preservation']['preserved'] and result['diagnostic_on_off_identical']
    assert before==g.stamp(model,opt,sched)


def test_calibration_real_autograd_and_preservation(p):
    model,opt,sched,batch,cfg,record,out=features()
    from src.clip_placement_preflight_run import calibration
    before=g.stamp(model,opt,sched)
    result=calibration(model,opt,sched,[batch]*4,[record]*4,cfg,1,d.rng_state(),.02,p)
    assert all(x['verified_scheduled_ratio']==pytest.approx(.02,rel=.05) for x in result['coefficients'].values())
    assert before==g.stamp(model,opt,sched) and result['optimizer_updates']==0


def test_failed_audit_keeps_restoration_evidence(p):
    model,opt,sched,batch,cfg,record,out=features()
    from src.clip_placement_preflight_run import audit_batch
    with pytest.raises(p2.ContractViolation) as failure:
        audit_batch(model,opt,sched,batch,cfg,record,1,d.rng_state(),2.,p)
    assert failure.value.evidence['preservation']['preserved']


@pytest.mark.parametrize('valid,complete,violations,status',[
    (True,True,[],'PASS'),(True,False,[],'INCOMPLETE'),
    (True,False,[{'requirement':'x'}],'FAIL'),(False,True,[],'INVALID')])
def test_acceptance_retains_failures_during_interruptions(valid,complete,violations,status):
    value=p2.acceptance(valid=valid,complete=complete,violations=violations,expected=4,completed=4 if complete else 1)
    assert value['status']==status and value['launch_eligible']==(status=='PASS') and not value['d2_launched']


def test_resume_authentication_and_attempt_history(tmp_path):
    first=Journal(tmp_path,{'version':1});first.put('cell',{'status':'PASS'})
    second=Journal(tmp_path,{'version':1})
    assert second.get('cell')=={'status':'PASS'}
    assert first.attempt+'.json' in second.prior_attempts
    # A crash leaves RUNNING evidence; the next invocation cannot erase it.
    assert json.loads((tmp_path/'attempts'/(first.attempt+'.json')).read_text())['status']=='RUNNING'
    with pytest.raises(p2.InvalidEvidence):Journal(tmp_path,{'version':2})
    entry=second.index['cell'];(tmp_path/entry['file']).write_text('{}')
    with pytest.raises(p2.InvalidEvidence):second.get('cell')


def test_gap_is_authenticated_complete_and_excludes_calibration(p):
    gap=json.loads((ROOT/p['gap_manifest']).read_text())
    assert validate_gap(p,gap)==pytest.approx(.0010687001049518585)
    assert sum(gap['counts'].values())==21334
    bad=deepcopy(gap);bad['records'][0]['r_C']+=.01
    with pytest.raises(p2.InvalidEvidence):validate_gap(p,bad)


def test_bundle_freezes_all_new_code_and_has_no_training_launch():
    import ast,base64,hashlib,zlib
    path=ROOT/'colabs/clip_placement_preflight_drive_one_cell.py'
    if not path.exists():pytest.skip('Builder runs after initial core tests')
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(assignments['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==assignments['BUNDLE_SHA256']
    sources=json.loads(raw);manifest=json.loads(sources['placement_source_manifest.json'])
    for name,sha in manifest.items():
        assert hashlib.sha256(sources[name].encode()).hexdigest()==sha
        assert (ROOT/name).read_text()==sources[name]
    assert "run_clip_placement_preflight.py" in path.read_text()
