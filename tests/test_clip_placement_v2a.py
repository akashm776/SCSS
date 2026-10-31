"""Implementation tests only, not a GPU acceptance or experimental gradient audit."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from src import clip_geometry_audit as g
from src import clip_placement_preflight as old
from src import clip_placement_v2a as v2
from tests.test_clip_placement_preflight import features, preserve_test_environment

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def p(): return json.loads((ROOT/'configs/clip_placement_preflight.json').read_text())


@pytest.fixture
def spec(): return json.loads((ROOT/'configs/clip_placement_v2a.json').read_text())


def test_archived_reference_and_old_sources_authenticate(spec):
    config, record=v2.reference(ROOT,spec)
    assert config['delta']==0.0010687001049518585
    assert len(record['context']['eligible_queries'])==34
    assert record['violation']['path']=='Wrong/mixed'


def test_exact_reproduction_and_mismatch_reporting(spec):
    _,record=v2.reference(ROOT,spec)
    original=record['evidence']['placements']
    assert v2.reproduction(original,original,spec)['status']=='PASS'
    changed=deepcopy(original)
    changed['Wrong']['e_aux'][0]+=1e-12
    result=v2.reproduction(changed,original,spec)
    assert result['status']=='INVALID'
    assert result['differences'][0]['path']=='placements/Wrong/e_aux/0'


def test_new_head_is_cast_only_and_restores_precision_settings(p):
    *_,record,out=features()
    # Non-unit live features make unintended renormalization observable.
    out.text_features.mul_(.999)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        shared,_=old.auxiliary_path(out,record,1,.02,p)
        before=g.backend_state()
        native,vals,settings=v2.fp32_head(out,shared)
        assert g.backend_state()==before
    expected=out.text_features.float()@out.image_features.float().T
    assert torch.equal(native['similarity'],expected)
    assert native['similarity'].dtype==torch.float32
    assert not settings['autocast_enabled'] and not settings['matmul_tf32']
    assert settings['matmul_precision']=='highest'
    for arm in old.ARMS:
        assert torch.equal(vals[arm]['z'],shared[arm]['z'].float())
        assert vals[arm]['z0'] is shared[arm]['z0']
        assert vals[arm]['target'] is shared[arm]['target']


def test_native_record_is_complete_symmetric_objective():
    torch.manual_seed(41)
    raw=torch.randn(7,7)
    native=v2.native_record(raw,raw*3)
    assert torch.equal(native['loss_native'],.5*(native['mean_row']+native['mean_col']))
    assert torch.allclose(native['loss_row'],native['A_row']-native['ell_plus'])
    assert torch.allclose(native['loss_col'],native['A_col']-native['ell_plus'])
    assert native['similarity'].shape==(7,7)
    delta=v2.paired_differences(native,native)
    assert all(x['max_absolute']==0 for x in delta.values())


def test_fork_uses_one_carrier_and_never_runs_v2_before_gate(p,spec,monkeypatch):
    *_,record,out=features()
    with torch.no_grad():
        shared,geom=old.auxiliary_path(out,record,1,.02,p)
        archived=dict(evidence=dict(placements=v2.describe(shared,geom,.02)))
        native=v2.native_record(out.raw_similarity,out.logits)
        original_candidate=old.candidate
        calls=[]
        def counted(*args,**kwargs):
            calls.append(1);return original_candidate(*args,**kwargs)
        monkeypatch.setattr(old,'candidate',counted)
        result=v2.paired_from_output(out,record,1,.02,p,spec,archived,native)
        assert result['status']=='PASS' and len(calls)==1
        assert result['evidence']['shared_inputs_unchanged']
        assert all(x['status']=='PASS' for x in result['evidence']['v2a']['contract']['arms'].values())
        archived['evidence']['placements']['Boundary']['p_sim'][0]+=.0001
        def forbidden(*args): raise AssertionError('v2a evaluated before gate passed')
        monkeypatch.setattr(v2,'fp32_head',forbidden)
        result=v2.paired_from_output(out,record,1,.02,p,spec,archived,native)
        assert result['status']=='INVALID' and result['reason']=='V1_REPRODUCTION_FAILED'
        assert 'v2a' not in result['evidence']


def test_all_arm_failures_and_signs_are_retained(p):
    *_,record,out=features()
    with torch.no_grad():
        shared,_=old.auxiliary_path(out,record,1,.02,p)
        _,values,_=v2.fp32_head(out,shared)
        values['Wrong']['g_logit'][0]=1
        values['Correct']['r_train'][1]=-1
        values['Boundary']['r_train'][2]=1
        result=v2.placement_contract(values,.02,p)
    assert result['status']=='FAIL'
    assert all(x['status']=='FAIL' for x in result['arms'].values())
    assert any(x['requirement']=='adjacent_separation' for x in result['failures'])


def test_numerical_nonfinite_is_not_geometry_failure(p):
    *_,record,out=features()
    shared,_=old.auxiliary_path(out,record,1,.02,p)
    shared['Wrong']['r_train'][0]=float('nan')
    with pytest.raises(FloatingPointError):v2.placement_contract(shared,.02,p)


def test_output_guard_cannot_overwrite_v1(tmp_path):
    from colabs.run_clip_placement_v2a import main
    output=tmp_path/'old';output.mkdir()
    original='{"experiment":"clip_placement_preflight_v1"}'
    (output/'run_config.json').write_text(original)
    with pytest.raises(ValueError,match='different experiment'):
        main(tmp_path/'d1',output,inputs_only=True)
    assert (output/'run_config.json').read_text()==original
    assert not (output/'acceptance.json').exists()
