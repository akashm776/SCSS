"""Synthetic audit tests do not stand in for the missing downloaded GPU record."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest

from scripts.audit_clip_placement_v2a import numerical_audit, read_input, stats


def fixture():
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/'configs/clip_placement_preflight.json').read_text())
    delta=.0010687001049518585
    placements={}
    ids=list(range(34))
    for arm,sign in [('Wrong',-1),('Boundary',0),('Correct',1)]:
        placements[arm]=dict(r_geom=[sign*delta]*34,r_train=[sign*delta]*34,
            g_logit=[100*sign*delta]*34,m_loss=[-2.]*34,p_sim=[.3]*34,
            ell_plus=[30.]*34,s_aux=[.3-sign*delta]*34,e_aux=[30-100*sign*delta]*34,
            A=[32.]*34,tau=100.,delta=delta,target_margin=sign*delta)
    native=dict(similarity=[[0.]*64 for _ in range(64)],logits=[[0.]*64 for _ in range(64)],
        p_sim=[0.]*64,ell_plus=[0.]*64,A_row=[0.]*64,A_col=[0.]*64,
        loss_row=[0.]*64,loss_col=[0.]*64,loss_native=0.)
    evidence=dict(v1=dict(native=native),v2a=dict(native=deepcopy(native),placements=placements))
    return evidence,p,delta,ids


def test_recomputed_margins_and_native_loss_changes():
    evidence,p,delta,ids=fixture()
    evidence['v2a']['native']['loss_native']=.0004
    result=numerical_audit(evidence,p,delta,ids)
    assert result['native_loss_change']==.0004
    assert result['arms']['Wrong']['maximum_logit_error']==0.
    assert result['spacing']['Wrong/Boundary']['minimum']==delta


def test_audit_rejects_actual_violation_regardless_of_recorded_status():
    evidence,p,delta,ids=fixture()
    evidence['v2a']['contract']={'status':'PASS'}
    evidence['v2a']['placements']['Correct']['g_logit'][0]=-1.
    with pytest.raises(ValueError,match='violate v2a contracts'):
        numerical_audit(evidence,p,delta,ids)


def test_zip_and_folder_reader_preserve_bytes_and_reject_traversal(tmp_path):
    folder=tmp_path/'input';folder.mkdir()
    data=b'{"status":"PASS"}\n'
    (folder/'acceptance.json').write_bytes(data)
    archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('input/acceptance.json',data)
    for path in [folder,archive]:
        read=read_input(path)
        assert read('acceptance.json')==data
        with pytest.raises(ValueError,match='Unsafe'):read('../escape')


def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):stats([float('nan')])
