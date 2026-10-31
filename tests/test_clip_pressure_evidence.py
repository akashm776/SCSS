"""CPU evidence integrity checks; no model loading or new training."""
import json
import pytest
from scripts import archive_clip_pressure_evidence as audit


@pytest.fixture
def files():
    folder=audit.ROOT/'experiment_results'/audit.NAME
    names=json.loads((folder/'export_audit.json').read_text())['files']
    return {n:(folder/n).read_bytes() for n in names}


def test_retained_evidence_and_diagnostics(files):
    assert audit.validate(files,full=False)==0
    derived=audit.diagnostics(files)
    expected=json.loads((audit.ROOT/'experiment_results'/audit.NAME/'diagnostics.json').read_text())
    assert derived==expected
    assert derived['Q_signs']['high']['w500']['negative']==6
    assert derived['Q_signs']['low']['w500']['negative']==3
    assert derived['effect_signs']['w500']['semihard_low']['negative']==1
    assert derived['primary_seed_signs']['negative']==3
    assert derived['calibration_max_relative_error']<.002


@pytest.mark.parametrize('mutation,error',[
    ('source','Changed source'),('summary','Pressure interaction'),
    ('calibration','Calibration coefficient'),('verification','Verified mean pressure'),
    ('coefficient','Coefficient differs'),('pressure','Pressure ratio'),
    ('clipping','Clipping scale'),('replay','Historical B replay'),('plan','Calibration plan changed')])
def test_rejects_inconsistent_evidence(files,mutation,error):
    put=lambda n,v:files.__setitem__(n,json.dumps(v).encode())
    if mutation=='source':files['source_overlay/src/clip_pressure.py']+=b'\n'
    elif mutation=='summary':
        v=json.loads(files['comparison.json'])
        v['timecourse']['50']['contrasts']['late_construction_x_pressure']['loss']*=-1
        put('comparison.json',v)
    elif mutation=='plan':
        v=json.loads(files['calibration_plan.json']);v['seeds']['271828']['batches'][0][0]+=1
        put('calibration_plan.json',v)
    elif mutation in ['calibration','verification']:
        prefix='seed_271828/';v=json.loads(files[prefix+'calibration.json'])
        record=v['weights']['100']
        field='coefficients' if mutation=='calibration' else 'mean_verified_ratios'
        record[field]['hard_low']*=2
        v['calibration_sha256']=audit.canonical_sha(v['weights'])
        put(prefix+'calibration.json',v);put(prefix+'calibration_w100.json',record)
    else:
        prefix='seed_271828/stream_2026100101/';v=json.loads(files[prefix+'result.json'])
        cell=v['cells']['w500'];arm='semihard_high'
        if mutation=='coefficient':cell['coefficients'][arm]*=2
        elif mutation=='pressure':
            probe=cell['pressure_probes'][arm]['10'];probe['ratio']+=1
            put(prefix+f'pressure_w500_{arm}_010.json',probe)
        elif mutation=='clipping':cell['traces'][arm][0]['clipping_scale']*=2
        else:
            arm='historical_semihard';cell['points'][arm]['50']['state_sha256']='wrong'
            put(prefix+f'point_w500_{arm}_050.json',cell['points'][arm]['50'])
        put(prefix+'result.json',v)
        put(prefix+f'branch_w500_{arm}.json',dict(points=cell['points'][arm],traces=cell['traces'][arm],pressure_probes=cell['pressure_probes'][arm]))
    with pytest.raises(ValueError,match=error):audit.validate(files,full=False)
