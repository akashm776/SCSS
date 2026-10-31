"""CPU checks for the downloaded evidence audit; no training or model loading."""
import json

import pytest

from scripts import archive_clip_difficulty_evidence as audit


@pytest.fixture
def files():
    folder = audit.ROOT/'experiment_results'/audit.NAME
    names = json.loads((folder/'export_audit.json').read_text())['files']
    return {n: (folder/n).read_bytes() for n in names}


def test_public_replay_arithmetic_and_derived_diagnostics(files):
    assert audit.validate(files, full=False) == 0
    expected = json.loads((audit.ROOT/'experiment_results'/audit.NAME/'diagnostics.json').read_text())
    assert audit.diagnostics(files) == expected
    assert expected['effect_signs']['w500']['synthetic_semihard']['positive'] == 6
    assert expected['difference_signs']['synthetic']['w100']['negative'] == 4
    assert expected['interaction_signs']['synthetic']['negative'] == 6


@pytest.mark.parametrize('mutation,error', [('source','Changed source'), ('summary','Interaction'),
    ('pressure','Pressure ratio'), ('overlap','support separation'), ('replay','Fixed-LR replay')])
def test_rejects_inconsistent_records(files, mutation, error):
    put = lambda n, v: files.__setitem__(n, json.dumps(v).encode())
    if mutation == 'source':
        files['source_overlay/src/clip_difficulty.py'] += b'\n'
    elif mutation == 'summary':
        value = json.loads(files['comparison.json'])
        value['timecourse']['50']['comparisons']['synthetic']['late_minus_early']['loss'] *= -1
        put('comparison.json', value)
    else:
        prefix = 'seed_271828/stream_2026100101/'
        row = json.loads(files[prefix+'result.json'])
        cell = row['cells']['w500']
        arm = 'synthetic_semihard'
        if mutation == 'pressure':
            cell['pressure_probes'][arm]['10']['ratio'] += 1
            put(prefix+f'pressure_w500_{arm}_010.json', cell['pressure_probes'][arm]['10'])
        elif mutation == 'overlap':
            cell['traces'][arm][1]['selection']['overlap_count'] = 1
        else:
            arm = 'synthetic_hard'
            cell['points'][arm]['50']['state_sha256'] = 'wrong'
            put(prefix+f'point_w500_{arm}_050.json', cell['points'][arm]['50'])
        put(prefix+'result.json',row)
        put(prefix+f'branch_w500_{arm}.json', dict(points=cell['points'][arm], traces=cell['traces'][arm],
                                                 pressure_probes=cell['pressure_probes'][arm]))
    with pytest.raises(ValueError, match=error):
        audit.validate(files, full=False)
