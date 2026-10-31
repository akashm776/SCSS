"""Regression checks for the CPU evidence audit; no training or cache loading."""
import json

import pytest

from scripts import archive_clip_weight_lr_evidence as audit


@pytest.fixture
def files():
    folder = audit.ROOT/'experiment_results'/audit.NAME
    inventory = json.loads((folder/'export_audit.json').read_text())['files']
    return {name: (folder/name).read_bytes() for name in inventory}


def test_retained_evidence_and_arithmetic(files):
    assert audit.validate(files, full=False) == 0


def test_crossed_interaction_convention():
    result = audit.contrasts(dict(zip(audit.CELLS, [-2., -1., 3., 5.])))
    assert result['weight_main_effect'] == 5.5
    assert result['lr_main_effect'] == 1.5
    assert result['weight_x_lr_interaction'] == 1.


@pytest.mark.parametrize('mutation,expected', [
    ('summary', 'Factorial contrast'), ('rng', 'Unmatched RNG/LR'),
    ('reset', 'Invalid optimizer reset'), ('source', 'Changed source'),
])
def test_rejects_inconsistent_evidence(files, mutation, expected):
    if mutation == 'source':
        files['source_overlay/src/clip_weight_lr.py'] += b'\n'
    elif mutation == 'summary':
        name = 'comparison.json'
        value = json.loads(files[name])
        value['timecourse']['50']['contrasts']['loss']['weight_x_lr_interaction'] *= -1
        files[name] = json.dumps(value).encode()
    else:
        name = 'seed_271828/stream_2026100101/result.json'
        row = json.loads(files[name])
        cell = row['cells']['w500_lr500']
        if mutation == 'rng':
            cell['traces']['sustained'][3]['rng_after_sha256'] = 'changed'
        else:
            cell['resets']['sustained']['new_parameter_states'] = 35
            start_name = name.removesuffix('result.json')+'start_w500_lr500_sustained.json'
            start = json.loads(files[start_name])
            start['reset'] = cell['resets']['sustained']
            files[start_name] = json.dumps(start).encode()
        files[name] = json.dumps(row).encode()
    with pytest.raises(ValueError, match=expected):
        audit.validate(files, full=False)
