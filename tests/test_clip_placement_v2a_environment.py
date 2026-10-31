from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.clip_placement_preflight import InvalidEvidence
from src.clip_placement_v2a_environment import validate_environment

ROOT=Path(__file__).resolve().parents[1]


def inputs():
    original=json.loads((ROOT/'configs/clip_placement_v2a_reference/run_config.json').read_text())['environment']
    spec=json.loads((ROOT/'configs/clip_placement_v2a_py31316.json').read_text())
    policy=spec['environment_revision']
    observed=deepcopy(original)
    observed.update(python=policy['required_python'],python_version=policy['required_python_version'],
                    host_ram_bytes=24*1024**3)
    return original,policy,observed


def test_only_explicit_python_build_changes():
    original,policy,observed=inputs()
    validate_environment(observed,original,policy)
    assert original['python'].startswith('3.13.15 ')
    parent=ROOT/'configs/clip_placement_v2a.json'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==policy['parent_specification_sha256']
    spec=json.loads((ROOT/'configs/clip_placement_v2a_py31316.json').read_text())
    spec.pop('environment_revision');spec['experiment']='clip_placement_v2a'
    assert spec==json.loads(parent.read_text())


@pytest.mark.parametrize('key,value',[
    ('python','3.13.17'),('python_version',[3,13,17]),('cuda','13.0'),
    ('cudnn',99999),('gpu','Different GPU'),('cublas_workspace',None),
    ('packages',{'torch':'different'}),('precision','different'),
])
def test_other_mismatches_stay_invalid_and_report_actual(key,value):
    original,policy,observed=inputs();observed[key]=value
    with pytest.raises(InvalidEvidence) as error:validate_environment(observed,original,policy)
    assert key in str(error.value) and 'expected' in str(error.value) and 'observed' in str(error.value)


def test_low_memory_is_operational():
    original,policy,observed=inputs();observed['host_ram_bytes']=1024
    with pytest.raises(RuntimeError,match='observed 1024'):
        validate_environment(observed,original,policy)
