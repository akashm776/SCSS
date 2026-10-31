"""Explicit Python patch/build revision; never alter frozen v1 environment checks."""
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from src.clip_placement_preflight import InvalidEvidence


def capture_environment(original):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('v2a requires one A100 CUDA GPU; no CPU fallback')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total',
                                 '--format=csv,noheader,nounits'],text=True).strip()
    ram=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
             if x.startswith('MemTotal:'))
    return dict(packages={n:importlib.metadata.version(n) for n in original['packages']},
        python=sys.version,python_version=list(sys.version_info[:3]),gpu=gpu,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        precision='D1 bf16, no GradScaler',host_ram_bytes=ram)


def validate_environment(observed, original, revision):
    expected=deepcopy(original)
    expected['python']=revision['required_python']
    expected['python_version']=revision['required_python_version']
    actual={k:v for k,v in observed.items() if k!='host_ram_bytes'}
    differences={k:dict(expected=expected.get(k),observed=actual.get(k))
                 for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k)}
    if differences:
        raise InvalidEvidence('Runtime environment mismatch: '+json.dumps(differences,sort_keys=True))
    if observed.get('host_ram_bytes',0)<20*1024**3:
        raise RuntimeError('High-RAM runtime with >=20 GiB host memory required; observed '+
                           str(observed.get('host_ram_bytes')))
