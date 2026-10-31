"""D1 Colab entry point: isolated source overlay, preflight, synchronous Drive backup."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from colabs.backup_clip_run import checksum
from colabs.transfer_drive_backup import DRIVE_ROOT, DriveMirror, require_drive
from colabs.run_clip_pressure import PACKAGES

TEST_FILES=['tests/test_clip_source_regime.py','tests/test_clip_geometry_audit.py',
    'tests/test_clip_pressure.py','tests/test_clip_difficulty.py','tests/test_clip_weight_lr.py',
    'tests/test_clip_optimizer_reset.py','tests/test_clip_native_history_policy.py',
    'tests/test_clip_fresh_split_policy.py','tests/test_clip_policy_pool_test.py','tests/test_clip_policy_test.py',
    'tests/test_clip_continuation_timecourse.py','tests/test_clip_checkpoint_diagnostic.py',
    'tests/test_clip_paired_updates.py','tests/test_clip_gated_training.py','tests/test_clip_transfer_drive_backup.py']


def require_no_previous_run(root,experiment):
    for marker in Path(root).glob('clip_source_regime_*/run_manifest.json'):
        if json.loads(marker.read_text()).get('experiment')==experiment:
            raise RuntimeError(f'Existing D1 run: {marker.parent}. Inspect first; no automatic resume.')


def preflight(bundle_id,p):
    if len(bundle_id)!=64 or any(c not in '0123456789abcdef' for c in bundle_id):raise ValueError('Invalid bundle SHA256')
    require_drive(DRIVE_ROOT/'preflight')
    for root in [Path('/content'),DRIVE_ROOT]:require_no_previous_run(root,p['experiment'])
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('No NVIDIA GPU runtime. In Colab choose Runtime > Change runtime type > A100 GPU and high RAM, then rerun this cell.')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(lines)!=1 or 'A100' not in lines[0] or int(lines[0].split(',')[1].strip())<39000:
        raise RuntimeError(f'One A100 with at least39 GB required; found {lines}')
    mem=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
    if mem<20*1024**3:raise RuntimeError('Select a high-RAM runtime with at least20 GiB host RAM')
    if shutil.disk_usage('/content').free<15*1024**3:raise RuntimeError('15 GiB free runtime disk required')
    return lines[0]


def main(bundle_id):
    os.chdir(ROOT);os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    sources=json.loads((ROOT/'source_regime_source_manifest.json').read_text())
    for relative,expected in sources.items():
        if checksum(ROOT/relative)!=expected:raise AssertionError('Source overlay changed: '+relative)
    p=json.loads((ROOT/'configs/clip_source_regime.json').read_text())
    gpu=preflight(bundle_id,p)
    for name in ['torch','torchvision']:
        if importlib.metadata.version(name)!=PACKAGES[name]:
            raise RuntimeError(f'{name} must be {PACKAGES[name]}; runtime replacement must be explicit.')
    subprocess.run([sys.executable,'-m','pip','install','datasets==2.21.0','transformers==4.57.3',
        'numpy==2.1.3','pyyaml==6.0.3','pytest','matplotlib'],check=True)
    subprocess.run([sys.executable,'-m','pytest','-q',*TEST_FILES],check=True)
    import torch
    from src.clip_source_regime import load_protocol, FREEZE_SHA256
    from src.clip_source_run import run
    from src.clip_checkpoint_diagnostic import write_json
    p=load_protocol()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One usable CUDA GPU required')
    versions={name:importlib.metadata.version(name) for name in PACKAGES}
    if versions!=PACKAGES:raise RuntimeError('Pinned numerical stack differs')
    run_id='clip_source_regime_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=Path('/content')/run_id;output.mkdir(exist_ok=False)
    mirror=DriveMirror(output,DRIVE_ROOT/run_id)
    manifest=dict(run_id=run_id,experiment=p['experiment'],bundle_id=bundle_id,status='running',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        freeze_manifest_sha256=FREEZE_SHA256,gpu=gpu,python=sys.version,packages=versions,
        support_size=2,common_population=True,source_ids_fixed=True,target_ratio=.02,
        starting_state_history='native_only',fresh_adamw=True,external_lr_source=100,
        optimizer_step_budget=6900,report_controls_treatment=False,independent_confirmation=False,
        subprocess_log=os.environ.get('OTCO_POLICY_LOG_PATH'),drive_destination=str(mirror.destination))
    try:
        for relative in sources:
            target=output/'source_overlay'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        write_json(output/'source_manifest.json',sources);write_json(output/'run_manifest.json',manifest)
        mirror.sync()
        print('PERSISTENT_DESTINATION:',mirror.destination,flush=True)
        print('D1: 1500 native preparation + 5400 branch updates. Probes perform paired virtual steps, then fully restore state.',flush=True)
        summary=run(output,mirror.sync,p)
        manifest['status']='complete_pending_drive_flush'
        write_json(output/'run_manifest.json',manifest);mirror.sync(final=True)
    except BaseException as error:
        detail=traceback.format_exc();print(detail,file=sys.stderr,flush=True)
        manifest.update(status='failed_or_sync_incomplete',error=f'{type(error).__name__}: {error}')
        try:
            write_json(output/'failure.json',dict(error=manifest['error'],traceback=detail))
            write_json(output/'run_manifest.json',manifest);mirror.sync()
        except BaseException as backup_error:print('BACKUP_ALSO_FAILED:',repr(backup_error),flush=True)
        raise
    print('D1_PRIMARY:',summary['primary_value'],flush=True)
    print('DRIVE_TREE_VERIFIED:',mirror.destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--bundle-id',required=True)
    main(parser.parse_args().bundle_id)
