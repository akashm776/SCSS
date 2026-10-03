"""New starting states: no archived checkpoint or recovery ZIP is required."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colabs.backup_clip_run import checksum
from colabs.transfer_drive_backup import DRIVE_ROOT, DriveMirror, require_drive

TEST_FILES = ['tests/test_clip_fresh_split_policy.py', 'tests/test_clip_native_history_policy.py', 'tests/test_clip_policy_pool_test.py',
    'tests/test_clip_policy_test.py', 'tests/test_clip_continuation_timecourse.py',
    'tests/test_clip_checkpoint_diagnostic.py', 'tests/test_clip_paired_updates.py',
    'tests/test_clip_gated_training.py', 'tests/test_clip_transfer_drive_backup.py']
PACKAGES = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128',
    'transformers': '4.57.3', 'datasets': '2.21.0', 'numpy': '2.1.3', 'PyYAML': '6.0.3'}


def require_no_previous_run(root, experiment):
    for marker in Path(root).glob('clip_native_history_policy_*/run_manifest.json'):
        if json.loads(marker.read_text()).get('experiment') == experiment:
            raise RuntimeError(f'Existing native-history run: {marker.parent}. Inspect first; no automatic resume.')


def preflight(bundle_id, p):
    if len(bundle_id) != 64 or any(c not in '0123456789abcdef' for c in bundle_id):
        raise ValueError('Expected source-bundle SHA256')
    require_drive(DRIVE_ROOT/'preflight')
    for root in [Path('/content'), DRIVE_ROOT]:
        require_no_previous_run(root, p['experiment'])
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(lines) != 1 or 'A100' not in lines[0] or int(lines[0].split(',')[1].strip()) < 39000:
        raise RuntimeError(f'One A100 with at least 39 GB required; found {lines}')
    if shutil.disk_usage('/content').free < 15*1024**3:
        raise RuntimeError('15 GiB free runtime disk required for model/data caches')
    return lines[0]


def main(bundle_id):
    os.chdir(ROOT)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sources = json.loads((ROOT/'native_history_policy_source_manifest.json').read_text())
    for relative, expected in sources.items():
        if checksum(ROOT/relative) != expected:
            raise AssertionError(f'Source overlay changed: {relative}')
    p = json.loads((ROOT/'configs/clip_native_history_policy.json').read_text())
    gpu = preflight(bundle_id, p)
    # Do not silently replace the runtime CUDA/optimizer stack.
    for name in ['torch', 'torchvision']:
        if importlib.metadata.version(name) != PACKAGES[name]:
            raise RuntimeError(f'{name} must be {PACKAGES[name]}; inspect runtime mismatch before running.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.21.0',
        'transformers==4.57.3', 'numpy==2.1.3', 'pyyaml==6.0.3', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *TEST_FILES], check=True)
    import torch
    from src.clip_native_history_policy import load_protocol, run
    from src.clip_checkpoint_diagnostic import write_json
    p = load_protocol()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one usable CUDA device required')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise RuntimeError(f'Numerical stack changed: {versions}')
    run_id = 'clip_native_history_policy_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path('/content')/run_id
    output.mkdir(exist_ok=False)
    mirror = DriveMirror(output, DRIVE_ROOT/run_id)
    manifest = dict(run_id=run_id, experiment=p['experiment'], bundle_id=bundle_id, status='running',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=gpu, python=sys.version, packages=versions, archived_checkpoints_required=False,
        starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        parent_experiment=p['parent_experiment'],parent_bundle_id=p['parent_bundle_id'],shared_design_with_parent=True,
        new_training_seeds=True, fresh_reporting_split_for_this_run=True,
        historically_unused_images_claimed=False, independent_test_set=False, new_external_dataset=False,
        rules_refitted=False, reports_control_treatment=False, literal_training_loader_resume=False,
        optimizer_step_budget=3900, trunk_optimizer_steps=1500, branch_optimizer_steps=2400,
        exploratory_followup=True, subprocess_log=os.environ.get('OTCO_POLICY_LOG_PATH'),
        drive_destination=str(mirror.destination))
    try:
        for relative in sources:
            target = output/'source_overlay'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
        write_json(output/'source_manifest.json', sources)
        write_json(output/'run_manifest.json', manifest)
        mirror.sync()
        print('PERSISTENT_DESTINATION:', mirror.destination, flush=True)
        print('3 new seeds, 512 newly held-out CUB images; NOT historically untouched or external data.', flush=True)
        print('1500 native-only starting-state updates + 2400 matched continuation updates = 3900.', flush=True)
        summary = run(output, mirror.sync, p)
        manifest['status'] = 'complete_pending_drive_flush'
        write_json(output/'run_manifest.json', manifest)
        mirror.sync(final=True)
    except BaseException as error:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        manifest.update(status='failed_or_sync_incomplete', error=f'{type(error).__name__}: {error}')
        try:
            write_json(output/'failure.json', dict(error=manifest['error'], traceback=detail))
            write_json(output/'run_manifest.json', manifest)
            mirror.sync()
        except BaseException as backup_error:
            print('BACKUP_ALSO_FAILED:', repr(backup_error), flush=True)
        raise
    print('FINAL_COMPARISON:', json.dumps(summary, indent=2), flush=True)
    print('DRIVE_TREE_VERIFIED:', mirror.destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle-id', required=True)
    main(parser.parse_args().bundle_id)
