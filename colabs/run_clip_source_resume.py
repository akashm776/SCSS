"""D1 automatic checkpoint resume, preserving the issued scientific protocol."""
import argparse
from contextlib import ExitStack
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


def preflight(bundle_id,p):
    if len(bundle_id)!=64 or any(c not in '0123456789abcdef' for c in bundle_id):raise ValueError('Invalid bundle SHA256')
    require_drive(DRIVE_ROOT/'preflight')

    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('No NVIDIA GPU runtime. In Colab choose Runtime > Change runtime type > A100 GPU and high RAM, then rerun this cell.')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(lines)!=1 or 'A100' not in lines[0] or int(lines[0].split(',')[1].strip())<39000:
        raise RuntimeError(f'One A100 with at least39 GB required; found {lines}')
    mem=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
    if mem<20*1024**3:raise RuntimeError('Select a high-RAM runtime with at least20 GiB host RAM')
    if shutil.disk_usage('/content').free<20*1024**3:raise RuntimeError('20 GiB free runtime disk required with resume checkpoints')
    return lines[0]


def main(bundle_id,resume_run=None):
    os.chdir(ROOT);os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from src.clip_source_resume_discovery import discover,restore_artifacts,lease
    from src.clip_source_resume_store import Checkpoints,ResumeMismatch
    sources=json.loads((ROOT/'source_resume_source_manifest.json').read_text())
    for relative,expected in sources.items():
        if checksum(ROOT/relative)!=expected:raise AssertionError('Source overlay changed: '+relative)
    old=json.loads((ROOT/'configs/clip_source_resume_reference.json').read_text())
    p=json.loads((ROOT/'configs/clip_source_regime.json').read_text())
    require_drive(DRIVE_ROOT/'preflight')
    prior=discover(DRIVE_ROOT,bundle_id,old['bundle_id'],resume_run)
    is_legacy=prior is not None and prior[1]['bundle_id']==old['bundle_id']
    run_id=(prior[0].name if prior and not is_legacy else
        'clip_source_resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output=Path('/content')/run_id;output.mkdir(exist_ok=True)
    destination=DRIVE_ROOT/run_id
    # Lock the old lineage while copying/migrating too; two runtimes must not fork it.
    lock_root=prior[0] if prior else destination
    with ExitStack() as locks:
        parent_check=locks.enter_context(lease(lock_root))
        child_check=locks.enter_context(lease(destination)) if destination!=lock_root else parent_check
        def check_lease():parent_check();child_check()
        if prior:
            restore_artifacts(prior[0],output,old['sources'] if is_legacy else sources)
            if prior[1].get('status')=='complete_pending_drive_flush' and (output/'completion.json').exists():
                completion=json.loads((output/'completion.json').read_text())
                if completion.get('status')!='complete':raise ResumeMismatch('Inconsistent completion marker')
                print('D1_ALREADY_COMPLETE:',prior[0],flush=True)
                return
        gpu=preflight(bundle_id,p)
        for name in ['torch','torchvision']:
            if importlib.metadata.version(name)!=PACKAGES[name]:
                raise RuntimeError(f'{name} must be {PACKAGES[name]}; runtime replacement must be explicit.')
        subprocess.run([sys.executable,'-m','pip','install','datasets==2.21.0','transformers==4.57.3',
            'numpy==2.1.3','pyyaml==6.0.3','pytest','matplotlib'],check=True)
        subprocess.run([sys.executable,'-m','pytest','-q',*TEST_FILES,'tests/test_clip_source_resume.py'],check=True)
        import torch
        from src.clip_source_regime import load_protocol, FREEZE_SHA256
        from src.clip_source_resume_run import run
        from src.clip_checkpoint_diagnostic import write_json
        p=load_protocol()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One usable CUDA GPU required')
        versions={name:importlib.metadata.version(name) for name in PACKAGES}
        if versions!=PACKAGES:raise RuntimeError('Pinned numerical stack differs')
        mirror=DriveMirror(output,destination)
        def sync():check_lease();return mirror.sync()
        contract=dict(bundle_id=bundle_id,freeze_sha256=FREEZE_SHA256,packages=versions,
            gpu=gpu,python_version=list(sys.version_info[:3]),cuda_version=torch.version.cuda,
            cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            population_sha256=checksum(ROOT/p['population']),schema=1)
        store=Checkpoints(destination/'resume_state',contract,sync=sync,local=output)
        attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        manifest=dict(run_id=run_id,experiment=p['experiment'],bundle_id=bundle_id,status='running',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            freeze_manifest_sha256=FREEZE_SHA256,gpu=gpu,python=sys.version,packages=versions,
            parent_run=prior[0].name if is_legacy else (prior[1].get('parent_run') if prior else None),
            legacy_import=is_legacy or bool(prior and prior[1].get('legacy_import')),
            attempt=attempt,automatic_resume=True,contract=contract,
            source_ids_fixed=True,target_ratio=.02,optimizer_step_budget=6900,
            reporting_controls_treatment=False,subprocess_log=os.environ.get('OTCO_POLICY_LOG_PATH'),
            drive_destination=str(destination))
        try:
            for relative in sources:
                target=output/'source_overlay'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target)
            write_json(output/'source_manifest.json',sources);write_json(output/'run_manifest.json',manifest)
            sync()
            print('PERSISTENT_DESTINATION:',destination,flush=True)
            print('AUTO_RESUME: native checkpoints every25 steps; branch checkpoints every10 steps and branch completion.',flush=True)
            if is_legacy:print('LEGACY_IMPORT: completed branches retained; unfinished branches replay because the old runner saved no state.',flush=True)
            summary=run(output,sync,p,store=store,legacy=manifest['legacy_import'])
            manifest['status']='complete_pending_drive_flush'
            write_json(output/'run_manifest.json',manifest)
            attempts=output/'resume_attempts';attempts.mkdir(exist_ok=True)
            write_json(attempts/(attempt+'.json'),dict(status='complete',events=store.events))
            check_lease();mirror.sync(final=True)
        except BaseException as error:
            detail=traceback.format_exc();print(detail,file=sys.stderr,flush=True)
            manifest.update(status='failed_or_sync_incomplete',error=f'{type(error).__name__}: {error}',
                resume_blocked=isinstance(error,(AssertionError,ValueError)))
            try:
                write_json(output/'failure.json',dict(error=manifest['error'],traceback=detail))
                write_json(output/'run_manifest.json',manifest)
                attempts=output/'resume_attempts';attempts.mkdir(exist_ok=True)
                write_json(attempts/(attempt+'.json'),dict(status='interrupted_or_failed',events=store.events,error=manifest['error']))
                sync()
            except BaseException as backup_error:print('BACKUP_ALSO_FAILED:',repr(backup_error),flush=True)
            raise
        print('D1_PRIMARY:',summary['primary_value'],flush=True)
        print('DRIVE_TREE_VERIFIED:',destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--bundle-id',required=True)
    parser.add_argument('--resume-run',default=None)
    args=parser.parse_args();main(args.bundle_id,args.resume_run)
