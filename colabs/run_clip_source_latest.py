"""Apply latest-only storage under the run lease, then launch the issued worker."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runner')
    parser.add_argument('--bundle-id', required=True)
    parser.add_argument('--resume-run')
    parser.add_argument('--cleanup-only', action='store_true')
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    runner = Path(args.runner).resolve()
    repo = runner.parents[1]
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('d1_latest_storage',
        Path(__file__).with_name('clip_source_latest_storage.py'))
    retention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retention)
    # Verify the immutable source overlay before loading any experiment module.
    sources = json.loads((repo/'source_resume_source_manifest.json').read_text())
    for name, digest in sources.items():
        if retention.sha256(repo/name) != digest:
            raise ValueError('Issued source overlay changed: ' + name)
    from colabs.transfer_drive_backup import DRIVE_ROOT, require_drive
    from src.clip_source_resume_discovery import discover, lease
    old = json.loads((repo/'configs/clip_source_resume_reference.json').read_text())
    require_drive(DRIVE_ROOT/'preflight')
    prior = discover(DRIVE_ROOT, args.bundle_id, old['bundle_id'], args.resume_run)
    if prior and (prior[0]/'resume_state').exists():
        with lease(prior[0]) as check:
            check()
            result = retention.prune(prior[0]/'resume_state', prior[1]['contract'])
            retention.record_policy(prior[0]/'resume_state')
            check()
            print('D1_STORAGE_CLEANUP:', prior[0], result,
                  'freed_GiB=', round(result['freed_bytes']/1024**3, 3), flush=True)
    else:
        print('D1_STORAGE_CLEANUP: no existing resume checkpoint directory.', flush=True)
    if args.cleanup_only:
        print('D1_CLEANUP_ONLY_COMPLETE: no training started.', flush=True)
        return
    retention.install()
    sys.argv = [str(runner), '--bundle-id', args.bundle_id]
    if args.resume_run:
        sys.argv += ['--resume-run', args.resume_run]
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
