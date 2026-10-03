"""Build a self-contained new-seed experiment; no historical checkpoint inputs."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_native_history_policy_cell import BASE, FILES as PARENT_FILES

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(set(PARENT_FILES + [
    'configs/clip_optimizer_reset.json', 'src/clip_optimizer_reset.py',
    'colabs/run_clip_optimizer_reset.py', 'tests/test_clip_optimizer_reset.py',
    'docs/clip-optimizer-reset.md', 'configs/clip_optimizer_reset_reference.json',
]))


def build():
    sources = {name: (ROOT/name).read_text() for name in FILES}
    sources['optimizer_reset_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}, indent=2)
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw,9)).decode()
    cell = '''"""ONE A100 Colab cell: retained versus reset AdamW at native states 100 and 500.
3900 total updates: 1500 native-only starting-state + 2400 matched branch updates.
No old checkpoint ZIP needed. Pinned fingerprints verify retained branches against the prior run.
Reset moments and parameter step counters only; never reset the scheduler or weights.
512 report images excluded from all new training/probes; NOT a new external dataset.
Requires torch 2.11.0+cu128 / torchvision 0.26.0+cu128; 15 GiB free runtime disk.
Prefer high-RAM. Starting snapshots remain in RAM; endpoint embeddings saved to Drive.
Paste the ENTIRE file. Wait for DRIVE_SYNC_COMPLETE. No automatic retries/resume.
"""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

BASE_COMMIT = __BASE__
BUNDLE_SHA256 = __DIGEST__
PAYLOAD = __PAYLOAD__

def run_all():
    if not Path('/content').is_dir():
        raise RuntimeError('Run in Google Colab with one A100 GPU.')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Embedded bundle checksum mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    staging = Path(tempfile.mkdtemp(prefix='otco_optimizer_reset_source_',dir='/content'))
    repo = staging/'repo'
    subprocess.run(['git','clone','--no-checkout','https://github.com/akashm776/otco.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',BASE_COMMIT],check=True)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Wrong pinned repository base')
    for name,source in json.loads(raw).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe source overlay path')
        target = repo/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(source,encoding='utf-8')
    spec = importlib.util.spec_from_file_location('otco_optimizer_reset_logging',repo/'colabs/policy_test_logging.py')
    logger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logger)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log = Path('/content/drive/MyDrive/OTCO/evaluation_transfer/clip_optimizer_reset_logs')/(stamp+'.log')
    print('FULL_SUBPROCESS_LOG:',log,flush=True)
    try:
        logger.run_logged([sys.executable,'-u',str(repo/'colabs/run_clip_optimizer_reset.py'),
                           '--bundle-id',BUNDLE_SHA256],repo,log)
    except BaseException:
        print('OPTIMIZER_RESET_FAILED: inspect full log; do not restart automatically.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean the experiment completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:',repr(error))
        raise
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: optimizer-reset results and log saved; Drive now unmounted.')
    print('Download the new clip_optimizer_reset run folder and its log. No large checkpoints written.')

run_all()
'''
    cell = cell.replace('__BASE__',repr(BASE)).replace('__DIGEST__',repr(digest)).replace('__PAYLOAD__',repr(payload))
    target = ROOT/'colabs/clip_optimizer_reset_drive_one_cell.py'
    target.write_text(cell)
    print(target,'bundle_sha256=',digest)
    return cell


if __name__ == '__main__':
    build()
