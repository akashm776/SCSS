"""Package D1 separately; never rebuild or modify historical bundles."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_geometry_audit_cell import BASE, FILES as PARENT_FILES

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT/'configs/clip_source_regime_freeze_manifest.json').read_text())
FILES = sorted(set(PARENT_FILES + [*FREEZE['files'],
    'configs/clip_source_regime_freeze_manifest.json', 'src/clip_source_regime.py',
    'src/clip_source_analysis.py','src/clip_source_run.py','colabs/run_clip_source_regime.py',
    'tests/test_clip_source_regime.py','docs/clip-source-regime-run.md',
    'scripts/build_clip_source_population.py']))



def build():
    sources = {name: (ROOT/name).read_text() for name in FILES}
    sources['source_regime_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}, indent=2)
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw,9)).decode()
    cell = '''"""ONE high-RAM A100 Colab cell: D1 frozen source-regime experiment.
Nine arms at native weights100/500; frozen k=2 cross-state common masks and source IDs from completed D0.
6900 committed updates:1500 native preparation +5400 branches; 2% initial mean masked-gradient pressure.
No old checkpoint ZIP/D0 folder needed. Native trunks/continuations and processed data must authenticate.
Current embeddings evolve; no online reselection or coefficient adaptation. C/I labels describe initial sources.
512 reporting images excluded from training/calibration. Reused development seeds/pool.
Requires torch2.11.0+cu128, torchvision0.26.0+cu128, one A100>=39 GB, host RAM>=20 GiB, disk>=15 GiB.
Per-update records synchronously backed up; paired virtual AdamW probes restore state. No saved model checkpoints.
Paste the ENTIRE cell. Wait for DRIVE_SYNC_COMPLETE. Inspect partial runs before explicit recovery; no automatic retries.
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
    staging = Path(tempfile.mkdtemp(prefix='otco_source_regime_source_',dir='/content'))
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
    spec = importlib.util.spec_from_file_location('otco_source_regime_logging',repo/'colabs/policy_test_logging.py')
    logger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logger)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log = Path('/content/drive/MyDrive/OTCO/evaluation_transfer/clip_source_regime_logs')/(stamp+'.log')
    print('FULL_SUBPROCESS_LOG:',log,flush=True)
    try:
        logger.run_logged([sys.executable,'-u',str(repo/'colabs/run_clip_source_regime.py'),
                           '--bundle-id',BUNDLE_SHA256],repo,log)
    except BaseException:
        print('D1_FAILED: inspect full log; do not restart automatically.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean the experiment completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:',repr(error))
        raise
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: D1 results and log saved; Drive now unmounted.')
    print('Download the new clip_source_regime run folder and its log. No large checkpoints written.')

run_all()
'''
    cell = cell.replace('__BASE__',repr(BASE)).replace('__DIGEST__',repr(digest)).replace('__PAYLOAD__',repr(payload))
    target = ROOT/'colabs/clip_source_regime_drive_one_cell.py'
    target.write_text(cell)
    print(target,'bundle_sha256=',digest)
    return cell


if __name__ == '__main__':
    build()
