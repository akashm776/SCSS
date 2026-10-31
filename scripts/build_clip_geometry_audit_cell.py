"""Build Experiment D0 as a separate source overlay; preserve the frozen protocol and prior bundles."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_pressure_cell import BASE, FILES as PARENT_FILES

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads((ROOT/'configs/clip_geometry_audit_freeze_manifest.json').read_text())
FILES = sorted(set(PARENT_FILES + [
    'src/clip_geometry_audit.py','src/clip_geometry_summary.py','src/clip_geometry_run.py',
    'colabs/run_clip_geometry_audit.py','tests/test_clip_geometry_audit.py',
    'docs/clip-geometry-audit-run.md','configs/clip_geometry_audit_freeze_manifest.json',
    *FREEZE['files'],*FREEZE['parent_evidence_files'],*FREEZE['reference_source_files'],
]))


def build():
    sources = {name: (ROOT/name).read_text() for name in FILES}
    sources['geometry_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}, indent=2)
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw,9)).decode()
    cell = '''"""ONE high-RAM A100 Colab cell: D0 geometry/feasibility audit, no treatment outcomes.
624 fixed-state batch forwards; 39,936 query exposures. Zero audit optimizer/backward steps.
Regenerating native checkpoints costs up to 1500 native preparation updates separately.
No old checkpoint ZIP required. Native trunks and both states must match pinned fingerprints.
No D1 continuations, auxiliary losses, gradient calibration, virtual AdamW probes, or reporting evaluation.
Requires torch 2.11.0+cu128 / torchvision 0.26.0+cu128, 20 GiB host RAM and 15 GiB free disk.
Writes caption-free compressed query records and summaries; no model checkpoints.
Paste the ENTIRE cell. Wait for DRIVE_SYNC_COMPLETE. No automatic retries or resume.
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
    staging = Path(tempfile.mkdtemp(prefix='otco_geometry_source_',dir='/content'))
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
    spec = importlib.util.spec_from_file_location('otco_geometry_logging',repo/'colabs/policy_test_logging.py')
    logger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logger)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log = Path('/content/drive/MyDrive/OTCO/evaluation_transfer/clip_geometry_audit_logs')/(stamp+'.log')
    print('FULL_SUBPROCESS_LOG:',log,flush=True)
    try:
        logger.run_logged([sys.executable,'-u',str(repo/'colabs/run_clip_geometry_audit.py'),
                           '--bundle-id',BUNDLE_SHA256],repo,log)
    except BaseException:
        print('D0_FAILED: inspect full log; do not restart automatically.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean the experiment completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:',repr(error))
        raise
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: D0 results and log saved; Drive now unmounted.')
    print('Download the new clip_geometry_audit run folder and its log. No large checkpoints written.')

run_all()
'''
    cell = cell.replace('__BASE__',repr(BASE)).replace('__DIGEST__',repr(digest)).replace('__PAYLOAD__',repr(payload))
    target = ROOT/'colabs/clip_geometry_audit_drive_one_cell.py'
    target.write_text(cell)
    print(target,'bundle_sha256=',digest)
    return cell


if __name__ == '__main__':
    build()
