"""Package D2-P separately. Never overwrite any issued D0/D1 bundle."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from scripts.build_clip_source_resume_cell import BASE, FILES as PARENT_FILES

ROOT=Path(__file__).resolve().parents[1]
FILES=sorted(set(PARENT_FILES+[
 'configs/clip_placement_preflight.json','configs/clip_placement_gap.json','configs/clip_placement_inputs.json',
 'src/clip_placement_preflight.py','src/clip_placement_evidence.py','src/clip_placement_preflight_run.py',
 'scripts/build_clip_placement_gap.py','scripts/build_clip_source_population.py',
 'scripts/build_clip_placement_preflight_cell.py',
 'colabs/clip_source_lease_repair.py','colabs/run_clip_placement_preflight.py','colabs/bootstrap_clip_source_resume.py',
 'tests/test_clip_placement_preflight.py','docs/clip-placement-preflight.md']))


def build():
    sources={n:(ROOT/n).read_text() for n in FILES}
    manifest={n:hashlib.sha256(v.encode()).hexdigest() for n,v in sources.items()}
    sources['placement_source_manifest.json']=json.dumps(manifest,sort_keys=True,indent=2)+'\n'
    (ROOT/'placement_source_manifest.json').write_text(sources['placement_source_manifest.json'])
    raw=json.dumps(sources,sort_keys=True).encode();sha=hashlib.sha256(raw).hexdigest()
    payload=base64.b64encode(zlib.compress(raw,9)).decode()
    cell='''"""D2-P v1 only: A100 40GB, high RAM, pinned D1 stack. ZERO optimizer updates.
Reads completed D1 checkpoints; writes small preflight JSON records, no new models.
Rerun the same cell to resume authenticated checks after an interruption.
FAIL stops at the first demonstrated violation; no gap/mask/precision adjustment.
Even PASS NEVER launches D2. Review acceptance.json first.
"""
import base64, hashlib, importlib.util, json, subprocess, sys, tempfile, zlib
from pathlib import Path
from datetime import datetime, timezone
BASE_COMMIT=__BASE__
BUNDLE_SHA256=__SHA__
PAYLOAD=__PAYLOAD__
D1_RUN='/content/drive/MyDrive/OTCO/evaluation_transfer/clip_source_resume_20261015T014921_451960Z'
OUTPUT='/content/drive/MyDrive/OTCO/evaluation_transfer/clip_placement_preflight_v1_'+BUNDLE_SHA256[:12]

def run_all():
    if not Path('/content').is_dir():raise RuntimeError('Run this cell in A100 high-RAM Colab')
    raw=zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest()!=BUNDLE_SHA256:raise RuntimeError('Embedded bundle mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    repo=Path(tempfile.mkdtemp(prefix='d2p_',dir='/content'))/'repo'
    subprocess.run(['git','clone','--no-checkout','https://github.com/akashm776/otco.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',BASE_COMMIT],check=True)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=BASE_COMMIT:raise RuntimeError('Wrong base')
    for name,value in json.loads(raw).items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe overlay path')
        target=repo/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(value)
    spec=importlib.util.spec_from_file_location('d2p_log',repo/'colabs/policy_test_logging.py')
    logger=importlib.util.module_from_spec(spec);spec.loader.exec_module(logger)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log=Path(OUTPUT)/'logs'/(stamp+'.log')
    print('D2P_OUTPUT:',OUTPUT,flush=True)
    print('FULL_SUBPROCESS_LOG:',log,flush=True)
    try:
        logger.run_logged([sys.executable,'-u',str(repo/'colabs/bootstrap_clip_source_resume.py'),
            str(repo/'colabs/run_clip_placement_preflight.py'),'--d1-run',D1_RUN,'--output',OUTPUT],repo,log)
    finally:
        drive.flush_and_unmount()
        print('D2P_DRIVE_FLUSH_COMPLETE. Read acceptance.json; D2 was NOT launched.',flush=True)

run_all()
'''
    cell=cell.replace('__BASE__',repr(BASE)).replace('__SHA__',repr(sha)).replace('__PAYLOAD__',repr(payload))
    path=ROOT/'colabs/clip_placement_preflight_drive_one_cell.py';path.write_text(cell)
    compile(cell,str(path),'exec')
    print(json.dumps(dict(path=str(path),bundle_sha256=sha,files=len(sources),bytes=len(cell))))
    return cell


if __name__=='__main__':build()
