"""Build a standalone CPU-only Colab delivery; no git clone or model downloads."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
FILES=['scripts/analyze_clip_image_influence.py','scripts/analyze_clip_policy_robustness.py']


def build():
    raw=json.dumps({n:(ROOT/n).read_text() for n in FILES},sort_keys=True).encode()
    digest=hashlib.sha256(raw).hexdigest()
    payload=base64.b64encode(zlib.compress(raw,9)).decode()
    cell='''"""Paste this ENTIRE file in CPU Colab. Zero training/model loading.
Input: completed clip_policy_pool_test_20260930T120952_487061Z folder or its single
ZIP in MyDrive/OTCO/evaluation_transfer or /content. No checkpoint recovery ZIP.
All 48 caches checked; outputs are small JSON/Markdown, not copied embeddings.
Wait for DRIVE_SYNC_COMPLETE. No automatic retry/resume. NumPy is required.
"""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zlib

BUNDLE_SHA256=__DIGEST__
PAYLOAD=__PAYLOAD__

def run_all():
    if not Path('/content').is_dir():
        raise RuntimeError('Use Colab CPU runtime; no GPU needed.')
    import numpy
    from google.colab import drive
    drive.mount('/content/drive')
    drive_root=Path('/content/drive/MyDrive/OTCO/evaluation_transfer')
    name='clip_policy_pool_test_20260930T120952_487061Z'
    roots=[drive_root,Path('/content')]
    folders=[r/name for r in roots if (r/name).is_dir()]
    candidates=folders or sorted({p for r in roots for p in r.glob(name+'*.zip')})
    if len(candidates)!=1:
        raise RuntimeError('Need one complete source folder or single ZIP: '+name)
    raw=zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest()!=BUNDLE_SHA256:
        raise RuntimeError('Embedded code checksum mismatch')
    stage=Path(tempfile.mkdtemp(prefix='otco_image_influence_',dir='/content'))
    for relative,source in json.loads(raw).items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe source overlay')
        target=stage/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(source,encoding='utf-8')
    run_id='clip_image_influence_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=Path('/content')/run_id
    result=subprocess.run([sys.executable,'-u','-m','scripts.analyze_clip_image_influence',
                           str(candidates[0]),'--output',str(output)],cwd=stage,
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout)
    output.mkdir(exist_ok=True)
    (output/'console.log').write_text(result.stdout+'\\nCHILD_EXIT_CODE: '+str(result.returncode)+'\\n')
    shutil.copytree(stage/'scripts',output/'source_overlay'/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    inventory={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in output.rglob('*') if p.is_file()}
    (output/'delivery_manifest.json').write_text(json.dumps(dict(bundle_id=BUNDLE_SHA256,
        status='complete' if result.returncode==0 else 'failed',files=inventory),indent=2))
    destination=drive_root/run_id
    shutil.copytree(output,destination)
    for p in output.rglob('*'):
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()!=hashlib.sha256((destination/p.relative_to(output)).read_bytes()).hexdigest():
            raise RuntimeError('Drive read-back mismatch: '+str(p))
    drive.flush_and_unmount()
    if result.returncode:
        raise RuntimeError('DIAGNOSTIC_FAILED; partial output synced. Inspect console.log; no automatic restart.')
    print('DRIVE_SYNC_COMPLETE:',destination)

run_all()
'''.replace('__DIGEST__',repr(digest)).replace('__PAYLOAD__',repr(payload))
    path=ROOT/'colabs/clip_image_influence_one_cell.py'
    path.write_text(cell)
    print(path,'bundle_sha256=',digest)


if __name__=='__main__':
    build()
