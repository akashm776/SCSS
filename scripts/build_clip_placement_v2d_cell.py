"""Package the full-population audit without rebuilding historical bundles."""

import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_placement_v2c_cell import BASE, FILES as PARENT_FILES

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    set(
        PARENT_FILES
        + [
            "configs/clip_placement_v2d.json",
            "src/clip_placement_v2d.py",
            "src/clip_placement_v2d_run.py",
            "colabs/run_clip_placement_v2d.py",
            "scripts/build_clip_placement_v2d_cell.py",
            "scripts/audit_clip_placement_v2c.py",
            "scripts/audit_clip_placement_v2d.py",
            "placement_v2c_source_manifest.json",
            "tests/test_clip_placement_v2d.py",
            "docs/clip-placement-v2d.md",
        ]
        + [
            str(p.relative_to(ROOT))
            for p in (ROOT / "configs/clip_placement_v2d_reference").rglob("*.json")
        ]
    )
)

TEMPLATE = '''"""Full-population D2 preflight: A100 40GB, high RAM, Python 3.13.16.
624 initial batch/state checks. Frozen v2c coefficients; zero optimizer updates.
Rerun this exact cell to resume. No D2 training is launched, even after PASS.
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
BUNDLE_SHA256 = __SHA__
PAYLOAD = __PAYLOAD__
D1_RUN = '/content/drive/MyDrive/OTCO/evaluation_transfer/clip_source_resume_20261015T014921_451960Z'
OUTPUT = '/content/drive/MyDrive/OTCO/evaluation_transfer/clip_placement_v2d_' + BUNDLE_SHA256[:12]


def run_all():
    if not Path('/content').is_dir():
        raise RuntimeError('Run this cell in A100 high-RAM Colab')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Embedded bundle mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    try:
        repo = Path(tempfile.mkdtemp(prefix='d2_population_', dir='/content')) / 'repo'
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/SCSS.git', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', BASE_COMMIT], check=True)
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != BASE_COMMIT:
            raise RuntimeError('Wrong repository base')
        for name, contents in json.loads(raw).items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe overlay path')
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
        spec = importlib.util.spec_from_file_location('d2_log', repo / 'colabs/policy_test_logging.py')
        logger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logger)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        log = Path(OUTPUT) / 'logs' / (stamp + '.log')
        print('D2P_OUTPUT:', OUTPUT, flush=True)
        print('FULL_SUBPROCESS_LOG:', log, flush=True)
        logger.run_logged([
            sys.executable, '-u', str(repo / 'colabs/bootstrap_clip_source_resume.py'),
            str(repo / 'colabs/run_clip_placement_v2d.py'),
            '--d1-run', D1_RUN, '--output', OUTPUT,
        ], repo, log)
    finally:
        drive.flush_and_unmount()
        print('D2P_DRIVE_FLUSH_COMPLETE. Read acceptance.json; D2 training was not launched.', flush=True)


run_all()
'''


def build():
    sources = {name: (ROOT / name).read_text() for name in FILES}
    manifest = {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in sources.items()
    }
    name = "placement_v2d_source_manifest.json"
    sources[name] = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    (ROOT / name).write_text(sources[name])
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    cell = (
        TEMPLATE.replace("__BASE__", repr(BASE))
        .replace("__SHA__", repr(digest))
        .replace("__PAYLOAD__", repr(payload))
    )
    destination = ROOT / "colabs/clip_placement_v2d_drive_one_cell.py"
    compile(cell, str(destination), "exec")
    destination.write_text(cell)
    print(
        json.dumps(
            dict(
                path=str(destination),
                bundle_sha256=digest,
                files=len(sources),
                bytes=len(cell),
            )
        )
    )
    return cell


if __name__ == "__main__":
    build()
