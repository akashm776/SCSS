"""Freeze a paired-history analysis before launching the unchanged parent cell."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'colabs/clip_fresh_split_policy_drive_one_cell.py'
FILES = [PARENT, 'configs/clip_paired_history_comparison.json',
    'scripts/analyze_clip_paired_history.py', 'scripts/analyze_clip_image_influence.py',
    'scripts/analyze_clip_policy_robustness.py', 'tests/test_clip_paired_history.py',
    'docs/clip-paired-history-comparison.md']


def build():
    spec = json.loads((ROOT/'configs/clip_paired_history_comparison.json').read_text())
    assert hashlib.sha256((ROOT/PARENT).read_bytes()).hexdigest() == spec['gated_history_cell_sha256']
    raw = json.dumps({n:(ROOT/n).read_text() for n in FILES},sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw,9)).decode()
    cell = '''"""Paired-history follow-up: paste this ENTIRE cell in A100 Colab.
Requires torch 2.11.0+cu128 and torchvision 0.26.0+cu128.
Freezes the analysis on Drive BEFORE launching the unchanged gated-history cell.
3900 additional updates. Does NOT rerun the native-history experiment.
No historical checkpoint ZIP or native-result upload required for this training.
Do not run the old parent cell separately as well. No automatic retries.
"""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zlib

BUNDLE_SHA256=__DIGEST__
PAYLOAD=__PAYLOAD__

def run_all():
    if not Path('/content').is_dir():
        raise RuntimeError('Use an A100 Google Colab runtime.')
    raw=zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest()!=BUNDLE_SHA256:
        raise RuntimeError('Paired-history delivery checksum mismatch')
    sources=json.loads(raw)
    from google.colab import drive
    drive.mount('/content/drive')
    root=Path('/content/drive/MyDrive/OTCO/evaluation_transfer')
    plan=root/'clip_paired_history_comparison_v1_plan'
    plan.mkdir(parents=True,exist_ok=True)
    # Exclusive writes or identical read-back only. Never replace an earlier plan.
    for name,source in sources.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe source path')
        target=plan/'source_overlay'/relative
        data=source.encode()
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(data)
        if target.read_bytes()!=data:
            raise RuntimeError('Existing frozen plan differs: '+str(target))
    marker=plan/'plan_manifest.json'
    if marker.exists():
        if json.loads(marker.read_text())['bundle_id']!=BUNDLE_SHA256:
            raise RuntimeError('Different paired-history analysis already frozen')
    else:
        manifest=dict(experiment='clip_paired_history_comparison_v1',bundle_id=BUNDLE_SHA256,
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            status='analysis_frozen_comparison_pending',selected_after_native_results=True,
            files={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()})
        with marker.open('x') as handle:
            json.dump(manifest,handle,indent=2)
    record=json.loads(marker.read_text())
    if record['files']!={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}:
        raise RuntimeError('Frozen plan inventory differs')
    # Flush the pre-analysis record before training, then let the parent remount.
    drive.flush_and_unmount()
    print('ANALYSIS_PLAN_FLUSHED:',plan,flush=True)
    parent=sources['colabs/clip_fresh_split_policy_drive_one_cell.py']
    spec=json.loads(sources['configs/clip_paired_history_comparison.json'])
    if hashlib.sha256(parent.encode()).hexdigest()!=spec['gated_history_cell_sha256']:
        raise RuntimeError('Preserved training cell changed')
    exec(compile(parent,'preserved_clip_fresh_split_policy_cell.py','exec'),{'__name__':'__main__'})
    print('PAIRED_HISTORY_TRAINING_FINISHED: download clip_fresh_split_policy run and log.')
    print('The paired comparison is still pending CPU analysis with the native-history results.')

run_all()
'''.replace('__DIGEST__',repr(digest)).replace('__PAYLOAD__',repr(payload))
    path = ROOT/'colabs/clip_paired_history_drive_one_cell.py'
    path.write_text(cell)
    print(path,'bundle_sha256=',digest)


if __name__ == '__main__':
    build()
