import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zlib

import pytest
import torch

from colabs import clip_source_latest_storage as latest
from src import clip_source_resume_store as storage
from src import clip_source_resume_training as rt
from src import clip_source_regime as original
from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from tests.test_clip_source_regime import toy, calibration

KEY = 'seed_271828_stream_2026100101'
TRUNK = 'seed_271828_trunk'
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate():
    before = deepcopy(d.rng_state())
    backend = g.backend_state()
    yield
    g.restore_rng(before)
    g.restore_backend(backend)


def populate(root):
    store = storage.Checkpoints(root, {'bundle': 'same'})
    for key in [TRUNK, KEY]:
        store.save(key, {'x': torch.zeros(3)})
        store.save(key, {'x': torch.ones(3)})
    return store


def test_cleanup_keeps_latest_each_phase_and_results(tmp_path):
    root = tmp_path/'resume_state'
    store = populate(root)
    (tmp_path/'result.json').write_text('results stay')
    (root/'unrelated.pt').write_bytes(b'untouched')
    (root/'.upload-abandoned').write_bytes(b'incomplete')
    (root/(KEY + '_' + 'a'*32 + '.pt')).write_bytes(b'orphan')
    heads = {p.stem: json.loads(p.read_text())['generations'][0]
             for p in root.glob('seed_*.json')}
    result = latest.prune(root, store.contract)
    assert result['removed_files'] == 4 and result['freed_bytes'] > 0
    assert result['retained_phases'] == 2
    for key, head in heads.items():
        assert json.loads((root/(key+'.json')).read_text())['generations'] == [head]
        assert torch.equal(store.load(key)['x'], torch.ones(3))
    assert (tmp_path/'result.json').read_text() == 'results stay'
    assert (root/'unrelated.pt').read_bytes() == b'untouched'
    assert latest.prune(root, store.contract)['removed_files'] == 0


@pytest.mark.parametrize('bad', ['corrupt', 'contract', 'path'])
def test_invalid_latest_prevents_all_deletion(tmp_path, bad):
    store = populate(tmp_path)
    pointer_path = tmp_path/(TRUNK+'.json')
    pointer = json.loads(pointer_path.read_text())
    if bad == 'corrupt':
        (tmp_path/pointer['generations'][0]['file']).write_bytes(b'corrupt')
    elif bad == 'contract':
        pointer['contract'] = {'different': 1}
        pointer_path.write_text(json.dumps(pointer))
    else:
        pointer['generations'][0]['file'] = '../escape.pt'
        pointer_path.write_text(json.dumps(pointer))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        latest.prune(tmp_path, store.contract)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_pointer_publication_failure_keeps_old_blobs(tmp_path, monkeypatch):
    store = populate(tmp_path)
    blobs = {p.name: p.read_bytes() for p in tmp_path.glob('*.pt')}
    def fail(*args):
        raise OSError('pointer publication interrupted')
    monkeypatch.setattr(latest, 'atomic_json', fail)
    with pytest.raises(OSError):
        latest.prune(tmp_path, store.contract)
    assert {p.name: p.read_bytes() for p in tmp_path.glob('*.pt')} == blobs
    assert torch.equal(store.load(KEY)['x'], torch.ones(3))


def test_latest_save_upload_failure_and_interrupted_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'Checkpoints', storage.Checkpoints)
    cls = latest.install()
    store = cls(tmp_path, {})
    store.save(KEY, {'x': torch.zeros(3)})
    original_copy = storage.copy_verified
    def fail(*args, **kwargs):
        raise OSError('upload failed')
    monkeypatch.setattr(storage, 'copy_verified', fail)
    with pytest.raises(OSError):
        store.save(KEY, {'x': torch.ones(3)})
    assert torch.equal(store.load(KEY)['x'], torch.zeros(3))
    monkeypatch.setattr(storage, 'copy_verified', original_copy)
    original_prune = latest.prune
    monkeypatch.setattr(latest, 'prune', fail)
    with pytest.raises(OSError):
        store.save(KEY, {'x': torch.ones(3)})
    # New state already committed; interrupted retention cannot lose it.
    assert torch.equal(store.load(KEY)['x'], torch.ones(3))
    monkeypatch.setattr(latest, 'prune', original_prune)
    recovered = cls(tmp_path, {})
    assert len(list(tmp_path.glob('*.pt'))) == 1
    assert torch.equal(recovered.load(KEY)['x'], torch.ones(3))


def test_latest_only_branch_resume_equals_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'Checkpoints', storage.Checkpoints)
    cls = latest.install()
    args = toy()
    model, opt, sched, batch, cfg, states, report, record, p = args
    kwargs = dict(calibrations=calibration(args), p=p, commit=lambda *_: None)
    baseline = original.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs)
    step = rt.step_update
    calls = []
    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 7:
            raise OSError('disconnect')
        return step(*args, **kwargs)
    monkeypatch.setattr(rt, 'step_update', interrupt)
    with pytest.raises(OSError):
        rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs,
                        store=cls(tmp_path,{}),key=KEY,interval=2)
    assert len(list(tmp_path.glob('*.pt'))) == 1
    monkeypatch.setattr(rt, 'step_update', step)
    resumed = rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs,
                             store=cls(tmp_path,{}),key=KEY,interval=2)
    assert resumed == baseline
    assert len(list(tmp_path.glob('*.pt'))) == 1


def test_latest_delivery_keeps_payload_and_embeds_operational_sources():
    def values(path):
        return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ['PAYLOAD', 'BUNDLE_SHA256']}
    issued = values(ROOT/'colabs/clip_source_resume_drive_one_cell.py')
    path = ROOT/'colabs/clip_source_latest_drive_one_cell.py'
    assert values(path) == issued
    raw = zlib.decompress(base64.b64decode(issued['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == issued['BUNDLE_SHA256']
    for name, source in json.loads(raw).items():
        if name != 'source_resume_source_manifest.json':
            assert (ROOT/name).read_text() == source
    literals = {node.value for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for name in ['run_clip_source_latest.py', 'clip_source_latest_storage.py']:
        assert (ROOT/'colabs'/name).read_text() in literals
        compile((ROOT/'colabs'/name).read_text(), name, 'exec')
    assert '*([\'--cleanup-only\'] if CLEANUP_ONLY else [])' in path.read_text()
