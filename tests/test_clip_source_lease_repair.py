import ast
import base64
import hashlib
import json
from pathlib import Path
import threading
import time
import zlib

import pytest

from colabs import clip_source_lease_repair as repair
from src import clip_source_resume_discovery as discovery
from src import clip_source_resume_store as storage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lease(monkeypatch):
    monkeypatch.setattr(discovery, 'lease', discovery.lease)
    return repair.install()


@pytest.mark.parametrize('failure', [FileNotFoundError(), json.JSONDecodeError('partial', '', 0)])
def test_transient_read_retries_without_writing(failure):
    value = dict(owner='same', active=True, heartbeat=1.)
    class Delayed:
        attempts = 0
        def read_text(self):
            self.attempts += 1
            if self.attempts < 3:
                raise failure
            return json.dumps(value)
    path = Delayed()
    assert repair.read_lease(path, delay=0) == value
    assert path.attempts == 3


def test_persistent_missing_active_lease_stops_without_recreating(tmp_path, lease, monkeypatch):
    monkeypatch.setattr(repair.time, 'sleep', lambda _: None)
    with lease(tmp_path, heartbeat_seconds=1000) as check:
        path = tmp_path/'resume_lease.json'
        path.unlink()
        with pytest.raises(FileNotFoundError):
            check()
        assert not path.exists()
    assert not path.exists()


def test_foreign_owner_not_overwritten_on_check_or_exit(tmp_path, lease):
    with lease(tmp_path, heartbeat_seconds=1000) as check:
        foreign = dict(owner='other-runtime', active=True, heartbeat=time.time())
        storage.atomic_json(tmp_path/'resume_lease.json', foreign)
        with pytest.raises(RuntimeError, match='Concurrent'):
            check()
    assert json.loads((tmp_path/'resume_lease.json').read_text()) == foreign


def test_active_other_runtime_prevents_acquisition(tmp_path, lease, monkeypatch):
    path = tmp_path/'resume_lease.json'
    original = dict(owner='other-runtime', active=True, heartbeat=time.time())
    storage.atomic_json(path, original)
    def update_heartbeat(_):
        storage.atomic_json(path, {**original, 'heartbeat': original['heartbeat'] + 1})
    monkeypatch.setattr(repair.time, 'sleep', update_heartbeat)
    with pytest.raises(RuntimeError, match='still active'):
        with lease(tmp_path):
            pytest.fail('Stole an active lease')
    assert json.loads(path.read_text())['owner'] == 'other-runtime'


def test_stale_recovery_and_same_runtime_exclusion(tmp_path, lease):
    storage.atomic_json(tmp_path/'resume_lease.json',
                        dict(owner='disconnected', active=True, heartbeat=0))
    with lease(tmp_path, heartbeat_seconds=1000) as check:
        check()
        with pytest.raises(RuntimeError, match='active process'):
            with lease(tmp_path):
                pytest.fail('Acquired a second local lease')
    assert json.loads((tmp_path/'resume_lease.json').read_text())['active'] is False


def test_check_waits_for_heartbeat_replacement_gap(tmp_path, monkeypatch):
    original = storage.atomic_json
    gap = threading.Event()
    release = threading.Event()
    checked = threading.Event()
    errors = []
    def delayed_write(path, value):
        if path.name == 'resume_lease.json' and threading.current_thread() is not threading.main_thread() and not gap.is_set():
            path.unlink()  # Simulate mounted-Drive missing entry during replace.
            gap.set()
            assert release.wait(3)
        original(path, value)
    monkeypatch.setattr(storage, 'atomic_json', delayed_write)
    monkeypatch.setattr(discovery, 'lease', discovery.lease)
    lease = repair.install()
    with lease(tmp_path, heartbeat_seconds=.01) as check:
        assert gap.wait(3)
        def reader():
            try:
                check()
            except BaseException as error:
                errors.append(error)
            finally:
                checked.set()
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        try:
            assert not checked.wait(.05), 'Check raced the heartbeat write'
        finally:
            release.set()
            reader_thread.join(3)
        assert checked.is_set() and not errors
        check()


def test_delivery_preserves_bundle_and_embeds_repair():
    def values(path):
        return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ['PAYLOAD', 'BUNDLE_SHA256', 'CLEANUP_ONLY', 'RESUME_RUN']}
    issued = values(ROOT/'colabs/clip_source_latest_drive_one_cell.py')
    path = ROOT/'colabs/clip_source_lease_repair_drive_one_cell.py'
    assert values(path) == issued
    raw = zlib.decompress(base64.b64decode(issued['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == issued['BUNDLE_SHA256']
    for name, source in json.loads(raw).items():
        if name != 'source_resume_source_manifest.json':
            assert (ROOT/name).read_text() == source
    literals = {n.value for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for name in ['run_clip_source_lease_repair.py', 'clip_source_lease_repair.py',
                 'run_clip_source_latest.py', 'clip_source_latest_storage.py']:
        source = (ROOT/'colabs'/name).read_text()
        assert source in literals
        compile(source, name, 'exec')
