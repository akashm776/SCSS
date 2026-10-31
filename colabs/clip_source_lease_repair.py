"""Bounded lease-read retries and serialized heartbeat/check operations.

Missing or changed ownership is never repaired in-place during training. Only a
new invocation acquires a new lease through the ordinary recovery procedure.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid


def read_lease(path, *, attempts=6, delay=.2):
    for attempt in range(attempts):
        try:
            value = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            continue
        if not isinstance(value, dict) or not isinstance(value.get('owner'), str):
            raise RuntimeError('Malformed D1 lease ownership record')
        if not isinstance(value.get('active'), bool) or not isinstance(value.get('heartbeat'), (int, float)):
            raise RuntimeError('Malformed D1 lease status record')
        return value


def install():
    from src import clip_source_resume_discovery as discovery
    from src.clip_source_resume_store import atomic_json

    @contextmanager
    def lease(remote, *, stale_seconds=180, heartbeat_seconds=20):
        root = Path(remote)
        root.mkdir(parents=True, exist_ok=True)
        local = Path('/tmp')/('d1-' + root.name + '.lock')
        with local.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('This D1 run already has an active process in this runtime')
            path = root/'resume_lease.json'
            owner = socket.gethostname() + ':' + str(os.getpid()) + ':' + uuid.uuid4().hex
            try:
                previous = read_lease(path)
            except FileNotFoundError:
                previous = None  # New invocation only; never inside a running lease.
            if previous is not None:
                stamp = previous['heartbeat']
                old_owner = previous['owner']
                while previous['active'] and time.time() - stamp < stale_seconds:
                    remaining = stale_seconds - (time.time() - stamp)
                    print('D1_WAITING_FOR_DISCONNECTED_RUN_LEASE:', round(remaining), flush=True)
                    time.sleep(min(20, max(.01, remaining)))
                    current = read_lease(path)
                    if current['owner'] != old_owner or (current['active'] and current['heartbeat'] != stamp):
                        raise RuntimeError('Another D1 runtime is still active; stop it before resuming')
                    previous = current
            atomic_json(path, dict(owner=owner, active=True, heartbeat=time.time()))
            mutex = threading.RLock()
            stop = threading.Event()
            errors = []

            def assert_owner():
                current = read_lease(path)
                if current['owner'] != owner or not current['active']:
                    raise RuntimeError('Concurrent D1 writer detected or lease released')

            def beat():
                while not stop.wait(heartbeat_seconds):
                    try:
                        with mutex:
                            assert_owner()
                            atomic_json(path, dict(owner=owner, active=True, heartbeat=time.time()))
                    except BaseException as error:
                        errors.append(error)
                        return

            def check():
                with mutex:
                    if errors:
                        raise OSError('D1 lease heartbeat failed') from errors[0]
                    assert_owner()

            check()
            atomic_json(root/'resume_lease_policy.json', dict(
                version='d1_lease_serialized_reads_v1', retries=6, retry_delay_seconds=.2,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                missing_active_lease='stop after bounded retries; never recreate',
                owner_change='stop immediately; never overwrite', scientific_bundle_unchanged=True))
            worker = threading.Thread(target=beat, daemon=True)
            worker.start()
            try:
                yield check
            finally:
                stop.set()
                worker.join(timeout=25)
                # Do not race a stalled Drive write or mask the original failure.
                if not worker.is_alive():
                    try:
                        with mutex:
                            current = read_lease(path)
                            if current['owner'] == owner:
                                atomic_json(path, dict(owner=owner, active=False, heartbeat=time.time()))
                    except (OSError, ValueError, RuntimeError) as error:
                        print('D1_LEASE_RELEASE_UNAVAILABLE:', repr(error), flush=True)

    discovery.lease = lease
    return lease
