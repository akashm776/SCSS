"""Storage-only amendment: one verified generation per required D1 phase.

Caller must hold the issued run lease. No result files or scientific inputs are
deleted. Keep old state until the replacement and its pointer are verified.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

KEY = r'seed_(?:271828|161803|141421)_(?:trunk|calibration|stream_(?:2026100101|2026100102))'
POINTER = re.compile(KEY + r'\.json')
BLOB = re.compile(KEY + r'_[0-9a-f]{32}\.pt')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix='.retention-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def prune(root, contract, *, key=None):
    """Validate every selected latest blob before changing pointers or deleting.

Full cleanup also removes unreferenced UUID blobs and abandoned upload temporaries
in this dedicated checkpoint directory. Scoped cleanup touches only its phase.
"""
    root = Path(root)
    if not root.exists():
        return dict(removed_files=0, freed_bytes=0, retained_phases=0)
    if root.is_symlink():
        raise ValueError('Refusing a symlink checkpoint directory')
    if key is not None and not re.fullmatch(KEY, key):
        raise ValueError('Unexpected D1 checkpoint key')
    pointers = sorted(p for p in root.iterdir() if POINTER.fullmatch(p.name)
                      and (key is None or p.stem == key))
    selected = []
    keep = set()
    for path in pointers:
        if path.is_symlink():
            raise ValueError('Symlink checkpoint pointer')
        pointer = json.loads(path.read_text())
        if pointer['contract'] != contract or not pointer['generations']:
            raise ValueError('Checkpoint contract mismatch or empty pointer')
        for entry in pointer['generations']:
            name = entry['file']
            if not re.fullmatch(re.escape(path.stem) + r'_[0-9a-f]{32}\.pt', name):
                raise ValueError('Unsafe checkpoint blob name')
        latest = pointer['generations'][0]
        blob = root/latest['file']
        if blob.is_symlink() or not blob.is_file() or sha256(blob) != latest['sha256']:
            raise ValueError('Latest checkpoint missing or corrupt; nothing pruned')
        keep.add(latest['file'])
        selected.append((path, pointer, latest))
    # Publish all reduced pointers first. A failure here deletes no blobs.
    for path, pointer, latest in selected:
        if len(pointer['generations']) != 1:
            atomic_json(path, dict(contract=contract, generations=[latest]))
        if json.loads(path.read_text()) != dict(contract=contract, generations=[latest]):
            raise OSError('Latest-only pointer read-back failed')
    count = freed = 0
    for path in sorted(root.iterdir()):
        owned_blob = BLOB.fullmatch(path.name) and (key is None or path.name.startswith(key + '_'))
        abandoned_upload = key is None and path.name.startswith('.upload-')
        if (owned_blob or abandoned_upload) and path.name not in keep:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Unexpected checkpoint entry type')
            size = path.stat().st_size
            path.unlink()
            freed += size
            count += 1
    return dict(removed_files=count, freed_bytes=freed, retained_phases=len(selected))


def record_policy(root):
    atomic_json(Path(root)/'latest_retention_policy.json', dict(
        version='d1_latest_only_v1', generations_per_phase=1,
        authorization='User requested stopping checkpoint history and continuing from latest state',
        storage_source_sha256=sha256(Path(__file__)),
        scientific_bundle_unchanged=True, results_preserved=True,
        temporary_replacement='Old latest retained until new blob and pointer verify',
        required_native_states='Retain starting states 100 and 500 used by issued runner'))


def install():
    from src import clip_source_resume_store as storage
    original = storage.Checkpoints

    class LatestCheckpoints(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            result = prune(self.root, self.contract)
            record_policy(self.root)
            self.events.append(dict(event='latest_only_cleanup', **result))
            print('D1_LATEST_ONLY:', result, flush=True)

        def save(self, key, payload):
            # The issued transaction remains authoritative. Only its retention
            # changes after successful publication, without touching payloads.
            super().save(key, payload)
            result = prune(self.root, self.contract, key=key)
            self.events.append(dict(event='latest_only_cleanup', key=key, **result))

    storage.Checkpoints = LatestCheckpoints
    return LatestCheckpoints
