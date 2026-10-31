"""Find one D1 lineage, authenticate artifacts, and prevent overlapping writers."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from colabs.backup_clip_run import checksum,copy_verified
from src.clip_source_resume_store import atomic_json,ResumeMismatch


def discover(root,bundle,legacy_bundle,requested=None):
    root=Path(root)
    if requested:
        path=root/requested
        if path.parent!=root or not (path/'run_manifest.json').is_file():raise ResumeMismatch('Unknown resume run')
        candidates=[path]
    else:
        candidates=[p.parent for pattern in ['clip_source_resume_*/run_manifest.json','clip_source_regime_*/run_manifest.json']
                    for p in root.glob(pattern)]
    compatible=[]
    incompatible=[]
    parents=set()
    for path in candidates:
        m=json.loads((path/'run_manifest.json').read_text())
        if m.get('experiment')!='clip_source_regime_v1':continue
        if m.get('parent_run'):parents.add(m['parent_run'])
        if m.get('bundle_id') in [bundle,legacy_bundle]:compatible.append((path,m))
        else:incompatible.append(path)
    incompatible=[p for p in incompatible if p.name not in parents]
    if incompatible:raise ResumeMismatch('Existing D1 run uses a different source bundle: '+', '.join(p.name for p in incompatible))
    compatible=[(p,m) for p,m in compatible if p.name not in parents]
    if len(compatible)>1:
        raise ResumeMismatch('Multiple D1 lineages; set RESUME_RUN to one: '+', '.join(p.name for p,_ in compatible))
    if not compatible:return None
    path,m=compatible[0]
    if m.get('resume_blocked'):raise ResumeMismatch('Previous scientific/integrity failure must be investigated: '+str(m.get('error')))
    if m.get('status')=='failed_or_sync_incomplete' and str(m.get('error','')).startswith(('AssertionError:','ValueError:','AuditFailure:','ResumeMismatch:')):
        raise ResumeMismatch('Prior scientific/integrity failure blocks automatic retry: '+m['error'])
    return path,m


def restore_artifacts(remote,local,expected_sources):
    """Only committed, checksum-authenticated result files are reused.

    Mutable transport status may be newer than its last inventory. Scientific
    artifacts never receive this exemption. No arbitrary Torch file is loaded here.
    """
    remote,local=Path(remote),Path(local)
    if not (remote/'DRIVE_BACKUP_MANIFEST.json').exists():
        # A process may die between creating the run marker and its first mirror.
        scientific=list(remote.glob('seed_*'))
        if scientific:raise ResumeMismatch('Missing inventory for existing scientific artifacts')
        return
    inventory=json.loads((remote/'DRIVE_BACKUP_MANIFEST.json').read_text())['files']
    if 'source_manifest.json' not in inventory:raise ResumeMismatch('Missing authenticated source manifest')
    for name,entry in inventory.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ResumeMismatch('Unsafe inventory path')
        if name in ['run_manifest.json','failure.json','resume_execution.json']:continue
        src=remote/path
        if not src.is_file() or checksum(src)!=entry['sha256']:
            if path.name=='progress.json':continue  # Mutable preparation progress is not a recovery checkpoint.
            raise ResumeMismatch('Saved artifact missing/corrupt: '+name)
        copy_verified(src,local/path)
    actual=json.loads((local/'source_manifest.json').read_text())
    if actual!=expected_sources:raise ResumeMismatch('Prior source manifest differs')
    for name,sha in expected_sources.items():
        path=local/'source_overlay'/name
        if not path.is_file() or checksum(path)!=sha:raise ResumeMismatch('Prior source overlay differs: '+name)


@contextmanager
def lease(remote,*,stale_seconds=180,heartbeat_seconds=20):
    """Single mounted-Drive writer; stale leases expire after a disconnected runtime.

    A local flock also excludes simultaneous invocations in the same runtime.
    Fresh remote leases are waited out, never stolen; an active heartbeat aborts.
    """
    root=Path(remote);root.mkdir(parents=True,exist_ok=True)
    local=Path('/tmp')/('d1-'+root.name+'.lock')
    with local.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This D1 run already has an active process in this runtime')
        path=root/'resume_lease.json';owner=socket.gethostname()+':'+str(os.getpid())+':'+uuid.uuid4().hex
        if path.exists():
            previous=json.loads(path.read_text());stamp=previous['heartbeat']
            while previous.get('active') and time.time()-stamp<stale_seconds:
                print('D1_WAITING_FOR_DISCONNECTED_RUN_LEASE:',round(stale_seconds-(time.time()-stamp)),flush=True)
                time.sleep(min(20,max(.01,stale_seconds-(time.time()-stamp))))
                current=json.loads(path.read_text())
                if current.get('active') and current['heartbeat']!=stamp:
                    raise RuntimeError('Another D1 runtime is still active; stop it before resuming')
                previous=current
        atomic_json(path,dict(owner=owner,active=True,heartbeat=time.time()))
        stop=threading.Event();errors=[]
        def beat():
            while not stop.wait(heartbeat_seconds):
                try:
                    if json.loads(path.read_text())['owner']!=owner:raise RuntimeError('D1 lease ownership changed')
                    atomic_json(path,dict(owner=owner,active=True,heartbeat=time.time()))
                except BaseException as e:errors.append(e);return
        worker=threading.Thread(target=beat,daemon=True);worker.start()
        def check():
            if errors:raise OSError('D1 lease heartbeat failed') from errors[0]
            if json.loads(path.read_text())['owner']!=owner:raise RuntimeError('Concurrent D1 writer detected')
        try:yield check
        finally:
            stop.set();worker.join(timeout=25)
            if path.exists() and json.loads(path.read_text()).get('owner')==owner:
                atomic_json(path,dict(owner=owner,active=False,heartbeat=time.time()))
