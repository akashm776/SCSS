from copy import deepcopy
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import pytest

from src import clip_geometry_audit as g
from src import clip_geometry_recovery as recovery
from src import clip_geometry_run as original
from src import clip_geometry_recovery_run as continued
from src import clip_checkpoint_diagnostic as d
from src import clip_optimizer_reset as reset
from tests.test_clip_geometry_audit import rows_fixture, test_complete_toy_run_preparation_loading_and_no_audit_updates as toy_run


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,allow_nan=False))


def make_source(tmp_path):
    rows,p=rows_fixture();p=deepcopy(p)
    p.update(training_seeds=[1],stream_seeds=[2],batch_size=32)
    p['populations']['extended']['offsets']=[0]
    p['populations']['calibration_feasibility']['batch_offsets']=[]
    p['budget'].update(total_batch_forwards=2,total_query_exposures=64)
    start={'native':'toy'};trunk={'trunk':'toy'}
    plan={'continuation_batches':{'1:2':[list(range(32))]},'calibration_batches':{'1':[]},
        'seeds':{'1':{'states':{str(w):{'archived_start':start} for w in [100,500]},
            'native_trunk_audit_canonical_sha256':reset.digest(trunk),
            'streams':{'2':{'training_batch_records_sha256':'toyidentity'}}}}}
    folder=tmp_path/'prior';folder.mkdir()
    sources=json.loads((d.ROOT/'configs/clip_geometry_recovery_reference.json').read_text())['source_files']
    for n in sources:
        dest=folder/'source_overlay'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(d.ROOT/n,dest)
    put(folder/'run_manifest.json',dict(experiment=p['experiment'],bundle_id=recovery.ORIGINAL_BUNDLE,
        run_id=recovery.ORIGINAL_RUN,freeze_manifest_sha256=g.FREEZE_SHA256,status='running'))
    put(folder/'source_manifest.json',sources);put(folder/'protocol.json',p);put(folder/'plan.json',plan)
    shutil.copyfile(d.ROOT/'configs/clip_geometry_audit_freeze_manifest.json',folder/'freeze_manifest.json')
    for w in [100,500]:put(folder/f'seed_1/start_{w:06d}.json',start)
    put(folder/'seed_1/trunk_audit.json',trunk)
    group=folder/'seed_1/stream_2';group.mkdir()
    put(group/'batch_identity.json',dict(canonical_sha256='toyidentity',source_indices=list(range(32))))
    record=original.write_gzip(group/'w100_batch000.jsonl.gz',rows,lines=True)
    original.write_gzip(group/'w100_batch000_summary.json.gz',{'summary':'toy'})
    put(group/'w100_batch000_audit.json',dict(training_seed=1,population='continuation',stream_seed=2,batch_offset=0,weight_step=100,
        preserved=True,before_sha256='same',after_sha256='same',optimizer_updates=0,backward_passes=0,queries=32,records=record))
    inventory(folder)
    return folder,p,plan,rows


def inventory(folder):
    put(folder/'DRIVE_BACKUP_MANIFEST.json',dict(run_id=recovery.ORIGINAL_RUN,status='in_progress',files={
        str(f.relative_to(folder)):dict(sha256=d.checksum(f),bytes=f.stat().st_size)
        for f in folder.rglob('*') if f.is_file() and f!=folder/'DRIVE_BACKUP_MANIFEST.json'}))


def test_reuse_and_missing_branch(tmp_path):
    folder,p,plan,rows=make_source(tmp_path)
    r=recovery.Recovery([folder],bundle_id='a'*64,p=p,plan=plan)
    ids=[{n:row[n] for n in ['source_index','caption_index','species_id']} for row in rows]
    cached=r.get(rows[0],ids);assert cached[0]==rows and len(r.used)==1
    with pytest.raises(g.AuditFailure,match='twice'):r.get(rows[0],ids)
    late={**rows[0],'weight_step':500};assert r.get(late,ids) is None
    r.finish(tmp_path,500)
    result=json.loads((tmp_path/'recovery_completion.json').read_text())
    assert result['reused_batch_forwards']==result['new_batch_forwards']==1


@pytest.mark.parametrize('kind',['bytes','identity','state_audit','failed_run','path_escape'])
def test_reject_bad_recovery(tmp_path,kind):
    folder,p,plan,rows=make_source(tmp_path)
    if kind=='bytes':
        with (folder/'seed_1/stream_2/w100_batch000.jsonl.gz').open('ab') as f:f.write(b'changed')
    elif kind=='identity':
        path=folder/'seed_1/stream_2/batch_identity.json';obj=json.loads(path.read_text());obj['canonical_sha256']='changed';put(path,obj);inventory(folder)
    elif kind=='state_audit':
        path=folder/'seed_1/stream_2/w100_batch000_audit.json';obj=json.loads(path.read_text());obj['after_sha256']='changed';put(path,obj);inventory(folder)
    elif kind=='failed_run':
        put(folder/'failure.json',{'error':'invariant failure'});inventory(folder)
    else:
        path=folder/'DRIVE_BACKUP_MANIFEST.json';obj=json.loads(path.read_text());obj['files']['../escape']={'sha256':'x','bytes':0};put(path,obj)
    with pytest.raises(g.AuditFailure):recovery.Recovery([folder],bundle_id='a'*64,p=p,plan=plan)


def test_incomplete_committed_batch_not_reused(tmp_path):
    folder,p,plan,rows=make_source(tmp_path)
    (folder/'seed_1/stream_2/w100_batch000_summary.json.gz').unlink();inventory(folder)
    with pytest.raises(g.AuditFailure,match='No authenticated'):recovery.Recovery([folder],bundle_id='a'*64,p=p,plan=plan)


def test_conflicting_duplicate_shards_rejected(tmp_path):
    folder,p,plan,rows=make_source(tmp_path)
    other=tmp_path/'second';shutil.copytree(folder,other)
    path=other/'seed_1/stream_2/w100_batch000.jsonl.gz'
    rows[0]['positive_cosine']+=.01;path.unlink();info=original.write_gzip(path,rows,lines=True)
    audit_path=path.with_name('w100_batch000_audit.json');audit=json.loads(audit_path.read_text());audit['records']=info;put(audit_path,audit);inventory(other)
    with pytest.raises(g.AuditFailure,match='Conflicting'):recovery.Recovery([folder,other],bundle_id='a'*64,p=p,plan=plan)


def test_complete_execution_reuses_without_encoder_forward(tmp_path,monkeypatch):
    # Run the existing independent toy native-reference fixture through the new
    # executor twice: first collect records, then reuse all six and forbid audit.
    cache={};calls=[];base_run=continued.run
    class Cache:
        def __init__(self):self.used=set();self.new_forwards=0;self.entries={k:{'run_id':'toy'} for k in cache}
        def save_provenance(self,*_):pass
        def get(self,ctx,ids):
            k=recovery.key(ctx)
            if k in cache:self.used.add(k);return deepcopy(cache[k])
            self.new_forwards+=1;return None
        def finish(self,directory,updates):
            assert len(list(Path(directory).glob('seed_*/start_*.json')))==2
            calls.append((len(self.used),self.new_forwards))
            for path in Path(directory).glob('seed_*/*/*.jsonl.gz'):
                with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
                audit=json.loads(path.with_name(path.name.replace('.jsonl.gz','_audit.json')).read_text())
                cache[recovery.key(audit)]=(rows,audit)
    audit=g.audit_batch
    def forbid_after_first(*args,**kw):
        assert not calls, 'Recovered observation performed an encoder audit'
        return audit(*args,**kw)
    monkeypatch.setattr(g,'audit_batch',forbid_after_first)
    monkeypatch.setattr(original,'run',lambda *a,**kw:base_run(*a,**kw,recovery=Cache()))
    # The toy fixture patches original fixed_probe_batches; mirror that injection.
    monkeypatch.setattr(continued,'fixed_probe_batches',lambda *a,**kw:original.fixed_probe_batches(*a,**kw))
    toy_run(tmp_path,monkeypatch)
    assert calls==[(0,6),(6,0)]



def test_can_continue_a_prior_recovery_with_checked_ancestry(tmp_path,monkeypatch):
    folder,p,plan,rows=make_source(tmp_path)
    local=tmp_path/'local_sources';shutil.copytree(folder/'source_overlay',local)
    shutil.copyfile(d.ROOT/'configs/clip_geometry_recovery_reference.json',local/'configs/clip_geometry_recovery_reference.json')
    sources=json.loads((folder/'source_manifest.json').read_text())
    put(local/'geometry_recovery_source_manifest.json',sources)
    manifest=json.loads((folder/'run_manifest.json').read_text())
    manifest.update(bundle_id='a'*64,recovery_root_run=recovery.ORIGINAL_RUN)
    put(folder/'run_manifest.json',manifest);inventory(folder)
    monkeypatch.setattr(d,'ROOT',local)
    r=recovery.Recovery([folder],bundle_id='a'*64,p=p,plan=plan)
    assert len(r.entries)==1
    manifest['recovery_root_run']='unrelated';put(folder/'run_manifest.json',manifest);inventory(folder)
    with pytest.raises(g.AuditFailure,match='ancestry'):recovery.Recovery([folder],bundle_id='a'*64,p=p,plan=plan)

def test_recovery_delivery_bundle():
    path=d.ROOT/'colabs/clip_geometry_recovery_drive_one_cell.py'
    if not path.exists():pytest.skip('Delivery cell excluded from overlay')
    vals={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(vals['PAYLOAD']));assert hashlib.sha256(raw).hexdigest()==vals['BUNDLE_SHA256']
    sources=json.loads(raw);manifest=json.loads(sources.pop('geometry_recovery_source_manifest.json'))
    assert set(sources)==set(manifest)
    for name,value in sources.items():
        assert hashlib.sha256(value.encode()).hexdigest()==manifest[name]
        if name!='geometry_source_manifest.json':assert value==(d.ROOT/name).read_text()
