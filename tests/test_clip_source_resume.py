from copy import deepcopy
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

import pytest
import torch

from src import clip_source_resume_training as rt
from src import clip_source_regime as original
from src import clip_source_resume_store as storage
from src import clip_source_resume_discovery as discovery
from src import clip_geometry_audit as g
from src import clip_checkpoint_diagnostic as d
from src import clip_native_history_policy as native
from tests.test_clip_source_regime import toy,calibration


@pytest.fixture(autouse=True)
def isolate():
    before=deepcopy(d.rng_state());backend=g.backend_state()
    yield
    g.restore_rng(before);g.restore_backend(backend)


def test_transactional_store_safe_load_corruption_and_contract(tmp_path,monkeypatch):
    store=storage.Checkpoints(tmp_path,{'bundle':'a'})
    store.save('phase',dict(x=torch.arange(3)))
    first=json.loads((tmp_path/'phase.json').read_text())
    original_copy=storage.copy_verified
    def fail(*a):raise OSError('interrupted upload')
    monkeypatch.setattr(storage,'copy_verified',fail)
    with pytest.raises(OSError):store.save('phase',{'x':torch.ones(3)})
    assert json.loads((tmp_path/'phase.json').read_text())==first
    assert torch.equal(store.load('phase')['x'],torch.arange(3))
    monkeypatch.setattr(storage,'copy_verified',original_copy)
    store.save('phase',{'x':torch.ones(3)});store.save('phase',{'x':torch.zeros(3)})
    assert len(list(tmp_path.glob('*.pt')))==2
    with pytest.raises(storage.ResumeMismatch):storage.Checkpoints(tmp_path,{'bundle':'b'}).load('phase')
    pointer=json.loads((tmp_path/'phase.json').read_text())
    (tmp_path/pointer['generations'][0]['file']).write_bytes(b'corrupt')
    with pytest.raises(storage.ResumeMismatch,match='corrupt'):store.load('phase')


def test_complete_training_state_serialization_and_frozen_fingerprint(tmp_path):
    model,opt,sched,*_=toy()
    model.eval();model.clip_model.text_projection.train()
    for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
    before=g.stamp(model,opt,sched)
    saved=storage.capture(model,opt,sched)
    store=storage.Checkpoints(tmp_path,{'test':1});store.save('live',saved)
    assert g.stamp(model,opt,sched)==before
    with torch.no_grad():model.clip_model.visual_projection.weight.add_(3.)
    opt.zero_grad();model.train();torch.rand(5)
    storage.restore(model,opt,sched,store.load('live'))
    assert g.stamp(model,opt,sched)==before
    with torch.no_grad():model.frozen.add_(1)
    with pytest.raises(storage.ResumeMismatch,match='frozen'):storage.restore(model,opt,sched,saved)


@pytest.mark.parametrize('fail_at',[3,6,17])
def test_interrupted_branches_equal_uninterrupted_and_skip_completed(tmp_path,monkeypatch,fail_at):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args;cal=calibration(args)
    kwargs=dict(calibrations=cal,p=p,commit=lambda *_:None)
    baseline=original.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs)
    store=storage.Checkpoints(tmp_path,{'bundle':'test'})
    step=rt.step_update;calls=[]
    def interrupt(*a,**k):
        calls.append(1)
        if len(calls)==fail_at:raise OSError('lost runtime')
        return step(*a,**k)
    monkeypatch.setattr(rt,'step_update',interrupt)
    with pytest.raises(OSError):rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,
        **kwargs,store=store,key='pair',interval=2)
    saved=store.load('pair')
    already=sum(len(v['traces']) for v in saved['branches'].values())
    resumed=[]
    def count(*a,**k):resumed.append(1);return step(*a,**k)
    monkeypatch.setattr(rt,'step_update',count)
    restored=rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,
        **kwargs,store=storage.Checkpoints(tmp_path,{'bundle':'test'}),key='pair',interval=2)
    assert restored==baseline
    assert len(resumed)==72-already
    # A finished pair is loaded without a single real update or virtual probe.
    monkeypatch.setattr(rt,'step_update',lambda *_a,**_k:pytest.fail('repeated completed training'))
    monkeypatch.setattr(rt,'optimizer_probe',lambda *_a,**_k:pytest.fail('repeated completed probe'))
    assert rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,
        **kwargs,store=store,key='pair',interval=2)==baseline


def test_second_disconnect_while_skipping_completed_keeps_active_checkpoint(tmp_path,monkeypatch):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args;cal=calibration(args)
    kwargs=dict(calibrations=cal,p=p,commit=lambda *_:None,store=storage.Checkpoints(tmp_path,{}),key='pair',interval=2)
    step=rt.step_update;count=[0]
    def fail(*a,**k):
        count[0]+=1
        if count[0]==7:raise OSError('disconnect')
        return step(*a,**k)
    monkeypatch.setattr(rt,'step_update',fail)
    with pytest.raises(OSError):rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs)
    expected=kwargs['store'].load('pair')['active']
    save=kwargs['store'].save
    def fail_after_first_save(*a,**k):save(*a,**k);raise OSError('second disconnect')
    monkeypatch.setattr(kwargs['store'],'save',fail_after_first_save)
    with pytest.raises(OSError):rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,**kwargs)
    assert d.state_digest(kwargs['store'].load('pair')['active'])==d.state_digest(expected)


def test_native_resume_restores_loader_rng_and_snapshot_hashes(tmp_path,monkeypatch):
    from tests.test_clip_paired_updates import TinyCLIP
    from src.clip_fresh_split_policy import training_batches
    from types import SimpleNamespace
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):return 24
        def set_epoch(self,n):self.epoch=n
        def __getitem__(self,i):
            return dict(pixel_values=torch.tensor([i/24.,0.,1.,0.,2.,3.]),input_ids=torch.tensor([0.,i/24.,1.,2.,3.,0.]),
                        source_indices=i,captions=f'caption_{self.epoch}_{i}')
    def init():
        torch.manual_seed(234);model=TinyCLIP()
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        dataset=Dataset();loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(123))
        return model,opt,sched,SimpleNamespace(train_dataset=dataset,train_loader=loader)
    config={'training':{'mixed_precision':'none'},'optimizer':{'max_gradient_norm':1.}}
    p={**native.load_protocol(),'checkpoint_steps':[2,6],'trunk_updates':6}
    model,opt,sched,data=init()
    states,audit=native.trunk(model,opt,sched,training_batches(data),config,p,seed=1,commit=lambda *_:None)
    model,opt,sched,data=init();store=storage.Checkpoints(tmp_path,{});step=d.step_update;calls=[0]
    def stop(*a,**k):
        calls[0]+=1
        if calls[0]==4:raise OSError('lost runtime')
        return step(*a,**k)
    monkeypatch.setattr(d,'step_update',stop)
    with pytest.raises(OSError):rt.resumable_trunk(model,opt,sched,training_batches(data),config,p,
        seed=1,commit=lambda *_:None,store=store,interval=2,loader=data.train_loader)
    model,opt,sched,data=init();actual=[]
    def count(*a,**k):actual.append(1);return step(*a,**k)
    monkeypatch.setattr(d,'step_update',count)
    recovered,actual_audit=rt.resumable_trunk(model,opt,sched,training_batches(data),config,p,
        seed=1,commit=lambda *_:None,store=store,interval=2,loader=data.train_loader)
    assert actual_audit==audit and d.state_digest(recovered)==d.state_digest(states)
    assert len(actual)==4
    model,opt,sched,data=init();actual.clear()
    recovered,_=rt.resumable_trunk(model,opt,sched,training_batches(data),config,p,seed=1,commit=lambda *_:None,
        store=store,interval=2,loader=data.train_loader)
    assert not actual and d.state_digest(recovered)==d.state_digest(states)


def test_legacy_completed_branch_reuse_and_partial_replay(tmp_path):
    args=toy();model,opt,sched,batch,cfg,states,report,record,p=args;cal=calibration(args)
    def commit(name,value):d.write_json(tmp_path/(name+'.json'),value)
    baseline=original.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,calibrations=cal,p=p,commit=commit)
    # Simulate old runner stopped after one branch. Only this complete branch is reused.
    legacy=tmp_path/'legacy';legacy.mkdir()
    (legacy/'branch_w1_native.json').write_bytes((tmp_path/'branch_w1_native.json').read_bytes())
    store=storage.Checkpoints(tmp_path/'checkpoints',{})
    result=rt.run_branches(model,opt,sched,states,[batch]*4,[record]*4,cfg,report,calibrations=cal,p=p,
        commit=lambda *_:None,store=store,key='pair',interval=2,legacy_folder=legacy)
    assert result==baseline and store.events[-1]['optimizer_steps_this_attempt']==68


def test_discovery_ambiguity_legacy_lineage_and_science_failure(tmp_path):
    def add(name,bundle,**extra):
        p=tmp_path/name;p.mkdir();storage.atomic_json(p/'run_manifest.json',dict(experiment='clip_source_regime_v1',bundle_id=bundle,**extra));return p
    assert discovery.discover(tmp_path,'new','old') is None
    old=add('clip_source_regime_old','old')
    assert discovery.discover(tmp_path,'new','old')[0]==old
    new=add('clip_source_resume_new','new',parent_run=old.name)
    assert discovery.discover(tmp_path,'new','old')[0]==new
    add('clip_source_resume_other','new')
    with pytest.raises(storage.ResumeMismatch,match='Multiple'):discovery.discover(tmp_path,'new','old')
    assert discovery.discover(tmp_path,'new','old',new.name)[0]==new
    storage.atomic_json(new/'run_manifest.json',dict(experiment='clip_source_regime_v1',bundle_id='new',resume_blocked=True,error='bad replay'))
    with pytest.raises(storage.ResumeMismatch,match='scientific'):discovery.discover(tmp_path,'new','old',new.name)


def test_result_inventory_authentication(tmp_path):
    remote=tmp_path/'remote';remote.mkdir();local=tmp_path/'local'
    sources={'a.py':hashlib.sha256(b'test').hexdigest()}
    (remote/'source_overlay').mkdir();(remote/'source_overlay/a.py').write_bytes(b'test')
    storage.atomic_json(remote/'source_manifest.json',sources)
    storage.atomic_json(remote/'branch.json',{'complete':True})
    files={str(p.relative_to(remote)):{'sha256':storage.checksum(p)} for p in remote.rglob('*') if p.is_file()}
    storage.atomic_json(remote/'DRIVE_BACKUP_MANIFEST.json',{'files':files})
    discovery.restore_artifacts(remote,local,sources)
    (remote/'branch.json').write_text('{}')
    with pytest.raises(storage.ResumeMismatch,match='corrupt'):discovery.restore_artifacts(remote,local,sources)


def test_resume_delivery_and_original_unchanged():
    path=d.ROOT/'colabs/clip_source_resume_drive_one_cell.py'
    if not path.exists():pytest.skip('Generated delivery absent from overlay')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(values['PAYLOAD']));assert hashlib.sha256(raw).hexdigest()==values['BUNDLE_SHA256']
    sources=json.loads(raw);manifest=json.loads(sources.pop('source_resume_source_manifest.json'))
    assert set(sources)==set(manifest)
    for name,value in sources.items():
        assert hashlib.sha256(value.encode()).hexdigest()==manifest[name] and (d.ROOT/name).read_text()==value
    old=json.loads((d.ROOT/'configs/clip_source_resume_reference.json').read_text())
    for name,sha in old['sources'].items():assert storage.checksum(d.ROOT/name)==sha


def test_complete_runner_restart_reuses_calibration_and_all_branches(tmp_path,monkeypatch):
    from src import clip_source_run as issued
    from src import clip_source_resume_run as resumed
    from tests.test_clip_source_regime import test_complete_d1_runner_persists_full_grid as exercise
    store=storage.Checkpoints(tmp_path/'remote_checkpoints',{'test':'end_to_end'})
    def execute(*a,**kw):
        monkeypatch.setattr(resumed,'fixed_probe_batches',issued.fixed_probe_batches)
        first=resumed.run(*a,**kw,store=store)
        with monkeypatch.context() as check:
            check.setattr(rt,'step_update',lambda *_a,**_kw:pytest.fail('Resumed complete branch trained again'))
            check.setattr(d,'step_update',lambda *_a,**_kw:pytest.fail('Resumed complete native trunk trained again'))
            check.setattr(resumed,'calibrate_states',lambda *_a,**_kw:pytest.fail('Resumed complete calibration recomputed'))
            second=resumed.run(*a,**kw,store=storage.Checkpoints(tmp_path/'remote_checkpoints',{'test':'end_to_end'}))
        assert first==second
        return second
    monkeypatch.setattr(issued,'run',execute)
    exercise(tmp_path,monkeypatch)


def test_changed_bundle_never_silently_starts_new_experiment(tmp_path):
    folder=tmp_path/'clip_source_resume_other_version';folder.mkdir()
    storage.atomic_json(folder/'run_manifest.json',dict(experiment='clip_source_regime_v1',bundle_id='different'))
    with pytest.raises(storage.ResumeMismatch,match='different source bundle'):discovery.discover(tmp_path,'current','legacy')


def test_lease_release_and_stale_recovery(tmp_path):
    import time
    folder=tmp_path/'run'
    with discovery.lease(folder,stale_seconds=.01,heartbeat_seconds=.005) as check:
        check()
        with pytest.raises(RuntimeError,match='active process'):
            with discovery.lease(folder):pass
    assert not json.loads((folder/'resume_lease.json').read_text())['active']
    storage.atomic_json(folder/'resume_lease.json',dict(owner='disconnected',active=True,heartbeat=time.time()-5))
    with discovery.lease(folder,stale_seconds=.01,heartbeat_seconds=.005) as check:check()


def test_pointer_publication_failure_retains_previous_checkpoint(tmp_path,monkeypatch):
    store=storage.Checkpoints(tmp_path,{})
    store.save('phase',{'step':10})
    original_pointer=(tmp_path/'phase.json').read_bytes()
    def fail(*_):raise OSError('pointer publish interrupted')
    monkeypatch.setattr(storage,'atomic_json',fail)
    with pytest.raises(OSError):store.save('phase',{'step':20})
    assert (tmp_path/'phase.json').read_bytes()==original_pointer
    assert store.load('phase')['step']==10
