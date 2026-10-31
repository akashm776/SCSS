"""Transactional, authenticated D1 resume checkpoints; no training RNG use."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np
import torch

from colabs.backup_clip_run import checksum, copy_verified
from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src.clip_fresh_split_policy import cpu_copy


class ResumeMismatch(ValueError):
    pass


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def safe_load(path):
    allowed=[np._core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True)


def compact(state,model):
    """Persist mutable parameters/buffers, authenticate pretrained frozen tensors."""
    names={name for name,v in model.named_parameters() if v.requires_grad}
    names |= {name for name,_ in model.named_buffers()}
    result=cpu_copy({k:v for k,v in state.items() if k!='model'})
    result['mutable_model']={k:cpu_copy(v) for k,v in state['model'].items() if k in names}
    result['frozen_sha256']=d.state_digest({k:v for k,v in state['model'].items() if k not in names})
    return result


def expand(state,model):
    mutable=state['mutable_model']
    expected={name for name,v in model.named_parameters() if v.requires_grad}
    expected |= {name for name,_ in model.named_buffers()}
    expected &= set(model.state_dict())
    if set(mutable)!=expected:raise ResumeMismatch('Checkpoint trainable policy changed')
    frozen={k:v for k,v in model.state_dict().items() if k not in mutable}
    if d.state_digest(frozen)!=state['frozen_sha256']:raise ResumeMismatch('Pretrained frozen weights differ')
    result={k:deepcopy(v) for k,v in state.items() if k not in ['mutable_model','frozen_sha256']}
    result['model']={**cpu_copy(frozen),**cpu_copy(mutable)}
    return result


def capture(model,opt,sched):
    value=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),**d.rng_state(),
        gradients=[v.grad for v in model.parameters()],modes=[m.training for m in model.modules()],backend=g.backend_state())
    return dict(state=compact(value,model),stamp=g.stamp(model,opt,sched))


def restore(model,opt,sched,payload):
    state=expand(payload['state'],model)
    d.restore(model,opt,sched,state)
    for parameter,grad in zip(model.parameters(),state['gradients']):
        parameter.grad=None if grad is None else grad.to(parameter.device).clone()
    for module,mode in zip(model.modules(),state['modes']):module.training=mode
    g.restore_backend(state['backend'])
    if g.stamp(model,opt,sched)!=payload['stamp']:raise ResumeMismatch('Restored training-state fingerprint differs')


class Checkpoints:
    """Two committed generations per phase. Blobs close and verify before pointer publication.

    Remote root is outside the result mirror's source tree. A failed write cannot
    replace the last pointer. Retired blobs are deleted only after publication.
    """
    def __init__(self,remote,contract,*,sync=lambda:None,local=None):
        self.root=Path(remote);self.root.mkdir(parents=True,exist_ok=True)
        self.contract=contract;self.sync=sync;self.local=Path(local) if local else None
        self.events=[]

    def load(self,key):
        index=self.root/(key+'.json')
        if not index.exists():return None
        pointer=json.loads(index.read_text())
        if pointer['contract']!=self.contract:raise ResumeMismatch('Resume contract/source bundle changed')
        # No automatic rollback of a committed corrupt checkpoint: never hide corruption.
        entry=pointer['generations'][0];path=self.root/entry['file']
        if not path.is_file() or checksum(path)!=entry['sha256']:
            raise ResumeMismatch('Committed checkpoint missing or corrupt: '+key)
        value=safe_load(path)
        if value['contract']!=self.contract or value['key']!=key:raise ResumeMismatch('Checkpoint identity mismatch')
        self.events.append(dict(event='load',key=key,sha256=entry['sha256']))
        return value['payload']

    def save(self,key,payload):
        if '/' in key or '..' in key:raise ValueError('Invalid checkpoint key')
        self.sync()  # Results referenced by the checkpoint are durable first.
        pointer_path=self.root/(key+'.json')
        prior=json.loads(pointer_path.read_text()) if pointer_path.exists() else dict(contract=self.contract,generations=[])
        if prior['contract']!=self.contract:raise ResumeMismatch('Checkpoint contract changed')
        name=key+'_'+uuid.uuid4().hex+'.pt'
        with tempfile.TemporaryDirectory(prefix='d1-checkpoint-') as scratch:
            source=Path(scratch)/name
            torch.save(dict(contract=self.contract,key=key,payload=payload),source)
            info,_=copy_verified(source,self.root/name)
        entry=dict(file=name,**info)
        generations=[entry,*prior['generations']][:2]
        atomic_json(pointer_path,dict(contract=self.contract,generations=generations))
        if json.loads(pointer_path.read_text())['generations'][0]!=entry:
            raise OSError('Checkpoint pointer read-back mismatch')
        self.events.append(dict(event='save',key=key,sha256=entry['sha256']))
        # Only files previously owned by this phase's committed pointer are pruned.
        keep={v['file'] for v in generations}
        for old in prior['generations']:
            if old['file'] not in keep:(self.root/old['file']).unlink(missing_ok=True)
        print('D1_CHECKPOINT_VERIFIED:',key,flush=True)
