import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zlib

import numpy as np
import pytest
import torch

from scripts import analyze_clip_image_influence as a


def test_directional_losses_match_independent_cross_entropy():
    logits=np.array([[12.,-4.,7.],[2.,-1.,.3],[-9.,1.,4.]])
    q=a.query_losses(logits)
    x=torch.tensor(logits,dtype=torch.float64)
    t=torch.nn.functional.cross_entropy(x,torch.arange(3),reduction='none').numpy()
    i=torch.nn.functional.cross_entropy(x.T,torch.arange(3),reduction='none').numpy()
    np.testing.assert_allclose(q,np.stack([t,i,(t+i)/2],axis=1),atol=1e-14)
    np.testing.assert_allclose(a.query_losses(logits+1000),q,atol=2e-13)


def test_paired_deletion_recomputes_candidates_not_just_queries():
    logits=np.array([[2.,10.,0.,0.],[1.,2.,0.,0.],[0.,0.,3.,1.],[0.,0.,1.,2.]])
    groups=[[0,1],[2,3]]
    actual=a.deletion_losses(logits,groups)
    np.testing.assert_allclose(actual[0],a.query_losses(logits[2:,2:]).mean(axis=0))
    assert not np.allclose(actual[1],a.query_losses(logits)[:2].mean(axis=0))
    with pytest.raises(ValueError,match='partition'):
        a.deletion_losses(logits,[[0,1],[1,2]])


def test_fixed_groups_cover_once_and_are_order_identity_based():
    ids=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(512)]
    groups=a.groups_for(ids,2026093004,16)
    assert len(groups)==32 and all(len(g)==16 for g in groups)
    assert sorted(i for g in groups for i in g)==list(range(512))
    reversed_ids=list(reversed(ids))
    assert [[ids[i] for i in g] for g in groups]==[[reversed_ids[i] for i in g] for g in a.groups_for(reversed_ids,2026093004,16)]
    with pytest.raises(ValueError):a.groups_for(['x']*512,1,16)


def test_concentration_exposes_dominant_query_without_net_share_instability():
    c=a.concentration(np.array([-10.]+[.2]*7),[1,5])
    assert c['mean']<0 and c['top'][0]['query_only_mean_after_removing_most_favorable']>0
    assert c['top'][0]['fraction_of_gross_benefit']==1.
    assert c['leave_one_query_out']['unfavorable']==1
    distributed=a.concentration(np.full(8,-.1),[1,5])
    assert distributed['leave_one_query_out']['favorable']==8
    assert distributed['top'][0]['fraction_of_gross_benefit']==pytest.approx(1/8)
    assert a.concentration(np.ones(8),[1])['top'][0]['fraction_of_gross_benefit'] is None


def test_hierarchical_mean_and_disaggregation():
    rows=[]; values=[]
    for si,s in enumerate(a.SPEC['seeds']):
        for k in a.SPEC['checkpoints']:
            for u in a.SPEC['streams']:
                rows.append(dict(training_seed=s,checkpoint_step=k,stream_seed=u))
                values.append(np.full((4,3),si+1.))
    np.testing.assert_equal(a.hierarchical(rows,values),np.full((4,3),2.))
    np.testing.assert_equal(a.hierarchical(rows,values,seeds=[789]),np.ones((4,3)))
    with pytest.raises(ValueError,match='Duplicate'):
        a.hierarchical(rows+[rows[0]],values+[values[0]])


def cache():
    images=np.eye(512,dtype=np.float32); texts=images.copy(); scale=2.
    f=io.BytesIO(); np.savez_compressed(f,images=images,texts=texts,scale=scale); data=f.getvalue()
    meta=dict(npz_sha256=hashlib.sha256(data).hexdigest(),row_order_sha256='a'*64,rows=512,dimension=512,dtype='float32',
        feature_sha256=hashlib.sha256(images.tobytes()+texts.tobytes()+json.dumps(scale).encode()).hexdigest())
    return data,meta


def test_cache_hash_identity_and_exact_decoding():
    data,meta=cache()
    np.testing.assert_equal(a.unpack(data,meta,'a'*64),np.eye(512)*2)
    with pytest.raises(ValueError,match='checksum'):a.unpack(data+b'x',meta,'a'*64)
    with pytest.raises(ValueError,match='identity'):a.unpack(data,meta,'b'*64)
    with pytest.raises(ValueError,match='Feature'):a.unpack(data,{**meta,'feature_sha256':'0'*64},'a'*64)


def test_no_overwrite(tmp_path):
    with pytest.raises(ValueError,match='overwrite'):
        a.main(tmp_path/'missing.zip',tmp_path)


def test_embedded_cell_sources():
    root=Path(__file__).resolve().parents[1]
    cell=root/'colabs/clip_image_influence_one_cell.py'
    tree=ast.parse(cell.read_text())
    v={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
       and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'PAYLOAD','BUNDLE_SHA256'}}
    raw=zlib.decompress(base64.b64decode(v['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==v['BUNDLE_SHA256']
    for name,source in json.loads(raw).items(): assert source==(root/name).read_text()
