import ast
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType
import zlib

import numpy as np
import pytest

from scripts import analyze_clip_paired_history as a


def rows(spec, interaction=2.):
    result = {}
    for si,s in enumerate(spec['seeds']):
        for k in [100,500]:
            for ui,u in enumerate(spec['streams']):
                value = (-1 if k == 100 else 1) + si*.1 + ui*.2
                def row(delta):
                    return {'points':{arm:{'50':{'report':{'native_loss':10.+(delta if arm=='sustained' else 0.),
                        'pool_retrieval':{'mean_r1_percent':3.}}}} for arm in spec['arms']}}
                result[s,k,u] = row(value),row(value+(interaction if k==500 else 0.))
    return {k:v[0] for k,v in result.items()},{k:v[1] for k,v in result.items()}


def test_paired_history_sign_and_seed_aggregation():
    spec=a.load_spec(); n,g=rows(spec)
    result=a.compare(n,g,spec)
    assert result['primary_value']==pytest.approx(2.)
    assert result['by_starting_step']['100']['interaction']['loss']==0
    assert result['by_starting_step']['500']['native_history']['loss']==pytest.approx(1.2)
    assert result['by_starting_step']['500']['gated_history']['loss']==pytest.approx(3.2)
    assert result['replication_units']==3 and len(result['paired_sets'])==12
    assert all(s['early_later_interaction']['loss']==pytest.approx(2.) for s in result['per_seed'])
    assert result['by_starting_step']['500']['interaction']['r1_pp']==0
    assert a.compare(g,n,spec)['primary_value']==pytest.approx(-2.)
    n.pop(next(iter(n)))
    with pytest.raises(ValueError,match='grid'): a.compare(n,g,spec)


def cache(scale):
    x=np.eye(512,dtype=np.float32)
    f=io.BytesIO(); np.savez_compressed(f,images=x,texts=x,scale=np.asarray(scale,dtype=np.float64))
    data=f.getvalue()
    feature=hashlib.sha256(x.tobytes()+x.tobytes()+json.dumps(float(scale)).encode()).hexdigest()
    meta=dict(npz_sha256=hashlib.sha256(data).hexdigest(),row_order_sha256='rowhash',rows=512,
              dimension=512,dtype='float32',feature_sha256=feature)
    logits=x.astype(np.float64)*scale
    loss=float(a.query_losses(logits)[:,2].mean())
    report=dict(native_loss=loss,features_sha256=feature,partition_losses={'full512':[loss]},
        pool_retrieval=dict(text_to_image_r1_percent=100.,image_to_text_r1_percent=100.,mean_r1_percent=100.))
    return data,meta,report


def evidence(spec,condition):
    """Small complete synthetic protocol grid; numerical caches are paired 512x512."""
    expected=spec['conditions'][condition]
    spec['conditions'][condition]['source_files']={}
    run_id=expected['experiment'].removesuffix('_v1')+'_fixture'
    if condition=='native': spec['conditions'][condition]['run_id']=run_id
    files={}
    put=lambda name,value:files.update({name:json.dumps(value,allow_nan=False).encode()})
    put('run_manifest.json',dict(run_id=run_id,experiment=expected['experiment'],bundle_id=expected['bundle_id'],
        status='complete_pending_drive_flush',packages={'torch':'2.11.0+cu128'},gpu='A100',source_commit='base'))
    put('completion.json',dict(status='complete',pairs=12,branches=48,endpoint_caches=48,
        trunk_optimizer_steps=1500,branch_optimizer_steps=2400,optimizer_steps=3900))
    put('protocol.json',expected['protocol']); put('source_manifest.json',{})
    for name in ['split_plan.json','branch_plans.json','timing_permutations.json','training_exclusion_indices.json']:
        put(name,{'same':True})
    grid={}; caches={scale:cache(scale) for scale in [1.,1.4,1.5]}
    alpha=expected['protocol']['coefficient']
    for seed in spec['seeds']:
        prefix=f'seed_{seed}/'
        traces=[dict(completed_updates=j,epoch=1,batch_identity_sha256=str(j),learning_rates=[.1],
            rng_after_sha256=str(j),coefficient=alpha if condition=='gated_history' and j>100 else 0.)
            for j in range(1,501)]
        audit=dict(actual_optimizer_steps=500,traces=traces,reporting_access=False,
            decisions=[] if condition=='native' else [dict(completed_updates=j,active=True,
                measurement={'mean_cosine':1.}) for j in [100,250]])
        if condition=='native':audit.update(history='native_only',auxiliary_updates=0,probe_calls=0)
        put(prefix+'trunk_audit.json',audit)
        for name in ['data_roles.json','training_batches.json','trainable_policy.json']:put(prefix+name,{'same':True})
        put(prefix+'training_config.json',{'dataset':{'diagnostic_holdout_indices':condition+'/exclusions'},'other':1})
        for step in [100,500]:
            state='early' if step==100 else condition+'later'
            put(prefix+f'start_{step:06d}.json',dict(epoch=1,state_sha256=state))
            for stream in spec['streams']:
                path=prefix+f'step_{step:06d}/stream_{stream}/'
                put(path+'training_batches.json',{'same':True})
                actions={arm:[arm=='sustained']*50 for arm in spec['arms']}
                branch_traces={arm:[dict(coefficient=alpha if arm=='sustained' else 0.,learning_rates=[.1],
                    rng_after_sha256=str(j)) for j in range(50)] for arm in spec['arms']}
                points={}
                for arm in spec['arms']:
                    scale=1. if arm!='sustained' else (1.4 if step==500 and condition=='gated_history' else 1.5)
                    data,meta,report=caches[scale]
                    files[path+f'features_{arm}.npz']=data;put(path+f'features_{arm}.json',meta)
                    points[arm]={'0':dict(report=caches[1.][2],state_sha256=state),
                                 '50':dict(report=report,state_sha256=state+arm)}
                audit=dict(actual_optimizer_steps=200,branch_start_hashes={arm:state for arm in spec['arms']},
                    source_immutable=True,frozen_parameters_unchanged=True,observations_preserve_state=True,
                    exact_no_update_report_replay=True,matched_rng_and_lrs=True,exposure_matched=True,reports_control_treatment=False)
                row=dict(training_seed=seed,checkpoint_step=step,stream_seed=stream,points=points,
                    actions=actions,traces=branch_traces,audit=audit,exposure={arm:sum(v) for arm,v in actions.items()},
                    report_row_order_sha256='rowhash')
                grid[seed,step,stream]=row;put(path+'result.json',row)
    put('comparison.json',dict(primary_value=0.,per_seed=[dict(training_seed=s,primary=0.) for s in spec['seeds']]))
    return dict(files=files,rows=grid,run=a.checked_json(files['run_manifest.json']))


@pytest.fixture
def pair():
    spec=deepcopy(a.load_spec())
    return spec,evidence(spec,'native'),evidence(spec,'gated_history')


def test_matching_and_native_trunk_contract(pair):
    spec,n,g=pair
    a.verify_matching(n,g,spec)
    for condition,run in [('native',n),('gated_history',g)]:
        audit=a.checked_json(run['files'][f"seed_{spec['seeds'][0]}/trunk_audit.json"])
        a.validate_trunk(audit,condition,spec['conditions'][condition]['protocol'])
        for row in run['rows'].values():a.validate_pair(row,spec)


@pytest.mark.parametrize('kind',['early_state','early_endpoint','early_actions','batch','environment','rng'])
def test_matching_rejects_changed_condition(pair,kind):
    spec,n,g=pair;s=spec['seeds'][0];u=spec['streams'][0]
    def change(name,fn):
        obj=a.checked_json(g['files'][name]);fn(obj);g['files'][name]=json.dumps(obj).encode()
    if kind=='early_state':change(f'seed_{s}/start_000100.json',lambda x:x.update(state_sha256='wrong'))
    if kind=='early_endpoint':g['rows'][s,100,u]['points']['sustained']['50']['state_sha256']='wrong'
    if kind=='early_actions':g['rows'][s,100,u]['actions']['gated'][0]=True
    if kind=='batch':change(f'seed_{s}/step_000500/stream_{u}/training_batches.json',lambda x:x.update(same=False))
    if kind=='environment':g['run']['packages']['torch']='cu130'
    if kind=='rng':g['rows'][s,500,u]['traces']['native'][0]['rng_after_sha256']='wrong'
    with pytest.raises(ValueError):a.verify_matching(n,g,spec)


def persist(tmp_path,run,spec,condition):
    path=tmp_path/condition;path.mkdir()
    for name,data in run['files'].items():
        target=path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    manifest=dict(run_id=run['run']['run_id'],status='verified_tree',files={n:dict(bytes=len(v),
        sha256=hashlib.sha256(v).hexdigest()) for n,v in run['files'].items()})
    marker=path/'DRIVE_BACKUP_MANIFEST.json';marker.write_text(json.dumps(manifest))
    if condition=='native':spec['conditions'][condition]['backup_manifest_sha256']=hashlib.sha256(marker.read_bytes()).hexdigest()
    return path


def test_complete_analysis_recomputes_caches_and_preserves_inputs(pair,tmp_path,monkeypatch):
    spec,n,g=pair
    npth=persist(tmp_path,n,spec,'native');gpth=persist(tmp_path,g,spec,'gated_history')
    monkeypatch.setattr(a,'load_spec',lambda:spec)
    out=tmp_path/'comparison'
    a.main(npth,gpth,out)
    result=json.loads((out/'comparison.json').read_text())
    assert result['primary_value']>0 and result['by_starting_step']['100']['interaction']['loss']==0
    assert result['history_exposure_by_seed'][str(spec['seeds'][0])]=={'native':0,'gated_history':400}
    with pytest.raises(ValueError,match='overwrite'):a.main(npth,gpth,out)
    # Input verification rechecks bytes, not just recorded status fields.
    (gpth/'comparison.json').write_text('{}')
    with pytest.raises(ValueError,match='Checksum'):a.read_run(gpth,'gated_history',spec)


def test_native_only_verification_never_claims_comparison(pair,tmp_path,monkeypatch,capsys):
    spec,n,_=pair;path=persist(tmp_path,n,spec,'native')
    monkeypatch.setattr(a,'load_spec',lambda:spec)
    out=tmp_path/'not_created';a.main(path,output=out)
    assert not out.exists() and 'Comparison pending' in capsys.readouterr().out


def test_delivery_and_parent_preservation():
    root=Path(__file__).resolve().parents[1]
    spec=a.load_spec()
    parent=root/'colabs/clip_fresh_split_policy_drive_one_cell.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['gated_history_cell_sha256']
    tree=ast.parse((root/'colabs/clip_paired_history_drive_one_cell.py').read_text())
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['PAYLOAD','BUNDLE_SHA256']}
    raw=zlib.decompress(base64.b64decode(values['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest()==values['BUNDLE_SHA256']
    for name,source in json.loads(raw).items():assert (root/name).read_text()==source


def test_launcher_flushes_plan_before_parent_and_rejects_changed_plan(tmp_path,monkeypatch):
    cell=a.ROOT/'colabs/clip_paired_history_drive_one_cell.py'
    tree=ast.parse(cell.read_text())
    tree.body.pop()  # Do not invoke run_all until the Colab/exec test doubles are installed.
    namespace={};exec(compile(tree,str(cell),'exec'),namespace)
    content=tmp_path/'content';content.mkdir()
    def path(value):
        text=str(value)
        return content/text.removeprefix('/content').lstrip('/') if text.startswith('/content') else Path(value)
    namespace['Path']=path
    calls=[]
    google=ModuleType('google');colab=ModuleType('google.colab');drive=ModuleType('drive')
    drive.mount=lambda _:calls.append('mount')
    drive.flush_and_unmount=lambda:calls.append('flush')
    colab.drive=drive;google.colab=colab
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    def parent(code,scope):
        assert calls[-1]=='flush'
        assert code.co_filename=='preserved_clip_fresh_split_policy_cell.py'
        assert (content/'drive/MyDrive/OTCO/evaluation_transfer/clip_paired_history_comparison_v1_plan/plan_manifest.json').exists()
        calls.append('parent')
    namespace['exec']=parent
    namespace['run_all']()
    assert calls==['mount','flush','parent']
    marker=content/'drive/MyDrive/OTCO/evaluation_transfer/clip_paired_history_comparison_v1_plan/plan_manifest.json'
    frozen=marker.read_bytes()
    namespace['run_all']()  # Same plan may be reused; parent's own duplicate-run guard remains intact.
    assert marker.read_bytes()==frozen and calls.count('parent')==2
    record=json.loads(frozen);record['bundle_id']='changed';marker.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='already frozen'):namespace['run_all']()
    assert calls.count('parent')==2
