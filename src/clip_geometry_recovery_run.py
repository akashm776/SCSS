"""Explicit recovery execution. Original D0 numerical and summary sources stay immutable."""
from copy import deepcopy
import gc
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import torch
import numpy as np

from src import clip_geometry_audit as g
from src.clip_geometry_summary import Summary, Accumulator, markdown_report
from src import clip_checkpoint_diagnostic as d
from src import clip_difficulty as b
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy
from src import clip_weight_lr as wl
from src import clip_train
from src.clip_fresh_split_policy import split_indices, training_batches
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol


def write_gzip(path,value,*,lines=False):
    path=Path(path)
    require_new=not path.exists()
    g.require(require_new,'Refusing to overwrite audit shard: '+str(path))
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as out:
            if lines:
                for row in value:out.write((json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode())
            else:out.write((json.dumps(value,allow_nan=False,separators=(',',':'))+'\n').encode())
    return dict(path=path.name,sha256=d.checksum(path),bytes=path.stat().st_size)


def authenticate_states(states,plan_seed,steps,trunk_audit):
    g.require(set(states)==set(steps),'Wrong native snapshot grid')
    g.require(reset.digest(trunk_audit)==plan_seed['native_trunk_audit_canonical_sha256'],'Native trunk replay mismatch')
    for w in steps:
        s=states[w];expected=plan_seed['states'][str(w)]
        g.require(d.state_digest(s)==expected['archived_start']['snapshot_sha256'] and
            d.state_digest(s['model'])==expected['weights_sha256'] and
            s['completed_updates']==w and s['starting_state_history']=='native_only' and
            s['prior_auxiliary_updates']==0 and s['arm']=='baseline','Native snapshot authentication failed')


def load_snapshots(directory,seed,steps,plan_seed):
    """Restricted deserialization of explicitly supplied local snapshots; no auto-discovery."""
    if directory is None:return None
    folder=Path(directory)/f'seed_{seed}'
    paths=[folder/f'step_{w:06d}.pt' for w in steps]
    exists=[p.exists() for p in paths]
    if not any(exists):return None
    g.require(all(exists) and (folder/'trunk_audit.json').is_file(),'Incomplete supplied snapshot pair; no fallback')
    # Native snapshots contain NumPy RNG arrays. Permit these concrete data types,
    # not arbitrary pickle functions, and authenticate the complete decoded state.
    allowed=[np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allowed):
        states={w:torch.load(path,map_location='cpu',weights_only=True) for w,path in zip(steps,paths)}
    audit=json.loads((folder/'trunk_audit.json').read_text())
    authenticate_states(states,plan_seed,steps,audit)
    return states,audit


def run(directory,sync,*,p=None,plan=None,preparation_protocol=None,native_reference=None,snapshot_dir=None,recovery=None):
    from transformers import AutoProcessor
    p=g.load_protocol() if p is None else p
    plan=json.loads((d.ROOT/p['plan']).read_text()) if plan is None else plan
    prep=b.load_protocol() if preparation_protocol is None else preparation_protocol
    ref=reset.load_reference() if native_reference is None else native_reference
    directory=Path(directory)
    if snapshot_dir is not None:g.require(Path(snapshot_dir).is_dir(),'Supplied snapshot directory does not exist')
    old=json.loads((d.ROOT/'configs/cub200_clip_diagnostic_holdout_indices.json').read_text())
    probes=[i for batch in json.loads((d.ROOT/'experiment_results/clip_paired_updates_2026-09/training_source_indices.json').read_text()) for i in batch]
    split=split_indices(list(range(prep['original_train_size'])),old,probes,prep)
    plans,_=policy.make_plans(split['training'],prep)
    exclusions=sorted(set(old)|set(split['report']))
    for name,value in [('split_plan.json',split),('branch_plans.json',plans),('training_exclusion_indices.json',exclusions)]:
        g.require(reset.digest(value)==ref[name],'Preparation data plan differs: '+name)
    g.require(wl.common_plans(split['training'],prep)==plan['continuation_batches'],'Frozen continuation membership differs')
    for seed in p['training_seeds']:
        ci=[i for batch in plan['calibration_batches'][str(seed)] for i in batch]
        used={i for u in p['stream_seeds'] for batch in plan['continuation_batches'][f'{seed}:{u}'] for i in batch}
        g.require(len(ci)==len(set(ci)) and set(ci)<=set(split['training']) and
            not set(ci)&(used|set(split['frozen_probes'])|set(split['report'])|set(old)),'Calibration data overlap')
    for name,value in [('protocol.json',p),('plan.json',plan),('freeze_manifest.json',g.load_freeze()),
        ('preparation_protocol.json',prep),('split_plan.json',split),('native_replay_reference.json',ref),
        ('training_exclusion_indices.json',exclusions)]:d.write_json(directory/name,value)
    g.require(recovery is not None, "Explicit authenticated recovery object required")
    recovery.save_provenance(directory)
    sync()
    summary=Summary(p);shards=[];preparation_updates=0;preparation_records=[]
    for seed in p['training_seeds']:
        clip_train.set_global_seed(seed)
        folder=directory/f'seed_{seed}';folder.mkdir(exist_ok=False)
        config=clip_train.load_training_config(d.ROOT/'configs/hf_cub200_clip_vit_b32_baseline.yaml')
        config['experiment']['seed']=config['training']['seed']=seed
        config['runtime']['num_workers']=0
        config['dataset']['diagnostic_holdout_indices']=str((directory/'training_exclusion_indices.json').resolve())
        device,info=clip_train.resolve_device(config['runtime'])
        processor=AutoProcessor.from_pretrained(config['model']['checkpoint'])
        data=clip_train.build_clip_training_data(config,processor,pin_memory=False)
        g.require(data.train_dataset.source_indices==split['training'] and len(data.train_loader)==prep['expected_batches_per_epoch'],'Training dataset differs')
        # Recreate the historical data-construction order and authenticate identities.
        # No reporting image is passed through a model, and no captions are persisted.
        report_batches,records=d.build_batches(data,split['report'],seed=0,epoch=0,canonical=True)
        keys=lambda indices:{data.train_dataset.grouped_split.groups[i].image_key for i in indices}
        train_keys,report_keys,old_keys,probe_keys=(keys(split[k]) for k in ['training','report','old_holdout','frozen_probes'])
        g.require(len(report_keys)==prep['report_size'] and not report_keys&(train_keys|old_keys|probe_keys) and probe_keys<=train_keys,'Image-level report overlap')
        ids=[hashlib.sha256(r['image_key'].encode()).hexdigest() for r in records]
        row_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        roles=dict(report=records,train_source_indices=split['training'],row_order_sha256=row_hash,
            image_key_sha256=ids,disjoint_by_image_key=True,historically_unused_images_claimed=False,original_validation_not_evaluated=True)
        expected=ref['seeds'][str(seed)]
        g.require(reset.digest(roles)==expected['data_roles.json'],'Historical dataset identities differ')
        del report_batches,records,roles
        with tempfile.TemporaryDirectory(prefix='d0_identity_') as scratch:
            fixed_probe_batches(data,gate_protocol(),Path(scratch)) # construction only; no model or gradient call
            probe_identity=json.loads((Path(scratch)/'training_batches.json').read_text())
            g.require(reset.digest(probe_identity)==expected['training_batches.json'],'Historical probe-data construction differs')
        del probe_identity
        model=d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
        trainable=d.configure_clip_trainable_parameters(model,config['model']['trainable_policy'])
        g.require((trainable['trainable_parameter_count'],trainable['trainable_tensor_count'])==(10895617,35),'Trainable policy differs')
        optimizer=d.build_clip_optimizer(model,config['optimizer'])
        scheduler=clip_train.build_scheduler(optimizer,warmup_steps=config['scheduler']['warmup_steps'],total_steps=prep['scheduler_horizon'])
        for name,value in [('training_config.json',config),('device.json',info),('trainable_policy.json',trainable)]:d.write_json(folder/name,value)
        sync()
        def commit(label,value):
            if label.startswith('start_'):
                g.require(value==expected['starts'][str(value['completed_updates'])],'Native starting state replay mismatch')
            d.write_json(folder/(label+'.json'),value);sync()
            if label=='progress':print('D0_PREPARATION:',seed,value['completed_updates'],flush=True)
        supplied=load_snapshots(snapshot_dir,seed,p['checkpoint_steps'],plan['seeds'][str(seed)])
        if supplied is None:
            states,trunk_audit=native.trunk(model,optimizer,scheduler,training_batches(data),config,prep,seed=seed,commit=commit)
            actual_updates=trunk_audit['actual_optimizer_steps']
        else:
            states,trunk_audit=supplied;actual_updates=0
        authenticate_states(states,plan['seeds'][str(seed)],p['checkpoint_steps'],trunk_audit)
        # Keep a self-contained authenticated lineage even when snapshots were supplied.
        for w in p['checkpoint_steps']:
            d.write_json(folder/f'start_{w:06d}.json',expected['starts'][str(w)])
        preparation_updates+=actual_updates
        preparation_records.append(dict(training_seed=seed,actual_native_preparation_updates=actual_updates,
            loaded_snapshots=supplied is not None,trunk_replay_verified=True))
        d.write_json(folder/'trunk_audit.json',trunk_audit)
        d.write_json(folder/'preparation.json',preparation_records[-1]);sync()
        source_digest=d.state_digest(states)
        common=states[min(p['checkpoint_steps'])]
        groups=[('continuation',stream,plan['continuation_batches'][f'{seed}:{stream}']) for stream in p['stream_seeds']]
        groups.append(('calibration_feasibility',None,plan['calibration_batches'][str(seed)]))
        for population,stream,indices in groups:
            flat=[i for batch in indices for i in batch]
            if stream is None:
                g.require(len(keys(flat))==len(flat) and not keys(flat)&(report_keys|old_keys|probe_keys|
                    keys([i for u in p['stream_seeds'] for batch in plan['continuation_batches'][f'{seed}:{u}'] for i in batch])), 'Calibration image overlap')
            with g.preserved(model,optimizer,scheduler):
                batches,identities=d.build_batches(data,flat,seed=seed,epoch=common['epoch'],canonical=False)
            g.require(len(batches)==len(indices) and len(identities)==len(flat),'Incorrect prepared audit batch count')
            if stream is None:
                identity=dict(records=identities,caption_epoch=common['epoch'],source_indices=flat,
                    disjoint_from_report_probes_and_continuation_by_image_key=True)
                wanted=plan['seeds'][str(seed)]['calibration_batch_records_canonical_sha256']
            else:
                identity=dict(records=identities,caption_epoch=common['epoch'])
                wanted=plan['seeds'][str(seed)]['streams'][str(stream)]['training_batch_records_sha256']
            g.require(reset.digest(identity)==wanted,'Processed batch/caption identity differs')
            group_dir=folder/(f'stream_{stream}' if stream is not None else 'calibration');group_dir.mkdir()
            d.write_json(group_dir/'batch_identity.json',dict(canonical_sha256=wanted,caption_epoch=common['epoch'],
                source_indices=flat,batches=len(indices),captions_omitted=True))
            sync()
            for offset,batch in enumerate(batches):
                records=identities[offset*p['batch_size']:(offset+1)*p['batch_size']]
                pair={}
                for w in p['checkpoint_steps']:
                    d.restore(model,optimizer,scheduler,states[w])
                    g.require(d.state_digest(model.state_dict())==plan['seeds'][str(seed)]['states'][str(w)]['weights_sha256'],'Restored weights differ')
                    context=dict(population=population,training_seed=seed,stream_seed=stream,batch_offset=offset,weight_step=w)
                    try:
                        cached=recovery.get(context,records)
                        if cached is None:
                            rows,audit=g.audit_batch(model,batch,records,context,common,config,p,optimizer,scheduler)
                        else:
                            rows,audit=cached
                        before=g.stamp(model,optimizer,scheduler)
                        base=f'w{w}_batch{offset:03d}'
                        info=write_gzip(group_dir/(base+'.jsonl.gz'),rows,lines=True)
                        local=Accumulator(p,[(seed,stream)])
                        for row in rows:local.add(row)
                        write_gzip(group_dir/(base+'_summary.json.gz'),local.finish())
                        saved_audit={**context,**audit,'queries':len(rows),'records':info}
                        d.write_json(group_dir/(base+'_audit.json'),saved_audit)
                        d.write_json(group_dir/(base+'_origin.json'),dict(reused=cached is not None,
                            source_run=None if cached is None else recovery.entries[tuple(context[n] for n in ['training_seed','population','stream_seed','batch_offset','weight_step'])]['run_id']))
                        sync()
                        g.require(g.stamp(model,optimizer,scheduler)==before,'Persistence changed audit state')
                        shards.append(dict(**context,records=str((group_dir/info['path']).relative_to(directory)),sha256=info['sha256'],queries=len(rows)))
                        pair[w]=rows
                    except BaseException as error:
                        failure=dict(**context,error=f'{type(error).__name__}: {error}',record=getattr(error,'record',None))
                        d.write_json(group_dir/f'failure_w{w}_batch{offset:03d}.json',failure)
                        sync();raise
                summary.add_batch_pair(pair)
                print('D0_RECOVERY:',seed,population,stream,offset+1,'/',len(batches),
                    'reused',len(recovery.used),'new',recovery.new_forwards,flush=True)
            del batches,identities,identity
        g.require(d.state_digest(states)==source_digest,'Audit changed native snapshots')
        del states,common,model,optimizer,scheduler,data,trunk_audit
        gc.collect();torch.cuda.empty_cache()
    g.require(preparation_updates<=p['budget']['native_preparation_updates_if_all_states_regenerated'],'Preparation budget exceeded')
    recovery.finish(directory,preparation_updates)
    result=summary.finish()
    d.write_json(directory/'summary.json',result)
    (directory/'REPORT.md').write_text(markdown_report(result))
    d.write_json(directory/'shards.json',shards)
    d.write_json(directory/'completion.json',dict(status='complete',experiment=p['experiment'],
        batch_forwards=summary.batch_count,query_exposures=summary.record_count,
        actual_native_preparation_updates=preparation_updates,preparation=preparation_records,
        audit_optimizer_updates=0,audit_backward_passes=0,audit_scheduler_steps=0,virtual_optimizer_steps=0,
        reporting_forward_passes=0,state_preservation_verified=True,native_states_authenticated=True,
        d1_protocol_frozen=False,all_query_candidate_sizes=result['all_query_candidate_sizes'],
        treatment_infeasibility_is_not_a_failed_audit=True))
    sync()
    return result
