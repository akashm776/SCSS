"""Native-only history variant. The earlier policy-replication sources stay immutable."""
import gc
import hashlib
import json
from pathlib import Path
import statistics

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src import clip_fresh_split_policy as parent
from src.clip_fresh_split_policy import split_indices, training_batches, FullPoolReport
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol


def load_protocol():
    p = json.loads((d.ROOT/'configs/clip_native_history_policy.json').read_text())
    expected = {**parent.load_protocol(),
        'experiment':'clip_native_history_policy_v1', 'trunk_gate_refresh_steps':[],
        'starting_state_history':'native_only', 'trunk_coefficient':0,
        'trunk_probes':False,
        'secondary_contrast':'sustained_minus_native_full512_loss_at_50_by_starting_step',
        'parent_experiment':'clip_fresh_split_policy_v1',
        'parent_bundle_id':'f84d5fd0dd5ec4ba3c47758bdf4e08ed43b31d92dbcc1d8c8ff3efb71a981a81',
        'shared_design_with_parent':True}
    if p != expected:
        raise ValueError('Frozen native-history protocol changed')
    return p


def snapshot(model,optimizer,scheduler,*,epoch,step,seed):
    state = parent.snapshot(model,optimizer,scheduler,epoch=epoch,step=step,seed=seed)
    state.update(arm='baseline',starting_state_history='native_only',prior_auxiliary_updates=0)
    return state


def trunk(model,optimizer,scheduler,batch_epochs,config,p,*,seed,commit):
    """No gate/report objects: every starting-state update is native-only."""
    if (p['starting_state_history'] != 'native_only' or p['trunk_coefficient'] != 0
            or p['trunk_probes'] or p['trunk_gate_refresh_steps']
            or not p['checkpoint_steps'] or min(p['checkpoint_steps']) < 1
            or max(p['checkpoint_steps']) > p['trunk_updates']):
        raise ValueError('Native-only trunk contract violated before updates')
    snapshots,traces = {},[]
    frozen = d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad})
    model.train()
    policy.guarded(model,optimizer,scheduler,lambda:commit('native_history_contract',
        dict(history='native_only',auxiliary_coefficient=0.,probe_calls=0,
             updates=p['trunk_updates'],checkpoint_steps=p['checkpoint_steps'])))
    for completed in range(p['trunk_updates']):
        epoch,batch = next(batch_epochs)
        trace = d.step_update(model,optimizer,scheduler,batch,config,0.)
        if trace['coefficient'] != 0.:
            raise AssertionError('Non-native starting-state update')
        trace.update(epoch=epoch,completed_updates=completed+1,
            batch_identity_sha256=d.state_digest(dict(source_indices=batch['source_indices'],captions=batch['captions'])))
        traces.append(trace)
        if completed+1 in p['checkpoint_steps']:
            state = snapshot(model,optimizer,scheduler,epoch=epoch,step=completed+1,seed=seed)
            snapshots[completed+1] = state
            record = dict(training_seed=seed,completed_updates=completed+1,epoch=epoch,
                arm='baseline',starting_state_history='native_only',prior_auxiliary_updates=0,
                state_sha256=d.live_digest(model,optimizer,scheduler),snapshot_sha256=d.state_digest(state))
            policy.guarded(model,optimizer,scheduler,lambda:commit(f'start_{completed+1:06d}',record))
        if (completed+1)%50 == 0:
            policy.guarded(model,optimizer,scheduler,lambda:commit('progress',
                dict(completed_updates=completed+1,auxiliary_updates=0)))
    if frozen != d.state_digest({n:v for n,v in model.named_parameters() if not v.requires_grad}):
        raise AssertionError('Frozen trunk parameters changed')
    return snapshots,dict(actual_optimizer_steps=len(traces),traces=traces,decisions=[],
        auxiliary_updates=0,probe_calls=0,reporting_access=False,
        frozen_parameters_unchanged=True,history='native_only')


def summarize(rows,p):
    result = parent.summarize(rows,p)
    if any(r['starting_state_history'] != 'native_only' or r['prior_auxiliary_updates'] != 0 for r in rows):
        raise AssertionError('Non-native history in continuation results')
    horizon = str(p['horizon'])
    per_seed = []
    for seed in p['training_seeds']:
        by_step = {}
        for step in p['checkpoint_steps']:
            group = [r for r in rows if (r['training_seed'],r['checkpoint_step']) == (seed,step)]
            by_step[str(step)] = dict(report_loss=statistics.fmean(
                r['points']['sustained'][horizon]['report']['native_loss'] -
                r['points']['native'][horizon]['report']['native_loss'] for r in group))
            by_step[str(step)]['pool_r1_pp'] = statistics.fmean(
                r['points']['sustained'][horizon]['report']['pool_retrieval']['mean_r1_percent'] -
                r['points']['native'][horizon]['report']['pool_retrieval']['mean_r1_percent'] for r in group)
        early,later = map(str,p['checkpoint_steps'])
        per_seed.append(dict(training_seed=seed,per_starting_step=by_step,
            later_minus_early={metric:by_step[later][metric]-by_step[early][metric]
                              for metric in ['report_loss','pool_r1_pp']}))
    result['sustained_minus_native'] = dict(
        per_seed=per_seed,
        per_starting_step={str(step):{metric:statistics.fmean(
            row['per_starting_step'][str(step)][metric] for row in per_seed)
            for metric in ['report_loss','pool_r1_pp']} for step in p['checkpoint_steps']},
        later_minus_early={metric:statistics.fmean(row['later_minus_early'][metric] for row in per_seed)
                          for metric in ['report_loss','pool_r1_pp']},
        interpretation='Secondary, sustained minus native at horizon 50. Negative loss helps; positive R1 helps. '
        'A positive later-minus-early loss contrast alone does not establish later harm: show both absolute effects.')
    result.update(starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        shared_design_with_parent=True,
        interpretation='Primary remains gated-minus-random-matched full512 loss at 50. '
        'All starting states have native-only history. Report sustained-minus-native separately at both ages. '
        'Three seeds, not twelve independent sets. Age still co-varies with model state, optimizer and LR. '
        'Same seeds, split and streams as the unrun gated-history variant: do not count the variants as independent replications. '
        'Reporting images are newly excluded for these experiments, not external or historically untouched data. No refitting.')
    return result


def run(directory,sync,p=None):
    from transformers import AutoProcessor
    p = load_protocol() if p is None else p
    directory = Path(directory)
    old = json.loads((d.ROOT/'configs/cub200_clip_diagnostic_holdout_indices.json').read_text())
    probe_layout = json.loads((d.ROOT/'experiment_results/clip_paired_updates_2026-09/training_source_indices.json').read_text())
    probes = [i for b in probe_layout for i in b]
    if len(old) != p['old_holdout_size'] or len(set(probes)) != p['frozen_probe_size']:
        raise AssertionError('Historical identities changed')
    split = split_indices(list(range(p['original_train_size'])),old,probes,p)
    if len(split['training']) != p['new_training_size']:
        raise AssertionError('Unexpected training population')
    exclusion = directory/'training_exclusion_indices.json'
    d.write_json(exclusion,sorted(set(old)|set(split['report'])))
    d.write_json(directory/'protocol.json',p)
    d.write_json(directory/'split_plan.json',split)
    plans,permutations = policy.make_plans(split['training'],p)
    d.write_json(directory/'branch_plans.json',plans)
    d.write_json(directory/'timing_permutations.json',permutations)
    sync()  # Source-index split and schedules committed before any model update.
    rows, trunk_steps = [],0
    for seed in p['training_seeds']:
        clip_train.set_global_seed(seed)
        folder = directory/f'seed_{seed}'
        folder.mkdir(exist_ok=False)
        config = clip_train.load_training_config(d.ROOT/'configs/hf_cub200_clip_vit_b32_baseline.yaml')
        config['experiment']['seed'] = config['training']['seed'] = seed
        config['runtime']['num_workers'] = 0
        config['dataset']['diagnostic_holdout_indices'] = str(exclusion.resolve())
        device,info = clip_train.resolve_device(config['runtime'])
        processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
        data = clip_train.build_clip_training_data(config,processor,pin_memory=False)
        if data.train_dataset.source_indices != split['training'] or len(data.train_loader) != p['expected_batches_per_epoch']:
            raise AssertionError('Training exclusions/batch count differ')
        report_batches,records = d.build_batches(data,split['report'],seed=0,epoch=0,canonical=True)
        keys = lambda indices: {data.train_dataset.grouped_split.groups[i].image_key for i in indices}
        train_keys,report_keys,old_keys,probe_keys = (keys(split[k]) for k in ['training','report','old_holdout','frozen_probes'])
        if len(report_keys) != p['report_size'] or report_keys & (train_keys|old_keys|probe_keys) or not probe_keys <= train_keys:
            raise AssertionError('Image-level report overlap')
        ids = [hashlib.sha256(r['image_key'].encode()).hexdigest() for r in records]
        row_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if rows and row_hash != common_row_hash:
            raise AssertionError('Reporting identities differ across seeds')
        common_row_hash = row_hash
        d.write_json(folder/'data_roles.json',dict(report=records,train_source_indices=split['training'],
            row_order_sha256=row_hash,image_key_sha256=ids,disjoint_by_image_key=True,
            historically_unused_images_claimed=False,original_validation_not_evaluated=True))
        d.write_json(folder/'training_config.json',config)
        d.write_json(folder/'device.json',info)
        probe = policy.AlignmentGate(fixed_probe_batches(data,gate_protocol(),folder),config,gate_protocol())
        model = d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
        trainable = d.configure_clip_trainable_parameters(model,config['model']['trainable_policy'])
        if (trainable['trainable_parameter_count'],trainable['trainable_tensor_count']) != (10895617,35):
            raise AssertionError('Trainable policy changed')
        d.write_json(folder/'trainable_policy.json',trainable)
        optimizer = d.build_clip_optimizer(model,config['optimizer'])
        scheduler = clip_train.build_scheduler(optimizer,warmup_steps=config['scheduler']['warmup_steps'],total_steps=p['scheduler_horizon'])
        sync()
        def commit(label,value):
            d.write_json(folder/(label+'.json'),value)
            sync()
            if label == 'progress':
                print('TRUNK:',seed,value['completed_updates'],flush=True)
        states,trunk_audit = trunk(model,optimizer,scheduler,training_batches(data),config,p,seed=seed,commit=commit)
        if (trunk_audit['auxiliary_updates'] != 0 or trunk_audit['decisions'] or trunk_audit['probe_calls'] != 0
                or any(t['coefficient'] != 0. for t in trunk_audit['traces'])
                or trunk_audit['actual_optimizer_steps'] != p['trunk_updates']):
            raise AssertionError('Starting states must have strictly native-only history')
        trunk_steps += trunk_audit['actual_optimizer_steps']
        d.write_json(folder/'trunk_audit.json',trunk_audit)
        report = FullPoolReport(report_batches,p['report_size'])
        for step in p['checkpoint_steps']:
            initial = states[step]
            if (initial['arm'] != 'baseline' or initial['starting_state_history'] != 'native_only'
                    or initial['prior_auxiliary_updates'] != 0 or initial['completed_updates'] != step
                    or initial['scheduler']['last_epoch'] != step):
                raise AssertionError('Starting-state identity/scheduler/history differs')
            for stream in p['stream_seeds']:
                key = f'{seed}:{step}:{stream}'
                pair_dir = folder/f'step_{step:06d}'/f'stream_{stream}'
                pair_dir.mkdir(parents=True,exist_ok=False)
                batches,batch_records = d.build_batches(data,[i for b in plans[key] for i in b],seed=seed,epoch=initial['epoch'],canonical=False)
                d.write_json(pair_dir/'training_batches.json',dict(records=batch_records,caption_epoch=initial['epoch']))
                sync()
                def pair_commit(label,value):
                    d.write_json(pair_dir/(label+'.json'),value)
                    sync()
                def save_point(arm,offset,value):
                    d.write_json(pair_dir/f'point_{arm}_{offset:03d}.json',value)
                    if offset == p['horizon']:
                        meta = cache.save_cache(pair_dir/f'features_{arm}.npz',report.encoded,row_hash)
                        if meta['feature_sha256'] != value['report']['features_sha256']:
                            raise AssertionError('Endpoint cache differs from evaluated state')
                    sync()
                result = policy.run_pair(model,optimizer,scheduler,initial,batches,config,probe,report,
                    coefficient=p['coefficient'],threshold=p['alignment_threshold'],refresh_offsets=p['gate_refresh_offsets'],
                    observation_steps=p['observation_steps'],permutation=permutations[key],commit=pair_commit,save_point=save_point,
                    progress=lambda text:print('NATIVE_HISTORY_PAIR:',key,text,flush=True))
                result.update(training_seed=seed,checkpoint_step=step,stream_seed=stream,report_row_order_sha256=row_hash,
                    starting_state_history='native_only',prior_auxiliary_updates=0,
                    source_state_sha256=d.state_digest(initial))
                d.write_json(pair_dir/'result.json',result)
                rows.append(result)
                sync()
                del batches,batch_records
                gc.collect()
        del states,initial,model,optimizer,scheduler,data,probe,report,report_batches
        gc.collect()
        torch.cuda.empty_cache()
    if trunk_steps != p['trunk_optimizer_steps']:
        raise AssertionError('Trunk budget differs')
    summary = summarize(rows,p)
    d.write_json(directory/'comparison.json',summary)
    d.write_json(directory/'completion.json',dict(status='complete',pairs=12,branches=48,endpoint_caches=48,
        trunk_optimizer_steps=trunk_steps,branch_optimizer_steps=summary['branch_optimizer_steps'],optimizer_steps=summary['optimizer_steps'],
        rules_refitted=False,new_training_seeds=True,independent_test_set=False,report_excluded_from_training_and_probes=True,
        starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        shared_design_with_parent=True))
    sync()
    return summary
