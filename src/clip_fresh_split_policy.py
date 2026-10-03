"""Prospective fresh-seed / held-out-split test within previously studied CUB."""
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol


def load_protocol():
    p = json.loads((d.ROOT/'configs/clip_fresh_split_policy.json').read_text())
    gate = gate_protocol()
    fixed = dict(experiment='clip_fresh_split_policy_v1',
        previous_stream_seeds=[2026092402,2026092601,2026092602,2026093002,2026093003],
        original_train_size=5994,old_holdout_size=1024,frozen_probe_size=1024,
        expected_batches_per_epoch=69,
        primary='seed_mean_gated_minus_random_matched_full512_loss_at_50',
        new_training_seeds=True,fresh_reporting_split_for_this_run=True,
        new_external_dataset=False,literal_training_loader_resume=False,exploratory_followup=True)
    if any(p.get(k) != v for k,v in fixed.items()):
        raise ValueError('Frozen fresh-split policy protocol changed')
    if (p['training_seeds'] != [271828,161803,141421] or p['checkpoint_steps'] != [100,500]
            or p['stream_seeds'] != [2026100101,2026100102] or p['arms'] != policy.ARMS
            or p['horizon'] != 50 or p['batch_size'] != 64 or p['report_size'] != 512
            or p['coefficient'] != gate['uniform_top8_coefficient']
            or p['alignment_threshold'] != gate['alignment_threshold']
            or p['trunk_gate_refresh_steps'] != [100,250] or p['trunk_updates'] != 500
            or p['gate_refresh_offsets'] != [0,10,25] or p['observation_steps'] != [0,50]
            or p['scheduler_horizon'] != 3850 or p['report_split_seed'] != 2026100104
            or p['permutation_seed'] != 2026100103 or p['new_training_size'] != 4458
            or p['optimizer_step_budget'] != 3900 or p['trunk_optimizer_steps'] != 1500
            or p['branch_optimizer_steps'] != 2400 or p['rules_refitted'] or p['report_controls_treatment']
            or p['independent_test_set'] or p['historically_unused_images_claimed']):
        raise ValueError('Frozen fresh-split policy protocol changed')
    return p


def split_indices(population, old_holdout, probe_indices, p):
    if len(set(population)) != len(population) or len(set(old_holdout)) != len(old_holdout):
        raise ValueError('Duplicate source identities')
    population, old, probes = set(population), set(old_holdout), set(probe_indices)
    if not old <= population or not probes <= population or old & probes:
        raise ValueError('Invalid historical exclusions/probes')
    candidates = population-old-probes
    if len(candidates) < p['report_size']:
        raise ValueError('Insufficient fresh reporting candidates')
    report = d.ranked(candidates,f"fresh-report-v1:{p['report_split_seed']}")[:p['report_size']]
    training = sorted(population-old-set(report))
    if set(report) & (set(training)|old|probes) or not probes <= set(training):
        raise AssertionError('Report leakage into training/probes')
    return dict(training=training, report=report, old_holdout=sorted(old), frozen_probes=sorted(probes))


def cpu_copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def snapshot(model, optimizer, scheduler, *, epoch, step, seed):
    return cpu_copy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        **d.rng_state(),epoch=epoch,completed_updates=step,training_seed=seed,
        arm='baseline' if step==100 else 'alignment_gated'))


def trunk(model, optimizer, scheduler, batch_epochs, config, probe, p, *, seed, commit):
    """No report object or reporting batches enter starting-state generation."""
    snapshots, traces, decisions = {}, [], []
    active = False
    model.train()
    for completed in range(p['trunk_updates']):
        if completed in p['trunk_gate_refresh_steps']:
            measured = policy.guarded(model,optimizer,scheduler,lambda:probe.measure(model))
            score = measured['mean_cosine']
            if not math.isfinite(score):
                raise ValueError('Nonfinite trunk gate score')
            active = score > p['alignment_threshold']
            decision=dict(completed_updates=completed,measurement=measured,active=active)
            decisions.append(decision)
            policy.guarded(model,optimizer,scheduler,lambda:commit(f'gate_{completed:06d}',decision))
        epoch,batch = next(batch_epochs)
        trace = d.step_update(model,optimizer,scheduler,batch,config,p['coefficient'] if active else 0.)
        trace.update(epoch=epoch,completed_updates=completed+1,
            batch_identity_sha256=d.state_digest(dict(source_indices=batch['source_indices'],captions=batch['captions'])))
        traces.append(trace)
        if completed+1 in p['checkpoint_steps']:
            state = snapshot(model,optimizer,scheduler,epoch=epoch,step=completed+1,seed=seed)
            snapshots[completed+1] = state
            record=dict(training_seed=seed,completed_updates=completed+1,epoch=epoch,
                state_sha256=d.live_digest(model,optimizer,scheduler),snapshot_sha256=d.state_digest(state))
            policy.guarded(model,optimizer,scheduler,lambda:commit(f'start_{completed+1:06d}',record))
        if (completed+1)%50 == 0:
            policy.guarded(model,optimizer,scheduler,lambda:commit('progress',dict(completed_updates=completed+1)))
    return snapshots,dict(actual_optimizer_steps=len(traces),traces=traces,decisions=decisions,
                          reporting_access=False,history='native through 100, frozen alignment gate thereafter')


def training_batches(data):
    epoch = 1
    while True:
        data.train_dataset.set_epoch(epoch)
        for batch in data.train_loader:
            yield epoch,batch
        epoch += 1


class FullPoolReport:
    def __init__(self,batches,size):
        self.batches,self.size = batches,size
        self.encoded = None

    def evaluate(self,model):
        with d.observe(model):
            self.encoded = d.encode_cached(model,self.batches)
        cache.validate_encoded(self.encoded,self.size)
        evaluated = cache.evaluate_encoded(self.encoded,{'full512':{'batches':[list(range(self.size))]}})
        return dict(native_loss=evaluated['full_pool_loss'],partition_losses=evaluated['partition_losses'],
            pool_retrieval=evaluated['pool_retrieval'],features_sha256=evaluated['features_sha256'],
            learned_logit_scale=evaluated['learned_logit_scale'])


def summarize(rows,p):
    result = policy.summarize(rows,p)
    # Reuse arithmetic, not the old experiment's provenance/interpretation.
    result.update(trunk_optimizer_steps=p['trunk_optimizer_steps'],branch_optimizer_steps=result.pop('optimizer_steps'),
        optimizer_steps=p['trunk_optimizer_steps']+sum(r['audit']['actual_optimizer_steps'] for r in rows),
        independent_test_set=False,new_training_seeds=True,fresh_reporting_split_for_this_run=True,
        interpretation='Gated minus control; negative full512 loss helps and positive R1 helps. '
        'Average two streams within starting states, two states within seeds, three seeds equally. '
        'Prospective fixed rule on new seeds and a newly held-out CUB split, not new external data or '
        'historically untouched images. Training population reduced to 4458. Do not pool with old runs '
        'as exact replications, claim statistical significance from streams, or tune after outcomes.')
    if result['optimizer_steps'] != p['optimizer_step_budget']:
        raise AssertionError('Optimizer budget differs')
    result['comparisons'] = {control: {metric: statistics.fmean(
        statistics.fmean(s['per_starting_step'][str(k)][str(p['horizon'])][control][metric]
                         for k in p['checkpoint_steps']) for s in result['per_seed'])
        for metric in ['report_loss','pool_r1_pp']} for control in policy.ARMS[1:]}
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
        states,trunk_audit = trunk(model,optimizer,scheduler,training_batches(data),config,probe,p,seed=seed,commit=commit)
        trunk_steps += trunk_audit['actual_optimizer_steps']
        d.write_json(folder/'trunk_audit.json',trunk_audit)
        report = FullPoolReport(report_batches,p['report_size'])
        for step in p['checkpoint_steps']:
            initial = states[step]
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
                    progress=lambda text:print('FRESH_PAIR:',key,text,flush=True))
                result.update(training_seed=seed,checkpoint_step=step,stream_seed=stream,report_row_order_sha256=row_hash)
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
        rules_refitted=False,new_training_seeds=True,independent_test_set=False,report_excluded_from_training_and_probes=True))
    sync()
    return summary
