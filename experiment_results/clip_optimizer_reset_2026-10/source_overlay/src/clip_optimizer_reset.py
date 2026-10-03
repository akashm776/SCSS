"""Matched AdamW-state reset, retaining weights, external LR schedule and streams."""
import gc
import hashlib
import json
from pathlib import Path
import statistics

import torch

from src import clip_native_history_policy as native
from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src.clip_fresh_split_policy import split_indices, training_batches, FullPoolReport
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol

ARMS = ['retained_native','retained_sustained','reset_native','reset_sustained']
REFERENCE_SHA256 = '8ac9fd614bb18c148f647b404c3a8536d0c4d62129fa200ef83657d5397c3e8b'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def load_protocol():
    p = json.loads((d.ROOT/'configs/clip_optimizer_reset.json').read_text())
    expected = {**native.load_protocol(), 'experiment':'clip_optimizer_reset_v1','arms':ARMS,
        'observation_steps':[0,1,5,10,25,50],'gate_refresh_offsets':[],
        'primary':'seed_mean_reset_minus_retained_sustained_effect_full512_loss_at_start500_horizon50',
        'primary_start':500,'secondary_contrast':'reset_by_age_timecourse',
        'parent_experiment':'clip_native_history_policy_v1',
        'parent_bundle_id':'db7f1a789e724f68b0a14a550ba12dd86f80b4458b1e14e9218646ee6e372914',
        'new_training_seeds':False,'fresh_reporting_split_for_this_run':False,
        'reset_moments':True,'reset_parameter_step_counters':True,'reset_scheduler':False,'reset_weights':False,
        'retained_replay_required':True,'optimizer_state_reset_not_momentum_only':True,
        'reference_sha256':REFERENCE_SHA256}
    if p != expected:
        raise ValueError('Frozen optimizer-reset protocol changed')
    return p


def load_reference():
    path = d.ROOT/'configs/clip_optimizer_reset_reference.json'
    if d.checksum(path) != REFERENCE_SHA256:
        raise ValueError('Pinned replay reference changed')
    return json.loads(path.read_text())


def reset_optimizer_state(model,optimizer,scheduler):
    """Remove all per-parameter moments/counters; retain param groups and scheduler."""
    if type(optimizer) is not torch.optim.AdamW or not optimizer.state:
        raise ValueError('Reset requires a populated AdamW optimizer')
    parameters = {p for group in optimizer.param_groups for p in group['params']}
    if set(optimizer.state) != parameters:
        raise ValueError('Missing or unexpected per-parameter AdamW state')
    counts = []
    for state in optimizer.state.values():
        if set(state) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Unexpected AdamW state layout (AMSGrad is not in this protocol)')
        counts.append(float(state['step']))
        if any(not torch.isfinite(state[k]).all() for k in state) or counts[-1] <= 0:
            raise ValueError('Invalid AdamW state')
    stamp = lambda:d.state_digest(dict(model=model.state_dict(),scheduler=scheduler.state_dict(),
        param_groups=optimizer.state_dict()['param_groups'],rng=d.rng_state(),
        gradients=[p.grad for p in model.parameters()]))
    invariant = stamp()
    before = d.state_digest(optimizer.state_dict())
    optimizer.state.clear()  # AdamW lazily initializes zero moments and step=0 at its next update.
    if stamp() != invariant or optimizer.state:
        raise AssertionError('Reset changed weights, scheduler, groups, RNG or gradients')
    return dict(optimizer_before_sha256=before,optimizer_after_sha256=d.state_digest(optimizer.state_dict()),
        cleared_parameter_states=len(parameters),old_step_counts=sorted(set(counts)),
        new_parameter_states=0,weights_scheduler_groups_rng_gradients_unchanged=True,
        lazy_initialization_next_update_step=1)


def run_pair(model,optimizer,scheduler,initial,batches,config,report,reference,*,
             coefficient,observation_steps,commit,save_point=lambda *_:None,progress=lambda _:None):
    horizon = len(batches)
    if (not batches or observation_steps != sorted(set(observation_steps))
            or observation_steps[0] != 0 or observation_steps[-1] != horizon
            or not 0 < coefficient < float('inf') or set(reference['arms']) != {'native','sustained'}):
        raise ValueError('Invalid optimizer-reset pair protocol/reference')
    source_hash = d.state_digest(initial)
    frozen = d.state_digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    points,traces,starts,resets = {},{},{},{}
    initial_report = None
    for arm in ARMS:
        d.restore(model,optimizer,scheduler,initial)
        original_start = d.live_digest(model,optimizer,scheduler)
        control = arm.split('_',1)[1]
        expected = reference['arms'][control]
        if original_start != expected['points']['0']['state_sha256']:
            raise AssertionError('Starting state does not reproduce native-history run')
        if arm.startswith('reset_'):
            resets[arm] = reset_optimizer_state(model,optimizer,scheduler)
        starts[arm] = dict(state_sha256=d.live_digest(model,optimizer,scheduler),
            weights_sha256=d.state_digest(model.state_dict()),scheduler_sha256=d.state_digest(scheduler.state_dict()),
            rng_sha256=d.state_digest(d.rng_state()),param_groups_sha256=d.state_digest(optimizer.state_dict()['param_groups']))
        for key in ['weights_sha256','scheduler_sha256','rng_sha256','param_groups_sha256']:
            if starts[arm][key] != starts[ARMS[0]][key]:
                raise AssertionError('Unmatched branch '+key)
        policy.guarded(model,optimizer,scheduler,lambda:commit('start_'+arm,
            dict(start=starts[arm],reset=resets.get(arm),coefficient=coefficient if control=='sustained' else 0.)))
        points[arm],traces[arm] = {},[]
        for offset in range(horizon+1):
            if offset in observation_steps:
                evaluated = policy.guarded(model,optimizer,scheduler,lambda:report.evaluate(model))
                if offset == 0:
                    if initial_report is None:
                        initial_report = evaluated
                    if evaluated != initial_report or evaluated != expected['points']['0']['report']:
                        raise AssertionError('No-update report mismatch after reset/restore')
                point = dict(report=evaluated,state_sha256=d.live_digest(model,optimizer,scheduler))
                if arm.startswith('retained_') and offset in [0,horizon]:
                    if point != expected['points'][str(offset)]:
                        raise AssertionError('Retained branch endpoint replay mismatch; no reset inference accepted')
                points[arm][str(offset)] = point
                policy.guarded(model,optimizer,scheduler,lambda:save_point(arm,offset,point))
                policy.guarded(model,optimizer,scheduler,lambda:progress(f'{arm}: {offset}/{horizon}'))
            if offset == horizon:
                break
            trace = d.step_update(model,optimizer,scheduler,batches[offset],config,
                coefficient if control=='sustained' else 0.)
            if arm != ARMS[0] and any(trace[k] != traces[ARMS[0]][offset][k]
                    for k in ['learning_rates','rng_after_sha256']):
                raise AssertionError('LR/RNG differs between reset/retained/treatment branches')
            expected_count = offset+1 if arm.startswith('reset_') else initial['completed_updates']+offset+1
            if any(float(s['step']) != expected_count for s in optimizer.state.values()):
                raise AssertionError('AdamW parameter step counter mismatch')
            traces[arm].append(trace)
        if arm.startswith('retained_') and digest(traces[arm]) != expected['traces_sha256']:
            raise AssertionError('Retained update trace replay mismatch')
    if (d.state_digest(initial) != source_hash or frozen != d.state_digest(
            {n:p for n,p in model.named_parameters() if not p.requires_grad})):
        raise AssertionError('Source snapshot or frozen parameters changed')
    return dict(points=points,traces=traces,starts=starts,resets=resets,
        audit=dict(actual_optimizer_steps=4*horizon,retained_exact_replay=True,source_immutable=True,
            frozen_parameters_unchanged=True,matched_rng_and_lrs=True,no_update_reports_identical=True,
            reset_only_optimizer_state=True,observations_preserve_state=True,report_controls_treatment=False))


def contrasts(row,offset):
    q = str(offset)
    effects = {}
    for state in ['retained','reset']:
        t,n = (row['points'][state+'_'+a][q]['report'] for a in ['sustained','native'])
        effects[state] = dict(loss=t['native_loss']-n['native_loss'],
            r1_pp=t['pool_retrieval']['mean_r1_percent']-n['pool_retrieval']['mean_r1_percent'])
    effects['interaction'] = {k:effects['reset'][k]-effects['retained'][k] for k in effects['reset']}
    return effects


def summarize(rows,p):
    grid = {(r['training_seed'],r['checkpoint_step'],r['stream_seed']):r for r in rows}
    expected = {(s,k,u) for s in p['training_seeds'] for k in p['checkpoint_steps'] for u in p['stream_seeds']}
    if len(grid) != len(rows) or set(grid) != expected:
        raise ValueError('Incomplete or duplicate optimizer-reset grid')
    if any(not r['audit']['retained_exact_replay'] for r in rows):
        raise ValueError('Retained replay did not pass')
    count = sum(r['audit']['actual_optimizer_steps'] for r in rows)
    if count != p['branch_optimizer_steps'] or count+p['trunk_optimizer_steps'] != p['optimizer_step_budget']:
        raise ValueError('Optimizer-reset budget mismatch')
    per_seed = []
    for s in p['training_seeds']:
        per_seed.append(dict(seed=s,by_starting_step={str(k):{str(t):{c:{m:statistics.fmean(
            contrasts(grid[s,k,u],t)[c][m] for u in p['stream_seeds']) for m in ['loss','r1_pp']}
            for c in ['retained','reset','interaction']} for t in p['observation_steps']} for k in p['checkpoint_steps']}))
    average = {str(k):{str(t):{c:{m:statistics.fmean(s['by_starting_step'][str(k)][str(t)][c][m] for s in per_seed)
        for m in ['loss','r1_pp']} for c in ['retained','reset','interaction']}
        for t in p['observation_steps']} for k in p['checkpoint_steps']}
    return dict(primary=p['primary'],primary_value=average[str(p['primary_start'])][str(p['horizon'])]['interaction']['loss'],
        per_seed=per_seed,by_starting_step=average,pairs=len(rows),branches=4*len(rows),replication_units=len(per_seed),
        branch_optimizer_steps=count,trunk_optimizer_steps=p['trunk_optimizer_steps'],optimizer_steps=p['optimizer_step_budget'],
        paired_timecourses=[dict(seed=r['training_seed'],step=r['checkpoint_step'],stream=r['stream_seed'],
            effects={str(t):contrasts(r,t) for t in p['observation_steps']}) for r in rows],
        interpretation='Primary at start500/offset50: reset minus retained sustained-minus-native loss. Negative means reset '
        'reduces treatment harm/increases benefit. Show both absolute effects, all seeds and fixed timepoints. '
        'Reset clears first/second moments and parameter step counters, changing bias correction and effective update size; '
        'not a momentum-direction-only test. External scheduler and LR trajectory retained. '
        'Persistence does not prove representation causation. Reused seeds/streams/reporting data; exploratory, no refitting.')


def run(directory,sync,p=None,reference=None):
    from transformers import AutoProcessor
    p = load_protocol() if p is None else p
    reference = load_reference() if reference is None else reference
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
    for name,value in [('split_plan.json',split),('branch_plans.json',plans),('training_exclusion_indices.json',sorted(set(old)|set(split['report'])))]:
        if digest(value) != reference[name]:
            raise AssertionError('Replay plan differs: '+name)
    d.write_json(directory/'replay_reference.json',reference)
    sync()  # Protocol and replay reference durable before any model update.
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
        fixed_probe_batches(data,gate_protocol(),folder)  # Reproduce archived data construction, no gradient probes.
        seed_reference = reference['seeds'][str(seed)]
        for name in ['data_roles.json','training_batches.json']:
            if digest(json.loads((folder/name).read_text())) != seed_reference[name]:
                raise AssertionError('Replay identities differ: '+name)
        model = d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
        trainable = d.configure_clip_trainable_parameters(model,config['model']['trainable_policy'])
        if (trainable['trainable_parameter_count'],trainable['trainable_tensor_count']) != (10895617,35):
            raise AssertionError('Trainable policy changed')
        d.write_json(folder/'trainable_policy.json',trainable)
        optimizer = d.build_clip_optimizer(model,config['optimizer'])
        scheduler = clip_train.build_scheduler(optimizer,warmup_steps=config['scheduler']['warmup_steps'],total_steps=p['scheduler_horizon'])
        sync()
        def commit(label,value):
            if label.startswith('start_') and value != seed_reference['starts'][str(value['completed_updates'])]:
                raise AssertionError('Native starting-state replay mismatch')
            d.write_json(folder/(label+'.json'),value)
            sync()
            if label == 'progress':
                print('TRUNK:',seed,value['completed_updates'],flush=True)
        states,trunk_audit = native.trunk(model,optimizer,scheduler,training_batches(data),config,p,seed=seed,commit=commit)
        if (trunk_audit['auxiliary_updates'] != 0 or trunk_audit['decisions'] or trunk_audit['probe_calls'] != 0
                or any(t['coefficient'] != 0. for t in trunk_audit['traces'])
                or trunk_audit['actual_optimizer_steps'] != p['trunk_updates']):
            raise AssertionError('Starting states must have strictly native-only history')
        if digest(trunk_audit) != seed_reference['trunk_audit.json']:
            raise AssertionError('Native trunk audit replay mismatch')
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
                batch_identity = dict(records=batch_records,caption_epoch=initial['epoch'])
                if digest(batch_identity) != reference['pairs'][key]['training_batches_sha256']:
                    raise AssertionError('Continuation batch/caption replay mismatch')
                d.write_json(pair_dir/'training_batches.json',batch_identity)
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
                result = run_pair(model,optimizer,scheduler,initial,batches,config,report,reference['pairs'][key],
                    coefficient=p['coefficient'],observation_steps=p['observation_steps'],
                    commit=pair_commit,save_point=save_point,
                    progress=lambda text:print('OPTIMIZER_RESET_PAIR:',key,text,flush=True))
                result.update(training_seed=seed,checkpoint_step=step,stream_seed=stream,report_row_order_sha256=row_hash,
                    starting_state_history='native_only',prior_auxiliary_updates=0,
                    source_state_sha256=d.state_digest(initial))
                d.write_json(pair_dir/'result.json',result)
                rows.append(result)
                sync()
                del batches,batch_records
                gc.collect()
        del states,initial,model,optimizer,scheduler,data,report,report_batches
        gc.collect()
        torch.cuda.empty_cache()
    if trunk_steps != p['trunk_optimizer_steps']:
        raise AssertionError('Trunk budget differs')
    summary = summarize(rows,p)
    d.write_json(directory/'comparison.json',summary)
    d.write_json(directory/'completion.json',dict(status='complete',pairs=12,branches=48,endpoint_caches=48,
        trunk_optimizer_steps=trunk_steps,branch_optimizer_steps=summary['branch_optimizer_steps'],optimizer_steps=summary['optimizer_steps'],
        rules_refitted=False,new_training_seeds=False,retained_exact_replay=True,independent_test_set=False,report_excluded_from_training_and_probes=True,
        starting_state_history='native_only',trunk_auxiliary_updates=0,trunk_probe_calls=0,
        shared_design_with_parent=True,reset_scheduler=False,reset_weights=False,
        reset_moments=True,reset_parameter_step_counters=True))
    sync()
    return summary
