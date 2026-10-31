"""Automatic-resume D1 runner; issued scientific sources remain unchanged."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from src import clip_pressure as pressure
from src import clip_difficulty as b
from src import clip_checkpoint_diagnostic as d
from src import clip_weight_lr as wl
from src import clip_native_history_policy as native
from src import clip_optimizer_reset as reset
from src import clip_policy_test as policy
from src import clip_policy_pool_test as cache
from src import clip_train
from src.clip_fresh_split_policy import split_indices, training_batches, FullPoolReport
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol
from src.clip_geometry_run import authenticate_states
from src.clip_source_regime import (load_protocol, validate_population, validate_batch_identity,
    calibrate_states, branch_coefficients)
from src.clip_source_resume_training import resumable_trunk, run_branches
from src.clip_source_resume_store import ResumeMismatch
from src.clip_source_analysis import summarize, markdown_report


def run(directory,sync,p=None,native_reference=None,reference=None,*,store,legacy=False):
    from transformers import AutoProcessor
    frozen_protocol = load_protocol() if p is None else p
    inherited = b.load_protocol()
    preparation_keys = ['previous_stream_seeds','scheduler_horizon','trunk_updates','trunk_gate_refresh_steps',
        'permutation_seed','report_split_seed','report_size','original_train_size','old_holdout_size',
        'frozen_probe_size','new_training_size','expected_batches_per_epoch','starting_state_history',
        'trunk_coefficient','trunk_probes']
    p = {**{key:inherited[key] for key in preparation_keys}, **frozen_protocol,
        'common_stream_checkpoint_step': frozen_protocol['common_rng_source_step'],
        'trunk_optimizer_steps': frozen_protocol['budget']['trunk_optimizer_steps'],
        'branch_optimizer_steps': frozen_protocol['budget']['branch_optimizer_steps'],
        'optimizer_step_budget': frozen_protocol['budget']['total_optimizer_steps'],
        'primary': frozen_protocol['analysis']['primary']}
    population = json.loads((d.ROOT/frozen_protocol['population']).read_text())
    plan = json.loads((d.ROOT/frozen_protocol['plan']).read_text())
    population_grid = validate_population(population, frozen_protocol, plan)
    native_reference = reset.load_reference() if native_reference is None else native_reference
    reference = pressure.load_reference() if reference is None else reference
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
    d.write_json(directory/'protocol.json',frozen_protocol)
    d.write_json(directory/'execution_protocol.json',json.loads((d.ROOT/'configs/clip_source_resume.json').read_text()))
    d.write_json(directory/'runtime_protocol.json',p)
    d.write_json(directory/'freeze_manifest.json',json.loads((d.ROOT/'configs/clip_source_regime_freeze_manifest.json').read_text()))
    d.write_json(directory/'frozen_population.json',population)
    d.write_json(directory/'split_plan.json',split)
    plans,permutations = policy.make_plans(split['training'],p)
    d.write_json(directory/'branch_plans.json',plans)
    d.write_json(directory/'timing_permutations.json',permutations)
    for name,value in [('split_plan.json',split),('branch_plans.json',plans),('training_exclusion_indices.json',sorted(set(old)|set(split['report'])))]:
        if reset.digest(value) != native_reference[name]:
            raise AssertionError('Replay plan differs: '+name)
    d.write_json(directory/'native_replay_reference.json',native_reference)
    d.write_json(directory/'native_continuation_reference.json',reference)
    plans = wl.common_plans(split['training'],p)
    d.write_json(directory/'common_branch_plans.json',plans)
    frozen_calibration = json.loads((d.ROOT/p['calibration']['plan']).read_text())
    if pressure.calibration_plan(split,plans,p) != frozen_calibration['seeds']:
        raise AssertionError('Frozen calibration membership differs')
    d.write_json(directory/'calibration_plan.json',frozen_calibration)
    sync()  # All protocol/data/reference files durable before model updates.
    rows, trunk_steps = [],0
    for seed in p['training_seeds']:
        clip_train.set_global_seed(seed)
        folder = directory/f'seed_{seed}'
        folder.mkdir(exist_ok=True)
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
        seed_reference = native_reference['seeds'][str(seed)]
        for name in ['data_roles.json','training_batches.json']:
            if reset.digest(json.loads((folder/name).read_text())) != seed_reference[name]:
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
        states,trunk_audit = resumable_trunk(model,optimizer,scheduler,training_batches(data),config,p,seed=seed,commit=commit,store=store,loader=data.train_loader)
        if (trunk_audit['auxiliary_updates'] != 0 or trunk_audit['decisions'] or trunk_audit['probe_calls'] != 0
                or any(t['coefficient'] != 0. for t in trunk_audit['traces'])
                or trunk_audit['actual_optimizer_steps'] != p['trunk_updates']):
            raise AssertionError('Starting states must have strictly native-only history')
        if reset.digest(trunk_audit) != seed_reference['trunk_audit.json']:
            raise AssertionError('Native trunk audit replay mismatch')
        trunk_steps += trunk_audit['actual_optimizer_steps']
        d.write_json(folder/'trunk_audit.json',trunk_audit)
        authenticate_states(states,plan['seeds'][str(seed)],p['checkpoint_steps'],trunk_audit)
        report = FullPoolReport(report_batches,p['report_size'])
        for step, initial in states.items():
            if (initial['arm'] != 'baseline' or initial['starting_state_history'] != 'native_only'
                    or initial['prior_auxiliary_updates'] != 0 or initial['completed_updates'] != step
                    or initial['scheduler']['last_epoch'] != step):
                raise AssertionError('Starting-state identity/scheduler/history differs')
        common = states[p['common_stream_checkpoint_step']]
        cal_indices = [i for batch in frozen_calibration['seeds'][str(seed)]['batches'] for i in batch]
        used_indices = [i for u in p['stream_seeds'] for batch in plans[f'{seed}:{u}'] for i in batch]
        cal_keys = keys(cal_indices)
        if len(cal_keys) != len(cal_indices) or cal_keys & (keys(used_indices)|report_keys|old_keys|probe_keys):
            raise AssertionError('Calibration image identity overlap')
        def build_calibration():
            with d.observe(model):
                return d.build_batches(data,cal_indices,seed=seed,epoch=common['epoch'],canonical=False)
        cal_batches,cal_records = policy.guarded(model,optimizer,scheduler,build_calibration)
        if len(cal_batches) != p['calibration']['batches_per_seed']:
            raise AssertionError('Calibration batch count differs')
        d.write_json(folder/'calibration_batches.json',dict(records=cal_records,caption_epoch=common['epoch'],
            source_indices=cal_indices,disjoint_from_report_probes_and_continuation_by_image_key=True))
        sync()
        def cal_commit(label,record):
            d.write_json(folder/(label+'.json'),record)
            sync()
        cal_population = [population_grid[seed,None,t] for t in range(4)]
        for t, record in enumerate(cal_population):
            validate_batch_identity(record,cal_records[t*p['batch_size']:(t+1)*p['batch_size']])
        cal_identity = dict(records=cal_records,caption_epoch=common['epoch'],source_indices=cal_indices,
            disjoint_from_report_probes_and_continuation_by_image_key=True)
        if reset.digest(cal_identity) != plan['seeds'][str(seed)]['calibration_batch_records_canonical_sha256']:
            raise AssertionError('Calibration processed identities differ from D0')
        cal_key=f'seed_{seed}_calibration'
        sealed=store.load(cal_key)
        cal_context=dict(population=reset.digest(cal_population),states=d.state_digest(states),config=reset.digest(config))
        if sealed is not None:
            if sealed['context']!=cal_context:raise ResumeMismatch('Saved calibration context differs')
            calibrations=sealed['weights']
        elif legacy and (folder/'calibration.json').exists():
            calibrations=json.loads((folder/'calibration.json').read_text())['weights']
            for w in p['checkpoint_steps']:
                branch_coefficients(calibrations[str(w)],p)
                if calibrations[str(w)]['population_sha256']!=reset.digest(cal_population):
                    raise ResumeMismatch('Legacy calibration population differs')
            store.save(cal_key,dict(context=cal_context,weights=calibrations))
        else:
            calibrations = calibrate_states(model,optimizer,scheduler,states,cal_batches,cal_population,config,p,cal_commit)
            store.save(cal_key,dict(context=cal_context,weights=calibrations))
        for w in p['checkpoint_steps']:
            branch_coefficients(calibrations[str(w)],p)
        d.write_json(folder/'calibration.json',dict(training_seed=seed,weights=calibrations,
            calibration_sha256=reset.digest(calibrations),optimizer_updates=0))
        sync()  # Both weight calibrations durable before ANY continuation for this seed.
        del cal_batches,cal_records
        for stream in p['stream_seeds']:
            key = f'{seed}:{stream}'
            pair_dir = folder/f'stream_{stream}'
            pair_dir.mkdir(parents=True,exist_ok=True)
            batches,batch_records = d.build_batches(data,[i for b in plans[key] for i in b],
                seed=seed,epoch=common['epoch'],canonical=False)
            if len(batches) != p['horizon']:
                raise AssertionError('Continuation length differs')
            batch_identity = dict(records=batch_records,caption_epoch=common['epoch'])
            old_key = f'{seed}:{p["common_stream_checkpoint_step"]}:{stream}'
            if reset.digest(batch_identity) != native_reference['pairs'][old_key]['training_batches_sha256']:
                raise AssertionError('Common early continuation batch/caption replay mismatch')
            d.write_json(pair_dir/'training_batches.json',batch_identity)
            sync()
            def pair_commit(label,value):
                d.write_json(pair_dir/(label+'.json'),value)
                sync()
            def save_point(cell,arm,offset,value):
                d.write_json(pair_dir/f'point_{cell}_{arm}_{offset:03d}.json',value)
                if offset == p['horizon']:
                    meta = cache.save_cache(pair_dir/f'features_{cell}_{arm}.npz',report.encoded,row_hash)
                    if meta['feature_sha256'] != value['report']['features_sha256']:
                        raise AssertionError('Endpoint cache differs from evaluated state')
                sync()
            population_records = [population_grid[seed,stream,t] for t in range(p['horizon'])]
            for t, record in enumerate(population_records):
                validate_batch_identity(record,batch_records[t*p['batch_size']:(t+1)*p['batch_size']])
            d.write_json(pair_dir/'population.json',population_records)
            sync()
            result = run_branches(model,optimizer,scheduler,states,batches,population_records,config,report,
                calibrations=calibrations,p=p,commit=pair_commit,save_point=save_point,
                native_reference=reference['pairs'][key]['weights'],
                progress=lambda text:print('D1_SET:',key,text,flush=True),
                store=store,key=f'seed_{seed}_stream_{stream}',legacy_folder=pair_dir if legacy else None)
            result.update(training_seed=seed,stream_seed=stream,report_row_order_sha256=row_hash,
                starting_state_history='native_only',prior_auxiliary_updates=0,
                common_caption_epoch=common['epoch'],common_rng_source_step=p['common_stream_checkpoint_step'],
                source_states_sha256=d.state_digest(states),batch_records_sha256=reset.digest(batch_identity))
            d.write_json(pair_dir/'result.json',result)
            rows.append(result)
            sync()
            del batches,batch_records
            gc.collect()
        del states,initial,common,model,optimizer,scheduler,data,report,report_batches
        gc.collect()
        torch.cuda.empty_cache()
    if trunk_steps != p['trunk_optimizer_steps']:
        raise AssertionError('Trunk budget differs')
    summary = summarize(rows,p)
    d.write_json(directory/'comparison.json',summary)
    (directory/'REPORT.md').write_text(markdown_report(summary))
    d.write_json(directory/'completion.json',dict(status='complete',experiment=p['experiment'],
        matched_sets=summary['matched_sets'],branches=summary['branches'],
        point_records=summary['branches']*len(p['observation_steps']),
        trunk_optimizer_steps=trunk_steps,branch_optimizer_steps=summary['branch_optimizer_steps'],
        optimizer_steps=summary['optimizer_steps'],native_trunk_exact_replay=True,native_continuation_exact_replay=True,
        frozen_population_sha256=reset.digest(population),initial_calibration_verified=True,
        independent_test_set=False,coefficients_fixed=True,support_ids_fixed=True,
        pressure_probe_records=summary['branches']*len(p['pressure_probe_offsets']),
        virtual_optimizer_step_calls=2*summary['branches']*len(p['pressure_probe_offsets']),
        committed_probe_optimizer_updates=0,calibration_optimizer_steps=0))
    d.write_json(directory/'resume_execution.json',dict(events=store.events,logical_optimizer_steps=summary['optimizer_steps'],
        note='Logical experiment steps count each treatment once; execution events separately identify work performed this attempt.'))
    sync()
    return summary
