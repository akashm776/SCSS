"""Restore one authenticated starting state; execute exactly one encoder forward."""
import json
from pathlib import Path

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src import clip_train
from src import clip_weight_lr as wl
from src.clip_geometry_run import authenticate_states
from src.clip_source_regime import load_protocol, validate_population, validate_batch_identity, forward
from src.clip_source_resume_store import safe_load, expand
from src.clip_placement_evidence import checksum, checked_path
from src.clip_placement_preflight import InvalidEvidence, ContractViolation
from src.clip_placement_v2a import native_record, paired_from_output


def run(d1, inputs, p, delta, spec, archived):
    from transformers import AutoProcessor
    seed, stream, w, offset = (spec[k] for k in ['training_seed','stream_seed','weight_step','batch_offset'])
    if (seed,stream,w,offset) != (271828,2026100101,100,0):
        raise InvalidEvidence('v2a is restricted to the archived first batch')
    population = json.loads((d.ROOT/p['population']).read_text())
    plan = json.loads((d.ROOT/p['plan']).read_text())
    if population != inputs['frozen_population.json'] or inputs['common_branch_plans.json'] != plan['continuation_batches']:
        raise InvalidEvidence('D1 population or continuation schedule differs')
    grid = validate_population(population,load_protocol(),plan)
    record = grid[seed,stream,offset]
    context = dict(training_seed=seed,stream_seed=stream,weight_step=w,batch_offset=offset,
                   population=record['population'],eligible_queries=record['eligible_queries'])
    if context != archived['context']:
        raise InvalidEvidence('Wrong first-batch exposure identity')
    clip_train.set_global_seed(seed)
    config = json.loads(json.dumps(inputs[f'seed_{seed}/training_config.json']))
    config['dataset']['diagnostic_holdout_indices'] = str((Path(d1)/'training_exclusion_indices.json').resolve())
    if config['training']['mixed_precision'] != 'bf16':
        raise InvalidEvidence('Original encoder/carrier execution must be bf16 autocast')
    device, _ = clip_train.resolve_device(config['runtime'])
    processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
    data = clip_train.build_clip_training_data(config,processor,pin_memory=False)
    if data.train_dataset.source_indices != inputs['split_plan.json']['training']:
        raise InvalidEvidence('Training data identity differs')
    model = d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
    trainable = d.configure_clip_trainable_parameters(model,config['model']['trainable_policy'])
    if (trainable['trainable_parameter_count'],trainable['trainable_tensor_count']) != (10895617,35):
        raise InvalidEvidence('Trainable policy differs')
    opt = d.build_clip_optimizer(model,config['optimizer'])
    sched = clip_train.build_scheduler(opt,warmup_steps=config['scheduler']['warmup_steps'],
                                      total_steps=inputs['runtime_protocol.json']['scheduler_horizon'])
    pointer = inputs[f'resume_state/seed_{seed}_trunk.json']
    entry = pointer['generations'][0]
    blob = checked_path(d1,'resume_state/'+entry['file'])
    if checksum(blob) != entry['sha256']: raise InvalidEvidence('Trunk changed after authentication')
    saved = safe_load(blob)
    if saved['contract'] != pointer['contract'] or saved['key'] != f'seed_{seed}_trunk':
        raise InvalidEvidence('Decoded trunk identity differs')
    states = {step:expand(saved['payload']['snapshots'][str(step)],model) for step in p['checkpoint_steps']}
    del saved
    try:
        authenticate_states(states,plan['seeds'][str(seed)],p['checkpoint_steps'],inputs[f'seed_{seed}/trunk_audit.json'])
    except AssertionError as error:
        raise InvalidEvidence('Native checkpoint authentication failed') from error
    common = states[100]
    indices = plan['continuation_batches'][f'{seed}:{stream}'][offset]
    with g.preserved(model,opt,sched):
        batches, identities = d.build_batches(data,indices,seed=seed,epoch=common['epoch'],canonical=False)
    original = inputs[f'seed_{seed}/stream_{stream}/training_batches.json']
    if (len(batches) != 1 or common['epoch'] != original['caption_epoch']
            or identities != original['records'][offset*64:(offset+1)*64]):
        raise InvalidEvidence('Processed first-batch images/captions differ')
    validate_batch_identity(record,identities)
    wl.prepare_branch(model,opt,sched,states[w],common,common)
    result = None
    with g.preserved(model,opt,sched) as preservation:
        g.restore_rng(common);model.train()
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        before = d.state_digest(model.state_dict())
        with torch.no_grad(),clip_train._autocast(device,config['training']['mixed_precision']):
            out = forward(model,batches[0],config)
            native_v1 = native_record(out.raw_similarity,out.logits)
            try:
                result = paired_from_output(out,record,w,delta,p,spec,archived,native_v1)
            except ContractViolation as error:
                # Even initial target reconstruction must reproduce the archived
                # passing ideal geometry; failure here is not a v2a head result.
                result = dict(status='INVALID',reason='ARCHIVED_GEOMETRY_NOT_REPRODUCED',
                              violation=error.record)
        if d.state_digest(model.state_dict()) != before:
            raise AssertionError('Forward observation mutated model state')
    result.update(context=context,preservation=preservation,encoder_forwards=1,
                  optimizer_updates=0,gradient_probes=0,calibration_batches=0)
    return result
