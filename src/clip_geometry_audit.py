"""Frozen D0 geometry audit. No loss, backward, or optimizer step in this module."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import clip_checkpoint_diagnostic as d
from src import clip_difficulty as historical
from src import clip_train
from src.clip_optimizer_reset import digest

FREEZE_SHA256 = '8ec6f7ff1fc6dd6a9e774a261704cb4a0278be391b0a045c06a5471c5776c0e8'
REGIMES = ('H', 'M', 'C', 'I')
PAIRS = tuple(itertools.combinations(REGIMES, 2))


class AuditFailure(ValueError):
    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record


def require(ok, message, record=None):
    if not ok:
        raise AuditFailure(message, record)


def load_freeze():
    path = d.ROOT/'configs/clip_geometry_audit_freeze_manifest.json'
    require(d.checksum(path) == FREEZE_SHA256, 'D0 freeze manifest changed')
    manifest = json.loads(path.read_text())
    for group in ['files', 'parent_evidence_files', 'reference_source_files']:
        for name, sha in manifest[group].items():
            require(d.checksum(d.ROOT/name) == sha, 'Frozen D0 dependency changed: '+name)
    return manifest


def load_protocol():
    load_freeze()
    return json.loads((d.ROOT/'configs/clip_geometry_audit.json').read_text())


def backend_state():
    return dict(matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark)


def restore_backend(s):
    torch.set_float32_matmul_precision(s['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = s['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = s['cudnn_tf32']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = s['bf16_reduction']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = s['fp16_reduction']
    torch.backends.cudnn.deterministic = s['cudnn_deterministic']
    torch.backends.cudnn.benchmark = s['cudnn_benchmark']


def restore_rng(s):
    random.setstate(s['python_rng'])
    np.random.set_state(s['numpy_rng'])
    torch.set_rng_state(s['torch_rng'])
    require(len(s['cuda_rng']) == torch.cuda.device_count(), 'Wrong CUDA RNG topology')
    if s['cuda_rng']:
        torch.cuda.set_rng_state_all(s['cuda_rng'])


def stamp(model, optimizer=None, scheduler=None):
    return d.state_digest(dict(model=model.state_dict(), gradients=[v.grad for v in model.parameters()],
        modes=[m.training for m in model.modules()], optimizer=None if optimizer is None else optimizer.state_dict(),
        scheduler=None if scheduler is None else scheduler.state_dict(), rng=d.rng_state(), backend=backend_state()))


@contextmanager
def preserved(model, optimizer=None, scheduler=None):
    """Restore actual values on success AND failure; verify restoration, not just a hash."""
    before = stamp(model, optimizer, scheduler)
    saved = deepcopy(model.state_dict())
    gradients = [None if v.grad is None else v.grad.detach().clone() for v in model.parameters()]
    modes = [m.training for m in model.modules()]
    opt = None if optimizer is None else deepcopy(optimizer.state_dict())
    sched = None if scheduler is None else deepcopy(scheduler.state_dict())
    rng, backend = deepcopy(d.rng_state()), backend_state()
    audit = dict(before_sha256=before)
    try:
        yield audit
    finally:
        model.load_state_dict(saved, strict=True)
        for v, g in zip(model.parameters(), gradients):
            v.grad = g
        for m, mode in zip(model.modules(), modes):
            m.training = mode
        if optimizer is not None:
            optimizer.load_state_dict(opt)
        if scheduler is not None:
            scheduler.load_state_dict(sched)
        restore_rng(rng)
        restore_backend(backend)
        audit.update(after_sha256=stamp(model, optimizer, scheduler))
        require(audit['after_sha256'] == before, 'D0 failed to restore complete state', audit)
        audit['preserved'] = True


@contextmanager
def scoring_precision(device):
    before = backend_state()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        restore_backend(before)


def ordered_supports(scores, k):
    """CPU ordering of exact FP32 scores; no extra rounding or sign tolerance."""
    size = len(scores)
    require(scores.shape == (size, size) and size-1 >= 15+k and np.isfinite(scores).all(), 'Invalid scoring matrix/support size')
    result = {r: [] for r in REGIMES}
    for i in range(size):
        candidates = [j for j in range(size) if i != j]
        order = sorted(candidates, key=lambda j: (-float(scores[i,j]), j))
        positive = [j for j in order if scores[i,i]-scores[i,j] > 0]
        nonpositive = sorted((j for j in order if scores[i,i]-scores[i,j] <= 0),
                             key=lambda j: (abs(float(scores[i,i]-scores[i,j])), j))
        result['H'].append(order[:k])
        result['M'].append(order[15:15+k])
        result['C'].append(positive[:k] if len(positive) >= k else None)
        result['I'].append(nonpositive[:k] if len(nonpositive) >= k else None)
    return result


def normalized(features, minimum, tolerance):
    x = features.detach().float()
    norms = torch.linalg.vector_norm(x, dim=-1)
    require(bool(torch.isfinite(x).all()) and bool((norms > minimum).all()), 'Invalid forward feature/norm')
    x = x/norms[:,None]
    require(bool((torch.linalg.vector_norm(x,dim=-1)-1).abs().le(tolerance).all()), 'Non-unit ideal feature')
    return x


def assert_candidate_invariants(record, p):
    tol = p['validity']['ideal_invariant_absolute_tolerance']
    if not record['ideal']['valid']:
        return
    real, syn = record['real']['ideal_margin'], record['ideal']['margin']
    if record['k'] == 1:
        require(abs(real-syn) <= tol, 'k1 Select/Synth invariant failed', record)
    if record['regime'] == 'I' and record['positive_cosine'] >= 0:
        require(syn <= real+tol and real <= tol, 'I-Synth invariant failed', record)
        record['I_invariant'] = 'passed'
    else:
        record['I_invariant'] = 'not_applicable'


def candidate_geometry(out, q, v, scores, supports, regime, k, p):
    """Vectorized GPU construction; transfer each batch's scalar geometry only once."""
    size = len(scores)
    mask = torch.zeros((size,size), device=v.device, dtype=torch.bool)
    for i, ids in enumerate(supports):
        if ids is not None:
            mask[i, ids] = True
    ideal_mu = (mask.float()/k) @ v
    ideal_norm = torch.linalg.vector_norm(ideal_mu,dim=-1)
    ideal_z = F.normalize(ideal_mu,dim=-1,eps=1e-12)
    ideal_similarity = (q*ideal_z).sum(-1)
    with clip_train._autocast(v.device, 'bf16'):
        weights = mask.to(out.raw_similarity.dtype)
        weights = (weights/weights.sum(-1,keepdim=True).clamp_min(1)).detach()
        mixed_mu = weights.to(out.image_features.dtype) @ out.image_features
        mixed_z = F.normalize(mixed_mu,dim=-1,eps=1e-12)
        training_similarity = (out.text_features*mixed_z).sum(-1)
    mixed_norm = torch.linalg.vector_norm(mixed_mu.float(),dim=-1)
    mixed_renorm = F.normalize(mixed_z.float(),dim=-1,eps=1e-12)
    mixed_selection_similarity = (q*mixed_renorm).sum(-1)
    arrays = {name:value.detach().float().cpu().numpy() for name,value in dict(
        ideal_norm=ideal_norm, mixed_norm=mixed_norm, ideal_similarity=ideal_similarity,
        mixed_selection_similarity=mixed_selection_similarity, training_similarity=training_similarity,
        ideal_unit_norm=torch.linalg.vector_norm(ideal_z,dim=-1)).items()}
    raw = out.raw_similarity.detach().float().cpu().numpy()
    result = []
    for i, ids in enumerate(supports):
        if ids is None:
            result.append(None)
            continue
        real = dict(index=ids[0],ideal_margin=float(scores[i,i]-scores[i,ids[0]]),
                    training_margin=float(raw[i,i]-raw[i,ids[0]]))
        entry = dict(regime=regime,k=k,positive_cosine=float(scores[i,i]),real=real,I_invariant='not_applicable')
        for path, norm_name, sim_name in [('ideal','ideal_norm','ideal_similarity'),
                                         ('mixed','mixed_norm','mixed_selection_similarity')]:
            norm = float(arrays[norm_name][i])
            valid = bool(np.isfinite(norm) and norm > p['validity']['mixture_norm_min_exclusive'])
            value = dict(valid=valid,norm=norm if np.isfinite(norm) else None,
                         reason=None if valid else ('nonfinite_mixture_norm' if not np.isfinite(norm) else 'near_zero_mixture_norm'),
                         margin=None,delta_margin=None,delta_absolute_margin=None)
            if valid:
                margin = float(scores[i,i]-arrays[sim_name][i])
                require(np.isfinite(margin), 'Nonfinite generated score', entry)
                value.update(margin=margin,delta_margin=margin-real['ideal_margin'],
                             delta_absolute_margin=abs(margin)-abs(real['ideal_margin']))
                if path == 'ideal':
                    require(abs(float(arrays['ideal_unit_norm'][i])-1) <= p['validity']['ideal_unit_norm_absolute_tolerance'], 'Non-unit ideal candidate', entry)
            if path == 'mixed':
                train_margin = float(raw[i,i]-arrays['training_similarity'][i]) if valid else None
                require(train_margin is None or np.isfinite(train_margin), 'Nonfinite mixed training score', entry)
                value.update(training_margin=train_margin,
                    training_delta_margin=None if train_margin is None else train_margin-real['training_margin'],
                    sign_disagreement=None if train_margin is None else int(np.sign(train_margin) != np.sign(value['margin'])))
            entry[path] = value
        assert_candidate_invariants(entry,p)
        result.append(entry)
    return result


@torch.no_grad()
def geometry(out, identities, context, p):
    minimum=p['validity']['input_embedding_norm_min_exclusive']
    tolerance=p['validity']['ideal_unit_norm_absolute_tolerance']
    device=out.image_features.device
    with scoring_precision(device):
        q=normalized(out.text_features,minimum,tolerance)
        v=normalized(out.image_features,minimum,tolerance)
        scores_tensor=q @ v.T
        scores=scores_tensor.cpu().numpy()
        raw=out.raw_similarity.detach().float().cpu().numpy()
        require(scores.shape==raw.shape and np.isfinite(scores).all() and np.isfinite(raw).all(), 'Nonfinite forward/scoring matrix')
        size=len(scores)
        require(len(identities)==size and len({r['source_index'] for r in identities})==size, 'Wrong batch identities')
        historical_masks=historical.select_supports(out.raw_similarity,k=8,start_rank=16)
        historical_ids={r:[np.flatnonzero(row).tolist() for row in historical_masks[a].cpu().numpy()]
                        for r,a in [('H','hard'),('M','semihard')]}
        records=[]
        for i, identity in enumerate(identities):
            ids=[j for j in range(size) if i!=j]
            margins=[float(scores[i,i]-scores[i,j]) for j in ids]
            old=[float(raw[i,i]-raw[i,j]) for j in ids]
            neg=sum(x<0 for x in margins); zero=sum(x==0 for x in margins); pos=sum(x>0 for x in margins)
            require(neg+zero+pos==size-1,'Margin partition failed')
            records.append(dict(**context,query_index=i,source_index=identity['source_index'],caption_index=identity['caption_index'],
                species_id=identity['species_id'],positive_cosine=float(scores[i,i]),training_positive_cosine=float(raw[i,i]),
                candidate_indices=ids,candidate_source_indices=[identities[j]['source_index'] for j in ids],
                candidate_species_ids=[identities[j]['species_id'] for j in ids],margins=margins,training_margins=old,
                N_negative=neg,N_zero=zero,N_positive=pos,N_nonpositive=neg+zero,
                precision_sign_disagreements=int(sum(np.sign(x)!=np.sign(y) for x,y in zip(margins,old))),
                sizes={},historical={}))
        for k in p['support_sizes']:
            ordered=ordered_supports(scores,k)
            old_ordered=ordered_supports(raw,k)
            calculated={r:candidate_geometry(out,q,v,scores,ordered[r],r,k,p) for r in REGIMES}
            for i, record in enumerate(records):
                entry=dict(eligible=record['N_positive']>=k and record['N_nonpositive']>=k,regimes={},overlaps={})
                for r in REGIMES:
                    ids=ordered[r][i]
                    same_old = None if ids is None or old_ordered[r][i] is None else set(ids)==set(old_ordered[r][i])
                    candidate=calculated[r][i]
                    if ids is None:
                        available=record['N_positive'] if r=='C' else record['N_nonpositive']
                        entry['regimes'][r]=dict(feasible=False,available=available,reason='insufficient_support',
                            support=None,support_source_indices=None,support_sha256=None,contributor_margins=None,
                            contributor_training_margins=None,similarity_ranks=None,same_species_fraction=None,
                            real_same_species=None,candidate=None,original_score_support_equal=same_old)
                        continue
                    require(len(ids)==len(set(ids))==k and i not in ids,'Invalid support')
                    src=[identities[j]['source_index'] for j in ids]
                    rank={j:rank+1 for rank,j in enumerate(sorted(record['candidate_indices'],key=lambda j:(-float(scores[i,j]),j)))}
                    entry['regimes'][r]=dict(feasible=True,available=(record['N_positive'] if r=='C' else record['N_nonpositive'] if r=='I' else size-1),
                        reason=None,support=ids,support_source_indices=src,support_sha256=digest(dict(batch_indices=ids,source_indices=src)),
                        contributor_margins=[float(scores[i,i]-scores[i,j]) for j in ids],
                        contributor_training_margins=[float(raw[i,i]-raw[i,j]) for j in ids],
                        similarity_ranks=[rank[j] for j in ids],
                        same_species_fraction=sum(identities[j]['species_id']==identity_species(record) for j in ids)/k,
                        real_same_species=identities[ids[0]]['species_id']==record['species_id'],
                        candidate=candidate,original_score_support_equal=same_old)
                for a,b in PAIRS:
                    x,y=ordered[a][i],ordered[b][i]
                    entry['overlaps'][a+b]=None if x is None or y is None else dict(fraction=len(set(x)&set(y))/k,equal=set(x)==set(y))
                require(entry['overlaps']['CI'] is None or entry['overlaps']['CI']['fraction']==0,'C/I overlap invariant failed',record)
                if ordered['C'][i] is not None:
                    nonpos=sum(scores[i,i]-scores[i,j]<=0 for j in ordered['H'][i])
                    require(len(set(ordered['H'][i])&set(ordered['C'][i]))==k-nonpos,'H/C overlap invariant failed',record)
                record['sizes'][str(k)]=entry
                if k==8:
                    for r in ['H','M']:
                        old_ids=historical_ids[r][i]
                        record['historical'][r]=dict(support=old_ids,equal=set(old_ids)==set(ordered[r][i]),
                            overlap=len(set(old_ids)&set(ordered[r][i]))/8)
        # Ensure strict JSON serializability before any persistence.
        json.dumps(records,allow_nan=False)
        return records,dict(encoder_feature_dtype=str(out.image_features.dtype),raw_similarity_dtype=str(out.raw_similarity.dtype),
                            selection_dtype=str(scores_tensor.dtype),scoring_backend=backend_state())


def identity_species(record):
    return record['species_id']


def audit_batch(model,batch,identities,context,common_rng,config,p,optimizer=None,scheduler=None):
    with preserved(model,optimizer,scheduler) as restoration:
        restore_rng(common_rng)
        model.train()
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        before_weights=d.state_digest(model.state_dict())
        device=next(model.parameters()).device
        with torch.no_grad(),clip_train._autocast(device,config['training']['mixed_precision']):
            out=model(batch['pixel_values'].to(device),clip_train._text_batch(batch,device))
        require(d.state_digest(model.state_dict())==before_weights,'Encoder forward mutated parameters or buffers')
        records,numerics=geometry(out,identities,context,p)
    return records,dict(**restoration,**numerics,optimizer_updates=0,backward_passes=0)
