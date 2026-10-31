"""Paired forward-only D2-P v2a. No optimizer, calibration or gradient probes."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.clip_training import native_clip_contrastive_loss
from src import clip_geometry_audit as g
from src import clip_checkpoint_diagnostic as d
from src.clip_placement_preflight import (
    ARMS, InvalidEvidence, ContractViolation, auxiliary_path, finite)
from src.clip_placement_preflight_run import describe, serial_tensor
from src.clip_placement_evidence import checksum, digest, read_checked


def reference(root, spec):
    """Authenticate original bytes and their internal links, not an offline summary."""
    folder = Path(root) / spec['reference_root']
    files = {n: read_checked(folder, n, sha) for n, sha in spec['reference_files'].items()}
    acceptance, config, index = (files[n] for n in ['acceptance.json', 'run_config.json', 'index.json'])
    key = f"geometry/{spec['training_seed']}/{spec['stream_seed']}/{spec['weight_step']}/{spec['batch_offset']}"
    if (acceptance['status'] != 'FAIL' or acceptance['completed_checks'] != 1
            or not acceptance['provenance_valid'] or acceptance['optimizer_updates'] != 0
            or acceptance['d2_launched'] or set(index) != {key}
            or digest(config) != acceptance['contract_sha256']
            or checksum(folder / 'index.json') != acceptance['committed_check_index_sha256']):
        raise InvalidEvidence('Wrong v1 failure lineage')
    entry = index[key]
    if spec['reference_files'].get(entry['file']) != entry['sha256']:
        raise InvalidEvidence('Wrong v1 check hash')
    record = files[entry['file']]
    if record['key'] != key or record['contract_sha256'] != digest(config):
        raise InvalidEvidence('Wrong v1 check contract')
    state = record['result']['evidence']['preservation']
    if not state['preserved'] or state['before_sha256'] != state['after_sha256']:
        raise InvalidEvidence('V1 state preservation not authenticated')
    # Freeze the entire original implementation dependency graph as well.
    for name, sha in config['implementation'].items():
        if checksum(Path(root) / name) != sha:
            raise InvalidEvidence('Original D1/v1 implementation changed: ' + name)
    return config, record['result']


def reproduction(actual, archived, spec):
    """Exact JSON-number equality: no bf16-sized allowance or post-hoc tolerance."""
    if spec['reproduction_atol'] != 0 or spec['reproduction_rtol'] != 0:
        raise InvalidEvidence('v2a reproduction tolerances must be exactly zero')
    differences = []

    def compare(a, b, path):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                differences.append(dict(path=path, reason='field_set')); return
            for k in b: compare(a[k], b[k], path + '/' + k)
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                differences.append(dict(path=path, reason='shape')); return
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + '/' + str(i))
        elif a != b:
            differences.append(dict(path=path, observed=a, archived=b))

    compare(actual, archived, 'placements')
    return dict(status='INVALID' if differences else 'PASS', differences=differences,
                atol=0., rtol=0., scope='All archived placement observables; no claim about unsaved intermediates')


def native_record(raw, logits):
    """Caller supplies the actual numerical context for this head."""
    labels = torch.arange(len(logits), device=logits.device)
    row = F.cross_entropy(logits, labels, reduction='none')
    col = F.cross_entropy(logits.T, labels, reduction='none')
    # Use the production symmetric loss, not a reconstructed partition identity.
    loss = native_clip_contrastive_loss(logits)
    return dict(similarity=raw, logits=logits, p_sim=raw.diagonal(), ell_plus=logits.diagonal(),
                A_row=torch.logsumexp(logits, 1), A_col=torch.logsumexp(logits, 0),
                loss_row=row, loss_col=col, mean_row=F.cross_entropy(logits, labels),
                mean_col=F.cross_entropy(logits.T, labels), loss_native=loss)


def fp32_head(out, shared):
    """Cast only. Original carrier/targets/scale generation are outside this block.

    Reusable head preserves native scale graph when grad is enabled. The v2a
    runner deliberately uses no_grad; it cannot certify the backward contract.
    """
    with g.scoring_precision(out.text_features.device):
        q, image = out.text_features.float(), out.image_features.float()
        scale = out.logit_scale.float()  # no recomputation of exp(logit_scale)
        raw = q @ image.T
        logits = raw * scale
        native = native_record(raw, logits)
        values = {}
        for arm, v in shared.items():
            queries = v['queries']
            z = v['z'].float()  # no normalization, no second ST construction
            aux_scale = scale.detach()
            sim = (q[queries] * z).sum(-1)
            extra = sim * aux_scale
            partition = torch.logsumexp(raw[queries] * aux_scale, 1)
            m = extra - partition
            p_sim, ell_plus = raw.diagonal()[queries], logits.diagonal()[queries]
            values[arm] = dict(v, z=z, s_aux=sim, e_aux=extra, A=partition,
                p_sim=p_sim, ell_plus=ell_plus, tau=aux_scale, m_loss=m,
                loss=F.softplus(m).mean(), r_train=p_sim-sim, g_logit=ell_plus-extra,
                epsilon_m=m-(extra-partition),
                epsilon_scale=(ell_plus-extra)-aux_scale*(p_sim-sim),
                P_plus=torch.exp(ell_plus-partition), sigmoid_m=torch.sigmoid(m))
        settings = dict(g.backend_state(), autocast_enabled=torch.is_autocast_enabled(q.device.type))
    return native, values, settings


def placement_contract(values, delta, p):
    """All per-exposure checks; do not stop at the first Wrong violation."""
    result, failures = {}, []
    for arm, v in values.items():
        for name in ['r_train', 'g_logit', 'm_loss', 'loss', 'tau']:
            finite(v[name], arm + '/' + name)
        r, gap = v['r_train'].float(), v['g_logit'].float()
        target, tau = ARMS[arm]*delta, float(v['tau'])
        sim_error, logit_error = (r-target).abs(), (gap-tau*target).abs()
        checks = dict(similarity=sim_error <= p['training_margin_rtol_delta']*delta,
                      logit=logit_error <= p['training_logit_rtol_scaled_delta']*tau*delta)
        if target:
            checks.update(similarity_sign=r*ARMS[arm] > 0, logit_sign=gap*ARMS[arm] > 0)
        mask = torch.stack(list(checks.values())).all(0)
        result[arm] = dict(status='PASS' if bool(mask.all()) else 'FAIL',
            query_indices=v['queries'].cpu().tolist(), per_exposure_pass=mask.cpu().tolist(),
            checks={k: x.cpu().tolist() for k, x in checks.items()},
            similarity_error=serial_tensor(sim_error), logit_error=serial_tensor(logit_error),
            similarity_tolerance=p['training_margin_rtol_delta']*delta,
            logit_tolerance=p['training_logit_rtol_scaled_delta']*tau*delta)
        for name, check in checks.items():
            if not bool(check.all()):
                failures.append(dict(arm=arm, requirement=name,
                    query_indices=v['queries'][~check].cpu().tolist()))
    spacing = {}
    for left, right in [('Wrong', 'Boundary'), ('Boundary', 'Correct')]:
        gap = values[right]['r_train'].float()-values[left]['r_train'].float()
        good = gap >= p['minimum_adjacent_separation_delta']*delta
        spacing[left+'/'+right] = dict(values=serial_tensor(gap), per_exposure_pass=good.cpu().tolist())
        if not bool(good.all()):
            failures.append(dict(requirement='adjacent_separation', arms=[left, right],
                query_indices=values[left]['queries'][~good].cpu().tolist()))
    return dict(status='FAIL' if failures else 'PASS', arms=result, spacing=spacing, failures=failures)


def serialize_native(record):
    for k, v in record.items(): finite(v, 'native/' + k)
    return {k: serial_tensor(v) for k, v in record.items()}


def paired_differences(old, new):
    result = {}
    for key in old:
        x = new[key].detach().double() - old[key].detach().double()
        flat = x.flatten()
        result[key] = dict(values=x.cpu().tolist(), minimum=float(flat.min()),
            maximum=float(flat.max()), mean=float(flat.mean()), max_absolute=float(flat.abs().max()),
            quantiles={str(p):float(torch.quantile(flat,p)) for p in [0.,.25,.5,.75,1.]})
    return result


def paired_from_output(out, record, w, delta, p, spec, archived, native_v1):
    """Called within the unchanged v1 autocast context, under no_grad.

    auxiliary_path invokes the original carrier and target family exactly once.
    Its ST candidates are then shared by both heads.
    """
    v1, geom = auxiliary_path(out, record, w, delta, p)
    old = describe(v1, geom, delta)
    gate = reproduction(old, archived['evidence']['placements'], spec)
    result = dict(reproduction=gate, v1=dict(native=serialize_native(native_v1), placements=old),
        native_partition_provenance='logsumexp diagnostics of actual native input logits in each head context; not captured inside fused cross_entropy',
        geometry=dict(status='PASS', evidence='unchanged v1 targets() assertions passed for all three targets'),
        captured=dict(image_dtype=str(out.image_features.dtype), text_dtype=str(out.text_features.dtype),
            carrier_dtype=str(v1['Wrong']['z0'].dtype), scale_dtype=str(out.logit_scale.dtype),
            scale_value=float(out.logit_scale),
            current_inputs_sha256=d.state_digest(dict(image=out.image_features, text=out.text_features,
                scale=out.logit_scale, carrier=v1['Wrong']['z0'], targets={a:v['target'] for a,v in v1.items()}))))
    if gate['status'] != 'PASS':
        return dict(status='INVALID', reason='V1_REPRODUCTION_FAILED', evidence=result)
    # This head is never evaluated before the reproduction gate has passed.
    before = d.state_digest(dict(image=out.image_features,text=out.text_features,scale=out.logit_scale,
                                carrier=v1['Wrong']['z0'],candidates={a:v['z'] for a,v in v1.items()}))
    native_v2, v2, settings = fp32_head(out, v1)
    after = d.state_digest(dict(image=out.image_features,text=out.text_features,scale=out.logit_scale,
                               carrier=v1['Wrong']['z0'],candidates={a:v['z'] for a,v in v1.items()}))
    if before != after: raise AssertionError('Scoring head mutated shared inputs')
    contract = placement_contract(v2, delta, p)
    result.update(v1_observed_contract=placement_contract(v1,delta,p),
        v2a=dict(native=serialize_native(native_v2),placements=describe(v2,geom,delta),contract=contract),
        native_differences=paired_differences(native_v1,native_v2),
        fp32_settings=settings, shared_inputs_unchanged=True)
    return dict(status=contract['status'], evidence=result)
