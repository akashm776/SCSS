"""Full-population initial placement audit using the authenticated v2 head.

The issued v1–v2c implementations stay unchanged. This stage extends coverage;
it does not refit calibration or execute continuation updates.
"""

from pathlib import Path

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src import clip_train
from src.clip_placement_evidence import read_checked
from src.clip_placement_preflight import (
    ARMS,
    ContractViolation,
    InvalidEvidence,
    auxiliary_path,
    finite,
    require,
)
from src.clip_placement_preflight_run import capture_contract_evidence, describe
from src.clip_placement_v2a import fp32_head, placement_contract, serialize_native
from src.clip_placement_v2b import gradients
from src.clip_source_regime import forward


def calibration_reference(root, spec):
    """Authenticate parent receipts and recompute every archived coefficient."""
    from scripts.audit_clip_placement_v2c import audit

    ref = spec["calibration_reference"]
    folder = Path(root) / ref["root"]
    try:
        records = {
            name: read_checked(folder, name, sha) for name, sha in ref["files"].items()
        }
        summary = audit(folder)
        if summary["contract_sha256"] != ref["contract_sha256"]:
            raise ValueError("Wrong calibration parent contract")
        cells = {}
        for key, entry in records["index.json"].items():
            if ref["files"].get(entry["file"]) != entry["sha256"]:
                raise ValueError("Calibration record missing from frozen reference")
            _, seed, weight = key.split("/")
            cells[int(seed), int(weight)] = records[entry["file"]]["result"]["evidence"]
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise InvalidEvidence(
            "Calibration parent authentication failed: " + str(error)
        ) from error
    return summary, cells


def expected_keys(p):
    return {
        f"geometry/{seed}/{stream}/{weight}/{offset}"
        for seed in p["training_seeds"]
        for stream in [*p["stream_seeds"], None]
        for weight in p["checkpoint_steps"]
        for offset in range(
            p["calibration_batches"] if stream is None else p["horizon"]
        )
    }


def completion(records, p):
    """A partial passing grid is never launch eligible; failures remain terminal."""
    expected = expected_keys(p)
    if not set(records) <= expected:
        raise InvalidEvidence("Unexpected population check key")
    statuses = [r["status"] for r in records.values()]
    if any(s not in ("PASS", "FAIL") for s in statuses):
        raise InvalidEvidence("Unexpected committed check status")
    if "FAIL" in statuses:
        return "FAIL"
    return "PASS" if set(records) == expected else "INCOMPLETE"


def verify_calibration_forward(evidence, archived, record):
    """Extend the archived calibration's exact weighted replay gate to this run."""
    actual = {
        name: evidence[name]
        for name in ("placements", "fp32_settings", "empty", "native_loss")
    }
    expected = {name: archived[name] for name in actual}
    if actual != expected or archived["eligible_queries"] != record["eligible_queries"]:
        raise InvalidEvidence(
            "Full-population forward does not reproduce calibrated treatment"
        )
    evidence["calibration_forward_reproduced"] = True


def audit_batch(
    model,
    opt,
    sched,
    batch,
    config,
    record,
    weight,
    common,
    delta,
    p,
    spec,
    archived=None,
):
    """One restored encoder/carrier execution, then forward and feature VJP checks.

    VJPs stop at live encoder features, avoiding 624 full-encoder gradient probes.
    Full model-gradient validity and calibration are authenticated upstream.
    """
    result = {"optimizer_updates": 0}
    device = next(model.parameters()).device
    with (
        capture_contract_evidence(result),
        g.preserved(model, opt, sched) as restoration,
    ):
        result["preservation"] = restoration
        g.restore_rng(common)
        model.train()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        before = d.state_digest(model.state_dict())
        with (
            torch.enable_grad(),
            clip_train._autocast(device, config["training"]["mixed_precision"]),
        ):
            out = forward(model, batch, config)
            empty = not record["eligible_queries"]
            shared, geom = (
                ({}, {}) if empty else auxiliary_path(out, record, weight, delta, p)
            )
            candidates = dict(shared)
            if not empty:
                candidates["H-Uniform"] = dict(shared["Wrong"], z=shared["Wrong"]["z0"])
            native, values, settings = fp32_head(out, candidates)
            placed = {a: values[a] for a in ARMS} if not empty else {}
            result.update(
                empty=empty,
                placements=describe(placed, geom, delta),
                fp32_settings=settings,
                native_loss=float(native["loss_native"].detach()),
                native=serialize_native(native),
            )
            if not empty:
                contract = placement_contract(placed, delta, p)
                result["placement_contract"] = contract
                require(
                    contract["status"] == "PASS",
                    "population_placement",
                    contract,
                    "PASS",
                    "FP32_head",
                )
                for arm, value in values.items():
                    finite(value["loss"], arm + "/loss")
                result["auxiliary_losses"] = {
                    a: float(v["loss"].detach()) for a, v in values.items()
                }
            else:
                result.update(
                    auxiliary_losses=dict.fromkeys(spec["arms"], 0.0),
                    auxiliary_gradient_norm=0.0,
                    gradient_cosine=None,
                )
        if archived is not None:
            verify_calibration_forward(result, archived, record)
        diagnostic_inputs = dict(
            image=out.image_features,
            text=out.text_features,
            scale=out.logit_scale,
            candidates={a: v["z"] for a, v in candidates.items()},
        )
        inputs_before = d.state_digest(diagnostic_inputs)
        diagnostic_before = g.stamp(model, opt, sched)
        if not empty:
            try:
                with torch.enable_grad(), g.scoring_precision(device):
                    measured = gradients(
                        out,
                        shared,
                        native,
                        placed,
                        (out.image_features, out.text_features),
                        model.clip_model.logit_scale,
                        p,
                        spec["gradient_contract"],
                    )
            except ContractViolation as error:
                result["feature_gradient_contract"] = getattr(error, "evidence", {})
                raise
            # The reusable validator accepts a parameter tuple; here it receives
            # features. Label those descriptive norms explicitly in the evidence.
            measured["auxiliary_feature_gradients"] = measured.pop("full_gradients")
            result["feature_gradient_contract"] = measured
        else:
            # An empty auxiliary term must be scale-disconnected and differentiable.
            zero = native["similarity"].sum() * 0
            scale, image, text = torch.autograd.grad(
                zero,
                (out.logit_scale, out.image_features, out.text_features),
                allow_unused=True,
                retain_graph=True,
            )
            require(
                scale is None
                and not bool(image.count_nonzero())
                and not bool(text.count_nonzero()),
                "empty_auxiliary_gradient",
                [scale is None, float(image.norm()), float(text.norm())],
                [True, 0.0, 0.0],
                "empty",
            )
        if g.stamp(model, opt, sched) != diagnostic_before:
            raise AssertionError("Gradient diagnostics mutated training state")
        if d.state_digest(diagnostic_inputs) != inputs_before:
            raise AssertionError("Gradient diagnostics mutated live scoring inputs")
        with torch.no_grad():
            after_native, after_values, after_settings = fp32_head(out, candidates)
        if settings != after_settings:
            raise AssertionError("Diagnostic replay changed head precision")
        for name, tensor in native.items():
            if not torch.equal(tensor, after_native[name]):
                raise AssertionError("Diagnostic replay changed native " + name)
        for arm, value in values.items():
            for name in ("z", "s_aux", "e_aux", "A", "m_loss", "loss"):
                if not torch.equal(value[name], after_values[arm][name]):
                    raise AssertionError(
                        "Diagnostic replay changed " + arm + "/" + name
                    )
        result["diagnostic_on_off_identical"] = True
        if d.state_digest(model.state_dict()) != before:
            raise AssertionError("Population audit mutated model state")
    return result
