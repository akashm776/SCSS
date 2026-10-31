"""Read-only audit of one authenticated D2-P failure; never changes D2 rules."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def audit(root):
    root = Path(root)
    acceptance = json.loads((root / "acceptance.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    index = json.loads((root / "index.json").read_text())
    assert canonical(config) == acceptance["contract_sha256"]
    assert sha(root / "index.json") == acceptance["committed_check_index_sha256"]
    assert (
        len(index) == 1
        and acceptance["completed_checks"] == 1
        and acceptance["status"] == "FAIL"
    )
    key, entry = next(iter(index.items()))
    path = root / entry["file"]
    assert sha(path) == entry["sha256"]
    record = json.loads(path.read_text())
    assert record["contract_sha256"] == canonical(config) and record["key"] == key
    result = record["result"]
    evidence = result["evidence"]
    placements = evidence["placements"]
    assert (
        evidence["preservation"]["preserved"]
        and evidence["preservation"]["before_sha256"]
        == evidence["preservation"]["after_sha256"]
    )
    design = config["design"]
    base = placements["Boundary"]
    ids = result["context"]["eligible_queries"]
    delta = config["delta"]
    tau = base["tau"]
    sim_tol = design["training_margin_rtol_delta"] * delta
    logit_tol = design["training_logit_rtol_scaled_delta"] * tau * delta
    shift = [p - pg for p, pg in zip(base["p_sim"], base["p_geom"])]
    native_scale = [ell - tau * p for ell, p in zip(base["ell_plus"], base["p_sim"])]
    bound = tau * sim_tol + logit_tol
    # Conservative guard for final FP32 products/subtractions, far smaller than
    # the observed native-logit discrepancies; this is not a changed tolerance.
    largest = (
        max(
            abs(x)
            for v in placements.values()
            for field in ["ell_plus", "e_aux"]
            for x in v[field]
        )
        + 1
    )
    roundoff_guard = 32 * 2.0 ** (math.floor(math.log2(largest)) - 23)
    arms = {}
    rows = []
    for arm, v in placements.items():
        assert (
            v["delta"] == delta
            and v["tau"] == tau
            and v["p_sim"] == base["p_sim"]
            and v["p_geom"] == base["p_geom"]
        )
        target = v["target_margin"]
        signed_errors = [r - target for r in v["r_train"]]
        geom_errors = [r - target for r in v["r_geom"]]
        geom_s = [pg - rg for pg, rg in zip(v["p_geom"], v["r_geom"])]
        auxiliary_path_error = [s - sg for s, sg in zip(v["s_aux"], geom_s)]
        residual = [err - sh for err, sh in zip(signed_errors, shift)]
        # Exact recorded-tensor decomposition (including final subtraction error).
        decomposition = [
            err - (sh - se + ge)
            for err, sh, se, ge in zip(
                signed_errors, shift, auxiliary_path_error, geom_errors
            )
        ]
        assert max(map(abs, decomposition)) < 1e-12
        logit_errors = [x - tau * target for x in v["g_logit"]]
        arms[arm] = dict(
            target=target,
            candidate_dtype=v["candidate_dtype"],
            max_fp32_geometry_error=max(map(abs, geom_errors)),
            max_auxiliary_similarity_path_error=max(map(abs, auxiliary_path_error)),
            max_training_error_unexplained_by_native_positive_offset=max(
                map(abs, residual)
            ),
            similarity_tolerance_failures=sum(abs(x) > sim_tol for x in signed_errors),
            logit_tolerance_failures=sum(abs(x) > logit_tol for x in logit_errors),
            wrong_similarity_sign_queries=[
                i for i, x in zip(ids, v["r_train"]) if target and x * target <= 0
            ],
            wrong_logit_sign_queries=[
                i for i, x in zip(ids, v["g_logit"]) if target and x * target <= 0
            ],
            similarity_margin_range=[min(v["r_train"]), max(v["r_train"])],
            logit_gap_range=[min(v["g_logit"]), max(v["g_logit"])],
            native_raw_exact_bf16=all(x == bf16(x) for x in v["p_sim"]),
            native_logits_exact_bf16_scaled_raw=all(
                e == bf16(p * tau) for p, e in zip(v["p_sim"], v["ell_plus"])
            ),
            auxiliary_logits_exact_fp32_scaled_similarity=all(
                e == f32(s * tau) for s, e in zip(v["s_aux"], v["e_aux"])
            ),
        )
        for j, query in enumerate(ids):
            rows.append(
                dict(
                    arm=arm,
                    query_index=query,
                    p_geom=v["p_geom"][j],
                    p_sim=v["p_sim"][j],
                    target_margin=target,
                    r_geom=v["r_geom"][j],
                    r_train=v["r_train"][j],
                    g_logit=v["g_logit"][j],
                    native_positive_offset=shift[j],
                    auxiliary_similarity_path_error=auxiliary_path_error[j],
                    signed_training_error=signed_errors[j],
                    native_logit_scaling_offset=native_scale[j],
                )
            )
    spacing = {}
    for left, right in [("Wrong", "Boundary"), ("Boundary", "Correct")]:
        gaps = [
            b - a
            for a, b in zip(placements[left]["r_train"], placements[right]["r_train"])
        ]
        spacing[left + "->" + right] = dict(
            minimum=min(gaps),
            maximum=max(gaps),
            max_error_from_delta=max(abs(x - delta) for x in gaps),
        )
    return dict(
        input=str(root),
        input_sha256={
            n: sha(root / n)
            for n in ["acceptance.json", "run_config.json", "index.json", entry["file"]]
        },
        contract_sha256=canonical(config),
        context=result["context"],
        state_restored=True,
        delta=delta,
        tau=tau,
        similarity_tolerance=sim_tol,
        logit_tolerance=logit_tol,
        arms=arms,
        adjacent_spacing=spacing,
        native_positive_offset=dict(
            minimum=min(shift),
            maximum=max(shift),
            median_absolute=statistics.median(map(abs, shift)),
        ),
        joint_contract=dict(
            necessary_bound=bound,
            roundoff_guard=roundoff_guard,
            native_scaling_offset_range=[min(native_scale), max(native_scale)],
            incompatible_queries=[
                i for i, x in zip(ids, native_scale) if abs(x) > bound + roundoff_guard
            ],
            derivation="g_logit - tau*r_train = ell_plus - tau*p_sim, apart from final FP32 rounding; simultaneous errors require |ell_plus-tau*p_sim| <= tau*eps_similarity + eps_logit",
        ),
        rows=rows,
        scope="Offline diagnostic of saved first-batch tensors only; no new preflight or training; v1 unchanged",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = audit(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: value[k]
                for k in ["delta", "tau", "arms", "adjacent_spacing", "joint_contract"]
            },
            indent=2,
        )
    )
