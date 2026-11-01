"""Authenticate full-population receipts and recheck saved numerical evidence.

This is a local archive audit, not an independent GPU replication.
"""

import argparse
import json
import math
from pathlib import Path

from scripts.audit_clip_placement_v2a import canonical, read_input, require, sha
from scripts.audit_clip_placement_v2c import validate_placements
from src.clip_placement_v2d import calibration_reference, completion, expected_keys

ROOT = Path(__file__).resolve().parents[1]


def validate_evidence(evidence, eligible, p, delta, gradient_contract):
    state = evidence["preservation"]
    require(
        state["preserved"] and state["before_sha256"] == state["after_sha256"],
        "State changed",
    )
    require(
        evidence["optimizer_updates"] == 0 and evidence["diagnostic_on_off_identical"],
        "Updates or diagnostic mutation",
    )
    require(evidence["empty"] == (not eligible), "Wrong eligibility")
    settings = evidence["fp32_settings"]
    require(
        settings["matmul_precision"] == "highest"
        and all(
            not settings[k]
            for k in (
                "autocast_enabled",
                "matmul_tf32",
                "cudnn_tf32",
                "bf16_reduction",
                "fp16_reduction",
            )
        ),
        "Wrong precision",
    )
    require(math.isfinite(evidence["native_loss"]), "Nonfinite native loss")
    require(
        set(evidence["auxiliary_losses"])
        == {"H-Uniform", "Wrong", "Boundary", "Correct"},
        "Missing auxiliary arm",
    )
    require(
        all(math.isfinite(v) for v in evidence["auxiliary_losses"].values()),
        "Nonfinite auxiliary loss",
    )
    if not eligible:
        require(
            evidence["auxiliary_gradient_norm"] == 0
            and evidence["gradient_cosine"] is None
            and all(v == 0 for v in evidence["auxiliary_losses"].values()),
            "Invalid empty auxiliary",
        )
        return 0
    validate_placements(evidence, len(eligible), p, delta)
    grads = evidence["feature_gradient_contract"]
    require(
        set(grads["per_query"]) == {str(q) for q in eligible}, "Missing exposure VJP"
    )
    for query in grads["per_query"].values():
        require(set(query) == {"Wrong", "Boundary", "Correct"}, "Missing VJP arm")
        for arm in query.values():
            require(set(arm) == {"source", "carrier", "query"}, "Missing VJP path")
            for measured in arm.values():
                require(
                    math.isfinite(measured["relative_vector_error"])
                    and measured["relative_vector_error"]
                    <= gradient_contract["vector_rtol"]
                    and math.isfinite(measured["reference_norm"])
                    and measured["reference_norm"] > gradient_contract["norm_min"]
                    and math.isfinite(measured["norm"])
                    and measured["norm"] > 0
                    and measured["cosine"] is not None
                    and measured["cosine"] >= gradient_contract["cosine_min"],
                    "VJP contract failed",
                )
    for name in ("tau", "logit_scale"):
        scale = grads["scale"][name]
        error = abs(scale["observed"] - scale["expected"])
        limit = gradient_contract["scale_atol"] + gradient_contract["scale_rtol"] * abs(
            scale["expected"]
        )
        require(
            math.isfinite(error)
            and error == scale["error"]
            and limit == scale["tolerance"]
            and error <= limit,
            "Native scale gradient failed",
        )
    for arm in ("Wrong", "Boundary", "Correct"):
        require(
            grads["scale"][arm]
            == {k: [True, True] for k in ("candidate", "partition", "loss")},
            "Auxiliary scale leak",
        )
    return 9 * len(eligible)


def audit(path):
    read = read_input(path)
    receipt = json.loads(read("acceptance.json"))
    config = json.loads(read("run_config.json"))
    index_bytes = read("index.json")
    index = json.loads(index_bytes)
    spec = json.loads((ROOT / "configs/clip_placement_v2d.json").read_text())
    manifest = json.loads((ROOT / "placement_v2d_source_manifest.json").read_text())
    p = json.loads((ROOT / spec["original_design"]).read_text())
    delta = json.loads((ROOT / p["gap_manifest"]).read_text())["delta"]
    require(
        config["specification"] == spec
        and config["implementation"] == manifest
        and config["original_design"] == p
        and config["delta"] == delta,
        "Issued configuration differs",
    )
    for name, digest in manifest.items():
        require(
            sha((ROOT / name).read_bytes()) == digest,
            "Issued implementation differs: " + name,
        )
    require(canonical(config) == receipt["contract_sha256"], "Wrong contract hash")
    require(
        sha(index_bytes) == receipt["committed_check_index_sha256"], "Wrong index hash"
    )
    require(
        receipt["status"] == "PASS" and receipt["operational_error"] is None,
        "No passing acceptance",
    )
    require(
        receipt["completed_checks"]
        == receipt["passing_checks"]
        == receipt["expected_checks"]
        == 624
        and receipt["execution_complete"]
        and receipt["launch_eligible"],
        "Incomplete receipt",
    )
    require(
        not receipt["d2_launched"]
        and receipt["optimizer_updates"] == receipt["calibration_refits"] == 0,
        "Unexpected training or recalibration",
    )
    summary, cells = calibration_reference(ROOT, spec)
    require(
        config["calibration_contract_sha256"] == summary["contract_sha256"],
        "Wrong calibration lineage",
    )
    coefficients = {f"{s}/{w}": cell["coefficients"] for (s, w), cell in cells.items()}
    require(
        json.loads(read("frozen_coefficients.json")) == coefficients,
        "Calibration coefficients changed",
    )
    require(set(index) == expected_keys(p), "Incomplete or unexpected grid")
    population = json.loads((ROOT / p["population"]).read_text())
    masks = {
        (r["training_seed"], r["stream_seed"], r["batch_offset"]): r
        for r in population["batches"]
    }
    records, exposures, vjps = {}, 0, 0
    for key, entry in index.items():
        data = read(entry["file"])
        require(sha(data) == entry["sha256"], "Wrong check content")
        envelope = json.loads(data)
        require(
            envelope["key"] == key and envelope["contract_sha256"] == canonical(config),
            "Wrong check lineage",
        )
        _, seed, stream, weight, offset = key.split("/")
        seed, weight, offset = int(seed), int(weight), int(offset)
        stream = None if stream == "None" else int(stream)
        batch = masks[seed, stream, offset]
        result = envelope["result"]
        require(
            result["context"]
            == dict(
                training_seed=seed,
                stream_seed=stream,
                weight_step=weight,
                batch_offset=offset,
                population=batch["population"],
                eligible_queries=batch["eligible_queries"],
            ),
            "Wrong exposure context",
        )
        require(result["status"] == "PASS", "Nonpassing batch")
        evidence = result["evidence"]
        vjps += validate_evidence(
            evidence, batch["eligible_queries"], p, delta, spec["gradient_contract"]
        )
        exposures += len(batch["eligible_queries"])
        if stream is None:
            archived = cells[seed, weight]["unweighted"][offset]
            require(
                evidence["calibration_forward_reproduced"], "Calibration replay missing"
            )
            for name in ("native_loss", "placements", "fp32_settings", "empty"):
                require(
                    evidence[name] == archived[name],
                    "Calibration forward changed: " + name,
                )
        records[key] = result
    require(completion(records, p) == "PASS", "Coverage not complete")
    return dict(
        status="PASS",
        batch_checks=len(records),
        eligible_state_exposures=exposures,
        candidate_vjp_checks=vjps,
        coefficients=24,
        optimizer_updates=0,
        d2_launched=False,
        contract_sha256=receipt["contract_sha256"],
        limitation="Rechecked archived measurements only; no independent GPU execution or future feasibility claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.path), indent=2))
