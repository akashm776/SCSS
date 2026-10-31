"""Verify archived v2c calibration records and recompute pressure summaries.

This audit reads saved norms and placement measurements. It does not execute the
model or recompute GPU gradients.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

from scripts.audit_clip_placement_v2a import canonical, read_input, require, sha
from src.clip_placement_v2c import gradient_reference

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["H-Uniform", "Wrong", "Boundary", "Correct"]
SIGNS = {"Wrong": -1, "Boundary": 0, "Correct": 1}
BATCHES_PER_CELL = 4
EXPECTED_CELLS = 6
EXPECTED_BATCHES = EXPECTED_CELLS * BATCHES_PER_CELL


def validate_run(receipt: dict, config: dict, index: dict, index_bytes: bytes) -> None:
    """Authenticate the issued implementation and the complete calibration grid."""
    specification = json.loads((ROOT / "configs/clip_placement_v2c.json").read_text())
    manifest = json.loads((ROOT / "placement_v2c_source_manifest.json").read_text())
    require(
        receipt["status"] == "PASS" and receipt["operational_error"] is None,
        "Not an uninterrupted completed PASS",
    )
    require(
        config["specification"] == specification
        and config["implementation"] == manifest,
        "Issued specification/implementation differs",
    )
    require(canonical(config) == receipt["contract_sha256"], "Contract hash differs")
    require(
        sha(index_bytes) == receipt["committed_check_index_sha256"],
        "Index hash differs",
    )
    for name, digest in manifest.items():
        require(
            sha((ROOT / name).read_bytes()) == digest, "Issued source changed: " + name
        )
    gradient_reference(ROOT, specification)

    design = config["original_design"]
    require(
        design == json.loads((ROOT / specification["original_design"]).read_text()),
        "Original design differs",
    )
    require(
        config["delta"]
        == json.loads((ROOT / design["gap_manifest"]).read_text())["delta"],
        "Gap differs",
    )
    expected_keys = {
        f"calibration/{seed}/{weight}"
        for seed in specification["training_seeds"]
        for weight in specification["checkpoint_steps"]
    }
    require(set(index) == expected_keys, "Incomplete cell grid")
    require(
        receipt["completed_cells"]
        == receipt["passing_cells"]
        == receipt["expected_cells"]
        == EXPECTED_CELLS,
        "Receipt coverage differs",
    )
    require(
        receipt["calibration_batches"]
        == receipt["verification_batches"]
        == EXPECTED_BATCHES,
        "Wrong batch counts",
    )
    require(
        receipt["optimizer_updates"] == 0
        and not receipt["d2_launched"]
        and not receipt["launch_eligible"],
        "Unexpected training permission",
    )


def validate_cell(envelope: dict, key: str, contract_digest: str) -> dict:
    seed, weight = map(int, key.split("/")[1:])
    result = envelope["result"]
    require(
        envelope["key"] == key and envelope["contract_sha256"] == contract_digest,
        "Wrong cell lineage",
    )
    require(
        result["status"] == "PASS"
        and result["context"] == dict(training_seed=seed, weight_step=weight),
        "Wrong cell identity/status",
    )
    evidence = result["evidence"]
    require(
        evidence["optimizer_updates"] == 0 and len(evidence["preservation"]) == 8,
        "Updates or missing restoration records",
    )
    for state in evidence["preservation"]:
        require(
            state["preserved"] and state["before_sha256"] == state["after_sha256"],
            "State preservation failed",
        )
    require(
        len(evidence["unweighted"])
        == len(evidence["verification"])
        == BATCHES_PER_CELL,
        "Missing scheduled batches",
    )
    return evidence


def validate_batch(
    row: dict, eligible_ids: list[int], offset: int, design: dict
) -> None:
    require(
        row["batch_offset"] == offset
        and row["eligible_queries"] == eligible_ids
        and row["eligible"] == len(eligible_ids),
        "Wrong eligibility",
    )
    require(row["empty"] == (not eligible_ids), "Wrong empty flag")
    settings = row["fp32_settings"]
    reduced_precision_flags = (
        "autocast_enabled",
        "matmul_tf32",
        "cudnn_tf32",
        "bf16_reduction",
        "fp16_reduction",
    )
    require(
        settings["matmul_precision"] == "highest"
        and all(not settings[key] for key in reduced_precision_flags),
        "FP32 context differs",
    )
    require(
        math.isfinite(row["native_norm"])
        and row["native_norm"] > design["gradient_norm_min"],
        "Invalid native norm",
    )
    require(
        set(row["auxiliary_norms"]) == set(ARMS)
        and row["scale_disconnected"] == dict.fromkeys(ARMS, True),
        "Missing arms or scale leakage",
    )
    for arm, norm in row["auxiliary_norms"].items():
        require(
            math.isfinite(norm)
            and (norm > design["gradient_norm_min"] if eligible_ids else norm == 0),
            "Invalid auxiliary norm",
        )
        cosine = row["native_auxiliary_cosines"][arm]
        require(
            (
                (cosine is None)
                if not eligible_ids
                else (math.isfinite(cosine) and abs(cosine) <= 1 + 1e-12)
            ),
            "Invalid cosine",
        )


def validate_placements(
    row: dict, count: int, design: dict, delta: float
) -> tuple[list, list, list]:
    geometry_errors, similarity_errors, logit_errors = [], [], []
    if not count:
        return geometry_errors, similarity_errors, logit_errors

    require(row["placement_contract"]["status"] == "PASS", "Placement not passed")
    placements = row["placements"]
    for arm, sign in SIGNS.items():
        placement = placements[arm]
        scale = placement["tau"]
        target = sign * delta
        require(
            math.isfinite(scale) and scale > 0 and placement["delta"] == delta,
            "Invalid scale/gap",
        )
        require(
            all(
                len(placement[key]) == count for key in ("r_geom", "r_train", "g_logit")
            ),
            "Incomplete exposure arrays",
        )
        for geometry, similarity, logit in zip(
            placement["r_geom"], placement["r_train"], placement["g_logit"]
        ):
            geometry_error = abs(geometry - target)
            similarity_error = abs(similarity - target)
            logit_error = abs(logit - scale * target)
            require(
                geometry_error
                <= max(
                    design["fp32_margin_atol"], design["fp32_margin_rtol_delta"] * delta
                ),
                "Geometry margin failed",
            )
            require(
                similarity_error <= design["training_margin_rtol_delta"] * delta
                and logit_error
                <= design["training_logit_rtol_scaled_delta"] * scale * delta,
                "Actual placement failed",
            )
            require(
                not sign or similarity * sign > 0 and logit * sign > 0,
                "Wrong placement sign",
            )
            geometry_errors.append(geometry_error)
            similarity_errors.append(similarity_error)
            logit_errors.append(logit_error)
    for left, right in (("Wrong", "Boundary"), ("Boundary", "Correct")):
        require(
            all(
                right_margin - left_margin
                >= design["minimum_adjacent_separation_delta"] * delta
                for left_margin, right_margin in zip(
                    placements[left]["r_train"], placements[right]["r_train"]
                )
            ),
            "Target separation failed",
        )
    return geometry_errors, similarity_errors, logit_errors


def validate_replay(evidence: dict) -> None:
    for unweighted, weighted in zip(evidence["unweighted"], evidence["verification"]):
        for field in (
            "native_norm",
            "native_loss",
            "eligible",
            "placements",
            "fp32_settings",
        ):
            require(
                unweighted[field] == weighted[field],
                "Weighted replay differs: " + field,
            )


def recompute_coefficients(evidence: dict, design: dict) -> tuple[dict, list[float]]:
    """Reproduce recorded coefficients from unweighted and weighted batch norms."""
    require(set(evidence["coefficients"]) == set(ARMS), "Missing coefficients")
    coefficients, batch_pressures = {}, []
    target_ratio = design["target_ratio"]
    for arm in ARMS:
        raw_ratios = [
            batch["auxiliary_norms"][arm] / batch["native_norm"]
            for batch in evidence["unweighted"]
        ]
        weighted_ratios = [
            batch["auxiliary_norms"][arm] / batch["native_norm"]
            for batch in evidence["verification"]
        ]
        raw_mean = statistics.fmean(raw_ratios)
        weighted_mean = statistics.fmean(weighted_ratios)
        error = abs(weighted_mean / target_ratio - 1)
        require(
            raw_mean > design["gradient_norm_min"]
            and error <= design["calibration_relative_tolerance"],
            "Calibration pressure fails",
        )
        expected = dict(
            coefficient=target_ratio / raw_mean,
            unweighted_scheduled_ratio=raw_mean,
            batch_ratios=raw_ratios,
            unweighted_treated_ratio=statistics.fmean(
                ratio
                for ratio, batch in zip(raw_ratios, evidence["unweighted"])
                if batch["eligible"]
            ),
            verified_batch_ratios=weighted_ratios,
            verified_scheduled_ratio=weighted_mean,
            relative_error=error,
            verified_treated_ratio=statistics.fmean(
                ratio
                for ratio, batch in zip(weighted_ratios, evidence["verification"])
                if batch["eligible"]
            ),
        )
        require(
            evidence["coefficients"][arm] == expected,
            "Coefficient or verification summary does not match saved norms",
        )
        coefficients[arm] = expected
        batch_pressures.extend(weighted_ratios)
    return coefficients, batch_pressures


def audit(path: str | Path) -> dict:
    read = read_input(path)
    raw = {
        name: read(name)
        for name in ("acceptance.json", "run_config.json", "index.json")
    }
    receipt = json.loads(raw["acceptance.json"])
    config = json.loads(raw["run_config.json"])
    index = json.loads(raw["index.json"])
    validate_run(receipt, config, index, raw["index.json"])

    design, delta = config["original_design"], config["delta"]
    contract_digest = canonical(config)
    population = json.loads((ROOT / design["population"]).read_text())
    masks = {
        (batch["training_seed"], batch["batch_offset"]): batch["eligible_queries"]
        for batch in population["batches"]
        if batch["stream_seed"] is None
    }
    cells, batch_pressures = [], []
    geometry_errors, similarity_errors, logit_errors = [], [], []
    exposures = empty_batches = 0
    for key, entry in sorted(index.items()):
        raw[entry["file"]] = read(entry["file"])
        require(sha(raw[entry["file"]]) == entry["sha256"], "Cell hash differs")
        envelope = json.loads(raw[entry["file"]])
        evidence = validate_cell(envelope, key, contract_digest)
        seed, weight = map(int, key.split("/")[1:])
        for phase in ("unweighted", "verification"):
            for offset, row in enumerate(evidence[phase]):
                eligible_ids = masks[seed, offset]
                validate_batch(row, eligible_ids, offset, design)
                geometry, similarity, logit = validate_placements(
                    row, len(eligible_ids), design, delta
                )
                # Both passes must satisfy the contract; summary counts cover only the first.
                if phase == "unweighted":
                    exposures += len(eligible_ids)
                    empty_batches += not eligible_ids
                    geometry_errors.extend(geometry)
                    similarity_errors.extend(similarity)
                    logit_errors.extend(logit)
        validate_replay(evidence)
        coefficients, pressures = recompute_coefficients(evidence, design)
        batch_pressures.extend(pressures)
        cells.append(dict(seed=seed, weight_step=weight, coefficients=coefficients))

    scheduled_means = [
        coefficient["verified_scheduled_ratio"]
        for cell in cells
        for coefficient in cell["coefficients"].values()
    ]
    return dict(
        status="PASS",
        contract_sha256=contract_digest,
        input_sha256={name: sha(contents) for name, contents in raw.items()},
        cells=cells,
        coefficients=24,
        calibration_batches=EXPECTED_BATCHES,
        verification_batches=EXPECTED_BATCHES,
        eligible_exposures=exposures,
        placement_checks=3 * exposures,
        empty_batches=empty_batches,
        state_preserved=True,
        optimizer_updates=0,
        scheduled_pressure_range=[min(scheduled_means), max(scheduled_means)],
        individual_batch_pressure_range=[min(batch_pressures), max(batch_pressures)],
        max_relative_pressure_error=max(
            abs(mean / design["target_ratio"] - 1) for mean in scheduled_means
        ),
        max_geometry_error=max(geometry_errors),
        max_similarity_error=max(similarity_errors),
        max_logit_error=max(logit_errors),
        limitation="Authentication and recomputation from saved metrics; no independent GPU-gradient recomputation. Calibration batches only; no training utility or full-population acceptance.",
    )


def report(v):
    lines = [
        "# Authenticated v2c calibration audit",
        "",
        "PASS: all six seed/state cells, 24 coefficients, 24 scheduled batches and 24 weighted-gradient replays.",
        f"Eligibility: {v['eligible_exposures']} state/exposure instances, {v['placement_checks']} arm/exposure checks; {v['empty_batches']} empty batches.",
        "",
        f"Verified scheduled pressure range: {100*v['scheduled_pressure_range'][0]:.6f}%–{100*v['scheduled_pressure_range'][1]:.6f}%.",
        f"Maximum relative error against the 2% target: {100*v['max_relative_pressure_error']:.6f}% (allowed 5%).",
        f"Individual batch pressures: {100*v['individual_batch_pressure_range'][0]:.4f}%–{100*v['individual_batch_pressure_range'][1]:.4f}%.",
        "",
        "| Seed | State | H-Uniform alpha | Wrong alpha | Boundary alpha | Correct alpha |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in v["cells"]:
        lines.append(
            f"| {cell['seed']} | {cell['weight_step']} | "
            + " | ".join(f"{cell['coefficients'][a]['coefficient']:.8f}" for a in ARMS)
            + " |"
        )
    lines += [
        "",
        f"Maximum errors: ideal margin {v['max_geometry_error']:.9g}; actual similarity margin {v['max_similarity_error']:.9g}; actual logit gap {v['max_logit_error']:.9g}.",
        "Weighted replay preserved native norms/losses and placement observables exactly. All 48 state-restoration records match. Auxiliary scale disconnection passed.",
        "",
        "The 2% target is a four-scheduled-batch average, not a per-batch equality or an optimizer-displacement match. All batches were treated, so conditional and scheduled means coincide.",
        "Larger placement-arm coefficients compensate for smaller raw gradients; they do not establish greater training utility. Keep these coefficients frozen for the next stage.",
        "",
        v["limitation"],
        "",
        "Next: full-population D2-P v2 audit. Zero optimizer updates; D2 training remains gated.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.is_file():
        result["source_zip_sha256"] = sha(args.input.read_bytes())
        shutil.copyfile(args.input, args.output / "source.zip")
    (args.output / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    summary = report(result)
    (args.output / "REPORT.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
