"""Verify the completed weight/LR experiment and export small, caption-free evidence.

CPU only. Full input authenticates every file and recomputes 48 cached endpoints.
Clone-only mode verifies retained bytes and recomputes factorial arithmetic; it
cannot evaluate omitted caches or recreate the GPU training invariants.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.analyze_clip_image_influence import query_losses, unpack
from scripts.analyze_clip_policy_robustness import (
    checked_json,
    close,
    require,
    verify_input,
)
from scripts.archive_clip_followup_evidence import digest
from scripts.archive_clip_october_evidence import (
    locate,
    retain,
    verify_export,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
NAME = "clip_weight_lr_2026-10"
SPEC = (
    "clip_weight_lr_20261007T024147_218129Z",
    "55d063b0b9d3dff61da1a353f7041573fcddc712da95c5279fd6f1ee422286cc",
    "9d1f48b37336cc190ad7be33ce77dbc325bc1f7422e0742ab86d26bba9bc07b0",
)
SOURCE_MANIFEST_SHA = "e660df656d94e777189a8ff6ebce32e0ea49f81c11a10e6a9f03ffc7fd33ad53"
CELLS = ["w100_lr100", "w100_lr500", "w500_lr100", "w500_lr500"]
ARMS = ["native", "sustained"]


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def contrasts(values):
    """Independent arithmetic on four sustained-minus-native effects."""
    ee, el, le, ll = [values[k] for k in CELLS]
    return dict(
        weight_main_effect=(le + ll - ee - el) / 2,
        lr_main_effect=(el + ll - ee - le) / 2,
        weight_x_lr_interaction=ll - le - el + ee,
        weight_effect_at_early_lr=le - ee,
        weight_effect_at_late_lr=ll - el,
        lr_effect_at_early_weights=el - ee,
        lr_effect_at_late_weights=ll - le,
    )


def validate(files, *, full):
    read = lambda name: checked_json(files[name])
    run, done = read("run_manifest.json"), read("completion.json")
    require(
        run["run_id"] == SPEC[0]
        and run["bundle_id"] == SPEC[2]
        and run["status"] == "complete_pending_drive_flush",
        "Wrong run/bundle/status",
    )
    require(
        done["status"] == "complete"
        and [
            done[k]
            for k in [
                "matched_sets",
                "cells",
                "branches",
                "endpoint_caches",
                "point_records",
                "trunk_optimizer_steps",
                "branch_optimizer_steps",
                "optimizer_steps",
            ]
        ]
        == [6, 24, 48, 48, 288, 1500, 2400, 3900],
        "Incomplete run",
    )
    require(
        digest(files["source_manifest.json"])["sha256"] == SOURCE_MANIFEST_SHA,
        "Wrong source manifest",
    )
    for path, sha in read("source_manifest.json").items():
        require(
            digest(files["source_overlay/" + path])["sha256"] == sha,
            "Changed source: " + path,
        )
    p = read("source_overlay/configs/clip_weight_lr.json")
    ref = read("source_overlay/configs/clip_weight_lr_reference.json")
    native = read("source_overlay/configs/clip_optimizer_reset_reference.json")
    require(
        read("protocol.json") == p
        and read("native_replay_reference.json") == native
        and read("early_diagonal_replay_reference.json") == ref,
        "Changed protocol/reference",
    )
    for name in [
        "split_plan.json",
        "branch_plans.json",
        "training_exclusion_indices.json",
    ]:
        require(canonical_sha(read(name)) == native[name], "Changed replay plan")
    common = {
        f"{s}:{u}": read("branch_plans.json")[f"{s}:100:{u}"]
        for s in p["training_seeds"]
        for u in p["stream_seeds"]
    }
    require(read("common_branch_plans.json") == common, "Changed common batch plan")
    rows, expected_paths, row_hashes, cache_count = {}, set(), set(), 0
    for seed in p["training_seeds"]:
        folder = f"seed_{seed}/"
        for name in ["trunk_audit.json"] + (
            ["data_roles.json", "training_batches.json"] if full else []
        ):
            require(
                canonical_sha(read(folder + name)) == native["seeds"][str(seed)][name],
                "Native trunk/data replay differs",
            )
        for step in [100, 500]:
            require(
                read(folder + f"start_{step:06d}.json")
                == native["seeds"][str(seed)]["starts"][str(step)],
                "Native start replay differs",
            )
        for stream in p["stream_seeds"]:
            prefix = folder + f"stream_{stream}/"
            expected_paths.add(prefix + "result.json")
            row = read(prefix + "result.json")
            require(
                (row["training_seed"], row["stream_seed"]) == (seed, stream)
                and row["starting_state_history"] == "native_only"
                and row["prior_auxiliary_updates"] == 0
                and row["common_rng_source_step"] == 100,
                "Wrong continuation identity",
            )
            row_hashes.add(row["report_row_order_sha256"])
            expected_batch = native["pairs"][f"{seed}:100:{stream}"][
                "training_batches_sha256"
            ]
            require(
                row["batch_records_sha256"] == expected_batch, "Wrong batch identity"
            )
            if full:
                batches = read(prefix + "training_batches.json")
                require(
                    canonical_sha(batches) == expected_batch
                    and batches["caption_epoch"] == row["common_caption_epoch"],
                    "Changed batches/captions",
                )
            audit = row["audit"]
            require(
                audit["actual_optimizer_steps"] == 400
                and audit["report_controls_treatment"] is False
                and all(
                    audit[k] is True
                    for k in [
                        "early_diagonal_exact_replay",
                        "source_immutable",
                        "frozen_parameters_unchanged",
                        "matched_rng_across_all_cells",
                        "matched_lr_within_lr_level",
                        "fresh_adamw_all_branches",
                        "observations_preserve_state",
                    ]
                ),
                "Failed runtime audit",
            )
            require(set(row["cells"]) == set(CELLS), "Wrong cell grid")
            rngs, rates, weights, schedulers, initial_reports = None, {}, {}, {}, {}
            for key, cell in row["cells"].items():
                w, lr = cell["weight_step"], cell["lr_step"]
                require(
                    key == f"w{w}_lr{lr}" and set(cell["points"]) == set(ARMS),
                    "Wrong cell identity",
                )
                require(
                    cell["starts"]["native"] == cell["starts"]["sustained"],
                    "Unmatched treatment start",
                )
                for arm in ARMS:
                    start, reset = cell["starts"][arm], cell["resets"][arm]
                    weights.setdefault(w, start["weights_sha256"])
                    schedulers.setdefault(
                        lr,
                        (
                            start["scheduler_sha256"],
                            start["optimizer_sha256"],
                            start["learning_rates"],
                        ),
                    )
                    require(
                        start["weights_sha256"] == weights[w]
                        and (
                            start["scheduler_sha256"],
                            start["optimizer_sha256"],
                            start["learning_rates"],
                        )
                        == schedulers[lr]
                        and start["rng_sha256"]
                        == row["cells"][CELLS[0]]["starts"]["native"]["rng_sha256"],
                        "Unmatched factorial starts",
                    )
                    coefficient = p["coefficient"] if arm == "sustained" else 0.0
                    require(
                        read(prefix + f"start_{key}_{arm}.json")
                        == dict(
                            start=start,
                            reset=reset,
                            weight_step=w,
                            lr_step=lr,
                            coefficient=coefficient,
                        ),
                        "Standalone start differs",
                    )
                    require(
                        reset["cleared_parameter_states"] == 35
                        and reset["old_step_counts"] == [w]
                        and reset["new_parameter_states"] == 0
                        and reset["weights_scheduler_groups_rng_gradients_unchanged"]
                        is True
                        and reset["lazy_initialization_next_update_step"] == 1,
                        "Invalid optimizer reset",
                    )
                    traces = cell["traces"][arm]
                    require(
                        len(traces) == 50
                        and all(t["coefficient"] == coefficient for t in traces),
                        "Bad treatment trace",
                    )
                    branch_rng = [t["rng_after_sha256"] for t in traces]
                    branch_lr = [t["learning_rates"] for t in traces]
                    rngs = branch_rng if rngs is None else rngs
                    rates.setdefault(lr, branch_lr)
                    require(
                        branch_rng == rngs
                        and branch_lr == rates[lr]
                        and branch_lr[0] == start["learning_rates"],
                        "Unmatched RNG/LR traces",
                    )
                    require(
                        set(cell["points"][arm])
                        == {str(t) for t in p["observation_steps"]},
                        "Wrong observations",
                    )
                    for t in p["observation_steps"]:
                        require(
                            read(prefix + f"point_{key}_{arm}_{t:03d}.json")
                            == cell["points"][arm][str(t)],
                            "Standalone observation differs",
                        )
                    initial = cell["points"][arm]["0"]["report"]
                    initial_reports.setdefault(w, initial)
                    require(
                        initial == initial_reports[w],
                        "LR/treatment changed initial report",
                    )
                    if key == CELLS[0]:
                        expected = ref["pairs"][f"{seed}:{stream}"]["arms"][arm]
                        require(
                            cell["points"][arm] == expected["points"]
                            and canonical_sha(traces) == expected["traces_sha256"],
                            "Early diagonal replay differs",
                        )
                    if full:
                        meta = read(prefix + f"features_{key}_{arm}.json")
                        logits = unpack(
                            files[prefix + f"features_{key}_{arm}.npz"],
                            meta,
                            row["report_row_order_sha256"],
                        )
                        report = cell["points"][arm]["50"]["report"]
                        require(
                            meta["feature_sha256"] == report["features_sha256"],
                            "Report/cache identity differs",
                        )
                        close(
                            float(query_losses(logits)[:, 2].mean()),
                            report["native_loss"],
                            "Cached loss",
                        )
                        t2i, i2t = [
                            100 * float(np.mean(logits.argmax(axis) == np.arange(512)))
                            for axis in [1, 0]
                        ]
                        for metric, value in [
                            ("text_to_image_r1_percent", t2i),
                            ("image_to_text_r1_percent", i2t),
                            ("mean_r1_percent", (t2i + i2t) / 2),
                        ]:
                            close(value, report["pool_retrieval"][metric], "Cached R1")
                        cache_count += 1
            rows[seed, stream] = row
    require(
        {n for n in files if n.endswith("/result.json")} == expected_paths
        and len(row_hashes) == 1,
        "Unexpected result grid/report identity",
    )
    summary = read("comparison.json")
    seed_summaries = {s["training_seed"]: s for s in summary["per_seed"]}
    require(
        len(seed_summaries) == len(summary["per_seed"]) == 3
        and set(seed_summaries) == set(p["training_seeds"]),
        "Wrong seed summary grid",
    )
    for t in p["observation_steps"]:
        for metric in ["loss", "r1_pp"]:
            per_seed = {}
            for seed in p["training_seeds"]:
                effects = {}
                for key in CELLS:
                    values = []
                    for stream in p["stream_seeds"]:
                        reports = rows[seed, stream]["cells"][key]["points"]
                        get = lambda arm: (
                            reports[arm][str(t)]["report"]["native_loss"]
                            if metric == "loss"
                            else reports[arm][str(t)]["report"]["pool_retrieval"][
                                "mean_r1_percent"
                            ]
                        )
                        values.append(get("sustained") - get("native"))
                    effects[key] = fmean(values)
                per_seed[seed] = effects
            aggregates = [
                (seed_summaries[s]["timecourse"][str(t)], e)
                for s, e in per_seed.items()
            ]
            aggregates.append(
                (
                    summary["timecourse"][str(t)],
                    {k: fmean(e[k] for e in per_seed.values()) for k in CELLS},
                )
            )
            for reported, effects in aggregates:
                for key, value in effects.items():
                    close(value, reported["cell_effects"][key][metric], "Cell effect")
                for key, value in contrasts(effects).items():
                    close(
                        value, reported["contrasts"][metric][key], "Factorial contrast"
                    )
    for k, v in summary["primary_values"].items():
        close(v, summary["timecourse"]["50"]["contrasts"]["loss"][k], "Primary value")
    require(cache_count == (48 if full else 0), "Wrong cache count")
    return cache_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiment_results")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = args.output_root / NAME
    if args.verify_only:
        verify_export(target, SPEC)
        audit = checked_json((target / "export_audit.json").read_bytes())
        validate({n: (target / n).read_bytes() for n in audit["files"]}, full=False)
        print(
            "Retained replay records and all seed/timecourse factorial arithmetic verified; caches omitted."
        )
        return
    require(not target.exists(), "Refusing to overwrite " + str(target))
    files, provenance = verify_input(
        locate(args.downloads, SPEC[0]),
        dict(source_run=SPEC[0], backup_manifest_sha256=SPEC[1]),
    )
    count = validate(files, full=True)
    selected = {n: b for n, b in files.items() if retain(n, b)}
    audit = dict(
        **provenance,
        bundle_id=SPEC[2],
        status="small_evidence_verified",
        experiment_complete=True,
        endpoint_caches_recomputed=count,
        omitted=sorted(set(files) - set(selected)),
        files={n: digest(b) for n, b in selected.items()},
        scope="Every full-input manifest member verified; 48 cached losses/R1 recomputed; native "
        "and early-diagonal replay records, matching/reset audits and factorial arithmetic checked. "
        "Caption-bearing inputs and NPZ caches omitted. Clone-only mode cannot recompute caches "
        "or reexecute training invariants. No proof of remote Drive flush or independent replication.",
    )
    for name, data in selected.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_json(target / "export_audit.json", audit)
    verify_export(target, SPEC)
    print(
        f"{count} cached endpoints recomputed; replay records and factorial contrasts verified."
    )


if __name__ == "__main__":
    main()
