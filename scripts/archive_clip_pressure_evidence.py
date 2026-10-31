"""Verify the completed pressure experiment and export small, caption-free evidence.

CPU only. Full input authenticates every file and recomputes 84 cached endpoints.
Clone-only mode verifies retained bytes and recomputes pressure arithmetic; it
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
NAME = "clip_pressure_2026-10"
SPEC = (
    "clip_pressure_20261007T235549_427102Z",
    "10df8ac016976a09f9c2a84b5b97d73999ccf6f2c79298e176b30030e3131a2d",
    "79f1ab673de6bc888b8dd85a56c04c01e5d481da4533b33700e1d607012424ce",
)
SOURCE_MANIFEST_SHA = "9397aee3cc8da15f03634c4270ebf5ecacf221f5334376b9baadaa8f23cea150"
CELLS = ["w100", "w500"]
ARMS = [
    "native",
    "historical_hard",
    "historical_semihard",
    "hard_low",
    "semihard_low",
    "hard_high",
    "semihard_high",
]
REPLAY = ARMS[:3]
CALIBRATED = ARMS[3:]
CORE_TRACE_KEYS = [
    "native_loss",
    "total_loss",
    "gradient_norm_before_clip",
    "learning_rates",
    "coefficient",
    "rng_after_sha256",
]


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def verify_summary(rows, summary, p):
    """Independently recompute the frozen estimands, nesting streams within seeds."""
    require(
        summary["primary"] == p["analysis"]["primary"]
        and [
            summary[k]
            for k in [
                "matched_sets",
                "cells",
                "branches",
                "replication_units",
                "trunk_optimizer_steps",
                "branch_optimizer_steps",
                "optimizer_steps",
            ]
        ]
        == [6, 12, 84, 3, 1500, 4200, 5700],
        "Wrong summary identity/budget",
    )
    seeds = {s["training_seed"]: s for s in summary["per_seed"]}
    require(
        len(seeds) == len(summary["per_seed"]) == 3
        and set(seeds) == set(p["training_seeds"]),
        "Wrong summary seeds",
    )
    for t in p["observation_steps"]:
        per_seed = {}
        for seed in p["training_seeds"]:
            effects = {}
            for key in CELLS:
                effects[key] = {}
                for arm in ARMS:
                    vals = {"loss": [], "r1_pp": []}
                    for stream in p["stream_seeds"]:
                        pts = rows[seed, stream]["cells"][key]["points"]
                        a, n = [pts[x][str(t)]["report"] for x in [arm, "native"]]
                        vals["loss"].append(a["native_loss"] - n["native_loss"])
                        vals["r1_pp"].append(
                            a["pool_retrieval"]["mean_r1_percent"]
                            - n["pool_retrieval"]["mean_r1_percent"]
                        )
                    effects[key][arm] = {m: fmean(v) for m, v in vals.items()}
            per_seed[seed] = effects
        avg = {
            w: {
                a: {
                    m: fmean(e[w][a][m] for e in per_seed.values())
                    for m in ["loss", "r1_pp"]
                }
                for a in ARMS
            }
            for w in CELLS
        }
        for reported, effects in [
            (seeds[s]["timecourse"][str(t)], e) for s, e in per_seed.items()
        ] + [(summary["timecourse"][str(t)], avg)]:
            for w in CELLS:
                for a in ARMS:
                    for m, v in effects[w][a].items():
                        close(v, reported["effects"][w][a][m], "Arm effect")
            cc = reported["contrasts"]
            for m in ["loss", "r1_pp"]:
                q = {
                    level: {
                        w: effects[w]["semihard_" + level][m]
                        - effects[w]["hard_" + level][m]
                        for w in CELLS
                    }
                    for level in ["low", "high"]
                }
                for level in q:
                    for w in CELLS:
                        close(q[level][w], cc["Q"][level][w][m], "Q")
                    close(
                        q[level]["w500"] - q[level]["w100"],
                        cc["age_interaction_by_target"][level][m],
                        "Age interaction",
                    )
                close(
                    fmean(q[level]["w500"] for level in q),
                    cc["late_geometry_main_effect"][m],
                    "Primary contrast",
                )
                close(
                    q["high"]["w500"] - q["low"]["w500"],
                    cc["late_construction_x_pressure"][m],
                    "Pressure interaction",
                )
                close(
                    fmean(q[level]["w500"] - q[level]["w100"] for level in q),
                    cc["mean_age_interaction"][m],
                    "Mean age interaction",
                )
    close(
        summary["primary_value"],
        summary["timecourse"]["50"]["contrasts"]["late_geometry_main_effect"]["loss"],
        "Primary",
    )


def verify_frozen_inputs(files, p):
    read = lambda n: checked_json(files[n])
    freeze = read("freeze_manifest.json")
    require(
        freeze == read("source_overlay/configs/clip_pressure_freeze_manifest.json")
        and digest(files["source_overlay/configs/clip_pressure_freeze_manifest.json"])[
            "sha256"
        ]
        == "3d7cdf94e080c646197935c43147daafeb1953933b512ab3548246ce19e7a954",
        "Freeze manifest changed",
    )
    for name, sha in freeze["files"].items():
        require(
            digest(files["source_overlay/" + name])["sha256"] == sha,
            "Frozen file changed",
        )
    require(
        read("calibration_plan.json")
        == read("source_overlay/" + p["calibration"]["plan"]),
        "Calibration plan changed",
    )
    runtime = {
        **read("source_overlay/configs/clip_difficulty.json"),
        **p,
        "common_stream_checkpoint_step": p["common_rng_source_step"],
        "primary": p["analysis"]["primary"],
        "trunk_optimizer_steps": 1500,
        "branch_optimizer_steps": 4200,
        "optimizer_step_budget": 5700,
    }
    require(read("runtime_protocol.json") == runtime, "Expanded protocol differs")
    done = read("completion.json")
    require(
        done["initial_calibration_verified"] is True
        and done["coefficients_fixed"] is True
        and done["calibration_states"] == 6
        and done["calibration_optimizer_steps"] == 0,
        "Failed calibration completion",
    )
    split, plans = read("split_plan.json"), read("common_branch_plans.json")
    for seed in p["training_seeds"]:
        used = {
            i
            for u in p["stream_seeds"]
            for batch in plans[f"{seed}:{u}"]
            for i in batch
        }
        eligible = set(split["training"]) - used - set(split["frozen_probes"])
        ordered = sorted(
            eligible,
            key=lambda i: (
                hashlib.sha256(
                    f"scss-pressure-calibration-v1:{seed}:{i}".encode()
                ).hexdigest(),
                i,
            ),
        )[:256]
        actual = read("calibration_plan.json")["seeds"][str(seed)]
        require(
            actual
            == dict(
                eligible_count=len(eligible),
                eligible_indices_sorted=sorted(eligible),
                batches=[ordered[i : i + 64] for i in range(0, 256, 64)],
            ),
            "Calibration membership differs",
        )
        require(
            not set(ordered) & (set(split["report"]) | set(split["old_holdout"])),
            "Calibration report overlap",
        )


def verify_calibration(files, seed, p, *, full):
    read = lambda n: checked_json(files[n])
    prefix = f"seed_{seed}/"
    cal = read(prefix + "calibration.json")
    require(
        cal["training_seed"] == seed
        and cal["optimizer_updates"] == 0
        and set(cal["weights"]) == {"100", "500"}
        and cal["calibration_sha256"] == canonical_sha(cal["weights"]),
        "Invalid calibration identity",
    )
    result = {}
    for w, record in cal["weights"].items():
        require(
            read(prefix + f"calibration_w{w}.json") == record
            and record["status"] == "complete"
            and record["training_state_preserved"] is True
            and record["optimizer_updates"] == 0
            and record["state_before_sha256"] == record["state_after_sha256"],
            "Failed calibration/state audit",
        )
        require(
            len(record["unweighted"]) == len(record["verification"]) == 4,
            "Wrong calibration batch count",
        )
        for phase in ["unweighted", "verification"]:
            expected = (
                {"hard", "semihard"} if phase == "unweighted" else set(CALIBRATED)
            )
            for batch in record[phase]:
                require(
                    batch["native_norm"] > 1e-12
                    and set(batch["ratios"])
                    == set(batch["auxiliary_norms"])
                    == expected,
                    "Invalid calibration norms",
                )
                for a, norm in batch["auxiliary_norms"].items():
                    require(
                        norm > (1e-12 if phase == "unweighted" else 0),
                        "Invalid auxiliary norm",
                    )
                    close(
                        norm / batch["native_norm"],
                        batch["ratios"][a],
                        "Calibration ratio",
                    )
        require(
            [x["native_norm"] for x in record["unweighted"]]
            == [x["native_norm"] for x in record["verification"]],
            "Native gradient verification differs",
        )
        means = {
            a: fmean(x["ratios"][a] for x in record["unweighted"])
            for a in ["hard", "semihard"]
        }
        for a, v in means.items():
            close(v, record["mean_unweighted_ratios"][a], "Mean calibration ratio")
        coefficients = {a["name"]: a["coefficient"] for a in p["arms"][:3]}
        for a in p["arms"][3:]:
            name, rho = a["name"], a["target_ratio"]
            alpha = rho / means[a["construction"]]
            close(alpha, record["coefficients"][name], "Calibration coefficient")
            close(rho, record["targets"][name], "Calibration target")
            measured = fmean(x["ratios"][name] for x in record["verification"])
            error = abs(measured / rho - 1)
            close(
                measured, record["mean_verified_ratios"][name], "Verified mean pressure"
            )
            close(error, record["relative_errors"][name], "Calibration relative error")
            require(
                error
                <= p["calibration"][
                    "weighted_gradient_verification_relative_tolerance"
                ],
                "Calibration tolerance failed",
            )
            coefficients[name] = alpha
        result[w] = coefficients
    if full:
        batches = read(prefix + "calibration_batches.json")
        indices = [
            i
            for batch in read("calibration_plan.json")["seeds"][str(seed)]["batches"]
            for i in batch
        ]
        require(
            batches["source_indices"]
            == indices
            == [r["source_index"] for r in batches["records"]]
            and batches["caption_epoch"] == read(prefix + "start_000100.json")["epoch"]
            and batches["disjoint_from_report_probes_and_continuation_by_image_key"]
            is True,
            "Calibration batch records differ",
        )
        keys = {r["image_key"] for r in batches["records"]}
        others = {r["image_key"] for r in read(prefix + "data_roles.json")["report"]}
        for stream in p["stream_seeds"]:
            others.update(
                r["image_key"]
                for r in read(prefix + f"stream_{stream}/training_batches.json")[
                    "records"
                ]
            )
        require(len(keys) == 256 and not keys & others, "Calibration image overlap")
    return cal, result


def check_clipping(record, norm, maximum):
    require(norm > 0, "Invalid total gradient norm")
    scale = min(1.0, maximum / (norm + 1e-6))
    close(record["clipping_scale"], scale, "Clipping scale")
    require(record["clipping_activated"] == (scale < 1), "Clipping activation")


def diagnostics(files):
    """Descriptive summaries; no new fitting or independent replication."""
    read = lambda n: checked_json(files[n])
    p = read("protocol.json")
    rows = [read(n) for n in sorted(files) if n.endswith("/result.json")]
    nested = lambda f: fmean(
        fmean(f(r) for r in rows if r["training_seed"] == s)
        for s in p["training_seeds"]
    )
    signs = lambda vals: dict(
        negative=sum(v < 0 for v in vals),
        zero=sum(v == 0 for v in vals),
        positive=sum(v > 0 for v in vals),
    )
    pairs = []
    for row in rows:
        e = {
            w: {
                a: row["cells"][w]["points"][a]["50"]["report"]["native_loss"]
                - row["cells"][w]["points"]["native"]["50"]["report"]["native_loss"]
                for a in ARMS
            }
            for w in CELLS
        }
        q = {
            l: {w: e[w]["semihard_" + l] - e[w]["hard_" + l] for w in CELLS}
            for l in ["low", "high"]
        }
        pairs.append(
            dict(
                training_seed=row["training_seed"],
                stream_seed=row["stream_seed"],
                effects=e,
                Q=q,
                primary=fmean(q[l]["w500"] for l in q),
                pressure_interaction=q["high"]["w500"] - q["low"]["w500"],
                age_interactions={l: q[l]["w500"] - q[l]["w100"] for l in q},
            )
        )
    pressure = {
        w: {
            str(t): {
                a: {
                    m: nested(lambda r: r["cells"][w]["pressure_probes"][a][str(t)][m])
                    for m in [
                        "native_norm",
                        "weighted_aux_norm",
                        "ratio",
                        "cosine",
                        "total_norm_before_clip",
                        "clipping_scale",
                    ]
                }
                for a in ARMS[1:]
            }
            for t in p["pressure_probe_offsets"]
        }
        for w in CELLS
    }
    calibrations = {
        str(s): read(f"seed_{s}/calibration.json")["weights"]
        for s in p["training_seeds"]
    }
    selections = [
        t["selection"]
        for r in rows
        for cell in r["cells"].values()
        for traces in cell["traces"].values()
        for t in traces
    ]
    clips = {
        w: {
            a: dict(
                activated=sum(
                    t["clipping_activated"]
                    for r in rows
                    for t in r["cells"][w]["traces"][a]
                ),
                total=sum(len(r["cells"][w]["traces"][a]) for r in rows),
                mean_scale=nested(
                    lambda r: fmean(
                        t["clipping_scale"] for t in r["cells"][w]["traces"][a]
                    )
                ),
            )
            for a in ARMS
        }
        for w in CELLS
    }
    summary = read("comparison.json")
    return dict(
        source_run=SPEC[0],
        source_comparison_sha256=digest(files["comparison.json"])["sha256"],
        analysis="Post-result descriptive audit of prespecified records; no fitting or independent replication.",
        pairs=pairs,
        pressure=pressure,
        clipping=clips,
        calibration_max_relative_error=max(
            v
            for s in calibrations.values()
            for c in s.values()
            for v in c["relative_errors"].values()
        ),
        calibrated_coefficients={
            s: {w: c["coefficients"] for w, c in cs.items()}
            for s, cs in calibrations.items()
        },
        effect_signs={
            w: {a: signs([r["effects"][w][a] for r in pairs]) for a in ARMS}
            for w in CELLS
        },
        Q_signs={
            l: {w: signs([r["Q"][l][w] for r in pairs]) for w in CELLS}
            for l in ["low", "high"]
        },
        primary_signs=signs([r["primary"] for r in pairs]),
        pressure_interaction_signs=signs([r["pressure_interaction"] for r in pairs]),
        primary_seed_signs=signs(
            [
                s["timecourse"]["50"]["contrasts"]["late_geometry_main_effect"]["loss"]
                for s in summary["per_seed"]
            ]
        ),
        selection_trace_records=len(selections),
        overlap_total=sum(s["overlap_count"] for s in selections),
        diagonal_total=sum(s["diagonal_count"] for s in selections),
        note="Gradient norms/cosines and selection records cannot be reconstructed from endpoint features. "
        "Pressure summaries average per-probe ratios; only the calibration-set initial mean was controlled. "
        "Streams are nested in three reused seeds; historical branches are replay controls.",
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
        == [6, 12, 84, 84, 504, 1500, 4200, 5700],
        "Incomplete run",
    )
    require(
        done["pressure_probe_records"] == 336
        and done["pressure_matched"] is False
        and done["pressure_controls_coefficient"] is False
        and done["historical_b_exact_replay"] is True,
        "Invalid pressure/replay completion",
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
    p = read("source_overlay/configs/clip_pressure.json")
    ref = read("source_overlay/configs/clip_pressure_reference.json")
    native = read("source_overlay/configs/clip_optimizer_reset_reference.json")
    require(
        read("protocol.json") == p
        and read("native_replay_reference.json") == native
        and read("pressure_replay_reference.json") == ref,
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
    verify_frozen_inputs(files, p)
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
        calibration, coefficients = verify_calibration(files, seed, p, full=full)
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
            require(
                row["calibration_sha256"] == calibration["calibration_sha256"],
                "Row calibration differs",
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
                audit["actual_optimizer_steps"] == 700
                and audit["reports_control_treatment"] is False
                and audit["pressure_controls_coefficient"] is False
                and all(
                    audit[k] is True
                    for k in [
                        "historical_b_exact_replay",
                        "calibration_verified",
                        "coefficients_fixed",
                        "source_immutable",
                        "frozen_parameters_unchanged",
                        "matched_rng_and_lr_all_branches",
                        "fresh_adamw_all_branches",
                        "observations_preserve_state",
                        "candidate_supports_disjoint",
                    ]
                ),
                "Failed runtime audit",
            )
            require(set(row["cells"]) == set(CELLS), "Wrong cell grid")
            rngs, rates, weights, schedulers, initial_reports = None, {}, {}, {}, {}
            for key, cell in row["cells"].items():
                w, lr = cell["weight_step"], cell["lr_step"]
                require(
                    cell["calibration_sha256"]
                    == canonical_sha(calibration["weights"][str(w)]),
                    "Cell calibration differs",
                )
                require(
                    key == f"w{w}" and lr == 100 and set(cell["points"]) == set(ARMS),
                    "Wrong cell identity",
                )
                require(
                    all(cell["starts"][a] == cell["starts"]["native"] for a in ARMS),
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
                    coefficient = coefficients[str(w)][arm]
                    require(
                        cell["coefficients"][arm] == coefficient,
                        "Coefficient differs from calibration",
                    )
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
                    if arm in REPLAY:
                        expected = ref["pairs"][f"{seed}:{stream}"]["weights"][str(w)][
                            arm
                        ]
                        core = [
                            {k: trace[k] for k in CORE_TRACE_KEYS} for trace in traces
                        ]
                        require(
                            cell["points"][arm] == expected["points"]
                            and canonical_sha(core) == expected["traces_sha256"],
                            "Historical B replay differs",
                        )
                    require(
                        read(prefix + f"branch_{key}_{arm}.json")
                        == dict(
                            points=cell["points"][arm],
                            traces=traces,
                            pressure_probes=cell["pressure_probes"][arm],
                        ),
                        "Standalone branch differs",
                    )
                    for trace in traces:
                        check_clipping(
                            trace,
                            trace["gradient_norm_before_clip"],
                            read(folder + "training_config.json")["optimizer"][
                                "max_gradient_norm"
                            ],
                        )
                        selection = trace["selection"]
                        require(
                            selection["overlap_count"]
                            == selection["diagonal_count"]
                            == 0
                            and selection["support_gap"]["min"] >= 0,
                            "Invalid support separation record",
                        )
                        for stats in [
                            selection["support_gap"],
                            *selection["support_margins"].values(),
                            *selection["candidate_margins"].values(),
                        ]:
                            require(
                                stats["min"] <= stats["mean"] <= stats["max"]
                                and 0 <= stats["nonpositive_fraction"] <= 1,
                                "Invalid geometry statistics",
                            )
                    require(
                        traces[0]["selection"]
                        == cell["traces"]["native"][0]["selection"],
                        "Initial candidates differ",
                    )
                    probes = cell["pressure_probes"][arm]
                    require(
                        set(probes) == {str(t) for t in p["pressure_probe_offsets"]},
                        "Wrong pressure probe grid",
                    )
                    for t, probe in probes.items():
                        check_clipping(
                            probe,
                            probe["total_norm_before_clip"],
                            read(folder + "training_config.json")["optimizer"][
                                "max_gradient_norm"
                            ],
                        )
                        require(
                            read(prefix + f"pressure_{key}_{arm}_{int(t):03d}.json")
                            == probe,
                            "Standalone pressure differs",
                        )
                        require(
                            probe["native_norm"] > 0
                            and probe["weighted_aux_norm"] >= 0,
                            "Invalid gradient norm",
                        )
                        close(
                            probe["ratio"],
                            probe["weighted_aux_norm"] / probe["native_norm"],
                            "Pressure ratio",
                        )
                        require(
                            (
                                arm == "native"
                                and probe["cosine"] is None
                                and probe["weighted_aux_norm"] == 0
                            )
                            or (
                                arm != "native"
                                and -1.000001 <= probe["cosine"] <= 1.000001
                            ),
                            "Invalid gradient cosine",
                        )
                    close(
                        probes["0"]["native_norm"],
                        cell["pressure_probes"]["native"]["0"]["native_norm"],
                        "Initial native pressure",
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
    verify_summary(rows, read("comparison.json"), p)
    require(cache_count == (84 if full else 0), "Wrong cache count")
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
        files = {n: (target / n).read_bytes() for n in audit["files"]}
        validate(files, full=False)
        require(
            checked_json((target / "diagnostics.json").read_bytes())
            == diagnostics(files),
            "Derived diagnostics differ",
        )
        print(
            "Retained replay records and all seed/timecourse pressure arithmetic verified; caches omitted."
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
        scope="Every full-input manifest member verified; 84 cached losses/R1 recomputed; native "
        "and historical B replay records, matching/reset audits and pressure arithmetic checked. "
        "Caption-bearing inputs and NPZ caches omitted. Clone-only mode cannot recompute caches "
        "or reexecute training invariants. No proof of remote Drive flush or independent replication.",
    )
    for name, data in selected.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_json(target / "export_audit.json", audit)
    write_json(target / "diagnostics.json", diagnostics(files))
    verify_export(target, SPEC)
    print(
        f"{count} cached endpoints recomputed; replay records and pressure contrasts verified."
    )


if __name__ == "__main__":
    main()
