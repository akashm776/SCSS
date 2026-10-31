"""Read-only verification and descriptive analysis of the incomplete D1 archive.

Never emits the frozen full-grid primary estimate or selects new treatments.
Outputs contain summaries only, without tensors, images or captions.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import fmean
from zipfile import ZipFile

from src.clip_source_analysis import effects, contrasts, summarize
from src.clip_source_regime import ARMS


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def average(entries):
    return {
        w: {
            a: {m: fmean(e[w][a][m] for e in entries) for m in ["loss", "r1_pp"]}
            for a in ARMS
        }
        for w in ["w100", "w500"]
    }


def audit(archive, logs, previous):
    result = dict(
        status="partial_development_evidence_not_final_D1",
        archive=archive.name,
        archive_sha256=digest_file(archive),
        logs_archive=logs.name,
        logs_sha256=digest_file(logs),
    )
    with ZipFile(archive) as z:
        root = z.namelist()[0].split("/")[0] + "/"
        names = set(z.namelist())
        read = lambda name: json.loads(z.read(root + name))
        inventory = read("DRIVE_BACKUP_MANIFEST.json")["files"]
        mismatches = []
        for name, entry in inventory.items():
            if (
                root + name not in names
                or hashlib.sha256(z.read(root + name)).hexdigest() != entry["sha256"]
            ):
                mismatches.append(name)
        if mismatches:
            raise ValueError("Archive inventory mismatch: " + repr(mismatches))
        source_manifest = read("source_manifest.json")
        for name, sha in source_manifest.items():
            assert (
                hashlib.sha256(z.read(root + "source_overlay/" + name)).hexdigest()
                == sha
            )
        contract = read("run_manifest.json")["contract"]
        checkpoints = []
        pointer_paths = sorted(
            n
            for n in names
            if n.startswith(root + "resume_state/seed_") and n.endswith(".json")
        )
        for name in pointer_paths:
            pointer = json.loads(z.read(name))
            assert pointer["contract"] == contract and len(pointer["generations"]) == 1
            entry = pointer["generations"][0]
            h = hashlib.sha256()
            with z.open(root + "resume_state/" + entry["file"]) as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    h.update(block)
            assert h.hexdigest() == entry["sha256"]
            checkpoints.append(entry)
        result["verification"] = dict(
            inventory_files=len(inventory),
            source_files=len(source_manifest),
            checkpoint_files=len(checkpoints),
            all_hashes_match=True,
        )
        result["manifest"] = {
            k: read("run_manifest.json").get(k)
            for k in [
                "status",
                "error",
                "resume_blocked",
                "packages",
                "python",
                "bundle_id",
            ]
        }
        result["final_completion_present"] = root + "completion.json" in names
        result["final_comparison_present"] = root + "comparison.json" in names
        p = read("protocol.json")
        rows = [
            json.loads(z.read(n))
            for n in sorted(names)
            if n.startswith(root + "seed_") and n.endswith("/result.json")
        ]
        try:
            summarize(rows, p)
        except ValueError as error:
            assert str(error) == "Incomplete/duplicate D1 grid; no survivor averaging"
            result["full_grid_analysis"] = str(error)
        else:
            raise ValueError(
                "Expected partial archive; use full analysis for complete data"
            )
        per_pair = []
        all_clipping = []
        pressure = {}
        for row in rows:
            for flag, value in row["audit"].items():
                if flag == "actual_optimizer_steps":
                    assert value == 900
                elif flag == "reports_control_treatment":
                    assert value is False
                else:
                    assert value is True
            for w, cell in row["cells"].items():
                assert set(cell["points"]) == set(ARMS)
                for arm in ARMS:
                    assert len(cell["traces"][arm]) == 50
                    assert set(cell["points"][arm]) == set(
                        map(str, p["observation_steps"])
                    )
                    assert set(cell["pressure_probes"][arm]) == set(
                        map(str, p["pressure_probe_offsets"])
                    )
                    assert cell["starts"][arm] == cell["starts"]["native"]
                    for trace in cell["traces"][arm]:
                        assert trace["coefficient"] == cell["coefficients"][arm]
                        all_clipping.append(trace["clipping_activated"])
                    for probe in cell["pressure_probes"][arm].values():
                        assert probe["state_preservation"]["preserved"] is True
                        assert (
                            probe["state_preservation"]["before_sha256"]
                            == probe["state_preservation"]["after_sha256"]
                        )
                for t in range(50):
                    assert (
                        len(
                            {
                                row["cells"][state]["traces"][a][t]["geometry"][
                                    "frozen_exposure_sha256"
                                ]
                                for state in row["cells"]
                                for a in ARMS
                            }
                        )
                        == 1
                    )
            times = {}
            for t in p["observation_steps"]:
                e = {w: effects(cell, t) for w, cell in row["cells"].items()}
                times[str(t)] = dict(effects=e, contrasts=contrasts(e, [100, 500]))
            per_pair.append(
                dict(
                    seed=row["training_seed"],
                    stream=row["stream_seed"],
                    timecourse=times,
                )
            )
        result["completed_pairs"] = per_pair
        result["completed_branches"] = 18 * len(rows)
        counts = Counter(row["training_seed"] for row in rows)
        complete_seeds = sorted(s for s, n in counts.items() if n == 2)
        result["complete_seeds"] = complete_seeds
        balanced_times = {}
        for t in map(str, p["observation_steps"]):
            seed_means = [
                average(
                    [r["timecourse"][t]["effects"] for r in per_pair if r["seed"] == s]
                )
                for s in complete_seeds
            ]
            e = average(seed_means)
            balanced_times[t] = dict(effects=e, contrasts=contrasts(e, [100, 500]))
        result["two_complete_seeds_descriptive_only"] = balanced_times
        result["seed_H50"] = {}
        for s, n in sorted(counts.items()):
            subset = [r for r in rows if r["training_seed"] == s]
            assert len({r["calibration_sha256"] for r in subset}) == 1
            for w in subset[0]["cells"]:
                assert all(
                    r["cells"][w]["coefficients"]
                    == subset[0]["cells"][w]["coefficients"]
                    for r in subset
                )
            e = average(
                [r["timecourse"]["50"]["effects"] for r in per_pair if r["seed"] == s]
            )
            result["seed_H50"][s] = dict(
                n_streams=n, effects=e, contrasts=contrasts(e, [100, 500])
            )
        result["clipping"] = dict(active=sum(all_clipping), total=len(all_clipping))
        for w in ["w100", "w500"]:
            pressure[w] = {}
            for arm in ARMS[1:]:
                pressure[w][arm] = {}
                for t in ["0", "49"]:
                    probes = [r["cells"][w]["pressure_probes"][arm][t] for r in rows]
                    pressure[w][arm][t] = {
                        k: fmean(x[k] for x in probes)
                        for k in ["ratio", "r_eff", "cosine"]
                    }
        result["pressure_five_completed_pairs_descriptive"] = pressure
        errors = []
        verified_ratios = []
        for seed in p["training_seeds"]:
            cal = read(f"seed_{seed}/calibration.json")
            for weight in cal["weights"].values():
                assert (
                    weight["training_state_preserved"]
                    and weight["optimizer_updates"] == 0
                )
                assert weight["state_before_sha256"] == weight["state_after_sha256"]
                errors.extend(weight["relative_errors"].values())
                verified_ratios.extend(weight["verified_mean_ratios"].values())
        result["calibration"] = dict(
            max_relative_error=max(errors),
            minimum_verified_ratio=min(verified_ratios),
            maximum_verified_ratio=max(verified_ratios),
            cells=len(errors),
        )
        # Restricted tensor deserialization only after verifying the saved SHA.
        from src.clip_source_resume_store import safe_load
        import tempfile

        active_entry = next(
            e
            for e in checkpoints
            if e["file"].startswith("seed_141421_stream_2026100102_")
        )
        with tempfile.TemporaryDirectory(prefix="d1-partial-audit-") as scratch:
            path = Path(scratch) / "active.pt"
            path.write_bytes(z.read(root + "resume_state/" + active_entry["file"]))
            saved = safe_load(path)
        assert saved["contract"] == contract
        active = saved["payload"]["active"]
        branch = saved["payload"]["branches"][active["branch"]]
        result["active_checkpoint"] = dict(
            file=active_entry["file"],
            branch=active["branch"],
            committed_updates=len(branch["traces"]),
            status=branch["status"],
        )
        result["remaining_optimizer_updates"] = (
            5400 - 50 * result["completed_branches"] - len(branch["traces"])
        )
        if previous:
            with ZipFile(previous) as old:
                old_branches = [
                    n
                    for n in old.namelist()
                    if n.startswith(root + "seed_")
                    and "/branch_w" in n
                    and n.endswith(".json")
                ]
                assert all(old.read(n) == z.read(n) for n in old_branches)
                result["older_archive_comparison"] = dict(
                    archive=previous.name,
                    unchanged_completed_branches=len(old_branches),
                )
    with ZipFile(logs) as z:
        result["logs"] = {}
        for name in sorted(z.namelist()):
            if name.endswith(".log"):
                lines = z.read(name).decode().splitlines()
                result["logs"][name] = [
                    line
                    for line in lines
                    if line.startswith(
                        (
                            "RuntimeError:",
                            "FileNotFoundError:",
                            "huggingface_hub.errors.",
                            "CHILD_EXIT_CODE:",
                            "D1_STORAGE_CLEANUP:",
                        )
                    )
                ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = audit(args.archive, args.logs, args.previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in [
                    "verification",
                    "completed_branches",
                    "active_checkpoint",
                    "remaining_optimizer_updates",
                    "calibration",
                    "clipping",
                    "older_archive_comparison",
                ]
            },
            indent=2,
        )
    )
