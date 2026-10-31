"""Verify a completed D1 download and recompute its frozen reporting contrasts.

Reads ZIPs without extracting model files. Writes only small evidence summaries.
This verifies recorded results; it does not rerun GPU training or reporting forwards.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from zipfile import ZipFile

from scripts.audit_clip_source_partial import digest_file
from src.clip_source_analysis import summarize
from src.clip_source_regime import ARMS


def audit(archive, previous, output):
    evidence = dict(archive=archive.name, archive_sha256=digest_file(archive))
    with ZipFile(archive) as z:
        root = z.namelist()[0].split("/")[0] + "/"
        names = set(z.namelist())
        read = lambda n: json.loads(z.read(root + n))

        def digest(name):
            h = hashlib.sha256()
            with z.open(root + name) as f:
                for block in iter(lambda: f.read(8 * 1024**2), b""):
                    h.update(block)
            return h.hexdigest()

        inventory = read("DRIVE_BACKUP_MANIFEST.json")["files"]
        for name, entry in inventory.items():
            assert digest(name) == entry["sha256"], name
        sources = read("source_manifest.json")
        for name, sha in sources.items():
            assert digest("source_overlay/" + name) == sha, name
        for name in ["src/clip_source_analysis.py", "src/clip_source_regime.py"]:
            assert (
                digest_file(Path(name)) == sources[name]
            ), "Local analysis differs from archived source"
        freeze = read("freeze_manifest.json")
        for name, sha in freeze["files"].items():
            assert digest("source_overlay/" + name) == sha, name
        manifest, completion = read("run_manifest.json"), read("completion.json")
        assert completion["status"] == "complete"
        assert completion["branches"] == 108 and completion["optimizer_steps"] == 6900
        for flag in [
            "native_trunk_exact_replay",
            "native_continuation_exact_replay",
            "initial_calibration_verified",
            "coefficients_fixed",
            "support_ids_fixed",
        ]:
            assert completion[flag] is True, flag
        pointers = [
            n.removeprefix(root)
            for n in names
            if n.startswith(root + "resume_state/seed_") and n.endswith(".json")
        ]
        for name in pointers:
            pointer = read(name)
            assert pointer["contract"] == manifest["contract"]
            assert len(pointer["generations"]) == 1
            entry = pointer["generations"][0]
            assert digest("resume_state/" + entry["file"]) == entry["sha256"]
        rows = [
            json.loads(z.read(n))
            for n in sorted(names)
            if n.startswith(root + "seed_") and n.endswith("/result.json")
        ]
        summary = summarize(rows, read("protocol.json"))
        assert summary == read("comparison.json"), "Recomputed comparison differs"
        probe_count = clipping_count = 0
        for row in rows:
            assert row["audit"]["native_exact_replay"] is True
            for cell in row["cells"].values():
                for arm in ARMS:
                    assert cell["starts"][arm] == cell["starts"]["native"]
                    clipping_count += sum(
                        t["clipping_activated"] for t in cell["traces"][arm]
                    )
                    for probe in cell["pressure_probes"][arm].values():
                        state = probe["state_preservation"]
                        assert (
                            state["preserved"]
                            and state["before_sha256"] == state["after_sha256"]
                        )
                        probe_count += 1
        assert probe_count == completion["pressure_probe_records"] == 432
        errors, ratios = [], []
        for seed in read("protocol.json")["training_seeds"]:
            for cal in read(f"seed_{seed}/calibration.json")["weights"].values():
                assert cal["training_state_preserved"] and cal["optimizer_updates"] == 0
                assert cal["state_before_sha256"] == cal["state_after_sha256"]
                errors.extend(cal["relative_errors"].values())
                ratios.extend(cal["verified_mean_ratios"].values())
        unchanged = 0
        with ZipFile(previous) as old:
            for name in old.namelist():
                if (
                    name.startswith(root + "seed_")
                    and "/branch_w" in name
                    and name.endswith(".json")
                ):
                    assert old.read(name) == z.read(name), (
                        "Completed branch changed: " + name
                    )
                    unchanged += 1
        pressure = {
            w: {
                arm: {
                    t: {
                        metric: fmean(
                            r["cells"][w]["pressure_probes"][arm][t][metric]
                            for r in rows
                        )
                        for metric in ["ratio", "r_eff", "cosine"]
                    }
                    for t in ["0", "49"]
                }
                for arm in ARMS[1:]
            }
            for w in ["w100", "w500"]
        }
        evidence.update(
            verification=dict(
                inventory_files=len(inventory),
                source_files=len(sources),
                checkpoint_files=len(pointers),
                hashes_match=True,
                comparison_recomputed_exactly=True,
                previous_archive=previous.name,
                unchanged_completed_branches=unchanged,
                probe_preservation_checks=probe_count,
            ),
            completion=completion,
            manifest=manifest,
            calibration=dict(
                cells=len(errors),
                max_relative_error=max(errors),
                verified_ratio_min=min(ratios),
                verified_ratio_max=max(ratios),
            ),
            clipping=dict(active=clipping_count, total=5400),
            pressure=pressure,
        )
        output.mkdir(parents=True, exist_ok=True)
        for filename, value in [("audit.json", evidence), ("comparison.json", summary)]:
            (output / filename).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
    print(
        json.dumps(
            dict(
                verification=evidence["verification"],
                primary=summary["primary_value"],
                complete=completion,
                calibration=evidence["calibration"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.previous, args.output)
