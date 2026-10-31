"""Read-only verification and filtered public export of four completed runs.

Full-input mode checks every original manifest member, recomputes endpoint losses
and R@1, and validates the paired-history/reset contrasts. Clone-only mode checks
the retained bytes against pinned original manifests; omitted caches cannot be
recomputed from the public export. Never launches training or loads checkpoints.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.archive_clip_followup_evidence import (
    DATA_FILES,
    EXPORT_SUFFIXES,
    digest,
    safe_path,
)
from scripts.analyze_clip_policy_robustness import (
    checked_json,
    close,
    require,
    verify_input,
)
from scripts.analyze_clip_image_influence import unpack, query_losses
from scripts import analyze_clip_paired_history as history

STUDIES = {
    "clip_policy_pool_test_2026-09": (
        "clip_policy_pool_test_20260930T120952_487061Z",
        "880459ebeef76ab14521e76a2808f48f66faa20d4392aff6348d9c39de14c9b2",
        "52ad1d1e45ae8cd8d39f623d028c890ed4ef73e57ad4f4301c1a5b929092bead",
    ),
    "clip_native_history_policy_2026-10": (
        "clip_native_history_policy_20261001T234400_383003Z",
        "7ff9f85f2329e3ac4443ec08731d9f4b1fc6d7030ab1fecbcaa588a4f687ecc5",
        "db7f1a789e724f68b0a14a550ba12dd86f80b4458b1e14e9218646ee6e372914",
    ),
    "clip_fresh_split_policy_2026-10": (
        "clip_fresh_split_policy_20261002T024902_873109Z",
        "ed2b96f08ca32c9ad1828b16d3459e60784c35b07643e55317ad60a37b8ecc08",
        "f84d5fd0dd5ec4ba3c47758bdf4e08ed43b31d92dbcc1d8c8ff3efb71a981a81",
    ),
    "clip_optimizer_reset_2026-10": (
        "clip_optimizer_reset_20261002T043855_470158Z",
        "e4911d0871e74ef23c592fd40d8de69604251dbd5992adf2b644f65408c53f31",
        "03fb0a137afdcb615d8cdd9ae2c35b7e799acc85706e8cf04eada81fad9eacfc",
    ),
}


def contains_caption(value):
    if isinstance(value, dict):
        return any(
            str(k).lower() in {"caption", "captions", "text", "texts"}
            or contains_caption(v)
            for k, v in value.items()
        )
    return isinstance(value, list) and any(contains_caption(v) for v in value)


def retain(name, data):
    p = safe_path(name)
    if p.name in DATA_FILES or p.suffix not in EXPORT_SUFFIXES:
        return False
    if p.suffix == ".json" and contains_caption(checked_json(data)):
        return False
    if p.suffix == ".jsonl" and any(
        contains_caption(checked_json(line))
        for line in data.splitlines()
        if line.strip()
    ):
        return False
    return True


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def locate(downloads, run):
    path = downloads / run
    if path.is_dir():
        return path
    matches = sorted(downloads.glob(run + "*.zip"))
    require(len(matches) == 1, "Need exactly one input ZIP or extracted run: " + run)
    return matches[0]


def validate_caches(files):
    count = 0
    read = lambda name: checked_json(files[name])
    for name in sorted(files):
        if not name.endswith("/result.json"):
            continue
        row = read(name)
        prefix = name.removesuffix("result.json")
        pool = "endpoints" in row
        arms = row["endpoints"] if pool else row["points"]
        for arm in arms:
            meta = read(prefix + f"features_{arm}.json")
            row_hash = (
                read("report_row_order.json")["row_order_sha256"]
                if pool
                else row["report_row_order_sha256"]
            )
            logits = unpack(files[prefix + f"features_{arm}.npz"], meta, row_hash)
            report = (
                row["endpoints"][arm] if pool else row["points"][arm]["50"]["report"]
            )
            require(
                meta["feature_sha256"] == report["features_sha256"], "Feature mismatch"
            )
            close(
                float(query_losses(logits)[:, 2].mean()),
                report["full_pool_loss" if pool else "native_loss"],
                "Endpoint loss",
            )
            truth = np.arange(512)
            t, i = [
                100 * float(np.mean(logits.argmax(axis) == truth)) for axis in [1, 0]
            ]
            retrieval = report["pool_retrieval"]
            for key, value in [
                ("text_to_image_r1_percent", t),
                ("image_to_text_r1_percent", i),
                ("mean_r1_percent", (t + i) / 2),
            ]:
                close(value, retrieval[key], "Endpoint R1")
            count += 1
    require(count == 48, "Incomplete endpoint cache grid")
    return count


def validate_reset(files):
    from src import clip_optimizer_reset as reset

    read = lambda name: checked_json(files[name])
    p, reference = reset.load_protocol(), reset.load_reference()
    require(
        read("protocol.json") == p and read("replay_reference.json") == reference,
        "Changed reset protocol/reference",
    )
    rows = []
    for seed in p["training_seeds"]:
        trunk = read(f"seed_{seed}/trunk_audit.json")
        require(
            reset.digest(trunk) == reference["seeds"][str(seed)]["trunk_audit.json"],
            "Native trunk replay differs",
        )
        for step in p["checkpoint_steps"]:
            for stream in p["stream_seeds"]:
                prefix = f"seed_{seed}/step_{step:06d}/stream_{stream}/"
                row = read(prefix + "result.json")
                require(
                    (row["training_seed"], row["checkpoint_step"], row["stream_seed"])
                    == (seed, step, stream),
                    "Reset row identity differs",
                )
                ref = reference["pairs"][f"{seed}:{step}:{stream}"]
                require(
                    reset.digest(read(prefix + "training_batches.json"))
                    == ref["training_batches_sha256"],
                    "Continuation data differs",
                )
                for arm in reset.ARMS:
                    traces = row["traces"][arm]
                    require(len(traces) == 50, "Incomplete update trace")
                    for trace, base in zip(traces, row["traces"]["retained_native"]):
                        require(
                            all(
                                trace[k] == base[k]
                                for k in ["learning_rates", "rng_after_sha256"]
                            ),
                            "Unmatched LR/RNG",
                        )
                        close(
                            trace["coefficient"],
                            p["coefficient"] if arm.endswith("sustained") else 0.0,
                            "Treatment coefficient",
                        )
                    for k in [
                        "weights_sha256",
                        "scheduler_sha256",
                        "rng_sha256",
                        "param_groups_sha256",
                    ]:
                        require(
                            row["starts"][arm][k]
                            == row["starts"]["retained_native"][k],
                            "Unmatched start",
                        )
                    for offset in p["observation_steps"]:
                        require(
                            read(prefix + f"point_{arm}_{offset:03d}.json")
                            == row["points"][arm][str(offset)],
                            "Standalone point differs",
                        )
                    if arm.startswith("retained_"):
                        expected = ref["arms"][arm.split("_", 1)[1]]
                        require(
                            reset.digest(traces) == expected["traces_sha256"],
                            "Retained trace differs",
                        )
                        require(
                            all(
                                row["points"][arm][t] == expected["points"][t]
                                for t in ["0", "50"]
                            ),
                            "Retained endpoint differs",
                        )
                    else:
                        audit = row["resets"][arm]
                        require(
                            audit["cleared_parameter_states"] == 35
                            and audit["new_parameter_states"] == 0
                            and audit[
                                "weights_scheduler_groups_rng_gradients_unchanged"
                            ]
                            and audit["old_step_counts"] == [step],
                            "Invalid reset audit",
                        )
                rows.append(row)
    require(
        read("comparison.json") == reset.summarize(rows, p), "Reset summary differs"
    )
    require(
        len([n for n in files if n.endswith("/result.json")]) == 12,
        "Unexpected result grid",
    )


def export_one(downloads, output, name, spec):
    run, sha, bundle = spec
    target = output / name
    require(not target.exists(), "Refusing to overwrite " + str(target))
    files, provenance = verify_input(
        locate(downloads, run), dict(source_run=run, backup_manifest_sha256=sha)
    )
    manifest = checked_json(files["run_manifest.json"])
    done = checked_json(files["completion.json"])
    require(
        manifest["bundle_id"] == bundle and done["status"] == "complete",
        "Wrong bundle/incomplete run",
    )
    require(done["pairs"] == 12 and done["branches"] == 48, "Incomplete run grid")
    for path, expected in checked_json(files["source_manifest.json"]).items():
        require(
            digest(files["source_overlay/" + path])["sha256"] == expected,
            "Source snapshot mismatch",
        )
    caches = validate_caches(files)
    if name == "clip_optimizer_reset_2026-10":
        validate_reset(files)
    selected = {n: b for n, b in files.items() if retain(n, b)}
    audit = dict(
        **provenance,
        bundle_id=bundle,
        status="small_evidence_verified",
        experiment_complete=True,
        endpoint_caches_recomputed=caches,
        omitted=sorted(set(files) - set(selected)),
        files={n: digest(b) for n, b in selected.items()},
        scope="All full-input manifest members SHA256/size verified before filtering; 48 endpoint losses/R1 "
        "recomputed. Filtered export excludes captions, tensors and NPZ caches. Clone-only verification "
        "cannot recompute omitted caches. Manifest is provenance, not a signature or proof of remote Drive flush.",
    )
    target.mkdir(parents=True)
    for path, data in selected.items():
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    write_json(target / "export_audit.json", audit)
    verify_export(target, spec)


def verify_export(target, spec):
    run, sha, bundle = spec
    audit = checked_json((target / "export_audit.json").read_bytes())
    marker = (target / "DRIVE_BACKUP_MANIFEST.json").read_bytes()
    require(digest(marker)["sha256"] == sha, "Original manifest pin differs")
    original = checked_json(marker)
    require(
        original["run_id"] == audit["run_id"] == run
        and audit["bundle_id"] == bundle
        and original["status"] == "verified_tree",
        "Export identity/status differs",
    )
    require(
        set(audit["files"]) | set(audit["omitted"])
        == set(original["files"]) | {"DRIVE_BACKUP_MANIFEST.json"}
        and not set(audit["files"]) & set(audit["omitted"]),
        "Export inventory differs",
    )
    for name, record in audit["files"].items():
        path = target / str(safe_path(name))
        require(not path.is_symlink(), "Symlink public evidence")
        data = path.read_bytes()
        require(digest(data) == record, "Changed public evidence: " + name)
        require(retain(name, data), "Unsafe public evidence: " + name)
        if name != "DRIVE_BACKUP_MANIFEST.json":
            require(
                record == {k: original["files"][name][k] for k in ["bytes", "sha256"]},
                "Export differs from original manifest",
            )
    require(
        checked_json((target / "completion.json").read_bytes())["status"] == "complete",
        "Incomplete export",
    )
    print(
        f'{target.name}: {len(audit["files"])} retained files verified; {len(audit["omitted"])} omitted'
    )


def archive_plan(downloads, output):
    source = downloads / "clip_paired_history_comparison_v1_plan"
    target = output / "clip_paired_history_comparison_2026-10/frozen_plan"
    require(not target.exists(), "Refusing to overwrite frozen plan")
    manifest = checked_json((source / "plan_manifest.json").read_bytes())
    spec = history.load_spec()
    require(
        manifest["bundle_id"]
        == "2395e02672eccdb64a29e9b8d9cc0271972b513dab6435aa382541ebebf0ac50"
        and manifest["frozen_at_utc"] == "2026-10-02T02:44:04.646220+00:00"
        and len(manifest["files"]) == 7,
        "Wrong frozen plan",
    )
    for name, sha in manifest["files"].items():
        safe_path(name)
        data = (source / "source_overlay" / name).read_bytes()
        require(
            digest(data)["sha256"]
            == sha
            == digest((history.ROOT / name).read_bytes())["sha256"],
            "Frozen plan/source differs",
        )
    require(
        checked_json(
            (
                source / "source_overlay/configs/clip_paired_history_comparison.json"
            ).read_bytes()
        )
        == spec,
        "Frozen comparison specification differs",
    )
    target.mkdir(parents=True)
    names = ["plan_manifest.json"] + ["source_overlay/" + n for n in manifest["files"]]
    for name in names:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source / name).read_bytes())


def verify_derived(output):
    target = output / "clip_paired_history_comparison_2026-10"
    spec = history.load_spec()
    require(
        checked_json((target / "protocol.json").read_bytes()) == spec,
        "Derived protocol differs",
    )
    grids = []
    for name in [
        "clip_native_history_policy_2026-10",
        "clip_fresh_split_policy_2026-10",
    ]:
        grid = {}
        for path in (output / name).glob("seed_*/step_*/stream_*/result.json"):
            row = checked_json(path.read_bytes())
            key = row["training_seed"], row["checkpoint_step"], row["stream_seed"]
            require(key not in grid, "Duplicate public pair")
            grid[key] = row
        grids.append(grid)
    actual = checked_json((target / "comparison.json").read_bytes())
    for key, value in history.compare(*grids, spec).items():
        require(actual[key] == value, "Derived comparison differs: " + key)
    plan = target / "frozen_plan"
    manifest = checked_json((plan / "plan_manifest.json").read_bytes())
    require(
        manifest["frozen_at_utc"] == "2026-10-02T02:44:04.646220+00:00"
        and manifest["bundle_id"]
        == "2395e02672eccdb64a29e9b8d9cc0271972b513dab6435aa382541ebebf0ac50",
        "Frozen plan identity differs",
    )
    for name, sha in manifest["files"].items():
        safe_path(name)
        require(
            digest((plan / "source_overlay" / name).read_bytes())["sha256"]
            == sha
            == digest((history.ROOT / name).read_bytes())["sha256"],
            "Frozen source differs",
        )
    print(
        "Paired-history comparison recomputed from public rows; frozen plan verified."
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--downloads", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("experiment_results"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.downloads is None:
        parser.error("--downloads required for full-input verification/export")
    if not args.verify_only:
        spec = history.load_spec()
        native = history.read_run(
            locate(args.downloads, STUDIES["clip_native_history_policy_2026-10"][0]),
            "native",
            spec,
        )
        gated = history.read_run(
            locate(args.downloads, STUDIES["clip_fresh_split_policy_2026-10"][0]),
            "gated_history",
            spec,
        )
        history.verify_matching(native, gated, spec)
    for name, spec in STUDIES.items():
        if not args.verify_only:
            export_one(args.downloads, args.output_root, name, spec)
        verify_export(args.output_root / name, spec)
    if not args.verify_only:
        history.main(
            locate(args.downloads, STUDIES["clip_native_history_policy_2026-10"][0]),
            locate(args.downloads, STUDIES["clip_fresh_split_policy_2026-10"][0]),
            args.output_root / "clip_paired_history_comparison_2026-10",
        )
        archive_plan(args.downloads, args.output_root)
    verify_derived(args.output_root)


if __name__ == "__main__":
    main()
