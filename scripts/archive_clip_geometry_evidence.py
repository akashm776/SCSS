"""Verify completed D0 artifacts and retain compact evidence; no GPU or training.

Recomputes selected scientific summaries from every query record. Does not rerun
embeddings or claim to independently recompute every metric in summary.json.
"""

import argparse
import ast
import base64
from collections import defaultdict, Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = "clip_geometry_recovery_20261009T011659_954111Z"
BUNDLE = "92f3431d62b1bb35e9a5740e3da28c92f38a2c1ff3e04dcdde2bbf16e964c254"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(run, out):
    run = Path(run)
    out = Path(out)
    manifest = read(run / "run_manifest.json")
    inventory = read(run / "DRIVE_BACKUP_MANIFEST.json")
    require(
        manifest["run_id"] == RUN and manifest["bundle_id"] == BUNDLE,
        "Unknown completed D0 run/bundle",
    )
    require(
        manifest["status"] == "complete_pending_drive_flush"
        and inventory["status"] == "verified_tree",
        "Incomplete run/tree",
    )
    for name, entry in inventory["files"].items():
        rel = Path(name)
        require(
            not rel.is_absolute() and ".." not in rel.parts, "Unsafe inventory path"
        )
        f = run / rel
        require(
            f.is_file()
            and f.stat().st_size == entry["bytes"]
            and sha(f) == entry["sha256"],
            "Inventory mismatch: " + name,
        )
    require(
        not any(Path(n).name.startswith("failure") for n in inventory["files"]),
        "Saved failure record",
    )
    vals = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(
            (ROOT / "colabs/clip_geometry_recovery_drive_one_cell.py").read_text()
        ).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ["PAYLOAD", "BUNDLE_SHA256"]
    }
    raw = zlib.decompress(base64.b64decode(vals["PAYLOAD"]))
    require(
        hashlib.sha256(raw).hexdigest() == BUNDLE == vals["BUNDLE_SHA256"],
        "Issued recovery bundle differs",
    )
    bundle = json.loads(raw)
    sources = read(run / "source_manifest.json")
    require(
        sources == json.loads(bundle["geometry_recovery_source_manifest.json"]),
        "Source inventory differs",
    )
    for name, h in sources.items():
        require(
            sha(run / "source_overlay" / name) == h
            and hashlib.sha256(bundle[name].encode()).hexdigest() == h,
            "Source mismatch: " + name,
        )
    p = read(run / "protocol.json")
    plan = read(run / "plan.json")
    done = read(run / "completion.json")
    rec = read(run / "recovery_completion.json")
    for name, frozen in [
        ("protocol.json", "configs/clip_geometry_audit.json"),
        ("plan.json", "configs/clip_geometry_audit_plan.json"),
        ("freeze_manifest.json", "configs/clip_geometry_audit_freeze_manifest.json"),
    ]:
        require(
            (run / name).read_bytes() == (ROOT / frozen).read_bytes(),
            "Frozen input differs",
        )
    require(
        done["status"] == "complete"
        and done["batch_forwards"] == 624
        and done["query_exposures"] == 39936
        and all(
            done[k] == 0
            for k in [
                "audit_optimizer_updates",
                "audit_backward_passes",
                "audit_scheduler_steps",
                "virtual_optimizer_steps",
                "reporting_forward_passes",
            ]
        ),
        "Completion budget differs",
    )
    require(
        (
            rec["reused_batch_forwards"],
            rec["new_batch_forwards"],
            rec["total_batch_forwards"],
        )
        == (166, 458, 624),
        "Recovery budget differs",
    )
    for seed in p["training_seeds"]:
        ref = plan["seeds"][str(seed)]
        folder = run / f"seed_{seed}"
        require(
            canonical(read(folder / "trunk_audit.json"))
            == ref["native_trunk_audit_canonical_sha256"],
            "Native trunk differs",
        )
        for w in p["checkpoint_steps"]:
            require(
                read(folder / f"start_{w:06d}.json")
                == ref["states"][str(w)]["archived_start"],
                "Native start differs",
            )
    prior = read(run / "recovery_provenance/source_0/DRIVE_BACKUP_MANIFEST.json")[
        "files"
    ]
    summary = read(run / "summary.json")
    reported = {
        (
            g["population"],
            int(g["weight"]) if g["weight"] != "paired" else "paired",
            g["seed"],
            g["stream"],
        ): g
        for g in summary["groups"]
    }
    values = defaultdict(lambda: defaultdict(list))
    states = {}
    counts = Counter()
    origins = Counter()
    flags = Counter()
    seen = set()
    expected = {
        (seed, pop, u, off, w)
        for seed in p["training_seeds"]
        for pop, streams, offsets in [
            (
                "continuation",
                p["stream_seeds"],
                p["populations"]["extended"]["offsets"],
            ),
            (
                "calibration_feasibility",
                [None],
                p["populations"]["calibration_feasibility"]["batch_offsets"],
            ),
        ]
        for u in streams
        for off in offsets
        for w in p["checkpoint_steps"]
    }

    def add(pop, w, seed, stream, metric, value):
        if value is None:
            return
        vals = value if isinstance(value, list) else [value]
        values[(pop, w, seed, stream)][metric].extend(vals)

    for shard in read(run / "shards.json"):
        key = tuple(
            shard[n]
            for n in [
                "training_seed",
                "population",
                "stream_seed",
                "batch_offset",
                "weight_step",
            ]
        )
        require(key in expected and key not in seen, "Unexpected/duplicate batch")
        seen.add(key)
        seed, pop, u, off, w = key
        path = run / shard["records"]
        require(sha(path) == shard["sha256"], "Shard checksum mismatch")
        stem = path.name[: -len(".jsonl.gz")]
        audit = read(path.with_name(stem + "_audit.json"))
        origin = read(path.with_name(stem + "_origin.json"))
        require(
            audit["preserved"]
            and audit["before_sha256"] == audit["after_sha256"]
            and audit["optimizer_updates"] == audit["backward_passes"] == 0,
            "Batch state audit failed",
        )
        require(
            audit["selection_dtype"] == "torch.float32"
            and not audit["scoring_backend"]["matmul_tf32"],
            "Selection precision differs",
        )
        require(audit["records"]["sha256"] == shard["sha256"], "Audit shard mismatch")
        origins["reused" if origin["reused"] else "new"] += 1
        if origin["reused"]:
            require(
                prior[shard["records"]]["sha256"] == shard["sha256"],
                "Reused observation differs from original",
            )
        indices = (
            plan["continuation_batches"][f"{seed}:{u}"][off]
            if pop == "continuation"
            else plan["calibration_batches"][str(seed)][off]
        )
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f]
        require(len(rows) == 64, "Incomplete query batch")
        for i, r in enumerate(rows):
            require(
                tuple(
                    r[n]
                    for n in [
                        "training_seed",
                        "population",
                        "stream_seed",
                        "batch_offset",
                        "weight_step",
                    ]
                )
                == key
                and r["query_index"] == i
                and r["source_index"] == indices[i]
                and r["candidate_source_indices"]
                == [v for j, v in enumerate(indices) if j != i],
                "Query identities differ",
            )
            require(
                r["N_positive"] + r["N_nonpositive"] == 63
                and r["N_negative"] + r["N_zero"] == r["N_nonpositive"],
                "Sign counts differ",
            )
            qkey = (seed, pop, u, off, i)
            states.setdefault(qkey, {})[w] = (
                r["source_index"],
                r["caption_index"],
                [r["sizes"][str(k)]["eligible"] for k in range(1, 9)],
            )
            scopes = (
                ["calibration_feasibility"]
                if pop == "calibration_feasibility"
                else ["extended"] + (["primary"] if off == 0 else [])
            )
            metrics = {
                "zero_nonpositive": r["N_nonpositive"] == 0,
                "positive_cosine": r["positive_cosine"],
                "N_positive": r["N_positive"],
                "N_nonpositive": r["N_nonpositive"],
            }
            margins = dict(zip(r["candidate_indices"], r["margins"]))
            for sign in ["positive", "nonpositive"]:
                metrics["semantics/" + sign] = [
                    s == r["species_id"]
                    for s, m in zip(r["candidate_species_ids"], r["margins"])
                    if (m > 0) == (sign == "positive")
                ]
            for k in range(1, 9):
                entry = r["sizes"][str(k)]
                metrics[f"k{k}/positive_feasible"] = r["N_positive"] >= k
                metrics[f"k{k}/nonpositive_feasible"] = r["N_nonpositive"] >= k
                metrics[f"k{k}/eligible"] = entry["eligible"]
                for regime, s in entry["regimes"].items():
                    if not s["feasible"]:
                        require(
                            s["support"] is None and s["candidate"] is None,
                            "Missing support imputed",
                        )
                        flags["missing_supports"] += 1
                        continue
                    support = s["support"]
                    require(
                        len(set(support)) == k and i not in support,
                        "Invalid support membership",
                    )
                    require(
                        s["contributor_margins"] == [margins[j] for j in support],
                        "Contributor margin differs",
                    )
                    if regime in ["C", "I"]:
                        require(
                            all(
                                (v > 0) == (regime == "C")
                                for v in s["contributor_margins"]
                            ),
                            "Wrong source margin sign",
                        )
                    c = s["candidate"]
                    for path_name in ["ideal", "mixed"]:
                        flags["invalid_mixtures"] += not c[path_name]["valid"]
                    if c["ideal"]["valid"]:
                        if k == 1:
                            require(
                                abs(c["ideal"]["margin"] - c["real"]["ideal_margin"])
                                <= 1e-5,
                                "k1 invariant violated",
                            )
                        if regime == "I" and r["positive_cosine"] >= 0:
                            require(
                                c["I_invariant"] == "passed"
                                and c["ideal"]["margin"]
                                <= c["real"]["ideal_margin"] + 1e-5,
                                "I invariant violated",
                            )
                            flags["I_invariants_passed"] += 1
                    if k != 8:
                        continue
                    pre = f"k8/{regime}/"
                    metrics[pre + "source_margin"] = s["contributor_margins"]
                    metrics[pre + "real_margin"] = c["real"]["ideal_margin"]
                    metrics[pre + "same_species"] = s["same_species_fraction"]
                    for path_name in ["ideal", "mixed"]:
                        v = c[path_name]
                        metrics[pre + path_name + "/margin"] = v["margin"]
                        metrics[pre + path_name + "/norm"] = v["norm"]
                        metrics[pre + path_name + "/nonpositive"] = (
                            None if v["margin"] is None else v["margin"] <= 0
                        )
                    for path_name in ["ideal", "mixed"]:
                        for metric in ["delta_margin", "delta_absolute_margin"]:
                            metrics[pre + path_name + "/" + metric] = c[path_name][
                                metric
                            ]
                for pair, o in entry["overlaps"].items():
                    if o is None:
                        continue
                    a, b = [entry["regimes"][x]["support"] for x in pair]
                    require(
                        o["fraction"] == len(set(a) & set(b)) / k
                        and o["equal"] == (set(a) == set(b)),
                        "Overlap arithmetic differs",
                    )
                    if pair == "CI":
                        require(o["fraction"] == 0, "C/I intersection")
                    if k == 8:
                        metrics["k8/paired/" + pair + "/overlap"] = o["fraction"]
                        metrics["k8/paired/" + pair + "/equal"] = o["equal"]
                if k == 8:
                    for regime, h in r["historical"].items():
                        metrics[f"historical/{regime}/equal"] = h["equal"]
                        metrics[f"historical/{regime}/overlap"] = h["overlap"]
            for scope in scopes:
                counts[(scope, w, seed, u)] += 1
                for metric, v in metrics.items():
                    add(scope, w, seed, u, metric, v)
        if len(seen) % 100 == 0:
            print("VERIFIED_BATCHES", len(seen), flush=True)
    require(
        seen == expected and origins == {"reused": 166, "new": 458},
        "Incomplete final grid/recovery counts",
    )
    for (seed, pop, u, off, i), pair in states.items():
        require(
            set(pair) == {100, 500} and pair[100][:2] == pair[500][:2],
            "Unmatched cross-state query",
        )
        scopes = (
            ["calibration_feasibility"]
            if pop == "calibration_feasibility"
            else ["extended"] + (["primary"] if off == 0 else [])
        )
        for scope in scopes:
            counts[(scope, "paired", seed, u)] += 1
            for k, (a, b) in enumerate(zip(pair[100][2], pair[500][2]), 1):
                label = (
                    "both"
                    if a and b
                    else "early_only" if a else "late_only" if b else "neither"
                )
                for status in ["both", "early_only", "late_only", "neither"]:
                    add(
                        scope,
                        "paired",
                        seed,
                        u,
                        f"k{k}/eligibility/{status}",
                        label == status,
                    )

    def summarize(v):
        a = np.asarray(v, dtype=float)
        return dict(
            count=len(a),
            mean=float(a.mean()),
            median=float(np.median(a)),
            q25=float(np.quantile(a, 0.25)),
            q75=float(np.quantile(a, 0.75)),
            minimum=float(a.min()),
            maximum=float(a.max()),
        )

    groups = []
    checked = 0
    for pop in ["primary", "extended", "calibration_feasibility"]:
        for w in [100, 500, "paired"]:
            strata = [key for key in values if key[:2] == (pop, w)]
            metric_names = sorted(set().union(*(values[key] for key in strata)))
            result = {}
            for metric in metric_names:
                combined = [x for key in strata for x in values[key].get(metric, [])]
                if not combined:
                    continue
                v = summarize(combined)
                means = defaultdict(list)
                for key in strata:
                    a = values[key].get(metric, [])
                    if a:
                        means[key[2]].append(float(np.mean(a)))
                streams = 1 if pop == "calibration_feasibility" else 2
                v["macro_mean"] = (
                    float(np.mean([np.mean(means[s]) for s in p["training_seeds"]]))
                    if all(len(means[s]) == streams for s in p["training_seeds"])
                    else None
                )
                result[metric] = v
                target = reported[(pop, w, "all", "all")]["metrics"].get(metric)
                if target is not None:
                    require(
                        target["count"] == v["count"]
                        and np.isclose(
                            target["pooled_mean"], v["mean"], rtol=0, atol=1e-12
                        )
                        and np.isclose(
                            target["quantiles"]["0.5"], v["median"], rtol=0, atol=1e-12
                        ),
                        "Recomputed statistic differs: " + metric,
                    )
                    require(
                        (target["macro_mean"] is None and v["macro_mean"] is None)
                        or (
                            target["macro_mean"] is not None
                            and v["macro_mean"] is not None
                            and np.isclose(
                                target["macro_mean"],
                                v["macro_mean"],
                                rtol=0,
                                atol=1e-12,
                            )
                        ),
                        "Macro arithmetic differs: " + metric,
                    )
                    checked += 1
            groups.append(
                dict(
                    population=pop,
                    weight=w,
                    query_exposures=sum(counts[key] for key in strata),
                    metrics=result,
                )
            )
    # Preserve per-seed rates, so pooled availability is not mistaken for independent replication.
    seed_groups = []
    for seed in p["training_seeds"]:
        for w in [100, 500]:
            keys = [
                key for key in values if key[:2] == ("extended", w) and key[2] == seed
            ]
            seed_groups.append(
                dict(
                    seed=seed,
                    weight=w,
                    metrics={
                        metric: summarize(
                            [x for key in keys for x in values[key][metric]]
                        )
                        for metric in [
                            "zero_nonpositive",
                            "k2/eligible",
                            "k8/eligible",
                            "k8/I/ideal/nonpositive",
                            "k8/C/ideal/nonpositive",
                        ]
                    },
                )
            )
    verification = dict(
        run_id=RUN,
        status="verified_complete_download",
        bundle_id=BUNDLE,
        verified_inventory_files=len(inventory["files"]),
        verified_source_files=len(sources),
        batch_forwards=624,
        query_exposures=39936,
        reused_forwards=166,
        new_forwards=458,
        audit_updates=0,
        current_native_preparation_updates=1500,
        previous_native_preparation_updates=500,
        total_observed_preparation_updates=2000,
        recomputed_aggregate_statistics=checked,
        raw_record_checks=dict(flags),
        scope="All inventory bytes, source bundle, complete grid, native identities, recovery lineage and raw numeric invariants verified. Selected aggregate means, medians and seed/stream macro means recomputed. Embeddings and GPU forwards not rerun; remaining summary metrics authenticated by checksum only.",
        final_drive_tree_verified=True,
        drive_flush_log_available=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_manifest.json",
        "completion.json",
        "recovery_completion.json",
        "recovery_sources.json",
        "DRIVE_BACKUP_MANIFEST.json",
        "source_manifest.json",
        "freeze_manifest.json",
        "protocol.json",
        "plan.json",
        "shards.json",
        "REPORT.md",
    ]:
        shutil.copyfile(run / name, out / name)
    # Retain the exact full summary as deterministic gzip; raw 624 shards stay in the original download.
    with (out / "summary.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
            with (run / "summary.json").open("rb") as source:
                shutil.copyfileobj(source, f)
    write(out / "verification.json", verification)
    write(out / "analysis.json", dict(groups=groups, per_seed_extended=seed_groups))
    write(
        out / "retained_manifest.json",
        {
            f.name: dict(sha256=sha(f), bytes=f.stat().st_size)
            for f in out.iterdir()
            if f.is_file() and f.name not in ["retained_manifest.json", "README.md"]
        },
    )
    print(json.dumps(verification, indent=2))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiment_results/clip_geometry_audit_2026-10",
    )
    args = parser.parse_args()
    verify(args.run, args.output)
