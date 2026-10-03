# State-Conditioned Synthetic Supervision (SCSS)

**Reusable generators of training signals, evaluated by when they help a learner.**

This research program asks whether some reliance on stored training examples can
be replaced by stored **generation functions**: rules or learned generators that
produce useful supervision for a model in a particular learning state. The aim is
not simply to accumulate more synthetic examples, but to understand which generated
signals to supply, when, at what strength, and for how long.

The current experimental test bed is synthetic negative construction in CLIP/CUB-200.
It tests a prerequisite of that broader idea: **the same construction rule can help
in one state and hurt in another**. We have not yet demonstrated a learned generator
library, reduced data requirements, a general curriculum, or reliable retrieval gains.

Formerly **OTCO: Optimal Transport Contrastive Learning**. OT weighting was an
initial construction hypothesis, not an established advantage. Historical experiment
IDs, source snapshots, Python project identifier and Colab Drive paths retain `otco`
or `OTCO` for reproducibility. The repository URL is unchanged.

[Research program](docs/research-program.md) ·
[Complete experiment ledger](docs/experiment-ledger.md) ·
[Latest results and interpretation](docs/research-status-2026-10-03.md) ·
[Reproduce and verify](docs/evidence-and-reproduction.md)

## What the evidence currently says

As of **October 3, 2026**, the strongest result is a controlled early/later contrast
in short matched continuations—not an overall synthetic-data performance win.
The latest studies use three paired training seeds, two continuation streams,
and a reused 512-image reporting pool. Streams are not independent training seeds.

| Starting update | Native history, retained AdamW | Prior gated history, retained AdamW | Native history, reset AdamW |
|---|---:|---:|---:|
| 100 | −0.001116 | −0.001116 | −0.001612 |
| 500 | +0.001491 | +0.001495 | +0.001637 |

Entries are **sustained auxiliary minus native reporting loss after 50 updates**;
negative is helpful. All six seed-stream pairs at each age have the corresponding
sign in each condition. The early history comparison is an exact matching control,
and the reset study exactly replays the retained branches: these are not extra
independent replications.

- Prior auxiliary exposure is not necessary for later harm in this setting.
- Resetting AdamW moments **and parameter step counters** does not remove it.
- This does not identify representation geometry as the cause: weights, learning
  rates and native-training exposure still differ between ages.
- Reporting-loss effects do not establish retrieval gains. Original long-run CLIP
  ablations found no final canonical R@1 win over baseline; policy effects remain
  small and sensitive to the comparison and reporting setup.

The [latest report](docs/research-status-2026-10-03.md) contains the interaction
contrasts, time course, per-seed results, R@1 and limitations. The
[ledger](docs/experiment-ledger.md) includes the earlier negative results, partial
run, replay controls and analyses rather than presenting only favorable findings.

## Research direction

The central object is **conditional intervention utility**: the change in a specified
reporting objective produced by a generated training signal, relative to a matched
native continuation, at a fixed horizon. State includes more than training step:
model parameters, optimizer state, schedule and training history can all matter.

The next proposed diagnostic crosses early/later native weights with controlled
learning-rate trajectories under freshly initialized AdamW. It is **proposed, not
implemented or run**. The broader program must subsequently test multiple generator
rules, prediction using separate meta data, transfer to unseen states/tasks, and
equal-budget comparisons against stored-example alternatives. See the
[research program and falsification criteria](docs/research-program.md).

## Repository map

| Location | Purpose |
|---|---|
| [`model/clip_training.py`](model/clip_training.py), [`src/`](src/) | Construction rules, objectives, matched-branch experiments |
| [`configs/`](configs/), [`tests/`](tests/) | Frozen protocols, replay references and local tests |
| [`colabs/`](colabs/) | Versioned pasteable experiment cells and runners |
| [`experiment_results/`](experiment_results/README.md) | Small numerical evidence, manifests, historical source snapshots and audits |
| [`scripts/`](scripts/) | Bundle builders, CPU analyses and evidence verifiers |
| [`docs/experiment-ledger.md`](docs/experiment-ledger.md) | Study-by-study questions, results and claim boundaries |

Protocol documents/source overlays retain their original preparation-time wording.
For current completion status, use the ledger and dated results—not an old “next
run” paragraph. Original inputs are not rewritten to fit the current interpretation.

## Run and verify

```bash
uv sync
uv run --with pytest python -m pytest -q
uv run python -m scripts.archive_clip_october_evidence --verify-only
```

Generic local dependencies are not the historical GPU replay environment. Recent
Colab studies require the issued cell's exact package and A100 checks, including
`torch 2.11.0+cu128` / `torchvision 0.26.0+cu128`; `cu130` is not an interchangeable
replay environment. Do not disable those guards or rerun completed studies blindly.

Public exports omit checkpoint tensors, raw captions/images and endpoint NPZ caches.
Clone-only verification checks retained evidence and paired-history arithmetic;
recomputing cache metrics requires the original downloaded runs. See
[verification scope and commands](docs/evidence-and-reproduction.md).
