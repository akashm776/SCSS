# 1. Introduction

[Front page](../../README.md) · [Contents](../../README.md#contents) · [Next](related-work.md)

Synthetic negatives offer a way to change a contrastive learner's training signal without collecting another example. But a construction that is difficult for a model to rank is not necessarily useful for updating that model. Its effect can depend on the current weights, optimization state, source examples, strength, and duration of the intervention.

We study **conditional intervention utility**: the change in a fixed reporting objective after a matched continuation with auxiliary supervision, relative to native training from the same starting state. Our test bed is CLIP fine-tuning on CUB image–caption pairs. The work progressed from testing negative constructions to asking when they help and which controls change that conclusion.

The motivating research program, **State-Conditioned Synthetic Supervision (SCSS)**, concerns reusable rules that generate useful training signals for a learner. The present evidence addresses a prerequisite: utility varies across tested learner states. It does not yet demonstrate a learned generator library, data compression, or a general curriculum.

## Research questions

1. Does the same synthetic-negative construction have different utility early and later in fine-tuning?
2. Can prior auxiliary exposure, optimizer state, learning-rate trajectory, or initial gradient pressure account for that pattern under the tested controls?
3. How do source selection and synthesis compare on a common treatment population?
4. Can generated ranking-margin placement be manipulated precisely enough to test its utility? This final question remains a validation and planned-training study.

## Contributions supported by the current evidence

**A matched-continuation experimental framework.** Shared inputs and explicit controls make short-horizon intervention effects measurable. History, optimizer-reset, and weight-by-learning-rate controls narrow several explanations without identifying a unique mechanism.

**Evidence of state-dependent loss utility.** In the completed source-regime study, all eight interventions improve aggregate reporting loss from early weights and worsen it from later weights. Two-source synthesis improves on its specified one-real-negative control early in all four source regimes for each seed average.

**Negative findings that constrain the interpretation.** The proposed correctly-ranked-versus-misranked source curriculum is not consistent across seeds. Early loss gains do not translate to aggregate R@1 gains. Geometry shows that a synthetic candidate can cross the ranking boundary even when all its sources are correctly ranked.

**An explicit distinction between a treatment and its numerical implementation.** Placement preflight exposed a mixed-precision inconsistency, followed by a staged FP32-head repair, gradient validation, and calibration. These establish execution properties, not placement utility.

[Results](results.md) connect each statement to measurements; the [evidence index](evidence.md) provides the supporting artifacts. The [research ledger](../experiment-ledger.md) preserves earlier experiments and changes in interpretation.

## Scope and terminology

This is an evolving research report, not a claim of a new state-of-the-art training algorithm. “Early” and “late” refer to specific checkpoints in this fine-tuning schedule, not universal stages of learning. “Synthetic” means a generated embedding used as an auxiliary negative, not a generated image or caption.

The project began as **OTCO**, investigating optimal-transport-derived constructions. An OT advantage was not established. Historical code names and artifact paths retain that name; the current scientific question is broader. The separate [research agenda](../research-program.md) describes longer-term hypotheses.

---

[Front page](../../README.md) · [Contents](../../README.md#contents) · [Next](related-work.md)
