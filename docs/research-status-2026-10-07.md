# SCSS research status — October 7, 2026

The completed weight-state × LR experiment preserves the early-benefit/later-harm
loss contrast under **both tested LR trajectories**. The contrast follows starting
weights in this controlled comparison. Swapping these natural schedule segments
does not remove it. This narrows the schedule explanation without identifying
representation geometry as the mechanism.

[Frozen protocol](clip-weight-lr.md) ·
[Verified evidence](../experiment_results/clip_weight_lr_2026-10/README.md) ·
[Prior findings](research-status-2026-10-03.md)

## Experiment and primary results

Three reused training seeds × two continuation streams × native-only starting
weights at updates 100/500 × external LR trajectories at positions 100/500 ×
native/sustained uniform-top-8 objectives. All branches start with fresh AdamW
moments and counters. Processed batches, captions and RNG are shared across all
eight branches within a seed/stream; coefficient is fixed at 0.06005351588542947.
There are 1,500 trunk and 2,400 branch updates, with a 50-update primary horizon.

Each entry below is **sustained minus native reporting loss**, averaged over streams
within seed, then equally across seeds. Negative is helpful. These are treatment
effects within each cell, not comparisons of raw loss between early/later models.

| Starting weights | LR from update 100 | LR from update 500 |
|---|---:|---:|
| Update 100 | −0.0016122062 | −0.0015408524 |
| Update 500 | +0.0017580303 | +0.0017639800 |

Every cell has the displayed sign in **all six seed–stream pairs**. These are three
training seeds with nested continuations, not 24 independent replications.

The three prespecified factorial contrasts on these loss effects are:

| Contrast | Value |
|---|---:|
| Weight-state main effect | +0.0033375344 |
| LR main effect | +0.0000386518 |
| Weight × LR interaction | −0.0000654041 |

The weight-state simple effects are +0.0033702365 under early LR and +0.0033048324
under later LR. LR simple effects are +0.0000713538 at early weights and
+0.0000059498 at later weights. The interaction is the difference of those LR
effects. Its small aggregate hides opposite signs across seeds; no equivalence or
statistical-significance claim is made.

## Individual seeds

Each row averages the two streams. E/L denote early/later weights first, LR second.

| Training seed | EE loss effect | EL loss effect | LE loss effect | LL loss effect |
|---|---:|---:|---:|---:|
| 271828 | −0.0022536430 | −0.0021740001 | +0.0012357265 | +0.0011448408 |
| 161803 | −0.0009484628 | −0.0009998253 | +0.0023025996 | +0.0025836602 |
| 141421 | −0.0016345128 | −0.0014487317 | +0.0017357647 | +0.0015634392 |

| Training seed | Weight main effect | LR main effect | Interaction |
|---|---:|---:|---:|
| 271828 | +0.0034041052 | −0.0000056214 | −0.0001705286 |
| 161803 | +0.0034172740 | +0.0001148490 | +0.0003324231 |
| 141421 | +0.0031912242 | +0.0000067278 | −0.0003581066 |

## Secondary horizons and retrieval

Mean loss effects retain the early/later sign distinction at every measured
nonzero horizon. This does not establish monotonicity across training ages; only
two weight checkpoints were tested here.

| Continuation updates | EE | EL | LE | LL |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | −0.000262312 | −0.000254621 | +0.000062816 | +0.000061669 |
| 5 | −0.001075912 | −0.001036722 | +0.000288270 | +0.000277725 |
| 10 | −0.001279102 | −0.001201067 | +0.000888414 | +0.000830451 |
| 25 | −0.001715388 | −0.001664267 | +0.001597110 | +0.001382870 |
| 50 | −0.001612206 | −0.001540852 | +0.001758030 | +0.001763980 |

Horizon-50 mean bidirectional reporting-pool R@1 effects are **0.000000,
+0.048828, −0.032552, 0.000000 percentage points**, respectively for EE, EL, LE,
LL. Seed-level retrieval effects are mixed, including opposite loss/retrieval
directions. This is not evidence of an overall retrieval improvement.

## What this control establishes and leaves open

- Swapping the tested LR trajectories does not reverse the sign associated with
  either weight checkpoint, after removing optimizer-state inheritance and
  matching continuation inputs/RNG across weight ages.
- The LR contrast is deliberately modest: the first group's early LR runs from
  1.000000e−5 to 9.995788e−6; its later LR runs from 9.721882e−6 to 9.650423e−6.
  Later is about 2.8–3.5% lower. Other groups are one tenth of these values.
  This does not rule out effects of substantially different LR schedules.
- Starting weights encode native learning history and many model properties.
  Weight association is not a specific representation-geometry explanation.
  Fresh AdamW still adapts during continuation; fixed auxiliary coefficient does
  not equal fixed effective gradient pressure.
- The seeds, early continuation streams and 512-image reporting pool are reused.
  The reporting images are excluded from this training, but the study is an
  exploratory follow-up rather than untouched external confirmation.
- The early-weight/early-LR branches exactly reproduce the prior reset study.
  The later diagonal uses common **early** continuation batches/RNG here, so its
  +0.0017639800 is not an exact replay of the old +0.0016371505 late reset effect.

The next planned question is whether **distinct difficulty regimes change their
relative utility across weight states**: compare extreme-hard versus an explicitly
separated semi-hard quantile band under a common LR, with native controls and
matched real/synthetic constructions. Audit candidate-set overlap first. A learned
selector comes after measuring the regime utilities. This experiment does not yet
show that semi-hard negatives rescue later usefulness.

## Verification and execution provenance

Run `clip_weight_lr_20261007T024147_218129Z`, issued bundle
`9d1f48b37336cc190ad7be33ce77dbc325bc1f7422e0742ab86d26bba9bc07b0`.
All **548 manifested files**, plus the pinned backup manifest, were checked.
All **48 endpoint caches** independently reproduce loss and both retrieval
directions. The audit also verifies source snapshots, native replay fingerprints,
early-diagonal observations/traces, reset and matching records, 288 standalone
observations, and every seed/timepoint factorial calculation.

The downloaded successful subprocess log records `CHILD_EXIT_CODE: 0` and
`DRIVE_TREE_VERIFIED`. Two earlier logs stop at preflight because torch was cu130
rather than the pinned cu128 build; they are not additional completed experiments.
The successful run records the pinned cu128 environment. The subprocess log does
not contain the outer cell's `DRIVE_SYNC_COMPLETE`; local manifest verification
establishes the downloaded tree's consistency, not remote flush completion.

The public export retains 489 files and omits 60 caption-bearing/binary input
files. Clone-only checks validate retained evidence and arithmetic; omitted NPZ
caches require the original download to recompute. No new training was launched
during this analysis. See [verification commands](evidence-and-reproduction.md).
