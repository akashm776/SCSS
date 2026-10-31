# D0 complete: feasibility and synthesis geometry

**D0 completed successfully. The proposed all-query nine-arm D1 is not feasible
as specified.** No k in 1–8 makes misranked-or-tied source selection available to
every intended query. Uniform mixtures also do not preserve the sources' ranking
side or closeness to the boundary. These are design findings, not training-utility
results.

The recovery contains the complete frozen grid: 624 forwards and 39,936 query
exposures, with 166 saved forwards reused and 458 newly computed. All three seeds
and both states are present. There are zero audit training updates, backward
passes, virtual optimizer steps or reporting evaluations. Checkpoint preparation
cost 1,500 native updates in recovery, plus 500 in the interrupted attempt.

All 2,668 downloaded inventory files and 118 issued source members verify.
Native fingerprints, batch membership, state-preservation records and recovery
lineage match. Selected statistics were independently recomputed from all query
records. No invalid mixtures or applicable I-Synth invariant violations were
found. This verifies saved evidence; it does not rerun the encoder.

## Feasibility

The primary population is the six initial batches, 384 query exposures per state.
The extended population contains those same batches plus the remaining planned
batches, 19,200 exposures per state. They are overlapping views, not additional
independent samples. Calibration is separate: 768 exposures per state.

| Population and measurement | Weights 100 | Weights 500 |
|---|---:|---:|
| Primary: no nonpositive-margin candidate | 18.75% | 28.91% |
| Primary: I support available at k=8 | 35.68% | 25.26% |
| Primary: common C/I eligibility at k=8 | 35.16% | 25.26% |
| Extended: no nonpositive-margin candidate | 19.19% | 26.92% |
| Extended: I support available at k=8 | 36.65% | 25.54% |
| Extended: C support available at k=8 | 99.45% | 99.88% |
| Extended: common C/I eligibility at k=8 | 36.10% | 25.42% |
| Calibration: common C/I eligibility at k=8 | 37.37% | 24.87% |

Zero nonpositive candidates means that reducing k cannot solve availability.
Even C at k=8 has occasional shortages, so its all-query hard-failure design
would also stop. An audit can complete successfully while finding these treatments
infeasible; that is what happened here.

Smaller k improves common coverage, but does not restore all-query feasibility:

| k | Extended C/I eligible early | Extended C/I eligible later | Eligible at both states |
|---:|---:|---:|---:|
| 1 | 80.74% | 73.08% | 69.72% |
| 2 | 69.17% | 59.11% | 55.56% |
| 3 | 60.80% | 49.50% | 45.98% |
| 4 | 54.12% | 42.36% | 38.93% |
| 5 | 48.52% | 36.91% | 33.49% |
| 6 | 43.64% | 32.51% | 29.09% |
| 7 | 39.73% | 28.73% | 25.58% |
| 8 | 36.10% | 25.42% | 22.47% |

At k=8, 22.47% are eligible at both states, 13.63% only early, 2.95% only later
and 60.95% at neither. k=1 remains a feasibility diagnostic: normalized uniform
synthesis from one source does not furnish a distinct ideal synthesis action.
The all-query candidate list for k>=2 is empty.

The state difference appears in each seed. Extended zero-nonpositive rates span
18.91–19.34% early versus 26.48–27.28% later. Extended k=8 common C/I eligibility
spans 34.81–37.19% early versus 24.73–26.27% later. These are descriptive checks
across three reused development seeds; query exposures are not independent
replications.

## Source selection and generated geometry differ

The table shows pooled median ideal FP32 margins at k=8 in the extended population.
Each regime uses its own feasible query population; cross-regime differences in
this table are not matched-population effects.

| Candidate | Early median | Later median |
|---|---:|---:|
| C-Select | +0.001361 (n=19,094) | +0.002011 (n=19,177) |
| C uniform synthesis | −0.026080 (n=19,094) | −0.016460 (n=19,177) |
| I-Select | −0.000535 (n=7,037) | −0.000634 (n=4,904) |
| I uniform synthesis | −0.037846 (n=7,037) | −0.031316 (n=4,904) |

Although every C contributor has a positive margin, **96.44% early and 87.97%
later** of its ideal synthetic candidates have nonpositive margins. The reference
mixed-precision path gives 96.41% and 87.97%, respectively. Thus this is not merely
a distinction between ideal and mixed-precision candidate scoring.

I-Select is close to zero on its available population, but I uniform synthesis is
substantially more negative. All 156,411 applicable I-Synth invariant checks across
populations, queries and k passed. Uniform normalized mixing of these misranked
sources is not a method for softening the nearest incorrect-side example toward
the boundary in the audited nonnegative-positive-cosine case.

The correct treatment names remain **source-restricted uniform mixtures**.
Positive-margin sources cannot be interpreted as delivering correct-side synthetic
supervision, and neither label establishes a boundary-targeted generator. This
finding motivates controlling generated placement if boundary refinement is the
scientific question. It does not show that a boundary-targeted candidate would
improve training.

## Separability, semantics and historical precision

H/I mean support overlap at k=8 is only 29.89% early and 35.80% later, conditional
on I availability. H/I support equality is 8.44% and 10.36%; their selected real
candidate is never identical on those feasible k=8 comparisons. Equal supports
can yield the same uniform mixture but a different Select control, so H and I
cannot simply be removed as interchangeable treatments.

H/C mean overlap is 45.86% early and 56.56% later, with identical supports in
19.29% and 26.95% of C-feasible queries. The formal rules are distinct but sometimes
apply the same action. Full conditional denominators and paired geometry are
retained in the original summary.

Same-species candidates account for 1.36% early and 1.76% later of nonpositive
candidate margins, versus 0.36% and 0.37% of positive margins. They are enriched
but do not dominate by this label. Species identity is not a verified
false-negative label, and these counts do not identify the cause of utility.

The new FP32 selection path also changes support identities relative to historical
B/C selection on the same forward. Exact H agreement is 68.10% early / 83.01%
later; M agreement is 40.08% / 58.87%. D0 must not be described as an exact replay
of historical hard/moderate selection. The frozen audit anticipated this comparison.

## Consequences for D1

The original all-query k=8 design should not be launched unchanged. Two decisions
must be made before a new D1 freeze:

1. **Treatment population:** define a common eligible-query experiment, or audit
   a larger candidate pool. A larger pool is not a guarantee that every query has
   a misranked candidate. Dynamic branch-specific eligibility also changes which
   queries receive treatment; it estimates the utility of that selection policy,
   rather than holding exposure fixed across negative regimes.
2. **Generated placement:** decide whether the question concerns uniform mixtures
   of different source regimes, or explicitly boundary-targeted supervision.
   Those are different interventions. The current uniform generator does not
   implement the intended correct-side boundary refinement.

D0 does not choose k on expected usefulness, select a winning arm, prove an
early-hard/later-boundary curriculum, or show an improvement over native training.
**D1 remains unfrozen.** Feasibility and the source-to-generated-margin mismatch
are now measured constraints on its design.

[Verified evidence and reproduction](../experiment_results/clip_geometry_audit_2026-10/README.md).
The original partial-run archive remains preserved; its 166 forwards are a subset
of these results and are not counted again.
