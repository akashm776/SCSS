# Experiment D0: fixed-state geometry and feasibility audit

Status: **protocol frozen; runner not implemented; audit not run**.
Experiment ID: `clip_geometry_audit_v1`. D1 is intentionally **not frozen**.
This design was selected after Experiment C and the historical source-margin
analysis; no D0 result or D1 treatment outcome has been generated.

Question: **Can the proposed negative regimes be defined feasibly, comparably and
distinctly at native learner weights 100 and 500?** D0 cannot establish utility.

The [machine-readable specification](../configs/clip_geometry_audit.json) is the
complete operational contract. The [data/state plan](../configs/clip_geometry_audit_plan.json)
pins ordered source indices, caption epochs, processed-batch identity hashes and
state fingerprints. The [freeze manifest](../configs/clip_geometry_audit_freeze_manifest.json)
pins these files, this document and parent evidence. This is a local content freeze,
not external preregistration. Any protocol amendment requires a new version and
reason before the amended computation; do not overwrite these issued files.

## States, data and compute accounting

Use training seeds **271828, 161803, 141421**, stream seeds **2026100101,
2026100102**, native-only weights **100/500**, and batches of **64**. Reuse C's
ordered continuation batches and seed-specific update-100 caption epoch (2), with
identical captions and processed inputs at both weights. Exclude the paired image
from the 64-image candidate set, leaving 63 negatives. Do not filter by species.

Three separately labelled populations:

| Population | Batches per seed/state | Exposures per state | Purpose |
|---|---:|---:|---|
| Primary | First batch of each of two streams | 384 | Initial feasibility/geometry |
| Extended | All 50 batches of each stream | 19,200 | Fixed-starting-model coverage |
| Calibration feasibility | C's four calibration batches | 768 | Availability on potential future calibration inputs |

Primary is a subset of extended, not additional observations. Calibration inputs
are disjoint from continuation/report/frozen-probe inputs and are not duplicated
by stream. Their separate inclusion checks whether a future treatment could even
be calibrated on C's planned inputs; **no gradients or coefficients are computed**.
Indices must pass the parent split and image-level exclusion checks. The model
stays at its starting weights across every batch; offsets label data sequence
positions, not elapsed D0 training.

Load authenticated snapshots if available. If all states must be regenerated,
reproduce the native trunks and verify their archived starts/traces: **1,500 native
optimizer updates**, reported as checkpoint preparation. C did not export large
checkpoint tensors. Loading unauthenticated weights or treating C's endpoints as
native starts is prohibited. Preparation performs no report evaluation.

D0 itself has **zero backward passes, optimizer updates, scheduler updates, virtual
optimizer steps or continuations**. It requires 600 continuation-batch forwards
plus 24 calibration-batch forwards: 624 forwards and 39,936 query exposures total,
including 38,400 continuation and 1,536 calibration exposures. Native regeneration
cost is separate. D0 still costs GPU forward computation.

## Forward, scoring and preservation

Use C's A100 and package versions (including torch 2.11.0+cu128 and torchvision
0.26.0+cu128), CLIP checkpoint, processor, trainable policy and training-forward
bf16 precision. No automatic package replacement. Inside each audit operation,
use training mode with gradients disabled and the seed's saved update-100 RNG
state. Reset this RNG for every batch and use it at both weight states. All regimes
share one encoder forward. This defines a fixed-state audit, not the future RNG
trajectory of any D1 branch.

Restore and verify parameters, buffers, per-module modes, parameter gradients,
Python/NumPy/CPU/CUDA RNG and any optimizer/scheduler state before and after every
observation, including exceptional exits. Preserve backend precision flags too.
A state-preservation failure stops the audit. `no_grad()` alone is insufficient.

Detach the forward's image/text features, cast to FP32, then renormalize each row.
Require finite vectors with norms strictly greater than **1e-6**. With autocast
disabled, compute `S = Q @ V.T` in FP32. Freeze highest float32 matmul precision,
disable TF32 and reduced-precision reduction settings for candidate/scoring
computations, and restore the historical forward/preparation settings afterward.
Do not clamp scores. Record actual backend settings and dtypes.

Let `p_i = S[i,i]` and `r_ij = p_i - S[i,j]`. Use strict `r>0` and `r<=0`, recording
negative and exact-zero counts separately. Numerical validation tolerances do not
change classification thresholds. Use ascending batch index to break exact ties
with stable sorting, identically for all regimes.

## Supports and Select candidates

Audit every **k=1,...,8**, with k=8 as the original proposed support size:

| Regime | Ordered support | Select candidate |
|---|---|---|
| H | First k negatives by descending S | First member |
| M | Global similarity ranks 16 through 15+k | Rank 16 |
| C | First k strictly positive margins, ascending r | Smallest positive margin |
| I | First k nonpositive margins, ascending absolute r | Smallest absolute nonpositive margin |

M at k=8 is the old rank band, but the new scoring/tie rule can change membership;
it is not an exact historical treatment replay. I means **misranked-or-tied nearest
available sources**, not severe errors. C means **positive-margin-restricted nearest
available sources**, not a guaranteed absolute near-boundary region.

For each query save `N_positive`, `N_negative`, `N_zero`, and `N_nonpositive`;
verify `N_positive+N_nonpositive=63` and `N_negative+N_zero=N_nonpositive`.
Report every feasibility curve and the zero-nonpositive rate.

Unavailable supports are null, with reason and available count. Do not fill,
truncate, repeat contributors or adapt k. Source overlap and real/synthetic
candidates are then undefined. This is an expected D0 observation, not a fatal
failure. If a support exists but its synthetic mixture is invalid, retain its
source IDs/statistics and real candidate; only affected synthetic fields are null.
Every statistic carries numerator/denominator where applicable, coverage and
missing-reason counts. Never replace missing values with zero.

## Two synthetic geometry paths

For every feasible support, calculate a uniform mean and normalized synthetic
candidate through both paths specified in the JSON contract:

1. **Ideal FP32:** use the FP32-renormalized source embeddings, FP32 mean and
   normalization, and FP32-renormalized query for all cosine measurements.
2. **Reference mixed precision:** use the same selected IDs, the existing uniform
   weight arithmetic, original forward image features, bf16-autocast matrix product
   and existing `F.normalize(..., eps=1e-12)` path. Record its actual forward score
   and also its cosine after FP32 renormalization against the FP32 query.

For the reference Select candidate record both the original forward matrix entry
and the new FP32 score. This separates changed selection geometry from changed
candidate arithmetic. It specifies the candidate mechanism D0 audits; it does not
freeze D1's arms, pressure, population or estimand.

Before normalizing either mixture, measure its actual pre-normalization norm in
FP32. A nonfinite norm or norm **<=1e-6** makes that synthetic candidate numerically
invalid: record it without a replacement and continue the audit. Nonfinite encoder
outputs or scoring matrices instead stop the run after saving a failure record.

Save positive cosine, contributor margins, selected real margin, synthetic margin,
mixture norm, `r_synth-r_real`, and `abs(r_synth)-abs(r_real)` with explicit path
labels. Signed displacement alone is not a measure of distance to the boundary.
Casting cannot recover precision lost upstream.

## Mathematical and implementation invariants

For unit embeddings, a valid ideal I mixture and `p_i>=0`, every contributor has
`q_i dot v_j >= p_i`. The uniform mean mu satisfies `||mu||<=1`, giving:

```text
q dot normalize(mu) >= q dot mu >= min_j(q dot v_j)
r_I_synth <= r_I_select <= 0.
```

Verify this with **absolute tolerance 1e-5**. Verify ideal unit norms and k=1
Select/Synth identity with the same tolerance. Do not apply the I theorem when
`p_i<0`, the mixture is invalid, or by mixing different numerical paths. Violations
beyond tolerance in the ideal path stop the audit, preserving the failing record;
no automatic retries or tolerance changes. Mixed-path discrepancies are separately
reported, not silently treated as violations of the exact ideal construction.

Other required checks: no diagonal member; correct support size; C/I disjointness;
and, when C is feasible, `|H intersect C| = k - count(nonpositive H margins)`.
At k=1, uniform synthesis is not a distinct treatment from Select.

## Eligibility, overlap and semantic summaries

Define `E_k^w = (N_nonpositive>=k and N_positive>=k)`. Match query exposures across
weights by population, seed, stream where present, batch offset, query position,
source index and caption index. Report both-state eligibility and all four
transitions: both, early-only, late-only, neither.

Report all six regime-pair support overlaps `|A intersect B|/k` and exact-set
agreement on their common feasible population. Also report candidate/path validity.
Comparisons across states, regimes or paths use their common eligible/valid
intersection with coverage; show each regime's available population separately.
Supports, candidate paths and Select choices must all be considered before calling
two interventions identical.

Record source and Select same-species rates and candidate-population rates
conditional on margin sign. Same-species association is a semantic descriptor,
not proof that a task negative is erroneous. Do not filter on species.

Save query-level numeric records, ordered support IDs and deterministic canonical
hashes; no captions, images or embedding vectors. Summaries split by population,
state, seed, stream, batch, k, regime and path. Report linear-interpolated quantiles,
frozen histograms, exact-zero mass and descriptive `|r| <= 0.001, 0.005, 0.01`
coverage. These thresholds are neither selection filters nor utility scores.

Query exposures and contributor entries are distinct units. Conditional pooled
histograms/quantiles must state their denominator. Report stream means within seed,
then equal seed means. A conditional macro mean is null if a required stratum has
no observations; do not silently drop it. Show separately labelled pooled results.
Calibration batches are averaged by seed without artificial stream replication.
No significance, independent replication or downstream-utility claim is made.

## Historical comparison and design decisions

On the same encoder forward, apply B's exact historical H/M k=8 support routine
and compare with new H/M. Record support equality/overlap and scoring sign changes.
This is a comparison of scoring rules on the same input/state, not replay of C's
model after subsequent native updates.

The earlier **168/384 and 217/384** figures are historical lower bounds on query
infeasibility at weights 100/500 inferred from C's aggregate source-support records.
They are not exact query counts, new FP32 measurements, or D0 results. The pinned
[derivation](../experiment_results/clip_pressure_2026-10/initial_hard_support_margin_audit.json)
contains the counts from which those bounds follow.

Frozen decision rules:

- An all-query C/I candidate design requires **100%** availability at both states
  on extended continuation and separate calibration exposures, with valid
  synthesis through both paths. Report every k>=2 satisfying this; do not
  automatically choose k. Observed feasibility is not a future-trajectory guarantee.
- Any zero-nonpositive query rules out all-query in-batch I for all k>=1 on that
  audited population. Smaller k does not fix it.
- Report eligible-query coverage and cross-state transitions without automatically
  enrolling a subset or selecting a new exposure-normalization scheme.
- Report overlap/equality distributions. Exact redundancy requires identical
  supports/candidates on all common feasible queries with nonempty coverage.
  No automatic near-redundancy cutoff or arm removal is authorized by D0.
- Do not choose k, supports or arms to optimize apparently helpful synthetic margins,
  expected utility, reporting loss or retrieval. Geometry characterizes the
  intervention; it is not an outcome-selection heuristic.
- A larger candidate pool or changed scoring rule requires a new audit version.
  D1 needs its own freeze after D0, including population, support size, arms,
  strength, primary estimand and branch-time infeasibility handling. Failed D1
  branches cannot silently be excluded from a nominal complete-grid analysis.

The report may conclude that the proposed experiment is infeasible. That is a
successful D0 finding. It may not conclude that I-Synth helps, that synthesis adds
utility, or that the model needs a particular curriculum.

## Deliverables and acceptance

Retain strict-JSON compressed query records, machine-readable summaries, a Markdown
report, protocol/plan/source/environment manifests, batch identity hashes, state
preservation checks, numerical validity counts and completion/failure records.
Null represents undefined values; NaN/Infinity are prohibited. No reporting-pool
forward evaluation, loss/R@1, auxiliary objective, calibration gradient, optimizer
probe or learned selector belongs in D0. No raw captions belong in the evidence
export. Runner/source freeze and implementation tests precede any GPU run; this
protocol freeze alone is not proof of implementation or completed auditing.
