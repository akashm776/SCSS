# SCSS Experiment B results — October 7, 2026

Synthetic semi-hard has a larger advantage over synthetic hard at later weights,
but **does not make the later intervention beneficial**. At the prespecified
50-update endpoint it is also better on average at early weights, so the predicted
“hard early, semi-hard later” crossover is not the aggregate endpoint result.
Real-negative controls show a different pattern, and semi-hard applies substantially
less gradient pressure. These distinctions determine what can be inferred.

[Frozen protocol](clip-difficulty.md) ·
[Verified archive](../experiment_results/clip_difficulty_2026-10/README.md) ·
[Experiment A](research-status-2026-10-07.md)

## Design and primary endpoint

Native-only weights 100/500; common update-100 LR trajectory; fresh AdamW moments
and counters. Five arms share continuation batches, captions and initial RNG within
each seed/stream. Hard support is the historical top eight; semi-hard support uses
disjoint ranks 16–23. Real arms select each support's most similar member; synthetic
arms normalize its uniform embedding mixture. Fixed coefficient 0.06005351588542947.
Three reused training seeds, two streams, 50-update continuations: 1,500 native
trunk + 3,000 branch = **4,500 optimizer updates**. Reporting uses the reused
512-image pool excluded from this training. No external confirmation claim.

Below: arm minus matched native loss at horizon 50. Negative helps. Average streams
within seed, then seeds equally. Native effects are zero by definition.

| Intervention | Weights 100 | Weights 500 |
|---|---:|---:|
| Hard synthetic | −0.0016122062 | +0.0017580303 |
| Semi-hard synthetic | −0.0018507969 | +0.0004180784 |
| Hard real | −0.0001464353 | +0.0001972352 |
| Semi-hard real | −0.0002021395 | +0.0003295696 |

Both synthetic arms help early in all six seed–stream pairs and harm later in all
six. Semi-hard reduces the **mean excess later loss by 76.2%** relative to hard;
that percentage is a ratio of small treatment effects, not a 76.2% reduction in
the model's total loss. Native remains preferable on the measured later endpoint.

For `Q(w) = D(w, synthetic_semihard) − D(w, synthetic_hard)`:

| Contrast | Loss difference |
|---|---:|
| Q(100) | −0.0002385907 |
| Q(500) | −0.0013399519 |
| **Primary: Q(500) − Q(100)** | **−0.0011013612** |

The primary interaction is negative in all three seed means and all six matched
stream contrasts. The directional crossover would require `Q(100) > 0` and
`Q(500) < 0`; the mean endpoint does not satisfy the first condition. Semi-hard
beats hard in four of six early pairs and all six later pairs. These are nested,
reused observations, not six new independent training replications or a significance
test. Native/hard-synthetic branches replay Experiment A exactly.

## Individual seeds

Two-stream means of loss effects. SH/SS mean synthetic hard/semi-hard.

| Seed | Early SH | Early SS | Later SH | Later SS | Primary interaction |
|---|---:|---:|---:|---:|---:|
| 271828 | −0.0022536430 | −0.0019408407 | +0.0012357265 | +0.0004033822 | −0.0011451466 |
| 161803 | −0.0009484628 | −0.0016041534 | +0.0023025996 | +0.0004509556 | −0.0011959534 |
| 141421 | −0.0016345128 | −0.0020073966 | +0.0017357647 | +0.0003998973 | −0.0009629835 |

Only seed 271828 has the directional endpoint crossover in its two-stream mean.
All six stream-level records are in [diagnostics.json](../experiment_results/clip_difficulty_2026-10/diagnostics.json).

The real-negative secondary contrast goes in the opposite direction: semi-minus-hard
is −0.0000557042 early, +0.0001323344 later; late-minus-early is **+0.0001880386**.
That interaction is positive in all three seed means and five of six stream
contrasts. Semi-hard real is worse than hard real in all six later pairs.
Thus a universal “prefer less-hard negatives later” rule is not supported by
these two constructions. Their gradient pressures also differ substantially.

## Continuation horizon matters

The shorter, prespecified secondary horizons do show the directional crossover
in aggregate synthetic effects. This does not replace the primary horizon-50 test.

| Horizon | Early semi-minus-hard | Later semi-minus-hard | Interaction |
|---|---:|---:|---:|
| 1 | +0.000156968 | −0.000029741 | −0.000186709 |
| 5 | +0.000508591 | −0.000188758 | −0.000697349 |
| 10 | +0.000454571 | −0.000535307 | −0.000989878 |
| 25 | +0.000503825 | −0.001079037 | −0.001582862 |
| **50** | **−0.000238591** | **−0.001339952** | **−0.001101361** |

Early hard gives the larger initial benefit, while semi-hard overtakes it by 50
updates in the aggregate. At later weights, semi-hard is less harmful throughout
these observed horizons, but its mean effect versus native remains harmful.
Training age and continuation duration are different axes; this is not evidence
of a monotonic curriculum over all training ages.

## Gradient pressure is an unresolved explanation

Each entry is the mean of per-probe `||coefficient × auxiliary gradient|| / ||native gradient||`
at offset zero, before branch trajectories diverge. Ratios are shown as percentages.
These are raw gradient norms before clipping/AdamW, not realized update ratios.

| Intervention | Weights 100 | Weights 500 |
|---|---:|---:|
| Hard synthetic | 10.523% | 6.603% |
| Semi-hard synthetic | 4.431% | 1.951% |
| Hard real | 1.475% | 1.682% |
| Semi-hard real | 0.163% | 0.168% |

Semi-hard synthetic pressure is about 42% of hard at early weights and 30% at
later weights (ratios of the displayed mean ratios). The gap persists at offsets
10 and 25. Matching the scalar coefficient did not match pressure. We therefore
cannot attribute reduced harm specifically to improved negative geometry; a
weaker auxiliary contribution is an alternative explanation to test.

Initial native/auxiliary gradient cosines are +0.053590/−0.028818 for hard/semi-hard
synthetic at early weights and −0.004538/−0.075837 at later weights. These are
training-batch native-gradient comparisons, not the older held-out alignment
predictor or an optimizer-aware utility estimate. They should not be substituted
for realized continuation utility.

## Did the candidate rules differ?

All 3,000 logged training-step records show zero hard/semi support overlap and no
diagonal inclusion (192,000 query selections, including replayed/correlated states).
The minimum hard-versus-semi support gap is nonnegative; boundary-gap ties occur
in approximately 0.0177% of logged queries. These records support distinct rank
bands, although their identities alone do not guarantee distinct synthetic geometry.

Initial candidate geometry uses one common-state record per seed/stream and weight,
not five copies from identical branch starts. Margin is positive similarity minus
candidate similarity; a nonpositive value means the candidate ties or outranks the
positive. It does not establish semantic false-negative status.

| Candidate | Early mean margin | Early nonpositive | Later mean margin | Later nonpositive |
|---|---:|---:|---:|---:|
| Hard synthetic | −0.032721 | 96.35% | −0.020950 | 88.54% |
| Semi-hard synthetic | −0.014145 | 78.39% | +0.001955 | 54.95% |
| Hard real | −0.011925 | 83.59% | −0.008263 | 73.70% |
| Semi-hard real | +0.016716 | 22.40% | +0.025088 | 12.50% |

The semi-hard synthetic construction is easier on average, but its label does not
mean all generated negatives have positive margins or lie near a decision boundary.
At later weights its mean margin is near zero while over half its candidates still
tie or outrank their positives. Generation changes geometry relative to contributors.

## Retrieval and next experimental question

Mean bidirectional reporting-pool R@1 effects, in percentage points:

| Intervention | Weights 100 | Weights 500 |
|---|---:|---:|
| Hard synthetic | 0.000000 | −0.032552 |
| Semi-hard synthetic | −0.065104 | −0.016276 |
| Hard real | −0.097656 | −0.016276 |
| Semi-hard real | −0.048828 | −0.032552 |

None has a positive aggregate R@1 effect at this endpoint; individual seeds are
mixed. Smaller reporting-loss harm is not a retrieval win.

The next discriminating question is whether the relative synthetic benefit survives
**a control for gradient pressure**. A follow-up could cross hard/semi-hard geometry
with fixed strengths chosen from separate training-only calibration, retain native
controls, and monitor pressure throughout continuation. Matching initial pressure
would not guarantee matching subsequent pressure or AdamW updates. Freeze that
design before new outcomes; no new experiment or controller was launched here.

## Verification and provenance

Run `clip_difficulty_20261007T114311_800923Z`; issued bundle
`85abe879f99d6411f710186e35ef404a9708a5e488745e67710fb63615739700`.
All **902 original manifested files** and the pinned backup manifest verify.
All **60 endpoint caches** reproduce full-pool loss and both retrieval directions.
Native trunk and fixed-LR replay records, starts, traces, 360 standalone report
records, 180 pressure records, support audit records and all seed/timepoint
arithmetic pass the CPU audit. Pressure gradients and per-step features were not
retained; their diagnostic records cannot be independently reconstructed from
endpoint caches. Runtime invariant checks are not a new execution of training.

The successful subprocess log ends with `DRIVE_TREE_VERIFIED` and `CHILD_EXIT_CODE: 0`.
The outer cell's flush marker is not in that log; a verified download establishes
local consistency rather than remote flush completion.

The filtered export retains **831 original files**, including the backup manifest,
and omits **72** caption-bearing/binary inputs. Its separately derived
`diagnostics.json` is recomputed by clone-only verification. Issued protocol/source
snapshots retain preparation-time wording and are unchanged. See
[verification commands](evidence-and-reproduction.md).
