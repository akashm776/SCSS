# SCSS Experiment C results — October 8, 2026

**Semi-hard retains a mean later advantage after initial gradient-pressure
calibration, especially at the 7% target. It still does not beat native on average
at either target.** The 2% comparison is mixed across continuations, and the
construction × pressure interaction is more consistent than the weight-state
interaction. This is not evidence for a universal hard-early/semi-hard-later rule.

[Frozen protocol](clip-pressure.md) ·
[Verified evidence](../experiment_results/clip_pressure_2026-10/README.md) ·
[Full comparisons](../experiment_results/clip_pressure_2026-10/comparison.json) ·
[Derived diagnostics](../experiment_results/clip_pressure_2026-10/diagnostics.json) ·
[Experiment B](research-status-2026-10-07-difficulty.md)

## What was tested

Hard synthetic top-8 versus semi-hard synthetic ranks 16–23, each calibrated to
2% and 7% initial mean auxiliary/native gradient-norm ratios. Native and the two
historical-coefficient synthetic arms provide replay controls. Native-only weights
100/500, common update-100 LR trajectory, fresh AdamW moments and counters, three
reused seeds × two streams, 50-update continuations. Total: **5,700 optimizer
updates**, including 1,500 trunk and 4,200 branch updates, across 84 branches.

Four deterministic training-only batches per seed determine each weight-state
coefficient through `alpha = rho / mean_b(||g_aux,b|| / ||g_native,b||)`, with
unweighted auxiliary gradients. The coefficient then stays fixed for both streams
and all 50 updates. Calibration membership is disjoint from continuation batches,
frozen probes and reporting. Outcomes were not used to select coefficients.
Only the initial calibration-set mean is matched, not each batch, parameter group,
AdamW update, clipping scale or future gradient trajectory.

All averages first average the two streams within each seed, then weight the three
seeds equally. Reporting uses the reused 512-image pool. These are not new training
seeds or an untouched final test.

## Primary endpoint and absolute native comparisons

`D = arm − matched native reporting loss` at horizon 50; negative helps.

| Calibrated intervention | Weights 100 | Weights 500 |
|---|---:|---:|
| Hard, 2% target | −0.0009868904 | +0.0003929658 |
| Semi-hard, 2% target | −0.0011100267 | +0.0003236494 |
| Hard, 7% target | −0.0018044836 | +0.0017727088 |
| Semi-hard, 7% target | −0.0023160732 | +0.0010872379 |

All four calibrated interventions help early in all six matched continuations.
All four harm later in their aggregate and each seed's two-stream mean. Late
semi-hard at 2% helps in one individual continuation (seed 161803, stream 2026100101,
`D = −0.0000258746`), but harms in the other five. The other three calibrated
late arms harm in all six. Thus there is no consistent absolute late-state success;
native remains preferable on the mean loss endpoint.

For `Q(w,rho) = D(semi-hard) − D(hard)`:

| Prespecified contrast | Loss difference |
|---|---:|
| Later Q, 2% | −0.0000693164 |
| Later Q, 7% | −0.0006854709 |
| **Primary M_late: mean of the two later Q values** | **−0.0003773937** |
| Construction × pressure: later Q(7%) − Q(2%) | −0.0006161545 |
| Early Q, 2% | −0.0001231363 |
| Early Q, 7% | −0.0005115897 |
| Later Q − early Q, 2% | +0.0000538198 |
| Later Q − early Q, 7% | −0.0001738813 |
| Mean weight-state interaction across targets | −0.0000600307 |

The primary favors semi-hard in **3/3 seed means and 5/6 stream contrasts**.
At 7%, later semi-hard beats hard in **3/3 seed means and 6/6 streams**. At 2%,
it wins in **2/3 seed means and 3/6 streams**. The construction × pressure
interaction is negative in all three seed means and all six stream contrasts:
the semi-hard advantage is larger at the higher target in every matched pair.
These sign counts are descriptive, not a significance or equivalence test.

| Seed (two-stream mean) | Later Q, 2% | Later Q, 7% | Primary M_late | Pressure interaction |
|---|---:|---:|---:|---:|
| 271828 | +0.0000956970 | −0.0003084764 | −0.0001063897 | −0.0004041734 |
| 161803 | −0.0002401631 | −0.0012122614 | −0.0007262123 | −0.0009720983 |
| 141421 | −0.0000634832 | −0.0005356750 | −0.0002995791 | −0.0004721917 |

The low-target weight-state interaction is positive in all three seed means;
the high-target interaction is negative in two and positive in one. Semi-hard is
already better early on average at both targets. The averaged age interaction is
small and mixed across seeds, and cannot support the proposed directional switch.

## Calibration worked; pressure trajectories subsequently diverged

All six weight-state calibrations pass. The largest relative deviation of a
remeasured mean from its target is **0.1377%**, well inside the frozen 5% tolerance.
All four individual ratios, norms and resulting coefficients are retained.
This verifies the recorded arithmetic and tolerance; CPU auditing does not recreate
the training gradients themselves.

The first continuation batch is distinct from the calibration batches. Its mean
ratios are nevertheless very close between constructions. Below are means of the
recorded per-batch ratios at later weights, expressed as percentages:

| Target / construction | Offset 0 | Offset 10 | Offset 25 | Offset 49 |
|---|---:|---:|---:|---:|
| 2%, hard | 1.909% | 2.056% | 1.829% | 1.804% |
| 2%, semi-hard | 1.902% | 2.069% | 1.701% | 1.684% |
| 7%, hard | 6.678% | 6.968% | 5.928% | 5.497% |
| 7%, semi-hard | 6.658% | 6.927% | 5.295% | 4.951% |

Earlier weights also start closely matched: 1.966%/1.985% at the low target and
6.880%/6.947% at the high target (hard/semi-hard). Subsequent semi-hard pressure
falls below hard pressure. These four observations do not measure the complete
50-step pressure integral.

**Gradient clipping is active on all 4,200 continuation updates**, including
native. Average clipping scales range approximately 0.063–0.070. The calibrated
raw norms therefore cannot be interpreted as matched AdamW intervention updates.
Clipping does not by itself explain the contrast; this audit has no causal
ablation of clipping or optimizer preconditioning.

## Continuation horizon and retrieval remain separate

Later semi-minus-hard loss effects at the prespecified reporting offsets:

| Horizon | Q, 2% | Q, 7% | Mean Q |
|---|---:|---:|---:|
| 1 | +0.0000121610 | +0.0000372522 | +0.0000247066 |
| 5 | +0.0000593772 | −0.0000686062 | −0.0000046145 |
| 10 | +0.0000404583 | −0.0001447224 | −0.0000521320 |
| 25 | +0.0001261973 | −0.0001870755 | −0.0000304391 |
| **50** | **−0.0000693164** | **−0.0006854709** | **−0.0003773937** |

At the low target, semi-hard is worse than hard at every shorter reporting horizon
and only overtakes it in the horizon-50 aggregate. At the high target it is worse
after one update and better at the later observations. This reinforces that
one-step utility and sustained intervention utility are different targets.

Horizon-50 mean bidirectional R@1 effects in percentage points:

| Intervention | Weights 100 | Weights 500 |
|---|---:|---:|
| Hard, 2% | −0.016276 | +0.016276 |
| Semi-hard, 2% | −0.016276 | −0.016276 |
| Hard, 7% | −0.032552 | −0.016276 |
| Semi-hard, 7% | −0.032552 | 0.000000 |

A tiny positive aggregate retrieval effect exists for later hard-low, unlike B's
all-nonpositive aggregate endpoint effects. It coexists with worse mean loss and
does not establish a reliable or canonical retrieval improvement.

## What C establishes and what remains open

The later mean semi-hard advantage, particularly at 7%, **survives the frozen
initial-pressure calibration procedure**. The large initial raw gradient-magnitude
difference in B is therefore insufficient as the sole account of that advantage.
However, later pressure drift, clipping, optimizer-conditioned directions and
construction-specific behavior remain possible contributors. C does not isolate
representation geometry or show equal optimization pressure throughout training.

The stronger empirical signal here is construction × strength, with additional
horizon dependence; weight-state differences in *relative preference* are weaker
than in B. Both constructions still show early benefit and later mean harm at
both strengths. A future selector should consider construction, strength and
duration jointly and retain native/abstention as an action. No learned selector,
new dataset/model generalization or final-task improvement was tested here.

## Verification and provenance

Run `clip_pressure_20261007T235549_427102Z`; issued bundle
`79f1ab673de6bc888b8dd85a56c04c01e5d481da4533b33700e1d607012424ce`.
All **1,322 manifested files** were authenticated and all **84 cached endpoints**
were recomputed for native loss and both retrieval directions. The 504 reporting
points, 336 probes, fixed coefficients, native trunks, optimizer-reset/matching
records and all three historical B replay arms were checked. Every seed/timepoint
contrast was independently recalculated. No GPU training was rerun.

The numerical archive retains 1,224 original files and omits 99 caption-bearing
or binary inputs, including NPZ caches. Clone-only checks validate retained bytes,
calibration arithmetic, replay records and comparisons; they cannot recreate
omitted endpoint metrics, training gradients or runtime state invariance. The
log ends with `DRIVE_TREE_VERIFIED` and child exit code 0; the downloaded tree is
complete. This is not independent proof of a remote Drive flush.

Frozen protocols, calibration membership, issued source files and earlier bundles
remain unchanged. Their preparation-time status is superseded by this report.
