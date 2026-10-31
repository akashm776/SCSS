# D1: frozen source regimes on common eligible exposures

Protocol frozen after completed D0 and before any D1 training outcome. The question is which starting-state source regime provides useful additional negative pressure at native weights100 versus500. Uniform two-source synthesis is a secondary factor. This is not explicit boundary-targeted generation.

## Treatment and population

Three development seeds (271828,161803,141421), two streams (2026100101,2026100102), native checkpoints100/500,64 pairs per batch. Use k=2, the smallest nondegenerate uniform synthesis support. Every exposure is keyed by seed, stream, offset, query row, source image and caption. Four separate calibration batches per seed use the same rule.

The frozen mask is E2_100 intersect E2_500 from D0's strict FP32 margin geometry. It contains10667/19200 continuation exposures (55.5573%) and416/768 calibration exposures (54.1667%). Every audited batch has at least26 eligible rows. Masks and ordered support IDs are bundled in `configs/clip_source_population.json`, derived only from authenticated D0 query records. Training does not recompute eligibility or select IDs. All native checkpoints and processed batches/captions must authenticate against D0. The model's current embeddings are recomputed and differentiated at every update.

H sources are ranks1–2 by descending similarity; M ranks16–17. C sources are the two smallest strictly positive margins; I the two smallest absolute nonpositive margins. Exact ties follow batch index. Select uses the first ordered source, frozen along with both support IDs. Synth uses equal weights on both sources and normalizes their current average. There is no species filter. C/I are initial source labels, not guaranteed current labels or generated placement.

Arms: native, H-Select, H-Synth, M-Select, M-Synth, C-Select, C-Synth, I-Select, I-Synth. Native loss is symmetric CLIP on all64 pairs. The auxiliary is the mean over eligible rows of softplus(extra_logit - native_row_logpartition), with all64 columns retained in the denominator. The auxiliary scale is detached; both native partition and current candidate features retain gradients. The Select candidate already appears in the native denominator and receives extra pressure. Support decisions and equal mixture coefficients are detached. The synthesis implementation uses the D0 reference mixed-precision matrix-product family, restricted to eligible rows; ideal FP32 geometry is diagnostic only.

Empty masks, invalid supports, nonfinite values or mixture norms <=1e-6 cause a recorded failure. No skipping queries, reselection, adaptive k, coefficient fallback or survivor-only analysis. Later side crossings do not invalidate frozen sources and never change treatment.

## Calibration and controls

For every seed/state/auxiliary arm, compute four individual raw auxiliary/native gradient norm ratios at coefficient1 using the exact masked objective. Alpha=.02/mean(ratios). Preserve all weights, gradients, buffers, modes, RNG, optimizer, scheduler and precision backend state. Checkpoint100 supplies common RNG, restored by observational calls. Verify weighted gradients on the same four batches; maximum relative target error5%, inherited from C. Fail on mismatch without tuning. Save individual norms/ratios and verification error. Commit both state calibrations before any continuation for that seed. Coefficients remain fixed and shared across streams.

Each branch starts with identical state-specific weights, fresh AdamW moments/counters and checkpoint100 external LR trajectory/RNG. Preserve native preprocessing, captions, trainable policy, loss, clipping and precision. Native trunks and native continuation points/core traces must replay the authenticated historical controls exactly. Auxiliary arms are new treatments, not B/C replays.

At each real update log frozen exposure hash, support IDs, eligible count, current source/Select/uniform/applied margins, mixture norms, auxiliary logits and source-side retention. At offsets0,10,25,49 record native/weighted auxiliary norms, ratio, cosine, total norm and clipping. Paired virtual native/treatment AdamW steps begin from identical current weights, evolved moments/counters and LR, use actual clipping, weight decay and scale clamp, and restore all state. Report norm(delta_treatment-delta_native)/norm(delta_native); null plus explicit reason if native displacement is zero. These864 virtual calls are diagnostic work, not committed training updates, and never control treatment.

## Outcomes and estimands

Horizon50 is primary; observations0,1,5,10,25,50. Define D=reporting loss minus matched native. Q(w)=D(C-Select)-D(I-Select); primary Delta_side=Q(500)-Q(100). Negative Delta is a relative shift toward C, not necessarily crossover. The directional crossover additionally requires Q100>0 and Q500<0. Actual usefulness requires D<0.

The endpoint is full512 reporting-pool loss: this estimates the effect of applying a masked training intervention, not reporting loss restricted to eligible queries. Shared exposure masks control query enrollment across states; state-specific support identities intentionally differ. This remains a comparison of starting-state-associated treatments, not a pure representation-only mechanism. Initial pressure calibration does not equalize future pressure, clipping or AdamW displacement.

Secondary G_R=D(R-Synth)-D(R-Select), its state interaction, H versus M/I/C, all absolute D, earlier horizons and bidirectional mean R@1. G compares a representation from two sources against extra pressure on one specified real representative; it does not establish intrinsic synthetic superiority. Average streams within each seed, then the three seeds equally; preserve all paired and per-seed contrasts. No significance/equivalence or independent-confirmation claim on reused development seeds/reporting data.

## Budget and persistence

108 branches x50=5400 committed branch updates; native preparation adds1500 (6900 total).648 reporting points,108 endpoint embedding caches and432 pressure/optimizer probes. Calibration commits zero optimizer steps. No GPU run is performed by local implementation or packaging.

All D0 sources/bundles and frozen evidence remain unchanged. D1 has its own source overlay, protocol/population checksum freeze, logs and Drive run folder. Save each completed real update, reporting point, probe and branch with synchronous checksummed backup. Do not average an incomplete grid or silently restart an interrupted run. Explicit recovery requires inspecting saved artifacts; no model checkpoints are persisted by this version.
