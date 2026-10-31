# 4. Experimental Setup

[Previous](formulation.md) · [Contents](../../README.md#contents) · [Next](results.md)

The table describes the completed D1 source-regime study. Earlier experiments deliberately vary parts of this setup; their results are interpreted within their own protocols.

| Component | D1 setup |
|---|---|
| Model | Pretrained CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Data | CUB image–caption pairs; frozen processed batch/caption identities |
| Trainable scope | Restricted fine-tuning policy; 10,895,617 parameters across 35 tensors |
| Starting weights | Native-only checkpoints at updates 100 and 500 |
| Continuation | 50 updates; 64 image–text pairs per training batch |
| Replication | Three training seeds; two continuation streams within each seed |
| Branch optimizer | Fresh AdamW state, including reset moments and step counters |
| Schedule | Common update-100 learning-rate trajectory at both weight states |
| Arms | Native plus Select/Synth for each of H, M, C, I |
| Auxiliary strength | Separate per-seed/state/arm coefficients; 2% initial scheduled raw-gradient target |
| Primary endpoint | Full 512-image reporting-pool loss after 50 continuation updates |
| Secondary endpoints | Mean bidirectional R@1 and horizons 0, 1, 5, 10, 25 |
| Budget | 108 branches; 5,400 continuation + 1,500 native preparation updates |

Full settings and the exact trainable policy are retained with the [frozen source-regime protocol](../clip-source-regime.md) and its input records. Generic package installation is not a replacement for the historical GPU environment.

## Matched comparisons

Within a matched comparison, branches share starting weights, continuation images/captions, stream plan, and controlled random state. D1 uses fresh branch optimizer state to avoid attributing an inherited moment-state difference to source construction. These controls narrow the intervention comparison; they do not separate every effect of representation, source geometry, and optimization.

## Eligibility and data roles

D0 established that some queries have no misranked source. D1 therefore uses the intersection of two-source C/I eligibility across both starting states. The frozen population includes **10,667 of 19,200 continuation exposures (55.56%)** and **416 of 768 calibration exposures (54.17%)**. Repeated appearances are exposure instances, not additional independent examples or seeds.

Native training uses every pair. The auxiliary loss alone is masked. Outcomes are measured on the full reporting pool, so the estimand is the effect of a masked training intervention on that pool, not performance restricted to treated queries.

Calibration batches are separated from continuation, reporting, and probe membership under the frozen plan. Reporting images are excluded from the recent training runs, but the reporting data and development seeds have been reused to guide this research. They are **not an untouched final confirmation set**.

## Aggregation and uncertainty

We first average the two continuation streams within each seed, then weight the three seeds equally. Seed means and matched stream contrasts remain available. Neither the six streams nor thousands of query exposures constitute thousands of independent replications.

The report presents descriptive effects and seed heterogeneity. It does not report invented confidence intervals, claim equivalence from a near-zero mean, or substitute a favorable earlier horizon for the prespecified endpoint. The [detailed results](detailed-results.md) show the primary interaction by seed.

## Metric conventions

Loss differences are treatment minus matched native; negative helps. R@1 changes are percentage points; positive helps. Reporting-pool retrieval, full canonical retrieval, and species classification are different endpoints with different populations. Historical studies with different denominators or horizons are not combined into a single performance ranking.

## Numerical regimes

D1 retains its issued mixed-precision computation. D2's placement checks subsequently motivated a consistent FP32 scoring/loss head while preserving the mixed-precision encoders and original carrier. That is a new numerical training regime. D1 outcomes are historical references for it, not interchangeable Native/H-Uniform controls. The D2 calibration stage recalibrates H-Uniform accordingly.

---

[Previous](formulation.md) · [Contents](../../README.md#contents) · [Next](results.md)
