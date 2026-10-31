# D2-P v1 first-batch failure audit — 2026-10-18

## Question

Which numerical path caused the initial placement failure, and can changing only
the synthetic target satisfy the frozen similarity and logit contracts?

Input: downloaded `clip_placement_preflight_v1_a0267d8132ca`, attempt
`20261018T190106_037877Z_61257397`. Seed271828, stream2026100101, native weight100,
batch0, 34 eligible query exposures. No new model forward, gradient probe or
training was run for this analysis. The issued D2-P v1 code/configuration remains
unchanged.

## Authentication and validation

The downloaded ZIP matches all nine extracted files byte-for-byte. `source.zip`
is an unchanged copy. Acceptance's run-configuration and check-index hashes match;
the index's check-record hash matches. Before/after state fingerprints in that
record are equal and preservation is true. `audit.json` contains input hashes,
per-query decompositions and numerical summaries.

The original run stopped after its first Wrong similarity assertion. The extra
arm/logit findings below are **offline calculations from saved actual tensors**,
not additional completed preflight checks. Gradient checks and calibration were
not reached. Zero optimizer updates; D2 not launched.

## Result: accurate candidate placement, inconsistent positive references

Frozen delta = 0.0010687001049518585; scale = 99.99642944335938.
Similarity error allowance = 0.00010687001049518585.
Logit error allowance = 0.010686619464092928.

All three candidates are recorded as **FP32**, not bf16. Maximum FP32 geometry
error is 7.08e-8; the auxiliary similarity differs from its reconstructed FP32
geometric similarity by at most 5.96e-8. Thus casting the candidate to FP32 would
not address this observed failure: it is already FP32.

The native raw positive values are exactly bf16-representable. Every native
positive logit equals the bf16-rounded product of its raw positive and scale.
Every auxiliary logit equals the FP32-rounded product of its auxiliary similarity
and the same scale. These arithmetic fingerprints also match the implementation's
autocast native matrix-product path and FP32 elementwise auxiliary path.

Let `d_positive = p_sim - p_geom`. Its range is
[-0.001146763563156128, 0.0010015666484832764], with median absolute value
0.0005537271499633789. For all three arms:

`r_train - r_target = d_positive + remainder`

The largest absolute remainder across all arms/queries is **1.01e-7**. The native
positive/reference mismatch therefore explains essentially the entire placement
error. This includes the difference between native scoring and the renormalized
FP32 geometric reference; the saved scalars do not further separate input rounding,
output rounding and renormalization contributions inside that native difference.

| Arm | Similarity-tolerance failures | Logit-tolerance failures | Actual similarity wrong sign | Actual logit wrong sign |
|---|---:|---:|---|---|
| Wrong | 30/34 | 30/34 | none | query30 |
| Boundary | 30/34 | 30/34 | not applicable | not applicable |
| Correct | 30/34 | 30/34 | query6 | queries1,4,6 |

Boundary's actual similarity margins range from -0.00114676 to +0.00100154;
its actual logit gaps range from -0.160612 to +0.132376. It is not numerically a tie
within the frozen tolerances for 30/34 exposures.

The three candidates **do retain their intended relative spacing**: both adjacent
similarity separations equal delta within 7.08e-8. The triplet is accurately spaced
but centered on a different positive reference from the native training scores.

## Stronger constraint: changing the target alone is insufficient

Write `r = p_sim - s_aux` and `g = ell_plus - e_aux`. With the current FP32
auxiliary scaling, up to final rounding:

`g - tau*r = ell_plus - tau*p_sim = k`.

To satisfy both frozen target contracts requires:

`abs(k) <= tau*eps_similarity + eps_logit = 0.021373238928185857`.

The native scaling residual k ranges from -0.06155157 to +0.05946198.
For **24/34 exposures**, it exceeds that necessary bound even after allowing a
conservative 6.10e-5 FP32 roundoff guard. The violation is independent of which
of the three target margins is requested. The similarity-acceptable and
logit-acceptable candidate intervals do not overlap for those exposures under
the existing scaling path.

Consequently, simply recentering the target on `p_sim`, keeping the present
native logits, cannot satisfy both frozen contracts on this batch. This is not
just a failure to find the right target vector. Widening tolerances or increasing
delta would change the protocol and must not be used to erase v1's failure.

## Structural update and next smallest test

D2-P v1 fails its prescribed numerical execution contract on the first batch.
This is evidence about that implementation/gap/precision combination, not evidence
that pairwise placement is useless or that straight-through gradients fail.

The next useful test is a separately versioned, zero-update diagnostic on this
same authenticated batch: use a consistent FP32 similarity/logit head for both
native and auxiliary paths while retaining mixed-precision encoder computation,
the same source IDs and gap; verify both contracts and the surrogate gradients.
That is a proposed protocol change, **not implemented or run in this audit**.
It would alter native numerical training, so future D2 Native and H-Uniform
controls would need to use that same path; archived D1 outcomes could no longer
be called exact replay controls for it. A logit-targeted policy retaining the old
native head is a different alternative, requiring revised cosine-target claims.

Reproduce the scalar audit:

```sh
.venv/bin/python scripts/audit_clip_placement_failure.py \
  '/Users/akashmittal/Downloads/clip_placement_preflight_v1_a0267d8132ca' \
  --output /tmp/d2p-first-batch-audit.json
```

The archived audit additionally records the ZIP checksum and byte-comparison count.
