# Authenticated v2b gradient audit

PASS on 34 exposures × 3 placement arms. All 306 recorded source/carrier/query VJP checks satisfy their frozen tolerances.
Both forward-reproduction gates were exact; complete state restoration was verified.

| Arm | Raw gradient norm | Raw ratio to native | Cosine to native | Cosine to H-Uniform |
|---|---:|---:|---:|---:|
| H-Uniform | 13.455213 | 73.817% | not recorded | not recorded |
| Wrong | 1.290015 | 7.077% | 0.120526 | 0.863122 |
| Boundary | 1.170060 | 6.419% | 0.120213 | 0.862609 |
| Correct | 1.055963 | 5.793% | 0.119900 | 0.861507 |

Native gradient norm: 18.227872448.

Source/carrier/query maximum recorded relative vector errors are all 0.0, 0.0, 0.0.
Native scale derivatives passed; auxiliary candidate, partition and loss were disconnected from both scale tensors.

These are unweighted gradients on the first continuation batch, not calibrated 2% pressure.
The smaller placed-candidate norms reinforce the need for separate frozen calibration; they do not rank training utility.
Cosines against Native or H-Uniform are not pairwise cosines between the three placement arms.

Audit of saved gradient metrics, not independent gradient recomputation. One uncalibrated starting batch only; no utility or full-population claim.

Next stage: four-batch scheduled pressure calibration under the FP32 head, followed by the full-population audit. No training is authorized by this result.
