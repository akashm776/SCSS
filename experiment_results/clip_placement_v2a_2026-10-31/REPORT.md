# Authenticated v2a paired forward audit

Offline audit of one completed v2a paired record; no new model execution

| Arm | Max geometry error | Max similarity error | Max logit error |
|---|---:|---:|---:|
| Wrong | 7.07805157e-08 | 1.00582838e-07 | 1.01786501e-05 |
| Boundary | 5.96046448e-08 | 5.96046448e-08 | 7.62939453e-06 |
| Correct | 7.07805157e-08 | 7.07805157e-08 | 6.36395282e-06 |

Native loss: 3.08009338379 → 3.07541131973; change -0.0046820640564.

All frozen placement checks passed when independently recomputed from saved observations.
The complete JSON records distributions of native matrix, partition and loss changes.

Does not validate gradients, calibration, full-population feasibility or utility.
