# Authenticated v2c calibration audit

PASS: all six seed/state cells, 24 coefficients, 24 scheduled batches and 24 weighted-gradient replays.
Eligibility: 832 state/exposure instances, 2496 arm/exposure checks; 0 empty batches.

Verified scheduled pressure range: 1.996835%–2.001782%.
Maximum relative error against the 2% target: 0.158268% (allowed 5%).
Individual batch pressures: 1.6313%–2.7054%.

| Seed | State | H-Uniform alpha | Wrong alpha | Boundary alpha | Correct alpha |
|---|---:|---:|---:|---:|---:|
| 141421 | 100 | 0.02245437 | 0.24430497 | 0.27019267 | 0.29907287 |
| 141421 | 500 | 0.02572129 | 0.18747578 | 0.20640975 | 0.22777773 |
| 161803 | 100 | 0.02094157 | 0.24379472 | 0.26925192 | 0.29780321 |
| 161803 | 500 | 0.02393817 | 0.20589223 | 0.22739980 | 0.25134731 |
| 271828 | 100 | 0.01990590 | 0.25116004 | 0.27772088 | 0.30714553 |
| 271828 | 500 | 0.02526097 | 0.19869922 | 0.21917863 | 0.24161578 |

Maximum errors: ideal margin 7.07805157e-08; actual similarity margin 1.34110451e-07; actual logit gap 1.39933474e-05.
Weighted replay preserved native norms/losses and placement observables exactly. All 48 state-restoration records match. Auxiliary scale disconnection passed.

The 2% target is a four-scheduled-batch average, not a per-batch equality or an optimizer-displacement match. All batches were treated, so conditional and scheduled means coincide.
Larger placement-arm coefficients compensate for smaller raw gradients; they do not establish greater training utility. Keep these coefficients frozen for the next stage.

Authentication and recomputation from saved metrics; no independent GPU-gradient recomputation. Calibration batches only; no training utility or full-population acceptance.

Next: full-population D2-P v2 audit. Zero optimizer updates; D2 training remains gated.
