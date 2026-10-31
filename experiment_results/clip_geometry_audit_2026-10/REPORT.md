# D0 geometry and feasibility audit

Feasibility/separability only. Conditional distributions are pooled query exposures or labelled source members. Macro means require every expected stratum. No k or arm chosen, no training-utility result. Primary is a subset of extended. Three reused seeds, not independent query replications.

Audit: 624 fixed-state batch forwards; 39936 query exposures; zero audit updates.

All-query C/I sizes passing structural and both-path numerical feasibility: [].
D1 remains unfrozen. Checkpoint preparation is recorded separately in completion.json.

Tables show pooled statistics and available/total denominators. Source margins pool source members; real/synthetic margins, species fractions and overlaps use query exposures. Undefined entries are never zero. Complete all-k, seed, stream and common-population distributions and macro means are in summary.json; per-batch summaries accompany each query shard. k=8 is a descriptive view, not a selected D1 design.

## calibration_feasibility: weights 100

### Feasibility

Zero nonpositive candidates: 0.192708 (768/768).

| k | Positive availability | Nonpositive availability | Common C/I eligibility |
|---:|---:|---:|---:|
| 1 | 1 (768/768) | 0.807292 (768/768) | 0.807292 (768/768) |
| 2 | 0.998698 (768/768) | 0.69401 (768/768) | 0.692708 (768/768) |
| 3 | 0.998698 (768/768) | 0.622396 (768/768) | 0.621094 (768/768) |
| 4 | 0.997396 (768/768) | 0.549479 (768/768) | 0.546875 (768/768) |
| 5 | 0.997396 (768/768) | 0.486979 (768/768) | 0.484375 (768/768) |
| 6 | 0.996094 (768/768) | 0.434896 (768/768) | 0.43099 (768/768) |
| 7 | 0.996094 (768/768) | 0.398438 (768/768) | 0.394531 (768/768) |
| 8 | 0.994792 (768/768) | 0.378906 (768/768) | 0.373698 (768/768) |

### Geometry at k=8

Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.

| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |
|---|---:|---:|---:|---:|
| H | -0.00187778 (6144/6144) | -0.0113551 (768/768) | -0.0342095 (768/768) | -0.0341216 (768/768) |
| M | 0.0169898 (6144/6144) | 0.0137594 (768/768) | -0.0175091 (768/768) | -0.017537 (768/768) |
| C | 0.00625829 (6112/6144) | 0.00143978 (764/768) | -0.0266156 (764/768) | -0.0264839 (764/768) |
| I | -0.00374699 (2328/6144) | -0.000616014 (291/768) | -0.0384284 (291/768) | -0.0384752 (291/768) |

### Separability at k=8

| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |
|---|---:|---:|---:|---:|
| HM | 0 (768/768) | 0 (768/768) | 0 (768/768) | 0 (768/768) |
| HC | 0.455825 (764/768) | 0.193717 (764/768) | 0.193717 (764/768) | 0.193717 (764/768) |
| HI | 0.370704 (291/768) | 0.140893 (291/768) | 0 (291/768) | 0 (291/768) |
| MC | 0.133671 (764/768) | 0.013089 (764/768) | 0.013089 (764/768) | 0.013089 (764/768) |
| MI | 0.160653 (291/768) | 0.0309278 (291/768) | 0.0378007 (291/768) | 0 (291/768) |
| CI | 0 (287/768) | 0 (287/768) | 0 (287/768) | 0 (287/768) |

Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; equal support sets alone need not give equal Select examples. Historical same-forward support equality: H: 0.703125 (768/768), M: 0.436198 (768/768).

### Semantics

Same-species associations are descriptive, not verified false-negative labels.

| Population | Same-species fraction |
|---|---:|
| Candidate members: positive | 0.00378427 (42016/42016) |
| Candidate members: nonpositive | 0.0152324 (6368/6368) |
| Candidate members: negative | 0.0152324 (6368/6368) |
| Candidate members: zero | undefined (0/0) |
| H support (k=8) | 0.0208333 (768/768) |
| H Select (k=8) | 0.0377604 (768/768) |
| M support (k=8) | 0.00439453 (768/768) |
| M Select (k=8) | 0.00260417 (768/768) |
| C support (k=8) | 0.0145615 (764/768) |
| C Select (k=8) | 0.0405759 (764/768) |
| I support (k=8) | 0.012457 (291/768) |
| I Select (k=8) | 0.0103093 (291/768) |

## calibration_feasibility: weights 500

### Feasibility

Zero nonpositive candidates: 0.269531 (768/768).

| k | Positive availability | Nonpositive availability | Common C/I eligibility |
|---:|---:|---:|---:|
| 1 | 1 (768/768) | 0.730469 (768/768) | 0.730469 (768/768) |
| 2 | 1 (768/768) | 0.583333 (768/768) | 0.583333 (768/768) |
| 3 | 1 (768/768) | 0.5 (768/768) | 0.5 (768/768) |
| 4 | 1 (768/768) | 0.429688 (768/768) | 0.429688 (768/768) |
| 5 | 1 (768/768) | 0.376302 (768/768) | 0.376302 (768/768) |
| 6 | 1 (768/768) | 0.328125 (768/768) | 0.328125 (768/768) |
| 7 | 1 (768/768) | 0.28776 (768/768) | 0.28776 (768/768) |
| 8 | 1 (768/768) | 0.248698 (768/768) | 0.248698 (768/768) |

### Geometry at k=8

Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.

| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |
|---|---:|---:|---:|---:|
| H | 0.00258722 (6144/6144) | -0.00858623 (768/768) | -0.0219145 (768/768) | -0.0218815 (768/768) |
| M | 0.0252264 (6144/6144) | 0.0210045 (768/768) | -0.0027671 (768/768) | -0.00275159 (768/768) |
| C | 0.00923093 (6144/6144) | 0.00217564 (768/768) | -0.0167138 (768/768) | -0.0166576 (768/768) |
| I | -0.00402144 (1528/6144) | -0.000666261 (191/768) | -0.0330899 (191/768) | -0.0329522 (191/768) |

### Separability at k=8

| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |
|---|---:|---:|---:|---:|
| HM | 0 (768/768) | 0 (768/768) | 0 (768/768) | 0 (768/768) |
| HC | 0.564453 (768/768) | 0.269531 (768/768) | 0.269531 (768/768) | 0.269531 (768/768) |
| HI | 0.347513 (191/768) | 0.0837696 (191/768) | 0 (191/768) | 0 (191/768) |
| MC | 0.100911 (768/768) | 0.0195312 (768/768) | 0.0195312 (768/768) | 0.0195312 (768/768) |
| MI | 0.174738 (191/768) | 0.0209424 (191/768) | 0.0157068 (191/768) | 0 (191/768) |
| CI | 0 (191/768) | 0 (191/768) | 0 (191/768) | 0 (191/768) |

Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; equal support sets alone need not give equal Select examples. Historical same-forward support equality: H: 0.838542 (768/768), M: 0.605469 (768/768).

### Semantics

Same-species associations are descriptive, not verified false-negative labels.

| Population | Same-species fraction |
|---|---:|
| Candidate members: positive | 0.00375572 (43933/43933) |
| Candidate members: nonpositive | 0.0204448 (4451/4451) |
| Candidate members: negative | 0.0204448 (4451/4451) |
| Candidate members: zero | undefined (0/0) |
| H support (k=8) | 0.0262044 (768/768) |
| H Select (k=8) | 0.0572917 (768/768) |
| M support (k=8) | 0.00439453 (768/768) |
| M Select (k=8) | 0.0078125 (768/768) |
| C support (k=8) | 0.0187174 (768/768) |
| C Select (k=8) | 0.0481771 (768/768) |
| I support (k=8) | 0.0111257 (191/768) |
| I Select (k=8) | 0.0104712 (191/768) |

## calibration_feasibility: weights paired

Eligibility transitions on matched query exposures; entries are fractions (available/total).

| k | Both | Early only | Late only | Neither |
|---:|---:|---:|---:|---:|
| 1 | 0.692708 (768/768) | 0.114583 (768/768) | 0.0377604 (768/768) | 0.154948 (768/768) |
| 2 | 0.541667 (768/768) | 0.151042 (768/768) | 0.0416667 (768/768) | 0.265625 (768/768) |
| 3 | 0.464844 (768/768) | 0.15625 (768/768) | 0.0351562 (768/768) | 0.34375 (768/768) |
| 4 | 0.39974 (768/768) | 0.147135 (768/768) | 0.0299479 (768/768) | 0.423177 (768/768) |
| 5 | 0.342448 (768/768) | 0.141927 (768/768) | 0.0338542 (768/768) | 0.481771 (768/768) |
| 6 | 0.294271 (768/768) | 0.136719 (768/768) | 0.0338542 (768/768) | 0.535156 (768/768) |
| 7 | 0.25651 (768/768) | 0.138021 (768/768) | 0.03125 (768/768) | 0.574219 (768/768) |
| 8 | 0.221354 (768/768) | 0.152344 (768/768) | 0.0273438 (768/768) | 0.598958 (768/768) |

## extended: weights 100

### Feasibility

Zero nonpositive candidates: 0.191875 (19200/19200).

| k | Positive availability | Nonpositive availability | Common C/I eligibility |
|---:|---:|---:|---:|
| 1 | 0.999271 (19200/19200) | 0.808125 (19200/19200) | 0.807396 (19200/19200) |
| 2 | 0.99875 (19200/19200) | 0.692917 (19200/19200) | 0.691667 (19200/19200) |
| 3 | 0.998073 (19200/19200) | 0.609948 (19200/19200) | 0.608021 (19200/19200) |
| 4 | 0.997656 (19200/19200) | 0.543542 (19200/19200) | 0.541198 (19200/19200) |
| 5 | 0.996927 (19200/19200) | 0.488229 (19200/19200) | 0.485156 (19200/19200) |
| 6 | 0.996094 (19200/19200) | 0.440312 (19200/19200) | 0.436406 (19200/19200) |
| 7 | 0.995156 (19200/19200) | 0.402187 (19200/19200) | 0.397344 (19200/19200) |
| 8 | 0.994479 (19200/19200) | 0.36651 (19200/19200) | 0.36099 (19200/19200) |

### Geometry at k=8

Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.

| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |
|---|---:|---:|---:|---:|
| H | -0.00166604 (153600/153600) | -0.0118988 (19200/19200) | -0.0338952 (19200/19200) | -0.0338925 (19200/19200) |
| M | 0.0164667 (153600/153600) | 0.0135065 (19200/19200) | -0.0177427 (19200/19200) | -0.0177165 (19200/19200) |
| C | 0.00617604 (152752/153600) | 0.0013615 (19094/19200) | -0.0260799 (19094/19200) | -0.0260747 (19094/19200) |
| I | -0.0032847 (56296/153600) | -0.000534803 (7037/19200) | -0.0378465 (7037/19200) | -0.0378281 (7037/19200) |

### Separability at k=8

| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |
|---|---:|---:|---:|---:|
| HM | 0 (19200/19200) | 0 (19200/19200) | 0 (19200/19200) | 0 (19200/19200) |
| HC | 0.45856 (19094/19200) | 0.19294 (19094/19200) | 0.19294 (19094/19200) | 0.19294 (19094/19200) |
| HI | 0.29892 (7037/19200) | 0.084411 (7037/19200) | 0 (7037/19200) | 0 (7037/19200) |
| MC | 0.127664 (19094/19200) | 0.0155023 (19094/19200) | 0.0155023 (19094/19200) | 0.0155023 (19094/19200) |
| MI | 0.184187 (7037/19200) | 0.0237317 (7037/19200) | 0.0419213 (7037/19200) | 0 (7037/19200) |
| CI | 0 (6931/19200) | 0 (6931/19200) | 0 (6931/19200) | 0 (6931/19200) |

Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; equal support sets alone need not give equal Select examples. Historical same-forward support equality: H: 0.681042 (19200/19200), M: 0.400781 (19200/19200).

### Semantics

Same-species associations are descriptive, not verified false-negative labels.

| Population | Same-species fraction |
|---|---:|
| Candidate members: positive | 0.00356973 (1041535/1041535) |
| Candidate members: nonpositive | 0.0136138 (168065/168065) |
| Candidate members: negative | 0.0136138 (168065/168065) |
| Candidate members: zero | undefined (0/0) |
| H support (k=8) | 0.0202279 (19200/19200) |
| H Select (k=8) | 0.0415625 (19200/19200) |
| M support (k=8) | 0.00450521 (19200/19200) |
| M Select (k=8) | 0.0053125 (19200/19200) |
| C support (k=8) | 0.0136037 (19094/19200) |
| C Select (k=8) | 0.0294333 (19094/19200) |
| I support (k=8) | 0.0109955 (7037/19200) |
| I Select (k=8) | 0.0108001 (7037/19200) |

## extended: weights 500

### Feasibility

Zero nonpositive candidates: 0.269219 (19200/19200).

| k | Positive availability | Nonpositive availability | Common C/I eligibility |
|---:|---:|---:|---:|
| 1 | 1 (19200/19200) | 0.730781 (19200/19200) | 0.730781 (19200/19200) |
| 2 | 0.999844 (19200/19200) | 0.59125 (19200/19200) | 0.591094 (19200/19200) |
| 3 | 0.999687 (19200/19200) | 0.495312 (19200/19200) | 0.495 (19200/19200) |
| 4 | 0.999583 (19200/19200) | 0.424063 (19200/19200) | 0.423646 (19200/19200) |
| 5 | 0.999427 (19200/19200) | 0.369688 (19200/19200) | 0.369115 (19200/19200) |
| 6 | 0.999427 (19200/19200) | 0.325677 (19200/19200) | 0.325104 (19200/19200) |
| 7 | 0.999167 (19200/19200) | 0.288177 (19200/19200) | 0.287344 (19200/19200) |
| 8 | 0.998802 (19200/19200) | 0.255417 (19200/19200) | 0.254219 (19200/19200) |

### Geometry at k=8

Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.

| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |
|---|---:|---:|---:|---:|
| H | 0.00251988 (153600/153600) | -0.00810441 (19200/19200) | -0.02226 (19200/19200) | -0.0222615 (19200/19200) |
| M | 0.0245302 (153600/153600) | 0.0207981 (19200/19200) | -0.00297752 (19200/19200) | -0.00298864 (19200/19200) |
| C | 0.00857889 (153416/153600) | 0.00201061 (19177/19200) | -0.01646 (19177/19200) | -0.0164628 (19177/19200) |
| I | -0.00389503 (39232/153600) | -0.000634469 (4904/19200) | -0.0313157 (4904/19200) | -0.0313442 (4904/19200) |

### Separability at k=8

| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |
|---|---:|---:|---:|---:|
| HM | 0 (19200/19200) | 0 (19200/19200) | 0 (19200/19200) | 0 (19200/19200) |
| HC | 0.565632 (19177/19200) | 0.269542 (19177/19200) | 0.269542 (19177/19200) | 0.269542 (19177/19200) |
| HI | 0.358024 (4904/19200) | 0.103589 (4904/19200) | 0 (4904/19200) | 0 (4904/19200) |
| MC | 0.101606 (19177/19200) | 0.0122021 (19177/19200) | 0.0122021 (19177/19200) | 0.0122021 (19177/19200) |
| MI | 0.164152 (4904/19200) | 0.0205954 (4904/19200) | 0.0440457 (4904/19200) | 0 (4904/19200) |
| CI | 0 (4881/19200) | 0 (4881/19200) | 0 (4881/19200) | 0 (4881/19200) |

Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; equal support sets alone need not give equal Select examples. Historical same-forward support equality: H: 0.830104 (19200/19200), M: 0.588698 (19200/19200).

### Semantics

Same-species associations are descriptive, not verified false-negative labels.

| Population | Same-species fraction |
|---|---:|
| Candidate members: positive | 0.00366843 (1096655/1096655) |
| Candidate members: nonpositive | 0.0175572 (112945/112945) |
| Candidate members: negative | 0.0175572 (112945/112945) |
| Candidate members: zero | undefined (0/0) |
| H support (k=8) | 0.0229492 (19200/19200) |
| H Select (k=8) | 0.0534375 (19200/19200) |
| M support (k=8) | 0.00390625 (19200/19200) |
| M Select (k=8) | 0.005 (19200/19200) |
| C support (k=8) | 0.0168952 (19177/19200) |
| C Select (k=8) | 0.0392658 (19177/19200) |
| I support (k=8) | 0.0128721 (4904/19200) |
| I Select (k=8) | 0.0124388 (4904/19200) |

## extended: weights paired

Eligibility transitions on matched query exposures; entries are fractions (available/total).

| k | Both | Early only | Late only | Neither |
|---:|---:|---:|---:|---:|
| 1 | 0.69724 (19200/19200) | 0.110156 (19200/19200) | 0.0335417 (19200/19200) | 0.159062 (19200/19200) |
| 2 | 0.555573 (19200/19200) | 0.136094 (19200/19200) | 0.0355208 (19200/19200) | 0.272813 (19200/19200) |
| 3 | 0.459844 (19200/19200) | 0.148177 (19200/19200) | 0.0351562 (19200/19200) | 0.356823 (19200/19200) |
| 4 | 0.389271 (19200/19200) | 0.151927 (19200/19200) | 0.034375 (19200/19200) | 0.424427 (19200/19200) |
| 5 | 0.334948 (19200/19200) | 0.150208 (19200/19200) | 0.0341667 (19200/19200) | 0.480677 (19200/19200) |
| 6 | 0.290938 (19200/19200) | 0.145469 (19200/19200) | 0.0341667 (19200/19200) | 0.529427 (19200/19200) |
| 7 | 0.255833 (19200/19200) | 0.14151 (19200/19200) | 0.0315104 (19200/19200) | 0.571146 (19200/19200) |
| 8 | 0.224688 (19200/19200) | 0.136302 (19200/19200) | 0.0295312 (19200/19200) | 0.609479 (19200/19200) |

## primary: weights 100

### Feasibility

Zero nonpositive candidates: 0.1875 (384/384).

| k | Positive availability | Nonpositive availability | Common C/I eligibility |
|---:|---:|---:|---:|
| 1 | 1 (384/384) | 0.8125 (384/384) | 0.8125 (384/384) |
| 2 | 1 (384/384) | 0.705729 (384/384) | 0.705729 (384/384) |
| 3 | 1 (384/384) | 0.601562 (384/384) | 0.601562 (384/384) |
| 4 | 0.997396 (384/384) | 0.544271 (384/384) | 0.541667 (384/384) |
| 5 | 0.994792 (384/384) | 0.484375 (384/384) | 0.479167 (384/384) |
| 6 | 0.994792 (384/384) | 0.421875 (384/384) | 0.416667 (384/384) |
| 7 | 0.994792 (384/384) | 0.403646 (384/384) | 0.398438 (384/384) |
| 8 | 0.994792 (384/384) | 0.356771 (384/384) | 0.351562 (384/384) |

### Geometry at k=8

Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.

| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |
|---|---:|---:|---:|---:|
| H | -0.00132202 (3072/3072) | -0.0107903 (384/384) | -0.0325505 (384/384) | -0.0325299 (384/384) |
| M | 0.0162774 (3072/3072) | 0.0137398 (384/384) | -0.018095 (384/384) | -0.0180611 (384/384) |
| C | 0.00624201 (3056/3072) | 0.00129192 (382/384) | -0.0252025 (382/384) | -0.0252773 (382/384) |
| I | -0.00317715 (1096/3072) | -0.000393271 (137/384) | -0.0365544 (137/384) | -0.0363423 (137/384) |

### Separability at k=8

| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |
|---|---:|---:|---:|---:|
| HM | 0 (384/384) | 0 (384/384) | 0 (384/384) | 0 (384/384) |
| HC | 0.46106 (382/384) | 0.188482 (382/384) | 0.188482 (382/384) | 0.188482 (382/384) |
| HI | 0.244526 (137/384) | 0.080292 (137/384) | 0 (137/384) | 0 (137/384) |
| MC | 0.112565 (382/384) | 0.0104712 (382/384) | 0.0104712 (382/384) | 0.0104712 (382/384) |
| MI | 0.205292 (137/384) | 0.0291971 (137/384) | 0.0291971 (137/384) | 0 (137/384) |
| CI | 0 (135/384) | 0 (135/384) | 0 (135/384) | 0 (135/384) |

Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; equal support sets alone need not give equal Select examples. Historical same-forward support equality: H: 0.692708 (384/384), M: 0.403646 (384/384).

### Semantics

Same-species associations are descriptive, not verified false-negative labels.

| Population | Same-species fraction |
|---|---:|
| Candidate members: positive | 0.00329473 (20639/20639) |
| Candidate members: nonpositive | 0.011821 (3553/3553) |
| Candidate members: negative | 0.011821 (3553/3553) |
| Candidate members: zero | undefined (0/0) |
| H support (k=8) | 0.0185547 (384/384) |
| H Select (k=8) | 0.0286458 (384/384) |
| M support (k=8) | 0.00358073 (384/384) |
| M Select (k=8) | 0.00260417 (384/384) |
| C support (k=8) | 0.0127618 (382/384) |
| C Select (k=8) | 0.0287958 (382/384) |
| I support (k=8) | 0.0100365 (137/384) |
| I Select (k=8) | 0.00729927 (137/384) |

## primary: weights 500

### Feasibility

Zero nonpositive candidates: 0.289062 (384/384).

| k | Positive availability | Nonpositive availability | Common C/I eligibility |
|---:|---:|---:|---:|
| 1 | 1 (384/384) | 0.710938 (384/384) | 0.710938 (384/384) |
| 2 | 1 (384/384) | 0.606771 (384/384) | 0.606771 (384/384) |
| 3 | 1 (384/384) | 0.481771 (384/384) | 0.481771 (384/384) |
| 4 | 1 (384/384) | 0.403646 (384/384) | 0.403646 (384/384) |
| 5 | 1 (384/384) | 0.330729 (384/384) | 0.330729 (384/384) |
| 6 | 1 (384/384) | 0.307292 (384/384) | 0.307292 (384/384) |
| 7 | 1 (384/384) | 0.286458 (384/384) | 0.286458 (384/384) |
| 8 | 1 (384/384) | 0.252604 (384/384) | 0.252604 (384/384) |

### Geometry at k=8

Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.

| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |
|---|---:|---:|---:|---:|
| H | 0.00301229 (3072/3072) | -0.00779813 (384/384) | -0.0216363 (384/384) | -0.0215874 (384/384) |
| M | 0.0240734 (3072/3072) | 0.020156 (384/384) | -0.00302891 (384/384) | -0.00307629 (384/384) |
| C | 0.0085529 (3072/3072) | 0.0018719 (384/384) | -0.0163321 (384/384) | -0.0165434 (384/384) |
| I | -0.00390849 (776/3072) | -0.000532538 (97/384) | -0.0301356 (97/384) | -0.0300879 (97/384) |

### Separability at k=8

| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |
|---|---:|---:|---:|---:|
| HM | 0 (384/384) | 0 (384/384) | 0 (384/384) | 0 (384/384) |
| HC | 0.577474 (384/384) | 0.289062 (384/384) | 0.289062 (384/384) | 0.289062 (384/384) |
| HI | 0.30799 (97/384) | 0.0824742 (97/384) | 0 (97/384) | 0 (97/384) |
| MC | 0.100911 (384/384) | 0.0182292 (384/384) | 0.0182292 (384/384) | 0.0182292 (384/384) |
| MI | 0.150773 (97/384) | 0.0103093 (97/384) | 0.0412371 (97/384) | 0 (97/384) |
| CI | 0 (97/384) | 0 (97/384) | 0 (97/384) | 0 (97/384) |

Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; equal support sets alone need not give equal Select examples. Historical same-forward support equality: H: 0.809896 (384/384), M: 0.585938 (384/384).

### Semantics

Same-species associations are descriptive, not verified false-negative labels.

| Population | Same-species fraction |
|---|---:|
| Candidate members: positive | 0.00388465 (21881/21881) |
| Candidate members: nonpositive | 0.0108178 (2311/2311) |
| Candidate members: negative | 0.0108178 (2311/2311) |
| Candidate members: zero | undefined (0/0) |
| H support (k=8) | 0.0179036 (384/384) |
| H Select (k=8) | 0.0390625 (384/384) |
| M support (k=8) | 0.0016276 (384/384) |
| M Select (k=8) | 0.00260417 (384/384) |
| C support (k=8) | 0.0175781 (384/384) |
| C Select (k=8) | 0.0416667 (384/384) |
| I support (k=8) | 0.0115979 (97/384) |
| I Select (k=8) | 0.0103093 (97/384) |

## primary: weights paired

Eligibility transitions on matched query exposures; entries are fractions (available/total).

| k | Both | Early only | Late only | Neither |
|---:|---:|---:|---:|---:|
| 1 | 0.695312 (384/384) | 0.117188 (384/384) | 0.015625 (384/384) | 0.171875 (384/384) |
| 2 | 0.572917 (384/384) | 0.132812 (384/384) | 0.0338542 (384/384) | 0.260417 (384/384) |
| 3 | 0.445312 (384/384) | 0.15625 (384/384) | 0.0364583 (384/384) | 0.361979 (384/384) |
| 4 | 0.369792 (384/384) | 0.171875 (384/384) | 0.0338542 (384/384) | 0.424479 (384/384) |
| 5 | 0.302083 (384/384) | 0.177083 (384/384) | 0.0286458 (384/384) | 0.492188 (384/384) |
| 6 | 0.276042 (384/384) | 0.140625 (384/384) | 0.03125 (384/384) | 0.552083 (384/384) |
| 7 | 0.260417 (384/384) | 0.138021 (384/384) | 0.0260417 (384/384) | 0.575521 (384/384) |
| 8 | 0.213542 (384/384) | 0.138021 (384/384) | 0.0390625 (384/384) | 0.609375 (384/384) |

