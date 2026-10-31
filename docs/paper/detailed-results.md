# Appendix A. Detailed D1 results

[Report contents](../../README.md#contents) · [Results](results.md) · [Evidence index](evidence.md)

These tables are generated from the [completed D1 comparison](../../experiment_results/clip_source_complete_2026-10-16/comparison.json). Streams are averaged within each seed; the three seeds receive equal weight. H50 is the frozen primary horizon. Loss differences are unscaled; retrieval differences are percentage points.

| Arm | Early loss Δ | Late loss Δ | Early R@1 Δ (pp) | Late R@1 Δ (pp) |
|---|---:|---:|---:|---:|
| H-Select | -0.000123700 | +0.000137865 | -0.08138 | -0.03255 |
| H-Synth | -0.000685671 | +0.000387684 | -0.03255 | +0.06510 |
| M-Select | -0.000272764 | +0.000215344 | -0.03255 | -0.01628 |
| M-Synth | -0.000824949 | +0.000152951 | -0.09766 | -0.01628 |
| C-Select | -0.000126386 | +0.000172882 | -0.04883 | +0.00000 |
| C-Synth | -0.000842937 | +0.000249853 | -0.03255 | +0.06510 |
| I-Select | -0.000028824 | +0.000142133 | -0.01628 | +0.00000 |
| I-Synth | -0.000766799 | +0.000244596 | -0.03255 | -0.03255 |

## Prespecified side interaction

Q = D(C-Select) − D(I-Select). The interaction is Q(late) − Q(early); negative favors the proposed relative shift toward C.

| Seed | Q early | Q late | Interaction |
|---|---:|---:|---:|
| 271828 | +0.000086814 | -0.000024934 | -0.000111748 |
| 161803 | -0.000251531 | +0.000037406 | +0.000288937 |
| 141421 | -0.000127970 | +0.000079777 | +0.000207747 |
| Equal-seed mean | -0.000097562 | +0.000030750 | +0.000128312 |

## Secondary synthesis contrast

Synth − Select within each source regime; negative favors synthesis.

| Regime | Early loss contrast | Late loss contrast |
|---|---:|---:|
| H | -0.000561972 | +0.000249819 |
| M | -0.000552185 | -0.000062393 |
| C | -0.000716551 | +0.000076971 |
| I | -0.000737975 | +0.000102463 |

The full comparison retains stream-level matches and all recorded horizons. These summaries do not imply statistical significance. See the [limitations](limitations.md) and [completed interpretation](../research-status-2026-10-16-source-complete.md).

Regenerate this appendix and the figure with `python -m scripts.build_research_summary`. Matplotlib is required for rendering.
