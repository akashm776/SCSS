"""Generate the D1 report figure and tables from archived comparison data."""

import json
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = (
    ROOT / "experiment_results/clip_source_complete_2026-10-16/comparison.json"
)
ARMS = tuple(
    f"{regime}_{method}"
    for regime in ("H", "M", "C", "I")
    for method in ("Select", "Synth")
)


def load_comparison(path: Path = COMPARISON_PATH) -> dict:
    comparison = json.loads(path.read_text())
    validate_aggregate(comparison)
    return comparison


def validate_aggregate(comparison: dict) -> None:
    """Reject aggregates that differ from the equal-weight mean of seed results."""
    effects = comparison["timecourse"]["50"]["effects"]
    for weight in ("w100", "w500"):
        for arm in ARMS:
            for metric in ("loss", "r1_pp"):
                mean = statistics.fmean(
                    seed["timecourse"]["50"]["effects"][weight][arm][metric]
                    for seed in comparison["per_seed"]
                )
                if not abs(mean - effects[weight][arm][metric]) < 1e-14:
                    raise ValueError(f"Aggregate mismatch: {weight}/{arm}/{metric}")


def render_figure(comparison: dict, destination: Path) -> None:
    # Keep plotting dependencies and font-cache setup out of table-only callers.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "otco-paper-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    effects = comparison["timecourse"]["50"]["effects"]
    arms = ARMS
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "scss-d1",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7), sharey=True)
    colors = ["#17789c", "#bd5427"]
    positions = np.arange(len(arms))
    for axis, metric, factor, title, label in zip(
        axes,
        ["loss", "r1_pp"],
        [1000, 1],
        ["Reporting loss", "Bidirectional retrieval (R@1)"],
        [
            "Treatment − native loss (×10⁻³)\n← lower is better",
            "Treatment − native R@1 (percentage points)\nhigher is better →",
        ],
    ):
        axis.axvline(0, color="#555555", linewidth=1)
        for weight, offset, color, legend in zip(
            ["w100", "w500"],
            [-0.15, 0.15],
            colors,
            ["Early weights (100)", "Late weights (500)"],
        ):
            values = [effects[weight][arm][metric] * factor for arm in arms]
            for seed_index, seed in enumerate(comparison["per_seed"]):
                seed_values = [
                    seed["timecourse"]["50"]["effects"][weight][arm][metric] * factor
                    for arm in arms
                ]
                axis.scatter(
                    seed_values,
                    positions + offset + (seed_index - 1) * 0.035,
                    s=16,
                    facecolors="none",
                    edgecolors=color,
                    alpha=0.5,
                    linewidths=0.8,
                )
            axis.scatter(
                values,
                positions + offset,
                s=46,
                c=color,
                marker="D",
                label=legend,
                zorder=3,
            )
        axis.set_title(title, fontweight="bold", loc="left", pad=12)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.set_yticks(
            positions,
            [arm.replace("_", "-").replace("Synth", "Synthesis") for arm in arms],
        )
        axis.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle(
        "D1: early loss gains do not imply retrieval gains",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.84, wspace=0.10)
    fig.text(
        0.02,
        0.025,
        "Horizon 50 · Diamonds: equal-seed means · Open circles: three seed means (two streams averaged per seed)\nNo confidence intervals are implied. Source regimes and controls are defined in the report.",
        fontsize=9,
        color="#444444",
    )
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "d1_research_summary.svg", metadata={"Date": None})
    fig.savefig(destination / "d1_research_summary.png", dpi=160)
    plt.close(fig)


def render_tables(comparison: dict) -> str:
    effects = comparison["timecourse"]["50"]["effects"]
    arms = ARMS
    lines = [
        "# Appendix A. Detailed D1 results",
        "",
        "[Report contents](../../README.md#contents) · [Results](results.md) · [Evidence index](evidence.md)",
        "",
        "These tables are generated from the [completed D1 comparison](../../experiment_results/clip_source_complete_2026-10-16/comparison.json). Streams are averaged within each seed; the three seeds receive equal weight. H50 is the frozen primary horizon. Loss differences are unscaled; retrieval differences are percentage points.",
        "",
        "| Arm | Early loss Δ | Late loss Δ | Early R@1 Δ (pp) | Late R@1 Δ (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        early, late = effects["w100"][arm], effects["w500"][arm]
        lines.append(
            f"| {arm.replace('_','-')} | {early['loss']:+.9f} | {late['loss']:+.9f} | {early['r1_pp']:+.5f} | {late['r1_pp']:+.5f} |"
        )
    lines += [
        "",
        "## Prespecified side interaction",
        "",
        "Q = D(C-Select) − D(I-Select). The interaction is Q(late) − Q(early); negative favors the proposed relative shift toward C.",
        "",
        "| Seed | Q early | Q late | Interaction |",
        "|---|---:|---:|---:|",
    ]
    for seed in comparison["per_seed"]:
        contrasts = seed["timecourse"]["50"]["contrasts"]
        lines.append(
            f"| {seed['training_seed']} | {contrasts['Q']['w100']['loss']:+.9f} | {contrasts['Q']['w500']['loss']:+.9f} | {contrasts['Delta_side']['loss']:+.9f} |"
        )
    contrasts = comparison["timecourse"]["50"]["contrasts"]
    lines.append(
        f"| Equal-seed mean | {contrasts['Q']['w100']['loss']:+.9f} | {contrasts['Q']['w500']['loss']:+.9f} | {contrasts['Delta_side']['loss']:+.9f} |"
    )
    lines += [
        "",
        "## Secondary synthesis contrast",
        "",
        "Synth − Select within each source regime; negative favors synthesis.",
        "",
        "| Regime | Early loss contrast | Late loss contrast |",
        "|---|---:|---:|",
    ]
    for regime in ["H", "M", "C", "I"]:
        lines.append(
            f"| {regime} | {contrasts['G'][regime]['w100']['loss']:+.9f} | {contrasts['G'][regime]['w500']['loss']:+.9f} |"
        )
    lines += [
        "",
        "The full comparison retains stream-level matches and all recorded horizons. These summaries do not imply statistical significance. See the [limitations](limitations.md) and [completed interpretation](../research-status-2026-10-16-source-complete.md).",
        "",
        "Regenerate this appendix and the figure with `python -m scripts.build_research_summary`. Matplotlib is required for rendering.",
        "",
    ]
    return "\n".join(lines)


def build() -> None:
    comparison = load_comparison()
    render_figure(comparison, ROOT / "docs/figures")
    (ROOT / "docs/paper/detailed-results.md").write_text(render_tables(comparison))
    print("Verified equal-seed arithmetic; wrote D1 figure and detailed results.")


if __name__ == "__main__":
    build()
