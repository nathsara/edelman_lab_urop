"""
fig-code/tdaicdiff-vs-metrics-plots.py

FINAL FIGURES (6): Diff (TD-AIC) % Change from Baseline vs. {metric}.

REVISED per user feedback -- NOT 24 individual (animal, metric) plots.
Now: SIX figures (one per metric), each pooling ALL 4 animals' points
together on one axes, PLUS one final combined figure with all six as
subplots.

Y-axis: (%change TD) - (%change AIC) [Impella-derived percdiff].
X-axis: RAW metric value (catheter-derived).
Scope: Nitro/Phen/Dobu phases only (matches the sketch's 3-entry legend).

Points colored by normalized-dose gradient (per drug) -- gradient colorbar
legend (not a plain scatter legend). Trendline (linear regression) per
drug, fit on the POOLED 4-animal data; equation + R^2 reported in BLACK
text, positioned in a fixed corner away from the data/trendlines.

STANDALONE test script: run directly, figures pop up on screen one at a
time (swap to savefig-only once approved). NOT wired into run_pipeline.py
yet.

Save locations (once approved):
  figures/tdaicdiff_vs_metrics/{metric}.png       (6 files)
  figures/tdaicdiff_vs_metrics/all_metrics.png    (1 combined file)
Code location: fig-code/tdaicdiff-vs-metrics-plots.py (this file)

COSMETIC PASS per user feedback: dose-gradient legend shrunk (thin bars,
small font) and moved to upper-right, aligned with the top of the main
plot(s), instead of the old tall box. For the combined 6-subplot figure,
axis labels/tick labels and the per-drug trendline equation text are
smaller so they don't run into each other, and the shared legend is
smaller/better placed.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import DRUG_BASE_COLOR, dose_gradient_color, add_dose_gradient_legend

REPO_ROOT = Path(__file__).resolve().parent.parent
CATHETER_DIR = REPO_ROOT / "data" / "processed" / "catheter_derived" / "summary_data"
IMPELLA_DIR = REPO_ROOT / "data" / "processed" / "impella_derived" / "summary_data"

ANIMAL_IDS = ["202", "203", "205", "221"]
DRUGS = ["Nitro", "Phen", "Dobu"]

METRICS = [
    ("HR_catheter", "HR (bpm)"),
    ("PP_catheter", "PP (mmHg)"),
    ("MAP_catheter", "MAP (mmHg)"),
    ("dpdt_max", "dp/dt max (mmHg/s)"),
    ("dpdt_min", "dp/dt min (mmHg/s)"),
    ("lvedp", "LVEDP (mmHg)"),
]


def load_joined_data(animal_id):
    catheter_path = CATHETER_DIR / f"{animal_id}_catheter_summary.pkl"
    impella_path = IMPELLA_DIR / f"{animal_id}_phase_summary_percdiff.pkl"
    if not catheter_path.exists() or not impella_path.exists():
        print(f"  WARNING: missing data for {animal_id}, skipping.")
        return None
    cath = pd.read_pickle(catheter_path)
    imp = pd.read_pickle(impella_path)[["phase_number", "med", "dose", "TD_pctchange_mean", "AIC_pctchange_mean"]]
    merged = cath.merge(imp, on="phase_number", suffixes=("", "_impella"))
    merged["diff_pctchange"] = merged["TD_pctchange_mean"] - merged["AIC_pctchange_mean"]
    return merged[merged["med"].isin(DRUGS)]


def _pooled_data(metric_prefix):
    """Pools all 4 animals' joined data into one DataFrame for a given metric."""
    frames = [d for d in (load_joined_data(a) for a in ANIMAL_IDS) if d is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_metric_axis(ax, data, metric_prefix, xlabel, label_fontsize=14,
                       tick_fontsize=None, eq_fontsize=9, point_size=35):
    metric_col = f"{metric_prefix}_mean"
    eq_lines = []
    for drug in DRUGS:
        subset = data[data["med"] == drug]
        if subset.empty:
            continue
        colors = [dose_gradient_color(drug, d) for d in subset["dose"]]
        ax.scatter(subset[metric_col], subset["diff_pctchange"], color=colors, s=point_size)

        if len(subset) >= 2:
            fit = linregress(subset[metric_col], subset["diff_pctchange"])
            xs = subset[metric_col].sort_values()
            ax.plot(xs, fit.slope * xs + fit.intercept, color=DRUG_BASE_COLOR[drug], linewidth=1.5)
            eq_lines.append(f"{drug}: y={fit.slope:.3g}x+{fit.intercept:.3g}, R\u00b2={fit.rvalue**2:.3f}")

    # Trendline equations in BLACK, fixed corner, away from data/trendlines.
    if eq_lines:
        ax.text(0.02, 0.02, "\n".join(eq_lines), transform=ax.transAxes,
                fontsize=eq_fontsize, va="bottom", ha="left", color="black",
                bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.85,
                          boxstyle="round", pad=0.3))

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel("Diff (TD-AIC) (% change from baseline)", fontsize=label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(labelsize=tick_fontsize)


def make_single_metric_figures(save=False, output_dir=None):
    figs = {}
    for metric_prefix, xlabel in METRICS:
        data = _pooled_data(metric_prefix)
        fig, ax = plt.subplots(figsize=(9, 7))
        _plot_metric_axis(ax, data, metric_prefix, xlabel)
        metric_name = xlabel.split(" (")[0]
        fig.suptitle(f"Diff (TD-AIC) % Change vs. {metric_name}\n(subjects 202, 203, 205, 221)", fontsize=14)
        fig.subplots_adjust(right=0.80, top=0.85)
        # Small legend, upper-right, top-aligned with the main plot (top=0.85).
        add_dose_gradient_legend(fig, rect=[0.81, 0.72, 0.13, 0.11])
        figs[metric_prefix] = fig

        if save:
            out_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "tdaicdiff_vs_metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{metric_prefix}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")
            plt.close(fig)
    if not save:
        plt.show()
    return figs


def make_combined_figure(save=False, output_dir=None):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Diff (TD-AIC) % Change vs. Metrics\n(subjects 202, 203, 205, 221)", fontsize=16)

    for (metric_prefix, xlabel), ax in zip(METRICS, axes.flat):
        data = _pooled_data(metric_prefix)
        _plot_metric_axis(ax, data, metric_prefix, xlabel,
                           label_fontsize=9, tick_fontsize=8,
                           eq_fontsize=6.5, point_size=20)

    fig.subplots_adjust(top=0.88, right=0.90, hspace=0.3, wspace=0.4)
    # Small legend, upper-right corner, aligned with top of the subplot grid.
    add_dose_gradient_legend(fig, rect=[0.91, 0.80, 0.06, 0.09])

    if save:
        out_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "tdaicdiff_vs_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all_metrics.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    make_single_metric_figures(save=True)
    make_combined_figure(save=True)