"""
fig-code/baseline-plots.py

FINAL FIGURES (1): Preliminary Analysis with Baseline Cohort.

One figure, four subplots (2x2): TD-AIC (raw, L/min) vs. each of
SmartPump, Pulse Pressure, HR, MAP (Impella-derived, raw units).

Data: ALL rows from the 8 baseline-cohort animals (103,104,105,106,107,
148,155,218) + ONLY the Baseline-phase rows (P6 and P3) from the 4
normal-cohort animals (202,203,205,221) -- their drug/washout/Esmo rows
are excluded, since this figure is about the resting/healthy state only.

No P-level color distinction (per user: "no color coding P-level" for this
figure specifically) -- every point plotted in one flat color.

Y-axis label per subplot: "TD-AIC [L/min]" (typo in hand-drawn notes said
"P-AIC" on one subplot -- confirmed with user this should read TD-AIC like
the other three).

SmartPump units: UNKNOWN -- flagged, left unitless on that one subplot's
x-axis rather than guessing a wrong unit.

STANDALONE test script: run directly, figures pop up on screen (per user
guideline 7 -- swap to savefig-only once approved). NOT wired into
run_pipeline.py yet (per guideline 8).

Save location (once approved): figures/baseline_analysis/
Code location: fig-code/baseline-plots.py (this file)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import BASELINE_COLOR

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPELLA_DIR = REPO_ROOT / "data" / "processed" / "impella_derived" / "summary_data"

BASELINE_COHORT_IDS = ["103", "104", "105", "106", "107", "148", "155", "218"]
NORMAL_COHORT_IDS = ["202", "203", "205", "221"]

# (metric column prefix, x-axis label, subplot position (row, col))
METRICS = [
    ("SmartPump", "SmartPump", (0, 0)),
    ("PP_impella", "Pulse Pressure (mmHg)", (0, 1)),
    ("HR_impella", "HR (bpm)", (1, 0)),
    ("MAP_impella", "MAP (mmHg)", (1, 1)),
]


def load_baseline_analysis_data():
    """
    Returns a single combined DataFrame: TD-AIC (raw) + each Impella metric
    (raw mean), for baseline-cohort animals (all rows) + normal-cohort
    animals (Baseline phase rows only).
    """
    rows = []

    for animal_id in BASELINE_COHORT_IDS:
        path = IMPELLA_DIR / f"{animal_id}_phase_summary.pkl"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping.")
            continue
        df = pd.read_pickle(path)
        rows.append(df)

    for animal_id in NORMAL_COHORT_IDS:
        path = IMPELLA_DIR / f"{animal_id}_phase_summary.pkl"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping.")
            continue
        df = pd.read_pickle(path)
        rows.append(df[df["med"] == "Baseline"])

    combined = pd.concat(rows, ignore_index=True)
    combined["TD_AIC_diff"] = combined["TD_mean"] - combined["AIC_mean"]
    return combined


def make_baseline_analysis_figure(save=False, output_dir=None):
    data = load_baseline_analysis_data()

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        "Effect of Impella-derived Hemodynamic Metrics on\nDifference between TD and AIC Cardiac Output",
        fontsize=16,
    )

    for metric_prefix, xlabel, (row, col) in METRICS:
        ax = axes[row][col]
        metric_col = f"{metric_prefix}_mean"
        ax.scatter(data[metric_col], data["TD_AIC_diff"], color=BASELINE_COLOR, alpha=0.7)
        ax.axhline(y=0, color="lightgray", linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TD-AIC (L/min)")

    fig.subplots_adjust(top=0.85, hspace=0.35, wspace=0.3)

    if save:
        output_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "baseline_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "baseline_analysis.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    make_baseline_analysis_figure(save=True)