"""
fig-code/ct-drug-effect-plots.py

FINAL FIGURES (2): CT Drug Effect Trajectory Analysis.

One figure, six subplots (3x2): HR, MAP, PP, dp/dt max, dp/dt min, LVEDP
(all catheter-derived). Each subplot shows percent-change-from-baseline
over time (minutes), one line per drug (Nitro=red, Phen=green, Dobu=blue)
-- SINGLE CROSS-ANIMAL AVERAGE LINE ONLY, no individual animal traces
(confirmed with user, differs from the PPT reference and from
sanity_check_ct_drug_effect.py, both of which also showed per-animal
lines).

Average = same truncate-to-shortest-then-average-by-row-position method
already used elsewhere in this pipeline (matches
sanity_check_ct_drug_effect.py's cross-animal average, and the original
legacy graphing.py convention) -- NOT true time-interpolation.

Dosage-increase vertical lines: SKIPPED per user instruction -- we don't
have the real timestamps for when dosage increased within a drug window
(drug_normalization.csv's Start/End are per whole low or high dose
administration, not a single "increase" instant within a continuous
trace). Only the trajectory lines are plotted.

Reads the already-built {label}_{metric}_MA.pkl files (window=240 rolling
average, per-beat elapsed-seconds time axis) from
data/processed/ct_drug_effect/{animal_id}/ -- converts seconds to minutes
for the x-axis per user's sketch.

STANDALONE test script: run directly, figure pops up on screen (swap to
savefig-only once approved). NOT wired into run_pipeline.py yet.

Save location (once approved): figures/ct_drug_effect/
Code location: fig-code/ct-drug-effect-plots.py (this file)
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import DRUG_BASE_COLOR

REPO_ROOT = Path(__file__).resolve().parent.parent
CT_DRUG_EFFECT_DIR = REPO_ROOT / "data" / "processed" / "ct_drug_effect"

ANIMAL_IDS = ["202", "203", "205", "221"]
DRUGS = ["Nitro", "Phen", "Dobu"]

# (metric key matching {label}_{metric}_MA.pkl, subplot y-label, subplot position)
METRICS = [
    ("hr_catheter", "HR [% change]", (0, 0)),
    ("map_catheter", "MAP [% change]", (0, 1)),
    ("pp_catheter", "PP [% change]", (0, 2)),
    ("dpdt_max", "dp/dt max [% change]", (1, 0)),
    ("dpdt_min", "dp/dt min [% change]", (1, 1)),
    ("lvedp", "LVEDP [% change]", (1, 2)),
]


def _elapsed_minutes(ma_df):
    start = datetime.combine(datetime.today(), ma_df["time"].iloc[0])
    return [(datetime.combine(datetime.today(), t) - start).total_seconds() / 60 for t in ma_df["time"]]


def _cross_animal_average(animal_id_list, drug, metric):
    """
    Loads {animal_id}_{drug}_{metric}_MA.pkl for each animal, truncates to
    the shortest series, and averages by row position (same convention as
    sanity_check_ct_drug_effect.py / legacy graphing.py).
    """
    ys, xs = [], []
    for animal_id in animal_id_list:
        path = CT_DRUG_EFFECT_DIR / animal_id / f"{animal_id}_{drug}_{metric}_MA.pkl"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping {animal_id} for {drug}/{metric}.")
            continue
        ma_df = pd.read_pickle(path)
        xs.append(_elapsed_minutes(ma_df))
        ys.append(ma_df["SMA"])

    if not ys:
        return None, None

    min_len = min(len(y) for y in ys)
    ys_trim = [y.iloc[:min_len] for y in ys]
    x_trim = xs[0][:min_len]
    y_avg = sum(ys_trim) / len(ys_trim)
    return x_trim, y_avg


def make_ct_drug_effect_figure(save=False, output_dir=None):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        "Continuous-Time Drug Effect on the\nTime-Evolution of Catheter-Derived Hemodynamic Metrics",
        fontsize=17,
    )

    for metric, ylabel, (row, col) in METRICS:
        ax = axes[row][col]
        metric_name = ylabel.split(" [")[0]
        all_x_mins = []
        for drug in DRUGS:
            x, y = _cross_animal_average(ANIMAL_IDS, drug, metric)
            if x is None:
                continue
            ax.plot(x, y, color=DRUG_BASE_COLOR[drug], label=drug, linewidth=1.5)
            all_x_mins.append(min(x))

        # Per-subplot title, in addition to the axis label -- unambiguous
        # which panel is which metric even before reading axis labels.
        ax.set_title(metric_name, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("minutes")
        ax.set_ylabel(ylabel)

        # Explicit xlim (not just ax.margins(x=0)) to guarantee zero
        # whitespace between the y-axis and the start of the trendlines --
        # margins alone can still leave a hair of padding depending on
        # renderer/backend, an explicit left bound does not.
        if all_x_mins:
            ax.set_xlim(left=min(all_x_mins))
        ax.margins(x=0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=13)

    # More generous spacing between subplots -- both directions -- so
    # x-axis labels of the top row never sit close to titles of the bottom
    # row, and y-axis labels never crowd the previous column's plot.
    fig.subplots_adjust(top=0.85, bottom=0.13, hspace=0.55, wspace=0.4, left=0.06, right=0.98)

    if save:
        output_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "ct_drug_effect"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "ct_drug_effect_trajectories.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    make_ct_drug_effect_figure(save=True)