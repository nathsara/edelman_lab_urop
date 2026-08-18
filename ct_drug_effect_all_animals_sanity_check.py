"""
ct_drug_effect_all_animals_sanity_check.py

STANDALONE, DISPOSABLE sanity-check script -- NOT part of run_pipeline.py,
not imported by anything, not in fig_code/. Delete this (and the figures it
produces) once the comparison against old reference figures is done.

Extends sanity_check_ct_drug_effect.py: instead of one metric per run via
CLI arg, generates all 6 figures (one per metric) in a single run, so the
full old-graph comparison can happen in one pass.

Purpose: two concerns were raised about the CURRENT (post-timestamp-bug-fix)
ct_drug_effect_plots.py output -- (1) HR was expected to rise under Phen but
dips instead, (2) not all lines start at y=0. This script shows each
individual animal's trajectory line (not just the cross-animal average),
using the SAME data location the real pipeline reads/writes
(data/processed/ct_drug_effect/{animal}/{animal}_{drug}_{metric}_MA.pkl) --
so it automatically reflects the current, regenerated, post-bug-fix data,
with no separate data-loading path to keep in sync.

Reproduced from legacy exactly (same as the original sanity check):
  - 3 stacked subplots, one per drug (Nitro/Phen/Dobu), sharing the x-axis
  - Each animal's MA line plotted individually (red/green/blue per drug)
  - x-axis = elapsed seconds from that animal's own window start (per-animal
    anchoring, not shared across animals -- matches legacy)
  - Cross-animal "Avg" dashed line = TRUNCATE each animal's series to the
    length of the shortest one, then average by row POSITION -- the OLD
    averaging method, unchanged here. (The NEW expanding-pool averaging
    method is being tested separately, in
    ct_drug_effect_expanding_avg_sanity_check.py -- kept deliberately apart
    from this file, which is only about verifying individual animal lines
    against old reference figures, not about changing the averaging method.)
  - Legend label only set on the first animal's line per drug (matches
    legacy's literal behavior -- avoids 4 duplicate legend entries)

Usage (run from repo root):
    python ct_drug_effect_all_animals_sanity_check.py
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

ANIMAL_IDS = ["202", "203", "205", "221"]

DRUGS = [
    ("Nitro", "red", "darkred"),
    ("Phen", "green", "darkgreen"),
    ("Dobu", "blue", "darkblue"),
]

METRIC_TITLES = {
    "dpdt_max": "dP/dt Max",
    "dpdt_min": "dP/dt Min",
    "lvedp": "LVEDP",
    "pp_catheter": "Pulse Pressure",
    "map_catheter": "MAP",
    "hr_catheter": "Heart Rate",
}

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "processed" / "ct_drug_effect"


def load_ma(animal_id, drug, metric):
    path = DATA_ROOT / animal_id / f"{animal_id}_{drug}_{metric}_MA.pkl"
    if not path.exists():
        return None
    return pd.read_pickle(path)


def elapsed_seconds(ma_df):
    """Anchors each animal's own timestamps to elapsed seconds from ITS OWN
    window start -- matches legacy exactly, NOT a shared/global time origin
    across animals."""
    start = datetime.combine(datetime.today(), ma_df["time"].iloc[0])
    return [(datetime.combine(datetime.today(), t) - start).total_seconds() for t in ma_df["time"]]


def make_figure_for_metric(metric):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    for ax, (drug, color, avg_color) in zip(axes, DRUGS):
        ys, xs = [], []
        for animal_id in ANIMAL_IDS:
            ma_df = load_ma(animal_id, drug, metric)
            if ma_df is None:
                print(f"  WARNING: no data for {animal_id}/{drug}/{metric}, skipping this animal.")
                continue
            x = elapsed_seconds(ma_df)
            y = ma_df["SMA"]
            ys.append(y)
            xs.append(x)
            # Legend label only on the first PLOTTED animal's line -- avoids
            # duplicate legend entries. Uses enumerate position, not
            # ANIMAL_IDS[0] directly, in case that animal's data is missing.
            label = drug if len(ys) == 1 else None
            ax.plot(x, y, color=color, label=label, alpha=0.7)

        if ys:
            # OLD averaging method (unchanged) -- truncate to shortest,
            # average by row position. See module docstring.
            min_len = min(len(y) for y in ys)
            ys_trim = [y.iloc[:min_len] for y in ys]
            x_trim = xs[0][:min_len]
            y_avg = sum(ys_trim) / len(ys_trim)
            ax.plot(x_trim, y_avg, color=avg_color, linestyle="--", linewidth=3, zorder=10, label=f"{drug} Avg")

        ax.axhline(y=0, color="lightgray", linestyle=":", linewidth=1, zorder=0)
        ax.set_title(f"{drug} - {METRIC_TITLES[metric]}")
        ax.set_ylabel("Percent change")
        ax.legend()

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"{METRIC_TITLES[metric]} — Percent change from baseline (all animals, sanity check)")
    plt.tight_layout()
    return fig


def main():
    for metric in METRIC_TITLES:
        print(f"Generating {metric}...")
        make_figure_for_metric(metric)
    plt.show()  # shows all 6 figures at once (separate windows)


if __name__ == "__main__":
    main()