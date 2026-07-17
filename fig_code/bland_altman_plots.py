"""
fig-code/bland-altman-plots.py

FINAL FIGURES (4): Bland-Altman (percent-change from baseline).

X-axis: Mean of Percent Change from Baseline ((AIC+TD)/2)
Y-axis: Difference (TD-AIC) of Percent Change from Baseline

Every individual phase-row, pooled across all 4 normal-cohort animals
(NOT averaged -- matches the PPT reference, many scattered points).

Scope: Baseline, Nitro, Phen, Dobu phases only -- Washout and Esmo
EXCLUDED. Flagged assumption, confirmed with user context: the legend has
exactly 4 categories (Baseline/Nitro/Phen/Dobu), no Washout entry shown.

Colors: flat/categorical (NOT dose-gradient) -- Baseline=black, Nitro=red,
Phen=green, Dobu=blue -- matches the sketch's legend exactly.

Lines: solid gray at mean difference, dashed gray at +-1.96 SD (computed
from the actual plotted differences, not hardcoded).

STANDALONE test script: run directly, figure pops up on screen (swap to
savefig-only once approved). NOT wired into run_pipeline.py yet.

Save location (once approved): figures/bland_altman/
Code location: fig-code/bland-altman-plots.py (this file)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import DRUG_BASE_COLOR, BASELINE_COLOR

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPELLA_DIR = REPO_ROOT / "data" / "processed" / "impella_derived" / "summary_data"

ANIMAL_IDS = ["202", "203", "205", "221"]
INCLUDED_MEDS = ["Baseline", "Nitro", "Phen", "Dobu", "Washout"]

CATEGORY_COLOR = {**DRUG_BASE_COLOR, "Baseline": BASELINE_COLOR, "Washout": "gray"}


def load_bland_altman_data():
    rows = []
    for animal_id in ANIMAL_IDS:
        path = IMPELLA_DIR / f"{animal_id}_phase_summary_percdiff.pkl"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping.")
            continue
        df = pd.read_pickle(path)
        df = df[df["med"].isin(INCLUDED_MEDS)].copy()
        df["mean_pctchange"] = (df["AIC_pctchange_mean"] + df["TD_pctchange_mean"]) / 2
        df["diff_pctchange"] = df["TD_pctchange_mean"] - df["AIC_pctchange_mean"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_bland_altman_figure(save=False, output_dir=None):
    data = load_bland_altman_data()

    fig, ax = plt.subplots(figsize=(10, 7))

    for med in INCLUDED_MEDS:
        subset = data[data["med"] == med]
        ax.scatter(subset["mean_pctchange"], subset["diff_pctchange"],
                   color=CATEGORY_COLOR[med], label=med, alpha=0.8)

    mean_diff = data["diff_pctchange"].mean()
    sd_diff = data["diff_pctchange"].std()
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    ax.axhline(mean_diff, color="gray", linestyle="-")
    ax.axhline(upper, color="gray", linestyle="--")
    ax.axhline(lower, color="gray", linestyle="--")

    ax.set_ylim(-80, 80)
    xmin, xmax = data["mean_pctchange"].min(), data["mean_pctchange"].max()
    xpad = (xmax - xmin) * 0.02
    label_x = xmax - xpad
    ax.text(label_x, mean_diff, f"mean diff: {mean_diff:.2f}", va="bottom", ha="right", fontsize=11)
    ax.text(label_x, upper, f"+1.96 SD: {upper:.2f}", va="bottom", ha="right", fontsize=11)
    ax.text(label_x, lower, f"-1.96 SD: {lower:.2f}", va="top", ha="right", fontsize=11)

    ax.set_xlabel("Mean of Percent Change from Baseline ((AIC+TD)/2) (%)")
    ax.set_ylabel("Difference (TD-AIC) of Percent Change from Baseline (%)")
    ax.set_title("Bland-Altman: TD vs. AIC Percent Change from Baseline")
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save:
        output_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "bland_altman"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "bland_altman_percdiff.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    make_bland_altman_figure(save=True)