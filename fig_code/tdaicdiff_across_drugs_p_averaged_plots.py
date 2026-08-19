"""
fig_code/tdaicdiff_across_drugs_averaged_plots.py

FINAL FIGURES (3), P3/P6-AVERAGED VERSION: TD-AIC (% change from baseline)
vs Drugs -- P6 and P3 merged into one averaged point per occurrence.

NEW variant, does NOT replace tdaicdiff_across_drugs_plots.py (that version,
showing P6/P3 as separate points, is kept as-is). This version averages P6
+ P3 together within each drug-dose occurrence (and within Baseline/each
Washout, which have no dose split) -- per user: P3 vs P6 showed no
significant difference (run_p3_p6_test), so averaging them is valid.
Result: Baseline -> 1 point, Nitro -> 2 points (low/high dose), Washout ->
1 point each, Phen/Dobu -> 1 or 2 points depending on the animal's actual
dosing (some animals only got a single, unsplit dose for these drugs).

Merging is done via fig_style.merge_p3_p6() -- groups by (drug group, dose
value), since P6 and P3 rows in the same occurrence share the identical
normalized dose value. Handles asymmetric cases (e.g. animal 205's
Dobu-high, P6 only, no matching P3) automatically -- a group of one row
just averages to itself.

Same data source, x-axis groups, and dose-gradient color scheme as the
non-averaged version -- only the P6/P3 merge step is new.

STANDALONE test script: run directly, figure pops up on screen (swap to
savefig-only once approved). NOT wired into run_pipeline.py yet.

Save location (once approved): figures/tdaicdiff_across_drugs_averaged/
Code location: fig_code/tdaicdiff_across_drugs_p_averaged_plots.py (this file)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import phase_color, add_dose_gradient_legend, merge_p3_p6

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPELLA_DIR = REPO_ROOT / "data" / "processed" / "impella_derived" / "summary_data"

ANIMAL_IDS = ["202", "203", "205", "221"]
GROUP_ORDER = ["Baseline", "Nitro", "Washout_1", "Phen", "Washout_2", "Dobu"]
GROUP_LABELS = ["Baseline", "Nitroprusside", "Washout", "Phenylephrine", "Washout", "Dobutamine"]


def _load_tag_and_merge(animal_id):
    path = IMPELLA_DIR / f"{animal_id}_phase_summary_percdiff.pkl"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping.")
        return None
    df = pd.read_pickle(path).sort_values("phase_number").reset_index(drop=True)
    df["diff_pctchange"] = df["TD_pctchange_mean"] - df["AIC_pctchange_mean"]

    # Same washout-block tagging as the non-averaged version (contiguous
    # blocks of med=="Washout" share one tag -- see that script for the
    # full bug-fix explanation).
    washout_count = 0
    groups = []
    prev_med = None
    for med in df["med"]:
        if med == "Washout":
            if prev_med != "Washout":
                washout_count += 1
            groups.append(f"Washout_{washout_count}" if washout_count <= 2 else None)
        elif med in ("Baseline", "Nitro", "Phen", "Dobu"):
            groups.append(med)
        else:
            groups.append(None)
        prev_med = med
    df["group"] = groups
    df = df[df["group"].notna()]

    return merge_p3_p6(df, ["diff_pctchange"])


def make_tdaicdiff_across_drugs_averaged_figure(save=False, output_dir=None):
    all_data = {animal_id: _load_tag_and_merge(animal_id) for animal_id in ANIMAL_IDS}
    all_data = {k: v for k, v in all_data.items() if v is not None}

    fig, ax = plt.subplots(figsize=(14, 7))

    x_cursor = 0
    group_boundaries = []
    for group_key, group_label in zip(GROUP_ORDER, GROUP_LABELS):
        group_start = x_cursor

        per_animal_rows = {
            animal_id: df[df["group"] == group_key].reset_index(drop=True)
            for animal_id, df in all_data.items()
        }
        max_positions = max((len(r) for r in per_animal_rows.values()), default=0)

        for pos in range(max_positions):
            vals = []
            doses = []
            for animal_id, rows in per_animal_rows.items():
                if pos < len(rows):
                    vals.append(rows.loc[pos, "diff_pctchange"])
                    doses.append(rows.loc[pos, "dose"])
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            std_val = (pd.Series(vals).std() if len(vals) > 1 else 0)
            mean_dose = pd.Series(doses).mean() if group_key not in ("Baseline",) and not all(pd.isna(doses)) else None

            if group_key == "Baseline":
                color = phase_color("Baseline", None)
            elif group_key.startswith("Washout"):
                color = phase_color("Washout", None)
            else:
                color = phase_color(group_key, mean_dose if mean_dose is not None else 0)

            ax.errorbar(x_cursor, mean_val, yerr=std_val, marker="o", color=color,
                        capsize=4, linestyle="none")
            x_cursor += 1

        group_center = (group_start + x_cursor - 1) / 2 if x_cursor > group_start else group_start
        group_boundaries.append((group_center, group_label))
        if group_key != GROUP_ORDER[-1]:
            ax.axvline(x=x_cursor - 0.5, color="lightgray", linestyle=":", linewidth=1)
        x_cursor += 0.5

    ax.set_xticks([c for c, _ in group_boundaries])
    ax.set_xticklabels([l for _, l in group_boundaries])
    ax.set_ylabel("(% change TD) - (% change AIC)")
    fig.text(0.5, 0.94, "TD-AIC Difference During Different Pharmacological States", ha="center", fontsize=15)
    fig.text(0.5, 0.905, "(P6/P3 averaged; averaged across subjects 202, 203, 205, 221)", ha="center", fontsize=11)

    fig.subplots_adjust(top=0.83, bottom=0.16, right=0.97)
    legend_ax = fig.add_axes([0.77, 0.87, 0.16, 0.09])
    add_dose_gradient_legend(fig, ax=legend_ax)

    if save:
        output_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "tdaicdiff_across_drugs_p_averaged"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "tdaicdiff_across_drugs_p_averaged.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    make_tdaicdiff_across_drugs_averaged_figure(save=True)