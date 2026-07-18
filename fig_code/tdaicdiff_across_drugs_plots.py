"""
fig-code/tdaicdiff-across-drugs-plots.py

FINAL FIGURES (3): TD-AIC (% change from baseline) vs Drugs.

X-axis groups: Baseline, Nitro, Washout(WO1), Phen, Washout(WO2), Dobu --
stops there (excludes the final washout before Esmo, and Esmo itself,
matching the sketch's 6 groups).

Y-axis: (%change TD) - (%change AIC), from the already-built Impella
percdiff files (TD_pctchange_mean - AIC_pctchange_mean).

Each point = cross-animal average (202,203,205,221) at that phase
POSITION within its group (averaging across whichever animals actually
have data at that position -- handles 202's single-dose Phen/Dobu not
having as many sub-phases as 203/205/221's low+high split).

FLAGGED DEVIATION FROM THE HAND SKETCH: the sketch shows exactly 2 points
per drug (implying P6+P3 merged into one point per dose level). This
script instead plots EVERY fine phase position separately (P6 and P3 as
distinct points, same color/gradient -- no P-level color distinction, per
user's "no P-level distinction in any plot", but not merged into fewer
points either).

BUG FIX (confirmed, this revision): washout-occurrence counting previously
incremented on EVERY row where med=="Washout" -- but each washout
occurrence is actually TWO rows (P6 and P3), so this split the first real
washout into "Washout_1"/"Washout_2" and silently dropped the real second
washout entirely (miscounted past the <=2 cutoff). Fixed: only increments
when a NEW washout block starts (previous row wasn't also "Washout"), so
both rows of one washout occurrence share the same tag. Verified with a
controlled test matching real ANIMAL_PHASES structure before shipping.

Color: Baseline=black, Washout=gray (no gradient), drug phases=gradient
by each point's own normalized dose value (continuous, not binary
low/high -- points naturally vary since normalized dose differs per
animal/phase).

STANDALONE test script: run directly, figure pops up on screen (swap to
savefig-only once approved). NOT wired into run_pipeline.py yet.

Save location (once approved): figures/tdaicdiff_across_drugs/
Code location: fig-code/tdaicdiff-across-drugs-plots.py (this file)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import phase_color, add_dose_gradient_legend

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPELLA_DIR = REPO_ROOT / "data" / "processed" / "impella_derived" / "summary_data"

ANIMAL_IDS = ["202", "203", "205", "221"]
GROUP_ORDER = ["Baseline", "Nitro", "Washout_1", "Phen", "Washout_2", "Dobu"]
GROUP_LABELS = ["Baseline", "Nitroprusside", "Washout", "Phenylephrine", "Washout", "Dobutamine"]


def _load_and_tag(animal_id):
    path = IMPELLA_DIR / f"{animal_id}_phase_summary_percdiff.pkl"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping.")
        return None
    df = pd.read_pickle(path).sort_values("phase_number").reset_index(drop=True)
    df["diff_pctchange"] = df["TD_pctchange_mean"] - df["AIC_pctchange_mean"]

    # Tag each row with its group. Washout occurrences are identified by
    # CONTIGUOUS BLOCKS of med=="Washout" rows -- the counter only advances
    # when a new block starts (previous row wasn't also Washout), so both
    # rows (P6 and P3) of one washout occurrence share the same tag.
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
        else:  # Esmo or anything else
            groups.append(None)
        prev_med = med
    df["group"] = groups
    return df[df["group"].notna()]


def make_tdaicdiff_across_drugs_figure(save=False, output_dir=None):
    all_data = {animal_id: _load_and_tag(animal_id) for animal_id in ANIMAL_IDS}
    all_data = {k: v for k, v in all_data.items() if v is not None}

    fig, ax = plt.subplots(figsize=(14, 7))

    x_cursor = 0
    group_boundaries = []
    for group_key, group_label in zip(GROUP_ORDER, GROUP_LABELS):
        group_start = x_cursor

        # position-index within this group, per animal
        per_animal_rows = {
            animal_id: df[df["group"] == group_key].reset_index(drop=True)
            for animal_id, df in all_data.items()
        }
        # Keep every available phase position. At each position, average across
        # the animals that actually have data there instead of truncating the
        # group to the shortest animal series.
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
            # std across animals at this position, for an error bar
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
        # dashed vertical separator between groups
        if group_key != GROUP_ORDER[-1]:
            ax.axvline(x=x_cursor - 0.5, color="lightgray", linestyle=":", linewidth=1)
        x_cursor += 0.5  # gap between groups

    ax.set_xticks([c for c, _ in group_boundaries])
    ax.set_xticklabels([l for _, l in group_boundaries])
    ax.set_ylabel("(% change TD) - (% change AIC)")
    fig.text(0.5, 0.94, "TD-AIC Difference During Different Pharmacological States", ha="center", fontsize=15)
    fig.text(0.5, 0.905, "(averaged across subjects 202, 203, 205, 221)", ha="center", fontsize=11)

    fig.subplots_adjust(top=0.83, bottom=0.16, right=0.97)
    legend_ax = fig.add_axes([0.77, 0.87, 0.16, 0.09])
    add_dose_gradient_legend(fig, ax=legend_ax)
    if save:
        output_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "tdaicdiff_across_drugs"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "tdaicdiff_across_drugs.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    make_tdaicdiff_across_drugs_figure(save=True)