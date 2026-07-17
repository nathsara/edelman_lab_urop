"""
fig-code/metrics-vs-drugs-plots.py

FINAL FIGURES (6): {Metric} (raw value, NOT percent change) vs Drugs.

Same x-axis grouping and dose-gradient/color logic as Figure 3
(tdaicdiff-across-drugs-plots.py) -- Baseline, Nitro, WO1, Phen, WO2, Dobu
-- but y-axis is each catheter-derived metric's RAW mean value (not a
percent-change), and this produces SIX separate figures, one per metric
(HR, PP, MAP, dp/dt max, dp/dt min, LVEDP), per user: "x6 for HR PP MAP
dp/dt max dp/dt min LVEDP".

Same flagged deviation from the hand sketch as Figure 3: plots every fine
phase position separately (P6 and P3 as distinct points, not merged into
one point per dose level) -- see that script's docstring for the full
reasoning.

STANDALONE test script: run directly, figures pop up on screen one at a
time (swap to savefig-only once approved). NOT wired into run_pipeline.py
yet.

Save location (once approved): figures/metrics_vs_drugs/{metric}.png
Code location: fig-code/metrics-vs-drugs-plots.py (this file)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import phase_color, add_dose_gradient_legend

REPO_ROOT = Path(__file__).resolve().parent.parent
CATHETER_DIR = REPO_ROOT / "data" / "processed" / "catheter_derived" / "summary_data"

ANIMAL_IDS = ["202", "203", "205", "221"]
GROUP_ORDER = ["Baseline", "Nitro", "Washout_1", "Phen", "Washout_2", "Dobu"]
GROUP_LABELS = ["Baseline", "Nitroprusside", "Washout", "Phenylephrine", "Washout", "Dobutamine"]

# (metric column prefix, y-axis label)
METRICS = [
    ("HR_catheter", "HR (bpm)"),
    ("PP_catheter", "PP (mmHg)"),
    ("MAP_catheter", "MAP (mmHg)"),
    ("dpdt_max", "dp/dt max (mmHg/s)"),
    ("dpdt_min", "dp/dt min (mmHg/s)"),
    ("lvedp", "LVEDP (mmHg)"),
]


def _load_and_tag(animal_id):
    path = CATHETER_DIR / f"{animal_id}_catheter_summary.pkl"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping.")
        return None
    df = pd.read_pickle(path).sort_values("phase_number").reset_index(drop=True)

    washout_count = 0
    groups = []
    for med in df["med"]:
        if med == "Washout":
            washout_count += 1
            groups.append(f"Washout_{washout_count}" if washout_count <= 2 else None)
        elif med in ("Baseline", "Nitro", "Phen", "Dobu"):
            groups.append(med)
        else:
            groups.append(None)
    df["group"] = groups
    return df[df["group"].notna()]


def make_metric_vs_drugs_figure(metric_prefix, ylabel, save=False, output_dir=None):
    all_data = {animal_id: _load_and_tag(animal_id) for animal_id in ANIMAL_IDS}
    all_data = {k: v for k, v in all_data.items() if v is not None}
    metric_col = f"{metric_prefix}_mean"

    fig, ax = plt.subplots(figsize=(14, 7))

    x_cursor = 0
    group_boundaries = []
    for group_key, group_label in zip(GROUP_ORDER, GROUP_LABELS):
        group_start = x_cursor
        per_animal_rows = {
            animal_id: df[df["group"] == group_key].reset_index(drop=True)
            for animal_id, df in all_data.items()
        }
        # Keep every available phase position. At each position, average across
        # the animals that actually have data there instead of truncating the
        # group to the shortest animal series.
        max_positions = max((len(r) for r in per_animal_rows.values()), default=0)

        for pos in range(max_positions):
            vals, doses = [], []
            for animal_id, rows in per_animal_rows.items():
                if pos < len(rows):
                    vals.append(rows.loc[pos, metric_col])
                    doses.append(rows.loc[pos, "dose"])
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            std_val = pd.Series(vals).std() if len(vals) > 1 else 0
            mean_dose = pd.Series(doses).mean() if group_key != "Baseline" and not all(pd.isna(doses)) else None

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
    ax.set_ylabel(ylabel)
    metric_name = ylabel.split(" (")[0]
    fig.text(0.5, 0.94, f"{metric_name} During Different Pharmacological States", ha="center", fontsize=15)
    fig.text(0.5, 0.905, "(averaged across subjects 202, 203, 205, 221)", ha="center", fontsize=11)

    fig.subplots_adjust(top=0.83, bottom=0.16, right=0.97)
    legend_ax = fig.add_axes([0.77, 0.87, 0.16, 0.09])
    add_dose_gradient_legend(fig, ax=legend_ax)

    if save:
        output_dir = Path(output_dir) if output_dir else REPO_ROOT / "figures" / "metrics_vs_drugs"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{metric_prefix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def make_all_metrics_vs_drugs_figures(save=False, output_dir=None):
    figs = {}
    for metric_prefix, ylabel in METRICS:
        figs[metric_prefix] = make_metric_vs_drugs_figure(metric_prefix, ylabel, save=save, output_dir=output_dir)
    return figs


if __name__ == "__main__":
    make_all_metrics_vs_drugs_figures(save=True)