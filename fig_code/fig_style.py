"""
fig-code/fig_style.py

Shared plotting utilities for all "Final Figures" scripts (per the Jul 17
2026 notes) -- NOT a pipeline module, just a small helper imported directly
by each standalone fig-code/*.py script for a consistent look.

Color convention (fixed, per notes): Nitro=red, Phen=green, Dobu=blue.
Baseline=black. Washout=gray, no gradient.
Dose gradient: lighter shade = lower normalized dose, darker = higher --
uses the ACTUAL continuous 0-1 normalized dose value (from
drug_dose_normalization.py), not just two fixed light/dark shades.
Font: >=14pt everywhere, per notes guideline 1.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

DRUG_BASE_COLOR = {
    "Nitro": "#b50c00",  # red
    "Phen": "#007a2f",   # green
    "Dobu": "#0028db",   # blue
}
BASELINE_COLOR = "black"
WASHOUT_COLOR = "gray"


def dose_gradient_color(drug, normalized_dose):
    """
    Returns an RGBA color for `drug` at a given normalized_dose (0-1),
    lighter for lower dose, darker/fully-saturated for higher dose.
    normalized_dose may be NaN (e.g. Washout) -- caller should use
    WASHOUT_COLOR/BASELINE_COLOR instead in that case, not this function.
    """
    if drug not in DRUG_BASE_COLOR:
        raise ValueError(f"Unknown drug {drug!r}, expected one of {list(DRUG_BASE_COLOR)}")
    base_rgb = np.array(mcolors.to_rgb(DRUG_BASE_COLOR[drug]))
    white = np.array([1.0, 1.0, 1.0])
    # blend: dose=0 -> mostly white (light tint), dose=1 -> full base color
    frac = 0.85 * float(normalized_dose) + 0.15  # keep a visible tint even at dose=0
    frac = max(0.15, min(1.0, frac))
    rgb = white * (1 - frac) + base_rgb * frac
    return tuple(rgb)


def phase_color(med, normalized_dose):
    """
    Convenience: returns the correct color for any phase given its `med`
    and (already-normalized 0-1) `dose` value -- Baseline=black,
    Washout=gray (no gradient), Esmo=gray (excluded from analysis but
    colored defensively if ever plotted), drug phases=gradient.
    """
    if med == "Baseline":
        return BASELINE_COLOR
    if med in ("Washout", "Esmo"):
        return WASHOUT_COLOR
    return dose_gradient_color(med, normalized_dose)


def merge_p3_p6(df, value_cols):
    """
    Merges P6+P3 rows into one averaged row per drug-dose occurrence (or per
    Baseline/Washout occurrence, which have no dose split). Used for the
    "averaged" figure versions, per user: P3 vs P6 showed no significant
    difference (confirmed via run_p3_p6_test), so it's valid to average them
    together rather than plot as separate points -- Nitro collapses from 4
    points to 2, Baseline from 2 to 1, etc.

    Grouping key: (group, dose) -- P6 and P3 rows within the same drug
    occurrence share the IDENTICAL normalized dose value (both came from the
    same dose-log row via time-matching in drug_dose_normalization.py), so
    grouping on dose directly identifies "same occurrence" without needing to
    guess a P6/P3 pairing rule. NaN dose (Baseline, Washout -- no dose split)
    is treated as one shared bucket per group tag, since there's only ever
    one occurrence to merge in those cases.

    Naturally handles asymmetric cases (e.g. animal 205's Dobu-high, which
    has only a P6 row, no matching P3) -- a "group" of one row just averages
    to itself, no special-casing needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'group' and 'dose' columns (already tagged, e.g. via
        _load_and_tag), sorted by phase_number so occurrence order (low dose
        before high dose) is preserved.
    value_cols : list of str
        Columns to average within each merged occurrence.

    Returns
    -------
    pd.DataFrame
        One row per merged occurrence: 'group', 'dose' (mean, though
        identical within the group by construction), plus averaged
        value_cols. Order preserved (first-appearance order of each
        (group, dose) key).
    """
    df = df.copy()
    # Rounded to avoid two rows that SHOULD be identical (same source dose
    # value) differing by float noise; NaN dose (Baseline/Washout) shares
    # one bucket per group tag.
    df["_dose_key"] = df["dose"].apply(lambda d: "NA" if pd.isna(d) else round(d, 8))

    merged_rows = []
    for (group_val, _dose_key), sub in df.groupby(["group", "_dose_key"], sort=False):
        row = {"group": group_val, "dose": sub["dose"].mean()}
        for col in value_cols:
            row[col] = sub[col].mean()
        merged_rows.append(row)
    return pd.DataFrame(merged_rows)


def add_dose_gradient_legend(fig, ax=None, loc="lower left", label_fontsize=11, tick_fontsize=10, title_fontsize=11):
    """
    Adds the small 'Normalized Drug Dosage 0->1' gradient legend shown in
    the notes for figs 3/5/6 -- one horizontal gradient bar per drug.
    """
    from matplotlib.patches import Rectangle
    ax = ax or fig.add_axes([0.15, 0.02, 0.2, 0.12])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(DRUG_BASE_COLOR))
    n_steps = 40
    for i, drug in enumerate(DRUG_BASE_COLOR):
        y = len(DRUG_BASE_COLOR) - i - 1
        for j in range(n_steps):
            frac = j / n_steps
            ax.add_patch(Rectangle((frac, y), 1 / n_steps, 0.8,
                                    color=dose_gradient_color(drug, frac), linewidth=0))
        ax.text(1.05, y + 0.4, drug, va="center", fontsize=label_fontsize)
    ax.text(0, -0.4, "0", fontsize=tick_fontsize)
    ax.text(1, -0.4, "1", fontsize=tick_fontsize)
    ax.set_title("Normalized Drug Dosage", fontsize=title_fontsize)
    ax.axis("off")
    return ax