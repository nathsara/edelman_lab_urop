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


def add_dose_gradient_legend(fig, ax=None, rect=None, bar_height=0.35,
                              fontsize=8, title_fontsize=9):
    """
    Adds a compact 'Normalized Drug Dosage 0->1' gradient legend -- one
    THIN horizontal gradient bar per drug (bars are deliberately thin so
    the legend doesn't eat up plot space).

    rect: [left, bottom, width, height] in figure coordinates for the
    legend axes (only used if `ax` is not already provided by the caller).
    Callers should pass a small rect (e.g. width ~0.12-0.16,
    height ~0.08-0.12) and position it wherever it fits without
    overlapping the main plot/other text.
    """
    from matplotlib.patches import Rectangle
    if ax is None:
        rect = rect or [0.83, 0.03, 0.14, 0.09]
        ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(DRUG_BASE_COLOR))
    n_steps = 30
    for i, drug in enumerate(DRUG_BASE_COLOR):
        y = len(DRUG_BASE_COLOR) - i - 1
        y_bar = y + (1 - bar_height) / 2
        for j in range(n_steps):
            frac = j / n_steps
            ax.add_patch(Rectangle((frac, y_bar), 1 / n_steps, bar_height,
                                    color=dose_gradient_color(drug, frac), linewidth=0))
        ax.text(1.06, y + 0.5, drug, va="center", fontsize=fontsize)
    ax.text(0, -0.35, "0", fontsize=fontsize - 1, ha="center")
    ax.text(1, -0.35, "1", fontsize=fontsize - 1, ha="center")
    ax.set_title("Norm. Dose", fontsize=title_fontsize, pad=2)
    ax.axis("off")
    return ax