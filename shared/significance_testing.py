"""
shared/significance_testing.py

Standalone significance testing module. NOT called from run_pipeline.py.

Reusable across datasets — Impella-derived (AIC, TD, PP_impella, HR_impella,
SmartPump, MAP_impella) and catheter-derived (dpdt_max, dpdt_min, lvedp,
PP_catheter, HR_catheter, MAP_catheter) pickles both go through the same
P3-vs-P6 and washout-vs-baseline logic below. Nothing about the tests
themselves is dataset-specific -- only the processed_dir, metric column
names, and pickle filename suffix change between the two.

Run manually, pointing at whichever dataset you want to test:

    # Impella-derived (default metrics/suffix shown explicitly for clarity)
    python -m shared.significance_testing \\
        --processed_dir data/processed/impella_derived/summary_data \\
        --metrics AIC TD PP_impella HR_impella SmartPump MAP_impella \\
        --pickle_suffix phase_summary

    # Catheter-derived
    python -m shared.significance_testing \\
        --processed_dir data/processed/catheter_derived/summary_data \\
        --metrics dpdt_max dpdt_min lvedp PP_catheter HR_catheter MAP_catheter \\
        --pickle_suffix catheter_summary

--processed_dir is REQUIRED (no default) -- there is no single "right" dataset
for this module to assume. --metrics and --pickle_suffix default to the
Impella-derived values so the original invocation shape still works if you
only pass --processed_dir.

Two tests are implemented:

1. P3 vs P6 significance test
   For each drug state (Baseline, Nitro, Phen, Dobu) and each metric in
   --metrics, tests whether P3 and P6 measurements differ significantly
   across the 4-animal normal cohort.
   Method: pool percent-change-from-own-p-level-baseline values across all
   4 animals separately for P3 and P6, then run an independent two-sample
   t-test. P6 measurements are normalized against P6 baseline mean; P3
   measurements are normalized against P3 baseline mean. This ensures we
   are testing whether P3 and P6 respond differently to drugs, not just
   whether they start from different absolute values.

2. Washout vs Baseline significance test
   For Washout1 (post-Nitro) and Washout2 (post-Phen) only — washouts
   following Dobu or Esmo are excluded since we only analyze up to Dobu.
   For each washout period and each metric in --metrics, tests whether the
   washout measurements differ significantly from baseline.
   Method: for each animal compute percent-change-from-baseline (combined
   P3+P6 baseline mean) for that washout phase, pool those values across
   all 4 animals, run a one-sample t-test against zero.
   Note: combined baseline mean is appropriate here since washout phases
   are not split by P-level — we are testing whether the animal returned
   to its overall baseline state.

Results printed to console AND saved to a .txt file for reference.
Decisions informed by these tests are baked directly into the pipeline
design rather than computed at runtime.

See PROJECT_DECISIONS.md for full reasoning.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys


# ── Constants ─────────────────────────────────────────────────────────────────

NORMAL_COHORT_IDS = ["202", "203", "205", "221"]

# Drug states to include — Esmo excluded entirely from all analyses
DRUG_STATES = ["Baseline", "Nitro", "Phen", "Dobu"]

# Default metrics/pickle suffix — Impella-derived dataset. Pass --metrics and
# --pickle_suffix explicitly to run against catheter-derived data instead.
DEFAULT_METRICS = ["AIC", "TD", "PP_impella", "HR_impella", "SmartPump", "MAP_impella"]
DEFAULT_PICKLE_SUFFIX = "phase_summary"

ALPHA = 0.05


# ── Output helper ─────────────────────────────────────────────────────────────

class Tee:
    """Writes output to both console and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_animal_summaries(processed_dir, pickle_suffix, animal_ids=NORMAL_COHORT_IDS):
    """
    Load per-animal phase-summary pickles and filter out Esmo phases.

    Parameters
    ----------
    processed_dir : str or Path
        Directory containing {animal_id}_{pickle_suffix}.pkl files.
    pickle_suffix : str
        e.g. "phase_summary" (Impella-derived) or "catheter_summary"
        (catheter-derived).
    animal_ids : list of str
        Defaults to the 4-animal normal cohort.

    Returns
    -------
    dict mapping animal_id -> DataFrame
    """
    processed_dir = Path(processed_dir)
    data = {}
    for animal_id in animal_ids:
        pkl_path = processed_dir / f"{animal_id}_{pickle_suffix}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Pickle not found for animal {animal_id} at {pkl_path}.\n"
                f"Run the appropriate processing step first (raw_data_processing.py "
                f"for Impella-derived, or the catheter-derived equivalent)."
            )
        df = pd.read_pickle(pkl_path)
        # Exclude Esmo phases entirely
        df = df[df["med"] != "Esmo"].copy()
        data[animal_id] = df
    return data


def _get_baseline_mean(df, metric, p_level=None):
    """
    Return the baseline mean for a given metric.

    Parameters
    ----------
    df : DataFrame
        Per-animal phase-summary dataframe.
    metric : str
        Metric column prefix (e.g. "AIC", "TD", "PP_impella", "dpdt_max").
        Matched exactly against "{metric}_mean" -- no case-folding or
        fuzzy matching, since e.g. PP_impella and PP_catheter are distinct
        quantities from different instruments and must never be conflated.
    p_level : str or None
        If provided (e.g. "P3" or "P6"), returns the baseline mean for that
        specific P-level only. Used in the P3 vs P6 test so that each group
        is normalized against its own baseline, not a combined mean.
        If None, returns the combined mean across all baseline phases (P3 + P6
        together). Used in the washout test where no P-level split is needed.

    Returns
    -------
    float
    """
    baseline_rows = df[df["med"] == "Baseline"]
    if p_level is not None:
        baseline_rows = baseline_rows[baseline_rows["p_level"] == p_level]
    if baseline_rows.empty:
        raise ValueError(
            f"No Baseline phase found for p_level={p_level!r}. "
            f"Check that the data contains the expected P-level."
        )
    return baseline_rows[f"{metric}_mean"].mean()


def _pct_change(value, reference_mean):
    """Compute percent change from a reference mean."""
    if reference_mean == 0 or pd.isna(reference_mean):
        return np.nan
    return ((value - reference_mean) / reference_mean) * 100


def _get_washouts_in_scope(df):
    """
    Return only the washout phases that fall within the Baseline->Dobu protocol
    window — i.e. Washout1 (post-Nitro) and Washout2 (post-Phen).

    Strategy: find the phase_number of the last Dobu phase, then keep only
    Washout rows whose phase_number is less than or equal to that cutoff.
    This excludes any post-Esmo or post-Dobu washouts that may exist in the
    raw data even after Esmo rows themselves have been filtered out.
    """
    dobu_rows = df[df["med"] == "Dobu"]
    if dobu_rows.empty:
        return pd.DataFrame()

    dobu_last_phase = dobu_rows["phase_number"].max()
    washouts = df[
        (df["med"] == "Washout") &
        (df["phase_number"] <= dobu_last_phase)
    ].sort_values("phase_number")

    # Keep only first two (Washout1, Washout2) — protocol has exactly two
    # within scope; guard against edge cases.
    return washouts.head(2)


# ── Test 1: P3 vs P6 ─────────────────────────────────────────────────────────

def run_p3_p6_test(processed_dir, metrics, pickle_suffix):
    """
    Test whether P3 and P6 measurements differ significantly across
    drug states and metrics, pooled across all 4 normal-cohort animals.

    Each P-level is normalized against its own baseline mean:
      - P6 drug-state values normalized against P6 baseline mean
      - P3 drug-state values normalized against P3 baseline mean

    This tests whether P3 and P6 respond differently to the drug,
    not merely whether they start from different absolute values.
    """
    print("\n" + "=" * 70)
    print("TEST 1: P3 vs P6 significance test")
    print(f"Processed dir: {processed_dir}")
    print(f"Animals: {NORMAL_COHORT_IDS}")
    print(f"Drug states: {DRUG_STATES}")
    print(f"Metrics: {metrics}")
    print(f"Alpha: {ALPHA}")
    print("Normalization: each P-level normalized against its own baseline mean")
    print("=" * 70)

    animal_data = _load_animal_summaries(processed_dir, pickle_suffix)
    any_significant = False

    for drug_state in DRUG_STATES:
        print(f"\n--- {drug_state} ---")
        for metric in metrics:
            col = f"{metric}_mean"
            p3_values, p6_values = [], []

            for animal_id, df in animal_data.items():
                state_rows = df[df["med"] == drug_state]

                for _, row in state_rows.iterrows():
                    if pd.isna(row[col]):
                        continue

                    p_level = row["p_level"]

                    # Normalize against this P-level's own baseline mean
                    try:
                        baseline_mean = _get_baseline_mean(df, metric, p_level=p_level)
                    except ValueError:
                        continue

                    pct = _pct_change(row[col], baseline_mean)
                    if pd.isna(pct):
                        continue

                    if p_level == "P3":
                        p3_values.append(pct)
                    elif p_level == "P6":
                        p6_values.append(pct)

            if len(p3_values) < 2 or len(p6_values) < 2:
                print(f"  {metric:>14}: insufficient data "
                      f"(P3 n={len(p3_values)}, P6 n={len(p6_values)}) — skipped")
                continue

            t_stat, p_val = stats.ttest_ind(p3_values, p6_values, equal_var=False)
            significant = p_val < ALPHA
            if significant:
                any_significant = True

            flag = " *** SIGNIFICANT ***" if significant else ""
            print(f"  {metric:>14}: t={t_stat:+.3f}, p={p_val:.4f} "
                  f"(P3 n={len(p3_values)}, P6 n={len(p6_values)}){flag}")

    print("\n" + "=" * 70)
    if any_significant:
        print("CONCLUSION: At least one significant P3 vs P6 difference found.")
        print("Review flagged results above before deciding whether to drop")
        print("P-level distinction.")
    else:
        print("CONCLUSION: No significant P3 vs P6 differences found across any")
        print("drug state or metric. P-level distinction can be dropped.")
    print("=" * 70)


# ── Test 2: Washout vs Baseline ───────────────────────────────────────────────

def run_washout_test(processed_dir, metrics, pickle_suffix):
    """
    Test whether Washout1 (post-Nitro) and Washout2 (post-Phen) differ
    significantly from baseline, pooled across all 4 normal-cohort animals.

    Washout phases are not split by P-level, so normalization uses the
    combined P3+P6 baseline mean per animal.

    Washouts following Dobu or Esmo are excluded — only the two washouts
    within the Baseline -> Nitro -> Washout -> Phen -> Washout -> Dobu
    protocol window are tested.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Washout vs Baseline significance test")
    print(f"Processed dir: {processed_dir}")
    print(f"Animals: {NORMAL_COHORT_IDS}")
    print(f"Metrics: {metrics}")
    print(f"Alpha: {ALPHA}")
    print("Note: only Washout1 (post-Nitro) and Washout2 (post-Phen) tested.")
    print("      Post-Dobu/Esmo washouts excluded.")
    print("      Normalization: combined P3+P6 baseline mean per animal.")
    print("=" * 70)

    animal_data = _load_animal_summaries(processed_dir, pickle_suffix)

    washout_labels = {0: "Washout1 (post-Nitro)", 1: "Washout2 (post-Phen)"}

    for washout_idx in range(2):
        print(f"\n--- {washout_labels[washout_idx]} ---")

        for metric in metrics:
            col = f"{metric}_mean"
            pct_changes = []

            for animal_id, df in animal_data.items():
                washouts = _get_washouts_in_scope(df)

                if washout_idx >= len(washouts):
                    print(f"  {metric:>14}: animal {animal_id} has no "
                          f"{washout_labels[washout_idx]} in scope — skipped")
                    continue

                washout_row = washouts.iloc[washout_idx]
                if pd.isna(washout_row[col]):
                    continue

                # Combined P3+P6 baseline mean — washout phases are not
                # split by P-level so no per-level normalization needed
                baseline_mean = _get_baseline_mean(df, metric, p_level=None)
                pct = _pct_change(washout_row[col], baseline_mean)
                if not pd.isna(pct):
                    pct_changes.append(pct)

            if len(pct_changes) < 2:
                print(f"  {metric:>14}: insufficient data "
                      f"(n={len(pct_changes)}) — skipped")
                continue

            t_stat, p_val = stats.ttest_1samp(pct_changes, popmean=0)
            significant = p_val < ALPHA
            mean_pct = np.mean(pct_changes)

            flag = " *** SIGNIFICANT ***" if significant else ""
            print(f"  {metric:>14}: mean pct change={mean_pct:+.2f}%, "
                  f"t={t_stat:+.3f}, p={p_val:.4f} "
                  f"(n={len(pct_changes)}){flag}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE:")
    print("  Significant result -> washout did NOT fully return to baseline.")
    print("  For Washout1: if significant, normalize Phen against Washout1 mean")
    print("                instead of original Baseline.")
    print("  For Washout2: if significant, normalize Dobu against Washout2 mean")
    print("                instead of original Baseline.")
    print("  Nitro always normalized against original Baseline regardless.")
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run significance tests for P3/P6 and washout vs baseline. "
                     "Reusable across datasets -- point --processed_dir, --metrics, "
                     "and --pickle_suffix at either impella_derived or "
                     "catheter_derived summary data."
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="Directory containing per-animal summary pickles. e.g. "
             "data/processed/impella_derived/summary_data or "
             "data/processed/catheter_derived/summary_data. No default -- "
             "must be specified explicitly.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Metric column prefixes to test (matched exactly against "
             f"'{{metric}}_mean'). Defaults to Impella-derived metrics: "
             f"{DEFAULT_METRICS}. For catheter-derived data pass e.g. "
             f"dpdt_max dpdt_min lvedp PP_catheter HR_catheter MAP_catheter.",
    )
    parser.add_argument(
        "--pickle_suffix",
        type=str,
        default=DEFAULT_PICKLE_SUFFIX,
        help=f"Pickle filename suffix -- files are expected at "
             f"{{processed_dir}}/{{animal_id}}_{{pickle_suffix}}.pkl. "
             f"Defaults to '{DEFAULT_PICKLE_SUFFIX}' (Impella-derived). "
             f"Use 'catheter_summary' for catheter-derived data.",
    )
    parser.add_argument(
        "--test",
        choices=["p3_p6", "washout", "both"],
        default="both",
        help="Which test(s) to run. Default: both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save results as a text file. Defaults to "
             "{processed_dir}/significance_results.txt.",
    )
    args = parser.parse_args()

    output_path = args.output or (args.processed_dir / "significance_results.txt")

    tee = Tee(output_path)
    sys.stdout = tee

    try:
        if args.test in ("p3_p6", "both"):
            run_p3_p6_test(args.processed_dir, args.metrics, args.pickle_suffix)
        if args.test in ("washout", "both"):
            run_washout_test(args.processed_dir, args.metrics, args.pickle_suffix)
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nResults saved to: {output_path}")