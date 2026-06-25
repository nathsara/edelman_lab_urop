"""
shared/significance_testing.py

Standalone significance testing module. NOT called from run_pipeline.py.
Run manually to reproduce the statistical decisions that informed the pipeline design:

    python -m shared.significance_testing

Two tests are implemented:

1. P3 vs P6 significance test
   For each drug state (Baseline, Nitro, Phen, Dobu) and each metric
   (AIC, TD, PP, HR, SmartPump, MAP), tests whether P3 and P6 measurements
   differ significantly across the 4-animal normal cohort.
   Method: pool percent-change-from-baseline values across all 4 animals
   separately for P3 and P6, then run an independent two-sample t-test.

2. Washout vs Baseline significance test
   For Washout1 (post-Nitro) and Washout2 (post-Phen) only — washouts
   following Dobu or Esmo are excluded since we only analyze up to Dobu.
   For each washout period and each metric (AIC, TD, PP, HR, SmartPump, MAP),
   tests whether the washout measurements differ significantly from baseline.
   Method: for each animal compute percent-change-from-baseline for that
   washout phase, pool those values across all 4 animals, run a one-sample
   t-test against zero.

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

# Metrics to test
METRICS = ["AIC", "TD", "PP", "HR", "SmartPump", "MAP"]

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

def _load_animal_summaries(processed_dir, animal_ids=NORMAL_COHORT_IDS):
    """
    Load per-animal phase-summary pickles and filter out Esmo phases.

    Returns
    -------
    dict mapping animal_id -> DataFrame
    """
    processed_dir = Path(processed_dir)
    data = {}
    for animal_id in animal_ids:
        pkl_path = processed_dir / f"{animal_id}_phase_summary.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Phase summary pickle not found for animal {animal_id} at {pkl_path}.\n"
                f"Run process_all_animals() first via run_pipeline.py."
            )
        df = pd.read_pickle(pkl_path)
        # Exclude Esmo phases entirely
        df = df[df["med"] != "Esmo"].copy()
        data[animal_id] = df
    return data


def _get_baseline_mean(df, metric):
    """
    Return the grand mean of a metric across both baseline phases (P3 + P6).
    Used as the reference value for percent-change computation.
    """
    baseline_rows = df[df["med"] == "Baseline"]
    if baseline_rows.empty:
        raise ValueError("No Baseline phase found in dataframe.")
    return baseline_rows[f"{metric}_mean"].mean()


def _pct_change_from_baseline(value, baseline_mean):
    """Compute percent change from baseline."""
    if baseline_mean == 0:
        return np.nan
    return ((value - baseline_mean) / baseline_mean) * 100


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

def run_p3_p6_test(processed_dir):
    """
    Test whether P3 and P6 measurements differ significantly across
    drug states and metrics, pooled across all 4 normal-cohort animals.
    """
    print("\n" + "=" * 70)
    print("TEST 1: P3 vs P6 significance test")
    print(f"Animals: {NORMAL_COHORT_IDS}")
    print(f"Drug states: {DRUG_STATES}")
    print(f"Metrics: {METRICS}")
    print(f"Alpha: {ALPHA}")
    print("=" * 70)

    animal_data = _load_animal_summaries(processed_dir)
    any_significant = False

    for drug_state in DRUG_STATES:
        print(f"\n--- {drug_state} ---")
        for metric in METRICS:
            col = f"{metric}_mean"
            p3_values, p6_values = [], []

            for animal_id, df in animal_data.items():
                baseline_mean = _get_baseline_mean(df, metric)
                state_rows = df[df["med"] == drug_state]

                for _, row in state_rows.iterrows():
                    if pd.isna(row[col]):
                        continue
                    pct = _pct_change_from_baseline(row[col], baseline_mean)
                    if row["p_level"] == "P3":
                        p3_values.append(pct)
                    elif row["p_level"] == "P6":
                        p6_values.append(pct)

            if len(p3_values) < 2 or len(p6_values) < 2:
                print(f"  {metric:>12}: insufficient data "
                      f"(P3 n={len(p3_values)}, P6 n={len(p6_values)}) — skipped")
                continue

            t_stat, p_val = stats.ttest_ind(p3_values, p6_values, equal_var=False)
            significant = p_val < ALPHA
            if significant:
                any_significant = True

            flag = " *** SIGNIFICANT ***" if significant else ""
            print(f"  {metric:>12}: t={t_stat:+.3f}, p={p_val:.4f} "
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

def run_washout_test(processed_dir):
    """
    Test whether Washout1 (post-Nitro) and Washout2 (post-Phen) differ
    significantly from baseline, pooled across all 4 normal-cohort animals.

    Washouts following Dobu or Esmo are excluded — only the two washouts
    within the Baseline -> Nitro -> Washout -> Phen -> Washout -> Dobu
    protocol window are tested.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Washout vs Baseline significance test")
    print(f"Animals: {NORMAL_COHORT_IDS}")
    print(f"Metrics: {METRICS}")
    print(f"Alpha: {ALPHA}")
    print("Note: only Washout1 (post-Nitro) and Washout2 (post-Phen) tested.")
    print("      Post-Dobu/Esmo washouts excluded.")
    print("=" * 70)

    animal_data = _load_animal_summaries(processed_dir)

    washout_labels = {0: "Washout1 (post-Nitro)", 1: "Washout2 (post-Phen)"}

    for washout_idx in range(2):
        print(f"\n--- {washout_labels[washout_idx]} ---")

        for metric in METRICS:
            col = f"{metric}_mean"
            pct_changes = []

            for animal_id, df in animal_data.items():
                washouts = _get_washouts_in_scope(df)

                if washout_idx >= len(washouts):
                    print(f"  {metric:>12}: animal {animal_id} has no "
                          f"{washout_labels[washout_idx]} in scope — skipped")
                    continue

                washout_row = washouts.iloc[washout_idx]
                if pd.isna(washout_row[col]):
                    continue

                baseline_mean = _get_baseline_mean(df, metric)
                pct = _pct_change_from_baseline(washout_row[col], baseline_mean)
                pct_changes.append(pct)

            if len(pct_changes) < 2:
                print(f"  {metric:>12}: insufficient data "
                      f"(n={len(pct_changes)}) — skipped")
                continue

            t_stat, p_val = stats.ttest_1samp(pct_changes, popmean=0)
            significant = p_val < ALPHA
            mean_pct = np.mean(pct_changes)

            flag = " *** SIGNIFICANT ***" if significant else ""
            print(f"  {metric:>12}: mean pct change={mean_pct:+.2f}%, "
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
        description="Run significance tests for P3/P6 and washout vs baseline."
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "summary_data",
        help="Directory containing per-animal phase-summary pickles.",
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
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "summary_data" / "significance_results.txt",
        help="Where to save results as a text file.",
    )
    args = parser.parse_args()

    # Tee output to both console and txt file
    tee = Tee(args.output)
    sys.stdout = tee

    try:
        if args.test in ("p3_p6", "both"):
            run_p3_p6_test(args.processed_dir)
        if args.test in ("washout", "both"):
            run_washout_test(args.processed_dir)
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nResults saved to: {args.output}")