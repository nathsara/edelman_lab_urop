"""
shared/percent_change_processing.py

Generates percent-change-from-baseline summary files for BOTH Impella-
derived and catheter-derived data. Normal cohort only (202/203/205/221) --
baseline-cohort animals have no drug states at all, "percent change from
baseline" doesn't map onto their structure.

BASELINE METHOD: each phase is normalized against its OWN P-level's
baseline mean -- P6 phases (including Baseline_P6 itself) compared to
Baseline_P6's mean; P3 phases compared to Baseline_P3's mean. This is the
SAME method significance_testing.py's P3-vs-P6 test already uses --
_get_baseline_mean() is imported and reused directly from that module
rather than reimplemented, so there's one source of truth for what counts
as "baseline" per P-level. NOT the combined P3+P6 average the washout test
uses -- see PROJECT_DECISIONS.md's flagged methodology note for the full
reasoning and the decision history (this was briefly considered, then
reversed back to per-P-level).

CONSEQUENCE: Baseline rows get a real value (0.0 -- a phase compared to its
own baseline is 0% change, by definition), not NaN. Different from the
NaN seen in the P3-vs-P6 t-test itself, which was a statistical degenerate
case (zero-variance input), not a data or logic problem here.

STD HANDLING: propagated via simple linear scaling --
pct_std = std / baseline_mean * 100 -- valid since percent-change is a
linear transform of the mean. Not used in any current figures, but kept as
real data for completeness. A more rigorous approach would also propagate
uncertainty in the baseline mean itself (delta method for a ratio of two
random variables) -- not done here, flagged as a simplification, not
currently needed.

DEPENDS ON drug dose normalization having already run: the `dose` column
is carried through UNCHANGED from the source summary pickle, so this MUST
run AFTER shared.drug_dose_normalization, or the carried-through `dose`
column would still hold the original categorical label instead of the
normalized 0-1 value.

Output: NEW files, saved alongside (NOT overwriting) the existing summary
pickles, in the SAME folder -- no new subfolder:
  {animal_id}_catheter_percdiff.pkl       (next to {animal_id}_catheter_summary.pkl)
  {animal_id}_phase_summary_percdiff.pkl  (next to {animal_id}_phase_summary.pkl)

HALTS on any error -- unlike drug_dose_normalization.py's deliberately-soft
warn-and-continue behavior, this computes real analytical values (not just
carrying through metadata), so a silent partial result would be worse than
stopping.
"""

from pathlib import Path
import pandas as pd

from .significance_testing import _get_baseline_mean

NORMAL_COHORT_IDS = ["202", "203", "205", "221"]

CATHETER_METRICS = ["dpdt_max", "dpdt_min", "lvedp", "PP_catheter", "MAP_catheter", "HR_catheter"]
IMPELLA_METRICS = ["AIC", "TD", "PP_impella", "HR_impella", "SmartPump", "MAP_impella"]


def _pct_change_columns(df, metric):
    """
    Returns (pctchange_mean_list, pctchange_std_list) for one metric, using
    each row's own p_level to select the matching per-P-level baseline mean.
    """
    pct_means = []
    pct_stds = []
    for _, row in df.iterrows():
        baseline_mean = _get_baseline_mean(df, metric, p_level=row["p_level"])
        value = row[f"{metric}_mean"]
        std = row[f"{metric}_std"]
        pct_means.append((value - baseline_mean) / baseline_mean * 100)
        pct_stds.append(std / baseline_mean * 100)
    return pct_means, pct_stds


def build_percdiff_df(summary_path, metrics):
    """
    Loads a summary pickle and returns a new DataFrame: all metadata columns
    (phase_number, med, dose, p_level) carried through unchanged, plus
    {metric}_pctchange_mean / {metric}_pctchange_std for each metric.
    """
    df = pd.read_pickle(summary_path)
    result = df[["phase_number", "med", "dose", "p_level"]].copy()
    for metric in metrics:
        pct_means, pct_stds = _pct_change_columns(df, metric)
        result[f"{metric}_pctchange_mean"] = pct_means
        result[f"{metric}_pctchange_std"] = pct_stds
    return result


def generate_all_percent_change_files(repo_root):
    """
    Generates percent-change summary files for all 4 normal-cohort animals,
    both catheter-derived and Impella-derived.
    """
    repo_root = Path(repo_root)
    catheter_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"
    impella_dir = repo_root / "data" / "processed" / "impella_derived" / "summary_data"

    for animal_id in NORMAL_COHORT_IDS:
        catheter_summary = catheter_dir / f"{animal_id}_catheter_summary.pkl"
        if catheter_summary.exists():
            percdiff_df = build_percdiff_df(catheter_summary, CATHETER_METRICS)
            out_path = catheter_dir / f"{animal_id}_catheter_percdiff.pkl"
            percdiff_df.to_pickle(out_path)
            print(f"  [OK] {out_path.name} saved ({len(percdiff_df)} phases).")
        else:
            print(f"  [SKIP] {catheter_summary.name} not found.")

        impella_summary = impella_dir / f"{animal_id}_phase_summary.pkl"
        if impella_summary.exists():
            percdiff_df = build_percdiff_df(impella_summary, IMPELLA_METRICS)
            out_path = impella_dir / f"{animal_id}_phase_summary_percdiff.pkl"
            percdiff_df.to_pickle(out_path)
            print(f"  [OK] {out_path.name} saved ({len(percdiff_df)} phases).")
        else:
            print(f"  [SKIP] {impella_summary.name} not found.")