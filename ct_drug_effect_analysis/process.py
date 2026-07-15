"""
ct_drug_effect_analysis/process.py

Stage 2: continuous-time drug-effect trajectory data.

Produces, per animal x drug state (Nitro/Phen/Dobu -- Stage 2's confirmed
scope; Baseline/Washout/Esmo out of scope) x metric (dp/dt max, dp/dt min,
LVEDP, PP-catheter, MAP-catheter, HR-catheter):

  {animal_id}_{drug}_{metric}.pkl           -- full per-beat data
  {animal_id}_{drug}_{metric}_percdiff.pkl  -- time + percent-change-from-baseline
  {animal_id}_{drug}_{metric}_MA.pkl        -- time + rolling(window=240) average

under data/processed/ct_drug_effect/{animal_id}/.

Reuses shared.catheter_data_processing.process_phase() and
process_phase_hemodynamics() UNCHANGED -- both are generic over any label
given matching raw files, and correctly route through the (corrected)
catheter_phase_config.PHASES_REQUIRING_EXTRA_PROCESSING check. Verified
directly against legacy process.py's real hardcoded list that none of the
12 Nitro/Phen/Dobu x animal windows need the segment-splitting path -- see
PROJECT_DECISIONS.md.

Baseline reference for percent-change: the Baseline_0_P6 row (index 0 after
sorting by phase_number) already sitting in each animal's existing
{animal_id}_catheter_summary.pkl -- matches legacy's continuous-trajectory
code exactly (unconditional index [0], no P6/P3 distinction; there IS no
P-level concept in continuous per-beat catheter data -- P-level is an
Impella pump-setting metadata field, not a property of individual
heartbeats). Baseline itself is never reprocessed as a coarse window.

Rolling average window: 240 for ALL six metrics, per explicit user
instruction -- legacy graphing.py was inconsistent (window=120 in most
blocks, window=240 in what looks like the final iteration for some metrics,
with several draft iterations of the same plots sitting in one file).
min_periods=1 (partial-window average for the first <240 beats, rather than
leaving them NaN) is a judgment call made here, not something the user
confirmed explicitly -- flagged. Legacy itself was inconsistent on this
point too (mixed across different blocks/draft iterations); min_periods=1
was used in what appeared to be the newest/final window=240 blocks, so it's
used uniformly here. Flag if you want NaN-padding instead.

HALTS on any error -- no per-animal/per-drug try/except/continue. Consistent
with every other stage in this pipeline.
"""

from pathlib import Path
import pandas as pd

from shared.catheter_data_processing import process_phase, process_phase_hemodynamics

COARSE_DRUGS = ("Nitro", "Phen", "Dobu")

SMA_WINDOW = 240

# metric key -> (baseline mean column in {animal_id}_catheter_summary.pkl,
#                 per-beat pickle filename suffix produced by process_phase /
#                 process_phase_hemodynamics, value column inside that per-beat
#                 DataFrame)
METRIC_SPECS = {
    "dpdt_max":    {"summary_mean_col": "dpdt_max_mean",     "per_beat_suffix": "dpdt_max",     "value_col": "dpdt_max"},
    "dpdt_min":    {"summary_mean_col": "dpdt_min_mean",     "per_beat_suffix": "dpdt_min",     "value_col": "dpdt_min"},
    "lvedp":       {"summary_mean_col": "lvedp_mean",        "per_beat_suffix": "lvedp",        "value_col": "lvedp"},
    "pp_catheter": {"summary_mean_col": "PP_catheter_mean",  "per_beat_suffix": "pp_catheter",  "value_col": "pulse_pressure"},
    "map_catheter":{"summary_mean_col": "MAP_catheter_mean", "per_beat_suffix": "map_catheter", "value_col": "map"},
    "hr_catheter": {"summary_mean_col": "HR_catheter_mean",  "per_beat_suffix": "hr_catheter",  "value_col": "heart_rate"},
}


def get_baseline_means(catheter_summary_path):
    """
    Reads {animal_id}_catheter_summary.pkl and returns a dict of
    {metric: baseline_mean}, using the Baseline_0_P6 row (first row after
    sorting by phase_number) as the single reference for all six metrics.
    """
    summary_df = pd.read_pickle(catheter_summary_path).sort_values("phase_number")
    baseline_row = summary_df.iloc[0]
    return {metric: baseline_row[spec["summary_mean_col"]] for metric, spec in METRIC_SPECS.items()}


def process_coarse_phase(animal_id, drug, raw_phase_dir, catheter_summary_dir, output_dir):
    """
    For one animal + one coarse drug state: runs full per-beat processing
    for all six metrics, then computes percent-change-from-baseline and a
    window=240 rolling average for each.

    Parameters
    ----------
    animal_id : str
    drug : str
        "Nitro", "Phen", or "Dobu".
    raw_phase_dir : str or Path
        Directory containing {animal_id}_{drug}_ecg_raw.pkl /
        {animal_id}_{drug}_lvp_raw.pkl / {animal_id}_{drug}_aop_raw.pkl, as
        produced by catheter_data_init.create_coarse_phase_data /
        create_coarse_aop_phase_data.
    catheter_summary_dir : str or Path
        Directory containing {animal_id}_catheter_summary.pkl (for the
        baseline reference).
    output_dir : str or Path
        Where to save per-beat / percdiff / MA pickles for this animal.
    """
    label = f"{animal_id}_{drug}"
    raw_phase_dir = Path(raw_phase_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catheter_summary_path = Path(catheter_summary_dir) / f"{animal_id}_catheter_summary.pkl"
    baseline_means = get_baseline_means(catheter_summary_path)

    print(f"[{label}] Processing dp/dt max/min + LVEDP...")
    process_phase(label, raw_phase_dir, plot=False, per_beat_dir=output_dir)

    print(f"[{label}] Processing PP/MAP/HR-catheter...")
    process_phase_hemodynamics(label, raw_phase_dir, per_beat_dir=output_dir)

    for metric, spec in METRIC_SPECS.items():
        per_beat_path = output_dir / f"{label}_{spec['per_beat_suffix']}.pkl"
        per_beat_df = pd.read_pickle(per_beat_path)

        baseline_mean = baseline_means[metric]
        value_col = spec["value_col"]
        percdiff = [((v - baseline_mean) / baseline_mean) * 100 for v in per_beat_df[value_col]]
        percdiff_df = pd.DataFrame({"time": per_beat_df["time"].values, f"{metric}_percdiff": percdiff})
        percdiff_df.to_pickle(output_dir / f"{label}_{metric}_percdiff.pkl")

        percdiff_df["SMA"] = percdiff_df[f"{metric}_percdiff"].rolling(window=SMA_WINDOW, min_periods=1).mean()
        ma_df = percdiff_df[["time", "SMA"]].copy()
        ma_df.to_pickle(output_dir / f"{label}_{metric}_MA.pkl")

        print(f"[{label}] {metric}: {len(per_beat_df)} beats -> percdiff + MA saved")


def process_all_coarse_phases(repo_root, animal_ids=("202", "203", "205", "221"), drugs=COARSE_DRUGS):
    """
    Runs process_coarse_phase() for every animal x drug combination.
    HALTS on any error -- a partial/incomplete set of trajectory data must
    never be silently left in place.
    """
    repo_root = Path(repo_root)
    raw_phase_root = repo_root / "data" / "processed" / "catheter_derived" / "raw_phase_data"
    catheter_summary_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"
    output_root = repo_root / "data" / "processed" / "ct_drug_effect"

    for animal_id in animal_ids:
        for drug in drugs:
            process_coarse_phase(
                animal_id=animal_id,
                drug=drug,
                raw_phase_dir=raw_phase_root / animal_id,
                catheter_summary_dir=catheter_summary_dir,
                output_dir=output_root / animal_id,
            )