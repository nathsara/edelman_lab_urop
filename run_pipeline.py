"""
run_pipeline.py

Single entry point to reproduce all results from raw data.
Run from the repo root:

    python run_pipeline.py

All figures are saved to figures/{stage_name}/ by default.
To override the figures destination:

    python run_pipeline.py --figures_dir /some/other/path

Note: significance testing (P3/P6, washout vs baseline) is handled separately
in shared/significance_testing.py and is not part of this pipeline. Pipeline
design reflects the decisions informed by those tests.
"""

import argparse
import sys
from pathlib import Path

from shared.raw_data_processing import process_all_animals
from shared.catheter_phase_config import ANIMAL_PHASES
from shared.catheter_data_init import (
    find_vbu_lvv_folder, create_data_df, create_raw_phase_data,
    create_aop_data_df, create_aop_phase_data,
    create_coarse_phase_data, create_coarse_aop_phase_data, COARSE_DRUGS,
)
from shared.catheter_data_processing import combined_phase_data
from shared.significance_testing import run_p3_p6_test, run_washout_test, Tee
from ct_drug_effect_analysis.process import process_coarse_phase
import pandas as pd
import plots

# Catheter-derived pipeline scope: normal cohort ONLY. Baseline cohort is out
# of scope for catheter-derived metrics -- see PROJECT_DECISIONS.md.
CATHETER_ANIMAL_IDS = ["202", "203", "205", "221"]

# Metrics plotted in Stage 0d, matching PPT slides 27 (dp/dt max), 28
# (dp/dt min), 29 (LVEDP). PP/MAP/HR-catheter have no legacy PPT figure to
# compare against (see PROJECT_DECISIONS.md) -- plotted using the same
# adapter with corrected units, for the user's own manual value comparison
# against their previously-saved data rather than a visual PPT diff.
CATHETER_PLOT_METRICS = [
    ("dpdt_max", "dp/dt max", "mmHG/s"),
    ("dpdt_min", "dp/dt min", "mmHG/s"),
    ("lvedp", "lvedp", "mmHG/s"),
    ("PP_catheter", "PP catheter", "mmHg"),
    ("MAP_catheter", "MAP catheter", "mmHg"),
    ("HR_catheter", "HR catheter", "bpm"),
]


def _generate_catheter_raw_data(repo_root, force=False):
    """
    Stage 0b: for each normal-cohort animal, load raw VBU ECG/LVP data and
    slice it into per-phase pickles (catheter_data_init.create_data_df +
    create_raw_phase_data), then do the same for AOP
    (create_aop_data_df + create_aop_phase_data), using the SAME per-phase
    timestamps as ECG/LVP so all three signals land at identical
    per-phase granularity.

    Skips animals whose expected output already exists on disk, unless
    force=True -- this step is expensive (~2.5 min + ~15 min across all 4
    animals for ECG/LVP, plus a further AOP pass) and its output doesn't
    change between runs unless the raw data itself changes, so re-running it
    every single pipeline invocation isn't worth the cost by default.

    HALTS on any error -- no per-animal try/except/continue. A partial or
    missing set of per-phase pickles must never be silently left in place.
    """
    repo_root = Path(repo_root)
    raw_root = repo_root / "data" / "raw"
    raw_phase_root = repo_root / "data" / "processed" / "catheter_derived" / "raw_phase_data"

    for animal_id in CATHETER_ANIMAL_IDS:
        animal_dir = raw_phase_root / animal_id
        raw_hd_pickle = animal_dir / f"raw_hd_data_{animal_id}.pkl"

        matches = list(raw_root.glob(f"VBU_{animal_id}_*"))
        if not matches:
            raise FileNotFoundError(f"No VBU folder found for animal {animal_id} under {raw_root}")
        vbu_folder = matches[0]

        if raw_hd_pickle.exists() and not force:
            print(f"[SKIP] {animal_id} raw combined ECG/LVP data already exists at {raw_hd_pickle}")
        else:
            lvv_folder = find_vbu_lvv_folder(vbu_folder, animal_id)
            create_data_df(lvv_folder, animal_id, output_dir=animal_dir)
            print(f"[OK]   {animal_id} raw combined ECG/LVP data -> {raw_hd_pickle}")

        expected_phase_files = []
        for label in ANIMAL_PHASES[animal_id]:
            expected_phase_files.append(animal_dir / f"{label}_ecg_raw.pkl")
            expected_phase_files.append(animal_dir / f"{label}_lvp_raw.pkl")

        if all(f.exists() for f in expected_phase_files) and not force:
            print(f"[SKIP] {animal_id} per-phase raw data already exists ({len(expected_phase_files) // 2} phases)")
            continue

        csv_path = vbu_folder / f"{animal_id}_TDvAIC.csv"
        if not csv_path.exists():
            alt_path = vbu_folder / f"{animal_id}_AICvTD.csv"
            if alt_path.exists():
                csv_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Could not find {animal_id}_TDvAIC.csv (or _AICvTD.csv) in {vbu_folder}"
                )

        create_raw_phase_data(
            animal_summary_csv=csv_path,
            raw_hd_data_pickle=raw_hd_pickle,
            animal_id=animal_id,
            output_dir=animal_dir,
        )
        print(f"[OK]   {animal_id} per-phase raw data ({len(ANIMAL_PHASES[animal_id])} phases) -> {animal_dir}")

    for animal_id in CATHETER_ANIMAL_IDS:
        # Separate loop from the ECG/LVP one above (rather than folding in) --
        # keeps this additive: AOP extraction never touches or re-runs the
        # already-confirmed-working ECG/LVP step, even if AOP fails or needs
        # re-running later.
        animal_dir = raw_phase_root / animal_id
        raw_aop_pickle = animal_dir / f"raw_aop_data_{animal_id}.pkl"

        matches = list(raw_root.glob(f"VBU_{animal_id}_*"))
        if not matches:
            raise FileNotFoundError(f"No VBU folder found for animal {animal_id} under {raw_root}")
        vbu_folder = matches[0]

        if raw_aop_pickle.exists() and not force:
            print(f"[SKIP] {animal_id} raw combined AOP data already exists at {raw_aop_pickle}")
        else:
            lvv_folder = find_vbu_lvv_folder(vbu_folder, animal_id)
            create_aop_data_df(lvv_folder, animal_id, output_dir=animal_dir)
            print(f"[OK]   {animal_id} raw combined AOP data -> {raw_aop_pickle}")

        expected_aop_files = [animal_dir / f"{label}_aop_raw.pkl" for label in ANIMAL_PHASES[animal_id]]
        if all(f.exists() for f in expected_aop_files) and not force:
            print(f"[SKIP] {animal_id} per-phase AOP data already exists ({len(expected_aop_files)} phases)")
            continue

        csv_path = vbu_folder / f"{animal_id}_TDvAIC.csv"
        if not csv_path.exists():
            alt_path = vbu_folder / f"{animal_id}_AICvTD.csv"
            if alt_path.exists():
                csv_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Could not find {animal_id}_TDvAIC.csv (or _AICvTD.csv) in {vbu_folder}"
                )

        create_aop_phase_data(
            animal_summary_csv=csv_path,
            raw_aop_data_pickle=raw_aop_pickle,
            animal_id=animal_id,
            output_dir=animal_dir,
        )
        print(f"[OK]   {animal_id} per-phase AOP data ({len(ANIMAL_PHASES[animal_id])} phases) -> {animal_dir}")


def _generate_catheter_summaries(repo_root, force=False):
    """
    Stage 0c: run the dp/dt max/min + LVEDP + PP/MAP/HR-catheter
    signal-processing pipeline for each normal-cohort animal, producing
    {animal_id}_catheter_summary.pkl (mean/std per phase) AND, alongside it,
    the full per-beat data for all six metrics under
    data/processed/catheter_derived/per_beat_data/{animal_id}/ (one pickle
    per phase per metric -- the final, fully-processed series each summary
    mean/std is computed from, not intermediate half-baked versions).

    Skips animals whose summary pickle already exists, unless force=True.

    HALTS on any error (combined_phase_data itself halts per-phase, and no
    try/except wraps the per-animal loop here either) -- a partial or
    missing summary pickle must never be silently left in place.
    """
    repo_root = Path(repo_root)
    raw_phase_root = repo_root / "data" / "processed" / "catheter_derived" / "raw_phase_data"
    summary_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"

    for animal_id in CATHETER_ANIMAL_IDS:
        summary_path = summary_dir / f"{animal_id}_catheter_summary.pkl"
        if summary_path.exists() and not force:
            print(f"[SKIP] {animal_id} catheter summary already exists at {summary_path}")
            continue

        combined_phase_data(
            animal_id=animal_id,
            raw_phase_dir=raw_phase_root / animal_id,
            output_dir=summary_dir,
            plot=False,
            per_beat_dir=repo_root / "data" / "processed" / "catheter_derived" / "per_beat_data" / animal_id,
        )


IMPELLA_SIGNIFICANCE_METRICS = ["AIC", "TD", "PP_impella", "HR_impella", "SmartPump", "MAP_impella"]


def _run_impella_significance_testing(repo_root):
    """
    Stage 0a2: runs both significance tests (P3 vs P6, washout vs baseline)
    against the Impella-derived summary data, for all six Impella-derived
    metrics -- the mirror of _run_catheter_significance_testing() below, for
    the other dataset. Same shared/significance_testing.py, no code changes
    needed there either; same Tee-redirection replication.

    Always runs fresh (no skip-if-exists), same rationale as the catheter
    version -- fast, pure statistics, and naturally overwrites prior results.

    Output: data/processed/impella_derived/summary_data/significance_results.txt
    """
    repo_root = Path(repo_root)
    processed_dir = repo_root / "data" / "processed" / "impella_derived" / "summary_data"
    output_path = processed_dir / "significance_results.txt"

    tee = Tee(output_path)
    real_stdout = sys.stdout
    sys.stdout = tee
    try:
        run_p3_p6_test(processed_dir, IMPELLA_SIGNIFICANCE_METRICS, "phase_summary")
        run_washout_test(processed_dir, IMPELLA_SIGNIFICANCE_METRICS, "phase_summary")
    finally:
        sys.stdout = real_stdout
        tee.close()
    print(f"Impella-derived significance testing results saved to: {output_path}")


CATHETER_SIGNIFICANCE_METRICS = ["dpdt_max", "dpdt_min", "lvedp", "PP_catheter", "MAP_catheter", "HR_catheter"]


def _run_catheter_significance_testing(repo_root):
    """
    Stage 0c2: runs both significance tests (P3 vs P6, washout vs baseline)
    against the catheter-derived summary data, for all six catheter-derived
    metrics. shared/significance_testing.py needed NO code changes for this
    -- it was already built fully generic over metric names/dataset location
    (see its module docstring); only the CLI's __main__ block (argument
    parsing + Tee output redirection) wasn't factored into an importable
    function, so that's replicated here directly.

    Always runs fresh (no skip-if-exists) -- this is fast (pure statistics
    over already-computed summary data, no signal processing), and Tee opens
    its output file in write mode, so this naturally overwrites whatever
    results were there before (e.g. an earlier partial run covering only
    dpdt_max/dpdt_min/lvedp, from before PP/HR/MAP-catheter existed) with
    the full six-metric result -- exactly the desired "more comprehensive,
    replaces previous" behavior, with no special-case code needed for it.

    Output: data/processed/catheter_derived/summary_data/significance_results.txt
    """
    repo_root = Path(repo_root)
    processed_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"
    output_path = processed_dir / "significance_results.txt"

    tee = Tee(output_path)
    real_stdout = sys.stdout
    sys.stdout = tee
    try:
        run_p3_p6_test(processed_dir, CATHETER_SIGNIFICANCE_METRICS, "catheter_summary")
        run_washout_test(processed_dir, CATHETER_SIGNIFICANCE_METRICS, "catheter_summary")
    finally:
        sys.stdout = real_stdout
        tee.close()
    print(f"Catheter-derived significance testing results saved to: {output_path}")


def _plot_catheter_summaries(repo_root):
    """
    Stage 0d: plots each normal-cohort animal's dp/dt max, dp/dt min, and
    LVEDP data, using the UNCHANGED legacy plot functions (via
    plots.plot_catheter_summary).

    This IS the real, permanent Stage 4 catheter-metrics plotting step, not a
    disposable QA-only script -- it stays in the pipeline going forward. What
    WILL change later, once the new data is confirmed correct against PPT
    slides 27-29: the plotting code itself gets a real style pass (legend,
    layout, grouping by metric across animals per the PPT slide-26 notes,
    etc.), and output moves from on-screen windows to saved files under
    figures/04_hemodynamic_metrics_vs_td_aic_diff/. None of that restyling
    happens here -- this function just calls the legacy plot functions as-is
    against the new pipeline's data.

    Uses plt.show() (inherited unchanged from legacy plots.py) rather than
    saving to disk -- this BLOCKS execution until each of the 12 figures
    (4 animals x 3 metrics) is manually closed. Deliberate for now, so every
    figure gets actively looked at during development. No skip-if-exists
    logic either, since nothing is persisted to disk to check against -- this
    stage just always (re)displays from whatever summary pickles exist.
    """
    repo_root = Path(repo_root)
    summary_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"

    for animal_id in CATHETER_ANIMAL_IDS:
        summary_path = summary_dir / f"{animal_id}_catheter_summary.pkl"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No catheter summary found for animal {animal_id} at {summary_path}. "
                f"Stage 0c should have produced this -- run the pipeline from the start."
            )
        summary_df = pd.read_pickle(summary_path)

        for metric, metric_label, ylabel in CATHETER_PLOT_METRICS:
            print(f"Plotting {animal_id} — {metric_label}...")
            plots.plot_catheter_summary(summary_df, animal_id, metric, metric_label, ylabel=ylabel)


def _generate_coarse_raw_data(repo_root, force=False):
    """
    Stage 0e: for each normal-cohort animal, slice the already-extracted
    full-animal raw ECG/LVP/AOP data (raw_hd_data_{animal_id}.pkl /
    raw_aop_data_{animal_id}.pkl, from Stage 0b) into coarse whole-drug-state
    windows (Nitro/Phen/Dobu -- Stage 2's confirmed scope), using REAL
    recorded timestamps from data/raw/drug_start_end_times.csv. Does not
    re-read the VBU CSVs -- just re-slices data already on disk.

    Skips animals whose expected coarse output already exists, unless
    force=True. HALTS on any error.
    """
    repo_root = Path(repo_root)
    raw_phase_root = repo_root / "data" / "processed" / "catheter_derived" / "raw_phase_data"
    drug_csv = repo_root / "data" / "raw" / "drug_start_end_times.csv"

    for animal_id in CATHETER_ANIMAL_IDS:
        animal_dir = raw_phase_root / animal_id
        raw_hd_pickle = animal_dir / f"raw_hd_data_{animal_id}.pkl"
        raw_aop_pickle = animal_dir / f"raw_aop_data_{animal_id}.pkl"

        expected_files = []
        for drug in COARSE_DRUGS:
            label = f"{animal_id}_{drug}"
            expected_files += [
                animal_dir / f"{label}_ecg_raw.pkl",
                animal_dir / f"{label}_lvp_raw.pkl",
                animal_dir / f"{label}_aop_raw.pkl",
            ]

        if all(f.exists() for f in expected_files) and not force:
            print(f"[SKIP] {animal_id} coarse raw data already exists ({len(COARSE_DRUGS)} drug states)")
            continue

        create_coarse_phase_data(drug_csv, raw_hd_pickle, animal_id, output_dir=animal_dir)
        create_coarse_aop_phase_data(drug_csv, raw_aop_pickle, animal_id, output_dir=animal_dir)
        print(f"[OK]   {animal_id} coarse raw data ({len(COARSE_DRUGS)} drug states) -> {animal_dir}")


def _generate_ct_drug_effect_data(repo_root, force=False):
    """
    Stage 0e: Stage 2's data layer -- continuous-time drug-effect trajectory
    data (per-beat + percent-change-from-baseline + rolling window=240
    average) for all six metrics, per animal x drug state. Code lives in
    ct_drug_effect_analysis/process.py; output lands in
    data/processed/ct_drug_effect/{animal_id}/.

    Skips animal-drug pairs whose output already exists (spot-checked via
    one representative file), unless force=True. HALTS on any error.
    """
    repo_root = Path(repo_root)
    raw_phase_root = repo_root / "data" / "processed" / "catheter_derived" / "raw_phase_data"
    catheter_summary_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"
    output_root = repo_root / "data" / "processed" / "ct_drug_effect"

    for animal_id in CATHETER_ANIMAL_IDS:
        for drug in COARSE_DRUGS:
            label = f"{animal_id}_{drug}"
            output_dir = output_root / animal_id
            check_file = output_dir / f"{label}_hr_catheter_MA.pkl"

            if check_file.exists() and not force:
                print(f"[SKIP] {label} continuous trajectory data already exists")
                continue

            process_coarse_phase(
                animal_id=animal_id,
                drug=drug,
                raw_phase_dir=raw_phase_root / animal_id,
                catheter_summary_dir=catheter_summary_dir,
                output_dir=output_dir,
            )


def main(repo_root, figures_dir, force_catheter=False):

    # ── Stage 0: Raw data processing ─────────────────────────────────────────
    print("=" * 60)
    print("STAGE 0: Raw data processing")
    print("=" * 60)
    process_all_animals(repo_root=repo_root)

    # ── Stage 0a: Impella-derived significance testing ───────────────────────
    print("=" * 60)
    print("STAGE 0a: Impella-derived significance testing (P3 vs P6, washout vs baseline)")
    print("=" * 60)
    _run_impella_significance_testing(repo_root=repo_root)

    # ── Stage 0b: Catheter-derived raw/phase data generation ────────────────
    print("=" * 60)
    print("STAGE 0b: Catheter-derived raw/phase data generation")
    print("=" * 60)
    _generate_catheter_raw_data(repo_root=repo_root, force=force_catheter)

    # ── Stage 0c: Catheter-derived signal processing (dp/dt, LVEDP) ─────────
    print("=" * 60)
    print("STAGE 0c: Catheter-derived signal processing")
    print("=" * 60)
    _generate_catheter_summaries(repo_root=repo_root, force=force_catheter)

    # ── Stage 0d: Catheter-derived significance testing ──────────────────────
    print("=" * 60)
    print("STAGE 0d: Catheter-derived significance testing (P3 vs P6, washout vs baseline)")
    print("=" * 60)
    _run_catheter_significance_testing(repo_root=repo_root)

    # ── Stage 0e: Coarse (whole-drug-state) raw data for continuous trajectories
    print("=" * 60)
    print("STAGE 0e: Coarse raw data (Nitro/Phen/Dobu whole-drug-state windows)")
    print("=" * 60)
    _generate_coarse_raw_data(repo_root=repo_root, force=force_catheter)

    # ── Stage 0f: Continuous-time drug-effect trajectory data (Stage 2 data) ─
    print("=" * 60)
    print("STAGE 0f: Continuous-time drug-effect trajectory data")
    print("=" * 60)
    _generate_ct_drug_effect_data(repo_root=repo_root, force=force_catheter)

    # ── Stage 0g: Catheter-derived metric plotting ───────────────────────────
    print("=" * 60)
    print("STAGE 0g: Catheter-derived metric plotting — close each figure to continue")
    print("=" * 60)
    _plot_catheter_summaries(repo_root=repo_root)

    # ── Stage 1: Baseline analysis ────────────────────────────────────────────
    # TODO: uncomment once written
    # from 01_baseline_analysis.process import run_baseline_analysis
    # from 01_baseline_analysis.plot import plot_baseline_analysis
    # run_baseline_analysis(repo_root=repo_root)
    # plot_baseline_analysis(figures_dir=figures_dir / "01_baseline_analysis")

    # ── Stage 2: Drug effect trajectory analysis ──────────────────────────────
    # Data layer already built and wired in above (Stages 0d/0e). Only the
    # actual figure generation remains -- deliberately deferred, see
    # ct_drug_effect_analysis/process.py's module docstring.
    # TODO: uncomment once plot.py is written
    # from ct_drug_effect_analysis.plot import plot_drug_trajectory
    # plot_drug_trajectory(figures_dir=figures_dir / "ct_drug_effect_analysis")

    # ── Stage 3: TD vs AIC across drug states ────────────────────────────────
    # TODO: uncomment once written
    # from 03_td_vs_aic_drug_states.process import run_td_vs_aic
    # from 03_td_vs_aic_drug_states.plot import plot_td_vs_aic
    # run_td_vs_aic(repo_root=repo_root)
    # plot_td_vs_aic(figures_dir=figures_dir / "03_td_vs_aic_drug_states")

    # ── Stage 4: Hemodynamic & catheter metrics vs TD-AIC diff ───────────────
    # TODO: uncomment once written
    # from 04_hemodynamic_metrics_vs_td_aic_diff.process import run_hemo_metrics
    # from 04_hemodynamic_metrics_vs_td_aic_diff.plot import plot_hemo_metrics
    # run_hemo_metrics(repo_root=repo_root)
    # plot_hemo_metrics(figures_dir=figures_dir / "04_hemodynamic_metrics_vs_td_aic_diff")

    print("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument(
        "--figures_dir",
        type=Path,
        default=None,
        help="Where to save figures. Defaults to {repo_root}/figures/",
    )
    parser.add_argument(
        "--force_catheter",
        action="store_true",
        help="Regenerate catheter-derived raw/phase data and summaries even if "
             "already present on disk. Default: skip animals whose expected "
             "output already exists (this stage is expensive, ~17.5 min across "
             "all 4 animals for the raw/phase step alone).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    figures_dir = args.figures_dir or repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    main(repo_root=repo_root, figures_dir=figures_dir, force_catheter=args.force_catheter)