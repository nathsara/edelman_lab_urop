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
from pathlib import Path

from shared.raw_data_processing import process_all_animals


def main(repo_root, figures_dir):

    # ── Stage 0: Raw data processing ─────────────────────────────────────────
    print("=" * 60)
    print("STAGE 0: Raw data processing")
    print("=" * 60)
    process_all_animals(repo_root=repo_root)

    # ── Stage 1: Baseline analysis ────────────────────────────────────────────
    # TODO: uncomment once written
    # from 01_baseline_analysis.process import run_baseline_analysis
    # from 01_baseline_analysis.plot import plot_baseline_analysis
    # run_baseline_analysis(repo_root=repo_root)
    # plot_baseline_analysis(figures_dir=figures_dir / "01_baseline_analysis")

    # ── Stage 2: Drug effect trajectory analysis ──────────────────────────────
    # TODO: uncomment once written
    # from 02_drug_effect_trajectory.process import run_drug_trajectory
    # from 02_drug_effect_trajectory.plot import plot_drug_trajectory
    # run_drug_trajectory(repo_root=repo_root)
    # plot_drug_trajectory(figures_dir=figures_dir / "02_drug_effect_trajectory")

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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    figures_dir = args.figures_dir or repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    main(repo_root=repo_root, figures_dir=figures_dir)