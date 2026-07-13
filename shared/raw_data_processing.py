"""
Shared raw-data processing utilities.

Converts a per-animal raw phase-summary CSV ({animal_id}_TDvAIC.csv) into a single
wide per-phase dataframe containing mean and standard deviation for AIC, TD, and
hemodynamic metrics (PP, HR, SmartPump, MAP), joined on phase identity rather than
row position.

These are all Impella-derived metrics — output now lives under
data/processed/impella_derived/summary_data/ (see PROJECT_DECISIONS.md for the
data/processed/{impella_derived,catheter_derived,ct_drug_effect} split).

Each raw CSV contains four column "blocks" side by side, each with its own
Number/number column. These blocks are NOT row-aligned by position -- they share
phase identity via the Number value itself. For example, every row tagged
Number=1 across all four blocks belongs to "phase 1," regardless of how many raw
timepoint rows each block happens to contain for that phase (AIC and TD are
sampled at different rates).

Column layout differs slightly between cohorts:
  - Normal cohort (202, 203, 205, 221): procedure block has Number, Med, Dose, P
  - Baseline cohort (103-218): procedure block has Number, Med, P (no Dose --
    these animals were never drugged, so there's no dose to record)

See PROJECT_DECISIONS.md for full provenance and reasoning.
"""
import pandas as pd
from pathlib import Path

# Column-index maps, confirmed against real column headers for all 12 animals.
COLUMN_MAPS = {
    "normal_cohort": {
        "aic_cols": [0, 1, 2],
        "td_cols": [4, 5, 6],
        "procedure_cols": [8, 9, 10, 11],
        "hemo_cols": [13, 14, 15, 16, 17, 18],
        "has_dose": True,
    },
    "baseline_cohort": {
        "aic_cols": [0, 1, 2],
        "td_cols": [4, 5, 6],
        "procedure_cols": [8, 9, 10],
        "hemo_cols": [12, 13, 14, 15, 16, 17],
        "has_dose": False,
    },
}


def _load_block(file_path, usecols, rename_map):
    """Load one column-block from the CSV, drop NA rows, rename to canonical names."""
    block = pd.read_csv(file_path, usecols=usecols).dropna()
    block = block.rename(columns=rename_map)
    block["phase_number"] = block["phase_number"].astype(int)
    return block


def load_animal_data(file_path, cohort):
    """
    Load and join all four blocks (AIC, TD, procedure, hemodynamics) from one
    raw CSV into a single wide per-phase mean/std dataframe.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw {animal_id}_TDvAIC.csv file.
    cohort : str
        Either "normal_cohort" or "baseline_cohort" -- selects the correct
        column-index map for this animal's raw file.

    Returns
    -------
    pd.DataFrame
        One row per phase, columns:
            phase_number, med, p_level, [dose if normal_cohort],
            AIC_mean, AIC_std, TD_mean, TD_std,
            PP_impella_mean, PP_impella_std, HR_impella_mean, HR_impella_std,
            SmartPump_mean, SmartPump_std, MAP_impella_mean, MAP_impella_std
    """
    if cohort not in COLUMN_MAPS:
        raise ValueError(f"Unknown cohort '{cohort}'. Expected one of {list(COLUMN_MAPS)}.")

    cmap = COLUMN_MAPS[cohort]
    file_path = Path(file_path)

    # --- AIC block ---
    aic_rename = {
        aic_label: canon
        for aic_label, canon in zip(
            pd.read_csv(file_path, usecols=cmap["aic_cols"], nrows=0).columns,
            ["phase_number", "time", "AIC"],
        )
    }
    aic = _load_block(file_path, cmap["aic_cols"], aic_rename)

    # --- TD block ---
    td_rename = {
        td_label: canon
        for td_label, canon in zip(
            pd.read_csv(file_path, usecols=cmap["td_cols"], nrows=0).columns,
            ["phase_number", "time", "TD"],
        )
    }
    td = _load_block(file_path, cmap["td_cols"], td_rename)

    # --- Procedure block (phase labels: Med, Dose [if present], P) ---
    proc_cols_header = pd.read_csv(file_path, usecols=cmap["procedure_cols"], nrows=0).columns
    if cmap["has_dose"]:
        proc_canon = ["phase_number", "med", "dose", "p_level"]
    else:
        proc_canon = ["phase_number", "med", "p_level"]
    proc_rename = dict(zip(proc_cols_header, proc_canon))
    procedure = _load_block(file_path, cmap["procedure_cols"], proc_rename)
    # One row per phase already (no repeated timepoints in this block) -- just
    # de-duplicate defensively in case the raw file has repeated label rows.
    procedure = procedure.drop_duplicates(subset="phase_number").set_index("phase_number")

    # --- Hemodynamics block ---
    # PP, HR, and MAP are renamed with an explicit _impella suffix because
    # catheter-derived data (dp/dt max/min, LVEDP) will ALSO produce PP, HR,
    # and MAP values from a different instrument (the pressure catheter, not
    # the Impella). Same underlying quantity, different source, potentially
    # different values -- so they must never share a column name. SmartPump
    # has no catheter-derived equivalent and is left unsuffixed.
    hemo_cols_header = pd.read_csv(file_path, usecols=cmap["hemo_cols"], nrows=0).columns
    hemo_canon = ["phase_number", "time", "PP_impella", "HR_impella", "SmartPump", "MAP_impella"]
    hemo_rename = dict(zip(hemo_cols_header, hemo_canon))
    hemo = _load_block(file_path, cmap["hemo_cols"], hemo_rename)

    # --- Aggregate each measurement block to one mean/std row per phase ---
    aic_stats = aic.groupby("phase_number")["AIC"].agg(["mean", "std"])
    aic_stats.columns = ["AIC_mean", "AIC_std"]

    td_stats = td.groupby("phase_number")["TD"].agg(["mean", "std"])
    td_stats.columns = ["TD_mean", "TD_std"]

    hemo_metrics = ["PP_impella", "HR_impella", "SmartPump", "MAP_impella"]
    hemo_stats = hemo.groupby("phase_number")[hemo_metrics].agg(["mean", "std"])
    hemo_stats.columns = [f"{metric}_{stat}" for metric, stat in hemo_stats.columns]

    # --- Join everything on phase_number ---
    result = procedure.join(aic_stats, how="outer")
    result = result.join(td_stats, how="outer")
    result = result.join(hemo_stats, how="outer")
    result = result.reset_index()

    return result


def process_animal(animal_id, raw_data_dir, cohort, output_dir):
    """
    Load one animal's raw CSV, compute the wide per-phase dataframe, and pickle it.

    Parameters
    ----------
    animal_id : str
        e.g. "202" or "103"
    raw_data_dir : str or Path
        Directory containing this animal's VBU folder (e.g. data/raw or
        data/raw/Baseline_CO)
    cohort : str
        "normal_cohort" or "baseline_cohort"
    output_dir : str or Path
        Where to save the resulting pickle.

    Returns
    -------
    pd.DataFrame
        The same dataframe that was pickled, for immediate use if needed.
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the animal's VBU folder (folder name includes a date suffix we don't
    # need to know in advance).
    matches = list(raw_data_dir.glob(f"VBU_{animal_id}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No VBU folder found for animal {animal_id} under {raw_data_dir}"
        )
    vbu_folder = matches[0]

    csv_path = vbu_folder / f"{animal_id}_TDvAIC.csv"
    if not csv_path.exists():
        # tolerate the legacy AICvTD naming if it hasn't been renamed yet
        alt_path = vbu_folder / f"{animal_id}_AICvTD.csv"
        if alt_path.exists():
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Could not find {animal_id}_TDvAIC.csv (or _AICvTD.csv) in {vbu_folder}"
            )

    df = load_animal_data(csv_path, cohort)
    output_path = output_dir / f"{animal_id}_phase_summary.pkl"
    df.to_pickle(output_path)
    return df


# Animal IDs and their cohort/data-directory assignments.
# Update raw_data_dir values to match your actual repo root if needed.
ANIMAL_REGISTRY = {
    # normal cohort
    "202": {"cohort": "normal_cohort", "subdir": ""},
    "203": {"cohort": "normal_cohort", "subdir": ""},
    "205": {"cohort": "normal_cohort", "subdir": ""},
    "221": {"cohort": "normal_cohort", "subdir": ""},
    # baseline cohort — lives one level deeper under Baseline_CO/
    "103": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "104": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "105": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "106": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "107": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "148": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "155": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
    "218": {"cohort": "baseline_cohort", "subdir": "Baseline_CO"},
}


def process_all_animals(repo_root, output_dir=None, animal_ids=None):
    """
    Process all animals (or a subset) and save per-animal phase-summary pickles.

    These pickles contain Impella-derived metrics only (AIC, TD, PP, HR,
    SmartPump, MAP) and are saved under
    {repo_root}/data/processed/impella_derived/summary_data/ by default.
    Catheter-derived metrics (dp/dt max/min, LVEDP) are handled by a separate
    module and saved under data/processed/catheter_derived/summary_data/ as
    their own per-animal pickles, joined to this data only at plot time.

    Parameters
    ----------
    repo_root : str or Path
        Root of the repo. Raw data is expected at {repo_root}/data/raw/.
    output_dir : str or Path, optional
        Where to save pickles. Defaults to
        {repo_root}/data/processed/impella_derived/summary_data/.
    animal_ids : list of str, optional
        Subset of animal IDs to process. Defaults to all 12.

    Returns
    -------
    dict
        Maps animal_id -> loaded DataFrame for each successfully processed animal.
    """
    repo_root = Path(repo_root)
    raw_root = repo_root / "data" / "raw"
    if output_dir is None:
        output_dir = repo_root / "data" / "processed" / "impella_derived" / "summary_data"
    if animal_ids is None:
        animal_ids = list(ANIMAL_REGISTRY.keys())

    results = {}
    errors = {}

    for animal_id in animal_ids:
        if animal_id not in ANIMAL_REGISTRY:
            print(f"[SKIP] {animal_id} not in ANIMAL_REGISTRY — skipping.")
            continue

        reg = ANIMAL_REGISTRY[animal_id]
        raw_data_dir = raw_root / reg["subdir"] if reg["subdir"] else raw_root

        try:
            df = process_animal(
                animal_id=animal_id,
                raw_data_dir=raw_data_dir,
                cohort=reg["cohort"],
                output_dir=output_dir,
            )
            results[animal_id] = df
            print(f"[OK]   {animal_id} — {len(df)} phases, pickled to {output_dir}")
        except Exception as e:
            errors[animal_id] = str(e)
            print(f"[FAIL] {animal_id} — {e}")

    if errors:
        print(f"\n{len(errors)} animal(s) failed: {list(errors.keys())}")
    else:
        print(f"\nAll {len(results)} animals processed successfully.")

    return results