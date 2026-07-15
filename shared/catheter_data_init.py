"""
shared/catheter_data_init.py

Prepares raw per-timepoint ECG/LVP data and per-phase time windows, feeding
the catheter-derived signal-processing pipeline (dp/dt max/min, LVEDP).

Two raw inputs per animal:

  1. The animal's VBU*Calibrated*LVV* subfolder -- many small per-timepoint
     CSVs, each containing (among other columns) a timestamp, LVP, and
     external ECG column. COLUMN POSITION for these three signals is NOT
     consistent across files -- confirmed empirically (e.g. LVP has been seen
     at column 45 in one file, 49 in another). Columns are therefore resolved
     by HEADER NAME here, never by position. See _resolve_signal_columns().
     If a file uses header text not yet in SIGNAL_ALIASES, this raises with
     the file's actual header shown, rather than silently reading the wrong
     column.

  2. The animal's {animal_id}_TDvAIC.csv -- same raw file raw_data_processing.py
     parses for Impella-derived metrics. This module reads the TD and
     Procedure blocks directly (not via raw_data_processing's aggregated
     output) because it needs each phase's raw per-timepoint start/end
     timestamps to slice the ECG/LVP data, not the phase-level mean/std.
     Column positions here (TD block: 4,5,6; label block: 9,10,11) match the
     already-validated normal_cohort layout in raw_data_processing.py's
     COLUMN_MAPS -- position-based reading is safe for this specific file,
     unlike the VBU log files above.

AOP (arterial/aortic pressure waveform) extraction, for PP/HR/MAP-catheter:
     ADDED after the dp/dt/LVEDP pipeline was confirmed working end-to-end.
     Legacy code (gen_map_pp_data.py) sourced AOP-derived PP/HR/MAP at a
     coarser, 6-phases-per-animal granularity, using a DIFFERENT timestamp
     source (Drugs/drug_start_end_times.csv) than everything else in this
     pipeline. DELIBERATE DEPARTURE FROM LEGACY, per user: AOP is instead
     sliced using the EXACT SAME per-fine-phase timestamps as ECG/LVP (i.e.
     the same timestamps_from_animal() call against the same TDvAIC.csv), so
     PP/HR/MAP-catheter comes out at the same per-phase (P-level x dose)
     granularity as dp/dt max/min and LVEDP, joinable into the same summary
     row per phase. Column position for AOP was seen to vary by animal in
     legacy code (column 43 for 202/203, 44 for 205/221) -- resolved by
     HEADER NAME here instead, same defensive pattern as ECG/LVP, which
     sidesteps that inconsistency entirely rather than needing a per-animal
     column map.

Scope: 4-animal normal cohort only (202, 203, 205, 221).
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


# Canonical signal name -> acceptable raw header spellings seen so far.
# Extend this if a new file uses header text not yet listed here --
# _resolve_signal_columns raises loudly (showing the actual header) rather
# than guessing, so any new variant surfaces immediately instead of silently
# reading the wrong column.
SIGNAL_ALIASES = {
    "timestamp": ["RTlog_Timestamp"],
    "lvp": ["LVP"],
    "ecg": ["Ext_ECG"],
}

# Separate alias map for the AOP extraction pass (see module docstring).
# "AOP" is the only spelling seen so far, per legacy gen_map_pp_data.py's
# split_into_columns() else-branch -- NOT YET CONFIRMED against a real file
# header the way LVP/ECG were. If a real file uses different header text,
# _resolve_signal_columns will raise loudly showing the actual header rather
# than silently reading the wrong column -- add the real spelling here if so.
AOP_SIGNAL_ALIASES = {
    "timestamp": ["RTlog_Timestamp"],
    "aop": ["AOP"],
}


def _resolve_signal_columns(header_columns, aliases=SIGNAL_ALIASES):
    """
    Map each canonical signal name to whichever column in THIS file's header
    matches one of its known alias spellings. Resolves by name, never by
    position, since position of these columns is not consistent file to file.

    Raises with the full actual header shown if a signal can't be found or
    matches more than one column, rather than guessing.
    """
    header_list = list(header_columns)
    resolved = {}
    for canonical_name, alias_list in aliases.items():
        matches = [col for col in header_list if col in alias_list]
        if not matches:
            raise ValueError(
                f"Could not find a column for '{canonical_name}' (looked for "
                f"{alias_list}) in header: {header_list}. If this file uses a "
                f"header spelling not yet seen, add it to SIGNAL_ALIASES "
                f"(or AOP_SIGNAL_ALIASES, if this is the AOP extraction pass)."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous match for '{canonical_name}': found {matches} in "
                f"header: {header_list}."
            )
        resolved[canonical_name] = matches[0]
    return resolved


def find_vbu_lvv_folder(vbu_parent_folder, animal_id):
    """
    Locate an animal's calibrated-LVV subfolder without needing to know its
    exact name in advance. Naming has been observed to vary between animals
    -- e.g. "VBU00221Calibrated_LVV" (with underscore) vs
    "VBU00205CalibratedLVV" (without) -- so this matches tolerantly on the
    animal ID plus "calibrated" and "lvv" appearing somewhere in the folder
    name, case-insensitive, rather than requiring one exact pattern.

    Parameters
    ----------
    vbu_parent_folder : str or Path
        The animal's top-level VBU_{animal_id}_{date} folder (i.e. the one
        containing the calibrated-LVV subfolder alongside the TDvAIC.csv).
    animal_id : str

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError if no subfolder matches.
    ValueError if more than one subfolder matches (naming too ambiguous to
    resolve automatically -- pass the exact path directly to create_data_df
    instead in that case).
    """
    vbu_parent_folder = Path(vbu_parent_folder)
    candidates = [
        p for p in vbu_parent_folder.iterdir()
        if p.is_dir()
        and animal_id in p.name
        and "calibrated" in p.name.lower()
        and "lvv" in p.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No calibrated-LVV subfolder found under {vbu_parent_folder} "
            f"for animal {animal_id}. Contents: {[p.name for p in vbu_parent_folder.iterdir()]}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple possible calibrated-LVV subfolders found for animal "
            f"{animal_id}: {[p.name for p in candidates]}. Pass the exact "
            f"path directly to create_data_df instead."
        )
    return candidates[0]


def create_data_df(vbu_lvv_folder, animal_id, output_dir):
    """
    Load and concatenate every per-timepoint CSV in an animal's
    calibrated-LVV folder into one raw timestamp/LVP/ECG dataframe.

    Parameters
    ----------
    vbu_lvv_folder : str or Path
        Path to the animal's calibrated-LVV subfolder. Use
        find_vbu_lvv_folder() to locate this without needing to know its
        exact name.
    animal_id : str
        e.g. "202"
    output_dir : str or Path
        Where to save the resulting pickle.

    Returns
    -------
    pd.DataFrame
        Columns: RTlog_Timestamp, LVP, Ext_ECG -- one row per raw timepoint,
        concatenated across every CSV in the folder (sorted filename order).
    """
    vbu_lvv_folder = Path(vbu_lvv_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(vbu_lvv_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {vbu_lvv_folder}")

    file_frames = []
    for file_path in csv_files:
        header = pd.read_csv(file_path, nrows=0).columns
        resolved = _resolve_signal_columns(header)
        usecols = [resolved["timestamp"], resolved["lvp"], resolved["ecg"]]

        file_df = pd.read_csv(file_path, usecols=usecols).dropna()
        # Normalize to canonical names regardless of this file's exact header
        # text, so downstream code never has to think about per-file naming.
        file_df = file_df.rename(columns={
            resolved["timestamp"]: "RTlog_Timestamp",
            resolved["lvp"]: "LVP",
            resolved["ecg"]: "Ext_ECG",
        })
        file_frames.append(file_df)

    # Concatenate once at the end rather than inside the loop. Legacy code did
    # pd.concat repeatedly inside the loop, which is O(n^2) -- each call
    # re-copies everything accumulated so far. Collecting frames in a list and
    # concatenating once is O(n) and produces an IDENTICAL result (same rows,
    # same order); this is a pure efficiency fix, not a behavior change.
    raw_hd_data = pd.concat(file_frames, ignore_index=True)

    output_path = output_dir / f"raw_hd_data_{animal_id}.pkl"
    raw_hd_data.to_pickle(output_path)
    return raw_hd_data


def create_raw_phase_data(animal_summary_csv, raw_hd_data_pickle, animal_id, output_dir):
    """
    Slice the raw per-timepoint LVP/ECG data (from create_data_df) into one
    LVP file and one ECG file per phase, using each phase's start/end
    timestamps read from the animal's TDvAIC.csv.

    Parameters
    ----------
    animal_summary_csv : str or Path
        Path to {animal_id}_TDvAIC.csv.
    raw_hd_data_pickle : str or Path
        Path to the raw_hd_data_{animal_id}.pkl produced by create_data_df.
    animal_id : str
    output_dir : str or Path
        Where to save the per-phase {label}_lvp_raw.pkl / {label}_ecg_raw.pkl
        files.
    """
    raw_hd_data_pickle = Path(raw_hd_data_pickle)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(raw_hd_data_pickle)
    timestamps, lvp, ecg = split_into_columns(data)
    timestamps = timestamps.apply(lambda x: datetime.strptime(x, "%Y%m%d %H:%M:%S.%f").time())

    start_times, end_times, labels = timestamps_from_animal(animal_summary_csv)

    for start, end, raw_label in zip(start_times, end_times, labels):
        label = f"{animal_id}_{raw_label}"
        time_data, lvp_data, ecg_data = filter_data(start, end, lvp, timestamps, ecg)

        lvp_df = pd.DataFrame({"time": time_data, "lvp": lvp_data})
        ecg_df = pd.DataFrame({"time": time_data, "ecg": ecg_data})

        lvp_df.to_pickle(output_dir / f"{label}_lvp_raw.pkl")
        ecg_df.to_pickle(output_dir / f"{label}_ecg_raw.pkl")


def split_into_columns(df):
    """Splits the raw combined dataframe into timestamp, LVP, and ECG series."""
    return df["RTlog_Timestamp"], df["LVP"], df["Ext_ECG"]


def timestamps_from_animal(animal_summary_csv):
    """
    Read per-phase start/end timestamps and phase labels from the raw
    TDvAIC.csv. Column positions (4,5,6 for TD block; 9,10,11 for the
    Med/Dose/P label block) match the already-validated normal_cohort layout
    in raw_data_processing.py's COLUMN_MAPS -- this file's structure is
    stable, unlike the VBU log files, so position-based reading is safe here.
    """
    animal_summary_csv = Path(animal_summary_csv)
    subject = pd.read_csv(animal_summary_csv, usecols=[4, 5, 6]).dropna()
    start_times = []
    end_times = []

    for phase in subject["number"].unique():
        phase_data = subject[subject["number"] == phase]
        start = datetime.strptime(phase_data["Time.1"].iloc[0], "%H:%M:%S").time()
        end = datetime.strptime(phase_data["Time.1"].iloc[-1], "%H:%M:%S").time()

        start_times.append(start)
        end_times.append(end)

    labels = []
    label_df = pd.read_csv(animal_summary_csv, usecols=[9, 10, 11]).dropna()
    for i in label_df.index:
        # NOTE: preserved exactly as in legacy code. The outer str() wraps
        # (Dose + "_" + str(P)) -- Dose itself is never explicitly cast to
        # str before concatenation, so this only works because Dose reads in
        # as string/object dtype from this particular CSV. Flagged, not
        # changed, since it evidently works on the real data as written.
        label = str(label_df.iloc[i]["Med"]) + "_" + str(label_df.iloc[i]["Dose"] + "_" + str(label_df.iloc[i]["P"]))
        labels.append(label)

    return start_times, end_times, labels


def filter_data(start, end, hd, timestamps, ecg):
    """Filters LVP/ECG/timestamp series down to a single phase's time window."""
    filtered = [(t, v1, v2) for t, v1, v2 in zip(timestamps, hd, ecg) if start <= t <= end]
    time_data, hd_data, ecg_data = zip(*filtered)

    return list(time_data), list(hd_data), list(ecg_data)


# ── AOP extraction (added for PP/HR/MAP-catheter) ────────────────────────────

def create_aop_data_df(vbu_lvv_folder, animal_id, output_dir):
    """
    Load and concatenate every per-timepoint CSV in an animal's
    calibrated-LVV folder into one raw timestamp/AOP dataframe. Mirrors
    create_data_df exactly, but pulls the AOP column instead of LVP/ECG --
    kept as a SEPARATE pass (re-reads the same CSVs a second time) rather
    than folding into create_data_df, so it doesn't require re-running the
    already-confirmed-working ECG/LVP extraction (which is expensive and
    doesn't need touching).

    Parameters
    ----------
    vbu_lvv_folder : str or Path
    animal_id : str
    output_dir : str or Path
        Where to save the resulting pickle.

    Returns
    -------
    pd.DataFrame
        Columns: RTlog_Timestamp, AOP.
    """
    vbu_lvv_folder = Path(vbu_lvv_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(vbu_lvv_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {vbu_lvv_folder}")

    file_frames = []
    for file_path in csv_files:
        header = pd.read_csv(file_path, nrows=0).columns
        resolved = _resolve_signal_columns(header, aliases=AOP_SIGNAL_ALIASES)
        usecols = [resolved["timestamp"], resolved["aop"]]

        file_df = pd.read_csv(file_path, usecols=usecols).dropna()
        file_df = file_df.rename(columns={
            resolved["timestamp"]: "RTlog_Timestamp",
            resolved["aop"]: "AOP",
        })
        file_frames.append(file_df)

    raw_aop_data = pd.concat(file_frames, ignore_index=True)

    output_path = output_dir / f"raw_aop_data_{animal_id}.pkl"
    raw_aop_data.to_pickle(output_path)
    return raw_aop_data


def create_aop_phase_data(animal_summary_csv, raw_aop_data_pickle, animal_id, output_dir):
    """
    Slice the raw per-timepoint AOP data (from create_aop_data_df) into one
    AOP file per phase, using the EXACT SAME per-phase start/end timestamps
    as create_raw_phase_data uses for ECG/LVP (same timestamps_from_animal()
    call against the same TDvAIC.csv) -- deliberately NOT the coarser,
    6-phase-per-animal timestamps legacy code used for AOP (see module
    docstring). This is what makes PP/HR/MAP-catheter come out at the same
    per-phase granularity as dp/dt max/min and LVEDP.

    Parameters
    ----------
    animal_summary_csv : str or Path
        Path to {animal_id}_TDvAIC.csv.
    raw_aop_data_pickle : str or Path
        Path to the raw_aop_data_{animal_id}.pkl produced by create_aop_data_df.
    animal_id : str
    output_dir : str or Path
        Where to save the per-phase {label}_aop_raw.pkl files. Pass the same
        raw_phase_data/{animal_id}/ directory used for ECG/LVP, so all three
        signals live together per phase.
    """
    raw_aop_data_pickle = Path(raw_aop_data_pickle)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(raw_aop_data_pickle)
    timestamps, aop = data["RTlog_Timestamp"], data["AOP"]
    timestamps = timestamps.apply(lambda x: datetime.strptime(x, "%Y%m%d %H:%M:%S.%f").time())

    start_times, end_times, labels = timestamps_from_animal(animal_summary_csv)

    for start, end, raw_label in zip(start_times, end_times, labels):
        label = f"{animal_id}_{raw_label}"
        time_data, aop_data = _filter_single_series(start, end, aop, timestamps)

        aop_df = pd.DataFrame({"time": time_data, "aop": aop_data})
        aop_df.to_pickle(output_dir / f"{label}_aop_raw.pkl")


def _filter_single_series(start, end, series, timestamps):
    """Filters one data series + its timestamps down to a single phase's time window."""
    filtered = [(t, v) for t, v in zip(timestamps, series) if start <= t <= end]
    time_data, series_data = zip(*filtered)
    return list(time_data), list(series_data)