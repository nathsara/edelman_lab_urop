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

    # CRITICAL FIX (confirmed real bug, not a hypothesis): concatenation
    # above was previously left in "sorted filename order" (per-file, via
    # sorted(vbu_lvv_folder.glob("*.csv")) at the top of this function) --
    # filename order is NOT guaranteed to match chronological order. This
    # was confirmed to produce non-monotonic timestamps in practice (found
    # via diagnose_ct_jump.py against real data: physically impossible
    # heart-rate values, including NEGATIVE bpm and 15000+ bpm, which only
    # occur when two "consecutive" R-peak timestamps are actually out of
    # chronological order -- 60/rp_diff with rp_diff near-zero or negative).
    # Fine phases (a few minutes) rarely span a file-boundary transition, so
    # this was invisible until the new continuous coarse windows (30-70+
    # minutes, crossing many file boundaries) started surfacing it.
    #
    # Fix: sort by the ACTUAL parsed timestamp, not filename order. Parses
    # into a temporary datetime column for correct chronological sorting
    # (rather than trusting the raw string's lexicographic order, which
    # would only be safe if the source format is perfectly zero-padded --
    # not verified, so not assumed), then drops the helper column --
    # RTlog_Timestamp itself is left as the original string, unchanged,
    # so every downstream consumer (which parses it themselves) needs no
    # changes.
    _sort_key = pd.to_datetime(raw_hd_data["RTlog_Timestamp"], format="%Y%m%d %H:%M:%S.%f")
    was_already_sorted = _sort_key.is_monotonic_increasing
    raw_hd_data = raw_hd_data.assign(_sort_key=_sort_key).sort_values("_sort_key").reset_index(drop=True)
    if not was_already_sorted:
        print(f"    [{animal_id}] WARNING: raw data was NOT in chronological order by filename "
              f"-- re-sorted by actual timestamp. This animal's data was affected by the "
              f"non-monotonic-timestamp bug.")

    # SECOND BUG surfaced by the sort fix above: exact-duplicate timestamps
    # (down to the microsecond) exist in some raw data -- e.g. from
    # overlapping time ranges between two source files. Before sorting,
    # duplicate rows were scattered through the data in filename order and
    # rarely landed adjacent to each other, so a zero time-difference
    # between "consecutive" rows almost never occurred. After sorting
    # correctly, true duplicates land next to each other -- and
    # _calc_deriv_lvp's dp/dt = diff(value)/diff(time) divides by zero.
    #
    # Fix: drop duplicate timestamps, keeping the first occurrence. If
    # duplicate timestamps have DIFFERING LVP/ECG values (a genuine data
    # conflict, not just a harmless re-logged duplicate), this is flagged
    # loudly rather than silently resolved -- worth real investigation
    # later, but not something to block an urgent run over right now.
    dup_mask = raw_hd_data["_sort_key"].duplicated(keep=False)
    n_dup_timestamps = dup_mask.sum()
    if n_dup_timestamps > 0:
        dup_rows = raw_hd_data[raw_hd_data["_sort_key"].duplicated(keep=False)]
        conflicting = dup_rows.groupby("_sort_key")[["LVP", "Ext_ECG"]].nunique()
        n_conflicting = (conflicting.max(axis=1) > 1).sum()
        print(f"    [{animal_id}] WARNING: {n_dup_timestamps} rows had exact-duplicate timestamps "
              f"-- keeping first occurrence of each, dropping the rest.")
        if n_conflicting > 0:
            print(f"    [{animal_id}] WARNING: {n_conflicting} of those duplicate timestamps had "
                  f"DIFFERING LVP/ECG values between the duplicates (not just harmless re-logged "
                  f"copies) -- kept the first occurrence arbitrarily. Worth investigating which "
                  f"source file/value is correct if this matters for final results.")
        raw_hd_data = raw_hd_data.drop_duplicates(subset="_sort_key", keep="first").reset_index(drop=True)

    raw_hd_data = raw_hd_data.drop(columns="_sort_key")

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

    # Same fix as create_data_df -- see that function's comment for full
    # explanation. Sort by actual parsed timestamp, not filename order.
    _sort_key = pd.to_datetime(raw_aop_data["RTlog_Timestamp"], format="%Y%m%d %H:%M:%S.%f")
    was_already_sorted = _sort_key.is_monotonic_increasing
    raw_aop_data = raw_aop_data.assign(_sort_key=_sort_key).sort_values("_sort_key").reset_index(drop=True)
    if not was_already_sorted:
        print(f"    [{animal_id}] WARNING: raw AOP data was NOT in chronological order by filename "
              f"-- re-sorted by actual timestamp.")

    # Same duplicate-timestamp fix as create_data_df -- see that function's
    # comment for full explanation.
    dup_mask = raw_aop_data["_sort_key"].duplicated(keep=False)
    n_dup_timestamps = dup_mask.sum()
    if n_dup_timestamps > 0:
        dup_rows = raw_aop_data[dup_mask]
        conflicting = dup_rows.groupby("_sort_key")["AOP"].nunique()
        n_conflicting = (conflicting > 1).sum()
        print(f"    [{animal_id}] WARNING: {n_dup_timestamps} AOP rows had exact-duplicate timestamps "
              f"-- keeping first occurrence of each, dropping the rest.")
        if n_conflicting > 0:
            print(f"    [{animal_id}] WARNING: {n_conflicting} of those duplicate timestamps had "
                  f"DIFFERING AOP values -- kept the first occurrence arbitrarily.")
        raw_aop_data = raw_aop_data.drop_duplicates(subset="_sort_key", keep="first").reset_index(drop=True)

    raw_aop_data = raw_aop_data.drop(columns="_sort_key")

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


# ── Coarse (whole-drug-state) extraction, for continuous-trajectory data ────
# ADDED for Stage 2 (continuous-time drug-effect trajectory). Distinct from
# everything above: instead of the fine per-phase (P-level x dose) windows
# in ANIMAL_PHASES, this slices ONE continuous window per whole drug state
# (e.g. "202_Nitro" spans both low AND high dose sub-periods as a single
# unbroken block -- confirmed via a legacy dosage-increase axvline sitting
# mid-trace in graphing.py, and via process.flex_combined_phase_data being
# called with coarse labels like "202_Nitro" in gen_map_pp_data.py's
# commented-out __main__).
#
# Timestamps come from data/raw/drug_start_end_times.csv (columns Subject,
# Phase, Start, End) -- REAL recorded protocol timestamps, not derived by
# assuming contiguity between fine sub-phases. Scope is restricted to
# Nitro/Phen/Dobu by default (Stage 2's confirmed scope; Baseline is not
# reprocessed here at all since the existing fine-phase Baseline_0_P6 row in
# {animal_id}_catheter_summary.pkl already provides the needed baseline
# reference; Washout and Esmo are out of scope per user decision).
#
# Reuses the ALREADY-EXTRACTED full-animal raw_hd_data_{animal_id}.pkl /
# raw_aop_data_{animal_id}.pkl pickles (from create_data_df / create_aop_data_df)
# rather than re-reading the VBU CSVs a third time -- just re-sliced with a
# different set of timestamps.

COARSE_DRUGS = ("Nitro", "Phen", "Dobu")


def _timestamps_from_drug_csv(drug_start_end_csv, animal_id, drugs=COARSE_DRUGS):
    """
    Reads data/raw/drug_start_end_times.csv and returns (start_times, end_times,
    labels) for one animal, restricted to `drugs`, in the same
    (start_times, end_times, labels) shape timestamps_from_animal() returns --
    so the same filtering logic downstream works for both.
    """
    drug_start_end_csv = Path(drug_start_end_csv)
    df = pd.read_csv(drug_start_end_csv)
    df = df[(df["Subject"].astype(str) == str(animal_id)) & (df["Phase"].isin(drugs))]
    if df.empty:
        raise ValueError(
            f"No rows found in {drug_start_end_csv} for Subject={animal_id!r}, "
            f"Phase in {drugs}."
        )

    start_times = [datetime.strptime(s, "%H:%M:%S").time() for s in df["Start"]]
    end_times = [datetime.strptime(s, "%H:%M:%S").time() for s in df["End"]]
    labels = list(df["Phase"])  # e.g. "Nitro" -- coarse, no P-level/dose suffix
    return start_times, end_times, labels


def create_coarse_phase_data(drug_start_end_csv, raw_hd_data_pickle, animal_id, output_dir, drugs=COARSE_DRUGS):
    """
    Slices the already-extracted full-animal raw ECG/LVP data (from
    create_data_df) into one coarse whole-drug-state ECG file and one LVP
    file per drug in `drugs`, using REAL recorded Start/End timestamps from
    drug_start_end_times.csv -- not derived from the fine per-phase windows.

    Output filenames: {animal_id}_{drug}_ecg_raw.pkl / {animal_id}_{drug}_lvp_raw.pkl
    (e.g. "202_Nitro_ecg_raw.pkl") -- distinct from fine-phase filenames
    (which always have a _{dose}_{P} suffix), so both can safely live in the
    same raw_phase_data/{animal_id}/ directory with no collision.

    Parameters
    ----------
    drug_start_end_csv : str or Path
        Path to data/raw/drug_start_end_times.csv.
    raw_hd_data_pickle : str or Path
        Path to the raw_hd_data_{animal_id}.pkl produced by create_data_df.
    animal_id : str
    output_dir : str or Path
    drugs : tuple of str
        Which drug states to extract. Default ("Nitro", "Phen", "Dobu") --
        Stage 2's confirmed scope.
    """
    raw_hd_data_pickle = Path(raw_hd_data_pickle)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(raw_hd_data_pickle)
    timestamps, lvp, ecg = split_into_columns(data)
    timestamps = timestamps.apply(lambda x: datetime.strptime(x, "%Y%m%d %H:%M:%S.%f").time())

    start_times, end_times, labels = _timestamps_from_drug_csv(drug_start_end_csv, animal_id, drugs)

    for start, end, drug in zip(start_times, end_times, labels):
        label = f"{animal_id}_{drug}"
        time_data, lvp_data, ecg_data = filter_data(start, end, lvp, timestamps, ecg)

        lvp_df = pd.DataFrame({"time": time_data, "lvp": lvp_data})
        ecg_df = pd.DataFrame({"time": time_data, "ecg": ecg_data})

        lvp_df.to_pickle(output_dir / f"{label}_lvp_raw.pkl")
        ecg_df.to_pickle(output_dir / f"{label}_ecg_raw.pkl")


def create_coarse_aop_phase_data(drug_start_end_csv, raw_aop_data_pickle, animal_id, output_dir, drugs=COARSE_DRUGS):
    """
    AOP equivalent of create_coarse_phase_data() -- slices the already-
    extracted full-animal raw_aop_data_{animal_id}.pkl into one coarse
    {animal_id}_{drug}_aop_raw.pkl per drug in `drugs`, using the same real
    recorded Start/End timestamps.
    """
    raw_aop_data_pickle = Path(raw_aop_data_pickle)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(raw_aop_data_pickle)
    timestamps, aop = data["RTlog_Timestamp"], data["AOP"]
    timestamps = timestamps.apply(lambda x: datetime.strptime(x, "%Y%m%d %H:%M:%S.%f").time())

    start_times, end_times, labels = _timestamps_from_drug_csv(drug_start_end_csv, animal_id, drugs)

    for start, end, drug in zip(start_times, end_times, labels):
        label = f"{animal_id}_{drug}"
        time_data, aop_data = _filter_single_series(start, end, aop, timestamps)

        aop_df = pd.DataFrame({"time": time_data, "aop": aop_data})
        aop_df.to_pickle(output_dir / f"{label}_aop_raw.pkl")