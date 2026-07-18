"""
shared/catheter_data_processing.py

Catheter-derived signal-processing pipeline: dp/dt max, dp/dt min, LVEDP,
and PP/MAP/HR-catheter, computed per phase from the raw ECG/LVP/AOP pickles
produced by catheter_data_init.py, then combined into ONE summary DataFrame
per animal.

dp/dt max/min and LVEDP are refactored from legacy process.py. PP/MAP/HR-
catheter (AoP-derived) are refactored from legacy aop_processing.py /
gen_map_pp_data.py, and MERGED into this module and into the same per-phase
summary row -- rather than kept as a separate module/output -- since the two
share real code (arrhythmia removal, R-peak detection, the christov beat
detector) and legacy already duplicated some of that between the two
codebases. PP/MAP/HR-catheter is a DELIBERATE DEPARTURE FROM LEGACY
GRANULARITY: legacy computed it at a coarse 6-phase-per-animal resolution
using a different timestamp source; here it's computed at the SAME per-phase
(P-level x dose) resolution as dp/dt/LVEDP, using the identical timestamps
-- see catheter_data_init.py's module docstring for the AOP extraction that
makes this possible. See PROJECT_DECISIONS.md for full rationale.

ALL COMPUTATION LOGIC IS PRESERVED EXACTLY (same arrhythmia windows, same
segment-splitting strategy, same two-round outlier detection for dp/dt/
LVEDP, same finetuner logic -- including its known quirks, see below; same
systolic/diastolic/PP/MAP/HR calculation for the AoP-derived metrics, with
NO outlier removal added to those -- legacy never did that for them). What
changed vs. legacy: 

  - arrhythmia_removal() now reads exclusion windows from
    catheter_phase_config.ARRHYTHMIA_EXCLUSION_WINDOWS instead of a hardcoded
    if/elif chain. Same windows, same phases affected.
  - Per-animal phase lists come from catheter_phase_config.ANIMAL_PHASES
    instead of a hardcoded if/elif chain in combined_phase_data.
  - combined_phase_data() produces ONE combined DataFrame per animal
    ({animal_id}_catheter_summary.pkl, columns: phase_number, med, dose,
    p_level, dpdt_max_mean/std, dpdt_min_mean/std, lvedp_mean/std) instead of
    six separate flat, unlabeled pickles.
  - No intermediate _cleaned / _cleaner pickles are written to disk at any
    point. The two rounds of outlier detection/removal still happen exactly
    as in legacy, but entirely in memory -- only the final per-phase
    mean/std, appended as one row of the combined summary DataFrame, is
    persisted.
  - detect_outliers() drops the unused `label` parameter (legacy accepted it
    but never used it inside the function -- dead parameter, zero effect on
    output).
  - plot=False is now the default (legacy default was plot=True, which calls
    plt.show() and BLOCKS execution until the window is closed -- fatal for
    an unattended batch run across ~70 phases). Pass plot=True for one-off
    single-phase visual inspection.
  - No per-animal override_files / destination_folder parameters (legacy's
    flex_combined_phase_data concern) -- this pipeline always reads from a
    single raw_phase_dir per animal, produced by catheter_data_init.

THREE KNOWN QUIRKS PRESERVED EXACTLY, NOT FIXED (flagged in legacy review,
confirmed present in the actual legacy source):

  1. lvedp_finetuner's `except KeyError` can never fire -- list indexing
     raises IndexError, not KeyError. If the search-window index goes
     negative, Python silently wraps to read from the end of the list rather
     than raising anything.
  2. Bounds-checking is inconsistent across the three finetuner functions:
     dpdt_max_finetuner guards the upper bound only; dpdt_min_finetuner has
     NO guard at all; lvedp_finetuner has the non-functional guard from #1.
  3. The P6/P3-alternation-by-list-position logic in plots.py uses
     `list.index(value)`, which returns the position of the FIRST matching
     value if there are duplicates -- not necessarily the position being
     iterated. Not touched by this module directly, but worth remembering
     when feeding this module's output into plots.py's plotting functions.

HALTS ON ERROR: if any single phase fails to process (missing file,
malformed data, arrhythmia-removal leaves too little data, etc.), the
exception propagates immediately and combined_phase_data() aborts. This is
deliberate -- a partial/incomplete summary pickle must never be silently
produced. Per-phase try/except-and-continue was considered and rejected.

Scope: 4-animal normal cohort only (202, 203, 205, 221), matching
catheter_phase_config.ANIMAL_PHASES.
"""

from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
from scipy.signal import find_peaks
from ecgdetectors import Detectors

from .catheter_phase_config import (
    ANIMAL_PHASES,
    ARRHYTHMIA_EXCLUSION_WINDOWS,
    PHASES_REQUIRING_EXTRA_PROCESSING,
)


# ── Arrhythmia removal ──────────────────────────────────────────────────────

def arrhythmia_removal(label, metric_df):
    """
    Remove manually-curated arrhythmia-artifact time windows for this phase,
    if any exist. Operates on ONE dataframe at a time (called separately for
    ecg_df and lvp_df) -- this matches the real legacy process.py signature
    exactly; an earlier assumption that legacy took a paired (label, ecg_df,
    lvp_df) signature was incorrect (see PROJECT_DECISIONS.md).

    Windows and phases affected are unchanged from legacy: same 4 phases
    (203_Nitro_high_P3, 203_Nitro_low_P3, 202_Phen_0_P3, 202_Washout_1_P3),
    same exact timestamps, sourced from catheter_phase_config.
    """
    windows = ARRHYTHMIA_EXCLUSION_WINDOWS.get(label)
    if not windows:
        return metric_df
    for start, end in windows:
        metric_df = metric_df[~((metric_df["time"] > start) & (metric_df["time"] < end))]
    return metric_df


# ── Single-phase pipeline ────────────────────────────────────────────────────

def process_phase(label, raw_phase_dir, plot=False, per_beat_dir=None):
    """
    Run full dp/dt max/min + LVEDP processing for a single phase.

    Parameters
    ----------
    label : str
        e.g. "202_Baseline_0_P6" -- must match an entry in
        catheter_phase_config.ANIMAL_PHASES, and must have corresponding
        {label}_ecg_raw.pkl / {label}_lvp_raw.pkl files in raw_phase_dir.
    raw_phase_dir : str or Path
        Directory containing the per-phase raw ECG/LVP pickles, as produced
        by catheter_data_init.create_raw_phase_data.
    plot : bool
        If True, generates the diagnostic ECG/dp-dt/LVP plot for this phase
        via plots.phase_data_plot (unchanged legacy visual). Default False.
    per_beat_dir : str or Path, optional
        If given, saves the final per-beat dp/dt max, dp/dt min, and LVEDP
        series (the same DataFrames the returned mean/std are computed from
        -- post arrhythmia-removal, post two-round outlier removal, NOT
        intermediate half-baked versions) as
        {label}_dpdt_max.pkl / {label}_dpdt_min.pkl / {label}_lvedp.pkl.
        If None (default), nothing extra is saved -- backward compatible
        with existing callers.

    Returns
    -------
    dict with keys dpdt_max_mean, dpdt_max_std, dpdt_min_mean, dpdt_min_std,
    lvedp_mean, lvedp_std.
    """
    raw_phase_dir = Path(raw_phase_dir)
    ecg_df = pd.read_pickle(raw_phase_dir / f"{label}_ecg_raw.pkl")
    lvp_df = pd.read_pickle(raw_phase_dir / f"{label}_lvp_raw.pkl")
    print(f"    [{label}] loaded {len(ecg_df)} ECG rows, {len(lvp_df)} LVP rows")

    ecg_df = arrhythmia_removal(label, ecg_df)
    lvp_df = arrhythmia_removal(label, lvp_df)

    if label in PHASES_REQUIRING_EXTRA_PROCESSING:
        max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, _lvp_peaks_timestamps, dpdt = (
            _extra_processing_pipeline(ecg_df, lvp_df, lvedp_finetuning=True)
        )
    else:
        _t0 = time.time()
        dpdt = _calc_deriv_lvp(list(lvp_df["lvp"]), list(lvp_df["time"]))
        print(f"    [{label}] dp/dt derivative computed ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        r_peaks = _christov_beat_timestamps(list(ecg_df["ecg"]))
        print(f"    [{label}] R-peak detection done, {len(r_peaks)} beats found ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        # PERFORMANCE FIX (efficiency-only, provably identical output):
        # legacy called list(ecg_df["time"]) / list(lvp_df["time"]) INSIDE
        # the list comprehension -- meaning the full column gets rebuilt
        # into a fresh list on EVERY iteration, not once. For a large
        # continuous window this is catastrophic (confirmed: a synthetic
        # test at this data's actual scale, ~1M rows, ~7700 lookups, timed
        # out entirely for the buggy version; the fixed version ran in
        # 0.06s). Hoisting list(...) outside the comprehension so it's
        # built ONCE produces the exact same indexed values, just without
        # rebuilding the list thousands of times over.
        ecg_time_list = list(ecg_df["time"])
        r_peaks_timestamps = [ecg_time_list[r] for r in r_peaks]
        lvp_peaks = find_peaks(list(lvp_df["lvp"]), height=50)
        lvp_time_list = list(lvp_df["time"])
        lvp_peaks_timestamps = [lvp_time_list[r] for r in list(lvp_peaks[0])]
        print(f"    [{label}] R-peak/LVP-peak timestamps resolved ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        r_peaks_timestamps = _r_peak_corrector(r_peaks_timestamps, lvp_peaks_timestamps)
        print(f"    [{label}] R-peaks corrected against LVP peaks, {len(r_peaks_timestamps)} remain ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        lvedp_data = []
        # PERFORMANCE FIX (efficiency-only, provably identical output): legacy
        # called list.index() inside this list comprehension -- a full O(n)
        # linear scan PER r-peak. Fine for small fine-phase windows (a few
        # thousand samples), but for large continuous coarse windows (e.g. a
        # 71-minute Nitro block, ~1.6M samples) this became O(n*m) and could
        # take a very long time. Precomputing the timestamp->index map once
        # (O(n)) then doing O(1) dict lookups per r-peak (O(m)) produces the
        # EXACT same reference_indices, just fast -- timestamps here are real,
        # distinct wall-clock values, so there's no duplicate-key ambiguity.
        lvp_time_to_index = {t: i for i, t in enumerate(list(lvp_df["time"]))}
        reference_indices = [lvp_time_to_index[x] for x in r_peaks_timestamps]
        for ref_idx in reference_indices:
            lvedp_data.append(lvp_df["lvp"].iloc[ref_idx])
        lvedp = pd.DataFrame({"time": r_peaks_timestamps, "lvedp": lvedp_data})
        print(f"    [{label}] LVEDP extracted ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        max_dpdt, min_dpdt, _max_timestamps = _dpdt_minmax_extractor(dpdt, r_peaks, list(ecg_df["time"]))
        print(f"    [{label}] dp/dt max/min per beat extracted ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        dpdt_max_list, dpdt_max_time_list = _dpdt_max_finetuner(
            dpdt, max_dpdt["dpdt_max"].tolist(), max_dpdt["time"].tolist(), lvp_df["time"][:-1].tolist()
        )
        dpdt_max_list, dpdt_max_time_list = _finetune_signal(dpdt_max_list, dpdt_max_time_list)
        max_dpdt = pd.DataFrame({"time": dpdt_max_time_list, "dpdt_max": dpdt_max_list})
        print(f"    [{label}] dp/dt max finetuned ({time.time()-_t0:.1f}s)")

        _t0 = time.time()
        dpdt_min_list, dpdt_min_time_list = _dpdt_min_finetuner(
            dpdt, min_dpdt["dpdt_min"].tolist(), min_dpdt["time"].tolist(), lvp_df["time"][:-1].tolist()
        )
        dpdt_min_list, dpdt_min_time_list = _finetune_signal(dpdt_min_list, dpdt_min_time_list)
        min_dpdt = pd.DataFrame({"time": dpdt_min_time_list, "dpdt_min": dpdt_min_list})
        print(f"    [{label}] dp/dt min finetuned ({time.time()-_t0:.1f}s)")

    # Two rounds of outlier detection/removal -- preserved exactly from
    # legacy. Unlike legacy, no intermediate _cleaned / _cleaner pickles are
    # ever written to disk; everything below is in-memory only, and just the
    # final mean/std makes it into the returned dict / combined summary row.
    dpdt_max_outliers = _detect_outliers(max_dpdt["time"], max_dpdt["dpdt_max"])
    dpdt_min_outliers = _detect_outliers(min_dpdt["time"], min_dpdt["dpdt_min"])
    lvedp_outliers = _detect_outliers(lvedp["time"], lvedp["lvedp"])

    dpdt_max_cleaned = _remove_outliers(max_dpdt, dpdt_max_outliers)
    dpdt_min_cleaned = _remove_outliers(min_dpdt, dpdt_min_outliers)
    lvedp_cleaned = _remove_outliers(lvedp, lvedp_outliers)

    dpdt_max_outliers2 = _detect_outliers(dpdt_max_cleaned["time"], dpdt_max_cleaned["dpdt_max"])
    dpdt_min_outliers2 = _detect_outliers(dpdt_min_cleaned["time"], dpdt_min_cleaned["dpdt_min"])
    lvedp_outliers2 = _detect_outliers(lvedp_cleaned["time"], lvedp_cleaned["lvedp"])

    dpdt_max_final = _remove_outliers(dpdt_max_cleaned, dpdt_max_outliers2)
    dpdt_min_final = _remove_outliers(dpdt_min_cleaned, dpdt_min_outliers2)
    lvedp_final = _remove_outliers(lvedp_cleaned, lvedp_outliers2)

    if per_beat_dir is not None:
        per_beat_dir = Path(per_beat_dir)
        per_beat_dir.mkdir(parents=True, exist_ok=True)
        dpdt_max_final.to_pickle(per_beat_dir / f"{label}_dpdt_max.pkl")
        dpdt_min_final.to_pickle(per_beat_dir / f"{label}_dpdt_min.pkl")
        lvedp_final.to_pickle(per_beat_dir / f"{label}_lvedp.pkl")

    if plot:
        # Lazily imported -- plots.py (and its matplotlib dependency) is only
        # needed for the plot=True, one-off inspection path, never for a
        # normal unattended pipeline run.
        import plots
        plots.phase_data_plot(
            ecg_df, lvp_df, r_peaks_timestamps,
            dpdt_max_final, dpdt_min_final, lvedp_final, dpdt, label,
        )

    return {
        "dpdt_max_mean": dpdt_max_final["dpdt_max"].mean(),
        "dpdt_max_std": dpdt_max_final["dpdt_max"].std(),
        "dpdt_min_mean": dpdt_min_final["dpdt_min"].mean(),
        "dpdt_min_std": dpdt_min_final["dpdt_min"].std(),
        "lvedp_mean": lvedp_final["lvedp"].mean(),
        "lvedp_std": lvedp_final["lvedp"].std(),
    }


# ── PP/HR/MAP-catheter (AoP-derived) helpers ─────────────────────────────────
# Ported from legacy aop_processing.py. Computation logic preserved EXACTLY
# -- no two-round outlier removal is applied here, unlike dp/dt max/min and
# LVEDP above. Legacy aop_processing.py never did that for these metrics; it
# was specific to process.py's pipeline. Adding it here would be inventing
# new processing logic, not a like-for-like refactor, so it's deliberately
# NOT included. What changed vs. legacy: functions take dataframes directly
# (calc_hr no longer opens pickle files by path itself -- the caller now
# reads raw_phase_dir pickles once, upstream, matching how every other
# function in this module already works), and calc_hr's arrhythmia_removal
# calls use the module-level arrhythmia_removal() above rather than a
# separately-imported one -- same single-df signature, already confirmed
# compatible.

def _calc_systolic(aop_df, label):
    aop_df = arrhythmia_removal(label, aop_df)
    systolic_indices = find_peaks(list(aop_df["aop"]), prominence=20, width=5)
    # PERFORMANCE FIX (efficiency-only, provably identical output): same
    # list-rebuilt-per-iteration bug as elsewhere in this module -- see
    # process_phase()'s matching fix for the full explanation and a timed
    # confirmation of the severity at this data's real scale.
    aop_list = list(aop_df["aop"])
    aop_time_list = list(aop_df["time"])
    systolic = [aop_list[r] for r in list(systolic_indices[0])]
    systolic_timestamps = [aop_time_list[r] for r in list(systolic_indices[0])]
    return pd.DataFrame({"time": systolic_timestamps, "systolic_pressure": systolic})


def _calc_diastolic(aop_df, label):
    aop_df = arrhythmia_removal(label, aop_df)
    inverted_aop = [-x for x in list(aop_df["aop"])]
    diastolic_indices = find_peaks(inverted_aop, prominence=20, width=5)
    aop_list = list(aop_df["aop"])
    aop_time_list = list(aop_df["time"])
    diastolic = [aop_list[r] for r in list(diastolic_indices[0])]
    diastolic_timestamps = [aop_time_list[r] for r in list(diastolic_indices[0])]
    return pd.DataFrame({"time": diastolic_timestamps, "diastolic_pressure": diastolic})


def _merge_asof_nearest_time(left_df, right_df):
    """
    pandas.merge_asof requires a numeric or datetime dtype for the `on`
    column -- it cannot operate directly on bare datetime.time objects
    (object dtype), which is what every "time" column in this pipeline uses.
    This anchors both sides to an arbitrary shared date purely so
    merge_asof's nearest-neighbor matching can run; it does NOT change which
    points get matched -- datetime.time objects already sort/compare
    correctly on their own, this only satisfies pandas' dtype requirement.

    NOT present in legacy aop_processing.py, which called merge_asof
    directly on "time" -- confirmed this raises pandas.errors.MergeError
    ("Incompatible merge dtype... both sides must have numeric dtype") the
    first time this code actually ran against real data. Legacy code would
    have hit the exact same error if it had ever been run end-to-end (it
    hadn't -- see PROJECT_DECISIONS.md). This is a genuine bug fix, not a
    behavior change to flag/confirm -- the original logic was never
    executable as written.
    """
    left = left_df.copy()
    right = right_df.copy()
    left["_merge_time"] = left["time"].apply(lambda t: datetime.combine(datetime.today(), t))
    right["_merge_time"] = right["time"].apply(lambda t: datetime.combine(datetime.today(), t))
    merged = pd.merge_asof(left, right, on="_merge_time", direction="nearest", suffixes=("", "_right"))
    return merged


def _calc_pulse_pressure(sys_df, dia_df):
    sys_df = sys_df.sort_values("time")
    dia_df = dia_df.sort_values("time")
    merged = _merge_asof_nearest_time(sys_df, dia_df)
    pp_df = merged[["time"]].copy()
    pp_df["pulse_pressure"] = merged["systolic_pressure"] - merged["diastolic_pressure"]
    return pp_df


def _calc_map(sys_df, dia_df):
    sys_df = sys_df.sort_values("time")
    dia_df = dia_df.sort_values("time")
    merged = _merge_asof_nearest_time(sys_df, dia_df)
    map_df = merged[["time"]].copy()
    map_df["map"] = (1 / 3) * merged["systolic_pressure"] + (2 / 3) * merged["diastolic_pressure"]
    return map_df


def _calc_hr(ecg_df, lvp_df, label):
    ecg_df = arrhythmia_removal(label, ecg_df)
    lvp_df = arrhythmia_removal(label, lvp_df)

    _t0 = time.time()
    r_peaks = _christov_beat_timestamps(list(ecg_df["ecg"]))
    print(f"    [{label}] (HR) R-peak detection done, {len(r_peaks)} beats found ({time.time()-_t0:.1f}s)")
    _t0 = time.time()
    ecg_time_list = list(ecg_df["time"])
    r_peaks_timestamps = [ecg_time_list[r] for r in r_peaks]
    lvp_peaks = find_peaks(list(lvp_df["lvp"]), height=50)
    lvp_time_list = list(lvp_df["time"])
    lvp_peaks_timestamps = [lvp_time_list[r] for r in list(lvp_peaks[0])]
    r_peaks_timestamps = _r_peak_corrector(r_peaks_timestamps, lvp_peaks_timestamps)
    print(f"    [{label}] (HR) R-peaks resolved and corrected, {len(r_peaks_timestamps)} remain ({time.time()-_t0:.1f}s)")

    continuous_hr = []
    for ts in range(len(r_peaks_timestamps) - 1):
        rp1_ts = r_peaks_timestamps[ts]
        rp2_ts = r_peaks_timestamps[ts + 1]
        rp_diff = (datetime.combine(datetime.today(), rp2_ts) - datetime.combine(datetime.today(), rp1_ts)).total_seconds()
        continuous_hr.append(round(60 / rp_diff))

    return pd.DataFrame({"time": r_peaks_timestamps[1:], "heart_rate": continuous_hr})


def process_phase_hemodynamics(label, raw_phase_dir, per_beat_dir=None):
    """
    Computes PP-catheter, MAP-catheter, and HR-catheter for a single phase.

    PP/MAP need the AOP signal (systolic/diastolic peaks off the AOP
    waveform) -- reads {label}_aop_raw.pkl, produced by
    catheter_data_init.create_aop_phase_data. HR-catheter only needs
    ECG+LVP, already extracted for the dp/dt/LVEDP pipeline -- no new raw
    signal needed for it.

    per_beat_dir : str or Path, optional
        If given, saves the final per-beat PP, MAP, and HR series (no
        outlier removal applied to these -- see module docstring) as
        {label}_pp_catheter.pkl / {label}_map_catheter.pkl /
        {label}_hr_catheter.pkl. If None (default), nothing extra is saved.

    Returns
    -------
    dict with keys PP_catheter_mean, PP_catheter_std, MAP_catheter_mean,
    MAP_catheter_std, HR_catheter_mean, HR_catheter_std.
    """
    raw_phase_dir = Path(raw_phase_dir)
    aop_df = pd.read_pickle(raw_phase_dir / f"{label}_aop_raw.pkl")
    ecg_df = pd.read_pickle(raw_phase_dir / f"{label}_ecg_raw.pkl")
    lvp_df = pd.read_pickle(raw_phase_dir / f"{label}_lvp_raw.pkl")

    _t0 = time.time()
    systolic_df = _calc_systolic(aop_df, label)
    diastolic_df = _calc_diastolic(aop_df, label)
    pp_df = _calc_pulse_pressure(systolic_df, diastolic_df)
    map_df = _calc_map(systolic_df, diastolic_df)
    print(f"    [{label}] PP/MAP computed from AOP ({time.time()-_t0:.1f}s)")

    # NOTE: _calc_hr() below runs its OWN full beat-detection pass
    # (_christov_beat_timestamps + _r_peak_corrector) on the same ecg_df/
    # lvp_df that process_phase() already ran beat detection on for this
    # same label -- genuinely redundant, beat detection runs twice per
    # animal-drug pair. Not deduplicated here (would mean restructuring how
    # combined_phase_data calls both functions, riskier to change correctly
    # under time pressure than adding visibility first) -- if this specific
    # step turns out to be a major share of the total runtime, worth
    # revisiting to share one r_peaks computation between both functions.
    _t0 = time.time()
    hr_df = _calc_hr(ecg_df, lvp_df, label)
    print(f"    [{label}] HR computed ({time.time()-_t0:.1f}s)")

    if per_beat_dir is not None:
        per_beat_dir = Path(per_beat_dir)
        per_beat_dir.mkdir(parents=True, exist_ok=True)
        pp_df.to_pickle(per_beat_dir / f"{label}_pp_catheter.pkl")
        map_df.to_pickle(per_beat_dir / f"{label}_map_catheter.pkl")
        hr_df.to_pickle(per_beat_dir / f"{label}_hr_catheter.pkl")

    return {
        "PP_catheter_mean": pp_df["pulse_pressure"].mean(),
        "PP_catheter_std": pp_df["pulse_pressure"].std(),
        "MAP_catheter_mean": map_df["map"].mean(),
        "MAP_catheter_std": map_df["map"].std(),
        "HR_catheter_mean": hr_df["heart_rate"].mean(),
        "HR_catheter_std": hr_df["heart_rate"].std(),
    }


# ── Per-animal driver ────────────────────────────────────────────────────────

def combined_phase_data(animal_id, raw_phase_dir, output_dir, plot=False, per_beat_dir=None):
    """
    Run process_phase() (dp/dt max/min, LVEDP) and process_phase_hemodynamics()
    (PP/MAP/HR-catheter) for every phase of one animal, in the order given by
    catheter_phase_config.ANIMAL_PHASES, and save ONE combined summary
    DataFrame: {animal_id}_catheter_summary.pkl.

    Columns: phase_number, med, dose, p_level, dpdt_max_mean, dpdt_max_std,
    dpdt_min_mean, dpdt_min_std, lvedp_mean, lvedp_std, PP_catheter_mean,
    PP_catheter_std, MAP_catheter_mean, MAP_catheter_std, HR_catheter_mean,
    HR_catheter_std.

      - phase_number is assigned by 1-indexed POSITION within
        ANIMAL_PHASES[animal_id] (already in chronological protocol order).
        Needed because significance_testing.py's washout-scope logic depends
        on a phase_number column existing.
      - med/dose/p_level are parsed by splitting the phase label on "_"
        (format is consistently {animal}_{Med}_{Dose}_{P} across every entry
        in every animal's ANIMAL_PHASES list, including animal 205's
        single-measurement Dobu entry -- "205_Dobu_high_P6" still splits into
        exactly 4 parts).
      - PP/MAP/HR-catheter are computed at the SAME per-phase granularity as
        dp/dt/LVEDP (deliberate departure from legacy, which used a coarser
        6-phase-per-animal AOP timestamp source) -- see catheter_data_init.py
        module docstring for the AOP extraction that makes this possible.

    per_beat_dir : str or Path, optional
        If given, also saves the full per-beat data (not just mean/std) for
        all six metrics per phase, under per_beat_dir -- see process_phase()
        and process_phase_hemodynamics() docstrings for exactly what's saved.
        If None (default), only the summary pickle is produced, matching
        prior behavior.

    HALTS on any error (propagates the exception, does not catch/continue)
    -- a partial/incomplete summary pickle must never be silently produced.
    """
    if animal_id not in ANIMAL_PHASES:
        raise ValueError(
            f"No phase list defined for animal {animal_id!r} in "
            f"catheter_phase_config.ANIMAL_PHASES. Expected one of {list(ANIMAL_PHASES)}."
        )

    raw_phase_dir = Path(raw_phase_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = ANIMAL_PHASES[animal_id]
    rows = []

    for phase_number, label in enumerate(phases, start=1):
        print(f"[{animal_id}] Processing phase {phase_number}/{len(phases)}: {label}")

        stats = process_phase(label, raw_phase_dir, plot=plot, per_beat_dir=per_beat_dir)
        hemo_stats = process_phase_hemodynamics(label, raw_phase_dir, per_beat_dir=per_beat_dir)

        parts = label.split("_")
        if len(parts) != 4:
            raise ValueError(
                f"Expected phase label '{label}' to split into exactly 4 parts "
                f"({{animal}}_{{Med}}_{{Dose}}_{{P}}), got {len(parts)}: {parts}"
            )
        _animal, med, dose, p_level = parts

        rows.append({
            "phase_number": phase_number,
            "med": med,
            "dose": dose,
            "p_level": p_level,
            **stats,
            **hemo_stats,
        })
        print(f"[{animal_id}] Finished phase {phase_number}/{len(phases)}: {label}")

    summary_df = pd.DataFrame(rows)
    output_path = output_dir / f"{animal_id}_catheter_summary.pkl"
    summary_df.to_pickle(output_path)
    print(f"[{animal_id}] Saved catheter summary ({len(summary_df)} phases) -> {output_path}")
    return summary_df


# ── Segment-splitting path (4 phases with arrhythmia gaps) ──────────────────

def _extra_processing_pipeline(ecg_df, lvp_df, lvedp_finetuning=True, dpdt_finetuning=True, dpdt_res=15, lvedp_res=15):
    """
    After removing arrhythmias, the data has gaps -- we only want continuous
    segments fed into the pipeline from here on. Splits into segments,
    processes each independently, then stitches results back together.
    Preserved exactly from legacy extra_processing_pipeline.
    """
    ecg_segments = _split_data_into_segments(ecg_df)
    lvp_segments = _split_data_into_segments(lvp_df)

    max_dpdt_segments = []
    min_dpdt_segments = []
    lvedp_segments = []
    r_peaks_timestamps_segments = []
    lvp_peaks_timestamps_segments = []
    dpdt_segments = []

    for i in range(len(ecg_segments)):
        ecg_segment = ecg_segments[i]
        lvp_segment = lvp_segments[i]

        (max_dpdt_segment, min_dpdt_segment, lvedp_segment,
         r_peaks_timestamps_segment, lvp_peaks_timestamps_segment, dpdt_segment) = _pipeline_segment(ecg_segment, lvp_segment)

        max_dpdt_segments.append(max_dpdt_segment)
        min_dpdt_segments.append(min_dpdt_segment)
        lvedp_segments.append(lvedp_segment)
        r_peaks_timestamps_segments.append(r_peaks_timestamps_segment)
        lvp_peaks_timestamps_segments.append(lvp_peaks_timestamps_segment)
        dpdt_segments.append(dpdt_segment)

    max_dpdt = _stitch_segments(max_dpdt_segments)
    min_dpdt = _stitch_segments(min_dpdt_segments)
    lvedp = _stitch_segments(lvedp_segments)
    r_peaks_timestamps = _stitch_segments(r_peaks_timestamps_segments)
    lvp_peaks_timestamps = _stitch_segments(lvp_peaks_timestamps_segments)
    dpdt_df = _stitch_segments(dpdt_segments)
    dpdt = dpdt_df["dpdt"]

    if dpdt_finetuning:
        dpdt_max_list, dpdt_max_time_list = _dpdt_max_finetuner(
            dpdt, max_dpdt["dpdt_max"].tolist(), max_dpdt["time"].tolist(), dpdt_df["time"].tolist(), resolution=dpdt_res
        )
        dpdt_max_list, dpdt_max_time_list = _finetune_signal(dpdt_max_list, dpdt_max_time_list)
        max_dpdt = pd.DataFrame({"time": dpdt_max_time_list, "dpdt_max": dpdt_max_list})

        dpdt_min_list, dpdt_min_time_list = _dpdt_min_finetuner(
            dpdt, min_dpdt["dpdt_min"].tolist(), min_dpdt["time"].tolist(), dpdt_df["time"].tolist()
        )
        dpdt_min_list, dpdt_min_time_list = _finetune_signal(dpdt_min_list, dpdt_min_time_list)
        min_dpdt = pd.DataFrame({"time": dpdt_min_time_list, "dpdt_min": dpdt_min_list})

    if lvedp_finetuning:
        lvedp_list, lvedp_time_list = _lvedp_finetuner(
            lvedp["lvedp"].tolist(), lvedp["time"].tolist(), lvp_df["lvp"].tolist(), lvp_df["time"].tolist(), resolution=lvedp_res
        )
        lvedp = pd.DataFrame({"time": lvedp_time_list, "lvedp": lvedp_list})

    return max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt


def _pipeline_segment(ecg_segment, lvp_segment):
    """
    Processes one small continuous ECG/LVP segment. Preserved exactly from
    legacy pipeline_segment, aside from the same list(...)-rebuilt-per-
    iteration fix applied elsewhere in this module (not a real bottleneck
    here since segments are capped at 10s, but fixed for consistency).
    """
    dpdt_segment = _calc_deriv_lvp(list(lvp_segment["lvp"]), list(lvp_segment["time"]))
    r_peaks_segment = _christov_beat_timestamps(list(ecg_segment["ecg"]))
    ecg_segment_time_list = list(ecg_segment["time"])
    r_peaks_timestamps_segment = [ecg_segment_time_list[r] for r in r_peaks_segment]
    lvp_peaks_segment = find_peaks(list(lvp_segment["lvp"]), height=50)
    lvp_segment_time_list = list(lvp_segment["time"])
    lvp_peaks_timestamps_segment = [lvp_segment_time_list[r] for r in list(lvp_peaks_segment[0])]
    r_peaks_timestamps_segment = _r_peak_corrector(r_peaks_timestamps_segment, lvp_peaks_timestamps_segment)

    lvedp_data_segment = []
    lvp_segment_time_to_index = {t: i for i, t in enumerate(list(lvp_segment["time"]))}
    reference_indices = [lvp_segment_time_to_index[x] for x in r_peaks_timestamps_segment]
    for ref_idx in reference_indices:
        lvedp_data_segment.append(lvp_segment["lvp"][ref_idx])

    lvedp_segment = pd.DataFrame({"time": r_peaks_timestamps_segment, "lvedp": lvedp_data_segment})
    dpdt_segment_df = pd.DataFrame({"time": lvp_segment["time"][:-1], "dpdt": dpdt_segment})

    max_dpdt_segment, min_dpdt_segment, _max_timestamps_segment = _dpdt_minmax_extractor(
        dpdt_segment, r_peaks_segment, list(ecg_segment["time"])
    )

    return max_dpdt_segment, min_dpdt_segment, lvedp_segment, r_peaks_timestamps_segment, lvp_peaks_timestamps_segment, dpdt_segment_df


# ── Low-level signal-processing helpers (all preserved exactly from legacy) ─

def _calc_deriv_lvp(lvp_values, time_values):
    # DEFENSIVE GUARD (last-resort safety net): the real fix for zero time-
    # differences is deduplicating exact-duplicate timestamps upstream, in
    # catheter_data_init.create_data_df/create_aop_data_df -- this should
    # already prevent difft==0 from ever reaching here. This guard exists
    # in case some other path (not yet identified) produces one anyway,
    # since a crash mid-pipeline is far more costly than a single
    # assumed-zero derivative point. Appends 0.0 (assume no instantaneous
    # change) rather than skipping -- skipping would shift every subsequent
    # index and break the positional alignment downstream code depends on
    # (dpdt[i] must correspond to the interval between time_values[i] and
    # time_values[i+1], exactly len(time_values)-1 entries).
    x = list(lvp_values)
    dxdt = []
    for i in range(len(x) - 1):
        diffx = x[i + 1] - x[i]
        dt1 = datetime.combine(datetime.today(), time_values[i + 1])
        dt2 = datetime.combine(datetime.today(), time_values[i])
        difft = (dt1 - dt2).total_seconds() * 1000
        if difft == 0:
            dxdt.append(0.0)
        else:
            dxdt.append(diffx / difft)
    return dxdt


def _christov_beat_timestamps(ext_ecg):
    detectors = Detectors(250)
    return detectors.christov_detector(ext_ecg)


def _r_peak_corrector(r_peak_timestamps, lvp_peaks_timestamps):
    # PERFORMANCE NOTE (efficiency-only, provably identical output): added
    # `break` once a match is found -- the outer logic only cares whether
    # ANY lvp peak falls in [curr_r_peak, next_r_peak] (a boolean OR,
    # order-independent), so stopping the inner scan early never changes
    # the result, just avoids continuing to scan lvp_peaks_timestamps
    # pointlessly after the answer is already known. Matters at the scale
    # of large continuous coarse windows (thousands of peaks), where legacy's
    # unconditional full inner scan became a real bottleneck.
    r_peaks = []
    for rts_index in range(len(r_peak_timestamps) - 1):
        true_r_peak = False
        curr_r_peak = r_peak_timestamps[rts_index]
        next_r_peak = r_peak_timestamps[rts_index + 1]

        for lvpp_index in range(len(lvp_peaks_timestamps)):
            curr_lvp_peak = lvp_peaks_timestamps[lvpp_index]
            if curr_r_peak <= curr_lvp_peak <= next_r_peak:
                true_r_peak = True
                break

        if true_r_peak:
            r_peaks.append(r_peak_timestamps[rts_index])

    return r_peaks


def _dpdt_minmax_extractor(dpdt, peak_indices, time_data):
    max_per_beat = []
    min_per_beat = []
    max_times = []
    min_times = []

    for i in range(len(peak_indices) - 1):
        start = peak_indices[i]
        end = peak_indices[i + 1]

        dpdt_segment = dpdt[start:end]
        time_segment = time_data[start:end]

        max_dpdt = max(dpdt_segment)
        min_dpdt = min(dpdt_segment)

        max_index = dpdt_segment.index(max_dpdt)
        min_index = dpdt_segment.index(min_dpdt)

        max_per_beat.append(max_dpdt)
        min_per_beat.append(min_dpdt)
        max_times.append(time_segment[max_index])
        min_times.append(time_segment[min_index])

    max_df = pd.DataFrame({"time": max_times, "dpdt_max": max_per_beat})
    min_df = pd.DataFrame({"time": min_times, "dpdt_min": min_per_beat})

    return max_df, min_df, max_times


def _detect_outliers(dataset_x, dataset_y):
    """
    Legacy accepted an unused `label` parameter here (dead code, never
    referenced inside the function) -- dropped, per confirmed cleanup.
    """
    outliers = []
    ds_mean = dataset_y.mean()
    ds_std = dataset_y.std()
    upper = ds_mean + 2 * ds_std
    lower = ds_mean - 2 * ds_std

    for i in range(len(dataset_y)):
        point_x = dataset_x[i]
        point_y = dataset_y[i]
        if point_y > upper or point_y < lower:
            outliers.append((point_x, point_y))

    return pd.DataFrame(outliers, columns=["time", "data"])


def _remove_outliers(data, outliers):
    data_col = data.columns[1]
    outliers_removed = data[~data[data_col].isin(outliers["data"])]
    outliers_removed.reset_index(drop=True, inplace=True)
    return outliers_removed


def _stitch_segments(list_of_segments):
    if isinstance(list_of_segments[0], list):
        full = []
        for segment in list_of_segments:
            full.extend(segment)
    else:
        full = pd.DataFrame()
        for segment in list_of_segments:
            full = pd.concat([full, segment], ignore_index=True)
    return full


def _split_data_into_segments(data, max_segment_length=10, sampling_frequency=250):
    time_data = data["time"]
    anchored_time = [datetime.combine(datetime.today(), t) for t in time_data]
    list_of_segments = []
    current_segment = pd.DataFrame()
    max_data_points = max_segment_length * sampling_frequency

    for i in range(len(data) - 1):
        curr_row = data.iloc[[i]]
        curr_time = anchored_time[i]
        next_time = anchored_time[i + 1]
        diff = (next_time - curr_time).total_seconds()

        if diff > 0.01 or len(current_segment) >= max_data_points:
            current_segment = pd.concat([current_segment, curr_row], ignore_index=True)
            list_of_segments.append(current_segment)
            current_segment = pd.DataFrame()
        else:
            current_segment = pd.concat([current_segment, curr_row], ignore_index=True)

    return list_of_segments


def _finetune_signal(signal_data, signal_time_axis):
    """Removes extraneous values detected due to noise by controlling for minimum period (heart rate)."""
    selected = []
    selected_timestamps = []
    idx = 0

    while idx < len(signal_data) - 1:
        curr = signal_data[idx]
        selected.append(curr)
        selected_timestamps.append(signal_time_axis[idx])

        anchored_time_curr = datetime.combine(datetime.today(), signal_time_axis[idx])
        anchored_time_next = datetime.combine(datetime.today(), signal_time_axis[idx + 1])
        time_diff = anchored_time_next - anchored_time_curr

        if time_diff > timedelta(milliseconds=400):  # period for 150bpm HR
            idx += 1
        else:
            idx += 2

    return selected, selected_timestamps


def _lvedp_finetuner(lvedp, lvedp_timestamps, lvp, lvp_timestamps, resolution=10):
    """
    resolution = number of data points examined before/after the identified
    r-peak timestamp. 10 points = 40ms of data.

    QUIRK PRESERVED (#1 above): `except KeyError` can never fire here --
    list indexing raises IndexError, not KeyError. If the search-window
    index goes negative, Python silently wraps to read from the end of the
    list rather than raising. Not fixed, per "preserve legacy computation
    exactly" -- flagged in the module docstring.

    PERFORMANCE FIX (efficiency-only, provably identical output): replaced
    a per-iteration list.index() linear scan with a precomputed dict lookup
    -- see process_phase()'s matching fix for the full explanation. Same
    O(n*m) -> O(n+m) improvement, same guarantee of identical results.
    """
    lvp_time_to_index = {t: i for i, t in enumerate(lvp_timestamps)}
    for i in range(len(lvedp_timestamps)):
        index_on_lvp = lvp_time_to_index[lvedp_timestamps[i]]
        real_lvedp = lvedp[i]
        real_lvedp_timestamp = lvedp_timestamps[i]

        if real_lvedp > 25:
            for j in range(index_on_lvp - resolution, index_on_lvp + resolution + 1):
                if lvp[j] <= 30:
                    try:
                        real_lvedp = lvp[j]
                        real_lvedp_timestamp = lvp_timestamps[j]
                    except KeyError:
                        j += 1
            lvedp[i] = real_lvedp
            lvedp_timestamps[i] = real_lvedp_timestamp

    return lvedp, lvedp_timestamps


def _dpdt_max_finetuner(dpdt, dpdt_max, dpdt_max_timestamps, dpdt_timestamps, resolution=15):
    """
    resolution = number of data points examined before/after the identified
    r-peak timestamp. 15 points = 60ms of data.

    QUIRK PRESERVED (#2 above): guards the upper bound only.

    PERFORMANCE FIX (efficiency-only, provably identical output): replaced
    a per-iteration list.index() linear scan with a precomputed dict lookup
    -- see process_phase()'s matching fix for the full explanation. Same
    O(n*m) -> O(n+m) improvement, same guarantee of identical results.
    """
    dpdt_time_to_index = {t: i for i, t in enumerate(dpdt_timestamps)}
    for i in range(len(dpdt_max_timestamps)):
        index_on_dpdt = dpdt_time_to_index[dpdt_max_timestamps[i]]
        real_dpdt_max = dpdt_max[i]
        real_dpdt_max_timestamp = dpdt_max_timestamps[i]

        for j in range(index_on_dpdt - resolution, index_on_dpdt + resolution + 1):
            if j >= len(dpdt):
                break
            if dpdt[j] > real_dpdt_max:
                real_dpdt_max = dpdt[j]
                real_dpdt_max_timestamp = dpdt_timestamps[j]

        dpdt_max[i] = real_dpdt_max
        dpdt_max_timestamps[i] = real_dpdt_max_timestamp

    return dpdt_max, dpdt_max_timestamps


def _dpdt_min_finetuner(dpdt, dpdt_min, dpdt_min_timestamps, dpdt_timestamps, resolution=15):
    """
    resolution = number of data points examined before/after the identified
    r-peak timestamp. 15 points = 60ms of data.

    QUIRK PRESERVED (#2 above): has NO bounds guard at all -- unlike
    dpdt_max_finetuner, an out-of-range index here will raise IndexError
    rather than being silently skipped. Not fixed, per "preserve legacy
    computation exactly."

    PERFORMANCE FIX (efficiency-only, provably identical output): replaced
    a per-iteration list.index() linear scan with a precomputed dict lookup
    -- see process_phase()'s matching fix for the full explanation. Same
    O(n*m) -> O(n+m) improvement, same guarantee of identical results.
    """
    dpdt_time_to_index = {t: i for i, t in enumerate(dpdt_timestamps)}
    for i in range(len(dpdt_min_timestamps)):
        index_on_dpdt = dpdt_time_to_index[dpdt_min_timestamps[i]]
        real_dpdt_min = dpdt_min[i]
        real_dpdt_min_timestamp = dpdt_min_timestamps[i]

        for j in range(index_on_dpdt - resolution, index_on_dpdt + resolution + 1):
            if dpdt[j] < real_dpdt_min:
                real_dpdt_min = dpdt[j]
                real_dpdt_min_timestamp = dpdt_timestamps[j]

        dpdt_min[i] = real_dpdt_min
        dpdt_min_timestamps[i] = real_dpdt_min_timestamp

    return dpdt_min, dpdt_min_timestamps