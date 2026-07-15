"""
shared/catheter_phase_config.py

Per-animal phase lists and arrhythmia-exclusion windows for the catheter-derived
(dp/dt max/min, LVEDP) processing pipeline. Extracted verbatim from the legacy
process.py, where they lived as hardcoded if/elif branches (per-animal phase
lists in combined_phase_data, exclusion windows in arrythmia_removal).

Values below are UNCHANGED from the legacy code -- same phases, same order,
same excluded timestamps. Only the representation changed: data structures
instead of branching code, so adding/correcting an entry doesn't require
touching pipeline logic.

Scope: 4-animal normal cohort only (202, 203, 205, 221).
"""

from datetime import time


# ── Per-animal phase lists ────────────────────────────────────────────────────
# Order matters -- consumed positionally downstream (e.g. plots.py's P6/P3
# alternation-by-index logic). Preserved exactly as in legacy combined_phase_data.
#
# NOTE: 205's list has 17 entries (odd) -- every other animal has an even count
# alternating P6/P3. 205 ends in "...Dobu_high_P6" with no matching
# "Dobu_high_P3". Flagged to user 2024 refactor session -- confirm this is a
# true data gap (P3 not recorded at that state) rather than a legacy omission,
# since any downstream code assuming strict P6/P3 alternation-by-position will
# silently misassign values past this point for animal 205.
ANIMAL_PHASES = {
    "221": [
        "221_Baseline_0_P6", "221_Baseline_0_P3",
        "221_Nitro_low_P6", "221_Nitro_low_P3",
        "221_Nitro_high_P6", "221_Nitro_high_P3",
        "221_Washout_0_P6", "221_Washout_0_P3",
        "221_Phen_low_P6", "221_Phen_low_P3",
        "221_Phen_high_P6", "221_Phen_high_P3",
        "221_Washout_1_P6", "221_Washout_1_P3",
        "221_Dobu_low_P6", "221_Dobu_low_P3",
    ],
    "205": [
        "205_Baseline_0_P6", "205_Baseline_0_P3",
        "205_Nitro_low_P6", "205_Nitro_low_P3",
        "205_Nitro_high_P6", "205_Nitro_high_P3",
        "205_Washout_1_P6", "205_Washout_1_P3",
        "205_Phen_low_P6", "205_Phen_low_P3",
        "205_Phen_high_P6", "205_Phen_high_P3",
        "205_Washout_2_P6", "205_Washout_2_P3",
        "205_Dobu_low_P6", "205_Dobu_low_P3",
        "205_Dobu_high_P6",  # <-- no matching _P3 entry; see NOTE above
    ],
    "203": [
        "203_Baseline_0_P6", "203_Baseline_0_P3",
        "203_Nitro_low_P6", "203_Nitro_low_P3",
        "203_Nitro_high_P6", "203_Nitro_high_P3",
        "203_Washout_0_P6", "203_Washout_0_P3",
        "203_Phen_low_P6", "203_Phen_low_P3",
        "203_Phen_high_P6", "203_Phen_high_P3",
        "203_Washout_1_P6", "203_Washout_1_P3",
        "203_Dobu_low_P6", "203_Dobu_low_P3",
        "203_Dobu_high_P6", "203_Dobu_high_P3",
        "203_Washout_2_P6", "203_Washout_2_P3",
        "203_Esmo_low_P6", "203_Esmo_low_P3",
    ],
    "202": [
        "202_Baseline_0_P6", "202_Baseline_0_P3",
        "202_Nitro_low_P6", "202_Nitro_low_P3",
        "202_Nitro_high_P6", "202_Nitro_high_P3",
        "202_Washout_0_P6", "202_Washout_0_P3",
        "202_Phen_0_P6", "202_Phen_0_P3",
        "202_Washout_1_P6", "202_Washout_1_P3",
        "202_Dobu_0_P6", "202_Dobu_0_P3",
        "202_Washout_2_P6", "202_Washout_2_P3",
        "202_Esmo_0_P6", "202_Esmo_0_P3",
    ],
}


# ── Arrhythmia exclusion windows ──────────────────────────────────────────────
# Keyed by phase label. Each value is a list of (start, end) time tuples to
# exclude from both the ECG and LVP data for that phase before processing.
# Timestamps preserved EXACTLY as in legacy arrythmia_removal -- these are
# manually curated research decisions (known arrhythmia artifacts at specific
# moments), not something to regenerate or approximate.
ARRHYTHMIA_EXCLUSION_WINDOWS = {
    "203_Nitro_high_P3": [
        (time(10, 58, 36), time(10, 58, 42)),
        (time(10, 59, 14), time(10, 59, 16)),
    ],
    "203_Nitro_low_P3": [
        (time(10, 32, 29), time(10, 32, 30)),
        (time(10, 32, 38), time(10, 32, 41)),
        (time(10, 32, 52), time(10, 32, 54)),
        (time(10, 33, 9), time(10, 33, 10)),
        (time(10, 33, 12), time(10, 33, 14)),
        (time(10, 33, 22), time(10, 33, 24)),
        (time(10, 33, 41), time(10, 33, 42)),
    ],
    "202_Phen_0_P3": [
        (time(12, 59, 46), time(12, 59, 55)),
        (time(12, 59, 57), time(13, 0, 24)),
        (time(13, 0, 47), time(13, 0, 49)),
        (time(13, 1, 6), time(13, 1, 10)),
        (time(13, 1, 56), time(13, 2, 13)),
        (time(13, 2, 49), time(13, 2, 50)),
    ],
    "202_Washout_1_P3": [
        (time(13, 36, 0), time(13, 36, 21)),
    ],
}

# Phases requiring the segment-splitting "extra processing" pipeline
# (lvedp_finetuning=True path in legacy pipeline()). In the legacy code this
# was a SEPARATE hardcoded list that happened to contain exactly the same four
# phases as ARRHYTHMIA_EXCLUSION_WINDOWS above -- which makes sense: removing
# arrhythmia windows creates gaps in the data, and those gaps are exactly why
# the segment-splitting extra-processing path is needed. Rather than maintain
# two lists that could silently drift apart, this is now derived directly from
# ARRHYTHMIA_EXCLUSION_WINDOWS. Confirm this equivalence is intentional (i.e.
# "needs extra processing" and "has arrhythmia exclusions" are meant to always
# be the same set) -- if there's ever a phase that needs one but not the other,
# this derivation would be wrong and they'd need to go back to being separate.
PHASES_REQUIRING_EXTRA_PROCESSING = list(ARRHYTHMIA_EXCLUSION_WINDOWS.keys())