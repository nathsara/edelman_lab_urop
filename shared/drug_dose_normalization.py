"""
shared/drug_dose_normalization.py

Post-processing step, run LAST in the pipeline: overwrites the `dose`
column in every already-generated summary pickle that has one, replacing
the categorical label ("low"/"high"/"0") with a continuous 0-1 value
representing how much of that drug the animal had been exposed to by that
phase -- per-drug min-max normalization of Cumulative mcg/kg, pooled across
all 4 normal-cohort animals.

Source: data/raw/drug_normalization.csv (real recorded dose-administration
log -- Study Date, VBU #, Drug Type, Dose, Unit, Start Time, End Time,
Animal Weight, Time elapsed, mcg/kg, Cumulative mcg/kg). Manually curated,
not regenerable -- tracked in git despite living under data/raw/, same
justified exception as drug_start_end_times.csv.

ALIGNMENT STRATEGY: each row's real Start Time/End Time is matched against
each fine phase's real start time (re-derived via
catheter_data_init.timestamps_from_animal(), reading the same TDvAIC.csv
already used everywhere else) via time-range CONTAINMENT -- a fine phase
matches a dose row if the fine phase's start time falls within
[row Start Time, row End Time]. This is precise, numeric matching, not
name-guessing -- a single dose-administration row typically contains TWO
fine phases (the P6 sub-window and P3 sub-window within that one dose
period), both getting the same normalized value.

CONVENTIONS (user-confirmed):
  - Baseline phases -> 0 (no drug exposure yet)
  - Washout phases -> NaN (dose info not used for washout downstream)
  - Esmo phases -> NaN (no Esmolol rows exist in the source CSV at all, and
    Esmo is excluded from all analysis anyway -- no relevant data for it)

KNOWN DISCREPANCY, not a bug: 221 has TWO recorded Dobu dose
administrations in the source CSV (1.5 and 2.5 mcg/kg/min), but
catheter_phase_config.ANIMAL_PHASES["221"] only has ONE Dobu fine-phase
pair (no "_high") -- the P-level tracking data never captured a second
Dobu phase for this animal, even though the drug really was given twice.
The second CSV row simply matches no fine phase and is silently skipped --
nothing to attach it to.

NOT a hard-halt-on-error step, unlike the rest of this pipeline: unmatched
CSV rows or unmatched drug-phase fine-phases print a WARNING and continue,
rather than raising. This is deliberate -- one legitimate unmatched-row
case is already known and expected (above), and this step assigns metadata
to already-correct, already-confirmed signal-processing output rather than
computing a new physiological value, so a partial, warning-annotated result
is more useful than an all-or-nothing halt. Flag if you'd rather this halt.

Always overwrites fully from the source CSV each run -- naturally
idempotent, safe to re-run.
"""

from pathlib import Path
from datetime import datetime

import pandas as pd

from .catheter_data_init import timestamps_from_animal
from .catheter_phase_config import ANIMAL_PHASES

NORMAL_COHORT_IDS = ["202", "203", "205", "221"]

DRUG_TYPE_TO_MED = {
    "Nitroprusside": "Nitro",
    "Phenylephrine": "Phen",
    "Dobutamine": "Dobu",
}


def _load_dose_log(drug_normalization_csv):
    """
    Loads and cleans the raw dose-administration log. Drops fully-blank
    trailing rows (the source file has many). Parses Start Time/End Time
    into time objects, maps Drug Type to this project's med naming
    (Nitro/Phen/Dobu), and maps VBU # (int) to animal_id (str) matching
    every other module's convention.
    """
    df = pd.read_csv(drug_normalization_csv).dropna(subset=["VBU #"])
    df = df.copy()
    df["animal_id"] = df["VBU #"].astype(int).astype(str)
    df["med"] = df["Drug Type"].map(DRUG_TYPE_TO_MED)
    unmapped = df[df["med"].isna()]
    if not unmapped.empty:
        raise ValueError(
            f"Unrecognized Drug Type value(s) in {drug_normalization_csv}: "
            f"{unmapped['Drug Type'].unique().tolist()}. Add to DRUG_TYPE_TO_MED "
            f"if this is a real new drug, rather than silently dropping it."
        )
    df["start_time"] = df["Start Time"].apply(lambda s: datetime.strptime(s, "%H:%M:%S").time())
    df["end_time"] = df["End Time"].apply(lambda s: datetime.strptime(s, "%H:%M:%S").time())
    return df


def _per_drug_min_max(dose_log):
    """Per-drug min/max of Cumulative mcg/kg, pooled across all animals -- per user's explicit method."""
    ranges = {}
    for med in dose_log["med"].unique():
        vals = dose_log[dose_log["med"] == med]["Cumulative mcg/kg"]
        ranges[med] = (vals.min(), vals.max())
    return ranges


def _normalized_dose_map(dose_log, min_max_by_drug, animal_id):
    """
    Builds {fine_phase_label: normalized_dose_value} for one animal, by
    matching each dose-log row (real Start/End time) against each fine
    phase's real start time (time-range containment).

    Returns a dict covering ALL of ANIMAL_PHASES[animal_id] -- Baseline
    entries mapped to 0, Washout and Esmo entries mapped to NaN, drug
    phases (Nitro/Phen/Dobu) mapped to their per-drug min-max normalized
    cumulative-exposure value.
    """
    animal_rows = dose_log[dose_log["animal_id"] == animal_id]

    # Re-derive real fine-phase start times from the same TDvAIC.csv source
    # every other module in this pipeline already uses -- not stored
    # anywhere as a standalone pickle, so recomputed here (cheap, just a
    # CSV read).
    animal_summary_csv = _find_tdvaic_csv(animal_id)
    start_times, _end_times, raw_labels = timestamps_from_animal(animal_summary_csv)
    # CASE-INSENSITIVE MATCH, confirmed necessary: the real TDvAIC.csv's Dose
    # column uses text casing (e.g. "Low"/"High") that doesn't always match
    # ANIMAL_PHASES's hardcoded casing (e.g. "low"/"high") -- confirmed via a
    # direct screenshot of 202's real CSV. This was invisible everywhere else
    # in the pipeline because every other use of these labels was as a FILE
    # PATH (e.g. "..._Low_P6_ecg_raw.pkl"), and Windows filesystems resolve
    # file paths case-insensitively -- so it silently never mattered until
    # now, the first place doing a plain in-memory string/dict match (which
    # is always case-sensitive, on any OS). Not fixing ANIMAL_PHASES itself
    # -- that's used successfully everywhere else; safer to make just this
    # new matching logic robust to casing than touch an established list.
    phase_starts = {raw_label.lower(): start for raw_label, start in zip(raw_labels, start_times)}

    result = {}
    for label in ANIMAL_PHASES[animal_id]:
        # label is e.g. "202_Nitro_low_P6" -- strip the animal prefix to
        # match phase_starts' keys (raw_labels from timestamps_from_animal
        # don't include it).
        raw_label = label[len(animal_id) + 1:]
        med = raw_label.split("_")[0]

        if med == "Baseline":
            result[label] = 0.0
            continue
        if med == "Washout":
            result[label] = float("nan")
            continue
        if med == "Esmo":
            result[label] = float("nan")
            continue

        phase_start = phase_starts.get(raw_label.lower())
        if phase_start is None:
            print(f"  WARNING: no start time found for fine phase {label!r} -- dose left unassigned.")
            continue

        matches = animal_rows[
            (animal_rows["med"] == med)
            & (animal_rows["start_time"] <= phase_start)
            & (phase_start <= animal_rows["end_time"])
        ]
        if matches.empty:
            print(f"  WARNING: no dose-log row matches fine phase {label!r} (med={med!r}, "
                  f"start={phase_start}) -- dose left unassigned.")
            continue
        if len(matches) > 1:
            print(f"  WARNING: {len(matches)} dose-log rows matched fine phase {label!r} -- "
                  f"using the first. Check for overlapping dose windows in the source CSV.")

        cumulative = matches.iloc[0]["Cumulative mcg/kg"]
        dose_min, dose_max = min_max_by_drug[med]
        result[label] = (cumulative - dose_min) / (dose_max - dose_min)

    return result


def _find_tdvaic_csv(animal_id, raw_root=None):
    """Locates {animal_id}_TDvAIC.csv under data/raw/, matching raw_data_processing.py's own lookup logic."""
    raw_root = Path(raw_root) if raw_root else Path("data") / "raw"
    matches = list(raw_root.glob(f"VBU_{animal_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No VBU folder found for animal {animal_id} under {raw_root}")
    vbu_folder = matches[0]
    csv_path = vbu_folder / f"{animal_id}_TDvAIC.csv"
    if not csv_path.exists():
        alt_path = vbu_folder / f"{animal_id}_AICvTD.csv"
        if alt_path.exists():
            return alt_path
        raise FileNotFoundError(f"Could not find {animal_id}_TDvAIC.csv (or _AICvTD.csv) in {vbu_folder}")
    return csv_path


def _overwrite_dose_column(pickle_path, dose_map):
    """
    Loads a summary pickle, overwrites its `dose` column in place using
    dose_map, and re-saves.

    IDEMPOTENCY FIX: labels are looked up via `phase_number` (1-indexed
    position within ANIMAL_PHASES[animal_id], immutable -- never touched by
    this function) rather than reconstructed from the `dose` column itself.
    The original approach reconstructed the match-key as
    "{animal_id}_{med}_{dose}_{p_level}" -- but `dose` is exactly the
    column THIS function overwrites, so on any run after the first, `dose`
    is already a float (or NaN), and the reconstructed key could never
    match anything again. This silently broke re-running the step at all
    -- confirmed via real output where already-normalized floats (e.g.
    0.8537859007832898) appeared in "not found in dose_map" warnings,
    meaning the row HAD been correctly processed before, but a second run
    couldn't verify or reapply anything. phase_number never changes, so
    this lookup works identically on the 1st, 2nd, or 100th run.
    """
    df = pd.read_pickle(pickle_path)
    if "dose" not in df.columns:
        print(f"  [SKIP] {pickle_path.name} has no 'dose' column.")
        return

    animal_id = pickle_path.name.split("_")[0]
    phases = ANIMAL_PHASES[animal_id]

    new_doses = []
    for _, row in df.iterrows():
        phase_number = int(row["phase_number"])
        if not (1 <= phase_number <= len(phases)):
            print(f"  WARNING: phase_number {phase_number} out of range for animal "
                  f"{animal_id} (expected 1-{len(phases)}) -- leaving original value.")
            new_doses.append(row["dose"])
            continue

        label = phases[phase_number - 1]
        if label in dose_map:
            new_doses.append(dose_map[label])
        else:
            # Should not happen now that dose_map covers every phase in
            # ANIMAL_PHASES (Baseline/Washout/Esmo all explicitly assigned)
            # -- defensive fallback only, in case of an unexpected label
            # mismatch. Leaves the original categorical value in place and
            # warns, rather than silently dropping data.
            print(f"  WARNING: {label!r} not found in dose_map -- leaving original value.")
            new_doses.append(row["dose"])

    df["dose"] = new_doses
    df.to_pickle(pickle_path)
    print(f"  [OK] {pickle_path.name} dose column updated.")


def normalize_all_dose_columns(repo_root, drug_normalization_csv=None):
    """
    Runs the full dose-normalization step: for every normal-cohort animal,
    overwrites the `dose` column in {animal_id}_catheter_summary.pkl and
    {animal_id}_phase_summary.pkl (baseline-cohort animals have no `dose`
    column at all -- skipped automatically via the has-column check in
    _overwrite_dose_column).
    """
    repo_root = Path(repo_root)
    drug_normalization_csv = Path(drug_normalization_csv) if drug_normalization_csv else (
        repo_root / "data" / "raw" / "drug_normalization.csv"
    )

    dose_log = _load_dose_log(drug_normalization_csv)
    min_max_by_drug = _per_drug_min_max(dose_log)
    print(f"Per-drug (Cumulative mcg/kg) min/max, pooled across all animals: {min_max_by_drug}")

    catheter_dir = repo_root / "data" / "processed" / "catheter_derived" / "summary_data"
    impella_dir = repo_root / "data" / "processed" / "impella_derived" / "summary_data"

    for animal_id in NORMAL_COHORT_IDS:
        print(f"[{animal_id}] Building normalized dose map...")
        dose_map = _normalized_dose_map(dose_log, min_max_by_drug, animal_id)

        catheter_path = catheter_dir / f"{animal_id}_catheter_summary.pkl"
        if catheter_path.exists():
            _overwrite_dose_column(catheter_path, dose_map)
        else:
            print(f"  [SKIP] {catheter_path.name} not found.")

        impella_path = impella_dir / f"{animal_id}_phase_summary.pkl"
        if impella_path.exists():
            _overwrite_dose_column(impella_path, dose_map)
        else:
            print(f"  [SKIP] {impella_path.name} not found.")