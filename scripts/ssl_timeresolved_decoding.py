"""Shared library for `projects/ssl-whisker-hitmiss-timeresolved-decoding/`:
time-resolved (10ms, disjoint) single-trial population decoding of
cohort-corrected whisker hit/miss, per brain area, split by day-stage,
cohort, and session-half. Deliberately reuses (does not reimplement) the
sibling `ssl-bwm-style-single-cell-decoding` project's already-locked
infrastructure:
- `ssl_bwm_trial_prep.list_whisker_training_ephys_sessions`/`prep_session`/
  `cohort_corrected_rewarded` for cohort-corrected trial labels.
- `ssl_bwm_windows.clip_window_for_whisker` for the mandatory whisker
  stimulus-artifact dead zone (`-1ms/+4ms` around `start_time`,
  `ssl-analyze/references/ssl_artifact_dead_zone.md`).
- `ssl_bwm_decoding`'s `population_design_matrix` pattern (per-unit
  `unit_rates_for_trials` against one loaded spike shard) and L1-logistic
  BWM-style decoding building block.

Design decisions locked with the user 2026-09-10 (see project `question.md`):
10ms disjoint bins (no smoothing), bin grid offset by 5ms so the dead zone
falls entirely inside one dropped bin, per-bin significance via label-shuffle
+ cluster-based permutation (not a full imposter-session null at every bin),
exploratory small-scale pass before any full sweep, both area parcellation
schemes (`area_group`, `area_acronym_custom`) from the start.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Cosmetic only: this environment's scikit-learn emitted a FutureWarning +
# UserWarning on every single `LogisticRegression(penalty="l1", ...)` fit
# (deprecated in favor of `l1_ratio`) -- with thousands of fits per
# condition this balloons log files by ~4x with no informational value
# (verified 2026-09-10, smoke test: 300k of ~300k log lines were this pair).
# Kept in place after the 2026-09-13 switch to `penalty="l2"` even though
# that specific pair was l1-triggered -- harmless if it no longer fires,
# and cheap insurance against l2 tripping something similar. Behavior is
# unaffected; does not silence warnings from other modules.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssl_bwm_decoding import C_GRID
from ssl_bwm_trial_prep import (
    list_whisker_training_ephys_sessions, load_reward_group, prep_session, session_training_day,
)
from ssl_bwm_windows import clip_window_for_whisker, unit_rates_for_trials
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
from axel_bisi_paths import axel_bisi_path

MOUSE_REF_PATH = Path("reports/ssl_analysis/derived/mouse_reference.parquet")
AREA_LABELS_PATH = Path("reports/ssl_analysis/derived/unit_area_labels.parquet")

QC_VALUES = ("good", "mua")
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_PER_CLASS = 3  # per class, evaluated *after* the session-half split (user decision 2026-09-28, down from 5;
# 2026-09-10: down from 8). Pooled CV then uses n_folds = min(5, minority count) -- see `_effective_folds`.

BIN_WIDTH_S = 0.010
DEAD_ZONE_START_S = -0.001  # start_time - 1ms
DEAD_ZONE_STOP_S = 0.004    # start_time + 4ms

# Corrected first-lick time (skills/ssl-lick-alignment, user request 2026-09-27): the stored
# `lick_time` is late by the per-session artifact window (`response_window_start_time - start_time`,
# 100 ms in 873/894 sessions, 50/150/300 ms in the rest), so every lick-aligned window and RT uses
# `first_lick_time` = start_time + (lick_time - response_window_start_time), computed per trial.
# SSL_CORRECT_LICK_TIME=0 reproduces runs made before 2026-09-27 (first_lick_time = stored lick_time).
CORRECT_LICK_TIME = __import__("os").environ.get("SSL_CORRECT_LICK_TIME", "1") != "0"
LICK_TIME_DEFINITION = ("first_lick_time = start_time + lick_time - response_window_start_time (corrected)"
                        if CORRECT_LICK_TIME else "stored lick_time (uncorrected, late by the artifact window)")


def add_first_lick_time(trials: pd.DataFrame) -> pd.DataFrame:
    """Adds `reaction_time` (s, from stimulus onset) and `first_lick_time` (physical first lick,
    trial clock) per trial, keeping the stored `lick_time` untouched (skills/ssl-lick-alignment).
    With CORRECT_LICK_TIME False both reproduce the old convention (stored lick_time,
    rt = lick_time - start_time). NaN where lick_time is NaN (no lick)."""
    trials = trials.copy()
    if CORRECT_LICK_TIME:
        trials["reaction_time"] = trials["lick_time"] - trials["response_window_start_time"]
        trials["first_lick_time"] = trials["start_time"] + trials["reaction_time"]
    else:
        trials["reaction_time"] = trials["lick_time"] - trials["start_time"]
        trials["first_lick_time"] = trials["lick_time"]
    return trials


# ---------------------------------------------------------------------------
# Session / mouse selection
# ---------------------------------------------------------------------------

def hitmiss_session_list(sessions_tbl: pd.DataFrame) -> pd.DataFrame:
    """(session_id, day_stage, subject_id, reward_group, learning_category)
    for every has-ephys whisker-training session whose subject passes the
    mandatory mouse-inclusion filters (`exclude==0`, `exclude_ephys==0`,
    `reward_group` in {R+, R-}, `R+proba` dropped) -- `ssl_task_semantics.md`.

    Fixes a gap in the sibling project's `list_whisker_training_ephys_sessions`/
    `load_reward_group`, which do not apply `exclude`/`exclude_ephys` (see
    `question.md`'s "Gap found" note).
    """
    pairs = list_whisker_training_ephys_sessions(sessions_tbl)
    ref = pd.read_parquet(MOUSE_REF_PATH)
    ref_ok = ref[
        (ref["exclude"] == 0) & (ref["exclude_ephys"] == 0) & (ref["reward_group"].isin(["R+", "R-"]))
    ]
    sid2sub = sessions_tbl.set_index("session_id")["subject_id"]
    rows = []
    for sid, stage in pairs:
        sub = sid2sub.get(sid)
        m = ref_ok[ref_ok["subject_id"] == sub]
        if len(m) != 1:
            continue
        rows.append(
            {
                "session_id": sid,
                "day_stage": stage,
                "subject_id": sub,
                "reward_group": m.iloc[0]["reward_group"],
                "learning_category": m.iloc[0]["learning_category"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trial prep: whisker-only, cohort-corrected hit/miss, session-half split
# ---------------------------------------------------------------------------

def prep_hitmiss_trials(dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> pd.DataFrame | None:
    """Active-epoch whisker trials for `session_id` with a cohort-corrected
    `rewarded` (hit) column and a `half` column (first/second, split at the
    median chronological trial index within this session). None if the
    session's subject has no usable reward_group."""
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    trials = prepped["trials"]
    trials = trials[trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
    if len(trials) == 0:
        return None
    median_idx = len(trials) // 2
    trials["half"] = ["first"] * median_idx + ["second"] * (len(trials) - median_idx)
    trials["reward_group"] = prepped["reward_group"]
    trials["subject_id"] = prepped["subject_id"]
    return trials


def prep_modality_trials(dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> pd.DataFrame | None:
    """Active-epoch whisker + auditory trials for `session_id`, **any**
    `lick_flag` (unlike `prep_lick_aligned_trials`, which restricts to
    lick_flag==1 for the lick-time alignment) -- for the start_time-aligned
    modality decode, which needs no lick event to anchor to, so uses the
    full trial set. `half` column: chronological median split, same
    convention as `prep_hitmiss_trials`."""
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    trials = prepped["trials"]
    trials = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
    if len(trials) == 0:
        return None
    median_idx = len(trials) // 2
    trials["half"] = ["first"] * median_idx + ["second"] * (len(trials) - median_idx)
    trials["reward_group"] = prepped["reward_group"]
    trials["subject_id"] = prepped["subject_id"]
    return trials


PERF_BLOCK_SIZE = 5  # consecutive whisker trials per performance-state block (user decision, 2026-09-11)


def prep_perfstate_trials(dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame,
                           block_size: int = PERF_BLOCK_SIZE) -> pd.DataFrame | None:
    """Active-epoch whisker trials for `session_id` with a `perf_state`
    column (high/low) instead of `prep_hitmiss_trials`'s chronological
    `half` -- consecutive blocks of `block_size` whisker trials, each
    block's cohort-corrected hit rate computed, then a per-session median
    split of block hit rates assigns every trial in a block to 'high' or
    'low'. Trailing whisker trials that don't fill a full block are
    dropped (kept blocks all have exactly `block_size` trials, so block
    hit rate is comparably granular throughout).

    Also computes `block_fa_rate`: the false-alarm rate (lick rate on
    `no_stim_trial` trials) within each block's own time span (from its
    first to its last whisker trial's `start_time`, inclusive) -- reported
    for validation/context, not used to define `perf_state` itself. NaN if
    no no-stim trials fall in that span (real gap, not imputed).
    """
    return prep_perfstate_trials_generic(
        dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"], block_size=block_size,
    )


def prep_perfstate_trials_generic(
    dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame,
    decode_trial_types: list[str], block_size: int = PERF_BLOCK_SIZE, licked_only: bool = False,
) -> pd.DataFrame | None:
    """Generalization of `prep_perfstate_trials` (user request 2026-09-11):
    performance-state blocks are always defined from **whisker** trials'
    cohort-corrected hit rate (blocks of `block_size` consecutive whisker
    trials, per-session median split), but the returned, perf_state-labeled
    trial set can be any `decode_trial_types` -- e.g. `['whisker_trial',
    'auditory_trial']` for the modality decode under perf-state, not just
    `['whisker_trial']` for hit/miss. A non-whisker decode trial is assigned
    to whichever whisker block's own time span (`min`..`max` start_time)
    contains its `start_time` (same time-span-membership logic already used
    for `block_fa_rate`'s no-stim trials); trials outside every block's span
    (or before the first / after the last block) are dropped, since they
    have no defined perf_state. `licked_only=True` restricts the returned
    decode trials to `lick_flag==1` (for a lick-time-aligned decode, which
    needs a real `lick_time` to anchor to) and adds `rt`/`half`-free `rt`
    column, mirroring `prep_lick_aligned_trials`."""
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    all_trials = prepped["trials"]
    whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
    if len(whisker) < block_size:
        return None

    n_full_blocks = len(whisker) // block_size
    whisker = whisker.iloc[: n_full_blocks * block_size].copy()
    whisker["block_id"] = np.arange(len(whisker)) // block_size

    block_hit_rate = whisker.groupby("block_id")["rewarded"].mean()
    median_hr = block_hit_rate.median()
    block_state = block_hit_rate.apply(lambda hr: "high" if hr >= median_hr else "low")
    block_bounds = whisker.groupby("block_id")["start_time"].agg(["min", "max"])

    no_stim = all_trials[all_trials["trial_type"] == "no_stim_trial"]
    block_fa_rate = {}
    for block_id, row in block_bounds.iterrows():
        in_span = no_stim[(no_stim["start_time"] >= row["min"]) & (no_stim["start_time"] <= row["max"])]
        block_fa_rate[block_id] = float(in_span["lick_flag"].mean()) if len(in_span) else float("nan")

    decode_trials = all_trials[all_trials["trial_type"].isin(decode_trial_types)].copy()
    if licked_only:
        decode_trials = decode_trials[(decode_trials["lick_flag"] == 1) & decode_trials["lick_time"].notna()].copy()
        decode_trials = add_first_lick_time(decode_trials)  # corrected first lick, 2026-09-27
    if len(decode_trials) == 0:
        return None

    # Assign each decode trial to the whisker block whose time span contains it.
    block_id_for_trial = pd.array([pd.NA] * len(decode_trials), dtype="Int64")
    starts = decode_trials["start_time"].to_numpy()
    for block_id, row in block_bounds.iterrows():
        in_span = (starts >= row["min"]) & (starts <= row["max"])
        block_id_for_trial[in_span] = block_id
    decode_trials["block_id"] = block_id_for_trial
    decode_trials = decode_trials[decode_trials["block_id"].notna()].reset_index(drop=True)
    if len(decode_trials) == 0:
        return None
    decode_trials["block_id"] = decode_trials["block_id"].astype(int)

    decode_trials["perf_state"] = decode_trials["block_id"].map(block_state)
    decode_trials["block_hit_rate"] = decode_trials["block_id"].map(block_hit_rate)
    decode_trials["block_fa_rate"] = decode_trials["block_id"].map(block_fa_rate)
    if licked_only:
        decode_trials["rt"] = decode_trials["reaction_time"]  # from stimulus onset (corrected, 2026-09-27)

    decode_trials["reward_group"] = prepped["reward_group"]
    decode_trials["subject_id"] = prepped["subject_id"]
    return decode_trials


def _active_trials_from_whisker_onset_for_curve(session_id: str, trials_tbl: pd.DataFrame) -> pd.DataFrame:
    """Context filter + `perf != 6` ("association" outcome) filter +
    drop-before-first-whisker-trial -- matches `cd_analysis/utils/
    performance.py`'s `keep_active_from_whisker_onset`. Deliberately NOT
    `ssl_bwm_trial_prep.prep_session`: that function additionally drops the
    first post-warmup trial for its own unrelated t-1-outcome feature,
    which breaks positional alignment with the learning-curve file's own
    trial indexing by exactly one (found and fixed 2026-09-15/16 -- see
    `043_perfstate_curve_labels_test.py`'s docstring for the investigation
    that found this)."""
    trials = trials_tbl[trials_tbl["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    has_context = trials["context"].notna() & (trials["context"] != "nan")
    if has_context.any():
        trials = trials[trials["context"] == "active"]
    if "perf" in trials.columns:
        trials = trials[trials["perf"] != 6]
    trials = trials.reset_index(drop=True)
    whisker_idx = trials.index[trials["trial_type"] == "whisker_trial"]
    if len(whisker_idx) > 0:
        trials = trials.loc[whisker_idx[0]:].reset_index(drop=True)
    return trials


def _active_trials_for_curve_untrimmed(session_id: str, trials_tbl: pd.DataFrame) -> pd.DataFrame:
    """Same context + `perf != 6` filter as
    `_active_trials_from_whisker_onset_for_curve`, WITHOUT the
    drop-before-first-whisker-trial step. Needed for the no-stim/
    false-alarm curve (`prep_perfquant_curve_targets`): its learning-curve
    file indexes every no-stim trial in the session, including the ones
    before the first whisker trial (the "auditory warm-up" block) --
    found 2026-09-19 via a systematic ~11-16-trial count mismatch between
    this project's warmup-trimmed no-stim subset and the curve file's own
    count, which turned out to match exactly the number of no-stim trials
    occurring before the first whisker trial. Using the untrimmed set here
    doesn't reintroduce warm-up trials into any DECODED trial set (this
    curve is only ever used as a smooth interpolation basis, never decoded
    itself) -- the existing "ignore auditory warm-up" convention still
    applies to every actual decode target."""
    trials = trials_tbl[trials_tbl["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    has_context = trials["context"].notna() & (trials["context"] != "nan")
    if has_context.any():
        trials = trials[trials["context"] == "active"]
    if "perf" in trials.columns:
        trials = trials[trials["perf"] != 6]
    return trials.reset_index(drop=True)


def _assign_expertise_blocks_positional(p_low: np.ndarray, p_chance: np.ndarray, reward_group_int: int,
                                         n_consecutive: int = 5) -> np.ndarray:
    """Port of `cd_analysis/utils/performance.py`'s `assign_expertise_blocks`
    (cohort-direction-aware criterion, >=5-consecutive-trial run
    requirement), applied to one session's positionally-aligned
    `p_low`/`p_chance` arrays. Ported rather than imported -- see
    `prep_perfstate_trials_curve`'s docstring for why."""
    if reward_group_int == 1:
        criterion = p_low > p_chance
    elif reward_group_int == 0:
        criterion = p_low < p_chance
    else:
        raise ValueError(f"unexpected reward_group_int {reward_group_int!r}")
    high_mask = np.zeros(len(criterion), dtype=bool)
    start_idx = 0
    while start_idx < len(criterion):
        if criterion[start_idx]:
            end_idx = start_idx
            while end_idx < len(criterion) and criterion[end_idx]:
                end_idx += 1
            if end_idx - start_idx >= n_consecutive:
                high_mask[start_idx:end_idx] = True
            start_idx = end_idx
        else:
            start_idx += 1
    return high_mask


def load_whisker_curve_row(curve_root: Path, subject_id: str, day: int, trial_type: str = "whisker_trial",
                            interp: bool = True) -> pd.Series | None:
    """Direct H5 read of Axel Bisi's per-session learning-curve file,
    generalized to any `whisker_N` day index (not just day 0) -- user
    finding 2026-09-19: `combined_results_ks4/<mouse>/whisker_N/
    learning_curve/` now exists with real curve data for `N>0` for
    AB-cohort mice (though not yet for the MH-cohort ephys mice this
    project's neural decoding actually runs on -- this function will pick
    that up automatically whenever it lands, no further code change
    needed). Reimplements (rather than calls)
    `ephys_utilities.helpers.load_helpers.load_learning_curves_data`,
    which hardcodes `whisker_0` in its filename template -- same file
    format, just day-parameterized, and drops the `ephys_utilities`
    sys.path/import dependency for this one read. Returns None if the
    file doesn't exist for this (mouse, day) (the normal, expected case
    for most MH-cohort mice, all expert-stage MH sessions today, and any
    mouse without a curve at all)."""
    suffix = "_interp" if interp else ""
    file_name = f"{subject_id}_whisker_{day}_{trial_type}_learning_curve{suffix}.h5"
    path_to_file = curve_root / subject_id / f"whisker_{day}" / "learning_curve" / file_name
    try:
        df = pd.read_hdf(path_to_file)
    except (FileNotFoundError, OSError):
        return None
    if len(df) == 0:
        return None
    return df.iloc[0]


def prep_perfstate_trials_curve(
    dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame,
    decode_trial_types: list[str], licked_only: bool = False,
) -> pd.DataFrame | None:
    """Curve-based perf-state definition (user decision 2026-09-16: adopt
    as the pipeline default going forward, replacing
    `prep_perfstate_trials_generic`'s fixed-block/session-median-split
    criterion for both the `perfstate` condition-split and the `perfstate`
    decode target itself). Uses Axel Bisi's own learning-curve model
    (`load_whisker_curve_row`'s per-whisker-trial `p_low`/`p_chance`,
    read directly rather than via `ephys_utilities.helpers.load_helpers.
    load_learning_curves_data`, which hardcodes day 0 -- see
    `load_whisker_curve_row`) instead of a purely session-relative
    hit-rate split: a whisker trial is `'high'` iff part
    of a run of >=5 consecutive trials where `p_low` is on the rewarded
    side of `p_chance` (cohort-direction-aware), `'low'` otherwise;
    non-whisker decode trials inherit the nearest whisker trial's label by
    `start_time` (`pd.merge_asof(..., direction='nearest')`). Ported (not
    imported) from `cd_analysis/utils/performance.py`'s
    `assign_expertise_blocks`/`propagate_expertise_inplace` --
    `cd_analysis.utils.performance` pulls in its own `utils.settings_haas`
    at import time, a dependency/environment risk this repo doesn't need
    for ~50 lines of portable logic (validated against real data
    2026-09-15/16: `043_perfstate_curve_labels_test.py`,
    `044_perfstate_curve_diagnostic.py`, `045_perfstate_curve_label_sweep.py`
    in the project's `exploratory-analyses/`).

    **Day-aware, generalized 2026-09-19** (was learning-stage-only /
    hardcoded to `whisker_0` until this date): the learning-curve H5 file
    lives at `<mouse>/whisker_<day>/learning_curve/<mouse>_whisker_<day>_
    whisker_trial_learning_curve_interp.h5`, `day` parsed from this
    session's own `session_description` via `session_training_day`
    (`day==0` -> the learning-stage file, `day>0` -> that expert session's
    own file, if it exists). As of 2026-09-19 no MH-cohort (ephys) mouse
    has a `day>0` curve file yet -- only `day==0` resolves for this
    project's actual decoding cohort today -- but AB-cohort mice do have
    `day>0` curve files now, so this now picks up expert-stage curves
    automatically for whichever mice/days actually have them, with no
    further code change needed when MH-cohort expert curves land. Returns
    None (same as every other "no usable trials" case in this module) for
    any session without a matching curve file (14/89 mice as of
    2026-09-16, mostly MH-prefix, at day 0) or a mismatched trial count,
    rather than silently falling back to the block-median definition --
    callers get a clean skip, not a mixed-definition result.

    Pre-whisker-onset "auditory warm-up" trials are never seen by this
    function at all (already dropped by the trimming below, same as this
    project's existing warmup-block fix) -- so unlike `cd_analysis`'s own
    `keep_active_from_whisker_onset` (which labels that prefix
    `'aud_block'`), there is no such category here: this pipeline ignores
    those trials for every decode target, modality included (user decision
    2026-09-16: "Ignore auditory blocks for modality decoding")."""
    subject_id = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "subject_id"].iloc[0]
    reward_group = load_reward_group(subject_id)
    if reward_group is None:
        return None
    reward_group_int = 1 if reward_group == "R+" else 0

    all_trials = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
    if len(whisker) == 0:
        return None

    curve_root = axel_bisi_path("combined_results_ks4")
    if curve_root is None:
        return None
    session_description = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "session_description"].iloc[0]
    day = session_training_day(session_description)
    if day is None:
        return None
    row = load_whisker_curve_row(curve_root, subject_id, day)
    if row is None:
        return None
    p_low, p_chance = np.asarray(row["p_low"]), np.asarray(row["p_chance"])
    if len(p_low) != len(whisker):
        return None

    high_mask = _assign_expertise_blocks_positional(p_low, p_chance, reward_group_int)
    whisker = whisker.copy()
    whisker["block_perf_type"] = np.where(high_mask, "high", "low")

    decode_trials = all_trials[all_trials["trial_type"].isin(decode_trial_types)].copy()
    if licked_only:
        decode_trials = decode_trials[(decode_trials["lick_flag"] == 1) & decode_trials["lick_time"].notna()].copy()
        decode_trials = add_first_lick_time(decode_trials)  # corrected first lick, 2026-09-27
    if len(decode_trials) == 0:
        return None
    decode_trials = decode_trials.sort_values("start_time").reset_index(drop=True)

    if set(decode_trial_types) == {"whisker_trial"}:
        decode_trials["perf_state"] = whisker.sort_values("start_time")["block_perf_type"].to_numpy()
    else:
        whisker_sorted = whisker.sort_values("start_time")[["start_time", "block_perf_type"]]
        decode_trials = pd.merge_asof(decode_trials, whisker_sorted, on="start_time", direction="nearest")
        decode_trials["perf_state"] = decode_trials["block_perf_type"]

    if licked_only:
        decode_trials["rt"] = decode_trials["reaction_time"]  # from stimulus onset (corrected, 2026-09-27)
    decode_trials["reward_group"] = reward_group
    decode_trials["subject_id"] = subject_id
    return decode_trials


def prep_perfquant_curve_targets(
    dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame,
) -> pd.DataFrame | None:
    """Per-whisker-trial continuous behavioral targets for the curve-value
    regression -> quantile-class ('perfquant') decoding target (user
    request 2026-09-19, added ALONGSIDE the existing 2-class `perfstate`
    target, not replacing it): `whisker_curve` (this trial's whisker-trial
    learning-curve `p_mean` -- same file/positional alignment
    `prep_perfstate_trials_curve` uses for its `p_low`-based discrete
    label, but the central estimate rather than the lower CI bound, since
    this is a regression target, not a threshold criterion),
    `falsealarm_curve` (the no-stim/catch-trial curve's `p_mean`, linearly
    interpolated onto this whisker trial's `start_time` -- there is no
    positional correspondence between whisker and no-stim trials the way
    there is within one trial type, unlike the same-trial-type alignment
    `whisker_curve` gets for free), and `performance_curve`
    (`whisker_curve - falsealarm_curve`).

    Returns None under the same conditions `prep_perfstate_trials_curve`
    does (no curve file for this (mouse, day) -- see `load_whisker_curve_row`
    -- or a whisker trial-count mismatch). A missing/empty/mismatched
    no-stim curve degrades gracefully to all-NaN `falsealarm_curve`/
    `performance_curve` rather than a hard failure -- a session can
    genuinely have zero no-stim trials, and `whisker_curve` alone is still
    usable as its own target."""
    subject_id = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "subject_id"].iloc[0]
    session_description = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "session_description"].iloc[0]
    day = session_training_day(session_description)
    if day is None:
        return None
    curve_root = axel_bisi_path("combined_results_ks4")
    if curve_root is None:
        return None
    reward_group = load_reward_group(subject_id)
    if reward_group is None:
        return None

    all_trials = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].sort_values("start_time").reset_index(drop=True)
    if len(whisker) == 0:
        return None

    wh_row = load_whisker_curve_row(curve_root, subject_id, day, trial_type="whisker_trial", interp=True)
    if wh_row is None:
        return None
    whisker_curve_full = np.asarray(wh_row["p_mean"])
    if len(whisker_curve_full) != len(whisker):
        return None

    out = pd.DataFrame({"start_time": whisker["start_time"].to_numpy()})
    out["session_id"] = session_id
    out["subject_id"] = subject_id
    out["reward_group"] = reward_group
    out["whisker_curve"] = whisker_curve_full

    fa_row = load_whisker_curve_row(curve_root, subject_id, day, trial_type="no_stim_trial", interp=False)
    no_stim_basis = _active_trials_for_curve_untrimmed(session_id, trials_tbl)
    no_stim = no_stim_basis[no_stim_basis["trial_type"] == "no_stim_trial"].sort_values("start_time").reset_index(drop=True)
    if fa_row is not None and len(no_stim) > 0 and len(np.asarray(fa_row["p_mean"])) == len(no_stim):
        fa_curve_full = np.asarray(fa_row["p_mean"])
        out["falsealarm_curve"] = np.interp(
            whisker["start_time"].to_numpy(), no_stim["start_time"].to_numpy(), fa_curve_full,
        )
    else:
        out["falsealarm_curve"] = np.nan
    out["performance_curve"] = out["whisker_curve"] - out["falsealarm_curve"]
    return out


def reconstruct_learning_trial(curve: pd.Series) -> tuple[str, float, str]:
    """Port of `behaviour_analysis/learning_utils.py`'s
    `identify_learning_trial_rewarded` (R+) /
    `identify_learning_trial_nonrewarded(..., interp_flag=True)` (R-), with
    the params hardcoded in `beh_plotting_functions.py`
    (n_trials_for_expert=20, min/max_perf_for_expert=0.8/0.2, n_consec=5),
    applied to one learning-curve H5 row (`load_whisker_curve_row`).
    Reproduces the stored `learning_trial` in 89/89 learning-stage
    sessions (validated 2026-09-24, `089_learning_trial_sanity.py`); used
    here to label HOW each value was reached, since the stored value alone
    hides two hardcoded defaults. Returns (mouse_cat, learning_trial,
    lt_source), lt_source one of:
      'learner' / 'expert'          -- criterion met at trial > 10
      'learner_floor10' / 'expert_floor10' -- criterion met at <= 10, clamped to 10
      'learner_fallback10'          -- R- only: no criterion met, hardcoded 10
      'non-learner'                 -- NaN
    `learning_trial` is a 0-based index into the curve-aligned whisker
    trials (`_active_trials_from_whisker_onset_for_curve`). R+: start of the
    first 5-run of p_low>p_chance. R-: LAST trial of the first 5-run of
    p_low>p_chance that is followed by 5 not-above-chance trials. Only
    trials >= the first whisker hit are considered."""
    pm, pl, ph, pc = (np.asarray(curve[k]) for k in ("p_mean", "p_low", "p_high", "p_chance"))
    o = np.asarray(curve["outcomes"])
    first_hit = int(np.argmax(o == 1)) if (o == 1).any() else 0
    above = np.where(pl > pc)[0]
    below = np.where(ph < pc)[0]
    above, below = above[above >= first_hit], below[below >= first_hit]

    def floored(cat, t):
        return (cat, 10, f"{cat}_floor10") if t <= 10 else (cat, int(t), cat)

    def consec(w):
        return np.all(np.diff(w) == 1)

    if curve["reward_group"] == 1:
        if len(above) == 0:
            return "non-learner", np.nan, "non-learner"
        for i in range(len(above) - 20 + 1):
            w = above[i:i + 20]
            if max(w) > 20:
                break
            if consec(w) and np.mean(pm[w]) >= 0.8:
                return floored("expert", w[0])
        for i in range(len(above) - 5 + 1):
            w = above[i:i + 5]
            if consec(w):
                return floored("learner", w[0])
        return "non-learner", np.nan, "non-learner"
    if len(below) == 0 and len(above) > 0:
        return "non-learner", np.nan, "non-learner"
    for i in range(len(below) - 20 + 1):
        w = below[i:i + 20]
        if consec(w) and np.mean(pm[w]) <= 0.2:
            return floored("expert", w[0])
    above_u = np.sort(np.unique(above))
    above_set = set(above_u)
    for i in range(len(above_u) - 5 + 1):
        w = above_u[i:i + 5]
        if consec(w):
            nxt = np.arange(w[-1] + 1, w[-1] + 6)
            if nxt[-1] <= len(pl) - 1 and all(t not in above_set for t in nxt):
                return floored("learner", w[-1])
    return "learner", 10, "learner_fallback10"


def detect_terminal_disengagement(session_id: str, trials_tbl: pd.DataFrame, min_whisker: int = 5,
                                   min_auditory: int = 1) -> dict:
    """End-of-session disengagement (user request 2026-09-24: "drop last run
    of misses at the end if it is clearly a stated state"). Default = rule A1
    (user 2026-09-30, chosen after comparing rules in ssl-learning-trial-
    identification 026): min_auditory 1 (was 3 = rule A; results computed
    before 2026-10-01 used 3 -- pass min_auditory=3 to reproduce). Uses the same
    active-trial set as the learning curve (`_active_trials_from_whisker_
    onset_for_curve`, all trial types). The trailing block = every trial
    after the session's LAST lick on any trial type (whisker, auditory or
    no-stim). It counts as a disengaged state only if it contains >=
    `min_whisker` whisker trials AND >= `min_auditory` auditory trials --
    auditory trials are rewarded on lick in both cohorts and normally hit at
    a high rate, so a run of unlicked auditory trials is the behavioral
    signature of disengagement rather than of a whisker-specific miss run.
    Returns dict(disengaged, t_cut, n_whisker_dropped, n_auditory_in_run)
    -- callers drop trials with start_time >= t_cut when `disengaged`."""
    all_t = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    out = dict(disengaged=False, t_cut=np.nan, n_whisker_dropped=0, n_auditory_in_run=0)
    licked = np.where(all_t["lick_flag"].to_numpy() == 1)[0]
    if len(licked) == 0:
        return out
    trail = all_t.iloc[licked[-1] + 1:]
    n_wh = int((trail["trial_type"] == "whisker_trial").sum())
    n_aud = int((trail["trial_type"] == "auditory_trial").sum())
    out.update(n_whisker_dropped=n_wh, n_auditory_in_run=n_aud)
    if n_wh >= min_whisker and n_aud >= min_auditory:
        out.update(disengaged=True, t_cut=float(trail["start_time"].iloc[0]))
    else:
        out["n_whisker_dropped"] = 0
    return out


def prep_hitmiss_trials_learning_split(
    dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict]:
    """`prep_hitmiss_trials` plus an `lt_epoch` column ('pre'/'post') split
    at this session's stored `learning_trial` (user request 2026-09-24,
    replacing the session-half split). The learning trial's `start_time` is
    looked up in the curve-aligned whisker trials and the DECODED trial set
    is split by start_time (not position -- `prep_session` drops one early
    trial, so positions are off by one); the learning trial itself is
    'post'. Returns (trials or None, info dict with learning_trial,
    lt_source, mouse_cat, t_learn, skip reason)."""
    info = dict(learning_trial=np.nan, lt_source=None, mouse_cat=None, t_learn=np.nan, lt_skip_reason=None)
    subject_id = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "subject_id"].iloc[0]
    desc = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "session_description"].iloc[0]
    day = session_training_day(desc)
    curve_root = axel_bisi_path("combined_results_ks4")
    curve = load_whisker_curve_row(curve_root, subject_id, day) if (day is not None and curve_root is not None) else None
    if curve is None:
        info["lt_skip_reason"] = "no learning-curve file"
        return None, info
    mouse_cat, lt_repro, lt_source = reconstruct_learning_trial(curve)
    lt = curve["learning_trial"]
    info.update(mouse_cat=curve["mouse_cat"], lt_source=lt_source)
    if pd.isna(lt):
        info["lt_skip_reason"] = "no learning_trial (non-learner)"
        return None, info
    lt = int(lt)
    info["learning_trial"] = lt
    if lt != lt_repro:
        info["lt_skip_reason"] = f"stored learning_trial {lt} != reconstructed {lt_repro}"
        return None, info
    cw = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    cw = cw[cw["trial_type"] == "whisker_trial"].reset_index(drop=True)
    if len(cw) != len(curve["outcomes"]) or lt >= len(cw):
        info["lt_skip_reason"] = f"curve/trial length mismatch ({len(curve['outcomes'])} vs {len(cw)})"
        return None, info
    trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
    if trials is None:
        info["lt_skip_reason"] = "no usable hit/miss trials"
        return None, info
    t_learn = float(cw["start_time"].iloc[lt])
    info["t_learn"] = t_learn
    trials["lt_epoch"] = np.where(trials["start_time"].to_numpy() >= t_learn, "post", "pre")
    return trials, info


def prep_lick_aligned_trials(dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> pd.DataFrame | None:
    """Active-epoch whisker + auditory trials for `session_id` restricted to
    `lick_flag==1` (the only trials with a real `lick_time` to anchor an
    event-aligned window to -- see `question.md`'s lick_time-aligned design
    note, 2026-09-10). Adds `reaction_time`/`first_lick_time` (corrected first lick,
    `add_first_lick_time`, 2026-09-27 -- align to `first_lick_time`, never the stored
    `lick_time`), `rt` (= `reaction_time`, from stimulus onset) and a `half`
    column (chronological median split, same convention as
    `prep_hitmiss_trials` but over this trial set, not whisker-only).
    `trial_type` is the decode target for this analysis (whisker vs
    auditory), not hit/miss."""
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    trials = prepped["trials"]
    trials = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"]) & (trials["lick_flag"] == 1)
                    & trials["lick_time"].notna()]
    trials = trials.reset_index(drop=True)
    if len(trials) == 0:
        return None
    trials = add_first_lick_time(trials)  # corrected first lick (skills/ssl-lick-alignment), 2026-09-27
    trials["rt"] = trials["reaction_time"]
    median_idx = len(trials) // 2
    trials["half"] = ["first"] * median_idx + ["second"] * (len(trials) - median_idx)
    trials["reward_group"] = prepped["reward_group"]
    trials["subject_id"] = prepped["subject_id"]
    return trials


# ---------------------------------------------------------------------------
# Area population selection
# ---------------------------------------------------------------------------

def area_units(session_id: str, area_col: str, area_value: str, area_labels: pd.DataFrame) -> np.ndarray:
    """Unit `cluster_id`s for one (session, area) cell, QC-filtered to
    `quality_label in {'good','mua'}` (switched from raw `bc_label` 2026-09-11
    per `skills/ssl-valid-data/SKILL.md` -- `quality_label`, from
    `unit_metrics_utils.classify_units_quality`, is a more complete
    good/mua split than bombcell's own `bc_label` alone. Note: since this
    filter pools good+mua either way, the pooled total is identical under
    both labels -- `quality_label` only reclassifies *within* the
    good/mua split, it does not change which units are `non-soma`-excluded
    -- the switch is for QC correctness, not for more units passing this
    specific pooled filter)."""
    sub = area_labels[
        (area_labels["session_id"] == session_id)
        & (area_labels[area_col] == area_value)
        & (area_labels["quality_label"].isin(QC_VALUES))
    ]
    return sub["cluster_id"].to_numpy()


def areas_with_enough_units(session_id: str, area_col: str, area_labels: pd.DataFrame, min_units: int = MIN_UNITS_PER_AREA) -> list[str]:
    sub = area_labels[(area_labels["session_id"] == session_id) & (area_labels["quality_label"].isin(QC_VALUES))]
    counts = sub.groupby(area_col)["cluster_id"].nunique()
    return counts[counts >= min_units].index.tolist()


# ---------------------------------------------------------------------------
# Dead-zone-offset 10ms bin grid
# ---------------------------------------------------------------------------

def dead_zone_offset_bin_edges(window: tuple[float, float], bin_width: float = BIN_WIDTH_S) -> list[tuple[float, float]]:
    """10ms disjoint bin edges relative to `start_time`, offset by half a
    bin width so the mandatory whisker dead zone (`-1ms/+4ms`) falls
    **entirely inside one bin** (`[-5ms, +5ms)` at the default 10ms width),
    which is then dropped -- avoids clipping/contaminating any bin with a
    partial real+artifact mix (user's explicit 2026-09-10 choice).
    """
    half = bin_width / 2.0
    start, stop = window
    # Grid aligned so a bin edge sits at `half` (i.e. one bin spans
    # [-half, +half), containing the whole dead zone since
    # DEAD_ZONE_START_S/STOP_S are within +-half for bin_width>=0.005).
    if not (DEAD_ZONE_START_S >= -half and DEAD_ZONE_STOP_S <= half):
        raise ValueError(
            f"bin_width {bin_width}s too small to contain the dead zone "
            f"[{DEAD_ZONE_START_S}, {DEAD_ZONE_STOP_S}] in one bin"
        )
    # Build edges directly: ..., -half-bin_width, -half, half, half+bin_width, ...
    n_pre = int(np.ceil((abs(start) - half) / bin_width)) if start < -half else 0
    n_post = int(np.ceil((stop - half) / bin_width)) if stop > half else 0
    edges = [-half - i * bin_width for i in range(n_pre, -1, -1)] + [half + i * bin_width for i in range(0, n_post + 1)]
    edges = sorted(set(round(e, 10) for e in edges))
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bins = [b for b in bins if b[0] >= start - 1e-9 and b[1] <= stop + 1e-9]
    # Drop the bin fully containing the dead zone.
    bins = [b for b in bins if not (b[0] <= DEAD_ZONE_START_S + 1e-9 and b[1] >= DEAD_ZONE_STOP_S - 1e-9)]
    return bins


# ---------------------------------------------------------------------------
# Per-bin population design matrices
# ---------------------------------------------------------------------------

def load_session_unit_spikes(dataset_root: Path, session_id: str) -> dict:
    """Load one session's spike shard once and index every unit's sorted
    spike times by `cluster_id`. Reuse the returned dict across every
    area/half cell for this session -- `bin_population_matrices` used to
    reload the shard from disk per (area, half) call (verified 2026-09-10
    smoke test: this was a real, avoidable cost, up to ~13s per repeat
    call within the same session)."""
    shard = load_spike_shard(dataset_root / "spikes" / session_id)
    spike_times_all = shard["spike_times_seconds"]
    spike_clusters_dense = shard["spike_clusters"]
    cluster_ids = shard["cluster_ids"]
    out = {}
    for cid in np.unique(cluster_ids):
        dense_idx = np.where(cluster_ids == cid)[0]
        out[cid] = np.sort(spike_times_all[spike_clusters_dense == dense_idx[0]]) if len(dense_idx) else np.array([])
    return out


def bin_population_matrices(
    unit_spikes_by_cluster: dict,
    unit_cluster_ids: np.ndarray,
    start_time: np.ndarray,
    is_whisker: np.ndarray,
    bin_edges: list[tuple[float, float]],
) -> list[np.ndarray]:
    """One trials-x-units firing-rate matrix per bin, from a
    `load_session_unit_spikes` dict already loaded for this session (pass
    the same dict across every area/half cell of one session)."""
    matrices = []
    for window in bin_edges:
        X = np.full((len(start_time), len(unit_cluster_ids)), np.nan)
        for j, cid in enumerate(unit_cluster_ids):
            spikes = unit_spikes_by_cluster.get(cid, np.array([]))
            X[:, j] = unit_rates_for_trials(spikes, start_time, is_whisker, window)
        matrices.append(X)
    return matrices


# ---------------------------------------------------------------------------
# Sliding (overlapping) start_time-aligned windows -- two-piece dead-zone
# excision instead of dropping a whole bin (user decision 2026-09-10: wider
# window than 10ms for richer per-bin features, "dealing with the artifact"
# rather than avoiding it via disjoint narrow bins).
# ---------------------------------------------------------------------------

def sliding_bin_edges(window: tuple[float, float], bin_width: float = 0.05, stride: float = 0.01) -> list[tuple[float, float]]:
    """Overlapping `bin_width`-wide windows relative to `start_time`,
    stepped every `stride`. Unlike `dead_zone_offset_bin_edges`, no bin is
    dropped -- every window (including ones straddling the dead zone) is
    handled by `sliding_window_rates_for_trials`'s two-piece excision."""
    start, stop = window
    starts = np.arange(start, stop - bin_width + 1e-9, stride)
    return [(round(float(s), 10), round(float(s) + bin_width, 10)) for s in starts]


def causal_bin_edges(window: tuple[float, float], bin_width: float = 0.05, stride: float = 0.005) -> list[tuple[float, float]]:
    """Backward-looking (causal) bin edges for the full-rebuild spec (user
    decision 2026-09-11): each bin covers `(t - bin_width, t]` and is
    labeled at its **end** `t`, so a bin decoding "at time t" never uses
    spikes after `t` (unlike `sliding_bin_edges`, whose bins are labeled at
    their center and so include `bin_width/2` of look-ahead). `window`
    gives the range of label times `t`, not of the underlying data (which
    extends `bin_width` before `window[0]`). The two-piece dead-zone
    excision in `sliding_window_rates_for_trials`/`event_aligned_rates_for_trials`
    is agnostic to this labeling convention -- only the edge generator and
    the plotting x-axis (bin end, not center) need to change."""
    start, stop = window
    labels = np.arange(start, stop + 1e-9, stride)
    return [(round(float(t - bin_width), 10), round(float(t), 10)) for t in labels]


def add_whole_brain_column(area_labels: pd.DataFrame, column: str = "whole_brain", value: str = "All units") -> pd.DataFrame:
    """Third area scheme (user request 2026-09-11): every QC-passing unit in
    a session pooled into a single pseudo-area, so the same `area_units`/
    `areas_with_enough_units` machinery (generic over any column name) works
    unchanged -- just pass `area_col='whole_brain'`."""
    df = area_labels.copy()
    df[column] = value
    return df


def sliding_window_rates_for_trials(
    spike_times_sorted: np.ndarray,
    trial_start_times: np.ndarray,
    trial_is_whisker: np.ndarray,
    window: tuple[float, float],
    dead_zone: tuple[float, float] | None = None,
) -> np.ndarray:
    """Per-trial firing rate (Hz), dead-zone-aware, for a window that may
    fully straddle the mandatory whisker dead zone (`clip_window_for_whisker`
    refuses this case -- designed for narrow bins that never straddle both
    sides at once). Here the dead zone is **excised** from the window: spikes
    are counted in whichever of the pre-dead-zone and post-dead-zone pieces
    the window actually has, and the rate denominator is the window's
    duration minus its overlap with the dead zone (not the raw window
    width) -- correct for a window entirely outside, one-sided against, or
    straddling both sides of the dead zone alike. NaN if the window falls
    entirely inside the dead zone (no valid duration left)."""
    w0, w1 = window
    n = len(trial_start_times)
    rates = np.full(n, np.nan)
    dz0, dz1 = dead_zone if dead_zone is not None else (DEAD_ZONE_START_S, DEAD_ZONE_STOP_S)

    for is_wh in (True, False):
        mask = trial_is_whisker == is_wh
        if not mask.any():
            continue
        starts = trial_start_times[mask]

        if not is_wh or w1 <= dz0 or w0 >= dz1:
            lo = np.searchsorted(spike_times_sorted, starts + w0, side="left")
            hi = np.searchsorted(spike_times_sorted, starts + w1, side="left")
            rates[mask] = (hi - lo).astype(np.float64) / (w1 - w0)
            continue

        overlap = min(w1, dz1) - max(w0, dz0)
        valid_duration = (w1 - w0) - overlap
        if valid_duration <= 0:
            continue  # stays NaN: window fully inside the dead zone

        counts = np.zeros(len(starts))
        if w0 < dz0:
            lo = np.searchsorted(spike_times_sorted, starts + w0, side="left")
            hi = np.searchsorted(spike_times_sorted, starts + min(w1, dz0), side="left")
            counts += (hi - lo)
        if w1 > dz1:
            lo = np.searchsorted(spike_times_sorted, starts + max(w0, dz1), side="left")
            hi = np.searchsorted(spike_times_sorted, starts + w1, side="left")
            counts += (hi - lo)
        rates[mask] = counts / valid_duration

    return rates


def sliding_bin_population_matrices(
    unit_spikes_by_cluster: dict,
    unit_cluster_ids: np.ndarray,
    start_time: np.ndarray,
    is_whisker: np.ndarray,
    bin_edges: list[tuple[float, float]],
    dead_zone: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """`bin_population_matrices` analog using `sliding_window_rates_for_trials`
    (two-piece dead-zone excision) instead of the disjoint-grid-with-a-
    dropped-bin approach. `dead_zone` overrides the module default (e.g. the
    wider `-10ms/+5ms` window per the 2026-09-11 full-rebuild spec)."""
    matrices = []
    for window in bin_edges:
        X = np.full((len(start_time), len(unit_cluster_ids)), np.nan)
        for j, cid in enumerate(unit_cluster_ids):
            spikes = unit_spikes_by_cluster.get(cid, np.array([]))
            X[:, j] = sliding_window_rates_for_trials(spikes, start_time, is_whisker, window, dead_zone=dead_zone)
        matrices.append(X)
    return matrices


# ---------------------------------------------------------------------------
# Event-aligned (e.g. lick_time) windows -- per-trial dead-zone masking
# ---------------------------------------------------------------------------

def lick_aligned_bin_edges(window: tuple[float, float], bin_width: float = BIN_WIDTH_S) -> list[tuple[float, float]]:
    """Plain disjoint bin edges relative to an event time (e.g.
    `lick_time`), no dead-zone offset trick needed -- unlike
    `dead_zone_offset_bin_edges`, the artifact here is not at a fixed
    position in this alignment (it depends on each trial's own RT, since
    the artifact is fixed relative to `start_time`, not the event), so
    correctness is handled per-trial by `event_aligned_rates_for_trials`
    instead of by dropping one global bin."""
    start, stop = window
    n_bins = int(round((stop - start) / bin_width))
    edges = [round(start + i * bin_width, 10) for i in range(n_bins + 1)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def event_aligned_rates_for_trials(
    spike_times_sorted: np.ndarray,
    event_time: np.ndarray,
    trial_start_time: np.ndarray,
    trial_is_whisker: np.ndarray,
    window: tuple[float, float],
    dead_zone: tuple[float, float] | None = None,
) -> np.ndarray:
    """Per-trial firing rate (Hz) for one unit, in `window` (seconds
    relative to `event_time`, e.g. `lick_time` -- NOT `start_time`).
    Unlike `unit_rates_for_trials`, the mandatory whisker dead-zone
    exclusion cannot be applied by clipping a single shared window,
    because the artifact's position within an event-aligned window shifts
    per trial (it is fixed relative to `trial_start_time`, not
    `event_time`, so it moves with each trial's own reaction time). A
    **Generalized 2026-09-10** (was blanket-NaN on any overlap -- correct
    but wasteful once bins got wider than 10ms, since a 50ms window
    overlaps some trial's dead zone far more often): now excises just the
    dead-zone portion from each trial's own valid duration (two-piece:
    pre-dead-zone + post-dead-zone spike counts, divided by the reduced
    duration), fully vectorized (every trial's overlap is an independent
    elementwise min/max). Only a trial whose entire window falls inside its
    own dead zone still gets NaN. Non-whisker trials are never masked."""
    w0, w1 = window
    win_start = event_time + w0
    win_stop = event_time + w1

    lo = np.searchsorted(spike_times_sorted, win_start, side="left")
    hi = np.searchsorted(spike_times_sorted, win_stop, side="left")
    rates = (hi - lo).astype(np.float64) / (w1 - w0)

    dz0, dz1 = dead_zone if dead_zone is not None else (DEAD_ZONE_START_S, DEAD_ZONE_STOP_S)
    dead_start = trial_start_time + dz0
    dead_stop = trial_start_time + dz1
    overlap_start = np.maximum(win_start, dead_start)
    overlap_stop = np.minimum(win_stop, dead_stop)
    overlap = np.clip(overlap_stop - overlap_start, 0, None)
    affected = trial_is_whisker & (overlap > 0)
    if not affected.any():
        return rates

    valid_duration = (w1 - w0) - overlap[affected]
    pre_stop = np.minimum(win_stop[affected], dead_start[affected])
    lo_pre = np.searchsorted(spike_times_sorted, win_start[affected], side="left")
    hi_pre = np.searchsorted(spike_times_sorted, pre_stop, side="left")
    counts = np.where(win_start[affected] < dead_start[affected], (hi_pre - lo_pre).astype(np.float64), 0.0)

    post_start = np.maximum(win_start[affected], dead_stop[affected])
    lo_post = np.searchsorted(spike_times_sorted, post_start, side="left")
    hi_post = np.searchsorted(spike_times_sorted, win_stop[affected], side="left")
    counts += np.where(win_stop[affected] > dead_stop[affected], (hi_post - lo_post).astype(np.float64), 0.0)

    rates[affected] = np.where(valid_duration > 0, counts / np.where(valid_duration > 0, valid_duration, 1.0), np.nan)
    return rates


def lick_aligned_bin_population_matrices(
    unit_spikes_by_cluster: dict,
    unit_cluster_ids: np.ndarray,
    event_time: np.ndarray,
    start_time: np.ndarray,
    is_whisker: np.ndarray,
    bin_edges: list[tuple[float, float]],
    dead_zone: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """`bin_population_matrices` analog for an event-aligned (e.g.
    lick_time) window -- per-trial dead-zone masking via
    `event_aligned_rates_for_trials` instead of a globally-dropped bin."""
    matrices = []
    for window in bin_edges:
        X = np.full((len(event_time), len(unit_cluster_ids)), np.nan)
        for j, cid in enumerate(unit_cluster_ids):
            spikes = unit_spikes_by_cluster.get(cid, np.array([]))
            X[:, j] = event_aligned_rates_for_trials(spikes, event_time, start_time, is_whisker, window, dead_zone=dead_zone)
        matrices.append(X)
    return matrices


# ---------------------------------------------------------------------------
# Fixed-C selection (once, on a wide summary window) + per-bin decoding
# ---------------------------------------------------------------------------

def wide_window_matrix_from_bins(matrices: list[np.ndarray]) -> np.ndarray:
    """Summary-window feature matrix built from the already-computed
    per-bin matrices (mean rate per unit across bins), rather than a
    direct `population_design_matrix` call on the full window -- a direct
    call would straddle the whole dead zone and
    `clip_window_for_whisker` deliberately refuses to auto-resolve that
    (ambiguous for one contiguous window). Since bins are equal-width and
    the dead-zone bin is already excluded from `matrices`, the plain mean
    is a faithful average-rate-over-the-window summary."""
    return np.nanmean(np.stack(matrices, axis=0), axis=0)


def _drop_nan_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop trials with a NaN feature value in this bin -- needed for
    event-aligned (e.g. lick_time) windows, where the per-trial dead-zone
    mask (see `lick_aligned_bin_population_matrices`) can invalidate a
    trial for one bin without invalidating it for every bin, unlike the
    start_time-aligned grid which drops one whole bin globally instead."""
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], y[valid]


def _make_classifier(C: float):
    """L2-logistic behind a per-fold StandardScaler (switched from L1 to L2
    2026-09-13 per user request, for the next sweep iteration onward --
    prior runs in this project used L1). The scaling rationale from the
    original 2026-09-10 decision still applies regardless of penalty type:
    the regularization penalty is applied in raw firing-rate units, so units
    with different baseline rates get an inconsistent effective penalty
    without scaling. Wrapped in a Pipeline so the scaler is fit on the
    training fold only, never on held-out data -- fitting it on the full X
    before CV would leak test-fold statistics into the training features."""
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear", C=C, max_iter=1000))


def select_fixed_c(X_wide: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_folds: int = 5) -> float:
    """Nested-CV-free but grid-honest C selection: k-fold CV over the C
    grid on the wide summary window once; returns the C with the best mean
    balanced accuracy. This is the one place nested-CV-equivalent cost is
    paid per (session, area, day_stage, half, scope) -- reused unchanged
    across every bin's decode, which is what makes the ~15-bin (at 10ms
    over a ~600ms window) sweep tractable."""
    X_wide, y = _drop_nan_rows(X_wide, y)
    best_c, best_score = C_GRID[0], -np.inf
    seed = int(rng.integers(0, 2**31 - 1))
    for c in C_GRID:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, te in skf.split(X_wide, y):
            clf = _make_classifier(c)
            clf.fit(X_wide[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], clf.predict(X_wide[te])))
        if np.mean(scores) > best_score:
            best_score, best_c = np.mean(scores), c
    return float(best_c)


def decode_bin(X: np.ndarray, y: np.ndarray, C: float, rng: np.random.Generator, n_repeats: int = 5, n_folds: int = 5) -> float:
    """Mean held-out balanced accuracy for one bin's design matrix, fixed C,
    `n_repeats` x `n_folds` CV. Rows with a NaN feature (dead-zone-masked
    trials for this bin, event-aligned windows only) are dropped first."""
    X, y = _drop_nan_rows(X, y)
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < n_folds:
        return float("nan")
    scores = []
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = _make_classifier(C)
            clf.fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores)) if scores else float("nan")


def decode_curve(matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_repeats: int = 5, n_folds: int = 5) -> np.ndarray:
    return np.array([decode_bin(X, y, C, rng, n_repeats, n_folds) for X in matrices])


# ---------------------------------------------------------------------------
# Pooled-prediction scoring + resample-on-missing-class (user request
# 2026-09-14): matches the BWM decoding paper's methodology on exactly two
# axes -- how a partition's held-out predictions are turned into one score,
# and how a fold missing a class is handled -- while deliberately KEEPING
# this project's existing single-C-across-all-bins cost tradeoff (user
# confirmed 2026-09-14: BWM's fully-nested per-bin C selection would
# multiply per-bin cost by the size of C_GRID; not adopted).
#
# Why pooled scoring: `decode_bin`'s mean-of-separately-scored-folds is not
# the same estimator as scoring once on every fold's concatenated held-out
# predictions -- they only agree when every fold has plenty of both
# classes. At this project's data-sufficiency floor (MIN_TRIALS_PER_CLASS
# as low as 5), a fold can land with only 1-2 minority-class trials, making
# that fold's own balanced accuracy a high-variance quantity that then gets
# averaged in at full weight. Pooling first means each class's recall is
# computed from its full held-out count across the whole partition at once.
#
# Why resample instead of skip: `decode_bin` silently drops ("continue")
# a fold whose training portion lacks a class -- fine when a score is a
# mean over many folds (one fewer number in the average), but incompatible
# with pooled scoring, where every trial must get exactly one held-out
# prediction per repeat; skipping a fold would leave some trials
# unpredicted and break that guarantee. Resampling (redraw the whole
# partition until every fold's training portion has both classes) is BWM's
# own fix for this and is required here, not just consistent with it.
#
# Added ALONGSIDE (not replacing) select_fixed_c/decode_bin/decode_curve --
# user request 2026-09-14, "keep the current code ... as backup ... do not
# overwrite them", since existing/queued sweeps still call the originals
# and their results are worth keeping for direct before/after comparison.
# ---------------------------------------------------------------------------

def _stratified_kfold_with_resample(
    X: np.ndarray, y: np.ndarray, n_folds: int, rng: np.random.Generator, max_resamples: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """One full n_folds-way partition of (X, y), redrawn (fresh shuffle
    seed) up to `max_resamples` times until every fold's training portion
    contains every class present in `y`. `max_resamples` lowered 50->20
    2026-09-15 (user decision, cost-reduction pass on the pooled estimator
    now that it's the production default -- see 024_master_sweep.py's
    `N_SHUF`/`NULL_N_REPEATS` comments for the sibling cost levers this
    was weighed against). Returns None if no valid partition was found in
    the budget -- can only happen when a class is so rare relative to
    n_folds that no split can avoid it, which the calling
    functions' own `min(bincount) < n_folds` pre-check should already rule
    out in practice."""
    n_classes = len(np.unique(y))
    for _ in range(max_resamples):
        seed = int(rng.integers(0, 2**31 - 1))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
        if all(len(np.unique(y[tr])) == n_classes for tr, _ in splits):
            return splits
    return None


def _effective_folds(y: np.ndarray, n_folds: int) -> int:
    """min(n_folds, minority-class count) (2026-09-28, with MIN_TRIALS_PER_CLASS lowered to 3): a class with
    fewer members than folds cannot be stratified. With >= n_folds members per class this returns n_folds, so
    results for such conditions are unchanged. 0 if a class is missing."""
    counts = np.bincount(np.asarray(y).astype(int), minlength=2)
    return int(min(n_folds, counts.min())) if (counts > 0).all() else 0


def select_fixed_c_pooled(X_wide: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_folds: int = 5, max_resamples: int = 20) -> float:
    """Same one-time, window-summary C selection as `select_fixed_c` (a
    single C is still reused unchanged across every bin) but scores each
    C_GRID candidate with pooled held-out predictions from one
    resample-guaranteed partition, for consistency with
    `decode_bin_pooled`'s scoring convention. Same cost as `select_fixed_c`
    (one partition per candidate C, no repeats) -- only the scoring
    convention changes, not how many models get fit."""
    X_wide, y = _drop_nan_rows(X_wide, y)
    n_folds = _effective_folds(y, n_folds)
    if n_folds < 2:
        return float(C_GRID[len(C_GRID) // 2])
    best_c, best_score = C_GRID[0], -np.inf
    for c in C_GRID:
        splits = _stratified_kfold_with_resample(X_wide, y, n_folds, rng, max_resamples)
        if splits is None:
            continue
        y_pred = np.empty(len(y), dtype=y.dtype)
        for tr, te in splits:
            clf = _make_classifier(c)
            clf.fit(X_wide[tr], y[tr])
            y_pred[te] = clf.predict(X_wide[te])
        score = balanced_accuracy_score(y, y_pred)
        if score > best_score:
            best_score, best_c = score, c
    return float(best_c)


def decode_bin_pooled(X: np.ndarray, y: np.ndarray, C: float, rng: np.random.Generator, n_repeats: int = 5, n_folds: int = 5, max_resamples: int = 20) -> float:
    """Pooled-prediction analog of `decode_bin`: within each of
    `n_repeats` independent, resample-guaranteed partitions, every trial
    gets exactly one held-out prediction (concatenated across that
    partition's n_folds folds), and balanced accuracy is computed once on
    the full pooled set -- then the `n_repeats` pooled scores are averaged
    (same total number of model fits as `decode_bin`, n_repeats x n_folds;
    only the aggregation step differs)."""
    X, y = _drop_nan_rows(X, y)
    n_folds = _effective_folds(y, n_folds)
    if n_folds < 2:
        return float("nan")
    repeat_scores = []
    for _ in range(n_repeats):
        splits = _stratified_kfold_with_resample(X, y, n_folds, rng, max_resamples)
        if splits is None:
            continue
        y_pred = np.empty(len(y), dtype=y.dtype)
        for tr, te in splits:
            clf = _make_classifier(C)
            clf.fit(X[tr], y[tr])
            y_pred[te] = clf.predict(X[te])
        repeat_scores.append(balanced_accuracy_score(y, y_pred))
    return float(np.mean(repeat_scores)) if repeat_scores else float("nan")


def decode_curve_pooled(matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_repeats: int = 5, n_folds: int = 5) -> np.ndarray:
    return np.array([decode_bin_pooled(X, y, C, rng, n_repeats, n_folds) for X in matrices])


def label_shuffle_null_curves_pooled(
    matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_shuf: int, n_repeats: int = 2, n_folds: int = 5,
) -> np.ndarray:
    """Pooled-scoring analog of `label_shuffle_null_curves`, for a null
    comparison that uses the same estimator as `decode_curve_pooled`'s real
    curve."""
    nulls = np.full((n_shuf, len(matrices)), np.nan)
    for s in range(n_shuf):
        y_shuf = rng.permutation(y)
        nulls[s] = decode_curve_pooled(matrices, y_shuf, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    return nulls


def linear_shift_null_curves_pooled(
    matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_shuf: int, n_repeats: int = 2,
    n_folds: int = 5, min_shift_frac: float = 0.1, max_shift_frac: float = 0.5,
) -> np.ndarray:
    """Pooled-scoring analog of `linear_shift_null_curves` (user request
    2026-09-14, added alongside the pooled-CV pilot's use of both null
    controls) -- same linear (truncating, no wraparound) shift construction,
    scored via `decode_curve_pooled` instead of `decode_curve` for
    consistency with the pooled estimator's real curve."""
    n = len(y)
    min_shift = max(1, int(min_shift_frac * n))
    max_shift = max(min_shift, int(max_shift_frac * n))
    nulls = np.full((n_shuf, len(matrices)), np.nan)
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
        if rng.random() < 0.5:
            y_shift = y[shift:]
            matrices_shift = [m[: n - shift] for m in matrices]
        else:
            y_shift = y[: n - shift]
            matrices_shift = [m[shift:] for m in matrices]
        nulls[s] = decode_curve_pooled(matrices_shift, y_shift, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    return nulls


# ---------------------------------------------------------------------------
# Curve-value regression -> quantile-class ("perfquant") decoding (user
# request 2026-09-19): instead of a 2-class high/low perf-state label,
# regress a continuous behavioral curve value (whisker learning-curve
# hit-rate, interpolated false-alarm rate, or their difference =
# "performance") from population activity, then score whether the
# regressor's continuous prediction falls in the same quintile (5-class,
# 20% pooled-population split) as the true value -- a classification
# accuracy computed from a regression, not a classifier. Added ALONGSIDE
# the existing 2-class curve-based `perfstate` target, not replacing it.
# Quantile edges are fit ONCE, globally, across the whole pooled
# population of trials (not per-session/per-mouse) -- see
# `049_perfquant_curve_regression_test.py`, which does that first pass.
# ---------------------------------------------------------------------------

RIDGE_ALPHA_GRID = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])


def _make_regressor(alpha: float):
    """Ridge regression behind a per-fold StandardScaler -- same rationale
    as `_make_classifier`: the penalty is applied in raw firing-rate units,
    so units with different baseline rates need scaling first; the scaler
    is fit train-fold-only to avoid leaking held-out statistics."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def quantile_edges_from_pooled_values(values: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Fixed quantile-class bin edges (20% splits for `n_classes=5`) fit
    once on a pooled population of curve values -- callers reuse this same
    array for every session's true-vs-predicted class comparison so a
    'class 3' means the same absolute performance level everywhere,
    per user decision 2026-09-19 ('pooled across all sessions/trials').
    Returns the `n_classes+1` bin edges (including -inf/+inf endpoints,
    so every value -- including a held-out prediction outside the training
    range -- always lands in a bin); pass `edges[1:-1]` to `np.digitize`."""
    cut_points = np.quantile(values, np.linspace(0, 1, n_classes + 1)[1:-1])
    return np.concatenate([[-np.inf], cut_points, [np.inf]])


def select_fixed_alpha_pooled(X_wide: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_folds: int = 5) -> float:
    """Regression analog of `select_fixed_c_pooled`: one-time alpha
    selection via out-of-fold R^2 on a single window-summary feature
    matrix, one `KFold` partition shared across every `RIDGE_ALPHA_GRID`
    candidate (no repeats -- unlike classification there's no class-
    imbalance degenerate-fold failure mode for a continuous target, so no
    resample-guarantee loop is needed)."""
    X_wide, y = _drop_nan_rows(X_wide, y)
    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_wide))
    best_alpha, best_score = RIDGE_ALPHA_GRID[0], -np.inf
    for a in RIDGE_ALPHA_GRID:
        y_pred = np.empty(len(y))
        for tr, te in splits:
            reg = _make_regressor(a)
            reg.fit(X_wide[tr], y[tr])
            y_pred[te] = reg.predict(X_wide[te])
        score = r2_score(y, y_pred)
        if score > best_score:
            best_score, best_alpha = score, a
    return float(best_alpha)


def decode_regression_quantile_pooled(
    X: np.ndarray, y: np.ndarray, alpha: float, quantile_edges: np.ndarray, rng: np.random.Generator,
    n_repeats: int = 5, n_folds: int = 5,
) -> dict:
    """Pooled-prediction regression analog of `decode_bin_pooled`: within
    each of `n_repeats` independent `KFold` partitions, every trial gets
    exactly one held-out continuous prediction (concatenated across that
    partition's folds); both the true values and the pooled predictions
    are then digitized into the same externally-supplied, globally-fit
    `quantile_edges` (see `quantile_edges_from_pooled_values`), and
    quantile-MATCH accuracy (fraction of trials whose predicted class
    equals its true class) is computed per repeat and averaged. Also
    returns the underlying pooled R^2 (reference only, not the primary
    metric -- the quantile-match accuracy is what's comparable to a
    shift-null's chance level of `1/n_classes`)."""
    X, y = _drop_nan_rows(X, y)
    if len(y) < n_folds:
        return dict(quantile_acc=float("nan"), r2=float("nan"))
    true_class = np.digitize(y, quantile_edges[1:-1])
    acc_scores, r2_scores = [], []
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        y_pred = np.empty(len(y))
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            reg = _make_regressor(alpha)
            reg.fit(X[tr], y[tr])
            y_pred[te] = reg.predict(X[te])
        pred_class = np.digitize(y_pred, quantile_edges[1:-1])
        acc_scores.append(float(np.mean(pred_class == true_class)))
        r2_scores.append(float(r2_score(y, y_pred)))
    return dict(quantile_acc=float(np.mean(acc_scores)), r2=float(np.mean(r2_scores)))


def linear_shift_null_regression_quantile_pooled(
    X: np.ndarray, y: np.ndarray, alpha: float, quantile_edges: np.ndarray, rng: np.random.Generator, n_shuf: int,
    n_repeats: int = 2, n_folds: int = 5, min_shift_frac: float = 0.1, max_shift_frac: float = 0.5,
) -> dict:
    """Regression analog of `linear_shift_null_curves_pooled`: same linear
    (truncating, no wraparound) shift between features and target, scored
    via `decode_regression_quantile_pooled`'s quantile-match accuracy AND
    R^2 (added 2026-09-20, user: "What about ... regression directly?" --
    the real fit's R^2 alone doesn't say whether it beats chance; a
    substantial real R^2 that's matched by an equally substantial null R^2
    is exactly the shared-slow-drift confound this null exists to catch,
    the same way it does for `hitmiss`/`perfstate`/`modality_lick`).
    Returns a dict of `(n_shuf,)` arrays: `quantile_acc` (chance level
    `1/n_classes`) and `r2` (chance level ~0, but the null's own
    distribution, not 0, is the real reference point)."""
    n = len(y)
    min_shift = max(1, int(min_shift_frac * n))
    max_shift = max(min_shift, int(max_shift_frac * n))
    acc_nulls, r2_nulls = np.full(n_shuf, np.nan), np.full(n_shuf, np.nan)
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
        if rng.random() < 0.5:
            y_shift, X_shift = y[shift:], X[: n - shift]
        else:
            y_shift, X_shift = y[: n - shift], X[shift:]
        res = decode_regression_quantile_pooled(
            X_shift, y_shift, alpha, quantile_edges, rng, n_repeats=n_repeats, n_folds=n_folds,
        )
        acc_nulls[s], r2_nulls[s] = res["quantile_acc"], res["r2"]
    return dict(quantile_acc=acc_nulls, r2=r2_nulls)


def residualize_curve(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Subtract each session's own centered rolling mean from a curve-value
    array (added 2026-09-20, user: "Why residual target, how would that
    help?"): removes the slow, session-spanning trend shared between the
    behavioral curve and any equally slow, task-irrelevant drift in
    population firing rates (electrode settling, satiety, arousal) -- a
    regressor can otherwise score well on EITHER the correctly-paired data
    or a `linear_shift_null_*` shuffle just by reading off "roughly where
    in the session is this trial" from either signal, with no real
    trial-by-trial neural-behavior coupling required (this is exactly why
    real quantile-match accuracy tracked its own shift-null so closely in
    the first, non-residualized validation run). What's left after
    residualizing is the trial-to-trial deviation from the local trend --
    if the population still predicts that, it's evidence of coupling
    beyond shared drift. `window=15` trials, centered, `min_periods=1` (so
    edge trials aren't dropped, just detrended against a smaller window)."""
    s = pd.Series(values)
    trend = s.rolling(window=window, center=True, min_periods=1).mean()
    return (s - trend).to_numpy()


# ---------------------------------------------------------------------------
# Cross-condition generalization (user request 2026-09-11): fit once on ALL
# of one condition's trials (e.g. first half), test once on ALL of another
# condition's trials (e.g. second half) -- the two trial sets are already
# disjoint by construction, so no CV is needed here (unlike decode_bin's
# within-condition curve). Deterministic given the classifier (liblinear
# logistic + StandardScaler have no randomness at fixed C), so no rng/n_repeats.
# ---------------------------------------------------------------------------

def decode_bin_crossgen(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, C: float) -> float:
    X_train, y_train = _drop_nan_rows(X_train, y_train)
    X_test, y_test = _drop_nan_rows(X_test, y_test)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    clf = _make_classifier(C)
    clf.fit(X_train, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(X_test)))


def decode_curve_crossgen(
    matrices_train: list[np.ndarray], y_train: np.ndarray, matrices_test: list[np.ndarray], y_test: np.ndarray, C: float
) -> np.ndarray:
    """Per-bin generalization accuracy: fit on condition A's trials (all of
    them, this bin's feature matrix), predict condition B's trials, same
    `C` A's within-condition curve was fit with. Call twice (A train/B test,
    then B train/A test) for a symmetric pair of cross-gen curves."""
    return np.array([
        decode_bin_crossgen(Xtr, y_train, Xte, y_test, C) for Xtr, Xte in zip(matrices_train, matrices_test)
    ])


def mean_in_window(curve: np.ndarray, bin_labels_s: np.ndarray, window: tuple[float, float]) -> float:
    """Mean decode accuracy over bins whose label time falls in `window`
    (e.g. the 5-35ms post-stimulus sensory window, or the 150ms-before-lick
    pre-lick window) -- used for the window-quantification figures/tests,
    not for the full time-resolved curves."""
    w0, w1 = window
    mask = (bin_labels_s >= w0) & (bin_labels_s <= w1)
    return float(np.nanmean(curve[mask])) if mask.any() else float("nan")


# ---------------------------------------------------------------------------
# Significance: label-shuffle null + cluster-based permutation across time
# ---------------------------------------------------------------------------

def label_shuffle_null_curves(
    matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_shuf: int, n_repeats: int = 2, n_folds: int = 5
) -> np.ndarray:
    """`n_shuf` x `n_bins` null accuracy curves. The same shuffled label
    vector is reused across all bins within one shuffle draw (preserves the
    across-bin correlation structure a label permutation removes only the
    trial-identity signal, not e.g. session-wide drift)."""
    nulls = np.full((n_shuf, len(matrices)), np.nan)
    for s in range(n_shuf):
        y_shuf = rng.permutation(y)
        nulls[s] = decode_curve(matrices, y_shuf, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    return nulls


def linear_shift_null_curves(
    matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_shuf: int, n_repeats: int = 2,
    n_folds: int = 5, min_shift_frac: float = 0.1, max_shift_frac: float = 0.5,
) -> np.ndarray:
    """Second, complementary null control (user request 2026-09-13, added
    for `hitmiss`/`perfstate` decoding only -- NOT `modality`, which has no
    comparable slow-drift confound to guard against): `n_shuf` x `n_bins`
    LINEAR-shift null curves. Unlike `label_shuffle_null_curves`'s full
    random permutation (which destroys ALL label structure, trial-order
    included), each draw here offsets the label sequence relative to the
    neural data by a random lag and simply **truncates** the non-overlapping
    trials at whichever end the shift pushes past (no wraparound) --
    switched from an earlier circular-shift (`np.roll`) version per user
    correction 2026-09-13 ("do linear shifts, not circular"): a circular
    wrap pairs the label sequence's end back onto its own start, an
    artificial discontinuity a real session's slow drift never has, whereas
    a linear (truncating) shift never manufactures a pairing that couldn't
    have occurred in the real trial order.

    This keeps the label vector's own local autocorrelation/block structure
    intact (e.g. perf_state's 5-trial blocks, or a slow session-wide drift
    in hit rate) while breaking its true trial-by-trial correspondence to
    the neural data -- the standard control for the failure mode a
    label-shuffle null can't catch: neural activity and the label each
    independently drifting slowly across the session, which would inflate
    decoding accuracy under a naive train/test split without the two
    actually being trial-locked.

    `min_shift_frac`/`max_shift_frac` bound the shift magnitude as a
    fraction of the trial count `n`: at least `min_shift_frac*n` (so no
    draw is a near-identity shift that leaks most of the true alignment
    back in) and at most `max_shift_frac*n` (so no draw truncates away more
    than half the trials, keeping the null itself reasonably powered).
    Shift direction (which end gets truncated) is randomized per draw."""
    n = len(y)
    min_shift = max(1, int(min_shift_frac * n))
    max_shift = max(min_shift, int(max_shift_frac * n))
    nulls = np.full((n_shuf, len(matrices)), np.nan)
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
        if rng.random() < 0.5:
            y_shift = y[shift:]
            matrices_shift = [m[: n - shift] for m in matrices]
        else:
            y_shift = y[: n - shift]
            matrices_shift = [m[shift:] for m in matrices]
        nulls[s] = decode_curve(matrices_shift, y_shift, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    return nulls


def cluster_permutation_test(observed: np.ndarray, null_curves: np.ndarray, z_threshold: float = 1.96, two_sided: bool = False) -> dict:
    """Cluster-mass permutation test across time bins. Per-bin z-score
    against the null distribution's mean/std, threshold at `z_threshold`,
    sum z within each contiguous run of above-threshold bins (cluster
    mass), then compare the observed max cluster mass against the null
    distribution of *each null curve's own* max cluster mass (built the
    same way against the remaining null curves' mean/std).

    `two_sided=False` (default, the above-chance-decoding use case):
    clusters only where `z > z_threshold`. `two_sided=True` (the R+/R-
    curve-difference use case, where the effect could point either way):
    clusters where `|z| > z_threshold`, cluster mass built from `|z|`, sign
    of each cluster reported separately in `observed_cluster_signs`."""
    null_mean = np.nanmean(null_curves, axis=0)
    null_std = np.nanstd(null_curves, axis=0)
    null_std = np.where(null_std == 0, np.nan, null_std)

    def cluster_masses_and_signs(curve: np.ndarray) -> tuple[list[float], list[int]]:
        z = (curve - null_mean) / null_std
        above = (z > z_threshold) if not two_sided else (np.abs(z) > z_threshold)
        masses, signs, run, run_sign = [], [], 0.0, 0
        for a, zi in zip(above, z):
            if a and np.isfinite(zi):
                run += abs(zi) if two_sided else zi
                run_sign = int(np.sign(zi))
            else:
                if run > 0:
                    masses.append(run)
                    signs.append(run_sign)
                run, run_sign = 0.0, 0
        if run > 0:
            masses.append(run)
            signs.append(run_sign)
        return masses, signs

    obs_masses, obs_signs = cluster_masses_and_signs(observed)
    obs_max = max(obs_masses) if obs_masses else 0.0
    obs_max_sign = obs_signs[int(np.argmax(obs_masses))] if obs_masses else 0

    null_max = np.array([max(cluster_masses_and_signs(null_curves[s])[0], default=0.0) for s in range(null_curves.shape[0])])
    p_value = float((np.sum(null_max >= obs_max) + 1) / (len(null_max) + 1))

    return {
        "observed_max_cluster_mass": obs_max,
        "observed_max_cluster_sign": obs_max_sign,
        "null_max_cluster_mass_dist": null_max,
        "p_value": p_value,
        "observed_z": (observed - null_mean) / null_std,
    }


# ---------------------------------------------------------------------------
# Data-sufficiency gate
# ---------------------------------------------------------------------------

def data_sufficiency_ok(n_units: int, y: np.ndarray, min_units: int = MIN_UNITS_PER_AREA, min_trials_per_class: int = MIN_TRIALS_PER_CLASS) -> tuple[bool, str]:
    if n_units < min_units:
        return False, f"only {n_units} units (< {min_units})"
    n_pos, n_neg = int(np.sum(y)), int(np.sum(~y.astype(bool)))
    if n_pos < min_trials_per_class or n_neg < min_trials_per_class:
        return False, f"class counts hit={n_pos}/miss={n_neg} (< {min_trials_per_class} each)"
    return True, "ok"


# ---------------------------------------------------------------------------
# Group-level significance (pivot, user decision 2026-09-10): the per-session
# label-shuffle null above is too expensive to run at the full sweep's scale
# (verified 2026-09-10 smoke test: ~1s/shuffle x hundreds of shuffles needed
# for a non-floor p-value x ~3500 (session, area, half) cells). Instead:
# compute the real decode curve AND exactly *one* shuffled-label surrogate
# curve per session (same cost as two real decodes, not n_shuf), then build
# the GROUP-level null by randomly choosing, per *mouse* (not per session --
# a mouse can contribute several expert sessions), whether to use that
# mouse's real or surrogate curves for every one of its sessions in the
# group. This is a standard subject-level sign-flip/randomization group
# test (the same permutation logic already used elsewhere in this dataset's
# analyses for a mouse-level factor, e.g. the PERMANOVA/cohort-test mouse-
# block permutation fix) -- cost is ~2x a real-only sweep, independent of
# how many permutation draws are used at the group level (those draws just
# resample already-computed numbers).
# ---------------------------------------------------------------------------

def session_real_and_shuffled_curves(
    matrices: list[np.ndarray], y: np.ndarray, C: float, rng: np.random.Generator, n_repeats: int = 5, n_folds: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """One session's (real_curve, surrogate_curve) pair -- the only
    per-session decode-curve cost this design pays (2x a real-only decode,
    not n_shuf x). `surrogate_curve` is from a single label-permutation
    draw of this session's own trials, same CV settings as the real curve."""
    real = decode_curve(matrices, y, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    y_shuf = rng.permutation(y)
    surrogate = decode_curve(matrices, y_shuf, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    return real, surrogate


def group_mouseblock_permutation_null(
    session_records: list[dict], rng: np.random.Generator, n_perm: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Group-level observed mean curve and its mouse-block sign-flip
    permutation null, from a list of per-session dicts each with keys
    `subject_id`, `real_curve`, `surrogate_curve` (all sessions must
    already be restricted to the one group being tested -- one cohort, one
    day_stage, one half, one area, one population scope).

    Returns (observed_mean_curve, null_curves) where `null_curves` has
    shape (n_perm, n_bins) -- directly usable as `cluster_permutation_test`'s
    `null_curves` argument.

    Permutation mechanics: for each of `n_perm` draws, every *mouse*
    contributing to this group independently gets a fair-coin choice of
    "real" or "surrogate" for **all** of its sessions in the group (mouse-
    block, not session-block -- a mouse's multiple expert sessions move
    together, so they can't be pseudo-independent evidence for the null the
    way naive per-session flipping would allow)."""
    subject_ids = sorted({r["subject_id"] for r in session_records})
    real_stack = np.stack([r["real_curve"] for r in session_records])  # (n_sessions, n_bins)
    surrogate_stack = np.stack([r["surrogate_curve"] for r in session_records])
    subj_idx = np.array([subject_ids.index(r["subject_id"]) for r in session_records])

    observed_mean_curve = np.nanmean(real_stack, axis=0)

    null_curves = np.full((n_perm, real_stack.shape[1]), np.nan)
    for p in range(n_perm):
        use_real_by_mouse = rng.integers(0, 2, size=len(subject_ids)).astype(bool)
        use_real = use_real_by_mouse[subj_idx]
        chosen = np.where(use_real[:, None], real_stack, surrogate_stack)
        null_curves[p] = np.nanmean(chosen, axis=0)

    return observed_mean_curve, null_curves


def rplus_rminus_group_test(session_summary: pd.DataFrame, rng: np.random.Generator, n_perm: int = 2000) -> dict:
    """Compare a per-session summary decoding metric (e.g. peak or
    mean-in-window balanced accuracy; `session_summary` needs columns
    `subject_id`, `reward_group`, `metric`) between R+ and R-. Runs the
    mandatory non-parametric+parametric test pair at session grain
    (`ssl_cohort_comparison_stats.md`) **and** a mouse-block permutation of
    the group-mean difference -- `reward_group` is mouse-level even though
    the row grain here is already sessions (`ssl_stats_unit_of_analysis.md`:
    session-grain does not exempt a mouse-level factor from block
    permutation when a mouse contributes multiple expert sessions)."""
    from scipy import stats

    rplus = session_summary.loc[session_summary.reward_group == "R+", "metric"].to_numpy()
    rminus = session_summary.loc[session_summary.reward_group == "R-", "metric"].to_numpy()
    mw = stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    welch = stats.ttest_ind(rplus, rminus, equal_var=False)
    observed_diff = float(np.mean(rplus) - np.mean(rminus))

    subject_group = session_summary.drop_duplicates("subject_id").set_index("subject_id")["reward_group"]
    subject_ids = subject_group.index.to_numpy()
    null_diffs = np.full(n_perm, np.nan)
    for p in range(n_perm):
        perm_labels = rng.permutation(subject_group.to_numpy())
        perm_map = dict(zip(subject_ids, perm_labels))
        perm_group = session_summary["subject_id"].map(perm_map)
        pr = session_summary.loc[perm_group == "R+", "metric"].to_numpy()
        pm = session_summary.loc[perm_group == "R-", "metric"].to_numpy()
        null_diffs[p] = np.mean(pr) - np.mean(pm) if len(pr) and len(pm) else np.nan
    perm_p = float((np.nansum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (np.sum(~np.isnan(null_diffs)) + 1))

    return {
        "mannwhitney_p": float(mw.pvalue),
        "welch_p": float(welch.pvalue),
        "observed_diff": observed_diff,
        "mouse_block_perm_p": perm_p,
        "n_rplus_sessions": len(rplus),
        "n_rminus_sessions": len(rminus),
        "n_mice": len(subject_ids),
    }


def learning_expert_group_test(session_summary: pd.DataFrame, rng: np.random.Generator, n_perm: int = 2000) -> dict:
    """Learning-vs-expert analog of `rplus_rminus_group_test` (added
    2026-09-14, user request: "side by side comparison learning vs expert"
    for the whole_brain statistics figures) -- same non-parametric+
    parametric pair plus mouse-block permutation, but grouping on
    `day_stage` ('learning'/'expert') instead of `reward_group`.

    `session_summary` needs columns `subject_id`, `day_stage`, `metric`.
    Mouse-block permutation matters *more* here than for R+/R-: a mouse
    typically contributes exactly one learning-stage session but several
    expert-stage sessions, so the two groups are not independent samples of
    sessions -- they overlap in which mice they come from. Permuting whole
    mice between the two labels (not individual sessions) is what makes the
    test respect that structure; a naive per-session Mann-Whitney/Welch
    would treat a mouse's several expert sessions as independent evidence
    against its own single learning session, which they are not.
    """
    from scipy import stats

    learning = session_summary.loc[session_summary.day_stage == "learning", "metric"].to_numpy()
    expert = session_summary.loc[session_summary.day_stage == "expert", "metric"].to_numpy()
    mw = stats.mannwhitneyu(learning, expert, alternative="two-sided")
    welch = stats.ttest_ind(learning, expert, equal_var=False)
    observed_diff = float(np.mean(learning) - np.mean(expert))

    subject_group = session_summary.drop_duplicates("subject_id").set_index("subject_id")["day_stage"]
    subject_ids = subject_group.index.to_numpy()
    null_diffs = np.full(n_perm, np.nan)
    for p in range(n_perm):
        perm_labels = rng.permutation(subject_group.to_numpy())
        perm_map = dict(zip(subject_ids, perm_labels))
        perm_group = session_summary["subject_id"].map(perm_map)
        pl = session_summary.loc[perm_group == "learning", "metric"].to_numpy()
        pe = session_summary.loc[perm_group == "expert", "metric"].to_numpy()
        null_diffs[p] = np.mean(pl) - np.mean(pe) if len(pl) and len(pe) else np.nan
    perm_p = float((np.nansum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (np.sum(~np.isnan(null_diffs)) + 1))

    return {
        "mannwhitney_p": float(mw.pvalue),
        "welch_p": float(welch.pvalue),
        "observed_diff": observed_diff,
        "mouse_block_perm_p": perm_p,
        "n_learning_sessions": len(learning),
        "n_expert_sessions": len(expert),
        "n_mice": len(subject_ids),
    }


def rplus_rminus_curve_difference(session_records: list[dict], rng: np.random.Generator, n_perm: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin R+ minus R- decoding-accuracy difference curve, with a
    mouse-block-permutation null for `cluster_permutation_test(...,
    two_sided=True)` -- answers "at which time bins does decoding ability
    differ between cohorts, and by how much" directly, not just a single
    summary-metric comparison (`rplus_rminus_group_test`). Uses only the
    already-computed *real* curves (no surrogate curves needed -- this is a
    two-group comparison, not an above-chance test), so it adds no extra
    per-session decoding cost.

    `session_records`: list of dicts with `subject_id`, `reward_group`
    ('R+'/'R-'), `real_curve`, already restricted to one (area, day_stage,
    half, population scope) cell. Mouse-block: the permutation reassigns
    `reward_group` labels to whole *mice*, keeping a mouse's multiple
    sessions together, matching the required treatment of `reward_group` as
    a mouse-level factor (`ssl_stats_unit_of_analysis.md`).

    Returns (observed_diff_curve, null_diff_curves) -- feed directly to
    `cluster_permutation_test(observed_diff_curve, null_diff_curves,
    two_sided=True)`.
    """
    subject_ids = sorted({r["subject_id"] for r in session_records})
    subject_group = {sid: next(r["reward_group"] for r in session_records if r["subject_id"] == sid) for sid in subject_ids}
    curve_stack = np.stack([r["real_curve"] for r in session_records])
    subj_idx = np.array([subject_ids.index(r["subject_id"]) for r in session_records])
    true_labels = np.array([subject_group[sid] for sid in subject_ids])

    def diff_curve_for_labels(labels_by_mouse: np.ndarray) -> np.ndarray:
        sess_labels = labels_by_mouse[subj_idx]
        rplus_mean = np.nanmean(curve_stack[sess_labels == "R+"], axis=0)
        rminus_mean = np.nanmean(curve_stack[sess_labels == "R-"], axis=0)
        return rplus_mean - rminus_mean

    observed_diff = diff_curve_for_labels(true_labels)

    null_curves = np.full((n_perm, curve_stack.shape[1]), np.nan)
    for p in range(n_perm):
        perm_labels = rng.permutation(true_labels)
        null_curves[p] = diff_curve_for_labels(perm_labels)

    return observed_diff, null_curves
