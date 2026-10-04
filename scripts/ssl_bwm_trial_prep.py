"""Shared per-session trial prep for the SSL BWM-style single-cell tests
(`projects/ssl-bwm-style-single-cell-decoding/`): cohort-corrected reward
status, the t-1-rewarded block analog, and its run index. Factored out of
the first (Modality) test script so Response and Prior-outcome reuse the
same, already-checked logic instead of re-deriving it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ssl_bwm_stats_util import t1_outcome_run_index

REF_PATH = Path("reports/ssl_analysis/derived/mouse_reference.parquet")

_DAY_RE = re.compile(r"whisker(?:_on_\d+_opto|_off_\d+_opto)?_([+-]?\d+)")


def session_training_day(session_description: str) -> int | None:
    """Parse the whisker-training day index (the `N` in a `whisker_N`
    session_description) -- this is the same index Axel Bisi's own
    per-mouse results tree uses for its `whisker_N/` subfolders
    (`combined_results_ks4/<mouse>/whisker_N/...`), so it doubles as the
    day-stage lookup (`day==0` -> "learning", `day>0` -> "expert") AND the
    folder-index needed to locate that session's learning-curve file.
    Returns None for a session_description that doesn't match the
    whisker-training pattern (opto/pretraining/etc.)."""
    m = _DAY_RE.match(str(session_description))
    return int(m.group(1)) if m else None


def list_whisker_training_ephys_sessions(sessions_tbl: pd.DataFrame) -> list[tuple[str, str]]:
    """(session_id, day_stage) for every has_ephys session whose
    session_description parses as a whisker-training day
    (day==0 -> "learning", day>0 -> "expert"); excludes
    auditory_*/free_licking_* pretraining sessions and day<0, per
    ssl_task_semantics.md's Day/training-stage semantics section."""
    out = []
    eph = sessions_tbl[sessions_tbl["has_ephys"]]
    for _, row in eph.iterrows():
        day = session_training_day(row["session_description"])
        if day is None or day < 0:
            continue
        out.append((row["session_id"], "learning" if day == 0 else "expert"))
    return out


def cohort_corrected_rewarded(trial_type: pd.Series, lick_flag: pd.Series, reward_group: str) -> np.ndarray:
    """True iff reward was delivered on that trial. whisker_trial flips by
    cohort (ssl_task_semantics.md); auditory_trial and no_stim_trial do not."""
    rewarded = np.zeros(len(trial_type), dtype=bool)
    is_wh = (trial_type == "whisker_trial").to_numpy()
    is_aud = (trial_type == "auditory_trial").to_numpy()
    lick = lick_flag.to_numpy().astype(bool)
    if reward_group == "R+":
        rewarded[is_wh] = lick[is_wh]
    elif reward_group == "R-":
        rewarded[is_wh] = ~lick[is_wh]
    else:
        raise ValueError(f"unexpected reward_group {reward_group!r}")
    rewarded[is_aud] = lick[is_aud]
    return rewarded


def load_reward_group(subject_id: str) -> str | None:
    ref = pd.read_parquet(REF_PATH)
    row = ref[ref["subject_id"] == subject_id]
    if len(row) != 1:
        return None
    rg = row.iloc[0]["reward_group"]
    if rg not in ("R+", "R-"):
        return None  # drop R+proba / NaN per established SSL practice
    return rg


def resolve_context(trials: pd.DataFrame) -> pd.Series:
    """Per-trial context (user, 2026-10-04): a trial whose gap to the previous or next trial is 3.0 +- 0.3 s (fixed
    passive ITI) is "passive"; otherwise "passive" if labelled so, else "active" (labelled active or the string "nan"
    of sessions without recorded context). `trials` sorted by start_time. perf == 6 trials inside active blocks stay
    active here and are dropped by the perf rule."""
    st = trials["start_time"].to_numpy()
    dp, dn = np.r_[np.inf, np.diff(st)], np.r_[np.diff(st), np.inf]
    fixed = (np.abs(dp - 3.0) < 0.3) | (np.abs(dn - 3.0) < 0.3)
    lab = trials["context"].astype(str).to_numpy()
    return pd.Series(np.where(fixed | (lab == "passive"), "passive", "active"), index=trials.index)


def prep_session(dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> dict | None:
    """Return active-epoch trials for `session_id` with `rewarded`,
    `t1_rewarded`, and `run_index` columns, or None if the session's subject
    has no usable reward_group. `trials_tbl`/`sessions_tbl` are the full
    metadata tables, passed in so callers only read parquet once."""
    trials = trials_tbl[trials_tbl["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    subject_id = sessions_tbl.loc[sessions_tbl["session_id"] == session_id, "subject_id"].iloc[0]
    reward_group = load_reward_group(subject_id)
    if reward_group is None:
        return None

    # Per-trial context (user rule 2026-10-04, skills/ssl-trial-exclusion): a trial in a fixed ~3 s ITI sequence is
    # passive whatever its label; otherwise labelled passive stays passive, labelled active and unlabelled "nan" are
    # active. Before 2026-10-04 only context == "active" was kept whenever a session had any label, which dropped
    # MH062_20260113_125836's 377-trial unlabelled training block.
    trials = trials[(resolve_context(trials) == "active").to_numpy()].reset_index(drop=True)
    # perf == 6 = excluded trials -- user rule 2026-10-01: always drop them in active-bound analyses (313 active trials in
    # 26 ssl_ephys sessions; results computed before 2026-10-01 included them).
    if "perf" in trials:
        trials = trials[trials["perf"] != 6].reset_index(drop=True)

    # Auditory-only warm-up block (Axel Bisi, the experimenter, 2026-09-14):
    # many whisker-training sessions begin with a block of auditory-only
    # trials (no whisker trials) to wake/engage the mouse before whisker
    # trials are introduced -- drop everything before the session's first
    # active whisker trial, in addition to the context filter above (not
    # instead of it). See ssl-analyze/references/ssl_auditory_warmup_block.md
    # for why: left in, this block's run structure was found to produce a
    # spurious above-chance pre-stimulus "modality" decode. No-op for a
    # session with no whisker trials at all (shouldn't occur for anything
    # that passed list_whisker_training_ephys_sessions, but safe either way).
    #
    # 2026-09-28 rule (Axel Bisi; skills/ssl-analyze/references/ssl_auditory_warmup_block.md): never remove the
    # first whisker trial, and always keep exactly 1 trial before it -- the warm-up block is removed except its
    # last trial. t-1 outcomes are computed on the full active sequence BEFORE the cut, so the kept pre-whisker
    # trial and the first whisker trial both have a real t-1; nothing else is dropped. Only a session whose very
    # first active trial is a whisker trial (no trial before it exists) keeps that trial with t1_rewarded = NaN;
    # t-1-dependent analyses drop NaN-t1 rows themselves (e.g. ssl_bwm_single_cell).
    # History: 2026-09-14 to 2026-09-28 the cut started AT the first whisker trial and a blanket "drop first trial
    # (no t-1)" then removed it; before 2026-09-14 that drop removed the first active trial (the first whisker
    # trial in sessions without a warm-up block).
    trials["rewarded"] = cohort_corrected_rewarded(trials["trial_type"], trials["lick_flag"], reward_group)
    trials["t1_rewarded"] = trials["rewarded"].shift(1)
    whisker_idx = trials.index[trials["trial_type"] == "whisker_trial"]
    if len(whisker_idx) > 0:
        trials = trials.loc[max(0, int(whisker_idx[0]) - 1):].reset_index(drop=True)
    trials["run_index"] = t1_outcome_run_index(trials["t1_rewarded"].to_numpy())

    return {
        "session_id": session_id,
        "subject_id": subject_id,
        "reward_group": reward_group,
        "trials": trials,
    }
