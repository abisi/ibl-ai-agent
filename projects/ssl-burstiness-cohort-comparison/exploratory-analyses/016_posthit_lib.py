"""Helper for restricting analysis to data after each session's first
cohort-corrected whisker-trial hit -- per-session "engagement onset" cutoff.

Cohort-corrected hit (per ssl_task_semantics.md, NOT the naive lick_flag
reading): R+ hit = whisker_trial & lick_flag==1 (licked); R- hit =
whisker_trial & lick_flag==0 (withheld, the rewarded action for that
cohort). Restricted to context=='active' (passive lick_flag is not
meaningful). reward_group here is the reference-sheet source (mouse-level,
stable) already merged into unit_table by apply_mouse_filters -- NOT the
native per-session wh_reward, which can flip within a mouse (see
question.md's cohort-factor-source decision) and would make "cohort-
corrected hit" ambiguous mid-session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def first_hit_time(sess_trials: pd.DataFrame, reward_group: str) -> float | None:
    wh = sess_trials[(sess_trials.trial_type == "whisker_trial") & (sess_trials.context == "active")]
    hit_mask = (wh.lick_flag == 1) if reward_group == "R+" else (wh.lick_flag == 0)
    hits = wh[hit_mask].sort_values("trial_id")
    if len(hits) == 0:
        return None
    return float(hits["start_time"].iloc[0])


def session_first_hit_times(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> dict[str, float | None]:
    """One reward_group per session (from unit_table, reference-sheet
    source) -> first-hit time, or None if the session has no qualifying hit
    (that session's units are excluded downstream, not silently included
    with an unrestricted window)."""
    out = {}
    session_reward = unit_table.drop_duplicates("session_id").set_index("session_id")["reward_group"]
    for session_id, sess_trials in trial_table.groupby("session_id"):
        if session_id not in session_reward.index:
            continue
        rg = session_reward.loc[session_id]
        out[session_id] = first_hit_time(sess_trials, rg)
    return out


def restrict_spikes(spikes: np.ndarray, t_restrict: float) -> np.ndarray:
    return spikes[spikes >= t_restrict]


def restrict_starts(starts: np.ndarray, t_restrict: float) -> np.ndarray:
    return starts[starts >= t_restrict]
