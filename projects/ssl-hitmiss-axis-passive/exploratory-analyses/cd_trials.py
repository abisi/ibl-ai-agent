"""Trial selection for the Figure 4 hit - miss CD pipeline (179, 183), user decisions 2026-11-24 (Method-1 review):

Active trials (axis): ephys_utilities data_utils.keep_active_trials (context != passive, perf != 6 -- perf 6 = excluded trial,
  e.g. particle / whisker stimulator off: ALWAYS removed --, early_lick == 0), whisker and auditory trials; nothing else is
  removed (no first-whisker-trial cut, no main-block restriction, no fast-lick rule). Hit = whisker trial with lick_flag 1.
  Known: 6 day-0 trials sit in a passive ITI sequence but are labelled active (AB128_20240829 x2, MH028_20250501 x4); they
  stay active (ephys_utilities rule).
  Trimming of the session end (SSL_179_TRIM, compared in the review): "none" (default), "a1" (terminal disengagement rule A1,
  ssl_timeresolved_decoding.detect_terminal_disengagement), "end10" (last 10 % of the active epoch duration removed).
Passive trials: ephys_utilities data_utils.get_passive_epochs (added 2026-11-24 at the user's request; passive context,
  whisker / auditory, lick_flag == 0 as get_passive_conditions_fast); mice via data_utils.keep_passive_data (units:
  keep_passive_units at the unit level, in the caller). Epochs (SSL_179_PASSIVE):
  "mid" = ephys_utilities neural_utils rule (every passive trial; pre / post split at the session's middle trial);
  "blocks" = the passive block before the first active trial / after the last active trial of the main block (excludes short
  passive runs interleaved with the first active trials: AB120, AB122, AB123, AB127, AB128, AB132, AB133, AB139, AB159, MH010,
  MH022, MH025, MH029, MH031, MH032, MH039, MH062). Default "mid" (ephys_utilities) until the user decides.
  A modality is kept if it has >= MIN_PASSIVE (5) trials in BOTH pre and post (user: 10 -> 5; AB124 post auditory = 7).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRIM = os.environ.get("SSL_179_TRIM", "none")
PASSIVE_RULE = os.environ.get("SSL_179_PASSIVE", "mid")
MIN_PASSIVE = 5
WA = ["whisker_trial", "auditory_trial"]


def _du():
    # NAS checkout (holds the uncommitted AB144 re-inclusion), not the haas ~/code copy
    for p in (Path("/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities"), Path("M:/analysis/Axel_Bisi/Github/ephys_utilities")):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            break
    from ephys_utilities.helpers import data_utils as DU
    return DU


def session_trials(sid: str, tt: pd.DataFrame, T, trim: str = TRIM, passive_rule: str = PASSIVE_RULE):
    """-> dict(passive_pre, active, passive_post, passive_ok={W: bool, A: bool}, info) or None"""
    DU = _du()
    t = tt[tt.session_id == sid].sort_values("start_time").reset_index(drop=True)
    if not len(DU.keep_passive_data(pd.DataFrame({"mouse_id": [sid.split("_")[0]]}))):
        return None                                                    # mouse without passive pre + post
    act = DU.keep_active_trials(t)
    act = act[act.trial_type.isin(WA)].sort_values("start_time").reset_index(drop=True)
    if not len(act):
        return None
    n0 = len(act)
    if trim == "a1":
        dis = T.detect_terminal_disengagement(sid, tt)
        if dis["disengaged"]:
            act = act[act.start_time < dis["t_cut"]]
    elif trim == "end10":
        t0, t1 = act.start_time.min(), act.start_time.max()
        act = act[act.start_time < t1 - 0.10 * (t1 - t0)]
    elif trim != "none":
        raise ValueError(trim)
    pas = t[(t.context.astype(str) == "passive") & t.trial_type.isin(WA)]
    n_lick = int((pas.lick_flag == 1).sum())
    pre, post = DU.get_passive_epochs(t, rule=passive_rule, drop_licks=True)   # ephys_utilities (added 2026-11-24)
    ok = {m: min((pre.trial_type == ty).sum(), (post.trial_type == ty).sum()) >= MIN_PASSIVE
          for m, ty in (("W", "whisker_trial"), ("A", "auditory_trial"))}
    if not any(ok.values()):
        return None
    info = dict(trim=trim, passive_rule=passive_rule, n_active_before_trim=n0, n_active=len(act), n_passive_licked=n_lick)
    return {"passive_pre": pre.reset_index(drop=True), "active": act.reset_index(drop=True),
            "passive_post": post.reset_index(drop=True), "passive_ok": ok, "info": info}
