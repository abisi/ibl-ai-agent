"""Shared Part III unit set (user 2026-10-05: "make sure the same stable units are used throughout"): ONE list of tracked units
per session, read by 133, 134, 135, 140 and 146 so that every Part III analysis uses exactly the same units.

tracked stable unit = 137 `stable` (coverage >= 0.9, presence >= 0.5, drift joint test passed; skills/ssl-valid-data "Unit sets")
AND firing >= MIN_RATE Hz in every segment below (132.session_trials trials; span = first trial - 1 s .. last trial + 1 s):
  passive pre; passive post; and the active epoch split in two at each of the three cut points used by the analyses:
  the middle active trial (135 / 146 active halves), the middle active whisker trial (140 midpoint split) and the
  (H//2 + 1)-th whisker hit (139 / 140 hit-median split).
tracked good unit = tracked stable AND quality_label good (good = good AND stable).
The scripts take the list as is (no further rate filter of their own), so the unit set per session is identical everywhere;
sessions can still differ between analyses through their own trial criteria (passive epochs, hit / miss counts).
Table: combined_results_ks4/<slug>/tables/137b_tracked_units.parquet (built by 137b_tracked_units.py).
"""

from __future__ import annotations

import os

import numpy as np

MIN_RATE = 0.5
# SSL_STAGE=expert: the same list for expert sessions (137b_tracked_units_expert.parquet), read by the scripts run with that stage
STAGE = os.environ.get("SSL_STAGE", "learning")
STAGE_SFX = "" if STAGE == "learning" else f"_{STAGE}"
TABLE_NAME = f"137b_tracked_units{STAGE_SFX}.parquet"


def table_path():
    from axel_bisi_paths import axel_bisi_root
    return axel_bisi_root() / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / TABLE_NAME


def segments(trs):
    """rate segments (start, end) in s for the session_trials dict"""
    pre, act, post = trs["passive_pre"], trs["active"].sort_values("start_time"), trs["passive_post"]
    span = lambda d: (d.start_time.min() - 1.0, d.start_time.max() + 1.0)
    a0, a1 = span(act)
    cuts = [act.start_time.iloc[len(act) // 2]]
    w = act[act.trial_type == "whisker_trial"]
    if len(w):
        cuts.append(w.start_time.iloc[len(w) // 2])
    hits = w[w.lick_flag == 1].start_time.to_numpy()
    if len(hits) >= 2:
        cuts.append(hits[len(hits) // 2])
    segs = [span(pre), span(post)]
    for t in cuts:
        segs += [(a0, t), (t, a1)]
    return segs


def is_tracked(sp, segs, min_rate=MIN_RATE):
    return len(sp) > 0 and all((np.searchsorted(sp, b) - np.searchsorted(sp, a)) / max(b - a, 1e-6) >= min_rate for a, b in segs)


def load(unit_set="stable"):
    """dict session_id -> sorted cluster_id array of the tracked units of `unit_set` ('stable' or 'good')"""
    import pandas as pd
    d = pd.read_parquet(table_path())
    d = d[d[unit_set]]
    return {s: np.sort(g.cluster_id.to_numpy()) for s, g in d.groupby("session_id")}


def load_table():
    import pandas as pd
    return pd.read_parquet(table_path())
