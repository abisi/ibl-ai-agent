"""186 -- Checks for the Method-1 review of the Figure 4 CD pipeline (user 2026-11-24): what changes under the user's rules.
  1. MH015: where are its tracked stable neurons (areas)?
  2. Fast licks: active whisker hits with a lick < 35 ms after the stimulus (dropped by 179): how many, per cohort?
  3. Active trials, current (keep_active_trials = not passive, perf != 6, early_lick == 0; from the first whisker trial;
     main block only; A1 trim) vs user rule (not passive, perf != 6 only): extra trials and hits / misses per session.
  4. Passive epochs, current (132: longest passive run before the main block / run right after it) vs ephys_utilities rule
     (neural_utils: every passive trial, pre / post split at the session's middle trial; keep_passive_data for mice / units):
     sessions whose pre / post trials differ.
  5. Passive trial counts per modality (< 10 -> modality dropped, session kept for the other).
Output: printed + 186_selection_checks.csv. Run (haas, repo root): .venv/bin/python .../186_selection_checks.py
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
os.chdir(EA.parents[2])
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
import ssl_timeresolved_decoding as T  # noqa: E402

sys.path.insert(0, str(EA.parents[1] / "ssl-paper-figures"))
import shared_passive_cd as S  # noqa: E402
from common import axel_bisi_root  # noqa: E402

sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
from ephys_utilities.helpers import data_utils as DU  # noqa: E402

M132 = importlib.import_module("132_modality_stim_passive_active")
TU = importlib.import_module("tracked_units")
root = resolve_dataset_dir("ssl_ephys")
st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
sess = T.hitmiss_session_list(st)
sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"]) & sess.subject_id.isin(S.learners())]

# 1. MH015
ids = TU.load("stable")
lab = pd.read_parquet(T.AREA_LABELS_PATH, columns=["session_id", "cluster_id", "area_group", "area_acronym_custom"])
for sid in sess[sess.subject_id == "MH015"].session_id:
    u = lab[(lab.session_id == sid) & lab.cluster_id.isin(ids.get(sid, []))].drop_duplicates("cluster_id")
    print(f"\n[1] {sid}: {len(u)} tracked stable neurons")
    print(u.groupby(["area_group", "area_acronym_custom"]).size().to_string())

rows = []
for r in sess.itertuples():
    sid = r.session_id
    t = tt[tt.session_id == sid].sort_values("start_time").reset_index(drop=True)
    trs = M132.session_trials(sid, st, tt, T)
    row = dict(session_id=sid, mouse_id=r.subject_id, cohort=r.reward_group)
    # 2-3. active trials
    rt = (t.lick_time - t.response_window_start_time)
    act_user = t[(t.context != "passive") & (t.perf != 6)]
    wu = act_user[act_user.trial_type == "whisker_trial"]
    row.update(user_W=len(wu), user_hit=int((wu.lick_flag == 1).sum()), user_early_lick=int((act_user.early_lick == 1).sum()))
    if trs is not None:
        a = trs["active"]
        w = a[a.trial_type == "whisker_trial"]
        rtw = (w.lick_time - w.response_window_start_time)
        fast = (w.lick_flag == 1) & (rtw <= 0.035)
        row.update(cur_W=len(w), cur_hit=int((w.lick_flag == 1).sum()), fast_hits=int(fast.sum()),
                   fast_rt_ms=";".join(f"{1000 * v:.0f}" for v in rtw[fast]))
    # 4. passive: 132 vs ephys_utilities (all passive, split at the middle trial)
    p = t[(t.context == "passive") & t.trial_type.isin(["whisker_trial", "auditory_trial"])]
    mid = t.start_time.values[len(t) // 2]
    eu = {"pre": p[p.start_time < mid], "post": p[p.start_time >= mid]}
    for k in ("pre", "post"):
        for m, ty in (("W", "whisker_trial"), ("A", "auditory_trial")):
            row[f"eu_{k}_{m}"] = int((eu[k].trial_type == ty).sum())
            row[f"cur_{k}_{m}"] = int((trs[f"passive_{k}"].trial_type == ty).sum()) if trs is not None else 0
        row[f"eu_{k}_licks"] = int((eu[k].lick_flag == 1).sum())
    rows.append(row)
d = pd.DataFrame(rows)
d["passive_kept_by_keep_passive_data"] = d.mouse_id.isin(DU.keep_passive_data(d[["mouse_id"]].drop_duplicates()).mouse_id)
d.to_csv(EA / "186_selection_checks.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

print("\n[2] fast hits (lick <= 35 ms) in the current active set:")
print(d.groupby("cohort")[["cur_hit", "fast_hits"]].sum().assign(frac=lambda x: x.fast_hits / x.cur_hit).round(3).to_string())
print(d[d.fast_hits > 0][["session_id", "cohort", "cur_hit", "fast_hits", "fast_rt_ms"]].to_string(index=False))

print("\n[3] whisker trials, current vs user rule (not passive, perf != 6):")
d["extra_W"], d["extra_hit"] = d.user_W - d.cur_W, d.user_hit - d.cur_hit
print(d.groupby("cohort")[["cur_W", "user_W", "cur_hit", "user_hit", "user_early_lick"]].sum().to_string())
print(d[(d.extra_W != 0)][["session_id", "cohort", "cur_W", "user_W", "cur_hit", "user_hit", "user_early_lick"]].to_string(index=False))

print("\n[4] passive epochs that differ (current 132 vs ephys_utilities all passive split at mid trial):")
c = [f"{s}_{k}_{m}" for k in ("pre", "post") for m in ("W", "A") for s in ("cur", "eu")]
diff = d[[any(d.loc[i, f"cur_{k}_{m}"] != d.loc[i, f"eu_{k}_{m}"] for k in ("pre", "post") for m in ("W", "A")) for i in d.index]]
print(diff[["session_id", "cohort"] + c].to_string(index=False))

print("\n[5] < 10 passive trials for a modality (ephys_utilities rule):")
low = d[(d[[f"eu_{k}_{m}" for k in ("pre", "post") for m in ("W", "A")]] < 10).any(axis=1)]
print(low[["session_id", "cohort"] + [f"eu_{k}_{m}" for k in ("pre", "post") for m in ("W", "A")]].to_string(index=False))
print("\nlicks in passive trials (eu pre / post):", int(d.eu_pre_licks.sum()), int(d.eu_post_licks.sum()))
print("not kept by keep_passive_data:", d[~d.passive_kept_by_keep_passive_data].session_id.tolist())
