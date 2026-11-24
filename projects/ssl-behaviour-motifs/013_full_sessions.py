"""013 -- session list of the full run: every whisker day-0 session in NWB_ks4 with DLC in the NWB, with exclusions.

Inclusion: NWB_ks4 file (no *_backup*), session_description == whisker_0; mouse in the reference sheet (share_internal)
with exclude == 0 and reward_group R+ or R- (cohort per mouse from the sheet; R+proba dropped); learner label from
learning_category (good / moderate = learner, bad = non-learner, missing = unknown); both DLC cameras in the NWB
(dlc_helpers.check_dlc) without frame / timestamp flags; side video not in data_utils.VIDEO_EXCLUDED_SESSIONS; known
behaviour setup (px -> mm calibration); >= 30 kept whisker trials.
Output: tables/full_sessions.csv (included), tables/full_exclusions.csv (session, reason).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import ssl_dlc_calibration as cal  # noqa: E402
from ephys_utilities.helpers import dlc_helpers  # noqa: E402


def check(f):
    sid = f.stem
    try:
        with h5py.File(f, "r") as h:
            if h["session_description"][()].decode() != "whisker_0":
                return None
    except Exception as e:  # noqa: BLE001
        return dict(session_id=sid, reason=f"NWB unreadable: {e}")
    r = dict(session_id=sid, mouse_id=sid.split("_")[0])
    try:
        from ephys_utilities.helpers.data_utils import VIDEO_EXCLUDED_SESSIONS
    except ImportError:
        VIDEO_EXCLUDED_SESSIONS = {}
    if sid in VIDEO_EXCLUDED_SESSIONS:
        return dict(r, reason=f"VIDEO_EXCLUDED_SESSIONS: {VIDEO_EXCLUDED_SESSIONS[sid]}")
    try:
        c = dlc_helpers.check_dlc(str(f))
    except Exception as e:  # noqa: BLE001
        return dict(r, reason=f"check_dlc failed: {e}")
    if len(c) == 0 or set(c.view) != {"side", "top"}:
        return dict(r, reason=f"DLC views in NWB: {sorted(set(c.view)) if len(c) else 'none'}")
    fl = ";".join(str(x) for x in c["flags"] if str(x))
    if fl:
        return dict(r, reason=f"check_dlc flags: {fl}")
    try:
        r["setup"] = cal.setup_of(sid)
    except KeyError:
        return dict(r, reason="behaviour setup unknown (no px -> mm calibration)")
    t = L.load_trials(sid)
    k = t[t.keep]
    r["n_whisker"] = int((k.stim == "whisker").sum())
    if r["n_whisker"] < 30:
        return dict(r, reason=f"{r['n_whisker']} kept whisker trials (< 30)")
    r["p_lick_whisker"] = float(k[k.stim == "whisker"].lick_flag.mean())
    r["rate_side"] = float(c.loc[c.view == "side", "rate"].iloc[0])
    r["rate_top"] = float(c.loc[c.view == "top", "rate"].iloc[0])
    r["top_h5"] = str(L.dlc_h5(sid, "top") or "")
    r["reason"] = ""
    return r


def main():
    sheet = L.reference_sheet()
    files = [f for f in sorted(L.NWB_DIR.glob("*.nwb")) if "_backup" not in f.name]
    with ProcessPoolExecutor(24) as ex:  # processes: check_dlc is CPU / GIL bound (threads took > 1 h)
        rows = [r for r in ex.map(check, files, chunksize=4) if r is not None]
    d = pd.DataFrame(rows)
    d["mouse_id"] = d.session_id.str.split("_").str[0]
    d = d.merge(sheet[["mouse_id", "exclude", "reward_group", "learning_category"]], on="mouse_id", how="left")
    d.loc[d.reward_group.isna() & (d.reason == ""), "reason"] = "mouse not in the reference sheet"
    d.loc[(d.exclude == 1) & (d.reason == ""), "reason"] = "exclude == 1 in the reference sheet"
    d.loc[~d.reward_group.isin(["R+", "R-"]) & (d.reason == ""), "reason"] = "reward_group not R+ / R- (" + d.reward_group.astype(str) + ")"
    d["learner"] = np.select([d.learning_category.isin(["good", "moderate"]), d.learning_category.eq("bad")],
                             ["learner", "non-learner"], "unknown")
    inc = d[d.reason == ""].drop(columns=["reason"]).sort_values(["reward_group", "session_id"])
    exc = d[d.reason != ""][["session_id", "mouse_id", "reward_group", "reason"]]
    inc.to_csv(L.TABLES / "full_sessions.csv", index=False)
    exc.to_csv(L.TABLES / "full_exclusions.csv", index=False)
    print(inc.groupby(["reward_group", "learner"]).size())
    print(exc.reason.str.split(":").str[0].value_counts())
    print("one session per mouse:", inc.mouse_id.is_unique, "top spout h5 missing:", (inc.top_h5 == "").sum())


if __name__ == "__main__":
    main()
