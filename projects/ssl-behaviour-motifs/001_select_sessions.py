"""001 -- choose the 8 day-0 whisker pilot sessions (4 R+, 4 R-; within each cohort 2 learners + 2 non-learners).

Candidates: NWB_ks4 whisker_0 sessions of mice with exclude == 0, reward_group R+ / R- and a learning_category in the
reference sheet (share_internal), merged-model filtered h5 for side AND top on the NAS (= Kuma copies), same behaviour
setup (camera geometry), no dlc_helpers.check_dlc flag.
Score per session (from the NWB DLC likelihoods, every 20th frame): median likelihood of the modelled keypoints per
view, fraction of frames < 0.5 for the static points; geometry = distance (px) of the median static-point positions from
the candidate median per view. Within each cohort x learner cell pick the best-scoring AB and MH session (mix of
owners), filling with the next best if one owner is missing.
Output: tables/001_candidates.csv, tables/001_selected_sessions.csv (selection + rationale), provenance/001.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import ssl_dlc_calibration as cal  # noqa: E402
from ephys_utilities.helpers import dlc_helpers  # noqa: E402

KUMA_LIST = L.TABLES / "000_kuma_dlc_sessions.txt"  # "session n_side n_top" listed on Kuma (inference_cspnext_m_330)
SIDE_LIK = ["nose_tip", "nose_base", "jaw", "jaw_ref", "spout"]
TOP_LIK = ["top_nose_tip", "top_nose_base", "whisker_base", "whisker_tip", "top_particle"]


def describe(sid):
    out = {"session_id": sid}
    f = L.NWB_DIR / f"{sid}.nwb"
    with h5py.File(f, "r") as h:
        out["session_description"] = h["session_description"][()].decode()
        if out["session_description"] != "whisker_0":
            return out
        g = h[dlc_helpers.TS_PATH]
        for kp in SIDE_LIK + TOP_LIK:
            n = f"{kp}_likelihood"
            if n in g:
                x = g[n]["data"][::20]
                out[f"lik_{kp}"] = float(np.nanmedian(x))
                out[f"low_{kp}"] = float(np.mean(x < 0.5))
            for c in ("x", "y"):
                if f"{kp}_{c}" in g:
                    out[f"pos_{kp}_{c}"] = float(np.nanmedian(g[f"{kp}_{c}"]["data"][::200]))
    for v in L.VIEWS:
        out[f"h5_{v}"] = str(L.dlc_h5(sid, v) or "")
    try:
        chk = dlc_helpers.check_dlc(str(f))
        out["dlc_flags"] = ";".join(str(x) for x in chk["flags"] if str(x))
        out["rate_side"] = float(chk.loc[chk.view == "side", "rate"].iloc[0])
        out["rate_top"] = float(chk.loc[chk.view == "top", "rate"].iloc[0])
    except Exception as e:  # noqa: BLE001
        out["dlc_flags"] = f"check_dlc failed: {e}"
    try:
        out["setup"] = cal.setup_of(sid)
    except KeyError:
        out["setup"] = np.nan
    t = L.load_trials(sid)
    k = t[t.keep]
    out["n_trials_kept"] = len(k)
    out["n_whisker"] = int((k.stim == "whisker").sum())
    out["p_lick_whisker"] = float(k.loc[k.stim == "whisker", "lick_flag"].mean())
    out["p_lick_no_stim"] = float(k.loc[k.stim == "no_stim", "lick_flag"].mean())
    return out


def main():
    sheet = L.reference_sheet()
    sheet = sheet[(sheet.exclude == 0) & sheet.reward_group.isin(["R+", "R-"]) & sheet.learning_category.notna()]
    kuma = pd.read_csv(KUMA_LIST, sep=" ", names=["session_id", "n_side", "n_top"])
    kuma = kuma[(kuma.n_side == 1) & (kuma.n_top == 1)]
    cand = [s for s in kuma.session_id if s.split("_")[0] in set(sheet.mouse_id) and (L.NWB_DIR / f"{s}.nwb").exists()]
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(describe, cand))
    d = pd.DataFrame(rows)
    d = d[d.session_description == "whisker_0"].copy()
    d["mouse_id"] = d.session_id.str.split("_").str[0]
    d = d.merge(sheet[["mouse_id", "reward_group", "learning_category"]], on="mouse_id", how="left")
    d["learner"] = np.where(d.learning_category.isin(["good", "moderate"]), "learner", "non-learner")
    d["owner"] = d.mouse_id.str[:2]
    d["ok_files"] = (d.h5_side != "") & (d.h5_top != "")
    d["ok_flags"] = d.dlc_flags.fillna("") == ""
    d["lik_side"] = d[[f"lik_{k}" for k in SIDE_LIK if f"lik_{k}" in d]].median(axis=1)
    d["lik_top"] = d[[f"lik_{k}" for k in TOP_LIK if f"lik_{k}" in d]].median(axis=1)
    d["low_static"] = d[[c for c in ("low_spout", "low_jaw_ref", "low_nose_base", "low_top_nose_base") if c in d]].max(axis=1)
    setup_mode = d.setup.mode().iloc[0]
    d["ok_setup"] = d.setup == setup_mode
    base = d[d.ok_files & d.ok_flags & d.ok_setup]
    geo = {}
    for v, pts in (("side", ["spout", "nose_base", "jaw_ref"]), ("top", ["top_nose_base", "whisker_base"])):
        cols = [f"pos_{p}_{c}" for p in pts for c in "xy"]
        med = base[cols].median()
        geo[v] = np.sqrt(((d[cols] - med) ** 2).values.reshape(len(d), -1, 2).sum(-1)).mean(1)
        d[f"geo_dist_{v}_px"] = geo[v]
    d["geo_dist"] = d[["geo_dist_side_px", "geo_dist_top_px"]].max(axis=1)
    d["ok_geometry"] = d.geo_dist <= 40  # px: same camera framing (static points within ~1.3 mm)
    d["score"] = d.lik_side + d.lik_top - d.low_static - d.geo_dist / 200
    d["eligible"] = d.ok_files & d.ok_flags & d.ok_setup & d.ok_geometry & (d.n_whisker >= 50)
    d.to_csv(L.TABLES / "001_candidates.csv", index=False)

    sel = []
    for (rg, ln), g in d[d.eligible].groupby(["reward_group", "learner"]):
        g = g.sort_values("score", ascending=False).drop_duplicates("mouse_id")
        pick = []
        for owner in ("AB", "MH"):
            go = g[g.owner == owner]
            if len(go):
                pick.append(go.index[0])
        for i in g.index:
            if len(pick) >= 2:
                break
            if i not in pick:
                pick.append(i)
        for rank, i in enumerate(pick[:2]):
            r = d.loc[i].copy()
            r["rationale"] = (f"{rg} {ln} (sheet learning_category={r.learning_category}); best score in cell for owner {r.owner}; "
                              f"setup {int(r.setup)}; median likelihood side {r.lik_side:.2f} / top {r.lik_top:.2f}; "
                              f"static points <0.5 on {100 * r.low_static:.1f}% frames; geometry offset {r.geo_dist:.0f} px; "
                              f"{len(g)} eligible sessions in cell")
            sel.append(r)
    sel = pd.DataFrame(sel)
    keep = ["session_id", "mouse_id", "reward_group", "learning_category", "learner", "owner", "setup", "n_trials_kept",
            "n_whisker", "p_lick_whisker", "p_lick_no_stim", "rate_side", "rate_top", "lik_side", "lik_top", "low_static",
            "geo_dist_side_px", "geo_dist_top_px", "score", "h5_side", "h5_top", "rationale"]
    sel[keep].to_csv(L.TABLES / "001_selected_sessions.csv", index=False)
    print(d.groupby(["reward_group", "learner"]).eligible.agg(["sum", "size"]))
    print(sel[keep[:12] + ["lik_side", "lik_top", "geo_dist_side_px", "geo_dist_top_px"]].to_string())
    json.dump({"script": "001_select_sessions.py", "n_candidates": int(len(d)), "n_eligible": int(d.eligible.sum()),
               "setup": int(setup_mode), "reference_sheet": "share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx",
               "learner_rule": "learning_category good/moderate = learner, bad = non-learner", "versions": L.versions()},
              open(L.RES / "provenance" / "001.json", "w"), indent=1)


if __name__ == "__main__":
    main()
