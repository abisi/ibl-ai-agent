"""045 -- NWB_ks4 reconversion without dlc2kinematics smoothing (user 2026-11-20: NWB_converter utils/dlc_utils
get_dlc_dataframe now loads the DLC h5 with smooth=False): check every staged NWB against the current NWB_ks4 file and
swap it in.

Staged files: Axel_Bisi/NWB_ks4_reconv_20261120/<session>.nwb, written by NWB_converter run_ab.py (personal runner,
`nwb --sessions-file ... --out <staging> --all-days`; haas for 264 sessions, Windows for the 4 large-AVI sessions).
Checks (new vs current NWB_ks4):
- identical: trials (every column), units (id, cluster_id, spike_times, spike_times_index, ccf labels / coordinates,
  bc_label), piezo lick times, acquisition (names; ImageSeries timestamps / external files), intervals tables (names),
  BehavioralEvents (names; DLC-derived event counts are reported, may change slightly);
- DLC BehavioralTimeSeries: same series set, per series same number of samples and identical timestamps; data changed
  only by the removed smoothing (reported: median correlation, max |diff| per series).
Swap only when every 'identical' check passes and the DLC structure matches: old -> Axel_Bisi/NWB_ks4_backup_20261120/,
new -> NWB_ks4. Report: Axel_Bisi/NWB_ks4_reconv_20261120/check_swap.csv.
Run (haas): ~/code/nwb_converter_env/bin/python 045_check_swap_unsmoothed.py [--no-swap] [--sessions ...]
"""
import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

AB = Path(r"M:\analysis\Axel_Bisi") if sys.platform.startswith("win") else Path("/mnt/lsens-analysis/Axel_Bisi")
NWB = AB / "NWB_ks4"
STAGE = AB / "NWB_ks4_reconv_20261120"
BACKUP = AB / "NWB_ks4_backup_20261120"
BTS, BEV = "processing/behavior/BehavioralTimeSeries", "processing/behavior/BehavioralEvents"
UNIT_COLS = ["id", "cluster_id", "spike_times", "spike_times_index", "ccf_atlas_acronym", "ccf_atlas_ap", "ccf_atlas_ml",
             "ccf_atlas_dv", "bc_label"]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9))
    return bool(np.array_equal(a, b))


def check(s):
    old, new = NWB / f"{s}.nwb", STAGE / f"{s}.nwb"
    r = dict(session=s, staged=new.exists(), current=old.exists())
    if not (new.exists() and old.exists()):
        r["verdict"] = "missing file"
        return r
    bad = []
    with h5py.File(old, "r") as fo, h5py.File(new, "r") as fn:
        for k in fo["intervals/trials"]:
            o = fo[f"intervals/trials/{k}"]
            if isinstance(o, h5py.Dataset) and not (k in fn["intervals/trials"] and same(o[()], fn[f"intervals/trials/{k}"][()])):
                bad.append(f"trials/{k}")
        if "units" in fo:
            for k in UNIT_COLS:
                if k in fo["units"] and not (k in fn.get("units", {}) and same(fo[f"units/{k}"][()], fn[f"units/{k}"][()])):
                    bad.append(f"units/{k}")
        p = f"{BEV}/piezo_lick_times/timestamps"
        if p in fo and not (p in fn and same(fo[p][()], fn[p][()])):
            bad.append("piezo")
        if set(fo.get("acquisition", {}).keys()) != set(fn.get("acquisition", {}).keys()):
            bad.append("acquisition names")
        else:
            for k in fo.get("acquisition", {}):
                for sub in ("timestamps", "external_file"):
                    q = f"acquisition/{k}/{sub}"
                    if q in fo and not (q in fn and same(fo[q][()], fn[q][()])):
                        bad.append(q)
        if set(fo.get("intervals", {}).keys()) != set(fn.get("intervals", {}).keys()):
            bad.append("intervals names")
        if set(fo.get(BEV, {}).keys()) != set(fn.get(BEV, {}).keys()):
            bad.append("BehavioralEvents names")
        so, sn = set(fo.get(BTS, {}).keys()), set(fn.get(BTS, {}).keys())
        r["n_series"] = len(so)
        if so != sn:
            bad.append(f"DLC series set (-{sorted(so - sn)[:3]} +{sorted(sn - so)[:3]})")
        cors, maxd, changed = [], [], 0
        for k in sorted(so & sn):
            do, dn = fo[f"{BTS}/{k}/data"], fn[f"{BTS}/{k}/data"]
            if do.shape != dn.shape:
                bad.append(f"DLC {k} shape")
                continue
            to, tn = f"{BTS}/{k}/timestamps", f"{BTS}/{k}/timestamps"
            if to in fo and not (tn in fn and same(fo[to][()], fn[tn][()])):
                bad.append(f"DLC {k} timestamps")
            a, b = np.asarray(do[()], float).ravel(), np.asarray(dn[()], float).ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 10 and not np.array_equal(a[ok], b[ok]):
                changed += 1
                if np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                    cors.append(np.corrcoef(a[ok], b[ok])[0, 1])
                maxd.append(float(np.nanmax(np.abs(a[ok] - b[ok]))))
        r.update(n_series_changed=changed, median_r_changed=float(np.median(cors)) if cors else np.nan,
                 min_r_changed=float(np.min(cors)) if cors else np.nan)
        for k in sorted(set(fo.get(BEV, {}).keys()) & set(fn.get(BEV, {}).keys())):
            if k != "piezo_lick_times" and f"{BEV}/{k}/timestamps" in fo:
                r[f"ev_{k}"] = f"{len(fo[f'{BEV}/{k}/timestamps'])}->{len(fn[f'{BEV}/{k}/timestamps'])}"
    r["differs"] = "; ".join(bad)
    r["verdict"] = "ok" if not bad else "CHECK"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--sessions", nargs="+", default=None)
    a = ap.parse_args()
    ss = a.sessions or sorted(p.stem for p in STAGE.glob("*.nwb"))
    rows = []
    for s in ss:
        r = check(s)
        if r["verdict"] == "ok" and not a.no_swap:
            BACKUP.mkdir(exist_ok=True)
            shutil.move(str(NWB / f"{s}.nwb"), str(BACKUP / f"{s}.nwb"))
            shutil.move(str(STAGE / f"{s}.nwb"), str(NWB / f"{s}.nwb"))
            r["swapped"] = True
        print(s, r["verdict"], r.get("n_series_changed"), "/", r.get("n_series"), "median r", r.get("median_r_changed"),
              r.get("differs", ""), "SWAPPED" if r.get("swapped") else "", flush=True)
        rows.append(r)
    out = STAGE / "check_swap.csv"
    D = pd.DataFrame(rows)
    if out.exists():
        D = pd.concat([pd.read_csv(out), D]).drop_duplicates("session", keep="last")
    D.to_csv(out, index=False)
    print(D.verdict.value_counts().to_string())


if __name__ == "__main__":
    main()
