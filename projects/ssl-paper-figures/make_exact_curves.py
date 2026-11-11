"""Exact learning curves (learning_curve_exact, grid HMM, sigma 1, 80 % CI) for every whisker-day session of NWB_ks4,
saved as combined_results_ks4/<mouse>/<behavior>_<day>/learning_curve/<mouse>_<behavior>_<day>_<trial_type>_learning_curve_exact.h5
(columns as in learning_curve_exact.plot_single_mouse_session_learning_curve_exact; p_chance and learning trial on the
whisker row). Trials: active (not passive), perf != 6, early_lick == 0, time-ordered; cohort from the reference sheet
(user, 2026-11-03: analyses on learning curves use the exact curves).

Run on haas (repo root):
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/make_exact_curves.py [--days 0] [--stale]
--stale: only sessions whose NWB is newer than their whisker-trial curve file (reconverted since), log written next to the
full-run log with a _stale_<date> suffix.
"""

import argparse
import glob
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import axel_bisi_root, import_external_repos  # noqa: E402

import_external_repos()
import learning_curve_exact as lce  # noqa: E402

ROOT = axel_bisi_root()
NWB = ROOT / "NWB_ks4"
OUT = ROOT / "combined_results_ks4"
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
SIGMA, CONF_INT = 1.0, 80


def _col(tr, c):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in tr[c][:]])


def one_session(args):
    path, cohort = args
    sid = Path(path).stem
    with h5py.File(path, "r") as f:
        desc = f["session_description"][()].decode()
        tr = f["intervals/trials"]
        st = pd.DataFrame({c: _col(tr, c) for c in ("trial_type", "lick_flag", "start_time", "context", "perf",
                                                    "early_lick")})
    st = st[(st.context != "passive") & (st.perf.astype(int) != 6) & (st.early_lick.astype(int) == 0)].copy()
    st["lick_flag"] = st.lick_flag.astype(int)
    st["start_time"] = st.start_time.astype(float)
    mouse = sid.split("_")[0]
    rg = {"R+": 1, "R-": 0}.get(cohort.get(mouse))
    if rg is None:
        return sid, desc, "no R+/R- cohort in the sheet"
    st["reward_group"] = rg
    res = lce.session_learning_curves_exact(st, SIGMA, CONF_INT)
    if "whisker_trial" not in res:
        return sid, desc, "no whisker trials"
    behavior, day = desc.rsplit("_", 1)
    day = int(day)
    out_dir = OUT / mouse / f"{behavior}_{day}" / "learning_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tt, rec in res.items():
        row = dict(mouse_id=mouse, session_id=sid, reward_group=rg, behavior=behavior, day=day, trial_type=tt,
                   outcomes=rec["outcomes"], start_time=rec["start_time"], p_samples=None, p_mean=rec["p_mean"],
                   p_low=rec["p_low"], p_high=rec["p_high"], ci_level=CONF_INT, p_chance=rec.get("p_chance"),
                   learning_trial=rec.get("learning_trial", np.nan), mouse_cat=rec.get("mouse_cat"),
                   method="exact_grid_hmm", sigma=SIGMA,
                   trials="active, perf != 6, early_lick == 0")
        pd.DataFrame([row]).to_hdf(out_dir / f"{mouse}_{behavior}_{day}_{tt}_learning_curve_exact.h5", key="df", mode="w")
    w = res["whisker_trial"]
    return sid, desc, f"ok: {len(w['outcomes'])} whisker trials, learning trial {w.get('learning_trial')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", type=int, help="whisker days to process (default all)")
    ap.add_argument("--stale", action="store_true", help="only sessions whose NWB is newer than their curve file")
    a = ap.parse_args()
    sheet = pd.read_excel(SHEET).drop_duplicates("mouse_id")
    cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
    jobs = []
    for p in sorted(glob.glob(str(NWB / "*.nwb"))):
        if "_backup_" in p:
            continue
        with h5py.File(p, "r") as f:
            desc = f["session_description"][()].decode()
        parts = desc.rsplit("_", 1)
        if parts[0] != "whisker":
            continue
        if a.days is not None and int(parts[1]) not in a.days:
            continue
        if a.stale:
            mouse = Path(p).stem.split("_")[0]
            h5 = OUT / mouse / desc / "learning_curve" / f"{mouse}_{desc}_whisker_trial_learning_curve_exact.h5"
            if h5.exists() and h5.stat().st_mtime > Path(p).stat().st_mtime:
                continue
        jobs.append((p, cohort))
    print(f"{len(jobs)} sessions")
    with Pool(min(24, max(len(jobs), 1))) as pool:
        res = pool.map(one_session, jobs)
    log = pd.DataFrame(res, columns=["session_id", "description", "status"])
    suffix = f"_stale_{pd.Timestamp.now():%Y%m%d}" if a.stale else ""
    log.to_csv(OUT / f"_exact_learning_curves_log{suffix}.csv", index=False)
    print(log.status.str.split(":").str[0].value_counts().to_string())
    print(log[~log.status.str.startswith("ok")].to_string(index=False))


if __name__ == "__main__":
    main()
