"""005 -- Per-trial raw rate matrices for the whisker-to-auditory similarity (006) and slow-drift (007) analyses (user 2026-10-16:
"Do 3 ... Do 4 as well"). Compute only.
Per session: units and trial selection as ssl-prelick-convergence 105 (081 tested units, stable run; 051 active trials: active context,
perf != 6, warm-up cut, rule A1), all trial types and outcomes (whisker / auditory hits and misses, correct rejections, false alarms).
Rates are raw (no baseline subtraction; downstream scripts subtract), in spikes/s:
  base   pre-trial window [-1.0, -0.015] s before trial start (stimulus onset)
  post   25-ms bins 0 .. 300 ms after stimulus onset (12 bins, left edges POST_E); not truncated at the lick -- the corrected first-lick
         time (rt, s after onset; NaN without lick) is stored so downstream scripts mask bins that end after the lick
Per trial: trial type (W / A / N), lick flag, rt, start time, tau (normalised time within the analysed epoch), whisker-trial index
(whisker trials, hits and misses, started before the trial) and trial index.
Output: tables/005_trial_windows/<session_id>.npz + sessions.csv
Run (haas, prelick environment): PRELICK_REF=sl PRELICK_UNITS=stable python 005_trial_windows.py [--n-proc 25]
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

import common as C

sys.path.insert(0, str(C.PL))
m51 = importlib.import_module("051_roc_prelick")
m81 = importlib.import_module("081_single_unit_classes")
OUT = C.TAB / "005_trial_windows"
BASE = (-1.0, -0.015)
BW = 0.025
POST_E = np.round(np.arange(0.0, 0.300, BW), 3)
TT = {"whisker_trial": "W", "auditory_trial": "A", "no_stim_trial": "N"}


def session(args):
    sid, units = args
    from pynwb import NWBHDF5IO
    try:
        with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
            nwb = io.read()
            U, _ = m51.ru.process_nwb_tables(nwb)
            t, log = m51.active_trials(nwb.trials.to_dataframe(), sid)
    except Exception as e:  # noqa: BLE001
        return dict(session_id=sid, error=repr(e)[:200])
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str)).merge(units, on=["electrode_group", "cluster_id"], how="inner")
    t = t[t.trial_type.isin(list(TT))].reset_index(drop=True)
    if len(U) < 5 or len(t) < 20:
        return dict(session_id=sid, error="too few units / trials")
    st = t.start_time.to_numpy(float)
    lick = (t.lick_flag == 1).to_numpy()
    rt = np.where(lick, (t.lick_time - t.response_window_start_time).to_numpy(float), np.nan)   # corrected first lick, s after onset
    tt = t.trial_type.map(TT).to_numpy()
    lo, hi = log["epoch"]
    tau = (st - lo) / (hi - lo)
    wtrial = np.r_[0, np.cumsum(tt == "W")[:-1]]
    nu, nt = len(U), len(t)
    base = np.empty((nu, nt), np.float32); post = np.empty((nu, nt, len(POST_E)), np.float32)
    for u, s in enumerate(U.spike_times):
        s = np.sort(np.asarray(s))
        base[u] = (np.searchsorted(s, st + BASE[1]) - np.searchsorted(s, st + BASE[0])) / (BASE[1] - BASE[0])
        for k, e in enumerate(POST_E):
            post[u, :, k] = (np.searchsorted(s, st + e + BW) - np.searchsorted(s, st + e)) / BW
    np.savez_compressed(OUT / f"{sid}.npz", electrode_group=U.electrode_group.to_numpy(), cluster_id=U.cluster_id.to_numpy(),
                        area_group=U.area_group.to_numpy(), trial_type=tt, lick=lick, rt=rt, start_time=st, tau=tau, wtrial=wtrial,
                        trial_index=np.arange(nt), base=base, post=post, post_e=POST_E, bw=BW)
    return dict(session_id=sid, n_units=nu, n_trials=nt, n_W=int((tt == "W").sum()), n_WH=int(((tt == "W") & lick).sum()),
                n_A=int((tt == "A").sum()), n_AH=int(((tt == "A") & lick).sum()), n_N=int((tt == "N").sum()), n_FA=int(((tt == "N") & lick).sum()))


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    jobs = [(sid, g[["electrode_group", "cluster_id", "area_group"]]) for sid, g in U0.groupby("session_id")]
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = list(pool.imap_unordered(session, jobs))
    S = pd.DataFrame(res).merge(U0.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]], on="session_id", how="left")
    S.to_csv(OUT / "sessions.csv", index=False)
    print(S.describe().round(1)); print("errors:", S.error.notna().sum() if "error" in S else 0)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    main(ap.parse_args())
