"""001 -- Response latency (half-time to peak) of whisker- and auditory-responsive units, all sessions pooled.

Spikes: KS4 NWB, whisker-artefact-corrected (roc_utils_new.correct_neuron_spike_train: -10..+5 ms around every whisker
onset replaced by a Poisson train at the pre-onset rate; seeded per session).
Trials: active, perf != 6, auditory warm-up cut (1 trial before the first whisker trial kept), rule A1 tail trim; all
whisker trials (whisker latency) and all auditory trials (auditory latency), lick or not.
Responsive units: significant in the rate-based ROC whisker_active / auditory_active (045 roc_long; response window
5-35 ms vs the pre-trial baseline), good + mua.
PSTH: 1-ms bins, -100..+200 ms from stimulus onset, mean over trials, minus the PSTH's own pre-stimulus mean in
-100..-10 ms (local baseline, before the whisker-artefact window; the pre-trial baseline [-1, -0.015] s is also stored);
Gaussian smoothing sigma = 2 ms. Response sign = sign of the ROC
selectivity (excited +, inhibited -); the signed response s * r(t) is used.
Peak: maximum of s * r(t) in 5..100 ms. Half-time to peak (latency): the last upward crossing of half the peak before
the peak, after stimulus onset (whisker: after +5 ms, the end of the artefact-replaced window; linear interpolation between 1-ms bins). NaN if the peak is not above 0 or if s * r(t)
stays above half the peak from stimulus onset to the peak (no crossing after onset).
Output: combined_results_ks4/_sensory_spatial_maps/unit_latency.parquet (keys mouse_id, session_id, electrode_group,
cluster_id; latency_whisker_ms, latency_auditory_ms, peak_*, t_peak_*, sign_*, n_trials_*) + provenance.
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
ROC = RES / "_roc_stage_analysis"
OUT = RES / "_sensory_spatial_maps"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
T0, T1, BIN = -0.100, 0.200, 0.001
BASE = (-1.0, -0.015)
LOCAL_BASE = (-100.0, -10.0)       # ms
SEARCH_START_MS = {"whisker": 5.0, "auditory": 0.0}   # whisker: spikes up to +5 ms are artefact-replaced
SIGMA_MS = 2.0
PEAK_WIN = (0.005, 0.100)
MEASURES = {"whisker": ("whisker_active", "whisker_trial"), "auditory": ("auditory_active", "auditory_trial")}


def stim_trials(trials):
    t = trials.sort_values("start_time").reset_index(drop=True)
    ctx = t["context"].astype(str)
    # unlabelled trials in a session with context labels (MH062_20260113_125836): passive trials are perf == 6 with a
    # fixed ITI, every other trial is active (user, 2026-10-04)
    if (ctx == "nan").any() and ctx.isin(["active", "passive"]).any():
        ctx = ctx.where(ctx != "nan", np.where(t["perf"] == 6, "passive", "active"))
    if (ctx == "active").any():
        t = t[ctx == "active"].reset_index(drop=True)
    if "perf" in t:
        t = t[t["perf"] != 6].reset_index(drop=True)
    wi = np.where(t["trial_type"].to_numpy() == "whisker_trial")[0]
    if len(wi):
        t = t.iloc[max(0, wi[0] - 1):].reset_index(drop=True)
    licked = np.where(t["lick_flag"].to_numpy() == 1)[0]
    if len(licked):
        tail = t.iloc[licked[-1] + 1:]
        if (tail.trial_type == "whisker_trial").sum() >= 5 and (tail.trial_type == "auditory_trial").sum() >= 1:
            t = t.iloc[:licked[-1] + 1].reset_index(drop=True)
    return t


def half_time(sr, tms, t_start=0.0):
    """sr: signed, smoothed response (Hz) on 1-ms grid tms; returns (latency ms, peak, t_peak ms)"""
    w = (tms >= PEAK_WIN[0] * 1000) & (tms <= PEAK_WIN[1] * 1000)
    ip = np.where(w)[0][np.argmax(sr[w])]
    pk = sr[ip]
    if not pk > 0:
        return np.nan, pk, np.nan
    i0 = np.searchsorted(tms, t_start)
    below = np.where(sr[i0:ip + 1] < 0.5 * pk)[0]
    if not len(below):
        return np.nan, pk, float(tms[ip])
    k = i0 + below[-1] + 1                                    # first bin of the last run above half-peak
    f = (0.5 * pk - sr[k - 1]) / (sr[k] - sr[k - 1])
    return float(tms[k - 1] + f * (tms[k] - tms[k - 1])), pk, float(tms[ip])


def run_session(sid):
    from pynwb import NWBHDF5IO
    from scipy.ndimage import gaussian_filter1d
    from roc_analysis import roc_utils_new as ru
    R = pd.read_parquet(ROC / "roc_long.parquet", filters=[("session_id", "==", sid)])
    R = R[R.analysis_type.isin([m[0] for m in MEASURES.values()]) & (R.sig == 1)]
    U = pd.read_parquet(ROC / "units.parquet", filters=[("session_id", "==", sid)])
    U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    R = R.merge(U[KEYS], on=KEYS)
    if not len(R):
        return pd.DataFrame()
    with NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
        trials = nwb.trials.to_dataframe()               # raw table (process_nwb_tables relabels `context`), as 051
    units["cluster_id"] = units["cluster_id"].astype(str)
    onsets = trials.loc[trials["whisker_stim"] == 1, "start_time"].to_numpy()
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    t = stim_trials(trials)
    edges = np.round(np.arange(T0, T1 + BIN / 2, BIN), 6)
    tms = 1000 * (edges[:-1] + BIN / 2)
    want = set(map(tuple, R[["electrode_group", "cluster_id"]].drop_duplicates().to_numpy()))
    out = []
    for u in units.itertuples():
        if (u.electrode_group, u.cluster_id) not in want:
            continue
        st = ru.correct_neuron_spike_train(np.asarray(u.spike_times), onsets, rng)
        row = dict(mouse_id=U.mouse_id.iloc[0], session_id=sid, electrode_group=u.electrode_group, cluster_id=u.cluster_id)
        for mod, (atype, ttype) in MEASURES.items():
            r = R[(R.electrode_group == u.electrode_group) & (R.cluster_id == u.cluster_id) & (R.analysis_type == atype)]
            if not len(r):
                continue
            s0 = t.start_time[t.trial_type == ttype].to_numpy()
            if len(s0) < 3:
                continue
            sign = 1.0 if r.sel.iloc[0] > 0 else -1.0
            base = ((np.searchsorted(st, s0 + BASE[1]) - np.searchsorted(st, s0 + BASE[0])) / (BASE[1] - BASE[0])).mean()
            cnt = np.zeros(len(edges) - 1)
            for s in s0:
                a, b = np.searchsorted(st, [s + T0, s + T1])
                cnt += np.histogram(st[a:b] - s, edges)[0]
            psth = cnt / len(s0) / BIN
            psth = psth - psth[(tms >= LOCAL_BASE[0]) & (tms < LOCAL_BASE[1])].mean()
            sr = sign * gaussian_filter1d(psth, SIGMA_MS)
            lat, pk, tp = half_time(sr, tms, SEARCH_START_MS[mod])
            row.update({f"latency_{mod}_ms": lat, f"peak_{mod}_hz": pk, f"t_peak_{mod}_ms": tp, f"sign_{mod}": sign,
                        f"n_trials_{mod}": len(s0), f"pretrial_baseline_{mod}_hz": base})
        out.append(row)
    return pd.DataFrame(out)


def main(a):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sids = sorted(pd.read_parquet(ROC / "units.parquet", columns=["session_id"]).session_id.unique())
    if a.sessions:
        sids = a.sessions.split(",")
    res = []
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        for k, d in enumerate(pool.imap_unordered(run_session, sids)):
            res.append(d)
            print(f"{time.time() - t0:6.0f}s {k + 1}/{len(sids)}", flush=True)
    L = pd.concat(res, ignore_index=True)
    f = OUT / ("unit_latency.parquet" if not a.sessions else "unit_latency_test.parquet")
    L.to_parquet(f.with_suffix(".tmp"))
    os.replace(f.with_suffix(".tmp"), f)
    json.dump(dict(script="001_unit_latency.py", psth=f"{T0}..{T1} s, {BIN * 1000:.0f}-ms bins", baseline_ms=LOCAL_BASE,
                   smoothing_sigma_ms=SIGMA_MS, peak_window_s=PEAK_WIN, latency="half-time to peak (last upward crossing of 50 % of peak before the peak, after onset)",
                   responsive="ROC whisker_active / auditory_active significant (045 roc_long), good+mua",
                   spikes="NWB_ks4, whisker-artefact Poisson correction, seeded per session",
                   trials="active, perf!=6, warm-up cut, A1; all whisker / all auditory trials", n_sessions=len(sids),
                   date=time.strftime("%Y-%m-%d %H:%M")), open(OUT / "provenance_001.json", "w"), indent=1)
    for mod in MEASURES:
        c = f"latency_{mod}_ms"
        if c in L:
            print(mod, "units", L[c].notna().sum(), "median latency", round(L[c].median(), 1), "ms")
    print("ALL DONE", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--sessions", default=None)
    main(ap.parse_args())
