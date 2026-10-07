"""077 -- Per-session quiet-window rates and behavioural performance (user review 2026-10-07), one pass over the NWB files.

Trials: the analysed task epoch of 051 (051.active_trials: active context, perf != 6, warm-up cut, rule A1), all trial
types and outcomes.
Quiet windows (user: "-200 to -100 ms before all start_times"; the spontaneous activity outside licking, raw rates, no
  baseline subtraction): spikes in [start_time - 0.2 s, start_time - 0.1 s) / 0.1 s for every trial of the epoch,
  artefact-corrected spike trains of 051 (roc_utils_new.process_nwb_tables, same unit order as the 051 event files).
  Windows with a piezo lick or a corrected trial first lick within [-1.0, 0] s of the window end are flagged
  (lick_free = False) so that analyses can restrict to lick-free windows.
Performance on the same trials: whisker hit rate (lick_flag on whisker trials), auditory hit rate, false-alarm rate
  (lick_flag on no-stim trials), d' = z(whisker hit rate) - z(false-alarm rate) with rates clipped to [1/(2n), 1 - 1/(2n)].
Output: per session <mouse>/whisker_<day>/roc_analysis/<mouse>_quiet_windows.npz (rates units x windows, raw; start
  times; lick_free; electrode_group, cluster_id) and across_days/<ref>/extras/performance_sessions.csv
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
OUT = m51.OUTROOT / "extras"
QW = (-0.2, -0.1)
LICK_GUARD = 1.0


def rate(t, n):
    return float(np.clip(t, 1 / (2 * n), 1 - 1 / (2 * n))) if n else np.nan


def session(args):
    sid, mouse, out_dir = args
    from pynwb import NWBHDF5IO
    try:
        with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
            nwb = io.read()
            units, _ = m51.ru.process_nwb_tables(nwb)
            trials_raw = nwb.trials.to_dataframe()
            t, log = m51.active_trials(trials_raw, sid)
            piezo = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                                       .time_series["piezo_lick_times"].data[:], float))
    except Exception as e:                                              # noqa: BLE001
        return dict(session_id=sid, error=str(e)[:200])
    st = t.start_time.to_numpy()
    lk = t.lick_flag.to_numpy() == 1
    fl = (t.start_time + t.lick_time - t.response_window_start_time).to_numpy()
    licks = np.sort(np.r_[piezo, fl[lk & np.isfinite(fl)]])
    we = st + QW[1]
    lick_free = np.searchsorted(licks, we) == np.searchsorted(licks, we - LICK_GUARD)
    R = np.zeros((len(units), len(st)), np.float32)
    for u, s in enumerate(units.spike_times):
        s = np.sort(np.asarray(s))
        R[u] = (np.searchsorted(s, st + QW[1]) - np.searchsorted(s, st + QW[0])) / (QW[1] - QW[0])
    np.savez_compressed(out_dir / f"{mouse}_quiet_windows.npz", rates=R, start_time=st, lick_free=lick_free,
                        trial_type=t.trial_type.to_numpy(str), electrode_group=units.electrode_group.astype(str).to_numpy(),
                        cluster_id=units.cluster_id.astype(str).to_numpy())
    perf = dict(session_id=sid, n_trials=len(t), n_quiet_lick_free=int(lick_free.sum()))
    for tt, key in (("whisker_trial", "wh"), ("auditory_trial", "ah"), ("no_stim_trial", "fa")):
        m = (t.trial_type == tt).to_numpy()
        perf[f"n_{key}"] = int(m.sum())
        perf[f"rate_{key}"] = float(lk[m].mean()) if m.any() else np.nan
    perf["dprime_wh_fa"] = (norm.ppf(rate(perf["rate_wh"], perf["n_wh"])) - norm.ppf(rate(perf["rate_fa"], perf["n_fa"]))
                            if perf["n_wh"] and perf["n_fa"] else np.nan)
    return perf


def main(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet", columns=["session_id", "cohort", "stage", "mouse_id"])
    meta = W.drop_duplicates("session_id").set_index("session_id")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(meta.index)]
    jobs = [(r.session_id, r.mouse, r.file.parent) for r in ss.itertuples()
            if a.redo or not (r.file.parent / f"{r.mouse}_quiet_windows.npz").exists()]
    rows = []
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        for k, r in enumerate(pool.imap_unordered(session, jobs)):
            rows.append(r)
            print(f"{k + 1}/{len(jobs)} {r.get('session_id')} {r.get('error', '')}", flush=True)
    P = pd.DataFrame(rows)
    P = P.join(meta[["cohort", "stage", "mouse_id"]], on="session_id")
    f = OUT / "performance_sessions.csv"
    if f.exists() and not a.redo:
        P = pd.concat([pd.read_csv(f), P]).drop_duplicates("session_id", keep="last")
    P.to_csv(f, index=False)
    print(P[["rate_wh", "rate_ah", "rate_fa", "dprime_wh_fa"]].describe().round(3).to_string())
    print(f"{len(P)} sessions, {round((time.time() - t0) / 60, 1)} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=12)
    ap.add_argument("--redo", action="store_true")
    main(ap.parse_args())
