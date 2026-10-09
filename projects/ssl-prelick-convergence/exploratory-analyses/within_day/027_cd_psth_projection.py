"""027 -- Time-resolved projection on the reward-lick CD (user 2026-10-08: "PSTH projected on the CD, -400 to 200 ms around the lick,
SEM over sessions").
Per learning-day session (026 sessions and units: >= 4 WH, >= 8 AH, >= 8 SL; stable units, mean pre-lick rate >= 0.1 Hz; whole brain):
  per event (WH, AH, SL; 051 trial table and order) spike counts in 10-ms bins from -400 to +200 ms around the first lick
  (spontaneous licks: the lick itself), Gaussian smoothing sigma = 20 ms, minus the event's own baseline rate (051 baseline:
  raw - baseline-corrected pre-lick rate), so the [-100, 0) ms mean of a trial equals its 026 pre-lick rate X.
  axis: exactly as 026 (5-fold CV over AH + SL, units z-scored on the training folds, u = unit(mean_z AH - mean_z SL)); the held-out AH
  and SL events are projected on their fold's u (no event is projected on an axis it helped build); WH on every fold's u, averaged.
  Time courses z-scored with the fold's mean / SD and projected; class means per session, per neuron (/ sqrt(N)).
Output: within_day/<ref>/cd_psth/cd_psth_projection.parquet (session_id, mouse_id, cohort, stage, n_units, cls, t, value)
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../027_cd_psth_projection.py [--n-proc 30]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
m51 = importlib.import_module("051_roc_prelick")
m002 = importlib.import_module("002_trial_slopes")
import os  # noqa: E402
RAW = bool(os.environ.get("SSL_026_RAW"))                      # 2026-10-09: no z-scoring (as 026 raw); output cd_psth_raw
OUT = m51.WITHIN / ("cd_psth_raw" if RAW else "cd_psth")
WIN_T, BIN, SIG = (-0.400, 0.200), 0.010, 2.0
EDGES = np.round(np.arange(WIN_T[0], WIN_T[1] + BIN / 2, BIN), 3)
TC = (EDGES[:-1] + EDGES[1:]) / 2


def session(args):
    sid, f, meta, W_s, seed = args
    from pynwb import NWBHDF5IO
    from sklearn.model_selection import StratifiedKFold
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"], kind="stable")
    X, raw, lab = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o]
    ev = (z["trial_start"].astype(float) + np.nan_to_num(z["rt"].astype(float)))[o]   # first lick (SL: the lick itself)
    if meta["stage"] != "learning" or (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return []
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    if ok.sum() < m002.MIN_UNITS:
        return []
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        units, _ = m51.ru.process_nwb_tables(io.read())
    assert (units.electrode_group.astype(str).to_numpy() == K.electrode_group.to_numpy()).all() and \
           (units.cluster_id.astype(str).to_numpy() == K.cluster_id.to_numpy()).all(), "unit order differs from the npz"
    spikes = [np.sort(np.asarray(s)) for s, k in zip(units.spike_times, ok) if k]
    X, Bl = X[ok], (raw - X)[ok]                                        # units x events; Bl = each event's baseline rate
    P = np.zeros((len(spikes), len(ev), len(TC)))
    for u, s in enumerate(spikes):
        lo, hi = np.searchsorted(s, ev + WIN_T[0]), np.searchsorted(s, ev + WIN_T[1])
        for j, (a, b) in enumerate(zip(lo, hi)):
            P[u, j] = np.histogram(s[a:b] - ev[j], EDGES)[0]
    P = gaussian_filter1d(P / BIN, SIG, axis=2) - Bl[:, :, None]       # units x events x time
    N = int(ok.sum())
    y_idx = np.where(np.isin(lab, ["AH", "FA"]))[0]; y = (lab[y_idx] == "AH").astype(int); w_idx = np.where(lab == "WH")[0]
    hold = np.full((len(y_idx), len(TC)), np.nan); pw = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(y_idx, y):     # as 026 cv_projection
        A = X[:, y_idx[tr]].T; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
        if RAW:
            sd = np.ones_like(sd)
        d = ((A[y[tr] == 1] - mu) / sd).mean(0) - ((A[y[tr] == 0] - mu) / sd).mean(0)
        u = d / max(np.linalg.norm(d), 1e-12)
        zt = lambda idx: np.einsum("uet,u->et", (P[:, idx] - mu[:, None, None]) / sd[:, None, None], u)
        hold[te] = zt(y_idx[te]); pw.append(zt(w_idx))
    cur = {"AH": hold[y == 1].mean(0), "SL": hold[y == 0].mean(0), "WH": np.mean(pw, 0).mean(0)}
    return [dict(meta, n_units=N, cls=c, t=float(t), value=float(v) / np.sqrt(N)) for c, tr_ in cur.items() for t, v in zip(TC, tr_)]


def main(a):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        Ws = W[W.session_id == r.session_id]
        if f.exists() and (Ws.stage == "learning").any():
            jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0],
                                               stage=Ws.stage.iloc[0]), Ws[["electrode_group", "cluster_id", "quality_label"]], i))
    print(len(jobs), "sessions", flush=True)
    rows = []
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        for r in pool.imap_unordered(session, jobs):
            rows += r
    pd.DataFrame(rows).to_parquet(OUT / "cd_psth_projection.parquet", index=False)
    print("ALL DONE", OUT / "cd_psth_projection.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    main(ap.parse_args())
