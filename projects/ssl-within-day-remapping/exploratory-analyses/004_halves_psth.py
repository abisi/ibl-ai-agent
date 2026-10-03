"""First-lick-aligned PSTHs per session half (early / late, split at the midpoint between the two middle auditory hits, as
001 / 003) for whisker hits (WH), auditory hits (AH) and spontaneous licks (SL).

Events, trial selection, spontaneous licks and artefact-corrected spike trains exactly as 051 (PRELICK_REF=sl).
Units: good + mua with mean raw pre-lick rate >= 0.1 Hz (051 trial files; same units as 001-003).
Per unit, class and half: PSTH (10 ms bins, [-0.6, 0.4] s around the first lick) minus the unit's mean baseline rate of
the same events (baseline windows as 051: 1 s before trial start for WH / AH, [-1, -0.5] s before the lick for SL);
then mean over units within the session. Output: one array per session (class x half x bins) with the number of events,
saved to combined_results_ks4/_within_day<TAG>/psth_halves/psth_halves.npz (+ sessions.csv).
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
OUT = m51.RES / f"_within_day{m51.TAG}" / "psth_halves"
UNIT_SET = ("good", "mua")
HALVES = ("early", "late")


def events(sid):
    """051 event table (WH, AH, SL as 'FA') and artefact-corrected spike trains"""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = m51.ru.process_nwb_tables(nwb)
        trials_raw = nwb.trials.to_dataframe()
        t, log = m51.select_trials(trials_raw)
        if m51.REF == "sl":
            sl = m51.spontaneous_licks(nwb, trials_raw, log["epoch"])
            t = t[t.cls != "FA"]
            t = pd.concat([t, pd.DataFrame(dict(first_lick_time=sl, start_time=sl, reaction_time=np.nan, cls="FA",
                                                base_lo=sl + m51.SL_BASE[0], base_hi=sl + m51.SL_BASE[1]))], ignore_index=True)
            t = t.sort_values("start_time").reset_index(drop=True)
    spikes = [np.sort(np.asarray(s)) for s in units.spike_times]
    return t, spikes, units


def session(args):
    sid, meta, W_s = args
    try:
        t, spikes, units = events(sid)
    except Exception as ex:
        print(f"[004] {sid} failed: {ex}", flush=True)
        return None
    st, cls = t.start_time.to_numpy(), t.cls.to_numpy()
    ta = np.sort(st[cls == "AH"])
    if len(ta) < 2:
        return None
    h = len(ta) // 2; cut = 0.5 * (ta[h - 1] + ta[h])
    half = np.where(st < cut, "early", "late")
    K = units[["electrode_group", "cluster_id"]].astype(str).reset_index(drop=True).merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    Wraw, _ = m51.unit_rates(spikes, t)
    ok = np.where(K.quality_label.isin(UNIT_SET).to_numpy() & (Wraw.mean(1) >= m51.MIN_FR))[0]
    if len(ok) < 5:
        return None
    edges = np.arange(m51.PSTH_WIN[0], m51.PSTH_WIN[1] + m51.PSTH_BIN / 2, m51.PSTH_BIN)
    fl, blo, bhi = t.first_lick_time.to_numpy(), t.base_lo.to_numpy(), t.base_hi.to_numpy()
    P = np.full((len(m51.CLASSES), 2, len(edges) - 1), np.nan, np.float32); N = np.zeros((len(m51.CLASSES), 2), int)
    for k, c in enumerate(m51.CLASSES):
        for j, hh in enumerate(HALVES):
            e = np.where((cls == c) & (half == hh))[0]
            N[k, j] = len(e)
            if len(e) == 0:
                continue
            acc = np.zeros(len(edges) - 1)
            for u in ok:
                s = spikes[u]
                lo, hi = np.searchsorted(s, fl[e] + edges[0]), np.searchsorted(s, fl[e] + edges[-1])
                rel = np.concatenate([s[a:b] - x for a, b, x in zip(lo, hi, fl[e])])
                rate = np.histogram(rel, edges)[0] / len(e) / m51.PSTH_BIN
                base = np.mean((np.searchsorted(s, bhi[e]) - np.searchsorted(s, blo[e])) / (bhi[e] - blo[e]))
                acc += rate - base
            P[k, j] = acc / len(ok)
    return dict(meta, n_units=len(ok), psth=P, n=N, edges=edges)


def main(a):
    t0 = time.time()
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    jobs = []
    for sid, Ws in W.groupby("session_id"):
        jobs.append((sid, dict(session_id=sid, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0]),
                     Ws[["electrode_group", "cluster_id", "quality_label"]].drop_duplicates()))
    if a.test:
        jobs = jobs[:a.test]
    res = []
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        for r in pool.imap_unordered(session, jobs):
            if r:
                res.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("psth", "n", "edges")} for r in res])
    meta.to_csv(OUT / "sessions.csv", index=False)
    np.savez_compressed(OUT / "psth_halves.npz", psth=np.stack([r["psth"] for r in res]), n=np.stack([r["n"] for r in res]),
                        edges=res[0]["edges"], session_id=meta.session_id.to_numpy(), classes=np.array(m51.CLASSES))
    print(f"ALL DONE {OUT} ({len(res)} sessions, {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    ap.add_argument("--test", type=int, default=0)
    main(ap.parse_args())
