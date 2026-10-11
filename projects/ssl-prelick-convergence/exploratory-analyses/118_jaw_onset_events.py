"""118 -- Pre-lick events realigned to the DLC jaw-opening onset (user 2026-10-12: "test the prelick CD whole analyses (within and across
days) but looking at 100 ms before jaw opening onset ... I want to see if the results hold"). Compute only; downstream scripts are run
unchanged with PRELICK_ALIGN=jaw (051 TAG "_sl_jaw", roots across_days/sl_stable_jaw, within_day/sl_stable_jaw).
Events and units: those of 051 (REF=sl): WH, AH = trials of the analysed epoch with a lick; SL = spontaneous-lick bout onsets (piezo).
Jaw signal: jaw_distance (px, unmasked as in the original detection; 113 load_series timestamps and length checks; skill ssl-dlc-units).
Onset (rule of behaviour_analysis find_dlc_jaw_onset_times: threshold = 7 x baseline SD of the jaw position, first crossing):
  WH, AH  baseline [start - 1 s, start); search (start + 50 ms, corrected first lick]  (50 ms: remove_fast_jaw_onset_trials)
  SL      baseline [lick - 1.0 s, lick - 0.5 s); search (lick - 0.5 s, lick]
  no crossing in the search window -> event dropped (the original falls back to the maximum; dropped here so every kept event has a
  real onset). Baseline windows of the neural rates are unchanged (pre-trial for WH / AH, [-1, -0.5] s before the lick for SL).
Rates: 051 unit_rates with first_lick_time := jaw onset: [onset - 100 ms, onset) minus the event's baseline.
Per-event file: <session folder>/<mouse>_roc_prelick_sl_jaw_trials.npz (same keys as 051: rates, raw, cls, rt, trial_start, electrode_group,
cluster_id; plus lick_time, jaw_onset, lead = lick - onset). Sessions without usable jaw DLC are skipped.
Validation: trial onsets vs the stored behaviour_analysis table (combined_results/<mouse>/dlc_jaw_onset_times.pkl, jaw_dlc_onset is
relative to start_time): median absolute difference and correlation printed and saved.
Output: across_days/sl_stable_jaw/jaw_onsets/{events.csv, sessions.csv, validation.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../118_jaw_onset_events.py [--n-proc 20]
"""
import argparse
import importlib
import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PRELICK_ALIGN"] = "lick"                    # read the lick-aligned definitions, write the jaw-tagged files
m51 = importlib.import_module("051_roc_prelick")
m113 = importlib.import_module("113_dlc_passive_pre")
TAG_JAW = m51.TAG + "_jaw"
OUT = m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}_jaw" / "jaw_onsets"
K_SD, MIN_AFTER_STIM = 7.0, 0.05
SL_BASE_JAW, SL_SEARCH = (-1.0, -0.5), 0.5
OLD = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results")


def onset(t, v, b0, b1, s0, s1):
    """first time in (s0, s1] where v - mean(baseline) > K_SD * sd(baseline); NaN if none"""
    bi = (t >= b0) & (t < b1); bv = v[bi]; bv = bv[np.isfinite(bv)]
    if len(bv) < 20:
        return np.nan
    thr = K_SD * bv.std()
    si = np.where((t > s0) & (t <= s1))[0]
    if len(si) == 0 or thr <= 0:
        return np.nan
    x = v[si] - bv.mean(); c = np.where(x > thr)[0]
    return float(t[si[c[0]]]) if len(c) else np.nan


def session(args):
    sid, folder, mouse = args
    from pynwb import NWBHDF5IO
    info, _, series = m113.load_series(sid)
    if "jaw distance raw" not in series:
        return dict(session_id=sid, status="no jaw DLC"), []
    tj, jd = series["jaw distance raw"]
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = m51.ru.process_nwb_tables(nwb)
        trials_raw = nwb.trials.to_dataframe()
        t, log = m51.select_trials(trials_raw, sid)
        sl = m51.spontaneous_licks(nwb, trials_raw, log["epoch"])
    t = t[t.cls != "FA"]
    t = pd.concat([t, pd.DataFrame(dict(first_lick_time=sl, start_time=sl, reaction_time=np.nan, cls="FA",
                                        base_lo=sl + m51.SL_BASE[0], base_hi=sl + m51.SL_BASE[1]))], ignore_index=True)
    t = t.sort_values("start_time").reset_index(drop=True)
    on = np.full(len(t), np.nan)
    for i, r in enumerate(t.itertuples()):
        if r.cls == "FA":
            on[i] = onset(tj, jd, r.first_lick_time + SL_BASE_JAW[0], r.first_lick_time + SL_BASE_JAW[1], r.first_lick_time - SL_SEARCH, r.first_lick_time)
        else:
            on[i] = onset(tj, jd, r.start_time - 1.0, r.start_time, r.start_time + MIN_AFTER_STIM, r.first_lick_time)
    t["lick_time_used"] = t.first_lick_time; t["jaw_onset"] = on
    ev = t.assign(session_id=sid, lead=t.first_lick_time - t.jaw_onset)[["session_id", "cls", "start_time", "lick_time_used", "jaw_onset", "lead"]]
    k = t[np.isfinite(on)].copy(); k["first_lick_time"] = k.jaw_onset
    spikes = [np.sort(np.asarray(s)) for s in units.spike_times]
    Wraw, X = m51.unit_rates(spikes, k)
    np.savez_compressed(folder / f"{mouse}_roc_prelick{TAG_JAW}_trials.npz", rates=X.astype(np.float32), raw=Wraw.astype(np.float32),
                        cls=k.cls.to_numpy(), rt=k.reaction_time.to_numpy(), trial_start=k.start_time.to_numpy(),
                        electrode_group=units.electrode_group.astype(str).to_numpy(), cluster_id=units.cluster_id.astype(str).to_numpy(),
                        lick_time=k.lick_time_used.to_numpy(), jaw_onset=k.jaw_onset.to_numpy(), lead=(k.lick_time_used - k.jaw_onset).to_numpy())
    st = dict(session_id=sid, status="ok", setup=info.get("setup"),
              **{f"n_{c}": int((t.cls == c).sum()) for c in ("WH", "AH", "FA")}, **{f"kept_{c}": int((k.cls == c).sum()) for c in ("WH", "AH", "FA")},
              **{f"lead_med_{c}": float(np.nanmedian((k.lick_time_used - k.jaw_onset)[k.cls == c])) if (k.cls == c).any() else np.nan
                 for c in ("WH", "AH", "FA")})
    return st, ev.to_dict("records")


def validate(E):
    rows = []
    for mouse, d in E.assign(mouse=E.session_id.str[:5]).groupby("mouse"):
        f = OLD / mouse / "dlc_jaw_onset_times.pkl"
        if not f.exists():
            continue
        P = pd.read_pickle(f)[["session_id", "start_time", "jaw_dlc_onset", "lick_flag"]]
        P = P[P.lick_flag == 1].assign(old=lambda q: q.start_time + q.jaw_dlc_onset)
        m = d[d.cls != "FA"].merge(P, on=["session_id", "start_time"], how="inner")
        rows.append(m)
    V = pd.concat(rows) if rows else pd.DataFrame()
    if len(V):
        v = V.dropna(subset=["jaw_onset", "old"]); dd = (v.jaw_onset - v.old).abs()
        print(f"validation vs dlc_jaw_onset_times.pkl: n = {len(v)} trials, median |diff| = {dd.median() * 1e3:.1f} ms, "
              f"within 10 ms: {(dd <= 0.010).mean():.2f}, r = {np.corrcoef(v.jaw_onset - v.start_time, v.old - v.start_time)[0, 1]:.3f}")
    return V


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = [(r.session_id, r.file.parent, r.mouse) for r in ss.itertuples() if (r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz").exists()]
    print(f"[118] {len(jobs)} sessions", flush=True)
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = pool.map(session, jobs)
    S = pd.DataFrame([r for r, _ in res]); E = pd.DataFrame([x for _, e in res for x in e])
    S.to_csv(OUT / "sessions.csv", index=False); E.to_csv(OUT / "events.csv", index=False)
    print(S.status.value_counts().to_string())
    ok = S[S.status == "ok"]
    for c in ("WH", "AH", "FA"):
        print(f"{c if c != 'FA' else 'SL'}: kept {ok[f'kept_{c}'].sum()} / {ok[f'n_{c}'].sum()} events; median lead lick - onset "
              f"{np.nanmedian(ok[f'lead_med_{c}']) * 1e3:.0f} ms")
    V = validate(E); V.to_csv(OUT / "validation.csv", index=False)
    # same unit table for the jaw-aligned roots (identical unit sets)
    for root in (m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}_jaw",):
        dst = root / "prelick_units.parquet"
        if not dst.exists():
            dst.symlink_to(m51.OUTROOT / "prelick_units.parquet")
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=20)
    main(ap.parse_args())
