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

v2 (2026-11-12, PRELICK_DLC=v2, dlc_version.py; user: "ALL analyses should use what is in NWB_ks4"): onsets from ephys_utilities
dlc_helpers (NAS checkout first on PYTHONPATH, plus NWB_reader for data_utils) with the agreed rule dh.JAW_ONSET_PARAMS (5 x pooled
baseline SD, walk back to 30 % of the first post-stimulus peak, prominent peak f = 0.5 with the 10-SD separate-opening guard), jaw =
load_dlc side camera (mm, own timestamps, converter reference fix) + clean_series default; side-video-excluded sessions skipped.
  WH, AH  dh.jaw_onset_times(nwb, all NWB trial rows of the session) as behaviour_analysis (pooled SD over the session's trials;
          search start - 0.5 s .. offset-corrected first lick); onset = start_time + jaw_dlc_onset, matched to the 051 events on
          start_time. Validated against combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl (validation.csv).
  SL      dh.movement_onset_times on the same jaw trace with pseudo-trials start = lick - 0.5 s = response_window_start, lick_time = lick:
          baseline_window (-0.5, 0) s = [lick - 1, lick - 0.5) as v1, pre_onset_window 0, time_start -0.5, time_stop 0.6 (search
          lick - 0.5 .. lick); threshold / walk-back / prominent peak from JAW_ONSET_PARAMS; pooled SD over the session's SL events.
  Dropped: events where the rule found no threshold-crossing bout (dh falls back to the peak time; detected explicitly by repeating
          the bout test of movement_onset_times on the returned trace, column crossing). Onsets before the stimulus are kept (user:
          real, earlier movements); the v1 MIN_AFTER_STIM 50-ms constraint is removed in v2 (it conflicts with the search from
          start - 0.5 s). For a pre-stimulus onset the [onset - 100, onset) rate window can overlap the pre-trial baseline window.
  Tags: _jaw2 (pre) / _jawc2 (centered); per-session <mouse>_roc_prelick_sl_jaw2_trials.npz / _sl_jawc2_; roots across_days/
  sl_stable_jaw2 (_jawc2); events.csv adds onset_dh, crossing, pre_stim; validation.csv (vs pkl), compare_v1.csv (vs the v1 events).
Run v2 (haas): PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:/mnt/lsens-analysis/Axel_Bisi/NWB_reader PRELICK_DLC=v2
  PRELICK_REF=sl PRELICK_UNITS=stable [PRELICK_JAW_WIN=centered] python .../118_jaw_onset_events.py [--n-proc 20] [--sessions a,b]
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
import dlc_version as dv  # noqa: E402  PRELICK_DLC switch (v1 default / v2 dlc_helpers)
# 2026-10-13: PRELICK_JAW_WIN=centered -> rate window [-50, +50) ms around jaw onset (tag / roots "_jawc"; run 051-downstream
# with PRELICK_ALIGN=jawc); default "pre" = [-100, 0) ms before jaw onset ("_jaw"). Both also store binned rates around jaw onset
# (jbins: 50-ms windows, centres -300 .. +200 ms, 25-ms step, minus the event's baseline) for the time courses of 120.
JWIN = os.environ.get("PRELICK_JAW_WIN", "pre")
JSFX = dv.JAW_SFX[JWIN]                                 # v1: _jaw / _jawc; v2: _jaw2 / _jawc2
if JWIN == "centered":
    m51.PRELICK = (-0.050, 0.050)
TAG_JAW = m51.TAG + JSFX
OUT = m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}{JSFX}" / "jaw_onsets"
JB_C = np.round(np.arange(-0.300, 0.2001, 0.025), 3)
K_SD, MIN_AFTER_STIM = 7.0, 0.05
SL_BASE_JAW, SL_SEARCH = (-1.0, -0.5), 0.5
OLD = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results" if dv.DLC == "v1" else "/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
SL_V2 = dict(time_start=-0.5, time_stop=0.6, baseline_window=(-0.5, 0.0), pre_onset_window=0.0)   # v2 SL pseudo-trial parameters


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


def crossing_found(trials, traces, thr, dt, P):
    """v2: per trial, True if dh.movement_onset_times found a threshold-crossing bout whose first local peak is at / after start_time
    (a real onset), False if it fell back to the peak time (no such bout). Same index arithmetic as movement_onset_times."""
    starts, rws = trials["start_time"].to_numpy(float), trials["response_window_start_time"].to_numpy(float)
    licks = trials["lick_time"].to_numpy(float)
    artifact = np.round(np.nanmean(rws - starts), 1)
    grid = np.arange(np.round(P["time_start"] / dt), np.round((P["time_stop"] + artifact) / dt) + 1) * dt
    t0 = int(np.argmin(np.abs(grid)))
    out = np.zeros(len(starts), bool)
    for k in range(len(starts)):
        trace = np.asarray(traces[k], float)
        if not np.isfinite(thr[k]) or np.all(np.isnan(trace)):
            continue
        stop = min(len(trace), t0 + int(round((P["time_stop"] + rws[k] - starts[k]) / dt)))
        lick = licks[k] - rws[k]
        if np.isfinite(lick):
            stop = min(stop, t0 + int(round(lick / dt)))
        start = max(0, t0 - int(round(P["pre_onset_window"] / dt)))
        w = np.nan_to_num(trace, nan=-np.inf)
        cross = np.flatnonzero(w[start:stop] > thr[k]) + start
        i = 0
        while i < len(cross):
            c = cross[i]
            end = c
            while end < stop and w[end] > thr[k]:
                end += 1
            p = c
            while p + 1 < end and w[p + 1] >= w[p]:
                p += 1
            if p >= t0:
                out[k] = True
                break
            i = np.searchsorted(cross, end)
    return out


def onsets_v2(sid, trials_raw, t):
    """v2 jaw onsets (session times) of the events in t (cls WH / AH = trials, FA = SL with first_lick_time = lick). Returns
    (onset, NaN where no crossing; raw dh onset incl. the peak fallback; crossing flag; per-trial table for validation) or a status"""
    dh, du = dv.dlc_helpers()
    f = str(m51.NWB / f"{sid}.nwb")
    if "side" in du.VIDEO_EXCLUDED_SESSIONS.get(sid, ((), ""))[0]:
        return "side video excluded"
    P = dict(dh.JAW_ONSET_PARAMS)
    trows = trials_raw.reset_index(drop=True)
    R = dh.jaw_onset_times(f, trows)
    if R is None:
        return "no jaw DLC"
    dt = 1 / float(R.freq.iloc[0])
    cr = crossing_found(trows, R.jaw_dlc_pos.to_list(), R.threshold.to_numpy(float), dt, P)
    V = pd.DataFrame(dict(session_id=sid, start_time=trows.start_time.to_numpy(float), trial_type=trows.trial_type.to_numpy(),
                          lick_flag=trows.lick_flag.to_numpy(), jaw_dlc_onset=R.jaw_dlc_onset.to_numpy(float), crossing=cr))
    side = dh.load_dlc(f, views=("side",))["side"]
    pos, _ = dh.clean_series(side, "jaw_distance")                     # the trace jaw_onset_times uses
    n = len(t)
    on_raw, crs = np.full(n, np.nan), np.zeros(n, bool)
    trial_ev = (t.cls != "FA").to_numpy()
    st = t.start_time.to_numpy(float)
    idx = pd.Index(V.start_time).get_indexer(st[trial_ev])
    assert (idx >= 0).all(), f"{sid}: 051 events not found in the NWB trial table"
    on_raw[trial_ev] = st[trial_ev] + V.jaw_dlc_onset.to_numpy()[idx]
    crs[trial_ev] = V.crossing.to_numpy()[idx]
    sl = t.first_lick_time.to_numpy(float)[~trial_ev]
    if len(sl):
        PS = dict(P, **SL_V2)
        pseudo = pd.DataFrame(dict(start_time=sl - 0.5, response_window_start_time=sl - 0.5, lick_time=sl))
        M = dh.movement_onset_times(pos, side.timestamps, pseudo, **PS)
        on_raw[~trial_ev] = pseudo.start_time.to_numpy() + M.onset.to_numpy(float)
        crs[~trial_ev] = crossing_found(pseudo, M.trace.to_list(), M.threshold.to_numpy(float), 1 / dh.frame_rate(side.timestamps), PS)
    return np.where(crs, on_raw, np.nan), on_raw, crs, V


def session(args):
    sid, folder, mouse = args
    from pynwb import NWBHDF5IO
    if dv.DLC == "v1":
        info, _, series = m113.load_series(sid)
        if "jaw distance raw" not in series:
            return dict(session_id=sid, status="no jaw DLC"), [], None
        tj, jd = series["jaw distance raw"]
    else:
        info = dict(setup=np.nan)
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
    V = None
    if dv.DLC == "v1":
        on = np.full(len(t), np.nan)
        for i, r in enumerate(t.itertuples()):
            if r.cls == "FA":
                on[i] = onset(tj, jd, r.first_lick_time + SL_BASE_JAW[0], r.first_lick_time + SL_BASE_JAW[1], r.first_lick_time - SL_SEARCH, r.first_lick_time)
            else:
                on[i] = onset(tj, jd, r.start_time - 1.0, r.start_time, r.start_time + MIN_AFTER_STIM, r.first_lick_time)
    else:
        res = onsets_v2(sid, trials_raw, t)
        if isinstance(res, str):
            return dict(session_id=sid, status=res), [], None
        on, on_raw, crs, V = res
        t["onset_dh"], t["crossing"] = on_raw, crs
        t["pre_stim"] = (t.cls != "FA").to_numpy() & (on_raw < t.start_time.to_numpy())
        try:
            info["setup"] = m113.cal.setup_of(sid)
        except Exception:  # noqa: BLE001
            pass
    t["lick_time_used"] = t.first_lick_time; t["jaw_onset"] = on
    cols = ["session_id", "cls", "start_time", "lick_time_used", "jaw_onset", "lead"] + (["onset_dh", "crossing", "pre_stim"] if dv.DLC == "v2" else [])
    ev = t.assign(session_id=sid, lead=t.first_lick_time - t.jaw_onset)[cols]
    k = t[np.isfinite(on)].copy(); k["first_lick_time"] = k.jaw_onset
    spikes = [np.sort(np.asarray(s)) for s in units.spike_times]
    Wraw, X = m51.unit_rates(spikes, k)
    on_k, blo, bhi = k.jaw_onset.to_numpy(), k.base_lo.to_numpy(), k.base_hi.to_numpy()
    JB = np.zeros((len(spikes), len(k), len(JB_C)), np.float32)
    for u, s_ in enumerate(spikes):
        b = (np.searchsorted(s_, bhi) - np.searchsorted(s_, blo)) / (bhi - blo)
        for j, c in enumerate(JB_C):
            JB[u, :, j] = (np.searchsorted(s_, on_k + c + 0.025) - np.searchsorted(s_, on_k + c - 0.025)) / 0.05 - b
    np.savez_compressed(folder / f"{mouse}_roc_prelick{TAG_JAW}_trials.npz", rates=X.astype(np.float32), raw=Wraw.astype(np.float32),
                        jbins=JB, jbins_c=JB_C,
                        cls=k.cls.to_numpy(), rt=k.reaction_time.to_numpy(), trial_start=k.start_time.to_numpy(),
                        electrode_group=units.electrode_group.astype(str).to_numpy(), cluster_id=units.cluster_id.astype(str).to_numpy(),
                        lick_time=k.lick_time_used.to_numpy(), jaw_onset=k.jaw_onset.to_numpy(), lead=(k.lick_time_used - k.jaw_onset).to_numpy())
    st = dict(session_id=sid, status="ok", setup=info.get("setup"),
              **{f"n_{c}": int((t.cls == c).sum()) for c in ("WH", "AH", "FA")}, **{f"kept_{c}": int((k.cls == c).sum()) for c in ("WH", "AH", "FA")},
              **{f"lead_med_{c}": float(np.nanmedian((k.lick_time_used - k.jaw_onset)[k.cls == c])) if (k.cls == c).any() else np.nan
                 for c in ("WH", "AH", "FA")})
    if dv.DLC == "v2":
        st.update({f"pre_stim_{c}": int((k.cls == c).mul(k.pre_stim).sum()) for c in ("WH", "AH")})
    return st, ev.to_dict("records"), V


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


def validate_v2(VT):
    """v2: every NWB trial's dh onset (VT, all rows of jaw_onset_times) vs combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl"""
    rows = []
    for mouse, d in VT.assign(mouse=VT.session_id.str.split("_").str[0]).groupby("mouse"):
        f = OLD / mouse / "dlc_jaw_onset_times.pkl"
        if not f.exists():
            rows.append(d.assign(pkl=np.nan, in_pkl=False)); continue
        P = pd.read_pickle(f)[["session_id", "start_time", "jaw_dlc_onset"]].rename(columns={"jaw_dlc_onset": "pkl"})
        rows.append(d.merge(P, on=["session_id", "start_time"], how="left", indicator=True)
                    .assign(in_pkl=lambda q: q._merge == "both").drop(columns="_merge"))
    V = pd.concat(rows, ignore_index=True)
    v = V[V.in_pkl]
    same = np.isclose(v.jaw_dlc_onset, v.pkl, atol=1e-9, rtol=0) | (v.jaw_dlc_onset.isna() & v.pkl.isna())
    print(f"validation vs combined_results_ks4 dlc_jaw_onset_times.pkl: {len(v)} / {len(V)} trials in a pkl "
          f"({V[~V.in_pkl].session_id.nunique()} sessions not in a pkl); identical {same.sum()} ({same.mean():.4f}); "
          f"max |diff| {np.nanmax(np.abs(v.jaw_dlc_onset - v.pkl)) * 1e3 if len(v) else np.nan:.3g} ms", flush=True)
    vl = v[v.lick_flag == 1]; sl_ = np.isclose(vl.jaw_dlc_onset, vl.pkl, atol=1e-9, rtol=0) | (vl.jaw_dlc_onset.isna() & vl.pkl.isna())
    print(f"  lick trials: {len(vl)}, identical {sl_.mean():.4f}; no-crossing (peak fallback) among lick trials: {(~vl.crossing).mean():.3f}")
    return V


def compare_v1(E):
    """v2 vs v1 events (sl_stable_jaw[c]/jaw_onsets/events.csv): kept counts, lead lick - onset per class, onset differences"""
    f1 = m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}{dv.JAW_SFX[JWIN].replace('2', '')}" / "jaw_onsets" / "events.csv"
    if not f1.exists():
        print("no v1 events at", f1); return pd.DataFrame()
    E1 = pd.read_csv(f1); E1 = E1[E1.session_id.isin(E.session_id.unique())]
    rows = []
    for c in ("WH", "AH", "FA"):
        a, b = E1[E1.cls == c], E[E.cls == c]
        rows.append(dict(cls="SL" if c == "FA" else c, n_v1=len(a), kept_v1=int(a.jaw_onset.notna().sum()), n_v2=len(b), kept_v2=int(b.jaw_onset.notna().sum()),
                         lead_med_v1_ms=a.lead.median() * 1e3, lead_med_v2_ms=b.lead.median() * 1e3,
                         lead_iqr_v1_ms=(a.lead.quantile(0.75) - a.lead.quantile(0.25)) * 1e3, lead_iqr_v2_ms=(b.lead.quantile(0.75) - b.lead.quantile(0.25)) * 1e3,
                         pre_stim_v2=int(b.pre_stim.fillna(False).astype(bool).sum()) if c != "FA" else 0))
        m = a.merge(b, on=["session_id", "cls", "start_time"], suffixes=("_v1", "_v2")).dropna(subset=["jaw_onset_v1", "jaw_onset_v2"])
        if len(m):
            d = (m.jaw_onset_v2 - m.jaw_onset_v1) * 1e3
            rows[-1].update(n_both=len(m), d_med_ms=d.median(), d_abs_med_ms=d.abs().median(), within_10ms=(d.abs() <= 10).mean(),
                            r=np.corrcoef(m.lead_v1, m.lead_v2)[0, 1])
    C = pd.DataFrame(rows); pd.set_option("display.width", 250); print(C.round(3).to_string(), flush=True)
    return C


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    if a.sessions:
        ss = ss[ss.session_id.isin(a.sessions.split(","))]
    jobs = [(r.session_id, r.file.parent, r.mouse) for r in ss.itertuples() if (r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz").exists()]
    print(f"[118] {len(jobs)} sessions, DLC {dv.DLC}, tag {TAG_JAW}, out {OUT}", flush=True)
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = pool.map(session, jobs)
    S = pd.DataFrame([r[0] for r in res]); E = pd.DataFrame([x for r in res for x in r[1]])
    S.to_csv(OUT / "sessions.csv", index=False); E.to_csv(OUT / "events.csv", index=False)
    print(S.status.value_counts().to_string())
    ok = S[S.status == "ok"]
    for c in ("WH", "AH", "FA"):
        print(f"{c if c != 'FA' else 'SL'}: kept {ok[f'kept_{c}'].sum()} / {ok[f'n_{c}'].sum()} events; median lead lick - onset "
              f"{np.nanmedian(ok[f'lead_med_{c}']) * 1e3:.0f} ms" + (f"; pre-stimulus onsets kept {ok[f'pre_stim_{c}'].sum()}" if dv.DLC == "v2" and c != "FA" else ""))
    if dv.DLC == "v1":
        V = validate(E); V.to_csv(OUT / "validation.csv", index=False)
    else:
        V = validate_v2(pd.concat([r[2] for r in res if r[2] is not None], ignore_index=True)); V.to_csv(OUT / "validation.csv", index=False)
        compare_v1(E).to_csv(OUT / "compare_v1.csv", index=False)
    # same unit table for the jaw-aligned roots (identical unit sets)
    for root in (m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}{JSFX}",):
        dst = root / "prelick_units.parquet"
        if not dst.exists():
            dst.symlink_to(m51.OUTROOT / "prelick_units.parquet")
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=20)
    ap.add_argument("--sessions", default="", help="comma-separated session ids (pilot; tables go to jaw_onsets/pilot/)")
    a_ = ap.parse_args()
    if a_.sessions:
        OUT = OUT / "pilot"
    main(a_)
