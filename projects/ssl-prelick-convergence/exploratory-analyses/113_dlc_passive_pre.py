"""113 -- DLC behaviour in passive-pre trials: whisker velocity and jaw opening, whisker and auditory stimuli, learning day vs expert
(user 2026-10-11; "careful of the sampling rate, which may differ"). Exploratory.
Sessions: those of 105 (learners rule for day 0; all expert sessions). Passive-pre trials: context == "passive" and start before the first
active trial. Stimulus time: whisker_stim_time / auditory_stim_time when present, else start_time.
DLC series (NWB processing/behavior/BehavioralTimeSeries, each with its own timestamps; cameras may run at different rates per session):
  length checks (2026-10-11): timestamps and data lengths differ in many files; equal -> used; timestamps empty (MH jaw) -> the whisker
  camera timestamps when the lengths match exactly; difference <= 0.5 % -> both truncated to the shorter (start-aligned); larger (e.g. data
  half the length of the timestamps: a ~100 Hz camera with 200 Hz timestamps) -> series excluded (listed in sessions.csv)
  whisker angle (top camera): frames with whisker_base likelihood < LIK (0.5) masked (2026-10-11: 0.9 masked most frames, DLC likelihoods are low;
  tip likelihood ignored, near 0 in MH sessions); velocity recomputed as d(angle)/dt from the
  timestamps (deg/s; np.gradient), not taken from the stored per-frame velocity (that scales with the frame rate); |velocity| also kept
  jaw opening: jaw_distance (side camera), frames with jaw_likelihood < LIK masked; 2026-10-11: converted to mm with the session's
  setup calibration (scripts/ssl_dlc_calibration.py, skills/ssl-dlc-units: setup 1 side 33.4 px/mm, setup 2 side 28.8 px/mm); the px
  trace is kept as "jaw opening (px)"; pixel scale differs between rigs (~15 vs ~470 px),
  but baseline-subtracted delta px is comparable (median auditory peak ~9 px AB vs ~25 px MH); the figure shows delta px, the version
  divided by the session's 5-95 percentile range is kept in traces.parquet ("jaw opening (norm.)")
Each trial is interpolated onto a common grid (GRID_DT = 5 ms, -0.5 .. +1.0 s from the stimulus; linear interpolation between valid
frames, gaps > MAX_GAP left NaN) and baseline-corrected (mean over [-0.2, 0) s). Per session: mean over trials of the trial type (>= 3).
Figure (compact, 2026-10-11): rows whisker speed |d angle/dt| (deg/s) and jaw opening (delta px; rigs differ ~2x in delta px, AB vs MH), columns
  whisker / auditory x day 0 / expert, R+ and R- overlaid, session mean +- SEM, 25-ms boxcar for display. Older layout:
R- magenta; mean +- s.e.m. over sessions. Per-session camera frame intervals saved and printed.
Added 2026-10-11: tongue out = P(tongue_likelihood >= LIK), not baseline-subtracted; nose movement = |d nose_angle/dt| (deg/s, nose_tip
  likelihood >= LIK); pupil area (pupil_likelihood >= LIK) as % change from the trial's [-0.5, 0) s mean; grid extended to +2 s.
Output: across_days/<ref>/dlc_passive_pre/{traces.parquet, sessions.csv, dlc_passive_pre_<pop>.{png,pdf}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../113_dlc_passive_pre.py [--n-proc 20] [--replot] [--pop learners]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import ssl_dlc_calibration as cal  # noqa: E402  pixel -> mm per setup and camera (skills/ssl-dlc-units)
m105 = importlib.import_module("105_trial_rates")
m51 = m105.m51
OUT = m51.OUTROOT / "dlc_passive_pre"
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
LIK, GRID_DT, MAX_GAP, MIN_TR = 0.5, 0.005, 0.05, 3
GRID = np.round(np.arange(-0.5, 2.0 + 1e-9, GRID_DT), 4)          # 2026-10-11: to +2 s for the pupil
BASE = (-0.2, 0.0)
BASE_PUPIL = (-0.5, 0.0)
# measures added 2026-10-11 (user: "also plot other behavioural metrics"); (name, baseline window or None = not subtracted)
EXTRA = {"tongue out": None, "nose movement": BASE, "pupil area": BASE_PUPIL}


def to_grid(t, v, events, base=BASE, grid=None):
    """t, v: series (NaN = invalid); events: stimulus times -> trials x grid (linear interpolation over valid frames, gaps > MAX_GAP NaN)"""
    GRID = globals()["GRID"] if grid is None else grid
    ok = np.isfinite(v); t, v = t[ok], v[ok]
    out = np.full((len(events), len(GRID)), np.nan)
    if len(t) < 10:
        return out
    for i, e in enumerate(events):
        g = e + GRID
        lo, hi = np.searchsorted(t, g[0] - MAX_GAP), np.searchsorted(t, g[-1] + MAX_GAP)
        if hi - lo < 5:
            continue
        tt, vv = t[lo:hi], v[lo:hi]
        y = np.interp(g, tt, vv, left=np.nan, right=np.nan)
        j = np.searchsorted(tt, g); j = np.clip(j, 1, len(tt) - 1)
        gap = tt[j] - tt[j - 1]
        y[gap > MAX_GAP] = np.nan
        out[i] = y
    if base is None:
        return out
    b = (GRID >= base[0]) & (GRID < base[1])
    return out - np.nanmean(out[:, b], 1, keepdims=True)


def load_series(sid):
    """DLC series of a session (also used by 116): returns info, trials table, {measure: (timestamps, values)}; trials None on error"""
    from pynwb import NWBHDF5IO
    try:
        with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
            n = io.read(); tr = n.trials.to_dataframe()
            ts = n.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
            flags = {}

            def get(k, alt_t=None):
                """(timestamps, data) with the length checks of 2026-10-11 (timestamps / data lengths differ in many NWB files)"""
                if k not in ts:
                    return None, None
                t, d = np.asarray(ts[k].timestamps[:], float), np.asarray(ts[k].data[:], float)
                if len(t) == len(d):
                    flags[k] = "ok"; return t, d
                if len(t) == 0 and alt_t is not None and len(alt_t) == len(d):
                    flags[k] = "borrowed whisker timestamps"; return alt_t, d
                if len(t) and abs(len(t) - len(d)) <= 0.005 * len(d):
                    n_ = min(len(t), len(d)); flags[k] = f"truncated {abs(len(t) - len(d))} frames"; return t[:n_], d[:n_]
                flags[k] = f"excluded (timestamps {len(t)} vs data {len(d)})"; return None, None
            tw, wa = get("whisker_angle"); _, wtl = get("whisker_tip_likelihood", tw); _, wbl = get("whisker_base_likelihood", tw)
            tj, jd = get("jaw_distance", tw); _, jl = get("jaw_likelihood", tw)
            if wtl is not None and tw is not None and len(wtl) != len(tw):
                wtl = None
            if wbl is not None and tw is not None and len(wbl) != len(tw):
                wbl = None
            if jl is not None and tj is not None and len(jl) != len(tj):
                jl = None
            ext = {}
            for name, (dk, lk) in {"tongue": ("tongue_likelihood", None), "nose": ("nose_angle", "nose_tip_likelihood"),
                                   "pupil": ("pupil_area", "pupil_likelihood")}.items():
                t_, d_ = get(dk, tw)
                l_ = get(lk, tw)[1] if lk else None
                ext[name] = (t_, d_, l_ if (l_ is not None and t_ is not None and len(l_) == len(t_)) else None)
    except Exception as e:  # noqa: BLE001
        return dict(session_id=sid, error=repr(e)[:150]), None, {}
    info = dict(session_id=sid, dt_whisker=float(np.median(np.diff(tw))) if tw is not None else np.nan,
                dt_jaw=float(np.median(np.diff(tj))) if tj is not None else np.nan,
                flag_whisker=flags.get("whisker_angle", "missing"), flag_jaw=flags.get("jaw_distance", "missing"))
    info_setup = np.nan
    series = {}
    if tw is not None:
        a = wa.copy()
        if wbl is not None:
            a[wbl < LIK] = np.nan
        vel = np.gradient(a, tw)                                       # deg/s from the timestamps
        series["whisker velocity"] = (tw, vel); series["|whisker velocity|"] = (tw, np.abs(vel))
    if tj is not None:
        series["jaw distance raw"] = (tj, jd.copy())                     # unmasked px (jaw onset detection, 118)
        d = jd.copy()
        if jl is not None:
            d[jl < LIK] = np.nan
        series["jaw opening (px)"] = (tj, d)
        try:
            series["jaw opening"] = (tj, cal.to_mm(d, sid, "jaw_distance")); info_setup = cal.setup_of(sid)
        except (KeyError, FileNotFoundError):
            info_setup = np.nan                                            # no setup -> no mm trace (session listed in sessions.csv)
        lo_, hi_ = np.nanpercentile(d, [5, 95]); series["jaw opening (norm.)"] = (tj, d / max(hi_ - lo_, 1e-9))
    t_, d_, _ = ext["tongue"]
    if t_ is not None:
        series["tongue out"] = (t_, (d_ >= LIK).astype(float))              # P(tongue detected), tongue_likelihood >= LIK
    t_, d_, l_ = ext["nose"]
    if t_ is not None:
        a = d_.copy()
        if l_ is not None:
            a[l_ < LIK] = np.nan
        series["nose movement"] = (t_, np.abs(np.gradient(a, t_)))           # |d nose angle / dt| (deg/s), side camera
    t_, d_, l_ = ext["pupil"]
    if t_ is not None:
        a = d_.copy()
        if l_ is not None:
            a[l_ < LIK] = np.nan
        series["pupil area"] = (t_, a)
    info["setup"] = info_setup if tj is not None else np.nan
    return info, tr, series


def session(args):
    sid, = args
    info, tr, series = load_series(sid)
    if tr is None:
        return info, []
    act = tr[tr.context == "active"] if "context" in tr else tr.iloc[:0]
    if act.empty:
        return dict(info, error="no active"), []
    pre = tr[(tr.context == "passive") & (tr.start_time < act.start_time.min())]
    rows = []
    for ttype, col, lab in (("whisker_trial", "whisker_stim_time", "whisker"), ("auditory_trial", "auditory_stim_time", "auditory")):
        d = pre[pre.trial_type == ttype]
        if len(d) < MIN_TR:
            continue
        ev = d[col].to_numpy(float) if col in d else d.start_time.to_numpy(float)
        ev = np.where(np.isfinite(ev), ev, d.start_time.to_numpy(float))
        for k, (t, v) in series.items():
            G = to_grid(t, v, ev, EXTRA.get(k, BASE))
            if k == "pupil area":                                          # % change from the trial's own [-0.5, 0) s baseline
                R = to_grid(t, v, ev, None); b0 = np.nanmean(R[:, (GRID >= BASE_PUPIL[0]) & (GRID < BASE_PUPIL[1])], 1, keepdims=True)
                G = 100 * G / np.where(b0 > 0, b0, np.nan)
            okt = np.isfinite(G).mean(1) > 0.5
            if okt.sum() >= MIN_TR:
                rows.append(dict(session_id=sid, stim=lab, measure=k, n_trials=int(okt.sum()), trace=np.nanmean(G[okt], 0).tolist()))
    info["n_pre_whisker"] = int((pre.trial_type == "whisker_trial").sum()); info["n_pre_auditory"] = int((pre.trial_type == "auditory_trial").sum())
    return info, rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(s,) for s in meta.session_id])
    pd.DataFrame([r for r, _ in res]).merge(meta, on="session_id").to_csv(OUT / "sessions.csv", index=False)
    pd.DataFrame([x for _, t in res for x in t]).merge(meta, on="session_id").to_parquet(OUT / "traces.parquet", index=False)


def figure(pop):
    """2026-10-11 (user: other behavioural metrics, 400 ms, tight y; pupil 2 s): rows whisker / auditory stimulus; columns whisker speed,
    nose movement, jaw opening, tongue out, pupil area; R+ / R- x day 0 (dashed) / expert (solid) overlaid; session mean +- SEM;
    display smoothing 25-ms boxcar; y limits = data range in the shown window (mean +- SEM over all groups and both stimuli) + 4 %"""
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH = m88.COH
    plt = m62.setup()
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    D = flt(pd.read_parquet(OUT / "traces.parquet"))
    meas = [("|whisker velocity|", "whisker speed (Δ°/s)", 400), ("nose movement", "nose speed (Δ°/s)", 400),
            ("jaw opening", "jaw opening (Δ mm)", 400), ("tongue out", "P(tongue out)", 400), ("pupil area", "pupil area (Δ %)", 2000)]
    rows = ["whisker", "auditory"]
    tms = GRID * 1e3; k = np.ones(5) / 5
    sm = lambda x: np.convolve(np.nan_to_num(x), k, "same")
    fig, axs = plt.subplots(2, len(meas), figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.45, hspace=0.35))
    for j, (mz, yl, tmax) in enumerate(meas):
        t0 = -500 if mz == "pupil area" else -200
        keep = (tms >= t0) & (tms <= tmax); lo, hi = np.inf, -np.inf
        for i, stim in enumerate(rows):
            ax = axs[i, j]
            for coh, st in GROUPS:
                d = D[(D.measure == mz) & (D.stim == stim) & (D.cohort == coh) & (D.stage == st)]
                if len(d) < 2:
                    continue
                A = np.stack(d.trace.apply(np.asarray).to_numpy()); A = np.stack([sm(a) for a in A])
                mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                ax.fill_between(tms[keep], (mu - se)[keep], (mu + se)[keep], color=COH[coh], alpha=0.18, lw=0)
                ax.plot(tms[keep], mu[keep], color=COH[coh], lw=0.8, ls="-" if st == "expert" else (0, (2.5, 1.5)),
                        label=f"{coh.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'} ({len(d)})")
                lo, hi = min(lo, np.nanmin((mu - se)[keep])), max(hi, np.nanmax((mu + se)[keep]))
            ax.axvline(0, color="0.4", lw=0.4, ls=":")
            if EXTRA.get(mz, BASE) is not None:
                ax.axhline(0, color="0.8", lw=0.4)
            ax.tick_params(labelsize=4.6); ax.set_xlim(t0, tmax)
            ax.set_xticks([0, 1000, 2000] if tmax == 2000 else [-200, 0, 200, 400])
            if i == 0:
                ax.set_title(yl, fontsize=5.4); ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time from stimulus (ms)", fontsize=4.8)
            if j == 0:
                ax.set_ylabel(f"{stim} stimulus", fontsize=5.4)
        pad = 0.04 * (hi - lo)
        for i in range(2):
            axs[i, j].set_ylim(lo - pad, hi + pad)
    h, _ = axs[0, 2].get_legend_handles_labels(); l = []
    Dm = D[D.measure.isin([m[0] for m in meas])]
    for coh, st in GROUPS:                                               # session-count range over panels
        c = Dm[(Dm.cohort == coh) & (Dm.stage == st)].groupby(["measure", "stim"]).session_id.nunique()
        rng_ = f"{c.min()}" if c.min() == c.max() else f"{c.min()}–{c.max()}"
        l.append(f"{coh.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'} (n = {rng_} sessions)")
    fig.legend(h, l, frameon=False, fontsize=4.8, loc="upper center", ncol=4, handlelength=2.0, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"dlc_passive_pre_{pop}.{ext}", dpi=300)
    print(D.groupby(["stim", "measure", "cohort", "stage"]).session_id.nunique().unstack([2, 3]))


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=20)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
