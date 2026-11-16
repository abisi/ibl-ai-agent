"""017 -- Do R+ mice whisk more during spontaneous periods, and is whisking associated with a higher readout? (user 2026-11-02)

Hypothesis (user): whisking contributes to consolidation (replay) of the whisker pattern, so R+ mice whisk more in spontaneous
periods and whisking goes with a higher whisker readout / correlation.

Data: 009 v7 saved files (stable units, whole brain "All"): session.npz (grid, valid, evoked, epoch, whisk mask, lick_dt),
behaviour.npz (raw whisker speed, deg/s), g_All_<target>{,_corr}.npz (continuous readouts), events parquets; trial
start / stop times from the NWB.
Spontaneous periods (SP) = iti: valid task time outside every trial (start_time .. stop_time) and the evoked tail (1 s after
  any stimulus onset), > LICK_FREE s from any lick (009 "iti" plus no licking); the passive blocks hold < 1 min of such time.
Whisking = the 009 per-session mask (whisker speed > median + 5 robust s.d., dilated by MOVE_MARGIN); bouts = runs of raw speed
> the session threshold lasting >= MIN_BOUT_S, merged across gaps < MERGE_S; raw speed in deg/s (absolute) as a second, between-
session-comparable measure (same tracking pipeline for all mice); fraction of SP time above FIXED_DEG deg/s.
Readout normalisation (threshold-free): (readout - median) / s.d., both over valid time outside the evoked tail (per session).
Per session x SP:
  whisking: frac_whisk, speed mean / median / 90th pct (deg/s), frac > FIXED_DEG, bouts / min, median bout duration
  readouts (decoder and correlation, whisker and auditory targets): mean normalised readout in whisking vs non-whisking SP time,
    Spearman rho with raw speed, event rate in whisking vs non-whisking SP time (009 events, first window), bout-onset-
    triggered readout (-2 .. +4 s, bouts preceded by >= PRE_FREE_S s of raw speed below the threshold), post-bout readout
    (non-whisking SP windows, raw speed < threshold, 0-3 s after bout offset minus 3-0 s before bout onset)
  specificity: rho(whisker target) - rho(auditory target), per readout
Tests: R+ vs R- per measure (Mann-Whitney and Welch, uncorrected, session = unit); across sessions: event rate / readout vs
  frac_whisk, OLS per cohort (scatter + line + 95 % CI, solid if p < 0.05) and pooled y ~ whisk * cohort.
Absolute whisking (fig E): angular velocity and whisking amplitude from the NWB angle (see AV_* / AMP_*), fractions of time
  above fixed thresholds; effect sizes (Hedges g, rank-biserial, ratio of medians with bootstrap CI) for every cohort test.
Output: figures/full_<unit_set>/analysis_017_whisking<tag>/fig_whisking_{amount,readout,bouts,sessions,angvel}.{png,pdf,svg},
<HOME>/analysis_017_whisking<tag>_{sessions,stats}.csv. Sessions: --sessions file (one session id per line).
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m9 = importlib.import_module("009_full_dataset_events")

LICK_FREE = 1.0                                  # s from the nearest lick
FIXED_DEG = 200.0                                # deg/s, absolute whisking threshold (between-session comparable)
MIN_BOUT_S, MERGE_S, PRE_FREE_S = 0.1, 0.2, 1.0
SPEED_EDGES = np.array([0, 25, 50, 100, 200, 400, 800, np.inf])
BTA = (-2.0, 4.0)
POST_S = 3.0
SPS = ("iti",)                                   # passive blocks: < 1 min of spontaneous time per session (pilot), dropped
RDS = [("whisker", ""), ("auditory", ""), ("whisker", "_corr"), ("auditory", "_corr")]
RD_NAME = {"": "decoder", "_corr": "correlation"}
COH = m9.COH
# angular velocity (user 2026-11-02): NWB whisker angle (deg; DLC frames with base likelihood < m1.LIK removed), gaps <= AV_GAP s
# linearly interpolated, 4th-order Butterworth low-pass AV_LP Hz (zero phase) on each finite stretch >= AV_MIN_RUN s, angular
# velocity = d angle / dt (deg/s); whisking amplitude = Hilbert envelope of the AMP_BAND band-passed angle (deg). Absolute
# units, not z-scored per mouse (z-scoring would remove the between-mouse difference being tested).
AV_GAP, AV_LP, AV_MIN_RUN = m9.AV_GAP, m9.AV_LP, m9.AV_MIN_RUN    # as the 009 partial correlation
AV_THRS = (25, 50, 100, 200, 400, 800)           # deg/s
AMP_BAND = (5.0, 25.0)                           # Hz, whisking band
AMP_THRS = (2, 5, 10, 20)                        # deg
MARK = {"AB": "o", "MH": "^"}                    # marker by experimenter (mouse prefix)


def angle_signals(sid):
    """camera times, low-passed angle, |angular velocity| (deg/s), whisking amplitude (deg); None without tracking"""
    from pynwb import NWBHDF5IO
    from scipy.signal import butter, sosfiltfilt, hilbert
    with NWBHDF5IO(str(m1.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        ts = io.read().processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
        if "whisker_angle" not in ts:
            return None
        tw, a = np.asarray(ts["whisker_angle"].timestamps[:], float), np.asarray(ts["whisker_angle"].data[:], float)
        lk = np.asarray(ts["whisker_base_likelihood"].data[:], float) if "whisker_base_likelihood" in ts else None
    n = min(len(tw), len(a))
    tw, a = tw[:n], a[:n].copy()
    if lk is not None and len(lk) >= n:
        a[lk[:n] < m1.LIK] = np.nan
    fs = 1 / np.median(np.diff(tw))
    fin = np.isfinite(a)
    if fin.sum() < fs * 10:
        return None
    on, off = runs(~fin)                                                 # interpolate short gaps
    ai = np.interp(tw, tw[fin], a[fin])
    for s0, s1 in zip(on, off):
        if (s1 - s0) / fs > AV_GAP or s0 == 0 or s1 == len(a):
            ai[s0:s1] = np.nan
    lp = butter(4, AV_LP, "low", fs=fs, output="sos")
    bp = butter(2, AMP_BAND, "band", fs=fs, output="sos")
    ang, amp = np.full(n, np.nan), np.full(n, np.nan)
    r0, r1 = runs(np.isfinite(ai))
    for s0, s1 in zip(r0, r1):
        if (s1 - s0) / fs < AV_MIN_RUN:
            continue
        ang[s0:s1] = sosfiltfilt(lp, ai[s0:s1])
        amp[s0:s1] = np.abs(hilbert(sosfiltfilt(bp, ai[s0:s1])))
    av = np.abs(np.gradient(ang, tw))
    return dict(t=tw, ang=ang, av=av, amp=amp, fs=fs, frac_ok=float(np.isfinite(av).mean()))


def av_measures(sid, t, SP, valid):
    """angular-velocity and amplitude measures in each spontaneous mask (camera samples mapped to the readout grid)"""
    A = angle_signals(sid)
    out = {}
    if A is None:
        return out, None
    gi = np.searchsorted(t, A["t"])
    inside = (gi > 0) & (gi < len(t))
    gi = np.clip(gi, 0, len(t) - 1)
    ok = inside & np.isfinite(A["av"])
    vmask = ok & valid[gi]
    med = np.median(A["av"][vmask])
    rsd = 1.4826 * np.median(np.abs(A["av"][vmask] - med))
    for sp_name, sp in SP.items():
        m = ok & sp[gi]
        if m.sum() < A["fs"] * 30:
            continue
        v, am = A["av"][m], A["amp"][m]
        d = dict(av_mean=v.mean(), av_median=np.median(v), av_p90=np.percentile(v, 90), av_p99=np.percentile(v, 99),
                 av_rel_frac=np.mean(v > med + m9.MOVE_K * rsd), av_rel_thr=med + m9.MOVE_K * rsd,
                 amp_median=np.nanmedian(am), amp_p90=np.nanpercentile(am, 90), setpoint_median=np.nanmedian(A["ang"][m]),
                 av_tracked_frac=A["frac_ok"])
        d.update({f"av_frac_{T}": np.mean(v > T) for T in AV_THRS})
        d.update({f"amp_frac_{T}": np.nanmean(am > T) for T in AMP_THRS})
        out[sp_name] = d
    return out, A


def trial_bounds(sid):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m1.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        tr = io.read().trials.to_dataframe()
    return tr.start_time.to_numpy(float), tr.stop_time.to_numpy(float)


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def session(sid, unit_set):
    out = m9.ROOT / unit_set / sid
    S = dict(np.load(out / "session.npz"))
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    P = __import__("json").loads((out / "params.json").read_text())
    t, valid, evk = S["t"], S["valid"], S["evoked"]
    st_, sp_ = trial_bounds(sid)
    in_trial = np.zeros(len(t), bool)
    for lo, hi in zip(np.searchsorted(t, st_), np.searchsorted(t, sp_, side="right")):
        in_trial[lo:hi] = True
    base = valid & ~evk
    nolick = S["lick_dt"] > LICK_FREE
    SP = {"iti": base & nolick & (S["epoch"] == 2) & ~in_trial}
    known = S["whisk_known"]
    tc = t - m1.WIDTH / 2
    wt, wv = B.get("whisker_speed_t", np.zeros(0)), B.get("whisker_speed_v", np.zeros(0))       # absent: no whisker tracking
    fin = np.isfinite(wv)
    spd = np.where(known, np.interp(tc, wt[fin], wv[fin]), np.nan) if fin.sum() > 100 else np.full(len(t), np.nan)
    thr = P.get("whisk_thr", np.nan)
    Z = {}
    for tg, sfx in RDS:
        f = out / f"g_All_{tg}{sfx}.npz"
        if not f.exists():
            continue
        z = np.load(f)["z"].astype(float)
        Z[(tg, sfx)] = (z - np.median(z[base])) / z[base].std()
    E = {}
    for tg, sfx in RDS:
        f = out / f"events_{tg}_All{sfx}.parquet"
        if f.exists():
            e = pd.read_parquet(f)
            E[(tg, sfx)] = np.clip(np.searchsorted(t, e.t_start.to_numpy()), 0, len(t) - 1)
    return dict(t=t, S=S, SP=SP, spd=spd, known=known, thr=thr, Z=Z, E=E, base=base)


def bouts(spd, thr, sp, step=m1.STEP):
    """bouts of raw speed > thr inside sp (gaps < MERGE_S merged, >= MIN_BOUT_S); returns onset / offset indices"""
    above = np.nan_to_num(spd, nan=0.0) > thr
    on, off = runs(above)
    if not len(on):
        return np.zeros(0, int), np.zeros(0, int)
    keep_on, keep_off = [on[0]], [off[0]]
    for a, b in zip(on[1:], off[1:]):
        if (a - keep_off[-1]) * step < MERGE_S:
            keep_off[-1] = b
        else:
            keep_on.append(a); keep_off.append(b)
    on, off = np.array(keep_on), np.array(keep_off)
    ok = ((off - on) * step >= MIN_BOUT_S) & sp[on]
    return on[ok], off[ok]


def measures(sid, coh, unit_set):
    Q = session(sid, unit_set)
    t, S, spd, known, thr = Q["t"], Q["S"], Q["spd"], Q["known"], Q["thr"]
    rows, curves = [], []
    step = m1.STEP
    AVM, _ = av_measures(sid, t, Q["SP"], S["valid"])
    for sp_name, sp in Q["SP"].items():
        spk = sp & known
        sec = spk.sum() * step
        d = dict(session_id=sid, cohort=coh, sp=sp_name, sp_min=sp.sum() * step / 60, whisk_known_frac=spk.sum() / max(sp.sum(), 1),
                 **AVM.get(sp_name, {}))
        if sec < 30 or not np.isfinite(thr):
            rows.append(d)
            continue
        w = S["whisk"] & spk
        s = spd[spk]
        d.update(frac_whisk=w.sum() / spk.sum(), speed_mean=np.nanmean(s), speed_median=np.nanmedian(s), speed_p90=np.nanpercentile(s, 90),
                 frac_fixed=np.mean(s > FIXED_DEG), whisk_thr=thr)
        on, off = bouts(spd, thr, spk)
        d.update(bout_rate_min=len(on) / (sec / 60), bout_dur_s=np.median((off - on) * step) if len(on) else np.nan)
        above = np.nan_to_num(spd, nan=np.inf) > thr                     # raw speed (not the dilated mask); unknown = not clean
        clean = np.array([not above[max(o - int(PRE_FREE_S / step), 0):o].any() for o in on], bool) if len(on) else np.zeros(0, bool)
        calm = ~above & sp                                                  # not whisking (raw speed), spontaneous
        lags = np.arange(int(BTA[0] / step), int(BTA[1] / step))
        for (tg, sfx), z in Q["Z"].items():
            k = f"{tg[0]}{RD_NAME[sfx][:3]}"
            nw = spk & ~S["whisk"]
            d[f"z_whisk_{k}"] = z[w].mean() if w.sum() > 200 else np.nan
            d[f"z_nowhisk_{k}"] = z[nw].mean() if nw.sum() > 200 else np.nan
            d[f"dz_whisk_{k}"] = d[f"z_whisk_{k}"] - d[f"z_nowhisk_{k}"]
            sub = np.flatnonzero(spk)[::10]
            d[f"rho_speed_{k}"] = stats.spearmanr(z[sub], spd[sub])[0] if len(sub) > 500 else np.nan
            if (tg, sfx) in Q["E"]:
                ei = Q["E"][(tg, sfx)]
                d[f"rate_whisk_{k}"] = w[ei].sum() / (w.sum() * step) if w.sum() * step > 10 else np.nan
                d[f"rate_nowhisk_{k}"] = nw[ei].sum() / (nw.sum() * step) if nw.sum() * step > 10 else np.nan
                d[f"rate_sp_{k}"] = spk[ei].sum() / sec
            # bout-triggered and post-bout still readout
            ob = on[clean]
            ob = ob[(ob + lags[0] >= 0) & (ob + lags[-1] < len(t))]
            if len(ob) >= 5:
                M = z[ob[:, None] + lags[None, :]]
                V = Q["base"][ob[:, None] + lags[None, :]]
                bta = np.nanmean(np.where(V, M, np.nan), 0)
                curves.append(dict(session_id=sid, cohort=coh, sp=sp_name, readout=k, n_bouts=len(ob), bta=bta))
            post, pre = [], []
            for a, b in zip(on[clean], off[clean]):
                m_post = slice(b, min(b + int(POST_S / step), len(t)))
                m_pre = slice(max(a - int(POST_S / step), 0), a)
                st_post, st_pre = calm[m_post], calm[m_pre]
                if st_post.sum() > 20 and st_pre.sum() > 20:
                    post.append(z[m_post][st_post].mean()); pre.append(z[m_pre][st_pre].mean())
            d[f"postbout_{k}"] = float(np.mean(np.array(post) - np.array(pre))) if len(post) >= 5 else np.nan
        for rd in ("dec", "cor"):
            d[f"spec_rho_{rd}"] = d.get(f"rho_speed_w{rd}", np.nan) - d.get(f"rho_speed_a{rd}", np.nan)
        rows.append(d)
    return rows, curves


def tests(F, cols):
    res = []
    for sp, q in F.groupby("sp"):
        for c in cols:
            if c not in q:
                continue
            a = q[q.cohort == "R+"][c].dropna().to_numpy(float)
            b = q[q.cohort == "R-"][c].dropna().to_numpy(float)
            if len(a) < 3 or len(b) < 3:
                continue
            res.append(dict(analysis="cohort", sp=sp, metric=c, n_rplus=len(a), n_rminus=len(b), median_rplus=np.median(a), median_rminus=np.median(b),
                            mean_rplus=a.mean(), mean_rminus=b.mean(), p_mannwhitney=stats.mannwhitneyu(a, b).pvalue,
                            p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue, **effect(a, b)))
    return res


def effect(a, b, n_boot=2000):
    """effect sizes R+ vs R-: Hedges g, rank-biserial r (MW: P(R+ > R-) - P(R+ < R-)), ratio of medians with 95 % bootstrap CI"""
    sp_ = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    g = (a.mean() - b.mean()) / sp_ * (1 - 3 / (4 * (len(a) + len(b)) - 9)) if sp_ > 0 else np.nan
    rb = 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1
    rng = np.random.default_rng(0)
    if (a > 0).all() and (b > 0).all():
        bs = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
        rat, lo, hi = np.median(a) / np.median(b), *np.percentile(bs, [2.5, 97.5])
    else:
        rat = lo = hi = np.nan
    return dict(hedges_g=g, rank_biserial=rb, ratio_medians=rat, ratio_ci_lo=lo, ratio_ci_hi=hi)


def strip(ax, q, col, title, log=False):
    for i, (coh, cl) in enumerate(COH.items()):
        qq = q[(q.cohort == coh) & q[col].notna()]
        v = qq[col].to_numpy(float)
        x = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        for mk_, sym in MARK.items():                                       # marker = experimenter (confound check)
            sel = qq.session_id.str.startswith(mk_).to_numpy()
            ax.plot(x[sel], v[sel], sym, ms=3, color=cl, mec="none")
        if len(v):
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1)
    a = q[q.cohort == "R+"][col].dropna(); b = q[q.cohort == "R-"][col].dropna()
    if len(a) >= 3 and len(b) >= 3:
        e = effect(a.to_numpy(float), b.to_numpy(float), 200)
        ax.set_title(f"{title}\np {stats.mannwhitneyu(a, b).pvalue:.2g} / {stats.ttest_ind(a, b, equal_var=False).pvalue:.2g}; g {e['hedges_g']:.2f}",
                     fontsize=5)
    else:
        ax.set_title(title, fontsize=5)
    ax.set_xticks([0, 1], [f"{c}\n{q[q.cohort == c][col].notna().sum()}" for c in COH], fontsize=5)
    ax.set_xlim(-0.6, 1.6)
    if log and (q[col].dropna() > 0).any():
        ax.set_yscale("log")
    ax.tick_params(labelsize=5)


def ols_panel(ax, q, x, y, res, sp, lab):
    import statsmodels.api as sm
    for coh, cl in COH.items():
        d = q[q.cohort == coh][[x, y]].dropna()
        ax.plot(d[x], d[y], "o", ms=3, color=cl, mec="none")
        if len(d) >= 4:
            X = sm.add_constant(d[x].to_numpy(float))
            f = sm.OLS(d[y].to_numpy(float), X).fit()
            xs = np.linspace(d[x].min(), d[x].max(), 50)
            pr = f.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
            ax.plot(xs, pr["mean"], color=cl, lw=1, ls="-" if f.pvalues[1] < 0.05 else "--")
            ax.fill_between(xs, pr["mean_ci_lower"], pr["mean_ci_upper"], color=cl, alpha=0.15, lw=0)
            res.append(dict(analysis="ols_cohort", sp=sp, metric=f"{y}~{x}", cohort=coh, n=len(d), slope=f.params[1], p_ols=f.pvalues[1]))
    d = q[[x, y, "cohort"]].dropna()
    if len(d) >= 8:
        import statsmodels.formula.api as smf
        f = smf.ols(f"{y} ~ {x} * C(cohort)", data=d).fit()
        pint = [p for k, p in f.pvalues.items() if ":" in k]
        res.append(dict(analysis="ols_pooled", sp=sp, metric=f"{y}~{x}*cohort", n=len(d), slope=f.params[x], p_ols=f.pvalues[x],
                        p_interaction=pint[0] if pint else np.nan))
        ax.set_title(f"{lab}\nslope p {f.pvalues[x]:.2g}, x cohort p {pint[0] if pint else np.nan:.2g}", fontsize=5)
    ax.tick_params(labelsize=5)


def fig_angvel(F, fdir, sp):
    """absolute angular velocity / amplitude, R+ vs R-: strips and fraction-above-threshold curves"""
    plt = m2.setup()
    q = F[F.sp == sp]
    if "av_mean" not in q:
        return
    fig = plt.figure(figsize=(7.4, 5.4), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1, 1])
    axs = sf[0].subplots(1, 5)
    for a, (c, lab, lg) in zip(axs, [("av_mean", "mean |angular velocity|\n(deg/s)", True), ("av_median", "median |angular\nvelocity| (deg/s)", True),
                                      ("av_p90", "90th pct |angular\nvelocity| (deg/s)", True), ("amp_median", "median whisking\namplitude (deg)", True),
                                      ("av_rel_frac", f"fraction > session median\n+ {m9.MOVE_K:g} robust s.d.", False)]):
        strip(a, q, c, lab, lg)
    axs = sf[1].subplots(1, 3)
    for a, (pre, thrs, unit) in zip(axs[:2], (("av_frac_", AV_THRS, "deg/s"), ("amp_frac_", AMP_THRS, "deg"))):
        for coh, cl in COH.items():
            M = q[q.cohort == coh][[f"{pre}{T}" for T in thrs]].dropna().to_numpy(float)
            if len(M):
                mu_, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
                a.errorbar(thrs, mu_, se, color=cl, marker="o", ms=3, lw=1, capsize=1.5, label=f"{coh} (n {len(M)})")
        ps = []
        for T in thrs:
            x_, y_ = q[q.cohort == "R+"][f"{pre}{T}"].dropna(), q[q.cohort == "R-"][f"{pre}{T}"].dropna()
            ps.append(f"{stats.mannwhitneyu(x_, y_).pvalue:.2g}" if len(x_) >= 3 and len(y_) >= 3 else "-")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks(thrs, [f"{T}" for T in thrs], fontsize=5)
        a.xaxis.set_minor_formatter(plt.NullFormatter())
        a.set_xlabel(f"threshold ({unit})", fontsize=5)
        a.set_ylabel("fraction of spontaneous time above", fontsize=5)
        a.set_title(("|angular velocity|" if unit == "deg/s" else "whisking amplitude") + "\nMW p: " + ", ".join(ps), fontsize=5)
        a.legend(fontsize=4.5, frameon=False)
        a.tick_params(labelsize=5)
    strip(axs[2], q, "setpoint_median", "median set point (deg;\ntracking geometry check)")
    fig.suptitle(f"017 E. whisking in absolute units ({sp}: valid task time outside trials and the evoked tail, > {LICK_FREE:g} s from licks), day 0. "
                 f"NWB whisker angle, DLC likelihood < {m1.LIK:g} removed, gaps <= {AV_GAP * 1000:.0f} ms interpolated, {AV_LP:g} Hz low-pass; "
                 f"angular velocity = d angle / dt; amplitude = Hilbert envelope of the {AMP_BAND[0]:g}-{AMP_BAND[1]:g} Hz angle. Circles AB, "
                 f"triangles MH mice; p Mann-Whitney / Welch, g = Hedges g (R+ - R-), uncorrected", fontsize=5.5, x=0.01, ha="left", wrap=True)
    m2.save(fig, fdir, f"fig_whisking_angvel_{sp}")
    plt.close(fig)


def figures(F, C, res, fdir, sp):
    plt = m2.setup()
    fig_angvel(F, fdir, sp)
    q = F[F.sp == sp]
    # amount
    cols = [("frac_whisk", "fraction of time whisking\n(session threshold)", False), ("frac_fixed", f"fraction > {FIXED_DEG:.0f} deg/s", False),
            ("speed_mean", "mean whisker speed (deg/s)", True), ("speed_median", "median whisker speed (deg/s)", True),
            ("bout_rate_min", "whisking bouts / min", False), ("bout_dur_s", "median bout duration (s)", False), ("whisk_thr", "session whisking\nthreshold (deg/s)", True),
            ("sp_min", "spontaneous minutes", False)]
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), layout="constrained")
    for a, (c, lab, lg) in zip(axs.ravel(), cols):
        strip(a, q, c, lab, lg)
    fig.suptitle(f"017 A. whisking in spontaneous periods ({sp}: valid, outside the evoked tail and > {LICK_FREE:g} s from licks), day 0; one point "
                 f"per session; p Mann-Whitney / Welch, uncorrected", fontsize=6, x=0.01, ha="left")
    m2.save(fig, fdir, f"fig_whisking_amount_{sp}")
    plt.close(fig)
    # readout vs whisking
    ks = [f"{tg[0]}{RD_NAME[sfx][:3]}" for tg, sfx in RDS]
    labs = {k: f"{'whisker' if k[0] == 'w' else 'auditory'} {RD_NAME['' if k.endswith('dec') else '_corr']}" for k in ks}
    fig, axs = plt.subplots(3, 4, figsize=(7.4, 5.6), layout="constrained")
    for j, k in enumerate(ks):
        strip(axs[0, j], q, f"dz_whisk_{k}", f"{labs[k]}: readout whisking -\nnot whisking (s.d.)")
        axs[0, j].axhline(0, color="0.6", lw=0.5, ls=":")
        strip(axs[1, j], q, f"rho_speed_{k}", f"{labs[k]}: Spearman rho\nreadout vs whisker speed")
        axs[1, j].axhline(0, color="0.6", lw=0.5, ls=":")
        strip(axs[2, j], q, f"postbout_{k}", f"{labs[k]}: non-whisking readout after -\nbefore a bout (s.d.)")
        axs[2, j].axhline(0, color="0.6", lw=0.5, ls=":")
    fig.suptitle(f"017 B. readout and whisking ({sp}), threshold-free (readout in session s.d. over valid time outside the evoked tail); "
                 f"bouts: speed > session threshold, >= {MIN_BOUT_S:g} s, {PRE_FREE_S:g} s below threshold before; non-whisking windows 0-{POST_S:g} s after vs "
                 f"{POST_S:g}-0 s before", fontsize=6, x=0.01, ha="left", wrap=True)
    m2.save(fig, fdir, f"fig_whisking_readout_{sp}")
    plt.close(fig)
    # bout-triggered averages
    lags = np.arange(int(BTA[0] / m1.STEP), int(BTA[1] / m1.STEP)) * m1.STEP
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.0), layout="constrained")
    cq = [c for c in C if c["sp"] == sp]
    for a, k in zip(axs, ks):
        for coh, cl in COH.items():
            M = np.array([c["bta"] for c in cq if c["readout"] == k and c["cohort"] == coh])
            if len(M):
                mu_, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(len(M))
                a.plot(lags, mu_, color=cl, lw=0.8, label=f"{coh} (n {len(M)})")
                a.fill_between(lags, mu_ - se, mu_ + se, color=cl, alpha=0.2, lw=0)
        a.axvline(0, color="0.6", lw=0.5, ls=":")
        a.set_title(labs[k], fontsize=5.5)
        a.set_xlabel("s from bout onset", fontsize=5)
        a.tick_params(labelsize=5)
    axs[0].set_ylabel("readout (session s.d.)", fontsize=5)
    axs[0].legend(fontsize=4.5, frameon=False)
    fig.suptitle(f"017 C. whisking-bout-triggered readout ({sp}), cohort mean +- sem over sessions", fontsize=6, x=0.01, ha="left")
    m2.save(fig, fdir, f"fig_whisking_bouts_{sp}")
    plt.close(fig)
    # across sessions
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.9), layout="constrained")
    for j, k in enumerate(ks):
        ols_panel(axs[0, j], q, "frac_whisk", f"rate_sp_{k}", res, sp, f"{labs[k]}: events / s vs\nfraction whisking")
        ols_panel(axs[1, j], q, "frac_whisk", f"z_whisk_{k}", res, sp, f"{labs[k]}: readout while whisking\nvs fraction whisking")
        axs[1, j].set_xlabel("fraction of spontaneous time whisking", fontsize=5)
    fig.suptitle(f"017 D. across sessions ({sp}): OLS per cohort (solid p < 0.05, band = 95 % CI); pooled y ~ whisking x cohort", fontsize=6, x=0.01, ha="left")
    m2.save(fig, fdir, f"fig_whisking_sessions_{sp}")
    plt.close(fig)


def main(ids, unit_set, tag):
    Sx = m9.sessions_table().set_index("session_id")
    rows, curves = [], []
    for sid in ids:
        coh = str(Sx.cohort.get(sid, "?"))
        r, c = measures(sid, coh, unit_set)
        rows += r
        curves += c
        print(sid, coh, flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(m1.HOME / f"analysis_017_whisking{tag}_sessions.csv", index=False)
    res = tests(F, [c for c in F.columns if c not in ("session_id", "cohort", "sp")])
    fdir = m1.HOME / "figures" / f"full_{unit_set}" / f"analysis_017_whisking{tag}"
    fdir.mkdir(parents=True, exist_ok=True)
    for sp in SPS:
        figures(F, curves, res, fdir, sp)
    T = pd.DataFrame(res)
    T.to_csv(m1.HOME / f"analysis_017_whisking{tag}_stats.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print(T.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--unit_set", default="stable")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main([l.strip() for l in open(a.sessions) if l.strip()], a.unit_set, a.tag)
