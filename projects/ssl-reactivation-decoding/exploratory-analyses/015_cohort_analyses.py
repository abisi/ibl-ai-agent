"""015 -- Large-scale R+ vs R- analyses of reactivation events from the 009 outputs (v5 2026-10-30; v6 shuffle threshold 2026-11-01).

Hypothesis: R+ mice show more reactivation events than R- mice (day 0, whisker decoder), nuanced by event type, content,
location relative to trials, change across the session, and relation to performance. Unit of analysis = session; R+ vs R-:
Mann-Whitney and Welch, uncorrected; all decoders (balanced accuracy < GOOD_BACC flagged, open markers); whole-brain decoder
primary, area-group decoders secondary. Threshold: 009 THR_MAIN (median over 100 rate-matched template shuffles of the
shuffled readout's 99.9th percentile). v7 (user 2026-11-01): --readout decoder (decision value) or corr (correlation with the
evoked template); outputs of the correlation readout carry the suffix _corr.
  A  three numbers per session (user 2026-11-01): total rate (events / s over all valid time; coarse primary), still rate (no
     movement, outside the evoked tail), movement-adjusted total rate (quasi-Poisson GLM: events ~ cohort + fraction of valid
     time whisking / jaw / lick / evoked / passive + balanced accuracy, offset log(valid seconds) -> R+ / R- rate ratio);
     also excess over the shuffled templates (real - mean of the 100 shuffles at the same threshold), all and still. fig30_primary
  B  movement: rates per movement mask (still / whisk / jaw / lick), fraction of events that are still; event rate aligned to
     whisking-bout onsets (speed > 200 deg/s after >= BOUT_GAP s below) and lick-bout onsets (first lick after >= LICK_GAP s).
                                                                                       fig31_movement, fig31b_movement_onsets
  C  content: fraction of events whose pattern matches the template better than rate-matched shuffled templates (identity
     p < 0.05, 009 / 007), fraction joint with the other decoder (its event within JOINT_S), rate of modality-specific still
     events.                                                                                                     fig32_content
  D  location relative to trials (task inter-trial time): still / all rate by the previous trial's type and outcome (W hit,
     W miss, A hit, A miss, CR, FA; R+ rewards W hits, R- does not); within-session W hit - W miss difference; rate vs time
     since the previous onset; quiet window (abort window) before W hits vs misses.                              fig33_trials
  E  change across the session: rate per 10-min bin of task time, late - early third; cohort x stage OLS interaction.
                                                                                                             fig34_timecourse
  F  performance: primary metric vs whisker hit rate, false-alarm rate, d' per cohort (OLS on the symlog scale + 95 % CI,
     solid if p < 0.05); trial level: per session logistic regression of the lick on whisker trials ~ still-event rate in the
     preceding inter-trial time + trial index, coefficient R+ vs R- and vs 0.                       fig35_performance_<stage>
  H  area groups: R+ - R- (median difference, symlog units) with p per area group decoder x metric x stage.  fig36_areas_<target>
  I  robustness: primary and still rate at the 99 / 99.5 / 99.9 / 99.99 shuffle thresholds, all vs good decoders. fig37_robustness
  G  HMM learning curves (ssl-learning-trial-identification 024_avg_curves_per_mouse.pkl, whisker day 0, sigma 1; source to be
     confirmed by the user): primary metric / still rate vs mean and final P(lick) whisker - FA per cohort; within session,
     Spearman of the still-event rate before each whisker trial vs the curve at that trial, R+ vs R- and vs 0.   fig38_learning_curves
Per-session tables are cached in <HOME>/analysis_015/<session>.pkl (--sessions); --figures aggregates. Stats in
<HOME>/analysis_015_stats_<unit_set>.csv.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m8 = importlib.import_module("008_sensitivity_threshold_cohorts")
m9 = importlib.import_module("009_full_dataset_events")
GROUP = m8.ALL
COH = m9.COH
OUT = m1.HOME / "analysis_015"
PRIMARY = ("rate", "events / s,\nall valid (total)")
SECONDARY = [("rate_still", "still events / s"), ("excess", "all valid: excess vs\nshuffled templates (/ s)"),
             ("excess_still", "still: excess vs\nshuffled templates (/ s)")]
SFX, PFX = "", ""                                # readout: "" / "" = decoder, "_corr" / "corr_" = correlation (set by --readout)
RD_TITLE = {"": "decoder", "_corr": "correlation", "_pcorr": "partial correlation | whisker velocity"}
ADJ = ["frac_whisk", "frac_jaw", "frac_lick", "frac_evoked", "frac_passive"]   # movement / context covariates of the adjusted GLM
PREV_CATS = ["W hit", "W miss", "A hit", "A miss", "CR", "FA"]
DT_EDGES = np.array([5, 7.5, 10, 15, 20, 30, 60])
TBIN = 600.0
BOUT_GAP, LICK_GAP = 1.0, 2.0
PSTH_HALF, PSTH_BIN = 3.0, 0.1
JOINT_S = 0.015
MIN_SECS = 30.0
STAGES2 = ("day 0", "expert")
CURVES = HERE.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "024_avg_curves_per_mouse.pkl"


# ------------------------------------------------------------------------------------------------ per-session tables
def load_trials(sid):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m1.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        tr = io.read().trials.to_dataframe()
    return tr.sort_values("start_time").reset_index(drop=True)


def trial_cats(tr):
    """category of every trial (all trials of the table): W / A hit-miss, CR, FA; passive; excluded (perf 6)"""
    tt, lf = tr.trial_type.to_numpy(), tr.lick_flag.to_numpy()
    pas = (m1.resolve_context(tr) == "passive").to_numpy()
    c = np.select([pas, tr.perf.to_numpy() == 6, (tt == "whisker_trial") & (lf == 1), tt == "whisker_trial",
                   (tt == "auditory_trial") & (lf == 1), tt == "auditory_trial", (tt == "no_stim_trial") & (lf == 0),
                   tt == "no_stim_trial"], ["passive", "excluded", "W hit", "W miss", "A hit", "A miss", "CR", "FA"], "other")
    return c


def cumsec(mask):
    return np.r_[0, np.cumsum(mask, dtype=np.int64)] * m1.STEP


def secs_between(cs, t, a, b):
    """seconds of a grid mask (cumulative cs) with window ends in (a, b]"""
    i, j = np.searchsorted(t, a, side="right"), np.searchsorted(t, b, side="right")
    return cs[j] - cs[i]


def bout_onsets(tv, vv, thr, gap):
    """onsets of supra-threshold periods preceded by >= gap s below threshold"""
    if tv is None or not len(tv):
        return np.zeros(0)
    above = np.nan_to_num(vv, nan=0) > thr
    on = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if not len(on):
        return np.zeros(0)
    last_above = np.maximum.accumulate(np.where(above, np.arange(len(above)), -1))
    prev = last_above[on - 1]
    ok = (prev < 0) | (tv[on] - tv[np.maximum(prev, 0)] >= gap)
    return tv[on[ok]]


def lick_onsets(licks, gap):
    licks = np.sort(np.asarray(licks, float))
    if not len(licks):
        return licks
    return licks[np.r_[True, np.diff(licks) >= gap]]


def psth(ev_t, onsets, t, cs_valid):
    """events / valid second in PSTH_BIN bins around onsets"""
    edges = np.arange(-PSTH_HALF, PSTH_HALF + 1e-9, PSTH_BIN)
    n = np.zeros(len(edges) - 1)
    s = np.zeros(len(edges) - 1)
    ev_t = np.sort(ev_t)
    for o in onsets:
        n += np.histogram(ev_t[(ev_t >= o + edges[0]) & (ev_t < o + edges[-1])] - o, edges)[0]
        s += secs_between(cs_valid, t, o + edges[:-1], o + edges[1:])
    return (edges[:-1] + edges[1:]) / 2, n, s


def session_tables(sid, unit_set):
    out = m9.ROOT / unit_set / sid
    S = dict(np.load(out / "session.npz"))
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    t, valid, still, pre = S["t"], S["valid"], S["still"] & S["valid"], S["prestart"] & S["valid"]
    cs_v, cs_s, cs_p = cumsec(valid), cumsec(still), cumsec(pre)
    E = {tg: pd.read_parquet(out / f"events_{tg}_{GROUP.replace(' ', '_')}{SFX}.parquet") for tg in m9.m6.TARGETS}
    tr = load_trials(sid)
    on = tr.start_time.to_numpy(float)
    cat = trial_cats(tr)
    R = dict(session_id=sid, prev=[], dt=[], prestart=[], tbin=[], content=[], psth=[], trial=[], beh={})
    # behaviour (kept active trials, 001 task_events rules)
    te = m1.task_events(tr)
    def rate_of(c, lick=True):
        m = te.cls.isin([c] if isinstance(c, str) else c)
        return float(te.licked[m].mean()) if m.any() else np.nan
    hw, fa, ha = rate_of("W"), float((te.cls == "FA").sum() / max(te.cls.isin(["FA", "N"]).sum(), 1)), rate_of("A")
    nW, nNS = int((te.cls == "W").sum()), int(te.cls.isin(["FA", "N"]).sum())
    from scipy.stats import norm
    cl = lambda p, n: np.clip(p, 1 / (2 * max(n, 1)), 1 - 1 / (2 * max(n, 1)))
    R["beh"] = dict(hit_w=hw, fa=fa, hit_a=ha, n_w=nW, n_ns=nNS,
                    dprime=float(norm.ppf(cl(hw, nW)) - norm.ppf(cl(fa, nNS))) if np.isfinite(hw) else np.nan)
    task_on = te.t.to_numpy(float)
    t0, t1 = (task_on.min(), task_on.max()) if len(task_on) else (np.nan, np.nan)
    for tg, ev in E.items():
        ev = ev.copy()
        tp = ev.t_peak.to_numpy(float)
        st = ev.still.to_numpy(bool)
        pi = np.searchsorted(on, tp, side="right") - 1                     # previous trial of each event
        # D1: previous-trial category
        gi_prev = np.searchsorted(on, t, side="right") - 1
        gcat = np.where(gi_prev >= 0, cat[np.maximum(gi_prev, 0)], "none")
        ecat = np.where(pi >= 0, cat[np.maximum(pi, 0)], "none")
        for c in PREV_CATS:
            gm = gcat == c
            R["prev"].append(dict(target=tg, cat=c, secs=float(valid[gm].sum() * m1.STEP), secs_still=float(still[gm].sum() * m1.STEP),
                                  n=int((ecat == c).sum()), n_still=int(((ecat == c) & st).sum())))
        # D2: time since the previous onset
        gdt = np.where(gi_prev >= 0, t - on[np.maximum(gi_prev, 0)], np.nan)
        edt = ev.dt_prev_onset.to_numpy(float)
        for a, b in zip(DT_EDGES[:-1], DT_EDGES[1:]):
            gm = (gdt >= a) & (gdt < b)
            em = (edt >= a) & (edt < b)
            R["dt"].append(dict(target=tg, dt_lo=a, dt_hi=b, secs=float(valid[gm].sum() * m1.STEP), secs_still=float(still[gm].sum() * m1.STEP),
                                n=int(em.sum()), n_still=int((em & st).sum())))
        # D3: quiet window (abort window) before the next trial, by its category
        gi_next = np.searchsorted(on, t, side="left")
        ncat = np.where(gi_next < len(on), cat[np.minimum(gi_next, len(on) - 1)], "none")
        ni = np.searchsorted(on, tp, side="left")
        encat = np.where(ni < len(on), cat[np.minimum(ni, len(on) - 1)], "none")
        epre = ev.prestart.to_numpy(bool)
        for c in PREV_CATS:
            gm = pre & (ncat == c)
            R["prestart"].append(dict(target=tg, cat=c, secs=float(gm.sum() * m1.STEP), n=int((epre & (encat == c)).sum()),
                                      n_still=int((epre & st & (encat == c)).sum()), secs_still=float((gm & still).sum() * m1.STEP)))
        # E: time course over the task (10-min bins from the first kept task trial), thirds
        if np.isfinite(t0):
            edges = np.arange(t0, t1 + TBIN, TBIN)
            for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                em = (tp > a) & (tp <= b)
                R["tbin"].append(dict(target=tg, kind="bin", k=k, t_lo=a - t0, secs=secs_between(cs_v, t, a, b), secs_still=secs_between(cs_s, t, a, b),
                                      n=int(em.sum()), n_still=int((em & st).sum())))
            th = np.linspace(t0, t1, 4)
            for k, (a, b) in enumerate(zip(th[:-1], th[1:])):
                em = (tp > a) & (tp <= b)
                R["tbin"].append(dict(target=tg, kind="third", k=k, t_lo=a - t0, secs=secs_between(cs_v, t, a, b), secs_still=secs_between(cs_s, t, a, b),
                                      n=int(em.sum()), n_still=int((em & st).sum())))
        # C: content
        ot = E["auditory" if tg == "whisker" else "whisker"]
        o0, o1 = np.sort(ot.t_start.to_numpy(float)) - JOINT_S, np.sort(ot.t_end.to_numpy(float)) + JOINT_S
        a0, a1 = ev.t_start.to_numpy(float), ev.t_end.to_numpy(float)
        # joint if any other-decoder event overlaps [a0, a1] (other extended by JOINT_S)
        j_ = np.searchsorted(o0, a1, side="right")                          # other events starting before this one ends
        joint = np.array([bool(j > 0 and np.max(o1[:j]) >= x) for j, x in zip(j_, a0)]) if len(ot) else np.zeros(len(ev), bool)
        idp = ev.p_identity.to_numpy(float) < 0.05
        ss = float(still.sum() * m1.STEP)
        R["content"].append(dict(target=tg, n=len(ev), n_still=int(st.sum()), frac_identity=float(idp.mean()) if len(ev) else np.nan,
                                 frac_identity_still=float(idp[st].mean()) if st.any() else np.nan,
                                 frac_joint=float(joint.mean()) if len(ev) else np.nan, frac_joint_still=float(joint[st].mean()) if st.any() else np.nan,
                                 rate_specific_still=float((st & ~joint).sum() / ss) if ss >= MIN_SECS else np.nan,
                                 frac_still=float(st.mean()) if len(ev) else np.nan))
        # B: event rate around whisking-bout and lick-bout onsets (onsets in valid time)
        ons = {}
        if "whisker_speed_t" in B:
            wthr = json.loads((out / "params.json").read_text()).get("whisk_thr") or m1.WHISK_MAX   # v7: per-session threshold
            ons["whisk bout"] = bout_onsets(B["whisker_speed_t"], B["whisker_speed_v"], wthr, BOUT_GAP)
        ons["lick bout"] = lick_onsets(B["licks"], LICK_GAP)
        for nm, o in ons.items():
            o = o[valid[np.clip(np.searchsorted(t, o), 0, len(t) - 1)]]
            if len(o) < 5:
                continue
            x, n_, s_ = psth(tp, o, t, cs_v)
            R["psth"].append(dict(target=tg, align=nm, n_onsets=len(o), x=x, n=n_, s=s_))
        # F (trial level): lick on kept W / no-stim trials ~ still-event rate in the preceding inter-trial time
        tj = te.t.to_numpy(float)
        prv = np.r_[np.nan, on][np.searchsorted(on, tj, side="left")]       # previous onset of any trial
        for j, (a, b) in enumerate(zip(prv, tj)):
            if not np.isfinite(a):
                continue
            em = (tp > a) & (tp <= b)
            R["trial"].append(dict(target=tg, idx=j, cls=te.cls.iloc[j], licked=bool(te.licked.iloc[j]), secs=secs_between(cs_v, t, a, b),
                                   secs_still=secs_between(cs_s, t, a, b), n=int(em.sum()), n_still=int((em & st).sum())))
    for k in ("prev", "dt", "prestart", "tbin", "content", "trial"):
        R[k] = pd.DataFrame(R[k]).assign(session_id=sid)
    return R


def run_sessions(sessions, unit_set):
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in sessions:
        try:
            R = session_tables(sid, unit_set)
            with open(OUT / f"{sid}_{unit_set}{SFX}.pkl", "wb") as f:
                pickle.dump(R, f)
            print("DONE", sid, flush=True)
        except Exception as e:
            import traceback
            print("FAILED", sid, repr(e), flush=True)
            traceback.print_exc()


# ------------------------------------------------------------------------------------------------ helpers
STATS = []


def test(a, b, **meta):
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    ok = len(a) >= m9.MIN_SESS and len(b) >= m9.MIN_SESS
    p_mw, p_t = m8.two_tests(a, b) if ok else (np.nan, np.nan)
    STATS.append(dict(meta, n_rplus=len(a), n_rminus=len(b), median_rplus=a.median(), median_rminus=b.median(), mean_rplus=a.mean(),
                      mean_rminus=b.mean(), p_mannwhitney=p_mw, p_welch=p_t, tested=ok))
    return p_mw, p_t


def ptxt(p_mw, p_t):
    return "" if not np.isfinite(p_mw) else f"p {p_mw:.2g} / {p_t:.2g}" + ("*" if p_mw < 0.05 and p_t < 0.05 else "")


def strip(ax, q, mt, sym=True, meta=None):
    """R+ vs R- strip (all decoders; open markers = balanced accuracy < GOOD_BACC), median bars, MW / Welch"""
    vals = []
    for x_, (coh, cl) in enumerate(COH.items()):
        s_ = q[q.cohort == coh]
        v = s_[mt].to_numpy(float)
        good = s_.bacc.to_numpy(float) >= m9.GOOD_BACC if "bacc" in s_ else np.ones(len(v), bool)
        jit = np.random.default_rng(x_).uniform(-0.17, 0.17, len(v))
        ax.scatter(x_ + jit, m9.symt(v) if sym else v, s=5, lw=0.4, edgecolors=cl, facecolors=np.where(good, cl, "w"), zorder=3)
        if np.isfinite(v).any():
            md = np.nanmedian(v)
            ax.plot([x_ - 0.28, x_ + 0.28], [m9.symt(md) if sym else md] * 2, color="k", lw=0.9, zorder=4)
        vals.append(v)
    p = test(vals[0], vals[1], metric=mt, **(meta or {}))
    ax.text(0.5, 1.0, ptxt(*p), transform=ax.transAxes, ha="center", va="bottom", fontsize=4)
    ax.set_xticks([0, 1], [f"{c}\n{np.isfinite(v).sum()}" for c, v in zip(COH, vals)], fontsize=4.2)
    ax.set_xlim(-0.6, 1.6)
    if sym:
        lo = min([np.nanmin(v) for v in vals if np.isfinite(v).any()] + [0])
        tk = [v for v in (-1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10) if v >= min(lo, 0) - 1e-12 and (lo < 0 or v >= 0)]
        ax.set_yticks(m9.symt(tk), [f"{v:g}" for v in tk])
        ax.axhline(0, color="0.7", lw=0.3, ls=":")
    ax.tick_params(labelsize=4.2)
    return p


def mean_sem(ax, x, Y, color, label=None, ls="-"):
    Y = np.asarray(Y, float)
    if not len(Y):
        return
    mu = np.nanmean(Y, 0)
    se = np.nanstd(Y, 0) / np.sqrt(np.maximum(np.isfinite(Y).sum(0), 1))
    ax.plot(x, mu, color=color, lw=0.8, ls=ls, label=label)
    ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.2, lw=0)


def corr_panel(ax, q, xcol, ycol, meta):
    from scipy import stats
    txt = []
    for coh, cl in COH.items():
        d = q[q.cohort == coh][[xcol, ycol]].dropna()
        x, y = d[xcol].to_numpy(float), m9.symt(d[ycol])
        ax.scatter(x, y, s=5, color=cl, lw=0, alpha=0.8)
        if len(d) >= m9.MIN_SESS and np.ptp(x) > 0:
            xs = np.linspace(x.min(), x.max(), 50)
            yh, ci, p = m9.ols_band(x, y, xs)
            ax.plot(xs, yh, color=cl, lw=0.8, ls="-" if p < 0.05 else "--")
            ax.fill_between(xs, yh - ci, yh + ci, color=cl, alpha=0.15, lw=0)
            rho, pr = stats.spearmanr(x, d[ycol])
            txt.append(f"{coh} rho {rho:.2f} (p {pr:.2g}), OLS p {p:.2g}")
            STATS.append(dict(meta, xcol=xcol, metric=ycol, cohort=coh, n=len(d), spearman_rho=rho, spearman_p=pr, ols_p=p))
    ax.text(0.01, 0.99, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=3.6)
    tk = [-0.1, -0.01, 0, 0.01, 0.1, 1]
    ax.set_yticks(m9.symt(tk), [f"{v:g}" for v in tk])
    ax.tick_params(labelsize=4.2)


# ------------------------------------------------------------------------------------------------ figures
def load_all(unit_set, sessions):
    R = {}
    for sid in sessions:
        f = OUT / f"{sid}_{unit_set}{SFX}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                R[sid] = pickle.load(fh)
    return R


def cat_table(R, key, info):
    D = pd.concat([r[key] for r in R.values() if len(r[key])], ignore_index=True)
    return D.merge(info, on="session_id", how="left")


def glm_ratio(q):
    """quasi-Poisson: still events ~ cohort + bacc, offset log(null expected); returns R+/R- ratio, CI, p"""
    import statsmodels.api as sm
    d = q.dropna(subset=["n_still", "null_exp_still", "bacc"])
    d = d[d.secs_still >= MIN_SECS]
    if min((d.cohort == c).sum() for c in COH) < m9.MIN_SESS:
        return np.nan, (np.nan, np.nan), np.nan
    X = sm.add_constant(np.c_[(d.cohort == "R+").to_numpy(float), d.bacc.to_numpy(float)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(d.n_still.to_numpy(float), X, family=sm.families.Poisson(), offset=np.log(np.maximum(d.null_exp_still, 0.5))).fit(scale="X2")
    b, se = f.params[1], f.bse[1]
    return float(np.exp(b)), (float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))), float(f.pvalues[1])


def glm_adjusted(q):
    """quasi-Poisson: events (all valid time) ~ cohort + ADJ fractions + bacc, offset log(valid seconds); R+/R- ratio, CI, p"""
    import statsmodels.api as sm
    d = q.dropna(subset=["n_events", "secs", "bacc"] + ADJ)
    d = d[d.secs >= MIN_SECS]
    if min((d.cohort == c).sum() for c in COH) < m9.MIN_SESS:
        return np.nan, (np.nan, np.nan), np.nan
    X = sm.add_constant(np.c_[(d.cohort == "R+").to_numpy(float), d[ADJ + ["bacc"]].to_numpy(float)], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(d.n_events.to_numpy(float), X, family=sm.families.Poisson(), offset=np.log(d.secs.to_numpy(float))).fit(scale="X2")
    b, se = f.params[1], f.bse[1]
    return float(np.exp(b)), (float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))), float(f.pvalues[1])


CONDS = [("qw", "quiet window\n(no lick)"), ("qw_still", "quiet window,\nno movement"), ("iti", "ITI\n(movement allowed)"),
         ("iti_still", "ITI,\nno movement")]
TUNE_LAB = {"wz": ("whisker speed (robust z)", [f"{a:g}-{b:g}".replace("-inf-", "<").replace("-inf", "+") for a, b in
                                                 zip(m9.Z_EDGES[:-1], m9.Z_EDGES[1:])]),
            "jz": ("jaw speed (robust z)", [f"{a:g}-{b:g}".replace("-inf-", "<").replace("-inf", "+") for a, b in
                                             zip(m9.Z_EDGES[:-1], m9.Z_EDGES[1:])]),
            "lk": ("time to nearest lick (s)", [f"{a:g}-{b:g}".replace("-inf", "+") for a, b in zip(m9.LICK_EDGES[:-1], m9.LICK_EDGES[1:])])}


def tuning(plt, W, fdir):
    """K (user 2026-11-01): event rate as a function of movement variables (009 tuning masks: whisker / jaw speed robust-z bins,
    time to the nearest lick; outside the evoked tail; event = its first window): cohort mean +- sem of log10 rate, real (solid)
    vs shuffled templates (dashed); second row: excess over the shuffles; R+ vs R- per bin (MW / Welch) in the stats table"""
    for stg in STAGES2:
        qs_ = W[W.st == stg]
        if qs_.empty:
            continue
        fig, axs = plt.subplots(4, 3, figsize=(7.4, 6.4), squeeze=False)
        for i, tg in enumerate(m9.m6.TARGETS):
            q = qs_[qs_.target == tg]
            for j, (key, (lab, ticks)) in enumerate(TUNE_LAB.items()):
                cols = [f"{key}{b}" for b in range(m9.TUNE[key])]
                if not all(f"rate_{c}" in q for c in cols):
                    axs[2 * i, j].axis("off")
                    axs[2 * i + 1, j].axis("off")
                    continue
                x = np.arange(len(cols))
                for coh, cl in COH.items():
                    d = q[q.cohort == coh]
                    for pre_, ls, nm in (("rate_", "-", "real"), ("rate_shuf_", "--", "shuffles")):
                        mean_sem(axs[2 * i, j], x, np.log10(np.maximum(d[[pre_ + c for c in cols]].to_numpy(float), 1e-4)), cl,
                                 f"{coh} {nm} (n {len(d)})", ls)
                    mean_sem(axs[2 * i + 1, j], x, d[[f"excess_{c}" for c in cols]].to_numpy(float), cl, coh)
                pt = []
                for c in cols:
                    p_ = test(q[q.cohort == "R+"][f"excess_{c}"], q[q.cohort == "R-"][f"excess_{c}"], analysis="K_tuning", target=tg,
                              stage=stg, metric=f"excess_{c}")
                    test(q[q.cohort == "R+"][f"rate_{c}"], q[q.cohort == "R-"][f"rate_{c}"], analysis="K_tuning", target=tg, stage=stg,
                         metric=f"rate_{c}")
                    pt.append("*" if (p_[0] < 0.05 and p_[1] < 0.05) else "")
                for a, yl in ((axs[2 * i, j], "log10 events / s"), (axs[2 * i + 1, j], "excess vs shuffles (/ s)")):
                    a.set_xticks(x, [f"{t}{s}" for t, s in zip(ticks, pt)], fontsize=3.8, rotation=30)
                    a.tick_params(labelsize=4.2)
                    if j == 0:
                        a.set_ylabel(f"{tg}\n{yl}", fontsize=4.6)
                axs[2 * i + 1, j].axhline(0, color="0.7", lw=0.3, ls=":")
                axs[2 * i + 1, j].set_xlabel(lab, fontsize=4.6)
                if i == 0:
                    axs[0, j].set_title(lab, fontsize=5)
                if i == 0 and j == 0:
                    axs[0, 0].legend(frameon=False, fontsize=3.6)
        fig.suptitle(f"K. {stg}: event rate vs movement ({RD_TITLE[SFX]} readout, whole brain), outside the evoked "
                     f"tail; speeds in robust z of each session's speed over valid time (movement threshold z = {m9.MOVE_K:g}); cohort mean "
                     f"+- sem; * = R+ vs R- excess, MW and Welch p < 0.05 (uncorrected)", x=0.01, ha="left", fontsize=5, weight="bold", wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        m2.save(fig, fdir, f"fig40_movement_tuning_{stg.replace(' ', '')}")
        plt.close(fig)


def conditions(plt, W, fdir):
    """J (user 2026-11-01): R+ vs R- per behavioural condition (009 COND, overlapping: quiet window, quiet window without
    movement, ITI, ITI without movement): rate and excess over the shuffled templates; cohort profiles across conditions"""
    for stg in STAGES2:
        qs_ = W[W.st == stg]
        if qs_.empty:
            continue
        for kind, pre_, yl in (("rate", "rate_", "events / s"), ("excess", "excess_", "excess vs shuffles (/ s)")):
            fig, axs = plt.subplots(2, len(CONDS), figsize=(7.4, 3.4), squeeze=False)
            for i, tg in enumerate(m9.m6.TARGETS):
                q = qs_[qs_.target == tg]
                for j, (c, lab) in enumerate(CONDS):
                    col = f"{pre_}{c}"
                    if col not in q or not q[col].notna().any():
                        axs[i, j].axis("off")
                        continue
                    strip(axs[i, j], q, col, meta=dict(analysis=f"J_{kind}", target=tg, stage=stg, condition=c))
                    if i == 0:
                        axs[i, j].set_title(lab, fontsize=4.6, pad=8)
                axs[i, 0].set_ylabel(f"{tg}\n{yl}", fontsize=4.6)
            fig.suptitle(f"J. {stg}: R+ vs R- by behavioural condition ({RD_TITLE[SFX]} readout, whole brain; one point "
                         f"per session, open: balanced accuracy < {m9.GOOD_BACC}); quiet window = task abort window with no lick within "
                         f"{m9.LICK_MARGIN:g} s; ITI = task time outside trials (start..stop) and the evoked tail, includes the quiet windows; "
                         f"no movement = no whisking / jaw movement (speed > median + {m9.MOVE_K:g} robust s.d., per session and body part) "
                         f"and, for the ITI, no lick; {kind} = "
                         f"{'events / s of condition time' if kind == 'rate' else 'real - mean of 100 shuffled templates at the same threshold'}; "
                         f"p Mann-Whitney / Welch, uncorrected", x=0.01, ha="left", fontsize=5, weight="bold", wrap=True)
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            m2.save(fig, fdir, f"fig39_conditions_{kind}_{stg.replace(' ', '')}")
            plt.close(fig)
        # profiles: cohort mean +- sem of log10 rate per condition, real (solid) vs shuffles (dashed)
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.4), squeeze=False)
        for i, tg in enumerate(m9.m6.TARGETS):
            ax = axs[0, i]
            q = qs_[qs_.target == tg]
            cs = [c for c, _ in CONDS if f"rate_{c}" in q]
            for coh, cl in COH.items():
                d = q[q.cohort == coh]
                for pre_, ls, lab in (("rate_", "-", "real"), ("rate_shuf_", "--", "shuffles")):
                    Y = np.log10(np.maximum(d[[f"{pre_}{c}" for c in cs]].to_numpy(float), 1e-4))
                    mean_sem(ax, np.arange(len(cs)), Y, cl, f"{coh} {lab} (n {len(d)})", ls)
            ax.set_xticks(range(len(cs)), [dict(CONDS)[c].replace("\n", " ") for c in cs], fontsize=4.2, rotation=35, ha="right")
            ax.set_ylabel("log10 events / s (floor 1e-4)", fontsize=4.6)
            ax.set_title(f"{tg}: cohort mean +- sem across conditions", fontsize=5)
            ax.legend(frameon=False, fontsize=3.8)
            ax.tick_params(labelsize=4.2)
        fig.suptitle(f"J. {stg}: event rate profiles across behavioural conditions, real vs shuffled templates", x=0.01, ha="left",
                     fontsize=5.2, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        m2.save(fig, fdir, f"fig39_condition_profiles_{stg.replace(' ', '')}")
        plt.close(fig)


def figures(unit_set, dec_csv=None, tag=""):
    plt = m2.setup()
    fdir = m1.HOME / "figures" / f"full_{unit_set}" / f"analysis_015{tag}{SFX}"
    Dd = pd.read_csv(dec_csv or m1.HOME / f"full_decoders_{unit_set}.csv")
    Dm = Dd[Dd.thr_source == PFX + m9.THR_MAIN].copy()
    import json
    secs = {}
    for sid in Dm.session_id.unique():
        P = json.loads((m9.ROOT / unit_set / sid / "params.json").read_text())
        secs[sid] = P["secs"]
    Dm["secs_still"] = Dm.session_id.map(lambda s: secs[s].get("still", np.nan))
    Dm["n_still"] = (Dm.rate_still * Dm.secs_still).round()
    Dm["null_exp_still"] = Dm.rate_shuf_still * Dm.secs_still                # v6: shuffled-template expected count
    Dm["st"] = np.where(Dm.stage.str.startswith("day 0"), "day 0", "expert")
    W = Dm[Dm.group == GROUP]
    info = W.drop_duplicates(["session_id", "target"])[["session_id", "target", "cohort", "st", "bacc"]]
    info_s = W.drop_duplicates("session_id")[["session_id", "cohort", "st"]]
    R = load_all(unit_set, W.session_id.unique())
    print(f"{len(R)} sessions with 015 tables / {W.session_id.nunique()}", flush=True)

    # ---------------- A: primary
    fig, axs = plt.subplots(4, 4, figsize=(7.4, 6.4), squeeze=False)
    for i, (tg, stg) in enumerate([(t_, s_) for t_ in m9.m6.TARGETS for s_ in STAGES2]):
        q = W[(W.target == tg) & (W.st == stg)]
        for j, (mt, lab) in enumerate([PRIMARY] + SECONDARY):
            strip(axs[i, j], q, mt, meta=dict(analysis="A", target=tg, stage=stg, group=GROUP))
            if i == 0:
                axs[i, j].set_title(lab, fontsize=5, pad=8)
        rr, ci, p = glm_adjusted(q)
        STATS.append(dict(analysis="A_glm_adjusted", target=tg, stage=stg, metric="events ~ cohort + movement / context fractions + bacc",
                          ratio_rplus_rminus=rr, ci_lo=ci[0], ci_hi=ci[1], p_glm=p))
        r2, ci2, p2 = glm_ratio(q)
        STATS.append(dict(analysis="A_glm_still_vs_shuffle", target=tg, stage=stg, metric="still events / shuffled templates",
                          ratio_rplus_rminus=r2, ci_lo=ci2[0], ci_hi=ci2[1], p_glm=p2))
        axs[i, 0].set_ylabel(f"{tg}, {stg}\nmovement-adjusted R+/R- {rr:.2f}\n[{ci[0]:.2f}, {ci[1]:.2f}], p {p:.2g}", fontsize=4.6)
    fig.suptitle(f"A. R+ vs R-, whole-brain decoders ({unit_set} units), one point per session (open: balanced accuracy < {m9.GOOD_BACC}); "
                 f"{RD_TITLE[SFX]} readout; {m9.THR_TEXT}; primary = total rate (all valid time); excess = real - mean of the "
                 f"shuffled templates at the same threshold; p Mann-Whitney / Welch, uncorrected; movement-adjusted: quasi-Poisson events ~ "
                 f"cohort + fraction of valid time whisking / jaw / lick / evoked / passive + balanced accuracy, offset log(valid s)", x=0.01, ha="left",
                 fontsize=5, weight="bold", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    m2.save(fig, fdir, "fig30_primary")
    plt.close(fig)
    conditions(plt, W, fdir)
    tuning(plt, W, fdir)

    # ---------------- B: movement
    mcols = [("rate_still", "still"), ("rate_whisk", "whisking"), ("rate_jaw", "jaw movement"), ("rate_lick", "licking"), ("frac_ev_still", "fraction of\nevents still")]
    C = cat_table(R, "content", info_s).merge(info[["session_id", "target", "bacc"]], on=["session_id", "target"], how="left")
    Wc = W.merge(C[["session_id", "target", "frac_still"]].rename(columns={"frac_still": "frac_ev_still"}), on=["session_id", "target"], how="left")   # W.frac_still = still fraction of valid time
    fig, axs = plt.subplots(4, len(mcols), figsize=(7.4, 6.0), squeeze=False)
    for i, (tg, stg) in enumerate([(t_, s_) for t_ in m9.m6.TARGETS for s_ in STAGES2]):
        q = Wc[(Wc.target == tg) & (Wc.st == stg)]
        for j, (mt, lab) in enumerate(mcols):
            strip(axs[i, j], q, mt, sym=mt != "frac_ev_still", meta=dict(analysis="B", target=tg, stage=stg))
            if i == 0:
                axs[i, j].set_title(f"events / s, {lab}" if mt.startswith("rate") else lab, fontsize=5, pad=8)
        axs[i, 0].set_ylabel(f"{tg}, {stg}", fontsize=5)
    fig.suptitle(f"B. events by movement (009 masks: whisking or jaw movement (speed > median + {m9.MOVE_K:g} robust s.d.) within {m9.MOVE_MARGIN * 1000:.0f} ms, lick within "
                 f"{m9.LICK_MARGIN:.0f} s); whole brain; rate = events / s of time in that state", x=0.01, ha="left", fontsize=5.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    m2.save(fig, fdir, "fig31_movement")
    plt.close(fig)
    rows = [dict(r, session_id=sid) for sid, Rs in R.items() for r in Rs["psth"]]
    if rows:
        Pt = pd.DataFrame(rows).merge(info_s, on="session_id")
        fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.4), squeeze=False)
        for i, tg in enumerate(m9.m6.TARGETS):
            for j, (al, stg) in enumerate([(a_, s_) for a_ in ("whisk bout", "lick bout") for s_ in STAGES2]):
                ax = axs[i, j]
                for coh, cl in COH.items():
                    d = Pt[(Pt.target == tg) & (Pt["align"] == al) & (Pt.st == stg) & (Pt.cohort == coh)]
                    Y = [np.where(s > 1, n / np.maximum(s, 1e-9), np.nan) for n, s in zip(d.n, d.s)]
                    if len(Y):
                        mean_sem(ax, d.x.iloc[0], Y, cl, f"{coh} ({len(Y)})")
                ax.axvline(0, color="0.6", lw=0.4, ls=":")
                ax.set_title(f"{tg}, {stg}: {al} onset", fontsize=5)
                ax.legend(frameon=False, fontsize=3.8)
                ax.tick_params(labelsize=4.2)
                if i == 1:
                    ax.set_xlabel("time from onset (s)", fontsize=4.6)
            axs[i, 0].set_ylabel("events / valid s", fontsize=4.6)
        fig.suptitle(f"B. event rate around whisking-bout onsets (above the session movement threshold after >= {BOUT_GAP:.0f} s below) and lick-bout "
                     f"onsets (first lick after >= {LICK_GAP:.0f} s); valid time only; mean +- s.e.m. over sessions", x=0.01, ha="left", fontsize=5.2, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        m2.save(fig, fdir, "fig31b_movement_onsets")
        plt.close(fig)

    # ---------------- C: content
    ccols = [("frac_identity", "identity p < 0.05,\nall events"), ("frac_identity_still", "identity p < 0.05,\nstill events"),
             ("frac_joint", "joint with the other\ndecoder, all"), ("frac_joint_still", "joint with the other\ndecoder, still"),
             ("rate_specific_still", "modality-specific\nstill events / s")]
    fig, axs = plt.subplots(4, len(ccols), figsize=(7.4, 6.0), squeeze=False)
    for i, (tg, stg) in enumerate([(t_, s_) for t_ in m9.m6.TARGETS for s_ in STAGES2]):
        q = C[(C.target == tg) & (C.st == stg)]
        for j, (mt, lab) in enumerate(ccols):
            strip(axs[i, j], q, mt, sym=mt.startswith("rate"), meta=dict(analysis="C", target=tg, stage=stg))
            if i == 0:
                axs[i, j].set_title(lab, fontsize=5, pad=8)
        axs[i, 0].set_ylabel(f"{tg}, {stg}", fontsize=5)
    fig.suptitle(f"C. event content, whole brain: identity = event pattern matches the template better than {m9.m7.N_PERM} rate-matched "
                 f"shuffled templates (p < 0.05); joint = the other decoder has an event within {JOINT_S * 1000:.0f} ms", x=0.01, ha="left",
                 fontsize=5.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    m2.save(fig, fdir, "fig32_content")
    plt.close(fig)

    # ---------------- D: trials
    Pv = cat_table(R, "prev", info_s)
    Pv["rate_still"] = np.where(Pv.secs_still >= MIN_SECS, Pv.n_still / Pv.secs_still.clip(lower=1e-9), np.nan)
    Pv["rate"] = np.where(Pv.secs >= MIN_SECS, Pv.n / Pv.secs.clip(lower=1e-9), np.nan)
    Dt = cat_table(R, "dt", info_s)
    Dt["rate_still"] = np.where(Dt.secs_still >= 10, Dt.n_still / Dt.secs_still.clip(lower=1e-9), np.nan)
    Dt["rate"] = np.where(Dt.secs >= 10, Dt.n / Dt.secs.clip(lower=1e-9), np.nan)
    Ps = cat_table(R, "prestart", info_s)
    Ps["rate"] = np.where(Ps.secs >= 10, Ps.n / Ps.secs.clip(lower=1e-9), np.nan)
    for stg in STAGES2:
        fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), squeeze=False)
        for i, tg in enumerate(m9.m6.TARGETS):
            ax = axs[i, 0]
            for k, coh in enumerate(COH):
                d = Pv[(Pv.target == tg) & (Pv.st == stg) & (Pv.cohort == coh)]
                for c_i, c in enumerate(PREV_CATS):
                    v = d[d.cat == c].rate_still.dropna()
                    x = c_i + (k - 0.5) * 0.35
                    ax.scatter(x + np.random.default_rng(c_i).uniform(-0.07, 0.07, len(v)), m9.symt(v), s=3, color=COH[coh], lw=0, alpha=0.7)
                    if len(v):
                        ax.plot([x - 0.12, x + 0.12], [m9.symt(v.median())] * 2, color="k", lw=0.8)
            labs = []
            for c in PREV_CATS:
                a_ = Pv[(Pv.target == tg) & (Pv.st == stg) & (Pv.cohort == "R+") & (Pv.cat == c)].rate_still
                b_ = Pv[(Pv.target == tg) & (Pv.st == stg) & (Pv.cohort == "R-") & (Pv.cat == c)].rate_still
                p_ = test(a_, b_, analysis="D_prev", target=tg, stage=stg, metric=f"rate_still after {c}")
                labs.append(f"{c}\n{p_[0]:.2g}/{p_[1]:.2g}" if np.isfinite(p_[0]) else c)
            ax.set_xticks(range(len(PREV_CATS)), labs, fontsize=3.6)
            tk = [0, 0.01, 0.1, 1]
            ax.set_yticks(m9.symt(tk), [f"{v:g}" for v in tk])
            ax.set_ylabel(f"{tg}\nstill events / s", fontsize=4.8)
            ax.set_title("by previous trial (R+ left, R- right)", fontsize=5)
            # W hit - W miss within session
            d = Pv[(Pv.target == tg) & (Pv.st == stg)].pivot_table(index=["session_id", "cohort"], columns="cat", values="rate_still").reset_index()
            if {"W hit", "W miss"} <= set(d.columns):
                d["hit_minus_miss"] = d["W hit"] - d["W miss"]
                strip(axs[i, 1], d.assign(bacc=1.0), "hit_minus_miss", meta=dict(analysis="D_hitmiss", target=tg, stage=stg))
            axs[i, 1].set_title("still rate after W hit - after W miss\n(R+: hit rewarded, R-: not)", fontsize=5)
            ax = axs[i, 2]
            for coh, cl in COH.items():
                d = Dt[(Dt.target == tg) & (Dt.st == stg) & (Dt.cohort == coh)]
                Y = d.pivot_table(index="session_id", columns="dt_lo", values="rate_still").to_numpy()
                if len(Y):
                    mean_sem(ax, DT_EDGES[:-1], Y, cl, coh)
            ax.set_xlabel("time since the previous trial onset (s, bin start)", fontsize=4.6)
            ax.set_title("still events / s vs time since trial", fontsize=5)
            ax.legend(frameon=False, fontsize=3.8)
            ax.tick_params(labelsize=4.2)
            d = Ps[(Ps.target == tg) & (Ps.st == stg)].pivot_table(index=["session_id", "cohort"], columns="cat", values="rate").reset_index()
            if {"W hit", "W miss"} <= set(d.columns):
                d["pre_hit_minus_miss"] = d["W hit"] - d["W miss"]
                strip(axs[i, 3], d.assign(bacc=1.0), "pre_hit_minus_miss", meta=dict(analysis="D_prestart", target=tg, stage=stg))
            axs[i, 3].set_title("quiet window before W hit - before W miss\n(events / s, all events)", fontsize=5)
        fig.suptitle(f"D. {stg}: events relative to trials (task inter-trial time, outside trial onset - 10 ms .. + 5 s), whole brain; per "
                     f"session rates (>= {MIN_SECS:.0f} s of still time per category); p R+ vs R- Mann-Whitney / Welch, uncorrected",
                     x=0.01, ha="left", fontsize=5.2, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        m2.save(fig, fdir, f"fig33_trials_{stg.replace(' ', '')}")
        plt.close(fig)

    # ---------------- E: time course
    Tb = cat_table(R, "tbin", info_s)
    Tb["rate_still"] = np.where(Tb.secs_still >= MIN_SECS, Tb.n_still / Tb.secs_still.clip(lower=1e-9), np.nan)
    Tb["rate"] = np.where(Tb.secs >= MIN_SECS, Tb.n / Tb.secs.clip(lower=1e-9), np.nan)
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), squeeze=False)
    import statsmodels.formula.api as smf
    for i, tg in enumerate(m9.m6.TARGETS):
        for j, (mt, stg) in enumerate([("rate_still", "day 0"), ("rate", "day 0")]):
            ax = axs[i, j]
            for coh, cl in COH.items():
                d = Tb[(Tb.target == tg) & (Tb.kind == "bin") & (Tb.st == stg) & (Tb.cohort == coh)]
                Y = d.pivot_table(index="session_id", columns="k", values=mt)
                Y = Y.loc[:, Y.notna().sum() >= m9.MIN_SESS]
                if Y.shape[1]:
                    mean_sem(ax, Y.columns.to_numpy() * TBIN / 60 + TBIN / 120, Y.to_numpy(), cl, coh)
            ax.set_title(f"{stg}: {'still events' if mt == 'rate_still' else 'events, all valid'} / s", fontsize=5)
            ax.set_xlabel("task time (min)", fontsize=4.6)
            ax.legend(frameon=False, fontsize=3.8)
            ax.tick_params(labelsize=4.2)
        th = Tb[(Tb.target == tg) & (Tb.kind == "third")].pivot_table(index=["session_id", "cohort", "st"], columns="k", values="rate_still").reset_index()
        if {0, 2} <= set(th.columns):
            th["late_minus_early"] = th[2] - th[0]
            strip(axs[i, 2], th[th.st == "day 0"].assign(bacc=1.0), "late_minus_early", meta=dict(analysis="E_thirds", target=tg, stage="day 0"))
        axs[i, 2].set_title("day 0: still rate, last - first third", fontsize=5)
        ax = axs[i, 3]
        q = W[W.target == tg][["session_id", "cohort", "st", PRIMARY[0]]].dropna()
        for k_, stg in enumerate(STAGES2):
            for c_i, (coh, cl) in enumerate(COH.items()):
                v = q[(q.st == stg) & (q.cohort == coh)][PRIMARY[0]]
                x = k_ + (c_i - 0.5) * 0.35
                ax.scatter(x + np.random.default_rng(c_i).uniform(-0.07, 0.07, len(v)), m9.symt(v), s=4, color=cl, lw=0)
                if len(v):
                    ax.plot([x - 0.12, x + 0.12], [m9.symt(v.median())] * 2, color="k", lw=0.8)
        if min(((q.st == s_) & (q.cohort == c_)).sum() for s_ in STAGES2 for c_ in COH) >= m9.MIN_SESS:
            q = q.assign(y=m9.symt(q[PRIMARY[0]]), rp=(q.cohort == "R+").astype(float), ex=(q.st == "expert").astype(float))
            f = smf.ols("y ~ rp * ex", data=q).fit()
            pint = float(f.pvalues["rp:ex"])
            STATS.append(dict(analysis="E_interaction", target=tg, metric=PRIMARY[0], p_interaction=pint, n=len(q)))
            ax.set_title(f"{PRIMARY[0]}: cohort x stage\nOLS (symlog) interaction p {pint:.2g}", fontsize=5)
        ax.set_xticks([0, 1], STAGES2, fontsize=4.2)
        tk = [-0.1, -0.01, 0, 0.01, 0.1]
        ax.set_yticks(m9.symt(tk), [f"{v:g}" for v in tk])
        ax.tick_params(labelsize=4.2)
        axs[i, 0].set_ylabel(f"{tg} decoder", fontsize=5)
    fig.suptitle(f"E. change across the session (10-min bins from the first kept task trial; bins with >= {m9.MIN_SESS} sessions) and across "
                 "stages, whole brain; mean +- s.e.m. over sessions; sessions of the same mouse at both stages are treated as independent",
                 x=0.01, ha="left", fontsize=5.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    m2.save(fig, fdir, "fig34_timecourse")
    plt.close(fig)

    # ---------------- F: performance
    Bh = pd.DataFrame([dict(session_id=s, **r["beh"]) for s, r in R.items()])
    Tr = cat_table(R, "trial", info_s)
    coefs = []
    import statsmodels.api as sm
    for (sid, tg), d in Tr[Tr.cls == "W"].groupby(["session_id", "target"]):
        d = d[d.secs_still >= 1]
        if len(d) < 20 or min(d.licked.sum(), (~d.licked).sum()) < 3:
            continue
        x = (d.n_still / d.secs_still).to_numpy(float)
        if np.std(x) == 0:
            continue
        X = sm.add_constant(np.c_[(x - x.mean()) / x.std(), (d.idx - d.idx.mean()) / d.idx.std()])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = sm.Logit(d.licked.to_numpy(float), X).fit(disp=0)
            coefs.append(dict(session_id=sid, target=tg, beta_iti_still=float(f.params[1]), beta_trial=float(f.params[2])))
        except Exception:
            pass
    Cf = pd.DataFrame(coefs)
    for stg in STAGES2:
        fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), squeeze=False)
        for i, tg in enumerate(m9.m6.TARGETS):
            q = W[(W.target == tg) & (W.st == stg)].merge(Bh, on="session_id", how="left")
            for j, (xc, xl) in enumerate([("hit_w", "whisker hit rate"), ("fa", "false-alarm rate"), ("dprime", "d' (whisker vs no stim)")]):
                corr_panel(axs[i, j], q, xc, PRIMARY[0], dict(analysis="F_perf", target=tg, stage=stg))
                axs[i, j].set_xlabel(xl, fontsize=4.6)
            axs[i, 0].set_ylabel(f"{tg}\n{PRIMARY[1]}", fontsize=4.6)
            if len(Cf):
                c = Cf[Cf.target == tg].merge(info_s, on="session_id")
                c = c[c.st == stg]
                strip(axs[i, 3], c.assign(bacc=1.0), "beta_iti_still", sym=False, meta=dict(analysis="F_trial", target=tg, stage=stg))
                from scipy.stats import wilcoxon
                for coh in COH:
                    v = c[c.cohort == coh].beta_iti_still.dropna()
                    if len(v) >= 5:
                        STATS.append(dict(analysis="F_trial_vs0", target=tg, stage=stg, cohort=coh, n=len(v), median=v.median(), p_wilcoxon=float(wilcoxon(v).pvalue)))
                axs[i, 3].axhline(0, color="0.6", lw=0.4, ls=":")
            axs[i, 3].set_title("trial level: lick on W trial ~ still-event\nrate in the preceding ITI (logit coef.)", fontsize=5)
        fig.suptitle(f"F. {stg}: event measures vs behaviour, whole brain; per session (kept active trials); OLS on the symlog scale + 95 % CI "
                     "(solid: p < 0.05); trial level: per-session logistic regression (z-scored predictors, + trial index)", x=0.01,
                     ha="left", fontsize=5.2, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        m2.save(fig, fdir, f"fig35_performance_{stg.replace(' ', '')}")
        plt.close(fig)

    # ---------------- G: HMM learning curves (ssl-learning-trial-identification 024, whisker day 0, sigma 1)
    from scipy.stats import spearmanr, wilcoxon
    with open(CURVES, "rb") as fh:
        cur = {d["session_id"]: d for d in pickle.load(fh)["sessions"]}
    Fe = pd.DataFrame([dict(session_id=s, wfa_mean=float(np.mean(d["curves"]["w-fa"])),
                            wfa_final=float(np.mean(np.asarray(d["curves"]["w-fa"])[-max(1, len(d["curves"]["w-fa"]) // 5):])),
                            n_whisker_curve=int(d["n_whisker"]), learning_category=d["learning_category"]) for s, d in cur.items()])
    tl = []
    for (sid, tg), d in Tr[Tr.cls == "W"].groupby(["session_id", "target"]):
        if sid not in cur:
            continue
        w = np.asarray(cur[sid]["curves"]["w-fa"])
        d = d.sort_values("idx").reset_index(drop=True)
        k = np.arange(len(d))                                           # k-th kept whisker trial = k-th whisker trial of the curve
        ok = (k < len(w)) & (d.secs_still >= 1).to_numpy()
        x = (d.n_still / d.secs_still.clip(lower=1e-9)).to_numpy()[ok]
        if ok.sum() < 20 or np.std(x) == 0:
            continue
        tl.append(dict(session_id=sid, target=tg, rho_iti_curve=float(spearmanr(x, w[k[ok]]).statistic), n_trials=int(ok.sum()),
                       n_kept_minus_curve=len(d) - len(w)))
    Tl = pd.DataFrame(tl)
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), squeeze=False)
    for i, tg in enumerate(m9.m6.TARGETS):
        q = W[(W.target == tg) & (W.st == "day 0")].merge(Fe, on="session_id", how="inner")
        for j, (xc, yc, xl) in enumerate([("wfa_mean", PRIMARY[0], "mean P(lick) whisker - FA (HMM)"),
                                          ("wfa_final", PRIMARY[0], "final P(lick) whisker - FA (last 20 %)"),
                                          ("wfa_final", "rate_still", "final P(lick) whisker - FA (last 20 %)")]):
            corr_panel(axs[i, j], q, xc, yc, dict(analysis="G_curve", target=tg, stage="day 0"))
            axs[i, j].set_xlabel(xl, fontsize=4.6)
            axs[i, j].set_title(yc, fontsize=5)
        if len(Tl):
            c = Tl[Tl.target == tg].merge(info_s, on="session_id")
            c = c[c.st == "day 0"]
            strip(axs[i, 3], c.assign(bacc=1.0), "rho_iti_curve", sym=False, meta=dict(analysis="G_trial", target=tg, stage="day 0"))
            for coh in COH:
                v = c[c.cohort == coh].rho_iti_curve.dropna()
                if len(v) >= 5:
                    STATS.append(dict(analysis="G_trial_vs0", target=tg, stage="day 0", cohort=coh, n=len(v), median=v.median(),
                                      p_wilcoxon=float(wilcoxon(v).pvalue)))
            axs[i, 3].axhline(0, color="0.6", lw=0.4, ls=":")
            STATS.append(dict(analysis="G_alignment", target=tg, n=len(c), median_kept_minus_curve=float(c.n_kept_minus_curve.median()),
                              max_abs_kept_minus_curve=float(c.n_kept_minus_curve.abs().max())))
        axs[i, 3].set_title("within session: Spearman of still-event\nrate before W trial vs curve (W - FA)", fontsize=5)
        axs[i, 0].set_ylabel(f"{tg} decoder", fontsize=5)
    fig.suptitle("G. day 0: events vs the HMM learning curves (ssl-learning-trial-identification 024: P(lick) per whisker trial, sigma 1, "
                 "whisker - false alarm); per session, OLS on the symlog scale + 95 % CI (solid: p < 0.05); trial level: k-th kept whisker "
                 "trial matched to the k-th curve trial", x=0.01, ha="left", fontsize=5.2, weight="bold", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    m2.save(fig, fdir, "fig38_learning_curves")
    plt.close(fig)

    # ---------------- H: areas
    import matplotlib.colors as mcolors
    hm = [PRIMARY[0], "rate_still", "rate"]
    for tg in m9.m6.TARGETS:
        G_ = [g for g in m8.GROUPS + sorted(set(Dm.group) - set(m8.GROUPS)) if g in set(Dm.group)]
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 0.22 * len(G_) + 1.2), squeeze=False)
        for k, stg in enumerate(STAGES2):
            M_ = np.full((len(G_), len(hm)), np.nan)
            T_ = np.full((len(G_), len(hm)), "", dtype=object)
            for gi, g in enumerate(G_):
                q = Dm[(Dm.group == g) & (Dm.target == tg) & (Dm.st == stg)]
                for mi, mt in enumerate(hm):
                    a_, b_ = q[q.cohort == "R+"][mt].dropna(), q[q.cohort == "R-"][mt].dropna()
                    if len(a_) >= m9.MIN_SESS and len(b_) >= m9.MIN_SESS:
                        p = test(a_, b_, analysis="H", target=tg, stage=stg, group=g, metric=mt)
                        M_[gi, mi] = m9.symt(a_.median()) - m9.symt(b_.median())
                        T_[gi, mi] = f"{p[0]:.2g}/{p[1]:.2g}{'*' if p[0] < 0.05 and p[1] < 0.05 else ''}\nn {len(a_)}/{len(b_)}"
            ax = axs[0, k]
            v = np.nanmax(np.abs(M_)) if np.isfinite(M_).any() else 1
            ax.imshow(M_, cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -v, v), aspect="auto")
            for gi in range(len(G_)):
                for mi in range(len(hm)):
                    ax.text(mi, gi, T_[gi, mi], ha="center", va="center", fontsize=3.2)
            ax.set_xticks(range(len(hm)), hm, fontsize=4.4)
            ax.set_yticks(range(len(G_)), ["whole brain" if g == GROUP else m2.short(g) for g in G_], fontsize=4.4)
            ax.set_title(f"{tg}, {stg}: median R+ - median R- (symlog units; red = R+ higher)", fontsize=5)
        fig.suptitle("H. area-group decoders: R+ vs R- per metric; text p Mann-Whitney / Welch (uncorrected), n R+ / R-; blank = < 3 sessions in a cohort",
                     x=0.01, ha="left", fontsize=5.2, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        m2.save(fig, fdir, f"fig36_areas_{tg}")
        plt.close(fig)

    # ---------------- I: robustness
    srcs = [f"{PFX}shuf{q:g}" for q in m9.NULL_Q]
    fig, axs = plt.subplots(4, 2 * len(srcs), figsize=(7.4, 5.6), squeeze=False)
    for i, (tg, stg) in enumerate([(t_, s_) for t_ in m9.m6.TARGETS for s_ in STAGES2]):
        for k, src in enumerate(srcs):
            for j, (sub, nm) in enumerate(((Dd, "all"), (Dd[Dd.bacc >= m9.GOOD_BACC], "good"))):
                q = sub[(sub.group == GROUP) & (sub.target == tg) & (sub.thr_source == src) & sub.stage.str.startswith(stg)]
                ax = axs[i, 2 * k + j]
                strip(ax, q, PRIMARY[0], meta=dict(analysis="I", target=tg, stage=stg, thr_source=src, decoders=nm))
                if i == 0:
                    ax.set_title(f"{src}, {nm} decoders", fontsize=4.8, pad=8)
        axs[i, 0].set_ylabel(f"{tg}, {stg}\n{PRIMARY[0]}", fontsize=4.6)
    fig.suptitle("I. robustness of the primary measure: shuffle-threshold percentile (99 / 99.5 / 99.9 / 99.99) x all vs good decoders "
                 f"(balanced accuracy >= {m9.GOOD_BACC}); whole brain", x=0.01, ha="left", fontsize=5.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    m2.save(fig, fdir, "fig37_robustness")
    plt.close(fig)

    St = pd.DataFrame(STATS)
    St["readout"] = m9.RNAME[SFX]
    St.to_csv(m1.HOME / f"analysis_015{tag}{SFX}_stats_{unit_set}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = St[St.analysis.isin(["A", "A_glm_adjusted", "A_glm_still_vs_shuffle", "E_interaction"]) | ((St.get("p_mannwhitney", 1) < 0.05) & (St.get("p_welch", 1) < 0.05))]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None, help="compute per-session tables ('all': every session with 009 outputs)")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--unit-set", default="stable")
    ap.add_argument("--decoders", default=None, help="decoders table (default: full_decoders_<unit_set>.csv)")
    ap.add_argument("--tag", default="", help="suffix of the output folder / stats file (tests)")
    ap.add_argument("--readout", choices=["decoder", "corr", "pcorr"], default="decoder",
                    help="v7: decoder readout, correlation readout or movement-partial correlation readout")
    a = ap.parse_args()
    if a.readout != "decoder":
        SFX = "_" + a.readout
        PFX = m9.PREFIX[SFX]
    if a.sessions is not None:
        ss = a.sessions
        if ss == ["all"]:
            ss = sorted(p.parent.name for p in (m9.ROOT / a.unit_set).glob(f"*/events_whisker_All{SFX}.parquet"))
        run_sessions(ss, a.unit_set)
    if a.figures:
        figures(a.unit_set, a.decoders, a.tag)
