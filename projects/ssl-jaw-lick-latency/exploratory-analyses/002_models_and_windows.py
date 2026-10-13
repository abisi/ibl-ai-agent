"""002 -- Jaw-onset / lick delay: detector sensitivity, prediction models, and lick-referenced pre-movement windows (user 2026-10-13:
"expand ... explain models, more figures, descriptions, caveats, and recommendations for my intended goal" = defining pre-movement
windows from lick times in sessions without video). Population: SSL_JAW_POP (default all sessions with jaw video; see 001).
A  Detector sensitivity (WH / AH / SL events of 118): onset re-detected on the same jaw traces with threshold K x baseline SD,
   K in {4, 7, 10}, and with a velocity criterion (first frame where d(jaw)/dt > K_V x baseline SD of the velocity, K_V = 5);
   lead distributions and per-event agreement with K = 7 (the rule used everywhere).
B  Prediction models of the lead (lick - jaw onset), leave-one-session-out (LOSO; the held-out session contributes nothing):
   M0 constant      global median lead (one number)
   M1 class         median lead per event class (WH / AH / SL)
   M2 mouse         median lead of the mouse's other sessions per class (M1 if the mouse has no other session)
   M3 RT linear     lead = a + b RT, per class (WH, AH), OLS on all other sessions
   M4 RT mixed      lead = a + b RT + mouse random intercept (statsmodels MixedLM, per class); the intercept of the held-out
                    session's mouse comes from its other sessions (0 if none)
   M5 stim-locked   jaw onset = stimulus + median stimulus-to-onset latency of other sessions (per class), i.e. lead = RT - latency
   M6 RT isotonic   lead = monotone non-decreasing function of RT (isotonic regression on other sessions)
   For each model: per-event error (predicted - actual onset, ms), and a split-conformal 90 % prediction interval (half-width =
   90th percentile of |error| on the training sessions) and its empirical coverage on the held-out session.
C  Pre-movement windows defined from the lick only: W(X) = [lick - X - 100, lick - X) ms. Contamination of an event = fraction of the
   window after the true jaw onset (0 = fully pre-movement). Curves of mean contamination and fraction of clean events (contamination
   = 0) vs X, per class; X needed for 80 / 90 / 95 % clean events, pooled and per session (session-to-session spread). Model-based
   windows: [predicted onset - 100, predicted onset) for M1, M3, M5, with their contamination.
Outputs: combined_results_ks4/ssl-jaw-lick-latency/[all_sessions/]{tables/*.csv, figures/jaw_lick_models.{png,pdf,svg},
figures/jaw_lick_windows.{png,pdf,svg}}
Run (haas): [SSL_JAW_POP=all] PRELICK_REF=sl PRELICK_UNITS=stable python .../002_models_and_windows.py [--n-proc 20]
"""
import argparse
import importlib
import multiprocessing as mp
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SSL_JAW_POP", "all")
m001 = importlib.import_module("001_jaw_lick_latency")
m113, m51 = m001.m113, m001.m51
OUT, CLS, COLC = m001.OUT, m001.CLS, m001.COLC
KS, K_V, SL_BASE, SL_SEARCH, MIN_AFTER = (4.0, 7.0, 10.0), 5.0, (-1.0, -0.5), 0.5, 0.05
WIN = 100.0
MODELS = ["M0 constant", "M1 class", "M2 mouse", "M3 RT linear", "M4 RT mixed", "M5 stim-locked", "M6 RT isotonic"]
COLM = {"M0 constant": "#bbbbbb", "M1 class": "#888888", "M2 mouse": "#4C72B0", "M3 RT linear": "#d94801", "M4 RT mixed": "#8c2d04",
        "M5 stim-locked": "#2CA02C", "M6 RT isotonic": "#9467BD"}


# ------------------------------------------------------------------ A detector sensitivity
def onset_k(t, v, b0, b1, s0, s1, k, vel=False):
    bi = (t >= b0) & (t < b1)
    x = np.gradient(v, t) if vel else v
    bv = x[bi]; bv = bv[np.isfinite(bv)]
    if len(bv) < 20:
        return np.nan
    si = np.where((t > s0) & (t <= s1))[0]
    if not len(si):
        return np.nan
    c = np.where(x[si] - bv.mean() > k * bv.std())[0]
    return float(t[si[c[0]]]) if len(c) else np.nan


def sens_session(args):
    sid, ev = args
    _, _, series = m113.load_series(sid)
    if "jaw distance raw" not in series:
        return []
    t, v = series["jaw distance raw"]
    out = []
    for r in ev.itertuples():
        if r.cls == "SL":
            win = (r.lick_time_used + SL_BASE[0], r.lick_time_used + SL_BASE[1], r.lick_time_used - SL_SEARCH, r.lick_time_used)
        else:
            win = (r.start_time - 1.0, r.start_time, r.start_time + MIN_AFTER, r.lick_time_used)
        d = dict(session_id=sid, cls=r.cls, lick=r.lick_time_used)
        for k in KS:
            d[f"on_K{k:g}"] = onset_k(t, v, *win, k)
        d["on_vel"] = onset_k(t, v, *win, K_V, vel=True)
        out.append(d)
    return out


def sensitivity(E, n_proc):
    ev = E[E.cls != "SL"]
    ev = pd.concat([ev, E[E.cls == "SL"].groupby("session_id", group_keys=False).apply(lambda g: g.sample(min(len(g), 60), random_state=0))])
    jobs = [(sid, g) for sid, g in ev.groupby("session_id")]
    with mp.get_context("fork").Pool(n_proc) as pool:
        R = pd.DataFrame([x for r in pool.map(sens_session, jobs) for x in r])
    for c in [k for k in R.columns if k.startswith("on_")]:
        R[c.replace("on_", "lead_")] = (R.lick - R[c]) * 1e3
    return R


# ------------------------------------------------------------------ B models
def fit_mixed(o):
    import statsmodels.formula.api as smf
    try:
        md = smf.mixedlm("lead ~ rt", o, groups=o["mouse_id"]).fit(reml=True)
        return md.params["Intercept"], md.params["rt"], {k: float(v.iloc[0]) for k, v in md.random_effects.items()}
    except Exception:  # noqa: BLE001
        b, a = np.polyfit(o.rt, o.lead, 1); return a, b, {}


def loso(E):
    from sklearn.isotonic import IsotonicRegression
    rows = []
    sids = E.session_id.unique()
    glob = {}
    for sid in sids:
        o_all = E[E.session_id != sid]
        g_all = E[E.session_id == sid]
        for c in CLS:
            g = g_all[g_all.cls == c]
            if not len(g):
                continue
            o = o_all[o_all.cls == c]; om = o[o.mouse_id == g.mouse_id.iloc[0]]
            P = {"M0 constant": np.full(len(g), o_all.lead.median()), "M1 class": np.full(len(g), o.lead.median()),
                 "M2 mouse": np.full(len(g), om.lead.median() if len(om) >= 10 else o.lead.median())}
            if c != "SL":
                oo = o[~o.censored & o.rt.notna()]
                b, a = np.polyfit(oo.rt, oo.lead, 1); P["M3 RT linear"] = a + b * g.rt.to_numpy()
                key = (sid, c)
                a4, b4, re = fit_mixed(oo.sample(min(len(oo), 2500), random_state=0))
                P["M4 RT mixed"] = a4 + b4 * g.rt.to_numpy() + re.get(g.mouse_id.iloc[0], 0.0)
                lat = ((o.jaw_onset - o.start_time) * 1e3).median(); P["M5 stim-locked"] = g.rt.to_numpy() - lat
                iso = IsotonicRegression(out_of_bounds="clip").fit(oo.rt, oo.lead); P["M6 RT isotonic"] = iso.predict(g.rt.to_numpy())
                # training residuals for conformal half-widths (per model, from the other sessions, in-sample)
                tr = {"M3 RT linear": oo.lead - (a + b * oo.rt), "M5 stim-locked": oo.lead - (oo.rt - lat),
                      "M6 RT isotonic": oo.lead - iso.predict(oo.rt), "M4 RT mixed": oo.lead - (a4 + b4 * oo.rt + oo.mouse_id.map(re).fillna(0))}
            else:
                tr = {}
            tr.update({"M0 constant": o.lead - o_all.lead.median(), "M1 class": o.lead - o.lead.median(), "M2 mouse": o.lead - o.lead.median()})
            for k, p in P.items():
                hw = np.nanpercentile(np.abs(tr[k]), 90)
                err = g.lead.to_numpy() - p
                rows.append(pd.DataFrame(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cls=c, model=k, err=err, pred_lead=p,
                                              lead=g.lead.to_numpy(), halfwidth90=hw, covered=np.abs(err) <= hw)))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ C windows
def contamination(lead, end_before_lick):
    """window [lick - X - WIN, lick - X); jaw onset at lick - lead; fraction of the window after the onset"""
    w0, w1 = -end_before_lick - WIN, -end_before_lick; on = -lead
    return np.clip((w1 - np.maximum(on, w0)) / WIN, 0, 1)


def windows(E, P):
    X = np.arange(0, 505, 10)
    curves = []
    for c in CLS:
        lead = E[E.cls == c].lead.to_numpy()
        for x in X:
            cc = contamination(lead, x)
            curves.append(dict(cls=c, X=x, mean_contam=cc.mean(), clean=np.mean(cc == 0)))
    C = pd.DataFrame(curves)
    need = []
    for c in CLS:
        d = E[E.cls == c]
        for q in (0.8, 0.9, 0.95):
            need.append(dict(cls=c, level=q, X_pooled=np.quantile(d.lead, q),   # a window ending X before the lick is clean iff lead <= X
                             X_session_median=d.groupby("session_id").lead.quantile(q).median(),
                             X_session_q10=d.groupby("session_id").lead.quantile(q).quantile(0.1),
                             X_session_q90=d.groupby("session_id").lead.quantile(q).quantile(0.9)))
    # model-based windows: [predicted onset - WIN, predicted onset); contamination = fraction after the true onset
    mw = []
    for k in ("M1 class", "M3 RT linear", "M5 stim-locked"):
        q = P[P.model == k]
        cc = np.clip((-q.pred_lead.to_numpy() - np.maximum(-q.lead.to_numpy(), -q.pred_lead.to_numpy() - WIN)) / WIN, 0, 1)
        # window = [-pred - WIN, -pred); onset at -lead; contaminated part = (-pred) - max(-lead, -pred - WIN) if onset < window end
        mw.append(pd.DataFrame(dict(model=k, cls=q.cls.to_numpy(), contam=cc, early=np.clip((q.lead - q.pred_lead).to_numpy(), None, 0) * -1)))
    # windows guaranteed (approximately) 90 % clean: fixed offset X90 vs model prediction + margin m90 (90th percentile of the model's
    # held-out errors lead - predicted lead, pooled over sessions); reported: distance of the window end from the lick (median over events)
    g90 = []
    for c in CLS:
        d = E[E.cls == c]
        g90.append(dict(cls=c, approach="fixed offset", dist_median=np.quantile(d.lead, 0.9), clean=np.mean(d.lead <= np.quantile(d.lead, 0.9))))
        for k in ("M1 class", "M2 mouse", "M3 RT linear", "M5 stim-locked", "M6 RT isotonic"):
            q = P[(P.model == k) & (P.cls == c)]
            if not len(q):
                continue
            m90 = np.quantile(q.err, 0.9); dist = q.pred_lead + m90
            g90.append(dict(cls=c, approach=f"{k} + margin", dist_median=np.median(dist), dist_q10=np.quantile(dist, 0.1), dist_q90=np.quantile(dist, 0.9),
                            clean=np.mean(q.err <= m90)))
    return C, pd.DataFrame(need), pd.concat(mw), pd.DataFrame(g90)


# ------------------------------------------------------------------ figures
def fig_models(R, P, PS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.4, "axes.labelsize": 5.4, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 4, wspace=0.6, hspace=0.85, left=0.07, right=0.985, top=0.9, bottom=0.1)
    L = []
    # a detector threshold: lead distributions
    ax = fig.add_subplot(gs[0, 0]); L.append(ax)
    for k, col in zip(["lead_K4", "lead_K7", "lead_K10", "lead_vel"], ["#9ecae1", "#3182bd", "#08519c", "#d94801"]):
        x = R[k].dropna(); x = x[x.between(0, 600)]
        ax.hist(x, bins=np.arange(0, 605, 15), histtype="step", density=True, color=col, lw=0.9,
                label=f"{k.replace('lead_', '').replace('K', 'K = ').replace('vel', 'velocity')} ({x.median():.0f} ms)")
    ax.set_xlabel("lead (ms)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=3.8)
    ax.set_title("Detector: threshold / criterion", fontsize=5.4)
    # b agreement with K = 7
    ax = fig.add_subplot(gs[0, 1]); L.append(ax)
    for k, col in zip(["lead_K4", "lead_K10", "lead_vel"], ["#9ecae1", "#08519c", "#d94801"]):
        d = (R[k] - R.lead_K7).dropna()
        e = np.sort(np.abs(d)); ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, lw=0.9, label=f"{k.replace('lead_', '')} (median {np.median(e):.0f} ms)")
    ax.set_xlim(0, 150); ax.set_xlabel("|lead − lead(K = 7)| (ms)"); ax.set_ylabel("cumulative fraction"); ax.legend(frameon=False, fontsize=3.8, loc="lower right")
    ax.set_title("Detector agreement with K = 7", fontsize=5.4)
    # c model error per class (median |error|)
    ax = fig.add_subplot(gs[0, 2:4]); L.append(ax)
    w = 0.11
    for i, k in enumerate(MODELS):
        for j, c in enumerate(CLS):
            q = PS[(PS.model == k) & (PS.cls == c)]
            if len(q):
                ax.bar(j + (i - 3) * w, q.median_abs.iloc[0], w * 0.95, color=COLM[k], label=k if j == 0 else None)
    ax.set_xticks(range(len(CLS)), [{"WH": "whisker hits", "AH": "auditory hits", "SL": "spontaneous licks"}[c] for c in CLS])
    ax.set_ylabel("median |predicted − actual onset| (ms)"); ax.legend(frameon=False, fontsize=3.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Prediction models (leave one session out)", fontsize=5.4); ax.set_ylim(0, 75); ax.tick_params(axis="x", labelsize=4.6)
    # d error vs RT for M1 and M3 (AH and WH pooled)
    ax = fig.add_subplot(gs[1, 0]); L.append(ax)
    E_ = P[(P.cls != "SL")].copy()
    rt = E_.groupby(["model"]).apply(lambda q: q)  # noqa: F841
    for k in ("M1 class", "M3 RT linear", "M5 stim-locked"):
        q = P[(P.model == k) & (P.cls != "SL")]
        bins = np.quantile(q.lead, np.linspace(0, 0.9, 10))
        b = pd.cut(q.lead, bins, include_lowest=True); s = q.groupby(b).err.median()
        ax.plot([iv.mid for iv in s.index], s.values, "o-", ms=2, lw=0.8, color=COLM[k], label=k)
    ax.axhline(0, color="0.6", lw=0.5); ax.set_xlabel("actual lead (ms; deciles, lowest 90 %)"); ax.set_ylabel("median error (ms)")
    ax.legend(frameon=False, fontsize=3.8); ax.set_title("Errors are largest for long leads", fontsize=5.4)
    # e conformal 90 % intervals: width and coverage
    ax = fig.add_subplot(gs[1, 1]); L.append(ax)
    for k in MODELS:
        q = P[P.model == k]
        cov = q.groupby(["cls"]).covered.mean(); hw = q.groupby("cls").halfwidth90.median()
        for c in CLS:
            if c in cov.index:
                ax.plot(2 * hw[c], cov[c], "o", ms=3, color=COLM[k], mec=COLC[c], mew=0.8)
    ax.axhline(0.9, color="0.6", lw=0.5, ls=":"); ax.set_xlabel("90 % interval width (ms)"); ax.set_ylabel("held-out coverage")
    ax.set_title("Prediction intervals (90 %)\nfill: model, edge: event class", fontsize=5.4)
    # f per-session bias of M3 (systematic offset of a session)
    ax = fig.add_subplot(gs[1, 2]); L.append(ax)
    for k in ("M1 class", "M2 mouse", "M3 RT linear"):
        b = P[(P.model == k) & (P.cls != "SL")].groupby("session_id").err.median()
        ax.hist(b, bins=np.arange(-200, 205, 10), histtype="step", color=COLM[k], lw=0.9, label=f"{k} (SD {b.std():.0f} ms)")
    ax.set_xlabel("session median error (ms)"); ax.set_ylabel("sessions"); ax.legend(frameon=False, fontsize=3.8)
    ax.set_title("Per-session bias (WH + AH)", fontsize=5.4)
    # g fraction within ±50 ms per model
    ax = fig.add_subplot(gs[1, 3]); L.append(ax)
    for i, k in enumerate(MODELS):
        for j, c in enumerate(CLS):
            q = PS[(PS.model == k) & (PS.cls == c)]
            if len(q):
                ax.plot(j + (i - 3) * 0.1, q.within50.iloc[0], "o", ms=3, color=COLM[k])
    ax.set_xticks(range(len(CLS)), CLS); ax.set_ylabel("fraction within ± 50 ms"); ax.set_ylim(0, 1)
    ax.set_title("Accuracy at ± 50 ms", fontsize=5.4)
    for ax, l in zip(L, "abcdefg"):
        ax.text(-0.3 if l != "c" else -0.1, 1.12, l, transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Jaw onset from the lick: detector sensitivity and prediction models", x=0.01, y=0.995, ha="left", fontsize=7, weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"jaw_lick_models.{ext}", dpi=300)


def fig_windows(E, C, N, MW, G90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.4, "axes.labelsize": 5.2, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 5, wspace=0.75, left=0.06, right=0.985, top=0.74, bottom=0.2)
    L = []
    ax = fig.add_subplot(gs[0, 0]); L.append(ax)
    for c in CLS:
        q = C[C.cls == c]; ax.plot(q.X, q.clean, color=COLC[c], lw=1.0, label=c)
    for v in (0.9, 0.95):
        ax.axhline(v, color="0.75", lw=0.5, ls=":")
    ax.set_xlabel("window end before the lick X (ms)"); ax.set_ylabel("fraction of events fully pre-movement"); ax.legend(frameon=False, fontsize=4.2)
    ax.set_title("Window [lick − X − 100, lick − X)", fontsize=5.4)
    ax = fig.add_subplot(gs[0, 1]); L.append(ax)
    for c in CLS:
        q = C[C.cls == c]; ax.plot(q.X, q.mean_contam, color=COLC[c], lw=1.0)
    ax.set_xlabel("X (ms)"); ax.set_ylabel("mean fraction of window after jaw onset"); ax.set_title("Movement contamination", fontsize=5.4)
    ax = fig.add_subplot(gs[0, 2]); L.append(ax)
    S = E.groupby(["session_id", "cls"]).lead.quantile(0.9).reset_index()
    for j, c in enumerate(CLS):
        v = S[S.cls == c].lead
        ax.plot(np.full(len(v), j) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, "o", ms=1.6, color=COLC[c], alpha=0.6)
        ax.plot([j - 0.25, j + 0.25], [np.quantile(E[E.cls == c].lead, 0.9)] * 2, color="k", lw=1)
    ax.set_xticks(range(len(CLS)), CLS); ax.set_ylabel("X for 90 % clean events (ms)")
    ax.set_title("Needed offset varies by session\n(dots: sessions; bar: pooled)", fontsize=5.4)
    ax = fig.add_subplot(gs[0, 3]); L.append(ax)
    for i, k in enumerate(("M1 class", "M3 RT linear", "M5 stim-locked")):
        for j, c in enumerate(("WH", "AH")):
            q = MW[(MW.model == k) & (MW.cls == c)]
            if len(q):
                ax.bar(j + (i - 1) * 0.25, (q.contam > 0).mean(), 0.23, color=COLM[k], label=k if j == 0 else None)
    ax.set_xticks([0, 1], ["WH", "AH"]); ax.set_ylabel("fraction of windows with movement")
    ax.legend(frameon=False, fontsize=3.8); ax.set_title("Window ending at the\npredicted onset", fontsize=5.2)
    ax = fig.add_subplot(gs[0, 4]); L.append(ax)
    ap_ = ["fixed offset", "M1 class + margin", "M2 mouse + margin", "M3 RT linear + margin", "M5 stim-locked + margin", "M6 RT isotonic + margin"]
    cols = ["k", COLM["M1 class"], COLM["M2 mouse"], COLM["M3 RT linear"], COLM["M5 stim-locked"], COLM["M6 RT isotonic"]]
    for i, (a_, col) in enumerate(zip(ap_, cols)):
        for j, c in enumerate(CLS):
            q = G90[(G90.approach == a_) & (G90.cls == c)]
            if len(q):
                ax.plot(j + (i - 2.5) * 0.12, q.dist_median.iloc[0], "o", ms=3, color=col, label=a_.replace(" + margin", "") if j == 0 else None)
    ax.set_xticks(range(len(CLS)), CLS); ax.set_ylabel("window end before the lick (ms; median)")
    ax.legend(frameon=False, fontsize=3.6, loc="lower left"); ax.set_title("Distance needed for\n90 % clean events", fontsize=5.2)
    for ax, l in zip(L, "abcde"):
        ax.text(-0.3, 1.15, l, transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Defining pre-movement windows without video", x=0.01, y=0.99, ha="left", fontsize=7, weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"jaw_lick_windows.{ext}", dpi=300)


def main(a):
    (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(parents=True, exist_ok=True)
    E = m001.load()
    pd.set_option("display.width", 220)
    R = sensitivity(E, a.n_proc); R.to_csv(OUT / "tables" / "detector_sensitivity.csv", index=False)
    print("detector: median lead and agreement with K = 7")
    for k in ("lead_K4", "lead_K7", "lead_K10", "lead_vel"):
        d = (R[k] - R.lead_K7).dropna()
        print(f"  {k:9s} median {R[k].median():6.1f} ms | detected {R[k].notna().mean():.2f} | median |diff| {np.median(np.abs(d)):5.1f} | within 10 ms {np.mean(np.abs(d) <= 10):.2f}")
    P = loso(E); P.to_csv(OUT / "tables" / "models_loso.csv", index=False)
    PS = P.groupby(["cls", "model"]).agg(median_abs=("err", lambda x: np.nanmedian(np.abs(x))), within25=("err", lambda x: np.nanmean(np.abs(x) <= 25)),
                                         within50=("err", lambda x: np.nanmean(np.abs(x) <= 50)), bias=("err", "median"),
                                         halfwidth90=("halfwidth90", "median"), coverage90=("covered", "mean")).reset_index()
    print(PS.round(3).to_string(index=False)); PS.to_csv(OUT / "tables" / "models_summary.csv", index=False)
    C, N, MW, G90 = windows(E, P)
    C.to_csv(OUT / "tables" / "window_curves.csv", index=False); N.to_csv(OUT / "tables" / "window_offsets.csv", index=False)
    G90.to_csv(OUT / "tables" / "window_90clean.csv", index=False); print(G90.round(1).to_string(index=False))
    print(N.round(1).to_string(index=False))
    print(MW.groupby(["model", "cls"]).agg(frac_contaminated=("contam", lambda x: np.mean(x > 0)), mean_contam=("contam", "mean")).round(3))
    fig_models(R, P, PS); fig_windows(E, C, N, MW, G90)
    print("ALL DONE", OUT / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=20)
    main(ap.parse_args())
