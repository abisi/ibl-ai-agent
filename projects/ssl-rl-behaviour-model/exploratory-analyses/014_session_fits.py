"""014 -- Can single-session behaviour be fitted well trial by trial? (user 2026-10-18: "for now I just want to see if we can fit mouse
trial by trial single-session behaviour well"). Every session is fitted on its own (its starting values are free, so nothing is carried
over from earlier sessions). Trials: 011 sequences (active, perf != 6, no trimming).
Models (outcomes of a lick as model v3: auditory +1 water, whisker +1 water R+ / -1 R-, no stimulus -0.5):
  static  constant P(lick) per trial type (3 parameters) -- benchmark without any within-session dynamics
  L       learning: value per trial type, starting values q0_A, q0_W, q0_N free; after a lick delta = o - Q, Q += alpha+- * delta;
          P(lick) = (1 - lapse) * [1 - (1 - p_sp) * (1 - sigmoid(beta * Q))]  (8)
  M       motivation without learning: fixed values scaled by thirst (T = 1 at session start, - rho per reward), Pavlovian lick bias
          B(T) = s_B * softplus(20 (T - theta)) / 20; beta fixed to 1 (only beta x Q identifiable without learning)  (8)
  LM      learning + motivation  (11)
  LMP     + perseveration (choice kernel C <- C + phi (lick - C), + pi (2C - 1))  (13)
Evaluation: 5-fold blocked cross-validation within the session (5 contiguous blocks; parameters fitted on 4, the held-out block scored
with the model running through the whole session on the actual licks) -> held-out log-likelihood per trial and held-out per-trial
predictions (each trial predicted by the fold in which it is held out); also a full-session fit per model for the examples.
Output: fits/014_sessions.csv (per session x model), tables/014_trials.parquet (per trial: held-out and full-fit P(lick) per model),
figures/014_*.{png,pdf,svg}
Run (haas): python 014_session_fits.py [--n-jobs 90] [--plot-only]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

import common as C

TT = {"auditory": 0, "whisker": 1, "no_stim": 2}
P_ALL = ["qA", "qW", "qN", "alpha_pos", "alpha_neg", "beta", "lapse", "p_sp", "rho", "s_B", "theta", "pi", "phi"]
BOUNDS = dict(qA=(-3, 3), qW=(-3, 3), qN=(-3, 3), alpha_pos=(1e-3, 1), alpha_neg=(1e-3, 1), beta=(0.1, 20), lapse=(0, 0.5), p_sp=(0, 0.5),
              rho=(0, 0.1), s_B=(0, 10), theta=(0, 1), pi=(-3, 3), phi=(1e-3, 1))
FIXED = dict(qA=0.0, qW=0.0, qN=0.0, alpha_pos=0.0, alpha_neg=0.0, beta=1.0, lapse=0.0, p_sp=0.0, rho=0.0, s_B=0.0, theta=0.5, pi=0.0, phi=0.5)
DEFAULT = dict(qA=0.5, qW=0.0, qN=-0.5, alpha_pos=0.05, alpha_neg=0.05, beta=3.0, lapse=0.02, p_sp=0.05, rho=0.01, s_B=1.0, theta=0.5,
               pi=0.3, phi=0.3)
MODELS = {"static": ["qA", "qW", "qN"],
          "L": ["qA", "qW", "qN", "alpha_pos", "alpha_neg", "beta", "lapse", "p_sp"],
          "M": ["qA", "qW", "qN", "lapse", "p_sp", "rho", "s_B", "theta"],
          "LM": ["qA", "qW", "qN", "alpha_pos", "alpha_neg", "beta", "lapse", "p_sp", "rho", "s_B", "theta"],
          "LMP": ["qA", "qW", "qN", "alpha_pos", "alpha_neg", "beta", "lapse", "p_sp", "rho", "s_B", "theta", "pi", "phi"]}
NESTED = {"static": [], "L": ["static"], "M": ["static"], "LM": ["L", "M"], "LMP": ["LM"]}
N_FOLD = 5
import os
FLOOR = float(os.environ.get("RL_FLOOR", "0"))         # lower bound on lapse and p_sp (test of overconfident predictions); 0 = none
TAG = os.environ.get("RL_TAG", "")
if FLOOR > 0:
    BOUNDS["lapse"] = (FLOOR, 0.5); BOUNDS["p_sp"] = (FLOOR, 0.5); FIXED["lapse"] = FLOOR; FIXED["p_sp"] = FLOOR
    DEFAULT["lapse"] = max(DEFAULT["lapse"], FLOOR); DEFAULT["p_sp"] = max(DEFAULT["p_sp"], FLOOR)


@njit(cache=True)
def kernel(tt, lick, o, water, P, mask, pout):
    qA, qW, qN, ap, an, beta, lapse, psp, rho, sB, th, pi_, phi = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[12]
    q = np.array([qA, qW, qN]); T = 1.0; Ck = 0.5; ll = 0.0
    for t in range(tt.shape[0]):
        k = tt[t]
        x = 20.0 * (T - th)
        sp = x if x > 30.0 else np.log1p(np.exp(x))
        dv = beta * T * q[k] + sB * sp / 20.0 + pi_ * (2.0 * Ck - 1.0)
        p = (1.0 - lapse) * (1.0 - (1.0 - psp) * (1.0 - 1.0 / (1.0 + np.exp(-dv))))
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        pout[t] = p
        l = lick[t]
        if mask[t] == 1:
            ll += np.log(p) if l == 1 else np.log(1.0 - p)
        if l == 1:
            d = o[t] - q[k]
            q[k] += (ap if d > 0 else an) * d
            if water[t] > 0:
                T = max(0.0, T - rho)
        Ck += phi * (l - Ck)
    return ll


def vec(model, x):
    p = dict(FIXED); p.update(dict(zip(MODELS[model], x)))
    return np.array([p[k] for k in P_ALL], float), p


def fit(model, A, mask, starts):
    fp = MODELS[model]; pout = np.empty(len(A[0]))

    def nll(x):
        return -kernel(*A, vec(model, x)[0], mask, pout)
    best = None
    for x0 in starts:
        r = minimize(nll, np.asarray(x0, float), method="L-BFGS-B", bounds=[BOUNDS[k] for k in fp])
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return vec(model, best.x)[1], float(best.fun)


def fit_models(A, mask, rng, n_random=4):
    res = {}
    for m in MODELS:                                          # ordered: every model also starts from its sub-models' optima
        starts = [[DEFAULT[k] for k in MODELS[m]]] + [[rng.uniform(*BOUNDS[k]) for k in MODELS[m]] for _ in range(n_random)]
        starts += [[float(np.clip(res[s][0][k], *BOUNDS[k])) for k in MODELS[m]] for s in NESTED[m]]
        res[m] = fit(m, A, mask, starts)
    return res


def session(args):
    sid, g = args
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    tt = g.trial_type.map(TT).to_numpy(np.int64); lick = np.array(g.lick_flag, dtype=np.int64); n = len(tt)
    rplus = g.reward_group.iloc[0] == "R+"
    o = np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if rplus else -1.0, -0.5))
    water = np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if rplus else 0.0, 0.0))
    A = (tt, lick, o, water)
    fold = np.minimum((np.arange(n) * N_FOLD) // n, N_FOLD - 1)
    held = {m: np.full(n, np.nan) for m in MODELS}; ll_held = {m: 0.0 for m in MODELS}
    pout = np.empty(n)
    for f in range(N_FOLD):
        test = (fold == f).astype(np.int64)
        res = fit_models(A, 1 - test, rng)
        for m, (p, _) in res.items():
            ll_held[m] += kernel(*A, vec(m, [p[k] for k in MODELS[m]])[0], test, pout)
            held[m][test == 1] = pout[test == 1]
    full = fit_models(A, np.ones(n, np.int64), rng)
    fullp = {}
    for m, (p, nll) in full.items():
        kernel(*A, vec(m, [p[k] for k in MODELS[m]])[0], np.ones(n, np.int64), pout); fullp[m] = pout.copy()
    meta = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.reward_group.iloc[0], day=int(g.day.iloc[0]), n=n,
                n_A=int((tt == 0).sum()), n_W=int((tt == 1).sum()), n_N=int((tt == 2).sum()), lick_rate=lick.mean())
    rows = [dict(meta, model=m, k=len(MODELS[m]), ll_held=ll_held[m], ll_held_per_trial=ll_held[m] / n, nll_full=full[m][1],
                 **{f"p_{k}": full[m][0][k] for k in P_ALL}) for m in MODELS]
    T = pd.DataFrame(dict(session_id=sid, trial_in_session=np.arange(n), trial_type=g.trial_type.to_numpy(), lick=lick,
                          **{f"held_{m}": held[m] for m in MODELS}, **{f"full_{m}": fullp[m] for m in MODELS}))
    return rows, T


def run(a):
    d = pd.read_parquet(C.TAB / "011_sequences.parquet")
    groups = [(s, g) for s, g in d.groupby("session_id", sort=False) if len(g) >= 50]
    print(f"{len(groups)} sessions")
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        res = pool.map(session, groups, chunksize=2)
    S = pd.DataFrame([r for x in res for r in x[0]]); S.to_csv(C.FITS / f"014_sessions{TAG}.csv", index=False)
    pd.concat([x[1] for x in res]).to_parquet(C.TAB / f"014_trials{TAG}.parquet")
    print(S.groupby("model").ll_held_per_trial.mean().round(4))


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.titlesize": 6.5, "legend.fontsize": 5.3})
    COH = {"R+": "#00B400", "R-": "#C800C8"}
    TC = {"auditory": "#1f78b4", "whisker": "#d95f02", "no_stim": "0.35"}
    S = pd.read_csv(C.FITS / f"014_sessions{TAG}.csv"); T = pd.read_parquet(C.TAB / f"014_trials{TAG}.parquet")
    S["stage"] = np.where(S.day < 0, "pretraining", np.where(S.day == 0, "day 0", "whisker days ≥ 1"))
    W = S.pivot(index="session_id", columns="model", values="ll_held_per_trial")
    meta = S.drop_duplicates("session_id").set_index("session_id")[["mouse_id", "cohort", "day", "stage", "n", "lick_rate"]]
    gain = W.sub(W["static"], axis=0).join(meta)
    # held-out AUC per session and model
    au = []
    for sid, g in T.groupby("session_id"):
        if g.lick.nunique() < 2:
            continue
        au.append(dict(session_id=sid, **{m: roc_auc_score(g.lick, g[f"held_{m}"]) for m in MODELS}))
    AU = pd.DataFrame(au).set_index("session_id").join(meta)
    AU.to_csv(C.TAB / f"014_auc{TAG}.csv"); gain.to_csv(C.TAB / f"014_gain{TAG}.csv")
    # 1 performance
    stages = ["pretraining", "day 0", "whisker days ≥ 1"]; ms = ["L", "M", "LM", "LMP"]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=0.5))
    for j, st in enumerate(stages):
        ax = axs[j]
        for i, m in enumerate(ms):
            for k_, c in enumerate(("R+", "R-")):
                v = gain[(gain.stage == st) & (gain.cohort == c)][m]; x = i + (k_ - 0.5) * 0.36
                ax.scatter(np.full(len(v), x) + np.random.default_rng(i).uniform(-0.07, 0.07, len(v)), v, s=2, color=COH[c], alpha=0.4, lw=0)
                ax.plot([x - 0.13, x + 0.13], [v.median()] * 2, color="k", lw=0.9)
            v = gain[gain.stage == st][m]
            ax.text(i, 1.0, f"{(v > 0).mean():.0%}", transform=ax.get_xaxis_transform(), ha="center", fontsize=4.8)
        ax.axhline(0, color="0.5", lw=0.5); ax.set_xticks(range(4), ms); ax.set_ylim(-0.08, 0.3)
        ax.set_title(f"{st} (n = {(gain.stage == st).sum()} sessions)", pad=9)
        if j == 0:
            ax.set_ylabel("held-out LL per trial − static\n(top: % sessions > static)")
    ax = axs[3]
    for i, m in enumerate(["static"] + ms):
        v = AU[m].dropna()
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k"))
    ax.axhline(0.5, color="0.5", lw=0.5, ls="--"); ax.set_xticks(range(5), ["static"] + ms); ax.set_ylabel("held-out AUC per session")
    ax.set_title("trial-by-trial discrimination\n(lick vs no lick, all sessions)")
    fig.suptitle("Single-session fits: held-out (5-fold blocked cross-validation) performance per session, relative to constant lick rates "
                 "per trial type", fontsize=6.6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"014_1_performance{TAG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # 2 within-session curves: data vs held-out predictions, averaged over sessions, by stage and cohort
    Tm = T.merge(meta[["cohort", "stage"]], left_on="session_id", right_index=True)
    Tm["b"] = (Tm.trial_in_session // 20) * 20 + 10
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.0), sharey=True, gridspec_kw=dict(hspace=0.5))
    for i, c in enumerate(("R+", "R-")):
        for j, st in enumerate(stages):
            ax = axs[i, j]; e = Tm[(Tm.cohort == c) & (Tm.stage == st) & (Tm.b < 500)]
            for tk, col in TC.items():
                f = e[e.trial_type == tk].groupby(["session_id", "b"])[["lick", "held_LM", "held_static"]].mean().groupby("b").agg(["mean", "count"])
                f = f[f[("lick", "count")] >= 5]
                ax.plot(f.index, f[("lick", "mean")], color=col, lw=1.6, alpha=0.4)
                ax.plot(f.index, f[("held_LM", "mean")], color=col, lw=0.9)
                ax.plot(f.index, f[("held_static", "mean")], color=col, lw=0.6, ls=":")
            ax.set_title(f"{c.replace('-', '−')} — {st}"); ax.set_ylim(-0.02, 1.02)
            if i == 1:
                ax.set_xlabel("trial in session (bins of 20)")
        axs[i, 0].set_ylabel("lick rate")
    fig.suptitle("Within-session curves, averaged over sessions: thick = data, thin = LM held-out prediction, dotted = static "
                 "(blue auditory, orange whisker, grey no stimulus)", fontsize=6.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"014_2_within_session{TAG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # 3 examples: trial-by-trial
    ex = []
    for c in ("R+", "R-"):
        for st in stages:
            cand = gain[(gain.cohort == c) & (gain.stage == st)].sort_values("LM")
            if len(cand):
                ex.append(cand.index[len(cand) // 2])
    fig, axs = plt.subplots(len(ex), 1, figsize=(7.4, 1.25 * len(ex)), gridspec_kw=dict(hspace=0.9))
    for ax, sid in zip(np.atleast_1d(axs), ex):
        g = T[T.session_id == sid]; mrow = meta.loc[sid]
        for tk, col in TC.items():
            e = g[g.trial_type == tk]
            ax.scatter(e.trial_in_session, np.where(e.lick == 1, 1.04, -0.04), marker="|", s=10, color=col, lw=0.6)
            ax.plot(e.trial_in_session, e.held_LM, color=col, lw=0.9)
            ax.plot(e.trial_in_session, e.lick.rolling(9, center=True, min_periods=3).mean(), color=col, lw=1.5, alpha=0.3)
        r = gain.loc[sid]
        ax.set_title(f"{sid} ({mrow.cohort.replace('-', '−')}, day {mrow.day}, {mrow.stage}): held-out ΔLL/trial vs static: L {r.L:.3f}, "
                     f"M {r.M:.3f}, LM {r.LM:.3f}; held-out AUC LM {AU.LM.get(sid, np.nan):.2f}", fontsize=5.6, color=COH[mrow.cohort])
        ax.set_ylim(-0.1, 1.1); ax.set_ylabel("P(lick)")
    np.atleast_1d(axs)[-1].set_xlabel("trial in session")
    fig.suptitle("Example sessions (median held-out gain per cohort and stage): ticks = licks (top) / no licks (bottom), thin = LM held-out "
                 "P(lick), thick = 9-trial running lick rate (blue auditory, orange whisker, grey no stimulus)", fontsize=6.3)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"014_3_examples{TAG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # summary table
    rows = []
    for st in stages + ["all"]:
        g_ = gain if st == "all" else gain[gain.stage == st]; a_ = AU if st == "all" else AU[AU.stage == st]
        for m in ms:
            rows.append(dict(stage=st, model=m, n=len(g_), median_gain=g_[m].median(), frac_better=(g_[m] > 0).mean(),
                             p_wilcoxon=stats.wilcoxon(g_[m]).pvalue, median_auc=a_[m].median(), median_auc_static=a_["static"].median()))
    pd.DataFrame(rows).to_csv(C.TAB / f"014_summary{TAG}.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=90)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run(a)
    figures()
