"""176 -- Lagged cross-correlation of the single-session hit / miss decoder readout with the learning curve (user 2026-10-13: "my decoder
readout single-session on hit/miss, can you cross-correlate with learning curves?"). Extends 127 (zero lag only). Exploratory.
Readout: 123 v2 single-trial held-out signed margin of the hit / miss decoder (5-50 and 5-100 ms after whisker onset), class-
residualised (margin minus the mean of its class, so that the changing hit / miss composition cannot create a trend), as 127.
Behaviour (024 HMM curves, sigma = 1, read at each decoded trial's time as in 127):
  c   whisker - FA curve (learning-curve level) and the whisker P(lick) curve
  dc  first difference of c across decoded trials (learning rate), smoothed with a 5-trial boxcar
Cross-correlation per session: r(lag) = corr(z[i], c[i + lag]) over decoded trials, lag in -LAG..+LAG decoded trials (lag > 0: neural
readout leads behaviour). Null per session: N_SHIFT linear shifts (k = 10-50 % of the trials, both directions, non-wrapping; as 127)
of z against c; excess(lag) = r(lag) - mean(null). Group: mean excess(lag) +- s.e.m. per cohort; Wilcoxon vs 0 per lag with BH over
lags; peak lag per session (lag of max |excess|).
Learning-trial-aligned readout: z averaged in 10-trial bins around each session's behavioural learning trial (ssl-learning-trial-
identification 017 sigma1 lt_final; learners only), against random pseudo-learning trials (same session, 200 draws).
Populations: all day-0 sessions with decoder scores and curves; learners = learner / immediate_learner of 017 (immediate learners
have no LT and enter only the cross-correlations).
Outputs: tables/176_xcorr_per_session.parquet, 176_xcorr_stats.csv, 176_lt_aligned.csv; figures/176_margin_learning_xcorr.{png,pdf,svg}
Run (haas, repo root): python .../176_margin_learning_xcorr.py
"""
from __future__ import annotations

import pickle
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}
LAG, N_SHIFT, LO, HI, MIN_TR = 30, 200, 0.1, 0.5, 40
BIN, NB = 10, 4                                     # LT-aligned: 4 bins of 10 trials before and after


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def corr_lag(z, c, lag):
    a, b = (z[:len(z) - lag], c[lag:]) if lag >= 0 else (z[-lag:], c[:len(c) + lag])
    return np.corrcoef(a, b)[0, 1] if len(a) > 10 and a.std() > 0 and b.std() > 0 else np.nan


def null(z, c, rng):
    n = len(z); out = []
    for _ in range(N_SHIFT):
        k = int(rng.integers(max(1, int(LO * n)), max(2, int(HI * n)) + 1))
        a, b = (z[k:], c[:n - k]) if rng.random() < 0.5 else (z[:n - k], c[k:])
        if len(a) > 10 and a.std() > 0 and b.std() > 0:
            out.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(out)


def main():
    curves = {p["session_id"]: p["curves"] for p in pickle.load(open(LTP / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    LT = pd.read_csv(LTP / "017_learning_trials_final_sigma1.csv").set_index("session_id")
    D = pickle.load(open(OUT / "123_singletrial_scores_all_v2.pkl", "rb"))
    rows, ltrows = [], []
    lags = np.arange(-LAG, LAG + 1)
    for (sid, w), out in D["res"].items():
        m = D["meta"][sid]
        if sid not in curves:
            continue
        y = np.asarray(m["y"], bool); z = np.asarray(out["real"]["margin"], float).copy(); ok = np.isfinite(z)
        for cls in (True, False):
            s_ = ok & (y == cls)
            if s_.any():
                z[s_] -= z[s_].mean()
        t, cw_t = np.asarray(m["t"], float), np.asarray(m["cw_t"], float)
        cat = LT.category.get(sid, "unknown") if sid in LT.index else "unknown"
        for cname in ("w-fa", "whisker"):
            cv = np.asarray(curves[sid][cname], float)
            if len(cv) != len(cw_t):
                continue
            c = np.interp(t, cw_t, cv)[ok]; zz = z[ok]
            if len(zz) < MIN_TR or c.std() == 0:
                continue
            dc = np.convolve(np.r_[0, np.diff(c)], np.ones(5) / 5, "same")
            for sig, cc in (("level", c), ("rate", dc)):
                rng = np.random.default_rng(zlib.crc32(f"{sid}|{w}|{cname}|{sig}".encode()))
                nl = null(zz, cc, rng)
                if len(nl) < 20:
                    continue
                r = np.array([corr_lag(zz, cc, int(l)) for l in lags]); ex = r - nl.mean()
                rows.append(dict(session_id=sid, mouse_id=sid[:5], cohort=m["reward_group"], category=cat, window=w, curve=cname, signal=sig,
                                 n=len(zz), null_mean=nl.mean(), null_sd=nl.std(), **{f"r_{l}": v for l, v in zip(lags, r)},
                                 **{f"ex_{l}": v for l, v in zip(lags, ex)},
                                 peak_lag=int(lags[np.nanargmax(np.abs(ex))]) if np.isfinite(ex).any() else np.nan))
        # learning-trial-aligned readout (whisker-trial index of the LT -> decoded-trial index via the curve axis)
        if sid in LT.index and np.isfinite(LT.loc[sid, "lt_final"]):
            lt_i = int(LT.loc[sid, "lt_final"])
            if lt_i < len(cw_t):
                t_lt = cw_t[lt_i]; tt = t[ok]; zz = z[ok]
                i0 = int(np.searchsorted(tt, t_lt))
                def binned(i):
                    return [np.nanmean(zz[max(0, i + b * BIN):max(0, i + (b + 1) * BIN)]) if 0 <= i + b * BIN and i + (b + 1) * BIN <= len(zz)
                            else np.nan for b in range(-NB, NB)]
                real = binned(i0)
                rng = np.random.default_rng(zlib.crc32(f"{sid}|{w}|lt".encode()))
                nul = np.nanmean([binned(int(rng.integers(NB * BIN, max(NB * BIN + 1, len(zz) - NB * BIN)))) for _ in range(200)], axis=0)
                for b, (a_, n_) in enumerate(zip(real, nul)):
                    ltrows.append(dict(session_id=sid, cohort=m["reward_group"], window=w, bin=(b - NB) * BIN + BIN / 2, real=a_, null=n_))
    X = pd.DataFrame(rows); X.to_parquet(OUT / "176_xcorr_per_session.parquet", index=False)
    A = pd.DataFrame(ltrows); A.to_csv(OUT / "176_lt_aligned.csv", index=False)
    srows = []
    for (w, cname, sig), g in X.groupby(["window", "curve", "signal"]):
        for pop, gg in (("all", g), ("learners", g[g.category.isin(["learner", "immediate_learner"])])):
            for coh in ("R+", "R-"):
                q = gg[gg.cohort == coh]
                ps = [stats.wilcoxon(q[f"ex_{l}"].dropna()).pvalue if q[f"ex_{l}"].notna().sum() >= 6 else np.nan for l in lags]
                qs = bh(ps)
                for l, p_, q_ in zip(lags, ps, qs):
                    srows.append(dict(window=w, curve=cname, signal=sig, pop=pop, cohort=coh, lag=int(l), n=q[f"ex_{l}"].notna().sum(),
                                      mean_excess=q[f"ex_{l}"].mean(), sem_excess=q[f"ex_{l}"].sem(), p=p_, q=q_))
    S = pd.DataFrame(srows); S.to_csv(OUT / "176_xcorr_stats.csv", index=False)
    pd.set_option("display.width", 220)
    sel = S[(S["pop"] == "learners") & (S.lag.isin([-20, -10, -5, 0, 5, 10, 20]))]
    print(sel.round(3).to_string(index=False))
    print("min q per curve x signal x cohort (learners):")
    print(S[S["pop"] == "learners"].groupby(["window", "curve", "signal", "cohort"]).q.min().round(3).to_string())
    print("peak lag (learners, median):"); print(X[X.category.isin(["learner", "immediate_learner"])].groupby(["window", "curve", "signal", "cohort"]).peak_lag.median())
    figure(X, S, A, lags)


def figure(X, S, A, lags):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 5.6})
    w = "5-50ms"
    fig, axs = plt.subplots(1, 5, figsize=(7.4, 1.8), gridspec_kw=dict(wspace=0.55))
    for ax, (cname, sig, tt) in zip(axs[:4], (("w-fa", "level", "whisker − FA curve"), ("w-fa", "rate", "Δ whisker − FA (learning rate)"),
                                             ("whisker", "level", "whisker P(lick)"), ("whisker", "rate", "Δ whisker P(lick)"))):
        for coh in ("R+", "R-"):
            q = S[(S.window == w) & (S.curve == cname) & (S.signal == sig) & (S["pop"] == "learners") & (S.cohort == coh)].sort_values("lag")
            ax.fill_between(q.lag, q.mean_excess - q.sem_excess, q.mean_excess + q.sem_excess, color=COL[coh], alpha=0.2, lw=0)
            ax.plot(q.lag, q.mean_excess, color=COL[coh], lw=0.9, label=f"{coh.replace('-', '−')} (n = {int(q.n.max())})")
            sig_ = q[q.q < 0.05]
            ax.plot(sig_.lag, np.full(len(sig_), ax.get_ylim()[1] if False else q.mean_excess.max() + q.sem_excess.max()), "s", ms=1.5, color=COL[coh])
        ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel("lag (trials; > 0: readout leads)"); ax.set_title(tt, fontsize=5.6)
    axs[0].set_ylabel("r(readout, behaviour) − shift null"); axs[0].legend(frameon=False, fontsize=4.4)
    ax = axs[4]
    for coh in ("R+", "R-"):
        q = A[(A.window == w) & (A.cohort == coh)].groupby("bin")[["real", "null"]].agg(["mean", "sem"])
        ax.errorbar(q.index, q[("real", "mean")], q[("real", "sem")], color=COL[coh], marker="o", ms=2, lw=0.9, capsize=0)
        ax.plot(q.index, q[("null", "mean")], color=COL[coh], lw=0.6, ls=":")
    ax.axvline(0, color="0.6", lw=0.5, ls=":"); ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("trials from behavioural learning trial"); ax.set_ylabel("readout (class-residualised)")
    ax.set_title("Readout around the learning trial\n(dotted: random pseudo-LT)", fontsize=5.6)
    fig.suptitle(f"Hit / miss decoder readout ({w}) vs learning curve: lagged cross-correlation (learners; squares: BH q < 0.05)",
                 x=0.01, y=1.05, ha="left", fontsize=6.5, weight="bold")
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"176_margin_learning_xcorr.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
