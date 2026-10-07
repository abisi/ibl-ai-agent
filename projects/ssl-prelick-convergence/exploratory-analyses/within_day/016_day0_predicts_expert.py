"""016 -- Do a mouse's learning-day (day 0) within-session effects predict its expert-stage performance? (user 2026-10-07)

Unit = mouse: mice with a day-0 session and >= 1 expert session (day >= 1); expert values averaged over the mouse's expert
sessions. Cohorts analysed separately (the sign of good performance differs: R+ should lick to whisker, R- withhold).
Day-0 predictors (x):
  neural (011 cd_trajectories sessions, 002 slopes sessions): drift of WH relative to SL (beta_WH - beta_SL) and to AH
    (beta_WH - beta_AH) on the reward-lick CD, WH score at the end of day 0 (OLS fit at tau = 1, md axis)
  behaviour (077 performance_sessions, 001 halves): day-0 whisker-hit rate, day-0 d' (WH vs false alarm on no-stim
    trials), day-0 change of the whisker-hit rate (late - early half, events / min)
Expert outcomes (y): whisker hit rate, d' (WH vs false alarm), expert mean WH CD score (011).
Statistics: Spearman rho and Pearson r per cohort; partial Spearman of each neural predictor given the day-0 whisker hit
rate (rank residuals) -- does the neural effect predict beyond day-0 behaviour?
Figure: scatter + OLS + 95% CI band per cohort (solid line only if p < 0.05), open = R+ / R- colours.
Output: within_day/<ref>/day0_predicts_expert/<pop>/day0_predicts_expert.{png,pdf,svg}, mice.csv, correlations.csv.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as tdist

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
BASE = m51.WITHIN
OUT = BASE / "day0_predicts_expert"
COH = m62.COH
XN = [("d0_drift_WH_SL", "Day-0 drift WH vs SL\n(β_WH − β_SL)"), ("d0_drift_WH_AH", "Day-0 drift WH vs AH\n(β_WH − β_AH)"),
      ("d0_WH_end", "Day-0 WH score at session end\n(CD, SL = 0, AH = 1)")]
XB = [("d0_rate_wh", "Day-0 whisker hit rate"), ("d0_dprime", "Day-0 d′ (WH vs FA)"),
      ("d0_dWHrate", "Day-0 change of WH rate\n(late − early, /min)")]
YN = [("ex_rate_wh", "Expert whisker hit rate"), ("ex_dprime", "Expert d′ (WH vs FA)"), ("ex_mean_WH", "Expert WH CD score")]


def mouse_table(pop):
    C = pd.read_csv(BASE / "cd_trajectories" / pop / "sessions.csv")
    T = pd.read_csv(BASE / "slopes" / "trial_slopes_sessions.csv")[["session_id", "md_WH_end"]]
    H = pd.read_csv(BASE / "halves" / "within_session_sessions.csv")
    H = H[H.split == "time"].assign(dWHrate=lambda d: d.rate_WH_late - d.rate_WH_early)[["session_id", "dWHrate"]]
    P = pd.read_csv(m51.OUTROOT / "extras" / "performance_sessions.csv")
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    P = flt(P[P.cohort.isin(["R+", "R-"])])
    S = P.merge(C[["session_id", "drift_WH_SL", "drift_WH_AH", "mean_WH"]], on="session_id", how="left") \
         .merge(T, on="session_id", how="left").merge(H, on="session_id", how="left")
    agg = dict(rate_wh="mean", dprime_wh_fa="mean", drift_WH_SL="mean", drift_WH_AH="mean", md_WH_end="mean",
               mean_WH="mean", dWHrate="mean", session_id="nunique")
    L = S[S.stage == "learning"].groupby(["mouse_id", "cohort"]).agg(agg).reset_index()
    E = S[S.stage == "expert"].groupby(["mouse_id", "cohort"]).agg(agg).reset_index()
    L = L.rename(columns=dict(rate_wh="d0_rate_wh", dprime_wh_fa="d0_dprime", drift_WH_SL="d0_drift_WH_SL",
                              drift_WH_AH="d0_drift_WH_AH", md_WH_end="d0_WH_end", mean_WH="d0_mean_WH", dWHrate="d0_dWHrate",
                              session_id="n_day0"))
    E = E.rename(columns=dict(rate_wh="ex_rate_wh", dprime_wh_fa="ex_dprime", mean_WH="ex_mean_WH", session_id="n_expert"))
    E = E[["mouse_id", "cohort", "ex_rate_wh", "ex_dprime", "ex_mean_WH", "n_expert"]]
    return L.merge(E, on=["mouse_id", "cohort"], how="inner")


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, _ = stats.pearsonr(ex, ey)
    n = len(x); tt = r * np.sqrt((n - 3) / max(1 - r ** 2, 1e-12))
    return r, 2 * tdist.sf(abs(tt), n - 3)


def panel(ax, M, x, y, rows):
    txt = []
    for coh in ("R+", "R-"):
        q = M[M.cohort == coh][list(dict.fromkeys([x, y, "d0_rate_wh"]))].dropna()
        ax.scatter(q[x], q[y], s=9, facecolor="white", edgecolor=COH[coh], lw=0.7, zorder=3)
        if len(q) < 5:
            txt.append(f"{coh.replace('-', '−')}: n = {len(q)}")
            continue
        rho, p = stats.spearmanr(q[x], q[y]); r, pr = stats.pearsonr(q[x], q[y])
        pr_, pp_ = partial_spearman(q[x].to_numpy(), q[y].to_numpy(), q.d0_rate_wh.to_numpy()) if x != "d0_rate_wh" else (np.nan, np.nan)
        b = np.polyfit(q[x], q[y], 1); xs = np.linspace(q[x].min(), q[x].max(), 50)
        res = q[y] - np.polyval(b, q[x]); s2 = (res ** 2).sum() / (len(q) - 2)
        se = np.sqrt(s2 * (1 / len(q) + (xs - q[x].mean()) ** 2 / ((q[x] - q[x].mean()) ** 2).sum()))
        tq = tdist.ppf(0.975, len(q) - 2)
        ax.fill_between(xs, np.polyval(b, xs) - tq * se, np.polyval(b, xs) + tq * se, color=COH[coh], alpha=0.12, lw=0)
        ax.plot(xs, np.polyval(b, xs), color=COH[coh], lw=0.9, ls="-" if p < 0.05 else (0, (2, 2)))
        txt.append(f"{coh.replace('-', '−')}: ρ = {rho:.2f}, {m62.fmt_p(p)}; r = {r:.2f}, {m62.fmt_p(pr)}; n = {len(q)}")
        rows.append(dict(x=x, y=y, cohort=coh, n=len(q), spearman_rho=rho, spearman_p=p, pearson_r=r, pearson_p=pr,
                         partial_rho_given_d0_rate=pr_, partial_p=pp_))
    ax.set_title("\n".join(txt), loc="left", fontsize=3.9, pad=2)


def main(a):
    plt = m62.setup()
    out = OUT / a.population
    out.mkdir(parents=True, exist_ok=True)
    M = mouse_table(a.population)
    M.to_csv(out / "mice.csv", index=False)
    print(M.groupby("cohort").size().to_dict(), "mice with day 0 and expert sessions")
    rows = []
    X = XN + XB
    fig, axs = plt.subplots(len(X), len(YN), figsize=(m62.W_IN * 0.8, 1.55 * len(X) + 0.5))
    for i, (x, xl) in enumerate(X):
        for j, (y, yl) in enumerate(YN):
            ax = axs[i, j]
            panel(ax, M, x, y, rows)
            ax.set_xlabel(xl, fontsize=4.8)
            if j == 0 or True:
                ax.set_ylabel(yl, fontsize=4.8)
    fig.suptitle(f"Day-0 effects per mouse vs expert-stage outcome ({a.population}; open: one mouse; R+ green, R− magenta;\n"
                 "OLS ± 95% CI per cohort, solid if Spearman p < 0.05)", x=0.02, y=0.998, ha="left", va="top", fontsize=6,
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.2, w_pad=0.8)
    m62.save(fig, out, "day0_predicts_expert"); plt.close(fig)
    R = pd.DataFrame(rows); R.to_csv(out / "correlations.csv", index=False)
    with pd.option_context("display.width", 200):
        print(R.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
