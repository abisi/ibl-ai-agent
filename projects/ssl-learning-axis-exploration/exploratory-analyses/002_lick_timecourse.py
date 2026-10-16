"""002 -- Lick-aligned time course on the reward-lick CD, day 0 vs expert (user 2026-10-16, "explore 1-4": item 4).
Rates: ssl-prelick-convergence 105 per-event lick-aligned 50-ms windows (centres -400 .. +100 ms, 25-ms step; baseline-corrected as 051) for
WH and AH (corrected first lick) and SL (spontaneous-lick bout onset). Units: stable good / mua with mean raw pre-lick rate >= 0.1 Hz (002 units).
CD: unit-norm AH - SL axis from the mean of the windows inside [-100, 0) ms (centres -75, -50, -25), 5-fold CV over AH + SL events, units
z-scored on the training folds; every time window of held-out AH / SL events and of all WH events (fold average) projected on the fold's axis,
per neuron. Per session: class time courses; lambda(t) = (WH - SL) / (AH - SL) at each window, with the AH - SL separation of the [-100, 0) ms
window as denominator (so lambda(t) is in units of the window separation; separation >= SEP_MIN); WH divergence onset: first window from
which WH - SL stays > 0.25 x the window separation for 3 consecutive windows.
Group: mean +- s.e.m. per cohort x stage; tests per window (day 0 vs expert, MWU), BH over windows; onset latency day 0 vs expert.
Output: tables/002_timecourse.csv, 002_onsets.csv, 002_stats.csv; figures/002_lick_timecourse.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C

m002 = importlib.import_module("002_trial_slopes")
m105 = importlib.import_module("105_trial_rates")
SEP_MIN, MIN_UNITS = 0.05, 10


def session(args):
    sid, meta, W_s, seed = args
    f = m105.OUT / f"{sid}.npz"
    if not f.exists():
        return None
    from sklearn.model_selection import StratifiedKFold
    z = np.load(f, allow_pickle=True)
    lab = z["lick_lab"].astype(str); L = z["lickbins"].astype(float); cc = z["lick_c"].astype(float) * (1e3 if z["lick_c"].max() < 5 else 1)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))).merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    win = (cc > -100) & (cc < 0)
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & np.isfinite(L).all(axis=(1, 2))
    if ok.sum() < MIN_UNITS or min((lab == c).sum() for c in ("AH", "SL")) < 8 or (lab == "WH").sum() < 4:
        return None
    X = np.transpose(L[ok], (1, 0, 2))                         # events x units x windows
    Xw = X[:, :, win].mean(2)
    ia = np.where(np.isin(lab, ["AH", "SL"]))[0]; y = (lab[ia] == "AH").astype(int); w = np.where(lab == "WH")[0]
    P = np.full((len(lab), len(cc)), np.nan); Pw = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(ia, y):
        A = Xw[ia[tr]]; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
        u = ((A[y[tr] == 1] - mu) / sd).mean(0) - ((A[y[tr] == 0] - mu) / sd).mean(0); u = u / max(np.linalg.norm(u), 1e-12)
        proj = lambda idx: np.einsum("eut,u->et", (X[idx] - mu[None, :, None]) / sd[None, :, None], u) / np.sqrt(X.shape[1])
        P[ia[te]] = proj(ia[te]); Pw.append(proj(w))
    P[w] = np.mean(Pw, 0)
    m = {c: np.nanmean(P[lab == c], 0) for c in ("WH", "AH", "SL")}
    sep = np.nanmean(m["AH"][win]) - np.nanmean(m["SL"][win])
    rows = [dict(meta, t=float(t), WH=m["WH"][k], AH=m["AH"][k], SL=m["SL"][k], sep=sep,
                 lam=(m["WH"][k] - m["SL"][k]) / sep if sep >= SEP_MIN else np.nan) for k, t in enumerate(cc)]
    d = (m["WH"] - m["SL"]) / sep if sep >= SEP_MIN else np.full(len(cc), np.nan)
    on = np.nan
    for k in range(len(cc) - 2):
        if np.all(d[k:k + 3] > 0.25):
            on = float(cc[k]); break
    return rows, dict(meta, onset=on, sep=sep)


def main(a):
    W = pd.read_parquet(C.m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy(); W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    W = W[W.mean_rate_raw >= C.m51.MIN_FR] if "mean_rate_raw" in W else W
    jobs = [(sid, dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0]),
             g[["electrode_group", "cluster_id", "quality_label"]], zlib.crc32(sid.encode())) for sid, g in W.groupby("session_id")]
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = [r for r in pool.map(session, jobs) if r is not None]
    T = C.population(pd.DataFrame([x for r, _ in res for x in r])); O = C.population(pd.DataFrame([o for _, o in res]))
    T.to_csv(C.TAB / "002_timecourse.csv", index=False); O.to_csv(C.TAB / "002_onsets.csv", index=False)
    St = []
    for c in ("R+", "R-"):
        for t, g in T[T.cohort == c].groupby("t"):
            for q in ("lam", "WH", "AH", "SL"):
                a0, b0 = g[g.stage == "learning"][q].dropna(), g[g.stage == "expert"][q].dropna()
                St.append(dict(cohort=c, t=t, measure=q, mean_d0=a0.mean(), mean_e=b0.mean(), p=stats.mannwhitneyu(a0, b0).pvalue if len(a0) > 2 and len(b0) > 2 else np.nan))
        a0, b0 = O[(O.cohort == c) & (O.stage == "learning")].onset.dropna(), O[(O.cohort == c) & (O.stage == "expert")].onset.dropna()
        St.append(dict(cohort=c, measure="onset", mean_d0=a0.mean(), mean_e=b0.mean(), n_d0=len(a0), n_e=len(b0),
                       p=stats.mannwhitneyu(a0, b0).pvalue if len(a0) > 2 and len(b0) > 2 else np.nan))
    St = pd.DataFrame(St)
    for (c, q), g in St[St.measure != "onset"].groupby(["cohort", "measure"]):
        St.loc[g.index, "q"] = C.bh(g.p)
    St.to_csv(C.TAB / "002_stats.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    CL = {"WH": "#f7b519", "AH": "#2c2cdb", "SL": "#262626"}
    fig, axs = plt.subplots(2, 4, figsize=(7.6, 3.9), gridspec_kw=dict(wspace=0.55, hspace=0.8))
    for i, c in enumerate(("R+", "R-")):
        for j, s_ in enumerate(("learning", "expert")):
            ax = axs[i, j]; g = T[(T.cohort == c) & (T.stage == s_)]
            for q in ("SL", "AH", "WH"):
                p = g.pivot_table(index="session_id", columns="t", values=q)
                ax.plot(p.columns, p.mean(), color=CL[q], lw=1, label=q); ax.fill_between(p.columns, p.mean() - p.sem(), p.mean() + p.sem(), color=CL[q], alpha=0.2, lw=0)
            ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0); ax.axvline(0, color="0.4", lw=0.5, ls="--")
            ax.set_title(f"{C.GLAB[(c, s_)]} ({g.session_id.nunique()})", color=C.COH[c], fontsize=6); ax.set_xlabel("time from lick (ms)")
            if j == 0:
                ax.set_ylabel("projection on CD (per neuron)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=4.6); ax.text(-0.4, 1.15, "a", transform=ax.transAxes, fontsize=8, weight="bold")
        ax = axs[i, 2]
        for s_, ls in (("learning", (0, (3, 1.5))), ("expert", "-")):
            p = T[(T.cohort == c) & (T.stage == s_)].pivot_table(index="session_id", columns="t", values="lam")
            ax.plot(p.columns, p.mean(), color=C.COH[c], ls=ls, lw=1, label="day 0" if s_ == "learning" else "expert")
            ax.fill_between(p.columns, p.mean() - p.sem(), p.mean() + p.sem(), color=C.COH[c], alpha=0.15, lw=0)
        sg = St[(St.cohort == c) & (St.measure == "lam") & (St.q < 0.05)]
        ytop = ax.get_ylim()[1]
        for t in sg.t:
            ax.plot([t - 12.5, t + 12.5], [ytop, ytop], color="k", lw=2)
        ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color="0.7", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.5, ls="--")
        ax.set_xlabel("time from lick (ms)"); ax.set_ylabel("λ(t) (re. window separation)"); ax.legend(frameon=False, fontsize=4.6)
        ax.set_title("λ over time (bars: D0 ≠ E, BH q < 0.05)", fontsize=5.6)
        if i == 0:
            ax.text(-0.4, 1.15, "b", transform=ax.transAxes, fontsize=8, weight="bold")
        ax = axs[i, 3]
        for k, s_ in enumerate(("learning", "expert")):
            v = O[(O.cohort == c) & (O.stage == s_)].onset.dropna()
            ax.scatter(k + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=4, color=C.COH[c], alpha=0.4, lw=0)
            ax.errorbar(k, v.mean(), v.sem(), fmt="o", ms=3.5, color=C.COH[c], mfc="white" if k == 0 else C.COH[c], capsize=0)
        r = St[(St.cohort == c) & (St.measure == "onset")].iloc[0]
        ax.set_xticks([0, 1], ["D0", "E"]); ax.set_xlim(-0.6, 1.6); ax.set_ylabel("WH divergence onset (ms)")
        ax.set_title(f"onset of WH − SL divergence\nD0 vs E {C.fp(r.p)} (n {int(r.n_d0)} / {int(r.n_e)})", fontsize=5.6)
        if i == 0:
            ax.text(-0.45, 1.15, "c", transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Lick-aligned trajectories on the reward-lick CD (CD from [−100, 0) ms; held-out AH / SL, fold-averaged WH; per neuron)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"002_lick_timecourse.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(St[St.measure == "onset"].to_string(index=False)); print(St[(St.measure == "lam") & (St.q < 0.1)].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20)
    main(ap.parse_args())
