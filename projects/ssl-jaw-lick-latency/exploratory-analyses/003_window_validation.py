"""003 -- Neural validation of lick-referenced pre-movement windows (user 2026-10-13, intended use: pre-movement analyses in sessions
without video). Question: does a reward-lick CD analysis run in a window defined from the lick alone reproduce the result obtained with
the true (video) jaw onset?
Data: ssl-prelick-convergence 105 lick-aligned rates (stable tested units; 50-ms bins centred -400 .. +100 ms around the corrected first
lick / spontaneous-lick onset, baseline-corrected); all R+ / R- sessions; learners rule for day-0 sessions in the statistics.
Windows W(X) = [lick - X - 100, lick - X) ms, X in {0, 100, 200, 300} (bins centred inside the window).
Per session (whole brain) and window: CD = AH - SL, 5-fold CV over AH + SL, units z-scored on the training folds; held-out AH / SL and all WH
projected (WH averaged over folds); per neuron (/ sqrt N); d_WS = p_WH - p_SL; d_AS = p_AH - p_SL (held-out separation).
Reference: d_WS of ssl-prelick-convergence 087 with PRELICK_ALIGN=jaw (100 ms before the video jaw onset, sessions with video).
Comparisons: (1) R+ day 0 vs expert and R+ vs R- at expert per window (Mann-Whitney, Welch); (2) across sessions with video, Spearman
correlation of d_WS(X) with jaw-aligned d_WS; (3) d_AS (signal strength) per window.
Caveat: unit sets differ slightly between 105 (stable tested units) and 087 (stable units, pre-lick rate >= 0.1 Hz).
Outputs: combined_results_ks4/ssl-jaw-lick-latency/[all_sessions/]{tables/window_validation*.csv, figures/jaw_lick_window_validation.*}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../003_window_validation.py [--n-proc 25]
"""
import argparse
import importlib
import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SSL_JAW_POP", "all")
m001 = importlib.import_module("001_jaw_lick_latency")
m51, m105, m61, OUT = m001.m51, m001.m105, m001.m61, m001.OUT
XS = (0, 100, 200, 300)
COH = {"R+": "#00B400", "R-": "#C800C8"}


def cd(X, lab, seed):
    from sklearn.model_selection import StratifiedKFold
    idx = np.where(np.isin(lab, ["AH", "SL"]))[0]; y = (lab[idx] == "AH").astype(int); wh = np.where(lab == "WH")[0]
    pa, ps, pw = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y):
        mu, sd = X[:, idx[tr]].mean(1), X[:, idx[tr]].std(1); sd[sd == 0] = 1
        Z = (X - mu[:, None]) / sd[:, None]
        u = Z[:, idx[tr][y[tr] == 1]].mean(1) - Z[:, idx[tr][y[tr] == 0]].mean(1); u /= max(np.linalg.norm(u), 1e-12)
        p = u @ Z
        pa += list(p[idx[te][y[te] == 1]]); ps += list(p[idx[te][y[te] == 0]]); pw.append(p[wh])
    s = np.sqrt(X.shape[0]); pwm = np.mean(pw, 0)
    return (np.mean(pwm) - np.mean(ps)) / s, (np.mean(pa) - np.mean(ps)) / s


def session(args):
    f, = args
    z = np.load(f, allow_pickle=True)
    lab, L, c = z["lick_lab"].astype(str), z["lickbins"].astype(float), z["lick_c"]
    if (lab == "WH").sum() < 4 or min((lab == k).sum() for k in ("AH", "SL")) < 8 or L.shape[0] < 10:
        return []
    out = []
    for x in XS:
        m = (c >= (-x - 100) / 1e3) & (c < -x / 1e3)
        X = L[:, :, m].mean(2)
        dws, das = cd(X, lab, 0)
        out.append(dict(session_id=f.stem, X=x, d_WS=dws, d_AS=das))
    return out


def two(a, b):
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def main(a):
    (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(parents=True, exist_ok=True)
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        V = pd.DataFrame([x for r in pool.map(session, [(f,) for f in files]) for x in r])
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    V = m61.learner_filter(V.merge(meta, on="session_id"))
    J = pd.read_csv(m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}_jaw" / "harmonised" / "sessions_areas.csv")
    J = J[J.area == "whole_brain"][["session_id", "d_WS"]].rename(columns={"d_WS": "d_WS_jaw"})
    V = V.merge(J, on="session_id", how="left"); V.to_csv(OUT / "tables" / "window_validation_sessions.csv", index=False)
    rows = []
    for x in XS:
        d = V[V.X == x]
        g = lambda co, st, col="d_WS": d[(d.cohort == co) & (d.stage == st)][col].dropna()
        r = dict(X=x)
        for co in ("R+", "R-"):
            r[f"{co} D0"], r[f"{co} E"] = g(co, "learning").mean(), g(co, "expert").mean()
            r[f"{co} D0vsE p_MWU"], r[f"{co} D0vsE p_Welch"] = two(g(co, "learning"), g(co, "expert"))
        r["E R+vsR- p_MWU"], r["E R+vsR- p_Welch"] = two(g("R+", "expert"), g("R-", "expert"))
        r["d_AS mean"] = d.d_AS.mean()
        q = d.dropna(subset=["d_WS_jaw"]); r["n_video"] = len(q)
        r["spearman_vs_jaw"], r["p_spearman"] = stats.spearmanr(q.d_WS, q.d_WS_jaw)
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(OUT / "tables" / "window_validation.csv", index=False)
    pd.set_option("display.width", 250); print(T.round(4).to_string(index=False))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.4, "axes.labelsize": 5.2, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.6))
    ax = axs[0]
    for co in ("R+", "R-"):
        for st, ls in (("learning", (0, (2.5, 1.5))), ("expert", "-")):
            m_ = [V[(V.X == x) & (V.cohort == co) & (V.stage == st)].d_WS.mean() for x in XS]
            s_ = [V[(V.X == x) & (V.cohort == co) & (V.stage == st)].d_WS.sem() for x in XS]
            ax.errorbar(XS, m_, s_, color=COH[co], ls=ls, marker="o", ms=2.5, lw=0.9, capsize=0, label=f"{co.replace('-', '−')} {'D0' if st == 'learning' else 'E'}")
    ax.axhline(0, color="0.7", lw=0.5); ax.invert_xaxis(); ax.set_xlabel("window end before the lick X (ms)"); ax.set_ylabel("d_WS (per neuron)")
    ax.legend(frameon=False, fontsize=4.0); ax.set_title("WH position on the CD\nby window", fontsize=5.4)
    ax = axs[1]
    ax.plot(XS, [-np.log10(T.set_index("X").loc[x, "R+ D0vsE p_MWU"]) for x in XS], "o-", color=COH["R+"], ms=2.5, lw=0.9, label="R+ D0 vs E")
    ax.plot(XS, [-np.log10(T.set_index("X").loc[x, "E R+vsR- p_MWU"]) for x in XS], "o-", color="0.2", ms=2.5, lw=0.9, label="E: R+ vs R−")
    ax.axhline(-np.log10(0.05), color="0.6", lw=0.5, ls=":"); ax.invert_xaxis(); ax.set_xlabel("X (ms)"); ax.set_ylabel("−log10 p (Mann-Whitney)")
    ax.legend(frameon=False, fontsize=4.0); ax.set_title("Effects by window", fontsize=5.4)
    ax = axs[2]
    ax.plot(XS, T["d_AS mean"], "o-", color="#2c2cdb", ms=2.5, lw=0.9); ax.invert_xaxis(); ax.set_xlabel("X (ms)"); ax.set_ylabel("held-out AH − SL (per neuron)")
    ax.set_title("CD signal strength", fontsize=5.4)
    ax = axs[3]
    ax.plot(XS, T.spearman_vs_jaw, "o-", color="k", ms=2.5, lw=0.9); ax.invert_xaxis(); ax.set_ylim(0, 1)
    ax.set_xlabel("X (ms)"); ax.set_ylabel("Spearman r with jaw-aligned d_WS")
    ax.set_title(f"Agreement with video-based window\n(n = {int(T.n_video.max())} sessions)", fontsize=5.4)
    for ax, l in zip(axs, "abcd"):
        ax.text(-0.35, 1.15, l, transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Lick-referenced windows vs the video jaw onset: does the CD result hold?", x=0.01, y=1.14, ha="left", fontsize=7, weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"jaw_lick_window_validation.{ext}", dpi=300, bbox_inches="tight")
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    main(ap.parse_args())
