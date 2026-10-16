"""123 -- Single-neuron similarity of whisker-hit pre-lick activity to spontaneous licks and auditory hits, signed ROC on all trials
(user 2026-10-14: no cross-validation; show that in expert R+ WH pre-lick activity is less SL-like and more AH-like than on day 0).
Per neuron, s = 2 AUC - 1 with a two-sided label-permutation p (N_PERM): s_WS = WH vs SL (> 0: WH above SL), s_WA = WH vs AH (< 0: WH
below AH), s_AS = AH vs SL (reward-lick modulation). Signed s averages to 0 under noise, so session means need no cross-validation.
Neurons: 081 tested units (stable, mean pre-lick rate >= 0.1 Hz) of the non-sensory classes (stimulus class not "sensory"; lick-gated
included, as 119 h). Two problems and their fixes (review of 2026-10-14):
  A  event counts differ (R+ expert ~90 WH per session, R- expert ~30) -> significance-based fractions are compared on count-matched
     draws: N_WH WH, N_AH AH, N_SL SL per draw, N_DRAW draws per session.
  B  selecting reward-lick neurons on AH vs SL reuses the AH and SL events (biases s_WS up and s_WA down, by an amount that depends on
     the AH / SL counts) -> B1 selection and measurement on the same count-matched draw (bias equal across groups); B2 no selection:
     all non-sensory neurons, means weighted by max(s_AS, 0).
Variants (session summaries):
  plain              selection (s_AS > 0, p < 0.05) and measures on all events                          [reference, biased]
  B1_matched         per draw: selection and measures on the draw; summaries averaged over draws with >= MIN_SEL selected neurons
  B2_weighted        all non-sensory neurons, all events, weights max(s_AS, 0)
  B2_weighted_matched  as B2 on the count-matched draws (weights from the draw), averaged over draws
  unweighted         all non-sensory neurons, all events, plain means (no selection at all; diluted by non-reward-lick neurons)
Measures: mean signed s_WS, s_WA, s_AS; for plain and B1_matched the categories of the selected neurons:
  AH-like       WH > SL significant, WH vs AH not significant
  intermediate  WH > SL significant and WH < AH significant
  SL-like       WH vs SL not significant, WH < AH significant
  other         neither significant, or opposite signs (WH < SL / WH > AH)
Statistics (unit = session, learners rule): 062 group_stats (MWU + Welch, mouse-level cohort x stage permutation).
Output: <OUTROOT>/single_neuron_signed/{units.parquet, sessions.csv, sessions_<pop>.csv, stats_<pop>.csv, means_<pop>, categories_<pop>,
        scatter_<pop>}.{png,pdf,svg}; run lick-aligned and with PRELICK_ALIGN=jaw (118 events, 100 ms before jaw-opening onset).
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable [PRELICK_ALIGN=jaw] python .../123_single_neuron_signed.py [--n-proc 30] [--replot]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m81 = importlib.import_module("081_single_unit_classes")
OUT = m51.OUTROOT / "single_neuron_signed"
LICKROOT = m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}"            # 081 stimulus classes (alignment independent)
N_PERM, N_DRAW = 200, 10
N_WH, N_AH, N_SL = 8, 8, 16
MIN_WH, MIN_AH, MIN_SL, MIN_SEL = 4, 4, 8, 3
CATS = ["AH-like", "intermediate", "SL-like", "other"]
CCOL = {"AH-like": "#2c2cdb", "intermediate": "#8f8fd6", "SL-like": "#262626", "other": "#d9d9d9"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
VARIANTS = ["plain", "B1_matched", "B2_weighted", "B2_weighted_matched", "unweighted"]
VLAB = {"plain": "Selected, all events\n(reference, biased)", "B1_matched": "Selected on\ncount-matched draws",
        "B2_weighted": "All non-sensory,\nweighted by s_AS", "B2_weighted_matched": "Weighted, on\ncount-matched draws",
        "unweighted": "All non-sensory,\nunweighted"}


def perm_s(X, a, b, rng):
    """s = 2 AUC - 1 (a > b) per unit (rows of X) and two-sided permutation p; one shared set of label permutations"""
    idx = np.r_[a, b]; na, nb = len(a), len(b)
    R = np.apply_along_axis(rankdata, 1, X[:, idx])
    f = lambda cols: 2 * (R[:, cols].sum(1) - na * (na + 1) / 2) / (na * nb) - 1
    obs = f(np.arange(na)); cnt = np.zeros(len(X))
    for _ in range(N_PERM):
        cnt += np.abs(f(rng.permutation(na + nb)[:na])) >= np.abs(obs) - 1e-12
    return obs, (1 + cnt) / (1 + N_PERM)


def tests(X, w, a, s, rng):
    sAS, pAS = perm_s(X, a, s, rng); sWS, pWS = perm_s(X, w, s, rng); sWA, pWA = perm_s(X, w, a, rng)
    return dict(sAS=sAS, pAS=pAS, sWS=sWS, pWS=pWS, sWA=sWA, pWA=pWA)


def category(t):
    ws = (t["pWS"] < 0.05) & (t["sWS"] > 0); wa = (t["pWA"] < 0.05) & (t["sWA"] < 0)
    nws, nwa = t["pWS"] >= 0.05, t["pWA"] >= 0.05
    return np.where(ws & nwa, "AH-like", np.where(ws & wa, "intermediate", np.where(nws & wa, "SL-like", "other")))


def sel_summary(t):
    """selected reward-lick neurons (s_AS > 0, p < 0.05): signed means and category fractions"""
    m = (t["pAS"] < 0.05) & (t["sAS"] > 0)
    if m.sum() < MIN_SEL:
        return None
    c = category({k: v[m] for k, v in t.items()})
    return dict(n=int(m.sum()), sWS=t["sWS"][m].mean(), sWA=t["sWA"][m].mean(), sAS=t["sAS"][m].mean(),
                **{f"frac_{k}": float(np.mean(c == k)) for k in CATS})


def w_summary(t):
    w = np.clip(t["sAS"], 0, None)
    if w.sum() <= 0:
        return None
    return dict(n=len(w), sWS=np.average(t["sWS"], weights=w), sWA=np.average(t["sWA"], weights=w), sAS=np.average(t["sAS"], weights=w))


def avg(ds):
    ds = [d for d in ds if d is not None]
    if len(ds) < N_DRAW / 2:
        return None
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]} | {"n_draws": len(ds)}


def session(args):
    sid, f, units = args
    z = np.load(f, allow_pickle=True)
    lab = z["cls"]; X = z["rates"].astype(float)
    iw, ia, isl = (np.where(lab == c)[0] for c in ("WH", "AH", "FA"))
    if len(iw) < MIN_WH or len(ia) < MIN_AH or len(isl) < MIN_SL:
        return [], []
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    K = K[K.nclass.isin(m81.CLASSES) & (K.nclass != "sensory")]
    if len(K) < MIN_SEL:
        return [], []
    Xs = X[K.row.to_numpy()]
    rng = np.random.default_rng(zlib.crc32(f"{sid}|123".encode()))
    t = tests(Xs, iw, ia, isl, rng)
    cat = category(t)
    urows = [dict(session_id=sid, electrode_group=r.electrode_group, cluster_id=r.cluster_id, nclass=r.nclass, category=cat[i],
                  **{k: float(v[i]) for k, v in t.items()}) for i, r in enumerate(K.itertuples())]
    base = dict(session_id=sid, n_wh=len(iw), n_ah=len(ia), n_sl=len(isl), n_ns=len(K))
    srows = []
    for v, d in [("plain", sel_summary(t)), ("B2_weighted", w_summary(t)),
                 ("unweighted", dict(n=len(K), sWS=t["sWS"].mean(), sWA=t["sWA"].mean(), sAS=t["sAS"].mean()))]:
        if d is not None:
            srows.append(dict(base, variant=v, **d))
    if len(iw) >= N_WH and len(ia) >= N_AH and len(isl) >= N_SL:
        sel, wts = [], []
        for _ in range(N_DRAW):
            td = tests(Xs, rng.choice(iw, N_WH, replace=False), rng.choice(ia, N_AH, replace=False), rng.choice(isl, N_SL, replace=False), rng)
            sel.append(sel_summary(td)); wts.append(w_summary(td))
        for v, d in [("B1_matched", avg(sel)), ("B2_weighted_matched", avg(wts))]:
            if d is not None:
                srows.append(dict(base, variant=v, **d))
    return urows, srows


def compute(n_proc):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(LICKROOT / "single_unit_classes" / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(U0.session_id.unique())]
    jobs = [(r.session_id, r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz",
             U0[U0.session_id == r.session_id][["electrode_group", "cluster_id", "nclass"]].drop_duplicates()) for r in ss.itertuples()]
    jobs = [j for j in jobs if j[1].exists()]
    print("sessions with trial files", len(jobs), "tag", m51.TAG, flush=True)
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = list(pool.imap_unordered(session, jobs))
    meta = U0.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    U = pd.DataFrame([x for r in res for x in r[0]]).merge(meta, on="session_id", how="left")
    S = pd.DataFrame([x for r in res for x in r[1]]).merge(meta, on="session_id", how="left")
    U.to_parquet(OUT / "units.parquet", index=False); S.to_csv(OUT / "sessions.csv", index=False)
    print("computed", len(U), "units", S.session_id.nunique(), "sessions", flush=True)


def save(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)


def figure(pop):
    plt = m62.setup()
    S = pd.read_csv(OUT / "sessions.csv"); U = pd.read_parquet(OUT / "units.parquet")
    S, U = S[S.cohort.isin(["R+", "R-"])], U[U.cohort.isin(["R+", "R-"])]
    if pop == "learners":
        S, U = m61.learner_filter(S), m61.learner_filter(U)
    S.to_csv(OUT / f"sessions_{pop}.csv", index=False)
    align = "lick" if m51.ALIGN == "lick" else "jaw-onset"
    rng = np.random.default_rng(1)
    m62.STATS.clear()
    # 1 signed means per variant
    rows = [("sWS", "Mean signed s_WS\n(WH vs SL; > 0: WH above SL)"), ("sWA", "Mean signed s_WA\n(WH vs AH; < 0: WH below AH)"),
            ("sAS", "Mean signed s_AS\n(AH vs SL)")]
    fig, axs = plt.subplots(len(rows), len(VARIANTS), figsize=(7.2, 4.9), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.85, bottom=0.06, hspace=0.75, wspace=0.6)
    for j, v in enumerate(VARIANTS):
        for i, (col, lab) in enumerate(rows):
            m62.dots_panel(axs[i, j], S[S.variant == v], col, lab if j == 0 else "", rng, f"123 {v} {col}", VLAB[v] if i == 0 else "",
                           ref=[(0, "0.6")])
    fig.suptitle(f"Non-sensory neurons, signed pre-lick ROC selectivity ({align}-aligned [-100, 0) ms; ○ day 0, ● expert; unit = session)",
                 fontsize=6.5, y=0.985)
    save(fig, f"means_{pop}"); plt.close(fig)
    # 2 categories: stacked bars + per-category statistics
    G = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
    fig = plt.figure(figsize=(6.6, 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 0.9, 0.9], wspace=0.6, left=0.07, right=0.99, top=0.78, bottom=0.2)
    for j, v in enumerate(["plain", "B1_matched"]):
        ax = fig.add_subplot(gs[j]); d = S[S.variant == v]
        for gi, g in enumerate(G):
            q = d[(d.cohort == g[0]) & (d.stage == g[1])]; b = 0
            for k in CATS:
                h = q[f"frac_{k}"].mean(); ax.bar(gi, h, bottom=b, color=CCOL[k], width=0.7, edgecolor="white", lw=0.4); b += h
            ax.text(gi, 1.02, f"n={len(q)}", ha="center", fontsize=4.6, color="0.4")
        ax.set_xticks(range(4)); ax.set_xticklabels(["R+\nday 0", "R+\nexpert", "R−\nday 0", "R−\nexpert"], fontsize=5)
        for tl, g in zip(ax.get_xticklabels(), G):
            tl.set_color(COH[g[0]])
        ax.set_ylim(0, 1.08); ax.set_ylabel("Fraction of reward-lick neurons" if j == 0 else "")
        ax.set_title(VLAB[v].replace("\n", " "), fontsize=5.8)
    h = [plt.Rectangle((0, 0), 1, 1, color=CCOL[k]) for k in CATS]
    fig.legend(h, ["AH-like (WH > SL; WH ≈ AH)", "intermediate (WH > SL; WH < AH)", "SL-like (WH ≈ SL; WH < AH)", "other / undetermined"],
               frameon=False, fontsize=5, ncol=4, loc="upper left", bbox_to_anchor=(0.05, 1.0))
    for j, k in enumerate(["AH-like", "SL-like"]):
        ax = fig.add_subplot(gs[2 + j])
        m62.dots_panel(ax, S[S.variant == "B1_matched"], f"frac_{k}", f"Fraction {k}" + "\n(count-matched)", rng, f"123 B1 frac_{k}", "")
    save(fig, f"categories_{pop}"); plt.close(fig)
    for k in CATS:
        m62.group_stats(S[S.variant == "plain"], f"frac_{k}", k, rng, f"123 plain frac_{k}")
    # 3 per-neuron scatter s_WS vs s_WA, selected reward-lick neurons (all events)
    R = U[(U.pAS < 0.05) & (U.sAS > 0)]
    fig, axs = plt.subplots(2, 2, figsize=(3.0, 3.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.88, bottom=0.13, wspace=0.12, hspace=0.25)
    for i, c in enumerate(["R+", "R-"]):
        for j, st in enumerate(["learning", "expert"]):
            ax = axs[i, j]; q = R[(R.cohort == c) & (R.stage == st)]
            ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
            ax.scatter(q.sWS, q.sWA, s=1.5, color=[CCOL[k] if k != "other" else "0.7" for k in q.category], lw=0, alpha=0.6, rasterized=True)
            ax.scatter(q.sWS.mean(), q.sWA.mean(), s=22, color=COH[c], edgecolor="white", lw=0.6, zorder=3)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
            if i == 0: ax.set_title("day 0" if st == "learning" else "expert")
            if j == 0: ax.set_ylabel(f"{c.replace('-', '−')}\ns_WA (WH vs AH)", color=COH[c])
            if i == 1: ax.set_xlabel("s_WS (WH vs SL)")
            ax.text(0.97, 0.04, f"n={len(q)}", transform=ax.transAxes, ha="right", fontsize=4.6, color="0.4")
    fig.suptitle("Reward-lick neurons (non-sensory, all events); AH-like = lower right on the x axis", fontsize=5.6, y=0.98)
    save(fig, f"scatter_{pop}"); plt.close(fig)
    St = pd.DataFrame(m62.STATS); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    cols = ["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert",
            "n_R− learning", "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R+ L vs E p_Welch", "R- L vs E p_MWU",
            "expert R+ vs R- p_MWU", "expert R+ vs R- p_Welch"]
    pd.set_option("display.width", 300)
    print(St[[c for c in cols if c in St]].round(4).to_string())
    print(S.groupby(["variant", "cohort", "stage"])["n"].mean().round(1).to_string())


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--n-proc", type=int, default=30); p.add_argument("--replot", action="store_true")
    p.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(p.parse_args())
