"""090 -- Exploratory (user 2026-10-10): does reaction time explain the across-days convergence of whisker hits onto the reward-lick CD?
Spontaneous licks have no RT, so two adjustments on the cued events only:
  long   whisker and auditory hits with RT >= RT_MIN (150 ms): the stimulus response then falls before the [-100, 0) ms pre-lick window
  match  whisker and auditory hits matched on RT: 20-ms bins, per bin min(n_WH, n_AH) events of each class drawn at random
         (N_DRAW draws, metrics averaged over draws); spontaneous licks all kept
  all    no adjustment (as 087)
Metrics per session and variant (087 metrics: units z-scored over AH + SL, 5-fold CV unit-norm CD = AH - SL, per neuron): d_WS
(WH - SL projection), rho_v (length of WH - SL), theta (angle of WH - SL to the CD); requirements as 026 (>= 4 WH, >= 8 AH, >= 8 SL)
after the adjustment. Units: stable, mean pre-lick rate >= 0.1 Hz; whole brain. Learners rule.
Figure: RT distributions (WH, AH) per cohort x stage; d_WS, rho, theta per variant, day 0 vs expert per cohort, with the number of
sessions. Statistics as 088 (MWU brackets, cohort x stage mouse permutation in the title).
Output: across_days/<ref>/rt_adjusted/{sessions.csv, rt_events.csv, rt_adjusted.{png,pdf}, stats.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../090_rt_adjusted_cd.py [--n-proc 30]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m51 = importlib.import_module("051_roc_prelick")
m002 = importlib.import_module("002_trial_slopes")
m87 = importlib.import_module("087_across_days_harmonised")
OUT = m51.OUTROOT / "rt_adjusted"
RT_MIN, BIN, N_DRAW = 0.150, 0.020, 10


def ok_counts(lab):
    return (lab == "WH").sum() >= m002.MIN_WH and min((lab == c).sum() for c in ("AH", "FA")) >= m002.MIN_EV


def session(args):
    sid, f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, rt = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["rt"].astype(float)[o]
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    ev = [dict(meta, cls=c, rt=x) for c, x in zip(lab, rt) if c in ("WH", "AH")]
    if ok.sum() < m002.MIN_UNITS:
        return [], ev
    X = X[ok]; rows = []
    # all
    if ok_counts(lab):
        rows.append(dict(meta, variant="all", n_WH=int((lab == "WH").sum()), n_AH=int((lab == "AH").sum()), **m87.metrics(X, lab, seed)))
    # long RT
    keep = (lab == "FA") | (rt >= RT_MIN)
    if ok_counts(lab[keep]):
        rows.append(dict(meta, variant="long", n_WH=int((lab[keep] == "WH").sum()), n_AH=int((lab[keep] == "AH").sum()),
                         **m87.metrics(X[:, keep], lab[keep], seed)))
    # RT-matched WH vs AH
    rng = np.random.default_rng(seed)
    b = np.floor(rt / BIN)
    iw, ia, isl = np.where(lab == "WH")[0], np.where(lab == "AH")[0], np.where(lab == "FA")[0]
    bins = np.intersect1d(b[iw], b[ia])
    nm = sum(min((b[iw] == k).sum(), (b[ia] == k).sum()) for k in bins)
    if nm >= max(m002.MIN_WH, m002.MIN_EV) and len(isl) >= m002.MIN_EV:
        acc = []
        for d in range(N_DRAW):
            sel = []
            for k in bins:
                w_, a_ = iw[b[iw] == k], ia[b[ia] == k]; n_ = min(len(w_), len(a_))
                sel += list(rng.choice(w_, n_, replace=False)) + list(rng.choice(a_, n_, replace=False))
            idx = np.sort(np.r_[np.array(sel, int), isl])
            acc.append(m87.metrics(X[:, idx], lab[idx], seed + d))
        mm = {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
        rows.append(dict(meta, variant="match", n_WH=int(nm), n_AH=int(nm), **mm))
    return rows, ev


def main(a):
    m62 = importlib.import_module("062_pub_convergence_figures")
    m61 = importlib.import_module("061_roc_prelick_learners")
    m88 = importlib.import_module("088_cosyne_across_days_v2")
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.replot:
        W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
        W = W[W.cohort.isin(["R+", "R-"])].copy()
        W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
        ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
        jobs = []
        for i, r in enumerate(ss.itertuples()):
            f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
            if f.exists():
                Ws = W[W.session_id == r.session_id]
                jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0]),
                             Ws[["electrode_group", "cluster_id", "quality_label"]], i))
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            res = pool.map(session, jobs)
        pd.DataFrame([x for r in res for x in r[0]]).to_csv(OUT / "sessions.csv", index=False)
        pd.DataFrame([x for r in res for x in r[1]]).to_csv(OUT / "rt_events.csv", index=False)
    D = m61.learner_filter(pd.read_csv(OUT / "sessions.csv")); E = m61.learner_filter(pd.read_csv(OUT / "rt_events.csv"))
    plt = m62.setup(); rng = np.random.default_rng(0)
    VAR = (("all", "all events"), ("long", f"RT ≥ {int(RT_MIN * 1e3)} ms"), ("match", "WH / AH RT-matched"))
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.95))
    gs = fig.add_gridspec(4, 3, hspace=1.0, wspace=0.6, left=0.08, right=0.98, top=0.95, bottom=0.06)
    # RT distributions
    edges = np.arange(0, 1.05, 0.025)
    for k, (c, s_) in enumerate(m88.GROUPS[:2] + m88.GROUPS[2:3]):
        pass
    sub = gs[0, :].subgridspec(1, 4, wspace=0.35)
    for k, g in enumerate(m88.GROUPS):
        ax = fig.add_subplot(sub[k]); d = E[(E.cohort == g[0]) & (E.stage == g[1])]
        for cl in ("AH", "WH"):
            x = d[d.cls == cl].rt.dropna()
            ax.hist(x, edges, histtype="step", density=True, color=m62.CL[cl], lw=1.0, label=f"{cl} (median {x.median() * 1e3:.0f} ms)")
        ax.axvline(RT_MIN, color="0.4", lw=0.6, ls=":")
        ax.set_title(m88.GLAB[g], color=m62.COH[g[0]], fontsize=5.5); ax.set_xlabel("RT (s)", fontsize=5); ax.legend(frameon=False, fontsize=4.2)
    for i, (col, ylab, ttl) in enumerate((("d_WS", "WH − SL projection on CD\n(per neuron)", "WH − SL"),
                                           ("rho_v", "length ρ of WH − SL\n(per neuron)", "Length ρ"),
                                           ("theta", "angle θ of WH − SL to CD (°)", "Angle θ"))):
        for j, (v, vl) in enumerate(VAR):
            ax = fig.add_subplot(gs[1 + i, j]); ax.set_box_aspect(0.8)
            m88.mean_panel2(ax, D[D.variant == v], col, ylab if j == 0 else "", rng, f"090 {v} {col}", f"{ttl}: {vl}", ref=0 if col == "d_WS" else None)
    fig.suptitle("Reaction-time adjustments of the reward-lick CD across days (exploratory)", x=0.02, y=0.995, ha="left", va="top", fontsize=7, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rt_adjusted.{ext}", dpi=300)
    S = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("090")]); S.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert",
             "n_R− learning", "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
