"""015 -- Within-session drift on the reward-lick coding direction per area group (user 2026-10-07: per-area breakdown as in
the across-day results, Fig. 4).

Same events, units and axis as 002 / 011 (PRELICK_REF=sl: WH, AH, SL 100 ms before the corrected first lick; good + mua,
pre-lick rate >= 0.1 Hz; >= 4 WH, >= 8 AH, >= 8 SL), but the 5-fold AH - SL mean-difference axis is built from the units
of one area group only (area_group of prelick_units; >= MIN_UNITS = 5 units, as 057), kept if the held-out AH-vs-SL
d' >= 0.3. Per session x area group and class k: OLS slope beta_k of the CD score against normalised session time tau;
drifts beta_WH - beta_SL and beta_WH - beta_AH (011 definitions).
Inclusion (062 area_qualify): an area group enters a cohort if it has >= 3 sessions at both stages in that cohort; R+ vs R-
tests only where both cohorts qualify.
Statistics per area group (unit = session): drift vs 0 per cohort x stage (Wilcoxon); day 0 R+ vs R- and expert R+ vs R-
(Mann-Whitney U and Welch t; mouse-level cohort permutation); cohort x stage interaction (mouse-level permutation, 001);
Benjamini-Hochberg over area groups for the day-0 permutation p.
Output: within_day/<ref>/area_cd_drift/area_sessions.csv (all sessions), area_trial_scores.parquet (every event's score
on its area's CD, used by 020), <pop>/area_cd_drift.{png,pdf,svg}, area_tests.csv.
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
m002 = importlib.import_module("002_trial_slopes")
OUT = m51.WITHIN / "area_cd_drift"
MIN_UNITS = 5
CLS = ["WH", "AH", "FA"]
STG = [("learning", "Day 0"), ("expert", "Expert")]


def session(args):
    f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["trial_start"].astype(float)[o]
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return [], []
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    tau = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    rows, trials = [], []
    for g in K.area_group.dropna().unique():
        u = ok & (K.area_group == g).to_numpy()
        if u.sum() < MIN_UNITS:
            continue
        S = m002.trial_scores(X[u].T, lab, seed)
        r = dict(meta, area_group=g, n_units=int(u.sum()), dprime=S["dprime"])
        if np.isfinite(S["md"]).any():
            for c in CLS:
                k = (lab == c) & np.isfinite(S["md"])
                if k.sum() >= 3:
                    r[f"beta_{c}"] = np.polyfit(tau[k], S["md"][k], 1)[0]
                    r[f"mean_{c}"] = S["md"][k].mean()
            if "beta_WH" in r:
                r["drift_WH_SL"] = r["beta_WH"] - r.get("beta_FA", np.nan)
                r["drift_WH_AH"] = r["beta_WH"] - r.get("beta_AH", np.nan)
            k = np.isin(lab, CLS) & np.isfinite(S["md"])         # single-event scores of this area's CD (020 figures)
            trials.append(pd.DataFrame(dict(session_id=meta["session_id"], mouse_id=meta["mouse_id"], cohort=meta["cohort"],
                                            stage=meta["stage"], area_group=g, cls=lab[k], tau=tau[k], t=t[k], cd=S["md"][k])))
        rows.append(r)
    return rows, trials


def compute(n_proc):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            jobs.append((f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0],
                                 stage=Ws.stage.iloc[0]),
                         Ws[["electrode_group", "cluster_id", "quality_label", "area_group"]].drop_duplicates(
                             ["electrode_group", "cluster_id"]), i))
    rows, trials = [], []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for res, tr in pool.imap_unordered(session, jobs):
            rows += res; trials += tr
    return pd.DataFrame(rows), pd.concat(trials, ignore_index=True)


def qualify(A, col):
    return m62.area_qualify(A.rename(columns={"area_group": "region"}), col=col)


def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full_like(p, np.nan)
    if ok.any():
        pv = p[ok]; o = np.argsort(pv); r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        q[np.where(ok)[0][o]] = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    return q


def tests(A, col):
    Q = qualify(A, col)
    rows = []
    for g in sorted(Q["R+"] | Q["R-"]):
        d = A[(A.area_group == g)].dropna(subset=[col])
        r = dict(measure=col, area_group=g, qualifies_Rplus=g in Q["R+"], qualifies_Rminus=g in Q["R-"])
        for c in ("R+", "R-"):
            if g not in Q[c]:
                continue
            for st, _ in STG:
                v = d[(d.cohort == c) & (d.stage == st)][col]
                r[f"n {c} {st}"], r[f"n mice {c} {st}"] = len(v), d[(d.cohort == c) & (d.stage == st)].mouse_id.nunique()
                r[f"mean {c} {st}"], r[f"sem {c} {st}"] = v.mean(), v.sem()
                r[f"p_wilcoxon {c} {st}"] = stats.wilcoxon(v).pvalue if len(v) >= 3 and (v != 0).any() else np.nan
        if g in Q["R+"] and g in Q["R-"]:
            for st, _ in STG:
                x = d[(d.cohort == "R+") & (d.stage == st)][col]; y = d[(d.cohort == "R-") & (d.stage == st)][col]
                r[f"{st} p_MWU"] = stats.mannwhitneyu(x, y).pvalue
                r[f"{st} p_Welch"] = stats.ttest_ind(x, y, equal_var=False).pvalue
                r[f"{st} R+ - R-"], r[f"{st} p_perm"] = m001.perm_interaction(d[d.stage == st].assign(stage="learning"), col,
                                                                             n_perm=5000, kind="learning")
            r["cxs"], r["cxs p_perm"] = m001.perm_interaction(d, col, n_perm=5000, kind="stage")
        rows.append(r)
    T = pd.DataFrame(rows)
    if "learning p_perm" in T:
        T["learning q_BH"] = bh(T["learning p_perm"])
    return T


def figure(plt, A, TT, out, pop):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    cols = [("drift_WH_SL", "Drift of WH relative to SL (β_WH − β_SL)"), ("drift_WH_AH", "Drift of WH relative to AH (β_WH − β_AH)")]
    order_all = [g for g in au.get_area_group_custom_order() if g in set(TT.area_group)]
    fig = plt.figure(figsize=(m62.W_IN, 0.24 * len(order_all) + 1.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.55, 1.3, 0.55], wspace=0.15, left=0.2, right=0.98, top=0.88, bottom=0.1)
    axs = []
    for j, (col, ttl) in enumerate(cols):
        T = TT[TT.measure == col].set_index("area_group")
        ax = fig.add_subplot(gs[0, 2 * j]); axs.append(ax)
        y = np.arange(len(order_all))[::-1]
        for i, g in enumerate(order_all):
            if g not in T.index:
                continue
            r = T.loc[g]
            for k, c in enumerate(("R+", "R-")):
                for s_, (st, _) in enumerate(STG):
                    m_, se_ = r.get(f"mean {c} {st}", np.nan), r.get(f"sem {c} {st}", np.nan)
                    if not np.isfinite(m_):
                        continue
                    yy = y[i] + 0.3 - (2 * k + s_) * 0.2
                    p0 = r.get(f"p_wilcoxon {c} {st}", np.nan)
                    ax.errorbar(m_, yy, xerr=se_, fmt="o", ms=2.6, color=m62.COH[c], lw=0.7, capsize=0,
                                mfc=("white" if st == "learning" else m62.COH[c]), alpha=1 if (np.isfinite(p0) and p0 < 0.05) else 0.55)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_yticks(y, order_all if j == 0 else [], fontsize=5)
        if j == 0:
            for tl, g in zip(ax.get_yticklabels(), order_all):
                tl.set_color(gcol.get(g, "k"))
        ax.set_ylim(-0.7, len(order_all) - 0.3)
        ax.set_xlabel("Change over one session (CD units)"); ax.set_title(ttl, fontsize=5.6)
        # p column: day 0 and expert R+ vs R- (mouse permutation), interaction
        axp = fig.add_subplot(gs[0, 2 * j + 1], sharey=ax); axs.append(axp)
        axp.axis("off")
        axp.text(0.0, len(order_all) - 0.2, "D0   Exp   ×", fontsize=4.6, va="bottom", weight="bold")
        for i, g in enumerate(order_all):
            if g in T.index and np.isfinite(T.loc[g].get("learning p_perm", np.nan)):
                r = T.loc[g]
                f = lambda p: ("<.001" if p < 0.001 else f"{p:.3f}".lstrip("0"))
                axp.text(0.0, y[i], f"{f(r['learning p_perm'])}  {f(r['expert p_perm'])}  {f(r['cxs p_perm'])}", fontsize=4.4,
                         va="center", color="#D62728" if r["learning p_perm"] < 0.05 else "0.2")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=m62.COH[c], mfc=("white" if st == "learning" else m62.COH[c]), ms=3,
                label=f"{c.replace('-', '−')} {sl.lower()}") for c in ("R+", "R-") for st, sl in STG]
    axs[0].legend(handles=h, frameon=False, fontsize=4.5, loc="lower left", ncol=2)
    fig.suptitle(f"Within-session drift on the reward-lick CD per area group ({pop}); mean ± s.e.m. over sessions, faded = n.s. vs 0;\n"
                 "p: R+ vs R− on day 0 (D0) and in experts (Exp), mouse-level permutation; ×: cohort × stage interaction",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=6, weight="bold")
    m62.save(fig, out, "area_cd_drift"); plt.close(fig)


def main(a):
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "area_sessions.csv"
    if a.replot and f.exists():
        A0 = pd.read_csv(f)
    else:
        A0, TR = compute(a.n_proc); A0.to_csv(f, index=False)
        TR.to_parquet(OUT / "area_trial_scores.parquet", index=False)
    for pop in ("learners", "all"):
        A = m61.learner_filter(A0) if pop == "learners" else A0
        out = OUT / pop; out.mkdir(exist_ok=True)
        TT = pd.concat([tests(A, c) for c in ("drift_WH_SL", "drift_WH_AH")], ignore_index=True)
        TT.to_csv(out / "area_tests.csv", index=False)
        figure(plt, A, TT, out, pop)
        cols = ["measure", "area_group", "n R+ learning", "mean R+ learning", "n R- learning", "mean R- learning",
                "learning p_perm", "learning p_MWU", "learning p_Welch", "expert p_perm", "cxs p_perm", "learning q_BH"]
        with pd.option_context("display.width", 250):
            print(pop); print(TT[[c for c in cols if c in TT]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
