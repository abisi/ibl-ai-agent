"""019 -- Whisker-hit quintile drift on the reward-lick CD per area, beyond a linear-shift null, with an area-level mixed model (user 2026-10-16:
"ok do it" -- shift null and area-level model for the 018 quintile result).
Per session and area group (units as 017 / 018; baseline-corrected rates): CD = unit-norm AH - SL axis of 5-fold CV over the AH + SL events
(units z-scored on the training folds); every event projected per neuron (/ sqrt N): AH and SL held-out, WH averaged over folds. Whisker-hit
quintiles as 018 (equal numbers of whisker hits, SL averaged over the same time windows); quantity: OLS slope over quintiles of WH - SL
(per neuron per quintile), and of WH alone.
Null: the AH / SL labels of the time-ordered AH + SL events shifted by k (10-50 % of them, both directions, non-wrapping; N_SHIFT shifts keeping
>= 8 events per class); the axis rebuilt the same way and every event (real classes) projected on it; the same quintile slopes. Excess = real -
mean null; percentile.
Statistics: per area and cohort, Wilcoxon of the excess vs 0 per stage, Mann-Whitney day 0 vs expert. Area-level model per cohort and measure
(OLS with standard errors clustered by mouse): excess ~ stage + area (pooled day 0 vs expert effect across areas) and excess ~ stage * area
(Wald test of the stage x area terms: does the stage effect differ between areas?).
Output: tables/019_sessions.csv, 019_stats.csv, 019_model.csv; figures/019_quint_shift_null.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C

warnings.filterwarnings("ignore")
m002 = importlib.import_module("002_trial_slopes")
st26 = importlib.import_module("026_roc_rates_all_sessions")
MIN_UNITS, MIN_EV, N_SHIFT, MIN_SESS = 10, 8, 50, 3


def project(X, y, seed):
    """X events x units; y: 1 AH, 0 SL, -1 not used for the axis. Held-out projection for y >= 0, fold average for the rest (per neuron)."""
    from sklearn.model_selection import StratifiedKFold
    ia = np.where(y >= 0)[0]; yy = y[ia]; rest = np.where(y < 0)[0]
    p = np.full(len(y), np.nan); pr = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(ia, yy):
        A = X[ia[tr]]; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
        z = lambda q: (q - mu) / sd
        u = z(A[yy[tr] == 1]).mean(0) - z(A[yy[tr] == 0]).mean(0); u = u / max(np.linalg.norm(u), 1e-12)
        p[ia[te]] = z(X[ia[te]]) @ u; pr.append(z(X[rest]) @ u)
    if len(rest):
        p[rest] = np.mean(pr, 0)
    return p / np.sqrt(X.shape[1])


def quint_slopes(p, cls, t):
    w = np.where(cls == "WH")[0]
    if len(w) < 5:
        return np.nan, np.nan
    idx = np.array_split(w, 5); tw = t; out_wh, out_rel = [], []
    for k, i in enumerate(idx):
        lo = -np.inf if k == 0 else 0.5 * (t[idx[k - 1][-1]] + t[i[0]]); hi = np.inf if k == 4 else 0.5 * (t[i[-1]] + t[idx[k + 1][0]])
        sl = p[(cls == "SL") & (tw >= lo) & (tw < hi)]
        out_wh.append(p[i].mean()); out_rel.append(p[i].mean() - sl.mean() if len(sl) else np.nan)
    x = np.arange(1, 6); r = np.array(out_rel); ok = np.isfinite(r)
    return np.polyfit(x, out_wh, 1)[0], (np.polyfit(x[ok], r[ok], 1)[0] if ok.sum() >= 3 else np.nan)


def session(args):
    f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o].astype(str), z["trial_start"].astype(float)[o]
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < MIN_EV:
        return []
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= C.m51.MIN_FR)
    cls = np.where(lab == "FA", "SL", lab)
    y = np.where(cls == "AH", 1, np.where(cls == "SL", 0, -1))
    ref = np.where(y >= 0)[0]; n = len(ref)
    rng = np.random.default_rng(seed)
    pairs = [(k, s) for k in range(max(1, int(0.1 * n)), int(0.5 * n) + 1) for s in (1, -1)]
    rng.shuffle(pairs)
    rows = []
    for area, m in [("whole_brain", ok)] + [(a, ok & (K.area_group == a).to_numpy()) for a in K.area_group.dropna().unique()]:
        if m.sum() < MIN_UNITS:
            continue
        Xa = X[m].T
        b_wh, b_rel = quint_slopes(project(Xa, y, seed), cls, t)
        nl = []
        for k, s in pairs:
            ys = np.full(len(y), -1)
            lab_ref = y[ref]
            if s > 0:
                ys[ref[:n - k]] = lab_ref[k:]
            else:
                ys[ref[k:]] = lab_ref[:n - k]
            if min((ys == 1).sum(), (ys == 0).sum()) < MIN_EV:
                continue
            nl.append(quint_slopes(project(Xa, ys, seed), cls, t))
            if len(nl) >= N_SHIFT:
                break
        nl = np.array(nl)
        if len(nl) < 10:
            continue
        rows.append(dict(meta, area=area, n_units=int(m.sum()), n_null=len(nl), wh=b_wh, rel=b_rel, wh_null=np.nanmean(nl[:, 0]), rel_null=np.nanmean(nl[:, 1]),
                         wh_ex=b_wh - np.nanmean(nl[:, 0]), rel_ex=b_rel - np.nanmean(nl[:, 1]), rel_pct=float(np.nanmean(nl[:, 1] < b_rel))))
    return rows


def compute(n_proc):
    W = pd.read_parquet(C.m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{C.m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            meta = dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0])
            jobs.append((f, meta, Ws[["electrode_group", "cluster_id", "quality_label", "area_group"]], zlib.crc32(r.session_id.encode())))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, jobs)
    S = C.population(pd.DataFrame([x for r in res for x in r]))
    S.to_csv(C.TAB / "019_sessions.csv", index=False)
    return S


def model(S):
    import statsmodels.formula.api as smf
    rows = []
    for c in ("R+", "R-"):
        for m in ("rel_ex", "wh_ex"):
            n = S[S.cohort == c].groupby(["area", "stage"]).size().unstack(fill_value=0)
            areas = [a for a in n.index if a != "whole_brain" and n.loc[a].get("learning", 0) >= MIN_SESS and n.loc[a].get("expert", 0) >= MIN_SESS]
            D = S[(S.cohort == c) & S.area.isin(areas)].dropna(subset=[m]).copy(); D["E"] = (D.stage == "expert").astype(int)
            try:
                D["y"] = D[m] * 100                               # x 100: per-neuron slopes are ~0.01
                cl = dict(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(D.mouse_id)[0]))
                r1 = smf.ols("y ~ E + area", D).fit(**cl)
                r2 = smf.ols("y ~ E * area", D).fit(**cl)
                names = [k for k in r2.params.index if k.startswith("E:")]
                wt = r2.wald_test(" , ".join(f"{k} = 0" for k in names), scalar=True) if names else None
                rows.append(dict(cohort=c, measure=m, n_areas=len(areas), n_obs=len(D), stage_effect=r1.params["E"] / 100, se=r1.bse["E"] / 100,
                                 p_stage=r1.pvalues["E"], p_stage_x_area=float(wt.pvalue) if wt is not None else np.nan))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(cohort=c, measure=m, error=repr(e)[:120]))
    return pd.DataFrame(rows)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    S = pd.read_csv(C.TAB / "019_sessions.csv") if a.replot else compute(a.n_proc)
    St = []
    w = lambda v: stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan
    for (area, c), g in S.groupby(["area", "cohort"]):
        for m in ("rel_ex", "wh_ex", "rel", "wh"):
            d0, e = g[g.stage == "learning"][m].dropna(), g[g.stage == "expert"][m].dropna()
            St.append(dict(area=area, cohort=c, measure=m, n_d0=len(d0), n_e=len(e), mean_d0=d0.mean(), mean_e=e.mean(), p_d0=w(d0), p_e=w(e),
                           p_stage=stats.mannwhitneyu(d0, e).pvalue if len(d0) >= 3 and len(e) >= 3 else np.nan))
    St = pd.DataFrame(St); St.to_csv(C.TAB / "019_stats.csv", index=False)
    M = model(S); M.to_csv(C.TAB / "019_model.csv", index=False)
    # figure: per area, excess of the WH - SL quintile slope, day 0 vs expert, per cohort (forest)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    gcol = au.get_custom_area_groups_colors()
    nn = S.groupby(["area", "cohort", "stage"]).size().unstack(["cohort", "stage"]).fillna(0)
    keep = [a_ for a_ in nn.index if any(nn.loc[a_].get((c, "learning"), 0) >= MIN_SESS and nn.loc[a_].get((c, "expert"), 0) >= MIN_SESS for c in ("R+", "R-"))]
    areas = ["whole_brain"] + [g for g in au.get_area_group_custom_order() if g in keep]
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 0.24 * len(areas) + 1.2), gridspec_kw=dict(wspace=0.35))
    for ax, (m, c, lab) in zip(axs, (("rel_ex", "R+", "R+: WH − SL"), ("rel_ex", "R-", "R−: WH − SL"), ("wh_ex", "R+", "R+: WH alone"), ("wh_ex", "R-", "R−: WH alone"))):
        y = np.arange(len(areas))[::-1]
        for i, a_ in enumerate(areas):
            r = St[(St.area == a_) & (St.cohort == c) & (St.measure == m)]
            if not len(r) or r.n_d0.iloc[0] < MIN_SESS or r.n_e.iloc[0] < MIN_SESS:
                continue
            r = r.iloc[0]; g = S[(S.area == a_) & (S.cohort == c)]
            for s_, dy, mfc in (("learning", 0.15, "white"), ("expert", -0.15, None)):
                v = g[g.stage == s_][m].dropna()
                ax.errorbar(v.mean(), y[i] + dy, xerr=v.sem(), fmt="o", ms=2.6, color=C.COH[c], mfc=mfc or C.COH[c], capsize=0, lw=0.8)
            if r.p_stage < 0.05:
                ax.text(1.0, y[i], "*", transform=ax.get_yaxis_transform(), fontsize=7, va="center", ha="right")
        ax.axvline(0, color="0.6", lw=0.5); ax.set_yticks(y, [a_.replace("whole_brain", "Whole brain").replace("Somatosensory-", "Ss-").replace(" areas", "") for a_ in areas] if ax is axs[0] else [])
        if ax is axs[0]:
            for tl, a_ in zip(ax.get_yticklabels(), areas):
                tl.set_color(gcol.get(a_, "k"))
        mm = M[(M.cohort == c) & (M.measure == m)]
        tt = f"pooled D0→E {C.fp(mm.p_stage.iloc[0])}; stage × area {C.fp(mm.p_stage_x_area.iloc[0])}" if len(mm) and "p_stage" in mm and np.isfinite(mm.p_stage.iloc[0]) else ""
        ax.set_title(f"{lab}\n{tt}", fontsize=5.6, color=C.COH[c]); ax.set_xlabel("quintile slope, real − shifted\n(per neuron per quintile)")
    fig.suptitle("Whisker-hit quintile drift on the reward-lick CD beyond the linear-shift null, per area (open day 0, filled expert; mean ± s.e.m.; "
                 "* day 0 vs expert MWU p < 0.05; titles: area-level OLS, SE clustered by mouse)", fontsize=6, x=0.01, y=1.02, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"019_quint_shift_null.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(St[St.measure == "rel_ex"].round(4).to_string(index=False)); print(M.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=24); ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
