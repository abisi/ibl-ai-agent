"""Passive post vs pre (user 2026-11-23: more signs of learning; compare the passive representations before and after
the task). A learned change of the whisker representation should be present after the task, without task context:
passive post differs from passive pre on whisker trials, more in R+ than in R-, and more than on auditory trials (same
blocks, same elapsed time, auditory already learned). Exploratory, uncorrected; per mouse; MW + Welch for R+ vs R-.

Three levels, whisker day 0 (expert sessions computed alongside, as the reference where nothing is learned any more):
A  TCA components (019 tables): passive post - pre whisker weight minus the same for auditory (SD units of the active
   whisker trials), per recurring cluster.
B  Units (no TCA): evoked response per unit and passive trial in three windows, z-scored per unit (rate - same-trial
   baseline -1..-0.01 s, / SD over trials and bins of the tensor): early 10-45 ms (stim_sensory, 10-ms bins), middle
   50-250 ms and late 250-1000 ms (stim_long, 50-ms bins). Delta = mean(post) - mean(pre), whisker and auditory;
   contrast = Delta_W - Delta_A; units = all kept units or passive-responsive units (evoked > 0, Wilcoxon p < 0.01 over
   the passive trials of that modality, pre and post pooled). Mouse value = mean over units of an area group.
C  Population (no TCA), per area group with >= 8 units and window: cross-validated squared distance between the post and
   pre mean response vectors (units x bins; two halves of alternating trials, inner product of the half-differences, /
   units), whisker and auditory, and the gain index g = <Delta, m_pre> / <m_pre, m_pre> (cross-validated halves), i.e.
   the change along the pre response (gain) vs. the rest (rotation).
Movement: all passive trials are used, jaw movements kept (user 2026-11-23); the jaw-movement rate per block is reported.

Run (haas): python 020_passive_pre_post.py [--n-jobs 40] [--reuse]
     -> tables/prepost_units.parquet, tables/prepost_pop.parquet, tables/prepost_clusters.csv,
        figures/aligned_gpu/prepost_{tca,units,population}.png|pdf
"""
import argparse
import importlib
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

A5 = importlib.import_module("005_build_aligned")
P4 = importlib.import_module("004_pilot_figures")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
WIN = {"early": ("stim_sensory", 10.0, 45.0), "middle": ("stim_long", 50.0, 250.0), "late": ("stim_long", 250.0, 1000.0)}
MIN_T, MIN_POP = 5, 8
FITN = {"_splitnn": "main fit", "_nn": "reference fit"}


def prep(sid, al):
    d = np.load(A5.OUT_DIR / sid / f"{al}.npz")
    R, base = d["rate"], d["baseline"]
    keep = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= 0.5
    D = R - base[:, :, None]
    D = np.nan_to_num(D / (np.nanstd(D, (0, 2), keepdims=True) + 1e-9))
    return d, D, keep


def cvdist(a, b):
    """a, b: trials x features of the pre / post block; unbiased squared distance of the means (alternate halves)"""
    if len(a) < 2 * 3 or len(b) < 2 * 3:
        return np.nan, np.nan
    d1 = b[0::2].mean(0) - a[0::2].mean(0)
    d2 = b[1::2].mean(0) - a[1::2].mean(0)
    m1, m2 = a[0::2].mean(0), a[1::2].mean(0)
    dist = float(d1 @ d2) / a.shape[1]
    den = float(m1 @ m2)
    g = float(0.5 * (d1 @ m2 + d2 @ m1) / den) if den > 0 else np.nan
    return dist, g


def session(sid, meta):
    t_all = pd.read_parquet(A5.OUT_DIR / sid / "trials.parquet").set_index("trial_index")
    units = pd.read_parquet(A5.OUT_DIR / sid / "units.parquet")
    _, _, grp = P4.area_colours(units.area_acronym_custom.str.strip().tolist())
    grp = np.asarray(grp)
    urows, prows = [], []
    for wn, (al, t0, t1) in WIN.items():
        d, D, keep = prep(sid, al)
        t = t_all.loc[d["trial_index"]]
        tc = d["t_centres"] * 1e3
        b = (tc > t0) & (tc < t1)
        ev = D[:, :, b]                                   # trials x units x bins
        e = ev.mean(2)                                    # trials x units
        ep, ty = t.epoch.to_numpy(), t.trial_type.to_numpy()
        variants = [("all", np.ones(len(t), bool))]   # all passive trials, jaw movements kept (user 2026-11-23)
        for vn, vm in variants:
            M = {(m, blk): np.flatnonzero((ty == f"{m}_trial") & (ep == f"passive_{blk}") & vm)
                 for m in ("whisker", "auditory") for blk in ("pre", "post")}
            if min(len(v) for v in M.values()) < MIN_T:
                continue
            resp = {}
            for m in ("whisker", "auditory"):
                x = e[np.r_[M[(m, "pre")], M[(m, "post")]]]
                with np.errstate(all="ignore"):
                    p = np.array([stats.wilcoxon(x[:, j]).pvalue if np.any(x[:, j] != 0) else 1.0
                                  for j in range(x.shape[1])])
                resp[m] = (x.mean(0) > 0) & (p < 0.01)
            dW = e[M[("whisker", "post")]].mean(0) - e[M[("whisker", "pre")]].mean(0)
            dA = e[M[("auditory", "post")]].mean(0) - e[M[("auditory", "pre")]].mean(0)
            for j in np.flatnonzero(keep):
                urows.append(dict(session_id=sid, **meta, window=wn, variant=vn,
                                  electrode_group=units.electrode_group.iloc[j], cluster_id=int(units.cluster_id.iloc[j]),
                                  area=units.area_acronym_custom.iloc[j].strip(), area_group=grp[j],
                                  resp_w=bool(resp["whisker"][j]), resp_a=bool(resp["auditory"][j]),
                                  pre_w=float(e[M[("whisker", "pre")], j].mean()), dW=float(dW[j]), dA=float(dA[j])))
            for g in sorted(set(grp[keep])) + ["all"]:
                sel = keep & ((grp == g) if g != "all" else True)
                if sel.sum() < MIN_POP:
                    continue
                F = ev[:, sel].reshape(len(t), -1)
                r = dict(session_id=sid, **meta, window=wn, variant=vn, area_group=g, n_units=int(sel.sum()))
                for m, k in (("whisker", "W"), ("auditory", "A")):
                    r[f"dist_{k}"], r[f"gain_{k}"] = cvdist(F[M[(m, "pre")]], F[M[(m, "post")]])
                prows.append(r)
    mv = []
    t = t_all
    for blk in ("pre", "post"):
        for m in ("whisker", "auditory"):
            g = t[(t.epoch == f"passive_{blk}") & (t.trial_type == f"{m}_trial")]
            o = g.jaw_dlc_onset.to_numpy(float)
            mv.append(dict(session_id=sid, **meta, block=blk, modality=m, n=len(g),
                           move=float(((o >= 0) & (o <= 1.0)).mean()) if len(g) else np.nan))
    return pd.DataFrame(urows), pd.DataFrame(prows), pd.DataFrame(mv)


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(stats.wilcoxon(v).pvalue) if len(v) >= 5 and np.any(v != 0) else np.nan


def two(a, b):
    a, b = (np.asarray(x, float) for x in (a, b))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.std(x) / np.sqrt(max(len(x) - 1, 1)) if len(x) else np.nan


def tca_level(out):
    am = pd.read_parquet(L.OUT / "tables" / "active_members.parquet")
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    am = am.assign(dW=am.p_post - am.p_pre, dA=am.p_post_au - am.p_pre_au)
    am["c"] = am.dW - am.dA
    rows = []
    for (meth, al, k), g in am.groupby(["method", "alignment", "cluster"]):
        pm = g.groupby(["mouse_id", "cohort"], as_index=False)[["dW", "dA", "c"]].mean()
        r = dict(method=meth, alignment=al, cluster=k, n_mice=pm.mouse_id.nunique(),
                 top_group=cl.loc[(meth, al, k), "top_group"], peak_ms=cl.loc[(meth, al, k), "peak_ms_median"],
                 kind=cl.loc[(meth, al, k), "kind"])
        for v in ("dW", "dA", "c"):
            for c in ("R+", "R-"):
                x = pm.loc[pm.cohort == c, v]
                r[f"{v}_{c}_median"], r[f"{v}_{c}_p"] = float(x.median()), wil(x)
            r[f"{v}_mw_p"], r[f"{v}_welch_p"] = two(pm.loc[pm.cohort == "R+", v], pm.loc[pm.cohort == "R-", v])
        rows.append(r)
    cs = pd.DataFrame(rows)
    cs["hit"] = (cs.c_mw_p < 0.05) & ((cs["c_R+_p"] < 0.05) | (cs["c_R-_p"] < 0.05))
    cs.to_csv(L.OUT / "tables" / "prepost_clusters.csv", index=False)
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.0), gridspec_kw=dict(hspace=0.45, wspace=0.4))
    for i, meth in enumerate(("_splitnn", "_nn")):
        for j, al in enumerate(("stim_long", "stim_sensory")):
            ax = axs[i, j]
            g = cs[(cs.method == meth) & (cs.alignment == al)]
            for _, r in g.iterrows():
                col = "k" if r.hit else ("0.45" if r.c_mw_p < 0.05 else "0.75")
                ax.scatter(r["c_R+_median"], r["c_R-_median"], s=4 + r.n_mice / 4, color=col, lw=0)
                if r.c_mw_p < 0.05:
                    ax.annotate(f"{r.cluster}: {r.top_group.replace('Somatosensory-', 'Ss-')[:12]}, {r.peak_ms:.0f} ms"
                                f" (MW {r.c_mw_p:.3f}, W {r.c_welch_p:.3f})", (r["c_R+_median"], r["c_R-_median"]),
                                fontsize=4, xytext=(3, 2), textcoords="offset points")
            lim = np.nanmax(np.abs(g[["c_R+_median", "c_R-_median"]].to_numpy())) * 1.2 if len(g) else 1
            ax.plot([-lim, lim], [-lim, lim], color="0.8", lw=0.4, ls=":")
            ax.axhline(0, color="0.6", lw=0.4)
            ax.axvline(0, color="0.6", lw=0.4)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="R+ median (post-pre whisker) - (post-pre auditory)",
                   ylabel="R- median", title=f"{al}, {FITN[meth]}")
    fig.suptitle("TCA clusters, passive post - pre (whisker minus auditory; SD units of active whisker trials), "
                 "whisker day 0. Grey = R+ vs R- MW p < 0.05; black = also != 0 within a cohort. Uncorrected.",
                 fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"prepost_tca.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)
    return cs


def units_page(U, out, order, stage="learning"):
    u = U[(U.stage == stage) & (U.variant == "all")].assign(c=lambda x: x.dW - x.dA)
    fig, axs = plt.subplots(3, 2, figsize=(7.6, 7.6), gridspec_kw=dict(hspace=1.1, wspace=0.3))
    rows = []
    for i, wn in enumerate(WIN):
        for j, (sel, lab) in enumerate(((u.resp_w, "whisker-responsive units"), (np.ones(len(u), bool), "all units"))):
            ax = axs[i, j]
            x = u[(u.window == wn) & sel]
            s = x.groupby(["mouse_id", "cohort", "area_group"])[["dW", "dA", "c"]].mean().reset_index()
            sa = x.groupby(["mouse_id", "cohort"])[["dW", "dA", "c"]].mean().reset_index().assign(area_group="all")
            s = pd.concat([sa, s])
            gl = ["all"] + [g for g in order if s[s.area_group == g].groupby("cohort").mouse_id.nunique()
                            .reindex(["R+", "R-"]).fillna(0).min() >= 5]
            for k, g in enumerate(gl):
                for q, coh in enumerate(("R+", "R-")):
                    v = s[(s.area_group == g) & (s.cohort == coh)].c.to_numpy()
                    ax.errorbar(k + (q - 0.5) * 0.3, np.mean(v), sem(v), color=COH[coh], marker="o", ms=2.3,
                                elinewidth=0.6)
                mw, we = two(s[(s.area_group == g) & (s.cohort == "R+")].c, s[(s.area_group == g) & (s.cohort == "R-")].c)
                ax.text(k, 0.99, f"{mw:.2f}\n{we:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=3.8, color="k" if mw < 0.05 else "0.6")
                for coh in ("R+", "R-"):
                    h = s[(s.area_group == g) & (s.cohort == coh)]
                    rows.append(dict(stage=stage, window=wn, units=lab, area_group=g, cohort=coh, n=len(h),
                                     dW=h.dW.median(), dA=h.dA.median(), c=h.c.median(), c_p=wil(h.c), mw_p=mw, welch_p=we))
            ax.axhline(0, color="0.5", lw=0.4)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.3 * (hi - lo))
            ax.set_xticks(range(len(gl)), [g.replace("Somatosensory-", "Ss-") for g in gl], rotation=45, ha="right",
                          fontsize=4.5)
            ax.set_ylabel("(post - pre) W - A, z")
            ax.set_title(f"{wn} {WIN[wn][1]:.0f}-{WIN[wn][2]:.0f} ms, {lab}", fontsize=6)
    fig.suptitle(f"Units, passive post - pre, whisker minus auditory ({'whisker day 0' if stage == 'learning' else 'expert'}"
                 "): mean ± SEM over mice of the mouse means; top: R+ vs R- MW / Welch p. Uncorrected.", fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"prepost_units_{stage}.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def pop_page(Pp, out, order):
    fig, axs = plt.subplots(3, 3, figsize=(7.6, 7.6), gridspec_kw=dict(hspace=1.1, wspace=0.45))
    rows = []
    x0 = Pp[(Pp.variant == "all")].assign(dd=lambda x: x.dist_W - x.dist_A)
    for i, wn in enumerate(WIN):
        x = x0[x0.window == wn]
        gl = ["all"] + [g for g in order if x[(x.area_group == g) & (x.stage == "learning")].groupby("cohort")
                        .mouse_id.nunique().reindex(["R+", "R-"]).fillna(0).min() >= 5]
        for j, (col, lab) in enumerate((("dd", "distance W - A"), ("gain_W", "gain whisker"), ("gain_A", "gain auditory"))):
            ax = axs[i, j]
            for k, g in enumerate(gl):
                h = x[(x.area_group == g) & (x.stage == "learning")].groupby(["mouse_id", "cohort"])[col].mean().reset_index()
                for q, coh in enumerate(("R+", "R-")):
                    v = h[h.cohort == coh][col].to_numpy()
                    ax.errorbar(k + (q - 0.5) * 0.3, np.nanmean(v), sem(v), color=COH[coh], marker="o", ms=2.3,
                                elinewidth=0.6)
                he = x[(x.area_group == g) & (x.stage == "expert")].groupby("cohort")[col].mean()
                for q, coh in enumerate(("R+", "R-")):
                    if coh in he:
                        ax.plot(k + (q - 0.5) * 0.3, he[coh], marker="x", ms=2.5, color=COH[coh], alpha=0.6, lw=0)
                mw, we = two(h[h.cohort == "R+"][col], h[h.cohort == "R-"][col])
                ax.text(k, 0.99, f"{mw:.2f}\n{we:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=3.8, color="k" if mw < 0.05 else "0.6")
                for coh in ("R+", "R-"):
                    v = h[h.cohort == coh][col]
                    rows.append(dict(window=wn, measure=col, area_group=g, cohort=coh, n=int(v.notna().sum()),
                                     median=v.median(), p=wil(v), mw_p=mw, welch_p=we))
            ax.axhline(0, color="0.5", lw=0.4)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.3 * (hi - lo))
            ax.set_xticks(range(len(gl)), [g.replace("Somatosensory-", "Ss-") for g in gl], rotation=45, ha="right",
                          fontsize=4.5)
            ax.set_title(f"{wn}: {lab}", fontsize=6)
    fig.suptitle("Population (no TCA), passive post vs pre per area group: cross-validated squared distance of the mean "
                 "responses (whisker minus auditory) and gain index (change along the pre response; 0 = none, < 0 = "
                 "weaker).\nDots: day 0 mean ± SEM over mice; x: expert cohort mean. Top: R+ vs R- MW / Welch p (day 0). "
                 "Uncorrected.", fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"prepost_population.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    from ephys_utilities.allen_utils import allen_utils as AU_
    order = AU_.get_area_group_custom_order()
    tdir, out = L.OUT / "tables", L.OUT / "figures" / "aligned_gpu"
    cs = tca_level(out)
    pd.set_option("display.width", 250)
    print(cs[cs.c_mw_p < 0.05][["method", "alignment", "cluster", "n_mice", "top_group", "peak_ms", "kind",
                                "c_R+_median", "c_R+_p", "c_R-_median", "c_R-_p", "c_mw_p", "c_welch_p"]].round(3)
          .to_string(index=False))
    print(cs.groupby(["method", "alignment"]).agg(n=("cluster", "size"), mw05=("c_mw_p", lambda x: int((x < 0.05).sum())),
                                                  hit=("hit", "sum")))
    fu, fp, fm = tdir / "prepost_units.parquet", tdir / "prepost_pop.parquet", tdir / "prepost_movement.parquet"
    if a.reuse and fu.exists():
        U, Pp, Mv = pd.read_parquet(fu), pd.read_parquet(fp), pd.read_parquet(fm)
    else:
        ses = pd.read_csv(tdir / "sessions.csv")
        jobs = [delayed(session)(r.session_id, dict(mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, day=int(r.day)))
                for r in ses.itertuples()]
        res = Parallel(n_jobs=a.n_jobs, verbose=2)(jobs)
        U = pd.concat([x[0] for x in res], ignore_index=True)
        Pp = pd.concat([x[1] for x in res], ignore_index=True)
        Mv = pd.concat([x[2] for x in res], ignore_index=True)
        U.to_parquet(fu, index=False)
        Pp.to_parquet(fp, index=False)
        Mv.to_parquet(fm, index=False)
    su = pd.concat([units_page(U, out, order, st) for st in ("learning", "expert")])
    sp = pop_page(Pp, out, order)
    su.to_csv(tdir / "prepost_units_stats.csv", index=False)
    sp.to_csv(tdir / "prepost_pop_stats.csv", index=False)
    print(su[(su.stage == "learning") & (su.mw_p < 0.05)].round(3).to_string(index=False))
    print(sp[sp.mw_p < 0.05].round(3).to_string(index=False))
    mv = Mv[Mv.stage == "learning"].pivot_table(index=["session_id", "cohort", "modality"], columns="block",
                                                 values="move").reset_index()
    mv["d"] = mv.post - mv.pre
    print("movement post-pre", mv.groupby(["cohort", "modality"]).d.median().round(3).to_dict())
    for m in ("whisker", "auditory"):
        x = mv[mv.modality == m]
        print(m, two(x[x.cohort == "R+"].d, x[x.cohort == "R-"].d))


if __name__ == "__main__":
    main()
