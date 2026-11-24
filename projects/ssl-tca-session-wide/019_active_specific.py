"""Active-specific components (user 2026-11-23): components whose whisker weight is high in the active block and drops in
both passive blocks. If a component reflects task engagement / learning, it should drop when the task context is gone
(passive post), not only differ by trial outcome. Exploratory, uncorrected.

Per member component of every recurring cluster (013; stim_long and stim_sensory, whisker day 0, both fits):
- trial weights in SD units of the active whisker trials, 0 = their mean (as 018);
- the active block split in halves by position; whisker misses (no lick, no reward) and hits per half; passive pre /
  post whisker trials; auditory hits and passive auditory trials likewise;
- context steps (>= 3 trials per mean):
    pre step   = mean(first-half active whisker misses) - mean(passive pre whisker)
    post step  = mean(second-half active whisker misses) - mean(passive post whisker)
  misses, so a lick / reward / movement component cannot make the step; a drop at both transitions (pre and post) is a
  context effect, a drift over the session gives steps of opposite sign. Same with hits and with auditory trials
  (auditory hit - passive auditory) to see whether the context effect is whisker-specific;
- session course: fifths of the active block per class plus passive pre / post (figures).
Per cluster (per-mouse values, members of a mouse averaged): median and Wilcoxon p of each step per cohort and pooled;
share of mice with both steps > 0; R+ vs R- Mann-Whitney and Welch. Active-specific: pooled pre and post miss steps both
> 0 with Wilcoxon p < 0.05. The within-session trend contrast d (017) is joined for each cluster.

Run (haas): python 019_active_specific.py [--n-jobs 40] [--reuse]
     -> tables/active_members.parquet, tables/active_clusters.csv,
        figures/aligned_gpu/active_overview.png|pdf, figures/aligned_gpu/active_clusters_<fit>.png|pdf
"""
import argparse
import importlib
import json
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

C12 = importlib.import_module("012_component_table")
A5 = importlib.import_module("005_build_aligned")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
WH, AU = "#f7b519", "#2c2cdb"
ALS = ["stim_long", "stim_sensory"]
FITN = {"_splitnn": "main fit", "_nn": "reference fit"}
NB, MIN3 = 5, 3
STEPS = ["pre_miss", "post_miss", "pre_hit", "post_hit", "pre_au", "post_au"]
COURSE = ["wh_hit", "wh_miss", "au_hit"]


def mean3(x):
    return float(np.mean(x)) if len(x) >= MIN3 else np.nan


def session_rows(sid, rows):
    out = []
    t0 = pd.read_parquet(A5.OUT_DIR / sid / "trials.parquet").set_index("trial_index")
    cache = {}
    for r in rows:
        key = (r["alignment"], r["method"])
        if key not in cache:
            d = np.load(C12.D / sid / f"{r['alignment']}_zscore{r['method']}.npz")
            cache[key] = (d, t0.loc[d["trial_index"]].reset_index())
        d, t = cache[key]
        w = d[f"r{int(r['rank'])}_f0"][:, int(r["comp"]) - 1].astype(float)
        ep, tt = t.epoch.to_numpy(), t.trial_type.to_numpy()
        lk = t.lick_flag.fillna(0).to_numpy().astype(int)
        act = ep == "active"
        aw = act & (tt == "whisker_trial")
        sd = np.std(w[aw])
        if not np.isfinite(sd) or sd <= 0:
            continue
        z = (w - w[aw].mean()) / sd
        pos = np.full(len(t), np.nan)
        pos[act] = np.arange(act.sum()) / max(act.sum() - 1, 1)
        h1, h2 = act & (pos < 0.5), act & (pos >= 0.5)
        W, A = tt == "whisker_trial", tt == "auditory_trial"
        pre, post = ep == "passive_pre", ep == "passive_post"
        res = {k: r[k] for k in ("session_id", "mouse_id", "cohort", "learner", "method", "alignment", "cluster", "comp")}
        res.update(
            pre_miss=mean3(z[h1 & W & (lk == 0)]) - mean3(z[pre & W]),
            post_miss=mean3(z[h2 & W & (lk == 0)]) - mean3(z[post & W]),
            pre_hit=mean3(z[h1 & W & (lk == 1)]) - mean3(z[pre & W]),
            post_hit=mean3(z[h2 & W & (lk == 1)]) - mean3(z[post & W]),
            pre_au=mean3(z[h1 & A & (lk == 1)]) - mean3(z[pre & A]),
            post_au=mean3(z[h2 & A & (lk == 1)]) - mean3(z[post & A]),
            p_pre=mean3(z[pre & W]), p_post=mean3(z[post & W]),
            p_pre_au=mean3(z[pre & A]), p_post_au=mean3(z[post & A]))
        for nm, m in (("wh_hit", W & (lk == 1)), ("wh_miss", W & (lk == 0)), ("au_hit", A & (lk == 1))):
            mm = act & m
            b = np.minimum((pos[mm] * NB).astype(int), NB - 1)
            res[f"f_{nm}"] = json.dumps([float(z[mm][b == i].mean()) if (b == i).sum() >= 2 else None for i in range(NB)])
        out.append(res)
    return out


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), float(stats.wilcoxon(v).pvalue) if len(v) >= 5 and np.any(v != 0) else np.nan, len(v))


def two(a, b):
    a, b = (np.asarray(x, float) for x in (a, b))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def per_mouse(g):
    return g.groupby(["mouse_id", "cohort"], as_index=False)[STEPS + ["p_pre", "p_post"]].mean()


def cluster_stats(am, tc):
    rows = []
    for (meth, al, k), g in am.groupby(["method", "alignment", "cluster"]):
        pm = per_mouse(g)
        row = dict(method=meth, alignment=al, cluster=k, n_mice=pm.mouse_id.nunique())
        for s in STEPS:
            row[f"{s}_median"], row[f"{s}_p"], row[f"{s}_n"] = wil(pm[s])
            for c in ("R+", "R-"):
                row[f"{s}_{c}_median"], row[f"{s}_{c}_p"], _ = wil(pm.loc[pm.cohort == c, s])
            row[f"{s}_mw_p"], row[f"{s}_welch_p"] = two(pm.loc[pm.cohort == "R+", s], pm.loc[pm.cohort == "R-", s])
        ok = pm[["pre_miss", "post_miss"]].notna().all(1)
        row["both_pos_share"] = float(((pm.pre_miss > 0) & (pm.post_miss > 0))[ok].mean()) if ok.any() else np.nan
        rows.append(row)
    cs = pd.DataFrame(rows)
    cs["active_specific"] = ((cs.pre_miss_median > 0) & (cs.pre_miss_p < 0.05) & (cs.post_miss_median > 0)
                             & (cs.post_miss_p < 0.05))
    keep = ["d_R+_median", "d_R+_p", "d_R-_median", "d_R-_p", "d_RpRm_mw_p", "d_RpRm_welch_p", "candidate"]
    cs = cs.merge(tc[["method", "alignment", "cluster"] + keep], on=["method", "alignment", "cluster"], how="left")
    return cs


def overview(cs, cl, out):
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.0), gridspec_kw=dict(hspace=0.45, wspace=0.4))
    for i, meth in enumerate(("_splitnn", "_nn")):
        for j, al in enumerate(ALS):
            ax = axs[i, j]
            g = cs[(cs.method == meth) & (cs.alignment == al)]
            for _, r in g.iterrows():
                col = "k" if r.active_specific else "0.7"
                ax.scatter(r.pre_miss_median, r.post_miss_median, s=4 + r.n_mice / 4, color=col, lw=0, zorder=2)
                if r.candidate is True:
                    ax.scatter(r.pre_miss_median, r.post_miss_median, s=40, facecolors="none", edgecolors="tab:red",
                               lw=0.8, zorder=3)
                if r.active_specific or r.candidate is True:
                    top = cl.loc[(meth, al, r.cluster), "top_group"].replace("Somatosensory-", "Ss-")
                    pk = cl.loc[(meth, al, r.cluster), "peak_ms_median"]
                    ax.annotate(f"{r.cluster}: {top[:12]}, {pk:.0f} ms", (r.pre_miss_median, r.post_miss_median),
                                fontsize=4.2, xytext=(3, 2), textcoords="offset points")
            ax.axhline(0, color="0.6", lw=0.4)
            ax.axvline(0, color="0.6", lw=0.4)
            lim = np.nanmax(np.abs(g[["pre_miss_median", "post_miss_median"]].to_numpy())) * 1.2 if len(g) else 1
            ax.plot([-lim, lim], [-lim, lim], color="0.8", lw=0.4, ls=":")
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Pre step: active (1st half) - passive pre (SD)",
                   ylabel="Post step: active (2nd half) - passive post (SD)",
                   title=f"{al}, {FITN[meth]}: {int(g.active_specific.sum())} / {len(g)} active-specific")
    fig.suptitle("Whisker-miss context steps per recurring cluster (median over mice, both cohorts; size = n mice). "
                 "Black = drop into both passive blocks (p < 0.05 each);\nred ring = trend candidate (017). Upper right "
                 "= higher in the active block than in both passive blocks; drift = upper left / lower right.",
                 fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"active_overview.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)


def sem(x):
    x = np.asarray(x, float)
    return np.nanstd(x, 0) / np.sqrt(np.maximum(np.sum(np.isfinite(x), 0) - 1, 1))


def clusters_page(cs, am, mem, cl, meth, out, nmax=8):
    sel = cs[(cs.method == meth) & (cs.active_specific | (cs.candidate == True))].copy()  # noqa: E712
    sel["score"] = sel[["pre_miss_median", "post_miss_median"]].min(1)
    sel = sel.sort_values(["active_specific", "score"], ascending=False).head(nmax)
    if not len(sel):
        return
    n = len(sel)
    fig, axs = plt.subplots(n, 5, figsize=(7.6, 1.6 * n + 0.6), squeeze=False,
                            gridspec_kw=dict(hspace=1.0, wspace=0.6, width_ratios=[0.9, 1.3, 1.3, 0.8, 0.8]))
    for i, (_, r) in enumerate(sel.iterrows()):
        g = am[(am.method == meth) & (am.alignment == r.alignment) & (am.cluster == r.cluster)]
        m = mem[(mem.method == meth) & (mem.alignment == r.alignment) & (mem.cluster == r.cluster)]
        c = cl.loc[(meth, r.alignment, r.cluster)]
        ax = axs[i, 0]
        tcen = np.array(json.loads(m.t_centres_ms.iloc[0]))
        for coh, h in m.groupby("cohort"):
            ax.plot(tcen, np.array([json.loads(x) for x in h.time_factor]).mean(0), color=COH[coh], lw=0.9)
        ax.axvline(0, color="0.6", lw=0.4)
        nm = g.groupby("cohort").mouse_id.nunique()
        ax.set_title(f"{r.alignment} cl. {r.cluster}{' (trend cand.)' if r.candidate is True else ''}\npeak "
                     f"{c.peak_ms_median:.0f} ms, {c.top_group.replace('Somatosensory-', 'Ss-')[:18]}\nR+ "
                     f"{nm.get('R+', 0)} / R- {nm.get('R-', 0)} mice", fontsize=5)
        ax.set_xlabel("Time (ms)", fontsize=5)
        for j, coh in enumerate(("R+", "R-")):
            ax = axs[i, 1 + j]
            h = g[g.cohort == coh]
            xb = np.arange(NB)
            for nm_, col, ls, mfc in (("wh_hit", WH, "-", WH), ("wh_miss", WH, "--", "white"), ("au_hit", AU, "-", AU)):
                Y = np.array([[np.nan if v is None else v for v in json.loads(x)] for x in h[f"f_{nm_}"]], float)
                if not len(Y):
                    continue
                Y = pd.DataFrame(Y).groupby(h.mouse_id.to_numpy()).mean().to_numpy()
                ax.errorbar(xb, np.nanmean(Y, 0), sem(Y), color=col, ls=ls, marker="o", ms=2, mfc=mfc, lw=0.7,
                            elinewidth=0.4)
            pmh = h.groupby("mouse_id")[["p_pre", "p_post"]].mean()
            for x, cc in ((-1, "p_pre"), (NB, "p_post")):
                v = pmh[cc].to_numpy(float)
                ax.errorbar(x, np.nanmean(v), sem(v), color=WH, marker="D", ms=2.5, mfc="white", elinewidth=0.4)
            ax.axvspan(-1.4, -0.6, color="0.93", lw=0, zorder=-1)
            ax.axvspan(NB - 0.4, NB + 0.4, color="0.93", lw=0, zorder=-1)
            ax.set_xticks(range(-1, NB + 1), ["pre", "1", "2", "3", "4", "5", "post"], fontsize=4.5)
            ax.set_title(f"{coh}: pre step {r[f'pre_miss_{coh}_median']:+.2f} (p {r[f'pre_miss_{coh}_p']:.3f})\n"
                         f"post step {r[f'post_miss_{coh}_median']:+.2f} (p {r[f'post_miss_{coh}_p']:.3f})",
                         fontsize=5, color=COH[coh])
            if j == 0:
                ax.set_ylabel("Weight (SD)", fontsize=5)
        pm = per_mouse(g)
        for jj, (col, lab, mw, we) in enumerate((("post_miss", "Post step (misses)", r.post_miss_mw_p, r.post_miss_welch_p),)):
            ax = axs[i, 3]
            for x, coh in enumerate(("R+", "R-")):
                v = pm.loc[pm.cohort == coh, col].dropna().to_numpy()
                ax.scatter(x + (np.random.default_rng(x).random(len(v)) - 0.5) * 0.3, v, s=2.5, color=COH[coh], lw=0)
                if len(v):
                    ax.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, color="k", lw=0.9)
            ax.axhline(0, color="0.6", lw=0.4)
            ax.set_xticks([0, 1], ["R+", "R-"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_title(f"{lab}\n{int(100 * r.both_pos_share)} % mice both > 0\nMW {mw:.3f}, W {we:.3f}", fontsize=5)
        ax = axs[i, 4]
        tm_ = mem[(mem.method == meth) & (mem.alignment == r.alignment) & (mem.cluster == r.cluster)]
        ax.axis("off")
        ax.text(0, 0.95, (f"trend d (017)\nR+ {r['d_R+_median']:+.2f} (p {r['d_R+_p']:.3f})\nR- {r['d_R-_median']:+.2f} "
                          f"(p {r['d_R-_p']:.3f})\nR+ vs R-: MW {r.d_RpRm_mw_p:.3f}\nWelch {r.d_RpRm_welch_p:.3f}\n\n"
                          f"auditory post step\n{r.post_au_median:+.2f} (p {r.post_au_p:.3f})\nhit post step\n"
                          f"{r.post_hit_median:+.2f} (p {r.post_hit_p:.3f})"), fontsize=4.8, va="top",
                transform=ax.transAxes)
        del tm_
    fig.subplots_adjust(top=1 - 0.7 / (1.6 * n + 0.6))
    fig.suptitle(f"Active-specific clusters, {FITN[meth]} (whisker day 0): session course in SD units of active whisker "
                 "trials (whisker hit filled, miss open, auditory hit blue; passive whisker diamonds), mean ± SEM over "
                 "mice;\npre / post steps of whisker misses (median, Wilcoxon p). Exploratory, uncorrected.",
                 fontsize=6, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"active_clusters{meth}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members.parquet")
    mem = mem[(mem.stage == "learning") & mem.alignment.isin(ALS)]
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    tc = pd.read_csv(L.OUT / "tables" / "trend_clusters.csv")
    f = L.OUT / "tables" / "active_members.parquet"
    if a.reuse and f.exists():
        am = pd.read_parquet(f)
    else:
        by = {}
        for r in mem.drop(columns=["time_factor", "area_profile", "t_centres_ms"]).to_dict("records"):
            by.setdefault(r["session_id"], []).append(r)
        res = Parallel(n_jobs=a.n_jobs)(delayed(session_rows)(s, rs) for s, rs in by.items())
        am = pd.DataFrame([x for xs in res for x in xs])
        am.to_parquet(f, index=False)
    cs = cluster_stats(am, tc)
    cs.to_csv(L.OUT / "tables" / "active_clusters.csv", index=False)
    out = L.OUT / "figures" / "aligned_gpu"
    overview(cs, cl, out)
    for meth in ("_splitnn", "_nn"):
        clusters_page(cs, am, mem, cl, meth, out)
    pd.set_option("display.width", 250)
    cols = ["method", "alignment", "cluster", "n_mice", "pre_miss_median", "pre_miss_p", "post_miss_median",
            "post_miss_p", "both_pos_share", "post_au_median", "post_miss_mw_p", "d_R+_median", "d_RpRm_mw_p",
            "candidate"]
    print(cs[cs.active_specific | (cs.candidate == True)][cols].round(3).to_string(index=False))  # noqa: E712
    print(cs.groupby(["method", "alignment"]).active_specific.agg(["sum", "size"]))
    for meth in ("_splitnn", "_nn"):   # descriptive: where do the candidates sit
        print(meth, cs[(cs.method == meth)][["pre_miss_median", "post_miss_median"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
