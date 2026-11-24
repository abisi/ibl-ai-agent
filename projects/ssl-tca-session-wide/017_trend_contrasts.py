"""Learning as within-outcome trends, controlled by contrasts (user 2026-11-22: split by outcome rather than regress it
out; compare whisker hits with auditory hits; R- as the contrasting cohort). Replaces the outcome-residualised partial
correlation of 012 (learn_ctrl) for the candidate selection. Exploratory, uncorrected.

Per member component of every recurring cluster (013; stimulus alignments, whisker day 0, both fits):
- trial weights on active trials, divided by their mean over active trials (scale comparable across mice);
- position of each active trial in the active block, 0 (first) .. 1 (last);
- slope of the normalised weight over position (least squares), separately for whisker hits, whisker misses,
  auditory hits (>= 8 trials each); the trend is kept, no detrending and no shift null;
- contrast d = slope(whisker hits) - slope(auditory hits): same lick and reward, but only the whisker association is
  being learned on day 0 (auditory was learned before). A lick / reward or drift component changes alike on both,
  d ~ 0; a whisker-learning component changes on whisker hits only;
- fifth-of-block means per class (for the figures).
Per cluster (per-mouse values; members of the same mouse averaged, as in 013): one-sample Wilcoxon per cohort of each slope and of d; R+ vs
R- Mann-Whitney and Welch of d (R- = same session structure without reward for whisker licks; the cohort contrast
isolates the reward-contingency part). Candidate: d differs from 0 in R+ (p < 0.05) and R+ vs R- d differs (Mann-Whitney
p < 0.05). Agreement with the reference fit through 014's cluster matching.

Run: python 017_trend_contrasts.py [--n-jobs 40]
     -> tables/trend_members.parquet, tables/trend_clusters.csv, figures/aligned_gpu/trend_overview.png|pdf,
        figures/aligned_gpu/trend_candidates_<fit>.png|pdf
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
CLASSES = {"wh_hit": ("whisker_trial", 1), "wh_miss": ("whisker_trial", 0), "au_hit": ("auditory_trial", 1)}
MIN_N, NB = 8, 5
ALS = ["stim_long", "stim_sensory"]
FITN = {"_splitnn": "main fit", "_nn": "reference fit"}


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
        w = d[f"r{int(r['rank'])}_f0"][:, int(r["comp"]) - 1]
        act = ~t.epoch.str.startswith("passive").to_numpy()
        wa, ta = w[act], t[act].reset_index(drop=True)
        scale = max(wa.mean(), 1e-12)
        wa = wa / scale
        pos = np.arange(len(wa)) / max(len(wa) - 1, 1)
        tt, lk = ta.trial_type.to_numpy(), ta.lick_flag.fillna(0).to_numpy().astype(int)
        res = {k: r[k] for k in ("session_id", "mouse_id", "cohort", "learner", "method", "alignment", "cluster", "comp")}
        for name, (ty, l_) in CLASSES.items():
            m = (tt == ty) & (lk == l_)
            res[f"n_{name}"] = int(m.sum())
            res[f"slope_{name}"] = float(np.polyfit(pos[m], wa[m], 1)[0]) if m.sum() >= MIN_N else np.nan
            b = np.minimum((pos[m] * NB).astype(int), NB - 1)
            res[f"fifths_{name}"] = json.dumps([float(wa[m][b == i].mean()) if (b == i).sum() >= 2 else None
                                                for i in range(NB)])
        res["d"] = res["slope_wh_hit"] - res["slope_au_hit"]
        out.append(res)
    return out


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), float(stats.wilcoxon(v).pvalue), len(v)) if len(v) >= 5 else (np.nan, np.nan, len(v))


def cluster_stats(tm):
    rows = []
    V = ["slope_wh_hit", "slope_wh_miss", "slope_au_hit", "d"]
    for (meth, al, k), g in tm.groupby(["method", "alignment", "cluster"]):
        # one value per mouse: a cluster can hold several components of the same session (013), averaged like 013
        g = g.groupby(["mouse_id", "cohort"], as_index=False)[V].mean()
        row = dict(method=meth, alignment=al, cluster=k)
        for coh in ("R+", "R-"):
            h = g[g.cohort == coh]
            for v in ("slope_wh_hit", "slope_wh_miss", "slope_au_hit", "d"):
                row[f"{v}_{coh}_median"], row[f"{v}_{coh}_p"], row[f"{v}_{coh}_n"] = wil(h[v])
        a, b = (g.loc[g.cohort == c, "d"].dropna().to_numpy() for c in ("R+", "R-"))
        ok = len(a) >= 5 and len(b) >= 5
        row["d_RpRm_mw_p"] = float(stats.mannwhitneyu(a, b).pvalue) if ok else np.nan
        row["d_RpRm_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue) if ok else np.nan
        rows.append(row)
    cs = pd.DataFrame(rows)
    cs["candidate"] = (cs["d_R+_p"] < 0.05) & (cs["d_RpRm_mw_p"] < 0.05)
    return cs


def overview(cs, cl, out):
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.6), gridspec_kw=dict(hspace=0.5, wspace=0.4))
    for i, meth in enumerate(("_splitnn", "_nn")):
        for j, al in enumerate(ALS):
            ax = axs[i, j]
            g = cs[(cs.method == meth) & (cs.alignment == al)]
            for _, r in g.iterrows():
                c = "k" if r.candidate else "0.65"
                ax.plot(r["d_R+_median"], r["d_R-_median"], "o", ms=3 if r.candidate else 2, color=c)
                if r.candidate:
                    ax.annotate(f"cl. {r.cluster}", (r["d_R+_median"], r["d_R-_median"]),
                                fontsize=5, xytext=(3, 2), textcoords="offset points")
            lim = np.nanmax(np.abs(g[["d_R+_median", "d_R-_median"]].to_numpy())) * 1.15 if len(g) else 1
            ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.4, ls=":")
            ax.axhline(0, color="0.7", lw=0.4)
            ax.axvline(0, color="0.7", lw=0.4)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="R+ median d (whisker-hit − auditory-hit slope)",
                   ylabel="R− median d", title=f"{al}, {FITN[meth]} ({len(g)} clusters)")
    fig.suptitle("Whisker-hit minus auditory-hit trend per recurring cluster (black = candidate: R+ d ≠ 0 and R+ ≠ R−, "
                 "p < 0.05 uncorrected)", fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"trend_overview.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)


def candidates_page(cs, tm, mem, cl, meth, out):
    sel = cs[(cs.method == meth) & cs.candidate].sort_values(["alignment", "d_RpRm_mw_p"])
    if not len(sel):
        print("no candidates", meth)
        return
    n = len(sel)
    fig, axs = plt.subplots(n, 4, figsize=(7.4, 1.75 * n + 0.4), squeeze=False,
                            gridspec_kw=dict(hspace=0.9, wspace=0.55, width_ratios=[1, 1.2, 1.2, 1]))
    for i, (_, r) in enumerate(sel.iterrows()):
        rd = r
        g = tm[(tm.method == meth) & (tm.alignment == r.alignment) & (tm.cluster == r.cluster)]
        m = mem[(mem.method == meth) & (mem.alignment == r.alignment) & (mem.cluster == r.cluster)]
        c = cl.loc[(meth, r.alignment, r.cluster)]
        ax = axs[i, 0]
        tc = np.array(json.loads(m.t_centres_ms.iloc[0]))
        for coh, h in m.groupby("cohort"):
            ax.plot(tc, np.array([json.loads(x) for x in h.time_factor]).mean(0), color=COH[coh], lw=1)
        ax.axvline(0, color="0.6", lw=0.4)
        nm = g.groupby("cohort").mouse_id.nunique()
        ax.set_title(f"{r.alignment} cl. {r.cluster}\npeak {c.peak_ms_median:.0f} ms, top {c.top_group.replace('Somatosensory-', 'Ss-')}\n"
                     f"R+ {nm.get('R+', 0)} / R− {nm.get('R-', 0)} mice", fontsize=5.5)
        ax.set_xlabel("Time from stimulus (ms)")
        for j, coh in enumerate(("R+", "R-")):
            ax = axs[i, 1 + j]
            h = g[g.cohort == coh]
            xb = np.arange(1, NB + 1)
            for cls_, col, ls, mfc in (("wh_hit", WH, "-", WH), ("wh_miss", WH, "--", "white"), ("au_hit", AU, "-", AU)):
                Y = np.array([[np.nan if v is None else v for v in json.loads(x)] for x in h[f"fifths_{cls_}"]], float)
                if len(Y):   # mean per mouse first
                    Y = pd.DataFrame(Y).groupby(h.mouse_id.to_numpy()).mean().to_numpy()
                if not len(Y):
                    continue
                mu = np.nanmean(Y, 0)
                se = np.nanstd(Y, 0) / np.sqrt(np.maximum(np.sum(np.isfinite(Y), 0) - 1, 1))
                ax.errorbar(xb, mu, se, color=col, ls=ls, marker="o", ms=2.5, mfc=mfc, lw=0.8, elinewidth=0.4,
                            label={"wh_hit": "whisker hit", "wh_miss": "whisker miss", "au_hit": "auditory hit"}[cls_])
            ax.set_xticks(xb)
            ax.set_xlabel("Fifth of the active block")
            ax.set_title(f"{coh}: slope W hit {rd[f'slope_wh_hit_{coh}_median']:+.2f}, A hit "
                         f"{rd[f'slope_au_hit_{coh}_median']:+.2f}\nd {rd[f'd_{coh}_median']:+.2f} (Wilcoxon p "
                         f"{rd[f'd_{coh}_p']:.3f})", fontsize=5.5, color=COH[coh])
            if j == 0:
                ax.set_ylabel("Weight / mean on active trials")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=5)
        ax = axs[i, 3]
        for x, coh in enumerate(("R+", "R-")):
            v = g[g.cohort == coh].groupby("mouse_id").d.mean().dropna().to_numpy()
            ax.scatter(x + (np.random.default_rng(x).random(len(v)) - 0.5) * 0.3, v, s=3, color=COH[coh], lw=0)
            if len(v):
                ax.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, color="k", lw=1)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xticks([0, 1], ["R+", "R−"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("d = slope W hit − A hit")
        ax.set_title(f"R+ vs R−: MW p {r.d_RpRm_mw_p:.3f}\nWelch p {r.d_RpRm_welch_p:.3f}", fontsize=5.5)
    fig.subplots_adjust(top=1 - 0.75 / (1.75 * n + 0.4))
    fig.suptitle(f"Trend candidates, {FITN[meth]} (whisker day 0; exploratory, uncorrected)", fontsize=7, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"trend_candidates{meth}.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true", help="reuse tables/trend_members.parquet")
    a = ap.parse_args()
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members.parquet")
    mem = mem[(mem.stage == "learning") & mem.alignment.isin(ALS)]
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    recs = mem.drop(columns=["time_factor", "area_profile", "t_centres_ms"]).to_dict("records")
    by = {}
    for r in recs:
        by.setdefault(r["session_id"], []).append(r)
    f_tm = L.OUT / "tables" / "trend_members.parquet"
    if a.reuse and f_tm.exists():
        tm = pd.read_parquet(f_tm)
    else:
        res = Parallel(n_jobs=a.n_jobs)(delayed(session_rows)(s, rs) for s, rs in by.items())
        tm = pd.DataFrame([x for xs in res for x in xs])
        tm.to_parquet(f_tm, index=False)
    cs = cluster_stats(tm)
    ag = pd.read_csv(L.OUT / "tables" / "method_agreement.csv")
    agm = ag[["alignment", "split_cluster", "ref_cluster"]].rename(columns={"split_cluster": "cluster"}).assign(method="_splitnn")
    cs = cs.merge(agm, on=["method", "alignment", "cluster"], how="left")
    keep = ["d_R+_median", "d_R+_p", "d_R-_median", "d_RpRm_mw_p", "d_RpRm_welch_p", "candidate"]
    refc = cs.loc[cs.method == "_nn", ["alignment", "cluster"] + keep].rename(
        columns={"cluster": "ref_cluster", **{c: f"ref_{c}" for c in keep}})
    cs = cs.merge(refc, on=["alignment", "ref_cluster"], how="left")
    cs.to_csv(L.OUT / "tables" / "trend_clusters.csv", index=False)
    out = L.OUT / "figures" / "aligned_gpu"
    overview(cs, cl, out)
    for meth in ("_splitnn", "_nn"):
        candidates_page(cs, tm, mem, cl, meth, out)
    pd.set_option("display.width", 250)
    cols = ["method", "alignment", "cluster", "d_R+_median", "d_R+_p", "d_R+_n", "d_R-_median", "d_R-_p", "d_R-_n",
            "d_RpRm_mw_p", "d_RpRm_welch_p", "slope_wh_hit_R+_median", "slope_au_hit_R+_median", "ref_cluster",
            "ref_d_R+_median", "ref_d_RpRm_mw_p", "ref_candidate"]
    print(cs[cs.candidate][[c for c in cols if c in cs]].round(3).to_string(index=False))
    print(cs.groupby(["method", "alignment"]).candidate.agg(["sum", "size"]))


if __name__ == "__main__":
    main()
