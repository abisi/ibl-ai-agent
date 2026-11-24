"""Recurring components across mice and their relation to performance / learning (exploratory, all day-0 sessions).

Input: tables/components.parquet (012). Per method x alignment, whisker day-0 sessions (one per mouse):
1. Similarity between two components of different mice = mean of three Pearson correlations on shared axes (centred,
   because raw cosines of non-negative profiles are high for almost any pair): time factor (same bins), area profile
   (mean net unit weight per area group, groups present in both; skipped if < 3 shared), trial-class profile (mean
   trial weight per class, classes present in both; skipped if < 3 shared).
2. Clustering: average-linkage hierarchical clustering on 1 - similarity; components of the same session are never
   merged (distance set to the maximum). Cut threshold from the data: the 95th percentile of the similarity between
   different components of the same session (what "two distinct components" look like).
3. Per cluster (members from >= 10 mice): cohort make-up (fraction of R+ / R- mice with a member; cohort bias tested by
   permuting cohort labels across mice, 2000 permutations), area consistency (share of members with the same top area
   group and specificity p < 0.05), time summary (median peak, share flat), directionality (split fits), and behaviour:
   per mouse (members averaged) excess of hit / miss AUC and of the partial learning-curve correlation over their
   shift nulls (012); one-sample Wilcoxon against 0 per cohort; R+ vs R- with Mann-Whitney and Welch (both reported);
   populations: all mice and learners. Uncorrected (exploratory).
4. Label per cluster: variability (median directionality |.| < 0.5, split fits), flat (median flat_cv < 0.2),
   area-specific (>= 50 % of members share the top group with specificity p < 0.05), else global; cohort-specific if
   the cohort-bias permutation p < 0.05.

Run: python 013_recurring_components.py -> tables/clusters.csv, tables/cluster_members.parquet,
     figures/aligned_gpu/clusters_<method>_<alignment>.png|pdf
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

import tca_lib as L  # noqa: E402
import os as _os
# optional subset of alignments written to separate tables (user 2026-11-23: go-signal tensors), e.g.
# TCA_ALIGNMENTS=stim_long_ns,jaw_all TCA_SUFFIX=_go; default: the four original alignments, original tables
SUFFIX = _os.environ.get("TCA_SUFFIX", "")


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
CLS = ["cls_A_whisker_1", "cls_A_whisker_0", "cls_A_auditory_1", "cls_A_auditory_0", "cls_P_whisker", "cls_P_auditory"]
CLS_LAB = ["W hit", "W miss", "A hit", "A miss", "W pass.", "A pass."]
MIN_MICE = 10
N_PERM = 2000


def corr(a, b):
    """Pearson correlation (centred cosine): raw cosines of non-negative profiles are high for almost any pair"""
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def similarity(x, y):
    parts = [corr(x["tf"], y["tf"])]
    g = [k for k in x["ap"] if k in y["ap"]]
    if len(g) >= 3:
        parts.append(corr(np.array([x["ap"][k] for k in g]), np.array([y["ap"][k] for k in g])))
    ok = np.isfinite(x["cp"]) & np.isfinite(y["cp"])
    if ok.sum() >= 3:
        parts.append(corr(x["cp"][ok], y["cp"][ok]))
    parts = [p for p in parts if np.isfinite(p)]
    return float(np.mean(parts)) if parts else 0.0


def group_tests(v, coh, lab):
    out = {}
    for c in ("R+", "R-"):
        x = v[coh == c]
        x = x[np.isfinite(x)]
        out[f"{lab}_{c}_n"] = len(x)
        out[f"{lab}_{c}_median"] = float(np.median(x)) if len(x) else np.nan
        out[f"{lab}_{c}_wilcoxon_p"] = float(stats.wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan
    a, b = v[(coh == "R+") & np.isfinite(v)], v[(coh == "R-") & np.isfinite(v)]
    if len(a) >= 3 and len(b) >= 3:
        out[f"{lab}_RpRm_mw_p"] = float(stats.mannwhitneyu(a, b).pvalue)
        out[f"{lab}_RpRm_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return out


def main():
    c = pd.read_parquet(L.OUT / "tables" / f"components{SUFFIX}.parquet")
    if "error" in c:
        c = c[c.error.isna()]
    c = c[c.stage == "learning"].reset_index(drop=True)
    all_clusters, members = [], []
    rng = np.random.default_rng(0)
    for (tag, al), g in c.groupby(["method", "alignment"]):
        nb = g.time_factor.map(lambda x: len(json.loads(x)))
        g = g[nb == nb.mode().iloc[0]].reset_index(drop=True)   # one time grid per alignment (guards stale files)
        feats = [dict(tf=np.array(json.loads(r.time_factor)), ap=json.loads(r.area_profile),
                      cp=np.array([getattr(r, k) for k in CLS], float)) for r in g.itertuples()]
        n = len(g)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                S[i, j] = S[j, i] = similarity(feats[i], feats[j])
        same = g.session_id.to_numpy()[:, None] == g.session_id.to_numpy()[None]
        within = S[np.triu(same, 1)]
        thr = float(np.percentile(within, 95))
        Dm = 1 - S
        Dm[same] = 2.0
        np.fill_diagonal(Dm, 0)
        lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=1 - thr, criterion="distance")
        g["cluster"] = lab
        mice = g.drop_duplicates("mouse_id").set_index("mouse_id")
        coh_m = mice.cohort
        n_coh = coh_m.value_counts()
        for k in np.unique(lab):
            m = g[g.cluster == k]
            mm = m.mouse_id.unique()
            if len(mm) < MIN_MICE:
                continue
            has = coh_m.index.isin(mm)
            frac = {cc: float(has[coh_m.to_numpy() == cc].mean()) for cc in ("R+", "R-")}
            stat = frac["R+"] - frac["R-"]
            perm = []
            cl = coh_m.to_numpy()
            for _ in range(N_PERM):
                pc = rng.permutation(cl)
                perm.append(has[pc == "R+"].mean() - has[pc == "R-"].mean())
            p_coh = float((np.sum(np.abs(perm) >= abs(stat)) + 1) / (N_PERM + 1))
            tops = m.top_group.value_counts()
            spec_share = float(((m.top_group == tops.index[0]) & (m.specificity_p < 0.05)).mean())
            per_mouse = m.groupby("mouse_id").agg(cohort=("cohort", "first"), learner=("learner", "first"),
                                                  perf=("perf_auc_excess", "mean"), learn=("learn_rho_excess", "mean"),
                                                  learnctrl=("learn_ctrl_excess", "mean"))
            # area coverage: share of each cohort's mice that have units in the cluster's top area group (a cohort
            # bias can come from which areas were recorded)
            cov = {cc: float(np.mean([tops.index[0] in json.loads(x) for x in
                                      g[g.cohort == cc].drop_duplicates("mouse_id").area_profile])) for cc in ("R+", "R-")}
            row = dict(method=tag, alignment=al, cluster=int(k), n_components=len(m), n_mice=len(mm),
                       top_group_coverage_Rp=cov["R+"], top_group_coverage_Rm=cov["R-"],
                       n_Rp=int(has[cl == "R+"].sum()), n_Rm=int(has[cl == "R-"].sum()),
                       frac_Rp=frac["R+"], frac_Rm=frac["R-"], cohort_bias=stat, cohort_bias_p=p_coh,
                       threshold=thr, top_group=tops.index[0], top_group_share=float(tops.iloc[0] / len(m)),
                       area_specific_share=spec_share, peak_ms_median=float(m.peak_ms.median()),
                       flat_share=float((m.flat_cv < 0.2).mean()),
                       directionality_median=float(m.directionality.median()) if m.directionality.notna().any()
                       else np.nan)
            for pop, sel in (("all", np.ones(len(per_mouse), bool)), ("learners", per_mouse.learner.astype(bool))):
                pm = per_mouse[sel]
                row.update({f"{pop}_{kk}": v for kk, v in group_tests(pm.perf.to_numpy(float), pm.cohort.to_numpy(),
                                                                      "perf").items()})
                row.update({f"{pop}_{kk}": v for kk, v in group_tests(pm.learn.to_numpy(float), pm.cohort.to_numpy(),
                                                                      "learn").items()})
                row.update({f"{pop}_{kk}": v for kk, v in group_tests(pm.learnctrl.to_numpy(float),
                                                                      pm.cohort.to_numpy(), "learnctrl").items()})
            kind = ("variability" if np.isfinite(row["directionality_median"]) and abs(row["directionality_median"]) < 0.5
                    else "flat" if m.flat_cv.median() < 0.2
                    else "area-specific" if spec_share >= 0.5 else "global")
            row["kind"] = kind
            row["cohort_specific"] = p_coh < 0.05
            all_clusters.append(row)
            members.append(m.assign(cluster=int(k)))
        print(f"{tag} {al}: {n} components, threshold {thr:.2f}, "
              f"{sum(1 for r in all_clusters if r['method'] == tag and r['alignment'] == al)} clusters with >= "
              f"{MIN_MICE} mice", flush=True)
        plot(g, [r for r in all_clusters if r["method"] == tag and r["alignment"] == al], tag, al, n_coh)
    cl = pd.DataFrame(all_clusters)
    cl.to_csv(L.OUT / "tables" / f"clusters{SUFFIX}.csv", index=False)
    pd.concat(members).to_parquet(L.OUT / "tables" / f"cluster_members{SUFFIX}.parquet", index=False)
    cols = ["method", "alignment", "cluster", "kind", "n_mice", "frac_Rp", "frac_Rm", "cohort_bias_p", "top_group",
            "peak_ms_median", "all_perf_R+_median", "all_perf_R-_median", "all_perf_RpRm_mw_p",
            "all_learn_R+_median", "all_learn_R-_median", "all_learn_RpRm_mw_p"]
    print(cl[[x for x in cols if x in cl]].round(3).to_string(index=False))


def plot(g, rows, tag, al, n_coh):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -r["n_mice"])[:10]
    fig, ax = plt.subplots(len(rows), 5, figsize=(7.4, 0.95 * len(rows) + 0.5), squeeze=False,
                           gridspec_kw={"width_ratios": [1.3, 1.8, 1.0, 0.7, 1.4]})
    groups = sorted({k for r in g.itertuples() for k in json.loads(r.area_profile)})
    for i, r in enumerate(rows):
        m = g[g.cluster == r["cluster"]]
        tcs = np.array(json.loads(m.t_centres_ms.iloc[0]))
        TF = np.array([json.loads(x) for x in m.time_factor])
        a = ax[i, 0]
        for coh in ("R+", "R-"):
            x = TF[m.cohort.to_numpy() == coh]
            if len(x):
                a.plot(tcs, x.mean(0), color=COH_COL[coh], lw=0.9)
        a.axvline(0, color="0.5", lw=0.4)
        a.set_ylabel(f"cl. {r['cluster']}\n{r['kind']}\n{r['n_mice']} mice", fontsize=5)
        AP = np.array([[json.loads(x).get(gg, np.nan) for gg in groups] for x in m.area_profile])
        ax[i, 1].bar(range(len(groups)), np.nanmean(AP, 0), color="0.4", width=0.8)
        ax[i, 1].axhline(0, color="0.6", lw=0.3)
        ax[i, 1].set_xticks(range(len(groups)) if i == len(rows) - 1 else [])
        if i == len(rows) - 1:
            ax[i, 1].set_xticklabels([x.replace("Somatosensory-", "Ss-") for x in groups], rotation=70, ha="right",
                                     fontsize=4.2)
        CP = m[CLS].to_numpy(float)
        ax[i, 2].bar(range(len(CLS)), np.nanmean(CP, 0), color=["#f7b519", "#fbd98c", "#2c2cdb", "#9595ed", "#fde7b5",
                                                                  "#c7c7f4"], width=0.8)
        ax[i, 2].set_xticks(range(len(CLS)) if i == len(rows) - 1 else [])
        if i == len(rows) - 1:
            ax[i, 2].set_xticklabels(CLS_LAB, rotation=70, ha="right", fontsize=4.5)
        ax[i, 3].bar([0, 1], [r["frac_Rp"], r["frac_Rm"]], color=[COH_COL["R+"], COH_COL["R-"]], width=0.7)
        ax[i, 3].set_ylim(0, 1.02)
        ax[i, 3].set_xticks([0, 1] if i == len(rows) - 1 else [])
        if i == len(rows) - 1:
            ax[i, 3].set_xticklabels(["R+", "R-"])
        ax[i, 3].set_title(f"p = {r['cohort_bias_p']:.3f}", fontsize=4.5)
        pm = m.groupby("mouse_id").agg(cohort=("cohort", "first"), learn=("learn_ctrl_excess", "mean"),
                                       perf=("perf_auc_excess", "mean"))
        for q, (col, lab_) in enumerate((("perf", "AUC"), ("learn", "rho"))):
            for jj, coh in enumerate(("R+", "R-")):
                v = pm[pm.cohort == coh][col].dropna().to_numpy()
                xx = q * 2.6 + jj + np.random.default_rng(0).uniform(-0.15, 0.15, len(v))
                ax[i, 4].scatter(xx, v, s=3, color=COH_COL[coh], lw=0)
                if len(v):
                    ax[i, 4].plot([q * 2.6 + jj - 0.25, q * 2.6 + jj + 0.25], [np.median(v)] * 2, color="k", lw=0.8)
        ax[i, 4].axhline(0, color="0.5", lw=0.4)
        ax[i, 4].set_xticks([0.5, 3.1] if i == len(rows) - 1 else [])
        if i == len(rows) - 1:
            ax[i, 4].set_xticklabels(["hit/miss AUC\nexcess", "learning rho excess\n(outcome-controlled)"],
                                     fontsize=4.5)
    ax[0, 0].set_title("Time factor (mean)", fontsize=5.5)
    ax[0, 1].set_title("Area profile (mean net weight)", fontsize=5.5)
    ax[0, 2].set_title("Trial-class weight", fontsize=5.5)
    ax[0, 4].set_title("Behaviour vs shift null\n(per mouse)", fontsize=5.5)
    ax[-1, 0].set_xlabel("Time from event (ms)")
    fig.suptitle(f"Recurring components (>= {MIN_MICE} mice), {al}, {tag.strip('_')}, whisker day 0 "
                 f"(R+ {n_coh.get('R+', 0)}, R- {n_coh.get('R-', 0)} mice); exploratory", fontsize=6, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    od = L.OUT / "figures" / "aligned_gpu"
    for ext in ("png", "pdf"):
        fig.savefig(od / f"clusters{tag}_{al}.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
