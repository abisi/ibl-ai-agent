"""Do the recurring components of the main fit (split non-negative, baseline-subtracted) reappear in the reference fit
(non-negative on rates, no baseline subtraction)? For every split cluster (013), the best-matching reference cluster of
the same alignment: similarity of the cluster means (Pearson of mean time factor, mean area profile on shared groups,
mean trial-class profile; averaged, as in 013). Reports both clusters' kinds, cohort make-up and behaviour statistics
side by side. Exploratory.

Run: python 014_method_agreement.py -> tables/method_agreement.csv
"""
import json

import numpy as np
import pandas as pd

import tca_lib as L

CLS = ["cls_A_whisker_1", "cls_A_whisker_0", "cls_A_auditory_1", "cls_A_auditory_0", "cls_P_whisker", "cls_P_auditory"]


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 and ok.sum() >= 3 else np.nan


def means(m):
    tf = np.mean([json.loads(x) for x in m.time_factor], 0)
    groups = sorted({k for x in m.area_profile for k in json.loads(x)})
    ap = {g: np.nanmean([json.loads(x).get(g, np.nan) for x in m.area_profile]) for g in groups}
    return tf, ap, m[CLS].mean().to_numpy(float)


def main():
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members.parquet")
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    keep = ["kind", "n_mice", "frac_Rp", "frac_Rm", "cohort_bias_p", "top_group", "peak_ms_median",
            "all_learnctrl_R+_median", "all_learnctrl_R+_wilcoxon_p", "all_learnctrl_R-_median",
            "all_learnctrl_R-_wilcoxon_p", "all_learnctrl_RpRm_mw_p", "all_learnctrl_RpRm_welch_p"]
    rows = []
    for al, g in mem.groupby("alignment"):
        ref = {k: means(m) for k, m in g[g.method == "_nn"].groupby("cluster")}
        for k, m in g[g.method == "_splitnn"].groupby("cluster"):
            tf, ap, cp = means(m)
            best, bs = None, -np.inf
            for k2, (tf2, ap2, cp2) in ref.items():
                shared = [x for x in ap if x in ap2]
                parts = [corr(tf, tf2), corr(np.array([ap[x] for x in shared]), np.array([ap2[x] for x in shared])),
                         corr(cp, cp2)]
                s = np.nanmean(parts)
                if s > bs:
                    best, bs = k2, s
            row = dict(alignment=al, split_cluster=int(k), ref_cluster=int(best), match_similarity=float(bs))
            for tag, kk in (("split", ("_splitnn", al, k)), ("ref", ("_nn", al, best))):
                for c in keep:
                    row[f"{tag}_{c}"] = cl.loc[kk, c] if c in cl.columns else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / "tables" / "method_agreement.csv", index=False)
    pd.set_option("display.width", 250)
    cand = df[(df["split_all_learnctrl_RpRm_mw_p"] < 0.05) | (df["split_cohort_bias_p"] < 0.05)
              | (df["split_all_learnctrl_R+_wilcoxon_p"] < 0.01)]
    cols = ["alignment", "split_cluster", "ref_cluster", "match_similarity", "split_kind", "ref_kind",
            "split_peak_ms_median", "ref_peak_ms_median", "split_cohort_bias_p", "ref_cohort_bias_p",
            "split_all_learnctrl_R+_median", "ref_all_learnctrl_R+_median", "split_all_learnctrl_R-_median",
            "ref_all_learnctrl_R-_median", "split_all_learnctrl_RpRm_mw_p", "ref_all_learnctrl_RpRm_mw_p"]
    print(cand[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
