"""run_stats_only (rastermap_utils, unchanged) on METACLUSTER labels, per rastermap variant.

For each variant, <method_dir>/metacluster_stats/ gets:
    rastermap_results_cv.npz        copy of the CV results with
                                      cluster_labels = metacluster - 1  (metaclusters 1..F in
                                                        dendrogram leaf order, see 003_metacluster_fmk.py)
                                      isort          = neurons ordered by metacluster, rastermap order within
                                      boundaries     = metacluster edges along isort
                                    (X = even trials, unit_ids, mouse/reward arrays, timing unchanged)
    neuron_cluster_labels_cv.csv    copy with cluster_label = metacluster - 1 (area lookup for run_stats_only)
    stats/                          everything run_reward_group_stats writes (F_mk computed there directly
                                    from the metacluster labels, PERMANOVA, per-metacluster Mann-Whitney
                                    + BH-FDR, dispersion stats, figS1-S5, post-hoc figures)
    stats/per_metacluster_stats_mwu_welch.csv
                                    sidecar: Mann-Whitney U and Welch t on F_mk, each BH-FDR (R+/R- test pair)
No minimum number of neurons per mouse (same mouse set as the fine-cluster stats).
Tables: unit_table / trial_table from the 000 driver's NAS cache (same loading as unit_spikes_analysis.py).
"""
import argparse
import json
import pathlib
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

import rastermap_psth.rastermap_utils as rmu
from rastermap_psth.rastermap_utils import run_stats_only, load_cfg

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CACHE = VARIANTS_ROOT / "_cache" / "tables_day0.pkl"
CONFIG_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def build_metacluster_folder(md):
    out = md / "metacluster_stats"
    mc_map = pd.read_csv(out / "metacluster_map.csv")          # from 003_metacluster_fmk.py
    mc_of = dict(zip(mc_map.cluster - 1, mc_map.metacluster))    # fine cluster (0-based) -> metacluster (1..F)
    d = dict(np.load(md / "rastermap_results_cv.npz", allow_pickle=True))
    labels = d["cluster_labels"].astype(int)
    mc0 = np.array([mc_of[c] - 1 for c in labels])               # 0-based metacluster per neuron
    isort_old = d["isort"]
    isort = np.concatenate([isort_old[mc0[isort_old] == j] for j in range(mc0.max() + 1)])
    sizes = np.bincount(mc0, minlength=mc0.max() + 1)
    d.update(cluster_labels=mc0, isort=isort, boundaries=np.cumsum(sizes)[:-1])
    np.savez_compressed(out / "rastermap_results_cv.npz", **d)
    meta = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    lab_col = "cluster_label" if "cluster_label" in meta else "cluster_label_cv"
    meta["fine_cluster_label"] = meta[lab_col]
    meta[lab_col] = meta[lab_col].map(lambda c: mc_of[c] - 1)
    meta.to_csv(out / "neuron_cluster_labels_cv.csv", index=False)
    return out, int(mc0.max() + 1)


def welch_sidecar(out, variant):
    fz = np.load(out / "stats" / "f_matrix.npz", allow_pickle=True)
    F, g = fz["f_matrix"], fz["reward_groups"].astype(str)
    rows = []
    for j in range(F.shape[1]):
        a, b = F[g == "R+", j], F[g == "R-", j]
        rows.append(dict(variant=variant, metacluster=j + 1, n_rplus=len(a), n_rminus=len(b),
                         mean_rplus=a.mean(), sem_rplus=a.std(ddof=1) / np.sqrt(len(a)),
                         mean_rminus=b.mean(), sem_rminus=b.std(ddof=1) / np.sqrt(len(b)),
                         p_mannwhitney=mannwhitneyu(a, b, alternative="two-sided").pvalue,
                         p_welch_t=ttest_ind(a, b, equal_var=False).pvalue))
    t = pd.DataFrame(rows)
    t["p_mannwhitney_fdr"] = bh(t.p_mannwhitney)
    t["p_welch_t_fdr"] = bh(t.p_welch_t)
    t.to_csv(out / "stats" / "per_metacluster_stats_mwu_welch.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    t0 = time.time()
    print(f"Loading cached tables {CACHE}", flush=True)
    with open(CACHE, "rb") as f:
        tables = pickle.load(f)
    print(f"  loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    for md in sorted(VARIANTS_ROOT.glob("qc_*/rastermap_clustering/*/*/clustering/*/rastermap")):
        variant = next(p for p in md.parts if p.startswith("qc_"))
        if a.variants and variant not in a.variants:
            continue
        t1 = time.time()
        print(f"\n=== {variant} ===", flush=True)
        try:
            out, n_mc = build_metacluster_folder(md)
            qc = ["good"] if variant.startswith("qc_good__") else ["good", "mua"]
            cfg = load_cfg(CONFIG_PATH, unit_quality_label=qc)
            (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS,
             rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
            run_stats_only(out, cfg, tables["unit_table"], tables["trial_table"])
            t = welch_sidecar(out, variant)
            summ = dict(variant=variant, n_metaclusters=n_mc,
                        n_sig_mwu_fdr=int((t.p_mannwhitney_fdr < 0.05).sum()),
                        n_sig_welch_fdr=int((t.p_welch_t_fdr < 0.05).sum()),
                        sig_mwu=t.loc[t.p_mannwhitney_fdr < 0.05, "metacluster"].tolist(),
                        sig_welch=t.loc[t.p_welch_t_fdr < 0.05, "metacluster"].tolist(),
                        runtime_min=round((time.time() - t1) / 60, 1))
            print(json.dumps(summ), flush=True)
            print((out / "stats" / "permanova_summary.txt").read_text(), flush=True)
        except Exception:
            import traceback
            print(f"=== {variant} FAILED ===\n{traceback.format_exc()}", flush=True)
    print("ALL DONE", flush=True)
