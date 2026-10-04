"""Control for reward/lick behaviour on whisker-hit trials (R+ rewarded, R- not): refit rastermap with
the whisker-hit conditions trimmed so they contain less (or no) post-lick / post-reward time.

Source feature matrix: qc_good_mua__learncat_good_moderate (same neurons, same PSTHs; bins are SLICED
from the existing condition columns, nothing recomputed). Settings = latest decisions: locality 0.25,
grid_upsample 10, clusters = rastermap k-means nodes (as 006), contiguous metaclusters (default).

    trimStim : "Whisker hit" (stimulus-aligned, -0.1..0.3 s)   -> keep t <= 0.2 s
    trimLick : "Whisker hit (lick)" (lick-aligned, -0.3..0.1 s) -> keep t <= 0.0 s (up to the lick)
    trimBoth : both
Auditory hit (rewarded in both cohorts) and all other conditions are unchanged.

Per test → rastermap_variants/qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__<test>/:
    run_rastermap (CV) + fig_flow_umap_cv, run_stats_only, cluster-mean | F_mk figure (MWU + Welch),
    matrix summary (incl. publication layout). Summary table → rastermap_variants/_trim_test_summary/
"""
import argparse
import importlib
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith("--src") and not a.startswith("--locality")]
g10 = importlib.import_module("006_grid10_node_clusters")    # installs the k-means-node fit_rastermap wrapper
drv = g10.drv
import rastermap_psth.rastermap_utils as rmu
import rastermap_psth.run_clustering_new as rcn
from rastermap_psth.population_matrix_summary import run_matrix_summary

SRC = "qc_good_mua__learncat_good_moderate"
QC = ["good", "mua"]
LOCALITY = 0.25
TESTS = {"trimStim": {"Whisker hit": 0.2},
         "trimLick": {"Whisker hit (lick)": 0.0},
         "trimBoth": {"Whisker hit": 0.2, "Whisker hit (lick)": 0.0}}
SUMMARY_DIR = drv.VARIANTS_ROOT / "_trim_test_summary"


def trimmed_data_folder(src_root, dst_root, cuts, cond_labels):
    src = sorted(src_root.glob("rastermap_clustering/*/*/*/feature_matrix.npz"))[0].parent
    dst = dst_root / src.relative_to(src_root)
    dst.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(src / "feature_matrix.npz", allow_pickle=True))
    nb = list(d["n_bins_list"].astype(int)); n_c = int(d["n_conds"])
    off = np.r_[0, np.cumsum(nb)]
    keep_cols, new_nb, report = [], [], {}
    for ci in range(n_c):
        t = d[f"t_ctr_{ci}"]
        lab = cond_labels[ci]
        keep = t <= cuts[lab] + 1e-9 if lab in cuts else np.ones(len(t), bool)
        keep_cols.append(np.arange(off[ci], off[ci + 1])[keep])
        d[f"t_ctr_{ci}"] = t[keep]; new_nb.append(int(keep.sum()))
        if lab in cuts:
            report[lab] = dict(t_min=float(t.min()), t_max_before=float(t.max()), t_max_after=float(t[keep].max()),
                               bins_before=int(len(t)), bins_after=int(keep.sum()))
    cols = np.concatenate(keep_cols)
    for k in ("X", "X_odd", "X_even"):
        d[k] = d[k][:, cols]
    d["n_bins_list"] = np.array(new_nb)
    np.savez_compressed(dst / "feature_matrix.npz", **d)
    for f in ("neuron_metadata.csv", "config_used.yaml"):
        if (src / f).exists():
            shutil.copyfile(src / f, dst / f)
    return dst, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="*", default=list(TESTS))
    a = ap.parse_args()
    rmu.DEFAULT_CFG["locality"] = LOCALITY
    tables = drv.load_tables()
    cfg = drv.load_cfg(drv.CONFIG_PATH, unit_quality_label=QC)
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS,
     rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
    cond_labels = list(rmu.COND_LABELS)
    SUMMARY_DIR.mkdir(exist_ok=True)
    rows = []
    for test in a.tests:
        name = f"{SRC}__loc0p25_grid10_nodes__{test}"
        root = drv.VARIANTS_ROOT / name
        data_folder, rep = trimmed_data_folder(drv.VARIANTS_ROOT / SRC, root, TESTS[test], cond_labels)
        print(f"\n=== {name}: {json.dumps(rep)} ===", flush=True)
        g10.CAPTURE.clear()
        rres = rcn.run_rastermap(data_folder=data_folder, config_path=drv.CONFIG_PATH, unit_quality_label=QC)
        md = pathlib.Path(rres["method_dir"])
        d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
        assert np.array_equal(d["cluster_labels"].astype(int), g10.CAPTURE[-1]["node"]), "labels != k-means nodes"
        drv.fig_flow_umap_cv(md, cfg)
        drv.run_stats_only(md, cfg, tables["unit_table"], tables["trial_table"])
        st = drv.fig_cluster_mean_matrix_with_fmk(md, cfg, name)
        both = st[(st.p_mannwhitney_fdr < 0.05) & (st.p_welch_t_fdr < 0.05)]
        perm = (md / "stats" / "permanova_summary.txt").read_text()
        F = float(perm.split("F-statistic :")[1].split()[0]); p = float(perm.split("p-value     :")[1].split()[0])
        row = dict(test=test, variant=name, cuts=json.dumps(TESTS[test]), n_neurons=int(len(d["unit_ids"])),
                   permanova_F=F, permanova_p=p, n_sig_mwu=int((st.p_mannwhitney_fdr < 0.05).sum()),
                   n_sig_welch=int((st.p_welch_t_fdr < 0.05).sum()), n_sig_both=int(len(both)),
                   sig_both_rplus=[int(r.cluster) for r in both.itertuples() if r.mean_f_rplus > r.mean_f_rminus],
                   sig_both_rminus=[int(r.cluster) for r in both.itertuples() if r.mean_f_rplus <= r.mean_f_rminus])
        rows.append(row)
        print(json.dumps(row), flush=True)
        run_matrix_summary(md, drv.CONFIG_PATH, dict(unit_quality_label=QC), title_prefix=f"{name}: ")
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "trim_test_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("ALL DONE", flush=True)
