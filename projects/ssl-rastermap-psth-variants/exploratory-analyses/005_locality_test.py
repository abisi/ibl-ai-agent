"""Rastermap locality test on learners + good units: locality 0, 0.25, 0.75 (all else identical).

Reuses the feature matrix of qc_good__learncat_good_moderate (copied, not recomputed), so only the
rastermap fit differs. fit_rastermap (rastermap_utils) reads locality / time_lag_window /
grid_upsample from the module-level DEFAULT_CFG, not from config.yaml, so locality is set there.
run_clustering_new.fit_rastermap is wrapped to build the IDENTICAL Rastermap(...) call and, in
addition, keep the model's k-means node assignment (model.embedding_clust, nodes in sorted order)
and 1D embedding — saved for the cluster-definition question (data-driven nodes vs equal slices).

Per locality → rastermap_variants/qc_good__learncat_good_moderate__loc<tag>/ (same inner layout):
    run_rastermap (CV), fig_flow_umap_cv, run_stats_only, cluster-mean | F_mk figure (MWU + Welch),
    rastermap/rastermap_node_labels_cv.csv (unit_ids, node label, embedding, equal-slice label)
Comparison (all localities) → rastermap_variants/_locality_test_summary/
"""
import importlib
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
from rastermap import Rastermap
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import rastermap_psth.rastermap_utils as rmu
import rastermap_psth.run_clustering_new as rcn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
drv = importlib.import_module("000_run_rastermap_variants")   # reuse loading / figure helpers

SRC_VARIANT = "qc_good__learncat_good_moderate"
LOCALITIES = {"0p00": 0.0, "0p25": 0.25, "0p75": 0.75}
SUMMARY_DIR = drv.VARIANTS_ROOT / "_locality_test_summary"
CAPTURE = []


def fit_rastermap_capture(X, n_clusters):
    """Same call as rastermap_utils.fit_rastermap, plus capture of node labels / embedding."""
    n_pcs = min(200, X.shape[0] - 1, X.shape[1] - 1)
    model = Rastermap(n_clusters=n_clusters, n_PCs=n_pcs,
                      locality=rmu.DEFAULT_CFG["locality"],
                      time_lag_window=rmu.DEFAULT_CFG["time_lag_window"],
                      grid_upsample=rmu.DEFAULT_CFG["grid_upsample"],
                      verbose=False).fit(X)
    isort = model.isort
    bounds = np.round(np.linspace(0, len(isort), n_clusters + 1)[1:-1]).astype(int)
    CAPTURE.append(dict(node=np.asarray(model.embedding_clust).copy(),
                        embedding=np.asarray(model.embedding)[:, 0].copy()))
    return isort, bounds


rcn.fit_rastermap = fit_rastermap_capture


def copy_data_folder(src_root, dst_root):
    src = sorted(src_root.glob("rastermap_clustering/*/*/*/feature_matrix.npz"))[0].parent
    dst = dst_root / src.relative_to(src_root)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("feature_matrix.npz", "neuron_metadata.csv", "config_used.yaml"):
        if (src / f).exists() and not (dst / f).exists():
            shutil.copyfile(src / f, dst / f)  # content only: NAS refuses copystat
    return dst


if __name__ == "__main__":
    tables = drv.load_tables()
    cfg = drv.load_cfg(drv.CONFIG_PATH, unit_quality_label=["good"])
    src_root = drv.VARIANTS_ROOT / SRC_VARIANT
    SUMMARY_DIR.mkdir(exist_ok=True)
    rows, labels_by = [], {}
    for tag, loc in LOCALITIES.items():
        name = f"{SRC_VARIANT}__loc{tag}"
        root = drv.VARIANTS_ROOT / name
        print(f"\n=== {name}: locality={loc} (time_lag_window={rmu.DEFAULT_CFG['time_lag_window']}, "
              f"grid_upsample={rmu.DEFAULT_CFG['grid_upsample']}) ===", flush=True)
        data_folder = copy_data_folder(src_root, root)
        rmu.DEFAULT_CFG["locality"] = loc
        CAPTURE.clear()
        rres = rcn.run_rastermap(data_folder=data_folder, config_path=drv.CONFIG_PATH, unit_quality_label=["good"])
        md = pathlib.Path(rres["method_dir"])
        cap = CAPTURE[-1]                                   # CV fit (odd trials); do_global is off
        d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
        assert len(cap["node"]) == len(d["unit_ids"])
        node_df = pd.DataFrame(dict(unit_ids=d["unit_ids"], node_label=cap["node"], embedding=cap["embedding"],
                                    equal_slice_label=d["cluster_labels"]))
        node_df.to_csv(md / "rastermap_node_labels_cv.csv", index=False)
        drv.fig_flow_umap_cv(md, cfg)
        drv.run_stats_only(md, cfg, tables["unit_table"], tables["trial_table"])
        st = drv.fig_cluster_mean_matrix_with_fmk(md, cfg, name)
        perm = (md / "stats" / "permanova_summary.txt").read_text()
        F = float(perm.split("F-statistic :")[1].split()[0]); p = float(perm.split("p-value     :")[1].split()[0])
        sizes = np.bincount(cap["node"])
        cvm = rres.get("cv_metrics")
        cv_flat = {}
        if isinstance(cvm, dict):
            for k, v in cvm.items():
                try:
                    cv_flat[f"cv_{k}"] = float(np.nanmean(np.asarray(v, dtype=float)))
                except (TypeError, ValueError):
                    pass
        rows.append(dict(locality=loc, variant=name, n_neurons=len(node_df), permanova_F=F, permanova_p=p,
                         n_sig_mwu=int((st.p_mannwhitney_fdr < 0.05).sum()), n_sig_welch=int((st.p_welch_t_fdr < 0.05).sum()),
                         node_size_min=int(sizes.min()), node_size_median=float(np.median(sizes)),
                         node_size_max=int(sizes.max()), n_nodes=len(sizes), **cv_flat))
        labels_by[loc] = node_df.set_index("unit_ids")
        print(json.dumps(rows[-1], default=float), flush=True)

    # agreement between localities (same neurons): node labels, equal slices, 1D order
    comp = []
    locs = list(labels_by)
    for i in range(len(locs)):
        for j in range(i + 1, len(locs)):
            a, b = labels_by[locs[i]], labels_by[locs[j]].reindex(labels_by[locs[i]].index)
            comp.append(dict(loc_a=locs[i], loc_b=locs[j],
                             ari_node_labels=adjusted_rand_score(a.node_label, b.node_label),
                             ari_equal_slices=adjusted_rand_score(a.equal_slice_label, b.equal_slice_label),
                             spearman_embedding=spearmanr(a.embedding, b.embedding)[0]))
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / "locality_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(SUMMARY_DIR / "locality_agreement.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(comp).round(3).to_string(index=False), flush=True)
    print("ALL DONE", flush=True)
