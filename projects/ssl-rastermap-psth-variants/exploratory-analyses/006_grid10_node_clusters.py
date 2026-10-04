"""Learners + good units, rastermap grid_upsample=10, clusters = rastermap's internal k-means nodes.

- Feature matrix: copied from qc_good__learncat_good_moderate (identical input, not recomputed).
- Rastermap: same call as rastermap_utils.fit_rastermap (locality / time_lag_window from DEFAULT_CFG,
  n_clusters=100, n_PCs=min(200, ...)) but grid_upsample=10 (rastermap default) instead of 0.
- Cluster labels: model.embedding_clust (each neuron's k-means node, nodes in rastermap-sorted order),
  NOT equal-size slices of isort. isort is returned as (node, upsampled embedding within node), and
  boundaries as node edges, so _cluster_labels_from_isort reproduces embedding_clust exactly (asserted).
  Every downstream step (stats, EDA, matrix summary, metaclusters) therefore uses the 100 node indices.

Output: rastermap_variants/<src>__grid10_nodes/   (--src, default qc_good__learncat_good_moderate)
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from rastermap import Rastermap

import rastermap_psth.rastermap_utils as rmu
import rastermap_psth.run_clustering_new as rcn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
drv = importlib.import_module("000_run_rastermap_variants")
loc_mod = importlib.import_module("005_locality_test")          # copy_data_folder

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default="qc_good__learncat_good_moderate", help="variant whose feature matrix is reused")
_ap.add_argument("--locality", type=float, default=None, help="rastermap locality (default: DEFAULT_CFG, 0.75)")
_args, _ = _ap.parse_known_args()
SRC_VARIANT = _args.src
QC = ["good"] if SRC_VARIANT.startswith("qc_good__") else ["good", "mua"]
LOCALITY = _args.locality
NAME = (f"{SRC_VARIANT}__grid10_nodes" if LOCALITY is None
        else f"{SRC_VARIANT}__loc{LOCALITY:.2f}_grid10_nodes".replace("0.", "0p"))
if LOCALITY is not None:
    rmu.DEFAULT_CFG["locality"] = LOCALITY          # fit_rastermap reads locality from DEFAULT_CFG
GRID_UPSAMPLE = 10
CAPTURE = []


def fit_rastermap_nodes(X, n_clusters):
    n_pcs = min(200, X.shape[0] - 1, X.shape[1] - 1)
    model = Rastermap(n_clusters=n_clusters, n_PCs=n_pcs,
                      locality=rmu.DEFAULT_CFG["locality"],
                      time_lag_window=rmu.DEFAULT_CFG["time_lag_window"],
                      grid_upsample=GRID_UPSAMPLE, verbose=False).fit(X)
    node = np.asarray(model.embedding_clust).astype(int)
    emb = np.asarray(model.embedding)[:, 0]
    isort = np.lexsort((emb, node))                     # primary: node (sorted order); secondary: position
    sizes = np.bincount(node, minlength=n_clusters)
    bounds = np.cumsum(sizes)[:-1].astype(int)
    CAPTURE.append(dict(node=node, embedding=emb, isort_model=np.asarray(model.isort), sizes=sizes))
    return isort, bounds


rcn.fit_rastermap = fit_rastermap_nodes

if __name__ == "__main__":
    tables = drv.load_tables()
    cfg = drv.load_cfg(drv.CONFIG_PATH, unit_quality_label=QC)
    root = drv.VARIANTS_ROOT / NAME
    data_folder = loc_mod.copy_data_folder(drv.VARIANTS_ROOT / SRC_VARIANT, root)
    print(f"=== {NAME}: grid_upsample={GRID_UPSAMPLE}, locality={rmu.DEFAULT_CFG['locality']}, "
          f"time_lag_window={rmu.DEFAULT_CFG['time_lag_window']}, labels = k-means nodes ===", flush=True)
    rres = rcn.run_rastermap(data_folder=data_folder, config_path=drv.CONFIG_PATH, unit_quality_label=QC)
    md = pathlib.Path(rres["method_dir"])
    cap = CAPTURE[-1]
    d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    labels = d["cluster_labels"].astype(int)
    assert np.array_equal(labels, cap["node"]), "cluster labels differ from rastermap k-means nodes"
    print(f"  labels == embedding_clust for all {len(labels)} neurons; node sizes min/median/max "
          f"{cap['sizes'].min()}/{int(np.median(cap['sizes']))}/{cap['sizes'].max()}, empty nodes "
          f"{int((cap['sizes'] == 0).sum())}", flush=True)
    # how contiguous are nodes along the model's own continuous order (upsampled isort)?
    pos = np.empty(len(labels), int); pos[cap["isort_model"]] = np.arange(len(labels))
    spread = pd.Series(pos).groupby(labels).agg(lambda x: x.max() - x.min() + 1) / np.bincount(labels)[np.unique(labels)]
    pd.DataFrame(dict(unit_ids=d["unit_ids"], node_label=cap["node"], embedding=cap["embedding"],
                      position_in_model_isort=pos)).to_csv(md / "rastermap_node_labels_cv.csv", index=False)
    drv.fig_flow_umap_cv(md, cfg)
    drv.run_stats_only(md, cfg, tables["unit_table"], tables["trial_table"])
    st = drv.fig_cluster_mean_matrix_with_fmk(md, cfg, NAME)
    summ = dict(variant=NAME, n_neurons=int(len(labels)), n_nodes=int((cap["sizes"] > 0).sum()),
                node_size_min=int(cap["sizes"].min()), node_size_median=float(np.median(cap["sizes"])),
                node_size_max=int(cap["sizes"].max()),
                node_span_ratio_in_model_isort_median=float(spread.median()),
                n_sig_mwu=int((st.p_mannwhitney_fdr < 0.05).sum()), n_sig_welch=int((st.p_welch_t_fdr < 0.05).sum()),
                permanova=(md / "stats" / "permanova_summary.txt").read_text().splitlines()[1:3])
    json.dump(summ, open(root / "grid10_nodes_summary.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)
    print("ALL DONE", flush=True)
