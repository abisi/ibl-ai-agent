"""Rastermap variant: learners, good+mua, WITH passive conditions, same parameters as the long-window lick-fixed run,
but excluding the whisker-hit time bins that contain the reward.

Source feature matrix: qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin (same neurons, same
PSTHs; bins are SLICED from the existing condition columns, nothing recomputed; as 009_trimmed_whisker_hit).
Cuts (keep bins whose END is <= cut; the reward-free cuts used for the reward-free MMD, 016):
    "Whisker hit"        (stimulus-aligned, -0.1..0.4 s) -> keep bins ending <= +0.20 s
    "Whisker hit (lick)" (lick-aligned,     -0.3..0.2 s) -> keep bins ending <= +0.05 s after the corrected first lick
Every other condition (passive whisker/auditory, whisker miss, auditory hit, spontaneous lick, auditory hit (lick)) is
unchanged. Settings as the source run: locality 0.25, grid_upsample 10, clusters = rastermap k-means nodes,
contiguous metaclusters (default).
Outputs -> rastermap_variants/<source>__noWhiskerReward/: rastermap (CV), flow/UMAP, stats, cluster-mean | F_mk figure,
matrix summary (incl. publication layout, metacluster summary), variant_provenance.json.
"""
import importlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]]
g10 = importlib.import_module("006_grid10_node_clusters")    # installs the k-means-node fit_rastermap wrapper
drv = g10.drv
import rastermap_psth.rastermap_utils as rmu                  # noqa: E402
import rastermap_psth.run_clustering_new as rcn               # noqa: E402
from rastermap_psth.population_matrix_summary import run_matrix_summary  # noqa: E402

SRC = "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin"
NAME = f"{SRC}__noWhiskerReward"
QC = ["good", "mua"]
LOCALITY = 0.25
CUTS = {"Whisker hit": 0.20, "Whisker hit (lick)": 0.05}
OVERRIDES = dict(unit_quality_label=QC, correct_lick_time=True)


def sliced_data_folder(src_root, dst_root, cuts, cond_labels):
    src = sorted(src_root.glob("rastermap_clustering/*/*/*/feature_matrix.npz"))[0].parent
    dst = dst_root / src.relative_to(src_root)
    dst.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(src / "feature_matrix.npz", allow_pickle=True))
    nb = list(d["n_bins_list"].astype(int)); n_c = int(d["n_conds"])
    assert n_c == len(cond_labels), (n_c, cond_labels)
    off = np.r_[0, np.cumsum(nb)]
    keep_cols, new_nb, report = [], [], {}
    for ci in range(n_c):
        t = d[f"t_ctr_{ci}"]; lab = cond_labels[ci]
        bw = float(np.median(np.diff(t)))
        keep = (t + bw / 2 <= cuts[lab] + 1e-9) if lab in cuts else np.ones(len(t), bool)
        keep_cols.append(np.arange(off[ci], off[ci + 1])[keep])
        d[f"t_ctr_{ci}"] = t[keep]; new_nb.append(int(keep.sum()))
        if lab in cuts:
            report[lab] = dict(bin_s=bw, t_ctr_min=float(t.min()), t_ctr_max_before=float(t.max()),
                               t_ctr_max_after=float(t[keep].max()), bins_before=int(len(t)), bins_after=int(keep.sum()))
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
    t0 = time.time()
    rmu.DEFAULT_CFG["locality"] = LOCALITY
    tables = drv.load_tables()
    cfg = drv.load_cfg(drv.CONFIG_PATH, **OVERRIDES)
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
    root = drv.VARIANTS_ROOT / NAME
    data_folder, rep = sliced_data_folder(drv.VARIANTS_ROOT / SRC, root, CUTS, list(rmu.COND_LABELS))
    print(f"=== {NAME}: {json.dumps(rep)} ===", flush=True)
    g10.CAPTURE.clear()
    rres = rcn.run_rastermap(data_folder=data_folder, config_path=drv.CONFIG_PATH, **OVERRIDES)
    md = pathlib.Path(rres["method_dir"])
    d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    assert np.array_equal(d["cluster_labels"].astype(int), g10.CAPTURE[-1]["node"]), "labels != k-means nodes"
    drv.set_conditions(cfg)
    drv.fig_flow_umap_cv(md, cfg)
    drv.run_stats_only(md, cfg, tables["unit_table"], tables["trial_table"])
    drv.fig_cluster_mean_matrix_with_fmk(md, cfg, NAME)
    run_matrix_summary(md, drv.CONFIG_PATH, OVERRIDES, title_prefix=f"{NAME}: ")
    json.dump(dict(variant=NAME, source_variant=SRC, config_overrides=OVERRIDES, locality=LOCALITY, grid_upsample=10,
                   learning_category="good_moderate", reward_cuts_s_bin_end=CUTS, slicing_report=rep,
                   n_neurons=int(len(d["unit_ids"])), method_dir=str(md), runtime_min=round((time.time() - t0) / 60, 1)),
              open(root / "variant_provenance.json", "w"), indent=2)
    print("ALL DONE", flush=True)
