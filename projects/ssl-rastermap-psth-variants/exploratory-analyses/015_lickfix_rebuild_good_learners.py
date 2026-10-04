"""Rebuild a learners feature matrix (--qc good | good_mua, --locality) with the corrected trial lick time, rerun rastermap
(grid_upsample=10, clusters = k-means nodes, locality = DEFAULT_CFG as in 006) and compare with the
uncorrected run qc_good__learncat_good_moderate__grid10_nodes.

Correction (rastermap_utils.precompute_event_map, cfg correct_lick_time=True): lick_time-aligned trial conditions
('Whisker hit (lick)', 'Auditory hit (lick)') use start_time + (lick_time - response_window_start_time).
NWB lick_time = response_window_start + reaction_time (NWB_converter behavior_converter_misc.py:527), while
behavior_control main_control.m computes reaction_time from stimulus onset -> NWB lick_time is late by the
per-trial artifact window. 'Spont. lick' (spontaneous-lick CSV, piezo clock) is unchanged.

Outputs -> rastermap_variants/qc_good__learncat_good_moderate__grid10_nodes__lickfix/ (+ lickfix_comparison/)
"""
import importlib
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--qc", default="good", choices=["good", "good_mua"])
_ap.add_argument("--locality", type=float, default=None, help="default: DEFAULT_CFG (0.75), as the good-only run")
_ap.add_argument("--learncat", default="good_moderate", choices=["good_moderate", "all"])
_ap.add_argument("--suffix", default="__lickfix", help="variant-name suffix, e.g. __lickfix_longwin for the longer config.yaml windows")
_ap.add_argument("--period", default=None, choices=["active", "passive", "passive_active"],
                 help="override config.yaml period (e.g. 'active' = no passive conditions; neurons then no longer need "
                      "passive trials, so sessions without passive blocks are included)")
_args = _ap.parse_args()
SRC = f"qc_{_args.qc}__learncat_{_args.learncat}"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "--src", SRC] + ([] if _args.locality is None else ["--locality", str(_args.locality)])
g10 = importlib.import_module("006_grid10_node_clusters")        # installs the k-means-node fit_rastermap wrapper
drv = g10.drv
import rastermap_psth.rastermap_utils as rmu                     # noqa: E402
import rastermap_psth.run_clustering_new as rcn                  # noqa: E402
import rastermap_psth.build_feature_matrix as bfm                # noqa: E402
from rastermap_psth.population_matrix_summary import run_matrix_summary  # noqa: E402

OLD = g10.NAME                                                   # e.g. qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes
NEW = f"{OLD}{_args.suffix}"
QC = g10.QC
LEARNCAT = _args.learncat


def method_dir(variant):
    return sorted((drv.VARIANTS_ROOT / variant).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]


def feature_npz(variant):
    return sorted((drv.VARIANTS_ROOT / variant).glob("rastermap_clustering/*/*/*/feature_matrix.npz"))[0]


def compare(md_old, md_new, out):
    out.mkdir(exist_ok=True)
    o, n = (np.load(m / "rastermap_results_cv.npz", allow_pickle=True) for m in (md_old, md_new))
    uo, un = o["unit_ids"], n["unit_ids"]
    common, io, inn = np.intersect1d(uo, un, return_indices=True)
    ari = adjusted_rand_score(o["cluster_labels"][io], n["cluster_labels"][inn])
    perm = {}
    for tag, m in (("old", md_old), ("new", md_new)):
        txt = (m / "stats" / "permanova_summary.txt").read_text()
        perm[tag] = dict(F=float(txt.split("F-statistic :")[1].split()[0]), p=float(txt.split("p-value     :")[1].split()[0]))
        st = pd.read_csv(m / "stats" / "cluster_mean_matrix_fmk_stats.csv")
        perm[tag]["n_sig_mwu"] = int((st.p_mannwhitney_fdr < 0.05).sum())
    # population-mean lick-aligned traces from the feature matrices (same neurons)
    fo, fn = (dict(np.load(feature_npz(v), allow_pickle=True)) for v in (OLD, NEW))
    labels = list(rmu.COND_LABELS)
    lick_conds = [c for c in labels if "lick" in c.lower()]
    fig, axes = plt.subplots(1, len(lick_conds), figsize=(4 * len(lick_conds), 3), dpi=200, squeeze=False)
    peaks = {}
    for ax, cond in zip(axes[0], lick_conds):
        ci = labels.index(cond)
        for tag, f, c in (("uncorrected lick_time", fo, "0.6"), ("corrected lick", fn, "k")):
            nb = f["n_bins_list"].astype(int); off = np.r_[0, np.cumsum(nb)]
            t = f[f"t_ctr_{ci}"]
            y = np.nanmean(f["X"][:, off[ci]:off[ci + 1]], 0)
            ax.plot(t * 1000, y, color=c, lw=1.2, label=f"{tag} (peak {t[np.argmax(y)] * 1000:+.0f} ms)")
            peaks[f"{cond} | {tag}"] = float(t[np.argmax(y)] * 1000)
        ax.axvline(0, color="k", lw=0.5, ls=":"); ax.set_title(cond, fontsize=9); ax.set_xlabel("ms from lick")
        ax.legend(fontsize=6, frameon=False)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0, 0].set_ylabel("population mean (feature matrix units)")
    fig.suptitle(f"{OLD} vs {NEW}: lick-aligned conditions", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "lick_conditions_population_mean.png"); plt.close(fig)
    nper = {}
    for tag, m in (("old", md_old), ("new", md_new)):
        c = pd.read_csv(m / "neuron_cluster_labels_cv.csv").groupby("mouse_id").size()
        nper[tag] = dict(min=int(c.min()), n_mice_lt20=int((c < 20).sum()), smallest={k: int(v) for k, v in c.nsmallest(5).items()})
    res = dict(old=OLD, new=NEW, n_neurons_old=int(len(uo)), n_neurons_new=int(len(un)), n_common=int(len(common)),
               ari_cluster_labels_common=float(ari), permanova=perm, population_peak_ms=peaks,
               neurons_per_mouse=nper)
    json.dump(res, open(out / "lickfix_comparison.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    t0 = time.time()
    tables = drv.load_tables()
    vroot = drv.VARIANTS_ROOT / NEW
    vroot.mkdir(parents=True, exist_ok=True)
    overrides = dict(unit_quality_label=QC, correct_lick_time=True)
    if _args.period:
        overrides["period"] = _args.period
    bfm.MOUSE_INFO = drv.mouse_info_for(LEARNCAT, vroot)
    print(f"=== {NEW}: build feature matrix with {overrides}, locality={rmu.DEFAULT_CFG['locality']} ===", flush=True)
    res = bfm.run_build_feature_matrix(tables["unit_table"].copy(), tables["trial_table"].copy(),
                                       lick_df=tables["lick_df"].copy(), config_path=drv.CONFIG_PATH,
                                       out_root=vroot, **overrides)
    g10.CAPTURE.clear()
    rres = rcn.run_rastermap(data_folder=res["data_folder"], config_path=drv.CONFIG_PATH, **overrides)
    md = pathlib.Path(rres["method_dir"])
    d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    assert np.array_equal(d["cluster_labels"].astype(int), g10.CAPTURE[-1]["node"]), "labels != k-means nodes"
    cfg = drv.load_cfg(drv.CONFIG_PATH, **overrides)
    drv.set_conditions(cfg)
    drv.fig_flow_umap_cv(md, cfg)
    drv.run_stats_only(md, cfg, tables["unit_table"], tables["trial_table"])
    drv.fig_cluster_mean_matrix_with_fmk(md, cfg, NEW)
    run_matrix_summary(md, drv.CONFIG_PATH, overrides,
                       title_prefix=f"{NEW}: ")
    try:
        cmp_ = compare(method_dir(OLD), md, vroot / "lickfix_comparison")
    except IndexError:                                          # no matching uncorrected run to compare with
        cmp_ = dict(comparison="skipped: no run named " + OLD)
    json.dump(dict(variant=NEW, config_overrides=overrides, locality=rmu.DEFAULT_CFG["locality"], grid_upsample=10,
                   learning_category=LEARNCAT, mouse_info_used=str(bfm.MOUSE_INFO), method_dir=str(md),
                   compared_to=OLD, runtime_min=round((time.time() - t0) / 60, 1),
                   psth_windows_s={k: cfg.get(k) for k in ("t_pre_passive", "t_post_passive", "t_pre_active",
                                                           "t_post_active", "t_pre_lick", "t_post_lick")}),
              open(vroot / "variant_provenance.json", "w"), indent=2)
    print(json.dumps(cmp_, indent=1), flush=True)
    print("ALL DONE", flush=True)
