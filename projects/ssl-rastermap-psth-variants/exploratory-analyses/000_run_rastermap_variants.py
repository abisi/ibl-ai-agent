"""Run Axel Bisi's rastermap_psth clustering pipeline (unit_spikes_analysis) for 4 variants:

    unit quality      : good           | good + mua
    learning_category : all categories | good + moderate (learners)

Loading follows unit_spikes_analysis.py (day 0 / 'learning', KS4 NWB, exclude==0,
exclude_ephys==0, R+/R- only, recording==1, AB068/AB077 excluded, drift-shift QC merge
-> quality_label, Allen labels + keep_shared_areas, anatomy merges, spontaneous licks,
waveform types). Each variant then runs, unmodified:
    rastermap_psth.build_feature_matrix.run_build_feature_matrix
    rastermap_psth.run_clustering_new.run_rastermap      (CV: fig5_population_matrix_cv,
                                                          fig_flow_umap_cv, ...)
    rastermap_psth.rastermap_utils.run_stats_only        (F_mk stats)
and adds one new figure (cluster-averaged CV population matrix | F_mk stats).

Folder layout (one root per variant, so data + clustering never collide):
    combined_results_ks4/rastermap_variants/qc_<good|good_mua>__learncat_<all|good_moderate>/
        rastermap_clustering/<period>/<modality>/<merge>_<bl>_<norm>_<qc>/   (feature matrix)
        rastermap_clustering/<period>/<modality>/clustering/n100_.../rastermap/ (clustering, stats)
        variant_provenance.json

Run on haas from ~/code/unit_spikes_analysis:
    PYTHONPATH=~/code/NWB_reader:. .venv/bin/python <this file> [--test] [--variants ...]
"""
import argparse
import json
import os
import pathlib
import pickle
import socket
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind

import ephys_utilities
import ephys_utilities.helpers.load_helpers
from ephys_utilities.helpers import data_utils, load_helpers
from ephys_utilities.neural_utils import unit_metrics_utils
import ephys_utilities.allen_utils.allen_utils as allen_utils
import ephys_utilities.plotting_utils.plotting_utils as plotting_utils

import rastermap_psth.build_feature_matrix as bfm
import rastermap_psth.rastermap_utils as rmu
from rastermap_psth.run_clustering_new import run_rastermap
from rastermap_psth.rastermap_utils import run_stats_only, load_cfg

# ── paths / constants (as in unit_spikes_analysis.py, haas branch) ───────────
assert "haas" in socket.gethostname(), "run on haas"
N_WORKERS = 110
ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis")
NWB_DIR = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
INFO_PATH = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info")
OUTPUT_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
CONFIG_PATH = ROOT / "rastermap_psth" / "config.yaml"
VARIANTS_ROOT = OUTPUT_PATH / "rastermap_variants"
# NAS, not haas local disk: the root fs is nearly full (a 10.5 GB cache in ~/tmp filled it on 2026-09-25)
CACHE = VARIANTS_ROOT / "_cache" / "tables_day0.pkl"
DAY_TO_ANALYZE = "learning"
EXCLUDED_MICE = ["AB068", "AB077"]

VARIANTS = {
    "qc_good__learncat_all":               dict(qc=["good"],        learncat="all"),
    "qc_good__learncat_good_moderate":     dict(qc=["good"],        learncat="good_moderate"),
    "qc_good_mua__learncat_all":           dict(qc=["good", "mua"], learncat="all"),
    "qc_good_mua__learncat_good_moderate": dict(qc=["good", "mua"], learncat="good_moderate"),
    # 2026-09-26: no firing-rate filters (session pre-filter global_fr_hz and per-condition
    # fr_threshold_hz disabled); odd/even reliability filter (min_odd_even_r) unchanged
    "qc_good__learncat_good_moderate__noFR": dict(qc=["good"], learncat="good_moderate",
                                                  cfg=dict(global_fr_hz=-1.0, fr_threshold_hz=0.0)),
}


def load_tables(test=False):
    """unit_spikes_analysis.py load block, verbatim logic."""
    if CACHE.exists() and not test:
        print(f"Loading cached tables {CACHE}", flush=True)
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    all_nwb_names = os.listdir(NWB_DIR)
    all_nwb_mice = [n.split("_")[0] for n in all_nwb_names]
    mouse_info_df = pd.read_excel(INFO_PATH / "joint_mouse_reference_weight.xlsx")
    mouse_info_df.rename(columns={"mouse_name": "mouse_id"}, inplace=True)
    mouse_info_df = mouse_info_df[(mouse_info_df["exclude"] == 0) &
                                  (mouse_info_df["exclude_ephys"] == 0) &
                                  (mouse_info_df["reward_group"].isin(["R+", "R-"])) &
                                  (mouse_info_df["recording"] == 1)]
    subject_ids = [m for m in mouse_info_df["mouse_id"].unique() if any(m in n for n in all_nwb_mice)]
    subject_ids = [s for s in subject_ids if s not in EXCLUDED_MICE]

    nwb_list = [str(NWB_DIR / n) for n in all_nwb_names if n.startswith("AB")]
    nwb_list += [str(NWB_DIR / n) for n in all_nwb_names if n.startswith("MH")]
    nwb_list = [nwb for nwb in nwb_list if any(s in nwb for s in subject_ids)]
    if test:  # a few mice from each cohort, learners and non-learners, with passive data
        keep = ["AB131", "AB132", "AB134", "AB161", "AB164", "MH022", "MH030", "MH036"]
        nwb_list = [n for n in nwb_list if any(m in n for m in keep)]
    print(f"{len(subject_ids)} subjects, {len(nwb_list)} NWB files", flush=True)

    trial_table, unit_table, nwb_neural_files = data_utils.combine_ephys_nwb(
        nwb_list, day_to_analyze=DAY_TO_ANALYZE, max_workers=N_WORKERS)

    dredge_df = load_helpers.load_motion_dredge_shift_test_results(
        nwb_neural_files, day_to_analyze=DAY_TO_ANALYZE, max_workers=N_WORKERS)
    dredge_df = dredge_df[["mouse_id", "session_id", "cluster_id", "electrode_group", "p_conservative", "r"]].rename(
        columns={"p_conservative": "drift_shift_test_pval"})
    dredge_df["drift_abs_r"] = dredge_df["r"].abs()
    unit_table["cluster_id"] = unit_table["cluster_id"].astype(str)
    dredge_df["cluster_id"] = dredge_df["cluster_id"].astype(str)
    unit_table = unit_table.merge(dredge_df, on=["mouse_id", "session_id", "cluster_id", "electrode_group"],
                                  how="left", validate="one_to_one")
    unit_table = unit_metrics_utils.compute_presence_coverage_metrics(unit_table)
    unit_table = unit_metrics_utils.classify_units_quality(unit_table, label_col="quality_label")
    unit_table = allen_utils.process_allen_labels(unit_table, split_merge_areas=True)
    unit_table, _ = data_utils.keep_shared_areas(unit_table, nomenclature="area_acronym_custom",
                                                 n_min_units=5, n_min_mice=3)
    unit_table = allen_utils.merge_liu_avg_ipsi(unit_table)
    unit_table = allen_utils.merge_hierarchy_columns_from_gao(unit_table)
    unit_table = allen_utils.merge_hierarchy_from_harris(unit_table)
    lick_times_df = load_helpers.load_spontaneous_reward_lick_times(
        nwb_neural_files, day_to_analyze=DAY_TO_ANALYZE, max_workers=N_WORKERS, load_summary=False)
    wf_df = ephys_utilities.helpers.load_helpers.load_wf_analysis_data(nwb_files=nwb_neural_files, experimenter="AB")
    wf_df = wf_df[wf_df.mouse_id.isin(unit_table.mouse_id.unique())]
    unit_table = unit_table.merge(wf_df, on=["mouse_id", "session_id", "electrode_group", "cluster_id"], how="left")

    out = dict(unit_table=unit_table, trial_table=trial_table, lick_df=lick_times_df,
               nwb_files=nwb_neural_files, subject_ids=subject_ids)
    if not test:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".pkl.partial")  # atomic: never leave a truncated cache
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CACHE)
    return out


def mouse_info_for(learncat, out_dir):
    """build_feature_matrix hardcodes learning_category in {good, moderate}. For the
    all-categories variant, point its MOUSE_INFO at a copy where every mouse passes that
    filter (learning_category set to 'good'); the original column is kept as
    learning_category_original. For good_moderate, use the original sheet."""
    orig = rmu.MOUSE_INFO
    if learncat == "good_moderate":
        return orig
    df = pd.read_excel(orig)
    df["learning_category_original"] = df["learning_category"]
    df["learning_category"] = "good"
    path = out_dir / "mouse_info_all_learning_categories.xlsx"
    df.to_excel(path, index=False)
    return str(path)


def set_conditions(cfg):
    """rastermap_utils keeps condition layout in module globals (set by run_rastermap);
    set them from cfg so the figure helpers also work standalone."""
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS,
     rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def fig_cluster_mean_matrix_with_fmk(method_dir, cfg, variant_name):
    """Cluster-averaged CV (even-trial) population matrix, side by side with per-cluster
    F_mk (per-mouse fractional representation) R+ vs R- stats (Mann-Whitney U + Welch t,
    BH-FDR each). Rows = rastermap clusters in rastermap (odd-trial fit) order."""
    set_conditions(cfg)
    d = np.load(method_dir / "rastermap_results_cv.npz", allow_pickle=True)
    X, labels = d["X"], d["cluster_labels"].astype(int)
    n_bins_list = list(d["n_bins_list"])
    K = labels.max() + 1
    # _cluster_labels_from_isort numbers clusters as consecutive blocks along isort,
    # so label order 0..K-1 is already the rastermap (odd-trial fit) order.
    order = np.arange(K)
    M = np.vstack([np.nanmean(X[labels == k], axis=0) if (labels == k).any()
                   else np.full(X.shape[1], np.nan) for k in order])
    n_per_cluster = np.array([(labels == k).sum() for k in order])

    fz = np.load(method_dir / "stats" / "f_matrix.npz", allow_pickle=True)
    F, groups, mice = fz["f_matrix"], fz["reward_groups"].astype(str), fz["mouse_ids"].astype(str)
    rp, rm = groups == "R+", groups == "R-"
    rows = []
    for k in order:
        a, b = F[rp, k], F[rm, k]
        p_mw = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) > 1 and len(b) > 1 else np.nan
        p_t = ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else np.nan
        rows.append(dict(cluster=k + 1, n_neurons=int((labels == k).sum()),
                         mean_f_rplus=a.mean(), sem_f_rplus=a.std(ddof=1) / np.sqrt(len(a)),
                         mean_f_rminus=b.mean(), sem_f_rminus=b.std(ddof=1) / np.sqrt(len(b)),
                         n_mice_rplus=int(rp.sum()), n_mice_rminus=int(rm.sum()),
                         p_mannwhitney=p_mw, p_welch_t=np.nan if p_t is None else p_t))
    st = pd.DataFrame(rows)
    st["p_mannwhitney_fdr"] = bh(st.p_mannwhitney.fillna(1))
    st["p_welch_t_fdr"] = bh(st.p_welch_t.fillna(1))
    st.insert(0, "row_in_figure", np.arange(len(st)))
    st.insert(0, "variant", variant_name)
    st.to_csv(method_dir / "stats" / "cluster_mean_matrix_fmk_stats.csv", index=False)

    vmax = np.nanpercentile(np.abs(M), cfg["vmax_pct"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 12), dpi=300,
                             gridspec_kw={"width_ratios": [10, 0.6, 3, 1.6], "wspace": 0.08})
    im = rmu._draw_matrix(axes[0], M, n_bins_list, [], vmax, cfg,
                          f"Cluster-mean PSTH, even trials (CV)  —  {K} clusters, n={len(X)} neurons")
    cax = axes[0].inset_axes([0.0, -0.09, 0.25, 0.012])
    fig.colorbar(im, cax=cax, orientation="horizontal",
                 label="Firing rate (z-score)" if cfg["normalize"] == "zscore" else "Firing rate (baseline-norm.)")
    axes[0].set_ylabel("Rastermap cluster (odd-trial fit order)")
    axes[0].set_yticks(np.arange(K) + 0.5)
    axes[0].set_yticklabels([str(k + 1) for k in order], fontsize=4)

    y = np.arange(K) + 0.5
    axes[1].barh(y, n_per_cluster, height=0.8, color="0.6")
    axes[1].set_ylim(K, 0); axes[1].set_yticks([]); axes[1].set_title("n neurons", fontsize=8)
    axes[1].tick_params(labelsize=6)

    c_rp, c_rm = plotting_utils.GROUP_COLORS["rplus"], plotting_utils.GROUP_COLORS["rminus"]
    ax = axes[2]
    ax.errorbar(st.mean_f_rplus, y - 0.15, xerr=st.sem_f_rplus, fmt="o", ms=2.5, lw=0.8, color=c_rp,
                label=f"R+ (n={int(rp.sum())} mice)")
    ax.errorbar(st.mean_f_rminus, y + 0.15, xerr=st.sem_f_rminus, fmt="o", ms=2.5, lw=0.8, color=c_rm,
                label=f"R− (n={int(rm.sum())} mice)")
    ax.set_ylim(K, 0); ax.set_yticks([])
    ax.set_xlabel(r"$F_{m,k}$ (fraction of mouse's neurons), mean ± SEM")
    ax.set_title("Fractional representation per cluster", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = axes[3]
    ax.plot(-np.log10(st.p_mannwhitney_fdr), y, "o", ms=2.5, color="k", label="Mann–Whitney U")
    ax.plot(-np.log10(st.p_welch_t_fdr), y, "s", ms=2.5, mfc="none", color="tab:blue", label="Welch t")
    ax.axvline(-np.log10(0.05), color="0.5", ls="--", lw=0.8)
    ax.set_ylim(K, 0); ax.set_yticks([])
    ax.set_xlabel(r"$-\log_{10}$ p (BH-FDR)")
    ax.set_title("R+ vs R−", fontsize=9)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    n_mw, n_t = int((st.p_mannwhitney_fdr < 0.05).sum()), int((st.p_welch_t_fdr < 0.05).sum())
    fig.suptitle(f"{variant_name}: cluster-averaged CV population matrix | F_mk R+ vs R− "
                 f"(FDR<0.05: MW {n_mw}/{K}, Welch {n_t}/{K})", fontsize=11)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(method_dir / f"fig5_cluster_mean_population_matrix_cv_fmk_stats.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved cluster-mean matrix + F_mk figure; FDR<0.05 MW {n_mw}/{K}, Welch {n_t}/{K}", flush=True)
    return st


def fig_flow_umap_cv(method_dir, cfg):
    """run_clustering_new.fig8_flow_umap builds fig_flow_umap_<prefix> but never saves it
    (no _save call before its return). Re-call it with the exact arguments run_rastermap
    uses in CV mode (X_even, cluster_labels_cv, edge_trim=1, prefix='cv') and save the
    figure it leaves open."""
    from rastermap_psth.run_clustering_new import fig8_flow_umap
    set_conditions(cfg)
    d = np.load(method_dir / "rastermap_results_cv.npz", allow_pickle=True)
    t_ctrs = [d[f"t_ctr_{ci}"] for ci in range(int(d["n_conds"]))]
    plt.close("all")
    fig8_flow_umap(d["X"], list(d["n_bins_list"]), t_ctrs, rmu.COND_LABELS, rmu.COND_COLORS, cfg,
                   method_dir, cluster_labels=d["cluster_labels"], edge_trim=1, prefix="cv")
    fig = plt.gcf()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(method_dir / f"fig_flow_umap_cv.{ext}", bbox_inches="tight", dpi=300)
    plt.close("all")
    print("  saved fig_flow_umap_cv", flush=True)


def run_variant(name, spec, tables, root):
    t0 = time.time()
    vroot = root / name
    vroot.mkdir(parents=True, exist_ok=True)
    overrides = dict(unit_quality_label=spec["qc"], **spec.get("cfg", {}))
    bfm.MOUSE_INFO = mouse_info_for(spec["learncat"], vroot)
    print(f"\n=== {name}: quality_label in {spec['qc']}, learning_category={spec['learncat']} "
          f"(MOUSE_INFO={bfm.MOUSE_INFO}) ===", flush=True)

    res = bfm.run_build_feature_matrix(tables["unit_table"].copy(), tables["trial_table"].copy(),
                                       lick_df=tables["lick_df"].copy(), config_path=CONFIG_PATH,
                                       out_root=vroot, **overrides)
    rres = run_rastermap(data_folder=res["data_folder"], config_path=CONFIG_PATH, **overrides)
    cfg = load_cfg(CONFIG_PATH, **overrides)
    fig_flow_umap_cv(rres["method_dir"], cfg)
    run_stats_only(rres["method_dir"], cfg, tables["unit_table"], tables["trial_table"])
    st = fig_cluster_mean_matrix_with_fmk(rres["method_dir"], cfg, name)

    meta = pd.read_csv(pathlib.Path(res["data_folder"]) / "neuron_metadata.csv")
    prov = dict(
        variant=name, unit_quality_label=spec["qc"], learning_category=spec["learncat"],
        created=datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
        config_path=str(CONFIG_PATH), config_overrides=overrides, day_to_analyze=DAY_TO_ANALYZE,
        mouse_info_used=str(bfm.MOUSE_INFO), data_folder=str(res["data_folder"]),
        clustering_folder=str(rres["out_folder"]), method_dir=str(rres["method_dir"]),
        n_neurons=int(len(meta)), n_mice=int(meta.mouse_id.nunique()),
        n_sessions=int(meta.session_id.nunique()),
        mice_by_reward_group={g: sorted(x.mouse_id.unique().tolist()) for g, x in meta.groupby("reward_group")},
        session_ids=sorted(meta.session_id.unique().tolist()),
        n_clusters_fdr_sig_mannwhitney=int((st.p_mannwhitney_fdr < 0.05).sum()),
        n_clusters_fdr_sig_welch=int((st.p_welch_t_fdr < 0.05).sum()),
        runtime_min=round((time.time() - t0) / 60, 1),
        loading="unit_spikes_analysis.py load block (see 000_run_rastermap_variants.py docstring)",
    )
    with open(vroot / "variant_provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"=== {name} done in {prov['runtime_min']} min: {prov['n_neurons']} neurons, "
          f"{prov['n_mice']} mice ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="8 mice, output under rastermap_variants/_test")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    root = VARIANTS_ROOT / "_test" if args.test else VARIANTS_ROOT
    tables = load_tables(test=args.test)
    for name in args.variants:
        try:
            run_variant(name, VARIANTS[name], tables, root)
        except Exception:
            import traceback
            print(f"=== {name} FAILED ===\n{traceback.format_exc()}", flush=True)
    print("ALL DONE", flush=True)
