"""050 -- Rastermap variants v2 (user 2026-11-23): one main configuration, one-factor robustness variants, all figures,
one organised folder per variant. Replaces the 000 -> 006 -> 015 chain (and 042 for the reward-bin control).

Main (every setting explicit here, so config.yaml defaults do not matter):
    units        STABLE good + MUA (stable = coverage >= 0.9, presence >= 0.5, drift-independent; build_feature_matrix
                 stable_unit_mask). Corrected lick times always (no "lickfix" in names).
    mice         learners (reference-sheet learning_category good / moderate), whisker day 0, one session per mouse
    PSTHs        passive -0.1..0.2, active -0.1..0.4, lick -0.3..0.2 s; 10-ms bins, 5-ms stride, sigma 2 ms; baseline-
                 normalised, per-trial baseline removed; FR filters global > 0.01 Hz, per condition > 0.1 Hz;
                 >= 3 odd trials per condition; odd/even r >= 0.1
    rastermap    CV (fit odd, evaluate even), 100 clusters, locality 0.25, time_lag_window 20, grid_upsample 10,
                 clusters = rastermap k-means nodes
    metaclusters contiguous agglomerative (average linkage, 1 - r, chain connectivity), cut 0.40
                 (population_matrix_summary.metaclustering default)

Variants (one change each):  main | units_all (stable filter off) | units_good | loc0p00 | loc0p75 | noFR | mice_all |
    noWhiskerReward (whisker-hit bins containing the reward removed, sliced from main's feature matrix as 042).
    loc0p00 / loc0p75 reuse main's feature matrix (same input, refit only).

Phases (run on haas from ~/code/unit_spikes_analysis, PYTHONPATH=~/code/NWB_reader:.):
    tables                      fresh day-0 unit / trial tables (000 load_tables logic) -> v2/_tables/ (+ slim copy
                                without spike times for the stats / figure steps)
    features --variants ...     feature matrices (one process, tables loaded once)
    fit --variants V            rastermap fit + pipeline figures (flow/UMAP, stats, cluster-mean | F_mk, matrix summary)
    figures --variants V        metacluster F_mk (003), metacluster stats (004), ROC EDA (001), significant-cluster
                                figures (023), metacluster description of all metaclusters (045)
    organize --variants V       <variant>/figures/<topic>/ copies + figures/index.md
    compare                     v2/_comparison/: AMI vs main (clusters, metaclusters), metaclusters matched to main by
                                mean-PSTH r (Hungarian), summary figure
Layout: v2/<variant>/{variant_provenance.json, rastermap_clustering/ (pipeline tree, unchanged), figures/}
"""
import argparse
import importlib
import json
import os
import pathlib
import pickle
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
V2 = RES / "rastermap_variants" / "v2"
TABLES = V2 / "_tables" / "tables_day0.pkl"
TABLES_SLIM = V2 / "_tables" / "tables_day0_slim.pkl"
TEST = "--test" in sys.argv                    # 8-mouse test run (000 load_tables(test=True)) in rastermap_variants/_v2_test
if TEST:
    V2 = RES / "rastermap_variants" / "_v2_test"
    TABLES, TABLES_SLIM = V2 / "_tables" / "tables_day0.pkl", V2 / "_tables" / "tables_day0_slim.pkl"
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
os.environ["RASTERMAP_TABLES_CACHE"] = str(TABLES_SLIM)

MAIN = dict(
    unit_quality_label=["good", "mua"], stable_only=True, learning_categories=["good", "moderate"],
    correct_lick_time=True, period="passive_active", modality="whisker_auditory", merge_passive=False,
    reward_filter="combined", include_lick_conditions=True,
    t_pre_passive=0.1, t_post_passive=0.2, t_pre_active=0.1, t_post_active=0.4, t_pre_lick=0.3, t_post_lick=0.2,
    t_pre_jaw=0.25, t_post_jaw=0.15, t_pre_reward=0.35, t_post_reward=0.05,
    bin_ms=10, stride_ms=5, sigma_ms=2, normalize="baseline", zscore_full=True, baseline_removal=True,
    global_fr_hz=0.01, fr_threshold_hz=0.1, n_min_trial_per_condition=3, min_odd_even_r=0.1,
    cross_validate=True, cv_zscore_independent=True, do_global=False, n_rastermap_clusters=100,
    locality=0.25, time_lag_window=20,
)
GRID_UPSAMPLE = 10
META_CUT = 0.40
REWARD_CUTS = {"Whisker hit": 0.20, "Whisker hit (lick)": 0.05}   # keep bins ending <= cut (s), as 042 / 016

VARIANTS = {   # name: (config changes, feature-matrix source: None = build, "main" = copy, "main:slice" = 042 slicing)
    "main":            ({}, None),
    "units_all":       (dict(stable_only=False), None),
    "units_good":      (dict(unit_quality_label=["good"]), None),
    "noFR":            (dict(global_fr_hz=-1.0, fr_threshold_hz=0.0), None),
    "mice_all":        (dict(learning_categories="all"), None),
    "loc0p00":         (dict(locality=0.0), "main"),
    "loc0p75":         (dict(locality=0.75), "main"),
    "noWhiskerReward": ({}, "main:slice"),
}
LABELS = {"main": "Main", "units_all": "All units (incl. unstable)", "units_good": "Good units only",
          "noFR": "No firing-rate threshold", "mice_all": "All mice (incl. non-learners)", "loc0p00": "Locality 0",
          "loc0p75": "Locality 0.75", "noWhiskerReward": "No whisker-reward bins"}


def cfg_of(name):
    return {**MAIN, **VARIANTS[name][0]}


def _drv():
    return importlib.import_module("000_run_rastermap_variants")


def method_dir(name):
    return sorted((V2 / name).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]


def feature_dir(name):
    return sorted((V2 / name).glob("rastermap_clustering/*/*/*/feature_matrix.npz"))[0].parent


# ── tables ────────────────────────────────────────────────────────────────────
def phase_tables():
    drv = _drv()
    drv.CACHE = TABLES
    TABLES.parent.mkdir(parents=True, exist_ok=True)
    t = drv.load_tables(test=TEST)                         # builds + writes TABLES (atomic) when absent
    ut = t["unit_table"]
    # feature matrix unit_ids = unit_table index, but run_stats_only looks neurons up by the unit_id column: make them
    # identical (the user's convention, raster_utils / xcorr_utils: unit_table['unit_id'] = unit_table.index)
    assert ut.index.is_unique
    if "unit_id" in ut.columns and "unit_id_nwb" not in ut.columns:
        ut = ut.rename(columns={"unit_id": "unit_id_nwb"})
    ut["unit_id"] = ut.index
    t["unit_table"] = ut
    tmp = TABLES.with_suffix(".pkl.partial")
    with open(tmp, "wb") as f:
        pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, TABLES)
    drop = [c for c in ut.columns if "spike_times" in c]
    slim = dict(t, unit_table=ut.drop(columns=drop))
    tmp = TABLES_SLIM.with_suffix(".pkl.partial")
    with open(tmp, "wb") as f:
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, TABLES_SLIM)
    json.dump(dict(created=datetime.now().isoformat(timespec="seconds"), n_units=int(len(ut)),
                   n_mice=int(ut.mouse_id.nunique()), n_sessions=int(ut.session_id.nunique()),
                   slim_dropped_columns=drop, loading="000_run_rastermap_variants.load_tables (fresh build)"),
              open(TABLES.parent / "tables_provenance.json", "w"), indent=2)
    print(f"tables: {len(ut)} units, {ut.mouse_id.nunique()} mice; slim drops {drop}", flush=True)


def load_tables(slim):
    with open(TABLES_SLIM if slim else TABLES, "rb") as f:
        return pickle.load(f)


# ── features ──────────────────────────────────────────────────────────────────
def phase_features(names):
    import rastermap_psth.build_feature_matrix as bfm
    drv = _drv()
    tables = load_tables(slim=False)
    for name in names:
        if VARIANTS[name][1] is not None:
            print(f"skip {name}: feature matrix from {VARIANTS[name][1]}", flush=True)
            continue
        t0 = time.time()
        cfg = cfg_of(name)
        print(f"\n=== features {name}: {VARIANTS[name][0] or 'main'} ===", flush=True)
        bfm.run_build_feature_matrix(tables["unit_table"].copy(), tables["trial_table"].copy(),
                                     lick_df=tables["lick_df"].copy(), config_path=drv.CONFIG_PATH,
                                     out_root=V2 / name, **cfg)
        print(f"=== features {name} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


def copy_feature_folder(src, dst_root):
    s = feature_dir(src)
    d = dst_root / s.relative_to(V2 / src)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("feature_matrix.npz", "neuron_metadata.csv", "config_used.yaml"):
        if (s / f).exists():
            shutil.copyfile(s / f, d / f)                  # content only: NAS refuses copystat
    return d


def slice_feature_folder(src, dst_root, cuts, cond_labels):
    """042.sliced_data_folder: drop the bins whose END is after the cut in the listed conditions; nothing recomputed."""
    d = copy_feature_folder(src, dst_root)
    z = dict(np.load(d / "feature_matrix.npz", allow_pickle=True))
    nb = list(z["n_bins_list"].astype(int)); n_c = int(z["n_conds"])
    assert n_c == len(cond_labels), (n_c, cond_labels)
    off = np.r_[0, np.cumsum(nb)]
    keep_cols, new_nb, rep = [], [], {}
    for ci in range(n_c):
        t = z[f"t_ctr_{ci}"]; lab = cond_labels[ci]
        bw = float(np.median(np.diff(t)))
        keep = (t + bw / 2 <= cuts[lab] + 1e-9) if lab in cuts else np.ones(len(t), bool)
        keep_cols.append(np.arange(off[ci], off[ci + 1])[keep])
        z[f"t_ctr_{ci}"] = t[keep]; new_nb.append(int(keep.sum()))
        if lab in cuts:
            rep[lab] = dict(bins_before=int(len(t)), bins_after=int(keep.sum()), t_ctr_max_after=float(t[keep].max()))
    cols = np.concatenate(keep_cols)
    for k in ("X", "X_odd", "X_even"):
        z[k] = z[k][:, cols]
    z["n_bins_list"] = np.array(new_nb)
    np.savez_compressed(d / "feature_matrix.npz", **z)
    return d, rep


# ── fit ───────────────────────────────────────────────────────────────────────
def install_node_fit(locality, time_lag_window, capture):
    """006: rastermap with grid_upsample=10; cluster labels = the model's k-means nodes (embedding_clust)."""
    from rastermap import Rastermap
    import rastermap_psth.run_clustering_new as rcn

    def fit_rastermap_nodes(X, n_clusters):
        n_pcs = min(200, X.shape[0] - 1, X.shape[1] - 1)
        model = Rastermap(n_clusters=n_clusters, n_PCs=n_pcs, locality=locality, time_lag_window=time_lag_window,
                          grid_upsample=GRID_UPSAMPLE, verbose=False).fit(X)
        node = np.asarray(model.embedding_clust).astype(int)
        emb = np.asarray(model.embedding)[:, 0]
        isort = np.lexsort((emb, node))
        sizes = np.bincount(node, minlength=n_clusters)
        capture.append(dict(node=node, sizes=sizes))
        return isort, np.cumsum(sizes)[:-1].astype(int)

    rcn.fit_rastermap = fit_rastermap_nodes
    return rcn


def phase_fit(name):
    import rastermap_psth.rastermap_utils as rmu
    from rastermap_psth.population_matrix_summary import run_matrix_summary
    drv = _drv()
    t0 = time.time()
    cfg_full = cfg_of(name)
    rmu.DEFAULT_CFG["locality"] = cfg_full["locality"]
    rmu.DEFAULT_CFG["time_lag_window"] = cfg_full["time_lag_window"]
    cap = []
    rcn = install_node_fit(cfg_full["locality"], cfg_full["time_lag_window"], cap)
    cfg = drv.load_cfg(drv.CONFIG_PATH, **cfg_full)
    drv.set_conditions(cfg)
    root = V2 / name
    src = VARIANTS[name][1]
    slicing = None
    if src is None:
        data_folder = feature_dir(name)
    elif src == "main":
        data_folder = copy_feature_folder("main", root)
    else:
        data_folder, slicing = slice_feature_folder("main", root, REWARD_CUTS, list(rmu.COND_LABELS))
    print(f"=== fit {name}: locality {cfg_full['locality']}, grid_upsample {GRID_UPSAMPLE}, data {data_folder} ===",
          flush=True)
    rres = rcn.run_rastermap(data_folder=data_folder, config_path=drv.CONFIG_PATH, **cfg_full)
    md = pathlib.Path(rres["method_dir"])
    d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    assert np.array_equal(d["cluster_labels"].astype(int), cap[-1]["node"]), "labels != k-means nodes"
    tables = load_tables(slim=True)
    drv.set_conditions(cfg)
    drv.fig_flow_umap_cv(md, cfg)
    drv.run_stats_only(md, cfg, tables["unit_table"], tables["trial_table"])
    st = drv.fig_cluster_mean_matrix_with_fmk(md, cfg, name)
    run_matrix_summary(md, drv.CONFIG_PATH, cfg_full, title_prefix=f"{LABELS[name]}: ", meta_cut=META_CUT)
    meta = pd.read_csv(pathlib.Path(data_folder) / "neuron_metadata.csv")
    prov = dict(
        variant=name, label=LABELS[name], change_from_main=VARIANTS[name][0], feature_source=src or "built",
        config=cfg_full, config_overrides=cfg_full, grid_upsample=GRID_UPSAMPLE, cluster_labels="rastermap k-means nodes",
        metaclustering=dict(method="contiguous", cut=META_CUT), reward_cuts_s=REWARD_CUTS if slicing else None,
        slicing_report=slicing, config_path=str(drv.CONFIG_PATH), tables=str(TABLES),
        mouse_info=str(rmu.MOUSE_INFO), data_folder=str(data_folder), method_dir=str(md),
        n_neurons=int(len(d["unit_ids"])), n_mice=int(meta.mouse_id.nunique()), n_sessions=int(meta.session_id.nunique()),
        mice_by_reward_group={g: sorted(x.mouse_id.unique().tolist()) for g, x in meta.groupby("reward_group")},
        n_clusters_fdr_sig_mannwhitney=int((st.p_mannwhitney_fdr < 0.05).sum()),
        n_clusters_fdr_sig_welch=int((st.p_welch_t_fdr < 0.05).sum()),
        created=datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
        runtime_min=round((time.time() - t0) / 60, 1))
    json.dump(prov, open(root / "variant_provenance.json", "w"), indent=2)
    print(f"=== fit {name} done in {prov['runtime_min']} min: {prov['n_neurons']} neurons, {prov['n_mice']} mice ===",
          flush=True)


# ── figures ───────────────────────────────────────────────────────────────────
def _run(args):
    print("+ " + " ".join(map(str, args)), flush=True)
    r = subprocess.run([sys.executable, "-u", *map(str, args)], env=os.environ.copy())
    if r.returncode:
        print(f"!! FAILED ({r.returncode}): {args[0]}", flush=True)
    return r.returncode


def phase_figures(name):
    from rastermap_psth.population_matrix_summary import metaclustering
    n_meta = int(max(metaclustering(method_dir(name), cut=META_CUT)["families"].values()))
    rc = [_run([HERE / "003_metacluster_fmk.py", "--variants", name]),
          _run([HERE / "004_metacluster_run_stats_only.py", "--variants", name]),
          _run([HERE / "001_cluster_roc_eda.py", "--variants", name]),
          _run([HERE / "023_sig_cluster_figures.py", "--variant", name, "--title", LABELS[name]]),
          _run([HERE / "045_metacluster_description.py", "--variant", name, "--order", *range(1, n_meta + 1),
                "--name", "metacluster_description_all"])]
    print(f"=== figures {name}: {sum(r != 0 for r in rc)} step(s) failed ===", flush=True)


# ── organise ──────────────────────────────────────────────────────────────────
TOPICS = [   # (figures/<topic>, folder relative to the method dir or variant root, glob)
    ("01_population", "md", "fig*population*"), ("01_population", "md", "fig_flow_umap*"),
    ("01_population", "md", "fig_cluster_mean*"), ("01_population", "md/matrix_summary", "*"),
    ("02_clusters", "md/sig_cluster_figures", "*"), ("02_clusters", "md/stats", "*"),
    ("03_metaclusters", "root/metacluster_description", "*"), ("03_metaclusters", "md/metacluster_stats", "fig*"),
    ("03_metaclusters", "md/metacluster_stats/stats", "*"),
    ("04_roc", "md/roc_eda", "*"), ("05_data", "data/diagnostics", "*"),
]
FIG_EXT = {".png", ".pdf", ".svg"}


def phase_organize(name):
    root, md = V2 / name, method_dir(name)
    base = dict(md=md, root=root, data=feature_dir(name))
    out = root / "figures"
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for topic, rel, pat in TOPICS:
        key, _, sub = rel.partition("/")
        src = base[key] / sub if sub else base[key]
        if not src.exists():
            continue
        for f in sorted(src.rglob(pat) if sub else src.glob(pat)):
            if f.is_file() and f.suffix in FIG_EXT:
                rel_src = f.relative_to(src)
                dst = out / topic / ("__".join(([sub.replace("/", "__")] if sub else []) + list(rel_src.parts)))
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(f, dst)
                    rows.append((topic, dst.name, str(f.relative_to(root))))
    prov = json.load(open(root / "variant_provenance.json"))
    lines = [f"# {LABELS[name]} (`{name}`)", "",
             f"Change from main: {VARIANTS[name][0] or 'none (main)'}; {prov['n_neurons']} neurons, {prov['n_mice']} mice, "
             f"{prov['n_sessions']} sessions. Copies of the pipeline figures, by topic (source path in the third column).",
             "", "| topic | figure | source |", "|---|---|---|"]
    lines += [f"| {t} | {n} | {s} |" for t, n, s in rows]
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"=== organize {name}: {len(rows)} figures -> {out} ===", flush=True)


# ── compare ───────────────────────────────────────────────────────────────────
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]


def _labels(name):
    from rastermap_psth.population_matrix_summary import metaclustering
    md = method_dir(name)
    d = pd.read_csv(md / "neuron_cluster_labels_cv.csv", usecols=KEYS + ["cluster_label"])
    fam = metaclustering(md, cut=META_CUT)["families"]
    d["metacluster"] = d.cluster_label.map(lambda k: fam[k])
    d["cluster_id"] = d.cluster_id.astype(str)
    return d


def _meta_psth(name, labels):
    """metacluster-mean PSTH (feature matrix X, full trials) with the bin times, rows = metaclusters 1..F."""
    z = np.load(feature_dir(name) / "feature_matrix.npz", allow_pickle=True)
    meta = pd.read_csv(feature_dir(name) / "neuron_metadata.csv")
    meta["cluster_id"] = meta.cluster_id.astype(str)
    m = meta[KEYS].merge(labels, on=KEYS, how="left")
    assert m.metacluster.notna().all(), name
    nb = z["n_bins_list"].astype(int)
    tags = np.concatenate([[(ci, round(float(t), 4)) for t in z[f"t_ctr_{ci}"]] for ci in range(len(nb))]).tolist()
    F = int(m.metacluster.max())
    P = np.vstack([z["X"][m.metacluster.to_numpy() == k].mean(0) for k in range(1, F + 1)])
    return P, [tuple(t) for t in tags]


def phase_compare():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_mutual_info_score
    out = V2 / "_comparison"
    out.mkdir(exist_ok=True)
    ref = _labels("main")
    P0, tag0 = _meta_psth("main", ref)
    rows, matches = [], []
    names = [n for n in VARIANTS if n != "main" and (V2 / n / "variant_provenance.json").exists()]
    for n in names:
        lab = _labels(n)
        j = ref.merge(lab, on=KEYS, suffixes=("_main", "_var"))
        P1, tag1 = _meta_psth(n, lab)
        common = [t for t in tag0 if t in set(tag1)]
        i0 = [tag0.index(t) for t in common]; i1 = [tag1.index(t) for t in common]
        A, B = P0[:, i0], P1[:, i1]
        A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
        B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
        R = A @ B.T / A.shape[1]                              # Pearson r, main metaclusters x variant metaclusters
        ri, ci = linear_sum_assignment(-R)
        r_match = np.full(len(A), np.nan); r_match[ri] = R[ri, ci]
        prov = json.load(open(V2 / n / "variant_provenance.json"))
        rows.append(dict(variant=n, label=LABELS[n], n_neurons=prov["n_neurons"], n_mice=prov["n_mice"],
                         n_common_neurons=len(j), frac_main_neurons_common=len(j) / len(ref),
                         n_metaclusters=int(lab.metacluster.max()),
                         ami_clusters=adjusted_mutual_info_score(j.cluster_label_main, j.cluster_label_var),
                         ami_metaclusters=adjusted_mutual_info_score(j.metacluster_main, j.metacluster_var),
                         median_matched_r=float(np.nanmedian(r_match)),
                         n_main_metaclusters_r_ge_0p8=int(np.nansum(r_match >= 0.8)), n_bins_compared=len(common)))
        matches += [dict(variant=n, main_metacluster=k + 1, matched_variant_metacluster=int(c) + 1, r=float(R[k, c]))
                    for k, c in zip(ri, ci)]
    S = pd.DataFrame(rows); M = pd.DataFrame(matches)
    S.to_csv(out / "variant_summary.csv", index=False); M.to_csv(out / "metacluster_matches.csv", index=False)
    if not len(S):
        print("compare: no finished variants", flush=True); return
    F0 = int(ref.metacluster.max())
    fig, ax = plt.subplots(1, 3, figsize=(13, 0.45 * len(S) + 1.8), dpi=200,
                           gridspec_kw=dict(width_ratios=[1, 1, 2.2]))
    y = np.arange(len(S))
    for a, col, title in ((ax[0], "ami_clusters", "AMI vs main: clusters"),
                          (ax[1], "ami_metaclusters", "AMI vs main: metaclusters")):
        a.barh(y, S[col], color="0.35"); a.set_xlim(0, 1); a.set_title(title, fontsize=8)
        a.set_yticks(y, S.label, fontsize=7); a.invert_yaxis()
    Mr = M.pivot(index="variant", columns="main_metacluster", values="r").reindex(S.variant)
    im = ax[2].imshow(Mr.to_numpy(), vmin=0, vmax=1, cmap="Greys", aspect="auto")
    ax[2].set_yticks(y, S.label, fontsize=7); ax[2].set_xticks(np.arange(F0), np.arange(1, F0 + 1), fontsize=6)
    ax[2].set_xlabel("main metacluster (rastermap order)", fontsize=7)
    ax[2].set_title("PSTH r of the matched variant metacluster (Hungarian)", fontsize=8)
    fig.colorbar(im, ax=ax[2], fraction=0.03, label="r")
    for a in ax[:2]:
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"variant_comparison.{ext}", bbox_inches="tight")
    print(S.to_string(index=False), flush=True)


if __name__ == "__main__":
    assert "haas" in socket.gethostname(), "run on haas"
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tables", "features", "fit", "figures", "organize", "compare"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--test", action="store_true", help="8-mouse test run in rastermap_variants/_v2_test")
    a = ap.parse_args()
    bad = [v for v in a.variants if v not in VARIANTS]
    assert not bad, f"unknown variants {bad}"
    V2.mkdir(parents=True, exist_ok=True)
    if a.phase == "tables":
        phase_tables()
    elif a.phase == "features":
        phase_features(a.variants)
    elif a.phase == "compare":
        phase_compare()
    else:
        fn = dict(fit=phase_fit, figures=phase_figures, organize=phase_organize)[a.phase]
        for v in a.variants:
            fn(v)
    print("ALL DONE", flush=True)
