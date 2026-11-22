"""048 -- post-hoc analyses of the Bayesian GMM (046, Dirichlet-process, 19 components) on the main rastermap variant,
the same analyses as for the rastermap clusters (user 2026-11-20: "a Bayesian GMM version of my current rastermap
variant with post hoc analyses too").

As 004 (metacluster stats), a rastermap-shaped folder is built and the existing pipeline is run on it unchanged:
  <clustering>/gmm_bayesian_<labels>_posthoc/ (next to rastermap/, as the pipeline expects)
    rastermap_results_cv.npz      copy of the rastermap CV results (X = even-trial PSTHs, held out from the fit;
                                  unit_ids, mouse / reward arrays, timing, UMAP embedding unchanged) with
                                    cluster_labels = BGMM component (0-based), labels permuted to match the
                                                     rastermap clusters by mean PSTH (Hungarian assignment on
                                                     1 - Pearson r of the raw cluster-mean PSTHs (population mean kept), rectangular when K
                                                     differs); components ordered by the rastermap position of
                                                     their matched cluster (user rule 2026-11-21)
                                    isort          = neurons by component, rastermap order within a component
                                    boundaries     = component edges along isort
    neuron_cluster_labels_cv.csv  copy with cluster_label = component (+ rastermap_cluster kept)
    component_map.csv             original 046 component id -> ordered component (1-based), size, mice,
                                  matched rastermap cluster + PSTH r, best-r cluster, dominant (membership)
                                  rastermap cluster and its share
  psth_match_matrix.npz / .png  r(component mean PSTH, rastermap cluster mean PSTH), matched order
  Steps (each also standalone with --steps):
    stats    rastermap_utils.run_stats_only (F_mk per mouse, PERMANOVA, per-component Mann-Whitney + BH, dispersion,
             figS1-S5, anatomy-axis stats) + Welch sidecar (R+ / R- test pair, both BH)
    figs     run_clustering_new fig6 / fig6b (component mean PSTHs, pooled and per reward group), fig8 (rastermap's
             UMAP coloured by component), fig11 / fig12 (area and reward group per component)
    matrix   population_matrix_summary.run_matrix_summary (population matrix + cluster-mean matrix, F_mk strip)
    roc      001 ROC-type EDA (fractions and enrichment of ROC categories per component; its "metaclusters" are the
             contiguous merge of the ordered components, cut 0.40, as for rastermap)
    sig      023 figures of the components with a significant R+ / R- difference
Labels: --labels dp (046, default) | k24 | k100 (047 maximum-likelihood fixed K) | k24_bayes | k100_bayes.
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. nice -n 10 .venv/bin/python -u <this>
"""
import argparse
import importlib.util
import json
import pathlib
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, ttest_ind

HERE = pathlib.Path(__file__).resolve().parent
VARIANT = "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin"
VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CLU = (VARIANTS_ROOT / VARIANT / "rastermap_clustering/passive_active/whisker_auditory/clustering/"
       "n100_passive_active_unmerge_baseline_whisker_auditory_combined_nobl")
RM = CLU / "rastermap"
BG = CLU / "gmm_bayesian_main"
CACHE = VARIANTS_ROOT / "_cache" / "tables_day0.pkl"
CONFIG_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")
LABEL_FILES = {"dp": "bgmm_labels.csv", "k24": "bgmm_k24_labels.csv", "k100": "bgmm_k100_labels.csv",
               "k24_bayes": "bgmm_k24_bayes_labels.csv", "k100_bayes": "bgmm_k100_bayes_labels.csv"}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def build_folder(labels):
    out = CLU / f"gmm_bayesian_{labels}_posthoc"   # sibling of rastermap/: the pipeline reads ../ inputs
    out.mkdir(exist_ok=True)
    d = dict(np.load(RM / "rastermap_results_cv.npz", allow_pickle=True))
    lab = pd.read_csv(BG / LABEL_FILES[labels])
    lab = lab.set_index("unit_ids").reindex(d["unit_ids"])
    assert lab.bgmm_component.notna().all(), "BGMM labels missing for some rastermap units"
    comp = lab.bgmm_component.to_numpy(int)
    rlab = np.asarray(d["cluster_labels"], int)                               # rastermap clusters
    pos = np.empty(len(comp)); pos[d["isort"]] = np.arange(len(comp))         # rastermap position of each neuron
    X = np.asarray(d["X"], float)
    ids, rids = np.unique(comp), np.unique(rlab)
    mc = np.stack([X[comp == c].mean(0) for c in ids])
    mr = np.stack([X[rlab == r].mean(0) for r in rids])
    z = lambda m: (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True)
    R = z(mc) @ z(mr).T / X.shape[1]                                          # Pearson r, components x clusters
    ri, ci = linear_sum_assignment(1 - R)                                     # one-to-one, rectangular if K differs
    match = dict(zip(ids[ri], rids[ci]))
    rpos = {r: np.median(pos[rlab == r]) for r in rids}
    order = np.array(sorted(ids, key=lambda c: rpos[match[c]]))
    new_of = {int(c): j for j, c in enumerate(order)}                         # original id -> 0-based matched id
    rord = np.array(sorted(rids, key=lambda r: rpos[r]))
    Rm = R[np.searchsorted(ids, order)][:, np.searchsorted(rids, rord)]
    np.savez_compressed(out / "psth_match_matrix.npz", r=Rm, component_046=order, rastermap_cluster=rord + 1,
                        matched_cluster=np.array([match[c] for c in order]) + 1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.6))
    im = ax.imshow(Rm, aspect="auto", cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
    jj = np.searchsorted(rord, [match[c] for c in order])
    ax.scatter(jj, np.arange(len(order)), s=6, facecolors="none", edgecolors="k", lw=0.6)
    ax.set_xlabel("rastermap cluster (rastermap order)"); ax.set_ylabel("component (matched order)")
    ax.set_title(f"Mean-PSTH match (Hungarian): median matched r = {np.median(R[ri, ci]):.2f}", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.03)
    fig.savefig(out / "psth_match_matrix.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    c0 = np.array([new_of[c] for c in comp])
    isort = np.concatenate([d["isort"][c0[d["isort"]] == j] for j in range(len(order))])
    sizes = np.bincount(c0, minlength=len(order))
    d.update(cluster_labels=c0, isort=isort, boundaries=np.cumsum(sizes)[:-1])
    np.savez_compressed(out / "rastermap_results_cv.npz", **d)
    meta = pd.read_csv(RM / "neuron_cluster_labels_cv.csv").set_index("unit_ids").reindex(d["unit_ids"]).reset_index()
    meta["rastermap_cluster"] = meta["cluster_label"]
    meta["cluster_label"] = c0
    meta.to_csv(out / "neuron_cluster_labels_cv.csv", index=False)
    rows = []
    for c in order:
        m = comp == c
        rc = pd.Series(lab.rastermap_cluster.to_numpy()[m]).value_counts()
        rows.append(dict(component=new_of[int(c)] + 1, bgmm_component_046=int(c), n_neurons=int(m.sum()),
                         n_mice=int(lab.mouse_id[m].nunique()), median_rastermap_position=float(np.median(pos[m])),
                         matched_rastermap_cluster=int(match[c]) + 1,
                         matched_psth_r=float(R[np.searchsorted(ids, c), np.searchsorted(rids, match[c])]),
                         best_r_rastermap_cluster=int(rids[R[np.searchsorted(ids, c)].argmax()]) + 1,
                         best_psth_r=float(R[np.searchsorted(ids, c)].max()),
                         dominant_rastermap_cluster=int(rc.index[0]) + 1, dominant_share=float(rc.iloc[0] / m.sum()),
                         n_rastermap_clusters_ge5pct=int((rc / m.sum() >= 0.05).sum())))
    pd.DataFrame(rows).to_csv(out / "component_map.csv", index=False)
    print(f"  folder {out}: {len(order)} components, sizes {sorted(sizes.tolist(), reverse=True)}", flush=True)
    return out


def welch_sidecar(out):
    fz = np.load(out / "stats" / "f_matrix.npz", allow_pickle=True)
    F, g = fz["f_matrix"], fz["reward_groups"].astype(str)
    rows = []
    for j in range(F.shape[1]):
        a, b = F[g == "R+", j], F[g == "R-", j]
        rows.append(dict(component=j + 1, n_rplus=len(a), n_rminus=len(b),
                         mean_rplus=a.mean(), sem_rplus=a.std(ddof=1) / np.sqrt(len(a)),
                         mean_rminus=b.mean(), sem_rminus=b.std(ddof=1) / np.sqrt(len(b)),
                         p_mannwhitney=mannwhitneyu(a, b, alternative="two-sided").pvalue,
                         p_welch_t=ttest_ind(a, b, equal_var=False).pvalue))
    t = pd.DataFrame(rows)
    t["p_mannwhitney_fdr"], t["p_welch_t_fdr"] = bh(t.p_mannwhitney), bh(t.p_welch_t)
    t.to_csv(out / "stats" / "per_component_stats_mwu_welch.csv", index=False)
    return t


def set_conditions(cfg):
    import rastermap_psth.rastermap_utils as rmu
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = \
        rmu.get_conditions(cfg)
    return rmu


def step_stats(out, cfg):
    from rastermap_psth.rastermap_utils import run_stats_only
    with open(CACHE, "rb") as f:
        tables = pickle.load(f)
    run_stats_only(out, cfg, tables["unit_table"], tables["trial_table"])
    t = welch_sidecar(out)
    print(json.dumps(dict(n_components=len(t), sig_mwu_fdr=t.loc[t.p_mannwhitney_fdr < 0.05, "component"].tolist(),
                          sig_welch_fdr=t.loc[t.p_welch_t_fdr < 0.05, "component"].tolist())), flush=True)
    print((out / "stats" / "permanova_summary.txt").read_text(), flush=True)


def step_figs(out, cfg):
    import rastermap_psth.run_clustering_new as rcn
    rmu = set_conditions(cfg)
    for name in ("CONDITIONS", "COND_LABELS", "COND_COLORS", "COND_LABELS_MATRIX", "COND_ALIGN_COLS"):
        setattr(rcn, name, getattr(rmu, name))                      # run_clustering_new reads its own module globals
    d = np.load(out / "rastermap_results_cv.npz", allow_pickle=True)
    X, lab, k = d["X"], d["cluster_labels"], int(d["cluster_labels"].max() + 1)
    nb = list(d["n_bins_list"])
    t_ctrs = [d[f"t_ctr_{i}"] for i in range(len(nb))]
    meta = pd.read_csv(out / "neuron_cluster_labels_cv.csv")
    rcn.fig6_cluster_profiles(X, t_ctrs, nb, d["isort"], d["boundaries"], out, prefix="cv")
    rcn.fig6b_cluster_profiles_reward_groups(X, t_ctrs, nb, d["isort"], d["boundaries"], rmu.COND_LABELS,
                                             rmu.COND_COLORS, d["reward_arr"], out, prefix="cv")
    rcn.fig8_umap(d["umap_embedding"], lab, k, out, prefix="cv")
    rcn.fig11_area_per_cluster(d["unit_ids"], lab, meta.area_acronym_custom.astype(str).to_numpy(), k, out)
    rcn.fig12_reward_per_cluster(d["unit_ids"], lab, d["reward_arr"], k, out)


def step_matrix(out, cfg):
    from rastermap_psth.population_matrix_summary import run_matrix_summary
    run_matrix_summary(out, CONFIG_PATH, dict(unit_quality_label=["good", "mua"]),
                       title_prefix=f"{VARIANT} (Bayesian GMM): ")


def step_roc(out, cfg):
    spec = importlib.util.spec_from_file_location("roc_eda_001", HERE / "001_cluster_roc_eda.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.run(out)


def step_sig(out, cfg):
    if not (pd.read_csv(out / "stats" / "per_cluster_stats.csv").p_fdr < 0.05).any():
        print("  no component with a significant R+ / R- difference (BH): no 023 figures", flush=True)
        return
    subprocess.run([sys.executable, "-u", str(HERE / "023_sig_cluster_figures.py"), "--variant", VARIANT,
                    "--method-dir", str(out), "--title", f"{VARIANT}: Bayesian GMM components"], check=True)


STEPS = {"stats": step_stats, "figs": step_figs, "matrix": step_matrix, "roc": step_roc, "sig": step_sig}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="dp", choices=list(LABEL_FILES))
    ap.add_argument("--steps", nargs="*", default=list(STEPS))
    a = ap.parse_args()
    from rastermap_psth.rastermap_utils import load_cfg
    cfg = load_cfg(CONFIG_PATH, unit_quality_label=["good", "mua"])
    set_conditions(cfg)
    out = build_folder(a.labels)
    json.dump(dict(script="048_bgmm_posthoc.py", labels=a.labels, label_file=LABEL_FILES[a.labels],
                   source_rastermap=str(RM), steps=a.steps, date=time.strftime("%Y-%m-%d %H:%M")),
              open(out / "posthoc_provenance.json", "w"), indent=1)
    for s in a.steps:
        t0 = time.time()
        print(f"\n=== step {s} ===", flush=True)
        try:
            STEPS[s](out, cfg)
            print(f"=== step {s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        except Exception:
            import traceback
            print(f"=== step {s} FAILED ===\n{traceback.format_exc()}", flush=True)
    print("ALL DONE", flush=True)
