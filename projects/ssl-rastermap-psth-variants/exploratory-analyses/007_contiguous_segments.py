"""Contiguity-constrained metaclustering ("segments") of rastermap clusters, sklearn AgglomerativeClustering.

Same input / similarity / linkage / cut as run_meta_clustering (cluster_postproc_analyses), but merges
are only allowed between clusters that are NEIGHBOURS in rastermap order, so every segment is one
contiguous stretch of the rastermap axis:
    input      cluster-mean PSTHs from neuron_psth_by_condition.npz (full trials), one row per cluster,
               rows in rastermap cluster order (cluster k = position k; true for k-means-node labels)
    distance   1 - Pearson r  (= cosine distance on row-mean-centred vectors)
    linkage    average (unweighted by cluster size, as run_meta_clustering)
    constraint connectivity = chain graph (cluster i <-> i-1, i+1)
    cut        distance_threshold = 0.35 (same as run_meta_clustering's meta_dendro_cut_dist)
Also: threshold sweep (0.15-0.80; n segments, silhouette on 1-r), contiguity assert.

Per variant → <method_dir>/segment_stats/:
    segment_map.csv                    cluster (1-based) -> segment (1..S, in rastermap order)
    segment_threshold_sweep.csv/.png
    rastermap_results_cv.npz, neuron_cluster_labels_cv.csv   relabelled (cluster_labels = segment-1)
    stats/                             run_stats_only on segments (F_mk computed there directly)
    stats/per_segment_stats_mwu_welch.csv
    fig_fmk_segment.png                per-mouse F_mk per segment, R+ vs R-
and <method_dir>/matrix_summary_segments/  (population_matrix_summary, rastermap order, segment strip)

Run on haas from ~/code/unit_spikes_analysis:  PYTHONPATH=~/code/NWB_reader:. .venv/bin/python -u <this> --variants ...
"""
import argparse
import json
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import rastermap_psth.rastermap_utils as rmu
from rastermap_psth.rastermap_utils import run_stats_only, load_cfg
from rastermap_psth import cluster_postproc_analyses as cpa
from rastermap_psth.population_matrix_summary import run_matrix_summary
from ephys_utilities.plotting_utils.plotting_utils import GROUP_COLORS

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CACHE = VARIANTS_ROOT / "_cache" / "tables_day0.pkl"
CONFIG_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")
CUT = cpa.DEFAULT_CFG["meta_dendro_cut_dist"]          # 0.35, same as run_meta_clustering
SWEEP = np.round(np.arange(0.15, 0.801, 0.05), 2)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def cluster_means(md):
    """Exactly run_meta_clustering's input: cluster-mean PSTHs over full-trial neuron PSTHs."""
    cluster_df = cpa.load_cluster_table(md)
    uid, cond_labels, psth, _ = cpa.load_psth_npz(md)
    X_full, sub, _ = cpa._build_neuron_matrix(cluster_df, cond_labels, psth, uid)
    K = int(sub.cluster_label.max()) + 1
    assert set(sub.cluster_label.unique()) == set(range(K)), "missing cluster labels"
    return np.vstack([X_full[sub.cluster_label.values == k].mean(0) for k in range(K)])


def segment(means, thr, return_model=False):
    K = len(means)
    C = means - means.mean(1, keepdims=True)                    # cosine on centred rows == 1 - Pearson r
    conn = diags([np.ones(K - 1), np.ones(K - 1)], [-1, 1], shape=(K, K))
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average",
                                    connectivity=conn, compute_full_tree=True, compute_distances=True).fit(C)
    lab = model.labels_
    # renumber 1..S in rastermap order and check contiguity
    order = list(dict.fromkeys(lab))
    seg = np.array([order.index(l) + 1 for l in lab])
    assert np.all(np.diff(seg) >= 0), "segments are not contiguous in rastermap order"
    return (seg, model) if return_model else seg

def tree_to_linkage(model, K):
    """sklearn merge tree -> scipy linkage Z. Every constrained merge joins two adjacent runs, so putting
    the child covering lower rastermap positions on the left makes the dendrogram leaf order equal the
    rastermap order (asserted)."""
    from scipy.cluster.hierarchy import dendrogram
    lo, cnt = {i: i for i in range(K)}, {i: 1 for i in range(K)}
    Z = []
    for i, (a, b) in enumerate(model.children_):
        a, b = int(a), int(b)
        if lo[b] < lo[a]:
            a, b = b, a
        node = K + i
        lo[node], cnt[node] = lo[a], cnt[a] + cnt[b]
        Z.append([a, b, float(model.distances_[i]), cnt[node]])
    Z = np.array(Z, float)
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    assert leaves == list(range(K)), "dendrogram leaf order differs from rastermap order"
    return Z


def qc_figures(means, seg, Z, out, variant):
    """Analogues of run_meta_clustering's QC figures for the contiguous segmentation."""
    import matplotlib.colors as mc
    from scipy.cluster.hierarchy import dendrogram
    from rastermap_psth.population_matrix_summary import family_palette
    K = len(seg); S = int(seg.max())
    pal = family_palette(range(1, S + 1))
    corr = 1 - squareform(np.clip(pdist(means, "correlation"), 0, None))
    sizes = np.bincount(seg)[1:]
    fig, ax = plt.subplots(figsize=(max(3, 0.12 * S + 1), 2.6), dpi=250)
    ax.bar(np.arange(1, S + 1), sizes, color=[pal[k] for k in range(1, S + 1)])
    ax.set_xlabel("Segment (rastermap order)"); ax.set_ylabel("n clusters"); ax.tick_params(labelsize=6)
    ax.set_title(f"{variant}: {S} contiguous segments ({int((sizes == 1).sum())} single-cluster)", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"qc_segment_sizes.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 6.5), dpi=250,
                             gridspec_kw={"width_ratios": [1.2, 0.18, 5, 5], "wspace": 0.05})
    lcf = cpa.build_link_color_func(Z, list(seg), pal, default_color="#9a9a9a")
    dn = dendrogram(Z, no_plot=True, link_color_func=lcf)
    for ic, dc, c in zip(dn["icoord"], dn["dcoord"], dn["color_list"]):
        axes[0].plot(dc, np.asarray(ic) / 10, color=c, lw=0.6)          # leaf i drawn at y = i + 0.5
    axes[0].axvline(CUT, color="k", ls="--", lw=0.5)
    axes[0].set_ylim(K, 0); axes[0].set_xlim(float(Z[:, 2].max()) * 1.03, 0); axes[0].set_yticks([])
    axes[0].set_xlabel("1 - r", fontsize=7); axes[0].set_title("Contiguous\ndendrogram", fontsize=8)
    for sp in ("top", "left", "right"):
        axes[0].spines[sp].set_visible(False)
    axes[1].imshow(np.array([[mc.to_rgb(pal[k])] for k in seg]), aspect="auto", extent=[0, 1, K, 0])
    axes[1].axis("off"); axes[1].set_title("Seg.", fontsize=8)
    v = np.nanpercentile(np.abs(means), 97)
    axes[2].imshow(means, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v, extent=[0, means.shape[1], K, 0],
                   interpolation="none")
    axes[2].set_yticks([]); axes[2].set_xticks([])
    axes[2].set_title("Cluster-mean PSTH (full trials, rastermap order)", fontsize=8)
    axes[3].imshow(corr, cmap="bwr", vmin=-1, vmax=1, extent=[0, K, K, 0], interpolation="none")
    edges = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, K]
    for a_, b_ in zip(edges[:-1], edges[1:]):
        axes[3].add_patch(plt.Rectangle((a_, a_), b_ - a_, b_ - a_, fill=False, ec="k", lw=0.7))
    axes[3].set_yticks([]); axes[3].set_xticks([])
    axes[3].set_title("Cluster correlation (Pearson r), segments boxed", fontsize=8)
    fig.suptitle(f"{variant}: contiguity-constrained metaclustering (average linkage, 1 - r <= {CUT})", fontsize=9)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"qc_segment_dendrogram_matrix_corr.{ext}", bbox_inches="tight")
    plt.close(fig)



def relabel_folder(md, seg, out):
    d = dict(np.load(md / "rastermap_results_cv.npz", allow_pickle=True))
    labels = d["cluster_labels"].astype(int)
    s0 = seg[labels] - 1
    isort_old = d["isort"]
    isort = np.concatenate([isort_old[s0[isort_old] == j] for j in range(s0.max() + 1)])
    d.update(cluster_labels=s0, isort=isort, boundaries=np.cumsum(np.bincount(s0))[:-1])
    np.savez_compressed(out / "rastermap_results_cv.npz", **d)
    meta = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    col = "cluster_label" if "cluster_label" in meta else "cluster_label_cv"
    meta["fine_cluster_label"] = meta[col]
    meta[col] = meta[col].map(lambda c: seg[c] - 1)
    meta.to_csv(out / "neuron_cluster_labels_cv.csv", index=False)


def welch_sidecar(out, variant):
    fz = np.load(out / "stats" / "f_matrix.npz", allow_pickle=True)
    F, g = fz["f_matrix"], fz["reward_groups"].astype(str)
    rows = []
    for j in range(F.shape[1]):
        a, b = F[g == "R+", j], F[g == "R-", j]
        rows.append(dict(variant=variant, segment=j + 1, n_rplus=len(a), n_rminus=len(b),
                         mean_rplus=a.mean(), sem_rplus=a.std(ddof=1) / np.sqrt(len(a)),
                         mean_rminus=b.mean(), sem_rminus=b.std(ddof=1) / np.sqrt(len(b)),
                         p_mannwhitney=mannwhitneyu(a, b, alternative="two-sided").pvalue,
                         p_welch_t=ttest_ind(a, b, equal_var=False).pvalue))
    t = pd.DataFrame(rows)
    t["p_mannwhitney_fdr"] = bh(t.p_mannwhitney); t["p_welch_t_fdr"] = bh(t.p_welch_t)
    t.to_csv(out / "stats" / "per_segment_stats_mwu_welch.csv", index=False)
    return t, F, g


def fig_fmk(F, g, out, variant, t):
    S = F.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, 0.38 * S + 2), 3.4), dpi=250)
    rng = np.random.default_rng(0)
    for off, grp, col in [(-0.18, "R+", GROUP_COLORS["rplus"]), (0.18, "R-", GROUP_COLORS["rminus"])]:
        Fg = F[g == grp]; x = np.arange(1, S + 1) + off
        for j in range(S):
            ax.scatter(np.full(len(Fg), x[j]) + rng.uniform(-0.06, 0.06, len(Fg)), Fg[:, j], s=3, color=col, alpha=0.35, lw=0)
        ax.errorbar(x, Fg.mean(0), yerr=Fg.std(0, ddof=1) / np.sqrt(len(Fg)), fmt="o", ms=3, color=col,
                    elinewidth=0.8, label=f"{grp} (n={len(Fg)} mice)")
    for j, r in t.iterrows():
        mark = "**" if (r.p_mannwhitney_fdr < 0.05 and r.p_welch_t_fdr < 0.05) else \
               "*" if (r.p_mannwhitney_fdr < 0.05 or r.p_welch_t_fdr < 0.05) else ""
        if mark:
            ax.text(r.segment, ax.get_ylim()[1] * 0.97, mark, ha="center", fontsize=7)
    ax.set_xticks(np.arange(1, S + 1)); ax.tick_params(labelsize=6)
    ax.set_xlabel("Contiguous segment (rastermap order)"); ax.set_ylabel(r"$F_{m,S}$")
    ax.set_title(f"{variant}: per-mouse fractional representation per segment "
                 "(** MWU & Welch FDR<0.05, * one test)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_fmk_segment.{ext}", bbox_inches="tight")
    plt.close(fig)


def sweep_fig(means, out, variant):
    D = squareform(np.clip(pdist(means, "correlation"), 0, None))
    rows = []
    for thr in SWEEP:
        seg = segment(means, thr)
        n = seg.max()
        sil = silhouette_score(D, seg, metric="precomputed") if 1 < n < len(seg) else np.nan
        rows.append(dict(threshold=thr, n_segments=int(n), silhouette=sil,
                         min_size_clusters=int(np.bincount(seg)[1:].min())))
    sw = pd.DataFrame(rows); sw.insert(0, "variant", variant)
    sw.to_csv(out / "segment_threshold_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=220)
    ax.plot(sw.threshold, sw.n_segments, "o-", color="k", ms=3); ax.set_ylabel("n segments")
    ax2 = ax.twinx(); ax2.plot(sw.threshold, sw.silhouette, "s--", color="tab:blue", ms=3)
    ax2.set_ylabel("silhouette (1 − r)", color="tab:blue")
    ax.axvline(CUT, color="0.5", ls=":", lw=0.8); ax.set_xlabel("distance threshold (1 − r)")
    ax.set_title(f"{variant}: contiguity-constrained segmentation", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"segment_threshold_sweep.{ext}", bbox_inches="tight")
    plt.close(fig)
    return sw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--cut", type=float, default=CUT, help="distance threshold (1 - r); default = run_meta_clustering's")
    a = ap.parse_args()
    CUT = a.cut                                   # module global, used by segment()/qc_figures()
    TAG = "" if abs(CUT - 0.35) < 1e-9 else f"_cut{CUT:.2f}".replace(".", "p")
    t0 = time.time()
    with open(CACHE, "rb") as f:
        tables = pickle.load(f)
    print(f"tables loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    for md in sorted(VARIANTS_ROOT.glob("qc_*/rastermap_clustering/*/*/clustering/*/rastermap")):
        variant = next(p for p in md.parts if p.startswith("qc_"))
        if variant not in a.variants:
            continue
        print(f"\n=== {variant} ===", flush=True)
        out = md / f"segment_stats{TAG}"; out.mkdir(exist_ok=True)
        means = cluster_means(md)
        sw = sweep_fig(means, out, variant)
        seg, model = segment(means, CUT, return_model=True)
        pd.DataFrame(dict(cluster=np.arange(len(seg)) + 1, family=seg, segment=seg)).to_csv(out / "segment_map.csv", index=False)
        Z = tree_to_linkage(model, len(seg))
        np.savez(out / "segment_linkage.npz", Z=Z, cut=CUT)
        qc_figures(means, seg, Z, out, variant)
        sizes_c = np.bincount(seg)[1:]
        print(f"  {seg.max()} contiguous segments at 1-r <= {CUT} (clusters per segment min/median/max "
              f"{sizes_c.min()}/{int(np.median(sizes_c))}/{sizes_c.max()})", flush=True)
        print(sw[["threshold", "n_segments", "silhouette"]].round(3).to_string(index=False), flush=True)
        relabel_folder(md, seg, out)
        qc = ["good"] if variant.startswith("qc_good__") else ["good", "mua"]
        cfg = load_cfg(CONFIG_PATH, unit_quality_label=qc)
        (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS,
         rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
        run_stats_only(out, cfg, tables["unit_table"], tables["trial_table"])
        t, F, g = welch_sidecar(out, variant)
        fig_fmk(F, g, out, variant, t)
        both = t[(t.p_mannwhitney_fdr < 0.05) & (t.p_welch_t_fdr < 0.05)]
        summ = dict(variant=variant, n_segments=int(seg.max()), cut=CUT,
                    permanova=(out / "stats" / "permanova_summary.txt").read_text().splitlines()[1:3],
                    n_sig_mwu=int((t.p_mannwhitney_fdr < 0.05).sum()), n_sig_welch=int((t.p_welch_t_fdr < 0.05).sum()),
                    sig_both=both.segment.tolist(),
                    sig_both_clusters={int(s): (np.flatnonzero(seg == s) + 1).tolist() for s in both.segment})
        json.dump(summ, open(out / "segment_summary.json", "w"), indent=2)
        print(json.dumps(summ), flush=True)
        run_matrix_summary(md, CONFIG_PATH, dict(unit_quality_label=qc), title_prefix=f"{variant}: ",
                           families_csv=out / "segment_map.csv", out_name=f"matrix_summary_segments{TAG}",
                           family_label=f"Segment (contiguous, 1-r<={CUT})", linkage_npz=out / "segment_linkage.npz")
    print("ALL DONE", flush=True)
