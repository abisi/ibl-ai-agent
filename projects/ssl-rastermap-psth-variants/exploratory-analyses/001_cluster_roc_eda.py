"""EDA: which rastermap clusters / metaclusters are made of which ROC-responsive neuron types.

For each rastermap variant produced by 000_run_rastermap_variants.py:
  1. Metaclustering with the user's own run_meta_clustering (cluster_postproc_analyses.py:
     hierarchical, average linkage on 1 - Pearson r of cluster-mean PSTHs, cut 0.35).
  2. Join day-0 ROC results (combined_results_ks4/<mouse>/whisker_0/roc_analysis/
     <mouse>_roc_results_new.csv) onto the clustered neurons on the mandatory unit key
     (mouse_id, session_id, electrode_group, cluster_id); verified one-to-one per analysis_type.
     NOTE: the user's run_roc_analysis joins without cluster_id (many-to-many) — not used here.
  3. Per-neuron features:
       - significance of each ROC analysis_type, split by direction (pos / neg);
         AUC == NaN (untestable, e.g. no auditory misses) -> NOT significant, although the
         ROC pipeline writes significant=True for those rows;
         choice-type directions inverted, as in unit_spikes_analysis.py ("positive and negative
         are inverted" fix).
       - derived labels from roc_utils_new.compute_neuron_labels (the user's implementation of
         combined-significance, Oryshchuk et al. 2024-style, classes):
           M_decision       = sig M_choice AND sig M_hit_vs_spontaneous
           decision = (whisker_choice | auditory_choice) & (whisker_hit_vs_spontaneous | auditory_hit_vs_spontaneous);
           gated = decision & !(whisker_sensory | auditory_sensory) & !spontaneous_licks_vs_cr
           (shown in column figures as a fraction of decision neurons, grey if < 10);
           motor = spontaneous_licks & spontaneous_licks_vs_cr, same sign (def. C)
           sensory_label    = sig in any passive (pre/post, whisker/auditory) analysis
           modality_preference = whisker / auditory / bimodal / non_responsive
         Recomputed here from the NaN-corrected significance (same formulas).
  4. Per cluster and per metacluster: fraction of neurons per feature; enrichment vs all other
     neurons (Fisher exact, two-sided, BH-FDR within feature). Descriptive only: neurons are
     not independent within mice (see mouse-level check in the summary csv: number of mice
     contributing significant neurons per cluster).
  5. Figures (in <method_dir>/roc_eda/):
       eda_cluster_roc_heatmap        clusters x features (fraction, FDR-enriched cells outlined)
       eda_metacluster_roc_heatmap    metaclusters x features
       eda_feature_heterogeneity      chi-square across clusters per feature (drives column pick)
       fig_rastermap_roc_columns      neuron-level CV rastermap matrix + metacluster strip +
                                      per-cluster ROC-fraction columns
       fig_clustermean_roc_columns    cluster-mean CV matrix + metacluster strip + ROC columns
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import fisher_exact, chi2_contingency

import rastermap_psth.rastermap_utils as rmu
from rastermap_psth.rastermap_utils import load_cfg
from rastermap_psth import cluster_postproc_analyses as cpa

OUTPUT_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
VARIANTS_ROOT = OUTPUT_PATH / "rastermap_variants"
CONFIG_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")
KEY = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
CHOICE_TYPES_INVERT = True  # as unit_spikes_analysis.py: choice directions are inverted
# derived labels = population_matrix_summary.ROC_CATEGORIES definitions (2026-09-27)
LABELS = ["decision", "gated_decision", "motor_lick_related"]
GATED_MIN_DECISION = 10   # column figures: gated as a fraction of each cluster's decision neurons
FDR = 0.05


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def find_method_dirs(root):
    return sorted(p for p in root.glob("*/rastermap_clustering/*/*/clustering/*/rastermap")
                  if (p / "rastermap_results_cv.npz").exists())


def load_roc(neurons):
    frames = []
    for (m, s), _ in neurons.groupby(["mouse_id", "session_id"]):
        f = OUTPUT_PATH / m / "whisker_0" / "roc_analysis" / f"{m}_roc_results_new.csv"
        if not f.exists():
            print(f"  [warn] no ROC file for {m}", flush=True)
            continue
        d = pd.read_csv(f, usecols=KEY + ["analysis_type", "auc", "selectivity", "significant", "direction"])
        frames.append(d[d.session_id == s])
    roc = pd.concat(frames, ignore_index=True)
    roc["cluster_id"] = roc["cluster_id"].astype(str)
    roc["significant"] = roc["significant"].astype(bool) & roc["auc"].notna()
    roc.loc[roc.auc.isna(), "selectivity"] = np.nan  # untestable rows carry no selectivity
    if CHOICE_TYPES_INVERT:  # label and sign flipped together so they stay consistent
        m = roc.analysis_type.str.contains("choice")
        roc.loc[m, "direction"] = roc.loc[m, "direction"].replace({"positive": "negative", "negative": "positive"})
        roc.loc[m, "selectivity"] = -roc.loc[m, "selectivity"]
    dup = roc.duplicated(KEY + ["analysis_type"]).sum()
    assert dup == 0, f"{dup} duplicated ROC rows on unit key + analysis_type"
    return roc


def neuron_features(neurons, roc):
    """Wide per-neuron table: <type>__sig, <type>__pos, <type>__neg, <type>__sel (+ derived labels).
    __pos/__neg = significant with selectivity > 0 / < 0. For wh_vs_aud_* types selectivity > 0
    is auditory-preferring and < 0 whisker-preferring (direction column 'auditory'/'whisker')."""
    r = roc.copy()
    r["pos"] = r.significant & (r.selectivity > 0)
    r["neg"] = r.significant & (r.selectivity < 0)
    r["sel"] = r.selectivity
    wide = r.pivot_table(index=KEY, columns="analysis_type", values=["significant", "pos", "neg", "sel"],
                         aggfunc="first", dropna=False)
    wide.columns = [f"{t}__{'sig' if v == 'significant' else v}" for v, t in wide.columns]
    wide = wide.reset_index()
    X = neurons.merge(wide, on=KEY, how="left", validate="one_to_one")
    types = sorted(roc.analysis_type.unique())
    sig = lambda t: X.get(f"{t}__sig", pd.Series(False, index=X.index)).fillna(False).astype(bool)
    # decision: (hit vs miss, either modality) AND (hit vs spontaneous lick, either modality)
    dec = (sig("whisker_choice") | sig("auditory_choice")) & \
          (sig("whisker_hit_vs_spontaneous") | sig("auditory_hit_vs_spontaneous"))
    X["decision"] = dec
    # gated: decision AND NOT stimulus-only response (miss vs CR, either modality) AND NOT lick vs CR
    X["gated_decision"] = dec & ~(sig("whisker_sensory") | sig("auditory_sensory")) & ~sig("spontaneous_licks_vs_cr")
    # motor (lick-related), def. C: lick vs pre-lick AND lick vs CR, both significant, same sign
    s1 = X.get("spontaneous_licks__sel", pd.Series(np.nan, index=X.index))
    s2 = X.get("spontaneous_licks_vs_cr__sel", pd.Series(np.nan, index=X.index))
    X["motor_lick_related"] = sig("spontaneous_licks") & sig("spontaneous_licks_vs_cr") & (np.sign(s1) == np.sign(s2))
    passive = [f"{m}_passive_{e}" for m in ("whisker", "auditory") for e in ("pre", "post")]
    X["sensory_passive_any"] = np.column_stack([sig(t) for t in passive]).any(1)
    wa, aa = sig("whisker_active"), sig("auditory_active")
    X["modality_bimodal"] = wa & aa
    X["modality_whisker_only"] = wa & ~aa
    X["modality_auditory_only"] = aa & ~wa
    X["roc_matched"] = X[[c for c in X.columns if c.endswith("__sig")]].notna().any(axis=1)
    return X, types


def enrichment(X, group_col, feats):
    rows = []
    for f in feats:
        v = X[f].fillna(False).astype(bool).values
        g = X[group_col].values
        tab = []
        for k in np.unique(g):
            ink = g == k
            a, b = int(v[ink].sum()), int(ink.sum() - v[ink].sum())
            c, d = int(v[~ink].sum()), int((~ink).sum() - v[~ink].sum())
            odds, p = fisher_exact([[a, b], [c, d]])
            n_mice = X.loc[ink & v, "mouse_id"].nunique()
            tab.append(dict(group=k, feature=f, n=int(ink.sum()), n_sig=a, frac=a / max(ink.sum(), 1),
                            frac_rest=c / max((~ink).sum(), 1), odds_ratio=odds, p=p,
                            n_mice_with_sig=n_mice, n_mice_in_group=X.loc[ink, "mouse_id"].nunique()))
        t = pd.DataFrame(tab)
        t["p_fdr"] = bh(t.p)
        t["enriched"] = (t.p_fdr < FDR) & (t.frac > t.frac_rest)
        t["depleted"] = (t.p_fdr < FDR) & (t.frac < t.frac_rest)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def heterogeneity(X, group_col, feats):
    rows = []
    for f in feats:
        v = X[f].fillna(False).astype(bool)
        ct = pd.crosstab(X[group_col], v)
        if ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        rows.append(dict(feature=f, chi2=chi2, dof=dof, p=p, overall_frac=v.mean(),
                         cramers_v=np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))))
    h = pd.DataFrame(rows)
    h["p_fdr"] = bh(h.p)
    return h.sort_values("cramers_v", ascending=False)


def heatmap(E, groups, feats, title, path, ylabel):
    F = E.pivot(index="group", columns="feature", values="frac").reindex(index=groups, columns=feats)
    enr = E.pivot(index="group", columns="feature", values="enriched").reindex(index=groups, columns=feats)
    fig, ax = plt.subplots(figsize=(0.32 * len(feats) + 3, max(4, 0.13 * len(groups) + 2)), dpi=200)
    im = ax.imshow(F.values, aspect="auto", cmap="viridis", vmin=0, vmax=np.nanpercentile(F.values, 98))
    yy, xx = np.where(enr.fillna(False).values)
    ax.scatter(xx, yy, marker="s", s=12, facecolors="none", edgecolors="w", linewidths=0.6)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=6)
    step = max(1, len(groups) // 50)
    ax.set_yticks(range(0, len(groups), step)); ax.set_yticklabels([str(g) for g in groups][::step], fontsize=5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.4, label="fraction of neurons significant")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    plt.close(fig)


def family_colors(fam_ids):
    ids = sorted(set(fam_ids))
    cmap = plt.get_cmap("tab20")
    return {f: mcolors.to_hex(cmap(i % 20)) for i, f in enumerate(ids)}


def feat_name(f):
    t, _, kind = f.partition("__")
    if t.startswith("wh_vs_aud") and kind in ("pos", "neg"):
        kind = {"pos": "aud-pref", "neg": "wh-pref"}[kind]
    else:
        kind = {"sig": "", "pos": "(+)", "neg": "(−)", "sel": "sel."}.get(kind, kind)
    return f"{t.replace('_', ' ')} {kind}".strip()


def selectivity_summary(X, group_col, types):
    """Per group x analysis_type: mean, SEM, mean |selectivity|, n neurons with a valid AUC."""
    rows = []
    for t in types:
        c = f"{t}__sel"
        if c not in X:
            continue
        for k, g in X.groupby(group_col):
            v = g[c].dropna().values
            rows.append(dict(group=k, analysis_type=t, n=len(v),
                             mean_sel=v.mean() if len(v) else np.nan,
                             sem_sel=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                             mean_abs_sel=np.abs(v).mean() if len(v) else np.nan,
                             n_mice=g.loc[g[c].notna(), "mouse_id"].nunique()))
    return pd.DataFrame(rows)


def selectivity_line_panels(S, K, families, types, title, path):
    """One panel per analysis_type: mean ± SEM selectivity vs rastermap cluster index,
    metacluster membership as background bands."""
    fam_col = family_colors(families.values())
    ncol = 4
    nrow = int(np.ceil(len(types) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.2 * nrow), dpi=200, sharex=True)
    for ax, t in zip(axes.flat, types):
        s = S[S.analysis_type == t].set_index("group").reindex(range(K))
        for k in range(K):
            ax.axvspan(k - 0.5, k + 0.5, color=fam_col[families[k]], alpha=0.18, lw=0)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.errorbar(range(K), s.mean_sel, yerr=s.sem_sel, fmt="o-", ms=1.8, lw=0.6, elinewidth=0.5, color="k")
        ax.set_title(t + (" (+ = auditory-pref.)" if t.startswith("wh_vs_aud") else ""), fontsize=7)
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for ax in axes.flat[len(types):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Rastermap cluster index", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean selectivity\n2·(AUC−0.5) ± SEM", fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    plt.close(fig)


def selectivity_heatmap(S, groups, types, title, path, ylabel):
    M = S.pivot(index="group", columns="analysis_type", values="mean_sel").reindex(index=groups, columns=types)
    v = np.nanpercentile(np.abs(M.values), 98)
    fig, ax = plt.subplots(figsize=(0.32 * len(types) + 3, max(4, 0.13 * len(groups) + 2)), dpi=200)
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=90, fontsize=6)
    step = max(1, len(groups) // 50)
    ax.set_yticks(range(0, len(groups), step)); ax.set_yticklabels([str(g) for g in groups][::step], fontsize=5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.4, label="mean selectivity 2·(AUC−0.5)")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    plt.close(fig)


def roc_columns_figure(Xmat, row_cluster, K, families, col_feats, frac_tab, n_bins_list, cfg,
                       title, path, per_neuron_rows, mode="fraction"):
    """Matrix (neuron-level in rastermap order, or cluster-mean) + metacluster strip +
    one column per ROC feature: each cluster's block colored by its fraction significant
    (mode='fraction', sequential 0 -> column max) or its mean selectivity (mode='selectivity',
    diverging, symmetric per column)."""
    n_rows = len(row_cluster)
    fam_col = family_colors(families.values())
    widths = [12, 0.35] + [0.3] * len(col_feats)
    fig, axes = plt.subplots(1, 2 + len(col_feats), figsize=(12 + 0.35 * (1 + len(col_feats)) + 2, 12),
                             dpi=300, gridspec_kw={"width_ratios": widths, "wspace": 0.04})
    vmax = np.nanpercentile(np.abs(Xmat), cfg["vmax_pct"])
    im = rmu._draw_matrix(axes[0], Xmat, n_bins_list, [], vmax, cfg, title)
    cax = axes[0].inset_axes([0.0, -0.1, 0.25, 0.012])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Firing rate (baseline-norm.)")
    edges = np.flatnonzero(np.diff(row_cluster)) + 1
    for e in edges:
        axes[0].axhline(e, color="k", lw=0.15 if per_neuron_rows else 0.05, alpha=0.6)

    # contiguous blocks of each cluster along the rows
    blocks = []
    starts = np.r_[0, edges]; stops = np.r_[edges, n_rows]
    for a, b in zip(starts, stops):
        blocks.append((a, b, row_cluster[a]))

    ax = axes[1]
    for a, b, k in blocks:
        ax.add_patch(plt.Rectangle((0, a), 1, b - a, color=fam_col[families[k]], lw=0))
    ax.set_xlim(0, 1); ax.set_ylim(n_rows, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Meta-\ncluster", fontsize=6, rotation=90, va="bottom")

    axes[0].set_ylabel("Neuron (rastermap order)" if per_neuron_rows else "Rastermap cluster")
    sel = mode == "selectivity"
    cmap = plt.get_cmap("RdBu_r" if sel else "magma_r")
    for ax, f in zip(axes[2:], col_feats):
        fr = frac_tab.get(f, {})
        vals = np.array([v for v in fr.values() if np.isfinite(v)])
        if sel:
            vmax_f = max(np.nanpercentile(np.abs(vals), 98), 1e-3) if len(vals) else 1
            norm = mcolors.Normalize(-vmax_f, vmax_f)
        else:
            vmax_f = max(np.nanpercentile(vals, 98), 1e-3) if len(vals) else 1
            norm = mcolors.Normalize(0, vmax_f)
        for a, b, k in blocks:
            v = fr.get(k, np.nan)
            ax.add_patch(plt.Rectangle((0, a), 1, b - a, color=cmap(norm(v)) if np.isfinite(v) else "#b5b5b5", lw=0))
        ax.set_xlim(0, 1); ax.set_ylim(n_rows, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{feat_name(f)}  [{'±' if sel else 'max '}{vmax_f:.2f}]", fontsize=5.5, rotation=90, va="bottom")
    p0, p1 = axes[2].get_position(), axes[-1].get_position()
    cax = fig.add_axes([p0.x0, p0.y0 - 0.07, p1.x1 - p0.x0, 0.01])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=mcolors.Normalize(-1 if sel else 0, 1), cmap=cmap),
                      cax=cax, orientation="horizontal")
    if sel:
        cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(["−[±]", "0", "+[±]"])
        cb.set_label("cluster mean selectivity 2·(AUC−0.5)  (wh vs aud: + = auditory-pref.)", fontsize=6)
        sup = ("Columns: mean ROC selectivity of each cluster's neurons; per-column symmetric colour scale "
               "±[value] (98th pct of |mean| across clusters)")
    else:
        cb.set_ticks([0, 1]); cb.set_ticklabels(["0", "column max"])
        cb.set_label("fraction of cluster's neurons significant", fontsize=6)
        sup = ("Columns: fraction of each cluster's neurons significant (ROC); per-column colour scale "
               "0 → [max] (98th pct across clusters)")
    cb.ax.tick_params(labelsize=5)
    fig.suptitle(sup, fontsize=8, y=0.995)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    plt.close(fig)


def run(method_dir):
    variant = next(p for p in method_dir.parts if p.startswith("qc_"))
    print(f"\n=== {variant} ===\n{method_dir}", flush=True)
    out = method_dir / "roc_eda"
    out.mkdir(exist_ok=True)
    cfg = load_cfg(CONFIG_PATH, unit_quality_label=["good"] if "qc_good__" in variant else ["good", "mua"])
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS,
     rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)

    # 1. metaclustering: population_matrix_summary default = contiguity-constrained agglomerative
    #    (sklearn, average linkage on 1 - r, rastermap-neighbour connectivity, cut 0.40)
    from rastermap_psth.population_matrix_summary import metaclustering as _pms_meta
    _meta = _pms_meta(method_dir)
    fam = pd.DataFrame({"cluster_label": list(_meta["families"]), "family": list(_meta["families"].values())})
    fam_csv = out / "metacluster_families_contiguous.csv"
    fam.to_csv(fam_csv, index=False)
    families = dict(zip(fam.cluster_label, fam.family))
    print(f"  metaclusters: {fam.family.nunique()} families from {len(fam)} clusters ({fam_csv})", flush=True)

    # 2. neurons + ROC
    neurons = pd.read_csv(method_dir / "neuron_cluster_labels_cv.csv").rename(
        columns={"unit_id": "unit_ids", "cluster_label_cv": "cluster_label"})
    neurons["cluster_id"] = neurons["cluster_id"].astype(str)
    neurons["metacluster"] = neurons.cluster_label.map(families)
    roc = load_roc(neurons)
    X, types = neuron_features(neurons, roc)
    print(f"  ROC matched: {X.roc_matched.sum()}/{len(X)} neurons", flush=True)

    feats_sig = [f"{t}__sig" for t in types]
    feats_dir = [f"{t}__{d}" for t in types for d in ("pos", "neg")]
    feats_lab = LABELS + ["sensory_passive_any", "modality_whisker_only", "modality_auditory_only", "modality_bimodal"]
    all_feats = feats_sig + feats_dir + feats_lab
    Xm = X[X.roc_matched].copy()

    # 3. enrichment + heterogeneity
    Ec = enrichment(Xm, "cluster_label", all_feats)
    Em = enrichment(Xm, "metacluster", all_feats)
    H = heterogeneity(Xm, "cluster_label", all_feats)
    for df, name in [(Ec, "cluster"), (Em, "metacluster")]:
        df.insert(0, "variant", variant)
        df.to_csv(out / f"eda_{name}_roc_enrichment.csv", index=False)
    H.insert(0, "variant", variant)
    H.to_csv(out / "eda_feature_heterogeneity_across_clusters.csv", index=False)
    X.to_csv(out / "neuron_roc_features.csv", index=False)

    K = int(neurons.cluster_label.max()) + 1
    clusters = list(range(K))
    fams = sorted(fam.family.unique())
    heatmap(Ec, clusters, feats_sig + feats_lab, f"{variant}: ROC significance by rastermap cluster "
            "(□ = enriched vs rest, Fisher BH-FDR<0.05)", out / "eda_cluster_roc_heatmap", "Rastermap cluster")
    heatmap(Ec, clusters, feats_dir, f"{variant}: ROC significance by direction", out / "eda_cluster_roc_direction_heatmap",
            "Rastermap cluster")
    heatmap(Em, fams, feats_sig + feats_lab, f"{variant}: ROC significance by metacluster", out / "eda_metacluster_roc_heatmap",
            "Metacluster (family)")

    fig, ax = plt.subplots(figsize=(6, 0.16 * len(H) + 1), dpi=200)
    ax.barh(range(len(H)), H.cramers_v, color=np.where(H.p_fdr < FDR, "k", "0.7"))
    ax.set_yticks(range(len(H))); ax.set_yticklabels(H.feature, fontsize=5); ax.invert_yaxis()
    ax.set_xlabel("Cramér's V across clusters (black: chi-square BH-FDR<0.05)")
    ax.set_title(f"{variant}: which ROC features differ between clusters", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"eda_feature_heterogeneity.{ext}", bbox_inches="tight")
    plt.close(fig)

    # 4. columns for the final figure: sensory / motor / decision anchors + gated labels,
    #    plus the most cluster-heterogeneous remaining significance features
    anchors = [  # sensory (passive, active) | modality | choice / lick | learning
        "whisker_passive_pre__sig", "whisker_passive_post__sig", "auditory_passive_pre__sig",
        "auditory_passive_post__sig", "whisker_active__sig", "auditory_active__sig",
        "wh_vs_aud_active__pos", "wh_vs_aud_active__neg",
        "whisker_choice__sig", "whisker_hit_vs_cr__sig", "spontaneous_licks__sig",
        "whisker_pre_vs_post_learning__sig"]
    top = [f for f in H[(H.p_fdr < FDR) & H.feature.str.endswith("__sig")].feature if f not in anchors][:4]
    col_feats = [f for f in anchors if f in all_feats] + LABELS + top
    frac_tab = {f: dict(zip(g.group, g.frac)) for f, g in Ec.groupby("feature")}
    # gated in the column figures: fraction of each cluster's DECISION neurons (NaN if < GATED_MIN_DECISION)
    n_dec = Xm.groupby("cluster_label").decision.sum()
    n_gat = Xm.groupby("cluster_label").gated_decision.sum()
    frac_tab["gated_decision"] = {int(k): (n_gat[k] / n_dec[k] if n_dec[k] >= GATED_MIN_DECISION else np.nan)
                                  for k in n_dec.index}
    json.dump(dict(variant=variant, columns=col_feats, anchors=anchors, data_driven_top=top,
                   metacluster_csv=str(fam_csv), n_neurons=int(len(X)), n_roc_matched=int(X.roc_matched.sum())),
              open(out / "figure_columns.json", "w"), indent=2)

    # 5. figures: neuron-level CV rastermap matrix, and cluster-mean matrix
    d = np.load(method_dir / "rastermap_results_cv.npz", allow_pickle=True)
    Xe, isort, labels = d["X"], d["isort"], d["cluster_labels"].astype(int)
    nb = list(d["n_bins_list"])
    roc_columns_figure(Xe[isort], labels[isort], K, families, col_feats, frac_tab, nb, cfg,
                       f"{variant}: CV population matrix (even trials, rastermap order), n={len(Xe)}",
                       out / "fig_rastermap_roc_columns", per_neuron_rows=True)
    M = np.vstack([Xe[labels == k].mean(0) for k in range(K)])
    roc_columns_figure(M, np.arange(K), K, families, col_feats, frac_tab, nb, cfg,
                       f"{variant}: cluster-mean CV matrix ({K} clusters)",
                       out / "fig_clustermean_roc_columns", per_neuron_rows=False)

    # 6. mean ROC selectivity (continuous, 2*(AUC-0.5)) per cluster / metacluster
    Sc = selectivity_summary(Xm, "cluster_label", types)
    Sm = selectivity_summary(Xm, "metacluster", types)
    for df, name in [(Sc, "cluster"), (Sm, "metacluster")]:
        df.insert(0, "variant", variant)
        df.to_csv(out / f"eda_{name}_mean_selectivity.csv", index=False)
    sel_types = [t for t in [  # same ordering as the fraction columns, continuous version
        "whisker_passive_pre", "whisker_passive_post", "auditory_passive_pre", "auditory_passive_post",
        "whisker_active", "auditory_active", "wh_vs_aud_active", "whisker_choice", "auditory_choice",
        "choice", "whisker_hit_vs_cr", "whisker_hit_vs_spontaneous", "spontaneous_licks",
        "spontaneous_licks_vs_cr", "whisker_sensory", "whisker_pre_vs_post_learning",
        "auditory_pre_vs_post_learning", "baseline_choice"] if t in types]
    selectivity_line_panels(Sc, K, families, types,
                            f"{variant}: mean ROC selectivity per rastermap cluster (background = metacluster)",
                            out / "eda_cluster_mean_selectivity_lines")
    selectivity_heatmap(Sc, clusters, types, f"{variant}: mean ROC selectivity by rastermap cluster",
                        out / "eda_cluster_mean_selectivity_heatmap", "Rastermap cluster")
    selectivity_heatmap(Sm, fams, types, f"{variant}: mean ROC selectivity by metacluster",
                        out / "eda_metacluster_mean_selectivity_heatmap", "Metacluster (family)")
    sel_tab = {f"{t}__sel": dict(zip(g.group, g.mean_sel)) for t, g in Sc.groupby("analysis_type")}
    sel_cols = [f"{t}__sel" for t in sel_types]
    roc_columns_figure(Xe[isort], labels[isort], K, families, sel_cols, sel_tab, nb, cfg,
                       f"{variant}: CV population matrix (even trials, rastermap order), n={len(Xe)}",
                       out / "fig_rastermap_roc_selectivity_columns", per_neuron_rows=True, mode="selectivity")
    roc_columns_figure(M, np.arange(K), K, families, sel_cols, sel_tab, nb, cfg,
                       f"{variant}: cluster-mean CV matrix ({K} clusters)",
                       out / "fig_clustermean_roc_selectivity_columns", per_neuron_rows=False, mode="selectivity")

    # console summary: top enriched clusters for decision / gated labels
    for f in LABELS + ["sensory_passive_any"]:
        t = Ec[(Ec.feature == f) & Ec.enriched].sort_values("p_fdr").head(6)
        print(f"  {f}: overall {Xm[f].mean():.3f}; enriched clusters "
              f"{[(int(r.group), round(r.frac, 2), int(r.n_mice_with_sig)) for r in t.itertuples()]}", flush=True)
    print(f"  saved → {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    root = VARIANTS_ROOT / "_test" if a.test else VARIANTS_ROOT
    for md in find_method_dirs(root):
        if a.variants and not any(v in str(md) for v in a.variants):
            continue
        try:
            run(md)
        except Exception:
            import traceback
            print(f"FAILED {md}\n{traceback.format_exc()}", flush=True)
    print("ALL DONE", flush=True)
