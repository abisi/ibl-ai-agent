"""SuppFigure3_rastermap_clustering -- one plot_<key>(ax, df) per panel; reads only the panel CSV (panel a is an image)."""
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from paperstyle import light

W, A, LICK, NEUTRAL = "#f7b519", "#2c2cdb", "#9b30d9", "0.55"


def kind_color(modality, kind):
    """metacluster response type -> colour (task colours; lick = ROC lick colour; weak = grey)"""
    if kind == "weak":
        return "0.75"
    if kind == "motor":
        return LICK
    return {"Whisker": W, "Auditory": A}.get(modality, "#3a9e6f")   # multisensory: green


def plot_meta_matrix(ax, df):
    r = df[df.kind == "r"]
    K = int(r.i.max()) + 1
    M = np.full((K, K), np.nan)
    M[r.i.astype(int), r.j.astype(int)] = r.value
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    fam = r[r.j == 0].sort_values("i").fam_i.to_numpy()
    edges = np.r_[0, np.flatnonzero(np.diff(fam)) + 1, K] - 0.5
    for a, b in zip(edges[:-1], edges[1:]):
        ax.add_patch(__import__("matplotlib").patches.Rectangle((a, a), b - a, b - a, fill=False, lw=0.4, ec="k"))
    ax.set_xticks([0, K - 1], ["1", str(K)])
    ax.set_yticks([0, K - 1], ["1", str(K)])
    ax.set_xlabel("Rastermap cluster")
    ax.set_ylabel("Rastermap cluster")
    Z = df[df.kind == "Z"][["i", "j", "value", "n"]].to_numpy(float)
    top = ax.inset_axes([0, 1.02, 1, 0.22])
    dendrogram(Z, ax=top, color_threshold=0, above_threshold_color="k", no_labels=True,
               link_color_func=lambda k: "k")
    top.axhline(0.40, color="#e8222e", lw=0.5, ls="--")
    top.text(10 * K, 0.38, "cut", color="#e8222e", fontsize=4.5, ha="left", va="center")
    top.set_xlim(0, 10 * K)
    top.set_yticks([0.4, 0.8], ["0.4", "0.8"])
    top.tick_params(labelsize=4.5, length=1.5)
    top.set_ylabel("1 − r", fontsize=5, labelpad=1)
    top.text(0.0, 1.0, "Dendrogram (average linkage)", transform=top.transAxes, fontsize=4.5, ha="left", va="bottom")
    for s in ("top", "right", "bottom"):
        top.spines[s].set_visible(False)
    top.set_xticks([])
    cax = ax.inset_axes([1.02, 0.0, 0.03, 0.3])
    cb = ax.figure.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=4.5, length=1.5)
    cb.set_label("r", fontsize=5, labelpad=0)
    cb.outline.set_linewidth(0.3)


def plot_meta_cut(ax, df):
    ax.plot(df.cut, df.n_metaclusters, color="k", lw=0.8)
    ax.axvline(0.40, color="#e8222e", lw=0.5, ls="--")
    n_cut = int(df.loc[(df.cut - 0.40).abs().idxmin(), "n_metaclusters"])
    ax.text(0.43, 0.97, f"Cut used:\n{n_cut} meta-\nclusters", transform=ax.get_xaxis_transform(), color="#e8222e",
            fontsize=4.5, ha="left", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("Dendrogram cut\n(1 − Pearson r)")
    ax.set_ylabel("Number of metaclusters")
    ax.set_xlim(0, 1)


def plot_meta_sizes(ax, df):
    d = df.sort_values("metacluster")
    x = np.arange(len(d))
    ax.bar(x, d.n_neurons, color=[kind_color(m, k) for m, k in zip(d.modality, d.kind)], width=0.8, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("Neurons per metacluster (bars)")
    ax.set_xticks(x[::4], d.metacluster.to_numpy()[::4])
    ax.set_xlabel("Metacluster (rastermap order)")
    ax.set_xlim(-0.8, len(d) - 0.2)
    t = ax.twinx()
    t.plot(x, d.n_mice, "o", ms=1.6, mfc="white", mec="k", mew=0.4)
    t.set_ylabel("Mice with ≥ 3 neurons (○)", fontsize=5)
    t.set_ylim(0, d.n_mice.max() * 1.08)
    t.spines["right"].set_visible(True)
    t.tick_params(labelsize=5)
    import matplotlib.patches as mp
    hs = [mp.Patch(color=c, label=l) for l, c in (("Whisker", W), ("Auditory", A), ("Multisensory", "#3a9e6f"),
                                                  ("Lick", LICK), ("Weak", "0.75"))]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=4.5, handlelength=0.8,
              columnspacing=0.6, handletextpad=0.3, borderaxespad=0.2, title="Metacluster response type",
              title_fontsize=4.5, alignment="left", frameon=False)


def _self_vs_other(ax, df, xlabel):
    """d / l: one dot per cluster; below y = x its PSTH is closer to its own replicate than to any other cluster."""
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.4, ls="--")
    ax.scatter(df.r_self, df.r_best_other, s=3, c="k", lw=0)
    lo = min(df.r_self.min(), df.r_best_other.min()) - 0.03
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.text(0.58, 0.66, "y = x", transform=ax.transAxes, rotation=45, fontsize=4.5, color="0.5", ha="center",
            va="bottom", rotation_mode="anchor")
    frac = (df.r_self > df.r_best_other).mean()
    ax.text(0.03, 0.97, f"{frac:.0%} of clusters below y = x:\ncloser to their own replicate\nthan to any other cluster",
            transform=ax.transAxes, fontsize=4.5, ha="left", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pearson r with the most\nsimilar other cluster")


def plot_cv(ax, df):
    _self_vs_other(ax, df, "Pearson r with itself,\nheld-out (even) trials")


def _pstr(p):
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _stat_text(ax, lines, loc=(0.97, 0.97), ha="right", fontsize=4.5):
    for k, (txt, c) in enumerate(lines):
        ax.text(loc[0], loc[1] - k * 0.075, txt, transform=ax.transAxes, ha=ha, va="top", fontsize=fontsize, color=c)


def plot_gmm_family_psth(ax, df):
    """two heatmaps, same rows (metaclusters, rastermap order): rastermap metacluster PSTH | matched GMM family PSTH;
    each row scaled by its own maximum |value| (shape comparison); r of the pair on the right"""
    import matplotlib as mpl
    ms = np.sort(df.metacluster.unique())
    ncol = int(df.col.max()) + 1
    mats = {}
    for m in ("rastermap", "gmm"):
        M = df[df.method == m].pivot(index="metacluster", columns="col", values="value").reindex(ms).to_numpy()
        mats[m] = M / np.nanmax(np.abs(M), axis=1, keepdims=True)
    gap = max(1, ncol // 25)
    img = np.hstack([mats["rastermap"], np.full((len(ms), gap), np.nan), mats["gmm"]])
    cm = mpl.colormaps["RdBu_r"].copy()
    cm.set_bad("white")
    ax.imshow(img, cmap=cm, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    edges = np.flatnonzero(np.diff(df[(df.method == "rastermap") & (df.metacluster == ms[0])].cond.to_numpy())) + 0.5
    for x0 in (0, ncol + gap):
        for e in edges:
            ax.axvline(x0 + e, color="k", lw=0.25)
    ax.set_xticks([ncol / 2, ncol + gap + ncol / 2], ["Rastermap metacluster", "Matched GMM family"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(np.arange(len(ms)), [str(m) for m in ms], fontsize=4)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_ylabel("Metacluster (rastermap order)", labelpad=9)
    first = df.drop_duplicates("metacluster").set_index("metacluster").reindex(ms)
    strip = ax.inset_axes([-0.02, 0, 0.014, 1])
    strip.imshow(np.array([mpl.colors.to_rgb(kind_color(mo, k)) for mo, k in zip(first.modality, first.kind)])[:, None],
                 aspect="auto", interpolation="nearest")
    strip.axis("off")
    for i, r in enumerate(first.r):
        ax.text(1.01, i, f"{r:.2f}", transform=ax.get_yaxis_transform(), fontsize=4, va="center", ha="left",
                color="k" if r > 0.8 else "0.55")
    ax.text(1.01, 1.0, "r", transform=ax.transAxes, fontsize=4.5, ha="left", va="bottom")
    for s in ax.spines.values():
        s.set_visible(False)


def plot_gmm_family_match(ax, df):
    c = [kind_color(m, k) for m, k in zip(df.modality, df.kind)]
    ax.scatter(df.n_neurons, df.r, s=7, c=c, lw=0.3, edgecolors="k")
    ax.set_xscale("log")
    ax.axhline(0.8, color="0.6", lw=0.4, ls="--")
    w = df.n_neurons / df.n_neurons.sum()
    _stat_text(ax, [(f"Weighted r = {(w * df.r).sum():.2f}", "k"),
                    (f"{w[df.r > 0.8].sum():.0%} of neurons", "k"), ("in r > 0.8", "k")],
               loc=(0.98, 0.3), ha="right")
    ax.set_ylim(min(-0.05, df.r.min() - 0.05), 1.02)
    ax.set_xlabel("Neurons in metacluster")
    ax.set_ylabel("Pearson r with the matched\nGMM family (mean PSTHs)")


def plot_latency(ax, df):
    lines = []
    for at, c in (("whisker_active", W), ("auditory_active", A)):
        d = df[df.analysis_type == at].sort_values("cluster")
        ax.scatter(d.cluster, d.latency_ms, s=2.5, color=c, lw=0, label=at.split("_")[0].capitalize())
        ax.plot(d.cluster, d.fit, color=c, lw=0.8)
        lines.append((f"{at.split('_')[0].capitalize()} trials:", c))
        lines.append((f"  ρ = {d.rho.iloc[0]:.2f}, {_pstr(d.p_perm.iloc[0])}", c))
    ax.set_xlabel("Rastermap cluster (order)")
    ax.set_ylabel("Median onset latency of\nresponsive neurons (ms)")
    ax.set_xlim(-2, 101)
    ax.set_ylim(0, 1.5 * df.latency_ms.max())
    _stat_text(ax, lines, loc=(0.98, 0.99), fontsize=4.3)


def plot_hierarchy(ax, df):
    _anatomy_vs_order(ax, df, "Hierarchy score, mean ± s.e.m.\n(Harris et al., 2019)")


def plot_liu_projection(ax, df):
    _anatomy_vs_order(ax, df, "wS1/wS2 projection strength,\nlog10, mean ± s.e.m. (Liu et al., 2024)")


def _anatomy_vs_order(ax, df, ylabel):
    """per-cluster mean anatomy score vs rastermap position, whisker- / auditory-dominated clusters"""
    lines = []
    for g, c in (("whisker", W), ("auditory", A)):
        d = df[df.dominant == g].sort_values("cluster")
        ax.errorbar(d.cluster, d.hierarchy, yerr=d["sem"], fmt="o", ms=1.4, color=c, lw=0.4, elinewidth=0.3,
                    label=f"{g.capitalize()}-dom.")
        ax.plot(d.cluster, d.fit, color=c, lw=0.8)
        lines.append((f"{g.capitalize()}-dominated clusters:", c))
        lines.append((f"  ρ = {d.rho.iloc[0]:.2f}, {_pstr(d.p_perm.iloc[0])}", c))
    ax.set_xlabel("Rastermap cluster (order)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-2, 101)
    lo, hi = (df.hierarchy - df["sem"]).min(), (df.hierarchy + df["sem"]).max()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.6 * (hi - lo))
    _stat_text(ax, lines, loc=(0.02, 0.99), ha="left", fontsize=4.3)


def _embed_axes(ax, xl, yl, df=None):
    if df is not None:   # a few isolated neurons far from the main cloud would shrink it
        for lim, v in ((ax.set_xlim, df.u1), (ax.set_ylim, df.u2)):
            lo, hi = np.percentile(v, [0.1, 99.9])
            pad = 0.04 * (hi - lo)
            lim(lo - pad, hi + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xl, labelpad=2)
    ax.set_ylabel(yl, labelpad=2)


def plot_umap_clusters(ax, df):
    sc = ax.scatter(df.u1, df.u2, c=df.cluster, cmap="turbo", vmin=0, vmax=99, s=0.15, lw=0, alpha=0.6,
                    rasterized=True)
    _embed_axes(ax, "UMAP 1", "UMAP 2", df)
    cax = ax.inset_axes([0.62, 0.1, 0.32, 0.035])
    cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 99])
    cb.set_ticklabels(["1", "100"])
    cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    cb.ax.set_title("Rastermap cluster", fontsize=4.5, pad=1.5)
    cb.outline.set_linewidth(0.3)
    cb.solids.set_alpha(1)
    _label_metaclusters(ax, df)


def _label_metaclusters(ax, df, min_neurons=1000):
    """metacluster number at its most central neuron (closest to the metacluster median), large metaclusters only"""
    import matplotlib.patheffects as pe
    for m, d in df.groupby("metacluster"):
        if len(d) < min_neurons:
            continue
        med = d[["u1", "u2"]].median().to_numpy()
        x, y = d[["u1", "u2"]].to_numpy()[np.argmin(((d[["u1", "u2"]].to_numpy() - med) ** 2).sum(1))]
        ax.text(x, y, str(m), fontsize=5, ha="center", va="center", color="k",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
    ax.text(0.99, 0.5, "Numbers:\nmetaclusters", transform=ax.transAxes, fontsize=4.3, ha="right", va="top")


def plot_umap_metaclusters(ax, df):
    ax.scatter(df.u1, df.u2, c=df.color, s=0.15, lw=0, alpha=0.6, rasterized=True)
    _embed_axes(ax, "UMAP 1", "UMAP 2", df)
    _label_metaclusters(ax, df)


def plot_umap_areas(ax, df):
    other = df.area_group == "Other"
    ax.scatter(df.u1[other], df.u2[other], c=df.color[other], s=0.15, lw=0, alpha=0.4, rasterized=True)
    ax.scatter(df.u1[~other], df.u2[~other], c=df.color[~other], s=0.15, lw=0, alpha=0.6, rasterized=True)
    _embed_axes(ax, "UMAP 1", "UMAP 2", df)
    import matplotlib.patches as mp
    g = df.groupby("area_group").agg(color=("color", "first"), n=("u1", "size")).sort_values("n", ascending=False)
    g = g.loc[[a for a in g.index if a != "Other"] + (["Other"] if "Other" in g.index else [])]
    ax.legend(handles=[mp.Patch(color=c, label=a) for a, c in zip(g.index, g.color)], loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=4, handlelength=0.7, handleheight=0.7, labelspacing=0.15,
              handletextpad=0.3, borderaxespad=0, frameon=False)


COND_NAMES = {"Whisker pre": "Whisker, passive pre", "Whisker post": "Whisker, passive post",
              "Auditory pre": "Auditory, passive pre", "Auditory post": "Auditory, passive post",
              "Whisker miss": "Whisker miss", "Whisker hit": "Whisker hit", "Auditory hit": "Auditory hit"}


def _pc_labels(ax, df):
    v1 = f" ({df.pc1_var.iloc[0]:.0%})" if "pc1_var" in df else ""
    v2 = f" ({df.pc2_var.iloc[0]:.0%})" if "pc2_var" in df else ""
    ax.set_xlabel(f"PC 1{v1}")
    ax.set_ylabel(f"PC 2{v2}")
    ax.set_xticks([])
    ax.set_yticks([])


def _shaded(col, frac):
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(col))
    shade = 0.15 + 0.85 * np.asarray(frac)[:, None]
    return (1 - shade) + base * shade


def plot_pca_flow(ax, df):
    import matplotlib.colors as mc
    for (c, lab, col), d in df.groupby(["cond_index", "condition", "color"], sort=True):
        d = d.sort_values("t")
        base = np.array(mc.to_rgb(col))
        shade = 0.15 + 0.85 * d.frac.to_numpy()[:, None]
        ax.plot(d.pc1, d.pc2, color=col, lw=0.5, alpha=0.6, zorder=1)
        ax.scatter(d.pc1, d.pc2, s=1.5, color=(1 - shade) + base * shade, lw=0, zorder=2)
        ax.plot([], [], color=col, lw=1.2, label=COND_NAMES.get(lab, lab))
        on = d.iloc[np.argmin(np.abs(d.t.to_numpy()))]
        ax.plot(on.pc1, on.pc2, marker="o", ms=2, mfc="white", mec=col, mew=0.5, zorder=3)
    _pc_labels(ax, df)
    ax.legend(loc="upper right", fontsize=4, handlelength=0.8, borderaxespad=0.1, labelspacing=0.2,
              handletextpad=0.3, frameon=False)
    ax.text(0.03, 0.98, "○ stimulus onset\nlight → dark: time", transform=ax.transAxes, fontsize=4.3, ha="left",
            va="top")


def plot_pca_flow_clusters(ax, df):
    for _, d in df.groupby("cond_index"):
        d = d.sort_values("t")
        ax.plot(d.pc1, d.pc2, color="0.75", lw=0.4, zorder=1)
        on = d.iloc[np.argmin(np.abs(d.t.to_numpy()))]   # user 2026-11-23: e introduces the trajectories (d removed)
        ax.plot(on.pc1, on.pc2, marker="o", ms=2, mfc="white", mec="0.3", mew=0.5, zorder=3)
    sc = ax.scatter(df.pc1, df.pc2, c=df.weighted_cluster, cmap="turbo", vmin=0, vmax=99, s=1.5, lw=0, zorder=2)
    _pc_labels(ax, df)
    ax.text(0.03, 0.98, "○ stimulus onset", transform=ax.transAxes, fontsize=4.3, ha="left", va="top")
    cax = ax.inset_axes([0.62, 0.92, 0.32, 0.035])
    cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 99])
    cb.set_ticklabels(["1", "100"])
    cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    cb.set_label("Activity-weighted mean\nrastermap cluster of the bin", fontsize=4.3, labelpad=1)
    cb.outline.set_linewidth(0.3)


def plot_pca_flow_by_type(ax, df):
    """one small panel per trial type: that type's trajectory (time-shaded), all other types grey (as
    run_clustering_new.fig8_flow_umap small multiples); shared limits"""
    ax.axis("off")
    conds = df.drop_duplicates("cond_index").sort_values("cond_index")
    n, gap = len(conds), 0.035
    w = (1 - gap * (n - 1)) / n
    pad_x = 0.05 * (df.pc1.max() - df.pc1.min())
    pad_y = 0.05 * (df.pc2.max() - df.pc2.min())
    for j, (_, r) in enumerate(conds.iterrows()):
        a = ax.inset_axes([j * (w + gap), 0, w, 1])
        other = df[df.cond_index != r.cond_index]
        a.scatter(other.pc1, other.pc2, s=0.6, color="0.85", lw=0, zorder=0)
        d = df[df.cond_index == r.cond_index].sort_values("t")
        a.plot(d.pc1, d.pc2, color=r.color, lw=0.5, alpha=0.7, zorder=1)
        a.scatter(d.pc1, d.pc2, s=1.2, color=_shaded(r.color, d.frac), lw=0, zorder=2)
        on = d.iloc[np.argmin(np.abs(d.t.to_numpy()))]
        a.plot(on.pc1, on.pc2, marker="o", ms=2, mfc="white", mec=r.color, mew=0.5, zorder=3)
        a.set_xlim(df.pc1.min() - pad_x, df.pc1.max() + pad_x)
        a.set_ylim(df.pc2.min() - pad_y, df.pc2.max() + pad_y)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(COND_NAMES.get(r.condition, r.condition), fontsize=5, color=r.color, pad=2)
        a.set_xlabel("PC 1", fontsize=5, labelpad=1)
        if j == 0:
            a.set_ylabel("PC 2", fontsize=5, labelpad=1)
            a.text(0.97, 0.97, "grey: other\ntrial types", transform=a.transAxes, fontsize=4.3, ha="right",
                   va="top", color="0.45")


def plot_area_specialization(ax, df):
    q = df.drop_duplicates("area").sort_values(["area_group", "kl_bits"], ascending=[True, False])
    areas = q.area.to_list()
    ks = np.sort(df.metacluster.unique())
    M = df.pivot(index="area", columns="metacluster", values="log2_enrichment").loc[areas, ks].to_numpy()
    ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(areas)), areas, fontsize=4)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xticks(range(0, len(ks), 4), ks[::4])
    ax.set_xlabel("Metacluster (rastermap order)")
    ax.text(-0.005, 1.045, "Response type", transform=ax.transAxes, fontsize=4.5, ha="right", va="center")
    ax.text(-0.016, -0.012, "Area\ngroup", transform=ax.transAxes, fontsize=4, ha="center", va="top")
    strip = ax.inset_axes([-0.025, 0, 0.018, 1])           # area-group colour per row
    import matplotlib.colors as mc
    strip.imshow(np.array([mc.to_rgb(c) for c in q.color])[:, None, :], aspect="auto", interpolation="nearest")
    strip.axis("off")
    top = ax.inset_axes([0, 1.02, 1, 0.05])                # metacluster response type
    top.imshow(np.array([mc.to_rgb(kind_color(m, k)) for m, k in
                         df.drop_duplicates("metacluster").set_index("metacluster").loc[ks, ["modality", "kind"]]
                         .itertuples(index=False)])[None], aspect="auto", interpolation="nearest")
    top.axis("off")
    side = ax.inset_axes([1.02, 0, 0.16, 1])               # specialization (KL, bits) vs within-mouse shuffle
    y = np.arange(len(areas))
    side.barh(y, q.kl_bits, height=0.75, color=["k" if p < 0.05 else "0.7" for p in q.p_perm], lw=0)
    side.scatter(q.null_p95, y, marker="|", s=6, color="#e8222e", lw=0.6)
    side.set_ylim(len(areas) - 0.5, -0.5)
    side.set_yticks([])
    side.spines["left"].set_visible(False)
    side.set_xlabel("Specialization:\nKL divergence from\ndataset (bits)", fontsize=4.5, labelpad=1)
    side.tick_params(labelsize=4.5, length=1.5)
    import matplotlib.patches as mp
    import matplotlib.lines as ml
    side.legend(handles=[mp.Patch(color="k", label="p < 0.05"), mp.Patch(color="0.7", label="n.s."),
                         ml.Line2D([], [], color="#e8222e", marker="|", lw=0, ms=4, label="Shuffle, 95th pct.")],
                loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=4.2, handlelength=0.8, borderaxespad=0.2,
                labelspacing=0.2, frameon=False)
    cax = ax.inset_axes([1.22, 0.0, 0.025, 0.3])
    import matplotlib as mpl
    cb = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-3, 3)), cax=cax,
                            ticks=[-3, 0, 3])
    cb.ax.tick_params(labelsize=4.5, length=1.5)
    cb.set_label("log2 enrichment\nvs dataset", fontsize=4.5, labelpad=1)
    cb.outline.set_linewidth(0.3)


def plot_area_ami(ax, df):
    fac = ["area", "area_group", "mouse", "reward_group"]
    names = ["Area", "Group", "Mouse", "R+/R−"]
    w = 0.38
    for k, (lv, c) in enumerate((("fine clusters", "k"), ("metaclusters", "0.6"))):
        d = df[df.level == lv].set_index("factor").loc[fac]
        x = np.arange(len(fac)) + (k - 0.5) * w
        ax.bar(x, d.AMI, w * 0.9, color=c, lw=0, label="Fine" if k == 0 else "Meta")
        for xi, v in zip(x, d.null_p95):
            ax.plot([xi - w * 0.45, xi + w * 0.45], [v, v], color="#e8222e", lw=0.7)
    ax.set_xticks(range(len(fac)), names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("AMI with cluster labels")
    ax.legend(loc="upper right", fontsize=4.5, handlelength=0.8, frameon=False)


ROBUST_LABELS = {   # variant compared with the main run, spelled out (user 2026-11-20); * = vs the good-unit run
    "Good units only": "Good units only (no multi-units)",
    "All learner categories": "All mice (incl. non-learners)",
    "No passive conditions": "Without passive-trial PSTHs",
    "No whisker-reward bins": "Without whisker post-reward bins",
    "Uncorrected lick times": "Uncorrected lick times",
    "No firing-rate threshold": "No firing-rate threshold*",
    "Locality 0 vs 0.25": "Rastermap locality 0*",
    "Locality 0.75 vs 0.25": "Rastermap locality 0.75*",
}


def plot_robustness(ax, df):
    y = np.arange(len(df))
    ax.barh(y - 0.2, df.AMI, 0.38, color="k", lw=0, label="AMI of cluster labels")
    ax.barh(y + 0.2, df.spearman_position, 0.38, color="0.6", lw=0, label="Spearman ρ of rastermap position")
    ax.axvline(df.gmm_ami.iloc[0], color="#e8222e", lw=0.6, ls="--")
    ax.text(df.gmm_ami.iloc[0] + 0.015, -0.65, "AMI, rastermap vs GMM", color="#e8222e", fontsize=4.3, ha="left",
            va="bottom")
    ax.set_yticks(y, [ROBUST_LABELS.get(c, c) for c in df.comparison], fontsize=4.3)
    ax.set_ylim(len(df) - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Agreement with the main run\n(*: with the good-unit run, locality 0.25)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.06), ncol=1, fontsize=4.3, handlelength=0.8, frameon=False,
              borderaxespad=0, labelspacing=0.2)


def plot_mouse_split(ax, df):
    _self_vs_other(ax, df, "Pearson r with itself,\nother half of the mice")


def plot_mouse_mixing(ax, df):
    d = df.sort_values("cluster")
    ax.fill_between(d.cluster, d.null_eff_p5, d.null_eff_p95, color="0.9", lw=0, step="mid",
                    label="Random clusters, same size")
    ax.fill_between(d.cluster, d.area_null_eff_p5, d.area_null_eff_p95, color="#9ecae1", lw=0, step="mid", alpha=0.8,
                    label="Labels permuted within area")
    ax.scatter(d.cluster, d.eff_mice, s=3, c="k", lw=0, label="Rastermap clusters", zorder=3)
    ax.set_xlim(-2, 101)
    ax.set_ylim(0, None)
    ax.set_xlabel("Rastermap cluster (order)")
    ax.set_ylabel("Effective number of mice\n(exp. of mouse entropy)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=4.3, handlelength=0.8, frameon=False, markerscale=1.5,
              borderaxespad=0, labelspacing=0.2, title="Bands: 5th–95th pct.", title_fontsize=4.3, alignment="left")


def plot_mouse_share(ax, df):
    bins = np.linspace(0, max(0.6, df.max_share.max()), 25)
    ax.hist(df.null_max_p95, bins, color="0.8", lw=0, label="Random clusters of the\nsame size, 95th pct.")
    ax.hist(df.max_share, bins, histtype="step", color="k", lw=0.7, label="Rastermap clusters")
    ax.set_xlabel("Largest share of one\nmouse in a cluster")
    ax.set_ylabel("Number of clusters")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=4.3, handlelength=0.8, frameon=False,
              borderaxespad=0, labelspacing=0.2)
