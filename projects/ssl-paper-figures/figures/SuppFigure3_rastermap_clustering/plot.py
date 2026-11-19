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


def _square(n):
    M = np.full((n, n), np.nan)
    return M


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
    top.set_xlim(0, 10 * K)
    top.set_yticks([0.4, 0.8], ["0.4", "0.8"])
    top.tick_params(labelsize=4.5, length=1.5)
    top.set_ylabel("1 − r", fontsize=5, labelpad=1)
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
    ax.set_yscale("log")
    ax.set_xlabel("Cut (1 − r)")
    ax.set_ylabel("Metaclusters")
    ax.set_xlim(0, 1)


def plot_meta_sizes(ax, df):
    d = df.sort_values("metacluster")
    x = np.arange(len(d))
    ax.bar(x, d.n_neurons, color=[kind_color(m, k) for m, k in zip(d.modality, d.kind)], width=0.8, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("Neurons")
    ax.set_xticks(x[::4], d.metacluster.to_numpy()[::4])
    ax.set_xlabel("Metacluster (rastermap order)")
    ax.set_xlim(-0.8, len(d) - 0.2)
    t = ax.twinx()
    t.plot(x, d.n_mice, "o", ms=1.6, mfc="white", mec="k", mew=0.4)
    t.set_ylabel("Mice (≥ 3 neurons)", fontsize=5)
    t.set_ylim(0, d.n_mice.max() * 1.08)
    t.spines["right"].set_visible(True)
    t.tick_params(labelsize=5)
    import matplotlib.patches as mp
    hs = [mp.Patch(color=c, label=l) for l, c in (("Whisker", W), ("Auditory", A), ("Multisensory", "#3a9e6f"),
                                                  ("Lick", LICK), ("Weak", "0.75"))]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=4.5, handlelength=0.8,
              columnspacing=0.6, handletextpad=0.3, borderaxespad=0.2)


def plot_cv(ax, df):
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.4, ls="--")
    ax.scatter(df.r_self, df.r_best_other, s=3, c="k", lw=0)
    lo = min(df.r_self.min(), df.r_best_other.min()) - 0.03
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.set_xlabel("r, same cluster (held-out)")
    ax.set_ylabel("r, best other cluster")


def plot_bgmm_contingency(ax, df):
    M = df.pivot(index="cluster", columns="component_rank", values="frac_of_cluster").fillna(0)
    ax.imshow(M.to_numpy(), cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    fam = df.drop_duplicates("cluster").sort_values("cluster").metacluster.to_numpy()
    strip = ax.inset_axes([-0.06, 0, 0.035, 1])
    strip.imshow((fam % 2)[:, None], cmap="Greys", vmin=-1, vmax=3, aspect="auto", interpolation="nearest")
    strip.axis("off")
    ax.set_yticks([])
    ax.set_xticks([0, M.shape[1] - 1], ["1", str(M.shape[1])])
    ax.set_ylabel("Rastermap cluster", labelpad=8)
    ax.set_xlabel("Bayesian GMM component")
    cax = ax.inset_axes([1.03, 0.0, 0.04, 0.35])
    import matplotlib as mpl
    cb = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap="Greys", norm=mpl.colors.Normalize(0, 1)), cax=cax, ticks=[0, 1])
    cb.ax.tick_params(labelsize=4.5, length=1.5)
    cb.set_label("Fraction", fontsize=5, labelpad=0)
    cb.outline.set_linewidth(0.3)


def plot_bgmm_agreement(ax, df):
    levels = ["fine clusters", "metaclusters"]
    for k, metric in enumerate(["AMI", "ARI"]):
        for i, lv in enumerate(levels):
            r = df[(df.level == lv) & (df.metric == metric)].iloc[0]
            x = i + (k - 0.5) * 0.38
            ax.bar(x, r.value, 0.34, color="k" if metric == "AMI" else "0.55", lw=0)
            ax.plot([x - 0.17, x + 0.17], [r.null_p95] * 2, color=light("k", 0.5), lw=0.8)
    ax.set_xticks([0, 1], ["Fine", "Meta"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Agreement with rastermap")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color="k", label="AMI"), mp.Patch(color="0.55", label="ARI")], loc="upper left",
              fontsize=4.5, handlelength=0.8)


def plot_bgmm_matched(ax, df):
    bins = np.linspace(min(0, df.r_best.min()), 1, 26)
    ax.hist(df[df.kind == "null"].r_best, bins, color="0.75", density=True, lw=0, edgecolor="none", label="Permuted")
    ax.hist(df[df.kind == "real"].r_best, bins, histtype="step", color="k", density=True, lw=0.7, label="BGMM")
    ax.set_xlabel("r with best rastermap cluster")
    ax.set_ylabel("Density (components)")
    ax.legend(loc="upper left", fontsize=4.5, handlelength=0.8)


def plot_latency(ax, df):
    for at, c in (("whisker_active", W), ("auditory_active", A)):
        d = df[df.analysis_type == at]
        ax.scatter(d.cluster, d.latency_ms, s=2.5, color=c, lw=0, label=at.split("_")[0].capitalize())
    ax.set_xlabel("Rastermap cluster")
    ax.set_ylabel("Onset latency (ms)")
    ax.set_xlim(-2, 101)
    ax.legend(loc="upper right", fontsize=4.5, handlelength=0.6, markerscale=1.5, borderaxespad=0.1)


def plot_hierarchy(ax, df):
    ax.errorbar(df.cluster, df.hierarchy, yerr=df["sem"], fmt="o", ms=1.4, color="k", lw=0.4, elinewidth=0.3)
    ax.set_xlabel("Rastermap cluster")
    ax.set_ylabel("Hierarchy score")
    ax.set_xlim(-2, 101)
