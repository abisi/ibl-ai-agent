"""SuppFigure3_embedding plot functions (read only the panel CSVs)."""
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from scipy.cluster.hierarchy import dendrogram

CUT = 0.40
RED = "#e8222e"
FS = 5.0


def _style(ax):
    ax.tick_params(labelsize=FS, length=1.5, pad=1, width=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)


def plot_schematic(ax, df):
    """pipeline boxes (left) and a thumbnail of the cluster-mean matrix with metacluster strip (right)"""
    ax.axis("off")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    cond = df[df.kind == "cond"].sort_values("i")
    fam = df[df.kind == "fam"].sort_values("i")
    n_cond = len(cond)
    steps = [("Condition PSTHs per neuron", f"{n_cond} conditions: passive W / A pre + post;\nW hit, W miss, A hit (stimulus);\nlicks: spont., W hit, A hit"),
             ("Concatenate", "baseline-subtracted, one\nfeature vector per neuron"),
             ("Rastermap (odd trials)", "locality 0.25, time lag 20,\n100 k-means nodes"),
             ("100 clusters  →  metaclusters", f"neighbours merged at\n1 − r = {CUT:.2f}: {int(fam.value.max())} metaclusters")]
    y = 0.98
    h = 0.205
    for k, (t, s) in enumerate(steps):
        ax.add_patch(mpatches.FancyBboxPatch((0.0, y - h), 0.50, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                             fc="0.95", ec="0.5", lw=0.4))
        ax.text(0.02, y - 0.02, t, fontsize=FS, fontweight="bold", va="top", ha="left")
        ax.text(0.02, y - 0.075, s, fontsize=FS - 0.5, va="top", ha="left", linespacing=1.0, color="0.25")
        if k < len(steps) - 1:
            ax.annotate("", xy=(0.25, y - h - 0.045), xytext=(0.25, y - h - 0.002),
                        arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.3", mutation_scale=4))
        y -= h + 0.047
    th = df[df.kind == "thumb"]
    M = th.pivot(index="i", columns="j", values="value").sort_index().to_numpy()
    vmax = np.nanpercentile(np.abs(M), 98)
    tax = ax.inset_axes([0.62, 0.06, 0.37, 0.86])
    tax.imshow(M, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    cb = th.drop_duplicates("j").sort_values("j").cond.to_numpy()
    for e in np.flatnonzero(np.diff(cb)) + 0.5:
        tax.axvline(e, color="k", lw=0.25)
    tax.set_xticks([]), tax.set_yticks([])
    for s in tax.spines.values():
        s.set_linewidth(0.3)
    sax = ax.inset_axes([0.585, 0.06, 0.03, 0.86])
    sax.imshow(np.array([mpl.colors.to_rgb(c) for c in fam.color])[:, None], aspect="auto", interpolation="nearest")
    sax.axis("off")
    ax.text(0.80, 0.95, "100 clusters × conditions\n(cluster-mean PSTHs)", fontsize=FS - 0.5, ha="center",
            va="bottom", linespacing=1.0)
    ax.annotate("", xy=(0.575, 0.12), xytext=(0.505, 0.12),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.3", mutation_scale=4))


def plot_cv_cluster(ax, df):
    """histograms over the 100 clusters: r with its own held-out replicate vs best r with another cluster"""
    bins = np.linspace(min(0.3, df.r_best_other.min()), 1.0, 36)
    ax.hist(df.r_best_other, bins=bins, color="0.75", lw=0, label="best other cluster")
    ax.hist(df.r_self, bins=bins, histtype="step", color="k", lw=0.8, label="same cluster")
    ax.axvline(np.median(df.r_self), color="k", lw=0.4, ls="--")
    ax.set_xlabel("r, odd-trial vs even-trial\ncluster-mean PSTH", fontsize=FS, labelpad=1)
    ax.set_ylabel("Clusters", fontsize=FS, labelpad=1)
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="upper left", handlelength=1.0)
    ax.text(0.02, 0.62, f"median {np.median(df.r_self):.2f}\n{(df.r_self > df.r_best_other).mean():.0%} closer to own\nreplicate",
            transform=ax.transAxes, fontsize=FS - 0.5, va="top", linespacing=1.0)
    _style(ax)


def plot_cv_neuron(ax, df):
    r = df.r_odd_even.dropna()
    ax.hist(r, bins=np.linspace(-0.2, 1.0, 49), color="0.55", lw=0)
    ax.axvline(r.median(), color="k", lw=0.4, ls="--")
    ax.set_xlabel("r, odd-trial vs even-trial\nneuron PSTH", fontsize=FS, labelpad=1)
    ax.set_ylabel("Neurons", fontsize=FS, labelpad=1)
    ax.text(0.03, 0.97, f"median {r.median():.2f}\nn = {len(r):,}", transform=ax.transAxes, fontsize=FS - 0.5, va="top")
    _style(ax)


def plot_meta_matrix(ax, df):
    r = df[df.kind == "r"]
    K = int(r.i.max()) + 1
    M = np.full((K, K), np.nan)
    M[r.i.astype(int), r.j.astype(int)] = r.value
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    fam = df[df.kind == "fam"].sort_values("i")
    f = fam.fam_i.to_numpy()
    edges = np.r_[0, np.flatnonzero(np.diff(f)) + 1, K] - 0.5
    for a, b in zip(edges[:-1], edges[1:]):
        ax.add_patch(mpatches.Rectangle((a, a), b - a, b - a, fill=False, lw=0.4, ec="k"))
    ax.set_xticks([0, K - 1], ["1", str(K)])
    ax.set_yticks([])
    ax.set_xlabel("Cluster (rastermap order)", fontsize=FS, labelpad=1)
    ax.tick_params(labelsize=FS, length=1.5, width=0.4)
    strip = ax.inset_axes([-0.045, 0, 0.03, 1])
    strip.imshow(np.array([mpl.colors.to_rgb(c) for c in fam.color])[:, None], aspect="auto", interpolation="nearest")
    strip.axis("off")
    for a, b, ff in zip(edges[:-1], edges[1:], f[(edges[:-1] + 0.5).astype(int)]):
        if b - a >= 3:
            ax.text(-0.065, 1 - (a + b) / 2 / K, str(int(ff)), transform=ax.transAxes, fontsize=FS - 0.5, ha="right", va="center")
    ax.text(-0.13, 0.5, "Metacluster", transform=ax.transAxes, rotation=90, fontsize=FS, ha="center", va="center")
    Z = df[df.kind == "Z"][["i", "j", "value", "n"]].to_numpy(float)
    top = ax.inset_axes([0, 1.02, 1, 0.2])
    dendrogram(Z, ax=top, color_threshold=0, above_threshold_color="k", no_labels=True, link_color_func=lambda k: "k")
    for c in top.collections:
        c.set_linewidth(0.4)
    top.axhline(CUT, color=RED, lw=0.5, ls="--")
    top.set_xlim(0, 10 * K)
    top.set_yticks([0.4, 0.8], ["0.4", "0.8"])
    top.tick_params(labelsize=FS - 0.5, length=1.5, width=0.4)
    top.set_ylabel("1 − r", fontsize=FS, labelpad=1)
    for s in ("top", "right", "bottom"):
        top.spines[s].set_visible(False)
    top.spines["left"].set_linewidth(0.4)
    top.set_xticks([])
    cax = ax.inset_axes([1.03, 0.0, 0.03, 0.3])
    cb = ax.figure.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=FS, length=1.5, width=0.4)
    cb.set_label("r", fontsize=FS, labelpad=0)
    cb.outline.set_linewidth(0.3)


def plot_meta_cut(ax, df):
    ax.plot(df.cut, df.n_metaclusters, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Dendrogram cut (1 − r)", fontsize=FS, labelpad=1)
    ax.set_ylabel("Number of metaclusters", fontsize=FS, labelpad=1)
    ax.set_xlim(0, 1)
    ax.axvline(CUT, color=RED, lw=0.5, ls="--")
    n_cut = int(df.loc[(df.cut - CUT).abs().idxmin(), "n_metaclusters"])
    ax.text(CUT + 0.02, 0.97, f"cut used: {n_cut}", transform=ax.get_xaxis_transform(), color=RED, fontsize=FS - 0.5,
            ha="left", va="top")
    _style(ax)
    t = ax.twinx()
    t.plot(df.cut, df.r_within, color="#4a6fa5", lw=0.7, label="within")
    t.plot(df.cut, df.r_between, color="#4a6fa5", lw=0.7, ls="--", label="between")
    t.set_ylabel("Median r between clusters", fontsize=FS, labelpad=1, color="#4a6fa5")
    t.tick_params(labelsize=FS, length=1.5, width=0.4, colors="#4a6fa5")
    t.set_ylim(-0.2, 1.0)
    for s in ("top", "left", "bottom"):
        t.spines[s].set_visible(False)
    t.spines["right"].set_linewidth(0.4)
    t.legend(frameon=False, fontsize=FS - 0.5, loc="lower left", handlelength=1.4, title="metaclusters",
             title_fontsize=FS - 0.5)
