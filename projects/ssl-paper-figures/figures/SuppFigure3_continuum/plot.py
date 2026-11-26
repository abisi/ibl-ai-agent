"""SuppFigure3_continuum plot functions (read only the panel CSVs)."""
import numpy as np

FS = 5.0
RED = "#e8222e"


def _style(ax):
    ax.tick_params(labelsize=FS, length=1.5, pad=1, width=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)


def plot_decay(ax, df):
    ax.fill_between(df.d, df.null_lo, df.null_hi, color="0.85", lw=0, label="random orders")
    ax.fill_between(df.d, df.q25, df.q75, color="#f2b8bc", lw=0)
    ax.plot(df.d, df["median"], color=RED, lw=0.9, label="rastermap order")
    ax.axhline(0, color="0.6", lw=0.3)
    ax.set_xlabel("Distance along the order (clusters)", fontsize=FS, labelpad=1)
    ax.set_ylabel("r between cluster means", fontsize=FS, labelpad=1)
    ax.set_xlim(1, df.d.max())
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="upper right", handlelength=1.0)
    _style(ax)


def plot_adjacent(ax, df):
    null = df[df.kind == "null_mean"].value
    obs = df[df.kind == "obs"].value
    ax.hist(null, bins=40, color="0.7", lw=0)
    ax.axvline(obs.mean(), color=RED, lw=0.9)
    n_ge = int((null >= obs.mean()).sum())
    ptxt = f"p < {1 / (1 + len(null)):.4f}" if n_ge == 0 else f"p = {(1 + n_ge) / (1 + len(null)):.3g}"
    ax.text(obs.mean() - 0.015, ax.get_ylim()[1] * 0.6, f"rastermap\norder\n{obs.mean():.2f}", color=RED,
            fontsize=FS - 0.5, va="top", ha="right")
    ax.text(0.03, 0.97, f"random orders\n(n = {len(null)})\n{ptxt}", transform=ax.transAxes, fontsize=FS - 0.5, va="top")
    ax.set_xlabel("Mean r of neighbouring clusters", fontsize=FS, labelpad=1)
    ax.set_ylabel("Permutations", fontsize=FS, labelpad=1)
    _style(ax)


def plot_nearest(ax, df):
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.4, ls="--")
    ax.scatter(df.r_rastermap_neighbour, df.r_nearest, s=5, c=df.color, lw=0.2, edgecolors="k")
    lo = min(df.r_rastermap_neighbour.min(), df.r_nearest.min()) - 0.03
    ax.set_xlim(lo, 1.01), ax.set_ylim(lo, 1.01)
    ax.set_xlabel("r with best rastermap neighbour", fontsize=FS, labelpad=1)
    ax.set_ylabel("r with most similar cluster", fontsize=FS, labelpad=1)
    ax.text(0.03, 0.97, f"most similar = neighbour:\n{(df.distance == 1).mean():.0%} of clusters\n(random order "
            f"{2 / (len(df) - 1):.0%})", transform=ax.transAxes, fontsize=FS - 0.5, va="top")
    _style(ax)
    inset = ax.inset_axes([1.08, 0.08, 0.22, 0.84])
    v = np.minimum(df.distance, 11)
    cnt = np.bincount(v, minlength=12)[1:]
    inset.barh(np.arange(1, 12), cnt, color="0.4", height=0.8, lw=0)
    inset.set_yticks([1, 5, 10, 11], ["1", "5", "10", ">10"])
    inset.invert_yaxis()
    inset.set_xlabel("Clusters", fontsize=FS - 0.5, labelpad=1)
    inset.set_ylabel("Distance to most similar cluster", fontsize=FS - 0.5, labelpad=2)
    inset.yaxis.set_label_position("right")
    _style(inset)
    inset.tick_params(labelsize=FS - 0.5)


def plot_assignment(ax, df):
    d = df[df.kind == "dist"]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.frac_obs, width=0.4, color="k", lw=0, label="neurons (held-out trials)")
    ax.bar(x + 0.2, d.frac_null, width=0.4, color="0.7", lw=0, label="random other cluster")
    ax.set_xticks(x, d.d.astype(str))
    ax.set_xlabel("Distance along the order to the best cluster other than its own", fontsize=FS, labelpad=1)
    ax.set_ylabel("Fraction of neurons", fontsize=FS, labelpad=1)
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="upper center", handlelength=1.0)
    _style(ax)


def plot_margin(ax, df):
    m = df[df.kind == "margin"]
    ax.bar((m.bin_lo + m.bin_hi) / 2, m["count"], width=(m.bin_hi - m.bin_lo), color="0.5", lw=0)
    ax.axvline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("r(own cluster) − r(best other)", fontsize=FS, labelpad=1)
    ax.set_ylabel("Neurons", fontsize=FS, labelpad=1)
    own = m[m.bin_lo >= -1e-9]["count"].sum() / m["count"].sum()
    ax.text(0.97, 0.97, f"own cluster best:\n{own:.0%} of neurons", transform=ax.transAxes, fontsize=FS - 0.5,
            va="top", ha="right")
    _style(ax)
