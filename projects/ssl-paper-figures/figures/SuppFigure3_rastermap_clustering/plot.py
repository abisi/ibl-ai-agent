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


def _pstr(p):
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _stat_text(ax, lines, loc=(0.97, 0.97), ha="right"):
    for k, (txt, c) in enumerate(lines):
        ax.text(loc[0], loc[1] - k * 0.075, txt, transform=ax.transAxes, ha=ha, va="top", fontsize=4.5, color=c)


def plot_bgmm_contingency(ax, df):
    M = df.pivot(index="cluster", columns="component_rank", values="frac_of_cluster").fillna(0)
    ax.imshow(M.to_numpy(), cmap="Greys", vmin=0, vmax=0.6, aspect="auto", interpolation="nearest")
    fam = df.drop_duplicates("cluster").sort_values("cluster").metacluster.to_numpy()
    strip = ax.inset_axes([-0.06, 0, 0.035, 1])
    strip.imshow((fam % 2)[:, None], cmap="Greys", vmin=-1, vmax=3, aspect="auto", interpolation="nearest")
    strip.axis("off")
    ax.set_yticks([])
    ax.set_xticks([0, M.shape[1] - 1], ["1", str(M.shape[1])])
    ax.set_ylabel("Rastermap cluster", labelpad=8)
    ax.set_xlabel("GMM component (matched order)")
    cax = ax.inset_axes([0.72, 1.03, 0.28, 0.035])
    import matplotlib as mpl
    cb = ax.figure.colorbar(mpl.cm.ScalarMappable(cmap="Greys", norm=mpl.colors.Normalize(0, 0.6)), cax=cax,
                            ticks=[0, 0.6], orientation="horizontal")
    cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_title("Fraction of cluster", fontsize=5, pad=5)
    cb.outline.set_linewidth(0.3)


def plot_bgmm_agreement(ax, df):
    metrics = [("matched", "Matched", "k"), ("AMI", "AMI", "0.4"), ("ARI", "ARI", "0.7")]
    w = 0.26
    for i, lv in enumerate(["fine clusters", "metaclusters"]):
        for k, (m, lab, c) in enumerate(metrics):
            r = df[(df.level == lv) & (df.metric == m)].iloc[0]
            x = i + (k - 1) * w
            ax.bar(x, r.value, w * 0.9, color=c, lw=0, label=lab if i == 0 else None)
            ax.plot([x - w * 0.45, x + w * 0.45], [r.null_p95] * 2, color="#e8222e", lw=0.7)
    K = int(df[df.level == "metaclusters"].K.iloc[0])
    ax.set_xticks([0, 1], ["Fine\n(K = 100)", f"Meta\n(K = {K})"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Agreement with rastermap")
    ax.legend(loc="upper left", fontsize=4.5, handlelength=0.8, borderaxespad=0.1)


def plot_bgmm_matched(ax, df):
    rng = np.random.default_rng(1)
    for i, lv in enumerate(["fine clusters", "metaclusters"]):
        d = df[df.level == lv]
        real, null = d[d.kind == "real"].r, d[d.kind == "perm"].r
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax.fill_between([i - 0.35, i + 0.35], lo, hi, color="0.88", lw=0)
        ax.plot([i - 0.35, i + 0.35], [np.median(null)] * 2, color="0.55", lw=0.6)
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(real)), real, s=1.6, c="k", lw=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(real)] * 2, color="#e8222e", lw=0.8)
    K = int(df[(df.level == "metaclusters") & (df.kind == "real")].shape[0])
    ax.set_xticks([0, 1], ["Fine\n(K = 100)", f"Meta\n(K = {K})"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.2, 1.02)
    ax.axhline(0, color="0.6", lw=0.3)
    ax.set_ylabel("r, matched mean PSTHs")


def plot_latency(ax, df):
    lines = []
    for at, c in (("whisker_active", W), ("auditory_active", A)):
        d = df[df.analysis_type == at].sort_values("cluster")
        ax.scatter(d.cluster, d.latency_ms, s=2.5, color=c, lw=0, label=at.split("_")[0].capitalize())
        ax.plot(d.cluster, d.fit, color=c, lw=0.8)
        lines.append((f"ρ = {d.rho.iloc[0]:.2f}, {_pstr(d.p_perm.iloc[0])}", c))
    ax.set_xlabel("Rastermap cluster")
    ax.set_ylabel("Onset latency (ms)")
    ax.set_xlim(-2, 101)
    ax.legend(loc="upper right", fontsize=4.5, handlelength=0.6, markerscale=1.5, borderaxespad=0.1)
    _stat_text(ax, lines, loc=(0.97, 0.80))


def plot_hierarchy(ax, df):
    lines = []
    for g, c in (("whisker", W), ("auditory", A)):
        d = df[df.dominant == g].sort_values("cluster")
        ax.errorbar(d.cluster, d.hierarchy, yerr=d["sem"], fmt="o", ms=1.4, color=c, lw=0.4, elinewidth=0.3,
                    label=f"{g.capitalize()}-dom.")
        ax.plot(d.cluster, d.fit, color=c, lw=0.8)
        lines.append((f"ρ = {d.rho.iloc[0]:.2f}, {_pstr(d.p_perm.iloc[0])}", c))
    ax.set_xlabel("Rastermap cluster")
    ax.set_ylabel("Hierarchy score")
    ax.set_xlim(-2, 101)
    ax.legend(loc="upper right", fontsize=4.5, handlelength=0.6, borderaxespad=0.1)
    _stat_text(ax, lines, loc=(0.2, 0.15), ha="left")
