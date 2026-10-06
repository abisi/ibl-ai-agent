"""002 -- Summaries and figures of 001 (whisker vs auditory stimulus decoding, all sessions pooled).

Per iteration i: corrected balanced accuracy d_i(t) = real_i(t) - null_i(t) (null = mean of the trial-shuffle decodes of the
same draw). Per area x N x bin: mean, s.d., 2.5 / 5 / 97.5 percentiles of d across iterations.
Above chance in a bin: 5th percentile of d > 0 (one-sided bootstrap over iterations, as ssl-pseudopopulation-area-decoding).
First significant bin (onset), searched after stimulus onset (bin end > 0), the onset bin itself above chance:
  wide (50-ms bins, 5-ms steps): >= 4 of the 5 bins starting at it above chance (25 ms; the previous rule);
  zoom (20-ms bins, 2-ms steps): >= 80 % of the bins in the 25 ms starting at it (>= 11 of 13 bins; the same rule at the
  finer step). Also stored: >= 20 of the 25 bins starting at it (50 ms) -- onset_zoom_20of25bins.
Early-window accuracy: mean d over the zoom bins ending in 5..50 ms after stimulus onset (per iteration -> mean, 95 % range).
Figures (OUT/figures/):
  arrival_curves_<level>     per area: wide | zoom, one curve per N (mean +- s.d. over iterations), onset ticks
  window_vs_N                5-50 ms corrected accuracy vs number of neurons, one panel per level, one line per area
  onset_vs_window            onset (ms) vs 5-50 ms accuracy (one point per area x N), per level and resolution
Tables: summary_bins.parquet, onsets.csv, window_accuracy.csv
"""
import importlib
import os
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = importlib.import_module("_style")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
AR = importlib.import_module("_areas")
OUT = AR.OUT
FIG = OUT / "figures"
COARSE, FINE, LEVELS = AR.COARSE, AR.FINE, AR.LEVELS
S.AREA_C.update(AR.colors())
LEVEL_NAME = {"area_group": "Area groups", "area_acronym_custom": "Areas"}
WIDE_T = np.round(np.arange(-0.2, 0.6 + 1e-9, 0.005), 6)
ZOOM_T = np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 6)
WIN = (0.005, 0.050)
N_GRID = [20, 50, 100, 200, 300, 500]


def load_raw():
    rows = []
    for f in sorted((OUT / "raw").glob("*.parquet")):
        if f.name.endswith(".tmp.parquet"):
            continue
        d = pd.read_parquet(f)
        d = d[d.skipped_reason.isna()]
        if len(d):
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def d_matrix(g):
    return np.stack(g.curve.map(np.asarray)) - np.stack(g.null_mean.map(np.asarray))     # iterations x bins


def onset(t, sig, span_bins, need):
    for i in np.where((t > 0) & sig)[0]:
        if i + span_bins > len(t):
            return np.nan
        if sig[i:i + span_bins].sum() >= need:
            return t[i]
    return np.nan


def summarise(D):
    bins, ons, win = [], [], []
    nw = len(WIDE_T)
    for (level, area, N), g in D.groupby(["level", "area", "N"]):
        d = d_matrix(g)
        for res, sl, t in (("wide", slice(0, nw), WIDE_T), ("zoom", slice(nw, None), ZOOM_T)):
            x = d[:, sl]
            p5 = np.percentile(x, 5, axis=0)
            sig = p5 > 0
            bins.append(pd.DataFrame(dict(level=level, area=area, N=N, resolution=res, t=t, mean=x.mean(0), sd=x.std(0),
                                          lo=np.percentile(x, 2.5, axis=0), hi=np.percentile(x, 97.5, axis=0), p5=p5,
                                          sig=sig, n_iter=len(x))))
            if res == "wide":
                ons.append(dict(level=level, area=area, N=N, resolution="wide", onset_ms=1000 * onset(t, sig, 5, 4)))
            else:
                ons.append(dict(level=level, area=area, N=N, resolution="zoom", onset_ms=1000 * onset(t, sig, 13, 11),
                                onset_zoom_20of25bins_ms=1000 * onset(t, sig, 25, 20)))
        zw = d[:, nw:][:, (ZOOM_T >= WIN[0] - 1e-9) & (ZOOM_T <= WIN[1] + 1e-9)].mean(1)
        win.append(dict(level=level, area=area, N=N, window_ms="5-50", mean=zw.mean(), sd=zw.std(), lo=np.percentile(zw, 2.5),
                        hi=np.percentile(zw, 97.5), p_le0=(1 + np.sum(zw <= 0)) / (1 + len(zw)), n_iter=len(zw),
                        n_eligible_sessions=int(g.n_eligible_sessions.iloc[0]), n_eligible_mice=int(g.n_eligible_mice.iloc[0])))
    return pd.concat(bins, ignore_index=True), pd.DataFrame(ons), pd.DataFrame(win)


def ncolors(Ns):
    import matplotlib.cm as cm
    Ns = sorted(Ns)
    return {N: cm.viridis_r(0.12 + 0.8 * i / max(1, len(Ns) - 1)) for i, N in enumerate(Ns)}


def fig_curves(plt, B, O, level):
    areas = [a for a in LEVELS[level] if a in set(B.area)]
    Ns = sorted(n for n in set(B.N) if n in N_GRID)
    nc = ncolors(Ns)
    ncols = 3
    nrows = int(np.ceil(len(areas) / ncols))
    H = 1.25 * nrows + 0.8
    fig = plt.figure(figsize=(S.W_IN, H))
    gs = fig.add_gridspec(nrows, ncols * 2, width_ratios=[1.35, 1] * ncols, wspace=0.45, hspace=0.75,
                          left=0.06, right=0.99, top=1 - 0.6 / H, bottom=0.35 / H)
    ymax = B[(B.level == level) & B.N.isin(Ns)]["mean"].max() + B[(B.level == level) & B.N.isin(Ns)]["sd"].max()
    for k, a in enumerate(areas):
        r, c = divmod(k, ncols)
        for j, res in enumerate(["wide", "zoom"]):
            ax = fig.add_subplot(gs[r, 2 * c + j])
            for N in Ns:
                q = B[(B.level == level) & (B.area == a) & (B.N == N) & (B.resolution == res)]
                if not len(q):
                    continue
                tt = 1000 * q.t.to_numpy()
                ax.fill_between(tt, q["mean"] - q.sd, q["mean"] + q.sd, color=nc[N], alpha=0.18, lw=0, edgecolor="none")
                ax.plot(tt, q["mean"], color=nc[N], lw=0.7, label=f"{N}")
                o = O[(O.level == level) & (O.area == a) & (O.N == N) & (O.resolution == res)].onset_ms
                if len(o) and np.isfinite(o.iloc[0]):
                    ax.plot([o.iloc[0]], [ymax * 1.02], marker="v", ms=2.6, color=nc[N], mew=0, clip_on=False)
            ax.axhline(0, color="0.5", lw=0.4)
            ax.axvline(0, color="0.3", lw=0.4, ls="--")
            ax.set_ylim(-0.05, ymax * 1.06)
            if res == "wide":
                ax.axvspan(-20, 100, color="0.85", alpha=0.5, lw=0, edgecolor="none", zorder=0)
                ax.set_xlim(-200, 600)
                ax.set_title(S.short(a), loc="left", fontsize=6.5, weight="bold", color=S.AREA_C.get(a, "k"))
                ax.set_ylabel("Corrected balanced\naccuracy" if c == 0 else "")
            else:
                ax.set_xlim(-20, 100)
                ax.set_title("zoom (20-ms bins)", fontsize=5, loc="left", color="0.35")
                ax.set_yticklabels([])
            if r == nrows - 1 or k + ncols >= len(areas):
                ax.set_xlabel("Time from stimulus (ms)", fontsize=5.5)
            if k == 0 and res == "wide":                 # one legend for the figure, in the header (not over the data)
                h, lab = ax.get_legend_handles_labels()
                fig.legend(h, lab, title="Neurons in the pseudo-population", frameon=False, fontsize=4.8, title_fontsize=4.8,
                           ncol=len(h), handlelength=1.2, columnspacing=0.9, loc="upper right", bbox_to_anchor=(0.99, 1 - 0.03 / H))
    fig.suptitle(f"Whisker vs auditory decoding, every N ({EPOCH_WORD}, all sessions pooled) -- {LEVEL_NAME[level].lower()}",
                 x=0.06, y=1 - 0.05 / H, va="top", ha="left", fontsize=7, weight="bold")
    S.save(fig, FIG, f"arrival_curves_{level}")
    plt.close(fig)


EPOCH_WORD = "task (active) trials" if AR.EPOCH == "active" else "passive trials"
N_BEST = 8


def best_areas(W, level, k=N_BEST):
    """the k areas with the most eligible sessions (at N = 200)"""
    q = W[(W.level == level) & (W.N == 200)].set_index("area").n_eligible_sessions
    return list(q.sort_values(ascending=False).index[:k])


def fig_n(plt, O, W):
    """early accuracy and onset vs number of neurons, every area of each level (one row per level)"""
    from matplotlib.ticker import NullFormatter, NullLocator
    fig = plt.figure(figsize=(S.W_IN, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.62], wspace=0.38, hspace=0.42, left=0.08, right=0.99, top=0.92, bottom=0.08)
    axes = []
    for r, level in enumerate(LEVELS):
        axa, axb = fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])
        axes.append((axa, axb))
        for a in LEVELS[level]:
            q = W[(W.level == level) & (W.area == a) & W.N.isin(N_GRID)].sort_values("N")
            if not len(q):
                continue
            axa.errorbar(q.N, q["mean"], [q["mean"] - q.lo, q.hi - q["mean"]], color=S.AREA_C[a], marker="o", ms=2.0,
                         lw=0.7, elinewidth=0.4, capsize=0, label=S.short(a))
            o = O[(O.level == level) & (O.area == a) & (O.resolution == "zoom") & O.N.isin(N_GRID)].sort_values("N")
            axb.plot(o.N, o.onset_ms, color=S.AREA_C[a], marker="o", ms=2.0, lw=0.7)
        for ax in (axa, axb):
            ax.set_xscale("log"); ax.set_xticks(N_GRID, [str(n) for n in N_GRID])
            ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("Neurons in the pseudo-population")
        axa.axhline(0, color="0.5", lw=0.4)
        axa.set_ylabel("Corrected balanced accuracy, 5-50 ms")
        axb.set_ylabel("Onset (ms; 20-ms bins, 2-ms steps)")
        axa.set_title(f"{LEVEL_NAME[level]}: early accuracy (95 % range over iterations)", loc="left", fontsize=5.6)
        axb.set_title(f"{LEVEL_NAME[level]}: onset (missing: no onset)", loc="left", fontsize=5.6)
        h, lab = axa.get_legend_handles_labels()
        lax = fig.add_subplot(gs[r, 2]); lax.set_axis_off()
        lax.legend(h, lab, frameon=False, fontsize=4.2, handlelength=1.0, loc="upper left", borderaxespad=0,
                   ncol=1 if len(h) <= 20 else 2, columnspacing=0.6, labelspacing=0.25)
    S.letter_row(fig, axes[0], "ab"); S.letter_row(fig, axes[1], "cd")
    fig.suptitle(f"Whisker vs auditory decoding: effect of the number of neurons ({EPOCH_WORD})", x=0.02, y=0.99, ha="left",
                 va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, "accuracy_onset_vs_N")
    plt.close(fig)


def fig_onset_corr(plt, O, W):
    """onset vs early accuracy, two ways per level (user 2026-10-06): (left) every area at N = 200; (right) the 8
    best-sampled areas at every N, dot size = N, lines join an area's N; decreasing-exponential fit + 95 % bootstrap band
    (OLS kept in the stats for comparison)"""
    M = O[O.resolution == "zoom"].merge(W[["level", "area", "N", "mean"]], on=["level", "area", "N"])
    M = M[M.N.isin(N_GRID)]
    fig = plt.figure(figsize=(S.W_IN, 6.0))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.42, left=0.08, right=0.98, top=0.92, bottom=0.08)
    size = lambda n: 4 + 30 * np.log(n / 20 + 1) / np.log(26)
    stats, axes = [], []
    for r, level in enumerate(LEVELS):
        ax1, ax2 = fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])
        axes.append((ax1, ax2))
        q = M[(M.level == level) & (M.N == 200)]
        st = S.exp_panel(ax1, q["mean"].to_numpy(), q.onset_ms.to_numpy(), colors=q.area.map(S.AREA_C).to_numpy(), sizes=np.full(len(q), 14))
        S.label_points(ax1, q["mean"].to_numpy(), q.onset_ms.to_numpy(), [S.short(a) for a in q.area],
                       [S.AREA_C[a] for a in q.area], fontsize=3.8 if level != "area_group" else 4.3)
        stats.append(dict(level=level, resolution="zoom", set="N200_all_areas", **st))
        best = best_areas(W, level)
        qb = M[(M.level == level) & M.area.isin(best)].sort_values(["area", "N"])
        for a in best:
            g = qb[qb.area == a]
            ax2.plot(g["mean"], g.onset_ms, color=S.AREA_C[a], lw=0.6, alpha=0.7, zorder=2)
        st2 = S.exp_panel(ax2, qb["mean"].to_numpy(), qb.onset_ms.to_numpy(), colors=qb.area.map(S.AREA_C).to_numpy(),
                          sizes=size(qb.N.to_numpy()))
        stats.append(dict(level=level, resolution="zoom", set="best8_all_N", **st2))
        qa = M[M.level == level]
        stats.append(dict(level=level, resolution="zoom", set="all_areas_all_N", **S.exp_panel(
            fig.add_axes([0, 0, 0.01, 0.01], visible=False), qa["mean"].to_numpy(), qa.onset_ms.to_numpy(), n_boot=0, scatter=False)))
        for ax, t, stt in ((ax1, f"{LEVEL_NAME[level]}, N = 200, every area", st),
                           (ax2, f"{LEVEL_NAME[level]}, {N_BEST} best-sampled, every N (dot size: N)", st2)):
            ax.set_title(t, loc="left", fontsize=5.6)
            ax.text(0.98, 0.97, f"ρ = {stt['rho']:.2f}, {S.fmt_p(stt['p_rho'])}, n = {stt['n']}\n"
                    f"exponential R² = {stt['r2_exp']:.2f}, linear R² = {stt['r2_ols']:.2f}\n"
                    f"ΔAIC (exp - linear) = {stt['aic_exp'] - stt['aic_ols']:.1f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=4.6)
            ax.set_xlabel("Corrected balanced accuracy, 5-50 ms")
            ax.set_ylabel("Onset (ms)")
        h = [plt.Line2D([], [], marker="o", ls="", color=S.AREA_C[a], ms=3, label=S.short(a)) for a in best]
        h += [plt.Line2D([], [], marker="o", ls="", color="0.5", ms=np.sqrt(size(n)), label=f"N = {n}") for n in (20, 100, 500)]
        ax2.legend(handles=h, loc="lower left", frameon=False, fontsize=4.2, handletextpad=0.3, labelspacing=0.35,
                   borderaxespad=0.2)
    S.letter_row(fig, axes[0], "ab"); S.letter_row(fig, axes[1], "cd")
    fig.suptitle(f"Whisker vs auditory decoding: onset vs early accuracy ({EPOCH_WORD})", x=0.02, y=0.99, ha="left", va="top",
                 fontsize=7, weight="bold")
    S.save(fig, FIG, "onset_vs_window")
    plt.close(fig)
    pd.DataFrame(stats).to_csv(OUT / "onset_vs_window_stats.csv", index=False)


def main():
    plt = S.setup()
    D = load_raw()
    B, O, W = summarise(D)
    B.to_parquet(OUT / "summary_bins.parquet")
    O.to_csv(OUT / "onsets.csv", index=False)
    W.to_csv(OUT / "window_accuracy.csv", index=False)
    for level in LEVELS:
        if (B.level == level).any():
            fig_curves(plt, B, O, level)
    fig_n(plt, O, W)
    fig_onset_corr(plt, O, W)
    print(O.pivot_table(index=["level", "area"], columns=["resolution", "N"], values="onset_ms").round(0).to_string())
    print(W.pivot_table(index=["level", "area"], columns="N", values="mean").round(3).to_string())
    print("ALL DONE", FIG)


if __name__ == "__main__":
    main()
