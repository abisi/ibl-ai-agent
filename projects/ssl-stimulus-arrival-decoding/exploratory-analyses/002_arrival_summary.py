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
    fig = plt.figure(figsize=(S.W_IN, 1.25 * nrows + 0.5))
    gs = fig.add_gridspec(nrows, ncols * 2, width_ratios=[1.35, 1] * ncols, wspace=0.45, hspace=0.75,
                          left=0.06, right=0.99, top=1 - 0.35 / (1.25 * nrows + 0.5), bottom=0.35 / (1.25 * nrows + 0.5))
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
                ax.set_title("zoom: 20-ms bins, 2-ms steps", fontsize=5, loc="left", color="0.35")
                ax.set_yticklabels([])
            if r == nrows - 1 or k + ncols >= len(areas):
                ax.set_xlabel("Time from stimulus (ms)", fontsize=5.5)
            if k == 0 and res == "wide":
                ax.legend(title="Neurons", frameon=False, fontsize=4.6, title_fontsize=4.8, ncol=2, handlelength=1,
                          columnspacing=0.6, loc="upper right")
    fig.suptitle(f"Whisker vs auditory stimulus decoding, all sessions pooled -- {LEVEL_NAME[level].lower()}",
                 x=0.06, ha="left", fontsize=7, weight="bold")
    S.save(fig, FIG, f"arrival_curves_{level}")
    plt.close(fig)


def fig_window(plt, W):
    fig, axs = plt.subplots(1, 2, figsize=(S.W_IN * 0.75, 2.1), gridspec_kw=dict(wspace=0.35))
    for ax, level in zip(axs, LEVELS):
        for a in LEVELS[level]:
            q = W[(W.level == level) & (W.area == a) & W.N.isin(N_GRID)].sort_values("N")
            if not len(q):
                continue
            ax.errorbar(q.N, q["mean"], [q["mean"] - q.lo, q.hi - q["mean"]], color=S.AREA_C[a], marker="o", ms=2.2,
                        lw=0.8, elinewidth=0.5, capsize=0, label=S.short(a))
        ax.set_xscale("log")
        ax.set_xticks(N_GRID, [str(n) for n in N_GRID])
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_xlabel("Neurons in the pseudo-population")
        ax.set_ylabel("Corrected balanced accuracy,\n5-50 ms after stimulus")
        ax.set_title(LEVEL_NAME[level], loc="left")
        ax.legend(frameon=False, fontsize=4.8, handlelength=1.2, loc="upper left")
    S.letter_row(fig, axs, "ab")
    S.save(fig, FIG, "window_vs_N")
    plt.close(fig)


def fig_onset_corr(plt, O, W):
    M = O.merge(W[["level", "area", "N", "mean"]], on=["level", "area", "N"])
    M = M[M.N.isin(N_GRID)]
    fig, axs = plt.subplots(2, 2, figsize=(S.W_IN * 0.7, 3.9), gridspec_kw=dict(wspace=0.4, hspace=0.6))
    stats = []
    for i, res in enumerate(["zoom", "wide"]):
        for j, level in enumerate(LEVELS):
            ax = axs[i, j]
            q = M[(M.level == level) & (M.resolution == res)]
            st = S.corr_panel(ax, q["mean"].to_numpy(), q.onset_ms.to_numpy(), colors=q.area.map(S.AREA_C).to_numpy(),
                              sizes=(4 + 14 * np.log(q.N / 20 + 1) / np.log(26)).to_numpy())
            stats.append(dict(level=level, resolution=res, **st))
            ax.set_title(f"{LEVEL_NAME[level]}, {'20-ms bins' if res == 'zoom' else '50-ms bins'}", loc="left")
            ax.text(0.98, 0.97, f"r = {st['r']:.2f}, {S.fmt_p(st['p'])}\nρ = {st['rho']:.2f}, {S.fmt_p(st['p_rho'])}\nn = {st['n']} area × N",
                    transform=ax.transAxes, ha="right", va="top", fontsize=4.8)
            ax.set_xlabel("Corrected balanced accuracy, 5-50 ms")
            ax.set_ylabel("First significant bin (ms)")
    h = [plt.Line2D([], [], marker="o", ls="", color=S.AREA_C[a], ms=3, label=S.short(a)) for lv in LEVELS for a in LEVELS[lv]]
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=4.8, bbox_to_anchor=(0.5, -0.06))
    S.letter_row(fig, axs[0], "ab")
    S.letter_row(fig, axs[1], "cd")
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
    fig_window(plt, W)
    fig_onset_corr(plt, O, W)
    print(O.pivot_table(index=["level", "area"], columns=["resolution", "N"], values="onset_ms").round(0).to_string())
    print(W.pivot_table(index=["level", "area"], columns="N", values="mean").round(3).to_string())
    print("ALL DONE", FIG)


if __name__ == "__main__":
    main()
