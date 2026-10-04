"""004 -- Main and summary figures, one set per level (area groups / fine areas, never mixed): where and when can
stimulus modality (whisker vs auditory) be decoded? Epoch from _areas (ARRIVAL_EPOCH=active|passive).

Colours: area groups from allen_utils.get_custom_area_groups_colors() (Axel Bisi's palette); fine areas in shades of their
parent group (_areas.colors). Rows of the heatmaps are sorted by onset (fast at the top).
Main figure per level (N = 200 neurons):
  a -- heatmap of the corrected balanced accuracy over time (50-ms bins, 5-ms steps), every area; bins not above chance
       shown faded; tick: onset;
  b -- the same for the first 50 ms (20-ms bins, 2-ms steps);
  c, d -- time courses (wide / first 50 ms) of the 8 best-sampled areas, mean +- s.d. over iterations, one bar per area
       marking the bins above chance, black tick: onset;
  e -- onset ranking (all areas), error bars: 95 % range over 1000 resamples of the iterations.
Summary figure per level: a method schematic, b first-50-ms heatmap, c onset ranking, d early accuracy vs number of
neurons, e onset vs early accuracy, f matched early accuracy (area groups) or best-sampled curves (areas),
g control: raw balanced accuracy and trial-shuffle null, h time course of the best-sampled areas.
Above chance: 5th percentile of the corrected accuracy over iterations > 0; onset: first post-stimulus bin above chance
with >= 80 % of the bins in the next 25 ms above chance.
Output: <OUT>/figures/arrival_main_N200_<level>.*, arrival_summary_<level>.*, arrival_figures_caption.md,
onset_bootstrap_N200.csv
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = importlib.import_module("_style")
AR = importlib.import_module("_areas")
m2 = importlib.import_module("002_arrival_summary")
OUT, FIG = m2.OUT, m2.FIG
N_MAIN, N_BOOT, N_CURVES = 200, 1000, 8
LEVEL_NAME = {"area_group": "area groups", "area_acronym_custom": "areas"}
SHORT = {"Somatosensory-whisker": "SS-whisker", "Somatosensory-orofacial": "SS-orofacial", "Somatosensory-body": "SS-body",
         "Auditory areas": "Auditory", "Motor areas": "Motor", "Frontal areas": "Frontal", "Retrosplenial areas": "Retrosplenial",
         "Posterior parietal areas": "Post. parietal", "Lateral septal complex": "Lat. septum", "Visual areas": "Visual",
         "Insular areas": "Insular", "Olfactory areas": "Olfactory", "Amygdala and hypothalamus": "Amygdala + hypoth."}
EPOCH_WORD = "task (active) trials" if AR.EPOCH == "active" else "passive trials"


def label(a):
    return SHORT.get(a, a)


def onset_boot(D, rng):
    rows = []
    nw = len(m2.WIDE_T)
    for (level, area), g in D[D.N == N_MAIN].groupby(["level", "area"]):
        d = m2.d_matrix(g)[:, nw:]
        k = len(d)
        obs = 1000 * m2.onset(m2.ZOOM_T, np.percentile(d, 5, axis=0) > 0, 13, 11)
        boots = np.array([1000 * m2.onset(m2.ZOOM_T, np.percentile(d[rng.integers(0, k, k)], 5, axis=0) > 0, 13, 11)
                          for _ in range(N_BOOT)])
        fin = np.isfinite(boots)
        rows.append(dict(level=level, area=area, N=N_MAIN, onset_ms=obs,
                         lo=np.percentile(boots[fin], 2.5) if fin.any() else np.nan,
                         hi=np.percentile(boots[fin], 97.5) if fin.any() else np.nan, frac_defined=fin.mean(),
                         n_iter=k, n_eligible_sessions=int(g.n_eligible_sessions.iloc[0])))
    return pd.DataFrame(rows)


def order(OB, level):
    q = OB[OB.level == level].copy()
    q["key"] = q.onset_ms.fillna(1e9)
    return q.sort_values(["key", "area"]).area.tolist()


def runs(sig):
    out, s = [], None
    for i, v in enumerate(sig):
        if v and s is None:
            s = i
        if (not v or i == len(sig) - 1) and s is not None:
            out.append((s, i if v else i - 1))
            s = None
    return out


def heatmap(ax, B, OB, level, areas, res, xlim, col, cax=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rows, sigs, t = [], [], None
    for a in areas:
        q = B[(B.level == level) & (B.area == a) & (B.N == N_MAIN) & (B.resolution == res)]
        if len(q):
            t = 1000 * q.t.to_numpy()
        rows.append(q["mean"].to_numpy() if len(q) else None)
        sigs.append(q.sig.to_numpy() if len(q) else None)
    n = len(t)
    M = np.array([r if r is not None else np.full(n, np.nan) for r in rows])
    Sg = np.array([s if s is not None else np.zeros(n, bool) for s in sigs])
    keep = (t >= xlim[0]) & (t <= xlim[1])
    M, Sg, t = M[:, keep], Sg[:, keep], t[keep]
    step = t[1] - t[0]
    ext = (t[0] - step / 2, t[-1] + step / 2, len(areas) - 0.5, -0.5)
    im = ax.imshow(np.where(Sg, M, np.nan), aspect="auto", extent=ext, cmap=plt.get_cmap("magma_r"), vmin=0, vmax=0.5,
                   interpolation="nearest")
    ax.imshow(np.where(~Sg & np.isfinite(M), 1.0, np.nan), aspect="auto", extent=ext, cmap=ListedColormap(["#ececec"]),
              interpolation="nearest")
    on = OB[OB.level == level].set_index("area").onset_ms
    for i, a in enumerate(areas):
        o = on.get(a, np.nan)
        if np.isfinite(o) and xlim[0] <= o <= xlim[1]:
            ax.plot([o, o], [i - 0.45, i + 0.45], color="white", lw=1.0)
            ax.plot([o, o], [i - 0.45, i + 0.45], color="k", lw=0.4)
    ax.axvline(0, color="0.2", lw=0.5, ls="--")
    ax.set_yticks(range(len(areas)), [label(a) for a in areas], fontsize=4.3 if len(areas) > 20 else 5)
    for tl, a in zip(ax.get_yticklabels(), areas):
        tl.set_color(col.get(a, "k"))
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.tick_params(axis="y", length=0)
    if cax is not None:
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Corrected balanced accuracy (grey: not above chance)", fontsize=4.6, labelpad=1)
        cb.ax.tick_params(labelsize=4.4, length=1.2, width=0.4); cb.outline.set_linewidth(0.4)
    return im


def curves(ax, B, OB, level, areas, res, xlim, col, ylim=(-0.03, 0.52)):
    on = OB[OB.level == level].set_index("area").onset_ms
    for a in areas:
        q = B[(B.level == level) & (B.area == a) & (B.N == N_MAIN) & (B.resolution == res)]
        t = 1000 * q.t.to_numpy()
        ax.fill_between(t, q["mean"] - q.sd, q["mean"] + q.sd, color=col[a], alpha=0.15, lw=0, edgecolor="none")
        ax.plot(t, q["mean"], color=col[a], lw=0.9)
    ax.axhline(0, color="0.5", lw=0.4); ax.axvline(0, color="0.3", lw=0.4, ls="--"); ax.set_xlim(*xlim)
    y0, dy = ylim[1] + 0.025, 0.034                  # row spacing of the significance bars (labels must not touch)
    for i, a in enumerate(areas):
        q = B[(B.level == level) & (B.area == a) & (B.N == N_MAIN) & (B.resolution == res)]
        t, sig = 1000 * q.t.to_numpy(), q.sig.to_numpy()
        step, yy = t[1] - t[0], y0 + i * dy
        for s_, e_ in runs(sig):
            x0_, x1_ = max(t[s_] - step / 2, xlim[0]), min(t[e_] + step / 2, xlim[1])
            if x1_ > x0_:
                ax.plot([x0_, x1_], [yy, yy], color=col[a], lw=1.6, solid_capstyle="butt", clip_on=False)
        o = on.get(a, np.nan)
        if np.isfinite(o) and xlim[0] <= o <= xlim[1]:
            ax.plot([o], [yy], marker="|", ms=4, mew=0.9, color="k", clip_on=False, zorder=6)
        ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), yy, label(a), color=col[a], fontsize=4.3, va="center", weight="bold")
    ax.set_ylim(ylim[0], y0 + len(areas) * dy)
    ax.spines["left"].set_bounds(*ylim)
    ax.set_yticks([v for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5) if v <= ylim[1]])
    ax.set_xlabel("Time from stimulus onset (ms)")


def ranking(ax, OB, level, areas, col):
    q = OB[OB.level == level].set_index("area").reindex(areas)
    y = np.arange(len(q))
    ax.barh(y, q.onset_ms, color=[col[a] for a in q.index], height=0.7, lw=0)
    ax.errorbar(q.onset_ms, y, xerr=[q.onset_ms - q.lo, q.hi - q.onset_ms], fmt="none", ecolor="0.2", lw=0.5, capsize=0)
    fs = 4.3 if len(areas) > 20 else 5
    ax.set_yticks(y, [label(a) for a in q.index], fontsize=fs)
    for tl, a in zip(ax.get_yticklabels(), q.index):
        tl.set_color(col.get(a, "k"))
    for yi, v, h in zip(y, q.onset_ms, q.hi):
        xt = (np.nanmax([v, h]) if np.isfinite(v) else 0) + 0.5
        ax.text(xt, yi, f"{v:.0f}" if np.isfinite(v) else "n.s.", va="center", fontsize=fs - 0.3, color="0.3")
    ax.set_ylim(len(q) - 0.5, -0.5)
    ax.set_xlim(0, np.nanmax(q.hi) * 1.2)
    ax.set_xlabel("Onset (ms)")
    ax.tick_params(axis="y", length=0)


def best_sampled(D, level, areas):
    n = D[(D.level == level) & (D.N == N_MAIN)].groupby("area").n_eligible_sessions.max()
    return list(n.reindex(areas).sort_values(ascending=False).index[:N_CURVES])


def main_figure(plt, D, B, OB, level, col):
    areas = order(OB, level)
    top = [a for a in areas if a in best_sampled(D, level, areas)]
    W = S.W_IN
    hh = max(2.2, 0.085 * len(areas) + 0.6)
    H = hh + 2.4 + 1.0
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 3, height_ratios=[hh, 2.2], width_ratios=[1.45, 0.8, 0.9], wspace=0.45, hspace=0.32,
                          left=0.12, right=0.98, top=1 - 0.9 / H, bottom=0.35 / H)
    axa, axb, axe = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    pb = axb.get_position()                           # colour bar between panel b and its title, ticks and label on top
    cax = fig.add_axes([pb.x0, pb.y1 + 0.1 / H, pb.width, 0.04 / H])
    heatmap(axa, B, OB, level, areas, "wide", (-50, 600), col, cax=cax)
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    pad = 26                                          # titles of the top row above the colour bar (points)
    axa.set_title("Time course (50-ms bins, 5-ms steps)", loc="left", fontsize=5.6, pad=pad)
    heatmap(axb, B, OB, level, areas, "zoom", (-20, 50), col)
    axb.set_yticklabels([]); axb.set_title("First 50 ms (20-ms bins)", loc="left", fontsize=5.6, pad=pad)
    ranking(axe, OB, level, areas, col)
    axe.set_title("Onset ranking (95 % range)", loc="left", fontsize=5.6, pad=pad)
    gb = gs[1, :].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
    axc, axd = fig.add_subplot(gb[0, 0]), fig.add_subplot(gb[0, 1])
    curves(axc, B, OB, level, top, "wide", (-200, 600), col)
    axc.set_ylabel("Corrected balanced accuracy")
    axc.set_title(f"{N_CURVES} best-sampled {LEVEL_NAME[level]} (bars: above chance)", loc="left", fontsize=5.6)
    curves(axd, B, OB, level, top, "zoom", (-20, 50), col)
    axd.set_title("First 50 ms", loc="left", fontsize=5.6)
    S.letter_row(fig, [axa, axb, axe], "abc", dx_in=0.62, dy_in=0.5)
    S.letter_row(fig, [axc, axd], "de")
    fig.suptitle(f"Stimulus-modality decoding across {LEVEL_NAME[level]} ({EPOCH_WORD}, all sessions pooled, N = {N_MAIN} neurons)",
                 x=0.02, y=1 - 0.04 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, f"arrival_main_N200_{level}")
    plt.close(fig)


def summary_figure(plt, D, B, O, W, OB, level, col):
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import NullFormatter, NullLocator
    areas = order(OB, level)
    top = [a for a in areas if a in best_sampled(D, level, areas)]
    hh = max(2.3, 0.085 * len(areas) + 0.5)
    H = hh + 2 * 2.0 + 1.3
    fig = plt.figure(figsize=(S.W_IN, H))
    gs = fig.add_gridspec(3, 3, height_ratios=[hh, 2.0, 2.0], wspace=0.55, hspace=0.45, left=0.1, right=0.98,
                          top=1 - 0.95 / H, bottom=0.3 / H)
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    boxes = [(0.3, 7.0, "122 sessions\n(2 cohorts,\nall days)"), (0.3, 4.0, "20 sessions\nx N/20 neurons\nof the area"),
             (0.3, 1.0, "pseudo-trials\nwhisker vs\nauditory"), (5.3, 7.0, "L2 logistic\nregression\nper bin, 3-fold CV"),
             (5.3, 4.0, "same draw,\nlabels shuffled\nwithin sessions"), (5.3, 1.0, "corrected\naccuracy =\nreal - shuffled")]
    for x0, y0, t in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), 4.2, 2.3, boxstyle="round,pad=0.1", fc="0.95", ec="0.5", lw=0.5))
        ax.text(x0 + 2.1, y0 + 1.15, t, ha="center", va="center", fontsize=4.4, linespacing=1.15)
    for (x0, y0), (x1, y1) in [((2.4, 6.9), (2.4, 6.4)), ((2.4, 3.9), (2.4, 3.4)), ((4.6, 2.15), (5.2, 8.15)),
                               ((7.4, 6.9), (7.4, 6.4)), ((7.4, 3.9), (7.4, 3.4))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax.set_title(f"Pseudo-population decoding\n({EPOCH_WORD})", loc="left", fontsize=5.8)
    axb = fig.add_subplot(gs[0, 1])
    pos = axb.get_position()
    cax = fig.add_axes([pos.x0, pos.y1 + 0.2 / H, pos.width, 0.045 / H])
    heatmap(axb, B, OB, level, areas, "zoom", (-20, 50), col, cax=cax)
    cax.xaxis.set_label_position("top")
    axc = fig.add_subplot(gs[0, 2]); ranking(axc, OB, level, areas, col)
    axc.set_yticklabels([]); axc.set_title("Onset (N = 200)", loc="left", fontsize=5.6)
    axd = fig.add_subplot(gs[1, 0])
    for a in areas:
        q = W[(W.level == level) & (W.area == a) & W.N.isin(m2.N_GRID)].sort_values("N")
        if len(q):
            hi = a in top
            axd.plot(q.N, q["mean"], color=col[a] if hi else "0.82", lw=0.9 if hi else 0.4, marker="o" if hi else None, ms=1.6,
                     zorder=3 if hi else 1)
    axd.set_xscale("log"); axd.set_xticks(m2.N_GRID, [str(n) for n in m2.N_GRID])
    axd.xaxis.set_minor_locator(NullLocator()); axd.xaxis.set_minor_formatter(NullFormatter())
    axd.set_xlabel("Neurons in the pseudo-population"); axd.set_ylabel("Corrected accuracy, 5-50 ms")
    axd.set_title(f"Control: number of neurons\n(colour: {N_CURVES} best-sampled)", loc="left", fontsize=5.6)
    axe = fig.add_subplot(gs[1, 1])
    M = O[(O.level == level) & (O.resolution == "zoom")].merge(W[["level", "area", "N", "mean"]], on=["level", "area", "N"])
    M = M[M.N.isin(m2.N_GRID)]
    st = S.corr_panel(axe, M["mean"].to_numpy(), M.onset_ms.to_numpy(), colors=M.area.map(col).to_numpy(),
                      sizes=(3 + 8 * np.log(M.N / 20 + 1) / np.log(26)).to_numpy())
    axe.text(0.98, 0.97, f"rho = {st['rho']:.2f}, {S.fmt_p(st['p_rho'])}\nn = {st['n']} area x N", transform=axe.transAxes,
             ha="right", va="top", fontsize=4.6)
    axe.set_xlabel("Corrected accuracy, 5-50 ms"); axe.set_ylabel("Onset (ms)")
    axe.set_title("Onset vs early accuracy", loc="left", fontsize=5.6)
    axf = fig.add_subplot(gs[1, 2])
    P = OUT / "matched_n.csv"
    if level == "area_group" and P.exists():
        Pm = pd.read_csv(P)
        Pm = Pm[(Pm.reference == "Somatosensory-whisker") & (Pm.level == level)]
        for r in Pm.itertuples():
            if not np.isfinite(r.matched_N) or r.area not in top:
                continue
            q = B[(B.level == level) & (B.area == r.area) & (B.N == int(r.matched_N)) & (B.resolution == "zoom")]
            if len(q):
                axf.plot(1000 * q.t, q["mean"], color=col[r.area], lw=1.1 if r.area == "Somatosensory-whisker" else 0.7,
                         label=f"{label(r.area)} ({int(r.matched_N)})")
        axf.legend(frameon=False, fontsize=4.0, loc="upper right", title="area (neurons)", title_fontsize=4.2)
        axf.set_title("Control: matched early accuracy\n(SS-whisker at 100 neurons)", loc="left", fontsize=5.6)
    else:
        for a in top:
            q = B[(B.level == level) & (B.area == a) & (B.N == N_MAIN) & (B.resolution == "zoom")]
            axf.plot(1000 * q.t, q["mean"], color=col[a], lw=0.8, label=label(a))
        axf.legend(frameon=False, fontsize=4.0, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)
        axf.set_title(f"{N_CURVES} best-sampled, first 100 ms", loc="left", fontsize=5.6)
    axf.set_ylim(-0.05, 0.8); axf.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])      # headroom above the curves for the legend
    axf.spines["left"].set_bounds(-0.05, 0.55)
    axf.axhline(0, color="0.5", lw=0.4); axf.axvline(0, color="0.3", lw=0.4, ls="--"); axf.set_xlim(-20, 100)
    axf.set_xlabel("Time from stimulus onset (ms)"); axf.set_ylabel("Corrected balanced accuracy")
    axg = fig.add_subplot(gs[2, 0])
    nw = len(m2.WIDE_T)
    for a in top[:3]:
        g = D[(D.area == a) & (D.N == N_MAIN) & (D.level == level)]
        real = np.stack(g.curve.map(np.asarray))[:, :nw]
        null = np.stack(g.null_mean.map(np.asarray))[:, :nw]
        t = 1000 * m2.WIDE_T
        axg.plot(t, real.mean(0), color=col[a], lw=0.9, label=label(a))
        axg.plot(t, null.mean(0), color=col[a], lw=0.8, ls=":")
    axg.axhline(0.5, color="0.5", lw=0.4); axg.axvline(0, color="0.3", lw=0.4, ls="--"); axg.set_xlim(-200, 600)
    axg.set_xlabel("Time from stimulus onset (ms)"); axg.set_ylabel("Balanced accuracy")
    axg.set_ylim(0.45, 1.3); axg.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); axg.spines["left"].set_bounds(0.45, 1.02)
    axg.legend(frameon=False, fontsize=4.3, loc="upper right", ncol=1, handlelength=1.2,
               title="solid: real, dotted: shuffled", title_fontsize=4.3)
    axg.set_title("Control: trial-shuffle null", loc="left", fontsize=5.6)
    axh = fig.add_subplot(gs[2, 1:])
    curves(axh, B, OB, level, top, "wide", (-200, 600), col)
    axh.set_ylabel("Corrected balanced accuracy")
    axh.set_title(f"{N_CURVES} best-sampled {LEVEL_NAME[level]}: time course (bars: above chance)", loc="left", fontsize=5.6)
    S.letter_row(fig, [ax, axb, axc], "abc", dy_in=0.45); S.letter_row(fig, [axd, axe, axf], "def"); S.letter_row(fig, [axg, axh], "gh")
    fig.suptitle(f"Where and when can stimulus modality be decoded? {LEVEL_NAME[level].capitalize()}, {EPOCH_WORD}, all sessions pooled",
                 x=0.02, y=1 - 0.04 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, f"arrival_summary_{level}")
    plt.close(fig)


def captions(OB):
    lines = [f"# Stimulus-modality decoding ({EPOCH_WORD})", "",
             f"**Where and when can stimulus modality be decoded?** Pseudo-population decoding of whisker vs auditory "
             f"{EPOCH_WORD}, all sessions pooled (both cohorts, learning day and expert days; good + mua neurons; whisker-"
             "artefact-corrected spikes). One iteration: 20 sessions drawn with replacement, N/20 neurons of the area drawn "
             "within each, pseudo-trials built by balanced reuse of each session's trials; L2 logistic regression per time bin "
             "(3-fold cross-validation on real trials, inner 2-fold for the regularisation); the same draw re-decoded 10 times "
             "with trial labels shuffled within sessions; corrected balanced accuracy = real - mean shuffled (0 = chance, "
             "0.5 = perfect). 100 iterations (pilot value). Above chance in a bin: 5th percentile over iterations > 0. Onset: "
             "first post-stimulus bin above chance with >= 80 % of the bins in the next 25 ms above chance (20-ms bins, 2-ms "
             "steps, labelled at their end); onset ranges: 95 % range over 1000 resamples of the iterations. Area groups and "
             "areas (40 best-sampled) are shown in separate figures; colours: allen_utils area-group palette, areas in shades "
             "of their group.", ""]
    for level in ("area_group", "area_acronym_custom"):
        q = OB[OB.level == level].sort_values("onset_ms")
        lines.append(f"Onsets at N = {N_MAIN}, {LEVEL_NAME[level]}: " + "; ".join(
            f"{label(r.area)} {r.onset_ms:.0f} ms" if np.isfinite(r.onset_ms) else f"{label(r.area)} n.s." for r in q.itertuples()) + ".")
    (OUT / "arrival_figures_caption.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    plt = S.setup()
    col = AR.colors()
    D = m2.load_raw()
    B, O, W = m2.summarise(D)
    OB = onset_boot(D, np.random.default_rng(0))
    OB.to_csv(OUT / "onset_bootstrap_N200.csv", index=False)
    print(OB.sort_values(["level", "onset_ms"]).round(1).to_string())
    for level in ("area_group", "area_acronym_custom"):
        if (OB.level == level).any():
            main_figure(plt, D, B, OB, level, col)
            summary_figure(plt, D, B, O, W, OB, level, col)
    captions(OB)
    print("ALL DONE", FIG)


if __name__ == "__main__":
    main()
