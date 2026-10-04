"""004 -- Main and summary figures: where and when can stimulus modality (whisker vs auditory) be decoded?

Inputs: 001 raw iterations (all sessions pooled), 002 summaries, 003 matched-accuracy runs (if present).
Colours: area groups from allen_utils.get_custom_area_groups_colors() (Axel Bisi's custom Allen-based palette; our group
names mapped to its keys); fine areas: shades of their group colour.
Main figure (N = 200 neurons): a, b -- time course (50-ms bins, 5-ms steps) of the corrected balanced accuracy for the
area groups and the fine areas, mean +- s.d. over iterations, with one horizontal bar per area marking the bins above
chance (5th percentile of the corrected accuracy > 0); c, d -- first 50 ms at 20-ms bins / 2-ms steps (same bars, onset
triangles); e, f -- onset (first significant bin, 25-ms rule) ranked fast -> slow, error bars: 95 % range of the onset
over 1000 resamples of the iterations.
Summary figure: a method schematic, b group time courses + bars, c first 50 ms, d onset ranking (groups + areas),
e early accuracy vs number of neurons, f onset vs early accuracy, g matched-accuracy curves (003), h control: raw
balanced accuracy and trial-shuffle null for two areas.
Output: _stimulus_arrival/figures/arrival_main_N200.{png,pdf,svg}, arrival_summary.{png,pdf,svg}, captions *.md,
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
m2 = importlib.import_module("002_arrival_summary")
OUT, FIG = m2.OUT, m2.FIG
N_MAIN = 200
N_BOOT = 1000
GROUP_KEY = {"Motor areas": "Motor and frontal areas", "Somatosensory-whisker": "Somatosensory areas-whisker",
             "Auditory areas": "Auditory areas", "Midbrain": "Midbrain", "Striatum": "Striatum and pallidum",
             "Thalamus": "Thalamus"}
FINE_PARENT = {"SSp-bfd": "Somatosensory-whisker", "SSs": "Somatosensory-whisker", "SCm": "Midbrain",
               "MO-wM1": "Motor areas", "MO-wM2": "Motor areas", "MO-ALM": "Motor areas", "DMS": "Striatum", "DLS": "Striatum"}
GROUP_LABEL = {"Somatosensory-whisker": "SS-whisker", "Auditory areas": "Auditory", "Motor areas": "Motor-frontal",
               "Midbrain": "Midbrain", "Striatum": "Striatum", "Thalamus": "Thalamus"}


def palette():
    try:
        sys.path.insert(0, str(pathlib.Path.home() / "code"))
        from allen_utils import allen_utils as au
        g = au.get_custom_area_groups_colors()
    except Exception:                                                # copy of allen_utils (2026-10-04)
        g = {"Motor and frontal areas": "#1f9d5a", "Somatosensory areas-whisker": "#188064", "Auditory areas": "#019399",
             "Striatum and pallidum": "#98d6f9", "Thalamus": "#ff7080", "Midbrain": "#ff64ff"}
    from matplotlib.colors import to_rgb
    col = {a: g[k] for a, k in GROUP_KEY.items()}
    shades = {}
    for parent in set(FINE_PARENT.values()):
        kids = [a for a, p in FINE_PARENT.items() if p == parent]
        base = np.array(to_rgb(col[parent]))
        for i, a in enumerate(kids):
            f = np.linspace(-0.35, 0.35, len(kids))[i] if len(kids) > 1 else 0
            shades[a] = tuple(np.clip(base * (1 - f) if f > 0 else base + (1 - base) * (-f), 0, 1))
    col.update(shades)
    return col


def load():
    D = m2.load_raw()
    B, O, W = m2.summarise(D)
    return D, B, O, W


def onset_boot(D, rng):
    rows = []
    nw = len(m2.WIDE_T)
    for (level, area, N), g in D[D.N == N_MAIN].groupby(["level", "area", "N"]):
        d = m2.d_matrix(g)[:, nw:]
        k = len(d)
        boots = []
        for _ in range(N_BOOT):
            x = d[rng.integers(0, k, k)]
            boots.append(1000 * m2.onset(m2.ZOOM_T, np.percentile(x, 5, axis=0) > 0, 13, 11))
        boots = np.array(boots)
        obs = 1000 * m2.onset(m2.ZOOM_T, np.percentile(d, 5, axis=0) > 0, 13, 11)
        rows.append(dict(level=level, area=area, N=N, onset_ms=obs, lo=np.nanpercentile(boots, 2.5), hi=np.nanpercentile(boots, 97.5),
                         frac_defined=np.isfinite(boots).mean()))
    return pd.DataFrame(rows)


def label(a):
    return GROUP_LABEL.get(a, a)


def curves_panel(ax, B, areas, level, res, col, xlim, ylim, bars=True, onset=None, lw=0.9, names=False):
    for a in areas:
        q = B[(B.level == level) & (B.area == a) & (B.N == N_MAIN) & (B.resolution == res)]
        if not len(q):
            continue
        t = 1000 * q.t.to_numpy()
        ax.fill_between(t, q["mean"] - q.sd, q["mean"] + q.sd, color=col[a], alpha=0.16, lw=0, edgecolor="none")
        ax.plot(t, q["mean"], color=col[a], lw=lw, label=label(a))
    ax.axhline(0, color="0.5", lw=0.4)
    ax.axvline(0, color="0.3", lw=0.4, ls="--")
    ax.set_xlim(*xlim)
    if bars:                                      # one significance bar per area above the curves
        y0, dy = ylim[1] + 0.025, 0.026
        for i, a in enumerate(areas):
            q = B[(B.level == level) & (B.area == a) & (B.N == N_MAIN) & (B.resolution == res)]
            if not len(q):
                continue
            t, sig = 1000 * q.t.to_numpy(), q.sig.to_numpy()
            yy = y0 + i * dy
            step = t[1] - t[0]
            for s_, e_ in runs(sig):
                x0_, x1_ = max(t[s_] - step / 2, xlim[0]), min(t[e_] + step / 2, xlim[1])
                if x1_ > x0_:
                    ax.plot([x0_, x1_], [yy, yy], color=col[a], lw=1.6, solid_capstyle="butt", clip_on=False)
            if names:
                ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), yy, label(a), color=col[a], fontsize=4.3, va="center",
                        ha="left", weight="bold")
            if onset is not None:
                o = onset.get(a, np.nan)
                if np.isfinite(o):
                    ax.plot([o], [yy], marker="|", ms=4, mew=0.9, color="k", clip_on=False, zorder=6)
        ax.set_ylim(ylim[0], y0 + len(areas) * dy)
        ax.spines["left"].set_bounds(ylim[0], ylim[1])
        ax.set_yticks([t_ for t_ in ax.get_yticks() if ylim[0] <= t_ <= ylim[1]])
    else:
        ax.set_ylim(*ylim)


def runs(sig):
    out, s = [], None
    for i, v in enumerate(sig):
        if v and s is None:
            s = i
        if (not v or i == len(sig) - 1) and s is not None:
            out.append((s, i if v else i - 1))
            s = None
    return out


def rank_panel(ax, OB, col, areas):
    q = OB[OB.area.isin(areas)].copy()
    q["key"] = q.onset_ms.fillna(1e9)
    q = q.sort_values("key").reset_index(drop=True)
    y = np.arange(len(q))
    ax.barh(y, q.onset_ms, color=[col[a] for a in q.area], height=0.68, lw=0)
    ax.errorbar(q.onset_ms, y, xerr=[q.onset_ms - q.lo, q.hi - q.onset_ms], fmt="none", ecolor="0.2", lw=0.6, capsize=0)
    ax.set_yticks(y, [label(a) for a in q.area], fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel("First significant bin (ms)")
    for yi, v, h in zip(y, q.onset_ms, q.hi):
        ax.text(max(v, h) + 0.6, yi, f"{v:.0f} ms", va="center", fontsize=4.6, color="0.25")
    ax.set_xlim(0, np.nanmax(q.hi) * 1.25)


def main_figure(plt, B, OB, col):
    G, Fn = m2.COARSE, m2.FINE
    on = {r.area: r.onset_ms for r in OB.itertuples()}
    W = S.W_IN
    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], height_ratios=[1.15, 1.25, 0.95], wspace=0.25, hspace=0.62,
                          left=0.08, right=0.98, top=0.9, bottom=0.07)
    yl = (-0.03, 0.52)
    axes = []
    for r, (areas, level, ttl) in enumerate([(G, "area_group", "Area groups"), (Fn, "area_acronym_custom", "Areas")]):
        ax = fig.add_subplot(gs[r, 0])
        curves_panel(ax, B, areas, level, "wide", col, (-200, 600), yl, onset=None, names=True)
        ax.axvspan(-20, 50, color="0.9", lw=0, edgecolor="none", zorder=0)
        ax.set_ylabel("Corrected balanced accuracy")
        ax.set_title(f"{ttl}: 50-ms bins, 5-ms steps (N = {N_MAIN} neurons)", loc="left", fontsize=6)
        ax2 = fig.add_subplot(gs[r, 1])
        curves_panel(ax2, B, areas, level, "zoom", col, (-20, 50), yl, onset=on)
        ax2.set_title("First 50 ms: 20-ms bins, 2-ms steps", loc="left", fontsize=6)
        axes += [ax, ax2]
    for ax in axes[2:]:
        ax.set_xlabel("Time from stimulus onset (ms)")
    axr1 = fig.add_subplot(gs[2, 0]); axr2 = fig.add_subplot(gs[2, 1])
    gq = gs[2, 0].subgridspec(1, 2, wspace=0.6)
    axr1.remove()
    a1, a2 = fig.add_subplot(gq[0, 0]), fig.add_subplot(gq[0, 1])
    rank_panel(a1, OB[OB.level == "area_group"], col, G)
    a1.set_title("Onset ranking: area groups", loc="left", fontsize=6)
    rank_panel(a2, OB[OB.level == "area_acronym_custom"], col, Fn)
    a2.set_title("Onset ranking: areas", loc="left", fontsize=6)
    axr2.set_axis_off()
    axr2.text(0, 0.95, "Bars above the curves: bins where the corrected\naccuracy is above chance (5th percentile over\n"
              "iterations > 0); black tick: onset (first bin with\n>= 80 % of the next 25 ms above chance).\n"
              "Ranking error bars: 95 % range over 1000\nresamples of the iterations.", va="top", fontsize=5, color="0.3",
              transform=axr2.transAxes)
    S.letter_row(fig, axes[:2], "ab"); S.letter_row(fig, axes[2:4], "cd"); S.letter_row(fig, [a1, a2], "ef")
    fig.suptitle("When does stimulus modality reach each area? Whisker vs auditory decoding, all sessions pooled",
                 x=0.08, y=0.995, ha="left", fontsize=7, weight="bold")
    S.save(fig, FIG, "arrival_main_N200")
    plt.close(fig)


def summary_figure(plt, D, B, O, W, OB, col):
    from matplotlib.patches import FancyBboxPatch
    G, Fn = m2.COARSE, m2.FINE
    on = {r.area: r.onset_ms for r in OB.itertuples()}
    fig = plt.figure(figsize=(S.W_IN, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], wspace=0.5, hspace=0.55, left=0.08, right=0.98, top=0.93, bottom=0.06)
    # a: schematic
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    boxes = [(0.2, 7.2, "122 sessions\n(2 cohorts,\nall days)"), (3.6, 7.2, "20 sessions\nx N/20 neurons\nof the area"),
             (7.0, 7.2, "pseudo-trials\nwhisker vs\nauditory"), (0.2, 2.2, "L2 logistic\nregression\nper bin, 3-fold CV"),
             (3.6, 2.2, "same draw,\nlabels shuffled\nwithin sessions"), (7.0, 2.2, "corrected\naccuracy =\nreal - shuffled")]
    for x0, y0, t in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), 2.8, 2.3, boxstyle="round,pad=0.1", fc="0.95", ec="0.5", lw=0.5))
        ax.text(x0 + 1.4, y0 + 1.15, t, ha="center", va="center", fontsize=4.2, linespacing=1.15)
    for (x0, y0), (x1, y1) in [((3.1, 8.35), (3.55, 8.35)), ((6.5, 8.35), (6.95, 8.35)), ((8.4, 7.1), (1.6, 4.6)),
                               ((3.1, 3.35), (3.55, 3.35)), ((6.5, 3.35), (6.95, 3.35))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax.set_title("Pseudo-population decoding", loc="left", fontsize=6)
    yl = (-0.03, 0.52)
    axb = fig.add_subplot(gs[0, 1:])
    curves_panel(axb, B, G, "area_group", "wide", col, (-200, 600), yl, names=True)
    axb.axvspan(-20, 50, color="0.9", lw=0, edgecolor="none", zorder=0)
    axb.set_ylabel("Corrected balanced accuracy"); axb.set_xlabel("Time from stimulus onset (ms)")
    axb.set_title(f"Area groups, N = {N_MAIN} neurons (bars: above chance)", loc="left", fontsize=6)
    axc = fig.add_subplot(gs[1, 0])
    curves_panel(axc, B, G + Fn, "area_group", "zoom", col, (-20, 50), yl, onset=on, bars=False)
    curves_panel(axc, B, Fn, "area_acronym_custom", "zoom", col, (-20, 50), yl, onset=on, bars=False, lw=0.5)
    axc.set_ylabel("Corrected balanced accuracy"); axc.set_xlabel("Time from stimulus onset (ms)")
    axc.set_title("First 50 ms (20-ms bins, 2-ms steps)\nthick: area groups, thin: areas (colours as in d)", loc="left", fontsize=5.6)
    axd = fig.add_subplot(gs[1, 1])
    rank_panel(axd, OB, col, G + Fn)
    axd.set_title("Onset ranking (N = 200)", loc="left", fontsize=6)
    axe = fig.add_subplot(gs[1, 2])
    for a in G + Fn:
        q = W[(W.area == a) & W.N.isin(m2.N_GRID)].sort_values("N")
        if len(q):
            axe.errorbar(q.N, q["mean"], [q["mean"] - q.lo, q.hi - q["mean"]], color=col[a], marker="o", ms=1.8,
                         lw=0.9 if a in G else 0.5, ls="-" if a in G else "--", elinewidth=0.4, capsize=0)
    from matplotlib.ticker import NullFormatter, NullLocator
    axe.set_xscale("log"); axe.set_xticks(m2.N_GRID, [str(n) for n in m2.N_GRID])
    axe.xaxis.set_minor_locator(NullLocator()); axe.xaxis.set_minor_formatter(NullFormatter())
    axe.set_xlabel("Neurons in the pseudo-population"); axe.set_ylabel("Corrected accuracy, 5-50 ms")
    axe.set_title("Control: number of neurons (solid: groups)", loc="left", fontsize=6)
    axf = fig.add_subplot(gs[2, 0])
    M = O[O.resolution == "zoom"].merge(W[["level", "area", "N", "mean"]], on=["level", "area", "N"])
    M = M[M.N.isin(m2.N_GRID)]
    st = S.corr_panel(axf, M["mean"].to_numpy(), M.onset_ms.to_numpy(), colors=M.area.map(col).to_numpy(),
                      sizes=(3 + 10 * np.log(M.N / 20 + 1) / np.log(26)).to_numpy())
    axf.text(0.98, 0.97, f"rho = {st['rho']:.2f}, {S.fmt_p(st['p_rho'])}", transform=axf.transAxes, ha="right", va="top", fontsize=4.8)
    axf.set_xlabel("Corrected accuracy, 5-50 ms"); axf.set_ylabel("First significant bin (ms)")
    axf.set_title("Onset vs early accuracy (area x N)", loc="left", fontsize=6)
    axg = fig.add_subplot(gs[2, 1])
    P = OUT / "matched_n.csv"
    if P.exists():
        Pm = pd.read_csv(P)
        Pm = Pm[Pm.reference == "Somatosensory-whisker"]
        for r in Pm.itertuples():
            if not np.isfinite(r.matched_N) or r.level != "area_group":
                continue
            q = B[(B.level == r.level) & (B.area == r.area) & (B.N == int(r.matched_N)) & (B.resolution == "zoom")]
            if len(q):
                axg.plot(1000 * q.t, q["mean"], color=col[r.area], lw=1.1 if r.area == "Somatosensory-whisker" else 0.7,
                         label=f"{label(r.area)} ({int(r.matched_N)})")
        axg.legend(frameon=False, fontsize=4.3, loc="upper right", title="neurons", title_fontsize=4.4)
    else:
        axg.text(0.5, 0.5, "matched-accuracy runs pending", ha="center", va="center", transform=axg.transAxes, fontsize=5)
    axg.axhline(0, color="0.5", lw=0.4); axg.axvline(0, color="0.3", lw=0.4, ls="--"); axg.set_xlim(-20, 100)
    axg.set_xlabel("Time from stimulus onset (ms)"); axg.set_ylabel("Corrected balanced accuracy")
    axg.set_title("Control: matched early accuracy\n(SS-whisker at 100 neurons)", loc="left", fontsize=6)
    axh = fig.add_subplot(gs[2, 2])
    nw = len(m2.WIDE_T)
    for a, lv in (("Somatosensory-whisker", "area_group"), ("Motor areas", "area_group")):
        g = D[(D.area == a) & (D.N == N_MAIN) & (D.level == lv)]
        real = np.stack(g.curve.map(np.asarray))[:, nw:]
        null = np.stack(g.null_mean.map(np.asarray))[:, nw:]
        t = 1000 * m2.ZOOM_T
        axh.plot(t, real.mean(0), color=col[a], lw=0.9, label=f"{label(a)}: real")
        axh.plot(t, null.mean(0), color=col[a], lw=0.9, ls=":", label=f"{label(a)}: shuffled")
    axh.axhline(0.5, color="0.5", lw=0.4); axh.axvline(0, color="0.3", lw=0.4, ls="--"); axh.set_xlim(-20, 100)
    axh.set_xlabel("Time from stimulus onset (ms)"); axh.set_ylabel("Balanced accuracy")
    axh.legend(frameon=False, fontsize=4.3, loc="upper left")
    axh.set_title("Control: trial-shuffle null", loc="left", fontsize=6)
    S.letter_row(fig, [ax, axb], "ab"); S.letter_row(fig, [axc, axd, axe], "cde"); S.letter_row(fig, [axf, axg, axh], "fgh")
    fig.suptitle("Where and when can stimulus modality be decoded? Whisker vs auditory, all sessions pooled",
                 x=0.08, y=0.995, ha="left", fontsize=7.2, weight="bold")
    S.save(fig, FIG, "arrival_summary")
    plt.close(fig)


def captions(D, OB, W):
    nses = D.n_eligible_sessions.max()
    g = OB[OB.level == "area_group"].sort_values("onset_ms")
    txt = f"""# Stimulus-modality decoding across the brain

**Where and when can stimulus modality be decoded?** Pseudo-population decoding of whisker vs auditory trials, all
sessions pooled (both cohorts, learning day and expert days; good + mua neurons; whisker-artefact-corrected spikes).
One iteration: 20 sessions drawn with replacement, N/20 neurons of the area drawn within each, pseudo-trials built by
balanced reuse of each session's trials; L2 logistic regression per time bin (3-fold cross-validation on real trials,
inner 2-fold for the regularisation); the same draw is re-decoded 10 times with trial labels shuffled within sessions;
corrected balanced accuracy = real - mean shuffled (0 = chance, 0.5 = perfect). 100 iterations (pilot value).
Above chance in a bin: 5th percentile of the corrected accuracy over iterations > 0. Onset: first post-stimulus bin
above chance with >= 80 % of the bins in the next 25 ms above chance. Wide time course: causal 50-ms bins, 5-ms steps;
first 50 ms: causal 20-ms bins, 2-ms steps (bins labelled at their end). Colours: area groups (allen_utils custom
palette), areas in shades of their group.
Onsets at N = {N_MAIN} (20-ms bins): """ + "; ".join(f"{label(r.area)} {r.onset_ms:.0f} ms" for r in g.itertuples()) + ".\n"
    (OUT / "arrival_figures_caption.md").write_text(txt, encoding="utf-8")


def main():
    plt = S.setup()
    col = palette()
    D, B, O, W = load()
    OB = onset_boot(D, np.random.default_rng(0))
    OB.to_csv(OUT / "onset_bootstrap_N200.csv", index=False)
    print(OB.round(1).to_string())
    main_figure(plt, B, OB, col)
    summary_figure(plt, D, B, O, W, OB, col)
    captions(D, OB, W)
    print("ALL DONE", FIG)


if __name__ == "__main__":
    main()
