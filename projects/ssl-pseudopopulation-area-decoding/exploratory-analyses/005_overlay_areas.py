"""005 -- Publication-style overlay of area groups per cohort (user requests 2026-09-29): areas coloured by area group
(ephys_utilities allen_utils.get_custom_area_groups_colors); no error bands on overlaid curves; whole brain, Amygdala
and hypothalamus, Cortical subplate, Visual areas and Pons and medulla not shown (user).
Layout (2 rows x 3 square cells):
  row 1  a  hit vs miss time course (-200..600 ms from stimulus): d = real - paired shift null, mean across iterations
            per area, thin lines;
         b  onset latency per area as a bar plot, fastest on top (bar from the stimulus to the onset); areas without an
            onset listed at the bottom as "none";
  row 2  c  whisker vs auditory time course in ONE square cell with two x segments separated by white space:
            stimulus-aligned (-10..50 ms) | first-lick-aligned (-600..200 ms), shared y axis;
         d  onset latency, stimulus-aligned;   e  onset latency, first-lick-aligned (bars from the lick to the onset).
Onset = first bin that is itself above chance with >= 4 of the 5 bins from it above (003.onset); above chance = 5th
percentile of d across iterations > 0; stimulus-aligned onsets searched after the stimulus only (bin end > 0).
The stats csv also keeps a window effect per area (mean d in the window, one-sided bootstrap p = fraction of iterations
with window d <= 0, Benjamini-Hochberg across the shown areas within target x cohort).
Only areas with the full iteration count are drawn.
Outputs: figures/005_overlay_areas_<cohort>.pdf/.png, ../artifacts/005_overlay_stats_<cohort>.csv
Run (repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/005_overlay_areas.py <cohort> [n_iter]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"
_spec = importlib.util.spec_from_file_location("s003", HERE / "003_summarize_pseudopop.py")
S003 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S003)
TARGETS = ["hitmiss", "modality_stim", "modality_lick"]
WINDOW = {"hitmiss": ("post-stimulus", (0.005, 0.6)), "modality_stim": ("sensory", (0.005, 0.05)),
          "modality_lick": ("pre-lick", (-0.10, 0.0))}
SHORT = {"hitmiss": "Hit vs miss", "modality_stim": "Whisker vs auditory\n(stimulus-aligned)",
         "modality_lick": "Whisker vs auditory\n(first-lick-aligned)"}
EXCLUDE = {"All units", "Amygdala and hypothalamus", "Cortical subplate", "Visual areas", "Pons and medulla",
           "Somatosensory-body", "Olfactory areas"}   # SS-body (2-4 mice) and olfactory excluded (user 2026-09-29)
VIEW = {"hitmiss": (-200, 600), "modality_stim": (-10, 50), "modality_lick": (-600, 200)}   # ms shown
POST_ONSET = {"hitmiss", "modality_stim"}
ABBR = {"Motor areas": "Motor", "Frontal areas": "Frontal", "Somatosensory-orofacial": "SS-orofacial",
        "Somatosensory-body": "SS-body", "Somatosensory-whisker": "SS-whisker", "Auditory areas": "Auditory",
        "Retrosplenial areas": "RSP", "Posterior parietal areas": "PPC", "Visual areas": "Visual",
        "Insular areas": "Insular", "Hippocampus": "HPC", "Striatum": "Striatum", "Pallidum": "Pallidum",
        "Lateral septal complex": "LSX", "Thalamus": "Thalamus", "Midbrain": "Midbrain", "Olfactory areas": "Olfactory"}
SEG = (1.0, 0.35, 2.2)                 # width ratios of the combined modality cell: stim | gap | lick


def allen():
    from axel_bisi_paths import axel_bisi_root
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
    import ephys_utilities.allen_utils.allen_utils as allen_utils
    return allen_utils.get_custom_area_groups_colors(), allen_utils.get_area_group_custom_order()


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def label_points(ax, pts, fs=5):
    """Short-name labels next to each point, avoiding the markers and already placed labels: tries right / left /
    above / below at increasing distance (in points); thin leader line when the label ended up away from its point.
    (Used by 006.)"""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    P = {a: ax.transData.transform(pts[a]) * to_pt for a in pts}
    r = 3.0
    boxes = [(px - r, py - r, 2 * r, 2 * r) for px, py in P.values()]
    hit = lambda x0, y0, w, h: any(x0 < bx + bw and bx < x0 + w and y0 < by + bh and by < y0 + h for bx, by, bw, bh in boxes)
    for a in sorted(pts, key=lambda k: -pts[k][1]):
        px, py = P[a]
        w, h = 0.55 * fs * len(ABBR.get(a, a)), 1.1 * fs
        cands = []
        for dd in np.arange(0, 30, 1.5):
            cands += [(px + r + 1, py - h / 2 + s * dd) for s in (1, -1)] + \
                     [(px - r - 1 - w, py - h / 2 + s * dd) for s in (1, -1)]
        x0, y0 = next(((cx, cy) for cx, cy in cands if not hit(cx, cy, w, h)), cands[0])
        boxes.append((x0, y0, w, h))
        far = abs(y0 + h / 2 - py) > h
        ax.annotate(ABBR.get(a, a), pts[a], xytext=(x0 - px, y0 - py), textcoords="offset points", fontsize=fs,
                    color="0.2", va="bottom", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="0.6", shrinkA=0, shrinkB=2.5) if far else None)


def compute(target, cohort, n_iter, order, T):
    align = "lick" if target == "modality_lick" else "stim"
    t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
    d = pd.read_parquet(ART / f"002_pseudo_{target}_{cohort}.parquet")
    d = d[d.skipped_reason.isna()]
    done = d.groupby("area").size()
    areas = [a for a in order if a not in EXCLUDE and done.get(a, 0) >= n_iter]
    wname, (w0, w1) = WINDOW[target]
    wm = (t >= w0) & (t <= w1)
    search = t > 0 if target in POST_ONSET else np.ones(len(t), bool)
    res, rows = {}, []
    for a in areas:
        g = d[d.area == a].sort_values("rep").head(n_iter)
        R = np.stack(g.curve.map(np.asarray).to_numpy())
        N = np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
        D = R - N
        above = np.nanpercentile(D, 5, 0) > 0
        dw = np.nanmean(D[:, wm], 1)
        res[a] = dict(D=D.mean(0), on=S003.onset(above[search], t[search]) * 1000, dw=dw.mean(),
                      p=max(float((dw <= 0).mean()), 1 / len(dw)), mice=int(g.n_eligible_mice.iloc[0]))
    q = bh([res[a]["p"] for a in areas])
    for a, qa in zip(areas, q):
        rows.append(dict(cohort=cohort, target=target, area=a, n_eligible_mice=res[a]["mice"], n_iterations=n_iter,
                         peak_d=np.nanmax(res[a]["D"]), t_peak_d_ms=t[np.nanargmax(res[a]["D"])] * 1000,
                         onset_ms=res[a]["on"], onset_search="post-stimulus" if target in POST_ONSET else "full window",
                         window=wname, d_window=res[a]["dw"], p_window=res[a]["p"], q_window_bh=qa))
    return t * 1000, areas, res, rows, d


def curves(ax, x, areas, res, colors, view):
    z = (x >= view[0]) & (x <= view[1])
    for a in areas:
        ax.plot(x[z], res[a]["D"][z], color=colors.get(a, "0.6"), lw=0.7)
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(*view)


def latency_bars(ax, areas, res, colors, ref):
    """Onset latency per area, fastest on top; bar from the alignment time (0) to the onset; no-onset areas at the
    bottom, labelled 'none'."""
    has = sorted([a for a in areas if np.isfinite(res[a]["on"])], key=lambda a: res[a]["on"])
    none = [a for a in areas if not np.isfinite(res[a]["on"])]
    ordered = has + none
    n = len(ordered)
    for k, a in enumerate(ordered):
        y = n - 1 - k
        if a in has:
            ax.barh(y, res[a]["on"], color=colors.get(a, "0.6"), height=0.72, lw=0)
            v = res[a]["on"]
            ax.text(v + (1 if v >= 0 else -1) * 0.02 * max(1, max(abs(res[b]["on"]) for b in has)), y, f"{v:.0f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=4.5, color="0.25")
        else:
            ax.text(0, y, " none", va="center", ha="left", fontsize=4.5, color="0.4")
    ax.set_yticks(range(n))
    ax.set_yticklabels([ABBR.get(a, a) for a in ordered[::-1]], fontsize=5)
    for lab, a in zip(ax.get_yticklabels(), ordered[::-1]):
        lab.set_color(colors.get(a, "0.3"))
    ax.set_ylim(-0.7, n - 0.3)
    ax.axvline(0, color="k", lw=0.5)
    ax.tick_params(axis="y", length=0)
    vals = [res[a]["on"] for a in has] or [0]
    lo, hi = min(0, min(vals)), max(0, max(vals))
    span = hi - lo or 1
    ax.set_xlim(lo - 0.18 * span * (lo < 0), hi + 0.18 * span * (hi > 0) + 0.02 * span)
    ax.set_xlabel(f"onset from {ref} (ms)")


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    cohort = sys.argv[1]
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    colors, order = allen()
    R = {tg: compute(tg, cohort, n_iter, order, T) for tg in TARGETS}
    rows = [r for tg in TARGETS for r in R[tg][3]]
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = GridSpec(2, 3, figure=fig, left=0.08, right=0.98, top=0.86, bottom=0.1, hspace=0.55, wspace=0.55)
    # row 1: hit vs miss
    x, areas, res, _, d0 = R["hitmiss"]
    axA = fig.add_subplot(gs[0, 0])
    curves(axA, x, areas, res, colors, VIEW["hitmiss"])
    axA.set_box_aspect(1)
    axA.set_xlabel("time from stimulus (ms)")
    axA.set_ylabel("Hit vs miss\nbalanced accuracy - null")
    axB = fig.add_subplot(gs[0, 1])
    latency_bars(axB, areas, res, colors, "stimulus")
    axB.set_box_aspect(1)
    # row 2: whisker vs auditory, stimulus- and lick-aligned in one square cell (white space between)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0], width_ratios=SEG, wspace=0)
    axS = fig.add_subplot(sub[0, 0])
    axL = fig.add_subplot(sub[0, 2], sharey=axS)
    tot = sum(SEG)
    axS.set_box_aspect(tot / SEG[0])
    axL.set_box_aspect(tot / SEG[2])
    xs, ars, rs, _, _ = R["modality_stim"]
    xl, arl, rl, _, _ = R["modality_lick"]
    curves(axS, xs, ars, rs, colors, VIEW["modality_stim"])
    curves(axL, xl, arl, rl, colors, VIEW["modality_lick"])
    axS.set_xticks([0, 40])
    axL.set_xticks([-400, 0, 200])
    axS.set_xlabel("from stimulus\n(ms)")
    axL.set_xlabel("from first lick (ms)")
    axS.set_ylabel("Whisker vs auditory\nbalanced accuracy - null")
    axL.spines["left"].set_visible(False)
    axL.tick_params(axis="y", left=False, labelleft=False)
    axD = fig.add_subplot(gs[1, 1])
    latency_bars(axD, ars, rs, colors, "stimulus")
    axD.set_box_aspect(1)
    axE = fig.add_subplot(gs[1, 2])
    latency_bars(axE, arl, rl, colors, "first lick")
    axE.set_box_aspect(1)
    for ax, lab, L in ((axA, "time course", "a"), (axB, "onset latency", "b"), (axS, "time course", "c"),
                       (axD, "onset latency, stimulus", "d"), (axE, "onset latency, first lick", "e")):
        ax.set_title(lab, pad=6, loc="left" if ax is axS else "center")
        ax.text(-0.3 if ax is not axS else -0.9, 1.1, L, transform=ax.transAxes, fontweight="bold", fontsize=9)
    # legend in the free cell of row 1
    from matplotlib.lines import Line2D
    axLeg = fig.add_subplot(gs[0, 2])
    axLeg.set_axis_off()
    names = [g for g in order if any(g in R[tg][1] for tg in TARGETS)]
    axLeg.legend(handles=[Line2D([], [], color=colors.get(g, "0.6"), lw=2) for g in names], labels=names,
                 loc="center left", fontsize=5.5, handlelength=1.4, title="area group", title_fontsize=6)
    fig.suptitle(f"{cohort}, learning day: pseudo-population decoding ({int(d0.n_mice.iloc[0])} mice x "
                 f"{int(d0.n_neurons.iloc[0])} neurons, {n_iter} iterations x {int(d0.n_null.median())} shifts, PILOT counts)\n"
                 "curves = means across iterations; onset = first above-chance bin (5th percentile of real - null > 0) with "
                 ">= 4 of 5 above;\nstimulus-aligned onsets searched after the stimulus only", fontsize=6.3, y=0.995)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"005_overlay_areas_{cohort}.{ext}", dpi=300)
    S = pd.DataFrame(rows)
    S.to_csv(ART / f"005_overlay_stats_{cohort}.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
