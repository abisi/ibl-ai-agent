"""011 -- Matched overlay + latency figures for every decoding type, per cohort (user 2026-09-30: "equivalent plots for
all decoding types then I choose which one to plot. Plot a barplot, no errorbar, of the latency of decoding from fast to
slow, vertical, color-legend, per cohort").
Per cohort: one row per decoding type (hit vs miss; whisker vs auditory stimulus-aligned; whisker vs auditory
first-lick-aligned), three square panels:
  a  time course, full window (stimulus -200..600 ms; lick -600..200 ms): real - paired shift null, mean over
     iterations, one line per area group (no error bands);
  b  time course, zoomed (hit vs miss -50..300 ms; whisker vs auditory stimulus -10..50 ms; lick -350..50 ms);
  c  onset latency, vertical bars from fast (left) to slow (right), no error bars, colours = area groups (legend);
     lick-aligned bars point down from the lick (earliest = most negative, left); areas without an onset listed under
     the axis.
Onset = first above-chance bin (5th percentile of real - null across iterations > 0) with >= 4 of the 5 bins from it
above (003.onset); stimulus-aligned targets searched after the stimulus. Areas as in the comparisons (005 EXCLUDE).
Every panel is also saved on its own (figures/011_panels/011_<panel>_<target>_<cohort>.pdf/.png, latency panels with
their own colour legend) so any combination can be assembled.
Outputs: figures/011_overlay_latency_<cohort>.pdf/.png, figures/011_panels/*, ../artifacts/011_onsets.csv
Run (repo root): python .../011_overlay_latency.py [n_iter] [cohorts, default R+ R-; "avg" = R+ and R- results averaged (stimulus decoding)]
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
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"
PAN = FIG / "011_panels"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S003 = _load("s003", "003_summarize_pseudopop.py")
S005 = _load("s005", "005_overlay_areas.py")
S012 = _load("s012", "012_rt_band.py")          # reaction-time band (grey, invisible y axis, bottom of the panel)
TARGETS = ["hitmiss", "modality_stim", "modality_lick"]
ROWLAB = {"hitmiss": "Hit vs miss", "modality_stim": "Whisker vs auditory\n(stimulus-aligned)",
          "modality_lick": "Whisker vs auditory\n(first-lick-aligned)"}
FULL = {"hitmiss": (-200, 600), "modality_stim": (-200, 600), "modality_lick": (-600, 200)}
ZOOM = {"hitmiss": (-50, 300), "modality_stim": (-10, 50), "modality_lick": (-350, 50)}
LAB = {"avg": "R+ and R− averaged"}


def cohort_D(tg, c, n_iter, order):
    d = pd.read_parquet(ART / f"002_pseudo_{tg}_{c}.parquet")
    d = d[d.skipped_reason.isna()]
    out = {}
    for a, g in d.groupby("area"):
        if a in S005.EXCLUDE or len(g) < n_iter:
            continue
        g = g.sort_values("rep").head(n_iter)
        out[a] = (np.stack(g.curve.map(np.asarray).to_numpy()) - np.stack(g.null_mean_curve.map(np.asarray).to_numpy()),
                  int(g.n_eligible_mice.iloc[0]))
    return out, d


def compute_avg(n_iter, order, T):
    """Cohort "avg" (user 2026-09-30): average of the R+ and R- pseudo-population results (no new decoding), stimulus
    decoding only. Per iteration i: d_i = (d_i(R+) + d_i(R-)) / 2 (iterations are independent draws, paired by index);
    curve = mean over iterations = average of the two cohort curves; above chance / onset from these averaged d_i."""
    out = {}
    for tg in ("modality_stim", "modality_lick"):
        align = "lick" if tg == "modality_lick" else "stim"
        t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
        (P, dp), (M, _) = cohort_D(tg, "R+", n_iter, order), cohort_D(tg, "R-", n_iter, order)
        areas = [a for a in order if a in P and a in M]
        search = t > 0 if align == "stim" else np.ones(len(t), bool)
        res = {}
        for a in areas:
            D = (P[a][0] + M[a][0]) / 2
            res[a] = dict(curve=D.mean(0), on=S003.onset((np.nanpercentile(D, 5, 0) > 0)[search], t[search]) * 1000,
                          mice=P[a][1] + M[a][1])
        out[tg] = dict(x=t * 1000, areas=areas, res=res, d=dp)
    return out


def compute(cohort, n_iter, order, T):
    if cohort == "avg":
        return compute_avg(n_iter, order, T)
    out = {}
    for tg in [x for x in TARGETS if (ART / f"002_pseudo_{x}_{cohort}.parquet").exists()]:
        align = "lick" if tg == "modality_lick" else "stim"
        t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
        d = pd.read_parquet(ART / f"002_pseudo_{tg}_{cohort}.parquet")
        d = d[d.skipped_reason.isna()]
        done = d.groupby("area").size()
        areas = [a for a in order if a not in S005.EXCLUDE and done.get(a, 0) >= n_iter]
        search = t > 0 if align == "stim" else np.ones(len(t), bool)
        res = {}
        for a in areas:
            g = d[d.area == a].sort_values("rep").head(n_iter)
            D = np.stack(g.curve.map(np.asarray).to_numpy()) - np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
            res[a] = dict(curve=D.mean(0), on=S003.onset((np.nanpercentile(D, 5, 0) > 0)[search], t[search]) * 1000,
                          mice=int(g.n_eligible_mice.iloc[0]))
        out[tg] = dict(x=t * 1000, areas=areas, res=res, d=d)
    return out


def tc(ax, X, colors, view, ref, ylabel, rt=None):
    x = X["x"]
    z = (x >= view[0]) & (x <= view[1])
    for a in X["areas"]:
        ax.plot(x[z], X["res"][a]["curve"][z], color=colors.get(a, "0.6"), lw=0.7)
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(*view)
    ax.set_xlabel(f"time from {ref} (ms)")
    ax.set_ylabel(ylabel)
    if rt is not None:
        S012.add_rt_band(ax, rt, view)


def latency(ax, X, colors, ref, legend):
    res = X["res"]
    has = sorted([a for a in X["areas"] if np.isfinite(res[a]["on"])], key=lambda a: res[a]["on"])
    none = [a for a in X["areas"] if not np.isfinite(res[a]["on"])]
    ax.bar(range(len(has)), [res[a]["on"] for a in has], color=[colors.get(a, "0.6") for a in has], width=0.8, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(has) - 0.3)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(f"onset from {ref} (ms)")
    ax.set_xlabel("fast → slow", labelpad=4)
    if none:
        ax.text(0.5, -0.16, "no onset: " + ", ".join(S005.ABBR.get(a, a) for a in none), transform=ax.transAxes,
                ha="center", va="top", fontsize=4.8, color="0.35")
    if legend:
        ax.legend(handles=[Patch(color=colors.get(a, "0.6"), label=a) for a in has], loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), fontsize=5, handlelength=1.0, handleheight=0.8, frameon=False,
                  title="fast → slow", title_fontsize=5.5)
    return has


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    colors, order = S005.allen()
    PAN.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort in (sys.argv[2:] or ["R+", "R-"]):          # "avg" = average of the R+ and R- results (stimulus decoding)
        R = compute(cohort, n_iter, order, T)
        TG = list(R)
        fig, axes = plt.subplots(len(TG), 3, figsize=(7.2, 2.35 * len(TG) + 0.35), squeeze=False)
        H = 2.35 * len(TG) + 0.35
        fig.subplots_adjust(left=0.1, right=0.98, top=1 - 0.55 / H, bottom=1.05 / H, hspace=0.62, wspace=0.55)
        for ax in axes.flat:
            ax.set_box_aspect(1)
        drawn = []
        for r, tg in enumerate(TG):
            X = R[tg]
            ref = "first lick" if tg == "modality_lick" else "stimulus"
            yl = f"{ROWLAB[tg]}\nbalanced accuracy - null"
            rt = S012.rt_values(tg, cohort)
            tc(axes[r, 0], X, colors, FULL[tg], ref, yl, rt)
            tc(axes[r, 1], X, colors, ZOOM[tg], ref, "balanced accuracy - null", rt)
            drawn += latency(axes[r, 2], X, colors, ref, legend=False)
            for a in X["areas"]:
                rows.append(dict(cohort=cohort, target=tg, area=a, n_eligible_mice=X["res"][a]["mice"],
                                 onset_ms=X["res"][a]["on"], n_iterations=n_iter))
            # single panels, each with its own legend where relevant
            for kind in ("full", "zoom", "latency"):
                f1, ax1 = plt.subplots(figsize=(2.9 if kind == "latency" else 2.2, 2.2))
                ax1.set_box_aspect(1)
                if kind == "latency":
                    latency(ax1, X, colors, ref, legend=True)
                    f1.subplots_adjust(left=0.2, right=0.62, bottom=0.2, top=0.9)
                else:
                    tc(ax1, X, colors, (FULL if kind == "full" else ZOOM)[tg], ref, yl, rt)
                    f1.subplots_adjust(left=0.28, right=0.95, bottom=0.2, top=0.9)
                ax1.set_title(f"{ROWLAB[tg].replace(chr(10), ' ')}, {LAB.get(cohort, cohort)}", fontsize=5.5)
                for ext in ("pdf", "png"):
                    f1.savefig(PAN / f"011_{kind}_{tg}_{cohort}.{ext}", dpi=300)
                plt.close(f1)
        for ax, lab, L in zip(axes[0], ("time course", "time course (zoom)", "onset latency"), "abc"):
            ax.set_title(lab, pad=7)
            ax.text(-0.35, 1.12, L, transform=ax.transAxes, fontweight="bold", fontsize=9)
        names = [g for g in order if g in set(drawn) | {a for tg in TG for a in R[tg]["areas"]}]
        fig.legend(handles=[Patch(color=colors.get(g, "0.6"), label=g) for g in names], loc="lower center", ncol=5,
                   fontsize=5.5, handlelength=1.0, handleheight=0.8, columnspacing=1.0, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        d0 = R[TG[0]]["d"]
        fig.suptitle(f"{LAB.get(cohort, cohort)}, learning day: pseudo-population decoding ({int(d0.n_mice.iloc[0])} mice x "
                     f"{int(d0.n_neurons.iloc[0])} neurons, {n_iter} iterations x {int(d0.n_null.median())} shifts"
                     f"{', PILOT' if n_iter < 1000 else ''})\ncurves = means across iterations; onset = first above-chance "
                     "bin (5th percentile of real - null > 0) with >= 4 of 5 above; stimulus-aligned: after the stimulus",
                     fontsize=6, y=0.995)
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"011_overlay_latency_{cohort}.{ext}", dpi=300)
        plt.close(fig)
    suffix = "" if not sys.argv[2:] else "_" + "_".join(sys.argv[2:])
    pd.DataFrame(rows).to_csv(ART / f"011_onsets{suffix}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
