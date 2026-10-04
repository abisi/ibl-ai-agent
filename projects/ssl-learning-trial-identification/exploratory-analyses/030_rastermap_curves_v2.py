"""030 -- Rastermap-sorted learning-curve heatmaps, v2 (user 2026-10-01: "Redo rastermap, changing parameters to find
structure, change cmap (use like the rplus color). Also do it joint on both cohorts (color code by group color). Do not do
further clustering for now. Also sort by difference of whisker and false alarm").
Data as 029: day-0 per-mouse HMM curves (sigma = 1; 024) -- whisker, false alarm (no stim), auditory, whisker - FA -- on
the whisker-trial axis starting at the first whisker trial, normalised to NPROG points of session progress. Versions:
untrimmed | trimA1.
Sort sources: whisker, FA, auditory, joint [whisker | FA], whisker - FA. Each sort is applied to every displayed curve.
Groups: R+ alone, R- alone, and both cohorts together (left strip = cohort colour).
Parameter search (per group x sort source x version): rastermap (n_clusters=None, i.e. the mice are sorted directly) over
locality {0, 0.5, 0.9} x n_PCs {5, 10, 20} x normalize {False, True}. Structure score = mean Pearson correlation between
ADJACENT rows of the sorted source matrix (how similar neighbours are; higher = smoother, more structured ordering),
reported relative to a random order (mean of 200 shuffles). The best-scoring parameters are used for the heatmaps; every
score is saved (artifacts/030_rastermap_param_scores.csv) and the sweep is shown for the joint sort
(030_param_sweep_<group><tag>.png: whisker curve under each parameter set, score in the title).
Colour maps (user 2026-10-01): P(lick) inferno; whisker - FA: R- purple (negative) -> white (0) -> R+ green (positive).
No clustering (user).
Stacked version (user 2026-10-01): R+ and R- each sorted within cohort (own best parameters), plotted in ONE matrix per
panel, R+ on top and R- below (line between): 030_rastermap_stacked<tag>.{png,pdf}.
Figures: 030_rastermap_<group><tag>.{png,pdf}: rows = sort source, columns = displayed curve; left strips = cohort
(group colour; both-cohort figure) and learning category; white tick = learning trial (lenient cascade, 028 table, all
100 mice).
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/030_rastermap_curves_v2.py
"""

from __future__ import annotations

import itertools
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}          # GROUP_COLORS rplus / rminus
CAT_COL = {"good": "#1a9850", "moderate": "#fee08b", "bad": "#d73027", "NA": "0.85"}
CMAP_P = "inferno"                                # user 2026-10-01
CMAP_D = LinearSegmentedColormap.from_list("rminus_white_rplus", [COL["R-"], "#ffffff", COL["R+"]])   # user: white centre
NPROG = 101
FS_M, FS_S = 7, 6
DISPLAY = [("whisker", "Whisker"), ("fa", "False alarm"), ("auditory", "Auditory"), ("w-fa", "Whisker − FA")]
SORTS = [("whisker", "whisker"), ("fa", "false alarm"), ("auditory", "auditory"), ("joint", "joint whisker | FA"),
         ("w-fa", "whisker − FA")]
GRID = list(itertools.product([0.0, 0.5, 0.9], [5, 10, 20], [False, True]))   # locality, n_PCs, normalize
LT_DEF = "lenient cascade"


def to_progress(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return np.full(NPROG, np.nan)
    return np.interp(np.linspace(0, 1, NPROG), np.linspace(0, 1, len(v)), v)


def neighbour_score(X, order):
    Z = X[order]
    Z = Z - Z.mean(1, keepdims=True)
    nrm = np.linalg.norm(Z, axis=1)
    nrm[nrm == 0] = 1
    return float(np.mean(np.sum(Z[1:] * Z[:-1], 1) / (nrm[1:] * nrm[:-1])))


def rm_sort(X, locality, n_pcs, normalize):
    from rastermap import Rastermap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Rastermap(n_clusters=None, n_PCs=min(n_pcs, X.shape[0] - 1), locality=locality, time_lag_window=0,
                      normalize=normalize, verbose=False, random_state=0).fit(X)
    return np.asarray(m.isort)


def best_sort(X, rng, rows, ctx):
    X = np.nan_to_num(X, nan=np.nanmean(X))
    rnd = np.mean([neighbour_score(X, rng.permutation(len(X))) for _ in range(200)])
    best, all_orders = None, {}
    for loc, npc, nz in GRID:
        try:
            o = rm_sort(X, loc, npc, nz)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(**ctx, locality=loc, n_pcs=npc, normalize=nz, score=np.nan, error=repr(e)[:100]))
            continue
        s = neighbour_score(X, o)
        all_orders[(loc, npc, nz)] = (o, s)
        rows.append(dict(**ctx, locality=loc, n_pcs=npc, normalize=nz, score=s, score_random=rnd, error=None))
        if best is None or s > best[1]:
            best = ((loc, npc, nz), s, o)
    return best, rnd, all_orders


def strip(ax, values, colors, where):
    sx = ax.inset_axes(where)
    keys = list(colors)
    sx.imshow(np.array([keys.index(v) for v in values])[:, None], aspect="auto", cmap=ListedColormap(list(colors.values())),
              vmin=0, vmax=len(keys) - 1, interpolation="nearest")
    sx.set_axis_off()


def heatmaps(D, meta, group, tag, sorts):
    n = len(meta)
    fig, axes = plt.subplots(len(SORTS), 4, figsize=(8.27, 11.0))
    fig.subplots_adjust(left=0.13, right=0.92, top=0.94, bottom=0.04, wspace=0.1, hspace=0.3)
    for r, (skey, slab) in enumerate(SORTS):
        (params, score, order), rnd = sorts[skey]
        for c, (dkey, dlab) in enumerate(DISPLAY):
            ax = axes[r, c]
            M = D[dkey][order]
            im = ax.imshow(M, aspect="auto", cmap=CMAP_D if dkey == "w-fa" else CMAP_P,
                           vmin=-1 if dkey == "w-fa" else 0, vmax=1, interpolation="nearest",
                           extent=(0, 1, n - 0.5, -0.5))
            ltf = meta["lt_frac"].to_numpy()[order]
            ok = np.isfinite(ltf)
            ax.scatter(ltf[ok], np.arange(n)[ok], marker="|", s=6, color="k" if dkey == "w-fa" else "w", lw=0.6)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(labelsize=FS_S, length=2)
            if r == 0:
                ax.set_title(dlab, fontsize=FS_M)
            if r == len(SORTS) - 1:
                ax.set_xlabel("session progress", fontsize=FS_M)
            else:
                ax.set_xticklabels([])
            if group == "stacked":
                ax.axhline((meta.reward_group == "R+").sum() - 0.5, color="k" if dkey == "w-fa" else "w", lw=1.0)
            if c == 0:
                if group == "stacked":
                    ax.set_ylabel(f"sorted by {slab}\n(within cohort)\n{params}", fontsize=FS_S, labelpad=16)
                else:
                    loc, npc, nz = params
                    ax.set_ylabel(f"sorted by {slab}\nloc {loc}, PCs {npc}, norm {'on' if nz else 'off'}\n"
                                  f"score {score:.2f} (random {rnd:.2f})", fontsize=FS_S,
                                  labelpad=16 if group == "both" else 9)
                cats = meta["learning_category"].map(lambda v: v if v in CAT_COL else "NA").to_numpy()[order]
                strip(ax, cats, CAT_COL, [-0.06, 0, 0.035, 1])
                if group in ("both", "stacked"):
                    strip(ax, meta["reward_group"].to_numpy()[order], COL, [-0.11, 0, 0.035, 1])
            if r == 0 and c in (2, 3):
                cax = ax.inset_axes([1.03, 0.0, 0.05, 0.7])
                cb = fig.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=FS_S, length=2)
                cb.set_label("Δ P(lick)" if dkey == "w-fa" else "P(lick)", fontsize=FS_S)
    name = {"R+": "R+", "R-": "R−", "both": "R+ and R−",
            "stacked": "R+ (top) and R− (bottom), each sorted within cohort"}[group]
    fig.text(0.13, 0.985, f"{name} (n = {n} mice), whisker day 0{', A1-trimmed' if tag else ''}: rastermap sort per row "
             "source (best parameters by neighbour-similarity score), applied to every curve. Strips: "
             + ("cohort (green R+, purple R−), " if group in ("both", "stacked") else "")
             + "learning category (green good, yellow moderate, red bad, grey NA). Tick (white; black on the difference): learning trial "
             f"({LT_DEF}).", fontsize=FS_M, va="top", wrap=True)
    gname = {"R+": "Rplus", "R-": "Rminus", "both": "both", "stacked": "stacked"}[group]
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"030_rastermap_{gname}{tag}.{ext}", dpi=250)
    plt.close(fig)


def sweep_figure(D, meta, group, tag, all_orders, rnd):
    fig, axes = plt.subplots(3, 6, figsize=(11.7, 6.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.04, wspace=0.12, hspace=0.35)
    n = len(meta)
    for ax, key in zip(axes.flat, GRID):
        if key not in all_orders:
            ax.set_axis_off()
            continue
        o, s = all_orders[key]
        ax.imshow(D["whisker"][o], aspect="auto", cmap=CMAP_P, vmin=0, vmax=1, interpolation="nearest")
        if group == "both":
            strip(ax, meta["reward_group"].to_numpy()[o], COL, [-0.06, 0, 0.04, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"loc {key[0]}, PCs {key[1]}, norm {'on' if key[2] else 'off'}\nscore {s:.2f}", fontsize=FS_S)
    name = {"R+": "R+", "R-": "R−", "both": "R+ and R−"}[group]
    fig.text(0.04, 0.975, f"Parameter sweep, joint [whisker | FA] sort, {name} (n = {n}){', A1-trimmed' if tag else ''}: "
             f"whisker curve shown; score = mean correlation of adjacent rows (random order {rnd:.2f})", fontsize=FS_M,
             va="top")
    gname = {"R+": "Rplus", "R-": "Rminus", "both": "both"}[group]
    fig.savefig(HERE / f"030_param_sweep_{gname}{tag}.png", dpi=200)
    plt.close(fig)


def main():
    lt = pd.read_csv(ART / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")
    rows = []
    rng = np.random.default_rng(0)
    for tag in ("", "_trimA1"):
        stack = {}
        per = pickle.load(open(ART / f"024_avg_curves_per_mouse{tag}.pkl", "rb"))["sessions"]
        for group in ("R+", "R-", "both"):
            ps = [p for p in per if group == "both" or p["reward_group"] == group]
            D = {k: np.vstack([to_progress(p["curves"][k]) for p in ps]) for k in ("whisker", "fa", "auditory", "w-fa")}
            meta = pd.DataFrame([dict(session_id=p["session_id"], mouse_id=p["mouse_id"], reward_group=p["reward_group"],
                                      learning_category=p["learning_category"], n_whisker=p["n_whisker"]) for p in ps])
            meta["lt_frac"] = [lt.loc[s, LT_DEF] / max(nw - 1, 1) if s in lt.index and pd.notna(lt.loc[s, LT_DEF]) else np.nan
                               for s, nw in zip(meta.session_id, meta.n_whisker)]
            src = dict(whisker=D["whisker"], fa=D["fa"], auditory=D["auditory"],
                       joint=np.hstack([D["whisker"], D["fa"]]), **{"w-fa": D["w-fa"]})
            sorts, sweep = {}, None
            for skey, X in src.items():
                best, rnd, allo = best_sort(X, rng, rows, dict(version=tag.strip("_") or "untrimmed", group=group,
                                                               sort_source=skey, n_mice=len(ps)))
                sorts[skey] = ((best[0], best[1], best[2]), rnd)
                if skey == "joint":
                    sweep = (allo, rnd)
            heatmaps(D, meta, group, tag, sorts)
            sweep_figure(D, meta, group, tag, *sweep)
            stack[group] = (D, meta, sorts)
            if group == "R-":
                # user 2026-10-01: each cohort sorted separately, plotted together (R+ on top, R- below)
                (Dp, mp, sp), (Dm, mm, sm) = stack["R+"], stack["R-"]
                Ds = {k: np.vstack([Dp[k], Dm[k]]) for k in Dp}
                ms = pd.concat([mp, mm], ignore_index=True)
                ss = {}
                for k in sp:
                    (_, sa, oa), _ = sp[k]
                    (_, sb, ob), _ = sm[k]
                    ss[k] = ((f"R+ score {sa:.2f}, R− {sb:.2f}", np.nan, np.r_[oa, ob + len(mp)]), np.nan)
                heatmaps(Ds, ms, "stacked", tag, ss)
            print(tag or "untrimmed", group, {k: (v[0][0], round(v[0][1], 2), round(v[1], 2)) for k, v in sorts.items()},
                  flush=True)
            pd.DataFrame([dict(version=tag.strip("_") or "untrimmed", group=group, sort_source=k, rank=i,
                               session_id=meta.session_id.iloc[j], mouse_id=meta.mouse_id.iloc[j],
                               reward_group=meta.reward_group.iloc[j], learning_category=meta.learning_category.iloc[j])
                          for k, v in sorts.items() for i, j in enumerate(v[0][2])]).to_csv(
                ART / f"030_rastermap_order_{group.replace('+', 'plus').replace('-', 'minus')}{tag}.csv", index=False)
    pd.DataFrame(rows).to_csv(ART / "030_rastermap_param_scores.csv", index=False)


if __name__ == "__main__":
    main()
