"""029 -- Rastermap-sorted heatmaps of day-0 learning curves (user 2026-10-01: "create heatmaps of learning curves aligned at
first whisker trial. Within cohort, sort mice using rastermap on this data. Normalize session length. Get a sorting per
trial type, apply sorting on all trials for every combination. Do also a joint sorting on the concatenation of whisker
curve and false curve. Try to find structure").
Data: per-mouse HMM curves (sigma = 1; 024: whisker, false alarm (no stim), auditory, all on the whisker-trial axis that
starts at the first whisker trial), interpolated onto NPROG points of normalised session progress (first -> last whisker
trial = 0 -> 1). One session per mouse (whisker day 0). Versions: untrimmed | trimA1 (024, disengagement rule A1).
Sorting (within cohort): rastermap (Stringer et al. 2024) on the mice x progress matrix of one trial type -- whisker, FA,
auditory -- or on the concatenation [whisker | FA] (joint). n_clusters=None (sorts the mice directly; cohorts have ~40-60
mice), n_PCs = min(20, n_mice - 1), locality 0.5, normalize=False (curve LEVEL is kept, not only shape: how much a mouse
licks is part of the structure).
Figures, per version x cohort (029_rastermap_<cohort><tag>.png/pdf): rows = sorting source (whisker, FA, auditory, joint
whisker+FA), columns = displayed curve (whisker, FA, auditory, whisker - FA); every panel is mice (in that row's order) x
session progress. Side strips: learning category (good / moderate / bad / NA) and the learning trial (lenient cascade, as
a fraction of the session; white tick) when defined.
Structure (029_structure_<cohort><tag>.png/pdf + artifacts/029_rastermap_clusters.csv): Ward clustering of the joint
[whisker | FA] matrix, k chosen by silhouette (2..6); per cluster mean +- SEM of the whisker, FA and auditory curves;
cross-tab with learning category, fraction with a defined LT (each definition), median LT position, session length.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/029_rastermap_curve_heatmaps.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}          # GROUP_COLORS rplus / rminus
TYPE_COL = {"whisker": None, "fa": "0.35", "auditory": "#1f5fbf"}
CAT_COL = {"good": "#1a9850", "moderate": "#fee08b", "bad": "#d73027"}
NPROG = 101
FS_L, FS_M, FS_S = 8, 7, 6
DISPLAY = [("whisker", "Whisker"), ("fa", "False alarm"), ("auditory", "Auditory"), ("w-fa", "Whisker − FA")]
SORTS = [("whisker", "whisker"), ("fa", "false alarm"), ("auditory", "auditory"), ("joint", "joint whisker + FA")]
LT_DEF = "lenient cascade"
LT_DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
           "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
           "lenient cascade + clean gate"]


def to_progress(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return np.full(NPROG, np.nan)
    return np.interp(np.linspace(0, 1, NPROG), np.linspace(0, 1, len(v)), v)


def rastermap_sort(X):
    from rastermap import Rastermap
    X = np.nan_to_num(X, nan=np.nanmean(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Rastermap(n_clusters=None, n_PCs=min(20, X.shape[0] - 1), locality=0.5, time_lag_window=0,
                      normalize=False, verbose=False, random_state=0).fit(X)
    return np.asarray(m.isort)


def ward_clusters(X):
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score
    X = np.nan_to_num(X, nan=np.nanmean(X))
    Z = linkage(X, "ward")
    best = None
    for k in range(2, 7):
        lab = fcluster(Z, k, "maxclust")
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if best is None or s > best[1]:
            best = (k, s, lab)
    return best


def heatmap_figure(D, meta, rg, tag, sorts):
    fig, axes = plt.subplots(4, 4, figsize=(8.27, 9.4))
    fig.subplots_adjust(left=0.1, right=0.93, top=0.93, bottom=0.05, wspace=0.12, hspace=0.32)
    n = len(meta)
    for r, (skey, slab) in enumerate(SORTS):
        order = sorts[skey]
        for c, (dkey, dlab) in enumerate(DISPLAY):
            ax = axes[r, c]
            M = D[dkey][order]
            if dkey == "w-fa":
                im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                               extent=(0, 1, n - 0.5, -0.5))
            else:
                im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
                               extent=(0, 1, n - 0.5, -0.5))
            for i, s in enumerate(order):
                ltf = meta.iloc[s]["lt_frac"]
                if np.isfinite(ltf):
                    ax.plot([ltf, ltf], [i - 0.45, i + 0.45], color="w", lw=0.8)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(labelsize=FS_S, length=2)
            if r == 0:
                ax.set_title(dlab, fontsize=FS_M)
            if r == 3:
                ax.set_xlabel("session progress", fontsize=FS_M)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"sorted by {slab}", fontsize=FS_M, labelpad=12)
                cats = meta.iloc[order]["learning_category"].map(lambda x: x if x in CAT_COL else "NA").to_numpy()
                sx = ax.inset_axes([-0.07, 0, 0.04, 1])
                cl = ["good", "moderate", "bad", "NA"]
                sx.imshow(np.array([cl.index(x) for x in cats])[:, None], aspect="auto",
                          cmap=ListedColormap([CAT_COL["good"], CAT_COL["moderate"], CAT_COL["bad"], "0.85"]),
                          vmin=0, vmax=3, interpolation="nearest")
                sx.set_axis_off()
            if r == 0 and c in (2, 3):
                cax = ax.inset_axes([1.03, 0.0, 0.05, 0.6]) if c == 3 else ax.inset_axes([1.03, 0.0, 0.05, 0.6])
                cb = fig.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=FS_S, length=2)
                cb.set_label("Δ P(lick)" if dkey == "w-fa" else "P(lick)", fontsize=FS_S)
    fig.text(0.1, 0.975, f"{'R+' if rg == 'R+' else 'R−'} (n = {n} mice), whisker day 0{', A1-trimmed' if tag else ''}: "
             "rastermap sort per row source, applied to every curve type. Left strip: learning category (green good, "
             f"yellow moderate, red bad, grey NA). White tick: learning trial ({LT_DEF}).", fontsize=FS_M, va="top",
             color=COL[rg], wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"029_rastermap_{rg.replace('+', 'plus').replace('-', 'minus')}{tag}.{ext}", dpi=250)
    plt.close(fig)


def structure_figure(D, meta, rg, tag, rows_out):
    J = np.hstack([D["whisker"], D["fa"]])
    k, sil, lab = ward_clusters(J)
    meta = meta.assign(cluster=lab)
    x = np.linspace(0, 1, NPROG)
    fig, axes = plt.subplots(2, k, figsize=(1.9 * k + 0.6, 4.2), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.12, wspace=0.3, hspace=0.75)
    order = np.argsort([np.nanmean(D["whisker"][lab == c]) for c in range(1, k + 1)])[::-1] + 1
    for j, c in enumerate(order):
        ax = axes[0, j]
        sel = lab == c
        for key, colr in (("auditory", TYPE_COL["auditory"]), ("whisker", COL[rg]), ("fa", TYPE_COL["fa"])):
            m = np.nanmean(D[key][sel], 0)
            se = np.nanstd(D[key][sel], 0, ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else 0 * m
            ax.fill_between(x, m - se, m + se, color=colr, alpha=0.25, lw=0)
            ax.plot(x, m, color=colr, lw=1.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"cluster {j + 1} (n = {int(sel.sum())})", fontsize=FS_M)
        ax.set_xlabel("session progress", fontsize=FS_S)
        ax.tick_params(labelsize=FS_S)
        if j == 0:
            ax.set_ylabel("P(lick)", fontsize=FS_M)
        g = meta[sel]
        cats = g.learning_category.map(lambda v: v if v in CAT_COL else "NA").value_counts()
        ax2 = axes[1, j]
        ax2.bar(range(4), [cats.get(cn, 0) for cn in ("good", "moderate", "bad", "NA")],
                color=[CAT_COL["good"], CAT_COL["moderate"], CAT_COL["bad"], "0.8"])
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(["good", "mod.", "bad", "NA"], fontsize=FS_S)
        ax2.tick_params(labelsize=FS_S)
        ndef = {d: int(g[f"has_{d}"].sum()) for d in LT_DEFS}
        ax2.set_title(f"LT defined ({LT_DEF}): {ndef[LT_DEF]}/{len(g)}\nmedian LT {np.nanmedian(g.lt_frac):.2f} · "
                      f"{int(np.median(g.n_whisker))} whisker trials", fontsize=FS_S)
        if j == 0:
            ax2.set_ylabel("mice", fontsize=FS_M)
        for _, r in g.iterrows():
            rows_out.append(dict(version=tag.strip("_") or "untrimmed", cohort=rg, cluster=j + 1, k=k, silhouette=sil,
                                 session_id=r.session_id, mouse_id=r.mouse_id, learning_category=r.learning_category,
                                 n_whisker=r.n_whisker, lt_frac=r.lt_frac,
                                 **{f"has_{d}": r[f"has_{d}"] for d in LT_DEFS}))
    fig.text(0.09, 0.97, f"{'R+' if rg == 'R+' else 'R−'}{', A1-trimmed' if tag else ''}: Ward clusters of the joint "
             f"[whisker | FA] curves (k = {k} by silhouette = {sil:.2f})\ncurves: whisker, FA grey, auditory blue; bottom: "
             "learning category per cluster", fontsize=FS_M, va="top", color=COL[rg])
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"029_structure_{rg.replace('+', 'plus').replace('-', 'minus')}{tag}.{ext}", dpi=250)
    plt.close(fig)
    return k, sil


def main():
    lt = pd.read_csv(ART / "013_learning_trials_all_methods.csv").set_index("session_id")
    f028 = ART / "028_learning_trials_all_methods_all_mice.csv"
    if f028.exists():                               # behaviour-only mice get LTs from 028 when available
        lt = pd.read_csv(f028).set_index("session_id")
    rows, summ = [], []
    for tag in ("", "_trimA1"):
        per = pickle.load(open(ART / f"024_avg_curves_per_mouse{tag}.pkl", "rb"))["sessions"]
        for rg in ("R+", "R-"):
            ps = [p for p in per if p["reward_group"] == rg]
            D = {k: np.vstack([to_progress(p["curves"][k]) for p in ps]) for k in ("whisker", "fa", "auditory", "w-fa")}
            meta = pd.DataFrame([dict(session_id=p["session_id"], mouse_id=p["mouse_id"],
                                      learning_category=p["learning_category"], n_whisker=p["n_whisker"]) for p in ps])
            for d in LT_DEFS:
                meta[f"has_{d}"] = meta.session_id.map(lambda s: s in lt.index and pd.notna(lt.loc[s, d]))
            meta["lt_frac"] = [lt.loc[s, LT_DEF] / max(n - 1, 1) if s in lt.index and pd.notna(lt.loc[s, LT_DEF]) else np.nan
                               for s, n in zip(meta.session_id, meta.n_whisker)]
            sorts = {k: rastermap_sort(D[k]) for k in ("whisker", "fa", "auditory")}
            sorts["joint"] = rastermap_sort(np.hstack([D["whisker"], D["fa"]]))
            heatmap_figure(D, meta, rg, tag, sorts)
            k, sil = structure_figure(D, meta, rg, tag, rows)
            # agreement between sortings (Spearman of rank positions)
            from scipy.stats import spearmanr
            pos = {s: np.argsort(o) for s, o in sorts.items()}
            for a in pos:
                for b in pos:
                    if a < b:
                        summ.append(dict(version=tag.strip("_") or "untrimmed", cohort=rg, sort_a=a, sort_b=b,
                                         spearman=spearmanr(pos[a], pos[b])[0]))
            # rastermap position (joint) vs mouse features
            pj = pos["joint"]
            for feat in ("n_whisker", "lt_frac"):
                v = meta[feat].to_numpy(float)
                ok = np.isfinite(v)
                summ.append(dict(version=tag.strip("_") or "untrimmed", cohort=rg, sort_a="joint position", sort_b=feat,
                                 spearman=spearmanr(pj[ok], v[ok])[0] if ok.sum() > 3 else np.nan))
            print(tag or "untrimmed", rg, "clusters k =", k, "silhouette", round(sil, 2), flush=True)
    pd.DataFrame(rows).to_csv(ART / "029_rastermap_clusters.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(ART / "029_rastermap_sort_agreement.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
