"""Figure 4G candidates (user 2026-11-25: "what would be other good ways to show the distribution of the effects across areas
... show all these alternatives on a separate figure in the same folder").

Data: compute_ag_areas (179 units, learners, whisker day 0; area groups with >= 10 neurons per session and >= 3 sessions per
cohort): per session x area group, evoked change post - pre (5-35 ms, Hz), whisker W and auditory A; WA = W - A.
Effect per area: delta = mean_R+(WA) - mean_R-(WA) (> 0: R- loses its whisker response relative to auditory more than R+).
Panels:
  a  current G: WA per cohort (mean +- s.e.m.)
  b  forest plot: delta with bootstrap 95 % CI (sessions resampled within cohort, 2000x), areas sorted by delta
  c  R+ vs R- scatter of the area means, identity line
  d  delta per area vs a cohort-label shuffle null (session labels permuted, the same permutation for all areas, 2000x):
     per-area 95 % null band; inset, mean delta over areas vs its null
  e  map: area-group centroids (mean CCF of their units, ROC table) in a dorsal (AP x ML) view, colour = delta,
     size = neurons
  f  delta vs the area's pre-task whisker response (both cohorts pooled), Spearman
Output: Figures/_explore/Figure4_passive_cd_<date>/area_effect_alternatives.{png,pdf,svg}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_area_alternatives.py
"""
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT, axel_bisi_root  # noqa: E402
from paperstyle import COLORS, MM, save_figure, set_nature_rc  # noqa: E402

FS = 5
rng = np.random.default_rng(0)


def area_table():
    u = S._ag_units()
    u = u.assign(b_pre=u[["base_pre_W", "base_pre_A"]].mean(1), b_post=u[["base_post_W", "base_post_A"]].mean(1))
    d = S.compute_ag_areas()
    d["WA"] = d.W - d.A
    pre = u.groupby("area_group").ev_pre_W.mean().rename("pre_W")
    n = u.groupby("area_group").size().rename("n_units")
    # CCF centroid per area group (ROC table has ccf_ap / ccf_ml / ccf_dv per unit)
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "ccf_ap", "ccf_ml", "ccf_dv"])
            rows.append(r.drop_duplicates(["session_id", "electrode_group", "cluster_id"]))
    c = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    uc = u.merge(c, on=["session_id", "electrode_group", "cid"], how="left")
    cen = uc.groupby("area_group")[["ccf_ap", "ccf_ml", "ccf_dv"]].median()
    return d, pd.concat([pre, n, cen], axis=1)


def delta(d):
    m = d.groupby(["area_group", "cohort"]).WA.mean().unstack()
    return (m["R+"] - m["R-"]).rename("delta")


def boot_ci(d, n=2000):
    out = {}
    for a_, g in d.groupby("area_group"):
        p, q = g[g.cohort == "R+"].WA.to_numpy(), g[g.cohort == "R-"].WA.to_numpy()
        b = [rng.choice(p, len(p)).mean() - rng.choice(q, len(q)).mean() for _ in range(n)]
        out[a_] = np.percentile(b, [2.5, 97.5])
    return pd.DataFrame(out, index=["lo", "hi"]).T


def shuffle_null(d, n=2000):
    ses = d[["session_id", "cohort"]].drop_duplicates()
    nulls = []
    for _ in range(n):
        perm = dict(zip(ses.session_id, rng.permutation(ses.cohort.to_numpy())))
        nulls.append(delta(d.assign(cohort=d.session_id.map(perm))))
    return pd.concat(nulls, axis=1)


def main():
    set_nature_rc()
    d, info = area_table()
    col, order, _ = S.area_palette()
    dl = delta(d)
    ci = boot_ci(d)
    null = shuffle_null(d)
    areas = [a for a in order if a in dl.index]
    srt = dl.loc[areas].sort_values().index.tolist()
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    W, H = 183, 120

    def ax_at(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def letter(x, y, s, title):
        fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top")
        fig.text((x + 4) / W, 1 - y / H, title, fontsize=6, va="top")

    # a: current G
    letter(0, 2, "a", "Current G: per cohort")
    a = ax_at(26, 10, 22, 40)
    m = d.groupby(["cohort", "area_group"]).WA.agg(["mean", "sem"]).reset_index()
    for k, ar in enumerate(areas):
        for c, off in (("R+", -0.17), ("R-", 0.17)):
            r = m[(m.cohort == c) & (m.area_group == ar)]
            a.errorbar(r["mean"], k + off, xerr=r["sem"], fmt="o", ms=1.8, color=COLORS[c], mfc="white", mew=0.5, lw=0.4)
    a.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_yticks(range(len(areas)))
    a.set_yticklabels(areas, fontsize=FS - 1)
    for tl, ar in zip(a.get_yticklabels(), areas):
        tl.set_color(col.get(ar, "0.2"))
    a.set_ylim(len(areas) - 0.5, -0.5)
    a.set_xlabel("Whisker − auditory change (Hz)", fontsize=FS)
    a.tick_params(labelsize=FS, length=1.5)

    # b: forest plot
    letter(60, 2, "b", "Forest: R+ − R−, bootstrap 95% CI, sorted")
    b = ax_at(86, 10, 22, 40)
    for k, ar in enumerate(srt):
        sig = (ci.loc[ar, "lo"] > 0) or (ci.loc[ar, "hi"] < 0)
        b.plot([ci.loc[ar, "lo"], ci.loc[ar, "hi"]], [k, k], color=col.get(ar, "0.3"), lw=0.8)
        b.plot(dl[ar], k, "o", ms=2.4, color=col.get(ar, "0.3"), mfc=col.get(ar, "0.3") if sig else "white", mew=0.6)
    b.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    b.set_yticks(range(len(srt)))
    b.set_yticklabels(srt, fontsize=FS - 1)
    for tl, ar in zip(b.get_yticklabels(), srt):
        tl.set_color(col.get(ar, "0.2"))
    b.set_ylim(len(srt) - 0.5, -0.5)
    b.set_xlabel("Δ (R+ − R−) of whisker − auditory\nchange (Hz)", fontsize=FS)
    b.tick_params(labelsize=FS, length=1.5)

    # c: R+ vs R- scatter
    letter(120, 2, "c", "R+ vs R− per area")
    c_ = ax_at(130, 10, 30, 30)
    mm = d.groupby(["area_group", "cohort"]).WA.mean().unstack()
    for ar in areas:
        c_.plot(mm.loc[ar, "R+"], mm.loc[ar, "R-"], "o", ms=3, color=col.get(ar, "0.3"))
        c_.text(mm.loc[ar, "R+"], mm.loc[ar, "R-"], " " + ar.split(" ")[0][:8], fontsize=FS - 1.5, color=col.get(ar, "0.3"),
                va="center")
    lim = [min(mm.min().min(), 0) - 0.5, max(mm.max().max(), 0) + 0.5]
    c_.plot(lim, lim, color="0.6", lw=0.4, ls=(0, (2, 2)))
    c_.axhline(0, color="0.85", lw=0.4)
    c_.axvline(0, color="0.85", lw=0.4)
    c_.set_xlim(lim)
    c_.set_ylim(lim)
    c_.set_xlabel("R+ whisker − auditory change (Hz)", fontsize=FS)
    c_.set_ylabel("R− whisker − auditory change (Hz)", fontsize=FS)
    c_.tick_params(labelsize=FS, length=1.5)
    r_, p_ = stats.wilcoxon(mm["R+"] - mm["R-"])
    c_.set_title(f"below identity: {(mm['R-'] < mm['R+']).sum()}/{len(mm)} areas; Wilcoxon over areas p={p_:.3f}",
                 fontsize=FS - 1)

    # d: delta vs shuffle null
    letter(0, 62, "d", "Δ per area vs cohort-shuffle null")
    e = ax_at(26, 70, 22, 40)
    for k, ar in enumerate(srt):
        lo, hi = np.nanpercentile(null.loc[ar], [2.5, 97.5])   # a shuffle can leave an area without one cohort
        e.plot([lo, hi], [k, k], color="0.8", lw=2.2, solid_capstyle="butt")
        out = (dl[ar] < lo) or (dl[ar] > hi)
        e.plot(dl[ar], k, "o", ms=2.4, color=col.get(ar, "0.3"), mfc=col.get(ar, "0.3") if out else "white", mew=0.6)
    e.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    e.set_yticks(range(len(srt)))
    e.set_yticklabels(srt, fontsize=FS - 1)
    for tl, ar in zip(e.get_yticklabels(), srt):
        tl.set_color(col.get(ar, "0.2"))
    e.set_ylim(len(srt) - 0.5, -0.5)
    e.set_xlabel("Δ (Hz); grey, null 95%", fontsize=FS)
    e.tick_params(labelsize=FS, length=1.5)
    ins = ax_at(52, 72, 14, 12)
    mn = null.loc[areas].mean(0, skipna=True)
    ins.hist(mn, bins=40, color="0.75")
    ins.axvline(dl.loc[areas].mean(), color="#B5524A", lw=0.9)
    pm = (np.sum(np.abs(mn) >= abs(dl.loc[areas].mean())) + 1) / (len(mn) + 1)
    ins.set_title(f"mean Δ over areas\np={pm:.3f}", fontsize=FS - 1)
    ins.set_yticks([])
    ins.tick_params(labelsize=FS - 1, length=1)

    # e: map (dorsal view of centroids)
    letter(72, 62, "e", "Map: area centroids, dorsal view")
    f = ax_at(80, 70, 40, 40)
    v = np.nanmax(np.abs(dl.loc[areas]))
    sc = None
    for ar in areas:
        x, y = info.loc[ar, "ccf_ml"], info.loc[ar, "ccf_ap"]
        if not np.isfinite(x):
            continue
        sc = f.scatter(x, y, s=8 + 40 * np.sqrt(info.loc[ar, "n_units"] / info.n_units.max()), c=[dl[ar]], cmap="RdBu_r",
                       vmin=-v, vmax=v, edgecolor=col.get(ar, "0.3"), lw=0.8)
        f.text(x, y, "  " + ar.split(" ")[0][:10], fontsize=FS - 1.5, va="center")
    f.set_xlabel("ML (CCF, µm)", fontsize=FS)
    f.set_ylabel("AP (CCF, µm; anterior up)", fontsize=FS)
    f.invert_yaxis()
    f.tick_params(labelsize=FS - 0.5, length=1.5)
    if sc is not None:
        cb = fig.colorbar(sc, ax=f, fraction=0.05, pad=0.02)
        cb.set_label("Δ (Hz)", fontsize=FS)
        cb.ax.tick_params(labelsize=FS - 1)

    # f: delta vs pre-task whisker response
    letter(130, 62, "f", "Δ vs pre-task whisker response")
    g = ax_at(140, 70, 30, 30)
    for ar in areas:
        g.plot(info.loc[ar, "pre_W"], dl[ar], "o", ms=3, color=col.get(ar, "0.3"))
    rs, ps = stats.spearmanr(info.loc[areas, "pre_W"], dl.loc[areas])
    g.axhline(0, color="0.7", lw=0.4)
    g.set_xlabel("Pre-task whisker response (Hz)", fontsize=FS)
    g.set_ylabel("Δ (R+ − R−) (Hz)", fontsize=FS)
    g.set_title(f"Spearman r={rs:.2f}, p={ps:.3f} ({len(areas)} areas)", fontsize=FS - 1)
    g.tick_params(labelsize=FS, length=1.5)

    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}"
    save_figure(fig, out, "area_effect_alternatives")
    print(pd.concat([dl, ci, info[["pre_W", "n_units"]]], axis=1).loc[srt].round(2).to_string())
    print("mean delta", round(dl.loc[areas].mean(), 3), "perm p", round(pm, 4), "| spearman", round(rs, 2), round(ps, 3))
    print("saved", out / "area_effect_alternatives.png")


if __name__ == "__main__":
    main()
