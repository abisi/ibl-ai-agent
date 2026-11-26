"""Figure 4Q (former S) candidates (user 2026-11-26: "S: alternatives to show area effects?").

Data: as panel Q (shared_passive_cd.compute_learning_curve; 195): per session and area group, Spearman r between the projection
of active whisker hits on the area's hit - miss axis and the exact whisker learning curve P(lick | W), data - shifted; areas with
>= 3 sessions per cohort. Learners, whisker day 0.
Panels:
  a  current: rows = area groups, R+ / R- mean +- s.e.m. (filled, different from 0)
  b  R+ vs R- per area (scatter, area colours, identity line): which areas sit above / below the diagonal
  c  ranked cohort difference R+ - R- with bootstrap 95 % CI over sessions
  d  per cohort, distribution of the area means (strip + box) with Wilcoxon over areas vs 0 and R+ vs R- (paired over areas)
  e  area mean r vs number of sessions (does the effect depend on sampling?)
Output: Figures/_explore/Figure4_passive_cd_<date>/lc_areas/lc_area_alternatives.{png,pdf,svg}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_lc_area_alternatives.py
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
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

FS = 5
COH = ["R+", "R-"]
COL = {"R+": "#00B400", "R-": "#C800C8"}


def main():
    set_nature_rc()
    d = S.compute_learning_curve()
    d = d[d.group != "whole_brain"]
    col, order, _ = S.area_palette()
    n = d.groupby(["group", "cohort"]).session_id.nunique().unstack().fillna(0)
    areas = [a for a in order if a in n.index and n.loc[a].get("R+", 0) >= 3 and n.loc[a].get("R-", 0) >= 3]
    d = d[d.group.isin(areas)]
    m = d.groupby(["group", "cohort"]).value.agg(["mean", "sem", "count"]).unstack()
    rng = np.random.default_rng(0)
    ci = {}
    for a_, g in d.groupby("group"):
        p, q = g[g.cohort == "R+"].value.to_numpy(), g[g.cohort == "R-"].value.to_numpy()
        b = [rng.choice(p, len(p)).mean() - rng.choice(q, len(q)).mean() for _ in range(2000)]
        ci[a_] = (p.mean() - q.mean(), *np.percentile(b, [2.5, 97.5]))
    fig = plt.figure(figsize=(183 * MM, 70 * MM))
    gs = fig.add_gridspec(1, 5, left=0.08, right=0.99, bottom=0.2, top=0.86, wspace=0.75)

    # a current
    a = fig.add_subplot(gs[0, 0])
    y = np.arange(len(areas))
    for c, off in (("R+", -0.15), ("R-", 0.15)):
        for k, a_ in enumerate(areas):
            v = d[(d.group == a_) & (d.cohort == c)].value
            mm, se = v.mean(), v.sem()
            p0 = max(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue) if len(v) >= 5 else 1
            a.errorbar(mm, k + off, xerr=se, fmt="o", ms=2.2, color=COL[c], mfc=COL[c] if p0 < 0.05 else "white", mew=0.5,
                       lw=0.5, capsize=0)
    a.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_yticks(y, [S.area_label(x) for x in areas], fontsize=FS - 1)
    for tl, a_ in zip(a.get_yticklabels(), areas):
        tl.set_color(col.get(a_, "0.3"))
    a.set_ylim(len(areas) - 0.5, -0.5)
    a.set_xlabel("W hits vs P(lick | W), r\n(data − shifted)", fontsize=FS)
    a.set_title("a  current (rows)", fontsize=FS + 0.5, loc="left")

    # b R+ vs R- scatter
    b = fig.add_subplot(gs[0, 1])
    x_, y_ = m[("mean", "R+")], m[("mean", "R-")]
    for a_ in areas:
        b.errorbar(x_[a_], y_[a_], xerr=m.loc[a_, ("sem", "R+")], yerr=m.loc[a_, ("sem", "R-")], fmt="o", ms=3,
                   color=col.get(a_, "0.4"), lw=0.4, capsize=0)
    lim = [min(x_.min(), y_.min()) - 0.05, max(x_.max(), y_.max()) + 0.05]
    b.plot(lim, lim, color="0.6", lw=0.5, ls=(0, (2, 2)))
    b.axhline(0, color="0.8", lw=0.4)
    b.axvline(0, color="0.8", lw=0.4)
    w = stats.wilcoxon(x_[areas], y_[areas]).pvalue
    b.text(0.03, 0.97, f"{int((x_[areas] > y_[areas]).sum())}/{len(areas)} areas R+ > R−\nWilcoxon over areas {S.p_text(w)}",
           transform=b.transAxes, ha="left", va="top", fontsize=FS - 1)
    b.set_xlabel("R+ r (data − shifted)", fontsize=FS)
    b.set_ylabel("R− r (data − shifted)", fontsize=FS)
    b.set_title("b  R+ vs R− per area", fontsize=FS + 0.5, loc="left")

    # c ranked difference with CI
    c_ = fig.add_subplot(gs[0, 2])
    rk = sorted(areas, key=lambda a_: ci[a_][0])
    for k, a_ in enumerate(rk):
        e, lo, hi = ci[a_]
        sig = lo > 0 or hi < 0
        c_.plot([lo, hi], [k, k], color=col.get(a_, "0.4"), lw=0.7)
        c_.plot(e, k, "o", ms=2.6, color=col.get(a_, "0.4"), mfc=col.get(a_, "0.4") if sig else "white", mew=0.6)
    c_.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    c_.set_yticks(range(len(rk)), [S.area_label(x) for x in rk], fontsize=FS - 1)
    for tl, a_ in zip(c_.get_yticklabels(), rk):
        tl.set_color(col.get(a_, "0.3"))
    c_.set_xlabel("R+ − R− (bootstrap 95 % CI)", fontsize=FS)
    c_.set_title("c  ranked difference", fontsize=FS + 0.5, loc="left")

    # d distribution of area means per cohort
    d_ = fig.add_subplot(gs[0, 3])
    for q, c in enumerate(COH):
        v = m[("mean", c)][areas].to_numpy()
        d_.boxplot(v, positions=[q], widths=0.5, showfliers=False, medianprops=dict(color=COL[c]),
                   boxprops=dict(color=COL[c], lw=0.6), whiskerprops=dict(color=COL[c], lw=0.6), capprops=dict(lw=0))
        d_.scatter(q + rng.uniform(-0.12, 0.12, len(v)), v, s=5, c=[col.get(a_, "0.4") for a_ in areas], lw=0, zorder=3)
        p0 = stats.wilcoxon(v).pvalue
        d_.text(q, 1.0, f"vs 0: {S.p_text(p0)}", transform=d_.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=FS - 1.2, color=COL[c])
    for a_ in areas:
        d_.plot([0, 1], [x_[a_], y_[a_]], color="0.85", lw=0.3, zorder=1)
    d_.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    d_.set_xticks([0, 1], ["R+", "R−"])
    d_.set_ylabel("Area mean r (data − shifted)", fontsize=FS)
    d_.set_title(f"d  area means (paired {S.p_text(w)})", fontsize=FS + 0.5, loc="left", pad=8)

    # e effect vs number of sessions
    e_ = fig.add_subplot(gs[0, 4])
    for c in COH:
        e_.scatter(m[("count", c)][areas], m[("mean", c)][areas], s=6, color=COL[c], lw=0)
        r, p = stats.spearmanr(m[("count", c)][areas], m[("mean", c)][areas])
        e_.text(0.03, 0.97 - 0.08 * COH.index(c), f"{c.replace('-', '−')}: ρ = {r:.2f}, {S.p_text(p)}", transform=e_.transAxes,
                ha="left", va="top", fontsize=FS - 1, color=COL[c])
    e_.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    e_.set_xlabel("Sessions with the area", fontsize=FS)
    e_.set_ylabel("Area mean r (data − shifted)", fontsize=FS)
    e_.set_title("e  vs sampling", fontsize=FS + 0.5, loc="left")
    for ax_ in (a, b, c_, d_, e_):
        ax_.tick_params(labelsize=FS, length=1.5)
    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}" / "lc_areas"
    out.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, "lc_area_alternatives")
    print("areas", len(areas), "R+ > R-:", int((x_[areas] > y_[areas]).sum()), "wilcoxon", w)
    print("saved", out)


if __name__ == "__main__":
    main()
