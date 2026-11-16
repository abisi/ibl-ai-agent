"""009 -- Figures of 007 (does learning orthogonalise the sensory and the hit / miss coding directions?). Day-0 learners (+ expert
sessions, all mice), session = unit, mean +- s.e.m. over sessions; style as 003. Cosines are cross-validated (007), signed:
> 0 = hits respond more along the sensory direction; orthogonalisation = a drop towards 0.
Figure 009_orthogonalization_{window}: whole brain
  A cos(sensory CD passive pre, hit / miss CD): early vs late half, before vs from the learning trial (+ linear-shift nulls, grey)
  B the same with the active sensory CD of the same trials
  C cos with the passive pre vs passive post sensory CD (whole active block)
  D day 0 vs expert (whole active block): passive pre and active sensory CD
  E hit / miss separation per unit split along the passive pre sensory CD (squared projection) and orthogonal to it, early vs late
  F stability of the directions: cos between the two passive sensory CDs, the two halves' sensory and hit / miss CDs
  G passive change projected on the hit / miss CD (units of the hit / miss distance): total, gain part (along the sensory CD),
    rotation part (orthogonal), and the relative gain change along the sensory CD
Stars: A, B, E paired Wilcoxon late - early (pre - post LT) above the pair; C paired post - pre; D Mann-Whitney day 0 vs expert
(Welch in the stats table); F, G Wilcoxon vs 0 (F: vs 1 would be 'same direction'; reported as is).
Figure 009_orthogonalization_areas_{window}: per area group, day 0: late - early and post - pre LT change of cos(pre CD, h),
  cos(pre CD, h) whole block day 0 and expert; * BH q < 0.05.
Output: figures/009_orthogonalization{_areas}_{window}.{png,pdf,svg}; tables/009_orthogonalization_{window}_stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/009_orthogonalization_figure.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
f3 = importlib.import_module("003_figures")
COH, CL, WB, TAB, FIG = f3.COH, f3.CL, f3.WB, f3.TAB, f3.FIG
OFF = {"R+": -0.12, "R-": 0.12}


def load():
    learners, _ = f3.m61.learner_mice()
    d0 = pd.read_parquet(TAB / "007_orthogonalization.parquet")
    d0 = d0[d0.skipped_reason.isna() & d0.mouse_id.isin(learners)].assign(stage="day 0")
    p = TAB / "007_orthogonalization_expert.parquet"
    ex = pd.read_parquet(p) if p.exists() else d0.iloc[:0]
    ex = ex[ex.skipped_reason.isna()].assign(stage="expert")
    D = pd.concat([d0, ex], ignore_index=True)
    for k in ("early", "late", "all", "preLT", "postLT"):
        if f"h2_{k}" in D:
            D[f"hpar2_{k}"] = D[f"hpar_pre_{k}"] ** 2
            D[f"horth2_{k}"] = D[f"h2_{k}"] - D[f"hpar2_{k}"]
    return D


def paired(ax, t, pairs, panel, what, null=False):
    """pairs: list of (x0, col0, x1, col1, label); per cohort mean +- s.e.m. at x + OFF, line, star on col1 - col0."""
    for x0, c0, x1, c1, lab in pairs:
        for c in COH:
            d = t[t.reward_group == c]
            if c0 not in d or c1 not in d:
                continue
            xs = (x0 + OFF[c], x1 + OFF[c])
            m = [d[c0].mean(), d[c1].mean()]
            ax.plot(xs, m, color=COH[c], lw=0.6, zorder=2)
            for x, col in zip(xs, (c0, c1)):
                f3.msem(ax, x, d[col], COH[c], ms=3, star=False, panel=panel, what=f"{what}: {col}")
                if null and f"{col}_null" in d:
                    nv = d[f"{col}_null"]
                    ax.errorbar(x + 0.06, nv.mean(), nv.sem(), fmt="o", color="0.55", mfc="white", ms=2.2, mew=0.5,
                                elinewidth=0.5, capsize=0, zorder=1)
                    f3.log(panel, f"{what}: {col} − shift null", c, d[col] - nv, f3.wil(d[col] - nv),
                           test="Wilcoxon real vs linear-shift null")
            diff = d[c1] - d[c0]
            p = f3.wil(diff)
            f3.log(panel, f"{what}: {lab} ({c1} − {c0})", c, diff, p, test="paired Wilcoxon")
            if f3.stars(p):
                top = max(d[c0].mean() + d[c0].sem(), d[c1].mean() + d[c1].sem())
                ax.annotate(f3.stars(p), (np.mean(xs), top), xytext=(0, 2 + 5 * (c == "R-")), textcoords="offset points",
                            ha="center", va="bottom", color=COH[c], fontsize=6)
        dd = {c: (t[t.reward_group == c][c1] - t[t.reward_group == c][c0]) if c1 in t else pd.Series(dtype=float) for c in COH}
        pm, pw = f3.two(dd["R+"], dd["R-"])
        f3.STATS.append(dict(panel=panel, measure=f"{what}: {lab} change, R+ vs R−", cohort="R+ vs R−", p=pm, p_welch=pw,
                             test="Mann-Whitney (p), Welch (p_welch)"))
    f3.zero(ax)


def cats(ax, t, items, panel, ref=0.0):
    for j, (lab, col) in enumerate(items):
        for c in COH:
            f3.msem(ax, j + OFF[c], t[t.reward_group == c][col] - ref, COH[c], ms=3, panel=panel, what=lab.replace("\n", " "))
    ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=5, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(items) - 0.5)
    f3.zero(ax)


def main_figure(D, window):
    W = D[(D.area == WB) & (D.window == window)]
    d0, ex = W[W.stage == "day 0"], W[W.stage == "expert"]
    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.985, top=0.9, bottom=0.13, wspace=0.6, hspace=0.95)
    ax = {k: fig.add_subplot(gs[i // 4, i % 4]) for i, k in enumerate("ABCDEFG")}
    halves = [(0, "{}_early", 1, "{}_late", "late − early"), (2.4, "{}_preLT", 3.4, "{}_postLT", "post − pre LT")]
    xt = ([0, 1, 2.4, 3.4], ["Early\nhalf", "Late\nhalf", "Before\nLT", "From\nLT"])
    for k, col, ttl in (("A", "cos_pre_h", "Passive sensory CD\nvs hit/miss CD"), ("B", "cos_s_h", "Active sensory CD\nvs hit/miss CD")):
        paired(ax[k], d0, [(a, c0.format(col), b, c1.format(col), lab) for a, c0, b, c1, lab in halves], k, ttl.replace("\n", " "),
               null=True)
        ax[k].set_xticks(*xt, fontsize=5)
        ax[k].set_ylabel("cos(sensory CD, hit/miss CD)\n(cross-validated)")
        ax[k].set_title(ttl)
    paired(ax["C"], d0, [(0, "cos_pre_h_all", 1, "cos_post_h_all", "post − pre")], "C", "passive pre vs post sensory CD")
    ax["C"].set_xticks([0, 1], ["Passive\npre CD", "Passive\npost CD"], fontsize=5)
    ax["C"].set_xlim(-0.5, 1.5); ax["C"].set_ylabel("cos(sensory CD, hit/miss CD)"); ax["C"].set_title("Sensory CD after\nthe task")

    a = ax["D"]
    for j, (col, lab) in enumerate((("cos_pre_h_all", "Passive pre CD"), ("cos_s_h_all", "Active CD"))):
        for c in COH:
            for s, (t, dx) in enumerate(((d0, -0.15), (ex, 0.15))):
                v = t[t.reward_group == c][col]
                x = j + dx + OFF[c] * 0.5
                f3.msem(a, x, v, COH[c], filled=(s == 0), ms=3, star=False, panel="D", what=f"{lab} {'day 0' if s == 0 else 'expert'}")
            pm, pw = f3.two(d0[d0.reward_group == c][col], ex[ex.reward_group == c][col])
            f3.STATS.append(dict(panel="D", measure=f"{lab}: day 0 vs expert", cohort=c, p=pm, p_welch=pw,
                                 test="Mann-Whitney (p), Welch (p_welch)"))
            if f3.stars(pm):
                a.annotate(f3.stars(pm), (j + OFF[c] * 0.5, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top",
                           color=COH[c], fontsize=6)
    a.set_xticks([0, 1], ["Passive\npre CD", "Active\nCD"], fontsize=5); a.set_xlim(-0.5, 1.5); f3.zero(a)
    a.set_ylabel("cos(sensory CD, hit/miss CD)"); a.set_title("Day 0 (filled) vs\nexpert (open)")

    paired(ax["E"], d0, [(0, "hpar2_early", 1, "hpar2_late", "along, late − early"),
                         (2.4, "horth2_early", 3.4, "horth2_late", "orthogonal, late − early")], "E", "hit/miss separation")
    ax["E"].set_xticks([0, 1, 2.4, 3.4], ["Early", "Late", "Early", "Late"], fontsize=5)
    ax["E"].text(0.5, -0.2, "along sensory CD", transform=ax["E"].get_xaxis_transform(), ha="center", va="top", fontsize=5)
    ax["E"].text(2.9, -0.2, "orthogonal", transform=ax["E"].get_xaxis_transform(), ha="center", va="top", fontsize=5)
    ax["E"].set_ylabel("Hit − miss distance² per unit"); ax["E"].set_title("Where the hit/miss\nseparation lies")

    cats(ax["F"], d0, [("Passive pre\nvs post", "cos_pre_post"), ("Sensory\nearly vs late", "cos_s_early_late"),
                       ("Hit/miss\nearly vs late", "cos_h_early_late"), ("Hit/miss\nbefore vs from LT", "cos_h_preLT_postLT")], "F")
    ax["F"].axhline(1, color="0.6", lw=0.4, ls="--"); ax["F"].set_ylabel("cos between the two CDs"); ax["F"].set_title("Stability of\nthe directions")
    cats(ax["G"], d0, [("Total", "dproj_h"), ("Gain part", "dproj_h_gain"), ("Rotation\npart", "dproj_h_rot"),
                       ("Relative gain\nchange", "gain_rel")], "G")
    ax["G"].set_ylabel("Passive change post − pre\n(hit/miss distance units)"); ax["G"].set_title("Passive change:\ngain vs rotation")

    for k, a in ax.items():
        bb = a.get_position()
        fig.text(bb.x0 - 0.06, bb.y1 + 0.035, k, fontsize=9, fontweight="bold", va="bottom")
    h = [plt.Line2D([], [], marker="o", ls="", color=COH[c], ms=3, label=CL[c]) for c in COH]
    h.append(plt.Line2D([], [], marker="o", ls="", color="0.55", mfc="white", ms=3, label="linear-shift null"))
    fig.legend(handles=h, loc="upper right", ncol=3, frameon=False, fontsize=5.5)
    fig.text(0.01, 0.985, f"Response window: {window} ({'5–35' if window == 'early' else '35–75'} ms)", fontsize=6, va="top")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"009_orthogonalization_{window}.{ext}", dpi=300)
    plt.close(fig)


def area_figure(D, window):
    W = D[(D.area != WB) & (D.window == window)].copy()
    W["d_half"] = W.cos_pre_h_late - W.cos_pre_h_early
    W["d_lt"] = W.cos_pre_h_postLT - W.cos_pre_h_preLT
    panels = [("d_half", "day 0", "cos(pre CD, h)\nlate − early half"), ("d_lt", "day 0", "cos(pre CD, h)\nfrom − before LT"),
              ("cos_pre_h_all", "day 0", "cos(pre CD, h)\nday 0"), ("cos_pre_h_all", "expert", "cos(pre CD, h)\nexpert")]
    cnt = W[W.stage == "day 0"].groupby(["area", "reward_group"]).session_id.nunique().unstack().fillna(0)
    keep = cnt[(cnt >= f3.MIN_SESS).any(axis=1)].index.tolist()
    pal, areas = f3.area_palette(False, keep)
    rows = []
    for m, st, _ in panels:
        for c in COH:
            for a in areas:
                v = W[(W.area == a) & (W.reward_group == c) & (W.stage == st)][m].dropna()
                if len(v) >= f3.MIN_SESS:
                    rows.append(dict(measure=m, stage=st, cohort=c, area=a, n=len(v), mean=v.mean(), sem=v.sem(), p=f3.wil(v)))
    R = pd.DataFrame(rows)
    R["q_bh"] = np.nan
    for _, g in R[R.p.notna()].groupby(["measure", "stage", "cohort"]):
        R.loc[g.index, "q_bh"] = multipletests(g.p, method="fdr_bh")[1]
    R.to_csv(TAB / f"009_orthogonalization_areas_{window}_stats.csv", index=False)
    h = 0.14 * len(areas) + 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2, h), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.99, top=1 - 0.55 / h, bottom=0.35 / h, wspace=0.25)
    y = np.arange(len(areas))
    for ax, (m, st, lab) in zip(axes, panels):
        for k, c in enumerate(COH):
            t = R[(R.measure == m) & (R.stage == st) & (R.cohort == c)].set_index("area").reindex(areas)
            yy = y + (k - 0.5) * 0.32
            ax.errorbar(t["mean"], yy, xerr=t["sem"], fmt="o", ms=2.2, color=COH[c], elinewidth=0.5, mew=0)
            for y0, x0, e0, q in zip(yy, t["mean"], t["sem"], t["q_bh"]):
                if np.isfinite(q) and q < 0.05:
                    ax.text(x0 + e0, y0, " *", color=COH[c], fontsize=6, va="center", ha="left")
        ax.axvline(0, color="0.6", lw=0.4, ls=":")
        for i in range(0, len(areas), 2):
            ax.axhspan(i - 0.5, i + 0.5, color="0.95", lw=0, edgecolor="none", zorder=0)
        ax.set_title(lab, fontsize=5.5); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y, [a.replace("Somatosensory-", "Ss-") for a in areas], fontsize=5)
    for t, a in zip(axes[0].get_yticklabels(), areas):
        t.set_color(pal[a])
    axes[0].set_ylim(len(areas) - 0.5, -0.5)
    fig.text(0.99, 0.01, f"{window} window; mean ± s.e.m. over sessions; * BH q < 0.05 (Wilcoxon vs 0)", ha="right", fontsize=4.5,
             color="0.3")
    for i, ax in enumerate(axes):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.01, bb.y1 + 0.32 / h, "ABCD"[i], fontsize=9, fontweight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"009_orthogonalization_areas_{window}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    D = load()
    pd.set_option("display.width", 250, "display.max_rows", 300)
    for w in ("early", "mid"):
        f3.STATS.clear()
        main_figure(D, w)
        area_figure(D, w)
        S = pd.DataFrame(f3.STATS)
        S.to_csv(TAB / f"009_orthogonalization_{w}_stats.csv", index=False)
        print(w); print(S.drop(columns=[c for c in ("test",) if c in S]).round(4).to_string())
