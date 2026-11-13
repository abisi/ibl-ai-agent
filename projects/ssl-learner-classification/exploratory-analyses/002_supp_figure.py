"""002 -- Supplementary figure: who is a learner (user 2026-11-13: "a supp figure that says what learners are, then compare
performance between learners and non-learners").

Reads the tables and curve caches of 001 (combined_results_ks4/ssl-learner-classification). Nature format (183 mm wide,
Arial 5-7 pt, paperstyle). Panels:
  A  criteria x mice: the nine learner criteria per mouse (sorted by consensus within cohort), consensus, agnostic label,
     sheet category, earlier DTW clusters, user-fixed non-learners
  B  agreement of each criterion and of the sheet with the agnostic label (Cohen's kappa)
  C  validation on day +1: AUC of each labelling for whisker P(lick) and d' on day +1
  D  what a learner is: fraction of whisker trials with the whisker curve credibly above chance, per mouse, with the
     best single threshold
  E  sheet category against the agnostic label
  F  mean exact learning curves (whisker and chance) of learners and non-learners, per cohort (session progress)
  G  performance of learners vs non-learners (day 0 and day +1)
Outputs: Figures/SuppFigure_learners_<date>/ (manuscript snapshot) and combined_results_ks4/ssl-learner-classification/figures/.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/002_supp_figure.py
"""

import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
sys.path.insert(0, str(REPO / "scripts"))
import paperstyle as ps  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root()
RES = ROOT / "combined_results_ks4" / "ssl-learner-classification"
SNAP = ROOT / "manuscripts" / "Bisi_Hamon" / "Figures" / f"SuppFigure_learners_{date.today()}"
COHORTS = ["R+", "R-"]
COH_LABEL = {"R+": "R+", "R-": "R−"}
C = ps.COLORS
CRITERIA = ["exact_rule", "high_state", "half2", "late30", "sustained", "L6", "cascade", "gmm", "dtw"]
CRIT_LABEL = {"exact_rule": "Exact-curve rule", "high_state": "High-performance state", "half2": "2nd half: W > FA",
              "late30": "Last 30 W trials: W > FA", "sustained": "Run ≥ 10 trials", "L6": "Joint change point",
              "cascade": "Lenient cascade", "gmm": "Gaussian mixture", "dtw": "DTW clustering"}
GREY_NO, GREY_NA = "0.86", "white"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def ptext(p):
    return f"p = {p:.2g}" if p >= 0.001 else "p < 0.001"


def order(d):
    """Mice sorted within cohort: consensus (high first), then Gaussian-mixture learner probability."""
    return {c: d[d.cohort == c].sort_values(["consensus", "gmm_p_learner", "frac_above"], ascending=False) for c in COHORTS}


def panel_matrix(ax, d):
    o = order(d)
    rows = CRITERIA + ["_gap", "consensus", "label", "sheet", "old_dtw", "fixed"]
    names = [CRIT_LABEL.get(r, "") for r in CRITERIA] + ["", "Consensus (fraction met)", "Agnostic label",
                                                          "Sheet category", "DTW clusters (old curves)", "Fixed non-learner (user)"]
    cols = []
    for k, c in enumerate(COHORTS):
        cols += [(c, r) for _, r in o[c].iterrows()]
        if k == 0:
            cols += [None, None]
    img = np.ones((len(rows), len(cols), 3))
    for j, cr in enumerate(cols):
        if cr is None:
            continue
        coh, r = cr
        col = np.array(plt.matplotlib.colors.to_rgb(C[coh]))
        for i, row in enumerate(rows):
            if row in CRITERIA:
                v = r[f"c_{row}"]
                img[i, j] = plt.matplotlib.colors.to_rgb(GREY_NA) if pd.isna(v) else col if v else plt.matplotlib.colors.to_rgb(GREY_NO)
            elif row == "consensus":
                img[i, j] = 1 - r.consensus * 0.85
            elif row == "label":
                img[i, j] = col if r.learner else plt.matplotlib.colors.to_rgb("0.25")
            elif row == "sheet":
                cat = r.learning_category
                img[i, j] = (col if cat == "good" else ps.light(C[coh], 0.55) if cat == "moderate"
                             else plt.matplotlib.colors.to_rgb("0.25") if cat == "bad" else (1, 1, 1))
            elif row == "old_dtw":
                v = r.old_cluster
                img[i, j] = (1, 1, 1) if pd.isna(v) else plt.matplotlib.colors.to_rgb(["0.3", "0.6", "0.85"][int(v)])
            elif row == "fixed":
                img[i, j] = plt.matplotlib.colors.to_rgb("0.15") if r.fixed_bad else (
                    plt.matplotlib.colors.to_rgb("0.65") if r.flexible else (1, 1, 1))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows)), names, fontsize=5)
    ax.tick_params(axis="y", length=0)
    n_p = len(o["R+"])
    ax.set_xticks([n_p / 2, n_p + 2 + len(o["R-"]) / 2], [f"R+ (n = {n_p})", f"R− (n = {len(o['R-'])})"])
    ax.tick_params(axis="x", length=0)
    for c, x0 in (("R+", 0), ("R-", n_p + 2)):
        ax.get_xticklabels()[COHORTS.index(c)].set_color(C[c])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Mice, sorted by consensus within cohort", labelpad=1)
    ax.axhline(len(CRITERIA) + 0.0, color="white", lw=2)
    # legend
    from matplotlib.patches import Patch
    h = [Patch(color=C["R+"], label="met / learner / good (R+ colour, R− colour)"),
         Patch(color=ps.light(C["R+"], 0.55), label="sheet moderate"), Patch(color="0.25", label="non-learner / sheet bad"),
         Patch(color=GREY_NO, label="not met"), Patch(color="0.15", label="fixed non-learner"),
         Patch(color="0.65", label="flexible (user)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=6, fontsize=4.5, handlelength=0.9,
              columnspacing=0.8, handletextpad=0.3)


def panel_kappa(ax, ag):
    labs = CRITERIA + ["sheet"]
    names = [CRIT_LABEL[c] for c in CRITERIA] + ["Sheet (good / moderate)"]
    g = ag[ag.reference == "final label"].set_index(["cohort", "labelling"])
    for k, c in enumerate(COHORTS):
        y = np.arange(len(labs)) + (k - 0.5) * 0.38
        v = np.array([g.loc[(c, l), "kappa"] if (c, l) in g.index else np.nan for l in labs])
        const = np.array([(c, l) in g.index and g.loc[(c, l), "n_learner"] in (0, g.loc[(c, l), "n"]) for l in labs])
        v[const] = np.nan  # a criterion met by every mouse has no kappa
        ax.barh(y, np.nan_to_num(v), 0.36, color=C[c], lw=0, label=COH_LABEL[c])
        for yy, cc in zip(y, const):
            if cc:
                ax.text(0.01, yy, "all mice met", va="center", fontsize=4, color=C[c])
    ax.set_yticks(range(len(labs)), names, fontsize=5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.4)
    ax.set_xlim(-0.2, 1.02)
    ax.set_xlabel("Agreement with the agnostic label\n(Cohen's κ)", labelpad=1)
    ax.legend(loc="lower right", fontsize=5, handlelength=0.8)


def panel_validation(ax, val):
    labs = ["final label", "sheet"] + CRITERIA
    names = ["Agnostic label", "Sheet"] + [CRIT_LABEL[c] for c in CRITERIA]
    g = val.set_index(["cohort", "outcome", "labelling"])
    for k, (c, out, mk) in enumerate([("R+", "hit_w_day1", "o"), ("R+", "dprime_day1", "s"), ("R-", "hit_w_day1", "o"),
                                      ("R-", "dprime_day1", "s")]):
        y = np.arange(len(labs)) + (k - 1.5) * 0.17
        v = np.array([g.loc[(c, out, l), "auc"] if (c, out, l) in g.index else np.nan for l in labs])
        p = np.array([g.loc[(c, out, l), "p"] if (c, out, l) in g.index else np.nan for l in labs])
        sig = p < 0.05
        ax.scatter(v[sig], y[sig], s=5, marker=mk, color=C[c], lw=0, zorder=3)
        ax.scatter(v[~sig], y[~sig], s=5, marker=mk, facecolor="white", edgecolor=C[c], lw=0.4, zorder=3)
    ax.axvline(0.5, color="0.5", lw=0.4, ls=":")
    ax.set_yticks(range(len(labs)), names, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("AUC for day +1 performance\n(0.5 = chance)", labelpad=1)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", lw=0, color="0.3", ms=2.5, label="W P(lick) day +1"),
                       Line2D([], [], marker="s", lw=0, color="0.3", ms=2.5, label="d' day +1"),
                       Line2D([], [], marker="o", lw=0, color=C["R+"], ms=2.5, label="R+"),
                       Line2D([], [], marker="o", lw=0, color=C["R-"], ms=2.5, label="R−"),
                       Line2D([], [], marker="o", lw=0, mfc="white", mec="0.3", ms=2.5, label="p ≥ 0.05")],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=4.5, handletextpad=0.2)


def panel_rule(ax, d, rule):
    r = rule[rule.feature == "frac_above"].set_index("cohort")
    for k, c in enumerate(COHORTS):
        g = d[d.cohort == c]
        rng = np.random.default_rng(k)
        for lr, dx in ((True, -0.18), (False, 0.18)):
            gg = g[g.learner == lr]
            x = k + dx + rng.uniform(-0.07, 0.07, len(gg))
            mk = gg.learning_category.map({"good": "o", "moderate": "^", "bad": "x"}).fillna("o")
            for m in ("o", "^", "x"):
                s = mk == m
                col = C[c] if lr else "0.25"
                ax.scatter(x[s], gg.frac_above[s], s=4 if m != "x" else 5, marker=m, lw=0.5 if m == "x" else 0,
                           color=col, zorder=3)
        th = r.loc[c, "threshold"]
        ax.plot([k - 0.4, k + 0.4], [th, th], color=C[c], lw=0.6, ls="--")
        ax.text(k, 1.04, f"≥ {th:.2f}" if r.loc[c, "side"] == ">=" else f"≤ {th:.2f}", ha="center", fontsize=4.5,
                color=C[c], transform=ax.get_xaxis_transform())
    ax.set_xticks([-0.18, 0.18, 0.82, 1.18], ["L", "NL", "L", "NL"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction of W trials with the\nW curve credibly above chance", labelpad=1)
    ax.set_xlabel("Agnostic label (L learner, NL non-learner)", labelpad=9, fontsize=5)
    for k, c in enumerate(COHORTS):
        ax.text(k, -0.2, COH_LABEL[c], color=C[c], ha="center", fontsize=5, transform=ax.get_xaxis_transform())
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", lw=0, color="0.4", ms=2, label="sheet good"),
                       Line2D([], [], marker="^", lw=0, color="0.4", ms=2, label="moderate"),
                       Line2D([], [], marker="x", lw=0, mew=0.5, color="0.4", ms=2.5, label="bad")],
              loc="upper right", fontsize=4.5, handletextpad=0.1, borderaxespad=0)


def panel_crosstab(ax, d):
    ax.axis("off")
    cats = ["good", "moderate", "bad"]
    y = 0.95
    for c in COHORTS:
        g = d[d.cohort == c]
        ax.text(0.0, y, f"{COH_LABEL[c]}", color=C[c], fontsize=6, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.45, y, "Learner", fontsize=5, transform=ax.transAxes, va="top", ha="center")
        ax.text(0.8, y, "Non-learner", fontsize=5, transform=ax.transAxes, va="top", ha="center")
        y -= 0.1
        for cat in cats:
            gg = g[g.learning_category == cat]
            ax.text(0.0, y, f"Sheet {cat}", fontsize=5, transform=ax.transAxes, va="top")
            for xx, lab in ((0.45, True), (0.8, False)):
                n = int((gg.learner == lab).sum())
                agree = (lab and cat != "bad") or (not lab and cat == "bad")
                ax.text(xx, y, str(n), fontsize=6, transform=ax.transAxes, va="top", ha="center",
                        fontweight="bold" if agree else "normal", color="k" if agree else "0.45")
            y -= 0.09
        y -= 0.06
    ax.text(0.0, y, "Bold: agnostic label agrees with the sheet", fontsize=4.5, color="0.4", transform=ax.transAxes, va="top")


def panel_curves(axs, d, crv):
    x = np.linspace(0, 1, 50)
    for ax, c in zip(axs, COHORTS):
        for lr, ls in ((True, "-"), (False, "--")):
            mice = d[(d.cohort == c) & (d.learner == lr)].mouse_id
            g = crv[crv.mouse_id.isin(mice)]
            for k, col in (("whisker_prog", C[c]), ("chance_prog", C["no_stim"])):
                m = g[[f"{k}_{i}" for i in range(50)]].to_numpy()
                mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m))
                ax.fill_between(x, mu - se, mu + se, color=ps.light(col, 0.8 if lr else 0.88), lw=0, edgecolor="none")
                ax.plot(x, mu, color=col, ls=ls, lw=0.8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Session progress (fraction of W trials)", labelpad=1)
        ax.set_ylabel("P(lick)", labelpad=1)
        nl, nn = int(d[(d.cohort == c)].learner.sum()), int((~d[(d.cohort == c)].learner).sum())
        ax.set_title(f"{COH_LABEL[c]}: {nl} learners, {nn} non-learners", color=C[c], fontsize=6, pad=2)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=C[c], label="Whisker"), Line2D([], [], color=C["no_stim"], label="False alarm"),
                           Line2D([], [], color="0.3", ls="-", label="Learners"), Line2D([], [], color="0.3", ls="--", label="Non-learners")],
                  loc="upper left", fontsize=4.5, ncol=2, handlelength=1.4, columnspacing=0.6)


PERF = [("hit_w", "W P(lick) day 0"), ("fa", "FA P(lick) day 0"), ("hit_a", "A P(lick) day 0"), ("dprime", "d' day 0"),
        ("hit_w_day1", "W P(lick) day +1"), ("dprime_day1", "d' day +1")]


def panel_perf(axs, d):
    for ax, (v, lab) in zip(axs, PERF):
        for k, c in enumerate(COHORTS):
            g = d[d.cohort == c]
            vals = {}
            for j, lr in enumerate((True, False)):
                y = g[g.learner == lr][v].dropna()
                x = k * 1.1 + j * 0.45
                rng = np.random.default_rng(10 * k + j)
                col = C[c] if lr else ps.light(C[c], 0.55)
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(y)), y, s=1.5, color=ps.light(col, 0.35), lw=0, zorder=2)
                ax.errorbar(x, y.mean(), y.sem(), color="0.15", marker="o", ms=2.2, mfc=col, mec="0.15", mew=0.4,
                            elinewidth=0.6, capsize=0, zorder=3)
                vals[lr] = y
            if len(vals[False]) >= 2:
                p = stats.mannwhitneyu(vals[True], vals[False]).pvalue
                top = np.nanmax([vals[True].max(), vals[False].max()])
                ax.text(k * 1.1 + 0.225, 1.0, stars(p), ha="center", va="bottom", fontsize=4.5,
                        transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 0.45, 1.1, 1.55], ["L", "NL", "L", "NL"], fontsize=4.5)
        ax.set_xlim(-0.3, 1.85)
        ax.set_ylabel(lab, labelpad=1, fontsize=5)
        ax.tick_params(labelsize=4.5)
        ax.text(0.225, -0.2, "R+", color=C["R+"], ha="center", fontsize=5, transform=ax.get_xaxis_transform())
        ax.text(1.325, -0.2, "R−", color=C["R-"], ha="center", fontsize=5, transform=ax.get_xaxis_transform())


def main():
    d = pd.read_csv(RES / "tables" / "001_per_mouse.csv")
    ag = pd.read_csv(RES / "tables" / "001_agreement.csv")
    val = pd.read_csv(RES / "tables" / "001_validation_day1.csv")
    rule = pd.read_csv(RES / "tables" / "001_simple_rule.csv")
    crv = pd.read_parquet(RES / "cache" / "001_curves_progress.parquet")
    ps.set_nature_rc()
    G = ps.PaperGrid(width_mm=183, n_cols=5, row_gap_mm=10, pad_left_mm=8, top_margin_mm=12)
    # row 1
    panel_matrix(G.axes_mm(0, 0, 125, 52, "a", margins_mm=(29, 9, 2, 1)), d)
    panel_kappa(G.axes_mm(127, 0, 56, 52, "b", margins_mm=(27, 9, 2, 1)), ag)
    # row 2
    panel_validation(G.axes_mm(0, 58, 62, 42, "c", margins_mm=(27, 8, 2, 1)), val)
    panel_rule(G.axes_mm(84, 58, 46, 42, "d", margins_mm=(11, 9, 4, 1)), d, rule)
    panel_crosstab(G.axes_mm(134, 58, 49, 42, "e", margins_mm=(2, 2, 3, 1)), d)
    # row 3
    box = G.axes_mm(0, 106, 85, 40, "f", margins_mm=(0, 0, 0, 0)); box.axis("off")
    axs = [G.axes_mm(x, 106, 41, 40, None, margins_mm=(9, 8, 4, 1)) for x in (0, 43)]
    panel_curves(axs, d, crv)
    box = G.axes_mm(88, 106, 95, 40, "g", margins_mm=(0, 0, 0, 0)); box.axis("off")
    axs = [G.axes_mm(88 + i * 15.8, 106, 15.8, 40, None, margins_mm=(7, 9, 4, 0.5)) for i in range(6)]
    panel_perf(axs, d)
    for out in (SNAP, RES / "figures"):
        ps.save_figure(G.fig, out, "SuppFigure_learners")
    G.fig.savefig(RES / "figures" / "SuppFigure_learners_tight.png", dpi=300, bbox_inches="tight", pad_inches=0.05)  # report copy
    print("height", G.figure_height_mm(), "->", SNAP)


if __name__ == "__main__":
    main()
