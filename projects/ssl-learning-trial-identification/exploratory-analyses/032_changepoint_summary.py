"""032 -- Summary of the Bayesian change-point (CP) learning-trial analysis (user 2026-10-01: "Show me a summary of the change
point analysis. Example mice. Distributions of CPs across cohorts. Definition summarized with examples").
Data: the LT chain on all 100 whisker day-0 sessions (028 workspace: 004 whisker-only CP = L5, 005 joint whisker + no-stim
CP = L6, L6 lenient), HMM curves (024, sigma = 1) for display only.
Figure 032_changepoint_summary (A4 portrait):
  a  definition (text): segment model, Bayes factor, categories, thresholds;
  b  example sessions (2 per row x 4 rows): raw whisker / no-stim licks, whisker and FA curves, joint-CP posterior over
     the change trial (shaded), posterior median (LT) with 90% credible interval, log10 BF and category;
  c  categories per cohort (L6), stacked counts;
  d  log10 BF distribution per cohort (L6 joint and L5 whisker-only), thresholds 0 (lenient) and 0.5 (strict);
  e  LT position (L6 / L6 lenient learners) in whisker trials and as fraction of the session, per cohort;
  f  credible-interval width vs log10 BF (how sharp the transitions are);
  g  L5 vs L6 LT agreement for sessions defined by both.
Outputs: exploratory-analyses/032_changepoint_summary.{pdf,png,svg}; artifacts/032_changepoint_table.csv
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/032_changepoint_summary.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
WS = ART / "028_chain_all"
COL = {"R+": "#00B400", "R-": "#C800C8"}
CAT_COL = {"learner": "#1a9850", "lenient": "#a6d96a", "high_from_start": "#fdae61", "never_licked": "#9e9ac8",
           "no_clear_transition": "#bdbdbd"}
FS_L, FS_M, FS_S = 8, 7, 6

DEF_TEXT = (
    "Change-point (CP) learning trial -- Bayesian segment model on RAW trial outcomes (not the smoothed curves)\n"
    "• Whisker trials are split into consecutive segments with constant lick rates; each rate has a flat Beta(1,1) prior and is\n"
    "  integrated out analytically, every allowed CP position gets equal prior weight (segments ≥ 5 whisker trials (MIN_SEG = 5)).\n"
    "• L5 (whisker only): R+ naive (a) → learned (b > a) [→ optional end decline]; R− generalising (a) → learned (b < a).\n"
    "• L6 (joint): each segment also has its own false-alarm rate f (no-stim trials assigned by time); learning = the\n"
    "  DISCRIMINATION b − f rises (R+) or falls (R−). Catches mice that learn by dropping false alarms.\n"
    "• Bayes factor (BF) = evidence for 'a change' vs 'no change' (flat, or flat → decline), averaged over all CP positions.\n"
    "• LT = posterior median of the first trial of the learned segment; 90% credible interval from the CP posterior.\n"
    "• Categories (L6): learner = log10 BF > 0.5 (≈ 3:1) and P(whisker > FA after) > 0.95 (R+) / P(whisker > FA before) > 0.95\n"
    "  and discrimination halved (R−); lenient = same with log10 BF > 0; high_from_start (R+) = already discriminating in\n"
    "  the first 20 whisker trials; never_licked (R−) = whisker never above FA (nothing to suppress); else no clear transition.")


def curve_xy(p):
    return np.arange(len(p["curves"]["whisker"])), p["curves"]["whisker"], p["curves"]["fa"]


def example_panel(ax, sid, inp, curves, cp, row, rg):
    d = inp[sid]
    x, w, fa = curve_xy(curves[sid])
    n = len(x)
    pk = np.asarray(cp[sid]["pk"], float) if cp.get(sid) is not None else None
    ax2 = ax.twinx()
    if pk is not None and np.nansum(pk) > 0:
        ax2.fill_between(np.arange(len(pk)), 0, pk / np.nanmax(pk), color="#1f77b4", alpha=0.18, step="mid", lw=0)
    ax2.set_ylim(0, 3.2)
    ax2.set_axis_off()
    ax.plot(x, fa, color="0.4", lw=1.0)
    ax.plot(x, w, color=COL[rg], lw=1.3)
    yw = np.asarray(d["w_outcomes"])
    ax.vlines(x[yw == 1], 1.05, 1.13, color=COL[rg], lw=0.5)
    ax.vlines(x[yw == 0], -0.13, -0.05, color=COL[rg], lw=0.5, alpha=0.5)
    tw, tn, yn = np.asarray(d["w_start"]), np.asarray(d["n_start"]), np.asarray(d["n_outcomes"])
    xn = np.interp(tn, tw, x)
    ax.vlines(xn[yn == 1], 1.15, 1.23, color="k", lw=0.5)
    lt, lo, hi = row["L6_cp_median"], row["L6_ci05"], row["L6_ci95"]
    defined = pd.notna(row["L6"]) or pd.notna(row.get("L6_lenient", np.nan))
    if pd.notna(lt):
        ax.axvspan(lo, hi, color="#1f77b4", alpha=0.08, lw=0)
        ax.axvline(lt, color="#1f77b4", lw=1.2, ls="-" if defined else ":")
    ax.set_ylim(-0.16, 1.27)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=FS_S, length=2)
    cat = row["L6_category"]
    if cat != "learner" and pd.notna(row.get("L6_lenient", np.nan)):
        cat = "lenient learner"
    ax.set_title(f"{d['mouse_id']} ({rg}) · {cat.replace('_', ' ')}\nlog10 BF {row['L6_log10_bf']:.1f} · "
                 + (f"LT {lt:.0f} [{lo:.0f}, {hi:.0f}]" if defined else "no LT"), fontsize=FS_S, color=COL[rg])


def pick_examples(t):
    ex = []
    for rg, cats in (("R+", [("learner", "max"), ("learner", "low"), ("lenient", None), ("high_from_start", None),
                             ("no_clear_transition", None)]),
                     ("R-", [("learner", "max"), ("learner", "low"), ("never_licked", None)])):
        g = t[t.reward_group == rg]
        for cat, how in cats:
            if cat == "lenient":
                s = g[(g.L6_category != "learner") & g.L6_lenient.notna()]
            else:
                s = g[g.L6_category == cat]
            if not len(s):
                continue
            s = s.sort_values("L6_log10_bf", ascending=False)
            if how == "low":
                s = s.iloc[len(s) // 2:]
            pick = s.iloc[0] if how in ("max", "low") else s.iloc[len(s) // 2]
            if pick.session_id not in [e[0] for e in ex]:
                ex.append((pick.session_id, rg))
    return ex[:8]


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS_M})
    t = pd.read_csv(WS / "005_learning_trials.csv")
    t4 = pd.read_csv(WS / "004_learning_trials.csv")[["session_id", "L5_cp_median", "L5_log10_bf", "L5_category", "L5"]]
    t = t.drop(columns=[c for c in t4.columns if c in t.columns and c != "session_id"]).merge(t4, on="session_id")
    inp = pickle.load(open(WS / "001_inputs.pkl", "rb"))
    cp = pickle.load(open(WS / "005_cp_posteriors.pkl", "rb"))
    curves = {p["session_id"]: p for p in pickle.load(open(ART / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    t["n_whisker"] = t.session_id.map(lambda s: len(inp[s]["w_outcomes"]))
    t["cat_plot"] = np.where((t.L6_category != "learner") & t.L6_lenient.notna(), "lenient", t.L6_category)
    t["ci_width"] = t.L6_ci95 - t.L6_ci05
    t.to_csv(ART / "032_changepoint_table.csv", index=False)
    ex = pick_examples(t)
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(7, 4, height_ratios=[1.15, 1, 1, 1, 1, 1.1, 1.1], hspace=0.9, wspace=0.45,
                          left=0.07, right=0.98, top=0.98, bottom=0.04)
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.text(0, 1, DEF_TEXT, fontsize=FS_S + 0.3, va="top", family="DejaVu Sans")
    ax.text(-0.04, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    tidx = t.set_index("session_id")
    for i, (sid, rg) in enumerate(ex):
        ax = fig.add_subplot(gs[1 + i // 2, (i % 2) * 2:(i % 2) * 2 + 2])
        example_panel(ax, sid, inp, curves, cp, tidx.loc[sid], rg)
        if i // 2 == 3 or i == len(ex) - 1:
            ax.set_xlabel("whisker trial", fontsize=FS_S)
        if i % 2 == 0:
            ax.set_ylabel("P(lick)", fontsize=FS_S)
        if i == 0:
            ax.text(-0.1, 1.25, "b", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
            ax.text(1.0, 1.32, "curves: whisker (cohort colour), false alarm (grey); ticks: whisker licks / misses, FA licks "
                    "(black); blue: CP posterior, LT and 90% CI (dotted = below threshold)", transform=ax.transAxes,
                    fontsize=FS_S - 0.5, ha="center")
    # c categories
    ax = fig.add_subplot(gs[5, 0])
    order = ["learner", "lenient", "high_from_start", "never_licked", "no_clear_transition"]
    for j, rg in enumerate(("R+", "R-")):
        g = t[t.reward_group == rg].cat_plot.value_counts()
        bottom = 0
        for c in order:
            v = g.get(c, 0)
            if v:
                ax.bar(j, v, bottom=bottom, color=CAT_COL[c], width=0.6)
                ax.text(j, bottom + v / 2, f"{v}", ha="center", va="center", fontsize=FS_S - 0.5)
                bottom += v
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R+", "R−"])
    for lab, cl in zip(ax.get_xticklabels(), (COL["R+"], COL["R-"])):
        lab.set_color(cl)
    ax.set_ylabel("mice", fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    for k, c in enumerate(order):
        ax.text(1.05, 0.95 - 0.13 * k, c.replace("_", " "), color=CAT_COL[c], transform=ax.transAxes, fontsize=FS_S - 0.5,
                fontweight="bold")
    ax.set_title("L6 categories", fontsize=FS_M)
    ax.text(-0.35, 1.1, "c", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    # d BF distributions
    ax = fig.add_subplot(gs[5, 2])
    bins = np.linspace(-3, 8, 23)
    for rg in ("R+", "R-"):
        v = t[t.reward_group == rg].L6_log10_bf.clip(-3, 8)
        ax.hist(v, bins=bins, histtype="step", color=COL[rg], lw=1.3, label=f"{'R+' if rg == 'R+' else 'R−'} L6")
        v5 = t[t.reward_group == rg].L5_log10_bf.clip(-3, 8)
        ax.hist(v5, bins=bins, histtype="step", color=COL[rg], lw=0.8, ls="--", label="L5 (whisker only)" if rg == "R+" else None)
    for th, ls in ((0, ":"), (0.5, "--")):
        ax.axvline(th, color="k", lw=0.7, ls=ls)
    ax.set_xlabel("log10 BF (clipped −3..8)", fontsize=FS_S)
    ax.set_ylabel("mice", fontsize=FS_S)
    ax.legend(fontsize=FS_S - 0.5, frameon=False)
    ax.tick_params(labelsize=FS_S)
    ax.set_title("evidence for a transition", fontsize=FS_M)
    ax.text(-0.3, 1.1, "d", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    # e LT distributions
    for k, (col, lab, xl) in enumerate((("L6_lenient", "LT (whisker trial)", None), ("frac", "LT (fraction of session)", (0, 1)))):
        ax = fig.add_subplot(gs[6, k])
        for j, rg in enumerate(("R+", "R-")):
            g = t[(t.reward_group == rg) & t.L6_lenient.notna()]
            v = g.L6_lenient / (g.n_whisker - 1) if col == "frac" else g.L6_lenient
            strict = g.L6.notna().to_numpy()
            xs = j + np.random.default_rng(0).uniform(-0.15, 0.15, len(v))
            ax.scatter(xs[strict], v[strict], s=10, color=COL[rg], lw=0)
            ax.scatter(xs[~strict], v[~strict], s=10, facecolor="white", edgecolor=COL[rg], lw=0.7)
            ax.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="k", lw=1)
        a = t[(t.reward_group == "R+") & t.L6_lenient.notna()]
        b = t[(t.reward_group == "R-") & t.L6_lenient.notna()]
        va, vb = ((a.L6_lenient / (a.n_whisker - 1), b.L6_lenient / (b.n_whisker - 1)) if col == "frac"
                  else (a.L6_lenient, b.L6_lenient))
        pm, pw = stats.mannwhitneyu(va, vb).pvalue, stats.ttest_ind(va, vb, equal_var=False).pvalue
        ax.set_title(f"{lab}\nMW p = {pm:.2f}, Welch p = {pw:.2f}", fontsize=FS_S)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"R+ (n={len(a)})", f"R− (n={len(b)})"], fontsize=FS_S)
        ax.tick_params(labelsize=FS_S)
        if xl:
            ax.set_ylim(*xl)
        if k == 0:
            ax.text(-0.35, 1.15, "e", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
            ax.text(0.0, -0.32, "filled = L6 strict, open = lenient only; bar = median", transform=ax.transAxes,
                    fontsize=FS_S - 0.5)
    # f CI width vs BF
    ax = fig.add_subplot(gs[6, 2])
    for rg in ("R+", "R-"):
        g = t[t.reward_group == rg]
        ax.scatter(g.L6_log10_bf.clip(-3, 8), g.ci_width / g.n_whisker, s=8, color=COL[rg], lw=0, alpha=0.8)
    ax.axvline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xlabel("log10 BF (clipped)", fontsize=FS_S)
    ax.set_ylabel("90% CI width / session length", fontsize=FS_S)
    ax.set_title("sharper transitions = narrower CI", fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    ax.text(-0.3, 1.15, "f", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    # g L5 vs L6
    ax = fig.add_subplot(gs[6, 3])
    for rg in ("R+", "R-"):
        g = t[(t.reward_group == rg) & t.L5.notna() & t.L6_lenient.notna()]
        ax.scatter(g.L5, g.L6_lenient, s=9, color=COL[rg], lw=0)
    lim = np.nanmax(t[["L5", "L6_lenient"]].to_numpy()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.6, ls="--")
    both = t[t.L5.notna() & t.L6_lenient.notna()]
    md = np.median(np.abs(both.L5 - both.L6_lenient)) if len(both) else np.nan
    ax.set_title(f"L5 vs L6 LT (n = {len(both)})\nmedian |diff| = {md:.0f} trials", fontsize=FS_S)
    ax.set_xlabel("L5 whisker-only LT", fontsize=FS_S)
    ax.set_ylabel("L6 joint LT", fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    ax.text(-0.3, 1.15, "g", transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(HERE / f"032_changepoint_summary.{ext}", dpi=250)
    plt.close(fig)
    print("examples:", ex)
    print(t.groupby(["reward_group", "cat_plot"]).size())
    for rg in ("R+", "R-"):
        g = t[(t.reward_group == rg) & t.L6_lenient.notna()]
        print(rg, "LT median", g.L6_lenient.median(), "frac", (g.L6_lenient / (g.n_whisker - 1)).median(),
              "BF median", t[t.reward_group == rg].L6_log10_bf.median())


if __name__ == "__main__":
    main()
