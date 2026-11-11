"""Exploration (user 2026-11-10): alternative views of Figure 1 panel i (whisker, false-alarm and auditory learning
aligned at the first whisker hit): learning curves only, the curves interpolated at whisker trials, or the data alone;
50 trials after the alignment, publication layout (user 2026-11-10). Whisker day 0, learners, early licks dropped; exact
curves (learning_curve_exact, sigma 1) as in compute_aligned_curves.

Rows (columns: whisker, false alarm, auditory; R+ / R- in panel-i colours; mean +- s.e.m. over mice):
  a  curves only, each trial type on its own trials (trial 0: first whisker hit; first no-stim / auditory trial at or
     after it), -5..49 (whisker from 0, planner);
  b  curves interpolated in time onto whisker trials (chance and auditory curves at each whisker trial, as in the example
     panels e-f), one shared x axis: whisker trials from the first whisker hit;
  c  data only, each trial type on its own trials, fraction licked per mouse in bins of 5 trials (-5..-1, 0..4, ...,
     45..49), plotted at the bin centre;
  d  data on the whisker-trial axis: every trial is placed at the number of whisker trials since the first whisker hit
     (no-stim and auditory trials between whisker trials k and k+1 count as k), bins of 5 whisker trials.
Statistics (user 2026-11-10: the cohort-level cluster test marked everything after the first whisker hit, since the
cohorts differ in level from there on): change over trials, per bin of 5 trials, from a reference bin (whisker: trials
0-4; false alarm and auditory: the 5 trials before the first whisker hit), per mouse (mean of the curve or of the licks in
the bin minus the same in the reference bin). Within cohort, Wilcoxon signed-rank vs 0; R+ vs R-, Mann-Whitney on the
change (difference in differences; Welch in the CSV); Benjamini-Hochberg over the bins of each family (per row, column
and test). Only trials / bins with >= 8 mice per cohort are shown and tested (later trials come from fewer mice).
Output: Figures/_explore/aligned_views_<date>.{pdf,png,svg}; _change_tests.csv; _trials.csv (per mouse and trial, unbinned, window -5..59;
--replot redraws from it without refitting).
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_aligned_views.py [--replot DATE]
"""

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import MM, light, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
LO, HI, HI_STORE, BIN, MIN_MICE = -5, 49, 59, 5, 8
REF_BIN = {"whisker_trial": 0, "no_stim_trial": -1, "auditory_trial": -1}  # trials 0-4 / the 5 trials before the hit
TTS = [("whisker_trial", "Whisker"), ("no_stim_trial", "False alarm"), ("auditory_trial", "Auditory")]
VIEWS = {"a": ("Curves", "own trials"), "b": ("Curves", "at whisker trials"), "c": ("Data", "own trials"),
         "d": ("Data", "at whisker trials")}


def build():
    """Unbinned per mouse and trial: view (A-D), x (trial index from the first whisker hit), p_lick."""
    import learning_curve_exact as lce
    from compute import CURVE_PARAMS, day0_table
    rows = []
    for sid, st in day0_table(shared_behaviour.load(), learners_only=True).groupby("session_id"):
        s0 = st[st.early_lick == 0].sort_values("start_time")
        out = lce.session_learning_curves_exact(s0, sigma=CURVE_PARAMS["sigma"], conf_int=CURVE_PARAMS["conf_int"])
        if "whisker_trial" not in out or not (out["whisker_trial"]["outcomes"] == 1).any():
            continue
        w = out["whisker_trial"]
        i_hit = int(np.argmax(w["outcomes"] == 1))
        t_hit, w_times = w["start_time"][i_hit], np.asarray(w["start_time"])
        w_rel = np.arange(len(w["outcomes"])) - i_hit
        at_w = {"whisker_trial": w["p_mean"], "no_stim_trial": w["p_chance"],
                "auditory_trial": out.get("auditory_trial", {}).get("at_whisker", {}).get("p_mean")}
        base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for tt, _ in TTS:
            if tt not in out:
                continue
            y, times = np.asarray(out[tt]["outcomes"], float), np.asarray(out[tt]["start_time"])
            lo = 0 if tt == "whisker_trial" else LO
            rel = np.arange(len(y)) - int(np.sum(times < t_hit))
            wrel = rel if tt == "whisker_trial" else np.searchsorted(w_times, times, side="right") - 1 - i_hit
            k, kd = (rel >= lo) & (rel <= HI_STORE), (wrel >= lo) & (wrel <= HI_STORE)
            rows += [pd.DataFrame({**base, "trial_type": tt, "view": "a", "x": rel[k], "p_lick": np.asarray(out[tt]["p_mean"])[k]}),
                     pd.DataFrame({**base, "trial_type": tt, "view": "c", "x": rel[k], "p_lick": y[k]}),
                     pd.DataFrame({**base, "trial_type": tt, "view": "d", "x": wrel[kd], "p_lick": y[kd]})]
            if at_w[tt] is not None:
                kw = (w_rel >= lo) & (w_rel <= HI_STORE)
                rows.append(pd.DataFrame({**base, "trial_type": tt, "view": "b", "x": w_rel[kw],
                                          "p_lick": np.asarray(at_w[tt], float)[kw]}))
    return pd.concat(rows, ignore_index=True)


def per_mouse(df):
    """Window LO..HI; data views binned per mouse (bin centre); several trials of one type at one whisker index are
    averaged per mouse (view d)."""
    df = df[(df.x >= LO) & (df.x <= HI)].copy()
    data = df.view.isin(["c", "d"])
    df.loc[data, "x"] = np.floor(df.loc[data, "x"] / BIN) * BIN + (BIN - 1) / 2
    return df.groupby(["mouse_id", "cohort", "trial_type", "view", "x"], as_index=False).p_lick.mean()


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def change_tests(d, tt):
    """Per bin of 5 trials: per-mouse mean minus the mouse's reference-bin mean; within cohort Wilcoxon vs 0, R+ vs R-
    Mann-Whitney on the change (difference in differences); Benjamini-Hochberg over the bins of each family."""
    from common import two_group_tests
    from statsmodels.stats.multitest import multipletests
    b = d.assign(bin=np.floor(d.x / BIN).astype(int)).groupby(["mouse_id", "cohort", "bin"]).p_lick.mean().unstack()
    if REF_BIN[tt] not in b:
        return pd.DataFrame(columns=["bin", "R+_q", "R-_q", "diff_q"])
    ch = b.sub(b[REF_BIN[tt]], axis=0).drop(columns=REF_BIN[tt])
    rows = []
    for bb in ch.columns:
        x = {c: ch.xs(c, level="cohort")[bb].dropna() for c in ("R+", "R-")}
        if min(len(v) for v in x.values()) < MIN_MICE:
            continue
        rows.append(dict(bin=bb, **{f"{c}_change": x[c].mean() for c in x}, **{f"{c}_n": len(x[c]) for c in x},
                         **{f"{c}_p": stats.wilcoxon(x[c]).pvalue if (x[c] != 0).any() else 1.0 for c in x},
                         **two_group_tests(x["R+"], x["R-"])))
    r = pd.DataFrame(rows)
    if len(r):
        for p, q in (("R+_p", "R+_q"), ("R-_p", "R-_q"), ("mannwhitney_p", "diff_q"), ("welch_p", "diff_welch_q")):
            r[q] = multipletests(r[p], method="fdr_bh")[1]
    return r


def plot(df, stem):
    from plot import COHORTS, group_color
    set_nature_rc()
    tests = []
    fig, axs = plt.subplots(4, 3, figsize=(120 * MM, 136 * MM), sharex=True, sharey=True,
                            gridspec_kw=dict(left=0.12, right=0.98, top=0.9, bottom=0.085, hspace=0.62, wspace=0.12))
    for r, (v, (kind, where)) in enumerate(VIEWS.items()):
        for c, (tt, tname) in enumerate(TTS):
            ax, d = axs[r, c], df[(df.view == v) & (df.trial_type == tt)]
            n = d.groupby(["x", "cohort"]).mouse_id.nunique().unstack()
            ok = n.index[(n.reindex(columns=COHORTS) >= MIN_MICE).all(axis=1)]
            d = d[d.x.isin(ok)]
            for i, co in enumerate(COHORTS):
                col, g = group_color(tt, co), d[d.cohort == co].groupby("x").p_lick
                m, e = g.mean(), g.sem()
                if kind == "Curves":
                    ax.fill_between(m.index, m - e, m + e, color=light(col, 0.65), lw=0)
                    ax.plot(m.index, m, color=col, lw=0.8)
                else:
                    ax.plot(m.index, m, color=col, lw=0.5)
                    ax.errorbar(m.index + (i - 0.5) * 0.9, m, e, color=col, marker="o", ms=2.2, mfc="white", mew=0.6, lw=0,
                                elinewidth=0.6, capsize=0, zorder=3)
            t = change_tests(d, tt)
            tests.append(t.assign(view=v, trial_type=tt))
            ref = REF_BIN[tt] * BIN + (BIN - 1) / 2
            ax.plot([ref - BIN / 2, ref + BIN / 2], [-0.03, -0.03], color="0.4", lw=1.2, solid_capstyle="butt",
                    clip_on=False, transform=ax.get_xaxis_transform())  # reference bin
            for _, rr in t.iterrows():
                xb = rr.bin * BIN + (BIN - 1) / 2
                for k, (key, col) in enumerate((("R+_q", group_color(tt, "R+")), ("R-_q", group_color(tt, "R-")),
                                                ("diff_q", "k"))):
                    if rr[key] < 0.05:
                        ax.text(xb, 1.0 + 0.075 * k, stars(rr[key]), color=col, ha="center", va="bottom", fontsize=5,
                                transform=ax.get_xaxis_transform(), clip_on=False)
            ax.axvline(0, color="0.6", lw=0.4, ls=":", zorder=0)
            ax.set_xlim(LO - 1, HI + 1)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 10, 20, 30, 40, 50])
            ax.tick_params(length=2, width=0.5, labelsize=5, pad=1.5)
            if r == 0:
                ax.set_title(tname, fontsize=6.5, pad=21)
                ax.legend(handles=[Line2D([], [], color=group_color(tt, co), lw=1.2, label=co) for co in COHORTS],
                          loc={"whisker_trial": "lower left", "no_stim_trial": "upper right",
                               "auditory_trial": "lower right"}[tt], fontsize=5, frameon=False, handlelength=1.2, borderaxespad=0.2)
            if c == 0:
                ax.set_ylabel(f"{kind}, {where}\nP(lick)", fontsize=5.5, labelpad=1)
                ax.text(-0.33, 1.12, v, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
            own = {"whisker_trial": "Whisker", "no_stim_trial": "No-stimulus", "auditory_trial": "Auditory"}[tt]
            ax.set_xlabel(("Whisker" if where == "at whisker trials" else own) + " trials from first whisker hit",
                          fontsize=5, labelpad=1)
            ax.xaxis.set_tick_params(labelbottom=True)
    fig.legend(handles=[Line2D([], [], color="0.3", lw=0.8, label="Learning curve (mean ± s.e.m.)"),
                        Line2D([], [], color="0.3", marker="o", mfc="white", ms=2.5, lw=0.5,
                               label=f"Fraction licked, bins of {BIN} (mean ± s.e.m.)"),
                        Line2D([], [], color="0.4", lw=1.2, label="Reference bin")],
               loc="upper center", ncol=3, fontsize=5, frameon=False, bbox_to_anchor=(0.55, 1.0), handlelength=1.6)
    fig.text(0.55, 0.008, "Stars per bin of 5 trials, change from the reference bin (per mouse; Benjamini-Hochberg over "
             "bins):\ncoloured, R+ (lower row) / R− (middle row) Wilcoxon vs 0; black (top row), R+ vs R− Mann-Whitney on "
             "the change", ha="center", va="bottom", fontsize=4.5)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=600)
    t = pd.concat(tests)
    t.to_csv(f"{stem}_change_tests.csv", index=False)
    with pd.option_context("display.width", 250):
        print(t[["view", "trial_type", "bin", "R+_n", "R-_n", "R+_change", "R+_q", "R-_change", "R-_q", "diff_q",
                 "diff_welch_q"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", help="date of an existing _trials.csv to redraw without refitting")
    a = ap.parse_args()
    if a.replot:
        stem = OUT / f"aligned_views_{a.replot}"
        trials = pd.read_csv(f"{stem}_trials.csv")
    else:
        stem = OUT / f"aligned_views_{datetime.date.today()}"
        trials = build()
        trials.to_csv(f"{stem}_trials.csv", index=False)
    plot(per_mouse(trials), stem)
    print(stem, trials.groupby("cohort").mouse_id.nunique().to_dict())


if __name__ == "__main__":
    main()
