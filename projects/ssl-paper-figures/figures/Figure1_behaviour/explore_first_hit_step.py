"""Panel i preview (user 2026-11-09): lick probability before and after the first whisker hit, data only.

Why: per-trial alignment (trials -5..-1) mixes mice (few mice have many trials of a type before their first whisker
hit, and those are the slow, low-licking ones), and the smoothed curves leak post-hit behaviour backwards. Here each
mouse contributes one value per position:
  'Before' = all trials of that type from the session start to the first whisker hit (no-stimulus and auditory only;
             whisker trials before the first hit are misses by definition);
  bins of 5 trials after it (trial 0 = first trial of the type at or after the first whisker hit; whisker trial 0, the
             hit itself, left out, so the first whisker bin is trials 1-4), up to trial 29.
Same day-0 learner sessions and trial filters as Figure 1 compute_aligned_curves (early licks dropped).
Stats: R+ vs R- cluster-based permutation test over positions (Figure 1 cluster_test) and Mann-Whitney + Welch per
position; within cohort, Before vs first bin (Wilcoxon signed-rank, paired mice).
Output: Figures/_explore/panel_i_first_hit_step_<date>.png and _stats_<date>.csv.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_first_hit_step.py
"""

import datetime
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, FIG_DIR / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C, P = _load("fig1_compute", "compute.py"), _load("fig1_plot", "plot.py")
TT = ["whisker_trial", "no_stim_trial", "auditory_trial"]
TITLE = {"whisker_trial": "Whisker", "no_stim_trial": "False alarm", "auditory_trial": "Auditory"}
BEFORE, LAST = -1, 29  # position of the 'Before' point; last trial after the hit
N_BIN = (LAST + 1) // 5


def per_mouse(table):
    rows = []
    for sid, st in C.day0_table(table, learners_only=True).groupby("session_id"):
        s0 = st[st.early_lick == 0].sort_values("start_time")
        w = s0[s0.trial_type == "whisker_trial"]
        if not (w.lick_flag == 1).any():
            continue
        t_hit = w[w.lick_flag == 1].start_time.iloc[0]
        base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for tt in TT:
            d = s0[s0.trial_type == tt]
            y = d.lick_flag.to_numpy(float)
            rel = np.arange(len(d)) - int(np.sum(d.start_time.to_numpy() < t_hit))
            if tt != "whisker_trial" and (rel < 0).any():
                rows.append({**base, "trial_type": tt, "position": BEFORE, "n_trials": int((rel < 0).sum()),
                             "p_lick": y[rel < 0].mean()})
            keep = (rel >= (1 if tt == "whisker_trial" else 0)) & (rel <= LAST)
            for b in range(N_BIN):
                sel = keep & (rel // 5 == b)
                if sel.any():
                    rows.append({**base, "trial_type": tt, "position": b, "n_trials": int(sel.sum()), "p_lick": y[sel].mean()})
    return pd.DataFrame(rows)


def tests(df):
    out = []
    for tt, g in df.groupby("trial_type"):
        for c in C.cluster_test(g, "position", "p_lick"):
            out.append({"trial_type": tt, "test": "cluster R+ vs R-", **c})
        for pos, gp in g.groupby("position"):
            a, b = gp[gp.cohort == "R+"].p_lick, gp[gp.cohort == "R-"].p_lick
            out.append({"trial_type": tt, "test": "per position R+ vs R-", "start": pos, "end": pos, "n_R+": len(a),
                        "n_R-": len(b), "mannwhitney_p": stats.mannwhitneyu(a, b).pvalue,
                        "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue})
        if tt != "whisker_trial":
            for c in P.COHORTS:
                w = g[g.cohort == c].pivot_table(index="mouse_id", columns="position", values="p_lick")[[BEFORE, 0]].dropna()
                out.append({"trial_type": tt, "test": f"{c} before vs trials 0-4", "n_mice": len(w),
                            "before": w[BEFORE].mean(), "after": w[0].mean(),
                            "wilcoxon_p": stats.wilcoxon(w[BEFORE], w[0]).pvalue})
    return pd.DataFrame(out)


def xpos(pos):
    return np.where(np.asarray(pos) == BEFORE, -4.0, np.asarray(pos) * 5 + 2.0)


def plot(df, st, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for a, tt in zip(axes, TT):
        g = df[df.trial_type == tt]
        for k, c in enumerate(P.COHORTS):
            gc, color, dx = g[g.cohort == c], P.group_color(tt, c), (0.5 if c == "R-" else -0.5)
            if tt != "whisker_trial":  # individual mice, Before -> trials 0-4
                w = gc.pivot_table(index="mouse_id", columns="position", values="p_lick")[[BEFORE, 0]].dropna()
                for _, r in w.iterrows():
                    a.plot(xpos([BEFORE, 0]) + dx, r.to_numpy(), color=color, lw=0.25, alpha=0.25)
            m = gc.groupby("position").p_lick
            x = xpos(m.mean().index) + dx
            after = m.mean().index >= 0
            a.errorbar(x, m.mean(), m.sem(), color=color, marker="o", ms=2.5, lw=0, elinewidth=0.7, capsize=0, zorder=3)
            a.plot(x[after], m.mean()[after], color=color, lw=0.7, zorder=3)
        for r in st[(st.trial_type == tt) & (st.test == "cluster R+ vs R-") & (st.p < 0.05)].itertuples():
            a.plot([xpos(r.start) - 2, xpos(r.end) + 2], [1.05, 1.05], color="k", lw=1.2, solid_capstyle="butt", clip_on=False)
        for r in st[(st.trial_type == tt) & (st.test == "per position R+ vs R-") & (st.mannwhitney_p < 0.05)].itertuples():
            a.text(xpos(r.start), 0.97, P.stars(r.mannwhitney_p), ha="center", fontsize=6)
        for k, r in enumerate(st[(st.trial_type == tt) & st.test.str.contains("before vs")].itertuples()):
            a.text(0.02, 0.06 + 0.07 * k, f"{r.test.split()[0]} before vs 0-4: p = {r.wilcoxon_p:.2g} (n = {r.n_mice})",
                   transform=a.transAxes, fontsize=5, color=P.group_color(tt, r.test.split()[0]))
        a.axvline(-1.5, color="0.6", lw=0.4, ls=":")
        ticks = [BEFORE] + list(range(N_BIN))
        labels = ["Before"] + [f"{5 * b + (1 if (tt == 'whisker_trial' and b == 0) else 0)}-{5 * b + 4}" for b in range(N_BIN)]
        if tt == "whisker_trial":
            ticks, labels = ticks[1:], labels[1:]
        a.set_xticks(xpos(ticks), labels, rotation=45, fontsize=5.5)
        a.set_xlim(-7, xpos(N_BIN - 1) + 3)
        a.set_ylim(-0.02, 1.02)
        a.set_title(TITLE[tt], fontsize=7, pad=8)
        a.set_xlabel("Trials from 1st W hit", fontsize=6)
        a.tick_params(labelsize=6)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(lick)", fontsize=7)
    fig.text(0.5, 0.005, "Before: all trials of the type from session start to the 1st whisker hit (one value per mouse); "
             "bars: cluster permutation R+ vs R-; stars: Mann-Whitney per position (uncorrected); faint lines: mice.",
             ha="center", fontsize=5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)


def main():
    date = datetime.date.today().isoformat()
    df = per_mouse(shared_behaviour.load())
    st = tests(df)
    out = FIGURES_ROOT / "_explore" / f"panel_i_first_hit_step_{date}.png"
    plot(df, st, out)
    df.to_csv(str(out).replace(".png", "_data.csv"), index=False)
    st.to_csv(str(out).replace(".png", "_stats.csv"), index=False)
    print(out)
    print(df[df.position == BEFORE].groupby(["trial_type", "cohort"]).n_trials.describe().round(1).to_string())
    print(st.round(4).to_string())


if __name__ == "__main__":
    main()
