"""What do the sheet's 'bad' mice have in common, per cohort, and which explicit curve-based criterion reproduces the
learner / non-learner split (user 2026-11-11: learning_category is not principled)?

Per mouse (whisker day 0, early licks dropped, as Figure 1): exact whisker and chance (no-stim) curves on whisker trials
(compute.session_curves), then candidate measures of a high-performance state:
  gap            = p_mean(whisker) - p_chance, per whisker trial (max, mean, mean over the last 20 trials)
  run_above      = longest run of whisker trials with p_low > p_chance (whisker curve credibly above chance)
  run_below      = longest run with p_high < p_chance (whisker curve credibly below chance)
  run_not_above  = longest run with p_low <= p_chance after the first hit (whisker not credibly above chance)
  repo rule      = learning_curve_exact.learning_trial (mouse_cat), assign_expertise_blocks high state
plus raw rates (whisker hit, false alarm, auditory), trial counts, first whisker hit, whisker hit rate on day +1.
Outputs (Figures/_explore/learner_criterion_<date>/): per-mouse table, per-cohort comparison of bad vs learners
(medians, Mann-Whitney, AUC), threshold sweeps for the best measures, and a figure.

Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python
  projects/ssl-paper-figures/figures/Figure1_behaviour/explore_learner_criterion.py
"""

import importlib.util
import sys
from datetime import date
from multiprocessing import Pool
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

_spec = importlib.util.spec_from_file_location("fig1_compute", FIG_DIR / "compute.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def mouse_measures(st):
    st = st[st.early_lick == 0].sort_values("start_time")
    curves, lt, cat = fig1.session_curves(st)
    y = curves.lick.to_numpy()
    gap = (curves.p_mean - curves.p_chance).to_numpy()
    above = (curves.p_low > curves.p_chance).to_numpy()
    below = (curves.p_high < curves.p_chance).to_numpy()
    first = int(np.argmax(y == 1)) if (y == 1).any() else len(y)
    rate = st.groupby("trial_type").lick_flag.mean()
    n = st.trial_type.value_counts()
    # d' (user 2026-11-11): whole session with the log-linear correction (counts + 0.5, trials + 1), and time-resolved
    # from the exact curves, z(p_whisker) - z(p_chance) per whisker trial
    hits, nw = st[st.trial_type == "whisker_trial"].lick_flag.sum(), n.get("whisker_trial", 0)
    fas, nf = st[st.trial_type == "no_stim_trial"].lick_flag.sum(), n.get("no_stim_trial", 0)
    dprime = stats.norm.ppf((hits + 0.5) / (nw + 1)) - stats.norm.ppf((fas + 0.5) / (nf + 1))
    clip = lambda v: np.clip(np.asarray(v, float), 0.01, 0.99)
    dp_t = stats.norm.ppf(clip(curves.p_mean)) - stats.norm.ppf(clip(curves.p_chance))
    dp_low = stats.norm.ppf(clip(curves.p_low)) - stats.norm.ppf(clip(curves.p_chance))
    return dict(
        mouse_id=st.mouse_id.iloc[0], session_id=st.session_id.iloc[0], cohort=st.cohort.iloc[0],
        learning_category=st.learning_category.iloc[0],
        n_whisker=int(n.get("whisker_trial", 0)), n_no_stim=int(n.get("no_stim_trial", 0)),
        n_auditory=int(n.get("auditory_trial", 0)),
        duration_min=float((st.start_time.max() - st.start_time.min()) / 60),
        hit_whisker=float(rate.get("whisker_trial", np.nan)), fa=float(rate.get("no_stim_trial", np.nan)),
        hit_auditory=float(rate.get("auditory_trial", np.nan)),
        hit_minus_fa=float(rate.get("whisker_trial", np.nan) - rate.get("no_stim_trial", np.nan)),
        first_whisker_hit=first + 1,
        gap_max=float(np.nanmax(gap)), gap_mean=float(np.nanmean(gap)), gap_last20=float(np.nanmean(gap[-20:])),
        gap_first50=float(np.nanmean(gap[:50])),
        p_whisker_max=float(curves.p_mean.max()),
        run_above=longest_run(above), frac_above=float(above.mean()),
        dprime=float(dprime), dprime_curve_max=float(np.nanmax(dp_t)), dprime_curve_mean=float(np.nanmean(dp_t)),
        dprime_curve_last20=float(np.nanmean(dp_t[-20:])), dprime_curve_first50=float(np.nanmean(dp_t[:50])),
        dprime_low_max=float(np.nanmax(dp_low)),
        **{f"run_dprime_ge{k}": longest_run(dp_t >= k) for k in (0.5, 1.0, 1.5)},
        run_below=longest_run(below[first:]), frac_below=float(below[first:].mean()) if first < len(y) else 0.0,
        run_not_above=longest_run(~above[first:]),
        repo_mouse_cat=cat, repo_learning_trial=lt, high_state=bool((curves.state == "high").any()),
    )


def auc(bad, good):
    """P(random learner > random bad mouse) on the measure (0.5 = no separation; < 0.5 = bad mice higher)."""
    u = stats.mannwhitneyu(good, bad).statistic
    return u / (len(good) * len(bad))


def main():
    table = shared_behaviour.load()
    d0 = fig1.day0_table(table)
    with Pool(16) as pool:
        rows = pool.map(mouse_measures, [st for _, st in d0.groupby("session_id")])
    df = pd.DataFrame(rows)
    d1 = table[(table.behavior == "whisker") & (table.day == 1) & (table.trial_type == "whisker_trial")
               & (table.early_lick == 0)]
    df["hit_whisker_day1"] = df.mouse_id.map(d1.groupby("mouse_id").lick_flag.mean())
    df["bad"] = df.learning_category == "bad"
    out = FIGURES_ROOT / "_explore" / f"learner_criterion_{date.today()}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_mouse.csv", index=False)

    measures = ["dprime", "dprime_curve_max", "dprime_curve_mean", "dprime_curve_last20", "dprime_curve_first50",
                "dprime_low_max", "run_dprime_ge0.5", "run_dprime_ge1.0", "run_dprime_ge1.5", "n_whisker", "n_no_stim", "duration_min", "hit_whisker", "fa", "hit_auditory", "hit_minus_fa",
                "first_whisker_hit", "gap_max", "gap_mean", "gap_last20", "gap_first50", "p_whisker_max", "run_above",
                "frac_above", "run_below", "frac_below", "run_not_above", "hit_whisker_day1"]
    comp = []
    for c in ["R+", "R-"]:
        d = df[df.cohort == c]
        for m in measures:
            b, g = d[d.bad][m].dropna(), d[~d.bad][m].dropna()
            if len(b) < 2:
                comp.append(dict(cohort=c, measure=m, n_bad=len(b), n_learner=len(g), bad_values=list(np.round(b, 3)),
                                 learner_median=g.median()))
                continue
            comp.append(dict(cohort=c, measure=m, n_bad=len(b), n_learner=len(g), bad_median=b.median(),
                             learner_median=g.median(), bad_min=b.min(), bad_max=b.max(), learner_min=g.min(),
                             learner_max=g.max(), mannwhitney_p=stats.mannwhitneyu(b, g).pvalue,
                             welch_p=stats.ttest_ind(b, g, equal_var=False).pvalue, auc_learner_gt_bad=auc(b, g)))
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "bad_vs_learners.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(comp.drop(columns=[c for c in ("bad_values",) if c in comp]).round(3).to_string())

    # repo rule and high state vs the sheet
    print(pd.crosstab([df.cohort, df.learning_category], df.repo_mouse_cat))
    print(pd.crosstab([df.cohort, df.learning_category], df.high_state))

    # threshold sweeps: agreement with the sheet (learner = good/moderate) per cohort
    sweeps = []
    for m in ["dprime", "dprime_curve_max", "dprime_curve_mean", "dprime_curve_last20", "run_dprime_ge0.5",
              "run_dprime_ge1.0", "run_dprime_ge1.5", "dprime_low_max", "run_above", "gap_max", "gap_mean", "gap_last20", "hit_minus_fa", "frac_above", "run_not_above",
              "hit_whisker"]:
        for c in ["R+", "R-"]:
            d = df[df.cohort == c].dropna(subset=[m])
            for thr in np.unique(np.round(d[m], 3)):
                for sign in (1, -1):  # learner if measure >= thr (1) or <= thr (-1)
                    pred = d[m] >= thr if sign == 1 else d[m] <= thr
                    sweeps.append(dict(measure=m, cohort=c, threshold=thr, learner_if=">=" if sign == 1 else "<=",
                                       agreement=float((pred == ~d.bad).mean()),
                                       bad_called_learner=int((pred & d.bad).sum()),
                                       learner_called_bad=int((~pred & ~d.bad).sum())))
    sweeps = pd.DataFrame(sweeps)
    sweeps.to_csv(out / "threshold_sweeps.csv", index=False)
    best = sweeps.sort_values("agreement", ascending=False).groupby(["cohort", "measure"]).head(1)
    print(best.sort_values(["cohort", "agreement"], ascending=[True, False]).to_string())

    # figure: bad vs learners on the main candidate measures
    show = ["run_above", "gap_max", "gap_mean", "hit_minus_fa", "hit_whisker", "fa", "first_whisker_hit", "n_whisker",
            "hit_whisker_day1"]
    fig, axes = plt.subplots(1, len(show), figsize=(2.0 * len(show), 2.6))
    rng = np.random.default_rng(0)
    for a, m in zip(axes, show):
        for i, c in enumerate(["R+", "R-"]):
            for j, (lab, sub) in enumerate((("learner", df[(df.cohort == c) & ~df.bad]), ("bad", df[(df.cohort == c) & df.bad]))):
                x = i * 2.5 + j + rng.uniform(-0.15, 0.15, len(sub))
                a.scatter(x, sub[m], s=8, color=COLORS[c] if lab == "learner" else "white", edgecolor=COLORS[c], lw=0.7)
        a.set_xticks([0, 1, 2.5, 3.5], ["R+ L", "R+ bad", "R- L", "R- bad"], fontsize=6, rotation=45)
        a.set_title(m, fontsize=7)
        a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "bad_vs_learners.png", dpi=200)
    print(out)


if __name__ == "__main__":
    main()
