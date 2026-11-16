"""Exploration (user 2026-11-13): does the number of whisker trials before the first whisker hit on day 0 correlate with
other markers of performance?

Per mouse (whisker day 0; early-lick trials dropped; shared behaviour table, so perf 6, passive and the warm-up block are
already cut): n_before = whisker trials before the first whisker hit (0 = the first whisker trial is a hit). Mice with no
whisker hit are censored (kept in the distribution plot, left out of the correlations).
Markers (many, on purpose), grouped:
  day 0 lick rates   whisker P(lick) overall (*), after the first hit, last 50 whisker trials; FA and auditory P(lick)
                     overall and before the first hit; whisker minus FA after the first hit
  day 0 curves       exact-curve learning trial (*), first high-state trial (*), fraction of whisker trials in the high state,
                     peak whisker curve, peak whisker minus chance curve, lag-0 r(whisker, chance curves)
  day 0 timing       median whisker and auditory RT, RT of the first hit, minutes to the first hit (*), session length
                     (whisker trials)
  chance             expected trials to a first hit if the mouse licked at its pre-hit FA rate ((1 - fa) / fa)
  other days         whisker P(lick) day +1 / +2; auditory and FA P(lick) day -1 / -2; auditory RT day -1;
                     pretraining days and trials
  (*) contains the pre-hit trials, so partly mechanical: read the clean versions next to them.
Stats: Spearman rho per cohort and pooled (mice), Fisher-z 95% CI, uncorrected p (user 2026-11-13: no multiple-comparison
correction); markers listed by decreasing pooled rho; R+ vs R-
on n_before Mann-Whitney + Welch; learners vs non-learners Mann-Whitney per cohort.
Output: Figures/_explore/trials_to_first_hit_<date>/ (overview, scatter grid, per-mouse table, stats csv).
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/SuppFigure1_behaviour/explore_trials_to_first_hit.py
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
from common import FIGURES_ROOT, two_group_tests  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig1 = _load("fig1_compute", FIG_DIR.parent / "Figure1_behaviour" / "compute.py")
fig1p = _load("fig1_plot", FIG_DIR.parent / "Figure1_behaviour" / "plot.py")
COLORS, COHORTS = fig1p.COLORS, ["R+", "R-"]
OUT = FIGURES_ROOT / "_explore" / f"trials_to_first_hit_{datetime.date.today()}"

MARKERS = {  # column: (label, group, partly mechanical)
    "wh_plick": ("W P(lick) d0, all", "Day 0 rates", True),
    "wh_plick_after": ("W P(lick) d0, after 1st hit", "Day 0 rates", False),
    "wh_plick_last50": ("W P(lick) d0, last 50 W", "Day 0 rates", False),
    "fa_plick": ("FA P(lick) d0", "Day 0 rates", False),
    "fa_plick_before": ("FA P(lick) d0, before 1st hit", "Day 0 rates", False),
    "aud_plick": ("A P(lick) d0", "Day 0 rates", False),
    "aud_plick_before": ("A P(lick) d0, before 1st hit", "Day 0 rates", False),
    "wh_minus_fa_after": ("W − FA d0, after 1st hit", "Day 0 rates", False),
    "learning_trial": ("Learning trial (exact curve)", "Day 0 curves", True),
    "first_high_trial": ("First high-state W trial", "Day 0 curves", True),
    "frac_high": ("Fraction W trials in high state", "Day 0 curves", False),
    "peak_curve": ("Peak W curve", "Day 0 curves", False),
    "peak_w_minus_chance": ("Peak W − chance curve", "Day 0 curves", False),
    "xcorr_lag0": ("r(W, chance curves), lag 0", "Day 0 curves", False),
    "rt_wh": ("W RT d0 (s)", "Day 0 timing", False),
    "rt_aud": ("A RT d0 (s)", "Day 0 timing", False),
    "rt_first_hit": ("RT of 1st W hit (s)", "Day 0 timing", False),
    "min_to_first_hit": ("Minutes to 1st W hit", "Day 0 timing", True),
    "n_whisker_trials": ("W trials in session", "Day 0 timing", False),
    "chance_expected": ("Chance trials to 1st hit (FA)", "Chance", False),
    "wh_day1": ("W P(lick) day +1", "Other days", False),
    "wh_day2": ("W P(lick) day +2", "Other days", False),
    "aud_daym1": ("A P(lick) day −1", "Other days", False),
    "fa_daym1": ("FA P(lick) day −1", "Other days", False),
    "aud_daym2": ("A P(lick) day −2", "Other days", False),
    "fa_daym2": ("FA P(lick) day −2", "Other days", False),
    "rt_aud_daym1": ("A RT day −1 (s)", "Other days", False),
    "n_pretraining_days": ("Pretraining days", "Other days", False),
    "n_pretraining_trials": ("Pretraining trials", "Other days", False),
}


def _rt(d):
    v = d[(d.lick_flag == 1) & d.computed_reaction_time.between(0, 1)].computed_reaction_time
    return v.median() if len(v) else np.nan


def day0_session(st, ctx):
    st = st[st.early_lick == 0].sort_values("start_time").reset_index(drop=True)
    w = st[st.trial_type == "whisker_trial"].reset_index(drop=True)
    hits = np.flatnonzero(w.lick_flag.to_numpy() == 1)
    r = {"mouse_id": st.mouse_id.iloc[0], "session_id": st.session_id.iloc[0], "cohort": st.cohort.iloc[0],
         "learning_category": st.learning_category.iloc[0], "n_whisker_trials": len(w),
         "wh_plick": w.lick_flag.mean(), "fa_plick": st[st.trial_type == "no_stim_trial"].lick_flag.mean(),
         "aud_plick": st[st.trial_type == "auditory_trial"].lick_flag.mean(),
         "rt_wh": _rt(w), "rt_aud": _rt(st[st.trial_type == "auditory_trial"]),
         "wh_plick_last50": w.lick_flag.iloc[-50:].mean(), "censored": len(hits) == 0}
    if len(hits):
        k = int(hits[0])
        t_hit = w.start_time.iloc[k]
        before = st[st.start_time < t_hit]
        fa_b = before[before.trial_type == "no_stim_trial"].lick_flag
        aft = w.iloc[k + 1:]
        fa_after = st[(st.start_time > t_hit) & (st.trial_type == "no_stim_trial")].lick_flag.mean()
        r.update(n_before=k, wh_plick_after=aft.lick_flag.mean() if len(aft) else np.nan,
                 fa_plick_before=fa_b.mean() if len(fa_b) else np.nan,
                 aud_plick_before=before[before.trial_type == "auditory_trial"].lick_flag.mean(),
                 wh_minus_fa_after=(aft.lick_flag.mean() - fa_after) if len(aft) else np.nan,
                 rt_first_hit=w.computed_reaction_time.iloc[k] if 0 < w.computed_reaction_time.iloc[k] <= 1 else np.nan,
                 min_to_first_hit=(t_hit - st.start_time.iloc[0]) / 60)
        fa_rate = fa_b.mean() if len(fa_b) >= 3 else np.nan
        r["chance_expected"] = (1 - fa_rate) / fa_rate if fa_rate and fa_rate > 0 else np.nan
    else:
        r["n_before"] = len(w)
    curves, learning_trial, _ = fig1.session_curves(st, ctx)
    high = np.flatnonzero(curves.state.to_numpy() == "high")
    r.update(learning_trial=learning_trial if learning_trial is not None else np.nan,
             first_high_trial=high[0] + 1 if len(high) else np.nan, frac_high=(curves.state == "high").mean(),
             peak_curve=curves.p_mean.max(), peak_w_minus_chance=(curves.p_mean - curves.p_chance).max(),
             xcorr_lag0=fig1.lagged_corr(curves.p_mean.to_numpy(), curves.p_chance.to_numpy(), 0))
    return r


def mouse_table(table):
    ctx = {}
    t = table[table.early_lick == 0]
    d0 = fig1.day0_table(t)
    rows = [day0_session(st, ctx) for _, st in d0.groupby("session_id")]
    s = pd.DataFrame(rows)
    num = s.select_dtypes("number").columns
    m = s.groupby(["mouse_id", "cohort", "learning_category"])[list(num)].mean()
    m["censored"] = s.groupby(["mouse_id", "cohort", "learning_category"]).censored.all()
    m = m.reset_index()
    p = (t.groupby(["mouse_id", "day", "session_id", "trial_type"]).lick_flag.mean()
         .groupby(["mouse_id", "day", "trial_type"]).mean().unstack(["day", "trial_type"]))
    other = pd.DataFrame({k: p.get((d, tt)) for k, d, tt in [
        ("wh_day1", 1, "whisker_trial"), ("wh_day2", 2, "whisker_trial"), ("aud_daym1", -1, "auditory_trial"),
        ("fa_daym1", -1, "no_stim_trial"), ("aud_daym2", -2, "auditory_trial"), ("fa_daym2", -2, "no_stim_trial")]})
    rt_m1 = t[(t.day == -1) & (t.trial_type == "auditory_trial")].groupby("mouse_id").apply(_rt).rename("rt_aud_daym1")
    pre, _ = fig1.compute_pretraining({"table": t})
    pre = pre[["mouse_id", "n_pretraining_trials", "n_auditory_days"]].rename(columns={"n_auditory_days": "n_pretraining_days"})
    m = m.merge(other.reset_index(), on="mouse_id", how="left").merge(rt_m1.reset_index(), on="mouse_id", how="left")
    m = m.merge(pre, on="mouse_id", how="left")
    m["learner"] = m.learning_category.isin(["good", "moderate"])
    return m


def corr_table(m, population):
    rows = []
    sel = m[~m.censored]
    if population == "learners":
        sel = sel[sel.learner]
    for grp in ("R+", "R-", "pooled"):
        g = sel if grp == "pooled" else sel[sel.cohort == grp]
        for v, (lab, group, mech) in MARKERS.items():
            d = g[["n_before", v]].dropna()
            if len(d) < 6:
                continue
            rho, p = stats.spearmanr(d.n_before, d[v])
            z, se = np.arctanh(np.clip(rho, -0.999, 0.999)), 1.06 / np.sqrt(len(d) - 3)
            rows.append(dict(population=population, cohort=grp, marker=v, label=lab, group=group, mechanical=mech,
                             n=len(d), rho=rho, p=p, ci_low=np.tanh(z - 1.96 * se), ci_high=np.tanh(z + 1.96 * se)))
    return pd.DataFrame(rows)  # uncorrected p (user 2026-11-13)


def order(c, population):
    """Markers by decreasing pooled rho (user 2026-11-13)."""
    g = c[(c.population == population) & (c.cohort == "pooled")].sort_values("rho", ascending=False)
    return list(g.marker) + [v for v in MARKERS if v not in set(g.marker)]


def _stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def plot_overview(m, c, stats_rows, path):
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 6.4))
    ax0 = fig.add_axes([0.07, 0.62, 0.17, 0.32])
    for k, coh in enumerate(COHORTS):
        g = m[m.cohort == coh]
        rng = np.random.default_rng(k)
        for cens, mk in ((False, "o"), (True, "x")):
            gg = g[g.censored == cens]
            ax0.scatter(k + rng.uniform(-0.15, 0.15, len(gg)), gg.n_before + 1, s=6, marker=mk, lw=0.6 if cens else 0,
                        color=COLORS[coh] if not cens else "0.3", alpha=0.85, zorder=2)
        g_un = g[~g.censored]
        ax0.errorbar(k + 0.3, g_un.n_before.median() + 1, 0, color="k", marker="_", ms=10, mew=1)
    ax0.set_yscale("log")
    ax0.set_xticks([0, 1], COHORTS)
    ax0.set_xlim(-0.5, 1.6)
    ax0.set_ylabel("W trials before 1st W hit + 1")
    t = stats_rows["rplus_vs_rminus"]
    ax0.set_title(f"Day 0, all mice\nMW p={t['mannwhitney_p']:.2g}, Welch p={t['welch_p']:.2g}\n× = no hit (censored)", fontsize=5.5)
    # learners vs non-learners
    ax1 = fig.add_axes([0.07, 0.12, 0.17, 0.36])
    for k, coh in enumerate(COHORTS):
        for j, lr in enumerate((True, False)):
            g = m[(m.cohort == coh) & (m.learner == lr)]
            x = k * 2.4 + j
            rng = np.random.default_rng(10 * k + j)
            ax1.scatter(x + rng.uniform(-0.15, 0.15, len(g)), g.n_before + 1, s=5, lw=0,
                        color=COLORS[coh] if lr else fig1p.light(COLORS[coh], 0.5))
            ax1.plot([x - 0.25, x + 0.25], [g.n_before.median() + 1] * 2, color="k", lw=1)
        p = stats_rows[f"learner_vs_non_{coh}"]["mannwhitney_p"]
        ax1.text(k * 2.4 + 0.5, 1.02, f"p={p:.2g}", transform=ax1.get_xaxis_transform(), ha="center", fontsize=5)
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1, 2.4, 3.4], ["R+ L", "R+ NL", "R− L", "R− NL"], fontsize=5)
    ax1.set_ylabel("W trials before 1st W hit + 1")
    ax1.set_title("Learners (L) vs non-learners (NL)\n(censored mice included)", fontsize=5.5, pad=8)
    # forest plots: all mice and learners
    for i, pop in enumerate(("all", "learners")):
        ax = fig.add_axes([0.44 + i * 0.37, 0.06, 0.17, 0.88])
        cc = c[c.population == pop]
        labels = order(c, pop)
        for k, coh in enumerate(("R+", "R-", "pooled")):
            g = cc[cc.cohort == coh].set_index("marker")
            col = COLORS.get(coh, "0.3")
            for y, v in enumerate(labels):
                if v not in g.index:
                    continue
                r = g.loc[v]
                yy = y + (k - 1) * 0.25
                ax.plot([r.ci_low, r.ci_high], [yy, yy], color=col, lw=0.6)
                ax.plot(r.rho, yy, "o", ms=2.2, color=col, mec="none" if r.p < 0.05 else col,
                        mfc=col if r.p < 0.05 else "white", mew=0.5)
                if r.p < 0.05:
                    ax.text(1.02, yy, _stars(r.p), color=col, fontsize=4.5, va="center",
                            transform=ax.get_yaxis_transform())
        ax.axvline(0, color="0.6", lw=0.5, ls=":")
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(range(len(labels)),
                      [MARKERS[v][0] + (" (*)" if MARKERS[v][2] else "") for v in labels], fontsize=5)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ with W trials before 1st hit")
        n = {coh: int(m[(~m.censored) & (m.cohort == coh) & (m.learner if pop == "learners" else True)].shape[0])
             for coh in COHORTS}
        ax.set_title(f"{'All mice' if pop == 'all' else 'Learners'} (R+ n={n['R+']}, R− n={n['R-']})", fontsize=6)
    fig.legend(handles=[plt.Line2D([], [], color=COLORS["R+"], marker="o", ms=3, label="R+"),
                        plt.Line2D([], [], color=COLORS["R-"], marker="o", ms=3, label="R−"),
                        plt.Line2D([], [], color="0.3", marker="o", ms=3, label="pooled"),
                        plt.Line2D([], [], color="0.3", marker="o", ms=3, mfc="white", lw=0, label="p ≥ 0.05")],
               loc="upper right", ncol=4, fontsize=5, frameon=False, bbox_to_anchor=(0.99, 1.0))
    fig.text(0.3, 0.005, "Markers by decreasing pooled ρ. (*) contains the pre-hit trials, partly mechanical. Lines, Fisher-z "
             "95% CI; stars, uncorrected Spearman p. Censored mice (no hit) excluded from correlations.", fontsize=4.5)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_scatter(m, c, path):
    labels = order(c, "all")
    nc = 6
    nr = int(np.ceil(len(labels) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.2, 1.25 * nr), constrained_layout=True)
    sel = m[~m.censored]
    cc = c[c.population == "all"].set_index(["cohort", "marker"])
    for a, v in zip(axs.flat, labels):
        for coh in COHORTS:
            g = sel[sel.cohort == coh]
            a.scatter(g.n_before + 1, g[v], s=3, lw=0, color=COLORS[coh], alpha=0.8)
        a.set_xscale("log")
        a.set_xlabel("W trials to 1st hit + 1", fontsize=5)
        a.set_ylabel(MARKERS[v][0], fontsize=5)
        a.tick_params(labelsize=4.5)
        txt = []
        for coh in COHORTS:
            if (coh, v) in cc.index:
                r = cc.loc[(coh, v)]
                txt.append((f"{coh} ρ={r.rho:.2f}{_stars(r.p)}", COLORS[coh]))
        for j, (s, col) in enumerate(txt):
            a.text(0.02 + 0.5 * j, 1.02, s, transform=a.transAxes, fontsize=4.3, color=col)
    for a in list(axs.flat)[len(labels):]:
        a.axis("off")
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = mouse_table(shared_behaviour.load())
    m.to_csv(OUT / "per_mouse.csv", index=False)
    c = pd.concat([corr_table(m, "all"), corr_table(m, "learners")], ignore_index=True)
    c.to_csv(OUT / "correlations.csv", index=False)
    st = {"rplus_vs_rminus": two_group_tests(m[(m.cohort == "R+") & ~m.censored].n_before,
                                             m[(m.cohort == "R-") & ~m.censored].n_before)}
    for coh in COHORTS:
        g = m[m.cohort == coh]
        st[f"learner_vs_non_{coh}"] = {"mannwhitney_p": float(stats.mannwhitneyu(g[g.learner].n_before,
                                                                                g[~g.learner].n_before).pvalue)}
    pd.json_normalize(st).to_csv(OUT / "group_tests.csv", index=False)
    plot_overview(m, c, st, OUT / "overview")
    plot_scatter(m, c, OUT / "scatter_all_mice")
    print(m.groupby("cohort").agg(mice=("mouse_id", "size"), censored=("censored", "sum"),
                                  median_n_before=("n_before", "median")))
    print(st)
    w = c.pivot_table(index=["population", "label"], columns="cohort", values=["rho", "p"]).round(3)
    for pop in ("all", "learners"):
        t = w.loc[pop].sort_values(("rho", "pooled"), ascending=False)
        t.columns = [f"{a}_{b}" for a, b in t.columns]
        print(pop); print(t[["rho_pooled", "p_pooled", "rho_R+", "p_R+", "rho_R-", "p_R-"]].to_string())
    print(OUT)


if __name__ == "__main__":
    main()
