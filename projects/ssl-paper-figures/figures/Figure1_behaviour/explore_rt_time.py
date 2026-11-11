"""Exploration (user 2026-11-10): "For the reaction time and learning, give alternate visualization and do stats within
cohort with respect to time." Panel k alternatives, whisker day 0, learners (Figure 1 compute.day0_table; early licks
dropped; reaction time = computed_reaction_time, 0-1 s, in ms), whisker hits (row 1) and auditory hits (row 2, control).

Views (columns):
  A  per-mouse median RT in bins of 5 hits (first 30), thin lines = mice, thick = cohort mean +- s.e.m. (as panel k + mice);
  B  within-mouse change: RT minus the mouse's own median over its first 30 hits, per hit, 5-hit running mean per mouse,
     then mean +- s.e.m. over mice (removes between-mouse offsets, so only change over time remains);
  C  RT over session time (0 = first, 1 = last whisker trial; 5 bins, all hits of the session), per-mouse median,
     mean +- s.e.m.;
  D  per-mouse Spearman rho(RT, hit number) over the first 30 hits; grey mice, mean +- s.e.m.; stars, Wilcoxon vs 0;
  E  RT against the whisker learning curve at the trial (exact curve, as in Figure 1 e-f; 4 bins of p(lick) 0-0.25, ...,
     0.75-1), per-mouse median, mean +- s.e.m.
Within-cohort tests over time (printed on the panels and saved in the JSON):
  - rho(RT, hit number) per mouse vs 0, Wilcoxon signed-rank (and one-sample t); R+ vs R- Mann-Whitney + Welch;
  - rho(RT, session time) per mouse, all hits of the session, same tests;
  - Friedman test across the six hit bins (mice with all six), and Wilcoxon hits 1-5 vs 26-30 (paired);
  - linear mixed model RT ~ hit number (random intercept and slope per mouse, statsmodels MixedLM, REML) per cohort:
    slope in ms per hit and its p; pooled model with a cohort x hit-number interaction.
Output: Figures/_explore/rt_time_<date>.{pdf,png}, _hits.csv (one row per licked trial) and _stats.json.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_time.py
"""

import datetime
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, two_group_tests  # noqa: E402
from paperstyle import MM, light, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
N_HITS, BIN, N_TIME, CURVE_BINS = 30, 5, 5, [0, 0.25, 0.5, 0.75, 1.0001]
TT = [("whisker_trial", "Whisker hits"), ("auditory_trial", "Auditory hits (control)")]


def hits_table():
    from compute import day0_table, session_curves
    ctx = {}
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    rows = []
    for sid, st in d0.groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        curves, _, _ = session_curves(st, ctx)
        isw = st.trial_type.to_numpy() == "whisker_trial"
        widx = np.cumsum(isw) - 1
        curve = np.where(widx >= 0, curves.p_mean.to_numpy()[np.clip(widx, 0, len(curves) - 1)], np.nan)
        tw = st.start_time.to_numpy()[isw]
        stime = (st.start_time.to_numpy() - tw[0]) / max(tw[-1] - tw[0], 1e-9)
        rt = (st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000).to_numpy()
        for tt, _ in TT:
            sel = np.flatnonzero((st.trial_type == tt).to_numpy() & (st.lick_flag == 1).to_numpy())
            rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                      "trial_type": tt, "hit_number": np.arange(1, len(sel) + 1), "rt_ms": rt[sel],
                                      "session_time": stime[sel], "whisker_curve": curve[sel]}))
    return pd.concat(rows, ignore_index=True)


def per_mouse_rho(h, x):
    out = {}
    for (m, c), g in h.dropna(subset=["rt_ms", x]).groupby(["mouse_id", "cohort"]):
        if len(g) >= 10 and g[x].nunique() > 1:
            out[(m, c)] = stats.spearmanr(g[x], g.rt_ms).statistic
    return pd.Series(out, name="rho").rename_axis(["mouse_id", "cohort"]).reset_index()


def mixed_slope(h, interaction=False):
    import statsmodels.formula.api as smf
    d = h.dropna(subset=["rt_ms"]).assign(hit=lambda d_: d_.hit_number - 1, rplus=lambda d_: (d_.cohort == "R+") * 1)
    form = "rt_ms ~ hit * rplus" if interaction else "rt_ms ~ hit"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = smf.mixedlm(form, d, groups=d.mouse_id, re_formula="~hit").fit(reml=True)
        except Exception:  # noqa: BLE001 - singular random slopes: fall back to a random intercept
            r = smf.mixedlm(form, d, groups=d.mouse_id).fit(reml=True)
    key = "hit:rplus" if interaction else "hit"
    return {"slope_ms_per_hit": float(r.params[key]), "p": float(r.pvalues[key]), "n_mice": int(d.mouse_id.nunique()),
            "n_hits": int(len(d))}


def within_stats(h30, hall):
    res = {}
    for tt, _ in TT:
        a, b = h30[h30.trial_type == tt], hall[hall.trial_type == tt]
        r = {}
        for name, d, x in (("rho_hit_number", a, "hit_number"), ("rho_session_time", b, "session_time")):
            rho = per_mouse_rho(d, x)
            r[name] = {**{c: {"mean": float(rho[rho.cohort == c].rho.mean()), "n": int((rho.cohort == c).sum()),
                              "wilcoxon_p": float(stats.wilcoxon(rho[rho.cohort == c].rho).pvalue),
                              "t_p": float(stats.ttest_1samp(rho[rho.cohort == c].rho, 0).pvalue)} for c in ("R+", "R-")},
                       "rplus_vs_rminus": two_group_tests(rho[rho.cohort == "R+"].rho, rho[rho.cohort == "R-"].rho)}
        pm = a.assign(bin=(a.hit_number - 1) // BIN).groupby(["mouse_id", "cohort", "bin"]).rt_ms.median().unstack()
        r["friedman_bins"], r["first_vs_last_bin"], r["mixed_model"] = {}, {}, {}
        for c in ("R+", "R-"):
            w = pm.xs(c, level="cohort").dropna()
            r["friedman_bins"][c] = {"n": len(w), "p": float(stats.friedmanchisquare(*w.T.to_numpy()).pvalue)} \
                if len(w) > 2 else {"n": len(w), "p": None}
            fl = pm.xs(c, level="cohort")[[0, N_HITS // BIN - 1]].dropna()
            r["first_vs_last_bin"][c] = {"n": len(fl), "median_first": float(fl[0].median()),
                                         "median_last": float(fl[N_HITS // BIN - 1].median()),
                                         "wilcoxon_p": float(stats.wilcoxon(fl[0], fl[N_HITS // BIN - 1]).pvalue)}
            r["mixed_model"][c] = mixed_slope(a[a.cohort == c])
        r["mixed_model"]["cohort_x_hit"] = mixed_slope(a, interaction=True)
        res[tt] = r
    return res


def pstr(p):
    return "n.s." if p is None or p >= 0.05 else ("***" if p < 0.001 else "**" if p < 0.01 else "*")


def plot(h30, hall, st, stem):
    from plot import COHORTS, group_color
    set_nature_rc()
    fig, axs = plt.subplots(2, 5, figsize=(183 * MM, 85 * MM))
    for r, (tt, title) in enumerate(TT):
        a30, aall, s = h30[h30.trial_type == tt], hall[hall.trial_type == tt], st[tt]
        txt = {c: [] for c in COHORTS}
        for c in COHORTS:
            col, d = group_color(tt, c), a30[a30.cohort == c]
            # A: bins, mice thin
            pm = d.assign(bin=(d.hit_number - 1) // BIN * BIN + 3).groupby(["mouse_id", "bin"]).rt_ms.median().reset_index()
            for _, g in pm.groupby("mouse_id"):
                axs[r, 0].plot(g.bin, g.rt_ms, color=light(col, 0.6), lw=0.25, zorder=1)
            g = pm.groupby("bin").rt_ms
            axs[r, 0].errorbar(g.mean().index, g.mean(), g.sem(), color=col, lw=1, marker="o", ms=2, capsize=0, zorder=3)
            # B: within-mouse change, running mean
            dd = d.dropna(subset=["rt_ms"]).copy()
            dd["delta"] = dd.rt_ms - dd.groupby("mouse_id").rt_ms.transform("median")
            dd["smooth"] = dd.groupby("mouse_id").delta.transform(lambda v: v.rolling(5, center=True, min_periods=3).mean())
            g = dd.groupby("hit_number").smooth
            m, e = g.mean(), g.sem()
            axs[r, 1].fill_between(m.index, m - e, m + e, color=light(col, 0.7), lw=0)
            axs[r, 1].plot(m.index, m, color=col, lw=0.8)
            # C: session time
            da = aall[aall.cohort == c]
            pt = da.assign(tb=np.clip((da.session_time * N_TIME).astype(int), 0, N_TIME - 1)).groupby(
                ["mouse_id", "tb"]).rt_ms.median().reset_index()
            g = pt.groupby("tb").rt_ms
            axs[r, 2].errorbar((g.mean().index + 0.5) / N_TIME, g.mean(), g.sem(), color=col, lw=0.8, marker="o", ms=2,
                               capsize=0)
            # D: per-mouse rho vs hit number
            rho = per_mouse_rho(d, "hit_number").rho
            x = COHORTS.index(c)
            axs[r, 3].scatter(x - 0.15 + np.random.default_rng(x).uniform(-0.06, 0.06, len(rho)), rho, s=1.5,
                              color="0.72", lw=0)
            axs[r, 3].errorbar(x + 0.1, rho.mean(), rho.sem(), color=col, marker="o", ms=2.4, capsize=0, elinewidth=0.7)
            p = s["rho_hit_number"][c]["wilcoxon_p"]
            axs[r, 3].text(x, 0.92, pstr(p), ha="center", fontsize=5, transform=axs[r, 3].get_xaxis_transform())
            # E: RT vs whisker curve
            dc = da.dropna(subset=["whisker_curve"])
            pc = dc.assign(cb=pd.cut(dc.whisker_curve, CURVE_BINS, labels=False, right=False)).groupby(
                ["mouse_id", "cb"]).rt_ms.median().reset_index()
            g = pc.groupby("cb").rt_ms
            axs[r, 4].errorbar(g.mean().index * 0.25 + 0.125 + (x - 0.5) * 0.03, g.mean(), g.sem(), color=col, lw=0.8,
                               marker="o", ms=2, capsize=0)
            mm = s["mixed_model"][c]
            txt[c] = (f"{c}: LMM {mm['slope_ms_per_hit']:+.1f} ms/hit {pstr(mm['p'])}; "
                      f"Friedman {pstr(s['friedman_bins'][c]['p'])}")
            axs[r, 1].text(0.02, 0.97 - 0.09 * x, txt[c], transform=axs[r, 1].transAxes, fontsize=4, color=col, va="top")
            rs = s["rho_session_time"][c]
            axs[r, 2].text(0.02, 0.97 - 0.09 * x, f"{c}: ρ {rs['mean']:+.2f} {pstr(rs['wilcoxon_p'])}",
                           transform=axs[r, 2].transAxes, fontsize=4.5, color=col, va="top")
        axs[r, 1].text(0.02, 0.79, f"R+ vs R- slope: {pstr(s['mixed_model']['cohort_x_hit']['p'])}",
                       transform=axs[r, 1].transAxes, fontsize=4, va="top")
        b = s["rho_hit_number"]["rplus_vs_rminus"]["mannwhitney_p"]
        axs[r, 3].plot([0, 1], [0.84, 0.84], color="0.2", lw=0.4, transform=axs[r, 3].get_xaxis_transform())
        axs[r, 3].text(0.5, 0.85, pstr(b), ha="center", fontsize=5, transform=axs[r, 3].get_xaxis_transform())
        axs[r, 0].set_xticks([3, 8, 13, 18, 23, 28], ["1", "6", "11", "16", "21", "26"])
        axs[r, 0].set_xlabel("Hit number (bin start)")
        axs[r, 0].set_ylabel(f"{title}\nReaction time (ms)")
        axs[r, 1].axhline(0, color="0.6", lw=0.4, ls=":")
        axs[r, 1].set_xlabel("Hit number")
        axs[r, 1].set_ylabel("Δ RT from own median (ms)")
        axs[r, 2].set_xlabel("Session time (norm.)")
        axs[r, 2].set_xlim(0, 1)
        axs[r, 3].axhline(0, color="0.6", lw=0.4, ls=":")
        axs[r, 3].set_xticks([0, 1], COHORTS)
        axs[r, 3].set_xlim(-0.6, 1.6)
        axs[r, 3].set_ylabel("ρ(RT, hit number)")
        axs[r, 4].set_xlabel("Whisker curve p(lick)")
        axs[r, 4].set_xticks([0, 0.5, 1])
        if r == 0:
            for a, t in zip(axs[0], ["A  bins + mice", "B  within-mouse change", "C  session time", "D  per-mouse trend",
                                     "E  vs learning curve"]):
                a.set_title(t, fontsize=6, loc="left")
    fig.suptitle("Reaction time over time within cohort, whisker day 0, learners (R+ green, R- magenta; auditory blue); "
                 "mean ± s.e.m. over mice; stars: within-cohort test vs no change", fontsize=5.5)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


def main():
    hall = hits_table()
    h30 = hall[hall.hit_number <= N_HITS]
    st = within_stats(h30, hall)
    stem = OUT / f"rt_time_{datetime.date.today()}"
    hall.to_csv(f"{stem}_hits.csv", index=False)
    json.dump(st, open(f"{stem}_stats.json", "w"), indent=1, default=float)
    plot(h30, hall, st, stem)
    print(json.dumps(st, indent=1, default=float))
    print(stem)


if __name__ == "__main__":
    main()
