"""Checks for the Figure 1 / Supp. Fig. 1 comments of 2026-11-22.

1. Fig. 1I (aligned to the first whisker hit): is the point at 0 a single trial; where do trials 0 of the false-alarm and
   auditory types fall relative to the first whisker hit; how fast do the cohorts diverge (trial-by-trial lick
   probability after the first hit, per-trial tests, and the 1-5 bin without trial 1).
2. Supp. Fig. 1G: why is the cross-correlation of the whisker and chance (FA) curves asymmetric? Per-mouse asymmetry index
   (mean r at negative lags minus positive lags) on the raw curves, on linearly detrended curves and on first differences;
   the same for the Spearman lag profiles of pre-stimulus DLC measures against the whisker curve (Supp. Fig. 1H).

Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/checks/003_fig1i_and_xcorr_asymmetry.py
Output: Figures/_explore/checks_2026-11-22/.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, two_group_tests  # noqa: E402

spec = importlib.util.spec_from_file_location("fig1", ROOT / "figures" / "Figure1_behaviour" / "compute.py")
fig1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig1)
OUT = FIGURES_ROOT / "_explore" / "checks_2026-11-22"
OUT.mkdir(parents=True, exist_ok=True)


def panel_i(ctx):
    data = fig1._aligned_lick_trials(ctx)
    hit = data[data.alignment == "first_whisker_hit"]
    w = hit[hit.trial_type == "whisker_trial"]
    print("\n=== Fig. 1I checks")
    print("bin edges", list(fig1.ALIGN_BIN_EDGES), "-> trial 0 is its own bin")
    print("whisker trial 0: n mice", w[w.trial_from_alignment == 0].mouse_id.nunique(),
          "rows per session", w[w.trial_from_alignment == 0].groupby("session_id").size().max(),
          "P(lick) =", w[w.trial_from_alignment == 0].p_lick.mean())
    print("whisker trials < 0 (before the first hit): P(lick) =", w[w.trial_from_alignment < 0].p_lick.mean(),
          "(all misses by definition)")
    # trial-by-trial after the first hit
    rows = []
    for tt, g in hit.groupby("trial_type"):
        for k in range(-3, 11):
            gk = g[g.trial_from_alignment == k]
            a, b = gk[gk.cohort == "R+"].p_lick, gk[gk.cohort == "R-"].p_lick
            if len(a) < 5 or len(b) < 5:
                continue
            ct = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
            rows.append(dict(trial_type=tt, trial=k, p_rplus=a.mean(), p_rminus=b.mean(), n_rplus=len(a),
                             n_rminus=len(b), fisher_p=stats.fisher_exact(ct)[1]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fig1i_trial_by_trial.csv", index=False)
    print(t.round(3).to_string(index=False))
    # bin 1-5 with and without trial 1, per mouse
    res = []
    for tt, g in hit.groupby("trial_type"):
        for name, sel in (("1-5", g.trial_from_alignment.between(1, 5)), ("2-5", g.trial_from_alignment.between(2, 5)),
                          ("1", g.trial_from_alignment == 1)):
            m = g[sel].groupby(["mouse_id", "cohort"]).p_lick.mean().reset_index()
            r = two_group_tests(m[m.cohort == "R+"].p_lick, m[m.cohort == "R-"].p_lick)
            res.append(dict(trial_type=tt, trials=name, mean_rplus=m[m.cohort == "R+"].p_lick.mean(),
                            mean_rminus=m[m.cohort == "R-"].p_lick.mean(), **r))
    r = pd.DataFrame(res)
    r.to_csv(OUT / "fig1i_bin1to5.csv", index=False)
    print(r.round(4).to_string(index=False))
    # where trial 0 of FA / auditory falls: delay (in trials of any type) after the first whisker hit
    d0 = shared_behaviour.load_untrimmed_day0()
    d0 = d0[d0.mouse_id.isin(set(hit.mouse_id)) & (d0.early_lick == 0)].sort_values("start_time")
    lag = []
    for sid, st in d0.groupby("session_id"):
        ws = st[(st.trial_type == "whisker_trial") & (st.lick_flag == 1)]
        if ws.empty:
            continue
        t_hit = ws.start_time.iloc[0]
        for tt in ("no_stim_trial", "auditory_trial"):
            nxt = st[(st.trial_type == tt) & (st.start_time >= t_hit)]
            if len(nxt):
                lag.append(dict(session_id=sid, trial_type=tt, after_hit=nxt.start_time.iloc[0] > t_hit,
                                trials_after=int(((st.start_time > t_hit) & (st.start_time <= nxt.start_time.iloc[0])).sum())))
    lag = pd.DataFrame(lag)
    print("trial 0 of FA / A after the first whisker hit:",
          lag.groupby("trial_type").agg(all_after=("after_hit", "all"), median_trials_after=("trials_after", "median")).to_dict())


def asym(r, lags):
    neg, pos = r[lags < 0], r[lags > 0]
    return np.nanmean(neg) - np.nanmean(pos)


def xcorr(ctx):
    print("\n=== Supp. Fig. 1G cross-correlation asymmetry (positive lag: whisker curve leads)")
    lags = fig1.LAGS
    rows, prof = [], []
    for sid, st in fig1.day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        curves, _, _ = fig1.session_curves(st, ctx)
        w, fa = curves.p_mean.to_numpy(), curves.p_chance.to_numpy()
        n = len(w)
        t = np.arange(n)
        variants = {"raw": (w, fa),
                    "detrended": (w - np.polyval(np.polyfit(t, w, 1), t), fa - np.polyval(np.polyfit(t, fa, 1), t)),
                    "diff": (np.diff(w), np.diff(fa))}
        trend = {"w_slope_per_100": np.polyfit(t / 100, w, 1)[0], "fa_slope_per_100": np.polyfit(t / 100, fa, 1)[0]}
        for v, (x, y) in variants.items():
            r = np.array([fig1.lagged_corr(x, y, int(k)) for k in lags])
            rows.append(dict(mouse_id=st.mouse_id.iloc[0], cohort=st.cohort.iloc[0], variant=v, n_trials=n,
                             asym_neg_minus_pos=asym(r, lags), peak_lag=int(lags[np.nanargmax(r)]), r0=r[lags == 0][0],
                             **trend))
            prof += [dict(mouse_id=st.mouse_id.iloc[0], cohort=st.cohort.iloc[0], variant=v, lag=int(k), r=rr)
                     for k, rr in zip(lags, r)]
    df = pd.DataFrame(rows)
    pd.DataFrame(prof).to_csv(OUT / "xcorr_profiles.csv", index=False)
    df.to_csv(OUT / "xcorr_asymmetry.csv", index=False)
    for (v, c), g in df.groupby(["variant", "cohort"]):
        a = g.asym_neg_minus_pos.dropna()
        print(f"{v:10s} {c}: asym (neg - pos lags) {a.mean():+.3f} +- {a.sem():.3f}, Wilcoxon p = {stats.wilcoxon(a).pvalue:.3g}, "
              f"t p = {stats.ttest_1samp(a, 0).pvalue:.3g}; median peak lag {g.peak_lag.median():+.0f}; r0 {g.r0.mean():.2f}; "
              f"W slope {g.w_slope_per_100.mean():+.3f}/100 tr, FA slope {g.fa_slope_per_100.mean():+.3f}/100 tr")
    raw = df[df.variant == "raw"]
    for c, g in raw.groupby("cohort"):  # does the asymmetry follow the opposite trends of the two curves?
        rho = stats.spearmanr(g.asym_neg_minus_pos, g.w_slope_per_100 - g.fa_slope_per_100)
        print(f"raw {c}: asymmetry vs (W slope - FA slope), Spearman rho {rho[0]:+.2f}, p = {rho[1]:.3g}")


def dlc_lag(ctx):
    print("\n=== Supp. Fig. 1H DLC lag profiles (positive lag: the DLC measure leads the whisker curve)")
    f = sorted((FIGURES_ROOT / "_explore").glob("movement_corr_20*-*-??_lagged.csv"))[-1]
    lag = pd.read_csv(f)
    learners = set(fig1.day0_table(ctx["table"], learners_only=True).mouse_id)
    lag = lag[lag.mouse_id.isin(learners)]
    rows = []
    for (m, c, met), g in lag.groupby(["mouse_id", "cohort", "metric"]):
        g = g.sort_values("lag")
        rows.append(dict(mouse_id=m, cohort=c, metric=met, asym=asym(g.spearman.to_numpy(), g.lag.to_numpy())))
    a = pd.DataFrame(rows)
    out = (a.groupby(["metric", "cohort"]).asym.agg(mean="mean", sem="sem", n="size",
                                                    p=lambda v: stats.wilcoxon(v.dropna()).pvalue).reset_index())
    out.to_csv(OUT / "dlc_lag_asymmetry.csv", index=False)
    print(f.name)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    ctx = {"table": shared_behaviour.load()}
    panel_i(ctx)
    xcorr(ctx)
    dlc_lag(ctx)
