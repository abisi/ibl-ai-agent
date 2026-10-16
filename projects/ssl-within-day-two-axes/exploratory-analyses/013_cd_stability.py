"""013 -- Test 4: does the reward-lick coding direction itself reorganise within the learning session? (user 2026-10-16).
Per session (008, both variants): AH and SL events split at the median tau; per half the unit-norm AH - SL axis (units z-scored on that half).
  cos_norm   cosine between the early and late axes from disjoint random halves of the events, divided by the square root of the two split-half
             reliabilities (50 splits, reliability floor 0.05; ~1 = same direction, clipped to +-1.5; not an angle)
  cos_raw    raw cosine between the early and late axes (biased low by noise; shown for reference)
  dp_early, dp_late   held-out separation d' of AH vs SL within each half (3-fold CV);  ddp = dp_late - dp_early
  gen        cross-generalisation: d' of the late AH / SL events on the early axis divided by the late axis' own held-out d' (1 = the early axis
             separates late events as well as the late axis)
A reorganisation of the reward-lick code on day 0 predicts lower cos_norm and gen on day 0 than at expert; a strengthening predicts ddp > 0.
Statistics: common.measure_stats (Wilcoxon vs 0 -- for cos_norm and gen also vs 1, in the csv --, day 0 vs expert, R+ vs R-, cohort x stage).
Output: tables/013_stats.csv; figures/013_cd_stability_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

MEAS = [("cos_norm", "early vs late CD: normalised cosine"), ("gen", "early-axis generalisation to late"), ("ddp", "Δ AH − SL separation d′ (late − early)"),
        ("dp_early", "AH − SL separation d′, early"), ("cos_raw", "early vs late CD: raw cosine")]


def main():
    S0 = C.population(pd.read_csv(C.TAB / "008_sessions.csv"))
    out = []
    for v in ("corr", "raw"):
        S = S0.rename(columns={f"{k}_{v}": k for k in ("cos_norm", "cos_raw", "gen", "dp_early", "dp_late")})
        S["ddp"] = S.dp_late - S.dp_early
        R = pd.DataFrame([r for m, _ in MEAS for r in C.measure_stats(S, m, "whole brain", m)]).assign(variant=v)
        for m in ("cos_norm", "gen"):
            for c in ("R+", "R-"):
                for s_ in ("learning", "expert"):
                    x = S[(S.cohort == c) & (S.stage == s_)][m].dropna()
                    R = pd.concat([R, pd.DataFrame([dict(variant=v, level="whole brain", measure=f"{m} vs 1", cohort=c, stage=s_, n_d0=len(x),
                                                         mean_d0=x.mean(), p_d0=stats.wilcoxon(x - 1).pvalue if len(x) >= 5 else np.nan)])])
        out.append(R)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
        fig = plt.figure(figsize=(7.6, 4.8)); gs = fig.add_gridspec(2, 3, wspace=0.6, hspace=1.2)
        for k, ((m, lab), l) in enumerate(zip(MEAS, "abcde")):
            ax = fig.add_subplot(gs[k // 3, k % 3]); Rm = R[R.measure == m]
            C.dot_panel(ax, S, m, Rm, lab, l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
            if m in ("cos_norm", "gen"):
                ax.axhline(1, color="0.4", lw=0.6, ls="--")
        ax = fig.add_subplot(gs[1, 2])
        for c in ("R+", "R-"):
            for s_, mk in (("learning", "o"), ("expert", "s")):
                d = S[(S.cohort == c) & (S.stage == s_)]
                ax.scatter(d.dp_early, d.dp_late, s=6, marker=mk, color=C.COH[c], facecolor="none" if s_ == "learning" else C.COH[c], lw=0.5)
        lim = [0, np.nanpercentile(np.r_[S.dp_early, S.dp_late], 99)]; ax.plot(lim, lim, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel("d′ early half"); ax.set_ylabel("d′ late half"); ax.set_title("AH − SL separation per session\n(circles day 0, squares expert)", fontsize=5.6)
        ax.text(-0.35, 1.25, "f", transform=ax.transAxes, fontsize=8, weight="bold")
        fig.suptitle(f"Test 4: stability of the reward-lick CD between the early and late half of the session "
                     f"({'baseline-corrected' if v == 'corr' else 'raw'} rates)", fontsize=6.4, x=0.01, y=1.0, ha="left")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(C.FIG / f"013_cd_stability_{v}.{ext}", dpi=300, bbox_inches="tight")
    R = pd.concat(out); R.to_csv(C.TAB / "013_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(R[R.cohort.isin(["R+", "R-"])][["variant", "measure", "cohort", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "cls"]].round(3).to_string())


if __name__ == "__main__":
    main()
