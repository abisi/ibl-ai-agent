"""006 -- Per-event linear mixed models of the reward-lick CD score over the session, day 0 vs expert (user 2026-10-16: use every event rather
than one slope per session, for more power with the few expert sessions).
Data: within_day 002 per-event scores (held-out SL = 0, AH = 1), normalised session time tau; learners rule on day 0, all expert sessions.
Per cohort and contrast (WH vs SL, WH vs AH), on the events of the two classes:
  score ~ tau * W * E + (1 + tau | session)      W = 1 for whisker hits, E = 1 for expert sessions
  tau:W        day-0 slope of whisker hits relative to the reference class (drift of WH relative to SL / AH)
  tau:W:E      expert minus day-0 difference of that relative slope (the learning-specific term)
  W            day-0 offset of whisker hits relative to the reference at tau = 0 (start level);  W:E  expert minus day-0 start level
REML fit (statsmodels MixedLM, lbfgs); Wald p values. Cohort x stage: the same model on both cohorts with the cohort interaction
(score ~ tau * W * E * R) reports tau:W:E:R.
Output: tables/006_mixed_model.csv; figures/006_mixed_model.{png,pdf,svg}
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import common as C

warnings.filterwarnings("ignore")
TERMS = [("tau:W", "relative slope, day 0"), ("tau:W:E", "relative slope, expert − day 0"), ("W", "start offset, day 0"),
         ("W:E", "start offset, expert − day 0")]


def fit(D, formula):
    m = smf.mixedlm(formula, D, groups=D.session_id, re_formula="~tau")
    try:
        r = m.fit(reml=True, method="lbfgs")
    except Exception:  # noqa: BLE001
        r = m.fit(reml=True, method="powell")
    return r


def main():
    T = pd.read_parquet(C.m51.WITHIN / "slopes" / "trial_scores.parquet")
    T["cls"] = T.cls.replace({"FA": "SL"})
    T = C.population(T.dropna(subset=["cd", "tau"]))
    T["E"] = (T.stage == "expert").astype(int); T["R"] = (T.cohort == "R-").astype(int)
    rows = []
    for ref in ("SL", "AH"):
        D0 = T[T.cls.isin(["WH", ref])].copy(); D0["W"] = (D0.cls == "WH").astype(int)
        for c in ("R+", "R-"):
            D = D0[D0.cohort == c]
            r = fit(D, "cd ~ tau * W * E")
            for t, lab in TERMS:
                rows.append(dict(ref=ref, cohort=c, term=t, label=lab, coef=r.params[t], se=r.bse[t], p=r.pvalues[t],
                                 n_sessions=D.session_id.nunique(), n_events=len(D)))
        r = fit(D0, "cd ~ tau * W * E * R")
        for t in ("tau:W:E:R", "W:E:R"):
            rows.append(dict(ref=ref, cohort="R+ vs R-", term=t, label="cohort × stage, " + ("relative slope" if t.startswith("tau") else "start offset"),
                             coef=r.params[t], se=r.bse[t], p=r.pvalues[t], n_sessions=D0.session_id.nunique(), n_events=len(D0)))
    M = pd.DataFrame(rows); M.to_csv(C.TAB / "006_mixed_model.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.9))
    for ax, ref, l in zip(axs, ("SL", "AH"), "ab"):
        d = M[(M.ref == ref) & M.cohort.isin(["R+", "R-"])]
        ylab = []
        for k, (t, lab) in enumerate(TERMS):
            for j, c in enumerate(("R+", "R-")):
                r = d[(d.term == t) & (d.cohort == c)].iloc[0]; y = -(k * 3 + j)
                ax.errorbar(r.coef, y, xerr=1.96 * r.se, fmt="o", ms=3, color=C.COH[c], capsize=0, mfc=C.COH[c] if r.p < 0.05 else "white")
                ax.text(ax.get_xlim()[1] if False else 1.02, y, C.fp(r.p), transform=ax.get_yaxis_transform(), fontsize=4.4, va="center", color=C.COH[c])
            ylab.append((-(k * 3 + 0.5), lab))
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_yticks([y for y, _ in ylab], [t for _, t in ylab], fontsize=5)
        ax.set_xlabel(f"coefficient (score units; ± 95 % CI)")
        rc = M[(M.ref == ref) & (M.term == "tau:W:E:R")].iloc[0]
        ax.set_title(f"WH relative to {'spontaneous licks' if ref == 'SL' else 'auditory hits'}\ncohort × stage (relative slope) {C.fp(rc.p)}", fontsize=6)
        ax.text(-0.7, 1.08, l, transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Per-event mixed model: score ~ τ × WH × expert + (1 + τ | session) (green R+, magenta R−; filled p < 0.05)", fontsize=6.4,
                 x=0.01, y=1.06, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"006_mixed_model.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(M.round(4).to_string())


if __name__ == "__main__":
    main()
