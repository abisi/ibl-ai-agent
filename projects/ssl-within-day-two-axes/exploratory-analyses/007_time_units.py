"""007 -- Whisker-hit drift on the reward-lick CD in other units of time and experience, day 0 vs expert (user 2026-10-16).
Data: within_day 002 per-event scores (held-out SL = 0, AH = 1). Per session, the drift of whisker hits relative to spontaneous licks is the
difference of the two OLS slopes (WH - SL) against:
  tau          normalised session time (reference; 0 -> 1)
  trial        event order among all analysed events (per 100 events)
  whisker hit  number of whisker hits so far (per 10 WH); also restricted to the first N_FIRST whisker hits of the session (same exposure
               range at both stages: wh_first)
  minutes      clock time from the first event (per 10 min)
  rewards      rewarded events so far (per 10 rewards): auditory hits in both cohorts plus whisker hits in R+ (among the analysed events,
               i.e. after the warm-up cut and the end trim; reward deliveries not read from the NWB)
Statistics: common.measure_stats per unit (Wilcoxon vs 0, day 0 vs expert, R+ vs R-, cohort x stage).
Output: tables/007_sessions.csv, 007_stats.csv; figures/007_time_units.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

N_FIRST, MIN_EV = 30, 4
UNITS = [("tau", "per session (τ)", 1.0), ("trial", "per 100 events", 100.0), ("wh", "per 10 whisker hits", 10.0),
         ("wh_first", f"per 10 WH, first {N_FIRST} WH", 10.0), ("minutes", "per 10 min", 10.0), ("rewards", "per 10 rewards", 10.0)]


def slope(x, y):
    return np.polyfit(x, y, 1)[0] if len(x) >= MIN_EV and np.ptp(x) > 0 else np.nan


def session(E, cohort):
    E = E.sort_values("t").reset_index(drop=True)
    E["trial"] = np.arange(len(E))
    E["wh"] = (E.cls == "WH").cumsum() - (E.cls == "WH")
    E["minutes"] = (E.t - E.t.iloc[0]) / 60
    rew = (E.cls == "AH") | ((E.cls == "WH") & (cohort == "R+"))
    E["rewards"] = rew.cumsum() - rew
    out = {}
    for u, _, scale in UNITS:
        x = "wh" if u == "wh_first" else u
        d = E if u != "wh_first" else E[E.wh < N_FIRST]
        w, s = d[d.cls == "WH"], d[d.cls == "SL"]
        out[f"s_{u}"] = (slope(w[x], w.cd) - slope(s[x], s.cd)) * scale
        out[f"sWH_{u}"] = slope(w[x], w.cd) * scale
    return out


def main():
    T = pd.read_parquet(C.m51.WITHIN / "slopes" / "trial_scores.parquet")
    T["cls"] = T.cls.replace({"FA": "SL"})
    meta = T.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    S = pd.DataFrame([dict(session_id=sid, **session(E, E.cohort.iloc[0])) for sid, E in T.groupby("session_id")]).merge(meta, on="session_id")
    S = C.population(S); S.to_csv(C.TAB / "007_sessions.csv", index=False)
    R = pd.DataFrame([r for u, _, _ in UNITS for r in C.measure_stats(S, f"s_{u}", "whole brain", f"s_{u}")])
    R.to_csv(C.TAB / "007_stats.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.8), gridspec_kw=dict(wspace=0.55, hspace=1.3))
    for ax, (u, lab, _), l in zip(axs.ravel(), UNITS, "abcdef"):
        Rm = R[R.measure == f"s_{u}"]
        C.dot_panel(ax, S, f"s_{u}", Rm, f"WH − SL drift, {lab}", l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle("Reward-lick CD: whisker-hit drift relative to spontaneous licks in different units of time and experience (score units; "
                 "green R+, magenta R−; open day 0, filled expert)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"007_time_units.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(R[["measure", "cohort", "n_d0", "n_e", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(4).to_string())


if __name__ == "__main__":
    main()
