"""006 -- Curve-based half-way rule (L7) for gradual learners + final
learning-trial table and review figure.

L7 works on the re-estimated smooth curves (002, 'eb' prior) through the
discrimination curve d(t) = p_whisker(t) - p_FA(t) (FA at actual no-stim
times). Step models (L5/L6) only express abrupt transitions; L7 targets
the half-way point of a gradual change:
  R+ : d0 = mean d over the first 10 whisker trials, d_hi = 90th percentile
       of d over the session. Transition required: d_hi - d0 >= MIN_DELTA.
       LT = first t with d(t) >= d0 + 0.5 (d_hi - d0) that holds on >= 16 of
       the 20 trials from t.
  R- : d_hi = max of d over the first 30 whisker trials (initial generalised
       licking), d_lo = 10th percentile of d after that maximum. Transition
       required: d_hi - d_lo >= MIN_DELTA and d_hi > MIN_DELTA (whisker
       really above FA initially). LT = first t after the maximum with
       d(t) <= d_lo + 0.5 (d_hi - d_lo) holding on >= 16 of the next 20.
MIN_DELTA = 0.2 (chosen by inspection of the review grids; stated, not fit).

Final table (artifacts/006_learning_trials_final.csv), one row per session
with full provenance: stored LT + source, L6 (joint whisker+FA Bayesian
change point: median, 90% CI, log10 BF, category), L6_lenient, L7, and
`lt_recommended` = L6 where L6 found a learner, else NaN; `lt_recommended_
broad` = L6 if defined, else L7 (larger N, mixes step and gradual
definitions -- flagged in `lt_recommended_broad_source`).
Figure: exploratory-analyses/006_final_review_<R+|R->.png.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
_s = importlib.util.spec_from_file_location("r003", HERE / "003_learning_trial_rules.py")
r003 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r003)

MIN_DELTA = 0.2
W, K = 20, 16
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def halfway(d: np.ndarray, rg: int) -> float:
    n = len(d)
    if rg == 1:
        d0, dhi = d[:10].mean(), np.percentile(d, 90)
        if dhi - d0 < MIN_DELTA:
            return np.nan
        thr = d0 + 0.5 * (dhi - d0)
        for t in range(n - W + 1):
            if d[t] >= thr and (d[t:t + W] >= thr).sum() >= K:
                return float(t)
        return np.nan
    i_max = int(np.argmax(d[:30]))
    dhi = d[i_max]
    rest = d[i_max:]
    dlo = np.percentile(rest, 10)
    if dhi - dlo < MIN_DELTA or dhi < MIN_DELTA:
        return np.nan
    thr = dlo + 0.5 * (dhi - dlo)
    for t in range(i_max, n - W + 1):
        if d[t] <= thr and (d[t:t + W] <= thr).sum() >= K:
            return float(t)
    return np.nan


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    cps = pickle.load(open(ART / "005_cp_posteriors.pkl", "rb"))
    df = pd.read_csv(ART / "005_learning_trials.csv").set_index("session_id")
    for sid, d in inputs.items():
        rg = 1 if d["reward_group"] == "R+" else 0
        e = curves[sid]["eb"]
        disc = e["p_mean"] - e["fa_time"]
        lt7 = halfway(disc, rg)
        df.loc[sid, "L7"] = lt7
        for k, v in r003.evaluate(lt7, d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"], rg).items():
            df.loc[sid, f"L7_{k}"] = v
    df["lt_recommended"] = df["L6"]
    df["lt_recommended_broad"] = df["L6"].fillna(df["L7"])
    df["lt_recommended_broad_source"] = np.where(df["L6"].notna(), "L6_changepoint",
                                                 np.where(df["L7"].notna(), "L7_halfway", "none"))
    df = df.reset_index()
    df.to_csv(ART / "006_learning_trials_final.csv", index=False)

    pd.set_option("display.width", 250)
    rows = []
    for rg, g in df.groupby("reward_group"):
        for r_ in ["L0_stored", "L6", "L7", "lt_recommended_broad"]:
            col_c = f"{r_}_contrast20" if r_ != "lt_recommended_broad" else None
            rows.append(dict(reward_group=rg, rule=r_, n_defined=int(g[r_].notna().sum()), median_lt=g[r_].median(),
                             n_at_10=int((g[r_] == 10).sum()),
                             contrast20=g[col_c].mean() if col_c else np.nan,
                             held_post=g[f"{r_}_held_post"].mean() if col_c else np.nan))
        both = g[g.L6.notna() & g.L7.notna()]
        print(f"{rg}: L6 & L7 both defined n={len(both)}, median |L6-L7|={np.median(np.abs(both.L6 - both.L7)):.1f}, "
              f"L7 inside L6 90% CI: {((both.L7 >= both.L6_ci05) & (both.L7 <= both.L6_ci95)).sum()}")
    summ = pd.DataFrame(rows)
    summ.to_csv(HERE / "006_rule_summary.csv", index=False)
    print(summ.round(3).to_string())
    print(pd.crosstab(df.reward_group, df.lt_recommended_broad_source))

    for rg_name in ("R+", "R-"):
        g = df[df.reward_group == rg_name].sort_values(["lt_recommended_broad_source", "session_id"])
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), constrained_layout=True)
        for ax, row in zip(axes.flat, g.itertuples()):
            sid = row.session_id
            d, e = inputs[sid], curves[sid]["eb"]
            o = d["w_outcomes"]
            x = np.arange(len(o))
            col = COHORT_COLOR[rg_name]
            ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.2, lw=0)
            ax.plot(x, e["p_mean"], color=col, lw=1.4)
            ax.plot(x, e["fa_time"], color="#444444", lw=1, ls="--")
            ax.plot(x, e["p_mean"] - e["fa_time"], color="#8c564b", lw=0.9, ls=":")
            ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=2, color="k", marker="|")
            r = cps.get(sid)
            if r is not None and not pd.isna(row.L6):
                ax.fill_between(x, -0.2, -0.2 + r["pk"] / r["pk"].max() * 0.25, color="#1f77b4", alpha=0.35, lw=0, step="mid")
                ax.axvspan(row.L6_ci05, row.L6_ci95, color="#1f77b4", alpha=0.12, lw=0)
            for val, c_, ls in ((row.L0_stored, "#d62728", "-"), (row.L6, "#1f77b4", "-"), (row.L7, "#ff7f0e", "--")):
                if not pd.isna(val):
                    ax.axvline(val, color=c_, lw=1.4, ls=ls)
            ax.axhline(0, color="#bbbbbb", lw=0.5)
            ax.set_ylim(-0.25, 1.12)
            ax.set_title(f"{sid[:5]} {row.learning_category} | {row.L6_category}\nstored={row.L0_stored:.0f} "
                         f"L6={row.L6:.0f} [{row.L6_ci05:.0f},{row.L6_ci95:.0f}] L7={row.L7:.0f}".replace("nan", "-"),
                         fontsize=6.5)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        fig.suptitle(f"{rg_name}: final learning-trial review. Thick = re-estimated whisker curve (80% CI), dashed = FA at "
                     f"actual times, dotted brown = discrimination (whisker - FA). Red = stored LT; blue = L6 joint Bayesian "
                     f"change point (shade 90% CI; posterior at bottom); orange dashed = L7 half-way rule (gradual)",
                     fontsize=9)
        fig.savefig(HERE / f"006_final_review_{rg_name.replace('+', 'plus').replace('-', 'minus')}.png", dpi=150)


if __name__ == "__main__":
    main()
