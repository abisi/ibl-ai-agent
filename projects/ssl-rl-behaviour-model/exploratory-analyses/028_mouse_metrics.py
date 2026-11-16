"""028 -- per-mouse learning measures and per-trial latents of the lean hierarchical fit (user 2026-10-24: "i want my mouse-level model
to yield parameters that I can relate to other analyses (learning speeds for whisker, other things)").
Per mouse: MAP parameters with posterior sd; whisker learning rate alpha_W (alpha_neg slot with +stim; else alpha+ in R+, alpha- in R-) and
its time constant tau_W = -1 / ln(1 - alpha_W) in licked whisker trials; whisker trials from whisker onset until |V_W| >= 0.5 (censored at
the last trial if never); V_W at the end of days 0, 1, 2; rewards to halve thirst 0.5 / exp(mu_rho). R+ vs R- (mouse = unit): Mann-Whitney
and Welch on log tau_W and log trials-to-half (censored values kept at their censoring point, Mann-Whitney is the primary test then).
Per trial: model latents keyed by mouse_id, session_id, trial_idx with the prediction error delta = o - V on licked stimulus trials
(o = +1 auditory, whisker +1 R+ / -1 R-; with +hom the rewarded o is Tbar, recomputed here).
Output: tables/028_<tag>_mouse_metrics.csv, tables/028_<tag>_stats.csv, tables/028_<tag>_trial_latents.parquet
Run (haas): python 028_mouse_metrics.py --tag 025L
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import common as C


def whisker_alpha(r, stim):
    if stim:
        return r.p_alpha_neg
    return r.p_alpha_pos if r.cohort_set == "R+" else r.p_alpha_neg


def per_mouse(R, L, stim):
    rows = []
    for _, r in R.iterrows():
        e = L[(L.mouse_id == r.mouse_id) & (L.day >= 0)]
        w = e[e.trial_type == "whisker"].reset_index(drop=True)
        reach = np.flatnonzero(np.abs(w.V_W.to_numpy()) >= 0.5)
        a = whisker_alpha(r, stim)
        out = dict(mouse_id=r.mouse_id, reward_group=r.cohort_set, n_trials=r.n, alpha_W=a, tau_W=-1.0 / np.log1p(-a),
                   whisker_trials_to_half=(reach[0] + 1) if len(reach) else len(w), censored=len(reach) == 0,
                   licked_whisker_to_half=int(w.lick_flag.iloc[:reach[0] + 1].sum()) if len(reach) else int(w.lick_flag.sum()),
                   rewards_to_half_thirst=0.5 / np.exp(r.p_mu_rho))
        for d in (0, 1, 2):
            x = w[w.day == d]
            out[f"V_W_end_day{d}"] = x.V_W.iloc[-1] if len(x) else np.nan
        out.update({c: r[c] for c in R if c.startswith(("p_", "ysd_"))})
        rows.append(out)
    return pd.DataFrame(rows)


def cohort_tests(M):
    rows = []
    for q in ["tau_W", "whisker_trials_to_half", "licked_whisker_to_half", "rewards_to_half_thirst", "p_beta",
              "V_W_end_day0", "V_W_end_day1", "V_W_end_day2"]:
        if q not in M:
            continue
        a, b = M[M.reward_group == "R+"][q].dropna(), M[M.reward_group == "R-"][q].dropna()
        lg = q not in ("V_W_end_day0", "V_W_end_day1", "V_W_end_day2")
        f = (lambda x: np.log(np.maximum(x, 1e-9))) if lg else (lambda x: x)
        rows.append(dict(measure=q, scale="log" if lg else "linear", median_rplus=a.median(), median_rminus=b.median(), n_rplus=len(a),
                         n_rminus=len(b), p_mwu=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(f(a), f(b), equal_var=False).pvalue,
                         n_censored_rplus=int(M[M.reward_group == "R+"].censored.sum()) if "whisker_to_half" in q or "trials_to_half" in q else 0,
                         n_censored_rminus=int(M[M.reward_group == "R-"].censored.sum()) if "whisker_to_half" in q or "trials_to_half" in q else 0))
    return pd.DataFrame(rows)


def trial_latents(L, R, model):
    L = L.copy()
    o = np.where(L.trial_type == "auditory", 1.0, np.where(L.reward_group == "R+", 1.0, -1.0))
    if "+hom" in model:
        rho = np.exp(L.mouse_id.map(R.set_index("mouse_id").p_mu_rho))
        rew = (L.trial_type == "auditory") | (L.reward_group == "R+")
        o = np.where(rew, np.maximum(0.0, 1.0 - rho * L.S), o)
    stim = L.trial_type.isin(["auditory", "whisker"]) & (L.lick_flag == 1)
    L["outcome"] = np.where(stim, o, np.nan)
    L["delta"] = np.where(stim, o - L.V, np.nan)
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="025L")
    a = ap.parse_args()
    R = pd.read_csv(C.FITS / f"{a.tag}_hier_mice.csv")
    model = R.model.iloc[0]
    stim = "+stim" in model
    L = pd.read_parquet(C.TAB / f"{a.tag}_latents" / f"{model}.parquet")
    M = per_mouse(R, L, stim)
    M.insert(0, "model", model)
    M.to_csv(C.TAB / f"028_{a.tag}_mouse_metrics.csv", index=False)
    S = cohort_tests(M)
    S.to_csv(C.TAB / f"028_{a.tag}_stats.csv", index=False)
    trial_latents(L, R, model).assign(model=model).to_parquet(C.TAB / f"028_{a.tag}_trial_latents.parquet")
    print(M.groupby("reward_group")[["alpha_W", "tau_W", "whisker_trials_to_half", "V_W_end_day0", "V_W_end_day1", "V_W_end_day2",
                                     "rewards_to_half_thirst"]].median().round(3).to_string())
    print(S.round(4).to_string(), flush=True)
