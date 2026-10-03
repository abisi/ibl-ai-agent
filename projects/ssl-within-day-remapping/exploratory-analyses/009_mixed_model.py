"""Mixed model on single trials: do whisker hits move along the reward-lick coding direction (CD) during the session,
relative to a reference lick type, and does this differ between R+ and R- mice?

Data: 002 trial_scores.parquet (one row per lick event: CD projection `cd`, SL = 0 and AH = 1 per session; normalised
session time `tau`, 0 -> 1). Learning day (day 0) and expert sessions are fitted separately.
Model (statsmodels MixedLM, REML), per stage and reference (SL, or AH):
    cd ~ wh * tau * rplus
    wh    = 1 for whisker hits, 0 for the reference lick type (spontaneous licks, or auditory hits)
    tau   = time in the session (0 = first analysed event, 1 = last)
    rplus = 1 for R+ mice, 0 for R- mice
    random effects: per session, an intercept, a time slope and a whisker-hit offset (re_formula ~ tau + wh);
    expert sessions additionally share a per-mouse intercept (variance component).
Plain-language terms:
    wh:tau          how much whisker hits move toward auditory hits along the CD during an R- session, beyond the
                    reference lick type (CD units per session; 1 = the full distance from SL to AH)
    wh:tau:rplus    how much more this happens in R+ than in R- sessions  (the test of interest)
    per-cohort drift = wh:tau (R-) and wh:tau + wh:tau:rplus (R+)
p values: Wald (model) and a permutation test of wh:tau:rplus that shuffles cohort labels across mice (N_PERM refits).
Output: combined_results_ks4/_within_day<TAG>/mixed_model/{mixed_model_terms.csv, mixed_model_summary.md}
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
BASE = m51.RES / f"_within_day{m51.TAG}"
OUT = BASE / "mixed_model"
SL_MAX = 150            # spontaneous licks per session subsampled to at most this many (balances events; seed 0)
FORMULA = "cd ~ wh * tau * rplus"
DATA = {}


def prepare(T, stage, ref):
    d = T[(T.stage == stage) & T.cls.isin(["WH", ref]) & np.isfinite(T.cd)].copy()
    if ref == "FA":                                    # cap spontaneous licks per session (they outnumber hits)
        rng = np.random.default_rng(0)
        keep = []
        for _, g in d.groupby("session_id"):
            sl = g[g.cls == "FA"]
            keep.append(pd.concat([g[g.cls == "WH"], sl.sample(min(len(sl), SL_MAX), random_state=int(rng.integers(1e9)))]))
        d = pd.concat(keep)
    d["wh"] = (d.cls == "WH").astype(float)
    d["rplus"] = (d.cohort == "R+").astype(float)
    return d.reset_index(drop=True)


def fit(d, stage):
    import statsmodels.formula.api as smf
    kw = dict(groups="session_id", re_formula="~tau + wh")
    if stage == "expert":
        kw["vc_formula"] = {"mouse": "0 + C(mouse_id)"}
    try:
        res = smf.mixedlm(FORMULA, d, **kw).fit(reml=True, method=["lbfgs", "powell"])
    except Exception:
        res = smf.mixedlm(FORMULA, d, groups="session_id", re_formula="~tau").fit(reml=True)
    return res


def perm_job(args):
    k, stage, ref = args
    d = DATA[(stage, ref)].copy()
    rng = np.random.default_rng(1000 + k)
    mice = d.mouse_id.unique(); lab = d.groupby("mouse_id").rplus.first().reindex(mice).to_numpy()
    d["rplus"] = pd.Series(rng.permutation(lab), index=mice).reindex(d.mouse_id).to_numpy()
    try:
        return fit(d, stage).params.get("wh:tau:rplus", np.nan)
    except Exception:
        return np.nan


def main(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_parquet(BASE / "slopes" / "trial_scores.parquet")
    rows, lines = [], []
    names = {"FA": "spontaneous licks", "AH": "auditory hits"}
    for stage in ["learning", "expert"]:
        for ref in ["FA", "AH"]:
            d = prepare(T, stage, ref); DATA[(stage, ref)] = d
            res = fit(d, stage)
            b, se, p = res.params, res.bse, res.pvalues
            with mp.get_context("fork").Pool(a.n_proc) as pool:
                null = np.array(pool.map(perm_job, [(k, stage, ref) for k in range(a.n_perm)]))
            null = null[np.isfinite(null)]
            obs = b["wh:tau:rplus"]
            p_perm = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))
            # per-cohort drift and its s.e. from the covariance of the fixed effects
            C = res.cov_params()
            drift_rm, se_rm = b["wh:tau"], se["wh:tau"]
            drift_rp = b["wh:tau"] + obs
            se_rp = float(np.sqrt(C.loc["wh:tau", "wh:tau"] + C.loc["wh:tau:rplus", "wh:tau:rplus"] + 2 * C.loc["wh:tau", "wh:tau:rplus"]))
            from scipy import stats
            p_rp = 2 * stats.norm.sf(abs(drift_rp / se_rp))
            r = dict(stage=stage, reference=names[ref], n_sessions=d.session_id.nunique(), n_mice=d.mouse_id.nunique(),
                     n_events=len(d), n_wh=int(d.wh.sum()), drift_Rminus=drift_rm, se_Rminus=se_rm, p_Rminus=p["wh:tau"],
                     drift_Rplus=drift_rp, se_Rplus=se_rp, p_Rplus=p_rp, diff_Rplus_minus_Rminus=obs,
                     se_diff=se["wh:tau:rplus"], p_wald_diff=p["wh:tau:rplus"], p_perm_diff=p_perm, n_perm=len(null),
                     converged=bool(getattr(res, "converged", True)))
            rows.append(r)
            day = "learning day (day 0)" if stage == "learning" else "expert days"
            lines.append(
                f"- **{day}, reference = {names[ref]}** ({r['n_sessions']} sessions, {r['n_mice']} mice, {r['n_wh']} whisker hits): "
                f"over one session, whisker hits move along the coding direction by {drift_rp:+.2f} ± {se_rp:.2f} in R+ "
                f"(p = {p_rp:.3g}) and {drift_rm:+.2f} ± {se_rm:.2f} in R− (p = {p['wh:tau']:.3g}) relative to {names[ref]} "
                f"(1 = the full distance from spontaneous licks to auditory hits). R+ minus R−: {obs:+.2f} ± {se['wh:tau:rplus']:.2f} "
                f"(model p = {p['wh:tau:rplus']:.3g}; shuffling cohort labels across mice, p = {p_perm:.3g}, {len(null)} shuffles).")
            print(lines[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "mixed_model_terms.csv", index=False)
    (OUT / "mixed_model_summary.md").write_text(
        "# Mixed model on single trials (coding-direction projections)\n\n"
        f"Model: `{FORMULA}`, random intercept, time slope and whisker-hit offset per session"
        " (expert: plus a per-mouse intercept). Spontaneous licks capped at "
        f"{SL_MAX} per session. Drift values are the change in the gap between whisker hits and the reference over one "
        "whole session, in coding-direction units (0 = spontaneous licks, 1 = auditory hits).\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    print(f"ALL DONE {OUT} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-proc", type=int, default=60)
    main(ap.parse_args())
