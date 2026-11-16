"""005 -- Movement / arousal vs choice state, and the state as a slow engagement variable (user 2026-11-12: "Do that ... yes
indeed it is supposed to be a slow engagement variable"). From the 001 per-trial tables (held-out state score s_state, early
whisker response g, held-out sensory hit / miss readout ro) and the Figure 1 movements cache (pre-stimulus jaw / nose / whisker
motion in [-0.2, 0) s, pupil area in [-2, 0) s; whisker trials of day 0). Day-0 learners, session = unit.

Part 1 -- joint model (active whisker trials with movements):
  gain     OLS of the early whisker response (rank, z) on lick, time (linear + quadratic), jaw, nose, whisker motion, pupil and
           s_state together (ranks, z): standardised coefficient of each pre-stimulus variable (b_*).
  decoding AUC of the lick from time only, state only, movements only, time + state, movements + state; then does the early (5-35 ms) sensory hit / miss readout survive the pre-stimulus variables? 5-fold CV AUC of the lick from
           [movements + pupil + time], [+ s_state], [+ s_state + ro], [ro only]: unique gain of ro = AUC(all) − AUC(no ro);
           partial rho(ro, lick | movements, pupil, time, s_state); rho(ro, each pre-stimulus variable).
Part 2 -- slow engagement (all active trials, whisker and auditory, in time order; s_state residualised on time, linear +
           quadratic):
  acf      autocorrelation of the state at lags 1..MAX_LAG trials (vs the same for a trial-order permutation null, N_PERM), and
           of the lick (whisker and auditory pooled, residualised on trial type and time); timescale = exponential fit of the
           state acf (tau in trials).
  history  previous trial -> current state: partial rho(s_state, previous reward | previous lick, current trial type, time)
           and rho(s_state, previous lick | ...); reward = auditory lick, or whisker lick in R+.
  past     slow state predicts the choice: AUC of the current whisker lick from the mean state of the previous K trials
           (excluding the current one) vs from the current trial's own state, K in PAST_K.
  blocks   non-overlapping blocks of BLOCK active trials: block-mean state vs block auditory hit rate and block whisker hit rate,
           and block-mean state vs block-mean early whisker response (| block time, linear-shift null over blocks, 10-50 %,
           non-wrapping, both directions) = slow gating.
Linear-shift nulls (user 2026-11-12; whole brain only): the outcome series shifted against the predictor series by 10-50 % of
  the trials (non-wrapping, both directions; covariates stay with the predictors), median over shifts -> *_null. The null keeps
  slow trends and autocorrelation and breaks the trial-by-trial pairing: null = slow, shared component; real - null =
  trial-specific component. Part 1 AUCs: lick shifted against the predictors (N_NULL_AUC shifts); coefficients: the early
  response shifted against all regressors; rho(ro, .): ro shifted. Part 2: whisker choice shifted against the current / past
  state; state history: the previous trial's lick (and reward) shifted against the state.
Output: tables/005_joint{_fine}.parquet, 005_engagement{_fine}.parquet, 005_acf{_fine}.parquet
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/005_engagement_movement.py [--fine]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(HERE), str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m01 = importlib.import_module("001_state_gating")
m04 = importlib.import_module("004_trial_tests")
m61 = importlib.import_module("061_roc_prelick_learners")
TAB = axel_bisi_root() / "combined_results_ks4" / "ssl-prestim-state-gating" / "tables"
MOVS = ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]
MAX_LAG, N_PERM, PAST_K, BLOCK, N_NULL, N_NULL_AUC = 15, 200, (1, 3, 5, 10), 10, 50, 20
WB = "whole_brain"


def rz(v):
    return m04.zc(stats.rankdata(v))


def resid(v, Z):
    A = np.c_[np.ones(len(v)), Z]
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def acf(x, max_lag):
    x = x - x.mean()
    d = x @ x
    return np.array([(x[:-k] @ x[k:]) / d if d > 0 else np.nan for k in range(1, max_lag + 1)])


def snull(fn, A, B, rng, n_null=N_NULL):
    """Median over linear shifts of fn(*A_shifted, *B_shifted): the arrays of B shifted against those of A by k trials
    (m01.shift_pairs: 10-50 %, non-wrapping, both directions). fn may return a scalar or a vector."""
    n = len(B[0])
    pairs = m01.shift_pairs(n)
    out = []
    for j in rng.permutation(len(pairs))[:n_null]:
        k, d = pairs[j]
        a = [x[k:] if d else x[: n - k] for x in A]
        b = [x[: n - k] if d else x[k:] for x in B]
        out.append(fn(*a, *b))
    out = np.asarray(out, float)
    if not len(out) or not np.isfinite(out).any():
        return np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(out, 0)


def joint(sid, area, d, rng):
    d = d.dropna(subset=["s_state", "g", "ro"])
    mv = [c for c in MOVS if d[c].notna().mean() > 0.8]
    d = d.dropna(subset=mv)
    y = d.lick.to_numpy(bool)
    if len(d) < 30 or not mv or min(y.sum(), (~y).sum()) < 5:
        return None
    t = m01.time_cov(d.start_time.to_numpy(float))
    M = np.c_[[rz(d[c]) for c in mv]].T
    s, ro = rz(d.s_state), m04.zc(d.ro)
    X = np.c_[np.ones(len(d)), y.astype(float), t, M, s]
    b = np.linalg.lstsq(X, rz(d.g), rcond=None)[0]
    row = dict(session_id=sid, area=area, n=len(d), movements=",".join(mv), b_state=b[-1])
    for c in MOVS:
        row[f"b_{c}"] = b[4 + mv.index(c)] if c in mv else np.nan
        row[f"rho_ro_{c}"] = stats.spearmanr(d.ro, d[c])[0] if c in mv else np.nan
    base = np.c_[M, t]
    row.update(auc_time=m04.cv_auc(t, y), auc_time_state=m04.cv_auc(np.c_[t, s], y), auc_mov=m04.cv_auc(M, y),
               auc_state=m04.cv_auc(s[:, None], y), auc_mov_state=m04.cv_auc(np.c_[M, s], y),
               auc_time_ro=m04.cv_auc(np.c_[t, ro], y), auc_pre=m04.cv_auc(base, y), auc_pre_state=m04.cv_auc(np.c_[base, s], y),
               auc_all=m04.cv_auc(np.c_[base, s, ro], y), auc_ro=m04.cv_auc(ro[:, None], y),
               rho_ro_state=stats.spearmanr(d.ro, d.s_state)[0],
               prho_ro_lick=m01.partial_rho(d.ro.to_numpy(), y.astype(float), np.c_[base, s]))
    row["auc_ro_unique"] = row["auc_all"] - row["auc_pre_state"]
    if area != WB:
        return row
    yf = y.astype(float)
    auc = lambda X, yy: m04.cv_auc(X, yy.astype(bool))
    for k, Xm in dict(auc_time=t, auc_mov=M, auc_state=s[:, None], auc_time_state=np.c_[t, s], auc_mov_state=np.c_[M, s],
                      auc_pre=base, auc_pre_state=np.c_[base, s], auc_all=np.c_[base, s, ro]).items():
        row[f"{k}_null"] = snull(auc, (Xm,), (yf,), rng, N_NULL_AUC)
    row["auc_ro_unique_null"] = row["auc_all_null"] - row["auc_pre_state_null"]
    bn = snull(lambda Xr, gg: np.linalg.lstsq(Xr, rz(gg), rcond=None)[0], (X,), (d.g.to_numpy(float),), rng)
    row["b_state_null"] = bn[-1] if np.ndim(bn) else np.nan
    for c in MOVS:
        row[f"b_{c}_null"] = bn[4 + mv.index(c)] if c in mv and np.ndim(bn) else np.nan
        row[f"rho_ro_{c}_null"] = (snull(lambda v, r: stats.spearmanr(r, v)[0], (d[c].to_numpy(float),),
                                         (d.ro.to_numpy(float),), rng) if c in mv else np.nan)
    row["rho_ro_state_null"] = snull(lambda v, r: stats.spearmanr(r, v)[0], (d.s_state.to_numpy(float),), (d.ro.to_numpy(float),), rng)
    row["prho_ro_lick_null"] = snull(lambda yy, Z, r: m01.partial_rho(r, yy, Z), (yf, np.c_[base, s]), (d.ro.to_numpy(float),), rng)
    return row


def block_null(a, b, Z, rng):
    m01.N_SHIFT_GATE = N_NULL
    nl = m01.gate_null(a, b, Z, rng) if len(a) >= 12 else np.array([])
    nl = nl[np.isfinite(nl)]
    return np.median(nl) if len(nl) else np.nan


def engagement(sid, area, d, rg, rng):
    d = d.dropna(subset=["s_state"]).sort_values("start_time")
    if len(d) < 60:
        return None, None
    isw = (d.trial_type == "whisker_trial").to_numpy()
    lick = d.lick.to_numpy(bool)
    t = m01.time_cov(d.start_time.to_numpy(float))
    s = resid(d.s_state.to_numpy(float), t)
    ls = resid(lick.astype(float), np.c_[t, isw])
    a_s, a_l = acf(s, MAX_LAG), acf(ls, MAX_LAG)
    null = np.array([acf(rng.permutation(s), MAX_LAG) for _ in range(N_PERM)])
    try:
        (tau,), _ = optimize.curve_fit(lambda k, tau: a_s[0] * np.exp(-(k - 1) / tau), np.arange(1, MAX_LAG + 1), a_s,
                                       p0=[3.0], bounds=(0.1, 500))
    except Exception:  # noqa: BLE001
        tau = np.nan
    row = dict(session_id=sid, area=area, n=len(d), acf1=a_s[0], acf1_null=null[:, 0].mean(), acf_mean_1_5=a_s[:5].mean(),
               acf_mean_1_5_null=null[:, :5].mean(), lick_acf1=a_l[0], lick_acf_mean_1_5=a_l[:5].mean(), tau=tau)
    # history: previous trial outcome
    rew = lick & (~isw | (rg == "R+"))
    Z = np.c_[lick[:-1], isw[1:], t[1:]]
    sc = d.s_state.to_numpy()[1:]
    row["prho_prev_reward"] = m01.partial_rho(sc, rew[:-1].astype(float), Z)
    row["prho_prev_lick"] = m01.partial_rho(sc, lick[:-1].astype(float), np.c_[isw[1:], t[1:]])
    if area == WB:  # previous outcome shifted against the state (and the current trial's covariates)
        row["prho_prev_lick_null"] = snull(lambda a, Zc, b: m01.partial_rho(a, b, Zc), (sc, np.c_[isw[1:], t[1:]]),
                                           (lick[:-1].astype(float),), rng)
        row["prho_prev_reward_null"] = snull(lambda a, Zc, b, lp: m01.partial_rho(a, b, np.c_[lp, Zc]),
                                             (sc, np.c_[isw[1:], t[1:]]), (rew[:-1].astype(float), lick[:-1].astype(float)), rng)
    # past state predicts the current whisker lick
    from sklearn.metrics import roc_auc_score
    w = np.where(isw)[0]
    yw = lick[w]
    if min(yw.sum(), (~yw).sum()) >= 3:
        def auc_ok(v, yy):
            ok = np.isfinite(v)
            yy = yy[ok].astype(bool)
            return roc_auc_score(yy, v[ok]) if min(yy.sum(), (~yy).sum()) >= 3 else np.nan
        row["auc_current"] = roc_auc_score(yw, s[w])
        pasts = {}
        for k in PAST_K:
            pasts[k] = np.array([s[max(0, i - k):i].mean() if i > 0 else np.nan for i in w])
            row[f"auc_past{k}"] = auc_ok(pasts[k], yw)
        if area == WB:  # whisker choice shifted against the (current / past) state series of the whisker trials
            row["auc_current_null"] = snull(auc_ok, (s[w],), (yw.astype(float),), rng)
            for k in PAST_K:
                row[f"auc_past{k}_null"] = snull(auc_ok, (pasts[k],), (yw.astype(float),), rng)
    # blocks
    blk = np.arange(len(d)) // BLOCK
    B = pd.DataFrame(dict(blk=blk, s=d.s_state.to_numpy(), lick=lick, isw=isw, g=d.g.to_numpy(), tt=d.start_time.to_numpy()))
    B = B.groupby("blk").apply(lambda b: pd.Series(dict(
        s=b.s.mean(), tt=b.tt.mean(), aud_hit=b.lick[~b.isw].mean() if (~b.isw).any() else np.nan,
        wh_hit=b.lick[b.isw].mean() if b.isw.any() else np.nan,
        g=np.nanmean(b.g[b.isw]) if b.isw.any() and np.isfinite(b.g[b.isw]).any() else np.nan))).dropna(subset=["s"])
    if len(B) >= 8:
        tb = m01.time_cov(B.tt.to_numpy(float))
        for col in ("aud_hit", "wh_hit", "g"):
            ok = B[col].notna().to_numpy()
            if ok.sum() >= 8:
                row[f"blk_{col}"] = m01.partial_rho(B.s.to_numpy()[ok], B[col].to_numpy()[ok], tb[ok])
                row[f"blk_{col}_null"] = block_null(B.s.to_numpy()[ok], B[col].to_numpy()[ok], tb[ok], rng)
        row["n_blocks"] = len(B)
    acfrow = pd.DataFrame(dict(session_id=sid, area=area, lag=np.arange(1, MAX_LAG + 1), acf_state=a_s,
                               acf_null=null.mean(0), acf_lick=a_l))
    return row, acfrow


def main(fine):
    sfx = "_fine" if fine else ""
    rng = np.random.default_rng(0)
    learners, _ = m61.learner_mice()
    R = pd.read_parquet(TAB / f"001_state_gating{sfx}.parquet")
    R = R[R.skipped_reason.isna() & R.mouse_id.isin(learners)]
    meta = R.drop_duplicates("session_id").set_index("session_id")[["mouse_id", "reward_group"]]
    T = pd.read_parquet(TAB / f"001_trials{sfx}.parquet")
    T = T[T.session_id.isin(meta.index) & (T.epoch == "active")].join(meta, on="session_id")
    T["lick"] = T.lick.astype(bool)

    W = T[T.trial_type == "whisker_trial"].merge(pd.read_parquet(m04.MOV)[["session_id", "start_time"] + MOVS],
                                                  on=["session_id", "start_time"], how="inner")
    J = [joint(sid, a, d, rng) for (sid, a), d in W.groupby(["session_id", "area"])]
    J = pd.DataFrame([j for j in J if j]).join(meta, on="session_id")
    J.to_parquet(TAB / f"005_joint{sfx}.parquet", index=False)

    rows, acfs = [], []
    for (sid, a), d in T.groupby(["session_id", "area"]):
        r, ac = engagement(sid, a, d, meta.reward_group[sid], rng)
        if r:
            rows.append(r); acfs.append(ac)
    E = pd.DataFrame(rows).join(meta, on="session_id")
    E.to_parquet(TAB / f"005_engagement{sfx}.parquet", index=False)
    pd.concat(acfs).join(meta, on="session_id").to_parquet(TAB / f"005_acf{sfx}.parquet", index=False)

    def wil(v):
        v = pd.Series(v).dropna()
        return (len(v), round(v.mean(), 3), round(stats.wilcoxon(v).pvalue, 4)) if len(v) >= 5 else (len(v),)
    for c in ("R+", "R-"):
        j, e = J[(J.area == WB) & (J.reward_group == c)], E[(E.area == WB) & (E.reward_group == c)]
        print(c, "joint", {k: wil(j[k]) for k in ["b_state"] + [f"b_{m}" for m in MOVS] + ["auc_pre", "auc_pre_state", "auc_all",
                                                                                               "auc_ro", "auc_ro_unique", "auc_time", "auc_time_state",
                                                                                               "auc_mov", "auc_state", "auc_mov_state",
                                                                                               "prho_ro_lick", "rho_ro_state"]
                             + [f"rho_ro_{m}" for m in MOVS]})
        print(c, "unique ro vs 0", wil(j.auc_ro_unique), "state over movements", wil(j.auc_pre_state - j.auc_pre))
        print(c, "engagement", {k: wil(e[k]) for k in ["acf1", "acf_mean_1_5", "lick_acf1", "lick_acf_mean_1_5",
                                                       "prho_prev_reward", "prho_prev_lick", "auc_current"]
                                + [f"auc_past{k}" for k in PAST_K]})
        print(c, "acf excess", wil(e.acf_mean_1_5 - e.acf_mean_1_5_null), "tau median", e.tau.median())
        print(c, "blocks", {k: wil(e[f"blk_{k}"] - e[f"blk_{k}_null"]) for k in ("aud_hit", "wh_hit", "g")},
              {k: wil(e[f"blk_{k}"]) for k in ("aud_hit", "wh_hit", "g")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", action="store_true")
    main(ap.parse_args().fine)
