"""008 -- The neural pre-stimulus state inside the behavioural GLM of Figure 1 (glm_v6), and movement -> early whisker response ->
choice mediation (user 2026-11-12: "do what you suggest": link the state and the movement effect on the response to the
behaviour GLM). Day-0 learner sessions of 001 (whole brain), active whisker trials present in both the GLM design matrix
(glm_v5_run.load_session on Myriam Hamon's module, as glm_v6 and Figure 1) and the 001 trial table (matched on start_time).

Models (logistic, L2 at the session's glm_v6 leave-one-session-out penalty C*, predictors standardised within the training
blocks; glm_v5 block CV: 5 contiguous blocks, gap 3 whisker trials, N_REP repeats with the glm_v5 offsets; scores as glm_v5:
held-out log-likelihood per trial and AUC relative to the fold's training lick rate, pooled out-of-fold, mean over repeats):
  glm        the 10 GLM predictors
  glm_state  + s_state (001 held-out state score)
  glm_g      + g (early whisker response on the passive-pre template: 007 g_u, uncentred; 001 g is not used, its template is
             relative to the session's grand mean)
  glm_ro     + ro (early sensory hit / miss readout)
  glm_all    + s_state + ro
  state      s_state alone
  d_<m>      = model m - glm (LL per trial, AUC). Linear-shift null of d_state / d_ro: the neural column shifted against the
             trials (10-50 %, non-wrapping, both directions; GLM predictors and lick stay), N_NULL shifts x N_REP_NULL
             repeats, median -> d_*_null.
What explains the state: standardised OLS coefficients of s_state on the 10 GLM predictors (beta_state_<f>), block-CV R^2
  (ridge) and its shift null, rho(s_state, out-of-fold GLM logit) and partial rho | session time (linear + quadratic).
Mediation (full-data fits, z-scored columns): a_f = coefficient of predictor f on the early response g (OLS g ~ all GLM
  predictors), b = coefficient of g on the lick (logistic lick ~ GLM + g at C*), c_f / c'_f = coefficient of f without / with
  g; indirect_f = a_f b; the same with ro in place of g (a_ro_f, b_ro, indirect_ro_f).
Output: tables/008_glm_state.parquet (session), 008_glm_state_beta.parquet (session x predictor)
Run (haas, repo root): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities nice \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/008_glm_state.py
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(HERE), str(PROJECTS / "ssl-paper-figures"), str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"),
                str(PROJECTS.parent / "scripts")]
m01 = importlib.import_module("001_state_gating")
m61 = importlib.import_module("061_roc_prelick_learners")
TAB = m01.TAB
GLM_DIR_NAME = "glm_v6"
FEATS = ["cumulative_reward_fraction", "time_since_reward", "whisker_speed_pre", "nose_speed_pre", "pupil_area_pre",
         "jaw_speed_pre", "spontaneous_lick_rate", "lick_whisker_trial_n-3", "lick_whisker_trial_n-2", "lick_whisker_trial_n-1"]
N_REP, N_NULL, N_REP_NULL, WB = 10, 20, 3, "whole_brain"


def zc(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1)


def cv_model(X, y, C, n_rep, v5, bs):
    """glm_v5 block CV at fixed C: mean over repeats of (LL per trial, AUC vs training rate); out-of-fold logit (mean)."""
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    rng = np.random.default_rng(7)
    offsets = [0] + [int(rng.integers(n)) for _ in range(n_rep - 1)]
    ll, auc, logit = [], [], np.full((n_rep, n), np.nan)
    for r, off in enumerate(offsets):
        P, THR = np.full(n, np.nan), np.full(n, np.nan)
        for tr, te in bs(n, v5.N_FOLDS, offset=off, gap=v5.GAP):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = v5._fit(sc.transform(X[tr]), y[tr], C)
            P[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
            THR[te] = y[tr].mean()
        ok = np.isfinite(P)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        s = v5.scores(y[ok], P[ok], THR[ok], THR[ok])
        ll.append(s["test_ll_per_trial"]); auc.append(s["test_auc"])
        logit[r, ok] = v5._logit(P[ok])
    return (np.mean(ll) if ll else np.nan), (np.mean(auc) if auc else np.nan), np.nanmean(logit, 0)


def block_r2(X, v, bs, v5):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    pred = np.full(len(v), np.nan)
    for tr, te in bs(len(v), v5.N_FOLDS, offset=0, gap=v5.GAP):
        sc = StandardScaler().fit(X[tr])
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), v[tr]).predict(sc.transform(X[te]))
    ok = np.isfinite(pred)
    return 1 - np.sum((v[ok] - pred[ok]) ** 2) / np.sum((v[ok] - v[ok].mean()) ** 2)


def session(sid, mouse, rg, tr, C, src):
    warnings.filterwarnings("ignore")
    import glm_v5_run as v5
    from common import NWB_DIR
    from glm_cv_compare import block_splits as bs
    try:
        X, y, t, names = v5.load_session(v5.glm_module(src), str(NWB_DIR / f"{sid}.nwb"))  # src built once in main (race)
    except Exception as e:  # noqa: BLE001
        return dict(session_id=sid, skipped_reason=f"glm load: {e!r}"), None
    G = pd.DataFrame(X, columns=names).assign(start_time=t, lick_glm=y)
    tr = tr.sort_values("start_time")
    M = pd.merge_asof(G.sort_values("start_time"), tr, on="start_time", direction="nearest", tolerance=0.05).dropna(
        subset=["s_state", "g", "ro"])
    base = dict(session_id=sid, mouse_id=mouse, reward_group=rg, C_star=C, n_glm=len(G), n=len(M), skipped_reason=None)
    if len(M) < 40 or min(M.lick_glm.sum(), (1 - M.lick_glm).sum()) < 8:
        return dict(base, skipped_reason="too few matched trials"), None
    base["lick_agree"] = float(np.mean(M.lick_glm.to_numpy() == M.lick.to_numpy().astype(int)))
    col = {f: [c for c in names if c == f or c.startswith(f + "_t")] for f in FEATS}
    Xg = M[names].to_numpy(float)
    y = M.lick_glm.to_numpy(int)
    s, g, ro = M.s_state.to_numpy(float), M.g.to_numpy(float), M.ro.to_numpy(float)
    row = dict(base)
    res = {}
    for k, Xm in dict(glm=Xg, glm_state=np.c_[Xg, s], glm_g=np.c_[Xg, g], glm_ro=np.c_[Xg, ro], glm_all=np.c_[Xg, s, ro],
                      state=s[:, None]).items():
        row[f"ll_{k}"], row[f"auc_{k}"], lg = cv_model(Xm, y, C, N_REP, v5, bs)
        res[k] = lg
    for k in ("glm_state", "glm_g", "glm_ro", "glm_all"):
        row[f"dll_{k}"], row[f"dauc_{k}"] = row[f"ll_{k}"] - row["ll_glm"], row[f"auc_{k}"] - row["auc_glm"]
    # shift null: the neural column against the trials
    n = len(y)
    rng = np.random.default_rng(int(sid[2:5]) if sid[2:5].isdigit() else 0)
    pairs = m01.shift_pairs(n)
    for key, v in (("glm_state", s), ("glm_ro", ro)):
        dl, da = [], []
        for j in rng.permutation(len(pairs))[:N_NULL]:
            k, d = pairs[j]
            idx = np.arange(k, n) if d else np.arange(n - k)
            vs = v[: n - k] if d else v[k:]
            if min(y[idx].sum(), (1 - y[idx]).sum()) < 5:
                continue
            l0, a0, _ = cv_model(Xg[idx], y[idx], C, N_REP_NULL, v5, bs)
            l1, a1, _ = cv_model(np.c_[Xg[idx], vs], y[idx], C, N_REP_NULL, v5, bs)
            dl.append(l1 - l0); da.append(a1 - a0)
        row[f"dll_{key}_null"] = float(np.nanmedian(dl)) if dl else np.nan
        row[f"dauc_{key}_null"] = float(np.nanmedian(da)) if da else np.nan
    # what explains the state
    tc = m01.time_cov(M.start_time.to_numpy(float))
    row["rho_state_glm_logit"] = stats.spearmanr(s, res["glm"], nan_policy="omit")[0]
    row["prho_state_glm_logit"] = m01.partial_rho(s, res["glm"], tc)
    row["r2_state_glm"] = block_r2(Xg, s, bs, v5)
    nl = []
    for j in rng.permutation(len(pairs))[:N_NULL]:
        k, d = pairs[j]
        nl.append(block_r2(Xg[k:], s[: n - k], bs, v5) if d else block_r2(Xg[: n - k], s[k:], bs, v5))
    row["r2_state_glm_null"] = float(np.nanmedian(nl))
    Z = np.c_[[zc(Xg[:, i]) for i in range(Xg.shape[1])]].T
    A = np.c_[np.ones(n), Z]
    beta = np.linalg.lstsq(A, zc(s), rcond=None)[0][1:]
    brows = [dict(session_id=sid, reward_group=rg, feature=f, beta_state=float(np.mean(beta[[names.index(c) for c in col[f]]])))
             for f in FEATS if col[f]]
    # mediation through the early response (g) and the readout (ro)
    for lab, v in (("g", g), ("ro", ro)):
        a = np.linalg.lstsq(A, zc(v), rcond=None)[0][1:]
        c_tot = v5._fit(Z, y, C).coef_[0]
        m_full = v5._fit(np.c_[Z, zc(v)], y, C)
        c_dir, b = m_full.coef_[0][:-1], m_full.coef_[0][-1]
        row[f"b_{lab}"] = float(b)
        for br in brows:
            ii = [names.index(c) for c in col[br["feature"]]]
            br.update({f"a_{lab}": float(np.mean(a[ii])), f"c_{lab}": float(np.mean(c_tot[ii])),
                       f"cprime_{lab}": float(np.mean(c_dir[ii])), f"indirect_{lab}": float(np.mean(a[ii]) * b)})
    return row, pd.DataFrame(brows)


def main():
    learners, _ = m61.learner_mice()
    R = pd.read_parquet(TAB / "001_state_gating.parquet")
    R = R[R.skipped_reason.isna() & R.mouse_id.isin(learners) & (R.area == WB)]
    T = pd.read_parquet(TAB / "001_trials.parquet")
    T = T[(T.area == WB) & (T.epoch == "active") & (T.trial_type == "whisker_trial")][["session_id", "start_time", "lick", "s_state",
                                                                                      "ro"]]
    G = pd.read_parquet(TAB / "007_trials.parquet").query("area == @WB and window == 'early' and epoch == 'active'")
    T = T.merge(G[["session_id", "start_time", "g_u"]].rename(columns={"g_u": "g"}), on=["session_id", "start_time"], how="inner")
    from common import CACHE_DIR
    cs = pd.read_csv(CACHE_DIR / GLM_DIR_NAME / "c_choice.csv").set_index("session_id").C_star
    jobs = [(r.session_id, r.mouse_id, r.reward_group, T[T.session_id == r.session_id], float(cs.get(r.session_id, 0.1)))
            for r in R.itertuples()]
    print(f"[008] {len(jobs)} sessions", flush=True)
    import glm_v5_run as v5
    src = v5.glm_source_dir()  # once: every call rewrites the patched copy, so parallel calls raced with the imports
    out = Parallel(n_jobs=24, verbose=5)(delayed(session)(*j, src) for j in jobs)
    S = pd.DataFrame([o[0] for o in out])
    S.to_parquet(TAB / "008_glm_state.parquet", index=False)
    B = pd.concat([o[1] for o in out if o[1] is not None], ignore_index=True)
    B.to_parquet(TAB / "008_glm_state_beta.parquet", index=False)
    S = S[S.skipped_reason.isna()]
    pd.set_option("display.width", 250)
    w = lambda v: (len(v.dropna()), round(v.mean(), 4), round(stats.wilcoxon(v.dropna()).pvalue, 4)) if v.notna().sum() >= 5 else None
    for c in ("R+", "R-"):
        d = S[S.reward_group == c]
        print(c, {k: w(d[k]) for k in ["auc_glm", "auc_state", "dauc_glm_state", "dll_glm_state", "dauc_glm_ro", "dll_glm_ro",
                                        "dauc_glm_g", "dauc_glm_all", "rho_state_glm_logit", "prho_state_glm_logit", "r2_state_glm",
                                        "b_g", "b_ro", "lick_agree"]})
        print(c, "vs null", {k: w(d[k] - d[f"{k}_null"]) for k in ["dauc_glm_state", "dll_glm_state", "dauc_glm_ro", "dll_glm_ro",
                                                                     "r2_state_glm"]})
        b = B[B.reward_group == c]
        print(b.groupby("feature")[["beta_state", "a_g", "c_g", "cprime_g", "indirect_g", "a_ro", "indirect_ro"]].mean().round(3))


if __name__ == "__main__":
    main()
