"""Shared input glm_v5 (user 2026-11-09): glm_v4 with 5 contiguous outer blocks (default; larger test blocks than 10) and,
for importance, the drop in held-out log-likelihood per trial when a predictor group is removed and the model refitted
(drop-and-refit, at the fold's C), next to the shuffled-predictor scores. Otherwise as glm_v4 below.

Former docstring (glm_v4):
Shared input glm_v4: the behaviour GLM of glm_v3 (same sessions, predictors and design matrix: Myriam Hamon's
glm_behavior_v3 loaders through glm_v3_run.glm_module, her file not edited) refitted with the cross-validation chosen
after glm_cv_compare.py (user 2026-11-09: "fix it and rerun the GLM").

Changes from glm_v3:
  - splits: 10 contiguous blocks of whisker trials in time order, GAP whisker trials on each side of the test block left
    out of training, inner CV for C also by contiguous blocks (5); repeated N_REPEATS times with the block boundaries
    shifted by a random offset, so each trial is predicted N_REPEATS times
  - no class weighting: predicted probabilities are calibrated to the session's lick rate (glm_v3 used
    class_weight='balanced', which centred predictions on 0.5 and gave negative pseudo-R2)
  - C chosen by held-out log-loss on 25 values 1e-5..1e3 (glm_v3: AUC on 1e-3..1e3, often at the 1e-3 edge)
  - balanced accuracy thresholds each fold's predictions at its training lick rate (equivalent to the balanced
    decision rule without distorting the probabilities)
  - AUC on each prediction relative to its fold's training lick rate (see scores; raw pooled AUC kept as test_auc_raw)
  - metrics on the pooled out-of-fold predictions of each repeat, then the mean over repeats; the per-trial mean
    prediction over repeats is saved for display (y_pred_proba_oof) and scored separately (*_avg)
  - shuffle importance: per repeat and fold, the predictor group is permuted in training and test trials and the
    model refitted at the fold's C (as glm_v3), N_PERM permutations; delta = full minus shuffled pooled AUC
    (and balanced accuracy, and held-out log-likelihood per trial in nats), averaged over permutations and repeats

Outputs, per session, in Figures/_cache/glm_v4/<mouse>/<mouse>_<session>_<kind> with the glm_v3 file names read by
Figure 1: results_metrics.csv, results.pkl (y_all, y_pred_proba_oof, result), coefficients.csv (mean over folds and
repeats), shuffle_importance.csv; plus predictions.parquet (every repeat: trial, fold, C, p_lick) and
coefficients_folds.csv. all_sessions_summary.csv and config.json at the top.

Run on haas (repo root), single-threaded BLAS:
    OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_v5_run.py [--n-jobs 24] [--mice AB164]
"""

import argparse
import json
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, axel_bisi_root, session_nwbs  # noqa: E402
from glm_cv_compare import block_splits  # noqa: E402
from glm_v3_run import HISTORY, MOVEMENT, SETTINGS, glm_module, glm_source_dir  # noqa: E402
from shared_behaviour import mouse_reference  # noqa: E402

OUT = CACHE_DIR / "glm_v5"
N_FOLDS, N_INNER, N_REPEATS, GAP, N_PERM = 5, 5, 10, 3, 10  # v5: 5 outer folds
C_VALUES = np.logspace(-5, 3, 25)
GROUPS = {**{name: [name] for name in HISTORY + MOVEMENT}, "all_movement": list(MOVEMENT)}


def _fit(Xtr, ytr, C):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000).fit(Xtr, ytr)


def choose_C(Xtr, ytr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        cv = LogisticRegressionCV(Cs=C_VALUES, cv=KFold(N_INNER), penalty="l2", solver="lbfgs", max_iter=2000,
                                  scoring="neg_log_loss", n_jobs=1).fit(Xtr, ytr)
    return float(cv.C_[0])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def scores(y, p, thr, base):
    """test_auc ranks each prediction relative to its fold's training lick rate (logit difference): with contiguous
    blocks the training rate is anti-correlated with the test block's rate, so the raw pooled AUC (test_auc_raw) mixes
    that between-fold offset into the ranking. Balanced accuracy uses the same reference (threshold = training rate)."""
    eps = 1e-12
    ll = np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    ll0 = np.mean(y * np.log(base + eps) + (1 - y) * np.log(1 - base + eps))
    two = 0 < y.mean() < 1
    return {"test_auc": roc_auc_score(y, _logit(p) - _logit(thr)) if two else np.nan,
            "test_auc_raw": roc_auc_score(y, p) if two else np.nan,
            "test_balanced_accuracy": balanced_accuracy_score(y, (p >= thr).astype(int)),
            "test_ll_per_trial": ll, "test_pseudo_r2": 1 - ll / ll0}


def group_columns(names):
    return {g: [i for i, fn in enumerate(names) if any(fn == pat or re.fullmatch(re.escape(pat) + r"_t\d+", fn)
                                                       for pat in pats)] for g, pats in GROUPS.items()}


def load_session(glm, nwb):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(nwb, mode="r", load_namespaces=True) as io:
        if io.read().session_description != "whisker_0":
            return None
    y, pred, ptypes, trials = glm.load_nwb_behavior_predictors(
        nwb_path=nwb, bin_size=SETTINGS["bin_size"], pre_stim_window=SETTINGS["pre_stim_window"],
        include_predictors=HISTORY, use_movement_energy=SETTINGS["use_movement_energy"],
        spont_licks_root=str(axel_bisi_root() / "combined_results"))
    X, names = glm.build_behavior_design_matrix(pred, ptypes["analog_keys"])
    w = (trials["trial_type"] == SETTINGS["trial_type"]).to_numpy()
    X, y, t = np.nan_to_num(X)[w], np.asarray(y)[w].astype(int), trials["start_time"].to_numpy()[w]
    order = np.argsort(t, kind="stable")
    return X[order], y[order], t[order], list(names)


def run_session(mouse, cohort, nwb, src):
    glm = glm_module(src)
    session = Path(nwb).stem
    try:
        loaded = load_session(glm, nwb)
    except Exception as e:  # noqa: BLE001 - reported and skipped, as in glm_v3
        print("skip", session, type(e).__name__, e)
        return None
    if loaded is None:
        return None
    X, y, t, names = loaded
    if len(np.unique(y)) < 2:
        print("skip", session, "single class")
        return None
    n = len(y)
    cols = group_columns(names)
    rng = np.random.default_rng(7)
    offsets = [0] + [int(rng.integers(n)) for _ in range(N_REPEATS - 1)]
    preds, coefs, rep_metrics, shuffle_rows, drop_rows = [], [], [], [], []
    P = np.full((N_REPEATS, n), np.nan)
    THR, BASE = np.full((N_REPEATS, n), np.nan), np.full((N_REPEATS, n), np.nan)
    for rep, off in enumerate(offsets):
        Psh = {g: np.full((N_PERM, n), np.nan) for g in GROUPS}
        Pdrop = {g: np.full(n, np.nan) for g in GROUPS}
        fold = np.full(n, -1)
        Cs = np.full(n, np.nan)
        for k, (tr, te) in enumerate(block_splits(n, N_FOLDS, offset=off, gap=GAP)):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            try:
                C = choose_C(Xtr, y[tr])
            except ValueError:  # an inner block with one class only: glm_v3 default grid midpoint
                C = 1.0
            m = _fit(Xtr, y[tr], C)
            P[rep, te], THR[rep, te], BASE[rep, te] = m.predict_proba(Xte)[:, 1], y[tr].mean(), y[tr].mean()
            fold[te], Cs[te] = k, C
            coefs.append(pd.DataFrame({"repeat": rep, "fold": k, "feature": names, "coefficient": m.coef_[0], "C": C,
                                       "intercept": m.intercept_[0]}))
            perm_rng = np.random.default_rng(1000 * rep + k)
            for g, idx in cols.items():  # drop-and-refit: model without the group's columns, same C
                keep = [i for i in range(Xtr.shape[1]) if i not in set(idx)]
                Pdrop[g][te] = _fit(Xtr[:, keep], y[tr], C).predict_proba(Xte[:, keep])[:, 1]
            for g, idx in cols.items():
                for j in range(N_PERM):
                    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
                    for i in idx:
                        Xtr_s[:, i] = perm_rng.permutation(Xtr_s[:, i])
                        Xte_s[:, i] = perm_rng.permutation(Xte_s[:, i])
                    Psh[g][j, te] = _fit(Xtr_s, y[tr], C).predict_proba(Xte_s)[:, 1]
        ok = np.isfinite(P[rep])
        full = scores(y[ok], P[rep, ok], THR[rep, ok], BASE[rep, ok])
        rep_metrics.append({"repeat": rep, "offset": off, "n_predicted": int(ok.sum()), **full})
        preds.append(pd.DataFrame({"repeat": rep, "trial": np.arange(n), "start_time": t, "y": y, "p_lick": P[rep],
                                   "fold": fold, "C": Cs, "train_lick_rate": THR[rep]}))
        for g in GROUPS:
            dr = scores(y[ok], Pdrop[g][ok], THR[rep, ok], BASE[rep, ok])
            drop_rows.append({"repeat": rep, "group": g,
                              "delta_test_ll_per_trial": full["test_ll_per_trial"] - dr["test_ll_per_trial"],
                              "delta_test_auc": full["test_auc"] - dr["test_auc"]})
            for j in range(N_PERM):
                s = scores(y[ok], Psh[g][j, ok], THR[rep, ok], BASE[rep, ok])
                shuffle_rows.append({"repeat": rep, "perm": j, "group": g,
                                     "delta_test_auc": full["test_auc"] - s["test_auc"],
                                     "delta_test_balanced_accuracy": full["test_balanced_accuracy"] - s["test_balanced_accuracy"],
                                     "delta_test_ll_per_trial": full["test_ll_per_trial"] - s["test_ll_per_trial"]})
    rm = pd.DataFrame(rep_metrics)
    p_avg = np.nanmean(P, 0)
    ok = np.isfinite(p_avg)
    avg = scores(y[ok], p_avg[ok], np.nanmean(THR, 0)[ok], np.nanmean(BASE, 0)[ok])
    metrics = {"mouse_id": mouse, "session_id": session, "cohort": cohort, "n_trials": n, "lick_rate": y.mean(),
               "n_folds": N_FOLDS, "n_repeats": N_REPEATS, "gap": GAP,
               **rm[["test_auc", "test_auc_raw", "test_balanced_accuracy", "test_ll_per_trial", "test_pseudo_r2"]].mean().to_dict(),
               "test_auc_sd_repeats": rm.test_auc.std(),
               **{f"{k}_avg": v for k, v in avg.items()},
               "C_median": float(pd.concat(coefs).C.median())}
    metrics["test_ll"] = metrics["test_ll_per_trial"] * n / N_FOLDS  # glm_v3 convention: summed over one test fold
    cf = pd.concat(coefs, ignore_index=True)
    sh = pd.DataFrame(shuffle_rows)
    d = OUT / mouse
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"{mouse}_{session}_"
    pd.DataFrame([metrics]).to_csv(f"{stem}results_metrics.csv", index=False)
    cf.groupby("feature", sort=False).coefficient.mean().reset_index().to_csv(f"{stem}coefficients.csv", index=False)
    cf.to_csv(f"{stem}coefficients_folds.csv", index=False)
    shm = sh.groupby("group", sort=False)[["delta_test_auc", "delta_test_balanced_accuracy", "delta_test_ll_per_trial"]].mean().reset_index()
    shm.insert(0, "model", "Shuffled " + shm.pop("group"))
    pd.concat([pd.DataFrame([{"model": "Full Model", "delta_test_auc": 0.0, "delta_test_balanced_accuracy": 0.0,
                                "delta_test_ll_per_trial": 0.0}]), shm]
              ).to_csv(f"{stem}shuffle_importance.csv", index=False)
    dm = pd.DataFrame(drop_rows).groupby("group", sort=False)[["delta_test_ll_per_trial", "delta_test_auc"]].mean().reset_index()
    dm.insert(0, "model", "Without " + dm.pop("group"))
    pd.concat([pd.DataFrame([{"model": "Full Model", "delta_test_ll_per_trial": 0.0, "delta_test_auc": 0.0}]), dm]
              ).to_csv(f"{stem}nested_model_comparison.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(f"{stem}predictions.parquet")
    pickle.dump({"y_all": y, "y_pred_proba_oof": p_avg, "start_time": t, "feature_names": names, "result": metrics,
                 "repeat_metrics": rm}, open(f"{stem}results.pkl", "wb"))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--mice", nargs="*")
    a = ap.parse_args()
    ref = mouse_reference()
    ref = ref[ref.learning_category.isin(["good", "moderate"])]
    if a.mice:
        ref = ref[ref.mouse_id.isin(a.mice)]
    src = glm_source_dir()
    jobs = [(r.mouse_id, r.cohort, str(f)) for r in ref.itertuples() for f in session_nwbs(r.mouse_id)]
    print(len(jobs), "NWB files ->", OUT)
    rows = [r for r in Parallel(n_jobs=a.n_jobs, verbose=2)(delayed(run_session)(*j, src) for j in jobs) if r]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    old = OUT / "all_sessions_summary.csv"
    if a.mice and old.exists():  # partial rerun: replace these mice's rows, keep the others
        prev = pd.read_csv(old)
        df = pd.concat([prev[~prev.mouse_id.isin(a.mice)], df], ignore_index=True)
    tmp = OUT / "all_sessions_summary.tmp"
    df.to_csv(tmp, index=False)
    tmp.rename(OUT / "all_sessions_summary.csv")
    json.dump({"n_folds": N_FOLDS, "n_inner": N_INNER, "n_repeats": N_REPEATS, "gap_whisker_trials": GAP,
               "n_perm": N_PERM, "C_values": C_VALUES.tolist(), "class_weight": None, "inner_scoring": "neg_log_loss",
               "groups": GROUPS, "glm_settings": {k: str(v) for k, v in SETTINGS.items()}, "n_sessions": len(df)},
              open(OUT / "config.json", "w"), indent=1)
    print(len(df), "sessions;", df[["test_auc", "test_balanced_accuracy", "test_pseudo_r2"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
