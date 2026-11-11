"""Cross-validation schemes for the behaviour GLM v3 (user 2026-11-09: "test blockwise CV and compare").

Same sessions (whisker day 0 of reference-sheet learners), same design matrix (glm_behavior_v3.load_nwb_behavior_predictors
and build_behavior_design_matrix, through glm_v3_run.glm_module, Myriam's file not edited) and same model (L2 logistic
regression, class_weight balanced, C chosen by an inner 5-fold CV on AUC over 20 values 1e-3..1e3, standardised on the
training trials). Only the splitting changes:

  random      : stratified 10-fold, trials shuffled (seed 42); outer folds as in glm_v3 (inner CV: sklearn default,
                stratified 5-fold without shuffling)
  random_rep  : the same, repeated with 10 seeds; each trial is predicted 10 times
  block       : 10 contiguous blocks of whisker trials in time order; inner CV also contiguous (5 blocks)
  block_gap   : as block, with the GAP whisker trials on each side of the test block removed from training (the
                lick n-1..n-3 predictors of the test trials then never contain training labels, and vice versa)
  block_rep   : block_gap repeated 10 times with the block boundaries shifted by a random offset (circular), so each
                trial is predicted 10 times by models trained on different neighbours

Per scheme and repeat, metrics are computed on the pooled out-of-fold predictions of the whole session (AUC, balanced
accuracy at p = 0.5, mean log-likelihood per trial and McFadden pseudo-R2 against the training-fold base rate), and as
the mean over folds (AUC only where the fold has both classes). For the repeated schemes, the per-trial mean
prediction over repeats is also scored ("avg" rows).

Outputs (Figures/_cache/glm_cv_compare/): predictions.parquet (one row per session x scheme x repeat x trial: trial
index, start time, y, p_lick, fold, C), metrics.csv, coefs.csv (per fold), config.json.

Run on haas (repo root):
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_cv_compare.py [--n-jobs 24] [--mice AB164]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, axel_bisi_root, session_nwbs  # noqa: E402
from glm_v3_run import HISTORY, SETTINGS, glm_module, glm_source_dir  # noqa: E402
from shared_behaviour import mouse_reference  # noqa: E402

OUT = CACHE_DIR / "glm_cv_compare"
N_FOLDS, N_INNER, N_REPEATS, GAP = 10, 5, 10, 3
C_VALUES = np.logspace(-3, 3, 20)


def fit_predict(Xtr, ytr, Xte, inner_cv):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        cv = LogisticRegressionCV(Cs=C_VALUES, cv=inner_cv, penalty="l2", solver="lbfgs", max_iter=2000,
                                  random_state=42, scoring="roc_auc", class_weight="balanced", n_jobs=1).fit(Xtr, ytr)
        C = float(cv.C_[0])
        m = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42,
                               class_weight="balanced").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], C, m.coef_[0], float(ytr.mean())


def block_splits(n, n_folds, offset=0, gap=0):
    """Contiguous test blocks in time order (circularly shifted by offset); training = the rest minus `gap` trials on
    each side of the test block."""
    order = np.roll(np.arange(n), -offset)
    for test in np.array_split(order, n_folds):
        lo, hi = test.min(), test.max()
        if offset and hi - lo + 1 != len(test):  # block wraps around the session end: keep it contiguous in time
            test = np.sort(test)
        excl = set(test.tolist())
        for t in test:
            excl.update(range(t - gap, t + gap + 1))
        train = np.array([i for i in range(n) if i not in excl])
        yield train, np.sort(test)


def schemes(n, y):
    """(scheme, repeat, list of (train, test), inner cv)"""
    out = [("random", 0, list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y)), N_INNER)]
    for r in range(N_REPEATS):
        out.append(("random_rep", r, list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=100 + r)
                                          .split(np.zeros(n), y)), N_INNER))
    out.append(("block", 0, list(block_splits(n, N_FOLDS)), KFold(N_INNER)))
    out.append(("block_gap", 0, list(block_splits(n, N_FOLDS, gap=GAP)), KFold(N_INNER)))
    rng = np.random.default_rng(7)
    for r in range(N_REPEATS):
        out.append(("block_rep", r, list(block_splits(n, N_FOLDS, offset=int(rng.integers(n)), gap=GAP)),
                    KFold(N_INNER)))
    return out


def scores(y, p, base):
    """Pooled metrics of out-of-fold predictions; base: per-trial training base rate (null model for pseudo-R2)."""
    eps = 1e-12
    ll = np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    ll0 = np.mean(y * np.log(base + eps) + (1 - y) * np.log(1 - base + eps))
    return {"auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y, (p >= 0.5).astype(int)), "ll_per_trial": ll,
            "pseudo_r2": 1 - ll / ll0}


def run_session(mouse, cohort, nwb, src):
    glm = glm_module(src)
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(nwb, mode="r", load_namespaces=True) as io:
        if io.read().session_description != "whisker_0":
            return None
    try:
        y, pred, ptypes, trials = glm.load_nwb_behavior_predictors(
            nwb_path=nwb, bin_size=SETTINGS["bin_size"], pre_stim_window=SETTINGS["pre_stim_window"],
            include_predictors=HISTORY, use_movement_energy=SETTINGS["use_movement_energy"],
            spont_licks_root=str(axel_bisi_root() / "combined_results"))
        X, names = glm.build_behavior_design_matrix(pred, ptypes["analog_keys"])
    except Exception as e:  # noqa: BLE001 - reported and skipped, as in glm_v3
        print("skip", Path(nwb).name, type(e).__name__, e)
        return None
    X = np.nan_to_num(X)
    w = (trials["trial_type"] == SETTINGS["trial_type"]).to_numpy()
    X, y, t = X[w], np.asarray(y)[w].astype(int), trials["start_time"].to_numpy()[w]
    order = np.argsort(t)  # time order (blocks are contiguous in time)
    X, y, t = X[order], y[order], t[order]
    session = Path(nwb).stem
    if len(np.unique(y)) < 2:
        return None
    preds, mets, coefs = [], [], []
    rep_p = {}
    for scheme, rep, splits, inner in schemes(len(y), y):
        p, base, fold, Cs = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), -1), np.full(len(y), np.nan)
        fold_auc = []
        for k, (tr, te) in enumerate(splits):
            if len(np.unique(y[tr])) < 2:
                continue
            try:
                p[te], C, coef, b = fit_predict(X[tr], y[tr], X[te], inner)
            except ValueError:  # inner fold with a single class (block inner CV in a very unbalanced session)
                continue
            base[te], fold[te], Cs[te] = b, k, C
            coefs.append(pd.DataFrame({"session_id": session, "scheme": scheme, "repeat": rep, "fold": k,
                                       "feature": names, "coefficient": coef, "C": C}))
            if 0 < y[te].mean() < 1:
                fold_auc.append(roc_auc_score(y[te], p[te]))
        ok = np.isfinite(p)
        mets.append({"session_id": session, "scheme": scheme, "repeat": rep, "n_trials": len(y),
                     "n_predicted": int(ok.sum()), "lick_rate": y.mean(), "mean_fold_auc": np.mean(fold_auc) if fold_auc else np.nan,
                     "n_fold_auc": len(fold_auc), **scores(y[ok], p[ok], base[ok])})
        preds.append(pd.DataFrame({"session_id": session, "scheme": scheme, "repeat": rep, "trial": np.arange(len(y)),
                                   "start_time": t, "y": y, "p_lick": p, "fold": fold, "C": Cs, "train_base_rate": base}))
        rep_p.setdefault(scheme, []).append((p, base))
    for scheme in ("random_rep", "block_rep"):  # per-trial mean over repeats, scored once
        P = np.nanmean([q for q, _ in rep_p[scheme]], 0)
        B = np.nanmean([b for _, b in rep_p[scheme]], 0)
        ok = np.isfinite(P)
        mets.append({"session_id": session, "scheme": scheme + "_avg", "repeat": -1, "n_trials": len(y),
                     "n_predicted": int(ok.sum()), "lick_rate": y.mean(), **scores(y[ok], P[ok], B[ok])})
    m = pd.DataFrame(mets).assign(mouse_id=mouse, cohort=cohort)
    return pd.concat(preds).assign(mouse_id=mouse, cohort=cohort), m, pd.concat(coefs).assign(mouse_id=mouse, cohort=cohort)


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
    res = [r for r in Parallel(n_jobs=a.n_jobs, verbose=2)(delayed(run_session)(*j, src) for j in jobs) if r is not None]
    OUT.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["predictions.parquet", "metrics.csv", "coefs.csv"]):
        df = pd.concat([r[i] for r in res], ignore_index=True)
        tmp = OUT / (name + ".tmp")
        df.to_parquet(tmp) if name.endswith("parquet") else df.to_csv(tmp, index=False)
        tmp.rename(OUT / name)
    json.dump({"n_folds": N_FOLDS, "n_inner": N_INNER, "n_repeats": N_REPEATS, "gap_whisker_trials": GAP,
               "C_values": C_VALUES.tolist(), "glm_settings": {k: str(v) for k, v in SETTINGS.items()},
               "history": HISTORY, "n_sessions": len(res)}, open(OUT / "config.json", "w"), indent=1)
    print(len(res), "sessions")


if __name__ == "__main__":
    main()
