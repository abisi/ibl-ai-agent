"""Full cross-mouse session-permutation sweep (user: "do the full sweep
of cross-animal tests" -- extending `079`'s 5-recipient pilot to ALL 75
learning-stage sessions as recipients, both cohorts, both directions --
a fully crossed 2x2 design, not just the R+-recipient half `079` piloted:

- **R+ recipient x R- donor** (`rminus_null`, reward-dependence, as in `079`)
- **R+ recipient x other-R+ donor** (`withinrplus_null`, specificity, as in `079`)
- **R- recipient x R+ donor** (`rplus_null`, NEW -- reward-dependence from
  the other direction)
- **R- recipient x other-R- donor** (`withinrminus_null`, NEW, user request
  2026-09-23: "Should there be within R- null too?" -- tests whether R-
  mice are ALSO individually specific to each other even without reward-
  driven learning. If yes, individuality in this decode is a property of
  being a distinct animal, not of having learned via reward -- it would
  soften how `withinrplus_null` should be read. If no, it strengthens the
  R+ specificity finding.)

Design (alignment, PLS+1SE recipe, per-cell exception handling) is
otherwise IDENTICAL to `079` -- same whisker-trial-index alignment
(confirmed with the user before `079` was built), same estimator, same
"one bad cell can't kill the run" convention.

**Reads `080`'s precomputed feature cache instead of recomputing from raw
spikes per pair** -- the speed optimization the user asked for. At full-
sweep scale (~5600 cells vs `079`'s 355), redundant per-pair feature
extraction (each session touched 70+ times as a donor) would dominate
runtime; this script's pairwise step is pure array slicing + PLS fitting.

**The cache is loaded ONCE into a module-level global before the worker
pool is created, and tasks carry only session IDs (small strings), not
the feature arrays themselves** -- passing large per-session arrays
(each up to ~5MB) as `ProcessPoolExecutor.submit()` arguments would force
them to be re-pickled and piped to a worker on EVERY task that uses them
(a given donor session appears in ~70+ tasks), which would itself become
the bottleneck this precompute step was meant to eliminate. Relying on
fork-based multiprocessing (this project's implicit assumption
throughout -- Linux default, same as every other script here) to give
each worker copy-on-write access to the already-populated global cache
at fork time, at zero serialization cost.

~75 recipients x ~74 donors (37 same-group minus self + 37-38 cross-
group) x 3 targets (free, joint fit) = ~5600 (recipient, donor, null_type)
cells, plus 75 real fits.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
MIN_TRIALS_FOR_REGRESSION = 25
METRICS = ["r2", "pearson", "spearman"]


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_1se_components(X, Y, rng, n_folds=5, grid=COMPONENT_GRID):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    max_comp = max(1, min(30, X.shape[1] - 1, int(X.shape[0] * 0.6)))
    candidates = sorted(set(min(c, max_comp) for c in grid if c <= max_comp)) or [max_comp]
    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    mean_scores, se_scores = [], []
    for n_comp in candidates:
        fold_scores = []
        for tr, te in splits:
            scaler = _floored_scaler()
            scaler.fit(X[tr])
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(scaler.transform(X[tr]), Y[tr])
            Y_pred = pls.predict(scaler.transform(X[te]))
            fold_scores.append(np.mean([r2_score(Y[te, k], Y_pred[:, k]) for k in range(Y.shape[1])]))
        mean_scores.append(np.mean(fold_scores))
        se_scores.append(np.std(fold_scores) / np.sqrt(n_folds))
    best_idx = int(np.argmax(mean_scores))
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    return next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)


def _pls_fit_predict(X_train, Y_train, X_test, n_components):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X_train), Y_train)
    return pls.predict(scaler.transform(X_test))


def _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS):
    from sklearn.model_selection import KFold
    Y_pred_accum = np.zeros_like(Y, dtype=float)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y, dtype=float)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te] = _pls_fit_predict(X[tr], Y[tr], X[te], n_components)
        Y_pred_accum += Y_pred_rep
    return Y_pred_accum / n_repeats


def _score(Y_true, Y_pred):
    from sklearn.metrics import r2_score
    n_targets = Y_true.shape[1]
    return dict(
        r2=[r2_score(Y_true[:, k], Y_pred[:, k]) for k in range(n_targets)],
        pearson=[pearsonr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
        spearman=[spearmanr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
    )


FEAT_CACHE = {}  # populated once in main(), before the worker pool is created -- see module docstring


def process_pair(args: tuple) -> dict:
    """Pure array-slicing + PLS fit -- looks up `recip`/`donor` arrays from
    the module-level FEAT_CACHE (inherited copy-on-write via fork, not
    passed as an argument), so tasks only carry small IDs."""
    recipient_mouse, recipient_sid, donor_mouse, donor_sid, null_type = args
    out = dict(recipient_mouse=recipient_mouse, recipient_sid=recipient_sid, donor_mouse=donor_mouse,
               donor_sid=donor_sid, null_type=null_type, ok=False)

    recip_X, recip_Y_raw = FEAT_CACHE[recipient_sid]["X"], FEAT_CACHE[recipient_sid]["Y_raw"]
    Y_mean, Y_std = recip_Y_raw.mean(axis=0), recip_Y_raw.std(axis=0)
    recip_Y = (recip_Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    if null_type == "real":
        X_pair, Y_pair = recip_X, recip_Y
    elif null_type in ("rminus_null", "rplus_null"):
        # donor's NEURAL ACTIVITY paired with recipient's own target
        donor_X = FEAT_CACHE[donor_sid]["X"]
        m = min(len(donor_X), len(recip_Y))
        X_pair, Y_pair = donor_X[:m], recip_Y[:m]
    elif null_type in ("withinrplus_null", "withinrminus_null"):
        # recipient's own activity paired with a DIFFERENT same-cohort mouse's target
        donor_Y_raw = FEAT_CACHE[donor_sid]["Y_raw"]
        m = min(len(recip_X), len(donor_Y_raw))
        dY_mean, dY_std = donor_Y_raw.mean(axis=0), donor_Y_raw.std(axis=0)
        donor_Y = (donor_Y_raw - dY_mean) / np.where(dY_std > 0, dY_std, 1.0)
        X_pair, Y_pair = recip_X[:m], donor_Y[:m]
    else:
        raise ValueError(null_type)

    if len(X_pair) < MIN_TRIALS_FOR_REGRESSION:
        return out

    rng = np.random.default_rng(abs(hash(f"{recipient_sid}_{donor_sid}_{null_type}")) % (2**31))
    n_1se = select_pls_1se_components(X_pair, Y_pair, rng)
    Y_pred = _pooled_cv_predict(X_pair, Y_pair, n_1se, rng)
    scores = _score(Y_pair, Y_pred)

    out.update(ok=True, n_trials=len(X_pair), **{f"{m}": scores[m] for m in METRICS})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    global FEAT_CACHE
    with open(OUT_DIR / "080_session_features_cache.pkl", "rb") as f:
        FEAT_CACHE = pickle.load(f)  # populated BEFORE the pool is created -- forked workers inherit it COW, free
    with open(OUT_DIR / "065_perfquant_allmice_cache.pkl", "rb") as f:
        meta_cache = pickle.load(f)
    meta = {r["session_id"]: r for r in meta_cache if r["day_stage"] == "learning"}

    rplus_sids = [sid for sid in FEAT_CACHE if meta.get(sid, {}).get("reward_group") == "R+"]
    rminus_sids = [sid for sid in FEAT_CACHE if meta.get(sid, {}).get("reward_group") == "R-"]
    print(f"{len(rplus_sids)} R+ sessions, {len(rminus_sids)} R- sessions usable from cache", flush=True)

    tasks = []
    for rsid in rplus_sids:
        rmouse = meta[rsid]["mouse"]
        tasks.append((rmouse, rsid, None, None, "real"))
        for dsid in rminus_sids:
            tasks.append((rmouse, rsid, meta[dsid]["mouse"], dsid, "rminus_null"))
        for dsid in rplus_sids:
            if dsid == rsid:
                continue
            tasks.append((rmouse, rsid, meta[dsid]["mouse"], dsid, "withinrplus_null"))
    for rsid in rminus_sids:
        rmouse = meta[rsid]["mouse"]
        tasks.append((rmouse, rsid, None, None, "real"))
        for dsid in rplus_sids:
            tasks.append((rmouse, rsid, meta[dsid]["mouse"], dsid, "rplus_null"))
        for dsid in rminus_sids:
            if dsid == rsid:
                continue
            tasks.append((rmouse, rsid, meta[dsid]["mouse"], dsid, "withinrminus_null"))
    print(f"{len(tasks)} total (recipient, donor, null_type) cells", flush=True)

    results = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=110) as ex:
        futures = {ex.submit(process_pair, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            try:
                res = fut.result()
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"  cell {futures[fut]} raised {type(e).__name__}: {e}", flush=True)
                continue
            if res.get("ok"):
                results.append(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} cells done ({n_failed} failed so far)", flush=True)
    print(f"{n_failed} cells raised an exception (of {len(tasks)} total)", flush=True)
    print(f"{len(results)}/{len(tasks)} cells usable", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "081_perfquant_crossmouse_fullsweep_raw.csv", index=False)
    print("saved 081_perfquant_crossmouse_fullsweep_raw.csv")
    print("DONE_081")


if __name__ == "__main__":
    main()
