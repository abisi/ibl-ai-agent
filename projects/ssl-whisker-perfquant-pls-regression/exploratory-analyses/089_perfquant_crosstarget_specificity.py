"""Cross-target specificity + PLS weight-vector comparison (user follow-up
to the "drift vs. code" open question this project has flagged repeatedly,
e.g. REPORT.md's Caveat 1 / open item 3): does each target's decoder carry
information specific to ITS OWN behavioral curve, or does one shared
"generic drift" axis explain all three targets about equally well?

Unlike `058`/`070` (which fit ONE joint, 3-output PLS model per session --
components chosen to jointly explain all 3 targets at once, which could
itself partly reflect shared structure by construction), this script fits
**three separate single-output PLS+1SE models per session**, one per
target, each with its own independently-selected component count and its
own out-of-fold pooled-CV predictions `d_k`. This lets each target find
its own best neuron-weighting independently, a fairer test of whether
those weightings converge (shared/generic) or diverge (specific).

Two analyses, same 3 single-output fits:

1. **Cross-target specificity**: build the 3x3 matrix
   `M[i,j] = Pearson(d_i, Y_j)` -- the diagonal (i==j) is each target's own
   real score (comparable in spirit, not number, to `058`'s joint-fit
   score); the off-diagonal is "does target i's decoder also predict
   target j's TRUE curve." If a single shared drift signal explains
   everything, off-diagonal should be close to the diagonal. If coding is
   target-specific, off-diagonal should sit closer to chance.
   **Null/chance reference, stated as an approximation, not exact**: each
   target's own linear-shift null (target i's shuffles scored against
   target i's own shifted curve, this project's standard construction) is
   reused as the chance-level reference band for BOTH the diagonal and
   every off-diagonal cell in that row. This is not a literal permutation
   test of "does d_i predict Y_j" (that would require shuffling the (i,j)
   pairing directly) -- it is the best available same-units reference
   without a second, more expensive null construction, and is presented
   as a visual band, not a p-value, for the off-diagonal cells.

2. **Weight-vector comparison (PLS only)**: cosine similarity between the
   three targets' `coef_` vectors (final full-data single-output PLS fit,
   standardized feature space, one weight per neuron -- same neurons,
   same ordering, directly comparable within a session). High |cosine|
   across a target pair means both targets are best explained by the same
   population axis (shared/generic); low |cosine| means distinct axes
   (specific).

**Pilot scope**: the same 12 "best to worst" example sessions
`059`-`061` already use (evenly spread across the full 88-session
ranking) -- not yet a full-pool sweep. `N_SHUF_NULL=200` (not `058`'s
1000) since this is 3x the single-output null cost of the joint-fit
recipe; still gives a ~0.5% p-value floor.

Run on haas (real compute, per this project's standing convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
N_SHUF_NULL = 200
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5

EXAMPLE_SESSIONS = [
    ("MH070", "MH070_20260121_140848", 1),
    ("AB087", "AB087_20231017_141901", 9),
    ("MH028", "MH028_20250501_104058", 17),
    ("AB164", "AB164_20250422_115457", 25),
    ("MH037", "MH037_20250524_143522", 33),
    ("MH039", "MH039_20250525_112720", 41),
    ("AB119", "AB119_20240731_102619", 48),
    ("AB142", "AB142_20241128_113227", 56),
    ("AB104", "AB104_20240313_145433", 64),
    ("AB094", "AB094_20231211_112445", 72),
    ("MH027", "MH027_20250422_111013", 80),
    ("MH036", "MH036_20250515_111838", 88),
]


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_1se_components(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                               n_folds: int = 5, grid: tuple = COMPONENT_GRID) -> int:
    """Single-output analog of `058`'s `select_pls_components` -- `y` is
    1-D here (one target), not the 3-column block."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    Y = y.reshape(-1, 1)
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
            fold_scores.append(r2_score(Y[te, 0], Y_pred[:, 0]))
        mean_scores.append(np.mean(fold_scores))
        se_scores.append(np.std(fold_scores) / np.sqrt(n_folds))

    best_idx = int(np.argmax(mean_scores))
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    return next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)


def _pls_fit_predict(X_train, y_train, X_test, n_components, return_coef=False):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X_train), y_train.reshape(-1, 1))
    pred = pls.predict(scaler.transform(X_test))[:, 0]
    if return_coef:
        return pred, pls.coef_.ravel()
    return pred


def _pooled_cv_predict(X, y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS):
    from sklearn.model_selection import KFold
    pred_accum = np.zeros(len(y), dtype=float)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        pred_rep = np.empty(len(y), dtype=float)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            pred_rep[te] = _pls_fit_predict(X[tr], y[tr], X[te], n_components)
        pred_accum += pred_rep
    return pred_accum / n_repeats


def null_distribution(X, y, n_components, rng, n_shuf=N_SHUF_NULL):
    n = len(y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    null_pearson = np.full(n_shuf, np.nan)
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, y_shift = X[: n - shift], y[shift:]
        else:
            X_shift, y_shift = X[shift:], y[: n - shift]
        pred_null = _pooled_cv_predict(X_shift, y_shift, n_components, rng, n_repeats=1, n_folds=N_FOLDS)
        null_pearson[s] = pearsonr(y_shift, pred_null)[0]
    return null_pearson


def process_one_session(args: tuple) -> dict:
    mouse, session_id, rank, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, rank=rank, ok=False)
    reward_group = load_reward_group(mouse)
    out["reward_group"] = reward_group

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return out
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    if len(unit_ids) < MIN_UNITS_PER_AREA:
        return out

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    col_std = np.nanstd(X, axis=0)
    X = X[:, col_std > 1e-6]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    n_components, d_stack, coef_stack, null_pearson = {}, [], [], {}
    for k, target in enumerate(TARGETS):
        y = Y[:, k]
        n_1se = select_pls_1se_components(X, y, rng)
        n_components[target] = n_1se
        d_k = _pooled_cv_predict(X, y, n_1se, rng)
        _, coef_k = _pls_fit_predict(X, y, X, n_1se, return_coef=True)
        null_k = null_distribution(X, y, n_1se, rng)
        d_stack.append(d_k)
        coef_stack.append(coef_k)
        null_pearson[target] = null_k

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1], n_components=n_components,
               Y_true=Y, d_stack=np.stack(d_stack, axis=1), coef_stack=np.stack(coef_stack, axis=0),
               null_pearson=null_pearson)
    diag = [pearsonr(out["d_stack"][:, k], Y[:, k])[0] for k in range(3)]
    print(f"{mouse}/{session_id} (rank {rank}/88, n={len(X)}, p={X.shape[1]}): "
          f"n_comp={n_components}, own-target pearson={[round(v, 3) for v in diag]}", flush=True)
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pickle

    tasks = [(mouse, sid, rank, SCRIPTS_DIR) for mouse, sid, rank in EXAMPLE_SESSIONS]
    results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  FAILED: {futures[fut]} -- {e!r}", flush=True)
                continue
            if res.get("ok"):
                results.append(res)
    results.sort(key=lambda r: r["rank"])
    print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)

    with open(OUT_DIR / "089_perfquant_crosstarget_cache.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"cached {len(results)} sessions -> 089_perfquant_crosstarget_cache.pkl", flush=True)

    # --- flat CSV: cross-prediction matrix cells + null reference stats ---
    records = []
    for r in results:
        d_stack, Y_true = r["d_stack"], r["Y_true"]
        for i, train_target in enumerate(TARGETS):
            null_i = r["null_pearson"][train_target]
            null_mean, null_std = float(np.nanmean(null_i)), float(np.nanstd(null_i))
            for j, test_target in enumerate(TARGETS):
                cross_r = pearsonr(d_stack[:, i], Y_true[:, j])[0]
                cross_rho = spearmanr(d_stack[:, i], Y_true[:, j])[0]
                p_vs_null = (1 + np.sum(null_i >= cross_r)) / (1 + len(null_i))
                records.append(dict(
                    mouse=r["mouse"], session_id=r["session_id"], rank=r["rank"], reward_group=r["reward_group"],
                    train_target=train_target, test_target=test_target, is_diagonal=(i == j),
                    n_components=r["n_components"][train_target], n_trials=r["n_trials"], n_units=r["n_units"],
                    cross_pearson=cross_r, cross_spearman=cross_rho,
                    null_pearson_mean=null_mean, null_pearson_std=null_std,
                    above_null=cross_r - null_mean, p_vs_own_null=p_vs_null,
                ))
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "089_perfquant_crosstarget_specificity.csv", index=False)
    print(f"saved 089_perfquant_crosstarget_specificity.csv ({len(df)} rows)", flush=True)

    # --- flat CSV: weight-vector cosine similarity, every target pair ---
    cos_records = []
    for r in results:
        coef = r["coef_stack"]
        for i, ti in enumerate(TARGETS):
            for j, tj in enumerate(TARGETS):
                if j <= i:
                    continue
                cos = float(np.dot(coef[i], coef[j]) / (np.linalg.norm(coef[i]) * np.linalg.norm(coef[j]) + 1e-12))
                cos_records.append(dict(mouse=r["mouse"], session_id=r["session_id"], rank=r["rank"],
                                         reward_group=r["reward_group"], target_a=ti, target_b=tj, cosine=cos))
    cos_df = pd.DataFrame(cos_records)
    cos_df.to_csv(OUT_DIR / "089_perfquant_crosstarget_weightcosine.csv", index=False)
    print(f"saved 089_perfquant_crosstarget_weightcosine.csv ({len(cos_df)} rows)", flush=True)

    if df.empty or cos_df.empty:
        print(f"\nNo usable sessions ({len(results)}/{len(tasks)}) -- skipping summary printout.", flush=True)
        print("DONE_089_COMPUTE")
        return

    print("\n=== cross-target specificity summary (own-target vs cross-target, mean across 12 sessions) ===")
    for i, train_target in enumerate(TARGETS):
        for j, test_target in enumerate(TARGETS):
            sub = df[(df.train_target == train_target) & (df.test_target == test_target)]
            print(f"  {train_target:>18} -> {test_target:<18}: mean cross_pearson={sub.cross_pearson.mean():+.3f}, "
                  f"mean above_null={sub.above_null.mean():+.3f}")
    print("\n=== weight-vector cosine similarity summary ===")
    for (ta, tb), sub in cos_df.groupby(["target_a", "target_b"]):
        print(f"  {ta} vs {tb}: mean cosine={sub.cosine.mean():+.3f}, median={sub.cosine.median():+.3f}, "
              f"range=[{sub.cosine.min():+.3f}, {sub.cosine.max():+.3f}]")
    print("DONE_089_COMPUTE")


if __name__ == "__main__":
    main()
