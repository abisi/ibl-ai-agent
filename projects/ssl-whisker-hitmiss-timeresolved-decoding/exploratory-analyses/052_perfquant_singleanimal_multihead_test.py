"""Single-animal, per-neuron-feature, multi-head regression test (user
request 2026-09-20: "I want a single-animal fully standardized prediction
scheme where one neuron is one feature, trying to predict three heads
i.e. three curves.")

Differs from both prior perfquant validations:
- vs `049` (single-animal, per-neuron features, one target at a time):
  this ALSO standardizes the TARGET (049 left curve values in raw units),
  and predicts all three curves JOINTLY as a multi-output ("three heads")
  regression sharing one input matrix, instead of three independent
  single-output regressions per session.
- vs `050` (cross-animal, area-averaged features): this is single-SESSION
  (no cross-subject pooling, no cross-animal generalization question) and
  uses the FULL per-neuron population vector as features -- only valid
  within one session's own neurons (see `050`'s docstring for why raw
  per-neuron features can't cross animals; that constraint doesn't apply
  here since there's no cross-subject step at all).

Feature space: whole_brain, single window. **Extended 2026-09-20** (user:
"Predict using baseline activity, 2000ms to -10ms before whisker trial
start, to predict learning curves?"): `FEATURE_WINDOW` now defaults to
`WIDE_BASELINE_WINDOW = (-2.000, -0.010)` -- the FULL pre-stimulus period
(not `050`'s narrow 45ms baseline snippet at exactly -2s) -- testing
whether pre-trial neural STATE predicts the curve, as opposed to
`SENSORY_WINDOW` (5-50ms post-stim, same window `049` used), which is
still defined and easy to swap back in for comparison. Not extended to
multiple windows or per-area here (an assumption, not requested this
time -- easy to add later).

"Fully standardized": features go through the usual per-fold
`StandardScaler` (same as every other decode in this project); the three
targets are ALSO z-scored, using each SESSION's own mean/std (not
cross-animal -- there's exactly one animal per session here, so no
between-subject calibration question the way `050` had). This also means
R^2 is not hobbled by the cross-animal miscalibration problem that made
it the wrong metric in `050` -- worth comparing against that result
directly.

"Three heads": one multi-output Ridge (`Ridge` natively accepts a
`(n_trials, 3)` target, fitting 3 independent linear readouts off the
SAME input matrix and regularization strength, in one call) rather than
3 separate single-output models -- the natural linear analog of a shared
"trunk" with three output heads. Cheaper than 3 separate fits, not more
expensive.

Null: full retrain-under-linear-shift (X shifted relative to Y before
BOTH fitting and scoring -- the project's standard, more rigorous
shift-null convention, same as `043`/`049`'s within-session nulls;
affordable here since single-session decoding is much cheaper than
`050`'s cross-subject pooled fits, unlike `050`'s cheaper post-hoc-only
shift null).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)          # matches 026-041's 2026-09-18 update, same window 049 used
WIDE_BASELINE_WINDOW = (-2.000, -0.010)  # user 2026-09-20: "baseline activity, 2000ms to -10ms
                                          # before whisker trial start" -- the FULL pre-stimulus
                                          # period (not a narrow snippet at -2s) -- does pre-trial
                                          # neural STATE predict the curve, as opposed to
                                          # SENSORY_WINDOW's stimulus-evoked response?
SHORT_BASELINE_WINDOW = (-0.055, -0.010)  # width-matched to SENSORY_WINDOW (45ms), immediately
                                           # pre-stimulus -- the conventional "baseline correction"
                                           # window, distinct from WIDE_BASELINE_WINDOW's full 2s span

# FEATURE_MODE selects what X is built from (user 2026-09-20: "Test with
# baseline-corrected sensory activity" -- a third option alongside the
# raw sensory and wide-pre-stimulus-baseline tests already run):
#   "sensory"   -> SENSORY_WINDOW alone (the original test)
#   "baseline"  -> WIDE_BASELINE_WINDOW alone (the "2000ms before" test)
#   "corrected" -> SENSORY_WINDOW minus SHORT_BASELINE_WINDOW, per neuron
#                  per trial -- the classic evoked-response baseline
#                  correction, meant to strip out each trial's ongoing/
#                  tonic rate before looking at the stimulus-evoked
#                  deviation, which should reduce the same shared-slow-
#                  drift confound that inflated WIDE_BASELINE_WINDOW's
#                  null if that confound is really a slow ADDITIVE offset
#                  common to both windows.
FEATURE_MODE = "sensory"  # re-running with the SCALE_FLOOR fix, to verify this window too
DEAD_ZONE = (-0.001, 0.004)      # production dead zone; irrelevant to any window above, harmless either way
N_REPEATS = 3
N_SHUF_NULL = 20
N_WORKERS = 10
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5

TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]  # the 3 "heads", in this fixed order

# Same 20-mouse subset used throughout this thread -- quick correctness
# check before deciding whether to scale to the full 88-session pool.
TEST_MICE = [
    "AB092", "AB120", "AB121", "AB126", "AB127", "AB128", "AB129", "AB130",
    "AB133", "AB138", "AB143", "AB144", "AB147", "AB154", "AB158", "AB159",
    "MH011", "MH014", "MH022", "MH030",
]
USE_FULL_POOL = True  # re-running full-pool baseline with the SCALE_FLOOR fix


SCALE_FLOOR = 1e-3  # Hz -- see _make_multihead_regressor


def _make_multihead_regressor(alpha: float):
    """Ridge behind a per-fold StandardScaler, with the fitted `scale_`
    floored to `SCALE_FLOOR`. Found running FEATURE_MODE='corrected'
    (though it turned out to also affect 'baseline' -- at least one of
    the 88 full-pool sessions hit this too): a column can have fine
    variance across the WHOLE session yet still be near-constant within
    one particular CV fold's training subset purely by chance. Vanilla
    StandardScaler's fitted std for that fold is then a tiny-but-nonzero
    float (not exactly 0, so sklearn's own zero-variance guard doesn't
    engage), so `1/std` explodes for any held-out test value that differs
    even slightly -- R^2 around -1e28, driven by scaled features around
    1e17, while coefficients stayed perfectly normal (verified by hand:
    changing alpha across the whole grid did nothing, confirming this is
    a scaler problem, not a regularization one). A whole-session variance
    pre-filter (`process_one_session`'s `col_std > 1e-6` drop) does NOT
    catch this -- it's a per-fold coincidence, not a property of the full
    column. Flooring the scale directly prevents the blow-up regardless
    of which fold's training subset happens to be near-degenerate.
    `SCALE_FLOOR` is in the same units as X (spike rate, Hz) -- 1e-3 Hz is
    far below any real signal variation (typical rates here span 0-500 Hz)
    and only ever engages for genuinely near-constant columns."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self

    return make_pipeline(_FlooredScaler(), Ridge(alpha=alpha))


def _r2_pearson_spearman_multihead(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Per-head (column) R^2/Pearson/Spearman. Y_true/Y_pred shape (n, 3)."""
    from sklearn.metrics import r2_score
    out = dict(r2=[], pearson=[], spearman=[])
    for k in range(Y_true.shape[1]):
        yt, yp = Y_true[:, k], Y_pred[:, k]
        out["r2"].append(float(r2_score(yt, yp)) if np.var(yt) > 0 else float("nan"))
        pr = pearsonr(yt, yp)[0] if len(yt) > 2 and np.std(yp) > 0 else float("nan")
        sr = spearmanr(yt, yp)[0] if len(yt) > 2 and np.std(yp) > 0 else float("nan")
        out["pearson"].append(float(pr))
        out["spearman"].append(float(sr))
    return out


def select_alpha_multihead(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_folds: int = 5) -> float:
    from sklearn.model_selection import KFold
    RIDGE_ALPHA_GRID = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    best_alpha, best_score = RIDGE_ALPHA_GRID[0], -np.inf
    for a in RIDGE_ALPHA_GRID:
        Y_pred = np.empty_like(Y)
        for tr, te in splits:
            reg = _make_multihead_regressor(a)
            reg.fit(X[tr], Y[tr])
            Y_pred[te] = reg.predict(X[te])
        res = _r2_pearson_spearman_multihead(Y, Y_pred)
        mean_score = np.nanmean(res["r2"])
        if mean_score > best_score:
            best_score, best_alpha = mean_score, a
    return float(best_alpha)


def decode_multihead_pooled(X: np.ndarray, Y: np.ndarray, alpha: float, rng: np.random.Generator,
                             n_repeats: int = N_REPEATS, n_folds: int = 5) -> dict:
    from sklearn.model_selection import KFold
    if len(Y) < n_folds:
        return dict(r2=[float("nan")] * Y.shape[1], pearson=[float("nan")] * Y.shape[1], spearman=[float("nan")] * Y.shape[1])
    r2s, pears, spears = [[] for _ in range(Y.shape[1])], [[] for _ in range(Y.shape[1])], [[] for _ in range(Y.shape[1])]
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            reg = _make_multihead_regressor(alpha)
            reg.fit(X[tr], Y[tr])
            Y_pred[te] = reg.predict(X[te])
        res = _r2_pearson_spearman_multihead(Y, Y_pred)
        for k in range(Y.shape[1]):
            r2s[k].append(res["r2"][k])
            pears[k].append(res["pearson"][k])
            spears[k].append(res["spearman"][k])
    return dict(r2=[float(np.nanmean(x)) for x in r2s],
                pearson=[float(np.nanmean(x)) for x in pears],
                spearman=[float(np.nanmean(x)) for x in spears])


def linear_shift_null_multihead(X: np.ndarray, Y: np.ndarray, alpha: float, rng: np.random.Generator, n_shuf: int,
                                 n_repeats: int = 1, n_folds: int = 5) -> dict:
    """Full retrain-under-shift null -- X shifted relative to Y before
    both fitting and scoring, same convention as `049`'s within-session
    shift-nulls (unlike `050`'s cheaper post-hoc-only variant)."""
    n = len(Y)
    min_shift = max(1, int(MIN_SHIFT_FRAC * n))
    max_shift = max(min_shift, int(MAX_SHIFT_FRAC * n))
    per_head = {m: [[] for _ in range(Y.shape[1])] for m in ("r2", "pearson", "spearman")}
    for _ in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        res = decode_multihead_pooled(X_shift, Y_shift, alpha, rng, n_repeats=n_repeats, n_folds=n_folds)
        for m in ("r2", "pearson", "spearman"):
            for k in range(Y.shape[1]):
                per_head[m][k].append(res[m][k])
    return per_head


def process_one_session(args: tuple) -> dict:
    mouse, session_id, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    out = dict(mouse=mouse, session_id=session_id, log=[], ok=False, reward_group=None)
    reward_group = load_reward_group(mouse)
    if reward_group is None or reward_group not in ("R+", "R-"):
        out["log"].append("no usable reward_group")
        return out
    out["reward_group"] = reward_group

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None:
        out["log"].append("no curve data")
        return out

    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    if len(unit_ids) < MIN_UNITS_PER_AREA:
        out["log"].append(f"only {len(unit_ids)} units (< {MIN_UNITS_PER_AREA})")
        return out
    if len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        out["log"].append(f"only {len(targets_df)} trials (< {MIN_TRIALS_FOR_REGRESSION})")
        return out

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    if FEATURE_MODE == "sensory":
        X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    elif FEATURE_MODE == "baseline":
        X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [WIDE_BASELINE_WINDOW], dead_zone=DEAD_ZONE)[0]
    elif FEATURE_MODE == "corrected":
        sensory_mat, short_baseline_mat = sliding_bin_population_matrices(
            unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW, SHORT_BASELINE_WINDOW], dead_zone=DEAD_ZONE,
        )
        X = sensory_mat - short_baseline_mat
    else:
        raise ValueError(f"unknown FEATURE_MODE {FEATURE_MODE!r}")

    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]

    # Drop near-zero-variance columns (found running FEATURE_MODE="corrected":
    # sensory-minus-baseline leaves many otherwise-silent neurons at ~0 for
    # nearly every trial; when a training fold sees that column as EXACTLY
    # constant, StandardScaler's fitted std is a tiny nonzero float, so
    # 1/std explodes -- fine for training data (which sits exactly at the
    # mean) but catastrophic for a held-out test row that differs even
    # slightly, producing scaled features ~1e17 and R^2 around -1e28. Not
    # a regularization issue (changing alpha did nothing) -- it happens in
    # the scaler, before Ridge ever sees the data. Applied to every
    # FEATURE_MODE for robustness, though raw single-window rates rarely
    # trigger this since they're almost never exactly constant session-wide.
    col_std = np.nanstd(X, axis=0)
    keep = col_std > 1e-6
    if not keep.all():
        X = X[:, keep]

    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION:
        out["log"].append(f"only {len(X)} trials after dropping NaN-feature rows")
        return out

    # "Fully standardized": z-score each of the 3 targets using THIS
    # session's own mean/std (single-animal -- no cross-subject
    # calibration question the way 050 had).
    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y_std_safe = np.where(Y_std > 0, Y_std, 1.0)
    Y = (Y_raw - Y_mean) / Y_std_safe

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    alpha = select_alpha_multihead(X, Y, rng)
    real = decode_multihead_pooled(X, Y, alpha, rng, n_repeats=N_REPEATS)
    null = linear_shift_null_multihead(X, Y, alpha, rng, n_shuf=N_SHUF_NULL)

    out["ok"] = True
    out["n_trials"] = len(X)
    out["n_units"] = len(unit_ids)
    out["alpha"] = alpha
    for k, target in enumerate(TARGETS):
        out[f"{target}_real_r2"] = real["r2"][k]
        out[f"{target}_real_pearson"] = real["pearson"][k]
        out[f"{target}_real_spearman"] = real["spearman"][k]
        out[f"{target}_null_r2_mean"] = float(np.nanmean(null["r2"][k]))
        out[f"{target}_null_r2_std"] = float(np.nanstd(null["r2"][k]))
        out[f"{target}_null_pearson_mean"] = float(np.nanmean(null["pearson"][k]))
        out[f"{target}_null_pearson_std"] = float(np.nanstd(null["pearson"][k]))
        out[f"{target}_null_spearman_mean"] = float(np.nanmean(null["spearman"][k]))
        out[f"{target}_null_spearman_std"] = float(np.nanstd(null["spearman"][k]))
        out["log"].append(
            f"[{target}] r2={real['r2'][k]:.3f} (null={np.nanmean(null['r2'][k]):.3f}) | "
            f"pearson={real['pearson'][k]:.3f} (null={np.nanmean(null['pearson'][k]):.3f}) | "
            f"spearman={real['spearman'][k]:.3f} (null={np.nanmean(null['spearman'][k]):.3f})"
        )
    return out


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")

    tasks = []
    if USE_FULL_POOL:
        hm = hitmiss_session_list(sessions_tbl)
        learning = hm[hm["day_stage"] == "learning"]
        for _, r in learning.iterrows():
            tasks.append((r["subject_id"], r["session_id"], SCRIPTS_DIR))
        print(f"USE_FULL_POOL=True: {len(tasks)} learning-stage has_ephys sessions")
    else:
        for mouse in TEST_MICE:
            sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                                 & (sessions_tbl["has_ephys"])]
            if len(sess) == 0:
                print(f"{mouse}: no learning-stage (whisker_0) has_ephys session, skipping")
                continue
            tasks.append((mouse, sess["session_id"].iloc[0], SCRIPTS_DIR))

    print(f"{len(tasks)} sessions, single-animal per-neuron multi-head decode ({N_WORKERS} workers)...")
    records = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if not res["ok"]:
                print(f"  {res['mouse']}: SKIPPED -- {res['log'][0] if res['log'] else '?'}")
                continue
            print(f"\n=== {res['mouse']} / {res['session_id']} ({res['reward_group']}): "
                  f"{res['n_trials']} trials, {res['n_units']} units, alpha={res['alpha']:.1e} ===")
            for line in res["log"]:
                print(f"  {line}")
            records.append(res)

    df = pd.DataFrame(records)
    out_csv = OUT_DIR / "052_perfquant_singleanimal_multihead_test.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    if len(df) == 0:
        print("no sessions decodable -- nothing to summarize")
        return

    print("\n=== summary across sessions (single-animal, standardized target, multi-head) ===")
    for target in TARGETS:
        for metric in ("r2", "pearson", "spearman"):
            real_col = f"{target}_real_{metric}"
            null_mean_col, null_std_col = f"{target}_null_{metric}_mean", f"{target}_null_{metric}_std"
            above = (df[real_col] > df[null_mean_col] + df[null_std_col]).sum()
            print(f"{target} [{metric}]: n={len(df)}, real mean={df[real_col].mean():.3f}, "
                  f"null mean={df[null_mean_col].mean():.3f}, {above}/{len(df)} sessions above null+1sd")


if __name__ == "__main__":
    main()
