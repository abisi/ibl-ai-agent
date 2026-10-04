"""Cross-animal, area-average-FR generalization test for the perfquant
curve-value regression target (user request 2026-09-20, following up on
`049_perfquant_curve_regression_test.py`'s within-session validation):

"Use both baseline and sensory window activity, baseline is 2 seconds
before start_time. Cross-validated by testing area-average FR across
subjects. Report generalization across animals and consistency of
predictions between animals across areas. Separate results per cohorts
and compare fit means between cohorts. Also report Pearson and Spearman.
Start with whole-brain first. Use raw learning curves, not residuals. Do
it with lags (-20,+20 trials). Do some test with example figures on a
grid before a bigger run."

This is that first, smaller "grid test" -- same 20-mouse `TEST_MICE`
subset `043`/`049` used (9 R+ / 11 R-, checked before writing this), NOT
yet the full 88-session sweep or any area level beyond whole_brain. NOT a
result to report as final -- a plausibility check on a genuinely different
validation design (cross-SUBJECT generalization, not within-session CV)
before deciding whether it's worth scaling up.

**Feature space, deliberately low-dimensional and cross-session-
comparable**: unlike `024`/`049`'s per-unit population vector (not
comparable across sessions -- different neurons every time), this uses
each session's AREA-AVERAGE firing rate (mean across every QC-passing
unit) in two windows: `SENSORY_WINDOW` (5-50ms post-stim, current
pipeline convention) and a new `BASELINE_WINDOW` (2000-1955ms
*before* start_time -- "baseline is 2 seconds before start_time"; width
matched to `SENSORY_WINDOW`'s 45ms since the user didn't specify a width,
noted here as an assumption, not a locked convention). Two scalar
features per trial at the whole_brain level -- this is what makes
leave-one-subject-out generalization possible at all: "mean whole-brain
firing rate in this window" means the same thing for any two sessions,
regardless of how many or which neurons each one recorded.

**Extended 2026-09-20** (user feedback on the first run's flat, near-
constant example predictions -- "How to improve that?" / "Aren't you
using the firing rates of different neurons? Use a multifeatures
regression where each neuron is a feature."): two corrections. (1)
Standardizing the TARGET (as first proposed) was wrong -- the curve
values are already a bounded, comparable 0-1 (or -1..1) probability
across animals, nothing to recalibrate there. The actual animal-specific
scale mismatch is in the INPUT firing rates (arbitrary units, different
baseline per animal) -- so features, not the target, are now z-scored
per subject before pooling for training. (2) Raw per-NEURON features
can't be used here at all (unlike `049`'s within-session validation,
where they're legitimate) -- there's no correspondence between "neuron
12" in one mouse's session and another's, so a per-neuron feature vector
from one animal is meaningless to a model trained on a different
animal's neurons. The portable middle ground is per-AREA (`AREA_COL`)
average rate: genuinely multi-feature (~2 x n_common_areas, up from 2),
still means the same thing across any two sessions that both recorded
that area.

**Cross-validation**: leave-one-subject-out (equivalently leave-one-
session-out here, since day-0 is one session per mouse), done SEPARATELY
within each cohort (R+ subjects only train/test on other R+ subjects, and
likewise R-) -- pooling cohorts would confound cohort identity with
whatever's being decoded. Ridge (same `_make_regressor`/
`select_fixed_alpha_pooled` as `049`, alpha selected on the pooled
training set) fit on every OTHER same-cohort subject's trials, scored on
the held-out subject's own trials via R^2, Pearson r, and Spearman rho.

**Lag cross-correlation (-20 to +20 trials)**: for each held-out subject,
correlate that subject's held-out predicted sequence against its true
sequence at trial-lags -20..+20 (trial-index lag, not time). A real,
same-trial coupling should peak sharply at lag 0 and decay quickly; a
pure shared-slow-drift artifact would instead stay broad and roughly flat
across many lags (both signals are smooth in their own right, so they'd
correlate at many nearby lags regardless of true trial-by-trial
correspondence) -- this is the cheaper, non-destructive alternative to
`049`'s hard residualization, which collapsed R^2 toward zero along with
the drift.

Raw curve values only (not the residualized variants) -- per this
request's explicit "use raw learning curves, not residuals".
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, ttest_ind

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)     # matches 026-041's 2026-09-18 update
BASELINE_WINDOW = (-2.000, -1.955)  # "2 seconds before start_time" -- width matched to SENSORY_WINDOW (45ms), an assumption
DEAD_ZONE = (-0.001, 0.004)         # production dead zone; irrelevant to BASELINE_WINDOW, harmless to pass either way
LAGS = list(range(-20, 21))
N_WORKERS = 10
MIN_TRIALS_FOR_REGRESSION = 25
MIN_AREA_COVERAGE_FRAC = 0.5  # an area must have >=5 units in at least this fraction of sessions to be a "common" feature

TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]

COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}  # project convention (memory: ssl_cross_area_rplus_rminus_colors)

# Same 20-mouse subset 043/049 used -- 9 R+ / 11 R- (checked before writing
# this script), enough subjects per cohort for a first-pass LOSO grid-test.
TEST_MICE = [
    "AB092", "AB120", "AB121", "AB126", "AB127", "AB128", "AB129", "AB130",
    "AB133", "AB138", "AB143", "AB144", "AB147", "AB154", "AB158", "AB159",
    "MH011", "MH014", "MH022", "MH030",
]

# 2026-09-20: "Once you're run, run the full thing" -- once the area-level
# feature fix is verified on TEST_MICE, scale to every curve-available
# learning-stage session (89 total, 88 with a curve file per 049's
# validation), not just the 20-mouse subset.
USE_FULL_POOL = True


AREA_COL = "area_group"  # multi-feature regression (user 2026-09-20: "use a multifeatures
                          # regression where each neuron is a feature") -- per-NEURON features
                          # can't transfer across animals at all (no correspondence between
                          # "neuron 12" in one mouse's session and another's), unlike the
                          # within-session validation (049) where they're legitimate. Per-AREA
                          # average rate is the portable middle ground: richer than one
                          # whole-brain scalar, still means the same thing across any two
                          # sessions that both recorded that area.


def process_one_session(args: tuple) -> dict:
    """Per session: per-AREA (`AREA_COL`) average FR (baseline + sensory,
    one pair of features per area with enough units in THIS session) and
    the 3 raw perfquant targets, for every whisker trial. Returns a plain
    dict (not a DataFrame -- keeps this picklable/simple across the
    ProcessPoolExecutor boundary)."""
    mouse, session_id, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, area_units, areas_with_enough_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    reward_group = load_reward_group(mouse)
    if reward_group is None or reward_group not in ("R+", "R-"):
        return dict(mouse=mouse, session_id=session_id, ok=False, reason="no usable reward_group")

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None:
        return dict(mouse=mouse, session_id=session_id, ok=False, reason="no curve data")
    if len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return dict(mouse=mouse, session_id=session_id, ok=False, reason=f"only {len(targets_df)} trials")

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    areas_here = areas_with_enough_units(session_id, AREA_COL, area_labels)

    area_features = {}
    for area in areas_here:
        unit_ids = area_units(session_id, AREA_COL, area, area_labels)
        baseline_mat = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [BASELINE_WINDOW], dead_zone=DEAD_ZONE)[0]
        sensory_mat = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
        area_features[area] = dict(baseline=np.nanmean(baseline_mat, axis=1), sensory=np.nanmean(sensory_mat, axis=1))

    return dict(
        mouse=mouse, session_id=session_id, ok=True, reward_group=reward_group,
        n_trials=len(targets_df), areas_available=areas_here, area_features=area_features,
        **{t: targets_df[t].to_numpy() for t in TARGETS},
    )


N_SHUF_NULL = 20
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5  # matches project's existing shift-null convention


def _r2_pearson_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float, float]:
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)) if np.var(y_true) > 0 else float("nan")
    pear_r, pear_p = pearsonr(y_true, y_pred) if len(y_true) > 2 else (float("nan"), float("nan"))
    spear_r, spear_p = spearmanr(y_true, y_pred) if len(y_true) > 2 else (float("nan"), float("nan"))
    return r2, float(pear_r), float(pear_p), float(spear_r), float(spear_p)


def loso_generalization(pooled: pd.DataFrame, cohort: str, target: str, feature_cols: list[str],
                         rng: np.random.Generator) -> tuple[list[dict], dict, list[dict]]:
    """Leave-one-subject-out within one cohort. Returns (per-subject
    records, pooled-across-subjects held-out predictions dict for the
    grid figure's scatter panel, per-subject example trial-sequences for
    the single-session example figure).

    Extended 2026-09-20 (user: "Describe train and test performance...
    Show performance distributions relative to null distributions"):
    - TRAIN performance: an extra pooled-CV pass over the training pool
      itself (other subjects only, never touching the held-out subject),
      at the SAME alpha `select_fixed_alpha_pooled` already chose --
      answers "how well does this feature set fit at all, across several
      animals" separately from "does it generalize to a NEW animal" (the
      held-out/test number).
    - NULL: `N_SHUF_NULL` linear-shift replicates of the held-out
      subject's own TRUE sequence (same `MIN_SHIFT_FRAC`/`MAX_SHIFT_FRAC`
      convention as `049`'s within-session shift-null), scored against
      the SAME (unchanged, already-computed) real held-out predictions.
      This is a cheaper, post-hoc variant of the project's usual
      shift-null (no retraining per shuffle) that specifically tests
      whether the model's predictions align with the held-out subject's
      curve at its REAL timing, or would match about as well against an
      arbitrary temporal offset of that same (autocorrelated) curve --
      i.e., it tests temporal specificity of an already-fit cross-animal
      model, not a full re-fit-under-shuffle null."""
    from ssl_timeresolved_decoding import _drop_nan_rows, _make_regressor, select_fixed_alpha_pooled
    from sklearn.model_selection import KFold

    sub = pooled[pooled["reward_group"] == cohort]
    subjects = sub["mouse"].unique()
    records = []
    all_true, all_pred, all_mouse = [], [], []
    examples = []
    for held_out in subjects:
        train = sub[sub["mouse"] != held_out]
        test = sub[sub["mouse"] == held_out].sort_values("trial_index")
        X_train = train[feature_cols].to_numpy()
        y_train = train[target].to_numpy()
        X_test = test[feature_cols].to_numpy()
        y_test = test[target].to_numpy()
        trial_index_test = test["trial_index"].to_numpy()
        X_train, y_train = _drop_nan_rows(X_train, y_train)
        valid_test = ~np.isnan(X_test).any(axis=1)
        X_test, y_test, trial_index_test = X_test[valid_test], y_test[valid_test], trial_index_test[valid_test]
        if len(X_train) < MIN_TRIALS_FOR_REGRESSION or len(X_test) < 5:
            continue
        alpha = select_fixed_alpha_pooled(X_train, y_train, rng)
        reg = _make_regressor(alpha)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2, pear_r, pear_p, spear_r, spear_p = _r2_pearson_spearman(y_test, y_pred)

        # Train performance: pooled-CV OOF predictions over the training
        # pool itself, at the already-chosen alpha (cheap -- one more
        # K-fold pass, no extra alpha search).
        seed = int(rng.integers(0, 2**31 - 1))
        y_train_oof = np.empty(len(y_train))
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X_train):
            reg_cv = _make_regressor(alpha)
            reg_cv.fit(X_train[tr], y_train[tr])
            y_train_oof[te] = reg_cv.predict(X_train[te])
        train_r2, train_pear_r, _, train_spear_r, _ = _r2_pearson_spearman(y_train, y_train_oof)

        lag_corrs = {}
        n = len(y_pred)
        for lag in LAGS:
            if lag >= 0:
                a, b = y_pred[: n - lag] if lag > 0 else y_pred, y_test[lag:]
            else:
                a, b = y_pred[-lag:], y_test[: n + lag]
            if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
                lag_corrs[lag] = float(pearsonr(a, b)[0])
            else:
                lag_corrs[lag] = float("nan")

        # Null: shift the TEST subject's own true sequence, rescore the
        # SAME real predictions against it -- see docstring.
        null_r2s, null_pears, null_spears = [], [], []
        min_shift = max(1, int(MIN_SHIFT_FRAC * n))
        max_shift = max(min_shift, int(MAX_SHIFT_FRAC * n))
        for _ in range(N_SHUF_NULL):
            shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
            if rng.random() < 0.5:
                y_true_shift, y_pred_shift = y_test[shift:], y_pred[: n - shift]
            else:
                y_true_shift, y_pred_shift = y_test[: n - shift], y_pred[shift:]
            nr2, npr, _, nsr, _ = _r2_pearson_spearman(y_true_shift, y_pred_shift)
            null_r2s.append(nr2)
            null_pears.append(npr)
            null_spears.append(nsr)

        records.append(dict(
            mouse=held_out, reward_group=cohort, target=target, n_train_trials=len(X_train), n_test_trials=len(X_test),
            alpha=alpha, r2=r2, pearson_r=pear_r, pearson_p=pear_p, spearman_r=spear_r, spearman_p=spear_p,
            train_r2=train_r2, train_pearson_r=train_pear_r, train_spearman_r=train_spear_r,
            null_r2_mean=float(np.nanmean(null_r2s)), null_r2_std=float(np.nanstd(null_r2s)),
            null_pearson_mean=float(np.nanmean(null_pears)), null_pearson_std=float(np.nanstd(null_pears)),
            null_spearman_mean=float(np.nanmean(null_spears)), null_spearman_std=float(np.nanstd(null_spears)),
            **{f"lag_{lag:+d}": v for lag, v in lag_corrs.items()},
        ))
        examples.append(dict(mouse=held_out, reward_group=cohort, target=target,
                              trial_index=trial_index_test, y_true=y_test, y_pred=y_pred))
        all_true.append(y_test)
        all_pred.append(y_pred)
        all_mouse.append(np.full(len(y_test), held_out))

    pooled_preds = dict(
        y_true=np.concatenate(all_true) if all_true else np.array([]),
        y_pred=np.concatenate(all_pred) if all_pred else np.array([]),
        mouse=np.concatenate(all_mouse) if all_mouse else np.array([]),
    )
    return records, pooled_preds, examples


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
        print(f"USE_FULL_POOL=True: {len(tasks)} learning-stage has_ephys sessions found "
              f"(curve availability checked per-session in process_one_session)")
    else:
        for mouse in TEST_MICE:
            sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                                 & (sessions_tbl["has_ephys"])]
            if len(sess) == 0:
                print(f"{mouse}: no learning-stage (whisker_0) has_ephys session, skipping")
                continue
            tasks.append((mouse, sess["session_id"].iloc[0], SCRIPTS_DIR))

    print(f"{len(tasks)} sessions, extracting per-{AREA_COL} baseline+sensory average FR ({N_WORKERS} workers)...")
    session_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if not res["ok"]:
                print(f"  {res['mouse']}: SKIPPED -- {res['reason']}")
                continue
            print(f"  {res['mouse']} / {res['session_id']} ({res['reward_group']}): {res['n_trials']} trials, "
                  f"{len(res['areas_available'])} areas with >=5 units")
            session_results.append(res)

    # Common areas: present with enough units in most sessions -- user
    # 2026-09-20 ("multifeatures regression where each neuron is a
    # feature") -- per-area average rate is the portable analog (see
    # `process_one_session`'s comment on why raw per-neuron features can't
    # transfer across animals at all). Requiring EVERY session to have the
    # SAME area (a strict intersection) found ZERO areas -- probe
    # placement varies too much mouse to mouse for that (checked directly:
    # even the best-covered areas, Motor/Striatum, are only in 18/20 of
    # the TEST_MICE sessions). Relaxed to majority coverage instead
    # (`MIN_AREA_COVERAGE_FRAC`); a session missing a "common" area gets
    # NaN for that area's features, which `_drop_nan_rows`/the test-side
    # NaN mask already handle gracefully (that session drops out of
    # training/testing for THAT feature set, not a hard failure).
    area_session_count = pd.Series(
        [a for res in session_results for a in res["areas_available"]]
    ).value_counts()
    min_sessions = max(1, int(round(MIN_AREA_COVERAGE_FRAC * len(session_results))))
    common_areas = sorted(area_session_count[area_session_count >= min_sessions].index.tolist())
    print(f"\narea coverage across {len(session_results)} sessions:\n{area_session_count.to_string()}")
    print(f"{len(common_areas)} areas at >={MIN_AREA_COVERAGE_FRAC:.0%} coverage (>={min_sessions}/{len(session_results)} sessions): {common_areas}")
    feature_cols_raw = [f"{w}_{a}" for a in common_areas for w in ("baseline", "sensory")]

    rows = []
    for res in session_results:
        n = res["n_trials"]
        for i in range(n):
            row = dict(mouse=res["mouse"], session_id=res["session_id"], reward_group=res["reward_group"], trial_index=i,
                       **{t: res[t][i] for t in TARGETS})
            for a in common_areas:
                if a in res["area_features"]:
                    row[f"baseline_{a}"] = res["area_features"][a]["baseline"][i]
                    row[f"sensory_{a}"] = res["area_features"][a]["sensory"][i]
                else:
                    row[f"baseline_{a}"] = np.nan
                    row[f"sensory_{a}"] = np.nan
            rows.append(row)
    pooled = pd.DataFrame(rows)

    # Standardize FEATURES per subject (NOT the target -- user 2026-09-20:
    # "they are already bounded within 0 and 1" -- the curve values are
    # already a comparable, bounded probability across animals; it's the
    # raw firing-rate features that have an animal-specific baseline/scale,
    # so THOSE are what needs standardizing before cross-animal pooling).
    #
    # Impute missing-area columns to 0 AFTER z-scoring (found running this:
    # with 7 majority-covered common areas but each individual session
    # only recording a subset of them -- e.g. one session might have
    # Motor+Striatum+Thalamus but not Hippocampus+Midbrain -- dropping any
    # row with an NaN feature, the previous behavior, dropped EVERY trial
    # in EVERY session, since no session had all 7 simultaneously). 0 is a
    # semantically reasonable fill here specifically because it comes
    # AFTER z-scoring: for areas a session DOES have, 0 means "this
    # subject's own typical level"; for areas it structurally lacks, 0 is
    # a neutral, non-informative placeholder Ridge can effectively learn
    # to downweight, rather than a fabricated real value. Caveat, disclosed
    # not silently assumed: which areas a session has is itself
    # nonrandom (probe placement), so this leans on the assumption that
    # missingness pattern isn't itself confounded with the target --
    # reasonable for a validation pass, not yet a production guarantee.
    feature_cols = [f"{c}_z" for c in feature_cols_raw]
    for c, cz in zip(feature_cols_raw, feature_cols):
        z = pooled.groupby("mouse")[c].transform(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0.0)
        pooled[cz] = z.fillna(0.0)

    print(f"\npooled trial table: {len(pooled)} trials across {pooled['mouse'].nunique()} subjects "
          f"({(pooled.groupby('mouse')['reward_group'].first() == 'R+').sum()} R+ / "
          f"{(pooled.groupby('mouse')['reward_group'].first() == 'R-').sum()} R-), "
          f"{len(feature_cols)} features ({len(common_areas)} areas x 2 windows, per-subject z-scored)")
    if USE_FULL_POOL:
        print("NOTE -- is that all data? Yes: USE_FULL_POOL=True, every curve-available learning-stage session.")
    else:
        print(f"NOTE -- is that all data? No: this is still the {len(TEST_MICE)}-mouse `TEST_MICE` subset, "
              f"not the full pool. 049's earlier validation found 88 learning-stage sessions with curve data "
              f"available in total; this cross-animal test has not yet been scaled to all of them.")

    rng = np.random.default_rng(20260920)
    all_records = []
    all_examples = []
    pooled_preds_by_target_cohort = {}
    for target in TARGETS:
        for cohort in ("R+", "R-"):
            records, preds, examples = loso_generalization(pooled, cohort, target, feature_cols, rng)
            all_records.extend(records)
            all_examples.extend(examples)
            pooled_preds_by_target_cohort[(target, cohort)] = preds
            if records:
                r2s = [r["r2"] for r in records]
                train_r2s = [r["train_r2"] for r in records]
                print(f"[{target}, {cohort}] n_subjects={len(records)} | "
                      f"TRAIN (pooled-CV, other subjects only) mean R2={np.nanmean(train_r2s):.3f} | "
                      f"TEST (held-out new subject) mean R2={np.nanmean(r2s):.3f} "
                      f"(std={np.nanstd(r2s):.3f}, range=[{np.nanmin(r2s):.3f}, {np.nanmax(r2s):.3f}])")

    df = pd.DataFrame(all_records)
    out_csv = OUT_DIR / "050_perfquant_crossanimal_arealevel_test.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    if len(df) == 0:
        print("no subjects were decodable -- nothing further to report")
        return

    METRICS = ["r2", "pearson_r", "spearman_r"]
    METRIC_LABELS = {"r2": "R2", "pearson_r": "Pearson r", "spearman_r": "Spearman rho"}

    print("\n=== cohort-mean comparison (R+ vs R-), per target x metric ===")
    cohort_stats = []
    for target in TARGETS:
        sub = df[df["target"] == target]
        for metric in METRICS:
            rplus = sub[sub["reward_group"] == "R+"][metric].dropna()
            rminus = sub[sub["reward_group"] == "R-"][metric].dropna()
            if len(rplus) < 2 or len(rminus) < 2:
                continue
            t_stat, t_p = ttest_ind(rplus, rminus, equal_var=False)
            u_stat, u_p = mannwhitneyu(rplus, rminus, alternative="two-sided")
            print(f"{target} [{METRIC_LABELS[metric]}]: R+ mean={rplus.mean():.3f}+-{rplus.sem():.3f} (n={len(rplus)}), "
                  f"R- mean={rminus.mean():.3f}+-{rminus.sem():.3f} (n={len(rminus)}) | "
                  f"Welch t={t_stat:.2f}, p={t_p:.3f} | Mann-Whitney U={u_stat:.1f}, p={u_p:.3f}")
            cohort_stats.append(dict(target=target, metric=metric, rplus_mean=rplus.mean(), rplus_sem=rplus.sem(),
                                      n_rplus=len(rplus), rminus_mean=rminus.mean(), rminus_sem=rminus.sem(),
                                      n_rminus=len(rminus), welch_t=t_stat, welch_p=t_p, mwu_u=u_stat, mwu_p=u_p))
    pd.DataFrame(cohort_stats).to_csv(OUT_DIR / "050_perfquant_cohort_comparison.csv", index=False)

    print("\n=== real vs null, per target x metric (pooled both cohorts) ===")
    for target in TARGETS:
        sub = df[df["target"] == target]
        for metric in METRICS:
            # metric->null column name mapping (r2->null_r2_*, pearson_r->null_pearson_*, spearman_r->null_spearman_*)
            base = {"r2": "r2", "pearson_r": "pearson", "spearman_r": "spearman"}[metric]
            null_mean_col, null_std_col = f"null_{base}_mean", f"null_{base}_std"
            above = (sub[metric] > sub[null_mean_col] + sub[null_std_col]).sum()
            print(f"{target} [{METRIC_LABELS[metric]}]: real mean={sub[metric].mean():.3f}, "
                  f"null mean={sub[null_mean_col].mean():.3f} | {above}/{len(sub)} subjects above null+1sd")

    # --- grid figure: rows = targets, columns = {scatter, lag-correlation} ---
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(10, 4 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        ax_scatter, ax_lag = axes[row_i]
        for cohort in ("R+", "R-"):
            preds = pooled_preds_by_target_cohort[(target, cohort)]
            if len(preds["y_true"]) == 0:
                continue
            ax_scatter.scatter(preds["y_true"], preds["y_pred"], s=8, alpha=0.35, color=COHORT_COLORS[cohort], label=cohort)
        ax_scatter.set_xlabel(f"true {target}")
        ax_scatter.set_ylabel(f"predicted {target}")
        ax_scatter.set_title(f"{target}: held-out predictions, all subjects pooled")
        ax_scatter.legend(fontsize=8, frameon=False)
        ax_scatter.spines[["top", "right"]].set_visible(False)

        for cohort in ("R+", "R-"):
            sub = df[(df["target"] == target) & (df["reward_group"] == cohort)]
            if len(sub) == 0:
                continue
            lag_cols = [f"lag_{lag:+d}" for lag in LAGS]
            lag_vals = sub[lag_cols].to_numpy(dtype=float)
            mean_corr = np.nanmean(lag_vals, axis=0)
            sem_corr = np.nanstd(lag_vals, axis=0) / np.sqrt(np.sum(~np.isnan(lag_vals), axis=0).clip(min=1))
            ax_lag.plot(LAGS, mean_corr, color=COHORT_COLORS[cohort], label=cohort)
            ax_lag.fill_between(LAGS, mean_corr - sem_corr, mean_corr + sem_corr, color=COHORT_COLORS[cohort], alpha=0.2)
        ax_lag.axvline(0, color="#888888", lw=1, linestyle=":")
        ax_lag.axhline(0, color="#888888", lw=1, linestyle=":")
        ax_lag.set_xlabel("trial lag (predicted[t] vs true[t+lag])")
        ax_lag.set_ylabel("Pearson r, mean +- SEM across subjects")
        ax_lag.set_title(f"{target}: lag cross-correlation")
        ax_lag.legend(fontsize=8, frameon=False)
        ax_lag.spines[["top", "right"]].set_visible(False)

    fig_path = OUT_DIR / "050_perfquant_crossanimal_grid.png"
    fig.savefig(fig_path, dpi=150)
    print(f"saved {fig_path.name}")

    # --- single-session example figure: best- and worst-Pearson subject per cohort, per target ---
    examples_df = pd.DataFrame(all_examples)
    fig2, axes2 = plt.subplots(len(TARGETS), 4, figsize=(18, 3.6 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        rec_sub = df[df["target"] == target]
        col_i = 0
        for cohort in ("R+", "R-"):
            cohort_recs = rec_sub[rec_sub["reward_group"] == cohort].sort_values("pearson_r")
            if len(cohort_recs) == 0:
                col_i += 2
                continue
            worst_mouse = cohort_recs.iloc[0]["mouse"]
            best_mouse = cohort_recs.iloc[-1]["mouse"]
            for label, mouse in (("best", best_mouse), ("worst", worst_mouse)):
                ax = axes2[row_i][col_i]
                ex = examples_df[(examples_df["target"] == target) & (examples_df["mouse"] == mouse)]
                if len(ex) == 1:
                    ex = ex.iloc[0]
                    order = np.argsort(ex["trial_index"])
                    ax.plot(np.array(ex["trial_index"])[order], np.array(ex["y_true"])[order], color="#333333", lw=1.5, label="true")
                    ax.plot(np.array(ex["trial_index"])[order], np.array(ex["y_pred"])[order], color=COHORT_COLORS[cohort], lw=1.5, label="predicted")
                    r_val = rec_sub[(rec_sub["mouse"] == mouse)]["pearson_r"].iloc[0]
                    ax.set_title(f"{cohort} {mouse} ({label}, r={r_val:.2f})", fontsize=9)
                ax.set_xlabel("trial index", fontsize=8)
                ax.set_ylabel(target, fontsize=8)
                ax.legend(fontsize=7, frameon=False)
                ax.spines[["top", "right"]].set_visible(False)
                col_i += 1
    fig2_path = OUT_DIR / "050_perfquant_crossanimal_examples.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- metrics-comparison figure: rows=targets, cols=metrics; per-subject real (colored) vs its own null (gray) ---
    fig3, axes3 = plt.subplots(len(TARGETS), len(METRICS), figsize=(5 * len(METRICS), 3.6 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        sub = df[df["target"] == target].sort_values(["reward_group", "r2"]).reset_index(drop=True)
        x = np.arange(len(sub))
        for col_i, metric in enumerate(METRICS):
            ax = axes3[row_i][col_i]
            base = {"r2": "r2", "pearson_r": "pearson", "spearman_r": "spearman"}[metric]
            colors = [COHORT_COLORS[c] for c in sub["reward_group"]]
            ax.scatter(x, sub[metric], color=colors, s=30, zorder=3, label="real")
            ax.errorbar(x, sub[f"null_{base}_mean"], yerr=sub[f"null_{base}_std"], fmt="x", color="#888888",
                        ecolor="#888888", elinewidth=1, capsize=2, ms=5, zorder=2, label="null (own shift)")
            for xi, (real_v, null_v) in enumerate(zip(sub[metric], sub[f"null_{base}_mean"])):
                ax.plot([xi, xi], [real_v, null_v], color="#cccccc", lw=0.8, zorder=1)
            ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["mouse"], rotation=90, fontsize=6)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_title(f"{target}: {METRIC_LABELS[metric]}, real vs own null", fontsize=9)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / "050_perfquant_crossanimal_metrics_comparison.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")


if __name__ == "__main__":
    main()
