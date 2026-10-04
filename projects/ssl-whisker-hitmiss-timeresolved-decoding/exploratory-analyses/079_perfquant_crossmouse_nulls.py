"""Cross-mouse session-permutation nulls for perfquant (Harris-style;
user spec 2026-09-23): tests whether the PLS+1SE decoding of a mouse's
behavioral curve from its own neural activity reflects something more
than (a) generic time/stimulus-locked drift any animal would show, or
(b) a trajectory shape common to all learners regardless of individual
timing.

Two nulls, same recipient, opposite element swapped:
- **R- null** (reward-dependence): recipient's OWN target Y (an R+
  mouse's curve), donor's NEURAL ACTIVITY X from an R- mouse (which by
  construction has no reward-driven learning). Tests whether ANY
  animal's generic activity can predict this specific R+ curve.
- **Within-R+ null** (specificity): recipient's OWN neural activity X,
  donor's TARGET Y from a DIFFERENT R+ mouse. Tests whether this mouse's
  activity predicts other learners' trajectories just as well as its
  own.

**Alignment**: whisker-trial-index position, truncated to
min(len(donor), len(recipient)) -- collapses to the same thing as
"whisker-stimulus count" alignment in this pipeline specifically, since
X/Y here are already built exclusively on the whisker-trial subsequence
(no auditory/no-stim rows mixed in) -- confirmed with the user before
building rather than assumed.

**Pilot scope** (not yet a full sweep): 5 R+ learning-stage recipient
sessions (top by `065`'s own whole_brain test_pearson -- MH070, MH069,
MH031, AB162, AB087), against ALL learning-stage R- sessions (37, R-
null) and ALL other learning-stage R+ sessions (37, within-R+ null) as
donor pools. All 3 targets scored jointly (free, one PLS fit per pair).

**Report per recipient x null_type x target**: corrected R2 (real minus
median donor score) and a rank-based p-value (fraction of donors scoring
>= real, standard permutation-style).

**Known limitation, stated not hidden** (user's own caveat): this
controls for generic time/stimulus-locked drift (R- null) and generic
"reward introduced" dynamics (within-R+ null), but NOT for anything else
R+ and R- groups systematically differ on (implant, cohort/batch,
recording quality). §4/§11 found no R+/R- difference in decoding
ACCURACY specifically, mild reassurance but not a guarantee against a
confound that would inflate the R- null itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
MIN_TRIALS_FOR_REGRESSION = 25
METRICS = ["r2", "pearson", "spearman"]

RECIPIENTS = [
    ("MH070", "MH070_20260121_140848"),
    ("MH069", "MH069_20260122_111455"),
    ("MH031", "MH031_20250507_104425"),
    ("AB162", "AB162_20250421_140550"),
    ("AB087", "AB087_20231017_141901"),
]


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


def session_X_valid(session_id: str, scripts_dir: str):
    """Whole_brain sensory-window feature matrix for one session, whisker
    trials only, NaN/zero-variance-column filtered -- the reusable X
    half of both a recipient and a donor."""
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return None, None
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
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
        return None, None
    return X, Y_raw


def process_pair(args: tuple) -> dict:
    """One (recipient, donor, null_type) cell, or null_type='real'."""
    recipient_mouse, recipient_sid, donor_mouse, donor_sid, null_type, scripts_dir = args
    out = dict(recipient_mouse=recipient_mouse, recipient_sid=recipient_sid, donor_mouse=donor_mouse,
               donor_sid=donor_sid, null_type=null_type, ok=False)

    recip_X, recip_Y_raw = session_X_valid(recipient_sid, scripts_dir)
    if recip_X is None:
        return out
    Y_mean, Y_std = recip_Y_raw.mean(axis=0), recip_Y_raw.std(axis=0)
    recip_Y = (recip_Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    if null_type == "real":
        X_pair, Y_pair = recip_X, recip_Y
    elif null_type == "rminus_null":
        donor_X, _ = session_X_valid(donor_sid, scripts_dir)
        if donor_X is None:
            return out
        m = min(len(donor_X), len(recip_Y))
        X_pair, Y_pair = donor_X[:m], recip_Y[:m]
    elif null_type == "withinrplus_null":
        _, donor_Y_raw = session_X_valid(donor_sid, scripts_dir)  # donor's TARGET, not X
        if donor_Y_raw is None:
            return out
        m = min(len(recip_X), len(donor_Y_raw))
        dY_mean, dY_std = donor_Y_raw.mean(axis=0), donor_Y_raw.std(axis=0)
        donor_Y = (donor_Y_raw - dY_mean) / np.where(dY_std > 0, dY_std, 1.0)
        X_pair, Y_pair = recip_X[:m], donor_Y[:m]
    else:
        raise ValueError(null_type)

    if len(X_pair) < MIN_TRIALS_FOR_REGRESSION:
        return out

    rng = np.random.default_rng(abs(hash(f"{recipient_sid}_{donor_sid}_{null_type}")) % (2**31))
    try:
        n_1se = select_pls_1se_components(X_pair, Y_pair, rng)
        Y_pred = _pooled_cv_predict(X_pair, Y_pair, n_1se, rng)
        scores = _score(Y_pair, Y_pred)
    except (ValueError, np.linalg.LinAlgError) as e:
        out["log"] = str(e)
        return out

    out.update(ok=True, n_trials=len(X_pair), **{f"{m}": scores[m] for m in METRICS})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pickle

    with open(OUT_DIR / "065_perfquant_allmice_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    rminus_learning = [(r["mouse"], r["session_id"]) for r in cache
                        if r["reward_group"] == "R-" and r["day_stage"] == "learning"]
    rplus_learning = [(r["mouse"], r["session_id"]) for r in cache
                       if r["reward_group"] == "R+" and r["day_stage"] == "learning"]
    recipient_sids = {sid for _, sid in RECIPIENTS}
    rplus_donors_pool = [(m, s) for m, s in rplus_learning if s not in recipient_sids]
    print(f"{len(rminus_learning)} R- donors, {len(rplus_donors_pool)} other-R+ donors, "
          f"{len(RECIPIENTS)} recipients", flush=True)

    tasks = []
    for rmouse, rsid in RECIPIENTS:
        tasks.append((rmouse, rsid, None, None, "real", SCRIPTS_DIR))
        for dmouse, dsid in rminus_learning:
            tasks.append((rmouse, rsid, dmouse, dsid, "rminus_null", SCRIPTS_DIR))
        for dmouse, dsid in rplus_donors_pool:
            tasks.append((rmouse, rsid, dmouse, dsid, "withinrplus_null", SCRIPTS_DIR))
    print(f"{len(tasks)} total (recipient, donor, null_type) cells", flush=True)

    results = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=100) as ex:
        futures = {ex.submit(process_pair, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            # A single cell's unhandled exception must never kill the whole
            # run (lesson from this script's own first launch, 2026-09-23:
            # an UnboundLocalError bug crashed the entire job after 2 of 355
            # cells -- same "one bad combination" resilience convention
            # 066's docstring already established for PLS numerical failures).
            try:
                res = fut.result()
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"  cell {futures[fut]} raised {type(e).__name__}: {e}", flush=True)
                continue
            if res.get("ok"):
                results.append(res)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} cells done ({n_failed} failed so far)", flush=True)
    print(f"{n_failed} cells raised an exception (of {len(tasks)} total)", flush=True)
    print(f"{len(results)}/{len(tasks)} cells usable", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "079_perfquant_crossmouse_nulls_raw.csv", index=False)

    summary = []
    for rmouse, rsid in RECIPIENTS:
        real_row = df[(df.recipient_sid == rsid) & (df.null_type == "real")]
        if len(real_row) == 0:
            print(f"{rmouse}/{rsid}: no real fit, skipped")
            continue
        for null_type in ("rminus_null", "withinrplus_null"):
            donors = df[(df.recipient_sid == rsid) & (df.null_type == null_type)]
            for k, target in enumerate(TARGETS):
                for metric in METRICS:
                    real_val = real_row.iloc[0][metric][k]
                    donor_vals = np.array([row[metric][k] for row in donors.itertuples()])
                    if len(donor_vals) == 0:
                        continue
                    corrected = real_val - np.median(donor_vals)
                    p = (1 + np.sum(donor_vals >= real_val)) / (1 + len(donor_vals))
                    summary.append(dict(recipient=rmouse, session_id=rsid, null_type=null_type, target=target,
                                         metric=metric, real=real_val, donor_median=float(np.median(donor_vals)),
                                         n_donors=len(donor_vals), corrected=corrected, p=p))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "079_perfquant_crossmouse_nulls_summary.csv", index=False)

    print("\n=== cross-mouse null results (pearson only shown; full table in CSV) ===")
    for rmouse, rsid in RECIPIENTS:
        sub = summary_df[(summary_df.recipient == rmouse) & (summary_df.metric == "pearson")]
        if len(sub) == 0:
            continue
        print(f"-- {rmouse} --")
        for _, row in sub.iterrows():
            print(f"    {row['null_type']:<18} {row['target']:<20} real={row['real']:+.3f}  "
                  f"donor_median={row['donor_median']:+.3f}  corrected={row['corrected']:+.3f}  "
                  f"p={row['p']:.3g}  (n_donors={int(row['n_donors'])})")

    # --- Figure: donor distributions + real score, per recipient x null_type, pearson only ---
    fig, axes = plt.subplots(len(RECIPIENTS), 2, figsize=(9, 3.2 * len(RECIPIENTS)), constrained_layout=True)
    null_labels = {"rminus_null": "R- null\n(reward-dependence)", "withinrplus_null": "within-R+ null\n(specificity)"}
    for row_i, (rmouse, rsid) in enumerate(RECIPIENTS):
        for col_i, null_type in enumerate(("rminus_null", "withinrplus_null")):
            ax = axes[row_i][col_i]
            donors = df[(df.recipient_sid == rsid) & (df.null_type == null_type)]
            real_row = df[(df.recipient_sid == rsid) & (df.null_type == "real")]
            if len(donors) == 0 or len(real_row) == 0:
                continue
            positions, data = [], []
            for k, target in enumerate(TARGETS):
                data.append(np.array([row["pearson"][k] for row in donors.itertuples()]))
                positions.append(k)
            parts = ax.violinplot(data, positions=positions, showmeans=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#999999")
                pc.set_alpha(0.5)
            for k, target in enumerate(TARGETS):
                ax.scatter([k], [real_row.iloc[0]["pearson"][k]], color="#d62728", s=50, zorder=5, marker="D")
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            ax.set_xticks(range(len(TARGETS)))
            ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=7.5)
            ax.set_title(f"{rmouse} -- {null_labels[null_type]}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cross-mouse session-permutation nulls (red diamond = real, violin = donor distribution)", fontsize=12)
    fig_path = OUT_DIR / "079_perfquant_crossmouse_nulls.png"
    fig.savefig(fig_path, dpi=140)
    print(f"\nsaved {fig_path.name}")
    print("DONE_079")


if __name__ == "__main__":
    main()
