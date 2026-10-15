"""Pilot (user request 2026-09-14): compare the new pooled-scoring +
resample-on-missing-class CV estimator (`decode_bin_pooled`/
`select_fixed_c_pooled`) against the existing mean-of-folds estimator
(`decode_bin`/`select_fixed_c`) -- same sessions, same population matrices,
same fixed-C-across-bins design, time-bin by time-bin.

Scope: whole_brain scheme, "whole" condition only (no half/perfstate
split), hitmiss + perfstate targets (the two flagged for CV concerns;
modality not included), learning stage, stim alignment -- matches
`024_master_sweep.py`'s constants exactly for direct comparability.

Null controls (added 2026-09-14, user request: "add the mean+-sem of the
appropriate shuffle distributions"): both null types this pipeline uses for
hitmiss/perfstate elsewhere -- label-shuffle and linear-shift -- computed
under BOTH estimators (old: `label_shuffle_null_curves`/
`linear_shift_null_curves`; new: their `_pooled` analogs), at a reduced
`N_SHUF`/`NULL_N_REPEATS` relative to the main sweep to keep this pilot
tractable (this is a diagnostic comparison, not a significance-test sweep --
these null bands are for visual reference, not a formal test). Comparing
the null bands between estimators matters in its own right: a well-behaved
label-shuffle null should sit at ~0.5 regardless of which CV estimator
computed it, and any systematic difference there would itself flag a CV
problem, separate from whatever the real curves show.

Checkpointed per session (resumable, same pattern as `024_master_sweep.py`).

Usage: python 039_pooled_cv_pilot_wholebrain_whole.py <hitmiss|perfstate>
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures" / "behavior"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STIM_WINDOW = (-0.2, 0.6)
BIN_WIDTH = 0.05
STRIDE = 0.005
WIDE_DEAD_ZONE = (-0.010, 0.005)
N_REPEATS = 5
# Worker-pool size: default 110 (user decision 2026-09-14, sized for the
# haas056.rcp.epfl.ch remote host's 112 cores now that it's set up as a
# compute target) -- override with SSL_DECODE_N_WORKERS when running on a
# smaller machine (e.g. this local machine has 32 cores; 110 workers here
# would badly oversubscribe it).
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "110"))
N_SHUF = 10  # reduced from the main sweep's 25 -- diagnostic reference band, not a significance test
NULL_N_REPEATS = 2
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
AGGREGATE_COLOR = "#2c5f5b"
DIFF_COLOR = "#B5651D"
SHIFT_NULL_COLOR = "#B5651D"


def process_one_session(args: tuple) -> dict | None:
    session_id, subject_id, reward_group, scripts_dir, decode_target = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH,
        add_whole_brain_column,
        area_units,
        causal_bin_edges,
        data_sufficiency_ok,
        decode_curve,
        decode_curve_pooled,
        label_shuffle_null_curves,
        label_shuffle_null_curves_pooled,
        linear_shift_null_curves,
        linear_shift_null_curves_pooled,
        load_session_unit_spikes,
        prep_hitmiss_trials,
        prep_perfstate_trials_generic,
        select_fixed_c,
        select_fixed_c_pooled,
        sliding_bin_population_matrices,
        wide_window_matrix_from_bins,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    bin_edges = causal_bin_edges(STIM_WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)

    if decode_target == "hitmiss":
        trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        y = trials["lick_flag"].to_numpy().astype(bool) if trials is not None else None
    else:
        trials = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"])
        y = (trials["perf_state"] == "high").to_numpy() if trials is not None else None
    if trials is None or len(trials) == 0:
        return None

    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    ok, reason = data_sufficiency_ok(len(unit_ids), y)
    if not ok:
        return dict(session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                    n_trials=len(y), skipped_reason=reason)

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    is_whisker = np.ones(len(trials), dtype=bool)
    start_time = trials["start_time"].to_numpy()
    matrices = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=WIDE_DEAD_ZONE)
    X_wide = wide_window_matrix_from_bins(matrices)

    rng_old = np.random.default_rng(abs(hash((session_id, "old"))) % (2**31))
    rng_new = np.random.default_rng(abs(hash((session_id, "new"))) % (2**31))
    C_old = select_fixed_c(X_wide, y, rng_old)
    C_new = select_fixed_c_pooled(X_wide, y, rng_new)
    curve_old = decode_curve(matrices, y, C_old, rng_old, n_repeats=N_REPEATS)
    curve_new = decode_curve_pooled(matrices, y, C_new, rng_new, n_repeats=N_REPEATS)

    # Both null controls this pipeline uses for hitmiss/perfstate elsewhere
    # (label-shuffle + linear-shift), under both estimators -- user request
    # 2026-09-14.
    null_old = label_shuffle_null_curves(matrices, y, C_old, rng_old, n_shuf=N_SHUF, n_repeats=NULL_N_REPEATS)
    shift_old = linear_shift_null_curves(matrices, y, C_old, rng_old, n_shuf=N_SHUF, n_repeats=NULL_N_REPEATS)
    null_new = label_shuffle_null_curves_pooled(matrices, y, C_new, rng_new, n_shuf=N_SHUF, n_repeats=NULL_N_REPEATS)
    shift_new = linear_shift_null_curves_pooled(matrices, y, C_new, rng_new, n_shuf=N_SHUF, n_repeats=NULL_N_REPEATS)

    return dict(session_id=session_id, subject_id=subject_id, reward_group=reward_group, n_trials=len(y),
                skipped_reason=None, C_old=C_old, C_new=C_new,
                curve_old=curve_old.tolist(), curve_new=curve_new.tolist(),
                null_old=null_old.tolist(), shift_old=shift_old.tolist(),
                null_new=null_new.tolist(), shift_new=shift_new.tolist())


def run_sweep(decode_target: str):
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import causal_bin_edges, hitmiss_session_list

    partial_path = OUT_DIR / f"039_pooled_cv_pilot_{decode_target}_stim_whole_brain.parquet"
    edges_path = OUT_DIR / f"039_bin_edges_{decode_target}.json"
    bin_edges = causal_bin_edges(STIM_WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)
    edges_path.write_text(json.dumps(bin_edges))

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    sessions = hitmiss_session_list(sessions_tbl)
    todo_sessions = sessions[sessions.day_stage == "learning"].reset_index(drop=True)

    done = set()
    if partial_path.exists():
        done = set(pd.read_parquet(partial_path, columns=["session_id"])["session_id"].unique())
    todo = todo_sessions[~todo_sessions.session_id.isin(done)]
    print(f"[{decode_target}] {len(todo_sessions)} learning-stage sessions, {len(done)} already done, {len(todo)} to run", flush=True)

    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    tasks = [(r.session_id, r.subject_id, r.reward_group, scripts_dir, decode_target) for r in todo.itertuples()]

    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_one_session, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            session_id = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{decode_target}] ERROR {session_id}: {e!r}", flush=True)
                continue
            if row is None:
                continue
            new_df = pd.DataFrame([row])
            if partial_path.exists():
                new_df = pd.concat([pd.read_parquet(partial_path), new_df], ignore_index=True)
            new_df.to_parquet(partial_path, index=False)
            n_done += 1
            print(f"[{decode_target}] [{n_done}/{len(tasks)}] {session_id}: "
                  f"{'skipped: ' + row['skipped_reason'] if row['skipped_reason'] else 'ok'}", flush=True)
    print(f"[{decode_target}] DONE", flush=True)


def plot_comparison(decode_target: str, suffix: str = "", partial_path: Path | None = None, edges_path: Path | None = None):
    """`suffix`/explicit path overrides let a smoke test (e.g. `040_pooled_
    cv_smoke_test.py`, a couple of sessions) reuse this exact plotting code
    against its own small result file instead of the full sweep's."""
    partial_path = partial_path or OUT_DIR / f"039_pooled_cv_pilot_{decode_target}{suffix}_stim_whole_brain.parquet"
    edges_path = edges_path or OUT_DIR / f"039_bin_edges_{decode_target}{suffix}.json"
    df = pd.read_parquet(partial_path)
    df = df[df["skipped_reason"].isna()].copy()
    bin_edges = json.loads(edges_path.read_text())
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])

    def stacked(col):
        return np.stack([np.array(c) for c in df[col]])

    def null_mean_sem(series):
        """Each row is a list of `N_SHUF` per-session null-shuffle curves --
        average across shuffles first (one mean-null curve per session),
        then mean+-SEM across sessions, same convention as
        `032_plot_decode_results.py`'s `null_mean_sem_curves`."""
        per_row_means = []
        for c in series:
            if c is None or len(c) == 0:
                continue
            arr = np.stack([np.asarray(s, dtype=float) for s in c])
            per_row_means.append(np.nanmean(arr, axis=0))
        if not per_row_means:
            return None
        stacked_rows = np.stack(per_row_means)
        return np.nanmean(stacked_rows, axis=0), np.nanstd(stacked_rows, axis=0) / np.sqrt(stacked_rows.shape[0])

    fig, axes = plt.subplots(1, 4, figsize=(19.6, 5.2))
    groups = [("R+", df[df.reward_group == "R+"]), ("R-", df[df.reward_group == "R-"]), ("R+ & R- aggregated", df)]
    for ax, (label, sub) in zip(axes[:3], groups):
        color = COHORT_COLOR.get(label, AGGREGATE_COLOR)
        for col, ls, tag in [("curve_old", "-", "old (mean-of-folds)"), ("curve_new", "--", "new (pooled+resample)")]:
            c = np.stack([np.array(x) for x in sub[col]])
            mean_c = np.nanmean(c, axis=0)
            sem_c = np.nanstd(c, axis=0) / np.sqrt(c.shape[0])
            ax.plot(bin_labels_ms, mean_c, color=color, lw=2.0, linestyle=ls, label=tag)
            ax.fill_between(bin_labels_ms, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.12, lw=0)
        # Null bands (user request 2026-09-14): thin=old estimator,
        # thick=new estimator; dotted=label-shuffle, dash-dot=linear-shift --
        # a well-behaved label-shuffle null should sit at ~0.5 under either
        # estimator, so old-vs-new divergence here is itself diagnostic.
        for null_col, null_ls, null_color, null_lw, null_tag in [
            ("null_old", ":", color, 1.0, "old shuffled null"),
            ("shift_old", "-.", SHIFT_NULL_COLOR, 1.0, "old shift null"),
            ("null_new", ":", color, 2.2, "new shuffled null"),
            ("shift_new", "-.", SHIFT_NULL_COLOR, 2.2, "new shift null"),
        ]:
            band = null_mean_sem(sub[null_col]) if null_col in sub.columns else None
            if band is None:
                continue
            null_mean, null_sem = band
            ax.plot(bin_labels_ms, null_mean, color=null_color, lw=null_lw, linestyle=null_ls, alpha=0.6, label=null_tag)
            ax.fill_between(bin_labels_ms, null_mean - null_sem, null_mean + null_sem, color=null_color, alpha=0.08, lw=0)
        ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_ylim(0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped
        ax.set_box_aspect(1)
        ax.set_xlabel("time from start_time (ms)", fontsize=8)
        ax.set_ylabel("balanced accuracy", fontsize=8)
        ax.set_title(f"{label}\n(n={sub['session_id'].nunique()} sessions)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    # 4th panel: per-bin paired difference (new - old), pooled across all sessions
    ax = axes[3]
    old_c = stacked("curve_old")
    new_c = stacked("curve_new")
    diff = new_c - old_c
    diff_mean = np.nanmean(diff, axis=0)
    diff_sem = np.nanstd(diff, axis=0) / np.sqrt(diff.shape[0])
    ax.plot(bin_labels_ms, diff_mean, color=DIFF_COLOR, lw=2.0)
    ax.fill_between(bin_labels_ms, diff_mean - diff_sem, diff_mean + diff_sem, color=DIFF_COLOR, alpha=0.18, lw=0)
    ax.axhline(0.0, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
    ax.set_box_aspect(1)
    ax.set_xlabel("time from start_time (ms)", fontsize=8)
    ax.set_ylabel("new - old (balanced accuracy)", fontsize=8)
    ax.set_title(f"paired difference, all cohorts\n(n={df['session_id'].nunique()} sessions)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{decode_target}_stim_whole_brain -- pooled+resample CV vs. existing mean-of-folds CV", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = FIG_DIR / f"039_{decode_target}{suffix}_stim_whole_brain_cv_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("hitmiss", "perfstate"):
        print("Usage: python 039_pooled_cv_pilot_wholebrain_whole.py <hitmiss|perfstate>")
        sys.exit(1)
    decode_target = sys.argv[1]
    run_sweep(decode_target)
    plot_comparison(decode_target)


if __name__ == "__main__":
    main()
