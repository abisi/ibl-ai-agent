"""Test run (user request 2026-09-15): sessions, whole_brain, comparing the
CURRENT block-based perf-state definition (`prep_perfstate_trials_generic`:
fixed-size whisker-trial blocks, per-session median split of
cohort-corrected hit rate) against the NEW curve-based definition (Axel
Bisi's own learning-curve model, via
`ephys_utilities.helpers.load_helpers.load_learning_curves_data` +
`cd_analysis/utils/performance.py`'s `assign_expertise_blocks`/
`propagate_expertise_inplace` logic, ported here with attribution rather
than imported -- `cd_analysis.utils.performance` pulls in its own
`utils.settings_haas` at import time, a dependency/environment risk this
repo doesn't need to take on for ~90 lines of portable logic).

**Scope note, discovered during this test's own setup**: the learning-curve
H5 file is NOT a mouse-lifetime, multi-session file -- one row per mouse,
`day==0` always, covering only that mouse's single learning-stage
(whisker_0) session's whisker trials. Learning-stage only, no cross-session
alignment risk. Off-by-one trial-count bug (from reusing
`ssl_bwm_trial_prep.prep_session`, which drops one extra trial for an
unrelated t-1-outcome feature) found and fixed -- see
`active_trials_from_whisker_onset` below; every session run through this
script now asserts an exact trial-count match rather than silently
tolerating a mismatch.

**Extended 2026-09-15 (user: "Try with a few more sessions")**: session
list now pulled from `045_perfstate_curve_label_sweep.py`'s full 71-session
sweep -- specifically the 20 sessions decodable (>=5 hits AND >=5 misses)
in all 4 (label_def x state) cells, so every session contributes a curve to
every panel instead of dropping out unpredictably. Parallelized across
sessions (`ProcessPoolExecutor`, mirroring `024_master_sweep.py`'s worker
pattern) since 4x the sessions at the same per-session cost would otherwise
take ~4x as long serially.

Pipeline, per test session: see `active_trials_from_whisker_onset` /
`assign_expertise_blocks_positional` / `new_perfstate_trials` docstrings
below. Decode: whole_brain, `condition_type='perfstate'`, high vs low, both
label definitions, today's pooled-CV estimator for both so the only
difference between the two curves is the label definition. REDUCED
settings vs. the production sweep (coarser stride, fewer CV repeats/
shuffles) -- a plausibility check, not a result to report as final.

Run from the repo root (`load_reward_group` resolves a repo-root-relative
path, same requirement `032`/`028` already have). Ran locally, not on
haas (deviates from `ssl_prefer_haas_compute.md`) -- haas was at
load average 185/112 cores (its own 110-worker sweep + other users) when
this was launched, so adding a competing worker pool there would have
slowed the primary job more than running this locally on this otherwise-
idle 32-core machine.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CURVE_ROOT = axel_bisi_path("combined_results_ks4")
EPHYS_UTILS_PATH = axel_bisi_path("Github", "ephys_utilities")

# Reduced-fidelity test settings (NOT the production 024_master_sweep.py
# values) -- coarser stride, fewer repeats/shuffles, purely to check the
# pipeline runs end to end and produces a plausible-shaped curve.
STIM_WINDOW = (-0.2, 0.6)
BIN_WIDTH = 0.05
STRIDE_TEST = 0.02          # vs. production 0.005 -- ~40 bins instead of ~161
DEAD_ZONE = (-0.010, 0.005)
N_REPEATS_TEST = 3          # vs. production 5
N_CONSECUTIVE = 5           # matches cd_analysis's assign_expertise_blocks default
# Null curves dropped in this expanded run (the aggregate figure only plots
# real_curve, same as before) -- saves N_SHUF_TEST x the decode cost per
# cell now that there are 4x the sessions.
N_WORKERS = 10               # local 32-core machine; leaves headroom, not a full-sweep worker count

# Pulled from 045's full sweep: sessions decodable (>=5 hits AND >=5 misses)
# in all 4 (label_def x state) cells, so every session contributes to every
# panel below. See 045_perfstate_curve_label_sweep.csv.
TEST_MICE = [
    "AB092", "AB120", "AB121", "AB126", "AB127", "AB128", "AB129", "AB130",
    "AB133", "AB138", "AB143", "AB144", "AB147", "AB154", "AB158", "AB159",
    "MH011", "MH014", "MH022", "MH030",
]


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    """`fig.savefig` occasionally hits a transient Windows file-lock
    (`OSError: [Errno 22] Invalid argument`, seen repeatedly this session --
    likely antivirus/cloud-sync briefly holding a newly-created file, not a
    real problem with the path or the figure) that a simple retry always
    clears. Losing an entire session's worth of already-computed decode
    curves to this (as happened once) isn't acceptable -- retry rather than
    let it propagate."""
    last_err = None
    for attempt in range(attempts):
        try:
            fig.savefig(out_path, **kwargs)
            return
        except OSError as e:
            last_err = e
            print(f"  savefig attempt {attempt+1}/{attempts} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    raise last_err


def active_trials_from_whisker_onset(session_id: str, trials_tbl: pd.DataFrame) -> pd.DataFrame:
    """Context filter + `perf != 6` ("association" outcome) filter +
    drop-before-first-whisker-trial -- matches `cd_analysis/utils/performance.py`'s
    `keep_active_from_whisker_onset` (both conditions applied together),
    NOT `ssl_bwm_trial_prep.prep_session` (which drops one extra trial for
    an unrelated t-1-outcome feature -- broke positional alignment with the
    curve file's trial count by exactly one, the original bug this helper
    was written to avoid). The `perf != 6` clause (added 2026-09-15, was
    missing) matters for subjects whose `context` field is the literal
    string "nan" (an older recording-era gap) -- context filtering is a
    no-op for those, so `perf != 6` is the only thing excluding
    'association'-outcome trials; its absence caused a genuine trial-count
    mismatch (distinct from the off-by-one bug) for 4/89 sessions in
    `045`'s first sweep."""
    trials = trials_tbl[trials_tbl["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    has_context = trials["context"].notna() & (trials["context"] != "nan")
    if has_context.any():
        trials = trials[trials["context"] == "active"]
    if "perf" in trials.columns:
        trials = trials[trials["perf"] != 6]
    trials = trials.reset_index(drop=True)
    whisker_idx = trials.index[trials["trial_type"] == "whisker_trial"]
    if len(whisker_idx) > 0:
        trials = trials.loc[whisker_idx[0]:].reset_index(drop=True)
    return trials


def assign_expertise_blocks_positional(p_low: np.ndarray, p_chance: np.ndarray, reward_group_int: int,
                                        n_consecutive: int = N_CONSECUTIVE) -> np.ndarray:
    """Port of `cd_analysis/utils/performance.py`'s `assign_expertise_blocks`
    criterion + contiguous-run detection, applied directly to one session's
    already-positionally-aligned `p_low`/`p_chance` arrays. Returns a
    boolean mask, True where `block_perf_type=='high'`."""
    if reward_group_int == 1:
        criterion = p_low > p_chance
    elif reward_group_int == 0:
        criterion = p_low < p_chance
    else:
        raise ValueError(f"unexpected reward_group_int {reward_group_int!r}")

    high_mask = np.zeros(len(criterion), dtype=bool)
    start_idx = 0
    while start_idx < len(criterion):
        if criterion[start_idx]:
            end_idx = start_idx
            while end_idx < len(criterion) and criterion[end_idx]:
                end_idx += 1
            if end_idx - start_idx >= n_consecutive:
                high_mask[start_idx:end_idx] = True
            start_idx = end_idx
        else:
            start_idx += 1
    return high_mask


def new_perfstate_trials(subject_id: str, reward_group: str, all_trials: pd.DataFrame,
                          decode_trial_types: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    """Curve-based analog of `prep_perfstate_trials_generic`, same output
    contract (a `decode_trials` dataframe with `perf_state`/`reward_group`/
    `subject_id` columns). Returns (dataframe_or_None, skip_reason_or_None).
    `all_trials` must already be `active_trials_from_whisker_onset`'s
    output for this session."""
    from ephys_utilities.helpers.load_helpers import load_learning_curves_data

    reward_group_int = 1 if reward_group == "R+" else 0
    whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
    if len(whisker) == 0:
        return None, "no whisker trials"

    if CURVE_ROOT is None or EPHYS_UTILS_PATH is None:
        return None, "Axel_Bisi share (combined_results_ks4 / ephys_utilities) not mounted on this machine"
    try:
        curves_df = load_learning_curves_data(str(CURVE_ROOT), [subject_id])
    except ValueError:
        return None, "no learning-curve file for this mouse"
    if len(curves_df) == 0:
        return None, "no learning-curve file for this mouse"
    row = curves_df.iloc[0]
    p_low, p_chance = np.asarray(row["p_low"]), np.asarray(row["p_chance"])
    if len(p_low) != len(whisker):
        return None, f"trial-count mismatch: ours={len(whisker)} vs curve={len(p_low)}"

    high_mask = assign_expertise_blocks_positional(p_low, p_chance, reward_group_int)
    whisker = whisker.copy()
    whisker["block_perf_type"] = np.where(high_mask, "high", "low")

    decode_trials = all_trials[all_trials["trial_type"].isin(decode_trial_types)].copy()
    if len(decode_trials) == 0:
        return None, "no decode-target trials"
    decode_trials = decode_trials.sort_values("start_time").reset_index(drop=True)

    if set(decode_trial_types) == {"whisker_trial"}:
        decode_trials["perf_state"] = whisker.sort_values("start_time")["block_perf_type"].to_numpy()
    else:
        # Port of `propagate_expertise_inplace`: nearest-in-time whisker
        # trial's label. Untested by this run (hitmiss is whisker-only).
        whisker_sorted = whisker.sort_values("start_time")[["start_time", "block_perf_type"]]
        decode_trials = pd.merge_asof(decode_trials, whisker_sorted, on="start_time", direction="nearest")
        decode_trials["perf_state"] = decode_trials["block_perf_type"]

    decode_trials["reward_group"] = reward_group
    decode_trials["subject_id"] = subject_id
    return decode_trials, None


def process_one_session(args: tuple) -> dict:
    """Worker: everything for one (mouse, session_id) -- labeling,
    comparison, and the 4 decode cells. Does its own imports/sys.path setup
    (mirrors `024_master_sweep.py`'s `process_one_session`) since
    `ProcessPoolExecutor` workers on Windows don't inherit the parent's
    already-modified `sys.path`."""
    mouse, session_id, scripts_dir, ephys_utils_path = args
    sys.path.insert(0, scripts_dir)
    if ephys_utils_path:
        sys.path.insert(0, ephys_utils_path)
    import numpy as np  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, causal_bin_edges, data_sufficiency_ok,
        decode_curve_pooled, label_shuffle_null_curves_pooled, load_session_unit_spikes,
        prep_perfstate_trials_generic, select_fixed_c_pooled, sliding_bin_population_matrices,
        wide_window_matrix_from_bins,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    bin_edges = causal_bin_edges(STIM_WINDOW, bin_width=BIN_WIDTH, stride=STRIDE_TEST)

    out = dict(mouse=mouse, session_id=session_id, log=[], cells={}, reward_group=None)
    reward_group = load_reward_group(mouse)
    if reward_group is None:
        out["log"].append("no usable reward_group")
        return out
    out["reward_group"] = reward_group

    all_trials = active_trials_from_whisker_onset(session_id, trials_tbl)
    old_trials = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl,
                                                 decode_trial_types=["whisker_trial"])
    new_trials, new_skip_reason = new_perfstate_trials(mouse, reward_group, all_trials, decode_trial_types=["whisker_trial"])

    if old_trials is None:
        out["log"].append("old def: no usable trials")
    else:
        out["log"].append(f"old def: {(old_trials['perf_state']=='high').sum()} high / "
                           f"{(old_trials['perf_state']=='low').sum()} low ({len(old_trials)} total)")
    if new_trials is None:
        out["log"].append(f"new def: SKIPPED -- {new_skip_reason}")
    else:
        out["log"].append(f"new def: {(new_trials['perf_state']=='high').sum()} high / "
                           f"{(new_trials['perf_state']=='low').sum()} low ({len(new_trials)} total)")

    if old_trials is not None and new_trials is not None:
        merged = old_trials[["trial_id", "perf_state"]].merge(
            new_trials[["trial_id", "perf_state"]], on="trial_id", suffixes=("_old", "_new"))
        agree = (merged["perf_state_old"] == merged["perf_state_new"]).mean() if len(merged) else float("nan")
        out["log"].append(f"agreement ({len(merged)} trials in both): {agree:.1%}")
        out["agreement"] = agree

    if new_trials is None:
        return out

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))

    def decode_one(trials_perfstate: pd.DataFrame, state_val: str) -> dict:
        sub = trials_perfstate[trials_perfstate["perf_state"] == state_val]
        y = sub["lick_flag"].to_numpy().astype(bool)
        ok, reason = data_sufficiency_ok(len(unit_ids), y)
        if not ok:
            return dict(ok=False, reason=reason, n_trials=len(y))
        start_time = sub["start_time"].to_numpy()
        is_whisker = np.ones(len(sub), dtype=bool)
        matrices = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=DEAD_ZONE)
        X_wide = wide_window_matrix_from_bins(matrices)
        C = select_fixed_c_pooled(X_wide, y, rng)
        real = decode_curve_pooled(matrices, y, C, rng, n_repeats=N_REPEATS_TEST)
        return dict(ok=True, n_trials=len(y), real_curve=real.tolist())

    for label_name, trials_for_decode in (("old", old_trials), ("new", new_trials)):
        if trials_for_decode is None:
            continue
        for state_val in ("high", "low"):
            res = decode_one(trials_for_decode, state_val)
            out["cells"][f"{label_name}_{state_val}"] = res
            if res["ok"]:
                out["log"].append(f"[{label_name}, {state_val}] n_trials={res['n_trials']}, peak_acc={np.nanmax(res['real_curve']):.3f}")
            else:
                out["log"].append(f"[{label_name}, {state_val}] SKIPPED -- {res['reason']}")
    return out


def main():
    sessions_tbl = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet")
    bin_edges_n = len(np.arange(STIM_WINDOW[0], STIM_WINDOW[1] + 1e-9, STRIDE_TEST))
    bin_labels_ms = np.array([round(t, 10) * 1000 for t in np.arange(STIM_WINDOW[0], STIM_WINDOW[1] + 1e-9, STRIDE_TEST)])
    print(f"bin grid: {bin_edges_n} bins, stride={STRIDE_TEST*1000:.0f}ms (test-reduced, production uses 5ms)")
    print(f"{len(TEST_MICE)} sessions, {N_WORKERS} parallel workers")

    tasks = []
    for mouse in TEST_MICE:
        sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                             & (sessions_tbl["has_ephys"])]
        if len(sess) == 0:
            print(f"{mouse}: no learning-stage (whisker_0) has_ephys session, skipping")
            continue
        tasks.append((mouse, sess["session_id"].iloc[0], SCRIPTS_DIR, str(EPHYS_UTILS_PATH) if EPHYS_UTILS_PATH else None))

    # curves[label_def][state][cohort] -- cohort in {"R+","R-"}; "aggregated"
    # is derived at plot time by pooling both, not stored separately.
    records = []
    curves = {ld: {sv: {"R+": [], "R-": []} for sv in ("high", "low")} for ld in ("old", "new")}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            print(f"\n=== {res['mouse']} / {res['session_id']} ({res['reward_group']}) ===")
            for line in res["log"]:
                print(f"  {line}")
            for cell_name, cell in res["cells"].items():
                label_name, state_val = cell_name.split("_")
                records.append(dict(mouse=res["mouse"], session_id=res["session_id"], reward_group=res["reward_group"],
                                     label_def=label_name, state=state_val,
                                     ok=cell["ok"], reason=cell.get("reason"), n_trials=cell.get("n_trials")))
                if cell["ok"] and res["reward_group"] in ("R+", "R-"):
                    curves[label_name][state_val][res["reward_group"]].append(np.array(cell["real_curve"]))

    out_csv = OUT_DIR / "043_perfstate_curve_test_results.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    # Persist raw curves before attempting the plot -- a plotting failure
    # (hit once this session, a transient Windows file-lock on savefig)
    # must never cost re-running the decode compute to recover from.
    npz_path = OUT_DIR / "043_perfstate_curve_test_curves.npz"
    npz_payload = {
        f"{ld}_{sv}_{coh.replace('+', 'plus').replace('-', 'minus')}": np.stack(arrs)
        for ld, d in curves.items() for sv, cd in d.items() for coh, arrs in cd.items() if arrs
    }
    np.savez(npz_path, **npz_payload, bin_labels_ms=bin_labels_ms)
    print(f"saved {npz_path.name}")

    colors = {"high": "#00838f", "low": "#c2185b"}
    cohort_panels = [("R+", "R+"), ("R-", "R-"), (None, "R+ & R- aggregated")]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.6), constrained_layout=True)
    for row_i, label_name in enumerate(("old", "new")):
        for col_i, (cohort_key, panel_title) in enumerate(cohort_panels):
            ax = axes[row_i][col_i]
            for state_val in ("high", "low"):
                if cohort_key is None:
                    stack_list = curves[label_name][state_val]["R+"] + curves[label_name][state_val]["R-"]
                else:
                    stack_list = curves[label_name][state_val][cohort_key]
                if not stack_list:
                    continue
                stack = np.stack(stack_list)
                mean_curve = np.nanmean(stack, axis=0)
                sem_curve = np.nanstd(stack, axis=0) / np.sqrt(stack.shape[0])
                ax.plot(bin_labels_ms, mean_curve, color=colors[state_val], lw=2, label=f"{state_val} (n={stack.shape[0]})")
                ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=colors[state_val], alpha=0.15, lw=0)
            ax.axhline(0.5, color="#888888", lw=1, linestyle=":")
            ax.axvline(0, color="#333333", lw=1, alpha=0.4)
            ax.set_ylim(0.4, 1.0)
            ax.set_box_aspect(1)
            def_label = "block-median, current" if label_name == "old" else "learning-curve, new"
            ax.set_title(f"{label_name} def ({def_label}) | {panel_title}", fontsize=9.5)
            ax.set_xlabel("time from start_time (ms)", fontsize=8.5)
            ax.set_ylabel("balanced accuracy (hit vs miss)", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Test run: {len(TEST_MICE)} sessions (decodable in all 4 cells, per 045's sweep), whole_brain, "
                 f"hitmiss x perfstate -- reduced settings (stride={STRIDE_TEST*1000:.0f}ms, n_repeats={N_REPEATS_TEST})", fontsize=10)
    out_png = OUT_DIR / "043_perfstate_curve_test.png"
    savefig_retry(fig, out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png.name}")


if __name__ == "__main__":
    main()
