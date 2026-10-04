"""Non-time-resolved perf-state decoding from two pre-trial windows (user
request 2026-09-16, following the decision to revert `decode_target=
'perfstate'`'s trial-type widening in `024_master_sweep.py`: mixing
whisker/auditory/no_stim trials into the SAME time-resolved curve adds
post-stimulus within-class heterogeneity unrelated to perf-state, since
evoked responses differ hugely by trial type. Pre-stimulus, that concern
doesn't apply -- so this is a dedicated, non-time-resolved analysis that
uses all 3 trial types specifically because everything here happens before
`start_time`.

Two windows, ONE aggregated (not sub-binned) feature per unit per window
-- not a curve:
1. **quiet_window**: `[start_time - 2000ms, start_time - 10ms]` per trial
   (~2s pre-trial period).
2. **ITI**: `2000ms` centered on the midpoint between each pair of
   time-consecutive trials' `quiet_window` START times (`start_time -
   2000ms`) -- one ITI point per (trial[i], trial[i+1]) pair, i.e. N-1 for
   N trials, NOT one per trial. Gets a perf_state label the same way
   auditory/no_stim trials do: nearest-in-time propagation
   (`merge_asof(..., direction='nearest')`) against the whisker-trial
   reference series.

Trial population for `quiet_window`: all 3 trial types
(`prep_perfstate_trials_curve(..., decode_trial_types=['whisker_trial',
'auditory_trial','no_stim_trial'])`) -- reinstates the trial-type widening
that was reverted for the time-resolved `perfstate` target, scoped here
instead where it's actually safe (no post-stimulus portion to contaminate).

Decode: high vs low perf-state, single aggregated window (no bin sweep),
`select_fixed_c_pooled`/`decode_bin_pooled` (this project's pooled-CV
estimator), linear-shift null (matches this project's convention for
hitmiss/perfstate: the meaningful confound is slow within-session drift,
which a label-shuffle null doesn't probe). whole_brain scheme first
(staged rollout, same convention `024_master_sweep.py` itself used
2026-09-11) -- area_group as a follow-up once this is validated.

Parallelized across sessions (`ProcessPoolExecutor`, mirrors
`043_perfstate_curve_labels_test.py`'s worker pattern). Run from the repo
root.

**Expert-stage support added 2026-09-17** (user: "For the perf state, use
the perf-state definition with old-median block and use the consensus
single bin decoders" -- i.e. this same non-time-resolved quiet_window/ITI
decode, but for expert-stage sessions, which the curve-based perf-state
definition above cannot label at all -- see `prep_perfstate_trials_curve`'s
own docstring: the per-mouse learning-curve file it needs only covers the
single day-0/whisker_0 session, there is no expert-stage equivalent. For
`day_stage='expert'`, trial labeling switches to
`prep_perfstate_trials_generic` (the older within-session block-median
hit-rate split, day-stage-agnostic by construction since it's derived
purely from that session's own behavior) and the session list switches
from the learning-stage-only `session_description=='whisker_0'` filter to
`hitmiss_session_list(sessions_tbl)` filtered to `day_stage=='expert'`
(the same canonical expert-stage session list `024_master_sweep.py`
itself uses). Everything else -- windows, estimator, null, plotting -- is
identical between stages. Output filenames get an `_expert` suffix
(matching `024_master_sweep.py`'s own day-stage suffix convention) so
expert results never overwrite the learning-stage ones.

Usage: `python 046_perfstate_baseline_decoding.py [n_test] [day_stage]`
  day_stage: 'learning' (default) or 'expert'.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
QUIET_WINDOW_S = 2.0   # quiet_window = [start_time - QUIET_WINDOW_S, start_time - QUIET_EDGE_S]
QUIET_EDGE_S = 0.010
ITI_HALF_WIDTH_S = 1.0  # ITI window = [midpoint - ITI_HALF_WIDTH_S, midpoint + ITI_HALF_WIDTH_S] (2000ms total)
N_REPEATS = 5
N_SHUF = 25
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "10"))  # default is local-machine scale;
# same env-var override 024_master_sweep.py uses to scale up to haas's ~110 cores for a full-scale run
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}  # this project's standing R+/R- color convention
AGGREGATE_COLOR = "#2c5f5b"  # matches 028/032's own "combined/aggregate" color


def linear_shift_null_bin_pooled(X: np.ndarray, y: np.ndarray, C: float, rng: np.random.Generator,
                                  n_shuf: int = N_SHUF, n_repeats: int = 2, n_folds: int = 5,
                                  min_shift_frac: float = 0.1, max_shift_frac: float = 0.5) -> np.ndarray:
    """Single-window analog of `linear_shift_null_curves_pooled` -- same
    linear (truncating, no wraparound) shift construction, scored via
    `decode_bin_pooled` on one already-aggregated feature matrix instead of
    a per-bin list."""
    from ssl_timeresolved_decoding import decode_bin_pooled
    n = len(y)
    min_shift = max(1, int(min_shift_frac * n))
    max_shift = max(min_shift, int(max_shift_frac * n))
    nulls = np.full(n_shuf, np.nan)
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
        if rng.random() < 0.5:
            y_shift, X_shift = y[shift:], X[: n - shift]
        else:
            y_shift, X_shift = y[: n - shift], X[shift:]
        nulls[s] = decode_bin_pooled(X_shift, y_shift, C, rng, n_repeats=n_repeats, n_folds=n_folds)
    return nulls


def process_one_session(args: tuple) -> dict:
    """Worker: everything for one (mouse, session_id) -- trial prep, ITI
    construction, feature extraction, and both windows' decode. Own
    imports/sys.path (ProcessPoolExecutor workers on Windows don't inherit
    the parent's already-modified sys.path)."""
    mouse, session_id, scripts_dir, day_stage = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, data_sufficiency_ok,
        decode_bin_pooled, event_aligned_rates_for_trials, load_session_unit_spikes,
        prep_perfstate_trials_curve, prep_perfstate_trials_generic, select_fixed_c_pooled,
    )

    out = dict(mouse=mouse, session_id=session_id, log=[], cells={})
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    # Curve-based labels (learning-stage only) vs the older within-session
    # block-median split (day-stage-agnostic) -- see module docstring's
    # "Expert-stage support added 2026-09-17" section.
    if day_stage == "expert":
        labeled = prep_perfstate_trials_generic(
            dataset_root, session_id, sessions_tbl, trials_tbl,
            decode_trial_types=["whisker_trial", "auditory_trial", "no_stim_trial"],
        )
        label_kind = "block-median"
    else:
        labeled = prep_perfstate_trials_curve(
            dataset_root, session_id, sessions_tbl, trials_tbl,
            decode_trial_types=["whisker_trial", "auditory_trial", "no_stim_trial"],
        )
        label_kind = "curve-based"
    if labeled is None:
        out["log"].append(f"SKIPPED -- no usable {label_kind} perf-state labels")
        return out
    labeled = labeled.sort_values("start_time").reset_index(drop=True)
    n_trials = len(labeled)
    out["log"].append(f"{n_trials} labeled trials (all 3 types): "
                       f"{(labeled['perf_state']=='high').sum()} high / {(labeled['perf_state']=='low').sum()} low")

    # Whisker-only reference series for propagating labels onto ITI points
    # (same mechanism auditory/no_stim trials already get inside
    # prep_perfstate_trials_curve -- reused here, not reimplemented, since
    # whisker rows' own perf_state was assigned directly, not propagated).
    whisker_ref = labeled[labeled["trial_type"] == "whisker_trial"][["start_time", "perf_state"]].sort_values("start_time")
    if len(whisker_ref) == 0:
        out["log"].append("SKIPPED -- no whisker-trial reference labels")
        return out

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))

    def decode_window(X: np.ndarray, y: np.ndarray) -> dict:
        ok, reason = data_sufficiency_ok(len(unit_ids), y)
        if not ok:
            return dict(ok=False, reason=reason, n=len(y))
        C = select_fixed_c_pooled(X, y, rng)
        real = decode_bin_pooled(X, y, C, rng, n_repeats=N_REPEATS)
        null = linear_shift_null_bin_pooled(X, y, C, rng, n_shuf=N_SHUF)
        return dict(ok=True, n=len(y), real=real, null_mean=float(np.nanmean(null)), null_std=float(np.nanstd(null)))

    # --- quiet_window: [start_time-2000ms, start_time-10ms], per trial ---
    start_time = labeled["start_time"].to_numpy()
    is_whisker = (labeled["trial_type"] == "whisker_trial").to_numpy()
    X_quiet = np.full((n_trials, len(unit_ids)), np.nan)
    for j, cid in enumerate(unit_ids):
        spikes = unit_spikes.get(cid, np.array([]))
        X_quiet[:, j] = event_aligned_rates_for_trials(
            spikes, event_time=start_time, trial_start_time=start_time, trial_is_whisker=is_whisker,
            window=(-QUIET_WINDOW_S, -QUIET_EDGE_S), dead_zone=None,
        )
    y_quiet = (labeled["perf_state"] == "high").to_numpy()
    res_quiet = decode_window(X_quiet, y_quiet)
    out["cells"]["quiet_window"] = res_quiet
    out["log"].append(f"[quiet_window] n={res_quiet['n']}" + (f" real={res_quiet['real']:.3f} null={res_quiet['null_mean']:.3f}+-{res_quiet['null_std']:.3f}"
                        if res_quiet["ok"] else f" SKIPPED -- {res_quiet['reason']}"))

    # --- ITI: midpoint of consecutive trials' quiet_window starts, 2000ms window centered on it ---
    quiet_starts = start_time - QUIET_WINDOW_S
    if len(quiet_starts) < 2:
        out["log"].append("[ITI] SKIPPED -- fewer than 2 trials")
        return out
    midpoints = (quiet_starts[:-1] + quiet_starts[1:]) / 2.0
    n_iti = len(midpoints)
    X_iti = np.full((n_iti, len(unit_ids)), np.nan)
    is_whisker_iti = np.zeros(n_iti, dtype=bool)  # ITI points aren't anchored to any one trial -- dead-zone n/a
    for j, cid in enumerate(unit_ids):
        spikes = unit_spikes.get(cid, np.array([]))
        X_iti[:, j] = event_aligned_rates_for_trials(
            spikes, event_time=midpoints, trial_start_time=midpoints, trial_is_whisker=is_whisker_iti,
            window=(-ITI_HALF_WIDTH_S, ITI_HALF_WIDTH_S), dead_zone=None,
        )
    iti_df = pd.DataFrame({"start_time": midpoints}).sort_values("start_time")
    iti_labeled = pd.merge_asof(iti_df, whisker_ref, on="start_time", direction="nearest")
    y_iti = (iti_labeled["perf_state"] == "high").to_numpy()
    res_iti = decode_window(X_iti, y_iti)
    out["cells"]["iti"] = res_iti
    out["log"].append(f"[iti] n={res_iti['n']}" + (f" real={res_iti['real']:.3f} null={res_iti['null_mean']:.3f}+-{res_iti['null_std']:.3f}"
                       if res_iti["ok"] else f" SKIPPED -- {res_iti['reason']}"))
    return out


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
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


def main():
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    day_stage = sys.argv[2] if len(sys.argv) > 2 else "learning"
    suffix = "" if day_stage == "learning" else f"_{day_stage}"  # matches 024_master_sweep.py's own convention
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    if day_stage == "expert":
        from ssl_timeresolved_decoding import hitmiss_session_list
        candidates = hitmiss_session_list(sessions_tbl)
        candidates = candidates[candidates["day_stage"] == "expert"]
        tasks = [(r["subject_id"], r["session_id"], SCRIPTS_DIR, day_stage)
                 for _, r in candidates.sort_values("subject_id").head(n_test).iterrows()]
        n_candidates = len(candidates)
    else:
        learning = sessions_tbl[(sessions_tbl["session_description"] == "whisker_0") & (sessions_tbl["has_ephys"])]
        tasks = [(r["subject_id"], r["session_id"], SCRIPTS_DIR, day_stage)
                 for _, r in learning.sort_values("subject_id").head(n_test).iterrows()]
        n_candidates = len(learning)
    print(f"{len(tasks)} sessions (of {n_candidates} candidates), whole_brain, {day_stage} stage, {N_WORKERS} workers")

    records = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            print(f"\n=== {res['mouse']} / {res['session_id']} ===")
            for line in res["log"]:
                print(f"  {line}")
            for window_name, cell in res["cells"].items():
                records.append(dict(mouse=res["mouse"], session_id=res["session_id"], window=window_name, **cell))

    df = pd.DataFrame(records)
    out_csv = OUT_DIR / f"046_perfstate_baseline_decoding_results{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    ok = df[df["ok"] == True]  # noqa: E712
    print(f"\n{len(ok)}/{len(df)} (session, window) cells decodable")
    for window_name in ("quiet_window", "iti"):
        sub = ok[ok.window == window_name]
        if len(sub):
            print(f"  {window_name}: n={len(sub)} sessions, real acc mean={sub['real'].mean():.3f}, "
                  f"null mean={sub['null_mean'].mean():.3f}")

    # Raw accuracy vs accuracy-minus-own-null (added 2026-09-16, user
    # follow-up after seeing raw accuracy alone: both windows sit around
    # real~0.86 next to null~0.85-0.86, i.e. raw accuracy here is dominated
    # by the same slow, drift-driven signal the shift-null is designed to
    # reveal and subtract out -- the above-null difference is what actually
    # reflects real-vs-null-additional perf-state coding, not raw accuracy
    # on its own), broken down by R+/R- cohort (user follow-up 2026-09-16,
    # "Break it down by R+/R- cohort") alongside the pooled/combined view --
    # `reward_group` isn't in the per-session decode output, so it's joined
    # in here post-hoc from the same mouse-reference table `load_reward_group`
    # uses (`REF_PATH`), keyed on `mouse` (== subject_id).
    from ssl_bwm_trial_prep import REF_PATH
    ref = pd.read_parquet(Path(REF_PATH))
    reward_group_map = ref.set_index("subject_id")["reward_group"].to_dict()
    ok = ok.copy()
    ok["diff"] = ok["real"] - ok["null_mean"]
    ok["reward_group"] = ok["mouse"].map(reward_group_map)
    n_no_group = int(ok["reward_group"].isna().sum())
    if n_no_group:
        print(f"  NOTE: {n_no_group}/{len(ok)} rows have no R+/R- reward_group (excluded from cohort panels)")

    def _paired_panel(ax, sub: pd.DataFrame, values_col: str, ref_line: float, ylim: tuple[float, float], color: str):
        """Points colored by COHORT (this panel's own `color`), not by
        window -- window is already unambiguous from x-position/tick label,
        so color is free to carry cohort identity instead (2026-09-16, user
        request "use correct colors for cohorts" -- previously points were
        colored teal/magenta by window regardless of cohort, not using this
        project's standing R+/R-/aggregate color convention at all)."""
        wide = sub.pivot_table(index="session_id", columns="window", values=values_col)
        wide = wide.dropna(subset=[c for c in ("quiet_window", "iti") if c in wide.columns])
        if {"quiet_window", "iti"}.issubset(wide.columns) and len(wide) >= 2:
            for _, r in wide.iterrows():
                ax.plot([0, 1], [r["quiet_window"], r["iti"]], color="#bbbbbb", lw=0.6, alpha=0.5, zorder=1)
            rng = np.random.default_rng(0)
            for x, col in enumerate(("quiet_window", "iti")):
                vals = wide[col].to_numpy()
                jitter = rng.uniform(-0.03, 0.03, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, zorder=2, color=color, alpha=0.85,
                           edgecolors="none" if x == 0 else color, facecolors=color if x == 0 else "none",
                           linewidths=1.1)
                ax.errorbar(x, vals.mean(), yerr=vals.std() / np.sqrt(len(vals)), fmt="D", color="black",
                            markersize=6, capsize=3, zorder=3)
            w_p = stats.wilcoxon(wide["quiet_window"], wide["iti"]).pvalue if len(wide) >= 3 else float("nan")
            t_p = stats.ttest_rel(wide["quiet_window"], wide["iti"]).pvalue
            # One-sample test per window: is it itself above `ref_line` (0
            # for the diff panel -- above its own null; not meaningful for
            # the raw-accuracy panel, whose ref_line is chance = 0.5 and
            # whose "above null" question the diff panel already answers).
            note = f"paired: Wilcoxon p={w_p:.3g}, t p={t_p:.3g}"
            if ref_line == 0.0:
                qw_p = stats.wilcoxon(wide["quiet_window"]).pvalue if len(wide) >= 3 else float("nan")
                iti_p = stats.wilcoxon(wide["iti"]).pvalue if len(wide) >= 3 else float("nan")
                note += f"\nvs 0: qw p={qw_p:.3g}, ITI p={iti_p:.3g}"
            ax.set_title(f"n={len(wide)}\n{note}", fontsize=7.5, color="#444444")
        else:
            ax.set_title(f"n={len(wide)} (too few for stats)", fontsize=7.5, color="#444444")
        ax.axhline(ref_line, color="#888888", lw=1, linestyle=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["quiet_window", "ITI"], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_box_aspect(1)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)

    diff_ylim = (min(-0.02, ok["diff"].min() - 0.02), max(0.05, ok["diff"].max() + 0.02))
    cohorts = [("combined", ok, AGGREGATE_COLOR), ("R+", ok[ok.reward_group == "R+"], COHORT_COLOR["R+"]),
               ("R-", ok[ok.reward_group == "R-"], COHORT_COLOR["R-"])]
    metric_rows = [("real", "raw accuracy\nbalanced accuracy (perf-state)", 0.5, (0.3, 1.02)),
                   ("diff", "above shift-null\naccuracy - shift-null mean", 0.0, diff_ylim)]
    # Landscape 2 (metric) x 3 (cohort) layout (2026-09-16, user request
    # "Improve layout" -- previously 3x2, very tall (17in) and awkward for
    # comparing cohorts, which now sit side by side in the same row instead
    # of stacked, matching how cohort comparisons are laid out elsewhere in
    # this project (e.g. `042`'s R+/R-/combined panels)).
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), constrained_layout=True)
    for row_i, (values_col, ylabel, ref_line, ylim) in enumerate(metric_rows):
        for col_i, (cohort_label, sub, color) in enumerate(cohorts):
            ax = axes[row_i, col_i]
            _paired_panel(ax, sub, values_col, ref_line, ylim, color)
            if row_i == 0:
                ax.set_title(f"{cohort_label}\n{ax.get_title()}", fontsize=9, color=color if cohort_label != "combined" else "#444444")
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
    fig.suptitle("quiet_window vs ITI, perf-state decode, by cohort", fontsize=12)
    savefig_retry(fig, OUT_DIR / f"046_perfstate_baseline_decoding{suffix}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved 046_perfstate_baseline_decoding{suffix}.png")


if __name__ == "__main__":
    main()
