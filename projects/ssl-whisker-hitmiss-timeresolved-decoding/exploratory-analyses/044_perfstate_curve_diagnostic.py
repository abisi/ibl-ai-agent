"""Diagnostic figure (user request 2026-09-15, following up on
`043_perfstate_curve_labels_test.py`'s trial-count fix): per test session,
the learning-curve model's own `p_mean` ("p_lick") and `p_chance` curves
over whisker-trial index, with the resulting high/low state allocation
shown as background shading -- a direct visual check of the new
curve-based perf-state definition before trusting it in a decode.

Reuses `043`'s (already-fixed) `active_trials_from_whisker_onset` +
`assign_expertise_blocks_positional` alignment logic verbatim (duplicated
here per this project's per-script self-containment convention, not
imported -- `043_...py` is a script, not an importable module name).
Explicitly asserts the trial count matches exactly (user 2026-09-15: "the
trials must match exactly") rather than silently skipping a mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_bwm_trial_prep import load_reward_group  # noqa: E402

EPHYS_UTILS_PATH = axel_bisi_path("Github", "ephys_utilities")
if EPHYS_UTILS_PATH is not None:
    sys.path.insert(0, str(EPHYS_UTILS_PATH))
from ephys_utilities.helpers.load_helpers import load_learning_curves_data  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CURVE_ROOT = axel_bisi_path("combined_results_ks4")
N_CONSECUTIVE = 5
TEST_MICE = ["AB161", "AB157", "AB087", "AB126", "AB139", "MH062"]


def active_trials_from_whisker_onset(session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> pd.DataFrame:
    """Verbatim copy of `043_perfstate_curve_labels_test.py`'s helper of the
    same name -- context filter + `perf != 6` ("association" outcome)
    filter + drop-before-first-whisker-trial, matching `cd_analysis/utils/
    performance.py`'s `keep_active_from_whisker_onset`. NOT
    `ssl_bwm_trial_prep.prep_session` (which additionally drops the first
    post-warmup trial for its own unrelated t-1-outcome feature -- the
    source of the off-by-one this script's assertion below guards against).
    The `perf != 6` clause (added 2026-09-15, was missing) matters for
    subjects whose `context` field is the literal string "nan" -- context
    filtering is a no-op for those, so `perf != 6` is the only thing
    excluding 'association'-outcome trials; its absence caused a genuine
    trial-count mismatch for 4/89 sessions in `045`'s first sweep."""
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
    """Verbatim copy of `043`'s helper of the same name -- port of
    `cd_analysis/utils/performance.py`'s `assign_expertise_blocks`."""
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


def shade_state_spans(ax, high_mask: np.ndarray, n_trials: int):
    """Contiguous-run background shading: light green for 'high' runs,
    light red for 'low' runs (not just high -- every trial belongs to one
    state or the other, so the whole axis background should say so)."""
    start = 0
    while start < n_trials:
        state = high_mask[start]
        end = start
        while end < n_trials and high_mask[end] == state:
            end += 1
        color = "#c8e6c9" if state else "#ffcdd2"
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.6, zorder=0, lw=0)
        start = end


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    panels = []
    for mouse in TEST_MICE:
        sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                             & (sessions_tbl["has_ephys"])]
        if len(sess) == 0:
            print(f"{mouse}: no learning-stage (whisker_0) has_ephys session, skipping")
            continue
        session_id = sess["session_id"].iloc[0]
        reward_group = load_reward_group(mouse)
        if reward_group is None:
            print(f"{mouse}: no usable reward_group, skipping")
            continue
        reward_group_int = 1 if reward_group == "R+" else 0

        all_trials = active_trials_from_whisker_onset(session_id, sessions_tbl, trials_tbl)
        whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
        if len(whisker) == 0:
            print(f"{mouse}: no whisker trials, skipping")
            continue

        try:
            curves_df = load_learning_curves_data(str(CURVE_ROOT), [mouse])
        except ValueError:
            print(f"{mouse}: no learning-curve file, skipping")
            continue
        if len(curves_df) == 0:
            print(f"{mouse}: no learning-curve file, skipping")
            continue
        row = curves_df.iloc[0]
        p_mean, p_low, p_high, p_chance = (np.asarray(row[c]) for c in ("p_mean", "p_low", "p_high", "p_chance"))

        # Hard assertion (user 2026-09-15: "the trials must match exactly")
        # -- not a soft skip-with-warning, since a silent mismatch here
        # would mean every downstream label is misattributed to the wrong
        # trial.
        assert len(p_mean) == len(whisker), (
            f"{mouse}: trial count mismatch (ours={len(whisker)}, curve={len(p_mean)}) -- "
            "alignment assumption violated, do not trust labels from this session"
        )

        high_mask = assign_expertise_blocks_positional(p_low, p_chance, reward_group_int)
        n_high, n_low = int(high_mask.sum()), int((~high_mask).sum())
        print(f"{mouse} / {session_id} (reward_group={reward_group}): {len(whisker)} whisker trials match exactly -- "
              f"{n_high} high / {n_low} low")
        panels.append(dict(mouse=mouse, session_id=session_id, reward_group=reward_group,
                            p_mean=p_mean, p_low=p_low, p_high=p_high, p_chance=p_chance, high_mask=high_mask))

    if not panels:
        print("nothing to plot")
        return

    ncols = int(np.ceil(np.sqrt(len(panels)))) or 1
    nrows = int(np.ceil(len(panels) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False, constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, p in zip(axes_flat, panels):
        n = len(p["p_mean"])
        t = np.arange(n)
        shade_state_spans(ax, p["high_mask"], n)
        ax.fill_between(t, p["p_low"], p["p_high"], color="#1f77b4", alpha=0.20, lw=0, zorder=1, label="p_low-p_high")
        ax.plot(t, p["p_mean"], color="#1f77b4", lw=1.8, zorder=2, label="p_mean (p_lick)")
        ax.plot(t, p["p_chance"], color="#333333", lw=1.2, linestyle="--", zorder=2, label="p_chance")
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
        ax.set_title(f"{p['mouse']} ({p['reward_group']}, n={n} whisker trials)", fontsize=9)
        ax.set_xlabel("whisker trial index", fontsize=8)
        ax.set_ylabel("probability", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    # axvspan patches aren't auto-labeled, so add the state-shading legend
    # (green='high', red='low') as explicit proxy patches alongside the
    # curve lines' own handles, on the first panel only.
    import matplotlib.patches as mpatches
    curve_handles, curve_labels = axes_flat[0].get_legend_handles_labels()
    state_handles = [mpatches.Patch(color="#c8e6c9", label="state: high"), mpatches.Patch(color="#ffcdd2", label="state: low")]
    axes_flat[0].legend(handles=curve_handles + state_handles, labels=curve_labels + ["state: high", "state: low"],
                         fontsize=6, frameon=True, framealpha=0.9, loc="lower right")
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Curve-based perf-state allocation: p_mean/p_chance vs. high/low state (>=5-consecutive-trial rule)", fontsize=11)
    out_png = OUT_DIR / "044_perfstate_curve_diagnostic.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png.name}")


if __name__ == "__main__":
    main()
