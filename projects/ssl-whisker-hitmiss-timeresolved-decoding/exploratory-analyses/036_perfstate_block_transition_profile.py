"""Perf-state block-transition behavioral profile (user idea, 2026-09-13:
"since perf-state is defined in blocks, check whether decoding accuracy
ramps gradually across a block or steps sharply at the block boundary").
Behavioral-only version of that question (no neural decoding, no new heavy
compute) -- characterizes the underlying signal that defines perf_state
itself (block hit-rate) around state transitions, as groundwork before
deciding whether a full neural per-trial-position decode (which WOULD need
a new compute-heavy sweep, not done here) is worth it.

For every state transition (a block whose median-split state differs from
the immediately preceding block, `low_to_high`/`high_to_low`), pulls
`N_BLOCKS_AROUND` blocks of whisker trials before and after the boundary
and records each trial's offset (in trials, not blocks) from the boundary
(trial 0 = first trial of the new block) and its cohort-corrected hit
(`rewarded`). Transitions too close to a session's start/end (window would
run off the trial sequence) are dropped, not padded, so every offset has
the same number of contributing sessions.

Unit of analysis: session (`ssl_stats_unit_of_analysis.md`) -- per session,
per direction, per offset, first averages across that session's own
(possibly several) transitions of that direction, then averages across
sessions for the plotted mean+-SEM. Purely descriptive (peri-transition
profile), no significance test.

Usage: python 036_perfstate_block_transition_profile.py
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
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import PERF_BLOCK_SIZE, hitmiss_session_list, prep_perfstate_trials_generic  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures" / "behavior"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_BLOCKS_AROUND = 2
DAY_STAGE = "learning"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
AGGREGATE_COLOR = "#2c5f5b"
DIRECTIONS = ["low_to_high", "high_to_low"]


def session_transition_profile(dataset_root, session_id, sessions_tbl, trials_tbl) -> pd.DataFrame | None:
    trials = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"])
    if trials is None or len(trials) == 0:
        return None
    trials = trials.sort_values("start_time").reset_index(drop=True)
    block_state = trials.groupby("block_id")["perf_state"].first().sort_index()
    block_start_idx = trials.groupby("block_id").apply(lambda d: d.index.min())
    block_ids = block_state.index.to_numpy()
    window_span = N_BLOCKS_AROUND * PERF_BLOCK_SIZE

    rows = []
    for pos in range(1, len(block_ids)):
        prev_b, cur_b = block_ids[pos - 1], block_ids[pos]
        prev_state, cur_state = block_state.loc[prev_b], block_state.loc[cur_b]
        if prev_state == cur_state:
            continue
        direction = f"{prev_state}_to_{cur_state}"
        if direction not in DIRECTIONS:
            continue
        boundary_idx = int(block_start_idx.loc[cur_b])
        lo, hi = boundary_idx - window_span, boundary_idx + window_span
        if lo < 0 or hi > len(trials):
            continue
        sub = trials.iloc[lo:hi][["rewarded"]].copy()
        sub["offset"] = np.arange(lo, hi) - boundary_idx
        sub["direction"] = direction
        rows.append(sub)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    sessions = hitmiss_session_list(sessions_tbl)
    sessions = sessions[sessions.day_stage == DAY_STAGE]

    per_session_rows = []
    n_with_data = 0
    for r in sessions.itertuples():
        prof = session_transition_profile(dataset_root, r.session_id, sessions_tbl, trials_tbl)
        if prof is None:
            continue
        n_with_data += 1
        session_mean = prof.groupby(["direction", "offset"])["rewarded"].mean().reset_index()
        session_mean["session_id"] = r.session_id
        session_mean["reward_group"] = r.reward_group
        per_session_rows.append(session_mean)
    print(f"{n_with_data}/{len(sessions)} sessions contributed at least one clean transition window")

    all_sessions = pd.concat(per_session_rows, ignore_index=True)

    fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(6.5 * len(DIRECTIONS), 5.5), squeeze=False)
    for ax, direction in zip(axes[0], DIRECTIONS):
        sub_dir = all_sessions[all_sessions.direction == direction]
        groups = [("R+", sub_dir[sub_dir.reward_group == "R+"], COHORT_COLOR["R+"]),
                  ("R-", sub_dir[sub_dir.reward_group == "R-"], COHORT_COLOR["R-"]),
                  ("R+ & R- aggregated", sub_dir, AGGREGATE_COLOR)]
        for label, g, color in groups:
            stats = g.groupby("offset")["rewarded"].agg(["mean", "sem", "count"]).reset_index()
            if len(stats) == 0:
                continue
            ax.plot(stats["offset"], stats["mean"], color=color, lw=2.0, label=f"{label} (n={g['session_id'].nunique()} sessions)")
            ax.fill_between(stats["offset"], stats["mean"] - stats["sem"], stats["mean"] + stats["sem"], color=color, alpha=0.15, lw=0)
        ax.axvline(-0.5, color="#333333", lw=1.2, linestyle="-", alpha=0.6, zorder=0)
        ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
        ax.set_ylim(-0.02, 1.02)  # headroom so markers/curves at 0/1 aren't clipped
        ax.set_box_aspect(1)
        ax.set_xlabel(f"trial offset from block boundary (block size={PERF_BLOCK_SIZE})")
        ax.set_ylabel("hit rate")
        ax.set_title(direction.replace("_", " "), fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="lower right" if direction == "low_to_high" else "upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Perf-state block-transition hit-rate profile ({DAY_STAGE}-stage, {N_BLOCKS_AROUND} blocks each side)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = FIG_DIR / "036_perfstate_block_transition_profile.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
