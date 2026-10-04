"""Per-session hit/miss whisker-trial counts, expert stage (user request
2026-09-17: "Plot the hit/miss trial counts for each expert sessions" --
follow-up to finding that `hitmiss_stim_expert_whole_brain` has ZERO
computed rows, every session failing `data_sufficiency_ok`'s
`MIN_TRIALS_PER_CLASS=5`-per-class floor).

Hit/miss = `lick_flag` on whisker trials, cohort-independent (same
`prep_hitmiss_trials`/`y = lick_flag` definition `024_master_sweep.py`
itself uses for `decode_target='hitmiss'`), counted over the WHOLE
session (the most favorable case for passing the sufficiency floor --
the `half`/`perfstate` conditions split this further and can only do
worse). Session list is `hitmiss_session_list(sessions_tbl)` filtered to
`day_stage=='expert'` -- the same canonical list `024_master_sweep.py`
uses for every decode target, so this includes every candidate session,
not just the 4 that happened to reach a decode attempt in the last sweep
(some fail earlier, e.g. insufficient units, and never even reach the
class-count check).

Diverging horizontal bar per session: hit trials extend right, miss
trials extend left, colored by cohort (this project's standing R+/R-
convention) -- with a dashed reference line at +-MIN_TRIALS_PER_CLASS (5)
so which sessions clear the sufficiency floor is visible directly. Sorted
by whichever count is smaller (the one `data_sufficiency_ok` actually
gates on), ascending, so the most severely imbalanced sessions are at top.

Usage: python 048_expert_hitmiss_trial_counts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import MIN_TRIALS_PER_CLASS, hitmiss_session_list, prep_hitmiss_trials  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    candidates = hitmiss_session_list(sessions_tbl)
    candidates = candidates[candidates["day_stage"] == "expert"]
    print(f"{len(candidates)} expert-stage candidate sessions")

    rows = []
    for _, r in candidates.iterrows():
        trials = prep_hitmiss_trials(dataset_root, r["session_id"], sessions_tbl, trials_tbl)
        if trials is None or len(trials) == 0:
            rows.append(dict(session_id=r["session_id"], subject_id=r["subject_id"], reward_group=r["reward_group"],
                              n_hit=0, n_miss=0, reason="no usable whisker trials"))
            continue
        n_hit = int((trials["lick_flag"] == 1).sum())
        n_miss = int((trials["lick_flag"] == 0).sum())
        rows.append(dict(session_id=r["session_id"], subject_id=r["subject_id"], reward_group=r["reward_group"],
                          n_hit=n_hit, n_miss=n_miss, reason=None))

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "048_expert_hitmiss_trial_counts.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv.name}")

    n_ok = int(((df["n_hit"] >= MIN_TRIALS_PER_CLASS) & (df["n_miss"] >= MIN_TRIALS_PER_CLASS)).sum())
    print(f"{n_ok}/{len(df)} sessions clear the {MIN_TRIALS_PER_CLASS}-per-class floor (whole-session count)")

    df["min_class"] = df[["n_hit", "n_miss"]].min(axis=1)
    df = df.sort_values("min_class", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(df) + 1.5))
    y = np.arange(len(df))
    colors = [COHORT_COLOR.get(rg, "#888888") for rg in df["reward_group"]]
    ax.barh(y, df["n_hit"], color=colors, alpha=0.9, label="hit (lick_flag=1)")
    ax.barh(y, -df["n_miss"], color=colors, alpha=0.45, label="miss (lick_flag=0)")
    ax.axvline(MIN_TRIALS_PER_CLASS, color="#333333", lw=1, linestyle="--")
    ax.axvline(-MIN_TRIALS_PER_CLASS, color="#333333", lw=1, linestyle="--")
    ax.axvline(0, color="#888888", lw=0.8)
    labels = [f"{sid} ({rg})" for sid, rg in zip(df["session_id"], df["reward_group"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    for tick, rg in zip(ax.get_yticklabels(), df["reward_group"]):
        tick.set_color(COHORT_COLOR.get(rg, "#888888"))
    xmax = max(df["n_hit"].max(), df["n_miss"].max()) * 1.1
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("trial count  <- miss  |  hit ->")
    ax.set_title(f"Expert-stage whisker hit/miss trial counts, whole session\n"
                 f"(dashed = MIN_TRIALS_PER_CLASS={MIN_TRIALS_PER_CLASS}; {n_ok}/{len(df)} sessions clear it on both sides)",
                 fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.9, label="hit"),
               plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.45, label="miss")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_png = OUT_DIR / "048_expert_hitmiss_trial_counts.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png.name}")


if __name__ == "__main__":
    main()
