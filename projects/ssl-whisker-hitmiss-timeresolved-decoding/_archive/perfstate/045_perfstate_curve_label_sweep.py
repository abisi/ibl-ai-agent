"""Full-scale (all available sessions), decode-free label comparison
between the current block-median perf-state definition and the new
learning-curve-based one (user request 2026-09-15: "Do a more extensive
test with the new state definition"). Cheap by design -- no decoding, just
loading + label assignment + comparison -- so it can cover every eligible
session in a couple of minutes, unlike `043`'s 5-session decode test.

For every learning-stage (day 0), has_ephys session whose mouse has a
learning-curve H5 file:
  1. Assert the trial count matches exactly (same alignment logic as `043`/
     `044`, duplicated here -- see those scripts for why `prep_session`
     itself can't be reused for this).
  2. Compute both label sets, trial-level agreement, and -- directly
     motivating whether `hitmiss` is even a fair target to test this
     definition against -- the hit/miss class balance within each state,
     both definitions, so the class-imbalance skip pattern `043` hit is
     characterized at full scale rather than guessed at from 5 sessions.

Writes `045_perfstate_curve_label_sweep.csv` (one row per session) and a
summary figure.
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
from ssl_timeresolved_decoding import prep_perfstate_trials_generic  # noqa: E402

EPHYS_UTILS_PATH = axel_bisi_path("Github", "ephys_utilities")
if EPHYS_UTILS_PATH is not None:
    sys.path.insert(0, str(EPHYS_UTILS_PATH))
from ephys_utilities.helpers.load_helpers import load_learning_curves_data  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CURVE_ROOT = axel_bisi_path("combined_results_ks4")
N_CONSECUTIVE = 5
MIN_TRIALS_PER_CLASS = 5  # matches ssl_timeresolved_decoding.MIN_TRIALS_PER_CLASS


def active_trials_from_whisker_onset(session_id: str, trials_tbl: pd.DataFrame) -> pd.DataFrame:
    """Context filter + `perf != 6` ("association" outcome) filter +
    drop-before-first-whisker-trial -- matches `cd_analysis/utils/performance.py`'s
    `keep_active_from_whisker_onset` (both conditions applied together, not
    just context), NOT `ssl_bwm_trial_prep.prep_session` (which drops one
    extra trial for an unrelated t-1-outcome feature). The `perf != 6`
    clause (added 2026-09-15, was missing) matters specifically for
    subjects whose `context` field is the literal string "nan" (an older
    recording-era gap already documented in `ssl_bwm_trial_prep.prep_session`)
    -- for those, context filtering is a no-op, so `perf != 6` is the only
    thing excluding 'association'-outcome trials. Its absence caused a
    genuine trial-count mismatch (distinct from the earlier off-by-one bug)
    for 4/89 sessions in this sweep's first run."""
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
    """Verbatim of 043/044's helper -- port of `cd_analysis/utils/performance.py`'s `assign_expertise_blocks`."""
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


def class_counts(trials: pd.DataFrame, state_val: str) -> tuple[int, int]:
    sub = trials[trials["perf_state"] == state_val]
    y = sub["lick_flag"].to_numpy().astype(bool)
    return int(y.sum()), int((~y).sum())


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    learning = sessions_tbl[(sessions_tbl["session_description"] == "whisker_0") & (sessions_tbl["has_ephys"])]
    print(f"{len(learning)} learning-stage has_ephys sessions total")

    records = []
    for _, srow in learning.sort_values("subject_id").iterrows():
        session_id, mouse = srow["session_id"], srow["subject_id"]
        reward_group = load_reward_group(mouse)
        if reward_group is None:
            records.append(dict(mouse=mouse, session_id=session_id, reason="no usable reward_group"))
            continue
        reward_group_int = 1 if reward_group == "R+" else 0

        try:
            curves_df = load_learning_curves_data(str(CURVE_ROOT), [mouse])
        except ValueError:
            records.append(dict(mouse=mouse, session_id=session_id, reward_group=reward_group, reason="no curve file"))
            continue
        if len(curves_df) == 0:
            records.append(dict(mouse=mouse, session_id=session_id, reward_group=reward_group, reason="no curve file"))
            continue

        all_trials = active_trials_from_whisker_onset(session_id, trials_tbl)
        whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
        row = curves_df.iloc[0]
        p_low, p_chance = np.asarray(row["p_low"]), np.asarray(row["p_chance"])

        if len(p_low) != len(whisker):
            records.append(dict(mouse=mouse, session_id=session_id, reward_group=reward_group,
                                 reason=f"trial-count mismatch (ours={len(whisker)}, curve={len(p_low)})"))
            continue

        high_mask = assign_expertise_blocks_positional(p_low, p_chance, reward_group_int)
        new_trials = whisker.copy()
        new_trials["perf_state"] = np.where(high_mask, "high", "low")
        new_trials["reward_group"] = reward_group

        old_trials = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl,
                                                     decode_trial_types=["whisker_trial"])

        rec = dict(mouse=mouse, session_id=session_id, reward_group=reward_group, reason=None,
                   n_whisker_trials=len(whisker),
                   new_n_high=int(high_mask.sum()), new_n_low=int((~high_mask).sum()))
        for sv in ("high", "low"):
            h, m = class_counts(new_trials, sv)
            rec[f"new_{sv}_hit"], rec[f"new_{sv}_miss"] = h, m
            rec[f"new_{sv}_decodable"] = (h >= MIN_TRIALS_PER_CLASS) and (m >= MIN_TRIALS_PER_CLASS)

        if old_trials is not None:
            rec["old_n_high"] = int((old_trials["perf_state"] == "high").sum())
            rec["old_n_low"] = int((old_trials["perf_state"] == "low").sum())
            for sv in ("high", "low"):
                h, m = class_counts(old_trials, sv)
                rec[f"old_{sv}_hit"], rec[f"old_{sv}_miss"] = h, m
                rec[f"old_{sv}_decodable"] = (h >= MIN_TRIALS_PER_CLASS) and (m >= MIN_TRIALS_PER_CLASS)
            merged = old_trials[["trial_id", "perf_state"]].merge(
                new_trials[["trial_id", "perf_state"]], on="trial_id", suffixes=("_old", "_new"), how="inner")
            rec["label_agreement"] = float((merged["perf_state_old"] == merged["perf_state_new"]).mean()) if len(merged) else np.nan
            rec["n_trials_compared"] = len(merged)
        else:
            rec["old_n_high"] = rec["old_n_low"] = None
            rec["label_agreement"] = np.nan
            rec["n_trials_compared"] = 0

        records.append(rec)
        both_decodable = rec.get("new_high_decodable") and rec.get("new_low_decodable")
        print(f"{mouse} / {session_id} ({reward_group}): {len(whisker)} trials match exactly, "
              f"new={rec['new_n_high']}h/{rec['new_n_low']}l, agreement={rec.get('label_agreement', float('nan')):.1%}, "
              f"both-states-decodable(new)={both_decodable}")

    df = pd.DataFrame(records)
    out_csv = OUT_DIR / "045_perfstate_curve_label_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    ok = df[df["reason"].isna()]
    skipped = df[df["reason"].notna()]
    print(f"\n{len(ok)}/{len(df)} sessions usable (exact trial-count match); skipped breakdown:")
    print(skipped["reason"].value_counts().to_string())
    print(f"\nlabel agreement (old vs new), n={ok['label_agreement'].notna().sum()}: "
          f"mean={ok['label_agreement'].mean():.1%}, median={ok['label_agreement'].median():.1%}, "
          f"range=[{ok['label_agreement'].min():.1%}, {ok['label_agreement'].max():.1%}]")
    for defn in ("old", "new"):
        for sv in ("high", "low"):
            col = f"{defn}_{sv}_decodable"
            if col in ok.columns:
                print(f"  {defn} def, {sv} state: {ok[col].sum()}/{ok[col].notna().sum()} sessions have >= {MIN_TRIALS_PER_CLASS} "
                      f"hits AND >= {MIN_TRIALS_PER_CLASS} misses (decodable)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].hist(ok["label_agreement"].dropna() * 100, bins=20, color="#1f77b4", edgecolor="white")
    axes[0].set_xlabel("trial-level label agreement (old vs new), %")
    axes[0].set_ylabel("sessions")
    axes[0].set_title(f"Label agreement (n={ok['label_agreement'].notna().sum()})", fontsize=10)
    axes[0].set_box_aspect(1)
    axes[0].spines[["top", "right"]].set_visible(False)

    decodable_counts = pd.DataFrame({
        "old high": ok.get("old_high_decodable", pd.Series(dtype=bool)),
        "old low": ok.get("old_low_decodable", pd.Series(dtype=bool)),
        "new high": ok.get("new_high_decodable", pd.Series(dtype=bool)),
        "new low": ok.get("new_low_decodable", pd.Series(dtype=bool)),
    }).sum()
    axes[1].bar(decodable_counts.index, decodable_counts.values, color=["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"])
    axes[1].set_ylabel("sessions with >=5 hits AND >=5 misses")
    axes[1].set_title("Hitmiss class-balance feasibility by state/def", fontsize=10)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_box_aspect(1)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].scatter(ok["new_n_high"], ok["new_n_low"], c=(ok["reward_group"] == "R+").map({True: "#00B400", False: "#C800C8"}), alpha=0.7)
    axes[2].set_xlabel("new def: n_high trials")
    axes[2].set_ylabel("new def: n_low trials")
    axes[2].set_title("New-def high/low trial counts per session\n(green=R+, magenta=R-)", fontsize=10)
    axes[2].set_box_aspect(1)
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Perf-state definition comparison, full sweep ({len(ok)} usable / {len(df)} candidate sessions)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = OUT_DIR / "045_perfstate_curve_label_sweep.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png.name}")


if __name__ == "__main__":
    main()
