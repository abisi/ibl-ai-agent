"""Detail view of the 8 sessions flagged by `detect_terminal_disengagement`
(user request 2026-09-24: "Show the sessions with disengagement. I am not
sure I want to remove disengagement yet.").

Per session, one column, x = active-epoch trial index (all trial types, the
curve-aligned trial set), shared across rows:
  1. running lick rate (RUN-trial centered window) per trial type: whisker
     (cohort color), auditory (blue), no-stim / false alarm (grey).
  2. lick raster per trial type (tick = lick, dot = no lick).
  3. whisker learning curve (p_mean, 80% CI, p_chance), on the whisker-trial
     positions within the full trial sequence.
Red line = learning_trial; black line = disengagement cut (start of the
trailing block after the last lick); the shaded block is what 095 drops.
Title: whisker / auditory / no-stim trials in the dropped block, and how many
whisker hits/misses the post epoch has with vs without the drop.

Output: figures/whole_brain/learning/095b_disengaged_sessions_detail.png/.pdf
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import (  # noqa: E402
    _active_trials_from_whisker_onset_for_curve, detect_terminal_disengagement, hitmiss_session_list,
    load_whisker_curve_row, prep_hitmiss_trials_learning_split,
)

_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
TYPE_STYLE = {"whisker_trial": ("whisker", None, 2), "auditory_trial": ("auditory", "#1f77b4", 1),
              "no_stim_trial": ("no-stim (FA)", "#777777", 0)}
RUN = 10


def running(x: np.ndarray, pos: np.ndarray, n_total: int, run: int) -> np.ndarray:
    out = np.full(n_total, np.nan)
    for i in range(n_total):
        m = np.abs(pos - i) <= run
        if m.sum() >= 3:
            out[i] = x[m].mean()
    return out


def main():
    root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(root / "metadata" / "trials.parquet")
    sess = hitmiss_session_list(sessions_tbl)
    sess = sess[sess.day_stage == "learning"]
    flagged = [s for s in sess.itertuples() if detect_terminal_disengagement(s.session_id, trials_tbl)["disengaged"]]
    curve_root = axel_bisi_path("combined_results_ks4")

    fig, axes = plt.subplots(3, len(flagged), figsize=(3.6 * len(flagged), 8.5), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.2, 0.7, 1.0]), constrained_layout=True, squeeze=False)
    for c, s in enumerate(flagged):
        a = _active_trials_from_whisker_onset_for_curve(s.session_id, trials_tbl)
        n = len(a)
        dis = detect_terminal_disengagement(s.session_id, trials_tbl)
        cut = int(np.where(a.start_time.to_numpy() == dis["t_cut"])[0][0])
        trials, info = prep_hitmiss_trials_learning_split(root, s.session_id, sessions_tbl, trials_tbl)
        wh_pos = np.where(a.trial_type.to_numpy() == "whisker_trial")[0]
        lt = int(info["learning_trial"]) if not np.isnan(info["learning_trial"]) else None
        col = COHORT_COLOR[s.reward_group]

        ax = axes[0, c]
        for tt, (lab, tcol, _) in TYPE_STYLE.items():
            pos = np.where(a.trial_type.to_numpy() == tt)[0]
            ax.plot(np.arange(n), running(a.lick_flag.to_numpy()[pos].astype(float), pos, n, RUN),
                    color=tcol or col, lw=1.6, label=lab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(f"lick rate (+-{RUN} trials)", fontsize=8)

        ax = axes[1, c]
        for tt, (lab, tcol, row) in TYPE_STYLE.items():
            m = a.trial_type.to_numpy() == tt
            lk = a.lick_flag.to_numpy() == 1
            ax.scatter(np.where(m & lk)[0], np.full((m & lk).sum(), row), marker="|", s=40, color=tcol or col)
            ax.scatter(np.where(m & ~lk)[0], np.full((m & ~lk).sum(), row), marker=".", s=4, color="#cccccc")
        ax.set_yticks([0, 1, 2], ["no-stim", "aud", "whisker"], fontsize=7)
        ax.set_ylim(-0.6, 2.6)

        ax = axes[2, c]
        curve = load_whisker_curve_row(curve_root, s.subject_id, 0)
        if curve is not None and len(curve["p_mean"]) == len(wh_pos):
            ax.fill_between(wh_pos, curve["p_low"], curve["p_high"], color=col, alpha=0.2, lw=0)
            ax.plot(wh_pos, curve["p_mean"], color=col, lw=1.5, label="p(lick | whisker)")
            ax.plot(wh_pos, curve["p_chance"], color="#555555", lw=1, ls="--", label="p_chance (FA)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("learning curve", fontsize=8)
        ax.set_xlabel("active trial index (all types)", fontsize=8)

        y = trials.lick_flag.to_numpy().astype(bool)
        post = (trials.lt_epoch == "post").to_numpy()
        kept = trials.start_time.to_numpy() < dis["t_cut"]
        for r in range(3):
            axes[r, c].axvspan(cut, n, color="#e0e0e0", zorder=0)
            axes[r, c].axvline(cut, color="k", lw=1.4)
            if lt is not None and lt < len(wh_pos):
                axes[r, c].axvline(wh_pos[lt], color="#d62728", lw=1.2)
            axes[r, c].tick_params(labelsize=7)
            axes[r, c].spines[["top", "right"]].set_visible(False)
        n_ns = int((a.iloc[cut:].trial_type == "no_stim_trial").sum())
        axes[0, c].set_title(
            f"{s.session_id} ({s.reward_group}, {s.learning_category})\n"
            f"dropped block: {dis['n_whisker_dropped']} whisker, {dis['n_auditory_in_run']} auditory, {n_ns} no-stim\n"
            f"post epoch whisker hit/miss: {int((y & post).sum())}/{int((~y & post).sum())} kept all -> "
            f"{int((y & post & kept).sum())}/{int((~y & post & kept).sum())} after drop", fontsize=7.5)
        if c == 0:
            axes[0, c].legend(frameon=False, fontsize=7, loc="upper right")
            axes[2, c].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Sessions flagged as terminally disengaged (095 rule: after the last lick on any trial type, >=5 whisker AND >=3 "
                 "auditory trials unlicked). Grey = dropped block; black = cut; red = learning_trial", fontsize=10)
    out = q034.fig_dir("whole_brain") / "095b_disengaged_sessions_detail.png"
    q034.savefig_retry(fig, out, dpi=220, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
