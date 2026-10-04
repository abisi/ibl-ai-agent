"""Sanity check for `detect_terminal_disengagement` (used by 095): for every
learning-stage hit/miss session, the trailing block after the last lick on
any trial type, and whether it passes the "clearly a state" rule (>=5
whisker AND >=3 auditory trials). Figure: per session, lick outcome per
trial by trial type (whisker / auditory / no-stim rows) across the active
epoch, learning_trial in red, disengagement cut in black (sessions that
pass) or grey dashed (trailing block too short to count).

Outputs: 095a_disengagement_check.csv, 095a_disengagement_check.png
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
from ssl_timeresolved_decoding import (  # noqa: E402
    _active_trials_from_whisker_onset_for_curve, detect_terminal_disengagement, hitmiss_session_list,
    prep_hitmiss_trials_learning_split,
)

OUT_DIR = Path(__file__).resolve().parent
ROWS = {"whisker_trial": 2, "auditory_trial": 1, "no_stim_trial": 0}
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def main():
    root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(root / "metadata" / "trials.parquet")
    sess = hitmiss_session_list(sessions_tbl)
    sess = sess[sess.day_stage == "learning"].reset_index(drop=True)

    recs, plot_data = [], []
    for s in sess.itertuples():
        all_t = _active_trials_from_whisker_onset_for_curve(s.session_id, trials_tbl)
        licked = np.where(all_t.lick_flag.to_numpy() == 1)[0]
        trail = all_t.iloc[licked[-1] + 1:] if len(licked) else all_t.iloc[0:0]
        dis = detect_terminal_disengagement(s.session_id, trials_tbl)
        _, info = prep_hitmiss_trials_learning_split(root, s.session_id, sessions_tbl, trials_tbl)
        recs.append(dict(session_id=s.session_id, reward_group=s.reward_group, n_active=len(all_t),
                         trail_whisker=int((trail.trial_type == "whisker_trial").sum()),
                         trail_auditory=int((trail.trial_type == "auditory_trial").sum()),
                         trail_nostim=int((trail.trial_type == "no_stim_trial").sum()),
                         disengaged=dis["disengaged"], learning_trial=info["learning_trial"]))
        plot_data.append((s, all_t, len(all_t) - len(trail), dis["disengaged"], info))
    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / "095a_disengagement_check.csv", index=False)
    print(df.groupby("reward_group").disengaged.agg(["sum", "count"]))
    print(df[df.disengaged][["trail_whisker", "trail_auditory"]].describe().round(1))
    print("near-misses (trailing block with >=3 whisker but rule not met):")
    print(df[~df.disengaged & (df.trail_whisker >= 3)].to_string())

    ncol = 6
    nrow = int(np.ceil(len(plot_data) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 1.25 * nrow), constrained_layout=True)
    for ax, (s, all_t, cut_pos, dis, info) in zip(axes.flat, plot_data):
        x = np.arange(len(all_t))
        yrow = all_t.trial_type.map(ROWS).to_numpy()
        lick = all_t.lick_flag.to_numpy() == 1
        ax.scatter(x[lick], yrow[lick], s=5, color=COHORT_COLOR[s.reward_group], marker="|")
        ax.scatter(x[~lick], yrow[~lick], s=2, color="#bbbbbb", marker=".")
        if not np.isnan(info["learning_trial"]):
            wh_pos = np.where(all_t.trial_type.to_numpy() == "whisker_trial")[0]
            lt = int(info["learning_trial"])
            if lt < len(wh_pos):
                ax.axvline(wh_pos[lt], color="#d62728", lw=1)
        if cut_pos < len(all_t):
            ax.axvline(cut_pos, color="k" if dis else "#999999", lw=1.4 if dis else 0.8, ls="-" if dis else "--")
        ax.set_yticks([0, 1, 2], ["ns", "aud", "wh"], fontsize=5)
        ax.tick_params(axis="x", labelsize=5)
        ax.set_title(f"{s.session_id[:5]} {s.reward_group}{' DISENGAGED' if dis else ''}", fontsize=6.5,
                     color="k" if dis else "#666666")
        ax.set_ylim(-0.6, 2.6)
    for ax in list(axes.flat)[len(plot_data):]:
        ax.axis("off")
    fig.suptitle("Terminal disengagement check: licks (ticks) vs no-lick (grey dots) per trial type; red = learning_trial; "
                 "black = cut (>=5 whisker & >=3 auditory unlicked after last lick), grey dashed = trailing block too short",
                 fontsize=9)
    fig.savefig(OUT_DIR / "095a_disengagement_check.png", dpi=170)


if __name__ == "__main__":
    main()
