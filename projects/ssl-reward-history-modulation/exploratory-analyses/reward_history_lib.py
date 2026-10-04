"""Shared helpers for the reward-history modulation project. See ../question.md
for the full locked definitions this code implements.
"""
import pandas as pd
import numpy as np

SSL_KS2_DIR = "reports/datasets/ssl_ks2_ephys/1.0.0"
SPIKE_QUANTIZATION_US = 100
RESPONSE_WINDOW_S = (0.005, 0.050)  # 5-50ms post-stimulus-onset
MIN_TRIALS_PER_BUCKET = 5

# perf -> raw stimulus-response label (TRIAL_MAP, from
# M:\analysis\Axel_Bisi\Github\ephys_utilities\neural_utils\neural_utils.py).
# NOT cohort-corrected: e.g. "whisker_hit" just means "licked on a whisker
# trial", regardless of whether that lick was rewarded for this mouse.
PERF_LABEL = {
    0: "whisker_miss", 1: "auditory_miss", 2: "whisker_hit", 3: "auditory_hit",
    4: "correct_rejection", 5: "false_alarm", 6: "association",
}


def normalize_context(trials):
    """Fix the context column session-by-session: some sessions store context
    as all-missing (real NaN or the literal string 'nan') because this NWB
    source has no BehavioralEpochs interface for them -- those sessions are
    single-context and should read as 'active' throughout (same fix as
    neural_utils.process_single_nwb). Mixed sessions get per-row NaN filled
    to 'active' too.
    """
    trials = trials.copy()
    trials["context"] = trials["context"].astype(str)

    def fix(group):
        if group.str.contains("nan").all():
            return pd.Series("active", index=group.index)
        return group.replace("nan", "active")

    trials["context"] = trials.groupby(trials["session_id"])["context"].transform(fix)
    return trials


def load_day0_trials(mouse_split):
    """Load trials.parquet restricted to day-0-whisker sessions of the given
    mice, with context normalized, non-active and association (perf==6)
    trials dropped.

    :param mouse_split: DataFrame with columns mouse_id, reward_group.
    :return: trials DataFrame, one row per trial, with an added
        'reward_group' column, sorted by (session_id, start_time).
    """
    sessions = pd.read_parquet(f"{SSL_KS2_DIR}/metadata/sessions.parquet")
    day0_sessions = sessions[
        (sessions["session_description"] == "whisker_0")
        & sessions["has_ephys"]
        & sessions["subject_id"].isin(mouse_split["mouse_id"])
    ]

    trials = pd.read_parquet(f"{SSL_KS2_DIR}/metadata/trials.parquet")
    trials = trials[trials["session_id"].isin(day0_sessions["session_id"])]
    trials = normalize_context(trials)
    trials = trials[(trials["context"] == "active") & (trials["perf"] != 6)]

    trials = trials.merge(day0_sessions[["session_id", "subject_id"]], on="session_id")
    trials = trials.rename(columns={"subject_id": "mouse_id"})
    trials = trials.merge(mouse_split[["mouse_id", "reward_group"]], on="mouse_id")

    return trials.sort_values(["session_id", "start_time"]).reset_index(drop=True)


def add_outcome_and_history(trials):
    """Add cohort-corrected outcome and trial-history columns.

    Adds:
    - modality: 'whisker' / 'auditory' / None (no_stim trials)
    - hit: True/False (cohort-corrected) for whisker/auditory trials, NaN for no_stim
    - prev_hit, prev_modality: same, for trial t-1 within the session
      (shift computed after already restricting to active, non-association
      trials, so t-1 is the previous *active* trial, not necessarily
      adjacent in raw trial_id).

    :param trials: output of load_day0_trials.
    :return: trials with the above columns added; first trial of each
        session has prev_hit/prev_modality == NaN (dropped downstream).
    """
    trials = trials.copy()
    modality_map = {"whisker_trial": "whisker", "auditory_trial": "auditory", "no_stim_trial": None}
    trials["modality"] = trials["trial_type"].map(modality_map)

    is_rplus = trials["reward_group"] == "R+"
    whisker = trials["modality"] == "whisker"
    auditory = trials["modality"] == "auditory"

    hit = pd.Series(pd.NA, index=trials.index, dtype="boolean")  # nullable: NaN stays NaN, not cast to False
    hit.loc[whisker & is_rplus] = trials.loc[whisker & is_rplus, "perf"] == 2   # whisker_hit
    hit.loc[whisker & ~is_rplus] = trials.loc[whisker & ~is_rplus, "perf"] == 0  # whisker_miss = R- hit
    hit.loc[auditory] = trials.loc[auditory, "perf"] == 3  # auditory_hit
    trials["hit"] = hit

    trials["prev_hit"] = trials.groupby("session_id")["hit"].shift(1)
    trials["prev_modality"] = trials.groupby("session_id")["modality"].shift(1)

    return trials
