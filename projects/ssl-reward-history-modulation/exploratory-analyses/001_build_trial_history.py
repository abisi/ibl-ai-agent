"""Build the day-0 trial table with cohort-corrected outcome and t-1 history
columns, for all eligible mice (exploration + confirmation). Saved once here
and reused by downstream scripts, split by mouse_split at analysis time.
"""
import pandas as pd
from reward_history_lib import load_day0_trials, add_outcome_and_history

mouse_split = pd.read_csv("projects/ssl-reward-history-modulation/artifacts/mouse_split.csv")

trials = load_day0_trials(mouse_split)
trials = add_outcome_and_history(trials)

print(f"{len(trials)} trials, {trials['session_id'].nunique()} sessions, {trials['mouse_id'].nunique()} mice")
print(trials.groupby(["reward_group", "modality"])["hit"].agg(["size", "mean"]))

trials.to_parquet("projects/ssl-reward-history-modulation/artifacts/trials_with_history.parquet", index=False)
