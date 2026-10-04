"""Step 1 of the open items in question.md: check whether events.parquet
carries a reward/spout/feedback-delivery timestamp distinct from
trials.start_time, and survey session/day-stage/trial counts to pick 1-2
sessions for the small-scale sanity check.
"""

from __future__ import annotations

import pandas as pd

from ibl_ai_agent.data_locations import resolve_dataset_dir

d = resolve_dataset_dir("ssl_ephys")
print(f"ssl_ephys root: {d}")

sessions = pd.read_parquet(d / "metadata" / "sessions.parquet")
trials = pd.read_parquet(d / "metadata" / "trials.parquet")
events = pd.read_parquet(d / "metadata" / "events.parquet")

print(f"\nsessions: {len(sessions)} rows")
print(f"trials: {len(trials)} rows, columns: {list(trials.columns)}")
print(f"events: {len(events)} rows, columns: {list(events.columns)}")

print("\nevent_type value counts (top 30):")
print(events["event_type"].value_counts().head(30))

# Sessions with ephys, for picking a small sanity-check subset.
ephys_sessions = sessions[sessions["has_ephys"]].copy()
ephys_sessions["day_tag"] = ephys_sessions["session_description"].fillna("")
print(f"\nephys sessions: {len(ephys_sessions)}")
print(ephys_sessions[["session_id", "subject_id", "session_description", "n_units", "n_trials"]].head(20))

# For one example ephys session, look at events near a few trial start_times
# to see if there's a reward/spout/lick event that isn't start_time itself.
example_session = ephys_sessions.iloc[0]["session_id"]
print(f"\nExample session for event-timing check: {example_session}")
sess_trials = trials[trials["session_id"] == example_session].sort_values("start_time").head(10)
sess_events = events  # events.parquet has no session_id column per schema doc; verify below
print("trials columns present:", list(sess_trials.columns))
print(sess_trials)
