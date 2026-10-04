"""Build the eligible-mouse population and the exploration/confirmation split.

Eligibility: reward_group in {R+, R-}, learning_category in {moderate, good},
exclude_ephys == 0 (joint_mouse_reference_weight.xlsx), AND at least one
day-0-whisker session with ephys in the ssl_ks2_ephys dataset.

Split: stratified by cohort, 15 mice held out for exploration (8 R+, 7 R-),
remaining 59 for confirmation. See ../question.md for rationale.
"""
import pandas as pd
import numpy as np

REF_WEIGHT_XLSX = r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx"
SESSIONS_PARQUET = "reports/datasets/ssl_ks2_ephys/1.0.0/metadata/sessions.parquet"
OUT_CSV = "projects/ssl-reward-history-modulation/artifacts/mouse_split.csv"

N_EXPLORATION_RPLUS = 8
N_EXPLORATION_RMINUS = 7
SEED = 0

ref = pd.read_excel(REF_WEIGHT_XLSX, sheet_name="Sheet1")
eligible_ref = ref[
    ref["reward_group"].isin(["R+", "R-"])
    & ref["learning_category"].isin(["moderate", "good"])
    & (ref["exclude_ephys"] == 0)
]

sessions = pd.read_parquet(SESSIONS_PARQUET)
day0_ephys_subjects = set(
    sessions.loc[(sessions["session_description"] == "whisker_0") & sessions["has_ephys"], "subject_id"]
)

eligible = eligible_ref[eligible_ref["mouse_id"].isin(day0_ephys_subjects)][["mouse_id", "reward_group"]].copy()
assert len(eligible) == 74, f"expected 74 eligible mice, got {len(eligible)}"

rng = np.random.default_rng(SEED)
eligible["split"] = "confirmation"
for group, n_explore in [("R+", N_EXPLORATION_RPLUS), ("R-", N_EXPLORATION_RMINUS)]:
    mice = eligible.loc[eligible["reward_group"] == group, "mouse_id"].to_numpy()
    explore_mice = rng.choice(mice, size=n_explore, replace=False)
    eligible.loc[eligible["mouse_id"].isin(explore_mice), "split"] = "exploration"

print(eligible.groupby(["split", "reward_group"]).size())

eligible.to_csv(OUT_CSV, index=False)
print(f"Saved {len(eligible)} mice to {OUT_CSV}")
