"""Session list: whisker day 0 (learning) and expert (day >= 1) sessions with stable good + mua units, cohort and
learner status from the reference sheet (exclude / exclude_ephys dropped, R+proba dropped). Pilot: 5 day-0 sessions per
cohort (stratified by learning_category, seed 0) + 2 expert sessions per cohort.

Run: python 001_sessions.py  ->  OUT/tables/sessions.csv, OUT/tables/units.parquet
"""
import numpy as np
import pandas as pd

import tca_lib as L


def main():
    u = L.stable_units()
    ref = L.reference_sheet()
    ref = ref[(ref.exclude != 1) & (ref.exclude_ephys != 1) & ref.reward_group.isin(["R+", "R-"])].set_index("subject_id")
    u = u[u.mouse_id.isin(ref.index)].copy()
    # recording gates day 0 only (ssl_dataset_inclusion rule 2)
    rec = u.mouse_id.map(ref.recording)
    u = u[(u.day > 0) | (rec == 1)]
    u["cohort"] = u.mouse_id.map(ref.reward_group)
    s = (u.groupby(["session_id", "mouse_id", "day", "cohort"])
         .agg(n_units=("cluster_id", "size"), n_areas=("area_acronym_custom", "nunique")).reset_index())
    s["learning_category"] = s.mouse_id.map(ref.learning_category)
    s["learner"] = s.learning_category.isin(["good", "moderate"])
    s["stage"] = np.where(s.day == 0, "learning", "expert")
    assert s.session_id.is_unique
    rng = np.random.default_rng(0)
    s["pilot"] = False
    for coh, g in s[s.day == 0].groupby("cohort"):
        # 5 per cohort, spread over learning categories (largest categories first), >= 300 units
        g = g[g.n_units >= 300]
        cats = g.learning_category.value_counts().index.tolist()
        pick = []
        while len(pick) < 5:
            for c in cats:
                pool = g[(g.learning_category == c) & ~g.session_id.isin(pick)]
                if len(pool) and len(pick) < 5:
                    pick.append(pool.session_id.iloc[rng.integers(len(pool))])
        s.loc[s.session_id.isin(pick), "pilot"] = True
    for coh, g in s[s.day > 0].groupby("cohort"):
        g = g[g.n_units >= 300]
        s.loc[s.session_id.isin(rng.choice(g.session_id, 2, replace=False)), "pilot"] = True
    (L.OUT / "tables").mkdir(parents=True, exist_ok=True)
    s.to_csv(L.OUT / "tables" / "sessions.csv", index=False)
    u[u.session_id.isin(s.session_id)].to_parquet(L.OUT / "tables" / "units.parquet", index=False)
    print(s.groupby(["stage", "cohort"]).agg(sessions=("session_id", "size"), mice=("mouse_id", "nunique"),
                                             learners=("learner", "sum"), med_units=("n_units", "median")).to_string())
    print(s[s.pilot].sort_values(["stage", "cohort"]).to_string(index=False))


if __name__ == "__main__":
    main()
