"""Session list (v2): sessions with >= MIN_UNITS stable good + mua units in SSp-bfd and in an M2 set (pooled MO-wM1 +
MO-wM2 = "wMO", and each separately), all days. Stable = 137_stable_units.stable. Cohort and learner status per mouse
from the canonical reference sheet; mice with exclude / exclude_ephys dropped. One row per (session, pair).

Run: python 002_sessions.py  ->  OUT/sessions_v2.csv
"""
import numpy as np
import pandas as pd

import s1m2_lib as L


def stable_units():
    """unit_info_all_days_v2 good / mua units joined to the stable flag (137 table, NWB cluster ids)."""
    c = L.CONFIG
    u = pd.read_parquet(L.UNIT_TABLE)
    u["cluster_id"] = u.cluster_id.astype(int)
    s = pd.read_parquet(L.STABLE_TABLE, columns=["session_id", "electrode_group", "nwb_cluster_id", "stable"])
    s = s.rename(columns={"nwb_cluster_id": "cluster_id"})
    u = u.merge(s, on=["session_id", "electrode_group", "cluster_id"], how="left")
    u = u[u.quality_label.isin(c["unit_quality"]) & u.stable.fillna(False).astype(bool)
          & (u.firing_rate >= c["min_rate_hz"])]
    return u


def main():
    c = L.CONFIG
    u = stable_units()
    ref = L.reference_sheet()
    ref = ref[(ref.exclude != 1) & (ref.exclude_ephys != 1)].set_index("subject_id")
    rows = []
    for (sid, mouse, day), g in u.groupby(["session_id", "mouse_id", "day"]):
        if mouse not in ref.index:
            continue
        s1 = g[g.area_acronym_custom == c["s1_area"]]
        for name, areas in c["m2_sets"].items():
            m2 = g[g.area_acronym_custom.isin(areas)]
            if len(s1) >= c["min_units"] and len(m2) >= c["min_units"]:
                rows.append(dict(session_id=sid, mouse_id=mouse, day=int(day), stage="learning" if day == 0 else "expert",
                                 cohort=ref.loc[mouse, "reward_group"], learning_category=ref.loc[mouse, "learning_category"],
                                 pair=f"{c['s1_area']}__{name}", m2_set=name, n_s1=len(s1), n_m2=len(m2),
                                 s1_targets=";".join(sorted(s1.target_region.dropna().unique())),
                                 m2_targets=";".join(sorted(m2.target_region.dropna().unique()))))
    df = pd.DataFrame(rows)
    df["learner"] = df.learning_category.isin(["good", "moderate"])
    df.to_csv(L.OUT / "sessions_v2.csv", index=False)
    print(df.groupby(["pair", "stage", "cohort"]).agg(n=("session_id", "size"), mice=("mouse_id", "nunique"),
                                                      med_s1=("n_s1", "median"), med_m2=("n_m2", "median")).to_string())


if __name__ == "__main__":
    main()
