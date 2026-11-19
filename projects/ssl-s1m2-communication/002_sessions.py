"""Session list: every session with >= MIN_UNITS SSp-bfd units and >= MIN_UNITS MO-wM1 or MO-wM2 units (good + mua,
rate >= 0.5 Hz; unit_info_all_days_v2), all days. Cohort and learner status per mouse from the canonical reference sheet
(share_internal), mice with exclude / exclude_ephys dropped. One row per (session, pair).

Run: python 002_sessions.py  ->  OUT/sessions.csv
"""
import pandas as pd

import s1m2_lib as L


def main():
    c = L.CONFIG
    u = pd.read_parquet(L.UNIT_TABLE)
    u = u[u.quality_label.isin(c["unit_quality"]) & (u.firing_rate >= c["min_rate_hz"])]
    ref = L.reference_sheet()
    ref = ref[(ref.exclude != 1) & (ref.exclude_ephys != 1)]
    ref = ref.set_index("subject_id")
    rows = []
    for (sid, mouse, day), g in u.groupby(["session_id", "mouse_id", "day"]):
        if mouse not in ref.index:
            continue
        n = g.area_acronym_custom.value_counts()
        for m2 in c["m2_areas"]:
            ns1, nm2 = int(n.get(c["s1_area"], 0)), int(n.get(m2, 0))
            if ns1 >= c["min_units"] and nm2 >= c["min_units"]:
                rows.append(dict(session_id=sid, mouse_id=mouse, day=int(day), stage="learning" if day == 0 else "expert",
                                 cohort=ref.loc[mouse, "reward_group"],
                                 learning_category=ref.loc[mouse, "learning_category"],
                                 pair=f"{c['s1_area']}__{m2}", m2_area=m2, n_s1=ns1, n_m2=nm2,
                                 s1_targets=";".join(sorted(g[g.area_acronym_custom == c["s1_area"]].target_region.dropna().unique())),
                                 m2_targets=";".join(sorted(g[g.area_acronym_custom == m2].target_region.dropna().unique()))))
    df = pd.DataFrame(rows)
    df["learner"] = df.learning_category.isin(["good", "moderate"])
    L.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(L.OUT / "sessions.csv", index=False)
    print(df.groupby(["pair", "stage", "cohort"]).agg(n=("session_id", "size"), learners=("learner", "sum"),
                                                      med_s1=("n_s1", "median"), med_m2=("n_m2", "median")).to_string())
    print("unique sessions", df.session_id.nunique(), "mice", df.mouse_id.nunique())


if __name__ == "__main__":
    main()
