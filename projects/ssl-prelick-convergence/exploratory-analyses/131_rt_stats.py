"""131 -- Reaction-time statistics of Fig1_task_data panel d, written to a table so the report can read them (user 2026-10-15:
stats for Fig1 d; previously computed ad hoc). Same session medians as 062 panel d: median RT per session and hit type
(>= 3 events), learners population, sessions of the tested-unit table; tests by m62.group_stats (MWU + Welch, cohort x stage
mouse permutation).
Output: across_days/<ref>/publication/learners/rt_stats.csv (+ rt_sessions.csv)
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../131_rt_stats.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m62 = importlib.import_module("062_pub_convergence_figures")
m51 = m62.m51
st26 = importlib.import_module("026_roc_rates_all_sessions")


def main():
    W = m62.load_population("learners")["W"]
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    info = W.drop_duplicates("session_id").set_index("session_id")[["cohort", "stage", "mouse_id"]]
    rows = []
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        for c in ("WH", "AH"):
            v = z["rt"][z["cls"] == c].astype(float); v = v[np.isfinite(v)]
            if len(v) >= 3 and r.session_id in info.index:
                rows.append(dict(session_id=r.session_id, cls=c, rt=np.median(v) * 1e3, **info.loc[r.session_id].to_dict()))
    RT = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for c in ("WH", "AH"):
        m62.group_stats(RT[RT.cls == c], "rt", f"median RT {c} (ms)", rng, f"131 RT {c}")
    out = m51.OUTROOT / "publication" / "learners"
    RT.to_csv(out / "rt_sessions.csv", index=False)
    pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("131")]).to_csv(out / "rt_stats.csv", index=False)
    print(pd.read_csv(out / "rt_stats.csv").iloc[:, :12].to_string())


if __name__ == "__main__":
    main()
