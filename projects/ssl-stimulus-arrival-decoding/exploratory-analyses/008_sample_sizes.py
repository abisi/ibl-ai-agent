"""008 -- Sample sizes per area (user 2026-10-06, supplementary table): for each epoch (task / passive) and level (area
groups / areas), the data the pseudo-populations are drawn from = the eligible sessions of 001 (>= MIN_UNITS good + mua
units of the area and >= MIN_TRIALS trials per class): sessions, mice, units (sum over eligible sessions; median per
session), whisker and auditory trials (sum), and eligible sessions by stage (learning day 0 / expert days).
Reads the per-session caches (<home>/<epoch>/cache_all/*.npz; unit labels and trial labels only).
Output: <home>/tables/sample_sizes.csv
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
AR = importlib.import_module("_areas")
m1 = importlib.import_module("001_arrival_pseudopop")
HOME = AR.HOME


def sessions(epoch):
    out = {}
    for f in sorted((HOME / epoch / "cache_all").glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        out[f.stem] = dict(y=z["y"].astype(bool), ag=z["area_group"], af=z["area_fine"])
    return out


def main():
    U = pd.read_parquet(AR.UNITS, columns=["session_id", "mouse_id", "stage"]).drop_duplicates("session_id").set_index("session_id")
    rows = []
    for epoch in ("passive", "active"):
        S = sessions(epoch)
        for level, areas in AR.LEVELS.items():
            col = "ag" if level == "area_group" else "af"
            for a in areas:
                el = []
                for sid, s in S.items():
                    n = int((s[col] == a).sum())
                    if n >= m1.MIN_UNITS and min(s["y"].sum(), (~s["y"]).sum()) >= m1.MIN_TRIALS:
                        el.append((sid, n, int(s["y"].sum()), int((~s["y"]).sum())))
                if not el:
                    rows.append(dict(epoch=epoch, level=level, area=a, n_sessions=0))
                    continue
                d = pd.DataFrame(el, columns=["session_id", "units", "n_y1", "n_y0"])
                d["mouse"] = d.session_id.map(U.mouse_id)
                d["stage"] = d.session_id.map(U.stage)
                rows.append(dict(epoch=epoch, level=level, area=a, n_sessions=len(d), n_mice=d.mouse.nunique(),
                                 n_units=int(d.units.sum()), units_per_session_median=float(d.units.median()),
                                 n_whisker_trials=int(d.n_y1.sum()), n_auditory_trials=int(d.n_y0.sum()),
                                 sessions_by_stage="; ".join(f"{k}: {v}" for k, v in d.stage.value_counts().sort_index().items())))
    T = pd.DataFrame(rows)
    (HOME / "tables").mkdir(exist_ok=True)
    T.to_csv(HOME / "tables" / "sample_sizes.csv", index=False)
    # trial-sequence diagnostic (2026-10-06, passive pre-stimulus decoding offset): per session, trials in time order;
    # P(same type as the previous trial) vs its value for an independent sequence, and the whisker fraction in the first
    # and second half of the session's trials (passive: roughly the pre- and post-task blocks)
    seq = []
    for epoch in ("passive", "active"):
        for sid, s in sessions(epoch).items():
            y = s["y"].astype(int)
            if len(y) < 10:
                continue
            p, h = y.mean(), len(y) // 2
            seq.append(dict(epoch=epoch, session_id=sid, n_trials=len(y), p_whisker=p, p_repeat=(y[1:] == y[:-1]).mean(),
                            p_repeat_independent=p ** 2 + (1 - p) ** 2, p_whisker_first_half=y[:h].mean(),
                            p_whisker_second_half=y[h:].mean()))
    pd.DataFrame(seq).to_csv(HOME / "tables" / "trial_sequence_check.csv", index=False)
    print(T[T.level == "area_group"].to_string(index=False))


if __name__ == "__main__":
    main()
