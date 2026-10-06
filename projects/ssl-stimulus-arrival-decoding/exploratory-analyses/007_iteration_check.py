"""007 -- Does the number of iterations change the result? (user 2026-10-06; led to the 500-iteration default)
Task (active) trials, N = 200: the N-sweep run (<home>/active/, 100 iterations x 10 shuffles) vs the areas of the final
run that reached 1000 iterations x 20 shuffles (<home>/active_final_n200/). Per area: onset (zoom resolution, 004 rule),
mean corrected accuracy 5-50 ms, max |difference| and correlation of the mean time courses, number of bins whose
above-chance call differs; and the spread of the onset over 200 random subsets of 100 of the 1000 final iterations
(how much a 100-iteration run can move by chance).
Output: <home>/tables/iteration_check_n200.csv, <home>/provenance_007.json
"""
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
AR = importlib.import_module("_areas")
m2 = importlib.import_module("002_arrival_summary")
HOME = AR.HOME
N_MAIN, N_FULL, N_PILOT, N_SUB = 200, 1000, 100, 200


def load(folder):
    m2.OUT = folder
    D = m2.load_raw()
    return D[D.N == N_MAIN]


def onset(x):
    return 1000 * m2.onset(m2.ZOOM_T, np.percentile(x, 5, axis=0) > 0, 13, 11)


def main():
    F = load(HOME / "active_final_n200")
    n = F.groupby(["level", "area"]).rep.nunique()
    full = n[n >= N_FULL].index
    P = load(HOME / "active")
    nw = len(m2.WIDE_T)
    win = (m2.ZOOM_T >= m2.WIN[0] - 1e-9) & (m2.ZOOM_T <= m2.WIN[1] + 1e-9)
    rng = np.random.default_rng(1)
    rows = []
    for level, area in full:
        dp = m2.d_matrix(P[(P.level == level) & (P.area == area)])
        df = m2.d_matrix(F[(F.level == level) & (F.area == area) & (F.rep < N_FULL)])
        zp, zf = dp[:, nw:], df[:, nw:]
        sub = np.array([onset(zf[rng.choice(len(zf), N_PILOT, replace=False)]) for _ in range(N_SUB)])
        rows.append(dict(level=level, area=area, n_iter_pilot=len(dp), n_iter_final=len(df),
                         onset_pilot=onset(zp), onset_final=onset(zf), onset_sub100_median=np.nanmedian(sub),
                         onset_sub100_lo=np.nanpercentile(sub, 2.5), onset_sub100_hi=np.nanpercentile(sub, 97.5),
                         early_pilot=zp[:, win].mean(), early_final=zf[:, win].mean(),
                         curve_max_abs_diff=np.abs(dp.mean(0) - df.mean(0)).max(),
                         curve_r=np.corrcoef(dp.mean(0), df.mean(0))[0, 1],
                         sig_bins_differ=int(((np.percentile(dp, 5, 0) > 0) != (np.percentile(df, 5, 0) > 0)).sum()),
                         n_bins=dp.shape[1]))
    R = pd.DataFrame(rows).sort_values(["level", "onset_final"])
    (HOME / "tables").mkdir(exist_ok=True)
    R.to_csv(HOME / "tables" / "iteration_check_n200.csv", index=False)
    print(R.round(3).to_string(index=False))
    (HOME / "provenance_007.json").write_text(json.dumps(dict(
        script="007_iteration_check.py", pilot=str(HOME / "active"), final=str(HOME / "active_final_n200"), N=N_MAIN,
        n_areas=len(R), n_subsets=N_SUB, date=time.strftime("%Y-%m-%d %H:%M")), indent=1))


if __name__ == "__main__":
    main()
