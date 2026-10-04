"""005 -- Final onsets at N = 200 (user 2026-10-04): 1000 iterations x 20 shuffles, all 18 area groups + top-40 areas.

Run after `ARRIVAL_OUT=<final folder> ARRIVAL_NSHUF=20 001_arrival_pseudopop.py --n-list 200 --n-iter 1000` (the final
folder links cache_all/ of the task-trial run, so no new cache). Same summaries and onset rule as 002 / 004; only the
N = 200 main figures are made (the N-dependence panels of the summary figure need every N, which stay at pilot sampling
in combined_results_ks4/_stimulus_arrival/).
Output (ARRIVAL_OUT): onset_bootstrap_N200.csv, onsets.csv, figures/arrival_main_N200_<level>.{png,pdf,svg}, provenance.
"""
import importlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
AR = importlib.import_module("_areas")
S = importlib.import_module("_style")
m1 = importlib.import_module("001_arrival_pseudopop")
m2 = importlib.import_module("002_arrival_summary")
m4 = importlib.import_module("004_main_figures")


def main():
    plt = S.setup()
    D = m2.load_raw()
    D = D[D.N == m4.N_MAIN]
    print("runs:", D.groupby(["level", "area"]).size().describe().round(1).to_dict())
    B, O, W = m2.summarise(D)
    O.to_csv(AR.OUT / "onsets.csv", index=False)
    OB = m4.onset_boot(D, np.random.default_rng(0))
    OB.to_csv(AR.OUT / "onset_bootstrap_N200.csv", index=False)
    print(OB.sort_values(["level", "onset_ms"]).round(1).to_string())
    col = AR.colors()
    m4.FIG = AR.OUT / "figures"
    for level in ("area_group", "area_acronym_custom"):
        if (OB.level == level).any():
            m4.main_figure(plt, D, B, OB, level, col)
    (AR.OUT / "provenance_005.json").write_text(json.dumps(dict(
        script="005_final_n200.py", n_iter=int(D.rep.nunique()), n_shuffles=m1.N_SHUF, N=m4.N_MAIN, epoch=AR.EPOCH,
        n_areas=int(D.groupby(["level", "area"]).ngroups)), indent=1))
    print("FINAL N200 DONE", AR.OUT)


if __name__ == "__main__":
    main()
