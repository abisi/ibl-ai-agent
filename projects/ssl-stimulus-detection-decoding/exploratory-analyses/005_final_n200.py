"""005 -- Final onsets at N = 200: 500 iterations x 20 linear shifts (as the arrival project), all area groups + top-40
areas, per target and epoch (DETECT_TARGET, DETECT_EPOCH).

Run after `DETECT_OUT=<final folder> 001_detection_pseudopop.py --n-list 200` --n-iter 500 with DETECT_NSHIFT=20; the
epoch's cache is shared, <HOME>/cache_<epoch>). Only iterations 0..499 are used. Same summaries and onset rule as 002 / 004; only the
N = 200 main figures are made (the N-dependence panels of the summary figure need every N, which stay at pilot sampling
in combined_results_ks4/ssl-stimulus-detection-decoding/<target>_<epoch>/).
Output (DETECT_OUT): onset_bootstrap_N200.csv, onsets.csv, summary_bins.parquet, window_accuracy.csv,
detection_figures_caption.md, figures/detection_main_N200_<level>.{png,pdf,svg}, provenance.
"""
import importlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
AR = importlib.import_module("_areas")
S = importlib.import_module("_style")
m1 = importlib.import_module("001_detection_pseudopop")
m2 = importlib.import_module("002_detection_summary")
m4 = importlib.import_module("004_main_figures")
N_FINAL = 500


def main():
    plt = S.setup()
    D = m2.load_raw()
    D = D[(D.N == m4.N_MAIN) & (D.rep < N_FINAL)]
    print("runs:", D.groupby(["level", "area"]).size().describe().round(1).to_dict())
    B, O, W = m2.summarise(D)
    O.to_csv(AR.OUT / "onsets.csv", index=False)
    B.to_parquet(AR.OUT / "summary_bins.parquet")
    W.to_csv(AR.OUT / "window_accuracy.csv", index=False)
    OB = m4.onset_boot(D, np.random.default_rng(0))
    OB.to_csv(AR.OUT / "onset_bootstrap_N200.csv", index=False)
    print(OB.sort_values(["level", "onset_ms"]).round(1).to_string())
    col = AR.colors()
    m4.FIG = AR.OUT / "figures"
    for level in ("area_group", "area_acronym_custom"):
        if (OB.level == level).any():
            m4.main_figure(plt, D, B, OB, level, col)
    m4.captions(OB, int(D.rep.nunique()), int(D.n_shifts.iloc[0]))
    (AR.OUT / "provenance_005.json").write_text(json.dumps(dict(
        script="005_final_n200.py", n_iter=int(D.rep.nunique()), n_shifts=int(D.n_shifts.iloc[0]), N=m4.N_MAIN, target=AR.TARGET, epoch=AR.EPOCH,
        n_areas=int(D.groupby(["level", "area"]).ngroups)), indent=1))
    print("FINAL N200 DONE", AR.OUT)


if __name__ == "__main__":
    main()
