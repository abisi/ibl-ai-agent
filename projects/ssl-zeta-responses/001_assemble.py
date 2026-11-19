"""001 -- pool the per-session ZETA / latenZy tables of the unit_spikes_analysis ZETA pipeline (all sessions) and join
unit metadata and the matching ROC result.

Input:  combined_results_ks4/<mouse>/whisker_<day>/zeta/<mouse>_zeta_results.csv (zeta_analysis, v2 2026-11-19)
        combined_results_ks4/_roc_stage_analysis/{units.parquet, roc_long.parquet} (area_group, cohort, day, quality;
        ROC sel / sig per analysis type; duplicate unit rows dropped; 045's sign flip of the choice
        types undone, so roc_sel > 0 = condition 2 higher, the ZETA convention)
Units:  good + mua (quality_label), as in the ROC analyses; passive analysis types via data_utils.keep_passive_units.
Output: combined_results_ks4/ssl-zeta-responses/zeta_long.parquet (one row per unit x analysis type) + provenance.json
Run (haas): PYTHONPATH=~/code/ephys_utilities ~/code/unit_spikes_analysis/.venv/bin/python 001_assemble.py
"""
import glob
import json
import pathlib
import time

import pandas as pd

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = RES / "ssl-zeta-responses"
ROC = RES / "_roc_stage_analysis"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
FLIP = {"choice", "whisker_choice", "auditory_choice", "baseline_choice", "baseline_whisker_choice",
        "baseline_auditory_choice"}   # 045_roc_stage_table.FLIP
COLS = KEYS + ["neuron_id", "analysis_type", "n_events_1", "n_events_2", "zeta_p", "zeta_p_full", "zeta_p_post", "zeta_z",
               "direction", "ifr_peak_s", "pos_peak_s", "pos_peak_z", "neg_peak_s", "neg_peak_z", "neg_peak_drop",
               "neg_peak", "latency_s", "latenzy_p", "significant", "mean_rate_1", "mean_rate_2"]


def main():
    from ephys_utilities.helpers import data_utils
    OUT.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(RES / "*" / "whisker_*" / "zeta" / "*_zeta_results.csv")))
    Z = pd.concat([pd.read_csv(f, usecols=lambda c: c in COLS, low_memory=False).assign(src=str(pathlib.Path(f).parent))
                   for f in files], ignore_index=True)
    Z["cluster_id"] = Z.cluster_id.astype(str)
    U = pd.read_parquet(ROC / "units.parquet", columns=KEYS + ["area_group", "area_acronym_custom", "cohort", "day",
                                                               "stage", "quality_label"])
    U["cluster_id"] = U.cluster_id.astype(str)
    Z = Z.merge(U.drop_duplicates(KEYS), on=KEYS, how="inner")
    Z = Z[Z.quality_label.isin(["good", "mua"]) & Z.zeta_p.notna()]
    pas = Z.analysis_type.str.contains("passive|pre_vs_post")
    Z = pd.concat([Z[~pas], data_utils.keep_passive_units(Z[pas])], ignore_index=True)
    roc = pd.read_parquet(ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sel", "sig"])
    roc["cluster_id"] = roc.cluster_id.astype(str)
    roc = roc.drop_duplicates(KEYS + ["analysis_type"]).rename(columns={"sel": "roc_sel", "sig": "roc_sig"})
    # 045 flips the choice types (positive = lick / hit); undo it so that positive = condition 2 higher, as in ZETA
    roc.loc[roc.analysis_type.isin(FLIP), "roc_sel"] *= -1
    Z = Z.merge(roc, on=KEYS + ["analysis_type"], how="left")
    Z["significant"] = Z.significant.astype(bool)
    Z.to_parquet(OUT / "zeta_long.parquet")
    json.dump(dict(script="001_assemble.py", n_files=len(files), n_rows=len(Z), n_sessions=int(Z.session_id.nunique()),
                   n_mice=int(Z.mouse_id.nunique()), units="good+mua, keep_passive_units for passive types",
                   date=time.strftime("%Y-%m-%d %H:%M")), open(OUT / "provenance.json", "w"), indent=1)
    print(len(files), "files", len(Z), "rows", Z.session_id.nunique(), "sessions", Z.roc_sig.notna().mean(), "with ROC")


if __name__ == "__main__":
    main()
