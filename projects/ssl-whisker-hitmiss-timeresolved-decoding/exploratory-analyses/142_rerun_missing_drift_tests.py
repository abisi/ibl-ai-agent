"""142 -- Drift-shift test (unit_fr_motion_shift_test_harris, as run_missing_drift_tests.py) for the UNITS that have no
result in the existing per-session files (user 2026-10-05: "run drift unit test where it is missing"). 137 found 10169
units without a drift result (6836 somatic, in 114 sessions): absent from the session's result CSV or present with NaN
r / p. Only those units are tested; results go to a SEPARATE root so the existing result files are never overwritten:
combined_results_ks4/_drift_rerun_20261005/<mouse>/<behaviour>_<day>/single_neuron_motion_shift_test/...
137_stable_units.py merges this root after the original one (original results win when both exist).
usage (haas, repo root): python .../142_rerun_missing_drift_tests.py [--n-workers 20] [--dry-run]
"""
import argparse
import os
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
sys.path.insert(0, str(pathlib.Path.home() / "code" / "ephys_utilities"))
from ephys_utilities.helpers import data_utils                                   # noqa: E402
from single_neuron_shift_test import unit_fr_motion_shift_test_harris as mst   # noqa: E402

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = RES / "_drift_rerun_20261005"
NWB_DIR = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
DATA = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/data")
STABLE = RES / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet"


def main(a):
    t0 = time.time()
    S = pd.read_parquet(STABLE)
    miss = S[~S.has_drift_test & (S.bc_label != "non-soma")][["session_id", "electrode_group", "nwb_cluster_id"]]
    sessions = sorted(miss.session_id.unique())
    print(f"{len(miss)} somatic units without drift results in {len(sessions)} sessions", flush=True)
    if a.dry_run:
        return
    nwbs = sorted(str(NWB_DIR / f"{s}.nwb") for s in sessions if (NWB_DIR / f"{s}.nwb").exists())
    trial_table, unit_table, _ = data_utils.combine_ephys_nwb(nwbs, day_to_analyze="all", max_workers=a.n_workers)
    unit_table["cluster_id_int"] = unit_table.cluster_id.astype(int)
    unit_table["electrode_group"] = unit_table.electrode_group.astype(str)
    key = miss.rename(columns={"nwb_cluster_id": "cluster_id_int"})
    ut = unit_table.merge(key, on=["session_id", "electrode_group", "cluster_id_int"], how="inner").drop(columns="cluster_id_int")
    tt = trial_table[trial_table.session_id.isin(ut.session_id.unique())]
    print(f"matched {len(ut)} of {len(miss)} units in the NWB unit tables; running the test", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = mst.run_motion_shift_test_analysis(ut, tt, output_path=str(OUT), config=dict(data_root=str(DATA), n_workers=a.n_workers))
    print(f"finished: {len(res)} unit rows in {round((time.time() - t0) / 60, 1)} min -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
