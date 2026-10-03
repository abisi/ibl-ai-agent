"""Run the single-unit motion-drift shift test (latest version: unit_spikes_analysis/single_neuron_shift_test/
unit_fr_motion_shift_test_harris.py, Harris 2021 exhaustive shift sweep, 2026-09-20) for sessions that have ephys but no
result file, so that quality_label (classify_units_quality) gets the joint drift check for every session.

A session is missing if combined_results_ks4/<mouse>/<behaviour>_<day>/single_neuron_motion_shift_test/
<mouse>_<behaviour>_<day>_motion_shift_test_results.csv does not exist. Units and trials are loaded from the NWB files
(NWB_ks4) with data_utils.combine_ephys_nwb, as for the unit-table build; DREDge motion from
Axel_Bisi/data/<mouse>/<session>/Ephys/catgt_*/..._imec*/dredge(_fast)/motion/motion.
usage: python run_missing_drift_tests.py [--mice MH020 ...] [--dry-run] [--n-workers 20]
"""
import argparse
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
from ephys_utilities.helpers import data_utils                                   # noqa: E402
from single_neuron_shift_test import unit_fr_motion_shift_test_harris as mst   # noqa: E402

NWB_DIR = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
DATA = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/data")
SHEET = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")


def result_path(mouse, session_day):
    return RES / mouse / session_day / "single_neuron_motion_shift_test" / f"{mouse}_{session_day}_motion_shift_test_results.csv"


def main(a):
    X = pd.read_excel(SHEET)
    mice = a.mice or sorted(X[(X.exclude == 0) & (X.exclude_ephys == 0) & X.reward_group.isin(["R+", "R-"])].mouse_id)
    nwbs = sorted(str(NWB_DIR / n) for n in os.listdir(NWB_DIR) if n.split("_")[0] in mice)
    trial_table, unit_table, _ = data_utils.combine_ephys_nwb(nwbs, day_to_analyze="all", max_workers=a.n_workers)
    unit_table["session_day"] = unit_table["behaviour"].astype(str) + "_" + unit_table["day"].astype(int).astype(str)
    keys = unit_table[["mouse_id", "session_id", "session_day"]].drop_duplicates()
    keys["done"] = [result_path(r.mouse_id, r.session_day).exists() for r in keys.itertuples()]
    todo = keys[~keys.done]
    print(f"{len(keys)} ephys sessions, {len(todo)} without drift-test results:\n{todo.to_string(index=False)}", flush=True)
    if a.dry_run or todo.empty:
        return
    ut = unit_table[unit_table.session_id.isin(todo.session_id)]
    tt = trial_table[trial_table.session_id.isin(todo.session_id)]
    res = mst.run_motion_shift_test_analysis(ut, tt, output_path=str(RES),
                                             config=dict(data_root=str(DATA), n_workers=a.n_workers))
    done = [result_path(r.mouse_id, r.session_day).exists() for r in todo.itertuples()]
    print(f"finished: {sum(done)}/{len(todo)} sessions now have results; {len(res)} unit rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mice", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-workers", type=int, default=20)
    main(ap.parse_args())
