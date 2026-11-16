"""020 -- Single-neuron drift shift test (unit_spikes_analysis run_motion_shift_test_analysis, Harris 2021) for the final
joined AB155 day-0 NWB (019). The DREDge motion is in the joined clock (no gap), the NWB in physical time (part 2 shifted
by the recording gap), so spike and trial times of part 2 are shifted back to the joined clock before the test (as 015).
The previous result file (part-1-only sorting) is renamed *.part1only_20261106 first.
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 020_drift_test_final.py
"""
import pathlib
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
from ephys_utilities.helpers import data_utils                                   # noqa: E402
from single_neuron_shift_test import unit_fr_motion_shift_test_harris as mst   # noqa: E402

A = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi")
NWB = A / "NWB_ks4" / "AB155_20250208_175103.nwb"
RES = A / "combined_results_ks4"
OLD = RES / "AB155" / "whisker_0" / "single_neuron_motion_shift_test" / "AB155_whisker_0_motion_shift_test_results.csv"


def main():
    with h5py.File(NWB, "r") as f:
        tj, t2 = f["processing/behavior/BehavioralEpochs/recording_gap/timestamps"][()]
    gap = t2 - tj
    print(f"recording gap {tj:.3f}-{t2:.3f} s ({gap:.3f} s): part-2 times shifted back by it")
    trial_table, unit_table, _ = data_utils.combine_ephys_nwb([str(NWB)], day_to_analyze="all", max_workers=1)
    unit_table["spike_times"] = [np.where(s >= t2, s - gap, s) for s in unit_table["spike_times"]]
    for c in ("start_time", "stop_time"):
        trial_table[c] = np.where(trial_table[c] >= t2, trial_table[c] - gap, trial_table[c])
    print(f"{len(unit_table)} units, {len(trial_table)} trials, trial starts "
          f"{trial_table.start_time.min():.1f}-{trial_table.start_time.max():.1f} s (joined clock)")
    if OLD.exists():
        OLD.rename(OLD.with_name(OLD.name + ".part1only_20261106"))
        print(f"previous result moved aside: {OLD.name}.part1only_20261106")
    res = mst.run_motion_shift_test_analysis(unit_table, trial_table, output_path=str(RES),
                                             config=dict(data_root=str(A / "data"), n_workers=20))
    fail = (res.p_conservative < 0.01) & (res.r.abs() > 0.5)
    print(f"{len(res)} units tested, r filled {res.r.notna().mean():.3f}; joint fail per probe:")
    print(res.assign(fail=fail).groupby("imec_id").fail.agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
