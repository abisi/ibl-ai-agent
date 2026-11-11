"""Spontaneous-lick detection for sessions missing it (user 2026-11-09: AB105, AB155, MH001, MH019 whisker day 0, needed
by the behaviour GLM's spontaneous_lick_rate predictor). Calls behaviour_analysis spontaneous_licks_utils.
process_spontaneous_licks unchanged, with the parameters of beh_plotting_functions ('spontaneous_lick_detection'),
writing to combined_results/<mouse>/whisker_<day>/spontaneous_licks/ like the existing files.

Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/spont_licks_run.py SESSION [SESSION ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NWB_DIR, axel_bisi_root, import_external_repos  # noqa: E402

PARAMS = {"bout_gap_s": 2.5, "bout_min_licks": 5, "bout_max_window_s": 1.5, "min_isi_debounce": 0.02, "rate_bin_s": 300,
          "raster_row_duration_s": 600, "peth_pre_onset_offset": 0.1, "peth_post_window": 2.0, "peth_bin_size": 0.02,
          "trial_exclusion_window_s": 1.0}  # as in beh_plotting_functions, 'spontaneous_lick_detection'


def main(sessions):
    import_external_repos()
    import matplotlib
    matplotlib.use("Agg")
    import spontaneous_licks_utils
    for s in sessions:
        nwb = NWB_DIR / f"{s}.nwb"
        out = axel_bisi_root() / "combined_results" / s.split("_")[0]
        spontaneous_licks_utils.process_spontaneous_licks(nwb_path=str(nwb), params=PARAMS, output_dir=str(out))


if __name__ == "__main__":
    main(sys.argv[1:])
