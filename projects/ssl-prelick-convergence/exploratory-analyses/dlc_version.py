"""DLC version switch of the jaw / kinematics chain (2026-11-12; user: "ALL analyses should use what is in NWB_ks4").

PRELICK_DLC=v1 (default): the original chain (113 load_series on the raw pynwb BehavioralTimeSeries, likelihood >= 0.5 masks,
  px -> mm with scripts/ssl_dlc_calibration.py; 118 own 7 x SD first-crossing jaw onsets). Outputs unchanged.
PRELICK_DLC=v2: every DLC series from the NWB_ks4 files through ephys_utilities dlc_helpers (NAS checkout first on PYTHONPATH):
  load_dlc + clean_series (default cleaning), each series on its own camera timestamps, video-excluded sessions skipped;
  jaw onsets with the dlc_helpers rule (JAW_ONSET_PARAMS). Outputs go to new names only:
    113 / 116 / 126 output folders suffixed DSUF ("_v2"): dlc_passive_pre_v2, prelick_kinematics_v2, jaw_control_v2
    118 alignment tags JAW_SFX ("_jaw2" = [-100, 0) ms before jaw onset, "_jawc2" = [-50, +50) ms around it); downstream scripts
    are run with PRELICK_ALIGN=jaw2 / jawc2 (051 TAG "_sl_jaw2", roots across_days/sl_stable_jaw2, within_day/sl_stable_jaw2)
    124 / 125 / 115 output files or folders suffixed DSUF.
"""
import os

DLC = os.environ.get("PRELICK_DLC", "v1")
assert DLC in ("v1", "v2"), f"PRELICK_DLC must be v1 or v2, got {DLC}"
DSUF = "" if DLC == "v1" else f"_{DLC}"
JAW_ALIGN = {"v1": {"pre": "jaw", "centered": "jawc"}, "v2": {"pre": "jaw2", "centered": "jawc2"}}[DLC]   # PRELICK_ALIGN values
JAW_SFX = {k: f"_{v}" for k, v in JAW_ALIGN.items()}


def dlc_helpers():
    """ephys_utilities dlc_helpers and data_utils from the NAS checkout (must be first on PYTHONPATH; the haas ~/code copy lags)"""
    from ephys_utilities.helpers import data_utils as du
    from ephys_utilities.helpers import dlc_helpers as dh
    assert "peak_prominence" in dh.JAW_ONSET_PARAMS and hasattr(dh, "pupil_trial_ok"), \
        f"dlc_helpers without the 2026-11-12 jaw-onset rule / pupil_trial_ok: {dh.__file__} (set PYTHONPATH to the NAS ephys_utilities)"
    return dh, du
