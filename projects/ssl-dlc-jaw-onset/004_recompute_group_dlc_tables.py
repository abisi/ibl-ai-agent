"""Recompute the cached group DLC tables of behaviour_analysis with ephys_utilities dlc_helpers (user 2026-11-10).

beh_plotting_functions.plot_group_behavior caches two tables and reloads them whenever they exist:
  data/processed_data/dlc_peth/dlc_peth_global.pkl     per-trial DLC PETHs (-2..5 s from trial start, 5 ms bins) of
      every whisker-day session: 'mean_movement_dlc_peths', '_by_perf', '_by_day', '_context', '_inflection' plots
  data/processed_data/dlc_trials/dlc_trial_global.pkl  per-trial movement / lick metrics: 'dlc_mvt_across_trials',
      'dlc_licks_across_trials'
Both were built in 2025-04/05 with NWB_reader (old NWBs, pixel units, nominal-rate frame mapping). This script runs the
same steps as plot_group_behavior (behaviour table from the session NWBs, reference-sheet merge, whisker-day reindex,
its mouse filters: PETH table without MH001-MH029, trial table AB mice only, as in its code) and calls its builders
with use_dlc_helpers=True. The existing pkls are renamed *_legacy_<mtime date>.pkl first (kept).
Run (haas, repo root): nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-dlc-jaw-onset/004_recompute_group_dlc_tables.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
from common import REFERENCE_SHEET, axel_bisi_root, import_external_repos, session_nwbs  # noqa: E402

PROC = axel_bisi_root() / "data" / "processed_data"


def keep_legacy(path):
    if path.exists():
        date = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
        if date == datetime.date.today().isoformat():  # this script's own earlier output today: replaced
            return
        dest = path.with_name(f"{path.stem}_legacy_{date}.pkl")
        assert not dest.exists(), dest
        path.rename(dest)
        print("kept", dest)


def main():
    import_external_repos()
    if "pymc" not in sys.modules:
        import types
        sys.modules["pymc"] = types.ModuleType("pymc")
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))  # NAS checkout first
    sys.path.insert(0, str(axel_bisi_root() / "behaviour_analysis"))
    import behavior_analysis_utils as bhv_utils

    info = pd.read_excel(REFERENCE_SHEET).rename(columns={"mouse_name": "mouse_id"})
    info["reward_group"] = info["reward_group"].map({"R+": 1, "R-": 0, "R+proba": 2})
    info = info[info["exclude"] == 0]
    nwb_list = [str(f) for f in session_nwbs() if f.stem[:5] in set(info.mouse_id)]
    bhv = bhv_utils.build_combined_behavior_table(nwb_list)
    cols = [c for c in ("mouse_id", "reward_group", "learning_category", "gradual", "expert_start") if c in info]
    bhv = bhv.merge(info[cols], on="mouse_id", how="left")
    bhv = bhv_utils.reindex_whisker_days(bhv)
    print("mice:", bhv.mouse_id.nunique(), flush=True)

    # PETH table (plot_group_behavior: dual-view mice, MH < 30 excluded, whisker days)
    dual = info[info.with_dual_view == 1].mouse_id.unique().tolist()
    include = [m for m in dual if not (m.startswith("MH") and int(m[2:]) < 30)]
    peth_in = bhv[bhv.mouse_id.isin(include) & bhv.behavior.isin(["whisker"])]
    params = {"recompute_flag": True, "bin_size": 0.005, "align": "trial_start", "time_start": -2, "time_stop": 5}
    keep_legacy(PROC / "dlc_peth" / "dlc_peth_global.pkl")
    peth = bhv_utils.build_combined_dlc_peth_table(peth_in, saving_path=str(PROC / "dlc_peth"), params=params,
                                                   use_dlc_helpers=True)
    print("PETH table:", peth.shape, peth.mouse_id.nunique(), "mice", flush=True)

    # trial metrics table (plot_group_behavior: whisker days, AB mice only)
    trial_in = bhv[bhv.behavior.isin(["whisker"]) & bhv.mouse_id.str.startswith("AB")]
    keep_legacy(PROC / "dlc_trials" / "dlc_trial_global.pkl")
    trials = bhv_utils.build_combined_dlc_trial_metrics_table(trial_in, saving_path=str(PROC / "dlc_trials"),
                                                              params={}, use_dlc_helpers=True)
    print("trial table:", trials.shape, trials.mouse_id.nunique(), "mice", flush=True)


if __name__ == "__main__":
    main()
