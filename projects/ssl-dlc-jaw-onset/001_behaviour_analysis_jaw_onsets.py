"""Jaw-opening onsets with the user's behaviour_analysis pipeline, one dlc_jaw_onset_times.pkl per mouse in
combined_results_ks4/<mouse>/ (user 2026-11-05: "the jaw onsets files need to use my behaviour_analysis function which
saves a .pkl per mouse in combined_results_ks4").

Calls beh_plotting_functions.plot_behavior(nwb_list, output_folder, plots=['single_trial_dlc_jaw_onset'], info_path)
unchanged (canonical NAS checkout of behaviour_analysis): it builds the behaviour table and, since 2026-11-10, computes
the onsets per session with ephys_utilities dlc_helpers.jaw_onset_times (side camera with its own frame timestamps, jaw
in mm with the converter reference fix, video-excluded sessions skipped; parameters: time window -1..2 s, baseline
(-1, -0.1) s, threshold 5 x pooled baseline SD, pre-onset window 0.5 s, onset at 30% of the peak; whisker days 0-9),
then saves dlc_jaw_onset_times.pkl plus its jaw-onset plots in the mouse folder. Only paths differ from its __main__:
NWB_ks4 session files (no *_backup_*.nwb), output combined_results_ks4/<mouse>, info_path = the reference sheet file.
The NAS ephys_utilities checkout is put first on sys.path (the haas env has the GitHub copy, which can lag behind).
An existing pkl is kept as dlc_jaw_onset_times_backup_<mtime date>.pkl before it is replaced.

Run (haas, repo root): nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-dlc-jaw-onset/001_behaviour_analysis_jaw_onsets.py [--n-proc 12] [--mice AB164]
"""

import argparse
import datetime
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
from common import REFERENCE_SHEET, axel_bisi_root, import_external_repos, session_nwbs  # noqa: E402

OUT_ROOT = axel_bisi_root() / "combined_results_ks4"


def run_mouse(mouse_id):
    import_external_repos()
    if "pymc" not in sys.modules:  # imported at module level for unrelated models; not installed on haas
        import types
        sys.modules["pymc"] = types.ModuleType("pymc")
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))  # NAS checkout first (see docstring)
    import beh_plotting_functions as bpf
    assert hasattr(bpf.dlc_helpers, "jaw_onset_times"), bpf.dlc_helpers.__file__
    out = OUT_ROOT / mouse_id
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "dlc_jaw_onset_times.pkl"
    if pkl.exists():
        date = datetime.date.fromtimestamp(pkl.stat().st_mtime).isoformat()
        backup, k = out / f"dlc_jaw_onset_times_backup_{date}.pkl", 1
        while backup.exists():  # never replace an earlier backup of the same date
            k += 1
            backup = out / f"dlc_jaw_onset_times_backup_{date}_{k}.pkl"
        pkl.rename(backup)
    files = [str(f) for f in session_nwbs(mouse_id)]
    try:
        bpf.plot_behavior(nwb_list=files, output_folder=str(out), plots=["single_trial_dlc_jaw_onset"],
                          info_path=str(REFERENCE_SHEET))
    except Exception as err:
        return mouse_id, f"{type(err).__name__}: {err}", traceback.format_exc(limit=3)
    if not pkl.exists():
        return mouse_id, "no pkl written (no DLC data?)", ""
    df = pd.read_pickle(pkl)
    return mouse_id, f"ok {len(df)} rows, {df.session_id.nunique() if 'session_id' in df else '?'} sessions", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proc", type=int, default=12)
    parser.add_argument("--mice", nargs="*")
    args = parser.parse_args()
    ref = pd.read_excel(REFERENCE_SHEET)
    mice = sorted(ref[(ref.with_dual_view == 1) & (ref.exclude == 0)].mouse_id.unique())
    if args.mice:
        mice = [m for m in mice if m in args.mice]
    mice = [m for m in mice if session_nwbs(m)]
    print(len(mice), "mice with dual-view video and NWBs", flush=True)
    with Pool(args.n_proc) as pool:
        results = pool.map(run_mouse, mice, chunksize=1)
    log = pd.DataFrame(results, columns=["mouse_id", "status", "traceback"])
    stamp = datetime.date.today().isoformat()
    log.to_csv(axel_bisi_root() / "combined_results_ks4" / "ssl-dlc-jaw-onset" / f"behaviour_analysis_run_{stamp}.csv",
               index=False)
    print(log[["mouse_id", "status"]].to_string())


if __name__ == "__main__":
    main()
