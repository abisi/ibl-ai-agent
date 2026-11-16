"""Jaw onsets: behaviour_analysis find_dlc_jaw_onset_times on dlc_helpers (2026-11-10 change) vs the existing per-mouse
dlc_jaw_onset_times.pkl (NWB_reader version), trial by trial, on a few mice (user 2026-11-10).

The trial table is the one stored in each mouse's existing pkl (same trials, same start / response-window / lick
times); only the DLC loading and the onset code change: ephys_utilities dlc_helpers.jaw_onset_times (NAS
checkout), as called by beh_plotting_functions.plot_behavior since 2026-11-10. The existing pkls are not modified.
Mice: a typical session set (AB164, AB131), DLC with the jaw reference at the image origin (MH062, MH070, AB153), and
sessions with known frame / pulse pairing exceptions (MH013 end-aligned 0218 and piecewise 0219, AB116 0726).
Outputs: combined_results_ks4/ssl-dlc-jaw-onset/compare_dlc_helpers/ (trials.csv, summary.csv, compare.png)
Run (haas): nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 002_compare_dlc_helpers.py [--mice AB164 ...]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AB = Path("/mnt/lsens-analysis/Axel_Bisi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl-paper-figures"))
from common import import_external_repos  # noqa: E402  (behaviour_analysis + NWB_utils on sys.path, as 001)

import_external_repos()
sys.path.insert(0, str(AB / "behaviour_analysis"))  # canonical NAS checkout first
sys.path.insert(0, str(AB / "Github" / "ephys_utilities"))
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402

RES = AB / "combined_results_ks4"
NWB = AB / "NWB_ks4"
OUT = RES / "ssl-dlc-jaw-onset" / "compare_dlc_helpers"
MICE = ["AB164", "AB131", "MH062", "MH070", "AB153", "MH013", "AB116"]
PARAMS = dict(time_start=-1, time_stop=2, baseline_window=(-1, -0.1), thresh_factor=5, pooled_baseline_std=True,
              pre_onset_window=0.5, onset_peak_fraction=0.3)  # beh_plotting_functions.plot_behavior


def main(mice):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mouse in mice:
        old = pd.read_pickle(RES / mouse / "dlc_jaw_onset_times.pkl")
        sessions = sorted(old.session_id.unique())
        nwbs = [f for s in sessions for f in glob.glob(str(NWB / f"{s}.nwb"))]
        trials = old.drop(columns=[c for c in ("threshold", "freq", "jaw_dlc_pos", "jaw_dlc_vel", "jaw_dlc_onset",
                                               "piezo_lick_time") if c in old.columns])
        new = pd.concat([r for r in (dh.jaw_onset_times(f, trials[trials.session_id == Path(f).stem].reset_index(drop=True),
                                                         **PARAMS) for f in nwbs) if r is not None], ignore_index=True)
        key = ["session_id", "start_time"]
        m = old[key + ["trial_type", "lick_flag", "jaw_dlc_onset", "threshold"]].merge(
            new[key + ["jaw_dlc_onset", "threshold"]], on=key, how="left", suffixes=("_old", "_new"))
        m["mouse_id"] = mouse
        rows.append(m)
        print(mouse, len(sessions), "sessions", len(m), "trials", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t["diff_ms"] = 1000 * (t.jaw_dlc_onset_new - t.jaw_dlc_onset_old)
    t.to_csv(OUT / "trials.csv", index=False)
    s = t.groupby(["mouse_id", "session_id"]).agg(
        n=("diff_ms", "size"), both=("diff_ms", lambda x: x.notna().sum()),
        median_abs_diff_ms=("diff_ms", lambda x: x.abs().median()),
        pct_within_10ms=("diff_ms", lambda x: 100 * (x.abs() <= 10).mean()),
        median_diff_ms=("diff_ms", "median"), thr_ratio=("threshold_new", "median")).reset_index()
    s["thr_ratio"] = s.thr_ratio / t.groupby(["mouse_id", "session_id"]).threshold_old.median().values
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.round(2).to_string())
    fig, axs = plt.subplots(2, len(mice), figsize=(2.6 * len(mice), 5), squeeze=False)
    for c, mouse in enumerate(mice):
        d = t[t.mouse_id == mouse]
        ax = axs[0, c]
        ax.scatter(d.jaw_dlc_onset_old, d.jaw_dlc_onset_new, s=2, alpha=0.4, edgecolors="none")
        ax.plot([-0.5, 2.2], [-0.5, 2.2], color="0.5", lw=0.5, ls=":")
        ax.set_title(mouse, fontsize=8); ax.set_xlabel("NWB_reader onset (s)", fontsize=7)
        ax.set_ylabel("dlc_helpers onset (s)" if c == 0 else "", fontsize=7)
        ax = axs[1, c]
        ax.hist(d.diff_ms.clip(-300, 300).dropna(), bins=np.arange(-300, 305, 10))
        ax.set_xlabel("new - old (ms, clipped)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "compare.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mice", nargs="+", default=MICE)
    main(ap.parse_args().mice)
