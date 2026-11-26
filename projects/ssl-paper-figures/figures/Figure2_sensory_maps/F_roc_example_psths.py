"""Figure 2F (right column): PSTHs of up-modulated, down-modulated and non-modulated units in one example area group per
row (user layout 2026-11-18): whisker trials -> Ss-whisker (Somatosensory-whisker), auditory trials -> Auditory,
spontaneous licks -> Motor areas.

Units: as F_roc_area_breakdown (day 0, stable set of skill ssl-valid-data, cohorts pooled); class from roc_long
(whisker_active / auditory_active / spontaneous_licks): up = sig & pos, down = sig & neg, non = not sig.
Events: whisker / auditory = active trials as in ssl-sensory-spatial-maps 001 (perf != 6, warm-up cut, rule A1 tail
trim, all trials lick or not); whisker spikes artefact-corrected (roc_utils_new.correct_neuron_spike_train, seeded per
session). Spontaneous licks = roc_utils_new.get_filtered_lick_times (piezo licks > 1 s from any stimulus), the ROC events.
PSTH: stimulus -100..+200 ms, 1-ms bins, Gaussian sigma 3 ms; licks -500..+500 ms, 1-ms bins, sigma 15 ms.
Delta rate = PSTH minus the unit's pre-event rate (stimulus: -1..-0.015 s, pre-trial baseline of ssl-figure-style rule
2; licks: -1..-0.5 s, before the ROC pre window -0.4..-0.2 s). Mean over units of a class within each session, then
mean +- s.e.m. over sessions (sessions with >= 1 unit of the class).
Output: manuscript Figures/Figure2_sensory_maps_explore_2026-11-17/F_roc/F_<row>_example_psth.* + _data.csv
Run (haas): PYTHONPATH=<NAS ephys_utilities>:~/code/NWB_reader ~/code/unit_spikes_analysis/.venv/bin/python
            F_roc_example_psths.py [--n-proc 40] [--plot-only]
"""
import argparse
import multiprocessing as mp
import os
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(Path.home() / "code" / "unit_spikes_analysis"))
import paperstyle as ps  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402
import F_roc_area_breakdown as fb  # noqa: E402

NWB = Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
LAT = Path("/home/bisi/code/ibl-ai-agent/projects/ssl-sensory-spatial-maps/exploratory-analyses/001_unit_latency.py")
OUT = fb.OUT
CACHE = fb.ROOT / "ssl-paper-figures" / "Figure2" / "F_example_psths.parquet"
EXAMPLE = {"whisker": "Somatosensory-whisker", "auditory": "Auditory areas", "lick": "Motor areas"}
LABEL = {"Somatosensory-whisker": "Ss-whisker", "Auditory areas": "Auditory", "Motor areas": "Motor areas"}
WIN = {"whisker": (-0.1, 0.2, 3.0, (-1.0, -0.015)), "auditory": (-0.1, 0.2, 3.0, (-1.0, -0.015)),
       "lick": (-0.5, 0.5, 15.0, (-1.0, -0.5))}
CLASSES = ["up", "down", "non"]


def units_table():
    d = fb.load()
    rows = []
    for row, at in fb.ROWS.items():
        x = d[(d.analysis_type == at) & (d.area_group == EXAMPLE[row])].copy()
        x["cls"] = np.where(x.sig & x.pos, "up", np.where(x.sig & x.neg, "down", "non"))
        x["row"] = row
        rows.append(x[fb.KEYS + ["row", "cls"]])
    return pd.concat(rows, ignore_index=True)


def _lat():
    import importlib.util
    s = importlib.util.spec_from_file_location("lat", LAT)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def run_session(args):
    sid, U = args
    from pynwb import NWBHDF5IO
    from scipy.ndimage import gaussian_filter1d
    from roc_analysis import roc_utils_new as ru
    lat = _lat()
    with NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
        trials = nwb.trials.to_dataframe()
        licks = np.asarray(ru.get_filtered_lick_times(nwb))
    units["cluster_id"] = units["cluster_id"].astype(str)
    onsets = trials.loc[trials["whisker_stim"] == 1, "start_time"].to_numpy()
    t = lat.stim_trials(trials)
    ev = {"whisker": t.start_time[t.trial_type == "whisker_trial"].to_numpy(),
          "auditory": t.start_time[t.trial_type == "auditory_trial"].to_numpy(), "lick": licks}
    sp = {(u.electrode_group, u.cluster_id): np.asarray(u.spike_times) for u in units.itertuples()}
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    out = []
    for row in EXAMPLE:
        s0 = ev[row]
        if len(s0) < 3:
            continue
        t0, t1, sig, base = WIN[row]
        edges = np.round(np.arange(t0, t1 + 5e-4, 1e-3), 6)
        for r in U[U.row == row].itertuples():
            st = sp.get((r.electrode_group, r.cluster_id))
            if st is None:
                continue
            if row == "whisker":
                st = ru.correct_neuron_spike_train(st, onsets, rng)
            b = ((np.searchsorted(st, s0 + base[1]) - np.searchsorted(st, s0 + base[0])) / (base[1] - base[0])).mean()
            cnt = np.zeros(len(edges) - 1)
            for s in s0:
                a, c = np.searchsorted(st, [s + t0, s + t1])
                cnt += np.histogram(st[a:c] - s, edges)[0]
            psth = gaussian_filter1d(cnt / len(s0) / 1e-3, sig) - b
            out.append(dict(mouse_id=r.mouse_id, session_id=sid, electrode_group=r.electrode_group,
                            cluster_id=r.cluster_id, row=row, cls=r.cls, n_events=len(s0), base_hz=b,
                            psth=psth.astype(np.float32)))
    return pd.DataFrame(out)


def compute(n_proc):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    U = units_table()
    jobs = [(s, g) for s, g in U.groupby("session_id")]
    res = []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for k, d in enumerate(pool.imap_unordered(run_session, jobs)):
            res.append(d)
            print(f"{k + 1}/{len(jobs)}", flush=True)
    P = pd.concat(res, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(CACHE.with_suffix(".tmp"))
    os.replace(CACHE.with_suffix(".tmp"), CACHE)
    return P


def plot(P):
    cols = au.get_custom_area_groups_colors()
    ps.set_nature_rc()
    W, H = 48.0, 28.0
    rows_csv = []
    for row, g in EXAMPLE.items():
        t0, t1, _, _ = WIN[row]
        x = P[P.row == row]
        n = len(x.psth.iloc[0])
        tt = (t0 + 5e-4 + 1e-3 * np.arange(n)) * (1000 if row != "lick" else 1)
        k = cols[g]
        style = {"up": dict(color=k), "down": dict(color=ps.light(k, 0.5)), "non": dict(color="0.6")}
        fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
        ax = fig.add_axes([8 / W, 7 / H, 22 / W, 17 / H])
        for c in CLASSES:
            y = x[x.cls == c]
            if not len(y):
                continue
            m = (y.groupby("session_id").psth.apply(lambda s: np.mean(np.stack(s.to_numpy()), 0)))
            M = np.stack(m.to_numpy())
            mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
            ax.fill_between(tt, mu - se, mu + se, color=ps.light(style[c]["color"], 0.6), lw=0, zorder=1)
            ax.plot(tt, mu, lw=0.6, zorder=2, **style[c])
            rows_csv.append(pd.DataFrame(dict(row=row, area_group=g, cls=c, t=tt, mean=mu, sem=se,
                                              n_sessions=len(M), n_units=len(y))))
            ax.text(1.04, {"up": 0.95, "non": 0.80, "down": 0.65}[c],
                    f"{ {'up': 'Up', 'down': 'Down', 'non': 'Non-mod.'}[c]} ({len(y):,})", transform=ax.transAxes,
                    fontsize=5, ha="left", va="top", color=style[c]["color"])
        ax.axvline(0, color="k", lw=0.4, ls=(0, (2, 1.5)), zorder=0)
        ax.axhline(0, color="0.3", lw=0.3, zorder=0)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_xlabel("Time from lick (s)" if row == "lick" else
                      f"Time from {row} stim. (ms)", fontsize=5.5, labelpad=1)
        ax.set_ylabel("Δ rate (spks/s)", fontsize=5.5, labelpad=1)
        ax.set_title(LABEL[g], fontsize=6, pad=2, color=k)
        ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_linewidth(0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ps.save_figure(fig, OUT, f"F_{row}_example_psth")
        plt.close(fig)
    pd.concat(rows_csv).to_csv(OUT / "F_example_psth_data.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    P = pd.read_parquet(CACHE) if a.plot_only else compute(a.n_proc)
    plot(P)
    print("done", P.groupby(["row", "cls"]).size().to_dict())
