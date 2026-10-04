"""Per-unit PSTHs (good units, all 124 sessions) for the conditions compared by the ROC analyses.

Spike trains: roc_utils_new.process_nwb_tables (whisker artifact [-10, +5] ms corrected). Trials: active / passive_pre
/ passive_post contexts (sessions whose context is the string 'nan' have no passive blocks -> all trials active).
Conditions (5 ms bins; stimulus-aligned -0.2..0.4 s, lick-aligned -0.4..0.4 s):
  W_pre, W_post, A_pre, A_post   passive whisker / auditory stimuli (pre- / post-session blocks)
  W_hit, W_miss, A_hit, A_miss   active whisker / auditory trials by outcome
  CR, FA                         catch (no-stim) trials without / with a lick
  SPONT                          spontaneous licks (roc_utils_new.get_filtered_lick_times, as the ROC), lick-aligned
Per unit: firing rate (Hz) per bin and condition, number of events, baseline mean / SD of the pre-stimulus rate
([-0.2, -0.01] s over all active stimulus and catch trials).
Output: combined_results_ks4/_roc_stage_analysis/psth/<session_id>.npz
"""
import argparse
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru                     # noqa: E402

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
OUT = RES / "_roc_stage_analysis" / "psth"
BW = 0.005
STIM_BINS = np.round(np.arange(-0.2, 0.4 + 1e-9, BW), 4)
LICK_BINS = np.round(np.arange(-0.4, 0.4 + 1e-9, BW), 4)


def rate_psth(st, t0, bins):
    if len(t0) == 0:
        return np.full(len(bins) - 1, np.nan, np.float32)
    idx = np.searchsorted(st, t0[:, None] + bins[None, :])
    return (np.diff(idx, axis=1).sum(0) / len(t0) / BW).astype(np.float32)


def worker(args):
    sid, keys = args
    out = OUT / f"{sid}.npz"
    if out.exists():
        return sid, "exists"
    nwb = NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r").read()
    units, _ = ru.process_nwb_tables(nwb)
    units = units.reset_index(drop=True)
    units["cluster_id"] = units.cluster_id.astype(str)
    units = units.merge(keys, on=["electrode_group", "cluster_id"], how="inner")
    t = nwb.trials.to_dataframe().sort_values("start_time")
    ctx = t["context"].astype(str).str.strip().str.lower()
    if ctx.isin(["nan", "none", ""]).all():
        ctx = pd.Series("active", index=t.index)
    elif "passive" in set(ctx):                                      # split passive blocks into pre / post
        mid = t.start_time[ctx == "active"].median()
        ctx = np.where(ctx == "passive", np.where(t.start_time < mid, "passive_pre", "passive_post"), ctx)
        ctx = pd.Series(ctx, index=t.index)
    t["ctx"] = ctx
    act = t[t.ctx == "active"]
    sel = {
        "W_pre": t[(t.ctx == "passive_pre") & (t.whisker_stim == 1)], "W_post": t[(t.ctx == "passive_post") & (t.whisker_stim == 1)],
        "A_pre": t[(t.ctx == "passive_pre") & (t.auditory_stim == 1)], "A_post": t[(t.ctx == "passive_post") & (t.auditory_stim == 1)],
        "W_hit": act[(act.whisker_stim == 1) & (act.lick_flag == 1)], "W_miss": act[(act.whisker_stim == 1) & (act.lick_flag == 0)],
        "A_hit": act[(act.auditory_stim == 1) & (act.lick_flag == 1)], "A_miss": act[(act.auditory_stim == 1) & (act.lick_flag == 0)],
        "CR": act[(act.no_stim == 1) & (act.lick_flag == 0)], "FA": act[(act.no_stim == 1) & (act.lick_flag == 1)]}
    ev = {k: np.sort(v.start_time.to_numpy(float)) for k, v in sel.items()}
    try:
        ev["SPONT"] = np.sort(np.asarray(ru.get_filtered_lick_times(nwb), float))
    except Exception as e:                                          # noqa: BLE001
        print(sid, "spont licks failed:", e, flush=True)
        ev["SPONT"] = np.array([])
    base_t = np.sort(act.start_time.to_numpy(float))
    res = {k: [] for k in ev}
    bmu, bsd = [], []
    for s in units.spike_times:
        st = np.sort(np.asarray(s, float))
        for k, t0 in ev.items():
            res[k].append(rate_psth(st, t0, LICK_BINS if k == "SPONT" else STIM_BINS))
        b = (np.searchsorted(st, base_t - 0.01) - np.searchsorted(st, base_t - 0.2)) / 0.19
        bmu.append(b.mean()); bsd.append(b.std())
    np.savez_compressed(out, electrode_group=units.electrode_group.to_numpy(str), cluster_id=units.cluster_id.to_numpy(str),
                        base_mu=np.array(bmu, np.float32), base_sd=np.array(bsd, np.float32),
                        stim_bins=STIM_BINS, lick_bins=LICK_BINS, **{f"n_{k}": len(v) for k, v in ev.items()},
                        **{k: np.stack(v) if len(v) else np.zeros((0, 1), np.float32) for k, v in res.items()})
    return sid, f"{len(units)} units, " + ", ".join(f"{k}={len(v)}" for k, v in ev.items())


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    U = pd.read_parquet(RES / "_roc_stage_analysis" / "units.parquet")
    U = U[U.quality_label == "good"]
    jobs = [(sid, g[["electrode_group", "cluster_id"]].drop_duplicates()) for sid, g in U.groupby("session_id")]
    print(len(jobs), "sessions", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for sid, msg in pool.map(worker, jobs):
            print(sid, msg, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
