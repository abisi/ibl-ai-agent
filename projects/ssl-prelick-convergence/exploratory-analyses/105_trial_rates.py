"""105 -- Per-trial rate matrices for the convergence / divergence analyses of 106 (user 2026-10-11: go / no-go remapping, decoding
transfer over time, subspace overlap). Compute only.
Per session (051 active selection; tested units of 081, stable run), events:
  W   all whisker trials (hits and misses; is_hit flag, RT), AH auditory hits, CR correct rejections (no-stimulus trials, no lick),
      FA false alarms (no-stimulus trials with a lick; added 2026-10-11)
  WH, AH, SL for the lick-aligned data (SL: 051 spontaneous licks, bout onsets outside trial windows within the analysed epoch)
Rates minus the event's own baseline (stimulus-aligned / WH / AH: mean rate in [-1, -0.015] s before trial start; SL: [-1, -0.5] s before
the lick, as 051), in spikes/s:
  stimwin  [50, 150) ms after stimulus onset, truncated at the trial's first lick (count / effective duration; NaN if < 20 ms left)
  stimbins 50-ms windows, centres 25 .. 175 ms (25-ms step) after stimulus onset; NaN where the window extends past the first lick
  lickbins 50-ms windows, centres -400 .. +100 ms (25-ms step) around the first lick (WH, AH) or the spontaneous lick (SL)
Output: across_days/<ref>/trial_rates/<session_id>.npz (unit keys, event labels, matrices units x events [x bins]) + sessions.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../105_trial_rates.py [--n-proc 25]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m81 = importlib.import_module("081_single_unit_classes")
OUT = m51.OUTROOT / "trial_rates"
W50 = 0.050
STIM_C = np.round(np.arange(0.025, 0.1751, 0.025), 3)
LICK_C = np.round(np.arange(-0.400, 0.1001, 0.025), 3)
BASE = (-1.0, -0.015)


def counts(s, ev, lo, hi):
    return np.searchsorted(s, ev + hi) - np.searchsorted(s, ev + lo)


def session(args):
    sid, units = args
    from pynwb import NWBHDF5IO
    try:
        with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
            nwb = io.read()
            U, _ = m51.ru.process_nwb_tables(nwb)
            raw = nwb.trials.to_dataframe()
            t, log = m51.active_trials(raw, sid)
            sl = m51.spontaneous_licks(nwb, raw, log["epoch"])
    except Exception as e:  # noqa: BLE001
        return dict(session_id=sid, error=repr(e)[:200])
    rt = (t.lick_time - t.response_window_start_time).to_numpy(); lick = (t.lick_flag == 1).to_numpy() & np.isfinite(rt)
    tt = t.trial_type.to_numpy(); st = t.start_time.to_numpy()
    iw = np.where(tt == "whisker_trial")[0]; ia = np.where((tt == "auditory_trial") & lick)[0]; ic = np.where((tt == "no_stim_trial") & ~lick)[0]
    ifa = np.where((tt == "no_stim_trial") & lick)[0]                                   # false alarms (2026-10-11, for CCGP 108)
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str)).merge(units, on=["electrode_group", "cluster_id"], how="inner")
    if len(U) < 5 or min(len(iw), len(ia), len(ic)) < 5:
        return dict(session_id=sid, error="too few units / events")
    spk = [np.sort(np.asarray(s)) for s in U.spike_times]
    stim_ev = np.r_[iw, ia, ic, ifa]; stim_lab = np.r_[np.full(len(iw), "W"), np.full(len(ia), "AH"), np.full(len(ic), "CR"), np.full(len(ifa), "FA")]
    s0 = st[stim_ev]; rts = np.where(lick[stim_ev], rt[stim_ev], np.inf)
    is_hit = lick[stim_ev] & (stim_lab == "W")
    wh = iw[lick[iw]]; fl_wh, fl_ah = st[wh] + rt[wh], st[ia] + rt[ia]
    lick_ev = np.r_[fl_wh, fl_ah, sl]; lick_lab = np.r_[np.full(len(wh), "WH"), np.full(len(ia), "AH"), np.full(len(sl), "SL")]
    base_lick = np.r_[st[wh], st[ia], np.full(len(sl), np.nan)]                         # trial start for WH / AH
    nu = len(spk)
    stimwin = np.full((nu, len(s0)), np.nan, np.float32); stimbins = np.full((nu, len(s0), len(STIM_C)), np.nan, np.float32)
    lickbins = np.full((nu, len(lick_ev), len(LICK_C)), np.nan, np.float32)
    hi_eff = np.minimum(0.150, rts); dur = hi_eff - 0.050
    okw = dur >= 0.020
    valid_bins = (STIM_C[None, :] + W50 / 2) <= rts[:, None]
    for u, s in enumerate(spk):
        b_stim = counts(s, s0, *BASE) / (BASE[1] - BASE[0])
        c = (np.searchsorted(s, s0 + hi_eff) - np.searchsorted(s, s0 + 0.050)).astype(float)
        stimwin[u, okw] = (c[okw] / dur[okw] - b_stim[okw])
        for k, cc in enumerate(STIM_C):
            v = counts(s, s0, cc - W50 / 2, cc + W50 / 2) / W50 - b_stim
            stimbins[u, :, k] = np.where(valid_bins[:, k], v, np.nan)
        b_l = np.where(np.isnan(base_lick), counts(s, lick_ev, *m51.SL_BASE) / (m51.SL_BASE[1] - m51.SL_BASE[0]),
                       counts(s, np.nan_to_num(base_lick), *BASE) / (BASE[1] - BASE[0]))
        for k, cc in enumerate(LICK_C):
            lickbins[u, :, k] = counts(s, lick_ev, cc - W50 / 2, cc + W50 / 2) / W50 - b_l
    np.savez_compressed(OUT / f"{sid}.npz", electrode_group=U.electrode_group.to_numpy(), cluster_id=U.cluster_id.to_numpy(),
                        area_group=U.area_group.to_numpy(), stim_lab=stim_lab, is_hit=is_hit, stim_rt=rts, stim_t=s0, stimwin=stimwin,
                        stimbins=stimbins, stim_c=STIM_C, lick_lab=lick_lab, lick_t=lick_ev, lickbins=lickbins, lick_c=LICK_C)
    return dict(session_id=sid, n_units=nu, n_W=len(iw), n_WH=int(lick[iw].sum()), n_AH=len(ia), n_CR=len(ic), n_FA=len(ifa), n_SL=len(sl))


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    jobs = [(sid, g[["electrode_group", "cluster_id", "area_group"]]) for sid, g in U0.groupby("session_id")]
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = list(pool.imap_unordered(session, jobs))
    S = pd.DataFrame(res).merge(U0.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]], on="session_id", how="left")
    S.to_csv(OUT / "sessions.csv", index=False)
    print(S.drop(columns=[c for c in S if c == "error"], errors="ignore").describe().round(1)); print("errors:", S.error.notna().sum() if "error" in S else 0)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    main(ap.parse_args())
