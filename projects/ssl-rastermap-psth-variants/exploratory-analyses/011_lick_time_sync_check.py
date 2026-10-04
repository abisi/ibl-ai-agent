"""Which lick time base is aligned with the neural data: trials.lick_time or BehavioralEvents piezo_lick_times?

Observation (AB130/AB154/MH065 day 0): trials.lick_time lags the first piezo lick after response-window start by
~70-120 ms, and the spontaneous-lick CSV onsets are exact piezo_lick_times. Test with neurons: lick-locked units
should respond at a short, fixed latency after the true tongue contact.

Per session (good+mua units, active-context hits):
    A = trials.lick_time
    B = first piezo lick after response_window_start_time (the same lick, piezo clock)
    S = spontaneous-lick onsets from the CSV (single/short_cluster/bout)   [piezo clock, independent event set]
Lick-locked units are selected on S only (peak |z| of the S-aligned PSTH in [-50, 100] ms), then the population
PSTH of those units is compared when aligned to A vs B. Also: whisker-stimulus-aligned PSTH (trial clock sanity).
Output -> combined_results_ks4/rpe_roc_pilot/lick_sync_check_<session>.png + lick_sync_check.csv
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru      # noqa: E402

ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi")
RES = ROOT / "combined_results_ks4"
OUT = RES / "rpe_roc_pilot"
SESSIONS = ["AB130_20240902_123634", "AB154_20250205_172319", "MH065_20260114_154021"]
QC = ["good", "mua"]
WIN, BW = (-0.3, 0.3), 0.005
N_TOP = 60


def psth_units(spikes, events, win=WIN, bw=BW):
    edges = np.arange(win[0], win[1] + bw / 2, bw)
    out = np.zeros((len(spikes), len(edges) - 1))
    for i, st in enumerate(spikes):
        for t in events:
            s = st[np.searchsorted(st, t + win[0]):np.searchsorted(st, t + win[1])] - t
            out[i] += np.histogram(s, edges)[0]
    return edges[:-1] + bw / 2, out / max(len(events), 1) / bw


MIN_RATE_HZ = 2.0


def norm_rows(P):
    """smoothed PSTH as fold change over each unit's mean rate in the window (robust for sparse units)"""
    from scipy.ndimage import gaussian_filter1d
    Ps = gaussian_filter1d(P, 2, axis=1)
    return Ps / np.maximum(P.mean(1, keepdims=True), 1e-6)


def run(session_id):
    mouse = session_id.split("_")[0]
    nwb = NWBHDF5IO(str(ROOT / "NWB_ks4" / f"{session_id}.nwb"), "r").read()
    units, trials = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
    bc = nwb.units.to_dataframe()["bc_label"]
    units = units[bc.loc[units["neuron_id"]].isin(QC).to_numpy()]
    spikes = [np.sort(np.asarray(s)) for s in units["spike_times"]]
    ev = nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
    pz = np.sort(np.asarray(ev.time_series["piezo_lick_times"].data[:]))
    hits = trials[(trials["context"] == "active") & (trials["lick_flag"] == 1) & trials["lick_time"].notna()]
    A = hits["lick_time"].to_numpy(float)
    i = np.searchsorted(pz, hits["response_window_start_time"].to_numpy(float))
    ok = i < len(pz)
    B = pz[i[ok]]; A_ok = A[ok]
    csv = pd.read_csv(RES / mouse / "whisker_0" / "spontaneous_licks" / f"{session_id}_spontaneous_licks.csv")
    S = np.sort(csv.loc[csv["event_type"].isin(["single", "short_cluster", "bout"]), "lick_time"].to_numpy(float))
    S = S[~np.isin(np.round(S, 6), np.round(A, 6))]                  # drop CSV events that are trial licks
    W = trials.loc[(trials["context"] == "active") & (trials["whisker_stim"] == 1), "start_time"].to_numpy(float)

    t, PS = psth_units(spikes, S)
    fS = norm_rows(PS)
    sel = (t >= -0.05) & (t <= 0.1)
    eligible = PS.mean(1) >= MIN_RATE_HZ
    score = np.where(eligible, fS[:, sel].max(1), -np.inf)          # lick-activated units (peak fold change)
    top = np.argsort(-score)[:N_TOP]
    res = {}
    for name, e in (("S spontaneous (piezo clock)", S), ("B first piezo lick in trial", B),
                    ("A trials.lick_time", A_ok)):
        _, P = psth_units([spikes[j] for j in top], e)
        res[name] = norm_rows(P).mean(0)
    tw, PW = psth_units(spikes, W, win=(-0.05, 0.1), bw=0.001)
    dAB = (A_ok - B) * 1000
    lat = {k: float(t[np.argmax(v)] * 1000) for k, v in res.items()}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), dpi=200)
    for (k, v), c in zip(res.items(), ["0.4", "#c0392b", "#1f4fbf"]):
        ax[0].plot(t * 1000, v, color=c, lw=1.2, label=f"{k} (peak {lat[k]:+.0f} ms)")
    ax[0].axvline(0, color="k", lw=0.5, ls=":"); ax[0].set_xlabel("ms from event"); ax[0].set_ylabel("rate / mean rate")
    ax[0].set_title(f"{N_TOP} most lick-locked units (selected on spontaneous licks)", fontsize=8)
    ax[0].legend(fontsize=6, frameon=False)
    ax[1].hist(dAB, bins=np.arange(-300, 301, 10), color="0.5")
    ax[1].set_xlabel("trials.lick_time - first piezo lick (ms)"); ax[1].set_title(
        f"median {np.median(dAB):.0f} ms, IQR [{np.percentile(dAB, 25):.0f}, {np.percentile(dAB, 75):.0f}]", fontsize=8)
    ax[2].plot(tw * 1000, PW.mean(0), color="#d4a017", lw=1)
    ax[2].axvline(0, color="k", lw=0.5, ls=":"); ax[2].set_xlabel("ms from whisker stim (trials.start_time)")
    ax[2].set_title("population rate, whisker trials (trial clock check)", fontsize=8)
    for a in ax:
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.suptitle(f"{session_id}: {len(spikes)} {'+'.join(QC)} units, {len(A_ok)} hits, {len(S)} spontaneous events",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / f"lick_sync_check_{session_id}.png"); plt.close(fig)
    return dict(session_id=session_id, n_units=len(spikes), n_hits=len(A_ok), n_spont=len(S),
                lickTime_minus_piezo_median_ms=float(np.median(dAB)),
                lickTime_minus_piezo_iqr_ms=[float(np.percentile(dAB, 25)), float(np.percentile(dAB, 75))],
                **{f"peak_ms__{k.split()[0]}": v for k, v in lat.items()})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run(s) for s in SESSIONS]
    df = pd.DataFrame(rows); df.to_csv(OUT / "lick_sync_check.csv", index=False)
    print(df.to_string(index=False)); print("ALL DONE")
