"""006 -- Direct evidence for the DLC frame <-> camera pulse pairing of one session (see ../question.md).

Plain event-triggered views, one per camera, under two pairings of DLC frame f with camera pulse f + D:
  NWB       : the pairing stored in the NWB (stored timestamp of frame f -> pulse index)
  proposed  : end-alignment, D = number of pulses - number of DLC frames (last frame <-> last pulse)
Side camera: tongue_likelihood around lick-bout onsets (piezo, no lick in the previous 1 s). Top camera: whisker_angle
(z) around the same lick-bout onsets (the tongue is not visible from the top). Each lick bout is one row of the
heatmaps (session order top to bottom), so a wrong pairing shows as the whole stripe of licking displaced from 0 by
(D_proposed - D_NWB) / 200 s, in every bout. Also: the cross-correlation of the DLC signal with the lick train as a
function of the assumed D (whole session and per third; one peak, at the same D in every third = a constant offset) and
the inter-pulse intervals at the start of the pulse run.
Pulses (ephys sessions): NIDQ sync_event_times (cam0 = top, cam1 = side) mapped to the NWB (imec) clock by a linear fit
on the frames the NWB pairs; when the NIDQ train is shorter than the stored timestamps (MH038_20250524, log pulses) the
stored timestamps are used, and the NIDQ pulses are compared with them.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/figures/<session>_pairing_evidence.png, tables/pairing_evidence.csv
Run (haas, ibl-ai-agent .venv): python 006_pairing_evidence.py SESSION_ID [SESSION_ID ...]
"""
import glob
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path("NWB_ks4")
DATA = axel_bisi_path("data")
OUT = axel_bisi_path("combined_results_ks4") / "ssl-dlc-alignment-debug"
FR = 200.0
LAG_RANGE = 3000  # frames scanned around 0 and around D_proposed in the cross-correlation panel


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def load(session_id):
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        b = f["processing/behavior/BehavioralTimeSeries"]
        licks = np.sort(f["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:])
        sig = {}
        if "tongue_likelihood" in b:
            sig["side"] = (np.nan_to_num(b["tongue_likelihood"]["data"][:].astype(float)), b["jaw_angle"]["timestamps"][:])
        if "whisker_angle" in b:
            sig["top"] = (np.nan_to_num(zs(b["whisker_angle"]["data"][:])), b["whisker_angle"]["timestamps"][:])
    sync = glob.glob(str(DATA / session_id[:5] / session_id / "Ephys" / "catgt_*" / "sync_event_times"))
    nidq = {}
    if sync:
        nidq = {"top": np.loadtxt(sync[0] + "/cam0_frame_times.txt"), "side": np.loadtxt(sync[0] + "/cam1_frame_times.txt")}
    bouts = licks[np.r_[True, np.diff(licks) > 1.0]]
    return sig, nidq, licks, bouts


def pulses_on_nwb_clock(stored, p_nidq):
    """NIDQ pulses mapped to the NWB clock, and D_NWB (pulse index of frame 0). Falls back to the stored timestamps."""
    if p_nidq is None or len(p_nidq) < len(stored) - 200:
        return stored, 0, "stored NWB timestamps"
    k0 = int(np.argmin(np.abs(p_nidq - stored[0])))
    n = min(len(stored), len(p_nidq) - k0)
    a, b = np.polyfit(p_nidq[k0:k0 + n], stored[:n], 1)
    res = stored[:n] - (a * p_nidq[k0:k0 + n] + b)
    return a * p_nidq + b, k0, f"NIDQ pulses -> NWB clock (fit residual max {1e3 * np.abs(res).max():.2f} ms)"


def xcorr(x, pulses, events, lags):
    """c[L] = mean of x over frames f whose pulse f + L carries an event, for each L in lags."""
    e = np.zeros(len(pulses))
    i = np.searchsorted(pulses, events)
    e[i[(i > 0) & (i < len(pulses))]] = 1
    c = fftconvolve(e, x[::-1], mode="full")
    allL = np.arange(len(c)) - (len(x) - 1)
    return np.interp(lags, allL, c)


def frame_times(n_frames, pulses, D):
    t = np.full(n_frames, np.nan)
    f = np.arange(n_frames)
    ok = (f + D >= 0) & (f + D < len(pulses))
    t[ok] = pulses[f[ok] + D]
    return t


def triggered(x, t, events, grid):
    ok = np.isfinite(t)
    x, t = x[ok], t[ok]
    q = events[:, None] + grid[None, :]
    i = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    j = np.where(np.abs(q - t[i - 1]) <= np.abs(t[i] - q), i - 1, i)
    out = x[j]
    out[np.abs(t[j] - q) > 0.02] = np.nan
    return out


def main(session_id):
    sig, nidq, licks, bouts = load(session_id)
    views = [v for v in ("side", "top") if v in sig]
    fig, axs = plt.subplots(len(views), 5, figsize=(22, 4.6 * len(views)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1, 0.9]))
    rows = []
    for r, view in enumerate(views):
        x, stored = sig[view]
        n = len(x)
        pulses, D_nwb, src = pulses_on_nwb_clock(stored, nidq.get(view))
        D_prop = len(pulses) - n
        shift_s = (D_prop - D_nwb) / FR
        half = max(1.0, 1.4 * abs(shift_s))
        grid = np.round(np.arange(-half, half + 1e-9, 0.01), 3)
        # a: cross-correlation vs assumed D
        lags = np.unique(np.r_[np.arange(D_nwb - LAG_RANGE, D_nwb + LAG_RANGE + 1), np.arange(D_prop - LAG_RANGE, D_prop + LAG_RANGE + 1)])
        ax = axs[r, 0]
        thirds = np.array_split(np.arange(n), 3)
        for k, (idx, col) in enumerate(zip(thirds, ["#9ecae1", "#4292c6", "#08306b"])):
            xi = np.zeros(n)
            xi[idx] = zs(x[idx])
            c = xcorr(xi, pulses, licks, lags)
            c = (c - np.median(c)) / (np.median(np.abs(c - np.median(c))) or 1)
            # whisker angle dips at licks (negative peak): take the largest |deviation|
            pk = int(lags[np.argmax(np.abs(c))]) if view == "top" else int(lags[np.argmax(c)])
            ax.plot(lags, c, color=col, lw=0.8, label=f"third {k + 1}: peak at D = {pk}")
            rows.append(dict(session_id=session_id, view=view, part=f"third{k + 1}", peak_D=pk,
                             D_nwb=D_nwb, D_proposed=D_prop, n_frames=n, n_pulses=len(pulses), pulse_source=src))
        ax.axvline(D_nwb, color="tab:red", lw=1, ls="--", label=f"NWB pairing D = {D_nwb}")
        ax.axvline(D_prop, color="tab:green", lw=1, ls="--", label=f"end-alignment D = {D_prop}")
        ax.set_xlabel("assumed D (DLC frame f <-> pulse f + D)", fontsize=8)
        ax.set_ylabel("DLC x licks correlation (robust z)", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.set_title(f"{view}: {n} DLC frames, {len(pulses)} pulses\n({src})", fontsize=8)
        # b, c: heatmaps per lick bout under both pairings
        ev = bouts[(bouts > pulses[0] + half) & (bouts < pulses[-1] - half)]
        for c_, (D, name) in zip((1, 2), ((D_nwb, "NWB pairing"), (D_prop, "end-alignment"))):
            m = triggered(x, frame_times(n, pulses, D), ev, grid)
            ax = axs[r, c_]
            vmax = np.nanpercentile(m, 99) if np.isfinite(m).any() else 1
            ax.imshow(m, aspect="auto", extent=[grid[0], grid[-1], len(ev), 0], cmap="magma", vmin=np.nanpercentile(m, 1) if np.isfinite(m).any() else 0, vmax=vmax, interpolation="nearest")
            ax.axvline(0, color="c", lw=0.6)
            ax.set_title(f"{name} (D = {D}): {'tongue likelihood' if view == 'side' else 'whisker angle (z)'}\nper lick-bout onset", fontsize=8)
            ax.set_xlabel("time from lick-bout onset (s)", fontsize=8)
            ax.set_ylabel("lick bout (session order)", fontsize=8)
        # d: means, zoom
        ax = axs[r, 3]
        zg = np.round(np.arange(-0.5, 0.5 + 1e-9, 0.005), 3)
        for D, name, col in ((D_nwb, "NWB", "tab:red"), (D_prop, "end-align", "tab:green")):
            m = triggered(x, frame_times(n, pulses, D), ev, zg)
            for k, (idx, ls) in enumerate(zip(np.array_split(np.arange(len(ev)), 3), ["-", "--", ":"])):
                ax.plot(zg * 1e3, np.nanmean(m[idx], 0), color=col, ls=ls, lw=1, label=f"{name}, third {k + 1}")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("time from lick-bout onset (ms)", fontsize=8)
        ax.set_title("mean per third of the session", fontsize=8)
        ax.legend(fontsize=6)
        # e: inter-pulse intervals at the start of the run
        ax = axs[r, 4]
        k = max(2 * max(D_prop, D_nwb, 0) + 200, 2000)
        ipi = np.diff(pulses[:k]) * 1e3
        ax.plot(np.arange(1, len(ipi) + 1), ipi, ".", ms=1.5, color="k")
        for D, col in ((D_nwb, "tab:red"), (D_prop, "tab:green")):
            if D > 0:
                ax.axvline(D, color=col, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("pulse index", fontsize=8)
        ax.set_ylabel("inter-pulse interval (ms)", fontsize=8)
        ax.set_title(f"first {k} pulses (first pulse at {pulses[0]:.2f} s)", fontsize=8)
        for a in axs[r]:
            a.tick_params(labelsize=7)
    if session_id == "MH038_20250524_105329" and "side" in nidq:
        stored = sig["side"][1]
        p = nidq["side"]
        j = np.clip(np.searchsorted(stored, p), 1, len(stored) - 1)
        j = np.where(np.abs(p - stored[j - 1]) <= np.abs(stored[j] - p), j - 1, j)
        d = (p - stored[j]) * 1e3
        print(f"MH038 0524 side: {len(p)} NIDQ pulses vs log-derived timestamps: nearest index {j[0]}..{j[-1]} "
              f"(consecutive {np.all(np.diff(j) == 1)}), diff median {np.median(d):.3f} ms, range {d.min():.3f}..{d.max():.3f} ms")
        rows.append(dict(session_id=session_id, view="side", part="nidq_vs_log", peak_D=int(j[0]),
                         D_nwb=np.median(d), D_proposed=d.max() - d.min(), n_frames=len(stored), n_pulses=len(p),
                         pulse_source="NIDQ pulses vs log-derived (ms)"))
    fig.suptitle(f"{session_id}: DLC frame <-> camera pulse pairing, NWB vs end-alignment", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{session_id}_pairing_evidence.png", dpi=120)
    plt.close(fig)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    out = OUT / "tables" / "pairing_evidence.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[old.session_id != session_id]
    pd.concat([old, t]).to_csv(out, index=False)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        try:
            main(s)
        except Exception as e:
            print(s, "ERROR", type(e).__name__, e)
