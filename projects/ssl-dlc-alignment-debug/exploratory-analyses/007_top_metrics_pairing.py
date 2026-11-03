"""007 -- Top-camera pairing from every top-view DLC metric (see ../question.md, 006).

For each top-view series in the NWB (same timestamps as whisker_angle), as value (z) and speed (|frame-to-frame change|,
z): correlation with the event train as a function of the assumed pairing D (DLC frame f <-> camera pulse f + D), for
all piezo licks and for whisker stimulus onsets, and the event-triggered mean under the NWB pairing and under
end-alignment. Signals clipped at +-5 z. Peak D = largest |robust z| (metrics can dip or rise at an event). A metric supports a pairing when its
peak sits a few frames from it, with the same lead / lag as in the controls.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/figures/<session>_top_metrics.png, tables/top_metrics_pairing.csv
Run (haas, ibl-ai-agent .venv): python 007_top_metrics_pairing.py SESSION_ID [SESSION_ID ...]
"""
import importlib.util
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m006", HERE / "006_pairing_evidence.py")
m006 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m006)
CLIP = 5.0
MARGIN = 400  # frames scanned beyond the NWB / end-alignment pairings


def load(session_id):
    with h5py.File(m006.NWB_DIR / f"{session_id}.nwb", "r") as f:
        b = f["processing/behavior/BehavioralTimeSeries"]
        ref = b["whisker_angle"]["timestamps"]
        n_ref, t0 = ref.shape[0], ref[0]
        series = {}
        for k in sorted(b):
            if "timestamps" in b[k] and b[k]["timestamps"].shape[0] == n_ref and b[k]["timestamps"][0] == t0:
                series[k] = b[k]["data"][:].astype(float)
        stored = ref[:]
        licks = np.sort(f["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:])
        t = f["intervals/trials"]
        ws = t["whisker_stim"][:] == 1
        wst = t["whisker_stim_time"][:][ws] if "whisker_stim_time" in t else t["start_time"][:][ws]
    _, nidq, _, _ = m006.load(session_id)
    return series, stored, nidq.get("top"), licks, np.sort(wst[np.isfinite(wst)])


def robust(c):
    return (c - np.median(c)) / (np.median(np.abs(c - np.median(c))) or 1)


def main(session_id):
    series, stored, p_nidq, licks, wst = load(session_id)
    n = len(next(iter(series.values())))
    pulses, D_nwb, src = m006.pulses_on_nwb_clock(stored, p_nidq)
    D_end = len(pulses) - n
    lags = np.arange(min(D_nwb, D_end) - MARGIN, max(D_nwb, D_end) + MARGIN + 1)
    grid = np.round(np.arange(-0.5, 0.5 + 1e-9, 0.005), 3)
    names = list(series)
    fig, axs = plt.subplots(len(names), 4, figsize=(18, 1.9 * len(names)), squeeze=False)
    rows = []
    for r, name in enumerate(names):
        # clipped at +-5 z: single-frame tracking glitches otherwise dominate sessions with few events
        x = np.clip(np.nan_to_num(m006.zs(series[name])), -CLIP, CLIP)
        sp = np.clip(np.nan_to_num(m006.zs(np.abs(np.diff(series[name], prepend=np.nan)))), -CLIP, CLIP)
        for c_, (ev, ev_name) in enumerate(((licks, "licks"), (wst, "whisker stim"))):
            ax = axs[r, c_]
            for sig, kind, col in ((x, "value", "tab:blue"), (sp, "speed", "tab:orange")):
                c = robust(m006.xcorr(sig, pulses, ev, lags))
                pk = int(lags[np.argmax(np.abs(c))])
                ax.plot(lags, c, color=col, lw=0.7, label=f"{kind}: peak {pk} (|z| {np.abs(c).max():.0f})")
                rows.append(dict(session_id=session_id, metric=name, signal=kind, events=ev_name, n_events=len(ev),
                                 peak_D=pk, peak_abs_z=float(np.abs(c).max()), D_nwb=D_nwb, D_end=D_end))
            ax.axvline(D_nwb, color="tab:red", ls="--", lw=0.8)
            ax.axvline(D_end, color="tab:green", ls="--", lw=0.8)
            ax.legend(fontsize=5.5, loc="upper right")
            ax.set_title(f"{name} x {ev_name} (n={len(ev)})", fontsize=7)
        for c_, (ev, ev_name) in enumerate(((licks, "licks"), (wst, "whisker stim")), start=2):
            ax = axs[r, c_]
            for D, lab, col in ((D_nwb, "NWB", "tab:red"), (D_end, "end-align", "tab:green")):
                m = m006.triggered(x, m006.frame_times(len(x), pulses, D), ev, grid)
                ax.plot(grid * 1e3, np.nanmean(m, 0), color=col, lw=0.9, label=f"{lab} (D={D})")
            ax.axvline(0, color="k", lw=0.5)
            ax.legend(fontsize=5.5)
            ax.set_title(f"{name} (z) around {ev_name}", fontsize=7)
        for a in axs[r]:
            a.tick_params(labelsize=6)
    for a in axs[-1, :2]:
        a.set_xlabel("assumed D (frame f <-> pulse f + D); red = NWB, green = end-alignment", fontsize=7)
    for a in axs[-1, 2:]:
        a.set_xlabel("time from event (ms)", fontsize=7)
    fig.suptitle(f"{session_id}: top-camera metrics vs pairing ({n} frames, {len(pulses)} pulses, {src})", fontsize=9)
    fig.tight_layout()
    fig.savefig(m006.OUT / "figures" / f"{session_id}_top_metrics.png", dpi=110)
    plt.close(fig)
    t = pd.DataFrame(rows)
    print(t[["metric", "signal", "events", "n_events", "peak_D", "peak_abs_z", "D_nwb", "D_end"]].to_string(index=False))
    out = m006.OUT / "tables" / "top_metrics_pairing.csv"
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
