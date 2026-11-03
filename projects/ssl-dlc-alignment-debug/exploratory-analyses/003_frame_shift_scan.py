"""003 -- Frame-precise check of the DLC frame <-> camera timestamp pairing stored in an NWB (see ../question.md).

A constant pairing error of k frames translates every DLC trace by k x 5 ms relative to the other NWB events, so it is
measured as a latency: the time of the peak of the lick-bout-onset-triggered mean of tongue_likelihood (side camera;
the tongue is visible at spout contact, when the piezo lick is detected) and the time the mean jaw_angle crosses half
of its peak (jaw opening precedes the lick). At the correct pairing these latencies are the same in every session and
in every third of a session (a change between thirds = frames dropped mid-session). Sessions whose video matches all
camera pulses are the controls. Grid: 1 ms, nearest frame within 10 ms; window [-0.3, 0.3] s.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/tables/lick_latency_check.csv, figures/lick_latency_check.png
Run (haas): python 003_frame_shift_scan.py SESSION_ID [SESSION_ID ...]
"""
import importlib.util
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m000", HERE / "000_dlc_event_psth.py")
m000 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m000)
m000.GRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.001), 4)
m000.MAX_GAP = 0.01


def latencies(session_id):
    trials, licks, series = m000.load_session(session_id)
    ev = m000.event_times(trials, licks)["lick"]
    parts = [("all", np.arange(len(ev)))] + [(f"third{j + 1}", th) for j, th in enumerate(np.array_split(np.arange(len(ev)), 3))]
    rows, curves = [], {}
    for name in ("tongue_likelihood", "jaw_angle"):
        if name not in series:
            continue
        data, ts = series[name]
        n = min(len(data), len(ts))
        z = (data[:n] - np.nanmean(data[:n])) / np.nanstd(data[:n])
        mat = m000.psth(z, ts[:n], ev)
        for part, idx in parts:
            m = np.nanmean(mat[idx], 0)
            if name == "tongue_likelihood":
                lat = m000.GRID[np.nanargmax(m)]
            else:
                base = np.nanmean(m[m000.GRID < -0.2])
                pk = np.nanargmax(m)
                half = base + (m[pk] - base) / 2
                above = np.where(m[:pk + 1] < half)[0]
                lat = m000.GRID[above[-1] + 1] if len(above) else np.nan
            rows.append(dict(session_id=session_id, series=name, part=part, n_licks=len(idx), latency_ms=lat * 1000,
                             n_frames=len(data), n_timestamps=len(ts)))
            curves[(name, part)] = m
    return pd.DataFrame(rows), curves


def main(sessions):
    tabs, allcurves = [], {}
    for s in sessions:
        df, curves = latencies(s)
        if df.empty:
            print(s, "no tongue / jaw series")
            continue
        tabs.append(df)
        allcurves[s] = curves
        print(s, df[df.part == "all"][["series", "latency_ms"]].to_dict("records"),
              "| thirds:", df[df.part != "all"].groupby("series").latency_ms.apply(list).to_dict())
    tab = pd.concat(tabs, ignore_index=True)
    out = m000.OUT / "tables" / "lick_latency_check.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.session_id.isin(sessions)]
    pd.concat([old, tab], ignore_index=True).to_csv(out, index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axs, ("tongue_likelihood", "jaw_angle")):
        for s, c in allcurves.items():
            if (name, "all") in c:
                ax.plot(m000.GRID * 1000, c[(name, "all")], lw=1, label=s)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{name}, lick-bout onset (piezo)", fontsize=9)
        ax.set_xlabel("time from lick (ms)")
    axs[0].set_ylabel("z")
    axs[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(m000.OUT / "figures" / "lick_latency_check.png", dpi=140)


if __name__ == "__main__":
    main(sys.argv[1:])
