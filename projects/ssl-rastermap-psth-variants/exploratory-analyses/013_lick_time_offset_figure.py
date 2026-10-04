"""Figure explaining the ~100 ms offset of trials.lick_time relative to the physical first lick.

NWB_converter/utils/behavior_converter_misc.py:527
    lick_time = response_window_start_time + reaction_time
response_window_start_time = start_time + 100 ms (artifact window). If the behaviour software measures
reaction_time from STIMULUS onset (the data say so), lick_time is placed ~100 ms after the real lick.

Panels: (A) timeline schematic; (B) per-trial piezo licks vs start_time, lick_time marked (one session);
(C) lick_time - first piezo lick, per trial; (D) lick-locked units aligned to the first piezo lick vs lick_time
vs lick_time - 100 ms.
Output -> combined_results_ks4/rpe_roc_pilot/lick_time_offset_<session>.png
"""
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sync = importlib.import_module("011_lick_time_sync_check")
ru = sync.ru

SESSION = sys.argv[1] if len(sys.argv) > 1 else "AB154_20250205_172319"
RW_DELAY = 0.1


def main():
    mouse = SESSION.split("_")[0]
    nwb = NWBHDF5IO(str(sync.ROOT / "NWB_ks4" / f"{SESSION}.nwb"), "r").read()
    units, trials = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
    bc = nwb.units.to_dataframe()["bc_label"]
    units = units[bc.loc[units["neuron_id"]].isin(sync.QC).to_numpy()]
    spikes = [np.sort(np.asarray(s)) for s in units["spike_times"]]
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    h = trials[(trials["context"] == "active") & (trials["lick_flag"] == 1) & trials["lick_time"].notna()].copy()
    i = np.searchsorted(pz, h["response_window_start_time"].to_numpy(float))
    h = h[i < len(pz)]; i = i[i < len(pz)]
    h["B"] = pz[i]
    h = h[(h["B"] <= h["lick_time"]) & (h["lick_time"] - h["B"] < 0.3)]          # same lick, piezo detected it
    rw = float(np.median(h["response_window_start_time"] - h["start_time"]))
    off = (h["lick_time"] - h["B"]).to_numpy() * 1000

    csv = pd.read_csv(sync.RES / mouse / "whisker_0" / "spontaneous_licks" / f"{SESSION}_spontaneous_licks.csv")
    S = np.sort(csv.loc[csv["event_type"].isin(["single", "short_cluster", "bout"]), "lick_time"].to_numpy(float))
    S = S[~np.isin(np.round(S, 6), np.round(h["lick_time"].to_numpy(), 6))]
    t, PS = sync.psth_units(spikes, S)
    fS = sync.norm_rows(PS)
    sel = (t >= -0.05) & (t <= 0.1)
    score = np.where(PS.mean(1) >= sync.MIN_RATE_HZ, fS[:, sel].max(1), -np.inf)
    top = [spikes[j] for j in np.argsort(-score)[:sync.N_TOP]]

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(12, 7.2), dpi=220)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.2], hspace=0.45, wspace=0.3)

    # (A) schematic
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-0.08, 0.62); ax.set_ylim(-1.3, 2.6); ax.axis("off")
    rt_true = 0.19                                    # illustrative reaction time from stimulus onset
    ev = [(0.0, "stimulus\n(start_time, synced:\nwhisker artefact at 0)", "#d4a017"),
          (rw, f"response window opens\n(start_time + {rw * 1000:.0f} ms)", "0.4"),
          (rt_true, "physical 1st lick\n(piezo; neurons respond\n+10-20 ms later)", "#c0392b"),
          (rw + rt_true, "trials.lick_time\n= response_window_start\n  + reaction_time", "#1f4fbf")]
    ax.axhline(0, color="k", lw=1)
    for k, (x, txt, c) in enumerate(ev):
        ax.plot([x, x], [-0.15, 0.15 + 0.75 * (k % 2)], color=c, lw=2.5 if k % 2 == 0 else 1)
        ax.plot([x, x], [-0.15, 0.15], color=c, lw=2.5)
        ax.text(x, 0.3 + 0.75 * (k % 2), txt, ha="center", va="bottom", color=c, fontsize=7.5)
    ax.annotate("", xy=(rt_true, -0.45), xytext=(0, -0.45), arrowprops=dict(arrowstyle="<->", color="#c0392b"))
    ax.text(rt_true / 2, -0.6, "reaction_time (measured by the behaviour software from stimulus onset)",
            ha="center", va="top", color="#c0392b", fontsize=7)
    ax.annotate("", xy=(rw + rt_true, -0.95), xytext=(rw, -0.95), arrowprops=dict(arrowstyle="<->", color="#1f4fbf"))
    ax.text(rw + rt_true / 2, -1.1, "same reaction_time, added to the response-window start in the NWB converter",
            ha="center", va="top", color="#1f4fbf", fontsize=7)
    ax.annotate("", xy=(rw + rt_true, 1.95), xytext=(rt_true, 1.95),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(rt_true + rw / 2, 2.02, f"offset = {rw * 1000:.0f} ms", ha="center", va="bottom", fontsize=8)
    ax.text(-0.08, 2.55, "A   NWB_converter/utils/behavior_converter_misc.py:527   "
            "lick_time = response_window_start_time + reaction_time", fontsize=8, va="top", family="monospace")

    # (B) raster
    ax = fig.add_subplot(gs[1, 0])
    h = h.assign(rt_rel=h["lick_time"] - h["start_time"]).sort_values("rt_rel")
    for r, (_, row) in enumerate(h.iterrows()):
        st0 = row["start_time"]
        lk = pz[(pz > st0 - 0.1) & (pz < st0 + 1.0)] - st0
        ax.scatter(lk, np.full(len(lk), r), marker="|", s=6, color="#c0392b", lw=0.8)
        ax.scatter(row["lick_time"] - st0, r, marker="o", s=4, color="#1f4fbf")
    ax.axvline(0, color="#d4a017", lw=1); ax.axvline(rw, color="0.4", lw=0.8, ls="--")
    ax.set_xlim(-0.1, 1.0); ax.set_xlabel("time from stimulus (s)"); ax.set_ylabel("hit trials (sorted by lick_time)")
    ax.set_title("B  piezo licks (red ticks) vs trials.lick_time (blue)", fontsize=8, loc="left")

    # (C) offset histogram
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(off, bins=np.arange(0, 301, 10), color="0.5")
    ax.axvline(rw * 1000, color="0.2", ls="--", lw=0.8)
    ax.set_xlabel("trials.lick_time - first piezo lick (ms)"); ax.set_ylabel("trials")
    ax.set_title(f"C  median {np.median(off):.0f} ms (IQR {np.percentile(off, 25):.0f}-{np.percentile(off, 75):.0f});"
                 f" dashed = {rw * 1000:.0f} ms", fontsize=8, loc="left")

    # (D) neural alignment
    ax = fig.add_subplot(gs[1, 2])
    for name, e, c in (("1st piezo lick", h["B"].to_numpy(), "#c0392b"),
                       ("trials.lick_time", h["lick_time"].to_numpy(), "#1f4fbf"),
                       (f"trials.lick_time - {rw * 1000:.0f} ms", h["lick_time"].to_numpy() - rw, "#7d3c98"),
                       ("spontaneous licks (CSV)", S, "0.5")):
        _, P = sync.psth_units(top, e)
        y = sync.norm_rows(P).mean(0)
        ax.plot(t * 1000, y, color=c, lw=1.2, label=f"{name} (peak {t[np.argmax(y)] * 1000:+.0f} ms)")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("ms from alignment event"); ax.set_ylabel("rate / mean rate")
    ax.set_title(f"D  {sync.N_TOP} lick-locked units (chosen on spontaneous licks)", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False)
    for a in fig.axes[1:]:
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.suptitle(f"{SESSION}: trials.lick_time is ~{rw * 1000:.0f} ms after the physical first lick", fontsize=10)
    out = sync.OUT / f"lick_time_offset_{SESSION}.png"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("saved", out, "n trials", len(h), "median offset", np.median(off))


if __name__ == "__main__":
    main()
