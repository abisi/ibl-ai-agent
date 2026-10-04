"""Corrected first-lick time and neural activity aligned to it (whisker and auditory hits separately).

The NWB converter stores lick_time = response_window_start_time + reaction_time, while the behaviour software's
reaction_time counts from stimulus onset. So:
    reaction_time          = lick_time - response_window_start_time
    corrected_lick_time    = start_time + reaction_time        (= lick_time - (response_window_start - start))

Figure per session:
    A  per-trial response-window delay (response_window_start - start_time)
    B  reaction time (corrected) per modality
    C  corrected_lick_time - nearest piezo lick, per modality (check: ~0 to +25 ms detection latency)
    D  piezo licks aligned to the corrected lick, whisker hits (sorted by RT; stimulus onset marked)
    E  same, auditory hits
    F  lick-locked units (chosen on spontaneous licks): population PSTH aligned to original vs corrected lick,
       whisker and auditory
    G  all units (>= 2 Hz), rate / mean rate aligned to the corrected lick, whisker hits (sorted by peak time)
    H  same units and order, auditory hits
Output -> combined_results_ks4/rpe_roc_pilot/corrected_lick_alignment_<session>.png
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

SESSIONS = sys.argv[1:] or ["AB130_20240902_123634", "AB154_20250205_172319"]
MOD = {"whisker": ("whisker_stim", "#d4a017"), "auditory": ("auditory_stim", "#1f4fbf")}


def corrected_hits(trials):
    h = trials[(trials["context"] == "active") & (trials["lick_flag"] == 1) & trials["lick_time"].notna()].copy()
    h["rw_delay"] = h["response_window_start_time"] - h["start_time"]
    h["reaction_time"] = h["lick_time"] - h["response_window_start_time"]
    h["lick_time_corrected"] = h["start_time"] + h["reaction_time"]
    h["modality"] = np.where(h["whisker_stim"] == 1, "whisker", np.where(h["auditory_stim"] == 1, "auditory", "none"))
    return h[h["modality"] != "none"]


def run(session_id):
    mouse = session_id.split("_")[0]
    nwb = NWBHDF5IO(str(sync.ROOT / "NWB_ks4" / f"{session_id}.nwb"), "r").read()
    units, trials = ru.process_nwb_tables(nwb, apply_artifact_correction=True)
    bc = nwb.units.to_dataframe()["bc_label"]
    units = units[bc.loc[units["neuron_id"]].isin(sync.QC).to_numpy()]
    spikes = [np.sort(np.asarray(s)) for s in units["spike_times"]]
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    h = corrected_hits(trials)
    near = lambda t: np.array([pz[np.argmin(np.abs(pz - x))] - x for x in t])
    h["corr_minus_piezo_ms"] = -near(h["lick_time_corrected"].to_numpy()) * 1000   # corrected - nearest piezo

    csv = pd.read_csv(sync.RES / mouse / "whisker_0" / "spontaneous_licks" / f"{session_id}_spontaneous_licks.csv")
    S = np.sort(csv.loc[csv["event_type"].isin(["single", "short_cluster", "bout"]), "lick_time"].to_numpy(float))
    S = S[~np.isin(np.round(S, 6), np.round(trials["lick_time"].dropna().to_numpy(), 6))]
    t, PS = sync.psth_units(spikes, S)
    fS = sync.norm_rows(PS)
    sel = (t >= -0.05) & (t <= 0.1)
    rate_ok = PS.mean(1) >= sync.MIN_RATE_HZ
    top = np.argsort(-np.where(rate_ok, fS[:, sel].max(1), -np.inf))[:sync.N_TOP]

    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(14, 8.2), dpi=200)
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.32)
    # A: response-window delay
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(h["rw_delay"] * 1000, bins=np.arange(90, 111, 0.5), color="0.5")
    ax.set_xlabel("response_window_start - start_time (ms)"); ax.set_ylabel("hit trials")
    ax.set_title("A  response-window delay per trial\n"
                 "reaction_time = lick_time - response_window_start\n"
                 "corrected_lick = start_time + reaction_time", fontsize=7.5, loc="left")
    # B: RT per modality
    ax = fig.add_subplot(gs[0, 1])
    for m, (_, c) in MOD.items():
        rt = h.loc[h["modality"] == m, "reaction_time"] * 1000
        ax.hist(rt, bins=np.arange(0, 1001, 25), color=c, alpha=0.6, label=f"{m} (n={len(rt)}, median {rt.median():.0f} ms)")
    ax.set_xlabel("reaction time from stimulus (ms)"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("B  corrected reaction time", fontsize=7.5, loc="left")
    # C: corrected - piezo
    ax = fig.add_subplot(gs[0, 2])
    for m, (_, c) in MOD.items():
        d = h.loc[h["modality"] == m, "corr_minus_piezo_ms"]
        ax.hist(d, bins=np.arange(-100, 101, 5), color=c, alpha=0.6, label=f"{m}: median {d.median():+.0f} ms")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("corrected lick - nearest piezo lick (ms)"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("C  check against the piezo licks", fontsize=7.5, loc="left")
    # F: lick-locked population, original vs corrected
    ax = fig.add_subplot(gs[0, 3])
    for m, (_, c) in MOD.items():
        hm = h[h["modality"] == m]
        for col, ls, lab in (("lick_time", ":", "original lick_time"), ("lick_time_corrected", "-", "corrected")):
            _, P = sync.psth_units([spikes[j] for j in top], hm[col].to_numpy())
            y = sync.norm_rows(P).mean(0)
            ax.plot(t * 1000, y, color=c, ls=ls, lw=1.2, label=f"{m}, {lab} (peak {t[np.argmax(y)] * 1000:+.0f} ms)")
    _, P = sync.psth_units([spikes[j] for j in top], S)
    y = sync.norm_rows(P).mean(0)
    ax.plot(t * 1000, y, color="0.5", lw=1, label=f"spontaneous (peak {t[np.argmax(y)] * 1000:+.0f} ms)")
    ax.axvline(0, color="k", lw=0.5, ls=":"); ax.set_xlabel("ms from lick"); ax.set_ylabel("rate / mean rate")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.set_title(f"F  {sync.N_TOP} lick-locked units (chosen on spontaneous licks)", fontsize=7.5, loc="left")
    # D/E: piezo rasters aligned to the corrected lick
    for k, (m, (_, c)) in enumerate(MOD.items()):
        ax = fig.add_subplot(gs[1, k])
        hm = h[h["modality"] == m].sort_values("reaction_time")
        for r, (_, row) in enumerate(hm.iterrows()):
            t0 = row["lick_time_corrected"]
            lk = pz[(pz > t0 - 0.6) & (pz < t0 + 0.6)] - t0
            ax.scatter(lk, np.full(len(lk), r), marker="|", s=5, color="#c0392b", lw=0.7)
            ax.scatter(row["start_time"] - t0, r, marker="o", s=3, color=c)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-0.6, 0.6); ax.set_xlabel("time from corrected lick (s)")
        ax.set_ylabel(f"{m} hits (sorted by RT)")
        ax.set_title(f"{'DE'[k]}  {m} hits: piezo licks (red),\nstimulus onset (dot)", fontsize=7.5, loc="left")
    # G/H: all units heatmap aligned to the corrected lick
    Pm, Po = {}, {}                                   # cross-validated: sort on odd trials, show even trials
    for m in MOD:
        e = h.loc[h["modality"] == m, "lick_time_corrected"].to_numpy()
        sp_ok = [spikes[j] for j in np.flatnonzero(rate_ok)]
        _, Pe = sync.psth_units(sp_ok, e[0::2]); _, Pod = sync.psth_units(sp_ok, e[1::2])
        Pm[m], Po[m] = sync.norm_rows(Pe), sync.norm_rows(Pod)
    order = np.argsort(np.argmax(Po["whisker"] + Po["auditory"], 1))
    for k, m in enumerate(MOD):
        ax = fig.add_subplot(gs[1, 2 + k])
        im = ax.imshow(Pm[m][order], aspect="auto", cmap="magma", vmin=0, vmax=3,
                       extent=[t[0] * 1000, t[-1] * 1000, len(order), 0], interpolation="none")
        ax.axvline(0, color="w", lw=0.6, ls=":")
        ax.set_xlabel("ms from corrected lick"); ax.set_ylabel("units (>= 2 Hz)")
        ax.set_title(f"{'GH'[k]}  {m} hits, all units: even trials,\nsorted by peak time on odd trials",
                     fontsize=7.5, loc="left")
        if k == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("rate / mean rate")
    for a in fig.axes:
        if a.get_label() != "<colorbar>":
            for s_ in ("top", "right"):
                a.spines[s_].set_visible(False)
    cohort = "R+" if trials.loc[(trials["context"] == "active") & (trials["whisker_stim"] == 1),
                                "reward_available"].mean() > 0.5 else "R-"
    fig.suptitle(f"{session_id} ({cohort}): lick alignment with corrected_lick_time = start_time + reaction_time  "
                 f"({len(spikes)} good+mua units, {int(rate_ok.sum())} >= {sync.MIN_RATE_HZ:g} Hz)", fontsize=9)
    out = sync.OUT / f"corrected_lick_alignment_{session_id}.png"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    summ = h.groupby("modality").agg(n=("reaction_time", "size"), rt_median_ms=("reaction_time", lambda x: 1000 * x.median()),
                                     corr_minus_piezo_median_ms=("corr_minus_piezo_ms", "median"),
                                     rw_delay_ms=("rw_delay", lambda x: 1000 * x.median()))
    summ.insert(0, "session_id", session_id)
    print(summ.round(1).to_string(), flush=True)
    return summ


if __name__ == "__main__":
    s = pd.concat([run(x) for x in SESSIONS])
    s.to_csv(sync.OUT / "corrected_lick_alignment_summary.csv")
    print("ALL DONE")
