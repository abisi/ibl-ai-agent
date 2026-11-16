"""Jaw onsets detected before the stimulus (user, 2026-11-12): how often, is it allowed, do piezo licks match, is it a
true behavioural response?

movement_onset_times searches from start_time - 0.5 s (pre_onset_window) and keeps the first threshold-crossing bout
whose first local PEAK is after start_time, walked back to 30 % of that peak: a bout that starts before the stimulus and
peaks after it gives a negative onset (allowed by design). Per trial (new pkls, lick trials, active, perf != 6):
  onset < 0; jaw distance at stimulus time (baseline SD units; > 5 = already above threshold); max jaw distance in the
  baseline window (-1, -0.1) s (the onset baseline); piezo lick events (NWB piezo_lick_times, own clock +-25 ms) in
  (-0.5, 0) s; offset-corrected first lick (lick_time - response_window_start_time).
Figure, one row per trial type (whisker hit, auditory hit, false alarm):
  col 1: onset histogram around the stimulus
  col 2: stimulus-aligned jaw distance (mean over mice), onset < 0 vs >= 0 trials
  col 3: first-lick time, onset < 0 vs >= 0 trials
  col 4: onset < 0 trials (up to 150, pooled), sorted by onset: onset (purple), first lick (pink), piezo events (black)
Output: combined_results_ks4/ssl-dlc-jaw-onset/figures/007_prestim_onsets.png (+ .pdf), tables/007_prestim_onsets.csv
(per trial), tables/007_prestim_onsets_summary.csv
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 007_prestim_onsets.py
"""
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

R = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = R / "ssl-dlc-jaw-onset"
TYPES = {"whisker_trial": ("Whisker hit", "#f7b519"), "auditory_trial": ("Auditory hit", "#2c2cdb"),
         "no_stim_trial": ("False alarm", "0.3")}
T0 = -1.0
PZ = "processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"


def one_mouse(pkl):
    d = pd.read_pickle(pkl)
    d = d[(d.context == "active") & (d.perf != 6) & (d.lick_flag == 1)].copy()
    rows, traces = [], []
    for sid, s in d.groupby("session_id"):
        try:
            with h5py.File(s.nwb_file.iloc[0], "r") as f:
                pz = f[PZ][:] if PZ in f else np.array([])
        except OSError:
            pz = np.array([])
        for r in s.itertuples():
            fs = float(r.freq)
            tr = np.asarray(r.jaw_dlc_pos, float) / (r.threshold / 5.0)
            i0 = int(round(-T0 * fs))
            ev = pz[(pz >= r.start_time - 1.0) & (pz < r.start_time + 1.5)] - r.start_time
            rows.append(dict(mouse_id=r.mouse_id, session_id=sid, trial_id=r.trial_id, trial_type=r.trial_type,
                             day=r.day, onset=r.jaw_dlc_onset, lick=r.lick_time - r.response_window_start_time,
                             pos_at_stim=tr[i0], base_max=np.nanmax(tr[:int(round(0.9 * fs))]),
                             n_piezo_pre=int(np.sum((ev >= -0.5) & (ev < 0))), early_lick=r.early_lick))
            traces.append((tr if round(fs) == 200 else None, ev))
    return rows, traces


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pkls = sorted(R.glob("*/dlc_jaw_onset_times.pkl"))
    out = Parallel(n_jobs=48)(delayed(one_mouse)(p) for p in pkls)
    tab = pd.DataFrame([r for rows, _ in out for r in rows])
    trs = [t for _, tr in out for t in tr]
    tab["pre"] = tab.onset < 0
    tab["above_thr_at_stim"] = tab.pos_at_stim > 5
    tab.to_csv(OUT / "tables" / "007_prestim_onsets.csv", index=False)
    per_mouse = tab.groupby(["trial_type", "mouse_id"]).pre.mean().mul(100)
    summ = []
    for tt in TYPES:
        t = tab[tab.trial_type == tt]
        for pre in (True, False):
            u = t[t.pre == pre]
            summ.append(dict(trial_type=tt, onset_before_stim=pre, n_trials=len(u),
                             pct_of_trials=100 * len(u) / len(t),
                             median_pct_per_mouse=per_mouse[tt].median() if pre else np.nan,
                             pct_above_thr_at_stim=100 * u.above_thr_at_stim.mean(),
                             pct_baseline_above_thr=100 * (u.base_max > 5).mean(),
                             pct_piezo_lick_pre_stim=100 * (u.n_piezo_pre > 0).mean(),
                             median_onset_ms=1000 * u.onset.median(), median_first_lick_ms=1000 * u.lick.median(),
                             median_onset_to_lick_ms=1000 * (u.lick - u.onset).median()))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "tables" / "007_prestim_onsets_summary.csv", index=False)
    print(summ.round(1).to_string())

    fig, axs = plt.subplots(3, 4, figsize=(16, 11), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    x = None
    for r_, (tt, (name, col)) in enumerate(TYPES.items()):
        idx = np.flatnonzero(tab.trial_type.to_numpy() == tt)
        t = tab.iloc[idx]
        ax = axs[r_, 0]
        ax.hist(t.onset, bins=np.arange(-0.5, 0.6, 0.01), color=col)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{name}: onset < 0 in {100 * t.pre.mean():.1f}% of trials "
                     f"(median mouse {per_mouse[tt].median():.1f}%)", fontsize=8)
        ax.set_xlabel("Jaw onset from stimulus (s)")
        for pre, ls, lab in ((True, "-", "onset < 0"), (False, "--", "onset >= 0")):
            sel = [k for k in idx if tab.pre.iat[k] == pre and trs[k][0] is not None]
            if not sel:
                continue
            per = pd.DataFrame({"m": tab.mouse_id.to_numpy()[sel], "k": sel}).groupby("m").k.apply(list)
            arr = np.array([np.nanmean(np.vstack([trs[k][0] for k in ks]), 0) for ks in per])
            x = T0 + np.arange(arr.shape[1]) / 200.0
            mu, se = np.nanmean(arr, 0), np.nanstd(arr, 0) / np.sqrt(len(arr))
            axs[r_, 1].plot(x, mu, ls=ls, color=col, label=f"{lab} ({len(sel)} trials, {len(arr)} mice)")
            axs[r_, 1].fill_between(x, mu - se, mu + se, color=col, alpha=0.2, lw=0)
            u = tab.iloc[sel]
            axs[r_, 2].hist(u.lick, bins=np.arange(0, 1.2, 0.02), density=True, histtype="step", ls=ls, color=col,
                            label=f"{lab}: median {1000 * u.lick.median():.0f} ms")
        axs[r_, 1].axhline(5, color="r", lw=0.6, ls=":")
        axs[r_, 1].axvline(0, color="k", lw=0.5)
        axs[r_, 1].set_xlim(-1, 1)
        axs[r_, 1].set_xlabel("Time from stimulus (s)")
        axs[r_, 1].set_ylabel("Jaw distance (baseline SD)")
        axs[r_, 1].legend(fontsize=6, frameon=False)
        axs[r_, 2].set_xlabel("First lick from stimulus (s)")
        axs[r_, 2].legend(fontsize=6, frameon=False)
        pre_idx = [k for k in idx if tab.pre.iat[k] and trs[k][0] is not None]
        rng = np.random.default_rng(0)
        if len(pre_idx) > 150:
            pre_idx = list(rng.choice(pre_idx, 150, replace=False))
        pre_idx = sorted(pre_idx, key=lambda k: tab.onset.iat[k])
        if pre_idx:
            img = np.vstack([trs[k][0] for k in pre_idx])
            ax = axs[r_, 3]
            ax.imshow(np.clip(img, -5, 30), aspect="auto", cmap="GnBu", interpolation="none",
                      extent=(T0, T0 + img.shape[1] / 200.0, len(pre_idx), 0))
            for j, k in enumerate(pre_idx):
                ev = trs[k][1]
                ax.plot(ev, np.full(len(ev), j + 0.5), ".", color="k", ms=1.2)
            ax.plot(tab.onset.to_numpy()[pre_idx], np.arange(len(pre_idx)) + 0.5, "|", color="mediumvioletred", ms=3)
            ax.plot(tab.lick.to_numpy()[pre_idx], np.arange(len(pre_idx)) + 0.5, "|", color="#e7298a", ms=3)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlim(-1, 1.2)
            ax.set_xlabel("Time from stimulus (s)")
            ax.set_title(f"onset < 0 trials ({len(pre_idx)} shown): purple onset, pink first lick, black piezo",
                         fontsize=7)
    fig.suptitle("Jaw onsets before the stimulus (new pkls, lick trials, active, perf != 6)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"007_prestim_onsets.{ext}", dpi=150)


if __name__ == "__main__":
    main()
