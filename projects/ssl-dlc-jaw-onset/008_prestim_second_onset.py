"""Trials whose jaw onset is before the stimulus (007): is there a second, stimulus-triggered opening after the first
one (first = preparation, second = reaction)? (user, 2026-11-12: "I would need to see this")

Per trial (new pkls, lick trials, active, perf != 6, 200 Hz sessions): jaw opening velocity = d(jaw distance)/dt (mm/s)
of the pkl trace after a 15 ms boxcar; velocity threshold = 5 x the session's median baseline SD of the velocity
(-1 to -0.1 s, as the position threshold). Opening bouts = runs of velocity above threshold. For onset < 0 trials:
second opening = the first bout STARTING at or after the stimulus (the velocity went back below threshold since the
first opening), before the first lick (offset-corrected). For onset >= 0 trials: the first bout starting at or after the
stimulus (its velocity onset), as reference latencies for a stimulus-triggered opening.
Figure 1 (008_prestim_second_onset.png), one row per trial type:
  col 1: stimulus-aligned velocity (mean over mice), onset < 0 vs >= 0 trials
  col 2: latency of the second opening (onset < 0 trials, % with one) vs latency of the velocity onset of onset >= 0
         trials
  col 3: onset < 0 trials (up to 120), velocity, sorted by second-opening latency: first onset (purple), second
         opening (green), first lick (pink)
Figure 2 (008_prestim_second_onset_examples.png): 8 random onset < 0 trials per type, jaw distance (black, baseline
SD) and velocity (orange, mm/s) with the same marks.
Output: combined_results_ks4/ssl-dlc-jaw-onset/figures/, tables/008_prestim_second_onset.csv (per trial)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 008_prestim_second_onset.py
"""
from pathlib import Path

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
FS, T0 = 200.0, -1.0
I0 = int(-T0 * FS)
BASE = slice(0, int(0.9 * FS))
SMOOTH = 3  # samples (15 ms boxcar)


def velocity(tr):
    x = pd.Series(tr).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
    return np.gradient(x) * FS


def bouts(v, thr):
    m = np.r_[False, v > thr, False]
    e = np.flatnonzero(np.diff(m.astype(int)))
    return e[::2], e[1::2]


def one_mouse(pkl):
    d = pd.read_pickle(pkl)
    d = d[(d.context == "active") & (d.perf != 6) & (d.lick_flag == 1) & (d.freq.round() == FS)]
    rows, keep = [], []
    for sid, s in d.groupby("session_id"):
        vel = [velocity(np.asarray(t, float)) for t in s.jaw_dlc_pos]
        vthr = 5 * np.nanmedian([np.nanstd(v[BASE]) for v in vel])
        for r, v in zip(s.itertuples(), vel):
            lick = r.lick_time - r.response_window_start_time
            stop = I0 + int(round(lick * FS)) if np.isfinite(lick) else len(v)
            st, en = bouts(np.nan_to_num(v, nan=-np.inf), vthr)
            after = [a for a in st if I0 <= a < stop]
            t2 = (after[0] - I0) / FS if after else np.nan
            rows.append(dict(mouse_id=r.mouse_id, session_id=sid, trial_id=r.trial_id, trial_type=r.trial_type,
                             onset=r.jaw_dlc_onset, lick=lick, second_opening=t2, vthr=vthr,
                             pre=bool(r.jaw_dlc_onset < 0)))
            keep.append((np.asarray(r.jaw_dlc_pos, float) / (r.threshold / 5.0), v))
    return rows, keep


def main():
    pkls = sorted(R.glob("*/dlc_jaw_onset_times.pkl"))
    out = Parallel(n_jobs=48)(delayed(one_mouse)(p) for p in pkls)
    tab = pd.DataFrame([r for rows, _ in out for r in rows])
    trs = [k for _, keep in out for k in keep]
    tab.to_csv(OUT / "tables" / "008_prestim_second_onset.csv", index=False)
    x = T0 + np.arange(len(trs[0][1])) / FS
    summ = []
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    rng = np.random.default_rng(0)
    examples = {}
    for r_, (tt, (name, col)) in enumerate(TYPES.items()):
        idx = np.flatnonzero(tab.trial_type.to_numpy() == tt)
        pre = idx[tab.pre.to_numpy()[idx]]
        post = idx[~tab.pre.to_numpy()[idx]]
        for sel, ls, lab in ((pre, "-", "onset < 0"), (post, "--", "onset >= 0")):
            per = pd.Series(sel).groupby(tab.mouse_id.to_numpy()[sel]).apply(list)
            arr = np.array([np.nanmean(np.vstack([trs[k][1] for k in ks]), 0) for ks in per])
            mu, se = np.nanmean(arr, 0), np.nanstd(arr, 0) / np.sqrt(len(arr))
            axs[r_, 0].plot(x, mu, ls=ls, color=col, label=f"{lab} ({len(sel)} trials, {len(arr)} mice)")
            axs[r_, 0].fill_between(x, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        axs[r_, 0].axvline(0, color="k", lw=0.5)
        axs[r_, 0].axhline(0, color="0.7", lw=0.5)
        axs[r_, 0].set_xlim(-0.5, 0.8)
        axs[r_, 0].set_title(name)
        axs[r_, 0].set_xlabel("Time from stimulus (s)")
        axs[r_, 0].set_ylabel("Jaw opening velocity (mm/s)")
        axs[r_, 0].legend(fontsize=7, frameon=False)
        t2 = tab.second_opening.to_numpy()
        has2 = np.isfinite(t2[pre])
        ref = t2[post][np.isfinite(t2[post])]
        bins = np.arange(0, 0.8, 0.01)
        axs[r_, 1].hist(ref, bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.3,
                        label=f"onset >= 0: velocity onset (median {1000 * np.median(ref):.0f} ms)")
        axs[r_, 1].hist(t2[pre][has2], bins=bins, density=True, histtype="step", color="g", lw=1.2,
                        label=f"onset < 0: second opening in {100 * has2.mean():.0f}% "
                              f"(median {1000 * np.nanmedian(t2[pre]):.0f} ms)")
        axs[r_, 1].set_xlabel("Latency from stimulus (s)")
        axs[r_, 1].legend(fontsize=7, frameon=False)
        summ.append(dict(trial_type=tt, n_pre=len(pre), pct_pre_with_second_opening=100 * has2.mean(),
                         median_second_opening_ms=1000 * np.nanmedian(t2[pre]),
                         median_velocity_onset_post_trials_ms=1000 * np.median(ref),
                         pct_post_with_velocity_onset=100 * np.isfinite(t2[post]).mean()))
        order = pre if len(pre) <= 120 else rng.choice(pre, 120, replace=False)
        order = sorted(order, key=lambda k: (np.isnan(t2[k]), t2[k]))
        img = np.vstack([trs[k][1] for k in order])
        ax = axs[r_, 2]
        lim = np.nanpercentile(np.abs(img), 98)
        ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="none",
                  extent=(x[0], x[-1] + 1 / FS, len(order), 0))
        yy = np.arange(len(order)) + 0.5
        ax.plot(tab.onset.to_numpy()[order], yy, "|", color="purple", ms=3)
        ax.plot(t2[order], yy, "|", color="g", ms=3, mew=1.5)
        ax.plot(tab.lick.to_numpy()[order], yy, "|", color="#e7298a", ms=3)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-0.5, 0.8)
        ax.set_xlabel("Time from stimulus (s)")
        ax.set_title("onset < 0 trials, velocity (red = opening): purple onset, green 2nd opening, pink lick",
                     fontsize=7)
        examples[tt] = rng.choice(pre, min(8, len(pre)), replace=False)
    fig.suptitle("Jaw onsets before the stimulus: is there a second, stimulus-triggered opening?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"008_prestim_second_onset.{ext}", dpi=150)

    fig, axs = plt.subplots(3, 8, figsize=(22, 8), sharex=True)
    for r_, (tt, (name, col)) in enumerate(TYPES.items()):
        for c, k in enumerate(examples[tt]):
            ax = axs[r_, c]
            pos, v = trs[k]
            ax.plot(x, pos, color="k", lw=0.8)
            ax2 = ax.twinx()
            ax2.plot(x, v, color="tab:orange", lw=0.7)
            ax2.axhline(tab.vthr.iat[k], color="tab:orange", lw=0.5, ls=":")
            for t, cc in ((tab.onset.iat[k], "purple"), (tab.second_opening.iat[k], "g"), (tab.lick.iat[k], "#e7298a")):
                if np.isfinite(t):
                    ax.axvline(t, color=cc, lw=1)
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlim(-0.5, 0.6)
            ax.set_title(f"{tab.session_id.iat[k][:14]} t{tab.trial_id.iat[k]}", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{name}\njaw distance (SD)", fontsize=8)
        axs[r_, 0].text(-0.5, 1.15, "", transform=axs[r_, 0].transAxes)
    for ax in axs[-1]:
        ax.set_xlabel("Time from stimulus (s)")
    fig.suptitle("onset < 0 examples: black jaw distance, orange velocity (dotted = threshold); purple onset, "
                 "green 2nd opening, pink first lick", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"008_prestim_second_onset_examples.{ext}", dpi=130)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "tables" / "008_prestim_second_onset_summary.csv", index=False)
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
