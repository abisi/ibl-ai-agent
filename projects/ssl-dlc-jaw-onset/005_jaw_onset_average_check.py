"""Does the jaw-onset detection look right on average? (user, 2026-11-12, after the new-DLC rerun of the per-mouse pkls)

Same trials as the user's plot_single_mouse_dlc_jaw_onset (beh_plotting_functions): lick trials, active context,
perf != 6, per trial type (whisker hit, auditory hit, false alarm). Old = dlc_jaw_onset_times_backup_2026-11-10.pkl
(dlc_helpers onsets on the old DLC), new = dlc_jaw_onset_times.pkl (new DLC, offset-corrected lick, 2026-11-12).
jaw_dlc_pos is the baseline-subtracted jaw distance; position traces are divided by the session's pooled baseline SD
(threshold / 5), so the detection threshold is 5 on the y axis. Rate of change = d(jaw distance)/dt from the same
traces (mm/s, positive = opening). Lick = first lick from the stimulus, offset-corrected (lick_time -
response_window_start_time; memory ssl-nwb-lick-time-offset).
  row 1: stimulus-aligned jaw distance, mean over mice (mouse = mean over its trials), old vs new
  row 2: stimulus-aligned rate of change of the jaw distance
  row 3: onset and first-lick time distributions (pooled trials)
  row 4: onset-aligned jaw distance (a good onset: flat before 0, sharp rise after)
  row 5: onset-aligned rate of change
  row 6: new onsets, up to 300 random trials pooled over mice, stimulus-aligned, sorted by onset (onset, lick ticks)
movement_onset_times falls back to the peak time, so every trial has an onset (no detection rate).
Output: combined_results_ks4/ssl-dlc-jaw-onset/figures/005_jaw_onset_average_check.png (+ .pdf),
        tables/005_jaw_onset_average_check.csv (per mouse)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 005_jaw_onset_average_check.py
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
T0 = -1.0                      # time_start of the traces
FS = 200.0
PRE, POST = 0.3, 0.5           # onset-aligned window (s)
COLS = ["mouse_id", "session_id", "trial_type", "lick_flag", "context", "perf", "jaw_dlc_pos", "jaw_dlc_onset",
        "lick_time", "start_time", "response_window_start_time", "threshold", "freq"]


def load_mouse(mouse):
    out = {}
    for tag, f in (("old", R / mouse / "dlc_jaw_onset_times_backup_2026-11-10.pkl"),
                   ("new", R / mouse / "dlc_jaw_onset_times.pkl")):
        if not f.exists():
            return None
        d = pd.read_pickle(f)
        d = d[[c for c in COLS if c in d.columns]]
        d = d[(d.context == "active") & (d.perf != 6) & (d.lick_flag == 1)].copy()
        d = d[d.freq.round() == FS]  # 100 Hz sessions left out of the averages (other grid)
        d["sd"] = d.threshold / 5.0
        d["lick"] = d.lick_time - d.response_window_start_time  # first lick from stimulus, offset-corrected
        d["pos"] = [np.asarray(t, float) / s for t, s in zip(d.jaw_dlc_pos, d.sd)]
        d["vel"] = [np.gradient(np.asarray(t, float)) * FS for t in d.jaw_dlc_pos]
        out[tag] = d.reset_index(drop=True)
    return out


def onset_locked(traces, onsets):
    a, b = int(PRE * FS), int(POST * FS)
    rows = []
    for tr, on in zip(traces, onsets):
        if not np.isfinite(on):
            continue
        i = int(round((on - T0) * FS))
        if i - a >= 0 and i + b <= len(tr):
            rows.append(tr[i - a:i + b])
    return np.nanmean(rows, 0) if rows else np.full(a + b, np.nan)


def summarise(mouse):
    dd = load_mouse(mouse)
    if dd is None:
        return None
    res = {}
    for tag, d in dd.items():
        for tt in TYPES:
            s = d[d.trial_type == tt]
            if len(s) < 5:
                continue
            res[(tag, tt)] = dict(stim_pos=np.nanmean(np.vstack(s.pos), 0), stim_vel=np.nanmean(np.vstack(s.vel), 0),
                                  on_pos=onset_locked(s.pos, s.jaw_dlc_onset), on_vel=onset_locked(s.vel, s.jaw_dlc_onset),
                                  onsets=s.jaw_dlc_onset.to_numpy(), licks=s.lick.to_numpy(), n=len(s))
    rng = np.random.default_rng(abs(hash(mouse)) % 2**32)
    new = dd["new"]
    take = new.sample(min(len(new), 8), random_state=rng.integers(1e9))
    res["sample"] = list(zip(take.pos, take.jaw_dlc_onset, take.lick, take.trial_type))
    return mouse, res


def band(ax, x, arr, ls, col, label):
    mu, se = np.nanmean(arr, 0), np.nanstd(arr, 0) / np.sqrt(np.isfinite(arr).sum(0))
    ax.plot(x, mu, ls=ls, color=col, lw=1.4, label=label)
    ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.2, lw=0)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    mice = sorted(p.parent.name for p in R.glob("*/dlc_jaw_onset_times_backup_2026-11-10.pkl"))
    res = [r for r in Parallel(n_jobs=48)(delayed(summarise)(m) for m in mice) if r is not None]
    print(len(res), "mice")
    fig, axs = plt.subplots(6, 3, figsize=(12, 19), gridspec_kw=dict(height_ratios=[1, 1, 0.7, 1, 1, 1.4]))
    rows = []
    x_on = np.arange(-int(PRE * FS), int(POST * FS)) / FS
    for c, (tt, (name, col)) in enumerate(TYPES.items()):
        for tag, ls, lc in (("old", "--", "0.55"), ("new", "-", col)):
            have = [r[(tag, tt)] for _, r in res if (tag, tt) in r]
            lab = f"{tag} DLC (n={len(have)} mice)"
            x = T0 + np.arange(len(have[0]["stim_pos"])) / FS
            band(axs[0, c], x, np.array([h["stim_pos"] for h in have]), ls, lc, lab)
            band(axs[1, c], x, np.array([h["stim_vel"] for h in have]), ls, lc, lab)
            band(axs[3, c], x_on, np.array([h["on_pos"] for h in have]), ls, lc, lab)
            band(axs[4, c], x_on, np.array([h["on_vel"] for h in have]), ls, lc, lab)
            ons = np.concatenate([h["onsets"] for h in have])
            bins = np.arange(-0.5, 1.2, 0.02)
            axs[2, c].hist(ons[np.isfinite(ons)], bins=bins, histtype="step", color=lc, ls=ls, density=True,
                           label=f"onset {tag} ({100 * np.mean(ons[np.isfinite(ons)] < 0):.1f}% < 0)")
            if tag == "new":
                lks = np.concatenate([h["licks"] for h in have])
                axs[2, c].hist(lks[np.isfinite(lks)], bins=bins, histtype="stepfilled", color="#e7298a", alpha=0.25,
                               density=True, label="first lick")
        for m, r in res:
            for tag in ("old", "new"):
                if (tag, tt) in r:
                    o, lk = r[(tag, tt)]["onsets"], r[(tag, tt)]["licks"]
                    lag = lk - o
                    ok = np.isfinite(o)
                    rows.append(dict(mouse_id=m, trial_type=tt, dlc=tag, n_trials=r[(tag, tt)]["n"],
                                     median_onset_ms=1000 * np.nanmedian(o), median_lick_ms=1000 * np.nanmedian(lk),
                                     median_lag_ms=1000 * np.nanmedian(lag),
                                     iqr_lag_ms=1000 * np.subtract(*np.nanpercentile(lag, [75, 25]))
                                     if np.isfinite(lag).any() else np.nan,
                                     pct_onset_before_stim=100 * np.mean(o[ok] < 0) if ok.any() else np.nan))
        axs[0, c].set_title(name)
        for r_, lab in ((0, "Time from stimulus (s)"), (1, "Time from stimulus (s)"), (2, "Time from stimulus (s)"),
                        (3, "Time from detected jaw onset (s)"), (4, "Time from detected jaw onset (s)")):
            axs[r_, c].axvline(0, color="k", lw=0.5)
            axs[r_, c].set_xlabel(lab)
            axs[r_, c].spines[["top", "right"]].set_visible(False)
            axs[r_, c].legend(fontsize=6, frameon=False)
        for r_ in (0, 1):
            axs[r_, c].set_xlim(-1, 1.5)
        for r_ in (0, 3):
            axs[r_, c].axhline(5, color="r", lw=0.6, ls=":")
        for r_ in (1, 4):
            axs[r_, c].axhline(0, color="0.7", lw=0.5)
        smp = sorted([s for _, r in res for s in r["sample"] if s[3] == tt and np.isfinite(s[1])],
                     key=lambda s: s[1])[:300]
        if smp:
            img = np.vstack([s[0] for s in smp])
            axs[5, c].imshow(np.clip(img, -5, 30), aspect="auto", cmap="GnBu", interpolation="none",
                             extent=(T0, T0 + img.shape[1] / FS, len(smp), 0))
            axs[5, c].plot([s[1] for s in smp], np.arange(len(smp)) + 0.5, "|", color="mediumvioletred", ms=2)
            axs[5, c].plot([s[2] for s in smp], np.arange(len(smp)) + 0.5, "|", color="#e7298a", ms=2, alpha=0.6)
            axs[5, c].axvline(0, color="k", lw=0.5)
            axs[5, c].set_xlim(-0.5, 1.2)
            axs[5, c].set_xlabel("Time from stimulus (s)")
            axs[5, c].set_title(f"new DLC, {len(smp)} trials (purple = onset, pink = first lick)", fontsize=8)
    axs[0, 0].set_ylabel("Jaw distance (baseline SD)")
    axs[1, 0].set_ylabel("d jaw distance / dt (mm/s)")
    axs[2, 0].set_ylabel("Density")
    axs[3, 0].set_ylabel("Jaw distance (baseline SD)")
    axs[4, 0].set_ylabel("d jaw distance / dt (mm/s)")
    axs[5, 0].set_ylabel("Trials (sorted by onset)")
    fig.suptitle("Jaw onset (lick trials, active, perf != 6; red dotted = threshold 5 SD; first lick offset-corrected; "
                 "every trial has an onset: fallback = peak time)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"005_jaw_onset_average_check.{ext}", dpi=150)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tables" / "005_jaw_onset_average_check.csv", index=False)
    print(tab.groupby(["trial_type", "dlc"])[["median_onset_ms", "median_lick_ms", "median_lag_ms", "iqr_lag_ms",
                                              "pct_onset_before_stim"]].median().round(1).to_string())


if __name__ == "__main__":
    main()
