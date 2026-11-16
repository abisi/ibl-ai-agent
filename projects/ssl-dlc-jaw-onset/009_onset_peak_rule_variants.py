"""Jaw onset: catch stimulus-triggered openings that follow a small pre-stimulus drift (user, 2026-11-12: most onset < 0
detections are true; MH031_20250507 t185 is one where the stimulus-triggered opening should be the onset; "how to get
more of this with minimal effort?").

Current rule (dlc_helpers.movement_onset_times): first threshold-crossing bout whose FIRST local peak is after the
stimulus, walked back to 30 % of that peak (never before the bout's threshold crossing). A slow drift that crosses the
threshold before the stimulus and has a small wiggle just after it gives a pre-stimulus onset.
Variant "prominent peak (f)": the current rule, except when the peak it picks is small (< f x the maximum of the trace
between the stimulus and the lick): the peak then becomes the first later local maximum reaching f x that maximum,
walked back to 30 % of it (never before the threshold crossing of its bout). A full opening just before the stimulus
(its peak as large as the response) keeps its onset; a small drift with a wiggle after the stimulus does not.
f = 0.3, 0.5, 0.7.
Applied to the pkl traces (jaw_dlc_pos, threshold, same search window: -0.5 s to the offset-corrected first lick); only
the peak choice differs, so trials with the same peak keep the same onset.
Outputs (combined_results_ks4/ssl-dlc-jaw-onset/): tables/009_onset_variants.csv (per trial),
tables/009_onset_variants_summary.csv, figures/009_onset_variants_examples.png (trials whose onset moves from before to
after the stimulus with f = 0.5, plus MH031_20250507 t185), figures/009_onset_variants_examples_poststim.png (hits
whose onset was already after the stimulus and moves > 10 ms with f = 0.5)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 009_onset_peak_rule_variants.py
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
T0, PRE_WIN, FRAC = -1.0, 0.5, 0.3
FS_LIST = (0.3, 0.5, 0.7)
TYPES = {"whisker_trial": "Whisker hit", "auditory_trial": "Auditory hit", "no_stim_trial": "False alarm"}


def current(w, thr, t0, start, stop):
    cross = np.flatnonzero(w[start:stop] > thr) + start
    i = 0
    while i < len(cross):
        c = cross[i]
        end = c
        while end < stop and w[end] > thr:
            end += 1
        p = c
        while p + 1 < end and w[p + 1] >= w[p]:
            p += 1
        if p >= t0:
            b = p
            while b > c and w[b] > FRAC * w[p]:
                b -= 1
            return b, c, p
        i = np.searchsorted(cross, end)
    return None


def prominent(w, thr, t0, start, stop, f):
    """Current rule; if its peak is small (< f x the trace maximum between the stimulus and the lick), the peak becomes
    the first local maximum after it that reaches f x that maximum, walked back to 30 % of it (never before the
    threshold crossing of its bout). Otherwise the current onset."""
    cur = current(w, thr, t0, start, stop)
    if cur is None or stop <= t0 + 1:
        return None if cur is None else cur[0]
    b0, c, p = cur
    top = np.max(w[t0:stop])
    if w[p] >= f * top:
        return b0
    P = None
    for k in range(p + 1, stop):
        if w[k] >= f * top and w[k] >= w[k - 1] and (k + 1 >= len(w) or w[k] >= w[k + 1]):
            P = k
            break
    if P is None:
        return b0
    b = P
    while b > start and w[b] > FRAC * w[P] and w[b - 1] > thr:
        b -= 1
    return b


def one_mouse(pkl):
    d = pd.read_pickle(pkl)
    d = d[(d.context == "active") & (d.perf != 6) & (d.lick_flag == 1)]
    rows, keep = [], []
    for r in d.itertuples():
        fs = float(r.freq)
        tr = np.asarray(r.jaw_dlc_pos, float)
        w = np.nan_to_num(tr, nan=-np.inf)
        t0 = int(round(-T0 * fs))
        rw = r.response_window_start_time - r.start_time
        stop = min(len(w), t0 + int(round((2.0 + rw) * fs)))
        lick = r.lick_time - r.response_window_start_time
        if np.isfinite(lick):
            stop = min(stop, t0 + int(round(lick * fs)))
        start = max(0, t0 - int(round(PRE_WIN * fs)))
        row = dict(mouse_id=r.mouse_id, session_id=r.session_id, trial_id=r.trial_id, trial_type=r.trial_type,
                   onset_pkl=r.jaw_dlc_onset, lick=lick)
        cur = current(w, r.threshold, t0, start, stop)
        row["onset_current"] = (cur[0] - t0) / fs if cur is not None else np.nan
        for f in FS_LIST:
            b = prominent(w, r.threshold, t0, start, stop, f)
            row[f"onset_prom{f}"] = (b - t0) / fs if b is not None else np.nan
        rows.append(row)
        keep.append(tr / (r.threshold / 5.0) if round(fs) == 200 else None)
    return rows, keep


def main():
    pkls = sorted(R.glob("*/dlc_jaw_onset_times.pkl"))
    out = Parallel(n_jobs=48)(delayed(one_mouse)(p) for p in pkls)
    tab = pd.DataFrame([r for rows, _ in out for r in rows])
    trs = [k for _, keep in out for k in keep]
    tab.to_csv(OUT / "tables" / "009_onset_variants.csv", index=False)
    print("current rule reproduces the pkl onsets (fallback = peak time excluded):",
          np.round(np.nanmean(np.isclose(tab.onset_current, tab.onset_pkl, atol=1e-3)
                              | tab.onset_current.isna()), 4))
    summ = []
    for tt in TYPES:
        t = tab[tab.trial_type == tt]
        for col in ["onset_pkl"] + [f"onset_prom{f}" for f in FS_LIST]:
            o = t[col].where(t[col].notna(), t.onset_pkl)  # no prominent peak -> keep the pkl onset (fallback)
            lag = t.lick - o
            summ.append(dict(trial_type=tt, rule=col, pct_onset_before_stim=100 * np.mean(o < 0),
                             pct_changed_10ms=100 * np.mean(np.abs(o - t.onset_pkl) > 0.01),
                             median_onset_ms=1000 * o.median(), median_lag_ms=1000 * lag.median(),
                             iqr_lag_ms=1000 * (lag.quantile(.75) - lag.quantile(.25))))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "tables" / "009_onset_variants_summary.csv", index=False)
    print(summ.round(1).to_string())

    x = None
    moved = tab.index[(tab.onset_pkl < 0) & (tab["onset_prom0.5"] >= 0)
                      & tab.trial_type.isin(["whisker_trial", "auditory_trial"])].to_numpy()
    moved = [k for k in moved if trs[k] is not None]
    rng = np.random.default_rng(1)
    pick = list(rng.choice(moved, min(15, len(moved)), replace=False))
    ref = tab.index[(tab.session_id.str.startswith("MH031_20250507")) & (tab.trial_id == 185)].to_list()
    pick = ref + pick
    fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for ax, k in zip(axs.flat, pick):
        tr = trs[k]
        x = T0 + np.arange(len(tr)) / 200.0
        ax.plot(x, tr, color="k", lw=0.8)
        ax.axhline(5, color="r", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axvline(tab.onset_pkl[k], color="purple", lw=1.2, label="current")
        ax.axvline(tab["onset_prom0.5"][k], color="g", lw=1.2, label="prominent peak 0.5")
        ax.axvline(tab.lick[k], color="#e7298a", lw=1, label="first lick")
        ax.set_xlim(-0.5, 0.6)
        ax.set_title(f"{TYPES[tab.trial_type[k]]} {tab.session_id[k][:14]} t{tab.trial_id[k]}", fontsize=8)
    axs.flat[0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("Time from stimulus (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Jaw distance (baseline SD)")
    n_pre = int(((tab.onset_pkl < 0) & tab.trial_type.isin(["whisker_trial", "auditory_trial"])).sum())
    fig.suptitle(f"Hits whose onset moves after the stimulus with the prominent-peak rule (f = 0.5): {len(moved)} of "
                 f"{n_pre} pre-stimulus onsets; first panel MH031_20250507 t185, then 15 random", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"009_onset_variants_examples.{ext}", dpi=130)

    # Hits whose onset was already after the stimulus and still changes (> 10 ms) with f = 0.5 (f = 0.3 shown too)
    hits = tab.trial_type.isin(["whisker_trial", "auditory_trial"])
    post = tab.index[hits & (tab.onset_pkl >= 0) & ((tab["onset_prom0.5"] - tab.onset_pkl).abs() > 0.01)].to_numpy()
    post = [k for k in post if trs[k] is not None]
    n_post = int((hits & (tab.onset_pkl >= 0)).sum())
    shift = (tab["onset_prom0.5"] - tab.onset_pkl)[post] * 1000
    print(f"post-stimulus hit onsets changed > 10 ms (f = 0.5): {len(post)} of {n_post}; shift ms median "
          f"{shift.median():.0f}, IQR {shift.quantile(.25):.0f}-{shift.quantile(.75):.0f}; changed with f = 0.3 too: "
          f"{100 * np.mean((tab['onset_prom0.3'] - tab.onset_pkl).abs()[post] > 0.01):.0f}%")
    pick = list(rng.choice(post, min(24, len(post)), replace=False))
    fig, axs = plt.subplots(4, 6, figsize=(22, 12), sharex=True)
    for ax, k in zip(axs.flat, pick):
        tr = trs[k]
        x = T0 + np.arange(len(tr)) / 200.0
        ax.plot(x, tr, color="k", lw=0.8)
        ax.axhline(5, color="r", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axvline(tab.onset_pkl[k], color="purple", lw=1.2, label="current")
        ax.axvline(tab["onset_prom0.3"][k], color="tab:orange", lw=1.2, ls=(0, (3, 2)), label="prominent 0.3")
        ax.axvline(tab["onset_prom0.5"][k], color="g", lw=1.2, ls=(0, (1, 1)), label="prominent 0.5")
        ax.axvline(tab.lick[k], color="#e7298a", lw=1, label="first lick")
        ax.set_xlim(-0.3, 0.7)
        ax.set_title(f"{TYPES[tab.trial_type[k]]} {tab.session_id[k][:14]} t{tab.trial_id[k]}", fontsize=8)
    axs.flat[0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("Time from stimulus (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Jaw distance (baseline SD)")
    fig.suptitle(f"Hits with a post-stimulus onset that changes with the prominent-peak rule (f = 0.5): {len(post)} of "
                 f"{n_post}; shift median {shift.median():.0f} ms; 24 random", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"009_onset_variants_examples_poststim.{ext}", dpi=120)


if __name__ == "__main__":
    main()
