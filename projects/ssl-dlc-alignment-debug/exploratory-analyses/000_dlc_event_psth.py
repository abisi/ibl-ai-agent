"""000 -- Event-aligned DLC traces under start / end / half-rate frame-to-timestamp alignment (see ../question.md).

For sessions whose DLC series and camera timestamps differ in length, pair frames with timestamps under each hypothesis:
  start: frame i <-> timestamp i (user hypothesis: data missing at the end), longer one truncated at its end
  end:   last frame <-> last timestamp, longer one truncated at its start
  half:  frame i <-> timestamp 2i, only when frames / timestamps is 0.45-0.55
then average z-scored DLC traces around trial starts per trial type and around lick-bout onsets, over all trials and per
session third. A hypothesis is supported if the traces are as sharp as in the control sessions and stay sharp to the end.
Sharpness = peak |mean z - baseline| in [0, 0.5] s, baseline = mean over [-0.5, 0] s.

Outputs: combined_results_ks4/ssl-dlc-alignment-debug/{figures/<session>_*.png|pdf, tables/sharpness.csv, cache/<session>.npz}
Run (haas): python 000_dlc_event_psth.py SESSION_ID [SESSION_ID ...]
"""
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path("NWB_ks4")
OUT = axel_bisi_path("combined_results_ks4") / "ssl-dlc-alignment-debug"
GRID = np.round(np.arange(-1.0, 2.0 + 1e-9, 0.01), 3)
BASE, POST = (GRID >= -0.5) & (GRID < 0), (GRID >= 0) & (GRID <= 0.5)
MAX_GAP = 0.05  # s: a grid point further than this from any frame is NaN
KEY_VARS = ["jaw_angle", "jaw_distance", "tongue_likelihood", "tongue_distance", "nose_angle", "pupil_area",
            "whisker_angle", "whisker_velocity", "top_nose_angle"]
EVENTS = ["WH", "WM", "AH", "AM", "FA", "CR", "pW", "pA", "lick"]
HYP_COLORS = {"start": "tab:blue", "end": "tab:orange", "half": "tab:green"}


def keep_series(name):
    return (name.endswith(("_angle", "_distance", "_velocity")) or name in ("pupil_area", "tongue_likelihood")) \
        and name != "lick_trace"


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load_session(session_id):
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        t = f["intervals/trials"]
        trials = pd.DataFrame({k: (dec(t[k][:]) if t[k].dtype.kind in "OS" else t[k][:])
                               for k in ["start_time", "trial_type", "lick_flag", "perf", "context"]})
        licks = f["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:]
        b = f["processing/behavior/BehavioralTimeSeries"]
        series = {k: (b[k]["data"][:].astype(float), b[k]["timestamps"][:] if "timestamps" in b[k] else np.array([]))
                  for k in b if keep_series(k)}
    return trials, licks, series


def event_times(trials, licks):
    tr = trials[~((trials.context != "passive") & (trials.perf == 6))]
    act, pas = tr[tr.context != "passive"], tr[tr.context == "passive"]
    w, a, n = (act.trial_type == "whisker_trial"), (act.trial_type == "auditory_trial"), (act.trial_type == "no_stim_trial")
    lk = act.lick_flag == 1
    ev = {"WH": act[w & lk], "WM": act[w & ~lk], "AH": act[a & lk], "AM": act[a & ~lk], "FA": act[n & lk],
          "CR": act[n & ~lk], "pW": pas[pas.trial_type == "whisker_trial"], "pA": pas[pas.trial_type == "auditory_trial"]}
    ev = {k: np.sort(v.start_time.to_numpy()) for k, v in ev.items()}
    lk = np.sort(licks[np.isfinite(licks)])
    ev["lick"] = lk[np.r_[True, np.diff(lk) > 1.0]]  # lick-bout onsets: no lick in the previous 1 s
    return ev


def pair(data, ts, hyp):
    n_d, n_t = len(data), len(ts)
    if n_t == 0:
        return None
    if hyp == "start":
        n = min(n_d, n_t)
        return data[:n], ts[:n]
    if hyp == "end":
        n = min(n_d, n_t)
        return data[n_d - n:], ts[n_t - n:]
    if hyp == "half" and 0.45 <= n_d / n_t <= 0.55:
        t2 = ts[::2]
        n = min(n_d, len(t2))
        return data[:n], t2[:n]
    return None


def psth(x, t, events):
    """Trials x GRID matrix of nearest-frame values; NaN where no frame within MAX_GAP."""
    q = events[:, None] + GRID[None, :]
    i = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    left, right = t[i - 1], t[i]
    j = np.where(np.abs(q - left) <= np.abs(right - q), i - 1, i)
    out = x[j]
    out[np.abs(t[j] - q) > MAX_GAP] = np.nan
    return out


def sharpness(m):
    if np.all(np.isnan(m)):
        return np.nan
    return np.nanmax(np.abs(m[POST] - np.nanmean(m[BASE])))


def run(session_id):
    trials, licks, series = load_session(session_id)
    ev = event_times(trials, licks)
    rows, cache = [], {}
    for name, (data, ts) in series.items():
        z = (data - np.nanmean(data)) / np.nanstd(data)
        for hyp in ("start", "end", "half"):
            p = pair(z, ts, hyp)
            if p is None:
                continue
            x, t = p
            for e, times in ev.items():
                if len(times) == 0:
                    continue
                mat = psth(x, t, times)
                thirds = np.array_split(np.arange(len(times)), 3)
                means = [np.nanmean(mat, 0)] + [np.nanmean(mat[i], 0) if len(i) else np.full(len(GRID), np.nan)
                                                 for i in thirds]
                for part, m in zip(["all", "third1", "third2", "third3"], means):
                    cache[f"{name}|{hyp}|{e}|{part}"] = m
                    rows.append(dict(session_id=session_id, series=name, hypothesis=hyp, event=e, part=part,
                                     n_events=len(times) if part == "all" else len(thirds[int(part[-1]) - 1]),
                                     n_frames=len(data), n_timestamps=len(ts), sharpness=sharpness(m),
                                     frac_nan=float(np.mean(np.isnan(mat if part == "all" else mat[thirds[int(part[-1]) - 1]])))))
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "cache" / f"{session_id}.npz", grid=GRID, keys=np.array(list(cache)),
                        values=np.array(list(cache.values())))
    df = pd.DataFrame(rows)
    plot(session_id, cache, series, ev)
    return df


def plot(session_id, cache, series, ev):
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vars_ = [v for v in KEY_VARS if v in series]
    lens = {v: f"{len(series[v][0])}/{len(series[v][1])}" for v in vars_}
    # A: hypotheses, all trials
    fig, axs = plt.subplots(len(vars_), len(EVENTS), figsize=(2.0 * len(EVENTS), 1.5 * len(vars_)), sharex=True,
                            squeeze=False)
    for r, v in enumerate(vars_):
        for c, e in enumerate(EVENTS):
            ax = axs[r, c]
            for hyp, col in HYP_COLORS.items():
                m = cache.get(f"{v}|{hyp}|{e}|all")
                if m is not None:
                    ax.plot(GRID, m, color=col, lw=1, label=hyp)
            ax.axvline(0, color="k", lw=0.5, ls=":")
            if r == 0:
                ax.set_title(f"{e} (n={len(ev[e])})", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{v}\nz ({lens[v]})", fontsize=7)
            ax.tick_params(labelsize=6)
    axs[0, 0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("time from event (s)", fontsize=7)
    fig.suptitle(f"{session_id}: DLC traces, frame-to-timestamp alignment hypotheses (frames/timestamps per row)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{session_id}_hypotheses.{ext}", dpi=150)
    plt.close(fig)
    # B: session thirds, per hypothesis
    for hyp in HYP_COLORS:
        if not any(k.split("|")[1] == hyp for k in cache):
            continue
        fig, axs = plt.subplots(len(vars_), len(EVENTS), figsize=(2.0 * len(EVENTS), 1.5 * len(vars_)), sharex=True,
                                squeeze=False)
        for r, v in enumerate(vars_):
            for c, e in enumerate(EVENTS):
                ax = axs[r, c]
                for k, col in zip(["third1", "third2", "third3"], ["#9ecae1", "#4292c6", "#08306b"]):
                    m = cache.get(f"{v}|{hyp}|{e}|{k}")
                    if m is not None:
                        ax.plot(GRID, m, color=col, lw=1, label=k)
                ax.axvline(0, color="k", lw=0.5, ls=":")
                if r == 0:
                    ax.set_title(e, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{v}\nz", fontsize=7)
                ax.tick_params(labelsize=6)
        axs[0, 0].legend(fontsize=6, frameon=False)
        for ax in axs[-1]:
            ax.set_xlabel("time from event (s)", fontsize=7)
        fig.suptitle(f"{session_id}: {hyp} alignment, first / middle / last third of events", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{session_id}_thirds_{hyp}.{ext}", dpi=150)
        plt.close(fig)


def main(sessions):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    tab = OUT / "tables" / "sharpness.csv"
    old = pd.read_csv(tab) if tab.exists() else pd.DataFrame()
    for s in sessions:
        print("session", s, flush=True)
        df = run(s)
        old = pd.concat([old[old.session_id != s] if len(old) else old, df], ignore_index=True)
        old.to_csv(tab, index=False)
    print("wrote", tab)


if __name__ == "__main__":
    main(sys.argv[1:])
