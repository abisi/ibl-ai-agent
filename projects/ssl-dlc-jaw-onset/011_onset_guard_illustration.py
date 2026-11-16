"""Option 2 illustrated (user, 2026-11-12): the prominent-peak rule (009, f = 0.5) with one guard: the peak is NOT
switched when the part it would skip is a separate opening of >= 10 baseline SD (after its peak the jaw closes at least
halfway before the new onset, 010). Then the current onset is kept.

Per trial (lick trials, active, perf != 6), same traces and search window as 009. Rules compared: current, f = 0.5
(option 1), f = 0.5 + guard (option 2).
Figure 011_onset_guard_illustration.png:
  row 1: MH031_20250507 t185 and 5 random hits whose onset still changes with the guard (was before the stimulus)
  row 2: 6 random hits whose onset still changes with the guard (was after the stimulus)
  row 3: 6 random hits where the guard keeps the current onset (option 1 would have moved it)
  purple = current onset, green = option 1, orange dashed = option 2 (= purple in row 3), pink = first lick; grey shading
  = the skipped part between the current and the option-1 onset.
Tables: tables/011_onset_guard.csv (per trial), tables/011_onset_guard_summary.csv
Run on haas after 009: ~/code/ibl-ai-agent/.venv/bin/python 011_onset_guard_illustration.py
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

spec = importlib.util.spec_from_file_location("v009", Path(__file__).with_name("009_onset_peak_rule_variants.py"))
v009 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v009)

R, OUT = v009.R, v009.OUT
T0, F, GUARD_SD = -1.0, 0.5, 10.0
TYPES = {"whisker_trial": "Whisker hit", "auditory_trial": "Auditory hit", "no_stim_trial": "False alarm"}


def guarded(w, thr, t0, start, stop, f=F):
    """(current onset index, option-1 index, option-2 index)."""
    cur = v009.current(w, thr, t0, start, stop)
    if cur is None:
        return None, None, None
    b0 = cur[0]
    b1 = v009.prominent(w, thr, t0, start, stop, f)
    if b1 is None or b1 <= b0:
        return b0, b1, b1
    seg = w[b0:b1 + 1]
    pk = b0 + int(np.argmax(seg[:-1])) if b1 > b0 + 1 else b0
    sd = thr / 5.0
    separate = pk < b1 and np.min(w[pk:b1 + 1]) <= 0.5 * w[pk] and w[pk] >= GUARD_SD * sd
    return b0, b1, (b0 if separate else b1)


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
        start = max(0, t0 - int(round(v009.PRE_WIN * fs)))
        b0, b1, b2 = guarded(w, r.threshold, t0, start, stop)
        cv = lambda b: (b - t0) / fs if b is not None else np.nan  # noqa: E731
        o0 = r.jaw_dlc_onset  # pkl onset (= current rule; fallback = peak time when the rule finds none)
        rows.append(dict(mouse_id=r.mouse_id, session_id=r.session_id, trial_id=r.trial_id, trial_type=r.trial_type,
                         lick=lick, onset_current=o0,
                         onset_opt1=cv(b1) if b1 is not None else o0, onset_opt2=cv(b2) if b2 is not None else o0))
        keep.append(tr / (r.threshold / 5.0) if round(fs) == 200 else None)
    return rows, keep


def main():
    pkls = sorted(R.glob("*/dlc_jaw_onset_times.pkl"))
    out = Parallel(n_jobs=48)(delayed(one_mouse)(p) for p in pkls)
    tab = pd.DataFrame([r for rows, _ in out for r in rows])
    trs = [k for _, keep in out for k in keep]
    tab.to_csv(OUT / "tables" / "011_onset_guard.csv", index=False)
    summ = []
    for tt in TYPES:
        t = tab[tab.trial_type == tt]
        for col, name in (("onset_current", "current"), ("onset_opt1", "option 1 (f 0.5)"),
                          ("onset_opt2", "option 2 (f 0.5 + guard)")):
            lag = t.lick - t[col]
            summ.append(dict(trial_type=tt, rule=name, pct_onset_before_stim=100 * np.mean(t[col] < 0),
                             pct_changed_10ms=100 * np.mean((t[col] - t.onset_current).abs() > 0.01),
                             pct_changed_pre=100 * np.mean((t.onset_current < 0) & ((t[col] - t.onset_current).abs() > 0.01)),
                             pct_changed_post=100 * np.mean((t.onset_current >= 0) & ((t[col] - t.onset_current).abs() > 0.01)),
                             median_lag_ms=1000 * lag.median(), iqr_lag_ms=1000 * (lag.quantile(.75) - lag.quantile(.25))))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "tables" / "011_onset_guard_summary.csv", index=False)
    print(summ.round(1).to_string())

    hits = tab.trial_type.isin(["whisker_trial", "auditory_trial"]).to_numpy()
    ok = np.array([x is not None for x in trs])
    ch2 = (tab.onset_opt2 - tab.onset_current).abs().to_numpy() > 0.01
    ch1 = (tab.onset_opt1 - tab.onset_current).abs().to_numpy() > 0.01
    pre = tab.onset_current.to_numpy() < 0
    rng = np.random.default_rng(5)
    ref = tab.index[(tab.session_id.str.startswith("MH031_20250507")) & (tab.trial_id == 185)].to_list()
    groups = [("still changed, was before the stimulus", ref + list(rng.choice(np.flatnonzero(hits & ok & ch2 & pre), 5, replace=False))),
              ("still changed, was after the stimulus", list(rng.choice(np.flatnonzero(hits & ok & ch2 & ~pre), 6, replace=False))),
              ("guard keeps the current onset", list(rng.choice(np.flatnonzero(hits & ok & ch1 & ~ch2), 6, replace=False)))]
    fig, axs = plt.subplots(3, 6, figsize=(22, 10.5), sharex=True)
    for r_, (label, ks) in enumerate(groups):
        for c, k in enumerate(ks):
            ax = axs[r_, c]
            x = trs[k]
            tt = T0 + np.arange(len(x)) / 200.0
            a, b = sorted([tab.onset_current[k], tab.onset_opt1[k]])
            ax.axvspan(a, b, color="0.85", lw=0)
            ax.plot(tt, x, color="k", lw=0.8)
            ax.axhline(5, color="r", lw=0.5, ls=":")
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.axvline(tab.onset_current[k], color="purple", lw=1.4, label="current")
            ax.axvline(tab.onset_opt1[k], color="g", lw=1.2, label="option 1")
            ax.axvline(tab.onset_opt2[k], color="tab:orange", lw=1.6, ls=(0, (3, 2)), label="option 2")
            ax.axvline(tab.lick[k], color="#e7298a", lw=1, label="first lick")
            ax.set_xlim(-0.4, 0.7)
            ax.set_title(f"{TYPES[tab.trial_type[k]]} {tab.session_id[k][:14]} t{tab.trial_id[k]}", fontsize=7.5)
        axs[r_, 0].set_ylabel(f"{label}\njaw distance (baseline SD)", fontsize=8)
    axs[0, 0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("Time from stimulus (s)")
    w = summ[summ.trial_type == "whisker_trial"].set_index("rule")
    fig.suptitle("Option 2 = prominent-peak rule (f = 0.5) + guard: keep the current onset when the skipped part (grey) is a "
                 f"separate opening >= {GUARD_SD:.0f} SD that closes at least halfway. Whisker hits changed: option 1 "
                 f"{w.loc['option 1 (f 0.5)', 'pct_changed_10ms']:.1f}%, option 2 {w.loc['option 2 (f 0.5 + guard)', 'pct_changed_10ms']:.1f}%",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"011_onset_guard_illustration.{ext}", dpi=120)


if __name__ == "__main__":
    main()
