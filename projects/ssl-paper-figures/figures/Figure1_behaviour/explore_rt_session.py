"""Exploration (user 2026-11-09): reaction times across normalised session time, next to the movement variables of
explore_movement_session.py. Learners, whisker day 0, hits of whisker and auditory trials (shared behaviour table,
early licks dropped).

Three latencies per hit, in ms from the stimulus (trial start_time):
  - registered reaction time: computed_reaction_time (lick_time - response_window_start_time; NWB offset removed)
  - piezo first lick: first NWB piezo_lick_times event in [0, 1] s (loader of explore_lick_proba)
  - jaw onset: jaw_dlc_onset of behaviour_analysis's per-mouse dlc_jaw_onset_times.pkl (combined_results_ks4/<mouse>),
    matched by session and start time (within 5 ms), kept in [0, 1] s
Per mouse: median per bin of normalised session time (10 bins) and per third; mean +- s.e.m. over mice.
Output: Figures/_explore/rt_session_time_<date>.{pdf,png} and the CSV.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_session.py
"""

import datetime
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, axel_bisi_root  # noqa: E402
from explore_lick_proba import session_licks  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
N_BINS = 10
JAW_DIR = axel_bisi_root() / "combined_results_ks4"
LAT = {"rt_registered_ms": "Registered RT", "rt_piezo_ms": "Piezo first lick", "jaw_onset_ms": "Jaw onset"}


def jaw_table(mouse):
    f = JAW_DIR / mouse / "dlc_jaw_onset_times.pkl"
    if not f.exists():
        return None
    d = pd.read_pickle(f)
    return d[["session_id", "start_time", "jaw_dlc_onset"]]


def one(args):
    sid, st = args
    st = st.sort_values("start_time")
    _, _, licks = session_licks(sid)
    t = st.start_time.to_numpy()
    tn = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    hits = st[(st.lick_flag == 1) & st.trial_type.isin(["whisker_trial", "auditory_trial"])].copy()
    hits["time_norm"] = tn[np.searchsorted(t, hits.start_time.to_numpy())]
    hits["rt_registered_ms"] = hits.computed_reaction_time.where(hits.computed_reaction_time.between(0, 1)) * 1000
    lat = []
    for t0 in hits.start_time.to_numpy():
        i = np.searchsorted(licks, t0) if len(licks) else 0
        lat.append((licks[i] - t0) * 1000 if len(licks) and i < len(licks) and licks[i] <= t0 + 1 else np.nan)
    hits["rt_piezo_ms"] = lat
    jaw = jaw_table(st.mouse_id.iloc[0])
    hits["jaw_onset_ms"] = np.nan
    if jaw is not None:
        j = jaw[jaw.session_id == sid].sort_values("start_time")
        if len(j):
            k = np.clip(np.searchsorted(j.start_time.to_numpy(), hits.start_time.to_numpy()), 0, len(j) - 1)
            ok = np.abs(j.start_time.to_numpy()[k] - hits.start_time.to_numpy()) < 0.005
            v = j.jaw_dlc_onset.to_numpy(float)[k]
            hits["jaw_onset_ms"] = np.where(ok & (v >= 0) & (v <= 1), v * 1000, np.nan)
    return hits[["mouse_id", "session_id", "cohort", "trial_type", "time_norm"] + list(LAT)]


def main():
    from compute import day0_table
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    d0 = d0[d0.early_lick == 0]
    with Pool(16) as pool:
        df = pd.concat(pool.map(one, list(d0.groupby("session_id"))), ignore_index=True)
    df["bin"] = np.minimum((df.time_norm * N_BINS).astype(int), N_BINS - 1)
    df["third"] = np.array(["early", "middle", "late"])[np.minimum((df.time_norm * 3).astype(int), 2)]
    stem = OUT / f"rt_session_time_{datetime.date.today()}"
    df.to_csv(f"{stem}.csv", index=False)
    plot(df, stem)
    print(stem, {k: int(df[k].notna().sum()) for k in LAT}, df.groupby("cohort").mouse_id.nunique().to_dict(),
          "jaw mice", df[df.jaw_onset_ms.notna()].mouse_id.nunique())


def plot(df, stem):
    set_nature_rc()
    A = {"R+": COLORS["auditory"], "R-": "#9a9aee"}
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 95 * MM))
    TH = ["early", "middle", "late"]
    for c, (y, lab) in enumerate(LAT.items()):
        a, b = axs[0, c], axs[1, c]
        for tt, cols, ls in (("whisker_trial", COLORS, "-"), ("auditory_trial", A, "--")):
            for co in ("R+", "R-"):
                d = df[(df.trial_type == tt) & (df.cohort == co)]
                m = d.groupby(["mouse_id", "bin"])[y].median().reset_index()
                g = m.groupby("bin")[y]
                x = (g.mean().index + 0.5) / N_BINS
                a.plot(x, g.mean(), color=cols[co], ls=ls, lw=0.8, label=f"{'W' if tt == 'whisker_trial' else 'A'} {co}")
                a.fill_between(x, g.mean() - g.sem(), g.mean() + g.sem(), color=cols[co], alpha=0.2, lw=0)
                w = d.groupby(["mouse_id", "third"])[y].median().unstack().reindex(columns=TH)
                dx = {("whisker_trial", "R+"): -0.09, ("whisker_trial", "R-"): -0.03, ("auditory_trial", "R+"): 0.03,
                      ("auditory_trial", "R-"): 0.09}[(tt, co)]
                b.errorbar(np.arange(3) + dx, w.mean(), w.sem(), color=cols[co], ls=ls, marker="o", ms=2, lw=0.8, capsize=0)
        a.set_title(lab, fontsize=6.5, pad=2)
        a.set_xlabel("Session time (norm.)", labelpad=1)
        b.set_xticks(range(3), ["Early", "Middle", "Late"])
        b.set_xlim(-0.4, 2.4)
        for ax in (a, b):
            ax.set_ylabel("Latency from stimulus (ms)", labelpad=1)
    axs[0, 0].legend(fontsize=4.5, frameon=False, ncol=2)
    n = df.groupby("cohort").mouse_id.nunique().to_dict()
    fig.suptitle(f"Hits, learners, whisker day 0 (R+ n = {n.get('R+')}, R- n = {n.get('R-')} mice; jaw onset: mice with "
                 "video); per-mouse median, mean ± s.e.m. over mice", fontsize=6)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
