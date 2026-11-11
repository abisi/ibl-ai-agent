"""Exploration (user 2026-11-09): reaction time and jaw onset across the first 30 licked trials of each trial type, whisker
day 0, learners (shared behaviour table: perf != 6, not passive, warm-up block trimmed to its last trial, early licks
dropped).

Hit number k = the k-th trial of that type with a lick in the session (whisker hit, auditory hit, false alarm on a
no-stimulus trial), counted in time order from the first trial kept. Latencies from the stimulus (trial start):
registered reaction time (computed_reaction_time, 0-1 s) and jaw onset (behaviour_analysis dlc_jaw_onset_times.pkl,
0-1 s; mice with video). First 30 hits (user 2026-11-09). Per mouse: median within bins of 5 hits (1-5, ..., 26-30); then mean +- s.e.m. over
mice (dot plot, per-mouse medians shown); bins shown and tested only where >= 8 mice of each cohort contribute (later hits come from fewer,
more active mice), mice per bin printed under the axis. R+ vs R-: cluster-based permutation test over bins (Welch t on the per-mouse medians, clusters of
adjacent bins with p < 0.05, 1,000 permutations of cohort labels; Figure 1 compute.cluster_test); bar = cluster p < 0.05.
Output: Figures/_explore/rt_hits_<date>.{pdf,png}, the per-hit CSV and the cluster table.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_hits.py
"""

import datetime
import json
import sys
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
from common import FIGURES_ROOT  # noqa: E402
from explore_rt_session import jaw_table  # noqa: E402
from paperstyle import MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
N_HITS, BIN, MIN_MICE = 30, 5, 8  # bins shown and tested only with >= MIN_MICE mice per cohort
TT = [("whisker_trial", "Whisker hits"), ("auditory_trial", "Auditory hits"), ("no_stim_trial", "False alarms")]
LAT = [("rt_ms", "Reaction time (ms)"), ("jaw_ms", "Jaw onset (ms)")]


def main():
    from compute import cluster_test, day0_table
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    d0 = d0[d0.early_lick == 0].sort_values("start_time")
    rows, jaws = [], {}
    for sid, st in d0.groupby("session_id"):
        mouse = st.mouse_id.iloc[0]
        if mouse not in jaws:
            jaws[mouse] = jaw_table(mouse)
        j = jaws[mouse]
        jaw = np.full(len(st), np.nan)
        if j is not None and len(j[j.session_id == sid]):
            jj = j[j.session_id == sid].sort_values("start_time")
            k = np.clip(np.searchsorted(jj.start_time.to_numpy(), st.start_time.to_numpy()), 0, len(jj) - 1)
            v = jj.jaw_dlc_onset.to_numpy(float)[k]
            ok = (np.abs(jj.start_time.to_numpy()[k] - st.start_time.to_numpy()) < 0.005) & (v >= 0) & (v <= 1)
            jaw = np.where(ok, v * 1000, np.nan)
        st = st.assign(jaw_ms=jaw, rt_ms=st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000)
        for tt, _ in TT:
            h = st[(st.trial_type == tt) & (st.lick_flag == 1)].head(N_HITS)
            rows.append(pd.DataFrame({"mouse_id": mouse, "session_id": sid, "cohort": st.cohort.iloc[0], "trial_type": tt,
                                      "hit_number": np.arange(1, len(h) + 1), "rt_ms": h.rt_ms.to_numpy(),
                                      "jaw_ms": h.jaw_ms.to_numpy()}))
    hits = pd.concat(rows, ignore_index=True)
    hits["bin"] = (hits.hit_number - 1) // BIN * BIN + (BIN + 1) / 2  # centre of 1-5, 6-10, ...
    per_mouse = hits.groupby(["mouse_id", "cohort", "trial_type", "bin"])[["rt_ms", "jaw_ms"]].median().reset_index()
    clusters = {}
    for tt, _ in TT:
        for y, _ in LAT:
            d = per_mouse[per_mouse.trial_type == tt].dropna(subset=[y])
            cnt = d.groupby(["bin", "cohort"]).mouse_id.nunique().unstack().fillna(0)
            keep = cnt.index[(cnt >= MIN_MICE).all(axis=1)] if cnt.shape[1] == 2 else []
            d = d[d.bin.isin(keep)]  # later hits come from fewer (more active) mice: test only well-populated bins
            clusters[f"{tt} {y}"] = cluster_test(d, "bin", y) if d.cohort.nunique() == 2 else []
    stem = OUT / f"rt_hits_{datetime.date.today()}"
    hits.to_csv(f"{stem}.csv", index=False)
    json.dump(clusters, open(f"{stem}_clusters.json", "w"), indent=1)
    plot(per_mouse, clusters, stem)
    for k, v in clusters.items():
        print(k, [(c["start"], c["end"], round(c["p"], 4)) for c in v])
    print(stem, hits.groupby(["trial_type", "cohort"]).mouse_id.nunique().to_dict())


def med_band(v, rng, n_boot=1000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = np.median(rng.choice(v, (n_boot, len(v))), axis=1)
    return np.median(v), np.percentile(b, 16), np.percentile(b, 84)


def plot(pm, clusters, stem):
    from plot import COHORTS, group_color
    set_nature_rc()
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 95 * MM), sharex=True)
    for r, (y, ylab) in enumerate(LAT):
        for c, (tt, title) in enumerate(TT):
            a = axs[r, c]
            ntxt = []
            for co in COHORTS:
                d = pm[(pm.trial_type == tt) & (pm.cohort == co)]
                col = group_color(tt, co)
                dx = (COHORTS.index(co) - 0.5) * 1.4
                for b, g in d.groupby("bin"):  # dot plot (user 2026-11-09): mice, then mean +- s.e.m. over mice
                    v = g[y].dropna()
                    if len(v) < MIN_MICE:
                        continue
                    a.scatter(b + dx + rng.uniform(-0.35, 0.35, len(v)), v, s=1, color=col, alpha=0.35, lw=0)
                    a.errorbar(b + dx, v.mean(), v.sem(), color=col, marker="o", ms=2.4, mec="0.15", mew=0.3, lw=0,
                               elinewidth=0.8, capsize=0, zorder=3)
                ntxt.append(f"{co} n={d.dropna(subset=[y]).mouse_id.nunique()}")
                for k2, (b, g) in enumerate(d.groupby("bin")):  # mice per bin under the axis
                    a.text(b, -0.17 - 0.07 * COHORTS.index(co), str(int(g[y].notna().sum())), color=col, fontsize=3.3,
                           ha="center", transform=a.get_xaxis_transform(), clip_on=False)
            top = a.get_ylim()[1]
            for cl in clusters.get(f"{tt} {y}", []):
                if cl["p"] < 0.05:
                    a.plot([cl["start"] - BIN / 2, cl["end"] + BIN / 2], [top, top], color="k", lw=1.2, solid_capstyle="butt")
            a.text(0.98, 0.04, ", ".join(ntxt), transform=a.transAxes, ha="right", fontsize=4.5)
            if r == 0:
                a.set_title(title, fontsize=6.5, pad=3)
            if c == 0:
                a.set_ylabel(ylab, labelpad=1)
            if r == 1:
                a.set_xlabel("Hit number (bins of 5); mice per bin below", labelpad=13)
            a.set_xlim(0, N_HITS + 1)
            a.set_xticks([3, 8, 13, 18, 23, 28], ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"], fontsize=4.5)
    fig.suptitle("First 30 licked trials of each type, whisker day 0, learners: dots, per-mouse median per bin of 5; "
                 "markers, mean ± s.e.m. over mice; bar, R+ vs R- cluster permutation P < 0.05", fontsize=5.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
