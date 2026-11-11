"""Exploration (user 2026-11-09): learning signatures in the piezo licks of whisker day 0 (NWB piezo_lick_times, synced;
loader of explore_lick_proba). Learners, trials of the shared behaviour table (perf != 6, not passive, warm-up trimmed).

1. lick bout per hit, against hit number (1st, 2nd, ... whisker hit; same for auditory hits): number of licks in the bout
   and time of its last lick (bout = licks from the first lick after the stimulus with inter-lick intervals < 0.5 s)
2. raster heatmap: P(lick in 50-ms bin) by whisker trial from the first whisker hit (rows) and time from stimulus
   (columns), mean over mice, per cohort (all whisker trials); auditory trials from the first auditory hit
3. reward-expectation index: licks in the late window [0.6, 1.5) s on each whisker hit / mean late licks on the
   session's auditory hits, against whisker-hit number
4. first-lick latency (piezo; first lick in [0, 1] s after the stimulus) on whisker and auditory hits, against hit number
5. licking outside trials (spontaneous lick rate between trials: licks more than 1 s from any trial start window
   [start - 1, start + 5] s) and the fraction of early-lick (aborted) trials, in 10 bins of normalised session time
Mean +- s.e.m. over mice; hit numbers shown while >= 10 mice per cohort contribute.
Output: Figures/_explore/lick_learning_{metrics,heatmaps}_<date>.{pdf,png} and the CSVs.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_lick_learning.py
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
from common import FIGURES_ROOT  # noqa: E402
from explore_lick_proba import session_licks  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
ILI, LATE, MAX_HIT, MIN_MICE = 0.5, (0.6, 1.5), 40, 10
H_EDGES = np.round(np.arange(-0.5, 2.0 + 1e-9, 0.05), 3)
N_TIME = 10


def bout(licks, t0):
    """(n licks, last lick time, first lick latency) of the bout starting at the first lick in [t0, t0 + 1]."""
    i = np.searchsorted(licks, t0)
    if i >= len(licks) or licks[i] > t0 + 1:
        return 0, np.nan, np.nan
    j = i
    while j + 1 < len(licks) and licks[j + 1] - licks[j] < ILI:
        j += 1
    return j - i + 1, licks[j] - t0, licks[i] - t0


def per_session(st, licks):
    st = st.sort_values("start_time")
    rows, heat = [], []
    t = st.start_time.to_numpy()
    for tt in ("whisker_trial", "auditory_trial"):
        s = st[(st.trial_type == tt) & (st.early_lick == 0)]
        hits = s[s.lick_flag == 1]
        a_late = None
        for k, t0 in enumerate(hits.start_time.to_numpy(), start=1):
            n, last, lat = bout(licks, t0)
            lo, hi = np.searchsorted(licks, [t0 + LATE[0], t0 + LATE[1]])
            tn0 = (t0 - t.min()) / max(t.max() - t.min(), 1e-9)  # normalised session time of the hit
            rows.append({"trial_type": tt, "hit_number": k, "n_licks": n, "last_lick_s": last, "latency_s": lat,
                         "late_licks": hi - lo, "third": ("early", "middle", "late")[min(int(tn0 * 3), 2)]})
        # heatmap: trials of this type from its first hit
        if len(hits):
            first = np.flatnonzero(s.lick_flag.to_numpy() == 1)[0]
            for r, t0 in enumerate(s.start_time.to_numpy()[first:first + 60]):
                lo, hi = np.searchsorted(licks, [t0 + H_EDGES[0], t0 + H_EDGES[-1]])
                heat.append({"trial_type": tt, "trial_from_first_hit": r,
                             **dict(zip(range(len(H_EDGES) - 1), np.histogram(licks[lo:hi] - t0, H_EDGES)[0] > 0))})
    hits = pd.DataFrame(rows)
    if len(hits):
        a = hits[hits.trial_type == "auditory_trial"].late_licks.mean()
        hits["expectation_index"] = np.where(hits.trial_type == "whisker_trial", hits.late_licks / a if a else np.nan,
                                             np.nan)
    # 5: spontaneous lick rate and early-lick fraction over normalised session time
    tn = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    b = np.minimum((tn * N_TIME).astype(int), N_TIME - 1)
    in_trial = np.zeros(len(licks), bool)
    for t0 in t:
        lo, hi = np.searchsorted(licks, [t0 - 1, t0 + 5])
        in_trial[lo:hi] = True
    spont = licks[~in_trial & (licks >= t.min()) & (licks <= t.max())]
    sb = np.minimum(((spont - t.min()) / max(t.max() - t.min(), 1e-9) * N_TIME).astype(int), N_TIME - 1)
    dur = (t.max() - t.min()) / N_TIME
    # time outside trial windows per bin, approximated by the bin duration minus 6 s per trial in the bin
    out_t = np.array([max(dur - 6 * (b == k).sum(), 1.0) for k in range(N_TIME)])
    early = pd.Series(st.early_lick.to_numpy()).groupby(b).mean().reindex(range(N_TIME))
    time = pd.DataFrame({"time_bin": range(N_TIME), "spont_rate_hz": np.bincount(sb, minlength=N_TIME) / out_t,
                         "early_fraction": early.to_numpy()})
    return hits, pd.DataFrame(heat), time


def job(args):
    sid, st = args
    _, _, licks = session_licks(sid)
    if not len(licks):
        return None
    out = per_session(st, licks)
    base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
    return tuple(o.assign(**base) for o in out)


def main():
    from compute import day0_table
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    with Pool(16) as pool:
        res = [r for r in pool.map(job, list(d0.groupby("session_id"))) if r is not None]
    hits, heat, time = (pd.concat([r[i] for r in res], ignore_index=True) for i in range(3))
    stem = datetime.date.today()
    for name, d in (("hits", hits), ("heat", heat), ("time", time)):
        d.to_csv(OUT / f"lick_learning_{name}_{stem}.csv", index=False)
    plot_metrics(hits, time, OUT / f"lick_learning_metrics_{stem}")
    plot_heat(heat, OUT / f"lick_learning_heatmaps_{stem}")
    plot_thirds(hits, OUT / f"lick_learning_thirds_{stem}")
    print(OUT, hits.groupby("cohort").mouse_id.nunique().to_dict())


def curve(ax, d, x, y, color, ls="-", label=None):
    m = d.groupby(["mouse_id", x])[y].mean().reset_index()
    g = m.groupby(x)[y]
    cnt = g.count()
    mu, se = g.mean()[cnt >= MIN_MICE], g.sem()[cnt >= MIN_MICE]
    ax.plot(mu.index, mu, color=color, ls=ls, lw=0.8, label=label)
    ax.fill_between(mu.index, mu - se, mu + se, color=color, alpha=0.25, lw=0)


def plot_metrics(hits, time, stem):
    set_nature_rc()
    A = {"R+": COLORS["auditory"], "R-": "#9a9aee"}
    fig, axs = plt.subplots(2, 4, figsize=(183 * MM, 90 * MM))
    h = hits[hits.hit_number <= MAX_HIT]
    panels = [("n_licks", "Licks per bout"), ("last_lick_s", "Last lick of bout (s)"), ("latency_s", "First-lick latency (s)")]
    for j, (y, lab) in enumerate(panels):
        a = axs[0, j]
        for c in ("R+", "R-"):
            curve(a, h[(h.trial_type == "whisker_trial") & (h.cohort == c)], "hit_number", y, COLORS[c], label=f"W {c}")
            curve(a, h[(h.trial_type == "auditory_trial") & (h.cohort == c)], "hit_number", y, A[c], ls="--", label=f"A {c}")
        a.set_ylabel(lab, labelpad=1)
        a.set_xlabel("Hit number", labelpad=1)
    axs[0, 0].legend(fontsize=4.5, frameon=False, ncol=2)
    a = axs[0, 3]
    for c in ("R+", "R-"):
        curve(a, h[(h.trial_type == "whisker_trial") & (h.cohort == c)], "hit_number", "expectation_index", COLORS[c])
    a.axhline(1, color="0.6", lw=0.4, ls=":")
    a.set_ylabel("Late licks W hit / A hit", labelpad=1)
    a.set_xlabel("Whisker hit number", labelpad=1)
    tn = time.assign(time_norm=(time.time_bin + 0.5) / N_TIME)
    for j, (y, lab) in enumerate((("spont_rate_hz", "Licks between trials (Hz)"), ("early_fraction", "Early-lick trials (fraction)"))):
        a = axs[1, j]
        for c in ("R+", "R-"):
            curve(a, tn[tn.cohort == c], "time_norm", y, COLORS[c], label=c)
        a.set_ylabel(lab, labelpad=1)
        a.set_xlabel("Session time (norm.)", labelpad=1)
    axs[1, 0].legend(fontsize=4.5, frameon=False)
    # bout length on the first 10 vs last 10 hits per mouse
    a = axs[1, 2]
    for k, (tt, cols) in enumerate((("whisker_trial", COLORS), ("auditory_trial", A))):
        for i, c in enumerate(("R+", "R-")):
            d = hits[(hits.trial_type == tt) & (hits.cohort == c)]
            first = d[d.hit_number <= 10].groupby("mouse_id").n_licks.mean()
            last = d.groupby("mouse_id").apply(lambda g: g.nlargest(10, "hit_number").n_licks.mean())
            x = k * 2.5 + i
            pair = pd.concat([first, last], axis=1).dropna()
            for _, r in pair.iterrows():
                a.plot([x - 0.2, x + 0.2], r.to_numpy(), color="0.8", lw=0.3)
            a.errorbar([x - 0.2, x + 0.2], pair.mean(), pair.sem(), color=cols[c], marker="o", ms=2, lw=0.8, capsize=0)
    a.set_xticks([0, 1, 2.5, 3.5], ["W R+", "W R-", "A R+", "A R-"], fontsize=5)
    a.set_ylabel("Licks per bout (first vs last 10 hits)", labelpad=1, fontsize=5)
    axs[1, 3].axis("off")
    n = hits.groupby("cohort").mouse_id.nunique().to_dict()
    axs[1, 3].text(0, 0.5, f"Learners, whisker day 0\nR+ n = {n.get('R+')}, R- n = {n.get('R-')} mice\nbout: ILI < {ILI} s\n"
                   f"late window {LATE[0]}-{LATE[1]} s\nhit numbers with >= {MIN_MICE} mice", fontsize=5.5, va="center")
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


def plot_thirds(hits, stem):
    """Per measure: early / middle / late thirds of normalised session time (x), whisker (solid) and auditory (dashed)
    hits per cohort, mean +- s.e.m. over mice (per-mouse mean over the hits of each third); grey lines, mice (whisker)."""
    set_nature_rc()
    A = {"R+": COLORS["auditory"], "R-": "#9a9aee"}
    TH = ["early", "middle", "late"]
    meas = [("n_licks", "Licks per bout"), ("last_lick_s", "Last lick of bout (s)"), ("latency_s", "First-lick latency (s)"),
            ("expectation_index", "Late licks W hit / A hit")]
    fig, axs = plt.subplots(1, 4, figsize=(183 * MM, 50 * MM))
    for a, (y, lab) in zip(axs, meas):
        for c in ("R+", "R-"):
            for tt, col, ls, dx in (("whisker_trial", COLORS[c], "-", -0.05), ("auditory_trial", A[c], "--", 0.05)):
                d = hits[(hits.trial_type == tt) & (hits.cohort == c)]
                if d[y].notna().sum() == 0:
                    continue
                m = d.groupby(["mouse_id", "third"])[y].mean().unstack().reindex(columns=TH)
                a.errorbar(np.arange(3) + dx, m.mean(), m.sem(), color=col, ls=ls, marker="o", ms=2, lw=0.8,
                           capsize=0, label=f"{'W' if tt == 'whisker_trial' else 'A'} {c}")
        if y == "expectation_index":
            a.axhline(1, color="0.6", lw=0.4, ls=":")
        a.set_xticks(range(3), ["Early", "Middle", "Late"])
        a.set_xlim(-0.4, 2.4)
        a.set_ylabel(lab, labelpad=1)
    axs[0].legend(fontsize=4.5, frameon=False, ncol=2)
    fig.suptitle("Thirds of normalised session time (learners, whisker day 0)", fontsize=6.5)
    fig.tight_layout(w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


def plot_heat(heat, stem):
    set_nature_rc()
    fig, axs = plt.subplots(1, 4, figsize=(183 * MM, 55 * MM), sharey=True)
    cols = [c for c in heat.columns if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())]
    for j, (tt, c) in enumerate([(t, c) for t in ("whisker_trial", "auditory_trial") for c in ("R+", "R-")]):
        d = heat[(heat.trial_type == tt) & (heat.cohort == c)]
        m = d.groupby(["mouse_id", "trial_from_first_hit"])[cols].mean().groupby("trial_from_first_hit").mean()
        im = axs[j].imshow(m.to_numpy(float), aspect="auto", cmap="magma", vmin=0, vmax=0.5,
                           extent=[H_EDGES[0], H_EDGES[-1], m.index.max() + 0.5, -0.5], interpolation="nearest")
        axs[j].axvline(0, color="w", lw=0.4, ls=":")
        axs[j].set_title(f"{'Whisker' if tt == 'whisker_trial' else 'Auditory'} {c} (n={d.mouse_id.nunique()})", fontsize=6, pad=2)
        axs[j].set_xlabel("Time from stimulus (s)", labelpad=1)
    axs[0].set_ylabel("Trials from first hit", labelpad=1)
    fig.colorbar(im, ax=axs, shrink=0.8, label="P(lick in 50-ms bin)")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
