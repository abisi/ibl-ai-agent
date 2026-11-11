"""Exploration (user 2026-11-10): false alarms and spontaneous licks before vs after the first whisker hit, R+ vs R-.
v2 (same day): all active data, the auditory warm-up block before the first whisker trial included (the untrimmed day-0
table; v1 used the trimmed one, which keeps only the last warm-up trial), plus versions in time.

Data: whisker day 0, learners (reference sheet good / moderate), active trials (perf != 6, context != passive), early-lick
trials dropped from the no-stim analyses. First whisker hit = first licked whisker trial.
Rows:
1  FA, no-stim trial count: 0 = first no-stim trial at or after the first whisker hit, -1 = the last one before it; bins
   of 5 trials; pre = trials -5..-1 (>= 3), post = 0..9.
2  FA, time: no-stim trials by start time from the first whisker hit, bins of 1 min; pre = -5..0 min (>= 3 trials),
   post = 0..5 min.
3  Spontaneous lick bouts, time: lick-bout onsets (NWB piezo_lick_times; a lick with no lick in the 1 s before) outside
   the excluded windows, per minute of time outside them, within the active session (first trial start to last trial
   start + 5 s); bins of 1 min (>= 20 s covered); pre = -5..0 min, post = 0..5 min (>= 60 s covered each).
   Excluded windows (v3, user 2026-11-10 "remove 5 seconds after each rewarded trial"): every trial's 0..1 s response
   window from trial start, and for rewarded trials (licked auditory trials; licked whisker trials in R+; early-lick
   trials never rewarded) from trial start to 5 s after the reward (trial start + reaction time + 5 s). Licks in the
   quiet window before a trial and after unrewarded responses (FAs, R- whisker hits) beyond 1 s count as spontaneous.
4  Spontaneous licks, time: every lick outside the excluded windows, per minute outside them.
Columns: a, time course, mean +- s.e.m. over mice; black bar, R+ vs R- per bin (Mann-Whitney on per-mouse values,
Benjamini-Hochberg over the bins of the row; Welch in the CSV; >= 8 mice per cohort); coloured stars, bin vs the mouse's own
pre value within cohort (Wilcoxon signed-rank, BH over the post bins). b, pre vs post per mouse, Wilcoxon signed-rank per
cohort. c, post - pre, R+ vs R- Mann-Whitney (Welch, and pre / post alone, in the CSV).
Output: Figures/_explore/fa_divergence_<date>.{pdf,png,svg}, _nostim.csv, _bouts.csv, _bins.csv, _prepost.csv, _tests.csv.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_fa_divergence.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, session_nwbs, two_group_tests  # noqa: E402
from paperstyle import MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
MIN_MICE, BOUT_GAP, RESP_WIN, MIN_COVER, REWARD_EXCL = 8, 1.0, 1.0, 20.0, 5.0
COHORTS = ("R+", "R-")
# row: (bin edges, pre window, post window, minimum pre trials or seconds, x label)
ROWS = {
    "fa_count": dict(edges=np.arange(-10, 51, 5), pre=(-5, -1), post=(0, 9), min_pre=3,
                     xlabel="No-stim trial from first whisker hit", title="False alarms, by no-stim trial"),
    "fa_time": dict(edges=np.arange(-6, 21, 1), pre=(-5, 0), post=(0, 5), min_pre=3,
                    xlabel="Time from first whisker hit (min)", title="False alarms, by time"),
    "bouts": dict(edges=np.arange(-6, 21, 1), pre=(-5, 0), post=(0, 5), min_pre=60,
                  xlabel="Time from first whisker hit (min)", title="Spontaneous lick bouts, by time"),
    "licks": dict(edges=np.arange(-6, 21, 1), pre=(-5, 0), post=(0, 5), min_pre=60,
                  xlabel="Time from first whisker hit (min)", title="Spontaneous licks, by time"),
}
YLABEL = {"fa_count": "False-alarm rate", "fa_time": "False-alarm rate", "bouts": "Lick bouts per min",
          "licks": "Licks per min"}
EVENTS = {"bouts": "onsets", "licks": "licks"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def mlabel(c):
    return c.replace("-", "−")


def bh(p):
    p = pd.Series(np.asarray(p, float))
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q.to_numpy()


def day0_active():
    from compute import LEARNER
    d = shared_behaviour.load_untrimmed_day0()
    d = d[(d.context != "passive") & d.learning_category.map(LEARNER).fillna(False).astype(bool)]
    return d.sort_values(["session_id", "start_time"])


def lick_times(session_id):
    f = [p for p in session_nwbs(session_id.split("_")[0]) if p.stem == session_id]
    with h5py.File(f[0], "r") as h:
        return np.sort(h["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:])


def session_tables(d, trial_types=("no_stim_trial",)):
    """Per-trial table of `trial_types` (default no-stim only, rows 1-2; Figure 1 also passes whisker and auditory to
    get the row-2 'fa_time' computation for those trial types; column trial_type) and per-session lick tables."""
    ns_rows, bout_rows, checks = [], [], []
    for sid, st in d.groupby("session_id"):
        hits = st[(st.trial_type == "whisker_trial") & (st.lick_flag == 1)]
        if not len(hits):
            continue
        t_hit, info = hits.start_time.iloc[0], dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for tt in trial_types:
            ns = st[(st.trial_type == tt) & (st.early_lick == 0)]
            after = (ns.start_time >= t_hit).to_numpy()
            k = np.where(after, np.cumsum(after) - 1, -(np.cumsum(~after[::-1])[::-1]))
            ns_rows.append(pd.DataFrame({**info, "trial_type": tt, "start_time": ns.start_time.to_numpy(), "fa_index": k,
                                         "minutes": (ns.start_time.to_numpy() - t_hit) / 60,
                                         "lick": ns.lick_flag.to_numpy(float)}))
        licks = lick_times(sid)
        starts = st.start_time.to_numpy()
        lk = st[st.lick_flag == 1]
        j = np.searchsorted(licks, lk.start_time.to_numpy())
        checks.append(np.mean([(jj < len(licks)) and (licks[jj] - s < RESP_WIN + 0.2)
                               for jj, s in zip(j, lk.start_time.to_numpy())]))
        t0, t1 = starts[0], starts[-1] + 5
        licks = licks[(licks >= t0) & (licks < t1)]
        rewarded = ((st.lick_flag == 1) & (st.early_lick == 0) & ((st.trial_type == "auditory_trial") | (
            (st.trial_type == "whisker_trial") & (st.cohort == "R+")))).to_numpy()
        rt = st.computed_reaction_time.where(st.computed_reaction_time.between(0, RESP_WIN), RESP_WIN).to_numpy()
        win = merge(np.c_[starts, np.where(rewarded, starts + rt + REWARD_EXCL, starts + RESP_WIN)])
        k = np.searchsorted(win[:, 0], licks, side="right") - 1
        excl = (k >= 0) & (licks < win[np.clip(k, 0, None), 1])
        first = np.r_[True, np.diff(licks) > BOUT_GAP]
        bout_rows.append(dict(**info, t_hit=t_hit, t0=t0, t1=t1, onsets=licks[first & ~excl], licks=licks[~excl],
                              win=win, n_rewarded=int(rewarded.sum())))
    print(f"licked trials with a piezo lick within {RESP_WIN + 0.2} s of start: median {np.median(checks):.3f}, "
          f"min {np.min(checks):.3f} (clock check)")
    return pd.concat(ns_rows, ignore_index=True), bout_rows


def merge(w):
    """Union of [start, end) intervals, sorted by start."""
    w = w[np.argsort(w[:, 0])]
    out = [list(w[0])]
    for a, b in w[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out)


def covered(s, lo, hi):
    """Seconds of [lo, hi) inside the active session and outside the excluded windows."""
    lo, hi = max(lo, s["t0"]), min(hi, s["t1"])
    if hi <= lo:
        return 0.0
    w = s["win"]
    return hi - lo - np.clip(np.minimum(w[:, 1], hi) - np.maximum(w[:, 0], lo), 0, None).sum()


def bout_rate(s, lo_min, hi_min, key="onsets"):
    lo, hi = s["t_hit"] + 60 * lo_min, s["t_hit"] + 60 * hi_min
    cov = covered(s, lo, hi)
    n = ((s[key] >= lo) & (s[key] < hi)).sum()
    return n / cov * 60 if cov > 0 else np.nan, cov


def per_mouse(kind, ns, bouts):
    """Per-mouse bin values (columns mouse_id, cohort, bin, value) and pre / post values."""
    r = ROWS[kind]
    edges = r["edges"]
    if kind in EVENTS:
        b, pp, key = [], [], EVENTS[kind]
        for s in bouts:
            for lo, hi in zip(edges[:-1], edges[1:]):
                v, cov = bout_rate(s, lo, hi, key)
                if cov >= MIN_COVER:
                    b.append(dict(mouse_id=s["mouse_id"], cohort=s["cohort"], bin=lo, value=v))
            (pre, cpre), (post, cpost) = bout_rate(s, *r["pre"], key), bout_rate(s, *r["post"], key)
            pp.append(dict(mouse_id=s["mouse_id"], cohort=s["cohort"], pre=pre if cpre >= r["min_pre"] else np.nan,
                           post=post if cpost >= r["min_pre"] else np.nan, pre_cover_s=cpre, post_cover_s=cpost))
        return pd.DataFrame(b), pd.DataFrame(pp)
    x = ns.fa_index if kind == "fa_count" else ns.minutes
    t = ns.assign(x=x)
    t = t[(t.x >= edges[0]) & (t.x < edges[-1])].assign(bin=lambda d: edges[np.searchsorted(edges, d.x, side="right") - 1])
    b = t.groupby(["mouse_id", "cohort", "bin"]).lick.mean().rename("value").reset_index()
    hi_inc = kind == "fa_count"  # trial windows are inclusive, time windows half-open

    def win(lo, hi, min_n):
        w = t[(t.x >= lo) & ((t.x <= hi) if hi_inc else (t.x < hi))].groupby(["mouse_id", "cohort"]).lick.agg(["mean", "count"])
        return w["mean"].where(w["count"] >= min_n)

    pp = pd.concat([win(*r["pre"], r["min_pre"]).rename("pre"), win(*r["post"], 1).rename("post")], axis=1).reset_index()
    return b, pp


def bin_tests(kind, b, pp):
    pre = pp.set_index("mouse_id").pre.dropna()
    rows = []
    for x in sorted(b.bin.unique()):
        v = {c: b[(b.bin == x) & (b.cohort == c)].set_index("mouse_id").value.dropna() for c in COHORTS}
        r = dict(row=kind, bin=x, **{f"{c}_n": len(v[c]) for c in COHORTS}, **{f"{c}_mean": v[c].mean() for c in COHORTS},
                 **{f"{c}_sem": v[c].sem() for c in COHORTS})
        if min(len(v["R+"]), len(v["R-"])) >= MIN_MICE:
            r.update(two_group_tests(v["R+"], v["R-"]))
        for c in COHORTS:
            m = pd.concat([v[c].rename("v"), pre.rename("pre")], axis=1, join="inner")
            r[f"{c}_n_paired"] = len(m)
            post_bin = x >= ROWS[kind]["post"][0]
            r[f"{c}_vs_pre_p"] = stats.wilcoxon(m.v - m.pre).pvalue if post_bin and len(m) > 5 and (m.v != m.pre).any() else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    for col in ("mannwhitney_p", "welch_p", "R+_vs_pre_p", "R-_vs_pre_p"):
        t[col[:-2] + "_q_bh"] = bh(t[col]) if col in t else np.nan
    return t


def prepost_tests(kind, pp):
    pp = pp.dropna(subset=["pre", "post"]).assign(change=lambda d: d.post - d.pre)
    rows = []
    for c in COHORTS:
        g = pp[pp.cohort == c]
        rows.append(dict(row=kind, test=f"{c} post vs pre (Wilcoxon signed-rank)", n=len(g), pre_mean=g.pre.mean(),
                         post_mean=g.post.mean(), p=stats.wilcoxon(g.post, g.pre).pvalue))
    for col in ("pre", "post", "change"):
        rows.append(dict(row=kind, test=f"R+ vs R- {col}",
                         **two_group_tests(pp[pp.cohort == "R+"][col], pp[pp.cohort == "R-"][col])))
    return pp.assign(row=kind), pd.DataFrame(rows)


def color(c):
    from plot import group_color
    return group_color("no_stim_trial", c)


def panel_curve(ax, kind, t):
    r = ROWS[kind]
    w = np.diff(r["edges"])[0]
    xc = (w - 1) / 2 if kind == "fa_count" else w / 2
    lo, hi = r["pre"]
    ax.axvspan(lo - 0.5 if kind == "fa_count" else lo, hi + 0.5 if kind == "fa_count" else hi, color="0.92", lw=0, zorder=0)
    ax.axvline(-0.5 if kind == "fa_count" else 0, color="0.5", lw=0.5, ls="--", zorder=0)
    for i, c in enumerate(COHORTS):
        dx = (i - 0.5) * 0.16 * w
        ax.errorbar(t.bin + xc + dx, t[f"{c}_mean"], t[f"{c}_sem"], color=color(c), marker="o", ms=2.2, lw=0.8,
                    elinewidth=0.6, capsize=0, zorder=3)
        for _, rr in t.iterrows():
            s = stars(rr[f"{c}_vs_pre_q_bh"]) if pd.notna(rr[f"{c}_vs_pre_q_bh"]) else ""
            if s:
                up = c == "R+"
                y = rr[f"{c}_mean"] + (rr[f"{c}_sem"] if up else -rr[f"{c}_sem"])
                ax.text(rr.bin + xc + dx, y, s, color=color(c), ha="center", va="bottom" if up else "top", fontsize=4.5)
    for _, rr in t.iterrows():
        if rr.get("mannwhitney_q_bh", np.nan) < 0.05:
            pad = 0.4 if kind == "fa_count" else 0.06
            x0, x1 = (rr.bin - pad, rr.bin + w - 1 + pad) if kind == "fa_count" else (rr.bin + pad, rr.bin + w - pad)
            ax.plot([x0, x1], [0.97, 0.97], transform=ax.get_xaxis_transform(), color="k", lw=2, solid_capstyle="butt",
                    clip_on=False)
    ax.set_xlim(r["edges"][0] - (1 if kind == "fa_count" else 0.2), r["edges"][-1] + (0 if kind == "fa_count" else 0.2))
    ax.set_ylim(0, max((t[f"{c}_mean"] + t[f"{c}_sem"]).max() for c in COHORTS) * 1.22)
    ax.set_xlabel(r["xlabel"], labelpad=1)
    ax.set_ylabel(YLABEL[kind], labelpad=1)
    ax.text(np.mean(r["pre"]), 0.02, "pre", ha="center", fontsize=4.5, color="0.4", transform=ax.get_xaxis_transform())


def panel_prepost(ax, kind, pp, tests):
    rng = np.random.default_rng(0)
    top = np.nanmax(pp[["pre", "post"]].to_numpy())
    for i, c in enumerate(COHORTS):
        g = pp[pp.cohort == c]
        x = np.array([0, 1]) + 2.6 * i
        for (_, r), jj in zip(g.iterrows(), rng.uniform(-0.08, 0.08, len(g))):
            ax.plot(x + jj, [r.pre, r.post], color=color(c), lw=0.25, alpha=0.3, zorder=1)
        ax.errorbar(x, [g.pre.mean(), g.post.mean()], [g.pre.sem(), g.post.sem()], color=color(c), marker="o", ms=3,
                    lw=1.2, elinewidth=0.9, capsize=0, zorder=3)
        p = tests[tests.test.str.startswith(f"{c} post vs pre")].p.iloc[0]
        ax.plot(x, [top * 1.04] * 2, color="k", lw=0.5)
        ax.text(x.mean(), top * 1.05, stars(p) or "n.s.", ha="center", va="bottom", fontsize=5.5 if stars(p) else 4.8)
        ax.text(x.mean(), -0.24, f"{mlabel(c)} (n={len(g)})", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=5, color=color(c))
    ax.set_xticks([0, 1, 2.6, 3.6], ["pre", "post"] * 2)
    ax.set_xlim(-0.5, 4.1)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel(YLABEL[kind], labelpad=1)


def panel_change(ax, kind, pp, tests):
    rng = np.random.default_rng(1)
    for i, c in enumerate(COHORTS):
        g = pp[pp.cohort == c].change
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(g)), g, s=2.5, color=color(c), alpha=0.5, lw=0, zorder=2)
        ax.errorbar(i + 0.32, g.mean(), g.sem(), color=color(c), marker="o", ms=3, elinewidth=0.9, capsize=0, zorder=3)
    ax.axhline(0, color="0.6", lw=0.4, ls=":", zorder=0)
    r = tests[tests.test == "R+ vs R- change"].iloc[0]
    top = pp.change.max() + 0.1 * np.ptp(pp.change)
    ax.plot([0, 1], [top, top], color="k", lw=0.5)
    ax.text(0.5, top, stars(r.mannwhitney_p) or "n.s.", ha="center", va="bottom", fontsize=5.5 if stars(r.mannwhitney_p) else 4.8)
    ax.set_ylim(None, top + 0.15 * np.ptp(pp.change))
    ax.set_xticks([0, 1], [mlabel(c) for c in COHORTS])
    for tl, c in zip(ax.get_xticklabels(), COHORTS):
        tl.set_color(color(c))
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("Post − pre", labelpad=1)


def main():
    d = day0_active()
    ns, bouts = session_tables(d)
    stem = OUT / f"fa_divergence_{datetime.date.today()}"
    ns.to_csv(f"{stem}_nostim.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k not in ("onsets", "licks", "win")}
                  | dict(n_bouts=len(s["onsets"]), n_licks=len(s["licks"]))
                  for s in bouts]).to_csv(f"{stem}_bouts.csv", index=False)
    set_nature_rc()
    fig = plt.figure(figsize=(150 * MM, 195 * MM))
    gs = fig.add_gridspec(4, 3, width_ratios=[2.3, 1.15, 0.75], left=0.07, right=0.985, top=0.96, bottom=0.105,
                          wspace=0.45, hspace=0.75)
    all_bins, all_pp, all_tests = [], [], []
    for row, kind in enumerate(ROWS):
        b, pp = per_mouse(kind, ns, bouts)
        t = bin_tests(kind, b, pp)
        pp, tests = prepost_tests(kind, pp)
        all_bins.append(t), all_pp.append(pp), all_tests.append(tests)
        axs = [fig.add_subplot(gs[row, i]) for i in range(3)]
        panel_curve(axs[0], kind, t)
        panel_prepost(axs[1], kind, pp, tests)
        panel_change(axs[2], kind, pp, tests)
        for ax, letter, title in zip(axs, "abc", [ROWS[kind]["title"], "Pre vs post, per mouse", "Change"]):
            ax.tick_params(length=2, labelsize=5.5, pad=1.5)
            ax.set_title(title, fontsize=6, pad=8, loc="left")
            ax.text(-0.02, 1.13, f"{row + 1}{letter}", transform=ax.transAxes, fontsize=7.5, fontweight="bold", ha="right")
        if row == 0:
            axs[0].legend(handles=[Line2D([], [], color=color(c), marker="o", ms=2.2, lw=0.8, label=mlabel(c))
                                   for c in COHORTS], loc="upper right", frameon=False, fontsize=5.5, handlelength=1.2,
                          bbox_to_anchor=(1, 0.93))
    n = d.groupby("cohort").mouse_id.nunique()
    fig.text(0.01, 0.008,
             f"Whisker day 0, learners (R+ {n['R+']}, R− {n['R-']} mice), all active trials incl. the auditory warm-up "
             f"block; mean ± s.e.m. over mice. Row 1: no-stim trials (early licks dropped) numbered from the first whisker "
             f"hit,\nbins of 5, pre −5..−1 (≥ 3), post 0..9. Row 2: same trials by time, bins of 1 min, pre −5..0 min "
             f"(≥ 3 trials), post 0..5 min. Row 3: lick-bout onsets (no lick in the previous {BOUT_GAP:.0f} s) outside "
             f"excluded windows,\nper minute outside them; excluded: every trial's 0–{RESP_WIN:.0f} s, and rewarded "
             f"trials from start to {REWARD_EXCL:.0f} s after the reward. Row 4: every lick outside the excluded "
             f"windows, per minute outside them.\na, black bar: R+ vs R− (Mann-Whitney, BH over bins, q < 0.05); coloured "
             f"stars: bin vs pre (Wilcoxon signed-rank, BH over post bins).\nb, Wilcoxon signed-rank. c, R+ vs R− Mann-Whitney (Welch in CSV).",
             fontsize=4.6, va="bottom")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=600)
    bins, pps, tests = pd.concat(all_bins), pd.concat(all_pp), pd.concat(all_tests)
    bins.to_csv(f"{stem}_bins.csv", index=False)
    pps.to_csv(f"{stem}_prepost.csv", index=False)
    tests.to_csv(f"{stem}_tests.csv", index=False)
    cols = ["row", "bin", "R+_n", "R-_n", "R+_mean", "R-_mean", "mannwhitney_q_bh", "welch_q_bh", "R+_n_paired",
            "R+_vs_pre_q_bh", "R-_n_paired", "R-_vs_pre_q_bh"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(bins[cols].round(4).to_string())
        print(tests.drop(columns=["mannwhitney_U", "welch_t"]).round(4).to_string())
    print(stem)


if __name__ == "__main__":
    main()
