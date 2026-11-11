"""Exploration (user 2026-11-10): candidate panel k, change in lick latency over the first 30 hits relative to each mouse's
first 5 hits, for reaction time and jaw onset; each bin tested against the first 5 hits within each cohort (no fit), and
R+ vs R- per bin, both corrected for multiple comparisons.

Data: whisker day 0, learners only (reference-sheet learning_category good / moderate; compute.day0_table), early licks
dropped; hit k = the k-th licked trial of that type (whisker hits, auditory hits), first 30. Latencies from trial start:
reaction time = computed_reaction_time (0-1 s); jaw onset = behaviour_analysis dlc_jaw_onset_times.pkl (0-1 s; mice with
video), matched by trial start time (explore_rt_hits logic).
Per mouse: delta = latency minus the mean of its first 5 hits (needs >= 3 of them); mean per bin of 5 hits (1-5, ...,
26-30), so bin 1-5 is 0 by construction. Then mean +- s.e.m. over mice.
Tests per bin 6-10 ... 26-30: within cohort, Wilcoxon signed-rank of the per-mouse delta vs 0 (= paired test of that bin
against the mouse's first 5 hits), Benjamini-Hochberg over the 5 bins per cohort, stars in the cohort colour over / under
the marker; R+ vs R-, Mann-Whitney on the deltas, Benjamini-Hochberg over the 5 bins, black stars at the top (Welch in
the CSV). Per-mouse least-squares slopes on single hits (ms per 10 hits; Wilcoxon vs 0, Mann-Whitney + Welch between
cohorts) are kept in the CSV, not drawn.
Output: Figures/_explore/rt_k_<date>.{pdf,png,svg}, _hits.csv (per hit), _bins.csv (tests), _slopes.csv.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_time_k.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from common import FIGURES_ROOT, two_group_tests  # noqa: E402
from paperstyle import MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
N_HITS, START, BIN = 30, 5, 5
TT = [("whisker_trial", "Whisker hits"), ("auditory_trial", "Auditory hits")]
LAT = [("rt_ms", "Reaction time"), ("jaw_ms", "Jaw onset")]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def hits_table():
    from compute import day0_table
    from explore_rt_session import jaw_table
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    rows, jaws = [], {}
    for sid, st in d0.groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        mouse = st.mouse_id.iloc[0]
        if mouse not in jaws:
            jaws[mouse] = jaw_table(mouse)
        j, jaw = jaws[mouse], np.full(len(st), np.nan)
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
    return pd.concat(rows, ignore_index=True)


def per_mouse_bins(h, y):
    """Per mouse x bin mean of latency minus the mean of the mouse's first START hits (>= 3 of them valid)."""
    d = h.dropna(subset=[y]).copy()
    ref = d[d.hit_number <= START].groupby("mouse_id")[y].agg(["mean", "count"])
    ref = ref[ref["count"] >= 3]["mean"]
    d = d[d.mouse_id.isin(ref.index)]
    d["delta"] = d[y] - d.mouse_id.map(ref)
    d["bin"] = (d.hit_number - 1) // BIN
    return d, d.groupby(["mouse_id", "cohort", "bin"]).delta.mean().reset_index()


def tests(b, tt, y):
    rows = []
    for bb in range(1, N_HITS // BIN):
        x = {c: b[(b.bin == bb) & (b.cohort == c)].delta.dropna() for c in ("R+", "R-")}
        rows.append(dict(trial_type=tt, latency=y, bin=bb, hits=f"{BIN * bb + 1}-{BIN * bb + BIN}",
                         **{f"{c}_mean": x[c].mean() for c in x}, **{f"{c}_n": len(x[c]) for c in x},
                         **{f"{c}_wilcoxon_p": stats.wilcoxon(x[c]).pvalue if len(x[c]) > 5 else np.nan for c in x},
                         **two_group_tests(x["R+"], x["R-"])))
    r = pd.DataFrame(rows)
    for col in ("R+_wilcoxon_p", "R-_wilcoxon_p", "mannwhitney_p", "welch_p"):
        q = col.replace("_p", "_q_bh")
        r[q] = np.nan
        ok = r[col].notna()
        if ok.any():
            r.loc[ok, q] = multipletests(r.loc[ok, col], method="fdr_bh")[1]
    return r


def slopes(d, tt, y):
    rows = []
    for (m, c), g in d.groupby(["mouse_id", "cohort"]):
        if len(g) >= 10:
            rows.append(dict(mouse_id=m, cohort=c, slope_per10=10 * np.polyfit(g.hit_number, g.delta, 1)[0]))
    s = pd.DataFrame(rows)
    out = [dict(trial_type=tt, latency=y, cohort=c, n=int((s.cohort == c).sum()), mean=s[s.cohort == c].slope_per10.mean(),
                wilcoxon_p=stats.wilcoxon(s[s.cohort == c].slope_per10).pvalue) for c in ("R+", "R-")]
    out.append(dict(trial_type=tt, latency=y, cohort="R+ vs R-",
                    **two_group_tests(s[s.cohort == "R+"].slope_per10, s[s.cohort == "R-"].slope_per10)))
    return out


def panel(ax, b, t, tt):
    from plot import COHORTS, group_color
    x0 = (BIN + 1) / 2
    for i, c in enumerate(COHORTS):
        col, dx = group_color(tt, c), (i - 0.5) * 1.0
        g = b[b.cohort == c].groupby("bin").delta
        m, e = g.mean(), g.sem().fillna(0)
        ax.errorbar(m.index * BIN + x0 + dx, m, e, color=col, marker="o", ms=2.4, lw=0.8, elinewidth=0.7, capsize=0, zorder=3)
        for _, r in t.iterrows():
            q = r[f"{c}_wilcoxon_q_bh"]
            if q < 0.05:
                up = m[r.bin] >= 0
                ax.text(r.bin * BIN + x0 + dx, m[r.bin] + e[r.bin] if up else m[r.bin] - e[r.bin], stars(q), color=col,
                        ha="center", va="bottom" if up else "top", fontsize=5.5)
    for _, r in t.iterrows():
        if r.mannwhitney_q_bh < 0.05:
            ax.text(r.bin * BIN + x0, 1.0, stars(r.mannwhitney_q_bh), transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6)
    ax.axhline(0, color="0.6", lw=0.4, ls=":", zorder=0)
    ax.set_xticks([3, 8, 13, 18, 23, 28], ["1–5", "6–10", "11–15", "16–20", "21–25", "26–30"], rotation=40, ha="right")
    ax.set_xlim(0, N_HITS + 1)
    ax.tick_params(length=2, labelsize=5, pad=1.5)


def main():
    from plot import COHORTS, group_color
    h = hits_table()
    stem = OUT / f"rt_k_{datetime.date.today()}"
    h.to_csv(f"{stem}_hits.csv", index=False)
    set_nature_rc()
    fig, axs = plt.subplots(2, 2, figsize=(95 * MM, 85 * MM),
                            gridspec_kw=dict(left=0.13, right=0.97, top=0.86, bottom=0.2, hspace=0.75, wspace=0.35))
    all_t, all_s = [], []
    for r, (tt, tname) in enumerate(TT):
        for c, (y, yname) in enumerate(LAT):
            d, b = per_mouse_bins(h[h.trial_type == tt], y)
            t = tests(b, tt, y)
            all_t.append(t)
            all_s += slopes(d, tt, y)
            ax = axs[r, c]
            panel(ax, b, t, tt)
            n = b.groupby("cohort").mouse_id.nunique()
            ax.set_title(f"{tname}, {yname.lower()}", fontsize=6, pad=7)
            ax.text(0.98, 0.03, f"R+ n={n.get('R+', 0)}, R− n={n.get('R-', 0)}", transform=ax.transAxes, ha="right",
                    fontsize=4.5)
            if c == 0:
                ax.set_ylabel(f"Δ latency from first {START} hits (ms)", fontsize=5.5, labelpad=1)
            if r == 1:
                ax.set_xlabel("Hit number", fontsize=5.5, labelpad=1)
        axs[r, 1].legend(handles=[Line2D([], [], color=group_color(tt, co), marker="o", ms=2.4, lw=0.8, label=co)
                                  for co in COHORTS], loc="upper left", fontsize=5, frameon=False, handlelength=1.2)
    fig.text(0.5, 0.965, "Lick latency over the first 30 hits, whisker day 0, learners (mean ± s.e.m. over mice)",
             ha="center", fontsize=6)
    fig.text(0.5, 0.015, "Coloured stars: bin vs first 5 hits within cohort (Wilcoxon signed-rank, Benjamini-Hochberg over "
             "5 bins).\nBlack stars: R+ vs R− (Mann-Whitney, Benjamini-Hochberg over 5 bins).", ha="center", fontsize=4.5)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=600)
    t = pd.concat(all_t)
    t.to_csv(f"{stem}_bins.csv", index=False)
    pd.DataFrame(all_s).to_csv(f"{stem}_slopes.csv", index=False)
    cols = ["trial_type", "latency", "hits", "R+_n", "R-_n", "R+_mean", "R+_wilcoxon_q_bh", "R-_mean", "R-_wilcoxon_q_bh",
            "mannwhitney_q_bh", "welch_q_bh"]
    with pd.option_context("display.width", 250):
        print(t[cols].round(4).to_string())
        print(pd.DataFrame(all_s).round(4).to_string())
    print(stem)


if __name__ == "__main__":
    main()
