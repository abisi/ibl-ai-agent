"""149 -- Passive pre -> passive post change of the active choice-decoder readout against the LINEAR-SHIFT null (146, whole
brain, shared tracked stable units). User 2026-10-05: "should it be a linear shift? ... frame results on passive pre to passive
post, mostly, using active for the design of the metric".
Why: the hit / miss labels drift over the session (R- hit rate falls, R+ rises), so a decoder can partly learn session time;
passive post, later still, would then read miss-like (R-) or hit-like (R+) without any sensory change. The linear-shift null
(labels shifted against the time-ordered active whisker trials by 10-50 %, non-wrapping) keeps that drift; the excess
(real change - mean null change) is the change beyond what session time alone produces.
Per response (rows: epochbase = main, trialbase = per-trial baseline removed, baseline = -55..-20 ms window alone):
  a  real post - pre change of the passive whisker readout vs the mean shifted-label change (per session, paired)
  b  excess, whisker (real - null mean)
  c  excess, auditory
  d  excess, whisker - auditory
Sessions: all 146 sessions (>= 3 active whisker hits and >= 3 misses; user 2026-10-05); shifts are drawn among those keeping
>= 3 hits and misses; n per cohort on the panels.
Tests: within cohort Wilcoxon | t (excess vs 0; real vs null paired); R+ vs R- Mann-Whitney | Welch. Session = unit; uncorrected.
Outputs: figures/publication/149_passive_readout_shift_null_<scope>.{png,pdf,svg}; 149_stats.csv (scope column)
Run (haas, repo root): python .../149_passive_readout_shift_null_figure.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
MIN_SHIFT = 1                  # every session with >= 3 hits and misses has a null (146 draws only valid shifts)
RESP = [("epochbase", "5-35 ms, epoch baseline (main)"), ("trialbase", "5-35 ms, per-trial baseline"),
        ("baseline", "baseline window alone (state)")]


def excess_panel(ax, d, col, title, rows, scope, resp):
    v = {}
    for i, c in enumerate(COH):
        v[c] = d[d.reward_group == c][col].to_numpy(float)
        x = np.full(len(v[c]), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v[c]))
        ax.plot(x, v[c], "o", ms=2.2, color=COL[c], alpha=0.45, mew=0)
        ax.errorbar(i, np.nanmean(v[c]), H.sem(v[c]), fmt="o", ms=4.5, color=COL[c], lw=1.2, capsize=0, zorder=5)
        pw, pt, n = H.one_sample(v[c])
        rows.append(dict(scope=scope, response=resp, measure=col, test="excess vs 0 (Wilcoxon | t)", cohort=c, n=n,
                         mean_a=np.nanmean(v[c]), mean_b=np.nan, p_nonparam=pw, p_param=pt))
        ax.text(i, 1.02, f"{H.pnum(pw)}|{H.pnum(pt)}", color=COL[c], ha="center", fontsize=5, transform=ax.get_xaxis_transform())
    mw, we = H.unpaired(v["R+"], v["R-"])
    rows.append(dict(scope=scope, response=resp, measure=col, test="R+ vs R- (Mann-Whitney | Welch)", cohort="R+ vs R-", n=np.nan,
                     mean_a=np.nanmean(v["R+"]), mean_b=np.nanmean(v["R-"]), p_nonparam=mw, p_param=we))
    ax.text(0.5, 1.1, f"R+ vs R- {H.pnum(mw)}|{H.pnum(we)}", ha="center", fontsize=5, transform=ax.get_xaxis_transform())
    ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"R+\n{np.isfinite(v['R+']).sum()}", f"R-\n{np.isfinite(v['R-']).sum()}"], fontsize=5)
    ax.set_xlim(-0.6, 1.6); ax.set_title(title, fontsize=6, pad=14)


def figure(D, scope, rows):
    fig, axes = plt.subplots(3, 4, figsize=(6.6, 6.0))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.07, hspace=0.95, wspace=0.6)
    for r, (resp, rl) in enumerate(RESP):
        d = D[D.response == resp].copy()
        d["real_dW"] = d.ro_std_passive_post_W - d.ro_std_passive_pre_W
        # a real vs null change
        ax = axes[r, 0]
        xp = {"R+": (0, 1), "R-": (2.4, 3.4)}
        a = {c: d[d.reward_group == c].shift_null_dW_mean.to_numpy(float) for c in COH}
        b = {c: d[d.reward_group == c].real_dW.to_numpy(float) for c in COH}
        H.mean_pair(ax, xp, a, b)
        for c in COH:
            pw, pt, n = H.paired(b[c], a[c])
            rows.append(dict(scope=scope, response=resp, measure="real_dW vs shift_null_dW_mean", test="paired Wilcoxon | t",
                             cohort=c, n=n, mean_a=np.nanmean(a[c]), mean_b=np.nanmean(b[c]), p_nonparam=pw, p_param=pt))
            ax.text(np.mean(xp[c]), 1.02, f"{H.pnum(pw)}|{H.pnum(pt)}", color=COL[c], ha="center", fontsize=5,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["null", "real"] * 2, fontsize=5); ax.set_xlim(-0.5, 3.9)
        ax.set_ylabel(f"{rl}\npost - pre whisker readout\n(SD units; - = miss-ward)", fontsize=5.5)
        ax.set_title("real vs shifted-label change", fontsize=6, pad=14)
        excess_panel(axes[r, 1], d, "shift_excess_dW", "excess, whisker", rows, scope, resp)
        excess_panel(axes[r, 2], d, "shift_excess_dA", "excess, auditory", rows, scope, resp)
        excess_panel(axes[r, 3], d, "shift_excess_dWA", "excess, whisker - auditory", rows, scope, resp)
        for j, l in enumerate("abcd"):
            fig.text(axes[r, j].get_position().x0 - 0.05, axes[r, j].get_position().y1 + 0.035, f"{l}{r + 1}", fontsize=8, weight="bold")
    fig.suptitle(f"Passive pre -> post change of the active choice-decoder readout beyond the linear-shift null\n"
                 f"(whole brain, shared tracked stable units, {scope}; sessions with >= 3 hits and >= 3 misses)", fontsize=7)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"149_passive_readout_shift_null_{scope}.{ext}", dpi=300)
    plt.close(fig)


def main():
    H.setup()
    D = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    D = D[D.skipped_reason.isna() & (D.area == "whole_brain") & (D.unit_set == "stable")]
    n_all = D[D.response == "epochbase"].groupby("reward_group").size()
    D = D[D.shift_n >= MIN_SHIFT]
    n_ok = D[D.response == "epochbase"].groupby("reward_group").size()
    print(f"[149] sessions with >= {MIN_SHIFT} shifts: {n_ok.to_dict()} of {n_all.to_dict()}")
    rows = []
    for scope in ("all", "learners"):
        d = D if scope == "all" else D[H.mouse_of(D).isin(H.learners())]
        figure(d, scope, rows)
    R = pd.DataFrame(rows)
    R.to_csv(EA / "149_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
