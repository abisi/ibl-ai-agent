"""004 -- Whisker-hit and spontaneous-lick projections on the reward-lick CD across the session, day 0 vs expert (user 2026-10-16: "where can
I find ... the projections of whisker hits ... how do you quantify the effect to be comparable to that of expert?").
Data: ssl-prelick-convergence within_day 026 bins.csv: per session and class (WH, SL), the mean projection per fifth of the session (normalised
time tau in 5 equal bins) on the real CD and on the CDs rebuilt from shifted AH / SL labels (mean over 50 shifts); per neuron (/ sqrt N).
Each session's curve is expressed relative to its first fifth (so sessions with different offsets are comparable); mean +- s.e.m. over
sessions. Columns: WH, SL, WH - SL; rows: R+, R-; day 0 (light, dashed open markers) vs expert (dark, filled); grey: shifted CDs.
Right column: the quantity tested in 003 (slope over tau of the per-event projections, real - shifted), day 0 vs expert.
Population: learners rule on day 0, all expert sessions.
Output: figures/004_rl_timecourse_stages.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C


def curves(B, cls_fn):
    """per session: relative-to-first-fifth curves of the real and shifted projections for a class combination"""
    out = {}
    for kind in ("real", "null"):
        piv = {c: B[B.cls == c].pivot_table(index="session_id", columns="bin", values=kind) for c in ("WH", "SL")}
        idx = piv["WH"].index.intersection(piv["SL"].index)
        v = cls_fn({c: piv[c].loc[idx].to_numpy() for c in piv})
        out[kind] = v - v[:, :1]
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    B = C.population(pd.read_csv(C.m51.WITHIN / "cd_shift_null" / "bins.csv"))
    S = C.population(pd.read_csv(C.m51.WITHIN / "cd_shift_null" / "sessions.csv"))
    R = pd.read_csv(C.TAB / "003_stats.csv"); R = R[R.level == "whole_brain"]
    cols = [("WH", lambda d: d["WH"], "Whisker hits"), ("SL", lambda d: d["SL"], "Spontaneous licks"),
            ("WH − SL", lambda d: d["WH"] - d["SL"], "Whisker hits − spontaneous licks")]
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), gridspec_kw=dict(wspace=0.55, hspace=0.75, width_ratios=[1, 1, 1, 0.9]))
    x = np.arange(1, 6)
    for i, c in enumerate(("R+", "R-")):
        for j, (lab, fn, title) in enumerate(cols):
            ax = axs[i, j]
            for s_, ls, mfc, alpha in (("learning", (0, (3, 1.5)), "white", 0.55), ("expert", "-", None, 1.0)):
                d = B[(B.cohort == c) & (B.stage == s_)]
                cv = curves(d, fn)
                for kind, col in (("null", "0.6"), ("real", C.COH[c])):
                    v = cv[kind]; m, se = np.nanmean(v, 0), np.nanstd(v, 0) / np.sqrt(np.sum(np.isfinite(v), 0))
                    ax.errorbar(x + (0.06 if s_ == "expert" else -0.06), m, se, color=col, ls=ls, marker="o", ms=2.6, lw=0.9, capsize=0,
                                mfc=(mfc or col) if kind == "real" else "white", alpha=alpha if kind == "real" else 0.9,
                                label=f"{'day 0' if s_ == 'learning' else 'expert'} ({len(v)})" if kind == "real" else None)
            ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.set_xticks(x)
            if i == 1:
                ax.set_xlabel("fifth of the session")
            if j == 0:
                ax.set_ylabel(f"{c.replace('-', '−')}\nprojection on CD\n(per neuron, re. first fifth)", color=C.COH[c])
            if i == 0:
                ax.set_title(title, fontsize=6)
            if j == 0:
                ax.legend(frameon=False, fontsize=4.6, loc="upper left", handlelength=1.6)
        # right column: per-session slope beyond the shift null (the 003 quantity), WH and WH - SL
        ax = axs[i, 3]
        for k, (m, lab) in enumerate((("excess_pu", "WH"), ("excessC_pu", "WH − SL"))):
            for s_, xo in (("learning", 0), ("expert", 1)):
                v = S[(S.cohort == c) & (S.stage == s_)][m].dropna(); xx = k * 2.4 + xo
                ax.scatter(xx + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=3, color=C.COH[c], alpha=0.35, lw=0)
                ax.errorbar(xx, v.mean(), v.sem(), fmt="o", ms=3.2, color=C.COH[c], mfc="white" if s_ == "learning" else C.COH[c], capsize=0)
            r = R[(R.measure == ("rl_WH" if m == "excess_pu" else "rl_WHSL")) & (R.cohort == c)].iloc[0]
            ax.text(k * 2.4 + 0.5, 1.0, f"D0 {C.fp(r.p_d0)}\nE {C.fp(r.p_e)}\nD0 vs E {C.fp(r.p_stage_mwu)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=4.0, color=C.COH[c])
        ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_xticks([0, 1, 2.4, 3.4], ["D0", "E", "D0", "E"])
        ax.text(0.5, -0.32 if i else -0.22, "WH        WH − SL", transform=ax.transAxes, ha="center", fontsize=5)
        ax.set_ylabel("slope over the session\n(real − shifted, per neuron)", fontsize=5)
        if i == 0:
            ax.set_title("Quantified: slope beyond null", fontsize=6, pad=22)
    fig.suptitle("Reward-lick CD across the session, day 0 (dashed, open) vs expert (solid, filled); grey: shifted-label CDs", fontsize=6.6,
                 x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"004_rl_timecourse_stages.{ext}", dpi=300, bbox_inches="tight")
    print("ALL DONE", C.FIG / "004_rl_timecourse_stages.png")


if __name__ == "__main__":
    main()
