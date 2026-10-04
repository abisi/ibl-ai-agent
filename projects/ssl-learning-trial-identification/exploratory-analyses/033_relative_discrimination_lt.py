"""033 -- A SINGLE learning-trial metric (user 2026-10-01: "Ideally I don't want a cascade but a single metric that satisfies
most things"; AB118 and MH070 are learners that start with high whisker licking).
Rule "relative discrimination onset" (LR):
  D_t = whisker - FA lick probability at whisker trial t, from the exact HMM posteriors (sigma = 1; whisker on its own
  trials, FA on no-stim trials interpolated in time onto the whisker trials; 024 / lt_lib), uncertainty kept:
  P(D_t > c) = sum_i P(W_t = p_i) * P(F_t < p_i - c) on the probability grid.
  Baseline B = posterior-mean D over the first BASE whisker trials (the mouse's own start).
  R+: LT = first trial t (>= BASE/2) such that P(D_s > B + m) >= p on >= K of the HOLD trials s = t .. t+HOLD-1;
  R-: mirror, P(D_s < B - m) >= p.
  Undefined when never reached: already discriminating from the start (nothing to rise), never licked (R-: B ~ 0, nothing to
  suppress), no sustained change. Lapses: a mid-session dip cannot satisfy the hold, so the LT lands after recovery.
Parameters swept: margin m in {0.10, 0.15, 0.20}, p in {0.8, 0.9}; BASE = 20, HOLD = 20, K = 16 (same hold as L7/L8).
Outputs: artifacts/033_lr_learning_trials.csv (all sessions x parameter sets), 033_lr_summary.csv (coverage per cohort,
agreement with other definitions), figures 033_lr_review_<cohort>.png (every mouse: curves, D, threshold, LT for the
default m = 0.15, p = 0.8 and the L6 / L7 / L8 LTs), 033_lr_examples.png (AB118, MH070 + sweep).
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/033_relative_discrimination_lt.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))
import lt_lib as L  # noqa: E402

COL = {"R+": "#00B400", "R-": "#C800C8"}
BASE, HOLD, K, SIGMA = 20, 20, 16, 1.0
MARGINS, PROBS = (0.10, 0.15, 0.20), (0.8, 0.9)
DEFAULT = (0.15, 0.8)
OTHER = ["L0 stored", "L6 joint CP", "L6 lenient", "L7 half-way", "L8 fixed margin", "lenient cascade"]


def p_diff_above(gw, gf, c):
    """P(W - F > c) per trial for grid marginals gw, gf (n x G)."""
    cdf_f = np.cumsum(gf, 1)
    idx = np.searchsorted(L.P_GRID, L.P_GRID - c, side="left") - 1          # P(F < p_i - c) = cdf_F[idx]
    pf = np.where(idx >= 0, cdf_f[:, np.clip(idx, 0, None)], 0.0)
    return np.sum(gw * pf, 1)


def lr_trial(gw, gf, rg, m, p):
    D = gw @ L.P_GRID - gf @ L.P_GRID
    n = len(D)
    B = float(D[:min(BASE, n)].mean())
    if rg == "R+":
        prob = p_diff_above(gw, gf, B + m)
    else:
        prob = 1.0 - p_diff_above(gw, gf, B - m)
    ok = prob >= p
    for t in range(BASE // 2, n - HOLD + 1):
        if ok[t] and ok[t:t + HOLD].sum() >= K:
            return t, B, prob
    return np.nan, B, prob


def main():
    inp = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    lt = pd.read_csv(ART / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")
    rows, keep = [], {}
    for sid, d in inp.items():
        tw = np.asarray(d["w_start"], float)
        gw = L.forward_backward(np.asarray(d["w_outcomes"], int), SIGMA)[0]
        gf = L.interp_marginals(L.forward_backward(np.asarray(d["n_outcomes"], int), SIGMA)[0],
                                np.asarray(d["n_start"], float), tw)
        rg = d["reward_group"]
        for m in MARGINS:
            for p in PROBS:
                t, B, prob = lr_trial(gw, gf, rg, m, p)
                rows.append(dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=rg,
                                 learning_category=d["learning_category"], n_whisker=len(tw), margin=m, p=p,
                                 baseline=B, LR=t, LR_frac=t / max(len(tw) - 1, 1) if np.isfinite(t) else np.nan,
                                 **{o: lt.loc[sid, o] if sid in lt.index else np.nan for o in OTHER}))
                if (m, p) == DEFAULT:
                    keep[sid] = dict(gw=gw, gf=gf, prob=prob, B=B, t=t, rg=rg, d=d)
    T = pd.DataFrame(rows)
    T.to_csv(ART / "033_lr_learning_trials.csv", index=False)
    summ = []
    for (m, p), g in T.groupby(["margin", "p"]):
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            r = dict(margin=m, p=p, cohort=rg, n=len(x), n_defined=int(x.LR.notna().sum()),
                     learners_defined=int(x[x.learning_category.isin(["good", "moderate"])].LR.notna().sum()),
                     n_learners=int(x.learning_category.isin(["good", "moderate"]).sum()),
                     median_LR_frac=x.LR_frac.median())
            for o in OTHER:
                both = x[x.LR.notna() & x[o].notna()]
                r[f"n_also_{o}"] = len(both)
                r[f"med_absdiff_{o}"] = float(np.median(np.abs(both.LR - both[o]))) if len(both) else np.nan
            summ.append(r)
    S = pd.DataFrame(summ)
    S.to_csv(ART / "033_lr_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[["margin", "p", "cohort", "n", "n_defined", "learners_defined", "n_learners", "median_LR_frac",
             "n_also_L6 lenient", "med_absdiff_L6 lenient", "med_absdiff_L7 half-way", "med_absdiff_L8 fixed margin"]]
          .round(2).to_string(index=False))
    for s in [k for k in keep if k.startswith(("AB118", "MH070"))]:
        print(s, T[T.session_id == s][["margin", "p", "baseline", "LR"]].to_string(index=False))
    review(keep, T)


def review(keep, T):
    plt.rcParams.update({"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False})
    for rg in ("R+", "R-"):
        sids = sorted([s for s, v in keep.items() if v["rg"] == rg], key=lambda s: (np.isnan(keep[s]["t"]), s))
        ncol = 6
        nrow = int(np.ceil(len(sids) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.7 * nrow + 0.6), squeeze=False)
        fig.subplots_adjust(left=0.03, right=0.995, top=1 - 0.5 / (1.7 * nrow + 0.6), bottom=0.02, hspace=0.6,
                            wspace=0.18)
        for ax, s in zip(axes.flat, sids):
            v = keep[s]
            w, f = v["gw"] @ L.P_GRID, v["gf"] @ L.P_GRID
            x = np.arange(len(w))
            ax.plot(x, w, color=COL[rg], lw=1)
            ax.plot(x, f, color="0.45", lw=0.9)
            ax.plot(x, w - f, color="k", lw=0.8)
            ax.fill_between(x, -0.25, -0.25 + 0.2 * v["prob"], color="#1f77b4", alpha=0.5, lw=0)
            thr = v["B"] + (DEFAULT[0] if rg == "R+" else -DEFAULT[0])
            ax.axhline(thr, color="k", lw=0.5, ls=":")
            ax.axhline(v["B"], color="0.6", lw=0.5, ls="--", xmax=BASE / max(len(w), 1))
            row = T[(T.session_id == s) & (T.margin == DEFAULT[0]) & (T.p == DEFAULT[1])].iloc[0]
            for o, c_, ls in (("L6 lenient", "#d62728", "-"), ("L7 half-way", "#ff7f0e", "--"), ("L8 fixed margin", "#17becf", ":")):
                if pd.notna(row[o]):
                    ax.axvline(row[o], color=c_, lw=0.8, ls=ls)
            if np.isfinite(v["t"]):
                ax.axvline(v["t"], color="#1f77b4", lw=1.6)
            ax.set_ylim(-0.27, 1.02)
            ax.set_title(f"{v['d']['mouse_id']} {v['d']['learning_category'] or ''} LR "
                         f"{'—' if not np.isfinite(v['t']) else int(v['t'])}", fontsize=6.5)
            ax.tick_params(labelsize=5)
        for ax in axes.flat[len(sids):]:
            ax.set_axis_off()
        fig.text(0.03, 0.998, f"{'R+' if rg == 'R+' else 'R−'}: relative-discrimination LT (blue line; margin {DEFAULT[0]}, "
                 f"p {DEFAULT[1]}, held {K}/{HOLD}); curves: whisker, FA grey, whisker − FA black; dotted = baseline "
                 f"{'+' if rg == 'R+' else '−'} margin; blue band = posterior P(beyond threshold); red L6 lenient, orange "
                 "L7, cyan L8", fontsize=7, va="top")
        fig.savefig(HERE / f"033_lr_review_{'Rplus' if rg == 'R+' else 'Rminus'}.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
