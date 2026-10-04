"""129 -- Is the mid-session split better than any other split, and which split gives the largest decoding change?
(user 2026-10-01: "compare the mid-session decoding change wrt a null distribution of [shifted splits], is mid-session
better than any random? By the same logic, what is the trial that brings the best change, excluding session borders").
Input: 122 placebo run with EVERY whisker trial as a candidate split (122_lt_placebo_whole_brain_step1.parquet; perf != 6,
A1-trimmed; separate size-matched decoders before / after each split): hit vs miss 5-50 / 5-100 ms, whisker vs auditory
-100..0 ms pre-lick.
Interior splits only: whisker-trial index k with BORDER <= k <= n_whisker - BORDER (and >= 2 trials per class per epoch).
  mid       percentile of the session-half split's delta (post - pre) among all interior splits >= EXCLUDE_NEAR trials away;
            excess = delta_half - mean(those); per cohort vs 0.5 / 0 (Wilcoxon AND t), R+ vs R- (MW AND Welch);
  best      k_best = argmax over interior splits of the delta profile smoothed over SMOOTH consecutive splits (raw argmax
            also saved); position as a fraction of the session; cohort distributions (MW AND Welch); relation to the
            behavioural change point (L6x forced, 034/036): Spearman, median |k_best - CP| vs CPs permuted across sessions
            within cohort (2000 permutations);
  profile   mean delta vs split position (10 bins of the session) per cohort -- where along the session splitting helps.
Note: the max of a noisy profile is biased upward; k_best is used for its POSITION, not its size.
Outputs: figures/129_mid_and_best_split.{pdf,png,svg}, 129_per_session.csv, 129_stats.csv
Run (haas): python 129_mid_and_best_split.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}
MEASURES = [("hitmiss", "5-50ms", "hit/miss 5-50 ms"), ("hitmiss", "5-100ms", "hit/miss 5-100 ms"),
            ("modality_lick", "-100-0ms", "whisker vs auditory pre-lick")]
BORDER, EXCLUDE_NEAR, SMOOTH, N_PERM = 10, 10, 5, 2000
FS = 6.5


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def two(a, b):
    if min(len(a), len(b)) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def main():
    d = pd.read_parquet(OUT / f"122_lt_placebo_whole_brain{os.environ.get('SSL_PLACEBO_TAG', '_step1')}.parquet")
    d = d[d.skipped_reason.isna()]
    cp = pd.read_csv(LTP / "036_joint_cp_relaxed.csv")
    cp = cp[(cp.version == "L6x") & (cp.bf_th == 0.3) & (cp.p_th == 0.9)].set_index("session_id").LT
    rows, prof = [], []
    for (sid, rg, dec, w), g in d.groupby(["session_id", "reward_group", "decoding", "window"]):
        nw = int(g.n_whisker_curve.iloc[0])
        sp = g[(g.split_k >= BORDER) & (g.split_k <= nw - BORDER)].sort_values("split_k")
        if len(sp) < 10:
            continue
        k, dl = sp.split_k.to_numpy(), sp.delta_matched.to_numpy()
        for kk, dd in zip(k, dl):
            prof.append(dict(reward_group=rg, decoding=dec, window=w, pos=kk / max(nw - 1, 1), delta=dd, session_id=sid))
        sm = pd.Series(dl).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
        kb, kr = int(k[np.argmax(sm)]), int(k[np.argmax(dl)])
        row = dict(session_id=sid, reward_group=rg, decoding=dec, window=w, n_whisker=nw, n_splits=len(sp),
                   k_best=kb, k_best_raw=kr, best_pos=kb / max(nw - 1, 1), best_delta_smooth=float(sm.max()),
                   cp_L6x=cp.get(sid, np.nan), cp_frac=cp.get(sid, np.nan) / max(nw - 1, 1))
        h = g[g.real_for == "half"]
        if len(h):
            hd = h.delta_matched.iloc[0]
            # position of the half split on the whisker axis: whisker trials before the split time
            far = np.abs(sp.n_pre.to_numpy() - h.n_pre.iloc[0]) >= EXCLUDE_NEAR
            pl = dl[far]
            if len(pl) >= 5:
                row.update(half_delta=hd, half_excess=hd - pl.mean(), half_pct=float(np.mean(pl < hd) + 0.5 * np.mean(pl == hd)))
        rows.append(row)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "129_per_session.csv", index=False)
    PR = pd.DataFrame(prof)
    srows = []
    rng = np.random.default_rng(0)
    for dec, w, _ in MEASURES:
        g = P[(P.decoding == dec) & (P.window == w)]
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            hx = x.dropna(subset=["half_pct"])
            r = dict(decoding=dec, window=w, cohort=rg, n=len(x), n_half=len(hx),
                     half_pct_mean=hx.half_pct.mean(), half_excess_mean=hx.half_excess.mean(),
                     p_half_pct_wilcoxon=stats.wilcoxon(hx.half_pct - 0.5).pvalue if len(hx) >= 5 else np.nan,
                     p_half_pct_t=stats.ttest_1samp(hx.half_pct, 0.5).pvalue if len(hx) >= 5 else np.nan,
                     best_pos_median=x.best_pos.median(), best_pos_mean=x.best_pos.mean())
            c = x.dropna(subset=["cp_frac"])          # FRACTIONS of the session (trial counts scale with length for both)
            if len(c) >= 5:
                r["spearman_best_cp"], r["p_spearman_best_cp"] = stats.spearmanr(c.best_pos, c.cp_frac)
                obs = np.median(np.abs(c.best_pos - c.cp_frac))
                perm = [np.median(np.abs(c.best_pos.to_numpy() - rng.permutation(c.cp_frac.to_numpy()))) for _ in range(N_PERM)]
                r.update(median_abs_best_cp=obs, median_abs_null=float(np.median(perm)),
                         p_perm_best_cp=(np.sum(np.array(perm) <= obs) + 1) / (N_PERM + 1))
            srows.append(r)
        a, b = g[g.reward_group == "R+"], g[g.reward_group == "R-"]
        pm1, pw1 = two(a.half_excess.dropna(), b.half_excess.dropna())
        pm2, pw2 = two(a.best_pos, b.best_pos)
        for r in srows[-2:]:
            r.update(p_half_cohorts_mw=pm1, p_half_cohorts_welch=pw1, p_bestpos_cohorts_mw=pm2, p_bestpos_cohorts_welch=pw2)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "129_stats.csv", index=False)
    figure(P, PR, S)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))


def figure(P, PR, S):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    fig, axes = plt.subplots(len(MEASURES), 4, figsize=(8.27, 7.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.07, wspace=0.45, hspace=0.75)
    rng = np.random.default_rng(0)
    for r, (dec, w, lab) in enumerate(MEASURES):
        g = P[(P.decoding == dec) & (P.window == w)]
        s = S[(S.decoding == dec) & (S.window == w)].set_index("cohort")
        ax = axes[r, 0]
        for j, rg in enumerate(("R+", "R-")):
            v = g[g.reward_group == rg].half_pct.dropna().to_numpy()
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=COL[rg], lw=0)
            if len(v) > 1:
                ax.errorbar(j, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="_", color="k", ms=12, elinewidth=1)
            if rg in s.index:
                ax.text(j, 1.0, f"W {pf(s.loc[rg, 'p_half_pct_wilcoxon'])}\nt {pf(s.loc[rg, 'p_half_pct_t'])}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 1.5, color=COL[rg])
        ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R+", "R−"])
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(f"{lab}\nmid-session percentile\namong all splits", fontsize=FS)
        if "R+" in s.index:
            ax.set_xlabel(f"MW {pf(s.loc['R+', 'p_half_cohorts_mw'])}, Welch {pf(s.loc['R+', 'p_half_cohorts_welch'])}", fontsize=FS - 1)
        ax = axes[r, 1]
        bins = np.linspace(0, 1, 11)
        for rg in ("R+", "R-"):
            ax.hist(g[g.reward_group == rg].best_pos, bins=bins, histtype="step", color=COL[rg], lw=1.2)
        ax.set_xlabel("best-split position (fraction of session)", fontsize=FS - 0.5)
        ax.set_ylabel("sessions")
        if "R+" in s.index:
            ax.set_title(f"cohorts MW {pf(s.loc['R+', 'p_bestpos_cohorts_mw'])}, Welch {pf(s.loc['R+', 'p_bestpos_cohorts_welch'])}",
                         fontsize=FS - 0.5)
        ax = axes[r, 2]
        pr = PR[(PR.decoding == dec) & (PR.window == w)].copy()
        pr["bin"] = np.clip((pr.pos * 10).astype(int), 0, 9)
        for rg in ("R+", "R-"):
            q = pr[pr.reward_group == rg].groupby(["session_id", "bin"]).delta.mean().reset_index()
            m = q.groupby("bin").delta.agg(["mean", "sem"])
            ax.errorbar((m.index + 0.5) / 10, m["mean"], yerr=m["sem"], color=COL[rg], lw=1, marker="o", ms=2.5, capsize=0)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel("split position (fraction of session)", fontsize=FS - 0.5)
        ax.set_ylabel("Δacc (post − pre)")
        ax = axes[r, 3]
        for rg in ("R+", "R-"):
            x = g[(g.reward_group == rg)].dropna(subset=["cp_frac"])
            ax.scatter(x.cp_frac, x.best_pos, s=6, color=COL[rg], lw=0)
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("behavioural CP (L6x, fraction)", fontsize=FS - 0.5)
        ax.set_ylabel("best neural split (fraction)")
        t = []
        for rg in ("R+", "R-"):
            if rg in s.index and np.isfinite(s.loc[rg].get("p_perm_best_cp", np.nan)):
                t.append(f"{rg}: |Δ| {s.loc[rg, 'median_abs_best_cp']:.2f} vs null {s.loc[rg, 'median_abs_null']:.2f}, "
                         f"p {pf(s.loc[rg, 'p_perm_best_cp'])}")
        ax.set_title("\n".join(t), fontsize=FS - 1.5)
    fig.suptitle(f"Mid-session vs all other splits, and the split with the largest decoding change (every whisker trial a "
                 f"candidate; first/last {BORDER} whisker trials excluded; best = argmax of the {SMOOTH}-split smoothed profile). "
                 "Uncorrected.", fontsize=FS + 0.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"129_mid_and_best_split.{ext}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
