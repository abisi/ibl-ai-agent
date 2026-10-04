"""123b -- Single-trial decoder margin (123, all learning sessions, A1-trimmed) vs the learning-trial (LT) definitions
(user 2026-10-01: "do the single-trial decoder margin run and compare with LT definitions").
Per session x window (hit vs miss, whole brain, 5-50 / 5-100 ms):
  z          held-out session-scaled signed margin per trial (> 0 = correct side), class-residualised (z - mean of its
             class) so the changing hit/miss composition does not create a step;
  neural CP  single change point of z: the split c (10-90% of the trials) maximising |Welch t| of mean(z[c:]) -
             mean(z[:c]); p_cp = P(max |t| of circularly shifted z >= observed), 500 shifts; sign = direction;
             converted to the curve-aligned whisker-trial index (whisker trials starting before decoded trial c);
  at the LT  margin change at each definition's LT (mean z of the WIN decoded trials from the LT minus the WIN before),
             ranked among the same change at every other split (>= EXCLUDE_NEAR trials away) -> percentile.
Group level per definition x cohort x window: neural CP vs LT (Spearman; median |CP - LT|; null = LTs permuted across
sessions within cohort, 2000 permutations -> p for the median |diff|); percentile (one-sample Wilcoxon AND t vs 0.5).
Also: mean smoothed z aligned to the LT (lenient cascade) per cohort; fraction of sessions with a significant neural CP.
Outputs: figures/123b_singletrial_vs_lt.{pdf,png,svg}, 123b_per_session.csv, 123b_stats.csv
Run (haas): python 123b_singletrial_vs_lt.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
COL = {"R+": "#00B400", "R-": "#C800C8"}
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate"]
WIN, EXCLUDE_NEAR, N_CIRC, N_PERM = 20, 10, 500, 2000
FS_L, FS_M, FS_S = 8, 7, 6


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def tstat(z, c):
    a, b = z[:c], z[c:]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (b.mean() - a.mean()) / se if se > 0 else 0.0


def change_point(z, rng):
    n = len(z)
    cs = np.arange(max(3, int(0.1 * n)), min(n - 3, int(0.9 * n)) + 1)
    ts = np.array([tstat(z, c) for c in cs])
    i = int(np.argmax(np.abs(ts)))
    obs = abs(ts[i])
    null = []
    for _ in range(N_CIRC):
        zs = np.roll(z, int(rng.integers(1, n)))
        null.append(np.max(np.abs([tstat(zs, c) for c in cs[::2]])))
    return int(cs[i]), float(ts[i]), float((np.sum(np.array(null) >= obs) + 1) / (N_CIRC + 1))


def margin_change(z, c):
    lo, hi = max(0, c - WIN), min(len(z), c + WIN)
    if c - lo < 5 or hi - c < 5:
        return np.nan
    return z[c:hi].mean() - z[lo:c].mean()


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS_M})
    D = pickle.load(open(OUT / "123_singletrial_scores_all.pkl", "rb"))
    res, meta = D["res"], D["meta"]
    rng = np.random.default_rng(0)
    rows, aligned = [], []
    for (sid, w), out in res.items():
        m = meta[sid]
        y = m["y"]
        z = np.asarray(out["real"]["margin"], float)
        ok = np.isfinite(z)
        if ok.sum() < 20:
            continue
        z = z.copy()
        for cls in (True, False):
            sel = ok & (y == cls)
            z[sel] -= z[sel].mean()
        z[~ok] = 0.0
        c, t, p = change_point(z, rng)
        cw_t, tt = m["cw_t"], m["t"]
        cp_w = int(np.sum(cw_t < tt[c]))
        allch = np.array([margin_change(z, k) for k in range(len(z))])
        base = dict(session_id=sid, reward_group=m["reward_group"], window=w, n_trials=len(z), n_whisker=len(cw_t),
                    cp_decoded=c, cp_whisker=cp_w, cp_t=t, cp_p=p, cp_frac=c / len(z))
        for d in DEFS:
            lt = m["lts"][d]
            r = dict(base, definition=d, lt_whisker=lt["lt_whisker"], lt_decoded=lt["lt_decoded"])
            if np.isfinite(lt["lt_decoded"]) and 0 < lt["lt_decoded"] < len(z):
                k = int(lt["lt_decoded"])
                real = allch[k]
                far = np.abs(np.arange(len(z)) - k) >= EXCLUDE_NEAR
                pl = allch[far & np.isfinite(allch)]
                r.update(margin_change=real,
                         percentile=float(np.mean(pl < real) + 0.5 * np.mean(pl == real)) if len(pl) >= 5 and np.isfinite(real)
                         else np.nan)
                if d == "lenient cascade":
                    zs = np.asarray(out["margin_smooth"], float)
                    zs = zs - np.nanmean(zs)
                    seg = np.full(81, np.nan)
                    for j, off in enumerate(range(-30, 51)):
                        if 0 <= k + off < len(zs):
                            seg[j] = zs[k + off]
                    aligned.append(dict(reward_group=m["reward_group"], window=w, seg=seg))
            rows.append(r)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "123b_per_session.csv", index=False)
    srows = []
    for (w, d), g in P.groupby(["window", "definition"]):
        for rg in ("R+", "R-"):
            x = g[(g.reward_group == rg) & g.lt_whisker.notna()]
            r = dict(window=w, definition=d, cohort=rg, n=len(x))
            if len(x) >= 5:
                r["spearman_cp_lt"], r["p_spearman"] = stats.spearmanr(x.cp_whisker, x.lt_whisker)
                obs = np.median(np.abs(x.cp_whisker - x.lt_whisker))
                perm = [np.median(np.abs(x.cp_whisker.to_numpy() - rng.permutation(x.lt_whisker.to_numpy())))
                        for _ in range(N_PERM)]
                r.update(median_abs_diff=obs, p_perm_median_abs_diff=(np.sum(np.array(perm) <= obs) + 1) / (N_PERM + 1),
                         median_abs_diff_null=float(np.median(perm)))
                pc = x.percentile.dropna()
                if len(pc) >= 5:
                    r.update(mean_percentile=pc.mean(), sem_percentile=pc.std(ddof=1) / np.sqrt(len(pc)),
                             p_wilcoxon_pct=stats.wilcoxon(pc - 0.5).pvalue if np.any(pc != 0.5) else np.nan,
                             p_t_pct=stats.ttest_1samp(pc, 0.5).pvalue)
            srows.append(r)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "123b_stats.csv", index=False)
    cps = P.drop_duplicates(["session_id", "window"])
    # figure
    wins = ["5-50ms", "5-100ms"]
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 5.6), gridspec_kw=dict(width_ratios=[0.9, 1.25, 1.25, 1.0]))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.09, wspace=0.55, hspace=0.55)
    yy = np.arange(len(DEFS))[::-1]
    for r_, w in enumerate(wins):
        ax = axes[r_, 0]
        x = P[(P.window == w) & (P.definition == "lenient cascade") & P.lt_whisker.notna()]
        for rg in ("R+", "R-"):
            g = x[x.reward_group == rg]
            sig = g.cp_p < 0.05
            ax.scatter(g.lt_whisker[sig], g.cp_whisker[sig], s=10, color=COL[rg], lw=0)
            ax.scatter(g.lt_whisker[~sig], g.cp_whisker[~sig], s=10, facecolor="white", edgecolor=COL[rg], lw=0.6)
        lim = max(x.lt_whisker.max(), x.cp_whisker.max()) * 1.05 if len(x) else 1
        ax.plot([0, lim], [0, lim], color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("LT (lenient cascade), whisker trial", fontsize=FS_S)
        ax.set_ylabel("neural change point, whisker trial", fontsize=FS_S)
        ax.set_title(f"hit vs miss {w}\nfilled: CP p < 0.05", fontsize=FS_S)
        ax.tick_params(labelsize=FS_S)
        s = S[S.window == w].set_index(["definition", "cohort"])
        ax = axes[r_, 1]
        for k, d in enumerate(DEFS):
            for rg, dy in (("R+", 0.15), ("R-", -0.15)):
                if (d, rg) in s.index and np.isfinite(s.loc[(d, rg)].get("median_abs_diff", np.nan)):
                    v = s.loc[(d, rg)]
                    ax.plot([v.median_abs_diff_null, v.median_abs_diff], [yy[k] + dy] * 2, color=COL[rg], lw=0.6,
                            alpha=0.6)
                    ax.scatter(v.median_abs_diff, yy[k] + dy, s=12, color=COL[rg] if v.p_perm_median_abs_diff < 0.05
                               else "white", edgecolor=COL[rg], lw=0.7, zorder=3)
                    ax.scatter(v.median_abs_diff_null, yy[k] + dy, s=8, marker="|", color="0.5")
        ax.set_yticks(yy)
        ax.set_yticklabels(DEFS, fontsize=FS_S - 0.5)
        ax.set_xlabel("median |neural CP − LT| (whisker trials)\n(grey tick: permuted-LT null)", fontsize=FS_S)
        ax.tick_params(labelsize=FS_S)
        ax = axes[r_, 2]
        for k, d in enumerate(DEFS):
            for rg, dy in (("R+", 0.15), ("R-", -0.15)):
                if (d, rg) in s.index and np.isfinite(s.loc[(d, rg)].get("mean_percentile", np.nan)):
                    v = s.loc[(d, rg)]
                    sig = v.p_wilcoxon_pct < 0.05
                    ax.errorbar(v.mean_percentile, yy[k] + dy, xerr=v.sem_percentile, fmt="o", color=COL[rg],
                                mfc=COL[rg] if sig else "white", ms=3.2, elinewidth=0.6, capsize=0)
        ax.axvline(0.5, color="0.6", lw=0.6, ls="--")
        ax.set_yticks(yy)
        ax.set_yticklabels([])
        ax.set_xlim(0.2, 0.9)
        ax.set_xlabel(f"percentile of margin change at LT\n(±{WIN} trials) among all splits", fontsize=FS_S)
        ax.tick_params(labelsize=FS_S)
        ax = axes[r_, 3]
        for rg in ("R+", "R-"):
            segs = np.array([a["seg"] for a in aligned if a["reward_group"] == rg and a["window"] == w])
            if not len(segs):
                continue
            mm = np.nanmean(segs, 0)
            se = np.nanstd(segs, 0, ddof=1) / np.sqrt(np.isfinite(segs).sum(0))
            xs = np.arange(-30, 51)
            v = np.isfinite(segs).sum(0) >= max(3, 0.5 * len(segs))
            ax.fill_between(xs[v], (mm - se)[v], (mm + se)[v], color=COL[rg], alpha=0.2, lw=0)
            ax.plot(xs[v], mm[v], color=COL[rg], lw=1.3, label=f"{'R+' if rg == 'R+' else 'R−'} (n={len(segs)})")
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("decoded trial − LT (lenient cascade)", fontsize=FS_S)
        ax.set_ylabel("smoothed margin (session-centred)", fontsize=FS_S)
        ax.legend(fontsize=FS_S, frameon=False)
        ax.tick_params(labelsize=FS_S)
        nsig = {rg: (cps[(cps.window == w) & (cps.reward_group == rg)].cp_p < 0.05).mean() for rg in ("R+", "R-")}
        ax.set_title(f"sessions with a significant neural CP:\nR+ {nsig['R+']:.0%}, R− {nsig['R-']:.0%}", fontsize=FS_S)
    for ax, L in zip(axes[0], "abcd"):
        ax.text(-0.25 if L != "b" else -0.75, 1.12, L, transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    fig.suptitle("Single-trial hit/miss decoder margin (whole brain, held-out, A1-trimmed) vs learning-trial definitions "
                 "(filled = p < 0.05, uncorrected)", fontsize=FS_M, y=0.985)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"123b_singletrial_vs_lt.{ext}", dpi=300)
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    print(cps.groupby(["window", "reward_group"]).apply(lambda g: pd.Series(dict(
        n=len(g), frac_sig=(g.cp_p < 0.05).mean(), frac_pos=(g.cp_t > 0).mean(), median_cp_frac=g.cp_frac.median()))))


if __name__ == "__main__":
    main()
