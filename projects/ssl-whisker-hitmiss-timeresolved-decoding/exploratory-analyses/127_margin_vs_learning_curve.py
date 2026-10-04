"""127 -- Within-session correlation of the single-trial decoder margin with the learning curve, against a LINEAR-SHIFT null
(user 2026-10-01: "Would it make sense to correlate change in a decoder metric with learning curves? Or circular?" -> do it
on hit/miss, then on whisker vs no-stim / whisker vs auditory, and compare distributions across cohorts).
Per session x decoding x window:
  z   held-out session-scaled signed margin per decoded trial (123 hit/miss v2, 128 stimulus decodings), class-residualised
      (z minus the mean of its class) so the changing class composition cannot create a trend;
  c   behaviour at the same trial: whisker - FA learning curve (HMM, sigma = 1, 024; also the whisker P(lick) curve), read at
      the trial's start time on the curve-aligned whisker-trial axis (linear interpolation in time);
  r   Pearson r(z, c);
  null linear shift (non-wrapping): z shifted against c by k = 10-50% of the trials, both directions, N_SHIFT random shifts
      -> r of each; both series keep their autocorrelation, so smooth-on-smooth correlation is in the null;
  excess = r - mean(null r); z_excess = excess / sd(null r); p = P(|null - mean| >= |r - mean|).
Group level (mouse = unit), per cohort: excess vs 0 (Wilcoxon AND one-sample t), fraction of sessions p < 0.05; R+ vs R-
(Mann-Whitney AND Welch). Uncorrected.
Sources: hitmiss = 123_singletrial_scores_all_v2.pkl (fallback to the v1 file), whisker_nostim / whisker_auditory =
128_singletrial_<decoding>.pkl (skipped if missing).
Outputs: figures/127_margin_vs_learning_curve.{pdf,png,svg}, 127_per_session.csv, 127_stats.csv
Run (haas): python 127_margin_vs_learning_curve.py
"""

from __future__ import annotations

import pickle
import zlib
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
N_SHIFT, LO, HI = 200, 0.1, 0.5
FS = 6.5


def sources():
    out = {}
    for f in ("123_singletrial_scores_all_v2.pkl", "123_singletrial_scores_all.pkl"):
        if (OUT / f).exists():
            out["hitmiss"] = OUT / f
            break
    for dec in ("whisker_nostim", "whisker_auditory"):
        if (OUT / f"128_singletrial_{dec}.pkl").exists():
            out[dec] = OUT / f"128_singletrial_{dec}.pkl"
    return out


def shift_null(z, c, rng):
    n = len(z)
    out = []
    for _ in range(N_SHIFT):
        k = int(rng.integers(max(1, int(LO * n)), max(2, int(HI * n)) + 1))
        a, b = (z[k:], c[:n - k]) if rng.random() < 0.5 else (z[:n - k], c[k:])
        if len(a) > 5 and np.std(a) > 0 and np.std(b) > 0:
            out.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(out)


def main():
    curves = {p["session_id"]: p["curves"] for p in pickle.load(open(LTP / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    rows = []
    for dec, f in sources().items():
        D = pickle.load(open(f, "rb"))
        for (sid, w), out in D["res"].items():
            m = D["meta"][sid]
            if sid not in curves:
                continue
            y = np.asarray(m["y"], bool)
            z = np.asarray(out["real"]["margin"], float).copy()
            ok = np.isfinite(z)
            for cls in (True, False):
                sel = ok & (y == cls)
                if sel.any():
                    z[sel] -= z[sel].mean()
            t, cw_t = np.asarray(m["t"], float), np.asarray(m["cw_t"], float)
            for cname in ("w-fa", "whisker"):
                cv = np.asarray(curves[sid][cname], float)
                if len(cv) != len(cw_t):
                    continue
                c = np.interp(t, cw_t, cv)
                zz, cc = z[ok], c[ok]
                if len(zz) < 20 or np.std(cc) == 0:
                    continue
                r = np.corrcoef(zz, cc)[0, 1]
                rng = np.random.default_rng(zlib.crc32(f"{sid}|{w}|{dec}|{cname}".encode()))
                nl = shift_null(zz, cc, rng)
                if len(nl) < 20:
                    continue
                mu, sd = nl.mean(), nl.std()
                rows.append(dict(session_id=sid, reward_group=m["reward_group"], decoding=dec, window=w, curve=cname,
                                 n=len(zz), r=r, null_mean=mu, null_sd=sd, excess=r - mu, z_excess=(r - mu) / sd if sd > 0 else np.nan,
                                 p=float(np.mean(np.abs(nl - mu) >= abs(r - mu)))))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "127_per_session.csv", index=False)
    srows = []
    for (dec, w, cname), g in P.groupby(["decoding", "window", "curve"]):
        ex = {}
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            ex[rg] = x.excess.to_numpy()
            r = dict(decoding=dec, window=w, curve=cname, cohort=rg, n=len(x), mean_r=x.r.mean(), mean_excess=x.excess.mean(),
                     sem_excess=x.excess.std(ddof=1) / np.sqrt(len(x)), frac_p05=(x.p < 0.05).mean(),
                     frac_pos_p05=((x.p < 0.05) & (x.excess > 0)).mean())
            if len(x) >= 5:
                r.update(p_wilcoxon=stats.wilcoxon(x.excess).pvalue, p_t=stats.ttest_1samp(x.excess, 0).pvalue)
            srows.append(r)
        pm = stats.mannwhitneyu(ex["R+"], ex["R-"]).pvalue if min(map(len, ex.values())) >= 3 else np.nan
        pw = stats.ttest_ind(ex["R+"], ex["R-"], equal_var=False).pvalue if min(map(len, ex.values())) >= 3 else np.nan
        for r in srows[-2:]:
            r.update(p_mw_cohorts=pm, p_welch_cohorts=pw)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "127_stats.csv", index=False)
    figure(P, S)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def figure(P, S):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    combos = [(d, w) for d in ("hitmiss", "whisker_nostim", "whisker_auditory") for w in ("5-50ms", "5-100ms")
              if ((P.decoding == d) & (P.window == w)).any()]
    fig, axes = plt.subplots(2, len(combos), figsize=(1.45 * len(combos) + 1, 5.2), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.08, wspace=0.45, hspace=0.6)
    rng = np.random.default_rng(0)
    for r_, cname in enumerate(("w-fa", "whisker")):
        for c_, (dec, w) in enumerate(combos):
            ax = axes[r_, c_]
            g = P[(P.decoding == dec) & (P.window == w) & (P.curve == cname)]
            for j, rg in enumerate(("R+", "R-")):
                v = g[g.reward_group == rg].excess.to_numpy()
                ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=COL[rg], lw=0, alpha=0.8)
                if len(v) > 1:
                    ax.errorbar(j, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="_", color="k", ms=12, elinewidth=1)
                s = S[(S.decoding == dec) & (S.window == w) & (S.curve == cname) & (S.cohort == rg)]
                if len(s):
                    s = s.iloc[0]
                    ax.text(j, 1.0, f"W {pf(s.get('p_wilcoxon', np.nan))}\nt {pf(s.get('p_t', np.nan))}",
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 1.5, color=COL[rg])
            s = S[(S.decoding == dec) & (S.window == w) & (S.curve == cname)]
            if len(s):
                ax.set_xlabel(f"R+ vs R−: MW {pf(s.p_mw_cohorts.iloc[0])}\nWelch {pf(s.p_welch_cohorts.iloc[0])}", fontsize=FS - 1)
            ax.axhline(0, color="0.6", lw=0.6, ls=":")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["R+", "R−"])
            ax.set_title(f"{dec.replace('_', ' vs ').replace('nostim', 'no-stim')}\n{w}", fontsize=FS, pad=16)
            if c_ == 0:
                ax.set_ylabel(f"r(margin, {'whisker − FA' if cname == 'w-fa' else 'whisker P(lick)'} curve)\n− linear-shift null")
    fig.suptitle("Within-session correlation of the single-trial decoder margin (class-residualised) with the learning "
                 "curve, beyond a linear-shift null; one dot per mouse (uncorrected)", fontsize=FS + 0.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"127_margin_vs_learning_curve.{ext}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
