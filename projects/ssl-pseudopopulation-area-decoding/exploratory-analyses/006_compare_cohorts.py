"""006 -- R+ vs R- comparison of the pseudo-population decoding (002 output), whole brain and area groups (user request
2026-09-29). The mouse-level cohort-label permutation test is NOT run here (deferred by the user); cohorts are compared
from their independent hierarchical bootstraps.

Per cohort, area and iteration i: d_i = real_i - paired shift null_i (as in 003/005). Across iterations the bootstrap
SD of d is the standard error of the cohort estimate (mice resampled with replacement), so two tests are reported
(unpaired R+ vs R-; both always reported, memory rule):
  - non-parametric: bootstrap overlap, p = 2 * min(P(d+ > d-), P(d+ < d-)) over all iteration pairs (floor 1/n_iter);
  - parametric: z = (mean d+ - mean d-) / sqrt(SD+^2 + SD-^2), two-sided normal p.
Only within-area R+ vs R- comparisons; a bin / area is marked when BOTH tests have UNCORRECTED p < 0.05 (user
2026-09-30: no BH correction). BH q values (across bins; across areas within target) are still written to the csv for
reference only. Reaction-time distributions (012) drawn as one translucent histogram per cohort at the bottom of every
time-course panel (own invisible y axis).

Figure A (006_compare_cohorts): rows = targets; a = whole brain d, R+ vs R- (mean +- SD across iterations), black
bar = bins where the cohorts differ; b = window d per area group, R+ vs R- (dumbbells, mean +- SD), * = both q < 0.05,
dagger = < 5 eligible mice in a cohort; c = onset latency R+ vs R- per area, same row order as b (descriptive, no test).
Figure B (006_compare_cohorts_areas_<target>): small multiples, one panel per area present in both cohorts, R+ vs R- d
(mean +- SD), black bar = bins where the cohorts differ.
Windows: hitmiss post-stimulus 5..600 ms (hit/miss decoding grows late), modality_stim sensory 5..50 ms,
modality_lick pre-lick -100..0 ms. Onsets as 003.onset (stimulus-aligned: searched after the stimulus).
Outputs: figures/006_compare_cohorts*.pdf/.png, ../artifacts/006_compare_cohorts_{bins,windows}.csv
Run (repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/006_compare_cohorts.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"



def perm_text(target):
    """Whole-brain mouse-level cohort-label permutation p per window (009, 200 permutations x 10 iterations per group,
    with the linear-shift null; artifacts/009_perm_wb_abstract_pvalues.csv), or None."""
    f = ART / "009_perm_wb_abstract_pvalues.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.target == target]
    if d.empty:
        return None
    return "cohort permutation (whole brain):\n" + "\n".join(f"{w.replace('ms', ' ms')}: p = {p:.2f}" for w, p in zip(d.window, d.p_perm))

def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S003 = _load("s003", "003_summarize_pseudopop.py")
S005 = _load("s005", "005_overlay_areas.py")
S012 = _load("s012", "012_rt_band.py")          # reaction-time bands, one per cohort
S005.ABBR.update({"Amygdala and hypothalamus": "Amyg./hypoth.", "Cortical subplate": "CTXsp",
                  "Pons and medulla": "Pons/medulla"})
COL = {"R+": "#00B400", "R-": "#C800C8"}
TARGETS = ["hitmiss", "modality_stim", "modality_lick"]
WINDOW = {"hitmiss": ("post-stimulus", (0.005, 0.6)), "modality_stim": ("sensory", (0.005, 0.05)),
          "modality_lick": ("pre-lick", (-0.10, 0.0))}
MIN_MICE = 5


def two_tests(A, B):
    """A (nA x k), B (nB x k) bootstrap samples -> (p_boot, p_z) per column."""
    gt = (A[:, None, :] > B[None, :, :]).mean((0, 1))
    lt = (A[:, None, :] < B[None, :, :]).mean((0, 1))
    p_boot = np.clip(2 * np.minimum(gt, lt), 1 / min(len(A), len(B)), 1)
    z = (A.mean(0) - B.mean(0)) / np.sqrt(A.var(0, ddof=1) + B.var(0, ddof=1))
    return p_boot, 2 * stats.norm.sf(np.abs(z)), z


def load(target, cohort, n_iter):
    d = pd.read_parquet(ART / f"002_pseudo_{target}_{cohort}.parquet")
    d = d[d.skipped_reason.isna()]
    out = {}
    for a, g in d.groupby("area"):
        if len(g) < n_iter:
            continue
        g = g.sort_values("rep").head(n_iter)
        R = np.stack(g.curve.map(np.asarray).to_numpy())
        N = np.stack(g.null_mean_curve.map(np.asarray).to_numpy())
        out[a] = dict(D=R - N, mice=int(g.n_eligible_mice.iloc[0]))
    return out, d


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    colors, order = S005.allen()
    bins_rows, win_rows = [], []
    figA, axesA = plt.subplots(3, 3, figsize=(7.2, 7.6))
    figA.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.07, hspace=0.55, wspace=0.75)
    for ax in axesA.flat:
        ax.set_box_aspect(1)
    for r, target in enumerate(TARGETS):
        align = "lick" if target == "modality_lick" else "stim"
        t = np.array([e[1] for e in T.causal_bin_edges(S003.WIN[align], bin_width=0.05, stride=0.005)])
        x = t * 1000
        search = t > 0 if align == "stim" else np.ones(len(t), bool)
        wname, (w0, w1) = WINDOW[target]
        wm = (t >= w0) & (t <= w1)
        data = {c: load(target, c, n_iter)[0] for c in ("R+", "R-")}
        areas = [a for a in ["All units"] + order if a in data["R+"] and a in data["R-"] and a not in ("Somatosensory-body", "Olfactory areas")]   # excluded (user)
        res = {}
        for a in areas:
            A, B = data["R+"][a]["D"], data["R-"][a]["D"]
            pb, pz, z = two_tests(A, B)
            qb, qz = S005.bh(pb), S005.bh(pz)
            sig = (pb < 0.05) & (pz < 0.05)            # uncorrected (user 2026-09-30: no BH); q kept in the csv for reference
            wa, wb = np.nanmean(A[:, wm], 1)[:, None], np.nanmean(B[:, wm], 1)[:, None]
            wpb, wpz, wz = two_tests(wa, wb)
            ons = {c: S003.onset((np.nanpercentile(data[c][a]["D"], 5, 0) > 0)[search], t[search]) * 1000
                   for c in ("R+", "R-")}
            res[a] = dict(A=A, B=B, sig=sig, wa=wa[:, 0], wb=wb[:, 0], wpb=wpb[0], wpz=wpz[0], wz=wz[0], ons=ons)
            for k in range(len(t)):
                bins_rows.append(dict(target=target, area=a, t_ms=x[k], d_rplus=A[:, k].mean(), d_rminus=B[:, k].mean(),
                                      sd_rplus=A[:, k].std(ddof=1), sd_rminus=B[:, k].std(ddof=1), z=z[k],
                                      p_boot=pb[k], p_z=pz[k], q_boot_bh=qb[k], q_z_bh=qz[k], sig_both=bool(sig[k])))
        # window tests: within-area R+ vs R-, marked on UNCORRECTED p (user 2026-09-30); BH q across areas kept in the csv only
        fam = [a for a in areas if a != "All units"]
        qwb = dict(zip(fam, S005.bh([res[a]["wpb"] for a in fam])))
        qwz = dict(zip(fam, S005.bh([res[a]["wpz"] for a in fam])))
        qwb["All units"], qwz["All units"] = res["All units"]["wpb"], res["All units"]["wpz"]
        for a in areas:
            ra = res[a]
            win_rows.append(dict(target=target, area=a, window=wname, window_ms=f"{w0 * 1000:.0f}..{w1 * 1000:.0f}",
                                 n_mice_rplus=data["R+"][a]["mice"], n_mice_rminus=data["R-"][a]["mice"],
                                 d_rplus=ra["wa"].mean(), sd_rplus=ra["wa"].std(ddof=1), d_rminus=ra["wb"].mean(),
                                 sd_rminus=ra["wb"].std(ddof=1), diff=ra["wa"].mean() - ra["wb"].mean(), z=ra["wz"],
                                 p_boot=ra["wpb"], p_z=ra["wpz"], q_boot_bh=qwb[a], q_z_bh=qwz[a],
                                 sig_both=bool(ra["wpb"] < 0.05 and ra["wpz"] < 0.05),   # uncorrected (user)
                                 onset_rplus_ms=ra["ons"]["R+"], onset_rminus_ms=ra["ons"]["R-"]))
        # --- figure A
        axa, axb, axc = axesA[r]
        wb_ = res["All units"]
        for c, M in (("R+", wb_["A"]), ("R-", wb_["B"])):
            mu, sd = M.mean(0), M.std(0, ddof=1)
            axa.fill_between(x, mu - sd, mu + sd, color=COL[c], alpha=0.2, lw=0)
            axa.plot(x, mu, color=COL[c], lw=0.9, label=f"{c} ({data[c]['All units']['mice']} mice)")
        top = axa.get_ylim()[1]
        axa.plot(np.where(wb_["sig"], x, np.nan), np.full(len(x), top), color="k", lw=2, solid_capstyle="butt")
        axa.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
        axa.axvline(0, color="k", lw=0.5)
        axa.axvspan(w0 * 1000, w1 * 1000, color="0.92", zorder=0, lw=0)
        axa.set_xlim(x[0], x[-1])
        RTB = [(S012.rt_values(target, c), COL[c]) for c in ("R+", "R-")]
        S012.add_rt_band(axa, RTB, (x[0], x[-1]))
        pt = perm_text(target)
        if pt:
            axa.text(0.98, 0.3, pt, transform=axa.transAxes, ha="right", va="bottom", fontsize=4.5, color="0.2")
        axa.set_xlabel(f"time from {'stimulus' if align == 'stim' else 'first lick'} (ms)")
        axa.set_ylabel(f"{S005.SHORT[target]}\nwhole brain, real - null")
        axa.legend(loc="upper left", fontsize=5.5, handlelength=1.2)
        # b: dumbbells of window d per area
        ar = [a for a in areas if a != "All units"]
        ar = sorted(ar, key=lambda a: res[a]["wa"].mean())
        ar = ar + ["All units"]
        for k, a in enumerate(ar):
            ra = res[a]
            mp, mm = ra["wa"].mean(), ra["wb"].mean()
            axb.plot([mp, mm], [k, k], color="0.75", lw=0.8, zorder=1)
            for c, m, s, dy in (("R+", mp, ra["wa"].std(ddof=1), 0.15), ("R-", mm, ra["wb"].std(ddof=1), -0.15)):
                axb.errorbar(m, k + dy, xerr=s, fmt="o", ms=2.6, color=COL[c], elinewidth=0.6, capsize=0, zorder=2)
            rw = win_rows[-len(areas):][areas.index(a)]
            mark = ("*" if rw["sig_both"] else "") + ("†" if min(rw["n_mice_rplus"], rw["n_mice_rminus"]) < MIN_MICE else "")
            if mark:
                axb.text(1.01, k, mark, transform=axb.get_yaxis_transform(), va="center", fontsize=6.5)
        axb.set_yticks(range(len(ar)))
        axb.set_yticklabels(["Whole brain" if a == "All units" else S005.ABBR.get(a, a) for a in ar], fontsize=5)
        for lab, a in zip(axb.get_yticklabels(), ar):
            lab.set_color("k" if a == "All units" else colors.get(a, "0.3"))
        axb.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
        axb.tick_params(axis="y", length=0)
        axb.set_ylim(-0.7, len(ar) - 0.3)
        axb.set_xlabel(f"real - null, {wname}\n({w0 * 1000:.0f} to {w1 * 1000:.0f} ms)")
        # c: onsets R+ vs R- per area, same row order as b (areas without onset in a "none" strip on the right)
        allon = [v for a in areas for v in res[a]["ons"].values() if np.isfinite(v)]
        lo, hi = (min(allon), max(allon)) if allon else (0, 1)
        pad = 0.08 * (hi - lo + 20)
        none = hi + 2.5 * pad
        for k, a in enumerate(ar):
            op, om = (res[a]["ons"][c] if np.isfinite(res[a]["ons"][c]) else none for c in ("R+", "R-"))
            axc.plot([op, om], [k, k], color="0.75", lw=0.8, zorder=1)
            axc.scatter([op], [k + 0.15], s=9, color=COL["R+"], zorder=2)
            axc.scatter([om], [k - 0.15], s=9, color=COL["R-"], zorder=2)
        axc.axvline(hi + 1.4 * pad, color="#dddddd", lw=0.5)
        axc.axvline(0, color="k", lw=0.5)
        axc.set_xlim(lo - pad, none + 1.6 * pad)
        ticks = [v for v in axc.get_xticks() if lo - pad <= v <= hi + pad]
        axc.set_xticks(ticks + [none])
        axc.set_xticklabels([f"{v:.0f}" for v in ticks] + ["none"])
        axc.set_yticks(range(len(ar)))
        axc.set_yticklabels([])
        axc.tick_params(axis="y", length=0)
        axc.set_ylim(-0.7, len(ar) - 0.3)
        axc.set_xlabel(f"onset from {'stimulus' if align == 'stim' else 'first lick'} (ms)")
        if r == 0:
            for ax, lab, L in ((axa, "whole brain", "a"), (axb, "area groups, window effect", "b"),
                               (axc, "onset latency", "c")):
                ax.set_title(lab, pad=8)
                ax.text(-0.35, 1.12, L, transform=ax.transAxes, fontweight="bold", fontsize=9)
        # --- figure B: small multiples per area
        n = len(areas)
        ncol = 5
        nrow = int(np.ceil(n / ncol))
        figB, axesB = plt.subplots(nrow, ncol, figsize=(7.2, 1.55 * nrow + 0.6), squeeze=False, sharex=True)
        figB.subplots_adjust(left=0.08, right=0.99, top=1 - 0.5 / (1.55 * nrow + 0.6), bottom=0.35 / (1.55 * nrow + 0.6),
                             hspace=0.45, wspace=0.35)
        for k, a in enumerate(areas):
            ax = axesB[k // ncol, k % ncol]
            ax.set_box_aspect(1)
            ra = res[a]
            for c, M in (("R+", ra["A"]), ("R-", ra["B"])):
                mu, sd = M.mean(0), M.std(0, ddof=1)
                ax.fill_between(x, mu - sd, mu + sd, color=COL[c], alpha=0.2, lw=0)
                ax.plot(x, mu, color=COL[c], lw=0.7)
            top = ax.get_ylim()[1]
            ax.plot(np.where(ra["sig"], x, np.nan), np.full(len(x), top), color="k", lw=1.6, solid_capstyle="butt")
            ax.axhline(0, color="#bbbbbb", lw=0.4, ls=":")
            ax.axvline(0, color="k", lw=0.4)
            ax.set_xlim(x[0], x[-1])
            S012.add_rt_band(ax, RTB, (x[0], x[-1]))
            ax.tick_params(labelsize=5)
            nm = f"{data['R+'][a]['mice']}/{data['R-'][a]['mice']} mice"
            ax.set_title(f"{'Whole brain' if a == 'All units' else a}\n{nm}", fontsize=5.5,
                         color="k" if a == "All units" else colors.get(a, "k"))
            if k % ncol == 0:
                ax.set_ylabel("real - null", fontsize=5.5)
            if k // ncol == nrow - 1 or k + ncol >= n:
                ax.set_xlabel(f"time from {'stimulus' if align == 'stim' else 'first lick'} (ms)", fontsize=5.5)
                ax.tick_params(labelbottom=True)
        for k in range(n, nrow * ncol):
            axesB[k // ncol, k % ncol].set_axis_off()
        figB.suptitle(f"{S005.SHORT[target].replace(chr(10), ' ')}: R+ (green) vs R- (magenta), learning day, mean +- SD "
                      f"across {n_iter} iterations (PILOT); black bar: cohorts differ (bootstrap and z test, both p < 0.05, uncorrected)",
                      fontsize=6)
        for ext in ("pdf", "png"):
            figB.savefig(FIG / f"006_compare_cohorts_areas_{target}.{ext}", dpi=300)
        plt.close(figB)
    figA.suptitle("R+ vs R-, learning day: pseudo-population decoding (20 mice x 10 neurons, "
                  f"{n_iter} iterations x 10 shifts, PILOT counts)\nmean +- SD across iterations; black bar / *: cohorts "
                  "differ (bootstrap overlap and z test, both p < 0.05, uncorrected); † < 5 mice in a cohort; "
                  "grey band in a = window used in b\nmouse-level cohort-label permutation test not run yet",
                  fontsize=6.2, y=0.995)
    for ext in ("pdf", "png"):
        figA.savefig(FIG / f"006_compare_cohorts.{ext}", dpi=300)
    W = pd.DataFrame(win_rows)
    W.to_csv(ART / "006_compare_cohorts_windows.csv", index=False)
    pd.DataFrame(bins_rows).to_csv(ART / "006_compare_cohorts_bins.csv", index=False)
    pd.set_option("display.width", 250)
    print(W[["target", "area", "n_mice_rplus", "n_mice_rminus", "d_rplus", "d_rminus", "diff", "p_boot", "p_z",
             "q_boot_bh", "q_z_bh", "sig_both", "onset_rplus_ms", "onset_rminus_ms"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
