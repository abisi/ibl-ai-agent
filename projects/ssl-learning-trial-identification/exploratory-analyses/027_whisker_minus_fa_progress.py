"""027 -- Stand-alone whisker - false alarm Delta P(lick) along normalised session progress (user 2026-09-30: "Plot stand
alone the delta plick whisker minus false alarm figure, scaling y axes to emphasize difference, along session scaled
progression").
Per mouse: whisker HMM curve minus the time-interpolated FA curve (sigma = 1; 024 per-mouse curves), interpolated onto
101 points of session progress (first -> last whisker trial = 0 -> 1). Mean +- SEM across mice, R+ and R- overlaid.
The y axis is fitted to the data (not -1..1).
Stats: per point R+ vs R- Mann-Whitney (mid-grey bar) and Welch (light bar), p < 0.05, uncorrected; and (user
2026-10-01) a mouse-level reward-group label permutation (5000 shuffles, difference of cohort means) with cluster-based
correction over progress points (black bar = clusters with p < 0.05); dashed line = divergence onset (first point of the
earliest significant cluster). Table: artifacts/027_divergence_permutation.csv.
Versions: scope all | learners (good + moderate) x untrimmed | trimA1 (024, end-of-session disengagement rule A1).
Trial-axis version (user 2026-10-01: "do the test also for trial-based plots"): x = whisker trial; each cohort drawn
while >= 50% of its mice have trials; the permutation and per-point tests run over the trials where BOTH cohorts have
>= 50% of their mice (missing trials ignored in the means); both cohorts are drawn only over that range (user: R-
longer sessions hidden).
Outputs: 027_w_minus_fa_{progress,trial}_<scope>{,_trimA1}.{pdf,png,svg} (progress files were 027_w_minus_fa_progress_*)
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/027_whisker_minus_fa_progress.py
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

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}          # GROUP_COLORS rplus / rminus
FS_M, FS_S = 8, 7
NPROG = 101
MIN_FRAC = 0.5                                   # trial axis: points where >= 50% of a cohort's mice have trials
N_PERM = 5000
CLUSTER_ALPHA = 0.05


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def reward_group_permutation(A, B, xg, rng=None):
    """Mouse-level reward-group label permutation (user 2026-10-01: "do a permutation on reward group to see when both
    cohorts diverge"). Statistic per progress point: difference of cohort means (R+ - R-). Labels shuffled across mice
    N_PERM times. Pointwise p (two-sided) and cluster-based correction over progress points (clusters = runs of
    pointwise p < CLUSTER_ALPHA with the same sign; mass = sum |diff|; cluster p = fraction of permutations whose
    largest cluster mass >= the observed mass). Divergence onset = first point of the earliest cluster with p < 0.05."""
    rng = rng or np.random.default_rng(0)
    X = np.vstack([A, B])
    na = len(A)
    obs = np.nanmean(A, 0) - np.nanmean(B, 0)
    null = np.empty((N_PERM, X.shape[1]))
    for i in range(N_PERM):
        idx = rng.permutation(len(X))
        null[i] = np.nanmean(X[idx[:na]], 0) - np.nanmean(X[idx[na:]], 0)
    p_point = (np.sum(np.abs(null) >= np.abs(obs)[None, :], 0) + 1) / (N_PERM + 1)

    def clusters(d, p):
        out = []
        for sgn in (1, -1):
            for i0, i1 in runs((p < CLUSTER_ALPHA) & (np.sign(d) == sgn)):
                out.append(((i0, i1), float(np.abs(d[i0:i1]).sum())))
        return out
    # null pointwise p for each permutation via its rank among all permutations
    rank = np.argsort(np.argsort(-np.abs(null), 0), 0)
    p_null = (rank + 1) / (N_PERM + 1)
    max_mass = np.array([max([m for _, m in clusters(null[i], p_null[i])], default=0.0) for i in range(N_PERM)])
    cl = [(span, float((np.sum(max_mass >= m) + 1) / (N_PERM + 1))) for span, m in clusters(obs, p_point)]
    cl.sort(key=lambda c: c[0][0])
    sig = [c for c in cl if c[1] < 0.05]
    first_pt = next((xg[j] for j in range(len(xg)) if p_point[j] < 0.05), np.nan)
    return dict(onset=xg[sig[0][0][0]] if sig else np.nan, onset_p=sig[0][1] if sig else np.nan,
                first_point=first_pt, cluster_list=cl,
                clusters="; ".join(f"{xg[a]:.2f}-{xg[b - 1]:.2f} p={p:.4f}" for (a, b), p in cl))


def main():
    plt.rcParams.update({"font.family": "Arial", "font.size": FS_M, "axes.labelsize": FS_M, "xtick.labelsize": FS_S,
                         "ytick.labelsize": FS_S, "axes.linewidth": 0.9, "xtick.major.width": 0.9,
                         "ytick.major.width": 0.9, "xtick.major.size": 3, "ytick.major.size": 3,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    perm_rows = []
    for tag in ("", "_trimA1"):
        per = pickle.load(open(ART / f"024_avg_curves_per_mouse{tag}.pkl", "rb"))["sessions"]
        for scope in ("all", "learners"):
            sel = [p for p in per if scope == "all" or p["learning_category"] in ("good", "moderate")]
            for axis in ("progress", "trial"):
                M, valid = {}, {}
                if axis == "progress":
                    x = np.linspace(0, 1, NPROG)
                    for rg in ("R+", "R-"):
                        M[rg] = np.vstack([np.interp(x, np.linspace(0, 1, len(p["curves"]["w-fa"])), p["curves"]["w-fa"])
                                           for p in sel if p["reward_group"] == rg])
                        valid[rg] = np.ones(NPROG, bool)
                else:
                    nmax = max(len(p["curves"]["w-fa"]) for p in sel)
                    x = np.arange(1, nmax + 1)
                    for rg in ("R+", "R-"):
                        M[rg] = np.vstack([np.pad(np.asarray(p["curves"]["w-fa"], float),
                                                  (0, nmax - len(p["curves"]["w-fa"])), constant_values=np.nan)
                                           for p in sel if p["reward_group"] == rg])
                        valid[rg] = np.isfinite(M[rg]).sum(0) >= MIN_FRAC * len(M[rg])
                both = valid["R+"] & valid["R-"]
                L = int(np.argmin(both)) if not both.all() else len(both)      # tested range: both cohorts >= 50%
                fig = plt.figure(figsize=(3.0, 3.0))
                ax = fig.add_axes([0.2, 0.165, 0.76, 0.72])
                lo_hi = []
                for rg in ("R+", "R-"):
                    n_ = np.isfinite(M[rg]).sum(0)
                    with np.errstate(all="ignore"):
                        m, se = np.nanmean(M[rg], 0), np.nanstd(M[rg], 0, ddof=1) / np.sqrt(n_)
                    v = valid[rg] & (np.arange(len(x)) < L)      # user 2026-10-01: R- shown only over the tested range
                    ax.fill_between(x[v], (m - se)[v], (m + se)[v], color=COL[rg], alpha=0.22, lw=0)
                    ax.plot(x[v], m[v], color=COL[rg], lw=1.9)
                    lo_hi += [np.nanmin((m - se)[v]), np.nanmax((m + se)[v])]
                A, B = M["R+"][:, :L], M["R-"][:, :L]
                pm = np.array([stats.mannwhitneyu(A[:, j][np.isfinite(A[:, j])], B[:, j][np.isfinite(B[:, j])]).pvalue
                               for j in range(L)])
                pw = np.array([stats.ttest_ind(A[:, j][np.isfinite(A[:, j])], B[:, j][np.isfinite(B[:, j])],
                                               equal_var=False).pvalue for j in range(L)])
                perm = reward_group_permutation(A, B, x[:L])
                perm_rows.append(dict(version=tag.strip("_") or "untrimmed", scope=scope, axis=axis,
                                      n_rplus=len(A), n_rminus=len(B), n_perm=N_PERM, tested_up_to=x[L - 1],
                                      onset=perm["onset"], onset_cluster_p=perm["onset_p"],
                                      first_pointwise=perm["first_point"], clusters=perm["clusters"]))
                lo, hi = min(lo_hi + [0.0]), max(lo_hi)
                pad = 0.08 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
                sig_cl = np.zeros(L, bool)
                for (i0, i1), pc in perm["cluster_list"]:
                    if pc < 0.05:
                        sig_cl[i0:i1] = True
                half = 0.5 * (x[1] - x[0])
                for p_mask, yy, c in ((sig_cl, 1.11, "k"), (pm < 0.05, 1.075, "0.45"), (pw < 0.05, 1.04, "0.72")):
                    for i0, i1 in runs(p_mask):
                        ax.plot([x[i0] - half, x[i1 - 1] + half], [yy, yy], color=c, lw=2.2, solid_capstyle="butt",
                                transform=ax.get_xaxis_transform(), clip_on=False)
                xmax = x[L - 1]
                if np.isfinite(perm["onset"]):
                    ax.axvline(perm["onset"], color="k", lw=0.8, ls="--")
                    lab = f"{perm['onset']:.2f}" if axis == "progress" else f"trial {int(perm['onset'])}"
                    ax.text(perm["onset"] + 0.015 * (xmax - x[0]), 0.03, f"diverge at {lab}",
                            transform=ax.get_xaxis_transform(), fontsize=FS_S, va="bottom")
                ax.axhline(0, color="0.6", lw=0.7, ls=":")
                if axis == "progress":
                    ax.set_xlim(0, 1)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xlabel("session progress")
                else:
                    ax.set_xlim(0.5, xmax + 0.5)
                    ax.set_xlabel("whisker trial")
                ax.set_ylabel("Δ P(lick), whisker − false alarm")
                ax.text(0.97, 0.95, f"R+ (n = {len(A)})", color=COL["R+"], transform=ax.transAxes, ha="right", va="top",
                        fontsize=FS_S)
                ax.text(0.97, 0.87, f"R− (n = {len(B)})", color=COL["R-"], transform=ax.transAxes, ha="right", va="top",
                        fontsize=FS_S)
                stem = HERE / f"027_w_minus_fa_{axis}_{scope}{tag}"
                for ext in ("pdf", "png", "svg"):
                    fig.savefig(f"{stem}.{ext}", dpi=300)
                plt.close(fig)
                print(stem.name, "tested up to", x[L - 1], "| divergence onset", perm["onset"], "cluster p",
                      perm["onset_p"], "| first pointwise", perm["first_point"])
    pd.DataFrame(perm_rows).to_csv(ART / "027_divergence_permutation.csv", index=False)


if __name__ == "__main__":
    main()
