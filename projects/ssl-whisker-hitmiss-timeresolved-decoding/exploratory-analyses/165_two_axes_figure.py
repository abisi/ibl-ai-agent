"""165 -- Two-row figure, one row per within-session axis, same quantities in both rows (user 2026-10-08).
Row 1, hit - miss axis (stimulus onset, 5-35 ms; 135 with SSL_135_HARM=1: unit mean / SD from the active trials, as the pre-lick CD):
  c  passive whisker and auditory projections on the unit hit - miss axis (per neuron, / sqrt(N)), relative to the session's own passive
     pre (= 0): pre -> post on the real axis (solid) and on axes rebuilt from linearly shifted hit / miss labels (dashed, mean of the
     shifts); mean +- s.e.m. over sessions per cohort
  d  excess beyond session time (real - shifted) for whisker, auditory and whisker - auditory; filled: Wilcoxon vs 0 p < 0.05; top:
     R+ vs R- Mann-Whitney
  e  per area group (axis from the group's units): whisker - auditory excess, R+ and R-; ANOVA excess ~ cohort x area with cohort labels
     permuted across mice (cohort, cohort x area); right: per-area Mann-Whitney p and Benjamini-Hochberg q
Row 2, reward-lick axis (pre-lick, 100 ms before the first lick; ssl-prelick-convergence within_day 026):
  c  whisker-hit (WH) and spontaneous-lick (SL) projections on the unit reward-lick CD (per neuron) in five bins of session time,
     relative to the session's own first bin (= 0): real CD (solid) vs CDs from linearly shifted AH / SL labels (dashed)
  d  excess of the slope over session time (real - shifted) for WH, SL and WH - SL
  e  per area group: WH - SL excess, as row 1
Columns a (schematic) and b (example session in state space: 163 / ssl-prelick-convergence 085) are left for the author.
Learners, learning day only; session = unit; non-parametric tests.
Outputs: figures/publication/165_two_axes_figure.{svg,pdf,png}; 165_stats.csv
Run (haas, repo root): python .../165_two_axes_figure.py
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
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
PL = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-prelick-convergence/within_day/sl_stable/cd_shift_null")
N_PERM = 2000
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def area_colors():
    try:
        sys.path.insert(0, str(Path.home() / "code" / "ephys_utilities"))
        from ephys_utilities.allen_utils import allen_utils
        return allen_utils.get_custom_area_groups_colors()
    except Exception:  # noqa: BLE001
        return {}


def wil(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan


def mwu(a, b):
    a, b = (np.asarray(v, float) for v in (a, b)); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue if min(len(a), len(b)) >= 3 else np.nan


def pn(p):
    return "n.s." if not np.isfinite(p) or p >= 0.05 else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def anova_perm(D, val, rng):
    """two-way ANOVA val ~ cohort x area (sum-of-squares F), cohort labels permuted across mice"""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    D = D.dropna(subset=[val]).copy()
    def fstats(d):
        a = sm.stats.anova_lm(smf.ols(f"{val} ~ C(cohort) * C(area)", d).fit(), typ=2)
        return a.loc["C(cohort)", "F"], a.loc["C(cohort):C(area)", "F"]
    f0 = fstats(D)
    mice = D.mouse_id.unique(); mc = D.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    idx = pd.Index(mice).get_indexer(D.mouse_id)
    null = []
    for _ in range(N_PERM):
        d = D.assign(cohort=rng.permutation(mc)[idx])
        try:
            null.append(fstats(d))
        except Exception:  # noqa: BLE001
            continue
    null = np.asarray(null)
    return [(1 + np.sum(null[:, j] >= f0[j])) / (1 + len(null)) for j in range(2)]


def change_panel(ax, items, R, row, ylab, xt):
    """items: list of (label, {cohort: (real (sessions x points), null (sessions x points))}); points relative to the first (= 0)"""
    nx = len(xt)
    for j, (lab, dd) in enumerate(items):
        off = j * (nx + 0.8)
        for c in COH:
            real, null = dd[c]
            for arr, ls, mfc in ((real, "-", COL[c]), (null, (0, (2.5, 1.5)), "white")):
                m, s = np.nanmean(arr, 0), np.array([H.sem(arr[:, i]) for i in range(arr.shape[1])])
                ax.errorbar(off + np.arange(nx), m, yerr=s, fmt="o", ls=ls, ms=2.8, color=COL[c], mfc=mfc, lw=1, capsize=0)
        ax.text(off + (nx - 1) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0, color="0.7", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([j * (nx + 0.8) + i for j in range(len(items)) for i in range(nx)]); ax.set_xticklabels(xt * len(items), fontsize=4.8)
    ax.set_ylabel(ylab)


def excess_panel(ax, D, cols, labs, R, row):
    for i, (col, lab) in enumerate(zip(cols, labs)):
        v = {c: D[D.cohort == c][col].to_numpy(float) for c in COH}
        for k, c in enumerate(COH):
            p = wil(v[c])
            ax.errorbar(i + (k - 0.5) * 0.3, np.nanmean(v[c]), H.sem(v[c]), fmt="o", ms=3.4, color=COL[c],
                        mfc=COL[c] if p < 0.05 else "white", lw=1.1, capsize=0)
            R.append(dict(row=row, panel="d", measure=col, cohort=c, n=int(np.isfinite(v[c]).sum()), mean=np.nanmean(v[c]), p_nonparam=p))
        pm = mwu(v["R+"], v["R-"])
        ax.text(i, 1.0, pn(pm), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.8)
        R.append(dict(row=row, panel="d", measure=col, cohort="R+ vs R-", p_nonparam=pm))
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=5); ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylabel("excess beyond session time\n(real − shifted, per neuron)")


def area_panel(ax, D, val, R, row, title, rng, AC):
    n = D.dropna(subset=[val]).groupby(["area", "cohort"]).size().unstack().fillna(0)
    areas = [a for a in n.index if min(n.loc[a].get("R+", 0), n.loc[a].get("R-", 0)) >= 3]
    areas = sorted(areas, key=lambda a: -n.loc[a].sum())
    Dq = D[D.area.isin(areas)]
    p_coh, p_int = anova_perm(Dq, val, rng)
    ps = []
    for i, a in enumerate(areas):
        v = {c: Dq[(Dq.area == a) & (Dq.cohort == c)][val].dropna().to_numpy(float) for c in COH}
        for k, c in enumerate(COH):
            p0 = wil(v[c])
            ax.errorbar(np.nanmean(v[c]), i + (k - 0.5) * 0.3, xerr=H.sem(v[c]), fmt="o", ms=2.8, color=COL[c],
                        mfc=COL[c] if p0 < 0.05 else "white", lw=0.9, capsize=0)
        ps.append(mwu(v["R+"], v["R-"]))
    q = bh(ps)
    for i, (a, p, qq) in enumerate(zip(areas, ps, q)):
        ax.text(1.02, i, f"{H.pnum(p)} (q {H.pnum(qq)})", transform=ax.get_yaxis_transform(), va="center", fontsize=4.4,
                weight="bold" if qq < 0.05 else "normal")
        R.append(dict(row=row, panel="e", measure=val, area=a, n_Rp=int(n.loc[a].get("R+", 0)), n_Rm=int(n.loc[a].get("R-", 0)),
                      p_nonparam=p, q_BH=qq))
    R.append(dict(row=row, panel="e", measure=val, area="ANOVA", p_cohort_perm=p_coh, p_cohort_x_area_perm=p_int))
    ax.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(len(areas))); ax.set_yticklabels([f"{a} ({int(n.loc[a]['R+'])}/{int(n.loc[a]['R-'])})" for a in areas], fontsize=4.8)
    for t, a in zip(ax.get_yticklabels(), areas):
        t.set_color(AC.get(a, "black"))
    ax.set_ylim(len(areas) - 0.5, -0.5)
    ax.set_xlabel("excess (per neuron)")
    ax.set_title(f"{title}\nANOVA: cohort {pn(p_coh)}, cohort × area {pn(p_int)}", fontsize=5.8)


def load_hm():
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked_harm.parquet")
    A = A[A.skipped_reason.isna()]; A = A[H.mouse_of(A).isin(H.learners())].copy()
    A["cohort"] = A.reward_group
    A["excess_W"], A["excess_A"], A["excess_WA"] = A.shift_excess_dWP, A.shift_excess_dAP, A.shift_excess_dWAP
    return A


def load_pl():
    S = pd.read_csv(PL / "sessions_areas.csv"); B = pd.read_csv(PL / "bins.csv")
    L = H.learners()
    S = S[(S.stage == "learning") & S.mouse_id.isin(L)].copy(); B = B[(B.stage == "learning") & B.mouse_id.isin(L)].copy()
    return S, B


def main():
    H.setup()
    rng = np.random.default_rng(0)
    AC = area_colors()
    R = []
    A = load_hm(); S, B = load_pl()
    fig = plt.figure(figsize=(11.6, 5.0))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1.25, 0.85, 0.12, 1.25], wspace=0.55, hspace=0.75, left=0.03, right=0.94, top=0.9, bottom=0.08)
    for r, (row, txt) in enumerate((("hit − miss axis", "stimulus onset"), ("reward-lick axis", "pre-lick"))):
        for j, t in enumerate(("a  schematic (author)", "b  example session (163 / 085)")):
            ax = fig.add_subplot(gs[r, j]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.85"); sp.set_visible(True)
            ax.text(0.5, 0.5, t, ha="center", va="center", fontsize=6, color="0.6", transform=ax.transAxes)
        fig.text(0.005, 0.93 - r * 0.47, f"{row} ({txt})", fontsize=8, weight="bold")
    # ---- row 1
    W = A[A.area == "All units"].dropna(subset=["shift_null_post_W_mean"])
    items = []
    for s_, lab in (("W", "passive whisker"), ("A", "passive auditory")):
        dd = {}
        for c in COH:
            d = W[W.cohort == c]
            real = np.c_[np.zeros(len(d)), (d[f"shift_real_post_{s_}"] - d[f"shift_real_pre_{s_}"]).to_numpy()]
            null = np.c_[np.zeros(len(d)), (d[f"shift_null_post_{s_}_mean"] - d[f"shift_null_pre_{s_}_mean"]).to_numpy()]
            dd[c] = (real, null)
        items.append((lab, dd))
    ax = fig.add_subplot(gs[0, 2]); change_panel(ax, items, R, "hit-miss", "projection on hit − miss axis\n(per neuron, relative to pre)", ["pre", "post"])
    ax.set_title("c  pre → post: real (solid) vs shifted-label axes (dashed)", fontsize=5.8, pad=12)
    ax = fig.add_subplot(gs[0, 3]); excess_panel(ax, W, ["excess_W", "excess_A", "excess_WA"], ["whisker", "auditory", "W − A"], R, "hit-miss")
    ax.set_title("d  beyond session time", fontsize=5.8, pad=12)
    ax = fig.add_subplot(gs[0, 5]); area_panel(ax, A[A.area != "All units"], "excess_WA", R, "hit-miss", "e  areas: whisker − auditory", rng, AC)
    # ---- row 2
    Sw = S[S.area == "whole_brain"]
    items = []
    for c_, lab in (("WH", "whisker hits"), ("SL", "spontaneous licks")):
        dd = {}
        for c in COH:
            b = B[(B.cls == c_) & (B.cohort == c)]
            real = b.pivot_table(index="session_id", columns="bin", values="real").to_numpy()
            null = b.pivot_table(index="session_id", columns="bin", values="null").to_numpy()
            dd[c] = (real - real[:, :1], null - null[:, :1])
        items.append((lab, dd))
    ax = fig.add_subplot(gs[1, 2]); change_panel(ax, items, R, "reward-lick", "projection on reward-lick CD\n(per neuron, relative to start)",
                                                 ["1", "2", "3", "4", "5"])
    ax.set_xlabel("fifth of the session", fontsize=5)
    ax.set_title("c  across the session: real CD (solid) vs shifted-label CDs (dashed)", fontsize=5.8, pad=12)
    ax = fig.add_subplot(gs[1, 3]); excess_panel(ax, Sw, ["excess_pu", "excessSL_pu", "excessC_pu"], ["WH", "SL", "WH − SL"], R, "reward-lick")
    ax.set_title("d  beyond session time (slopes)", fontsize=5.8, pad=12)
    ax = fig.add_subplot(gs[1, 5]); area_panel(ax, S[S.area != "whole_brain"], "excessC_pu", R, "reward-lick", "e  areas: WH − SL", rng, AC)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=COL["R+"], marker="o", ms=3), Line2D([], [], color=COL["R-"], marker="o", ms=3),
                Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls=(0, (2.5, 1.5)))],
               ["R+", "R−", "real axis", "shifted-label axes"], loc="upper right", ncol=4, frameon=False, fontsize=6)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIGDIR / f"165_two_axes_figure.{ext}", dpi=300)
    pd.DataFrame(R).to_csv(EA / "165_stats.csv", index=False)
    print(pd.DataFrame(R).round(4).to_string())
    print("ALL DONE", FIGDIR / "165_two_axes_figure.png")


if __name__ == "__main__":
    main()
