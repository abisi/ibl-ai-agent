"""Publication figure of the communication analyses (user request 2026-11-09: "gather all in one place, report, and make
publication quality figure"). Reads the statistics written by 013 (subspace methods, one 016 run) and 015 (CD-projection
coupling, one 014 run) for one population and assembles one Nature-style figure (183 mm wide; ssl-paper-figures
paperstyle):
  a-c  communication over time, RRR held-out R^2 divided by the within-area ceiling, mean over each session's pairs, then
       mean +- s.e.m. over sessions per cohort; whisker lick, auditory hit, whisker no lick; bars: cohort cluster p < 0.05;
  d-f  the same for partial RRR (the rest of the recording regressed out; pCCA when the run has no pRRR);
  g    agreement between methods (Spearman of the early rise per session x pair x condition);
  h    share of pairs whose early rise on whisker licks differs from 0 within each cohort, per method;
  i    R+ minus R- early rise on whisker licks per ordered pair (RRR / ceiling); * Mann-Whitney p < 0.05, ** BH q < 0.05;
  j-k  CD-projection coupling (Chen et al. 2024): hit-miss CD, hits + misses pooled (stimulus-aligned) and pre-lick CD,
       auditory hits + spontaneous licks pooled (lick-aligned); solid r, dashed r | rest of recording; cluster bars;
  l    held-out d' of each CD per area group and cohort.
Output: OUT/report_figures/fig_communication_<comm-run>_<pop>.{pdf,svg,png,...} (paperstyle.save_figure) + panel CSVs.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-communication-methods-pilot/017_publication_figure.py
    [--comm-run comm_v3_seven] [--cd-run cd_coupling_seven] [--pop learners]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "ssl-paper-figures")]
import comm_lib as cl  # noqa: E402
import paperstyle as ps  # noqa: E402

_spec = importlib.util.spec_from_file_location("rrr_stats_v1", HERE / "011_rrr_timeresolved_stats.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

COL = {"R+": ps.COLORS["R+"], "R-": ps.COLORS["R-"]}
COHORTS = ["R+", "R-"]
SHORT = {**v1.SHORT, "Somatosensory cortex": "SSp", "Motor and frontal areas": "Frontal", "Striatum and pallidum": "Str",
         "Hippocampus": "HPC", "Midbrain": "MB", "Thalamus": "TH", "Auditory areas": "AUD"}
COND_LAB = {"W_lick": "Whisker hit", "A_hit": "Auditory hit", "W_nolick": "Whisker miss"}
METHOD_LAB = {"r2": "RRR", "cca": "CCA", "pcca": "pCCA", "prrr": "pRRR", "dr2": "ΔR²"}
CD_EPOCH_STIM, CD_EPOCH_LICK = (0.1, 0.2), (-0.1, 0.0)


def mean_sem(df, by="window_center", val="excess"):
    g = df.groupby(["cohort", by])[val]
    return g.mean().unstack(0), g.sem().unstack(0)


def pooled(sl, q, cond):
    """Per session mean over its pairs (sl already restricted to included cells), per window."""
    d = sl[(sl.quantity == q) & (sl.condition == cond) & ~sl.source.str.endswith("(within)")]
    return d.groupby(["session_id", "cohort", "window_center"], as_index=False).excess.mean()


def time_panel(ax, df, clus, q, cond, ylabel=None, transform=None, ls="-", band=True, xlab="Time from stimulus (s)"):
    m, s = mean_sem(df)
    f = transform or (lambda x: x)
    for c in COHORTS:
        if c not in m:
            continue
        ax.plot(m.index, f(m[c]), color=COL[c], ls=ls, lw=0.8 if ls == "-" else 0.6)
        if band:
            ax.fill_between(m.index, f(m[c] - s[c]), f(m[c] + s[c]), color=ps.light(COL[c], 0.7), lw=0, zorder=0)
    ax.axvline(0, color="0.5", lw=0.4, ls=":")
    ax.axhline(0, color="0.7", lw=0.4)
    if clus is not None and len(clus):
        sig = clus[(clus.quantity == q) & (clus.condition == cond) & (clus.p < 0.05)]
        y = ax.get_ylim()[1]
        for r in sig.itertuples():
            ax.plot([r.start - 0.05, r.end + 0.05], [y, y], color="k", lw=1.2, solid_capstyle="butt", clip_on=False)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlab)


def label(ax, letter):
    ax.text(-0.28, 1.12, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm-run", default="comm_v3_seven")
    ap.add_argument("--cd-run", default="cd_coupling_seven")
    ap.add_argument("--pop", default="learners")
    a = ap.parse_args()
    ps.set_nature_rc()
    rd, cd = cl.OUT / a.comm_run / a.pop, cl.OUT / a.cd_run / a.pop
    sl = pd.read_parquet(rd / "session_level.parquet")
    ep = pd.read_parquet(rd / "epochs_session.parquet")
    clus = pd.read_csv(rd / "stats_clusters.csv")
    clus_pool = clus[clus.source == "all"] if "source" in clus else clus
    tpairs = pd.read_csv(rd / "stats_pairs.csv")
    counts = pd.read_csv(rd / "method_counts.csv")
    csl = pd.read_parquet(cd / "session_level.parquet")
    cclus = pd.read_csv(cd / "stats_clusters.csv")
    dp = pd.read_csv(cd / "dprime.csv")
    partial = "prrr" if "prrr" in set(sl.quantity) else "pcca"
    partial_n = partial + "n" if partial + "n" in set(sl.quantity) else partial
    n_s = {c: sl[sl.cohort == c].session_id.nunique() for c in COHORTS}

    fig = plt.figure(figsize=(183 * ps.MM, 205 * ps.MM))
    gs = fig.add_gridspec(4, 3, left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.75, wspace=0.42)
    out_csv = {}
    row_axes = {0: [], 1: []}
    # a-c, d-f: subspace communication over time
    for j, cond in enumerate(["W_lick", "A_hit", "W_nolick"]):
        for i, (q, ylab) in enumerate((("r2n", "RRR R² / ceiling"),
                                       (partial_n, f"{METHOD_LAB[partial]} R² / ceiling" if partial == "prrr"
                                        else "pCCA cc1 / ceiling"))):
            ax = fig.add_subplot(gs[i, j])
            d = pooled(sl, q, cond)
            out_csv[f"{'abcdef'[3 * i + j]}_{q}_{cond}"] = d
            time_panel(ax, d, clus_pool, q, cond, ylab if j == 0 else None)
            ax.set_title(COND_LAB[cond] + (" (rest of brain removed)" if i == 1 else ""), fontsize=6)
            label(ax, "abcdef"[3 * i + j])
            row_axes[i].append(ax)
    for axs in row_axes.values():  # shared y within a row
        lo, hi = min(a_.get_ylim()[0] for a_ in axs), max(a_.get_ylim()[1] for a_ in axs)
        for a_ in axs:
            a_.set_ylim(lo, hi)
    # g: method agreement
    meths = [m for m in ["r2", "cca", "pcca", "prrr", "dr2"] if m in set(ep.quantity)]
    r = ep[ep.quantity.isin(meths) & ep.condition.isin(COND_LAB) & (ep.source != "all")].pivot_table(
        index=["session_id", "condition", "source", "target"], columns="quantity", values="rise_early")
    M = np.array([[stats.spearmanr(*r[[x, y]].dropna().to_numpy().T).statistic if x != y else 1 for y in meths]
                  for x in meths])
    out_csv["g_method_agreement"] = pd.DataFrame(M, index=meths, columns=meths)
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    for ii in range(len(meths)):
        for jj in range(len(meths)):
            ax.text(jj, ii, f"{M[ii, jj]:.2f}", ha="center", va="center", fontsize=5, color="w" if M[ii, jj] > 0.6 else "k")
    lab = [METHOD_LAB[m] for m in meths]
    ax.set_xticks(range(len(meths)), lab, rotation=45)
    ax.set_yticks(range(len(meths)), lab)
    ax.spines[:].set_visible(False)
    ax.set_title("Agreement between methods\n(Spearman ρ, early rise)", fontsize=6)
    label(ax, "g")
    # h: share of pairs with a significant early rise on whisker hits, per method
    ax = fig.add_subplot(gs[2, 1])
    sub = counts[(counts.condition == "W_lick") & (counts.variable == "rise_early") & counts.quantity.isin(meths)]
    sub = sub.set_index("quantity").reindex(meths)
    x = np.arange(len(meths))
    ax.bar(x - 0.18, sub.rplus_rise / sub.n_pairs, 0.34, color=COL["R+"], label=f"R+ (n = {n_s['R+']})")
    ax.bar(x + 0.18, sub.rminus_rise / sub.n_pairs, 0.34, color=COL["R-"], label=f"R- (n = {n_s['R-']})")
    for xx, (a_, b_, n_) in enumerate(zip(sub.rplus_rise, sub.rminus_rise, sub.n_pairs)):
        for dx_, v_ in ((-0.18, a_), (0.18, b_)):
            ax.text(xx + dx_, v_ / n_ + 0.02, f"{int(v_)}", ha="center", va="bottom", fontsize=4.5)
    ax.text(0.02, 0.98, f"of {int(sub.n_pairs.max())} pairs", transform=ax.transAxes, va="top", fontsize=5)
    ax.set_xticks(x, lab, rotation=45)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairs with early rise ≠ 0")
    ax.set_title("Whisker hit, per method", fontsize=6)
    ax.legend(loc="upper right")
    out_csv["h_method_counts"] = sub.reset_index()
    label(ax, "h")
    # i: R+ - R- early rise per pair (whisker hit, RRR / ceiling)
    ax = fig.add_subplot(gs[2, 2])
    t = tpairs[(tpairs.quantity == "r2n") & (tpairs.condition == "W_lick") & (tpairs.variable == "rise_early")]
    groups = v1._groups(t)
    Mi = np.full((len(groups), len(groups)), np.nan)
    for rr in t.itertuples():
        Mi[groups.index(rr.source), groups.index(rr.target)] = rr.diff
        mk = "**" if rr.mw_q < 0.05 else "*" if rr.mannwhitney_p < 0.05 else ""
        if mk:
            ax.text(groups.index(rr.target), groups.index(rr.source), mk, ha="center", va="center", fontsize=6)
    lim = np.nanpercentile(np.abs(Mi), 98) if np.isfinite(Mi).any() else 1  # a few pairs with 3-4 sessions saturate
    im = ax.imshow(Mi, cmap="PiYG", vmin=-lim, vmax=lim)
    im.cmap.set_bad("0.92")
    acol = v1.area_colors(groups)
    ax.set_xticks(range(len(groups)), [SHORT.get(g, g) for g in groups], rotation=60)
    ax.set_yticks(range(len(groups)), [SHORT.get(g, g) for g in groups])
    for tl, g in zip(list(ax.get_xticklabels()) + list(ax.get_yticklabels()), groups + groups):
        tl.set_color(acol[g])
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("R+ − R- early rise", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    ax.set_title("Whisker hit, RRR / ceiling", fontsize=6)
    out_csv["i_pair_diff"] = t
    label(ax, "i")
    # j-k: CD coupling
    for k, (cond, ttl, xlab) in enumerate((("hitmiss:hm_pooled", "Hit-miss axis, hits + misses", "Time from stimulus (s)"),
                                            ("prelick:ahsl_pooled", "Pre-lick axis, AH + SL", "Time from first lick (s)"))):
        ax = fig.add_subplot(gs[3, k])
        for q, ls in (("r", "-"), ("r_partial", "--")):
            d = csl[(csl.quantity == q) & (csl.condition == cond)].groupby(
                ["session_id", "cohort", "window_center"], as_index=False).excess.mean()
            out_csv[f"{'jk'[k]}_{q}"] = d
            time_panel(ax, d, cclus if q == "r" else None, q, cond, "Correlation across areas (r)" if k == 0 else None,
                       transform=np.tanh, ls=ls, band=(q == "r"), xlab=xlab)
        ax.set_title(ttl, fontsize=6)
        label(ax, "jk"[k])
    # l: d' per area
    ax = fig.add_subplot(gs[3, 2])
    dp = dp[dp.cohort.isin(COHORTS)]
    order = [g for g in v1._groups(dp.rename(columns={"area": "source"}).assign(target=dp.area)) if (dp.area == g).sum() >= 6]
    for k, cdn in enumerate(("hitmiss", "prelick")):
        for c, dx in (("R+", -0.12), ("R-", 0.12)):
            g = dp[(dp.cd == cdn) & (dp.cohort == c)].groupby("area").cd_dprime.agg(["mean", "sem"]).reindex(order)
            y = np.arange(len(order)) + (0.0 if k == 0 else 0.0)
            ax.errorbar(g["mean"], y + dx, xerr=g["sem"], fmt="o" if k == 1 else "s", ms=2.2, color=COL[c],
                        mfc=COL[c] if k == 1 else "white", elinewidth=0.6, mew=0.6, lw=0)
    ax.set_yticks(range(len(order)), [SHORT.get(g, g) for g in order])
    for tl, g in zip(ax.get_yticklabels(), order):
        tl.set_color(v1.area_colors(order)[g])
    ax.invert_yaxis()
    ax.axvline(0, color="0.7", lw=0.4)
    ax.set_xlabel("Held-out d' along the axis")
    ax.set_title("Axis strength (□ hit-miss, ● pre-lick)", fontsize=6)
    out_csv["l_dprime"] = dp
    label(ax, "l")
    fig.text(0.5, 0.985, f"Communication between area groups on the first whisker day ({a.pop}: R+ {n_s['R+']}, R- {n_s['R-']} "
             "sessions; R+ green, R- magenta)", ha="center", va="top", fontsize=7)
    od = cl.OUT / "report_figures"
    name = f"fig_communication_{a.comm_run}_{a.pop}"
    ps.save_figure(fig, od, name)
    (od / f"{name}_data").mkdir(parents=True, exist_ok=True)
    for k, v in out_csv.items():
        v.to_csv(od / f"{name}_data" / f"{k}.csv")
    print(od / name)


if __name__ == "__main__":
    main()
