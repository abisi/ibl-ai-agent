"""Replication figure: Ranjbar-Slamloo & Kamigaki 2026 S1 -> M2 time-shifted CCA, paper data vs SSL (SSp-bfd vs pooled
MO-wM1 + MO-wM2, all stable units, PCA -> CCA, v3 maps). One page per trial type, multi-page PDF + one PNG per page.

Per page:
  A  mean baseline-subtracted CCV maps (delay x time): paper, SSL R+ / R- at learning and expert; one shared scale.
  B  S1-leading CCV over time (delays 0..+15 ms), one panel per cohort x stage; grey = paper (matching trial type).
  C  per-session readouts: in-sample CCV, 10-fold cross-validated CCV, feedforward minus feedback.

Run: python 005_replication_figure.py [--kind total]  ->  OUT/replication/replication_<kind>.pdf
"""
import argparse
import importlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import s1m2_lib as L

F4 = importlib.import_module("004_stats_figures")
COL = {"R+": "#00B400", "R-": "#C800C8"}
GROUPS = [("learning", "R+"), ("learning", "R-"), ("expert", "R+"), ("expert", "R-")]
PAIR = "SSp-bfd__wMO"
# SSL trial type -> (title, matching paper trial type or None)
TRIAL_TYPES = [("All", "All active trials", "All"),
               ("W_all", "Whisker trials (all outcomes)", "Go"),
               ("W_hit", "Whisker hits", "hit"),
               ("W_miss", "Whisker misses", "miss"),
               ("A_hit", "Auditory hits", None)]
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0})


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "p<1e-4"
    return "p=" + (f"{p:.2f}" if p >= 0.01 else f"{p:.1g}").lstrip("0")


def crop(m, T, D, tlim=(-0.15, 0.25), dlim=0.05):
    w, tm = np.abs(D) <= dlim + 1e-9, (T >= tlim[0] - 1e-9) & (T <= tlim[1] + 1e-9)
    return m[np.ix_(w, tm)], [T[tm][0] * 1e3, T[tm][-1] * 1e3, D[w][0] * 1e3, D[w][-1] * 1e3]


def mean_sem(x):
    x = np.asarray(x, float)
    return np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(np.sum(np.isfinite(x), 0).clip(1))


def page(cond, title, pc, kind, df_all, maps, traces, T, D, tr, ref_all):
    pk = "Total" if kind == "total" else "Noise"
    df = df_all[(df_all.pair == PAIR) & (df_all.condition == cond) & (df_all.kind == kind)]
    ref = ref_all[(ref_all["corr"] == pk) & (ref_all.condition == pc)] if pc else None
    fig = plt.figure(figsize=(7.4, 7.0))
    g1 = fig.add_gridspec(1, 5, wspace=0.55, left=0.08, right=0.9, top=0.89, bottom=0.71)
    g2 = fig.add_gridspec(1, 4, wspace=0.55, left=0.09, right=0.98, top=0.6, bottom=0.45)
    g3 = fig.add_gridspec(1, 3, wspace=0.6, left=0.09, right=0.98, top=0.33, bottom=0.16)

    # ---- A: maps, shared colour scale
    panels = []
    if pc:
        Tp = tr[f"{pk}_{pc}_t"]
        pm = tr[f"{pk}_{pc}_map_mean"]
        pm = pm - np.nanmean(pm[:, (Tp >= -0.25) & (Tp <= 0)], axis=1, keepdims=True)
        panels.append(("Paper, expert\n" + f"{pc} trials (n = 50)", *crop(pm, Tp, tr["delays"]), "k"))
    else:
        panels.append(("Paper: no matching\ntrial type", None, None, "0.4"))
    for st, co in GROUPS:
        sess = df[(df.stage == st) & (df.cohort == co)].session_id.unique()
        ms = [maps[(s, PAIR, cond, kind)] for s in sess if (s, PAIR, cond, kind) in maps]
        m = crop(np.nanmean(ms, axis=0), T, D) if ms else (None, None)
        panels.append((f"SSL {co}, {st}\n(n = {len(ms)} sessions)", *m, COL[co]))
    vals = np.concatenate([p[1].ravel() for p in panels if p[1] is not None])
    vmax = np.nanpercentile(np.abs(vals), 99)
    for j, (ttl, m, ext, c) in enumerate(panels):
        ax = fig.add_subplot(g1[0, j])
        ax.set_title(ttl, color=c)
        if m is None:
            ax.set_axis_off()
            continue
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext,
                       interpolation="nearest")
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_xticks([-100, 0, 100, 200])
        ax.set_xlabel("Time from stimulus (ms)")
        ax.set_ylabel("Delay (ms)")
    cax = fig.add_axes([0.915, 0.73, 0.01, 0.14])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("CCV minus\nbaseline", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)

    # ---- B: time courses, one panel per group
    ylo, yhi = np.inf, -np.inf
    axes_b = []
    for j, (st, co) in enumerate(GROUPS):
        ax = fig.add_subplot(g2[0, j])
        axes_b.append(ax)
        if pc:
            m, se = mean_sem(tr[f"{pk}_{pc}_FF"])
            ax.plot(Tp * 1e3, m, color="0.55", lw=0.9, label="paper")
            ax.fill_between(Tp * 1e3, m - se, m + se, color="0.55", alpha=.2, lw=0, edgecolor="none")
        sess = df[(df.stage == st) & (df.cohort == co)].session_id.unique()
        x = [traces[(s, PAIR, cond, kind)][0] for s in sess if (s, PAIR, cond, kind) in traces]
        if x:
            m, se = mean_sem(x)
            ax.plot(T * 1e3, m, color=COL[co], lw=1, label=f"SSL {co} {st}")
            ax.fill_between(T * 1e3, m - se, m + se, color=COL[co], alpha=.2, lw=0, edgecolor="none")
            sel = (T >= -0.25) & (T <= 0.4)
            ylo, yhi = min(ylo, np.nanmin((m - se)[sel])), max(yhi, np.nanmax((m + se)[sel]))
        ax.axvspan(0, 100, color="gold", alpha=.15, lw=0, edgecolor="none")
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_xlim(-250, 400)
        ax.set_xlabel("Time from stimulus (ms)")
        ax.set_ylabel("S1-leading CCV\nminus baseline")
        ax.set_title(f"SSL {co}, {st} (n = {len(x)})", color=COL[co])
        ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    if pc:
        m, se = mean_sem(tr[f"{pk}_{pc}_FF"])
        ylo, yhi = min(ylo, np.nanmin(m - se)), max(yhi, np.nanmax(m + se))
    pad = 0.08 * (yhi - ylo)
    for ax in axes_b:
        ax.set_ylim(ylo - pad, yhi + 3 * pad)

    # ---- C: per-session readouts
    rng = np.random.default_rng(0)
    specs = (("FF", "S1 -> M2 coupling, 0-100 ms\n(in-sample CCV minus baseline)", True),
             ("CV_FF", "S1 -> M2 coupling, 0-100 ms\n(cross-validated CCV minus baseline)", False),
             ("FF_minus_FB", "Feedforward minus feedback\n(S1 leading minus M2 leading)", True))
    for j, (col, lab, with_paper) in enumerate(specs):
        ax = fig.add_subplot(g3[0, j])
        xt, xl, xc = [], [], []
        if with_paper and pc:
            v = ref[ref.direction == "FF"].set_index("session_idx").value
            if col == "FF_minus_FB":
                v = v - ref[ref.direction == "FB"].set_index("session_idx").value
            ax.errorbar(0, v.mean(), v.std() / np.sqrt(len(v)), color="0.4", marker="s", ms=3.5, capsize=2, lw=0.8)
            xt.append(0); xl.append(f"paper\n\nn=50\n{fmt_p(F4.wil(v))}"); xc.append("0.3")
        for x0, (st, co) in enumerate(GROUPS, start=1):
            v = df[(df.stage == st) & (df.cohort == co)][col].dropna()
            if not len(v):
                continue
            ax.scatter(x0 - .15 + rng.uniform(-.08, .08, len(v)), v, s=4, c=COL[co], alpha=.45, lw=0)
            ax.errorbar(x0 + .15, v.mean(), v.std() / np.sqrt(len(v)), color=COL[co], capsize=2, lw=0.8,
                        marker="o", ms=3.5, mfc="white" if st == "learning" else COL[co])
            xt.append(x0); xc.append(COL[co])
            xl.append(f"{co}\n{'learn' if st == 'learning' else 'expert'}\nn={len(v)}\n{fmt_p(F4.wil(v))}")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=5)
        for t_, c_ in zip(ax.get_xticklabels(), xc):
            t_.set_color(c_)
        ax.set_xlim(-0.6 if (with_paper and pc) else 0.4, 4.6)
        ax.set_ylabel(lab)
    for y, l in ((0.925, "A"), (0.635, "B"), (0.365, "C")):
        fig.text(0.01, y, l, fontsize=9, fontweight="bold")
    fig.suptitle(f"{title}: S1 (SSp-bfd) -> M2 (MO-wM1 + MO-wM2) time-shifted CCA, {kind} coupling",
                 fontsize=8, y=0.985)
    cap = ("Method as Ranjbar-Slamloo & Kamigaki 2026 (PLoS Biol e3004095): 5-ms bins, 50-ms windows every 10 ms, M2 "
           "window shifted by the delay (delay > 0: S1 leads), first canonical correlation (CCV). A: maps baseline-"
           "subtracted per delay (-250..0 ms), shared colour scale. B, C: readout as their fig. 3, CCV averaged over "
           "delays 0..+15 ms minus its -250..0 ms mean, averaged over 0-100 ms (yellow). Paper values recomputed from "
           "their Zenodo maps (expert mice). SSL: all stable good + MUA units (>= 10 per area), each area reduced to "
           "its top 10 PCs before CCA (with all units in-sample CCA saturates at ~1); spikes in -10..+5 ms around every "
           "trial start removed (the diagonal band in A is where a window covers this gap). Session = unit of "
           "analysis, mean +- s.e.m. over sessions; p: Wilcoxon signed-rank vs 0, uncorrected. Open markers learning "
           "day (whisker day 0), filled expert days.")
    fig.text(0.07, 0.012, "\n".join(textwrap.wrap(cap, 190)), fontsize=5.2, va="bottom")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="total")
    a = ap.parse_args()
    df, maps, traces, T, D = F4.load()
    tr = np.load(L.OUT / "paper_reference" / "traces.npz")
    ref = pd.read_parquet(L.OUT / "paper_reference" / "per_session.parquet")
    out = L.OUT / "replication"
    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / f"replication_{a.kind}.pdf") as pdf:
        for cond, title, pc in TRIAL_TYPES:
            fig = page(cond, title, pc, a.kind, df, maps, traces, T, D, tr, ref)
            pdf.savefig(fig)
            fig.savefig(out / f"replication_{a.kind}_{cond}.png", dpi=300)
            plt.close(fig)
    df[(df.pair == PAIR) & (df.kind == a.kind)].to_csv(out / f"replication_{a.kind}_per_session.csv", index=False)
    print(out / f"replication_{a.kind}.pdf")


if __name__ == "__main__":
    main()
