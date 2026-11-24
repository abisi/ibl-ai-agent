"""Figures of the connected-pairs CCA (010 tables). Output OUT/pairs/figures/:
- pairs_readouts.pdf (+ png per page): one page per trial type; columns = key readouts (hires CV source-leading,
  windows starting 5-15 ms; v3 CV source-leading 0-100 ms, artefact-free cells), rows = learning / expert; x = pairs
  ordered by Allen enrichment (label colour = class); mean +- s.e.m. over sessions per cohort; filled marker = Wilcoxon
  p < 0.05 vs 0; open grey-edged marker = cohort does not pass the >= 3 sessions at both stages rule.
- pairs_scaling.{png,pdf,svg}: pair-mean readout vs log10 Allen enrichment, learning, whisker trials and all trials, per
  cohort (OLS line + analytic 95 % band, solid if p < 0.05; Pearson r and p; Spearman in the table).
- pairs_hires_maps_<cond>.pdf: mean hires CCV maps (delay x window start, baseline-subtracted, grey = excluded cells),
  one panel per pair, pages R+ / R- learning.
Run (haas): ../../.venv/bin/python 011_pairs_figures.py
"""
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import pairs_lib as P

S10 = importlib.import_module("010_pairs_stats")
COL = {"R+": "#00B400", "R-": "#C800C8"}
CLS_COL = {"strong": "#1f4e8c", "weak": "#5b8fd1", "none": "#8a8a8a"}
TRIAL_TYPES = [("W_all", "Whisker trials (all outcomes)"), ("W_hit", "Whisker hits"), ("W_miss", "Whisker misses"),
               ("A_hit", "Auditory hits"), ("NS_all", "No-stimulus trials (negative control)"), ("All", "All active trials")]
RO_LAB = {"hr_cv_e0": "CV canonical correlation, source leads 0-15 ms\n(10-ms windows starting 5-15 ms; minus baseline)",
          "v3_CV_FF": "CV canonical correlation, source leads 0-15 ms\n(50-ms windows, 0-100 ms, artefact-free; minus baseline)"}
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 6, "legend.fontsize": 5.5, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0})


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "p<1e-4" if p < 1e-4 else "p=" + (f"{p:.2f}" if p >= 0.01 else f"{p:.1g}")


def pair_order(df):
    o = df.groupby("pair").allen_enrichment.first().sort_values(ascending=False)
    return list(o.index)


def readout_page(pdf, df, st, cond, title, order, cls):
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    d = df[(df.kind == "total") & (df.condition == cond)]
    x = np.arange(len(order))
    for j, ro in enumerate(S10.KEY):
        for i, stage in enumerate(("learning", "expert")):
            ax = axs[i, j]
            for k, coh in enumerate(("R+", "R-")):
                s = st[(st.readout == ro) & (st.condition == cond) & (st.stage == stage) & (st.cohort == coh)].set_index("pair")
                for xi, p in enumerate(order):
                    if p not in s.index or s.loc[p, "n_sessions"] == 0:
                        continue
                    r = s.loc[p]
                    xx = xi + (k - 0.5) * 0.3
                    ok = bool(r.cohort_passes)
                    filled = ok and np.isfinite(r.p) and r.p < 0.05
                    ax.errorbar(xx, r["mean"], yerr=r["sem"] if r.n_sessions > 1 else None, fmt="o", ms=3,
                                color=COL[coh] if ok else "0.6", mfc=COL[coh] if filled else "white", lw=0.7,
                                mew=0.7, label=f"{coh}" if xi == 0 else None)
            mm = d[d.stage == stage]
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_xticks(x, [f"{p.replace('__', '-')}" for p in order], rotation=60, ha="right")
            for t, p in zip(ax.get_xticklabels(), order):
                t.set_color(CLS_COL.get(cls[p], "k"))
            ax.set_ylabel(RO_LAB[ro], fontsize=5.5)
            n = mm.groupby("cohort").session_id.nunique().to_dict()
            ax.set_title(f"{stage.capitalize()} (sessions: R+ {n.get('R+', 0)}, R- {n.get('R-', 0)})")
            ax.set_xlim(-0.6, len(order) - 0.4)
            from matplotlib.lines import Line2D
            hs = [Line2D([], [], marker="o", ls="", color=COL[c], mfc=COL[c], ms=3, label=f"{c}, p<0.05 vs 0")
                  for c in COL] + [Line2D([], [], marker="o", ls="", color=COL["R+"], mfc="white", ms=3, label="n.s."),
                                   Line2D([], [], marker="o", ls="", color="0.6", mfc="white", ms=3,
                                          label="cohort < 3 sessions at a stage")]
            hs += [Line2D([], [], ls="", marker="s", color=v, label={"strong": "strong", "weak": "weak", "none": "no"}[k] + " projection") for k, v in CLS_COL.items()]
            if i == 0 and j == 1:
                fig.legend(handles=hs, frameon=False, loc="outside lower center", ncol=7)
    fig.suptitle(f"{title}: time-shifted CCA across area pairs (pairs ordered by Allen projection density; mean +- "
                 f"s.e.m. over sessions)", fontsize=7)
    pdf.savefig(fig)
    fig.savefig(P.FIGS / f"pairs_readouts_{cond}.png", dpi=300)
    plt.close(fig)


def regress(ax, x, y, color):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return
    r, p = stats.pearsonr(x, y)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    n = len(x)
    s_err = np.sqrt(np.sum((y - (a + b * x)) ** 2) / (n - 2))
    ci = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.plot(xs, a + b * xs, color=color, ls="-" if p < 0.05 else "--", lw=0.8)
    ax.fill_between(xs, a + b * xs - ci, a + b * xs + ci, color=color, alpha=0.12, lw=0, edgecolor="none")
    rho, prho = stats.spearmanr(x, y)
    return f"pairs: n = {n}, r = {r:.2f}, {fmt_p(p)}; rho = {rho:.2f}, {fmt_p(prho)}"


def scaling_fig(df, ss):
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 5.4), constrained_layout=True)
    d = df[(df.kind == "total") & (df.stage == "learning")]
    for i, ro in enumerate(S10.KEY):
        for j, (cond, coh) in enumerate([(c, k) for c in ("W_all", "All") for k in ("R+", "R-")]):
            ax = axs[i, j]
            g = d[(d.condition == cond) & (d.cohort == coh)]
            pm = g.groupby("pair").agg(y=(ro, "mean"), se=(ro, "sem"), x=("log_enrichment", "first"), n=(ro, "count"),
                                       cls=("class", "first"))
            pm = pm[pm.n >= 3]
            for p, r in pm.iterrows():
                ax.errorbar(r.x, r.y, yerr=r.se, fmt="o", ms=3, color=CLS_COL[r.cls], lw=0.6)
                ax.text(r.x, r.y, " " + p.replace("__", "-"), fontsize=3.5, color="0.45", va="center")
            txt = regress(ax, pm.x, pm.y, COL[coh]) or "pairs: < 4"
            q = ss[(ss.readout == ro) & (ss.condition == cond) & (ss.cohort == coh) & (ss.stage == "learning")]
            if len(q):
                q = q.iloc[0]
                txt += (f"\nwithin-session rho {q.mean_rho:.2f} (n = {int(q.n_sessions_rho)}, {fmt_p(q.p_rho_wilcoxon)})"
                        f"\nmixed model slope {q.get('lmm_slope', np.nan):.3f}, {fmt_p(q.get('lmm_p', np.nan))}")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.axvline(0, color="k", lw=0.4, ls=":")
            ax.set_xlabel("Allen projection density at target sites\n(log10, x brain average)", fontsize=5.5)
            ax.set_ylabel(RO_LAB[ro], fontsize=5)
            ax.set_title(f"{coh} learning, {dict(TRIAL_TYPES)[cond].lower()}\n{txt}", color=COL[coh], fontsize=5)
    fig.suptitle("Does source-leading coupling scale with projection strength? (pair means, >= 3 sessions; colour = "
                 "class: dark blue strong, light blue weak, grey none)", fontsize=7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(P.FIGS / f"pairs_scaling.{ext}", dpi=300)
    plt.close(fig)


def maps_pdf(df, extra, cond, order):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.82")
    with PdfPages(P.FIGS / f"pairs_hires_maps_{cond}.pdf") as pdf:
        for stage in ("learning", "expert"):
            for coh in ("R+", "R-"):
                fig, axs = plt.subplots(3, 6, figsize=(7.4, 4.6), constrained_layout=True)
                for ax, p in zip(axs.flat, order):
                    sess = df[(df.pair == p) & (df.stage == stage) & (df.cohort == coh) & (df.kind == "total")
                              & (df.condition == cond)].session_id.unique()
                    ms = [extra[(s, p, cond)] for s in sess if (s, p, cond) in extra]
                    ax.set_title(f"{p.replace('__', ' -> ')} (n={len(ms)})", fontsize=5.5)
                    if not ms:
                        ax.set_axis_off()
                        continue
                    _, _, ws, D = ms[0]
                    tm = (ws >= -0.05 - 1e-9) & (ws <= 0.1 + 1e-9)
                    m = np.nanmean([x[1] for x in ms], axis=0)[:, tm]
                    v = np.nanpercentile(np.abs(m), 99) or 1e-3
                    ax.imshow(np.ma.masked_invalid(m), aspect="auto", origin="lower", cmap=cmap, vmin=-v, vmax=v,
                              extent=[ws[tm][0] * 1e3, ws[tm][-1] * 1e3, D[0] * 1e3, D[-1] * 1e3], interpolation="nearest")
                    ax.axhline(0, color="k", lw=0.3)
                    ax.set_xlabel("Window start (ms)", fontsize=5)
                    ax.set_ylabel("Delay (ms; >0 source leads)", fontsize=5)
                    ax.text(0.98, 0.98, f"+-{v:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=4.5)
                for ax in list(axs.flat)[len(order):]:
                    ax.set_axis_off()
                fig.suptitle(f"{coh} {stage}, {dict(TRIAL_TYPES)[cond].lower()}: high-resolution CCV maps (1-ms bins, "
                             f"10-ms windows), baseline-subtracted per delay; grey = overlaps the artefact gap; colour "
                             f"scale per panel (+-99th percentile)", fontsize=6.5, color=COL[coh])
                pdf.savefig(fig)
                plt.close(fig)


def main():
    P.FIGS.mkdir(parents=True, exist_ok=True)
    df, extra = S10.load()
    st = pd.read_csv(P.TABLES / "stats.csv")
    ss = pd.read_csv(P.TABLES / "scaling_session.csv")
    order = pair_order(df)
    cls = df.groupby("pair")["class"].first().to_dict()
    with PdfPages(P.FIGS / "pairs_readouts.pdf") as pdf:
        for cond, title in TRIAL_TYPES:
            readout_page(pdf, df, st, cond, title, order, cls)
    scaling_fig(df, ss)
    for cond in ("W_all", "A_hit"):
        maps_pdf(df, extra, cond, order)
    print("figures in", P.FIGS)


if __name__ == "__main__":
    main()
