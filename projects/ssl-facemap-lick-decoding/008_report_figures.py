"""Facemap lick decoding, step 8 (haas): the report's main figures, from the existing result tables (no recomputation).

Reads (written by 003, 004, 006, 007; none is modified):
  analysis/decoding_per_session.csv        per-session held-out r / balanced accuracy / AUC of every feature set (003)
  analysis/learning_curve_correlations.csv per-session r and linear-shift z of the trial-wise measures vs the curve (003)
  roi/rois.csv, roi/previews/<session>.png automatic ROIs (004)
  roi/compare/per_session_deltas.csv       feature sets, real minus shift-null median (006)
  roi/compare/features_ranking.csv         single features, median (r - shift-null median) (007)
Figures (combined_results_ks4/ssl-facemap-lick-decoding/figures/, png + pdf):
  fig1_rois             automatic face ROIs (eye, snout, whisker pad, jaw) on example sessions of both cohorts
  fig2_decoding_auc     AUC of the upcoming whisker lick: base, DLC kpca, Facemap motion SVD, and each added to base;
                        added AUC over base (Facemap and kpca), and Facemap vs kpca gain per session
  fig3_shift_null       decoding r relative to its linear-shift null (z), per feature set
  fig4_learning_curve   pre-stimulus motion energy and Facemap lick probability vs the exact learning curve (shift z)
  fig5_roi_vs_dlc       ROI Facemap vs DLC feature sets, real minus shift-null median (lick, learning curve)
  fig6_single_features  single DLC and ROI features, median (r - shift-null median) per stage, cohort and target
Run on haas: nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 008_report_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-facemap-lick-decoding")
OUT = BASE / "figures"
COLORS = {"R+": "#00B400", "R-": "#C800C8"}
STAGES = {"day0": "day 0", "expert": "expert (day > 0)"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def stage(day):
    return np.where(np.asarray(day) == 0, "day0", "expert")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def strip(ax, x, v, color, rng, w=0.12):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    ax.scatter(x + rng.uniform(-w, w, len(v)), v, s=6, color=color, alpha=0.45, linewidths=0)
    if len(v):
        ax.plot([x - 1.6 * w, x + 1.6 * w], [np.median(v)] * 2, color=color, lw=2.2)


def cohort_legend(ax, loc="best"):
    for c, col in COLORS.items():
        ax.plot([], [], color=col, lw=2, label=f"{c} (bar = median)")
    ax.legend(frameon=False, fontsize=7, loc=loc)


def fig1():
    """Three sessions per cohort, evenly spaced through the sorted session list (deterministic)."""
    rois = pd.read_csv(BASE / "roi" / "rois.csv").set_index("session_id")
    D = pd.read_csv(BASE / "analysis" / "decoding_per_session.csv").set_index("session_id")
    ex = []
    for coh in COLORS:
        s = sorted(x for x in D.index[D.cohort == coh] if (BASE / "roi" / "previews" / f"{x}.png").exists())
        ex += [(s[i], coh) for i in np.linspace(0, len(s) - 1, 3).astype(int)]
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    for ax, (sid, coh) in zip(axs.flat, ex):
        ax.imshow(plt.imread(BASE / "roi" / "previews" / f"{sid}.png"))
        fb = ", whisker fallback" if bool(rois.loc[sid].get("whisker_fallback", False)) else ""
        ax.set_title(f"{sid} ({coh}{fb})", fontsize=7, color=COLORS[coh])
        ax.axis("off")
    save(fig, "fig1_rois")


def fig2():
    D = pd.read_csv(BASE / "analysis" / "decoding_per_session.csv").copy()
    D["stage"] = stage(D.day)
    models = [("base", "base\n(time+history)"), ("kpca_mean_1s", "DLC kpca"), ("fm_mot_1s", "Facemap\nmotion SVD"),
              ("base+kpca_mean_1s", "base +\nDLC kpca"), ("base+fm_mot_1s", "base +\nFacemap")]
    D["gain_fm"] = D["auc_base+fm_mot_1s"] - D.auc_base
    D["gain_kpca"] = D["auc_base+kpca_mean_1s"] - D.auc_base
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])
    rng = np.random.default_rng(0)
    for j, st in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, j])
        g = D[D.stage == st]
        for i, (m, _) in enumerate(models):
            for q, coh in enumerate(COLORS):
                strip(ax, i + (q - 0.5) * 0.36, g[g.cohort == coh][f"auc_{m}"], COLORS[coh], rng)
        ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([lab for _, lab in models], fontsize=7)
        ax.set_ylabel("AUC, held-out trials (blocked 5-fold CV)")
        n = {c: int(g[g.cohort == c].auc_base.notna().sum()) for c in COLORS}
        ax.set_title(f"{chr(97 + j)}  {STAGES[st]}: R+ n = {n['R+']}, R- n = {n['R-']} sessions", loc="left")
        if j == 0:
            cohort_legend(ax, "lower right")
    ax = fig.add_subplot(gs[1, 0:2])
    xt, xl = [], []
    for j, st in enumerate(STAGES):
        for i, (col, lab) in enumerate((("gain_fm", "Facemap"), ("gain_kpca", "DLC kpca"))):
            x0 = j * 2.6 + i * 1.1
            for q, coh in enumerate(COLORS):
                g = D[(D.stage == st) & (D.cohort == coh)]
                strip(ax, x0 + (q - 0.5) * 0.4, g[col], COLORS[coh], rng, w=0.13)
            xt.append(x0)
            xl.append(f"{STAGES[st]}\n{lab}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel("added AUC: AUC(base + set) - AUC(base)")
    ax.set_title("c  Added value of face features over time + history", loc="left")
    ax = fig.add_subplot(gs[1, 2])
    for coh in COLORS:
        g = D[(D.stage == "expert") & (D.cohort == coh)]
        ax.scatter(g.gain_kpca, g.gain_fm, s=8, color=COLORS[coh], alpha=0.6, linewidths=0, label=coh)
    lim = [np.nanmin(D[["gain_fm", "gain_kpca"]].values) - 0.02, np.nanmax(D[["gain_fm", "gain_kpca"]].values) + 0.02]
    ax.plot(lim, lim, color="k", lw=0.6)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlabel("added AUC, DLC kpca")
    ax.set_ylabel("added AUC, Facemap")
    ax.set_title("d  expert sessions, paired", loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_decoding_auc")


def fig3():
    D = pd.read_csv(BASE / "analysis" / "decoding_per_session.csv").copy()
    D["stage"] = stage(D.day)
    keys = [("time", "time"), ("base", "base"), ("kpca_mean_1s", "DLC kpca"), ("fm_mot_1s", "Facemap"),
            ("base+kpca_mean_1s", "base +\nDLC kpca"), ("base+fm_mot_1s", "base +\nFacemap")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, st in zip(axs, STAGES):
        g = D[D.stage == st]
        for i, (k, _) in enumerate(keys):
            for q, coh in enumerate(COLORS):
                strip(ax, i + (q - 0.5) * 0.36, g[g.cohort == coh][f"r_null_z_{k}"], COLORS[coh], rng)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(1.645, color="0.5", lw=0.6, ls=":")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab for _, lab in keys], fontsize=7)
        ax.set_title(STAGES[st], loc="left")
    axs[0].set_ylabel("held-out r vs linear-shift null (z)")
    cohort_legend(axs[0], "upper left")
    fig.tight_layout()
    save(fig, "fig3_shift_null")


def fig4():
    C = pd.read_csv(BASE / "analysis" / "learning_curve_correlations.csv")
    C = C[C.curve == "p_mean"]
    C["stage"] = stage(C.day)
    meas = [("motion_energy_1s", "pre-stimulus motion energy (-1..0 s)"),
            ("lick_proba_fm_mot_1s", "cross-validated P(lick), Facemap motion SVD")]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (m, lab) in zip(axs, meas):
        g = C[C.measure == m]
        for j, st in enumerate(STAGES):
            for q, coh in enumerate(COLORS):
                strip(ax, j + (q - 0.5) * 0.4, g[(g.stage == st) & (g.cohort == coh)].z_shift, COLORS[coh], rng, w=0.14)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(list(STAGES.values()))
        ax.set_title(lab, loc="left", fontsize=8)
    axs[0].set_ylabel("r(measure, learning curve p_mean)\nvs linear-shift null (z)")
    cohort_legend(axs[0], "lower left")
    fig.tight_layout()
    save(fig, "fig4_learning_curve")


def fig5():
    Dd = pd.read_csv(BASE / "roi" / "compare" / "per_session_deltas.csv")
    Dd["stage"] = stage(Dd.day)
    sets = [("dlc_bp4_1s", "DLC 4 parts"), ("dlc_kpca_1s", "DLC kpca"), ("roi_energy_1s", "ROI energy"),
            ("roi_svd_1s", "ROI motion SVD"), ("full_svd_1s", "full-frame SVD")]
    mets = [("lick_bacc", "upcoming lick: balanced accuracy\n(real - shift-null median)"),
            ("curve_pmean", "learning curve p_mean: in-sample R2\n(real - shift-null median)")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))
    rng = np.random.default_rng(0)
    for i, (met, lab) in enumerate(mets):
        for j, st in enumerate(STAGES):
            ax = axs[i, j]
            g = Dd[Dd.stage == st]
            for k, (s, _) in enumerate(sets):
                for q, coh in enumerate(COLORS):
                    strip(ax, k + (q - 0.5) * 0.36, g[g.cohort == coh][f"d_{met}_{s}"], COLORS[coh], rng)
            ax.axvspan(-0.5, 1.5, color="0.93", zorder=-1)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xticks(range(len(sets)))
            ax.set_xticklabels([lab_ for _, lab_ in sets], fontsize=7)
            ax.set_ylabel(lab, fontsize=7)
            ax.set_title(f"{chr(97 + 2 * i + j)}  {STAGES[st]} (n = {g.session_id.nunique()} sessions; grey = DLC)",
                         loc="left")
    cohort_legend(axs[0, 0], "upper right")
    fig.tight_layout()
    save(fig, "fig5_roi_vs_dlc")


def fig6():
    R = pd.read_csv(BASE / "roi" / "compare" / "features_ranking.csv")
    order = []
    for w in ("1s", "0.2s"):
        order += [f"dlc_{b}_{k}_{w}" for b in ("jaw", "nose", "whisker", "pupil") for k in ("posture", "movement")]
        order += [f"roi_{r}_energy_{w}" for r in ("eye", "jaw", "whisker", "snout")]
    order = [o for o in order if o in set(R.feature)]
    cols = [(t, st, c) for t in ("lick", "pmean") for st in STAGES for c in COLORS]
    M = np.full((len(order), len(cols)), np.nan)
    for j, (t, st, c) in enumerate(cols):
        g = R[(R.target == t) & (R.stage == st) & (R.cohort == c)].set_index("feature").median_delta
        M[:, j] = g.reindex(order).to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 8))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="w" if abs(M[i, j]) > 0.6 * lim else "k")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    for i, o in enumerate(order):
        ax.get_yticklabels()[i].set_color("#b8860b" if o.startswith("roi_") else "k")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{'lick' if t == 'lick' else 'p_mean'}\n{'day 0' if st == 'day0' else 'expert'}\n{c}"
                        for t, st, c in cols], fontsize=6.5)
    ax.axvline(3.5, color="k", lw=1)
    ax.axhline(len(order) / 2 - 0.5, color="k", lw=0.6)
    fig.colorbar(im, ax=ax, shrink=0.5, label="median over sessions of r - shift-null median")
    fig.tight_layout()
    save(fig, "fig6_single_features")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()


if __name__ == "__main__":
    main()
