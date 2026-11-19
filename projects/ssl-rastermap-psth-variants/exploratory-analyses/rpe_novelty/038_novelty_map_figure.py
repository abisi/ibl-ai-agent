"""Figure for step 5 (region map of the novelty / reward model, 037 map) + model-free exposure curves.

Row 1-2 (per window): forest plots across area groups of the recoverable quantities (parameter-recovery step):
  initial novelty amplitude (bN x (N_first - N_asymptote), z), reward weights bR (R+, R-) on RW value, shared-component
  weight lambda (coarsely recoverable), asymptote relative to the auditory response, and the held-out (leave-one-mouse-
  out) gain in within-mouse R^2 of the novelty model over nuisance-only. 95% CIs: mouse-level bootstrap (100x).
Row 3: model-free curves: whisker-trial response vs whisker exposure number (active trials, bins of 5) per cohort, with
  the auditory response vs auditory trial number as the familiar reference, for selected regions (late window;
  per-session mean then mean +- SEM across sessions).
Output: combined_results_ks4/_novelty_model/model/novelty_map.{png,pdf}
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m = importlib.import_module("029_roc_rpe_v2")
BASE = m.RES / "_novelty_model"
COH = {"R+": "#00B400", "R-": "#C800C8"}
CURVE_REGIONS = ["all", "Somatosensory-whisker", "Retrosplenial areas", "Motor areas", "Thalamus"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    M = pd.read_csv(BASE / "model" / "region_map.csv")
    order = (M[M.window == "late"].sort_values("initial_amplitude").region.tolist())
    fig = plt.figure(figsize=(7.2, 9.2))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.18, left=0.2, right=0.98, top=0.95, bottom=0.06)
    panels = [("initial_amplitude", "Initial novelty\namplitude (z)"), ("bR", "Reward weight bR\n(RW value)"),
              ("lam", "Shared weight λ\n(aud→whisker)"), ("asymptote_minus_auditory", "Asymptote −\nauditory (z)"),
              ("cv_gain_freq_shared_vs_nuis", "Held-out R² gain\nnovelty vs nuisance")]
    for r, win in enumerate(["early", "late"]):
        S = M[M.window == win].set_index("region").reindex(order)
        y = np.arange(len(order))
        for c, (key, lab) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if key == "bR":
                for k, (coh, col) in enumerate([("Rplus", COH["R+"]), ("Rminus", COH["R-"])]):
                    v, lo, hi = S[f"bR_{coh}"], S[f"bR_{coh}_lo"], S[f"bR_{coh}_hi"]
                    ax.errorbar(v, y + (k - 0.5) * 0.3, xerr=[v - lo, hi - v], fmt="o", ms=2.5, color=col, elinewidth=0.7,
                                label="R+" if k == 0 else "R−")
                ax.set_xlim(-1.5, 2.5)
                if r == 0:
                    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
            elif key == "cv_gain_freq_shared_vs_nuis":
                v = S[key]
                sig = (S["p_wilcoxon_freq_shared_vs_nuis"] < .05) & (S["p_ttest_freq_shared_vs_nuis"] < .05)
                ax.barh(y, v, color=np.where(sig, "k", "0.7"), height=0.6)
            else:
                v = S[key]
                if f"{key}_lo" in S:
                    lo, hi = S[f"{key}_lo"], S[f"{key}_hi"]
                    excl = (lo > 0) | (hi < 0)
                    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor="0.4", elinewidth=0.7)
                    ax.scatter(v, y, s=10, c=np.where(excl & (key != "lam"), "k", "0.6"), zorder=3)
                else:
                    ax.scatter(v, y, s=10, c="0.4")
            ax.axvline(0, color="k", lw=0.5)
            ax.set_yticks(y, order if c == 0 else [""] * len(order), fontsize=6); ax.invert_yaxis()
            if r == 0:
                ax.set_title(lab, fontsize=6.5)
            if c == 0:
                ax.text(-0.95, 1.02, f"{'ab'[r]}  {win} window ({'5-50' if win == 'early' else '50-150'} ms)",
                        transform=ax.transAxes, fontsize=7.5, weight="bold")
    # model-free exposure curves
    R = pd.read_parquet(BASE / "trial_region_table.parquet")
    C = pd.read_parquet(BASE / "trial_covariates.parquet")
    D = R[(R.level.isin(["area_group", "all"])) & (R.n_units >= 5)].merge(C, on=["session_id", "trial"])
    for c, reg in enumerate(CURVE_REGIONS):
        ax = fig.add_subplot(gs[2, c])
        g = D[D.region == reg]
        for coh, col in COH.items():
            for tt, idx, ls in [("whisker", "n_whisker_before", "-"), ("auditory", "n_auditory_before", ":")]:
                h = g[(g.cohort == coh) & (g.ttype == tt)].copy()
                if tt == "auditory":
                    h = h[h.aud_rank >= 5]; h["x"] = h[idx] - 5
                else:
                    h["x"] = h[idx]
                h = h[h.x < 100]; h["b"] = (h.x // 5).astype(int)
                ps = h.groupby(["session_id", "b"]).z_late.mean().unstack()
                mu, se = ps.mean(), ps.std() / np.sqrt(ps.notna().sum())
                ok = ps.notna().sum() >= 5
                x = (mu.index[ok] + 0.5) * 5
                ax.plot(x, mu[ok], color=col, ls=ls, lw=1, label=f"{coh.replace('-', chr(8722))} {tt}")
                ax.fill_between(x, (mu - se)[ok], (mu + se)[ok], color=col, alpha=0.15, lw=0)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_title(reg if reg != "all" else "all good units", fontsize=6.5)
        ax.set_xlabel("trial number of that type")
        if c == 0:
            ax.set_ylabel("late response (z)")
            ax.text(-0.45, 1.12, "c  model-free: response vs exposure (late window)", transform=ax.transAxes,
                    fontsize=7.5, weight="bold")
        if c == len(CURVE_REGIONS) - 1:
            ax.legend(frameon=False, fontsize=5, loc="upper right")
    for ext in ["png", "pdf"]:
        fig.savefig(BASE / "model" / f"novelty_map.{ext}", dpi=250)
    print("saved", BASE / "model" / "novelty_map.png")
    # ---- hits vs misses (is the R+/R- divergence explained by licking?)
    fig2, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True)
    fig2.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12, hspace=0.45, wspace=0.35)
    for r, reg in enumerate(["all", "Somatosensory-whisker"]):
        g = D[(D.region == reg) & (D.ttype == "whisker")].copy()
        g["b"] = (g.n_whisker_before // 10).astype(int)
        g = g[g.b < 10]
        for c, (win, outc) in enumerate([("z_early", 0), ("z_early", 1), ("z_late", 0), ("z_late", 1)]):
            ax = axs[r, c]
            for coh, col in COH.items():
                h = g[(g.cohort == coh) & (g.lick == outc)]
                ps = h.groupby(["session_id", "b"])[win].mean().unstack()
                mu, se = ps.mean(), ps.std() / np.sqrt(ps.notna().sum()); ok = ps.notna().sum() >= 5
                x = (mu.index[ok] + 0.5) * 10
                ax.errorbar(x, mu[ok], se[ok], color=col, marker="o", ms=2.5, lw=1, capsize=1.5,
                            label=coh.replace("-", "−"))
            ax.axhline(0, color="0.6", lw=0.4)
            ax.set_title(f"{reg if reg != 'all' else 'all good units'}\n{win[2:]} window, {'hits' if outc else 'misses (no lick)'}",
                         fontsize=6.5)
            if r == 1:
                ax.set_xlabel("whisker trial number")
            if c == 0:
                ax.set_ylabel("response (z)")
    axs[0, 0].legend(frameon=False, fontsize=6)
    fig2.suptitle("d  Whisker-trial responses split by outcome (bins of 10 trials; mean ± SEM across sessions)",
                  fontsize=8, x=0.02, ha="left")
    for ext in ["png", "pdf"]:
        fig2.savefig(BASE / "model" / f"novelty_hits_misses.{ext}", dpi=250)
    print("saved", BASE / "model" / "novelty_hits_misses.png")


if __name__ == "__main__":
    main()
