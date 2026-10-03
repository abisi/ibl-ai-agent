"""COSYNE summary (5 panels): in R+ mice, convergence of whisker hits (WH) toward auditory hits (AH) begins within the
learning day and continues across days; R- diverge on day 0. Reference: spontaneous licks (SL).

  a  First-lick-aligned PSTHs (004): rows R+ / R-, columns day 0 early half, day 0 late half, expert (both halves);
     WH, AH, SL; rate minus baseline, mean over units per session, mean +- s.e.m. over sessions.
  b  Trial level (002): WH - SL score on the session's cross-validated SL -> AH axis (0 = SL, 1 = AH - SL) across the
     learning-day session (5 bins), R+ vs R-; expert levels (session mean) at the right.
  c  Day-0 divergence (002): per-session slope of WH relative to SL across the session, R+ vs R- (MWU, Welch, mouse-level
     cohort permutation).
  d  Common footing (003, 4 events per class, 150 units, phase-matched halves): whole-session decoder readout
     P(AH | WH) - P(AH | SL) - chance per epoch (day 0 early / late, expert early / late), R+ and R-, hierarchical-bootstrap
     95% CI.
  e  Contrasts of d: within-day, across-day (early phase) and carry-over (expert early - day 0 late), per cohort, with the
     R+ vs R- permutation p.
Output: combined_results_ks4/_within_day<TAG>/cosyne/COSYNE_convergence_timeline.{png,pdf,svg} + stats csv.
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
BASE = m51.RES / f"_within_day{m51.TAG}"
OUT = BASE / "cosyne"
COH, CL, CLAB = m62.COH, m62.CL, m62.CLAB
EPOCH_DIR = "epochs_n4"
rng = np.random.default_rng(0)


def main():
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with plt.rc_context({"font.size": 5.5, "axes.titlesize": 5.6, "axes.labelsize": 5.4, "xtick.labelsize": 4.9,
                         "ytick.labelsize": 4.9, "legend.fontsize": 4.7}):
        fig = plt.figure(figsize=(m62.W_IN, 4.9))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.6], wspace=0.22, left=0.07, right=0.985, top=0.88, bottom=0.1)
        # a: PSTH grid
        ga = outer[0].subgridspec(2, 3, hspace=0.55, wspace=0.12)
        z = np.load(BASE / "psth_halves" / "psth_halves.npz", allow_pickle=True)
        S = pd.read_csv(BASE / "psth_halves" / "sessions.csv")
        P, Nev, edges = z["psth"], z["n"], z["edges"]
        tc = (edges[:-1] + np.diff(edges) / 2) * 1e3
        sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")
        axs_a = np.empty((2, 3), object)
        for i, c in enumerate(["R+", "R-"]):
            for j, (stg, halves, ttl) in enumerate([("learning", [0], "Day 0, early"), ("learning", [1], "Day 0, late"),
                                                    ("expert", [0, 1], "Expert")]):
                ax = fig.add_subplot(ga[i, j]); axs_a[i, j] = ax
                msk = ((S.cohort == c) & (S.stage == stg)).to_numpy()
                ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
                for k, cl in enumerate(m51.CLASSES):
                    ok = msk & (Nev[:, k, halves].min(1) >= 3)
                    A = np.array([sm(np.nanmean(p[halves], 0)) for p in P[ok, k]])
                    if len(A) < 2:
                        continue
                    mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(tc, mu - se, mu + se, color=CL[cl], alpha=0.25, lw=0)
                    ax.plot(tc, mu, color=CL[cl], lw=0.85, label=CLAB[cl])
                ax.set_xlim(-400, 200); ax.set_xticks([-400, -200, 0, 200]); ax.set_xticklabels(["−400", "", "0", ""])
                ax.set_title(f"{c.replace('-', '−')} {ttl} ({int(msk.sum())})", color=COH[c], fontsize=5.0, pad=2)
                if j == 0:
                    ax.set_ylabel("Rate − baseline (Hz)", fontsize=5)
                if i == 0 and j == 0:
                    ax.legend(frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.05, fontsize=4.3)
        yl = (min(a.get_ylim()[0] for a in axs_a.ravel()), max(a.get_ylim()[1] for a in axs_a.ravel()))
        for i in range(2):
            for j in range(3):
                axs_a[i, j].set_ylim(*yl)
                if j:
                    axs_a[i, j].set_yticklabels([])
        p0 = axs_a[1, 0].get_position(); p2 = axs_a[1, 2].get_position()
        fig.text((p0.x0 + p2.x1) / 2, p0.y0 - 0.075, "Time from first lick (ms); mean ± s.e.m. over sessions", ha="center", fontsize=5)
        # right block: b, c (top), d, e (bottom)
        gr = outer[1].subgridspec(2, 2, hspace=0.75, wspace=0.45, width_ratios=[1.35, 1])
        TR = pd.read_csv(BASE / "slopes" / "trial_slopes_trajectories.csv")
        D2 = pd.read_csv(BASE / "slopes" / "trial_slopes_sessions.csv")
        md = TR[TR.axis == "md"].pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "bin"], columns="cls",
                                             values="score").reset_index()
        md["rel"] = md["WH"] - md["FA"]
        xb = (np.arange(5) + 0.5) / 5
        ax_b = fig.add_subplot(gr[0, 0])
        for c in ["R+", "R-"]:
            q = md[(md.cohort == c) & (md.stage == "learning")].groupby("bin").rel.agg(["mean", "sem"]).reindex(range(5))
            ax_b.fill_between(xb, q["mean"] - q["sem"], q["mean"] + q["sem"], color=COH[c], alpha=0.15, lw=0)
            ax_b.plot(xb, q["mean"], color=COH[c], lw=1.2, marker="o", ms=2.5, mfc="white",
                      label=f"{c.replace('-', '−')} day 0 ({md[(md.cohort == c) & (md.stage == 'learning')].session_id.nunique()})")
            e = md[(md.cohort == c) & (md.stage == "expert")].groupby("session_id").rel.mean()
            ax_b.errorbar(1.12, e.mean(), e.sem(), fmt="o", ms=4, color=COH[c], capsize=0, lw=1.0, clip_on=False)
        ax_b.text(1.12, 1.0, "expert", transform=ax_b.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.6)
        ax_b.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax_b.set_xlim(0, 1.2); ax_b.set_xticks([0, 0.5, 1])
        ax_b.set_xlabel("Time in the learning-day session (normalised)")
        ax_b.set_ylabel("WH − SL score\n(session SL → AH axis)")
        ax_b.set_title("Single trials: within day 0, WH move toward AH\nin R+ and back to SL in R−", fontsize=5.3)
        ax_b.legend(frameon=False, loc="upper left")
        # c: day-0 relative slopes
        ax_c = fig.add_subplot(gr[0, 1])
        col = "md_WH-FA_slope"
        d0 = D2[D2.stage == "learning"].dropna(subset=[col])
        for k_, c in enumerate(["R+", "R-"]):
            v = d0[d0.cohort == c][col]
            ax_c.scatter(k_ + rng.uniform(-0.12, 0.12, len(v)), v, s=5, color=COH[c], alpha=0.5, lw=0)
            ax_c.errorbar(k_ + 0.27, v.mean(), v.sem(), fmt="o", ms=4, color=COH[c], mfc="white", capsize=0, lw=1.0)
            rows.append(dict(panel="c", cohort=c, mean=v.mean(), sem=v.sem(), n=len(v), p_wilcoxon=stats.wilcoxon(v).pvalue))
        a_, b_ = d0[d0.cohort == "R+"][col], d0[d0.cohort == "R-"][col]
        _, pp = m001.perm_interaction(d0.assign(stage="learning"), col, kind="learning")
        pm, pw = stats.mannwhitneyu(a_, b_).pvalue, stats.ttest_ind(a_, b_, equal_var=False).pvalue
        rows.append(dict(panel="c", cohort="R+ vs R-", p_MWU=pm, p_Welch=pw, p_perm=pp))
        lo, hi = np.nanpercentile(d0[col], [1, 99])
        ax_c.set_ylim(lo - 0.2 * (hi - lo), hi + 0.35 * (hi - lo))
        ax_c.plot([0, 0, 1, 1], [hi + 0.12 * (hi - lo), hi + 0.17 * (hi - lo), hi + 0.17 * (hi - lo), hi + 0.12 * (hi - lo)], color="0.2", lw=0.6)
        ax_c.text(0.5, hi + 0.19 * (hi - lo), m62.fmt_p(pp), ha="center", fontsize=4.8)
        ax_c.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax_c.set_xticks([0, 1], ["R+", "R−"]); ax_c.set_xlim(-0.5, 1.6)
        for t, c in zip(ax_c.get_xticklabels(), ["R+", "R-"]):
            t.set_color(COH[c])
        ax_c.set_ylabel("Day-0 slope of WH − SL\n(score per session)")
        ax_c.set_title("Day 0: R+ and R− diverge\n(mouse permutation)", fontsize=5.3)
        # d / e: common footing epochs (003)
        E = pd.read_csv(BASE / EPOCH_DIR / "all" / "epoch_contrasts.csv")
        meas = "dec"
        ax_d = fig.add_subplot(gr[1, 0])
        xe = {"L-early": 0, "L-late": 1, "E-early": 2.4, "E-late": 3.4}
        for k_, c in enumerate(["R+", "R-"]):
            q = E[(E.measure == meas) & (E.cohort == c) & (E.kind == "epoch")].set_index("name").reindex(list(xe))
            dx = (k_ - 0.5) * 0.16
            for seg in (("L-early", "L-late"), ("E-early", "E-late")):
                ax_d.plot([xe[s_] + dx for s_ in seg], q.loc[list(seg), "value"], color=COH[c], lw=1.1)
            ax_d.plot([xe["L-late"] + dx, xe["E-early"] + dx], q.loc[["L-late", "E-early"], "value"], color=COH[c], lw=0.7, ls=(0, (2, 2)))
            for ep, x in xe.items():
                ax_d.errorbar(x + dx, q.loc[ep, "value"], [[q.loc[ep, "value"] - q.loc[ep, "lo"]], [q.loc[ep, "hi"] - q.loc[ep, "value"]]],
                              fmt="o", ms=3.4, color=COH[c], mfc="white" if ep.startswith("L") else COH[c], lw=0.9, capsize=0,
                              label=c.replace("-", "−") if ep == "L-early" else None)
            for ep in xe:
                rows.append(dict(panel="d", cohort=c, epoch=ep, value=q.loc[ep, "value"], lo=q.loc[ep, "lo"], hi=q.loc[ep, "hi"]))
        ax_d.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax_d.set_xticks(list(xe.values()), ["early", "late", "early", "late"])
        for x_, s_ in ((0.5, "Day 0"), (2.9, "Expert")):
            ax_d.annotate(s_, xy=(x_, 0), xycoords=ax_d.get_xaxis_transform(), xytext=(0, -11), textcoords="offset points",
                          ha="center", va="top", fontsize=5)
        ax_d.set_ylabel("P(AH | WH) − P(AH | SL) − chance\n(whole-session decoder)")
        ax_d.set_title("Decoder readout per session half\n(phase-matched; 4 events / class; 95% CI)", fontsize=5.3)
        ax_d.legend(frameon=False, loc="upper left")
        # e: contrasts
        ax_e = fig.add_subplot(gr[1, 1])
        names = ["within-day", "across-day (early)", "carry-over"]
        labs = ["within\nday 0", "across\ndays", "carry-\nover"]
        for k_, c in enumerate(["R+", "R-"]):
            q = E[(E.measure == meas) & (E.cohort == c) & (E.kind == "contrast")].set_index("name").reindex(names)
            x = np.arange(len(names)) + (k_ - 0.5) * 0.3
            for xi, (_, r) in zip(x, q.iterrows()):
                ax_e.errorbar(xi, r.value, [[r.value - r.lo], [r.hi - r.value]], fmt="o", ms=3.4, color=COH[c], lw=0.9, capsize=0,
                              mfc=COH[c] if (r.p_boot if np.isfinite(r.p_boot) else 1) < 0.05 else "white")
            for n_, (_, r) in zip(names, q.iterrows()):
                rows.append(dict(panel="e", cohort=c, contrast=n_, value=r.value, lo=r.lo, hi=r.hi, p_boot=r.p_boot))
        pp = E[(E.measure == meas) & (E.cohort == "R+ - R-")].set_index("name").reindex(names)
        for i, n_ in enumerate(names):
            ax_e.text(i, 1.01, m62.fmt_p(pp.loc[n_, "p_perm"]).replace("p = ", "").replace("p < ", "<"),
                      transform=ax_e.get_xaxis_transform(), ha="center", fontsize=4.5)
            rows.append(dict(panel="e", cohort="R+ - R-", contrast=n_, value=pp.loc[n_, "value"], p_perm=pp.loc[n_, "p_perm"]))
        ax_e.axhline(0, color="0.3", lw=0.5)
        ax_e.set_xticks(range(len(names)), labs, fontsize=4.8)
        ax_e.set_ylabel("Change in decoder readout")
        ax_e.set_title("R+ converges within and across days\n(filled: p < 0.05; top: R+ vs R−)", fontsize=5.3, pad=8)
        m62.letter_row(fig, [axs_a[0, 0]], "a", dx_in=0.33, dy_in=0.15)
        m62.letter_row(fig, [ax_b, ax_c], "bc", dx_in=0.4, dy_in=0.25)
        m62.letter_row(fig, [ax_d, ax_e], "de", dx_in=0.4, dy_in=0.25)
        fig.suptitle("In R+ mice, whisker hits converge toward auditory hits within the learning day and across days; "
                     "R− diverge on day 0", x=0.07, y=0.995, ha="left", fontsize=6.9, weight="bold")
        m62.save(fig, OUT, "COSYNE_convergence_timeline"); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "COSYNE_convergence_timeline_stats.csv", index=False)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
