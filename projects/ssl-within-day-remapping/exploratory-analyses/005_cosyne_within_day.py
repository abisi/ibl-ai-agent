"""COSYNE summary (5 panels): R- whisker-hit (WH) activity changes more within the learning day than within expert sessions.

  a  First-lick-aligned PSTHs of R- sessions (004): day 0 early / late halves and expert early / late halves; WH, AH, SL;
     rate minus baseline, mean over units per session, mean +- s.e.m. over sessions.
  b  Behaviour: whisker-hit rate across the session (002 trajectories), R- day 0 vs expert (R+ day 0 for reference).
  c  Trial level (002): WH - SL score on the session's SL -> AH axis across the session (5 bins), R- day 0 vs expert
     (R+ day 0 for reference); mean +- s.e.m. over sessions.
  d  Epoch means on a common footing (003, 4 events per class): distance difference normalised by the session's d(AH, SL),
     day 0 early / late and expert early / late, R- (R+ for reference); hierarchical-bootstrap 95% CI.
  e  Size of the within-session change |late - early| per session (003), day 0 vs expert, R- (R+ for reference), for the
     position and the normalised distance; p = hierarchical bootstrap (mice, then sessions) of the day 0 - expert difference
     and Mann-Whitney U.
Output: combined_results_ks4/_within_day<TAG>/cosyne/COSYNE_within_day.{png,pdf,svg} + stats csv.
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
BASE = m51.RES / f"_within_day{m51.TAG}"
OUT = BASE / "cosyne"
COH, CL, CLAB = m62.COH, m62.CL, m62.CLAB
STAGE_LS = {"learning": "-", "expert": (0, (2.5, 1.5))}
rng = np.random.default_rng(0)


def hb_mean(df, col, B=4000):
    g = {k: v[col].to_numpy() for k, v in df.groupby("mouse_id")}
    mice = list(g)
    return np.array([np.mean([rng.choice(g[k], len(g[k])).mean() for k in rng.choice(mice, len(mice))]) for _ in range(B)])


def main():
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with plt.rc_context({"font.size": 5.5, "axes.titlesize": 5.8, "axes.labelsize": 5.5, "xtick.labelsize": 5,
                         "ytick.labelsize": 5, "legend.fontsize": 4.8}):
        fig = plt.figure(figsize=(m62.W_IN, 4.6))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.6, left=0.07, right=0.985, top=0.89, bottom=0.11)
        top = outer[0].subgridspec(1, 5, wspace=0.45, width_ratios=[1, 1, 1, 1, 1.5])
        bot = outer[1].subgridspec(1, 3, wspace=0.5, width_ratios=[1.25, 1.1, 1.25])
        # a: PSTHs (R-)
        z = np.load(BASE / "psth_halves" / "psth_halves.npz", allow_pickle=True)
        S = pd.read_csv(BASE / "psth_halves" / "sessions.csv")
        P, Nev, edges = z["psth"], z["n"], z["edges"]
        tc = (edges[:-1] + np.diff(edges) / 2) * 1e3
        sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")
        axs_a = []
        for j, (stg, hh) in enumerate([("learning", 0), ("learning", 1), ("expert", 0), ("expert", 1)]):
            ax = fig.add_subplot(top[j]); axs_a.append(ax)
            msk = ((S.cohort == "R-") & (S.stage == stg)).to_numpy()
            ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
            for k, c in enumerate(m51.CLASSES):
                ok = msk & (Nev[:, k, hh] >= 3)
                A = np.array([sm(p) for p in P[ok, k, hh]])
                if len(A) < 2:
                    continue
                mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                ax.fill_between(tc, mu - se, mu + se, color=CL[c], alpha=0.25, lw=0)
                ax.plot(tc, mu, color=CL[c], lw=0.9, label=CLAB[c])
            ax.set_xlim(-500, 300); ax.set_xticks([-400, 0, 200]); ax.set_xticklabels(["−400", "0", "200"])
            ax.set_title(f"{'Day 0' if stg == 'learning' else 'Expert'}, {'early' if hh == 0 else 'late'} half\n"
                         f"({int(msk.sum())} sessions)", color=COH["R-"], fontsize=5.2, pad=2)
            if j == 0:
                ax.set_ylabel("Rate − baseline (Hz)")
                ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1)
        yl = (min(a.get_ylim()[0] for a in axs_a), max(a.get_ylim()[1] for a in axs_a))
        for j, a in enumerate(axs_a):
            a.set_ylim(*yl)
            if j:
                a.set_yticklabels([])
        x0, x1 = axs_a[0].get_position().x0, axs_a[-1].get_position().x1
        fig.text((x0 + x1) / 2, axs_a[0].get_position().y0 - 0.07, "R− mice: time from first lick (ms); mean ± s.e.m. over sessions",
                 ha="center", fontsize=5.2, color=COH["R-"])
        # b: whisker-hit rate across the session
        TR = pd.read_csv(BASE / "slopes" / "trial_slopes_trajectories.csv")
        xb = (np.arange(5) + 0.5) / 5
        ax_b = fig.add_subplot(top[4])
        for (c, stg), alpha in [(("R-", "learning"), 1), (("R-", "expert"), 1), (("R+", "learning"), 0.45)]:
            q = TR[(TR.axis == "rate") & (TR.cls == "WH") & (TR.cohort == c) & (TR.stage == stg)].groupby("bin").score.agg(["mean", "sem"]).reindex(range(5))
            ax_b.errorbar(xb, q["mean"], q["sem"], color=COH[c], ls=STAGE_LS[stg], lw=1.0, marker="o", ms=2.2, capsize=0,
                          alpha=alpha, mfc="white" if stg == "learning" else COH[c],
                          label=f"{c.replace('-', '−')} {'day 0' if stg == 'learning' else 'expert'}")
        ax_b.set_xlabel("Time in session (normalised)"); ax_b.set_ylabel("Whisker hits / min")
        ax_b.set_title("R− stop licking to whisker\nwithin day 0", fontsize=5.4)
        ax_b.legend(frameon=False, loc="upper right")
        # c: trial-level WH - SL score across the session
        ax_c = fig.add_subplot(bot[0])
        md = TR[TR.axis == "md"].pivot_table(index=["session_id", "cohort", "stage", "bin"], columns="cls", values="score").reset_index()
        md["rel"] = md["WH"] - md["FA"]
        for (c, stg), alpha in [(("R-", "learning"), 1), (("R-", "expert"), 1), (("R+", "learning"), 0.45)]:
            q = md[(md.cohort == c) & (md.stage == stg)].groupby("bin").rel.agg(["mean", "sem", "count"]).reindex(range(5))
            ax_c.fill_between(xb, q["mean"] - q["sem"], q["mean"] + q["sem"], color=COH[c], alpha=0.12 * alpha, lw=0)
            ax_c.plot(xb, q["mean"], color=COH[c], ls=STAGE_LS[stg], lw=1.1, marker="o", ms=2.2, alpha=alpha,
                      mfc="white" if stg == "learning" else COH[c],
                      label=f"{c.replace('-', '−')} {'day 0' if stg == 'learning' else 'expert'} "
                            f"({md[(md.cohort == c) & (md.stage == stg)].session_id.nunique()})")
        ax_c.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax_c.set_xlabel("Time in session (normalised)"); ax_c.set_ylabel("WH − SL score\n(session SL → AH axis; 1 = AH − SL)")
        ax_c.set_title("Single trials: R− whisker hits converge\nonto spontaneous licks within day 0", fontsize=5.4)
        ax_c.legend(frameon=False, loc="lower left", fontsize=4.5)
        # d / e: common-footing epochs (003, 4 events per class)
        E = pd.read_csv(BASE / "epochs_n4" / "epoch_sessions.csv")
        E["half"] = E.epoch.str.split("-").str[1]
        ax_d = fig.add_subplot(bot[1])
        xe = {("learning", "early"): 0, ("learning", "late"): 1, ("expert", "early"): 2.4, ("expert", "late"): 3.4}
        for k_, (c, alpha) in enumerate([("R-", 1), ("R+", 0.4)]):
            pts = {}
            for (stg, hh), x in xe.items():
                d = E[(E.cohort == c) & (E.stage == stg) & (E.half == hh)].dropna(subset=["ddn"])
                m = d.groupby("mouse_id").ddn.mean().mean(); bt = hb_mean(d, "ddn", 2000)
                pts[(stg, hh)] = m
                dx = (k_ - 0.5) * 0.18
                ax_d.errorbar(x + dx, m, [[m - np.percentile(bt, 2.5)], [np.percentile(bt, 97.5) - m]], fmt="o", ms=3.5,
                              color=COH[c], mfc="white" if stg == "learning" else COH[c], lw=0.9, capsize=0, alpha=alpha)
                rows.append(dict(panel="d", cohort=c, stage=stg, half=hh, ddn=m, lo=np.percentile(bt, 2.5),
                                 hi=np.percentile(bt, 97.5), n_sessions=len(d), n_mice=d.mouse_id.nunique()))
            for stg in ("learning", "expert"):
                ax_d.plot([xe[(stg, "early")] + (k_ - 0.5) * 0.18, xe[(stg, "late")] + (k_ - 0.5) * 0.18],
                          [pts[(stg, "early")], pts[(stg, "late")]], color=COH[c], lw=1.0, alpha=alpha,
                          label=c.replace("-", "−") if stg == "learning" else None)
        ax_d.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax_d.set_xticks(list(xe.values()), ["early", "late", "early", "late"])
        for x_, s_ in ((0.5, "Day 0"), (2.9, "Expert")):
            ax_d.annotate(s_, xy=(x_, 0), xycoords=ax_d.get_xaxis_transform(), xytext=(0, -11), textcoords="offset points",
                          ha="center", va="top", fontsize=5.2)
        ax_d.set_ylabel("Δd / d(AH, SL)\n(> 0: WH closer to AH)")
        ax_d.set_title("Population distance per half\n(4 events / class, 150 units; 95% CI)", fontsize=5.4)
        ax_d.legend(frameon=False, loc="lower right")
        # e: size of the within-session change, day 0 vs expert
        ax_e = fig.add_subplot(bot[2])
        names = [("pos", "Position on\nSL → AH axis"), ("ddn", "Normalised\ndistance")]
        for i, (col, lab) in enumerate(names):
            Pv = E.pivot_table(index=["session_id", "mouse_id", "cohort", "stage"], columns="half", values=col).reset_index()
            Pv["ad"] = (Pv["late"] - Pv["early"]).abs()
            for k_, (c, alpha) in enumerate([("R-", 1), ("R+", 0.4)]):
                L = Pv[(Pv.cohort == c) & (Pv.stage == "learning")].dropna(subset=["ad"])
                X = Pv[(Pv.cohort == c) & (Pv.stage == "expert")].dropna(subset=["ad"])
                scale = Pv.dropna(subset=["ad"]).ad.median()                      # per-measure scale (median |Δ|)
                diff = hb_mean(L, "ad") - hb_mean(X, "ad")
                pb = min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                pm = stats.mannwhitneyu(L.ad, X.ad).pvalue
                base = i * 2.2 + k_ * 0.95
                for j_, (d_, st_) in enumerate([(L, "learning"), (X, "expert")]):
                    v = d_.ad / scale
                    ax_e.scatter(base + j_ * 0.42 + rng.uniform(-0.08, 0.08, len(v)), v, s=3, color=COH[c], alpha=0.35 * alpha, lw=0)
                    ax_e.errorbar(base + j_ * 0.42, v.mean(), v.sem(), fmt="o", ms=3.5, color=COH[c],
                                  mfc="white" if st_ == "learning" else COH[c], lw=0.9, capsize=0, alpha=alpha)
                top_ = np.nanpercentile(Pv.ad / scale, 97)
                ax_e.text(base + 0.21, top_ * 1.04, m62.fmt_p(pb), ha="center", fontsize=4.4, color=COH[c], alpha=max(alpha, 0.6))
                rows.append(dict(panel="e", measure=col, cohort=c, abs_change_day0=L.ad.mean(), abs_change_expert=X.ad.mean(),
                                 n_day0=len(L), n_expert=len(X), n_mice_expert=X.mouse_id.nunique(), p_boot=pb, p_MWU=pm))
            ax_e.text(i * 2.2 + 0.68, -0.13, lab, transform=ax_e.get_xaxis_transform(), ha="center", va="top", fontsize=5)
        ax_e.set_xticks([b + o for i in range(2) for b in (i * 2.2, i * 2.2 + 0.95) for o in (0, 0.42)],
                        ["D0", "E", "D0", "E"] * 2, fontsize=4.6)
        ax_e.set_xlim(-0.35, 4.0); ax_e.set_ylim(bottom=0)
        ax_e.set_ylabel("|late − early| within session\n(÷ median across sessions)")
        ax_e.set_title("Within-session change: day 0 (D0) vs expert (E)\nR− (dark), R+ (light); hierarchical bootstrap p",
                       fontsize=5.2)
        m62.letter_row(fig, [axs_a[0], ax_b], "ab", dx_in=0.3, dy_in=0.12)
        m62.letter_row(fig, [ax_c, ax_d, ax_e], "cde", dx_in=0.35, dy_in=0.25)
        fig.suptitle("R− whisker-hit activity changes more within the learning day than within expert sessions",
                     x=0.07, y=0.995, ha="left", fontsize=7, weight="bold")
        m62.save(fig, OUT, "COSYNE_within_day"); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "COSYNE_within_day_stats.csv", index=False)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
