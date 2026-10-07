"""013 -- Within-session change on the learning day and in expert sessions, R+ and R- at equal weight (user review
2026-10-07: generalises the COSYNE within-day panels b-e of 005, which showed R- with R+ day 0 as a faint reference).

  a-c  Behaviour across the session (002 trajectories, 5 equal bins of normalised session time): whisker-hit, auditory-hit
       and spontaneous-lick rate (events / min); mean +- s.e.m. over sessions; day 0 dashed with open markers, expert solid.
  d-e  Epoch means on a common footing (003, 4 events per class, 150 units, epochs_n4): normalised distance difference
       Delta d / d(AH, SL) and position on the SL -> AH axis, early and late half of day 0 and of expert sessions; mean over
       mice of session values, hierarchical-bootstrap 95% CI (mice, then sessions).
  f-h  Size of the within-session change |late - early| per session for the normalised distance, the position and the
       held-out decoder transfer (dec), day 0 vs expert per cohort; p = hierarchical bootstrap of the day 0 - expert
       difference (4000 resamples); the table also gives Mann-Whitney U and R+ vs R- per stage (MWU, Welch, mouse-level
       cohort permutation) and the cohort x stage interaction (mouse-level permutation).
Output: within_day/<ref>/within_change/<pop>/within_session_change.{png,pdf,svg}, within_session_change_stats.csv.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
BASE = m51.WITHIN
OUT = BASE / "within_change"
COH = m62.COH
LS = {"learning": (0, (2.5, 1.5)), "expert": "-"}
SL_ = {"learning": "day 0", "expert": "expert"}
rng = np.random.default_rng(0)


def hb_mean(df, col, B=4000):
    g = {k: v[col].to_numpy() for k, v in df.groupby("mouse_id")}
    mice = list(g)
    return np.array([np.mean([rng.choice(g[k], len(g[k])).mean() for k in rng.choice(mice, len(mice))]) for _ in range(B)])


def main(a):
    plt = m62.setup()
    out = OUT / a.population
    out.mkdir(parents=True, exist_ok=True)
    flt = m61.learner_filter if a.population == "learners" else (lambda d: d)
    TR = flt(pd.read_csv(BASE / "slopes" / "trial_slopes_trajectories.csv"))
    E = flt(pd.read_csv(BASE / "epochs_n4" / "epoch_sessions.csv"))
    E["half"] = E.epoch.str.split("-").str[1]
    rows = []
    fig = plt.figure(figsize=(m62.W_IN, 4.9))
    gs = fig.add_gridspec(2, 1, hspace=0.62, left=0.07, right=0.985, top=0.9, bottom=0.1)
    top = gs[0].subgridspec(1, 5, wspace=0.55)
    bot = gs[1].subgridspec(1, 3, wspace=0.45)
    xb = (np.arange(5) + 0.5) / 5
    axs_t = []
    for j, (cls, ttl) in enumerate((("WH", "Whisker hits"), ("AH", "Auditory hits"), ("FA", "Spontaneous licks"))):
        ax = fig.add_subplot(top[j]); axs_t.append(ax)
        for c in ("R+", "R-"):
            for stg in ("learning", "expert"):
                d = TR[(TR.axis == "rate") & (TR.cls == cls) & (TR.cohort == c) & (TR.stage == stg)]
                q = d.groupby("bin").score.agg(["mean", "sem"]).reindex(range(5))
                ax.errorbar(xb, q["mean"], q["sem"], color=COH[c], ls=LS[stg], lw=1.0, marker="o", ms=2.2, capsize=0,
                            mfc="white" if stg == "learning" else COH[c],
                            label=f"{c.replace('-', '−')} {SL_[stg]} ({d.session_id.nunique()})")
        ax.set_xlabel("Time in session (normalised)"); ax.set_ylabel("Events / min" if j == 0 else "")
        ax.set_title(f"{ttl} across the session", fontsize=5.4)
        if j == 0:
            ax.legend(frameon=False, fontsize=4.2, loc="best")
    xe = {("learning", "early"): 0, ("learning", "late"): 1, ("expert", "early"): 2.4, ("expert", "late"): 3.4}
    for j, (col, ylab, ttl) in enumerate((("ddn", "Δd / d(AH, SL)\n(> 0: WH closer to AH)", "Normalised distance per half"),
                                          ("pos", "Position on SL → AH axis\n(0 = SL, 1 = AH)", "Position per half"))):
        ax = fig.add_subplot(top[3 + j]); axs_t.append(ax)
        for k_, c in enumerate(("R+", "R-")):
            pts = {}
            dx = (k_ - 0.5) * 0.2
            for (stg, hh), x in xe.items():
                d = E[(E.cohort == c) & (E.stage == stg) & (E.half == hh)].dropna(subset=[col])
                m = d.groupby("mouse_id")[col].mean().mean(); bt = hb_mean(d, col, 2000)
                lo, hi = np.percentile(bt, [2.5, 97.5])
                pts[(stg, hh)] = m
                ax.errorbar(x + dx, m, [[m - lo], [hi - m]], fmt="o", ms=3.2, color=COH[c],
                            mfc="white" if stg == "learning" else COH[c], lw=0.9, capsize=0)
                rows.append(dict(panel="per_half", measure=col, cohort=c, stage=stg, half=hh, mean=m, lo=lo, hi=hi,
                                 n_sessions=len(d), n_mice=d.mouse_id.nunique()))
            for stg in ("learning", "expert"):
                ax.plot([xe[(stg, "early")] + dx, xe[(stg, "late")] + dx], [pts[(stg, "early")], pts[(stg, "late")]],
                        color=COH[c], lw=1.0, ls=LS[stg], label=c.replace("-", "−") if stg == "expert" else None)
        if col == "ddn":
            ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(list(xe.values()), ["early", "late", "early", "late"], fontsize=4.8)
        for x_, s_ in ((0.5, "Day 0"), (2.9, "Expert")):
            ax.annotate(s_, xy=(x_, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=5)
        ax.set_ylabel(ylab); ax.set_title(f"{ttl}\n(4 events / class; 95% CI)", fontsize=5.2)
        ax.legend(frameon=False, fontsize=4.4, loc="upper left")
    axs_b = []
    for j, (col, ttl) in enumerate((("ddn", "Normalised distance Δd / d(AH, SL)"), ("pos", "Position on SL → AH axis"),
                                    ("dec", "Decoder transfer (held-out)"))):
        ax = fig.add_subplot(bot[j]); axs_b.append(ax)
        Pv = E.pivot_table(index=["session_id", "mouse_id", "cohort", "stage"], columns="half", values=col).reset_index()
        Pv["ad"] = (Pv["late"] - Pv["early"]).abs()
        Pv = Pv.dropna(subset=["ad"])
        r = dict(panel="abs_change", measure=col)
        for k_, c in enumerate(("R+", "R-")):
            L = Pv[(Pv.cohort == c) & (Pv.stage == "learning")]; X = Pv[(Pv.cohort == c) & (Pv.stage == "expert")]
            diff = hb_mean(L, "ad") - hb_mean(X, "ad")
            pb = min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            for j_, (d_, st_) in enumerate(((L, "learning"), (X, "expert"))):
                x = k_ * 1.2 + j_ * 0.5
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(d_)), d_.ad, s=3, color=COH[c], alpha=0.35, lw=0)
                ax.errorbar(x, d_.ad.mean(), d_.ad.sem(), fmt="o", ms=3.5, color=COH[c],
                            mfc="white" if st_ == "learning" else COH[c], lw=0.9, capsize=0)
            ax.text(k_ * 1.2 + 0.25, 1.0, m62.fmt_p(pb), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=4.6, color=COH[c])
            r.update({f"{c} day0 mean": L.ad.mean(), f"{c} expert mean": X.ad.mean(), f"{c} n day0": len(L),
                      f"{c} n expert": len(X), f"{c} day0 vs expert p_boot": pb,
                      f"{c} day0 vs expert p_MWU": stats.mannwhitneyu(L.ad, X.ad).pvalue})
        for stg in ("learning", "expert"):
            A_ = Pv[(Pv.cohort == "R+") & (Pv.stage == stg)].ad; B_ = Pv[(Pv.cohort == "R-") & (Pv.stage == stg)].ad
            r[f"{stg} R+ vs R- p_MWU"] = stats.mannwhitneyu(A_, B_).pvalue
            r[f"{stg} R+ vs R- p_Welch"] = stats.ttest_ind(A_, B_, equal_var=False).pvalue
            r[f"{stg} R+ - R-"], r[f"{stg} p_perm"] = m001.perm_interaction(Pv[Pv.stage == stg].assign(stage="learning"), "ad", kind="learning")
        r["cohort x stage"], r["cxs p_perm"] = m001.perm_interaction(Pv, "ad", kind="stage")
        rows.append(r)
        ax.set_xticks([0, 0.5, 1.2, 1.7], ["R+ D0", "R+ E", "R− D0", "R− E"], fontsize=4.8)
        ax.set_xlim(-0.35, 2.05); ax.set_ylim(bottom=0)
        ax.set_ylabel("|late − early| within session" if j == 0 else "")
        ax.set_title(f"{ttl}\nchange size, day 0 (D0) vs expert (E)", fontsize=5.2, pad=9)
    m62.letter_row(fig, axs_t, "abcde", dx_in=0.3, dy_in=0.12)
    m62.letter_row(fig, axs_b, "fgh", dx_in=0.3, dy_in=0.2)
    fig.suptitle(f"Within-session change on the learning day and in expert sessions, R+ and R− ({a.population})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "within_session_change"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "within_session_change_stats.csv", index=False)
    print(pd.DataFrame([r for r in rows if r["panel"] == "abs_change"]).round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
