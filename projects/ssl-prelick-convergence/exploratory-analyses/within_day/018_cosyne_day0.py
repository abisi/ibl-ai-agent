"""018 -- COSYNE abstract figure on the learning day only (user 2026-10-07: 4 panels, no behaviour, day-0 area data only).

  a, b  Day-0 single-trial scores on the session's reward-lick coding direction (CD; 5-fold AH - SL mean-difference axis,
        SL = 0, AH = 1) across normalised session time, WH / AH / SL, R+ (a) and R- (b); 5 bins, mean +- s.e.m. over
        sessions (011 data: 002 trial_scores.parquet).
  c     Day-0 drift per session: beta_WH - beta_SL and beta_WH - beta_AH (OLS slopes of the class scores against tau), R+ vs
        R-; dots = sessions, mean +- s.e.m.; p = mouse-level cohort permutation (Mann-Whitney U and Welch t in the stats table).
  d     Day-0 drift per area group (015 per-area CD): area group x [R+, R-] for each drift, mean over sessions; '*' = drift
        differs from 0 (Wilcoxon p < 0.05); right: R+ vs R- mouse-level permutation p per drift; areas with >= 3 day-0
        sessions in both cohorts (017 stage rule). q = Benjamini-Hochberg FDR of the permutation p over the areas shown, per
        drift (user 2026-10-07).
Output: within_day/<ref>/cosyne/<pop>/COSYNE_day0.{png,pdf,svg}, COSYNE_day0_stats.csv; --strict: COSYNE_day0_strict.* (panel d
restricted to areas with >= 3 sessions in every cohort x stage condition; user 2026-10-07); --dots: panel d as a dot plot
(per-area mean +- s.e.m., R+ and R-, as area_cd_drift) instead of the matrix, suffix _dots.
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
m011 = importlib.import_module("011_cd_trajectories")
m015 = importlib.import_module("015_area_cd_drift")
m017 = importlib.import_module("017_area_cd_drift_views")
COH, CL = m62.COH, m62.CL
OUT = m51.WITHIN / "cosyne"


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup()
    out = OUT / a.population
    out.mkdir(parents=True, exist_ok=True)
    flt = m61.learner_filter if a.population == "learners" else (lambda d: d)
    T = flt(pd.read_parquet(m51.WITHIN / "slopes" / "trial_scores.parquet"))
    S, _, D = m011.session_table(T)
    S0 = S[S.stage == "learning"]
    A = flt(pd.read_csv(m015.OUT / "area_sessions.csv"))
    rows = []
    rng = np.random.default_rng(0)
    with plt.rc_context({"font.size": 5.5, "axes.titlesize": 5.8, "axes.labelsize": 5.5, "xtick.labelsize": 5,
                         "ytick.labelsize": 5, "legend.fontsize": 4.8}):
        fig = plt.figure(figsize=(m62.W_IN, 2.55))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.05, 1.55], wspace=0.55, left=0.06, right=0.9, top=0.78, bottom=0.17)
        axs = []
        # a, b: day-0 trajectories
        for j, coh in enumerate(("R+", "R-")):
            ax = fig.add_subplot(gs[0, j]); axs.append(ax)
            n = m011.traj(ax, D, coh, "learning", ["WH", "AH", "FA"], [CL["WH"], CL["AH"], CL["FA"]],
                          ["Whisker hit", "Auditory hit", "Spont. lick"])
            ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color=CL["AH"], lw=0.5, ls=":")
            ax.set_xlim(0, 1); ax.set_ylim(-0.35, 1.35)
            ax.set_xlabel("Time in session (normalised)")
            ax.set_title(f"{coh.replace('-', '−')} mice, day 0 ({n} sessions)", color=COH[coh])
            if j == 0:
                ax.set_ylabel("Pre-lick score on reward-lick axis\n(spont. lick = 0, auditory hit = 1)")
                ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1)
        # c: day-0 drift per session
        ax = fig.add_subplot(gs[0, 2]); axs.append(ax)
        for i, (col, lab) in enumerate((("drift_WH_SL", "vs spont.\nlick"), ("drift_WH_AH", "vs auditory\nhit"))):
            r = dict(panel="c", measure=col)
            for k, coh in enumerate(("R+", "R-")):
                v = S0[S0.cohort == coh][col].dropna()
                x = i * 1.15 + k * 0.45
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(v)), v, s=3, color=COH[coh], alpha=0.4, lw=0)
                ax.errorbar(x, v.mean(), v.sem(), fmt="o", ms=3.6, color=COH[coh], lw=1, capsize=0, mfc="white")
                r[f"mean {coh}"], r[f"sem {coh}"], r[f"n {coh}"] = v.mean(), v.sem(), len(v)
            x_, y_ = S0[S0.cohort == "R+"][col].dropna(), S0[S0.cohort == "R-"][col].dropna()
            r["p_MWU"] = stats.mannwhitneyu(x_, y_).pvalue; r["p_Welch"] = stats.ttest_ind(x_, y_, equal_var=False).pvalue
            r["diff"], r["p_perm"] = m001.perm_interaction(S0.dropna(subset=[col]), col, kind="learning")
            top = np.nanpercentile(S0[col], 98)
            ax.plot([i * 1.15, i * 1.15 + 0.45], [top * 1.08] * 2, color="0.3", lw=0.6)
            ax.text(i * 1.15 + 0.225, top * 1.12, m62.fmt_p(r["p_perm"]), ha="center", va="bottom", fontsize=4.6)
            rows.append(r)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0.225, 1.375], ["vs spont. lick", "vs auditory hit"])
        ax.set_xlim(-0.3, 1.9)
        ax.set_ylabel("Whisker-hit drift over the session\n(β_WH − β_ref, CD units)")
        ax.set_title("Whisker hits drift toward (R+) or\naway from (R−) rewarded licks", fontsize=5.4)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COH[c], mfc="white", ms=3, label=c.replace("-", "−"))
                           for c in ("R+", "R-")], frameon=False, loc="lower left", fontsize=4.6)
        # d: per-area day-0 drift (matrix, or dot plot with --dots)
        TT = pd.concat([m017.stage_tests(A, c, "learning") for c, _ in m017.MEAS], ignore_index=True)
        both = TT.dropna(subset=["p_perm"])
        keep = set(both[both.measure == "drift_WH_SL"].area_group) & set(both[both.measure == "drift_WH_AH"].area_group)
        if a.strict:                                         # only areas with >= 3 sessions in every cohort x stage condition
            keep &= m017.strict_areas(A, "drift_WH_SL") & m017.strict_areas(A, "drift_WH_AH")
        TT["q_BH"] = np.nan                                  # FDR (Benjamini-Hochberg) over the areas shown, per drift
        for c_, _ in m017.MEAS:
            k_ = (TT.measure == c_) & TT.area_group.isin(keep)
            TT.loc[k_, "q_BH"] = m015.bh(TT.loc[k_, "p_perm"])
        order = m017.order_of(keep)
        Ti = TT.set_index(["measure", "area_group"])
        for col, _ in m017.MEAS:
            for g in order:
                rows.append(dict(panel="d", measure=col, area_group=g, **{k_: Ti.loc[(col, g)].get(k_) for k_ in
                            ("n R+", "mean R+", "sem R+", "p_wilcoxon R+", "n R-", "mean R-", "sem R-", "p_wilcoxon R-",
                             "p_MWU", "p_Welch", "p_perm", "q_BH")}))
        gcol = au.get_custom_area_groups_colors()
        if a.dots:
            dots_d(fig, gs[0, 3], TT, order, gcol, axs)
        else:
            matrix_d(fig, gs[0, 3], Ti, order, gcol, axs)
        m62.letter_row(fig, axs[:3], "abc", dx_in=0.35, dy_in=0.15)
        m62.letter_row(fig, axs[3:4], "d", dx_in=0.55, dy_in=0.3)
        fig.suptitle("On the learning day, whisker-hit pre-lick activity drifts toward the rewarded-lick pattern in R+ mice "
                     "and away from it in R− mice", x=0.02, y=0.995, ha="left", va="top", fontsize=6.6, weight="bold")
        tag = ("_strict" if a.strict else "") + ("_dots" if a.dots else "")
        m62.save(fig, out, f"COSYNE_day0{tag}"); plt.close(fig)
    R = pd.DataFrame(rows); R.to_csv(out / f"COSYNE_day0{tag}_stats.csv", index=False)
    with pd.option_context("display.width", 250):
        print(R.round(4).to_string())


def dots_d(fig, spec, TT, order, gcol, axs):
    """panel d as dot plot: day-0 drift per area group, R+ and R- (mean +- s.e.m. over sessions, filled = drift != 0),
    one column per drift, then R+ vs R- permutation p and FDR q per drift"""
    sub = spec.subgridspec(1, 3, width_ratios=[1, 1, 0.75], wspace=0.08)
    y = None
    for j, (col, lab) in enumerate((("drift_WH_SL", "vs spont. lick"), ("drift_WH_AH", "vs auditory hit"))):
        ax = fig.add_subplot(sub[0, j]); axs.append(ax)
        y, _ = m017.forest(ax, TT[TT.measure == col], order, j == 0, gcol, xlabel=False)
        ax.tick_params(axis="y", labelsize=4.8)
        ax.set_title(lab, fontsize=5.2, pad=2)
        ax.set_xlabel("Drift (CD units)", fontsize=4.8)
    axp = fig.add_subplot(sub[0, 2], sharey=axs[-1]); axp.axis("off")
    Ti = TT.set_index(["measure", "area_group"])
    axp.text(0.0, len(order) - 0.45, "R+ vs R− p (q)\n  SL        AH", fontsize=4.3, va="bottom", weight="bold")
    for i, g in enumerate(order):
        for k, col in enumerate(("drift_WH_SL", "drift_WH_AH")):
            p, q = Ti.loc[(col, g), "p_perm"], Ti.loc[(col, g), "q_BH"]
            axp.text(0.0 + 0.5 * k, y[i], f"{m017.fp(p)}\n({m017.fp(q)})", fontsize=3.9, va="center", linespacing=1.0,
                     color="#D62728" if p < 0.05 else "0.35", weight="bold" if q < 0.05 else "normal")
    axp.set_xlim(0, 1)
    axs[3].set_title(axs[3].get_title(), fontsize=5.2, pad=2)
    fig.text(axs[3].get_position().x0, axs[3].get_position().y1 + 0.075, "Day-0 drift per area group (filled: ≠ 0; bold q < 0.05)",
             fontsize=5.4, ha="left")


def matrix_d(fig, spec, Ti, order, gcol, axs):
    ax = fig.add_subplot(spec); axs.append(ax)
    if True:
        cols = [("drift_WH_SL", "R+"), ("drift_WH_SL", "R-"), ("drift_WH_AH", "R+"), ("drift_WH_AH", "R-")]
        M = np.full((len(order), 4), np.nan); star = np.zeros_like(M, bool)
        for i, g in enumerate(order):
            for j, (col, c) in enumerate(cols):
                r = Ti.loc[(col, g)]
                M[i, j] = r[f"mean {c}"]; star[i, j] = r[f"p_wilcoxon {c}"] < 0.05
        vlim = float(np.nanpercentile(np.abs(M), 95))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:+.2f}".replace("-", "−") + ("*" if star[i, j] else ""), ha="center", va="center",
                        fontsize=4.1, color="white" if abs(M[i, j]) > 0.6 * vlim else "k")
        ax.axvline(1.5, color="white", lw=1.5)
        ax.set_xticks(range(4), ["R+", "R−", "R+", "R−"]); ax.xaxis.tick_top()
        for t, c in zip(ax.get_xticklabels(), ["R+", "R-", "R+", "R-"]):
            t.set_color(COH[c])
        for x_, lab in ((0.5, "vs spont. lick"), (2.5, "vs auditory hit")):
            ax.annotate(lab, xy=(x_, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 13), textcoords="offset points", ha="center", va="bottom", fontsize=4.8)
        ax.set_yticks(range(len(order)), [m017.SHORT(g) for g in order])
        for tl, g in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(g, "k"))
        for k, col in enumerate(("drift_WH_SL", "drift_WH_AH")):
            ax.annotate("SL" if k == 0 else "AH", xy=(4.05 + 0.9 * k, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=4.6, weight="bold")
            for i, g in enumerate(order):
                p, q = Ti.loc[(col, g), "p_perm"], Ti.loc[(col, g), "q_BH"]
                ax.text(4.05 + 0.9 * k, i, f"{m017.fp(p)}\n({m017.fp(q)})", ha="center", va="center", fontsize=3.9, clip_on=False,
                        linespacing=1.0, color="#D62728" if p < 0.05 else "0.35", weight="bold" if q < 0.05 else "normal")
        ax.annotate("R+ vs R− p (q)", xy=(4.5, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 13), textcoords="offset points", ha="center", va="bottom", fontsize=4.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.25, location="bottom", aspect=30)
        cb.set_label("Day-0 whisker-hit drift (CD units)", fontsize=4.8); cb.ax.tick_params(labelsize=4.5)
        ax.set_title("Day-0 drift per area group (* ≠ 0; bold q < 0.05)", pad=24, fontsize=5.4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    ap.add_argument("--strict", action="store_true", help="panel d: only areas with >= 3 sessions in every cohort x stage condition")
    ap.add_argument("--dots", action="store_true", help="panel d as a dot plot (per-area means, as area_cd_drift) instead of a matrix")
    main(ap.parse_args())
