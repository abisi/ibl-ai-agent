"""019 -- COSYNE abstract figure on the learning day, version 2 (user 2026-10-07): row 1 = the pre-lick score of every
event type per cohort and the drift measured in example sessions; row 2 = every quantification relative to spontaneous
licks (SL); row 3 = every quantification relative to auditory hits (AH).

Data: learning day (day 0), learners by default; 002 trial_scores.parquet (5-fold reward-lick coding direction, CD:
mean AH - mean SL of the training folds, unit norm, units z-scored on the training folds; held-out projections for AH and
SL, fold-averaged for WH; one normalisation per session, SL = 0, AH = 1; session kept if held-out AH-vs-SL d' >= 0.3);
015 area_sessions.csv (same CD from the units of one area group, >= 5 units).

Row 1
  a, b  Score of every event type across normalised session time tau (5 bins; per-session class means, mean +- s.e.m. over
        sessions), R+ (a) and R- (b).
  c, d  The drift in one example session per cohort (the day-0 session with >= 20 WH closest to the cohort medians of
        the two drifts, in s.d. units): every event's score against tau, OLS line per class (slope beta_k), and the
        two drifts beta_WH - beta_SL and beta_WH - beta_AH.
Row 2 (reference SL), row 3 (reference AH)
  e / h  Score difference across the session, both cohorts: WH - SL (e; rises when WH move away from SL toward AH) and
         AH - WH (h; the distance of WH to AH, falls when WH approach AH).
  f / i  Drift per session, beta_WH - beta_ref (> 0: WH move toward AH relative to the reference); dots = sessions, open
         circle = mean +- s.e.m.; p = mouse-level cohort permutation (MWU, Welch in the stats table).
  g / j  The same drift per area group (CD from that area's units), R+ and R- mean +- s.e.m. over sessions, filled = drift
         != 0 (Wilcoxon p < 0.05); areas with >= 3 sessions in every cohort x stage condition (017 strict rule); p =
         R+ vs R- mouse-level permutation, q = Benjamini-Hochberg over the areas shown.
Output: within_day/<ref>/cosyne/<pop>/COSYNE_day0_v2.{png,pdf,svg}, COSYNE_day0_v2_stats.csv.
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
m008 = importlib.import_module("008_cosyne_timeline_expanded")
m011 = importlib.import_module("011_cd_trajectories")
m015 = importlib.import_module("015_area_cd_drift")
m017 = importlib.import_module("017_area_cd_drift_views")
COH, CL = m62.COH, m62.CL
OUT = m51.WITHIN / "cosyne"
CLS_LAB = {"WH": "Whisker hit", "AH": "Auditory hit", "FA": "Spont. lick"}
REFS = [("FA", "SL", "drift_WH_SL", "WH_SL", "spontaneous licks"), ("AH", "AH", "drift_WH_AH", "AH_WH", "auditory hits")]


EX_YLIM = (-1.0, 2.9)                                       # example panels (c, d)
DOT_YLIM = (-2.9, 3.4)                                      # drift panels (f, i); sessions beyond are drawn at the edge


EX_PCT = {"R+": 85, "R-": 15}                              # illustrative examples (user 2026-10-07): larger drifts


def example_session(S0, coh, n_wh=20):
    """illustrative example (user 2026-10-07: "larger drifts towards AH in R+ and away in R-, relative to AH"): among the
    day-0 sessions with >= n_wh WH whose drift relative to SL has the cohort's sign, the one whose beta_WH - beta_AH is
    closest to the EX_PCT percentile of its cohort (85th for R+, 15th for R-); returns (session_id, percentile rank)"""
    d = S0[S0.cohort == coh].dropna(subset=["drift_WH_AH", "drift_WH_SL"])
    target = np.percentile(d.drift_WH_AH, EX_PCT[coh])
    sign = 1 if coh == "R+" else -1
    c = d[(d.n_WH >= n_wh) & (sign * d.drift_WH_SL > 0)]
    r = c.loc[(c.drift_WH_AH - target).abs().idxmin()]
    return r.session_id, 100 * (d.drift_WH_AH <= r.drift_WH_AH).mean()




def example_panel(ax, lab, tau, s, coh, nu, rows):
    out = {}
    for cl in ("FA", "AH", "WH"):
        k = (lab == cl) & np.isfinite(s)
        ax.scatter(tau[k], s[k], s=2.5 if cl == "FA" else 5, color=CL[cl], alpha=0.3 if cl == "FA" else 0.75, lw=0,
                   zorder=2 if cl == "FA" else 3)
        b, a0 = np.polyfit(tau[k], s[k], 1)
        ax.plot([0, 1], [a0, a0 + b], color=CL[cl], lw=1.4, zorder=5, solid_capstyle="round")
        out[cl] = b
    ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.axhline(1, color=CL["AH"], lw=0.4, ls=":")
    lo, hi = np.nanpercentile(s[np.isin(lab, ["WH", "AH", "FA"])], [1, 99])
    ax.set_ylim(*EX_YLIM); ax.set_xlim(0, 1)                # same range in both examples
    d_sl, d_ah = out["WH"] - out["FA"], out["WH"] - out["AH"]
    txt = (f"{nu} units; β_WH {out['WH']:+.2f}, β_SL {out['FA']:+.2f}, β_AH {out['AH']:+.2f}\n"
           f"β_WH − β_SL = {d_sl:+.2f}\nβ_WH − β_AH = {d_ah:+.2f}").replace("-", "−")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=4.3, linespacing=1.15,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.set_title(f"{coh.replace('-', '−')} example session ({nu} units)", color=COH[coh])
    rows.append(dict(panel="example", cohort=coh, n_units=nu, beta_WH=out["WH"], beta_SL=out["FA"], beta_AH=out["AH"],
                     drift_WH_SL=d_sl, drift_WH_AH=d_ah))


def drift_dots(ax, S0, col, rng, rows, key):
    r = dict(panel=key, measure=col)
    for k, coh in enumerate(("R+", "R-")):
        v = S0[S0.cohort == coh][col].dropna()
        jit = k + rng.uniform(-0.12, 0.12, len(v))
        inside = v.between(*DOT_YLIM).to_numpy()
        ax.scatter(jit[inside], v[inside], s=3, color=COH[coh], alpha=0.4, lw=0)
        for xx, vv in zip(jit[~inside], v[~inside]):         # clipped sessions: triangle at the edge with the value
            yy = DOT_YLIM[0] + 0.05 if vv < 0 else DOT_YLIM[1] - 0.05
            ax.scatter(xx, yy, s=7, marker="v" if vv < 0 else "^", color=COH[coh], alpha=0.7, lw=0)
            ax.text(xx + 0.1, yy, f"{vv:.1f}".replace("-", "−"), fontsize=3.8, va="center", color=COH[coh])
        ax.errorbar(k, v.mean(), v.sem(), fmt="o", ms=3.8, color=COH[coh], lw=1, capsize=0, mfc="white", zorder=4)
        r[f"mean {coh}"], r[f"sem {coh}"], r[f"n {coh}"] = v.mean(), v.sem(), len(v)
        r[f"p_wilcoxon {coh}"] = stats.wilcoxon(v).pvalue
    x_, y_ = S0[S0.cohort == "R+"][col].dropna(), S0[S0.cohort == "R-"][col].dropna()
    r["p_MWU"] = stats.mannwhitneyu(x_, y_).pvalue; r["p_Welch"] = stats.ttest_ind(x_, y_, equal_var=False).pvalue
    r["diff"], r["p_perm"] = m001.perm_interaction(S0.dropna(subset=[col]), col, kind="learning")
    rows.append(r)
    top = DOT_YLIM[1] * 0.9
    ax.plot([0, 1], [top] * 2, color="0.3", lw=0.6)
    ax.text(0.5, top * 1.03, m62.fmt_p(r["p_perm"]), ha="center", va="bottom", fontsize=4.6)
    ax.set_ylim(*DOT_YLIM)
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1], ["R+", "R−"])
    for t, c in zip(ax.get_xticklabels(), ("R+", "R-")):
        t.set_color(COH[c])
    ax.set_xlim(-0.55, 1.55)


def area_panel(fig, spec, TT, order, gcol, col, show_labels, rows, key):
    sub = spec.subgridspec(1, 2, width_ratios=[1, 0.32], wspace=0.04)
    ax = fig.add_subplot(sub[0])
    T = TT[TT.measure == col]
    y, Ti = m017.forest(ax, T, order, show_labels, gcol, xlabel=False)
    ax.tick_params(axis="y", labelsize=4.8)
    axp = fig.add_subplot(sub[1], sharey=ax); axp.axis("off"); axp.set_xlim(0, 1)
    axp.text(0.1, len(order) - 0.45, "p (q)", fontsize=4.5, va="bottom", weight="bold")
    for i, g in enumerate(order):
        p, q = Ti.loc[g, "p_perm"], Ti.loc[g, "q_BH"]
        axp.text(0.1, y[i], f"{m017.fp(p)}\n({m017.fp(q)})", fontsize=4.0, va="center", linespacing=1.0,
                 color="#D62728" if p < 0.05 else "0.35", weight="bold" if q < 0.05 else "normal")
        rows.append(dict(panel=key, measure=col, area_group=g, **{k_: Ti.loc[g].get(k_) for k_ in
                    ("n R+", "mean R+", "sem R+", "p_wilcoxon R+", "n R-", "mean R-", "sem R-", "p_wilcoxon R-", "p_MWU",
                     "p_Welch", "p_perm", "q_BH")}))
    return ax


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup()
    out = OUT / a.population
    out.mkdir(parents=True, exist_ok=True)
    flt = m61.learner_filter if a.population == "learners" else (lambda d: d)
    T = flt(pd.read_parquet(m51.WITHIN / "slopes" / "trial_scores.parquet"))
    S, _, D = m011.session_table(T)
    S0, D0 = S[S.stage == "learning"], D[D.stage == "learning"]
    A = flt(pd.read_csv(m015.OUT / "area_sessions.csv"))
    D2 = flt(pd.read_csv(m51.WITHIN / "slopes" / "trial_slopes_sessions.csv"))
    gcol = au.get_custom_area_groups_colors()
    rows = []
    rng = np.random.default_rng(0)
    # area tests: day 0, strict area set (>= 3 sessions in every cohort x stage condition), FDR over the areas shown
    TT = pd.concat([m017.stage_tests(A, c, "learning") for c, _ in m017.MEAS], ignore_index=True)
    keep = set(TT.dropna(subset=["p_perm"]).area_group)
    for c_, _ in m017.MEAS:
        keep &= m017.strict_areas(A, c_)
    TT["q_BH"] = np.nan
    for c_, _ in m017.MEAS:
        k_ = (TT.measure == c_) & TT.area_group.isin(keep)
        TT.loc[k_, "q_BH"] = m015.bh(TT.loc[k_, "p_perm"])
    order = m017.order_of(keep)
    with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9,
                         "ytick.labelsize": 4.9, "legend.fontsize": 4.7, "axes.titlepad": 3}):
        fig = plt.figure(figsize=(m62.W_IN, 6.9))
        outer = fig.add_gridspec(3, 1, height_ratios=[1, 1.05, 1.05], hspace=0.58, left=0.085, right=0.985, top=0.93, bottom=0.06)
        r1 = outer[0].subgridspec(1, 4, wspace=0.42)
        axs1 = []
        for j, coh in enumerate(("R+", "R-")):
            ax = fig.add_subplot(r1[j]); axs1.append(ax)
            n = m011.traj(ax, D0, coh, "learning", ["WH", "AH", "FA"], [CL[k] for k in ("WH", "AH", "FA")],
                          [CLS_LAB[k] for k in ("WH", "AH", "FA")])
            ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.axhline(1, color=CL["AH"], lw=0.4, ls=":")
            ax.set_xlim(0, 1); ax.set_ylim(-0.35, 1.35)
            ax.set_xlabel("Time in session τ")
            ax.set_title(f"{coh.replace('-', '−')} mice, day 0 ({n} sessions)", color=COH[coh])
            if j == 0:
                ax.set_ylabel("Pre-lick score on reward-lick axis\n(spont. lick = 0, auditory hit = 1)")
                ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1)
        for j, coh in enumerate(("R+", "R-")):
            ax = fig.add_subplot(r1[2 + j]); axs1.append(ax)
            sid, pct = example_session(S0, coh)
            lab, _, tau, s, nu = m008.session_scores(sid)
            example_panel(ax, lab, tau, s, coh, nu, rows)
            rows[-1].update(session_id=sid, percentile_drift_WH_AH=pct)
            ax.set_title(f"{coh.replace('-', '−')} example session ({pct:.0f}th percentile)", color=COH[coh])
            ax.set_xlabel("Time in session τ")
            if j == 0:
                ax.set_ylabel("Pre-lick score (single events)")
        rowax = {}
        for i, (ref, rl, dcol, tcol, rname) in enumerate(REFS):
            g = outer[1 + i].subgridspec(1, 3, width_ratios=[1.05, 0.62, 1.55], wspace=0.42)
            axs = []
            ax = fig.add_subplot(g[0]); axs.append(ax)
            for coh in ("R+", "R-"):
                m011.traj(ax, D0, coh, "learning", [tcol], [COH[coh]], [coh.replace("-", "−")])
            ax.axhline(0, color="0.6", lw=0.4, ls=":")
            ax.set_xlim(0, 1); ax.set_xlabel("Time in session τ")
            if ref == "FA":
                ax.set_ylabel("Whisker hit − spont. lick\n(score difference)")
                ax.set_title("WH move away from SL (R+)")
            else:
                ax.set_ylabel("Auditory hit − whisker hit\n(distance of WH to AH)")
                ax.set_title("WH approach AH (R+)")
            ax.legend(frameon=False, loc="upper left", handlelength=1.0)
            ax = fig.add_subplot(g[1]); axs.append(ax)
            drift_dots(ax, S0, dcol, rng, rows, "f" if ref == "FA" else "i")
            ax.set_ylabel(f"Drift β_WH − β_{rl}\n(change over one session)")
            ax.set_title("Per session")
            ax = area_panel(fig, g[2], TT, order, gcol, dcol, True, rows, "g" if ref == "FA" else "j"); axs.append(ax)
            ax.set_xlabel(f"Drift β_WH − β_{rl} (change over one session)")
            AN = m017.stage_area_anova(A, order, dcol, "learning")
            rows.append(dict(panel=("g" if ref == "FA" else "j") + " ANOVA", measure=dcol,
                             **{f"{k} {s_}": AN[k][s_] for k in ("cohort", "area", "cohort x area") for s_ in ("F", "p", "p_perm")},
                             n_sessions=AN["n_sessions"], n_mice=AN["n_mice"], n_values=AN["n_values"]))
            ax.set_title("Per area group: R+ vs R− p (FDR q); filled ≠ 0\n" + m017.anova_line(AN), fontsize=5.0)
            rowax[ref] = axs
            # row label
            y0 = np.mean([axs[0].get_position().y0, axs[0].get_position().y1])
            fig.text(0.008, y0, f"Relative to {rname}", rotation=90, ha="left", va="center", fontsize=6, weight="bold",
                     color="0.25")
        # harmonise: drift dots share y within the two rows? (different references: separate), trajectories rows 2-3 same height
        m62.letter_row(fig, axs1, "abcd", dx_in=0.33, dy_in=0.13)
        m62.letter_row(fig, rowax["FA"], "efg", dx_in=0.33, dy_in=0.13)
        m62.letter_row(fig, rowax["AH"], "hij", dx_in=0.33, dy_in=0.13)
        from matplotlib.lines import Line2D
        rowax["FA"][2].legend(handles=[Line2D([], [], marker="o", ls="", color=COH[c], ms=3, label=c.replace("-", "−"))
                                       for c in ("R+", "R-")], frameon=False, loc="upper left", fontsize=4.6)
        fig.suptitle("On the learning day, whisker-hit pre-lick activity converges toward the rewarded-lick pattern in R+ "
                     "mice, not in R− mice", x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold")
        m62.save(fig, out, "COSYNE_day0_v2"); plt.close(fig)
    R = pd.DataFrame(rows); R.to_csv(out / "COSYNE_day0_v2_stats.csv", index=False)
    with pd.option_context("display.width", 250):
        print(R[R.panel.isin(["example", "f", "i"])].dropna(axis=1, how="all").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
