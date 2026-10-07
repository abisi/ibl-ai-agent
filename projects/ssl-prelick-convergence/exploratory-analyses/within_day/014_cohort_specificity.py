"""014 -- Which within-day cohort differences (R+ vs R-) are specific to the learning day? (user question 2026-10-07: "is
there a cohort difference specific to the learning day, not visible in expert sessions?")

For every session-level within-day measure: R+ vs R- on day 0 and in expert sessions (Mann-Whitney U, Welch t, mouse-level
cohort permutation, 001 perm_interaction kind='learning' applied per stage), and the cohort x stage interaction
[(E - L)(R+) - (E - L)(R-)] (mouse-level permutation, 10000). Effect size: Hedges' g of R+ - R- per stage with a normal
approximation 95% CI. Benjamini-Hochberg FDR over measures, separately for the day-0 test and for the interaction.
A measure is called learning-day specific when the day-0 permutation p < 0.05, the expert permutation p >= 0.05 and the
interaction p < 0.05; "day 0 only (interaction n.s.)" when the interaction does not reach 0.05.

Measures (sources):
  cd_trajectories/<pop>/sessions.csv (011): drift_WH_SL, drift_WH_AH, beta_WH, beta_SL, beta_AH, mean WH CD score.
  halves/within_session_sessions.csv (001, split 'time'): late - early of dd, lambda, d(WH,SL), d(WH,AH), d(AH,SL),
      decoder transfer (ws / xh balanced accuracy, chance-corrected), WH / AH / SL rate, WH / AH reaction time; session
      level (mean of halves) of dd and lambda.
  slopes/trial_slopes_sessions.csv (002): slopes of WH - SL and WH - AH on the mean-difference axis (md) and decoder (dec).
  epochs_n4/epoch_sessions.csv (003): late - early and |late - early| of position, normalised distance and decoder transfer.
Output: within_day/<ref>/cohort_specificity/<pop>/cohort_specificity.csv + cohort_specificity.{png,pdf,svg}.
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
OUT = BASE / "cohort_specificity"
KEY = ["session_id", "mouse_id", "cohort", "stage"]


def measures(pop):
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    M = []                                                    # (family, name, label, frame with KEY + 'v')
    add = lambda fam, name, lab, d, col: M.append((fam, name, lab, d[KEY].assign(v=d[col].to_numpy())))
    C = pd.read_csv(BASE / "cd_trajectories" / pop / "sessions.csv")
    for col, lab in (("drift_WH_SL", "β_WH − β_SL"), ("drift_WH_AH", "β_WH − β_AH"), ("beta_WH", "β_WH"),
                     ("beta_FA", "β_SL"), ("beta_AH", "β_AH"), ("mean_WH", "mean WH score")):
        add("Reward-lick CD (011)", col, lab, C, col)
    H = flt(pd.read_csv(BASE / "halves" / "within_session_sessions.csv"))
    H = H[H.split == "time"].copy()
    for m, lab in (("dd", "Δd"), ("lam", "λ"), ("d_WH_FA", "d(WH, SL)"), ("d_WH_AH", "d(WH, AH)"), ("d_AH_FA", "d(AH, SL)"),
                   ("ws_bacc_c", "decoder within-half"), ("xh_bacc_c", "decoder cross-half"), ("rate_WH", "WH rate"),
                   ("rate_AH", "AH rate"), ("rate_FA", "SL rate"), ("rt_WH", "WH reaction time"), ("rt_AH", "AH reaction time")):
        H["_d"] = H[f"{m}_late"] - H[f"{m}_early"]
        add("Halves, late − early (001)", f"halves_delta_{m}", f"{lab}: late − early", H, "_d")
    for m, lab in (("dd", "Δd"), ("lam", "λ")):
        H["_m"] = H[[f"{m}_early", f"{m}_late"]].mean(1)
        add("Halves, session level (001)", f"halves_level_{m}", f"{lab}: session level", H, "_m")
    T = flt(pd.read_csv(BASE / "slopes" / "trial_slopes_sessions.csv"))
    for col, lab in (("md_WH-FA_slope", "md axis: WH − SL slope"), ("md_WH-AH_slope", "md axis: WH − AH slope"),
                     ("dec_WH-FA_slope", "decoder: WH − SL slope"), ("dec_WH-AH_slope", "decoder: WH − AH slope")):
        add("Trial slopes (002)", f"slopes_{col}", lab, T, col)
    E = flt(pd.read_csv(BASE / "epochs_n4" / "epoch_sessions.csv"))
    E["half"] = E.epoch.str.split("-").str[1]
    for m, lab in (("pos", "position"), ("ddn", "Δd / d(AH, SL)"), ("dec", "decoder transfer")):
        P = E.pivot_table(index=KEY, columns="half", values=m).reset_index()
        P["_d"] = P.late - P.early; P["_a"] = P._d.abs()
        add("Epochs, common footing (003)", f"epochs_delta_{m}", f"{lab}: late − early", P, "_d")
        add("Epochs, common footing (003)", f"epochs_abs_{m}", f"{lab}: |late − early|", P, "_a")
    return M


def hedges(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g, g - 1.96 * se, g + 1.96 * se


def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full_like(p, np.nan)
    if ok.any():
        pv = p[ok]; o = np.argsort(pv); r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        q[np.where(ok)[0][o]] = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    return q


def main(a):
    plt = m62.setup()
    out = OUT / a.population
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fam, name, lab, d in measures(a.population):
        d = d.dropna(subset=["v"])
        r = dict(family=fam, measure=name, label=lab)
        for st in ("learning", "expert"):
            x = d[(d.cohort == "R+") & (d.stage == st)].v; y = d[(d.cohort == "R-") & (d.stage == st)].v
            r[f"{st} n R+"], r[f"{st} n R-"] = len(x), len(y)
            r[f"{st} mean R+"], r[f"{st} mean R-"] = x.mean(), y.mean()
            if len(x) >= 3 and len(y) >= 3:
                r[f"{st} g"], r[f"{st} g lo"], r[f"{st} g hi"] = hedges(x, y)
                r[f"{st} p_MWU"] = stats.mannwhitneyu(x, y).pvalue
                r[f"{st} p_Welch"] = stats.ttest_ind(x, y, equal_var=False).pvalue
                _, r[f"{st} p_perm"] = m001.perm_interaction(d[d.stage == st].assign(stage="learning").rename(columns={"v": "val"}),
                                                             "val", kind="learning")
        r["cxs"], r["cxs p_perm"] = m001.perm_interaction(d.rename(columns={"v": "val"}), "val", kind="stage")
        rows.append(r)
    R = pd.DataFrame(rows)
    R["learning q_BH"] = bh(R["learning p_perm"]); R["cxs q_BH"] = bh(R["cxs p_perm"])
    d0 = R["learning p_perm"] < 0.05; ex = R["expert p_perm"] < 0.05; ix = R["cxs p_perm"] < 0.05
    R["call"] = np.select([d0 & ~ex & ix, d0 & ~ex, d0 & ex, ~d0 & ex], ["learning-day specific", "day 0 only (interaction n.s.)",
                                                                       "both stages", "expert only"], "none")
    R.to_csv(out / "cohort_specificity.csv", index=False)
    # forest plot: Hedges' g (R+ - R-) per stage
    n = len(R)
    fig, ax = plt.subplots(figsize=(m62.W_IN * 0.62, 0.17 * n + 0.9))
    y = np.arange(n)[::-1]
    for st, dy, mfc in (("learning", 0.15, "white"), ("expert", -0.15, "0.2")):
        ax.errorbar(R[f"{st} g"], y + dy, xerr=[R[f"{st} g"] - R[f"{st} g lo"], R[f"{st} g hi"] - R[f"{st} g"]], fmt="o",
                    ms=2.8, color="0.2", mfc=mfc, lw=0.7, capsize=0, label="Day 0" if st == "learning" else "Expert")
        sig = R[f"{st} p_perm"] < 0.05
        ax.scatter(R[f"{st} g"][sig], (y + dy)[sig.to_numpy()], marker="*", s=14, color="#D62728", zorder=5, lw=0)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [f"{l}" + ("  ◆" if c == "learning-day specific" else "") for l, c in zip(R.label, R.call)], fontsize=4.6)
    fams = R.family.to_numpy()
    for i in range(1, n):
        if fams[i] != fams[i - 1]:
            ax.axhline(y[i] + 0.5, color="0.85", lw=0.5)
    ax.set_xlabel("Hedges' g, R+ − R− (± 95% CI)")
    ax.set_title(f"Cohort difference on day 0 (open) and in expert sessions (filled) ({a.population})\n"
                 "red star: mouse-level permutation p < 0.05; ◆ learning-day specific\n(day 0 p < 0.05, expert n.s., "
                 "interaction p < 0.05)", fontsize=5.2)
    ax.legend(frameon=False, fontsize=4.6, loc="lower right")
    fig.tight_layout()
    m62.save(fig, out, "cohort_specificity"); plt.close(fig)
    cols = ["label", "learning mean R+", "learning mean R-", "learning p_perm", "expert mean R+", "expert mean R-",
            "expert p_perm", "cxs p_perm", "learning q_BH", "call"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(R[cols].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
