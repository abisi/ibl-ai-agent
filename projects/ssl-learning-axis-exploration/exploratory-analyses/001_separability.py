"""001 -- Separability of whisker hits from auditory hits and from spontaneous licks on the reward-lick CD, and the spread of whisker hits,
day 0 vs expert (user 2026-10-16, "explore 1-4": item 3). Events: 008 scores (both variants; held-out SL = 0, AH = 1).
Per session (whole session and early / late half by tau):
  auc_WA    AUC of WH vs AH scores (0.5 = indistinguishable; < 0.5 = WH below AH); d_WA = |d'| WH vs AH
  auc_WS    AUC of WH vs SL scores (> 0.5 = WH above SL)
  sd_WH     SD of WH scores; sd_ratio = SD(WH) / mean(SD(AH), SD(SL)) (spread relative to the reference classes)
  dauc_WA, dauc_WS: late - early
Statistics: common.measure_stats (Wilcoxon vs 0 -- vs 0.5 for AUCs and vs 1 for the SD ratio in the csv --, day 0 vs expert, R+ vs R-,
cohort x stage).
Output: tables/001_sessions.csv, 001_stats.csv; figures/001_separability_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

MEAS = [("auc_WA", "AUC WH vs AH (0.5 = same)"), ("auc_WS", "AUC WH vs SL (0.5 = same)"), ("dauc_WA", "Δ AUC WH vs AH (late − early)"),
        ("dauc_WS", "Δ AUC WH vs SL (late − early)"), ("sd_ratio", "WH spread / reference spread"), ("auc_WA_late", "AUC WH vs AH, late half")]


def auc(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))


def session(E):
    E = E.dropna(subset=["cd"]); g = {c: E[E.cls == c] for c in ("WH", "AH", "SL")}
    out = dict(auc_WA=auc(g["WH"].cd, g["AH"].cd), auc_WS=auc(g["WH"].cd, g["SL"].cd), sd_WH=g["WH"].cd.std(),
               sd_ratio=g["WH"].cd.std() / np.mean([g["AH"].cd.std(), g["SL"].cd.std()]))
    med = E.tau.median()
    for h, m in (("early", E.tau <= med), ("late", E.tau > med)):
        e = E[m]
        out[f"auc_WA_{h}"] = auc(e[e.cls == "WH"].cd, e[e.cls == "AH"].cd); out[f"auc_WS_{h}"] = auc(e[e.cls == "WH"].cd, e[e.cls == "SL"].cd)
    out["dauc_WA"] = out["auc_WA_late"] - out["auc_WA_early"]; out["dauc_WS"] = out["auc_WS_late"] - out["auc_WS_early"]
    return out


def main():
    E0 = pd.read_parquet(C.TAB_WD / "008_events.parquet")
    rows = []
    for v in ("corr", "raw"):
        E = E0.rename(columns={f"cd_{v}": "cd"})
        for sid, g in E.groupby("session_id"):
            rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0], variant=v, **session(g)))
    S = C.population(pd.DataFrame(rows)); S.to_csv(C.TAB / "001_sessions.csv", index=False)
    R = pd.concat([pd.DataFrame(C.measure_stats(S[S.variant == v], m, "whole brain", m)).assign(variant=v) for v in ("corr", "raw") for m, _ in MEAS])
    extra = []
    for v in ("corr", "raw"):
        for m, ref in (("auc_WA", 0.5), ("auc_WS", 0.5), ("auc_WA_late", 0.5), ("sd_ratio", 1.0)):
            for c in ("R+", "R-"):
                for s_ in ("learning", "expert"):
                    x = S[(S.variant == v) & (S.cohort == c) & (S.stage == s_)][m].dropna()
                    extra.append(dict(variant=v, measure=f"{m} vs {ref}", cohort=c, stage=s_, n=len(x), mean=x.mean(), p=stats.wilcoxon(x - ref).pvalue if len(x) >= 5 else np.nan))
    R.to_csv(C.TAB / "001_stats.csv", index=False); pd.DataFrame(extra).to_csv(C.TAB / "001_stats_ref.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    for v in ("corr", "raw"):
        fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.8), gridspec_kw=dict(wspace=0.55, hspace=1.3))
        for ax, (m, lab), l in zip(axs.ravel(), MEAS, "abcdef"):
            Rm = R[(R.variant == v) & (R.measure == m)]
            C.dot_panel(ax, S[S.variant == v], m, Rm, lab, l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
            if m.startswith("auc") and not m.startswith("dauc"):
                ax.axhline(0.5, color="0.4", lw=0.6, ls="--")
            if m == "sd_ratio":
                ax.axhline(1, color="0.4", lw=0.6, ls="--")
        fig.suptitle(f"Separability of whisker hits on the reward-lick CD ({'baseline-corrected' if v == 'corr' else 'raw'} rates; dashed: chance / "
                     "equal spread; p in panels: Wilcoxon vs 0 and day 0 vs expert)", fontsize=6.4, x=0.01, y=1.0, ha="left")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(C.FIG / f"001_separability_{v}.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(R[(R.variant == "corr") & R.cohort.isin(["R+", "R-"])][["measure", "cohort", "mean_d0", "mean_e", "p_stage_mwu", "cls"]].round(3).to_string(index=False))
    print(pd.DataFrame(extra)[lambda d: d.variant == "corr"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
