"""002 -- Statistics and figures of 001 (pre-stimulus state axis, gating, passive change).

Population: day-0 learners (sheet rule, learner_filter of ssl-prestim / prelick 061) and expert sessions (all); session = unit.
Per cohort, whole brain and per area (>= MIN_SESS sessions): mean +- s.e.m., Wilcoxon vs 0 (BH over areas within measure x
cohort), R+ vs R- Mann-Whitney + Welch; paired tests: active half 2 vs half 1 (state decodability, gating), passive change raw
vs controlled for state (ro_dW vs ro_dW_adj, g_dW vs g_dW_adj); day 0 vs expert (Mann-Whitney). Across sessions: Spearman of the
passive state shift with the passive readout change.
Output: tables/002_stats{_fine}.csv; figures/002_summary{_fine}.{png,pdf}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/002_summary.py [--fine]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m61 = importlib.import_module("061_roc_prelick_learners")
ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-prestim-state-gating"
TAB, FIG = ROOT / "tables", ROOT / "figures"
COH = {"R+": "#00B400", "R-": "#C800C8"}
MIN_SESS = 5
MEAS = [("state_bal_acc_excess", "State decodability\n(bal. acc. − shift null)"),
        ("gate_rho", "Gating ρ(state, W gain | lick, time)"), ("gate_rho_excess", "Gating ρ − shift null"),
        ("gate_rho_hits", "Gating within hits"), ("gate_rho_misses", "Gating within misses"),
        ("gate_rho_passive", "Gating, passive W trials"), ("gate_rho_aud", "Gating, auditory (A gain)"),
        ("state_dW", "Passive state shift post − pre\n(W trials; > 0 hit-like)"), ("state_dA", "Passive state shift (A trials)"),
        ("ro_dW", "Passive W sensory readout\npost − pre"), ("ro_dW_adj", "… controlled for state"),
        ("g_dW", "Passive W gain post − pre"), ("g_dW_adj", "… controlled for state")]


def two(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def wil(v):
    v = pd.Series(v).dropna()
    return stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan


def main(fine):
    sfx = "_fine" if fine else ""
    learners, _ = m61.learner_mice()
    D0 = pd.read_parquet(TAB / f"001_state_gating{sfx}.parquet")
    D0 = D0[D0.skipped_reason.isna() & D0.mouse_id.isin(learners)].assign(stage="day 0")
    fe = TAB / f"001_state_gating{sfx}_expert.parquet"
    DE = pd.read_parquet(fe) if fe.exists() else pd.DataFrame()
    if len(DE):
        DE = DE[DE.skipped_reason.isna()].assign(stage="expert")
    D = pd.concat([D0, DE], ignore_index=True)
    D["state_bal_acc_dh"] = D.state_bal_acc_h2 - D.state_bal_acc_h1
    D["gate_rho_dh"] = D.gate_rho_h2 - D.gate_rho_h1
    rows = []
    for (stage, area), g in D.groupby(["stage", "area"]):
        for m in [m for m, _ in MEAS] + ["state_bal_acc_dh", "gate_rho_dh"]:
            vals = {c: g[g.reward_group == c][m].dropna() for c in ("R+", "R-")}
            for c, v in vals.items():
                if len(v) >= MIN_SESS:
                    rows.append(dict(stage=stage, area=area, measure=m, cohort=c, n=len(v), mean=v.mean(), sem=v.sem(), p_vs0=wil(v)))
            mw, we = two(vals["R+"], vals["R-"])
            if np.isfinite(mw):
                rows.append(dict(stage=stage, area=area, measure=m, cohort="R+ vs R-", n=sum(len(v) for v in vals.values()),
                                 mean=vals["R+"].mean() - vals["R-"].mean(), p_vs0=mw, p_welch=we))
        for c in ("R+", "R-"):
            gc = g[g.reward_group == c]
            for a, b, lab in (("ro_dW", "ro_dW_adj", "ro raw vs adj"), ("g_dW", "g_dW_adj", "g raw vs adj")):
                pr = gc[[a, b]].dropna()
                if len(pr) >= MIN_SESS:
                    rows.append(dict(stage=stage, area=area, measure=lab, cohort=c, n=len(pr), mean=(pr[a] - pr[b]).mean(),
                                     p_vs0=stats.wilcoxon(pr[a], pr[b]).pvalue))
            pr = gc[["state_dW", "ro_dW"]].dropna()
            if len(pr) >= MIN_SESS:
                r, p = stats.spearmanr(pr.state_dW, pr.ro_dW)
                rows.append(dict(stage=stage, area=area, measure="rho(state_dW, ro_dW)", cohort=c, n=len(pr), mean=r, p_vs0=p))
    S = pd.DataFrame(rows)
    S["q_bh"] = np.nan
    sel = (S.area != "whole_brain") & S.cohort.isin(["R+", "R-"]) & S.p_vs0.notna()
    for _, g in S[sel].groupby(["stage", "measure", "cohort"]):
        S.loc[g.index, "q_bh"] = multipletests(g.p_vs0, method="fdr_bh")[1]
    # day 0 vs expert per cohort (whole brain)
    for m, _ in MEAS[:7]:
        for c in ("R+", "R-"):
            a = D[(D.stage == "day 0") & (D.area == "whole_brain") & (D.reward_group == c)][m]
            b = D[(D.stage == "expert") & (D.area == "whole_brain") & (D.reward_group == c)][m]
            mw, we = two(a, b)
            if np.isfinite(mw):
                S = pd.concat([S, pd.DataFrame([dict(stage="day 0 vs expert", area="whole_brain", measure=m, cohort=c,
                                                     n=a.notna().sum() + b.notna().sum(), mean=a.mean() - b.mean(), p_vs0=mw,
                                                     p_welch=we)])], ignore_index=True)
    S.to_csv(TAB / f"002_stats{sfx}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    wb = S[S.area == "whole_brain"]
    print(wb.round(4).to_string())
    sig = S[(S.area != "whole_brain") & (S.q_bh < 0.05)]
    print("\nareas with BH q < 0.05:\n", sig.round(4).to_string())

    # figure: whole brain (day 0 and expert) for every measure, then per-area gating and state decodability (day 0)
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, len(MEAS) // 2 + 1)
    rng = np.random.default_rng(0)
    for i, (m, lab) in enumerate(MEAS):
        ax = fig.add_subplot(gs[i // (len(MEAS) // 2 + 1), i % (len(MEAS) // 2 + 1)])
        for j, stage in enumerate(("day 0", "expert")):
            for k, c in enumerate(("R+", "R-")):
                v = D[(D.stage == stage) & (D.area == "whole_brain") & (D.reward_group == c)][m].dropna()
                x = j * 2.4 + k
                ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=COH[c], alpha=0.35 if stage == "expert" else 0.6)
                if len(v):
                    ax.errorbar(x + 0.3, v.mean(), v.sem(), fmt="o", color=COH[c], mfc="white" if stage == "expert" else COH[c], ms=4)
                    p = wil(v)
                    if np.isfinite(p) and p < 0.05:
                        ax.text(x + 0.3, ax.get_ylim()[1], "*" if p >= 0.01 else "**" if p >= 0.001 else "***", ha="center",
                                va="top", fontsize=7, color=COH[c])
        ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_xticks([0.5, 2.9], ["day 0", "expert"], fontsize=7)
        ax.set_title(lab, fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Pre-stimulus state and gating, whole brain (day-0 learners; expert open); * Wilcoxon vs 0", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"002_summary{sfx}.{ext}", dpi=170)

    # per-area figure (day 0): state decodability, gating, gating passive, state shift
    am = ["state_bal_acc_excess", "gate_rho", "gate_rho_passive", "state_dW"]
    A = S[(S.stage == "day 0") & (S.area != "whole_brain") & S.cohort.isin(["R+", "R-"]) & S.measure.isin(am)]
    areas = sorted(A.area.unique(), key=lambda a: -A[(A.measure == "state_bal_acc_excess") & (A.area == a)]["mean"].mean())
    fig, axes = plt.subplots(1, len(am), figsize=(4 * len(am), 0.28 * len(areas) + 1.5), sharey=True)
    for ax, m in zip(axes, am):
        for k, c in enumerate(("R+", "R-")):
            t = A[(A.measure == m) & (A.cohort == c)].set_index("area").reindex(areas)
            yy = np.arange(len(areas)) + (k - 0.5) * 0.3
            ax.errorbar(t["mean"], yy, xerr=t["sem"], fmt="o", ms=3, color=COH[c], elinewidth=0.6)
            for y0, x0, q in zip(yy, t["mean"], t["q_bh"]):
                if np.isfinite(q) and q < 0.05:
                    ax.text(x0, y0 - 0.15, "*", color=COH[c], fontsize=8, ha="center")
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_yticks(range(len(areas)), areas, fontsize=6)
        ax.set_title(dict(MEAS).get(m, m).replace("\n", " "), fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0].invert_yaxis()
    fig.suptitle("Per area (day-0 learners), mean ± s.e.m. over sessions; * BH q < 0.05 vs 0", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"002_areas{sfx}.{ext}", dpi=170)
    print(FIG / f"002_summary{sfx}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", action="store_true")
    main(ap.parse_args().fine)
