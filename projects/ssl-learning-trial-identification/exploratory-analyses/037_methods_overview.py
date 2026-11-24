"""037 -- Overview of the learning-trial (LT) definitions on the same whisker day-0 sessions (user 2026-11-23: "let us go
back to ways to create a learning trial, different methods"; the user does not like false-alarm-relative metrics because R+
mice get more rewards and are expected to false-alarm more).

Definitions (one LT per session or undefined), from the existing tables:
  exact rule     behaviour_analysis learning_curve_exact rule on the exact curves (paper Figure 1; learner-classification 001
                 exact_lt, only when exact_cat is learner / expert)                        [uses FA: p_low > p_chance]
  L0 stored      old PyMC pipeline LT                                                     [FA]
  L3 sustained   sustained P(whisker > FA) rule (003)                                     [FA]
  L5 whisker CP  whisker-only Bayesian change point (004)                                 [no FA]
  L6 joint CP    joint whisker + FA change point, strict (005)                            [FA]
  L6 lenient     same, log10 BF > 0                                                       [FA]
  L7 half-way    half-way point of the smooth whisker - FA curve (006)                    [FA]
  L8 margin      fixed whisker - FA margin held (013)                                     [FA]
  cascade        lenient cascade L6 lenient -> L5w -> L7 (007)                            [FA]
  LR             relative discrimination onset vs the mouse's own start (033; m 0.15, p 0.8) [FA]
  CP shuffle     single joint CP with lapse, shuffle-calibrated (034, FA stream)          [FA]
  CP delta       joint CP with effect floor delta 0.1, circular-shift null (035)          [FA]
  L6x relaxed    extended joint CP, log10 BF > 0.3, P(W > FA) > 0.9 (036)                 [FA]
Per cohort: coverage (sessions with an LT), LT position (trials, fraction of the whisker trials), pairwise agreement (median
|LT difference| where both defined, and the share of sessions where both are defined or both undefined), agreement of the
implied learner label (LT defined) with the sheet (Cohen's kappa).
Outputs: artifacts/037_methods_lt.csv (session x method), artifacts/037_methods_summary.csv, 037_methods_overview.png|pdf.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-learning-trial-identification/exploratory-analyses/037_methods_overview.py [--exact]
  --exact: L0-L8 / cascade from the 038 chain on the current exact curves (outputs with suffix _exact).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
LC = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-learner-classification/tables/001_per_mouse.csv")
COH = {"R+": "#00B400", "R-": "#C800C8"}
USES_FA = {"exact rule": True, "L0 stored": True, "L3 sustained": True, "L5 whisker CP": False, "L6 joint CP": True,
           "L6 lenient": True, "L7 half-way": True, "L8 margin": True, "cascade": True, "LR": True, "CP shuffle": True,
           "CP delta": True, "L6x relaxed": True}


EXACT = "--exact" in sys.argv   # 038 chain on the current exact curves (user 2026-11-23) instead of 028
TAG = "_exact" if EXACT else ""


def build():
    a = pd.read_csv(ART / ("038_learning_trials_all_methods_exact.csv" if EXACT else
                           "028_learning_trials_all_methods_all_mice.csv"))
    m = a[["session_id", "reward_group", "learning_category"]].copy()
    for src, dst in (("L0 stored", "L0 stored"), ("L3 sustained prob.", "L3 sustained"), ("L5 whisker CP", "L5 whisker CP"),
                     ("L6 joint CP", "L6 joint CP"), ("L6 lenient", "L6 lenient"), ("L7 half-way", "L7 half-way"),
                     ("L8 fixed margin", "L8 margin"), ("lenient cascade", "cascade")):
        m[dst] = a[src]
    lr = pd.read_csv(ART / "033_lr_learning_trials.csv")
    lr = lr[(lr.margin == 0.15) & (lr.p == 0.8)][["session_id", "LR", "n_whisker"]]
    m = m.merge(lr, on="session_id", how="left")
    c4 = pd.read_csv(ART / "034_single_cp.csv")
    c4 = c4[c4.stream == "FA"].assign(**{"CP shuffle": lambda x: x.LT.where(x.learner.astype(bool))})
    m = m.merge(c4[["session_id", "CP shuffle"]], on="session_id", how="left")
    c5 = pd.read_csv(ART / "035_single_cp_delta.csv")
    c5 = c5[np.isclose(c5.delta, 0.1)].assign(**{"CP delta": lambda x: x.LT.where(x.learner.astype(bool))})
    m = m.merge(c5[["session_id", "CP delta"]], on="session_id", how="left")
    c6 = pd.read_csv(ART / "036_joint_cp_relaxed.csv")
    c6 = c6[(c6.version == "L6x") & np.isclose(c6.bf_th, 0.3) & np.isclose(c6.p_th, 0.9)]
    c6 = c6.assign(**{"L6x relaxed": lambda x: x.LT.where(x.learner.astype(bool))})
    m = m.merge(c6[["session_id", "L6x relaxed"]], on="session_id", how="left")
    lc = pd.read_csv(LC)
    lc["exact rule"] = lc.exact_lt.where(lc.exact_cat.isin(["learner", "expert"]))
    m = m.merge(lc[["session_id", "exact rule"]], on="session_id", how="left")
    m = m.rename(columns={"reward_group": "cohort"})
    return m


def main():
    m = build()
    meths = list(USES_FA)
    m.to_csv(ART / f"037_methods_lt{TAG}.csv", index=False)
    rows = []
    for coh, g in m.groupby("cohort"):
        sheet = g.learning_category.map({"good": True, "moderate": True, "bad": False})
        for k in meths:
            v = g[k]
            lab = v.notna()
            ok = sheet.notna()
            frac = (v / g.n_whisker).dropna()
            rows.append(dict(cohort=coh, method=k, uses_fa=USES_FA[k], n=len(g), n_lt=int(lab.sum()),
                             coverage=float(lab.mean()), lt_median=float(v.median()), lt_frac_median=float(frac.median()),
                             kappa_sheet=float(cohen_kappa_score(lab[ok], sheet[ok].astype(bool)))
                             if lab[ok].nunique() > 1 else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(ART / f"037_methods_summary{TAG}.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))

    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(3, 2, hspace=0.7, wspace=0.45, height_ratios=[1, 1, 1.3])
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(meths))
    for i, coh in enumerate(("R+", "R-")):
        s = S[S.cohort == coh].set_index("method").reindex(meths)
        ax.barh(y + (i - 0.5) * 0.38, s.coverage, 0.36, color=COH[coh], label=coh)
    ax.set_yticks(y, [f"{k}{'' if USES_FA[k] else '  (no FA)'}" for k in meths], fontsize=5.5)
    ax.invert_yaxis()
    ax.set(xlabel="Sessions with a learning trial", xlim=(0, 1))
    ax.legend(fontsize=5, frameon=False)
    ax.set_title("A  Coverage", fontsize=7, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    for i, coh in enumerate(("R+", "R-")):
        s = S[S.cohort == coh].set_index("method").reindex(meths)
        ax.barh(y + (i - 0.5) * 0.38, s.kappa_sheet, 0.36, color=COH[coh])
    ax.set_yticks(y, meths, fontsize=5.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.4)
    ax.set(xlabel="Learner (LT defined) vs sheet, Cohen's κ")
    ax.set_title("B  Implied learner label", fontsize=7, loc="left")
    for i, coh in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[1, i])
        g = m[m.cohort == coh]
        data = [(g[k] / g.n_whisker).dropna().to_numpy() for k in meths]
        ax.boxplot(data, vert=False, widths=0.6, showfliers=False, medianprops=dict(color="k"),
                   boxprops=dict(color=COH[coh]), whiskerprops=dict(color=COH[coh]), capprops=dict(color=COH[coh]))
        for j, dd in enumerate(data):
            ax.scatter(dd, np.full(len(dd), j + 1) + np.random.default_rng(j).uniform(-0.2, 0.2, len(dd)), s=2,
                       color=COH[coh], alpha=0.5, lw=0)
        ax.set_yticks(range(1, len(meths) + 1), meths, fontsize=5.5)
        ax.invert_yaxis()
        ax.set(xlabel="Learning trial (fraction of whisker trials)", xlim=(0, 1))
        ax.set_title(f"{'CD'[i]}  LT position, {coh}", fontsize=7, loc="left")
    for i, coh in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[2, i])
        g = m[m.cohort == coh]
        M = np.full((len(meths), len(meths)), np.nan)
        for a_, ka in enumerate(meths):
            for b_, kb in enumerate(meths):
                both = g[ka].notna() & g[kb].notna()
                if a_ < b_ and both.sum() >= 3:
                    M[a_, b_] = np.median(np.abs(g[ka][both] - g[kb][both]))
                elif a_ > b_:
                    M[a_, b_] = float((g[ka].notna() == g[kb].notna()).mean())
        im = ax.imshow(np.where(np.triu(np.ones_like(M), 1) > 0, np.nan, M), cmap="Greys", vmin=0, vmax=1)
        im2 = ax.imshow(np.where(np.triu(np.ones_like(M), 1) > 0, M, np.nan), cmap="Oranges", vmin=0, vmax=40)
        for a_ in range(len(meths)):
            for b_ in range(len(meths)):
                if np.isfinite(M[a_, b_]):
                    ax.text(b_, a_, f"{M[a_, b_]:.0f}" if a_ < b_ else f"{M[a_, b_]:.1f}"[1:], ha="center", va="center",
                            fontsize=3.6, color="k")
        ax.set_xticks(range(len(meths)), meths, rotation=60, ha="right", fontsize=4.8)
        ax.set_yticks(range(len(meths)), meths, fontsize=4.8)
        ax.set_title(f"{'EF'[i]}  Agreement, {coh}: upper = median |ΔLT| (trials),\nlower = share defined / undefined alike",
                     fontsize=6, loc="left")
    fig.suptitle(f"Learning-trial definitions on whisker day 0 ({'038 chain on the current exact curves' if EXACT else '028 chain'}"
                 "; 033-036 and the exact rule of Figure 1). Exploratory.", fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"037_methods_overview{TAG}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
