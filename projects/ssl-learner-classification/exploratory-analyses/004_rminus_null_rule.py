"""004 -- R+ learner rule with R- mice as the null (user 2026-11-23: "try recommendation 1"; FA-relative metrics are not
ideal because R+ mice get more rewards and are expected to false-alarm more, which lowers whisker - FA measures).

R- mice receive no whisker reward: whatever whisker licking they show on day 0 (generalisation from the auditory
association, impulsive licking) is what a mouse does without learning the whisker-reward association. A statistic s is
computed identically for every mouse; an R+ mouse is a learner if s exceeds the 95th percentile of s over R- mice (90th as
sensitivity). Statistics (exact day-0 curves on the whisker-trial axis, 001 cache):
  FA-relative (for comparison)
    frac_above      fraction of whisker trials with the whisker curve credibly above chance (80 % band, 001)
    peak_dprime     peak within-session d' (whisker vs chance curve, 001)
  whisker only (no FA)
    w_half2         mean whisker-curve P(lick) over the second half of the whisker trials
    w_sustained     maximum over 20-trial windows of the mean whisker-curve P(lick)
    frac_w_high     fraction of whisker trials whose whisker-curve P(lick) exceeds the R- level (95th percentile of all
                    R- whisker-curve values, pooled over mice and trials)
Per statistic: R- threshold, R+ learners, Cohen's kappa with the sheet (good / moderate) and with the agnostic consensus
label (001), AUC of the label for day +1 whisker P(lick) (FA-free) and d', disagreements with the sheet.
Outputs: tables/004_rminus_rule_per_mouse.csv, tables/004_rminus_rule_summary.csv, figures/004_rminus_rule.png|pdf.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/004_rminus_null_rule.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
sys.path.insert(0, str(REPO / "scripts"))
import paperstyle as ps  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

RES = axel_bisi_root() / "combined_results_ks4" / "ssl-learner-classification"
C = ps.COLORS
WIN = 20
QS = (95, 90)
STATS = {"frac_above": ("Fraction W trials credibly > chance", "FA-relative"),
         "peak_dprime": ("Peak within-session d'", "FA-relative"),
         "w_half2": ("W P(lick), second half", "whisker only"),
         "w_sustained": (f"Sustained W P(lick) (best {WIN}-trial window)", "whisker only"),
         "frac_w_high": ("Fraction W trials above the R- level", "whisker only")}


def whisker_stats(curves, level):
    rows = []
    for m, c in curves.items():
        w = np.asarray(c["whisker"], float)
        n = len(w)
        k = min(WIN, n)
        run = np.convolve(w, np.ones(k) / k, mode="valid")
        rows.append(dict(mouse_id=m, w_half2=float(w[n // 2:].mean()), w_sustained=float(run.max()),
                         frac_w_high=float((w > level).mean())))
    return pd.DataFrame(rows)


def kappa(a, b):
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(bool), b[ok].astype(bool)
    return float(cohen_kappa_score(a, b)) if a.nunique() > 1 or b.nunique() > 1 else np.nan


def auc(lab, y):
    ok = y.notna()
    return float(roc_auc_score(lab[ok], y[ok])) if lab[ok].nunique() == 2 else np.nan


def main():
    d = pd.read_csv(RES / "tables" / "001_per_mouse.csv")
    curves = pd.read_pickle(RES / "cache" / "001_curves_trials.pkl")
    rm = d[d.cohort == "R-"].mouse_id
    level = float(np.percentile(np.concatenate([np.asarray(curves[m]["whisker"]) for m in rm if m in curves]), 95))
    d = d.merge(whisker_stats(curves, level), on="mouse_id", how="left")
    p = d[d.cohort == "R+"].copy()
    rows = []
    for s, (lab, fam) in STATS.items():
        for q in QS:
            th = float(np.percentile(d[d.cohort == "R-"][s].dropna(), q))
            L = p[s] > th
            p[f"L_{s}_q{q}"] = L
            dis = p[L != p.sheet_learner.astype(bool)]
            rows.append(dict(statistic=s, label=lab, family=fam, q=q, threshold=th, n_rplus=len(p), n_learner=int(L.sum()),
                             kappa_sheet=kappa(L, p.sheet_learner), kappa_consensus=kappa(L, p.learner),
                             agree_sheet=float((L == p.sheet_learner.astype(bool)).mean()),
                             auc_w_day1=auc(L, p.hit_w_day1), auc_dprime_day1=auc(L, p.dprime_day1),
                             sheet_bad_learners=",".join(dis[L[dis.index]].mouse_id),
                             sheet_learner_non=",".join(dis[~L[dis.index]].mouse_id)))
    S = pd.DataFrame(rows)
    p.to_csv(RES / "tables" / "004_rminus_rule_per_mouse.csv", index=False)
    S.to_csv(RES / "tables" / "004_rminus_rule_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("R- whisker-curve level (95th pct pooled):", round(level, 3))
    print(S.drop(columns=["label"]).round(3).to_string(index=False))

    ps.set_nature_rc()
    fig, axs = plt.subplots(2, len(STATS), figsize=(183 / 25.4, 3.6), gridspec_kw=dict(hspace=0.9, wspace=0.55))
    for j, (s, (lab, fam)) in enumerate(STATS.items()):
        ax = axs[0, j]
        th = S[(S.statistic == s) & (S.q == 95)].threshold.iloc[0]
        for k, coh in enumerate(("R+", "R-")):
            g = d[d.cohort == coh]
            rng = np.random.default_rng(k)
            mk = g.learning_category.map({"good": "o", "moderate": "^", "bad": "x"}).fillna("o")
            for m in ("o", "^", "x"):
                sel = (mk == m).to_numpy()
                ax.scatter(k + rng.uniform(-0.15, 0.15, sel.sum()), g[s][sel], s=4 if m != "x" else 5, marker=m,
                           lw=0.5 if m == "x" else 0, color=C[coh] if m != "x" else "0.2", zorder=3)
        ax.axhline(th, color=C["R-"], lw=0.6, ls="--")
        ax.set_xticks([0, 1], ["R+", "R−"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"{lab}\n({fam})", fontsize=5)
        ax.tick_params(labelsize=5)
    axs[0, 0].set_ylabel("Statistic (day 0)", fontsize=5)
    for j, (col, ylab) in enumerate((("n_learner", "R+ learners (of %d)" % len(p)), ("kappa_sheet", "κ with sheet"),
                                     ("kappa_consensus", "κ with consensus"), ("auc_w_day1", "AUC, W P(lick) day +1"),
                                     ("auc_dprime_day1", "AUC, d′ day +1"))):
        ax = axs[1, j]
        for q, dx, a in ((95, -0.18, 1.0), (90, 0.18, 0.45)):
            v = S[S.q == q].set_index("statistic").reindex(list(STATS))[col]
            ax.bar(np.arange(len(STATS)) + dx, v, 0.34, color=["0.55" if STATS[s][1] == "FA-relative" else C["R+"]
                                                              for s in STATS], alpha=a, lw=0)
        ax.set_xticks(range(len(STATS)), ["frac > chance", "peak d′", "W 2nd half", "W sustained", "frac > R− level"],
                      rotation=45, ha="right")
        ax.set_title(ylab, fontsize=5)
        ax.tick_params(axis="y", labelsize=5)
        ax.tick_params(axis="x", labelsize=4.2)
        if col == "n_learner":
            ax.axhline(int(p.sheet_learner.sum()), color="0.3", lw=0.5, ls=":")
            ax.text(4.4, int(p.sheet_learner.sum()), "sheet", fontsize=4, va="bottom", ha="right")
    fig.suptitle("R+ learner = day-0 statistic above the R− 95th percentile (dashed; R− mice get no whisker reward). Top: "
                 "statistic per mouse (×, sheet bad; ▲, moderate; ●, good). Bottom: dark bars 95th, light 90th percentile;"
                 "\ngrey = relative to false alarms, green = whisker only. Exploratory.", fontsize=5.5, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"004_rminus_rule.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
