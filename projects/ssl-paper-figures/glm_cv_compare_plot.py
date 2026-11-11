"""Figure for glm_cv_compare.py: behaviour GLM v3 performance and weights under each cross-validation scheme.

a-c  pooled out-of-fold AUC, balanced accuracy and pseudo-R2 per session (grey lines join a session's schemes; for
     the repeated schemes, the mean over repeats; "avg" = the per-trial mean prediction over repeats, scored once)
d    per session, AUC with random folds vs contiguous blocks with a gap (identity line)
e    spread of the AUC over the 10 repeats (s.d. per session), random vs block
f    chosen C per scheme (all folds)
g    weights (mean over folds and sessions, standardised predictors) per scheme
h    example session: observed licks (running mean), and the mean prediction over repeats for random and block folds

Output: Figures/_explore/glm_cv_compare_<date>.{pdf,png} and glm_cv_compare_<date>_summary.csv.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/glm_cv_compare_plot.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, FIGURES_ROOT  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

IN = CACHE_DIR / "glm_cv_compare"
SCHEMES = ["random", "random_rep", "random_rep_avg", "block", "block_gap", "block_rep", "block_rep_avg"]
LABELS = {"random": "random\n(glm_v3)", "random_rep": "random\n×10", "random_rep_avg": "random\n×10 avg",
          "block": "block", "block_gap": "block\n+gap", "block_rep": "block+gap\n×10", "block_rep_avg": "block+gap\n×10 avg"}
COL = {s: ("#4a6fa5" if s.startswith("random") else "#c26b2b") for s in SCHEMES}


def per_session(m):
    return m.groupby(["session_id", "cohort", "scheme"])[["auc", "balanced_accuracy", "pseudo_r2"]].mean().reset_index()


def paired(ax, s, metric, ylabel, chance=None):
    w = s.pivot_table(index="session_id", columns="scheme", values=metric)[SCHEMES]
    x = np.arange(len(SCHEMES))
    for _, r in w.iterrows():
        ax.plot(x, r.to_numpy(), color="0.85", lw=0.3, zorder=1)
    for i, sc in enumerate(SCHEMES):
        v = w[sc].dropna()
        ax.errorbar(i, v.mean(), v.sem(), color=COL[sc], marker="o", ms=2.5, capsize=0, elinewidth=0.8, zorder=3)
    if chance is not None:
        ax.axhline(chance, color="0.5", lw=0.4, ls=":")
    ax.set_xticks(x, [LABELS[s] for s in SCHEMES], fontsize=4.2)
    ax.set_ylabel(ylabel, labelpad=1)
    ref = w["random"]
    for i, sc in enumerate(SCHEMES[1:], 1):  # paired Wilcoxon vs the glm_v3 scheme
        d = (w[sc] - ref).dropna()
        p = stats.wilcoxon(d).pvalue if len(d) > 5 and d.abs().sum() > 0 else np.nan
        ax.text(i, 1.0, "n.s." if not p < 0.05 else ("***" if p < 0.001 else "**" if p < 0.01 else "*"),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.5)
    return w


def main():
    set_nature_rc()
    m = pd.read_csv(IN / "metrics.csv")
    coefs = pd.read_csv(IN / "coefs.csv")
    preds = pd.read_parquet(IN / "predictions.parquet")
    s = per_session(m)
    n_ses, n_mice = s.session_id.nunique(), m.mouse_id.nunique()

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.99, top=0.95, bottom=0.07, hspace=0.75, wspace=0.45)
    axes = {}
    axes["a"] = fig.add_subplot(gs[0, 0:2])
    axes["b"] = fig.add_subplot(gs[0, 2:4])
    axes["c"] = fig.add_subplot(gs[1, 0:2])
    axes["d"], axes["e"] = fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    axes["f"], axes["g"] = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    axes["h"] = fig.add_subplot(gs[2, 2:4])
    w_auc = paired(axes["a"], s, "auc", "AUC (pooled out-of-fold)", 0.5)
    paired(axes["b"], s, "balanced_accuracy", "Balanced accuracy", 0.5)
    paired(axes["c"], s, "pseudo_r2", "Pseudo-R² (McFadden)", 0)

    ax = axes["d"]
    for c in ("R+", "R-"):
        ids = s[s.cohort == c].session_id.unique()
        ax.scatter(w_auc.loc[w_auc.index.isin(ids), "random"], w_auc.loc[w_auc.index.isin(ids), "block_gap"], s=3,
                   color=COLORS[c], lw=0, label=c)
    ax.plot([0.3, 1], [0.3, 1], color="0.5", lw=0.4, ls=":")
    ax.set_xlim(0.3, 1.01); ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("AUC, random folds", labelpad=1); ax.set_ylabel("AUC, blocks + gap", labelpad=1)
    ax.legend(fontsize=4.5, frameon=False, handletextpad=0.2, loc="upper left")

    ax = axes["e"]
    sd = m[m.scheme.isin(["random_rep", "block_rep"])].groupby(["session_id", "scheme"]).auc.std().unstack()
    for i, sc in enumerate(["random_rep", "block_rep"]):
        v = sd[sc].dropna()
        ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=2, color="0.7", lw=0)
        ax.errorbar(i + 0.25, v.mean(), v.sem(), color=COL[sc], marker="o", ms=2.5, capsize=0)
    ax.set_xticks([0, 1], ["random", "block+gap"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("AUC s.d. over 10 repeats", labelpad=1)

    ax = axes["f"]
    cf = coefs.drop_duplicates(["session_id", "scheme", "repeat", "fold"])
    for i, sc in enumerate(["random", "random_rep", "block", "block_gap", "block_rep"]):
        v = np.log10(cf[cf.scheme == sc].C)
        ax.violinplot(v, [i], widths=0.7, showextrema=False)
        ax.plot(i, v.median(), "o", ms=2, color=COL[sc])
    ax.set_xticks(range(5), ["rnd", "rnd×10", "blk", "blk+gap", "blk×10"], fontsize=4.2)
    ax.set_ylabel("log10 C (inner CV)", labelpad=1)

    ax = axes["g"]
    base = coefs[~coefs.feature.str.contains("_t[1-9]$", regex=True)]
    mean_w = base.groupby(["scheme", "feature"]).coefficient.mean().unstack(0)
    order = mean_w["random"].abs().sort_values(ascending=True).index
    y = np.arange(len(order))
    for k, sc in enumerate(["random", "block_gap", "block_rep"]):
        ax.plot(mean_w.loc[order, sc], y + (k - 1) * 0.2, "o", ms=1.8, color=COL[sc] if sc != "block_rep" else "#7a3b10",
                label=LABELS[sc].replace("\n", " "))
    ax.axvline(0, color="0.5", lw=0.4, ls=":")
    ax.set_yticks(y, [f.replace("lick_whisker_trial_", "lick ").replace("_", " ") for f in order], fontsize=3.8)
    ax.set_xlabel("Weight (mean)", labelpad=1)
    ax.legend(fontsize=3.8, frameon=False, loc="lower right", handletextpad=0.1)

    ax = axes["h"]
    best = s[s.scheme == "random"].sort_values("auc").session_id.iloc[len(s[s.scheme == "random"]) // 2]  # median AUC
    for sc, lab in (("random_rep", "random ×10, mean"), ("block_rep", "block+gap ×10, mean")):
        p = preds[(preds.session_id == best) & (preds.scheme == sc)].groupby("trial").p_lick.mean()
        ax.plot(p.index + 1, p, color=COL[sc], lw=0.7, label=lab)
    obs = preds[(preds.session_id == best) & (preds.scheme == "random")].sort_values("trial")
    ax.plot(obs.trial + 1, obs.y.rolling(10, center=True, min_periods=3).mean(), color="0.5", lw=0.7,
            label="observed (10-trial mean)")
    ax.scatter(obs.trial + 1, obs.y * 1.04 - 0.02, s=1.5, color="0.3", lw=0)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Whisker trial", labelpad=1); ax.set_ylabel("P(lick)", labelpad=1)
    ax.set_title(f"{best} (median random-fold AUC)", fontsize=5, pad=2)
    ax.legend(fontsize=4, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))

    for k, ax in axes.items():
        ax.text(-0.12 if k not in "ab c" else -0.06, 1.08, k, transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.text(0.01, 0.005, f"{n_ses} sessions (whisker day 0, learners), {n_mice} mice. Stars in a-c: paired Wilcoxon "
             "against random folds (glm_v3). 10 outer folds, inner 5-fold CV for C; gap = 3 whisker trials.", fontsize=4.5)
    out = FIGURES_ROOT / "_explore"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"glm_cv_compare_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)
    summ = s.groupby("scheme")[["auc", "balanced_accuracy", "pseudo_r2"]].agg(["mean", "sem"]).round(4)
    summ.to_csv(f"{stem}_summary.csv")
    print(summ.to_string())
    print("AUC s.d. over repeats:", sd.mean().round(4).to_dict())
    print("median C:", cf.groupby("scheme").C.median().to_dict())


if __name__ == "__main__":
    main()
