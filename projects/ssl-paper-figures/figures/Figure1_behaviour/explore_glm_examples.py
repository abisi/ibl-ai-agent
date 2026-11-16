"""Exploration (user 2026-11-12: "Change prediction example"): candidate sessions for Figure 1 l, GLM v6 with the new DLC.
Learner sessions of whisker day 0 with >= 100 whisker trials and a whisker lick rate of 0.25-0.75 (not the flat, mostly
licking MH070), per cohort the 6 with the highest held-out balanced accuracy; each drawn as panel l (held-out prediction,
mean over the 10 repeats, min-max spread; data = 10-trial running lick fraction).
Output: Figures/_explore/glm_examples_<date>.{pdf,png} + .csv (all ranked sessions).
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_glm_examples.py
"""

import datetime
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import compute  # noqa: E402
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import COLORS, MM, light, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
N_PER_COHORT, MIN_TRIALS, RATE = 6, 100, (0.25, 0.75)


def main():
    set_nature_rc()
    ctx = {"table": shared_behaviour.load()}
    learners = set(compute.day0_table(ctx["table"], learners_only=True).session_id)
    rows = []
    for r, f in compute._glm_files(ctx, "results_metrics.csv"):
        if r.session_id in learners:
            m = pd.read_csv(f).iloc[0]
            rows.append(dict(mouse_id=r.mouse_id, session_id=r.session_id, cohort=r.cohort, n_trials=m.n_trials,
                             lick_rate=m.lick_rate, test_balanced_accuracy=m.test_balanced_accuracy, test_auc=m.test_auc))
    allm = pd.DataFrame(rows)
    ok = allm[(allm.n_trials >= MIN_TRIALS) & allm.lick_rate.between(*RATE)]
    pick = ok.sort_values("test_balanced_accuracy", ascending=False).groupby("cohort").head(N_PER_COHORT)
    pick = pick.sort_values(["cohort", "test_balanced_accuracy"], ascending=[True, False])
    fig, axes = plt.subplots(2, N_PER_COHORT, figsize=(180 * MM, 62 * MM), squeeze=False)
    for i, cohort in enumerate(compute.COHORTS):
        for j, r in enumerate(pick[pick.cohort == cohort].itertuples()):
            ax = axes[i][j]
            f = compute._glm_dir() / r.mouse_id / f"{r.mouse_id}_{r.session_id}_results.pkl"
            res = pickle.load(open(f, "rb"))
            y, pred = np.asarray(res["y_all"]), np.asarray(res["y_pred_proba_oof"])
            pr = pd.read_parquet(str(f).replace("results.pkl", "predictions.parquet")).pivot(
                index="trial", columns="repeat", values="p_lick")
            x = np.arange(1, len(y) + 1)
            col = COLORS[cohort]
            ax.fill_between(x, pr.min(axis=1), pr.max(axis=1), color=light(col, 0.6), lw=0)
            ax.plot(x, pd.Series(y).rolling(10, center=True, min_periods=3).mean(), color="0.6", lw=0.6)
            ax.plot(x, pred, color=col, lw=0.7)
            ax.scatter(x[y == 1], np.full((y == 1).sum(), 1.06), s=1.5, marker="|", lw=0.3, color=COLORS["R+"])
            ax.scatter(x[y == 0], np.full((y == 0).sum(), -0.06), s=1.5, marker="|", lw=0.3, color=COLORS["R-"])
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(f"{r.mouse_id} ({cohort})\nbal. acc. {r.test_balanced_accuracy:.2f}, "
                         f"lick {r.lick_rate:.2f}, n {int(r.n_trials)}", fontsize=5, pad=2)
            ax.set_xlabel("Whisker trial", fontsize=5, labelpad=1)
            ax.set_ylabel("P(lick)", fontsize=5, labelpad=1)
            ax.tick_params(labelsize=4.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"glm_examples_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    allm.sort_values(["cohort", "test_balanced_accuracy"], ascending=[True, False]).to_csv(f"{stem}.csv", index=False)
    print(pick.to_string())
    print(stem)


if __name__ == "__main__":
    main()
