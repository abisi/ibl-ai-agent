"""Cohort-level (not session-level) trial-index-wise true-vs-predicted
learning curves (user follow-up to `065`'s per-session grid: "I want the
figure that replace session by cohorts to predict trial index wise the
learning curves").

Pure post-hoc on `065`'s already-computed cache (`065_perfquant_allmice_
cache.pkl` -- per-session z-scored `Y_true`/`Y_pred`, PLS+1SE, sensory
window, every good/moderate-learner session, both day_stages), no new
compute. `065` plots one panel PER SESSION; this instead pools sessions
WITHIN each cohort (R+/R-) onto a common trial-index axis and plots one
curve per cohort -- same targets, same PLS+1SE predictions, just grouped
by cohort instead of by session.

Sessions differ in trial count, so pooling requires a common x-axis:
each session's trial_index is rescaled to a fraction of that session's
own length (0=first trial, 1=last trial), then binned into N_BINS equal-
width bins. Within each (day_stage, cohort, target, bin), the mean (+/-
SEM across SESSIONS, not trials) of Y_true and of Y_pred is plotted --
i.e. "does the average predicted trajectory track the average real
trajectory, across a cohort's sessions, as a function of where you are
in the session" rather than "does it track one session's actual trial
sequence" (065's question). This is a distinct, coarser question:
real-trial-order fine structure within a session is exactly what
averaging across sessions on a normalized axis washes out.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
DAY_STAGES = ["learning", "expert"]
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
N_BINS = 20


def bin_session_onto_frac_axis(trial_index: np.ndarray, values: np.ndarray, n_bins: int) -> np.ndarray:
    """`values` is 1-D (one target column). Returns an (n_bins,) array of
    the within-session mean value per bin, NaN where the session has no
    trials in that bin (short sessions can skip bins near the tail)."""
    n = trial_index.max() if len(trial_index) else 0
    frac = trial_index / n if n > 0 else np.zeros_like(trial_index, dtype=float)
    bin_idx = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            out[b] = np.nanmean(values[mask])
    return out


def main():
    with open(OUT_DIR / "065_perfquant_allmice_cache.pkl", "rb") as f:
        results = pickle.load(f)

    bin_centers = (np.arange(N_BINS) + 0.5) / N_BINS

    fig, axes = plt.subplots(len(DAY_STAGES), len(TARGETS), figsize=(4.6 * len(TARGETS), 3.8 * len(DAY_STAGES)),
                              constrained_layout=True, squeeze=False)
    for row_i, day_stage in enumerate(DAY_STAGES):
        stage_results = [r for r in results if r["day_stage"] == day_stage]
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            k = TARGETS.index(target)
            n_sessions_note = []
            for cohort in ("R+", "R-"):
                coh_results = [r for r in stage_results if r["reward_group"] == cohort]
                if not coh_results:
                    continue
                true_binned = np.stack([bin_session_onto_frac_axis(r["trial_index"], r["Y_true"][:, k], N_BINS)
                                         for r in coh_results])
                pred_binned = np.stack([bin_session_onto_frac_axis(r["trial_index"], r["Y_pred"][:, k], N_BINS)
                                         for r in coh_results])
                true_mean = np.nanmean(true_binned, axis=0)
                true_sem = np.nanstd(true_binned, axis=0) / np.sqrt(np.sum(~np.isnan(true_binned), axis=0))
                pred_mean = np.nanmean(pred_binned, axis=0)
                pred_sem = np.nanstd(pred_binned, axis=0) / np.sqrt(np.sum(~np.isnan(pred_binned), axis=0))
                color = COHORT_COLOR[cohort]
                ax.plot(bin_centers, true_mean, color=color, lw=2.0, linestyle="-", label=f"{cohort} true")
                ax.fill_between(bin_centers, true_mean - true_sem, true_mean + true_sem, color=color, alpha=0.15, linewidth=0)
                ax.plot(bin_centers, pred_mean, color=color, lw=1.6, linestyle="--", alpha=0.85, label=f"{cohort} pred")
                ax.fill_between(bin_centers, pred_mean - pred_sem, pred_mean + pred_sem, color=color, alpha=0.10, linewidth=0)
                n_sessions_note.append(f"{cohort} n={len(coh_results)}")
            ax.set_title(f"{day_stage} -- {target}\n({', '.join(n_sessions_note)} sessions)", fontsize=9)
            ax.set_xlabel("fraction of session (trial index, normalized)", fontsize=7.5)
            ax.set_ylabel("z-scored target", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="best")
    fig.suptitle("PLS+1SE: cohort-pooled true vs predicted learning curves, trial-index-wise\n"
                 "(solid=true, dashed=predicted, shaded=SEM across sessions within cohort, binned onto normalized trial position)",
                 fontsize=11)
    fig_path = OUT_DIR / "087_perfquant_cohort_trialindex_curves.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig_path.name} (+ .pdf)")
    print("DONE_087")


if __name__ == "__main__":
    main()
