"""Behavioral comparison plots for the new curve-based perf-state definition
(user request 2026-09-16, after adopting `prep_perfstate_trials_curve` as
the pipeline default: "in behavior folder, plots comparative of
performance, learning curves across cohorts and states and trial types").
First script in this new `behavior/` folder (sibling to
`exploratory-analyses/`) -- a home for behavior-only comparisons that don't
depend on any neural decoding, as opposed to `exploratory-analyses/017`/
`031`/`037`'s older one-off behavioral scripts.

Two figures, using every learning-stage (day 0), has_ephys session with a
usable learning-curve file (88/89 as of 2026-09-16 -- see
`045_perfstate_curve_label_sweep.py`'s coverage sweep; the 1 exception,
MH038, has an unexplained 0-whisker-trial mismatch, logged not debugged):

1. `001_learning_curves_by_cohort.png`: grand-average `p_mean`/`p_chance`
   learning curve per cohort (R+, R-) -- each mouse's curve resampled onto
   a common 0-100% trial-index grid (sessions have different trial counts)
   before averaging, mean+-SEM across mice.

2. `001_performance_by_state_trialtype_cohort.png`: P(lick) grid, rows =
   trial_type (whisker/auditory/no_stim), cols = cohort (R+/R-), high vs
   low perf-state (new curve-based definition) as paired points per
   session -- paired because both states come from the same session, so
   the mandatory non-parametric+parametric pair
   (`ssl_rplus_rminus_test_pair.md`'s pairing convention, applied here to
   high-vs-low instead of R+-vs-R-) is Wilcoxon signed-rank + paired t-test,
   both annotated, never just one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_bwm_trial_prep import load_reward_group  # noqa: E402
from ssl_timeresolved_decoding import prep_perfstate_trials_curve  # noqa: E402

EPHYS_UTILS_PATH = axel_bisi_path("Github", "ephys_utilities")
if EPHYS_UTILS_PATH is not None:
    sys.path.insert(0, str(EPHYS_UTILS_PATH))
from ephys_utilities.helpers.load_helpers import load_learning_curves_data  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CURVE_ROOT = axel_bisi_path("combined_results_ks4")
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
STATE_COLOR = {"high": "#00838f", "low": "#c2185b"}
TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
PCT_GRID = np.linspace(0, 100, 50)


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    import time
    last_err = None
    for attempt in range(attempts):
        try:
            fig.savefig(out_path, **kwargs)
            return
        except OSError as e:
            last_err = e
            print(f"  savefig attempt {attempt+1}/{attempts} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    raise last_err


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    learning = sessions_tbl[(sessions_tbl["session_description"] == "whisker_0") & (sessions_tbl["has_ephys"])]
    print(f"{len(learning)} learning-stage has_ephys sessions")

    curves_by_cohort = {"R+": [], "R-": []}
    perf_records = []
    n_ok, n_skip = 0, 0
    for _, srow in learning.sort_values("subject_id").iterrows():
        session_id, mouse = srow["session_id"], srow["subject_id"]
        reward_group = load_reward_group(mouse)
        if reward_group is None:
            n_skip += 1
            continue

        # --- learning curve (p_mean/p_chance), resampled to 0-100% ---
        try:
            curves_df = load_learning_curves_data(str(CURVE_ROOT), [mouse])
        except ValueError:
            curves_df = pd.DataFrame()
        if len(curves_df) > 0:
            row = curves_df.iloc[0]
            p_mean, p_chance = np.asarray(row["p_mean"]), np.asarray(row["p_chance"])
            if len(p_mean) > 1:
                pct = np.linspace(0, 100, len(p_mean))
                curves_by_cohort[reward_group].append(dict(
                    mouse=mouse, p_mean=np.interp(PCT_GRID, pct, p_mean), p_chance=np.interp(PCT_GRID, pct, p_chance)))

        # --- performance by trial_type x state ---
        labeled = prep_perfstate_trials_curve(dataset_root, session_id, sessions_tbl, trials_tbl,
                                                decode_trial_types=TRIAL_TYPES)
        if labeled is None:
            n_skip += 1
            continue
        n_ok += 1
        for trial_type in TRIAL_TYPES:
            for state_val in ("high", "low"):
                sub = labeled[(labeled["trial_type"] == trial_type) & (labeled["perf_state"] == state_val)]
                if len(sub) == 0:
                    continue
                perf_records.append(dict(mouse=mouse, session_id=session_id, reward_group=reward_group,
                                          trial_type=trial_type, state=state_val,
                                          p_lick=float(sub["lick_flag"].mean()), n_trials=len(sub)))

    print(f"learning curves: {len(curves_by_cohort['R+'])} R+ mice, {len(curves_by_cohort['R-'])} R- mice")
    print(f"performance-by-state: {n_ok} sessions usable, {n_skip} skipped (no reward_group or no curve file)")

    perf_df = pd.DataFrame(perf_records)
    out_csv = OUT_DIR / "001_performance_by_state_trialtype_cohort.csv"
    perf_df.to_csv(out_csv, index=False)
    print(f"saved {out_csv.name}")

    # --- Figure 1: learning curves by cohort ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    for ax, cohort in zip(axes, ("R+", "R-")):
        entries = curves_by_cohort[cohort]
        if not entries:
            ax.axis("off")
            continue
        p_mean_stack = np.stack([e["p_mean"] for e in entries])
        p_chance_stack = np.stack([e["p_chance"] for e in entries])
        mean_curve, sem_curve = np.nanmean(p_mean_stack, axis=0), np.nanstd(p_mean_stack, axis=0) / np.sqrt(len(entries))
        chance_curve = np.nanmean(p_chance_stack, axis=0)
        ax.plot(PCT_GRID, mean_curve, color=COHORT_COLOR[cohort], lw=2.2, label="p_mean (mean+-SEM)")
        ax.fill_between(PCT_GRID, mean_curve - sem_curve, mean_curve + sem_curve, color=COHORT_COLOR[cohort], alpha=0.2, lw=0)
        ax.plot(PCT_GRID, chance_curve, color="#333333", lw=1.2, linestyle="--", label="p_chance (mean)")
        ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
        ax.set_title(f"{cohort} (n={len(entries)} mice)", fontsize=10)
        ax.set_xlabel("% of learning-stage session (whisker trials)", fontsize=8.5)
        ax.set_ylabel("probability", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("Learning curves by cohort (curve-based perf-state model)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_retry(fig, OUT_DIR / "001_learning_curves_by_cohort.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved 001_learning_curves_by_cohort.png")

    # --- Figure 2: performance by trial_type x state x cohort, paired stats ---
    fig, axes = plt.subplots(len(TRIAL_TYPES), 2, figsize=(8, 3.6 * len(TRIAL_TYPES)), squeeze=False)
    rng = np.random.default_rng(0)
    for row_i, trial_type in enumerate(TRIAL_TYPES):
        for col_i, cohort in enumerate(("R+", "R-")):
            ax = axes[row_i][col_i]
            sub = perf_df[(perf_df.trial_type == trial_type) & (perf_df.reward_group == cohort)]
            wide = sub.pivot_table(index="session_id", columns="state", values="p_lick")
            wide = wide.dropna(subset=["high", "low"]) if "high" in wide.columns and "low" in wide.columns else wide.iloc[0:0]
            n = len(wide)
            for _, r in wide.iterrows():
                ax.plot([0, 1], [r["high"], r["low"]], color="#bbbbbb", lw=0.6, alpha=0.6, zorder=1)
            for x, state_val in enumerate(("high", "low")):
                if state_val in wide.columns and len(wide) > 0:
                    vals = wide[state_val].to_numpy()
                    jitter = rng.uniform(-0.04, 0.04, size=len(vals))
                    ax.scatter(np.full(len(vals), x) + jitter, vals, color=STATE_COLOR[state_val], s=14, alpha=0.7, zorder=2)
                    ax.errorbar(x, vals.mean(), yerr=vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                                 fmt="D", color="black", markersize=6, capsize=3, zorder=3)
            stat_note = "n/a"
            if n >= 3:
                try:
                    w_stat, w_p = stats.wilcoxon(wide["high"], wide["low"])
                except ValueError:
                    w_p = float("nan")
                t_stat, t_p = stats.ttest_rel(wide["high"], wide["low"])
                stat_note = f"Wilcoxon p={w_p:.3g}, paired-t p={t_p:.3g}"
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["high", "low"])
            ax.set_ylim(-0.05, 1.05)
            ax.set_box_aspect(1)
            ax.set_title(f"{trial_type} | {cohort} (n={n} sessions)\n{stat_note}", fontsize=8)
            ax.set_ylabel("P(lick)", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Performance by trial type x perf-state x cohort (curve-based definition, paired per session)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_retry(fig, OUT_DIR / "001_performance_by_state_trialtype_cohort.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved 001_performance_by_state_trialtype_cohort.png")


if __name__ == "__main__":
    main()
