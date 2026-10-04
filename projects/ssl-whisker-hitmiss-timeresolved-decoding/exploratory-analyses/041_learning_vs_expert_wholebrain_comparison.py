"""Side-by-side learning-vs-expert comparison for the whole_brain statistics
figures (user request 2026-09-14: "for statistics figures whole-brain, do
side by side comparison learning vs expert").

One combined figure, 4 rows (hitmiss, perfstate, modality-stim, modality-
lick) x 3 columns (R+, R-, aggregated): each panel is a two-group jittered
strip plot (learning vs expert) of window-quantified decoding accuracy,
mean+-SEM, chance line at 0.5, annotated with the same non-parametric+
parametric test pair used everywhere else in this project (Mann-Whitney,
Welch) plus a mouse-block permutation p-value.

The mouse-block permutation matters more here than for the R+/R- cohort
comparisons elsewhere: a mouse typically contributes exactly one
learning-stage session but several expert-stage sessions, so the two
groups being compared are NOT independent samples of sessions -- see
`learning_expert_group_test`'s docstring in ssl_timeresolved_decoding.py.
The plain Mann-Whitney/Welch pair is still reported (matching this
project's established R+/R- convention and for comparability with the
other cohort-comparison figures), but the mouse-block permutation p-value
is the one to trust if it disagrees with them.

Restricted to condition_type=='whole' (the only condition_type common to
all four targets, including perfstate which has no half/perfstate
sub-conditions of its own) -- this is a stage comparison, not a
within-condition one, so 'whole' is the right level regardless.

Usage: python 041_learning_vs_expert_wholebrain_comparison.py
(no argument -- always whole_brain, always all 4 targets; area_group/
area_acronym_custom are out of scope here, same reasoning as documented in
025-030: their multi-row-per-session shape needs per-area faceting this
script doesn't do.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ssl_timeresolved_decoding import learning_expert_group_test, mean_in_window  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures" / "whole_brain" / "learning_vs_expert"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SENSORY_WINDOW = (0.005, 0.050)  # updated 2026-09-18 from (0.005, 0.035), user request
PRE_LICK_WINDOW = (-0.100, 0.0)
CHANCE = 0.5
STAGE_COLOR = {"learning": "#1f77b4", "expert": "#d62728"}

# (learning_tag, expert_tag, window_name, window, row_title)
ROWS = [
    ("hitmiss_stim_whole_brain", "hitmiss_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Hit vs miss"),
    ("perfstate_stim_whole_brain", "perfstate_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Performance state"),
    ("modality_stim_whole_brain", "modality_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Modality (stim)"),
    ("modality_lick_whole_brain", "modality_lick_expert_whole_brain", "pre_lick", PRE_LICK_WINDOW, "Modality (lick)"),
]
COHORTS = ["R+", "R-", "aggregated"]


def load_whole(tag: str, window: tuple[float, float]) -> pd.DataFrame | None:
    partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
    edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
    if not partial_path.exists() or not edges_path.exists():
        return None
    df = pd.read_parquet(partial_path)
    df = df[(df["skipped_reason"].isna()) & (df["condition_type"] == "whole")].copy()
    if len(df) == 0:
        return None
    edges = json.loads(edges_path.read_text())
    bin_labels = np.array([e[1] for e in edges])
    df["metric"] = df["real_curve"].apply(lambda c: mean_in_window(np.asarray(c, dtype=float), bin_labels, window))
    return df[["session_id", "subject_id", "reward_group", "metric"]]


def stage_panel(ax, learning: np.ndarray, expert: np.ndarray, title: str, n_mice: int | None = None,
                 perm_p: float | None = None):
    rng_jitter = np.random.default_rng(0)
    for i, (vals, stage) in enumerate([(learning, "learning"), (expert, "expert")]):
        vals = vals[~np.isnan(vals)]
        color = STAGE_COLOR[stage]
        jitter = rng_jitter.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, alpha=0.4, s=10, zorder=2)
        if len(vals):
            mean_v, sem_v = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
            ax.errorbar([i], [mean_v], yerr=[sem_v], color=color, lw=2.5, marker="o", markersize=7, zorder=3, capsize=3)
    from scipy import stats

    l_valid, e_valid = learning[~np.isnan(learning)], expert[~np.isnan(expert)]
    n_l, n_e = len(l_valid), len(e_valid)
    if n_l >= 2 and n_e >= 2:
        mw_p = stats.mannwhitneyu(l_valid, e_valid, alternative="two-sided").pvalue
        welch_p = stats.ttest_ind(l_valid, e_valid, equal_var=False).pvalue
        p_text = f"learning n={n_l}, expert n={n_e}\nMann-Whitney p={mw_p:.3g}\nWelch p={welch_p:.3g}"
        if perm_p is not None:
            p_text += f"\nmouse-block perm p={perm_p:.3g} (n_mice={n_mice})"
    else:
        p_text = f"learning n={n_l}, expert n={n_e}"
    ax.text(0.5, 0.02, p_text, transform=ax.transAxes, fontsize=7, ha="center", va="bottom")
    ax.axhline(CHANCE, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["learning", "expert"], fontsize=9)
    ax.set_ylim(0.35, 1.02)  # headroom so markers/errorbars at the accuracy ceiling aren't clipped
    ax.set_title(title, fontsize=9.5)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    rows_ready = []
    for learning_tag, expert_tag, window_name, window, row_title in ROWS:
        learning_df = load_whole(learning_tag, window)
        expert_df = load_whole(expert_tag, window)
        if learning_df is None or expert_df is None:
            print(f"{row_title}: missing learning and/or expert results, skipping "
                  f"({'learning ready' if learning_df is not None else 'learning missing'}, "
                  f"{'expert ready' if expert_df is not None else 'expert missing'})")
            continue
        rows_ready.append((row_title, learning_df, expert_df))

    if not rows_ready:
        print("nothing ready yet -- no learning+expert pair both computed")
        return

    fig, axes = plt.subplots(len(rows_ready), 3, figsize=(3.2 * 3, 3.2 * len(rows_ready)), squeeze=False)
    for r, (row_title, learning_df, expert_df) in enumerate(rows_ready):
        for c, cohort in enumerate(COHORTS):
            ax = axes[r][c]
            if cohort == "aggregated":
                l_sub, e_sub = learning_df, expert_df
            else:
                l_sub, e_sub = learning_df[learning_df.reward_group == cohort], expert_df[expert_df.reward_group == cohort]

            combined = pd.concat([
                l_sub.assign(day_stage="learning"),
                e_sub.assign(day_stage="expert"),
            ], ignore_index=True)
            n_mice = combined["subject_id"].nunique()
            perm_p = None
            if l_sub["metric"].notna().sum() >= 2 and e_sub["metric"].notna().sum() >= 2:
                try:
                    perm_p = learning_expert_group_test(combined, np.random.default_rng(0))["mouse_block_perm_p"]
                except Exception:
                    perm_p = None

            stage_panel(ax, l_sub["metric"].to_numpy(), e_sub["metric"].to_numpy(),
                        f"{row_title} | {cohort}", n_mice=n_mice, perm_p=perm_p)
    fig.suptitle("whole_brain -- learning vs expert, window-quantified decoding accuracy", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = FIG_DIR / "041_learning_vs_expert_whole_brain_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name} ({len(rows_ready)}/4 targets ready)")


if __name__ == "__main__":
    main()
