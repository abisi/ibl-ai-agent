"""Regenerable progress figures for `024_master_sweep.py`'s whole-brain-only
runs, using the locked figure template from `023_initial_check_hitmiss_halves.py`
(3-panel R+/R-/aggregated, square, n-annotated, larger axis text). Since
`whole_brain` is a single pseudo-area, no per-area faceting is needed here --
one figure per condition_type present in the data (whole / half / perfstate),
each showing the relevant within-condition comparison. Causal bins are
labeled at their **end** (not center, unlike the old sliding-window figures)
-- `bin_labels_ms = edge[1] * 1000`.

Run this repeatedly as `024_master_sweep.py ... whole_brain` accumulates
sessions; safe to call before any given combo/condition has landed (skips
what's missing, plots what's there).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent


def fig_dir(tag: str) -> Path:
    """Figures organized per area level and, under that, per day-stage --
    both read off the tag itself, matching `032_plot_decode_results.py`'s
    `fig_dir()` (2026-09-14: this script previously wrote to a flat
    `figures/whole_brain/` directory, orphaning the day-stage-nested copies
    `032` had already been writing next to; unified onto one convention)."""
    if "whole_brain" in tag:
        sub = "whole_brain"
    elif "area_acronym_custom" in tag:
        sub = "area_acronym_custom"
    elif "area_group" in tag:
        sub = "area_group"
    else:
        sub = "other"
    stage = "expert" if "_expert" in tag else "learning"
    d = OUT_DIR / "figures" / sub / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
AGGREGATE_COLOR = "#2c5f5b"

# (combo_tag, title_prefix)
COMBOS = [
    ("hitmiss_stim_whole_brain", "Hit vs miss, start_time-aligned, whole brain"),
    ("modality_stim_whole_brain", "Modality, start_time-aligned, whole brain"),
    ("modality_lick_whole_brain", "Modality, first-lick-aligned (corrected), whole brain"),
    # Expert-stage (whole_brain) variants (user request 2026-09-14: "extend
    # that fuller treatment to all"). perfstate excluded throughout -- it
    # only ever computes 'whole', no half/perfstate sub-conditions to
    # compare against itself. area_group NOT added here: its results have
    # ~30 rows/session (one per area x condition, via `area_col`/
    # `area_value`), but this script's per-session grouping assumes one row
    # per session (true for whole_brain's single pseudo-area) -- adding
    # area_group tags as-is would silently pool different areas together.
    # Needs per-area faceting (like 032/034 already do) before extending
    # here; flagged to the user 2026-09-14, not yet resolved.
    ("hitmiss_stim_expert_whole_brain", "Hit vs miss, start_time-aligned, whole brain (expert)"),
    ("modality_stim_expert_whole_brain", "Modality, start_time-aligned, whole brain (expert)"),
    ("modality_lick_expert_whole_brain", "Modality, first-lick-aligned (corrected), whole brain (expert)"),
]


def summarize(sub: pd.DataFrame) -> str:
    return f"n={sub['session_id'].nunique()} sessions, {sub['subject_id'].nunique()} mice"


def plot_condition_comparison(df: pd.DataFrame, bin_labels_ms: np.ndarray, condition_type: str,
                               values: tuple[str, str], title: str, out_path: Path, xlabel: str):
    sub_ct = df[df["condition_type"] == condition_type]
    if len(sub_ct) == 0:
        print(f"  no '{condition_type}' rows yet, skipping {out_path.name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [("R+", sub_ct[sub_ct.reward_group == "R+"]), ("R-", sub_ct[sub_ct.reward_group == "R-"]), ("R+ & R- aggregated", sub_ct)]

    any_data = False
    for ax, (label, sub) in zip(axes, panels):
        color = COHORT_COLOR.get(label, AGGREGATE_COLOR)
        for value, ls in zip(values, ("-", "--")):
            val_sub = sub[sub["condition_value"] == value]
            curves = [np.array(c) for c in val_sub["real_curve"]]
            if len(curves) == 0:
                continue
            any_data = True
            stacked = np.stack(curves)
            mean_curve = np.nanmean(stacked, axis=0)
            sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
            ax.plot(bin_labels_ms, mean_curve, color=color, lw=2.2, linestyle=ls, label=value)
            ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18, lw=0)

        ax.axhline(0.5, color="#888888", lw=1.2, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_title(f"{label}\n({summarize(sub)})", fontsize=13)
        ax.set_ylim(0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped
        _xpad = 0.02 * (bin_labels_ms.max() - bin_labels_ms.min())
        ax.set_xlim(bin_labels_ms.min() - _xpad, bin_labels_ms.max() + _xpad)
        ax.set_box_aspect(1)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, frameon=False, loc="upper left")

    if not any_data:
        plt.close(fig)
        print(f"  '{condition_type}' rows present but no curves yet, skipping {out_path.name}")
        return

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name} ({summarize(sub_ct)})")


def plot_whole(df: pd.DataFrame, bin_labels_ms: np.ndarray, title: str, out_path: Path, xlabel: str):
    sub_ct = df[df["condition_type"] == "whole"]
    if len(sub_ct) == 0:
        print(f"  no 'whole' rows yet, skipping {out_path.name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [("R+", sub_ct[sub_ct.reward_group == "R+"]), ("R-", sub_ct[sub_ct.reward_group == "R-"]), ("R+ & R- aggregated", sub_ct)]
    for ax, (label, sub) in zip(axes, panels):
        color = COHORT_COLOR.get(label, AGGREGATE_COLOR)
        curves = [np.array(c) for c in sub["real_curve"]]
        if curves:
            stacked = np.stack(curves)
            mean_curve = np.nanmean(stacked, axis=0)
            sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
            ax.plot(bin_labels_ms, mean_curve, color=color, lw=2.2)
            ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18, lw=0)
        ax.axhline(0.5, color="#888888", lw=1.2, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_title(f"{label}\n({summarize(sub)})", fontsize=13)
        ax.set_ylim(0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped
        _xpad = 0.02 * (bin_labels_ms.max() - bin_labels_ms.min())
        ax.set_xlim(bin_labels_ms.min() - _xpad, bin_labels_ms.max() + _xpad)
        ax.set_box_aspect(1)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name} ({summarize(sub_ct)})")


def main():
    for tag, title_prefix in COMBOS:
        partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
        edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
        if not partial_path.exists():
            print(f"{tag}: no results file yet, skipping")
            continue
        if not edges_path.exists():
            print(f"{tag}: no bin-edges file at {edges_path.name}, skipping")
            continue

        df = pd.read_parquet(partial_path)
        df = df[df["skipped_reason"].isna()].copy()
        if len(df) == 0:
            print(f"{tag}: results file exists but no computed rows yet")
            continue

        bin_edges = json.loads(edges_path.read_text())
        bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
        xlabel = "time from start_time (ms)" if "stim" in tag else "time from first lick (ms, corrected)"

        print(f"{tag}:")
        out_dir = fig_dir(tag)
        plot_whole(df, bin_labels_ms, f"{title_prefix} -- whole session", out_dir / f"025_{tag}_whole.png", xlabel)
        plot_condition_comparison(df, bin_labels_ms, "half", ("first", "second"),
                                   f"{title_prefix} -- session-half comparison", out_dir / f"025_{tag}_half.png", xlabel)
        plot_condition_comparison(df, bin_labels_ms, "perfstate", ("high", "low"),
                                   f"{title_prefix} -- performance-state comparison", out_dir / f"025_{tag}_perfstate.png", xlabel)


if __name__ == "__main__":
    main()
