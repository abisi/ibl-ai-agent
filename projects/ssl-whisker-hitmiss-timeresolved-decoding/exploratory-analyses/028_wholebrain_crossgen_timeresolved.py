"""Time-resolved cross-generalization figures (user request 2026-09-12:
"similar time-resolved figures for the cross-condition evaluation").
Extends the `025_plot_wholebrain_progress.py` half/perfstate template
(3-panel R+/R-/aggregated, square, causal bins labeled at bin end) by
adding the cross-generalization curve alongside the two within-condition
curves: solid = within level A, dashed = within level B, dotted = crossgen
(mean of both directions' `crossgen_curve_to_other`, since a session's A-row
carries "trained on A, tested on B" and its B-row carries "trained on B,
tested on A" -- averaging them per bin gives one symmetric "generalization
accuracy" curve, matching how `026`/`027`'s within-vs-crossgen test already
pools both directions).

RT / stim-time histogram overlay (user request 2026-09-15: "all
stim-aligned time-resolved figures must show the little histogram at the
bottom", then "For the lick time time-resolved figures, plot the stim-time,
the same way RTs are plotted") -- same `build_rt_cache`/
`build_stim_time_cache`/`add_rt_histogram` implementation as
`032_plot_decode_results.py` (duplicated here rather than imported, per
this project's existing convention of each numbered script being
self-contained -- see e.g. `EXCLUDED_OVERLAY_AREAS`/`get_area_color_map`
duplicated across `032`/`034`/`042`): stim-aligned combos
(`hitmiss_stim_whole_brain`, `modality_stim_whole_brain`, and the
expert-stage variants) get the whisker-trial hit RT histogram; lick-aligned
combos (`modality_lick_whole_brain`, `modality_lick_expert_whole_brain`)
get the mirror-image stim-onset-time-relative-to-lick histogram instead
(`-reaction_time`, corrected first lick since 2026-09-27; was `start_time - lick_time`) -- every combo gets one or the other,
never neither. Needs to run from the repo root (`prep_hitmiss_trials` ->
`load_reward_group` resolves a repo-root-relative path), same requirement
`032` already has.
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
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import add_first_lick_time, prep_hitmiss_trials  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def build_rt_cache(session_ids) -> dict[str, np.ndarray]:
    """Whisker-trial hit reaction times (ms) per session -- identical to
    `032_plot_decode_results.py`'s helper of the same name."""
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    cache: dict[str, np.ndarray] = {}
    for session_id in session_ids:
        trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        if trials is None:
            continue
        hits = add_first_lick_time(trials[trials["lick_flag"] == 1])  # corrected first lick (2026-09-27)
        cache[session_id] = hits["reaction_time"].to_numpy() * 1000
    return cache


def build_stim_time_cache(session_ids) -> dict[str, np.ndarray]:
    """Whisker-trial hit stimulus-onset times (ms) relative to the corrected first lick (-reaction_time; stored lick_time before 2026-09-27),
    per session -- the lick-aligned mirror of `build_rt_cache`, identical to
    `032_plot_decode_results.py`'s helper of the same name."""
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    cache: dict[str, np.ndarray] = {}
    for session_id in session_ids:
        trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        if trials is None:
            continue
        hits = add_first_lick_time(trials[trials["lick_flag"] == 1])  # corrected first lick (2026-09-27)
        cache[session_id] = (-hits["reaction_time"]).to_numpy() * 1000
    return cache


def add_rt_histogram(ax, session_ids, rt_cache: dict[str, np.ndarray], xlim: tuple[float, float], ylim: tuple[float, float]):
    if rt_cache is None:
        return
    arrays = [rt_cache[s] for s in session_ids if s in rt_cache and len(rt_cache[s])]
    if not arrays:
        return
    rt = np.concatenate(arrays)
    rt = rt[(rt >= xlim[0]) & (rt <= xlim[1])]
    if len(rt) == 0:
        return
    frac_at_chance = (0.5 - ylim[0]) / (ylim[1] - ylim[0])
    if frac_at_chance <= 0:
        return
    # Drawn directly on `ax` in its own data coords (bars scaled to sit
    # below the chance line) rather than via `ax.twinx()` -- a second axes
    # sharing the same position as `ax` in a multi-column figure confuses
    # matplotlib's `bbox_inches="tight"` bbox union and can silently clip
    # the leftmost subplot's own ylabel/yticklabels (root-caused 2026-09-16
    # in `032_plot_decode_results.py`'s identical helper, same fix here).
    counts, edges = np.histogram(rt, bins=40, range=xlim)
    if counts.max() <= 0:
        return
    top = ylim[0] + frac_at_chance * 0.85 * (ylim[1] - ylim[0])
    heights = (counts / counts.max()) * (top - ylim[0])
    ax.bar(edges[:-1], heights, bottom=ylim[0], width=np.diff(edges), align="edge",
           color="#888888", alpha=0.35, edgecolor="#333333", linewidth=1.2, zorder=0)


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

COMBOS = [
    ("hitmiss_stim_whole_brain", "Hit vs miss, start_time-aligned, whole brain", "time from start_time (ms)"),
    ("modality_stim_whole_brain", "Modality, start_time-aligned, whole brain", "time from start_time (ms)"),
    ("modality_lick_whole_brain", "Modality, first-lick-aligned (corrected), whole brain", "time from first lick (ms, corrected)"),
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
    ("hitmiss_stim_expert_whole_brain", "Hit vs miss, start_time-aligned, whole brain (expert)", "time from start_time (ms)"),
    ("modality_stim_expert_whole_brain", "Modality, start_time-aligned, whole brain (expert)", "time from start_time (ms)"),
    ("modality_lick_expert_whole_brain", "Modality, first-lick-aligned (corrected), whole brain (expert)", "time from first lick (ms, corrected)"),
]


def summarize(sub: pd.DataFrame) -> str:
    return f"n={sub['session_id'].nunique()} sessions, {sub['subject_id'].nunique()} mice"


def stack_curves(series: pd.Series) -> np.ndarray | None:
    curves = [np.array(c) for c in series if c is not None]
    return np.stack(curves) if curves else None


def plot_condition_crossgen(df: pd.DataFrame, bin_labels_ms: np.ndarray, condition_type: str,
                             values: tuple[str, str], title: str, out_path: Path, xlabel: str,
                             rt_cache: dict[str, np.ndarray] | None = None):
    sub_ct = df[df["condition_type"] == condition_type]
    if len(sub_ct) == 0:
        print(f"  no '{condition_type}' rows, skipping {out_path.name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [("R+", sub_ct[sub_ct.reward_group == "R+"]), ("R-", sub_ct[sub_ct.reward_group == "R-"]), ("R+ & R- aggregated", sub_ct)]

    any_data = False
    for ax, (label, sub) in zip(axes, panels):
        color = COHORT_COLOR.get(label, AGGREGATE_COLOR)

        # within-condition curves, solid/dashed
        for value, ls in zip(values, ("-", "--")):
            val_sub = sub[sub["condition_value"] == value]
            stacked = stack_curves(val_sub["real_curve"])
            if stacked is None:
                continue
            any_data = True
            mean_curve = np.nanmean(stacked, axis=0)
            sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
            ax.plot(bin_labels_ms, mean_curve, color=color, lw=2.2, linestyle=ls, label=f"within {value}")
            ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.15, lw=0)

        # crossgen curve: average both directions' crossgen_curve_to_other
        crossgen_stacked = stack_curves(sub["crossgen_curve_to_other"])
        if crossgen_stacked is not None:
            any_data = True
            mean_cg = np.nanmean(crossgen_stacked, axis=0)
            sem_cg = np.nanstd(crossgen_stacked, axis=0) / np.sqrt(crossgen_stacked.shape[0])
            ax.plot(bin_labels_ms, mean_cg, color=color, lw=2.0, linestyle=":", label="crossgen", alpha=0.9)
            ax.fill_between(bin_labels_ms, mean_cg - sem_cg, mean_cg + sem_cg, color=color, alpha=0.10, lw=0)

        ax.axhline(0.5, color="#888888", lw=1.2, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_title(f"{label}\n({summarize(sub)})", fontsize=13)
        ax.set_ylim(0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped
        xlim_raw = (float(bin_labels_ms.min()), float(bin_labels_ms.max()))
        xpad = 0.02 * (xlim_raw[1] - xlim_raw[0])
        xlim = (xlim_raw[0] - xpad, xlim_raw[1] + xpad)
        ax.set_xlim(*xlim)
        if rt_cache is not None:
            add_rt_histogram(ax, sub["session_id"].unique(), rt_cache, xlim, (0.4, 1.02))
        ax.set_box_aspect(1)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, frameon=False, loc="upper left")

    if not any_data:
        plt.close(fig)
        print(f"  no computable curves, skipping {out_path.name}")
        return

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main():
    for tag, title_prefix, xlabel in COMBOS:
        partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
        edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
        if not partial_path.exists() or not edges_path.exists():
            print(f"{tag}: missing results/bin-edges, skipping")
            continue

        df = pd.read_parquet(partial_path)
        df = df[df["skipped_reason"].isna()].copy()
        if len(df) == 0:  # file exists but every row skipped (e.g. class-imbalance at expert stage) -- nothing to plot
            print(f"{tag}: results file exists but no computed rows yet")
            continue
        bin_edges = json.loads(edges_path.read_text())
        bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
        alignment = df["alignment"].iloc[0] if "alignment" in df.columns else ("stim" if "_stim_" in tag else "lick")

        # RT / stim-time histogram overlay (user request 2026-09-15) -- see
        # module docstring.
        rt_cache = None
        if alignment == "stim":
            rt_cache = build_rt_cache(df["session_id"].unique())
            print(f"  RT histogram cache: {len(rt_cache)}/{df['session_id'].nunique()} sessions")
        elif alignment == "lick":
            rt_cache = build_stim_time_cache(df["session_id"].unique())
            print(f"  stim-time histogram cache: {len(rt_cache)}/{df['session_id'].nunique()} sessions")

        print(f"{tag}:")
        out_dir = fig_dir(tag)
        plot_condition_crossgen(df, bin_labels_ms, "half", ("first", "second"),
                                 f"{title_prefix} -- session-half within vs cross-generalization",
                                 out_dir / f"028_{tag}_half_crossgen.png", xlabel, rt_cache=rt_cache)
        plot_condition_crossgen(df, bin_labels_ms, "perfstate", ("high", "low"),
                                 f"{title_prefix} -- performance-state within vs cross-generalization",
                                 out_dir / f"028_{tag}_perfstate_crossgen.png", xlabel, rt_cache=rt_cache)


if __name__ == "__main__":
    main()
