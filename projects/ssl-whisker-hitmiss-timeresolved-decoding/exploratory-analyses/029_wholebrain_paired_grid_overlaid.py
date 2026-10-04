"""Cohort-overlaid variant of `027_wholebrain_window_stats_figures.py`'s
paired grid (user request 2026-09-12: "plot these grid figures with cohorts
overlaid, without the single mouse lines"). Instead of 3 separate columns
(R+, R-, aggregated) each with per-session spaghetti lines, this puts R+,
R-, and aggregated as three overlaid mean+-SEM lines in ONE square panel per
comparison -- no individual-session lines, just the group summary.

Same 4 comparisons as `027`'s paired grid (2x2 layout): half/perfstate x
condition_pair/crossgen_vs_within.

Above-chance ALTERNATE figures (user request 2026-09-17, pointing at
`029_hitmiss_stim_whole_brain_sensory_paired_grid_overlaid.png`: "Make
alternate figs with this" [decoding above chance]): a second PNG per tag,
`..._abovechance.png`, plotting `window_acc - null_window_acc` instead of
raw `window_acc` -- each row's OWN `shift_null_curves` (hitmiss/perfstate;
one per condition_value, e.g. half:first and half:second each carry their
own null, confirmed by checking the data directly, not assumed) when
available, falling back to theoretical chance (0.5) for tags without it
(modality -- same fallback convention `034`/`042` already use). The
`crossgen` value has no null of its own (it's a deterministic accuracy
from cross-condition generalization, not a separately-nulled decode) --
its above-chance value uses the SAME row's `within`-condition null as a
shared baseline, since that's the only null available for that session at
all. Original raw-accuracy figures are untouched -- this adds a second
figure, per "alternate," it doesn't replace the first.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ssl_timeresolved_decoding import mean_in_window  # noqa: E402

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


SENSORY_WINDOW = (0.005, 0.050)  # updated 2026-09-18 from (0.005, 0.035), user request
PRE_LICK_WINDOW = (-0.100, 0.0)
CHANCE = 0.5
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8", "aggregated": "#2c5f5b"}

STAGE_PAIRS = [
    # (learning_tag, expert_tag, window_name, window, title_prefix)
    # 2026-09-18, user request "Do the same with paired_grid plots,
    # across-learning-stage with two cohorts overlaid" -- learning-stage
    # and expert-stage sessions compared as two UNPAIRED groups (updated
    # same day, "For the across learning stage, compared sessions, do not
    # pair mice"): no subject_id matching, a mouse need not have sessions
    # in both stages to be included; see `render_stage_pair_figure`.
    ("hitmiss_stim_whole_brain", "hitmiss_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Hit vs miss, start_time-aligned"),
    ("modality_stim_whole_brain", "modality_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Modality, start_time-aligned"),
    ("modality_lick_whole_brain", "modality_lick_expert_whole_brain", "pre_lick", PRE_LICK_WINDOW, "Modality, first-lick-aligned (corrected)"),
]

COMBOS = [
    ("hitmiss_stim_whole_brain", "sensory", SENSORY_WINDOW, "Hit vs miss, start_time-aligned"),
    ("modality_stim_whole_brain", "sensory", SENSORY_WINDOW, "Modality, start_time-aligned"),
    ("modality_lick_whole_brain", "pre_lick", PRE_LICK_WINDOW, "Modality, first-lick-aligned (corrected)"),
    # Expert-stage (whole_brain) variants (user request 2026-09-14: "extend
    # that fuller treatment to all"). perfstate excluded throughout -- it
    # only ever computes 'whole', no half/perfstate sub-conditions to
    # compare against itself. area_group NOT added here: its results have
    # ~30 rows/session (one per area x condition, via `area_col`/
    # `area_value`), but this script's per-session pivots assume one row
    # per session (true for whole_brain's single pseudo-area) -- adding
    # area_group tags as-is would silently pool different areas together.
    # Needs per-area faceting (like 032/034 already do) before extending
    # here; flagged to the user 2026-09-14, not yet resolved.
    ("hitmiss_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Hit vs miss, start_time-aligned (expert)"),
    ("modality_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Modality, start_time-aligned (expert)"),
    ("modality_lick_expert_whole_brain", "pre_lick", PRE_LICK_WINDOW, "Modality, first-lick-aligned (corrected) (expert)"),
]


def load_combo(tag: str) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Returns None when results/bin-edges aren't ready yet, so `main`'s
    per-tag loop can skip and continue to later tags instead of dying on
    the first not-yet-ready one (2026-09-14 fix, needed now that `COMBOS`
    spans multiple sweeps that finish at different times)."""
    partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
    edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
    if not partial_path.exists() or not edges_path.exists():
        return None
    df = pd.read_parquet(partial_path)
    df = df[df["skipped_reason"].isna()].copy()
    if len(df) == 0:  # file exists but every row skipped (e.g. class-imbalance at expert stage) -- nothing to plot
        return None
    edges = json.loads(edges_path.read_text())
    bin_labels = np.array([e[1] for e in edges])
    return df, bin_labels


def curve_window_mean(curve, bin_labels: np.ndarray, window: tuple[float, float]) -> float:
    if curve is None:
        return float("nan")
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        return float("nan")
    return mean_in_window(arr, bin_labels, window)


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    """`fig.savefig` occasionally hits a transient Windows file-lock
    (`OSError: [Errno 22] Invalid argument`) that a simple retry always
    clears -- same helper as `030`/`032`/`034`/`043`/`046`, added here
    2026-09-17 after it crashed this script outright with no retry."""
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


def shift_null_window_mean(shift_curves, bin_labels: np.ndarray, window: tuple[float, float]) -> float:
    """Mean-across-draws-then-window-mean of a row's `shift_null_curves` --
    same convention as `034_area_window_quant_grid.py`'s helper of the same
    name (one empirical per-session chance-level estimate for this window)."""
    if shift_curves is None:
        return float("nan")
    per_shuf = [np.asarray(s, dtype=float) for s in shift_curves if s is not None]
    if not per_shuf:
        return float("nan")
    mean_curve = np.nanmean(np.stack(per_shuf), axis=0)
    return mean_in_window(mean_curve, bin_labels, window)


def overlaid_panel(ax, group_arrays: dict[str, tuple[np.ndarray, np.ndarray]], label_a: str, label_b: str, title: str,
                    ylim: tuple[float, float], ref_line: float = CHANCE, paired: bool = True):
    """group_arrays: {group_label: (a_values, b_values)}. `paired=True`
    (default, every existing caller): `a`/`b` are the SAME sessions'/mice's
    two values (same length, index-corresponding) -- Wilcoxon signed-rank
    + paired-t. `paired=False` (2026-09-18, user request "For the across
    learning stage, compared sessions, do not pair mice" --
    `render_stage_pair_figure`'s stage comparison specifically): `a`/`b`
    are independent samples (different sessions/mice in each, not
    required to be the same length or have any correspondence) -- Mann-
    Whitney U + Welch's t instead, and NaNs are dropped independently per
    array rather than requiring both entries of a pair to be valid. The
    connecting line between the two group means is still drawn either way
    -- it tracks the group-level average, which is informative even
    without individual-level pairing."""
    xs = [0, 1]
    p_lines = []
    for group_label, (a, b) in group_arrays.items():
        if paired:
            valid = ~(np.isnan(a) | np.isnan(b))
            a, b = a[valid], b[valid]
        else:
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        n = len(a) if paired else min(len(a), len(b))
        color = COHORT_COLOR[group_label]
        if len(a) < 2 or len(b) < 2:
            p_lines.append(f"{group_label}: n_a={len(a)}, n_b={len(b)}")
            continue
        mean_a, mean_b = np.mean(a), np.mean(b)
        sem_a, sem_b = np.std(a) / np.sqrt(len(a)), np.std(b) / np.sqrt(len(b))
        ax.errorbar(xs, [mean_a, mean_b], yerr=[sem_a, sem_b], color=color, lw=2.5, marker="o",
                    markersize=7, zorder=3, capsize=3, label=group_label)
        if paired:
            try:
                w_p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else np.nan
            except ValueError:
                w_p = np.nan
            t_p = stats.ttest_rel(a, b).pvalue
            p_lines.append(f"{group_label}: n={n}, W p={w_p:.2g}, t p={t_p:.2g}")
        else:
            mw_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            welch_p = stats.ttest_ind(a, b, equal_var=False).pvalue
            p_lines.append(f"{group_label}: n_a={len(a)}, n_b={len(b)}, MW p={mw_p:.2g}, Welch p={welch_p:.2g}")

    # Between-cohort (R+ vs R-, unpaired) test at each condition level -- added per user request 2026-09-12.
    if "R+" in group_arrays and "R-" in group_arrays:
        a_rplus, b_rplus = group_arrays["R+"]
        a_rminus, b_rminus = group_arrays["R-"]
        for level_label, x, vals_rplus, vals_rminus in ((label_a, xs[0], a_rplus, a_rminus), (label_b, xs[1], b_rplus, b_rminus)):
            vp, vm = vals_rplus[~np.isnan(vals_rplus)], vals_rminus[~np.isnan(vals_rminus)]
            if len(vp) >= 2 and len(vm) >= 2:
                mw_p = stats.mannwhitneyu(vp, vm, alternative="two-sided").pvalue
                welch_p = stats.ttest_ind(vp, vm, equal_var=False).pvalue
                p_lines.append(f"R+ vs R- @{level_label}: MW p={mw_p:.2g}, Welch p={welch_p:.2g}")

    ax.text(0.5, 0.02, "\n".join(p_lines), transform=ax.transAxes, fontsize=7, ha="center", va="bottom", linespacing=1.5)
    if ylim[0] <= ref_line <= ylim[1]:
        ax.axhline(ref_line, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_a, label_b], fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10.5)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    return len(p_lines)


SPECS = [
    ("half: first vs second", "half", ("first", "second"), False),
    ("perfstate: high vs low", "perfstate", ("high", "low"), False),
    ("half: within vs crossgen", "half", ("first", "second"), True),
    ("perfstate: within vs crossgen", "perfstate", ("high", "low"), True),
]


def render_figure(df: pd.DataFrame, value_col: str, crossgen_col: str, tag: str, window_name: str,
                   window: tuple[float, float], title_prefix: str, ylabel: str, ref_line: float,
                   value_clip: tuple[float, float] | None, out_suffix: str):
    """Builds and saves one 2x2 overlaid-panel figure from `value_col`
    (paired against itself for half/perfstate, against `crossgen_col` for
    the crossgen rows). Factored out of `main` (2026-09-17) so the raw and
    above-chance figures -- same panel logic, different input columns --
    don't duplicate ~40 lines each."""
    # First pass: build all panels' group_arrays and find the shared y-range from actual data (mean +- SEM span).
    panel_data = []
    all_bounds = []
    for row_label, condition_type, values, use_crossgen in SPECS:
        sub = df[df.condition_type == condition_type]
        group_arrays = {}
        label_a = label_b = None
        for group_label, group_sub in (("R+", sub[sub.reward_group == "R+"]), ("R-", sub[sub.reward_group == "R-"])):
            if use_crossgen:
                a = group_sub[value_col].to_numpy()
                b = group_sub[crossgen_col].to_numpy()
                label_a, label_b = "within", "crossgen"
            else:
                piv = group_sub.pivot_table(index="session_id", columns="condition_value", values=value_col)
                if values[0] not in piv.columns or values[1] not in piv.columns:
                    continue
                a, b = piv[values[0]].to_numpy(), piv[values[1]].to_numpy()
                label_a, label_b = values
            group_arrays[group_label] = (a, b)
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.any():
                n = valid.sum()
                mean_a, mean_b = np.mean(a[valid]), np.mean(b[valid])
                sem_a, sem_b = np.std(a[valid]) / np.sqrt(n), np.std(b[valid]) / np.sqrt(n)
                all_bounds.extend([mean_a - sem_a, mean_a + sem_a, mean_b - sem_b, mean_b + sem_b])
        panel_data.append((row_label, group_arrays, label_a, label_b))

    pad = 0.03
    data_lo = min(all_bounds) if all_bounds else (0.4 if value_clip else -0.05)
    data_hi = max(all_bounds) + pad if all_bounds else (1.0 if value_clip else 0.15)
    text_room = 0.55 * (data_hi - data_lo)  # room for up to 4 annotation lines below the data, shared across all 4 panels
    y_lo = data_lo - text_room
    y_hi = data_hi
    if value_clip is not None:
        y_lo, y_hi = max(value_clip[0], y_lo), min(value_clip[1], y_hi)
    ylim = (y_lo, y_hi)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    for ax, (row_label, group_arrays, label_a, label_b) in zip(axes.flat, panel_data):
        overlaid_panel(ax, group_arrays, label_a, label_b, row_label, ylim, ref_line=ref_line)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("condition", fontsize=9)

    fig.suptitle(f"{title_prefix} -- {window_name} window {window}, cohorts overlaid (whole brain)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = fig_dir(tag) / f"029_{tag}_{window_name}_paired_grid_overlaid{out_suffix}.png"
    savefig_retry(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


def per_subject_whole_values(tag: str, window: tuple[float, float]) -> pd.DataFrame | None:
    """One row per (subject_id, reward_group) for `tag`'s 'whole' condition
    -- `window_acc` (raw) and `null_window_acc` (that mouse's own sessions'
    shift-null, mean-across-sessions-then-mean-across-draws; NaN/absent
    when the tag has no shift-null, e.g. modality_stim) -- averaged across
    a mouse's own sessions first (a mouse can have several sessions within
    one stage, same as `047`'s per-mouse aggregation). NOT used by
    `render_stage_pair_figure` any more (2026-09-18, see
    `per_session_whole_values`) -- kept in case a per-mouse-aggregated
    version is wanted again later."""
    loaded = load_combo(tag)
    if loaded is None:
        return None
    df, bin_labels = loaded
    sub = df[df.condition_type == "whole"].copy()
    if len(sub) == 0:
        return None
    sub["window_acc"] = sub["real_curve"].apply(lambda c: curve_window_mean(c, bin_labels, window))
    if "shift_null_curves" in sub.columns and sub["shift_null_curves"].notna().any():
        sub["null_window_acc"] = sub["shift_null_curves"].apply(lambda c: shift_null_window_mean(c, bin_labels, window))
    else:
        sub["null_window_acc"] = np.nan
    return sub.groupby(["subject_id", "reward_group"], as_index=False)[["window_acc", "null_window_acc"]].mean()


def per_session_whole_values(tag: str, window: tuple[float, float]) -> pd.DataFrame | None:
    """One row per session for `tag`'s 'whole' condition -- `window_acc`
    (raw) and `null_window_acc` (that session's own shift-null; NaN/absent
    when the tag has no shift-null). Unlike `per_subject_whole_values`, NOT
    aggregated to one value per mouse -- `render_stage_pair_figure` compares
    sessions directly, unpaired, per user request 2026-09-18 ("For the
    across learning stage, compared sessions, do not pair mice")."""
    loaded = load_combo(tag)
    if loaded is None:
        return None
    df, bin_labels = loaded
    sub = df[df.condition_type == "whole"].copy()
    if len(sub) == 0:
        return None
    sub["window_acc"] = sub["real_curve"].apply(lambda c: curve_window_mean(c, bin_labels, window))
    if "shift_null_curves" in sub.columns and sub["shift_null_curves"].notna().any():
        sub["null_window_acc"] = sub["shift_null_curves"].apply(lambda c: shift_null_window_mean(c, bin_labels, window))
    else:
        sub["null_window_acc"] = np.nan
    return sub[["session_id", "subject_id", "reward_group", "window_acc", "null_window_acc"]]


def render_stage_pair_figure(tag_learning: str, tag_expert: str, window_name: str, window: tuple[float, float],
                              title_prefix: str, value_col: str, ylabel: str, ref_line: float,
                              value_clip: tuple[float, float] | None, out_suffix: str):
    """Across-learning-stage analog of `render_figure`'s SPECS panels
    (see `STAGE_PAIRS`), one square panel instead of a 2x2 grid since
    there's a single comparison (learning vs expert) here, not four.
    UNPAIRED (2026-09-18, user request "For the across learning stage,
    compared sessions, do not pair mice"): learning-stage and expert-stage
    sessions are compared as two independent groups -- no subject_id
    matching, no requirement that the same mouse appear in both (a mouse
    contributing sessions to only one stage still counts), Mann-Whitney U
    + Welch's t (`overlaid_panel(..., paired=False)`) instead of Wilcoxon
    + paired-t."""
    learn = per_session_whole_values(tag_learning, window)
    expert = per_session_whole_values(tag_expert, window)
    if learn is None or expert is None:
        print(f"{tag_learning} / {tag_expert}: missing results/bin-edges for one stage, skipping stage-pair")
        return

    def values(df: pd.DataFrame) -> pd.Series:
        return df["window_acc"] if value_col == "window_acc" else df["window_acc"] - df["null_window_acc"]

    group_arrays = {}
    all_bounds = []
    for group_label in ("R+", "R-"):
        a = values(learn[learn.reward_group == group_label]).to_numpy()
        b = values(expert[expert.reward_group == group_label]).to_numpy()
        group_arrays[group_label] = (a, b)
        a_valid, b_valid = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a_valid) and len(b_valid):
            mean_a, mean_b = np.mean(a_valid), np.mean(b_valid)
            sem_a, sem_b = np.std(a_valid) / np.sqrt(len(a_valid)), np.std(b_valid) / np.sqrt(len(b_valid))
            all_bounds.extend([mean_a - sem_a, mean_a + sem_a, mean_b - sem_b, mean_b + sem_b])

    pad = 0.03
    data_lo = min(all_bounds) if all_bounds else (0.4 if value_clip else -0.05)
    data_hi = max(all_bounds) + pad if all_bounds else (1.0 if value_clip else 0.15)
    text_room = 0.6 * (data_hi - data_lo)
    y_lo, y_hi = data_lo - text_room, data_hi
    if value_clip is not None:
        y_lo, y_hi = max(value_clip[0], y_lo), min(value_clip[1], y_hi)
    ylim = (y_lo, y_hi)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    overlaid_panel(ax, group_arrays, "learning", "expert", "learning vs expert (unpaired sessions)", ylim,
                   ref_line=ref_line, paired=False)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("day stage", fontsize=9)
    fig.suptitle(f"{title_prefix} -- {window_name} window {window}, cohorts overlaid (whole brain)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = fig_dir(tag_learning) / f"029_{tag_learning}_{window_name}_paired_grid_overlaid_stage{out_suffix}.png"
    savefig_retry(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


def main():
    for tag, window_name, window, title_prefix in COMBOS:
        loaded = load_combo(tag)
        if loaded is None:
            print(f"{tag}: missing results/bin-edges, skipping")
            continue
        df, bin_labels = loaded
        df["window_acc"] = df["real_curve"].apply(lambda c: curve_window_mean(c, bin_labels, window))
        df["crossgen_window_acc"] = df["crossgen_curve_to_other"].apply(lambda c: curve_window_mean(c, bin_labels, window))
        render_figure(df, "window_acc", "crossgen_window_acc", tag, window_name, window, title_prefix,
                      ylabel="balanced accuracy", ref_line=CHANCE, value_clip=(0.0, 1.0), out_suffix="")

        # Above-chance alternate (user request 2026-09-17): each row's own
        # shift-null (hitmiss/perfstate) or theoretical 0.5 (modality,
        # no shift-null computed) as the "chance" being subtracted --
        # `crossgen` reuses the SAME row's within-condition null, since it
        # has no null of its own.
        has_shift_null = "shift_null_curves" in df.columns and df["shift_null_curves"].notna().any()
        if has_shift_null:
            df["null_window_acc"] = df["shift_null_curves"].apply(lambda c: shift_null_window_mean(c, bin_labels, window))
            ylabel_ac = "balanced accuracy - shift-null"
        else:
            df["null_window_acc"] = CHANCE
            ylabel_ac = "balanced accuracy - chance"
        df["window_acc_abovechance"] = df["window_acc"] - df["null_window_acc"]
        df["crossgen_window_acc_abovechance"] = df["crossgen_window_acc"] - df["null_window_acc"]
        render_figure(df, "window_acc_abovechance", "crossgen_window_acc_abovechance", tag, window_name, window,
                      title_prefix, ylabel=ylabel_ac, ref_line=0.0, value_clip=None, out_suffix="_abovechance")

    # Across-learning-stage, cohorts overlaid (2026-09-18, user request
    # "Do the same with paired_grid plots, across-learning-stage with two
    # cohorts overlaid") -- UNPAIRED sessions (2026-09-18 follow-up:
    # "do not pair mice"), Mann-Whitney U + Welch's t.
    for tag_learning, tag_expert, window_name, window, title_prefix in STAGE_PAIRS:
        render_stage_pair_figure(tag_learning, tag_expert, window_name, window, title_prefix,
                                  "window_acc", "balanced accuracy", CHANCE, (0.0, 1.0), "")
        render_stage_pair_figure(tag_learning, tag_expert, window_name, window, title_prefix,
                                  "window_acc_abovechance", "balanced accuracy - shift-null/chance", 0.0, None, "_abovechance")


if __name__ == "__main__":
    main()
