"""Figures for `026_wholebrain_window_stats.py`'s tests (user request
2026-09-12: "make figures for the stats, with small square subplots per
comparison"). Two figures per combo:

  1. `..._paired_grid.png`: every PAIRED comparison (condition_pair: half
     first-vs-second, perfstate high-vs-low; crossgen_vs_within: within-vs-
     cross for each condition type), x R+/R-/aggregated -- one small square
     panel per comparison, each a per-session dot-and-line plot (session's
     own line connecting its two values) plus the group mean+-SEM, with the
     Wilcoxon/paired-t p-values annotated.
  2. `..._cohort_grid.png`: every COHORT (R+ vs R-, unpaired) comparison,
     one small square panel per condition_value, each a two-group jittered
     strip plot with mean+-SEM and the Mann-Whitney/Welch p-values annotated.

Chance level (0.5) always drawn. Reuses the same window definitions as
026 (sensory: 5-35ms post-stim; pre-lick: 150ms before lick).

Above-chance ALTERNATE figures (user request 2026-09-17, after the same
addition to `029_wholebrain_paired_grid_overlaid.py`: "Yes I do" [redo
027 the same way]): a second pair of PNGs per tag,
`..._paired_grid_abovechance.png`/`..._cohort_grid_abovechance.png`,
plotting `window_acc - null_window_acc` instead of raw `window_acc` --
each row's own `shift_null_curves` (hitmiss/perfstate) when available,
falling back to theoretical chance (0.5) for tags without it (modality).
`crossgen` has no null of its own -- reuses the same row's within-
condition null, same convention as `029`. Original raw-accuracy figures
are untouched.
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
    # area_group tags as-is would silently pool different areas together
    # in the paired/cohort tests. Needs per-area faceting (like 032/034
    # already do) before extending here; flagged to the user 2026-09-14,
    # not yet resolved.
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


def shift_null_window_mean(shift_curves, bin_labels: np.ndarray, window: tuple[float, float]) -> float:
    """Mean-across-draws-then-window-mean of a row's `shift_null_curves` --
    same convention as `034`/`029`'s helper of the same name."""
    if shift_curves is None:
        return float("nan")
    per_shuf = [np.asarray(s, dtype=float) for s in shift_curves if s is not None]
    if not per_shuf:
        return float("nan")
    mean_curve = np.nanmean(np.stack(per_shuf), axis=0)
    return mean_in_window(mean_curve, bin_labels, window)


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    """`fig.savefig` occasionally hits a transient Windows file-lock
    (`OSError: [Errno 22] Invalid argument`) that a simple retry always
    clears -- same helper as `029`/`030`/`032`/`034`/`043`/`046`, added
    here 2026-09-17 (this script had no retry logic at all before)."""
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


def paired_panel(ax, a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, color: str, title: str,
                  ref_line: float = CHANCE, ylim: tuple[float, float] = (0.35, 1.02)):
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    xs = [0, 1]
    for ai, bi in zip(a, b):
        ax.plot(xs, [ai, bi], color=color, alpha=0.15, lw=0.8, zorder=1)
    if n >= 2:
        try:
            w_p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else np.nan
        except ValueError:
            w_p = np.nan
        t_p = stats.ttest_rel(a, b).pvalue
        mean_a, mean_b = np.mean(a), np.mean(b)
        sem_a, sem_b = np.std(a) / np.sqrt(n), np.std(b) / np.sqrt(n)
        ax.errorbar(xs, [mean_a, mean_b], yerr=[sem_a, sem_b], color=color, lw=2.5, marker="o",
                    markersize=6, zorder=3, capsize=3)
        p_text = f"n={n}\nWilcoxon p={w_p:.3g}\npaired-t p={t_p:.3g}"
    else:
        p_text = f"n={n}"
    ax.text(0.5, 0.02, p_text, transform=ax.transAxes, fontsize=7.5, ha="center", va="bottom")
    ax.axhline(ref_line, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_a, label_b], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def cohort_panel(ax, rplus: np.ndarray, rminus: np.ndarray, title: str,
                  ref_line: float = CHANCE, ylim: tuple[float, float] = (0.35, 1.02)):
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate([(rplus, COHORT_COLOR["R+"]), (rminus, COHORT_COLOR["R-"])]):
        vals = vals[~np.isnan(vals)]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, alpha=0.4, s=10, zorder=2)
        if len(vals):
            mean_v, sem_v = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
            ax.errorbar([i], [mean_v], yerr=[sem_v], color=color, lw=2.5, marker="o", markersize=7, zorder=3, capsize=3)
    n_p, n_m = np.sum(~np.isnan(rplus)), np.sum(~np.isnan(rminus))
    if n_p >= 2 and n_m >= 2:
        mw_p = stats.mannwhitneyu(rplus[~np.isnan(rplus)], rminus[~np.isnan(rminus)], alternative="two-sided").pvalue
        welch_p = stats.ttest_ind(rplus[~np.isnan(rplus)], rminus[~np.isnan(rminus)], equal_var=False).pvalue
        p_text = f"R+ n={n_p}, R- n={n_m}\nMann-Whitney p={mw_p:.3g}\nWelch p={welch_p:.3g}"
    else:
        p_text = f"R+ n={n_p}, R- n={n_m}"
    ax.text(0.5, 0.02, p_text, transform=ax.transAxes, fontsize=7.5, ha="center", va="bottom")
    ax.axhline(ref_line, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R+", "R-"], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def render_paired_grid(df: pd.DataFrame, value_col: str, crossgen_col: str, tag: str, window_name: str,
                        window: tuple[float, float], title_prefix: str, ylabel: str, ref_line: float,
                        ylim: tuple[float, float], out_suffix: str):
    paired_specs = []  # (row_label, condition_type, values, use_crossgen)
    for condition_type, values in (("half", ("first", "second")), ("perfstate", ("high", "low"))):
        paired_specs.append((f"{condition_type}: {values[0]} vs {values[1]}", condition_type, values, False))
    for condition_type, values in (("half", ("first", "second")), ("perfstate", ("high", "low"))):
        paired_specs.append((f"{condition_type}: within vs crossgen", condition_type, values, True))

    n_rows, n_cols = len(paired_specs), 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    for r, (row_label, condition_type, values, use_crossgen) in enumerate(paired_specs):
        sub = df[df.condition_type == condition_type]
        for c, (group_label, group_sub) in enumerate([
            ("R+", sub[sub.reward_group == "R+"]), ("R-", sub[sub.reward_group == "R-"]), ("aggregated", sub)
        ]):
            ax = axes[r, c]
            if use_crossgen:
                a = group_sub[value_col].to_numpy()
                b = group_sub[crossgen_col].to_numpy()
                label_a, label_b = "within", "crossgen"
            else:
                piv = group_sub.pivot_table(index="session_id", columns="condition_value", values=value_col)
                if values[0] not in piv.columns or values[1] not in piv.columns:
                    ax.axis("off")
                    continue
                a, b = piv[values[0]].to_numpy(), piv[values[1]].to_numpy()
                label_a, label_b = values
            paired_panel(ax, a, b, label_a, label_b, COHORT_COLOR[group_label], f"{row_label}\n{group_label}",
                         ref_line=ref_line, ylim=ylim)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
    fig.suptitle(f"{title_prefix} -- {window_name} window {window}, paired comparisons (whole brain)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out1 = fig_dir(tag) / f"027_{tag}_{window_name}_paired_grid{out_suffix}.png"
    savefig_retry(fig, out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out1.name}")


def render_cohort_grid(df: pd.DataFrame, value_col: str, tag: str, window_name: str, window: tuple[float, float],
                        title_prefix: str, ylabel: str, ref_line: float, ylim: tuple[float, float], out_suffix: str):
    cohort_specs = []
    for condition_type in ("whole", "half", "perfstate"):
        for cv in sorted(df[df.condition_type == condition_type]["condition_value"].dropna().unique()):
            cohort_specs.append((condition_type, cv))
    n_panels = len(cohort_specs)
    n_cols2 = 3
    n_rows2 = int(np.ceil(n_panels / n_cols2))
    fig2, axes2 = plt.subplots(n_rows2, n_cols2, figsize=(3.0 * n_cols2, 3.0 * n_rows2), squeeze=False)
    for i, (condition_type, cv) in enumerate(cohort_specs):
        ax = axes2[i // n_cols2, i % n_cols2]
        sub = df[(df.condition_type == condition_type) & (df.condition_value == cv)]
        rplus = sub[sub.reward_group == "R+"][value_col].to_numpy()
        rminus = sub[sub.reward_group == "R-"][value_col].to_numpy()
        cohort_panel(ax, rplus, rminus, f"{condition_type}={cv}", ref_line=ref_line, ylim=ylim)
        if i % n_cols2 == 0:
            ax.set_ylabel(ylabel, fontsize=9)
    for j in range(n_panels, n_rows2 * n_cols2):
        axes2[j // n_cols2, j % n_cols2].axis("off")
    fig2.suptitle(f"{title_prefix} -- {window_name} window {window}, R+ vs R- comparisons (whole brain)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = fig_dir(tag) / f"027_{tag}_{window_name}_cohort_grid{out_suffix}.png"
    savefig_retry(fig2, out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2.name}")


def abovechance_ylim(df: pd.DataFrame, value_col: str) -> tuple[float, float]:
    """Shared y-range for the above-chance figures, from the actual data
    span (values can be negative, unlike raw accuracy's fixed 0.35-1.02)."""
    vals = df[value_col].dropna().to_numpy()
    if len(vals) == 0:
        return (-0.10, 0.30)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    return (min(-0.02, lo - pad), max(0.05, hi + pad))


def main():
    for tag, window_name, window, title_prefix in COMBOS:
        loaded = load_combo(tag)
        if loaded is None:
            print(f"{tag}: missing results/bin-edges, skipping")
            continue
        df, bin_labels = loaded
        df["window_acc"] = df["real_curve"].apply(lambda c: curve_window_mean(c, bin_labels, window))
        df["crossgen_window_acc"] = df["crossgen_curve_to_other"].apply(lambda c: curve_window_mean(c, bin_labels, window))

        render_paired_grid(df, "window_acc", "crossgen_window_acc", tag, window_name, window, title_prefix,
                            ylabel="balanced accuracy", ref_line=CHANCE, ylim=(0.35, 1.02), out_suffix="")
        render_cohort_grid(df, "window_acc", tag, window_name, window, title_prefix,
                            ylabel="balanced accuracy", ref_line=CHANCE, ylim=(0.35, 1.02), out_suffix="")

        # Above-chance alternates (user request 2026-09-17, "Yes I do" --
        # same treatment as 029): each row's own shift-null (hitmiss/
        # perfstate) or theoretical 0.5 (modality) as the "chance" being
        # subtracted; crossgen reuses the within-condition's own null.
        has_shift_null = "shift_null_curves" in df.columns and df["shift_null_curves"].notna().any()
        if has_shift_null:
            df["null_window_acc"] = df["shift_null_curves"].apply(lambda c: shift_null_window_mean(c, bin_labels, window))
            ylabel_ac = "balanced accuracy - shift-null"
        else:
            df["null_window_acc"] = CHANCE
            ylabel_ac = "balanced accuracy - chance"
        df["window_acc_abovechance"] = df["window_acc"] - df["null_window_acc"]
        df["crossgen_window_acc_abovechance"] = df["crossgen_window_acc"] - df["null_window_acc"]
        ylim_ac = abovechance_ylim(df, "window_acc_abovechance")

        render_paired_grid(df, "window_acc_abovechance", "crossgen_window_acc_abovechance", tag, window_name, window,
                            title_prefix, ylabel=ylabel_ac, ref_line=0.0, ylim=ylim_ac, out_suffix="_abovechance")
        render_cohort_grid(df, "window_acc_abovechance", tag, window_name, window, title_prefix,
                            ylabel=ylabel_ac, ref_line=0.0, ylim=ylim_ac, out_suffix="_abovechance")


if __name__ == "__main__":
    main()
