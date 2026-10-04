"""Generic, tag-driven progress/results plotting script (added 2026-09-12
to keep results and figures organized consistently across runs, per user
request: "organize cleanly results and figures, as they should remain
consistent across runs"). Supersedes writing a new one-off plotting script
per (decode_target, alignment, area_scheme) combination -- this one just
takes the results tag and figures out what's actually in the data.

Usage: `python 032_plot_decode_results.py <tag>`
  tag matches `024_master_results_<tag>.parquet` / `024_bin_edges_<tag>.json`,
  e.g. "perfstate_stim_whole_brain", "hitmiss_stim_area_group".

Produces, for whichever condition_types are actually present in the data
(gracefully skipping any that aren't -- e.g. `perfstate`-target results have
no 'perfstate' condition_type, only 'whole'):
  - `032_<tag>_whole.png`: whole-session curve.
  - `032_<tag>_half.png`: within-condition halves comparison (not produced
    for the `perfstate` decode target, which is full-session-only -- see
    024_master_sweep.py).
  - `032_<tag>_perfstate.png`: within-condition perfstate comparison (only
    for targets where 'perfstate' is a condition, i.e. hitmiss/modality, not
    the perfstate target itself).

Each figure is 4 panels per area/row: R+, R-, R+ & R- aggregated, and a 4th
"R+ & R- overlaid" panel with both cohorts' curves on one axes (per user
request 2026-09-12: "as fourth subplots, an overlay of cohorts" -- applies
to every time-resolved figure this script makes). Every panel also shows the
label-shuffle null as a mean+-SEM band across sessions -- same treatment as
the real curve, so the two are directly comparable (user request
2026-09-12: "instead of plotting CI for the shuffle, plot sem").

For area schemes with more than one area (area_group, area_acronym_custom),
also facets by area_value (one row of the 4-panel layout per area, capped
at a reasonable number per figure to stay legible).

Additionally, for area schemes with >1 area, produces one
`032_<tag>_<condition_type>_area_overlay.png` per condition_type: within
each cohort (R+, R-) and each condition_value, all areas' real curves are
overlaid on one panel (rows = condition_value, cols = cohort; square
panels), colored by the user's own allen_utils area colors -- per user
request 2026-09-12: "plot within group, within condition figures with area
overlaid for comparison, using colors of my allen_utils function." These
area-overlay figures drop a fixed set of areas not relevant to this
whisker/auditory task (`EXCLUDED_OVERLAY_AREAS`) per user request
2026-09-12: "Ignore somatosensory-body, amygdala, cortical subplate,
insular areas, visual areas, olfactory."

Every figure above (both the per-area faceted ones and the area-overlay
ones) also gets a "_zoom" companion with a restricted x-axis (user-specified
2026-09-12: stim-aligned -20 to 100ms, lick-aligned -100 to 20ms) -- same
underlying curves/data, just a closer look at the response onset.

Reaction-time histogram overlay (user request 2026-09-13, extended
2026-09-15 to every stim-aligned time-resolved figure this script makes,
area-overlay included -- "all stim-aligned time-resolved figures must show
the little histogram at the bottom"), stim-aligned (`alignment=='stim'`)
figures: every R+/R-/aggregated/overlay panel, AND every area-overlay
panel, gets a whisker-trial hit reaction-time (`reaction_time` = lick_time - response_window_start_time, corrected 2026-09-27; was `lick_time - start_time`, ~100 ms too long)
histogram overlaid via an independent (`ax.twinx()`) y-axis, scaled so the
tallest bar sits just under the 0.5 chance line rather than sharing the
curve's own scale -- same layout/styling as the `035` preview (grey,
semi-transparent, thicker dark-grey edge). Deliberately always
whisker-trial hit RT regardless of `decode_target`/`condition_type` -- it's
a stable behavioral timing reference for "when do mice respond in this
task," not tied to whatever's being decoded in a given panel, so one RT
cache per tag covers every figure.

Stim-time histogram overlay (user request 2026-09-15: "For the lick time
time-resolved figures, plot the stim-time, the same way RTs are plotted"),
lick-aligned (`alignment=='lick'`) figures: the mirror-image counterpart --
every panel gets a whisker-trial hit **stimulus-onset** time
(`-reaction_time`, corrected first lick since 2026-09-27; was `start_time - lick_time`) histogram, same
overlay/scaling/styling mechanism as the RT histogram above (in fact the
same underlying per-session RT values, just negated -- `start_time` is
`lick_time` minus RT). `build_rt_cache`/`build_stim_time_cache` are
alignment-specific builders; `add_rt_histogram` itself is alignment-
agnostic (just overlays whatever per-session ms-offset array it's given),
reused unchanged for both.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import add_first_lick_time, prep_hitmiss_trials  # noqa: E402
from axel_bisi_paths import axel_bisi_path  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def build_rt_cache(session_ids) -> dict[str, np.ndarray]:
    """Whisker-trial hit reaction times (ms) per session -- see module
    docstring's "Reaction-time histogram overlay" section."""
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
    per session -- the lick-aligned mirror of `build_rt_cache` (same hits,
    same start_time/lick_time pair, opposite sign: RT = lick_time -
    start_time, this is start_time - lick_time = -RT). See module
    docstring's "Stim-time histogram overlay" section."""
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
    """Draws the histogram directly on `ax` in its own data coordinates
    (bars scaled to sit below the chance line) rather than via `ax.twinx()`
    -- a second axes sharing the same position as `ax` in a multi-column
    figure confuses matplotlib's `bbox_inches="tight"` bbox union and can
    silently clip the leftmost subplot's own ylabel/yticklabels (user report
    2026-09-16: "032_hitmiss_stim_area_group_whole_area_overlay.png still
    has the clipped y label on the left" -- reproduced and root-caused to
    this `twinx()` call, confirmed fixed by removing it)."""
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
    counts, edges = np.histogram(rt, bins=40, range=xlim)
    if counts.max() <= 0:
        return
    top = ylim[0] + frac_at_chance * 0.85 * (ylim[1] - ylim[0])
    heights = (counts / counts.max()) * (top - ylim[0])
    ax.bar(edges[:-1], heights, bottom=ylim[0], width=np.diff(edges), align="edge",
           color="#888888", alpha=0.35, edgecolor="#333333", linewidth=1.2, zorder=0)


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    """`fig.savefig` occasionally hits a transient Windows file-lock
    (`OSError: [Errno 22] Invalid argument` -- seen repeatedly this
    session, likely antivirus/cloud-sync briefly holding a newly-created
    file, not a real problem with the path or the figure) that a simple
    retry always clears."""
    fig.canvas.draw()  # force a full draw before bbox_inches="tight" measures it -- twinx()
    # axes (RT histogram) + set_box_aspect(1) can otherwise leave stale layout
    # info that clips the leftmost panel's ylabel/yticklabels (user report
    # 2026-09-16: "032_hitmiss_stim_area_group_whole_area_overlay.png still
    # has the clipped y label on the left").
    # 2026-09-27: also writes a vector .pdf next to the .png (user: all figures in pdf), same as 034's helper.
    for target in (out_path, out_path.with_suffix(".pdf")):
        last_err = None
        for attempt in range(attempts):
            try:
                fig.savefig(target, **kwargs)
                last_err = None
                break
            except OSError as e:
                last_err = e
                print(f"  savefig attempt {attempt+1}/{attempts} failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
        if last_err is not None:
            raise last_err


def fig_dir(tag: str) -> Path:
    """Figures organized per area level (user request 2026-09-13) and, under
    that, per day-stage (user request 2026-09-13: keep expert-stage results
    clearly separated from learning) -- both read off the tag itself (e.g.
    'hitmiss_stim_expert_whole_brain' -> 'whole_brain/expert'), not passed
    separately. `DAY_STAGE`'s tag suffix is literally '_expert' (see
    `024_master_sweep.py`'s `_suffix`); anything without it is learning-stage
    (the default, no suffix)."""
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
MAX_AREAS_PER_FIGURE = 8
# Machine-independent share root (see ssl-load/references/ssl_external_repo_paths.md
# and axel_bisi_paths.py) -- resolves to M:\... locally, /mnt/lsens-analysis/... on haas.
ALLEN_UTILS_PATH = axel_bisi_path("Github", "ephys_utilities", "ephys_utilities", "allen_utils")
# Areas dropped from every area_group figure (user request 2026-09-12:
# "Ignore somatosensory-body, amygdala, cortical subplate, insular areas,
# visual areas, olfactory" -- decluttering areas that aren't the focus of
# this whisker/auditory task; extended 2026-09-16, user request "Replot the
# area_group figures but excluding pons and medulla everywhere" -- "Pons and
# medulla" is the actual area_group category name for that combined region).
EXCLUDED_OVERLAY_AREAS = {
    "Somatosensory-body", "Amygdala and hypothalamus", "Cortical subplate",
    "Insular areas", "Visual areas", "Olfactory areas", "Pons and medulla",
}


def get_area_color_map(area_list: list[str]) -> dict[str, str]:
    """User's own allen_utils Allen-atlas-derived colors (per user request
    2026-09-12: "using colors of my allen_utils function"). `area_group`
    values are looked up directly in `get_custom_area_groups_colors()`
    (group-name keyed); finer `area_acronym_custom` values fall back to
    `get_custom_area_color_per_group()`'s single-acronym-to-group-color map;
    anything unmatched gets a neutral grey rather than erroring. Also falls
    back to grey for every area (with a warning, not a crash) if
    `ALLEN_UTILS_PATH` itself isn't reachable -- it lives on the `M:` network
    drive, which isn't always mounted for a non-interactive process (same
    class of issue `run_ssl_ks4_build.ps1` already handles for its own `M:`
    dependency)."""
    if ALLEN_UTILS_PATH is None:
        print("  WARNING: Axel_Bisi share not mounted on this machine (M:\\ / /mnt/lsens-analysis) -- using grey for all areas")
        return {area: "#888888" for area in area_list}
    sys.path.insert(0, str(ALLEN_UTILS_PATH))
    try:
        import allen_utils  # noqa: E402
    except ModuleNotFoundError:
        print(f"  WARNING: allen_utils not importable from {ALLEN_UTILS_PATH} -- using grey for all areas")
        return {area: "#888888" for area in area_list}

    group_colors = allen_utils.get_custom_area_groups_colors()
    acronym_colors, _ = allen_utils.get_custom_area_color_per_group()
    return {area: group_colors.get(area) or acronym_colors.get(area) or "#888888" for area in area_list}


def summarize(sub: pd.DataFrame) -> str:
    return f"n={sub['session_id'].nunique()} sessions, {sub['subject_id'].nunique()} mice"


def stack_curves(series: pd.Series) -> np.ndarray | None:
    curves = [np.array(c) for c in series if c is not None]
    return np.stack(curves) if curves else None


def null_mean_sem_curves(series: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Per row (session), `null_curves` is a set of label-shuffle null curves
    (see 024_master_sweep.py's N_SHUF) -- average across shuffles first (one
    mean-null curve per session), then stack across sessions exactly like
    `stack_curves` does for the real curve, giving an apples-to-apples
    mean+-SEM-across-sessions shuffled reference curve for the same panel.
    (Switched from a pooled-shuffle 95% CI band to this mean+-SEM treatment
    per user request 2026-09-12: "instead of plotting CI for the shuffle,
    plot sem" -- matches the real curve's own mean+-SEM-across-sessions
    style exactly, so the two are directly comparable.) `null_curves`
    round-trips through parquet as a 1-D object array of per-shuffle 1-D
    arrays (not a clean 2-D numeric array), so shuffle curves are stacked
    explicitly per row rather than relying on np.asarray on the whole column."""
    per_row_means = []
    for c in series:
        if c is None:
            continue
        shuf_curves = [np.asarray(s, dtype=float) for s in c if s is not None]
        if not shuf_curves:
            continue
        arr = np.stack(shuf_curves)
        if arr.ndim == 2 and arr.shape[0] > 0:
            per_row_means.append(np.nanmean(arr, axis=0))
    if not per_row_means:
        return None
    stacked = np.stack(per_row_means)  # (n_sessions, n_bins)
    mean_curve = np.nanmean(stacked, axis=0)
    sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
    return mean_curve, sem_curve


def _plot_curve(ax, sub: pd.DataFrame, bin_labels_ms: np.ndarray, color: str, values: tuple[str, str] | None,
                 null_col: str, null_label: str, label_prefix: str = "") -> bool:
    """Draws the real curve (mean+-SEM band) and its ONE null curve
    (mean+-SEM band) for one cohort/aggregate `sub` onto `ax`. Shared by the
    R+/R-/aggregated panels and by the cohort-overlay panel.

    `null_col`/`null_label` pick which null to show (user decision
    2026-09-16: "For hit/miss and perf-state decoding, it should always be
    a linear shift null plotted... For modality the label shuffle is
    good") -- one null per decode target, shown in EVERY panel including
    the pooled aggregate (supersedes the earlier policy of showing the
    label-shuffle null everywhere plus the shift null only in within-cohort
    panels)."""
    any_data = False
    if values is None:
        stacked = stack_curves(sub["real_curve"])
        if stacked is not None:
            any_data = True
            mean_curve = np.nanmean(stacked, axis=0)
            sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
            ax.plot(bin_labels_ms, mean_curve, color=color, lw=2.0, label=f"{label_prefix}real".strip())
            ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18, lw=0)
        null_band = null_mean_sem_curves(sub[null_col]) if null_col in sub.columns else None
        if null_band is not None:
            null_mean, null_sem = null_band
            ax.plot(bin_labels_ms, null_mean, color=color, lw=1.0, linestyle=":", alpha=0.6,
                     label=f"{label_prefix}{null_label} (mean+-SEM)".strip())
            ax.fill_between(bin_labels_ms, null_mean - null_sem, null_mean + null_sem, color=color, alpha=0.10, lw=0)
    else:
        for value, ls in zip(values, ("-", "--")):
            val_sub = sub[sub["condition_value"] == value]
            stacked = stack_curves(val_sub["real_curve"])
            if stacked is None:
                continue
            any_data = True
            mean_curve = np.nanmean(stacked, axis=0)
            sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
            ax.plot(bin_labels_ms, mean_curve, color=color, lw=2.0, linestyle=ls, label=f"{label_prefix}{value}".strip())
            ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.15, lw=0)
            null_band = null_mean_sem_curves(val_sub[null_col]) if null_col in val_sub.columns else None
            if null_band is not None:
                null_mean, null_sem = null_band
                ax.plot(bin_labels_ms, null_mean, color=color, lw=0.9, linestyle=ls, alpha=0.5,
                         label=f"{label_prefix}{value} {null_label}".strip())
                ax.fill_between(bin_labels_ms, null_mean - null_sem, null_mean + null_sem, color=color, alpha=0.07, lw=0)
    return any_data


def plot_row(axes_row, sub_area: pd.DataFrame, condition_type: str, values: tuple[str, str] | None, xlabel: str, area_label: str,
             zoom_xlim: tuple[float, float] | None = None, rt_cache: dict[str, np.ndarray] | None = None,
             null_col: str = "null_curves", null_label: str = "shuffled"):
    bin_labels_ms = sub_area.attrs["bin_labels_ms"]
    rplus = sub_area[sub_area.reward_group == "R+"]
    rminus = sub_area[sub_area.reward_group == "R-"]
    panels = [("R+", rplus), ("R-", rminus), ("R+ & R- aggregated", sub_area)]
    any_data = False
    for ax, (label, sub) in zip(axes_row[:3], panels):
        color = COHORT_COLOR.get(label, AGGREGATE_COLOR)
        got = _plot_curve(ax, sub, bin_labels_ms, color, values, null_col, null_label)
        any_data = any_data or got
        if got:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_title(f"{area_label} | {label}\n({summarize(sub)})", fontsize=8.5)

    # 4th panel, every time-resolved figure: R+ and R- overlaid on one axes
    # (not pooled into one aggregate curve like panel 3) -- per user request
    # 2026-09-12: "as fourth subplots, an overlay of cohorts."
    ax_overlay = axes_row[3]
    got_p = _plot_curve(ax_overlay, rplus, bin_labels_ms, COHORT_COLOR["R+"], values, null_col, null_label, label_prefix="R+ ")
    got_m = _plot_curve(ax_overlay, rminus, bin_labels_ms, COHORT_COLOR["R-"], values, null_col, null_label, label_prefix="R- ")
    if got_p or got_m:
        any_data = True
        ax_overlay.legend(fontsize=6, frameon=False, loc="upper left", ncol=1)
    ax_overlay.set_title(f"{area_label} | R+ & R- overlaid\n({summarize(sub_area)})", fontsize=8.5)

    xlim_raw = zoom_xlim if zoom_xlim is not None else (float(bin_labels_ms.min()), float(bin_labels_ms.max()))
    xpad = 0.02 * (xlim_raw[1] - xlim_raw[0])
    xlim = (xlim_raw[0] - xpad, xlim_raw[1] + xpad)
    acc_ylim = (0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped by the axes border
    if rt_cache is not None:
        for ax, sub in zip(axes_row, (rplus, rminus, sub_area, sub_area)):
            add_rt_histogram(ax, sub["session_id"].unique(), rt_cache, xlim, acc_ylim)

    for ax in axes_row:
        ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
        ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
        ax.set_ylim(*acc_ylim)
        ax.set_box_aspect(1)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("balanced accuracy", fontsize=8)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    return any_data


def make_figure(df: pd.DataFrame, bin_labels_ms: np.ndarray, condition_type: str, values: tuple[str, str] | None,
                 title: str, out_path: Path, xlabel: str, zoom_xlim: tuple[float, float] | None = None,
                 rt_cache: dict[str, np.ndarray] | None = None, null_col: str = "null_curves", null_label: str = "shuffled"):
    sub_ct = df[df["condition_type"] == condition_type].copy()
    if len(sub_ct) == 0:
        print(f"  no '{condition_type}' rows, skipping {out_path.name}")
        return
    sub_ct.attrs["bin_labels_ms"] = bin_labels_ms

    areas = sorted((a for a in sub_ct["area_value"].dropna().unique() if a not in EXCLUDED_OVERLAY_AREAS),
                    key=lambda a: -sub_ct[sub_ct.area_value == a]["session_id"].nunique())
    areas = areas[:MAX_AREAS_PER_FIGURE]
    if len(areas) > 1:
        print(f"  {condition_type}: {len(sub_ct['area_value'].unique())} areas total, plotting top {len(areas)} by session count")

    fig, axes = plt.subplots(len(areas), 4, figsize=(14.7, 3.6 * len(areas)), squeeze=False)
    any_data_overall = False
    for row_i, area in enumerate(areas):
        sub_area = sub_ct[sub_ct.area_value == area].copy()
        sub_area.attrs["bin_labels_ms"] = bin_labels_ms
        got_data = plot_row(axes[row_i], sub_area, condition_type, values, xlabel, area, zoom_xlim=zoom_xlim, rt_cache=rt_cache,
                             null_col=null_col, null_label=null_label)
        any_data_overall = any_data_overall or got_data

    if not any_data_overall:
        plt.close(fig)
        print(f"  no computable curves, skipping {out_path.name}")
        return

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_retry(fig, out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def make_area_overlay_figure(df: pd.DataFrame, bin_labels_ms: np.ndarray, condition_type: str, values: tuple[str, str] | None,
                              title: str, out_path: Path, xlabel: str, zoom_xlim: tuple[float, float] | None = None,
                              rt_cache: dict[str, np.ndarray] | None = None, null_col: str = "null_curves", null_label: str = "shuffled"):
    """Within-cohort (R+, R-), within-condition-value figure with all areas'
    real curves overlaid on one panel each, colored by the user's allen_utils
    area colors -- for directly comparing areas rather than comparing
    cohorts or conditions. Rows = condition_value (just "whole" when
    `values` is None), cols = cohort. Skipped when <2 areas are present
    (e.g. whole_brain), since there is nothing to overlay/compare.

    RT histogram overlay (user request 2026-09-15: "all stim-aligned
    time-resolved figures must show the little histogram at the bottom" --
    previously this figure was explicitly excluded; that exclusion is
    superseded).

    Null distribution overlay (user request 2026-09-16: "plot the
    corresponding null distributions", then corrected: "For hit/miss and
    perf-state decoding, it should always be a linear shift null plotted
    ... For modality the label shuffle is good"): each area's ONE null
    (`null_col`/`null_label`, mean+-SEM across sessions, same
    `null_mean_sem_curves` helper the per-area faceted figures use) is
    drawn as a thin dotted line in that area's own color, directly under
    its real curve."""
    sub_ct = df[df["condition_type"] == condition_type].copy()
    if len(sub_ct) == 0:
        print(f"  no '{condition_type}' rows, skipping {out_path.name}")
        return
    areas = sorted(a for a in sub_ct["area_value"].dropna().unique() if a not in EXCLUDED_OVERLAY_AREAS)
    if len(areas) < 2:
        print(f"  {condition_type}: only {len(areas)} area(s) after exclusions, skipping area-overlay figure")
        return
    color_map = get_area_color_map(areas)

    cond_values = values if values is not None else ("whole",)
    cohorts = ["R+", "R-"]
    fig, axes = plt.subplots(len(cond_values), len(cohorts), figsize=(6.0 * len(cohorts), 6.0 * len(cond_values)), squeeze=False)
    any_data = False
    for row_i, cond_val in enumerate(cond_values):
        for col_i, cohort in enumerate(cohorts):
            ax = axes[row_i][col_i]
            sub = sub_ct[sub_ct.reward_group == cohort]
            if values is not None:
                sub = sub[sub.condition_value == cond_val]
            for area in areas:
                area_sub = sub[sub.area_value == area]
                stacked = stack_curves(area_sub["real_curve"])
                if stacked is None:
                    continue
                any_data = True
                mean_curve = np.nanmean(stacked, axis=0)
                sem_curve = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])
                color = color_map[area]
                ax.plot(bin_labels_ms, mean_curve, color=color, lw=1.6, label=area)
                ax.fill_between(bin_labels_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.12, lw=0)
                null_band = null_mean_sem_curves(area_sub[null_col]) if null_col in area_sub.columns else None
                if null_band is not None:
                    null_mean, null_sem = null_band
                    ax.plot(bin_labels_ms, null_mean, color=color, lw=0.9, linestyle=":", alpha=0.6)
                    ax.fill_between(bin_labels_ms, null_mean - null_sem, null_mean + null_sem, color=color, alpha=0.07, lw=0)
            ax.axhline(0.5, color="#888888", lw=1, linestyle=":", zorder=0)
            ax.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.4, zorder=0)
            acc_ylim = (0.4, 1.02)  # headroom so markers/curves at the accuracy ceiling aren't clipped by the axes border
            ax.set_ylim(*acc_ylim)
            ax.set_box_aspect(1)
            xlim_raw = zoom_xlim if zoom_xlim is not None else (float(bin_labels_ms.min()), float(bin_labels_ms.max()))
            xpad = 0.02 * (xlim_raw[1] - xlim_raw[0])
            xlim = (xlim_raw[0] - xpad, xlim_raw[1] + xpad)
            if rt_cache is not None:
                add_rt_histogram(ax, sub["session_id"].unique(), rt_cache, xlim, acc_ylim)
            ax.set_xlim(*xlim)
            ax.set_title(f"{cohort} | {cond_val}\n({summarize(sub)})", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("balanced accuracy", fontsize=8)
            ax.tick_params(axis="both", labelsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                null_proxy = plt.Line2D([0], [0], color="#888888", lw=0.9, linestyle=":", alpha=0.7)
                ax.legend(handles + [null_proxy], labels + [f"{null_label} (per area, dotted)"], fontsize=6, frameon=False, loc="upper left", ncol=2)

    if not any_data:
        plt.close(fig)
        print(f"  no computable curves, skipping {out_path.name}")
        return
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_retry(fig, out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 032_plot_decode_results.py <tag>")
        sys.exit(1)
    tag = sys.argv[1]

    partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
    edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
    if not partial_path.exists() or not edges_path.exists():
        print(f"{tag}: missing results/bin-edges file(s), nothing to plot")
        return

    df = pd.read_parquet(partial_path)
    df = df[df["skipped_reason"].isna()].copy()
    if len(df) == 0:
        print(f"{tag}: results file exists but no computed rows yet")
        return

    bin_edges = json.loads(edges_path.read_text())
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
    alignment = df["alignment"].iloc[0] if "alignment" in df.columns else ("stim" if "stim" in tag else "lick")
    xlabel = "time from start_time (ms)" if alignment == "stim" else "time from first lick (ms, corrected)"

    print(f"{tag}: {df['session_id'].nunique()} sessions, condition_types present: {sorted(df['condition_type'].dropna().unique())}")
    out_dir = fig_dir(tag)

    # Which null to plot (user decision 2026-09-16; extended 2026-09-17):
    # hitmiss/perfstate always show the linear-shift null. modality_lick
    # ALSO shows the shift null now (2026-09-17 finding: its lick_flag==1
    # hit-filter makes whisker-hit trials non-uniform in time within a
    # session, unlike auditory-hit trials, which a label-shuffle null
    # can't catch -- see `024_master_sweep.py`'s `compute_shift_null`
    # comment for the full reasoning). modality_stim keeps the label-
    # shuffle null (unaffected -- it uses the full, unfiltered trial set).
    # Falls back to label-shuffle if decode_target/alignment are somehow
    # absent from the data (shouldn't happen for current tags).
    decode_target = df["decode_target"].iloc[0] if "decode_target" in df.columns else (
        "hitmiss" if "hitmiss" in tag else "perfstate" if "perfstate" in tag else "modality")
    if decode_target in ("hitmiss", "perfstate") or (decode_target == "modality" and alignment == "lick"):
        null_col, null_label = "shift_null_curves", "shift null"
    else:
        null_col, null_label = "null_curves", "shuffled"

    # RT / stim-time histogram overlay -- see module docstring's "Reaction-
    # time histogram overlay" / "Stim-time histogram overlay" sections.
    rt_cache = None
    if alignment == "stim":
        rt_cache = build_rt_cache(df["session_id"].unique())
        print(f"  RT histogram cache: {len(rt_cache)}/{df['session_id'].nunique()} sessions")
    elif alignment == "lick":
        rt_cache = build_stim_time_cache(df["session_id"].unique())
        print(f"  stim-time histogram cache: {len(rt_cache)}/{df['session_id'].nunique()} sessions")

    # Zoomed x-axis range, user-specified 2026-09-12: stim-aligned -20 to
    # 100ms (fast sensory-onset window), lick-aligned -100 to 20ms (pre-lick
    # motor-preparation window) -- same values used for every zoomed figure
    # below, per "the zoomed in version figures I mentioned earlier."
    zoom_xlim = (-20, 100) if alignment == "stim" else (-100, 20)

    condition_specs = [
        ("whole", None, "whole session"),
        ("half", ("first", "second"), "session-half comparison"),
        ("perfstate", ("high", "low"), "performance-state comparison"),
    ]
    for condition_type, values, desc in condition_specs:
        make_figure(df, bin_labels_ms, condition_type, values, f"{tag} -- {desc}",
                    out_dir / f"032_{tag}_{condition_type}.png", xlabel, rt_cache=rt_cache, null_col=null_col, null_label=null_label)
        make_figure(df, bin_labels_ms, condition_type, values, f"{tag} -- {desc} (zoomed)",
                    out_dir / f"032_{tag}_{condition_type}_zoom.png", xlabel, zoom_xlim=zoom_xlim, rt_cache=rt_cache,
                    null_col=null_col, null_label=null_label)
        make_area_overlay_figure(df, bin_labels_ms, condition_type, values, f"{tag} -- {desc}, areas overlaid",
                                  out_dir / f"032_{tag}_{condition_type}_area_overlay.png", xlabel, rt_cache=rt_cache,
                                  null_col=null_col, null_label=null_label)
        make_area_overlay_figure(df, bin_labels_ms, condition_type, values, f"{tag} -- {desc}, areas overlaid (zoomed)",
                                  out_dir / f"032_{tag}_{condition_type}_area_overlay_zoom.png", xlabel, zoom_xlim=zoom_xlim,
                                  rt_cache=rt_cache, null_col=null_col, null_label=null_label)


if __name__ == "__main__":
    main()
