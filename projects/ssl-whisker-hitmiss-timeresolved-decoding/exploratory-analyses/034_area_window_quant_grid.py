"""Area-level window-quantification grid (user request 2026-09-13, revised
twice after seeing examples -- see conversation history for the two earlier
layouts this superseded).

One figure per tag (per cohort, plus a cohort-comparison version): a grid
with **rows = window metric** (baseline -200,-10ms; secondary = sensory
5-35ms for stim-aligned tags or pre-lick -100-0ms for lick-aligned tags) and
**columns = condition** (`whole`, `half:first`, `half:second` -- the
`perfstate` condition and every cross-generalization curve are dropped per
user request as redundant/out of scope for this figure).

Each subplot: x-axis = areas common to both cohorts (same exclusion list as
`032`'s area-overlay figures), ordered by the user's own allen_utils
canonical order and colored by the user's own allen_utils area colors (both
the point and the x-tick label -- no legend for area identity, too many
areas to stay legible). In the comparison figure, cohort is distinguished by
marker EDGE color (R+ = `COHORT_COLOR["R+"]` #00B400, R- = `COHORT_COLOR
["R-"]` #C800C8 -- established SSL R+/R- convention), while the marker FACE
stays the area color (2026-09-24, user: "keep area colors but change
circles and triangles by shades of colors: R+ are the rplus color and R-
are rminus color" -- supersedes the original circle=R+/triangle=R- shape
convention). y-axis is decoding accuracy minus chance, shared across every
subplot.

Linear-shift null (user request 2026-09-13), **per-cohort figures only**
(not the comparison figure -- the shift null is a within-cohort control):
when a tag's rows carry `shift_null_curves` (hitmiss/perfstate only -- see
`024_master_sweep.py`), "chance" for the colored real point becomes that
session's own shift-null window value (paired per session) instead of the
theoretical 0.5, per user request ("figures showing perf relative to chance
should use this as chance when relevant (hit/miss, perf state)"), and a
second grey point per area shows where the empirical shift-null itself sits
relative to 0.5 -- so both "real vs. empirical chance" and "how far the
empirical chance drifted from 0.5" are visible. Tags without
`shift_null_curves` (modality, or any not-yet-rerun tag) fall back to the
original real-minus-0.5 behavior unchanged.

Cohort-comparison stats (user request 2026-09-13), comparison figure only:
each subplot's title reports a two-way ANOVA (cohort x area, with
interaction) main effect of cohort across areas, and a non-parametric
Mann-Whitney U post-hoc per area, FDR-corrected (Benjamini-Hochberg) across
the areas in that one subplot (user request 2026-09-13: FDR for the
post-ANOVA post-hoc specifically, not the ANOVA itself), is annotated as an
asterisk above any area where R+ vs R- differs at q<0.05 --
see `cohort_anova_and_posthoc`. Per-cohort and comparison-figure legends
both removed per the same request (area identity is already carried by
color/x-tick-label; cohort marker edge color -- #00B400=R+, #C800C8=R- --
stays defined only here, in `COHORT_COLOR`/this docstring, not in-figure).

Combined-cohort variant (user request 2026-09-13), performance-state
decoding only (`decode_target=='perfstate'`, i.e. tags actually decoding
perf-state itself -- not perfstate as a condition split of some other
target): a third grid (`..._grid_combined.png`, alongside `_grid_plus`/
`_grid_minus`) pools R+ and R- sessions into one group rather than
comparing them, via the same `plot_group_grid` used for the per-cohort
grids (just given the full, unfiltered-by-cohort dataframe).

Usage: python 034_area_window_quant_grid.py <tag>
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
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_1samp
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ssl_timeresolved_decoding import mean_in_window  # noqa: E402


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    """`fig.savefig` occasionally hits a transient Windows file-lock
    (`OSError: [Errno 22] Invalid argument`, likely antivirus/cloud-sync
    briefly holding a newly-created file) that a simple retry always
    clears -- same helper as `030`/`032`/`043`/`046`, added here 2026-09-17
    (this was the last figure script in the project still using a bare
    `fig.savefig`).

    2026-09-24 update (user: "Regenerate the figures in higher resolution,
    also in pdf"): now saves BOTH a `.png` (at whatever dpi the caller
    passes -- bumped from 140 to 300 at every call site in this file) and
    a `.pdf` (vector -- dpi only matters for the PNG raster) at
    the same base path, each with its own independent retry."""
    for target in (out_path, out_path.with_suffix(".pdf")):
        last_err = None
        for attempt in range(attempts):
            try:
                fig.savefig(target, **kwargs)
                break
            except OSError as e:
                last_err = e
                print(f"  savefig attempt {attempt+1}/{attempts} failed for {target.name} ({e}), retrying in {delay}s...")
                time.sleep(delay)
        else:
            raise last_err

OUT_DIR = Path(__file__).resolve().parent


def fig_dir(tag: str) -> Path:
    """Figures organized per area level and day-stage (user request
    2026-09-13) -- same convention as `032_plot_decode_results.py`'s
    `fig_dir`."""
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


# Default assumes this machine's `M:` network drive; on haas056 (or any
# other remote host), the same NAS is mounted at `/mnt/lsens-analysis`
# instead -- override via SSL_ALLEN_UTILS_PATH (user 2026-09-15: "the
# location of the utils from the Haas are not visible at M:\analysis but at
# /mnt/lsens-analysis").
# 2026-09-28: default resolved via scripts/axel_bisi_paths.py (M: locally, /mnt/lsens-analysis on haas) instead of
# a hardcoded M: path, which silently fell back to grey colors / alphabetical order on haas.
from axel_bisi_paths import axel_bisi_path  # noqa: E402

ALLEN_UTILS_PATH = os.environ.get("SSL_ALLEN_UTILS_PATH") or str(
    axel_bisi_path("Github", "ephys_utilities", "ephys_utilities", "allen_utils"))
COHORT_MARKER = {"R+": "o", "R-": "^"}  # superseded 2026-09-24 for cohort-comparison
# plots (user: "change circles and triangles by shades of colors") -- kept
# only for any leftover generic callers of `plot_factor_comparison_grid`
# with non-cohort factors (e.g. day_stage), which still use shape.
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}  # established SSL R+/R-
# convention (`ssl_cross_area_rplus_rminus_colors.md`) -- now also used as
# the marker EDGE color to distinguish R+/R- in the cohort-comparison
# grids, replacing circle-vs-triangle shape (2026-09-24, user: "keep area
# colors but change circles and triangles by shades of colors: R+ are the
# rplus color and R- are rminus color"). Area color stays the marker FACE.
YLIM_DIFF = (-0.10, 0.30)
BASELINE_WINDOW = (-0.200, -0.010)
SENSORY_WINDOW = (0.005, 0.050)  # updated 2026-09-18 from (0.005, 0.035), user request
PRE_LICK_WINDOW = (-0.100, 0.0)
EXCLUDED_OVERLAY_AREAS = {
    "Somatosensory-body", "Amygdala and hypothalamus", "Cortical subplate",
    "Insular areas", "Visual areas", "Olfactory areas", "Pons and medulla",
}  # extended 2026-09-16, user request "Replot the area_group figures but excluding pons and medulla everywhere"
# Columns: (condition_type, condition_value, display_label). `perfstate` and
# every cross-gen curve dropped per user request 2026-09-13 ("ignore the
# perf state decoding per state, as it is redundant, ignore the
# cross-condition tests").
CONDITION_COLUMNS = [
    ("whole", "whole", "whole"),
    ("half", "first", "half:first"),
    ("half", "second", "half:second"),
]
# Rows: (metric prefix, window, display label).
METRIC_ROWS = [
    ("baseline", BASELINE_WINDOW, "baseline (-200,-10ms)"),
    ("secondary", None, None),  # window/label resolved per-tag alignment in load()
]


def _import_allen_utils():
    """`ALLEN_UTILS_PATH` lives on the `M:` network drive, which isn't always
    mounted for a non-interactive process (same class of issue
    `run_ssl_ks4_build.ps1` already handles for its own `M:` dependency) --
    return None rather than crashing when it's unreachable; callers already
    have a defined fallback (grey color / alphabetical order)."""
    sys.path.insert(0, ALLEN_UTILS_PATH)
    try:
        import allen_utils  # noqa: E402
        return allen_utils
    except ModuleNotFoundError:
        print(f"  WARNING: allen_utils not importable from {ALLEN_UTILS_PATH} (M: drive not mounted?) -- using grey/alphabetical fallback")
        return None


def get_area_color_map(area_list: list[str]) -> dict[str, str]:
    allen_utils = _import_allen_utils()
    if allen_utils is None:
        return {a: "#888888" for a in area_list}
    group_colors = allen_utils.get_custom_area_groups_colors()
    acronym_colors, _ = allen_utils.get_custom_area_color_per_group()
    return {a: group_colors.get(a) or acronym_colors.get(a) or "#888888" for a in area_list}


def lighten(hex_color: str, amount: float = 0.55) -> str:
    """Blend `hex_color` toward white by `amount` (0=unchanged, 1=white) --
    used to make the 'first half' condition marker a visibly less
    saturated version of the area color, while 'second half' stays at
    full saturation (user request 2026-09-21: "Make the first half
    condition a less saturated than the second half")."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


AREA_COUNT_SQUARE_THRESHOLD = 20  # areas at/below this stay square (paired with `style_ax`'s box_aspect)


def area_fig_width_mult(areas: list[str], base: float = 3.6) -> float:
    """Per-column subplot-width multiplier (2026-09-24, user: "make the
    figures non-square, slightly rectangular to account for large number
    of areas on the x-axis") -- `base` (the multiplier every `figsize=
    (base * len(columns), ...)` call used unconditionally before) for
    area_group/whole_brain-sized area lists (~13, <= `AREA_COUNT_SQUARE_
    THRESHOLD`), scaled up proportionally for area_acronym_custom's much
    longer list (~46) so its x-axis area labels have room instead of
    overlapping into illegibility. Paired with `style_ax`'s box_aspect,
    scaled down by the same ratio, so the extra width is actually used by
    the axes box rather than left as whitespace around a still-square plot."""
    return max(base, base * len(areas) / AREA_COUNT_SQUARE_THRESHOLD)


def area_order_for(area_col: str) -> list[str]:
    allen_utils = _import_allen_utils()
    if allen_utils is None:
        return []
    if area_col == "area_group":
        return allen_utils.get_area_group_custom_order()
    if area_col == "area_acronym_custom":
        return allen_utils.get_area_acronym_custom_order()
    return []


def curve_window_mean(curve, bin_labels_s: np.ndarray, window: tuple[float, float]) -> float:
    if curve is None:
        return float("nan")
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        return float("nan")
    return mean_in_window(arr, bin_labels_s, window)


def shift_null_window_mean(shift_curves, bin_labels_s: np.ndarray, window: tuple[float, float]) -> float:
    """`shift_curves` is a per-row set of linear-shift-null curves (same
    round-trip-through-parquet shape as `null_curves` elsewhere in this
    pipeline: a 1-D object sequence of per-shuffle 1-D arrays). Average
    across shuffles first, then take the window mean -- one empirical
    per-session chance-level estimate for this window and metric."""
    if shift_curves is None:
        return float("nan")
    per_shuf = [np.asarray(s, dtype=float) for s in shift_curves if s is not None]
    if not per_shuf:
        return float("nan")
    mean_curve = np.nanmean(np.stack(per_shuf), axis=0)
    return mean_in_window(mean_curve, bin_labels_s, window)


def load(tag: str):
    """Returns `(None, None, None)` when results/bin-edges aren't ready yet
    or the file exists but every row was skipped (e.g. class-imbalance at
    expert stage, 2026-09-17 fix -- this previously crashed with an
    out-of-bounds `.iloc[0]` on an empty, all-skipped dataframe) -- same
    graceful-skip convention `025`/`027`-`030` already use."""
    partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
    edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
    if not partial_path.exists() or not edges_path.exists():
        return None, None, None, None
    df = pd.read_parquet(partial_path)
    df = df[df["skipped_reason"].isna()].copy()
    if len(df) == 0:
        return None, None, None, None
    bin_edges = json.loads(edges_path.read_text())
    bin_labels_s = np.array([e[1] for e in bin_edges])
    alignment = df["alignment"].iloc[0] if "alignment" in df.columns else ("stim" if "stim" in tag else "lick")
    decode_target = df["decode_target"].iloc[0] if "decode_target" in df.columns else ("modality" if "modality" in tag else "hitmiss")
    secondary_window = SENSORY_WINDOW if alignment == "stim" else PRE_LICK_WINDOW
    secondary_label = (
        f"sensory ({int(SENSORY_WINDOW[0] * 1000)},{int(SENSORY_WINDOW[1] * 1000)}ms)"
        if alignment == "stim"
        else f"pre-lick ({int(PRE_LICK_WINDOW[0] * 1000)},{int(PRE_LICK_WINDOW[1] * 1000)}ms)"
    )
    rows = [("secondary", secondary_window, secondary_label)]
    # Baseline (pre-stimulus, -200,-10ms relative to start_time) only makes
    # sense for hitmiss/perfstate decoding -- per user request 2026-09-13:
    # "Baseline is relative to stim-alignment only, per target type but only
    # meaningful for hit/miss decoding and state decoding, not modality
    # decoding." Modality (whisker vs auditory trial_type) can't be
    # meaningfully decoded before the stimulus defines it, so a baseline row
    # there would just measure incidental structure, not real coding -- and
    # since hitmiss/perfstate are stim-aligned only anyway, this also
    # naturally excludes baseline from ever being computed relative to lick.
    if decode_target in ("hitmiss", "perfstate"):
        rows.insert(0, ("baseline", BASELINE_WINDOW, "baseline (-200,-10ms)"))
    # 2026-09-24 (user: "you can do the equivalent which is accuracy minus
    # chance, they have trial shuffles"): tags without a true linear-shift
    # null (modality_stim -- see `compute_shift_null` in `024_master_
    # sweep.py`) still carry a label-shuffle `null_curves` column (the
    # decode_target=='modality'/alignment=='stim' null), same per-shuffle
    # array-of-arrays shape as `shift_null_curves` (see `shift_null_
    # window_mean`'s own docstring) -- fall back to that instead of
    # leaving the "_shift"-suffixed control column entirely absent, so
    # every downstream `{metric_prefix}_shift`-consuming figure (above-
    # control, cohort ANOVA, the grey null reference) gets a real,
    # per-session-paired null instead of silently rendering empty.
    # `null_kind` tracks which one was actually used, purely for honest
    # figure labeling -- the two are NOT interchangeable statistically
    # (shift-null is a within-session temporal-structure control,
    # label-shuffle is a trial-identity control) and downstream text says
    # "shift null" vs "label-shuffle null" accordingly.
    has_true_shift = "shift_null_curves" in df.columns and df["shift_null_curves"].notna().any()
    has_label_null = "null_curves" in df.columns and df["null_curves"].notna().any()
    if has_true_shift:
        null_source_col, null_kind = "shift_null_curves", "shift"
    elif has_label_null:
        null_source_col, null_kind = "null_curves", "label-shuffle"
    else:
        null_source_col, null_kind = None, None
    has_shift_null = null_source_col is not None
    for metric_prefix, window, _ in rows:
        df[metric_prefix] = df["real_curve"].apply(lambda c: curve_window_mean(c, bin_labels_s, window))
        if has_shift_null:
            df[f"{metric_prefix}_shift"] = df[null_source_col].apply(
                lambda c: shift_null_window_mean(c, bin_labels_s, window))
    return df, rows, has_shift_null, null_kind


def common_areas(df: pd.DataFrame, area_col: str) -> list[str]:
    areas_rplus = set(df[df.reward_group == "R+"]["area_value"].dropna().unique())
    areas_rminus = set(df[df.reward_group == "R-"]["area_value"].dropna().unique())
    common = (areas_rplus & areas_rminus) - EXCLUDED_OVERLAY_AREAS
    order = area_order_for(area_col)
    ordered = [a for a in order if a in common]
    return ordered if ordered else sorted(common)


def present_columns(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    have = set(df[["condition_type", "condition_value"]].dropna().drop_duplicates().itertuples(index=False, name=None))
    return [c for c in CONDITION_COLUMNS if (c[0], c[1]) in have]


def area_means(sub: pd.DataFrame, areas: list[str], metric_col: str) -> tuple[np.ndarray, np.ndarray]:
    means, sems = [], []
    for area in areas:
        vals = sub[sub.area_value == area][metric_col].dropna().to_numpy()
        means.append(np.mean(vals) if len(vals) else np.nan)
        sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(sems)


def area_paired_diff(sub: pd.DataFrame, areas: list[str], metric_prefix: str, has_shift_null: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-area mean+-SEM of (real - chance), paired per session. Chance is
    the session's own linear-shift-null window value when available
    (hitmiss/perfstate decoding only) rather than the theoretical 0.5, per
    user request 2026-09-13: "figures showing perf relative to chance
    should use this as chance when relevant (hit/miss, perf state)"."""
    means, sems = [], []
    shift_col = f"{metric_prefix}_shift"
    for area in areas:
        a_sub = sub[sub.area_value == area]
        if has_shift_null and shift_col in a_sub.columns:
            paired = (a_sub[metric_prefix] - a_sub[shift_col]).dropna().to_numpy()
        else:
            paired = (a_sub[metric_prefix] - 0.5).dropna().to_numpy()
        means.append(np.mean(paired) if len(paired) else np.nan)
        sems.append(np.std(paired) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0)
    return np.array(means), np.array(sems)


def area_shift_ref(sub: pd.DataFrame, areas: list[str], metric_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-area mean+-SEM of (shift-null - 0.5) -- where the empirical
    linear-shift null itself sits relative to theoretical chance, shown as
    a grey reference point per user request 2026-09-13: shift-null curves
    "should be plotted on ... area quantification figures (within
    cohort)"."""
    means, sems = [], []
    shift_col = f"{metric_prefix}_shift"
    for area in areas:
        vals = (sub[sub.area_value == area][shift_col] - 0.5).dropna().to_numpy()
        means.append(np.mean(vals) if len(vals) else np.nan)
        sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(sems)


def area_above_control_diff(sub: pd.DataFrame, areas: list[str], metric_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-area mean+-SEM of (real - shift-null), paired per session --
    added 2026-09-15 (user request: "the equivalent figures with the
    decoding accuracy above the control", as its own explicit figure
    rather than mixed into `area_paired_diff`'s chance-or-shift-null
    fallback). Unlike `area_paired_diff`, this does NOT fall back to 0.5
    when shift-null is unavailable (modality) -- returns NaN for that
    row/area instead, since "above control" is meaningless without a real
    control to be above."""
    means, sems = [], []
    shift_col = f"{metric_prefix}_shift"
    for area in areas:
        a_sub = sub[sub.area_value == area]
        if shift_col in a_sub.columns:
            paired = (a_sub[metric_prefix] - a_sub[shift_col]).dropna().to_numpy()
        else:
            paired = np.array([])
        means.append(np.mean(paired) if len(paired) else np.nan)
        sems.append(np.std(paired) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0)
    return np.array(means), np.array(sems)


def compute_shared_ylim_raw(df: pd.DataFrame, areas: list[str], columns: list[tuple[str, str, str]],
                             rows: list[tuple[str, tuple, str]], pad_frac: float = 0.1) -> tuple[float, float]:
    """Raw-accuracy analog of `compute_shared_ylim` -- same tailor-to-data
    approach, but on `area_means` (no chance/control subtraction) and
    guaranteeing the 0.5 chance line stays visible instead of 0.0."""
    los, his = [], []
    for cohort in ("R+", "R-"):
        sub_cohort = df[df.reward_group == cohort]
        for metric_prefix, _, _ in rows:
            for ctype, cval, _ in columns:
                sub = sub_cohort[(sub_cohort.condition_type == ctype) & (sub_cohort.condition_value == cval)]
                mean, sem = area_means(sub, areas, metric_prefix)
                valid = ~np.isnan(mean)
                if valid.any():
                    los.append(np.nanmin((mean - sem)[valid]))
                    his.append(np.nanmax((mean + sem)[valid]))
                shift_col = f"{metric_prefix}_shift"
                if shift_col in sub.columns:
                    nmean, nsem = area_means(sub, areas, shift_col)
                    nvalid = ~np.isnan(nmean)
                    if nvalid.any():
                        los.append(np.nanmin((nmean - nsem)[nvalid]))
                        his.append(np.nanmax((nmean + nsem)[nvalid]))
    if not los:
        return (0.3, 1.0)
    lo, hi = min(los), max(his)
    pad = pad_frac * (hi - lo) if hi > lo else 0.02
    lo, hi = lo - pad, hi + pad
    lo, hi = min(lo, 0.45), max(hi, 0.55)  # always show a sliver on both sides of the 0.5 chance line
    return (lo, hi)


def compute_shared_ylim_abovecontrol(df: pd.DataFrame, areas: list[str], columns: list[tuple[str, str, str]],
                                      rows: list[tuple[str, tuple, str]], pad_frac: float = 0.15) -> tuple[float, float]:
    """Above-control analog of `compute_shared_ylim`, using
    `area_above_control_diff` -- falls back to `YLIM_DIFF` when nothing is
    plottable (e.g. modality, which has no shift-null at all)."""
    los, his = [], []
    for cohort in ("R+", "R-"):
        sub_cohort = df[df.reward_group == cohort]
        for metric_prefix, _, _ in rows:
            for ctype, cval, _ in columns:
                sub = sub_cohort[(sub_cohort.condition_type == ctype) & (sub_cohort.condition_value == cval)]
                mean, sem = area_above_control_diff(sub, areas, metric_prefix)
                valid = ~np.isnan(mean)
                if valid.any():
                    los.append(np.nanmin((mean - sem)[valid]))
                    his.append(np.nanmax((mean + sem)[valid]))
    if not los:
        return YLIM_DIFF
    lo, hi = min(los), max(his)
    pad = pad_frac * (hi - lo) if hi > lo else 0.02
    lo, hi = lo - pad, hi + pad
    lo, hi = min(lo, -0.01), max(hi, 0.02)
    return (lo, hi)


def factor_anova_and_posthoc(sub: pd.DataFrame, areas: list[str], metric_prefix: str,
                              factor_col: str = "reward_group", level_a: str = "R+", level_b: str = "R-",
                              shift_col: str | None = None) -> tuple[float, float, dict[str, float]]:
    """Two-way ANOVA (`factor_col` x area, with interaction) on per-session
    (metric - 0.5) values, testing the main effect of `factor_col` (its
    `level_a`/`level_b` levels) across areas (user request 2026-09-13,
    generalized 2026-09-18 from the original cohort-only version -- see
    `cohort_anova_and_posthoc` below for the reward_group-specific wrapper
    every existing caller still uses unchanged) -- the parametric half of
    this project's established paired-test convention
    (`ssl_rplus_rminus_test_pair.md`: always report both a parametric and a
    non-parametric test), now reused for any two-level factor (cohort,
    within-session condition, or learning/expert stage). Mann-Whitney U is
    the non-parametric post-hoc, run separately per area then FDR-corrected
    (Benjamini-Hochberg, across the areas tested in this one subplot) --
    user request 2026-09-13: FDR only for the post-ANOVA post-hoc, NOT for
    the ANOVA itself (one omnibus test per subplot, nothing to correct
    across) and unlike `026_wholebrain_window_stats.py`'s uncorrected
    per-cell tests (a different, coarser-grained test family -- this
    correction is local to one subplot's areas, not across this script's
    other subplots/tags). Returns (F, p) for the `factor_col` main effect
    and a per-area FDR-corrected MWU p-value (q-value) dict. Treats each
    (session, area) row as one independent observation -- a simplification
    (not mouse-block permutation like this project's confirmatory
    group-level tests, and not a repeated-measures/paired ANOVA even when
    `level_a`/`level_b` are two conditions of the SAME session -- same
    simplification the original cohort version already made) appropriate
    for this exploratory area-level grid.

    `shift_col` (added 2026-09-15, for the above-control figure variant):
    when given, subtracts that row's own shift-null value instead of the
    constant 0.5 -- unlike a constant shift, this changes the actual F/p
    values (per-session null varies), so it needs its own pass rather than
    reusing the 0.5-subtracted result."""
    cols = [factor_col, "area_value", metric_prefix] + ([shift_col] if shift_col else [])
    long = sub[cols].dropna().copy()
    long = long[long.area_value.isin(areas)]
    long = long[long[factor_col].isin([level_a, level_b])]
    long["value"] = long[metric_prefix] - (long[shift_col] if shift_col else 0.5)

    raw_p: dict[str, float] = {}
    for area in areas:
        a_vals = long[(long.area_value == area) & (long[factor_col] == level_a)]["value"].to_numpy()
        b_vals = long[(long.area_value == area) & (long[factor_col] == level_b)]["value"].to_numpy()
        if len(a_vals) > 0 and len(b_vals) > 0:
            try:
                _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        raw_p[area] = p

    testable = [a for a in areas if not np.isnan(raw_p[a])]
    posthoc_p: dict[str, float] = dict(raw_p)
    if testable:
        _, qvals, _, _ = multipletests([raw_p[a] for a in testable], method="fdr_bh")
        posthoc_p.update(dict(zip(testable, qvals)))

    if long[factor_col].nunique() < 2 or long["area_value"].nunique() < 2:
        return float("nan"), float("nan"), posthoc_p
    try:
        model = smf.ols(f"value ~ C({factor_col}) * C(area_value)", data=long).fit()
        table = anova_lm(model, typ=2)
        f_val = float(table.loc[f"C({factor_col})", "F"])
        p_val = float(table.loc[f"C({factor_col})", "PR(>F)"])
    except Exception:
        f_val, p_val = float("nan"), float("nan")
    return f_val, p_val, posthoc_p


def cohort_anova_and_posthoc(sub: pd.DataFrame, areas: list[str], metric_prefix: str,
                              shift_col: str | None = None) -> tuple[float, float, dict[str, float]]:
    """Thin cohort-specific wrapper around `factor_anova_and_posthoc` --
    kept so every existing call site (unchanged) reads the same as before."""
    return factor_anova_and_posthoc(sub, areas, metric_prefix, "reward_group", "R+", "R-", shift_col)


def compute_shared_ylim(df: pd.DataFrame, areas: list[str], columns: list[tuple[str, str, str]],
                         rows: list[tuple[str, tuple, str]], has_shift_null: bool, pad_frac: float = 0.15) -> tuple[float, float]:
    """Tailor the y-range to the actual data (user request 2026-09-13:
    "tailor the range of the y-axis to be close to the data") rather than a
    fixed constant -- computed once from every cohort/row/column combo
    (including the shift-null reference points, when present, so they never
    clip) so the per-cohort and comparison figures for this tag all share
    the same scale (still directly comparable to each other), with padding
    and a guaranteed sliver of room around the chance line at 0."""
    los, his = [], []
    for cohort in ("R+", "R-"):
        sub_cohort = df[df.reward_group == cohort]
        for metric_prefix, _, _ in rows:
            for ctype, cval, _ in columns:
                sub = sub_cohort[(sub_cohort.condition_type == ctype) & (sub_cohort.condition_value == cval)]
                mean, sem = area_paired_diff(sub, areas, metric_prefix, has_shift_null)
                valid = ~np.isnan(mean)
                if valid.any():
                    los.append(np.nanmin((mean - sem)[valid]))
                    his.append(np.nanmax((mean + sem)[valid]))
                if has_shift_null and f"{metric_prefix}_shift" in sub.columns:
                    nmean, nsem = area_shift_ref(sub, areas, metric_prefix)
                    nvalid = ~np.isnan(nmean)
                    if nvalid.any():
                        los.append(np.nanmin((nmean - nsem)[nvalid]))
                        his.append(np.nanmax((nmean + nsem)[nvalid]))
    if not los:
        return YLIM_DIFF
    lo, hi = min(los), max(his)
    pad = pad_frac * (hi - lo) if hi > lo else 0.02
    lo, hi = lo - pad, hi + pad
    lo, hi = min(lo, -0.01), max(hi, 0.02)  # always show a sliver on both sides of the chance line
    return (lo, hi)


def style_ax(ax, areas: list[str], colors: dict[str, str], title: str, ylim: tuple[float, float],
             ref_line: float = 0.0, ylabel: str = "balanced accuracy - chance"):
    """`ref_line`/`ylabel` added 2026-09-15 (user request: raw-accuracy and
    above-control figure variants alongside the original diff-from-chance
    one) -- defaults reproduce the original behavior exactly for every
    existing caller."""
    ax.axhline(ref_line, color="#888888", lw=1, linestyle=":", zorder=0)
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas, rotation=60, ha="right", fontsize=6.5)
    for tick, area in zip(ax.get_xticklabels(), areas):
        tick.set_color(colors[area])
    ax.set_ylim(*ylim)
    # 2026-09-24 (user: "make the figures non-square, slightly rectangular
    # to account for large number of areas on the x-axis"): square (aspect
    # 1) unchanged for area_group/whole_brain-sized lists (<= AREA_COUNT_
    # SQUARE_THRESHOLD areas), widened proportionally for area_acronym_
    # custom's much longer area list -- paired with `area_fig_width_mult`
    # below, which grows each subplot's actual figure-width allocation by
    # the same ratio, so the wider box has real room instead of just
    # stretching into whitespace.
    ax.set_box_aspect(min(1.0, AREA_COUNT_SQUARE_THRESHOLD / len(areas)) if areas else 1.0)
    ax.set_title(title, fontsize=8.5, pad=8)
    ax.set_xlabel("area", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)


def area_above_chance_stats(sub: pd.DataFrame, areas: list[str], metric_prefix: str, use_shift_null: bool,
                             allow_theoretical_fallback: bool = True) -> dict[str, tuple[float, float]]:
    """Per-area test of whether decoding is above chance/null (user request
    2026-09-16: "On figures comparing area-wise decoding vs null, plot stat
    annotations to test whether areas are above chance (e.g. anova against
    0) and post-hoc test per area (MWU)"). A one-sample t-test of
    (real - chance) against 0 is mathematically a one-way ANOVA with a
    single group tested against a fixed value (F = t^2) -- implemented
    directly as a one-sample t-test, alongside Mann-Whitney U comparing the
    area's real window-values against its own null window-values as two
    unpaired samples (same session-level-independence simplification
    `cohort_anova_and_posthoc` already uses for R+/R-, not mouse-block
    permutation) -- this project's usual parametric+nonparametric pair
    (`ssl_rplus_rminus_test_pair.md`), applied here to "above chance"
    instead of "R+ vs R-". FDR (Benjamini-Hochberg) across the areas in one
    panel, same convention as `cohort_anova_and_posthoc`. When
    `use_shift_null` and the shift-null column is actually present, that's
    the null; otherwise either falls back to theoretical 0.5
    (`allow_theoretical_fallback=True`, matching `area_paired_diff`) or
    returns NaN for that area (`False`, matching `area_above_control_diff`
    -- "above control" is meaningless without a real control). Returns
    `{area: (one_sample_t_q, mwu_q)}`."""
    shift_col_name = f"{metric_prefix}_shift"
    raw_p_t, raw_p_mwu = {}, {}
    for area in areas:
        a_sub = sub[sub.area_value == area]
        real_vals = a_sub[metric_prefix].dropna().to_numpy()
        have_shift = use_shift_null and shift_col_name in a_sub.columns and a_sub[shift_col_name].notna().any()
        if have_shift:
            paired = (a_sub[metric_prefix] - a_sub[shift_col_name]).dropna().to_numpy()
            null_vals = a_sub[shift_col_name].dropna().to_numpy()
        elif allow_theoretical_fallback:
            paired = (a_sub[metric_prefix] - 0.5).dropna().to_numpy()
            null_vals = np.full(len(real_vals), 0.5)
        else:
            raw_p_t[area], raw_p_mwu[area] = float("nan"), float("nan")
            continue
        raw_p_t[area] = float(ttest_1samp(paired, 0.0).pvalue) if len(paired) > 1 else float("nan")
        if len(real_vals) > 0 and len(null_vals) > 0:
            try:
                raw_p_mwu[area] = float(mannwhitneyu(real_vals, null_vals, alternative="two-sided").pvalue)
            except ValueError:
                raw_p_mwu[area] = float("nan")
        else:
            raw_p_mwu[area] = float("nan")

    def _fdr(raw: dict[str, float]) -> dict[str, float]:
        testable = [a for a in areas if not np.isnan(raw[a])]
        out = dict(raw)
        if testable:
            _, qvals, _, _ = multipletests([raw[a] for a in testable], method="fdr_bh")
            out.update(dict(zip(testable, qvals)))
        return out

    q_t, q_mwu = _fdr(raw_p_t), _fdr(raw_p_mwu)
    return {area: (q_t[area], q_mwu[area]) for area in areas}


def plot_group_grid(sub_group: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                     rows: list[tuple[str, tuple, str]], label: str, tag: str, out_path: Path, ylim: tuple[float, float],
                     has_shift_null: bool, null_kind: str | None = "shift"):
    """Per-cohort grid (`label`='R+' or 'R-', `sub_group` pre-filtered to
    that cohort) -- also reused, unchanged, for the pooled-cohort 'combined'
    variant (`label`='R+ & R- combined', `sub_group`=both cohorts together,
    i.e. every session treated as one group regardless of reward_group) --
    performance-state ('perfstate' decode target) figures only, per user
    request 2026-09-13."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(area_fig_width_mult(areas) * len(columns), 3.6 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, cval, col_label) in enumerate(columns):
            ax = axes[r][c]
            sub = sub_group[(sub_group.condition_type == ctype) & (sub_group.condition_value == cval)]
            show_null_ref = has_shift_null and f"{metric_prefix}_shift" in sub.columns and sub[f"{metric_prefix}_shift"].notna().any()
            diff_mean, diff_sem = area_paired_diff(sub, areas, metric_prefix, show_null_ref)
            real_off = -0.1 if show_null_ref else 0.0
            for i, area in enumerate(areas):
                ax.errorbar(i + real_off, diff_mean[i], yerr=diff_sem[i], fmt="o", color=colors[area], markersize=4, capsize=2)
            if show_null_ref:
                # Grey reference: where the empirical linear-shift null itself
                # sits relative to theoretical chance (0.5) -- per user
                # request 2026-09-13 to plot the shift null on this figure.
                null_mean, null_sem = area_shift_ref(sub, areas, metric_prefix)
                for i, area in enumerate(areas):
                    ax.errorbar(i + 0.1, null_mean[i], yerr=null_sem[i], fmt="s", color="#888888",
                                markersize=3, capsize=2, alpha=0.75)
            # Above-chance stat annotations (user request 2026-09-16): '*'
            # = MWU q<0.05 (real vs null, per area), '^' = one-sample
            # t-test / ANOVA-against-0 q<0.05 (real-minus-chance per area).
            stats_q = area_above_chance_stats(sub, areas, metric_prefix, show_null_ref)
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                q_t, q_mwu = stats_q[area]
                marker = ("*" if (not np.isnan(q_mwu) and q_mwu < 0.05) else "") + \
                         ("^" if (not np.isnan(q_t) and q_t < 0.05) else "")
                if marker:
                    top = diff_mean[i] + diff_sem[i] if not np.isnan(diff_mean[i]) else 0.0
                    ax.text(i + real_off, min(top + 0.04 * y_span, ylim[1] - 0.02 * y_span), marker,
                            ha="center", va="bottom", fontsize=9, color="black")
            n_sessions = sub["session_id"].nunique()
            chance_note = f" (chance = {null_kind} null)" if show_null_ref else ""
            style_ax(ax, areas, colors, f"{row_label} | {col_label}{chance_note}\n(n={n_sessions} sessions)", ylim)
    fig.suptitle(f"{tag} -- {label} -- window quantification, decoding minus chance "
                 f"('*'=MWU q<.05 vs null, '^'=one-sample-t/ANOVA-vs-0 q<.05, FDR per panel)", fontsize=11)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_factor_comparison_grid(df: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                                 rows: list[tuple[str, tuple, str]], tag: str, out_path: Path, ylim: tuple[float, float],
                                 factor_col: str = "reward_group", level_a: str = "R+", level_b: str = "R-",
                                 marker_a: str = "o", marker_b: str = "^", suptitle_label: str = "R+ vs R-",
                                 edge_a: str | None = None, edge_b: str = "black", edge_width: float = 0.6):
    """Generalized `plot_comparison_grid` (2026-09-18, user request "make
    within-cohort across-condition ... and within-cohort across-learning-
    stage ... anova/posthoc across area figures, as two other ways to
    compare the data"): same rows(metric) x columns(condition) grid, same
    ANOVA + per-area MWU post-hoc, but the compared FACTOR is now
    parametrized instead of hardcoded to `reward_group`/R+/R- -- e.g.
    `factor_col='day_stage'`, `level_a='learning'`, `level_b='expert'` for
    the across-stage comparison (see `plot_comparison_grid` below for the
    reward_group-specific wrapper every existing caller still uses
    unchanged, and `plot_within_cohort_condition_grid` for the
    across-CONDITION comparison, which needs a different grid axis
    entirely since the condition values themselves become the compared
    factor rather than a fixed column).

    `edge_a`/`edge_b` (2026-09-24, user: "change circles and triangles by
    shades of colors: R+ are the rplus color and R- are rminus color")
    let a caller override the marker edge color per level -- the
    reward_group wrapper below passes the R+/R- `COHORT_COLOR`s and the
    same circle shape for both levels, so cohort is now shown by edge
    shading rather than shape, while the marker FACE stays the area
    color. Non-cohort callers (day_stage) are unaffected: default
    `edge_a=None` ('none', i.e. no edge) / `edge_b='black'` keeps their
    original shape-only distinction."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(area_fig_width_mult(areas) * len(columns), 4.2 * len(rows)),
                              squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, cval, col_label) in enumerate(columns):
            ax = axes[r][c]
            sub = df[(df.condition_type == ctype) & (df.condition_value == cval)]
            n_parts = []
            top_y = {area: -np.inf for area in areas}
            for level, marker, edge, off in ((level_a, marker_a, edge_a, -0.12), (level_b, marker_b, edge_b, 0.12)):
                lsub = sub[sub[factor_col] == level]
                mean, sem = area_means(lsub, areas, metric_prefix)
                diff = mean - 0.5
                for i, area in enumerate(areas):
                    ax.errorbar(i + off, diff[i], yerr=sem[i], fmt=marker, color=colors[area],
                                markersize=4, capsize=2, markeredgecolor=edge or "none", markeredgewidth=edge_width)
                    if not np.isnan(diff[i]):
                        top_y[area] = max(top_y[area], diff[i] + sem[i])
                n_parts.append(f"{level} n={lsub['session_id'].nunique()}")

            # Parametric (two-way ANOVA, factor main effect) + non-parametric
            # post-hoc (per-area Mann-Whitney U), per user request 2026-09-13
            # (cohort) / 2026-09-18 (generalized) -- see
            # `factor_anova_and_posthoc` docstring.
            f_val, p_val, posthoc_q = factor_anova_and_posthoc(sub, areas, metric_prefix, factor_col, level_a, level_b)
            anova_note = f"ANOVA {factor_col}: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else f"ANOVA {factor_col}: n/a"
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                q = posthoc_q.get(area, float("nan"))
                if not np.isnan(q) and q < 0.05 and np.isfinite(top_y[area]):
                    ax.text(i, min(top_y[area] + 0.03 * y_span, ylim[1] - 0.02 * y_span), "*",
                            ha="center", va="bottom", fontsize=11, color="black")

            style_ax(ax, areas, colors, f"{row_label} | {col_label}\n({', '.join(n_parts)})\n{anova_note}", ylim)
    fig.suptitle(f"{tag} -- {suptitle_label} -- window quantification, decoding minus chance", fontsize=12)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_comparison_grid(df: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                          rows: list[tuple[str, tuple, str]], tag: str, out_path: Path, ylim: tuple[float, float]):
    """Thin reward_group-specific wrapper around `plot_factor_comparison_grid`
    -- kept so every existing call site (unchanged) reads the same as before.
    2026-09-24: cohort is now shown as a `COHORT_COLOR`-shaded marker edge
    (same circle shape for both R+/R-) instead of circle-vs-triangle."""
    plot_factor_comparison_grid(df, areas, colors, columns, rows, tag, out_path, ylim,
                                 "reward_group", "R+", "R-", "o", "o", "R+ vs R-",
                                 COHORT_COLOR["R+"], COHORT_COLOR["R-"], 1.0)


def plot_within_cohort_condition_grid(df_cohort: pd.DataFrame, areas: list[str], colors: dict[str, str],
                                       condition_pairs: list[tuple[str, str, str, str]], rows: list[tuple[str, tuple, str]],
                                       cohort_label: str, tag: str, out_path: Path, ylim: tuple[float, float]):
    """Within-cohort, across-CONDITION comparison (user request 2026-09-18):
    `df_cohort` is pre-filtered to one cohort; `condition_pairs` is a list
    of (condition_type, value_a, value_b, column_label), e.g. `('half',
    'first', 'second', 'half: first vs second')` -- unlike
    `plot_factor_comparison_grid`, the compared factor (`condition_value`)
    is ALSO what selects the data (there's no separate fixed "column" to
    hold still), so this gets its own grid: rows=metric, columns=condition
    pairs, comparing `value_a` vs `value_b` (same ANOVA/MWU machinery,
    `factor_col='condition_value'`) at each area, within this one cohort.
    Both values share one marker shape (circle); `value_a` ('first') is
    plotted in a `lighten()`-desaturated version of the area color,
    `value_b` ('second') at full saturation -- color alone (not shape)
    distinguishes the two (user request 2026-09-22: "Instead of circles
    and triangles for first and second half, do colored for second and
    less saturated version of that color for first")."""
    fig, axes = plt.subplots(len(rows), len(condition_pairs), figsize=(area_fig_width_mult(areas) * len(condition_pairs), 4.2 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, val_a, val_b, col_label) in enumerate(condition_pairs):
            ax = axes[r][c]
            sub = df_cohort[df_cohort.condition_type == ctype]
            n_parts = []
            top_y = {area: -np.inf for area in areas}
            for val, off in ((val_a, -0.12), (val_b, 0.12)):
                vsub = sub[sub.condition_value == val]
                mean, sem = area_means(vsub, areas, metric_prefix)
                diff = mean - 0.5
                for i, area in enumerate(areas):
                    marker_color = colors[area] if val == val_b else lighten(colors[area])
                    ax.errorbar(i + off, diff[i], yerr=sem[i], fmt="o", color=marker_color,
                                markersize=4, capsize=2, markeredgecolor="black", markeredgewidth=0.5)
                    if not np.isnan(diff[i]):
                        top_y[area] = max(top_y[area], diff[i] + sem[i])
                n_parts.append(f"{val} n={vsub['session_id'].nunique()}")

            f_val, p_val, posthoc_q = factor_anova_and_posthoc(sub, areas, metric_prefix, "condition_value", val_a, val_b)
            anova_note = f"ANOVA condition: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else "ANOVA condition: n/a"
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                q = posthoc_q.get(area, float("nan"))
                if not np.isnan(q) and q < 0.05 and np.isfinite(top_y[area]):
                    ax.text(i, min(top_y[area] + 0.03 * y_span, ylim[1] - 0.02 * y_span), "*",
                            ha="center", va="bottom", fontsize=11, color="black")

            style_ax(ax, areas, colors, f"{row_label} | {col_label}\n({', '.join(n_parts)})\n{anova_note}", ylim)
    fig.suptitle(f"{tag} -- {cohort_label}, across condition -- window quantification, decoding minus chance", fontsize=12)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_condition_grid_cohorts_sidebyside(df: pd.DataFrame, areas: list[str], colors: dict[str, str],
                                            condition_pairs: list[tuple[str, str, str, str]], rows: list[tuple[str, tuple, str]],
                                            tag: str, out_path: Path, ylim: tuple[float, float]):
    """Same across-CONDITION comparison as `plot_within_cohort_condition_grid`
    (same ANOVA/MWU-posthoc machinery, run separately per cohort), but both
    cohorts placed as adjacent subplots in ONE figure instead of two
    separate `_plus_condition`/`_minus_condition` files (user request
    2026-09-21: "Plot the two cohorts side by side as subplots"). Columns
    are (condition pair x cohort): for each condition pair, the R+ subplot
    is immediately followed by the R- subplot."""
    n_cols = len(condition_pairs) * 2
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(area_fig_width_mult(areas) * n_cols, 4.2 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, val_a, val_b, col_label) in enumerate(condition_pairs):
            for ci, cohort in enumerate(("R+", "R-")):
                ax = axes[r][c * 2 + ci]
                sub = df[(df.reward_group == cohort) & (df.condition_type == ctype)]
                n_parts = []
                top_y = {area: -np.inf for area in areas}
                for val, off in ((val_a, -0.12), (val_b, 0.12)):
                    vsub = sub[sub.condition_value == val]
                    mean, sem = area_means(vsub, areas, metric_prefix)
                    diff = mean - 0.5
                    for i, area in enumerate(areas):
                        marker_color = colors[area] if val == val_b else lighten(colors[area])
                        ax.errorbar(i + off, diff[i], yerr=sem[i], fmt="o", color=marker_color,
                                    markersize=4, capsize=2, markeredgecolor="black", markeredgewidth=0.5)
                        if not np.isnan(diff[i]):
                            top_y[area] = max(top_y[area], diff[i] + sem[i])
                    n_parts.append(f"{val} n={vsub['session_id'].nunique()}")

                f_val, p_val, posthoc_q = factor_anova_and_posthoc(sub, areas, metric_prefix, "condition_value", val_a, val_b)
                anova_note = f"ANOVA condition: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else "ANOVA condition: n/a"
                y_span = ylim[1] - ylim[0]
                for i, area in enumerate(areas):
                    q = posthoc_q.get(area, float("nan"))
                    if not np.isnan(q) and q < 0.05 and np.isfinite(top_y[area]):
                        ax.text(i, min(top_y[area] + 0.03 * y_span, ylim[1] - 0.02 * y_span), "*",
                                ha="center", va="bottom", fontsize=11, color="black")

                style_ax(ax, areas, colors, f"{cohort} | {row_label} | {col_label}\n({', '.join(n_parts)})\n{anova_note}", ylim)
    fig.suptitle(f"{tag} -- across condition, cohorts side by side -- window quantification, decoding minus chance", fontsize=12)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_group_grid_raw(sub_group: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                         rows: list[tuple[str, tuple, str]], label: str, tag: str, out_path: Path, ylim: tuple[float, float],
                         null_kind: str | None = "shift"):
    """Raw-accuracy analog of `plot_group_grid` (user request 2026-09-15:
    "plot the equivalent figures with raw decoding accuracy") -- same
    layout, but the colored dot is the area's raw balanced accuracy (no
    chance/control subtraction) and the grey square (when shift-null
    exists) is the null's own raw accuracy, at the same small offset."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(area_fig_width_mult(areas) * len(columns), 3.6 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, cval, col_label) in enumerate(columns):
            ax = axes[r][c]
            sub = sub_group[(sub_group.condition_type == ctype) & (sub_group.condition_value == cval)]
            shift_col = f"{metric_prefix}_shift"
            show_null_ref = shift_col in sub.columns and sub[shift_col].notna().any()
            real_off = -0.1 if show_null_ref else 0.0
            mean, sem = area_means(sub, areas, metric_prefix)
            for i, area in enumerate(areas):
                ax.errorbar(i + real_off, mean[i], yerr=sem[i], fmt="o", color=colors[area], markersize=4, capsize=2)
            if show_null_ref:
                null_mean, null_sem = area_means(sub, areas, shift_col)
                for i, area in enumerate(areas):
                    ax.errorbar(i + 0.1, null_mean[i], yerr=null_sem[i], fmt="s", color="#888888",
                                markersize=3, capsize=2, alpha=0.75)
            n_sessions = sub["session_id"].nunique()
            chance_note = f" (grey = {null_kind}-null control)" if show_null_ref else ""
            style_ax(ax, areas, colors, f"{row_label} | {col_label}{chance_note}\n(n={n_sessions} sessions)", ylim,
                      ref_line=0.5, ylabel="balanced accuracy (raw)")
    fig.suptitle(f"{tag} -- {label} -- window quantification, raw decoding accuracy", fontsize=12)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_comparison_grid_raw(df: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                              rows: list[tuple[str, tuple, str]], tag: str, out_path: Path, ylim: tuple[float, float]):
    """Raw-accuracy analog of `plot_comparison_grid` -- same R+/R- layout
    and the same ANOVA/post-hoc test (a constant -0.5 shift changes neither
    the F-statistic nor the p-values, so the stats are identical to the
    diff-from-chance version; only the plotted y-values differ)."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(area_fig_width_mult(areas) * len(columns), 4.2 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, cval, col_label) in enumerate(columns):
            ax = axes[r][c]
            sub = df[(df.condition_type == ctype) & (df.condition_value == cval)]
            n_parts = []
            top_y = {area: -np.inf for area in areas}
            for cohort, off in (("R+", -0.12), ("R-", 0.12)):
                csub = sub[sub.reward_group == cohort]
                mean, sem = area_means(csub, areas, metric_prefix)
                for i, area in enumerate(areas):
                    ax.errorbar(i + off, mean[i], yerr=sem[i], fmt="o", color=colors[area],
                                markersize=4, capsize=2, markeredgecolor=COHORT_COLOR[cohort], markeredgewidth=1.0)
                    if not np.isnan(mean[i]):
                        top_y[area] = max(top_y[area], mean[i] + sem[i])
                n_parts.append(f"{cohort} n={csub['session_id'].nunique()}")

            f_val, p_val, posthoc_q = cohort_anova_and_posthoc(sub, areas, metric_prefix)
            anova_note = f"ANOVA cohort: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else "ANOVA cohort: n/a"
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                q = posthoc_q.get(area, float("nan"))
                if not np.isnan(q) and q < 0.05 and np.isfinite(top_y[area]):
                    ax.text(i, min(top_y[area] + 0.03 * y_span, ylim[1] - 0.02 * y_span), "*",
                            ha="center", va="bottom", fontsize=11, color="black")

            style_ax(ax, areas, colors, f"{row_label} | {col_label}\n({', '.join(n_parts)})\n{anova_note}", ylim,
                      ref_line=0.5, ylabel="balanced accuracy (raw)")
    fig.suptitle(f"{tag} -- R+ vs R- -- window quantification, raw decoding accuracy", fontsize=12)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_group_grid_abovecontrol(sub_group: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                                  rows: list[tuple[str, tuple, str]], label: str, tag: str, out_path: Path, ylim: tuple[float, float],
                                  null_kind: str | None = "shift"):
    """Above-control analog of `plot_group_grid` (user request 2026-09-15:
    "the decoding accuracy above the control" as its own explicit figure)
    -- always real-minus-shift-null (never falls back to 0.5), one dot per
    area, no separate grey null marker (the null is already what's being
    subtracted, so a second reference to it would be redundant here)."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(area_fig_width_mult(areas) * len(columns), 3.6 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, cval, col_label) in enumerate(columns):
            ax = axes[r][c]
            sub = sub_group[(sub_group.condition_type == ctype) & (sub_group.condition_value == cval)]
            diff_mean, diff_sem = area_above_control_diff(sub, areas, metric_prefix)
            for i, area in enumerate(areas):
                ax.errorbar(i, diff_mean[i], yerr=diff_sem[i], fmt="o", color=colors[area], markersize=4, capsize=2)
            stats_q = area_above_chance_stats(sub, areas, metric_prefix, use_shift_null=True, allow_theoretical_fallback=False)
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                q_t, q_mwu = stats_q[area]
                marker = ("*" if (not np.isnan(q_mwu) and q_mwu < 0.05) else "") + \
                         ("^" if (not np.isnan(q_t) and q_t < 0.05) else "")
                if marker:
                    top = diff_mean[i] + diff_sem[i] if not np.isnan(diff_mean[i]) else 0.0
                    ax.text(i, min(top + 0.04 * y_span, ylim[1] - 0.02 * y_span), marker,
                            ha="center", va="bottom", fontsize=9, color="black")
            n_sessions = sub["session_id"].nunique()
            style_ax(ax, areas, colors, f"{row_label} | {col_label}\n(n={n_sessions} sessions)", ylim,
                      ref_line=0.0, ylabel=f"balanced accuracy - {null_kind}-null control")
    fig.suptitle(f"{tag} -- {label} -- window quantification, decoding above its own {null_kind}-null control "
                 f"('*'=MWU q<.05 vs null, '^'=one-sample-t/ANOVA-vs-0 q<.05, FDR per panel)", fontsize=11)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_comparison_grid_abovecontrol(df: pd.DataFrame, areas: list[str], colors: dict[str, str], columns: list[tuple[str, str, str]],
                                       rows: list[tuple[str, tuple, str]], tag: str, out_path: Path, ylim: tuple[float, float],
                                       null_kind: str | None = "shift"):
    """Above-control analog of `plot_comparison_grid` -- the ANOVA/post-hoc
    here is NOT the same numbers as the diff-from-chance version: the
    per-session shift-null being subtracted varies session to session
    (unlike the constant 0.5), so `cohort_anova_and_posthoc`'s `shift_col`
    argument recomputes it properly."""
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(area_fig_width_mult(areas) * len(columns), 4.2 * len(rows)), squeeze=False, constrained_layout=True)
    for r, (metric_prefix, _, row_label) in enumerate(rows):
        for c, (ctype, cval, col_label) in enumerate(columns):
            ax = axes[r][c]
            sub = df[(df.condition_type == ctype) & (df.condition_value == cval)]
            shift_col = f"{metric_prefix}_shift"
            n_parts = []
            top_y = {area: -np.inf for area in areas}
            for cohort, off in (("R+", -0.12), ("R-", 0.12)):
                csub = sub[sub.reward_group == cohort]
                diff, sem = area_above_control_diff(csub, areas, metric_prefix)
                for i, area in enumerate(areas):
                    ax.errorbar(i + off, diff[i], yerr=sem[i], fmt="o", color=colors[area],
                                markersize=4, capsize=2, markeredgecolor=COHORT_COLOR[cohort], markeredgewidth=1.0)
                    if not np.isnan(diff[i]):
                        top_y[area] = max(top_y[area], diff[i] + sem[i])
                n_parts.append(f"{cohort} n={csub['session_id'].nunique()}")

            if shift_col in sub.columns:
                f_val, p_val, posthoc_q = cohort_anova_and_posthoc(sub, areas, metric_prefix, shift_col=shift_col)
            else:
                f_val, p_val, posthoc_q = float("nan"), float("nan"), {}
            anova_note = f"ANOVA cohort: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else f"ANOVA cohort: n/a (no {null_kind or 'shift'}-null)"
            y_span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                q = posthoc_q.get(area, float("nan"))
                if not np.isnan(q) and q < 0.05 and np.isfinite(top_y[area]):
                    ax.text(i, min(top_y[area] + 0.03 * y_span, ylim[1] - 0.02 * y_span), "*",
                            ha="center", va="bottom", fontsize=11, color="black")

            style_ax(ax, areas, colors, f"{row_label} | {col_label}\n({', '.join(n_parts)})\n{anova_note}", ylim,
                      ref_line=0.0, ylabel=f"balanced accuracy - {null_kind}-null control")
    fig.suptitle(f"{tag} -- R+ vs R- -- window quantification, decoding above its own {null_kind}-null control", fontsize=12)
    savefig_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# Within-cohort, across-CONDITION comparison (2026-09-18): only `half:
# first vs second` -- `perfstate` is deliberately excluded as a condition
# dimension throughout this script already (user request 2026-09-13:
# "ignore the perf state decoding per state, as it is redundant"), so this
# stays consistent with `CONDITION_COLUMNS` rather than reintroducing it
# just for this one new comparison.
CONDITION_PAIRS = [("half", "first", "second", "half: first vs second")]


def expert_tag_for(tag: str, alignment: str) -> str | None:
    """`hitmiss_stim_area_group` + alignment='stim' -> `hitmiss_stim_expert_
    area_group` -- matches `024_master_sweep.py`'s own `_suffix` tag
    convention (`_expert` inserted right after the alignment token).
    Returns None if `tag` doesn't contain the expected `_{alignment}_`
    marker (shouldn't happen for any real tag) or already IS an expert tag."""
    marker = f"_{alignment}_"
    if marker not in tag or "_expert" in tag:
        return None
    return tag.replace(marker, f"_{alignment}_expert_", 1)


def load_stage_merged(tag_learning: str, alignment: str) -> tuple[pd.DataFrame, list] | None:
    """Loads and concatenates the learning + expert versions of the same
    (decode_target, alignment, area scheme) tag, tagging each row's
    `day_stage` (2026-09-18, for the across-learning-stage comparison).
    Returns None if the expert counterpart's tag can't be derived, or its
    data isn't ready/computable yet."""
    tag_expert = expert_tag_for(tag_learning, alignment)
    if tag_expert is None:
        return None
    df_learn, rows, _, _ = load(tag_learning)
    if df_learn is None:
        return None
    df_expert, _, _, _ = load(tag_expert)
    if df_expert is None:
        return None
    df_learn = df_learn.copy()
    df_learn["day_stage"] = "learning"
    df_expert = df_expert.copy()
    df_expert["day_stage"] = "expert"
    return pd.concat([df_learn, df_expert], ignore_index=True), rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python 034_area_window_quant_grid.py <tag>")
        sys.exit(1)
    tag = sys.argv[1]

    df, rows, has_shift_null, null_kind = load(tag)
    if df is None:
        print(f"{tag}: missing results/bin-edges, or results file exists but no computed rows yet -- nothing to plot")
        return
    area_col = df["area_col"].iloc[0] if "area_col" in df.columns else "area_group"
    areas = common_areas(df, area_col)
    colors = get_area_color_map(areas)
    columns = present_columns(df)
    print(f"{tag}: {len(areas)} common areas, {len(columns)} conditions present: {[c[2] for c in columns]}, "
          f"rows: {[r[2] for r in rows]}, null control available: {has_shift_null} (kind={null_kind})")
    if not areas or not columns:
        print("  nothing to plot")
        return

    ylim = compute_shared_ylim(df, areas, columns, rows, has_shift_null)
    ylim_raw = compute_shared_ylim_raw(df, areas, columns, rows)
    ylim_abovecontrol = compute_shared_ylim_abovecontrol(df, areas, columns, rows)
    print(f"  y-range tailored to data: {ylim} (diff-from-chance), {ylim_raw} (raw), {ylim_abovecontrol} (above-control)")
    out_dir = fig_dir(tag)
    for cohort in ("R+", "R-"):
        suffix = "plus" if cohort == "R+" else "minus"
        plot_group_grid(df[df.reward_group == cohort], areas, colors, columns, rows, cohort, tag,
                         out_dir / f"034_{tag}_grid_{suffix}.png", ylim, has_shift_null, null_kind)
        plot_group_grid_raw(df[df.reward_group == cohort], areas, colors, columns, rows, cohort, tag,
                             out_dir / f"034_{tag}_grid_{suffix}_raw.png", ylim_raw, null_kind)
        plot_group_grid_abovecontrol(df[df.reward_group == cohort], areas, colors, columns, rows, cohort, tag,
                                      out_dir / f"034_{tag}_grid_{suffix}_abovecontrol.png", ylim_abovecontrol, null_kind)
    plot_comparison_grid(df, areas, colors, columns, rows, tag, out_dir / f"034_{tag}_grid_comparison.png", ylim)
    plot_comparison_grid_raw(df, areas, colors, columns, rows, tag, out_dir / f"034_{tag}_grid_comparison_raw.png", ylim_raw)
    plot_comparison_grid_abovecontrol(df, areas, colors, columns, rows, tag,
                                       out_dir / f"034_{tag}_grid_comparison_abovecontrol.png", ylim_abovecontrol, null_kind)

    # Combined-cohort variant (user request 2026-09-13), performance-state
    # decoding only: pools R+ and R- sessions into one group rather than
    # comparing them, for tags where decode_target=='perfstate' (decoding
    # performance state itself, not splitting some other decode by it).
    decode_target = df["decode_target"].iloc[0] if "decode_target" in df.columns else ("perfstate" if "perfstate" in tag else "")
    if decode_target == "perfstate":
        plot_group_grid(df, areas, colors, columns, rows, "R+ & R- combined", tag,
                         out_dir / f"034_{tag}_grid_combined.png", ylim, has_shift_null, null_kind)
        plot_group_grid_raw(df, areas, colors, columns, rows, "R+ & R- combined", tag,
                             out_dir / f"034_{tag}_grid_combined_raw.png", ylim_raw, null_kind)
        plot_group_grid_abovecontrol(df, areas, colors, columns, rows, "R+ & R- combined", tag,
                                      out_dir / f"034_{tag}_grid_combined_abovecontrol.png", ylim_abovecontrol, null_kind)

    # Within-cohort, across-CONDITION comparison (2026-09-18, user request:
    # "make ... within-cohort across-condition ... anova/posthoc across
    # area figures ... as two other ways to compare the data") -- diff-
    # from-chance only (not also raw/abovecontrol, scoped down to keep this
    # addition bounded); reuses `ylim` since it's the same diff-from-chance
    # scale as the other comparison grids for this tag.
    available_condition_types = set(df["condition_type"].dropna().unique())
    usable_pairs = [p for p in CONDITION_PAIRS if p[0] in available_condition_types]
    if usable_pairs:
        for cohort in ("R+", "R-"):
            suffix = "plus" if cohort == "R+" else "minus"
            plot_within_cohort_condition_grid(df[df.reward_group == cohort], areas, colors, usable_pairs, rows,
                                               cohort, tag, out_dir / f"034_{tag}_grid_{suffix}_condition.png", ylim)
        # both cohorts as adjacent subplots in one figure (user request 2026-09-21:
        # "Plot the two cohorts side by side as subplots")
        plot_condition_grid_cohorts_sidebyside(df, areas, colors, usable_pairs, rows, tag,
                                                out_dir / f"034_{tag}_grid_condition_cohorts.png", ylim)

    # Within-cohort, across-LEARNING-STAGE comparison (2026-09-18, same
    # user request) -- only for learning-stage tags whose expert-stage
    # sibling has usable data (auto-detected; silently skipped otherwise,
    # same "nothing to plot yet" convention as the rest of this script).
    alignment = df["alignment"].iloc[0] if "alignment" in df.columns else ("stim" if "stim" in tag else "lick")
    if "_expert" not in tag:
        stage_merged = load_stage_merged(tag, alignment)
        if stage_merged is not None:
            merged_df, merged_rows = stage_merged
            merged_areas = common_areas(merged_df, area_col)
            merged_columns = present_columns(merged_df)
            merged_ylim = compute_shared_ylim(merged_df, merged_areas, merged_columns, merged_rows,
                                               "shift_null_curves" in merged_df.columns and merged_df["shift_null_curves"].notna().any())
            tag_expert = expert_tag_for(tag, alignment)
            print(f"  stage-pair with {tag_expert}: {len(merged_areas)} common areas")
            for cohort in ("R+", "R-"):
                suffix = "plus" if cohort == "R+" else "minus"
                plot_factor_comparison_grid(
                    merged_df[merged_df.reward_group == cohort], merged_areas, get_area_color_map(merged_areas),
                    merged_columns, merged_rows, f"{tag} vs {tag_expert}",
                    out_dir / f"034_{tag}_grid_{suffix}_stage.png", merged_ylim,
                    "day_stage", "learning", "expert", "o", "s", f"{cohort}: learning vs expert")
        else:
            print(f"  no usable expert-stage sibling for {tag} yet -- skipping across-stage comparison")


if __name__ == "__main__":
    main()
