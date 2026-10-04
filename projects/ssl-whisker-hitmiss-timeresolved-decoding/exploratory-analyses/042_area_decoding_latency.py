"""Per-area decoding-onset latency, area-based (`area_group`/`area_acronym_custom`)
tags only. Three independent latency methods, compared side by side (user
request 2026-09-15, second pass):

1. **Cluster-mass permutation** (as before) -- group-mean curve, mouse-block
   sign-flip null, earliest cluster-corrected-significant (p<ALPHA) cluster's
   onset bin.
2/3. **Sigmoid fit** -- a 4-parameter logistic (floor, ceiling, latency,
   slope) fit to the decoding-accuracy-vs-time curve via least squares;
   latency is the fitted inflection-point parameter directly (continuous,
   not bin-quantized).
4/5. **50%-of-peak threshold crossing** -- first bin where the curve reaches
   halfway from its own null-baseline level to its own peak.

Each of the fit-based methods (2-5) is computed twice: once on the
**group-mean curve** (one estimate) and once **per mouse** (one estimate per
mouse, averaged sessions within a mouse first), reported as mean +- SEM
across mice -- per user request ("do so for both mean curves and mouse
curves and report sem across mice").

**Null source, 2026-09-15 correction**: all methods above now reference the
**linear-shift null** (`shift_null_curves`, hitmiss/perfstate only), not the
label-shuffle null used in the first version of this script. The
label-shuffle null destroys the label sequence's own autocorrelation, so a
genuine pre-existing session-level bias (e.g. hit-vs-miss trials already
differing in engagement before the stimulus) still reads as "above null" at
literally the very first bin -- which is what produced the wall of 0ms
latencies the user flagged. The shift null preserves that bias (it's still
present in the shifted-label surrogate), so decoding above *this* null is
closer to a genuine stimulus-evoked signal. Falls back to the label-shuffle
null (with a printed warning) for tags where shift-null was never computed
(modality). See `034_area_window_quant_grid.py`'s `has_shift_null` handling
for the established precedent of this same fallback.

Unlike the first version of this script, **no post-stimulus clamping is
applied to any method** -- a method reporting a negative (pre-stimulus)
latency is itself the diagnostic signal we're now comparing across methods,
not something to hide.

RT / stim-time histogram overlay (user request 2026-09-15: "all
stim-aligned time-resolved figures must show the little histogram at the
bottom", then "For the lick time time-resolved figures, plot the stim-time,
the same way RTs are plotted"), fit-diagnostics figure only -- same
`build_rt_cache`/`build_stim_time_cache`/`add_rt_histogram` implementation
as `032_plot_decode_results.py`/`028_wholebrain_crossgen_timeresolved.py`
(duplicated per this project's established per-script self-containment
convention): stim-aligned tags get whisker-trial hit RT, lick-aligned tags
get the mirror-image stim-onset-time-relative-to-lick instead. Uses each
area's own contributing session set (not the whole tag's) since different
areas draw from different session subsets here.

`session_real_and_shuffled_curves`/`group_mouseblock_permutation_null`/
`cluster_permutation_test` (method 1's machinery) live in
`scripts/ssl_timeresolved_decoding.py` -- see that project's own docstring
history for why this, not `mne.stats`, is used for the permutation test.

Usage: python 042_area_decoding_latency.py <tag> [condition_type]
  tag matches `024_master_results_<tag>.parquet`, e.g. "hitmiss_stim_area_group".
  condition_type defaults to "whole" (session-level curve).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import (  # noqa: E402
    cluster_permutation_test, group_mouseblock_permutation_null, mean_in_window, prep_hitmiss_trials, add_first_lick_time,
)

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


def add_rt_histogram(ax, session_ids, rt_cache: dict[str, np.ndarray], xlim: tuple[float, float], ylim: tuple[float, float],
                      anchor_level: float = 0.5):
    """`anchor_level` (added for this script specifically): 032/028 fix
    their panels' y-range to (0.4, 1.0), so the 0.5 chance line is always
    inside it and the histogram reliably sits "just under" it. This
    script's diagnostic panels instead auto-scale to each area's own curve
    range, which usually does NOT include 0.5 at all (real curves here
    start around 0.52-0.57) -- anchoring to 0.5 would silently drop the
    histogram in most panels. Callers here pass each panel's own
    `baseline_level` (the shift-null level, already guaranteed visible)
    instead."""
    if rt_cache is None:
        return
    arrays = [rt_cache[s] for s in session_ids if s in rt_cache and len(rt_cache[s])]
    if not arrays:
        return
    rt = np.concatenate(arrays)
    rt = rt[(rt >= xlim[0]) & (rt <= xlim[1])]
    if len(rt) == 0:
        return
    frac_at_chance = (anchor_level - ylim[0]) / (ylim[1] - ylim[0])
    if frac_at_chance <= 0 or frac_at_chance >= 1:
        return
    ax_hist = ax.twinx()
    counts, edges = np.histogram(rt, bins=40, range=xlim)
    ax_hist.set_ylim(0, counts.max() / (frac_at_chance * 0.85) if counts.max() > 0 else 1)
    ax_hist.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                color="#888888", alpha=0.35, edgecolor="#333333", linewidth=1.2, zorder=0)
    ax_hist.set_yticks([])
    ax_hist.spines[["top", "left", "right"]].set_visible(False)
    ax.set_zorder(ax_hist.get_zorder() + 1)
    ax.patch.set_visible(False)

Z_THRESHOLD = 1.96
ALPHA = 0.05
N_PERM = 2000
MIN_MICE = 4  # minimum mice per (area, group) for any method here to be meaningful
SEED = 0
BASELINE_WINDOW = (-0.200, -0.010)  # same pre-stimulus window 034_area_window_quant_grid.py uses
MIN_RISE = 0.02  # minimum peak-minus-baseline accuracy rise required to attempt a sigmoid fit / half-peak crossing
SIGMOID_R2_MIN = 0.3  # below this, treat the sigmoid fit as unreliable and report NaN
HALFPEAK_FRAC = 0.5

EXCLUDED_OVERLAY_AREAS = {
    "Somatosensory-body", "Amygdala and hypothalamus", "Cortical subplate",
    "Insular areas", "Visual areas", "Olfactory areas", "Pons and medulla",
}  # "Pons and medulla" added 2026-09-16 for consistency with 032/034's
   # "exclude pons and medulla everywhere" -- this script was off-limits at
   # the time of that request (latency method undecided), now caught up.
# Same M:\ vs /mnt/lsens-analysis machine-independence issue as 034_area_window_quant_grid.py.
# 2026-09-28: default resolved via scripts/axel_bisi_paths.py (M: locally, /mnt/lsens-analysis on haas).
from axel_bisi_paths import axel_bisi_path  # noqa: E402

ALLEN_UTILS_PATH = os.environ.get("SSL_ALLEN_UTILS_PATH") or str(
    axel_bisi_path("Github", "ephys_utilities", "ephys_utilities", "allen_utils"))


def _import_allen_utils():
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


def area_order_for(area_col: str) -> list[str]:
    allen_utils = _import_allen_utils()
    if allen_utils is None:
        return []
    if area_col == "area_group":
        return allen_utils.get_area_group_custom_order()
    if area_col == "area_acronym_custom":
        return allen_utils.get_area_acronym_custom_order()
    return []


def fig_dir(tag: str) -> Path:
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


# ---------------------------------------------------------------------------
# Null source: shift-null preferred, label-shuffle fallback
# ---------------------------------------------------------------------------

def session_surrogate_curve(row) -> tuple[np.ndarray | None, str]:
    """One session's null-reference curve for every method below. Prefers
    the linear-shift null (`shift_null_curves`), averaged across its N_SHUF
    draws (they're already cached -- no added cost, and less noisy than the
    single first draw `024_master_sweep.py` stores as `surrogate_curve`).
    Falls back to the label-shuffle null (`null_curves`, first draw only,
    matching `024_master_sweep.py`'s own `surrogate_curve` convention) only
    for tags where shift-null was never computed (modality)."""
    shift = getattr(row, "shift_null_curves", None)
    if shift is not None:
        draws = [np.asarray(s, dtype=float) for s in shift if s is not None]
        if draws:
            return np.nanmean(np.stack(draws), axis=0), "shift"
    null_curves = getattr(row, "null_curves", None)
    if null_curves is not None:
        draws = [np.asarray(s, dtype=float) for s in null_curves if s is not None]
        if draws:
            return np.asarray(draws[0], dtype=float), "label_shuffle_fallback"
    return None, "none"


def mouse_level_curves(sub: pd.DataFrame) -> dict[str, dict]:
    """One (real_curve, surrogate_curve) pair per mouse -- averages a
    mouse's sessions together first (mouse is this project's established
    unit of independence, not session -- avoids a mouse with several
    expert-stage sessions dominating the per-mouse latency methods below)."""
    out = {}
    for subj, g in sub.groupby("subject_id"):
        reals = [np.asarray(c, dtype=float) for c in g["real_curve"] if c is not None]
        if not reals:
            continue
        surrs = [c for c in g["surrogate_curve_used"] if c is not None]
        out[subj] = dict(real_curve=np.nanmean(np.stack(reals), axis=0),
                          surrogate_curve=np.nanmean(np.stack(surrs), axis=0) if surrs else None,
                          n_sessions=len(g))
    return out


def baseline_level_of(curve: np.ndarray | None, bin_labels_s: np.ndarray) -> float:
    if curve is None:
        return 0.5
    val = mean_in_window(curve, bin_labels_s, BASELINE_WINDOW)
    return val if np.isfinite(val) else 0.5


def mean_sem(values) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    n = len(finite)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(finite))
    sem = float(np.std(finite) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


# ---------------------------------------------------------------------------
# Method 1: cluster-mass permutation (group-mean curve, mouse-block null)
# ---------------------------------------------------------------------------

def find_clusters(z: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Contiguous runs of `z > threshold` -> (start_idx, end_idx, cluster_mass=sum(z)).
    Same one-sided above-chance convention as `cluster_permutation_test`
    (two_sided=False)."""
    clusters, run_start, run_mass = [], None, 0.0
    for i, zi in enumerate(z):
        above = np.isfinite(zi) and zi > threshold
        if above:
            if run_start is None:
                run_start = i
            run_mass += zi
        else:
            if run_start is not None:
                clusters.append((run_start, i - 1, run_mass))
            run_start, run_mass = None, 0.0
    if run_start is not None:
        clusters.append((run_start, len(z) - 1, run_mass))
    return clusters


def cluster_pvalues(clusters: list[tuple[int, int, float]], null_max: np.ndarray) -> list[tuple[int, int, float, float]]:
    """Corrected p-value per cluster (not just the single max-mass cluster
    `cluster_permutation_test` itself reports): fraction of the
    mouse-block-permutation max-cluster-mass null distribution >= this
    cluster's own mass -- the standard multi-cluster extension of the same
    max-statistic correction."""
    return [(s, e, m, float((np.sum(null_max >= m) + 1) / (len(null_max) + 1))) for s, e, m in clusters]


def cluster_latency(sub: pd.DataFrame, bin_labels_ms: np.ndarray, rng: np.random.Generator) -> dict:
    sub = sub.dropna(subset=["real_curve", "surrogate_curve_used", "subject_id"])
    n_mice = sub["subject_id"].nunique()
    n_sessions = len(sub)
    if n_mice < MIN_MICE:
        return dict(latency_ms=np.nan, n_mice=n_mice, n_sessions=n_sessions,
                     clusters=[], reason=f"only {n_mice} mice (< {MIN_MICE})")

    session_records = [
        dict(subject_id=row.subject_id, real_curve=np.asarray(row.real_curve, dtype=float),
             surrogate_curve=np.asarray(row.surrogate_curve_used, dtype=float))
        for row in sub.itertuples()
    ]
    observed_mean_curve, null_curves = group_mouseblock_permutation_null(session_records, rng, n_perm=N_PERM)
    result = cluster_permutation_test(observed_mean_curve, null_curves, z_threshold=Z_THRESHOLD, two_sided=False)
    z = result["observed_z"]
    null_max = result["null_max_cluster_mass_dist"]
    clusters = cluster_pvalues(find_clusters(z, Z_THRESHOLD), null_max)
    sig = [(s, e, m, p) for s, e, m, p in clusters if p < ALPHA]
    earliest = min(sig, key=lambda c: c[0]) if sig else None
    latency_ms = float(bin_labels_ms[earliest[0]]) if earliest else np.nan
    latency_p = float(earliest[3]) if earliest else np.nan
    return dict(latency_ms=latency_ms, latency_p=latency_p, n_mice=n_mice, n_sessions=n_sessions,
                 clusters=[dict(start_ms=float(bin_labels_ms[s]), end_ms=float(bin_labels_ms[e]),
                                 mass=float(m), p_value=p) for s, e, m, p in clusters],
                 reason=None if sig else "no significant cluster")


# ---------------------------------------------------------------------------
# Methods 2-3: sigmoid fit (mean curve / per mouse)
# ---------------------------------------------------------------------------

def sigmoid(t, floor, ceiling, latency, slope):
    # curve_fit explores extreme slope*distance products while searching --
    # clip the exponent so those probe steps don't spam RuntimeWarnings;
    # doesn't change the converged fit (a clipped exp saturates to the same
    # 0/1 asymptote the true value would anyway).
    exponent = np.clip(-slope * (t - latency), -500, 500)
    return floor + (ceiling - floor) / (1.0 + np.exp(exponent))


def fit_sigmoid_latency(curve: np.ndarray, bin_labels_ms: np.ndarray) -> dict:
    y = np.asarray(curve, dtype=float)
    t = bin_labels_ms
    valid = np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < 8:
        return dict(latency_ms=np.nan, r2=np.nan, params=None)
    floor0, ceiling0 = float(np.nanmin(y)), float(np.nanmax(y))
    if ceiling0 - floor0 < MIN_RISE:
        return dict(latency_ms=np.nan, r2=np.nan, params=None)
    half = floor0 + 0.5 * (ceiling0 - floor0)
    above = np.where(y >= half)[0]
    t0_guess = float(t[above[0]]) if len(above) else float(np.median(t))
    p0 = [floor0, ceiling0, t0_guess, 0.1]
    bounds = ([0.0, 0.0, float(t.min()), 1e-3], [1.0, 1.0, float(t.max()), 5.0])
    try:
        popt, _ = curve_fit(sigmoid, t, y, p0=p0, bounds=bounds, maxfev=8000)
    except Exception:
        return dict(latency_ms=np.nan, r2=np.nan, params=None)
    pred = sigmoid(t, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if not np.isfinite(r2) or r2 < SIGMOID_R2_MIN:
        return dict(latency_ms=np.nan, r2=r2, params=popt)
    return dict(latency_ms=float(popt[2]), r2=r2, params=popt)


# ---------------------------------------------------------------------------
# Methods 4-5: 50%-of-peak threshold crossing (mean curve / per mouse)
# ---------------------------------------------------------------------------

def halfpeak_latency(curve: np.ndarray, bin_labels_ms: np.ndarray, baseline_level: float, frac: float = HALFPEAK_FRAC) -> dict:
    y = np.asarray(curve, dtype=float)
    t = bin_labels_ms
    valid = np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(y) == 0:
        return dict(latency_ms=np.nan, threshold=np.nan, peak=np.nan)
    peak = float(np.nanmax(y))
    if peak - baseline_level < MIN_RISE:
        return dict(latency_ms=np.nan, threshold=np.nan, peak=peak)
    threshold = baseline_level + frac * (peak - baseline_level)
    idx = np.where(y >= threshold)[0]
    if len(idx) == 0:
        return dict(latency_ms=np.nan, threshold=threshold, peak=peak)
    return dict(latency_ms=float(t[idx[0]]), threshold=threshold, peak=peak)


# ---------------------------------------------------------------------------
# Per (area, group) driver -- runs all 5 latency estimates
# ---------------------------------------------------------------------------

def compute_area_group(sub: pd.DataFrame, bin_labels_ms: np.ndarray, bin_labels_s: np.ndarray, rng: np.random.Generator) -> dict:
    mouse_curves = mouse_level_curves(sub)
    n_mice = len(mouse_curves)
    result = dict(n_mice=n_mice, n_sessions=len(sub), group_real_curve=None, group_surrogate_curve=None,
                  baseline_level=np.nan, peak_level=np.nan, sigmoid_params_mean=None,
                  session_ids=sorted(sub["session_id"].dropna().unique().tolist()))
    if n_mice < MIN_MICE:
        result.update(reason=f"only {n_mice} mice (< {MIN_MICE})", latency_cluster=np.nan, latency_cluster_p=np.nan, clusters=[],
                       latency_sigmoid_mean=np.nan, r2_sigmoid_mean=np.nan,
                       latency_sigmoid_mouse_mean=np.nan, latency_sigmoid_mouse_sem=np.nan, n_mice_sigmoid_fit=0,
                       latency_halfpeak_mean=np.nan,
                       latency_halfpeak_mouse_mean=np.nan, latency_halfpeak_mouse_sem=np.nan, n_mice_halfpeak_fit=0)
        return result

    real_stack = np.stack([m["real_curve"] for m in mouse_curves.values()])
    group_real = np.nanmean(real_stack, axis=0)
    surr_list = [m["surrogate_curve"] for m in mouse_curves.values() if m["surrogate_curve"] is not None]
    group_surrogate = np.nanmean(np.stack(surr_list), axis=0) if surr_list else None
    baseline_level = baseline_level_of(group_surrogate, bin_labels_s)
    peak_level = float(np.nanmax(group_real))

    cl = cluster_latency(sub, bin_labels_ms, rng)
    sig_mean = fit_sigmoid_latency(group_real, bin_labels_ms)
    sig_mouse_mean, sig_mouse_sem, n_fit_sig = mean_sem(
        [fit_sigmoid_latency(m["real_curve"], bin_labels_ms)["latency_ms"] for m in mouse_curves.values()])
    hp_mean = halfpeak_latency(group_real, bin_labels_ms, baseline_level)
    hp_mouse_vals = []
    for m in mouse_curves.values():
        bl_m = baseline_level_of(m["surrogate_curve"], bin_labels_s)
        hp_mouse_vals.append(halfpeak_latency(m["real_curve"], bin_labels_ms, bl_m)["latency_ms"])
    hp_mouse_mean, hp_mouse_sem, n_fit_hp = mean_sem(hp_mouse_vals)

    result.update(reason=None, latency_cluster=cl["latency_ms"], latency_cluster_p=cl["latency_p"], clusters=cl["clusters"],
                  latency_sigmoid_mean=sig_mean["latency_ms"], r2_sigmoid_mean=sig_mean["r2"],
                  latency_sigmoid_mouse_mean=sig_mouse_mean, latency_sigmoid_mouse_sem=sig_mouse_sem, n_mice_sigmoid_fit=n_fit_sig,
                  latency_halfpeak_mean=hp_mean["latency_ms"],
                  latency_halfpeak_mouse_mean=hp_mouse_mean, latency_halfpeak_mouse_sem=hp_mouse_sem, n_mice_halfpeak_fit=n_fit_hp,
                  group_real_curve=group_real, group_surrogate_curve=group_surrogate,
                  baseline_level=baseline_level, peak_level=peak_level, sigmoid_params_mean=sig_mean["params"])
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _value_labels(ax, x: np.ndarray, values: np.ndarray, abs_max: float, fontsize: float = 6,
                   extra: dict[float, str] | None = None):
    """`extra` (added for latency-figure annotation, user request 2026-09-16
    "Annotate"): optional area-keyed second line of text (e.g. the cluster
    method's own p-value) placed just beyond the ms-value label."""
    for xi, v in zip(x, values):
        if not np.isfinite(v):
            continue
        va = "bottom" if v >= 0 else "top"
        ax.text(xi, v + (0.02 * abs_max if v >= 0 else -0.02 * abs_max), f"{v:.0f}",
                 ha="center", va=va, fontsize=fontsize, rotation=90)
        if extra and xi in extra:
            ax.text(xi, v + (0.14 * abs_max if v >= 0 else -0.14 * abs_max), extra[xi],
                     ha="center", va=va, fontsize=fontsize - 1, rotation=90, color="#444444")


def _area_tick_labels(areas: list[str], n_mice: dict[str, int] | None) -> list[str]:
    """`n_mice` (added 2026-09-16, "Annotate"): appends each area's
    contributing-mice count to its tick label, e.g. "Midbrain (n=12)"."""
    if not n_mice:
        return list(areas)
    return [f"{a} (n={n_mice.get(a, 0)})" for a in areas]


def plot_latency_bars(ax, latencies: dict[str, float], colors: dict[str, str], title: str, ylabel: str,
                       n_mice: dict[str, int] | None = None, p_values: dict[str, float] | None = None):
    """Sorted-ascending (earliest to latest) single-method bar plot (used for
    the cohort-split cluster-method figure). `ax.set_box_aspect(1)` keeps
    every panel square per the project-wide strict-square rule. Handles
    negative (pre-stimulus) values: no clamping is applied anymore, so
    "earliest" can be < 0. `n_mice`/`p_values` (2026-09-16, "Annotate"):
    per-area mice count appended to the tick label, cluster p-value printed
    just past each bar's ms-value label."""
    present = {a: v for a, v in latencies.items() if np.isfinite(v)}
    ordered = sorted(present, key=lambda a: present[a])
    ax.set_box_aspect(1)
    if not ordered:
        ax.set_title(f"{title}\n(no area reached significance)", fontsize=9)
        ax.axis("off")
        return
    x = np.arange(len(ordered))
    values = np.array([present[a] for a in ordered])
    ax.bar(x, values, color=[colors[a] for a in ordered], width=0.7)
    abs_max = float(np.max(np.abs(values))) or 1.0
    extra = {xi: f"p={p_values[a]:.3f}" for xi, a in zip(x, ordered) if p_values and np.isfinite(p_values.get(a, np.nan))}
    _value_labels(ax, x, values, abs_max, extra=extra)
    ax.axhline(0, color="#888888", lw=0.8, linestyle=":", zorder=0)
    ax.set_ylim(min(0, values.min()) - 0.22 * abs_max, max(0, values.max()) + 0.22 * abs_max)
    ax.set_xticks(x)
    ax.set_xticklabels(_area_tick_labels(ordered, n_mice), rotation=80, ha="right", fontsize=6.5)
    for tick, area in zip(ax.get_xticklabels(), ordered):
        tick.set_color(colors[area])
    ax.set_ylabel(ylabel, fontsize=8.5)
    missing = sorted(set(latencies) - set(ordered))
    # A full name list here can run to 30+ areas and visually collide with
    # the panel above under constrained_layout -- collapse to a count once
    # it gets long (full names are always in the saved CSV/log).
    if missing:
        subtitle = f"\n({len(missing)} areas: no significant cluster)" if len(missing) > 8 \
            else f"\n(no significant cluster: {', '.join(missing)})"
    else:
        subtitle = ""
    ax.set_title(f"{title}{subtitle}", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_method_panel(ax, areas: list[str], values: dict[str, float], sems: dict[str, float] | None,
                       colors: dict[str, str], title: str, ylabel: str = "",
                       n_mice: dict[str, int] | None = None, p_values: dict[str, float] | None = None):
    """Sorted-ascending (earliest to latest) bar panel for the
    method-comparison grid (changed 2026-09-16, user request "Order areas
    from earliest to latest" -- previously a FIXED area order shared across
    all 5 panels, so the same area sat at the same x position in every
    panel; sorting each panel by its own method's values trades that
    cross-panel positional alignment for making each panel readable at a
    glance on its own, which is what was asked for). Areas with no finite
    value for this method (insufficient mice, no fit, no crossing) are
    appended after the sorted ones and drawn as a grey x-mark at y=0, so
    "no result" is never confused with a real (e.g. 0ms) value.
    `n_mice`/`p_values` (2026-09-16, "Annotate"): per-area mice count
    appended to the tick label; `p_values` (cluster panel only) printed
    just past each bar's ms-value label."""
    ax.set_box_aspect(1)
    present = {a: values[a] for a in areas if np.isfinite(values.get(a, np.nan))}
    missing = [a for a in areas if a not in present]
    ordered = sorted(present, key=lambda a: present[a]) + missing
    x = np.arange(len(ordered))
    vals = np.array([values.get(a, np.nan) for a in ordered], dtype=float)
    errs = np.array([sems.get(a, 0.0) if sems and np.isfinite(values.get(a, np.nan)) else 0.0 for a in ordered]) if sems else None
    finite = np.isfinite(vals)
    if finite.any():
        ax.bar(x[finite], vals[finite], yerr=(errs[finite] if errs is not None else None),
               color=[colors[a] for a, f in zip(ordered, finite) if f], width=0.7, capsize=3)
        abs_max = float(np.max(np.abs(vals[finite]) + (errs[finite] if errs is not None else 0))) or 1.0
        extra = {xi: f"p={p_values[a]:.3f}" for xi, a, f in zip(x, ordered, finite)
                 if f and p_values and np.isfinite(p_values.get(a, np.nan))}
        _value_labels(ax, x[finite], vals[finite], abs_max, extra=extra)
    else:
        abs_max = 1.0
    ax.plot(x[~finite], np.zeros((~finite).sum()), marker="x", color="#999999", linestyle="none", markersize=5)
    ax.axhline(0, color="#888888", lw=0.8, linestyle=":", zorder=0)
    lo = min(0, np.nanmin(vals[finite]) if finite.any() else 0) - 0.28 * abs_max
    hi = max(0, np.nanmax(vals[finite]) if finite.any() else 0) + 0.28 * abs_max
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels(_area_tick_labels(ordered, n_mice), rotation=80, ha="right", fontsize=6)
    for tick, area in zip(ax.get_xticklabels(), ordered):
        tick.set_color(colors[area])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_comparison_grid(areas: list[str], colors: dict[str, str], per_area: dict[str, dict], null_source: str,
                          out_path: Path, tag: str, condition_type: str, ylabel: str):
    panels = [
        ("latency_cluster", None, "n_mice", "latency_cluster_p", "Cluster permutation\n(group mean, mouse-block null)"),
        ("latency_sigmoid_mean", None, "n_mice", None, "Sigmoid fit\n(group mean curve)"),
        ("latency_sigmoid_mouse_mean", "latency_sigmoid_mouse_sem", "n_mice_sigmoid_fit", None, "Sigmoid fit\n(per mouse, mean +- SEM)"),
        ("latency_halfpeak_mean", None, "n_mice", None, "50%-of-peak\n(group mean curve)"),
        ("latency_halfpeak_mouse_mean", "latency_halfpeak_mouse_sem", "n_mice_halfpeak_fit", None, "50%-of-peak\n(per mouse, mean +- SEM)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5), constrained_layout=True)
    for i, (ax, (key, sem_key, n_key, p_key, label)) in enumerate(zip(axes.flat, panels)):
        values = {a: per_area[a][key] for a in areas}
        sems = {a: per_area[a][sem_key] for a in areas} if sem_key else None
        n_mice = {a: per_area[a][n_key] for a in areas}
        p_values = {a: per_area[a][p_key] for a in areas} if p_key else None
        plot_method_panel(ax, areas, values, sems, colors, label, ylabel=(ylabel if i % 3 == 0 else ""),
                           n_mice=n_mice, p_values=p_values)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{tag} -- {condition_type} -- decoding-onset latency, method comparison (null: {null_source})", fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fit_diagnostics(areas: list[str], per_area: dict[str, dict], colors: dict[str, str], bin_labels_ms: np.ndarray,
                          out_path: Path, tag: str, condition_type: str, null_source: str,
                          rt_cache: dict[str, np.ndarray] | None = None):
    """Small multiples, one square panel per area: real (mouse-balanced mean)
    curve, null-baseline reference curve, the fitted sigmoid overlay (dense,
    when the fit was reliable), and vertical markers for all three methods'
    latency estimates -- so a fit can be sanity-checked visually rather than
    trusted blindly (user request: "show relevant fitting when necessary").
    `rt_cache` (stim-aligned tags only) overlays each area's own whisker-hit
    RT histogram -- see module docstring. Panels ordered earliest-to-latest
    (2026-09-16, user request "Order areas from earliest to latest"): by
    cluster-permutation latency where available, falling back to the
    sigmoid-fit then 50%-of-peak group-mean latency for an area with no
    significant cluster, with areas that have no usable latency at all from
    any method pushed to the end in their original (atlas) order."""
    def _sort_key(area: str) -> float:
        d = per_area[area]
        for key in ("latency_cluster", "latency_sigmoid_mean", "latency_halfpeak_mean"):
            v = d.get(key, np.nan)
            if np.isfinite(v):
                return v
        return np.inf
    areas = sorted(areas, key=_sort_key)
    n = len(areas)
    ncols = int(np.ceil(np.sqrt(n))) or 1
    nrows = int(np.ceil(n / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), constrained_layout=True, squeeze=False)
    axes_flat = axes.flatten()
    t_dense = np.linspace(bin_labels_ms.min(), bin_labels_ms.max(), 400)
    for ax, area in zip(axes_flat, areas):
        ax.set_box_aspect(1)
        d = per_area[area]
        if d["group_real_curve"] is None:
            ax.set_title(f"{area}\n({d['reason']})", fontsize=7, color=colors[area])
            ax.axis("off")
            continue
        ax.plot(bin_labels_ms, d["group_real_curve"], color=colors[area], lw=1.7, zorder=3)
        if d["group_surrogate_curve"] is not None:
            ax.plot(bin_labels_ms, d["group_surrogate_curve"], color="#888888", lw=1.0, linestyle=":", zorder=2)
        if d["sigmoid_params_mean"] is not None and np.isfinite(d["latency_sigmoid_mean"]):
            ax.plot(t_dense, sigmoid(t_dense, *d["sigmoid_params_mean"]), color="black", lw=1.0, linestyle="--", alpha=0.8, zorder=2)
        xlim = (float(bin_labels_ms.min()), float(bin_labels_ms.max()))
        if rt_cache is not None:
            add_rt_histogram(ax, d["session_ids"], rt_cache, xlim, ax.get_ylim(), anchor_level=d["baseline_level"])
        if np.isfinite(d["latency_cluster"]):
            ax.axvline(d["latency_cluster"], color="#1f77b4", lw=1.3, linestyle="-", alpha=0.85)
        if np.isfinite(d["latency_sigmoid_mean"]):
            ax.axvline(d["latency_sigmoid_mean"], color="black", lw=1.3, linestyle="--", alpha=0.85)
        if np.isfinite(d["latency_halfpeak_mean"]):
            ax.axvline(d["latency_halfpeak_mean"], color="#d62728", lw=1.3, linestyle="-.", alpha=0.85)
        ax.axhline(d["baseline_level"], color="#888888", lw=0.7, linestyle=":", alpha=0.5)
        r2_note = f", r2={d['r2_sigmoid_mean']:.2f}" if np.isfinite(d.get("r2_sigmoid_mean", np.nan)) else ""
        p_note = f", p={d['latency_cluster_p']:.3f}" if np.isfinite(d.get("latency_cluster_p", np.nan)) else ""
        ax.set_title(f"{area} (n={d['n_mice']} mice{r2_note}{p_note})", fontsize=6.8, color=colors[area])
        ax.tick_params(labelsize=6)
        ax.set_xlim(bin_labels_ms.min(), bin_labels_ms.max())
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes_flat[len(areas):]:
        ax.axis("off")

    # A shared fig.legend(loc="outside upper center") collided with
    # fig.suptitle under constrained_layout (legend text rendered on top of,
    # not below, the title) -- putting the legend inside the first panel
    # instead sidesteps that interaction entirely, matching how 032/034
    # already place their own small in-panel legends.
    handles = [
        mlines.Line2D([0], [0], color=colors[areas[0]], lw=1.7, label="real curve (mouse-bal. mean)"),
        mlines.Line2D([0], [0], color="#888888", lw=1.0, linestyle=":", label=f"{null_source}-null (mouse-bal. mean)"),
        mlines.Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="sigmoid fit"),
        mlines.Line2D([0], [0], color="#1f77b4", lw=1.5, label="cluster-perm. latency"),
        mlines.Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="sigmoid-fit latency"),
        mlines.Line2D([0], [0], color="#d62728", lw=1.5, linestyle="-.", label="50%-of-peak latency"),
    ]
    axes_flat[0].legend(handles=handles, loc="lower right", fontsize=4.8, frameon=True, framealpha=0.9)
    fig.suptitle(f"{tag} -- {condition_type} -- fit diagnostics (null: {null_source})", fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 042_area_decoding_latency.py <tag> [condition_type]")
        sys.exit(1)
    tag = sys.argv[1]
    condition_type = sys.argv[2] if len(sys.argv) > 2 else "whole"

    partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
    edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
    if not partial_path.exists() or not edges_path.exists():
        print(f"{tag}: missing results/bin-edges file(s), nothing to compute")
        return

    df = pd.read_parquet(partial_path)
    df = df[(df["skipped_reason"].isna()) & (df["condition_type"] == condition_type)].copy()
    if len(df) == 0:
        print(f"{tag}: no computed '{condition_type}' rows yet")
        return

    area_col = df["area_col"].iloc[0] if "area_col" in df.columns else ("area_group" if "area_group" in tag else "area_acronym_custom")
    if area_col == "whole_brain":
        print(f"{tag}: whole_brain has no areas to rank, nothing to plot")
        return

    bin_edges = json.loads(edges_path.read_text())
    bin_labels_ms = np.array([e[1] * 1000 for e in bin_edges])
    bin_labels_s = np.array([e[1] for e in bin_edges])
    alignment = df["alignment"].iloc[0] if "alignment" in df.columns else ("stim" if "_stim_" in tag else "lick")

    # RT / stim-time histogram overlay (user request 2026-09-15).
    rt_cache = None
    if alignment == "stim":
        rt_cache = build_rt_cache(df["session_id"].unique())
        print(f"  RT histogram cache: {len(rt_cache)}/{df['session_id'].nunique()} sessions")
    elif alignment == "lick":
        rt_cache = build_stim_time_cache(df["session_id"].unique())
        print(f"  stim-time histogram cache: {len(rt_cache)}/{df['session_id'].nunique()} sessions")

    surrogate_curves, surrogate_sources = [], []
    for row in df.itertuples():
        c, src = session_surrogate_curve(row)
        surrogate_curves.append(c)
        surrogate_sources.append(src)
    df["surrogate_curve_used"] = surrogate_curves
    df["surrogate_source_used"] = surrogate_sources
    n_fallback = int((df["surrogate_source_used"] == "label_shuffle_fallback").sum())
    n_none = int((df["surrogate_source_used"] == "none").sum())
    null_source_label = "shift" if (df["surrogate_source_used"] == "shift").any() else "label_shuffle_fallback"
    if n_fallback:
        print(f"  NOTE: shift-null unavailable for {n_fallback}/{len(df)} rows -- falling back to label-shuffle null for those")
    if n_none:
        print(f"  WARNING: no usable null curve at all for {n_none}/{len(df)} rows -- those sessions are dropped")

    order = area_order_for(area_col)
    present_areas = set(df["area_value"].dropna().unique()) - EXCLUDED_OVERLAY_AREAS
    areas = [a for a in order if a in present_areas] or sorted(present_areas)
    colors = get_area_color_map(areas)
    print(f"{tag} ({condition_type}, {alignment}-aligned): {len(areas)} areas, null={null_source_label}, "
          f"z>{Z_THRESHOLD}, n_perm={N_PERM}, alpha={ALPHA}, min_mice={MIN_MICE}, sigmoid_r2_min={SIGMOID_R2_MIN}, "
          f"min_rise={MIN_RISE}, halfpeak_frac={HALFPEAK_FRAC}")

    rng = np.random.default_rng(SEED)
    per_group_area = {"R+": {}, "R-": {}, "combined": {}}
    records = []
    for group_name, group_df in (("R+", df[df.reward_group == "R+"]), ("R-", df[df.reward_group == "R-"]), ("combined", df)):
        for area in areas:
            res = compute_area_group(group_df[group_df.area_value == area], bin_labels_ms, bin_labels_s, rng)
            per_group_area[group_name][area] = res
            records.append(dict(
                tag=tag, condition_type=condition_type, group=group_name, area=area, null_source=null_source_label,
                n_mice=res["n_mice"], n_sessions=res["n_sessions"], reason=res["reason"],
                latency_cluster_ms=res["latency_cluster"], latency_cluster_p=res["latency_cluster_p"],
                latency_sigmoid_mean_ms=res["latency_sigmoid_mean"], r2_sigmoid_mean=res["r2_sigmoid_mean"],
                latency_sigmoid_mouse_mean_ms=res["latency_sigmoid_mouse_mean"],
                latency_sigmoid_mouse_sem_ms=res["latency_sigmoid_mouse_sem"], n_mice_sigmoid_fit=res["n_mice_sigmoid_fit"],
                latency_halfpeak_mean_ms=res["latency_halfpeak_mean"],
                latency_halfpeak_mouse_mean_ms=res["latency_halfpeak_mouse_mean"],
                latency_halfpeak_mouse_sem_ms=res["latency_halfpeak_mouse_sem"], n_mice_halfpeak_fit=res["n_mice_halfpeak_fit"],
                baseline_level=res["baseline_level"], peak_level=res["peak_level"],
                clusters_json=json.dumps(res["clusters"]),
                z_threshold=Z_THRESHOLD, n_perm=N_PERM, alpha=ALPHA, min_mice=MIN_MICE, seed=SEED,
                sigmoid_r2_min=SIGMOID_R2_MIN, min_rise=MIN_RISE, halfpeak_frac=HALFPEAK_FRAC,
                baseline_window=str(BASELINE_WINDOW)))
            print(f"  {group_name:>8} | {area:<28} n_mice={res['n_mice']:>2} "
                  f"cluster={res['latency_cluster']} sig_mean={res['latency_sigmoid_mean']} "
                  f"sig_mouse={res['latency_sigmoid_mouse_mean']}+-{res['latency_sigmoid_mouse_sem']} "
                  f"hp_mean={res['latency_halfpeak_mean']} hp_mouse={res['latency_halfpeak_mouse_mean']}+-{res['latency_halfpeak_mouse_sem']}"
                  + (f"  ({res['reason']})" if res["reason"] else ""))

    out_dir = fig_dir(tag)
    results_df = pd.DataFrame(records)
    results_csv = out_dir / f"042_{tag}_{condition_type}_area_latency.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"  saved {results_csv.name}")

    xlabel_note = "time from start_time (ms)" if alignment == "stim" else "time from first lick (ms, corrected)"
    ylabel = f"onset latency, ms ({xlabel_note}), vs {null_source_label}-null"

    # Cohort-split figure for the cluster method specifically -- information
    # (R+ vs R- split) not present in the pooled comparison grid below.
    cluster_latencies = {g: {a: per_group_area[g][a]["latency_cluster"] for a in areas} for g in ("R+", "R-", "combined")}
    cluster_n_mice = {g: {a: per_group_area[g][a]["n_mice"] for a in areas} for g in ("R+", "R-", "combined")}
    cluster_p_values = {g: {a: per_group_area[g][a]["latency_cluster_p"] for a in areas} for g in ("R+", "R-", "combined")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
    for ax, group_name, show_ylabel in zip(axes.flat, ("R+", "R-", "combined"), (True, False, True)):
        plot_latency_bars(ax, cluster_latencies[group_name], colors, f"{group_name}", ylabel if show_ylabel else "",
                           n_mice=cluster_n_mice[group_name], p_values=cluster_p_values[group_name])
    axes.flat[3].axis("off")
    fig.suptitle(f"{tag} -- {condition_type} -- cluster-permutation onset latency by area (vs {null_source_label}-null)", fontsize=12)
    fig.savefig(out_dir / f"042_{tag}_{condition_type}_latency_bycohort.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 042_{tag}_{condition_type}_latency_bycohort.png")

    # Primary ask: all 5 method variants, pooled (combined) group, one grid.
    plot_comparison_grid(areas, colors, per_group_area["combined"], null_source_label,
                          out_dir / f"042_{tag}_{condition_type}_method_comparison.png", tag, condition_type, ylabel)
    print(f"  saved 042_{tag}_{condition_type}_method_comparison.png")

    plot_fit_diagnostics(areas, per_group_area["combined"], colors, bin_labels_ms,
                          out_dir / f"042_{tag}_{condition_type}_fit_diagnostics.png", tag, condition_type, null_source_label,
                          rt_cache=rt_cache)
    print(f"  saved 042_{tag}_{condition_type}_fit_diagnostics.png")


if __name__ == "__main__":
    main()
