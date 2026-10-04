"""Behavior-correlation figures (revised 2026-09-12 per user instruction:
"Redo the correlation but with the delta p(lick)_whisker - p(lick)_false_alarm,
and the delta of that" -- replaces the earlier hit-rate/d-prime behavioral
measures with the simpler raw lick-probability-difference discriminability
index already used for the behavioral-state validation elsewhere in this
project (`017_behavioral_states_summary`): `delta_lick_prob = P(lick|whisker)
- P(lick|no_stim)`.

Pairing rule (locked 2026-09-12): always correlate a decoding measure with
a behavioral measure of the SAME grain -- whole-session behavior vs.
whole-session decoding; a given condition's delta-of-delta_lick_prob vs.
that same condition's delta decoding.

Behavioral metrics, all from trial tables:
  - whole-session: delta_lick_prob = P(lick|whisker) - P(lick|no_stim) over
    the whole session.
  - delta_half: delta_lick_prob in the first half minus the second half of
    whisker trials (same chronological median-split convention the decoding
    half-condition uses; no_stim trials assigned to a half by whether their
    start_time falls in that half's own whisker-trial time span).
  - delta_perfstate: delta_lick_prob in high-perf-state blocks minus
    low-perf-state blocks (same `prep_perfstate_trials_generic` block
    definition the decoding perfstate-condition uses; FA rate aggregated
    from each block's own `block_fa_rate`).

Decoding metrics (from the master-sweep whole-brain results): session_decoding
('whole' condition, window-mean accuracy) and the matching delta_half /
delta_perfstate decoding deltas.

One figure per combo, 3 rows (whole / delta_half / delta_perfstate) x
{pooled, percohort} = 2 figures per combo, one column (single X-metric now).
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
from ssl_timeresolved_decoding import (  # noqa: E402
    hitmiss_session_list, mean_in_window, prep_perfstate_trials_generic,
)
from ssl_bwm_trial_prep import prep_session  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402

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
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
POOLED_COLOR = "#2c5f5b"

COMBOS = [
    ("hitmiss_stim_whole_brain", "sensory", SENSORY_WINDOW, "Hit vs miss, start_time-aligned"),
    ("modality_stim_whole_brain", "sensory", SENSORY_WINDOW, "Modality, start_time-aligned"),
    ("modality_lick_whole_brain", "pre_lick", PRE_LICK_WINDOW, "Modality, first-lick-aligned (corrected)"),
    # Expert-stage (whole_brain) variants (user request 2026-09-14: "extend
    # that fuller treatment to all"). perfstate excluded throughout -- it
    # only ever computes 'whole', no half/perfstate sub-conditions to
    # compare against itself. area_group NOT added here: its results have
    # ~30 rows/session (one per area x condition, via `area_col`/
    # `area_value`), but this script's per-session merge (with the
    # behavioral table) assumes one row per session (true for whole_brain's
    # single pseudo-area) -- adding area_group tags as-is would silently
    # pool different areas together. Needs per-area faceting (like 032/034
    # already do) before extending here; flagged to the user 2026-09-14,
    # not yet resolved.
    ("hitmiss_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Hit vs miss, start_time-aligned (expert)"),
    ("modality_stim_expert_whole_brain", "sensory", SENSORY_WINDOW, "Modality, start_time-aligned (expert)"),
    ("modality_lick_expert_whole_brain", "pre_lick", PRE_LICK_WINDOW, "Modality, first-lick-aligned (corrected) (expert)"),
]


def curve_window_mean(curve, bin_labels: np.ndarray, window: tuple[float, float]) -> float:
    if curve is None:
        return float("nan")
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        return float("nan")
    return mean_in_window(arr, bin_labels, window)


def whole_session_behavior(trials: pd.DataFrame) -> dict:
    whisker = trials[trials["trial_type"] == "whisker_trial"]
    no_stim = trials[trials["trial_type"] == "no_stim_trial"]
    p_whisker = whisker["lick_flag"].mean() if len(whisker) else np.nan
    p_fa = no_stim["lick_flag"].mean() if len(no_stim) else np.nan
    return dict(delta_lick_prob=p_whisker - p_fa)


def half_behavior(trials: pd.DataFrame) -> dict:
    whisker = trials[trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
    no_stim = trials[trials["trial_type"] == "no_stim_trial"]
    if len(whisker) == 0:
        return dict(delta_lick_prob_half=np.nan)
    median_idx = len(whisker) // 2
    whisker["half"] = ["first"] * median_idx + ["second"] * (len(whisker) - median_idx)
    per_half = {}
    for half in ("first", "second"):
        wh = whisker[whisker["half"] == half]
        if len(wh) == 0:
            per_half[half] = np.nan
            continue
        t0, t1 = wh["start_time"].min(), wh["start_time"].max()
        ns = no_stim[(no_stim["start_time"] >= t0) & (no_stim["start_time"] <= t1)]
        p_whisker = wh["lick_flag"].mean()
        p_fa = ns["lick_flag"].mean() if len(ns) else np.nan
        per_half[half] = p_whisker - p_fa
    return dict(delta_lick_prob_half=per_half["first"] - per_half["second"])


def perfstate_behavior(dataset_root, session_id, sessions_tbl, trials_tbl) -> dict:
    whisker = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"])
    if whisker is None or len(whisker) == 0:
        return dict(delta_lick_prob_perfstate=np.nan)
    per_state = {}
    for state in ("high", "low"):
        st = whisker[whisker["perf_state"] == state]
        if len(st) == 0:
            per_state[state] = np.nan
            continue
        p_whisker = st["lick_flag"].mean()
        p_fa = st["block_fa_rate"].mean()
        per_state[state] = p_whisker - p_fa
    return dict(delta_lick_prob_perfstate=per_state["high"] - per_state["low"])


def build_behavior_table() -> pd.DataFrame:
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    rows = []
    for r in hm.itertuples():
        prepped = prep_session(dataset_root, r.session_id, sessions_tbl, trials_tbl)
        if prepped is None:
            continue
        trials = prepped["trials"]
        if len(trials[trials["trial_type"] == "whisker_trial"]) == 0:
            continue
        row = dict(session_id=r.session_id)
        row.update(whole_session_behavior(trials))
        row.update(half_behavior(trials))
        row.update(perfstate_behavior(dataset_root, r.session_id, sessions_tbl, trials_tbl))
        rows.append(row)
    return pd.DataFrame(rows)


def load_decode_metrics(tag: str, window: tuple[float, float]) -> pd.DataFrame | None:
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
    df["window_acc"] = df["real_curve"].apply(lambda c: curve_window_mean(c, bin_labels, window))

    whole = df[df.condition_type == "whole"][["session_id", "subject_id", "reward_group", "window_acc"]].rename(
        columns={"window_acc": "session_decoding"})

    half = df[df.condition_type == "half"].pivot_table(index="session_id", columns="condition_value", values="window_acc")
    half["delta_half"] = half.get("first", np.nan) - half.get("second", np.nan)

    perf = df[df.condition_type == "perfstate"].pivot_table(index="session_id", columns="condition_value", values="window_acc")
    perf["delta_perfstate"] = perf.get("high", np.nan) - perf.get("low", np.nan)

    out = whole.merge(half[["delta_half"]], on="session_id", how="left").merge(perf[["delta_perfstate"]], on="session_id", how="left")
    return out


def regress_and_plot(ax, x: np.ndarray, y: np.ndarray, color: str, label: str | None = None):
    """Standard correlation-figure style for this analysis (locked
    2026-09-12, user rule -- apply to every correlation figure going
    forward): scatter + OLS line + a shaded 95% CI band around the fit
    (analytic CI on the regression line, i.e. `t * s_err * sqrt(1/n +
    (x-xbar)^2/Sxx)`, not just a naked line)."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    ax.scatter(x, y, color=color, alpha=0.5, s=18, label=None)
    if n < 3:
        return f"{label + ': ' if label else ''}n={n} (too few)"
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    y_line = slope * xs + intercept
    ls = "-" if (label is None or p < 0.05) else "--"

    dof = n - 2
    if dof > 0:
        resid = y - (slope * x + intercept)
        s_err = np.sqrt(np.sum(resid**2) / dof)
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        if sxx > 0:
            se_pred = s_err * np.sqrt(1.0 / n + (xs - x_mean) ** 2 / sxx)
            t_val = stats.t.ppf(0.975, dof)
            ci = t_val * se_pred
            ax.fill_between(xs, y_line - ci, y_line + ci, color=color, alpha=0.08, lw=0)

    ax.plot(xs, y_line, color=color, lw=2, linestyle=ls, label=label)
    sig = "*" if p < 0.05 else ""
    return f"{label + ': ' if label else ''}n={n}, r={r:.2f}, p={p:.3g}{sig}"


# (row_label, y_col, y_label, x_col, x_label)
ROWS = [
    ("whole session", "session_decoding", "session decoding accuracy", "delta_lick_prob", "P(lick|whisker) - P(lick|FA)"),
    ("delta: first - second half", "delta_half", "delta decoding (first - second half)", "delta_lick_prob_half",
     "delta of [P(lick|whisker)-P(lick|FA)]\n(first - second half)"),
    ("delta: high - low perfstate", "delta_perfstate", "delta decoding (high - low perfstate)", "delta_lick_prob_perfstate",
     "delta of [P(lick|whisker)-P(lick|FA)]\n(high - low perfstate)"),
]


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    """`fig.savefig` occasionally hits a transient Windows file-lock
    (`OSError: [Errno 22] Invalid argument`, likely antivirus/cloud-sync
    briefly holding a newly-created file) that a simple retry always
    clears -- same helper as `032`/`043`/`046`, added here 2026-09-17 after
    it crashed this script outright with no retry."""
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
    print("Computing per-session delta_lick_prob (whole-session + delta_half + delta_perfstate)...")
    behavior = build_behavior_table()
    print(f"  {len(behavior)} sessions with behavior metrics")

    for tag, window_name, window, title_prefix in COMBOS:
        decode = load_decode_metrics(tag, window)
        if decode is None:
            print(f"{tag}: missing results/bin-edges, skipping")
            continue
        merged = decode.merge(behavior, on="session_id", how="inner")
        print(f"{tag}: {len(merged)} sessions merged")

        for suffix, per_cohort in (("pooled", False), ("percohort", True)):
            fig, axes = plt.subplots(1, len(ROWS), figsize=(4.6 * len(ROWS), 4.6))
            for ax, (row_label, y_col, y_label, x_col, x_label) in zip(axes, ROWS):
                x, y = merged[x_col].to_numpy(dtype=float), merged[y_col].to_numpy(dtype=float)
                if per_cohort:
                    texts = []
                    for cohort in ("R+", "R-"):
                        mask = (merged["reward_group"] == cohort).to_numpy()
                        texts.append(regress_and_plot(ax, x[mask], y[mask], COHORT_COLOR[cohort], cohort))
                    ax.legend(fontsize=8, frameon=False, loc="best")
                else:
                    texts = [regress_and_plot(ax, x, y, POOLED_COLOR)]
                ax.text(0.02, 0.98, "\n".join(texts), transform=ax.transAxes, fontsize=7.5, va="top", ha="left")
                ax.axhline(0, color="#888888", lw=1, linestyle=":", zorder=0)
                ax.axvline(0, color="#888888", lw=1, linestyle=":", zorder=0)
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_ylabel(y_label, fontsize=9.5)
                ax.set_title(row_label, fontsize=10)
                ax.set_box_aspect(1)
                ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"{title_prefix} -- {window_name} window {window}, decoding vs. delta_lick_prob ({suffix}, whole brain)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            out_path = fig_dir(tag) / f"030_{tag}_{window_name}_behavior_corr_{suffix}.png"
            savefig_retry(fig, out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_path.name}")


if __name__ == "__main__":
    main()
