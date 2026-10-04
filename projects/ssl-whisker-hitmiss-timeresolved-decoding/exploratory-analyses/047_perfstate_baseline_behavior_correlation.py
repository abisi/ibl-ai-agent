"""Correlate `046_perfstate_baseline_decoding.py`'s per-mouse decoding
accuracy (above its own shift-null) against how behaviorally distinct the
mouse's high/low perf-states actually are (user request 2026-09-16:
"Correlate performance state decoding with difference in performance
(whisker hit rate, false alarm and difference of the two) per mouse across
states").

Three (high - low) behavioral gaps, computed from the SAME curve-based
perf-state labels `046` itself uses (`prep_perfstate_trials_curve`, all 3
trial types) -- NOT `037_perfstate_confound_and_behavior_correlations.py`'s
older block-median-split labels, and NOT that script's precomputed
`block_fa_rate` column (not produced by the curve-based function) --
false-alarm rate is computed directly here from `no_stim_trial` rows:
- `hit_rate_gap`: P(lick|whisker_trial)(high) - P(lick|whisker_trial)(low).
  Raw, not cohort-corrected -- same caveat `037` already documents: for R+
  sessions this is close to tautological with the reward-contingent
  variable that (indirectly, via the learning curve) drives the state
  split; for R- sessions it's a genuinely independent measure. Included
  anyway per the user's explicit request, caveat flagged not hidden.
- `fa_rate_gap`: P(lick|no_stim_trial)(high) - P(lick|no_stim_trial)(low).
  Independent of the split by construction.
- `delta_lick_prob_gap`: `hit_rate_gap - fa_rate_gap` (a simple hit-minus-
  false-alarm discriminability gap, per the user's "difference of the two").

Unit of analysis is the MOUSE, not the session (user's explicit "per
mouse" -- overriding this project's usual session-level default for
expert-stage analyses, [[ssl_stats_unit_of_analysis]], which doesn't apply
here anyway since this is learning-stage/day-0 only): sessions are
averaged within `subject_id` before correlating (a no-op for the common
case of one learning-stage session per mouse).

Decode metric: `real - null_mean` (accuracy above the session's own
linear-shift null) from `046_perfstate_baseline_decoding_results.csv`,
already computed -- NOT raw accuracy, per the 2026-09-16 finding that raw
accuracy here is dominated by shared slow drift the null already explains
(see `046`'s own figure). Both `quiet_window` and `iti` windows get their
own row.

Correlation-figure style (`ssl_correlation_figure_style.md`, locked
convention): scatter + OLS + shaded analytic 95% CI, solid line only if
p<0.05 -- `regress_and_plot` copied verbatim from
`030_wholebrain_behavior_correlation.py` / `037`'s own copy of it (this
project's established per-script self-containment convention, see `042`'s
docstring). Produces both a pooled-cohort and a per-cohort figure, matching
037's own pairing convention.

Usage: python 047_perfstate_baseline_behavior_correlation.py
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
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_bwm_trial_prep import REF_PATH  # noqa: E402
from ssl_timeresolved_decoding import prep_perfstate_trials_curve  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures" / "behavior"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
POOLED_COLOR = "#2c5f5b"
DECODE_TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]  # matches 046


def regress_and_plot(ax, x: np.ndarray, y: np.ndarray, color: str, label: str | None = None) -> str:
    """Verbatim copy of `030_wholebrain_behavior_correlation.py`'s reference
    implementation -- see `ssl_correlation_figure_style.md`."""
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


def load_decode_metric() -> pd.DataFrame:
    """Per-mouse, per-window `real - null_mean` -- averaged across a
    mouse's sessions first (no-op for the common 1-session-per-mouse case)."""
    df = pd.read_csv(OUT_DIR / "046_perfstate_baseline_decoding_results.csv")
    df = df[df["ok"] == True].copy()  # noqa: E712
    df["diff"] = df["real"] - df["null_mean"]
    per_mouse = df.groupby(["mouse", "window"], as_index=False)["diff"].mean()
    wide = per_mouse.pivot(index="mouse", columns="window", values="diff").reset_index()
    wide = wide.rename(columns={"quiet_window": "decode_diff_quiet_window", "iti": "decode_diff_iti"})
    return wide


def per_mouse_behavior_gaps(mice, sessions_tbl, trials_tbl, dataset_root) -> pd.DataFrame:
    """Per-session hit/FA-rate gaps (curve-based perf-state labels, see
    module docstring), averaged to per-mouse across a mouse's sessions."""
    rows = []
    for mouse in mice:
        sessions = sessions_tbl.loc[
            (sessions_tbl["subject_id"] == mouse)
            & (sessions_tbl["session_description"] == "whisker_0")
            & (sessions_tbl["has_ephys"]), "session_id",
        ].tolist()
        session_gaps = []
        for session_id in sessions:
            trials = prep_perfstate_trials_curve(
                dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=DECODE_TRIAL_TYPES,
            )
            if trials is None or len(trials) == 0:
                continue
            per_state_hit, per_state_fa = {}, {}
            for state in ("high", "low"):
                wh = trials[(trials.trial_type == "whisker_trial") & (trials.perf_state == state)]
                ns = trials[(trials.trial_type == "no_stim_trial") & (trials.perf_state == state)]
                per_state_hit[state] = wh["lick_flag"].mean() if len(wh) else np.nan
                per_state_fa[state] = ns["lick_flag"].mean() if len(ns) else np.nan
            hit_rate_gap = per_state_hit["high"] - per_state_hit["low"]
            fa_rate_gap = per_state_fa["high"] - per_state_fa["low"]
            session_gaps.append(dict(hit_rate_gap=hit_rate_gap, fa_rate_gap=fa_rate_gap))
        if not session_gaps:
            continue
        g = pd.DataFrame(session_gaps)
        hit_rate_gap = g["hit_rate_gap"].mean()
        fa_rate_gap = g["fa_rate_gap"].mean()
        rows.append(dict(mouse=mouse, hit_rate_gap=hit_rate_gap, fa_rate_gap=fa_rate_gap,
                          delta_lick_prob_gap=hit_rate_gap - fa_rate_gap, n_sessions=len(session_gaps)))
    return pd.DataFrame(rows)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    decode = load_decode_metric()
    print(f"{len(decode)} mice with a decode metric")
    behavior = per_mouse_behavior_gaps(decode["mouse"].unique(), sessions_tbl, trials_tbl, dataset_root)
    merged = decode.merge(behavior, on="mouse", how="inner")

    ref = pd.read_parquet(Path(REF_PATH))
    reward_group_map = ref.set_index("subject_id")["reward_group"].to_dict()
    merged["reward_group"] = merged["mouse"].map(reward_group_map)
    n_no_group = int(merged["reward_group"].isna().sum())
    merged = merged.dropna(subset=["reward_group"])
    print(f"  {len(merged)} mice merged with behavior + reward_group"
          + (f" ({n_no_group} dropped, no reward_group)" if n_no_group else ""))

    y_specs = [("decode_diff_quiet_window", "quiet_window decode acc. - shift-null"),
               ("decode_diff_iti", "ITI decode acc. - shift-null")]
    x_specs = [
        ("hit_rate_gap", "P(lick|whisker)(high) - P(lick|whisker)(low)", "whisker hit-rate gap"),
        ("fa_rate_gap", "P(lick|no_stim)(high) - P(lick|no_stim)(low)", "false-alarm-rate gap"),
        ("delta_lick_prob_gap", "hit_rate_gap - fa_rate_gap", "hit-minus-FA gap"),
    ]

    for suffix, per_cohort in (("pooled", False), ("percohort", True)):
        fig, axes = plt.subplots(len(y_specs), len(x_specs), figsize=(5.2 * len(x_specs), 5.2 * len(y_specs)), squeeze=False)
        for row_i, (y_col, y_label) in enumerate(y_specs):
            y = merged[y_col].to_numpy(dtype=float)
            for ax, (x_col, x_label, title) in zip(axes[row_i], x_specs):
                x = merged[x_col].to_numpy(dtype=float)
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
                ax.set_xlabel(x_label, fontsize=8.5)
                ax.set_ylabel(y_label, fontsize=8.5)
                ax.set_title(title, fontsize=10)
                ax.set_box_aspect(1)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"046 perf-state baseline decoding vs. behavioral state-separation, per mouse ({suffix})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = FIG_DIR / f"047_perfstate_baseline_behavior_corr_{suffix}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path.name}")

    out_csv = OUT_DIR / "047_perfstate_baseline_behavior_corr.csv"
    merged.to_csv(out_csv, index=False)
    print(f"saved {out_csv.name}")


if __name__ == "__main__":
    main()
