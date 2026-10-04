"""Perf-state decoding: behavioral-gap correlation + session-time-confound
check (user ideas, 2026-09-13). Two per-session scalar correlations against
the perf-state decoding accuracy (`perfstate_stim_whole_brain`, sensory
window 5-35ms, real_curve - 0.5 -- originally read the pre-shift-null
backup since it was already complete while the live rerun was still in
progress; switched to the live file 2026-09-14 once that rerun finished
(89/89) -- this control's own math doesn't depend on the shift-null
addition either way, so results are unaffected by the switch):

1. **Behavioral-gap correlation** (extended 2026-09-13 to three measures,
   user request): does decoding strength scale with how behaviorally
   distinct the two states actually are? Three (high - low) gaps, all raw
   (not cohort-corrected) trial-level rates:
   - `hit_rate_gap`: P(lick|whisker) per state. Note this is close to
     tautological for R+ sessions specifically, where raw `lick_flag` on
     whisker trials IS the cohort-corrected `rewarded` that the median
     split is actually done on (`ssl_bwm_trial_prep.
     cohort_corrected_rewarded`) -- for R- sessions `rewarded` is the
     *inverse* of `lick_flag`, so there it's a genuinely independent
     measure. Included anyway per explicit request, caveat flagged rather
     than hidden.
   - `fa_rate_gap`: mean `block_fa_rate` per state -- independent of the
     split by construction (not used to define `perf_state`).
   - `delta_lick_prob_gap`: gap of (P(lick|whisker) - block_fa_rate) per
     state, i.e. the same discriminability index and per-state computation
     as `030_wholebrain_behavior_correlation.py`'s `perfstate_behavior`.

2. **Session-time confound check**: perf-state blocks may covary with plain
   time-in-session (e.g. a slow session-wide drift), which a decoder could
   pick up without encoding anything about "state" per se. x = ROC AUC of
   normalized trial index (0-1 within session) alone predicting perf_state
   (high=1) -- no neural data, just "how well does raw session position
   alone separate high/low blocks." High neural-decode-vs-this-AUC
   correlation would flag the same class of confound the linear-shift null
   already guards against in the main sweep, from a different angle (this
   is about the LABEL's own time-structure, not neural-vs-label drift).

Both follow this project's locked correlation-figure style
(`ssl_correlation_figure_style.md`): scatter + OLS + shaded analytic 95% CI,
solid line only if p<0.05 -- `regress_and_plot` copied verbatim from
`030_wholebrain_behavior_correlation.py` (the reference implementation).
Produces both a pooled-cohort and a per-cohort figure, matching 030's own
pairing convention.

Second row (user request 2026-09-13): same four x-metrics, but as absolute
magnitude rather than signed value -- tests whether decoding scales with
the *size* of the behavioral/confound separation regardless of its sign
(which, per `hit_rate_gap`, can be cohort-dependent and label-artifactual
rather than meaningful on its own). The session-time confound panel uses
|AUC-0.5| rather than |AUC|, since 0.5 (chance), not 0, is that metric's
own "no effect" reference point.

Usage: python 037_perfstate_confound_and_behavior_correlations.py
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
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import hitmiss_session_list, mean_in_window, prep_perfstate_trials_generic  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures" / "behavior"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TAG = "perfstate_stim_whole_brain"
# Was reading the pre-shift-null backup (see prior comments in git history) --
# switched 2026-09-14 once the live rerun with the shift-null control
# actually finished (89/89), so this now uses genuinely current data.
BACKUP_SUFFIX = ""
SENSORY_WINDOW = (0.005, 0.050)  # updated 2026-09-18 from (0.005, 0.035), user request
DAY_STAGE = "learning"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
POOLED_COLOR = "#2c5f5b"


def regress_and_plot(ax, x: np.ndarray, y: np.ndarray, color: str, label: str | None = None):
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
    df = pd.read_parquet(OUT_DIR / f"024_master_results_{TAG}.parquet{BACKUP_SUFFIX}")
    df = df[(df["skipped_reason"].isna()) & (df["condition_type"] == "whole")]
    bin_edges = json.loads((OUT_DIR / f"024_bin_edges_{TAG}.json{BACKUP_SUFFIX}").read_text())
    bin_labels_s = np.array([e[1] for e in bin_edges])
    df = df.copy()
    df["decode_metric"] = df["real_curve"].apply(lambda c: mean_in_window(np.asarray(c), bin_labels_s, SENSORY_WINDOW) - 0.5)
    return df[["session_id", "reward_group", "decode_metric"]]


def per_session_confounds(session_ids, sessions_tbl, trials_tbl, dataset_root) -> pd.DataFrame:
    """Per-session (high - low) gaps for three behavioral measures (user
    request 2026-09-13) plus the idea-5 session-time confound:
    - `hit_rate_gap`: raw P(lick|whisker) per state (NOT the cohort-corrected
      `rewarded`/`block_hit_rate` that actually defines the high/low split
      -- using the raw rate keeps this a genuinely distinct measure from the
      label-defining one for R- sessions, where `rewarded` is the *inverse*
      of `lick_flag` on whisker trials (`ssl_bwm_trial_prep.
      cohort_corrected_rewarded`); for R+ sessions `rewarded==lick_flag` on
      whisker trials, so this gap is unavoidably close to tautological
      there -- flagged, not hidden.
    - `fa_rate_gap`: mean `block_fa_rate` per state (independent of the
      split by construction, as before).
    - `delta_lick_prob_gap` (user definition, 2026-09-13: "whisker hit rate
      minus the false alarm rate"): `hit_rate_gap - fa_rate_gap` directly
      (algebraically identical to gapping the per-state (P(lick|whisker) -
      block_fa_rate) discriminability index the way
      `030_wholebrain_behavior_correlation.py`'s `perfstate_behavior` does
      it -- verified numerically equal -- but computed directly here so the
      definition is unambiguous from the code alone).
    """
    rows = []
    for session_id in session_ids:
        trials = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"])
        if trials is None or len(trials) == 0:
            continue
        trials = trials.sort_values("start_time").reset_index(drop=True)

        per_state_hit, per_state_fa = {}, {}
        for state in ("high", "low"):
            st = trials[trials.perf_state == state]
            per_state_hit[state] = st["lick_flag"].mean() if len(st) else np.nan
            per_state_fa[state] = st["block_fa_rate"].mean() if len(st) else np.nan
        hit_rate_gap = per_state_hit["high"] - per_state_hit["low"]
        fa_rate_gap = per_state_fa["high"] - per_state_fa["low"]
        delta_lick_prob_gap = hit_rate_gap - fa_rate_gap

        trial_idx_norm = np.arange(len(trials)) / max(len(trials) - 1, 1)
        y_true = (trials["perf_state"] == "high").to_numpy().astype(int)
        trivial_auc = float("nan")
        if 0 < y_true.sum() < len(y_true):
            trivial_auc = roc_auc_score(y_true, trial_idx_norm)

        rows.append(dict(session_id=session_id, hit_rate_gap=hit_rate_gap, fa_rate_gap=fa_rate_gap,
                          delta_lick_prob_gap=delta_lick_prob_gap, trivial_auc=trivial_auc))
    return pd.DataFrame(rows)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    decode = load_decode_metric()
    print(f"{TAG}: {len(decode)} sessions with a decode metric")
    confounds = per_session_confounds(decode["session_id"].unique(), sessions_tbl, trials_tbl, dataset_root)
    merged = decode.merge(confounds, on="session_id", how="inner")
    print(f"  {len(merged)} sessions merged with confound/behavior covariates")

    # Absolute-value row (user request 2026-09-13): does decoding scale with
    # the *magnitude* of the behavioral/confound separation regardless of
    # its (cohort-dependent, sometimes label-artifactual) sign? For the
    # trivial_auc panel the natural "no-effect" point is 0.5, not 0, so its
    # magnitude is |AUC-0.5| rather than |AUC| directly.
    merged["abs_hit_rate_gap"] = merged["hit_rate_gap"].abs()
    merged["abs_fa_rate_gap"] = merged["fa_rate_gap"].abs()
    merged["abs_delta_lick_prob_gap"] = merged["delta_lick_prob_gap"].abs()
    merged["abs_trivial_auc_dev"] = (merged["trivial_auc"] - 0.5).abs()

    rows = [
        ("hit_rate_gap", "P(lick|whisker)(high) - P(lick|whisker)(low)", "whisker hit-rate gap"),
        ("fa_rate_gap", "FA-rate(high) - FA-rate(low)", "false-alarm-rate gap"),
        ("delta_lick_prob_gap", "hit_rate_gap - fa_rate_gap", "delta P(lick) gap"),
        ("trivial_auc", "trial-index-alone AUC for perf_state", "session-time confound check"),
    ]
    rows_abs = [
        ("abs_hit_rate_gap", "|P(lick|whisker)(high) - P(lick|whisker)(low)|", "|whisker hit-rate gap|"),
        ("abs_fa_rate_gap", "|FA-rate(high) - FA-rate(low)|", "|false-alarm-rate gap|"),
        ("abs_delta_lick_prob_gap", "|hit_rate_gap - fa_rate_gap|", "|delta P(lick) gap|"),
        ("abs_trivial_auc_dev", "|trial-index-alone AUC - 0.5|", "|session-time confound| (dev. from chance)"),
    ]
    y = merged["decode_metric"].to_numpy(dtype=float)

    for suffix, per_cohort in (("pooled", False), ("percohort", True)):
        fig, axes = plt.subplots(2, len(rows), figsize=(5.2 * len(rows), 10.0), squeeze=False)
        for row_i, row_spec in enumerate((rows, rows_abs)):
            for ax, (x_col, x_label, title) in zip(axes[row_i], row_spec):
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
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_ylabel(
                    f"perf-state decoding accuracy - 0.5\n(sensory window, "
                    f"{int(SENSORY_WINDOW[0] * 1000)}-{int(SENSORY_WINDOW[1] * 1000)}ms)",
                    fontsize=9,
                )
                ax.set_title(title, fontsize=10)
                ax.set_box_aspect(1)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{TAG} -- decoding vs. behavioral-gap / session-time confound, signed (top) vs. magnitude (bottom) ({suffix})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = FIG_DIR / f"037_{TAG}_confound_behavior_corr_{suffix}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
