"""Window-quantification + cross-generalization statistics on the completed
whole-brain-only `024_master_sweep.py` results (user request 2026-09-12):
"Run the cross-generalization and window statistics on the whole-brain
results first."

Windows (per the full-rebuild spec, `question.md`): sensory (5-35ms
post-stimulus, for the two stim-aligned combos) and pre-lick (150ms before
corrected first lick since 2026-09-27, for the lick-aligned combo). No multiple-comparison correction
across these tests (explicit user instruction) -- only the time-bin-wise
cluster-mass permutation (unchanged, done elsewhere) gets that treatment.

Three test families, per combo:
  1. condition_pair: PAIRED test (Wilcoxon signed-rank + paired t-test)
     comparing the two levels of a condition type (half: first vs second;
     perfstate: high vs low) on window-mean accuracy -- paired because both
     levels come from the same session. Run separately for R+, R-, and
     aggregated (pooled, still paired by session).
  2. cohort: UNPAIRED test (Mann-Whitney + Welch, via the already-locked
     `rplus_rminus_group_test`) comparing R+ vs R- window-mean accuracy, per
     condition_value (whole/first/second/high/low) -- different sessions
     across cohorts, so unpaired.
  3. crossgen_vs_within: PAIRED test comparing a session's own within-
     condition window accuracy against its cross-generalization accuracy
     (trained on this condition, tested on the other) -- same session, so
     paired.

Writes `026_wholebrain_window_stats.csv`.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ssl_timeresolved_decoding import mean_in_window, rplus_rminus_group_test  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)  # updated 2026-09-18 from (0.005, 0.035), user request
PRE_LICK_WINDOW = (-0.100, 0.0)

COMBOS = [
    ("hitmiss_stim_whole_brain", "sensory", SENSORY_WINDOW),
    ("modality_stim_whole_brain", "sensory", SENSORY_WINDOW),
    ("modality_lick_whole_brain", "pre_lick", PRE_LICK_WINDOW),
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
    ("hitmiss_stim_expert_whole_brain", "sensory", SENSORY_WINDOW),
    ("modality_stim_expert_whole_brain", "sensory", SENSORY_WINDOW),
    ("modality_lick_expert_whole_brain", "pre_lick", PRE_LICK_WINDOW),
]


def load_combo(tag: str) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Returns None (rather than raising) when the results/bin-edges files
    don't exist yet -- e.g. modality mid-redo -- so `main`'s single
    end-of-loop CSV write still captures whatever combos ARE ready instead
    of losing them to an unhandled crash on a not-yet-ready one (2026-09-14
    fix, matching how 028/029/030 already skip missing combos per-tag)."""
    partial_path = OUT_DIR / f"024_master_results_{tag}.parquet"
    edges_path = OUT_DIR / f"024_bin_edges_{tag}.json"
    if not partial_path.exists() or not edges_path.exists():
        return None
    df = pd.read_parquet(partial_path)
    df = df[df["skipped_reason"].isna()].copy()
    edges = json.loads(edges_path.read_text())
    bin_labels = np.array([e[1] for e in edges])
    return df, bin_labels


def _curve_window_mean(curve, bin_labels: np.ndarray, window: tuple[float, float]) -> float:
    if curve is None:
        return float("nan")
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        return float("nan")
    return mean_in_window(arr, bin_labels, window)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    if n < 2:
        return dict(n=n, wilcoxon_p=np.nan, ttest_p=np.nan, mean_a=np.nan, mean_b=np.nan, mean_diff=np.nan)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else np.nan
    except ValueError:
        w_p = np.nan
    t_p = float(stats.ttest_rel(a, b).pvalue)
    return dict(n=n, wilcoxon_p=w_p, ttest_p=t_p, mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                mean_diff=float(np.mean(a - b)))


def main():
    all_rows = []
    for tag, window_name, window in COMBOS:
        loaded = load_combo(tag)
        if loaded is None:
            print(f"{tag}: missing results/bin-edges, skipping")
            continue
        df, bin_labels = loaded
        df["window_acc"] = df["real_curve"].apply(lambda c: _curve_window_mean(c, bin_labels, window))
        df["crossgen_window_acc"] = df["crossgen_curve_to_other"].apply(lambda c: _curve_window_mean(c, bin_labels, window))
        n_sessions = df["session_id"].nunique()
        print(f"\n=== {tag} ({window_name} window {window}, {n_sessions} sessions) ===")

        # --- 1. condition_pair ---
        for condition_type, values in (("half", ("first", "second")), ("perfstate", ("high", "low"))):
            sub = df[df.condition_type == condition_type]
            for group_label, group_sub in (("R+", sub[sub.reward_group == "R+"]), ("R-", sub[sub.reward_group == "R-"]), ("aggregated", sub)):
                piv = group_sub.pivot_table(index="session_id", columns="condition_value", values="window_acc")
                if values[0] not in piv.columns or values[1] not in piv.columns:
                    continue
                res = paired_test(piv[values[0]].to_numpy(), piv[values[1]].to_numpy())
                row = dict(combo=tag, window=window_name, test_family="condition_pair", condition_type=condition_type,
                           group=group_label, level_a=values[0], level_b=values[1], **res)
                all_rows.append(row)
                print(f"  [condition_pair] {condition_type} {values[0]} vs {values[1]}, {group_label}: "
                      f"n={row['n']} mean_a={row['mean_a']:.3f} mean_b={row['mean_b']:.3f} "
                      f"wilcoxon_p={row['wilcoxon_p']:.4f} ttest_p={row['ttest_p']:.4f}")

        # --- 2. cohort (R+ vs R-) ---
        rng = np.random.default_rng(0)
        for condition_type in ("whole", "half", "perfstate"):
            sub = df[df.condition_type == condition_type]
            for cv in sorted(sub["condition_value"].dropna().unique()):
                cv_sub = sub[sub.condition_value == cv][["session_id", "subject_id", "reward_group", "window_acc"]].rename(columns={"window_acc": "metric"}).dropna()
                if cv_sub["reward_group"].nunique() < 2 or cv_sub["reward_group"].value_counts().min() < 2:
                    continue
                res = rplus_rminus_group_test(cv_sub, rng)
                row = dict(combo=tag, window=window_name, test_family="cohort", condition_type=condition_type,
                           group=cv, n=len(cv_sub), mannwhitney_p=res["mannwhitney_p"], welch_p=res["welch_p"],
                           mouse_block_perm_p=res["mouse_block_perm_p"], observed_diff=res["observed_diff"],
                           n_rplus=res["n_rplus_sessions"], n_rminus=res["n_rminus_sessions"])
                all_rows.append(row)
                print(f"  [cohort] {condition_type}={cv}: n={row['n']} (R+={row['n_rplus']}, R-={row['n_rminus']}) "
                      f"diff={row['observed_diff']:.3f} mannwhitney_p={row['mannwhitney_p']:.4f} welch_p={row['welch_p']:.4f}")

        # --- 3. crossgen_vs_within ---
        for condition_type in ("half", "perfstate"):
            sub = df[df.condition_type == condition_type]
            for group_label, group_sub in (("R+", sub[sub.reward_group == "R+"]), ("R-", sub[sub.reward_group == "R-"]), ("aggregated", sub)):
                res = paired_test(group_sub["window_acc"].to_numpy(), group_sub["crossgen_window_acc"].to_numpy())
                row = dict(combo=tag, window=window_name, test_family="crossgen_vs_within", condition_type=condition_type,
                           group=group_label, level_a="within", level_b="crossgen", **res)
                all_rows.append(row)
                print(f"  [crossgen_vs_within] {condition_type}, {group_label}: n={row['n']} "
                      f"within={row['mean_a']:.3f} crossgen={row['mean_b']:.3f} "
                      f"wilcoxon_p={row['wilcoxon_p']:.4f} ttest_p={row['ttest_p']:.4f}")

    out = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "026_wholebrain_window_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
