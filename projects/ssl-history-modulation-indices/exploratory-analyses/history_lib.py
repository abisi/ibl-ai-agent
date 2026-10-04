"""Shared helpers for the RHMI/EHMI history-modulation-index project. See
../question.md for the full locked metric definition.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf

DATASET_ROOT = Path("reports/datasets/ssl_ephys/1.0.0")
EVOKED_WINDOW_S = (0.005, 0.050)  # starts exactly at the dead-zone boundary (-10ms/+5ms per this project); no clipping needed
DEAD_ZONE_S = (-0.010, 0.005)  # whisker trials only
MIN_TRIALS_PER_BUCKET = 5
QC_VALUES = ("good",)


def cohort_corrected_correct(trial_type: pd.Series, lick_flag: pd.Series, reward_group: str) -> np.ndarray:
    """Task-rule correctness (NOT reward receipt). R+: lick=correct on both
    modalities. R-: withhold=correct on whisker, lick=correct on auditory."""
    lick = lick_flag.to_numpy().astype(bool)
    is_wh = (trial_type == "whisker_trial").to_numpy()
    is_aud = (trial_type == "auditory_trial").to_numpy()
    correct = np.zeros(len(trial_type), dtype=bool)
    if reward_group == "R+":
        correct[is_wh] = lick[is_wh]
    elif reward_group == "R-":
        correct[is_wh] = ~lick[is_wh]
    else:
        raise ValueError(reward_group)
    correct[is_aud] = lick[is_aud]
    return correct


def literal_reward_delivered(trial_type: pd.Series, lick_flag: pd.Series, reward_group: str) -> np.ndarray:
    """Actual water delivery. Whisker trials only deliver water for R+
    licks; R- whisker trials never deliver water, correct or not."""
    lick = lick_flag.to_numpy().astype(bool)
    is_wh = (trial_type == "whisker_trial").to_numpy()
    is_aud = (trial_type == "auditory_trial").to_numpy()
    rewarded = np.zeros(len(trial_type), dtype=bool)
    if reward_group == "R+":
        rewarded[is_wh] = lick[is_wh]
    rewarded[is_aud] = lick[is_aud]
    return rewarded


def unit_rates_for_trials(spike_times_sorted, start_time, window):
    """Per-trial firing rate (Hz) in `window` (s, relative to start_time).
    Window is [5ms,50ms], entirely outside the whisker dead zone
    (-10ms/+5ms or -1ms/+4ms alike) by construction -- no clipping needed."""
    w0, w1 = window
    lo = np.searchsorted(spike_times_sorted, start_time + w0, side="left")
    hi = np.searchsorted(spike_times_sorted, start_time + w1, side="left")
    return (hi - lo).astype(np.float64) / (w1 - w0)


def prep_trials(session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame, reward_group: str) -> pd.DataFrame:
    trials = trials_tbl[trials_tbl["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    has_context = trials["context"].notna() & (trials["context"] != "nan")
    if has_context.any():
        trials = trials[trials["context"] == "active"].reset_index(drop=True)
    trials = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)

    trials["correct"] = cohort_corrected_correct(trials["trial_type"], trials["lick_flag"], reward_group)
    trials["reward_delivered"] = literal_reward_delivered(trials["trial_type"], trials["lick_flag"], reward_group)
    trials["t1_reward_delivered"] = trials["reward_delivered"].shift(1)
    trials = trials.iloc[1:].reset_index(drop=True)  # drop first trial (no t-1)
    return trials


def prep_trials_auditory_only(session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> pd.DataFrame:
    """"Within-task" trial prep (2026-08-22): restricts the ENTIRE candidate
    sequence to auditory_trial rows only, for both `t` and `t-1` -- removes
    the whisker Go/No-Go cohort asymmetry entirely (auditory contingency is
    lick=reward in both cohorts, so `correct`==`reward_delivered` here by
    construction and the formula is identical for R+/R-, unlike
    `prep_trials`). `t-1` is the previous AUDITORY trial specifically, not
    the previous active trial of any modality -- a stricter within-task
    sequence than prep_trials's cross-modality t-1. `reward_group` is not
    needed as an argument since auditory correctness doesn't depend on it."""
    trials = trials_tbl[trials_tbl["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    has_context = trials["context"].notna() & (trials["context"] != "nan")
    if has_context.any():
        trials = trials[trials["context"] == "active"].reset_index(drop=True)
    trials = trials[trials["trial_type"] == "auditory_trial"].reset_index(drop=True)

    lick = trials["lick_flag"].to_numpy().astype(bool)
    trials["correct"] = lick
    trials["reward_delivered"] = lick
    trials["t1_reward_delivered"] = trials["reward_delivered"].shift(1)
    trials = trials.iloc[1:].reset_index(drop=True)  # drop first trial (no t-1)
    return trials


def compute_index_for_unit(spike_times_sorted, start_time, t_mask, t1_rewarded, window):
    """Returns dict with n_a, n_b, fr_a, fr_b, denom_mean_fr, index (or NaNs)."""
    rates = unit_rates_for_trials(spike_times_sorted, start_time, window)
    a_mask = t_mask & t1_rewarded
    b_mask = t_mask & ~t1_rewarded
    n_a, n_b = int(a_mask.sum()), int(b_mask.sum())
    if n_a < MIN_TRIALS_PER_BUCKET or n_b < MIN_TRIALS_PER_BUCKET:
        return {"n_a": n_a, "n_b": n_b, "fr_a": np.nan, "fr_b": np.nan, "denom_mean_fr": np.nan, "index": np.nan}
    fr_a = rates[a_mask].mean()
    fr_b = rates[b_mask].mean()
    denom = rates[t_mask].mean()  # mean FR across ALL t-trials, both t-1 buckets pooled
    index = (fr_a - fr_b) / denom if denom > 0 else np.nan
    return {"n_a": n_a, "n_b": n_b, "fr_a": fr_a, "fr_b": fr_b, "denom_mean_fr": denom, "index": index}


def mouse_level_test(df: pd.DataFrame, metric: str) -> dict:
    """Non-parametric (Mann-Whitney) + parametric (Welch's t) R+ vs R- test
    on one value per mouse (median index across all its valid units) --
    avoids neuron-level pseudoreplication by construction. Per established
    SSL practice, always report both."""
    valid = df[df[metric].notna()]
    per_mouse = valid.groupby(["mouse_id", "reward_group"])[metric].median().reset_index()
    rplus = per_mouse.loc[per_mouse.reward_group == "R+", metric].to_numpy()
    rminus = per_mouse.loc[per_mouse.reward_group == "R-", metric].to_numpy()

    u_stat, u_p = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    t_stat, t_p = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)  # Welch's

    return {
        "n_mice_rplus": len(rplus), "n_mice_rminus": len(rminus),
        "median_rplus": float(np.median(rplus)), "median_rminus": float(np.median(rminus)),
        "mean_rplus": float(np.mean(rplus)), "mean_rminus": float(np.mean(rminus)),
        "mannwhitney_u": float(u_stat), "mannwhitney_p": float(u_p),
        "welch_t": float(t_stat), "welch_p": float(t_p),
    }


def neuron_level_lmm(df: pd.DataFrame, metric: str) -> dict:
    """index ~ C(reward_group) with a mouse-level random intercept
    (statsmodels MixedLM) -- full neuron-level N while accounting for
    non-independence of units within the same mouse."""
    valid = df[df[metric].notna()][["mouse_id", "reward_group", metric]].copy()
    valid = valid.rename(columns={metric: "y"})
    model = smf.mixedlm("y ~ C(reward_group, Treatment(reference='R+'))", valid, groups=valid["mouse_id"])
    result = model.fit(reml=True)
    coef_name = [c for c in result.params.index if "reward_group" in c][0]
    return {
        "n_units": len(valid), "n_mice": valid["mouse_id"].nunique(),
        "coef_rminus_vs_rplus": float(result.params[coef_name]),
        "se": float(result.bse[coef_name]),
        "z": float(result.tvalues[coef_name]),
        "p": float(result.pvalues[coef_name]),
        "converged": bool(result.converged),
    }


def mouse_level_proportion_test(df: pd.DataFrame, metric: str, threshold: float = 0.0) -> dict:
    """Per-mouse fraction of units with index > threshold (default: sign
    test, fraction "positively"/enhanced-modulated), R+ vs R- compared with
    Mann-Whitney + Welch's t -- same mouse-as-replicate logic as
    mouse_level_test, applied to a proportion instead of a location
    statistic. Answers "does the balance of enhanced- vs suppressed-
    modulation units differ by cohort", distinct from "does the typical
    index value differ"."""
    valid = df[df[metric].notna()]
    per_mouse = valid.groupby(["mouse_id", "reward_group"])[metric].apply(
        lambda s: float((s > threshold).mean())
    ).reset_index(name="frac_positive")
    rplus = per_mouse.loc[per_mouse.reward_group == "R+", "frac_positive"].to_numpy()
    rminus = per_mouse.loc[per_mouse.reward_group == "R-", "frac_positive"].to_numpy()

    u_stat, u_p = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
    t_stat, t_p = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)

    return {
        "threshold": threshold,
        "n_mice_rplus": len(rplus), "n_mice_rminus": len(rminus),
        "mean_frac_rplus": float(np.mean(rplus)), "mean_frac_rminus": float(np.mean(rminus)),
        "median_frac_rplus": float(np.median(rplus)), "median_frac_rminus": float(np.median(rminus)),
        "mannwhitney_u": float(u_stat), "mannwhitney_p": float(u_p),
        "welch_t": float(t_stat), "welch_p": float(t_p),
    }


def neuron_level_ks_test(df: pd.DataFrame, metric: str) -> dict:
    """Two-sample Kolmogorov-Smirnov test on pooled units -- tests for ANY
    distributional difference (location, spread, shape), not just a mean/
    median shift. Pseudoreplication-caveated (pools units across mice with
    unequal counts per mouse) -- see mouse_block_permutation_ks_test for
    the mouse-respecting version."""
    valid = df[df[metric].notna()]
    rplus = valid.loc[valid.reward_group == "R+", metric].to_numpy()
    rminus = valid.loc[valid.reward_group == "R-", metric].to_numpy()
    res = scipy_stats.ks_2samp(rplus, rminus)
    return {"n_rplus": len(rplus), "n_rminus": len(rminus), "ks_stat": float(res.statistic), "p": float(res.pvalue)}


def mouse_block_permutation_ks_test(df: pd.DataFrame, metric: str, n_perm: int = 2000, seed: int = 0) -> dict:
    """KS statistic between cohort distributions, with a null built by
    permuting the R+/R- label across MICE (not units) and recomputing the
    pooled-unit KS statistic each time -- respects non-independence of
    units within a mouse, same mouse-block-permutation logic used to fix
    the pseudoreplication bug in the ssl-reward-history-modulation
    project's PERMANOVA. This is the rigorous version of
    neuron_level_ks_test; prefer this one when reporting a distributional
    test result."""
    valid = df[df[metric].notna()][["mouse_id", "reward_group", metric]].copy()
    mouse_labels = valid.drop_duplicates("mouse_id")[["mouse_id", "reward_group"]].reset_index(drop=True)
    mouse_ids = mouse_labels["mouse_id"].to_numpy()
    true_labels = mouse_labels["reward_group"].to_numpy()

    def ks_stat_for_labels(group_of_row):
        a = valid.loc[group_of_row == "R+", metric]
        b = valid.loc[group_of_row == "R-", metric]
        return scipy_stats.ks_2samp(a, b).statistic

    observed = ks_stat_for_labels(valid["reward_group"])

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(true_labels)
        label_map = dict(zip(mouse_ids, perm_labels))
        shuffled_group = valid["mouse_id"].map(label_map)
        null_stats[i] = ks_stat_for_labels(shuffled_group)

    p_perm = float((np.sum(null_stats >= observed) + 1) / (n_perm + 1))
    return {
        "n_mice_rplus": int((true_labels == "R+").sum()), "n_mice_rminus": int((true_labels == "R-").sum()),
        "observed_ks_stat": float(observed), "n_perm": n_perm, "p_perm": p_perm,
        "null_mean": float(null_stats.mean()), "null_std": float(null_stats.std()),
    }


def run_distribution_proportion_tests(in_path: Path, out_path: Path, n_perm: int = 2000) -> dict:
    df = pd.read_parquet(in_path)
    print(f"{len(df)} unit rows, {df['session_id'].nunique()} sessions, {df['mouse_id'].nunique()} mice")

    results = {}
    for metric in ("rhmi_index", "ehmi_index"):
        print(f"\n=== {metric.upper()} ===")
        try:
            prop_res = mouse_level_proportion_test(df, metric)
            print("Mouse-level proportion (frac index>0):", json.dumps(prop_res, indent=2))
            ks_res = neuron_level_ks_test(df, metric)
            print("Neuron-level KS (pseudoreplication-caveated):", json.dumps(ks_res, indent=2))
            perm_res = mouse_block_permutation_ks_test(df, metric, n_perm=n_perm)
            print(f"Mouse-block permutation KS ({n_perm} perms):", json.dumps(perm_res, indent=2))
            results[metric] = {
                "mouse_level_proportion": prop_res,
                "neuron_level_ks": ks_res,
                "mouse_block_permutation_ks": perm_res,
            }
        except (ValueError, ZeroDivisionError) as e:
            print(f"  SKIPPED {metric}: {e} (likely too little data in one cohort)")
            results[metric] = {"error": str(e)}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


def paired_metric_test(df: pd.DataFrame, cohort: str) -> dict:
    """Within one cohort, paired RHMI vs EHMI comparison (same unit has
    both) -- "between metrics" question, kept separate per cohort per user
    request 2026-08-22 rather than pooling R+/R-. Mouse-level: per-mouse
    median RHMI and median EHMI, paired Wilcoxon signed-rank across mice
    (avoids neuron-level pseudoreplication, same logic as mouse_level_test).
    Neuron-level: paired Wilcoxon on matched units directly, reported as a
    secondary/pseudoreplication-caveated check."""
    sub = df[df.reward_group == cohort]
    paired = sub[sub["rhmi_index"].notna() & sub["ehmi_index"].notna()]

    per_mouse = paired.groupby("mouse_id")[["rhmi_index", "ehmi_index"]].median()
    mouse_w_stat, mouse_w_p = scipy_stats.wilcoxon(per_mouse["rhmi_index"], per_mouse["ehmi_index"])

    unit_w_stat, unit_w_p = scipy_stats.wilcoxon(paired["rhmi_index"], paired["ehmi_index"])

    return {
        "cohort": cohort, "n_mice": len(per_mouse), "n_units_paired": len(paired),
        "mouse_level_median_rhmi": float(per_mouse["rhmi_index"].median()),
        "mouse_level_median_ehmi": float(per_mouse["ehmi_index"].median()),
        "mouse_level_wilcoxon_stat": float(mouse_w_stat), "mouse_level_wilcoxon_p": float(mouse_w_p),
        "unit_level_wilcoxon_stat": float(unit_w_stat), "unit_level_wilcoxon_p": float(unit_w_p),
    }


def run_paired_metric_tests(in_path: Path, out_path: Path) -> dict:
    df = pd.read_parquet(in_path)
    results = {}
    for cohort in ("R+", "R-"):
        print(f"\n=== {cohort}: RHMI vs EHMI (paired, within cohort) ===")
        try:
            res = paired_metric_test(df, cohort)
            print(json.dumps(res, indent=2))
            results[cohort] = res
        except (ValueError, ZeroDivisionError) as e:
            print(f"  SKIPPED {cohort}: {e} (likely too little paired data)")
            results[cohort] = {"error": str(e)}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


def run_stats(in_path: Path, out_path: Path) -> dict:
    df = pd.read_parquet(in_path)
    print(f"{len(df)} unit rows, {df['session_id'].nunique()} sessions, {df['mouse_id'].nunique()} mice")

    results = {}
    for metric in ("rhmi_index", "ehmi_index"):
        print(f"\n=== {metric.upper()} ===")
        try:
            mouse_res = mouse_level_test(df, metric)
            print("Mouse-level:", json.dumps(mouse_res, indent=2))
            lmm_res = neuron_level_lmm(df, metric)
            print("Neuron-level LMM:", json.dumps(lmm_res, indent=2))
            results[metric] = {"mouse_level": mouse_res, "neuron_level_lmm": lmm_res}
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            print(f"  SKIPPED {metric}: {e} (likely too little data in one cohort)")
            results[metric] = {"error": str(e)}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results
