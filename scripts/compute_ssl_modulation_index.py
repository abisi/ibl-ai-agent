"""Suppression-selectivity modulation index (MI) per unit, plus the
cohort-difference test and the firing-rate-confound check.

MI = (|delta_w| - |delta_a|) / (|delta_w| + |delta_a|), using the
evoked-baseline-corrected post-minus-pre delta (delta_corrected) as "delta
spikes" -- the most direct measure of a stimulus-specific response change,
isolated from generic baseline drift and from the raw (non-baseline-
corrected) evoked signal. This is a judgment call (the user did not specify
which of the three windows "delta spikes" refers to); documented here and
in the report.

Absolute value is required for boundedness: with signed deltas, (a-b)/(a+b)
is only guaranteed in [-1,1] when a and b share a sign. A unit suppressed by
one modality and excited by the other can drive a+b through zero, making
the raw signed ratio unbounded/undefined -- the opposite of the user's
"bounded between -1 and 1" requirement. Using |delta_w|, |delta_a| trades
away the excitation-vs-suppression sign (this is a magnitude-selectivity
index, not a direction index) in exchange for actually satisfying that
requirement.

Cohort-difference test: MI rows are per-unit (neuron-level) but reward_group
is a mouse-level factor -- a naive Mann-Whitney U on pooled units would be
the unit-level-test-of-a-mouse-level-factor pseudoreplication issue flagged
in ssl-analyze/references/ssl_analysis_patterns.md. Implemented instead as a
mouse-block permutation test on the observed difference of per-mouse median
MI between reward groups (nonparametric, unpaired, respects the mouse as
the unit of random assignment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DERIVED_DIR = Path("reports/ssl_analysis/derived")
RESULTS_DIR = Path("reports/ssl_analysis/results")
RNG_SEED = 20260814  # per ibl-analyze reproducibility_qc.md: fixed seed for stochastic steps

N_PERMUTATIONS = 100_000


def mouse_block_permutation_test(mouse_values: pd.Series, mouse_groups: pd.Series, n_perm: int, seed: int) -> dict:
    """Two-sided permutation test on the difference of group medians,
    permuting group labels at the mouse level."""
    rng = np.random.default_rng(seed)
    values = mouse_values.to_numpy()
    groups = mouse_groups.to_numpy()
    group_levels = np.unique(groups)
    assert len(group_levels) == 2, f"expected 2 groups, got {group_levels}"
    g0, g1 = group_levels

    observed = np.median(values[groups == g1]) - np.median(values[groups == g0])
    n1 = int((groups == g1).sum())
    n = len(values)

    perm_diffs = np.empty(n_perm)
    idx = np.arange(n)
    for i in range(n_perm):
        perm = rng.permutation(idx)
        perm_g1 = perm[:n1]
        perm_g0 = perm[n1:]
        perm_diffs[i] = np.median(values[perm_g1]) - np.median(values[perm_g0])

    p_value = float((np.abs(perm_diffs) >= np.abs(observed)).mean())
    return {
        "observed_diff_median": float(observed),
        "n_mice_group0": int((groups == g0).sum()),
        "n_mice_group1": int((groups == g1).sum()),
        "group0": str(g0), "group1": str(g1),
        "n_permutations": n_perm,
        "p_value": p_value,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lmm_table = pd.read_parquet(DERIVED_DIR / "lmm_table.parquet")
    rates = pd.read_parquet(DERIVED_DIR / "unit_condition_rates.parquet")

    idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group"]
    wide = lmm_table.pivot_table(index=idx_cols, columns="stim_type", values="delta_corrected").reset_index()
    wide = wide.dropna(subset=["whisker", "auditory"])

    denom = wide["whisker"].abs() + wide["auditory"].abs()
    numer = wide["whisker"].abs() - wide["auditory"].abs()
    wide["modulation_index"] = np.where(denom > 0, numer / denom, np.nan)
    n_undefined = wide["modulation_index"].isna().sum()
    print(f"Units with MI undefined (both deltas exactly 0): {n_undefined} / {len(wide)}")
    wide = wide.dropna(subset=["modulation_index"])
    assert wide["modulation_index"].between(-1, 1).all(), "MI escaped [-1,1] -- bug"

    overall_rate = (
        rates.groupby(["session_id", "cluster_id"])[["baseline_rate_hz", "evoked_rate_hz"]]
        .mean().mean(axis=1).rename("overall_firing_rate_hz")
    )
    wide = wide.merge(overall_rate, on=["session_id", "cluster_id"], how="left")

    wide.to_parquet(DERIVED_DIR / "modulation_index.parquet", index=False)
    print(f"Wrote {len(wide)} unit MI rows.")
    print(wide["modulation_index"].describe())
    print()
    print("By reward_group:")
    print(wide.groupby("reward_group")["modulation_index"].describe())

    per_mouse = wide.groupby(["mouse_id", "reward_group"])["modulation_index"].median().reset_index()
    print()
    print(f"Per-mouse median MI, n={len(per_mouse)} mice")

    cohort_test = mouse_block_permutation_test(
        per_mouse["modulation_index"], per_mouse["reward_group"], N_PERMUTATIONS, RNG_SEED
    )
    print("Mouse-block permutation test (reward_group difference in median MI):")
    print(cohort_test)

    rho, rho_p = stats.spearmanr(wide["modulation_index"], wide["overall_firing_rate_hz"])
    fr_confound = {"spearman_rho": float(rho), "p_value": float(rho_p), "n": int(len(wide))}
    print()
    print(f"MI vs overall firing rate (Spearman): rho={rho:.4f}, p={rho_p:.4g}")

    import json
    with open(RESULTS_DIR / "modulation_index_results.json", "w") as f:
        json.dump({"cohort_permutation_test": cohort_test, "firing_rate_confound": fr_confound, "n_undefined_dropped": int(n_undefined)}, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'modulation_index_results.json'}")


if __name__ == "__main__":
    main()
