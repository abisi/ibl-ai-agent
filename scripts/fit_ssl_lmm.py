"""Parsimonious LMM hierarchy, per analysis window (baseline / evoked /
evoked-baseline-corrected), random intercept for mouse:

  A: delta ~ stim_type * reward_group
  B: A + n_stim (per-row stim_type-matched active-portion stimulus count)
  C_intercept: B + area_group (main effect / intercept shift only)
  C_threeway:  B + stim_type * reward_group * area_group (full factorial)

C_intercept is nested in C_threeway (same terms with the higher-order
area interactions constrained to 0), so a likelihood-ratio test between
them (both fit with ML, reml=False) answers whether the data actually
need area-specific stim x reward interactions, or whether area intercepts
suffice -- rather than assuming either structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DERIVED_DIR = Path("reports/ssl_analysis/derived")
RESULTS_DIR = Path("reports/ssl_analysis/results")


def fit(formula: str, data: pd.DataFrame, groups: pd.Series):
    # lbfgs produces degenerate zero-variance boundary solutions (llf=inf) on
    # this data. No single remaining optimizer is reliable across all model
    # sizes either (bfgs/cg fail to converge on the 51-param three-way model;
    # powell found a higher, converged likelihood there). Try several, keep
    # the converged fit with the highest likelihood; if none converge, keep
    # the highest-likelihood attempt but flag it.
    best = None
    best_converged_llf = -np.inf
    attempts = []
    for method in ["bfgs", "cg", "powell", "nm"]:
        model = smf.mixedlm(formula, data, groups=groups)
        result = model.fit(reml=False, method=method, maxiter=1000)
        attempts.append((method, bool(result.converged), float(result.llf)))
        if not np.isfinite(result.llf):
            continue
        if result.converged and result.llf > best_converged_llf:
            best = result
            best_converged_llf = result.llf
    if best is None:
        # nothing converged cleanly -- fall back to the highest finite llf
        finite = [(m, c, l) for m, c, l in attempts if np.isfinite(l)]
        if not finite:
            raise RuntimeError(f"No optimizer produced a finite likelihood: {attempts}")
        best_method = max(finite, key=lambda t: t[2])[0]
        model = smf.mixedlm(formula, data, groups=groups)
        best = model.fit(reml=False, method=best_method, maxiter=1000)
        print(f"  WARNING: no optimizer converged for this formula; using best non-converged fit ({best_method}). Attempts: {attempts}")
    return best


def summarize(result, label: str) -> dict:
    return {
        "label": label,
        "converged": bool(result.converged),
        "llf": float(result.llf),
        "n_params": int(result.params.shape[0]) + 1,  # + residual variance
        "aic": float(result.aic),
        "bic": float(result.bic),
        "fixed_effects": {k: float(v) for k, v in result.params.items()},
        "pvalues": {k: float(v) for k, v in result.pvalues.items()},
    }


def lrt(reduced_llf: float, reduced_k: int, full_llf: float, full_k: int) -> dict:
    stat = 2 * (full_llf - reduced_llf)
    df = full_k - reduced_k
    p = float(stats.chi2.sf(stat, df)) if df > 0 else float("nan")
    return {"lr_stat": float(stat), "df": int(df), "p_value": p}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DERIVED_DIR / "lmm_table.parquet")
    df["n_stim_z"] = (df["n_stim"] - df["n_stim"].mean()) / df["n_stim"].std()

    # "Pons and medulla" has 0 R+ units (33 R- units only, all one mouse's
    # incidental probe placement) -- a structurally empty cell in the
    # stim_type x reward_group x area_group design, which produced a
    # degenerate (~1e16) coefficient in an initial fit. Area-specific models
    # cannot estimate anything for an area entirely missing one factor
    # level, so it is excluded here rather than left in to silently corrupt
    # the area-model fixed effects and the LRT.
    excluded_area = "Pons and medulla"
    n_before = len(df)
    df = df[df["area_group"] != excluded_area].copy()
    if hasattr(df["area_group"], "cat"):
        df["area_group"] = df["area_group"].cat.remove_unused_categories()
    print(f"Excluded {excluded_area} ({n_before - len(df)} rows, 0 R+ units) from area-stratified models.")

    outcomes = ["delta_baseline", "delta_evoked", "delta_corrected"]
    all_results: dict[str, dict] = {}

    for outcome in outcomes:
        print(f"\n===== outcome: {outcome} =====", flush=True)
        data = df.dropna(subset=[outcome, "area_group"]).copy()
        groups = data["mouse_id"]

        formulas = {
            "A_global": f"{outcome} ~ C(stim_type) * C(reward_group)",
            "B_plus_nstim": f"{outcome} ~ C(stim_type) * C(reward_group) + n_stim_z",
            "C_area_intercept": f"{outcome} ~ C(stim_type) * C(reward_group) + n_stim_z + C(area_group)",
            "C_area_threeway": f"{outcome} ~ C(stim_type) * C(reward_group) * C(area_group) + n_stim_z",
        }

        outcome_results = {}
        for label, formula in formulas.items():
            print(f"Fitting {label}: {formula}", flush=True)
            res = fit(formula, data, groups)
            summary = summarize(res, label)
            outcome_results[label] = summary
            print(f"  converged={summary['converged']} llf={summary['llf']:.1f} aic={summary['aic']:.1f} n_params={summary['n_params']}")

        comparison = lrt(
            outcome_results["C_area_intercept"]["llf"], outcome_results["C_area_intercept"]["n_params"],
            outcome_results["C_area_threeway"]["llf"], outcome_results["C_area_threeway"]["n_params"],
        )
        preferred = "C_area_threeway" if comparison["p_value"] < 0.05 else "C_area_intercept"
        print(f"LRT (intercept-only vs three-way): {comparison} -> preferred = {preferred}")

        all_results[outcome] = {
            "models": outcome_results,
            "area_structure_lrt": comparison,
            "preferred_area_model": preferred,
            "n_rows": int(len(data)),
            "n_mice": int(data["mouse_id"].nunique()),
        }

    with open(RESULTS_DIR / "lmm_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'lmm_results.json'}")


if __name__ == "__main__":
    main()
