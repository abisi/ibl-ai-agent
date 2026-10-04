"""Focality-index summary, per Axel's 2026-09-01 "summarize overall
interaction differently, with a focality index between cohort and for
each trial type." Reads the same cached main-compute-pass results as
`027_cross_pair_summary.py` (no recompute).

**Focality index, defined**: for a given area X (or for the whole
network), take the vector of mean |dim-1 correlation| between X and each
of its partner areas (the 39-pair dataset), and compute the Gini
coefficient of that vector. Gini=0 means X interacts equally strongly
with all its partners (diffuse/non-focal communication); Gini->1 means
X's communication is concentrated on very few strong partners (focal),
with the rest near zero. This is the standard inequality/concentration
measure from economics (Lorenz-curve-based), repurposed here as a
concentration-of-connectivity-strength measure -- a common move in
network neuroscience (same idea as node "selectivity"/"sparseness"
indices).

Two views, split by cohort (R+ vs R-, "not averaged together" per the
prior instruction) and trial type:
1. `barplot_focality_per_area.png`: per-area focality -- for each of the
   10 areas, Gini coefficient over that area's own partner-interaction
   vector, one panel per condition, R+/R- bars with SEM (bootstrap over
   the area's set of partner-pair point estimates -- see note in code).
2. `barplot_focality_overall.png`: one whole-network focality index per
   (cohort, condition) -- Gini coefficient over ALL 39 pairs' dim-1
   correlation values at once -- "is total cross-area communication
   concentrated in a few standout pairs, or spread broadly across most
   of them."

Both computed for variant A and variant B.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
fig024 = importlib.import_module("024_updated_figures")
driver = importlib.import_module("026_all_pairs_driver")
cov_lib = importlib.import_module("000_coverage_lib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CACHE_DIR = ARTIFACTS_DIR / "cache"
SUMMARY_DIR = RUNS_DIR / "cross_pair_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = fig024.CONDITIONS
COHORT_COLOR = fig024.COHORT_COLOR
MIN_UNITS = fig024.MIN_UNITS
N_BOOTSTRAP = 500

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "figure.titlesize": 15,
})


def _stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def units_fingerprint(session_ids: list[str]) -> str:
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    ref_fp = _stable_hash(ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group"]].astype(str).values.tolist())
    return _stable_hash({"session_ids": sorted(session_ids), "day_to_analyze": "learning",
                          "quality_tiers": ["good", "mua"], "ref_fp": ref_fp})


def load_pair_main_pass(area_a: str, area_b: str, session_ids: list[str]) -> dict | None:
    units_fp = units_fingerprint(session_ids)
    pair_fp = _stable_hash({"units_fp": units_fp, "area_a": area_a, "area_b": area_b, "min_units": MIN_UNITS})
    params_fp = fig024.main_pass_params_fingerprint()
    cache_file = CACHE_DIR / f"main_pass_{pair_fp}_{params_fp}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def dim1_mean(data: dict, cond: tuple, variant: str, cohort: str) -> float:
    d = data[cond][variant]
    rg = np.array(d["reward_group"])
    idx = np.where(rg == cohort)[0]
    if len(idx) == 0:
        return np.nan
    vals = np.abs(np.array([d["dim_corrs"][i][0] for i in idx], dtype=float))
    return float(np.nanmean(vals))


def gini(values: np.ndarray) -> float:
    """Standard Gini coefficient of a non-negative vector. 0 = perfectly
    equal (diffuse); ->1 = maximally concentrated (focal). NaN if fewer
    than 2 finite, non-all-zero values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or np.all(v == 0):
        return np.nan
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def main() -> None:
    pairs_df, session_lists = driver.derive_session_lists()
    areas = sorted(set(pairs_df["area_a"]) | set(pairs_df["area_b"]))
    pair_order = [(row["area_a"], row["area_b"]) for _, row in pairs_df.iterrows()]
    area_partners = {a: [] for a in areas}
    for a, b in pair_order:
        area_partners[a].append(b)
        area_partners[b].append(a)

    print(f"[028] Loading cached main-pass results for {len(pairs_df)} pairs...")
    pair_data: dict[tuple[str, str], dict] = {}
    for a, b in pair_order:
        d = load_pair_main_pass(a, b, session_lists[(a, b)])
        if d is not None:
            pair_data[(a, b)] = d
    print(f"[028] {len(pair_data)}/{len(pairs_df)} pairs loaded from cache")

    def pair_value(a: str, b: str, cond: tuple, variant: str, cohort: str) -> float:
        data = pair_data.get((a, b)) or pair_data.get((b, a))
        if data is None or cond not in data:
            return np.nan
        return dim1_mean(data, cond, variant, cohort)

    # ==== Per-area focality, R+/R- bars, one figure per variant ====
    rng = np.random.default_rng(0)
    for variant in ("A", "B"):
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5.5 * len(CONDITIONS), 6.5), constrained_layout=True)
        x = np.arange(len(areas))
        for col, cond in enumerate(CONDITIONS):
            ax = axes[col]
            for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
                means, sems = [], []
                for area in areas:
                    partners = area_partners[area]
                    vals = np.array([pair_value(area, p, cond, variant, cohort) for p in partners])
                    g = gini(vals)
                    means.append(g)
                    # Bootstrap SEM: resample partners with replacement,
                    # recompute Gini, take the SD of the bootstrap
                    # distribution as the SEM estimate (Gini has no closed-
                    # form SEM; only sensible if there are >=4 partners).
                    valid = vals[~np.isnan(vals)]
                    if len(valid) >= 4:
                        boot = [gini(rng.choice(valid, size=len(valid), replace=True)) for _ in range(N_BOOTSTRAP)]
                        sems.append(np.nanstd(boot))
                    else:
                        sems.append(np.nan)
                ax.bar(x + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=2)
            ax.set_xticks(x); ax.set_xticklabels(areas, rotation=90, fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"{cond[0]} (lick={cond[1]})", fontsize=9)
            if col == 0:
                ax.set_ylabel("focality index (Gini)\nof partner-interaction strength\n+/- bootstrap SEM")
                ax.legend(fontsize=8)
        fig.suptitle(f"Per-area focality of cross-area communication (variant {variant}) -- R+ vs R- "
                     f"(not averaged), all areas x all conditions")
        out = SUMMARY_DIR / f"barplot_focality_per_area_variant{variant}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out.name}")

    # ==== Overall (whole-network) focality, R+/R- bars, one figure ====
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, variant in zip(axes, ("A", "B")):
        x = np.arange(len(CONDITIONS))
        for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
            means, sems = [], []
            for cond in CONDITIONS:
                vals = np.array([pair_value(a, b, cond, variant, cohort) for a, b in pair_order])
                g = gini(vals)
                means.append(g)
                valid = vals[~np.isnan(vals)]
                boot = [gini(rng.choice(valid, size=len(valid), replace=True)) for _ in range(N_BOOTSTRAP)]
                sems.append(np.nanstd(boot))
            ax.bar(x + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n(lick={c[1]})" for c in CONDITIONS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"variant {variant}")
        ax.set_ylabel("focality index (Gini)\nof all 39 pairs' interaction strength")
        ax.legend(fontsize=9)
    fig.suptitle("Overall network focality of cross-area communication -- R+ vs R- (not averaged), by trial type")
    out = SUMMARY_DIR / "barplot_focality_overall.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out.name}")

    print(f"\nAll focality figures written to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
