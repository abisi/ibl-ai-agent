"""Overall cross-area communication strength per cohort, across trial
types, per Axel's 2026-09-01 "Compare just overall communication per
cohort across trial types." Simpler than the focality-index figures --
this is the mean (not Gini-concentration) of dim-1 |correlation|, but
pooled across ALL SESSIONS from ALL 39 PAIRS (not one point per pair) --
session_id is the established unit of analysis for this project's stats
(per [[ssl_stats_unit_of_analysis]]), giving real N per condition/cohort
instead of the pair-level N=39 used in the focality figures. Reads the
same cached main-compute-pass results as `027`/`028`/`029` (no recompute).

R+ vs R- compared with the established test pair (Mann-Whitney U + Welch
t-test, both always reported per [[ssl_rplus_rminus_test_pair]]).

Figure: `barplot_overall_communication_by_cohort.png` -- x-axis = 5 trial
types, R+/R- bars with SEM across pooled sessions, one panel per variant
(A/B), p-values annotated per condition.
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
from scipy import stats
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

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
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


def main() -> None:
    pairs_df, session_lists = driver.derive_session_lists()
    pair_order = [(row["area_a"], row["area_b"]) for _, row in pairs_df.iterrows()]

    print(f"[030] Loading cached main-pass results for {len(pairs_df)} pairs...")
    pair_data: dict[tuple[str, str], dict] = {}
    for a, b in pair_order:
        d = load_pair_main_pass(a, b, session_lists[(a, b)])
        if d is not None:
            pair_data[(a, b)] = d
    print(f"[030] {len(pair_data)}/{len(pairs_df)} pairs loaded from cache")

    # Pool every (pair, session) dim-1 |correlation| value, per cohort/condition/variant.
    stats_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    for ax, variant in zip(axes, ("A", "B")):
        x = np.arange(len(CONDITIONS))
        pooled = {cohort: {cond: [] for cond in CONDITIONS} for cohort in ("R+", "R-")}
        for data in pair_data.values():
            for cond in CONDITIONS:
                if cond not in data:
                    continue
                d = data[cond][variant]
                rg = np.array(d["reward_group"])
                vals = np.abs(np.array([dc[0] for dc in d["dim_corrs"]], dtype=float))
                for cohort in ("R+", "R-"):
                    idx = np.where(rg == cohort)[0]
                    pooled[cohort][cond].extend(vals[idx].tolist())

        for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
            means = [np.nanmean(pooled[cohort][c]) for c in CONDITIONS]
            sems = [np.nanstd(pooled[cohort][c], ddof=1) / np.sqrt(len(pooled[cohort][c]))
                    if len(pooled[cohort][c]) > 1 else np.nan for c in CONDITIONS]
            ax.bar(x + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=3)

        y_top = 0
        for col, cond in enumerate(CONDITIONS):
            vp = np.array(pooled["R+"][cond])
            vm = np.array(pooled["R-"][cond])
            vp, vm = vp[~np.isnan(vp)], vm[~np.isnan(vm)]
            if len(vp) > 1 and len(vm) > 1:
                u_stat, p_mw = stats.mannwhitneyu(vp, vm, alternative="two-sided")
                t_stat, p_welch = stats.ttest_ind(vp, vm, equal_var=False)
                stats_rows.append({"variant": variant, "condition": f"{cond[0]}(lick={cond[1]})",
                                    "n_Rplus": len(vp), "n_Rminus": len(vm),
                                    "mean_Rplus": np.mean(vp), "mean_Rminus": np.mean(vm),
                                    "mannwhitney_p": p_mw, "welch_p": p_welch})
                y_top = max(y_top, np.nanmean(vp) + (np.nanstd(vp, ddof=1) / np.sqrt(len(vp))),
                            np.nanmean(vm) + (np.nanstd(vm, ddof=1) / np.sqrt(len(vm))))
        for col, cond in enumerate(CONDITIONS):
            row = next((r for r in stats_rows if r["variant"] == variant
                        and r["condition"] == f"{cond[0]}(lick={cond[1]})"), None)
            if row is not None:
                sig = "*" if row["mannwhitney_p"] < 0.05 else ""
                ax.text(col, y_top * 1.05, f"p={row['mannwhitney_p']:.3f}{sig}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n(lick={c[1]})" for c in CONDITIONS], fontsize=9)
        ax.set_ylim(0, y_top * 1.25)
        ax.set_ylabel("mean |dim-1 correlation|\n+/- SEM, pooled across sessions x all 39 pairs")
        ax.set_title(f"variant {variant}")
        ax.legend(fontsize=10)

    fig.suptitle("Overall cross-area communication strength, R+ vs R-, by trial type "
                 "(session-level pooling across all 39 pairs, Mann-Whitney p shown)")
    out_png = SUMMARY_DIR / "barplot_overall_communication_by_cohort.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_png.name}")

    stats_df = pd.DataFrame(stats_rows)
    out_csv = SUMMARY_DIR / "overall_communication_by_cohort_stats.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.name}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
