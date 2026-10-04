"""Fine-area-level analog of `031_windowed_summary.py`, per Axel's
2026-09-01 "do the same but for the fine area level... only pairs with at
least 5 recordings of each." Reads the cached main-compute-pass results
from the 82-pair fine-level batch (`032_all_fine_pairs_driver.py`, all
succeeded) -- no recompute. Same windowed (baseline/sensory, always
separate) + R+/R- (triangle-split matrices, separate bars, never
averaged) + uncorrected-p-value design as `031`, just keyed on
`area_acronym_custom` (25 fine areas actually appear across the 82
qualifying pairs) instead of the 10 coarse areas.

Figures (all in `artifacts/runs/cross_pair_summary/windowed_fine/`):
- `matrix_{window}.png`, `barplot_pairs_{window}.png`,
  `focality_per_area_{window}_variant{A,B}.png`,
  `focality_overall_{window}.png`,
  `overall_communication_by_cohort_{window}.png` + `.csv`.
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
cca_lib = importlib.import_module("003_cca_lib")
fine_driver = importlib.import_module("032_all_fine_pairs_driver")
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
OUT_DIR = RUNS_DIR / "cross_pair_summary" / "windowed_fine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = fig024.CONDITIONS
COHORT_COLOR = fig024.COHORT_COLOR
MIN_UNITS = fig024.MIN_UNITS
AREA_COLUMN = "area_acronym_custom"
N_BOOTSTRAP = 500
WINDOWS = ["baseline", "sensory"]

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
    pair_fp = _stable_hash({"units_fp": units_fp, "area_a": area_a, "area_b": area_b, "min_units": MIN_UNITS,
                             "area_column": AREA_COLUMN})
    params_fp = fig024.main_pass_params_fingerprint()
    cache_file = CACHE_DIR / f"main_pass_{pair_fp}_{params_fp}.pkl"
    if not cache_file.exists():
        print(f"  MISSING cache for {area_a} vs {area_b}: {cache_file.name}")
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def gini(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or np.all(v == 0):
        return np.nan
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def windowed_session_values(data: dict, cond: tuple, variant: str, cohort: str, window_mask: np.ndarray) -> np.ndarray:
    d = data[cond][variant]
    rg = np.array(d["reward_group"])
    idx = np.where(rg == cohort)[0]
    if len(idx) == 0:
        return np.array([])
    return np.array([np.nanmean(np.abs(d["corr_t"][i][window_mask])) for i in idx])


def main() -> None:
    pairs, session_lists = fine_driver.derive_fine_pairs()
    pair_order = pairs
    print(f"[033] {len(pair_order)} fine pairs to summarize")

    areas = sorted(set(a for a, b in pair_order) | set(b for a, b in pair_order))
    area_idx = {a: i for i, a in enumerate(areas)}
    n_areas = len(areas)
    area_partners = {a: [] for a in areas}
    for a, b in pair_order:
        area_partners[a].append(b)
        area_partners[b].append(a)

    print(f"[033] Loading cached main-pass results for {len(pair_order)} pairs...")
    pair_data: dict[tuple[str, str], dict] = {}
    for a, b in pair_order:
        d = load_pair_main_pass(a, b, session_lists[(a, b)])
        if d is not None:
            pair_data[(a, b)] = d
    print(f"[033] {len(pair_data)}/{len(pair_order)} pairs loaded from cache")

    bin_starts = cca_lib.sliding_window_starts()
    bin_centers = cca_lib.sliding_window_centers(bin_starts)
    t_ms = bin_centers * 1000
    win_masks_by_cond = {}
    for cond in CONDITIONS:
        is_whisker = cond[0] == "whisker_trial"
        baseline_mask, sensory_mask = fig024.window_masks(t_ms, is_whisker)
        win_masks_by_cond[cond] = {"baseline": baseline_mask, "sensory": sensory_mask}

    def pv(a: str, b: str, cond: tuple, variant: str, cohort: str, window: str) -> np.ndarray:
        data = pair_data.get((a, b)) or pair_data.get((b, a))
        if data is None or cond not in data:
            return np.array([])
        return windowed_session_values(data, cond, variant, cohort, win_masks_by_cond[cond][window])

    def pv_mean(a: str, b: str, cond: tuple, variant: str, cohort: str, window: str) -> float:
        vals = pv(a, b, cond, variant, cohort, window)
        return float(np.nanmean(vals)) if len(vals) else np.nan

    rng = np.random.default_rng(0)

    for window in WINDOWS:
        print(f"\n[033] ==== window = {window} ====")

        # ---- Matrices ----
        mats, all_vals = {}, []
        for variant in ("A", "B"):
            for cond in CONDITIONS:
                mat = np.full((n_areas, n_areas), np.nan)
                for a, b in pair_order:
                    if (a, b) not in pair_data:
                        continue
                    v_p = pv_mean(a, b, cond, variant, "R+", window)
                    v_m = pv_mean(a, b, cond, variant, "R-", window)
                    i, j = area_idx[a], area_idx[b]
                    lo, hi = (i, j) if i > j else (j, i)
                    mat[hi, lo] = v_m
                    mat[lo, hi] = v_p
                key = f"variant {variant}, {cond[0]} (lick={cond[1]})"
                mats[key] = mat
                all_vals.append(mat)
        vmax = np.nanmax([np.nanmax(np.abs(m)) for m in all_vals if np.any(~np.isnan(m))])
        nrows, ncols = 2, len(CONDITIONS)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 12.5), constrained_layout=True)
        im = None
        for k, key in enumerate(mats):
            ax = axes[k // ncols, k % ncols]
            im = ax.imshow(mats[key], vmin=0, vmax=vmax, cmap="viridis")
            ax.set_xticks(range(n_areas)); ax.set_xticklabels(areas, rotation=90, fontsize=6.5)
            ax.set_yticks(range(n_areas)); ax.set_yticklabels(areas, fontsize=6.5)
            ax.plot([-0.5, n_areas - 0.5], [-0.5, n_areas - 0.5], color="white", linewidth=1.0)
            ax.set_title(key, fontsize=9)
        fig.colorbar(im, ax=axes, shrink=0.6, label=f"mean |correlation-across-time|, {window} window")
        fig.suptitle(f"Fine-area cross-pair {window}-window communication -- R+ (lower tri) vs R- (upper tri), "
                     f"{len(pair_data)} pairs (>= 5 sessions/cohort)")
        fig.savefig(OUT_DIR / f"matrix_{window}.png", dpi=120)
        plt.close(fig)
        print(f"Wrote matrix_{window}.png")

        # ---- Barplots ----
        pair_labels = [f"{a}-{b}" for a, b in pair_order]
        fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(0.4 * len(pair_order) + 3, 13), constrained_layout=True)
        for row_i, variant in enumerate(("A", "B")):
            for col, cond in enumerate(CONDITIONS):
                ax = axes[row_i, col]
                xarr = np.arange(len(pair_order))
                for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
                    means, sems = [], []
                    for a, b in pair_order:
                        vals = pv(a, b, cond, variant, cohort, window)
                        means.append(np.nanmean(vals) if len(vals) else np.nan)
                        sems.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)
                    ax.bar(xarr + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=2)
                ax.set_xticks(xarr)
                if row_i == 1:
                    ax.set_xticklabels(pair_labels, rotation=90, fontsize=5)
                else:
                    ax.set_xticklabels([])
                ax.set_title(f"variant {variant}, {cond[0]} (lick={cond[1]})", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"mean |correlation|, {window} window\n+/- SEM across sessions")
                    ax.legend(fontsize=8)
        fig.suptitle(f"Per-pair {window}-window communication, R+ vs R- (not averaged), "
                     f"{len(pair_data)} fine-area pairs")
        fig.savefig(OUT_DIR / f"barplot_pairs_{window}.png", dpi=110)
        plt.close(fig)
        print(f"Wrote barplot_pairs_{window}.png")

        # ---- Focality (per-area + overall) ----
        for variant in ("A", "B"):
            fig, ax = plt.subplots(figsize=(0.5 * len(areas) + 3, 6.5), constrained_layout=True)
            xarr = np.arange(len(areas))
            for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
                means, sems = [], []
                for area in areas:
                    partners = area_partners[area]
                    partner_vals = []
                    for p in partners:
                        cond_vals = [pv_mean(area, p, c, variant, cohort, window) for c in CONDITIONS]
                        partner_vals.append(np.nanmean(cond_vals))
                    partner_vals = np.array(partner_vals)
                    means.append(gini(partner_vals))
                    valid = partner_vals[~np.isnan(partner_vals)]
                    if len(valid) >= 4:
                        boot = [gini(rng.choice(valid, size=len(valid), replace=True)) for _ in range(N_BOOTSTRAP)]
                        sems.append(np.nanstd(boot))
                    else:
                        sems.append(np.nan)
                ax.bar(xarr + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=2)
            ax.set_xticks(xarr); ax.set_xticklabels(areas, rotation=90, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"focality index (Gini), {window} window\n+/- bootstrap SEM")
            ax.legend(fontsize=9)
            ax.set_title(f"Per-area focality (fine), {window} window (variant {variant}, conditions averaged)")
            fig.savefig(OUT_DIR / f"focality_per_area_{window}_variant{variant}.png", dpi=120)
            plt.close(fig)
            print(f"Wrote focality_per_area_{window}_variant{variant}.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
        for ax, variant in zip(axes, ("A", "B")):
            means, sems = {}, {}
            for cohort in ("R+", "R-"):
                pair_vals = []
                for a, b in pair_order:
                    cond_vals = [pv_mean(a, b, c, variant, cohort, window) for c in CONDITIONS]
                    pair_vals.append(np.nanmean(cond_vals))
                pair_vals = np.array(pair_vals)
                means[cohort] = gini(pair_vals)
                valid = pair_vals[~np.isnan(pair_vals)]
                boot = [gini(rng.choice(valid, size=len(valid), replace=True)) for _ in range(N_BOOTSTRAP)]
                sems[cohort] = np.nanstd(boot)
            ax.bar([0, 1], [means["R+"], means["R-"]], yerr=[sems["R+"], sems["R-"]],
                   color=[COHORT_COLOR["R+"], COHORT_COLOR["R-"]], capsize=4, width=0.5)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R-"])
            ax.set_ylim(0, 1)
            ax.set_title(f"variant {variant}")
            ax.set_ylabel(f"focality index (Gini), {window} window\nof all {len(pair_data)} fine pairs")
        fig.suptitle(f"Overall (fine-level) network focality, {window} window, R+ vs R- (conditions averaged)")
        fig.savefig(OUT_DIR / f"focality_overall_{window}.png", dpi=120)
        plt.close(fig)
        print(f"Wrote focality_overall_{window}.png")

        # ---- Overall communication by cohort ----
        stats_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
        for ax, variant in zip(axes, ("A", "B")):
            xarr = np.arange(len(CONDITIONS))
            pooled = {cohort: {cond: [] for cond in CONDITIONS} for cohort in ("R+", "R-")}
            for a, b in pair_order:
                if (a, b) not in pair_data:
                    continue
                for cond in CONDITIONS:
                    for cohort in ("R+", "R-"):
                        vals = pv(a, b, cond, variant, cohort, window)
                        pooled[cohort][cond].extend(vals.tolist())
            for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
                means = [np.nanmean(pooled[cohort][c]) for c in CONDITIONS]
                sems = [np.nanstd(pooled[cohort][c], ddof=1) / np.sqrt(len(pooled[cohort][c]))
                        if len(pooled[cohort][c]) > 1 else np.nan for c in CONDITIONS]
                ax.bar(xarr + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=3)
            y_top = 0
            for cond in CONDITIONS:
                vp = np.array(pooled["R+"][cond]); vm = np.array(pooled["R-"][cond])
                vp, vm = vp[~np.isnan(vp)], vm[~np.isnan(vm)]
                if len(vp) > 1 and len(vm) > 1:
                    u_stat, p_mw = stats.mannwhitneyu(vp, vm, alternative="two-sided")
                    t_stat, p_welch = stats.ttest_ind(vp, vm, equal_var=False)
                    stats_rows.append({"variant": variant, "window": window,
                                        "condition": f"{cond[0]}(lick={cond[1]})",
                                        "n_Rplus": len(vp), "n_Rminus": len(vm),
                                        "mean_Rplus": np.mean(vp), "mean_Rminus": np.mean(vm),
                                        "mannwhitney_p": p_mw, "welch_p": p_welch})
                    y_top = max(y_top, np.mean(vp) + np.std(vp, ddof=1) / np.sqrt(len(vp)),
                                np.mean(vm) + np.std(vm, ddof=1) / np.sqrt(len(vm)))
            for col, cond in enumerate(CONDITIONS):
                row = next((r for r in stats_rows if r["variant"] == variant and r["window"] == window
                            and r["condition"] == f"{cond[0]}(lick={cond[1]})"), None)
                if row is not None:
                    sig = "*" if row["mannwhitney_p"] < 0.05 else ""
                    ax.text(col, y_top * 1.05, f"p={row['mannwhitney_p']:.3f}{sig}", ha="center", fontsize=8)
            ax.set_xticks(xarr)
            ax.set_xticklabels([f"{c[0]}\n(lick={c[1]})" for c in CONDITIONS], fontsize=9)
            ax.set_ylim(0, y_top * 1.25)
            ax.set_ylabel(f"mean |correlation|, {window} window\n+/- SEM, pooled sessions x pairs")
            ax.set_title(f"variant {variant}")
            ax.legend(fontsize=10)
        fig.suptitle(f"Fine-level overall communication, {window} window, R+ vs R-, by trial type "
                     f"({len(pair_data)} pairs, session-pooled, Mann-Whitney p shown)")
        fig.savefig(OUT_DIR / f"overall_communication_by_cohort_{window}.png", dpi=130)
        plt.close(fig)
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(OUT_DIR / f"overall_communication_by_cohort_{window}_stats.csv", index=False)
        print(f"Wrote overall_communication_by_cohort_{window}.png + stats.csv")
        print(stats_df.to_string(index=False))

    print(f"\nAll fine-level windowed cross-pair figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
