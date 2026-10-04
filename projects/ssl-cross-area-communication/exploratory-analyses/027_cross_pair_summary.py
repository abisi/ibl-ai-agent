"""Cross-pair aggregation figures for the completed 39-pair batch run
(`026_all_pairs_driver.py`). Reads each pair's ALREADY-COMPUTED results --
the cached main-compute-pass pickle (via the same deterministic
fingerprint chain `024` uses to write it, so no recompute) and each
pair's saved `10_window_quantification_stats.csv` -- rather than
rerunning anything.

Revision (Axel, 2026-09-01, "do not correct for multiple comparisons and
plot R+ as bottom triangle and R- as upper triangle, not averaged
together. Also plot as barplots with the cohorts"):
- No FDR correction -- raw Mann-Whitney p<0.05 flags significance.
- Every matrix figure now shows R+ (lower triangle, row>col) and R-
  (upper triangle, row<col) SEPARATELY instead of averaging the two
  cohorts into one symmetric value per cell.
- Added barplot versions (area pair on x-axis, R+/R- as separate colored
  bars with SEM) alongside the matrices, for the dim-1 correlation and
  the window-quantification results.

Pairs are symmetric (only A<->B counted once, per Axel's earlier "pairs
are symmetric, only count A<->B") -- there is no row=target/column=source
directionality in dim-1 correlation strength; that would require the
lagged/lead-lag analysis, which was not run per-pair in this batch
(07/lag-sweep lives in `023`, scoped to a representative subset).

Figures:
1. `matrix_dim1_correlation.png` / `matrix_excess_correlation.png`: area x
   area grid, R+ in the lower triangle, R- in the upper triangle, one
   panel per (variant A/B x condition).
2. `matrix_scree_dimensionality.png`: same R+/R- triangle split, cell =
   mean # leading dimensions above that session's own shuffle-null level.
3. `rplus_rminus_difference_summary.png` + `.csv`: all 39 pairs' window-
   quantification stats concatenated, raw p<0.05 (no correction), R+/R-
   triangle-split matrix of which pairs differ.
4. `barplot_dim1_correlation.png`: same dim-1 data as (1), as grouped bar
   charts (all 39 pairs on the x-axis, R+/R- bars with SEM), one panel per
   (variant, condition).
5. `barplot_window_quantification.png`: window-quantification means (R+/R-
   bars with SEM) across all 39 pairs, one panel per (window, variant).
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

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "figure.titlesize": 15,
})


def _stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def units_fingerprint(session_ids: list[str]) -> str:
    """Cheap re-derivation of load_raw_units_cached's fingerprint -- reads
    only the reference sheet (fast), never touches the (now-deleted, were
    huge) cached unit-table pickles themselves."""
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    ref_fp = _stable_hash(ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group"]].astype(str).values.tolist())
    quality_tiers = ["good", "mua"]
    return _stable_hash({
        "session_ids": sorted(session_ids), "day_to_analyze": "learning",
        "quality_tiers": quality_tiers, "ref_fp": ref_fp,
    })


def load_pair_main_pass(area_a: str, area_b: str, session_ids: list[str]) -> dict | None:
    units_fp = units_fingerprint(session_ids)
    pair_fp = _stable_hash({"units_fp": units_fp, "area_a": area_a, "area_b": area_b, "min_units": MIN_UNITS})
    params_fp = fig024.main_pass_params_fingerprint()
    cache_file = CACHE_DIR / f"main_pass_{pair_fp}_{params_fp}.pkl"
    if not cache_file.exists():
        print(f"  MISSING cache for {area_a} vs {area_b}: {cache_file.name}")
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def dim1_summary(data: dict, cond: tuple, variant: str, cohort: str) -> tuple[float, float, float, int]:
    """Mean |dim-1 correlation|, its SEM, mean |dim-1 shuffle-null|, n sessions."""
    d = data[cond][variant]
    rg = np.array(d["reward_group"])
    idx = np.where(rg == cohort)[0]
    if len(idx) == 0:
        return np.nan, np.nan, np.nan, 0
    true_vals = np.abs(np.array([d["dim_corrs"][i][0] for i in idx], dtype=float))
    null_vals = np.abs(np.array([np.nanmean(d["null_dim_corrs"][i][:, 0]) for i in idx], dtype=float))
    sem = np.nanstd(true_vals, ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else np.nan
    return float(np.nanmean(true_vals)), float(sem), float(np.nanmean(null_vals)), len(idx)


def dimensionality(data: dict, cond: tuple, variant: str, cohort: str) -> tuple[float, int]:
    """Mean (across this cohort's sessions) # leading dims whose
    |correlation| exceeds that session's own shuffle-null (dim-1) level."""
    d = data[cond][variant]
    rg = np.array(d["reward_group"])
    idx = np.where(rg == cohort)[0]
    vals = []
    for i in idx:
        dims = np.abs(d["dim_corrs"][i])
        null_level = np.nanmean(np.abs(d["null_dim_corrs"][i][:, 0]))
        if np.isnan(null_level):
            continue
        above = dims > null_level
        n = 0
        for v in above:
            if not v or np.isnan(v):
                break
            n += 1
        vals.append(n)
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def fill_triangle(mat: np.ndarray, area_idx: dict, a: str, b: str, v_rplus: float, v_rminus: float) -> None:
    """R+ -> lower triangle (row>col), R- -> upper triangle (row<col).
    `a`/`b` order doesn't matter -- the two areas are placed by their
    fixed index positions, not by which came first in the pair list."""
    i, j = area_idx[a], area_idx[b]
    lo, hi = (i, j) if i > j else (j, i)  # lo > hi always
    mat[hi, lo] = v_rminus  # row<col -> upper triangle -> R-
    mat[lo, hi] = v_rplus   # row>col -> lower triangle -> R+


def plot_matrix_grid(mats: dict, panel_keys: list, areas: list[str], n_areas: int, vmin: float, vmax: float,
                      cmap: str, cbar_label: str, suptitle: str, out_path: Path, ncols: int, figsize_per: float) -> None:
    nrows = int(np.ceil(len(panel_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for k, key in enumerate(panel_keys):
        ax = axes[k // ncols, k % ncols]
        im = ax.imshow(mats[key], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(n_areas)); ax.set_xticklabels(areas, rotation=90, fontsize=7)
        ax.set_yticks(range(n_areas)); ax.set_yticklabels(areas, fontsize=7)
        ax.plot([-0.5, n_areas - 0.5], [-0.5, n_areas - 0.5], color="white", linewidth=1.2)
        ax.set_title(str(key), fontsize=9)
    for k in range(len(panel_keys), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label=cbar_label)
    fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def main() -> None:
    pairs_df, session_lists = driver.derive_session_lists()
    areas = sorted(set(pairs_df["area_a"]) | set(pairs_df["area_b"]))
    n_areas = len(areas)
    area_idx = {a: i for i, a in enumerate(areas)}
    pair_order = [(row["area_a"], row["area_b"]) for _, row in pairs_df.iterrows()]

    print(f"[027] Loading cached main-pass results for {len(pairs_df)} pairs...")
    pair_data: dict[tuple[str, str], dict] = {}
    for a, b in pair_order:
        d = load_pair_main_pass(a, b, session_lists[(a, b)])
        if d is not None:
            pair_data[(a, b)] = d
    print(f"[027] {len(pair_data)}/{len(pairs_df)} pairs loaded from cache")

    # ==== Matrices: dim-1 correlation and excess-over-null, R+/R- triangle-split ====
    for metric_name, use_excess in [("dim1_correlation", False), ("excess_correlation", True)]:
        mats, all_vals = {}, []
        for variant in ("A", "B"):
            for cond in CONDITIONS:
                mat = np.full((n_areas, n_areas), np.nan)
                for (a, b), data in pair_data.items():
                    if cond not in data:
                        continue
                    per_cohort = {}
                    for cohort in ("R+", "R-"):
                        true_r, _, null_r, n = dim1_summary(data, cond, variant, cohort)
                        per_cohort[cohort] = (true_r - null_r) if use_excess else true_r if n > 0 else np.nan
                    fill_triangle(mat, area_idx, a, b, per_cohort["R+"], per_cohort["R-"])
                key = f"variant {variant}, {cond[0]} (lick={cond[1]})"
                mats[key] = mat
                all_vals.append(mat)
        vmax = np.nanmax([np.nanmax(np.abs(m)) for m in all_vals if np.any(~np.isnan(m))])
        vmin = 0 if not use_excess else -vmax
        cmap = "viridis" if not use_excess else "RdBu_r"
        label = "|dim-1 correlation - shuffle-null|" if use_excess else "|dim-1 correlation|"
        plot_matrix_grid(mats, list(mats.keys()), areas, n_areas, vmin, vmax, cmap, label,
                          f"Cross-pair {'excess' if use_excess else 'dim-1'} correlation -- "
                          f"R+ (lower triangle) vs R- (upper triangle), all 39 pairs",
                          SUMMARY_DIR / f"matrix_{metric_name}.png", ncols=len(CONDITIONS), figsize_per=5.5)

    # ==== Matrix: dimensionality, R+/R- triangle-split (variant A) ====
    dim_mats = {}
    for cond in CONDITIONS:
        mat = np.full((n_areas, n_areas), np.nan)
        for (a, b), data in pair_data.items():
            if cond not in data:
                continue
            v_plus, n_plus = dimensionality(data, cond, "A", "R+")
            v_minus, n_minus = dimensionality(data, cond, "A", "R-")
            fill_triangle(mat, area_idx, a, b, v_plus if n_plus else np.nan, v_minus if n_minus else np.nan)
        dim_mats[f"{cond[0]} (lick={cond[1]})"] = mat
    vmax_dim = np.nanmax([np.nanmax(m) for m in dim_mats.values() if np.any(~np.isnan(m))])
    plot_matrix_grid(dim_mats, list(dim_mats.keys()), areas, n_areas, 0, vmax_dim, "magma",
                      "# leading dims above shuffle-null",
                      "Cross-pair \"dimensionality\" of communication (variant A) -- R+ (lower triangle) vs "
                      "R- (upper triangle), all 39 pairs",
                      SUMMARY_DIR / "matrix_scree_dimensionality.png", ncols=len(CONDITIONS), figsize_per=5.5)

    # ==== R+/R- difference summary: raw p<0.05, no correction ====
    all_stats = []
    for a, b in pair_order:
        run_dir = RUNS_DIR / f"{driver.slugify(a)}_vs_{driver.slugify(b)}"
        stats_csv = run_dir / "10_window_quantification_stats.csv"
        if not stats_csv.exists():
            continue
        df = pd.read_csv(stats_csv)
        df["area_a"] = a
        df["area_b"] = b
        all_stats.append(df)
    combined = pd.concat(all_stats, ignore_index=True)
    combined["significant_raw_p05"] = combined["mannwhitney_p"] < 0.05
    combined.to_csv(SUMMARY_DIR / "rplus_rminus_difference_summary.csv", index=False)
    print(f"Wrote rplus_rminus_difference_summary.csv ({combined['significant_raw_p05'].sum()}/{len(combined)} "
          f"raw p<0.05, NOT corrected for multiple comparisons)")

    # Direction of the effect determines triangle placement here (not
    # cohort identity, since "significant" is a single yes/no per pair):
    # lower triangle = R+ mean > R- mean, upper = R- mean > R+ mean. A
    # SEPARATE panel per condition (not just per window x variant) --
    # collapsing 5 conditions into one cell per pair silently let the
    # last-processed condition overwrite the others (a real bug caught
    # while reviewing this figure). Only the winning direction's cell is
    # ever written (the other stays NaN/blank) so pale "0.0" Reds-cmap
    # cells can't be mistaken for a tested-but-null result.
    for window in ["baseline", "sensory"]:
        fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(5.5 * len(CONDITIONS), 11.5), constrained_layout=True)
        for row_i, variant in enumerate(("A", "B")):
            for col, cond in enumerate(CONDITIONS):
                ax = axes[row_i, col]
                mat = np.full((n_areas, n_areas), np.nan)
                cond_label = f"{cond[0]}(lick={cond[1]})"
                sub = combined[(combined["window"] == window) & (combined["variant"] == variant)
                                & (combined["condition"] == cond_label)]
                for _, r in sub.iterrows():
                    if not r["significant_raw_p05"]:
                        continue
                    i, j = area_idx[r["area_a"]], area_idx[r["area_b"]]
                    lo, hi = (i, j) if i > j else (j, i)
                    if r["mean_Rplus"] > r["mean_Rminus"]:
                        mat[lo, hi] = 1.0  # lower triangle -> R+ higher
                    else:
                        mat[hi, lo] = 1.0  # upper triangle -> R- higher
                im = ax.imshow(mat, vmin=0, vmax=1, cmap="Reds")
                ax.set_xticks(range(n_areas)); ax.set_xticklabels(areas, rotation=90, fontsize=7)
                ax.set_yticks(range(n_areas)); ax.set_yticklabels(areas, fontsize=7)
                ax.plot([-0.5, n_areas - 0.5], [-0.5, n_areas - 0.5], color="black", linewidth=1.0)
                ax.set_title(f"variant {variant}, {cond[0]} (lick={cond[1]})", fontsize=9)
        fig.suptitle(f"R+ vs R- difference, {window} window (Mann-Whitney, raw p<0.05, NOT corrected for "
                     f"multiple comparisons; red = significant, lower tri = R+ higher, upper tri = R- higher), "
                     f"all 39 pairs")
        out_path = SUMMARY_DIR / f"rplus_rminus_difference_summary_{window}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Wrote {out_path.name}")

    # ==== Barplots: dim-1 correlation, all pairs, R+/R- bars ====
    area_abbrev = {
        "Auditory areas": "Aud", "Hippocampus": "Hipp", "Midbrain": "Mid",
        "Motor and frontal areas": "MFr", "PPC": "PPC", "Retrosplenial areas": "Retro",
        "SSp-orofacial": "SSpO", "SSp-w": "SSpW", "Striatum and pallidum": "Str", "Thalamus": "Thal",
    }
    pair_labels = [f"{area_abbrev[a]}-{area_abbrev[b]}" for a, b in pair_order]
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(0.5 * len(pair_order) + 3, 13), constrained_layout=True)
    for row_i, variant in enumerate(("A", "B")):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row_i, col]
            x = np.arange(len(pair_order))
            for cohort, offset in (("R+", -0.18), ("R-", 0.18)):
                means, sems = [], []
                for a, b in pair_order:
                    data = pair_data.get((a, b))
                    if data is None or cond not in data:
                        means.append(np.nan); sems.append(np.nan)
                        continue
                    m, s, _, n = dim1_summary(data, cond, variant, cohort)
                    means.append(m if n > 0 else np.nan)
                    sems.append(s if n > 0 else np.nan)
                ax.bar(x + offset, means, width=0.36, yerr=sems, color=COHORT_COLOR[cohort], label=cohort, capsize=2)
            ax.set_xticks(x)
            if row_i == 1:
                ax.set_xticklabels(pair_labels, rotation=90, fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.set_title(f"variant {variant}, {cond[0]} (lick={cond[1]})", fontsize=9)
            if col == 0:
                ax.set_ylabel("mean |dim-1 correlation|\n+/- SEM across sessions")
                ax.legend(fontsize=8)
    fig.suptitle("Dim-1 correlation per area pair, R+ vs R- (not averaged), all 39 pairs")
    fig.savefig(SUMMARY_DIR / "barplot_dim1_correlation.png", dpi=110)
    plt.close(fig)
    print("Wrote barplot_dim1_correlation.png")

    # ==== Barplot: window quantification, all pairs, R+/R- bars ====
    fig, axes = plt.subplots(2, 2, figsize=(0.5 * len(pair_order) + 3, 13), constrained_layout=True)
    for i, window in enumerate(["baseline", "sensory"]):
        for j, variant in enumerate(["A", "B"]):
            ax = axes[i, j]
            x = np.arange(len(pair_order))
            for cohort, offset, mean_col, n_col in (("R+", -0.18, "mean_Rplus", "n_Rplus"),
                                                       ("R-", 0.18, "mean_Rminus", "n_Rminus")):
                means = []
                for a, b in pair_order:
                    sub = combined[(combined["area_a"] == a) & (combined["area_b"] == b)
                                    & (combined["window"] == window) & (combined["variant"] == variant)]
                    means.append(sub[mean_col].iloc[0] if len(sub) else np.nan)
                ax.bar(x + offset, means, width=0.36, color=COHORT_COLOR[cohort], label=cohort)
            ax.set_xticks(x)
            if i == 1:
                ax.set_xticklabels(pair_labels, rotation=90, fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.set_title(f"{window} window, variant {variant}", fontsize=9)
            if j == 0:
                ax.set_ylabel("mean CCA correlation-across-time")
                ax.legend(fontsize=8)
    fig.suptitle("Baseline/sensory-window CCA-variate alignment per area pair, R+ vs R- (not averaged), all 39 pairs")
    fig.savefig(SUMMARY_DIR / "barplot_window_quantification.png", dpi=110)
    plt.close(fig)
    print("Wrote barplot_window_quantification.png")

    print(f"\nAll cross-pair summary figures written to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
