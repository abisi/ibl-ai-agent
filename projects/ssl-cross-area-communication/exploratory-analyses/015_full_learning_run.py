"""FULL learning-arm run, per Axel's 2026-08-31 "confirm then run" go-ahead.
50 sessions (26 R+, 24 R-) with all 3 area pairs at >=30 good+MUA units
(from artifacts/full_learning_session_list.csv, checked directly against
full_unit_table_metadata.parquet before launching). Scope decisions, stated
explicitly:
- good_mua tier ONLY (not 'good') -- the checkpoint runs consistently showed
  the same qualitative pattern in both tiers; dropping 'good' here is a
  compute-budget call, not a finding.
- N_SHUFFLES=300 (down from the checkpoint's 500) -- still gives ~0.003
  p-value resolution; a compute-budget call given the ~8x larger session
  count (50 vs 6). 50 sessions x 3 pairs x 3 conditions x 3 variants x 300
  shuffles ~= 405,000 PartialCCA fits, estimated ~2.5-3h from empirical
  timing on the checkpoint run.
Everything else (5ms bins, dead-zone exclusion, 3 area pairs, 3 conditions
with auditory x lick0 dropped, per-variant output folders, pooled paired
Wilcoxon+t-test significance) is unchanged from the checkpoint
(011_variant_subplots_by_outcome.py). Output goes to a SEPARATE
artifacts/full_learning/ tree so it never overwrites the already-reviewed
6-session checkpoint results in artifacts/variant_A|B|C/.

Plotting is updated to the latest agreed style (uniform dots, y-axis shared
per area-pair row across variant columns -- see 013_uniform_dots_and_heatmaps.py)
rather than the superseded significance-dot styling.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cov_lib = importlib.import_module("000_coverage_lib")
cca_lib = importlib.import_module("003_cca_lib")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from partial_CCA import PartialCCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FULL_DIR = ARTIFACTS_DIR / "full_learning"
FULL_DIR.mkdir(exist_ok=True)
VARIANT_DIRS = {v: FULL_DIR / f"variant_{v}" for v in ["A", "B", "C"]}
for d in VARIANT_DIRS.values():
    d.mkdir(exist_ok=True)

AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1)]
N_UNITS_CAP = 30
MIN_UNITS_GOOD_MUA = 30
MIN_TRIALS = 15
N_DIMS_MAX = 20
N_SHUFFLES = 300
REGULARIZATION = 1e-3
BIN_WIDTH = 0.005
WINDOW = cca_lib.WINDOW
TIER_NAME = "good_mua"
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"


def square(ax):
    ax.set_box_aspect(1)


def flatten(tensor: np.ndarray) -> np.ndarray:
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


def get_units_for_area(unit_table, session_id, area, rng):
    sess = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == area)]
    sess = sess[TIER_FN(sess)]
    if len(sess) < MIN_UNITS_GOOD_MUA:
        return None
    n = min(N_UNITS_CAP, len(sess))
    return sess.sample(n=n, random_state=int(rng.integers(1e9)))


def spikes_of(units: pd.DataFrame) -> list[np.ndarray]:
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def build_Z(resid_other_v, lick_v, variant):
    if variant == "A":
        return None
    if variant == "B":
        if resid_other_v is None:
            return None
        flat_o = flatten(resid_other_v)
        pca_o, k = cca_lib.fit_nuisance_pca(flat_o)
        return pca_o.transform(flat_o)[:, :k]
    if lick_v is None:
        return None
    return flatten(lick_v)


def main() -> None:
    session_list = pd.read_csv(ARTIFACTS_DIR / "full_learning_session_list.csv")
    files = [f"{s}.nwb" for s in session_list["session_id"]]
    print(f"Loading {len(files)} learning-arm sessions...")

    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")

    bin_edges = cca_lib.time_bin_edges(WINDOW, BIN_WIDTH)
    rng_master = np.random.default_rng(20260831)

    rows = {v: [] for v in ["A", "B", "C"]}

    for _, srow in session_list.iterrows():
        session_id, cohort = srow["session_id"], srow["reward_group"]
        trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
        mouse_id = session_id.split("_")[0]
        lick_times = np.sort(events[(events["session_id"] == session_id)
                                     & (events["event_type"] == "piezo_lick_times")]["time"].to_numpy(dtype=float))

        for area_a, area_b in AREA_PAIRS:
            units_a = get_units_for_area(unit_table, session_id, area_a, rng_master)
            units_b = get_units_for_area(unit_table, session_id, area_b, rng_master)
            if units_a is None or units_b is None:
                continue
            spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)

            for trial_type, lick_flag in CONDITIONS:
                is_whisker = trial_type == "whisker_trial"
                tt = trials_sess[(trials_sess["trial_type"] == trial_type)
                                  & (trials_sess["lick_flag"] == lick_flag)].sort_values("start_time")
                starts = tt["start_time"].to_numpy(dtype=float)
                if len(starts) < MIN_TRIALS:
                    continue
                dz_mask = cca_lib.dead_zone_bin_mask(bin_edges, is_whisker)
                valid_bins = ~dz_mask

                tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
                tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)
                resid_a_v = cca_lib.noise_correlation_residuals(tensor_a)[:, valid_bins, :]
                resid_b_v = cca_lib.noise_correlation_residuals(tensor_b)[:, valid_bins, :]

                others = unit_table[(unit_table["session_id"] == session_id) & TIER_FN(unit_table)
                                     & (~unit_table["area_group_coarse"].isin([area_a, area_b]))]
                resid_other_v, lick_v = None, None
                if len(others) >= 5:
                    tensor_o = cca_lib.population_tensor(spikes_of(others), starts, is_whisker, bin_edges)
                    resid_other_v = cca_lib.noise_correlation_residuals(tensor_o)[:, valid_bins, :]
                if len(lick_times):
                    lick_reg = cca_lib.lick_rate_regressor(lick_times, starts, bin_edges)
                    lick_v = lick_reg[:, valid_bins, None]

                n_dims = min(N_DIMS_MAX, len(units_a), len(units_b))
                X = flatten(resid_a_v)

                for variant in ["A", "B", "C"]:
                    Z = build_Z(resid_other_v, lick_v, variant)
                    if variant != "A" and Z is None:
                        continue
                    Y = flatten(resid_b_v)
                    model = PartialCCA(regularization=REGULARIZATION).fit(X, Y, Z, verbose=False)
                    observed = model.canonical_correlations_[:n_dims]

                    n_trials = resid_b_v.shape[0]
                    rng_shuf = np.random.default_rng(
                        hash((session_id, area_a, area_b, trial_type, lick_flag, variant)) % (2**32))
                    shuf_corrs = np.full((N_SHUFFLES, n_dims), np.nan)
                    for s in range(N_SHUFFLES):
                        perm = rng_shuf.permutation(n_trials)
                        m_shuf = PartialCCA(regularization=REGULARIZATION).fit(
                            X, flatten(resid_b_v[perm]), Z, verbose=False)
                        shuf_corrs[s] = m_shuf.canonical_correlations_[:n_dims]
                    null_mean, null_sem = shuf_corrs.mean(axis=0), shuf_corrs.std(axis=0) / np.sqrt(N_SHUFFLES)

                    for d in range(n_dims):
                        rows[variant].append({
                            "reward_group": cohort, "session_id": session_id, "mouse_id": mouse_id,
                            "area_a": area_a, "area_b": area_b, "trial_type": trial_type, "lick_flag": lick_flag,
                            "n_trials": len(starts), "dimension": d + 1, "observed_corr": observed[d],
                            "shuffle_null_mean": null_mean[d], "shuffle_null_sem": null_sem[d],
                        })
        print(f"{session_id} ({cohort}) done")

    for variant in ["A", "B", "C"]:
        df = pd.DataFrame(rows[variant])
        out = VARIANT_DIRS[variant] / "results.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")

    # ================= Pooled paired Wilcoxon + t-test significance (stats only, not plotted as dot styling) =================
    def paired_test(df: pd.DataFrame) -> tuple[float, float]:
        obs, null = df["observed_corr"].to_numpy(), df["shuffle_null_mean"].to_numpy()
        if len(obs) < 2 or np.allclose(obs, null):
            return np.nan, np.nan
        try:
            w_p = scipy_stats.wilcoxon(obs, null).pvalue
        except ValueError:
            w_p = np.nan
        t_p = scipy_stats.ttest_rel(obs, null).pvalue
        return w_p, t_p

    dims = np.arange(1, N_DIMS_MAX + 1)
    all_dfs = {v: pd.DataFrame(rows[v]) for v in ["A", "B", "C"]}
    stat_rows = []
    for trial_type, lick_flag in CONDITIONS:
        for area_a, area_b in AREA_PAIRS:
            for variant in ["A", "B", "C"]:
                df = all_dfs[variant]
                sub = df[(df.trial_type == trial_type) & (df.lick_flag == lick_flag)
                         & (df.area_a == area_a) & (df.area_b == area_b)]
                for d in dims:
                    dd = sub[sub.dimension == d]
                    w_p, t_p = paired_test(dd)
                    stat_rows.append({"trial_type": trial_type, "lick_flag": lick_flag, "area_a": area_a,
                                       "area_b": area_b, "variant": variant, "n_sessions_pooled": len(dd),
                                       "dimension": d, "wilcoxon_p": w_p, "paired_ttest_p": t_p})
    pd.DataFrame(stat_rows).to_csv(FULL_DIR / "variant_significance_tests.csv", index=False)
    print(f"Wrote variant_significance_tests.csv ({len(stat_rows)} rows)")

    # ================= Figures: uniform dots, y shared per row across variants =================
    def short(area: str) -> str:
        return area.replace(" areas", "").replace(" and pallidum", "")

    for trial_type, lick_flag in CONDITIONS:
        for kind, ylabel_suffix in [("raw", ""), ("excess", " (observed - shuffle null)")]:
            fig, axes = plt.subplots(3, 3, figsize=(19, 20))
            for row, (area_a, area_b) in enumerate(AREA_PAIRS):
                row_aggs = {}
                for variant in ["A", "B", "C"]:
                    df = all_dfs[variant]
                    sub = df[(df.trial_type == trial_type) & (df.lick_flag == lick_flag)
                             & (df.area_a == area_a) & (df.area_b == area_b)]
                    aggs = {}
                    for cohort in ["R+", "R-"]:
                        coh = sub[sub.reward_group == cohort].copy()
                        if coh.empty:
                            continue
                        coh["excess"] = coh["observed_corr"] - coh["shuffle_null_mean"]
                        value_col = "observed_corr" if kind == "raw" else "excess"
                        aggs[cohort] = coh.groupby("dimension")[value_col].agg(["mean", "sem"]).reindex(dims)
                    row_aggs[variant] = aggs

                all_vals = []
                for variant, aggs in row_aggs.items():
                    for cohort, agg in aggs.items():
                        all_vals.append((agg["mean"] - agg["sem"]).dropna())
                        all_vals.append((agg["mean"] + agg["sem"]).dropna())
                nonempty_vals = [v for v in all_vals if len(v)]
                if nonempty_vals:
                    y_lo = min(v.min() for v in nonempty_vals)
                    y_hi = max(v.max() for v in nonempty_vals)
                    pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.01
                    if kind == "excess":
                        y_lo = min(y_lo, 0)
                else:
                    # No data for this (pair, condition) in any variant/cohort --
                    # leave y-axis on auto-scale rather than crashing; panels will
                    # render empty, which is itself informative (no usable sessions).
                    y_lo, y_hi, pad = 0.0, 1.0, 0.0

                for col, variant in enumerate(["A", "B", "C"]):
                    ax = axes[row, col]
                    for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                        if cohort not in row_aggs[variant]:
                            continue
                        agg = row_aggs[variant][cohort]
                        ax.plot(dims, agg["mean"], color=color, linewidth=1, alpha=0.7)
                        ax.fill_between(dims, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=color, alpha=0.13)
                        ax.scatter(dims, agg["mean"], s=45, color=color, alpha=0.85, edgecolor="none", zorder=5)
                    if kind == "excess":
                        ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylim(y_lo - pad, y_hi + pad)
                    ax.tick_params(labelleft=True)
                    ax.set_title(f"{short(area_a)} vs {short(area_b)}, variant {variant}", fontsize=10)
                    ax.set_xlabel("canonical dimension")
                    ax.set_ylabel(f"canonical correlation{ylabel_suffix}")
                    square(ax)
            fig.suptitle(f"FULL learning dataset (50 sessions, 26 R+/24 R-), {trial_type} (lick_flag={lick_flag}), "
                         f"{kind} correlation, variants A/B/C as subplots\n(good+MUA, 5ms bins, dead zone excluded, "
                         f"300 shuffles; y-axis shared within each row)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out_name = f"full_variants_ABC_{kind}_{trial_type}_lick{lick_flag}.png"
            fig.savefig(FULL_DIR / out_name, dpi=110)
            plt.close(fig)
            print(f"Wrote {out_name}")

    # ================= Lag-0 summary heatmaps =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for col, variant in enumerate(["A", "B", "C"]):
        df = all_dfs[variant]
        d1 = df[df.dimension == 1]
        mat = np.full((len(AREA_PAIRS), len(CONDITIONS)), np.nan)
        for i, (area_a, area_b) in enumerate(AREA_PAIRS):
            for j, (trial_type, lick_flag) in enumerate(CONDITIONS):
                cell = d1[(d1.area_a == area_a) & (d1.area_b == area_b)
                          & (d1.trial_type == trial_type) & (d1.lick_flag == lick_flag)]
                if len(cell):
                    mat[i, j] = cell["observed_corr"].mean()
        ax = axes[col]
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([f"{tt.replace('_trial','')}\nlick={lf}" for tt, lf in CONDITIONS], fontsize=8)
        ax.set_yticks(range(len(AREA_PAIRS)))
        ax.set_yticklabels([f"{short(a)} vs\n{short(b)}" for a, b in AREA_PAIRS], fontsize=8)
        for i in range(len(AREA_PAIRS)):
            for j in range(len(CONDITIONS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if mat[i, j] < np.nanmean(mat) else "black")
        ax.set_title(f"variant {variant}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="dim-1 canonical correlation")
    fig.suptitle("FULL learning dataset -- lag-0 canonical correlation summary (dim 1, mean across sessions x cohorts)")
    fig.tight_layout()
    fig.savefig(FULL_DIR / "full_heatmap_lag0_summary.png", dpi=130)
    plt.close(fig)
    print("Wrote full_heatmap_lag0_summary.png")


if __name__ == "__main__":
    main()
