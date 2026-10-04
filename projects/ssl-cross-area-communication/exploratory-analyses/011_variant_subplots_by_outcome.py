"""Extended per Axel's 2026-08-31 follow-up:
- All three variants (A/B/C) now get their own refit-per-shuffle null (only
  A had one before) -- figures show them as subplots side by side.
- Trial-type x lick_flag stratification: whisker_trial x {0,1},
  auditory_trial x {0,1} -- CHECKED FIRST against trials.parquet for these 6
  sessions: auditory_trial lick_flag==0 has only 0-6 trials/session (near-
  ceiling auditory performance, a known property of this dataset -- see
  ssl_task_semantics.md) -- not analyzable at this subset scale, so that one
  condition is DROPPED here, explicitly, rather than run on unusable data.
  The other three conditions (whisker x {0,1}, auditory x 1) all have
  >=19 trials/session in this 6-session set.
- "Use all mice, not only learners": already true of this pipeline's mouse
  filter (`apply_mouse_filters` in 000_coverage_lib.py only applies
  exclude==0/exclude_ephys==0/reward_group in {R+,R-} -- no
  learning_category restriction has ever been applied here).
- Results saved per-variant into separate subfolders
  (artifacts/variant_A|B|C/); the cross-variant comparison figures
  (the explicit ask) are necessarily cross-cutting and live at the
  artifacts/ top level.
- Significance changed: no more "fraction of sessions individually
  significant" counting. Now a single distributional test per (dimension,
  pair, condition, variant, tier): paired Wilcoxon signed-rank + paired
  t-test between the across-session distribution of observed_corr and the
  across-session distribution of each session's own mean shuffle value
  (paired by session). "Significant" (large/dark dot) = both p<0.05.

Still 5ms bins (dead zone excluded) and the same 3 area pairs, same 6-session
checkpoint subset -- a large compute expansion (3 variants x 3 conditions
x 2 tiers x 6 sessions x 3 pairs = 324 iterations x 500 shuffles), run here
before any full-dataset scaling.
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
from matplotlib.colors import to_rgb

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
VARIANT_DIRS = {v: ARTIFACTS_DIR / f"variant_{v}" for v in ["A", "B", "C"]}
for d in VARIANT_DIRS.values():
    d.mkdir(exist_ok=True)

AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1)]  # auditory/0 dropped, see docstring
N_UNITS_CAP = 30
MIN_UNITS_GOOD, MIN_UNITS_GOOD_MUA = 10, 30
MIN_TRIALS = 15
N_DIMS_MAX = 20
N_SHUFFLES = 500
REGULARIZATION = 1e-3
BIN_WIDTH = 0.005
WINDOW = cca_lib.WINDOW

SESSIONS = {
    "R+": ["AB127_20240821_103757", "AB130_20240902_123634", "AB125_20240817_123403"],
    "R-": ["MH034_20250514_104756", "AB126_20240822_114405", "AB085_20231005_152636"],
}
ALL_FILES = [f"{s}.nwb" for sessions in SESSIONS.values() for s in sessions]

QUALITY_TIERS = {
    "good": (lambda ut: ut["quality_label"] == "good", MIN_UNITS_GOOD),
    "good_mua": (lambda ut: ut["quality_label"].isin(["good", "mua"]), MIN_UNITS_GOOD_MUA),
}


def square(ax):
    ax.set_box_aspect(1)


def flatten(tensor: np.ndarray) -> np.ndarray:
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


def get_units_for_area(unit_table, session_id, area, tier_fn, min_units, rng):
    sess = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == area)]
    sess = sess[tier_fn(sess)]
    if len(sess) < min_units:
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


REPO_ROOT = Path(__file__).resolve().parents[3]
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"
# Variants A and B already ran to completion and are saved on disk (a prior
# run crashed only because variant C's lick_v was never computed -- bug
# fixed below) -- re-fitting them would waste ~2/3 of an already-expensive
# 324-iteration x 500-shuffle run. Recompute only C, then merge with the
# saved A/B CSVs for the plotting/stats section.
VARIANTS_TO_COMPUTE = ["C"]


def main() -> None:
    unit_table, trial_table = cov_lib.load_units(ALL_FILES, day_to_analyze="learning", max_workers=6)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)  # entire-dataset scope, no learner restriction
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")

    bin_edges = cca_lib.time_bin_edges(WINDOW, BIN_WIDTH)
    rng_master = np.random.default_rng(1101)

    rows = {v: [] for v in VARIANTS_TO_COMPUTE}

    for tier_name, (tier_fn, min_units) in QUALITY_TIERS.items():
        for cohort, session_ids in SESSIONS.items():
            for session_id in session_ids:
                trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
                mouse_id = session_id.split("_")[0]
                lick_times = np.sort(events[(events["session_id"] == session_id)
                                             & (events["event_type"] == "piezo_lick_times")]["time"].to_numpy(dtype=float))

                for area_a, area_b in AREA_PAIRS:
                    units_a = get_units_for_area(unit_table, session_id, area_a, tier_fn, min_units, rng_master)
                    units_b = get_units_for_area(unit_table, session_id, area_b, tier_fn, min_units, rng_master)
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

                        others = unit_table[(unit_table["session_id"] == session_id) & tier_fn(unit_table)
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

                        for variant in VARIANTS_TO_COMPUTE:
                            Z = build_Z(resid_other_v, lick_v, variant)
                            if variant != "A" and Z is None:
                                continue
                            Y = flatten(resid_b_v)
                            model = PartialCCA(regularization=REGULARIZATION).fit(X, Y, Z, verbose=False)
                            observed = model.canonical_correlations_[:n_dims]

                            n_trials = resid_b_v.shape[0]
                            rng_shuf = np.random.default_rng(
                                hash((tier_name, session_id, area_a, area_b, trial_type, lick_flag, variant)) % (2**32))
                            shuf_corrs = np.full((N_SHUFFLES, n_dims), np.nan)
                            for s in range(N_SHUFFLES):
                                perm = rng_shuf.permutation(n_trials)
                                m_shuf = PartialCCA(regularization=REGULARIZATION).fit(
                                    X, flatten(resid_b_v[perm]), Z, verbose=False)
                                shuf_corrs[s] = m_shuf.canonical_correlations_[:n_dims]
                            null_mean, null_sem = shuf_corrs.mean(axis=0), shuf_corrs.std(axis=0) / np.sqrt(N_SHUFFLES)

                            for d in range(n_dims):
                                rows[variant].append({
                                    "quality_tier": tier_name, "reward_group": cohort, "session_id": session_id,
                                    "mouse_id": mouse_id, "area_a": area_a, "area_b": area_b,
                                    "trial_type": trial_type, "lick_flag": lick_flag, "n_trials": len(starts),
                                    "dimension": d + 1, "observed_corr": observed[d], "shuffle_null_mean": null_mean[d],
                                    "shuffle_null_sem": null_sem[d],
                                })
                print(f"[{tier_name}] {session_id} ({cohort}) done")

    for variant in VARIANTS_TO_COMPUTE:
        df = pd.DataFrame(rows[variant])
        out = VARIANT_DIRS[variant] / "results.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")

    # ================= Distributional significance test (replaces the >=2/3-sessions rule) =================
    def paired_test(df: pd.DataFrame) -> tuple[float, float]:
        """Paired Wilcoxon + paired t-test, observed_corr vs. shuffle_null_mean,
        one pair per session (session-level values, not per-shuffle-draw)."""
        obs, null = df["observed_corr"].to_numpy(), df["shuffle_null_mean"].to_numpy()
        if len(obs) < 2 or np.allclose(obs, null):
            return np.nan, np.nan
        try:
            w_p = scipy_stats.wilcoxon(obs, null).pvalue
        except ValueError:
            w_p = np.nan
        t_p = scipy_stats.ttest_rel(obs, null).pvalue
        return w_p, t_p

    def sig_scatter(ax, x, y, color, is_sig):
        for xi, yi, sig in zip(x, y, is_sig):
            if sig:
                ax.scatter(xi, yi, s=140, color=to_rgb(color), alpha=1.0, edgecolor="black", linewidth=0.8, zorder=5)
            else:
                ax.scatter(xi, yi, s=35, color=to_rgb(color), alpha=0.35, zorder=5)

    dims = np.arange(1, N_DIMS_MAX + 1)
    all_dfs = {}
    for v in ["A", "B", "C"]:
        if v in VARIANTS_TO_COMPUTE:
            all_dfs[v] = pd.DataFrame(rows[v])
        else:
            all_dfs[v] = pd.read_csv(VARIANT_DIRS[v] / "results.csv")
    print({v: len(df) for v, df in all_dfs.items()})
    stat_rows = []

    # Significance computed ONCE per (tier, condition, pair, variant, dimension) --
    # pools BOTH cohorts (n=6 sessions), not per-cohort (n=3): a paired Wilcoxon
    # signed-rank test cannot reach p<0.05 at n=3 by construction (min possible
    # p=0.25), which would silently make every per-cohort test powerless. "The
    # distribution of correlation values vs. the distribution of mean shuffle
    # values" is one pooled test per (pair, variant, dimension), applied to both
    # cohorts' points in the figures below (and computed only once, not per
    # raw/excess figure -- the test itself doesn't depend on that choice).
    sig_cache = {}
    for tier_name in QUALITY_TIERS:
        for trial_type, lick_flag in CONDITIONS:
            for area_a, area_b in AREA_PAIRS:
                for variant in ["A", "B", "C"]:
                    df = all_dfs[variant]
                    sub = df[(df.quality_tier == tier_name) & (df.trial_type == trial_type)
                             & (df.lick_flag == lick_flag) & (df.area_a == area_a) & (df.area_b == area_b)]
                    is_sig = np.zeros(len(dims), dtype=bool)
                    for i, d in enumerate(dims):
                        dd = sub[sub.dimension == d]
                        w_p, t_p = paired_test(dd)
                        is_sig[i] = (not np.isnan(w_p)) and (not np.isnan(t_p)) and w_p < 0.05 and t_p < 0.05
                        stat_rows.append({"quality_tier": tier_name, "trial_type": trial_type,
                                           "lick_flag": lick_flag, "area_a": area_a, "area_b": area_b,
                                           "variant": variant, "n_sessions_pooled": len(dd), "dimension": d,
                                           "wilcoxon_p": w_p, "paired_ttest_p": t_p})
                    sig_cache[(tier_name, trial_type, lick_flag, area_a, area_b, variant)] = is_sig

    for tier_name in QUALITY_TIERS:
        for trial_type, lick_flag in CONDITIONS:
            for kind, ylabel_suffix in [("raw", ""), ("excess", " (observed - shuffle null)")]:
                fig, axes = plt.subplots(3, 3, figsize=(19, 19))
                for row, (area_a, area_b) in enumerate(AREA_PAIRS):
                    for col, variant in enumerate(["A", "B", "C"]):
                        ax = axes[row, col]
                        df = all_dfs[variant]
                        sub = df[(df.quality_tier == tier_name) & (df.trial_type == trial_type)
                                 & (df.lick_flag == lick_flag) & (df.area_a == area_a) & (df.area_b == area_b)]
                        is_sig = sig_cache[(tier_name, trial_type, lick_flag, area_a, area_b, variant)]

                        for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                            coh = sub[sub.reward_group == cohort].copy()
                            if coh.empty:
                                continue
                            coh["excess"] = coh["observed_corr"] - coh["shuffle_null_mean"]
                            value_col = "observed_corr" if kind == "raw" else "excess"
                            agg = coh.groupby("dimension")[value_col].agg(["mean", "sem"]).reindex(dims)

                            ax.plot(dims, agg["mean"], color=color, linewidth=1, alpha=0.6)
                            ax.fill_between(dims, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=color, alpha=0.13)
                            sig_scatter(ax, dims, agg["mean"].to_numpy(), color, is_sig)
                        if kind == "excess":
                            ax.axhline(0, color="black", linewidth=0.8)
                        a_short = area_a.replace(" areas", "").replace(" and pallidum", "")
                        b_short = area_b.replace(" areas", "").replace(" and pallidum", "")
                        ax.set_title(f"{a_short} vs {b_short}, variant {variant}", fontsize=10)
                        ax.set_xlabel("canonical dimension")
                        ax.set_ylabel(f"canonical correlation{ylabel_suffix}")
                        square(ax)
                fig.suptitle(f"{tier_name}, {trial_type} (lick_flag={lick_flag}), {kind} correlation, variants A/B/C "
                             f"as subplots\n(5ms bins, dead zone excluded; large/dark dots = paired Wilcoxon AND "
                             f"paired t-test both p<0.05, observed vs. each session's own shuffle null)")
                fig.tight_layout()
                out_name = f"variants_ABC_{kind}_{tier_name}_{trial_type}_lick{lick_flag}.png"
                fig.savefig(ARTIFACTS_DIR / out_name, dpi=110)
                plt.close(fig)
                print(f"Wrote {out_name}")

    pd.DataFrame(stat_rows).to_csv(ARTIFACTS_DIR / "variant_significance_tests.csv", index=False)
    print(f"Wrote variant_significance_tests.csv ({len(stat_rows)} rows)")


if __name__ == "__main__":
    main()
