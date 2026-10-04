"""All-pairs run per Axel's 2026-08-31 extension list, scoped per his
follow-up answer: "keep only pairs with three mice per group, with at
least 20 (good+mua) units in each area" -- 40 coarse pairs + 226 fine
pairs (computed and confirmed with Axel before launching; see
artifacts/pairs_final_{coarse,fine}.csv).

Changes from the 3-pair full-learning run (015):
- BOTH area hierarchy levels (coarse AND fine), all qualifying pairs at
  each, not 3 hand-picked coarse pairs.
- Single unit floor: >=20 good+MUA units/area (replaces the earlier
  tier-specific 10-good/30-good+MUA split -- good-only tier dropped).
- NO unit subsampling: use ALL available good+MUA units per area (removes
  the earlier 30-unit cap). Timed directly with synthetic data at the
  relevant real dimensions before launching (single fit: 0.23s at
  coarse-median size 519x260, 1.3s at the largest coarse size 1410x668,
  ~0.01s at fine-level sizes) -- this is why N_SHUFFLES is reduced below.
- N_SHUFFLES=100 (down from 300) -- a compute-budget call given the ~9x
  more pairs and the unit-count-driven per-fit cost increase for coarse
  pairs; estimated ~2 days total (vs ~4-5 days at 300 shuffles), stated
  to Axel before launching. Still ~0.01 p-value resolution.
- NEW: within-area (split-half) correlation control for every area that
  appears in >=1 qualifying pair -- randomly split that area's units into
  two halves, run the identical pipeline between the halves, as a
  same-population reference/ceiling against which between-area
  correlations can be judged.
- R+/R- colors fixed to #00B400 / #C800C8 throughout (Axel's standing
  convention, saved to memory).

Everything else (5ms bins, dead-zone exclusion, whisker x {0,1} /
auditory x 1 conditions, noise-correlation construction, the pCCA engine,
variants A/B/C, refit-per-shuffle null, pooled paired Wilcoxon+t-test
significance) is unchanged from 011/015.

Output: artifacts/all_pairs/{coarse,fine}/variant_{A,B,C}/results.csv,
artifacts/all_pairs/{coarse,fine}/within_area_results.csv.
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
from partial_CCA import PartialCCA

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ALL_PAIRS_DIR = ARTIFACTS_DIR / "all_pairs"
ALL_PAIRS_DIR.mkdir(exist_ok=True)

R_PLUS_COLOR, R_MINUS_COLOR = "#00B400", "#C800C8"

CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1)]
MIN_UNITS = 20
MIN_MICE_PER_GROUP = 3
MIN_TRIALS = 15
N_DIMS_MAX = 20
N_SHUFFLES = 100
REGULARIZATION = 1e-3
BIN_WIDTH = 0.005
WINDOW = cca_lib.WINDOW
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"

LEVELS = {"coarse": "area_group_coarse", "fine": "area_acronym_custom"}


def flatten(tensor: np.ndarray) -> np.ndarray:
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


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


def eligible_area_sessions(unit_table: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """(session_id, mouse_id, reward_group, area) rows with >=MIN_UNITS good+MUA units."""
    ut = unit_table[TIER_FN(unit_table)]
    piv = ut.groupby(["session_id", "mouse_id", "reward_group", area_col]).size().reset_index(name="n_units")
    return piv[piv["n_units"] >= MIN_UNITS]


def shuffle_null(X, resid_b_v, Z, n_dims, n_trials, rng_shuf):
    shuf_corrs = np.full((N_SHUFFLES, n_dims), np.nan)
    for s in range(N_SHUFFLES):
        perm = rng_shuf.permutation(n_trials)
        m_shuf = PartialCCA(regularization=REGULARIZATION).fit(X, flatten(resid_b_v[perm]), Z, verbose=False)
        shuf_corrs[s] = m_shuf.canonical_correlations_[:n_dims]
    return shuf_corrs.mean(axis=0), shuf_corrs.std(axis=0) / np.sqrt(N_SHUFFLES)


def run_level(level_name: str, area_col: str, pairs_df: pd.DataFrame, unit_table: pd.DataFrame,
              trial_table: pd.DataFrame, events: pd.DataFrame, rng_master: np.random.Generator) -> None:
    level_dir = ALL_PAIRS_DIR / level_name
    variant_dirs = {v: level_dir / f"variant_{v}" for v in ["A", "B", "C"]}
    for d in variant_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    elig = eligible_area_sessions(unit_table, area_col)
    bin_edges = cca_lib.time_bin_edges(WINDOW, BIN_WIDTH)

    rows = {v: [] for v in ["A", "B", "C"]}
    within_rows = []
    areas_done_within = set()

    n_pairs = len(pairs_df)
    for pair_i, prow in pairs_df.reset_index().iterrows():
        area_a, area_b = prow["area_a"], prow["area_b"]
        # sessions where BOTH areas are eligible
        sess_a = elig[elig[area_col] == area_a][["session_id", "mouse_id", "reward_group"]]
        sess_b = elig[elig[area_col] == area_b][["session_id", "mouse_id", "reward_group"]]
        both = sess_a.merge(sess_b, on=["session_id", "mouse_id", "reward_group"])

        for _, srow in both.iterrows():
            session_id, mouse_id, cohort = srow["session_id"], srow["mouse_id"], srow["reward_group"]
            trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
            lick_times = np.sort(events[(events["session_id"] == session_id)
                                         & (events["event_type"] == "piezo_lick_times")]["time"].to_numpy(dtype=float))

            units_a_full = unit_table[(unit_table["session_id"] == session_id) & (unit_table[area_col] == area_a) & TIER_FN(unit_table)]
            units_b_full = unit_table[(unit_table["session_id"] == session_id) & (unit_table[area_col] == area_b) & TIER_FN(unit_table)]
            spikes_a, spikes_b = spikes_of(units_a_full), spikes_of(units_b_full)

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
                                     & (~unit_table[area_col].isin([area_a, area_b]))]
                resid_other_v, lick_v = None, None
                if len(others) >= 5:
                    tensor_o = cca_lib.population_tensor(spikes_of(others), starts, is_whisker, bin_edges)
                    resid_other_v = cca_lib.noise_correlation_residuals(tensor_o)[:, valid_bins, :]
                if len(lick_times):
                    lick_reg = cca_lib.lick_rate_regressor(lick_times, starts, bin_edges)
                    lick_v = lick_reg[:, valid_bins, None]

                n_dims = min(N_DIMS_MAX, len(units_a_full), len(units_b_full))
                X = flatten(resid_a_v)

                for variant in ["A", "B", "C"]:
                    Z = build_Z(resid_other_v, lick_v, variant)
                    if variant != "A" and Z is None:
                        continue
                    Y = flatten(resid_b_v)
                    model = PartialCCA(regularization=REGULARIZATION).fit(X, Y, Z, verbose=False)
                    observed = model.canonical_correlations_[:n_dims]
                    rng_shuf = np.random.default_rng(
                        hash((level_name, session_id, area_a, area_b, trial_type, lick_flag, variant)) % (2**32))
                    null_mean, null_sem = shuffle_null(X, resid_b_v, Z, n_dims, resid_b_v.shape[0], rng_shuf)

                    for d in range(n_dims):
                        rows[variant].append({
                            "reward_group": cohort, "session_id": session_id, "mouse_id": mouse_id,
                            "area_a": area_a, "area_b": area_b, "trial_type": trial_type, "lick_flag": lick_flag,
                            "n_units_a": len(units_a_full), "n_units_b": len(units_b_full), "n_trials": len(starts),
                            "dimension": d + 1, "observed_corr": observed[d],
                            "shuffle_null_mean": null_mean[d], "shuffle_null_sem": null_sem[d],
                        })

                # Within-area split-half control -- once per (area, session, condition), not per pair
                # (an area appears in several pairs; only compute its within-area control the first time seen).
                for area, units_full, tensor_full in [(area_a, units_a_full, tensor_a), (area_b, units_b_full, tensor_b)]:
                    key = (level_name, area, session_id, trial_type, lick_flag)
                    if key in areas_done_within or len(units_full) < 2 * MIN_UNITS:
                        continue
                    areas_done_within.add(key)
                    idx = rng_master.permutation(len(units_full))
                    half = len(idx) // 2
                    resid_full_v = cca_lib.noise_correlation_residuals(tensor_full)[:, valid_bins, :]
                    Xw = flatten(resid_full_v[:, :, idx[:half]])
                    Yw = flatten(resid_full_v[:, :, idx[half:2 * half]])
                    n_dims_w = min(N_DIMS_MAX, half, len(idx) - half)
                    model_w = PartialCCA(regularization=REGULARIZATION).fit(Xw, Yw, None, verbose=False)
                    observed_w = model_w.canonical_correlations_[:n_dims_w]
                    rng_shuf_w = np.random.default_rng(hash((level_name, "within", area, session_id, trial_type, lick_flag)) % (2**32))
                    null_mean_w, null_sem_w = shuffle_null(Xw, resid_full_v[:, :, idx[half:2 * half]], None, n_dims_w, resid_full_v.shape[0], rng_shuf_w)
                    for d in range(n_dims_w):
                        within_rows.append({
                            "reward_group": cohort, "session_id": session_id, "mouse_id": mouse_id, "area": area,
                            "trial_type": trial_type, "lick_flag": lick_flag, "n_units_half": half,
                            "dimension": d + 1, "observed_corr": observed_w[d],
                            "shuffle_null_mean": null_mean_w[d], "shuffle_null_sem": null_sem_w[d],
                        })
        print(f"[{level_name}] pair {pair_i + 1}/{n_pairs} done ({area_a} vs {area_b}, {len(both)} session-instances)")

    for variant in ["A", "B", "C"]:
        df = pd.DataFrame(rows[variant])
        df.to_csv(variant_dirs[variant] / "results.csv", index=False)
        print(f"[{level_name}] wrote {len(df)} rows to variant_{variant}/results.csv")
    pd.DataFrame(within_rows).to_csv(level_dir / "within_area_results.csv", index=False)
    print(f"[{level_name}] wrote {len(within_rows)} rows to within_area_results.csv")


def main() -> None:
    pairs_coarse = pd.read_csv(ARTIFACTS_DIR / "pairs_final_coarse.csv")
    pairs_fine = pd.read_csv(ARTIFACTS_DIR / "pairs_final_fine.csv")
    all_areas_coarse = set(pairs_coarse["area_a"]) | set(pairs_coarse["area_b"])
    all_areas_fine = set(pairs_fine["area_a"]) | set(pairs_fine["area_b"])

    # Load every session that appears in the coverage report as having >=1
    # eligible pair at either level (metadata-only check against the cached
    # full_unit_table_metadata.parquet, no NWB needed for this selection step).
    cached = pd.read_parquet(ARTIFACTS_DIR / "full_unit_table_metadata.parquet")
    cached = cached[cached.day_stage == "learning"]
    cand_sessions = set(cached.loc[cached.area_group_coarse.isin(all_areas_coarse), "session_id"]) | \
        set(cached.loc[cached.area_acronym_custom.isin(all_areas_fine), "session_id"])
    files = sorted(f"{s}.nwb" for s in cand_sessions)
    print(f"Loading {len(files)} candidate learning-arm sessions...")

    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")

    rng_master = np.random.default_rng(2026831)

    run_level("coarse", LEVELS["coarse"], pairs_coarse, unit_table, trial_table, events, rng_master)
    run_level("fine", LEVELS["fine"], pairs_fine, unit_table, trial_table, events, rng_master)


if __name__ == "__main__":
    main()
