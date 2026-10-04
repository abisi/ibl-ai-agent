"""Quantifies and corrects the session-pooling caveat flagged throughout
the cross-pair "overall communication by cohort" analyses (030/031/033),
per Axel's 2026-09-01 "Let's treat the caveat. Explain why it is one and
what is the quantification associated?"

The caveat: those analyses pooled one value per (session, pair) INSTANCE,
not one value per session -- a session that qualifies for multiple area
pairs (common: a session with >=20 units in 3+ areas contributes to every
pair among them) gets counted once per pair it appears in. Mann-Whitney/
Welch assume independent observations; repeated instances from the same
session violate that, inflating the effective N and anti-conservatively
shrinking p-values below what the TRUE number of independent sessions
would support.

This script quantifies the inflation (instances / unique sessions, per
cohort/condition) and reruns the R+ vs R- test PROPERLY: one value per
UNIQUE session (averaged across whichever pairs that session contributed
to, within a given condition/variant/window), for both the coarse
(35-pair) and fine (82-pair) analyses. Reports old (instance-level) vs
new (session-deduplicated) p-values side by side so the caveat's actual
impact is visible, not just asserted.
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
driver = importlib.import_module("026_all_pairs_driver")
fine_driver = importlib.import_module("032_all_fine_pairs_driver")
cov_lib = importlib.import_module("000_coverage_lib")

import numpy as np
import pandas as pd
from scipy import stats

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CACHE_DIR = ARTIFACTS_DIR / "cache"
OUT_DIR = RUNS_DIR / "cross_pair_summary"

CONDITIONS = fig024.CONDITIONS
MIN_UNITS = fig024.MIN_UNITS
MIN_SESSIONS_PER_COHORT = 5
WINDOWS = ["baseline", "sensory"]


def _stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def units_fingerprint(session_ids: list[str]) -> str:
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    ref_fp = _stable_hash(ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group"]].astype(str).values.tolist())
    return _stable_hash({"session_ids": sorted(session_ids), "day_to_analyze": "learning",
                          "quality_tiers": ["good", "mua"], "ref_fp": ref_fp})


def load_pair_main_pass(area_a: str, area_b: str, session_ids: list[str], area_column: str,
                         include_area_column_in_fp: bool) -> dict | None:
    """`include_area_column_in_fp=False` for coarse pairs: those caches
    were written by `026_all_pairs_driver.py` BEFORE `AREA_COLUMN` was
    added to `024`'s fingerprint (added specifically for the fine-level
    work) -- so the coarse cache filenames don't have that field baked
    in. Fine pairs (written by `032`, after that addition) DO need it."""
    units_fp = units_fingerprint(session_ids)
    fp_dict = {"units_fp": units_fp, "area_a": area_a, "area_b": area_b, "min_units": MIN_UNITS}
    if include_area_column_in_fp:
        fp_dict["area_column"] = area_column
    pair_fp = _stable_hash(fp_dict)
    params_fp = fig024.main_pass_params_fingerprint()
    cache_file = CACHE_DIR / f"main_pass_{pair_fp}_{params_fp}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def coarse_pairs():
    pairs_df, session_lists = driver.derive_session_lists()
    ut = pd.read_parquet(ARTIFACTS_DIR / "full_unit_table_metadata_v3.parquet")
    ut = ut[(ut["exclude"] == 0) & (ut["exclude_ephys"] == 0) & (ut["reward_group"].isin(["R+", "R-"]))
            & (ut["quality_label"].isin(["good", "mua"])) & (ut["day_stage"] == "learning")]
    rg_lookup = ut.drop_duplicates("session_id").set_index("session_id")["reward_group"]
    pair_order = []
    for _, row in pairs_df.iterrows():
        a, b = row["area_a"], row["area_b"]
        sess = session_lists[(a, b)]
        n_p = (rg_lookup.reindex(sess) == "R+").sum()
        n_m = (rg_lookup.reindex(sess) == "R-").sum()
        if n_p >= MIN_SESSIONS_PER_COHORT and n_m >= MIN_SESSIONS_PER_COHORT:
            pair_order.append((a, b))
    return pair_order, session_lists, "area_group_coarse_v3"


def fine_pairs():
    pairs, session_lists = fine_driver.derive_fine_pairs()
    return pairs, session_lists, "area_acronym_custom"


def analyze(level_name: str, pair_order, session_lists, area_column, include_area_column_in_fp: bool):
    print(f"\n{'='*70}\n{level_name} ({len(pair_order)} pairs)\n{'='*70}")
    pair_data = {}
    for a, b in pair_order:
        d = load_pair_main_pass(a, b, session_lists[(a, b)], area_column, include_area_column_in_fp)
        if d is not None:
            pair_data[(a, b)] = d
    print(f"  {len(pair_data)}/{len(pair_order)} pairs loaded from cache")

    bin_starts = cca_lib.sliding_window_starts()
    bin_centers = cca_lib.sliding_window_centers(bin_starts)
    t_ms = bin_centers * 1000
    win_masks_by_cond = {}
    for cond in CONDITIONS:
        is_whisker = cond[0] == "whisker_trial"
        baseline_mask, sensory_mask = fig024.window_masks(t_ms, is_whisker)
        win_masks_by_cond[cond] = {"baseline": baseline_mask, "sensory": sensory_mask}

    rows = []
    for window in WINDOWS:
        for variant in ("A", "B"):
            for cond in CONDITIONS:
                mask = win_masks_by_cond[cond][window]
                # session_vals[cohort][session_id] = list of this session's
                # per-pair values for this (window, variant, cond) -- one
                # entry per pair the session contributed to.
                session_vals = {"R+": {}, "R-": {}}
                n_instances = {"R+": 0, "R-": 0}
                for (a, b), data in pair_data.items():
                    if cond not in data:
                        continue
                    d = data[cond]["A" if variant == "A" else "B"]
                    rg = np.array(d["reward_group"])
                    sids = d["session_id"]
                    for i, sid in enumerate(sids):
                        cohort = rg[i]
                        if cohort not in ("R+", "R-"):
                            continue
                        v = np.nanmean(np.abs(d["corr_t"][i][mask]))
                        session_vals[cohort].setdefault(sid, []).append(v)
                        n_instances[cohort] += 1

                # OLD (instance-level, pseudo-replicated): flatten all instances.
                old_p = np.concatenate([np.array(v) for v in session_vals["R+"].values()]) if session_vals["R+"] else np.array([])
                old_m = np.concatenate([np.array(v) for v in session_vals["R-"].values()]) if session_vals["R-"] else np.array([])
                # NEW (session-deduplicated): one value per unique session
                # (averaged across whichever pairs it appeared in).
                new_p = np.array([np.nanmean(v) for v in session_vals["R+"].values()])
                new_m = np.array([np.nanmean(v) for v in session_vals["R-"].values()])

                if len(old_p) > 1 and len(old_m) > 1:
                    _, p_old_mw = stats.mannwhitneyu(old_p, old_m, alternative="two-sided")
                    _, p_old_welch = stats.ttest_ind(old_p, old_m, equal_var=False)
                else:
                    p_old_mw = p_old_welch = np.nan
                if len(new_p) > 1 and len(new_m) > 1:
                    _, p_new_mw = stats.mannwhitneyu(new_p, new_m, alternative="two-sided")
                    _, p_new_welch = stats.ttest_ind(new_p, new_m, equal_var=False)
                else:
                    p_new_mw = p_new_welch = np.nan

                n_unique_p, n_unique_m = len(session_vals["R+"]), len(session_vals["R-"])
                rows.append({
                    "level": level_name, "window": window, "variant": variant,
                    "condition": f"{cond[0]}(lick={cond[1]})",
                    "n_instances_Rplus": n_instances["R+"], "n_unique_sessions_Rplus": n_unique_p,
                    "inflation_Rplus": round(n_instances["R+"] / n_unique_p, 2) if n_unique_p else np.nan,
                    "n_instances_Rminus": n_instances["R-"], "n_unique_sessions_Rminus": n_unique_m,
                    "inflation_Rminus": round(n_instances["R-"] / n_unique_m, 2) if n_unique_m else np.nan,
                    "mean_Rplus_dedup": np.nanmean(new_p) if len(new_p) else np.nan,
                    "mean_Rminus_dedup": np.nanmean(new_m) if len(new_m) else np.nan,
                    "p_instance_mw": p_old_mw, "p_instance_welch": p_old_welch,
                    "p_dedup_mw": p_new_mw, "p_dedup_welch": p_new_welch,
                })
    return pd.DataFrame(rows)


def main() -> None:
    c_pairs, c_sessions, c_col = coarse_pairs()

    # 2026-09-05: fine batch NOT rerun with the current (10ms/5ms bins + causal smoothing +
    # 0.5Hz rate filter) pipeline -- its caches are stale relative to today's coarse rerun,
    # so mixing them in here would silently blend two different methodologies. Coarse only.
    # Also: include_area_column_in_fp=True now (was False) -- today's coarse batch was
    # computed with the CURRENT 024, which puts area_column in its own fingerprint (unlike
    # the much earlier 026-driven batch this flag was originally written for).
    combined = analyze("coarse", c_pairs, c_sessions, c_col, include_area_column_in_fp=True)

    out_csv = OUT_DIR / "deduplication_check.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    sub = combined[["window", "variant", "condition", "inflation_Rplus", "inflation_Rminus",
                     "p_instance_mw", "p_dedup_mw", "p_instance_welch", "p_dedup_welch"]]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
