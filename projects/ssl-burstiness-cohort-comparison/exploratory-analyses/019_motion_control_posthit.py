"""Motion/licking control, restricted to post-first-hit data -- direct
answer to "does bursting correlate with movement": correlate session-level
median continuous_burstiness (post-hit) against orofacial motion (DLC
keypoint velocity, ssl_behavior) and licking (piezo_lick_times,
ssl_ephys), both ALSO restricted to time >= that session's first-hit time,
for a fair like-for-like comparison. Uses the compressed ssl_ephys dataset's
own trials.parquet for first-hit detection (independent of our Path B
loader, but same underlying NWB fields) -- avoids re-loading raw NWB a
second time just for trial timing, consistent with ssl_loading_policy.md's
local-first rule for this side-analysis.

Run with the repo's own .venv:
    .venv\\Scripts\\python.exe projects\\ssl-burstiness-cohort-comparison\\exploratory-analyses\\019_motion_control_posthit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ibl_ai_agent.datasets.ssl_behavior import load_tracking_shard

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"
SSL_BEHAVIOR_TRACKING_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_behavior" / "1.0.0" / "tracking"
KEYPOINTS = ["jaw_velocity", "whisker_velocity", "nose_velocity"]


def _normalize_context(sess: pd.DataFrame) -> pd.Series:
    """The compressed ssl_ephys dataset's `context` column is the literal
    string "nan" (not a real null) for every row of subjects below AB116
    (which never had passive epochs recorded in the raw NWB, so `context`
    was never populated) -- found directly: AB080's session has context=="nan"
    for all 347 rows, silently zeroing out any `context=='active'` filter.
    Path B's own process_single_nwb normalizes this (an all-"nan" session ->
    'active'; a mixed session -> fillna/replace 'nan'->'active'); the
    compressed dataset does not apply that normalization, so it must be
    replicated here rather than trusting the raw column."""
    ctx = sess["context"].astype(str)
    if (ctx == "nan").all():
        return pd.Series("active", index=sess.index)
    return ctx.replace("nan", "active")


def session_first_hit_times(reward_group_by_session: dict) -> dict:
    trials = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "trials.parquet")
    out = {}
    for session_id, rg in reward_group_by_session.items():
        sess = trials[trials.session_id == session_id].copy()
        sess["context"] = _normalize_context(sess)
        wh = sess[(sess.trial_type == "whisker_trial") & (sess.context == "active")]
        hit_mask = (wh.lick_flag == 1) if rg == "R+" else (wh.lick_flag == 0)
        hits = wh[hit_mask].sort_values("trial_id")
        out[session_id] = float(hits["start_time"].iloc[0]) if len(hits) else None
    return out


def session_lick_rate_posthit(session_id: str, t_restrict: float) -> float | None:
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")
    sessions = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "sessions.parquet")
    t_end = sessions.loc[sessions.session_id == session_id, "duration_s"]
    if len(t_end) == 0:
        return None
    t_end = float(t_end.iloc[0])
    licks = events[(events.session_id == session_id) & (events.event_type == "piezo_lick_times")]
    licks_post = licks[licks.time >= t_restrict]
    duration = t_end - t_restrict
    if duration <= 0:
        return None
    return len(licks_post) / duration


def session_motion_energy_posthit(session_id: str, t_restrict: float) -> float | None:
    shard_path = SSL_BEHAVIOR_TRACKING_DIR / session_id
    if not shard_path.exists():
        return None
    try:
        shard = load_tracking_shard(shard_path)
    except Exception:
        return None
    per_keypoint = []
    for kp in KEYPOINTS:
        matches = [k for k in shard if k.endswith(f".{kp}")]
        if not matches:
            continue
        key = matches[0]
        group_key = key.rsplit(".", 1)[0]
        ts_key = f"{group_key}.timestamps_seconds"
        if ts_key not in shard:
            continue
        values = shard[key]
        timestamps = shard[ts_key]
        n = min(len(values), len(timestamps))
        values, timestamps = values[:n], timestamps[:n]
        mask = timestamps >= t_restrict
        finite = values[mask][np.isfinite(values[mask])]
        if len(finite):
            per_keypoint.append(np.mean(np.abs(finite)))
    return float(np.mean(per_keypoint)) if per_keypoint else None


def main() -> None:
    rows = []
    for day_stage in ["learning", "expert"]:
        metrics = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics_posthit.parquet")
        session_reward = metrics.drop_duplicates("session_id").set_index("session_id")["reward_group"].to_dict()
        hit_times = session_first_hit_times(session_reward)
        n_no_hit = sum(v is None for v in hit_times.values())
        print(f"{day_stage}: {len(hit_times)} sessions, {n_no_hit} with no qualifying hit (excluded from motion control too)")

        for tier_name, tier_filter in [("good", metrics["quality_label"] == "good"),
                                        ("good_mua", metrics["quality_label"].isin(["good", "mua"]))]:
            tiered = metrics[tier_filter]
            per_session = tiered.groupby("session_id").agg(
                median_continuous_burstiness=("continuous_burstiness", "median"),
                n_units=("continuous_burstiness", "size"),
                reward_group=("reward_group", "first"),
            ).reset_index()
            motion, lick_rate = {}, {}
            for sid in per_session["session_id"]:
                t_restrict = hit_times.get(sid)
                if t_restrict is None:
                    continue
                motion[sid] = session_motion_energy_posthit(sid, t_restrict)
                lick_rate[sid] = session_lick_rate_posthit(sid, t_restrict)
            # .map() with None-valued dict entries produces an object-dtype
            # column (not a proper float NaN), which crashes scipy's
            # spearmanr (found directly: AttributeError deep in np.average,
            # only on the fresh in-memory run -- a parquet round-trip
            # silently coerces to float64/NaN, masking it on reload).
            # Cast explicitly rather than relying on that side effect.
            per_session["orofacial_motion_energy"] = pd.to_numeric(per_session["session_id"].map(motion), errors="coerce")
            per_session["lick_rate_hz"] = pd.to_numeric(per_session["session_id"].map(lick_rate), errors="coerce")
            per_session["day_stage"] = day_stage
            per_session["tier"] = tier_name
            rows.append(per_session)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(ARTIFACTS_DIR / "motion_control_posthit_per_session.parquet", index=False)

    print("\nSpearman correlations (post-hit median continuous_burstiness vs. motion/licking):")
    corr_rows = []
    for day_stage in ["learning", "expert"]:
        for tier in ["good", "good_mua"]:
            sub = combined[(combined.day_stage == day_stage) & (combined.tier == tier)]
            for var in ["orofacial_motion_energy", "lick_rate_hz"]:
                valid = sub.dropna(subset=["median_continuous_burstiness", var])
                if len(valid) < 4:
                    continue
                rho, p = scipy_stats.spearmanr(valid["median_continuous_burstiness"], valid[var])
                corr_rows.append({"day_stage": day_stage, "tier": tier, "variable": var,
                                   "n_sessions": len(valid), "spearman_rho": rho, "p_value": p})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(ARTIFACTS_DIR / "motion_control_posthit_correlations.csv", index=False)
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
