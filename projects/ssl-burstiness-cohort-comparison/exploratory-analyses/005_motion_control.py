"""Motion/licking control: correlate session-level continuous burstiness
against orofacial motion (DLC keypoint velocity, ssl_behavior) and licking
(piezo_lick_times events, ssl_ephys) -- per ../question.md.

Uses the compressed ssl_ephys/ssl_behavior datasets (not raw NWB) for this
control only -- appropriate per ssl_loading_policy.md's Core policy, since
this is a per-session summary well within the compressed schema's scope, not
part of the mandatory Path B burstiness computation itself.

Run with the repo's own .venv (needs ibl_ai_agent's own dataset modules,
not pynwb/ephys_utilities):
    .venv\\Scripts\\python.exe projects\\ssl-burstiness-cohort-comparison\\exploratory-analyses\\005_motion_control.py
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


def session_lick_rate() -> pd.DataFrame:
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")
    sessions = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "sessions.parquet")
    licks = events[events["event_type"] == "piezo_lick_times"]
    lick_counts = licks.groupby("session_id").size().rename("n_licks").reset_index()
    out = sessions[["session_id", "duration_s"]].merge(lick_counts, on="session_id", how="left")
    out["n_licks"] = out["n_licks"].fillna(0)
    out["lick_rate_hz"] = out["n_licks"] / out["duration_s"]
    return out[["session_id", "lick_rate_hz"]]


def session_motion_energy(session_id: str) -> float | None:
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
        values = shard[matches[0]]
        finite = values[np.isfinite(values)]
        if len(finite):
            per_keypoint.append(np.mean(np.abs(finite)))
    return float(np.mean(per_keypoint)) if per_keypoint else None


def main() -> None:
    lick_df = session_lick_rate()

    rows = []
    for day_stage in ["learning", "expert"]:
        metrics = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_unit_metrics.parquet")
        session_ids = metrics["session_id"].unique()
        print(f"{day_stage}: {len(session_ids)} sessions")
        motion = {sid: session_motion_energy(sid) for sid in session_ids}
        n_with_motion = sum(v is not None for v in motion.values())
        print(f"  motion energy available for {n_with_motion}/{len(session_ids)} sessions")

        for tier_name, tier_filter in [("good", metrics["quality_label"] == "good"),
                                        ("good_mua", metrics["quality_label"].isin(["good", "mua"]))]:
            tiered = metrics[tier_filter]
            per_session = tiered.groupby("session_id").agg(
                median_continuous_burstiness=("continuous_burstiness", "median"),
                n_units=("continuous_burstiness", "size"),
            ).reset_index()
            per_session["orofacial_motion_energy"] = per_session["session_id"].map(motion)
            per_session = per_session.merge(lick_df, on="session_id", how="left")
            per_session["day_stage"] = day_stage
            per_session["tier"] = tier_name
            rows.append(per_session)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(ARTIFACTS_DIR / "motion_control_per_session.parquet", index=False)

    print("\nSpearman correlations (session-level median continuous_burstiness vs. motion/licking):")
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
    corr_df.to_csv(ARTIFACTS_DIR / "motion_control_correlations.csv", index=False)
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
