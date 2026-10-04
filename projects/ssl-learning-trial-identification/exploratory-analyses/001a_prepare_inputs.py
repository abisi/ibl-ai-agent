"""001a -- Build the exact per-session inputs of the learning-curve model
(run with the ibl-ai-agent venv; 001b fits in the behaviour_analysis conda
env, which has PyMC). For every ephys learning-stage (day 0) session with a
stored whisker curve:
  - whisker outcomes + start/stop times: curve-aligned active whisker trials
    (`_active_trials_from_whisker_onset_for_curve`) -- verified identical
    to the stored `outcomes` (length and values), else skipped.
  - no-stim outcomes + times: `_active_trials_for_curve_untrimmed` no-stim
    trials (matches the stored no-stim curve's trial count, 2026-09-19
    finding) -- checked against the stored no-stim `outcomes`, else skipped.
  - stored curves (p_mean/p_low/p_high/p_chance/learning_trial) for later
    comparison.
Output: artifacts/001_inputs.pkl (dict session_id -> dict).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_bwm_trial_prep import session_training_day  # noqa: E402
from ssl_timeresolved_decoding import (  # noqa: E402
    _active_trials_for_curve_untrimmed, _active_trials_from_whisker_onset_for_curve, hitmiss_session_list,
    load_whisker_curve_row,
)

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    root = resolve_dataset_dir("ssl_ephys")
    st = pd.read_parquet(root / "metadata" / "sessions.parquet")
    tt = pd.read_parquet(root / "metadata" / "trials.parquet")
    curve_root = axel_bisi_path("combined_results_ks4")
    sess = hitmiss_session_list(st)
    sess = sess[sess.day_stage == "learning"]
    out, log = {}, []
    for s in sess.itertuples():
        desc = st.loc[st.session_id == s.session_id, "session_description"].iloc[0]
        day = session_training_day(desc)
        wc = load_whisker_curve_row(curve_root, s.subject_id, day, "whisker_trial", interp=True)
        nc = load_whisker_curve_row(curve_root, s.subject_id, day, "no_stim_trial", interp=False)
        if wc is None or nc is None:
            log.append((s.session_id, "missing curve file"))
            continue
        a = _active_trials_from_whisker_onset_for_curve(s.session_id, tt)
        w = a[a.trial_type == "whisker_trial"]
        u = _active_trials_for_curve_untrimmed(s.session_id, tt)
        n = u[u.trial_type == "no_stim_trial"]
        wo, no = w.lick_flag.to_numpy().astype(int), n.lick_flag.to_numpy().astype(int)
        if len(wo) != len(wc["outcomes"]) or not np.array_equal(wo, np.asarray(wc["outcomes"]).astype(int)):
            log.append((s.session_id, f"whisker mismatch {len(wo)} vs {len(wc['outcomes'])}"))
            continue
        if len(no) != len(nc["outcomes"]) or not np.array_equal(no, np.asarray(nc["outcomes"]).astype(int)):
            log.append((s.session_id, f"no-stim mismatch {len(no)} vs {len(nc['outcomes'])}"))
            continue
        out[s.session_id] = dict(
            session_id=s.session_id, mouse_id=s.subject_id, reward_group=s.reward_group, learning_category=s.learning_category,
            w_outcomes=wo, w_start=w.start_time.to_numpy(), w_stop=w.stop_time.to_numpy() if "stop_time" in w else w.start_time.to_numpy(),
            n_outcomes=no, n_start=n.start_time.to_numpy(),
            stored=dict(p_mean=np.asarray(wc["p_mean"]), p_low=np.asarray(wc["p_low"]), p_high=np.asarray(wc["p_high"]),
                        p_chance=np.asarray(wc["p_chance"]), learning_trial=wc["learning_trial"], mouse_cat=wc["mouse_cat"],
                        fa_p_mean=np.asarray(nc["p_mean"])),
        )
    ART.mkdir(exist_ok=True)
    with open(ART / "001_inputs.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"{len(out)} sessions prepared; skipped: {log}")


if __name__ == "__main__":
    main()
