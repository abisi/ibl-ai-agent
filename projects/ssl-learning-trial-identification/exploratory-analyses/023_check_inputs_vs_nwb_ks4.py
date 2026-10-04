"""023 -- Check that the learning-curve inputs match the NWB_ks4 files (user 2026-09-30: "make sure the data matches
NWB_ks4 NWB files"). For every session in 001 (ephys, trials from ssl_ephys) and 022 (behaviour-only, from NWB_ks4):
read M:/analysis/Axel_Bisi/NWB_ks4/<session_id>.nwb, apply the curve trial selection (whisker:
_active_trials_from_whisker_onset_for_curve; no-stim and auditory: _active_trials_for_curve_untrimmed) and compare per
trial type: count, lick outcomes (exact), start times (max |diff| < 1 ms). Auditory for 001 sessions is compared with the
ssl_ephys trials table (the figure's source).
Output: artifacts/023_inputs_vs_nwb_ks4.csv, printed summary.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/023_check_inputs_vs_nwb_ks4.py
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")


def main():
    os.chdir(REPO)
    from pynwb import NWBHDF5IO
    from axel_bisi_paths import axel_bisi_root
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed as U
    from ssl_timeresolved_decoding import _active_trials_from_whisker_onset_for_curve as W
    nwb = axel_bisi_root() / "NWB_ks4"
    tt = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "trials.parquet")
    inputs = {**{k: dict(v, _src="001 ephys") for k, v in pickle.load(open(ART / "001_inputs.pkl", "rb")).items()},
              **{k: dict(v, _src="022 NWB_ks4") for k, v in pickle.load(open(ART / "022_inputs_behaviour_only.pkl", "rb")).items()}}
    rows = []
    for sid, inp in inputs.items():
        f = nwb / f"{sid}.nwb"
        r = dict(session_id=sid, mouse_id=inp["mouse_id"], source=inp["_src"], nwb_exists=f.exists())
        if not f.exists():
            rows.append(r)
            continue
        with NWBHDF5IO(str(f), "r") as io:
            n = io.read()
            r["nwb_description"] = str(n.session_description)
            t = n.trials.to_dataframe().assign(session_id=sid)
        a, u = W(sid, t), U(sid, t)
        nw = {"whisker": a[a.trial_type == "whisker_trial"], "no_stim": u[u.trial_type == "no_stim_trial"],
              "auditory": u[u.trial_type == "auditory_trial"]}
        if "a_outcomes" in inp:
            aud = (inp["a_outcomes"], inp["a_start"])
        else:
            ue = U(sid, tt)
            ue = ue[ue.trial_type == "auditory_trial"]
            aud = (ue.lick_flag.to_numpy().astype(int), ue.start_time.to_numpy())
        mine = {"whisker": (inp["w_outcomes"], inp["w_start"]), "no_stim": (inp["n_outcomes"], inp["n_start"]), "auditory": aud}
        ok_all = True
        for k in nw:
            yo, to = np.asarray(mine[k][0], int), np.asarray(mine[k][1], float)
            yn, tn = nw[k].lick_flag.to_numpy().astype(int), nw[k].start_time.to_numpy().astype(float)
            same_n = len(yo) == len(yn)
            same_y = same_n and np.array_equal(yo, yn)
            dt = float(np.max(np.abs(to - tn))) if same_n and len(yo) else np.nan
            ok = same_y and (not np.isfinite(dt) or dt < 1e-3)
            ok_all &= ok
            r.update({f"{k}_n_input": len(yo), f"{k}_n_nwb": len(yn), f"{k}_outcomes_equal": same_y, f"{k}_max_dt_s": dt,
                      f"{k}_ok": ok})
        r["all_ok"] = ok_all
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(ART / "023_inputs_vs_nwb_ks4.csv", index=False)
    print(len(d), "sessions |", int(d.nwb_exists.sum()), "with an NWB_ks4 file |", int(d.get("all_ok", pd.Series(dtype=bool)).fillna(False).sum()), "fully matching")
    bad = d[~d.get("all_ok", pd.Series(False, index=d.index)).fillna(False).astype(bool)]
    cols = ["session_id", "source", "nwb_exists"] + [c for c in d.columns if c.endswith(("_n_input", "_n_nwb", "_ok", "_max_dt_s"))]
    pd.set_option("display.width", 250)
    print(bad[[c for c in cols if c in bad]].to_string(index=False))


if __name__ == "__main__":
    main()
