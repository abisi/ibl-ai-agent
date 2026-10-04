"""022 -- Learning-curve inputs for mice NOT in the ephys learning-session set (user 2026-09-30: "Include the ... mice
in the ssl dataset, they are not ephys, but behavioural analyses should also include them").
NWB source: NWB_ks4 only (user). A stored curve that equals the NWB set minus its last trial is accepted and
flagged. Mice = (ssl_behavior subjects U mice with a stored combined_results_ks4/<mouse>/whisker_0 learning curve) minus the 88
ephys learning sessions of 001. For each: the whisker day-0 NWB (session_description 'whisker_0'; M:/analysis/Axel_Bisi/
NWB), trials table, the SAME trial selection as the curve pipeline (ssl_timeresolved_decoding
_active_trials_from_whisker_onset_for_curve for whisker trials, _active_trials_for_curve_untrimmed for no-stim and
auditory trials: active context, perf != 6), and a check against the stored curve outcomes where a stored curve file
exists (whisker, no-stim, auditory: length and values). Reward group / learning category from the mouse reference
(reports/ssl_analysis/derived/mouse_reference.parquet).
Output: artifacts/022_inputs_behaviour_only.pkl (session_id -> dict, same keys as 001 + a_outcomes, a_start, source,
stored_match), printed per-mouse log.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/022_behaviour_only_inputs.py
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


def stored_outcomes(curve_dir, mouse, ttype):
    f = curve_dir / f"{mouse}_whisker_0_{ttype}_learning_curve.h5"
    if not f.exists():
        return None
    return np.asarray(pd.read_hdf(f).iloc[0]["outcomes"]).astype(int)


def main():
    os.chdir(REPO)
    from pynwb import NWBHDF5IO
    from axel_bisi_paths import axel_bisi_root
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed, _active_trials_from_whisker_onset_for_curve
    root = axel_bisi_root()
    beh = set(pd.read_parquet(resolve_dataset_dir("ssl_behavior") / "metadata" / "sessions.parquet").subject_id)
    cr = root / "combined_results_ks4"
    curve = {m for m in os.listdir(cr) if (cr / m / "whisker_0" / "learning_curve" / f"{m}_whisker_0_whisker_trial_learning_curve.h5").exists()}
    done = {v["mouse_id"] for v in pickle.load(open(ART / "001_inputs.pkl", "rb")).values()}
    ref = pd.read_parquet(REPO / "reports" / "ssl_analysis" / "derived" / "mouse_reference.parquet").set_index("subject_id")
    mice = sorted((beh | curve) - done)
    print(len(mice), "mice:", mice)
    # NWB folders searched in order (user's shares: behaviour NWBs are split across these); every whisker_0 file found is
    # a candidate; with a stored curve, the first candidate whose outcomes match it exactly is used
    nwb_dirs = [root / "NWB_ks4"]                  # user 2026-09-30: only NWB_ks4
    out, log = {}, []
    for m in mice:
        cands, seen = [], set()
        for nd in nwb_dirs:
            if not nd.exists():
                continue
            for f in sorted(x for x in os.listdir(nd) if x.startswith(m + "_") and x.endswith(".nwb")):
                try:
                    with NWBHDF5IO(str(nd / f), "r") as io:
                        n = io.read()
                        if str(n.session_description).strip() == "whisker_0" and (f, len(n.trials or [])) not in seen:
                            seen.add((f, len(n.trials or [])))
                            cands.append((nd.name, f, n.trials.to_dataframe() if n.trials is not None else pd.DataFrame()))
                except Exception as e:  # noqa: BLE001
                    log.append((m, f"unreadable {nd.name}/{f}: {e!r}"[:160]))
        if not cands:
            log.append((m, "no whisker_0 NWB in " + ", ".join(d.name for d in nwb_dirs)))
            continue
        cdir = cr / m / "whisker_0" / "learning_curve"
        chosen, tried = None, []
        for dname, f, tr in cands:
            if not len(tr):
                tried.append(f"{dname}/{f}: no trials")
                continue
            sid = f[:-4]
            tr = tr.assign(session_id=sid)
            a = _active_trials_from_whisker_onset_for_curve(sid, tr)
            u = _active_trials_for_curve_untrimmed(sid, tr)
            w, ns, au = a[a.trial_type == "whisker_trial"], u[u.trial_type == "no_stim_trial"], u[u.trial_type == "auditory_trial"]
            match = {}
            for tt_, df in {"whisker_trial": w, "no_stim_trial": ns, "auditory_trial": au}.items():
                so = stored_outcomes(cdir, m, tt_)
                mine = df.lick_flag.to_numpy().astype(int)
                if so is None:
                    match[tt_] = None
                elif len(so) == len(mine) and np.array_equal(so, mine):
                    match[tt_] = True
                elif len(so) == len(mine) - 1 and np.array_equal(so, mine[:-1]):
                    match[tt_] = "stored lacks the last trial"   # accepted, flagged (AB105 whisker, 2026-09-30)
                else:
                    match[tt_] = False
            tried.append(f"{dname}/{f}: whisker {len(w)}, no-stim {len(ns)}, auditory {len(au)}, match {match}")
            if not any(v is False for v in match.values()):
                chosen = (dname, f, sid, w, ns, au, match)
                break
        if chosen is None:
            log.append((m, "no candidate matches the stored curve: " + " | ".join(tried)))
            continue
        dname, f, sid, w, ns, au, match = chosen
        f = f"{dname}/{f}"
        rg = ref.loc[m, "reward_group"] if m in ref.index else None
        if rg not in ("R+", "R-"):
            log.append((m, f"no reward group ({rg})"))
            continue
        out[sid] = dict(session_id=sid, mouse_id=m, reward_group=rg,
                        learning_category=ref.loc[m, "learning_category"] if m in ref.index else "",
                        w_outcomes=w.lick_flag.to_numpy().astype(int), w_start=w.start_time.to_numpy(),
                        w_stop=w.stop_time.to_numpy(), n_outcomes=ns.lick_flag.to_numpy().astype(int),
                        n_start=ns.start_time.to_numpy(), a_outcomes=au.lick_flag.to_numpy().astype(int),
                        a_start=au.start_time.to_numpy(), source=f"NWB {f}", stored_match=match)
        log.append((m, f"ok {sid}: whisker {len(w)}, no-stim {len(ns)}, auditory {len(au)}; stored match {match}"))
    pickle.dump(out, open(ART / "022_inputs_behaviour_only.pkl", "wb"))
    for m, msg in log:
        print(m, msg)
    print(len(out), "sessions saved")


if __name__ == "__main__":
    main()
