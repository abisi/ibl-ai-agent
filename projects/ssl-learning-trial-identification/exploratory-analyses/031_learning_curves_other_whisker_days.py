"""031 -- Learning curves + single-session figures for every whisker session other than whisker_0 (user 2026-10-01: "run
for all non whisker_0 sessions the learning curves and the respective single session figures"; auditory days skipped --
user; free-licking days have no trial structure).
Sessions: every NWB_ks4 file whose session_description starts with "whisker" and is not "whisker_0" (whisker_+1..+4,
whisker_-1, whisker_on/off_*, whisker_psy_*).
Per session, the SAME method as day 0 (021/024): trial sets from ssl_timeresolved_decoding (whisker:
_active_trials_from_whisker_onset_for_curve; no-stim and auditory: _active_trials_for_curve_untrimmed; active context,
perf != 6), exact HMM curves (sigma = 1), FA and auditory fitted on their own trials and time-interpolated onto the
whisker-trial axis; figure = 021's figure() (square, ticks above = licks, below = no licks; whisker hits R+ green, misses
R- purple, FA black / CR grey, auditory blue / misses light blue). Reward group from the mouse reference.
Output folder: combined_results_ks4/<mouse>/<behaviour>_<day>/learning_curve/ (description with '+' removed, as the
existing folders: whisker_+1 -> whisker_1, whisker_on_1_+2 -> whisker_on_1_2), created if missing;
<mouse>_<folder>_learning_curves_exact_sigma1.{pdf,png,svg}. Curves (posterior mean, 80% band, outcomes, times) for every
session: artifacts/031_curves_other_whisker_days.pkl; log: artifacts/031_log.csv.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/031_learning_curves_other_whisker_days.py
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))


def folder_name(desc):
    return desc.replace("+", "")


def one(args):
    fpath, rg = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "scripts"))
    import lt_lib as L
    from pynwb import NWBHDF5IO
    from axel_bisi_paths import axel_bisi_root
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed as U
    from ssl_timeresolved_decoding import _active_trials_from_whisker_onset_for_curve as W
    m021 = importlib.import_module("021_learning_curve_figures")
    sid = Path(fpath).stem
    mouse = sid.split("_")[0]
    with NWBHDF5IO(fpath, "r") as io:
        n = io.read()
        desc = str(n.session_description).strip()
        t = n.trials.to_dataframe().assign(session_id=sid) if n.trials is not None else None
    if t is None or not len(t):
        return dict(session_id=sid, description=desc, status="no trials"), None
    a, u = W(sid, t), U(sid, t)
    w, ns, au = a[a.trial_type == "whisker_trial"], u[u.trial_type == "no_stim_trial"], u[u.trial_type == "auditory_trial"]
    if len(w) < 3:
        return dict(session_id=sid, description=desc, status=f"{len(w)} whisker trials"), None
    inp = dict(session_id=sid, mouse_id=mouse, reward_group=rg, w_outcomes=w.lick_flag.to_numpy().astype(int),
               w_start=w.start_time.to_numpy(), n_outcomes=ns.lick_flag.to_numpy().astype(int),
               n_start=ns.start_time.to_numpy())
    aud = (au.lick_flag.to_numpy().astype(int), au.start_time.to_numpy()) if len(au) else None
    out_dir = axel_bisi_root() / "combined_results_ks4" / mouse / folder_name(desc) / "learning_curve"
    if len(ns) < 3:
        return dict(session_id=sid, description=desc, status=f"{len(ns)} no-stim trials"), None
    m021.figure(inp, aud, out_dir, label=folder_name(desc))
    gw = L.forward_backward(inp["w_outcomes"], 1.0)[0]
    tw = inp["w_start"]
    curves = dict(whisker=gw, fa=L.interp_marginals(L.forward_backward(inp["n_outcomes"], 1.0)[0], inp["n_start"], tw))
    if aud is not None and len(aud[0]) >= 3:
        curves["auditory"] = L.interp_marginals(L.forward_backward(aud[0], 1.0)[0], aud[1], tw)
    summ = {k: dict(p_mean=g @ L.P_GRID, p_low80=L.quantiles(g, [0.1])[0], p_high80=L.quantiles(g, [0.9])[0])
            for k, g in curves.items()}
    rec = dict(inp, description=desc, folder=folder_name(desc), a_outcomes=None if aud is None else aud[0],
               a_start=None if aud is None else aud[1], curves=summ, sigma=1.0)
    return dict(session_id=sid, mouse_id=mouse, reward_group=rg, description=desc, n_whisker=len(w), n_nostim=len(ns),
                n_auditory=len(au), status="ok", figure=str(out_dir)), rec


def main():
    os.chdir(REPO)
    import h5py
    from axel_bisi_paths import axel_bisi_root
    nwb = axel_bisi_root() / "NWB_ks4"
    ref = pd.read_parquet(REPO / "reports" / "ssl_analysis" / "derived" / "mouse_reference.parquet").set_index("subject_id")
    jobs, log = [], []
    for f in sorted(os.listdir(nwb)):
        if not f.endswith(".nwb"):
            continue
        try:
            with h5py.File(nwb / f, "r") as h:
                d = h["session_description"][()]
                d = d.decode() if isinstance(d, bytes) else str(d)
        except Exception as e:  # noqa: BLE001
            log.append(dict(session_id=f[:-4], status=f"unreadable: {e!r}"[:120]))
            continue
        if not d.startswith("whisker") or d == "whisker_0":
            continue
        m = f.split("_")[0]
        rg = ref.loc[m, "reward_group"] if m in ref.index else None
        if rg not in ("R+", "R-"):
            log.append(dict(session_id=f[:-4], description=d, status=f"no R+/R- reward group ({rg})"))
            continue
        jobs.append((str(nwb / f), rg))
    print(len(jobs), "whisker sessions (not whisker_0)", flush=True)
    recs = {}
    with ProcessPoolExecutor(16) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                row, rec = fu.result()
            except Exception as e:  # noqa: BLE001
                row, rec = dict(status=f"error: {e!r}"[:200]), None
            log.append(row)
            if rec is not None:
                recs[rec["session_id"]] = rec
            if i % 25 == 0:
                print(i, "/", len(jobs), flush=True)
    pickle.dump(recs, open(ART / "031_curves_other_whisker_days.pkl", "wb"))
    lg = pd.DataFrame(log)
    lg.to_csv(ART / "031_log.csv", index=False)
    print(lg.status.value_counts().to_string())
    print(lg[lg.status == "ok"].groupby("description").size().to_string())


if __name__ == "__main__":
    main()
