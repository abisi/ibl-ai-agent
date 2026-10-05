"""137b -- Build the shared Part III unit list (tracked_units.py): per learning-stage session with passive pre AND post, the 137
stable units firing >= 0.5 Hz in passive pre, passive post and both active halves at every cut point (middle trial, middle
whisker trial, hit median). Columns: mouse_id, session_id, electrode_group, cluster_id, area_group, quality_label, stable
(tracked stable), good (tracked AND quality good). Sessions without passive pre + post are absent.
Output: combined_results_ks4/<slug>/tables/137b_tracked_units.parquet (+ per-session counts printed)
Run (haas, repo root): python .../137b_tracked_units.py
"""

from __future__ import annotations

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS); sys.path.insert(0, str(OUT))
import tracked_units as TU  # noqa: E402

N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "30"))


def process(args):
    sid, cands = args
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, SCRIPTS); sys.path.insert(0, str(OUT))
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return sid, None
    segs = TU.segments(trs)
    spikes = T.load_session_unit_spikes(root, sid)
    keep = [c for c in cands if TU.is_tracked(spikes.get(c, spikes.get(int(c), np.array([]))), segs)]
    return sid, np.asarray(keep, dtype=np.int64)


def main():
    os.chdir(OUT.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    S = pd.read_parquet(TU.table_path().with_name("137_stable_units.parquet"))
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    S = S[S.session_id.isin(sess.session_id) & S.stable]
    args = [(sid, g.cluster_id.to_numpy()) for sid, g in S.groupby("session_id")]
    print(f"[137b] {len(args)} sessions", flush=True)
    keep = {}
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for sid, k in ex.map(process, args):
            if k is not None:
                keep[sid] = k
    rows = [S[(S.session_id == sid) & S.cluster_id.isin(k)] for sid, k in keep.items()]
    out = pd.concat(rows, ignore_index=True)[["mouse_id", "session_id", "electrode_group", "cluster_id", "area_group", "quality_label",
                                              "stable", "good"]]
    out["stable"] = True                       # tracked stable
    out = out.merge(sess[["session_id", "reward_group"]], on="session_id", how="left")
    tmp = TU.table_path().with_suffix(".partial.parquet"); out.to_parquet(tmp, index=False); os.replace(tmp, TU.table_path())
    n = out.groupby(["reward_group", "session_id"]).agg(stable=("stable", "sum"), good=("good", "sum")).reset_index()
    print(n.groupby("reward_group")[["stable", "good"]].describe().round(0).T.to_string(), flush=True)
    print(f"[137b] DONE {len(keep)} sessions, {len(out)} tracked stable units, {int(out.good.sum())} tracked good", flush=True)


if __name__ == "__main__":
    main()
