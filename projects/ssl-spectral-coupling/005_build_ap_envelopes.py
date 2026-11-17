"""Build AP-band envelope caches (ap_lib) for all sessions of 002's session table.

Per session: up to MAX_PER_COND (25) random trials of each condition W_lick, W_nolick, A_hit, NS_cr (same trial rules as
002), envelope [trial, 5-ms bin, channel] on the peak channels of the stable units, clock fitted per probe.
Outputs: OUT/ap/cache/<session>.npz (E float32, trial_index into sl.load_session trials), <session>_channels.parquet,
<session>_prov.json. Validity check of the method: 004_ap_check.py.
Run (haas): OMP_NUM_THREADS=1 nice -n 10 python projects/ssl-spectral-coupling/005_build_ap_envelopes.py [--n-proc 20]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ap_lib as al  # noqa: E402
import spec_lib as sl  # noqa: E402

CONDS = ["W_lick", "W_nolick", "A_hit", "NS_cr"]
CACHE = sl.OUT / "ap" / "cache"


def build(sid):
    from ephys_utilities.helpers import data_utils
    t0 = time.time()
    al.MAX_PER_COND = 25
    units, trials, rule = sl.load_session(sid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = data_utils.process_single_nwb(str(sl.cl.NWB_DIR / f"{sid}.nwb"), day_to_analyze="learning")["trial_table"]
    wst = full[full.trial_type == "whisker_trial"].start_time.to_numpy()
    sub = trials[trials.condition.isin(CONDS)]
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    E, chans, tsel, prov = al.build_session(sid, units, sub, wst, rng)
    tidx = sub.index.to_numpy()[tsel]
    tmp = CACHE / f"{sid}.tmp.npz"
    np.savez(tmp, E=E.astype(np.float32), trial_index=tidx, start_time=trials.start_time.to_numpy()[tidx])
    os.replace(tmp, CACHE / f"{sid}.npz")
    chans.to_parquet(CACHE / f"{sid}_channels.parquet")
    json.dump(dict(session_id=sid, unit_rule=rule, n_trials=int(len(tidx)), n_channels=int(len(chans)), probes=prov,
                   runtime_s=round(time.time() - t0)), open(CACHE / f"{sid}_prov.json", "w"), indent=1, default=str)
    return sid, E.shape, round(time.time() - t0)


def safe(sid):
    try:
        return build(sid)
    except Exception as err:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return sid, f"ERROR {type(err).__name__}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=20)
    ap.add_argument("--sessions", nargs="*")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("run002", HERE / "002_run_spikes.py")
    run002 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run002)
    s, _ = run002.session_table()
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    CACHE.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in CACHE.glob("*.npz") if not p.stem.endswith(".tmp")}
    # learners first (primary population), larger cohort balance
    s = s.sort_values("learner", ascending=False)
    jobs = [x for x in s.session_id if x not in done]
    print(f"{len(jobs)} sessions to build ({len(done)} done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
