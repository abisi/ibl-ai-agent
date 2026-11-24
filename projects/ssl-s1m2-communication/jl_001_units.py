"""Jules Lebert's context dataset (S1M2_DATASET=jl): unit table with SSL area labels and stability, plus a per-session
trial summary. Every NWB in /mnt/lsens-analysis/Jules_Lebert/NWB that holds a units table is read.

Units: area_acronym_custom from ephys_utilities.allen_utils.process_allen_labels(split_merge_areas=True), i.e. exactly
the SSL definition (Allen atlas-space labels ccf_atlas_*, layered labels -> parent; MOs / MOp units on probes targeted
at wM1 / wM2 -> MO-wM1 / MO-wM2). Unit set = bombcell bc_label good or mua (non-soma excluded, as SSL unit_set_all).
Stability as SSL 137 where possible: presence_ratio >= 0.5 (60-s bins) and coverage_ratio >= 0.9
(unit_metrics_utils.compute_presence_coverage_metrics, over all units of the session); no DREDge drift test exists for
this dataset, so that third SSL criterion is not applied.

Outputs (OUT = combined_results_ks4/ssl-s1m2-communication/jules_lebert):
  tables/jl_units.parquet          one row per unit (no spike times), join keys mouse_id, session_id, electrode_group,
                                   cluster_id
  tables/jl_trial_summary.csv      per session: trial counts by type x context, reward_available per context,
                                   stimulus parameters, hit rates
  tables/jl_area_counts.csv        per session x target probe x area_acronym_custom: all / stable units
Run on haas: S1M2_DATASET=jl python jl_001_units.py --n-jobs 16
"""
import argparse
import json
import os
import time
import traceback
import warnings

os.environ.setdefault("S1M2_DATASET", "jl")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import s1m2_lib as L

TAB = L.OUT / "tables"


def one(path):
    from ephys_utilities.helpers import data_utils
    from ephys_utilities.allen_utils import allen_utils
    from ephys_utilities.neural_utils import unit_metrics_utils as umu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = data_utils.process_single_nwb(str(path), day_to_analyze="all", apply_artifact_correction=False)
        u, t = res["unit_table"], res["trial_table"]
        u = umu.compute_presence_coverage_metrics(u)
        n_raw = len(u)
        u = allen_utils.process_allen_labels(u, split_merge_areas=True)
    sid = path.stem
    u["cluster_id"] = u.cluster_id.astype(int)
    u["unit_set_all"] = u.bc_label.isin(["good", "mua"])
    u["stable"] = u.unit_set_all & (u.presence_ratio >= 0.5) & (u.coverage_ratio >= 0.9)
    u["n_spikes_session"] = u.spike_times.apply(len)
    keep = ["mouse_id", "session_id", "electrode_group", "cluster_id", "target_region", "area_acronym_custom",
            "area_group", "ccf_atlas_acronym", "ccf_atlas_parent_acronym", "ccf_acronym", "ccf_parent_acronym",
            "ap", "ml", "dv", "bc_label", "ks_label", "firing_rate", "presenceRatio", "presence_ratio",
            "coverage_ratio", "n_spikes_session", "unit_set_all", "stable", "day", "behaviour"]
    u = u[[c for c in keep if c in u.columns]].copy()
    u["electrode_group"] = u.electrode_group.astype(str)
    u["n_units_nwb"] = n_raw
    # ---- trials
    t = t.sort_values("start_time").reset_index(drop=True)
    r = dict(session_id=sid, mouse_id=u.mouse_id.iloc[0], behaviour=str(t.behaviour.iloc[0]), day=int(t.day.iloc[0]),
             n_trials=len(t), session_dur_min=float((t.stop_time.max() - t.start_time.min()) / 60),
             context_values=json.dumps(sorted(map(str, t.context.unique()))),
             context_backgrounds=json.dumps(sorted(map(str, t.context_background.unique()))),
             early_lick=int(t.early_lick.sum()), perf_counts=json.dumps(t.perf.value_counts().sort_index().to_dict()),
             wh_amp=json.dumps(sorted(t.loc[t.trial_type == "whisker_trial", "whisker_stim_amplitude"].unique().tolist())),
             wh_dur=json.dumps(sorted(t.loc[t.trial_type == "whisker_trial", "whisker_stim_duration"].unique().tolist())),
             stim_lag_ms=float(np.nanmax(np.abs(t.stim_onset - t.start_time)) * 1e3),
             rw_start_ms=float(np.nanmedian(t.response_window_start_time - t.start_time) * 1e3),
             n_blocks=int((t.context.astype(str) != t.context.astype(str).shift()).sum()),
             opto=int(t.opto_stim.fillna(0).astype(bool).sum()) if "opto_stim" in t else 0)
    for ctx in sorted(t.context.astype(str).unique()):
        tc = t[t.context.astype(str) == ctx]
        for tt, nm in (("whisker_trial", "W"), ("auditory_trial", "A"), ("no_stim_trial", "NS")):
            x = tc[tc.trial_type == tt]
            r[f"n_{nm}_ctx{ctx}"] = len(x)
            r[f"plick_{nm}_ctx{ctx}"] = float(x.lick_flag.mean()) if len(x) else np.nan
            r[f"reward_avail_{nm}_ctx{ctx}"] = float(x.reward_available.mean()) if len(x) else np.nan
    return u, r


def safe(p):
    try:
        out = one(p)
        print(f"done {p.stem} {time.strftime('%H:%M')}", flush=True)
        return out
    except Exception:
        print(f"FAIL {p.stem}\n{traceback.format_exc()}", flush=True)
        return None


def main():
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=16)
    a = ap.parse_args()
    files = []
    for p in sorted(L.NWB_DIR.glob("*.nwb")):
        with h5py.File(p, "r") as h:
            if "units" in h:
                files.append(p)
    print(f"{len(files)} NWBs with units", flush=True)
    res = [r for r in Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(safe)(p) for p in files) if r is not None]
    U = pd.concat([r[0] for r in res], ignore_index=True)
    T = pd.DataFrame([r[1] for r in res])
    TAB.mkdir(parents=True, exist_ok=True)
    tmp = TAB / "jl_units.tmp.parquet"
    U.to_parquet(tmp, index=False)
    tmp.replace(TAB / "jl_units.parquet")
    T.to_csv(TAB / "jl_trial_summary.csv", index=False)
    C = U[U.unit_set_all].groupby(["session_id", "target_region", "area_acronym_custom"]).agg(
        n_all=("cluster_id", "size"), n_stable=("stable", "sum")).reset_index()
    C.to_csv(TAB / "jl_area_counts.csv", index=False)
    print(f"{U.session_id.nunique()} sessions, {len(U)} units", flush=True)


if __name__ == "__main__":
    main()
