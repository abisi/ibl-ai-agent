"""Per-trial, per-region response table for the novelty / reward-exposure models (learning day, good units).

For every ACTIVE trial (whisker, auditory, catch = no-stim) of every learning session:
  region responses  mean over the region's quality_label == 'good' units of the per-unit z-scored rate in
                    base  [-310, -10] ms   (pre-stimulus baseline; covariate)
                    early [  5,  50] ms    (stimulus-locked, before almost every lick; whisker artifact window
                                            [-10, +5] ms is corrected upstream and excluded here)
                    late  [ 50, 150] ms
                    post  [150, 300] ms
                    relative to start_time (= whisker/auditory onset; no-stim reference time for catch trials).
                    z = (rate - unit mean pre-stim rate over all active trials) / max(SD, 1 Hz).
                    Levels: area_acronym_custom, area_group and 'all' (every good unit).
  trial covariates  type, lick, first-lick latency (corrected), reward, exposure counts before the trial (whisker /
                    auditory / catch; active only AND including passive blocks), total active-trial index, clock
                    time, time since the last whisker stimulus (any context), previous trial type / lick / reward,
                    warm-up flag (active trials before the first whisker trial), video covariates in [-500, 0] ms
                    (pupil area, whisker-angle SD = whisking, jaw-angle SD; each robust-z within session, clipped +-5), learning trial.
Output: combined_results_ks4/_novelty_model/trial_region_table.parquet, trial_covariates.parquet
"""
import argparse
import importlib
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m = importlib.import_module("029_roc_rpe_v2")
OUT = m.RES / "_novelty_model"
WIN = {"base": (-0.31, -0.01), "early": (0.005, 0.05), "late": (0.05, 0.15), "post": (0.15, 0.30)}


def counts_in(st, t0, w):
    return np.searchsorted(st, t0 + w[1]) - np.searchsorted(st, t0 + w[0])


def video_cov(nwb, t0):
    ts = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
    out = {}
    for name, key, fn in [("pupil", "pupil_area", np.nanmean), ("whisking", "whisker_angle", np.nanstd),
                          ("jaw", "jaw_angle", np.nanstd)]:
        if key not in ts or ts[key].timestamps is None:
            out[name] = np.full(len(t0), np.nan); continue
        d = np.asarray(ts[key].data[:], float); t = np.asarray(ts[key].timestamps[:], float)
        n = min(len(d), len(t)); d, t = d[:n], t[:n]
        a, b = np.searchsorted(t, t0 - 0.5), np.searchsorted(t, t0)
        out[name] = np.array([fn(d[i:j]) if j > i else np.nan for i, j in zip(a, b)])
    for k in out:                                   # robust z within session (tracking artifacts), clipped at +-5
        v = out[k]
        if np.isfinite(v).sum() > 10:
            med = np.nanmedian(v); mad = 1.4826 * np.nanmedian(np.abs(v - med))
            out[k] = np.clip((v - med) / (mad if mad > 0 else np.nanstd(v) or 1.0), -5, 5)
    return out


def worker(sid):
    mouse = sid.split("_")[0]
    nwb, units, act = m.load_session(sid)
    allt = nwb.trials.to_dataframe().sort_values("start_time")
    ctx = allt["context"].astype(str).str.strip().str.lower()
    allt["ctx"] = "active" if ctx.isin(["nan", "none", ""]).all() else ctx
    q = pd.read_csv(m.RES / mouse / "whisker_0" / m.SUBDIR / f"{mouse}_roc_rpe_v2.csv",
                    usecols=["neuron_id", "quality_label", "area_acronym_custom", "area_group", "cohort"])
    cohort = q.cohort.iloc[0]
    good = q[q.quality_label == "good"].set_index("neuron_id")
    units = units[units.neuron_id.isin(good.index)].reset_index(drop=True)
    act = act.reset_index(drop=True)
    act["ttype"] = np.select([act.whisker_stim == 1, act.auditory_stim == 1, act.no_stim == 1],
                             ["whisker", "auditory", "catch"], "other")
    t0 = act.start_time.to_numpy(float)
    # ---- per-unit z-scored window rates (units x trials)
    Z = {}
    base_c = np.stack([counts_in(np.sort(np.asarray(s)), t0, WIN["base"]) for s in units.spike_times]) / 0.30
    mu, sd = base_c.mean(1, keepdims=True), np.maximum(base_c.std(1, keepdims=True), 1.0)
    for w, (a, b) in WIN.items():
        rate = base_c if w == "base" else np.stack([counts_in(np.sort(np.asarray(s)), t0, (a, b))
                                                   for s in units.spike_times]) / (b - a)
        Z[w] = (rate - mu) / sd
    reg = good.loc[units.neuron_id]
    rows = []
    for level in ["area_acronym_custom", "area_group", "all"]:
        labs = np.full(len(units), "all") if level == "all" else reg[level].astype(str).to_numpy()
        for r in np.unique(labs):
            sel = labs == r
            df = pd.DataFrame(dict(session_id=sid, trial=np.arange(len(act)), level=level, region=r, n_units=int(sel.sum())))
            for w in WIN:
                df[f"z_{w}"] = Z[w][sel].mean(0)
            rows.append(df)
    R = pd.concat(rows, ignore_index=True)
    # ---- trial covariates
    C = pd.DataFrame(dict(session_id=sid, mouse_id=mouse, cohort=cohort, trial=np.arange(len(act)), ttype=act.ttype,
                          start_time=t0, lick=act.lick_flag.to_numpy(int),
                          rt=np.where(act.lick_flag == 1, (act.lick_time - act.response_window_start_time).to_numpy(float), np.nan),
                          reward=((act.lick_flag == 1) & (act.reward_available == 1)).to_numpy(int)))
    for k in ["whisker", "auditory", "catch"]:
        C[f"n_{k}_before"] = np.r_[0, np.cumsum(C.ttype.to_numpy() == k)[:-1]]
    # exposures including passive blocks (any context), counted by stimulus time
    all_w = np.sort(allt.loc[allt.whisker_stim == 1, "start_time"].to_numpy(float))
    all_a = np.sort(allt.loc[allt.auditory_stim == 1, "start_time"].to_numpy(float))
    C["n_whisker_before_all"] = np.searchsorted(all_w, t0 - 1e-6)
    C["n_auditory_before_all"] = np.searchsorted(all_a, t0 - 1e-6)
    C["n_passive_pre_whisker"] = int(((allt.whisker_stim == 1) & allt.ctx.str.startswith("passive") &
                                      (allt.start_time < t0.min())).sum())
    j = np.searchsorted(all_w, t0 - 1e-6) - 1
    C["time_since_whisker"] = np.where(j >= 0, t0 - all_w[np.maximum(j, 0)], np.nan)
    C["clock_min"] = (t0 - t0.min()) / 60
    C["prev_ttype"] = C.ttype.shift(1); C["prev_lick"] = C.lick.shift(1); C["prev_reward"] = C.reward.shift(1)
    first_w = np.where(C.ttype == "whisker")[0]
    C["warmup"] = C.trial < (first_w[0] if len(first_w) else len(C))
    C["aud_rank"] = np.where(C.ttype == "auditory", C.n_auditory_before, np.nan)
    for k, v in video_cov(nwb, t0).items():
        C[k] = v
    lt, _ = m.r27.learning_split_time(mouse, 0, np.sort(act.loc[act.whisker_stim == 1, "start_time"].to_numpy()))
    C["learning_trial_time"] = lt if lt is not None else np.nan
    return R, C


def main(a):
    OUT.mkdir(exist_ok=True)
    d = pd.read_parquet(m.SUMMARY / "learning" / "rpe_v2_neurons.parquet", columns=["session_id"])
    sessions = sorted(d.session_id.unique())
    Rs, Cs = [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (R, C) in enumerate(pool.map(worker, sessions)):
            Rs.append(R); Cs.append(C)
            print(f"done {i + 1}/{len(sessions)} {C.session_id.iloc[0]}", flush=True)
    R = pd.concat(Rs, ignore_index=True); C = pd.concat(Cs, ignore_index=True)
    R.to_parquet(OUT / "trial_region_table.parquet", index=False)
    C.to_parquet(OUT / "trial_covariates.parquet", index=False)
    print(f"saved {len(R)} region-trial rows, {len(C)} trials, {C.session_id.nunique()} sessions", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
