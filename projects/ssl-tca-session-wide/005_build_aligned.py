"""Event-aligned firing-rate tensors from spike times, one script for all alignments (user 2026-11-22).

Data source: NWB_ks4 (spike times + trials via data_utils.process_single_nwb), stable good + mua units
(tables/units.parquet from 001), trial selection tca_lib.select_trials, jaw onsets from the per-mouse
combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl (`jaw_dlc_onset`, s from start_time), first lick corrected for
the artefact window: lick = start_time + (lick_time - response_window_start_time) (ssl-nwb-lick-time-offset).
Spikes in -10..+5 ms of every trial start are removed (tca_lib.load_session) and, before binning, replaced by Poisson
surrogate spikes at the unit's baseline rate of that trial (FILL, user 2026-11-22: correction before the tensor; the
earlier valid-time rates, count / valid time, amplified the noise ~3.5x in the 2.5-ms-valid sensory bins).

Alignments (50-ms bins, unsmoothed; user 2026-11-22):
  stim_long      start_time, -0.2 .. +1.5 s (34 bins)      whisker + auditory trials, passive + active (no-stim
                 trials dropped, user 2026-11-22)
  stim_sensory   start_time, -52.5 .. +107.5 ms, 10-ms bins (16) on a grid shifted by -2.5 ms, whisker + auditory
                 trials (user 2026-11-22: pre-event bins; artefact bins kept, valid-time corrected, 2.5 ms valid each)
  jaw            jaw onset, -0.3 .. +0.3 s (12 bins)       active lick trials with a valid jaw onset
  lick           corrected first lick, -0.3 .. +0.3 s      same trials as jaw (comparable)
  stim_long_ns   as stim_long, all trials incl. no-stimulus (false alarms / correct rejections) (user 2026-11-23)
  jaw_all        jaw onset, -0.3 .. +0.3 s, 10-ms bins (60), all trials: licks and jaw movements without a lick at
                 their onset, still trials at the type-matched median lick onset (jaw_all_events; user 2026-11-23)
Valid jaw onset: 0.015 s <= onset <= first-lick latency (pre-stimulus / artefact-time onsets dropped).
Partial rebuild: --align <names> builds only those alignments (trials.parquet / units.parquet rewritten, identical
trial selection).
Baseline (all alignments): per trial and unit, rate in start_time - 1.0 .. start_time - 0.010 s of the same trial
(jaw / lick tensors borrow the baseline before that trial's start).

Run: python 005_build_aligned.py [--sessions <sid> ...] [--n-jobs 20]   (default: all sessions)
     -> OUT/artifacts/aligned/<sid>/<alignment>.npz (rate [trials x units x bins], valid [trials x bins] in s,
        baseline [trials x units], t_centres, trial_index), units.parquet, trials.parquet
"""
import argparse
import importlib
import json
import time
import traceback
import zlib

import numpy as np
import pandas as pd

import tca_lib as L

B = importlib.import_module("002_build_tensors")
ALIGN = {
    "stim_long": dict(event="stim", win=(-0.2, 1.5), bin=0.05, trials="stim"),
    # pre-event bins in every window (user 2026-11-22); sensory: one regular 10-ms grid shifted by -2.5 ms (user): no
    # bin lies entirely inside the artefact (-10..+5 ms); the two bins touching it keep 2.5 ms of valid time each
    # (rate = spikes / valid time, noisy there)
    "stim_sensory": dict(event="stim", win=(-0.0525, 0.1075), bin=0.01, trials="stim"),
    "jaw": dict(event="jaw", win=(-0.3, 0.3), bin=0.05, trials="lick"),
    "lick": dict(event="lick", win=(-0.3, 0.3), bin=0.05, trials="lick"),
    # go-signal tensors (user 2026-11-23): long stimulus window with the no-stimulus trials (false alarms / correct
    # rejections), and a fine jaw-aligned window over all trials (non-lick trials at their own jaw onset or at a pseudo
    # onset, see jaw_all_events)
    "stim_long_ns": dict(event="stim", win=(-0.2, 1.5), bin=0.05, trials="all"),
    "jaw_all": dict(event="jaw_all", win=(-0.3, 0.3), bin=0.01, trials="all"),
}
MOVE_RANGE = (0.015, 0.6)   # jaw onset of a movement without a lick (jaw_all)
BASE_WIN = (-1.0, -0.010)
FILL = True   # Poisson fill of the artefact window at the trial's baseline rate, before binning (user 2026-11-22)
JAW_RANGE = (0.015, 1.2)
OUT_DIR = L.OUT / "artifacts" / "aligned"


def bin_counts(spike_times, ev, edges):
    """counts [trials x units x bins] for bins ev + edges (edges relative, ascending)."""
    E = ev[:, None] + edges[None]                       # trials x (nb + 1)
    C = np.zeros((len(ev), len(spike_times), len(edges) - 1), np.float32)
    for j, sp in enumerate(spike_times):
        k = np.searchsorted(np.asarray(sp, float), E.ravel()).reshape(E.shape)
        C[:, j] = np.diff(k, axis=1)
    return C


def valid_time(st, ev, edges):
    """seconds of each bin outside the trial's own artefact window [st - 10 ms, st + 5 ms)."""
    lo, hi = L.CONFIG["artifact_window_s"]
    a0, a1 = (st + lo)[:, None], (st + hi)[:, None]
    b0, b1 = ev[:, None] + edges[None, :-1], ev[:, None] + edges[None, 1:]
    ov = np.clip(np.minimum(b1, a1) - np.maximum(b0, a0), 0, None)
    return (b1 - b0) - ov


def fill_artefact(sp, st, base, sid):
    """Poisson surrogate spikes in each selected trial's artefact window [st - 10 ms, st + 5 ms), at the unit's
    baseline rate of that trial (-1.0 .. -0.010 s, which ends where the window starts); uniform times; seeded per
    session. Mean = baseline, Poisson variance as in any other bin (the valid-time rate of the previous version
    amplified the variance about 3.5x in the 2.5-ms-valid sensory bins)."""
    lo, hi = L.CONFIG["artifact_window_s"]
    dur = hi - lo
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    out = []
    for j, x in enumerate(sp):
        n = rng.poisson(base[:, j] * dur)
        if n.sum():
            x = np.sort(np.r_[x, np.repeat(st + lo, n) + rng.random(int(n.sum())) * dur])
        out.append(x)
    return out


def jaw_all_events(t):
    """jaw_all alignment (user 2026-11-23): event time and class per trial.
    lick  = active lick trial with a valid jaw onset (jaw_valid), aligned to it;
    move  = no lick, jaw onset in MOVE_RANGE (active miss / correct rejection or passive trial), aligned to it;
    still = no jaw onset in 0..1 s, aligned to a pseudo onset: the session's median onset of the lick trials of the
            same trial type (fallback: all lick trials);
    other = the rest (late / early onsets), pseudo-aligned like still (kept, labelled)."""
    st = t.start_time.to_numpy(float)
    on = t.jaw_dlc_onset.to_numpy(float)
    lick = t.lick_flag.fillna(0).to_numpy().astype(int) == 1
    valid = t.jaw_valid.to_numpy(bool)
    mv = ~lick & (on >= MOVE_RANGE[0]) & (on <= MOVE_RANGE[1])
    still = ~((on >= 0) & (on <= 1.0))
    cls = np.where(valid, "lick", np.where(mv, "move", np.where(still & ~lick, "still", "other")))
    ty = t.trial_type.to_numpy()
    allmed = np.nanmedian(on[valid]) if valid.any() else 0.15
    med = {k: (np.nanmedian(on[valid & (ty == k)]) if (valid & (ty == k)).sum() >= 3 else allmed) for k in np.unique(ty)}
    rel = np.where(np.isin(cls, ["lick", "move"]), on, np.array([med[k] for k in ty]))
    return st + rel, cls, rel


def build(sid, mouse, day, cohort, keys, only=None):
    t0 = time.time()
    units, trials = L.load_session(sid, keys)
    t = L.select_trials(trials)
    t["lick_latency"] = t.lick_time - t.response_window_start_time
    t["cohort"] = cohort
    t["trial_index"] = np.arange(len(t))
    t, meta = L.learning_curves(mouse, day, t)
    t = t.sort_values("trial_index").reset_index(drop=True)
    jf = L.ROOT / "combined_results_ks4" / mouse / "dlc_jaw_onset_times.pkl"
    if jf.exists():
        j = pd.read_pickle(jf)
        j = j[j.session_id == sid][["start_time", "jaw_dlc_onset"]].astype(float).sort_values("start_time")
    else:   # no jaw-onset table for this mouse (AB080, AB082, MH026): stimulus alignments only
        j = pd.DataFrame({"start_time": pd.Series(dtype=float), "jaw_dlc_onset": pd.Series(dtype=float)})
    t = pd.merge_asof(t.sort_values("start_time"), j, on="start_time", direction="nearest", tolerance=1e-3)
    t = t.sort_values("trial_index").reset_index(drop=True)
    st = t.start_time.to_numpy(float)
    act_lick = ((t.epoch == "active") & (t.lick_flag == 1) & (t.lick_latency > 0)).to_numpy()
    on = t.jaw_dlc_onset.to_numpy(float)
    lat = t.lick_latency.to_numpy(float)
    t["jaw_valid"] = act_lick & (on >= JAW_RANGE[0]) & (on <= np.minimum(lat, JAW_RANGE[1]))
    t["jaw_time"] = st + on
    t["lick_abs"] = st + lat
    t["jaw_all_time"], t["jaw_class"], t["jaw_all_rel"] = jaw_all_events(t)
    od = OUT_DIR / sid
    od.mkdir(parents=True, exist_ok=True)
    sp = [np.sort(np.asarray(x, float)) for x in units.spike_times]
    # baseline (same for all alignments)
    be = np.array(BASE_WIN)
    base = bin_counts(sp, st, be)[:, :, 0] / (be[1] - be[0])
    if FILL:   # artefact correction before binning (user 2026-11-22): surrogate spikes in the removed window
        sp = fill_artefact(sp, st, base, sid)
    info = {}
    for name, a in ALIGN.items():
        if only and name not in only:
            continue
        if a["trials"] == "all":
            idx = np.arange(len(t))
        elif a["trials"] == "stim":   # whisker + auditory trials, passive and active (no-stim dropped, user 2026-11-22)
            idx = np.flatnonzero((t.trial_type != "no_stim_trial").to_numpy())
        else:
            idx = np.flatnonzero(t.jaw_valid.to_numpy())
        if len(idx) == 0:   # e.g. no jaw onsets: alignment skipped
            info[name] = dict(n_trials=0, n_bins=0, valid_frac=float("nan"))
            continue
        ev = {"stim": st, "jaw": t.jaw_time.to_numpy(float), "lick": t.lick_abs.to_numpy(float),
              "jaw_all": t.jaw_all_time.to_numpy(float)}[a["event"]][idx]
        segs = a["win"] if isinstance(a["win"], list) else [a["win"]]   # one or more contiguous segments
        Cs, Vs, tcs = [], [], []
        for w0, w1 in segs:
            nb_s = int(round((w1 - w0) / a["bin"]))
            edges = w0 + a["bin"] * np.arange(nb_s + 1)
            Cs.append(bin_counts(sp, ev, edges))
            Vs.append(valid_time(st[idx], ev, edges))
            tcs.append((edges[:-1] + edges[1:]) / 2)
        C, V, tcen = np.concatenate(Cs, 2), np.concatenate(Vs, 1), np.concatenate(tcs)
        nb = len(tcen)
        if FILL:   # every bin is a full-width count after the fill
            V = np.full_like(V, a["bin"])
        R = C / np.maximum(V, 1e-9)[:, None, :]
        R[np.repeat((V <= 0)[:, None, :], R.shape[1], 1)] = np.nan
        np.savez_compressed(od / f"{name}.npz", rate=R.astype(np.float32), valid=V.astype(np.float32),
                            baseline=base[idx].astype(np.float32), t_centres=tcen,
                            trial_index=idx, event=a["event"], win=np.array(segs), bin=a["bin"])
        info[name] = dict(n_trials=int(len(idx)), n_bins=int(nb), valid_frac=round(float(V.sum() / V.size / a["bin"]), 4))
    B.flat(units.drop(columns=["spike_times"])).to_parquet(od / "units.parquet", index=False)
    B.flat(t).to_parquet(od / "trials.parquet", index=False)
    fi = od / "build_info.json"
    if only and fi.exists():   # partial build: keep the other alignments' info
        info = {**json.loads(fi.read_text()).get("align", {}), **info}
    (od / "build_info.json").write_text(json.dumps(dict(session_id=sid, secs=round(time.time() - t0), meta={
        k: (float(v) if np.isscalar(v) and v is not None else None) for k, v in meta.items()}, align=info,
        config=L.CONFIG, alignments={k: {**v, "win": [list(w) for w in (v["win"] if isinstance(v["win"], list)
                                                                         else [v["win"]])]} for k, v in ALIGN.items()},
        baseline_win=list(BASE_WIN), jaw_range=list(JAW_RANGE), artefact_fill="poisson_baseline" if FILL else None), indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", help="default: all sessions in tables/sessions.csv")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--align", nargs="*", help="build only these alignments (others kept); default: all")
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    if a.sessions:
        s = s[s.session_id.isin(a.sessions)]
    u = pd.read_parquet(L.OUT / "tables" / "units.parquet")

    def one(r):
        keys = u[u.session_id == r.session_id][["electrode_group", "cluster_id", "area_acronym_custom",
                                                 "target_region"]].reset_index(drop=True)
        try:
            info = build(r.session_id, r.mouse_id, r.day, r.cohort, keys, only=a.align)
            print(r.session_id, info, flush=True)
            return dict(session_id=r.session_id, ok=True, **{f"{k}_trials": v["n_trials"] for k, v in info.items()})
        except Exception:
            print(r.session_id, "FAILED", traceback.format_exc()[-800:], flush=True)
            return dict(session_id=r.session_id, ok=False, error=traceback.format_exc()[-800:])
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(one)(r) for r in s.itertuples())
    logn = "aligned_build_log.csv" if not a.align else f"aligned_build_log_{'_'.join(a.align)}.csv"
    pd.DataFrame(res).to_csv(L.OUT / "tables" / logn, index=False)


if __name__ == "__main__":
    main()
