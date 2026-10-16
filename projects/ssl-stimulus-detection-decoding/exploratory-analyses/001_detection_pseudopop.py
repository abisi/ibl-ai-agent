"""001 -- Time-resolved pseudo-population decoding of stimulus presence: whisker vs no stimulus and auditory vs no stimulus
(stimulus-aligned), all sessions pooled (both cohorts, learning day and expert days), at two time resolutions and for several
neuron counts. Same pipeline as ssl-stimulus-arrival-decoding 001 (whisker vs auditory); changes (user 2026-10-16):
  target    DETECT_TARGET = whisker | auditory: stimulus trials of that modality (lick or not) vs no-stimulus events.
  epoch     DETECT_EPOCH = active: no-stimulus events = correct rejections (no_stim_trial, lick_flag 0), same trial rules as
            the arrival project (perf != 6, auditory warm-up cut keeping 1 trial before the first whisker trial, rule A1 tail
            trim). passive: passive blocks contain no no-stimulus trials, so the no-stimulus events are sham onsets
            0.8 s before the stimulus of the OTHER modality's passive trials (whisker target: windows before auditory
            stimuli; auditory target: before whisker stimuli); the sham's full decoding window (-200..+600 ms) ends 200 ms
            before that stimulus. Each trial gives one event, so the labels follow the passive trial sequence and the
            linear-shift null stays valid. Shams with a piezo lick in [-200, +600] ms are dropped.
  spikes    KS4 NWB; the whisker-artefact Poisson replacement (-10..+5 ms, roc_utils_new) is applied at every whisker onset
            (any context) AND at every cached event onset of every class, so no class can be told apart in that window.
  null      linear shift (ssl-pseudopopulation-area-decoding 002 make_parts): per session the labels of the time-ordered
            events are shifted by 10-50 % of the events (non-wrapping, random direction, new shift per session and per
            decode; shifts leaving < MIN_TRIALS of a class are redrawn), same draw and same per-bin C as the real decode;
            N_SHIFT shifts averaged -> null_i; corrected accuracy d_i = real_i - null_i. (No shuffle null.)
Unchanged: decoder (002 decode / pseudo / split_folds), sampling (N_SESS sessions with replacement, N / N_SESS units of the
area per session), eligibility (>= 5 units in the area, >= 3 events per class), bins (wide causal 50 ms / 5 ms, -200..+600 ms;
zoom causal 20 ms / 2 ms, -20..+100 ms), areas (_areas: all groups + top-40 fine areas), N list.

Stage 1 (--cache): one cache per epoch with every event class -> <HOME>/cache_<epoch>/<sid>.npz
  active: cls W (whisker trial), A (auditory trial), N (correct rejection); passive: W, A, SW (sham before a whisker
  trial), SA (sham before an auditory trial). Rates bins x events x units (float32), event times, class labels.
Stage 2 (default): decoding jobs (area x N x iteration chunk) for DETECT_TARGET -> <OUT>/raw/<level>__<area>__N<N>.parquet
Run on haas: cd ~/code/unit_spikes_analysis; DETECT_EPOCH=.. DETECT_TARGET=.. PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python <this> [--cache]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "projects" / "ssl-pseudopopulation-area-decoding" / "exploratory-analyses"), str(REPO / "scripts"),
                str(pathlib.Path.home() / "code" / "unit_spikes_analysis")]
m2 = importlib.import_module("002_pseudopop_decoding")
sys.path.insert(0, str(HERE))
AR = importlib.import_module("_areas")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
UNITS, EPOCH, TARGET, OUT, CACHE = AR.UNITS, AR.EPOCH, AR.TARGET, AR.OUT, AR.CACHE
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]

COARSE, FINE, LEVELS = AR.COARSE, AR.FINE, AR.LEVELS
N_LIST = [20, 50, 100, 200, 300, 500]
N_SESS = 20
MIN_UNITS, MIN_TRIALS = 5, 3
N_SHIFT = int(os.environ.get("DETECT_NSHIFT", "10"))   # 10 for the N sweep (pilot), 20 for the final N = 200 runs (as arrival)
N_ITER = 100                                           # N sweep (pilot); final N = 200 runs: --n-iter 500
CHUNK = 5
C_GRID = 1.0 / np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])     # = 002's grid
SHAM = 0.8                                             # passive sham onset, s before the other modality's stimulus
LICK_WIN = (-0.2, 0.6)                                 # passive shams with a lick in this window (re. sham) are dropped
# decoded classes per target and epoch: (stimulus class, no-stimulus class)
CLASSES = {("whisker", "active"): ("W", "N"), ("auditory", "active"): ("A", "N"),
           ("whisker", "passive"): ("W", "SA"), ("auditory", "passive"): ("A", "SW")}


def edges():
    wide = [(t - 0.05, t) for t in np.round(np.arange(-0.2, 0.6 + 1e-9, 0.005), 6)]
    zoom = [(t - 0.02, t) for t in np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 6)]
    return np.array(wide), np.array(zoom)


WIDE, ZOOM = edges()
EDGES = np.concatenate([WIDE, ZOOM])
RESOLUTION = np.r_[np.zeros(len(WIDE), int), np.ones(len(ZOOM), int)]      # 0 wide, 1 zoom


# ------------------------------------------------------------------------------------------------ stage 1: cache
def resolve_context(t):
    """as arrival 001: a trial in a fixed ~3 s ITI sequence (gap to the previous or next trial 3.0 +- 0.3 s) is passive;
    otherwise the context column decides. t sorted by time."""
    st = t["start_time"].to_numpy()
    dp, dn = np.r_[np.inf, np.diff(st)], np.r_[np.diff(st), np.inf]
    fixed = (np.abs(dp - 3.0) < 0.3) | (np.abs(dn - 3.0) < 0.3)
    lab = t["context"].astype(str).to_numpy()
    return pd.Series(np.where(fixed | (lab == "passive"), "passive", "active"), index=t.index)


def active_events(trials):
    """active (resolve_context), perf != 6, warm-up cut (keep 1 trial before the first whisker trial), rule A1 (as arrival
    stim_trials, applied to the full sequence); W = whisker trials, A = auditory trials, N = no_stim trials with lick_flag 0"""
    t = trials.sort_values("start_time").reset_index(drop=True)
    ctx = resolve_context(t)
    if (ctx == "active").any():
        t = t[ctx == "active"].reset_index(drop=True)
    if "perf" in t:
        t = t[t["perf"] != 6].reset_index(drop=True)
    wi = np.where(t["trial_type"].to_numpy() == "whisker_trial")[0]
    if len(wi):
        t = t.iloc[max(0, wi[0] - 1):].reset_index(drop=True)
    licked = np.where(t["lick_flag"].to_numpy() == 1)[0]
    if len(licked):
        tail = t.iloc[licked[-1] + 1:]
        if (tail.trial_type == "whisker_trial").sum() >= 5 and (tail.trial_type == "auditory_trial").sum() >= 1:
            t = t.iloc[:licked[-1] + 1].reset_index(drop=True)
    tt = t.trial_type.to_numpy()
    cls = np.select([tt == "whisker_trial", tt == "auditory_trial", (tt == "no_stim_trial") & (t.lick_flag.to_numpy() == 0)],
                    ["W", "A", "N"], "")
    keep = cls != ""
    return t.start_time.to_numpy()[keep], cls[keep]


def passive_events(trials, licks):
    """passive trials (fixed ~3 s ITI or labelled passive; pre and post blocks pooled; perf 6 kept): W / A at the stimulus,
    SW / SA = sham onsets SHAM s before a whisker / auditory stimulus (dropped if a lick falls in LICK_WIN)"""
    t = trials.sort_values("start_time").reset_index(drop=True)
    t = t[(resolve_context(t) == "passive").to_numpy()]
    t = t[t.trial_type.isin(["whisker_trial", "auditory_trial"])]
    st = t.start_time.to_numpy(); w = (t.trial_type == "whisker_trial").to_numpy()
    sham = st - SHAM
    quiet = np.searchsorted(licks, sham + LICK_WIN[1]) == np.searchsorted(licks, sham + LICK_WIN[0])
    times = np.r_[st, sham[quiet]]
    cls = np.r_[np.where(w, "W", "A"), np.where(w, "SW", "SA")[quiet]]
    o = np.argsort(times, kind="stable")
    return times[o], cls[o]


def cache_session(sid):
    f = CACHE / f"{sid}.npz"
    if f.exists():
        return sid, "exists"
    from pynwb import NWBHDF5IO
    from roc_analysis import roc_utils_new as ru
    U = pd.read_parquet(UNITS)
    U = U[(U.session_id == sid) & U.quality_label.isin(["good", "mua"])
          & (U.area_group.isin(COARSE) | U.area_acronym_custom.isin(FINE))].drop_duplicates(KEYS)
    if not len(U):
        return sid, "no target units"
    with NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
        trials = nwb.trials.to_dataframe()               # raw table (process_nwb_tables relabels `context`), as arrival
        licks = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                                   .time_series["piezo_lick_times"].data[:], float))
    units["cluster_id"] = units["cluster_id"].astype(str)
    units = units.merge(U[["electrode_group", "cluster_id", "area_group", "area_acronym_custom", "quality_label"]],
                        on=["electrode_group", "cluster_id"], how="inner", validate="one_to_one")
    times, cls = active_events(trials) if EPOCH == "active" else passive_events(trials, licks)
    if not len(times):
        return sid, "no events"
    wh = trials.loc[trials["whisker_stim"] == 1, "start_time"].to_numpy()        # every whisker onset, any context
    onsets = np.unique(np.r_[wh, times])                                          # + every cached event onset
    rng = np.random.default_rng(zlib.crc32(f"{sid}|detect|{EPOCH}".encode()))
    lo, hi = times[:, None] + EDGES[None, :, 0], times[:, None] + EDGES[None, :, 1]
    width = (EDGES[:, 1] - EDGES[:, 0])[None, :]
    R = np.empty((len(EDGES), len(times), len(units)), np.float32)
    for j, st in enumerate(units.spike_times):
        st = ru.correct_neuron_spike_train(np.asarray(st), onsets, rng)
        R[:, :, j] = ((np.searchsorted(st, hi) - np.searchsorted(st, lo)) / width).T
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, R=R, cls=cls, t=times, area_group=units.area_group.to_numpy(str), area_fine=units.area_acronym_custom.to_numpy(str),
             cluster_id=units.cluster_id.to_numpy(str), electrode_group=units.electrode_group.to_numpy(str), edges=EDGES)
    os.replace(tmp, f)
    n = {c: int((cls == c).sum()) for c in np.unique(cls)}
    return sid, f"{len(times)} events {n}, {len(units)} units"


# ------------------------------------------------------------------------------------------------ stage 2: decoding
SESS, VIEWS, CGRID = {}, {}, None


def load_views():
    """target classes only: y = True for the stimulus class, False for the no-stimulus class, events in time order"""
    U = pd.read_parquet(UNITS)
    meta = U.drop_duplicates("session_id").set_index("session_id")
    cs, cn = CLASSES[(TARGET, EPOCH)]
    for f in sorted(CACHE.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        sid = f.stem
        cls, t = z["cls"].astype(str), z["t"]
        keep = np.where(np.isin(cls, [cs, cn]))[0]
        keep = keep[np.argsort(t[keep], kind="stable")]
        SESS[sid] = dict(R=np.ascontiguousarray(z["R"][:, keep, :]), y=(cls[keep] == cs), ag=z["area_group"], af=z["area_fine"],
                         stage=meta.stage.get(sid), cohort=meta.cohort.get(sid), mouse=meta.mouse_id.get(sid))
    for level, areas in LEVELS.items():
        col = "ag" if level == "area_group" else "af"
        for a in areas:
            v = []
            for sid, s in SESS.items():
                u = np.where(s[col] == a)[0]
                if len(u) >= MIN_UNITS and min(s["y"].sum(), (~s["y"]).sum()) >= MIN_TRIALS:
                    v.append((sid, u))
            VIEWS[(level, a)] = v


def parts_for(draw, rng, shift):
    """as 002 make_parts: real (shift=False) or linearly shifted labels (non-wrapping, 10-50 % of the events, random
    direction, redrawn until both classes keep >= MIN_TRIALS; non-overlapping ends dropped)"""
    parts = []
    for sid, units in draw:
        s = SESS[sid]
        n = len(s["y"])
        for _ in range(100):
            if shift:
                k = int(rng.integers(int(0.1 * n), int(0.5 * n) + 1))
                nidx, bidx = (np.arange(0, n - k), np.arange(k, n)) if rng.random() < 0.5 else (np.arange(k, n), np.arange(0, n - k))
            else:
                nidx = bidx = np.arange(n)
            y = s["y"][bidx]
            if min(y.sum(), (~y).sum()) >= MIN_TRIALS:
                break
        else:
            return None
        f = m2.split_folds(y, m2.N_OUTER, rng)
        if f is None:
            return None
        parts.append(dict(R=s["R"][:, nidx][:, :, units], y=y, fold=f, sid=sid))
    return parts


def job(args):
    level, area, N, iters = args
    views = VIEWS[(level, area)]
    rows = []
    base = dict(level=level, area=area, N=N, target=TARGET, epoch=EPOCH, n_eligible_sessions=len(views),
                n_eligible_mice=len({SESS[s]["mouse"] for s, _ in views}), n_sess_per_draw=N_SESS, n_shifts=N_SHIFT)
    for it in iters:
        rng = np.random.default_rng(zlib.crc32(f"{TARGET}|{EPOCH}|{level}|{area}|{N}|{it}".encode()))
        sess = rng.choice(len(views), N_SESS, replace=True)
        cnt = np.full(N_SESS, N // N_SESS)
        cnt[rng.choice(N_SESS, N % N_SESS, replace=False)] += 1
        draw = [(views[k][0], rng.choice(views[k][1], c, replace=True)) for k, c in zip(sess, cnt) if c > 0]
        parts = parts_for(draw, rng, False)
        if parts is None:
            rows.append(dict(base, rep=it, skipped_reason="folds"))
            continue
        real, cidx = m2.decode(parts, rng, CGRID)
        nulls = []
        for _ in range(N_SHIFT):
            p = parts_for(draw, rng, True)
            if p is not None:
                nulls.append(m2.decode(p, rng, CGRID, fixed=cidx)[0])
        nulls = np.array(nulls)
        rows.append(dict(base, rep=it, curve=real.astype(np.float32).tolist(),
                         null_mean=np.nanmean(nulls, 0).astype(np.float32).tolist(),
                         null_sd=np.nanstd(nulls, 0).astype(np.float32).tolist(), n_null=len(nulls),
                         sessions=[d[0] for d in draw], n_units_per_session=[len(d[1]) for d in draw],
                         c_index=cidx.tolist(), skipped_reason=None))
    return level, area, N, rows


def raw_path(level, area, N):
    return OUT / "raw" / f"{level}__{area.replace(' ', '_')}__N{N}.parquet"


def main(a):
    global CGRID
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import multiprocessing as mp
    t0 = time.time()
    if a.cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        sids = sorted(pd.read_parquet(UNITS, columns=["session_id"]).session_id.unique())
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            for sid, msg in pool.imap_unordered(cache_session, sids):
                print(f"{time.time() - t0:7.0f}s {sid}: {msg}", flush=True)
        print("CACHE DONE", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    CGRID = C_GRID
    load_views()
    print(f"{TARGET} {EPOCH}: loaded {len(SESS)} sessions in {time.time() - t0:.0f}s", flush=True)
    if a.runs_file:                                   # 003 --plan: "level|area|N" per line
        combos = [tuple(l.split("|")) for l in pathlib.Path(a.runs_file).read_text().split("\n") if l.strip()]
        combos = [(l, ar, int(n)) for l, ar, n in combos]
    else:
        levels = [a.level] if a.level else list(LEVELS)
        combos = [(level, area, N) for level in levels for area in (a.areas.split(",") if a.areas else LEVELS[level])
                  for N in (a.n_list or N_LIST)]
    jobs, todo = [], {}
    for level, area, N in combos:
        if (level, area) not in VIEWS:
            print("unknown area", level, area); continue
        f = raw_path(level, area, N)
        done = set(pd.read_parquet(f, columns=["rep"]).rep) if f.exists() else set()
        its = [i for i in range(a.n_iter) if i not in done]
        todo[(level, area, N)] = []
        jobs += [(level, area, N, its[k:k + CHUNK]) for k in range(0, len(its), CHUNK)]
    for (level, area) in sorted({c[:2] for c in combos if c[:2] in VIEWS}):
        print(f"{level} {area}: {len(VIEWS[(level, area)])} eligible sessions", flush=True)
    jobs.sort(key=lambda j: -j[2])                    # largest N first (longest jobs)
    print(f"{len(jobs)} jobs", flush=True)
    (OUT / "raw").mkdir(exist_ok=True)
    pending = {k: sum(1 for j in jobs if j[:3] == k) for k in todo}
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        for level, area, N, rows in pool.imap_unordered(job, jobs):
            k = (level, area, N)
            todo[k] += rows
            pending[k] -= 1
            if pending[k] == 0:
                f = raw_path(level, area, N)
                df = pd.DataFrame(todo[k])
                if f.exists():
                    df = pd.concat([pd.read_parquet(f), df], ignore_index=True)
                tmp = f.with_suffix(".tmp.parquet")
                df.to_parquet(tmp)
                os.replace(tmp, f)
                todo[k] = []
                print(f"{time.time() - t0:7.0f}s wrote {f.name} ({len(df)} rows)", flush=True)
    cs, cn = CLASSES[(TARGET, EPOCH)]
    json.dump(dict(script="001_detection_pseudopop.py", target=TARGET, epoch=EPOCH, classes=[cs, cn], n_sess_per_draw=N_SESS,
                   min_units=MIN_UNITS, min_trials=MIN_TRIALS, n_shifts=N_SHIFT, n_iter=a.n_iter, t_train=m2.T_TRAIN,
                   t_test=m2.T_TEST, n_outer=m2.N_OUTER, n_inner=m2.N_INNER, engine=m2.ENGINE, c_grid=CGRID.tolist(),
                   wide="causal 50 ms / 5 ms, -200..600 ms", zoom="causal 20 ms / 2 ms, -20..100 ms",
                   null="linear label shift within session (10-50 % of events, non-wrapping), C fixed from the real decode",
                   spikes="NWB_ks4, artefact Poisson replacement -10..+5 ms at every whisker onset and every event onset, seeded per session",
                   events=("active, perf!=6, warm-up cut, A1 trim; stimulus trials (lick or not) vs correct rejections" if EPOCH == "active" else
                           f"passive (fixed ~3 s ITI or labelled passive; pre + post pooled); stimulus vs sham onsets {SHAM} s before "
                           f"the other modality's stimulus, lick-free in {LICK_WIN}"),
                   units="good+mua, v2 unit table", sessions=sorted(SESS), stages_cohorts="pooled",
                   date=time.strftime("%Y-%m-%d %H:%M")), open(OUT / "provenance_001.json", "w"), indent=1)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--level", default=None)
    ap.add_argument("--areas", default=None)
    ap.add_argument("--n-list", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--runs-file", default=None)
    ap.add_argument("--n-proc", type=int, default=100)
    main(ap.parse_args())
