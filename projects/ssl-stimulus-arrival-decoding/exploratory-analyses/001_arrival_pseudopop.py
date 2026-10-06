"""001 -- Time-resolved pseudo-population decoding of whisker vs auditory stimulus (stimulus-aligned), all sessions pooled
(both cohorts, learning day and expert days), at two time resolutions and for several neuron counts.

Decoder and pseudo-population construction: the method of ssl-pseudopopulation-area-decoding 002 (decode(), pseudo(),
split_folds(), balanced trial reuse are imported from there, unchanged): per time bin an L2 logistic regression; 3-fold
outer cross-validation built on each session's real trials of each class; regularisation chosen per bin and fold by an
inner 2-fold split of the training trials; T_TRAIN = T_TEST = 100 pseudo-trials per class; features z-scored on the
training pseudo-trials; balanced accuracy, mean over outer folds.
Changes decided with the user (2026-10-04, question.md):
  data      all whisker-training ephys sessions of the v2 unit table (both cohorts, day 0 and expert days); sessions are
            the independent sampling unit (mice ignored).
  spikes    KS4 NWB (NWB_ks4), whisker-artefact-corrected spike trains (roc_utils_new.correct_neuron_spike_train: spikes
            in -10..+5 ms around every whisker onset replaced by a Poisson train at the pre-onset rate; seeded per
            session) -- no dead-zone excision on top.
  trials    active, perf != 6, auditory warm-up cut (1 trial before the first whisker trial kept), rule A1 tail trim;
            all whisker vs all auditory trials (lick or not).
  units     good + mua (v2 quality_label); areas from the v2 table (area_group, area_acronym_custom). A session is
            eligible for an area with >= 5 units there and >= 3 trials of each class.
  sampling  one draw = N_SESS sessions with replacement, then N / N_SESS units of the area within each sampled session
            (with replacement; the remainder of N spread over randomly chosen sessions), then trials (balanced reuse).
  N         neuron counts 20, 50, 100, 200, 300, 500 (plus matched counts, 003).
  null      trial shuffling: class labels permuted within each session (before pooling), same draw, same per-bin C as
            the real decode; N_SHUF shuffles averaged -> null_i; corrected accuracy d_i = real_i - null_i.
  bins      causal (labelled at bin end). wide: 50-ms bins, 5-ms steps, -200..+600 ms; zoom: 20-ms bins, 2-ms steps,
            -20..+100 ms. Both resolutions are decoded on the same draw (bins are independent decoders).
Iterations (100) and shuffles (10) are PILOT values.

Stage 1 (--cache): per-session rate cache (bins x trials x units of the target areas) -> CACHE/<sid>.npz
Stage 2 (default): decoding jobs (area x N x iteration chunk) -> OUT/raw/<level>__<area>__N<N>.parquet
Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python <this> [--cache] ...
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time
import zlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "projects" / "ssl-pseudopopulation-area-decoding" / "exploratory-analyses"), str(REPO / "scripts"),
                str(pathlib.Path.home() / "code" / "unit_spikes_analysis")]
m2 = importlib.import_module("002_pseudopop_decoding")

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
UNITS = RES / "_roc_stage_analysis" / "units.parquet"          # v2 unit table (045): keys, quality, areas, ccf, stage
sys.path.insert(0, str(HERE))
AR = importlib.import_module("_areas")
EPOCH, OUT = AR.EPOCH, AR.OUT                                  # ARRIVAL_EPOCH=active|passive (user, 2026-10-04)
CACHE = OUT / "cache_all"                                      # every area group + top-40 fine areas (2026-10-04)
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]

COARSE, FINE, LEVELS = AR.COARSE, AR.FINE, AR.LEVELS
N_LIST = [20, 50, 100, 200, 300, 500]
N_SESS = 20
MIN_UNITS, MIN_TRIALS = 5, 3
N_SHUF = int(os.environ.get("ARRIVAL_NSHUF", "20"))  # 20 (final N = 200 runs, user 2026-10-04); N-sweep runs used 10
N_ITER = 500                                  # default (user 2026-10-06: 500 vs 1000 iterations agree, see change-log); N-sweep runs before that used 100
CHUNK = 5
C_GRID = 1.0 / np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])     # = scripts/ssl_bwm_decoding.C_GRID (002's grid)


def edges():
    wide = [(t - 0.05, t) for t in np.round(np.arange(-0.2, 0.6 + 1e-9, 0.005), 6)]
    zoom = [(t - 0.02, t) for t in np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 6)]
    return np.array(wide), np.array(zoom)


WIDE, ZOOM = edges()
EDGES = np.concatenate([WIDE, ZOOM])
RESOLUTION = np.r_[np.zeros(len(WIDE), int), np.ones(len(ZOOM), int)]      # 0 wide, 1 zoom


# ------------------------------------------------------------------------------------------------ stage 1: cache
def resolve_context(t):
    """per-trial context (user, 2026-10-04): a trial in a fixed ~3 s ITI sequence (gap to the previous or next trial
    3.0 +- 0.3 s) is passive; otherwise the context column decides (passive stays passive; active and unlabelled "nan"
    are active; perf == 6 trials inside active blocks stay active and are dropped by the perf rule). t sorted by time."""
    st = t["start_time"].to_numpy()
    dp, dn = np.r_[np.inf, np.diff(st)], np.r_[np.diff(st), np.inf]
    fixed = (np.abs(dp - 3.0) < 0.3) | (np.abs(dn - 3.0) < 0.3)
    lab = t["context"].astype(str).to_numpy()
    return pd.Series(np.where(fixed | (lab == "passive"), "passive", "active"), index=t.index)


def passive_trials(trials):
    """passive epoch (user, 2026-10-04): passive trials (fixed ~3 s ITI or labelled passive; pre and post blocks pooled,
    perf 6 kept as it codes passive trials), whisker vs auditory"""
    t = trials.sort_values("start_time").reset_index(drop=True)
    t = t[(resolve_context(t) == "passive").to_numpy()]
    return t[t.trial_type.isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)


def stim_trials(trials):
    """active (resolve_context), perf != 6, warm-up cut (keep 1 trial before the first whisker trial), rule A1; whisker + auditory trials"""
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
    return t[t.trial_type.isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)


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
        trials = nwb.trials.to_dataframe()               # raw table (process_nwb_tables relabels `context`), as 051
    units["cluster_id"] = units["cluster_id"].astype(str)
    units = units.merge(U[["electrode_group", "cluster_id", "area_group", "area_acronym_custom", "quality_label"]],
                        on=["electrode_group", "cluster_id"], how="inner", validate="one_to_one")
    onsets = trials.loc[trials["whisker_stim"] == 1, "start_time"].to_numpy()        # every whisker onset, any context
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    t = stim_trials(trials) if EPOCH == "active" else passive_trials(trials)
    starts = t.start_time.to_numpy()
    y = (t.trial_type == "whisker_trial").to_numpy()
    lo, hi = starts[:, None] + EDGES[None, :, 0], starts[:, None] + EDGES[None, :, 1]
    width = (EDGES[:, 1] - EDGES[:, 0])[None, :]
    R = np.empty((len(EDGES), len(t), len(units)), np.float32)
    for j, st in enumerate(units.spike_times):
        st = ru.correct_neuron_spike_train(np.asarray(st), onsets, rng)
        R[:, :, j] = ((np.searchsorted(st, hi) - np.searchsorted(st, lo)) / width).T
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, R=R, y=y, area_group=units.area_group.to_numpy(str), area_fine=units.area_acronym_custom.to_numpy(str),
             cluster_id=units.cluster_id.to_numpy(str), electrode_group=units.electrode_group.to_numpy(str),
             start_time=starts, edges=EDGES)
    os.replace(tmp, f)
    return sid, f"{len(t)} trials ({int(y.sum())} whisker), {len(units)} units"


# ------------------------------------------------------------------------------------------------ stage 2: decoding
SESS, VIEWS, CGRID = {}, {}, None


def load_views():
    U = pd.read_parquet(UNITS)
    meta = U.drop_duplicates("session_id").set_index("session_id")
    for f in sorted(CACHE.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        sid = f.stem
        SESS[sid] = dict(R=z["R"], y=z["y"].astype(bool), ag=z["area_group"], af=z["area_fine"],
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


def parts_for(draw, rng, shuffle):
    parts = []
    for sid, units in draw:
        s = SESS[sid]
        y = rng.permutation(s["y"]) if shuffle else s["y"]
        f = m2.split_folds(y, m2.N_OUTER, rng)
        if f is None:
            return None
        parts.append(dict(R=s["R"][:, :, units], y=y, fold=f, sid=sid))
    return parts


def job(args):
    level, area, N, iters = args
    views = VIEWS[(level, area)]
    rows = []
    base = dict(level=level, area=area, N=N, n_eligible_sessions=len(views),
                n_eligible_mice=len({SESS[s]["mouse"] for s, _ in views}), n_sess_per_draw=N_SESS, n_shuffles=N_SHUF)
    for it in iters:
        rng = np.random.default_rng(zlib.crc32(f"{level}|{area}|{N}|{it}".encode()))
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
        for _ in range(N_SHUF):
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
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.cache:
        sids = sorted(pd.read_parquet(UNITS, columns=["session_id"]).session_id.unique())
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            for sid, msg in pool.imap_unordered(cache_session, sids):
                print(f"{time.time() - t0:7.0f}s {sid}: {msg}", flush=True)
        print("CACHE DONE", flush=True)
        return
    CGRID = C_GRID
    load_views()
    print(f"loaded {len(SESS)} sessions in {time.time() - t0:.0f}s", flush=True)
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
    json.dump(dict(script="001_arrival_pseudopop.py", n_sess_per_draw=N_SESS, min_units=MIN_UNITS, min_trials=MIN_TRIALS,
                   n_shuffles=N_SHUF, n_iter=a.n_iter, t_train=m2.T_TRAIN, t_test=m2.T_TEST, n_outer=m2.N_OUTER,
                   n_inner=m2.N_INNER, engine=m2.ENGINE, c_grid=CGRID.tolist(), wide="causal 50 ms / 5 ms, -200..600 ms",
                   zoom="causal 20 ms / 2 ms, -20..100 ms", null="trial shuffle within session, C fixed from real decode",
                   spikes="NWB_ks4, whisker-artefact Poisson correction (roc_utils_new), seeded per session",
                   trials=("active, perf!=6, warm-up cut, A1 trim; all whisker vs all auditory" if AR.EPOCH == "active" else
                           "passive (fixed ~3 s ITI or labelled passive; pre + post pooled; perf 6 kept); all whisker vs all auditory"),
                   epoch=AR.EPOCH,
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
