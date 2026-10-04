"""002 (v1, sklearn per-bin fits; SUPERSEDED 2026-09-29 by the vectorised 002_pseudopop_decoding.py, kept for provenance of pilots 1-2) -- Hierarchical-bootstrap pseudo-population decoding, learning day, per target x cohort x area (parameters in
../question.md, decided with the user 2026-09-29).

Per (target, cohort, area):
  eligible sessions = learning-day sessions of the cohort with >= MIN_UNITS QC units (good + mua) in the area and
  >= MIN_TRIALS trials of each class. One repetition:
  1. sample N_MICE mice (= sessions; one learning session per mouse) with replacement;
  2. per sampled session: N_NEURONS units with replacement; each class's real trials split into 3 folds (every fold
     holds >= 1 trial of every class; redrawn otherwise);
  3. for each outer fold f: training pseudo-trials (T per class) built from the other folds' trials, test
     pseudo-trials (T per class) from fold f's trials, both by sampling one trial of the class per session with
     replacement and concatenating the sessions' units (N_MICE x N_NEURONS features);
     C chosen per bin by an inner 2-fold split of the training trials (same per-session/per-class construction),
     StandardScaler + L2 logistic, balanced accuracy on the test pseudo-trials; accuracy = mean over outer folds.
  Paired linear-shift null (user design, 2026-09-29): in every iteration the SAME draw (mice, units) is also decoded
  N_SHIFTS times with the labels linearly shifted within each session (k = 10-50% of its trials, random direction, new
  shift per session and per shifted decode; neural trial i paired with the label of trial i + k, non-overlapping ends
  dropped; shifts leaving < MIN_TRIALS of a class redrawn); null_i = mean of the N_SHIFTS curves, d_i = real_i - null_i.
  Pseudo-trials: T_TRAIN per class for training and inner C selection, T_TEST per class for testing.
  Iterations (100) and N_SHIFTS (10) are PILOT values.
Features: causal 50-ms bins, 5-ms steps (labelled at bin end), whisker-artefact dead zone (-10..+5 ms from stimulus
onset) excised per trial before any computation; stim-aligned -200..+600 ms (hitmiss, modality_stim) or aligned to the
corrected first lick (start_time + lick_time - response_window_start_time) -600..+200 ms (modality_lick).
Trial prep: prep_session rule (warm-up block removed except the trial before the first whisker trial).
Targets: hitmiss (whisker trials: lick vs no lick), modality_stim (whisker vs auditory, all trials),
modality_lick (licked whisker vs licked auditory trials).
Output: ../artifacts/002_pseudo_<target>_<cohort>.parquet (appendable; one row per area x iteration: real curve,
null curves and their mean, with
the sampled mice, eligible-session count and all parameters).
Run (haas, repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/002_pseudopop_decoding.py
    <target> <cohort> <areas|all> [n_iterations]
"""

from __future__ import annotations

import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART = HERE.parent / "artifacts"

N_MICE, MIN_UNITS, MIN_TRIALS = 10, 5, 3
N_NEURONS = int(os.environ.get("SSL_PSEUDO_N_NEURONS", "20"))   # per session; user 2026-09-29: 20 (was 5)
T_TRAIN = int(os.environ.get("SSL_PSEUDO_T_TRAIN", "100"))      # training pseudo-trials per class; user 2026-09-29: 100
T_TEST, N_OUTER, N_INNER = 100, 3, 2
OUT_SUFFIX = os.environ.get("SSL_PSEUDO_SUFFIX", "")                # e.g. _sens_n20_t100 for the sensitivity pilot
N_SHIFTS = 10   # paired linear shifts per iteration (PILOT value; user may increase)
BIN_W, STRIDE = 0.05, 0.005
DZ = (-0.010, 0.005)
WIN = {"stim": (-0.2, 0.6), "lick": (-0.6, 0.2)}
N_WORKERS = int(os.environ.get("SSL_PSEUDO_N_WORKERS", "12"))
CHUNK_REPS = 5
PARAMS = dict(n_mice=N_MICE, n_neurons=N_NEURONS, min_units=MIN_UNITS, min_trials=MIN_TRIALS, t_train=T_TRAIN, t_test=T_TEST,
              n_shifts=N_SHIFTS,
              n_outer=N_OUTER, n_inner=N_INNER, bin_w=BIN_W, stride=STRIDE, dead_zone=str(DZ), stage="learning")


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def target_trials(target, root, sid, st, tt):
    """(trials, y, event_time, is_whisker, alignment) for one session, or None."""
    import ssl_timeresolved_decoding as T
    if target == "hitmiss":
        tr = T.prep_hitmiss_trials(root, sid, st, tt)
        if tr is None:
            return None
        return tr, tr.lick_flag.to_numpy().astype(bool), tr.start_time.to_numpy(), np.ones(len(tr), bool), "stim"
    if target == "modality_stim":
        tr = T.prep_modality_trials(root, sid, st, tt)
        if tr is None:
            return None
        w = (tr.trial_type == "whisker_trial").to_numpy()
        return tr, w, tr.start_time.to_numpy(), w, "stim"
    tr = T.prep_lick_aligned_trials(root, sid, st, tt)   # corrected first lick (add_first_lick_time)
    if tr is None:
        return None
    w = (tr.trial_type == "whisker_trial").to_numpy()
    return tr, w, tr.first_lick_time.to_numpy(), w, "lick"


class Session:
    """Trials, labels and lazily computed per-unit rate matrices (bins x trials) for one session and area."""

    def __init__(self, sid, subject, units, spikes, tr, y, event, is_whisker, align, edges):
        self.sid, self.subject, self.units, self.spikes = sid, subject, np.asarray(units), spikes
        self.y, self.event, self.start, self.is_whisker, self.align, self.edges = y, event, tr.start_time.to_numpy(), is_whisker, align, edges
        self.cache = {}

    def rates(self, unit_ids):
        import ssl_timeresolved_decoding as T
        out = []
        for u in unit_ids:
            if u not in self.cache:
                if self.align == "stim":
                    mats = T.sliding_bin_population_matrices(self.spikes, np.array([u]), self.start, self.is_whisker, self.edges, dead_zone=DZ)
                else:
                    mats = T.lick_aligned_bin_population_matrices(self.spikes, np.array([u]), self.event, self.start, self.is_whisker,
                                                                  self.edges, dead_zone=DZ)
                self.cache[u] = np.stack([m[:, 0] for m in mats]).astype(np.float32)   # bins x trials
            out.append(self.cache[u])
        return np.stack(out, axis=-1)   # bins x trials x units


def split_folds(y, n_folds, rng, max_tries=50):
    """Per-class fold labels on real trials; every fold must contain every class (redrawn otherwise)."""
    for _ in range(max_tries):
        f = np.empty(len(y), int)
        for c in (False, True):
            idx = np.where(y == c)[0]
            f[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
        if all(((f == k) & (y == c)).any() for k in range(n_folds) for c in (False, True)):
            return f
    return None


def pseudo(parts, mask_fn, rng, T):
    """Pseudo-trials: T per class; each concatenates one trial of that class per session (with replacement) drawn
    from the trials selected by mask_fn(part). Returns X (bins x 2T x features), y (2T)."""
    Xs, ys = [], []
    for c in (False, True):
        cols = []
        for p in parts:
            pool = np.where(mask_fn(p) & (p["y"] == c))[0]
            pick = rng.choice(pool, T, replace=True)
            cols.append(p["R"][:, pick, :])
        Xs.append(np.concatenate(cols, axis=2))
        ys.append(np.full(T, c))
    return np.concatenate(Xs, axis=1), np.concatenate(ys)


def fit_score(Xtr, ytr, Xte, yte, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    ok_tr, ok_te = ~np.isnan(Xtr).any(1), ~np.isnan(Xte).any(1)
    if len(np.unique(ytr[ok_tr])) < 2 or len(np.unique(yte[ok_te])) < 2:
        return np.nan
    sc = StandardScaler().fit(Xtr[ok_tr])
    m = LogisticRegression(penalty="l2", solver="liblinear", C=C, max_iter=1000).fit(sc.transform(Xtr[ok_tr]), ytr[ok_tr])
    return balanced_accuracy_score(yte[ok_te], m.predict(sc.transform(Xte[ok_te])))


def decode(parts, rng):
    """3-fold pseudo-population decoding of one set of per-session parts (R: bins x trials x units, y, fold)."""
    from ssl_timeresolved_decoding import C_GRID
    n_bins = parts[0]["R"].shape[0]
    acc = np.full((N_OUTER, n_bins), np.nan)
    for fo in range(N_OUTER):
        Xtr, ytr = pseudo(parts, lambda p: p["fold"] != fo, rng, T_TRAIN)
        Xte, yte = pseudo(parts, lambda p: p["fold"] == fo, rng, T_TEST)
        for p in parts:   # inner 2-fold C selection per bin, on the training folds' trials only
            tr_idx = p["fold"] != fo
            g = np.full(len(p["y"]), -1)
            g[tr_idx] = split_folds(p["y"][tr_idx], N_INNER, rng)
            p["inner"] = g
        inner_sets = [(pseudo(parts, lambda p: (p["inner"] >= 0) & (p["inner"] != fi), rng, T_TRAIN),
                       pseudo(parts, lambda p: p["inner"] == fi, rng, T_TRAIN)) for fi in range(N_INNER)]
        for bi in range(n_bins):
            best, bestC = -1, C_GRID[0]
            for C in C_GRID:
                sc = np.nanmean([fit_score(a[0][bi], a[1], b[0][bi], b[1], C) for a, b in inner_sets])
                if sc > best:
                    best, bestC = sc, C
            acc[fo, bi] = fit_score(Xtr[bi], ytr, Xte[bi], yte, bestC)
    return np.nanmean(acc, 0)


def make_parts(draw, rng, shift):
    """Per-session parts for one draw (sampled sessions + units), real or with a fresh linear shift per session."""
    parts = []
    for s, units in draw:
        n = len(s.y)
        for _ in range(100):
            if shift:
                k = int(rng.integers(int(0.1 * n), int(0.5 * n) + 1))
                nidx, bidx = (np.arange(0, n - k), np.arange(k, n)) if rng.random() < 0.5 else (np.arange(k, n), np.arange(0, n - k))
            else:
                nidx = bidx = np.arange(n)
            y = s.y[bidx]
            if min(y.sum(), (~y).sum()) >= MIN_TRIALS:
                break
        else:
            return None
        f = split_folds(y, N_OUTER, rng)
        if f is None:
            return None
        parts.append(dict(R=s.rates(units)[:, nidx, :], y=y, fold=f, sid=s.sid))
    return parts


def one_iteration(sessions, eligible, rng):
    """User design (2026-09-29): one draw of mice (with replacement) and units per session; decode the real labels and
    N_SHIFTS linearly shifted versions of the SAME draw (new random shift per session each time). Returns the real
    curve, the N_SHIFTS null curves and the sampled sessions."""
    mice = rng.choice(len(eligible), N_MICE, replace=True)
    draw = [(sessions[eligible[mi]], rng.choice(sessions[eligible[mi]].units, N_NEURONS, replace=True)) for mi in mice]
    parts = make_parts(draw, rng, shift=False)
    if parts is None:
        return None
    real = decode(parts, rng)
    nulls = []
    for _ in range(N_SHIFTS):
        sp = make_parts(draw, rng, shift=True)
        if sp is not None:
            nulls.append(decode(sp, rng))
    if not nulls:
        return None
    return real, np.array(nulls), [s.sid for s, _ in draw]


def fast_unit_spikes(root, sid, wanted):
    """Same output as ssl_timeresolved_decoding.load_session_unit_spikes (sorted spike times per cluster_id), but
    one argsort instead of one full scan per unit (13 s -> ~1 s per session), restricted to `wanted` clusters."""
    from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
    shard = load_spike_shard(root / "spikes" / sid)
    t, dense, cids = np.asarray(shard["spike_times_seconds"]), np.asarray(shard["spike_clusters"]), np.asarray(shard["cluster_ids"])
    order = np.argsort(dense, kind="stable")
    ts = t[order]
    bounds = np.searchsorted(dense[order], np.arange(len(cids) + 1))
    first = {}
    for i, c in enumerate(cids.tolist()):
        first.setdefault(c, i)          # library convention: first dense index of the cluster id
    out = {}
    for cid in np.asarray(wanted).tolist():
        i = first.get(cid)
        if i is not None:
            out[cid] = np.sort(ts[bounds[i]:bounds[i + 1]])
    return out


def load_area(target, cohort, area_col, area):
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & (sess.reward_group == cohort)]
    align = "lick" if target == "modality_lick" else "stim"
    edges = T.causal_bin_edges(WIN[align], bin_width=BIN_W, stride=STRIDE)
    out = {}
    for r in sess.itertuples():
        units = T.area_units(r.session_id, area_col, area, labels)
        if len(units) < MIN_UNITS:
            continue
        tt_ = target_trials(target, root, r.session_id, st, tt)
        if tt_ is None:
            continue
        tr, y, event, is_w, al = tt_
        if min(y.sum(), (~y).sum()) < MIN_TRIALS:
            continue
        out[r.session_id] = Session(r.session_id, r.subject_id, units, fast_unit_spikes(root, r.session_id, units), tr, y, event,
                                    is_w, al, edges)
    return out, edges


def task(args):
    """One (target, cohort, area) chunk: sessions loaded once, then the listed paired iterations."""
    target, cohort, area_col, area, todo = args
    sys.path.insert(0, str(ROOT_REPO / "scripts"))
    os.chdir(ROOT_REPO)
    sessions, edges = load_area(target, cohort, area_col, area)
    eligible = sorted(sessions)
    base = dict(target=target, cohort=cohort, area_col=area_col, area=area, n_eligible=len(eligible),
                n_eligible_mice=len({sessions[s].subject for s in eligible}), **PARAMS)
    if len(eligible) < 2:
        return [dict(base, rep=-1, skipped_reason=f"{len(eligible)} eligible sessions")]
    rows = []
    for it in todo:
        rng = np.random.default_rng(zlib.crc32(f"{target}|{cohort}|{area}|paired|{it}".encode()))
        res = one_iteration(sessions, eligible, rng)
        if res is None:
            rows.append(dict(base, rep=it, skipped_reason="could not build folds"))
            continue
        real, nulls, sids = res
        rows.append(dict(base, rep=it, curve=real.tolist(), null_mean_curve=np.nanmean(nulls, 0).tolist(),
                         null_curves=nulls.tolist(), n_null=len(nulls), sessions_sampled=sids, skipped_reason=None))
    return rows


def main():
    target, cohort, areas_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 100   # PILOT value (user may increase)
    os.chdir(ROOT_REPO)
    import ssl_timeresolved_decoding as T
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    if areas_arg == "all":
        jobs = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(labels.area_group.dropna().unique())]
    else:
        jobs = [("whole_brain", "All units") if a == "whole_brain" else ("area_group", a) for a in areas_arg.split(",")]
    out_path = ART / f"002_pseudo_{target}_{cohort}{OUT_SUFFIX}.parquet"
    done = set()
    if out_path.exists():
        d = pd.read_parquet(out_path, columns=["area", "rep"])
        done = set(zip(d.area, d.rep))
    tasks = []
    for area_col, area in jobs:
        todo = [r for r in range(n_iter) if (area, r) not in done]
        for i in range(0, len(todo), CHUNK_REPS):     # chunks run in parallel; each loads the area's sessions once
            tasks.append((target, cohort, area_col, area, todo[i:i + CHUNK_REPS]))
    print(f"[002] {target} {cohort}{OUT_SUFFIX}: {len(tasks)} tasks ({len(jobs)} areas), {N_NEURONS} neurons/session, "
          f"T_train {T_TRAIN}, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(task, a): a for a in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                new = pd.DataFrame(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"[002] ERROR {a[3]}: {e!r}", flush=True)
                continue
            out = pd.concat([pd.read_parquet(out_path), new], ignore_index=True) if out_path.exists() else new
            out.to_parquet(out_path, index=False)
            print(f"[002] [{i}/{len(tasks)}] {a[3]}: {new.skipped_reason.isna().sum()} reps, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print("[002] DONE", flush=True)


if __name__ == "__main__":
    main()
