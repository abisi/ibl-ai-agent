"""002 (v2, vectorised; 2026-09-29) -- Hierarchical-bootstrap pseudo-population decoding, learning day, per cohort x
area, all decoding targets in one job (parameters in ../question.md, decided with the user 2026-09-29).
v1 (sklearn per-bin fits, pilots 1-2) is kept as 002_pseudopop_decoding_v1_sklearn.py.

Per (cohort, area) job -- spikes loaded once, rates shared across targets:
  stimulus-aligned rates are computed once per unit on the session's whisker + auditory trials (prep_modality_trials)
  and sliced for hitmiss (whisker rows) and modality_stim (all rows); first-lick-aligned rates once per unit on the
  licked whisker + auditory trials (prep_lick_aligned_trials) for modality_lick.
Per target, eligible sessions = >= MIN_UNITS QC units in the area and >= MIN_TRIALS trials of each class.
One iteration (per target):
  1. sample N_MICE mice (= sessions) with replacement and N_NEURONS units per session with replacement;
  2. real decode: each session's real trials of each class split into 3 folds (every fold holds every class; redrawn
     otherwise); per outer fold, training pseudo-trials (T_TRAIN per class) from the other folds' trials and test
     pseudo-trials (T_TEST per class) from the fold's trials, each pseudo-trial concatenating one trial of the class per
     session, drawn by balanced reuse (without replacement until the session's trials of that class in the fold are
     exhausted, then through a fresh random permutation; user option (c), 2026-09-29); mice and units are drawn with
     replacement; per-bin C chosen by an inner 2-fold split of the training trials; z-scoring on the
     training pseudo-trials; balanced accuracy; mean over outer folds;
  3. N_SHIFTS shifted decodes of the SAME draw (labels linearly shifted within each session by 10-50% of its trials,
     random direction, new shift per session and per decode; non-overlapping ends dropped; shifts leaving < MIN_TRIALS
     of a class redrawn), each with the SAME per-bin, per-fold C as the real decode (user: "fix C across fits
     strictly"); null_i = mean of the N_SHIFTS curves, d_i = real_i - null_i.
Decoder: L2 logistic regression per time bin. Default ENGINE "liblinear" (sklearn per bin, as the per-session
pipeline): at this problem size it was faster than the batched Newton solver (33 s vs 56 s per real decode, same
accuracy; 2026-09-29), so the vectorised solver is kept as ENGINE "batched" (all bins at once, warm-started along the C
grid, intercept unpenalised; `--selftest` compares it with sklearn lbfgs). The main speed-up is the fixed C: shifted
decodes skip the inner CV (about 1/15 of a real decode).
Features: causal 50-ms bins, 5-ms steps (labelled at bin end), whisker-artefact dead zone (-10..+5 ms) excised per trial
before any computation; stim-aligned -200..+600 ms; lick-aligned to the corrected first lick -600..+200 ms.
Iterations (100) and N_SHIFTS (10) are PILOT values.
Output: ../artifacts/002_pseudo_<target>_<cohort>.parquet (appendable; one row per area x iteration).
Run (haas, repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/002_pseudopop_decoding.py
    <cohort> <areas|all> [n_iterations] [targets]      |      ... 002_pseudopop_decoding.py --selftest
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

TARGETS = ["hitmiss", "modality_stim", "modality_lick"]
N_MICE = int(os.environ.get("SSL_PSEUDO_N_MICE", "20"))       # user 2026-09-29: 20 mice per draw (was 10)
MIN_UNITS, MIN_TRIALS = 5, 3
N_NEURONS = int(os.environ.get("SSL_PSEUDO_N_NEURONS", "20"))
T_TRAIN = int(os.environ.get("SSL_PSEUDO_T_TRAIN", "100"))
T_TEST, N_OUTER, N_INNER = 100, 3, 2
N_SHIFTS = 10                      # PILOT value
BIN_W, STRIDE = 0.05, 0.005
DZ = (-0.010, 0.005)
WIN = {"stim": (-0.2, 0.6), "lick": (-0.6, 0.2)}
N_WORKERS = int(os.environ.get("SSL_PSEUDO_N_WORKERS", "60"))
CHUNK = int(os.environ.get("SSL_PSEUDO_CHUNK", "10"))
ENGINE = os.environ.get("SSL_PSEUDO_ENGINE", "liblinear")   # "liblinear" (per-bin sklearn, default) or "batched" (Newton)
MAX_NEWTON, TOL = 30, 1e-6
PARAMS = dict(trial_sampling="balanced reuse (without replacement until exhausted)", n_mice=N_MICE, n_neurons=N_NEURONS, min_units=MIN_UNITS, min_trials=MIN_TRIALS, t_train=T_TRAIN, t_test=T_TEST,
              n_outer=N_OUTER, n_inner=N_INNER, n_shifts=N_SHIFTS, bin_w=BIN_W, stride=STRIDE, dead_zone=str(DZ),
              stage="learning", engine=f"v2 {ENGINE} L2 logistic, C fixed across real+shifts")


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


# ------------------------------------------------------------------------------------------------ decoder
def fit_logreg(X, y, invC, w0=None):
    """Batched L2 logistic regression. X (B, n, p) z-scored, y (n,) in {0, 1}, invC (B,) = 1/C per batch element.
    Minimises sum(log-loss) + 0.5 * invC * ||w||^2 (intercept unpenalised) = sklearn's C*sum(loss) + 0.5||w||^2 / C.
    Newton steps with backtracking. Returns W (B, p + 1), last column = intercept."""
    B, n, p = X.shape
    Xa = np.concatenate([X, np.ones((B, n, 1))], axis=2)
    reg = np.ones(p + 1)
    reg[-1] = 0.0
    lam = invC[:, None] * reg[None, :]
    w = np.zeros((B, p + 1)) if w0 is None else w0.copy()
    yf = y.astype(float)

    def obj(w_):
        z = np.einsum("bnp,bp->bn", Xa, w_)
        return np.logaddexp(0, z).sum(1) - (z * yf).sum(1) + 0.5 * (lam * w_ ** 2).sum(1)

    f = obj(w)
    for _ in range(MAX_NEWTON):
        z = np.einsum("bnp,bp->bn", Xa, w)
        s = 0.5 * (1 + np.tanh(0.5 * z))
        g = np.einsum("bnp,bn->bp", Xa, s - yf) + lam * w
        H = np.matmul((Xa * (s * (1 - s))[:, :, None]).transpose(0, 2, 1), Xa)
        H[:, np.arange(p + 1), np.arange(p + 1)] += lam + 1e-9
        step = np.linalg.solve(H, g[:, :, None])[:, :, 0]
        t = np.ones(B)
        for _ in range(20):                      # backtracking per batch element
            fn = obj(w - t[:, None] * step)
            bad = fn > f + 1e-12
            if not bad.any():
                break
            t[bad] *= 0.5
        w = w - t[:, None] * step
        f = obj(w)
        if np.max(np.abs(t[:, None] * step)) < TOL:
            break
    return w


def fit_bins(X, y, invC, w0=None):
    """Fit one L2 logistic per bin. ENGINE "liblinear": sklearn liblinear per bin (as the per-session pipeline;
    intercept penalised) -- measured 2026-09-29 faster than the batched Newton at this size (200 pseudo-trials x 200
    units: 33 s vs 56 s per real decode, identical accuracy). ENGINE "batched": fit_logreg. Returns W (B, p + 1)."""
    if ENGINE == "batched":
        return fit_logreg(X, y, invC, w0)
    from sklearn.linear_model import LogisticRegression
    B, n, p = X.shape
    W = np.empty((B, p + 1))
    for b in range(B):
        m = LogisticRegression(penalty="l2", solver="liblinear", C=1.0 / invC[b], max_iter=1000).fit(X[b], y)
        W[b, :p], W[b, p] = m.coef_[0], m.intercept_[0]
    return W


def standardise(Xtr, Xte):
    mu = np.nanmean(Xtr, axis=1, keepdims=True)
    sd = np.nanstd(Xtr, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def bal_acc(W, Xte, yte):
    z = np.einsum("bnp,bp->bn", Xte, W[:, :-1]) + W[:, -1:]
    pred = z > 0
    return 0.5 * (pred[:, yte].mean(1) + (~pred[:, ~yte]).mean(1))


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


def balanced_reuse(pool, T, rng):
    """Option (c), user 2026-09-29: T trial indices from `pool` without replacement until the pool is exhausted, then
    again through a fresh random permutation (each real trial used floor(T/len) or ceil(T/len) times)."""
    reps = int(np.ceil(T / len(pool)))
    return np.concatenate([rng.permutation(pool) for _ in range(reps)])[:T]


def pseudo(parts, mask_fn, rng, T):
    """T pseudo-trials per class; each concatenates one trial of that class per session, drawn by balanced reuse from
    the trials selected by mask_fn(part) (independently per session). Returns X (bins x 2T x features), y (2T,) bool."""
    Xs = []
    for c in (False, True):
        cols = [p["R"][:, balanced_reuse(np.where(mask_fn(p) & (p["y"] == c))[0], T, rng), :] for p in parts]
        Xs.append(np.concatenate(cols, axis=2))
    return np.concatenate(Xs, axis=1).astype(np.float64), np.r_[np.zeros(T, bool), np.ones(T, bool)]


def decode(parts, rng, C_grid, fixed=None):
    """3-fold pseudo-population decoding. fixed=None: per-bin C chosen by inner 2-fold CV (real decode); returns
    (accuracy curve, chosen C index per fold x bin). fixed=array (folds x bins): those C used (shifted decodes)."""
    n_bins = parts[0]["R"].shape[0]
    acc = np.full((N_OUTER, n_bins), np.nan)
    cidx = np.zeros((N_OUTER, n_bins), int)
    order = np.argsort(C_grid)                     # small C (strong penalty) first: warm-start path
    for fo in range(N_OUTER):
        Xtr, ytr = pseudo(parts, lambda p: p["fold"] != fo, rng, T_TRAIN)
        Xte, yte = pseudo(parts, lambda p: p["fold"] == fo, rng, T_TEST)
        if fixed is None:
            for p in parts:
                trm = p["fold"] != fo
                g = np.full(len(p["y"]), -1)
                g[trm] = split_folds(p["y"][trm], N_INNER, rng)
                p["inner"] = g
            score = np.zeros((len(C_grid), n_bins))
            for fi in range(N_INNER):
                a, ya = pseudo(parts, lambda p: (p["inner"] >= 0) & (p["inner"] != fi), rng, T_TRAIN)
                b, yb = pseudo(parts, lambda p: p["inner"] == fi, rng, T_TEST)
                a, b = standardise(a, b)
                w = None
                for ci in order:
                    w = fit_bins(a, ya, np.full(n_bins, 1.0 / C_grid[ci]), w)
                    score[ci] += bal_acc(w, b, yb) / N_INNER
            best = np.max(score, axis=0)
            cidx[fo] = np.array([order[np.where(score[order, j] >= best[j] - 1e-12)[0][0]] for j in range(n_bins)])
        else:
            cidx[fo] = fixed[fo]
        Xtr, Xte = standardise(Xtr, Xte)
        W = fit_bins(Xtr, ytr, 1.0 / np.asarray(C_grid)[cidx[fo]])
        acc[fo] = bal_acc(W, Xte, yte)
    return acc.mean(0), cidx


# ------------------------------------------------------------------------------------------------ data
def fast_unit_spikes(root, sid, wanted):
    """Same output as ssl_timeresolved_decoding.load_session_unit_spikes (verified identical 2026-09-29), one argsort."""
    from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
    shard = load_spike_shard(root / "spikes" / sid)
    t, dense, cids = np.asarray(shard["spike_times_seconds"]), np.asarray(shard["spike_clusters"]), np.asarray(shard["cluster_ids"])
    order = np.argsort(dense, kind="stable")
    ts = t[order]
    bounds = np.searchsorted(dense[order], np.arange(len(cids) + 1))
    first = {}
    for i, c in enumerate(cids.tolist()):
        first.setdefault(c, i)
    return {cid: np.sort(ts[bounds[first[cid]]:bounds[first[cid] + 1]]) for cid in np.asarray(wanted).tolist() if cid in first}


class SessionData:
    """One session x area: spikes of the area's units and per-unit rate caches shared by all targets."""

    def __init__(self, sid, subject, units, spikes, stim_tr, lick_tr, edges):
        self.sid, self.subject, self.units, self.spikes, self.edges = sid, subject, np.asarray(units), spikes, edges
        self.stim_start = stim_tr.start_time.to_numpy()
        self.stim_w = (stim_tr.trial_type == "whisker_trial").to_numpy()
        self.lick = None if lick_tr is None or not len(lick_tr) else dict(
            event=lick_tr.first_lick_time.to_numpy(), start=lick_tr.start_time.to_numpy(),
            w=(lick_tr.trial_type == "whisker_trial").to_numpy())
        self.cache = {"stim": {}, "lick": {}}

    def rates(self, align, units):
        import ssl_timeresolved_decoding as T
        c = self.cache[align]
        for u in set(units.tolist()) - set(c):
            if align == "stim":
                mats = T.sliding_bin_population_matrices(self.spikes, np.array([u]), self.stim_start, self.stim_w, self.edges["stim"],
                                                         dead_zone=DZ)
            else:
                L = self.lick
                mats = T.lick_aligned_bin_population_matrices(self.spikes, np.array([u]), L["event"], L["start"], L["w"],
                                                              self.edges["lick"], dead_zone=DZ)
            c[u] = np.stack([m[:, 0] for m in mats]).astype(np.float32)
        return np.stack([c[u] for u in units.tolist()], axis=-1)     # bins x trials x units


class View:
    """One target's view of a SessionData: trial rows (into the shared rate cache) and labels."""

    def __init__(self, sd, align, rows, y):
        self.sd, self.align, self.rows, self.y, self.sid, self.subject, self.units = sd, align, rows, y, sd.sid, sd.subject, sd.units

    def rates(self, units):
        return self.sd.rates(self.align, units)[:, self.rows, :]


def load_cohort_area(cohort, area_col, area):
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    sess = T.hitmiss_session_list(st)
    # cohort "pooled" (user 2026-09-30, stimulus decoding): R+ and R- learning sessions together, mice drawn from both
    sess = sess[(sess.day_stage == "learning") & (sess.reward_group.isin(["R+", "R-"]) if cohort == "pooled"
                                                   else sess.reward_group == cohort)]
    edges = {a: T.causal_bin_edges(WIN[a], bin_width=BIN_W, stride=STRIDE) for a in WIN}
    views = {t: {} for t in TARGETS}
    for r in sess.itertuples():
        units = T.area_units(r.session_id, area_col, area, labels)
        if len(units) < MIN_UNITS:
            continue
        mod = T.prep_modality_trials(root, r.session_id, st, tt)
        if mod is None or not len(mod):
            continue
        hm = T.prep_hitmiss_trials(root, r.session_id, st, tt)
        lk = T.prep_lick_aligned_trials(root, r.session_id, st, tt)
        sd = SessionData(r.session_id, r.subject_id, units, fast_unit_spikes(root, r.session_id, units), mod, lk, edges)
        w_rows = np.where(sd.stim_w)[0]
        if hm is not None and np.array_equal(hm.start_time.to_numpy(), sd.stim_start[w_rows]):   # same whisker trials
            views["hitmiss"][r.session_id] = View(sd, "stim", w_rows, mod.lick_flag.to_numpy()[w_rows].astype(bool))
        views["modality_stim"][r.session_id] = View(sd, "stim", np.arange(len(mod)), sd.stim_w.copy())
        if sd.lick is not None:
            views["modality_lick"][r.session_id] = View(sd, "lick", np.arange(len(lk)), sd.lick["w"].copy())
    for t in TARGETS:
        views[t] = {s: v for s, v in views[t].items() if min(v.y.sum(), (~v.y).sum()) >= MIN_TRIALS}
    return views


# ------------------------------------------------------------------------------------------------ iteration
def make_parts(draw, rng, shift):
    parts = []
    for v, units in draw:
        n = len(v.y)
        for _ in range(100):
            if shift:
                k = int(rng.integers(int(0.1 * n), int(0.5 * n) + 1))
                nidx, bidx = (np.arange(0, n - k), np.arange(k, n)) if rng.random() < 0.5 else (np.arange(k, n), np.arange(0, n - k))
            else:
                nidx = bidx = np.arange(n)
            y = v.y[bidx]
            if min(y.sum(), (~y).sum()) >= MIN_TRIALS:
                break
        else:
            return None
        f = split_folds(y, N_OUTER, rng)
        if f is None:
            return None
        parts.append(dict(R=v.rates(units)[:, nidx, :], y=y, fold=f, sid=v.sid))
    return parts


def one_iteration(views, rng, C_grid):
    eligible = sorted(views)
    mice = rng.choice(len(eligible), N_MICE, replace=True)
    draw = [(views[eligible[m]], rng.choice(views[eligible[m]].units, N_NEURONS, replace=True)) for m in mice]
    parts = make_parts(draw, rng, shift=False)
    if parts is None:
        return None
    real, cidx = decode(parts, rng, C_grid)
    nulls = []
    for _ in range(N_SHIFTS):
        sp = make_parts(draw, rng, shift=True)
        if sp is not None:
            nulls.append(decode(sp, rng, C_grid, fixed=cidx)[0])
    if not nulls:
        return None
    return real, np.array(nulls), cidx, [v.sid for v, _ in draw]


def task(args):
    cohort, area_col, area, targets, todo = args
    sys.path.insert(0, str(ROOT_REPO / "scripts"))
    os.chdir(ROOT_REPO)
    from ssl_timeresolved_decoding import C_GRID
    C_grid = np.asarray(C_GRID, float)
    views = load_cohort_area(cohort, area_col, area)
    out = {t: [] for t in targets}
    for t in targets:
        v = views[t]
        base = dict(target=t, cohort=cohort, area_col=area_col, area=area, n_eligible=len(v),
                    n_eligible_mice=len({x.subject for x in v.values()}), **PARAMS)
        if len(v) < 2:
            out[t].append(dict(base, rep=-1, skipped_reason=f"{len(v)} eligible sessions"))
            continue
        for it in todo.get(t, []):
            rng = np.random.default_rng(zlib.crc32(f"{t}|{cohort}|{area}|paired|{it}".encode()))
            res = one_iteration(v, rng, C_grid)
            if res is None:
                out[t].append(dict(base, rep=it, skipped_reason="could not build folds"))
                continue
            real, nulls, cidx, sids = res
            out[t].append(dict(base, rep=it, curve=real.tolist(), null_mean_curve=np.nanmean(nulls, 0).tolist(),
                               null_curves=nulls.tolist(), n_null=len(nulls), c_index=cidx.tolist(), sessions_sampled=sids,
                               skipped_reason=None))
    return out


def selftest():
    """Batched solver vs sklearn lbfgs (unpenalised intercept) on random data of the pipeline's size."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    B, n, p = 6, 200, 200
    X = rng.normal(size=(B, n, p))
    y = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    X[:, y, :5] += 0.5
    for C in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        t0 = time.time()
        W = fit_logreg(X, y, np.full(B, 1.0 / C))
        tb = time.time() - t0
        md, agree = 0.0, []
        for b in range(B):
            m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000, tol=1e-10).fit(X[b], y)
            ref = np.r_[m.coef_[0], m.intercept_]
            md = max(md, np.max(np.abs(ref - W[b])) / (np.max(np.abs(ref)) + 1e-12))
            agree.append(np.mean(m.predict(X[b]) == ((X[b] @ W[b, :-1] + W[b, -1]) > 0)))
        print(f"C={C:g}: max rel. coef diff {md:.2e}, prediction agreement {np.mean(agree):.4f}, batched fit {tb:.2f}s for B={B}")


def main():
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    cohort, areas_arg = sys.argv[1], sys.argv[2]
    n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 100          # PILOT value
    targets = sys.argv[4].split(",") if len(sys.argv) > 4 else TARGETS
    os.chdir(ROOT_REPO)
    import ssl_timeresolved_decoding as T
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    if areas_arg == "all":
        jobs = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(labels.area_group.dropna().unique())]
    else:
        jobs = [("whole_brain", "All units") if a == "whole_brain" else ("area_group", a) for a in areas_arg.split(",")]
    paths = {t: ART / f"002_pseudo_{t}_{cohort}.parquet" for t in targets}
    done = {t: set() for t in targets}
    for t, pth in paths.items():
        if pth.exists():
            d = pd.read_parquet(pth, columns=["area", "rep"])
            done[t] = set(zip(d.area, d.rep))
    tasks = []
    for area_col, area in jobs:
        for c0 in range(0, n_iter, CHUNK):
            todo = {t: [r for r in range(c0, min(c0 + CHUNK, n_iter)) if (area, r) not in done[t]] for t in targets}
            if any(todo.values()):
                tasks.append((cohort, area_col, area, targets, todo))
    print(f"[002v2] {cohort}: {len(tasks)} tasks ({len(jobs)} areas x {targets}), {N_NEURONS} neurons/session, "
          f"T_train {T_TRAIN}, T_test {T_TEST}, {N_SHIFTS} shifts, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(task, a): a for a in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                res = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[002v2] ERROR {a[2]}: {e!r}", flush=True)
                continue
            for t, rows in res.items():
                if not rows:
                    continue
                new = pd.DataFrame(rows)
                out = pd.concat([pd.read_parquet(paths[t]), new], ignore_index=True) if paths[t].exists() else new
                out.to_parquet(paths[t], index=False)
            print(f"[002v2] [{i}/{len(tasks)}] {a[2]}: " + ", ".join(f"{t} {sum(r.get('skipped_reason') is None for r in rows)}"
                                                                   for t, rows in res.items()) + f" -- {time.time() - t0:.0f}s", flush=True)
    print("[002v2] DONE", flush=True)


if __name__ == "__main__":
    main()
