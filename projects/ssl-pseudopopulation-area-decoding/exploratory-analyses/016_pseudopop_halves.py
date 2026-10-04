"""016 -- Pseudo-population decoding across SESSION HALVES (user 2026-10-02: "Test the pseudopopulation decoding across session
halves for hit/miss at stim time and auditory vs whisker prelick (-100 ms to lick)").
Decodings (learning-stage sessions, per cohort):
  hitmiss        whisker trials, y = lick, stimulus-aligned windows 5-50 and 5-100 ms (dead zone -10..+5 ms excised);
  modality_lick  licked whisker + auditory trials, y = whisker, -100..0 ms before the corrected first lick.
Trials (skills/ssl-trial-exclusion): prep_hitmiss_trials / prep_lick_aligned_trials (active context, perf != 6, warm-up cut;
the kept pre-whisker trial removed), end-of-session disengagement dropped with rule A1; halves = median split of each
session's retained trials of that decoding.
Units: QC good + mua (as the pseudo-population project), whole brain and each area_group; >= MIN_UNITS per session.
Stage 1 (cache, per session): window rates (Hz) per trial x unit, labels, halves -> NAS cache (haas root disk is full).
Stage 2 (per cohort x decoding/window x area, N_ITER iterations):
  1. draw N_MICE eligible sessions with replacement (eligible = >= MIN_UNITS units and >= MIN_TRIALS trials of each class in
     BOTH halves) and N_NEURONS units per session with replacement;
  2. for each half separately: each session's trials of each class (of that half) split into 3 folds; per outer fold,
     T_TRAIN training and T_TEST test pseudo-trials per class (one trial per session per pseudo-trial, drawn with
     replacement from that session's trials of the class in the training / test folds); z-score on training pseudo-trials;
     L2 logistic regression, C by inner 2-fold CV on the training pseudo-trials; balanced accuracy;
  3. null: the same with each session's labels shuffled within the half (N_NULL), same C; corrected = acc - mean null;
  4. delta = corrected(half 2) - corrected(half 1).
Group level: per cohort, median and 95% interval of corrected accuracy per half and of delta over iterations (mouse-level
bootstrap), p(delta) = 2 min(P(delta > 0), P(delta < 0)); cohort difference of delta from independent pairing of the two
cohorts' iterations (difference of differences), same two-sided p. N_ITER = 100 is a PILOT value.
Outputs: ../artifacts/016_pseudopop_halves.parquet, 016_pseudopop_halves_stats.csv, figures/016_pseudopop_halves.{pdf,png,svg}
Run (haas, repo root): python .../016_pseudopop_halves.py [cache|decode|plot|all]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
ART = HERE.parent / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}
JOBS = [("hitmiss", "5-50ms", (0.005, 0.050)), ("hitmiss", "5-100ms", (0.005, 0.100)),
        ("modality_lick", "-100-0ms", (-0.100, 0.0))]
DZ = (-0.010, 0.005)
MIN_UNITS, MIN_TRIALS = 5, 3
N_MICE = int(os.environ.get("SSL_PSEUDO_N_MICE", "20"))
N_NEURONS = int(os.environ.get("SSL_PSEUDO_N_NEURONS", "10"))
N_ITER = int(os.environ.get("SSL_PSEUDO_N_ITER", "100"))             # PILOT value
T_TRAIN, T_TEST, N_FOLD, N_NULL = 100, 50, 3, 5
C_GRID = (0.001, 0.01, 0.1, 1.0)
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))


def cache_dir():
    from axel_bisi_paths import axel_bisi_root
    d = axel_bisi_root() / "ibl_ai_agent_cache" / "016_pseudopop_halves"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------------------- stage 1
def cache_session(args):
    sid, rg = args
    _init()
    sys.path.insert(0, str(REPO / "scripts"))
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    lab = labels[(labels.session_id == sid) & labels.quality_label.isin(T.QC_VALUES)]
    units = lab.cluster_id.to_numpy()
    if len(units) < MIN_UNITS:
        return sid, None
    spikes = T.load_session_unit_spikes(root, sid)
    dis = T.detect_terminal_disengagement(sid, tt)
    out = dict(session_id=sid, reward_group=rg, units=units, area_group=lab.area_group.to_numpy(), data={})
    for dec in ("hitmiss", "modality_lick"):
        tr = T.prep_hitmiss_trials(root, sid, st, tt) if dec == "hitmiss" else T.prep_lick_aligned_trials(root, sid, st, tt)
        if tr is None or not len(tr):
            continue
        if dec == "modality_lick":
            p = T.prep_modality_trials(root, sid, st, tt)
            fw = p.loc[p.trial_type == "whisker_trial", "start_time"].min() if p is not None else -np.inf
            tr = tr[tr.start_time >= fw]
        if dis["disengaged"]:
            tr = tr[tr.start_time < dis["t_cut"]]
        tr = tr.reset_index(drop=True)
        if len(tr) < 4 * MIN_TRIALS:
            continue
        y = (tr.lick_flag == 1).to_numpy() if dec == "hitmiss" else (tr.trial_type == "whisker_trial").to_numpy()
        half = (np.arange(len(tr)) >= len(tr) // 2).astype(int)
        for d_, w, win in JOBS:
            if d_ != dec:
                continue
            if dec == "hitmiss":
                X = T.sliding_bin_population_matrices(spikes, units, tr.start_time.to_numpy(), np.ones(len(tr), bool), [win],
                                                      dead_zone=DZ)[0]
            else:
                X = T.lick_aligned_bin_population_matrices(spikes, units, tr.first_lick_time.to_numpy(), tr.start_time.to_numpy(),
                                                           (tr.trial_type == "whisker_trial").to_numpy(), [win], dead_zone=DZ)[0]
            if np.isnan(X).any():
                X = np.where(np.isnan(X), np.nanmean(X, 0, keepdims=True), X)
            out["data"][(dec, w)] = dict(X=X.astype(np.float32), y=y, half=half)
    pickle.dump(out, open(cache_dir() / f"{sid}.pkl", "wb"))
    return sid, {k: (int(v["y"].sum()), int((~v["y"]).sum())) for k, v in out["data"].items()}


def run_cache():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    st = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet")
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    todo = [(r.session_id, r.reward_group) for r in sess.itertuples() if not (cache_dir() / f"{r.session_id}.pkl").exists()]
    print(f"[016 cache] {len(todo)} sessions", flush=True)
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        for i, (sid, info) in enumerate(ex.map(cache_session, todo), 1):
            if i % 10 == 0:
                print(f"[016 cache] {i}/{len(todo)}", flush=True)


# ------------------------------------------------------------------------------------------------------------- stage 2
def fit_lr(Xtr, ytr, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000).fit(Xtr, ytr)


def bacc(y, p):
    return 0.5 * (np.mean(p[y]) + np.mean(~p[~y]))


def pseudo(sessions, cls, idx_sets, n, rng):
    """n pseudo-trials of class cls: per session one trial drawn (with replacement) from idx_sets[s][cls]."""
    cols = []
    for s, (X, idx) in enumerate(sessions):
        pick = rng.choice(idx_sets[s][cls], n, replace=True)
        cols.append(X[pick])
    return np.hstack(cols)


def decode_half(blocks, rng, C=None, shuffle=False):
    """blocks: list of (X (trials x units), y) for one half, one per drawn session. Returns (balanced acc, C used)."""
    folds = []
    for X, y in blocks:
        yy = rng.permutation(y) if shuffle else y
        f = {}
        for cls in (True, False):
            i = rng.permutation(np.where(yy == cls)[0])
            f[cls] = np.array_split(i, N_FOLD)
        folds.append((X, f))
    accs, Cs = [], []
    for k in range(N_FOLD):
        tr_idx = [{c: np.concatenate([f[c][j] for j in range(N_FOLD) if j != k]) for c in (True, False)} for _, f in folds]
        te_idx = [{c: f[c][k] if len(f[c][k]) else np.concatenate([f[c][j] for j in range(N_FOLD)]) for c in (True, False)}
                  for _, f in folds]
        S = [(X, None) for X, _ in folds]
        Xtr = np.vstack([pseudo(S, True, tr_idx, T_TRAIN, rng), pseudo(S, False, tr_idx, T_TRAIN, rng)])
        ytr = np.r_[np.ones(T_TRAIN, bool), np.zeros(T_TRAIN, bool)]
        Xte = np.vstack([pseudo(S, True, te_idx, T_TEST, rng), pseudo(S, False, te_idx, T_TEST, rng)])
        yte = np.r_[np.ones(T_TEST, bool), np.zeros(T_TEST, bool)]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        c = C
        if c is None:
            perm = rng.permutation(len(ytr))
            a_, b_ = perm[: len(perm) // 2], perm[len(perm) // 2:]
            sc = [np.mean([bacc(ytr[b_], fit_lr(Xtr[a_], ytr[a_], cc).predict(Xtr[b_]).astype(bool)),
                           bacc(ytr[a_], fit_lr(Xtr[b_], ytr[b_], cc).predict(Xtr[a_]).astype(bool))]) for cc in C_GRID]
            c = C_GRID[int(np.argmax(sc))]
        Cs.append(c)
        accs.append(bacc(yte, fit_lr(Xtr, ytr, c).predict(Xte).astype(bool)))
    return float(np.mean(accs)), float(np.median(Cs))


def iterate(args):
    cohort, dec, w, area, seeds = args
    _init()
    sys.path.insert(0, str(REPO / "scripts"))
    pool = []
    for f in sorted(cache_dir().glob("*.pkl")):
        c = pickle.load(open(f, "rb"))
        if c["reward_group"] != cohort or (dec, w) not in c["data"]:
            continue
        um = np.ones(len(c["units"]), bool) if area == "All units" else (c["area_group"] == area)
        if um.sum() < MIN_UNITS:
            continue
        dd = c["data"][(dec, w)]
        ok = all(min(int((dd["y"] & (dd["half"] == h)).sum()), int((~dd["y"] & (dd["half"] == h)).sum())) >= MIN_TRIALS
                 for h in (0, 1))
        if ok:
            pool.append((c["session_id"], dd["X"][:, um], dd["y"], dd["half"]))
    rows = []
    if len(pool) < 5:
        return [dict(cohort=cohort, decoding=dec, window=w, area=area, n_pool=len(pool), skipped=True)]
    for it in seeds:
        rng = np.random.default_rng(zlib.crc32(f"{cohort}|{dec}|{w}|{area}|{it}".encode()))
        draw = [pool[i] for i in rng.integers(0, len(pool), N_MICE)]
        units = [rng.integers(0, X.shape[1], N_NEURONS) for _, X, _, _ in draw]
        r = dict(cohort=cohort, decoding=dec, window=w, area=area, iteration=it, n_pool=len(pool), skipped=False)
        for h in (0, 1):
            blocks = [(X[half == h][:, u], y[half == h]) for (_, X, y, half), u in zip(draw, units)]
            acc, C = decode_half(blocks, rng)
            null = [decode_half(blocks, rng, C=C, shuffle=True)[0] for _ in range(N_NULL)]
            r[f"acc_h{h + 1}"], r[f"null_h{h + 1}"], r[f"corr_h{h + 1}"] = acc, float(np.mean(null)), acc - float(np.mean(null))
        r["delta"] = r["corr_h2"] - r["corr_h1"]
        rows.append(r)
    return rows


def run_decode():
    caches = [pickle.load(open(f, "rb")) for f in sorted(cache_dir().glob("*.pkl"))]
    areas = ["All units"] + sorted({a for c in caches for a in pd.unique(c["area_group"]) if isinstance(a, str)})
    out_path = ART / "016_pseudopop_halves.parquet"
    chunks = [list(range(i, min(i + 10, N_ITER))) for i in range(0, N_ITER, 10)]
    jobs = [(cohort, dec, w, area, ch) for cohort in ("R+", "R-") for dec, w, _ in JOBS for area in areas for ch in chunks]
    print(f"[016 decode] {len(jobs)} jobs ({len(areas)} areas, {N_ITER} iterations, {N_MICE} mice x {N_NEURONS} units)", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = [ex.submit(iterate, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 50 == 0 or i == len(jobs):
                pd.DataFrame(rows).to_parquet(out_path, index=False)
                print(f"[016 decode] {i}/{len(jobs)} -- {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


# ------------------------------------------------------------------------------------------------------------- stats / plot
def p2(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min(np.mean(x > 0), np.mean(x < 0)) + 1 / (len(x) + 1))) if len(x) else np.nan


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_parquet(ART / "016_pseudopop_halves.parquet")
    d = d[~d.skipped.astype(bool)]
    rng = np.random.default_rng(0)
    rows = []
    for (dec, w, area), g in d.groupby(["decoding", "window", "area"]):
        r = dict(decoding=dec, window=w, area=area)
        dl = {}
        for rg in ("R+", "R-"):
            x = g[g.cohort == rg]
            r[f"n_pool_{rg}"] = int(x.n_pool.max()) if len(x) else 0
            for k in ("corr_h1", "corr_h2", "delta"):
                v = x[k].to_numpy()
                r[f"{k}_{rg}_med"] = float(np.median(v)) if len(v) else np.nan
                r[f"{k}_{rg}_lo"], r[f"{k}_{rg}_hi"] = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
            r[f"p_delta_{rg}"] = p2(x.delta)
            dl[rg] = x.delta.to_numpy()
        if len(dl["R+"]) and len(dl["R-"]):
            n = min(len(dl["R+"]), len(dl["R-"]))
            dd = rng.permutation(dl["R+"])[:n] - rng.permutation(dl["R-"])[:n]
            r["diffdiff_med"], r["p_diffdiff"] = float(np.median(dd)), p2(dd)
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(ART / "016_pseudopop_halves_stats.csv", index=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 6.5})
    areas = ["All units"] + sorted(a for a in S.area.unique() if a != "All units")
    fig, axes = plt.subplots(len(JOBS), 2, figsize=(11.7, 2.9 * len(JOBS) + 0.6),
                             gridspec_kw=dict(width_ratios=[0.6, 2.4]))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.12, wspace=0.18, hspace=0.75)
    for r_, (dec, w, _) in enumerate(JOBS):
        g = d[(d.decoding == dec) & (d.window == w) & (d.area == "All units")]
        ax = axes[r_, 0]
        for k, rg in enumerate(("R+", "R-")):
            x = g[g.cohort == rg]
            for h in (1, 2):
                v = x[f"corr_h{h}"].to_numpy()
                if not len(v):
                    continue
                xx = (h - 1) + (k - 0.5) * 0.25
                ax.violinplot(v, positions=[xx], widths=0.22, showextrema=False)
                ax.plot([xx - 0.08, xx + 0.08], [np.median(v)] * 2, color=COL[rg], lw=2)
            if len(x):
                ax.plot([0 + (k - 0.5) * 0.25, 1 + (k - 0.5) * 0.25], [x.corr_h1.median(), x.corr_h2.median()], color=COL[rg],
                        lw=1.2)
        s = S[(S.decoding == dec) & (S.window == w) & (S.area == "All units")]
        if len(s):
            s = s.iloc[0]
            ax.set_title(f"whole brain · {dec} {w}\nΔ R+ {s.get('delta_R+_med', np.nan):+.3f} (p {s.get('p_delta_R+', np.nan):.3f}), "
                         f"R− {s.get('delta_R-_med', np.nan):+.3f} (p {s.get('p_delta_R-', np.nan):.3f})\n"
                         f"ΔΔ {s.get('diffdiff_med', np.nan):+.3f} (p {s.get('p_diffdiff', np.nan):.3f})", fontsize=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["half 1", "half 2"])
        ax.set_ylabel("acc − shuffle (pseudo-pop.)")
        ax = axes[r_, 1]
        s = S[(S.decoding == dec) & (S.window == w)].set_index("area")
        for i, ar in enumerate(areas):
            if ar not in s.index:
                continue
            for k, rg in enumerate(("R+", "R-")):
                m, lo, hi = s.loc[ar, f"delta_{rg}_med"], s.loc[ar, f"delta_{rg}_lo"], s.loc[ar, f"delta_{rg}_hi"]
                if not np.isfinite(m):
                    continue
                sig = s.loc[ar, f"p_delta_{rg}"] < 0.05
                ax.errorbar(i + (k - 0.5) * 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[rg],
                            mfc=COL[rg] if sig else "white", ms=3, elinewidth=0.8, capsize=0)
            if np.isfinite(s.loc[ar].get("p_diffdiff", np.nan)) and s.loc[ar, "p_diffdiff"] < 0.05:
                ax.text(i, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas, rotation=40, ha="right", fontsize=5.5)
        ax.set_ylabel("Δ (half 2 − half 1)")
        ax.set_title(f"{dec} {w}: change across halves per area (median, 95% bootstrap interval; filled = p < .05; "
                     "* = R+ vs R− difference of differences p < .05)", fontsize=6)
    fig.suptitle(f"Pseudo-population decoding across session halves ({N_MICE} mice × {N_NEURONS} units per draw, "
                 f"{d.iteration.nunique()} iterations — PILOT), trial-shuffle-corrected; perf ≠ 6, warm-up removed, A1 trim; "
                 "pre-lick window −100..0 ms.", fontsize=7)
    (HERE / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(HERE / "figures" / f"016_pseudopop_halves.{ext}", dpi=220)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(S[S.area == "All units"].round(3).T.to_string())


if __name__ == "__main__":
    os.chdir(REPO)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("cache", "all"):
        run_cache()
    if mode in ("decode", "all"):
        run_decode()
    if mode in ("plot", "all"):
        plot()
