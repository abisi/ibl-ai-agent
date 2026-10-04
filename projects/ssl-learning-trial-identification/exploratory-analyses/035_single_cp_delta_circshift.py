"""035 -- Single learning-trial definition: joint whisker + FA change point (034 model: optional lapse, no start gates) with
an EFFECT-SIZE FLOOR inside the model and a DRIFT-PRESERVING (circular-shift) null (user 2026-10-01: "forget the
spontaneous lick rate and use the false alarm rate, sampled like the whisker trial. Try the circular shift and the sweep").
Data: whisker day 0, all 100 mice (028 inputs; active context, perf != 6 -- user rule), whisker trials and no-stim (FA) trials
(FA = lick in the response window of a no-stim trial, i.e. sampled exactly like whisker trials).
Effect-size floor delta: only change configurations whose learned segment has discrimination (whisker - FA, segment MLEs) at
least delta ABOVE the naive segment (R+) / BELOW the generalising segment (R-) count as "change"; the Bayes factor compares
"a change of >= delta" vs no change. Swept: delta in {0, 0.1, 0.2, 0.3}.
Null: circular shift -- the whisker outcome sequence is rotated by a random offset (10-90% of the session) relative to the FA
stream (FA unchanged). Each stream keeps its own slow drift and autocorrelation; only their alignment (the
whisker-specific change relative to FA) is broken. N_SHIFT shifts per session; p = P(BF_shift >= BF_real); learner iff
p < ALPHA. Also reported: the within-type shuffle p (stationary null, 034) for delta = 0.
LT = posterior median of the learned-segment start (90% CI).
Outputs: artifacts/035_single_cp_delta.csv (session x delta), figures 035_coverage_sweep.png,
035_review_<cohort>_d<delta>.png (every mouse at the chosen delta).
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/035_single_cp_delta_circshift.py [mouse ...]
"""

from __future__ import annotations

import importlib
import os
import zlib
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
M = importlib.import_module("034_single_cp_calibrated")
Streams, MIN_SEG, MAX_LAPSE, summarize_pk = M.Streams, M.MIN_SEG, M.MAX_LAPSE, M.summarize_pk
DELTAS = (0.0, 0.1, 0.2, 0.3)
N_SHIFT, ALPHA = 50, 0.05
N_WORKERS = int(os.environ.get("SSL_N_WORKERS", "40"))


def fit(S, rg, delta):
    n = S.n
    fam_no, fam_lapse, l_no, l_lapse = [], [], [], []
    if rg == "R+":
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            dn = S.disc(0, k)
            for j in list(range(k + MIN_SEG, n - MIN_SEG + 1)) + [n]:
                if S.disc(k, j) < dn + max(delta, 1e-9) or (j < n and S.wrate(j, n) >= S.wrate(k, j)):
                    continue
                fam_no.append(S.ml(0, k) + S.ml(k, j) + (S.ml(j, n) if j < n else 0.0))
                l_no.append(k)
        for k in range(MIN_SEG, n - 2 * MIN_SEG + 1):
            dn, wn = S.disc(0, k), S.wrate(0, k)
            for l in range(k + MIN_SEG, min(k + MAX_LAPSE, n - MIN_SEG) + 1):
                wl = S.wrate(k, l)
                if wl >= wn:
                    continue
                for j in list(range(l + MIN_SEG, n - MIN_SEG + 1)) + [n]:
                    if (S.disc(l, j) < dn + max(delta, 1e-9) or wl >= S.wrate(l, j)
                            or (j < n and S.wrate(j, n) >= S.wrate(l, j))):
                        continue
                    fam_lapse.append(S.ml(0, k) + S.ml(k, l) + S.ml(l, j) + (S.ml(j, n) if j < n else 0.0))
                    l_lapse.append(l)
        null = [S.ml(0, n)] + [S.ml(0, j) + S.ml(j, n) for j in range(MIN_SEG, n - MIN_SEG + 1)
                               if S.wrate(j, n) < S.wrate(0, j)]
        null_lapse = [S.ml_pooled(0, k, l, n) + S.ml(k, l)
                      for k in range(MIN_SEG, n - 2 * MIN_SEG + 1)
                      for l in range(k + MIN_SEG, min(k + MAX_LAPSE, n - MIN_SEG) + 1) if S.wrate(k, l) < S.wrate(0, k)]
        lnull = [logsumexp(null) - np.log(len(null))]
        if null_lapse:
            lnull.append(logsumexp(null_lapse) - np.log(len(null_lapse)))
        log_null = logsumexp(lnull) - np.log(len(lnull))
    else:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            if S.disc(k, n) > S.disc(0, k) - max(delta, 1e-9):
                continue
            fam_no.append(S.ml(0, k) + S.ml(k, n))
            l_no.append(k)
        log_null = S.ml(0, n)
    fams = [(np.asarray(lg), np.asarray(ls)) for lg, ls in ((fam_no, l_no), (fam_lapse, l_lapse)) if lg]
    if not fams:
        return -np.inf, None
    fam_ev = [logsumexp(lg) - np.log(len(lg)) for lg, _ in fams]
    log_change = logsumexp(fam_ev) - np.log(len(fam_ev))
    pk = np.zeros(n)
    for (lg, ls), ev in zip(fams, fam_ev):
        pk += np.bincount(ls, weights=np.exp(lg - logsumexp(lg)) * np.exp(ev - logsumexp(fam_ev)), minlength=n)
    return float((log_change - log_null) / np.log(10)), pk


def one(args):
    sid, d = args
    warnings.filterwarnings("ignore")
    o, wt = np.asarray(d["w_outcomes"], float), np.asarray(d["w_start"], float)
    ft, fy = np.asarray(d["n_start"], float), np.asarray(d["n_outcomes"], float)
    rg, n = d["reward_group"], len(o)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))   # reproducible (hash() is salted per process)
    shifts = rng.integers(max(1, int(0.1 * n)), max(2, int(0.9 * n)) + 1, N_SHIFT)
    perms = [rng.permutation(n) for _ in range(N_SHIFT)]
    fperm = [rng.permutation(len(fy)) for _ in range(N_SHIFT)]
    out = []
    for delta in DELTAS:
        bf, pk = fit(Streams(o, wt, fy, ft), rg, delta)
        nb = np.array([fit(Streams(np.roll(o, s), wt, fy, ft), rg, delta)[0] for s in shifts])
        p_circ = float((np.sum(nb >= bf) + 1) / (N_SHIFT + 1))
        p_shuf = np.nan
        if delta == 0.0:
            ns = np.array([fit(Streams(o[pp], wt, fy[fp], ft), rg, delta)[0] for pp, fp in zip(perms, fperm)])
            p_shuf = float((np.sum(ns >= bf) + 1) / (N_SHIFT + 1))
        lt, lo, hi = summarize_pk(pk) if pk is not None and pk.sum() > 0 else (np.nan, np.nan, np.nan)
        out.append(dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=rg, learning_category=d["learning_category"],
                        n_whisker=n, n_fa=len(fy), delta=delta, log10_bf=bf, p_circ=p_circ, p_shuffle=p_shuf,
                        LT=lt, LT_ci05=lo, LT_ci95=hi, learner=bool(p_circ < ALPHA and np.isfinite(lt)), pk=pk))
    return out


def main():
    os.chdir(REPO)
    inp = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    only = set(sys.argv[1:])
    jobs = [(sid, d) for sid, d in inp.items() if not only or d["mouse_id"] in only]
    rows = []
    with ProcessPoolExecutor(min(N_WORKERS, len(jobs))) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 10 == 0:
                print(i, "/", len(jobs), flush=True)
    R = pd.DataFrame(rows)
    tag = ("_" + "_".join(sorted(only))) if only else ""
    pickle.dump({(r.session_id, r.delta): r.pk for r in R.itertuples()}, open(ART / f"035_single_cp_pk{tag}.pkl", "wb"))
    R = R.drop(columns=["pk"])
    R.to_csv(ART / f"035_single_cp_delta{tag}.csv", index=False)
    R["cat"] = R.learning_category.fillna("NA")
    pd.set_option("display.width", 220)
    print(R.groupby(["delta", "reward_group", "cat"]).learner.sum().unstack(fill_value=0))
    print("n per category:", R[R.delta == 0].groupby(["reward_group", "cat"]).size().to_dict())
    print("delta 0: learners circ-shift", R[(R.delta == 0)].groupby("reward_group").learner.sum().to_dict(),
          "| shuffle p<.05", R[(R.delta == 0) & (R.p_shuffle < 0.05)].groupby("reward_group").size().to_dict())
    print(R[R.mouse_id.isin(["AB118", "MH070", "AB104", "MH068", "MH013"])][
        ["mouse_id", "delta", "log10_bf", "p_circ", "LT", "LT_ci05", "LT_ci95", "learner"]].to_string(index=False))


if __name__ == "__main__":
    main()
