"""183 -- From the hit - miss axis to subspaces (user 2026-11-24: "1. do this"; Kingsbury et al. 2025 bioRxiv, subspace
divergence across contexts). Question: do R- passive whisker responses after the task leave the active-task subspace, or rotate
within it, away from the hit - miss direction?

Same sessions, units, trials, scaling and shift null as 179 (whole brain only): responses = rate in 10-ms bins starting 5, 10,
..., 35 ms (7 bins, 5-45 ms; active trials with a lick before 35 ms excluded, as 179) minus the epoch's mean baseline (-55 to
-20 ms), / the unit's s.d. over the active trials (5-35 ms window, as 179). N = neurons.
Subspaces (uncentred, they keep the mean evoked pattern; SVD of condition-mean bins):
  task      T: top K_TASK right singular vectors of [mean hit (7 bins); mean miss (7 bins)]        (14 x N)
  decision  D: top K_DEC right singular vectors of [mean hit - mean miss, 7 bins]                  (7 x N)
            + the same on linearly shifted hit / miss labels (179 shifts) -> D_k (null)
  stimulus  S: top K_DEC of [mean active whisker - mean active auditory, 7 bins] (reference)
Readout for a passive response (whisker W or auditory A; epoch pre / post): cross-validated fraction of its energy in a subspace,
  f(V) = mean_b <P_V r1_b, P_V r2_b> / mean_b <r1_b, r2_b>, r1 / r2 = bin means of two random halves of the trials (N_HALF
  splits; numerator and denominator averaged over splits before the ratio). Noise does not inflate it (cross-validated).
  f_task, f_dec (+ f_dec_null = mean over D_k), f_stim; also the epoch's temporal halves (h1, h2) for a within-epoch floor.
  Chance for a random K-dim subspace = K / N.
Principal angles (reference): between T and the passive pre / post subspaces is not used (passive trial counts too small for a
stable PCA); the fractions above are the projection version of the same question.
Outputs: 183_subspace_alignment{_expert}.parquet (one row per session x stim). Run (haas, repo root):
  .venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/183_subspace_alignment.py [--pilot]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
C179 = importlib.import_module("179_cd_clean")
BINS = [(s, s + 0.010) for s in np.round(np.arange(0.005, 0.0351, 0.005), 3)]
K_TASK, K_DEC = 3, 2
N_HALF = 50
STAGE = os.environ.get("SSL_STAGE", "learning")
OUT = EA / f"183_subspace_alignment{'' if STAGE == 'learning' else '_' + STAGE}.parquet"


def top(M, k):
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    return vt[:k].T                                                    # N x k, orthonormal


def cv_frac(R, Vs, rng, idx=None):
    """R: bins x trials x N (passive, one stimulus); Vs: list of N x k bases -> cv energy fraction per basis"""
    if idx is not None:
        R = R[:, idx]
    n = R.shape[1]
    num, den = np.zeros(len(Vs)), 0.0
    for _ in range(N_HALF):
        p = rng.permutation(n)
        a, b = R[:, p[: n // 2]].mean(1), R[:, p[n // 2:]].mean(1)     # bins x N
        den += np.sum(a * b)
        for i, V in enumerate(Vs):
            num[i] += np.sum((a @ V) * (b @ V))
    return num / den if den > 0 else num * np.nan


def process(args):
    sid, subject, rg = args
    C179._init()
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(base, skipped_reason="no passive pre + post or no active")]
    act = trs["active"].sort_values("start_time").reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    keep = ~licked | (rt > C179.WIN[1])
    act, licked = act[keep].reset_index(drop=True), licked[keep]
    trs = dict(trs, active=act)
    allu = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), C179.SET_IDS.get(sid, np.array([], dtype=np.int64)))
    if len(allu) < C179.MIN_UNITS_ALL:
        return [dict(base, skipped_reason="too few tracked units")]
    isw = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in ("passive_pre", "active", "passive_post")}
    widx = np.where(isw["active"])[0]
    y = licked[widx]
    if min(y.sum(), (~y).sum()) < C179.MIN_CLASS:
        return [dict(base, skipped_reason="< 3 hits or misses")]
    spikes = T.load_session_unit_spikes(root, sid)
    B, X = {}, {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy()
        one = np.ones(len(t0), bool)
        out = T.sliding_bin_population_matrices(spikes, allu, t0, one, BINS + [C179.WIN, C179.BASE], dead_zone=C179.DZ)
        bm = np.nanmean(out[-1], 0, keepdims=True)
        B[e] = np.stack(out[: len(BINS)], 0) - bm[None]                 # bins x trials x units
        X[e] = out[-2] - bm
    sd = X["active"].std(0)
    ok = sd > 1e-6
    N = int(ok.sum())
    Z = {e: B[e][:, :, ok] / sd[ok] for e in B}
    hit, miss = widx[y], widx[~y]
    shifts = C179.shifted_labels(y, widx)
    A = Z["active"]
    Tb = top(np.vstack([A[:, hit].mean(1), A[:, miss].mean(1)]), K_TASK)
    Db = top(A[:, hit].mean(1) - A[:, miss].mean(1), K_DEC)
    Dk = [top(A[:, xi[ys]].mean(1) - A[:, xi[~ys]].mean(1), K_DEC) for xi, ys in shifts]
    Sb = top(A[:, isw["active"]].mean(1) - A[:, ~isw["active"]].mean(1), K_DEC)
    Vs = [Tb, Db, Sb] + Dk
    rng = np.random.default_rng(zlib.crc32(f"{sid}|183".encode()))
    rows = []
    for s_k, sel in (("W", True), ("A", False)):
        r = dict(base, stim=s_k, n_units=N, n_shifts=len(Dk), chance_task=K_TASK / N, chance_dec=K_DEC / N, skipped_reason=None)
        for e, k in (("passive_pre", "pre"), ("passive_post", "post")):
            R = Z[e][:, isw[e] == sel]
            order = np.argsort(trs[e].start_time.to_numpy()[isw[e] == sel], kind="stable")
            n = R.shape[1]
            for part, idx in (("", None), ("_h1", order[: n // 2]), ("_h2", order[n // 2:])):
                f = cv_frac(R, Vs, rng, idx)
                r[f"f_task_{k}{part}"], r[f"f_dec_{k}{part}"], r[f"f_stim_{k}{part}"] = f[0], f[1], f[2]
                r[f"f_dec_{k}{part}_null"] = float(np.nanmean(f[3:])) if len(Dk) else np.nan
        rows.append(r)
    print(sid, rg, N, "units", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    TU = importlib.import_module("tracked_units")
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == STAGE) & sess.reward_group.isin(["R+", "R-"])]
    if a.pilot:                                                        # 3 per cohort among the sessions 179 kept
        ok = pd.read_parquet(C179.OUT_M if C179.OUT_M.exists() else EA / "tables" / C179.OUT_M.name)
        ok = set(ok[ok.skipped_reason.isna()].session_id)
        sess = sess[sess.session_id.isin(ok)].groupby("reward_group").head(3)
    C179.SET_IDS.update(TU.load("stable"))
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples()]
    print(f"[183] {len(args)} sessions, stage {STAGE}", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(int(os.environ.get("SSL_DECODE_N_WORKERS", "40")), initializer=C179._init) as ex:
        futs = {ex.submit(process, x): x for x in args}
        for f in as_completed(futs):
            try:
                rows += f.result()
            except Exception as e:  # noqa: BLE001
                x = futs[f]
                rows.append(dict(session_id=x[0], mouse_id=x[1], reward_group=x[2], skipped_reason=f"error: {e!r}"[:300]))
                print("FAILED", x[0], repr(e)[:300], flush=True)
    out = OUT.with_name(OUT.stem + ("_pilot" if a.pilot else "") + ".parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    print("ALL DONE", out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
