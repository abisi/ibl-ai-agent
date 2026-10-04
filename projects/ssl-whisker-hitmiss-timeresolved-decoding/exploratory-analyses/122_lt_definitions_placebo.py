"""122 -- Placebo-split test for every learning-trial (LT) definition (user 2026-10-01: "run the placebo test").
Question: is the pre/post change in decodability at a definition's REAL learning trial larger than at arbitrary split
points of the same session? The definitions are built from the same behaviour that changes, so a pre/post gain at the
LT is partly expected; a split anywhere else in the session also contains drift, engagement and hit-rate change. The
placebo distribution of each session is the null.
Same decoding as 118 (whole brain; hit vs miss 5-50 / 5-100 ms stimulus-aligned; whisker vs auditory -150..0 ms from the
corrected first lick; one C per session x window; size-matched balanced accuracy per epoch; min 2 trials per class per
epoch), terminal disengagement dropped with rule A1 (library default since 2026-10-01).
Split points: every STEP-th whisker trial of the curve-aligned index (the index LTs count in) that leaves >= 2 trials of
each class in both epochs, PLUS every definition's real LT (the session-half split included). Decoded trials split by
the start time of whisker trial k (k itself -> post), as in 095/118.
Statistic per split: delta = acc_post_matched - acc_pre_matched (raw; no per-split shift null -- the placebo splits share
the session's drift and class structure, so the placebo distribution is the null). All splits use the SAME estimator
(N_SUBSAMPLE x N_REPEATS), so real and placebo values are comparable.
Per session x definition (in 122b): percentile of the real delta among placebo deltas with |k - LT| >= EXCLUDE_NEAR.
Output: 122_lt_placebo_whole_brain.parquet (one row per session x decoding x window x split).
Run (haas, repo root): python .../122_lt_definitions_placebo.py
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = str(OUT_DIR.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, str(OUT_DIR))
M118 = importlib.import_module("118_lt_definitions_split_decoding")
LT_TABLE, DEFS, WINDOWS, DZ = M118.LT_TABLE, M118.DEFS, M118.WINDOWS, M118.DZ
MIN_PER_CLASS = M118.MIN_TRIALS_PER_CLASS_EPOCH
STEP = int(os.environ.get("SSL_PLACEBO_STEP", "4"))
N_SUBSAMPLE, N_REP_SUB, N_REP_FULL = 20, 5, 5
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
OUT_PATH = OUT_DIR / f"122_lt_placebo_whole_brain{os.environ.get('SSL_PLACEBO_TAG', '')}.parquet"   # "_step1": every whisker trial (2026-10-01)


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def process(args):
    sid, subject, rg, decoding = args
    sys.path.insert(0, SCRIPTS_DIR)
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg, decoding=decoding, area_col="whole_brain",
                n_subsample=N_SUBSAMPLE, step=STEP, disengagement_rule="A1 (>=5 whisker, >=1 auditory)")
    tr = T.prep_hitmiss_trials(root, sid, st, tt) if decoding == "hitmiss" else T.prep_lick_aligned_trials(root, sid, st, tt)
    if tr is None or not len(tr):
        return [dict(base, skipped_reason="no usable trials")]
    dis = T.detect_terminal_disengagement(sid, tt)
    n0 = len(tr)
    if dis["disengaged"]:
        tr = tr[tr["start_time"] < dis["t_cut"]].reset_index(drop=True)
    base.update(disengaged=dis["disengaged"], n_dropped_disengaged=n0 - len(tr))
    y = (tr["lick_flag"] == 1).to_numpy() if decoding == "hitmiss" else (tr["trial_type"] == "whisker_trial").to_numpy()
    start = tr["start_time"].to_numpy()
    n = len(y)
    units = T.area_units(sid, "whole_brain", "All units", labels)
    if len(units) < T.MIN_UNITS_PER_AREA:
        return [dict(base, skipped_reason="too few units")]
    spikes = T.load_session_unit_spikes(root, sid)
    wins = list(WINDOWS[decoding].values())
    if decoding == "hitmiss":
        mats = T.sliding_bin_population_matrices(spikes, units, start, np.ones(n, bool), wins, dead_zone=DZ)
    else:
        mats = T.lick_aligned_bin_population_matrices(spikes, units, tr["first_lick_time"].to_numpy(), start,
                                                      (tr["trial_type"] == "whisker_trial").to_numpy(), wins, dead_zone=DZ)
    feats = dict(zip(WINDOWS[decoding], mats))
    rng0 = np.random.default_rng(zlib.crc32(f"{decoding}|{sid}|placebo".encode()))
    ok_rows = {w: ~np.isnan(X).any(1) for w, X in feats.items()}
    Cs = {w: T.select_fixed_c_pooled(X[ok_rows[w]], y[ok_rows[w]], rng0, n_folds=M118.MAX_FOLDS) for w, X in feats.items()}
    cw = T._active_trials_from_whisker_onset_for_curve(sid, tt)
    cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
    lt_tab = pd.read_csv(LT_TABLE).set_index("session_id")
    # candidate splits (whisker-trial index k); real LTs of every definition added
    real = {}
    for d in DEFS:
        if d == "half":
            continue
        v = lt_tab[d].get(sid, np.nan) if d in lt_tab.columns else np.nan
        if pd.notna(v) and int(v) < len(cw_t):
            real.setdefault(int(v), []).append(d)
    ks = sorted(set(range(STEP, len(cw_t), STEP)) | set(real))
    splits = [(k, float(cw_t[k]), "|".join(real.get(k, []))) for k in ks]
    # session-half split of the decoded trials (as 118 "half")
    splits.append((-1, float(start[n // 2]) if n else np.nan, "half"))
    rows = []
    for k, t_split, is_real_for in splits:
        is_pre = start < t_split
        cnt = (int(y[is_pre].sum()), int((~y[is_pre]).sum()), int(y[~is_pre].sum()), int((~y[~is_pre]).sum()))
        b = dict(base, split_k=k, t_split=t_split, real_for=is_real_for, n_trials=n, n_pre=int(is_pre.sum()),
                 pre_pos=cnt[0], pre_neg=cnt[1], post_pos=cnt[2], post_neg=cnt[3], n_whisker_curve=len(cw_t))
        if min(cnt) < MIN_PER_CLASS:
            if is_real_for:
                rows.append(dict(b, skipped_reason="class counts < min"))
            continue
        rng = np.random.default_rng(zlib.crc32(f"{decoding}|{sid}|{k}".encode()))
        for w, X in feats.items():
            t0 = time.time()
            r = M118.epoch_scores(X, y, is_pre, Cs[w], rng, T.decode_bin_pooled, N_SUBSAMPLE, N_REP_SUB, N_REP_FULL)
            if r is None:
                continue
            rows.append(dict(b, window=w, n_units=len(units), C=Cs[w], **r,
                             delta_matched=r["acc_post_matched"] - r["acc_pre_matched"],
                             delta_full=r["acc_post_full"] - r["acc_pre_full"], compute_s=time.time() - t0,
                             skipped_reason=None))
    return rows or [dict(base, skipped_reason="no valid split")]


def main():
    os.chdir(OUT_DIR.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    done = set()
    if OUT_PATH.exists():
        d = pd.read_parquet(OUT_PATH, columns=["session_id", "decoding"])
        done = set(zip(d.session_id, d.decoding))
    args = [(r.session_id, r.subject_id, r.reward_group, dec) for dec in ("modality_lick", "hitmiss")
            for r in sess.itertuples() if (r.session_id, dec) not in done]
    print(f"[122] {len(args)} session x decoding tasks, STEP {STEP}, N_SUBSAMPLE {N_SUBSAMPLE}, {N_WORKERS} workers",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, a): a for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                rows = [dict(session_id=a[0], mouse_id=a[1], reward_group=a[2], decoding=a[3], skipped_reason=f"error: {e!r}")]
            new = pd.DataFrame(rows)
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            print(f"[122] [{i}/{len(args)}] {a[0]} {a[3]}: {len(new)} rows -- {time.time() - t0:.0f}s", flush=True)
    print("[122] DONE", flush=True)


if __name__ == "__main__":
    main()
