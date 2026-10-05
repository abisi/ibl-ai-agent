"""139 -- Hit-median session split vs midpoint split, whole brain, learning stage (user 2026-10-05: "update results but
forgetting about learning trials, splitting sessions at halves that are not mid-point but 50% hits before and after ... both
whole-session decoders and separate decoders evaluated"; design reviewed with the user the same day).
Trials (skills/ssl-trial-exclusion): prep_hitmiss_trials / prep_lick_aligned_trials (active context, perf != 6, warm-up cut),
end-of-session disengagement trimmed with rule A1.
Splits (per session):
  hitmedian  whisker trials in time order; H = number of hits (lick on a whisker trial, same label in both cohorts); the split
             is at the (H//2 + 1)-th hit: "before" holds the first H//2 hits (the A1 trim removes only trailing misses, so it
             does not move the split). Lick-aligned trials are assigned by start time.
  mid        median split of the decoded trials (reference).
Decodings and windows: hit vs miss (whisker trials, y = lick): baseline -200..-10 ms, 5-35, 5-50, 5-100 ms after the stimulus
(dead zone -10..+5 ms); whisker vs auditory (licked trials, y = whisker): -100..0 ms before the corrected first lick.
Decoders (StandardScaler + L2 logistic regression; ONE C per session x window from all trials, select_fixed_c_pooled):
  separate   one decoder per half; count-matched across halves (each half subsampled to the smaller half's count of each
             class, N_SUB subsamples, pooled stratified CV); >= MIN_CLASS trials per class per half.
             cross-half generalisation: train on one half's matched subsample, test on the other's (both directions).
  single     one decoder on all trials (pooled stratified CV, N_REP repeats); held-out predictions scored per half.
Null (within half): labels shifted against the neural trials WITHIN each half (non-wrapping, 10-50 % of that half's trials,
independently in each half), the same decoder rerun; N_SHIFT shifts. Values: accuracy - mean null, per half.
Output: 139_hitmedian_split_whole_brain.parquet (session x decoding x split x window, with split description columns).
Run (haas, repo root): python .../139_hitmedian_split_decoding.py
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

OUT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = str(OUT_DIR.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
WINDOWS = {"hitmiss": {"baseline": (-0.200, -0.010), "5-35ms": (0.005, 0.035), "5-50ms": (0.005, 0.050), "5-100ms": (0.005, 0.100)},
           "modality_lick": {"-100-0ms": (-0.100, 0.0)}}
DZ = (-0.010, 0.005)
MIN_CLASS, MAX_FOLDS = 3, 5
N_SUB, N_REP_SUB = 30, 3
N_REP_SINGLE = 5
N_SHIFT, N_SUB_SHIFT = 20, 3
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
OUT_PATH = OUT_DIR / "139_hitmedian_split_whole_brain.parquet"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def counts(y):
    return int(y.sum()), int((~y).sum())


def separate(Xa, ya, Xb, yb, C, rng, T, n_sub, n_rep, cross=True):
    """count-matched per-half accuracy (pooled CV) and cross-half generalisation; None if a class is short"""
    from ssl_timeresolved_decoding import _make_classifier
    from sklearn.metrics import balanced_accuracy_score
    (pa, na), (pb, nb) = counts(ya), counts(yb)
    tp, tn = min(pa, pb), min(na, nb)
    if min(tp, tn) < MIN_CLASS:
        return None
    nf = min(MAX_FOLDS, tp, tn)
    ia = (np.where(ya)[0], np.where(~ya)[0]); ib = (np.where(yb)[0], np.where(~yb)[0])
    acc_a, acc_b, x_ab, x_ba = [], [], [], []
    for _ in range(n_sub):
        sa = np.r_[rng.choice(ia[0], tp, replace=False), rng.choice(ia[1], tn, replace=False)]
        sb = np.r_[rng.choice(ib[0], tp, replace=False), rng.choice(ib[1], tn, replace=False)]
        acc_a.append(T.decode_bin_pooled(Xa[sa], ya[sa], C, rng, n_repeats=n_rep, n_folds=nf))
        acc_b.append(T.decode_bin_pooled(Xb[sb], yb[sb], C, rng, n_repeats=n_rep, n_folds=nf))
        if cross:
            x_ab.append(balanced_accuracy_score(yb[sb], _make_classifier(C).fit(Xa[sa], ya[sa]).predict(Xb[sb])))
            x_ba.append(balanced_accuracy_score(ya[sa], _make_classifier(C).fit(Xb[sb], yb[sb]).predict(Xa[sa])))
    out = dict(matched_n_pos=tp, matched_n_neg=tn, acc_1=float(np.nanmean(acc_a)), acc_2=float(np.nanmean(acc_b)))
    if cross:
        out.update(cross_1to2=float(np.mean(x_ab)), cross_2to1=float(np.mean(x_ba)))
    return out


def single(X, y, is1, C, rng, n_rep):
    """one whole-session decoder, held-out predictions scored per half (balanced accuracy)"""
    from sklearn.metrics import balanced_accuracy_score
    from ssl_timeresolved_decoding import _effective_folds, _make_classifier, _stratified_kfold_with_resample
    nf = _effective_folds(y, MAX_FOLDS)
    halves = (is1, ~is1)
    if nf < 2 or any(min(counts(y[h])) < MIN_CLASS for h in halves):
        return None
    acc = [[], []]
    for _ in range(n_rep):
        splits = _stratified_kfold_with_resample(X, y, nf, rng, 20)
        if splits is None:
            continue
        pred = np.empty(len(y), bool)
        for tr, te in splits:
            pred[te] = _make_classifier(C).fit(X[tr], y[tr]).predict(X[te])
        for j, h in enumerate(halves):
            acc[j].append(balanced_accuracy_score(y[h], pred[h]))
    return None if not acc[0] else dict(acc_1=float(np.mean(acc[0])), acc_2=float(np.mean(acc[1])))


def shift_within(y, idx, rng):
    """labels of the trials idx shifted against their neural trials (non-wrapping, 10-50 % of len(idx)); returns
    (neural index, label) arrays of the kept pairs"""
    m = len(idx)
    k = int(rng.integers(max(1, int(MIN_SHIFT_FRAC * m)), max(2, int(MAX_SHIFT_FRAC * m)) + 1))
    if rng.random() < 0.5:
        return idx[: m - k], y[idx[k:]]
    return idx[k:], y[idx[: m - k]]


def hit_median_time(tr_w):
    """start time of the (H//2 + 1)-th hit of the time-ordered whisker trials (first H//2 hits before the split)"""
    lick = (tr_w["lick_flag"] == 1).to_numpy()
    H = int(lick.sum())
    if H < 2:
        return np.nan, H
    k = np.flatnonzero(lick)[H // 2]
    return float(tr_w["start_time"].to_numpy()[k]), H


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
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg, decoding=decoding, area_col="whole_brain", min_class=MIN_CLASS,
                n_sub=N_SUB, n_shift=N_SHIFT, disengagement_rule="A1")
    trw = T.prep_hitmiss_trials(root, sid, st, tt)
    tr = trw if decoding == "hitmiss" else T.prep_lick_aligned_trials(root, sid, st, tt)
    if tr is None or not len(tr) or trw is None or not len(trw):
        return [dict(base, skipped_reason="no usable trials")]
    dis = T.detect_terminal_disengagement(sid, tt)
    if dis["disengaged"]:
        tr = tr[tr["start_time"] < dis["t_cut"]].reset_index(drop=True)
        trw = trw[trw["start_time"] < dis["t_cut"]].reset_index(drop=True)
    trw = trw.sort_values("start_time").reset_index(drop=True)
    tr = tr.sort_values("start_time").reset_index(drop=True)
    y = (tr["lick_flag"] == 1).to_numpy() if decoding == "hitmiss" else (tr["trial_type"] == "whisker_trial").to_numpy()
    start = tr["start_time"].to_numpy()
    n = len(y)
    t_hm, H = hit_median_time(trw)
    wl = (trw["lick_flag"] == 1).to_numpy(); wt = trw["start_time"].to_numpy()
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
    rng0 = np.random.default_rng(zlib.crc32(f"139|{decoding}|{sid}".encode()))
    Cs = {}
    for w, X in feats.items():
        ok = ~np.isnan(X).any(1)
        Cs[w] = T.select_fixed_c_pooled(X[ok], y[ok], rng0, n_folds=MAX_FOLDS)
    rows = []
    for split in ("hitmedian", "mid"):
        if split == "hitmedian":
            if not np.isfinite(t_hm):
                rows.append(dict(base, split=split, skipped_reason=f"{H} whisker hits"))
                continue
            is1 = start < t_hm
            t_split = t_hm
        else:
            is1 = np.arange(n) < n // 2
            t_split = float(start[n // 2]) if n else np.nan
        w1 = wt < t_split
        b = dict(base, split=split, t_split=t_split, n_trials=n, n_1=int(is1.sum()), n_2=int((~is1).sum()),
                 pos_1=int(y[is1].sum()), neg_1=int((~y[is1]).sum()), pos_2=int(y[~is1].sum()), neg_2=int((~y[~is1]).sum()),
                 n_whisker=len(wt), whisker_hits=H, split_frac_whisker=float(w1.mean()) if len(wt) else np.nan,
                 split_frac_time=float((t_split - wt[0]) / (wt[-1] - wt[0])) if len(wt) > 1 else np.nan,
                 hit_rate_1=float(wl[w1].mean()) if w1.any() else np.nan, hit_rate_2=float(wl[~w1].mean()) if (~w1).any() else np.nan,
                 whisker_miss_1=int((~wl[w1]).sum()), whisker_miss_2=int((~wl[~w1]).sum()))
        for w, X in feats.items():
            t0 = time.time()
            ok = ~np.isnan(X).any(1)
            Xo, yo, i1 = X[ok], y[ok], is1[ok]
            rng = np.random.default_rng(zlib.crc32(f"139|{decoding}|{sid}|{split}|{w}".encode()))
            row = dict(b, window=w, n_units=len(units), C=Cs[w])
            sep = separate(Xo[i1], yo[i1], Xo[~i1], yo[~i1], Cs[w], rng, T, N_SUB, N_REP_SUB)
            sgl = single(Xo, yo, i1, Cs[w], rng, N_REP_SINGLE)
            if sep is None and sgl is None:
                rows.append(dict(row, skipped_reason=f"class counts < {MIN_CLASS} per half"))
                continue
            idx1, idx2 = np.where(i1)[0], np.where(~i1)[0]
            ns1, ns2, ng1, ng2 = [], [], [], []
            for _ in range(N_SHIFT):
                a1, l1 = shift_within(yo, idx1, rng); a2, l2 = shift_within(yo, idx2, rng)
                if sep is not None:
                    r = separate(Xo[a1], l1, Xo[a2], l2, Cs[w], rng, T, N_SUB_SHIFT, 2, cross=False)
                    ns1.append(np.nan if r is None else r["acc_1"]); ns2.append(np.nan if r is None else r["acc_2"])
                if sgl is not None:
                    Xs, ys = Xo[np.r_[a1, a2]], np.r_[l1, l2]
                    hs = np.r_[np.ones(len(a1), bool), np.zeros(len(a2), bool)]
                    r = single(Xs, ys, hs, Cs[w], rng, 2)
                    ng1.append(np.nan if r is None else r["acc_1"]); ng2.append(np.nan if r is None else r["acc_2"])
            mean = lambda v: float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else np.nan
            if sep is not None:
                row.update(sep_acc_1=sep["acc_1"], sep_acc_2=sep["acc_2"], sep_null_1=mean(ns1), sep_null_2=mean(ns2),
                           cross_1to2=sep["cross_1to2"], cross_2to1=sep["cross_2to1"], matched_n_pos=sep["matched_n_pos"],
                           matched_n_neg=sep["matched_n_neg"])
                row.update(sep_corr_1=row["sep_acc_1"] - row["sep_null_1"], sep_corr_2=row["sep_acc_2"] - row["sep_null_2"])
                row["sep_change"] = row["sep_corr_2"] - row["sep_corr_1"]
                row["cross_minus_within"] = 0.5 * ((sep["cross_1to2"] - sep["acc_2"]) + (sep["cross_2to1"] - sep["acc_1"]))
            if sgl is not None:
                row.update(sgl_acc_1=sgl["acc_1"], sgl_acc_2=sgl["acc_2"], sgl_null_1=mean(ng1), sgl_null_2=mean(ng2))
                row.update(sgl_corr_1=row["sgl_acc_1"] - row["sgl_null_1"], sgl_corr_2=row["sgl_acc_2"] - row["sgl_null_2"])
                row["sgl_change"] = row["sgl_corr_2"] - row["sgl_corr_1"]
            row.update(skipped_reason=None, compute_s=time.time() - t0)
            rows.append(row)
    return rows


def main():
    os.chdir(OUT_DIR.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    lim = int(os.environ.get("SSL_139_LIMIT", "0"))
    if lim:
        sess = pd.concat([sess[sess.reward_group == c].head(lim) for c in ("R+", "R-")])
    done = set()
    if OUT_PATH.exists():
        d = pd.read_parquet(OUT_PATH, columns=["session_id", "decoding"])
        done = set(zip(d.session_id, d.decoding))
    args = [(r.session_id, r.subject_id, r.reward_group, dec) for dec in ("modality_lick", "hitmiss")
            for r in sess.itertuples() if (r.session_id, dec) not in done]
    print(f"[139] {len(args)} session x decoding tasks, {N_WORKERS} workers", flush=True)
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
            tmp = OUT_PATH.with_suffix(".partial.parquet"); out.to_parquet(tmp, index=False); os.replace(tmp, OUT_PATH)
            print(f"[139] [{i}/{len(args)}] {a[0]} {a[3]}: {int(new.skipped_reason.isna().sum()) if 'skipped_reason' in new else 0} rows ok"
                  f" -- {time.time() - t0:.0f}s", flush=True)
    print("[139] DONE", flush=True)


if __name__ == "__main__":
    main()
