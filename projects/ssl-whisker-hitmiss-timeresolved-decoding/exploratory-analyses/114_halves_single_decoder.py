"""114 -- Within-session change with ONE decoder per session (user request 2026-09-28): instead of decoding each
session half separately (which needs enough hits AND misses in each half), train on the whole session and score the
held-out predictions separately in the 1st and 2nd half (same median split as the 024 'half' condition).

Hit/miss (stimulus-aligned), learning stage, whole brain and area_group. Pipeline decoder (StandardScaler + L2
logistic, one C per session/area from select_fixed_c_pooled on the wide window), pooled stratified CV with
n_folds = min(5, minority count), N_REP repeats. Whole-session training needs >= 3 hits and >= 3 misses; a half is
scored only if it has >= MIN_HALF hits and >= MIN_HALF misses (MIN_HALF = 3). Per half: balanced accuracy of the
pooled held-out predictions, averaged over repeats.
  whole brain: time course on causal 50-ms bins with STRIDE_S stride (coarser than 024's 5 ms, for cost) + windows
  area_group:  windows only (baseline -200..-10 ms, sensory 5..50 ms)
Chance: linear-shift null scored the same way (N_SHIFT shifts of 10-50% of the session, labels shifted relative to
the neural data, decoder retrained, held-out predictions scored per half). The half assignment follows the NEURAL
trial index (the trials being predicted). Values reported as accuracy - null per half.
Output: 114_halves_single_decoder.parquet (session x area x half; curves for whole brain), rows compatible with the
110/113 loaders (condition_type = "half").
Run (haas): python 114_halves_single_decoder.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
OUT_PATH = OUT / "114_halves_single_decoder.parquet"
WINDOWS = {"baseline": (-0.200, -0.010), "sensory": (0.005, 0.050)}
STIM_WINDOW = (-0.2, 0.6)
BIN_W, STRIDE_S = 0.05, 0.02
DZ = (-0.010, 0.005)
N_REP, N_SHIFT, MIN_HALF = 5, 10, 3
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "60"))


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def half_scores(X, y, half2, C, rng):
    """Pooled held-out predictions of one whole-session decoder -> balanced accuracy per half (NaN if a half lacks
    MIN_HALF of either class)."""
    from sklearn.metrics import balanced_accuracy_score
    from ssl_timeresolved_decoding import _effective_folds, _make_classifier, _stratified_kfold_with_resample
    ok = ~np.isnan(X).any(axis=1)
    X, y, half2 = X[ok], y[ok], half2[ok]
    nf = _effective_folds(y, 5)
    out = np.full(2, np.nan)
    if nf < 2:
        return out
    halves = (~half2, half2)
    scorable = [min(y[h].sum(), (~y[h]).sum()) >= MIN_HALF for h in halves]
    if not any(scorable):
        return out
    acc = [[], []]
    for _ in range(N_REP):
        splits = _stratified_kfold_with_resample(X, y, nf, rng, 20)
        if splits is None:
            continue
        pred = np.empty(len(y), bool)
        for tr, te in splits:
            pred[te] = _make_classifier(C).fit(X[tr], y[tr]).predict(X[te])
        for j, h in enumerate(halves):
            if scorable[j]:
                acc[j].append(balanced_accuracy_score(y[h], pred[h]))
    return np.array([np.mean(a) if a else np.nan for a in acc])


def process(args):
    sid, subject_id, rg = args
    sys.path.insert(0, SCRIPTS)
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, areas_with_enough_units, causal_bin_edges,
        load_session_unit_spikes, prep_hitmiss_trials, select_fixed_c_pooled, sliding_bin_population_matrices,
        wide_window_matrix_from_bins,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    tr = prep_hitmiss_trials(root, sid, st, tt)
    base = dict(session_id=sid, subject_id=subject_id, reward_group=rg, decode_target="hitmiss", alignment="stim",
                method="single whole-session decoder, held-out predictions scored per half", n_rep=N_REP, n_shift=N_SHIFT,
                min_half_per_class=MIN_HALF)
    if tr is None:
        return [dict(base, skipped_reason="no usable trials")]
    y = tr["lick_flag"].to_numpy().astype(bool)
    n = len(y)
    half2 = (tr["half"] == "second").to_numpy()
    if min(y.sum(), (~y).sum()) < 3:
        return [dict(base, n_trials=n, skipped_reason=f"whole-session class counts hit={int(y.sum())}/miss={int((~y).sum())} (< 3)")]
    t = tr["start_time"].to_numpy()
    spikes = load_session_unit_spikes(root, sid)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    edges = causal_bin_edges(STIM_WINDOW, bin_width=BIN_W, stride=STRIDE_S)
    shifts = rng.choice(np.arange(int(0.1 * n), int(0.5 * n) + 1), size=N_SHIFT, replace=True) * rng.choice([-1, 1], N_SHIFT)
    rows = []
    for area_col in ("whole_brain", "area_group"):
        for area in areas_with_enough_units(sid, area_col, labels):
            units = area_units(sid, area_col, area, labels)
            wins = sliding_bin_population_matrices(spikes, units, t, np.ones(n, bool), list(WINDOWS.values()), dead_zone=DZ)
            W = dict(zip(WINDOWS, wins))
            bins = sliding_bin_population_matrices(spikes, units, t, np.ones(n, bool), edges, dead_zone=DZ) if area_col == "whole_brain" else []
            C = select_fixed_c_pooled(wide_window_matrix_from_bins(bins) if bins else W["sensory"], y, rng)
            real = {w: half_scores(W[w], y, half2, C, rng) for w in WINDOWS}
            curve = np.array([half_scores(X, y, half2, C, rng) for X in bins]) if bins else None
            null = {w: [] for w in WINDOWS}
            null_curve = []
            for k in shifts:
                ak = abs(int(k))
                nsl, bsl = (slice(0, n - ak), slice(ak, n)) if k > 0 else (slice(ak, n), slice(0, n - ak))
                ys, hs = y[bsl], half2[nsl]
                for w in WINDOWS:
                    null[w].append(half_scores(W[w][nsl], ys, hs, C, rng))
                if bins:
                    null_curve.append(np.array([half_scores(X[nsl], ys, hs, C, rng) for X in bins]))
            for j, hv in enumerate(("first", "second")):
                h = half2 if hv == "second" else ~half2
                r = dict(base, area_col=area_col, area_value=area, condition_type="half", condition_value=hv, n_units=len(units),
                         n_trials=int(h.sum()), n_hits=int(y[h].sum()), n_misses=int((~y[h]).sum()), C=C,
                         skipped_reason=None if np.isfinite(real["sensory"][j]) else "half not scorable (< 3 hits or misses)")
                for w in WINDOWS:
                    nv = np.array([x[j] for x in null[w]])
                    r[f"real_{w}"] = real[w][j]
                    r[f"null_{w}"] = np.nanmean(nv) if np.isfinite(nv).any() else np.nan
                    r[f"corr_{w}"] = r[f"real_{w}"] - r[f"null_{w}"]
                if bins:
                    nc = np.array([x[:, j] for x in null_curve])
                    r["real_curve"] = curve[:, j].tolist()
                    r["null_mean_curve"] = np.nanmean(nc, 0).tolist()
                rows.append(r)
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import causal_bin_edges, hitmiss_session_list
    sess = hitmiss_session_list(pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"]
    (OUT / "114_bin_edges.json").write_text(json.dumps(causal_bin_edges(STIM_WINDOW, bin_width=BIN_W, stride=STRIDE_S)))
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    print(f"[114] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, (r.session_id, r.subject_id, r.reward_group)): r.session_id for r in todo.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                new = pd.DataFrame(f.result())
            except Exception as e:  # noqa: BLE001
                new = pd.DataFrame([dict(session_id=futs[f], skipped_reason=f"error: {e!r}")])
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            print(f"[114] [{i}/{len(futs)}] {futs[f]} {time.time() - t0:.0f}s", flush=True)
    print("[114] DONE", flush=True)


if __name__ == "__main__":
    main()
