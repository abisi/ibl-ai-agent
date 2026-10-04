"""116 -- Within-session change with ONE decoder per session (as 114), generalised (user request 2026-09-30, COSYNE
figure): targets
  hitmiss        stimulus-aligned, whisker trials, y = lick (prep_hitmiss_trials); windows baseline -200..-10 ms,
                 sensory 5..50 ms, sensory100 5..100 ms (the user's 100-ms sensory window);
  modality_lick  first-lick-aligned (corrected first lick), licked whisker + auditory trials, y = whisker
                 (prep_lick_aligned_trials); windows prelick -100..0 ms, early -600..-450 ms;
whole brain (time course: causal 50-ms bins, 20-ms stride, stim -200..600 / lick -600..200 ms, + windows) and
area_group (windows only). Decoder as 114: StandardScaler + L2 logistic, one C per session/area (select_fixed_c_pooled on
the wide window), pooled stratified CV with n_folds = min(5, minority), N_REP repeats; ONE decoder trained on all trials,
held-out predictions scored separately in the 1st and 2nd half (chronological median split, the prep functions' `half`);
a half is scored with >= 3 trials of each class; whole session needs >= 3 of each class. Chance: linear-shift null (N_SHIFT
shifts of 10-50% of the session, labels shifted relative to the neural data, decoder retrained, scored per half the same
way). Whisker-artefact dead zone (-10..+5 ms from stimulus onset) excised per trial.
Output: 116_halves_single_decoder_<target>.parquet (session x area x half; curves for whole brain) and
        116_bin_edges_<target>.json. Rows compatible with 114/115.
Run (haas): python 116_halves_single_decoder_v2.py <hitmiss|modality_lick>
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
SPEC = {"hitmiss": dict(align="stim", window=(-0.2, 0.6), main="sensory100",
                        windows={"baseline": (-0.200, -0.010), "sensory": (0.005, 0.050), "sensory100": (0.005, 0.100)}),
        "modality_lick": dict(align="lick", window=(-0.6, 0.2), main="prelick",
                              windows={"prelick": (-0.100, 0.0), "early": (-0.600, -0.450)})}
BIN_W, STRIDE_S = 0.05, 0.02
DZ = (-0.010, 0.005)
N_REP, N_SHIFT, MIN_HALF = 5, 10, 3
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "30"))


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def half_scores(X, y, half2, C, rng):
    """Pooled held-out predictions of one whole-session decoder -> balanced accuracy per half (NaN if a half lacks
    MIN_HALF of either class). Same as 114."""
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
    target, sid, subject_id, rg = args
    sys.path.insert(0, SCRIPTS)
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    sp = SPEC[target]
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    tr = T.prep_hitmiss_trials(root, sid, st, tt) if target == "hitmiss" else T.prep_lick_aligned_trials(root, sid, st, tt)
    base = dict(session_id=sid, subject_id=subject_id, reward_group=rg, decode_target=target, alignment=sp["align"],
                method="single whole-session decoder, held-out predictions scored per half", n_rep=N_REP, n_shift=N_SHIFT,
                min_half_per_class=MIN_HALF, windows=json.dumps(sp["windows"]))
    if tr is None or not len(tr):
        return [dict(base, skipped_reason="no usable trials")]
    y = (tr["lick_flag"] == 1).to_numpy() if target == "hitmiss" else (tr["trial_type"] == "whisker_trial").to_numpy()
    n = len(y)
    half2 = (tr["half"] == "second").to_numpy()
    if min(y.sum(), (~y).sum()) < 3:
        return [dict(base, n_trials=n, skipped_reason=f"whole-session class counts {int(y.sum())}/{int((~y).sum())} (< 3)")]
    spikes = T.load_session_unit_spikes(root, sid)
    rng = np.random.default_rng(zlib.crc32(f"{target}|{sid}".encode()))
    edges = T.causal_bin_edges(sp["window"], bin_width=BIN_W, stride=STRIDE_S)
    wlist = list(sp["windows"].values())

    def mats(units, e):
        if sp["align"] == "stim":
            return T.sliding_bin_population_matrices(spikes, units, tr["start_time"].to_numpy(), np.ones(n, bool), e,
                                                     dead_zone=DZ)
        return T.lick_aligned_bin_population_matrices(spikes, units, tr["first_lick_time"].to_numpy(),
                                                      tr["start_time"].to_numpy(),
                                                      (tr["trial_type"] == "whisker_trial").to_numpy(), e, dead_zone=DZ)

    shifts = rng.choice(np.arange(int(0.1 * n), int(0.5 * n) + 1), size=N_SHIFT, replace=True) * rng.choice([-1, 1], N_SHIFT)
    rows = []
    for area_col in ("whole_brain", "area_group"):
        for area in T.areas_with_enough_units(sid, area_col, labels):
            units = T.area_units(sid, area_col, area, labels)
            W = dict(zip(sp["windows"], mats(units, wlist)))
            bins = mats(units, edges) if area_col == "whole_brain" else []
            C = T.select_fixed_c_pooled(T.wide_window_matrix_from_bins(bins) if bins else W[sp["main"]], y, rng)
            real = {w: half_scores(W[w], y, half2, C, rng) for w in W}
            curve = np.array([half_scores(X, y, half2, C, rng) for X in bins]) if bins else None
            null = {w: [] for w in W}
            null_curve = []
            for k in shifts:
                ak = abs(int(k))
                nsl, bsl = (slice(0, n - ak), slice(ak, n)) if k > 0 else (slice(ak, n), slice(0, n - ak))
                ys, hs = y[bsl], half2[nsl]
                for w in W:
                    null[w].append(half_scores(W[w][nsl], ys, hs, C, rng))
                if bins:
                    null_curve.append(np.array([half_scores(X[nsl], ys, hs, C, rng) for X in bins]))
            for j, hv in enumerate(("first", "second")):
                h = half2 if hv == "second" else ~half2
                r = dict(base, area_col=area_col, area_value=area, condition_type="half", condition_value=hv,
                         n_units=len(units), n_trials=int(h.sum()), n_pos=int(y[h].sum()), n_neg=int((~y[h]).sum()), C=C,
                         skipped_reason=None if np.isfinite(real[sp["main"]][j]) else "half not scorable (< 3 per class)")
                for w in W:
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
    target = sys.argv[1]
    sys.path.insert(0, SCRIPTS)
    os.chdir(OUT.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st = pd.read_parquet(root / "metadata" / "sessions.parquet")
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    json.dump(T.causal_bin_edges(SPEC[target]["window"], bin_width=BIN_W, stride=STRIDE_S),
              open(OUT / f"116_bin_edges_{target}.json", "w"))
    out_path = OUT / f"116_halves_single_decoder_{target}.parquet"
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, (target, r.session_id, r.subject_id, r.reward_group)): r.session_id for r in sess.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception as e:  # noqa: BLE001
                rows.append(dict(session_id=futs[f], decode_target=target, skipped_reason=f"error: {e!r}"))
            print(f"[116 {target}] [{i}/{len(futs)}] {futs[f]} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[116 {target}] DONE", flush=True)


if __name__ == "__main__":
    main()
