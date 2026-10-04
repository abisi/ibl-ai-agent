"""118 -- Pre/post learning-trial split decoding for EVERY learning-trial (LT) definition, whole brain, window
decoders (user request 2026-09-30: "run the split-part decoding (hit/miss and modality) for all sessions ... restricting
to the stat windows ... and on a single figure compare the results"; the user may still refine the LT definition, so
this is a quick overview of the differences between definitions).

Definitions: the 12 columns of ssl-learning-trial-identification/artifacts/013_learning_trials_all_methods.csv
(whisker-trial index, same convention as the stored LT: L0 stored, L1, L2, L3, L5, L7, L8, L6, L6 lenient, L5w lenient,
lenient cascade, lenient cascade + clean gate) + "half" (median split of the decoded trials; reference, no LT needed).
The LT index is converted to a time as in 095 (start time of whisker trial LT in the learning-curve trial list,
_active_trials_from_whisker_onset_for_curve); trials are 'pre' if they start before it.
Decodings (whole brain, one decoder per epoch, features = mean rate per unit in the window):
  hitmiss         stimulus-aligned whisker trials (prep_hitmiss_trials), y = lick; windows 5-50 ms and 5-100 ms
                  (dead zone -10..+5 ms from stimulus onset);
  modality_lick   licked whisker + auditory trials aligned to the corrected first lick (prep_lick_aligned_trials),
                  y = whisker; window -100..0 ms from the first lick (dead zone relative to stimulus onset).
Same scheme as 095 (hit/miss pre/post split with shift null):
  - terminal disengaged block dropped before the split (detect_terminal_disengagement; user: keep);
  - both epochs need >= MIN_TRIALS_PER_CLASS_EPOCH of each class;
  - primary = size-matched: each epoch subsampled to min(pre, post) per class, N_SUBSAMPLE subsamples; secondary =
    unmatched (all epoch trials);
  - one C per (session, decoding, window) from all trials (select_fixed_c_pooled), decode_bin_pooled;
  - chance = session-level linear shift null (non-wrapping, 10-50% of the trials): labels shifted against the neural
    trials, each pair keeps the NEURAL trial's epoch, the same matched/unmatched epoch decoding re-run; the first N_SHIFT
    shifts that keep >= MIN_TRIALS_PER_CLASS_EPOCH per class in both epochs are used;
  - reported: acc and null mean per epoch; delta_matched_nullcorr = (post - null_post) - (pre - null_pre).
Overview settings (quicker than 095; env to change): N_SUBSAMPLE 50 (095: 100), N_SHIFT 25 (095: 50).
Output: 118_lt_definitions_split_whole_brain.parquet (one row per session x decoding x window x definition).
Run (haas, repo root): python .../118_lt_definitions_split_decoding.py
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
LT_TABLE = OUT_DIR.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "013_learning_trials_all_methods.csv"
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate", "half"]
WINDOWS = {"hitmiss": {"5-50ms": (0.005, 0.050), "5-100ms": (0.005, 0.100)},
           "modality_lick": {"-100-0ms": (-0.100, 0.0)}}
DZ = (-0.010, 0.005)
MIN_TRIALS_PER_CLASS_EPOCH = 2
MAX_FOLDS = 5
N_SUBSAMPLE = int(os.environ.get("SSL_LT_N_SUBSAMPLE", "50"))
N_REPEATS_PER_SUBSAMPLE, N_REPEATS_FULL = 5, 20
N_SHIFT = int(os.environ.get("SSL_LT_N_SHIFT", "25"))
MAX_SHIFT_ATTEMPTS, N_SUBSAMPLE_SHIFT, N_REPEATS_SHIFT = 1000, 5, 2
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
RUN_TAG = os.environ.get("SSL_LT_OUT_TAG", "")      # "_A1" = rerun with disengagement rule A1 (2026-10-01)
OUT_PATH = OUT_DIR / f"118_lt_definitions_split_whole_brain{RUN_TAG}.parquet"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def epoch_scores(X, y, is_pre, C, rng, decode_bin_pooled, n_sub, n_rep_sub, n_rep_full):
    """As 095: size-matched and unmatched balanced accuracy per epoch, None if an epoch lacks a class."""
    idx = {"pre": np.where(is_pre)[0], "post": np.where(~is_pre)[0]}
    counts = {ep: (int(y[i].sum()), int((~y[i]).sum())) for ep, i in idx.items()}
    if min(min(c) for c in counts.values()) < MIN_TRIALS_PER_CLASS_EPOCH:
        return None
    tp, tn = min(counts["pre"][0], counts["post"][0]), min(counts["pre"][1], counts["post"][1])
    nf = min(MAX_FOLDS, tp, tn)
    out = dict(matched_n_pos=tp, matched_n_neg=tn)
    for ep, ie in idx.items():
        pos, neg = ie[y[ie]], ie[~y[ie]]
        accs = []
        for _ in range(n_sub):
            sel = np.r_[rng.choice(pos, tp, replace=False), rng.choice(neg, tn, replace=False)]
            accs.append(decode_bin_pooled(X[sel], y[sel], C, rng, n_repeats=n_rep_sub, n_folds=nf))
        ye = y[ie]
        out[f"acc_{ep}_matched"] = float(np.nanmean(accs))
        out[f"acc_{ep}_full"] = decode_bin_pooled(X[ie], ye, C, rng, n_repeats=n_rep_full,
                                                  n_folds=min(MAX_FOLDS, int(ye.sum()), int((~ye).sum())))
    return out


def valid_shifts(y, is_pre, rng):
    n = len(y)
    lo = max(1, int(MIN_SHIFT_FRAC * n))
    hi = max(lo, int(MAX_SHIFT_FRAC * n))
    out, tried = [], 0
    while len(out) < N_SHIFT and tried < MAX_SHIFT_ATTEMPTS:
        k, fwd = int(rng.integers(lo, hi + 1)), bool(rng.random() < 0.5)
        tried += 1
        ys, ps = (y[k:], is_pre[: n - k]) if fwd else (y[: n - k], is_pre[k:])
        if min(ys[ps].sum(), (~ys[ps]).sum(), ys[~ps].sum(), (~ys[~ps]).sum()) >= MIN_TRIALS_PER_CLASS_EPOCH:
            out.append((k, fwd))
    return out, tried


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
                n_subsample=N_SUBSAMPLE, n_shift_target=N_SHIFT, min_trials_per_class_epoch=MIN_TRIALS_PER_CLASS_EPOCH)
    tr = T.prep_hitmiss_trials(root, sid, st, tt) if decoding == "hitmiss" else T.prep_lick_aligned_trials(root, sid, st, tt)
    if tr is None or not len(tr):
        return [dict(base, definition=d, skipped_reason="no usable trials") for d in DEFS]
    dis = T.detect_terminal_disengagement(sid, tt)       # library default = rule A1 since 2026-10-01 (was >= 3 auditory)
    n0 = len(tr)
    if dis["disengaged"]:
        tr = tr[tr["start_time"] < dis["t_cut"]].reset_index(drop=True)
    base.update(disengaged=dis["disengaged"], n_dropped_disengaged=n0 - len(tr), run_tag=RUN_TAG or "A(min_aud 3)")
    y = (tr["lick_flag"] == 1).to_numpy() if decoding == "hitmiss" else (tr["trial_type"] == "whisker_trial").to_numpy()
    start = tr["start_time"].to_numpy()
    n = len(y)
    units = T.area_units(sid, "whole_brain", "All units", labels)
    if len(units) < T.MIN_UNITS_PER_AREA:
        return [dict(base, definition=d, skipped_reason="too few units") for d in DEFS]
    spikes = T.load_session_unit_spikes(root, sid)
    wins = list(WINDOWS[decoding].values())
    if decoding == "hitmiss":
        mats = T.sliding_bin_population_matrices(spikes, units, start, np.ones(n, bool), wins, dead_zone=DZ)
    else:
        mats = T.lick_aligned_bin_population_matrices(spikes, units, tr["first_lick_time"].to_numpy(), start,
                                                      (tr["trial_type"] == "whisker_trial").to_numpy(), wins, dead_zone=DZ)
    feats = dict(zip(WINDOWS[decoding], mats))
    ok_rows = {w: ~np.isnan(X).any(1) for w, X in feats.items()}
    rng0 = np.random.default_rng(zlib.crc32(f"{decoding}|{sid}".encode()))
    Cs = {w: T.select_fixed_c_pooled(X[ok_rows[w]], y[ok_rows[w]], rng0, n_folds=MAX_FOLDS) for w, X in feats.items()}
    lt_tab = pd.read_csv(LT_TABLE).set_index("session_id")
    cw = T._active_trials_from_whisker_onset_for_curve(sid, tt)
    cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
    rows = []
    for d in DEFS:
        b = dict(base, definition=d)
        if d == "half":
            is_pre = np.arange(n) < n // 2
            b.update(learning_trial=np.nan, t_split=float(start[n // 2]) if n else np.nan)
        else:
            lt = lt_tab[d].get(sid, np.nan) if d in lt_tab.columns else np.nan
            if pd.isna(lt):
                rows.append(dict(b, skipped_reason=f"no learning trial under {d}"))
                continue
            if int(lt) >= len(cw_t):
                rows.append(dict(b, learning_trial=lt, skipped_reason="LT beyond curve trials"))
                continue
            t_lt = cw_t[int(lt)]
            is_pre = start < t_lt
            b.update(learning_trial=float(lt), t_split=float(t_lt))
        b.update(n_trials=n, n_pre=int(is_pre.sum()), n_post=int((~is_pre).sum()), pre_pos=int(y[is_pre].sum()),
                 pre_neg=int((~y[is_pre]).sum()), post_pos=int(y[~is_pre].sum()), post_neg=int((~y[~is_pre]).sum()))
        if min(b["pre_pos"], b["pre_neg"], b["post_pos"], b["post_neg"]) < MIN_TRIALS_PER_CLASS_EPOCH:
            rows.append(dict(b, skipped_reason=f"class counts pre {b['pre_pos']}/{b['pre_neg']} post "
                                               f"{b['post_pos']}/{b['post_neg']} (< {MIN_TRIALS_PER_CLASS_EPOCH})"))
            continue
        rng = np.random.default_rng(zlib.crc32(f"{decoding}|{sid}|{d}".encode()))
        shifts, tried = valid_shifts(y, is_pre, rng)
        for w, X in feats.items():
            t0 = time.time()
            real = epoch_scores(X, y, is_pre, Cs[w], rng, T.decode_bin_pooled, N_SUBSAMPLE, N_REPEATS_PER_SUBSAMPLE,
                                N_REPEATS_FULL)
            if real is None:
                rows.append(dict(b, window=w, skipped_reason="epoch lacks a class after NaN rows"))
                continue
            null = {k: [] for k in ("pre_matched", "post_matched", "pre_full", "post_full")}
            for k, fwd in shifts:
                Xs, ys, ps = (X[: n - k], y[k:], is_pre[: n - k]) if fwd else (X[k:], y[: n - k], is_pre[k:])
                r = epoch_scores(Xs, ys, ps, Cs[w], rng, T.decode_bin_pooled, N_SUBSAMPLE_SHIFT, N_REPEATS_SHIFT,
                                 N_REPEATS_SHIFT)
                for key in null:
                    null[key].append(np.nan if r is None else r[f"acc_{key}"])
            row = dict(b, window=w, n_units=len(units), C=Cs[w], n_valid_shifts=len(shifts), n_shift_attempts=tried,
                       **real, skipped_reason=None)
            for key, vals in null.items():
                vals = np.asarray(vals, float)
                row[f"nullmean_{key}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            for ep in ("pre", "post"):
                for m in ("matched", "full"):
                    row[f"corr_{ep}_{m}"] = row[f"acc_{ep}_{m}"] - row[f"nullmean_{ep}_{m}"]
            row["delta_matched_nullcorr"] = row["corr_post_matched"] - row["corr_pre_matched"]
            row["delta_full_nullcorr"] = row["corr_post_full"] - row["corr_pre_full"]
            row["compute_s"] = time.time() - t0
            rows.append(row)
    return rows


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
    # lick-aligned tasks first (fewer trials, faster), then hit/miss
    args = [(r.session_id, r.subject_id, r.reward_group, dec) for dec in ("modality_lick", "hitmiss")
            for r in sess.itertuples() if (r.session_id, dec) not in done]
    print(f"[118] {len(args)} session x decoding tasks, {len(DEFS)} definitions, N_SUBSAMPLE {N_SUBSAMPLE}, "
          f"N_SHIFT {N_SHIFT}, {N_WORKERS} workers", flush=True)
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
            nok = int(new.skipped_reason.isna().sum()) if "skipped_reason" in new else 0
            print(f"[118] [{i}/{len(args)}] {a[0]} {a[3]}: {nok} rows ok -- {time.time() - t0:.0f}s", flush=True)
    print("[118] DONE", flush=True)


if __name__ == "__main__":
    main()
