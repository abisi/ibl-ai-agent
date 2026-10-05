"""140 -- Hit/miss coding direction across session halves (rotation vs gain), its relation to passive whisker responses, and
noise correlations along it (user 2026-10-05: "3. yes do this, but generalize to the cross epoch case (i.e. project passive pre
and passive post data onto that) ... 4. noise correlations ... use tracked units with drift-robust metrics"; run in full).
Sessions: learning stage with passive trials before AND after the active block (132.session_trials: active = prep_modality_trials
from the first whisker trial, perf != 6, A1-trimmed; passive = labelled passive whisker / auditory trials).
Units: whole brain, the shared Part III tracked stable units (tracked_units.py / 137b: 137 'stable', drift-robust, MUA allowed,
firing >= 0.5 Hz in passive pre, passive post and both active halves of every split), identical to 133 / 134 / 135 / 146.
Responses: rate (Hz) 5-35 ms after stimulus onset minus the unit's mean -55..-20 ms baseline WITHIN its epoch; z-scored per
unit over all trials (pooled epochs). Active whisker trials with a lick before 35 ms are excluded (no motor contamination).
Splits of the active whisker trials: hitmedian (first H//2 hits before, as 139) and mid (median split of the whisker trials).
Per split, count matching: each half's hits and misses subsampled to the smaller half's counts; N_SPLIT random repetitions,
each splitting every half's trials into disjoint subsets A / B.
  rotation   CD_h = mean(hits) - mean(misses) in half h; cos between the two halves' CDs from disjoint subsets, normalised by
             the split-half reliabilities: cosnorm = mean cos(CD_1^x, CD_2^y) / sqrt(rel_1 rel_2), rel_h = cos(CD_h^A, CD_h^B),
             reliabilities floored at 0.05 (1 = same direction, 0 = unrelated).
  gain       d' of hits vs misses projected on an axis (axis from subset A, d' on subset B): own axis (dprime_own_h) and the other
             half's axis (dprime_cross_h: half h data on half h' axis). Rotation without gain: own > cross; gain: own_2 > own_1
             with cross_2 ~ own_2.
  passive    for each half's CD (unit vector from subset A): projection of passive pre / post whisker - auditory difference,
             as a fraction of the active hit - miss difference on the same axis (subset B): frac_<epoch>_h; also the reliability-
             normalised cos between the passive whisker axis (whisker - auditory) and CD_h, and the projection of the passive
             whisker-EVOKED pattern (baseline-subtracted, not centred).
  noise      within-class residuals of half h (matched subsample); noise_ratio_h = variance of the residuals along CD_h (axis from
             the other subset) / mean variance per unit (> 1: noise concentrated on the coding direction); linear Fisher
             information of hits vs misses in the top K principal components of all active residuals (shared subspace),
             bias-corrected for finite trials (Kanitscheider et al. 2015, two classes of T trials, N = K dimensions):
             FI = d' Sigma^-1 d (2T - N - 3) / (2T - 2) - 2N / T.
  shift null (hitmedian split; 2026-10-05, shift_null.py): CD_2 rebuilt from labels shifted against the time-ordered whisker
             trials (10-50 %, non-wrapping; N_SHIFT shifts among those keeping >= MIN_CLASS matched hits and misses after the
             hit-median split of the shifted labels; N_SPLIT_SHIFT splits); passive pre -> post change of noise-corrected cos
             (passive axis, W, A, W - A), raw cos and projections (W, A, W - A): shift_real / _null / _excess / _pct columns.
Output: 140_coding_direction_noise.parquet (session x split). Run (haas, repo root): python .../140_coding_direction_noise.py
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

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, str(OUT))
WIN, BASE, DZ = (0.005, 0.035), (-0.055, -0.020), (-0.010, 0.005)
MIN_RATE, MIN_UNITS, MIN_CLASS = 0.5, 20, 4          # MIN_CLASS per class per half after matching (split into A/B of >= 2)
N_SPLIT, K_PC = 50, 10
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "30"))
OUT_PATH = OUT / "140_coding_direction_noise.parquet"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else np.nan


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def dprime(ph, pm):
    s = np.sqrt(0.5 * (ph.var(ddof=1) + pm.var(ddof=1)))
    return float((ph.mean() - pm.mean()) / s) if s > 0 else np.nan


def fisher_bc(Zh, Zm):
    """bias-corrected linear Fisher information, two classes with T trials each (subsampled to equal T), N dims"""
    T = min(len(Zh), len(Zm)); N = Zh.shape[1]
    if 2 * T - N - 3 <= 0:
        return np.nan
    Zh, Zm = Zh[:T], Zm[:T]
    d = Zh.mean(0) - Zm.mean(0)
    S = 0.5 * (np.cov(Zh, rowvar=False) + np.cov(Zm, rowvar=False))
    try:
        fi = float(d @ np.linalg.solve(S, d))
    except np.linalg.LinAlgError:
        return np.nan
    return fi * (2 * T - N - 3) / (2 * T - 2) - 2 * N / T


N_SHIFT, N_SPLIT_SHIFT = 50, 10
PAS = ("passive_pre", "passive_post")


def shift_null(Za, Z, E, isw, lick, m, rng):
    """Linear-shift null (shift_null.py) of the passive pre -> post change of the alignment with the half-2 coding direction
    (hit-median split, whole brain). Per shift: hit-median split of the SHIFTED labels (in the truncated, time-ordered trials),
    count matching across halves (>= MIN_CLASS hits and misses), CD_2 from subset B (cos metrics) / subset A (projection) and its
    reliability cos(CD_2^A, CD_2^B), as the real one; passive patterns and their reliabilities are the real ones. Metrics:
    noise-corrected cos (axis = passive whisker - auditory axis, W, A evoked, W - A), raw cos (axis, W, A, W - A), projection of
    the evoked patterns on unit CD_2 / sqrt(n) (W, A, W - A)."""
    SN = importlib.import_module("shift_null")
    n_u = Za.shape[1]

    def halves(ys):
        hp = np.flatnonzero(ys)
        if len(hp) < 2:
            return None
        h1 = np.arange(len(ys)) < hp[len(hp) // 2]
        idx = {(h, c): np.where((h1 if h == 1 else ~h1) & (ys if c else ~ys))[0] for h in (1, 2) for c in (True, False)}
        tp = min(len(idx[(1, True)]), len(idx[(2, True)])); tn = min(len(idx[(1, False)]), len(idx[(2, False)]))
        return (idx, tp, tn) if min(tp, tn) >= MIN_CLASS else None

    valid = SN.valid_shifts(lick, lambda ys, pos: halves(ys) is not None)
    pats, rel = {}, {}
    for e in PAS:
        w, a = np.where(isw[e])[0], np.where(~isw[e])[0]
        pats[e] = (w, a)
        rel[e] = {k: max(m[f"{r}_{e}_2"], 0.05) for k, r in (("axis", "rpas"), ("W", "rW"), ("A", "rA"))}

    def metrics(c, p, rc):
        v = {}
        for e in PAS:
            for k in ("axis", "W", "A"):
                v[(e, f"{k}N")] = float(np.clip(c[(e, k)] / np.sqrt(rel[e][k] * max(rc, 0.05)), -1.5, 1.5))
                v[(e, f"{k}R")] = c[(e, k)]
            v[(e, "WAN")] = v[(e, "WN")] - v[(e, "AN")]; v[(e, "WAR")] = c[(e, "W")] - c[(e, "A")]
            v[(e, "WP")], v[(e, "AP")] = p[(e, "W")], p[(e, "A")]; v[(e, "WAP")] = p[(e, "W")] - p[(e, "A")]
        return {mm: v[("passive_post", mm)] - v[("passive_pre", mm)] for mm in {k for _, k in v}}

    c_r = {(e, k): m[f"{key}_{e}_2"] for e in PAS for k, key in (("axis", "cpas"), ("W", "cW"), ("A", "cA"))}
    p_r = {(e, k): m[f"p{k}_{e}_2"] for e in PAS for k in ("W", "A")}
    real = metrics(c_r, p_r, m["rel_2"])
    null, n_floor = {mm: [] for mm in real}, 0
    for j in rng.permutation(len(valid))[:N_SHIFT]:
        ys, pos, _ = SN.cut(lick, *valid[j])
        idx, tp, tn = halves(ys)
        c, p, rcs = {}, {}, []
        for _ in range(N_SPLIT_SHIFT):
            s_h = rng.choice(idx[(2, True)], tp, replace=False); s_m = rng.choice(idx[(2, False)], tn, replace=False)
            A_ = pos[np.r_[s_h[: tp // 2]]], pos[np.r_[s_m[: tn // 2]]]
            B_ = pos[np.r_[s_h[tp // 2:]]], pos[np.r_[s_m[tn // 2:]]]
            cdA = Za[A_[0]].mean(0) - Za[A_[1]].mean(0); cdB = Za[B_[0]].mean(0) - Za[B_[1]].mean(0)
            rcs.append(cos(cdA, cdB)); uA = unit(cdA)
            for e in PAS:
                w, a = pats[e]
                pw, pa = rng.permutation(w), rng.permutation(a)
                c.setdefault((e, "axis"), []).append(cos(Z[e][pw[: len(w) // 2]].mean(0) - Z[e][pa[: len(a) // 2]].mean(0), cdB))
                for k, idx_ in (("W", w), ("A", a)):
                    q = rng.permutation(idx_)
                    c.setdefault((e, k), []).append(cos(E[e][q[: len(q) // 2]].mean(0), cdB))
                    p.setdefault((e, k), []).append(float(E[e][idx_].mean(0) @ uA) / np.sqrt(n_u))
        rc = float(np.nanmean(rcs)); n_floor += rc < 0.05
        d = metrics({kk: np.nanmean(v) for kk, v in c.items()}, {kk: np.nanmean(v) for kk, v in p.items()}, rc)
        for mm, v in d.items():
            null[mm].append(v)
    out = SN.summarize(real, null)
    out.update(shift_n_valid=len(valid), shift_n=len(null["WN"]), shift_frac_rel_floor=n_floor / max(len(null["WN"]), 1))
    return out


def tracked_stable(sid, stable, spikes, segments):
    cands = stable.get(sid, np.array([], dtype=np.int64))
    out = []
    for cid in cands:
        sp = spikes.get(cid, spikes.get(int(cid), np.array([])))
        if not len(sp):
            continue
        if all((np.searchsorted(sp, b) - np.searchsorted(sp, a)) / max(b - a, 1e-6) >= MIN_RATE for a, b in segments):
            out.append(cid)
    return np.asarray(out)


def process(args):
    sid, subject, rg, stable = args
    _init()
    sys.path.insert(0, SCRIPTS); sys.path.insert(0, str(OUT))
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(base, skipped_reason="no passive pre+post or no active")]
    act = trs["active"].sort_values("start_time").reset_index(drop=True)
    actw = act[act.trial_type == "whisker_trial"].reset_index(drop=True)
    rt = (actw.lick_time - actw.response_window_start_time).to_numpy()
    lick = (actw.lick_flag == 1).to_numpy()
    H = int(lick.sum())
    t_w = actw.start_time.to_numpy()
    if H < 2 or (~lick).sum() < 2:
        return [dict(base, skipped_reason=f"hits {H}, misses {int((~lick).sum())}")]
    splits = {"hitmedian": t_w[np.flatnonzero(lick)[H // 2]], "mid": t_w[len(t_w) // 2]}
    ok = ~lick | ~np.isfinite(rt) | (rt > WIN[1])                       # no lick before the window ends
    actw, lick, t_w = actw[ok].reset_index(drop=True), lick[ok], t_w[ok]
    spikes = T.load_session_unit_spikes(root, sid)
    pre, post = trs["passive_pre"], trs["passive_post"]
    a0, a1 = act.start_time.min() - 1.0, act.start_time.max() + 1.0
    segs = [(pre.start_time.min() - 1, pre.start_time.max() + 1), (post.start_time.min() - 1, post.start_time.max() + 1)]
    for t in splits.values():
        segs += [(a0, t), (t, a1)]
    units = np.asarray(stable.get(sid, np.array([], dtype=np.int64)))    # shared tracked stable units (137b), taken as is
    if len(units) < MIN_UNITS:
        return [dict(base, n_units=len(units), skipped_reason="too few tracked stable units")]
    X = {}
    for e, d in (("passive_pre", pre), ("active", actw), ("passive_post", post)):
        t0 = d.start_time.to_numpy()
        r, b = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
        X[e] = r - np.nanmean(b, 0, keepdims=True)
    good = np.all([~np.isnan(v).any(0) for v in X.values()], axis=0)
    Pall = np.vstack([v[:, good] for v in X.values()])
    mu, sd = Pall.mean(0), Pall.std(0)
    keep = sd > 0
    if keep.sum() < MIN_UNITS:
        return [dict(base, n_units=int(keep.sum()), skipped_reason="too few units after cleaning")]
    Z = {e: (v[:, good][:, keep] - mu[keep]) / sd[keep] for e, v in X.items()}
    E = {e: v[:, good][:, keep] / sd[keep] for e, v in X.items()}
    isw = {e: (d.trial_type == "whisker_trial").to_numpy() for e, d in (("passive_pre", pre), ("passive_post", post))}
    # shared noise subspace: top K PCs of within-class residuals of all active whisker trials
    Za = Z["active"]
    R = np.vstack([Za[lick] - Za[lick].mean(0), Za[~lick] - Za[~lick].mean(0)])
    _, _, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    Vk = Vt[:K_PC].T
    rows = []
    for sname, t_s in splits.items():
        h1 = t_w < t_s
        idx = {(h, c): np.where((h1 if h == 1 else ~h1) & (lick if c else ~lick))[0] for h in (1, 2) for c in (True, False)}
        tp, tn = min(len(idx[(1, True)]), len(idx[(2, True)])), min(len(idx[(1, False)]), len(idx[(2, False)]))
        row = dict(base, split=sname, n_units=int(keep.sum()), n_whisker=len(t_w), split_frac=float(h1.mean()),
                   hits_1=len(idx[(1, True)]), misses_1=len(idx[(1, False)]), hits_2=len(idx[(2, True)]), misses_2=len(idx[(2, False)]),
                   matched_hits=tp, matched_misses=tn)
        if min(tp, tn) < MIN_CLASS:
            rows.append(dict(row, skipped_reason=f"matched hits/misses {tp}/{tn} < {MIN_CLASS}"))
            continue
        rng = np.random.default_rng(zlib.crc32(f"140|{sid}|{sname}".encode()))
        acc = {k: [] for k in ("rel_1", "rel_2", "between", "own_1", "own_2", "cross_1", "cross_2", "nr_1", "nr_2", "fi_1", "fi_2")}
        for e in ("passive_pre", "passive_post"):
            for h in (1, 2):
                for k in ("frac", "cpas", "rpas", "evokedW", "cW", "rW", "cA", "rA", "pW", "pA"):
                    acc[f"{k}_{e}_{h}"] = []
        for _ in range(N_SPLIT):
            sub = {}
            for (h, c), ii in idx.items():
                s = rng.choice(ii, tp if c else tn, replace=False)
                m = len(s) // 2
                sub[(h, c)] = (s[:m], s[m:])
            cd = {(h, p): Za[sub[(h, True)][p]].mean(0) - Za[sub[(h, False)][p]].mean(0) for h in (1, 2) for p in (0, 1)}
            acc["rel_1"].append(cos(cd[(1, 0)], cd[(1, 1)])); acc["rel_2"].append(cos(cd[(2, 0)], cd[(2, 1)]))
            acc["between"].append(np.mean([cos(cd[(1, p)], cd[(2, q)]) for p in (0, 1) for q in (0, 1)]))
            for h, o in ((1, 2), (2, 1)):
                hb, mb = Za[sub[(h, True)][1]], Za[sub[(h, False)][1]]
                u_own, u_oth = unit(cd[(h, 0)]), unit(cd[(o, 0)])
                acc[f"own_{h}"].append(dprime(hb @ u_own, mb @ u_own))
                acc[f"cross_{h}"].append(dprime(hb @ u_oth, mb @ u_oth))
                res = np.vstack([hb - hb.mean(0), mb - mb.mean(0)])
                acc[f"nr_{h}"].append(float(np.var(res @ u_own, ddof=1) / np.mean(np.var(res, axis=0, ddof=1))))
                Hh = np.r_[sub[(h, True)][0], sub[(h, True)][1]]; Mh = np.r_[sub[(h, False)][0], sub[(h, False)][1]]
                acc[f"fi_{h}"].append(fisher_bc(rng.permutation(Za[Hh] @ Vk), rng.permutation(Za[Mh] @ Vk)))
                sep = float((hb.mean(0) - mb.mean(0)) @ u_own)
                for e in ("passive_pre", "passive_post"):
                    w, a = np.where(isw[e])[0], np.where(~isw[e])[0]
                    if min(len(w), len(a)) < 4:
                        continue
                    pw = rng.permutation(w); pa = rng.permutation(a)
                    w1, w2, a1_, a2_ = pw[: len(w) // 2], pw[len(w) // 2:], pa[: len(a) // 2], pa[len(a) // 2:]
                    D = Z[e][w].mean(0) - Z[e][a].mean(0)
                    acc[f"frac_{e}_{h}"].append(float(D @ u_own) / sep if sep != 0 else np.nan)
                    acc[f"cpas_{e}_{h}"].append(cos(Z[e][w1].mean(0) - Z[e][a1_].mean(0), cd[(h, 1)]))
                    acc[f"rpas_{e}_{h}"].append(cos(Z[e][w1].mean(0) - Z[e][a1_].mean(0), Z[e][w2].mean(0) - Z[e][a2_].mean(0)))
                    acc[f"evokedW_{e}_{h}"].append(float(E[e][w].mean(0) @ u_own) / sep if sep != 0 else np.nan)
                    # evoked patterns (baseline-subtracted, not centred) of each stimulus vs the coding direction (subset B,
                    # disjoint from the trials of the passive pattern halves), with split-half reliabilities: auditory = control
                    for key, idx_ in (("W", w), ("A", a)):
                        q = rng.permutation(idx_); q1, q2 = q[: len(q) // 2], q[len(q) // 2:]
                        acc[f"c{key}_{e}_{h}"].append(cos(E[e][q1].mean(0), cd[(h, 1)]))
                        acc[f"r{key}_{e}_{h}"].append(cos(E[e][q1].mean(0), E[e][q2].mean(0)))
                        acc[f"p{key}_{e}_{h}"].append(float(E[e][idx_].mean(0) @ u_own) / np.sqrt(len(u_own)))
        m = {k: float(np.nanmean(v)) if len(v) else np.nan for k, v in acc.items()}
        r1, r2 = max(m["rel_1"], 0.05), max(m["rel_2"], 0.05)
        row.update(rel_cd_1=m["rel_1"], rel_cd_2=m["rel_2"], cos_between=m["between"],
                   cosnorm_between=float(np.clip(m["between"] / np.sqrt(r1 * r2), -1.5, 1.5)),
                   dprime_own_1=m["own_1"], dprime_own_2=m["own_2"], dprime_cross_1=m["cross_1"], dprime_cross_2=m["cross_2"],
                   noise_ratio_1=m["nr_1"], noise_ratio_2=m["nr_2"], fisher_1=m["fi_1"], fisher_2=m["fi_2"])
        for e in ("passive_pre", "passive_post"):
            for h in (1, 2):
                rr = max(m[f"rpas_{e}_{h}"], 0.05) if np.isfinite(m[f"rpas_{e}_{h}"]) else np.nan
                rc = r1 if h == 1 else r2
                row[f"frac_{e}_{h}"] = m[f"frac_{e}_{h}"]
                row[f"evokedW_frac_{e}_{h}"] = m[f"evokedW_{e}_{h}"]
                row[f"cosnorm_{e}_{h}"] = float(np.clip(m[f"cpas_{e}_{h}"] / np.sqrt(rr * rc), -1.5, 1.5)) if np.isfinite(rr) else np.nan
                for key in ("W", "A"):
                    rk = max(m[f"r{key}_{e}_{h}"], 0.05) if np.isfinite(m[f"r{key}_{e}_{h}"]) else np.nan
                    row[f"cosnorm_evoked{key}_{e}_{h}"] = (float(np.clip(m[f"c{key}_{e}_{h}"] / np.sqrt(rk * rc), -1.5, 1.5))
                                                           if np.isfinite(rk) else np.nan)
                    row[f"cos_evoked{key}_{e}_{h}"] = m[f"c{key}_{e}_{h}"]
                    row[f"proj_evoked{key}_{e}_{h}"] = m[f"p{key}_{e}_{h}"]
        if sname == "hitmedian":
            row.update(shift_null(Za, Z, E, isw, lick, m, rng))
        rows.append(dict(row, skipped_reason=None))
    return rows


def main():
    os.chdir(OUT.parents[2])
    sys.path.insert(0, SCRIPTS)
    from axel_bisi_paths import axel_bisi_root
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    # 2026-10-05: the shared Part III tracked stable units (tracked_units.py / 137b; "same stable units throughout")
    stable_by_sid = importlib.import_module("tracked_units").load("stable")
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    args = [(r.session_id, r.subject_id, r.reward_group, {r.session_id: stable_by_sid.get(r.session_id, np.array([]))})
            for r in sess.itertuples() if r.session_id not in done]
    print(f"[140] {len(args)} sessions, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, a): a for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                rows = [dict(session_id=a[0], mouse_id=a[1], reward_group=a[2], skipped_reason=f"error: {e!r}")]
            new = pd.DataFrame(rows)
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            tmp = OUT_PATH.with_suffix(".partial.parquet"); out.to_parquet(tmp, index=False); os.replace(tmp, OUT_PATH)
            print(f"[140] [{i}/{len(args)}] {a[0]}: {new.skipped_reason.isna().sum()} ok -- {time.time() - t0:.0f}s", flush=True)
    print("[140] DONE", flush=True)


if __name__ == "__main__":
    main()
