"""007 -- Does learning orthogonalise the sensory coding direction and the hit / miss coding direction? (user 2026-11-12: "test
that learning induces orthogonalization of the sensory response with that of the response predicting hit/miss using the
coding directions"). Same sessions, trials and tracked stable units as 001; response = window - LBASE divided by the unit's SD
over all trials of the session, NOT mean-centred (001 centres, so its mean responses are relative to the session's grand mean
over all trial types and epochs, not to the baseline).

Coding directions (mean differences, per session x area x window):
  s_X   sensory CD of trial set X: mean whisker response (active: class-balanced, 0.5 (mean hits + mean misses), so a change
        of hit rate does not leak the hit / miss difference into it; passive: mean of passive whisker trials)
  h_X   hit / miss CD: mean hits - mean misses (active whisker trials of X)
  X in  pre (passive pre), post (passive post), all (active), early / late (chronological halves of the active whisker
        trials), preLT / postLT (before / from the session's learning trial, ssl-learning-trial-identification 006
        lt_recommended_broad, whisker-trial ordinal before the early-lick drop)
Cross-validated cosine: N_SPLIT random stratified half splits (hits, misses, passive trials each halved); for vectors u, v:
  cos = mean[0.5 (u_a.v_b + u_b.v_a)] / sqrt(mean[u_a.u_b] mean[v_a.v_b]) (independent halves only: unbiased by noise;
  NaN when a split-half reliability is <= 0). Signed: > 0 = hits respond more along the sensory direction.
  Pairs: (s_pre, h_X), (s_post, h_X), (s_X, h_X), (s_pre, s_post), (s_early, s_late), (h_early, h_late), (h_preLT, h_postLT);
  squared norms per unit |h_X|^2 / N, |s_X|^2 / N (unbiased, cross-half dot products); hpar_{pre,s}_X = projection of h_X
  on the unit sensory CD of the other half, per sqrt(N) (the hit / miss separation along the sensory direction).
Linear-shift null: hit / miss labels shifted against the trials (10-50 %, non-wrapping, both directions, within the trial set),
  N_NULL shifts x N_SPLIT_NULL splits: median null cos for (s_pre, h_X) and (s_X, h_X) -> *_null.
Gain vs rotation of the passive change (K vs L of 003): D = mean post - mean pre (passive whisker), u_s = unit(s_pre),
  u_h = unit(h_all); pre trials split in halves (u_s from one, D from the other and all post, both ways, N_SPLIT): projection of
  D on u_h in units of |h_all| (fraction of the hit / miss distance) = gain part (D.u_s)(u_s.u_h) + rotation part
  (D - (D.u_s) u_s).u_h; relative gain change (D.u_s) / |s_pre|.
Windows: early (5-35 ms, 001 SENS) and mid (35-75 ms); active trials with a lick before the window end are dropped.
Per trial: g_u = response projected on unit(mean passive-pre whisker response) of the uncentred matrices (the gain measure of
001 g without the centring), all epochs -> 007_trials{...}.parquet (session, area, window, epoch, trial type, lick, start time).
Output: tables/007_orthogonalization{_fine}{_expert}.parquet (session x area x window), 007_trials{_fine}{_expert}.parquet
Run (haas, repo root): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities nice \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/007_orthogonalization.py
  [SSL_STAGE=expert] [SSL_001_AREA=area_acronym_custom]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m01 = importlib.import_module("001_state_gating")
REPO, TAB = m01.REPO, m01.TAB
LT = HERE.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "006_learning_trials_final.csv"
WINDOWS = {"early": (0.005, 0.035), "mid": (0.035, 0.075)}
N_SPLIT, N_NULL, N_SPLIT_NULL, MIN_HALF = 100, 50, 10, 4
SFX = ("_fine" if m01.AREA_COL != "area_group" else "") + ("" if m01.STAGE == "learning" else f"_{m01.STAGE}")
OUT = TAB / f"007_orthogonalization{SFX}.parquet"
OUT_TRIALS = TAB / f"007_trials{SFX}.parquet"


def halves(idx, rng):
    p = rng.permutation(idx)
    return p[: len(p) // 2], p[len(p) // 2: 2 * (len(p) // 2)]


def cds(R, y, rng, n_split, P=None):
    """Per split: (s_a, s_b, h_a, h_b) of the active trials R with labels y (and (p_a, p_b) of passive matrices P)."""
    hi, mi = np.where(y)[0], np.where(~y)[0]
    out = []
    for _ in range(n_split):
        (h1, h2), (m1, m2) = halves(hi, rng), halves(mi, rng)
        H = [R[h1].mean(0), R[h2].mean(0)]
        Mi = [R[m1].mean(0), R[m2].mean(0)]
        d = dict(s=[0.5 * (H[0] + Mi[0]), 0.5 * (H[1] + Mi[1])], h=[H[0] - Mi[0], H[1] - Mi[1]])
        for k, M in (P or {}).items():
            a, b = halves(np.arange(len(M)), rng)
            d[k] = [M[a].mean(0), M[b].mean(0)]
        out.append(d)
    return out


def xcos(su, sv, ku, kv):
    """Cross-validated cosine between vector ku of split list su and kv of split list sv (paired splits)."""
    num = np.mean([0.5 * (a[ku][0] @ b[kv][1] + a[ku][1] @ b[kv][0]) for a, b in zip(su, sv)])
    uu = np.mean([a[ku][0] @ a[ku][1] for a in su])
    vv = np.mean([b[kv][0] @ b[kv][1] for b in sv])
    return float(num / np.sqrt(uu * vv)) if uu > 0 and vv > 0 else np.nan


def xproj(su, ks, kh, n):
    """Cross-validated projection of kh on the unit direction of ks (other half), per sqrt(unit)."""
    v = [0.5 * (a[kh][1] @ a[ks][0] / np.linalg.norm(a[ks][0]) + a[kh][0] @ a[ks][1] / np.linalg.norm(a[ks][1])) for a in su]
    return float(np.mean(v) / np.sqrt(n))


def sq(su, k, n):
    return float(np.mean([a[k][0] @ a[k][1] for a in su]) / n)


def shift_null(R, y, Ppre, rng):
    n = len(y)
    pairs = [(k, d) for k, d in m01.shift_pairs(n)
             if min((y[k:] if d else y[: n - k]).sum(), (~(y[k:] if d else y[: n - k])).sum()) >= MIN_HALF]
    a, b = [], []
    for j in rng.permutation(len(pairs))[:N_NULL]:
        k, d = pairs[j]
        ys, Rs = (y[k:], R[: n - k]) if d else (y[: n - k], R[k:])
        sp = cds(Rs, ys, rng, N_SPLIT_NULL, {"pre": Ppre})
        a.append(xcos(sp, sp, "pre", "h"))
        b.append(xcos(sp, sp, "s", "h"))
    f = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    return f(np.asarray(a, float)), f(np.asarray(b, float))


def process(args):
    sid, subject, rg, sets, lt = args
    m01._init()
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(base, skipped_reason="no passive pre+post or no active")], []
    act0 = trs["active"].sort_values("start_time").reset_index(drop=True)
    act0["w_ordinal"] = np.nan
    w0 = (act0.trial_type == "whisker_trial").to_numpy()
    act0.loc[w0, "w_ordinal"] = np.arange(1, w0.sum() + 1)
    pre = trs["passive_pre"].sort_values("start_time").reset_index(drop=True)
    post = trs["passive_post"].sort_values("start_time").reset_index(drop=True)
    wpre, wpost = (pre.trial_type == "whisker_trial").to_numpy(), (post.trial_type == "whisker_trial").to_numpy()
    if wpre.sum() < 2 * MIN_HALF or wpost.sum() < 2 * MIN_HALF:
        return [dict(base, skipped_reason="too few passive whisker trials")], []
    spikes = T.load_session_unit_spikes(root, sid)
    rows, trows = [], []
    for wname, win in WINDOWS.items():
        rt = (act0.lick_time - act0.response_window_start_time).to_numpy()
        act = act0[~((act0.lick_flag == 1).to_numpy() & np.isfinite(rt) & (rt <= win[1]))].reset_index(drop=True)
        isw = (act.trial_type == "whisker_trial").to_numpy()
        y = (act.lick_flag == 1).to_numpy()[isw]
        if min(y.sum(), (~y).sum()) < 2 * MIN_HALF:
            rows.append(dict(base, window=wname, skipped_reason="too few hits / misses")); continue
        ordn = act.w_ordinal.to_numpy()[isw]
        for area, units in sets.items():
            units = np.asarray(units)
            row = dict(base, area=area, window=wname, n_hits=int(y.sum()), n_misses=int((~y).sum()), skipped_reason=None)
            if len(units) < m01.MIN_UNITS:
                rows.append(dict(row, skipped_reason="too few units")); continue
            Rr = {}
            for e, d in (("passive_pre", pre), ("active", act), ("passive_post", post)):
                t0 = d.start_time.to_numpy()
                m = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [m01.LBASE, win],
                                                      dead_zone=m01.DZ)
                Rr[e] = m[1] - m[0]
            ok_u = np.all([~np.isnan(v).any(0) for v in Rr.values()], axis=0)
            Rp = np.vstack([v[:, ok_u] for v in Rr.values()])
            keep = Rp.std(0) > 0
            if keep.sum() < m01.MIN_UNITS:
                rows.append(dict(row, skipped_reason="too few units after cleaning")); continue
            z = {e: v[:, ok_u][:, keep] / Rp[:, keep].std(0) for e, v in Rr.items()}
            nU = int(keep.sum())
            row["n_units"] = nU
            rng = np.random.default_rng(zlib.crc32(f"007|{sid}|{area}|{wname}".encode()))
            u_pre = m01.unit_vec(z["passive_pre"][wpre].mean(0))
            for e, d in (("passive_pre", pre), ("active", act), ("passive_post", post)):
                trows.append(pd.DataFrame(dict(session_id=sid, area=area, window=wname, epoch=e, trial_type=d.trial_type.to_numpy(),
                                               lick=d.lick_flag.to_numpy(), start_time=d.start_time.to_numpy(), g_u=z[e] @ u_pre)))
            Rw, Ppre, Ppost = z["active"][isw], z["passive_pre"][wpre], z["passive_post"][wpost]
            n = len(y)
            subsets = {"all": np.ones(n, bool), "early": np.arange(n) < n // 2, "late": np.arange(n) >= n // 2}
            if np.isfinite(lt):
                subsets.update(preLT=ordn < lt, postLT=ordn >= lt)
                row["lt"] = lt
            SP = {}
            for k, m in subsets.items():
                if min(y[m].sum(), (~y[m]).sum()) >= MIN_HALF:
                    SP[k] = cds(Rw[m], y[m], rng, N_SPLIT, {"pre": Ppre, "post": Ppost})
                    row[f"n_hits_{k}"], row[f"n_misses_{k}"] = int(y[m].sum()), int((~y[m]).sum())
            for k, sp in SP.items():
                row[f"cos_pre_h_{k}"] = xcos(sp, sp, "pre", "h")
                row[f"cos_post_h_{k}"] = xcos(sp, sp, "post", "h")
                row[f"cos_s_h_{k}"] = xcos(sp, sp, "s", "h")
                row[f"cos_pre_s_{k}"] = xcos(sp, sp, "pre", "s")
                row[f"h2_{k}"], row[f"s2_{k}"] = sq(sp, "h", nU), sq(sp, "s", nU)
                row[f"hpar_pre_{k}"], row[f"hpar_s_{k}"] = xproj(sp, "pre", "h", nU), xproj(sp, "s", "h", nU)
                nl = shift_null(Rw[subsets[k]], y[subsets[k]], Ppre, rng)
                row[f"cos_pre_h_{k}_null"], row[f"cos_s_h_{k}_null"] = nl
            if "all" in SP:
                sp = SP["all"]
                row["cos_pre_post"] = xcos(sp, sp, "pre", "post")
                row["pre2"], row["post2"] = sq(sp, "pre", nU), sq(sp, "post", nU)
            for a, b in (("early", "late"), ("preLT", "postLT")):
                if a in SP and b in SP:
                    row[f"cos_h_{a}_{b}"] = xcos(SP[a], SP[b], "h", "h")
                    row[f"cos_s_{a}_{b}"] = xcos(SP[a], SP[b], "s", "s")
            # gain vs rotation of the passive change, on the whole-session hit / miss CD
            hfull = Rw[y].mean(0) - Rw[~y].mean(0)
            hn = np.linalg.norm(hfull)
            if hn > 0:
                uh, parts = hfull / hn, []
                post_m = Ppost.mean(0)
                for _ in range(N_SPLIT // 4):
                    a, b = halves(np.arange(len(Ppre)), rng)
                    for i1, i2 in ((a, b), (b, a)):
                        s_pre = Ppre[i1].mean(0)
                        us = m01.unit_vec(s_pre)
                        D = post_m - Ppre[i2].mean(0)
                        g = D @ us
                        parts.append((D @ uh / hn, g * (us @ uh) / hn, (D - g * us) @ uh / hn,
                                      g / np.linalg.norm(Ppre[i2].mean(0)), us @ uh))
                P = np.mean(parts, 0)
                row.update(dproj_h=P[0], dproj_h_gain=P[1], dproj_h_rot=P[2], gain_rel=P[3], cos_us_uh_naive=P[4])
            rows.append(row)
    return rows, trows


def main():
    m01._init()
    os.chdir(REPO)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    S = importlib.import_module("tracked_units").load_table()
    S = S[S.stable]
    if m01.AREA_COL != "area_group":
        L = pd.read_parquet(T.AREA_LABELS_PATH, columns=["session_id", "cluster_id", m01.AREA_COL])
        S = S.drop(columns=[m01.AREA_COL], errors="ignore").merge(L, on=["session_id", "cluster_id"], how="left")
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == m01.STAGE) & sess.reward_group.isin(["R+", "R-"])]
    lt = pd.read_csv(LT).set_index("session_id").lt_recommended_broad
    args = []
    for r in sess.itertuples():
        s = S[S.session_id == r.session_id]
        sets = {"whole_brain": s.cluster_id.to_numpy()}
        for a, gdf in s.groupby(m01.AREA_COL):
            if len(gdf) >= m01.MIN_UNITS:
                sets[a] = gdf.cluster_id.to_numpy()
        args.append((r.session_id, r.subject_id, r.reward_group, sets, float(lt.get(r.session_id, np.nan))))
    print(f"[007] {len(args)} sessions -> {OUT.name}", flush=True)
    t0, rows, trows = time.time(), [], []
    with ProcessPoolExecutor(min(m01.N_WORKERS, max(1, len(args))), initializer=m01._init) as ex:
        futs = {ex.submit(process, a): a[0] for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r, tr = f.result()
                rows += r
                trows += tr
            except Exception as err:  # noqa: BLE001
                rows.append(dict(session_id=futs[f], skipped_reason=f"error: {err!r}"))
            print(f"[007] [{i}/{len(args)}] {futs[f]} -- {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    tmp = OUT.with_suffix(".partial.parquet"); out.to_parquet(tmp, index=False); os.replace(tmp, OUT)
    if trows:
        tl = pd.concat(trows, ignore_index=True)
        tmp = OUT_TRIALS.with_suffix(".partial.parquet"); tl.to_parquet(tmp, index=False); os.replace(tmp, OUT_TRIALS)
    print(f"[007] done in {(time.time() - t0) / 60:.1f} min: {len(out)} rows, {out.skipped_reason.isna().sum()} ok", flush=True)


if __name__ == "__main__":
    main()
