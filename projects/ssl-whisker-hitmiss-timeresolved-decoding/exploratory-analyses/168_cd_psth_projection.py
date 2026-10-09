"""168 -- Time-resolved projection on the hit - miss axis (user 2026-10-08: "after the schematic, PSTH projected on the CD: whole brain or
an area with large separation, -100 to 200 ms, SEM over sessions").
Per learning-day session (135 session list, R+ / R-) and unit group (whole brain + each area_group; Part III tracked stable units, >= 5),
harmonised scaling as 135 SSL_135_HARM=1: rates in 10-ms windows (5-ms steps, centres -95 ... +195 ms after stimulus onset; dead zone
-10 to +5 ms excised, windows inside it are NaN), minus the epoch's mean baseline (-55 to -20 ms), divided by the unit's SD of the
5-35 ms active response (units with SD <= 1e-6 dropped). Active trials with a lick before 35 ms are excluded (as 135).
Axis (cross-validated): active whisker trials split in two random halves stratified by hit / miss (N_SPLIT splits); per split and
direction, u = unit(mean hit - mean miss) of the 5-35 ms window in one half, and the OTHER half's hit and miss time courses are
projected on u (held out, so the in-window separation is not inflated); passive whisker / auditory trials (pre, post; never used for
the axis) are projected on both halves' axes. Averaged over directions and splits; per neuron (/ sqrt(N)).
Also stored per session x group: held-out separation hit - miss in the 5-35 ms window (sep) and its d'.
Output: 168_cd_psth_projection.parquet (long: session_id, mouse_id, reward_group, area, n_units, cls, t, value) and
168_cd_psth_separation.parquet. Run (haas, repo root): python .../168_cd_psth_projection.py
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EA = Path(__file__).resolve().parent
SCRIPTS = str(EA.parents[2] / "scripts")
sys.path[:0] = [str(EA), SCRIPTS]
M135 = importlib.import_module("135_whisker_lick_alignment_epochs")
WIN, BASE, DZ = M135.WIN, M135.BASE, M135.DZ
TS = np.round(np.arange(-0.100, 0.1901, 0.005), 3)          # window starts; 10-ms windows
TC = TS + 0.005                                               # window centres
N_SPLIT, MIN_UNITS, MIN_CLASS = 10, 5, 5
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "30"))
TAB = EA                                                      # sync_results.sh moves tables to the NAS tables/ and symlinks them
STAGE = os.environ.get("SSL_STAGE", "learning")             # 2026-10-08: SSL_STAGE=expert for the across-days port; suffix _expert
SFX = "" if STAGE == "learning" else f"_{STAGE}"
RAW = bool(os.environ.get("SSL_168_RAW"))                      # 2026-10-09: raw rates, no SD scaling; suffix _raw
SFX = SFX + ("_raw" if RAW else "")
OUT_P, OUT_S = TAB / f"168_cd_psth_projection{SFX}.parquet", TAB / f"168_cd_psth_separation{SFX}.parquet"
SET_IDS = {}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def process(args):
    sid, subject, rg = args
    M135._init()
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [], []
    act = trs["active"].reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    keep_t = ~licked | (rt > WIN[1])
    act, licked = act[keep_t].reset_index(drop=True), licked[keep_t]
    trs = dict(trs, active=act)
    spikes = T.load_session_unit_spikes(root, sid)
    ids = SET_IDS.get(sid, np.array([], dtype=np.int64))
    sl = labels[labels.session_id == sid]
    groups = [("whole_brain", "All units")] + [("area_group", a) for a in sorted(sl.area_group.dropna().unique())]
    allu = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), ids)
    if len(allu) < MIN_UNITS:
        return [], []
    # rates for all tracked units once; groups are column subsets
    X, B, TR = {}, {}, {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy(); one = np.ones(len(t0), bool)
        r, b = T.sliding_bin_population_matrices(spikes, allu, t0, one, [WIN, BASE], dead_zone=DZ)
        B[e] = np.nanmean(b, 0, keepdims=True); X[e] = r - B[e]
        TR[e] = np.stack(T.sliding_bin_population_matrices(spikes, allu, t0, one, [(s, s + 0.010) for s in TS], dead_zone=DZ), 0) - B[e][None]
    isw = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in X}
    hit, miss = np.where(isw["active"] & licked)[0], np.where(isw["active"] & ~licked)[0]
    if min(len(hit), len(miss)) < 2 * MIN_CLASS:
        return [], []
    rows, seps = [], []
    for col, area in groups:
        m = np.isin(allu, np.intersect1d(T.area_units(sid, col, area, labels), ids))
        if m.sum() < MIN_UNITS:
            continue
        sd = X["active"][:, m].std(0); ok = sd > 1e-6
        if ok.sum() < MIN_UNITS:
            continue
        cols = np.where(m)[0][ok]; s_ = np.ones(int(ok.sum())) if RAW else sd[ok]; N = len(cols)
        E = X["active"][:, cols] / s_
        Tr = {e: TR[e][:, :, cols] / s_ for e in TR}                   # time x trials x units
        rng = np.random.default_rng(zlib.crc32(f"{sid}|{area}|168".encode()))
        acc = {k: [] for k in ("hit", "miss", "pre_W", "post_W", "pre_A", "post_A")}
        sep, dpr = [], []
        for _ in range(N_SPLIT):
            ph, pm = rng.permutation(hit), rng.permutation(miss)
            H_ = (ph[: len(ph) // 2], ph[len(ph) // 2:]); M_ = (pm[: len(pm) // 2], pm[len(pm) // 2:])
            for a, b in ((0, 1), (1, 0)):
                u = unit(E[H_[a]].mean(0) - E[M_[a]].mean(0))
                acc["hit"].append(np.nanmean(Tr["active"][:, H_[b]], 1) @ u)
                acc["miss"].append(np.nanmean(Tr["active"][:, M_[b]], 1) @ u)
                for ep, k in (("passive_pre", "pre"), ("passive_post", "post")):
                    acc[f"{k}_W"].append(np.nanmean(Tr[ep][:, isw[ep]], 1) @ u)
                    acc[f"{k}_A"].append(np.nanmean(Tr[ep][:, ~isw[ep]], 1) @ u)
                ph_, pm_ = E[H_[b]] @ u, E[M_[b]] @ u
                sep.append(ph_.mean() - pm_.mean())
                sp = np.sqrt(0.5 * (ph_.var(ddof=1) + pm_.var(ddof=1))); dpr.append(sep[-1] / sp if sp > 0 else np.nan)
        base = dict(session_id=sid, mouse_id=subject, reward_group=rg, area=area, n_units=N)
        for k, v in acc.items():
            mv = np.nanmean(np.stack(v), 0) / np.sqrt(N)
            rows += [dict(base, cls=k, t=float(t), value=float(x)) for t, x in zip(TC, mv)]
        seps.append(dict(base, n_hit=len(hit), n_miss=len(miss), sep=float(np.mean(sep)) / np.sqrt(N), dprime=float(np.nanmean(dpr))))
    return rows, seps


def main():
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == STAGE) & sess.reward_group.isin(["R+", "R-"])]
    SET_IDS.update(importlib.import_module("tracked_units").load("stable"))
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples()]
    print(f"[168] {len(args)} sessions", flush=True)
    rows, seps, t0 = [], [], time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=M135._init) as ex:
        futs = {ex.submit(process, a): a for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r, s = f.result(); rows += r; seps += s
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f][0], repr(e)[:200], flush=True)
            print(f"[168] [{i}/{len(args)}] {futs[f][0]} -- {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(OUT_P, index=False); pd.DataFrame(seps).to_parquet(OUT_S, index=False)
    print("ALL DONE", OUT_P)


if __name__ == "__main__":
    main()
