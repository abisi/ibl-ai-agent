"""169 -- Is the passive pre -> post change on the hit - miss axis a move along the axis or a change of response size (gain)?
(user 2026-10-08: "split into along axis and orthogonal, show me results before updating the figure"). Also stores each session's
behavioural change for linking the two within-session axes (170).
Per learning-day session (135 list; Part III tracked stable units, whole brain), harmonised scaling as 135 SSL_135_HARM=1 / 166:
5-35 ms response minus the epoch's mean baseline (-55 to -20 ms), / unit SD over the active trials (units with SD <= 1e-6 dropped);
active trials with a lick before 35 ms excluded. u = unit(mean_z hit - mean_z miss) over all active whisker trials (passive trials
never build it); shifted axes u_k as 166 (labels shifted 10-50 % against the time-ordered whisker trials, 50 shifts, >= 10 per class).
For each passive stimulus (W, A) and epoch (pre, post), m = mean evoked vector; all per neuron (/ sqrt(N)):
  a      = m . u                               projection on the axis (along)
  rho    = |m|                                 response size; cross-validated: sqrt(m_A . m_B), m_A / m_B = means of random
                                               halves of the trials (N_HALF splits), so trial noise does not inflate it
  cos    = a / rho                             cosine of the angle between the response and the axis (gain-free)
Change d = m_post - m_pre:
  d_along = d . u        (= a_post - a_pre)
  d_orth  = |d - (d . u) u| cross-validated (halves of pre and of post)
  f_along = d_along^2 / (d_along^2 + d_orth^2)  fraction of the change's (cross-validated) energy along the axis
  g       = cos(d, m_pre) cross-validated: +1 / -1 = the change is a pure gain up / down of the pre pattern
Pure gain change (m_post = k m_pre): cos unchanged, d_along = (k - 1) a_pre. Remapping: cos changes.
Shifted axes: a, cos, d_along, f_along recomputed on every u_k; stored as the mean over shifts (_null).
Behaviour (active, all whisker / auditory trials of the analysed epoch): lick probability in the last third minus the first third
(dlick_W, dlick_A) and in the whole session (plick_W, plick_A).
Output: 169_along_orthogonal.parquet (one row per session x stimulus). Run (haas, repo root): python .../169_along_orthogonal_hitmiss.py
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
N_SHIFT, FRAC, MIN_CLASS, N_HALF, MIN_UNITS = 50, (0.10, 0.50), 10, 50, 20
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "30"))
OUT_P = EA / "169_along_orthogonal.parquet"
SET_IDS = {}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def thirds(x):
    k = len(x) // 3
    return (np.nanmean(x[-k:]) - np.nanmean(x[:k])) if k >= 3 else np.nan


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
        return []
    act0 = trs["active"].sort_values("start_time").reset_index(drop=True)
    beh = {}
    for s_, tt_ in (("W", "whisker_trial"), ("A", "auditory_trial")):
        lk = (act0[act0.trial_type == tt_].lick_flag == 1).to_numpy(float)
        beh[s_] = (thirds(lk), np.nanmean(lk) if len(lk) else np.nan)
    rt = (act0.lick_time - act0.response_window_start_time).to_numpy()
    licked = (act0.lick_flag == 1).to_numpy() & np.isfinite(rt)
    keep = ~licked | (rt > WIN[1]); act = act0[keep].reset_index(drop=True); licked = licked[keep]
    trs = dict(trs, active=act)
    units = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), SET_IDS.get(sid, np.array([], dtype=np.int64)))
    if len(units) < MIN_UNITS:
        return []
    spikes = T.load_session_unit_spikes(root, sid)
    X = {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy()
        r, b = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
        X[e] = r - np.nanmean(b, 0, keepdims=True)
    mu, sd = X["active"].mean(0), X["active"].std(0); ok = sd > 1e-6
    if ok.sum() < MIN_UNITS:
        return []
    Z = {e: (v[:, ok] - mu[ok]) / sd[ok] for e, v in X.items()}
    E = {e: v[:, ok] / sd[ok] for e, v in X.items()}
    N = int(ok.sum()); sN = np.sqrt(N)
    widx = np.where((act.trial_type == "whisker_trial").to_numpy())[0]
    y = licked[widx]
    if min(y.sum(), (~y).sum()) < MIN_CLASS:
        return []
    u = unit(Z["active"][widx[y]].mean(0) - Z["active"][widx[~y]].mean(0))
    n = len(widx); ks = np.unique(np.linspace(FRAC[0] * n, FRAC[1] * n, N_SHIFT // 2).astype(int))
    us = []
    for k in np.r_[ks, -ks]:
        i = np.arange(max(0, -k), min(n, n - k)); ys, xi = y[i], widx[i + k]
        if min(ys.sum(), (~ys).sum()) >= MIN_CLASS:
            us.append(unit(Z["active"][xi[ys]].mean(0) - Z["active"][xi[~ys]].mean(0)))
    U = np.stack([u] + us, 1)                                          # units x (1 + shifts); column 0 = real axis
    rng = np.random.default_rng(zlib.crc32(f"{sid}|169".encode()))
    rows = []
    for s_, tt_ in (("W", "whisker_trial"), ("A", "auditory_trial")):
        P = {e: E[e][(trs[e].trial_type == tt_).to_numpy()] for e in ("passive_pre", "passive_post")}
        if min(len(v) for v in P.values()) < 10:
            continue
        m = {e: v.mean(0) for e, v in P.items()}
        a = {e: m[e] @ U / sN for e in m}                              # (1 + shifts,)
        d = m["passive_post"] - m["passive_pre"]; dU = d @ U / sN
        rho2 = {e: [] for e in m}; dorth2, g_num, g_den1, g_den2 = [], [], [], []
        for _ in range(N_HALF):
            h = {}
            for e, v in P.items():
                pi = rng.permutation(len(v)); h[e] = (v[pi[: len(v) // 2]].mean(0), v[pi[len(v) // 2:]].mean(0))
                rho2[e].append(h[e][0] @ h[e][1] / N)
            dA, dB = h["passive_post"][0] - h["passive_pre"][0], h["passive_post"][1] - h["passive_pre"][1]
            dorth2.append((dA @ dB - (dA @ U) * (dB @ U)) / N)         # per axis (1 + shifts,)
            g_num.append(dA @ h["passive_pre"][1] / N); g_den1.append(dA @ dB / N); g_den2.append(h["passive_pre"][0] @ h["passive_pre"][1] / N)
        rho = {e: np.sqrt(max(np.mean(rho2[e]), 0)) for e in m}
        dorth = np.sqrt(np.clip(np.mean(dorth2, 0), 0, None))
        cos_ = {e: a[e] / rho[e] if rho[e] > 0 else a[e] * np.nan for e in m}
        f = dU ** 2 / (dU ** 2 + dorth ** 2)
        gden = np.sqrt(max(np.mean(g_den1), 0) * max(np.mean(g_den2), 0))
        g = np.mean(g_num) / gden if gden > 0 else np.nan
        r = dict(session_id=sid, mouse_id=subject, reward_group=rg, stim=s_, n_units=N, n_shifts=len(us),
                 n_pre=len(P["passive_pre"]), n_post=len(P["passive_post"]), g=g,
                 rho_pre=rho["passive_pre"], rho_post=rho["passive_post"],
                 dlick_W=beh["W"][0], plick_W=beh["W"][1], dlick_A=beh["A"][0], plick_A=beh["A"][1])
        for key, v in (("a_pre", a["passive_pre"]), ("a_post", a["passive_post"]), ("cos_pre", cos_["passive_pre"]),
                       ("cos_post", cos_["passive_post"]), ("d_along", dU), ("d_orth", dorth), ("f_along", f)):
            r[key], r[f"{key}_null"] = float(v[0]), float(np.nanmean(v[1:])) if len(us) else np.nan
        rows.append(r)
    print(sid, rg, N, "units", flush=True)
    return rows


def main():
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    SET_IDS.update(importlib.import_module("tracked_units").load("stable"))
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples()]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=M135._init) as ex:
        futs = {ex.submit(process, a_): a_ for a_ in args}
        for i, f_ in enumerate(as_completed(futs), 1):
            try:
                rows += f_.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f_][0], repr(e)[:200], flush=True)
    pd.DataFrame(rows).to_parquet(OUT_P, index=False)
    print("ALL DONE", OUT_P, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
