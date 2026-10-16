"""008 -- Per-event table for the day-0-specificity tests (user 2026-10-16: "try without baseline correction ... run all 6 tests").
For every session of within_day 002 (both stages; same events, stable units, mean raw pre-lick rate >= 0.1 Hz, >= 4 WH, 8 AH, 8 SL), the
reward-lick CD scores of 002 (trial_scores: 5-fold CV over AH + SL, units z-scored on the training folds, unit-norm AH - SL mean difference,
held-out AH / SL and fold-averaged WH projections; one session normalisation, held-out SL = 0, AH = 1; d' >= 0.3) are computed twice:
  corr   pre-lick rate [-100, 0) ms minus the event's own baseline (trial classes: [-1.0, -0.015] s before trial start; SL: before the lick),
         as in 002 (sanity check: equal to 002's `cd`)
  raw    the same rate without baseline subtraction
Per event: class, trial start, tau, RT, whisker-trial index (whisker trials, hits and misses, started before the event; 105 stimulus table),
the session's learning-curve values at that whisker trial (ssl-learning-trial-identification 024, sigma 1: whisker P(lick) and whisker - FA;
day-0 sessions with curves) and the learning trial (017 lt_final, whisker-trial index -> time).
Per session (test 4, CD stability): events split at the median tau into early / late; per half and variant the unit-norm AH - SL axis
(z-scored on that half), held-out separation d' (3-fold), cross-generalisation (late events on the early axis: d'_cross / d'_late), raw
cosine between early and late axes and the split-half-reliability-normalised cosine (disjoint random halves, 50 splits, reliability floor
0.05, as 135).
Output: tables/008_events.parquet, 008_sessions.csv
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import pickle
import zlib

import numpy as np
import pandas as pd

import common as C

m002 = importlib.import_module("002_trial_slopes")
st26 = importlib.import_module("026_roc_rates_all_sessions")
m105 = importlib.import_module("105_trial_rates")
LTP = C.PROJECTS / "ssl-learning-trial-identification" / "artifacts"
N_SPLIT, FLOOR = 50, 0.05


def dprime(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def axis(X, y):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    u = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    return u / max(np.linalg.norm(u), 1e-12), mu, sd


def cv_dprime(X, y, rng, k=3):
    s = np.full(len(y), np.nan)
    idx = rng.permutation(len(y)); folds = np.array_split(idx, k)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if min((y[tr] == 1).sum(), (y[tr] == 0).sum()) < 2:
            return np.nan
        u, mu, sd = axis(X[tr], y[tr]); s[f] = ((X[f] - mu) / sd) @ u
    return dprime(s[y == 1], s[y == 0])


def stability(X, lab, tau, rng):
    m = np.isin(lab, ["AH", "FA"]); y = (lab == "AH").astype(int)
    med = np.median(tau[m])
    E, L = m & (tau <= med), m & (tau > med)
    if min((y[E] == 1).sum(), (y[E] == 0).sum(), (y[L] == 1).sum(), (y[L] == 0).sum()) < 4:
        return {}
    ue, mue, sde = axis(X[E], y[E]); ul, _, _ = axis(X[L], y[L])
    sl_late = ((X[L] - mue) / sde) @ ue
    d_e, d_l = cv_dprime(X[E], y[E], rng), cv_dprime(X[L], y[L], rng)
    d_x = dprime(sl_late[y[L] == 1], sl_late[y[L] == 0])
    cn, re_, rl_ = [], [], []
    iE, iL = np.where(E)[0], np.where(L)[0]
    for _ in range(N_SPLIT):
        a = rng.permutation(iE); b = rng.permutation(iL)
        e1, e2, l1, l2 = a[: len(a) // 2], a[len(a) // 2:], b[: len(b) // 2], b[len(b) // 2:]
        if min(min((y[s_] == 1).sum(), (y[s_] == 0).sum()) for s_ in (e1, e2, l1, l2)) < 1:
            continue
        ae1, ae2, al1, al2 = (axis(X[s_], y[s_])[0] for s_ in (e1, e2, l1, l2))
        re_.append(ae1 @ ae2); rl_.append(al1 @ al2); cn.append(ae1 @ al2)
    rel_e, rel_l = max(np.mean(re_), FLOOR) if re_ else np.nan, max(np.mean(rl_), FLOOR) if rl_ else np.nan
    return dict(cos_raw=float(ue @ ul), cos_norm=float(np.clip(np.mean(cn) / np.sqrt(rel_e * rel_l), -1.5, 1.5)) if cn else np.nan,
                rel_early=rel_e, rel_late=rel_l, dp_early=d_e, dp_late=d_l, dp_cross=d_x, gen=d_x / d_l if d_l and np.isfinite(d_l) else np.nan)


def session(args):
    sid, f, meta, W_s, seed, curves, lt = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    Xc, Xr, lab, t, rt = (z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o].astype(str),
                         z["trial_start"].astype(float)[o], z["rt"].astype(float)[o])
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return None, None
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (Xr.mean(1) >= C.m51.MIN_FR)
    if ok.sum() < m002.MIN_UNITS:
        return None, None
    tau = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    ev = pd.DataFrame(dict(meta, cls=np.where(lab == "FA", "SL", lab), t=t, tau=tau, rt=rt))
    srow = dict(meta, n_units=int(ok.sum()))
    rng = np.random.default_rng(seed)
    for v, X in (("corr", Xc[ok].T), ("raw", Xr[ok].T)):
        S = m002.trial_scores(X, lab, seed)
        ev[f"cd_{v}"] = S["md"]; srow[f"dprime_{v}"] = S["dprime"]
        srow.update({f"{k}_{v}": x for k, x in stability(X, lab, tau, rng).items()})
    # whisker-trial index, learning curves and learning trial (105 stimulus table: 'W' = whisker trials, hits and misses)
    f105 = m105.OUT / f"{meta['session_id']}.npz"
    if f105.exists():
        q = np.load(f105, allow_pickle=True)
        wt = np.sort(q["stim_t"][q["stim_lab"].astype(str) == "W"].astype(float))
        ev["wtrial"] = np.searchsorted(wt, ev.t.to_numpy() - 1e-6)            # whisker trials started before the event
        srow["n_wtrials"] = len(wt)
        if curves is not None and all(len(c_) == len(wt) for c_ in curves.values()):
            k = np.clip(ev.wtrial.to_numpy() - 1, 0, len(wt) - 1)                # last whisker trial before (or at) the event
            ev["curve_whisker"], ev["curve_wfa"] = np.asarray(curves["whisker"])[k], np.asarray(curves["w-fa"])[k]
        if lt is not None and np.isfinite(lt) and int(lt) < len(wt):
            srow["lt_wtrial"], srow["lt_time"] = int(lt), float(wt[int(lt)])
    return ev, srow


def main(a):
    W = pd.read_parquet(C.m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    curves = {p["session_id"]: p["curves"] for p in pickle.load(open(LTP / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    LT = pd.read_csv(LTP / "017_learning_trials_final_sigma1.csv").set_index("session_id")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{C.m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            meta = dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0])
            lt = LT.lt_final.get(r.session_id, np.nan) if meta["stage"] == "learning" else np.nan
            jobs.append((r.session_id, f, meta, Ws[["electrode_group", "cluster_id", "quality_label"]], zlib.crc32(r.session_id.encode()),
                         curves.get(r.session_id) if meta["stage"] == "learning" else None, lt))
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = pool.map(session, jobs)
    E = pd.concat([e for e, _ in res if e is not None], ignore_index=True)
    S = pd.DataFrame([s for _, s in res if s is not None])
    if "category" in LT:
        S["lt_category"] = S.session_id.map(LT.category)
    E.to_parquet(C.TAB / "008_events.parquet", index=False); S.to_csv(C.TAB / "008_sessions.csv", index=False)
    # sanity check against 002
    T = pd.read_parquet(C.m51.WITHIN / "slopes" / "trial_scores.parquet")
    T["cls"] = T.cls.replace({"FA": "SL"}); T["t_r"] = T.t.round(3); E["t_r"] = E.t.round(3)
    M = E.merge(T[["session_id", "cls", "t_r", "cd"]], on=["session_id", "cls", "t_r"])
    print("events", len(E), "sessions", S.session_id.nunique(), "| match 002:", len(M), "max |diff|", float(np.nanmax(np.abs(M.cd - M.cd_corr))))
    print(S.groupby(["cohort", "stage"])[[c for c in S.columns if c.startswith(("cos_norm", "gen", "dprime"))]].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20)
    main(ap.parse_args())
