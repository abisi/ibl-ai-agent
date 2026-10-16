"""127 -- Unified single-neuron analysis (user 2026-10-14): one neuron set and one event selection for every panel of the single-neuron
figure (examples, time courses, whole-brain summary, per area), three baseline definitions, a "WH-specific" measure for a second
dimension, and the session / mouse exclusion table.
Events: 051 per-event files (WH, AH = corrected first lick, start_time + RT; SL = spontaneous-lick bout onsets). Units: 081 tested
units of the non-sensory classes (stimulus class not "sensory"; lick-gated included; 081 neuron_classes), mean raw pre-lick
rate >= 0.1 Hz, from the 053 unit table (not the 081 table, which drops sessions with < 8 events per class): stable good / mua
(137_stable_units); sessions absent from the 137 table (drift file missing, user 2026-10-14): all good / mua units; sessions
without quality labels (AB105_20240314): good / mua from the NWB bc_label (non-soma excluded). The rule used is saved per session (unit_rule).
Rates: spikes in 50-ms windows centred TC = -500 .. +200 ms (5-ms step) around the lick, from the NWB spike times.
Baselines (each event):
  orig    051: WH / AH [stimulus - 1 s, stimulus - 15 ms); SL [lick - 1, lick - 0.5 s)  (different windows per event type)
  shared  the same lick-relative window for every event, [lick - 1.5, lick - 1.0) s; events kept only if no piezo lick falls in the
          window and, for WH / AH, the window ends before the stimulus (RT <= 1 s)
  raw     no baseline subtraction
Selection and measures (per baseline variant), count-matched: N_DRAW draws of N_WH WH, N_AH AH, N_SL SL events; in each draw reward-lick
neurons = non-sensory units with s_AS > 0, p < 0.05 (label permutation, N_PERM) in the window [-100, 0) ms; signed s = 2 AUC - 1 of WH vs
SL (s_WS), WH vs AH (s_WA), AH vs SL (s_AS) per time bin for the selected units of that draw. Per unit: selection frequency f (fraction
of draws) and mean s(t) over the draws in which it was selected; session (and session x area) value = sum f s(t) / sum f, kept if
sum f >= MIN_SEL (>= 3 neuron-equivalents). Window values = mean of the bins centred at -75 and -25 ms (= [-100, 0) ms).
Second dimension ("WH-specific"): per draw, fraction of all non-sensory units whose WH response lies outside the AH-SL range
(s_WS and s_WA both significant, p < 0.05, same sign: WH above both or below both), and the same among units not selected in that draw.
Exclusions: all sessions of 026 (cohort from the mouse reference sheet, 053 rule; stage from the day) through the steps cohort ->
learners rule -> trial file and tested units -> count-matched event minimum -> >= MIN_SEL selected; written per cohort x stage
(sessions and mice), per baseline variant.
Output: <OUTROOT>/unified_single_neuron/{sessions.csv, tc.npz, areas.csv, units.parquet, exclusions.csv, exclusion_table.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../127_unified_single_neuron.py [--n-proc 25]
"""
import argparse
import os
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m53 = importlib.import_module("053_roc_prelick_table")
m61 = importlib.import_module("061_roc_prelick_learners")
m81 = importlib.import_module("081_single_unit_classes")
st26 = importlib.import_module("026_roc_rates_all_sessions")
# neuron set (user 2026-10-14): "all" = every non-sensory neuron, no p-value selection (default); "pval" = reward-lick selection
# (s_AS > 0, p < 0.05) per draw, kept as an option (results in unified_single_neuron/)
SELECT = os.environ.get("PRELICK_SELECT", "all")
OUT = m51.OUTROOT / ("unified_single_neuron" if SELECT == "pval" else "unified_single_neuron_all")
TC = np.round(np.arange(-0.5, 0.2 + 1e-9, 0.005), 3); HW = 0.025
WINB = np.isin(np.round(TC, 3), [-0.075, -0.025])
SHARED = (-1.5, -1.0)
N_WH, N_AH, N_SL, N_DRAW, N_PERM, MIN_SEL = 4, 4, 8, 200, 200, 3      # 4 / 4 / 8 standard (user 2026-10-14); 200 draws final (2026-10-15)
VARIANTS = ["shared", "orig", "raw"]
PAIRS = ["WS", "WA", "AS"]
# absolute measures (user 2026-10-14): |s| minus the |s| of a label-shuffled draw (same events, classes pooled and relabelled), the
# noise floor of |s| for that draw; "aWS" etc. = |s_WS| - |s_WS|_shuffle
ALLP = PAIRS + ["a" + p_ for p_ in PAIRS]


def ranks_auc(X, a, b):
    """X units x events; s = 2 AUC - 1 (a > b) per unit"""
    Y = X[:, np.r_[a, b]]; R = rankdata(Y, axis=1); na, nb = len(a), len(b)
    return 2 * (R[:, :na].sum(1) - na * (na + 1) / 2) / (na * nb) - 1


def perm_s(X, a, b, rng):
    idx = np.r_[a, b]; na, nb = len(a), len(b)
    R = rankdata(X[:, idx], axis=1)
    f = lambda cols: 2 * (R[:, cols].sum(1) - na * (na + 1) / 2) / (na * nb) - 1
    obs = f(np.arange(na)); cnt = np.zeros(len(X))
    for _ in range(N_PERM):
        cnt += np.abs(f(rng.permutation(na + nb)[:na])) >= np.abs(obs) - 1e-12
    return obs, (1 + cnt) / (1 + N_PERM)


def session(args):
    sid, f, units = args
    from pynwb import NWBHDF5IO
    z = np.load(f, allow_pickle=True)
    lab = z["cls"].astype(str); start = z["trial_start"].astype(float); rt = z["rt"].astype(float)
    lick = start + np.nan_to_num(rt); trial = lab != "FA"
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    K = K[K.nclass.isin(m81.CLASSES) & (K.nclass != "sensory")]
    K = K[z["raw"].astype(float)[K.row.to_numpy()].mean(1) >= m51.MIN_FR].reset_index(drop=True)
    info = dict(session_id=sid, n_ns=len(K), n_WH=int((lab == "WH").sum()), n_AH=int((lab == "AH").sum()), n_SL=int((lab == "FA").sum()))
    if len(K) < MIN_SEL:
        return info, [], {}, [], []
    base_orig = (z["raw"].astype(float) - z["rates"].astype(float))[K.row.to_numpy()]
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        Uq, _ = m51.ru.process_nwb_tables(nwb)
        lk = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"].time_series["piezo_lick_times"].data[:], float))
    Uq = Uq.assign(electrode_group=Uq.electrode_group.astype(str), cluster_id=Uq.cluster_id.astype(str)).set_index(["electrode_group", "cluster_id"])
    nu, ne = len(K), len(lab)
    Rt = np.full((nu, ne, len(TC)), np.nan, np.float32); Bs = np.full((nu, ne), np.nan)
    for u, r in enumerate(K.itertuples()):
        if (r.electrode_group, r.cluster_id) not in Uq.index:
            continue
        st = np.sort(np.asarray(Uq.loc[(r.electrode_group, r.cluster_id)].spike_times))
        lo = np.searchsorted(st, lick[:, None] + TC[None] - HW); hi = np.searchsorted(st, lick[:, None] + TC[None] + HW)
        Rt[u] = (hi - lo) / (2 * HW)
        Bs[u] = (np.searchsorted(st, lick + SHARED[1]) - np.searchsorted(st, lick + SHARED[0])) / (SHARED[1] - SHARED[0])
    lickfree = np.searchsorted(lk, lick + SHARED[1]) == np.searchsorted(lk, lick + SHARED[0])
    prestim = ~trial | (lick + SHARED[1] <= start)
    ok_shared = lickfree & prestim
    info.update(shared_ok_WH=int((ok_shared & (lab == "WH")).sum()), shared_ok_AH=int((ok_shared & (lab == "AH")).sum()),
                shared_ok_SL=int((ok_shared & (lab == "FA")).sum()))
    rng = np.random.default_rng(zlib.crc32(f"{sid}|127".encode()))
    srows, arows, urows, tcs = [], [], [], {}
    for v in VARIANTS:
        if v == "raw":
            X = Rt.astype(float)
        else:
            X = Rt - (Bs if v == "shared" else base_orig)[..., None]
        valid = ok_shared if v == "shared" else np.ones(ne, bool)
        iw, ia, isl = (np.where((lab == c) & valid)[0] for c in ("WH", "AH", "FA"))
        if len(iw) < N_WH or len(ia) < N_AH or len(isl) < N_SL:
            srows.append(dict(info, variant=v, status="too few events")); continue
        Xw = np.nanmean(X[:, :, WINB], 2)
        cnt = np.zeros(nu); S = {p: np.zeros((nu, len(TC))) for p in ALLP}; spec_all, spec_out = [], []
        for _ in range(N_DRAW):
            w, a, s = rng.choice(iw, N_WH, replace=False), rng.choice(ia, N_AH, replace=False), rng.choice(isl, N_SL, replace=False)
            if SELECT == "all":
                sel = np.isfinite(Xw).all(1)
                spec_all.append(np.nan); spec_out.append(np.nan)
            else:
                sAS, pAS = perm_s(Xw, a, s, rng); sel = (pAS < 0.05) & (sAS > 0)
                sWS, pWS = perm_s(Xw, w, s, rng); sWA, pWA = perm_s(Xw, w, a, rng)
                spec = (pWS < 0.05) & (pWA < 0.05) & (np.sign(sWS) == np.sign(sWA))
                spec_all.append(spec.mean()); spec_out.append(spec[~sel].mean() if (~sel).any() else np.nan)
            if not sel.any():
                continue
            cnt[sel] += 1
            Xs = X[sel]
            for p_, (A_, B_) in {"WS": (w, s), "WA": (w, a), "AS": (a, s)}.items():
                v_ = np.stack([ranks_auc(Xs[:, :, k], A_, B_) for k in range(len(TC))], 1)
                pool = rng.permutation(np.r_[A_, B_]); A0, B0 = pool[: len(A_)], pool[len(A_):]
                v0 = np.stack([ranks_auc(Xs[:, :, k], A0, B0) for k in range(len(TC))], 1)
                S[p_][sel] += v_; S["a" + p_][sel] += np.abs(v_) - np.abs(v0)
        fsel = cnt / N_DRAW
        m = cnt > 0
        mean_s = {p_: np.where(m[:, None], S[p_] / np.maximum(cnt, 1)[:, None], np.nan) for p_ in ALLP}
        base_row = dict(info, variant=v, n_WH_used=len(iw), n_AH_used=len(ia), n_SL_used=len(isl), sum_f=float(fsel.sum()),
                        frac_WHspec=float(np.mean(spec_all)), frac_WHspec_nonsel=float(np.nanmean(spec_out)))
        if fsel.sum() < MIN_SEL:
            srows.append(dict(base_row, status="too few selected")); continue
        tc_ = {p_: np.nansum(fsel[m, None] * mean_s[p_][m], 0) / fsel[m].sum() for p_ in ALLP}
        tcs[v] = np.stack([tc_[p_] for p_ in ALLP])
        srows.append(dict(base_row, status="ok", **{f"s{p_}": float(tc_[p_][WINB].mean()) for p_ in ALLP}))
        for g, q in K[m].groupby("area_group"):
            ii = q.index.to_numpy()
            if fsel[ii].sum() < MIN_SEL:
                continue
            arows.append(dict(session_id=sid, variant=v, area=g, sum_f=float(fsel[ii].sum()),
                              **{f"s{p_}": float(np.nansum(fsel[ii, None] * mean_s[p_][ii], 0)[WINB].mean() / fsel[ii].sum()) for p_ in ALLP}))
        for i in np.where(m)[0]:
            urows.append(dict(session_id=sid, electrode_group=K.electrode_group[i], cluster_id=K.cluster_id[i], nclass=K.nclass[i],
                              area_group=K.area_group[i], variant=v, f_sel=fsel[i], hz=float(np.nanmean(Rt[i][:, WINB])),
                              **{f"s{p_}": float(mean_s[p_][i][WINB].mean()) for p_ in ALLP}))
    return info, srows, tcs, arows, urows


def units_table():
    """candidate units per session: 053 unit table + 081 stimulus classes; stable good / mua, or all good / mua where the session is
    absent from 137_stable_units (no drift file yet), or all units where the session has no quality labels"""
    m53_ = importlib.import_module("053_roc_prelick_table")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W.assign(electrode_group=W.electrode_group.astype(str), cluster_id=W.cluster_id.astype(str))
    in137 = set(pd.read_parquet(m53_.STABLE, columns=["session_id"]).session_id)
    good = ("good", "mua")
    qa = W.quality_label_all if "quality_label_all" in W else W.quality_label
    nolab = W.groupby("session_id").quality_label_all.transform(lambda x: x.isna().all()) if "quality_label_all" in W else False
    rule = np.select([W.session_id.isin(in137), nolab], ["stable good/mua", "good/mua from NWB bc_label (no quality labels)"],
                     "all good/mua (no drift file)")
    # sessions without quality labels in the unit table (AB105_20240314): good / mua from the NWB units bc_label (non-soma excluded)
    bc = []
    for sid in W.session_id[rule == "good/mua from NWB bc_label (no quality labels)"].unique():
        from pynwb import NWBHDF5IO
        with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
            u = io.read().units.to_dataframe()
        bc.append(pd.DataFrame(dict(session_id=sid, electrode_group=[getattr(g, "name", str(g)) for g in u.electrode_group],
                                    cluster_id=u.cluster_id.astype(str).to_numpy(), bc_label=u.bc_label.astype(str).to_numpy())))
    if bc:
        W = W.merge(pd.concat(bc), on=["session_id", "electrode_group", "cluster_id"], how="left")
    else:
        W["bc_label"] = np.nan
    keep = np.select([rule == "stable good/mua", rule == "good/mua from NWB bc_label (no quality labels)"],
                     [W.quality_label.isin(good), W.bc_label.isin(good)], qa.isin(good))
    W = W.assign(unit_rule=rule)[keep.astype(bool)]
    W = W.merge(m81.neuron_classes(), on=["session_id", "electrode_group", "cluster_id"], how="left")
    W["nclass"] = W.nclass.fillna("unclassified")
    return W[["session_id", "electrode_group", "cluster_id", "area_group", "nclass", "unit_rule"]].drop_duplicates(["session_id", "electrode_group", "cluster_id"])


def meta_all():
    """all 026 sessions with cohort (mouse reference sheet, 053 rule) and stage (day 0 = learning, > 0 = expert)"""
    ss = st26.all_sessions().rename(columns={"mouse": "mouse_id"})
    X = pd.read_excel(m53.MOUSE_REF).drop_duplicates("mouse_id").set_index("mouse_id")
    flag = lambda c: X[c].fillna(0).astype(str).isin(["1", "1.0", "True", "yes"]) if c in X else pd.Series(False, index=X.index)
    ref = X.reward_group.where(X.reward_group.isin(["R+", "R-"]) & ~flag("exclude") & ~flag("exclude_ephys"))
    ss["cohort"] = ss.mouse_id.map(ref); ss["sheet_group"] = ss.mouse_id.map(X.reward_group)
    ss["stage"] = np.where(ss.day.astype(int) == 0, "learning", "expert")
    return ss


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    ss = meta_all()
    U0 = units_table()
    jobs, rule = [], {}
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse_id}_roc_prelick{m51.TAG}_trials.npz"
        u = U0[U0.session_id == r.session_id][["electrode_group", "cluster_id", "nclass", "area_group"]].drop_duplicates()
        rule[r.session_id] = U0[U0.session_id == r.session_id].unit_rule.iloc[0] if len(u) else "none"
        if f.exists() and len(u) and isinstance(r.cohort, str):
            jobs.append((r.session_id, f, u))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = list(pool.imap_unordered(session, jobs))
    meta = ss[["session_id", "mouse_id", "cohort", "stage", "day"]]
    S = pd.DataFrame([x for r in res for x in r[1]]).merge(meta, on="session_id", how="left")
    S["unit_rule"] = S.session_id.map(rule)
    A = pd.DataFrame([x for r in res for x in r[3]]).merge(meta, on="session_id", how="left")
    Uo = pd.DataFrame([x for r in res for x in r[4]]).merge(meta, on="session_id", how="left")
    S.to_csv(OUT / "sessions.csv", index=False); A.to_csv(OUT / "areas.csv", index=False); Uo.to_parquet(OUT / "units.parquet", index=False)
    tc = {f"{v}|{r[0]['session_id']}": r[2][v] for r in res for v in r[2]}
    np.savez_compressed(OUT / "tc.npz", t=TC, pairs=np.array(ALLP), **tc)
    # exclusion table
    learners, _ = m61.learner_mice()
    done = {r[0]["session_id"]: r for r in res}
    rows = []
    for r in ss.itertuples():
        for v in VARIANTS:
            if not isinstance(r.cohort, str):
                why = f"cohort not R+/R- or excluded in sheet ({r.sheet_group})"
            elif r.stage == "learning" and r.mouse_id not in learners:
                why = "learners rule (non-learner Day 0)"
            elif r.session_id not in done:
                why = "no trial file or no tested units"
            else:
                sr = [x for x in done[r.session_id][1] if x["variant"] == v]
                why = "fewer than 3 non-sensory units" if not sr else ("included" if sr[0]["status"] == "ok" else sr[0]["status"])
            rows.append(dict(session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort if isinstance(r.cohort, str) else "excluded",
                             stage=r.stage, variant=v, reason=why))
    E = pd.DataFrame(rows); E.to_csv(OUT / "exclusions.csv", index=False)
    T = E.groupby(["variant", "cohort", "stage", "reason"]).agg(sessions=("session_id", "nunique"), mice=("mouse_id", "nunique")).reset_index()
    T.to_csv(OUT / "exclusion_table.csv", index=False)
    pd.set_option("display.width", 250); print(T.to_string(), flush=True)
    print("computed", S[S.status == "ok"].groupby(["variant", "cohort", "stage"]).session_id.nunique().to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--n-proc", type=int, default=25)
    compute(p.parse_args().n_proc)
