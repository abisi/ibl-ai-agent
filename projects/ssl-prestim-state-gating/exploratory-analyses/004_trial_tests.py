"""004 -- Further trial-level tests of the state-gating hypothesis (user 2026-11-12: "explore this hypothesis more"), from the
001 per-trial tables (held-out state scores s_state, sensory gain g, readout ro), day-0 learners, session = unit.

T1 dose-response   per session, active whisker trials split in state quintiles (ranks of s_state residualised on
                   session time, linear + quadratic; g residualised the same way): P(lick) per quintile, and g
                   per quintile within hits and within misses (does the sensory response scale with the state?).
T2 movement / arousal confound   pre-stimulus jaw, nose and whisker motion in [-0.2, 0) s and pupil area in [-2, 0) s
                   (Figure 1 movements cache, ssl-paper-figures shared_movements, dlc_helpers; whisker trials of day 0,
                   joined on session_id + start_time): Spearman rho(s_state, movement); does s_state predict the lick beyond
                   movements (logistic, held-out AUC of movements only vs movements + s_state, 5-fold CV); gating
                   controlled for movements rho(s_state, g | lick, time, movements); arousal gating rho(pupil, g | lick, time).
T3 cross-area gating   whole-brain state vs each area's gain g (and each area's own state vs the Ss-whisker gain):
                   rho(s, g | lick, time), linear-shift null (10-50 % of the trials, non-wrapping, both directions).
T4 learning trial  s_state and g aligned to the session's learning trial (ssl-learning-trial-identification 006
                   lt_recommended_broad; whisker-trial ordinal of the active whisker trials kept in 001), within hits and
                   within misses separately (so the trajectory is not the hit-rate change), bins of 10 whisker trials;
                   not separable from session time (the learning trial is a position in the session).
T5 modality        AUC of the whisker-trained state for auditory hits vs misses (is the state whisker-specific or general
                   engagement?), sessions with >= 3 auditory misses.
Output: tables/004_{dose,movement,crossarea,learning,auditory}{_fine}.parquet
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/004_trial_tests.py [--fine]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(HERE), str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m01 = importlib.import_module("001_state_gating")
m61 = importlib.import_module("061_roc_prelick_learners")
ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-prestim-state-gating"
TAB = ROOT / "tables"
MOV = axel_bisi_root() / "manuscripts" / "Bisi_Hamon" / "Figures" / "_cache" / "movements_day0_v3.parquet"
LT = PROJECTS / "ssl-learning-trial-identification" / "artifacts" / "006_learning_trials_final.csv"
MOVS = ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]
NQ, LT_BIN, LT_SPAN, N_NULL = 5, 10, 30, 50
WB, SSW = "whole_brain", {"area_group": "Somatosensory-whisker", "fine": "SSp-bfd"}


def zc(v):
    v = np.asarray(v, float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 0 else 1)


def rho_with_null(s, g, Z, rng):
    r = m01.partial_rho(s, g, Z)
    m01.N_SHIFT_GATE = N_NULL
    nl = m01.gate_null(s, g, Z, rng)
    nl = nl[np.isfinite(nl)]
    return r, (np.median(nl) if len(nl) else np.nan)


def cv_auc(X, y, k=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    if min(y.sum(), (~y).sum()) < k:
        return np.nan
    p = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        p[te] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr]).decision_function(X[te])
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def main(fine):
    sfx = "_fine" if fine else ""
    rng = np.random.default_rng(0)
    learners, _ = m61.learner_mice()
    R = pd.read_parquet(TAB / f"001_state_gating{sfx}.parquet")
    R = R[R.skipped_reason.isna() & R.mouse_id.isin(learners)]
    meta = R.drop_duplicates("session_id").set_index("session_id")[["mouse_id", "reward_group"]]
    T = pd.read_parquet(TAB / f"001_trials{sfx}.parquet")
    T = T[T.session_id.isin(meta.index)].join(meta, on="session_id")
    A = T[(T.epoch == "active") & (T.trial_type == "whisker_trial")].sort_values(["session_id", "area", "start_time"]).copy()
    A["lick"] = A.lick.astype(bool)
    A["ordinal"] = A.groupby(["session_id", "area"]).cumcount() + 1
    ssw = SSW["fine" if fine else "area_group"]

    # T1 dose-response (every area)
    dose = []
    for (sid, area), d in A.groupby(["session_id", "area"]):
        d = d.dropna(subset=["s_state"])
        if len(d) < 4 * NQ:
            continue
        # state and response residualised on session time (linear + quadratic): both drift over the session
        tc = np.c_[np.ones(len(d)), m01.time_cov(d.start_time.to_numpy(float))]
        res = lambda v: v - tc @ np.linalg.lstsq(tc, v, rcond=None)[0]
        q = pd.qcut(stats.rankdata(res(d.s_state.to_numpy(float))), NQ, labels=False)
        gv = d.g.to_numpy(float)
        ok = np.isfinite(gv)
        gz = np.full(len(d), np.nan)
        gz[ok] = zc(gv[ok] - tc[ok] @ np.linalg.lstsq(tc[ok], gv[ok], rcond=None)[0])
        for i in range(NQ):
            m = q == i
            row = dict(session_id=sid, area=area, quintile=i + 1, n=int(m.sum()), p_lick=d.lick[m].mean(),
                       s_state=d.s_state[m].mean(), g=np.nanmean(gz[m]))
            for lab, mm in (("hits", d.lick.to_numpy()), ("misses", ~d.lick.to_numpy())):
                row[f"g_{lab}"] = np.nanmean(gz[m & mm]) if (m & mm).sum() >= 2 else np.nan
            dose.append(row)
    pd.DataFrame(dose).join(meta, on="session_id").to_parquet(TAB / f"004_dose{sfx}.parquet", index=False)

    # T3 cross-area gating
    cross = []
    for sid, d in A.groupby("session_id"):
        P = d.pivot_table(index="start_time", columns="area", values=["s_state", "g"])
        lick = d.drop_duplicates("start_time").set_index("start_time").lick.reindex(P.index).to_numpy(float)
        Z = np.c_[lick, m01.time_cov(P.index.to_numpy(float))]
        areas = P["g"].columns
        if WB in areas:
            for a in areas:
                r, nl = rho_with_null(P["s_state"][WB].to_numpy(), P["g"][a].to_numpy(), Z, rng)
                cross.append(dict(session_id=sid, kind="whole-brain state -> area gain", state_area=WB, gain_area=a, rho=r, null=nl))
        if ssw in areas:
            for a in P["s_state"].columns:
                r, nl = rho_with_null(P["s_state"][a].to_numpy(), P["g"][ssw].to_numpy(), Z, rng)
                cross.append(dict(session_id=sid, kind=f"area state -> {ssw} gain", state_area=a, gain_area=ssw, rho=r, null=nl))
    C = pd.DataFrame(cross).join(meta, on="session_id")
    C["excess"] = C.rho - C.null
    C.to_parquet(TAB / f"004_crossarea{sfx}.parquet", index=False)

    # T5 auditory hits vs misses on the whisker state axis
    from sklearn.metrics import roc_auc_score
    aud = []
    AU = T[(T.epoch == "active") & (T.trial_type == "auditory_trial")].dropna(subset=["s_state"])
    for (sid, area), d in AU.groupby(["session_id", "area"]):
        y = d.lick.astype(bool).to_numpy()
        if min(y.sum(), (~y).sum()) >= 3:
            aud.append(dict(session_id=sid, area=area, n_aud_miss=int((~y).sum()), auc_aud=roc_auc_score(y, d.s_state)))
    for (sid, area), d in A.dropna(subset=["s_state"]).groupby(["session_id", "area"]):
        y = d.lick.to_numpy()
        if min(y.sum(), (~y).sum()) >= 3:
            aud.append(dict(session_id=sid, area=area, auc_whisker=roc_auc_score(y, d.s_state)))
    U = pd.DataFrame(aud).groupby(["session_id", "area"]).first().reset_index().join(meta, on="session_id")
    U.to_parquet(TAB / f"004_auditory{sfx}.parquet", index=False)

    # T4 learning trial
    L = pd.read_csv(LT)[["session_id", "lt_recommended_broad"]].dropna()
    lt = L.set_index("session_id").lt_recommended_broad
    LA = A[A.session_id.isin(lt.index)].copy()
    LA["rel"] = LA.ordinal - LA.session_id.map(lt)
    LA = LA[LA.rel.abs() <= LT_SPAN]
    LA["bin"] = (np.floor(LA.rel / LT_BIN) * LT_BIN).astype(int)
    LA["gz"] = LA.groupby(["session_id", "area"]).g.transform(zc)
    LA["sz"] = LA.groupby(["session_id", "area"]).s_state.transform(zc)
    lrows = []
    for (sid, area, b), d in LA.groupby(["session_id", "area", "bin"]):
        row = dict(session_id=sid, area=area, bin=b, n=len(d), p_lick=d.lick.mean())
        for lab, m in (("all", np.ones(len(d), bool)), ("hits", d.lick.to_numpy()), ("misses", ~d.lick.to_numpy())):
            row[f"s_{lab}"] = d.sz[m].mean() if m.sum() else np.nan
            row[f"g_{lab}"] = d.gz[m].mean() if m.sum() else np.nan
        lrows.append(row)
    pd.DataFrame(lrows).join(meta, on="session_id").to_parquet(TAB / f"004_learning{sfx}.parquet", index=False)

    # T2 movements / pupil (whisker trials of day 0)
    M = pd.read_parquet(MOV)[["session_id", "start_time"] + MOVS]
    J = A.merge(M, on=["session_id", "start_time"], how="inner")
    mrows = []
    for (sid, area), d in J.groupby(["session_id", "area"]):
        d = d.dropna(subset=["s_state", "g"])
        if len(d) < 30:
            continue
        mv = [c for c in MOVS if d[c].notna().mean() > 0.8]
        d = d.dropna(subset=mv)
        if len(d) < 30 or not mv:
            continue
        y = d.lick.to_numpy()
        Xm = np.c_[[zc(d[c]) for c in mv]].T
        Zt = np.c_[y.astype(float), m01.time_cov(d.start_time.to_numpy(float))]
        row = dict(session_id=sid, area=area, n=len(d), movements=",".join(mv),
                   auc_mov=cv_auc(Xm, y), auc_mov_state=cv_auc(np.c_[Xm, zc(d.s_state)], y),
                   auc_state=cv_auc(zc(d.s_state)[:, None], y),
                   gate_rho=m01.partial_rho(d.s_state.to_numpy(), d.g.to_numpy(), Zt),
                   gate_rho_movctrl=m01.partial_rho(d.s_state.to_numpy(), d.g.to_numpy(), np.c_[Zt, Xm]),
                   state_on_lick_given_mov=m01.partial_rho(d.s_state.to_numpy(), y.astype(float),
                                                           np.c_[m01.time_cov(d.start_time.to_numpy(float)), Xm]))
        for c in MOVS:
            ok = d[c].notna()
            row[f"rho_state_{c}"] = stats.spearmanr(d.s_state[ok], d[c][ok])[0] if c in mv else np.nan
            row[f"gate_{c}"] = m01.partial_rho(d[c].to_numpy(), d.g.to_numpy(), Zt) if c in mv else np.nan
        mrows.append(row)
    pd.DataFrame(mrows).join(meta, on="session_id").to_parquet(TAB / f"004_movement{sfx}.parquet", index=False)

    # whole-brain print
    pd.set_option("display.width", 250)
    def wil(v):
        v = pd.Series(v).dropna()
        return (len(v), round(v.mean(), 3), round(stats.wilcoxon(v).pvalue, 4)) if len(v) >= 5 else (len(v), np.nan, np.nan)
    Mv = pd.read_parquet(TAB / f"004_movement{sfx}.parquet")
    Mv = Mv[Mv.area == WB]
    for c in ("R+", "R-"):
        g = Mv[Mv.reward_group == c]
        print(c, "movement", {k: wil(g[k]) for k in ["auc_mov", "auc_state", "auc_mov_state", "state_on_lick_given_mov",
                                                    "gate_rho", "gate_rho_movctrl"] + [f"rho_state_{m}" for m in MOVS]
                              + [f"gate_{m}" for m in MOVS]})
        print(c, "auc_mov_state - auc_mov", wil(g.auc_mov_state - g.auc_mov))
        u = U[(U.area == WB) & (U.reward_group == c)]
        print(c, "auditory AUC - 0.5", wil(u.auc_aud - 0.5), "whisker", wil(u.auc_whisker - 0.5))
        cc = C[(C.reward_group == c) & (C.kind.str.startswith("whole"))]
        print(c, "cross wb->area", cc.groupby("gain_area").excess.apply(lambda v: wil(v)).to_dict())
        cc = C[(C.reward_group == c) & (~C.kind.str.startswith("whole"))]
        print(c, f"cross area->{ssw}", cc.groupby("state_area").excess.apply(lambda v: wil(v)).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", action="store_true")
    main(ap.parse_args().fine)
