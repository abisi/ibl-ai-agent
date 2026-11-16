"""001 -- Pre-stimulus state axis and single-trial gating of the early whisker response (user 2026-11-11: test the hypothesis that
an attentional decision variable changes with learning and sets the strength of sensory responses; see question.md).

Sessions / trials / units as ssl-hitmiss-axis-passive 146: sessions with passive trials before AND after the active block
(132.session_trials: active from the first whisker trial, perf != 6, A1 trimmed), active trials with a lick before 35 ms
dropped, >= MIN_CLASS active whisker hits and misses; shared tracked stable units (137b); whole brain + area groups (+ fine
custom acronyms with SSL_001_AREA=area_acronym_custom, outputs *_fine); >= MIN_UNITS units per cell.
Windows (s from stimulus onset; stimulus-artefact dead zone -10..+5 ms as 146):
  STATE  -0.250..-0.060  pre-stimulus state (does not overlap the sensory baseline)
  LBASE  -0.055..-0.020  the trial's own baseline for the sensory response (146 BASE)
  SENS    0.005.. 0.035  early sensory window (146 WIN)
Per session x area:
  state axis   L2 logistic hit vs miss on z-scored STATE rates of active whisker trials (one C per cell, select_fixed_c_pooled),
               N_REP balanced subsamples x K_FOLD CV; every trial (active whisker / auditory, passive pre / post) scored only by
               models that did not train on it; score standardised as 146 ((score - hit/miss midpoint) / SD of held-out scores):
               s_state > 0 = hit-like state. state_bal_acc, and its linear-shift null (labels shifted 10-50 % of the trials,
               non-wrapping, both directions, N_SHIFT shifts x N_REP_SHIFT) -> state_bal_acc_excess.
  sensory gain g = projection of the trial's sensory response (SENS - LBASE, z-scored per unit over all trials) on the unit
               mean passive-pre whisker response (independent of the active trials); gA the same on the passive-pre auditory
               response (auditory control).
  gating       active whisker trials: partial Spearman rho(s_state, g | lick, session time [linear + quadratic]) =
               gate_rho; within hits / within misses (| time); linear-shift null (g shifted against s_state by 10-50 % of the
               trials, non-wrapping, both directions; covariates follow s_state) -> gate_rho_excess; auditory active trials:
               rho(s_state, gA | lick, time); passive whisker trials (no choice): rho(s_state, g | epoch, time within epoch).
  passive      state shift: mean s_state post - pre (whisker, auditory trials); sensory hit / miss decoder (146-like, on the
               SENS - LBASE responses) read out on passive whisker trials: change post - pre raw (ro_dW) and controlled for
               s_state (OLS ro ~ post + s_state: ro_dW_adj), same for g (g_dW, g_dW_adj).
  learning     per active half (chronological): state decodability (CV within half) and gate_rho.
Output (combined_results_ks4/ssl-prestim-state-gating/tables/): 001_state_gating{_fine}{_expert}.parquet (session x area),
001_trials{_fine}{_expert}.parquet (per trial: epoch, type, lick, time, s_state, g, gA, ro).
Run (haas, ibl-ai-agent repo root): nice .venv/bin/python projects/ssl-prestim-state-gating/exploratory-analyses/001_state_gating.py
  [SSL_STAGE=expert] [SSL_001_AREA=area_acronym_custom] [SSL_001_SESSIONS=sid1,sid2 for a pilot (outputs *_pilot)]
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
REPO = HERE.parents[2]
SCRIPTS = str(REPO / "scripts")
HM = REPO / "projects" / "ssl-hitmiss-axis-passive" / "exploratory-analyses"
TD = REPO / "projects" / "ssl-whisker-hitmiss-timeresolved-decoding" / "exploratory-analyses"
for p in (SCRIPTS, str(HM), str(TD)):
    sys.path.insert(0, p)
from axel_bisi_paths import axel_bisi_root  # noqa: E402

STATE, LBASE, SENS, DZ = (-0.250, -0.060), (-0.055, -0.020), (0.005, 0.035), (-0.010, 0.005)
MIN_UNITS, MIN_CLASS = int(os.environ.get("SSL_001_MIN_UNITS", "10")), 3
N_REP, K_FOLD, N_SHIFT, N_REP_SHIFT, N_SHIFT_GATE = 20, 5, 20, 5, 100
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "24"))
STAGE = os.environ.get("SSL_STAGE", "learning")
AREA_COL = os.environ.get("SSL_001_AREA", "area_group")
PILOT = [s for s in os.environ.get("SSL_001_SESSIONS", "").split(",") if s]
SFX = ("_fine" if AREA_COL != "area_group" else "") + ("" if STAGE == "learning" else f"_{STAGE}") + ("_pilot" if PILOT else "")
TAB = axel_bisi_root() / "combined_results_ks4" / "ssl-prestim-state-gating" / "tables"
OUT_ROWS, OUT_TRIALS = TAB / f"001_state_gating{SFX}.parquet", TAB / f"001_trials{SFX}.parquet"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import warnings
    warnings.filterwarnings("ignore")


def unit_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def cv_scores(X, y, C, rng, others, T, n_rep=N_REP):
    """Repeated balanced CV of an L2 logistic decoder. Returns (standardised held-out score per row of X, dict of standardised
    scores per matrix in `others`, balanced accuracy). Rows of X not in a repetition's subsample are scored by that
    repetition's fold models (none trained on them)."""
    hits, miss = np.where(y)[0], np.where(~y)[0]
    k = min(len(hits), len(miss))
    acc = [[] for _ in range(len(y))]
    oth = {n: [] for n in others}
    bal = []
    for _ in range(n_rep):
        sel = np.r_[rng.choice(hits, k, replace=False), rng.choice(miss, k, replace=False)]
        rest = np.setdiff1d(np.arange(len(y)), sel)
        yy = y[sel]
        nf = min(K_FOLD, k)
        perm = rng.permutation(len(sel))
        folds = [perm[i::nf] for i in range(nf)]
        s_sel, s_rest, s_oth = np.full(len(sel), np.nan), [], {n: [] for n in others}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(sel)), te)
            if len(np.unique(yy[tr])) < 2:
                continue
            clf = T._make_classifier(C).fit(X[sel[tr]], yy[tr])
            s_sel[te] = clf.decision_function(X[sel[te]])
            if len(rest):
                s_rest.append(clf.decision_function(X[rest]))
            for n, M in others.items():
                if len(M):
                    s_oth[n].append(clf.decision_function(M))
        ok = np.isfinite(s_sel)
        if not (ok & yy).any() or not (ok & ~yy).any():
            continue
        mid = 0.5 * (s_sel[ok & yy].mean() + s_sel[ok & ~yy].mean())
        sd = np.std(s_sel[ok])
        if not sd > 0:
            continue
        bal.append(0.5 * (np.mean(s_sel[ok & yy] > mid) + np.mean(s_sel[ok & ~yy] <= mid)))
        for i, s in zip(sel[ok], s_sel[ok]):
            acc[i].append((s - mid) / sd)
        if s_rest:
            for i, s in zip(rest, np.mean(s_rest, 0)):
                acc[i].append((s - mid) / sd)
        for n in others:
            if s_oth[n]:
                oth[n].append((np.mean(s_oth[n], 0) - mid) / sd)
    score = np.array([np.mean(v) if v else np.nan for v in acc])
    return score, {n: (np.mean(v, 0) if v else np.full(len(others[n]), np.nan)) for n, v in oth.items()}, (
        float(np.mean(bal)) if bal else np.nan)


def partial_rho(x, y, Z):
    """Spearman partial correlation of x and y given the columns of Z (ranks of x and y residualised on Z with an intercept)."""
    from scipy import stats
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    if ok.sum() < 8:
        return np.nan
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    A = np.c_[np.ones(ok.sum()), Z[ok]]
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    ex, ey = res(rx), res(ry)
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d > 0 else np.nan


def time_cov(t):
    tz = (t - t.mean()) / (t.std() if t.std() > 0 else 1)
    return np.c_[tz, tz ** 2]


def shift_pairs(n):
    lo, hi = max(1, int(0.1 * n)), max(1, int(0.5 * n))
    return [(k, d) for k in range(lo, hi + 1) for d in (0, 1) if n - k >= 10]


def gate_null(s, g, Z, rng):
    """Linear-shift null of partial_rho(s, g | Z): g shifted against s (and its covariates) by k trials, non-wrapping."""
    n = len(s)
    pairs = shift_pairs(n)
    if not pairs:
        return np.array([])
    out = []
    for j in rng.permutation(len(pairs))[:N_SHIFT_GATE]:
        k, d = pairs[j]
        if d:
            out.append(partial_rho(s[k:], g[: n - k], Z[k:]))
        else:
            out.append(partial_rho(s[: n - k], g[k:], Z[: n - k]))
    return np.asarray(out, float)


def process(args):
    sid, subject, rg, sets = args
    _init()
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    trs = M132.session_trials(sid, st, tt, T)
    if trs is None:
        return [dict(base, skipped_reason="no passive pre+post or no active")], []
    act = trs["active"].sort_values("start_time").reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    act = act[~((act.lick_flag == 1).to_numpy() & np.isfinite(rt) & (rt <= SENS[1]))].reset_index(drop=True)
    isw = (act.trial_type == "whisker_trial").to_numpy()
    lick = (act.lick_flag == 1).to_numpy()
    yw = lick[isw]
    if min(yw.sum(), (~yw).sum()) < MIN_CLASS:
        return [dict(base, skipped_reason=f"active whisker hits / misses {int(yw.sum())} / {int((~yw).sum())}")], []
    pre = trs["passive_pre"].sort_values("start_time").reset_index(drop=True)
    post = trs["passive_post"].sort_values("start_time").reset_index(drop=True)
    half2 = np.arange(len(act)) >= len(act) // 2
    spikes = T.load_session_unit_spikes(root, sid)
    rows, trows = [], []
    for area, units in sets.items():
        units = np.asarray(units)
        row = dict(base, area=area, n_units=len(units), n_hits=int(yw.sum()), n_misses=int((~yw).sum()),
                   n_pre=len(pre), n_post=len(post), skipped_reason=None)
        if len(units) < MIN_UNITS:
            rows.append(dict(row, skipped_reason="too few units")); continue
        M = {}
        for e, d in (("passive_pre", pre), ("active", act), ("passive_post", post)):
            t0 = d.start_time.to_numpy()
            M[e] = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [STATE, LBASE, SENS], dead_zone=DZ)
        S = {e: m[0] for e, m in M.items()}
        R = {e: m[2] - m[1] for e, m in M.items()}
        ok_u = np.all([~np.isnan(v).any(0) for v in list(S.values()) + list(R.values())], axis=0)
        Sp, Rp = np.vstack([v[:, ok_u] for v in S.values()]), np.vstack([v[:, ok_u] for v in R.values()])
        keep = (Sp.std(0) > 0) & (Rp.std(0) > 0)
        if keep.sum() < MIN_UNITS:
            rows.append(dict(row, skipped_reason="too few units after cleaning")); continue
        zS = {e: (v[:, ok_u][:, keep] - Sp[:, keep].mean(0)) / Sp[:, keep].std(0) for e, v in S.items()}
        zR = {e: (v[:, ok_u][:, keep] - Rp[:, keep].mean(0)) / Rp[:, keep].std(0) for e, v in R.items()}
        row["n_units"] = int(keep.sum())
        rng = np.random.default_rng(zlib.crc32(f"001|{sid}|{area}".encode()))
        wpre = (pre.trial_type == "whisker_trial").to_numpy()
        wpost = (post.trial_type == "whisker_trial").to_numpy()
        # sensory axes from passive-pre responses (independent of the active trials)
        uW = unit_vec(zR["passive_pre"][wpre].mean(0)) if wpre.any() else None
        uA = unit_vec(zR["passive_pre"][~wpre].mean(0)) if (~wpre).any() else None
        g = {e: zR[e] @ uW for e in zR} if uW is not None else None
        gA = {e: zR[e] @ uA for e in zR} if uA is not None else None
        # state axis on active whisker trials
        Xw = zS["active"][isw]
        C = T.select_fixed_c_pooled(Xw, yw, rng, n_folds=5)
        others = {"active_aud": zS["active"][~isw], "passive_pre": zS["passive_pre"], "passive_post": zS["passive_post"]}
        sW, sO, bal = cv_scores(Xw, yw, C, rng, others, T)
        row.update(C_state=C, state_bal_acc=bal)
        # linear-shift null of state decodability
        n = len(yw)
        valid = [(k, d) for k, d in shift_pairs(n)
                 if min((yw[k:] if d else yw[: n - k]).sum(), (~(yw[k:] if d else yw[: n - k])).sum()) >= MIN_CLASS]
        nulls = []
        for j in rng.permutation(len(valid))[:N_SHIFT]:
            k, d = valid[j]
            ys, Xs = (yw[k:], Xw[: n - k]) if d else (yw[: n - k], Xw[k:])
            nulls.append(cv_scores(Xs, ys, C, rng, {}, T, n_rep=N_REP_SHIFT)[2])
        row.update(state_bal_acc_null=float(np.nanmean(nulls)) if nulls else np.nan, state_n_shift=len(nulls))
        row["state_bal_acc_excess"] = row["state_bal_acc"] - row["state_bal_acc_null"]
        # per active half
        h2w = half2[isw]
        for h, m in (("h1", ~h2w), ("h2", h2w)):
            if min(yw[m].sum(), (~yw[m]).sum()) >= MIN_CLASS:
                row[f"state_bal_acc_{h}"] = cv_scores(Xw[m], yw[m], C, rng, {}, T, n_rep=10)[2]
        tw = act.start_time.to_numpy()[isw]
        Zc = np.c_[yw.astype(float), time_cov(tw)]
        if g is not None:
            gw = g["active"][isw]
            row["gate_rho"] = partial_rho(sW, gw, Zc)
            nl = gate_null(sW, gw, Zc, rng)
            row.update(gate_rho_null=float(np.nanmean(nl)) if len(nl) else np.nan,
                       gate_rho_pct=float((np.sum(nl >= row["gate_rho"]) + 1) / (len(nl) + 1)) if len(nl) and np.isfinite(row["gate_rho"]) else np.nan)
            row["gate_rho_excess"] = row["gate_rho"] - row["gate_rho_null"]
            for lab, m in (("hits", yw), ("misses", ~yw)):
                row[f"gate_rho_{lab}"] = partial_rho(sW[m], gw[m], time_cov(tw[m]))
            for h, m in (("h1", ~h2w), ("h2", h2w)):
                row[f"gate_rho_{h}"] = partial_rho(sW[m], gw[m], np.c_[yw[m].astype(float), time_cov(tw[m])])
            # passive whisker trials: no choice
            sp = np.r_[sO["passive_pre"][wpre], sO["passive_post"][wpost]]
            gp = np.r_[g["passive_pre"][wpre], g["passive_post"][wpost]]
            ep = np.r_[np.zeros(wpre.sum()), np.ones(wpost.sum())]
            tp = np.r_[pre.start_time.to_numpy()[wpre], post.start_time.to_numpy()[wpost]]
            tcov = np.zeros((len(tp), 2))
            for v in (0, 1):
                if (ep == v).sum() > 2:
                    tcov[ep == v] = time_cov(tp[ep == v])
            row["gate_rho_passive"] = partial_rho(sp, gp, np.c_[ep, tcov])
            row["g_dW"] = float(g["passive_post"][wpost].mean() - g["passive_pre"][wpre].mean())
            okp = np.isfinite(sp) & np.isfinite(gp)
            if okp.sum() > 6:
                A = np.c_[np.ones(okp.sum()), ep[okp], sp[okp]]
                row["g_dW_adj"] = float(np.linalg.lstsq(A, gp[okp], rcond=None)[0][1])
        if gA is not None:
            ia = ~isw
            if ia.sum() >= 8 and np.isfinite(sO["active_aud"]).sum() >= 8:
                row["gate_rho_aud"] = partial_rho(sO["active_aud"], gA["active"][ia],
                                                  np.c_[lick[ia].astype(float), time_cov(act.start_time.to_numpy()[ia])])
        # passive state shift (hit-like > 0)
        for stim, mpre, mpost in (("W", wpre, wpost), ("A", ~wpre, ~wpost)):
            row[f"state_pre_{stim}"] = float(np.nanmean(sO["passive_pre"][mpre])) if mpre.any() else np.nan
            row[f"state_post_{stim}"] = float(np.nanmean(sO["passive_post"][mpost])) if mpost.any() else np.nan
            row[f"state_d{stim}"] = row[f"state_post_{stim}"] - row[f"state_pre_{stim}"]
        row["state_active_hit"] = float(np.nanmean(sW[yw]))
        row["state_active_miss"] = float(np.nanmean(sW[~yw]))
        # sensory hit / miss decoder (SENS - LBASE) read out on passive whisker trials; change raw and controlled for state
        Rw = zR["active"][isw]
        Cr = T.select_fixed_c_pooled(Rw, yw, rng, n_folds=5)
        roW, roO, bal_r = cv_scores(Rw, yw, Cr, rng, {"passive_pre": zR["passive_pre"], "passive_post": zR["passive_post"]}, T)
        row.update(sens_bal_acc=bal_r)
        rp = np.r_[roO["passive_pre"][wpre], roO["passive_post"][wpost]] if (wpre.any() and wpost.any()) else np.array([])
        if len(rp):
            sp = np.r_[sO["passive_pre"][wpre], sO["passive_post"][wpost]]
            ep = np.r_[np.zeros(wpre.sum()), np.ones(wpost.sum())]
            row["ro_dW"] = float(np.nanmean(roO["passive_post"][wpost]) - np.nanmean(roO["passive_pre"][wpre]))
            okp = np.isfinite(rp) & np.isfinite(sp)
            if okp.sum() > 6:
                A = np.c_[np.ones(okp.sum()), ep[okp], sp[okp]]
                b = np.linalg.lstsq(A, rp[okp], rcond=None)[0]
                row.update(ro_dW_adj=float(b[1]), ro_on_state=float(b[2]))
            row["ro_state_rho_passive"] = partial_rho(sp, rp, np.c_[ep])
        rows.append(row)
        # trial-level table (whole brain and area cells)
        for e, d in (("passive_pre", pre), ("active", act), ("passive_post", post)):
            s_e = np.full(len(d), np.nan)
            if e == "active":
                s_e[isw] = sW
                s_e[~isw] = sO["active_aud"]
                r_e = np.full(len(d), np.nan); r_e[isw] = roW
            else:
                s_e = sO[e]
                r_e = roO[e]
            trows.append(pd.DataFrame(dict(session_id=sid, area=area, epoch=e, trial_type=d.trial_type.to_numpy(),
                                           lick=d.lick_flag.to_numpy(), start_time=d.start_time.to_numpy(), s_state=s_e,
                                           g=g[e] if g is not None else np.nan, gA=gA[e] if gA is not None else np.nan,
                                           ro=r_e)))
    return rows, trows


def main():
    _init()
    os.chdir(REPO)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    S = importlib.import_module("tracked_units").load_table()
    S = S[S.stable]
    if AREA_COL != "area_group":
        L = pd.read_parquet(T.AREA_LABELS_PATH, columns=["session_id", "cluster_id", AREA_COL])
        S = S.drop(columns=[AREA_COL], errors="ignore").merge(L, on=["session_id", "cluster_id"], how="left")
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == STAGE) & sess.reward_group.isin(["R+", "R-"])]
    if PILOT:
        sess = sess[sess.session_id.isin(PILOT)]
    args = []
    for r in sess.itertuples():
        s = S[S.session_id == r.session_id]
        sets = {"whole_brain": s.cluster_id.to_numpy()}
        for a, gdf in s.groupby(AREA_COL):
            if len(gdf) >= MIN_UNITS:
                sets[a] = gdf.cluster_id.to_numpy()
        args.append((r.session_id, r.subject_id, r.reward_group, sets))
    TAB.mkdir(parents=True, exist_ok=True)
    print(f"[001] {len(args)} sessions, {N_WORKERS} workers -> {OUT_ROWS.name}", flush=True)
    t0, rows, trows = time.time(), [], []
    with ProcessPoolExecutor(min(N_WORKERS, max(1, len(args))), initializer=_init) as ex:
        futs = {ex.submit(process, a): a[0] for a in args}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r, tr = f.result()
            except Exception as err:  # noqa: BLE001
                r, tr = [dict(session_id=futs[f], skipped_reason=f"error: {err!r}")], []
            rows += r
            trows += tr
            print(f"[001] [{i}/{len(args)}] {futs[f]} -- {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    tmp = OUT_ROWS.with_suffix(".partial.parquet"); out.to_parquet(tmp, index=False); os.replace(tmp, OUT_ROWS)
    if trows:
        tl = pd.concat(trows, ignore_index=True)
        tmp = OUT_TRIALS.with_suffix(".partial.parquet"); tl.to_parquet(tmp, index=False); os.replace(tmp, OUT_TRIALS)
    print(f"[001] done in {(time.time() - t0) / 60:.1f} min: {len(out)} rows, {out.skipped_reason.isna().sum()} ok", flush=True)


if __name__ == "__main__":
    main()
