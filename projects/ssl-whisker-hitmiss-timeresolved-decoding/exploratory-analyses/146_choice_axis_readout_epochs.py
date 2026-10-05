"""146 -- Does the sensory (5-35 ms) response to the whisker stimulus become less aligned with the axis that predicts the choice,
in R- but not R+? (user 2026-10-05: "test the hypothesis that at passive_post (or perhaps active), sensory activity becomes
less aligned with an axis that predicts choices in R-, not in R+ ... do the active decoder tested on passive pre and passive
post ... do all of these except the time course; pilot on a few mice first").
Sessions / trials: learning stage with passive trials before AND after the active block (132.session_trials: active =
prep_modality_trials from the first whisker trial, perf != 6, A1-trimmed; passive = labelled passive whisker / auditory
trials). Active trials with a lick before 35 ms are excluded. Epochs: passive_pre, active_1 / active_2 (chronological halves of
the active trials), passive_post.
Units (skills/ssl-valid-data "Unit sets", 137 table): `stable` and `good` (= good AND stable), each restricted to units firing
>= 0.5 Hz over the span of every epoch (tracked).
Responses: rate (Hz) 5-35 ms after stimulus onset minus the unit's mean -55..-20 ms baseline within its epoch; z-scored per unit
over all trials (pooled epochs) for the decoder; evoked patterns = baseline-subtracted rates scaled by the same SD (not centred).
1 choice decoder: hit vs miss (lick on active whisker trials), L2 logistic regression (one C per session), N_REP repetitions of a
  balanced subsample (min(hits, misses) of each class) with stratified K-fold CV. A trial is scored only by models that did not
  train on it: held-out active whisker trials, all active auditory trials, all passive trials.
  readout (anchored): (score - midpoint) / half-distance between the held-out hit and miss means of that repetition
  (+1 = like an active hit, -1 = like an active miss); readout (standardised): (score - midpoint) / SD of the held-out active
  whisker scores. Null: the same with the hit / miss labels shuffled before training (standardised readout only).
2 decomposition with the mean-difference choice axis CD = mean(hits) - mean(misses) from a random half A of the balanced
  subsample (reliability = cos(CD_A, CD_B)): per epoch and stimulus, size of the mean evoked pattern (norm / sqrt(n units)),
  cos(evoked pattern, CD_A), projection on unit CD_A; active patterns from trials not in A.
3 state space: condition means (evoked patterns) projected on the plane of unit(CD, all active whisker trials) and the passive-pre
  whisker evoked pattern orthogonalised to it.
4 behaviour: whisker hit rate in each active half.
Output: 146_choice_axis_readout.parquet (session x unit set). Run (haas, repo root): python .../146_choice_axis_readout_epochs.py
Pilot: SSL_146_SESSIONS="sid1,sid2,..." restricts the sessions (output 146_choice_axis_readout_pilot.parquet).
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
MIN_RATE, MIN_UNITS, MIN_CLASS = 0.5, 20, 6
N_REP, K_FOLD, N_NULL = 20, 5, 20
EP4 = ["passive_pre", "active_1", "active_2", "passive_post"]
UNIT_SETS = ["stable", "good"]
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "24"))
PILOT = [s for s in os.environ.get("SSL_146_SESSIONS", "").split(",") if s]
OUT_PATH = OUT / ("146_choice_axis_readout_pilot.parquet" if PILOT else "146_choice_axis_readout.parquet")


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else np.nan


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def tracked(cands, spikes, segs):
    out = []
    for cid in cands:
        sp = spikes.get(cid, spikes.get(int(cid), np.array([])))
        if len(sp) and all((np.searchsorted(sp, b) - np.searchsorted(sp, a)) / max(b - a, 1e-6) >= MIN_RATE for a, b in segs):
            out.append(cid)
    return np.asarray(out)


def decoder_readout(Za, lick, others, C, rng, T, shuffle=False):
    """others: dict name -> (n x units) matrices scored by every model. Returns per-trial mean readouts (anchored, standardised)
    for the held-out active whisker trials and for each matrix in others."""
    from ssl_timeresolved_decoding import _make_classifier
    hits, miss = np.where(lick)[0], np.where(~lick)[0]
    k = min(len(hits), len(miss))
    acc_act = {key: [[] for _ in range(len(lick))] for key in ("anch", "std")}
    acc_oth = {n: {key: [] for key in ("anch", "std")} for n in others}
    bal = []
    for _ in range(N_REP):
        sel = np.r_[rng.choice(hits, k, replace=False), rng.choice(miss, k, replace=False)]
        y = lick[sel].copy()
        if shuffle:
            y = rng.permutation(y)
        nf = min(K_FOLD, k)
        perm = rng.permutation(len(sel))
        folds = [perm[i::nf] for i in range(nf)]
        score_sel = np.full(len(sel), np.nan)
        oth = {n: [] for n in others}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(sel)), te)
            if len(np.unique(y[tr])) < 2:
                continue
            clf = _make_classifier(C).fit(Za[sel[tr]], y[tr])
            score_sel[te] = clf.decision_function(Za[sel[te]])
            for n, M in others.items():
                oth[n].append(clf.decision_function(M) if len(M) else np.array([]))
        ok = np.isfinite(score_sel)
        yt = lick[sel]                                   # true labels for anchoring (also under the shuffle null)
        mh, mm = score_sel[ok & yt].mean(), score_sel[ok & ~yt].mean()
        mid, half, sd = 0.5 * (mh + mm), 0.5 * (mh - mm), np.std(score_sel[ok])
        bal.append(0.5 * (np.mean(score_sel[ok & yt] > 0) + np.mean(score_sel[ok & ~yt] <= 0)))
        for i, s in zip(sel[ok], score_sel[ok]):
            acc_act["anch"][i].append((s - mid) / half if half != 0 else np.nan)
            acc_act["std"][i].append((s - mid) / sd if sd > 0 else np.nan)
        for n in others:
            if oth[n] and len(oth[n][0]):
                s = np.mean(oth[n], axis=0)
                acc_oth[n]["anch"].append((s - mid) / half if half != 0 else s * np.nan)
                acc_oth[n]["std"].append((s - mid) / sd if sd > 0 else s * np.nan)
    act = {key: np.array([np.nanmean(v) if v else np.nan for v in acc_act[key]]) for key in acc_act}
    oth = {n: {key: np.nanmean(acc_oth[n][key], axis=0) if acc_oth[n][key] else np.array([]) for key in ("anch", "std")} for n in others}
    return act, oth, float(np.mean(bal))


def process(args):
    sid, subject, rg, sets = args
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
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    early = (act.lick_flag == 1).to_numpy() & np.isfinite(rt) & (rt <= WIN[1])
    act = act[~early].reset_index(drop=True)
    half2 = np.arange(len(act)) >= len(act) // 2
    isw_a = (act.trial_type == "whisker_trial").to_numpy()
    lick_a = (act.lick_flag == 1).to_numpy()
    wl = lick_a[isw_a]
    if min(wl.sum(), (~wl).sum()) < MIN_CLASS:
        return [dict(base, skipped_reason=f"active whisker hits / misses {int(wl.sum())} / {int((~wl).sum())} < {MIN_CLASS}")]
    pre, post = trs["passive_pre"].reset_index(drop=True), trs["passive_post"].reset_index(drop=True)
    t_mid = act.start_time.iloc[len(act) // 2]
    segs = [(pre.start_time.min() - 1, pre.start_time.max() + 1), (act.start_time.min() - 1, t_mid),
            (t_mid, act.start_time.max() + 1), (post.start_time.min() - 1, post.start_time.max() + 1)]
    spikes = T.load_session_unit_spikes(root, sid)
    beh = dict(hit_rate_1=float(lick_a[isw_a & ~half2].mean()), hit_rate_2=float(lick_a[isw_a & half2].mean()),
               n_whisker_active=int(isw_a.sum()), n_hits=int(wl.sum()), n_misses=int((~wl).sum()),
               n_pre_w=int((pre.trial_type == "whisker_trial").sum()), n_pre_a=int((pre.trial_type == "auditory_trial").sum()),
               n_post_w=int((post.trial_type == "whisker_trial").sum()), n_post_a=int((post.trial_type == "auditory_trial").sum()))
    rows = []
    for (uset, area), cands in sets.items():
        units = tracked(cands, spikes, segs)
        row = dict(base, unit_set=uset, area=area, n_units=len(units), **beh)
        if len(units) < MIN_UNITS:
            rows.append(dict(row, skipped_reason="too few tracked units")); continue
        X = {}
        for e, d in (("passive_pre", pre), ("active", act), ("passive_post", post)):
            t0 = d.start_time.to_numpy()
            r, b = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
            X[e] = r - np.nanmean(b, 0, keepdims=True)
        ok_u = np.all([~np.isnan(v).any(0) for v in X.values()], axis=0)
        P = np.vstack([v[:, ok_u] for v in X.values()]); mu, sd = P.mean(0), P.std(0); keep = sd > 0
        if keep.sum() < MIN_UNITS:
            rows.append(dict(row, skipped_reason="too few units after cleaning")); continue
        Z = {e: (v[:, ok_u][:, keep] - mu[keep]) / sd[keep] for e, v in X.items()}
        E = {e: v[:, ok_u][:, keep] / sd[keep] for e, v in X.items()}
        row["n_units"] = int(keep.sum())
        rng = np.random.default_rng(zlib.crc32(f"146|{sid}|{uset}|{area}".encode()))
        Zw, lick_w = Z["active"][isw_a], wl
        C = T.select_fixed_c_pooled(Zw, lick_w, rng, n_folds=5)
        others = {"passive_pre": Z["passive_pre"], "passive_post": Z["passive_post"], "active_aud": Z["active"][~isw_a]}
        act_r, oth_r, bal = decoder_readout(Zw, lick_w, others, C, rng, T)
        _, oth_n, _ = decoder_readout(Zw, lick_w, others, C, rng, T, shuffle=True)
        row.update(decoder_bal_acc=bal, C=C)
        # drift control: decoder trained on the FIRST active half only (cannot learn a late-session state); the second
        # active half is scored as an unseen block like the passive epochs
        h2w_ = half2[isw_a]
        if min(lick_w[~h2w_].sum(), (~lick_w[~h2w_]).sum()) >= MIN_CLASS:
            oth1 = dict(others, active_2W=Zw[h2w_])
            _, o1, bal1 = decoder_readout(Zw[~h2w_], lick_w[~h2w_], oth1, C, rng, T)
            row["decoder1_bal_acc"] = bal1
            for e in ("passive_pre", "passive_post"):
                w_ = (pre if e == "passive_pre" else post).trial_type.eq("whisker_trial").to_numpy()
                row[f"ro_h1dec_{e}_W"] = float(np.nanmean(o1[e]["std"][w_]))
                row[f"ro_h1dec_{e}_A"] = float(np.nanmean(o1[e]["std"][~w_]))
            row["ro_h1dec_active_2_W"] = float(np.nanmean(o1["active_2W"]["std"]))
        wi = {"passive_pre": (pre.trial_type == "whisker_trial").to_numpy(), "passive_post": (post.trial_type == "whisker_trial").to_numpy()}
        h2w = half2[isw_a]; h2a = half2[~isw_a]
        for key in ("anch", "std"):
            for e in ("passive_pre", "passive_post"):
                row[f"ro_{key}_{e}_W"] = float(np.nanmean(oth_r[e][key][wi[e]]))
                row[f"ro_{key}_{e}_A"] = float(np.nanmean(oth_r[e][key][~wi[e]]))
                if key == "std":
                    row[f"null_std_{e}_W"] = float(np.nanmean(oth_n[e][key][wi[e]]))
                    row[f"null_std_{e}_A"] = float(np.nanmean(oth_n[e][key][~wi[e]]))
            for h, m in ((1, ~h2w), (2, h2w)):
                row[f"ro_{key}_active_{h}_W"] = float(np.nanmean(act_r[key][m]))
                row[f"ro_{key}_active_{h}_hit"] = float(np.nanmean(act_r[key][m & lick_w]))
                row[f"ro_{key}_active_{h}_miss"] = float(np.nanmean(act_r[key][m & ~lick_w]))
            for h, m in ((1, ~h2a), (2, h2a)):
                row[f"ro_{key}_active_{h}_A"] = float(np.nanmean(oth_r["active_aud"][key][m]))
        # 2 decomposition with the mean-difference axis
        hits, miss = np.where(lick_w)[0], np.where(~lick_w)[0]
        k = min(len(hits), len(miss))
        Ew = E["active"][isw_a]; Ea = E["active"][~isw_a]
        acc = {}
        rel = []
        for _ in range(N_REP):
            ph, pm = rng.permutation(hits)[:k], rng.permutation(miss)[:k]
            A_h, B_h, A_m, B_m = ph[: k // 2], ph[k // 2:], pm[: k // 2], pm[k // 2:]
            cdA = Zw[A_h].mean(0) - Zw[A_m].mean(0); cdB = Zw[B_h].mean(0) - Zw[B_m].mean(0)
            rel.append(cos(cdA, cdB)); u = unit(cdA)
            notA = np.setdiff1d(np.arange(len(lick_w)), np.r_[A_h, A_m])
            pats = {}
            for e, d in (("passive_pre", pre), ("passive_post", post)):
                pats[(e, "W")] = E[e][wi[e]].mean(0); pats[(e, "A")] = E[e][~wi[e]].mean(0)
            for h, mW, mA in ((1, ~h2w, ~h2a), (2, h2w, h2a)):
                nw = notA[mW[notA]]
                pats[(f"active_{h}", "W")] = Ew[nw].mean(0) if len(nw) else np.full(Ew.shape[1], np.nan)
                pats[(f"active_{h}", "A")] = Ea[mA].mean(0) if mA.any() else np.full(Ea.shape[1], np.nan)
            for (e, s), p in pats.items():
                for name, val in (("size", np.linalg.norm(p) / np.sqrt(len(p))), ("cos", cos(p, cdA)), ("proj", float(p @ u))):
                    acc.setdefault(f"{name}_{e}_{s}", []).append(val)
        row["cd_reliability"] = float(np.nanmean(rel))
        for kk, v in acc.items():
            row[kk] = float(np.nanmean(v))
        # 3 state space: plane of unit(CD all) and passive-pre whisker pattern orthogonalised
        u1 = unit(Zw[lick_w].mean(0) - Zw[~lick_w].mean(0))
        w0 = E["passive_pre"][wi["passive_pre"]].mean(0); w0 = w0 - (w0 @ u1) * u1; u2 = unit(w0)
        # variant y-axis: passive-pre whisker - auditory axis (stimulus identity), orthogonalised to the choice axis
        v0 = E["passive_pre"][wi["passive_pre"]].mean(0) - E["passive_pre"][~wi["passive_pre"]].mean(0)
        v0 = v0 - (v0 @ u1) * u1; u3 = unit(v0)
        conds = {"pre_W": E["passive_pre"][wi["passive_pre"]], "pre_A": E["passive_pre"][~wi["passive_pre"]],
                 "post_W": E["passive_post"][wi["passive_post"]], "post_A": E["passive_post"][~wi["passive_post"]]}
        for h, mW, mA in ((1, ~h2w, ~h2a), (2, h2w, h2a)):
            conds[f"act{h}_hit"] = Ew[mW & lick_w]; conds[f"act{h}_miss"] = Ew[mW & ~lick_w]; conds[f"act{h}_A"] = Ea[mA]
        for cname, M in conds.items():
            m = M.mean(0) if len(M) else np.full(len(u1), np.nan)
            row[f"ss_{cname}_x"], row[f"ss_{cname}_y"], row[f"ss2_{cname}_y"] = float(m @ u1), float(m @ u2), float(m @ u3)
        rows.append(dict(row, skipped_reason=None))
    return rows


def main():
    os.chdir(OUT.parents[2])
    from axel_bisi_paths import axel_bisi_root
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    S = pd.read_parquet(axel_bisi_root() / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet")
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    if PILOT:
        sess = sess[sess.session_id.isin(PILOT)]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    args = []
    for r in sess.itertuples():
        if r.session_id in done:
            continue
        s = S[S.session_id == r.session_id]
        sets = {}
        for u in UNIT_SETS:
            su = s[s[u]]
            sets[(u, "whole_brain")] = su.cluster_id.to_numpy()
            for ag, g in su.groupby("area_group"):
                if len(g) >= MIN_UNITS:
                    sets[(u, ag)] = g.cluster_id.to_numpy()
        args.append((r.session_id, r.subject_id, r.reward_group, sets))
    print(f"[146] {len(args)} sessions, {N_WORKERS} workers -> {OUT_PATH.name}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(min(N_WORKERS, max(1, len(args))), initializer=_init) as ex:
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
            print(f"[146] [{i}/{len(args)}] {a[0]}: {new.skipped_reason.isna().sum()} ok -- {time.time() - t0:.0f}s", flush=True)
    print("[146] DONE", flush=True)


if __name__ == "__main__":
    main()
