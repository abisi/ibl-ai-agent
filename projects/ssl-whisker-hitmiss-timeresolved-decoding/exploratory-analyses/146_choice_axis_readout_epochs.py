"""146 -- Does the sensory (5-35 ms) response to the whisker stimulus become less aligned with the axis that predicts the choice,
in R- but not R+? (user 2026-10-05: "test the hypothesis that at passive_post (or perhaps active), sensory activity becomes
less aligned with an axis that predicts choices in R-, not in R+ ... do the active decoder tested on passive pre and passive
post ... do all of these except the time course; pilot on a few mice first").
Sessions / trials: learning stage with passive trials before AND after the active block (132.session_trials: active =
prep_modality_trials from the first whisker trial, perf != 6, A1-trimmed; passive = labelled passive whisker / auditory
trials). Active trials with a lick before 35 ms are excluded. Epochs: passive_pre, active_1 / active_2 (chronological halves of
the active trials), passive_post.
Session selection (documented 2026-10-05): learning-stage sessions (one per mouse, R+ / R- from the mouse sheet) with passive
trials before AND after the active block, >= 3 active whisker hits and >= 3 misses (after excluding licks before 35 ms; was 6),
and >= 20 tracked units in the unit set (per area: >= 20). Skipped sessions and the reason are kept in the output.
Response variants (whole brain): epochbase (rate minus the epoch-mean baseline, main), trialbase (minus the trial's own
baseline), baseline (the -55..-20 ms baseline window alone: state without the sensory response).
Controls stored per session: epoch durations, gaps between epochs, clock time, rewards collected, mean baseline rate per epoch.
Units: the shared Part III tracked units (tracked_units.py / 137b, identical to 133 / 134 / 135 / 140): `stable` = 137 stable
firing >= 0.5 Hz in passive pre, passive post and both active halves at every cut point; `good` = those with quality good.
Responses: rate (Hz) 5-35 ms after stimulus onset minus the unit's mean -55..-20 ms baseline within its epoch; z-scored per unit
over all trials (pooled epochs) for the decoder; evoked patterns = baseline-subtracted rates scaled by the same SD (not centred).
1 choice decoder: hit vs miss (lick on active whisker trials), L2 logistic regression (one C per session), N_REP repetitions of a
  balanced subsample (min(hits, misses) of each class) with stratified K-fold CV. A trial is scored only by models that did not
  train on it: held-out active whisker trials, all active auditory trials, all passive trials.
  readout (anchored): (score - midpoint) / half-distance between the held-out hit and miss means of that repetition
  (+1 = like an active hit, -1 = like an active miss); readout (standardised): (score - midpoint) / SD of the held-out active
  whisker scores. Null: the same with the hit / miss labels shuffled before training (standardised readout only).
  Linear-shift null (whole brain, 2026-10-05): labels shifted against the time-ordered active whisker trials by 10-50 % of
  the trials (non-wrapping; N_SHIFT shifts x N_REP_SHIFT repetitions, drawn without replacement from the (lag, direction) pairs
  that keep >= MIN_CLASS hits and misses, so every included session has a null), readout scale anchored on the trials'
  true labels; per shift the passive post - pre readout change (W, A, W - A). Stored: null mean / SD, excess = real - null
  mean, one-sided percentile. It keeps slow drift in labels and activity, so a decoder that learned session time is in the null.
2 decomposition with the mean-difference choice axis CD = mean(hits) - mean(misses) from a random half A of the balanced
  subsample (reliability = cos(CD_A, CD_B)): per epoch and stimulus, size of the mean evoked pattern (norm / sqrt(n units)),
  cos(evoked pattern, CD_A), projection on unit CD_A; active patterns from trials not in A.
3 state space: condition means (evoked patterns) projected on unit(CD, all active whisker trials) (x) and on three y axes, each
  orthogonalised to x: passive-pre whisker pattern (ss_), passive-pre whisker - auditory (ss2_), passive-pre auditory pattern (ss3_).
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
MIN_RATE, MIN_UNITS, MIN_CLASS = 0.5, 20, 3     # MIN_CLASS: >= 3 active whisker hits and >= 3 misses (user 2026-10-05; was 6)
RESPONSES = ["epochbase", "trialbase", "baseline"]
N_REP, K_FOLD, N_NULL = 20, 5, 20
N_SHIFT, N_REP_SHIFT = 50, 5                   # linear-shift null: shifts per session, balanced repetitions per shift
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


def decoder_readout(Za, lick, others, C, rng, T, shuffle=False, anchor=None, n_rep=None):
    """others: dict name -> (n x units) matrices scored by every model. Returns per-trial mean readouts (anchored, standardised)
    for the held-out active whisker trials and for each matrix in others. `lick` = training labels (balanced on them);
    `anchor` = the trials' true labels for the readout scale (default `lick`; differs under the linear-shift null)."""
    from ssl_timeresolved_decoding import _make_classifier
    hits, miss = np.where(lick)[0], np.where(~lick)[0]
    k = min(len(hits), len(miss))
    acc_act = {key: [[] for _ in range(len(lick))] for key in ("anch", "std")}
    acc_oth = {n: {key: [] for key in ("anch", "std")} for n in others}
    bal = []
    anchor = lick if anchor is None else anchor
    for _ in range(n_rep or N_REP):
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
        yt = anchor[sel]                                 # true labels for anchoring (also under the shuffle / shift nulls)
        if not (ok & yt).any() or not (ok & ~yt).any():
            continue
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
    if not bal:
        bal = [np.nan]
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
    # engagement / timing controls (session level)
    act_all = trs["active"]
    is_rew = lambda d: (d.lick_flag == 1) & ((d.trial_type == "auditory_trial") | ((d.trial_type == "whisker_trial") & (rg == "R+")))
    ts = tt[tt.session_id == sid]
    ts_act = ts[(ts.context.astype(str) != "passive") & (ts.perf != 6)]
    srow = st[st.session_id == sid].iloc[0]
    try:
        t0c = pd.to_datetime(srow.session_start_time); clock_h = t0c.hour + t0c.minute / 60
    except Exception:  # noqa: BLE001
        clock_h = np.nan
    a0, a1 = act_all.start_time.min(), act_all.start_time.max()
    eng = dict(active_dur_min=(a1 - a0) / 60, pre_dur_min=(pre.start_time.max() - pre.start_time.min()) / 60,
               post_dur_min=(post.start_time.max() - post.start_time.min()) / 60,
               gap_pre_active_min=(a0 - pre.start_time.max()) / 60, gap_active_post_min=(post.start_time.min() - a1) / 60,
               pre_to_post_min=(post.start_time.min() - pre.start_time.max()) / 60,
               post_from_session_start_min=(post.start_time.min() - ts.start_time.min()) / 60, clock_start_h=clock_h,
               n_rewards_active=int(is_rew(act_all).sum()), reward_rate_active_per_min=float(is_rew(act_all).sum() / max((a1 - a0) / 60, 1e-6)),
               n_rewards_session=int(is_rew(ts_act).sum()),
               n_rewards_whisker_active=int(((act_all.trial_type == "whisker_trial") & (act_all.lick_flag == 1) & (rg == "R+")).sum()),
               n_rewards_auditory_active=int(((act_all.trial_type == "auditory_trial") & (act_all.lick_flag == 1)).sum()))
    rows = []
    for (uset, area), cands in sets.items():
        units = np.asarray(cands)                  # shared tracked units (137b), taken as is
        row0 = dict(base, unit_set=uset, area=area, n_units=len(units), **beh, **eng)
        if len(units) < MIN_UNITS:
            rows.append(dict(row0, skipped_reason="too few tracked units")); continue
        RB = {}
        for e, d in (("passive_pre", pre), ("active", act), ("passive_post", post)):
            t0 = d.start_time.to_numpy()
            RB[e] = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
        # state: mean baseline rate (Hz, -55..-20 ms) over units per epoch (active split in halves)
        bm = {"passive_pre": np.nanmean(RB["passive_pre"][1]), "active_1": np.nanmean(RB["active"][1][~half2]),
              "active_2": np.nanmean(RB["active"][1][half2]), "passive_post": np.nanmean(RB["passive_post"][1])}
        row0.update({f"base_rate_{k}": float(v) for k, v in bm.items()})
        for resp in (RESPONSES if area == "whole_brain" else RESPONSES[:1]):
            row = dict(row0, response=resp)
            if resp == "epochbase":      # 5-35 ms rate minus the unit's mean baseline rate in that epoch
                X = {e: r - np.nanmean(b, 0, keepdims=True) for e, (r, b) in RB.items()}
            elif resp == "trialbase":    # minus the same trial's own baseline rate
                X = {e: r - b for e, (r, b) in RB.items()}
            else:                        # baseline window alone (-55..-20 ms): state, no sensory response
                X = {e: b.copy() for e, (r, b) in RB.items()}
            ok_u = np.all([~np.isnan(v).any(0) for v in X.values()], axis=0)
            P = np.vstack([v[:, ok_u] for v in X.values()]); mu, sd = P.mean(0), P.std(0); keep = sd > 0
            if keep.sum() < MIN_UNITS:
                rows.append(dict(row, skipped_reason="too few units after cleaning")); continue
            Z = {e: (v[:, ok_u][:, keep] - mu[keep]) / sd[keep] for e, v in X.items()}
            E = {e: (v[:, ok_u][:, keep] - (mu[keep] if resp == "baseline" else 0)) / sd[keep] for e, v in X.items()}
            row["n_units"] = int(keep.sum())
            rng = np.random.default_rng(zlib.crc32(f"146|{sid}|{uset}|{area}|{resp}".encode()))
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
            # linear-shift null (whole brain; user 2026-10-05 "should it be a linear shift?"): labels shifted against the active
            # whisker trials (time order) by k = 10-50 % of the trials, non-wrapping, random direction (as the Part I null);
            # keeps the slow drift of both series, so a decoder that learned session time is in the null. Per shift: the passive
            # post - pre change of the standardised readout (whisker, auditory, whisker - auditory).
            if uset == "stable":                     # whole brain and area groups (2026-10-05)
                n = len(lick_w); lo, hi = max(1, int(0.1 * n)), max(1, int(0.5 * n))
                dnull = {kk: [] for kk in ("W", "A", "WA")}
                # every session with >= MIN_CLASS hits and misses gets a null (user 2026-10-05 "use at least 3 hits to include
                # the session"): shifts are drawn from the (lag, direction) pairs whose shifted labels keep >= MIN_CLASS hits
                # and misses (and the anchored trials both classes), so low-hit sessions are not lost to truncation
                def _cut(kk_, dr):
                    return (lick_w[kk_:], Zw[: n - kk_], lick_w[: n - kk_]) if dr else (lick_w[: n - kk_], Zw[kk_:], lick_w[kk_:])
                valid = [(kk_, dr) for kk_ in range(lo, hi + 1) for dr in (0, 1)
                         if min(_cut(kk_, dr)[0].sum(), (~_cut(kk_, dr)[0]).sum()) >= MIN_CLASS
                         and min(_cut(kk_, dr)[2].sum(), (~_cut(kk_, dr)[2]).sum()) >= 1]
                row["shift_n_valid"] = len(valid)
                pick = rng.permutation(len(valid))[:N_SHIFT] if valid else []
                for j in pick:
                    y_s, Zs, anc = _cut(*valid[j])
                    _, o_s, _ = decoder_readout(Zs, y_s, {e: others[e] for e in ("passive_pre", "passive_post")}, C, rng, T,
                                                anchor=anc, n_rep=N_REP_SHIFT)
                    if not len(o_s["passive_pre"]["std"]):
                        continue
                    r_ = {(e, s): float(np.nanmean(o_s[e]["std"][wi[e] if s == "W" else ~wi[e]])) for e in ("passive_pre", "passive_post") for s in "WA"}
                    dW = r_[("passive_post", "W")] - r_[("passive_pre", "W")]; dA = r_[("passive_post", "A")] - r_[("passive_pre", "A")]
                    dnull["W"].append(dW); dnull["A"].append(dA); dnull["WA"].append(dW - dA)
                real = {"W": float(np.nanmean(oth_r["passive_post"]["std"][wi["passive_post"]]) - np.nanmean(oth_r["passive_pre"]["std"][wi["passive_pre"]])),
                        "A": float(np.nanmean(oth_r["passive_post"]["std"][~wi["passive_post"]]) - np.nanmean(oth_r["passive_pre"]["std"][~wi["passive_pre"]]))}
                real["WA"] = real["W"] - real["A"]
                row["shift_n"] = len(dnull["W"])
                for kk, v in dnull.items():
                    v = np.asarray(v, float)
                    row[f"shift_null_d{kk}_mean"] = float(np.nanmean(v)) if len(v) else np.nan
                    row[f"shift_null_d{kk}_sd"] = float(np.nanstd(v)) if len(v) else np.nan
                    row[f"shift_excess_d{kk}"] = real[kk] - row[f"shift_null_d{kk}_mean"]
                    # one-sided percentile: fraction of null changes <= the real change (small = real more negative / miss-ward)
                    row[f"shift_pct_d{kk}"] = float((np.sum(v <= real[kk]) + 1) / (len(v) + 1)) if len(v) else np.nan
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
            # variant y-axis: passive-pre AUDITORY evoked pattern, orthogonalised to the choice axis (2026-10-05)
            a0 = E["passive_pre"][~wi["passive_pre"]].mean(0); a0 = a0 - (a0 @ u1) * u1; u4 = unit(a0)
            conds = {"pre_W": E["passive_pre"][wi["passive_pre"]], "pre_A": E["passive_pre"][~wi["passive_pre"]],
                     "post_W": E["passive_post"][wi["passive_post"]], "post_A": E["passive_post"][~wi["passive_post"]]}
            for h, mW, mA in ((1, ~h2w, ~h2a), (2, h2w, h2a)):
                conds[f"act{h}_hit"] = Ew[mW & lick_w]; conds[f"act{h}_miss"] = Ew[mW & ~lick_w]; conds[f"act{h}_A"] = Ea[mA]
            for cname, M in conds.items():
                m = M.mean(0) if len(M) else np.full(len(u1), np.nan)
                row[f"ss_{cname}_x"], row[f"ss_{cname}_y"], row[f"ss2_{cname}_y"], row[f"ss3_{cname}_y"] = (
                    float(m @ u1), float(m @ u2), float(m @ u3), float(m @ u4))
            rows.append(dict(row, skipped_reason=None))
    return rows


def main():
    os.chdir(OUT.parents[2])
    from axel_bisi_paths import axel_bisi_root
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    # 2026-10-05: the shared Part III tracked units (tracked_units.py / 137b; "same stable units throughout")
    S = importlib.import_module("tracked_units").load_table()
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
