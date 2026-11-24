"""179 -- Clean single pipeline for the Figure 4 hit - miss coding-direction (CD) analysis (user 2026-11-23: "rerun everything on
this pipeline in one clean way"; step 1 decisions: 132 session_trials fixed, AB144 re-included, ">= 3 hits and 3 misses for the
axis"). Replaces, for the figure, 135 (split-half axes: cannot work with 3 trials per class), 168 (time course) and 169 (along /
orthogonal); same conventions as 169, extended to areas, the time course, the stimulus axis and the unit level.

Sessions: hitmiss_session_list, stage SSL_STAGE (learning default; expert -> suffix _expert), R+ / R-. Trials: cd_trials.py
(2026-11-24 review; ephys_utilities keep_active_trials / keep_passive_data / keep_passive_units; passive = every passive
trial without a lick, pre / post split at the session's middle trial; a modality kept with >= 5 passive trials in pre and post;
variants SSL_179_TRIM none / a1 / end10 and SSL_179_PASSIVE mid / blocks -> output suffixes). Units (SSL_179_UNITS): 'stable' (default
2026-11-24) = 137 stable good + MUA (coverage, presence, drift; no rate criterion), 'tracked' = 137b list; every group (whole brain, area group, fine area) needs >= 10 units (no separate
whole-session minimum). Before 2026-11-24: 132.session_trials, >= 20 units whole brain, >= 10 passive trials, fast licks
(< 35 ms) dropped (none existed).
Scaling (harmonised, as 135 SSL_135_HARM / 169): 5-35 ms rate minus the epoch's mean baseline (-55 to -20 ms), / the unit's SD
over the active trials (units with SD <= 1e-6 dropped). Projections per neuron (/ sqrt(N)).
Axis: u = unit(mean hit - mean miss), ALL active whisker trials, >= MIN_CLASS (3) hits and misses (passive trials never build it,
so no split is needed). Shift null: labels shifted 10-50 % against the time-ordered whisker trials (non-wrapping, N_SHIFT
shifts, each keeping >= MIN_CLASS per class) -> u_k; every axis quantity also on the u_k, stored as the mean (_null).
Stimulus axis (reference): s = unit(mean active whisker - mean active auditory), same window; cos(u, s) stored.
Groups: "All units", each area_group and each fine area (area_acronym_custom) with >= MIN_UNITS_AREA (10) units.
Per group x passive stimulus (W, A), with m_e = mean evoked vector in epoch e (pre, post):
  a_pre, a_post, d_along = a_post - a_pre (+ _null)        projection on u
  rho_pre, rho_post   cross-validated |m| (sqrt(m_half1 . m_half2), N_HALF splits)
  cos_pre, cos_post, dtheta = arccos(cos_post) - arccos(cos_pre) in degrees (+ _null)
  d_orth (cross-validated), f_along, g (cross-validated cos(d, m_pre): gain sign)
  s_pre, s_post       projection on the stimulus axis s
  a_/th_{pre,post}_{h1,h2} (+ _null)  projection and angle on the temporal halves of each passive epoch (noise floor:
  within = h2 - h1, across = post_h1 - pre_h2; user 2026-11-24)
  ev_pre, ev_post, ev_abs_change   population mean evoked rate (spikes/s, unscaled) and mean |post - pre| over units
  (the link with the passive response change of panels A-G)
Held-out separation: leave-one-out over active whisker trials (axis rebuilt without the trial), hit - miss mean and d'.
Time course (All units and area_group): 10-ms windows, 5-ms steps, centres -95 ... +195 ms (dead zone -10..+5 ms NaN);
passive pre / post W / A projected on u (and on the u_k: _null mean); active hits / misses leave-one-out.
Unit table (All units): u weight, evoked rates pre / post W / A (spikes/s), keys mouse_id, session_id, electrode_group, cluster_id.
Outputs (this folder, moved to the NAS by sync_results.sh): 179_cd_clean{sfx}.parquet, 179_cd_time{sfx}.parquet,
179_cd_units{sfx}.parquet. Run (haas, repo root): python .../179_cd_clean.py
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
WIN, BASE, DZ = (0.005, 0.035), (-0.055, -0.020), (-0.010, 0.005)
TS = np.round(np.arange(-0.100, 0.1901, 0.005), 3)
TC = TS + 0.005
N_SHIFT, FRAC, N_HALF = 50, (0.10, 0.50), 50
MIN_CLASS = int(os.environ.get("SSL_179_MIN_CLASS", "3"))
MIN_UNITS_AREA = 10                                    # every group incl. whole brain (2026-11-24; was 20 for whole brain)
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
STAGE = os.environ.get("SSL_STAGE", "learning")
CDT = importlib.import_module("cd_trials")
UNITS = os.environ.get("SSL_179_UNITS", "stable")                 # stable (default 2026-11-24) | tracked
SFX = (("" if STAGE == "learning" else f"_{STAGE}") + ("" if MIN_CLASS == 3 else f"_hm{MIN_CLASS}")
       + ("" if CDT.TRIM == "none" else f"_trim-{CDT.TRIM}") + ("" if CDT.PASSIVE_RULE == "mid" else f"_passive-{CDT.PASSIVE_RULE}")
       + ("" if UNITS == "stable" else f"_units-{UNITS}"))
OUT_M, OUT_T, OUT_U = (EA / f"179_cd_{k}{SFX}.parquet" for k in ("clean", "time", "units"))
FINE = "area_acronym_custom"
SET_IDS, UNIT_META = {}, {}


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    warnings.filterwarnings("ignore")


def unit(v):
    n = np.linalg.norm(v, axis=0)
    return v / np.where(n > 0, n, np.nan)


def shifted_labels(y, widx):
    """non-wrapping linear shifts of the time-ordered labels (10-50 %), both directions -> list of (trial idx, labels)"""
    n = len(widx)
    ks = np.unique(np.linspace(FRAC[0] * n, FRAC[1] * n, N_SHIFT // 2).astype(int))
    out = []
    for k in np.r_[ks, -ks]:
        if k == 0:
            continue
        i = np.arange(max(0, -k), min(n, n - k))
        ys, xi = y[i], widx[i + k]
        if min(ys.sum(), (~ys).sum()) >= MIN_CLASS:
            out.append((xi, ys))
    return out


def loo_proj(Z, hit, miss, Tr=None):
    """leave-one-out: each active whisker trial projected on the axis built without it (window Z; time course Tr if given)"""
    sh, sm = Z[hit].sum(0), Z[miss].sum(0)
    nh, nm = len(hit), len(miss)
    p, tc = {"hit": [], "miss": []}, {"hit": [], "miss": []}
    for cls, idx in (("hit", hit), ("miss", miss)):
        for i in idx:
            if cls == "hit":
                v = (sh - Z[i]) / (nh - 1) - sm / nm
            else:
                v = sh / nh - (sm - Z[i]) / (nm - 1)
            uu = unit(v)
            p[cls].append(float(Z[i] @ uu))
            if Tr is not None:
                tc[cls].append(Tr[:, i] @ uu)
    return p, tc


def process(args):
    sid, subject, rg = args
    _init()
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    base = dict(session_id=sid, mouse_id=subject, reward_group=rg)
    trs = CDT.session_trials(sid, tt, T)                            # 2026-11-24: ephys_utilities selection (cd_trials.py)
    if trs is None:
        return [dict(base, skipped_reason="no passive pre + post (>= 5 trials of a modality) or no active")], [], []
    pok = trs["passive_ok"]
    act = trs["active"].sort_values("start_time").reset_index(drop=True)
    beh = {s_: float((act[act.trial_type == t_].lick_flag == 1).mean()) for s_, t_ in (("W", "whisker_trial"), ("A", "auditory_trial"))}
    licked = (act.lick_flag == 1).to_numpy()                         # fast-lick rule removed 2026-11-24 (it removed 0 trials)
    allu = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), SET_IDS.get(sid, np.array([], dtype=np.int64)))
    meta = UNIT_META.get(sid)
    if meta is not None:                                               # ephys_utilities keep_passive_units (AB155 imec4)
        kp = CDT._du().keep_passive_units(meta.assign(session_id=sid))
        allu = np.intersect1d(allu, kp.cluster_id.to_numpy())
    if len(allu) < MIN_UNITS_AREA:                                     # 2026-11-24: no whole-session filter, area-level only
        return [dict(base, skipped_reason=f"{len(allu)} tracked units < {MIN_UNITS_AREA}")], [], []
    isw = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in ("passive_pre", "active", "passive_post")}
    widx = np.where(isw["active"])[0]
    y = licked[widx]
    if min(y.sum(), (~y).sum()) < MIN_CLASS:
        return [dict(base, n_hit=int(y.sum()), n_miss=int((~y).sum()),
                     skipped_reason=f"< {MIN_CLASS} hits or misses ({int(y.sum())}/{int((~y).sum())})")], [], []
    spikes = T.load_session_unit_spikes(root, sid)
    X, TR = {}, {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy()
        one = np.ones(len(t0), bool)
        r, b = T.sliding_bin_population_matrices(spikes, allu, t0, one, [WIN, BASE], dead_zone=DZ)
        bm = np.nanmean(b, 0, keepdims=True)
        X[e] = r - bm
        TR[e] = np.stack(T.sliding_bin_population_matrices(spikes, allu, t0, one, [(s, s + 0.010) for s in TS],
                                                           dead_zone=DZ), 0) - bm[None]
    hit, miss = widx[y], widx[~y]
    shifts = shifted_labels(y, widx)
    sl = (labels[(labels.session_id == sid) & labels.cluster_id.isin(allu)].drop_duplicates("cluster_id")
          .set_index("cluster_id").reindex(allu))
    groups = [("whole_brain", "All units", np.ones(len(allu), bool))]
    for col in ("area_group", FINE):
        for a in sorted(sl[col].dropna().unique()):
            groups.append((col, a, (sl[col] == a).to_numpy()))
    rng = np.random.default_rng(zlib.crc32(f"{sid}|179".encode()))
    rng_h = np.random.default_rng(zlib.crc32(f"{sid}|179|halves".encode()))
    rows, trows, urows = [], [], []
    for col, area, m in groups:
        sd = X["active"][:, m].std(0)
        ok = sd > 1e-6
        cols = np.where(m)[0][ok]
        N = len(cols)
        if N < MIN_UNITS_AREA:                                         # same threshold for whole brain and areas
            continue
        s_ = sd[ok]
        sN = np.sqrt(N)
        Z = X["active"][:, cols] / s_                                  # centring irrelevant for class differences
        E = {e: X[e][:, cols] / s_ for e in X}
        u = unit(Z[hit].mean(0) - Z[miss].mean(0))
        us = [unit(Z[xi[ys]].mean(0) - Z[xi[~ys]].mean(0)) for xi, ys in shifts]
        U = np.stack([u] + us, 1)                                      # units x (1 + shifts)
        sax = unit(Z[isw["active"]].mean(0) - Z[~isw["active"]].mean(0))
        p, _ = loo_proj(Z, hit, miss)
        ph, pm = np.array(p["hit"]) / sN, np.array(p["miss"]) / sN
        spool = np.sqrt(0.5 * (ph.var(ddof=1) + pm.var(ddof=1)))
        g0 = dict(base, area_col=col, area=area, n_units=N, n_hit=len(hit), n_miss=len(miss), n_shifts=len(us),
                  plick_W=beh["W"], plick_A=beh["A"], cos_u_s=float(u @ sax), sep_loo=float(ph.mean() - pm.mean()),
                  dprime_loo=float((ph.mean() - pm.mean()) / spool) if spool > 0 else np.nan, skipped_reason=None)
        for s_k, sel in (("W", True), ("A", False)):
            if not pok[s_k]:                                           # < 5 passive trials of this modality in pre or post
                continue
            P = {e: E[e][isw[e] == sel] for e in ("passive_pre", "passive_post")}
            raw = {e: X[e][isw[e] == sel][:, cols] for e in P}
            mm = {e: v.mean(0) for e, v in P.items()}
            a = {e: mm[e] @ U / sN for e in mm}
            d = mm["passive_post"] - mm["passive_pre"]
            dU = d @ U / sN
            rho2 = {e: [] for e in mm}
            dorth2, gn, gd1, gd2 = [], [], [], []
            for _ in range(N_HALF):
                h = {}
                for e, v in P.items():
                    pi = rng.permutation(len(v))
                    h[e] = (v[pi[: len(v) // 2]].mean(0), v[pi[len(v) // 2:]].mean(0))
                    rho2[e].append(h[e][0] @ h[e][1] / N)
                dA, dB = h["passive_post"][0] - h["passive_pre"][0], h["passive_post"][1] - h["passive_pre"][1]
                dorth2.append((dA @ dB - (dA @ U) * (dB @ U)) / N)
                gn.append(dA @ h["passive_pre"][1] / N)
                gd1.append(dA @ dB / N)
                gd2.append(h["passive_pre"][0] @ h["passive_pre"][1] / N)
            rho = {e: np.sqrt(max(np.mean(rho2[e]), 0)) for e in mm}
            cs = {e: a[e] / rho[e] if rho[e] > 0 else a[e] * np.nan for e in mm}
            th = {e: np.degrees(np.arccos(np.clip(cs[e], -1, 1))) for e in mm}
            dorth = np.sqrt(np.clip(np.mean(dorth2, 0), 0, None))
            f = dU ** 2 / (dU ** 2 + dorth ** 2)
            gden = np.sqrt(max(np.mean(gd1), 0) * max(np.mean(gd2), 0))
            r = dict(g0, stim=s_k, n_pre=len(P["passive_pre"]), n_post=len(P["passive_post"]),
                     g=float(np.mean(gn) / gden) if gden > 0 else np.nan, rho_pre=rho["passive_pre"], rho_post=rho["passive_post"],
                     s_pre=float(mm["passive_pre"] @ sax / sN), s_post=float(mm["passive_post"] @ sax / sN),
                     ev_pre=float(raw["passive_pre"].mean()), ev_post=float(raw["passive_post"].mean()),
                     ev_abs_change=float(np.abs(raw["passive_post"].mean(0) - raw["passive_pre"].mean(0)).mean()))
            # temporal halves of each passive epoch (user 2026-11-24, noise floor as Kingsbury et al. 2025: within = h2 - h1 of
            # one epoch, across = post h1 - pre h2, adjacent in time, matched trial counts); own rng -> other values unchanged
            for e, k in (("passive_pre", "pre"), ("passive_post", "post")):
                order = np.argsort(trs[e].start_time.to_numpy()[isw[e] == sel], kind="stable")
                v = P[e][order]
                for hk, hv in (("h1", v[: len(v) // 2]), ("h2", v[len(v) // 2:])):
                    mh = hv.mean(0)
                    ah = mh @ U / sN
                    r2 = []
                    for _ in range(N_HALF):
                        pi = rng_h.permutation(len(hv))
                        r2.append(hv[pi[: len(hv) // 2]].mean(0) @ hv[pi[len(hv) // 2:]].mean(0) / N)
                    rh = np.sqrt(max(np.mean(r2), 0))
                    thh = np.degrees(np.arccos(np.clip(ah / rh, -1, 1))) if rh > 0 else ah * np.nan
                    for key, val in ((f"a_{k}_{hk}", ah), (f"th_{k}_{hk}", thh)):
                        r[key] = float(val[0])
                        r[f"{key}_null"] = float(np.nanmean(val[1:])) if len(us) else np.nan
                    r[f"n_{k}_{hk}"] = len(hv)
            for key, v in (("a_pre", a["passive_pre"]), ("a_post", a["passive_post"]), ("d_along", dU),
                           ("cos_pre", cs["passive_pre"]), ("cos_post", cs["passive_post"]),
                           ("dtheta", th["passive_post"] - th["passive_pre"]), ("d_orth", dorth), ("f_along", f)):
                r[key] = float(v[0])
                r[f"{key}_null"] = float(np.nanmean(v[1:])) if len(us) else np.nan
            rows.append(r)
        if col in ("whole_brain", "area_group"):
            Tr = {e: TR[e][:, :, cols] / s_ for e in TR}               # time x trials x units
            n_tr0 = len(trows)
            for e, k in (("passive_pre", "pre"), ("passive_post", "post")):
                for s_k, sel in (("W", True), ("A", False)):
                    if not pok[s_k]:
                        continue
                    tm = np.nanmean(Tr[e][:, isw[e] == sel], 1) @ U / sN        # time x (1 + shifts)
                    trows.append(pd.DataFrame(dict(t=TC, value=tm[:, 0], null=np.nanmean(tm[:, 1:], 1) if len(us) else np.nan,
                                                   cls=f"{k}_{s_k}")))
            _, tc = loo_proj(Z, hit, miss, Tr["active"])
            for cls in ("hit", "miss"):
                trows.append(pd.DataFrame(dict(t=TC, value=np.nanmean(np.stack(tc[cls]), 0) / sN, null=np.nan, cls=cls)))
            for df in trows[n_tr0:]:                                   # 4-6 frames (a modality may be dropped)
                for k_, v_ in dict(base, area_col=col, area=area, n_units=N).items():
                    df[k_] = v_
        if col == "whole_brain":
            meta = UNIT_META.get(sid)
            cid = allu[cols]
            U_ = pd.DataFrame(dict(cluster_id=cid, u_weight=u, sd_active=s_,
                                   **{f"ev_{k}_{s_k}": X[e][isw[e] == sel][:, cols].mean(0)
                                      for e, k in (("passive_pre", "pre"), ("passive_post", "post"))
                                      for s_k, sel in (("W", True), ("A", False))},
                                   ev_active_hit=X["active"][hit][:, cols].mean(0), ev_active_miss=X["active"][miss][:, cols].mean(0)))
            U_ = U_.assign(**base).merge(meta, on="cluster_id", how="left") if meta is not None else U_.assign(**base)
            urows.append(U_)
    print(sid, rg, len(allu), "units", flush=True)
    return rows, trows, urows


def main():
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    TU = importlib.import_module("tracked_units")
    root = resolve_dataset_dir("ssl_ephys")
    sess = T.hitmiss_session_list(pd.read_parquet(root / "metadata" / "sessions.parquet"))
    sess = sess[(sess.day_stage == STAGE) & sess.reward_group.isin(["R+", "R-"])]
    if UNITS == "tracked":                                             # legacy: 137b tracked list (stable + >= 0.5 Hz per segment)
        SET_IDS.update(TU.load("stable"))
        tab = TU.load_table()
    else:                                                              # 2026-11-24: 137 stable (good + mua), no rate criterion
        tab = pd.read_parquet(TU.table_path().with_name("137_stable_units.parquet"))
        tab = tab[tab.stable & tab.quality_label.isin(["good", "mua"])]
        SET_IDS.update({s: np.sort(g.cluster_id.to_numpy().astype(np.int64)) for s, g in tab.groupby("session_id")})
    lab = pd.read_parquet(T.AREA_LABELS_PATH, columns=["session_id", "cluster_id", "area_group", FINE])
    tab = tab[["session_id", "cluster_id", "electrode_group"]].merge(lab, on=["session_id", "cluster_id"], how="left")
    UNIT_META.update({s: g.drop(columns="session_id") for s, g in tab.groupby("session_id")})
    args = [(r.session_id, r.subject_id, r.reward_group) for r in sess.itertuples()]
    print(f"[179] {len(args)} sessions, MIN_CLASS {MIN_CLASS}, stage {STAGE}", flush=True)
    rows, trows, urows, t0 = [], [], [], time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, a_): a_ for a_ in args}
        for i, f_ in enumerate(as_completed(futs), 1):
            a_ = futs[f_]
            try:
                r, tr, ur = f_.result()
                rows += r; trows += tr; urows += ur
            except Exception as e:  # noqa: BLE001
                rows.append(dict(session_id=a_[0], mouse_id=a_[1], reward_group=a_[2], skipped_reason=f"error: {e!r}"[:300]))
                print("FAILED", a_[0], repr(e)[:300], flush=True)
            print(f"[179] [{i}/{len(args)}] {a_[0]} -- {time.time() - t0:.0f}s", flush=True)
    for df, p in ((pd.DataFrame(rows), OUT_M), (pd.concat(trows, ignore_index=True) if trows else pd.DataFrame(), OUT_T),
                  (pd.concat(urows, ignore_index=True) if urows else pd.DataFrame(), OUT_U)):
        tmp = p.with_suffix(".partial.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
    print("ALL DONE", OUT_M, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
