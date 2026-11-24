"""009 -- state fingerprints, cross-method / cross-view / cross-session mapping and meta-states (haas).

Fingerprint (v3) = mean of 15 features over the frames of a state, on the side 200 Hz grid (cameras synchronous), in
PHYSICAL units (deg, mm, mm/s, P(tongue out), video motion energy ME in grey levels / frame) and, for display and
clustering, z-scored with POOLED statistics over all frames of the 8 sessions (tables/009_feature_pooled_scaling.csv;
no per-session z-scoring: a state must mean the same in every animal):
  top: whisker set-point, whisking amplitude, nose_tip displacement from rest and speed, ME of snout, whisker pad, whiskers;
  side: jaw opening (from rest), jaw angle, tongue out, nose_tip displacement and speed, ME of snout, mouth, cheek.
States used (>= 0.5 % of the session) by fewer than 3 mice are flagged (shared = False) and left out of the meta-state
clustering and the cross-animal comparisons; cross-mouse consistency = correlation of each mouse's fingerprint with the
mean of the other mice (009_fingerprint_consistency.csv).
- pooled fingerprints: per method x view x state (kpms seed 0, AR-HMM, lightning-action; all 8 sessions);
- session fingerprints: per method x view x session x state (>= 0.5 % of the session's frames and >= 250 frames).
Meta-states: Ward clustering of the session fingerprints (shared states only) of all methods and views (one unit per session-state, so
session-specific states are mapped by what they look like); K from 4-12 by silhouette (ties -> smaller K). Every
session-state takes its cluster; every frame of every method / view / session is relabelled with it (states below the
unit thresholds -> nearest meta-state centroid by fingerprint, else -1).
Contingency: P(state B | state A) over synchronous frames pooled across sessions, top 10 states of each; Hungarian
matching (scipy linear_sum_assignment) on the joint probabilities.
Names: features with |z| >= 0.4 (at most 2, largest |z| first) mapped to words; none -> "neutral"; all ME <= -0.4 and
no positive feature -> "still".
Output: tables/009_*.csv, cache/009_meta.npz, provenance/009.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import bm_states as BS  # noqa: E402
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402

FEATS = ["whisker_setpoint", "whisking_amp", "top_nose_disp", "top_nose_speed", "me_top_snout", "me_top_whisker_pad",
         "me_top_whiskers", "jaw_open", "jaw_angle", "tongue_out", "nose_disp", "nose_speed", "me_side_snout", "me_side_mouth",
         "me_side_cheek"]
UNITS = {"whisker_setpoint": "deg", "whisking_amp": "deg", "top_nose_disp": "mm", "top_nose_speed": "mm/s", "jaw_open": "mm",
         "jaw_angle": "deg", "tongue_out": "P", "nose_disp": "mm", "nose_speed": "mm/s"}  # me_*: grey levels / frame
WORDS = {"whisker_setpoint": ("set-point high", "set-point low"), "whisking_amp": ("whisk", "no whisk"),
         "top_nose_disp": ("nose displaced (top)", "nose at rest (top)"), "top_nose_speed": ("nose moving", "nose still"),
         "me_top_snout": ("snout motion", "snout quiet"), "me_top_whisker_pad": ("whisker-pad motion", "pad quiet"),
         "me_top_whiskers": ("whisker motion", "whiskers quiet"), "jaw_open": ("jaw open", "jaw closed"),
         "jaw_angle": ("jaw angle high", "jaw angle low"),
         "tongue_out": ("tongue out", "tongue in"), "nose_disp": ("nose displaced (side)", "nose at rest (side)"),
         "nose_speed": ("nose moving", "nose still"), "me_side_snout": ("snout motion", "snout quiet"),
         "me_side_mouth": ("mouth motion", "mouth quiet"), "me_side_cheek": ("cheek motion", "cheek quiet")}
MIN_MICE = 3  # states used (>= 0.5 % of the session) by fewer mice are flagged and left out of cross-animal comparisons
ROWS = [(m, v) for m in BS.METHODS for v in L.VIEWS]
FULL = BS.SET == "full"
PRE = "full_009_" if FULL else "009_"
CDIR = L.CACHE / "full" if FULL else L.CACHE
LA_IDX = list(range(1, len(BS.LA_CLASSES)))
if FULL:  # no video pass in the full run: fingerprints from the DLC features only
    FEATS = [f for f in FEATS if not f.startswith("me_")]
    ROWS = [r for r in ROWS if r != ("lightning_action", "top")]  # full run: top lightning-action = final model only (no held-out predictions)
NTOP = 10
THR = 0.4


def name_of(fp):
    fp = np.asarray(fp, float)
    me = [i for i, f in enumerate(FEATS) if f.startswith("me_")]
    if np.all(fp[me] <= -THR) and not np.any(fp >= THR):
        return "still"
    order = [i for i in np.argsort(-np.abs(fp)) if abs(fp[i]) >= THR]
    words = []
    for i in order:
        w = WORDS[FEATS[i]][0 if fp[i] > 0 else 1]
        if w not in words:
            words.append(w)
        if len(words) == 2:
            break
    return " + ".join(words) if words else "neutral"


def features(sid):
    sg = BS.session_grid(sid)
    t = sg["t"]
    out = {}
    for view in L.VIEWS:
        f = np.load(BS.INP / f"{sid}_{view}_features.npz")
        X = pd.DataFrame(f["X"], columns=[str(n) for n in f["names"]])
        for c in X.columns:
            out[c] = dh.to_timestamps(X[c].to_numpy(float), f["t"], t, max_gap_s=0.0075)
        if FULL:
            continue
        v = np.load(L.RES / "kuma" / "video" / f"{sid}_{view}.npz")
        for r in v["regions"]:
            out[f"me_{view}_{r}"] = dh.to_timestamps(v[f"me_{r}"].astype(float), v["t"], t, max_gap_s=0.0075)
    return pd.DataFrame({k: out[k] for k in FEATS}).astype(np.float32)  # v3: physical units, no per-session z-scoring


def main():
    sel = pd.read_csv(L.TABLES / ("full_sessions.csv" if FULL else "001_selected_sessions.csv"))
    if FULL:  # 100 Hz videos (AB102, AB104, AB107): models / velocities are 200 Hz -> left out (change-log 2026-11-20)
        sel = sel[sel.rate_side > 150]
    sids = list(sel.session_id)
    coh = dict(zip(sel.session_id, sel.reward_group))
    Fp = {sid: features(sid) for sid in sids}  # physical units
    allp = pd.concat(Fp.values(), ignore_index=True)
    mu, sd = allp.mean(), allp.std()  # pooled statistics (all frames of the 8 sessions)
    Fz = {sid: (F - mu) / sd for sid, F in Fp.items()}
    pd.DataFrame({"pooled_mean": mu, "pooled_sd": sd}).to_csv(L.TABLES / f"{PRE}feature_pooled_scaling.csv")
    Z = {(sid, m, v): BS.load_states(sid, m, v).astype(np.int16) for sid in sids for m, v in ROWS}
    # pooled and session fingerprints
    pooled, sess = [], []
    for m, v in ROWS:
        allz = np.concatenate([Z[(s, m, v)] for s in sids])
        allF = pd.concat([Fz[s] for s in sids], ignore_index=True)
        cnt = np.bincount(allz[allz >= 0])
        for s in np.flatnonzero(cnt / cnt.sum() > 0.005):
            fp = allF[allz == s].mean()
            pooled.append(dict(method=m, view=v, state=int(s), usage=cnt[s] / cnt.sum(), name=name_of(fp), **fp.to_dict()))
        for sid in sids:
            z = Z[(sid, m, v)]
            c = np.bincount(z[z >= 0])
            for s in np.flatnonzero((c / c.sum() >= 0.005) & (c >= 250)):
                fp = Fz[sid][z == s].mean()
                fphys = Fp[sid][z == s].mean()
                sess.append(dict(method=m, view=v, session_id=sid, cohort=coh[sid], state=int(s), n_frames=int(c[s]),
                                 usage=c[s] / c.sum(), **fp.to_dict(), **{f"phys_{k}": x for k, x in fphys.items()}))
    P, S = pd.DataFrame(pooled), pd.DataFrame(sess)
    nm_ = S.groupby(["method", "view", "state"]).session_id.nunique().rename("n_mice").reset_index()
    P = P.merge(nm_, on=["method", "view", "state"], how="left").fillna({"n_mice": 0})
    S = S.merge(nm_, on=["method", "view", "state"], how="left")
    P["shared"], S["shared"] = P.n_mice >= MIN_MICE, S.n_mice >= MIN_MICE
    # cross-mouse consistency of the physical fingerprint: per mouse vs the mean of the other mice (z on pooled stats)
    cons = []
    for (m, v, st), g in S[S.shared].groupby(["method", "view", "state"]):
        for _, q in g.iterrows():
            o = g[g.session_id != q.session_id][FEATS].mean().to_numpy(float)
            x = q[FEATS].to_numpy(float)
            ok = np.isfinite(o) & np.isfinite(x)
            cons.append(dict(method=m, view=v, state=st, session_id=q.session_id, r_vs_other_mice=float(np.corrcoef(x[ok], o[ok])[0, 1])))
    pd.DataFrame(cons).to_csv(L.TABLES / f"{PRE}fingerprint_consistency.csv", index=False)
    X = S.loc[S.shared, FEATS].fillna(0).to_numpy()  # meta-states from states shared by >= 3 mice
    lk = linkage(X, "ward")
    sil = {k: silhouette_score(X, fcluster(lk, k, "maxclust")) for k in range(4, 13)}
    K = max(sil, key=lambda k: (round(sil[k], 2), -k))
    lab = fcluster(lk, K, "maxclust") - 1
    # order meta-states by total frames
    w = np.bincount(lab, weights=S.loc[S.shared, "n_frames"], minlength=K)
    order = np.argsort(-w)
    remap = np.empty(K, int)
    remap[order] = np.arange(K)
    Ssh = S[S.shared]
    cent = np.array([np.average(X[remap[lab] == k], 0, weights=Ssh.n_frames[remap[lab] == k]) for k in range(K)])
    S["meta"] = -1
    S.loc[S.shared, "meta"] = remap[lab]
    nf = ~S.shared  # flagged states: nearest meta-state centroid, kept out of the clustering
    if nf.any():
        S.loc[nf, "meta"] = np.argmin(((S.loc[nf, FEATS].fillna(0).to_numpy()[:, None] - cent[None]) ** 2).sum(-1), 1)
    meta_names = []
    for k in range(K):
        nm = name_of(cent[k])
        meta_names.append(nm if nm not in meta_names else f"{nm} ({k})")
    M = pd.DataFrame(cent, columns=FEATS)
    phys = pd.DataFrame([np.average(S.loc[S.shared & (S.meta == k), [f"phys_{f}" for f in FEATS]].to_numpy(float), 0,
                                    weights=S.loc[S.shared & (S.meta == k), "n_frames"]) for k in range(K)],
                        columns=[f"phys_{f}" for f in FEATS])
    M = pd.concat([M, phys], axis=1)
    M.insert(0, "meta", range(K))
    M.insert(1, "name", [f"M{k} {meta_names[k]}" for k in range(K)])
    M.insert(2, "frac_frames", w[order] / w.sum())
    # pooled state -> meta: frame-weighted majority of its session units, else nearest centroid
    def pooled_meta(r):
        u = S[(S.method == r.method) & (S.view == r.view) & (S.state == r.state)]
        if len(u):
            return int(u.groupby("meta").n_frames.sum().idxmax())
        return int(np.argmin(((cent - r[FEATS].fillna(0).to_numpy(float)) ** 2).sum(1)))
    P["meta"] = [pooled_meta(r) for _, r in P.iterrows()]
    P["meta_name"] = [M.name[k] for k in P.meta]
    # frame relabelling
    meta_seq = {}
    for (sid, m, v), z in Z.items():
        u = S[(S.session_id == sid) & (S.method == m) & (S.view == v)].set_index("state").meta.to_dict()
        pm = P[(P.method == m) & (P.view == v)].set_index("state").meta.to_dict()
        lut = np.full(max(z.max() + 1, 1), -1)
        for s in range(len(lut)):
            lut[s] = u.get(s, pm.get(s, -1))
        meta_seq[f"{sid}|{m}|{v}"] = np.where(z >= 0, lut[np.clip(z, 0, None)], -1)
    np.savez_compressed(CDIR / "009_meta.npz", **meta_seq)
    # session specificity -> which meta-state do session-specific states belong to
    spec_rows = []
    for (m, v), g in S.groupby(["method", "view"]):
        for s, gs in g.groupby("state"):
            share = gs.usage.max() / gs.usage.sum() if len(gs) > 1 else 1.0
            spec_rows.append(dict(method=m, view=v, state=s, n_sessions=len(gs), dominant_share=share,
                                  sessions=";".join(gs.session_id.str[:5]), metas=";".join(map(str, sorted(set(gs.meta))))))
    SP = pd.DataFrame(spec_rows)
    # contingency + Hungarian
    cont_rows, hung_rows = [], []
    for (a, b) in itertools.combinations(ROWS, 2):
        za = np.concatenate([Z[(s, *a)] for s in sids])
        zb = np.concatenate([Z[(s, *b)] for s in sids])
        ok = (za >= 0) & (zb >= 0)
        ta = list(np.argsort(-np.bincount(za[ok]))[:NTOP]) if a[0] != "lightning_action" else LA_IDX
        tb = list(np.argsort(-np.bincount(zb[ok]))[:NTOP]) if b[0] != "lightning_action" else LA_IDX
        nb = int(zb[ok].max()) + 1
        full = np.bincount(za[ok] * nb + zb[ok], minlength=(int(za[ok].max()) + 1) * nb).reshape(-1, nb)
        J = full[np.ix_(ta, tb)].astype(float)
        Pc = J / np.maximum(full[ta].sum(1)[:, None], 1)
        r, c = linear_sum_assignment(-J / J.sum())
        for i, j in zip(r, c):
            hung_rows.append(dict(A=f"{a[0]} {a[1]}", B=f"{b[0]} {b[1]}", state_A=int(ta[i]), state_B=int(tb[j]),
                                  p_B_given_A=Pc[i, j], joint=J[i, j] / ok.sum()))
        for i, sa in enumerate(ta):
            for j, sb in enumerate(tb):
                cont_rows.append(dict(A=f"{a[0]} {a[1]}", B=f"{b[0]} {b[1]}", state_A=int(sa), state_B=int(sb), p_B_given_A=Pc[i, j]))
    # agreement in meta-state space vs original states
    ag = []
    for a, b in itertools.combinations(ROWS, 2):
        for sid in sids:
            act = BS.session_grid(sid)["epoch"] != "other"
            x, y = Z[(sid, *a)], Z[(sid, *b)]
            mx, my = meta_seq[f"{sid}|{a[0]}|{a[1]}"], meta_seq[f"{sid}|{b[0]}|{b[1]}"]
            ok = act & (x >= 0) & (y >= 0) & (mx >= 0) & (my >= 0)
            ag.append(dict(A=f"{a[0]} {a[1]}", B=f"{b[0]} {b[1]}", session_id=sid, ami_states=ami(x[ok][::20], y[ok][::20]),
                           ami_meta=ami(mx[ok][::20], my[ok][::20]), frac_same_meta=float(np.mean(mx[ok] == my[ok]))))
    AG = pd.DataFrame(ag)
    P.to_csv(L.TABLES / f"{PRE}fingerprints_pooled.csv", index=False)
    S.to_csv(L.TABLES / f"{PRE}fingerprints_session.csv", index=False)
    M.to_csv(L.TABLES / f"{PRE}metastates.csv", index=False)
    SP.to_csv(L.TABLES / f"{PRE}state_sessions.csv", index=False)
    pd.DataFrame(cont_rows).to_csv(L.TABLES / f"{PRE}contingency.csv", index=False)
    pd.DataFrame(hung_rows).to_csv(L.TABLES / f"{PRE}hungarian.csv", index=False)
    AG.to_csv(L.TABLES / f"{PRE}agreement_meta.csv", index=False)
    json.dump(dict(script="009_fingerprints_metastates.py", features=FEATS, K=int(K), silhouette={int(k): float(v) for k, v in sil.items()},
                   name_threshold_z=THR, unit_thresholds="usage >= 0.5 % and >= 250 frames per session", versions=L.versions()),
              open(L.RES / "provenance" / ("009_full.json" if FULL else "009.json"), "w"), indent=1)
    print("K", K, {k: round(v, 3) for k, v in sil.items()})
    print(M[["name", "frac_frames"]].to_string())
    print(P.groupby(["method", "view"]).apply(lambda g: g.sort_values("usage", ascending=False).head(10)[["state", "usage", "name", "meta"]].to_string()).to_string()[:6000])
    print(AG.groupby(["A", "B"])[["ami_states", "ami_meta", "frac_same_meta"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
