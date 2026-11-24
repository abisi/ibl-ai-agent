"""035 (haas) -- evaluation of the action-segmentation comparison (kuma/034) and of the representations (keypoints,
Facemap, BEAST, existing state sequences) against signals the models never saw (2026-11-22, pilot, side view).

seg   per held-out session x model (head_input from kuma/full/segcompare), on the side 200 Hz grid:
      lick   piezo licks (NWB): AUC of P(lick) for frames <= 25 ms from a piezo lick vs frames > 0.5 s from any lick;
             recall = share of piezo licks with a predicted lick frame within +-50 ms; precision = share of predicted
             lick bouts (>= 2 frames) with a piezo lick within the bout +-50 ms.
      whisk  top-camera whisking amplitude (not an input of any side model; the heuristic whisking LABEL is derived from
             it, so this checks how well side-only models reproduce it on an unseen mouse): AUC of P(whisking) for
             frames above the session's 75th vs below its 25th percentile, lick-free frames only.
      still  AUC of P(still): frames with top nose speed AND whisking amplitude below the 25th percentile and > 0.5 s from
             licks vs frames with either above the 75th percentile.
      jaw    trials with a jaw onset (dlc_jaw_onset_times; derived from the side jaw keypoint, so independent of the
             video-only inputs, not of kp): first frame predicted jaw or lick after the stimulus vs the jaw onset
             (median |difference|, share within 50 ms).
      plus agreement with the heuristic labels on labelled frames (circular).
repr  linear read-out of the same independent targets from each representation (+-20 ms context: lags -4..4 frames,
      step 2), leave-one-mouse-out, 40 k frames per session: lick-near vs far (logistic, AUC), whisking amplitude and
      top nose speed (ridge, R2). Representations: kp, fm (Facemap motion SVD 50), beast (PCA 64), kp_fm, kp_beast,
      AR-HMM side states (one-hot), full-run lightning-action side probabilities (in-sample: the pilot sessions were in
      its training set).
Outputs: tables/035_seg_per_session.csv, 035_seg_summary.csv, 035_repr_per_session.csv, 035_repr_summary.csv,
         figures/035_segment_compare.png
Run (haas, repo root): python projects/ssl-behaviour-motifs/035_segment_eval.py seg|repr|fig
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import bm_lib as L  # noqa: E402
import bm_states as BS  # noqa: E402

PILOT = ["AB117_20240723_125437", "AB121_20240813_125401", "AB137_20241116_134442", "AB139_20241119_115957",
         "MH026_20250422_181339", "MH031_20250507_104425", "MH037_20250524_143522", "MH039_20250525_112720"]
CLASSES = ["background", "still", "whisking", "jaw", "lick"]
KR = L.RES / "kuma"
SEG = KR / "full" / "segcompare"
EXT = KR / "ext_data"
INP = L.CACHE / "full" / "inputs"
FPS = 200.0


def targets(sid):
    S = np.load(INP / f"{sid}_session.npz", allow_pickle=True)
    t = S["t"]
    top = np.load(INP / f"{sid}_top_features.npz")
    names = list(top["names"])
    j = np.clip(np.searchsorted(top["t"], t), 0, len(top["t"]) - 1)
    wamp = top["X"][j, names.index("whisking_amp")].astype(float)
    nspd = top["X"][j, names.index("top_nose_speed")].astype(float)
    licks = np.asarray(S["licks"], float)
    k = np.clip(np.searchsorted(licks, t), 1, len(licks) - 1)
    dl = np.minimum(np.abs(t - licks[k - 1]), np.abs(licks[k] - t))
    return dict(t=t, labels=S["labels"], licks=licks, dlick=dl, wamp=wamp, nspd=nspd)


def auc(score, pos, neg):
    m = (pos | neg) & np.isfinite(score)
    if pos[m].sum() < 20 or neg[m].sum() < 20:
        return np.nan
    return roc_auc_score(pos[m], score[m])


def bouts(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def seg_metrics(sid, P, T, jo):
    t = T["t"]
    n = min(len(P), len(t))
    P, t = P[:n], t[:n]
    dl, wamp, nspd = T["dlick"][:n], T["wamp"][:n], T["nspd"][:n]
    pred = np.nanargmax(np.nan_to_num(P, nan=-1), 1)
    out = {}
    near, far = dl <= 0.025, dl > 0.5
    out["lick_auc"] = auc(P[:, 4], near, far)
    lf = np.flatnonzero(pred == 4)
    tl = t[lf]
    lk = T["licks"][(T["licks"] > t[0]) & (T["licks"] < t[-1])]
    if len(tl) and len(lk):
        i = np.clip(np.searchsorted(tl, lk), 1, len(tl) - 1)
        out["lick_recall_50ms"] = float(np.mean(np.minimum(np.abs(lk - tl[i - 1]), np.abs(tl[i] - lk)) <= 0.05))
    else:
        out["lick_recall_50ms"] = 0.0
    on, off = bouts(pred == 4)
    keep = (off - on) >= 2
    on, off = on[keep], off[keep]
    if len(on):
        a = np.searchsorted(lk, t[on] - 0.05)
        b = np.searchsorted(lk, t[np.minimum(off, n - 1)] + 0.05)
        out["lick_bout_precision"] = float(np.mean(b > a))
    else:
        out["lick_bout_precision"] = np.nan
    lo_w, hi_w = np.nanpercentile(wamp, [25, 75])
    lo_n, hi_n = np.nanpercentile(nspd, [25, 75])
    out["whisk_auc"] = auc(P[:, 2], (wamp > hi_w) & far, (wamp < lo_w) & far)
    still_pos = (wamp < lo_w) & (nspd < lo_n) & far
    still_neg = (wamp > hi_w) | (nspd > hi_n)
    out["still_auc"] = auc(P[:, 1], still_pos, still_neg)
    jl = []
    for _, r in jo.iterrows():
        if not np.isfinite(r.jaw_onset_abs):
            continue
        a, b = np.searchsorted(t, r.start_time), np.searchsorted(t, r.start_time + 1.0)
        w = np.flatnonzero(np.isin(pred[a:b], (3, 4)))
        if len(w):
            jl.append(t[a + w[0]] - r.jaw_onset_abs)
    jl = np.asarray(jl)
    out["jaw_trials"] = len(jl)
    out["jaw_abs_err_ms"] = float(np.median(np.abs(jl)) * 1000) if len(jl) else np.nan
    out["jaw_within_50ms"] = float(np.mean(np.abs(jl) <= 0.05)) if len(jl) else np.nan
    lab = T["labels"][:n].astype(int)
    lab = np.array([0, 1, 2, 0, 3, 4])[lab]  # 002 codes -> full-run classes (nose -> background)
    m = lab > 0
    out["agree_heuristic"] = float(np.mean(pred[m] == lab[m]))
    return out


def seg():
    rows = []
    T = {s: targets(s) for s in PILOT}
    JO = {s: L.jaw_onsets(s) for s in PILOT}
    for d in sorted(SEG.iterdir()):
        if not d.is_dir():
            continue
        head, inp = d.name.split("_", 1) if not d.name.startswith("daart") else ("_".join(d.name.split("_")[:2]), "_".join(d.name.split("_")[2:]))
        for s in PILOT:
            f = d / f"{s}_probs.npz"
            if not f.exists():
                continue
            P = np.load(f)["probs"].astype(float)
            rows.append(dict(model=d.name, head=head, input=inp, session_id=s, **seg_metrics(s, P, T[s], JO[s])))
            print(rows[-1], flush=True)
    # reference: the heuristic labels themselves as a one-hot 'prediction'
    for s in PILOT:
        lab = np.array([0, 1, 2, 0, 3, 4])[T[s]["labels"].astype(int)]
        rows.append(dict(model="thresholds", head="rules", input="kp+top", session_id=s,
                         **seg_metrics(s, np.eye(5)[lab].astype(float), T[s], JO[s])))
    D = pd.DataFrame(rows)
    D.to_csv(L.TABLES / "035_seg_per_session.csv", index=False)
    cols = ["lick_auc", "lick_recall_50ms", "lick_bout_precision", "whisk_auc", "still_auc", "jaw_abs_err_ms", "jaw_within_50ms", "agree_heuristic"]
    S = D.groupby("model")[cols].agg(["mean", "std"]).round(3)
    S["n_sessions"] = D.groupby("model").size()
    S.to_csv(L.TABLES / "035_seg_summary.csv")
    pd.set_option("display.width", 250)
    print(D.groupby("model")[cols].mean().round(3).to_string())


def lagged(X, lags=(-4, -2, 0, 2, 4)):
    out = []
    for l in lags:
        out.append(np.roll(X, l, axis=0))
    return np.hstack(out)


def reps(sid, n):
    R = {}
    k = np.load(INP / f"{sid}_side_keypoints.npz")
    xy = (k["xy"] - k["rest"][None]).reshape(len(k["t"]), -1)
    vel = np.vstack([np.zeros((1, xy.shape[1])), np.diff(xy, axis=0)])
    R["kp"] = np.hstack([xy, vel])
    f = EXT / "facemap" / "side" / f"{sid}_side_facemap.npz"
    if f.exists():
        R["fm"] = np.load(f)["motSVD"][:, :50]
    f = EXT / "beast" / "side" / "embeddings" / f"{sid}_side_pca64.npy"
    if f.exists():
        R["beast"] = np.load(f)
    for a in ("fm", "beast"):
        if a in R:
            R[a] = R[a][:n] if len(R[a]) >= n else np.vstack([R[a], np.repeat(R[a][-1:], n - len(R[a]), 0)])
            R[f"kp_{a}"] = np.hstack([R["kp"][:n], R[a]])
    z = BS.load_states(sid, "arhmm", "side")
    if z is not None:
        R["arhmm_side_states"] = np.eye(65)[z + 1][:, 1:]  # fixed width: global model, state ids shared across sessions
    f = KR / "full" / "lightning_action" / "side" / f"{sid}_probs.npz"
    if f.exists():
        R["la_full_probs (in-sample)"] = np.load(f)["probs"]
    return {k_: np.nan_to_num(v[:n].astype(np.float32)) for k_, v in R.items()}


def repr_():
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    data = {}
    for s in PILOT:
        T = targets(s)
        n = len(T["t"])
        R = reps(s, n)
        idx = np.sort(rng.choice(np.arange(10, n - 10), 40000, replace=False))
        data[s] = dict(R={k: StandardScaler().fit_transform(lagged(v))[idx] for k, v in R.items()},
                       lick=T["dlick"][idx], wamp=T["wamp"][idx], nspd=T["nspd"][idx])
        print(s, list(R), flush=True)
    names = sorted(set.intersection(*[set(d["R"]) for d in data.values()]))
    rows = []
    for name in names:
        for test in PILOT:
            tr = [s for s in PILOT if s != test]
            X = np.vstack([data[s]["R"][name] for s in tr])
            Xt = data[test]["R"][name]
            row = dict(representation=name, session_id=test, n_features=X.shape[1])
            y = np.concatenate([data[s]["lick"] for s in tr])
            m = (y <= 0.025) | (y > 0.5)
            yt = data[test]["lick"]
            mt = (yt <= 0.025) | (yt > 0.5)
            clf = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced").fit(X[m], y[m] <= 0.025)
            row["lick_auc"] = auc(clf.decision_function(Xt), (yt <= 0.025) & mt, (yt > 0.5) & mt)
            for tgt in ("wamp", "nspd"):
                y = np.concatenate([np.log1p(np.abs(data[s][tgt])) for s in tr])
                yt = np.log1p(np.abs(data[test][tgt]))
                ok, okt = np.isfinite(y), np.isfinite(yt)
                rg = Ridge(alpha=10.0).fit(X[ok], y[ok])
                pr = rg.predict(Xt[okt])
                row[f"{tgt}_r2"] = 1 - np.sum((yt[okt] - pr) ** 2) / np.sum((yt[okt] - yt[okt].mean()) ** 2)
            rows.append(row)
            print(row, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(L.TABLES / "035_repr_per_session.csv", index=False)
    S = D.groupby("representation")[["n_features", "lick_auc", "wamp_r2", "nspd_r2"]].agg(["mean", "std"]).round(3)
    S.to_csv(L.TABLES / "035_repr_summary.csv")
    print(D.groupby("representation")[["lick_auc", "wamp_r2", "nspd_r2"]].mean().round(3).to_string())


def fig():
    D = pd.read_csv(L.TABLES / "035_seg_per_session.csv")
    R = pd.read_csv(L.TABLES / "035_repr_per_session.csv") if (L.TABLES / "035_repr_per_session.csv").exists() else None
    seg_cols = [("lick_auc", "lick AUC (piezo)"), ("lick_recall_50ms", "lick recall ±50 ms"),
                ("lick_bout_precision", "lick-bout precision"), ("whisk_auc", "whisking AUC (top amp.)"),
                ("still_auc", "still AUC"), ("jaw_within_50ms", "jaw/lick onset within 50 ms")]
    order = list(D.groupby("model")["lick_auc"].mean().sort_values().index)
    nrow = 2 if R is not None else 1
    fig, axs = plt.subplots(nrow, len(seg_cols), figsize=(2.0 * len(seg_cols), 2.6 * nrow + 0.6), squeeze=False)
    for a, (c, lab) in zip(axs[0], seg_cols):
        for i, m in enumerate(order):
            v = D[D.model == m][c].to_numpy(float)
            a.plot(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, "o", ms=2, color="#888888")
            a.plot([i - 0.3, i + 0.3], [np.nanmean(v)] * 2, "k-", lw=1.2)
        a.set_xticks(range(len(order)))
        a.set_xticklabels(order, rotation=70, ha="right", fontsize=5)
        a.set_title(lab, fontsize=6)
        a.tick_params(labelsize=5)
    if R is not None:
        rc = [("lick_auc", "read-out: lick AUC"), ("wamp_r2", "read-out: whisking amp. R²"), ("nspd_r2", "read-out: top nose speed R²")]
        ro = list(R.groupby("representation")["lick_auc"].mean().sort_values().index)
        for a, (c, lab) in zip(axs[1], rc):
            for i, m in enumerate(ro):
                v = R[R.representation == m][c].to_numpy(float)
                a.plot(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, "o", ms=2, color="#888888")
                a.plot([i - 0.3, i + 0.3], [np.nanmean(v)] * 2, "k-", lw=1.2)
            a.set_xticks(range(len(ro)))
            a.set_xticklabels(ro, rotation=70, ha="right", fontsize=5)
            a.set_title(lab, fontsize=6)
            a.tick_params(labelsize=5)
        for a in axs[1][len(rc):]:
            a.axis("off")
    fig.suptitle("Action segmentation and representations vs independent signals (8 pilot sessions, held-out mouse; "
                 "dots = sessions, bar = mean)", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(L.FIGS / f"035_segment_compare.{ext}", dpi=250)


if __name__ == "__main__":
    {"seg": seg, "repr": repr_, "fig": fig}[sys.argv[1]]()
