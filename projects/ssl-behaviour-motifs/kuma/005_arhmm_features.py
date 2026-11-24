"""005 (v3, Kuma GPU) -- sticky HDP AR-HMM (jax_moseq.models.arhmm, Gibbs) on interpretable features at 200 Hz, per view.

Features (002 v3, physical units, each camera's 200 Hz grid, NaN gaps linearly interpolated), scaled with POOLED
statistics (mean and SD over all frames of the 8 sessions; no per-session z-scoring, user 2026-11-19); the scaling
constants are saved in done.json so that they can be recomputed on the full dataset:
  top: whisker set-point (deg), whisking amplitude (deg), phase (sin, cos), nose_tip displacement from rest (mm), speed (mm/s);
  side: jaw opening (mm from rest), jaw angle (deg), tongue out (0 / 1), nose_tip displacement from rest (mm), speed (mm/s).
AR order 10 (50 ms), alpha 5.7, gamma 1e3. Fitted on the same training chunks as kpms 003 (60 s every 240 s), then
applied to every full session with the most probable state sequence (jax_moseq arhmm.stateseq_mode, Viterbi) with the
fitted parameters.
Stages (python 005_arhmm_features.py <view> <stage> key=value ...):
  sweep kappa= ns=                  -> 50 Gibbs iterations, seed 0 -> sweep_ns<ns>_k<kappa>.json (median duration,
                                       states > 1 %, session specificity)
  full  kappa= ns= seed= train= iters= -> fit + apply -> <tag>/<sid>.npz (t, z), done.json
"""
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
jax.config.update("jax_enable_x64", os.environ.get("BM_X64", "1") == "1")  # float32 collapsed to one state at 200 Hz
from jax_moseq.models import arhmm  # noqa: E402
from jax_moseq.models.arhmm.gibbs import stateseq_mode  # noqa: E402
from jax_moseq.utils import batch, unbatch  # noqa: E402

BASE = Path("/work/lsens/bisi/behaviour_motifs")
SET = os.environ.get("BM_SET", "pilot")  # pilot or full (all sessions; fitted on data/full_train_sessions.txt)
INP = BASE / "data" / ("inputs" if SET == "pilot" else "full_inputs")
RES_ROOT = BASE / "results" / ("" if SET == "pilot" else "full")
FPS, NLAGS = 200, 10
CHUNK, EVERY = 60 * FPS, 240 * FPS
NOPHASE = os.environ.get("BM_NOPHASE", "0") == "1"  # pilot variant: without the whisking phase
FEATS = {"top": ["whisker_setpoint", "whisking_amp"] + ([] if NOPHASE else ["whisk_phase_sin", "whisk_phase_cos"]) + ["top_nose_disp", "top_nose_speed"],
         "side": ["jaw_open", "jaw_angle", "tongue_out", "nose_disp", "nose_speed"]}


def sessions():
    if SET == "pilot":
        sel = pd.read_csv(BASE / "data" / "001_selected_sessions.csv")
        rp = list(sel[sel.reward_group == "R+"].session_id)
        rm = list(sel[sel.reward_group == "R-"].session_id)
        return list(sel.session_id), [[rp[i], rm[i]] for i in range(4)]
    allx = list(pd.read_csv(BASE / "data" / "full_sessions.csv").session_id)
    tr = open(BASE / "data" / "full_train_sessions.txt").read().split()
    return allx, [[s for s in allx if s not in tr]]


def load_one(view, sid):
    d = np.load(INP / f"{sid}_{view}_features.npz")
    names = [str(n) for n in d["names"]]
    x = d["X"][:, [names.index(f) for f in FEATS[view]]].astype(float)
    idx = np.arange(len(x))
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        x[:, j] = np.interp(idx, idx[ok], x[ok, j])
    return x, d["t"]


def pooled_scaling(view, sids):
    """pooled mean and SD over every 10th frame of all sessions (full run: recomputed on the full dataset)"""
    allx = np.concatenate([load_one(view, s)[0][::10] for s in sids])
    return allx.mean(0), allx.std(0)


def load(view, only=None):
    sids, _ = sessions()
    if SET == "full":
        mu, sd = pooled_scaling(view, sids)
        X, T = {}, {}
        for sid in only:
            x, T[sid] = load_one(view, sid)
            X[sid] = (x - mu) / sd
        return X, T, (mu.tolist(), sd.tolist())
    X, T = {}, {}
    for sid in sids:
        d = np.load(INP / f"{sid}_{view}_features.npz")
        names = [str(n) for n in d["names"]]
        x = d["X"][:, [names.index(f) for f in FEATS[view]]].astype(float)
        idx = np.arange(len(x))
        for j in range(x.shape[1]):
            ok = np.isfinite(x[:, j])
            x[:, j] = np.interp(idx, idx[ok], x[ok, j])
        X[sid], T[sid] = x, d["t"]
    allx = np.concatenate(list(X.values()))
    mu, sd = allx.mean(0), allx.std(0)
    return {k: (v - mu) / sd for k, v in X.items()}, T, (mu.tolist(), sd.tolist())


def durations(zs):
    out = []
    for z in zs.values():
        b = np.diff(np.r_[0, np.flatnonzero(np.diff(z) != 0) + 1, len(z)])
        out.append(b[1:-1] if len(b) > 2 else b)
    return np.concatenate(out) * 1000 / FPS


def specificity(zs):
    per = {}
    for k, z in zs.items():
        per.setdefault(k.split("__")[0], []).append(z)
    C = np.array([np.bincount(np.concatenate(v), minlength=100)[:100] for v in per.values()], float)
    N = C / C.sum(1, keepdims=True)
    tot = C.sum(0)
    used = np.flatnonzero(tot / tot.sum() > 0.005)
    share = N[:, used].max(0) / N[:, used].sum(0)
    return float(tot[used][share > 0.8].sum() / tot.sum()), float(np.mean(share > 0.8))


def unb(z, meta):
    keys, bounds = meta
    zz = unbatch(np.asarray(z), keys, np.stack([bounds[:, 0] + NLAGS, bounds[:, 1]], axis=1))
    return {k: np.pad(v, (NLAGS, 0), mode="edge") for k, v in zz.items()}


def fit(X, train, kappa, ns, iters, seed):
    cc = {}
    for s in train:
        for i, a in enumerate(range(0, len(X[s]) - CHUNK, EVERY)):
            cc[f"{s}__c{i}"] = X[s][a:a + CHUNK]
    x, mask, meta = batch(cc, seg_length=10000)
    data = dict(x=jnp.asarray(x), mask=jnp.asarray(mask))
    D = x.shape[-1]
    model = arhmm.init_model(data, seed=jr.PRNGKey(seed), trans_hypparams=dict(num_states=ns, gamma=1e3, alpha=5.7, kappa=kappa),
                             ar_hypparams=dict(latent_dim=D, nlags=NLAGS, S_0_scale=0.01, K_0_scale=10.0))
    t0 = time.time()
    for it in range(iters):
        model = arhmm.resample_model(data, **model, parallel_message_passing=True)
        if it in (0, 9) or it % 50 == 49:
            print(f"  iter {it + 1} {time.time() - t0:.0f} s", flush=True)
    return model, unb(jax.device_get(model["states"]["z"]), meta)


def apply(model, x):
    xb, mask, meta = batch({"s": x}, seg_length=10000)
    p = model["params"]
    z = stateseq_mode(jnp.asarray(xb), jnp.asarray(mask), p["Ab"], p["Q"], p["pi"])
    return unb(jax.device_get(z), meta)["s"]


def main():
    view, stage = sys.argv[1], sys.argv[2]
    a = dict(x.split("=") for x in sys.argv[3:])
    out = RES_ROOT / "arhmm" / (view + ("_nophase" if NOPHASE and view == "top" else ""))
    out.mkdir(parents=True, exist_ok=True)
    sids, folds = sessions()
    kappa, ns = float(a["kappa"]), int(a.get("ns", 30))
    trn = a.get("train", "all")
    train = sids if trn == "all" else [s for s in sids if s not in folds[int(trn[2:])]]
    if SET == "full":
        train = [s for s in sids if s not in folds[0]]
    X, T, scaling = load(view, only=train if SET == "full" else None)
    if stage == "sweep":
        _, z = fit(X, train, kappa, ns, 50, 0)
        dur = durations(z)
        allz = np.concatenate(list(z.values()))
        u = np.bincount(allz) / len(allz)
        fr, fs = specificity(z)
        r = dict(kappa=kappa, ns=ns, median_ms=float(np.median(dur)), n_states_1pct=int(np.sum(u > 0.01)),
                 frac_frames_session_specific=fr, frac_states_session_specific=fs)
        json.dump(r, open(out / f"sweep_ns{ns}_k{kappa:.0e}.json", "w"))
        print(r, flush=True)
        return
    seed, iters = int(a["seed"]), int(a["iters"])
    tag = f"ns{ns}_k{kappa:.0e}_s{seed}_{trn}"
    model, ztr = fit(X, train, kappa, ns, iters, seed)
    allz = np.concatenate(list(ztr.values()))
    order = np.argsort(-np.bincount(allz, minlength=ns))
    remap = np.empty(ns, int)
    remap[order] = np.arange(ns)
    od = out / tag
    od.mkdir(exist_ok=True)
    zs = {}
    p = jax.device_get(model["params"])
    np.savez_compressed(od / "params.npz", Ab=p["Ab"], Q=p["Q"], pi=p["pi"], remap=remap, pooled_mean=scaling[0], pooled_sd=scaling[1])
    for sid in sids:
        if SET == "full" and sid not in X:
            x, T[sid] = load_one(view, sid)
            X[sid] = (x - np.array(scaling[0])) / np.array(scaling[1])
        z = remap[apply(model, X[sid])]
        if SET == "full":
            del X[sid]
        zs[sid] = z
        np.savez_compressed(od / f"{sid}.npz", z=z.astype(np.int16), t=T[sid][:len(z)])
        print("applied", sid, flush=True)
    dur = durations(zs)
    allz = np.concatenate([zs[s] for s in train])
    usage = np.bincount(allz, minlength=ns) / len(allz)
    fr, fs = specificity({f"{s}__x": zs[s] for s in sids})
    json.dump(dict(view=view, features=FEATS[view], pooled_mean=scaling[0], pooled_sd=scaling[1], kappa=kappa, ns=ns, seed=seed, iters=iters, nlags=NLAGS, train=trn,
                   train_sessions=train, median_ms=float(np.median(dur)), usage=usage.tolist(),
                   n_states_1pct=int(np.sum(usage > 0.01)), frac_frames_session_specific=fr, frac_states_session_specific=fs),
              open(od / "done.json", "w"), indent=1)
    print("done", tag, np.median(dur), fr, flush=True)


if __name__ == "__main__":
    main()
