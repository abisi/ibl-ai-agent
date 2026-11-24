"""003 (v3, Kuma GPU) -- keypoint-MoSeq per view at 200 Hz on movement only (002 v3: NWB keypoints in mm, spout-tip
aligned; modelled = moving keypoints only: side nose_tip, jaw, jaw_ref, tongue; top nose_tip, whisker_base, whisker_tip,
particle). Coordinates are passed to kpms in pseudo-pixels (mm x 30) so that its noise priors keep their scale.

Fitting on a subset, applying to all frames: the model is fitted on 60 s chunks taken every 240 s of every training
session (25 % of the frames, ~1.4 M frames for 8 sessions), then applied (kpms.apply_model, 50 iterations) to every
full session separately. AR order nlags = 10 (50 ms at 200 Hz); fix_heading = True (head-fixed).
Per-session normalisation of the keypoints (option `norm`; geometry check = before / after):
  raw:    spout-aligned mm only (no per-session normalisation);
  centre: each keypoint minus its session median position, plus the mean of the session medians (v1 / v2 choice);
  rest:   each keypoint minus the session's resting pose (002: median over pooled-threshold rest frames), plus the mean
          resting pose: the model sees deviations from rest only (v3 default).
Training sets: all = 8 sessions; ho<f> = all but fold f (one R+ and one R- session, as lightning-action 004).
Stages (python 003_kpms.py <view> <stage> key=value ...):
  pca   norm=                         -> <norm>/pca.json (latent_dim = all pose dimensions, whitened)
  sweep norm= kappa= ns=              -> AR-only, 50 iterations, seed 0: median syllable duration, n syllables > 1 %,
                                         session specificity on the training chunks -> <norm>/sweep_ns<ns>_k<kappa>.json
  full  norm= kappa_ar= kappa_full= ns= seed= train= iters=
                                      -> 50 AR + <iters> full iterations, then applied to the 8 full sessions ->
                                         <norm>/<tag>/<sid>.npz (t, z at 200 Hz), done.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/work/lsens/bisi/behaviour_motifs")
SET = os.environ.get("BM_SET", "pilot")  # pilot (8 sessions) or full (all sessions of data/full_sessions.csv)
INP = BASE / "data" / ("inputs" if SET == "pilot" else "full_inputs")
RES_ROOT = BASE / "results" / ("" if SET == "pilot" else "full")
FPS = 200
NLAGS = int(os.environ.get("BM_NLAGS", 10))
APPLY_ITERS = int(os.environ.get("BM_APPLY_ITERS", 50 if os.environ.get("BM_SET", "pilot") == "pilot" else 20))  # full: cost
X64 = os.environ.get("BM_X64", "1") == "1"  # float64: at 200 Hz the 10 AR lags are nearly collinear; float32 gave NaN Ab / Q
CHUNK = 60 * FPS
EVERY = (240 if os.environ.get("BM_SET", "pilot") == "pilot" else 360) * FPS  # full run: 16 training mice, ~17 % of frames
PSEUDO_PX = 30.0  # mm -> pseudo-pixels for kpms


def sessions():
    """pilot: 8 sessions, folds of one R+ and one R- session. full: all sessions; training = data/full_train_sessions.txt
    (balanced subset of mice), the single 'fold' = the sessions not in the training subset."""
    if SET == "pilot":
        sel = pd.read_csv(BASE / "data" / "001_selected_sessions.csv")
        rp = list(sel[sel.reward_group == "R+"].session_id)
        rm = list(sel[sel.reward_group == "R-"].session_id)
        return list(sel.session_id), [[rp[i], rm[i]] for i in range(4)]
    allx = list(pd.read_csv(BASE / "data" / "full_sessions.csv").session_id)
    tr = open(BASE / "data" / "full_train_sessions.txt").read().split()
    return allx, [[s for s in allx if s not in tr]]


def load(view, norm, only=None):
    sids, _ = sessions()
    sids = only or sids
    coords, confs, ts = {}, {}, {}
    for sid in sids:
        d = np.load(INP / f"{sid}_{view}_keypoints.npz")
        coords[sid], confs[sid], ts[sid] = d["xy"].astype(float), np.clip(d["conf"].astype(float), 1e-3, 1), d["t"]
        bps = [str(b) for b in d["bodyparts"]]
    allx, folds = sessions()
    train_all = allx if SET == "pilot" else [s for s in allx if s not in folds[0]]
    if norm == "rest":  # reference = mean resting pose of the training sessions (identical in every job)
        rest = {s: np.load(INP / f"{s}_{view}_keypoints.npz")["rest"].astype(float) for s in set(sids) | set(train_all)}
        ref = np.mean([rest[s] for s in train_all], 0)
        coords = {s: c - rest[s] + ref for s, c in coords.items()}
    elif norm == "centre":
        med = {s: np.nanmedian(np.load(INP / f"{s}_{view}_keypoints.npz")["xy"].astype(float), 0) for s in set(sids) | set(train_all)}
        ref = np.mean([med[s] for s in train_all], 0)
        coords = {s: c - med[s] + ref for s, c in coords.items()}
    coords = {s: c * PSEUDO_PX for s, c in coords.items()}
    return coords, confs, ts, bps


def chunks(coords, confs, train):
    cc, kk = {}, {}
    for s in train:
        for i, a in enumerate(range(0, len(coords[s]) - CHUNK, EVERY)):
            cc[f"{s}__c{i}"], kk[f"{s}__c{i}"] = coords[s][a:a + CHUNK], confs[s][a:a + CHUNK]
    return cc, kk


def durations(zs):
    out = []
    for z in zs.values():
        b = np.diff(np.r_[0, np.flatnonzero(np.diff(z) != 0) + 1, len(z)])
        out.append(b[1:-1] if len(b) > 2 else b)
    return np.concatenate(out) * 1000 / FPS


def specificity(zs):
    """share of frames in states whose usage-normalised frequency is > 80 % in one session (states > 0.5 %)"""
    per = {}
    for k, z in zs.items():
        per.setdefault(k.split("__")[0], []).append(z)
    C = np.array([np.bincount(np.concatenate(v), minlength=200)[:200] for v in per.values()], float)
    N = C / C.sum(1, keepdims=True)
    tot = C.sum(0)
    used = np.flatnonzero(tot / tot.sum() > 0.005)
    share = N[:, used].max(0) / N[:, used].sum(0)
    return float(tot[used][share > 0.8].sum() / tot.sum()), float(np.mean(share > 0.8))


def fit_full(kpms, jax, data, metadata, pca, config, proj, tag, ka, kf, seed, iters):
    """50 AR-only iterations at kappa ka, then the full model at kappa kf; checkpoints in proj / tag"""
    model = kpms.init_model(data, pca=pca, seed=jax.random.PRNGKey(seed), **config())
    model = kpms.update_hypparams(model, kappa=ka)
    model = kpms.fit_model(model, data, metadata, str(proj), tag, ar_only=True, num_iters=50, save_every_n_iters=50,
                           generate_progress_plots=False)[0]
    model = kpms.update_hypparams(model, kappa=kf)
    return kpms.fit_model(model, data, metadata, str(proj), tag, ar_only=False, start_iter=50, num_iters=50 + iters,
                          save_every_n_iters=50, generate_progress_plots=False)[0]


def main():
    import jax
    if X64:
        jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    view, stage = sys.argv[1], sys.argv[2]
    a = dict(x.split("=") for x in sys.argv[3:])
    norm = a.get("norm", "rest")
    proj = RES_ROOT / "kpms" / view / (norm + (f"_no_{os.environ['BM_EXCLUDE']}" if os.environ.get("BM_EXCLUDE") else ""))
    proj.mkdir(parents=True, exist_ok=True)
    sids, folds = sessions()
    train = sids if a.get("train", "all") == "all" else [s for s in sids if s not in folds[int(a["train"][2:])]]
    if stage == "apply":  # full run: apply a fitted model (tag=) to a slice of the sessions (part=i/n)
        i, n = map(int, a["part"].split("/"))
        todo = sids[i::n]
        coords, confs, ts, bps = load(view, norm, only=sorted(set(train) | set(todo)))
    else:
        todo = sids
        coords, confs, ts, bps = load(view, norm, only=None if SET == "pilot" else train)
    cc, kk = chunks(coords, confs, train)
    # side tongue excluded from kpms (v3 pilot): its rare, multi-mm excursions carry ~100 % of the pose variance; with
    # whitened PCs the remaining (sub-mm jaw / nose) dimensions became numerically degenerate and the AR stage produced
    # NaNs (median syllable 20 ms at every kappa). Tongue / licking stay in the AR-HMM features and lightning-action labels.
    use = [b for b in bps if not (view == "side" and b == "tongue")]
    use = [b for b in use if b not in os.environ.get("BM_EXCLUDE", "").split(",")]  # pilot variant (e.g. whisker_tip)
    if stage == "pca":  # shared settings are written once; per-job settings (num_states) only in memory (parallel jobs)
        kpms.setup_project(str(proj), overwrite=True)
        cfg = kpms.load_config(str(proj), check_if_valid=False, build_indexes=False)
        kpms.update_config(str(proj), bodyparts=bps, use_bodyparts=use, anterior_bodyparts=["nose_tip"],
                           posterior_bodyparts=[bps[1]], skeleton=[], fps=FPS, fix_heading=True, video_dir=str(BASE),
                           ar_hypparams=dict(cfg["ar_hypparams"], nlags=NLAGS))

    def config():
        c = kpms.load_config(str(proj))
        c["trans_hypparams"] = dict(c["trans_hypparams"], num_states=int(a.get("ns", 50)))
        return c
    data, metadata = kpms.format_data(cc, kk, **config())
    if stage == "pca":
        pca = kpms.fit_pca(**data, **config())
        kpms.save_pca(pca, str(proj))
        cs = np.cumsum(pca.explained_variance_ratio_)
        # v3: all pose dimensions (2 x (n keypoints - 1)), whitened by kpms. The 90 %-variance rule gave latent_dim 1 on
        # the side view: rare, large tongue excursions carry ~100 % of the variance and would hide jaw / nose movements.
        ld = int(min(len(cs), 2 * (len(use) - 1)))
        kpms.update_config(str(proj), latent_dim=ld)
        json.dump(dict(latent_dim=ld, cumvar=cs.tolist(), norm=norm), open(proj / "pca.json", "w"), indent=1)
        print("latent_dim", ld, cs.round(3), flush=True)
        return
    pca = kpms.load_pca(str(proj))
    if stage == "sweep":
        kappa, ns = float(a["kappa"]), int(a.get("ns", 50))
        model = kpms.init_model(data, pca=pca, **config())
        model = kpms.update_hypparams(model, kappa=kappa)
        name = f"sweep_ns{ns}_k{kappa:.0e}"
        __import__("shutil").rmtree(proj / name, ignore_errors=True)  # stale checkpoint of an earlier attempt
        model = kpms.fit_model(model, data, metadata, str(proj), name, ar_only=True, num_iters=50, save_every_n_iters=50,
                               generate_progress_plots=False)[0]
        z = {k: np.asarray(v["syllable"]) for k, v in kpms.extract_results(model, metadata, save_results=False).items()}
        dur = durations(z)
        allz = np.concatenate(list(z.values()))
        u = np.bincount(allz) / len(allz)
        fr, fs = specificity(z)
        r = dict(norm=norm, kappa=kappa, ns=ns, median_ms=float(np.median(dur)), mean_ms=float(np.mean(dur)),
                 n_syll_1pct=int(np.sum(u > 0.01)), frac_frames_session_specific=fr, frac_states_session_specific=fs)
        json.dump(r, open(proj / f"sweep_ns{ns}_k{kappa:.0e}.json", "w"))
        print(r, flush=True)
        return
    if stage == "apply":
        tag = a["tag"]
        model = kpms.load_checkpoint(str(proj), tag)[0]
        out = proj / tag
        out.mkdir(exist_ok=True)
        for sid in todo:
            if (out / f"{sid}.npz").exists():
                continue
            d1, m1 = kpms.format_data({sid: coords[sid]}, {sid: confs[sid]}, **config())
            res = kpms.apply_model(model, d1, m1, save_results=False, num_iters=APPLY_ITERS, **config())
            z = np.asarray(res[sid]["syllable"])
            np.savez_compressed(out / f"{sid}.npz", z=z.astype(np.int16), t=ts[sid][:len(z)])
            print("applied", sid, flush=True)
        print("done apply", tag, a["part"], flush=True)
        return
    if stage == "full":
        ka, kf, ns, seed, iters = float(a["kappa_ar"]), float(a["kappa_full"]), int(a.get("ns", 50)), int(a["seed"]), int(a["iters"])
        tag = f"ns{ns}_ka{ka:.0e}_kf{kf:.0e}_s{seed}_{a.get('train', 'all')}"
        resume = os.environ.get("BM_RESUME") == "1" and (proj / tag / "checkpoint.h5").exists()  # fitted, apply failed
        if not resume:
            __import__("shutil").rmtree(proj / tag, ignore_errors=True)
            model = fit_full(kpms, jax, data, metadata, pca, config, proj, tag, ka, kf, seed, iters)
        else:
            print("resume: using the saved checkpoint", flush=True)
        kpms.reindex_syllables_in_checkpoint(str(proj), tag)
        model = kpms.load_checkpoint(str(proj), tag)[0]
        ztrain = {k: np.asarray(v["syllable"]) for k, v in kpms.extract_results(model, metadata, save_results=False).items()}
        out = proj / tag
        out.mkdir(exist_ok=True)
        zs = {}
        if SET == "full":  # full run: application to all sessions in separate 'apply' jobs
            json.dump(dict(norm=norm, kappa_ar=ka, kappa_full=kf, ns=ns, seed=seed, iters=iters, train_sessions=train,
                           median_ms_train_chunks=float(np.median(durations(ztrain))), version=kpms.__version__),
                      open(out / "fit.json", "w"), indent=1)
            print("fitted", tag, flush=True)
            return
        for sid in sids:  # apply to every full session (held-out ones included)
            d1, m1 = kpms.format_data({sid: coords[sid]}, {sid: confs[sid]}, **config())
            res = kpms.apply_model(model, d1, m1, save_results=False, num_iters=APPLY_ITERS, **config())
            z = np.asarray(res[sid]["syllable"])
            zs[sid] = z
            np.savez_compressed(out / f"{sid}.npz", z=z.astype(np.int16), t=ts[sid][:len(z)])
            print("applied", sid, flush=True)
        dur = durations(zs)
        allz = np.concatenate([zs[s] for s in train])
        usage = np.bincount(allz, minlength=ns) / len(allz)
        fr, fs = specificity({f"{s}__x": zs[s] for s in sids})
        json.dump(dict(norm=norm, kappa_ar=ka, kappa_full=kf, ns=ns, seed=seed, iters=iters, train=a.get("train", "all"),
                       train_sessions=train, median_ms=float(np.median(dur)), n_syll_1pct=int(np.sum(usage > 0.01)),
                       usage=usage.tolist(), frac_frames_session_specific=fr, frac_states_session_specific=fs,
                       median_ms_train_chunks=float(np.median(durations(ztrain))), version=kpms.__version__),
                  open(out / "done.json", "w"), indent=1)
        print("done", tag, np.median(dur), fr, flush=True)


if __name__ == "__main__":
    main()
