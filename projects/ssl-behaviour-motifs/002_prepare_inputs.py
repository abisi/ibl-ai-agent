"""002 (v3) -- model inputs for the 8 pilot sessions (haas, CPU): geometry-normalised keypoints in mm, features in
physical units, resting pose, heuristic labels with POOLED thresholds, epochs, DLC dropout; native 200 Hz.

Rule (user 2026-11-19): states and actions must mean the same in every animal -> nothing is defined relative to a
session. Geometry (user correction 2026-11-19): pixels -> mm with the setup calibration (scripts/ssl_dlc_calibration,
per camera), then a translation so that the spout tip (median over frames with likelihood > 0.5) sits at the mean spout
position; no rotation or extra scaling (the camera framing is very similar across sessions). The top spout is not in the
NWB: taken from the merged-model h5 (frame mapping verified); not tracked in the top view of MH037 / MH039 -> nose_base +
mean (spout - nose_base) vector of the other sessions. particle and jaw_ref move (whisker / jaw angle kinematics):
modelled, never used as references. Modelled keypoints: side nose_tip, jaw, jaw_ref, tongue; top nose_tip,
whisker_base, whisker_tip, particle.
Source: NWB_ks4 DLC series (dlc_helpers.load_dlc, pixels, per-keypoint likelihoods, camera timestamps); no h5. Check
vs the v1 h5: tables/002_nwb_vs_h5.csv (lag 0 frames, r >= 0.98 in all 16 session x views in v2).
Gating: jaw never gated; tongue < 0.6 -> placed on the jaw ("in the mouth"); other points < 0.1 -> interpolated; running
median 5 frames.
Features (200 Hz, each camera's grid): top whisker angle (deg, NWB whisker_angle, dlc_helpers clean_series),
set-point (< 3 Hz), whisking amplitude and phase (Hilbert, 5-15 Hz band), nose_tip displacement from rest (mm) and speed
(mm/s, 25 ms smoothing); side jaw angle (deg, NWB jaw_angle), jaw opening (jaw displacement from rest, mm), jaw speed, tongue out (likelihood > 0.6),
nose_tip displacement from rest and speed; motion = summed speed of the modelled keypoints of both views (mm/s, 50 ms).
Resting pose (per session, used to remove face shape / camera geometry): median aligned position of every keypoint over
rest frames = motion < pooled 30th percentile AND whisking amplitude < pooled 40th percentile AND no piezo lick within
0.5 s (pooled = percentiles over all frames of the 8 sessions). kpms (003, norm=rest) subtracts it.
Heuristic labels (side grid; 0 background, 1 still, 2 whisking, 3 nose, 4 jaw (no lick), 5 lick; priority lick > jaw >
nose > whisking > still), every threshold a pooled statistic in physical units (tables/label_thresholds.json, with the
percentile used so that it can be recomputed on the full dataset):
- lick: [-40, +60] ms around piezo licks;
- jaw: jaw opening > half the pooled median jaw opening within 50 ms of piezo licks (mm), no lick within 150 ms;
- whisking: amplitude > pooled 75th percentile (deg);
- nose: mean nose speed (side, top) > pooled 85th percentile (mm/s) and amplitude < pooled 50th percentile;
- still: motion < pooled 30th percentile, amplitude < pooled 40th percentile, jaw opening < half the jaw threshold, no
  lick within 0.5 s.
Outputs: cache/inputs/<sid>_{side,top}_keypoints.npz (aligned mm, rest pose), _features.npz, _session.npz, _trials,
_jaw_onsets, _dropout; tables/label_thresholds.json, 002_label_fractions.csv, 002_dropout.csv, 002_nwb_vs_h5.csv.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import ssl_dlc_calibration as cal  # noqa: E402
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402

OUT = L.CACHE / "inputs"
CUT_DEFAULT = 0.1
LABELS = {0: "background", 1: "still", 2: "whisking", 3: "nose", 4: "jaw (no lick)", 5: "lick"}
DROP_KPS = {"side": ["nose_tip", "nose_base", "jaw", "jaw_ref", "tongue", "spout"],
            "top": ["nose_tip", "nose_base", "whisker_base", "whisker_tip", "particle"]}
PCT = dict(rest_motion=30, rest_amp=40, whisk=75, nose=85, nose_amp=50, still_motion=30, still_amp=40)


def cutoff(kp):
    return {"jaw": None, "tongue": 0.6}.get(kp, CUT_DEFAULT)


def interp_nan(y):
    y = np.asarray(y, float).copy()
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    idx = np.arange(len(y))
    y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def mm_per_px(sid, view):
    return 1.0 / cal.px_per_mm(sid, "jaw_x" if view == "side" else "whisker_tip_x")


def nwb_view(sid, view):
    return dh.load_dlc(str(L.NWB_DIR / f"{sid}.nwb"), views=(view,), physical_units=False)[view]


def kp_xyl(v, view, kp):
    n = L.NWB_NAME[view](kp)
    return v.data[f"{n}_x"].to_numpy(float), v.data[f"{n}_y"].to_numpy(float), v.likelihood[f"{n}_likelihood"].to_numpy(float)


def spout_ref(sid, v, view):
    """median spout-tip position (mm, camera frame) over frames with likelihood > 0.5, and the nose_base median.
    Side: NWB spout series. Top: the NWB has no top spout series -> merged-model h5 (frame mapping to the NWB verified:
    lag 0, r >= 0.98, tables/002_nwb_vs_h5.csv); None where the top spout is not tracked (median likelihood < 0.5:
    MH037, MH039)."""
    k = mm_per_px(sid, view)
    x, y, lk = kp_xyl(v, view, "nose_base")
    nb = np.array([np.nanmedian(x[lk > 0.5]), np.nanmedian(y[lk > 0.5])]) * k
    if view == "side":
        x, y, lk = kp_xyl(v, view, "spout")
    else:
        df = L.read_h5(L.dlc_h5(sid, view)).iloc[:len(v.timestamps)]
        x, y, lk = (df[("spout", c)].to_numpy(float) for c in ("x", "y", "likelihood"))
    if np.median(lk) < 0.5:
        return None, nb
    return np.array([np.nanmedian(x[lk > 0.5]), np.nanmedian(y[lk > 0.5])]) * k, nb


def aligned(sid, v, view, shift):
    """modelled keypoints + nose_base, gated, interpolated, in mm, translated so that the spout tip is at the common
    reference position (no rotation / scaling beyond the mm calibration: the camera framing is similar across sessions)"""
    kps = L.KP_MODEL[view] + ["nose_base"]
    k = mm_per_px(sid, view)
    xy, conf = np.full((len(v.timestamps), len(kps), 2), np.nan), np.zeros((len(v.timestamps), len(kps)))
    jx, jy, _ = kp_xyl(v, view, "jaw") if view == "side" else (None, None, None)
    for i, kp in enumerate(kps):
        x, y, lk = kp_xyl(v, view, kp)
        c = cutoff(kp)
        if kp == "tongue":
            bad = lk < c
            x, y = np.where(bad, jx, x), np.where(bad, jy, y)
        elif c is not None:
            bad = lk < c
            x, y = np.where(bad, np.nan, x), np.where(bad, np.nan, y)
        x, y = median_filter(interp_nan(x), 5) * k, median_filter(interp_nan(y), 5) * k
        xy[:, i] = np.vstack([x, y]).T + shift
        conf[:, i] = lk
    return xy, conf, kps


def bandpass(x, fs, lo, hi):
    return sosfiltfilt(butter(3, [lo, hi], btype="band", fs=fs, output="sos"), x)


def lowpass(x, fs, f):
    return sosfiltfilt(butter(3, f, btype="low", fs=fs, output="sos"), x)


def speed(xy, t, sm):
    return uniform_filter1d(np.linalg.norm(np.gradient(xy, t, axis=0), axis=-1), sm)


def nwb_vs_h5(sid, view, v):
    df = L.read_h5(L.dlc_h5(sid, view))
    kp, c = ("jaw", "y") if view == "side" else ("whisker_tip", "x")
    a = interp_nan(v.data[f"{L.NWB_NAME[view](kp)}_{c}"].to_numpy(float))
    b = interp_nan(df[(kp, c)].to_numpy(float))
    n = min(len(a), len(b))
    return dict(session_id=sid, view=view, series=f"{kp}_{c}", n_nwb=len(a), n_h5=len(b),
                r_lag0_whole=float(np.corrcoef(a[:n], b[:n])[0, 1]),
                max_abs_diff_px_median=float(np.nanmedian(np.abs(a[:n] - b[:n]))))


def stage1(sid, refs):
    """geometry, raw (rest-independent) features; returns arrays for pooling"""
    views = {view: nwb_view(sid, view) for view in L.VIEWS}
    out = {"sid": sid, "check": [nwb_vs_h5(sid, view, v) for view, v in views.items()]}
    for view, v in views.items():
        ref = refs[(sid, view)]
        shift = refs[("template", view)] - ref["spout_mm"]
        xy, conf, kps = aligned(sid, v, view, shift)
        out[view] = dict(t=v.timestamps, xy=xy.astype(np.float32), conf=conf.astype(np.float32), kps=kps, shift=shift,
                         ref_method=ref["method"], mm_per_px=mm_per_px(sid, view), rate=v.rate, fixed=v.fixed_reference)
        out[view]["lik"] = {kp: kp_xyl(v, view, kp)[2].astype(np.float32) for kp in DROP_KPS[view]}
    top, side = out["top"], out["side"]
    kt = top["kps"]
    # whisker angle (deg) as stored in the NWB (particle / whisker_base kinematics), dlc_helpers cleaning; jaw angle idem
    vt = dh.load_dlc(str(L.NWB_DIR / f"{sid}.nwb"), views=("top",))["top"]
    wa, _ = dh.clean_series(vt, "whisker_angle")
    wl = np.isnan(wa)
    ang = interp_nan(wa)
    vs_ = dh.load_dlc(str(L.NWB_DIR / f"{sid}.nwb"), views=("side",))["side"]
    jang = interp_nan(dh.clean_series(vs_, "jaw_angle")[0])
    fs = top["rate"]
    an = hilbert(bandpass(ang, fs, 5, 15))
    sm = int(0.025 * fs)
    top["f"] = dict(whisker_angle=ang, whisker_setpoint=lowpass(ang, fs, 3.0), whisking_amp=np.abs(an),
                    whisk_phase_sin=np.sin(np.angle(an)), whisk_phase_cos=np.cos(np.angle(an)),
                    top_nose_speed=speed(top["xy"][:, kt.index("nose_tip")], top["t"], sm))
    for k in ("whisker_angle", "whisker_setpoint", "whisking_amp", "whisk_phase_sin", "whisk_phase_cos"):
        top["f"][k] = np.where(wl, np.nan, top["f"][k])
    ks = side["kps"]
    side["f"] = dict(jaw_angle=jang, jaw_speed=speed(side["xy"][:, ks.index("jaw")], side["t"], sm),
                     nose_speed=speed(side["xy"][:, ks.index("nose_tip")], side["t"], sm),
                     tongue_out=(side["lik"]["tongue"] > 0.6).astype(np.float32))
    t = side["t"]
    mt = lambda view: uniform_filter1d(sum(speed(out[view]["xy"][:, i], out[view]["t"], 1)  # noqa: E731
                                           for i in range(len(L.KP_MODEL[view]))), int(0.05 * fs))
    side["motion"] = mt("side") + dh.to_timestamps(mt("top"), top["t"], t, max_gap_s=0.0075)
    side["amp_on_side"] = dh.to_timestamps(top["f"]["whisking_amp"], top["t"], t, max_gap_s=0.0075)
    side["tnose_on_side"] = dh.to_timestamps(top["f"]["top_nose_speed"], top["t"], t, max_gap_s=0.0075)
    side["licks"] = np.sort(L.piezo_licks(sid))
    j = np.clip(np.searchsorted(side["licks"], t), 1, len(side["licks"]) - 1)
    side["dlick"] = np.minimum(np.abs(t - side["licks"][j - 1]), np.abs(side["licks"][j] - t))
    return out


def epochs(t, trials):
    lab = np.full(len(t), "other", dtype=object)
    k = trials[trials.keep]
    act = trials[trials.active]
    if len(act):
        lab[(t >= act.start_time.min() - 2) & (t <= act.stop_time.max() + 5)] = "iti"
    for _, r in k.iterrows():
        lab[(t >= r.start_time - 2) & (t < r.start_time)] = "pre"
        lab[(t >= r.start_time) & (t <= r.response_window_stop_time)] = "resp"
    for _, r in trials[trials.context == "passive"].iterrows():
        lab[(t >= r.start_time - 2) & (t <= r.stop_time + 2)] = "other"
    return lab


def dropout(sid, S):
    rows, arr = [], {}
    lags = np.arange(-100, 101)
    licks = S["side"]["licks"]
    for view in L.VIEWS:
        t = S[view]["t"]
        li = np.searchsorted(t, licks)
        li = li[(li > 250) & (li < len(t) - 250)]
        minute = ((t - t[0]) // 60).astype(int)
        for kp, lk in S[view]["lik"].items():
            c = cutoff(kp) or 0.5
            low = lk < c
            peri = low[li[:, None] + 2 * lags[None, :]].mean(0)
            arr[f"{view}|{kp}|per_minute"] = np.bincount(minute, weights=low) / np.bincount(minute)
            arr[f"{view}|{kp}|peri_lick"] = peri
            rows.append(dict(session_id=sid, view=view, bodypart=kp, cutoff=c, frac_low=float(low.mean()),
                             frac_low_peri_lick_0_200ms=float(peri[(lags >= 0) & (lags < 20)].mean())))
    np.savez_compressed(OUT / f"{sid}_dropout.npz", lag_s=lags * 0.01, **arr)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(L.TABLES / "001_selected_sessions.csv")
    sids = list(sel.session_id)
    refs, ref_rows = {}, []
    for view in L.VIEWS:
        got = {s: spout_ref(s, nwb_view(s, view), view) for s in sids}
        ok = [s for s in sids if got[s][0] is not None]
        off = np.mean([got[s][0] - got[s][1] for s in ok], 0)  # mean spout - nose_base vector (fallback)
        for s in sids:
            sp, nb = got[s]
            method = "spout tip" if sp is not None else "nose_base + mean (spout - nose_base) of the other sessions"
            refs[(s, view)] = dict(spout_mm=sp if sp is not None else nb + off, method=method)
            ref_rows.append(dict(session_id=s, view=view, method=method, spout_x_mm=refs[(s, view)]["spout_mm"][0],
                                 spout_y_mm=refs[(s, view)]["spout_mm"][1], nose_base_x_mm=nb[0], nose_base_y_mm=nb[1]))
        refs[("template", view)] = np.mean([refs[(s, view)]["spout_mm"] for s in sids], 0)
    pd.DataFrame(ref_rows).to_csv(L.TABLES / "002_alignment_reference.csv", index=False)
    with ProcessPoolExecutor(8) as ex:
        S = dict(zip(sids, ex.map(stage1, sids, [refs] * len(sids))))
    # pooled thresholds, step 1: rest frames
    pool = lambda f: np.concatenate([f(S[s]) for s in sids])  # noqa: E731
    thr = {"units": {"motion": "mm/s", "amp": "deg", "nose_speed": "mm/s", "jaw": "mm"}, "percentiles": PCT,
           "definition": "percentiles over all frames of the pooled sessions (nan ignored); see 002 docstring"}
    thr["rest_motion"] = float(np.nanpercentile(pool(lambda x: x["side"]["motion"]), PCT["rest_motion"]))
    thr["rest_amp"] = float(np.nanpercentile(pool(lambda x: x["side"]["amp_on_side"]), PCT["rest_amp"]))
    for s in sids:
        sd = S[s]["side"]
        rest = (sd["motion"] < thr["rest_motion"]) & (sd["amp_on_side"] < thr["rest_amp"]) & (sd["dlick"] > 0.5)
        sd["rest_mask"] = rest
        for view in L.VIEWS:
            v = S[s][view]
            rv = rest if view == "side" else (dh.to_timestamps(rest.astype(float), sd["t"], v["t"], max_gap_s=0.0075) > 0.5)
            v["rest"] = np.nanmedian(v["xy"][rv], 0)
            v["rest_frac"] = float(rv.mean())
        ks, kt = sd["kps"], S[s]["top"]["kps"]
        sd["f"]["jaw_open"] = np.linalg.norm(sd["xy"][:, ks.index("jaw")] - sd["rest"][ks.index("jaw")], axis=-1)
        sd["f"]["nose_disp"] = np.linalg.norm(sd["xy"][:, ks.index("nose_tip")] - sd["rest"][ks.index("nose_tip")], axis=-1)
        tp = S[s]["top"]
        tp["f"]["top_nose_disp"] = np.linalg.norm(tp["xy"][:, kt.index("nose_tip")] - tp["rest"][kt.index("nose_tip")], axis=-1)
    # step 2: label thresholds
    thr["jaw"] = 0.5 * float(np.nanmedian(pool(lambda x: x["side"]["f"]["jaw_open"][x["side"]["dlick"] < 0.05])))
    amp = pool(lambda x: x["side"]["amp_on_side"])
    thr["whisk"] = float(np.nanpercentile(amp, PCT["whisk"]))
    thr["nose_amp"] = float(np.nanpercentile(amp, PCT["nose_amp"]))
    thr["still_amp"] = float(np.nanpercentile(amp, PCT["still_amp"]))
    ns = lambda x: np.nanmean([x["side"]["f"]["nose_speed"], x["side"]["tnose_on_side"]], 0)  # noqa: E731
    thr["nose_speed"] = float(np.nanpercentile(pool(ns), PCT["nose"]))
    thr["still_motion"] = float(np.nanpercentile(pool(lambda x: x["side"]["motion"]), PCT["still_motion"]))
    fr_rows, metas = [], []
    for s in sids:
        sd, tp = S[s]["side"], S[s]["top"]
        t = sd["t"]
        lab = np.zeros(len(t), np.int8)
        lick = np.zeros(len(t), bool)
        for a, b in zip(np.searchsorted(t, sd["licks"] - 0.04), np.searchsorted(t, sd["licks"] + 0.06)):
            lick[a:max(b, a + 1)] = True
        a_ = sd["amp_on_side"]
        jaw = (sd["f"]["jaw_open"] > thr["jaw"]) & (sd["dlick"] > 0.15)
        whisk = a_ > thr["whisk"]
        nose = (ns(S[s]) > thr["nose_speed"]) & (a_ < thr["nose_amp"])
        still = (sd["motion"] < thr["still_motion"]) & (a_ < thr["still_amp"]) & (sd["dlick"] > 0.5) & (sd["f"]["jaw_open"] < 0.5 * thr["jaw"])
        lab[still], lab[whisk], lab[nose & ~whisk], lab[jaw], lab[lick] = 1, 2, 3, 4, 5
        trials = L.load_trials(s)
        ep = epochs(t, trials)
        for view in L.VIEWS:
            v = S[s][view]
            nm = len(L.KP_MODEL[view])
            np.savez_compressed(OUT / f"{s}_{view}_keypoints.npz", t=v["t"], xy=v["xy"][:, :nm], conf=v["conf"][:, :nm],
                                bodyparts=np.array(L.KP_MODEL[view]), rest=v["rest"][:nm].astype(np.float32),
                                xy_static=v["xy"][:, nm:], static_names=np.array(["nose_base"]), shift=v["shift"],
                                mm_per_px=v["mm_per_px"], ref_method=v["ref_method"],
                                units="mm, translated so that the spout tip is at the common reference")
            fk = list(v["f"])
            np.savez_compressed(OUT / f"{s}_{view}_features.npz", t=v["t"], X=np.vstack([v["f"][k] for k in fk]).T.astype(np.float32),
                                names=np.array(fk))
        np.savez_compressed(OUT / f"{s}_session.npz", t=t, labels=lab, epoch=ep.astype(str), speed=sd["motion"].astype(np.float32),
                            whisking_amp=a_.astype(np.float32), jaw_open=sd["f"]["jaw_open"].astype(np.float32),
                            tongue_out=sd["f"]["tongue_out"], licks=sd["licks"], rest_mask=sd["rest_mask"])
        trials.to_parquet(OUT / f"{s}_trials.parquet")
        L.jaw_onsets(s).to_parquet(OUT / f"{s}_jaw_onsets.parquet")
        fr_rows.append(dict(session_id=s, **{LABELS[k]: float(np.mean(lab == k)) for k in LABELS}, rest_frac=sd["rest_frac"]))
        metas.append(dict(session_id=s, side_n=len(t), top_n=len(tp["t"]), side_rate=sd["rate"], top_rate=tp["rate"],
                          side_ref=sd["ref_method"], top_ref=tp["ref_method"], side_mm_per_px=sd["mm_per_px"], top_mm_per_px=tp["mm_per_px"],
                          label_frac={k: float(np.mean(lab == k)) for k in LABELS}))
    pd.DataFrame([c for s in sids for c in S[s]["check"]]).to_csv(L.TABLES / "002_nwb_vs_h5.csv", index=False)
    pd.DataFrame([r for s in sids for r in dropout(s, S[s])]).to_csv(L.TABLES / "002_dropout.csv", index=False)
    pd.DataFrame(fr_rows).to_csv(L.TABLES / "002_label_fractions.csv", index=False)
    json.dump(thr, open(L.TABLES / "label_thresholds.json", "w"), indent=1)
    json.dump({"script": "002_prepare_inputs.py (v3)", "fs": L.FS_DS, "source": "NWB_ks4 DLC series via dlc_helpers.load_dlc",
               "geometry": "px -> mm (setup calibration), then translation so that the spout tip (median, likelihood > 0.5) is at the "
                           "mean spout position (no rotation / scaling); top view without a tracked spout: nose_base + mean "
                           "(spout - nose_base) vector of the other sessions; tables/002_alignment_reference.csv",
               "keypoints_modelled": L.KP_MODEL, "reference_mm": {v: refs[("template", v)].tolist() for v in L.VIEWS},
               "thresholds": thr, "sessions": metas, "label_names": LABELS, "versions": L.versions()},
              open(L.RES / "provenance" / "002.json", "w"), indent=1, default=str)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in thr.items() if k not in ("definition",)}, indent=0))
    print(pd.DataFrame(fr_rows).round(3).to_string())
    print(pd.DataFrame(ref_rows).round(2).to_string())


if __name__ == "__main__":
    main()
