"""014 -- model inputs of the full run (all sessions of tables/full_sessions.csv), with exactly the definitions of 002 v3
(geometry, features, resting pose, heuristic labels), thresholds and pooled statistics recomputed on the full dataset.

Scalable version of 002 (same functions, imported from it): per session, stage 1 (geometry, features) is written to a
temporary pickle on the NAS and only every 10th frame of the pooled quantities is returned; pooled percentiles are taken
over these subsamples (every 10th frame of every session); stage 2 (resting pose, displacements) and stage 3 (labels,
outputs) reload the pickle. Output: cache/full/inputs/ (same file layout as cache/inputs/), tables/full_label_thresholds.json,
full_label_fractions.csv, full_alignment_reference.csv, full_dropout.csv; provenance/014.json.
Usage: python 014_full_inputs.py [n_workers]
"""
import importlib.util
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bm_lib as L  # noqa: E402
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402

spec = importlib.util.spec_from_file_location("p002", HERE / "002_prepare_inputs.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
OUT = L.CACHE / "full" / "inputs"
_nwb_vs_h5 = P.nwb_vs_h5


def _safe_check(sid, view, v):  # the merged-model h5 is missing for some sessions: the check is then skipped
    try:
        return _nwb_vs_h5(sid, view, v)
    except Exception as e:  # noqa: BLE001
        return dict(session_id=sid, view=view, note=f"no h5 check: {e}")


P.nwb_vs_h5 = _safe_check
TMP = L.CACHE / "full" / "_tmp"
SUB = 10


def ref_one(args):
    sid, view = args
    try:
        sp, nb = P.spout_ref(sid, P.nwb_view(sid, view), view)
    except Exception:  # noqa: BLE001  (no h5 for the top spout)
        sp, nb = None, None
        if view == "top":
            v = P.nwb_view(sid, view)
            x, y, lk = P.kp_xyl(v, view, "nose_base")
            nb = np.array([np.nanmedian(x[lk > 0.5]), np.nanmedian(y[lk > 0.5])]) * P.mm_per_px(sid, view)
    return sid, view, sp, nb


def s1(args):
    sid, refs = args
    S = P.stage1(sid, refs)
    sd = S["side"]
    pickle.dump(S, open(TMP / f"{sid}.pkl", "wb"), protocol=4)
    return sid, sd["motion"][::SUB], sd["amp_on_side"][::SUB]


def s2(args):
    sid, thr = args
    S = pickle.load(open(TMP / f"{sid}.pkl", "rb"))
    sd = S["side"]
    rest = (sd["motion"] < thr["rest_motion"]) & (sd["amp_on_side"] < thr["rest_amp"]) & (sd["dlick"] > 0.5)
    sd["rest_mask"] = rest
    for view in L.VIEWS:
        v = S[view]
        rv = rest if view == "side" else (dh.to_timestamps(rest.astype(float), sd["t"], v["t"], max_gap_s=0.0075) > 0.5)
        v["rest"] = np.nanmedian(v["xy"][rv], 0)
        v["rest_frac"] = float(rv.mean())
    ks, kt = sd["kps"], S["top"]["kps"]
    sd["f"]["jaw_open"] = np.linalg.norm(sd["xy"][:, ks.index("jaw")] - sd["rest"][ks.index("jaw")], axis=-1)
    sd["f"]["nose_disp"] = np.linalg.norm(sd["xy"][:, ks.index("nose_tip")] - sd["rest"][ks.index("nose_tip")], axis=-1)
    tp = S["top"]
    tp["f"]["top_nose_disp"] = np.linalg.norm(tp["xy"][:, kt.index("nose_tip")] - tp["rest"][kt.index("nose_tip")], axis=-1)
    pickle.dump(S, open(TMP / f"{sid}.pkl", "wb"), protocol=4)
    ns = np.nanmean([sd["f"]["nose_speed"], sd["tnose_on_side"]], 0)
    return sid, sd["f"]["jaw_open"][sd["dlick"] < 0.05], ns[::SUB]


def s3(args):
    sid, thr = args
    S = pickle.load(open(TMP / f"{sid}.pkl", "rb"))
    sd, tp = S["side"], S["top"]
    t = sd["t"]
    lab = np.zeros(len(t), np.int8)
    lick = np.zeros(len(t), bool)
    for a, b in zip(np.searchsorted(t, sd["licks"] - 0.04), np.searchsorted(t, sd["licks"] + 0.06)):
        lick[a:max(b, a + 1)] = True
    a_ = sd["amp_on_side"]
    ns = np.nanmean([sd["f"]["nose_speed"], sd["tnose_on_side"]], 0)
    jaw = (sd["f"]["jaw_open"] > thr["jaw"]) & (sd["dlick"] > 0.15)
    whisk = a_ > thr["whisk"]
    nose = (ns > thr["nose_speed"]) & (a_ < thr["nose_amp"])
    still = (sd["motion"] < thr["still_motion"]) & (a_ < thr["still_amp"]) & (sd["dlick"] > 0.5) & (sd["f"]["jaw_open"] < 0.5 * thr["jaw"])
    lab[still], lab[whisk], lab[nose & ~whisk], lab[jaw], lab[lick] = 1, 2, 3, 4, 5
    trials = L.load_trials(sid)
    ep = P.epochs(t, trials)
    for view in L.VIEWS:
        v = S[view]
        nm = len(L.KP_MODEL[view])
        np.savez_compressed(OUT / f"{sid}_{view}_keypoints.npz", t=v["t"], xy=v["xy"][:, :nm], conf=v["conf"][:, :nm],
                            bodyparts=np.array(L.KP_MODEL[view]), rest=v["rest"][:nm].astype(np.float32),
                            xy_static=v["xy"][:, nm:], static_names=np.array(["nose_base"]), shift=v["shift"],
                            mm_per_px=v["mm_per_px"], ref_method=v["ref_method"],
                            units="mm, translated so that the spout tip is at the common reference")
        fk = list(v["f"])
        np.savez_compressed(OUT / f"{sid}_{view}_features.npz", t=v["t"], X=np.vstack([v["f"][k] for k in fk]).T.astype(np.float32),
                            names=np.array(fk))
    np.savez_compressed(OUT / f"{sid}_session.npz", t=t, labels=lab, epoch=ep.astype(str), speed=sd["motion"].astype(np.float32),
                        whisking_amp=a_.astype(np.float32), jaw_open=sd["f"]["jaw_open"].astype(np.float32),
                        tongue_out=sd["f"]["tongue_out"], licks=sd["licks"], rest_mask=sd["rest_mask"])
    trials.to_parquet(OUT / f"{sid}_trials.parquet")
    try:
        L.jaw_onsets(sid).to_parquet(OUT / f"{sid}_jaw_onsets.parquet")
    except Exception as e:  # noqa: BLE001
        print("no jaw onsets", sid, e, flush=True)
    P.OUT = OUT
    drop = P.dropout(sid, S)
    fr = dict(session_id=sid, **{P.LABELS[k]: float(np.mean(lab == k)) for k in P.LABELS}, rest_frac=sd["rest_frac"],
              side_ref=sd["ref_method"], top_ref=tp["ref_method"])
    (TMP / f"{sid}.pkl").unlink()
    return fr, drop


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for d in (OUT, TMP):
        d.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(L.TABLES / "full_sessions.csv")
    sids = list(sel.session_id)
    with ProcessPoolExecutor(nw) as ex:
        got = list(ex.map(ref_one, [(s, v) for v in L.VIEWS for s in sids]))
    refs, ref_rows = {}, []
    for view in L.VIEWS:
        g = {s: (sp, nb) for s, v, sp, nb in got if v == view}
        ok = [s for s in sids if g[s][0] is not None]
        off = np.mean([g[s][0] - g[s][1] for s in ok], 0)
        for s in sids:
            sp, nb = g[s]
            method = "spout tip" if sp is not None else "nose_base + mean (spout - nose_base) of the other sessions"
            refs[(s, view)] = dict(spout_mm=sp if sp is not None else nb + off, method=method)
            ref_rows.append(dict(session_id=s, view=view, method=method, spout_x_mm=refs[(s, view)]["spout_mm"][0],
                                 spout_y_mm=refs[(s, view)]["spout_mm"][1]))
        refs[("template", view)] = np.mean([refs[(s, view)]["spout_mm"] for s in sids], 0)
    pd.DataFrame(ref_rows).to_csv(L.TABLES / "full_alignment_reference.csv", index=False)
    thr = {"units": {"motion": "mm/s", "amp": "deg", "nose_speed": "mm/s", "jaw": "mm"}, "percentiles": P.PCT,
           "definition": f"percentiles over every {SUB}th frame of all sessions of tables/full_sessions.csv (nan ignored); "
                         "jaw = half the median jaw opening within 50 ms of piezo licks (all such frames); see 002 docstring",
           "n_sessions": len(sids)}
    with ProcessPoolExecutor(nw) as ex:
        r1 = list(ex.map(s1, [(s, refs) for s in sids]))
    mo, am = np.concatenate([r[1] for r in r1]), np.concatenate([r[2] for r in r1])
    thr["rest_motion"] = float(np.nanpercentile(mo, P.PCT["rest_motion"]))
    thr["rest_amp"] = float(np.nanpercentile(am, P.PCT["rest_amp"]))
    thr["whisk"] = float(np.nanpercentile(am, P.PCT["whisk"]))
    thr["nose_amp"] = float(np.nanpercentile(am, P.PCT["nose_amp"]))
    thr["still_amp"] = float(np.nanpercentile(am, P.PCT["still_amp"]))
    thr["still_motion"] = float(np.nanpercentile(mo, P.PCT["still_motion"]))
    with ProcessPoolExecutor(nw) as ex:
        r2 = list(ex.map(s2, [(s, thr) for s in sids]))
    thr["jaw"] = 0.5 * float(np.nanmedian(np.concatenate([r[1] for r in r2])))
    thr["nose_speed"] = float(np.nanpercentile(np.concatenate([r[2] for r in r2]), P.PCT["nose"]))
    json.dump(thr, open(L.TABLES / "full_label_thresholds.json", "w"), indent=1)
    with ProcessPoolExecutor(nw) as ex:
        r3 = list(ex.map(s3, [(s, thr) for s in sids]))
    fr = [r[0] for r in r3]
    pd.DataFrame([d for r in r3 for d in r[1]]).to_csv(L.TABLES / "full_dropout.csv", index=False)
    pd.DataFrame(fr).to_csv(L.TABLES / "full_label_fractions.csv", index=False)
    json.dump(dict(script="014_full_inputs.py", n_sessions=len(sids), thresholds=thr, definitions="as 002 v3",
                   reference_mm={v: refs[("template", v)].tolist() for v in L.VIEWS}, versions=L.versions()),
              open(L.RES / "provenance" / "014.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in thr.items() if isinstance(v, float)}, indent=0))
    print(pd.DataFrame(fr).describe().round(3).to_string())


if __name__ == "__main__":
    main()
