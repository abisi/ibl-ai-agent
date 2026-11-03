"""004 -- Validate NWB piezo_lick_times of every ephys NWB_ks4 session against the NIDQ piezo channel (read-only).

NWB piezo_lick_times (converter): detect_piezo_lick_times on log_continuous.bin (5 kHz), mapped to the imec clock by
trial-by-trial interpolation through the trial TTLs. Check: the same detector (NWB_converter
utils.continuous_processing.detect_piezo_lick_times, same threshold from the session yaml) on the NIDQ piezo channel
(25 kHz; channel 7, or 3 on the 'Myri' rig, imDatBsc_sn 543, as in ephys_preprocessing run_tprime.py), mapped NIDQ -> imec
through the NIDQ trial TTLs (CatGT nidq.xa_1_0.txt) and TPrime trial_start_times.txt (TPrime-equivalent), then matched
to the NWB licks (nearest within 20 ms). Reported per session: % NWB licks matched, % NIDQ licks matched, median and IQR
of the difference, median per session third (drift).
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/tables/piezo_lick_validation.csv
Run (haas, ibl-ai-agent .venv): python 004_piezo_lick_validation.py NWB_CONVERTER_DIR [SESSION_ID ...]
"""
import glob
import importlib
import os
import pathlib
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path("NWB_ks4")
DATA = axel_bisi_path("data")
RAW = "/mnt/lsens-data"
OUT = axel_bisi_path("combined_results_ks4") / "ssl-dlc-alignment-debug" / "tables"
TOL = 0.02


def import_converter(path):
    """Import NWB_converter detection helpers, stubbing optional modules absent from this environment."""
    sys.path.insert(0, path)
    for _ in range(20):
        try:
            cp = importlib.import_module("utils.continuous_processing")
            rs = importlib.import_module("utils.readSLGX")
            return cp.detect_piezo_lick_times, rs
        except ModuleNotFoundError as e:
            sys.modules[e.name] = types.ModuleType(e.name)
            for attr in ("ScanImageTiffReader",):
                setattr(sys.modules[e.name], attr, None)
    raise RuntimeError("could not import NWB_converter helpers")


def one(args):
    session_id, conv = args
    detect, rs = import_converter(conv)
    m = session_id[:5]
    r = dict(session_id=session_id)
    try:
        cfg = yaml.safe_load(open(DATA / m / session_id / f"config_{session_id}.yaml"))
        thr = cfg["log_continuous_metadata"]["threshold_dict"].get("lick_trace")
        metas = sorted(glob.glob(f"{RAW}/{m}/Recording/{session_id}/Ephys/*/*nidq.meta"))
        if len(metas) != 1:
            r["error"] = f"{len(metas)} nidq meta files"
            return r
        meta = rs.readMeta(pathlib.Path(metas[0]))
        sr = float(meta["niSampRate"])
        rig = meta.get("imDatBsc_sn", "")
        ap_meta = glob.glob(f"{RAW}/{m}/Recording/{session_id}/Ephys/*/*imec0/*.ap.meta")
        bsc = rs.readMeta(pathlib.Path(ap_meta[0])).get("imDatBsc_sn", "") if ap_meta else rig
        chan = 3 if str(bsc).strip() in ("543", "543.0") else 7
        raw = rs.makeMemMapRaw(pathlib.Path(metas[0].replace(".meta", ".bin")), meta)
        data = np.abs(rs.GainCorrectNI(raw[[chan], :], [chan], meta)[0])
        nidq, _ = detect({"lick_trace": data}, ni_session_sr=sr, lick_threshold=thr)
        catgt = glob.glob(str(DATA / m / session_id / "Ephys" / "catgt_*"))[0]
        ttl_n = np.loadtxt(glob.glob(f"{catgt}/*nidq.xa_1_0.txt")[0])
        ttl_i = np.loadtxt(f"{catgt}/sync_event_times/trial_start_times.txt")
        n = min(len(ttl_n), len(ttl_i))
        nidq_i = np.interp(nidq, ttl_n[:n], ttl_i[:n])
        with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
            nwb = f["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:]
        j = np.clip(np.searchsorted(nwb, nidq_i), 1, len(nwb) - 1)
        near = np.where(np.abs(nwb[j - 1] - nidq_i) < np.abs(nwb[j] - nidq_i), nwb[j - 1], nwb[j])
        d = nidq_i - near
        ok = np.abs(d) < TOL
        k = np.clip(np.searchsorted(nidq_i, nwb), 1, max(len(nidq_i) - 1, 1))
        dn = np.minimum(np.abs(nidq_i[k - 1] - nwb), np.abs(nidq_i[np.minimum(k, len(nidq_i) - 1)] - nwb))
        dd = d[ok] * 1000
        thirds = np.array_split(np.arange(len(d)), 3)
        r.update(rig_bsc=bsc, piezo_channel=chan, threshold=thr, n_ttl_nidq=len(ttl_n), n_ttl_imec=len(ttl_i),
                 n_nwb=len(nwb), n_nidq=len(nidq), pct_nidq_matched=100 * ok.mean(),
                 pct_nwb_matched=100 * np.mean(dn < TOL), median_ms=np.median(dd),
                 iqr_ms=np.percentile(dd, 75) - np.percentile(dd, 25),
                 third_medians_ms=[round(float(np.median(d[t][ok[t]] * 1000)), 3) if ok[t].any() else None for t in thirds])
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def ephys_sessions():
    out = []
    for f in sorted(os.listdir(NWB_DIR)):
        if re.match(r"^[A-Z]{2}\d{3}_\d{8}_\d{6}\.nwb$", f):
            with h5py.File(NWB_DIR / f, "r") as h:
                if "units" in h:
                    out.append(f[:-4])
    return out


if __name__ == "__main__":
    conv = sys.argv[1]
    sessions = sys.argv[2:] or ephys_sessions()
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(one, [(s, conv) for s in sessions]))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "piezo_lick_validation.csv", index=False)
    print(df.drop(columns=[c for c in ("third_medians_ms",) if c in df]).to_string())
