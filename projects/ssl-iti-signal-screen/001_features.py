"""ITI signal screen, step 1 (haas): per whisker trial, candidate ITI / pre-stimulus behavioural features.

Exploratory screen (2026-11-04, user): find any behavioural signal in the ITI that (1) correlates with the learning
curve, (2) predicts the upcoming whisker choice and (3) relates to overall performance; rank by effect size, no
multiple-comparison correction, day 0 and expert (day > 0) sessions separately.

Sessions: every whisker-day session with an exact learning curve (combined_results_ks4/<mouse>/whisker_<day>/
learning_curve/*_whisker_trial_learning_curve_exact.h5, trials = active whisker trials, perf != 6, early_lick == 0),
matched to NWB_ks4 trials on start_time. Cohort from the reference sheet.

Per whisker trial (t0 = start_time; previous trial = previous trial of any type, any context):
  windows   pre0.2 (-0.2, 0), pre1 (-1, 0), pre2 (-2, 0), iti (previous stop_time, t0 - 0.2),
            postprev (previous stop_time, + 2 s)
  signals   DLC series from the NWB (jaw distance/angle/velocity, nose side distance/angle/velocity, top nose
            distance/angle/velocity, whisker angle/velocity, pupil area, tongue distance); series whose data and
            timestamps differ by > 200 samples are skipped
  stats     mean, speed (mean |diff| per s), std, slope (per s), range (max - min)
  extra     whisking-band power fraction (6-20 Hz / 1-50 Hz, Welch) of whisker angle in pre1, pre2, iti;
            licks: ITI duration (t0 - previous stop), piezo licks in iti / postprev, piezo lick rate in iti, time
            since the last piezo lick before t0, DLC jaw and tongue licks in iti
Also stored per trial: lick (upcoming choice), exact-curve p_mean / p_chance / p_low / p_high, and the history
predictors of ssl-face-lick-decoding (trial index, previous lick / reward / whisker lick, log time since lick).
Output: combined_results_ks4/ssl-iti-signal-screen/features/<session>.parquet, features/sessions.csv
Run on haas: nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 001_features.py --jobs 32
"""

import argparse
import glob
import importlib.util
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "dlc_kpca", HERE.parent / "ssl-facemap-lick-decoding" / "dlc_kpca.py")  # DLC helpers (ex ssl-face-lick-decoding pilot)
dlc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dlc)  # load (trials), trial_table (history predictors)

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
NWB = ROOT / "NWB_ks4"
OUT = ROOT / "combined_results_ks4" / "ssl-iti-signal-screen" / "features"
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
SIGNALS = ["jaw_distance", "jaw_angle", "jaw_velocity", "nose_distance", "nose_angle", "nose_velocity",
           "top_nose_distance", "top_nose_angle", "top_nose_velocity", "whisker_angle", "whisker_velocity",
           "pupil_area", "tongue_distance"]
FIXED = {"pre0.2": (-0.2, 0.0), "pre1": (-1.0, 0.0), "pre2": (-2.0, 0.0)}
MAX_MISMATCH = 200


def _str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load_nwb(path):
    with h5py.File(path, "r") as f:
        tr = f["intervals/trials"]
        trials = pd.DataFrame({c: [_str(v) if isinstance(v, bytes) else v for v in tr[c][:]] for c in
                               ("trial_type", "start_time", "stop_time", "lick_flag", "lick_time", "context", "perf",
                                "early_lick", "reward_available")})
        b = f.get("processing/behavior/BehavioralTimeSeries", {})
        series = {}
        for s in SIGNALS:
            if s in b:
                d, t = b[s]["data"][:], b[s]["timestamps"][:]
                if abs(len(d) - len(t)) <= MAX_MISMATCH:
                    n = min(len(d), len(t))
                    series[s] = (t[:n], d[:n].astype(float))
        ev = f.get("processing/behavior/BehavioralEvents", {})
        events = {k: ev[k]["timestamps"][:] for k in ("piezo_lick_times", "jaw_dlc_licks", "tongue_dlc_licks") if k in ev}
    return trials, series, events


def stats_in(t, x, a, b):
    i0, i1 = np.searchsorted(t, [a, b])
    seg, ts = x[i0:i1], t[i0:i1]
    ok = np.isfinite(seg)
    if ok.sum() < 5:
        return dict(mean=np.nan, speed=np.nan, std=np.nan, slope=np.nan, range=np.nan)
    seg, ts = seg[ok], ts[ok]
    dt = np.median(np.diff(ts)) if len(ts) > 1 else np.nan
    return dict(mean=seg.mean(), speed=np.mean(np.abs(np.diff(seg))) / dt, std=seg.std(),
                slope=np.polyfit(ts - ts[0], seg, 1)[0], range=np.ptp(seg))


def whisk_power(t, x, a, b):
    i0, i1 = np.searchsorted(t, [a, b])
    seg = x[i0:i1]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 64:
        return np.nan
    fs = 1 / np.median(np.diff(t[i0:i1]))
    f, p = welch(seg - seg.mean(), fs=fs, nperseg=min(128, len(seg)))
    tot = p[(f >= 1) & (f <= 50)].sum()
    return p[(f >= 6) & (f <= 20)].sum() / tot if tot > 0 else np.nan


def count(ev, a, b):
    return int(np.sum((ev >= a) & (ev < b)))


def process(args):
    curve_path, cohort = args
    cur = pd.read_hdf(curve_path).iloc[0]
    sid = cur.session_id
    out = OUT / f"{sid}.parquet"
    if out.exists():
        return dict(session_id=sid, status="exists")
    trials, series, events = load_nwb(NWB / f"{sid}.nwb")
    trials = trials.sort_values("start_time").reset_index(drop=True)
    hist = dlc.trial_table(trials.drop(columns=["stop_time"]))  # exact-curve trials with history predictors
    cst = np.asarray(cur.start_time, float)
    d = np.abs(hist.start_time.to_numpy()[:, None] - cst[None, :])
    if len(hist) != len(cst) or d.min(1).max() > 1e-3:
        return dict(session_id=sid, status=f"trial mismatch ({len(hist)} vs {len(cst)} curve trials)")
    order = d.argmin(1)
    starts_all, stops_all = trials.start_time.to_numpy(float), trials.stop_time.to_numpy(float)
    piezo = events.get("piezo_lick_times", np.array([]))
    rows = []
    for k, r in hist.iterrows():
        t0 = r.start_time
        j = np.searchsorted(starts_all, t0 - 1e-6) - 1  # previous trial of any type
        prev_stop = stops_all[j] if j >= 0 else np.nan
        wins = dict(FIXED)
        if np.isfinite(prev_stop) and t0 - 0.2 - prev_stop > 0.5:
            wins["iti"] = (prev_stop - t0, -0.2)
        if np.isfinite(prev_stop):
            wins["postprev"] = (prev_stop - t0, prev_stop - t0 + 2.0)
        row = dict(trial=k, start_time=t0, lick=int(r.lick_flag), p_mean=cur.p_mean[order[k]],
                   p_chance=cur.p_chance[order[k]], p_low=cur.p_low[order[k]], p_high=cur.p_high[order[k]],
                   trial_idx=r.trial_idx, prev_lick=r.prev_lick, prev_reward=r.prev_reward,
                   prev_whisker_lick=r.prev_whisker_lick, log_since_lick=r.log_since_lick)
        for s, (ts, x) in series.items():
            for w, (a, b) in wins.items():
                for st, v in stats_in(ts, x, t0 + a, t0 + b).items():
                    row[f"{s}|{w}|{st}"] = v
            if s == "whisker_angle":
                for w in ("pre1", "pre2", "iti"):
                    if w in wins:
                        row[f"whisker_angle|{w}|whisk_power"] = whisk_power(ts, x, t0 + wins[w][0], t0 + wins[w][1])
        iti = t0 - prev_stop if np.isfinite(prev_stop) else np.nan
        row["iti|duration"] = iti
        if np.isfinite(prev_stop):
            row["piezo_licks|iti|count"] = count(piezo, prev_stop, t0 - 0.2)
            row["piezo_licks|iti|rate"] = row["piezo_licks|iti|count"] / max(iti - 0.2, 1e-3)
            row["piezo_licks|postprev|count"] = count(piezo, prev_stop, prev_stop + 2.0)
            for e in ("jaw_dlc_licks", "tongue_dlc_licks"):
                if e in events:
                    row[f"{e}|iti|count"] = count(events[e], prev_stop, t0 - 0.2)
        before = piezo[piezo < t0]
        row["piezo_licks|pre|time_since_last"] = t0 - before[-1] if len(before) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    m = sid.split("_")[0]
    df.insert(0, "session_id", sid)
    df.insert(1, "mouse_id", m)
    df.insert(2, "cohort", cohort)
    df.insert(3, "day", int(cur.day))
    df.insert(4, "learning_trial", float(cur.learning_trial))  # NaN: criterion never reached
    df.to_parquet(out)
    return dict(session_id=sid, status="ok", n_trials=len(df), n_features=df.shape[1] - 18,
                signals=len(series), day=int(cur.day), cohort=cohort)


def safe(args):
    try:
        return process(args)
    except Exception as e:
        return dict(session_id=Path(args[0]).name, status=f"error: {type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_excel(SHEET).drop_duplicates("mouse_id")
    cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
    paths = sorted(glob.glob(str(ROOT / "combined_results_ks4" / "*" / "whisker_*" / "learning_curve" /
                                 "*_whisker_trial_learning_curve_exact.h5")))[: a.limit]
    jobs = [(p, cohort.get(Path(p).name[:5], "?")) for p in paths]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(safe, jobs):
            print(r["session_id"], r["status"], r.get("n_trials", ""), r.get("n_features", ""), flush=True)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "sessions.csv", index=False)
    print("ALL DONE")


if __name__ == "__main__":
    main()
