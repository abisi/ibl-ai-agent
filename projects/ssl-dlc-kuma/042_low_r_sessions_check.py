"""Which tracking is right where old and new disagree most (036 sessions with the lowest r)? (2026-11-11)

Independent of either model: the behaviour each trace should follow.
  side jaw_distance:  locked to piezo lick bouts (onset = first lick after >= 1 s without licks): response =
                      mean(0..150 ms) - mean(-400..-200 ms), in units of the trace's SD (robust: MAD x 1.4826)
  top whisker_angle:  locked to whisker stimuli (trials.whisker_stim_time): |mean(0..40 ms) - mean(-100..0 ms)| / SD
Old and new rebuilt as in 036 (025 build_nwb_like, dlc_helpers series), session NWB timestamps; frames by timestamp.
A real trace has a clearly larger locked response; noise or a wrong keypoint has ~0.
Output: inference_assessment/low_r_check.csv (+ figure low_r_check.png: mean locked traces old vs new per session)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 042_low_r_sessions_check.py [--n 25]
"""

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("a036", "036_inference_assessment.py")
ba, pv = A.ba, A.pv
MAIN = {"side": "jaw_distance", "top": "whisker_angle"}
WIN = {"side": (-0.6, 0.6), "top": (-0.3, 0.3)}


def events(session, view):
    with h5py.File(ba.NWB / f"{session}.nwb", "r") as f:
        if view == "top":
            t = f["intervals/trials/whisker_stim_time"][:]
            ws = f["intervals/trials/whisker_stim"][:]
            return t[(ws == 1) & np.isfinite(t)]
        lk = f["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:]
    return lk[np.r_[True, np.diff(lk) >= 1.0]]


def locked(x, t, ev, win):
    rate = 1 / np.median(np.diff(t[:2000]))
    lags = np.arange(int(win[0] * rate), int(win[1] * rate))
    i0 = np.searchsorted(t, ev)
    i0 = i0[(i0 + lags[0] >= 0) & (i0 + lags[-1] < len(x))]
    m = np.nanmean(x[i0[:, None] + lags[None, :]], 0)
    return lags / rate, m


def score(lag, m, x, view):
    sd = 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
    if view == "side":
        resp = np.nanmean(m[(lag >= 0) & (lag < 0.15)]) - np.nanmean(m[(lag >= -0.4) & (lag < -0.2)])
    else:
        resp = abs(np.nanmean(m[(lag >= 0) & (lag < 0.04)]) - np.nanmean(m[(lag >= -0.1) & (lag < 0)]))
    return resp / sd if sd > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    kin = pd.read_csv(A.OUT / "kinematics.csv")
    main_q = kin[kin.quantity.isin(MAIN.values())].copy()
    main_q = main_q[((main_q.view == "side") & (main_q.quantity == "jaw_distance"))
                    | ((main_q.view == "top") & (main_q.quantity == "whisker_angle"))]
    low = main_q.dropna(subset=["r"]).sort_values("r").head(args.n)
    rows, traces = [], []
    for r in low.itertuples():
        job = json.loads((A.OUT / "rows" / f"{r.session_id}_{r.view}.json").read_text())
        ev = events(r.session_id, r.view)
        res = dict(session_id=r.session_id, view=r.view, quantity=MAIN[r.view], r_old_new=r.r, n_events=len(ev))
        with tempfile.TemporaryDirectory(dir=A.SHM) as tmp:
            for name in ("old", "new"):
                h5 = job.get(f"{name}_h5")
                p = Path(tmp) / f"{name}.h5"
                ba.build_nwb_like(Path(h5), r.session_id, r.view, p)
                v = ba.load_view(p, r.view, r.session_id)
                x = np.asarray(ba.series(v, MAIN[r.view]), float)
                t = np.asarray(v.timestamps, float)[:len(x)]
                lag, m = locked(x, t, ev, WIN[r.view])
                res[f"locked_{name}"] = score(lag, m, x, r.view)
                traces.append((r.session_id, r.view, name, lag, (m - np.nanmean(m[lag < 0])) /
                               (1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))))
        res["better"] = "new" if res["locked_new"] > 1.2 * res["locked_old"] else (
            "old" if res["locked_old"] > 1.2 * res["locked_new"] else "similar")
        rows.append(res)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(A.OUT / "low_r_check.csv", index=False)
    n = len(rows)
    fig, axs = plt.subplots(int(np.ceil(n / 5)), 5, figsize=(14, 2.3 * np.ceil(n / 5)), squeeze=False)
    for ax, (s, v, name, lag, m) in zip(np.repeat(axs.flat, 2), traces):
        ax.plot(lag * 1000, m, color={"old": "#d6604d", "new": "#2ca02c"}[name], lw=1, label=name)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_title(f"{s[:14]} {v} ({'lick' if v == 'side' else 'whisker stim'})", fontsize=7)
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6, frameon=False)
    fig.supxlabel("Time from event (ms)")
    fig.supylabel("Event-locked mean (robust SD units)")
    fig.tight_layout()
    fig.savefig(A.OUT / "figures" / "low_r_check.png", dpi=150)
    print(out.better.value_counts().to_string())


if __name__ == "__main__":
    main()
