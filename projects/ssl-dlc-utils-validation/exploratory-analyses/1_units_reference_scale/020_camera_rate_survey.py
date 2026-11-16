"""How often does the true camera frame rate differ from the metadata frame rate? (user 2026-11-10)

Code that maps session times to video frames with the metadata rate (frame = (t - video start) x camera_freq, e.g.
behaviour_analysis find_dlc_jaw_onset_times before 2026-11-10) drifts by the difference between the true camera clock
and camera_freq, and is offset if the video start is taken from the other camera. dlc_helpers locates every frame by
its own camera timestamp and is not affected; this survey sizes the error of the nominal-rate mapping.

Every NWB_ks4 session file (no *_backup_*) with DLC, per camera (dlc_helpers._camera_groups: side = jaw_angle group,
top = whisker_angle group):
  meta_fps           session metadata camera_freq (NWB_reader.get_video_sampling_rate)
  true_fps           (n timestamps - 1) / (last - first timestamp): mean rate over the session (includes missed pulses)
  median_dt_fps      1 / median inter-frame interval (what dlc_helpers.frame_rate returns)
  drift_ms_end       time error of the nominal mapping at the last frame: (n - 1) / meta_fps - (last - first), ms
  drift_ms_per_hour  (true_fps / meta_fps - 1) x 3.6e6
  n_gaps             intervals > 1.5 median intervals (missed pulses / dropped frames)
  start_offset_ms    this camera's first timestamp - the other camera's first timestamp
Output: combined_results_ks4/ssl-dlc-utils-validation/tables/020_camera_rates.csv, figures/020_camera_rates.png
Run on haas: nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 020_camera_rate_survey.py
"""

import ast
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1]  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))
from common import NWB_DIR, OUT, dh  # noqa: E402

TS = "processing/behavior/BehavioralTimeSeries"


def camera_freq(f):
    """camera_freq of the session metadata (NWB_reader.get_session_metadata: ast.literal_eval of
    experiment_description)."""
    try:
        raw = f["general/experiment_description"][()]
        meta = ast.literal_eval(raw.decode() if isinstance(raw, bytes) else str(raw))
        return float(meta["camera_freq"])
    except (KeyError, ValueError, SyntaxError, TypeError):
        return np.nan


def main():
    files = sorted(p for p in NWB_DIR.glob("*.nwb") if "_backup_" not in p.name)
    rows = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                if TS not in f:
                    continue
                cams = dh._camera_groups(f[TS])
                meta = camera_freq(f)
                ts = {v: f[TS][names[0]]["timestamps"][:] for v, names in cams.items()}
        except OSError as err:
            rows.append({"session_id": p.stem, "error": str(err)})
            continue
        for v, t in ts.items():
            if len(t) < 2:
                continue
            dt = np.diff(t)
            dur = t[-1] - t[0]
            other = [o for o in ts if o != v]
            rows.append({"session_id": p.stem, "mouse_id": p.stem[:5], "camera": v, "n_frames": len(t),
                         "duration_s": dur, "meta_fps": meta, "true_fps": (len(t) - 1) / dur,
                         "median_dt_fps": 1 / np.median(dt),
                         "drift_ms_end": 1000 * ((len(t) - 1) / meta - dur) if np.isfinite(meta) else np.nan,
                         "drift_ms_per_hour": ((len(t) - 1) / dur / meta - 1) * 3.6e6 if np.isfinite(meta) else np.nan,
                         "n_gaps": int((dt > 1.5 * np.median(dt)).sum()),
                         "start_offset_ms": 1000 * (t[0] - ts[other[0]][0]) if other else np.nan})
    d = pd.DataFrame(rows)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "tables" / "020_camera_rates.csv", index=False)
    ok = d.dropna(subset=["drift_ms_end"])
    print(f"{d.session_id.nunique()} sessions, {len(d)} cameras; metadata fps values: "
          f"{ok.meta_fps.value_counts().to_dict()}")
    for v, g in ok.groupby("camera"):
        a = g.drift_ms_end.abs()
        print(f"{v}: |drift at end| median {a.median():.0f} ms, 90th pct {a.quantile(0.9):.0f} ms, max {a.max():.0f} ms; "
              f"> 5 ms (1 frame) {100 * (a > 5).mean():.0f}%, > 20 ms {100 * (a > 20).mean():.0f}%, "
              f"> 100 ms {100 * (a > 100).mean():.0f}%; true fps median {g.true_fps.median():.4f}; "
              f"sessions with gaps {100 * (g.n_gaps > 0).mean():.0f}%; |start offset| > 5 ms "
              f"{100 * (g.start_offset_ms.abs() > 5).mean():.0f}% (median {g.start_offset_ms.abs().median():.1f} ms)")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for v, col in (("side", "0.3"), ("top", "#9e9ac8")):
        g = ok[ok.camera == v]
        axs[0].hist(g.true_fps - g.meta_fps, bins=60, color=col, alpha=0.7, label=v)
        axs[1].scatter(g.duration_s / 60, g.drift_ms_end, s=5, color=col, label=v)
        axs[2].hist(g.start_offset_ms.clip(-300, 300).dropna(), bins=60, color=col, alpha=0.7, label=v)
    axs[0].set_xlabel("true fps - metadata fps"); axs[0].set_ylabel("Cameras"); axs[0].legend(frameon=False)
    axs[1].set_xlabel("Video duration (min)"); axs[1].set_ylabel("Nominal-rate time error at last frame (ms)")
    axs[1].set_yscale("symlog", linthresh=10)
    axs[2].set_xlabel("Camera start - other camera start (ms, clipped)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "020_camera_rates.png", dpi=150)


if __name__ == "__main__":
    main()
