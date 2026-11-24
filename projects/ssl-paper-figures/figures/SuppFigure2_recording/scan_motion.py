"""Scan DREDge motion of every probe (data/<mouse>/<session>/Ephys/catgt_*/*_imec*/dredge/motion/motion): per probe the
session-wide max |displacement| and the best 100-s window (largest peak-to-peak range of one depth bin's trace).
Output: <root>/combined_results_ks4/ssl-paper-figures/SuppFigure2_motion_scan.csv
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python scan_motion.py"""
import glob
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
WIN = 100.0


def one(d):
    m = Path(d)
    try:
        D = np.load(m / "displacement_seg0.npy")
        t = np.load(m / "temporal_bins_s_seg0.npy")
        y = np.load(m / "spatial_bins_um.npy")
    except Exception:
        return None
    dt = np.median(np.diff(t))
    n = max(2, int(round(WIN / dt)))
    best = (0, 0, 0)
    for j in range(D.shape[1]):
        x = D[:, j]
        if len(x) <= n:
            continue
        sw = np.lib.stride_tricks.sliding_window_view(x, n)
        rng = sw.max(1) - sw.min(1)
        i = int(np.argmax(rng))
        if rng[i] > best[0]:
            best = (float(rng[i]), float(t[i]), float(y[j]))
    p = m.parts
    return dict(path=str(m.parent.parent.parent), session=p[-7], probe=p[-4].split("_")[-1], n_tbins=len(t),
                dur_s=float(t[-1]), max_abs=float(np.abs(D).max()), p2p_session=float(D.max() - D.min()),
                win_range=best[0], win_t0=best[1], win_depth=best[2])


if __name__ == "__main__":
    dirs = glob.glob(str(ROOT / "data" / "*" / "*" / "Ephys" / "catgt_*" / "*_imec*" / "dredge" / "motion" / "motion"))
    print(len(dirs), "probes", flush=True)
    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(one, dirs) if r]
    d = pd.DataFrame(rows)
    out = ROOT / "combined_results_ks4" / "ssl-paper-figures"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "SuppFigure2_motion_scan.csv", index=False)
    print(d.describe().round(2).to_string())
    c = d[(d.max_abs < 20)].sort_values("win_range", ascending=False)
    print(c.head(15).round(2).to_string())
