"""Rank probe x depth-band examples for the motion figure: the band's tracked depth (peak of the spike depth
histogram per 5-s bin, top 50 % amplitudes) should follow the DREDge motion before correction and be flat after.
Candidates: probes with session max |displacement| < 20 um and a 100-s window range >= 10 um (scan_motion.py).
Output: <root>/combined_results_ks4/ssl-paper-figures/SuppFigure2_example_ranking.csv"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
OUTD = ROOT / "combined_results_ks4" / "ssl-paper-figures"
WIN, TB, HALF = 300.0, 5.0, 50.0


def disp_at(t, y, D, tb, yb):
    Dt = np.stack([np.interp(t, tb, D[:, j]) for j in range(len(yb))], 1)
    j = np.clip(np.searchsorted(yb, y) - 1, 0, len(yb) - 2)
    w = np.clip((y - yb[j]) / (yb[j + 1] - yb[j]), 0, 1)
    i = np.arange(len(t))
    return Dt[i, j] * (1 - w) + Dt[i, j + 1] * w


def track(t, yy, c, t0):
    tc, pk = [], []
    for b in np.arange(t0, t0 + WIN - 1e-9, TB):
        s = (t >= b) & (t < b + TB) & (np.abs(yy - c) < HALF)
        if s.sum() < 30:
            continue
        h, e = np.histogram(yy[s] - c, bins=np.arange(-HALF, HALF + 1e-9, 1.0))
        h = gaussian_filter1d(h.astype(float), 3.0)
        tc.append(b + TB / 2)
        pk.append(((e[:-1] + e[1:]) / 2)[np.argmax(h)])
    return np.array(tc), np.array(pk)


def one(row):
    try:
        d = Path(row["path"]) / "dredge" / "motion"
        fs = json.load(open(d / "parameters.json"))["sampling_frequency"]
        D = np.load(d / "motion" / "displacement_seg0.npy")
        tb = np.load(d / "motion" / "temporal_bins_s_seg0.npy")
        yb = np.load(d / "motion" / "spatial_bins_um.npy")
        # window: 300 s centred on the best 100-s window
        t0 = max(0.0, row["win_t0"] - 100.0)
        pk = np.load(d / "peaks.npy", mmap_mode="r")
        lo, hi = np.searchsorted(pk["sample_index"], [t0 * fs, (t0 + WIN) * fs])
        p = np.array(pk[lo:hi])
        y = np.array(np.load(d / "peak_locations.npy", mmap_mode="r")["y"][lo:hi])
        t = p["sample_index"] / fs
        a = np.abs(p["amplitude"])
        keep = a >= np.median(a)
        t, y = t[keep], y[keep]
        yc = y - disp_at(t, y, D, tb, yb)
        # dense, sharp bands: peaks of the corrected depth histogram (2-um bins)
        h, e = np.histogram(yc, bins=np.arange(0, 3900, 2.0))
        hs = gaussian_filter1d(h.astype(float), 2.0)
        peaks, prop = find_peaks(hs, prominence=hs.max() * 0.05, distance=30)
        out = []
        for i in peaks[np.argsort(-prop["prominences"])][:12]:
            c = (e[i] + e[i + 1]) / 2
            tc, pb = track(t, y, c, t0)
            _, pa = track(t, yc, c, t0)
            if len(tc) < 40 or len(pa) != len(pb):
                continue
            mo = disp_at(tc, np.full_like(tc, c), D, tb, yb)
            rng = np.ptp(mo)
            rb, ra = np.corrcoef(mo, pb)[0, 1], np.corrcoef(mo, pa)[0, 1]
            sb, sa = np.polyfit(mo, pb, 1)[0], np.polyfit(mo, pa, 1)[0]
            out.append(dict(path=row["path"], session=row["session"], probe=row["probe"], t0=t0, depth=c,
                            motion_range=rng, r_before=rb, r_after=ra, slope_before=sb, slope_after=sa,
                            resid_sd_after=float(np.std(pa - pa.mean())), n_bins=len(tc),
                            max_abs_session=row["max_abs"]))
        return out
    except Exception as err:  # noqa: BLE001
        print(row["session"], row["probe"], "ERROR", err, flush=True)
        return []


if __name__ == "__main__":
    sc = pd.read_csv(OUTD / "SuppFigure2_motion_scan.csv")
    cand = sc[(sc.max_abs < 20) & (sc.win_range >= 10)].sort_values("win_range", ascending=False)
    print(len(cand), "candidate probes", flush=True)
    with ProcessPoolExecutor(12) as ex:
        res = [r for rr in ex.map(one, cand.to_dict("records")) for r in rr]
    d = pd.DataFrame(res)
    nwb = {p.stem for p in (ROOT / "NWB_ks4").glob("*.nwb")}
    d["in_dataset"] = d.session.isin(nwb)
    d["score"] = d.r_before.clip(lower=0) * d.motion_range * (1 - d.r_after.abs()) * d.slope_before.clip(0, 1.2)
    d = d.sort_values("score", ascending=False)
    d.to_csv(OUTD / "SuppFigure2_example_ranking.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.head(20).round(2).drop(columns=["path"]).to_string())
