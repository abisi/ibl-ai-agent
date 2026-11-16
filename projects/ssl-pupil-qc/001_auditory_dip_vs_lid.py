"""Is the pupil-area dip after auditory stimuli explained by blinks / squints? (user question, 2026-11-03)

For each pilot session (pupil_qc outputs), aligned to trial start (-1 to 2 s), per trial type:
  row 1: fraction of frames per state (blink, squint, failed) -> lid / tracking events time-locked to the stimulus
  row 2: normalised height/width of the pupil (drops when the lid clips the pupil)
  row 3: mean pupil-point likelihood
  row 4: area change (baseline -1..0 s), valid frames of all trials (solid) vs trials whose whole window is valid
         (dashed); a lid artefact shrinks in the all-valid trials and coincides with row 1-2 changes.
Output: combined_results_ks4/ssl-pupil-qc/figures/001_auditory_dip_vs_lid.png (+ .pdf, table csv).
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 001_auditory_dip_vs_lid.py
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pupil_qc as pq

SESSIONS = ["AB131_20240905_123601", "AB164_20250422_115457", "MH022_20250311_140628", "AB116_20240725_094541",
            "AB102_20240309_114107"]
OUT = pq.RESULTS / "ssl-pupil-qc"
EDGES = np.arange(-1.0, 2.0 + 1e-9, 0.02)
GROUPS = {"Auditory": ["A hit", "A miss"], "Whisker": ["W hit", "W miss"], "No stim.": ["FA", "CR"]}
COLORS = {"Auditory": "#2c2cdb", "Whisker": "#f7b519", "No stim.": "0.3"}


def session_traces(sid):
    with h5py.File(pq.NWB_DIR / f"{sid}.nwb") as f:
        desc = f["session_description"][()].decode()
    mouse = sid.split("_")[0]
    d = pq.RESULTS / mouse / desc.replace("+", "") / "pupil_qc"
    fr = pd.read_parquet(d / f"{sid}_frames.parquet")
    tq = pd.read_csv(d / f"{sid}_trials_qc.csv")
    df, _, meta = pq.load_session(sid)
    meanL = np.column_stack([df[f"{p}_likelihood"] for p in pq.POINTS]).mean(1)
    t = fr.t.to_numpy()
    st, area, ratio = fr.state.to_numpy(), fr.area_clean_px2.to_numpy(), fr.hw_ratio_norm.to_numpy()
    nb = len(EDGES) - 1
    out = {}
    for g, outcomes in GROUPS.items():
        rows = tq[tq.outcome.isin(outcomes)]
        acc = {k: np.full((len(rows), nb), np.nan) for k in ("blink", "squint", "failed", "ratio", "lik", "area")}
        allvalid = np.zeros(len(rows), bool)
        for j, r in enumerate(rows.itertuples()):
            idx = np.searchsorted(t, r.start_time + EDGES)
            for k in range(nb):
                a, b = idx[k], idx[k + 1]
                if b <= a:
                    continue
                s = st[a:b]
                acc["blink"][j, k] = (s == pq.STATE["blink"]).mean()
                acc["squint"][j, k] = (s == pq.STATE["squint"]).mean()
                acc["failed"][j, k] = (s == pq.STATE["failed"]).mean()
                acc["ratio"][j, k] = np.nanmean(ratio[a:b]) if np.isfinite(ratio[a:b]).any() else np.nan
                acc["lik"][j, k] = meanL[a:b].mean()
                acc["area"][j, k] = np.nanmean(area[a:b]) if np.isfinite(area[a:b]).any() else np.nan
            allvalid[j] = (st[idx[0]:idx[-1]] == pq.STATE["valid"]).all()
        base = (EDGES[:-1] >= -1) & (EDGES[:-1] < 0)
        da = acc["area"] - np.nanmean(acc["area"][:, base], 1, keepdims=True)
        da = da / np.nanmedian(area) * 100  # % of the session median area (within-session units)
        out[g] = dict(n=len(rows), n_allvalid=int(allvalid.sum()), **{k: np.nanmean(v, 0) for k, v in acc.items()},
                      darea=np.nanmean(da, 0), darea_allvalid=np.nanmean(da[allvalid], 0) if allvalid.any() else
                      np.full(nb, np.nan))
    return desc, out


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    x = (EDGES[:-1] + EDGES[1:]) / 2
    fig, axs = plt.subplots(4, len(SESSIONS), figsize=(3.0 * len(SESSIONS), 8), sharex=True)
    rows = []
    for c, sid in enumerate(SESSIONS):
        desc, out = session_traces(sid)
        for g, o in out.items():
            col = COLORS[g]
            axs[0, c].plot(x, 100 * o["blink"], color=col, lw=1)
            axs[0, c].plot(x, 100 * o["squint"], color=col, lw=1, ls="--")
            axs[0, c].plot(x, 100 * o["failed"], color=col, lw=0.6, ls=":")
            axs[1, c].plot(x, o["ratio"], color=col, lw=1)
            axs[2, c].plot(x, o["lik"], color=col, lw=1)
            axs[3, c].plot(x, o["darea"], color=col, lw=1, label=f"{g} (n = {o['n']})")
            axs[3, c].plot(x, o["darea_allvalid"], color=col, lw=1, ls="--", label=f"  all-valid (n = {o['n_allvalid']})")
            w = (x >= 0) & (x < 0.5)
            rows.append(dict(session=sid, day=desc, group=g, n=o["n"], n_allvalid=o["n_allvalid"],
                             min_darea_0_500ms=np.nanmin(o["darea"][w]),
                             min_darea_allvalid_0_500ms=np.nanmin(o["darea_allvalid"][w]),
                             t_min_ms=1000 * x[w][np.nanargmin(o["darea"][w])],
                             max_blink_pct=100 * np.nanmax(o["blink"][w]), max_squint_pct=100 * np.nanmax(o["squint"][w]),
                             min_ratio=np.nanmin(o["ratio"][w]), min_lik=np.nanmin(o["lik"][w])))
        axs[0, c].set_title(f"{sid}\n{desc}", fontsize=8)
        for r in range(4):
            axs[r, c].axvline(0, color="k", lw=0.5)
        axs[3, c].axhline(0, color="0.7", lw=0.5)
        axs[3, c].legend(fontsize=5, frameon=False)
        axs[3, c].set_xlabel("Time from trial start (s)")
    axs[0, 0].set_ylabel("% frames\nblink (solid), squint (--), failed (:)")
    axs[1, 0].set_ylabel("Height/width (norm.)")
    axs[2, 0].set_ylabel("Mean likelihood")
    axs[3, 0].set_ylabel("Δ area (% session median)\nall trials (solid), all-valid (--)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"001_auditory_dip_vs_lid.{ext}", dpi=150)
    tab = pd.DataFrame(rows)
    (OUT / "tables").mkdir(exist_ok=True)
    tab.to_csv(OUT / "tables" / "001_auditory_dip_vs_lid.csv", index=False)
    print(tab.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
