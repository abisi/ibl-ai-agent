"""Is the fast early pupil dip after stimuli a real constriction or the eyelid? (user question, 2026-11-09)

The lid clips the top / bottom of the pupil but not its sides: a real constriction shrinks the horizontal diameter
(pupil_left <-> pupil_right) as much as the vertical one; a lid artefact shrinks the area and the vertical diameter
(pupil_top <-> pupil_bottom) only. Per pilot session (001), aligned to trial start (-1 to 2 s, 20 ms bins), whisker and
auditory trials (hits + misses), no-stim trials as reference:
  col 1: 4-point area (as the NWB pupil_area), frames with all four points >= cutoff
  col 2: horizontal diameter, frames with left and right >= cutoff
  col 3: vertical diameter, frames with top and bottom >= cutoff
  col 4: % of frames valid for the area at each cutoff
Change from the trial baseline (-1..0 s) in % of the session median; likelihood cutoffs 0.1 (dlc_helpers), 0.5
(converter), 0.8. Raw NWB keypoints (THREE_POINT correction applied by pupil_qc.load_session), no QC states, no
smoothing, so the cutoff is the only frame selection.
Output: combined_results_ks4/ssl-pupil-qc/figures/006_dip_horizontal_diameter.png (+ .pdf), tables/006_dip.csv
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 006_dip_horizontal_diameter.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pupil_qc as pq  # noqa: E402

SESSIONS = ["AB131_20240905_123601", "AB164_20250422_115457", "MH022_20250311_140628", "AB116_20240725_094541",
            "AB102_20240309_114107"]
OUT = pq.RESULTS / "ssl-pupil-qc"
EDGES = np.arange(-1.0, 2.0 + 1e-9, 0.02)
GROUPS = {"Auditory": ["A hit", "A miss"], "Whisker": ["W hit", "W miss"], "No stim.": ["FA", "CR"]}
COLORS = {"Auditory": "#2c2cdb", "Whisker": "#f7b519", "No stim.": "0.3"}
CUTOFFS = {0.1: "-", 0.5: "--", 0.8: ":"}
MEASURES = {"area": ("4-point area", ["top", "left", "right", "bottom"]),
            "diam_h": ("Horizontal diameter", ["left", "right"]),
            "diam_v": ("Vertical diameter", ["top", "bottom"])}


def measures(df):
    p = {q: (df[f"{q}_x"].to_numpy(), df[f"{q}_y"].to_numpy()) for q in pq.POINTS}
    order = ["top", "right", "bottom", "left"]
    a = sum(p[order[i]][0] * p[order[(i + 1) % 4]][1] - p[order[i]][1] * p[order[(i + 1) % 4]][0] for i in range(4))
    return {"area": 0.5 * np.abs(a),
            "diam_h": np.hypot(p["left"][0] - p["right"][0], p["left"][1] - p["right"][1]),
            "diam_v": np.hypot(p["top"][0] - p["bottom"][0], p["top"][1] - p["bottom"][1])}


def locked(x, t, starts):
    """Trials x bins mean of x (NaN-aware)."""
    nb = len(EDGES) - 1
    out = np.full((len(starts), nb), np.nan)
    for j, s in enumerate(starts):
        idx = np.searchsorted(t, s + EDGES)
        for k in range(nb):
            seg = x[idx[k]:idx[k + 1]]
            if len(seg) and np.isfinite(seg).any():
                out[j, k] = np.nanmean(seg)
    return out


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    x = (EDGES[:-1] + EDGES[1:]) / 2
    base = x < 0
    early = (x >= 0) & (x < 0.5)
    fig, axs = plt.subplots(len(SESSIONS), 4, figsize=(13, 2.4 * len(SESSIONS)), sharex=True)
    rows = []
    for r, sid in enumerate(SESSIONS):
        df, trials, meta = pq.load_session(sid)
        t = df.t.to_numpy()
        m = measures(df)
        lik = {q: df[f"{q}_likelihood"].to_numpy() for q in pq.POINTS}
        for g, outcomes in GROUPS.items():
            starts = trials.loc[trials.outcome.isin(outcomes), "start_time"].to_numpy()
            for cut, ls in CUTOFFS.items():
                if g == "No stim." and cut != 0.1:
                    continue
                for c, (name, (label, pts)) in enumerate(MEASURES.items()):
                    ok = np.all([lik[q] >= cut for q in pts], axis=0)
                    v = np.where(ok, m[name], np.nan)
                    tr = locked(v, t, starts)
                    d = (tr - np.nanmean(tr[:, base], 1, keepdims=True)) / np.nanmedian(v) * 100
                    trace = np.nanmean(d, 0)
                    axs[r, c].plot(x, trace, color=COLORS[g], ls=ls, lw=0.9)
                    rows.append(dict(session=sid, day=meta["description"], group=g, cutoff=cut, measure=name,
                                     n_trials=len(starts), min_0_500ms=np.nanmin(trace[early]),
                                     t_min_ms=1000 * x[early][np.nanargmin(trace[early])]))
                    if name == "area":
                        val = locked(ok.astype(float), t, starts)
                        axs[r, 3].plot(x, 100 * np.nanmean(val, 0), color=COLORS[g], ls=ls, lw=0.9)
        axs[r, 0].set_ylabel(f"{sid}\n{meta['description']}\nΔ (% session median)", fontsize=7)
        axs[r, 3].set_ylabel("% frames valid (area)", fontsize=7)
        for c in range(4):
            axs[r, c].axvline(0, color="k", lw=0.5)
            if c < 3:
                axs[r, c].axhline(0, color="0.7", lw=0.5)
    for c, (label, _) in enumerate(MEASURES.values()):
        axs[0, c].set_title(label, fontsize=9)
    axs[0, 3].set_title("Frames valid for the area", fontsize=9)
    for c in range(4):
        axs[-1, c].set_xlabel("Time from trial start (s)")
    handles = [plt.Line2D([], [], color=col, label=g) for g, col in COLORS.items()] + \
              [plt.Line2D([], [], color="k", ls=ls, label=f"cutoff {cut}") for cut, ls in CUTOFFS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"006_dip_horizontal_diameter.{ext}", dpi=150)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tables" / "006_dip.csv", index=False)
    w = tab[tab.group != "No stim."].pivot_table(index=["session", "group", "cutoff"], columns="measure",
                                                 values="min_0_500ms")
    print(w.round(1).to_string())


if __name__ == "__main__":
    main()
