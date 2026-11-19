"""012 -- Excited sensory neurons whose response peaks late (50-95 ms) (user 2026-11-18). Uses the 011 PSTHs.

Units: 010/011 excited, ROC-responsive (5-35 ms window), peak z >= 5; late = t_peak 50-95 ms; compared with early
(peak < 50 ms). Sustained index = mean(100-200 ms) / peak (pre-trial baseline subtracted).
Outputs (fast_slow_peaks/): late_peaks_areas.csv; late_peaks_<modality>.{png,pdf,svg}:
  left  single-neuron heatmap of peak-normalised PSTHs of the late neurons, sorted by peak time
  right mean +- s.e.m. over sessions of late neurons per area group (>= 3 sessions, >= 20 neurons)
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 012_late_peaks.py
"""
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402

OUT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-sensory-spatial-maps/fast_slow_peaks")
COL = {"whisker": "#f7b519", "auditory": "#2c2cdb"}
LATE = (50.0, 95.0)


def main():
    z = np.load(OUT / "psths.npz")
    P, t = z["psth"], z["t_ms"]
    R = pd.read_parquet(OUT / "psth_units.parquet").sort_values("row").reset_index(drop=True)
    U = pd.read_parquet(OUT / "units.parquet")      # all excited incl. the 95-100 ms edge class (no PSTH for those)
    Y = P - R.base_pretrial_hz.to_numpy()[:, None]
    pk = np.array([y[(t >= 5) & (t <= 100)].max() for y in Y])
    R["sust"] = Y[:, (t >= 100) & (t < 200)].mean(1) / np.where(pk > 0, pk, np.nan)
    R["late"] = R.t_peak_ms.between(*LATE)
    Yn = Y / np.where(pk > 0, pk, np.nan)[:, None]

    rows = []
    for (m, g), x in R.groupby(["modality", "area_group"]):
        u = U[(U.modality == m) & (U.area_group == g) & (U.z_peak >= 5)]
        rows.append(dict(modality=m, area_group=g, n_excited=len(u), n_late=int(x.late.sum()),
                         frac_late=x.late.sum() / max(len(u), 1), sessions_late=x[x.late].session_id.nunique(),
                         sust_late_median=x[x.late].sust.median(), sust_early_median=x[x.t_peak_ms < 50].sust.median()))
    A = pd.DataFrame(rows).sort_values(["modality", "frac_late"], ascending=[True, False])
    A.to_csv(OUT / "late_peaks_areas.csv", index=False)
    pd.set_option("display.width", 200)
    print(A.round(3).to_string())
    for m in ("whisker", "auditory"):
        x = R[(R.modality == m) & R.late]
        e = U[(U.modality == m) & (U.z_peak >= 5)]
        print(m, "late", len(x), "of", len(e), f"({len(x) / len(e):.3f}); edge 95-100 ms:", int((e.t_peak_ms >= 95).sum()),
              "; sust median late", round(x.sust.median(), 3),
              "; frac transient (<0.25)", round((x.sust < 0.25).mean(), 3))

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    pal = au.get_custom_area_groups_colors()
    order = au.get_area_group_custom_order()
    show = (t >= -50) & (t <= 200)
    for m in ("whisker", "auditory"):
        x = R[(R.modality == m) & R.late].sort_values("t_peak_ms")
        ns = x.groupby("area_group").agg(n=("row", "size"), s=("session_id", "nunique"))
        groups = [g for g in order if g in ns.index and ns.loc[g, "s"] >= 3 and ns.loc[g, "n"] >= 20]
        nc = 4
        nr = max(2, int(np.ceil(len(groups) / (nc - 1))))
        fig = plt.figure(figsize=(7.4, 1.5 * nr + 0.4))
        gs = fig.add_gridspec(nr, nc, width_ratios=[1.3, 1, 1, 1], hspace=0.9, wspace=0.55)
        a = fig.add_subplot(gs[:, 0])
        im = a.imshow(Yn[x.row.to_numpy()][:, show], aspect="auto", cmap="magma", vmin=-0.2, vmax=1,
                      extent=(t[show][0], t[show][-1], len(x), 0), interpolation="none", rasterized=True)
        a.axvline(0, color="w", lw=0.5, ls="--")
        a.set_xlabel("Time from stimulus onset (ms)")
        a.set_ylabel("Neurons, sorted by peak time")
        a.set_title(f"{m.capitalize()}: {len(x):,} neurons peaking {LATE[0]:.0f}-{LATE[1]:.0f} ms", fontsize=6.5)
        cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.04, pad=0.08)
        cb.set_label("Rate / unit's peak (pre-trial baseline subtracted)", fontsize=5.5)
        for i, g in enumerate(groups):
            a = fig.add_subplot(gs[i // (nc - 1), 1 + i % (nc - 1)])
            s = x[x.area_group == g]
            S = np.stack([np.nanmean(Yn[h.row.to_numpy()], 0) for _, h in s.groupby("session_id")])
            S = S[np.isfinite(S).all(1)]
            mu, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))
            a.fill_between(t[show], (mu - se)[show], (mu + se)[show], color=COL[m], alpha=0.3, lw=0, edgecolor="none")
            a.plot(t[show], mu[show], color=COL[m], lw=0.8)
            a.axvline(0, color="0.3", lw=0.5, ls="--")
            a.axvspan(*LATE, facecolor="0.92", lw=0, edgecolor="none", zorder=0)
            a.axhline(0, color="0.6", lw=0.4, ls=":")
            a.set_title(f"{g}\n{len(s)} neurons / {len(S)} sessions", fontsize=5.8, color=pal[g])
            a.set_xlim(-50, 200)
            a.set_xlabel("Time from stimulus (ms)", fontsize=5.5)
            a.set_ylabel("Rate / unit's peak", fontsize=5.5)
        fig.suptitle(f"{m.capitalize()} stimulus: excited responsive neurons (ROC 5-35 ms, peak z ≥ 5) whose response "
                     f"peaks at {LATE[0]:.0f}-{LATE[1]:.0f} ms (grey band); right: mean ± s.e.m. over sessions per area "
                     "group (≥ 3 sessions, ≥ 20 neurons)", fontsize=6.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"late_peaks_{m}.{ext}", dpi=300 if ext == "png" else 1200, bbox_inches="tight")
        plt.close(fig)
    print("written", OUT)


if __name__ == "__main__":
    main()
