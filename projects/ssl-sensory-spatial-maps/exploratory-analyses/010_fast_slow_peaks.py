"""010 -- Fast (peak < 20 ms) and slow (peak 20-100 ms) transient sensory responses, from the 001 latency table.

User question (2026-11-18): is there evidence for both fast peaks (< 20 ms) and slower peaks (< 100 ms) in sensory
responses (transients only, i.e. using latency)?
Input:  ssl-sensory-spatial-maps/unit_latency.parquet (001: responsive = ROC whisker_active / auditory_active significant,
        response window 5-35 ms; 1-ms PSTH, sigma 2 ms, peak searched 5-100 ms, latency = half-time to peak);
        _roc_stage_analysis/units.parquet for area_group, cohort, stage (first row per unit key).
Units:  excited responses (sign +). Peak reliability: z = peak / noise SD, noise SD = Poisson SD of the smoothed PSTH at
        the pre-trial rate, sqrt(rate / (n_trials * 1 ms)) / sqrt(2 sqrt(pi) sigma_bins). Main set z >= Z_MIN (a noise
        maximum over the ~95 searched bins reaches z ~ 3); all units also reported.
Classes: fast t_peak < 20 ms; slow 20 <= t_peak < 95 ms; edge t_peak >= 95 ms (still rising at the search limit).
Tests:  Hartigan dip test on log10(t_peak) and 1- vs 2-component Gaussian mixture (BIC) on log10(t_peak), all units and
        per area group (>= 100 units); co-occurrence of both classes per session x area group (>= 5 units).
Outputs: ssl-sensory-spatial-maps/fast_slow_peaks/{units.parquet, area_groups.csv, bimodality.csv, cooccurrence.csv,
         fast_slow_peaks.{png,pdf,svg}}
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 010_fast_slow_peaks.py
"""
import json
import pathlib
import time

import diptest
import matplotlib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = RES / "ssl-sensory-spatial-maps" / "fast_slow_peaks"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
FAST, EDGE, Z_MIN, SIGMA_BINS = 20.0, 95.0, 5.0, 2.0
COL = {"whisker": "#f7b519", "auditory": "#2c2cdb"}
MIN_AREA, MIN_CELL = 100, 5


def load():
    L = pd.read_parquet(RES / "ssl-sensory-spatial-maps" / "unit_latency.parquet")
    U = pd.read_parquet(RES / "_roc_stage_analysis" / "units.parquet",
                        columns=KEYS + ["area_group", "area_acronym_custom", "cohort", "stage", "day"])
    U["cluster_id"] = U.cluster_id.astype(str)
    L["cluster_id"] = L.cluster_id.astype(str)
    L = L.merge(U.drop_duplicates(KEYS), on=KEYS, how="left")
    rows = []
    for m in ("whisker", "auditory"):
        x = L[L[f"sign_{m}"] == 1].copy()
        rate = np.maximum(x[f"pretrial_baseline_{m}_hz"], 0.5)
        sd = np.sqrt(rate / (x[f"n_trials_{m}"] * 0.001)) / np.sqrt(2 * np.sqrt(np.pi) * SIGMA_BINS)
        rows.append(pd.DataFrame(dict(
            **{k: x[k] for k in KEYS + ["area_group", "area_acronym_custom", "cohort", "stage", "day"]},
            modality=m, t_peak_ms=x[f"t_peak_{m}_ms"], latency_ms=x[f"latency_{m}_ms"], peak_hz=x[f"peak_{m}_hz"],
            z_peak=x[f"peak_{m}_hz"] / sd)))
    d = pd.concat(rows, ignore_index=True).dropna(subset=["t_peak_ms"])
    d["cls"] = np.select([d.t_peak_ms < FAST, d.t_peak_ms < EDGE], ["fast", "slow"], "edge")
    return d


def kde_peaks(v, lo=5.0, hi=60.0, bw=1.5):
    """local maxima (ms) of a Gaussian KDE (bandwidth bw ms) of peak times in lo..hi, with their density dips between"""
    v = v[np.isfinite(v)]
    g = np.arange(lo, hi + 0.01, 0.1)
    k = np.exp(-0.5 * ((g[:, None] - v[None, :]) / bw) ** 2).sum(1)
    i = np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] >= k[2:])) + 1
    i = i[k[i] > 0.15 * k.max()]
    dips = [g[a + np.argmin(k[a:b])] for a, b in zip(i[:-1], i[1:])]
    return "/".join(f"{x:.1f}" for x in g[i]), "/".join(f"{x:.1f}" for x in dips)


def bimodality(v):
    pk, dp = kde_peaks(v)
    v = np.log10(v[np.isfinite(v)])
    dip, p = diptest.diptest(v)
    X = v.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=0).fit(X)
    g2 = GaussianMixture(2, random_state=0, n_init=5).fit(X)
    mu = np.sort(10 ** g2.means_.ravel())
    w = g2.weights_[np.argsort(g2.means_.ravel())]
    return dict(n=len(v), kde_peaks_ms=pk, kde_dips_ms=dp, dip=dip, p_dip=p, dBIC_1minus2=g1.bic(X) - g2.bic(X), mode1_ms=mu[0], mode2_ms=mu[1],
                w_mode2=w[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    d.to_parquet(OUT / "units.parquet")
    order = [g for g in au.get_area_group_custom_order() if g in set(d.area_group.dropna())]
    pal = au.get_custom_area_groups_colors()

    B, A, C = [], [], []
    for m in ("whisker", "auditory"):
        for zset, x0 in (("all", d), (f"z>={Z_MIN:g}", d[d.z_peak >= Z_MIN])):
            x = x0[(x0.modality == m) & (x0.cls != "edge")]
            B.append(dict(modality=m, units=zset, area_group="all", **bimodality(x.t_peak_ms.to_numpy())))
            for g, y in x.groupby("area_group"):
                if len(y) >= MIN_AREA:
                    B.append(dict(modality=m, units=zset, area_group=g, **bimodality(y.t_peak_ms.to_numpy())))
            x = x0[x0.modality == m]
            for g, y in x.groupby("area_group"):
                A.append(dict(modality=m, units=zset, area_group=g, n=len(y), frac_fast=(y.cls == "fast").mean(),
                              frac_slow=(y.cls == "slow").mean(), frac_edge=(y.cls == "edge").mean(),
                              med_tpeak_fast=y.t_peak_ms[y.cls == "fast"].median(),
                              med_tpeak_slow=y.t_peak_ms[y.cls == "slow"].median(),
                              med_lat_fast=y.latency_ms[y.cls == "fast"].median(),
                              med_lat_slow=y.latency_ms[y.cls == "slow"].median()))
            for (s, g), y in x.groupby(["session_id", "area_group"]):
                if len(y) >= MIN_CELL:
                    C.append(dict(modality=m, units=zset, session_id=s, area_group=g, n=len(y),
                                  n_fast=(y.cls == "fast").sum(), n_slow=(y.cls == "slow").sum()))
    B, A, C = pd.DataFrame(B), pd.DataFrame(A), pd.DataFrame(C)
    B.to_csv(OUT / "bimodality.csv", index=False)
    A.to_csv(OUT / "area_groups.csv", index=False)
    C.to_csv(OUT / "cooccurrence.csv", index=False)

    pd.set_option("display.width", 220)
    print(B[B.area_group == "all"].round(4).to_string())
    print(B[B.area_group != "all"].round(3).to_string())
    print(A[A.units != "all"].round(3).to_string())
    c = C.assign(both=(C.n_fast > 0) & (C.n_slow > 0))
    print(c.groupby(["modality", "units"]).both.agg(["mean", "size"]).round(3))

    # figure: row 1 peak-time histograms (all vs reliable); row 2 fraction fast / slow per area group
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6.5,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.4), gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.5, wspace=0.75))
    bins = np.arange(4.5, 101.5, 1)
    for j, m in enumerate(("whisker", "auditory")):
        a = ax[0, j]
        x = d[d.modality == m]
        a.hist(x.t_peak_ms, bins, color="0.75", lw=0, edgecolor="none", label=f"all excited (n = {len(x):,})")
        y = x[x.z_peak >= Z_MIN]
        a.hist(y.t_peak_ms, bins, color=COL[m], lw=0, edgecolor="none", label=f"peak z ≥ {Z_MIN:g} (n = {len(y):,})")
        a.axvline(FAST, color="k", lw=0.6, ls="--")
        a.axvline(EDGE, color="0.4", lw=0.6, ls=":")
        r = B[(B.modality == m) & (B.units != "all") & (B.area_group == "all")].iloc[0]
        a.set_title(f"{m.capitalize()} responses: time of the response peak\n"
                    f"density peaks {r.kde_peaks_ms.replace('/', ', ')} ms (dip at {r.kde_dips_ms.replace('/', ', ')} ms; "
                    f"peak z ≥ {Z_MIN:g}); dip test p < 0.001", fontsize=7)
        a.set_xlabel("Peak time after stimulus onset (ms)")
        a.set_ylabel("Neurons per 1-ms bin")
        a.set_xlim(3, 102)
        a.legend(frameon=False, fontsize=6)
        a = ax[1, j]
        t = A[(A.modality == m) & (A.units != "all") & (A.n >= 30)].set_index("area_group")
        t = t.reindex([g for g in order if g in t.index])
        yy = np.arange(len(t))[::-1]
        a.barh(yy, t.frac_fast, color=COL[m], lw=0, label="fast (peak < 20 ms)")
        a.barh(yy, t.frac_slow, left=t.frac_fast, color=matplotlib.colors.to_rgba(COL[m], 0.35), lw=0,
               label="slow (peak 20-95 ms)")
        a.barh(yy, t.frac_edge, left=t.frac_fast + t.frac_slow, color="0.8", lw=0, label="peak at the 95-100 ms limit")
        a.set_yticks(yy)
        a.set_yticklabels([f"{g} (n = {int(n)})" for g, n in zip(t.index, t.n)], fontsize=5.5)
        for lab, g in zip(a.get_yticklabels(), t.index):
            lab.set_color(pal[g])
        a.set_xlim(-0.02, 1.02)
        a.set_xlabel(f"Fraction of excited neurons (peak z ≥ {Z_MIN:g})")
        a.set_title(f"{m.capitalize()}: fast vs slow peaks per area group", fontsize=7)
        a.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.suptitle("Fast and slow transient sensory responses (active task, all sessions pooled; responsive = ROC 5-35 ms)",
                 fontsize=8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fast_slow_peaks.{ext}", dpi=300, bbox_inches="tight")
    json.dump(dict(script="010_fast_slow_peaks.py", fast_ms=FAST, edge_ms=EDGE, z_min=Z_MIN, sigma_bins=SIGMA_BINS,
                   input="unit_latency.parquet (001)", date=time.strftime("%Y-%m-%d %H:%M")),
              open(OUT / "provenance.json", "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
