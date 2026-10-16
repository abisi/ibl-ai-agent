"""010 -- Sanity check (user 2026-10-16): does stimulus-presence decoding per area relate to the anatomical input from the
whisker and auditory cortices? Allen Mouse Connectivity part (Liu et al. 2024 wS1 / wS2 data: pending, see README).
Anatomy per area: Allen projection density (ssl-sensory-spatial-maps 002: line-balanced, normalised per experiment, smoothed
50 um, right hemisphere, source area itself excluded) of SSp-bfd, SSs, AUDp, AUD-sec and the merged whisker (SSp-bfd + SSs) and
auditory (AUDp + AUD-sec) sources, sampled at the CCF position of every good + mua unit of the area (ML folded onto the right
hemisphere, 50-um voxels) and averaged over the area's units ("input at the recorded sites"); also the fraction of the
area's units inside the source's 70 % projection zone (zone70, "locality").
Decoding per area (N = 200; final run if complete, else the N-sweep rows; 006 source()): onset (zoom, 004 rule; unreliable
onsets excluded from onset tests), early corrected accuracy 5-50 ms, peak corrected accuracy (009).
Tests across areas, per level (area groups / fine areas), excluding the source areas (no density there by construction):
Spearman rho (and Pearson r on log density) of each decoding measure with the matching source (whisker target ~ whisker
density, auditory target ~ auditory density) and with the other source as a specificity control; per epoch.
Output: <HOME>/anatomy/anatomy_by_area.csv, anatomy_tests.csv, figures/anatomy_vs_decoding_<level>.{png,pdf,svg}
Run (haas): PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python .../010_anatomy_link.py
"""
import importlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = importlib.import_module("_style")
AR = importlib.import_module("_areas")
HOME = AR.HOME
OUTA = HOME / "anatomy"
ZONES = AR.RES / "_sensory_spatial_maps" / "projection_zones.npz"
RES_UM, MID_UM = 50, 5700
SOURCES = ["whisker", "auditory", "SSp-bfd", "SSs", "AUDp", "AUD-sec"]
SOURCE_AREAS = {"whisker": {"Somatosensory-whisker", "SSp-bfd", "SSs"}, "auditory": {"Auditory areas", "AUDp", "AUDd", "AUDv", "AUD"}}
N_MAIN = 200
TARGETS, EPOCHS = ("whisker", "auditory"), ("active", "passive")
MEASURES = {"early": "Early accuracy 5-50 ms", "peak": "Peak accuracy", "onset": "Onset (ms)"}


def anatomy():
    """per area (both levels): mean density of each source at the area's unit positions, fraction inside zone70"""
    z = np.load(ZONES)
    U = pd.read_parquet(AR.UNITS, columns=AR.KEYS + ["quality_label", "area_group", "area_acronym_custom",
                                                     "ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"])
    U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(AR.KEYS).dropna(subset=["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"])
    ml = MID_UM + np.abs(U.ccf_atlas_ml.to_numpy(float) - MID_UM)
    ijk = np.round(np.c_[U.ccf_atlas_ap.to_numpy(float), U.ccf_atlas_dv.to_numpy(float), ml] / RES_UM).astype(int)
    shp = z["density_whisker"].shape
    ok = np.all((ijk >= 0) & (ijk < np.array(shp)), axis=1)
    U, ijk = U[ok].copy(), ijk[ok]
    for s in SOURCES:
        U[f"dens_{s}"] = z[f"density_{s}"][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        U[f"zone_{s}"] = z[f"zone70_{s}"][ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(float)
    rows = []
    for level, col in (("area_group", "area_group"), ("area_acronym_custom", "area_acronym_custom")):
        for a, g in U[U[col].isin(AR.LEVELS[level])].groupby(col):
            rows.append(dict(level=level, area=a, n_units=len(g), **{f"dens_{s}": g[f"dens_{s}"].mean() for s in SOURCES},
                             **{f"zone_{s}": g[f"zone_{s}"].mean() for s in SOURCES}))
    return pd.DataFrame(rows)


def decoding():
    """per target x epoch x level x area: onset, early and peak accuracy at N = 200 (006 / 009 helpers)"""
    out = []
    for t in TARGETS:
        os.environ["DETECT_TARGET"] = t
        for m in ("_areas", "002_detection_summary", "004_main_figures", "006_active_vs_passive", "009_area_ranking"):
            sys.modules.pop(m, None)
        m9 = importlib.import_module("009_area_ranking")
        for e in EPOCHS:
            for level in AR.LEVELS:
                try:
                    q = m9.metrics(e, level)
                    folder, _ = m9.m6.source(e, level)
                    w = pd.read_csv(folder / "window_accuracy.csv")
                except FileNotFoundError:
                    continue
                w = w[(w.level == level) & (w.N == N_MAIN)].set_index("area")["mean"]
                out.append(q.assign(target=t, early=q.area.map(w)))
    D = pd.concat(out, ignore_index=True)
    return D.rename(columns={"onset_ms": "onset"})


def tests(A, D):
    rows = []
    M = D.merge(A, on=["level", "area"], how="inner")
    for (t, e, level), g in M.groupby(["target", "epoch", "level"]):
        g = g[~g.area.isin(SOURCE_AREAS["whisker"] | SOURCE_AREAS["auditory"])]
        for src in ("whisker", "auditory"):
            for mname in MEASURES:
                x, y = g[f"dens_{src}"].to_numpy(float), g[mname].to_numpy(float)
                if mname == "onset" and "onset_flag" in g:
                    keep = ~g.onset_flag.fillna(True).to_numpy(bool)
                    x, y = x[keep], y[keep]
                k = np.isfinite(x) & np.isfinite(y) & (x > 0)
                if k.sum() < 5:
                    continue
                rs = stats.spearmanr(x[k], y[k]); rp = stats.pearsonr(np.log10(x[k]), y[k])
                rows.append(dict(target=t, epoch=e, level=level, source=src, matching=src == t, measure=mname, n_areas=int(k.sum()),
                                 spearman_rho=rs.statistic, spearman_p=rs.pvalue, pearson_r_logdens=rp.statistic, pearson_p=rp.pvalue))
    return pd.DataFrame(rows), M


def figure(plt, M, T, level, col):
    fig, axs = plt.subplots(len(TARGETS) * len(EPOCHS), len(MEASURES), figsize=(S.W_IN, 7.0))
    for i, (t, e) in enumerate([(t, e) for t in TARGETS for e in EPOCHS]):
        g = M[(M.target == t) & (M.epoch == e) & (M.level == level)]
        g = g[~g.area.isin(SOURCE_AREAS["whisker"] | SOURCE_AREAS["auditory"])]
        for j, mname in enumerate(MEASURES):
            ax = axs[i, j]
            x, y = g[f"dens_{t}"], g[mname]
            ax.scatter(x, y, s=10, c=[col.get(a, "0.5") for a in g.area], lw=0)
            for a, xx, yy in zip(g.area, x, y):
                if np.isfinite(xx) and np.isfinite(yy) and level == "area_group":
                    ax.text(xx, yy, " " + S.SHORT.get(a, a), fontsize=3.6, va="center")
            ax.set_xscale("log")
            r = T[(T.target == t) & (T.epoch == e) & (T.level == level) & (T.source == t) & (T.measure == mname)]
            if len(r):
                ax.set_title(f"rho = {r.spearman_rho.iloc[0]:+.2f}, p = {r.spearman_p.iloc[0]:.3f} (n = {r.n_areas.iloc[0]})", fontsize=4.8)
            if j == 0:
                ax.set_ylabel(f"{t} vs no stim., {e}\n{MEASURES[mname]}", fontsize=5)
            else:
                ax.set_ylabel(MEASURES[mname], fontsize=5)
            ax.set_xlabel(f"{t} cortex projection density\nat recorded units (Allen)", fontsize=5)
    fig.suptitle(f"Stimulus-presence decoding vs Allen projection density from the stimulus' cortex, {level} (N = {N_MAIN}; source areas excluded)",
                 fontsize=6.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    S.save(fig, OUTA / "figures", f"anatomy_vs_decoding_{level}")
    plt.close(fig)


def main():
    plt = S.setup()
    OUTA.mkdir(parents=True, exist_ok=True)
    A = anatomy(); A.to_csv(OUTA / "anatomy_by_area.csv", index=False)
    D = decoding()
    T, M = tests(A, D)
    T.to_csv(OUTA / "anatomy_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[T.matching].round(3).to_string(index=False))
    col = AR.colors()
    for level in AR.LEVELS:
        figure(plt, M, T, level, col)
    (OUTA / "provenance_010.json").write_text(json.dumps(dict(script="010_anatomy_link.py", zones=str(ZONES), N=N_MAIN,
                                                              date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print("ALL DONE", OUTA)


if __name__ == "__main__":
    main()
