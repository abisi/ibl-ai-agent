"""010 -- Sanity check (user 2026-10-16): does stimulus-presence decoding per area relate to the anatomical input from the
whisker and auditory cortices? Two anatomical sources:
  Liu et al. 2024 (eLife; ephys_utilities allen_utils/data/Liu_et_al_Group_averages_ranked.xlsx): axonal projection strength
  from wS1 and wS2 per target structure, 6 Cre lines; used (user 2026-10-16): ipsilateral, both injection sites combined =
  column avg_ipsi (mean of the 12 ipsi values, 2 sites x 6 lines). Each unit gets the value of the deepest Liu target
  structure containing it (CCF annotation, structure tree; Liu mixes levels, e.g. TH and VP); per area: mean over its units.
  Allen Mouse Connectivity:
Anatomy per area: Allen projection density (ssl-sensory-spatial-maps 002: line-balanced, normalised per experiment, smoothed
50 um, right hemisphere, source area itself excluded) of SSp-bfd, SSs, AUDp, AUD-sec and the merged whisker (SSp-bfd + SSs) and
auditory (AUDp + AUD-sec) sources, sampled at the CCF position of every good + mua unit of the area (ML folded onto the right
hemisphere, 50-um voxels) and averaged over the area's units ("input at the recorded sites"); also the fraction of the
area's units inside the source's 70 % projection zone (zone70, "locality").
Decoding per area (N = 200; final run if complete, else the N-sweep rows; 006 source()): onset (zoom, 004 rule; unreliable
onsets excluded from onset tests), early corrected accuracy 5-50 ms, peak corrected accuracy (009).
Tested Allen measure per modality (dens_<modality>_incl): per unit, the mean of the modality's source densities leaving out
the source the unit sits in (SSp-bfd units -> SSs projection, SSs units -> SSp-bfd; AUDp <-> AUD-sec), so the source areas
themselves are included (user 2026-10-16: include Ss-whisker).
Tests across areas, per level (area groups / fine areas), all areas with a value: Spearman rho (and Pearson r on log density) of each decoding measure with the matching source (whisker target ~ whisker
density, auditory target ~ auditory density) and with the other source as a specificity control; per epoch.
Output: <HOME>/anatomy/anatomy_by_area.csv, anatomy_tests.csv, figures/anatomy_{allen,liu}_vs_decoding_<level>.{png,pdf,svg}
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
LIU = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities/ephys_utilities/allen_utils/data/Liu_et_al_Group_averages_ranked.xlsx")
ATLAS = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0")
TESTED = {"whisker": "dens_whisker_incl", "auditory": "dens_auditory_incl", "liu": "liu_ipsi"}      # anatomical measure per source
MIN_MAPPED = 0.5                       # Liu value used only if >= 50 % of the area's units fall in a Liu target structure
MATCH = {"whisker": "whisker", "auditory": "auditory", "liu": "whisker"}                  # target each source should predict
MODALITY_SOURCES = {"whisker": ["SSp-bfd", "SSs"], "auditory": ["AUDp", "AUD-sec"]}
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
    # merged whisker / auditory input including the source areas themselves (user 2026-10-16: include Ss-whisker): per unit,
    # mean of the normalised densities of the modality's sources, leaving out the source the unit sits in (its own injection
    # area, excluded from that source's density by construction): SSp-bfd units -> SSs projection, SSs units -> SSp-bfd, etc.
    for mod, srcs in MODALITY_SOURCES.items():
        D = np.stack([U[f"dens_{s}"].to_numpy(float) for s in srcs], 1)
        own = np.stack([z[f"source_{s}"][ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(bool) for s in srcs], 1)
        D[own] = np.nan
        U[f"dens_{mod}_incl"] = np.nanmean(D, 1)
    for k, v in liu_units(U, ml).items():
        U[k] = v
    rows = []
    for level, col in (("area_group", "area_group"), ("area_acronym_custom", "area_acronym_custom")):
        for a, g in U[U[col].isin(AR.LEVELS[level])].groupby(col):
            rows.append(dict(level=level, area=a, n_units=len(g), **{f"dens_{s}": g[f"dens_{s}"].mean() for s in SOURCES},
                             **{f"dens_{m}_incl": g[f"dens_{m}_incl"].mean() for m in MODALITY_SOURCES},
                             **{f"zone_{s}": g[f"zone_{s}"].mean() for s in SOURCES},
                             liu_ipsi=g.liu_ipsi.mean(), liu_s1_ipsi=g.liu_s1_ipsi.mean(), liu_s2_ipsi=g.liu_s2_ipsi.mean(),
                             liu_mapped=g.liu_ipsi.notna().mean()))
    A = pd.DataFrame(rows)
    A.loc[A.liu_mapped < MIN_MAPPED, ["liu_ipsi", "liu_s1_ipsi", "liu_s2_ipsi"]] = np.nan     # area mostly outside Liu's targets
    return A


def liu_units(U, ml):
    """per unit: Liu et al. ipsilateral projection strength of the deepest Liu target structure containing the unit
    (avg_ipsi = both sites; also wS1-only and wS2-only means over the 6 lines)"""
    import tifffile
    X = pd.read_excel(LIU, header=None)
    site, line = X.iloc[0].ffill(), X.iloc[1]
    V = X.iloc[3:].set_index(0)
    s1 = [c for c in X.columns if site[c] == "s1_ispi" and isinstance(line[c], str)]
    s2 = [c for c in X.columns if site[c] == "s2_ipsi" and isinstance(line[c], str)]
    avg = [c for c in X.columns if X.iloc[0, c] == "avg_ipsi"][0]
    assert len(s1) == len(s2) == 6, (s1, s2)
    tab = pd.DataFrame(dict(ipsi=V[avg].astype(float), s1=V[s1].astype(float).mean(1), s2=V[s2].astype(float).mean(1)))
    assert np.allclose(tab.ipsi, (tab.s1 + tab.s2) / 2, rtol=1e-6), "avg_ipsi is not the mean of the 12 ipsi values"
    St = pd.read_csv(ATLAS / "structures.csv")
    acr = dict(zip(St.id, St.acronym))
    path = {i: [int(x) for x in p.strip("/").split("/")] for i, p in zip(St.id, St.structure_id_path)}
    def deepest(i):
        for a in reversed(path.get(i, [])):
            if acr.get(a) in tab.index:
                return acr[a]
        return None
    ann = tifffile.imread(ATLAS / "annotation.tiff")
    ijk = np.round(np.c_[U.ccf_atlas_ap.to_numpy(float), U.ccf_atlas_dv.to_numpy(float), ml] / 10).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(ann.shape)), axis=1)
    ids = np.zeros(len(U), int); ids[ok] = ann[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    m = {i: deepest(int(i)) for i in np.unique(ids)}
    tgt = np.array([m[i] for i in ids], dtype=object)
    get = lambda c: np.array([tab[c].get(t, np.nan) if t else np.nan for t in tgt], float)
    return dict(liu_target=tgt, liu_ipsi=get("ipsi"), liu_s1_ipsi=get("s1"), liu_s2_ipsi=get("s2"))


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
                if q.empty:
                    continue
                w = w[(w.level == level) & (w.N == N_MAIN)].set_index("area")["mean"]
                out.append(q.assign(target=t, early=q.area.map(w)))
    D = pd.concat(out, ignore_index=True)
    return D.rename(columns={"onset_ms": "onset"})


def tests(A, D):
    rows = []
    M = D.merge(A, on=["level", "area"], how="inner")
    for (t, e, level), g in M.groupby(["target", "epoch", "level"]):
        for src, colname in TESTED.items():
            for mname in MEASURES:
                x, y = g[colname].to_numpy(float), g[mname].to_numpy(float)
                if mname == "onset" and "onset_flag" in g:
                    keep = ~g.onset_flag.fillna(True).to_numpy(bool)
                    x, y = x[keep], y[keep]
                k = np.isfinite(x) & np.isfinite(y) & (x > 0)
                if k.sum() < 5:
                    continue
                rs = stats.spearmanr(x[k], y[k]); rp = stats.pearsonr(np.log10(x[k]), y[k])
                rows.append(dict(target=t, epoch=e, level=level, source=src, matching=MATCH[src] == t, measure=mname, n_areas=int(k.sum()),
                                 spearman_rho=rs.statistic, spearman_p=rs.pvalue, pearson_r_logdens=rp.statistic, pearson_p=rp.pvalue))
    return pd.DataFrame(rows), M


XNAME = {"allen": "projection density from the stimulus' cortex\nat recorded units (Allen)",
         "liu": "wS1 + wS2 ipsi projection\n(Liu et al. 2024)"}


def figure(plt, M, T, level, col, src):
    """rows: target x epoch; columns: measures; x: Allen density of the target's own cortex (src "allen") or Liu wS1 + wS2
    ipsi projection for both targets (src "liu"; auditory rows = specificity control)"""
    fig, axs = plt.subplots(len(TARGETS) * len(EPOCHS), len(MEASURES), figsize=(S.W_IN, 7.0), squeeze=False)
    for i, (t, e) in enumerate([(t, e) for t in TARGETS for e in EPOCHS]):
        g = M[(M.target == t) & (M.epoch == e) & (M.level == level)]
        key = t if src == "allen" else "liu"
        for j, mname in enumerate(MEASURES):
            ax = axs[i, j]
            x, y = g[TESTED[key]], g[mname]
            ax.scatter(x, y, s=10, c=[col.get(a, "0.5") for a in g.area], lw=0)
            for a, xx, yy in zip(g.area, x, y):
                if np.isfinite(xx) and np.isfinite(yy) and xx > 0 and level == "area_group":
                    ax.text(xx, yy, " " + S.SHORT.get(a, a), fontsize=3.6, va="center")
            ax.set_xscale("log")
            if mname == "onset":
                ax.invert_yaxis()
            r = T[(T.target == t) & (T.epoch == e) & (T.level == level) & (T.source == key) & (T.measure == mname)]
            if len(r):
                ax.set_title(f"rho = {r.spearman_rho.iloc[0]:+.2f}, p = {r.spearman_p.iloc[0]:.3f} (n = {r.n_areas.iloc[0]})", fontsize=4.8)
            ax.set_ylabel((f"{t} vs no stim., {e}\n" if j == 0 else "") + MEASURES[mname], fontsize=5)
            ax.set_xlabel(XNAME[src], fontsize=4.6)
    what = "Allen projection density from the stimulus' own cortex" if src == "allen" else "wS1 + wS2 ipsilateral projections (Liu et al. 2024)"
    fig.suptitle(f"Stimulus-presence decoding vs {what}, {level} (N = {N_MAIN}; source areas included; onset axis inverted)",
                 fontsize=6.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    S.save(fig, OUTA / "figures", f"anatomy_{src}_vs_decoding_{level}")
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
        for src in ("allen", "liu"):
            if (M.level == level).any():
                figure(plt, M, T, level, col, src)
    (OUTA / "provenance_010.json").write_text(json.dumps(dict(script="010_anatomy_link.py", zones=str(ZONES), N=N_MAIN,
                                                              date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print("ALL DONE", OUTA)


if __name__ == "__main__":
    main()
