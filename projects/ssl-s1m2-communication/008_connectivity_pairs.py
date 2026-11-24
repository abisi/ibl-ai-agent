"""Step 1 of the connected-pairs extension: projection strength of each area pair from already-extracted anatomy, and
SSL session coverage per pair.

Anatomy (sources wS1 = SSp-bfd, wS2 = SSs, A1 = AUDp only; no other source has extracted data):
- Allen Mouse Connectivity (ssl-sensory-spatial-maps 002, projection_zones.npz): line-balanced projection density per
  source (each experiment normalised to its total over candidate voxels, line means averaged, smoothed 50 um, right
  hemisphere, source area and fibre tracts excluded). Target voxels = the unique 50-um CCF voxels holding >= 1 stable
  good + mua unit of the target area (all sessions, ML folded to the right), minus voxels inside the source structure.
  allen_enrichment = mean density over the target voxels / mean density over all candidate voxels (1 = brain average).
  Thresholds (mass contours, as the 002 projection zones): thr70 / thr90 = density at which the densest candidate
  voxels hold 70 % / 90 % of the projection. allen_class: strong if mean target density >= thr70 (the average
  recorded site is as dense as the core-zone boundary), weak if >= thr90, none otherwise. Also the fraction of target
  voxels inside the 70 % zone.
- Liu et al. 2024 (wS1 / wS2 injections, 6 Cre lines, ipsilateral, mean of lines): each target unit takes the value of
  the deepest Liu structure containing it (as ssl-stimulus-detection-decoding 010); per area mean over units. Liu values
  are per-structure totals (size-biased, coarse: DLS / DMS / TS all take the CP value; injection site not masked).
  liu_class from the value's percentile among Liu's structures for that source: strong >= 90th, weak >= 50th, none below.
- class = allen_class (primary; Liu disagreement flagged). Pairs of two sources: both directions, class = stronger.
  Pairs with no source end: "connectivity not extracted".
Coverage: sessions with >= PAIR_CONFIG min_units (5) stable good + mua units in both areas; cohort per mouse from the reference sheet (exclude /
exclude_ephys dropped); stage learning = day 0, expert = day > 0. Pair passes for a cohort if >= 3 sessions at both
stages (ssl_area_inclusion_within_cohort).

Outputs (OUT/pairs): tables/connectivity_pairs.csv, tables/pair_sessions.csv, tables/area_anatomy.csv,
figures/connectivity_pairs.{png,pdf,svg}.
Run (haas): ../../.venv/bin/python 008_connectivity_pairs.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pairs_lib as P

SSM = P.L.ROOT / "combined_results_ks4" / "ssl-sensory-spatial-maps"
ATLAS = P.L.ROOT / "Anatomy" / "allen_mouse_bluebrain_barrels_10um_v1.0"
LIU = P.L.ROOT / "Github" / "ephys_utilities" / "ephys_utilities" / "allen_utils" / "data" / "Liu_et_al_Group_averages_ranked.xlsx"
RES, MID = 50, 5700
SRC_IDS = {"SSp-bfd": [329], "SSs": [378], "AUDp": [1002]}
LIU_PCT = (0.90, 0.50)
CLASS_ORDER = {"strong": 2, "weak": 1, "none": 0}


def descendants(structs, ids):
    ids = set(ids)
    return {s["id"] for s in structs if ids & set(s["structure_id_path"])}


def allen(units):
    import tifffile
    z = np.load(SSM / "projection_zones.npz")
    structs = json.load(open(ATLAS / "structures.json"))
    ann = tifffile.imread(ATLAS / "annotation.tiff")
    s = RES // 10
    ann = ann[s // 2::s, s // 2::s, s // 2::s]
    right = np.zeros(ann.shape, bool)
    right[:, :, int(MID / RES):] = True
    brain = (ann != 0) & ~np.isin(ann, list(descendants(structs, [1009, 73]))) & right
    u = units.dropna(subset=["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"])
    ml = MID + np.abs(u.ccf_atlas_ml.to_numpy(float) - MID)
    ijk = np.round(np.c_[u.ccf_atlas_ap.to_numpy(float), u.ccf_atlas_dv.to_numpy(float), ml] / RES).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(ann.shape)), axis=1)
    u, ijk = u[ok], ijk[ok]
    rows = []
    for src, key in P.ALLEN_SRC.items():
        dens = z[f"density_{key}"].astype(float)
        srcmask = np.isin(ann, list(descendants(structs, SRC_IDS[key])))
        cand = brain & ~srcmask
        v = np.sort(dens[cand])[::-1]
        cm = np.cumsum(v) / v.sum()
        thr70, thr90 = v[np.searchsorted(cm, 0.70)], v[np.searchsorted(cm, 0.90)]
        mean_c = dens[cand].mean()
        zone70 = cand & (dens >= thr70)
        for tgt in P.AREAS:
            if tgt == src:
                continue
            g = ijk[(u.area == tgt).to_numpy()]
            vox = np.unique(g, axis=0) if len(g) else np.zeros((0, 3), int)
            keep = cand[vox[:, 0], vox[:, 1], vox[:, 2]] if len(vox) else np.zeros(0, bool)
            vox = vox[keep]
            d = dens[vox[:, 0], vox[:, 1], vox[:, 2]] if len(vox) else np.array([np.nan])
            m = float(np.nanmean(d))
            rows.append(dict(source=src, target=tgt, allen_source=key, n_target_voxels=len(vox),
                             n_target_voxels_in_source=int((~keep).sum()) if len(keep) else 0,
                             allen_density=m, allen_enrichment=m / mean_c, allen_thr70_enrichment=thr70 / mean_c,
                             allen_thr90_enrichment=thr90 / mean_c,
                             allen_frac_zone70=float(zone70[vox[:, 0], vox[:, 1], vox[:, 2]].mean()) if len(vox) else np.nan,
                             allen_class=("strong" if m >= thr70 else "weak" if m >= thr90 else "none") if len(vox) else "no sites"))
    return pd.DataFrame(rows)


def liu(units):
    import tifffile
    X = pd.read_excel(LIU, header=None)
    site, line = X.iloc[0].ffill(), X.iloc[1]
    V = X.iloc[3:].set_index(0)
    cols = {"s1": [c for c in X.columns if site[c] == "s1_ispi" and isinstance(line[c], str)],
            "s2": [c for c in X.columns if site[c] == "s2_ipsi" and isinstance(line[c], str)]}
    assert len(cols["s1"]) == len(cols["s2"]) == 6
    tab = pd.DataFrame({k: V[c].astype(float).mean(1) for k, c in cols.items()})
    St = pd.read_csv(ATLAS / "structures.csv")
    acr = dict(zip(St.id, St.acronym))
    path = {i: [int(x) for x in p.strip("/").split("/")] for i, p in zip(St.id, St.structure_id_path)}

    def deepest(i):
        for a in reversed(path.get(i, [])):
            if acr.get(a) in tab.index:
                return acr[a]
        return None

    u = units.dropna(subset=["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"])
    ml = MID + np.abs(u.ccf_atlas_ml.to_numpy(float) - MID)
    ann = tifffile.imread(ATLAS / "annotation.tiff")
    ijk = np.round(np.c_[u.ccf_atlas_ap.to_numpy(float), u.ccf_atlas_dv.to_numpy(float), ml] / 10).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(ann.shape)), axis=1)
    ids = np.zeros(len(u), int)
    ids[ok] = ann[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    m = {i: deepest(int(i)) for i in np.unique(ids)}
    tgt_struct = np.array([m[i] for i in ids], dtype=object)
    rows = []
    for src, site_ in P.LIU_SRC.items():
        q_hi, q_lo = tab[site_].quantile(LIU_PCT[0]), tab[site_].quantile(LIU_PCT[1])
        for tgt in P.AREAS:
            if tgt == src:
                continue
            sel = (u.area == tgt).to_numpy()
            ts = tgt_struct[sel]
            vals = np.array([tab[site_].get(t, np.nan) if t else np.nan for t in ts], float)
            mapped = np.isfinite(vals).mean() if len(vals) else 0
            val = np.nanmean(vals) if mapped >= 0.5 else np.nan
            top = pd.Series(ts).value_counts().head(3)
            rows.append(dict(source=src, target=tgt, liu=val, liu_mapped=mapped,
                             liu_pct=float((tab[site_] < val).mean()) if np.isfinite(val) else np.nan,
                             liu_structures="; ".join(f"{k} {v}" for k, v in top.items()),
                             liu_class=("strong" if val >= q_hi else "weak" if val >= q_lo else "none")
                             if np.isfinite(val) else "n/a"))
    return pd.DataFrame(rows)


def coverage(units):
    ref = P.reference()
    rows = []
    for (sid, mouse, day), g in units[units.area.notna()].groupby(["session_id", "mouse_id", "day"]):
        if mouse not in ref.index:
            continue
        n = g.area.value_counts()
        for a, b, _ in P.all_pairs():
            if n.get(a, 0) >= P.PAIR_CONFIG["min_units"] and n.get(b, 0) >= P.PAIR_CONFIG["min_units"]:
                rows.append(dict(session_id=sid, mouse_id=mouse, day=int(day), stage="learning" if day == 0 else "expert",
                                 cohort=ref.loc[mouse, "reward_group"], learning_category=ref.loc[mouse, "learning_category"],
                                 pair=P.pair_name(a, b), source=a, target=b, n_source=int(n[a]), n_target=int(n[b]),
                                 same_probe_frac=P.same_probe_frac(g, a, b)))
    df = pd.DataFrame(rows)
    df["learner"] = df.learning_category.isin(["good", "moderate"])
    return df


def classify(A, Lq):
    """pair-level class: Allen primary; source-source pairs take the stronger direction"""
    rows = []
    for a, b, ext in P.all_pairs():
        r = dict(source=a, target=b, pair=P.pair_name(a, b), extracted=ext)
        if not ext:
            r["class"] = "connectivity not extracted"
            rows.append(r)
            continue
        dirs = [(a, b)] + ([(b, a)] if b in P.SOURCES else [])
        best = None
        for s, t in dirs:
            al = A[(A.source == s) & (A.target == t)].iloc[0]
            if best is None or CLASS_ORDER.get(al.allen_class, -1) > CLASS_ORDER.get(best[1].allen_class, -1):
                best = ((s, t), al)
        (s, t), al = best
        lq = Lq[(Lq.source == s) & (Lq.target == t)]
        lq = lq.iloc[0] if len(lq) else None
        r.update(anat_direction=f"{s}->{t}", allen_enrichment=al.allen_enrichment, allen_frac_zone70=al.allen_frac_zone70,
                 allen_thr70_enrichment=al.allen_thr70_enrichment, allen_thr90_enrichment=al.allen_thr90_enrichment,
                 allen_class=al.allen_class, n_target_voxels=al.n_target_voxels,
                 liu=lq.liu if lq is not None else np.nan, liu_pct=lq.liu_pct if lq is not None else np.nan,
                 liu_class=lq.liu_class if lq is not None else "n/a",
                 liu_structures=lq.liu_structures if lq is not None else "")
        if len(dirs) == 2:
            o = A[(A.source == b) & (A.target == a)].iloc[0]
            r.update(allen_enrichment_reverse=o.allen_enrichment, allen_class_reverse=o.allen_class)
        r["class"] = al.allen_class
        r["liu_disagrees"] = r["liu_class"] not in ("n/a", al.allen_class)
        rows.append(r)
    return pd.DataFrame(rows)


def add_n(C, S):
    rule = P.PAIR_CONFIG["min_sessions"]
    for coh in ("R+", "R-"):
        for st in ("learning", "expert"):
            n = S[(S.cohort == coh) & (S.stage == st)].groupby("pair").session_id.nunique()
            C[f"n_{coh}_{st}"] = C.pair.map(n).fillna(0).astype(int)
            m = S[(S.cohort == coh) & (S.stage == st)].groupby("pair").mouse_id.nunique()
            C[f"mice_{coh}_{st}"] = C.pair.map(m).fillna(0).astype(int)
        C[f"pass_{coh}"] = (C[f"n_{coh}_learning"] >= rule) & (C[f"n_{coh}_expert"] >= rule)
    C["pass_any"] = C["pass_R+"] | C["pass_R-"]
    sp = S.groupby("pair").same_probe_frac.mean()
    C["same_probe_frac_mean"] = C.pair.map(sp)
    return C


def figure(C, A):
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.linewidth": 0.6})
    areas = list(P.AREAS)
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 6.4), gridspec_kw=dict(height_ratios=[1, 3.2], hspace=0.45))
    # A: Allen enrichment source x target
    M = np.full((len(P.SOURCES), len(areas)), np.nan)
    for i, s in enumerate(P.SOURCES):
        for j, t in enumerate(areas):
            r = A[(A.source == s) & (A.target == t)]
            if len(r):
                M[i, j] = r.allen_enrichment.iloc[0]
    ax = axs[0]
    im = ax.imshow(np.log10(np.clip(M, 1e-2, None)), cmap="Greys", vmin=-1, vmax=1.5, aspect="auto")
    for i, s in enumerate(P.SOURCES):
        for j, t in enumerate(areas):
            r = A[(A.source == s) & (A.target == t)]
            if len(r):
                c = r.allen_class.iloc[0]
                ax.text(j, i, f"{M[i, j]:.1f}\n{c}", ha="center", va="center", fontsize=5,
                        color="w" if np.log10(max(M[i, j], 1e-2)) > 0.6 else "k")
    ax.set_xticks(range(len(areas)), areas, rotation=45, ha="right")
    ax.set_yticks(range(len(P.SOURCES)), [f"{s} ({P.ALLEN_SRC[s]})" for s in P.SOURCES])
    ax.set_title("A  Allen projection density at the recorded sites of the target (x brain-average; class by 70 % / "
                 "90 % mass contours)", loc="left", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.set_label("log10 enrichment", fontsize=6)
    # B: pair matrix: class colour + learning sessions R+/R-
    col = {"strong": "#7fb2e5", "weak": "#cfe6fb", "none": "#eeeeee", "connectivity not extracted": "white"}
    ax = axs[1]
    ax.set_xlim(-0.5, len(areas) - 0.5)
    ax.set_ylim(len(areas) - 0.5, -0.5)
    for _, r in C.iterrows():
        i, j = areas.index(r.source), areas.index(r.target)
        for (y, x) in ((i, j), (j, i)):
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=col[r["class"]], edgecolor="0.75", lw=0.3))
            ax.text(x, y - 0.17, f"{r['n_R+_learning']}/{r['n_R+_expert']}", ha="center", va="center", fontsize=4.6,
                    color="#00B400")
            ax.text(x, y + 0.2, f"{r['n_R-_learning']}/{r['n_R-_expert']}", ha="center", va="center", fontsize=4.6,
                    color="#C800C8")
            if r.pass_any:
                ax.add_patch(plt.Rectangle((x - 0.47, y - 0.47), 0.94, 0.94, fill=False, edgecolor="k", lw=0.8))
    ax.set_xticks(range(len(areas)), areas, rotation=45, ha="right")
    ax.set_yticks(range(len(areas)), areas)
    ax.set_aspect("equal")
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=v, edgecolor="0.6", lw=0.3) for v in col.values()]
    hs.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="k", lw=0.8))
    ax.legend(hs, [*col.keys(), ">= 3 sessions at both stages\nin >= 1 cohort"], fontsize=5.5, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"B  Connectivity class (Allen; sources wS1 / wS2 / A1 only) and sessions with >= {P.PAIR_CONFIG['min_units']} stable units in both "
                 "areas:\n    R+ (green) and R- (magenta), learning / expert", loc="left", fontsize=7)
    P.FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(P.FIGS / f"connectivity_pairs.{ext}", dpi=300, bbox_inches="tight")


def main():
    P.TABLES.mkdir(parents=True, exist_ok=True)
    u = P.stable_units(with_coords=True)
    ref = P.reference()
    u = u[u.mouse_id.isin(ref.index)]
    print("units per area:\n", u.area.value_counts().to_string(), "\nmissing coords:", u.ccf_atlas_ap.isna().mean(), flush=True)
    A = allen(u[u.area.notna()])
    Lq = liu(u[u.area.notna()])
    A.merge(Lq, on=["source", "target"], how="left").to_csv(P.TABLES / "area_anatomy.csv", index=False)
    S = coverage(u)
    S.to_csv(P.TABLES / "pair_sessions.csv", index=False)
    C = add_n(classify(A, Lq), S)
    C["config"] = json.dumps(dict(P.PAIR_CONFIG, liu_pct=LIU_PCT, allen="projection_zones.npz density, thr70 / thr90",
                                  unit_table=str(P.L.UNIT_TABLE), stable_table=str(P.L.STABLE_TABLE)))
    C.to_csv(P.TABLES / "connectivity_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(C[C.extracted][["pair", "anat_direction", "allen_enrichment", "allen_frac_zone70", "allen_class", "liu_pct",
                          "liu_class", "n_R+_learning", "n_R+_expert", "n_R-_learning", "n_R-_expert", "pass_any",
                          "same_probe_frac_mean"]].round(2).to_string())
    print(C[~C.extracted & C.pass_any][["pair", "n_R+_learning", "n_R+_expert", "n_R-_learning", "n_R-_expert"]].to_string())
    print("thresholds:", A.groupby("source")[["allen_thr70_enrichment", "allen_thr90_enrichment"]].first().round(2).to_string())
    figure(C, A)


if __name__ == "__main__":
    main()
