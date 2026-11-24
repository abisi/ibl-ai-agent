"""Review: is SSp-un (and SSp-tr) functionally closer to Ss-body, Ss-orofacial or Ss-whisker? (user request 2026-11-18)

Current pipeline labels (ephys_utilities allen_utils): SSp-un -> SSp-oro (Somatosensory-orofacial), SSp-tr -> SSp-body
(Somatosensory-body); handle_ssp_oro / handle_ssp_body run before handle_ppc, so neither becomes PPC on PPC probes.

Anatomical classes from the unit's CCF atlas acronym (layer stripped): SSp-un, SSp-tr, body (SSp-ul, SSp-ll),
oro (SSp-m, SSp-n), bfd (SSp-bfd), SSs, PPC (VISa, VISrl).
1. Inventory: insertions (session x probe) with SSp-un / SSp-tr units, target, day, cohort, unit counts, areas on the
   same track.
2. Anatomy: distance (um) from each SSp-un / SSp-tr unit to the nearest voxel of each class (Allen 25 um).
3. Function (all days, cohorts pooled; ROC units = quality good + mua, PSTH = good units only):
   % significant for whisker / auditory passive (pre), whisker / auditory active, spontaneous licks, choice;
   passive whisker PSTH (z vs pre-stim baseline): early response 5-30 ms, peak latency of responsive units.
   Insertion-level values (pooled units within insertion x class) -> class mean and bootstrap CI over insertions.
   Within-probe comparison on the SSp-un / SSp-tr insertions themselves.
   Profile distance: standardised profile of SSp-un vs each class, bootstrap over insertions -> how often each class is
   nearest.
Outputs: combined_results_ks4/ssl-ssp-un-review/{inventory.csv, anatomy_distance.csv, insertion_profiles.csv,
         class_summary.csv, within_probe.csv, nearest_class_bootstrap.csv, review.png/pdf}
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 001_ssp_un_review.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = R / "ssl-ssp-un-review"
ROC = R / "_roc_stage_analysis"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
CLASS = {"SSp-un": "SSp-un", "SSp-tr": "SSp-tr", "SSp-ul": "body", "SSp-ll": "body", "SSp-m": "oro", "SSp-n": "oro",
         "SSp-bfd": "bfd", "SSs": "SSs", "VISa": "PPC", "VISrl": "PPC"}
ORDER = ["SSp-un", "SSp-tr", "body", "oro", "bfd", "SSs", "PPC"]
COLC = {"SSp-un": "k", "SSp-tr": "0.45", "body": "#e0c92f", "oro": "#a66829", "bfd": "#f7b519", "SSs": "#d99a00",
        "PPC": "#009fac"}
TYPES = ["whisker_passive_pre", "auditory_passive_pre", "whisker_active", "auditory_active", "spontaneous_licks",
         "choice"]
RNG = np.random.default_rng(0)
NBOOT = 2000


def base_acr(a):
    a = str(a)
    m = re.match(r"^(SSp-[a-z]+|SSs|VISa|VISrl)", a)
    return m.group(1) if m else None


def psth_metrics():
    """Per good unit: early passive whisker response (z, 5-30 ms) and peak latency (ms, 0-100 ms)."""
    rows = []
    for f in sorted((ROC / "psth").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "W_pre" not in z.files or int(z["n_W_pre"]) < 10:
            continue
        b = z["stim_bins"]
        c = (b[:-1] + b[1:]) / 2
        sd = np.maximum(z["base_sd"], 0.5)
        zz = (z["W_pre"] - z["base_mu"][:, None]) / sd[:, None]
        early = zz[:, (c > 0.005) & (c < 0.03)].mean(1)
        w = (c > 0) & (c < 0.1)
        lat = c[w][np.argmax(zz[:, w], 1)] * 1000
        rows.append(pd.DataFrame(dict(session_id=f.stem, electrode_group=z["electrode_group"],
                                      cluster_id=z["cluster_id"].astype(str), early_z=early, peak_lat_ms=lat)))
    return pd.concat(rows, ignore_index=True)


def boot_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    bs = RNG.choice(v, (NBOOT, len(v))).mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    lab = pd.read_parquet(OUT / "unit_labels_all_days_v2.parquet")
    lab = lab.drop_duplicates(KEYS)
    lab["cls"] = lab.ccf_atlas_acronym.map(base_acr).map(CLASS)
    lab["ins"] = lab.session_id + "|" + lab.electrode_group

    # ---------------- 1. inventory
    q = lab[lab.quality_label.isin(["good", "mua"])]
    inv = []
    for c in ("SSp-un", "SSp-tr"):
        s = q[q.cls == c]
        for ins, g in s.groupby("ins"):
            p = q[q.ins == ins]
            near = (p[p.cls.notna() & (p.cls != c)].cls.value_counts().to_dict())
            inv.append(dict(cls=c, mouse_id=g.mouse_id.iloc[0], session_id=g.session_id.iloc[0],
                            electrode_group=g.electrode_group.iloc[0], target_region=g.target_region.iloc[0],
                            day=g.day.iloc[0], reward_group=g.reward_group.iloc[0], n_units=len(g),
                            n_good=int((g.quality_label == "good").sum()), n_probe=len(p),
                            labels_now=";".join(sorted(g.area_acronym_custom.astype(str).unique())),
                            groups_now=";".join(sorted(g.area_group.astype(str).unique())),
                            other_ss_on_probe=str(near), dv_range_um=f"{g.dv.min():.0f}-{g.dv.max():.0f}"))
    inv = pd.DataFrame(inv)
    inv.to_csv(OUT / "inventory.csv", index=False)
    print(inv.to_string())

    # ---------------- 2. anatomy: distance to each class (Allen 25 um)
    from brainglobe_atlasapi import BrainGlobeAtlas
    from scipy import ndimage
    atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    ann = np.asarray(atlas.annotation)
    ids = {c: [] for c in ORDER}
    for sid, s in atlas.structures.items():
        c = CLASS.get(base_acr(s["acronym"]) or "")
        if c:
            ids[c].append(sid)
    pts = q[q.cls.isin(["SSp-un", "SSp-tr"])].copy()
    vox = np.round(pts[["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"]].to_numpy() / 25).astype(int)
    vox = np.clip(vox, 0, np.array(ann.shape) - 1)
    for c in ORDER[2:] + ["SSp-un", "SSp-tr"]:
        dist = ndimage.distance_transform_edt(~np.isin(ann, ids[c])) * 25
        pts[f"dist_{c}"] = dist[vox[:, 0], vox[:, 1], vox[:, 2]]
        del dist
    dcols = [f"dist_{c}" for c in ORDER[2:]]
    an = pts.groupby("cls")[dcols].median().round(0)
    an["nearest_neighbour_class_frac"] = pts.groupby("cls")[dcols].apply(
        lambda x: x.idxmin(axis=1).str.replace("dist_", "").value_counts(normalize=True).round(2).to_dict())
    an.to_csv(OUT / "anatomy_distance.csv")
    print(an.to_string())

    # ---------------- 3. function
    roc = pd.read_parquet(ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sig", "pos"])
    roc = roc[roc.analysis_type.isin(TYPES)]
    roc["cluster_id"] = roc.cluster_id.astype(str)
    wide = roc.pivot_table(index=KEYS, columns="analysis_type", values="sig", aggfunc="first").astype(float)
    wide = wide.reset_index()
    u = q[q.cls.notna()].merge(wide, on=KEYS, how="inner")
    ps = psth_metrics()
    u = u.merge(ps, on=["session_id", "electrode_group", "cluster_id"], how="left")
    u.loc[u.quality_label != "good", ["early_z", "peak_lat_ms"]] = np.nan
    resp = u.whisker_passive_pre == 1
    u.loc[~resp, "peak_lat_ms"] = np.nan
    feats = TYPES + ["early_z", "peak_lat_ms"]
    ip = u.groupby(["cls", "ins"]).agg(n=("cluster_id", "size"), **{f: (f, "mean") for f in feats}).reset_index()
    ip = ip[ip.n >= 10]
    ip.to_csv(OUT / "insertion_profiles.csv", index=False)
    rows = []
    for c in ORDER:
        g = ip[ip.cls == c]
        r = dict(cls=c, n_insertions=len(g), n_units=int(g.n.sum()))
        for f in feats:
            lo, hi = boot_ci(g[f])
            r[f] = g[f].mean()
            r[f + "_lo"], r[f + "_hi"] = lo, hi
        rows.append(r)
    cs = pd.DataFrame(rows)
    cs.to_csv(OUT / "class_summary.csv", index=False)
    print(cs[["cls", "n_insertions", "n_units"] + feats].round(3).to_string())

    # within probe: SSp-un / SSp-tr insertions, every class on the same insertion
    wp = ip[ip.ins.isin(inv.session_id + "|" + inv.electrode_group)].sort_values(["ins", "cls"])
    wp.to_csv(OUT / "within_probe.csv", index=False)
    print(wp.round(3).to_string())

    # profile distance, bootstrap over insertions
    prof_feats = TYPES + ["early_z"]
    zsc = ip[prof_feats].std()
    near = []
    for target in ("SSp-un", "SSp-tr"):
        cnt = {c: 0 for c in ORDER[2:]}
        dsum = {c: [] for c in ORDER[2:]}
        tg = ip[ip.cls == target]
        for _ in range(NBOOT):
            t = tg.sample(len(tg), replace=True, random_state=RNG.integers(1e9))[prof_feats].mean() / zsc
            dd = {}
            for c in ORDER[2:]:
                g = ip[ip.cls == c]
                m = g.sample(len(g), replace=True, random_state=RNG.integers(1e9))[prof_feats].mean() / zsc
                dd[c] = float(np.sqrt(((t - m) ** 2).sum()))
                dsum[c].append(dd[c])
            cnt[min(dd, key=dd.get)] += 1
        for c in ORDER[2:]:
            near.append(dict(target=target, cls=c, frac_nearest=cnt[c] / NBOOT, dist_median=np.median(dsum[c]),
                             dist_lo=np.percentile(dsum[c], 2.5), dist_hi=np.percentile(dsum[c], 97.5)))
    near = pd.DataFrame(near)
    near.to_csv(OUT / "nearest_class_bootstrap.csv", index=False)
    print(near.round(3).to_string())

    # ---------------- figure
    lab_f = {"whisker_passive_pre": "Whisker passive", "auditory_passive_pre": "Auditory passive",
             "whisker_active": "Whisker active", "auditory_active": "Auditory active",
             "spontaneous_licks": "Spontaneous licks", "choice": "Choice", "early_z": "Whisker 5-30 ms (z)",
             "peak_lat_ms": "Whisker peak latency (ms)"}
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.6))
    for ax, f in zip(axs.ravel(), feats):
        for k, c in enumerate(ORDER):
            g = ip[ip.cls == c][f].dropna()
            ax.scatter(k + RNG.uniform(-0.15, 0.15, len(g)), g, s=7, color=COLC[c], alpha=0.5, lw=0)
            r = cs[cs.cls == c].iloc[0]
            ax.errorbar(k, r[f], [[r[f] - r[f + "_lo"]], [r[f + "_hi"] - r[f]]], color="k", marker="o", ms=3,
                        capsize=2, lw=0.8)
        ax.set_xticks(range(len(ORDER)), ORDER, rotation=45, fontsize=7)
        ax.set_title(lab_f[f] + ("" if f in ("early_z", "peak_lat_ms") else " (% sig.)"), fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SSp-un / SSp-tr vs neighbouring somatosensory and parietal areas (dots = insertions, "
                 "mean and 95% bootstrap CI over insertions; all days, cohorts pooled)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"review.{ext}", dpi=200)


if __name__ == "__main__":
    main()
