"""005 figures -- global (NWB CCF) vs local (D2) registration on each brain's own tissue, and the C2 functional test.

Inputs: 005_registration_validation.py compute (tables/005_l4_<mouse>.npz, 005_units_flat_<mouse>.parquet), 002 responses.
Everything is drawn in the brain's own flatmap (before Lana's barrel correction), so both registrations sit on exactly
the same tissue image and only the barrel outlines and unit positions differ:
  global = Blue Brain barrels of the coarse (brainreg) registration, units at their NWB CCF (inverse field);
  D2 = Lana's corrected barrels (barrels_corrected_full), units on the physical DiI track (001 track transfer).
Per brain: 005_tissue_<mouse> (tissue alone | global | D2). Population: 005_functional_c2_<set> (whisker response vs
distance from C2 under each registration, plus the two mixed routes that separate the barrel map from the unit
placement), 005_fit_qc (Lana's landmark error and Dice, coarse placement vs D2).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 005_plot.py [mice]
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

_spec = importlib.util.spec_from_file_location("val005", Path(__file__).with_name("005_registration_validation.py"))
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)
m3, TABLES, OUT, LO, PIX, SHAPE = v.m3, v.TABLES, v.OUT, v.LO, v.PIX, v.SHAPE

plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
C2 = 9
LG = {"L2/3": ["1", "2/3"], "L4": ["4"], "L5/6": ["5", "6a", "6b"]}
LG_COLORS = {"L2/3": "#74c476", "L4": "#ff7f0e", "L5/6": "#3182bd"}
GLOBAL_C, D2_C = "#ffd400", "#00e5ff"
ROUTES = {  # name: (barrel map, unit position prefix, unit barrel column, unit layer column)
    "Global registration (NWB CCF)": ("lab_glob_glob", "glob_direct", "glob_barrel", "glob_layer"),
    "Local barrel registration (D2)": ("lab_d2_glob", "glob_trk", "trk_d2_barrel", "trk_d2_layer"),
}
MIXED = {  # C2 distance routes: barrel map x unit placement
    "global": ("lab_glob_glob", "glob_direct"),
    "D2": ("lab_d2_glob", "glob_trk"),
    "D2 barrels, NWB-CCF units": ("lab_d2_glob", "glob_direct"),
    "global barrels, DiI-track units": ("lab_glob_glob", "glob_trk"),
}
MIXED_COLORS = {"global": "#d4a000", "D2": "#008b9e", "D2 barrels, NWB-CCF units": "#7fc5ce",
                "global barrels, DiI-track units": "#e8cf7a"}
BINS = [0, 150, 300, 500, 800, 3000]
BIN_LABELS = ["<150", "150-300", "300-500", "500-800", ">800"]


def extent():
    return [LO[0], LO[0] + SHAPE[0] * PIX, LO[1], LO[1] + SHAPE[1] * PIX]


def tissue_rgb(z):
    """Bandpassed L4 autofluorescence (grey) with the DiI column maximum in red; image rows = dorsal, columns = anterior."""
    img = z["l4_glob"]
    valid = np.isfinite(img)
    bp = v.bandpass(np.where(valid, img, np.nanmedian(img)), valid)
    lo, hi = np.nanpercentile(bp[valid], [2, 98])
    g = np.clip((bp - lo) / (hi - lo), 0, 1)
    dii = z["dii_glob"]
    dv = np.isfinite(dii)
    base = np.nanpercentile(dii[dv], 50)
    r = np.clip((np.where(dv, dii, base) - base) / (np.nanpercentile(dii[dv], 99.7) - base), 0, 1)
    rgb = np.stack([np.maximum(g, r), g * (1 - 0.8 * r), g * (1 - 0.8 * r)], -1)
    rgb[~valid] = 1.0
    return np.transpose(rgb, (1, 0, 2))


def outlines(ax, lab, color, lw=0.45):
    cols = np.where(lab < 34, lab, 0).T
    ax.contour(find_boundaries(cols, mode="inner").astype(float), levels=[0.5], colors=color, linewidths=lw,
               extent=extent(), origin="lower")
    if (cols == C2).any():
        ax.contour((cols == C2).astype(float), levels=[0.5], colors=color, linewidths=lw * 3, extent=extent(),
                   origin="lower")
    for k in np.unique(cols[cols > 0]):
        yy, xx = np.nonzero(cols == k)
        ax.text(LO[0] + (xx.mean() + 0.5) * PIX, LO[1] + (yy.mean() + 0.5) * PIX, m3.NAMES[int(k)], color=color,
                fontsize=4.5, ha="center", va="center", fontweight="bold", clip_on=True)


def centroid(lab, k=C2):
    ii, jj = np.nonzero(lab == k)
    return np.array([LO[0] + (ii.mean() + 0.5) * PIX, LO[1] + (jj.mean() + 0.5) * PIX]) if len(ii) else np.full(2, np.nan)


def layer_group(layer):
    return layer.fillna("").map({l: g for g, ls in LG.items() for l in ls})


def calls(d, col):
    x = d[d[col].fillna("") != ""][col].value_counts()
    return ", ".join(f"{b} {n}" for b, n in x.head(3).items()) or "no barrel"


def decorate(ax, c):
    ax.set_xlim(c[0] - 900, c[0] + 900)
    ax.set_ylim(c[1] + 900, c[1] - 900)  # row A on top, as in 004
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    x0, x1 = ax.get_xlim()
    y0 = ax.get_ylim()[0]
    ax.plot([x1 - 600, x1 - 100], [y0 - 90] * 2, color="k", lw=1.2)
    ax.text(x1 - 350, y0 - 120, "500 µm", ha="center", va="bottom", fontsize=5)
    ax.annotate("", (x0 + 450, y0 - 90), (x0 + 100, y0 - 90), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6))
    ax.text(x0 + 275, y0 - 120, "anterior", ha="center", va="bottom", fontsize=5)


dii_rows = []


def dii_check(mouse, z, u, radius=450.0):
    """Per probe: DiI footprint in the tissue (pixels within `radius` of the midpoint of both routes' cortical units
    whose DiI column maximum is above the field's 99.5th percentile; intensity-weighted centroid) and the distance of
    each route's cortical units (median position) to it. A probe counts as having a visible DiI spot if >= 5 such
    pixels and their mean is >= 2x the local median."""
    dii = z["dii_glob"]
    fin = np.isfinite(dii)
    thr = np.nanpercentile(dii[fin], 99.5)
    ii, jj = np.nonzero(fin)
    P = np.c_[LO[0] + (ii + 0.5) * PIX, LO[1] + (jj + 0.5) * PIX]
    val = dii[ii, jj]
    rows = []
    for (sid, grp), d in u.groupby(["session_id", "electrode_group"]):
        ctx = d[d.trk_d2_layer.fillna("").ne("") | d.glob_layer.fillna("").ne("")]
        pos = {r: ctx[[f"{p}_ant", f"{p}_dor"]].median().to_numpy() for r, p in
               (("global", "glob_direct"), ("D2", "glob_trk"))}
        if not all(np.isfinite(x).all() for x in pos.values()):
            continue
        mid = (pos["global"] + pos["D2"]) / 2
        near = np.linalg.norm(P - mid, axis=1) <= radius
        hot = near & (val >= thr)
        visible = hot.sum() >= 5 and val[hot].mean() >= 2 * np.median(val[near])
        c = (P[hot] * val[hot, None]).sum(0) / val[hot].sum() if hot.any() else np.full(2, np.nan)
        rows.append(dict(mouse_id=mouse, session_id=sid, electrode_group=grp, n_units=len(ctx), dii_visible=visible,
                         n_dii_pixels=int(hot.sum()), dii_contrast=val[hot].mean() / np.median(val[near]) if hot.any() else np.nan,
                         dist_global_um=np.linalg.norm(pos["global"] - c), dist_D2_um=np.linalg.norm(pos["D2"] - c),
                         global_vs_D2_um=np.linalg.norm(pos["global"] - pos["D2"])))
    return rows


def dii_figure(t):
    t = t[t.dii_visible]
    fig, ax = plt.subplots(figsize=(2.4, 2.3), constrained_layout=True)
    m = max(t.dist_global_um.max(), t.dist_D2_um.max()) * 1.05
    ax.plot([0, m], [0, m], color="0.6", lw=0.5, ls="--")
    ax.scatter(t.dist_global_um, t.dist_D2_um, s=8, color="k", lw=0)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_aspect("equal")
    ax.set_xlabel("Global (NWB CCF) units to DiI spot (µm)")
    ax.set_ylabel("D2 (DiI track) units to DiI spot (µm)")
    w = stats.wilcoxon(t.dist_global_um, t.dist_D2_um) if len(t) > 5 else None
    ax.set_title(f"{len(t)} probes with a visible DiI spot\nmedian {t.dist_global_um.median():.0f} vs "
                 f"{t.dist_D2_um.median():.0f} µm" + (f", Wilcoxon P = {w.pvalue:.2g}" if w else ""), fontsize=6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"005_dii_distance.{ext}", dpi=400)
    plt.close(fig)


def brain_figure(mouse):
    z = np.load(TABLES / f"005_l4_{mouse}.npz")
    u = pd.read_parquet(TABLES / f"005_units_flat_{mouse}.parquet")
    u = u[u.bc_label.isin(["good", "mua"])]
    rgb = tissue_rgb(z)
    tracks = {k.split("::")[1]: z[k] for k in z.files if k.startswith("track_glob::")}
    probes = sorted(u.track_file.dropna().unique())
    pos = u[["glob_trk_ant", "glob_trk_dor"]].to_numpy()
    pos = pos[np.isfinite(pos).all(1)]
    c = np.median(pos, 0) if len(pos) else centroid(z["lab_d2_glob"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    axes[0].set_title("Tissue only: L4 autofluorescence (grey),\nDiI column maximum (red), clicked DiI track (dotted)")
    for ax in axes:
        ax.imshow(rgb, extent=extent(), origin="lower", interpolation="nearest")
        for rel in probes:
            t = tracks.get(rel)
            if t is not None:
                ax.plot(t[:, 0], t[:, 1], color="#ff3030", lw=0.6, ls=":", zorder=4)
    for ax, (name, (lab, p, bcol, lcol)), color in zip(axes[1:], ROUTES.items(), (GLOBAL_C, D2_C)):
        outlines(ax, z[lab], color)
        g = layer_group(u[lcol])
        for lg, lc in LG_COLORS.items():
            x = u[g == lg]
            ax.scatter(x[f"{p}_ant"], x[f"{p}_dor"], s=3, color=lc, edgecolor="k", lw=0.15, zorder=5, label=lg)
        txt = "\n".join(f"{r.split('/')[-1].replace('.points', '')}"
                        f"{' ' + r.split('/')[0][9:15] if '/' in r else ''}: {calls(u[u.track_file == r], bcol)}"
                        for r in probes)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=4.5, va="top", bbox=dict(fc="white", ec="none",
                alpha=0.85, pad=1), zorder=6)
        ax.set_title(f"{name}\nunits by layer (that registration's layers)")
    axes[2].legend(frameon=True, loc="lower right", fontsize=5, markerscale=2, framealpha=0.85, edgecolor="none")
    for ax in axes:
        decorate(ax, c)
    fig.suptitle(f"{mouse}: one tissue image (the brain's own L4 flatmap), the barrels and unit positions of each "
                 f"registration; thick outline = C2", fontsize=6.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"005_tissue_{mouse}.{ext}", dpi=400)
    plt.close(fig)
    dii_rows.extend(dii_check(mouse, z, u))
    out = u[["mouse_id", "session_id", "electrode_group", "cluster_id", "track_file", "glob_layer", "trk_d2_layer",
             "glob_barrel", "trk_d2_barrel"]].copy()
    for name, (lab, p) in MIXED.items():
        out[f"dist_c2::{name}"] = np.linalg.norm(u[[f"{p}_ant", f"{p}_dor"]].to_numpy() - centroid(z[lab]), axis=1)
    return out


def functional_figure(df, tag):
    """Whisker response vs distance from the C2 centre under each route, L2/3 and L4 units (that map's layers)."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    rows = []
    x = np.arange(len(BINS) - 1)
    for name in MIXED:
        lay = df["trk_d2_layer" if name.startswith("D2") else "glob_layer"].fillna("")
        dc = f"dist_c2::{name}"
        d = df[lay.isin(["2/3", "4"]) & np.isfinite(df[dc]) & df.whisker_auc.notna()]
        b = pd.cut(d[dc], BINS)
        g = d.groupby(b, observed=False)
        resp = g.whisker_responsive.apply(lambda s: s.astype(float).mean())
        auc = g.whisker_auc.median()
        lat = d[d.whisker_responsive.fillna(False).astype(bool)].groupby(b, observed=False).whisker_latency_ms.median()
        main = name in ("global", "D2")
        for ax, y in zip(axes[:3], (resp, auc, lat)):
            ax.plot(x, y.values, "-" if main else "--", color=MIXED_COLORS[name], marker="o", ms=2.5 if main else 1.5,
                    lw=1.2 if main else 0.7, label=name)
        r = stats.spearmanr(d[dc], d.whisker_auc)
        rows.append(dict(route=name, n_units=len(d), n_mice=d.mouse_id.nunique(), spearman_auc_vs_dist=r.statistic,
                         p=r.pvalue, n_within150=int((d[dc] < 150).sum()),
                         frac_resp_within150=d[d[dc] < 150].whisker_responsive.astype(float).mean(),
                         frac_resp_beyond500=d[d[dc] > 500].whisker_responsive.astype(float).mean()))
    for ax, yl in zip(axes[:3], ("Fraction whisker-responsive", "Whisker AUC (median)",
                                 "Whisker latency, responsive (ms)")):
        ax.set_xticks(x, BIN_LABELS, rotation=30)
        ax.set_xlabel("Distance from C2 centre (µm)")
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False, fontsize=4.5)
    tab = pd.DataFrame(rows)
    ax = axes[3]
    ax.barh(range(len(tab)), tab.spearman_auc_vs_dist, color=[MIXED_COLORS[n] for n in tab.route])
    ax.set_yticks(range(len(tab)), [n.replace(", ", ",\n") for n in tab.route], fontsize=4.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("Spearman r, AUC vs distance\n(more negative = tuned to C2)")
    fig.suptitle(f"{tag}: L2/3 and L4 units, {df.mouse_id.nunique()} mice; the right C2 whisker is stimulated, so the "
                 f"correct map should show responses peaking at C2", fontsize=6.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"005_functional_c2_{tag}.{ext}", dpi=400)
    plt.close(fig)
    return tab.assign(set=tag)


def qc_figure():
    s = pd.read_csv(v.D2 / "summary.csv")
    inv = pd.read_csv(TABLES / "000_inventory.csv")
    s = s[s.brain.isin(inv.d2_folder)]
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.1), constrained_layout=True)
    for ax, (a, b, yl) in zip(axes, [("err_initial_um", "err_tps_loo_um", "Barrel landmark error vs tissue (µm)"),
                                     ("dice_initial", "dice_after", "Overlap with tissue barrels (Dice)")]):
        for r in s.itertuples():
            ax.plot([0, 1], [getattr(r, a), getattr(r, b)], color="0.7", lw=0.4)
        ax.boxplot([s[a].dropna(), s[b].dropna()], positions=[0, 1], widths=0.35, showfliers=False)
        ax.set_xticks([0, 1], ["Global\n(coarse placement)", "D2\n(local fit; error left-one-out)"])
        ax.set_ylabel(yl)
        ax.set_title(f"median {s[a].median():.2f} -> {s[b].median():.2f}" if "dice" in a else
                     f"median {s[a].median():.0f} -> {s[b].median():.0f} µm")
    fig.suptitle(f"Lana's barrel fit QC, {len(s)} brains", fontsize=6.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"005_fit_qc.{ext}", dpi=400)
    plt.close(fig)
    return s


def main(mice):
    OUT.mkdir(parents=True, exist_ok=True)
    mice = mice or sorted(p.stem.split("_")[-1] for p in TABLES.glob("005_l4_*.npz"))
    dist = []
    for m in mice:
        try:
            dist.append(brain_figure(m))
            print("figure", m, flush=True)
        except Exception as e:  # noqa: BLE001
            print(m, "FAILED", type(e).__name__, e, flush=True)
    keys = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
    df = pd.concat(dist, ignore_index=True)
    resp = pd.concat([pd.read_parquet(TABLES / f"002_responses_{m}.parquet") for m in df.mouse_id.unique()],
                     ignore_index=True)
    for d in (df, resp):
        d["cluster_id"] = d.cluster_id.astype(float)
    df = df.merge(resp, on=keys, how="left")
    df.to_parquet(TABLES / "005_c2_distance_units.parquet")
    inv = pd.read_csv(TABLES / "000_inventory.csv").drop_duplicates("mouse_id").set_index("mouse_id")
    vis = df.mouse_id.map(inv.barrels_visible).fillna("").eq("yes")
    tabs = pd.concat([functional_figure(df, "all"), functional_figure(df[vis], "visible")])
    tabs.to_csv(TABLES / "005_functional_c2.csv", index=False)
    print(tabs.round(3).to_string(index=False))
    t = pd.DataFrame(dii_rows)
    t.to_csv(TABLES / "005_dii_check.csv", index=False)
    dii_figure(t)
    print(t.round(1).to_string(index=False))
    s = qc_figure()
    print(s[["err_initial_um", "err_tps_loo_um", "dice_initial", "dice_after"]].describe().round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
