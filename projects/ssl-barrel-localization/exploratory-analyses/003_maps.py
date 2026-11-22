"""003 -- Localization and functional maps of the linked units (001 track transfer `trk_`, 002), good / mua units on wS1 probes.

Per mouse (one row): flat L4 map of the barrel field (D2 template, final user-picked fits, all brains in the same flat frame; C2 shaded)
with the units at their corrected flat coordinate, coloured by (1) layer, (2) whisker AUC (evoked vs baseline),
(3) whisker latency, (4) auditory AUC; (5) depth profile: whisker delta rate against the unit depth on the probe,
coloured by barrel. Global vs barrel-specific registration: shift between the global CCF (IBL alignment) and the D2
corrected CCF per unit, and global vs D2 barrel per unit. Pooled: whisker / auditory responsiveness and latency by barrel class (C2, C2 neighbours,
other barrels, septa / outside barrels within SSp-bfd) and layer. Tables: per-probe barrels and layers crossed.

Outputs: combined_results_ks4/ssl-barrel-localization/figures/ (per mouse in per_mouse/: flat maps, global vs D2),
tables/003_*.csv. Summaries pooled over all brains and over brains with visible barrels (Lana's visual check), by cohort
(reward_group from the reference sheet).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 003_maps.py AB156 AB164 MH064   [no args: every 002 table]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

D2 = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ws1-barrel-registration/D2_all_brains")  # template.npz (same file as Lana's D2 run)
ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-barrel-localization"
TABLES, FIGS = ROOT / "tables", ROOT / "figures"
TPL = np.load(D2 / "template.npz")
NAMES = pd.read_csv(D2 / "barrel_index.csv").set_index("index").barrel.to_dict()
NEIGHBOURS = {"B1", "B2", "B3", "C1", "C3", "D1", "D2", "D3"}
LAYERS = ["1", "2/3", "4", "5", "6a", "6b"]
LAYER_COLORS = dict(zip(LAYERS, ["#c7e9c0", "#74c476", "#e6550d", "#3182bd", "#756bb1", "#bcbddc"]))
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})


def load(mouse):
    u = pd.read_parquet(TABLES / f"001_units_{mouse}.parquet")
    r = pd.read_parquet(TABLES / f"002_responses_{mouse}.parquet")
    for d in (u, r):
        d["cluster_id"] = d.cluster_id.astype(float)
    df = u.merge(r, on=["mouse_id", "session_id", "electrode_group", "cluster_id"], how="left")
    df = df[df.bc_label.isin(["good", "mua"]) & df.track_file.notna()]
    glob = df.get("registration", pd.Series("D2", index=df.index)).eq("global")
    df = df[glob | df.trk_brainreg_region.notna()]  # D2 brains: units placed on their probe's DiI track
    glob = glob.reindex(df.index)
    df["barrel"] = df.trk_d2_barrel.fillna("")
    df["layer"] = df.trk_d2_layer.fillna("").replace("", np.nan).fillna(df.trk_brainreg_layer.replace("", np.nan))
    # brains whose local fit was rejected (final decision use_global): the global registration of the NWB files, i.e.
    # barrel / layer of the annotation at the unit's NWB CCF and its global flat position
    df.loc[glob, "barrel"] = df.loc[glob, "glob_barrel"].fillna("")
    df.loc[glob, "layer"] = df.loc[glob, "glob_layer"].replace("", np.nan)
    df.loc[glob, "trk_flat_anterior_um"] = df.loc[glob, "glob_flat_anterior_um"]
    df.loc[glob, "trk_flat_dorsal_um"] = df.loc[glob, "glob_flat_dorsal_um"]
    df["registration"] = np.where(glob, "global", "D2")
    in_bfd = df.glob_region.fillna("").str.startswith("SSp-bfd") | (df.barrel != "")
    df["barrel_class"] = np.select([df.barrel == "C2", df.barrel.isin(NEIGHBOURS), df.barrel != "", in_bfd],
                                   ["C2", "C2 neighbours", "other barrels", "septa / between columns"], "outside SSp-bfd")
    return df


def barrel_map(ax):
    lab = TPL["alab"]
    cols = np.where(lab < 34, lab, 0)
    lo, pix = TPL["lo"], float(TPL["pix"])
    ext = [lo[1], lo[1] + cols.shape[1] * pix, lo[0], lo[0] + cols.shape[0] * pix]  # x = dorsal, y = anterior
    ax.imshow(np.where(cols == 9, 0.85, 1.0), cmap="gray", vmin=0, vmax=1, extent=ext, origin="lower", interpolation="nearest")
    ax.contour(find_boundaries(cols, mode="inner").astype(float), levels=[0.5], colors="0.55", linewidths=0.3,
               extent=ext, origin="lower")
    for k in np.unique(cols[cols > 0]):
        yy, xx = np.nonzero(cols == k)
        ax.text(lo[1] + xx.mean() * pix, lo[0] + yy.mean() * pix, NAMES.get(int(k), ""), fontsize=4, ha="center",
                va="center", color="0.4")
    ax.set_aspect("equal")
    ax.set_xlabel("Flat dorsal (um)")
    ax.set_ylabel("Flat anterior (um)")
    return ext


def flat_panel(ax, df, col, title, cmap, norm=None, vmin=None, vmax=None):
    ext = barrel_map(ax)
    d = df[np.isfinite(df.trk_flat_anterior_um) & df[col].notna()]
    sc = ax.scatter(d.trk_flat_dorsal_um, d.trk_flat_anterior_um, c=d[col], cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                    s=5, lw=0, alpha=0.8, zorder=3)
    if len(d):
        pad = 450
        ax.set_xlim(max(ext[0], d.trk_flat_dorsal_um.min() - pad), min(ext[1], d.trk_flat_dorsal_um.max() + pad))
        ax.set_ylim(max(ext[2], d.trk_flat_anterior_um.min() - pad), min(ext[3], d.trk_flat_anterior_um.max() + pad))
    ax.set_title(title, fontsize=7)
    return sc


def mouse_row(axes, df, mouse):
    lay = df.assign(layer_code=df.layer.map({l: i for i, l in enumerate(LAYERS)}))
    ext = barrel_map(axes[0])
    d = lay[np.isfinite(lay.trk_flat_anterior_um)]
    for l in LAYERS:
        x = d[d.layer == l]
        axes[0].scatter(x.trk_flat_dorsal_um, x.trk_flat_anterior_um, s=5, lw=0, color=LAYER_COLORS[l], label=f"L{l}", zorder=3)
    if len(d):
        axes[0].set_xlim(max(ext[0], d.trk_flat_dorsal_um.min() - 450), min(ext[1], d.trk_flat_dorsal_um.max() + 450))
        axes[0].set_ylim(max(ext[2], d.trk_flat_anterior_um.min() - 450), min(ext[3], d.trk_flat_anterior_um.max() + 450))
    axes[0].legend(fontsize=4.5, frameon=False, markerscale=1.5, loc="upper right")
    axes[0].set_title(f"{mouse}: units by layer", fontsize=7)
    for ax, col, title in [(axes[1], "whisker_auc", "Whisker AUC (evoked vs baseline)"),
                           (axes[3], "auditory_auc", "Auditory AUC")]:
        sc = flat_panel(ax, df, col, title, "RdBu_r", norm=TwoSlopeNorm(0.5, 0.2, 0.8))
        plt.colorbar(sc, ax=ax, shrink=0.6)
    sc = flat_panel(axes[2], df[df.whisker_responsive.fillna(False).astype(bool)], "whisker_latency_ms",
                    "Whisker latency (ms), responsive units", "viridis", vmin=5, vmax=30)
    plt.colorbar(sc, ax=axes[2], shrink=0.6)
    ax = axes[4]
    for b, x in df.groupby("barrel"):
        ax.scatter(x.whisker_delta_hz, x.depth, s=5, lw=0, label=b or "none", zorder=3,
                   color="#009E73" if b == "C2" else ("0.75" if not b else None))
    ax.axvline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xlabel("Whisker evoked - baseline (Hz)")
    ax.set_ylabel("Depth on probe from tip (um; surface up)")
    top = df.depth.max()
    ax.set_ylim(top - 1600, top + 50)
    ax.legend(fontsize=4.5, frameon=False, markerscale=1.5, ncol=2)
    ax.set_title("Depth profile (upper 1.6 mm)", fontsize=7)


def registration_figure(data, path):
    """Global (IBL alignment, brainreg) vs barrel-specific (D2) position of every unit in a D2 barrel column: shift
    magnitude, horizontal (AP x ML) arrows from global to corrected position, and global vs D2 barrel counts."""
    mice = list(data)
    fig, axes = plt.subplots(2, len(mice), figsize=(3.2 * len(mice), 6), constrained_layout=True, squeeze=False)
    rows = []
    for j, m in enumerate(mice):
        d = data[m][data[m].barrel != ""]
        ax = axes[0, j]
        ax.hist(d.trk_shift_um, bins=np.arange(0, 800, 20), color="0.4", lw=0)
        ax.set_xlabel("|D2 (on DiI track) - global CCF| (um)")
        ax.set_ylabel("Units")
        ax.set_title(f"{m}: median {d.trk_shift_um.median():.0f} um", fontsize=7)
        ax = axes[1, j]
        ax.quiver(d.ccf_atlas_ml, d.ccf_atlas_ap, d.trk_shift_ml_um, d.trk_shift_ap_um, angles="xy", scale_units="xy",
                  scale=1, width=0.004, color="0.5")
        for b, x in d.groupby("barrel"):
            ax.scatter(x.trk_ccf_ml_um, x.trk_ccf_ap_um, s=4, lw=0, label=b, zorder=3)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("CCF ML (um)")
        ax.set_ylabel("CCF AP (um)")
        ax.legend(fontsize=4.5, frameon=False, title="D2 barrel", title_fontsize=5)
        rows.append(pd.crosstab(d.glob_barrel.replace("", "none"), d.barrel).stack().rename("n_units").reset_index()
                    .assign(mouse_id=m))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    tab = pd.concat(rows, ignore_index=True)
    return tab[tab.n_units > 0].rename(columns={"glob_barrel": "global_barrel", "barrel": "d2_barrel"})


def summary(df):
    s = (df[df.barrel_class != "outside SSp-bfd"].groupby(["barrel_class", "layer"])
         .agg(n_units=("cluster_id", "size"), n_mice=("mouse_id", "nunique"),
              whisker_responsive=("whisker_responsive", "mean"), whisker_auc=("whisker_auc", "median"),
              whisker_latency_ms=("whisker_latency_ms", "median"), auditory_responsive=("auditory_responsive", "mean"),
              auditory_auc=("auditory_auc", "median")).reset_index())
    return s


def summary_figure(df, path):
    order = ["C2", "C2 neighbours", "other barrels", "septa / between columns"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), constrained_layout=True)
    d = df[df.barrel_class.isin(order)]
    for ax, (col, ylabel) in zip(axes, [("whisker_responsive", "Fraction whisker responsive"),
                                        ("whisker_latency_ms", "Whisker latency (ms)"),
                                        ("auditory_responsive", "Fraction auditory responsive")]):
        for i, cls in enumerate(order):
            for j, l in enumerate(LAYERS[:5]):
                x = d[(d.barrel_class == cls) & (d.layer == l)][col].dropna().astype(float)
                if len(x) < 3:
                    continue
                v = x.mean() if col.endswith("responsive") else x.median()
                ax.bar(i + (j - 2) * 0.15, v, width=0.15, color=LAYER_COLORS[l], lw=0, label=f"L{l}" if i == 0 else None)
        ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=5, frameon=False)
    fig.suptitle(f"{df.mouse_id.nunique()} mice, {len(df)} good/mua units", fontsize=7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def cohorts():
    sheet = next(p for p in (Path("/mnt/share_internal"), Path(r"M:\share_internal")) if p.exists())
    ref = pd.read_excel(sheet / "Axel_Bisi_Share" / "dataset_info" / "joint_mouse_reference_weight.xlsx")
    return ref.set_index("mouse_id").reward_group.to_dict()


def main(mice):
    mice = mice or sorted(p.stem.split("_")[-1] for p in TABLES.glob("002_responses_*.parquet"))
    tag = "all" if len(mice) > 5 else "_".join(mice)
    per_mouse = FIGS / "per_mouse"
    per_mouse.mkdir(parents=True, exist_ok=True)
    data = {}
    for m in mice:
        d = load(m)
        if len(d) == 0:
            print(m, "no units on a DiI track; skipped")
            continue
        data[m] = d
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.3), constrained_layout=True)
        mouse_row(axes, d, m)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(per_mouse / f"003_flat_maps_{m}.{ext}", dpi=300)
        plt.close(fig)
        registration_figure({m: d}, per_mouse / f"003_global_vs_d2_{m}")
    inv = pd.read_csv(TABLES / "000_inventory.csv").drop_duplicates("mouse_id").set_index("mouse_id")
    coh = cohorts()
    df = pd.concat(data.values(), ignore_index=True)
    df["cohort"] = df.mouse_id.map(coh)
    df["barrels_visible"] = df.mouse_id.map(inv.barrels_visible).fillna("").eq("yes")
    reg = pd.concat([registration_figure({m: d}, per_mouse / f"003_global_vs_d2_{m}") for m, d in data.items()])
    reg.to_csv(TABLES / f"003_global_vs_d2_barrel_{tag}.csv", index=False)
    for name, sub in [("visible", df[df.barrels_visible]), ("allbrains", df)]:
        for coh_name, x in [("pooled", sub), *((c, sub[sub.cohort == c]) for c in ("R+", "R-"))]:
            if len(x):
                summary(x).to_csv(TABLES / f"003_summary_{tag}_{name}_{coh_name}.csv", index=False)
                summary_figure(x, FIGS / f"003_summary_{tag}_{name}_{coh_name}")
    probes = (df.groupby(["mouse_id", "session_id", "electrode_group", "track_file"])
              .apply(lambda x: pd.Series({"n_units": len(x),
                                          "barrels": " ".join(f"{b}:{n}" for b, n in x.barrel.replace("", np.nan).dropna().value_counts().items()),
                                          "layers_in_barrels": " ".join(f"L{l}:{n}" for l, n in x[x.barrel != ""].layer.value_counts().items()),
                                          "n_whisker_responsive": int(x.whisker_responsive.fillna(False).sum())}), include_groups=False)
              .reset_index())
    probes.to_csv(TABLES / f"003_probes_{tag}.csv", index=False)
    print(probes.to_string(index=False))
    print(summary(df[df.barrels_visible]).round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
