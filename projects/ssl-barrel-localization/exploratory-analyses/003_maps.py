"""003 -- Localization and functional maps of the linked units (001 track route `trk_`, 002 responses), good / mua units.

Per mouse (one row): flat L4 map of the barrel field (Lana's D2 template, all brains in the same flat frame; C2 shaded)
with the units at their corrected flat coordinate, coloured by (1) layer, (2) whisker AUC (evoked vs baseline),
(3) whisker latency, (4) auditory AUC; (5) depth profile: whisker delta rate against the distance from the brain
surface, coloured by barrel. Pooled: whisker / auditory responsiveness and latency by barrel class (C2, C2 neighbours,
other barrels, septa / outside barrels within SSp-bfd) and layer. Tables: per-probe barrels and layers crossed.

Outputs: combined_results_ks4/ssl-barrel-localization/figures/003_*.{png,pdf,svg}, tables/003_*.csv
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

D2 = Path("/raid0/lmsmith/local_barrel_reg/D2_neuropixels")
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
    df = df[df.bc_label.isin(["good", "mua"]) & df.trk_file.notna()]
    df["barrel"] = df.trk_d2_barrel.fillna("")
    df["layer"] = df.trk_d2_layer.fillna("").replace("", np.nan).fillna(df.trk_brainreg_layer.replace("", np.nan))
    in_bfd = df.trk_brainreg_region.fillna("").str.startswith("SSp-bfd") | (df.barrel != "")
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
        ax.scatter(x.whisker_delta_hz, x.d_surface_um, s=5, lw=0, label=b or "none", zorder=3,
                   color="#009E73" if b == "C2" else ("0.75" if not b else None))
    ax.invert_yaxis()
    ax.axvline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xlabel("Whisker evoked - baseline (Hz)")
    ax.set_ylabel("Distance from brain surface (um)")
    ax.set_ylim(1600, -50)
    ax.legend(fontsize=4.5, frameon=False, markerscale=1.5, ncol=2)
    ax.set_title("Depth profile (cortex)", fontsize=7)


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
    fig.suptitle(f"Pilot: {', '.join(sorted(df.mouse_id.unique()))} (good/mua units, track route)", fontsize=7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main(mice):
    mice = mice or sorted(p.stem.split("_")[-1] for p in TABLES.glob("002_responses_*.parquet"))
    FIGS.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in mice}
    fig, axes = plt.subplots(len(mice), 5, figsize=(16, 3.3 * len(mice)), constrained_layout=True, squeeze=False)
    for row, m in zip(axes, mice):
        mouse_row(row, data[m], m)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"003_flat_maps_{'_'.join(mice)}.{ext}", dpi=300)
    plt.close(fig)
    df = pd.concat(data.values(), ignore_index=True)
    probes = (df.groupby(["mouse_id", "session_id", "electrode_group", "trk_file"])
              .apply(lambda x: pd.Series({"n_units": len(x),
                                          "barrels": " ".join(f"{b}:{n}" for b, n in x.barrel.replace("", np.nan).dropna().value_counts().items()),
                                          "layers_in_barrels": " ".join(f"L{l}:{n}" for l, n in x[x.barrel != ""].layer.value_counts().items()),
                                          "n_whisker_responsive": int(x.whisker_responsive.fillna(False).sum())}), include_groups=False)
              .reset_index())
    probes.to_csv(TABLES / f"003_probes_{'_'.join(mice)}.csv", index=False)
    summary(df).to_csv(TABLES / f"003_summary_{'_'.join(mice)}.csv", index=False)
    summary_figure(df, FIGS / f"003_summary_{'_'.join(mice)}")
    print(probes.to_string(index=False))
    print(summary(df).round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
