"""004 -- Top-down barrel-field figures of probes, units and their whisker / auditory responses (all brains pooled).

Every brain is in Lana's common flat frame (D2 template; units placed on their DiI track, 001 `trk_flat_*`), drawn
as the standard barrel-map schematic: anterior to the right (arcs 1 -> 8 left to right, straddlers on the left), row A
at the top and row E at the bottom (the frame's flat-dorsal axis points down). Good / mua units on wS1 probes (003 `load`).
  004_probe_locations: each probe's path through the barrel field (its units from superficial to deep), dot at its
      L4 units (else its median position), coloured by cohort and labelled; probes and mice per barrel.
  004_unit_locations: units coloured by layer and by cohort; units and mice per barrel.
  004_response_maps_<set>: barrel-filled maps per layer group (all, L2/3, L4, L5/6): fraction of whisker-responsive units,
      median whisker AUC, median whisker latency (responsive units), fraction of auditory-responsive units. Barrels with
      >= MIN_UNITS units; <set> = all (pooled), R+, R-, visible (brains with visible barrels, Lana's check).
  004_response_units_<set>: the same measures per unit (scatter), per layer group.
Outputs: combined_results_ks4/ssl-barrel-localization/figures/topdown/ (png, pdf, svg) and tables/004_barrel_maps_<set>.csv.
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 004_topdown_maps.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

_spec = importlib.util.spec_from_file_location("maps003", Path(__file__).with_name("003_maps.py"))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

OUT = m3.FIGS / "topdown"
MIN_UNITS = 10
WHISKER, AUDITORY = "#f7b519", "#2c2cdb"
COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}
LAYER_GROUPS = {"All layers": None, "L2/3": ["2/3"], "L4": ["4"], "L5/6": ["5", "6a", "6b"]}
NO_DATA = "#f2f2f2"
plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                     "ytick.labelsize": 5.5, "legend.fontsize": 5})

LAB = np.where(m3.TPL["alab"] < 34, m3.TPL["alab"], 0).T  # rows = dorsal, columns = anterior
LO, PIX = m3.TPL["lo"], float(m3.TPL["pix"])
EXT = [LO[0], LO[0] + LAB.shape[1] * PIX, LO[1], LO[1] + LAB.shape[0] * PIX]  # anterior (x), dorsal (y)
IDX = {v: k for k, v in m3.NAMES.items()}
_yy, _xx = np.nonzero(LAB > 0)
VIEW = [EXT[0] + _xx.min() * PIX - 120, EXT[0] + _xx.max() * PIX + 120,
        EXT[2] + _yy.min() * PIX - 120, EXT[2] + _yy.max() * PIX + 120]
CENTRE = {m3.NAMES[int(k)]: (EXT[0] + np.nonzero(LAB == k)[1].mean() * PIX, EXT[2] + np.nonzero(LAB == k)[0].mean() * PIX)
          for k in np.unique(LAB[LAB > 0])}


def single_hue(color, reverse=False):
    cols = ["#ffffff", color] if not reverse else [color, "#ffffff"]
    return LinearSegmentedColormap.from_list("h", cols)


def base_map(ax, values=None, cmap=None, norm=None, labels=True, label_color="0.35"):
    """Barrel outlines from above; barrels filled with `values` (dict barrel -> value) through cmap / norm."""
    rgba = np.ones(LAB.shape + (4,))
    if values is not None:
        rgba[LAB > 0] = matplotlib.colors.to_rgba(NO_DATA)
        for b, v in values.items():
            if b in IDX and np.isfinite(v):
                rgba[LAB == IDX[b]] = cmap(norm(v))
    ax.imshow(rgba, extent=EXT, origin="lower", interpolation="nearest", zorder=0)
    ax.contour(find_boundaries(LAB, mode="inner").astype(float), levels=[0.5], colors="0.6", linewidths=0.3,
               extent=EXT, origin="lower", zorder=1)
    if labels:
        for b, (x, y) in CENTRE.items():
            ax.text(x, y, b, fontsize=3.5, ha="center", va="center", color=label_color, zorder=2)
    ax.set_xlim(VIEW[:2])
    ax.set_ylim(VIEW[3], VIEW[2])  # row A on top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def compass(ax):
    """Anterior arrow (bottom left) and 500-um scale bar (bottom right)."""
    x0, y0, L = VIEW[0] + 60, VIEW[3] - 60, 400
    ax.annotate("", (x0 + L, y0), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.3"))
    ax.text(x0 + L / 2, y0 - 40, "anterior", fontsize=4.5, ha="center", va="bottom", color="0.3")
    ax.plot([VIEW[1] - 560, VIEW[1] - 60], [y0] * 2, color="k", lw=0.8)
    ax.text(VIEW[1] - 310, y0 - 40, "500 µm", fontsize=4.5, ha="center", va="bottom")


def xy(d):
    return d.trk_flat_anterior_um.to_numpy(), d.trk_flat_dorsal_um.to_numpy()


def colorbar(fig, ax, cmap, norm, label):
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55, pad=0.01, aspect=15)
    cb.set_label(label, fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400)
    plt.close(fig)
    print("saved", OUT / name, flush=True)


def probes_table(df):
    rows = []
    for (m, s, g), x in df.groupby(["mouse_id", "session_id", "electrode_group"]):
        x = x[np.isfinite(x.trk_flat_anterior_um) & (x.barrel_class != "outside SSp-bfd")].sort_values("depth", ascending=False)
        if not len(x):
            continue
        l4 = x[x.layer == "4"]
        ref = l4 if len(l4) else x
        top, bot = x.iloc[0], x.iloc[-1]
        rows.append(dict(mouse_id=m, session_id=s, electrode_group=g, cohort=x.cohort.iloc[0], n_units=len(x),
                         ant=ref.trk_flat_anterior_um.median(), dor=ref.trk_flat_dorsal_um.median(),
                         ant_top=top.trk_flat_anterior_um, dor_top=top.trk_flat_dorsal_um,
                         ant_bot=bot.trk_flat_anterior_um, dor_bot=bot.trk_flat_dorsal_um,
                         barrel=ref.barrel.replace("", np.nan).mode().iloc[0] if ref.barrel.ne("").any() else ""))
    p = pd.DataFrame(rows)
    p["label"] = p.mouse_id + " " + p.electrode_group.str.replace("_shank0", "").str.replace("imec", "i")
    p["n_sessions"] = p.groupby(["mouse_id", "electrode_group"]).session_id.transform("nunique")
    p.loc[p.n_sessions > 1, "label"] += " " + p.session_id.str[9:13]
    return p


def per_barrel(df):
    d = df[df.barrel != ""]
    d = d.assign(probe=d.session_id + "/" + d.electrode_group)
    return d.groupby("barrel").agg(n_units=("cluster_id", "size"), n_mice=("mouse_id", "nunique"),
                                   n_probes=("probe", "nunique"))


def probe_figure(df, probes):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    base_map(ax)
    for r in probes.itertuples():
        c = COHORT_COLORS.get(r.cohort, "0.4")
        ax.plot([r.ant_top, r.ant_bot], [r.dor_top, r.dor_bot], color=c, lw=0.6, alpha=0.7, zorder=3)
        ax.scatter(r.ant, r.dor, s=9, color=c, edgecolor="white", lw=0.3, zorder=4)
        ax.text(r.ant + 30, r.dor - 20, r.label, fontsize=2.6, color=c, zorder=5, clip_on=True)
    for k, c in COHORT_COLORS.items():
        ax.scatter([], [], s=9, color=c, label=f"{k} ({(probes.cohort == k).sum()} probes, "
                                               f"{probes[probes.cohort == k].mouse_id.nunique()} mice)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, handletextpad=0.2)
    compass(ax)
    ax.set_title("Probe paths in SSp-bfd\n(line: superficial to deep units; dot: L4 units)")
    ax = axes[1]
    pb = per_barrel(df)
    norm = Normalize(0, max(pb.n_probes.max(), 1))
    cmap = single_hue("0.25")
    base_map(ax, pb.n_probes.to_dict(), cmap, norm, labels=False)
    for b, (x, y) in CENTRE.items():
        n = pb.n_probes.get(b, 0)
        ax.text(x, y + 40, b, fontsize=3.5, ha="center", va="bottom", color="white" if norm(n) > 0.6 else "0.3")
        if n:
            ax.text(x, y - 30, f"{n} | {pb.n_mice[b]}", fontsize=3.3, ha="center", va="top",
                    color="white" if norm(n) > 0.6 else "0.15")
    colorbar(fig, ax, cmap, norm, "Probes with units in the barrel")
    ax.set_title("Probes | mice per barrel")
    save(fig, "004_probe_locations")


def unit_figure(df):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    d = df[np.isfinite(df.trk_flat_anterior_um)]
    ax = axes[0, 0]
    base_map(ax)
    for l in m3.LAYERS:
        x = d[d.layer == l]
        ax.scatter(*xy(x), s=1.2, lw=0, color=m3.LAYER_COLORS[l], label=f"L{l} ({len(x)})", zorder=3, alpha=0.8,
                   rasterized=True)
    ax.legend(frameon=False, loc="upper right", markerscale=3, handletextpad=0.1)
    compass(ax)
    ax.set_title(f"Units by layer ({len(d)} units, {d.mouse_id.nunique()} mice)")
    ax = axes[0, 1]
    base_map(ax)
    for k, c in COHORT_COLORS.items():
        x = d[d.cohort == k]
        ax.scatter(*xy(x), s=1.2, lw=0, color=c, label=f"{k} ({len(x)} units, {x.mouse_id.nunique()} mice)", zorder=3,
                   alpha=0.6, rasterized=True)
    ax.legend(frameon=False, loc="upper right", markerscale=3, handletextpad=0.1)
    ax.set_title("Units by cohort")
    pb = per_barrel(df)
    for ax, col, title, label in [(axes[1, 0], "n_units", "Units per barrel (all layers)", "Units"),
                                  (axes[1, 1], "n_mice", "Mice per barrel", "Mice")]:
        norm = Normalize(0, pb[col].max())
        cmap = single_hue("0.25")
        base_map(ax, pb[col].to_dict(), cmap, norm, labels=False)
        for b, (x, y) in CENTRE.items():
            n = pb[col].get(b, 0)
            ax.text(x, y + 40, b, fontsize=3.5, ha="center", va="bottom", color="white" if norm(n) > 0.6 else "0.3")
            if n:
                ax.text(x, y - 30, f"{n}", fontsize=3.5, ha="center", va="top", color="white" if norm(n) > 0.6 else "0.15")
        colorbar(fig, ax, cmap, norm, label)
        ax.set_title(title)
    save(fig, "004_unit_locations")


MEASURES = [  # column, aggregation, title, cmap, norm
    ("whisker_responsive", "frac", "Whisker-responsive units", single_hue(WHISKER), Normalize(0, 0.8)),
    ("whisker_auc", "median", "Whisker AUC (median)", plt.get_cmap("RdBu_r"), TwoSlopeNorm(0.5, 0.4, 0.7)),
    ("whisker_latency_ms", "median", "Whisker latency (median, responsive)", single_hue("#7a5200", True),
     Normalize(5, 25)),
    ("auditory_responsive", "frac", "Auditory-responsive units", single_hue(AUDITORY), Normalize(0, 0.5)),
]
UNIT_MEASURES = [
    ("whisker_auc", "Whisker AUC", plt.get_cmap("RdBu_r"), TwoSlopeNorm(0.5, 0.2, 0.8)),
    ("whisker_latency_ms", "Whisker latency (responsive units)", single_hue("#7a5200", True), Normalize(5, 30)),
    ("auditory_auc", "Auditory AUC", plt.get_cmap("RdBu_r"), TwoSlopeNorm(0.5, 0.2, 0.8)),
]
CB_LABELS = {"whisker_responsive": "Fraction of units", "whisker_auc": "AUC, evoked vs baseline (0.5 = none)",
             "whisker_latency_ms": "Latency (ms)", "auditory_responsive": "Fraction of units",
             "auditory_auc": "AUC, evoked vs baseline (0.5 = none)"}


def map_grid(nrows, ncols, title_h=0.5, cb_h=0.55):
    """Equal-size barrel maps (sized from the map aspect) with a horizontal colorbar under each column."""
    W, left, gap = 7.2, 0.3, 0.08
    w = (W - left - 0.1 - (ncols - 1) * gap) / ncols
    h, vgap = w * (VIEW[3] - VIEW[2]) / (VIEW[1] - VIEW[0]), 0.05
    H = title_h + nrows * h + (nrows - 1) * vgap + cb_h
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_axes([(left + j * (w + gap)) / W, 1 - (title_h + (i + 1) * h + i * vgap) / H,
                                       w / W, h / H])
    caxes = [fig.add_axes([(left + j * (w + gap) + 0.2 * w) / W, 0.3 / H, 0.6 * w / W, 0.06 / H]) for j in range(ncols)]
    return fig, axes, caxes


def hcolorbar(fig, cax, cmap, norm, label):
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2, pad=1)
    cb.outline.set_linewidth(0.4)


def row_label(ax, text):
    ax.text(-0.02, 0.5, text, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6)


def layer_subset(df, layers):
    return df if layers is None else df[df.layer.isin(layers)]


def barrel_values(d, col, how):
    d = d[d.barrel != ""]
    if col == "whisker_latency_ms":
        d = d[d.whisker_responsive.fillna(False).astype(bool)]
        ok = d.groupby("barrel").size() >= 3
    else:
        ok = d.groupby("barrel")[col].count() >= MIN_UNITS
    g = d.groupby("barrel")[col]
    v = g.apply(lambda s: s.astype(float).mean()) if how == "frac" else g.median()
    return v[ok.reindex(v.index, fill_value=False)]


def response_maps(df, tag, title):
    fig, axes, caxes = map_grid(len(LAYER_GROUPS), len(MEASURES))
    rows = []
    for i, (lg, layers) in enumerate(LAYER_GROUPS.items()):
        d = layer_subset(df, layers)
        for j, (col, how, mtitle, cmap, norm) in enumerate(MEASURES):
            v = barrel_values(d, col, how)
            ax = axes[i, j]
            base_map(ax, v.to_dict(), cmap, norm)
            if i == 0:
                ax.set_title(mtitle, fontsize=6, pad=2)
                hcolorbar(fig, caxes[j], cmap, norm, CB_LABELS[col])
            if j == 0:
                row_label(ax, f"{lg} ({len(d[d.barrel != ''])} units)")
            if i == len(LAYER_GROUPS) - 1 and j == 0:
                compass(ax)
            rows.append(v.rename("value").reset_index().assign(layer_group=lg, measure=col))
    fig.suptitle(f"{title} ({df.mouse_id.nunique()} mice): barrels with >= {MIN_UNITS} units (latency: >= 3 responsive "
                 f"units); light grey = fewer units", fontsize=6.5, y=0.99)
    save(fig, f"004_response_maps_{tag}")
    tab = pd.concat(rows, ignore_index=True)
    n = df[df.barrel != ""].groupby("barrel").agg(n_units=("cluster_id", "size"), n_mice=("mouse_id", "nunique"))
    tab.merge(n, left_on="barrel", right_index=True).to_csv(m3.TABLES / f"004_barrel_maps_{tag}.csv", index=False)


def response_units(df, tag, title):
    fig, axes, caxes = map_grid(len(LAYER_GROUPS) - 1, len(UNIT_MEASURES))
    for i, (lg, layers) in enumerate(list(LAYER_GROUPS.items())[1:]):
        d = layer_subset(df, layers)
        d = d[np.isfinite(d.trk_flat_anterior_um)]
        for j, (col, mtitle, cmap, norm) in enumerate(UNIT_MEASURES):
            x = d[d.whisker_responsive.fillna(False).astype(bool)] if col == "whisker_latency_ms" else d
            x = x[x[col].notna()]
            x = x.reindex(x[col].sub(0.5).abs().sort_values().index) if "auc" in col else x.sort_values(col, ascending=False)
            ax = axes[i, j]
            base_map(ax, labels=True, label_color="0.6")
            ax.scatter(*xy(x), c=x[col], cmap=cmap, norm=norm, s=2.5, lw=0, zorder=3, rasterized=True)
            if i == 0:
                ax.set_title(mtitle, fontsize=6, pad=2)
                hcolorbar(fig, caxes[j], cmap, norm, CB_LABELS[col])
            if j == 0:
                row_label(ax, f"{lg} ({len(d)} units)")
            if i == len(LAYER_GROUPS) - 2 and j == 0:
                compass(ax)
    fig.suptitle(f"{title} ({df.mouse_id.nunique()} mice): every unit at its barrel-field position, strongest responses "
                 f"on top", fontsize=6.5, y=0.99)
    save(fig, f"004_response_units_{tag}")


def main():
    mice = sorted(p.stem.split("_")[-1] for p in m3.TABLES.glob("002_responses_*.parquet"))
    df = pd.concat([m3.load(m) for m in mice], ignore_index=True)
    df["cohort"] = df.mouse_id.map(m3.cohorts())
    inv = pd.read_csv(m3.TABLES / "000_inventory.csv").drop_duplicates("mouse_id").set_index("mouse_id")
    df["barrels_visible"] = df.mouse_id.map(inv.barrels_visible).fillna("").eq("yes")
    probes = probes_table(df)
    probes.to_csv(m3.TABLES / "004_probe_positions.csv", index=False)
    probe_figure(df, probes)
    unit_figure(df)
    sets = [("all", df, "All brains"), ("R+", df[df.cohort == "R+"], "R+ mice"), ("R-", df[df.cohort == "R-"], "R- mice"),
            ("visible", df[df.barrels_visible], "Brains with visible barrels")]
    for tag, d, title in sets:
        response_maps(d, tag, title)
        response_units(d, tag, title)


if __name__ == "__main__":
    main()
