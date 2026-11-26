"""SuppFigure3_annotated_matrix plot functions (read only the panel CSVs)."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _box(ax):
    fig = ax.figure
    fw, fh = fig.get_size_inches() * 25.4
    p = ax.get_position()
    ax.axis("off")
    return fig, fw, fh, p.x0 * fw, (1 - p.y1) * fh, p.width * fw, p.height * fh


def plot_metacluster_unit_properties(ax, df):
    """Panel b: metaclusters (rows, rastermap order, strip colours as the matrix) x unit features (columns); colour =
    median of the feature in the metacluster, z-scored across metaclusters; the fraction of good units printed."""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    feats = list(dict.fromkeys(df.feature))
    mcs = df.drop_duplicates("metacluster").sort_values("mc_rank")
    Z = df.pivot(index="metacluster", columns="feature", values="z").loc[mcs.metacluster, feats].to_numpy()
    lab_w, top = 9.0, 14.0
    a = fig.add_axes([(X0 + lab_w) / fw, 1 - (Y0 + H) / fh, (W - lab_w - 14.0) / fw, (H - top) / fh])
    im = a.imshow(Z, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto", interpolation="none")
    a.set_yticks(range(len(mcs))); a.set_yticklabels([f"MC {m}" for m in mcs.metacluster], fontsize=5.0)
    for t, c in zip(a.get_yticklabels(), mcs.mc_color):
        t.set_color(c); t.set_fontweight("bold")
    labs = df.drop_duplicates("feature").set_index("feature").label
    a.set_xticks(range(len(feats))); a.set_xticklabels([labs[f] for f in feats], rotation=45, ha="left", fontsize=5.0,
                                                       rotation_mode="anchor")
    a.xaxis.tick_top()
    a.tick_params(length=0, pad=1)
    for s in a.spines.values():
        s.set_visible(False)
    good = df[df.feature == "frac_good"].set_index("metacluster").value
    n = df.drop_duplicates("metacluster").set_index("metacluster").n
    for i, m in enumerate(mcs.metacluster):
        a.text(len(feats) - 0.3, i, f"{good[m]:.0%} good · n = {n[m]:,}", fontsize=5.0, va="center", ha="left")
    cax = fig.add_axes([(X0 + lab_w) / fw, 1 - (Y0 + H + 3.0) / fh, 22.0 / fw, 1.2 / fh])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=5.0, length=1.2, pad=0.8, width=0.35)
    cb.outline.set_linewidth(0.35)
    cb.set_label("median, z-scored across metaclusters", fontsize=5.0, labelpad=1)
