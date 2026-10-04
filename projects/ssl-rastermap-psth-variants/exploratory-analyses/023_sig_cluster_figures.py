"""Figures for the post-hoc significant (R+ vs R-, Mann-Whitney BH-FDR) rastermap clusters of one variant.

1  sig_cluster_psth_summary : publication summary in the style of the metacluster PSTH summary. One row per significant
   cluster (sorted by R+ share): label (cluster number, activity-based functional name, R+ share, q, n neurons / mice),
   mean +- SEM PSTH for ALL PSTH types (two-level Passive | Active headers), area-group composition bar (top-3 areas),
   sagittal + top-down atlas views (neurons, density bands, centroid).
2  cluster_<k>_slide        : one figure per significant cluster for single slides. Row 1: all PSTH types.
   Row 2: area-group composition (no top/right frame), top areas, large sagittal and top-down atlas views.
Cluster numbers are 1-based, as in stats/per_cluster_stats.csv and the matrix enrichment column.
CCF atlas coordinates are read once from the cached unit table and stored in <out>/neuron_ccf_coords.parquet.
Outputs -> <method_dir>/sig_cluster_figures/
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import rastermap_psth.population_matrix_summary as pms            # noqa: E402
import rastermap_psth.rastermap_utils as rmu                       # noqa: E402
import ephys_utilities.allen_utils.allen_utils as allen_utils     # noqa: E402

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CONFIG = "/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml"
ATLAS_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0")
CACHE = VARIANTS_ROOT / "_cache" / "tables_day0.pkl"
RP, RM = "#00B400", "#C800C8"
FS = 7


def ccf_coords(md, unit_ids, out):
    f = out / "neuron_ccf_coords.parquet"
    if f.exists():
        d = pd.read_parquet(f).set_index("unit_id").reindex(unit_ids)
        return d[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].to_numpy(float)
    import pickle
    print("[023] loading cached unit table for CCF coordinates (once) ...", flush=True)
    ut = pickle.load(open(CACHE, "rb"))["unit_table"]
    coords = rmu._load_ccf_atlas_coords(ut, unit_ids)
    pd.DataFrame(dict(unit_id=unit_ids, ccf_atlas_ap=coords[:, 0], ccf_atlas_ml=coords[:, 1],
                      ccf_atlas_dv=coords[:, 2])).to_parquet(f, index=False)
    return coords


class Atlas:
    def __init__(self, coords):
        self.ref = rmu.load_reference(ATLAS_PATH)
        self.mask = rmu.load_annotation_mask(ATLAS_PATH)
        self.px = rmu._reorder_coords_for_brainrender(coords) / rmu.VOXEL_SIZE_UM
        self.sag = rmu._translucent_gray_rgba(self.ref.max(axis=2).T)
        self.top = rmu._translucent_gray_rgba(self.ref.max(axis=1).T)

    def draw(self, ax_sag, ax_top, sel, color, dot=6, bands=True):
        pts = self.px[sel]
        pts = pts[~np.isnan(pts).any(axis=1)]
        pts = pts[rmu._filter_points_inside_brain(pts, self.mask)]
        ap, dv, ml = pts[:, 0], pts[:, 1], pts[:, 2]
        for ax, img, y in ((ax_sag, self.sag, dv), (ax_top, self.top, ml)):
            ax.imshow(img, origin="upper", interpolation="nearest", zorder=1)
            ax.scatter(ap, y, s=dot, color=color, alpha=0.5, edgecolors="none", zorder=4, rasterized=True)
            if bands:
                rmu._kde_density_bands_on_ax(ax, ap, y, color)
            ax.scatter([ap.mean()], [y.mean()], marker="+", s=40 if dot < 1 else 90, linewidths=1.0 if dot < 1 else 1.6,
                       color="k", zorder=5)
            ax.axis("off")


def area_legend(fig, groups, y, fs, only=None, ncol=6):
    """one-row-wrapped area-group colour legend (canonical order), centred at figure height y"""
    from matplotlib.patches import Patch
    colors = allen_utils.get_custom_area_groups_colors()
    present = set(groups) if only is None else set(only)
    names = [g for g in allen_utils.get_area_group_custom_order() if g in present] + \
            sorted(g for g in present if g not in allen_utils.get_area_group_custom_order())
    fig.legend(handles=[Patch(facecolor=colors.get(g, "0.7"), label=g) for g in names], loc="upper center",
               bbox_to_anchor=(0.5, y), ncol=min(ncol, len(names)), fontsize=fs, frameon=False, handlelength=0.9,
               handleheight=0.9, columnspacing=1.0, title="Area group", title_fontsize=fs)


def area_bar(ax, groups, area, sel, fs, label_top=True, pct_labels=True):
    order = [g for g in allen_utils.get_area_group_custom_order()]
    colors = allen_utils.get_custom_area_groups_colors()
    cnt = pd.Series(groups[sel]).value_counts()
    cnt = cnt.reindex([g for g in order if g in cnt.index] + [g for g in cnt.index if g not in order])
    frac = cnt.values / sel.sum()
    bars = ax.bar(range(len(cnt)), frac * 100, color=[colors.get(g, "0.7") for g in cnt.index], width=0.8)
    for b, f in zip(bars, frac):
        if pct_labels and f >= 0.05:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{f * 100:.0f}", ha="center", va="bottom",
                    fontsize=fs - 2)
    ax.set_xticks([]); ax.set_ylabel("% of cluster", fontsize=fs - 1)
    ax.set_ylim(0, max(frac * 100) * 1.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    top3 = pd.Series(area[sel]).value_counts().head(3)
    if label_top:
        ax.set_title("Top areas: " + ", ".join(f"{a} ({n})" for a, n in top3.items()), fontsize=fs - 1, loc="left")
    return cnt


def cluster_names(R):
    """activity-based name per cluster (same rules as the metacluster names, one 'family' per cluster)"""
    K = int(R["labels"].max()) + 1
    names, tab = pms.family_names_psth(R, {k: k for k in range(K)})
    return names, tab


def psth_panels(axes, R, sel, labels_c, fs, show_titles=True, ylabel=True, xlabels=True):
    X, off = R["X"], np.r_[0, np.cumsum(R["n_bins_list"])]
    segs = [X[sel, off[c]:off[c + 1]] for c in range(len(labels_c))]
    mu = [s.mean(0) for s in segs]; se = [s.std(0) / np.sqrt(len(s)) for s in segs]
    lo = min((m - e).min() for m, e in zip(mu, se)); hi = max((m + e).max() for m, e in zip(mu, se))
    pad = 0.08 * (hi - lo + 1e-9)
    for c, ax in enumerate(axes):
        t = R["t_ctrs"][c]
        ax.fill_between(t, mu[c] - se[c], mu[c] + se[c], color="0.75", lw=0)
        ax.plot(t, mu[c], color="k", lw=0.9)
        ax.axvline(0, color="0.5", lw=0.4, ls=":"); ax.axhline(0, color="0.85", lw=0.4)
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(t[0], t[-1])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=fs - 2, length=1.5, pad=1)
        if c > 0:
            ax.set_yticklabels([])
        elif ylabel:
            ax.set_ylabel("z-score", fontsize=fs - 1)
        if show_titles:
            ax.set_title(pms.cond_ticklabels([labels_c[c]])[0], fontsize=fs - 1.5, pad=2, linespacing=0.95)
        if not xlabels:
            ax.set_xticklabels([])
        else:                                          # 3 ticks (start, 0, end), edge labels kept inside the panel
            edge = t[0] if abs(t[0]) > abs(t[-1]) else t[-1]     # the window edge farther from 0
            ticks = sorted([edge, 0.0])
            ax.set_xticks(ticks)
            tl = ax.set_xticklabels([f"{v:.1f}" if v else "0" for v in ticks])
            tl[0].set_ha("left" if ticks[0] == edge else "center"); tl[-1].set_ha("right" if ticks[-1] == edge else "center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    md = sorted((VARIANTS_ROOT / a.variant).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]
    out = md / "sig_cluster_figures"; out.mkdir(exist_ok=True)
    qc = ["good", "mua"] if a.variant.startswith("qc_good_mua") else ["good"]
    cfg = pms.load_cfg(CONFIG, unit_quality_label=qc)
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = \
        rmu.get_conditions(cfg)
    labels_c = list(rmu.COND_LABELS)
    R = pms.load_rastermap(md); fmk = pms.load_fmk_stats(md)
    st = pd.read_csv(md / "stats" / "per_cluster_stats.csv")          # cluster is 1-based
    F, g = fmk["F"], fmk["groups"]
    mrp, mrm = F[g == "R+"].mean(0), F[g == "R-"].mean(0)
    share = mrp / (mrp + mrm)
    sig = np.flatnonzero(fmk["reject"])
    sig = sig[np.argsort(-share[sig])]                                 # R+ enriched first
    names, name_tab = cluster_names(R)
    name_tab.to_csv(out / "cluster_function_names.csv", index=False)
    coords = ccf_coords(md, R["unit_ids"], out)
    atlas = Atlas(coords)
    groups, area, mice = R["area_group_arr"].astype(str), R["area_arr"].astype(str), R["mouse_arr"]
    title = a.title or a.variant
    pms.use_pub_font()
    plt.rcParams.update({"font.size": FS, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    nC = len(labels_c)

    def label_txt(k):
        q = st.loc[st.cluster == k + 1, "p_fdr"]
        qtxt = f"q = {q.iloc[0]:.2g}" if len(q) else ""
        sel = R["labels"] == k
        return (f"Cluster {k + 1}", names.get(k, ""),
                f"R+ share {share[k]:.0%}, {qtxt}\n{int(sel.sum()):,} neurons · {len(np.unique(mice[sel]))} mice")

    # ── 1. summary ────────────────────────────────────────────────────────────
    row_mm = 17.0                                                      # airier rows
    H = 36 + row_mm * len(sig) + 22                                    # + area-group legend
    fig = plt.figure(figsize=(183 / 25.4, H / 25.4))
    # columns: label | spacer (room for the y label) | PSTHs | spacer | area bar | sagittal | top-down
    gs = fig.add_gridspec(len(sig), 1 + 1 + nC + 1 + 3,
                          width_ratios=[2.6, 0.75] + [1.0] * nC + [0.45, 1.4, 1.25, 1.25],
                          left=0.005, right=0.995, top=1 - 26 / H, bottom=32 / H, wspace=0.28, hspace=0.6)
    P0 = 2                                                             # first PSTH column
    top_axes = None
    for r, k in enumerate(sig):
        sel = R["labels"] == k
        col = RP if share[k] > 0.5 else RM
        axl = fig.add_subplot(gs[r, 0]); axl.axis("off")
        axl.add_patch(plt.Rectangle((0, 0.1), 0.04, 0.8, color=col, transform=axl.transAxes, clip_on=False))
        l1, l2, l3 = label_txt(k)
        axl.text(0.07, 0.86, l1, transform=axl.transAxes, va="center", fontsize=FS)
        axl.text(0.07, 0.6, l2, transform=axl.transAxes, va="center", fontsize=FS - 1.2)
        axl.text(0.07, 0.22, l3, transform=axl.transAxes, va="center", fontsize=FS - 1.5, color="0.35", linespacing=1.1)
        axs = [fig.add_subplot(gs[r, P0 + c]) for c in range(nC)]
        psth_panels(axs, R, sel, labels_c, FS, show_titles=(r == 0), ylabel=True, xlabels=(r == len(sig) - 1))
        if r == 0:
            top_axes = axs
        axb = fig.add_subplot(gs[r, P0 + nC + 1])
        area_bar(axb, groups, area, sel, FS, label_top=False, pct_labels=False)
        axb.set_ylabel(""); axb.tick_params(labelsize=FS - 2)
        atlas.draw(fig.add_subplot(gs[r, P0 + nC + 2]), fig.add_subplot(gs[r, P0 + nC + 3]), sel, col, dot=0.25,
                   bands=False)
    area_legend(fig, groups, y=24 / H, fs=FS - 1)
    fig.canvas.draw()
    pms.draw_cond_headers(fig, top_axes, labels_c, y_top=1 - 9 / H, y_sub=1 - 14.5 / H, fs=FS)
    fig.text(top_axes[-1].get_position().x1 + 0.03, 1 - 22 / H, "Area groups (% of cluster)", fontsize=FS - 1)
    fig.text(0.005, 1 - 3 / H, f"{title}: post-hoc significant rastermap clusters (R+ vs R−, Mann-Whitney BH-FDR)",
             fontsize=FS + 1, va="top")
    fig.text(0.005, 3 / H, "PSTH: mean ± SEM over the cluster's neurons (z-score, even trials); time (s) from alignment "
             "event. Bars: area-group composition. Atlas: sagittal and top-down, + = centroid.",
             fontsize=FS - 1.5, color="0.35")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"sig_cluster_psth_summary.{ext}", dpi=400)
    plt.close(fig)

    # ── 2. one slide figure per cluster ───────────────────────────────────────
    for k in sig:
        sel = R["labels"] == k
        col = RP if share[k] > 0.5 else RM
        fig = plt.figure(figsize=(254 / 25.4, 143 / 25.4))                 # 16:9 slide
        g1 = fig.add_gridspec(1, nC, left=0.06, right=0.99, top=0.8, bottom=0.55, wspace=0.18)
        g2 = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.35], left=0.06, right=0.99, top=0.46, bottom=0.16,
                              wspace=0.12)
        axs = [fig.add_subplot(g1[0, c]) for c in range(nC)]
        psth_panels(axs, R, sel, labels_c, FS + 1)
        fig.canvas.draw()
        pms.draw_cond_headers(fig, axs, labels_c, y_top=0.93, y_sub=0.875, fs=FS + 1)
        axb = fig.add_subplot(g2[0, 0])
        area_bar(axb, groups, area, sel, FS + 1, label_top=False)
        atlas.draw(fig.add_subplot(g2[0, 1]), fig.add_subplot(g2[0, 2]), sel, col, dot=1.2)
        area_legend(fig, groups, y=0.115, fs=FS, only=np.unique(groups[sel]), ncol=8)
        l1, l2, l3 = label_txt(k)
        fig.text(0.01, 0.985, f"{l1} — {l2}", fontsize=FS + 3, va="top")
        fig.text(0.99, 0.985, l3.replace("\n", "  ·  "), fontsize=FS, va="top", ha="right", color="0.35")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"cluster_{k + 1:03d}_slide.{ext}", dpi=300)
        plt.close(fig)
    print(f"[023] {len(sig)} significant clusters: {[int(k) + 1 for k in sig]}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
