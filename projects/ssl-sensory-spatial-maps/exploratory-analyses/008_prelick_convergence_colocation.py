"""008 -- Pre-lick converging neurons and the convergence of whisker- and auditory-cortex projections.

Hypothesis (user 2026-10-04): neurons whose pre-lick activity converges across modalities sit where the projections of
whisker and auditory cortex overlap.
Neurons: pre-lick ROC with the spontaneous-lick reference (ssl-prelick-convergence 051, PRELICK_REF=sl, locked set:
100 ms before the corrected first lick, active trials, perf != 6, warm-up cut, A1 trim; good + mua; tested = both ROCs
tested, mean raw pre-lick rate >= 0.1 Hz). Converging neuron (locked definition, 065): significant whisker-hit vs SL AND
auditory-hit vs SL, same sign. Quantity: P = fraction of tested neurons that converge.
Anatomy (002, line-balanced zones): whisker zone, auditory zone and their overlap (ZONE_PCT contour), 50-um grid, right
hemisphere (neurons folded onto it).
Tests (as 006; no session pairing): P inside the overlap vs P among all tested neurons -- hierarchical bootstrap of the
difference (sessions with replacement, then neurons, binomial; one-sided) and Fisher's exact test; pooled (all sessions)
and per cohort x stage (Holm over the 4 groups). Exploratory: P in each zone category (outside both, whisker zone only,
auditory zone only, overlap) and overlap vs each single-modality zone (same bootstrap).
Maps: converging fraction density = Gaussian-smoothed converging count / smoothed tested count (3-D, sigma 150 um,
>= 3 neurons in the kernel), coronal 500-um slabs through the overlap sub-regions (as 006) and the isocortex flatmap
(as 007).
Outputs: combined_results_ks4/_sensory_spatial_maps/prelick_convergence/ (tests, zone categories, unit table with
mouse_id, session_id, electrode_group, cluster_id, provenance, caption) and figures<ZTAG>/prelick_convergence_{colocation,
groups,flatmaps}.{png,pdf,svg}.
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import ndimage, stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "py_extra"))          # ccf_streamlines
m3 = importlib.import_module("003_spatial_maps")
m2 = importlib.import_module("002_projection_zones")
m6 = importlib.import_module("006_colocation_figure")
m7 = importlib.import_module("007_cortical_flatmaps")
S, FIG, ZTAG, ZONE_PCT = m3.S, m3.FIG, m3.ZTAG, m3.ZONE_PCT
PRE = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/_roc_prelick_sl")
POUT = m3.OUT / "prelick_convergence"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
AF, WF = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"   # "fa" class = spontaneous licks here
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
GLAB = {("R+", "learning"): "R+ learning", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− learning",
        ("R-", "expert"): "R− expert"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
CONV, PURPLE, WCOL, ACOL = "#d94801", m6.PURPLE, m6.WCOL, m6.ACOL
ZCAT = ["outside both zones", "whisker zone only", "auditory zone only", "overlap"]
ZCOL = ["0.6", WCOL, ACOL, PURPLE]


# ------------------------------------------------------------------------------------------------ data
def load():
    W = pd.read_parquet(PRE / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    tested = W[f"sig:{AF}"].notna() & W[f"sig:{WF}"].notna()
    W = W[tested & W[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].notna().all(1)].reset_index(drop=True)
    sa, sw = np.sign(W[f"sel:{AF}"]), np.sign(W[f"sel:{WF}"])
    W["converging"] = (W[f"sig:{AF}"] == 1) & (W[f"sig:{WF}"] == 1) & (sa == sw)
    W["converging_sign"] = np.where(W.converging, sa, 0)
    W["bimodal"] = W.converging              # 006 statistics read the binary label from this column
    W["ml_f"] = m3.MID + np.abs(W.ccf_atlas_ml - m3.MID)
    W["lat_mm"] = (W.ml_f - m3.MID) / 1000
    W["ap_mm"] = W.ccf_atlas_ap / 1000
    W["dv_mm"] = W.ccf_atlas_dv / 1000
    return W


def group_sel(D):
    out = [("All sessions", np.ones(len(D), bool))]
    for g in GROUPS:
        out.append((GLAB[g], ((D.cohort == g[0]) & (D.stage == g[1])).to_numpy()))
    return out


# ------------------------------------------------------------------------------------------------ statistics
def tests(D, Z, rng):
    ov = Z["zone70_whisker"] & Z["zone70_auditory"]
    ijk, ok = m6.unit_voxels(D, ov.shape)
    D["in_whisker_zone"] = m6.in_mask(Z["zone70_whisker"], ijk, ok)
    D["in_auditory_zone"] = m6.in_mask(Z["zone70_auditory"], ijk, ok)
    D["in_overlap"] = D.in_whisker_zone & D.in_auditory_zone
    D["zone_cat"] = D.in_whisker_zone.astype(int) + 2 * D.in_auditory_zone.astype(int)
    rows, cats = [], []
    for name, sel in group_sel(D):
        Dg = D[sel].reset_index(drop=True)
        r, _ = m6.test_region(Dg, ijk[sel], ok[sel], ov, np.ones(len(Dg), bool), rng, "whole overlap zone")
        r.update(group=name, n_sessions=int(Dg.session_id.nunique()), n_mice=int(Dg.mouse_id.nunique()),
                 n_tested=len(Dg), P_all=float(Dg.converging.mean()))
        rows.append(r)
        for k, cn in enumerate(ZCAT):
            s = (Dg.zone_cat == k).to_numpy()
            lo, hi = m6.boot_ci(Dg, s, rng)
            row = dict(group=name, zone=cn, n_tested=int(s.sum()), n_converging=int(Dg.converging[s].sum()),
                       n_sessions=int(Dg.session_id[s].nunique()), P=float(Dg.converging[s].mean()) if s.any() else np.nan,
                       ci_lo=lo, ci_hi=hi)
            if k in (1, 2) and s.sum() >= m6.MIN_UNITS and (Dg.zone_cat == 3).sum() >= m6.MIN_UNITS:
                d = m6.boot_diff(Dg, (Dg.zone_cat == 3).to_numpy(), s, rng)       # overlap vs this single zone
                row.update(diff_overlap_minus=float(Dg.converging[Dg.zone_cat == 3].mean() - Dg.converging[s].mean()),
                           p_boot_overlap_gt=(1 + np.sum(d <= 0)) / (1 + len(d)))
            cats.append(row)
    T, C = pd.DataFrame(rows), pd.DataFrame(cats)
    g4 = T.group != "All sessions"
    T.loc[g4 & T.p_boot.notna(), "p_boot_holm"] = m6.holm(T.loc[g4 & T.p_boot.notna(), "p_boot"])
    return T, C


# ------------------------------------------------------------------------------------------------ maps
def slab_density(A, D, key, c):
    if key not in m3._DCACHE:
        m3._DCACHE[key] = m3.density_volumes(D, D.converging.to_numpy(float))
    M, extd = m3.slab_ratio(*m3._DCACHE[key], "cor", c)
    sec = A.section("cor", c)
    _, inside = A.boundaries(sec)
    gy = ((np.arange(M.shape[0]) + 0.5) * 5).astype(int)
    gx = ((np.arange(M.shape[1]) + 0.5) * 5).astype(int)
    inb = inside[np.clip(gy, 0, inside.shape[0] - 1)][:, np.clip(gx, 0, inside.shape[1] - 1)]
    return np.where(inb, M, np.nan), extd


def cmap_conv():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("white_conv", ["#ffffff", "#fdbe85", "#e6550d", "#7f2704"])


def overlap_contour(ax, A, c):
    zo = A.zone_section("whisker", "cor", c) & A.zone_section("auditory", "cor", c)
    r = A.zres / 1000
    yy, xx = (np.arange(zo.shape[0]) + 0.5) * r, (np.arange(zo.shape[1]) + 0.5) * r
    if zo.any():
        ax.contour(xx, yy, ndimage.gaussian_filter(zo.astype(float), 0.8), levels=[0.5], colors=[PURPLE], linewidths=0.7, zorder=6)


def finish_ax(ax):
    ax.set_xlim(0, 5.8); ax.set_ylim(7.2, 0); ax.set_aspect("equal"); ax.set_axis_off()


def p_label(p):
    return "n/a" if not np.isfinite(p) else S.fmt_p(p)


# ------------------------------------------------------------------------------------------------ figure 1
def fig_colocation(plt, A, D, T, C, slabs, vmax):
    from matplotlib.colors import ListedColormap
    W = S.W_IN
    ncol = len(slabs)
    pw = (W - 0.45) / ncol
    ph = pw * 7.2 / 5.8
    H = 0.55 + 3 * ph + 0.45 + 2.1
    fig = plt.figure(figsize=(W, H))
    top_frac = (3 * ph) / H
    gs = fig.add_gridspec(3, ncol, left=0.4 / W, right=1 - 0.05 / W, top=1 - 0.52 / H, bottom=1 - 0.52 / H - top_frac,
                          wspace=0.03, hspace=0.05)
    gb = fig.add_gridspec(1, 3, width_ratios=[0.55, 1.5, 1.15], wspace=0.55, left=0.55 / W, right=1 - 0.1 / W,
                          top=1.85 / H, bottom=0.42 / H)
    cm = cmap_conv()
    for c_i, c in enumerate(slabs):
        sec = A.section("cor", c)
        zw, za = A.zone_section("whisker", "cor", c), A.zone_section("auditory", "cor", c)
        zo = zw & za
        r = A.zres / 1000
        ext = (0, zw.shape[1] * r, zw.shape[0] * r, 0)
        axs = [fig.add_subplot(gs[k, c_i]) for k in range(3)]
        ax = axs[0]
        for z_, col in ((zw, WCOL), (za, ACOL)):
            ax.imshow(np.ma.masked_where(~z_, z_.astype(float)), extent=ext, cmap=ListedColormap([col]), alpha=0.32,
                      interpolation="nearest", zorder=1)
        ax.imshow(np.ma.masked_where(~zo, zo.astype(float)), extent=ext, cmap=ListedColormap([PURPLE]), alpha=0.75,
                  interpolation="nearest", zorder=1.3)
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        ax = axs[1]
        m, x, y = m3.in_slab(D, "cor", c)
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        cv = D.converging.to_numpy()
        ax.scatter(x[m & ~cv], y[m & ~cv], s=0.35, c="0.62", lw=0, zorder=3, rasterized=True)
        ax.scatter(x[m & cv], y[m & cv], s=0.45, c=CONV, lw=0, zorder=4, rasterized=True)
        ax.text(0.98, 0.02, f"{int((m & cv).sum())}/{int(m.sum())}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.3, color="0.4")
        ax = axs[2]
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        M, extd = slab_density(A, D, "conv_all", c)
        imd = ax.imshow(M, extent=extd, cmap=cm, vmin=0, vmax=vmax, interpolation="bilinear", zorder=1.5)
        for ax in axs[1:]:
            overlap_contour(ax, A, c)
        for ax in axs:
            finish_ax(ax)
        axs[0].set_title(f"AP {(5400 - c) / 1000:+.2f} mm", fontsize=5.8, pad=2)
        if c_i == 0:
            for ax, t in zip(axs, ["Projection zones", "Tested neurons", "Converging fraction"]):
                ax.text(-0.05, 0.5, t, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6)
            axs[2].plot([0.2, 1.2], [7.0, 7.0], color="k", lw=0.9, solid_capstyle="butt")
            axs[2].text(0.7, 6.85, "1 mm", ha="center", va="bottom", fontsize=4.6)
    hz = [plt.Rectangle((0, 0), 1, 1, color=WCOL, alpha=0.4, lw=0, label="whisker-cortex zone"),
          plt.Rectangle((0, 0), 1, 1, color=ACOL, alpha=0.4, lw=0, label="auditory-cortex zone"),
          plt.Rectangle((0, 0), 1, 1, color=PURPLE, alpha=0.75, lw=0, label="overlap (line in b, c)"),
          plt.Line2D([], [], marker="o", ls="", ms=2.5, color="0.75", label="tested"),
          plt.Line2D([], [], marker="o", ls="", ms=2.5, color=CONV, label="converging")]
    fig.legend(handles=hz, loc="upper left", bbox_to_anchor=(0.4 / W, 1 - 0.2 / H), ncol=5, frameon=False, fontsize=5.0,
               handlelength=1.2, columnspacing=1.0)
    c_bottom = 1 - 0.52 / H - top_frac
    cax = fig.add_axes([2.4 / W, c_bottom - 0.16 / H, 1.5 / W, 0.05 / H])
    cb = fig.colorbar(imd, cax=cax, orientation="horizontal")
    cb.set_label("Converging fraction of tested neurons (3-D Gaussian, sigma 150 um)", fontsize=4.8, labelpad=1)
    cb.ax.tick_params(labelsize=4.4, length=1.2, width=0.4); cb.outline.set_linewidth(0.4)
    # d: pooled, inside vs all
    axd = fig.add_subplot(gb[0, 0])
    g = T[T.group == "All sessions"].iloc[0]
    axd.bar([0, 1], [100 * g.P_in, 100 * g.P_ref], 0.62, color=[PURPLE, "0.6"], lw=0,
            yerr=[[100 * (g.P_in - g.ci_lo), 100 * (g.P_ref - g.ref_ci_lo)], [100 * (g.ci_hi - g.P_in), 100 * (g.ref_ci_hi - g.P_ref)]],
            error_kw=dict(lw=0.6))
    axd.set_xticks([0, 1], ["inside\noverlap", "all\ntested"], fontsize=5.2)
    axd.set_ylabel("Converging neurons (% of tested)")
    top = 100 * max(g.ci_hi, g.ref_ci_hi)
    axd.plot([0, 0, 1, 1], [top * 1.06, top * 1.1, top * 1.1, top * 1.06], color="k", lw=0.5)
    axd.text(0.5, top * 1.13, f"bootstrap {p_label(g.p_boot)}\nFisher {p_label(g.p_fisher)}", ha="center", va="bottom", fontsize=4.5)
    axd.set_ylim(0, top * 1.55)
    axd.set_title("All sessions", fontsize=5.4, loc="left")
    # e: per cohort x stage
    axe = fig.add_subplot(gb[0, 1])
    Tg = T[T.group != "All sessions"].reset_index(drop=True)
    tops = []
    for i, r in Tg.iterrows():
        coh = "R+" if r.group.startswith("R+") else "R-"
        fc = COH[coh] if "expert" in r.group else "white"
        x0 = 3 * i
        axe.bar(x0, 100 * r.P_in, 0.9, color=fc, ec=COH[coh], lw=0.8)
        axe.bar(x0 + 1, 100 * r.P_ref, 0.9, color="0.85", ec=COH[coh], lw=0.8)
        for xx, p, lo, hi in ((x0, r.P_in, r.get("ci_lo", np.nan), r.get("ci_hi", np.nan)),
                              (x0 + 1, r.P_ref, r.get("ref_ci_lo", np.nan), r.get("ref_ci_hi", np.nan))):
            if np.isfinite(lo):
                axe.plot([xx, xx], [100 * lo, 100 * hi], color="k", lw=0.6)
        t = 100 * np.nanmax([r.get("ci_hi", np.nan), r.get("ref_ci_hi", np.nan), r.P_in, r.P_ref])
        tops.append(t)
        ph_ = r.get("p_boot_holm", np.nan)
        axe.text(x0 + 0.5, t * 1.08, f"{p_label(ph_)}\nn = {int(r.n_resp_in)}/{int(r.n_tested)}", ha="center", va="bottom", fontsize=4.2)
    axe.set_xticks([3 * i + 0.5 for i in range(len(Tg))], [g_.replace(" ", "\n") for g_ in Tg.group], fontsize=5.2)
    axe.set_ylim(0, max(tops) * 1.5)
    axe.set_ylabel("Converging neurons (% of tested)")
    axe.set_title("Per cohort and stage (left: inside overlap; right, grey: all tested; p Holm)", fontsize=5.0, loc="left")
    # f: zone categories, pooled
    axf = fig.add_subplot(gb[0, 2])
    q = C[C.group == "All sessions"].set_index("zone").reindex(ZCAT)
    x = np.arange(4)
    axf.bar(x, 100 * q.P, 0.65, color=ZCOL, lw=0, alpha=0.9)
    axf.errorbar(x, 100 * q.P, [100 * (q.P - q.ci_lo), 100 * (q.ci_hi - q.P)], fmt="none", ecolor="k", lw=0.6)
    axf.set_xticks(x, ["outside", "whisker\nonly", "auditory\nonly", "overlap"], fontsize=5.0)
    for xi, n in zip(x, q.n_tested):
        axf.text(xi, 0.3, f"{int(n)}", ha="center", va="bottom", fontsize=4.0, color="white")
    tf = 100 * np.nanmax(q.ci_hi)
    for k, xi in ((1, 1), (2, 2)):
        p = q.iloc[k].get("p_boot_overlap_gt", np.nan)
        if np.isfinite(p):
            yy = tf * (1.08 + 0.17 * (k - 1))
            axf.plot([xi, xi, 3, 3], [yy - tf * 0.03, yy, yy, yy - tf * 0.03], color="k", lw=0.5)
            axf.text((xi + 3) / 2, yy + tf * 0.01, p_label(p), ha="center", va="bottom", fontsize=4.3)
    axf.set_ylim(0, tf * 1.55)
    axf.set_ylabel("Converging neurons (% of tested)")
    axf.set_title("By projection zone (all sessions)", fontsize=5.4, loc="left")
    S.letter_row(fig, [axd, axe, axf], "def", dx_in=0.42, dy_in=0.25)
    for k, ltr in enumerate("abc"):
        fig.text(0.04 / W, 1 - 0.52 / H - k * ph / H - 0.005, ltr, fontsize=9, weight="bold", va="top")
    fig.suptitle("Pre-lick converging neurons and the convergence of whisker- and auditory-cortex projections",
                 x=0.4 / W, y=1 - 0.02 / H, ha="left", va="top", fontsize=7.2, weight="bold")
    S.save(fig, FIG, "prelick_convergence_colocation")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 2
def fig_groups(plt, A, D, slabs, vmax):
    W = S.W_IN
    ncol = len(slabs)
    pw = (W - 0.6) / ncol
    ph = pw * 7.2 / 5.8
    H = 0.5 + 4 * ph + 0.45
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(4, ncol, left=0.55 / W, right=1 - 0.05 / W, top=1 - 0.45 / H, bottom=0.45 / H, wspace=0.03, hspace=0.05)
    cm = cmap_conv()
    for r_i, g in enumerate(GROUPS):
        sel = ((D.cohort == g[0]) & (D.stage == g[1])).to_numpy()
        Dg = D[sel].reset_index(drop=True)
        for c_i, c in enumerate(slabs):
            ax = fig.add_subplot(gs[r_i, c_i])
            m3.draw_section(ax, A, A.section("cor", c), (0, 5.7, 8.0, 0))
            M, extd = slab_density(A, Dg, f"conv_{g[0]}_{g[1]}", c)
            im = ax.imshow(M, extent=extd, cmap=cm, vmin=0, vmax=vmax, interpolation="bilinear", zorder=1.5)
            overlap_contour(ax, A, c)
            m, _, _ = m3.in_slab(Dg, "cor", c)
            ax.text(0.98, 0.02, f"{int((m & Dg.converging.to_numpy()).sum())}/{int(m.sum())}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=4.2, color="0.4")
            finish_ax(ax)
            if r_i == 0:
                ax.set_title(f"AP {(5400 - c) / 1000:+.2f} mm", fontsize=5.8, pad=2)
            if c_i == 0:
                ax.text(-0.05, 0.5, f"{GLAB[g]}\n({Dg.session_id.nunique()} sessions)", transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=5.8, color=COH[g[0]])
    cax = fig.add_axes([0.55 / W, 0.2 / H, 1.5 / W, 0.05 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Converging fraction of tested neurons", fontsize=4.8, labelpad=1)
    cb.ax.tick_params(labelsize=4.4, length=1.2, width=0.4); cb.outline.set_linewidth(0.4)
    fig.text(2.4 / W, 0.22 / H, "purple line: overlap of the whisker- and auditory-cortex projection zones; grey numbers: "
             "converging / tested neurons in the slab", fontsize=4.8, va="center")
    fig.suptitle("Pre-lick converging neurons by cohort and stage (coronal 500-um slabs)", x=0.55 / W, y=1 - 0.05 / H,
                 ha="left", va="top", fontsize=7.2, weight="bold")
    S.save(fig, FIG, "prelick_convergence_groups")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 3 (flatmap)
def fig_flatmaps(plt, A, D, vmax):
    import matplotlib.patheffects as pe
    from matplotlib.path import Path
    D = D.copy()
    D["atlas_id"] = A.unit_ids(D)
    iso = {k for k, p in A.path.items() if 315 in p}
    D = D[D.atlas_id.isin(iso)].reset_index(drop=True)
    ml_left = m7.MID - (D.ml_f.to_numpy() - m7.MID)
    P = m7.projector().project_coordinates(np.c_[D.ccf_atlas_ap, D.ccf_atlas_dv, ml_left].astype(float), scale="voxels",
                                           hemisphere="left", **m7.VIEW)
    ok = np.isfinite(P).all(1)
    D = D[ok].reset_index(drop=True); P = P[ok]
    B = m7.boundaries()
    Z = np.load(m3.ZONES_NPZ)
    zw, za = m7.zone_flat(Z["zone70_whisker"]), m7.zone_flat(Z["zone70_auditory"])
    allp = np.vstack(list(B.values()))
    x0, x1 = int(allp[:, 0].min()) - 10, int(allp[:, 0].max()) + 10
    y0, y1 = int(allp[:, 1].min()) - 10, int(allp[:, 1].max()) + 10
    nx, ny = x1 - x0, y1 - y0
    gx, gy = np.meshgrid(np.arange(nx) + x0 + 0.5, np.arange(ny) + y0 + 0.5)
    pts = np.c_[gx.ravel(), gy.ravel()]
    inside = np.zeros(len(pts), bool)
    for v in B.values():
        inside |= Path(v).contains_points(pts)
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(inside.reshape(ny, nx), iterations=4))

    def zcrop(z):
        zz = np.asarray(z).T
        out = np.zeros((ny, nx), bool)
        ys, xs = np.arange(ny) + y0, np.arange(nx) + x0
        okx, oky = (xs >= 0) & (xs < zz.shape[1]), (ys >= 0) & (ys < zz.shape[0])
        out[np.ix_(oky, okx)] = zz[np.ix_(ys[oky], xs[okx])] >= 0.5
        return out & inside
    zW, zA = zcrop(zw), zcrop(za)
    x, y = P[:, 0] - x0, P[:, 1] - y0
    s = m7.SIGMA_UM / m7.PX_UM
    norm = 2 * np.pi * s ** 2
    cm = cmap_conv()
    sets = [("All sessions", np.ones(len(D), bool), "k")] + [
        (GLAB[g], ((D.cohort == g[0]) & (D.stage == g[1])).to_numpy(), COH[g[0]]) for g in GROUPS]
    W = S.W_IN
    pw = (W - 0.4) / len(sets)
    phh = pw * ny / nx
    H = 0.5 + 2 * phh + 0.55
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, len(sets), left=0.3 / W, right=1 - 0.05 / W, top=1 - 0.5 / H, bottom=0.5 / H, wspace=0.04, hspace=0.1)

    def base(ax):
        for v in B.values():
            ax.plot(v[:, 0] - x0, v[:, 1] - y0, color="0.7", lw=0.25, zorder=2)
        ax.contour(inside.astype(float), levels=[0.5], colors="0.25", linewidths=0.6, zorder=3)
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0); ax.set_aspect("equal"); ax.set_axis_off()

    def zones(ax):
        for z, col, ls in ((zW, m3.SRC_STYLE["whisker"][0], "-"), (zA, m3.SRC_STYLE["auditory"][0], "--")):
            if z.any():
                ax.contour(ndimage.gaussian_filter(z.astype(float), 3), levels=[0.5], colors=[col], linewidths=0.7,
                           linestyles=[ls], zorder=5)
    cv = D.converging.to_numpy()
    for j, (name, sel, col) in enumerate(sets):
        ax = fig.add_subplot(gs[0, j]); base(ax)
        ax.scatter(x[sel & ~cv], y[sel & ~cv], s=0.15, c="0.62", lw=0, zorder=3.5, rasterized=True)
        ax.scatter(x[sel & cv], y[sel & cv], s=0.25, c=CONV, lw=0, zorder=4, rasterized=True)
        ax.set_title(f"{name}\n({int(D.session_id[sel].nunique())} sessions)", fontsize=5.4, pad=2, color=col)
        ax.text(0.98, 0.02, f"{int((sel & cv).sum())}/{int(sel.sum())}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.2, color="0.4")
        xi, yi = x[sel].astype(int), y[sel].astype(int)
        g_ = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        num, den = np.zeros((ny, nx)), np.zeros((ny, nx))
        np.add.at(num, (yi[g_], xi[g_]), cv[sel][g_].astype(float))
        np.add.at(den, (yi[g_], xi[g_]), 1)
        num, den = ndimage.gaussian_filter(num, s), ndimage.gaussian_filter(den, s)
        M = np.where((den * norm >= m7.DENS_MIN) & inside, num / np.maximum(den, 1e-12), np.nan)
        ax = fig.add_subplot(gs[1, j]); base(ax)
        im = ax.imshow(M, cmap=cm, vmin=0, vmax=vmax, interpolation="bilinear", zorder=1)
        zones(ax)
    fig.text(0.05 / W, 1 - 0.5 / H - phh / 2 / H, "Neurons", rotation=90, va="center", fontsize=6)
    fig.text(0.05 / W, 1 - 0.5 / H - 1.5 * phh / H, "Converging fraction", rotation=90, va="center", fontsize=6)
    cax = fig.add_axes([0.3 / W, 0.22 / H, 1.4 / W, 0.05 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Converging fraction of tested neurons", fontsize=4.8, labelpad=1)
    cb.ax.tick_params(labelsize=4.4, length=1.2, width=0.4); cb.outline.set_linewidth(0.4)
    h = [plt.Line2D([], [], color=m3.SRC_STYLE["whisker"][0], lw=1, label=f"whisker-cortex zone ({ZONE_PCT} %)"),
         plt.Line2D([], [], color=m3.SRC_STYLE["auditory"][0], lw=1, ls="--", label=f"auditory-cortex zone ({ZONE_PCT} %)"),
         plt.Line2D([], [], marker="o", ls="", ms=2.5, color=CONV, label="converging neuron")]
    fig.legend(handles=h, loc="lower left", ncol=3, frameon=False, fontsize=4.8, bbox_to_anchor=(2.0 / W, 0.08 / H))
    fig.suptitle("Pre-lick converging neurons across the isocortex (Allen butterfly flatmap, left hemisphere)",
                 x=0.3 / W, y=1 - 0.05 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, "prelick_convergence_flatmaps")
    plt.close(fig)
    return len(D)


# ------------------------------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(0)
    plt = S.setup()
    POUT.mkdir(parents=True, exist_ok=True)
    D = load()
    A = m3.Atlas()
    Z = np.load(m3.ZONES_NPZ)
    T, C = tests(D, Z, rng)
    lab = m2.merged_ids(m2.atlas_50um())
    SR, masks, ov = m6.subregions(Z, lab, A.acr)
    iU, okU = m6.unit_voxels(D, ov.shape)
    SR["n_tested"] = [int(m6.in_mask(mk, iU, okU).sum()) for mk in masks]
    SR = SR[SR.n_tested >= m6.MIN_REC].reset_index(drop=True)
    slabs = m6.pick_slabs(SR)
    p0 = D.converging.mean()
    vmax = max(0.1, np.ceil(3 * p0 * 20) / 20)
    T.to_csv(POUT / f"prelick_colocation_tests{ZTAG}.csv", index=False)
    C.to_csv(POUT / f"prelick_zone_categories{ZTAG}.csv", index=False)
    D[KEYS + ["cohort", "stage", "day", "quality_label", "area_group", "area_acronym_custom", "ccf_atlas_ap", "ccf_atlas_ml",
              "ccf_atlas_dv", f"sel:{AF}", f"sel:{WF}", f"sig:{AF}", f"sig:{WF}", "converging", "converging_sign",
              "in_whisker_zone", "in_auditory_zone", "in_overlap"]].to_parquet(POUT / f"prelick_units_zones{ZTAG}.parquet")
    print(T.drop(columns=[c for c in T.columns if c.endswith(("ci_lo", "ci_hi"))]).round(4).to_string())
    print(C.round(4).to_string())
    fig_colocation(plt, A, D, T, C, slabs, vmax)
    fig_groups(plt, A, D, slabs, vmax)
    n_flat = fig_flatmaps(plt, A, D, vmax)
    prov = dict(script="008_prelick_convergence_colocation.py", prelick_source=str(PRE / "prelick_units.parquet"),
                reference="spontaneous licks (PRELICK_REF=sl)", definition="sig AH vs SL and sig WH vs SL, same sign (065)",
                units="good + mua, tested in both ROCs (mean raw pre-lick rate >= 0.1 Hz)", zones=str(m3.ZONES_NPZ),
                zone_pct=ZONE_PCT, n_tested=len(D), n_converging=int(D.converging.sum()), n_sessions=int(D.session_id.nunique()),
                n_mice=int(D.mouse_id.nunique()), n_flatmap_isocortex=n_flat, n_boot=m6.N_BOOT, slabs_um=list(map(float, slabs)),
                colour_max=float(vmax), groups=[GLAB[g] for g in GROUPS])
    (POUT / f"provenance{ZTAG}.json").write_text(json.dumps(prov, indent=1))
    caption(D, T, C, n_flat)
    print("ALL DONE", FIG / "prelick_convergence_colocation.png")


def caption(D, T, C, n_flat):
    g = T[T.group == "All sessions"].iloc[0]
    q = C[C.group == "All sessions"].set_index("zone")
    lines = [
        "# Pre-lick converging neurons and the convergence of whisker- and auditory-cortex projections", "",
        f"**Pre-lick converging neurons and the convergence of whisker- and auditory-cortex projections.** Neurons: good and "
        f"multi-unit clusters tested in both pre-lick ROCs (100 ms before the first lick; reference: spontaneous licks; mean "
        f"raw pre-lick rate >= 0.1 Hz); {len(D)} neurons from {D.session_id.nunique()} sessions ({D.mouse_id.nunique()} mice). "
        f"Converging neuron: whisker hit vs spontaneous lick and auditory hit vs spontaneous lick both significant (1000 label "
        f"permutations, p < 0.05) with the same sign; {int(D.converging.sum())} neurons ({100 * D.converging.mean():.1f} %).", "",
        f"**a**, Projection zones on coronal 500-um slabs (right hemisphere; neurons folded onto it): {ZONE_PCT} % contours of "
        f"the line-balanced anterograde projection density of whisker cortex (SSp-bfd + SSs, yellow) and auditory cortex (AUDp + "
        f"AUDd/v, blue); purple: overlap. Slabs as in the bimodal co-location figure.",
        "**b**, Tested neurons (grey) and converging neurons (orange); purple line: overlap; numbers: converging / tested.",
        "**c**, Converging fraction: converging and tested neurons counted on the 50-um CCF grid, each smoothed with a 3-D "
        "Gaussian (sigma 150 um) and divided, averaged over the slab; shown where >= 3 neurons fall within the kernel.",
        f"**d**, Pooled converging fraction inside the overlap ({g.n_resp_in} neurons) vs all tested neurons: "
        f"{100 * g.P_in:.1f} % vs {100 * g.P_ref:.1f} % (hierarchical bootstrap {p_label(g.p_boot)}, Fisher {p_label(g.p_fisher)}, "
        f"one-sided); error bars: 95 % hierarchical-bootstrap CI (sessions, then neurons).",
        "**e**, Same per cohort and stage (filled: expert; open: learning day; grey: all tested neurons of the group); p: "
        "hierarchical bootstrap, Holm-corrected over the 4 groups; n: neurons inside the overlap / tested.",
        "**f**, Converging fraction by projection-zone category (all sessions); brackets: overlap vs whisker-zone-only and vs "
        "auditory-zone-only neurons (hierarchical bootstrap, one-sided, exploratory).", "",
        "**Results.** All sessions: " + f"{100 * g.P_in:.1f} % inside the overlap vs {100 * g.P_ref:.1f} % overall "
        f"({100 * g['diff']:+.1f} points, 95 % CI {100 * g.diff_ci_lo:+.1f} to {100 * g.diff_ci_hi:+.1f}; bootstrap "
        f"{p_label(g.p_boot)}). Zone categories: " + "; ".join(
            f"{z} {100 * q.loc[z, 'P']:.1f} % (n = {int(q.loc[z, 'n_tested'])})" for z in ZCAT if z in q.index) + ". Groups: " +
        "; ".join(f"{r.group} {100 * r.P_in:.1f} % vs {100 * r.P_ref:.1f} % (Holm {p_label(r.get('p_boot_holm', np.nan))})"
                  for _, r in T[T.group != "All sessions"].iterrows()) + ".", "",
        f"Supplementary: converging fraction per cohort and stage on the same slabs; isocortex flatmap ({n_flat} isocortical "
        f"tested neurons; Allen butterfly flatmap, left hemisphere, anterior up; density as in c on the 10-um flat grid).", "",
        "Interpretation: co-location only. Projection zones come from other mice (population-averaged excitatory cortical "
        "axons, including axons of passage); converging neurons are mostly lick-related (pre-lick window), so enrichment would "
        "show where modality-general pre-lick activity and converging cortical inputs coincide, not that one drives the other. "
        "CCF uncertainty ~100-200 um. Cohorts and stages differ in sampled areas, so group differences in P are confounded "
        "with sampling; the within-group inside-vs-all comparison is the main test."]
    (POUT / f"prelick_convergence_caption{ZTAG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-6:-4]))


if __name__ == "__main__":
    main()
