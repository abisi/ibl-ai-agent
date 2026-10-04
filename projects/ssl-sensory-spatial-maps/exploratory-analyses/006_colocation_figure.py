"""006 -- Co-location of bimodal neurons with converging whisker and auditory cortical projections (publication figure).

Neurons: good + mua, all sessions pooled; responsive / bimodal from 003.bimodal_classes (active, passive pre, passive
post stimulus-vs-baseline ROC tests; Bonferroni over the modality's number of tests in the session; bimodal = responsive
to both modalities). Quantity: P = fraction of sensory-responsive neurons that are bimodal.
Anatomy (002): whisker zone = 70 % contour of the merged SSp-bfd + SSs projection density, auditory zone = same for
AUDp + AUDd/v; overlap = whisker zone AND auditory zone (50-um grid, right hemisphere; neurons folded onto it).
Sub-regions: the overlap volume cut by Allen structure (layers merged) into 3-D connected pieces (26-connectivity) of
>= MIN_VOL mm^3, named by position within their structure (rostral / intermediate / caudal by AP tertile of the
structure, medial / lateral by its median ML; CP behind AP -1.0 mm from bregma = "CP tail"); generic labels skipped.
Tests (global, no session pairing; user 2026-10-04):
  pooled P inside vs outside the overlap (or a sub-region vs outside the whole overlap);
  Fisher's exact test (= unit-level label permutation), one-sided;
  hierarchical bootstrap of the difference (main test): sessions resampled with replacement from all sessions, then
  neurons within each session (binomial); pooled P inside and outside recomputed per resample; p = fraction of resampled
  differences <= 0 (one-sided); keeps the clustering of neurons within sessions without pairing sessions
  (the spatial-shift null was removed, user 2026-10-04);
  95 % CI of each P: same hierarchical bootstrap;
  Holm across sub-regions; control: sub-region vs the rest of the same structure (outside the overlap), same tests.
Output: combined_results_ks4/_sensory_spatial_maps/colocation_{subregions,tests}.csv, figures/colocation_figure.{png,pdf,svg},
colocation_figure_caption.md
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
m3 = importlib.import_module("003_spatial_maps")
m2 = importlib.import_module("002_projection_zones")
S, OUT = m3.S, m3.OUT
FIG, ZTAG, ZONE_PCT = m3.FIG, m3.ZTAG, m3.ZONE_PCT
MIN_VOL, MIN_UNITS, N_BOOT = 0.04, 30, 2000
MIN_REC = 10                 # sub-regions shown / listed only with >= 10 recorded good + mua neurons inside
GENERIC = {"MB", "TH", "HY", "CTX", "grey", "root", "STR", "PAL", "CB", "P", "MY"}
R50 = 50.0
PURPLE, WCOL, ACOL = "#7b3294", "#f7b519", "#2c2cdb"


# ------------------------------------------------------------------------------------------------ anatomy
def subregions(Z, lab, acr):
    ov = Z["zone70_whisker"] & Z["zone70_auditory"]
    v = (R50 / 1000) ** 3
    rows, masks = [], []
    for k in np.unique(lab[ov]):
        a = acr.get(int(k), str(k))
        if k == 0 or a in GENERIC:
            continue
        mk = ov & (lab == k)
        if mk.sum() * v < MIN_VOL:
            continue
        st = np.argwhere(lab == k)
        st = st[st[:, 2] >= int(5700 / R50)]
        ap_t = np.quantile(st[:, 0], [1 / 3, 2 / 3])
        ml_m = np.median(st[:, 2])
        L, n = ndimage.label(mk, structure=np.ones((3, 3, 3)))
        for c in range(1, n + 1):
            cm = L == c
            if cm.sum() * v < MIN_VOL:
                continue
            ctr = np.argwhere(cm).mean(0)
            ap_mm = 5.4 - (ctr[0] * R50 + R50 / 2) / 1000            # mm from bregma (approx.)
            if a == "CP" and ap_mm < -1.0:
                name = "CP tail"
            else:
                apw = "rostral" if ctr[0] < ap_t[0] else "intermediate" if ctr[0] < ap_t[1] else "caudal"
                mlw = "medial" if ctr[2] < ml_m else "lateral"
                name = f"{a} {apw}-{mlw}"
            rows.append(dict(structure=a, name=name, volume_mm3=cm.sum() * v, ap_bregma_mm=ap_mm,
                             ml_mm=(ctr[2] * R50 + R50 / 2 - 5700) / 1000, dv_mm=(ctr[1] * R50 + R50 / 2) / 1000,
                             frac_of_overlap=cm.sum() / ov.sum()))
            masks.append(cm)
    T = pd.DataFrame(rows)
    # unique names (two pieces with the same label get a, b)
    names, seen = T.name.tolist(), {}
    counts = pd.Series(names).value_counts()
    for i, nm in enumerate(names):
        if counts[nm] > 1:
            seen[nm] = seen.get(nm, 0) + 1
            names[i] = f"{nm} {'abcdefgh'[seen[nm] - 1]}"
    T["name"] = names
    order = np.argsort(-T.volume_mm3.to_numpy())
    T = T.iloc[order].reset_index(drop=True)
    masks = [masks[i] for i in order]
    T.insert(0, "id", np.arange(1, len(T) + 1))
    return T, masks, ov


def unit_voxels(U, shape):
    ijk = np.floor(np.c_[U.ccf_atlas_ap, U.ccf_atlas_dv, U.ml_f].astype(float) / R50)
    ok = np.isfinite(ijk).all(1)
    ijk = np.where(ok[:, None], ijk, -1).astype(int)
    ok &= (ijk >= 0).all(1) & (ijk < np.array(shape)).all(1)
    return ijk, ok


def in_mask(mask, ijk, ok, shift=(0, 0, 0)):
    q = ijk - np.asarray(shift)[None, :]
    good = ok & (q >= 0).all(1) & (q < np.array(mask.shape)).all(1)
    out = np.zeros(len(ijk), bool)
    out[good] = mask[q[good, 0], q[good, 1], q[good, 2]]
    return out


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(D, sel, rng):
    """hierarchical bootstrap of the pooled proportion: sessions with replacement, then neurons (binomial) within"""
    g = D[sel].groupby("session_id").bimodal.agg(["size", "sum"])
    if not len(g):
        return np.nan, np.nan
    n, k = g["size"].to_numpy(), g["sum"].to_numpy()
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    b = rng.binomial(n[idx], (k / n)[idx])
    p = b.sum(1) / n[idx].sum(1)
    return np.percentile(p, 2.5), np.percentile(p, 97.5)


def boot_diff(D, sel_in, sel_ref, rng):
    """hierarchical bootstrap of P(in) - P(ref): sessions with replacement (all sessions contributing to either set), then
    neurons within each session (binomial); returns the resampled differences"""
    g = pd.DataFrame(dict(s=D.session_id.to_numpy(), y=D.bimodal.to_numpy(float), i=sel_in, r=sel_ref & ~sel_in))
    g = g[g.i | g.r]
    a = g[g.i].groupby("s").y.agg(["size", "sum"])
    b = g[g.r].groupby("s").y.agg(["size", "sum"])
    t = a.join(b, how="outer", lsuffix="_i", rsuffix="_r").fillna(0)
    ni, ki, nr, kr = (t[c].to_numpy() for c in ("size_i", "sum_i", "size_r", "sum_r"))
    idx = rng.integers(0, len(t), (N_BOOT, len(t)))
    bi = rng.binomial(ni[idx].astype(int), np.divide(ki, ni, out=np.zeros_like(ki), where=ni > 0)[idx])
    br = rng.binomial(nr[idx].astype(int), np.divide(kr, nr, out=np.zeros_like(kr), where=nr > 0)[idx])
    Ni, Nr = ni[idx].sum(1), nr[idx].sum(1)
    ok = (Ni > 0) & (Nr > 0)
    return bi.sum(1)[ok] / Ni[ok] - br.sum(1)[ok] / Nr[ok]


def test_region(D, ijk, ok, mask, ref_sel, rng, label):
    y = D.bimodal.to_numpy()
    ins = in_mask(mask, ijk, ok)
    a, b = y[ins], y[ref_sel & ~ins]
    row = dict(region=label, n_resp_in=int(ins.sum()), n_bimodal_in=int(a.sum()), n_sessions_in=int(D.session_id[ins].nunique()),
               n_mice_in=int(D.mouse_id[ins].nunique()), P_in=a.mean() if len(a) else np.nan, n_resp_ref=len(b),
               P_ref=b.mean() if len(b) else np.nan)
    if ins.sum() < MIN_UNITS:
        return row, None
    row["diff"] = row["P_in"] - row["P_ref"]
    row["p_fisher"] = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]], alternative="greater")[1]
    null = boot_diff(D, ins, ref_sel, rng)
    row["n_boot"] = len(null)
    row["p_boot"] = (1 + np.sum(null <= 0)) / (1 + len(null))
    row["diff_ci_lo"], row["diff_ci_hi"] = np.percentile(null, [2.5, 97.5])
    row["ci_lo"], row["ci_hi"] = boot_ci(D, ins, rng)
    row["ref_ci_lo"], row["ref_ci_hi"] = boot_ci(D, ref_sel & ~ins, rng)
    return row, null


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


# ------------------------------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(0)
    plt = S.setup()
    U = m3.load_units()
    A = m3.Atlas()
    U["atlas_id"] = A.unit_ids(U)
    U["structure"] = U.atlas_id.map(lambda k: A.acr.get(int(k), ""))
    Z = np.load(m3.ZONES_NPZ)
    lab = m2.merged_ids(m2.atlas_50um())
    SR, masks, ov = subregions(Z, lab, A.acr)
    # keep sub-regions with recorded neurons only (user): >= MIN_REC good + mua neurons inside; renumbered
    iU, okU = unit_voxels(U, ov.shape)
    SR["n_neurons"] = [int(in_mask(mk, iU, okU).sum()) for mk in masks]
    keep = (SR.n_neurons >= MIN_REC).to_numpy()
    print(f"sub-regions: {len(SR)}, with >= {MIN_REC} recorded neurons: {int(keep.sum())}")
    SR, masks = SR[keep].reset_index(drop=True), [m for m, k in zip(masks, keep) if k]
    SR["id"] = np.arange(1, len(SR) + 1)
    D = U[U.bimodal_cat.isin([1, 2, 3])].reset_index(drop=True)
    D["bimodal"] = D.bimodal_cat == 3
    ijk, ok = unit_voxels(D, ov.shape)
    D["in_overlap"] = in_mask(ov, ijk, ok)
    allref = np.ones(len(D), bool)
    rows, nulls = [], {}
    r, nl = test_region(D, ijk, ok, ov, allref, rng, "whole overlap zone")
    rows.append(dict(r, kind="global")); nulls["whole overlap zone"] = nl
    outside = ~D.in_overlap.to_numpy()
    for sr, mk in zip(SR.itertuples(), masks):
        r, nl = test_region(D, ijk, ok, mk, outside, rng, sr.name)
        r.update(kind="sub-region", id=sr.id, structure=sr.structure, volume_mm3=sr.volume_mm3)
        # control: rest of the same structure, outside the overlap
        rest = (D.structure == sr.structure).to_numpy() & outside
        yb = D.bimodal.to_numpy()
        ins = in_mask(mk, ijk, ok)
        if ins.sum() >= MIN_UNITS and rest.sum() >= MIN_UNITS:
            a, b = yb[ins], yb[rest]
            r.update(P_rest_structure=b.mean(), n_rest_structure=int(rest.sum()),
                     p_fisher_vs_rest=stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]],
                                                         alternative="greater")[1])
            r["rest_ci_lo"], r["rest_ci_hi"] = boot_ci(D, rest, rng)
            bd = boot_diff(D, ins, rest, rng)
            r["p_boot_vs_rest"] = (1 + np.sum(bd <= 0)) / (1 + len(bd))
        rows.append(r); nulls[sr.name] = nl
    T = pd.DataFrame(rows)
    sub = (T.kind == "sub-region") & T.p_boot.notna()
    T.loc[sub, "p_boot_holm"] = holm(T.loc[sub, "p_boot"])
    T.loc[sub, "p_fisher_holm"] = holm(T.loc[sub, "p_fisher"])
    if "p_boot_vs_rest" in T:
        sv = sub & T.p_boot_vs_rest.notna()
        T.loc[sv, "p_vs_rest_holm"] = holm(T.loc[sv, "p_boot_vs_rest"])
        T.loc[sv, "p_fisher_vs_rest_holm"] = holm(T.loc[sv, "p_fisher_vs_rest"])
    SR.to_csv(OUT / f"colocation_subregions{ZTAG}.csv", index=False)
    T.to_csv(OUT / f"colocation_tests{ZTAG}.csv", index=False)
    print(SR.round(3).to_string())
    print(T.drop(columns=[c for c in T.columns if c.endswith("ci_lo") or c.endswith("ci_hi")]).round(4).to_string())
    figure(plt, A, U, D, SR, masks, T, nulls, Z)
    caption(SR, T, D)
    print("ALL DONE", FIG / "colocation_figure.png")


# ------------------------------------------------------------------------------------------------ figure
def pick_slabs(SR, n=6):
    """coronal slab centres (um) through the tested sub-regions, >= 0.6 mm apart, sorted rostral -> caudal"""
    cands = (5400 - 1000 * SR.ap_bregma_mm).tolist()
    picked = []
    for c in cands:
        if all(abs(c - p) >= 600 for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    for c in np.arange(4000, 9600, 900):                          # fill if fewer sub-regions
        if len(picked) >= n:
            break
        if all(abs(c - p) >= 600 for p in picked):
            picked.append(c)
    return sorted(picked)


def short_p(p):
    return "n/a" if not np.isfinite(p) else "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def figure(plt, A, U, D, SR, masks, T, nulls, Z):
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    W = S.W_IN
    slabs = pick_slabs(SR[SR.id.isin(T[T.kind == "sub-region"].dropna(subset=["p_boot"]).id)] if (T.kind == "sub-region").any() else SR)
    ncol = len(slabs)
    pw = (W - 0.45) / ncol
    ph = pw * 7.2 / 5.8
    H = 0.45 + 3 * ph + 0.35 + 2.3
    fig = plt.figure(figsize=(W, H))
    top_frac = (3 * ph) / H
    gs = fig.add_gridspec(3, ncol, left=0.4 / W, right=1 - 0.05 / W, top=1 - 0.42 / H, bottom=1 - 0.42 / H - top_frac,
                          wspace=0.03, hspace=0.05)
    gb = fig.add_gridspec(1, 4, left=0.5 / W, right=1 - 0.42 / W, top=1.85 / H, bottom=0.42 / H,
                          width_ratios=[0.75, 1.0, 2.5, 1.15], wspace=0.75)
    cmap_b = LinearSegmentedColormap.from_list("white_bimodal", ["#ffffff", "#c2a5cf", "#7b3294", "#40004b"])
    ijk_all = None
    tested = T[(T.kind == "sub-region") & T.p_boot.notna()]
    v_bim = np.where(D.bimodal, 1.0, 0.0)
    for c_i, c in enumerate(slabs):
        sec = A.section("cor", c)
        zw, za = A.zone_section("whisker", "cor", c), A.zone_section("auditory", "cor", c)
        zo = zw & za
        r = A.zres / 1000
        ext = (0, zw.shape[1] * r, zw.shape[0] * r, 0)
        yy, xx = (np.arange(zw.shape[0]) + 0.5) * r, (np.arange(zw.shape[1]) + 0.5) * r
        axs = [fig.add_subplot(gs[k, c_i]) for k in range(3)]
        # a: anatomy
        ax = axs[0]
        for z_, col in ((zw, WCOL), (za, ACOL)):
            ax.imshow(np.ma.masked_where(~z_, z_.astype(float)), extent=ext, cmap=ListedColormap([col]), alpha=0.32,
                      interpolation="nearest", zorder=1)
        ax.imshow(np.ma.masked_where(~zo, zo.astype(float)), extent=ext, cmap=ListedColormap([PURPLE]), alpha=0.75,
                  interpolation="nearest", zorder=1.3)
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        i0, i1 = int((c - 250) / 50), int((c + 250) / 50)
        for sr, mk in zip(SR.itertuples(), masks):
            sl = mk[max(i0, 0):i1].any(0)[:, int(5700 / 50):]
            if sl.sum() >= 4:
                yy_, xx_ = np.where(sl)
                ax.text((xx_.mean() + 0.5) * r, (yy_.mean() + 0.5) * r, str(sr.id), fontsize=5.0, weight="bold",
                        ha="center", va="center", color="white" if sr.id in tested.id.values else "0.85", zorder=8,
                        path_effects=[pe.withStroke(linewidth=1.3, foreground="k")])
        # b: functional (neurons)
        ax = axs[1]
        m, x, y = m3.in_slab(D, "cor", c)
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        nb = m & ~D.bimodal.to_numpy()
        ax.scatter(x[nb], y[nb], s=0.15, c="0.72", lw=0, zorder=3, rasterized=True)
        bb = m & D.bimodal.to_numpy()
        ax.scatter(x[bb], y[bb], s=0.5, c=PURPLE, lw=0, zorder=4, rasterized=True)
        # c: density of P
        ax = axs[2]
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        if "bim" not in m3._DCACHE:
            m3._DCACHE["bim"] = m3.density_volumes(D, v_bim)
        M, extd = m3.slab_ratio(*m3._DCACHE["bim"], "cor", c)
        _, inside = A.boundaries(sec)
        gy = ((np.arange(M.shape[0]) + 0.5) * 5).astype(int)
        gx = ((np.arange(M.shape[1]) + 0.5) * 5).astype(int)
        inb = inside[np.clip(gy, 0, inside.shape[0] - 1)][:, np.clip(gx, 0, inside.shape[1] - 1)]
        imd = ax.imshow(np.where(inb, M, np.nan), extent=extd, cmap=cmap_b, vmin=0, vmax=0.5, interpolation="bilinear", zorder=1.5)
        for ax in axs[1:]:
            if zo.any():
                ax.contour(xx, yy, ndimage.gaussian_filter(zo.astype(float), 0.8), levels=[0.5], colors=[PURPLE],
                           linewidths=0.7, zorder=6)
        for ax in axs:
            ax.set_xlim(0, 5.8); ax.set_ylim(7.2, 0); ax.set_aspect("equal"); ax.set_axis_off()
        axs[0].set_title(f"AP {(5400 - c) / 1000:+.2f} mm", fontsize=5.8, pad=2)
        axs[1].text(0.98, 0.02, f"{int(m.sum())}", transform=axs[1].transAxes, ha="right", va="bottom", fontsize=4.4, color="0.4")
        if c_i == 0:
            for ax, t in zip(axs, ["Projection zones", "Responsive neurons", "Bimodal fraction"]):
                ax.text(-0.05, 0.5, t, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6)
            axs[2].plot([0.2, 1.2], [7.0, 7.0], color="k", lw=0.9, solid_capstyle="butt")
            axs[2].text(0.7, 6.85, "1 mm", ha="center", va="bottom", fontsize=4.6)
    # legends for the slab rows
    hz = [plt.Rectangle((0, 0), 1, 1, color=WCOL, alpha=0.4, lw=0, label="whisker-cortex zone"),
          plt.Rectangle((0, 0), 1, 1, color=ACOL, alpha=0.4, lw=0, label="auditory-cortex zone"),
          plt.Rectangle((0, 0), 1, 1, color=PURPLE, alpha=0.75, lw=0, label="overlap (numbers: sub-regions)"),
          plt.Line2D([], [], marker="o", ls="", ms=2.5, color="0.72", label="responsive, one modality"),
          plt.Line2D([], [], marker="o", ls="", ms=2.5, color=PURPLE, label="bimodal")]
    fig.legend(handles=hz, loc="upper left", bbox_to_anchor=(0.4 / W, 1 - 0.1 / H), ncol=5, frameon=False, fontsize=5.0,
               handlelength=1.2, columnspacing=1.0)
    c_bottom = 1 - 0.42 / H - top_frac
    cax = fig.add_axes([0.45 / W, c_bottom - 0.16 / H, 1.5 / W, 0.05 / H])
    cb = fig.colorbar(imd, cax=cax, orientation="horizontal")
    cb.set_label("Bimodal fraction of responsive neurons (3-D Gaussian, sigma 150 um)", fontsize=4.8, labelpad=1)
    cb.ax.tick_params(labelsize=4.4, length=1.2, width=0.4); cb.outline.set_linewidth(0.4)
    # bottom row
    axd, axe, axf, axg = [fig.add_subplot(gb[0, k]) for k in range(4)]
    g = T[T.kind == "global"].iloc[0]
    axd.bar([0, 1], [100 * g.P_in, 100 * g.P_ref], 0.62, color=[PURPLE, "0.6"], lw=0,
            yerr=[[100 * (g.P_in - g.ci_lo), 100 * (g.P_ref - g.ref_ci_lo)], [100 * (g.ci_hi - g.P_in), 100 * (g.ref_ci_hi - g.P_ref)]],
            error_kw=dict(lw=0.6))
    axd.set_xticks([0, 1], ["inside\noverlap", "all\nresponsive"], fontsize=5.2)
    axd.set_ylabel("Bimodal neurons (% of responsive)")
    topd = 100 * max(g.ci_hi, g.ref_ci_hi)
    axd.plot([0, 0, 1, 1], [topd + 1.5, topd + 2.5, topd + 2.5, topd + 1.5], color="k", lw=0.5)
    axd.text(0.5, topd + 3, f"bootstrap {S.fmt_p(g.p_boot)}\nFisher {S.fmt_p(g.p_fisher)}", ha="center", va="bottom", fontsize=4.5)
    axd.set_ylim(0, topd + 13)
    for xx_, nn in ((0, g.n_resp_in), (1, g.n_resp_ref)):
        axd.text(xx_, 1.0, f"{nn}", ha="center", va="bottom", fontsize=4.3, color="white")
    axd.set_title("Whole overlap", fontsize=5.4, loc="left")
    # e: bootstrap distribution of the difference
    nl = nulls.get("whole overlap zone")
    if nl is not None and len(nl):
        axe.hist(100 * nl, bins=40, color=PURPLE, alpha=0.55, lw=0, edgecolor="none")
        axe.axvline(0, color="k", lw=0.6)
        axe.axvline(100 * g["diff"], color=PURPLE, lw=1.0)
        axe.text(100 * g["diff"], axe.get_ylim()[1] * 0.97, " observed", color=PURPLE, fontsize=4.6, va="top")
        axe.set_xlabel("Inside minus all responsive\n(% points)")
        axe.set_ylabel("Bootstrap resamples")
        axe.set_title(f"Hierarchical bootstrap ({len(nl)})", fontsize=5.4, loc="left")
    # f: tested sub-regions
    Q = T[(T.kind == "sub-region") & T.p_boot.notna()].sort_values("id").reset_index(drop=True)
    axf.axvspan(100 * g.ref_ci_lo, 100 * g.ref_ci_hi, color="0.88", lw=0, edgecolor="none", zorder=0)
    axf.axvline(100 * g.P_ref, color="0.45", lw=0.6, ls="--", zorder=1)
    for i, q in Q.iterrows():
        axf.errorbar(100 * q.P_in, i, xerr=[[100 * (q.P_in - q.ci_lo)], [100 * (q.ci_hi - q.P_in)]], fmt="o", ms=2.6,
                     color=PURPLE, lw=0.7, capsize=0, zorder=3)
        if np.isfinite(q.get("P_rest_structure", np.nan)):
            axf.plot(100 * q.P_rest_structure, i, marker="|", ms=5.5, color="0.2", mew=0.9, zorder=2)
        axf.text(1.02, i, short_p(q.p_boot_holm), va="center", fontsize=4.3, transform=axf.get_yaxis_transform())
    axf.text(1.02, -0.9, "bootstrap p\n(Holm)", fontsize=4.3, va="bottom", transform=axf.get_yaxis_transform())
    axf.set_yticks(range(len(Q)), [f"{int(q.id)}  {q.region}  ({int(q.n_resp_in)} / {int(q.n_sessions_in)})"
                                   for q in Q.itertuples()], fontsize=4.6)
    axf.set_ylim(len(Q) - 0.4, -0.6)
    axf.set_xlim(0, max(70, 100 * Q.ci_hi.max() + 5))
    axf.set_xlabel("Bimodal neurons (% of responsive)")
    axf.set_title("Overlap sub-regions (responsive neurons / sessions): dot = inside\n(95 % CI), | = rest of the "
                  "structure, dashed = all responsive neurons", fontsize=5.0, loc="left")
    # g: within-structure difference (same rows as f)
    for j, q in Q.iterrows():
        if not np.isfinite(q.get("P_rest_structure", np.nan)):
            continue
        d = 100 * (q.P_in - q.P_rest_structure)
        axg.barh(j, d, color=PURPLE if d > 0 else "0.6", height=0.62, lw=0)
        axg.text(1.02, j, short_p(q.p_vs_rest_holm), va="center", fontsize=4.3, transform=axg.get_yaxis_transform())
    axg.text(1.02, -0.9, "bootstrap p\n(Holm)", fontsize=4.3, va="bottom", transform=axg.get_yaxis_transform())
    axg.axvline(0, color="k", lw=0.5)
    axg.set_yticks(range(len(Q)), [str(int(q.id)) for q in Q.itertuples()], fontsize=4.8)
    axg.set_ylim(len(Q) - 0.4, -0.6)
    lim = max(5, np.nanmax(np.abs(100 * (Q.P_in - Q.P_rest_structure)))) * 1.1
    axg.set_xlim(-lim, lim)
    axg.set_xlabel("Inside minus rest of the\nstructure (% points)")
    axg.set_title("Same structure", fontsize=5.4, loc="left")
    S.letter_row(fig, [axd, axe, axf, axg], "defg", dx_in=0.42, dy_in=0.25)
    for k, ltr in enumerate("abc"):
        fig.text(0.04 / W, 1 - 0.42 / H - k * ph / H - 0.005, ltr, fontsize=9, weight="bold", va="top")
    fig.suptitle("Bimodal neurons and the convergence of whisker- and auditory-cortex projections",
                 x=0.4 / W, y=1 - 0.01 / H, ha="left", va="top", fontsize=7.2, weight="bold")
    S.save(fig, FIG, "colocation_figure")
    plt.close(fig)


def caption(SR, T, D):
    g = T[T.kind == "global"].iloc[0]
    Q = T[T.kind == "sub-region"].sort_values("id")
    sig = Q[Q.p_boot_holm < 0.05] if "p_boot_holm" in Q else Q.iloc[:0]
    lines = [
        "# Bimodal neurons and the convergence of whisker- and auditory-cortex projections",
        "",
        f"**Bimodal neurons and the convergence of whisker- and auditory-cortex projections.** Neurons: good and multi-unit "
        f"clusters, all sessions (both cohorts, learning day and expert days); {len(D)} sensory-responsive neurons from "
        f"{D.session_id.nunique()} sessions. A neuron is responsive to a modality if any of its stimulus-vs-baseline ROC tests "
        f"(active, passive pre, passive post; 5-35 ms after stimulus onset vs the pre-trial baseline, 1000 label permutations) "
        f"is significant after Bonferroni correction over the number of such tests in its session (1-3); bimodal = responsive "
        f"to both whisker and auditory stimuli (excited or inhibited).",
        "",
        f"**a**, Projection zones on coronal 500-um slabs (right hemisphere; neurons folded onto it). Whisker zone (yellow): {ZONE_PCT} % "
        "contour of the merged anterograde projection density of SSp-bfd and SSs (Allen Mouse Brain Connectivity Atlas; wild-type "
        "and Emx1-IRES-Cre injections, 8 and 3 experiments, each normalised to its total); auditory zone (blue): same for AUDp "
        "and AUDd/AUDv (6 and 3); purple: overlap of the two zones. Numbers: overlap sub-regions (the overlap volume cut by "
        f"Allen structure into connected 3-D pieces >= {MIN_VOL} mm^3, named by their position within the structure); only "
        f"sub-regions holding >= {MIN_REC} recorded neurons are shown and listed (structures without recorded neurons omitted); "
        f"white numbers: sub-regions with >= {MIN_UNITS} sensory-responsive neurons, tested.",
        "**b**, Sensory-responsive neurons in the same slabs (grey: one modality; purple: bimodal); line: overlap zone; grey "
        "number: responsive neurons in the slab.",
        "**c**, Bimodal fraction: bimodal neurons and responsive neurons counted on the 50-um CCF grid, each smoothed with a "
        "3-D Gaussian (sigma 150 um) and divided (bimodal density normalised by the density of recorded responsive neurons), "
        "averaged over the 500-um slab; shown where >= 3 neurons fall within the kernel.",
        f"**d**, Pooled bimodal fraction inside the overlap zone ({g.n_resp_in} responsive neurons, {g.n_sessions_in} sessions, "
        f"{g.n_mice_in} mice) and among all responsive neurons ({g.n_resp_ref}): {100 * g.P_in:.1f} % vs {100 * g.P_ref:.1f} %; "
        f"error bars: 95 % hierarchical-bootstrap CI (sessions, then neurons). p: hierarchical bootstrap of the difference "
        f"({S.fmt_p(g.p_boot)}) and Fisher's exact test ({S.fmt_p(g.p_fisher)}, one-sided; neurons treated as independent).",
        f"**e**, Hierarchical bootstrap of the difference (inside minus all responsive neurons; {int(g.n_boot)} resamples): "
        "sessions resampled with replacement, then neurons within each session; p = fraction of resampled differences <= 0. "
        "It keeps the clustering of neurons within sessions and does not require the same sessions inside and outside.",
        "**f**, Bimodal fraction per overlap sub-region (dots, 95 % bootstrap CI), the rest of the same structure outside the "
        "overlap (vertical tick) and all responsive neurons (dashed line, grey band: 95 % CI). p: hierarchical bootstrap of the "
        f"difference to all responsive neurons outside the overlap, Holm-corrected across the {Q.p_boot.notna().sum()} tested "
        "sub-regions; n: responsive neurons / sessions inside.",
        "**g**, Sub-region minus the rest of its structure (percentage points); p: hierarchical bootstrap, Holm-corrected "
        "(Fisher's exact test in colocation_tests.csv).",
        "",
        "Interpretation: co-location. The projection zones come from other mice (population-averaged tracing of excitatory "
        "cortical axons, including axons of passage), so overlap marks where whisker and auditory cortical inputs can converge; "
        "it does not show that the bimodal responses are driven by these inputs. CCF positions of the recorded neurons carry an "
        "uncertainty of roughly 100-200 um.",
        "",
        "Sub-regions (id, name, volume, centroid AP from bregma):",
    ]
    vr = Q[Q.get("p_vs_rest_holm", pd.Series(np.nan, index=Q.index)) < 0.05] if "p_vs_rest_holm" in Q else Q.iloc[:0]
    res = (f"**Results.** Inside the overlap, {100 * g.P_in:.1f} % of responsive neurons were bimodal vs {100 * g.P_ref:.1f} % "
           f"overall (difference {100 * g['diff']:+.1f} % points, 95 % CI {100 * g.diff_ci_lo:+.1f} to {100 * g.diff_ci_hi:+.1f}; "
           f"bootstrap {S.fmt_p(g.p_boot)}, Fisher {S.fmt_p(g.p_fisher)}). {len(sig)} of {Q.p_boot.notna().sum()} tested "
           "sub-regions had more bimodal neurons than all responsive neurons outside the overlap after Holm correction"
           + (f" ({', '.join(sig.region)})" if len(sig) else "") + ". Compared with the rest of their own structure, "
           + (", ".join(f"{q.region} ({100 * q.P_in:.0f} % vs {100 * q.P_rest_structure:.0f} %, Holm {S.fmt_p(q.p_vs_rest_holm)})"
                        for q in vr.itertuples()) if len(vr) else "no sub-region")
           + " had more bimodal neurons. Sub-regions with < "
           f"{MIN_UNITS} responsive neurons were not tested.")
    lines.insert(lines.index("Interpretation: co-location. The projection zones come from other mice (population-averaged tracing of excitatory "
                             "cortical axons, including axons of passage), so overlap marks where whisker and auditory cortical inputs can converge; "
                             "it does not show that the bimodal responses are driven by these inputs. CCF positions of the recorded neurons carry an "
                             "uncertainty of roughly 100-200 um."), res + "\n")
    for sr in SR.itertuples():
        q = Q[Q.id == sr.id]
        extra = ""
        if len(q) and np.isfinite(q.iloc[0].get("p_boot", np.nan)):
            qq = q.iloc[0]
            extra = (f"; bimodal {100 * qq.P_in:.1f} % (n = {qq.n_resp_in}, {qq.n_sessions_in} sessions), bootstrap p (Holm) "
                     f"{qq.p_boot_holm:.3g}")
        lines.append(f"- {sr.id}: {sr.name}, {sr.volume_mm3:.2f} mm^3, AP {sr.ap_bregma_mm:+.2f} mm{extra}")
    (OUT / f"colocation_figure_caption{ZTAG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
