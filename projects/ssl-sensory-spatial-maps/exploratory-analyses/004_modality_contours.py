"""004 -- Do whisker-preferring and auditory-preferring neurons occupy different locations within an area?

Neurons: good + mua, all sessions pooled, significant in the rate-based ROC wh_vs_aud_active (whisker-preferring:
selectivity < 0; auditory-preferring: > 0). Slabs: the coronal slabs centred on projection zones / areas of 003 (500 um);
within a slab, its target areas (SLAB_AREAS) with >= 15 neurons of each preference and >= 3 sessions (every other area
of the slab is tested too and saved separately as exploratory: modality_contours_all_areas.csv).
Per slab x area, positions in the slab plane (lateral distance from the midline, depth; mm):
  80 % contour: 2-D Gaussian kernel density (Scott bandwidth; positions jittered by 10 um, seeded, for the density only --
  neurons of one probe track are collinear) of each group; the contour enclosing the highest-density
  region that holds 80 % of the group's density.
  Location difference: distance between the two groups' centroids (mean positions), and the signed shifts along each
  axis (auditory - whisker: lateral, depth). Overlap: Dice coefficient of the two 80 % regions.
  Test: preference labels permuted among the area's significant neurons WITHIN each session (keeps each session's probe
  positions and its whisker / auditory counts), N_PERM permutations; p = (1 + #null >= observed) / (1 + N_PERM) for the
  centroid distance (one-sided) and two-sided for each axis shift. Holm correction across all slab x area tests is also
  reported over the target tests (uncorrected p shown in the figure).
Output: combined_results_ks4/_sensory_spatial_maps/modality_contours.csv + figures/modality_contours.{png,pdf,svg}
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m3 = importlib.import_module("003_spatial_maps")
S, OUT, FIG = m3.S, m3.OUT, m3.FIG
# areas tested in each target slab (area_acronym_custom; TEa, PO and MG have no separate custom label -> AUD, TH)
SLAB_AREAS = {"SSp-bfd": ["SSp-bfd"], "SSs": ["SSs"], "wM1": ["MO-wM1"], "wM2": ["MO-wM2"], "ORB": ["ORB"],
              "Anterior striatum": ["DMS", "DLS", "VS"], "Caudal striatum": ["DMS", "DLS"], "Striatum tail": ["TS"],
              "SCm": ["SCm"], "AUDp": ["AUD"], "TEa": ["AUD"], "PO (thalamus)": ["TH"], "MG (thalamus)": ["TH"]}
MIN_PER_GROUP, MIN_SESSIONS, N_PERM, LEVEL, JITTER_MM = 15, 3, 5000, 0.80, 0.010


def hdr_level(kde, pts, frac):
    """density threshold enclosing `frac` of the probability mass (sample-based: quantile of the density at the points)"""
    return np.quantile(kde(pts.T), 1 - frac)


def grid_kde(pts, xr, yr, n=200):
    from scipy.stats import gaussian_kde
    pts = pts + np.random.default_rng(1).normal(0, JITTER_MM, pts.shape)   # probe tracks can be collinear
    k = gaussian_kde(pts.T)
    gx, gy = np.linspace(*xr, n), np.linspace(*yr, n)
    X, Y = np.meshgrid(gx, gy)
    Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return gx, gy, Z, hdr_level(k, pts, LEVEL)


def stat(P, lab):
    a, w = P[lab], P[~lab]
    d = a.mean(0) - w.mean(0)
    return np.hypot(*d), d


def perm_test(P, lab, sess, rng):
    obs_d, obs_v = stat(P, lab)
    groups = [np.where(sess == s)[0] for s in np.unique(sess)]
    nd, nv = np.empty(N_PERM), np.empty((N_PERM, 2))
    for k in range(N_PERM):
        l2 = lab.copy()
        for g in groups:
            l2[g] = rng.permutation(lab[g])
        nd[k], nv[k] = stat(P, l2)
    p_d = (1 + np.sum(nd >= obs_d)) / (1 + N_PERM)
    p_v = [(1 + np.sum(np.abs(nv[:, j]) >= abs(obs_v[j]))) / (1 + N_PERM) for j in range(2)]
    return obs_d, obs_v, p_d, p_v


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


def main():
    plt = S.setup()
    U = m3.load_units()
    A = m3.Atlas()
    sets = m3.slab_sets(U, A)
    v, sig = m3.values(U, "modality")
    U = U.assign(mod_sel=v, mod_sig=sig)
    rng = np.random.default_rng(0)
    rows, panels = [], []
    for lab_slab, c in sets["targets"][1]:
        m, x, y = m3.in_slab(U, "cor", c)
        W = U[m & U.mod_sig.to_numpy()].assign(x=x[m & U.mod_sig.to_numpy()], y=y[m & U.mod_sig.to_numpy()])
        for area, g in W.groupby("area_acronym_custom"):
            aud = (g.mod_sel > 0).to_numpy()
            if aud.sum() < MIN_PER_GROUP or (~aud).sum() < MIN_PER_GROUP or g.session_id.nunique() < MIN_SESSIONS:
                continue
            P = g[["x", "y"]].to_numpy()
            d, dv, p_d, p_v = perm_test(P, aud, g.session_id.to_numpy(), rng)
            xr = (P[:, 0].min() - 0.3, P[:, 0].max() + 0.3)
            yr = (P[:, 1].min() - 0.3, P[:, 1].max() + 0.3)
            ka, kw = grid_kde(P[aud], xr, yr), grid_kde(P[~aud], xr, yr)
            ina, inw = ka[2] >= ka[3], kw[2] >= kw[3]
            dice = 2 * (ina & inw).sum() / max(1, ina.sum() + inw.sum())
            rows.append(dict(slab=lab_slab, slab_centre_um=c, area=area,
                             target=area in SLAB_AREAS.get(lab_slab.split(",")[0], []), n_whisker_pref=int((~aud).sum()),
                             n_auditory_pref=int(aud.sum()), n_sessions=g.session_id.nunique(), n_mice=g.mouse_id.nunique(),
                             centroid_distance_um=1000 * d, shift_lateral_um=1000 * dv[0], shift_depth_um=1000 * dv[1],
                             dice_80=dice, p_distance=p_d, p_lateral=p_v[0], p_depth=p_v[1], n_perm=N_PERM))
            panels.append((lab_slab, c, area, P, aud, ka, kw))
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "modality_contours_all_areas.csv", index=False)          # exploratory: every area of every slab
    keep = T.target.to_numpy()
    T = T[keep].reset_index(drop=True)
    T["p_distance_holm"] = holm(T.p_distance)
    T.to_csv(OUT / "modality_contours.csv", index=False)
    figure(plt, A, T, [p for p, k in zip(panels, keep) if k])
    print("ALL DONE", OUT / "modality_contours.csv")


def figure(plt, A, T, panels):
    n = len(panels)
    nc = 5
    nr = int(np.ceil(n / nc))
    W = S.W_IN
    H = 1.75 * nr + 0.5
    fig, axs = plt.subplots(nr, nc, figsize=(W, H), squeeze=False, gridspec_kw=dict(wspace=0.12, hspace=0.6))
    for ax in axs.ravel()[n:]:
        ax.set_axis_off()
    for ax, (lab_slab, c, area, P, aud, ka, kw), r in zip(axs.ravel(), panels, T.itertuples()):
        sec = A.section("cor", c)
        m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
        for pts, col in ((P[~aud], S.WH_C), (P[aud], S.AUD_C)):
            ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c=col, lw=0, alpha=0.6, zorder=4, rasterized=True)
        for (gx, gy, Z, lev), col in ((kw, S.WH_C), (ka, S.AUD_C)):
            ax.contour(gx, gy, Z, levels=[lev], colors=[col], linewidths=0.9, zorder=5)
        for pts, col in ((P[~aud], S.WH_C), (P[aud], S.AUD_C)):
            mu = pts.mean(0)
            ax.plot(*mu, marker="+", ms=5, mew=1.1, color="k", zorder=7)
            ax.plot(*mu, marker="+", ms=4, mew=0.7, color=col, zorder=8)
        pad = 0.6
        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].max() + pad, P[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{area}\n{lab_slab}", fontsize=5.0, pad=2, linespacing=1.1)
        ax.text(0.0, -0.02, f"Δ = {r.centroid_distance_um:.0f} µm, {S.fmt_p(r.p_distance)}\n"
                            f"lat {r.shift_lateral_um:+.0f} ({S.fmt_p(r.p_lateral)}), depth {r.shift_depth_um:+.0f} ({S.fmt_p(r.p_depth)})\n"
                            f"n = {r.n_whisker_pref} W / {r.n_auditory_pref} A, {r.n_sessions} sessions",
                transform=ax.transAxes, fontsize=4.2, va="top", ha="left", clip_on=False)
        x1, y1 = ax.get_xlim()[1] - 0.1, ax.get_ylim()[1] + 0.12
        ax.plot([x1 - 1.0, x1], [y1, y1], color="k", lw=0.8, solid_capstyle="butt")
    h = [plt.Line2D([], [], color=S.WH_C, lw=1, label="whisker-preferring, 80 % contour"),
         plt.Line2D([], [], color=S.AUD_C, lw=1, label="auditory-preferring, 80 % contour"),
         plt.Line2D([], [], color="k", marker="+", ls="", label="centroid")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Location of whisker- vs auditory-preferring neurons within areas (coronal 500-um slabs)",
                 x=0.02, ha="left", fontsize=7, weight="bold")
    fig.text(0.02, 0.985 - 0.25 / H, "Δ: distance between centroids; lat / depth: auditory − whisker shift (µm); p: preference "
             "labels permuted within sessions (shifts due only to which sessions contributed each preference are not "
             "counted), uncorrected; Holm-corrected values in modality_contours.csv. Scale bar 1 mm.",
             fontsize=4.8, ha="left", va="top", color="0.25")
    S.save(fig, FIG, "modality_contours")
    plt.close(fig)


if __name__ == "__main__":
    main()
