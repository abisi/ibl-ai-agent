"""Figure 3 panel C, option (iv) (user 2026-11-25): left, time-resolved decoding per area group as square panels stacked
vertically (one per decoder); right, the rastermap cluster axis (100 clusters, metacluster boundaries as in panel A) and
one dot column per decoder (dot per fine cluster).

Left: accuracy minus null over time, one line per area group (palette: ephys_utilities custom area-group colours).
  pseudo-population decoders (whisker / auditory detection, whisker vs auditory, movement): N = 200 area-group
  pseudo-populations, mean over iterations of (accuracy - that iteration's null mean);
  hit vs miss: per-session area-group decoders ('whole' condition), real minus mean linear-shift null, mean over sessions
  (area groups with >= MIN_SESS sessions).
Right: per fine cluster and decoder, the peak (25-ms smoothed) z of the cluster-mean decoder weight (log2 |w| relative to
  the decoder's average neuron) against cluster labels permuted within session (N_PERM); dot area = peak z (clipped),
  colour = onset (first time z >= 3 for 2 bins); grey ring = z < 3.
Movement = the first movement run (earliest onset, partial); the jaw-aligned run replaces it when finished.
Output: manuscript Figures/_explore/Figure3_C_options_<date>/option_iv_*.{png,pdf,csv}
Run (haas): cd ~/code/unit_spikes_analysis; RASTERMAP_META_CUT=0.30 PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python <this>
"""
import datetime
import glob
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "projects" / "ssl-decoding-rerun-202611"))
f005 = importlib.import_module("005_rastermap_decoding_figure")
import ephys_utilities.allen_utils.allen_utils as allen_utils       # noqa: E402

OUT = (pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/_explore")
       / f"Figure3_C_options_{datetime.date.today().isoformat()}")
H = f005.p004.H
DEC = [("wdet", "Whisker detection"), ("adet", "Auditory detection"), ("wva", "Whisker vs auditory"),
       ("hm", "Hit vs miss (whisker)"), ("move", "Movement (FA vs CR)")]
WIN = {"wdet": (-0.05, 0.3), "adet": (-0.05, 0.3), "wva": (-0.05, 0.3), "hm": (-0.05, 0.3), "move": (-0.2, 0.3)}
PEAK_WIN = {"wdet": (0, 0.3), "adet": (0, 0.3), "wva": (0, 0.3), "hm": (0, 0.3), "move": (-0.2, 0.3)}
N_PERM, ZTHR, ZCLIP, MIN_SESS = 200, 3.0, 12.0, 3


def area_curves(key, d):
    """{area_group: accuracy - null over time} for the left panels"""
    src = {k: (s_, w_) for k, _, s_, w_, _ in f005.p004.DEC + [f005.DEC_MOVE]}[key]
    n = len(d["t"])
    if src[0] == "pseudo":
        fs = sorted(glob.glob(str(H / src[1] / "raw_chunks" / "area_group__*__N200__*.parquet")))
        r = pd.concat([pd.read_parquet(f, columns=["area", "curve", "null_mean"]) for f in fs], ignore_index=True)
        return {a: np.mean([np.asarray(c)[:n] - np.asarray(m)[:n] for c, m in zip(g.curve, g.null_mean)], 0)
                for a, g in r.groupby("area")}
    tag = src[1].replace("whole_brain", "area_group")
    fs = sorted(glob.glob(str(H / "hitmiss" / "rows" / tag / "*.parquet")))
    r = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    r = r[(r.condition_type == "whole") & r.real_curve.notna()]
    out = {}
    for a, g in r.groupby("area_value"):
        if g.session_id.nunique() < MIN_SESS:
            continue
        L = []
        for _, x in g.iterrows():
            nul = x.shift_null_curves if x.shift_null_curves is not None and len(x.shift_null_curves) else x.null_curves
            L.append(np.asarray(x.real_curve, float)[:n] - np.stack(list(nul)).astype(float).mean(0)[:n])
        out[a] = np.nanmean(L, 0)
    return out


ZOOM_T = np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 3)        # detection / arrival zoom grid (20-ms bins, bin ends)


def area_curves_zoom(key, n_wide):
    """{area_group: accuracy - null on the zoom grid} (pseudo-population stimulus decoders only; user 2026-11-25: zoom
    on the first 50 ms)"""
    src = {k: (s_, w_) for k, _, s_, w_, _ in f005.p004.DEC + [f005.DEC_MOVE]}[key]
    if src[0] != "pseudo" or key == "move":
        return {}
    fs = sorted(glob.glob(str(H / src[1] / "raw_chunks" / "area_group__*__N200__*.parquet")))
    r = pd.concat([pd.read_parquet(f, columns=["area", "curve", "null_mean"]) for f in fs], ignore_index=True)
    z = slice(n_wide, n_wide + len(ZOOM_T))
    return {a: np.mean([np.asarray(c)[z] - np.asarray(m)[z] for c, m in zip(g.curve, g.null_mean)], 0)
            for a, g in r.groupby("area")}


def cluster_peak_z(d, w0, w1, K=100, seed=0):
    """per fine cluster: peak (25-ms smoothed) log2 mean enrichment in [w0, w1], z vs within-session label permutation;
    onset = first time z >= ZTHR for 2 bins (time-resolved z with the same permutations)"""
    df = d["df"]
    selw = (d["t"] >= w0) & (d["t"] <= w1)
    t = d["t"][selw]
    E = df[list(df.columns[:len(d["t"])])].to_numpy()[:, selw]
    lab0 = df.cluster.to_numpy().astype(int)
    n = len(df)
    cnt = np.bincount(lab0, minlength=K).astype(float)
    k5 = np.ones(5) / 5

    def means(lab):
        A = sp.csr_matrix((np.ones(n), (lab, np.arange(n))), shape=(K, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(np.asarray(A @ E) / cnt[:, None])

    obs = means(lab0)
    rng = np.random.default_rng(seed)
    groups = [np.where(df.session_id.to_numpy() == s)[0] for s in df.session_id.unique()]
    s1 = np.zeros_like(obs); s2 = np.zeros_like(obs); pk = []
    smooth = lambda M: np.apply_along_axis(lambda r: np.convolve(r, k5, mode="same"), 1, M)
    for _ in range(N_PERM):
        lab = lab0.copy()
        for g in groups:
            lab[g] = lab0[rng.permutation(g)]
        m = means(lab); s1 += m; s2 += m ** 2; pk.append(np.nanmax(smooth(m), 1))
    mu = s1 / N_PERM; sd = np.sqrt(np.maximum(s2 / N_PERM - mu ** 2, 1e-12))
    zt = (obs - mu) / sd
    pk = np.array(pk); peak_obs = np.nanmax(smooth(obs), 1)
    zpk = (peak_obs - pk.mean(0)) / pk.std(0)
    rows = []
    zs = smooth(np.nan_to_num(zt))
    for k in range(K):
        ok = zt[k] >= ZTHR
        on = next((t[i] for i in range(len(t) - 1) if ok[i] and ok[i + 1]), np.nan)
        # half-rise onset (user 2026-11-25: colour every dot): first time the smoothed weight-z curve reaches half its
        # peak, searching back from the peak; defined whenever the peak is positive
        ip = int(np.argmax(zs[k])); pk_k = zs[k, ip]
        half = np.nan
        if pk_k > 0:
            j = ip
            while j > 0 and zs[k, j - 1] >= pk_k / 2:
                j -= 1
            half = t[j]
        rows.append(dict(cluster=k, n=int(cnt[k]), peak_z=float(zpk[k]), onset_s=float(on), onset_half_s=float(half),
                         peak_t_s=float(t[ip])))
    ztime = pd.DataFrame([dict(cluster=k, t=float(tt), z=float(v), z_smooth=float(vs))
                          for k in range(K) for tt, v, vs in zip(t, zt[k], zs[k])])
    return pd.DataFrame(rows), ztime


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m050 = f005.m050
    drv = m050._drv()
    drv.set_conditions(drv.load_cfg(drv.CONFIG_PATH, **m050.cfg_of("main")))
    md = m050.method_dir("main")
    fam = f005.pms.metaclustering(md)["families"]
    neu = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    neu["cluster_id"] = neu["cluster_id"].astype(str)
    neu["metacluster"] = neu["cluster_label"].map(fam)
    dec = f005.load_decoders(neu)
    order_ref = allen_utils.get_area_group_custom_order()
    acol = allen_utils.get_custom_area_groups_colors()
    K = 100

    curves, dots, ztimes, zoom = {}, [], [], {}
    for key, _ in DEC:
        if key not in dec:
            continue
        curves[key] = area_curves(key, dec[key])
        zoom[key] = area_curves_zoom(key, len(dec[key]["t"]))
        z, zt_ = cluster_peak_z(dec[key], *PEAK_WIN[key], K=K)
        dots.append(z.assign(decoder=key)); ztimes.append(zt_.assign(decoder=key))
    pd.concat(ztimes, ignore_index=True).to_parquet(OUT / "option_iv_cluster_z_time.parquet", index=False)
    pd.DataFrame([dict(decoder=k, area_group=a, t=float(t), acc_minus_null=float(v))
                  for k, cs in zoom.items() for a, c in cs.items() for t, v in zip(ZOOM_T, c)]).to_csv(
        OUT / "option_iv_area_curves_zoom.csv", index=False)
    dots = pd.concat(dots, ignore_index=True)
    dots["metacluster"] = dots.cluster.map(fam)
    dots.to_csv(OUT / "option_iv_cluster_dots.csv", index=False)
    pd.DataFrame([dict(decoder=k, area_group=a, t=float(t), acc_minus_null=float(v))
                  for k, cs in curves.items() for a, c in cs.items() for t, v in zip(dec[k]["t"], c)]).to_csv(
        OUT / "option_iv_area_curves.csv", index=False)

    plt.rcParams.update({"font.size": 6, "font.family": "Arial", "axes.linewidth": 0.5, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "xtick.major.size": 2, "ytick.major.size": 2})
    nd = len(curves)
    side = 30.0                                                     # mm, square panels on the left
    fig_h = nd * side + (nd - 1) * 6 + 18
    fig = plt.figure(figsize=(183 / 25.4, fig_h / 25.4))
    mm = lambda x, y, w, h: fig.add_axes([x / 183, 1 - (y + h) / fig_h, w / 183, h / fig_h])
    groups_seen = []
    for i, key in enumerate(curves):
        ax = mm(12, 8 + i * (side + 6), side, side)
        lo, hi = WIN[key]
        t = dec[key]["t"]; sel = (t >= lo) & (t <= hi)
        for a in [g for g in order_ref if g in curves[key]] + [g for g in curves[key] if g not in order_ref]:
            ax.plot(t[sel] * 1000, curves[key][a][sel], color=acol.get(a, "#aaaaaa"), lw=0.6)
            if a not in groups_seen:
                groups_seen.append(a)
        ax.axhline(0, color="0.6", lw=0.4); ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_title(dict(DEC)[key], fontsize=6, pad=2)
        ax.set_xlabel("ms from movement onset" if key == "move" else "ms from stimulus", fontsize=5, labelpad=1)
        ax.set_ylabel("accuracy − null", fontsize=5, labelpad=1)
        ax.tick_params(labelsize=5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    lax = mm(12, 8 + nd * (side + 6) - 4, side + 10, 10); lax.axis("off")
    lax.legend(handles=[plt.Line2D([], [], color=acol.get(a, "#aaaaaa"), lw=1, label=a) for a in groups_seen],
               ncol=2, fontsize=3.8, frameon=False, loc="upper left", handlelength=0.9, columnspacing=0.6)

    # right: cluster axis (metacluster strip + boundaries) and dot columns
    top_y, h_ax = 8, fig_h - 26
    x0 = 62
    pal = f005.pms.family_palette(fam.values(), display_order=[fam[k] for k in range(K)])
    ax_s = mm(x0, top_y, 4, h_ax)
    runs = {}
    for k in range(K):
        ax_s.barh(k, 1, height=1, color=pal[fam[k]], edgecolor="none")
        a, b = runs.get(fam[k], (k, k)); runs[fam[k]] = (min(a, k), max(b, k))
    for f, (a, b) in runs.items():
        if b - a >= 2:
            ax_s.text(-0.3, (a + b) / 2, str(f), ha="right", va="center", fontsize=4.5)
    ax_s.set_ylim(K - 0.5, -0.5); ax_s.set_xlim(0, 1); ax_s.axis("off")
    ax_s.set_title("MC", fontsize=5, pad=2)
    cmap = plt.get_cmap("viridis")
    onset_norm = matplotlib.colors.Normalize(-0.1, 0.25)
    colw, gap = 18, 4
    for j, key in enumerate(curves):
        ax = mm(x0 + 6 + j * (colw + gap), top_y, colw, h_ax)
        d = dots[dots.decoder == key].set_index("cluster").reindex(range(K))
        z = d.peak_z.clip(0, ZCLIP).fillna(0).to_numpy()
        sig = d.peak_z.to_numpy() >= ZTHR
        size = 1 + 14 * (z / ZCLIP)
        y = np.arange(K)
        ax.scatter(np.zeros(K)[sig], y[sig], s=size[sig], c=cmap(onset_norm(d.onset_s.to_numpy()[sig])), lw=0)
        ax.scatter(np.zeros(K)[~sig], y[~sig], s=size[~sig], facecolor="none", edgecolor="0.75", lw=0.3)
        for f, (a, b) in runs.items():
            ax.axhline(b + 0.5, color="0.85", lw=0.3, zorder=0)
        ax.set_ylim(K - 0.5, -0.5); ax.set_xlim(-1, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(dict(DEC)[key].replace(" (", "\n("), fontsize=5, pad=2, linespacing=0.9)
    # keys
    kx = x0 + 6 + len(curves) * (colw + gap)
    cax = mm(kx, top_y + 2, 2.5, 30)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=onset_norm, cmap=cmap), cax=cax)
    cb.set_label("onset (s; first z >= 3)", fontsize=5); cb.ax.tick_params(labelsize=4.5)
    kax = mm(kx - 2, top_y + 40, 14, 30); kax.axis("off")
    for i, zz in enumerate([3, 6, 12]):
        kax.scatter(0.2, 1 - i * 0.3, s=1 + 14 * zz / ZCLIP, color="0.3", lw=0)
        kax.text(0.45, 1 - i * 0.3, f"z = {zz}", va="center", fontsize=4.5)
    kax.scatter(0.2, 1 - 3 * 0.3, s=4, facecolor="none", edgecolor="0.75", lw=0.3)
    kax.text(0.45, 1 - 3 * 0.3, "z < 3", va="center", fontsize=4.5)
    kax.set_xlim(0, 1.5); kax.set_ylim(-0.2, 1.2)
    fig.text(0.01, 0.995, "(iv) Left: decoding per area group (accuracy − null). Right: decoder weight per fine cluster on "
             "the rastermap axis (dot size = peak weight z vs label permutation, colour = onset)", fontsize=6, va="top")
    fig.savefig(OUT / "option_iv_areas_and_cluster_dots.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "option_iv_areas_and_cluster_dots.pdf", bbox_inches="tight")
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
