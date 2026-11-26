"""SuppFigure3_robustness plot functions (read only the panel CSVs)."""
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

FS = 5.0
CMAP_M = "bwr"


def _box(ax):
    fig = ax.figure
    fw, fh = fig.get_size_inches() * 25.4
    p = ax.get_position()
    ax.axis("off")
    return fig, fw, fh, p.x0 * fw, (1 - p.y1) * fh, p.width * fw, p.height * fh


def _mm(fig, x, y, w, h):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x / fw, 1 - (y + h) / fh, w / fw, h / fh])


def _style(a):
    a.tick_params(labelsize=FS, length=1.2, pad=0.8, width=0.35)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_linewidth(0.35)


def _bins(df):
    cols = [c for c in df.columns if re.fullmatch(r"c\d+_b\d+", c)]
    cond = np.array([int(c.split("_")[0][1:]) for c in cols])
    bounds = np.where(np.diff(cond) != 0)[0] + 0.5
    return cols, bounds


def _matrix(a, M, bounds, vmax):
    a.imshow(M, aspect="auto", cmap=CMAP_M, vmin=-vmax, vmax=vmax, interpolation="none")
    for b in bounds:
        a.axvline(b, color="k", lw=0.25)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.35)


def plot_param_grid(ax, df):
    """a: the reference fit's 100 cluster-mean PSTHs drawn in each fit's order (rows = locality, columns = time-lag
    window); reference setting boxed"""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    cols, bounds = _bins(df)
    M = df[cols].to_numpy()
    vmax = np.nanpercentile(np.abs(M), 97)
    pc = [c for c in df.columns if c.startswith("pos_")]
    loc = sorted({float(re.search(r"_l([\d.]+)_", c).group(1)) for c in pc})
    tlw = sorted({int(c.split("_t")[-1]) for c in pc})
    gx, gy = 1.5, 2.0
    w = (W - (len(tlw) - 1) * gx) / len(tlw)
    h = (H - 3.0 - (len(loc) - 1) * gy) / len(loc)
    for i, l in enumerate(loc):
        for j, t in enumerate(tlw):
            pos = df[f"pos_l{l:.2f}_t{t}"].to_numpy()
            a = _mm(fig, X0 + j * (w + gx), Y0 + 3.0 + i * (h + gy), w, h)
            _matrix(a, M[np.argsort(pos)], bounds, vmax)
            if (l, t) == (0.25, 20):
                for s in a.spines.values():
                    s.set_linewidth(1.2); s.set_color("#C8102E")
            if i == 0:
                a.set_title(f"time-lag window {t}", fontsize=FS, pad=1.5)
            if j == 0:
                a.set_ylabel(f"locality {l:g}", fontsize=FS, labelpad=1.5)
    cax = _mm(fig, X0 + W - 14.0, Y0 + H + 1.2, 12.0, 1.1)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-vmax, vmax), cmap=CMAP_M), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax]); cb.set_ticklabels([f"{-vmax:.0f}", "0", f"{vmax:.0f}"])
    cb.ax.tick_params(labelsize=FS, length=1.0, pad=0.5, width=0.3); cb.outline.set_linewidth(0.3)
    fig.text((X0 + W - 15.0) / fw, 1 - (Y0 + H + 1.75) / fh, "Δ rate (spks/s)", fontsize=FS, ha="right", va="center")
    fig.text(X0 / fw, 1 - (Y0 + H + 1.75) / fh, "rows: the 100 clusters (same in every fit) in each fit's order; "
             "columns: conditions; red box: setting used (0.25 / 20)", fontsize=FS, va="center", color="0.3")


def _heat(a, P, title, vmin, vmax, loc, tlw, fmt="{:.2f}"):
    a.imshow(P, cmap="Greys", vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P[i, j]
            a.text(j, i, fmt.format(v), ha="center", va="center", fontsize=FS - 0.2,
                   color="w" if (v - vmin) / (vmax - vmin) > 0.6 else "k")
    a.set_xticks(range(len(tlw))); a.set_xticklabels([str(t) for t in tlw])
    a.set_yticks(range(len(loc))); a.set_yticklabels([f"{l:g}" for l in loc])
    i0, j0 = loc.index(0.25), tlw.index(20)
    a.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec="#C8102E", lw=1.0))
    a.set_title(title, fontsize=FS, pad=1.5)
    a.set_xlabel("time-lag window", fontsize=FS, labelpad=0.8); a.set_ylabel("locality", fontsize=FS, labelpad=0.8)
    a.tick_params(labelsize=FS, length=1.0, pad=0.6, width=0.3)


def plot_grid_summary(ax, df):
    """b: order RSA vs the reference fit (top) and vs the clusters' PSTH dissimilarity (bottom); clusters identical"""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    loc = sorted(df.locality.unique().tolist())
    tlw = sorted(df.time_lag_window.unique().tolist())
    piv = lambda c: df.pivot(index="locality", columns="time_lag_window", values=c).loc[loc, tlw].to_numpy()
    h = (H - 24.0) / 2
    a1 = _mm(fig, X0, Y0 + 3.0, W, h)
    _heat(a1, piv("rsa_order_vs_reference"), "Order vs setting used (RSA)", 0, 1, loc, tlw)
    a2 = _mm(fig, X0, Y0 + 3.0 + h + 9.0, W, h)
    null = df.rsa_order_vs_psth_null_mean.mean() + 2 * df.rsa_order_vs_psth_null_sd.mean()
    _heat(a2, piv("rsa_order_vs_psth"), "Order vs PSTH similarity (RSA)", 0, 1, loc, tlw)
    fig.text(X0 / fw, 1 - (Y0 + 3.0 + 2 * h + 9.0 + 6.5) / fh,
             f"Clusters identical in all {len(df)} fits\n(matched r = {df.matched_r_min.min():.2f}).\nRandom orders: "
             f"RSA {df.rsa_order_vs_psth_null_mean.mean():.2f} ± {df.rsa_order_vs_psth_null_sd.mean():.2f}",
             fontsize=FS, va="top", color="0.3", linespacing=1.0)


def plot_variants(ax, df):
    """c: per variant, matched r of the 100 main clusters (dots + median) and the order RSA (bars)"""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    order = list(dict.fromkeys(df.sort_values("rsa_order", ascending=False).variant))
    lab = df.drop_duplicates("variant").set_index("variant")
    w1 = W * 0.6
    a = _mm(fig, X0, Y0, w1, H)
    rng = np.random.default_rng(0)
    for i, v in enumerate(order):
        r = df[df.variant == v].r.to_numpy()
        a.scatter(i + rng.uniform(-0.25, 0.25, len(r)), r, s=1.2, color="0.55", lw=0)
        a.plot([i - 0.3, i + 0.3], [np.median(r)] * 2, color="k", lw=0.9)
        a.text(i, 1.012, f"{np.median(r):.2f}", ha="center", va="bottom", fontsize=FS)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([lab.label[v] for v in order], rotation=30, ha="right", fontsize=FS)
    a.set_ylim(min(0.4, df.r.min() - 0.02), 1.0)
    a.set_ylabel("matched cluster r", fontsize=FS, labelpad=1)
    a.text(-0.6, 1.012, "median", ha="right", va="bottom", fontsize=FS, color="0.3")
    _style(a)
    b = _mm(fig, X0 + w1 + 12.0, Y0, W - w1 - 12.0, H)
    rs = [lab.rsa_order[v] for v in order]
    b.bar(range(len(order)), rs, color="0.45", width=0.65)
    for i, v in enumerate(rs):
        b.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=FS)
    b.set_xticks(range(len(order)))
    b.set_xticklabels([lab.label[v] for v in order], rotation=30, ha="right", fontsize=FS)
    b.set_ylim(0, 1); b.set_ylabel("order RSA vs main", fontsize=FS, labelpad=1)
    _style(b)


def plot_gmm_matrices(ax, df):
    """d: rastermap clusters / metaclusters vs matched GMM components (K = 100 and K = n metaclusters)"""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    cols, bounds = _bins(df)
    vmax = np.nanpercentile(np.abs(df[df.K == 100][cols].to_numpy()), 97)
    Ks = sorted(df.K.unique(), reverse=True)
    gx = 2.0
    w = (W - 3 * gx - 3.0) / 4
    x = X0
    for K in Ks:
        lvl = "cluster" if K == 100 else "metacluster"
        for blk, title in ((f"rastermap_{lvl}", f"Rastermap, {K} {lvl}s"), (f"gmm_{lvl}", f"GMM, K = {K} (matched order)")):
            d = df[(df.K == K) & (df.block == blk)].sort_values("row")
            a = _mm(fig, x, Y0 + 3.0, w, H - 3.0)
            _matrix(a, d[cols].to_numpy(), bounds, vmax)
            a.set_title(title, fontsize=FS, pad=1.5)
            if blk.startswith("gmm"):
                n_un = int(d.matched_r.isna().sum())
                if n_un:
                    a.axhline(len(d) - n_un - 0.5, color="k", lw=0.5, ls=":")
            x += w + gx
        x += 3.0 - gx + gx


def plot_gmm_match(ax, df):
    """e: matched r vs rastermap cluster size, K = 100 (grey) and K = n metaclusters (black)"""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    a = _mm(fig, X0, Y0 + 3.0, W, H - 3.0)
    for K, col, s in ((100, "0.6", 4), (int(df.K.min()), "k", 12)):
        d = df[df.K == K]
        lvl = "clusters" if K == 100 else "metaclusters"
        a.scatter(d.n_rastermap, d.r, s=s, color=col, lw=0,
                  label=f"{len(d)} {lvl} (K = {K}): median r {d.r.median():.2f}")
    a.set_xscale("log")
    a.set_ylim(min(0.0, df.r.min() - 0.05), 1.02)
    a.set_xlabel("neurons in the rastermap cluster", fontsize=FS, labelpad=1)
    a.set_ylabel("r with matched GMM component", fontsize=FS, labelpad=1)
    a.legend(frameon=False, fontsize=FS, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    _style(a)
