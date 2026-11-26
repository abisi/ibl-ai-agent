"""Figure 3 plot functions (read only the panel CSVs)."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DEC = [("wdet", "Whisker detection"), ("adet", "Auditory detection"), ("wva", "Whisker vs auditory"),
       ("hm", "Hit vs miss"), ("move", "Movement")]
EVENT = {"wdet": "stimulus", "adet": "stimulus", "wva": "stimulus", "hm": "stimulus", "move": "jaw onset"}
# movement = false alarms vs correct rejections at jaw-opening onset (jaw run, 2026-11-26)
XWIN = {"wdet": (-0.01, 0.05), "adet": (-0.01, 0.05), "wva": (-0.01, 0.05),      # user: zoom on the first 50 ms
        "hm": (-0.05, 0.3), "move": (-0.5, 0.3)}
XTICKS = {"wdet": [0, 25, 50], "adet": [0, 25, 50], "wva": [0, 25, 50], "hm": [0, 150, 300], "move": [-400, 0, 200]}
ZTHR, ZCLIP = 3.0, 12.0
REL = matplotlib.colors.Normalize(-0.05, 0.2)          # dot colour: weight rise time relative to the decoder's onset
# user 2026-11-26: cmasher flamingo, reversed (fast = light pink, late = dark), near-white end cut (no white on white)
import cmasher                                           # noqa: E402
CMAP = cmasher.get_sub_cmap(cmasher.flamingo_r, 0.20, 0.92)
HEAT = plt.get_cmap("Greys")
TICK = "#C8102E"                                        # area onset tick, crimson (user 2026-11-26)


def _box(ax):
    fig = ax.figure
    fw, fh = fig.get_size_inches() * 25.4
    p = ax.get_position()
    ax.axis("off")
    return fig, fw, fh, p.x0 * fw, (1 - p.y1) * fh, p.width * fw, p.height * fh


def _mm_axes(fig, x_mm, y_mm, w_mm, h_mm):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x_mm / fw, 1 - (y_mm + h_mm) / fh, w_mm / fw, h_mm / fh])


def _style(a):
    a.tick_params(labelsize=5.0, length=1.2, pad=0.8, width=0.35)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_linewidth(0.35)


def _size(z):
    return 0.8 + 9 * (np.clip(z, 0, ZCLIP) / ZCLIP)


def plot_area_decoding(ax, df):
    """Panel C: one row per decoder. Left, accuracy - null over time per area group (area colours as panel A); right,
    separated, the same as an area x time heatmap with the areas sorted by onset (tick = first time the area reaches half
    its maximum). Stimulus decoders zoomed on the first 50 ms (2-ms grid, 20-ms bins)."""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    decs = [d for d in DEC if d[0] in set(df.decoder)]
    nd = len(decs)
    gap, lab, sep = 7.0, 8.0, 7.0                           # row gap, y-label room, gap lines -> heatmap (user: separate)
    side = (H - (nd - 1) * gap - 12.0) / nd
    hw = W - lab - side - sep - 1.0                         # C box narrowed instead (user: heatmaps less wide)
    vmax = 0.5
    for i, (key, title) in enumerate(decs):
        y = Y0 + i * (side + gap)
        lo, hi = XWIN[key]
        d = df[(df.decoder == key) & df.t.between(lo, hi)]
        a = _mm_axes(fig, X0 + lab, y, side, side)
        for _, g in sorted(d.groupby("area_group"), key=lambda x: x[1].area_rank.iloc[0]):
            a.plot(g.t * 1000, g.acc_minus_null, color=g.area_color.iloc[0], lw=0.5)
        a.axhline(0, color="0.6", lw=0.3); a.axvline(0, color="k", lw=0.3, ls=":")
        a.set_title(title, fontsize=5.0, pad=1.5, loc="left")
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        a.set_xticks(XTICKS[key]); a.set_xlim(lo * 1000, hi * 1000)
        _style(a)
        if i == nd // 2:
            a.set_ylabel("Accuracy − chance (shift null)", fontsize=5, labelpad=1)
        P = d.pivot_table(index="area_group", columns="t", values="acc_minus_null")
        # onset = first significant bin relative to the pre-event level (user 2026-11-26; compute.py _sig_onsets_baseline)
        so = d.drop_duplicates("area_group").set_index("area_group").sig_onset_s
        on = {ar: (so.get(ar) if np.isfinite(so.get(ar, np.nan)) else np.nan) for ar in P.index}
        order = sorted(P.index, key=lambda ar: (np.isnan(on[ar]), on[ar] if np.isfinite(on[ar]) else 0))
        h = _mm_axes(fig, X0 + lab + side + sep, y, hw, side)
        tt = P.columns.to_numpy() * 1000
        im = h.imshow(P.loc[order].to_numpy(), aspect="auto", cmap=HEAT, vmin=0, vmax=vmax,
                      extent=[tt[0], tt[-1], len(order) - 0.5, -0.5], interpolation="none")
        span = tt[-1] - tt[0]
        for r_, ar in enumerate(order):
            if np.isfinite(on[ar]):
                h.plot([on[ar] * 1000] * 2, [r_ - 0.45, r_ + 0.45], color=TICK, lw=0.8)
            h.add_patch(plt.Rectangle((tt[0] - span * 0.05, r_ - 0.5), span * 0.035, 1,
                                      color=d[d.area_group == ar].area_color.iloc[0], clip_on=False, lw=0))
        h.axvline(0, color="k", lw=0.3, ls=":")
        h.set_yticks([]); h.set_xticks(XTICKS[key]); h.set_xlim(lo * 1000, hi * 1000)
        _style(h)
        if i == 0:
            h.set_title("Area groups sorted by onset", fontsize=5.0, pad=1.5, loc="right")
        h.text(1.0, -0.18, f"ms from {EVENT[key]}", transform=h.transAxes, fontsize=5.0, ha="right", va="top",
               color="0.3")
    # keys under the heatmaps: grey = accuracy - chance; tick = area onset
    ky = Y0 + nd * (side + gap) - gap + 4.0
    cax = _mm_axes(fig, X0 + lab + side + sep, ky, min(hw * 0.4, 12.0), 1.2)   # user: smaller colour bar
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
    cb.set_ticks([0, 0.25, 0.5]); cb.ax.tick_params(labelsize=5.0, length=1.2, pad=0.8, width=0.35)
    cb.outline.set_linewidth(0.35); cb.set_label("Accuracy − chance", fontsize=5.0, labelpad=1)
    tx = X0 + lab + side + sep + min(hw * 0.4, 12.0) + 4.0
    fig.add_artist(plt.Line2D([tx / fw, (tx + 2.5) / fw], [1 - (ky + 0.8) / fh] * 2, color=TICK, lw=0.9))
    fig.text((tx + 3.5) / fw, 1 - (ky + 0.8) / fh, "area onset (first bin above baseline)", fontsize=5.0, va="center")


def plot_weight_example(ax, df):
    """Panel D (user 2026-11-26): how a dot of E is built, on one example cluster (strongest whisker-detection cluster of
    MC 12). Left, the cluster's weight (|w| relative to the average neuron, log2, 25-ms smoothed) against same-size
    'clusters' drawn by shuffling cluster labels within session; middle, its peak against the shuffled peaks -> z (dot
    size); right, its mean response averaged in the decoder's own causal 20-ms bins above its weight z on the same grid
    (the weight rises with the response; half-rise -> dot colour)."""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    X0 = X0 + 4.0; W = W - 4.0                              # room for the panel letter
    r = df.iloc[0]
    col = r.mc_color
    ttl = dict(fontsize=5.0, pad=1.2, loc="left")
    lab = dict(fontsize=5.0, labelpad=0.6)
    top, axh = Y0 + 4.2, H - 10.5
    # 1. cluster vs shuffled clusters
    a = _mm_axes(fig, X0 + 4.5, top, 25.0, axh)
    nc = df[df.part == "null_curve"]
    for _, g in nc.groupby("series"):
        a.plot(g.t * 1000, g.y, color="0.75", lw=0.25)
    o = df[df.part == "obs_curve"]
    a.plot(o.t * 1000, o.y, color=col, lw=1.0)
    a.axhline(0, color="k", lw=0.3, ls=":")
    a.set_xticks([0, 150, 300]); a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    a.set_xlabel("ms from stimulus", **lab); a.set_ylabel("log2 weight", **lab)
    a.set_title(f"Cluster ({int(r.n_neurons)} neurons) vs\nshuffled same-size clusters", **ttl)
    _style(a)
    # 2. peak vs shuffled peaks -> z
    b = _mm_axes(fig, X0 + 36.0, top, 18.0, axh)
    pk = df[df.part == "null_peak"].y.to_numpy()
    b.hist(pk, bins=25, color="0.7", lw=0)
    b.axvline(r.peak_obs, color=col, lw=1.0)
    b.set_xlabel("peak log2 weight", **lab); b.set_ylabel("shuffles", **lab)
    b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2)); b.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    b.set_title(f"Peak vs shuffled peaks\nz = {r.peak_z:.0f} → dot size", **ttl)
    _style(b)
    # 3. response (decoder-matched bins) above weight z
    x3, w3 = X0 + 62.0, W - 63.0
    hh = (axh - 1.5) / 2
    c1 = _mm_axes(fig, x3, top, w3, hh)
    for name, cc in (("Whisker hit", "#f7b519"), ("Whisker miss", "#c8a27a")):
        g = df[(df.part == "psth_raw") & (df.series == name)]
        g = g[g.t.between(-0.02, 0.1)]
        c1.plot(g.t * 1000, g.y, color=cc, lw=0.35, alpha=0.6)
        g = df[(df.part == "psth_dec") & (df.series == name)]
        c1.plot(g.t * 1000, g.y, color=cc, lw=0.9, label=name.replace("Whisker ", "W "))
    c1.set_xlim(-20, 100); c1.set_xticks([0, 50, 100]); c1.set_xticklabels([])
    c1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    c1.set_ylabel("Δ Hz", **lab)
    c1.legend(frameon=False, fontsize=5.0, loc="upper right", handlelength=0.8, borderaxespad=0, labelspacing=0.2)
    c1.set_title("Response in the decoder's 20-ms bins (thin: 5 ms)\nand weight z on the same grid", **ttl)
    _style(c1)
    c2 = _mm_axes(fig, x3, top + hh + 1.5, w3, hh)
    zz = df[df.part == "z_zoom"]
    zs = zz.y.to_numpy(); tz = zz.t.to_numpy() * 1000
    c2.plot(tz, zs, color=col, lw=0.9)
    c2.axhline(0, color="0.8", lw=0.3)
    ip = int(np.argmax(zs)); j = ip
    while j > 0 and zs[j - 1] >= zs[ip] / 2:
        j -= 1
    c2.scatter([tz[j]], [zs[ip] / 2], s=6, color="k", zorder=4, lw=0)
    c2.annotate("half rise → colour", (tz[j], zs[ip] / 2), xytext=(4, -1), textcoords="offset points", fontsize=5.0,
                va="center")
    c2.set_xlim(-20, 100); c2.set_xticks([0, 50, 100]); c2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    c2.set_xlabel("ms from stimulus", **lab); c2.set_ylabel("weight z", **lab)
    _style(c2)


def _dot_key(fig, fw, fh, x, y):
    """one-line key of the E dots: size = peak weight z, colour = half-rise time from the decoder onset, open = z < 3"""
    fig.text(x / fw, 1 - (y + 1.0) / fh, "Dot size = peak weight z", fontsize=5.0, va="center")
    ka = _mm_axes(fig, x + 21.0, y, 16.0, 2.0); ka.axis("off")
    for i, zz in enumerate([3, 6, 12]):
        ka.scatter(0.1 + i * 0.36, 0.5, s=_size(zz), color="0.3", lw=0)
        ka.text(0.1 + i * 0.36 + 0.07, 0.5, str(zz), fontsize=5.0, va="center")
    ka.set_xlim(0, 1); ka.set_ylim(0, 1)
    fig.text((x + 39.0) / fw, 1 - (y + 1.0) / fh, "open: z < 3", fontsize=5.0, va="center")
    fig.text((x + 53.0) / fw, 1 - (y + 1.0) / fh, "Colour = weight rise (ms from decoder onset)", fontsize=5.0,
             va="center")
    cax = _mm_axes(fig, x + 53.5, y + 2.6, 22.0, 1.1)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=REL, cmap=CMAP), cax=cax, orientation="horizontal",
                      extend="both")
    cb.set_ticks([0, 0.1, 0.2]); cb.set_ticklabels(["0", "100", "200"])
    cb.ax.tick_params(labelsize=5.0, length=1.0, pad=0.5, width=0.3); cb.outline.set_linewidth(0.3)


def plot_cluster_weights(ax, df):
    """Panel E: the rastermap cluster axis (metacluster strip as panel A) and one dot column per decoder (see D)."""
    fig, fw, fh, X0, Y0, W, H = _box(ax)
    decs = [d for d in DEC if d[0] in set(df.decoder)]
    nd = len(decs)
    K = int(df.cluster.max()) + 1
    strip_w, mc_lab = 2.2, 4.0
    top, hh = Y0 + 9.5, H - 9.5
    _dot_key(fig, fw, fh, X0 + 6.0, Y0 + 0.4)            # user 2026-11-26: compact dot key above E
    colw = (W - mc_lab - strip_w - 1.0) / nd
    mcs = df.drop_duplicates("cluster").set_index("cluster").reindex(range(K))
    sa = _mm_axes(fig, X0 + mc_lab, top, strip_w, hh)
    runs = {}
    for k in range(K):
        sa.barh(k, 1, height=1, color=mcs.mc_color[k], edgecolor="none")
        f = mcs.metacluster[k]
        a_, b_ = runs.get(f, (k, k)); runs[f] = (min(a_, k), max(b_, k))
    for f, (a_, b_) in runs.items():
        if b_ - a_ >= 2:
            inb = bool(mcs[mcs.metacluster == f].in_b.iloc[0]) if "in_b" in mcs else False
            sa.text(-0.4, (a_ + b_) / 2, str(int(f)), ha="right", va="center", fontsize=5.5 if inb else 5.0,
                    fontweight="bold" if inb else "normal", color="k" if inb else "0.45")   # bold = shown in B
    sa.set_ylim(K - 0.5, -0.5); sa.set_xlim(0, 1); sa.axis("off")
    sa.text(0.5, 1.0, "MC", transform=sa.transAxes, ha="center", va="bottom", fontsize=5)
    for j, (key, title) in enumerate(decs):
        a = _mm_axes(fig, X0 + mc_lab + strip_w + 1.0 + j * colw, top, colw, hh)
        d = df[df.decoder == key].set_index("cluster").reindex(range(K))
        z = d.peak_z.fillna(0).to_numpy()
        sig = d.peak_z.to_numpy() >= ZTHR
        size = _size(z)
        yy = np.arange(K)
        col = CMAP(REL(np.nan_to_num(d.onset_rel_s.to_numpy(), nan=REL.vmax)))
        a.scatter(np.zeros(K)[sig], yy[sig], s=size[sig], c=col[sig], lw=0)
        a.scatter(np.zeros(K)[~sig], yy[~sig], s=np.maximum(size[~sig], 1.5), facecolor="none", edgecolor=col[~sig],
                  lw=0.35)
        for f, (a_, b_) in runs.items():
            a.axhline(b_ + 0.5, color="0.88", lw=0.25, zorder=0)
        a.set_ylim(K - 0.5, -0.5); a.set_xlim(-1, 1); a.axis("off")
        a.text(0.5, 1.0, title.replace(" ", "\n", 1), transform=a.transAxes, ha="center", va="bottom", fontsize=5.0,
               linespacing=0.95)
