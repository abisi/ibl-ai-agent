"""SuppFigure3_areas_cohorts -- one plot_<key>(ax, df) per panel; reads only the panel CSV."""
import matplotlib as mpl
import matplotlib.colors as mcol
import numpy as np

from paperstyle import light

COH = {"R+": "#00B400", "R-": "#C800C8"}
FS = 5.0
ENR = mpl.colors.Normalize(-3, 3)


def _style(a):
    a.tick_params(labelsize=FS, length=1.5, pad=1, width=0.4)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_linewidth(0.4)


def _pfmt(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.2g}"


def _short(g):
    return (g.replace(" areas", "").replace("Somatosensory-", "SS-").replace("Lateral septal complex", "Lat. septum")
            .replace("Amygdala and hypothalamus", "Amygdala / hypoth.").replace("Cortical subplate", "Cort. subplate")
            .replace("Pons and medulla", "Pons / medulla"))


def _cohort_dots(ax, x, vals_p, vals_m, w=0.18, s=2.0, seed=0):
    r = np.random.default_rng(seed)
    for dx, v, c in ((-w, vals_p, "R+"), (w, vals_m, "R-")):
        v = np.asarray(v, float)
        ax.scatter(x + dx + r.uniform(-0.07, 0.07, len(v)), v, s=s, color=COH[c], lw=0, zorder=3, rasterized=True)
        ax.plot([x + dx - 0.12, x + dx + 0.12], [np.nanmean(v)] * 2, color="k", lw=0.6, zorder=4)


def plot_area_sampling(ax, df):
    groups = list(dict.fromkeys(df.area_group))
    for i, g in enumerate(groups):
        d = df[df.area_group == g]
        _cohort_dots(ax, i, d[d.cohort == "R+"].frac, d[d.cohort == "R-"].frac, seed=i)
        p1, p2 = d.mw_p.iloc[0], d.welch_p.iloc[0]
        if min(p1, p2) < 0.05:
            ax.text(i, d.frac.max() + 0.02, "*" if max(p1, p2) < 0.05 else "(*)", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(range(len(groups)), [_short(g) for g in groups], rotation=45, ha="right", rotation_mode="anchor")
    for t, g in zip(ax.get_xticklabels(), groups):
        t.set_color(df[df.area_group == g].color.iloc[0])
    ax.set_ylabel("Fraction of the mouse's neurons", fontsize=FS)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    for c, lab in (("R+", "R+"), ("R-", "R−")):
        ax.scatter([], [], s=6, color=COH[c], label=lab)
    ax.legend(loc="upper right", fontsize=FS, frameon=False, handletextpad=0.1, borderaxespad=0.1)
    ax.text(0.99, 0.60, "* both tests P < 0.05\n(*) one test", transform=ax.transAxes, ha="right", va="top", fontsize=FS,
            color="0.35")
    _style(ax)


def plot_ccf(ax, df):
    ax.axis("off")
    ml = (df.ccf_atlas_ml - 5700) / 1000          # mm from the midline (Allen CCF midline 5.7 mm)
    ap = df.ccf_atlas_ap / 1000
    dv = df.ccf_atlas_dv / 1000
    for j, c in enumerate(("R+", "R-")):
        m = (df.cohort == c).to_numpy()
        nm = df[m].mouse_id.nunique()
        top = ax.inset_axes([j * 0.52, 0.56, 0.46, 0.41])
        top.scatter(ml[m], ap[m], s=0.04, c=df.color[m], lw=0, rasterized=True)
        top.set_xlim(ml.min() - 0.2, ml.max() + 0.2); top.set_ylim(ap.max() + 0.2, ap.min() - 0.2)
        top.set_aspect("equal", adjustable="box")
        top.set_title(f"{'R+' if c == 'R+' else 'R−'} ({nm} mice)", fontsize=FS, color=COH[c], pad=1)
        side = ax.inset_axes([j * 0.52, 0.0, 0.46, 0.36])
        side.scatter(ap[m], dv[m], s=0.04, c=df.color[m], lw=0, rasterized=True)
        side.set_xlim(ap.min() - 0.2, ap.max() + 0.2); side.set_ylim(dv.max() + 0.2, dv.min() - 0.2)
        side.set_aspect("equal", adjustable="box")
        for a, (xl, yl) in ((top, ("ML (mm)", "AP (mm)")), (side, ("AP (mm)", "DV (mm)"))):
            _style(a)
            a.set_xlabel(xl, fontsize=FS, labelpad=0.5); a.set_ylabel(yl, fontsize=FS, labelpad=0.5)
            a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3)); a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))


def plot_mouse_yield(ax, df):
    ax.axis("off")
    cell = df[df.kind == "cell"]
    per = df[df.kind == "cluster"].sort_values("cluster")
    K = len(per)
    mice = (cell.drop_duplicates("mouse_id").sort_values(["cohort", "n_mouse"], ascending=[True, False]).mouse_id.to_list())
    M = cell.pivot(index="mouse_id", columns="cluster", values="frac").loc[mice, range(K)].to_numpy()
    h = ax.inset_axes([0.0, 0.40, 0.93, 0.52])
    norm = mcol.LogNorm(1e-3, 0.2)
    im = h.imshow(np.clip(M, 1e-3, None), aspect="auto", cmap="Greys", norm=norm, interpolation="nearest")
    h.set_xticks([]); h.set_yticks([])
    h.set_ylabel("Mice", fontsize=FS, labelpad=4)
    coh = cell.drop_duplicates("mouse_id").set_index("mouse_id").cohort.loc[mice]
    st = h.inset_axes([-0.012, 0, 0.01, 1])
    st.imshow(np.array([mcol.to_rgb(COH[c]) for c in coh])[:, None], aspect="auto", interpolation="nearest")
    st.axis("off")
    top = h.inset_axes([0, 1.02, 1, 0.05])
    top.imshow(np.array([mcol.to_rgb(c) for c in per.mc_color])[None], aspect="auto", interpolation="nearest")
    top.axis("off")
    cax = ax.inset_axes([0.945, 0.48, 0.012, 0.36])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_ticks([1e-3, 1e-2, 1e-1]); cb.set_ticklabels(["≤0.1%", "1%", "10%"])
    cb.ax.tick_params(labelsize=FS, length=1, pad=0.5, width=0.3); cb.outline.set_linewidth(0.3)
    cb.set_label("f_mk", fontsize=FS, labelpad=1)
    x = np.arange(K)
    t = ax.inset_axes([0.0, 0.235, 0.93, 0.11])           # R+ share of each cluster's neurons
    t.plot(x, per.rplus_share.to_numpy(), color="k", lw=0.6)
    t.axhline(per.rplus_share_all.iloc[0], color="k", lw=0.5, ls="--")
    t.set_xlim(-0.5, K - 0.5); t.set_ylim(0, 1); t.set_yticks([0, 0.5, 1]); t.set_xticks([])
    t.set_ylabel("R+ share", fontsize=FS, labelpad=1)
    _style(t)
    t.spines["bottom"].set_visible(False)
    b = ax.inset_axes([0.0, 0.0, 0.93, 0.20])             # mice contributing >= 3 neurons, per cohort
    b.bar(x, per.n_mice_ge3_rplus.to_numpy(), width=0.9, color=light(COH["R+"], 0.25), lw=0, label="R+ mice")
    b.bar(x, per.n_mice_ge3_rminus.to_numpy(), bottom=per.n_mice_ge3_rplus.to_numpy(), width=0.9,
          color=light(COH["R-"], 0.25), lw=0, label="R− mice")
    b.set_xlim(-0.5, K - 0.5)
    b.set_ylabel("Mice ≥3\nneurons", fontsize=FS, labelpad=1)
    b.set_xlabel("Cluster (rastermap order)", fontsize=FS, labelpad=1)
    b.set_xticks([0, 49, 99], ["1", "50", "100"])
    b.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    _style(b)
    h_, l_ = b.get_legend_handles_labels()
    ax.legend(h_, l_, loc="lower right", bbox_to_anchor=(0.93, 0.345), ncol=2, fontsize=FS, frameon=False,
              handlelength=0.8, borderaxespad=0.1, columnspacing=0.8)


def plot_dispersion(ax, df):
    ax.axis("off")
    for j, (m, lab) in enumerate((("shannon_entropy_norm", "Normalised entropy"), ("max_frac", "Largest cluster\nfraction"))):
        a = ax.inset_axes([j * 0.58, 0.0, 0.40, 0.80])
        _cohort_dots(a, 0, df[df.cohort == "R+"][m], df[df.cohort == "R-"][m], w=0.22, s=3, seed=j)
        a.set_xlim(-0.6, 0.6)
        a.set_xticks([-0.22, 0.22], ["R+", "R−"])
        for tl, c in zip(a.get_xticklabels(), ("R+", "R-")):
            tl.set_color(COH[c])
        a.set_ylabel(lab, fontsize=FS, labelpad=1)
        a.text(0.5, 1.03, f"MW {_pfmt(df[f'mw_p_{m}'].iloc[0])}" + "\n" + f"Welch {_pfmt(df[f'welch_p_{m}'].iloc[0])}",
               transform=a.transAxes, fontsize=FS, ha="center", va="bottom")
        a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        _style(a)


def plot_occupancy(ax, df):
    o = df[df.matrix == "observed"].sort_values("cluster")
    e = df[df.matrix == "expected"].sort_values("cluster")
    x = o.cluster.to_numpy()
    ax.bar(x, o["diff"] * 100, width=0.85, color=o.mc_color, lw=0)
    ax.plot(x, e["diff"] * 100, color="k", lw=0.6, label="expected from area sampling")
    s = o.sig_both.to_numpy()
    y = (o["diff"] * 100).to_numpy()
    ax.scatter(x[s], y[s] + np.sign(y[s]) * 0.35, s=3, color="k", lw=0, zorder=4, label="R+ ≠ R− (both tests, q < 0.05)")
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks([0, 49, 99], ["1", "50", "100"])
    ax.set_xlabel("Cluster (rastermap order)", fontsize=FS, labelpad=1)
    ax.set_ylabel("R+ − R− mean f_mk (%)", fontsize=FS, labelpad=1)
    r = df.iloc[0]
    ax.text(0.01, 0.98, f"PERMANOVA observed F = {r.permanova_F_observed:.2f}, {_pfmt(r.permanova_p_observed)}; expected "
            f"{_pfmt(r.permanova_p_expected)}; residual F = {r.permanova_F_residual:.2f}, {_pfmt(r.permanova_p_residual)}",
            transform=ax.transAxes, fontsize=FS, va="top")
    ax.legend(loc="lower left", bbox_to_anchor=(0.50, 0.0), fontsize=FS, frameon=False, handlelength=1,
              borderaxespad=0.1)
    _style(ax)


def plot_occupancy_scatter(ax, df):
    o = df[df.matrix == "observed"].sort_values("cluster")
    e = df[df.matrix == "expected"].sort_values("cluster")
    ax.scatter(e["diff"] * 100, o["diff"] * 100, s=4, c=o.mc_color, lw=0)
    lim = max(np.abs(o["diff"]).max(), np.abs(e["diff"]).max()) * 100 * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.4, ls="--")
    ax.axhline(0, color="0.85", lw=0.3); ax.axvline(0, color="0.85", lw=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Expected (%)", fontsize=FS, labelpad=1)
    ax.set_ylabel("Observed R+ − R− (%)", fontsize=FS, labelpad=1)
    r = df.iloc[0]
    ax.text(0.03, 0.97, f"r = {r.r_obs_vs_expected:.2f}\n{_pfmt(r.p_obs_vs_expected)}", transform=ax.transAxes,
            fontsize=FS, va="top")
    _style(ax)


def plot_area_enrichment(ax, df):
    ax.axis("off")
    groups = list(dict.fromkeys(df.area_group))
    ks = sorted(df.metacluster.unique())
    for j, c in enumerate(("R+", "R-")):
        d = df[df.cohort == c]
        M = d.pivot(index="area_group", columns="metacluster", values="log2_enrichment").loc[groups, ks].to_numpy()
        h = ax.inset_axes([j * 0.40, 0.0, 0.37, 0.88])
        im = h.imshow(M, cmap="RdBu_r", norm=ENR, aspect="auto", interpolation="nearest")
        h.set_xticks(range(len(ks)), [str(k) for k in ks], fontsize=FS)
        h.set_yticks(range(len(groups)), [_short(g) for g in groups] if j == 0 else [], fontsize=FS)
        if j == 0:
            for tl, g in zip(h.get_yticklabels(), groups):
                tl.set_color(df[df.area_group == g].color.iloc[0])
        h.tick_params(length=0, pad=1)
        h.set_xlabel("Metacluster", fontsize=FS, labelpad=1)
        h.set_title("R+" if c == "R+" else "R−", fontsize=FS + 0.5, color=COH[c], pad=5)
        st = h.inset_axes([0, 1.01, 1, 0.045])
        cols = d.drop_duplicates("metacluster").set_index("metacluster").mc_color.loc[ks]
        st.imshow(np.array([mcol.to_rgb(x) for x in cols])[None], aspect="auto", interpolation="nearest")
        st.axis("off")
        for s in h.spines.values():
            s.set_visible(False)
    q = df.drop_duplicates("area_group").set_index("area_group").loc[groups]
    b = ax.inset_axes([0.83, 0.0, 0.12, 0.88])
    y = np.arange(len(groups))
    b.barh(y, q.js, height=0.7, color=["k" if p < 0.05 else "0.7" for p in q.p_perm], lw=0)
    b.scatter(q.null_p95, y, marker="|", s=8, color="#e8222e", lw=0.6)
    b.set_ylim(len(groups) - 0.5, -0.5); b.set_yticks([])
    b.set_xlabel("R+ vs R− JS\ndivergence (bits)", fontsize=FS, labelpad=1)
    b.xaxis.set_major_locator(mpl.ticker.MaxNLocator(2))
    _style(b)
    b.spines["left"].set_visible(False)
    b.text(1.0, 1.02, "black, P < 0.05\nred tick, perm. 95th pct.", transform=b.transAxes, fontsize=FS, ha="right",
           va="bottom", color="0.3")
    cax = ax.inset_axes([0.775, 0.25, 0.012, 0.5])
    cb = ax.figure.colorbar(im, cax=cax, ticks=[-3, 0, 3])
    cb.ax.tick_params(labelsize=FS, length=1, pad=0.5, width=0.3); cb.outline.set_linewidth(0.3)
    cb.set_label("log2 enrich.", fontsize=FS, labelpad=1)


def plot_specialization(ax, df):
    groups = list(dict.fromkeys(df.area_group))
    x = np.arange(len(groups))
    for k, (lvl, a_) in enumerate((("fine clusters", 1.0), ("metaclusters", 0.45))):
        d = df[df.level == lvl].set_index("area_group").loc[groups]
        xx = x + (k - 0.5) * 0.4
        cols = [light(c, 0.0 if k == 0 else 0.55) for c in d.color]
        ax.bar(xx, d.kl_bits, width=0.38, color=cols, lw=0, label="fine clusters" if k == 0 else "metaclusters")
        for xi, v in zip(xx, d.null_p95):
            ax.plot([xi - 0.17, xi + 0.17], [v, v], color="#e8222e", lw=0.6)
    ax.set_xticks(x, [_short(g) for g in groups], rotation=45, ha="right", rotation_mode="anchor")
    for tl, g in zip(ax.get_xticklabels(), groups):
        tl.set_color(df[df.area_group == g].color.iloc[0])
    ax.set_ylabel("Specialization (KL, bits)", fontsize=FS, labelpad=1)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    import matplotlib.patches as mp
    import matplotlib.lines as ml
    ax.legend(handles=[mp.Patch(color="0.25", label="fine clusters"), mp.Patch(color="0.7", label="metaclusters"),
                       ml.Line2D([], [], color="#e8222e", lw=0.8, label="permutation 95th pct.")],
              loc="upper right", fontsize=FS, frameon=False, handlelength=0.9, borderaxespad=0.1)
    _style(ax)


def plot_fine_areas(ax, df):
    d = df[df.level == "metaclusters"].sort_values(["area_group", "kl_bits"], ascending=[True, False])
    x = np.arange(len(d))
    ax.bar(x, d.kl_bits, width=0.75, color=d.color, lw=0)
    for xi, v in zip(x, d.null_p95):
        ax.plot([xi - 0.35, xi + 0.35], [v, v], color="#e8222e", lw=0.5)
    st = ax.inset_axes([0, 1.02, 1, 0.06])
    st.scatter(x, np.zeros(len(x)), marker="s", s=5, c=d.top_mc_color, lw=0)
    st.set_xlim(-0.6, len(x) - 0.4); st.axis("off")
    st.text(-0.8, 0, "top MC", fontsize=FS, ha="right", va="center")
    ym = d.kl_bits.max()
    ax.set_xticks(x, d.area, rotation=90, fontsize=FS)
    for tl, c in zip(ax.get_xticklabels(), d.color):
        tl.set_color(c)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, ym * 1.05)
    ax.set_ylabel("Specialization over\nmetaclusters (KL, bits)", fontsize=FS, labelpad=1)
    _style(ax)
