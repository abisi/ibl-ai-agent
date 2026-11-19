"""Publication figures for the reward-free R+ vs R- MMD^2 analysis (one variant), in two parts.

Part 1  data & method (183 x 175 mm)
  a  reward-free windows per PSTH type      b  pipeline schematic                c  goals, metrics, criteria
  d  cohort densities (PC1-PC2)             e  mean kernel per mouse pair (all)  f  MMD^2 decomposition (all neurons)
  g  mouse-level permutation null           h  witness in PC space               i  mouse-mean witness per cohort
     (example PSTH type, --example)
Part 2  results (183 x 235 mm)
  a  MMD^2 z per PSTH type (both halves)    b  per-PSTH-type statistics table
  c  mouse-mean witness per PSTH type and cohort, with mouse-level Mann-Whitney U and Welch p-values
  d  z per area group x PSTH type; * p<0.05 both halves, ** BH-FDR q<0.05 across area x PSTH tests
  e  number of significant area groups per PSTH type (uncorrected vs BH-FDR)
  f..  mouse-level mean PSTHs per cohort for the strongest area x PSTH cells (+ time-resolved z if 019 was run)
Correction family: active PSTH types + passive post (Whisker/Auditory post, Whisker miss/hit, Auditory hit, Spont.
lick, Whisker/Auditory hit (lick)); p_max = max(p_odd->even, p_even->odd) enters BH-FDR.
Outputs -> <method_dir>/<run-dir>/pub_overview/{mmd_pub_part1_method, mmd_pub_part2_results}.{png,pdf,svg},
           per_condition_stats.csv, per_area_family_fdr.csv
"""
import argparse
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/rastermap_psth"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import cluster_mmd_analysis_new as mmd                                  # noqa: E402
d017 = importlib.import_module("017_mmd_method_figures")
d016 = d017.d016

RP, RM = d017.RP, d017.RM
CONDS = ["Whisker pre", "Whisker post", "Auditory pre", "Auditory post", "Whisker miss", "Whisker hit", "Auditory hit",
         "Spont. lick", "Whisker hit (lick)", "Auditory hit (lick)"]
FAMILY = ["Whisker post", "Auditory post", "Whisker miss", "Whisker hit", "Auditory hit", "Spont. lick",
          "Whisker hit (lick)", "Auditory hit (lick)"]
ALPHA, FS = 0.05, 6.5
DIRS = ("fit_odd_test_even", "fit_even_test_odd")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def letter(fig, ax, s, dx_mm=9.0, dy_mm=5.0):
    """panel letter in FIGURE coordinates, dx_mm left of / dy_mm above the axes box (never off-canvas)"""
    pos = ax.get_position()
    W, H = fig.get_size_inches() * 25.4
    fig.text(max(0.004, pos.x0 - dx_mm / W), min(0.995, pos.y1 + dy_mm / H), s, fontsize=FS + 3, ha="left", va="bottom")


def setup_style():
    import rastermap_psth.population_matrix_summary as pms
    pms.use_pub_font()
    plt.rcParams.update({"font.size": FS, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                         "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5})
    return pms


def mouse_witness(wn, cond):
    we = wn[wn.feature_version == cond]
    mm = we.groupby(["mouse_id", "cohort"]).witness_mean_dirs.mean().reset_index()
    mm["w"] = mm.witness_mean_dirs - we.witness_mean_dirs.mean()
    a_, b_ = mm[mm.cohort == "R+"].w, mm[mm.cohort == "R-"].w
    return mm, mannwhitneyu(a_, b_).pvalue, ttest_ind(a_, b_, equal_var=False).pvalue


# ══════════════════════════════════════════════════════════════════════════
def part1(a, ctx, out):
    D, res, nulls, wn = ctx["D"], ctx["res"], ctx["nulls"], ctx["wn"]
    t_ctrs, cond_labels = ctx["t_ctrs"], ctx["cond_labels"]
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(183 / 25.4, 175 / 25.4))
    top = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.0], left=0.13, right=0.985, top=0.9, bottom=0.66, wspace=0.4)
    mid = fig.add_gridspec(1, 3, left=0.09, right=0.93, top=0.55, bottom=0.35, wspace=0.75)
    bot = fig.add_gridspec(1, 3, left=0.09, right=0.93, top=0.215, bottom=0.07, wspace=0.75)

    # a  windows
    ax = fig.add_subplot(top[0]); style(ax)
    for i, c in enumerate(CONDS[::-1]):
        t = t_ctrs[cond_labels.index(c)]
        cut = d016.REWARD_FREE_CUTS.get(c)
        ax.plot([t[0], t[-1]], [i, i], color="0.85", lw=4.5, solid_capstyle="butt")
        ax.plot([t[0], min(t[-1], cut) if cut is not None else t[-1]], [i, i], color="0.35", lw=4.5, solid_capstyle="butt")
        ax.plot([0, 0], [i - 0.32, i + 0.32], color="k", lw=0.8)
    ax.set_yticks(range(len(CONDS))); ax.set_yticklabels(CONDS[::-1]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time from stimulus / lick (s)")
    ax.set_title("Reward-free windows\n(dark: kept; light: removed)", loc="left")
    # b  pipeline
    ax = fig.add_subplot(top[1]); ax.axis("off")
    steps = ["neuron × PSTH features\n(odd / even trial halves)", "PCA 99% + σ² (median),\nfit on one half",
             "RBF kernel k(x, x′) on\nthe other half", "MMD² = k̄(R+,R+) + k̄(R−,R−)\n− 2k̄(R+,R−)",
             "null: permute cohort\nlabels between mice", "witness f(x): R+- or\nR−-like neurons"]
    n = len(steps); h = 0.118; gap = (1 - n * h) / (n - 1)
    for i, s in enumerate(steps):
        y1 = 1 - i * (h + gap); y0 = y1 - h
        ax.add_patch(FancyBboxPatch((0.05, y0), 0.9, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc="#f3f3f3", ec="0.55", lw=0.5, transform=ax.transAxes))
        ax.text(0.5, (y0 + y1) / 2, s, transform=ax.transAxes, ha="center", va="center", fontsize=FS - 1,
                linespacing=1.0)
        if i < n - 1:
            ax.annotate("", xy=(0.5, y0 - gap + 0.004), xytext=(0.5, y0 - 0.004), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35", mutation_scale=5))
    ax.set_title("Method (per PSTH type;\nalso per area group)", loc="left")
    # c  goals & metrics
    ax = fig.add_subplot(top[2]); ax.axis("off")
    blocks = [("Goal", "Do R+ and R− neurons differ in their\nresponse profiles before any reward?"),
              ("Metrics", "MMD²: distance between the two\nneuron-response distributions\n"
                          "z: MMD² relative to mouse-level null\nwitness f(x) > 0: R+-like; < 0: R−-like"),
              ("Criteria", "p < 0.05 in both trial halves\nBH-FDR across areas × PSTH types")]
    y = 0.98
    for head, body in blocks:
        ax.text(0.0, y, head, transform=ax.transAxes, va="top", fontsize=FS + 0.5)
        ax.text(0.0, y - 0.09, body, transform=ax.transAxes, va="top", fontsize=FS - 0.5, linespacing=1.15)
        y -= 0.09 + 0.085 * (body.count("\n") + 1) + 0.07
    ax.set_title("Goal and metrics", loc="left")

    # d  densities
    coh = D["cohorts"]; Z = D["test_proj"][:, :2]
    xl, yl = np.percentile(Z[:, 0], [1, 99]), np.percentile(Z[:, 1], [1, 99])
    axs = [fig.add_subplot(mid[i]) for i in range(3)] + [fig.add_subplot(bot[i]) for i in range(3)]
    for ax in axs:
        style(ax); ax.set_box_aspect(1)
    ax = axs[0]
    gx, gy = np.meshgrid(np.linspace(*xl, 90), np.linspace(*yl, 90))
    for lab, col in (("R+", RP), ("R-", RM)):
        zz = Z[coh == lab]; zz = zz[rng.choice(len(zz), min(4000, len(zz)), replace=False)]
        dens = gaussian_kde(zz.T)(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contour(gx, gy, dens, levels=np.quantile(dens, [0.6, 0.8, 0.9, 0.97]), colors=col, linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.plot([], [], color=RP, lw=0.8, label="R+"); ax.plot([], [], color=RM, lw=0.8, label="R−")
    ax.legend(frameon=False, loc="best", handlelength=1)
    ax.set_title("Cohort densities (PC1–PC2)", loc="left")
    # e  kernel per mouse pair
    ax = axs[1]
    S, codes, is_pos = D["S"], D["codes"], D["is_pos"]
    M = S.shape[1]; B = np.zeros((M, M)); np.add.at(B, codes, S)
    nn = np.bincount(codes, minlength=M).astype(float)
    Kbar = B / np.outer(nn, nn)
    Kbar[np.diag_indices(M)] = (np.diag(B) - nn) / np.maximum(nn * (nn - 1), 1)
    order = np.r_[np.flatnonzero(is_pos)[np.argsort(-Kbar[np.ix_(is_pos, is_pos)].mean(1))],
                  np.flatnonzero(~is_pos)[np.argsort(-Kbar[np.ix_(~is_pos, ~is_pos)].mean(1))]]
    im = ax.imshow(Kbar[np.ix_(order, order)], cmap="viridis", interpolation="none")
    npos = int(is_pos.sum())
    ax.axhline(npos - 0.5, color="w", lw=0.6); ax.axvline(npos - 0.5, color="w", lw=0.6)
    ax.set_xticks([npos / 2, npos + (M - npos) / 2]); ax.set_xticklabels(["R+ mice", "R− mice"])
    ax.set_yticks([npos / 2, npos + (M - npos) / 2]); ax.set_yticklabels(["R+", "R−"]); ax.tick_params(length=0)
    cax = ax.inset_axes([1.05, 0, 0.05, 1]); cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4)
    lo_, hi_ = float(np.nanmin(Kbar)), float(np.nanmax(Kbar))
    cb.set_ticks([lo_, hi_]); cb.set_ticklabels([f"{lo_:.2f}", f"{hi_:.2f}"]); cb.ax.tick_params(labelsize=FS - 1.5)
    ax.set_title(f"Mean kernel per mouse pair\n(all {len(codes):,} neurons, {M} mice)", loc="left")
    # f  decomposition
    ax = axs[2]
    tx, ty, txy, m2 = d017.mmd_terms(S, codes, is_pos)
    ax.bar([0, 1, 2], [tx, ty, txy], color=[RP, RM, "0.6"], width=0.65)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["within\nR+", "within\nR−", "across"], linespacing=0.95)
    ax.set_ylim(0, max(tx, ty, txy) * 1.3); ax.set_ylabel("mean kernel similarity")
    ax.text(0.5, 0.98, f"MMD² = {m2:.2g}", transform=ax.transAxes, ha="center", va="top")
    ax.set_title("MMD² decomposition", loc="left")
    # g  null
    ax = axs[3]
    rr = res[res.feature_version == a.example].set_index("direction")
    scale = 10 ** np.floor(np.log10(max(rr.mmd2_obs.max(), 1e-12)))
    for d, col, lab in ((DIRS[0], "0.35", "odd → even"), (DIRS[1], "#d9822b", "even → odd")):
        ax.hist(nulls[f"{a.example}__{d}"] / scale, bins=40, color=col, alpha=0.45, lw=0, label=lab)
        ax.axvline(rr.loc[d, "mmd2_obs"] / scale, color=col, lw=1)
    xmax = max(rr.mmd2_obs.max(), max(np.percentile(nulls[f"{a.example}__{d}"], 99.5) for d in DIRS)) / scale
    ax.set_xlim(right=xmax * 1.12); ax.set_yticks([])
    ax.set_xlabel(f"MMD² (×10^{int(np.log10(scale))})"); ax.set_ylabel("permutations")
    ax.legend(frameon=False, fontsize=FS - 1, loc="best", handlelength=0.8)
    ax.set_title(f"Mouse-level null ({a.example})\np = {rr.p_value.max():.2g} in both halves", loc="left")
    # h  witness
    ax = axs[4]
    w = D["w"] - D["w"].mean(); vw = np.percentile(np.abs(w), 98)
    sub = rng.choice(len(Z), min(6000, len(Z)), replace=False); sub = sub[np.argsort(np.abs(w[sub]))]
    ax.scatter(Z[sub, 0], Z[sub, 1], c=w[sub], cmap=d017.WIT_CMAP, vmin=-vw, vmax=vw, s=0.7, lw=0, rasterized=True)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Witness (centered)\ngreen: R+-like, magenta: R−-like", loc="left")
    # i  per mouse
    ax = axs[5]
    mm, pm, pw = mouse_witness(wn, a.example)
    for k, (lab, col) in enumerate((("R+", RP), ("R-", RM))):
        v = mm[mm.cohort == lab].w.values
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=col, lw=0)
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color="k", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R−"]); ax.axhline(0, color="0.6", lw=0.4)
    ax.set_ylabel("mouse mean witness"); ax.set_xlim(-0.6, 1.6)
    ax.set_title(f"Per mouse (MWU p = {pm:.2g};\nWelch p = {pw:.2g})", loc="left")

    for ax, l in zip([fig.axes[0], fig.axes[1], fig.axes[2]], "abc"):
        letter(fig, ax, l, dx_mm=22 if l == "a" else 4)
    for ax, l in zip(axs, "defghi"):
        letter(fig, ax, l, dx_mm=11)
    lab = "whole-brain, all PSTH types concatenated (full)" if a.example == "full" else a.example
    fig.suptitle(f"{ctx['title']} — data and method; panels d–i: {lab}", fontsize=FS + 1.5, x=0.01, y=0.995,
                 ha="left", va="top")
    for ext in ("png", "pdf", "svg"):
        tag = "" if a.example == "Auditory hit (lick)" else "_" + a.example.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(out / f"mmd_pub_part1_method{tag}.{ext}", dpi=400)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
def part2(a, ctx, out, pms):
    res, wn, pa = ctx["res"], ctx["wn"], ctx["pa"]
    rng = np.random.default_rng(1)
    # per-condition statistics
    rows = []
    for c in CONDS:
        g = res[res.feature_version == c].set_index("direction")
        _, pm, pw = mouse_witness(wn, c)
        rows.append(dict(condition=c, n_neurons=int(g.n_neurons.iloc[0]),
                         n_mice=f"{int(g.n_mice_pos.iloc[0])}/{int(g.n_mice_neg.iloc[0])}",
                         mmd2=g.mmd2_obs.mean(), z_oe=g.loc[DIRS[0], "mmd2_z"], z_eo=g.loc[DIRS[1], "mmd2_z"],
                         p_oe=g.loc[DIRS[0], "p_value"], p_eo=g.loc[DIRS[1], "p_value"],
                         p_max=g.p_value.max(), mouse_mwu_p=pm, mouse_welch_p=pw))
    st = pd.DataFrame(rows)
    st["q_bh_family"] = np.nan
    fm = st.condition.isin(FAMILY)
    st.loc[fm, "q_bh_family"] = multipletests(st.loc[fm, "p_max"], method="fdr_bh")[1]
    st.to_csv(out / "per_condition_stats.csv", index=False)
    fam = pa[pa.feature_version.isin(FAMILY)].copy()
    fam["q"] = multipletests(fam.p_max, method="fdr_bh")[1]
    fam["sig_raw"], fam["sig_fdr"] = fam.p_max < ALPHA, fam.q < ALPHA
    fam.to_csv(out / "per_area_family_fdr.csv", index=False)

    fig = plt.figure(figsize=(183 / 25.4, 240 / 25.4))
    g1 = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.55], left=0.15, right=0.99, top=0.935, bottom=0.765, wspace=0.08)
    g2 = fig.add_gridspec(1, 1, left=0.09, right=0.99, top=0.69, bottom=0.555)
    g3 = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], left=0.19, right=0.98, top=0.475, bottom=0.235, wspace=0.75)
    top = (fam[fam.sig_raw].sort_values(["sig_fdr", "z_mean"], ascending=False).drop_duplicates("feature_version")
           .head(4))
    tr_path = ctx["run"] / "time_resolved" / "time_resolved_summary.csv"
    n_tc = len(top) + (1 if tr_path.exists() else 0)
    g4 = fig.add_gridspec(1, max(n_tc, 1), left=0.09, right=0.99, top=0.155, bottom=0.04, wspace=0.55)

    # a  forest
    ax = fig.add_subplot(g1[0]); style(ax)
    y = np.arange(len(CONDS))[::-1]
    for (zc, col, dy, lab) in (("z_oe", "0.35", 0.15, "odd → even"), ("z_eo", "#d9822b", -0.15, "even → odd")):
        ax.scatter(st[zc], y + dy, s=7, color=col, zorder=3, label=lab)
    for yy, r in zip(y, st.itertuples()):
        mark = "**" if (np.isfinite(r.q_bh_family) and r.q_bh_family < ALPHA) else ("*" if r.p_max < ALPHA else "")
        ax.text(max(r.z_oe, r.z_eo) + 0.4, yy, mark, va="center", fontsize=FS + 1)
    ax.axvline(0, color="0.6", lw=0.4); ax.set_yticks(y); ax.set_yticklabels(CONDS)
    ax.set_xlabel("MMD² z (all neurons)"); ax.set_xlim(right=st[["z_oe", "z_eo"]].max().max() * 1.2 + 1)
    ax.legend(frameon=False, loc="lower right", fontsize=FS - 1.5, handletextpad=0.2)
    ax.set_title("Per PSTH type", loc="left")
    # b  table (rows aligned with the forest plot)
    axt = fig.add_subplot(g1[1]); axt.axis("off")
    fmtp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    cell = [[f"{r.n_neurons:,}", r.n_mice, f"{r.mmd2:.1e}", f"{r.z_oe:.1f} / {r.z_eo:.1f}",
             f"{fmtp(r.p_oe)} / {fmtp(r.p_eo)}", "–" if not np.isfinite(r.q_bh_family) else fmtp(r.q_bh_family),
             fmtp(r.mouse_mwu_p), fmtp(r.mouse_welch_p)] for r in st.itertuples()]
    tb = axt.table(cellText=cell, colLabels=["neurons", "mice\nR+/R−", "MMD²", "z\n(o→e / e→o)",
                                             "p\n(o→e / e→o)", "q BH\n(family)", "mouse\nMWU p", "mouse\nWelch p"],
                   loc="upper left", cellLoc="center", bbox=[0, -0.02, 1, 1.0])
    tb.auto_set_font_size(False); tb.set_fontsize(FS - 1.5)
    for (r_, c_), cl in tb.get_celld().items():
        cl.set_linewidth(0.3); cl.set_edgecolor("0.8")
        if r_ == 0:
            cl.set_text_props(fontsize=FS - 2.3); cl.set_height(cl.get_height() * 2.2)
        elif st.iloc[r_ - 1].p_max < ALPHA:
            cl.set_facecolor("#f2f2f2")
    axt.set_title("Statistics per PSTH type (grey rows: p<0.05 both halves; q: BH over active + passive-post types)",
                  loc="left", fontsize=FS - 0.5, pad=16)
    # c  per mouse per condition
    ax = fig.add_subplot(g2[0]); style(ax)
    for i, c in enumerate(CONDS):
        mm, pm, pw = mouse_witness(wn, c)
        for lab, col, dx in (("R+", RP, -0.17), ("R-", RM, 0.17)):
            v = mm[mm.cohort == lab].w.values
            ax.scatter(i + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=3, color=col, lw=0, alpha=0.85)
            ax.plot([i + dx - 0.12, i + dx + 0.12], [np.median(v)] * 2, color="k", lw=0.8)
        ax.text(i, 1.0, f"MWU {fmtp(pm)}\nWelch {fmtp(pw)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=FS - 2.2, color="k" if min(pm, pw) < ALPHA else "0.55")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(pms.cond_ticklabels(CONDS), fontsize=FS - 1, linespacing=0.95)
    ax.set_ylabel("mouse mean witness\n(centered)")
    ax.text(0, 1.2, "Per mouse and PSTH type (green R+, magenta R−; line: median; mouse-level tests above)",
            transform=ax.transAxes, fontsize=FS)
    # d  per-area heatmap
    ax = fig.add_subplot(g3[0])
    import ephys_utilities.allen_utils.allen_utils as allen_utils
    rws = [g for g in allen_utils.get_area_group_custom_order() if g in set(fam.area_group)]
    Zp = fam.pivot(index="area_group", columns="feature_version", values="z_mean").reindex(rws)[FAMILY]
    Sr = fam.pivot(index="area_group", columns="feature_version", values="sig_raw").reindex(rws)[FAMILY]
    Sf = fam.pivot(index="area_group", columns="feature_version", values="sig_fdr").reindex(rws)[FAMILY]
    Nn = fam.groupby("area_group")[["n_neurons", "n_mice_pos", "n_mice_neg"]].first().reindex(rws)
    vmax = max(4.0, float(np.nanpercentile(Zp.values, 98)))
    im = ax.imshow(Zp.values, cmap="magma_r", vmin=0, vmax=vmax, aspect="auto")
    for i in range(Zp.shape[0]):
        for j in range(Zp.shape[1]):
            v = Zp.values[i, j]
            if np.isfinite(v):
                mark = "**" if Sf.values[i, j] else ("*" if Sr.values[i, j] else "")
                ax.text(j, i, f"{v:.1f}{mark}", ha="center", va="center", fontsize=FS - 2,
                        color="w" if v > 0.55 * vmax else "k")
    ax.set_xticks(range(len(FAMILY))); ax.set_xticklabels(pms.cond_ticklabels(FAMILY), fontsize=FS - 1.5, linespacing=0.95)
    ax.set_yticks(range(len(rws)))
    ax.set_yticklabels([f"{g} ({int(Nn.loc[g, 'n_mice_pos'])}/{int(Nn.loc[g, 'n_mice_neg'])})" for g in rws],
                       fontsize=FS - 1.5)
    cax = ax.inset_axes([1.02, 0, 0.025, 1]); cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4)
    cb.set_label("MMD² z", labelpad=1); cb.ax.tick_params(labelsize=FS - 1.5)
    ax.set_title(f"Per area group (mice R+/R−): * p<0.05 both halves; ** BH-FDR q<0.05 over {len(fam)} tests",
                 loc="left", fontsize=FS - 0.5)
    # e  counts
    ax = fig.add_subplot(g3[1]); style(ax)
    cnt = fam.groupby("feature_version")[["sig_raw", "sig_fdr"]].sum().reindex(FAMILY)
    yy = np.arange(len(FAMILY))[::-1]
    ax.barh(yy + 0.18, cnt.sig_raw, height=0.36, color="0.7", label="uncorrected")
    ax.barh(yy - 0.18, cnt.sig_fdr, height=0.36, color="0.2", label="BH-FDR")
    ax.set_yticks(yy); ax.set_yticklabels(FAMILY, fontsize=FS - 1)
    ax.set_xlabel("# area groups significant"); ax.legend(frameon=False, loc="lower right", fontsize=FS - 1.5)
    ax.set_title("Significant area groups", loc="left", fontsize=FS - 0.5)
    # f..  time courses
    Xr, mice_r, area_r, coh_r = ctx["Xr"], ctx["mice_r"], ctx["area_r"], ctx["coh_r"]
    t_ctrs, cond_labels, slices = ctx["t_ctrs"], ctx["cond_labels"], ctx["slices"]
    tc_axes = []
    for k, r in enumerate(top.itertuples()):
        ax = fig.add_subplot(g4[k]); style(ax); tc_axes.append(ax)
        s0, s1 = slices[r.feature_version]; t = t_ctrs[cond_labels.index(r.feature_version)]
        sel = area_r == r.area_group
        for lab, col in (("R+", RP), ("R-", RM)):
            m_ids = [m for m in np.unique(mice_r[sel & (coh_r == lab)]) if (sel & (mice_r == m)).sum() >= 5]
            pmm = np.stack([Xr[sel & (mice_r == m), s0:s1].mean(0) for m in m_ids])
            mu, se = pmm.mean(0), pmm.std(0) / np.sqrt(len(pmm))
            ax.fill_between(t, mu - se, mu + se, color=col, alpha=0.25, lw=0)
            ax.plot(t, mu, color=col, lw=0.8, label=f"{lab} ({len(pmm)})")
        cut = d016.REWARD_FREE_CUTS.get(r.feature_version)
        if cut is not None:
            ax.axvspan(cut, t[-1], color="0.9", lw=0, zorder=0)
        ax.axvline(0, color="0.5", lw=0.4, ls=":")
        ax.set_xlabel(f"time from {'lick' if 'lick' in r.feature_version.lower() else 'stim.'} (s)")
        if k == 0:
            ax.set_ylabel("Hz (baseline-sub.),\nmouse mean ± SEM")
        ax.legend(frameon=False, fontsize=FS - 2, loc="best", handlelength=0.8)
        ax.set_title(f"{r.area_group}\n{r.feature_version}, z = {r.z_mean:.1f}{'**' if r.sig_fdr else '*'}",
                     loc="left", fontsize=FS - 1)
    if tr_path.exists():
        ax = fig.add_subplot(g4[n_tc - 1]); style(ax); tc_axes.append(ax)
        tr = pd.read_csv(tr_path); g = tr[tr.condition == a.example]
        ax.plot(g.t_center, g.z_mean, color="k", lw=0.8, label="profile")
        ax.plot(g.t_center, g.amp_z_mean, color="0.55", lw=0.8, ls="--", label="amplitude")
        sig = g[g.significant_both]
        ax.scatter(sig.t_center, np.full(len(sig), g.z_mean.max() * 1.12), marker="s", s=3, color="k")
        ax.axhline(0, color="0.6", lw=0.4); ax.axvline(0, color="0.5", lw=0.4, ls=":")
        ax.set_xlabel(f"time from {'lick' if 'lick' in a.example.lower() else 'stim.'} (s)"); ax.set_ylabel("z")
        ax.legend(frameon=False, fontsize=FS - 2, loc="best")
        ax.set_title(f"Time-resolved MMD²\n{a.example} (all neurons)", loc="left", fontsize=FS - 1)

    for ax, l in zip([fig.axes[0], fig.axes[1]], "ab"):
        letter(fig, ax, l, dx_mm=27 if l == "a" else 2)
    letter(fig, fig.axes[2], "c", dx_mm=15, dy_mm=9)
    letter(fig, fig.axes[3], "d", dx_mm=34)
    letter(fig, fig.axes[4], "e", dx_mm=25)
    for ax, l in zip(tc_axes, "fghij"):
        letter(fig, ax, l, dx_mm=11 if l == "f" else 7)
    fig.suptitle(f"{ctx['title']} — results", fontsize=FS + 1.5, x=0.01, y=0.995, ha="left", va="top")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"mmd_pub_part2_results.{ext}", dpi=400)
    plt.close(fig)
    print(st.round(4).to_string(index=False))
    print(fam.sort_values("q").head(12)[["area_group", "feature_version", "z_mean", "p_max", "q"]].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", default="mmd_rewardfree")
    ap.add_argument("--example", default="Auditory hit (lick)", help="example for Part 2 time-resolved panel")
    ap.add_argument("--part1-examples", nargs="*", default=["full", "Auditory hit (lick)", "Whisker miss"],
                    help="Part 1 (method) is rendered once per example feature version")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    root, fm_path, md = d016.paths(a.variant)
    run = md / a.run_dir
    out = run / "pub_overview"; out.mkdir(exist_ok=True)
    import yaml
    with np.load(fm_path, allow_pickle=True) as z:
        t_ctrs = [z[f"t_ctr_{i}"] for i in range(int(z["n_conds"]))]
    pipe_cfg = yaml.unsafe_load(open(fm_path.parent / "config_used.yaml"))
    _, cond_labels, _, _, _ = d016.rmu.get_conditions(pipe_cfg)
    cond_labels = list(cond_labels)
    wn = pd.read_parquet(run / "witness_per_neuron.parquet"); wn = wn[wn.loo == "mouse"]
    X, meta, _, _ = mmd._load_npz_data(fm_path)
    rm = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(meta["unit_ids"])), index=meta["unit_ids"]).loc[rm["unit_ids"]].to_numpy()
    ctx = dict(run=run, res=pd.read_csv(run / "mmd_cv_results.csv"), nulls=np.load(run / "mmd_cv_nulls.npz"), wn=wn,
               pa=pd.read_csv(run / "per_area" / "mmd_per_area_summary.csv"), t_ctrs=t_ctrs, cond_labels=cond_labels,
               D=d017.prepare(a.variant, a.example, "fit_odd_test_even", None),
               Xr=X[pos], mice_r=meta["mouse"][pos], area_r=meta["area_group"][pos].astype(str),
               coh_r=meta["reward_group"][pos].astype(str),
               slices=mmd._build_default_trial_type_slices(meta["n_bins_list"], cond_labels=cond_labels),
               title=a.title or f"Reward-free R+ vs R− comparison (MMD²), {a.variant}")
    pms = setup_style()
    import copy
    for ex in a.part1_examples:
        a1 = copy.copy(a); a1.example = ex
        part1(a1, {**ctx, "D": ctx["D"] if ex == a.example else d017.prepare(a.variant, ex, "fit_odd_test_even", None)},
              out)
    part2(a, ctx, out, pms)
    print("ALL DONE")


if __name__ == "__main__":
    main()
