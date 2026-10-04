"""Method figures (square panels) for the reward-free MMD run of 016, one set per feature version and CV direction.

Panels (each also saved alone):
  a  input: reward-free PSTH feature matrix (fit / test split), neurons sorted by cohort then rastermap order
  b  PCA (99% variance, fit on the FIT split): cumulative variance
  c  test split in PC1-PC2, R+ / R- with density contours (illustration only; the test uses all PCs)
  d  kernel bandwidth: pairwise squared distances (label-blind) with the median-heuristic sigma^2
  e  kernel similarity matrix on a random subsample, ordered by cohort then rastermap position
  f  MMD^2 decomposition (mean within R+, within R-, cross) computed on ALL neurons from mouse block sums
  g  mouse-level permutation: true cohort vs one random relabelling (each mouse moves as a block)
  h  null distribution (mouse-level permutations) with the observed MMD^2
  i  witness function (mouse-LOO) on the PC1-PC2 projection
  j  witness distribution per cohort
  k  witness along the rastermap order (neurons), cluster means overlaid
Usage: 017_mmd_method_figures.py --variant V --run-dir mmd_example_oldwindows [--versions ...] [--direction fit_odd_test_even]
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/rastermap_psth"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import importlib                                                     # noqa: E402
import cluster_mmd_analysis_new as mmd                                # noqa: E402
d016 = importlib.import_module("016_mmd_rewardfree_cv")

RP, RM = "#00B400", "#C800C8"
WIT_CMAP = LinearSegmentedColormap.from_list("rpm", [RM, "white", RP])
SQ = 3.4                     # inches per square panel
N_KERNEL_SHOW = 800          # neurons shown in the kernel-matrix panel (random, stratified by cohort)
N_SCATTER = 4000             # neurons shown in scatter panels


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)


def prepare(variant, version, direction, n_perm_from_run):
    root, fm_path, md = d016.paths(variant)
    X, meta, Xo, Xe = mmd._load_npz_data(fm_path)
    import yaml
    with np.load(fm_path, allow_pickle=True) as z:
        t_ctrs = [z[f"t_ctr_{i}"] for i in range(int(z["n_conds"]))]
    pipe_cfg = yaml.unsafe_load(open(fm_path.parent / "config_used.yaml"))
    _, cond_labels, _, _, _ = d016.rmu.get_conditions(pipe_cfg)
    cond_labels = list(cond_labels)
    rm = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    rm_units, rm_labels, isort = rm["unit_ids"], rm["cluster_labels"].astype(int), rm["isort"]
    pos = pd.Series(np.arange(len(meta["unit_ids"])), index=meta["unit_ids"]).loc[rm_units].to_numpy()
    mice, cohorts = meta["mouse"][pos], meta["reward_group"][pos].astype(str)
    slices = mmd._build_default_trial_type_slices(meta["n_bins_list"], cond_labels=cond_labels)
    exclude = mmd.time_cut_exclude_bins(t_ctrs, cond_labels, d016.REWARD_FREE_CUTS, pipe_cfg.get("bin_ms", 10) / 1000)
    V_odd = d016.build_versions(Xo[pos], cond_labels, slices, exclude)[version]
    V_even = d016.build_versions(Xe[pos], cond_labels, slices, exclude)[version]
    fit, test = (V_odd, V_even) if direction == "fit_odd_test_even" else (V_even, V_odd)
    cfg = {**mmd.DEFAULT_MMD_CFG, "cohort_labels": ("R+", "R-"), "pca_test_n_components": None,
           "pca_test_variance_retained": 0.99}
    fit_proj, pca = mmd._maybe_pca_reduce(fit, cfg)
    sigma2 = mmd.median_sq_dist(fit_proj)
    test_proj = pca.transform(test)
    codes, uniq, is_pos = mmd._mouse_codes_and_cohorts(mice, cohorts, ("R+", "R-"))
    S = mmd.kernel_mouse_sums(test_proj, codes, len(uniq), sigma2)
    w_n, w_m = mmd.witness_from_mouse_sums(S, codes, is_pos)
    rank = np.empty(len(isort), int); rank[isort] = np.arange(len(isort))      # rastermap position per neuron
    # untrimmed test-split PSTHs of the version's conditions (panel a: what is kept / removed)
    conds = [version] if version in cond_labels else (d016.GROUPS[version] or cond_labels)
    Xtest = (Xe if direction == "fit_odd_test_even" else Xo)[pos]
    psth = []
    for c in conds:
        s0, s1 = slices[c]
        psth.append(dict(cond=c, t=np.asarray(t_ctrs[cond_labels.index(c)]), X=Xtest[:, s0:s1],
                         cut=d016.REWARD_FREE_CUTS.get(c)))
    return dict(fit=fit, test=test, fit_proj=fit_proj, test_proj=test_proj, pca=pca, sigma2=sigma2, S=S,
                codes=codes, uniq=uniq, is_pos=is_pos, cohorts=cohorts, mice=mice, w=w_m, rank=rank,
                rm_labels=rm_labels, md=md, psth=psth, bin_s=pipe_cfg.get("bin_ms", 10) / 1000)


def mmd_terms(S, codes, is_pos):
    M = S.shape[1]
    B = np.zeros((M, M)); np.add.at(B, codes, S)
    n = np.bincount(codes, minlength=M).astype(float)
    p, q = is_pos.astype(float), (~is_pos).astype(float)
    m_, n_ = n @ p, n @ q
    tx = (p @ B @ p - m_) / (m_ * (m_ - 1)); ty = (q @ B @ q - n_) / (n_ * (n_ - 1)); txy = p @ B @ q / (m_ * n_)
    return tx, ty, txy, tx + ty - 2 * txy


def panels(D, res_row, null, version, direction, out):
    rng = np.random.default_rng(0)
    coh = D["cohorts"]; pos_m = coh == "R+"
    order = np.lexsort((D["rank"], ~pos_m))
    figs = {}

    def new(name):
        f, ax = plt.subplots(figsize=(SQ, SQ), dpi=200); figs[name] = f; _style(ax); return f, ax

    # a: input matrix
    f, ax = new("a_input_matrix")
    Xs = D["test"][order]; v = np.nanpercentile(np.abs(Xs), 99)
    ax.imshow(Xs, aspect="auto", cmap="bwr", vmin=-v, vmax=v, interpolation="none")
    ax.axhline(pos_m.sum() - 0.5, color="k", lw=1)
    ax.set_xlabel("reward-free PSTH bins", fontsize=7); ax.set_ylabel(f"neurons (R+ | R−; n={len(Xs):,})", fontsize=7)
    ax.set_title("a  Input (test split)", fontsize=8, loc="left")
    # b: PCA variance
    f, ax = new("b_pca_variance")
    ev = np.cumsum(D["pca"].explained_variance_ratio_)
    ax.plot(np.arange(1, len(ev) + 1), ev, color="k", lw=1)
    ax.axhline(0.99, color="0.5", ls="--", lw=0.7)
    ax.set_xlabel("# PCs", fontsize=7); ax.set_ylabel("cumulative variance", fontsize=7)
    ax.set_title(f"b  PCA fit on fit split: {len(ev)} PCs (99%)", fontsize=8, loc="left")
    # c: PC1-PC2 by cohort
    f, ax = new("c_pca_cohorts")
    Z = D["test_proj"][:, :2]; sub = rng.choice(len(Z), min(N_SCATTER, len(Z)), replace=False)
    for lab, col in (("R+", RP), ("R-", RM)):
        s = sub[coh[sub] == lab]
        ax.scatter(Z[s, 0], Z[s, 1], s=2, color=col, alpha=0.35, lw=0, label=lab)
        k = gaussian_kde(Z[coh == lab].T[:, rng.choice((coh == lab).sum(), min(3000, (coh == lab).sum()), replace=False)])
        xl, yl = np.percentile(Z[:, 0], [1, 99]), np.percentile(Z[:, 1], [1, 99])
        gx, gy = np.meshgrid(np.linspace(*xl, 80), np.linspace(*yl, 80))
        ax.contour(gx, gy, k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape), levels=4, colors=col, linewidths=0.7)
    ax.set_xlim(*np.percentile(Z[:, 0], [0.5, 99.5])); ax.set_ylim(*np.percentile(Z[:, 1], [0.5, 99.5]))
    ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7); ax.legend(fontsize=6, frameon=False, markerscale=3)
    ax.set_title("c  Test split, PC1–PC2 (illustration)", fontsize=8, loc="left")
    # d: bandwidth
    f, ax = new("d_bandwidth")
    sd = pdist(D["test_proj"][rng.choice(len(Z), min(3000, len(Z)), replace=False)], "sqeuclidean")
    ax.hist(sd[sd > 0], bins=np.geomspace(sd[sd > 0].min(), sd.max(), 60), color="0.7"); ax.set_xscale("log")
    ax.axvline(D["sigma2"], color="crimson", lw=1.3, label=r"$\sigma^2$ (median, fit split)")
    ax.set_xlabel(r"$\|x_i-x_j\|^2$", fontsize=7); ax.set_ylabel("pairs", fontsize=7); ax.legend(fontsize=6, frameon=False)
    ax.set_title("d  Kernel bandwidth", fontsize=8, loc="left")
    # e: kernel matrix (subsample)
    f, ax = new("e_kernel_matrix")
    n_half = N_KERNEL_SHOW // 2
    si = np.concatenate([rng.choice(np.flatnonzero(pos_m), n_half, replace=False),
                         rng.choice(np.flatnonzero(~pos_m), n_half, replace=False)])
    si = si[np.lexsort((D["rank"][si], ~pos_m[si]))]
    P = D["test_proj"][si]
    K = np.exp(-((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) / (2 * D["sigma2"]))
    # within each cohort block: decreasing mean similarity to all shown neurons (smooth display)
    ks = K.mean(1)
    o = np.r_[np.argsort(-ks[:n_half]), n_half + np.argsort(-ks[n_half:])]
    K = K[np.ix_(o, o)]
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=np.percentile(K, 99), interpolation="none")
    ax.axhline(n_half - 0.5, color="w", lw=0.8); ax.axvline(n_half - 0.5, color="w", lw=0.8)
    ax.set_xticks([n_half / 2, 1.5 * n_half]); ax.set_xticklabels(["R+", "R−"], fontsize=7)
    ax.set_yticks([n_half / 2, 1.5 * n_half]); ax.set_yticklabels(["R+", "R−"], fontsize=7)
    f.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    ax.set_title(f"e  Kernel k(i,j), {N_KERNEL_SHOW} random neurons\n    (per cohort: decreasing mean similarity)",
                 fontsize=7.5, loc="left")
    # f: MMD terms
    f, ax = new("f_mmd_terms")
    tx, ty, txy, m2 = mmd_terms(D["S"], D["codes"], D["is_pos"])
    ax.bar(["within R+", "within R−", "cross"], [tx, ty, txy], color=[RP, RM, "0.55"])
    lo = min(tx, ty, txy); ax.set_ylim(lo - (max(tx, ty, txy) - lo) * 3, max(tx, ty, txy) * 1.002)
    ax.set_ylabel("mean kernel similarity", fontsize=7)
    ax.set_title(f"f  MMD² = {tx:.4f} + {ty:.4f} − 2·{txy:.4f}\n    = {m2:.2e} (all neurons)", fontsize=7.5, loc="left")
    # g: permutation schematic
    f, ax = new("g_permutation")
    M = len(D["uniq"]); order_m = np.argsort(~D["is_pos"])
    perm = np.zeros(M, bool); perm[rng.choice(M, D["is_pos"].sum(), replace=False)] = True
    n_m = np.bincount(D["codes"], minlength=M)[order_m]
    x = np.cumsum(np.r_[0, n_m[:-1]])
    ax.bar(x, np.ones(M), width=n_m, align="edge", color=[RP if p else RM for p in D["is_pos"][order_m]], edgecolor="w", lw=0.3)
    ax.bar(x, -np.ones(M), width=n_m, align="edge", color=[RP if p else RM for p in perm[order_m]], edgecolor="w", lw=0.3)
    ax.set_yticks([0.5, -0.5]); ax.set_yticklabels(["true", "one\npermutation"], fontsize=7)
    ax.set_xlabel(f"{M} mice (width = n neurons)", fontsize=7)
    ax.set_title("g  Mouse-level permutation", fontsize=8, loc="left")
    # h: null
    f, ax = new("h_null")
    ax.hist(null, bins=50, color="0.7")
    ax.axvline(res_row["mmd2_obs"], color="crimson", lw=1.3)
    ax.set_xlabel("MMD²", fontsize=7); ax.set_ylabel("permutations", fontsize=7)
    ax.set_title(f"h  Null: p = {res_row['p_value']:.3g}, z = {res_row['mmd2_z']:.1f}", fontsize=8, loc="left")
    # i: witness on PCA
    f, ax = new("i_witness_pca")
    w = D["w"]; vw = np.percentile(np.abs(w), 98)
    s_ = sub[np.argsort(np.abs(w[sub]))]
    sc = ax.scatter(Z[s_, 0], Z[s_, 1], c=w[s_], cmap=WIT_CMAP, vmin=-vw, vmax=vw, s=3, lw=0)
    ax.set_xlim(*np.percentile(Z[:, 0], [0.5, 99.5])); ax.set_ylim(*np.percentile(Z[:, 1], [0.5, 99.5]))
    f.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7)
    ax.set_title("i  Witness (mouse-LOO; R+ > 0 > R−)", fontsize=8, loc="left")
    # j: witness distribution
    f, ax = new("j_witness_distribution")
    bins = np.linspace(-vw, vw, 60)
    for lab, col in (("R+", RP), ("R-", RM)):
        ax.hist(w[coh == lab], bins=bins, color=col, alpha=0.5, density=True, label=lab)
    ax.axvline(0, color="k", lw=0.5); ax.axvline(w.mean(), color="0.3", ls="--", lw=0.8, label="population mean")
    ax.set_xlabel("witness", fontsize=7); ax.set_ylabel("density", fontsize=7); ax.legend(fontsize=6, frameon=False)
    ax.set_title("j  Witness per cohort", fontsize=8, loc="left")
    # k: witness along rastermap order
    f, ax = new("k_witness_rastermap")
    ordr = np.argsort(D["rank"])
    ax.scatter(np.arange(len(w)), w[ordr], s=0.5, color="0.7", lw=0)
    cm = pd.Series(w).groupby(D["rm_labels"]).mean()
    pos_c = pd.Series(D["rank"]).groupby(D["rm_labels"]).median()
    ax.plot(pos_c.values, cm.loc[pos_c.index].values, color="k", lw=0.9, marker="o", ms=1.5)
    ax.axhline(w.mean(), color="0.3", ls="--", lw=0.7); ax.axhline(0, color="k", lw=0.4)
    ax.set_ylim(-vw, vw); ax.set_xlabel("rastermap position (neurons)", fontsize=7); ax.set_ylabel("witness", fontsize=7)
    ax.set_title("k  Witness along rastermap (line: cluster means)", fontsize=8, loc="left")

    out.mkdir(parents=True, exist_ok=True)
    tag = f"{version.replace(' ', '_').replace('(', '').replace(')', '')}__{direction}"
    for name, fig in figs.items():
        fig.tight_layout(); fig.savefig(out / f"{tag}__{name}.png"); fig.savefig(out / f"{tag}__{name}.pdf")
    # combined 3 x 4 grid
    grid = plt.figure(figsize=(4 * SQ, 3 * SQ), dpi=150)
    for i, (name, fig) in enumerate(figs.items()):
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        a = grid.add_subplot(3, 4, i + 1); a.imshow(img); a.axis("off")
        plt.close(fig)
    grid.suptitle(f"Reward-free MMD², {version} ({direction.replace('_', ' ')})", fontsize=11)
    grid.tight_layout(); grid.savefig(out / f"{tag}__overview.png"); grid.savefig(out / f"{tag}__overview.pdf")
    plt.close(grid)


def pub_figure(D, res_df, nulls, version, out, width_mm=183, fs=6.5):
    """Publication figure (2 x 4 square panels, 183 mm): a reward-free PSTHs, b PC1-PC2 densities,
    c sorted kernel matrix, d MMD^2 decomposition, e CV null distributions, f witness in PC space,
    g witness per cohort, h per-version summary (z, both CV directions)."""
    import rastermap_psth.population_matrix_summary as pms
    pms.use_pub_font()
    plt.rcParams.update({"font.size": fs, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.titlesize": fs, "axes.labelsize": fs, "xtick.labelsize": fs - 0.5,
                         "ytick.labelsize": fs - 0.5, "legend.fontsize": fs - 0.5})
    rng = np.random.default_rng(0)
    coh = D["cohorts"]; pos_m = coh == "R+"
    W = width_mm / 25.4
    fig = plt.figure(figsize=(W, W * 0.54))
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.975, top=0.83, bottom=0.1, wspace=0.75, hspace=0.8)
    axs = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    for ax, letter in zip(axs, "abcdefgh"):
        _style(ax)
        ax.text(-0.3, 1.13, letter, transform=ax.transAxes, fontsize=fs + 2.5, va="bottom", ha="left")
    for ax in axs:
        ax.set_box_aspect(1)

    # a  mean PSTH per cohort (test split), removed bins shaded
    ax = axs[0]
    x0 = 0.0
    for k, p in enumerate(D["psth"]):
        t = p["t"]; tt = t - t[0] + x0
        for lab, col in (("R+", RP), ("R-", RM)):
            m = p["X"][coh == lab]
            mu, se = m.mean(0), m.std(0) / np.sqrt(len(m))
            ax.fill_between(tt, mu - se, mu + se, color=col, alpha=0.25, lw=0)
            ax.plot(tt, mu, color=col, lw=0.8, label=lab if k == 0 else None)
        if p["cut"] is not None:
            ax.axvspan(p["cut"] + D["bin_s"] / 2 - t[0] + x0, tt[-1], color="0.88", zorder=0, lw=0)
        ax.axvline(-t[0] + x0, color="0.5", lw=0.4, ls=":")
        x0 = tt[-1] + 0.05
    ax.set_xticks([]); ax.set_xlabel("time (conditions concatenated)" if len(D["psth"]) > 1 else
                                     f"time from {'lick' if 'lick' in version.lower() else 'stimulus'}")
    if len(D["psth"]) == 1:
        t = D["psth"][0]["t"]; ticks = [v for v in (-0.2, 0, 0.2, 0.4) if t[0] <= v <= t[-1]]
        ax.set_xticks([v - t[0] for v in ticks]); ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_ylabel("population mean (a.u.)")
    ax.legend(frameon=False, loc="upper right", handlelength=1, bbox_to_anchor=(1.0, 1.02))
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))            # room for the legend
    ax.set_title("Reward-free input\n(grey: removed)", loc="left")

    # b  PC1-PC2 densities
    ax = axs[1]
    Z = D["test_proj"][:, :2]
    xl, yl = np.percentile(Z[:, 0], [1, 99]), np.percentile(Z[:, 1], [1, 99])
    gx, gy = np.meshgrid(np.linspace(*xl, 90), np.linspace(*yl, 90))
    sub = rng.choice(len(Z), min(N_SCATTER, len(Z)), replace=False)
    ax.scatter(Z[sub, 0], Z[sub, 1], s=0.6, color="0.75", lw=0, rasterized=True)
    for lab, col in (("R+", RP), ("R-", RM)):
        zz = Z[coh == lab]; zz = zz[rng.choice(len(zz), min(3000, len(zz)), replace=False)]
        dens = gaussian_kde(zz.T)(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contour(gx, gy, dens, levels=np.quantile(dens, [0.6, 0.8, 0.9, 0.97]), colors=col, linewidths=0.6)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Neurons in PC space\n({D['pca'].n_components_} PCs, 99% var.)", loc="left")

    # c  kernel matrix (sorted)
    ax = axs[2]
    n_half = N_KERNEL_SHOW // 2
    si = np.concatenate([rng.choice(np.flatnonzero(pos_m), n_half, replace=False),
                         rng.choice(np.flatnonzero(~pos_m), n_half, replace=False)])
    P = D["test_proj"][si]
    K = np.exp(-((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) / (2 * D["sigma2"]))
    ks = K.mean(1); o = np.r_[np.argsort(-ks[:n_half]), n_half + np.argsort(-ks[n_half:])]
    im = ax.imshow(K[np.ix_(o, o)], cmap="viridis", vmin=0, vmax=1, interpolation="none", rasterized=True)
    ax.axhline(n_half - 0.5, color="w", lw=0.6); ax.axvline(n_half - 0.5, color="w", lw=0.6)
    ax.set_xticks([n_half / 2, 1.5 * n_half]); ax.set_xticklabels(["R+", "R−"])
    ax.set_yticks([n_half / 2, 1.5 * n_half]); ax.set_yticklabels(["R+", "R−"])
    ax.tick_params(length=0)
    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 1]); cb.outline.set_linewidth(0.4)
    ax.set_title(f"Kernel similarity k(x, x′)\n({N_KERNEL_SHOW} random neurons)", loc="left")

    # d  MMD^2 decomposition
    ax = axs[3]
    tx, ty, txy, m2 = mmd_terms(D["S"], D["codes"], D["is_pos"])
    vals = [tx, ty, txy]
    ax.bar([0, 1, 2], vals, color=[RP, RM, "0.6"], width=0.65)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["R+/R+", "R−/R−", "R+/R−"])
    ax.set_ylim(0, max(vals) * 1.45); ax.set_ylabel("mean k")
    ax.text(0.5, 0.98, f"MMD² = {m2:.3g}", transform=ax.transAxes, ha="center", va="top")
    ax.text(0.5, 0.88, "k̄(R+,R+) + k̄(R−,R−)\n− 2k̄(R+,R−)", transform=ax.transAxes, ha="center", va="top",
            fontsize=fs - 1.5, color="0.35", linespacing=1.0)
    ax.set_ylim(0, max(vals) * 1.6)
    ax.set_title("MMD² decomposition\n(all neurons)", loc="left")

    # e  CV nulls
    ax = axs[4]
    rr = res_df[res_df.feature_version == version].set_index("direction")
    for d, col, lab in (("fit_odd_test_even", "0.35", "odd → even"), ("fit_even_test_odd", "#d9822b", "even → odd")):
        nl = nulls[f"{version}__{d}"]
        ax.hist(nl, bins=40, color=col, alpha=0.45, lw=0, label=f"{lab}\np = {rr.loc[d, 'p_value']:.3g}")
        ax.axvline(rr.loc[d, "mmd2_obs"], color=col, lw=1)
    ax.set_xlabel("MMD²"); ax.set_ylabel("permutations")
    xmax = max(rr["mmd2_obs"].max(), max(np.percentile(nulls[f"{version}__{d}"], 99.5) for d in rr.index))
    ax.set_xlim(right=xmax * 1.12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.22, 1.0), handlelength=0.8, fontsize=fs - 1.5)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2)); ax.xaxis.get_offset_text().set_fontsize(fs - 1)
    ax.set_title("Mouse-level permutation null\n(fit / test on trial halves)", loc="left")

    # f  witness in PC space
    ax = axs[5]
    w = D["w"]; vw = np.percentile(np.abs(w), 98)
    s_ = sub[np.argsort(np.abs(w[sub]))]
    sc = ax.scatter(Z[s_, 0], Z[s_, 1], c=w[s_], cmap=WIT_CMAP, vmin=-vw, vmax=vw, s=0.8, lw=0, rasterized=True)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(sc, cax=cax); cb.outline.set_linewidth(0.4)
    ax.set_title("Witness f(x)\n(>0: R+-like, <0: R−-like)", loc="left")

    # g  witness per cohort
    ax = axs[6]
    bins = np.linspace(-vw, vw, 50)
    for lab, col in (("R+", RP), ("R-", RM)):
        ax.hist(w[coh == lab], bins=bins, color=col, alpha=0.5, density=True, lw=0, label=lab)
    ax.axvline(0, color="k", lw=0.4); ax.axvline(w.mean(), color="0.3", ls="--", lw=0.6)
    ax.set_xlabel("witness (mouse left out)"); ax.set_ylabel("density"); ax.legend(frameon=False, loc="best")
    ax.set_title("Witness per cohort\n(dashed: population mean)", loc="left")

    # h  summary across feature versions
    ax = axs[7]
    piv = res_df.pivot(index="feature_version", columns="direction", values="mmd2_z")
    pmax = res_df.groupby("feature_version").p_value.max()
    order = list(dict.fromkeys(res_df.feature_version))[::-1]
    y = np.arange(len(order))
    for d, col, dy in (("fit_odd_test_even", "0.35", -0.15), ("fit_even_test_odd", "#d9822b", 0.15)):
        ax.scatter(piv.loc[order, d], y + dy, s=5, color=col, zorder=3)
    sig = [pmax[v] < 0.05 for v in order]
    ax.scatter([piv.loc[v].max() + 0.8 for v, s_ in zip(order, sig) if s_], [yy for yy, s_ in zip(y, sig) if s_],
               marker="*", s=12, color="k")
    ax.axvline(0, color="0.6", lw=0.4)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=fs - 1.5); ax.set_xlabel("MMD² z")
    ax.set_box_aspect(None)
    ax.set_title("All feature versions\n(*: p<0.05 both halves)", loc="left")

    fig.suptitle(f"Reward-free neuron-level R+ vs R− comparison (MMD²): {version}", fontsize=fs + 1.5, x=0.02,
                 y=0.985, ha="left", va="top")
    out.mkdir(parents=True, exist_ok=True)
    tag = version.replace(' ', '_').replace('(', '').replace(')', '')
    fig.savefig(out / f"pub_mmd_{tag}.png", dpi=400); fig.savefig(out / f"pub_mmd_{tag}.pdf")
    fig.savefig(out / f"pub_mmd_{tag}.svg")
    plt.close(fig); plt.rcdefaults()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", required=True, help="016 output folder name inside the rastermap method dir")
    ap.add_argument("--versions", nargs="*", default=["Whisker miss", "Auditory hit", "full"])
    ap.add_argument("--direction", default="fit_odd_test_even")
    ap.add_argument("--pub", action="store_true", help="publication figure only")
    a = ap.parse_args()
    _, _, md = d016.paths(a.variant)
    run = md / a.run_dir
    res = pd.read_csv(run / "mmd_cv_results.csv")
    nulls = np.load(run / "mmd_cv_nulls.npz")
    for v in a.versions:
        row = res[(res.feature_version == v) & (res.direction == a.direction)].iloc[0]
        D = prepare(a.variant, v, a.direction, None)
        if a.pub:
            pub_figure(D, res, nulls, v, run / "pub_figures")
        else:
            panels(D, row, nulls[f"{v}__{a.direction}"], v, a.direction, run / "method_figures")
        print("saved", v, flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
