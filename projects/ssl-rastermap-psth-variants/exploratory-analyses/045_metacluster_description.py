"""045 -- Metacluster description for Figure 2 (user 2026-11-12): selected metaclusters of one rastermap variant, one row
each, A4 width (183 mm). Close to pub_metacluster_psth_summary.

Per row: a one-line header (metacluster number, data-driven functional name from
population_matrix_summary.family_names_psth, n neurons, n mice), then left to right:
  PSTHs     three square sub-panels sharing the y axis: passive (whisker / auditory, pre dashed, post solid), active
            stimulus-aligned (whisker hit, whisker miss light brown, auditory hit), lick-aligned (spontaneous lick, whisker
            hit, auditory hit); whisker traces drawn above auditory ones. Even trials, Hz baseline-subtracted (the
            rastermap input); per mouse the mean over its neurons in the metacluster (mice with >= MIN_NEURONS_MOUSE),
            then mean +- s.e.m. over mice (user 2026-11-12: s.e.m. over mice, not neurons)
  areas     polar plot of log2(metacluster fraction / dataset fraction) per area group, wedges from the black circle
            (= the dataset fraction, i.e. the area represented as in the recordings) outwards (enriched) or inwards
            (depleted); allen_utils custom area-group colours, minor groups pooled in Other; faded = n.s. (Fisher exact,
            metacluster vs rest, BH over area groups and metaclusters; values in _table.csv)
Layout: a third of an A4 page (user 2026-11-12): 12.6 mm square panels, rows 16 mm apart, x labels on the bottom row.
  ROC       fraction of the metacluster's neurons significant per ROC category (day-0 ROC,
            population_matrix_summary.ROC_CATEGORIES; gated as a fraction of the decision neurons); tick, all neurons;
            * Fisher exact vs the rest, BH. "Passive change" = learning_modulated: whisker_ or auditory_pre_vs_post_learning
            significant (roc_utils_new: passive-post vs passive-pre trials, post-stimulus window 5-35 ms, each trial minus
            its own baseline [-1, -0.015] s scaled to the window, rates)
Metaclusters: population_matrix_summary.metaclustering defaults (contiguous, 1 - r cut 0.40), numbered in rastermap order.
Output: <variant folder>/metacluster_description/metacluster_description.{png,pdf,svg} + _table.csv (with variant,
method_dir, order) + _psth_mice.csv (per-mouse PSTH means).
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. .venv/bin/python \
  ~/code/ibl-ai-agent/projects/ssl-rastermap-psth-variants/exploratory-analyses/045_metacluster_description.py \
  [--variant V] [--order 23 18 14 15 12]
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import rastermap_psth.rastermap_utils as rmu  # noqa: E402
from rastermap_psth import population_matrix_summary as pms  # noqa: E402

ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CONFIG = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")
VARIANT = "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin"   # user 2026-11-12
ORDER = [23, 18, 14, 12]   # user 2026-11-12 (15 removed)                                                              # user 2026-11-12
MIN_NEURONS_MOUSE = 3
N_COLS, COL_GAP = 2, 5.0   # two metaclusters per row, row-major (user 2026-11-12)
W, A, NS, WMISS = "#f7b519", "#2c2cdb", "#262626", "#c8a27a"   # task colours; whisker miss light brown (user)
Z = {W: 3, WMISS: 3, A: 2, NS: 2.5}                            # whisker above auditory (user)
PSTH_GROUPS = [  # (title, [(condition label, colour, line style, legend)])
    ("Passive", [("Whisker pre", W, "--", "W pre"), ("Whisker post", W, "-", "W post"),
                 ("Auditory pre", A, "--", "A pre"), ("Auditory post", A, "-", "A post")]),
    ("Active, stimulus-aligned", [("Whisker hit", W, "-", "W hit"), ("Whisker miss", WMISS, "-", "W miss"),
                                  ("Auditory hit", A, "-", "A hit")]),
    ("Active, lick-aligned", [("Spont. lick", NS, "-", "Spont. lick"), ("Whisker hit (lick)", W, "-", "W hit"),
                              ("Auditory hit (lick)", A, "-", "A hit")]),
]
ROC_SHOWN = ["whisker_sensory", "auditory_sensory", "lick_responsive", "decision", "gated_decision"]
# user 2026-11-12 (half the height): stimulus-responsive and passive change dropped from the panel (still in _table.csv)
ROC_ALL = ["stim_responsive", "whisker_sensory", "auditory_sensory", "lick_responsive", "decision", "gated_decision",
           "learning_modulated"]
ROC_SHORT = {"stim_responsive": "Stimulus", "whisker_sensory": "Whisker", "auditory_sensory": "Auditory",
             "lick_responsive": "Lick", "decision": "Decision", "gated_decision": "Gated / dec.",
             "learning_modulated": "Passive change"}
ROC_VIVID = {"stim_responsive": "#0077bb", "whisker_sensory": "#f7b519", "auditory_sensory": "#2c2cdb",
             "lick_responsive": "#9b30d9", "decision": "#e8222e", "gated_decision": "#b5005b",
             "learning_modulated": "#00a087"}   # user 2026-11-12: more vivid than pms.ROC_COLORS


def method_dir(v):
    return sorted((ROOT / v).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]


def fisher(k, n, K, N):
    """metacluster: k of n; rest: K - k of N - n."""
    return fisher_exact([[k, n - k], [K - k, (N - n) - (K - k)]])[1] if n and N - n else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--order", type=int, nargs="+", default=ORDER)
    a = ap.parse_args()
    md = method_dir(a.variant)
    prov = json.load(open(ROOT / a.variant / "variant_provenance.json"))
    cfg = rmu.load_cfg(CONFIG, **prov["config_overrides"])
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
    R = pms.load_rastermap(md)
    roc = pms.build_roc_categories(md, R["unit_ids"])
    meta = pms.metaclustering(md)
    names, name_tab = pms.family_names_psth(R, meta["families"])
    fam = np.array([meta["families"][k] for k in R["labels"]])
    top = list(a.order)
    missing = [f for f in top if f not in set(fam)]
    assert not missing, f"metaclusters {missing} not in this variant (1..{fam.max()})"
    labels_c = list(rmu.COND_LABELS)
    off = np.r_[0, np.cumsum(R["n_bins_list"])]
    major, acol = pms._area_categories(R)
    groups = major + ["Other"]
    acol["Other"] = pms.OTHER_AREA_COLOR
    area = np.where(np.isin(R["area_group_arr"], major), R["area_group_arr"], "Other")
    mouse = R["mouse_arr"]
    N = len(fam)

    rows, psth_rows = [], []
    for f in top:
        m = fam == f
        n = int(m.sum())
        for g in groups:
            k, K = int((area[m] == g).sum()), int((area == g).sum())
            rows.append(dict(metacluster=f, block="area", item=g, frac=k / n, frac_all=K / N,
                             log2_ratio=np.log2((k / n) / (K / N)) if k and K else np.nan, p=fisher(k, n, K, N)))
        matched = roc["roc_matched"].values.astype(bool)
        for c in ROC_ALL:
            s = roc[f"{c}__sig"].values == 1
            base = matched.copy()
            if c == "gated_decision":   # within decision neurons, as the matrix column
                base &= roc["decision__sig"].values == 1
            k, nn = int((s & base & m).sum()), int((base & m).sum())
            K, NN = int((s & base).sum()), int(base.sum())
            rows.append(dict(metacluster=f, block="roc", item=c, frac=k / nn if nn else np.nan, frac_all=K / NN,
                             n_base=nn, p=fisher(k, nn, K, NN)))
    tab = pd.DataFrame(rows)
    for b in ("area", "roc"):
        sel = (tab.block == b) & tab.p.notna()
        tab.loc[sel, "q_bh"] = multipletests(tab.loc[sel, "p"], method="fdr_bh")[1]
    tab["variant"], tab["method_dir"], tab["order"] = a.variant, str(md), " ".join(map(str, top))
    tab["name"] = tab.metacluster.map(names)

    pms.use_pub_font()
    fs = 4.0
    plt.rcParams.update({"font.size": fs, "axes.linewidth": 0.4, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 1.0,
                         "ytick.major.size": 1.0, "xtick.labelsize": fs - 0.5, "ytick.labelsize": fs - 0.5,
                         "xtick.major.pad": 1, "ytick.major.pad": 1})
    # layout in mm (user 2026-11-12: a third of an A4 page, no white space between rows, square panels)
    # user 2026-11-12: half the height, one row per metacluster: label block on the left of each row, ~7 mm panels
    S, ROW, HDR, TOP, BOT = 9.4, 10.6, 0.0, 4.4, 9.4
    LAB = 15.0                                                # label block width
    x_p = [LAB + 5.5, LAB + 5.5 + S + 1.4, LAB + 5.5 + 2 * (S + 1.4)]   # PSTHs (shared y: no tick labels on 2 and 3)
    x_pol = x_p[2] + S + 3.0
    x_roc = x_pol + S + 10.0
    # user 2026-11-12: half the height -> metaclusters in N_COLS columns of rows (left column first)
    n_rows = int(np.ceil(len(top) / N_COLS))
    block_w = x_roc + S + 3.0
    width_mm = N_COLS * block_w + (N_COLS - 1) * COL_GAP
    height_mm = TOP + ROW * n_rows + BOT
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)

    def add_ax(x, y_top, w, h, **kw):   # mm, y from the top
        return fig.add_axes([x / width_mm, 1 - (y_top + h) / height_mm, w / width_mm, h / height_mm], **kw)

    theta = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
    w_th = 2 * np.pi / len(groups)
    xlims = {}
    for r, f in enumerate(top):
        m = fam == f
        mice = [mm for mm in np.unique(mouse[m]) if (m & (mouse == mm)).sum() >= MIN_NEURONS_MOUSE]
        rr, col = divmod(r, N_COLS)   # row-major: two metaclusters per row
        dx = col * (block_w + COL_GAP)
        y0 = TOP + rr * ROW + HDR
        last = rr == n_rows - 1 or r + N_COLS >= len(top)   # bottom panel of its column: x labels
        fig.text((dx + 0.5) / width_mm, 1 - (y0 + 0.2) / height_mm, f"Metacluster {f}", fontsize=fs + 0.2, va="top")
        fig.text((dx + 0.5) / width_mm, 1 - (y0 + 2.4) / height_mm, names.get(f, ""), fontsize=fs - 0.4, va="top",
                 color="0.2")
        fig.text((dx + 0.5) / width_mm, 1 - (y0 + 4.4) / height_mm, f"{int(m.sum()):,} neurons, {len(mice)} mice",
                 fontsize=fs - 0.9, va="top", color="0.4")
        axs, lo, hi = [], np.inf, -np.inf
        for j, (title, conds) in enumerate(PSTH_GROUPS):
            ax = add_ax(dx + x_p[j], y0, S, S, sharey=axs[0] if axs else None); axs.append(ax)
            for cond, color, ls, lab in conds:
                c = labels_c.index(cond)
                t = R["t_ctrs"][c]
                per_mouse = np.vstack([np.nanmean(R["X"][m & (mouse == mm)][:, off[c]:off[c + 1]], 0) for mm in mice])
                mu = np.nanmean(per_mouse, 0)
                se = np.nanstd(per_mouse, 0, ddof=1) / np.sqrt(np.isfinite(per_mouse).sum(0))
                psth_rows += [dict(metacluster=f, condition=cond, mouse_id=mm, time_s=tt, rate_hz=v)
                              for mm, row in zip(mice, per_mouse) for tt, v in zip(t, row)]
                ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.2, lw=0, zorder=Z[color] - 0.5)
                ax.plot(t, mu, color=color, ls=ls, lw=0.6, label=lab, zorder=Z[color])
                lo, hi = min(lo, np.nanmin(mu - se)), max(hi, np.nanmax(mu + se))
                xlims[j] = (t.min(), t.max())
            ax.set_xlim(*xlims[j])
            ax.axvline(0, color="0.5", lw=0.35, ls=":", zorder=0)
            ax.axhline(0, color="0.85", lw=0.3, zorder=0)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            if last:
                ax.set_xlabel("Time (s)", labelpad=0.3)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel("Δ Hz", labelpad=0.3)
            else:   # shared y axis: no label (user 2026-11-12)
                ax.tick_params(labelleft=False)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if rr == 0:
                ax.set_title({"Passive": "Passive", "Active, stimulus-aligned": "Stimulus",
                              "Active, lick-aligned": "Lick"}[title], fontsize=fs, pad=1.2)
        pad = 0.08 * (hi - lo)
        axs[0].set_ylim(lo - pad, hi + pad)
        # areas: polar log2 ratio to the dataset (circle = dataset fraction); faded = n.s.
        d = tab[(tab.metacluster == f) & (tab.block == "area")].set_index("item").reindex(groups)
        ax = add_ax(dx + x_pol, y0, S, S, projection="polar")
        L = max(0.5, np.ceil(2 * np.nanmax(np.abs(d.log2_ratio))) / 2)   # per-plot scale (user 2026-11-12)
        v = d.log2_ratio.fillna(0).to_numpy()
        sig = (d.q_bh < 0.05).to_numpy()
        ax.bar(theta, v, bottom=L, width=w_th * 0.9, zorder=2, lw=0,
               color=[matplotlib.colors.to_rgba(acol[g], 1.0 if sg else 0.35) for g, sg in zip(groups, sig)])
        ring = np.linspace(0, 2 * np.pi, 200)
        ax.plot(ring, np.full_like(ring, L), color="k", lw=0.5, zorder=3)
        ax.set_ylim(0, 2 * L)
        ax.set_yticks([0, 2 * L], [])
        ax.text(1.0, 0.0, f"±{L:g}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs - 1, color="0.35")
        ax.set_xticks([]); ax.grid(lw=0.25, color="0.85"); ax.spines["polar"].set_linewidth(0.35)
        if rr == 0:
            ax.set_title("Areas", fontsize=fs, pad=1.2)
        # ROC fractions (square)
        ax = add_ax(dx + x_roc, y0, S, S)
        d = tab[(tab.metacluster == f) & (tab.block == "roc")].set_index("item").reindex(ROC_SHOWN)
        y = np.arange(len(ROC_SHOWN))
        ax.barh(y, d.frac, color=[ROC_VIVID[c] for c in ROC_SHOWN], height=0.75, lw=0)
        ax.scatter(d.frac_all, y, marker="|", s=4, color="k", lw=0.5, zorder=3)
        for i, c in enumerate(ROC_SHOWN):
            q = d.q_bh[c]
            if q < 0.05:   # Fisher exact vs the other neurons, BH (user 2026-11-12: annotate the statistics)
                ax.text(max(d.frac[c], d.frac_all[c]) + 0.03, i, "***" if q < 0.001 else "**" if q < 0.01 else "*",
                        va="center", fontsize=fs - 0.5)
        ax.set_yticks(y, [ROC_SHORT[c] for c in ROC_SHOWN], fontsize=fs - 0.8)
        ax.set_ylim(len(ROC_SHOWN) - 0.4, -0.6)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        if last:
            ax.set_xlabel("Fraction sig.", labelpad=0.3)
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(axis="y", length=0, pad=0.8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if rr == 0:
            ax.set_title("ROC", fontsize=fs, pad=1.2)
    free = n_rows * N_COLS > len(top)   # empty slot at the bottom of the last column: the area key goes there
    keyax = (add_ax((N_COLS - 1) * (block_w + COL_GAP) + 4, TOP + (n_rows - 1) * ROW + HDR + 5, block_w - 8, S - 2) if free
             else add_ax(1, height_mm - 2.6, width_mm - 2, 2.4)); keyax.axis("off")
    from matplotlib.lines import Line2D
    trax = add_ax(1, height_mm - 6.2, width_mm - 2, 3.6); trax.axis("off")   # trace key (user: no legends on top)
    trax.legend(handles=[Line2D([], [], color=W, ls="--", lw=0.6, label="W pre"), Line2D([], [], color=W, lw=0.6, label="W post / hit"),
                         Line2D([], [], color=WMISS, lw=0.6, label="W miss"), Line2D([], [], color=A, ls="--", lw=0.6, label="A pre"),
                         Line2D([], [], color=A, lw=0.6, label="A post / hit"), Line2D([], [], color=NS, lw=0.6, label="Spont. lick"),
                         Line2D([], [], color="k", lw=0, marker="|", ms=2.5,
                                label="ROC: all neurons; * / ** / *** q < 0.05 / 0.01 / 0.001 (Fisher vs other neurons, BH)"),
                         Patch(color="none", label="Areas: log2(fraction / dataset), circle = dataset, scale per plot, faded = n.s.")],
                loc="center", ncol=4, fontsize=fs - 0.8, frameon=False, handlelength=1.2, columnspacing=0.6,
                handletextpad=0.25)
    keyax.legend(handles=[Patch(color=acol[g], label=g) for g in groups], loc="center", ncol=4 if free else 7, fontsize=fs - 0.9,
                 frameon=False, handlelength=0.8, handleheight=0.8, columnspacing=0.6, labelspacing=0.1,
                 handletextpad=0.3)
    out = ROOT / a.variant / "metacluster_description"
    out.mkdir(exist_ok=True)
    stem = out / "metacluster_description"
    rmu._save(fig, stem, dpi=600)
    tab.to_csv(f"{stem}_table.csv", index=False)
    pd.DataFrame(psth_rows).assign(variant=a.variant).to_csv(f"{stem}_psth_mice.csv", index=False)
    name_tab.to_csv(out / "metacluster_names_all.csv", index=False)
    print(stem)


if __name__ == "__main__":
    main()
