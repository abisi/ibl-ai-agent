"""06 -- Slide figures from the cached tables (no sorting, no metrics recomputed). 16:9, PDF + PNG, one fixed colour per
sorter. Figure 2 agreement, 3 KS4 variability, 4 curation, 5 summary; 2-4 per recording, 5 across recordings.
Usage: python 06_figures.py [--slice] [--only 2,3,4,5]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import npw  # noqa: E402

L = npw.SORTER_LABEL


def table(cfg, name, sfx):
    p = npw.out(cfg, "tables", f"{name}{sfx}.csv")
    return pd.read_csv(p) if p.exists() else None


def fig2(cfg, rec_id, sfx):
    pw = table(cfg, f"04_pairwise_{rec_id}", sfx)
    sets = table(cfg, f"04_agreement_sets_{rec_id}", sfx)
    au = table(cfg, f"04_agreement_units_{rec_id}", sfx)
    if pw is None:
        return
    pw = pw.set_index(pw.columns[0])
    names = list(pw.index)
    col = cfg["colors"]
    c = cfg["comparison"]
    fig = plt.figure(figsize=cfg["figures"]["size_in"])
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.3, 0.9], height_ratios=[1, 1.1], wspace=0.75, hspace=0.12,
                          left=0.12, right=0.98, top=0.8, bottom=0.2)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(pw.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), [L[n] for n in names], rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(len(names)), [L[n] for n in names], fontsize=13)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                v = pw.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12, color="w" if v < 0.6 else "k")
    ax.set_title("Fraction of row sorter's units\nmatched in column sorter", fontsize=14)
    # UpSet: bars of the most frequent sorter combinations + membership dots
    top = sets.sort_values("n_units", ascending=False).head(15).reset_index(drop=True)
    axb = fig.add_subplot(gs[0, 1])
    axb.bar(top.index, top.n_units, color="0.3")
    axb.set_xticks([])
    axb.set_ylabel("units")
    by_k = sets.groupby("n_sorters").n_units.sum()
    axb.set_title("Units found by k sorters\n" + ", ".join(f"k={k}: {n}" for k, n in by_k.items()), fontsize=14)
    axm = fig.add_subplot(gs[1, 1], sharex=axb)
    for x, s in enumerate(top.sorters):
        mem = s.split("+")
        ys = [names.index(m) for m in mem if m in names]
        axm.scatter([x] * len(names), range(len(names)), color="0.85", s=60)
        axm.scatter([x] * len(ys), ys, color=[col[names[y]] for y in ys], s=80)
        if len(ys) > 1:
            axm.plot([x, x], [min(ys), max(ys)], color="0.3", lw=2)
    axm.set_yticks(range(len(names)), [L[n] for n in names], fontsize=13)
    axm.invert_yaxis()
    axm.set_xticks([])
    axm.spines[["left", "bottom"]].set_visible(False)
    # SNR: units found by all sorters vs by one sorter only
    axs = fig.add_subplot(gs[:, 2])
    n = len(names)
    for i, s in enumerate(names):
        d = au[au.sorter == s]
        for k, off, alpha in ((n, -0.18, 1.0), (1, 0.18, 0.45)):
            v = d.loc[d.n_sorters == k, "snr"].dropna()
            if len(v):
                bp = axs.boxplot(v, positions=[i + off], widths=0.3, vert=False, showfliers=False,
                                 patch_artist=True)
                bp["boxes"][0].set(facecolor=col[s], alpha=alpha)
    axs.set_yticks(range(n), [L[s] for s in names], fontsize=13)
    axs.invert_yaxis()
    axs.set_xlabel("SNR")
    axs.set_title(f"SNR: found by all {n} (dark)\nvs by one only (light)", fontsize=14)
    fig.suptitle(f"Agreement between sorters, {rec_id} (same drift-corrected input; match score >= "
                 f"{c['match_score']}, delta {c['delta_time_ms']} ms)", fontsize=17, y=0.98)
    npw.save_fig(cfg, fig, f"fig2_agreement_{rec_id}{sfx}")
    plt.close(fig)


def fig3(cfg, rec_id, sfx):
    v = table(cfg, f"04_ks4_variability_{rec_id}", sfx)
    s = table(cfg, f"04_ks4_scores_{rec_id}", sfx)
    if v is None:
        return
    order = list(v.run_id)
    lab = [r.replace("kilosort4_", "").replace("kilosort4", "default (seed 5, Th 9)") for r in order]
    x = np.arange(len(order))
    fig, axs = plt.subplots(1, 3, figsize=cfg["figures"]["size_in"], gridspec_kw=dict(wspace=0.35))
    axs[0].bar(x - 0.2, v.n_units, 0.4, color=cfg["colors"]["kilosort4"], label="all units")
    axs[0].bar(x + 0.2, v.n_good, 0.4, color=cfg["colors"]["kilosort4"], alpha=0.45, label="Bombcell GOOD")
    axs[0].set_ylabel("units")
    axs[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=12)
    axs[1].bar(x, v.frac_default_matched, color=cfg["colors"]["kilosort4"])
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("fraction of default units matched")
    for i, r in enumerate(order):
        d = s[s.run_id == r].best_score
        axs[2].violinplot(d, positions=[i], widths=0.8, showmedians=True)
    axs[2].set_ylim(0, 1.02)
    axs[2].set_ylabel("best agreement score (default units)")
    for ax in axs:
        ax.set_xticks(x, lab, rotation=60, ha="right", fontsize=13)
    rep = v[v.run_id == "kilosort4_repeat"]
    det = "" if rep.empty else (f"identical rerun: {rep.frac_default_matched.iloc[0]:.2f} matched, "
                               f"{int(rep.n_units.iloc[0])} vs {int(v.n_units.iloc[0])} units")
    fig.suptitle(f"Kilosort 4 run-to-run variability, {rec_id}   {det}", fontsize=18)
    fig.subplots_adjust(top=0.82, bottom=0.32)
    npw.save_fig(cfg, fig, f"fig3_ks4_variability_{rec_id}{sfx}")
    plt.close(fig)


def fig4(cfg, rec_id, sfx):
    ret = table(cfg, "05_retained", sfx)
    sw = table(cfg, "05_sweeps", sfx)
    if ret is None:
        return
    names = [s for s in cfg["sorters"]] + [cfg["asrun"]["run_id"]]
    ret = ret[(ret.rec_id == rec_id) & ret.run_id.isin(names)]
    if ret.empty:   # slice test: only the first recording is sorted
        return
    names = [r for r in names if r in set(ret.run_id)]
    sw = sw[(sw.rec_id == rec_id) & sw.run_id.isin(names)]
    schemes = ["none", *cfg["curation"]]
    col = {**cfg["colors"]}
    fig = plt.figure(figsize=cfg["figures"]["size_in"])
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.75, top=0.8, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    tot = ret[ret.area == "all"]
    w = 0.8 / len(names)
    for i, r in enumerate(names):
        d = tot[tot.run_id == r].set_index("scheme").reindex(schemes).n
        ax.bar(np.arange(len(schemes)) + (i - len(names) / 2 + 0.5) * w, d, w, color=col[r], label=L[r])
    ax.set_xticks(range(len(schemes)), ["no curation", *[npw.SCHEME_LABEL[s] for s in cfg["curation"]]], fontsize=13,
                  rotation=25, ha="right")
    ax.set_ylabel("retained units")
    ax.set_title("Total")
    # per area, under the user's own criteria (Bombcell GOOD alone keeps too few units to compare areas)
    ax2 = fig.add_subplot(gs[0, 1])
    pa = ret[(ret.scheme == "axel") & (ret.area != "all")]
    areas = pa.groupby("area").n.sum().sort_values(ascending=False).index[:8]
    for i, r in enumerate(names):
        d = pa[pa.run_id == r].set_index("area").reindex(areas).n.fillna(0)
        ax2.bar(np.arange(len(areas)) + (i - len(names) / 2 + 0.5) * w, d, w, color=col[r])
    ax2.set_xticks(range(len(areas)), areas, rotation=30, ha="right", fontsize=13)
    ax2.set_ylabel(f"{npw.SCHEME_LABEL['axel']} units", fontsize=14)
    ax2.set_title("Per area (IBL alignment)")
    for k, (metric, s) in enumerate(cfg["sweeps"].items()):
        ax3 = fig.add_subplot(gs[1, k])
        for r in names:
            d = sw[(sw.run_id == r) & (sw.metric == metric)]
            ax3.plot(d.threshold, d.n, color=col[r], lw=2.5)
        for scheme, t in s["marks"].items():
            ax3.axvline(t, color="k", ls="--", lw=1.2)
            ax3.text(t, ax3.get_ylim()[1], f" {npw.SCHEME_LABEL[scheme]}", rotation=90, va="top", fontsize=12)
        ax3.set_xscale("log")
        ax3.set_xlabel(f"{metric} threshold (keep <)")
        ax3.set_ylabel("retained units")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"Curation, {rec_id}", fontsize=18, y=0.99)
    npw.save_fig(cfg, fig, f"fig4_curation_{rec_id}{sfx}")
    plt.close(fig)


def fig5(cfg, sfx):
    ret = table(cfg, "05_retained", sfx)
    if ret is None:
        return
    names = [s for s in cfg["sorters"]] + [cfg["asrun"]["run_id"]]
    ret = ret[(ret.area == "all") & ret.run_id.isin(names)]
    schemes = ["none", *cfg["curation"]]
    recs = [r for r in cfg["recordings"] if r in set(ret.rec_id)]
    fig, axs = plt.subplots(1, len(recs), figsize=cfg["figures"]["size_in"], sharey=False)
    for ax, rec_id in zip(np.atleast_1d(axs), recs):
        d = ret[ret.rec_id == rec_id]
        for j, sc in enumerate(schemes):
            for i, r in enumerate(names):
                v = d[(d.scheme == sc) & (d.run_id == r)].n
                if len(v):
                    ax.scatter(j + (i - len(names) / 2) * 0.09, v, color=cfg["colors"][r], s=90,
                               marker="D" if r == cfg["asrun"]["run_id"] else "o", label=L[r] if j == 0 else None)
            v = d[d.scheme == sc].n
            if len(v):
                ax.text(j, v.max() * 1.04, f"{v.min()}-{v.max()}", ha="center", fontsize=12)
        ax.set_xticks(range(len(schemes)), ["none", *[npw.SCHEME_LABEL[s] for s in cfg["curation"]]], rotation=40,
                      ha="right", fontsize=13)
        ax.set_title(rec_id)
        ax.set_ylim(0, None)
    np.atleast_1d(axs)[0].set_ylabel("units")
    h, lb = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Unit count across pipeline choices (sorter x curation), drift-corrected input", fontsize=18)
    fig.subplots_adjust(top=0.76, bottom=0.25, wspace=0.3)
    npw.save_fig(cfg, fig, f"fig5_summary{sfx}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true")
    ap.add_argument("--only", default="2,3,4,5")
    a = ap.parse_args()
    cfg = npw.load_config()
    npw.style(cfg)
    sfx = "_slice" if a.slice else ""
    only = set(a.only.split(","))
    for rec_id in cfg["recordings"]:
        if "2" in only:
            fig2(cfg, rec_id, sfx)
        if "3" in only:
            fig3(cfg, rec_id, sfx)
        if "4" in only:
            fig4(cfg, rec_id, sfx)
    if "5" in only:
        fig5(cfg, sfx)


if __name__ == "__main__":
    main()
