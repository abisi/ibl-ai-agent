"""07 -- Spike detection and unit yield across sorters (user 2026-11-17), all units, no curation. Pipeline container.

Per recording: units and spikes of every sorter, split by how many sorters found the unit (only this sorter / some /
all; multi-sorter agreement as in 04), and spike-level agreement: fraction of sorter i's spikes that coincide
(+-delta_time) with a spike of the unit Hungarian-matched to it in sorter j (unmatched units count as undetected).
Matching parameters: config `comparison` (same as 04). Sorters missing on disk are left out (noted on the figure).
Outputs tables/07_units_<rec>.csv (one row per sorter unit: n_spikes, n_sorters), 07_spike_agreement_<rec>.csv
(sorter x other), figures/fig6_yield_<rec>.{pdf,png}, fig6_yield_summary.{pdf,png}.
Usage: python 07_detection_yield.py --rec <rec_id>    (compute, one recording; Kuma, sbatch/yield.sbatch, which copies
                                                        the tables to the NAS combined_results_ks4/npworkshop-sorting)
       NPW_CONFIG=config_haas.yaml python 07_detection_yield.py --plot   (figures from the NAS tables, haas;
                                                        config_haas.yaml = config.yaml with out_root on the NAS)
"""
import argparse
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import npw  # noqa: E402

L = npw.SORTER_LABEL
GROUPS = (("all", "found by all sorters", 1.0), ("some", "found by 2+ (not all)", 0.55),
          ("only", "found by this sorter only", 0.2))


def compute(cfg, rec_id):
    import spikeinterface.comparison as sc
    c = cfg["comparison"]
    kw = dict(delta_time=c["delta_time_ms"], match_score=c["match_score"], chance_score=c["chance_score"])
    names = [s for s in cfg["sorters"] if (npw.sorting_dir(cfg, rec_id, s) / "sorting").exists()]
    sortings = {s: npw.load_sorting(cfg, rec_id, s) for s in names}
    n_sp = {s: sortings[s].count_num_spikes_per_unit(outputs="dict") for s in names}
    rows = []
    for i, j in itertools.combinations(names, 2):
        cmp = sc.compare_two_sorters(sortings[i], sortings[j], sorting1_name=i, sorting2_name=j, **kw)
        ev = cmp.match_event_count
        for a, b, m in ((i, j, cmp.hungarian_match_12), (j, i, cmp.hungarian_match_21)):
            m = m[m != -1]
            hit = sum(int(ev.at[u, v] if a == i else ev.at[v, u]) for u, v in m.items())
            rows.append(dict(rec_id=rec_id, sorter=a, other=b, n_units=len(n_sp[a]), units_matched=len(m),
                             spikes=sum(n_sp[a].values()), spikes_matched=hit))
        print(rec_id, i, j, flush=True)
    pd.DataFrame(rows).to_csv(npw.out(cfg, "tables", f"07_spike_agreement_{rec_id}.csv"), index=False)
    mc = sc.compare_multiple_sorters([sortings[s] for s in names], name_list=names, **kw)
    agr = mc.get_agreement_sorting(minimum_agreement_count=1)
    k_of = {}
    for au in agr.unit_ids:
        k = int(agr.get_unit_property(au, "agreement_number"))
        for s, u in agr.get_unit_property(au, "unit_ids").items():
            k_of[(s, str(u))] = k
    units = pd.DataFrame([dict(rec_id=rec_id, sorter=s, unit_id=u, n_spikes=int(n), n_sorters=k_of.get((s, str(u)), 1),
                               n_sorters_present=len(names)) for s in names for u, n in n_sp[s].items()])
    units.to_csv(npw.out(cfg, "tables", f"07_units_{rec_id}.csv"), index=False)
    print(rec_id, units.groupby("sorter").agg(units=("unit_id", "size"), spikes=("n_spikes", "sum")).to_string())


def group(u):
    n = u.n_sorters_present.iloc[0]
    return np.where(u.n_sorters == 1, "only", np.where(u.n_sorters == n, "all", "some"))


def stacked(ax, u, names, col, value, scale, xlabel, title):
    for y, s in enumerate(names):
        d = u[u.sorter == s]
        left = 0.0
        for g, _, alpha in GROUPS:
            v = (d.loc[d.grp == g, value].sum() if value != "units" else (d.grp == g).sum()) / scale
            ax.barh(y, v, left=left, color=col[s], alpha=alpha, edgecolor=col[s], lw=1)
            left += v
        tot = (d[value].sum() if value != "units" else len(d)) / scale
        ax.text(left, y, f" {tot:.0f}" if scale == 1 else f" {tot:.1f}", va="center", fontsize=13)
    ax.set_yticks(range(len(names)), [L[s] for s in names], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.15)


def fig_rec(cfg, rec_id):
    u_p = npw.out(cfg, "tables", f"07_units_{rec_id}.csv")
    if not u_p.exists():
        return
    u = pd.read_csv(u_p)
    u["grp"] = group(u)
    pw = pd.read_csv(npw.out(cfg, "tables", f"07_spike_agreement_{rec_id}.csv"))
    names = [s for s in cfg["sorters"] if s in set(u.sorter)]
    missing = [L[s] for s in cfg["sorters"] if s not in names]
    col = cfg["colors"]
    fig = plt.figure(figsize=cfg["figures"]["size_in"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.6, left=0.16, right=0.96, top=0.78, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    stacked(ax, u, names, col, "units", 1, "units", "Unit yield")
    ax = fig.add_subplot(gs[1])
    stacked(ax, u, names, col, "n_spikes", 1e6, "spikes (millions)", "Spikes detected")
    ax.set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=a) for _, _, a in GROUPS]
    fig.legend(handles, [lab for _, lab, _ in GROUPS], loc="lower center", ncol=3, frameon=False, fontsize=14,
               bbox_to_anchor=(0.36, 0.0))
    ax = fig.add_subplot(gs[2])
    m = pw.assign(f=pw.spikes_matched / pw.spikes).pivot(index="sorter", columns="other", values="f")
    m = m.reindex(index=names, columns=names)
    ax.imshow(m.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                v = m.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12, color="w" if v < 0.6 else "k")
    ax.set_xticks(range(len(names)), [L[n] for n in names], rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(len(names)), [L[n] for n in names], fontsize=13)
    ax.set_title("Fraction of row sorter's spikes\nalso found by column sorter", fontsize=16)
    c = cfg["comparison"]
    sub = f"all units, no curation; matched units (score >= {c['match_score']}), spikes within {c['delta_time_ms']} ms"
    if missing:
        sub += f"; not yet sorted: {', '.join(missing)}"
    fig.suptitle(f"Spike detection and yield across sorters, {rec_id}\n", fontsize=19, y=0.97)
    fig.text(0.5, 0.88, sub, ha="center", fontsize=13, color="0.35")
    npw.save_fig(cfg, fig, f"fig6_yield_{rec_id}")
    plt.close(fig)


def fig_summary(cfg):
    us = [pd.read_csv(p) for r in cfg["recordings"] if (p := npw.out(cfg, "tables", f"07_units_{r}.csv")).exists()]
    if not us:
        return
    u = pd.concat(us)
    u["grp"] = np.concatenate([group(d) for _, d in u.groupby("rec_id", sort=False)])
    names = [s for s in cfg["sorters"] if s in set(u.sorter)]
    recs = list(dict.fromkeys(u.rec_id))
    col = cfg["colors"]
    per = pd.DataFrame([dict(rec_id=r, sorter=s, units=len(d), only_units=(d.grp == "only").mean() * 100,
                             only_spikes=d.loc[d.grp == "only", "n_spikes"].sum() / d.n_spikes.sum() * 100)
                        for (r, s), d in u.groupby(["rec_id", "sorter"])]).set_index(["rec_id", "sorter"])
    fig, axs = plt.subplots(1, 3, figsize=cfg["figures"]["size_in"], gridspec_kw=dict(wspace=0.45, left=0.09,
                                                                                    right=0.98, top=0.8, bottom=0.3))
    marks = ("o", "s", "^")
    for ax, key, lab in zip(axs, ("units", "only_units", "only_spikes"),
                            ("units", "% of units found by this sorter only",
                             "% of spikes in units found\nby this sorter only")):
        for x, s in enumerate(names):
            v = [per.loc[(r, s), key] if (r, s) in per.index else np.nan for r in recs]
            ax.bar(x, np.nanmean(v), color=col[s], alpha=0.35, width=0.7)
            for r, val, mk in zip(recs, v, marks):
                ax.scatter(x, val, color=col[s], marker=mk, s=70, zorder=3, edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(names)), [L[s] for s in names], rotation=40, ha="right")
        ax.set_ylabel(lab)
    handles = [plt.Line2D([], [], marker=mk, ls="", color="0.4", markeredgecolor="k", ms=9) for mk in marks[:len(recs)]]
    fig.legend(handles, recs, frameon=False, loc="lower center", ncol=len(recs), fontsize=14)
    n_by = u.groupby("rec_id").n_sorters_present.first()
    note = ", ".join(f"{r}: {n_by[r]} sorters" for r in recs)
    fig.suptitle("Unit yield and sorter-specific detections (all units, no curation)", fontsize=19, y=0.97)
    fig.text(0.5, 0.88, f"bars = mean over recordings, markers = recordings ({note})", ha="center", fontsize=13,
             color="0.35")
    npw.save_fig(cfg, fig, "fig6_yield_summary")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    cfg = npw.load_config()
    if a.rec:
        compute(cfg, a.rec)
    if a.plot:
        npw.style(cfg)
        for rec_id in cfg["recordings"]:
            fig_rec(cfg, rec_id)
        fig_summary(cfg)


if __name__ == "__main__":
    main()
