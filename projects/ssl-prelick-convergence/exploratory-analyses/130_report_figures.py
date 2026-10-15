"""130 -- Report figures for ssl-prelick-convergence (user 2026-10-15: report figures assembled from the latest COSYNE panels and
re-arranged into the report's storyline; every dimension shown at each level). Plot only; panels and statistics are those of 119
(three-row COSYNE figure), 087 (CD metrics) and 088 (panel helpers), on the same 1.2-in square grid.
  report_fig2_single_neurons   a population PSTHs (all tested neurons, 2 x 2: cohort x stage); b example non-sensory reward-lick neuron
                               (stimulus- and lick-aligned); c ROC similarity index of non-sensory reward-lick neurons, day 0 vs expert;
                               d per area (omnibus ANOVA + post hoc)
  report_fig3_population_cd    a projections on the CD over time (WH / AH / SL, 2 x 2); b class projections, day 0 vs expert tests per
                               class; c d_WS; d theta and rho; e per area: d_WS change (omnibus ANOVA + post hoc)
Outputs: across_days/<ref>/publication/learners/report_fig{2,3}_*.{png,pdf,svg}; stats in report_figures_stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../130_report_figures.py [--n-proc 30]
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m119 = importlib.import_module("119_cosyne_three_rows")
m88, m62, m83, m51 = m119.m88, m119.m62, m119.m83, m119.m51
S, GAP, L0, A, TOP, B, T = 1.2, 0.5, 0.45, 0.6, 0.12, 0.42, 0.62
EXTRA = {1: 0.35}                    # room for the example-neuron legends (Fig. 2b)


def grid(plt, ncol, area_last=True, extra=None):
    extra = extra or {}
    xs = [L0]
    for c in range(1, ncol - 1):
        xs.append(xs[-1] + S + GAP + extra.get(c - 1, 0))
    xs.append(xs[-1] + S + GAP + extra.get(ncol - 2, 0) + (A if area_last else 0))
    W_, H_ = xs[-1] + S + 0.1, TOP + T + S + B
    fig = plt.figure(figsize=(W_, H_)); y0 = B
    box = lambda c: [xs[c] / W_, y0 / H_, S / W_, S / H_]
    spec = lambda c: fig.add_gridspec(1, 1, left=xs[c] / W_, right=(xs[c] + S) / W_, bottom=y0 / H_, top=(y0 + S) / H_)[0]
    return fig, box, spec


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    plt = m62.setup(); rng = np.random.default_rng(0); rows = []
    L = m119.load_data(a.n_proc, rows)
    out = m51.OUTROOT / "publication" / "learners"
    rc = {"font.size": 5.2, "axes.titlesize": 5.2, "axes.labelsize": 5.0, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6,
          "legend.fontsize": 4.4, "svg.fonttype": "none"}
    with plt.rc_context(rc):
        # ---------------- Figure 2: single neurons
        fig, box, spec = grid(plt, 4, extra=EXTRA)
        Wc = L["Wc"]; LS = []
        tested = (Wc["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wc["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
        ib = m83.psth_insets(fig, spec(0), Wc, L["PB"], L["tc"], tested, "a", rows)
        for x in ib:
            x.set_box_aspect(1)
        LS.append(ib[0])
        ex = m119.pick_clean(Wc)
        ig = m119.example_neuron(fig, spec(1), ex, rng, wspace=0.75, stacked=True, legend_out=True); LS.append(ig[0])
        ig[0].axvspan(5, 35, color="#C6DBEF", alpha=0.7, lw=0, zorder=0)
        SIU = L["SIU"]
        si = SIU[(SIU.session_id == ex.session_id) & (SIU.electrode_group == str(ex.electrode_group)) & (SIU.cluster_id == str(ex.cluster_id))].SI
        ig[0].set_title(f"Non-sensory reward-lick neuron\n(R+ expert, {ex.get('area_acronym_custom', ex.get('area_group', ''))}"
                        f"{f', SI = {si.iloc[0]:+.2f}' if len(si) else ''})", fontsize=4.8, pad=3)
        ax = fig.add_axes(box(2)); LS.append(ax)
        m88.mean_panel2(ax, L["S_ns"], "si", "SI (−1 SL-like … +1 AH-like)", rng, "130 si", "Non-sensory reward-lick neurons\n"
                        "ROC similarity index SI", ref=0)
        ax = fig.add_axes(box(3)); LS.append(ax)
        m119.area_forest(ax, L["AD"]["i"], L["ORD"]["i"], L["AQ"]["i"], "Δ SI (expert − day 0)", "Per area: SI (non-sensory)", rng, rows, "f2d", gcol)
        m62.letter_row(fig, LS, "abcd", dx_in=0.3, dy_in=0.12)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"report_fig2_single_neurons.{ext}", dpi=300)
        plt.close(fig)
        # ---------------- Figure 3: population CD
        fig, box, spec = grid(plt, 5)
        LS = []
        ip = L["m89"].proj_insets(fig, spec(0), L["TP"], rows)
        for x in ip:
            x.set_box_aspect(1)
        LS.append(ip[0])
        P = L["P"]
        ax = fig.add_axes(box(1)); m88.proj_levels(ax, P); m119.level_stats(ax, P, rows); LS.append(ax)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], color=m88.CL[k_], lw=1.2, marker="o", ms=2.4) for k_ in ("WH", "AH", "FA")], ["WH", "AH", "SL"], frameon=False,
                  fontsize=4.2, loc="center right", handlelength=1.2)
        ax = fig.add_axes(box(2)); LS.append(ax)
        m88.mean_panel2(ax, P, "d_WS", "d_WS = p_WH − p_SL (per neuron)", rng, "130 dWS", "Whisker hits on the CD", ref=0)
        sub = spec(3).subgridspec(1, 2, wspace=1.1); ah = [fig.add_subplot(sub[i]) for i in range(2)]; LS.append(ah[0])
        m88.small_panel(ah[0], P, "theta", "θ (°)", rng, "Angle θ")
        m88.small_panel(ah[1], P, "rho_v", "ρ (per neuron)", rng, "Length ρ")
        ax = fig.add_axes(box(4)); LS.append(ax)
        m119.area_forest(ax, L["AD"]["m"], L["ORD"]["m"], L["AQ"]["m"], "Δ WH − SL projection\n(per neuron; expert − day 0)",
                         "Per area: WH position on the CD", rng, rows, "f3e", gcol)
        m62.letter_row(fig, LS, "abcde", dx_in=0.3, dy_in=0.12)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"report_fig3_population_cd.{ext}", dpi=300)
        plt.close(fig)
    R = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith(("130", "088"))])
    R.to_csv(out / "report_figures_stats.csv", index=False)
    print("ALL DONE", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    main(ap.parse_args())
