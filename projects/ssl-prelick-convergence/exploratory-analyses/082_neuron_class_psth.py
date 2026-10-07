"""082 -- First-lick-aligned PSTHs of converging neurons by stimulus-defined class, across learning stages (user 2026-10-08).

Units and classes from 081 (units.parquet): good / mua (stable run: stable only), pre-lick rate >= 0.1 Hz; classes from the
existing stimulus-aligned ROC analyses (045 roc_long, joined on session_id, electrode_group, cluster_id; all analysed units
match): sensory = significant in whisker_active or auditory_active (user 2026-10-08); lick-gated = not sensory,
whisker_choice significant; non-sensory = neither. Pre-lick ROC (051, all events): reward-lick neurons = AH > SL
significant; converging = reward-lick and WH > SL significant; reward-lick-only = reward-lick, WH vs SL not significant.
PSTHs (062 load_psth_prestart): first-lick aligned, rate minus the unit's pre-trial baseline per class, 10-ms bins, 3-bin
boxcar; mean over units within session, then mean +- s.e.m. over sessions (062 mouse_psth); shaded: pre-lick window.
Layout (square panels): rows 1-3 = converging neurons of each class, columns = R+ day 0, R+ expert, R- day 0, R- expert;
row 4 = reward-lick-only neurons (all classes; whisker hits stay with spontaneous licks); right column = converging
fraction among reward-lick neurons per class (mean +- s.e.m. over sessions, 081) and the composition of converging neurons
by class.
Output: across_days/<ref>[_units]/single_unit_classes/<pop>/neuron_class_psth.{png,pdf,svg}.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m81 = importlib.import_module("081_single_unit_classes")
COH, CL = m62.COH, m62.CL
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
GLAB = {("R+", "learning"): "R+ day 0", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− day 0", ("R-", "expert"): "R− expert"}
CCOL = {"sensory": "#4C72B0", "lick-gated": "#DD8452", "non-sensory": "#55A868"}


def main(a):
    plt = m62.setup()
    U = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda c: U[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U["rl"] = sig(f"sig:{m81.SEL_AS}") & (U[f"sel:{m81.SEL_AS}"] > 0)
    U["conv"] = U.rl & sig(f"sig:{m81.SEL_WS}") & (U[f"sel:{m81.SEL_WS}"] > 0)
    U["rl_only"] = U.rl & ~sig(f"sig:{m81.SEL_WS}")
    PB, KB, tc = m62.load_psth_prestart(U)
    U = U.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    C0 = pd.read_csv(m81.OUT / "classes.csv")
    for pop in ("learners", "all"):
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        W, C = flt(U), flt(C0)
        out = m81.OUT / pop; out.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(m62.W_IN, 6.6))
        gs = fig.add_gridspec(4, 5, hspace=0.75, wspace=0.45, left=0.09, right=0.985, top=0.88, bottom=0.08)
        axs_psth, rows_ax = [], []
        specs = [(c, W.conv & (W.nclass == c), f"Converging, {c}") for c in m81.CLASSES] + \
                [("reward-lick-only", W.rl_only & W.nclass.isin(m81.CLASSES), "Reward-lick-only, all classes")]
        for i, (key, mask, rlab) in enumerate(specs):
            row = []
            for j, (coh, st) in enumerate(GROUPS):
                ax = fig.add_subplot(gs[i, j]); ax.set_box_aspect(1); axs_psth.append(ax); row.append(ax)
                mk = (mask & (W.cohort == coh) & (W.stage == st)).to_numpy()
                ns, nu = m62.mouse_psth(ax, W, PB, tc, mk, min_units=1, legend=False)
                ax.set_xlim(-400, 200); ax.set_xticks([-300, 0])
                ax.set_xticklabels(["−300", "0"] if i == 3 else [])
                if i == 0:
                    ax.set_title(GLAB[(coh, st)], color=COH[coh], fontsize=6)
                ax.text(0.03, 0.97, f"{nu} units / {ns} sessions", transform=ax.transAxes, ha="left", va="top", fontsize=4.2, color="0.3")
                if j == 0:
                    ax.set_ylabel(f"{rlab}\nrate − baseline (Hz)", fontsize=5)
                if i == 3:
                    ax.set_xlabel("Time from first lick (ms)", fontsize=5)
            rows_ax.append(row)
        yl = (min(x.get_ylim()[0] for x in axs_psth), max(x.get_ylim()[1] for x in axs_psth))
        for x in axs_psth:
            x.set_ylim(*yl)
        # right column: converging fraction per class (rows 1-3) and composition of converging neurons (row 4)
        xs = {g: k for k, g in enumerate(GROUPS)}
        for i, c in enumerate(m81.CLASSES):
            ax = fig.add_subplot(gs[i, 4]); ax.set_box_aspect(1); rows_ax[i].append(ax)
            d = C[C.nclass == c].dropna(subset=["conv"])
            for g in GROUPS:
                v = d[(d.cohort == g[0]) & (d.stage == g[1])].conv
                ax.errorbar(xs[g], v.mean(), v.sem(), fmt="o", ms=3.8, color=COH[g[0]], lw=1, capsize=0,
                            mfc="white" if g[1] == "learning" else COH[g[0]])
            ax.plot([0, 1], [d[(d.cohort == "R+") & (d.stage == s)].conv.mean() for s in ("learning", "expert")], color=COH["R+"], lw=0.8)
            ax.plot([2, 3], [d[(d.cohort == "R-") & (d.stage == s)].conv.mean() for s in ("learning", "expert")], color=COH["R-"], lw=0.8)
            ax.set_xticks(range(4), ["D0", "E", "D0", "E"], fontsize=4.8); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 1)
            ax.set_title(f"Converging fraction\n({c})", fontsize=5.2)
        ax = fig.add_subplot(gs[3, 4]); ax.set_box_aspect(1); rows_ax[3].append(ax)
        cv = W[W.conv & W.nclass.isin(m81.CLASSES)]
        bottom = np.zeros(4)
        for c in m81.CLASSES:
            frac = np.array([((cv.cohort == g[0]) & (cv.stage == g[1]) & (cv.nclass == c)).sum()
                             / max(((cv.cohort == g[0]) & (cv.stage == g[1])).sum(), 1) for g in GROUPS])
            ax.bar(range(4), frac, bottom=bottom, color=CCOL[c], width=0.7)
            bottom += frac
        ax.set_xticks(range(4), ["R+\nD0", "R+\nE", "R−\nD0", "R−\nE"], fontsize=4.6); ax.set_ylim(0, 1)
        ax.set_title("Converging neurons\nby class", fontsize=5.2)
        for k_, c_ in enumerate(m81.CLASSES):
            ax.text(4.0, 0.15 + 0.3 * k_, c_, color=CCOL[c_], fontsize=4.4, va="center", clip_on=False)
        for r_, letters in zip(rows_ax, ("abcde", "fghij", "klmno", "pqrst")):
            m62.letter_row(fig, r_, letters)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], color=CL[k], lw=1.2, label=l) for k, l in (("WH", "Whisker hit"), ("AH", "Auditory hit"), ("FA", "Spontaneous lick"))],
                   loc="upper left", bbox_to_anchor=(0.09, 0.955), ncol=3, frameon=False, fontsize=5)
        fig.suptitle(f"Converging neurons by stimulus class across learning ({pop})",
                     x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold")
        m62.save(fig, out, "neuron_class_psth"); plt.close(fig)
        print(pop, "done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
