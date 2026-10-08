"""160 -- COSYNE figure, decoder version of 158 (user 2026-10-08: "the equivalent cosyne figure but with the decoder null-corrected").
Same layout, learners, whole brain, shared tracked stable units, evoked 5-35 ms responses (epoch baseline):
  a  state space before the task (as 158a)
  b  choice-decoder readout of whisker trials per epoch (146: L2 logistic hit vs miss on active whisker trials; every trial scored
     by models that did not train on it; standardised readout = (score - hit / miss midpoint) / SD of the held-out active whisker
     scores; > 0 hit side, < 0 miss side); passive pre, active 1st / 2nd half, passive post; R+ vs R-; active hit (^) and miss (v)
     levels as reference; top: post - pre within cohort (Wilcoxon) and R+ vs R- (Mann-Whitney)
  c  same for auditory trials (control)
  d  pre -> post change of the readout beyond session time: real change minus the mean change of decoders trained on linearly
     shifted labels (146 shift null), whisker, auditory and whisker - auditory; mean +- s.e.m.; filled: Wilcoxon vs 0 p < 0.05;
     top: R+ vs R- Mann-Whitney
  e  state-space summary per cohort (as 158e)
Statistics non-parametric only; session = unit; uncorrected. SSL_160_RESPONSE=trialbase uses the per-trial baseline variant.
Outputs: figures/publication/160_cosyne_decoder_learners[_trialbase].{png,pdf,svg}; 160_stats_learners[_trialbase].csv
Run (haas, repo root): python .../160_cosyne_decoder.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
F = importlib.import_module("158_cosyne_within_day")
H, COL, COH, FIGDIR, pn, wil, mwu = F.H, F.COL, F.COH, F.FIGDIR, F.pn, F.wil, F.mwu
RESP = os.environ.get("SSL_160_RESPONSE", "epochbase")
SFX = "" if RESP == "epochbase" else f"_{RESP}"
EPOCHS = F.EPOCHS


def load():
    L = H.learners()
    V = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    V = V[V.skipped_reason.isna() & (V.area == "whole_brain") & (V.unit_set == "stable") & (V.response == RESP)]
    return V[H.mouse_of(V).isin(L)].copy()


def panel_readout(ax, V, s, title, R, key):
    x = np.arange(len(EPOCHS))
    for k, c in enumerate(COH):
        d = V[V.reward_group == c]
        m = [d[f"ro_std_{e}_{s}"].mean() for e, _ in EPOCHS]; se = [H.sem(d[f"ro_std_{e}_{s}"]) for e, _ in EPOCHS]
        off = (k - 0.5) * 0.12
        ax.errorbar(x + off, m, yerr=se, fmt="-o", ms=3, color=COL[c], lw=1.1, capsize=0,
                    label=f"{c.replace('-', '−')} (n = {d[f'ro_std_passive_pre_{s}'].notna().sum()})")
        if s == "W":                                              # active hit / miss levels (held-out)
            for j, e in ((1, "active_1"), (2, "active_2")):
                for cl, mk in (("hit", "^"), ("miss", "v")):
                    ax.plot(j + off, d[f"ro_std_{e}_{cl}"].mean(), mk, ms=3.2, color=COL[c], alpha=0.45, mew=0)
    ch = {c: (V[V.reward_group == c][f"ro_std_passive_post_{s}"] - V[V.reward_group == c][f"ro_std_passive_pre_{s}"]).to_numpy(float)
          for c in COH}
    pc = {c: wil(ch[c]) for c in COH}; pm = mwu(ch["R+"], ch["R-"])
    ax.text(0.5, 1.0, f"post − pre: R+ {pn(pc['R+'])}, R− {pn(pc['R-'])}\nR+ vs R− {pn(pm)}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=4.8)
    ax.axhline(0, color="0.7", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in EPOCHS], fontsize=5); ax.set_xlim(-0.4, len(EPOCHS) - 0.6)
    ax.set_ylabel("decoder readout (SD)\n(+ hit side, − miss side)" if s == "W" else "")
    ax.set_title(title, pad=17)
    if s == "W":
        ax.legend(frameon=False, fontsize=4.8, loc="lower left", handlelength=1.2)
        ax.text(0.98, 0.98, "faint ▲▼: active\nhits / misses", transform=ax.transAxes, ha="right", va="top", fontsize=4.3, color="0.4")
    for c in COH:
        R.append(dict(panel=key, measure=f"{s} readout post - pre", cohort=c, n=int(np.isfinite(ch[c]).sum()), mean_a=np.nanmean(ch[c]),
                      p_nonparam=pc[c]))
    R.append(dict(panel=key, measure=f"{s} readout post - pre", cohort="R+ vs R-", mean_a=np.nanmean(ch["R+"]), mean_b=np.nanmean(ch["R-"]),
                  p_nonparam=pm))


def panel_d(ax, V, R):
    items = [("shift_excess_dW", "whisker"), ("shift_excess_dA", "auditory"), ("shift_excess_dWA", "whisker −\nauditory")]
    for i, (col, lab) in enumerate(items):
        v = {c: V[V.reward_group == c][col].to_numpy(float) for c in COH}
        for k, c in enumerate(COH):
            p = wil(v[c]); m = np.nanmean(v[c])
            ax.errorbar(i + (k - 0.5) * 0.3, m, H.sem(v[c]), fmt="o", ms=3.6, color=COL[c], mfc=COL[c] if p < 0.05 else "white", lw=1.1, capsize=0)
            R.append(dict(panel="d", measure=col, cohort=c, n=int(np.isfinite(v[c]).sum()), mean_a=m, p_nonparam=p))
        pm = mwu(v["R+"], v["R-"])
        ax.text(i, 1.0, pn(pm), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.8)
        R.append(dict(panel="d", measure=col, cohort="R+ vs R-", mean_a=np.nanmean(v["R+"]), mean_b=np.nanmean(v["R-"]), p_nonparam=pm))
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for _, l in items], fontsize=5); ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylabel("pre → post readout change\nbeyond session time (SD)")
    ax.set_title("Beyond session time\n(R+ vs R−; filled: ≠ 0)", pad=9)


def main():
    H.setup()
    V = load()
    _, D = F.load()                                              # state-space table (146, epoch baseline) for a and e
    R = []
    fig = plt.figure(figsize=(11.6, 2.4))
    gs = fig.add_gridspec(1, 6, wspace=0.68, left=0.05, right=0.985, top=0.74, bottom=0.2)
    axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
    F.panel_a(axs[0], D, R)
    panel_readout(axs[1], V, "W", "Whisker trials:\nchoice-decoder readout", R, "b")
    panel_readout(axs[2], V, "A", "Auditory trials\n(control)", R, "c")
    lo = min(a.dataLim.y0 for a in axs[1:3]); hi = max(a.dataLim.y1 for a in axs[1:3]); pd_ = 0.08 * (hi - lo)
    yl = (lo - pd_, hi + pd_)
    for a in axs[1:3]:
        a.set_ylim(*yl)
    panel_d(axs[3], V, R)
    cols = ["ss_hit", "ss_miss", "ss_pre_W", "ss_post_W", "ss_pre_A", "ss_post_A"]
    xs = np.concatenate([D[f"{c}_x"].groupby(D.reward_group).mean().to_numpy() for c in cols])
    ys = np.concatenate([D[f"{c}_y"].groupby(D.reward_group).mean().to_numpy() for c in cols])
    pad = lambda v: (np.nanmin(v) - 0.12 * np.ptp(v) - 0.3, np.nanmax(v) + 0.12 * np.ptp(v) + 0.3)
    lims = (pad(np.r_[xs, 0]), pad(np.r_[ys, 0]))
    for ax, c in zip(axs[4:], COH):
        F.panel_e(ax, D, c, lims, R)
    axs[0].set_xlim(*lims[0]); axs[0].set_ylim(*lims[1])
    for a in axs:
        a.set_box_aspect(1)
    for a, l in zip(axs[:5], "abcde"):
        bb = a.get_position(); fig.text(bb.x0 - 0.035, 0.965, l, fontsize=9, weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"160_cosyne_decoder_learners{SFX}.{ext}", dpi=300)
    S = pd.DataFrame(R); S.to_csv(EA / f"160_stats_learners{SFX}.csv", index=False)
    print(S[S.panel.isin(["b", "c", "d"])].round(4).to_string())
    print("ALL DONE", FIGDIR / f"160_cosyne_decoder_learners{SFX}.png")


if __name__ == "__main__":
    main()
