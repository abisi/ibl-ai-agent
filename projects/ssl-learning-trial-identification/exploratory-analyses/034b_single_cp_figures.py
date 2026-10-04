"""034b -- Figures for the shuffle-calibrated single change-point LT (034).
  034_single_cp_summary.{png,pdf}: a coverage per cohort (learners p < 0.05 / p < 0.10, FA vs SPONT stream, with L6 /
     L6 lenient for reference); b shuffle-p distributions; c LT FA vs SPONT stream; d LT (calibrated, FA) vs L6 lenient;
     e spontaneous vs FA lick probability (session means; validity of the spontaneous stream); f per-session correlation
     of the spontaneous and FA curves over the session.
  034_single_cp_review_<cohort>.png: every mouse -- whisker (cohort colour), FA (grey), spontaneous (orange) HMM curves,
     LT +- 90% CI for both streams (blue = FA stream, orange = SPONT stream; solid = p < 0.05, dotted = not), p values.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/034b_single_cp_figures.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
import importlib  # noqa: E402

import lt_lib as L  # noqa: E402

M034 = importlib.import_module("034_single_cp_calibrated")
COL = {"R+": "#00B400", "R-": "#C800C8"}
SCOL = {"FA": "#1f77b4", "SPONT": "#ff7f0e"}
FS = 6.5


def main():
    plt.rcParams.update({"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False, "font.size": FS})
    R = pd.read_csv(ART / "034_single_cp.csv")
    lt = pd.read_csv(ART / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")
    inp = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    from axel_bisi_paths import axel_bisi_root
    nwb = axel_bisi_root() / "NWB_ks4"
    curves, val = {}, []
    for sid, d in inp.items():
        tw = np.asarray(d["w_start"], float)
        w = L.forward_backward(np.asarray(d["w_outcomes"], int), 1.0)[0] @ L.P_GRID
        f = L.interp_marginals(L.forward_backward(np.asarray(d["n_outcomes"], int), 1.0)[0], np.asarray(d["n_start"], float),
                               tw) @ L.P_GRID
        try:
            bt, by = M034.spont_stream(sid, nwb)
            s = L.interp_marginals(L.forward_backward(by.astype(int), 1.0)[0], bt, tw) @ L.P_GRID
        except Exception:  # noqa: BLE001
            s, by = np.full(len(tw), np.nan), np.array([np.nan])
        curves[sid] = (w, f, s)
        ok = np.isfinite(s)
        val.append(dict(session_id=sid, reward_group=d["reward_group"], fa_rate=float(np.mean(d["n_outcomes"])),
                        spont_rate=float(np.nanmean(by)),
                        curve_r=stats.pearsonr(f[ok], s[ok])[0] if ok.sum() > 5 and np.std(s[ok]) > 0 and np.std(f[ok]) > 0
                        else np.nan))
    V = pd.DataFrame(val)
    V.to_csv(ART / "034_spont_vs_fa.csv", index=False)
    W = R.pivot_table(index=["session_id", "reward_group", "learning_category"], columns="stream",
                      values=["LT", "p_shuffle", "log10_bf"]).reset_index()
    W.columns = ["_".join([c for c in col if c]) for col in W.columns]
    fig, axes = plt.subplots(2, 3, figsize=(8.27, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6)
    ax = axes[0, 0]
    labels = ["FA p<.05", "FA p<.10", "SPONT p<.05", "SPONT p<.10", "L6 strict", "L6 lenient"]
    for j, rg in enumerate(("R+", "R-")):
        g = R[R.reward_group == rg]
        vals = [int(g[(g.stream == "FA")].learner.sum()), int(g[(g.stream == "FA")].learner_p10.sum()),
                int(g[(g.stream == "SPONT")].learner.sum()), int(g[(g.stream == "SPONT")].learner_p10.sum())]
        sids = g.session_id.unique()
        vals += [int(sum(pd.notna(lt.loc[s, "L6 joint CP"]) for s in sids if s in lt.index)),
                 int(sum(pd.notna(lt.loc[s, "L6 lenient"]) for s in sids if s in lt.index))]
        x = np.arange(len(labels)) + (j - 0.5) * 0.38
        ax.bar(x, vals, width=0.36, color=COL[rg], label=f"{'R+' if rg == 'R+' else 'R−'} (n={len(sids)})")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.5, str(v), ha="center", fontsize=FS - 1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=FS - 0.5)
    ax.set_ylabel("mice with an LT")
    ax.legend(frameon=False, fontsize=FS - 0.5)
    ax.set_title("a  coverage", loc="left", fontweight="bold")
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 21)
    for st in ("FA", "SPONT"):
        for rg, ls in (("R+", "-"), ("R-", "--")):
            ax.hist(R[(R.stream == st) & (R.reward_group == rg)].p_shuffle, bins=bins, histtype="step", color=SCOL[st],
                    ls=ls, lw=1.1, label=f"{st} {rg}")
    ax.axvline(0.05, color="k", lw=0.6)
    ax.set_xlabel("shuffle p (per session)")
    ax.set_ylabel("mice")
    ax.legend(frameon=False, fontsize=FS - 1)
    ax.set_title("b  calibrated evidence", loc="left", fontweight="bold")
    ax = axes[0, 2]
    for rg in ("R+", "R-"):
        g = W[(W.reward_group == rg) & (W.p_shuffle_FA < 0.05) & (W.p_shuffle_SPONT < 0.05)]
        ax.scatter(g.LT_FA, g.LT_SPONT, s=9, color=COL[rg], lw=0)
    lim = np.nanmax(W[["LT_FA", "LT_SPONT"]].to_numpy()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.6, ls="--")
    both = W[(W.p_shuffle_FA < 0.05) & (W.p_shuffle_SPONT < 0.05)]
    ax.set_title(f"c  LT, FA vs SPONT stream (both p<.05, n={len(both)})\nmedian |diff| "
                 f"{np.median(np.abs(both.LT_FA - both.LT_SPONT)):.0f} trials", loc="left", fontweight="bold", fontsize=FS)
    ax.set_xlabel("LT, FA stream")
    ax.set_ylabel("LT, SPONT stream")
    ax = axes[1, 0]
    W["L6len"] = W.session_id.map(lambda s: lt.loc[s, "L6 lenient"] if s in lt.index else np.nan)
    for rg in ("R+", "R-"):
        g = W[(W.reward_group == rg) & (W.p_shuffle_FA < 0.05) & W.L6len.notna()]
        ax.scatter(g.L6len, g.LT_FA, s=9, color=COL[rg], lw=0)
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.6, ls="--")
    g = W[(W.p_shuffle_FA < 0.05) & W.L6len.notna()]
    ax.set_title(f"d  calibrated LT (FA) vs L6 lenient (n={len(g)})\nmedian |diff| {np.median(np.abs(g.LT_FA - g.L6len)):.0f}",
                 loc="left", fontweight="bold", fontsize=FS)
    ax.set_xlabel("L6 lenient LT")
    ax.set_ylabel("calibrated LT (FA)")
    ax = axes[1, 1]
    for rg in ("R+", "R-"):
        g = V[V.reward_group == rg]
        ax.scatter(g.fa_rate, g.spont_rate, s=9, color=COL[rg], lw=0)
    r, p = stats.spearmanr(V.fa_rate, V.spont_rate, nan_policy="omit")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("FA rate (no-stim trials)")
    ax.set_ylabel("spontaneous lick probability (1-s bins)")
    ax.set_title(f"e  session rates, Spearman ρ {r:.2f} (p {p:.1g})", loc="left", fontweight="bold", fontsize=FS)
    ax = axes[1, 2]
    for j, rg in enumerate(("R+", "R-")):
        v = V[V.reward_group == rg].curve_r.dropna()
        ax.scatter(j + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=8, color=COL[rg], lw=0)
        ax.plot([j - 0.25, j + 0.25], [v.median()] * 2, color="k")
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R+", "R−"])
    ax.set_ylabel("r(FA curve, spontaneous curve)")
    ax.set_title("f  within-session co-variation", loc="left", fontweight="bold", fontsize=FS)
    fig.suptitle("Single learning-trial definition: joint change point with optional lapse, no start gates, per-session "
                 "shuffle calibration (50 shuffles); baseline stream = FA trials or spontaneous licks", fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"034_single_cp_summary.{ext}", dpi=250)
    plt.close(fig)
    for rg in ("R+", "R-"):
        g = W[W.reward_group == rg].sort_values("p_shuffle_FA")
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.75 * nrow + 0.5), squeeze=False)
        fig.subplots_adjust(left=0.03, right=0.995, top=1 - 0.45 / (1.75 * nrow + 0.5), bottom=0.02, hspace=0.75, wspace=0.15)
        for ax, r in zip(axes.flat, g.itertuples()):
            w, f, s = curves[r.session_id]
            x = np.arange(len(w))
            ax.plot(x, s, color=SCOL["SPONT"], lw=0.8)
            ax.plot(x, f, color="0.45", lw=0.9)
            ax.plot(x, w, color=COL[rg], lw=1.1)
            for st, yy in (("FA", 1.08), ("SPONT", 1.16)):
                row = R[(R.session_id == r.session_id) & (R.stream == st)]
                if not len(row) or not np.isfinite(row.LT.iloc[0]):
                    continue
                row = row.iloc[0]
                sig = row.p_shuffle < 0.05
                ax.plot([row.LT_ci05, row.LT_ci95], [yy, yy], color=SCOL[st], lw=1.2)
                ax.axvline(row.LT, color=SCOL[st], lw=1.2 if sig else 0.8, ls="-" if sig else ":")
            l6 = lt.loc[r.session_id, "L6 lenient"] if r.session_id in lt.index else np.nan
            if pd.notna(l6):
                ax.scatter([l6], [-0.08], marker="^", s=10, color="k")
            ax.set_ylim(-0.12, 1.22)
            ax.set_title(f"{r.session_id[:5]} {r.learning_category if isinstance(r.learning_category, str) else ''} "
                         f"pFA {r.p_shuffle_FA:.2f} pSP {r.p_shuffle_SPONT:.2f}", fontsize=6)
            ax.tick_params(labelsize=5)
        for ax in axes.flat[len(g):]:
            ax.set_axis_off()
        fig.text(0.03, 0.998, f"{'R+' if rg == 'R+' else 'R−'}: curves whisker / FA (grey) / spontaneous (orange); LT ± 90% CI: "
                 "blue = FA stream, orange = SPONT stream (solid = shuffle p < 0.05, dotted = not); black ▲ = L6 lenient. "
                 "Sorted by FA-stream p.", fontsize=7, va="top")
        fig.savefig(HERE / f"034_single_cp_review_{'Rplus' if rg == 'R+' else 'Rminus'}.png", dpi=160)
        plt.close(fig)
    print(V.groupby("reward_group")[["fa_rate", "spont_rate", "curve_r"]].median())


if __name__ == "__main__":
    main()
