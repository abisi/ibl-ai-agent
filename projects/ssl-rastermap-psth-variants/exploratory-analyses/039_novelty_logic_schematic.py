"""Schematic of the logic of the exposure/reward model analysis (035-038), built with the real model functions.

a  Design: trial sequence of an example session (passive-pre whisker stimuli, auditory warm-up, then interleaved
   whisker / auditory / catch trials); whisker trials rewarded in R+, never in R-.
b  Novelty model: whisker novelty N_w = -log p(whisker) from leaky counts over ALL trial types; examples on the real
   sequences (no sharing vs auditory->whisker sharing lambda), and the trial-mix prediction (same number of whisker
   exposures, different whisker fraction -> different novelty).
c  Reward model: Rescorla-Wagner value of the whisker stimulus fit to whisker-trial licking (real example sessions).
d  Response model and the question it answers (equation + nested models).
e  Predicted region classes (signatures in R+ vs R-).
f  Controls and inference pipeline.
Output: combined_results_ks4/_novelty_model/logic_schematic.{png,pdf}
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
nm = importlib.import_module("037_novelty_model")
BASE = nm.BASE
COH = {"R+": "#00B400", "R-": "#C800C8"}
TCOL = {"whisker": "#E08214", "auditory": "#2166AC", "catch": "0.6"}
EX = {"R+": "MH069_20260122_111455", "R-": "AB085_20231005_152636"}


def textbox(ax, x, y, s, **kw):
    ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=kw.pop("fs", 6.3),
            bbox=dict(boxstyle="round,pad=0.4", fc=kw.pop("fc", "#F7F7F7"), ec="0.7", lw=0.6), **kw)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    C = pd.read_parquet(BASE / "trial_covariates.parquet").sort_values(["session_id", "trial"])
    fig = plt.figure(figsize=(7.4, 9.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 1.05, 1.05, 1.25], hspace=0.62, wspace=0.28,
                          left=0.08, right=0.98, top=0.965, bottom=0.03)

    # a: design / trial sequence
    ax = fig.add_subplot(gs[0, :])
    for k, (coh, sid) in enumerate(EX.items()):
        g = C[C.session_id == sid]
        n_pw = int(g.n_passive_pre_whisker.iloc[0]); n_pa = int(g.n_auditory_before_all.iloc[0])
        y0 = 1 - k
        x0 = 0
        if n_pw + n_pa:
            ax.add_patch(plt.Rectangle((0, y0 - 0.3), 25, 0.6, color="0.92"))
            ax.text(12.5, y0 + 0.38, f"passive pre\n({n_pw} W, {n_pa} A)", ha="center", fontsize=5.5)
            x0 = 30
        for t, (tt, lick) in enumerate(zip(g.ttype, g.lick)):
            ax.plot([x0 + t, x0 + t], [y0 - 0.25, y0 + 0.25], color=TCOL.get(tt, "0.6"), lw=0.6 if tt != "catch" else 0.3)
            if lick and tt == "whisker":
                ax.plot(x0 + t, y0 + 0.33, marker="v", ms=1.8, color=COH[coh] if coh == "R+" else "k", lw=0)
        wu = int(g.warmup.sum())
        ax.annotate("auditory warm-up", xy=(x0 + wu / 2, y0 - 0.3), xytext=(x0 + wu / 2, y0 - 0.62), ha="center",
                    fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=0.5))
        ax.text(-4, y0, f"{coh.replace('-', chr(8722))}\n{sid.split('_')[0]}", ha="right", va="center", color=COH[coh], fontsize=6.5)
    ax.set_xlim(-2, 520); ax.set_ylim(-0.8, 1.7); ax.axis("off")
    for i, (lab, col) in enumerate(TCOL.items()):
        ax.plot([], [], color=col, lw=2, label=f"{lab} trial")
    ax.plot([], [], "v", color="k", ms=3, label="whisker lick (rewarded in R+ only)")
    ax.legend(frameon=False, ncol=4, fontsize=6, loc="upper right", bbox_to_anchor=(1, 1.12))
    ax.set_title("a  Design: auditory is pretrained (familiar, rewarded in both); whisker is introduced within the session",
                 loc="left", fontsize=8)

    # b: novelty model
    ax = fig.add_subplot(gs[1, 0])
    NC = nm.NoveltyCache(C, list(EX.values()))
    for coh, sid in EX.items():
        for lam, ls in [(0.0, "-"), (0.5, "--")]:
            N = NC.N(sid, 1.0, lam, 100.0)
            ax.plot(np.arange(1, len(N) + 1), N, color=COH[coh], ls=ls, lw=1,
                    label=f"{coh.replace('-', chr(8722))}, λ={lam}")
    ax.set_xlim(0, 120); ax.set_xlabel("whisker exposure (active trials)"); ax.set_ylabel("whisker novelty N_w = −log p(W)")
    ax.legend(frameon=False, fontsize=5.5)
    ax.set_title("b  Novelty model (count-based familiarity)", loc="left", fontsize=8)
    ins = ax.inset_axes([0.5, 0.42, 0.47, 0.3])
    for frac, col in [(0.2, "0.2"), (0.5, "0.6")]:
        rng = np.random.default_rng(0)
        ev = np.where(rng.random(400) < frac, 0, rng.choice([1, 2], 400))
        N = nm.novelty_series(ev, np.ones(len(ev), bool), 1.0, 0.0, 100.0)[ev == 0]
        ins.plot(np.arange(1, len(N) + 1), N, color=col, lw=0.9, label=f"{int(frac * 100)}% whisker trials")
    ins.set_xlim(0, 60); ins.tick_params(labelsize=5); ins.set_title("trial-mix prediction", fontsize=5.5)
    ins.legend(frameon=False, fontsize=4.5); ins.set_xlabel("whisker exposure", fontsize=5)
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    textbox(ax, 0.0, 1.0,
            "Leaky counts over every trial type\n"
            "  c_j <- alpha*c_j + 1[trial activates j]\n"
            "  W whisker, A auditory, C catch, S shared\n"
            "  pretraining: c_A = c_C = c_S = A0, c_W = 0\n\n"
            "p_mod(W) = (c_W+e) / (c_W+c_A+c_C+3e)\n"
            "p_sh(W)  = (c_S+e) / (c_S+c_C+2e)\n"
            "p(W) = (1-lam) p_mod + lam p_sh\n"
            "N_w = -log p(W)   (before each whisker trial)\n\n"
            "Why this and not exp(-n/tau):\n"
            " - familiarity depends on the FRACTION of\n"
            "   whisker trials (interleaved familiar\n"
            "   auditory/catch trials enter the\n"
            "   normalisation), not only the count\n"
            " - lam > 0: familiarity with auditory\n"
            "   'task stimuli' transfers to whisker\n"
            "   -> smaller initial whisker novelty\n"
            " - passive-pre whisker stimuli count too")

    # c: reward model
    ax = fig.add_subplot(gs[2, 0])
    for coh, sid in EX.items():
        w = C[(C.session_id == sid) & (C.ttype == "whisker")]
        V, par = nm.fit_rw(w.lick.to_numpy(), w.reward.to_numpy())
        x = np.arange(1, len(V) + 1)
        ax.plot(x, V, color=COH[coh], lw=1.2, label=f"{coh.replace('-', chr(8722))} value V (η={par['eta']:.2f})")
        ax.plot(x, w.lick.rolling(10, min_periods=1, center=True).mean(), color=COH[coh], lw=0.7, ls=":",
                label=f"{coh.replace('-', chr(8722))} P(lick), 10-trial mean")
    ax.set_xlabel("whisker trial"); ax.set_ylabel("value / P(lick)"); ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    ax.set_title("c  Reward model: value learned from whisker licks", loc="left", fontsize=8)
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    textbox(ax, 0.0, 1.0,
            "Rescorla-Wagner on whisker trials\n"
            "  P(lick_t) = sigmoid(kappa (V_t - theta))\n"
            "  after a lick: V <- V + eta (r - V)\n"
            "  r = 1 in R+ (reward), 0 in R-\n"
            "  V_0 = v0 (generalised from auditory)\n"
            "R(t) = V_t  (the cue windows precede the\n"
            "outcome: a cue response can only carry\n"
            "the learned value, not the prediction\n"
            "error of the current trial)\n\n"
            "Response model, whisker trials of one\n"
            "region / window (early 5-50, late 50-150 ms):\n"
            "  y = mouse + bN*N_w + bR[cohort]*V\n"
            "      + nuisance + noise\n"
            "Novelty parameters shared by both cohorts,\n"
            "only bR is cohort-specific (joint fit; a\n"
            "two-stage 'fit R-, subtract from R+' would\n"
            "misassign the shared monotonic trends)",
            fc="#FFF8E7")

    # d: predicted classes
    ax = fig.add_subplot(gs[3, 0])
    x = np.arange(0, 100)
    sig = {"exposure-driven": (np.exp(-x / 15), np.exp(-x / 15)),
           "reward-maintained": (0.6 + 0.4 * np.exp(-x / 15), np.exp(-x / 15)),
           "reward-emergent": (0.8 * (1 - np.exp(-x / 15)), 0 * x),
           "generalizing": (0.25 * np.exp(-x / 15), 0.25 * np.exp(-x / 15))}
    for i, (lab, (yp, ym)) in enumerate(sig.items()):
        off = -1.25 * i
        ax.plot(x, yp + off, color=COH["R+"], lw=1.1); ax.plot(x, ym + off, color=COH["R-"], lw=1.1, ls="--")
        ax.text(102, off + 0.35, lab, fontsize=6, va="center")
    ax.plot([], [], color=COH["R+"], label="R+"); ax.plot([], [], color=COH["R-"], ls="--", label="R−")
    ax.legend(frameon=False, fontsize=6, loc="lower left"); ax.set_xlim(0, 150); ax.set_yticks([])
    ax.set_xlabel("whisker exposure"); ax.set_ylabel("whisker response (schematic)")
    ax.spines["left"].set_visible(False)
    ax.set_title("d  What each region class predicts", loc="left", fontsize=8)
    ax = fig.add_subplot(gs[3, 1]); ax.axis("off")
    ax.set_title("e  Controls and inference", loc="left", fontsize=8)
    textbox(ax, 0.0, 0.98,
            "Nuisance (null layer, step 2):\n"
            " auditory & catch trends of the same region\n"
            "   (diff-in-diff: non-specific drift)\n"
            " pre-stimulus rate, pupil, whisking, jaw\n"
            " log time since last whisker (adaptation)\n"
            " previous lick / reward, current lick\n"
            "Behavioural check: R- whisker vs catch licking\n\n"
            "Fitting: grid over (alpha, lam, A0) or tau\n"
            "  -> OLS with mouse fixed effects\n"
            "Model choice: leave-one-mouse-out CV R^2\n"
            "  nuisance | exp(count/trial/clock) | novelty\n"
            "  | +sharing | +reward | cohort-specific alpha\n"
            "Inference (mouse = unit): bootstrap over mice,\n"
            "  cohort-label permutation for bR(R+)-bR(R-)\n"
            "Step 1: recovery on the real trial sequences\n"
            "  decides what can be mapped (initial\n"
            "  amplitude, bR, asymptote; not alpha/half-life)\n"
            "Step 5: repeat per region -> map",
            fs=6.0)
    fig.savefig(BASE / "logic_schematic.png", dpi=250); fig.savefig(BASE / "logic_schematic.pdf")
    print("saved", BASE / "logic_schematic.png")


if __name__ == "__main__":
    main()
