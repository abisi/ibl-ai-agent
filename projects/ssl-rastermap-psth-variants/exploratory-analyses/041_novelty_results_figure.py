"""One multi-panel figure explaining the exposure / reward-model results (040_novelty_refit), learning day, good units.

a  Lick model: observed vs predicted P(lick) on whisker trials and fitted value V (R+ rises, R- extinguishes).
b  Model-free: whole-brain whisker response (late window) vs whisker trial, hits vs misses, per cohort.
c  Value vs response on hits (model-light): within-session demeaned response vs within-session demeaned V (quintiles),
   per cohort, whole brain and motor areas (late window).
d  Misses (exposure test): initial novelty amplitude per region (early / late), bootstrap 95% CI.
e  Misses, model-free: response vs whisker trial for regions with habituation / sensitisation (cohorts pooled).
f/g Hits (reward test): value weight bR in R+ and R- per region (early / late), 95% CI, cohort-label permutation p.
h  Held-out (leave-one-mouse-out) gain in within-mouse R^2: novelty vs nuisance (misses), reward vs novelty (hits).
i  Session subsets: bR(R+) - bR(R-) on hits and novelty amplitude, all vs passive-pre vs no-passive sessions.
j  Summary.
Output: combined_results_ks4/_novelty_model/refit/novelty_results_figure.{png,pdf}
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
nm = importlib.import_module("037_novelty_model")
BASE = nm.BASE; OUT = BASE / "refit"
COH = {"R+": "#00B400", "R-": "#C800C8"}
REG = ["all", "Somatosensory-whisker", "Retrosplenial areas", "Motor areas", "Thalamus", "Hippocampus"]
RLAB = {"all": "Whole brain", "Somatosensory-whisker": "Whisker S1", "Retrosplenial areas": "Retrosplenial",
        "Motor areas": "Motor", "Thalamus": "Thalamus", "Hippocampus": "Hippocampus"}


def pstar(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def binned(df, x, y, nb=10, maxb=12):
    df = df.assign(b=(df[x] // nb).astype(int)); df = df[df.b < maxb]
    ps = df.groupby(["session_id", "b"])[y].mean().unstack()
    mu, se = ps.mean(), ps.std() / np.sqrt(ps.notna().sum()); ok = ps.notna().sum() >= 5
    return (mu.index[ok] + 0.5) * nb, mu[ok], se[ok]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 5.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    T = pd.read_csv(OUT / "refit_results.csv")
    Vdf = pd.read_parquet(OUT / "lick_model_values.parquet")
    R, C = nm.load()
    fig = plt.figure(figsize=(7.4, 10.4))
    gs = fig.add_gridspec(4, 3, hspace=0.66, wspace=0.62, left=0.115, right=0.975, top=0.955, bottom=0.05)

    def letter(ax, s, title):
        ax.set_title(title, loc="left", fontsize=7)
        ax.text(-0.2, 1.1, s, transform=ax.transAxes, fontsize=9, weight="bold")

    # a lick model
    ax = fig.add_subplot(gs[0, 0])
    W = C[C.ttype == "whisker"].merge(Vdf, on=["session_id", "trial"])
    for coh, col in COH.items():
        g = W[W.cohort == coh]
        for y, ls, lab in [("lick", "-", "P(lick) observed"), ("p_lick_fit", "--", "P(lick) model"), ("V", ":", "value V")]:
            x, mu, se = binned(g, "n_whisker_before", y, maxb=15)
            ax.plot(x, mu, color=col, ls=ls, lw=1 if ls != ":" else 1.4, label=f"{coh.replace('-', chr(8722))} {lab}")
            if ls == "-":
                ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.15, lw=0)
    ax.set_ylim(0, 1.02); ax.set_xlabel("whisker trial"); ax.set_ylabel("probability / value")
    ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=4.8)
    letter(ax, "a", "Lick model: value learned from licks")

    # b model-free hits vs misses
    ax = fig.add_subplot(gs[0, 1])
    Rall = R[(R.level == "all")].merge(C, on=["session_id", "trial"])
    for coh, col in COH.items():
        for outc, ls, lab in [(1, "-", "hits"), (0, "--", "misses")]:
            g = Rall[(Rall.ttype == "whisker") & (Rall.cohort == coh) & (Rall.lick == outc)]
            x, mu, se = binned(g, "n_whisker_before", "z_late", maxb=10)
            ax.errorbar(x, mu, se, color=col, ls=ls, lw=1, marker="o", ms=2, capsize=1,
                        label=f"{coh.replace('-', chr(8722))} {lab}")
    ax.set_xlabel("whisker trial"); ax.set_ylabel("whole-brain response, 50-150 ms (z)")
    ax.legend(frameon=False, fontsize=5)
    letter(ax, "b", "Model-free: hits vs misses")

    # c value vs response on hits
    ax = fig.add_subplot(gs[0, 2])
    for reg, mk in [("all", "o"), ("Motor areas", "s")]:
        Wr = nm.assemble(R, C, reg, "late").merge(Vdf[["session_id", "trial", "V"]], on=["session_id", "trial"])
        Wr = Wr[Wr.lick == 1].copy()
        Wr["y_dm"] = Wr.y - Wr.groupby("session_id").y.transform("mean")
        Wr["V_dm"] = Wr.V - Wr.groupby("session_id").V.transform("mean")
        for coh, col in COH.items():
            g = Wr[Wr.cohort == coh].copy()
            g["q"] = pd.qcut(g.V_dm, 5, labels=False, duplicates="drop")
            ps = g.groupby(["session_id", "q"]).agg(x=("V_dm", "mean"), y=("y_dm", "mean")).reset_index()
            agg = ps.groupby("q").agg(x=("x", "mean"), y=("y", "mean"), se=("y", lambda s: s.std() / np.sqrt(len(s))))
            ax.errorbar(agg.x, agg.y, agg.se, color=col, marker=mk, ms=3, lw=1, capsize=1, ls="-" if reg == "all" else ":",
                        label=f"{RLAB[reg]}, {coh.replace('-', chr(8722))}")
    ax.axhline(0, color="0.6", lw=0.4); ax.axvline(0, color="0.6", lw=0.4)
    ax.set_xlabel("value V (within-session deviation)"); ax.set_ylabel("hit response, 50-150 ms (within-session dev., z)")
    ax.legend(frameon=False, fontsize=4.8)
    letter(ax, "c", "Hits: response follows value in R+ only")

    # d misses novelty amplitude
    ax = fig.add_subplot(gs[1, 0])
    S = T[(T.subset == "all") & (T.outcome == "miss")].set_index(["region", "window"])
    y = np.arange(len(REG))
    for k, (win, col) in enumerate([("early", "0.25"), ("late", "#E08214")]):
        v = np.array([S.loc[(r, win), "initial_amplitude"] for r in REG]); lo = np.array([S.loc[(r, win), "amp_lo"] for r in REG])
        hi = np.array([S.loc[(r, win), "amp_hi"] for r in REG])
        yy = y + (k - 0.5) * 0.3
        ax.errorbar(v, yy, xerr=[v - lo, hi - v], fmt="none", ecolor=col, elinewidth=0.8, zorder=1)
        ax.plot([], [], "o", ms=3, color=col, label=f"{win} ({'5-50' if win == 'early' else '50-150'} ms)")
        for yv, l, h, vv in zip(yy, lo, hi, v):
            if l > 0 or h < 0:
                ax.plot(vv, yv, "o", ms=3.5, color=col, zorder=3)
            else:
                ax.plot(vv, yv, "o", ms=3.5, mfc="white", mec=col, zorder=3)
    ax.axvline(0, color="k", lw=0.5); ax.set_yticks(y, [RLAB[r] for r in REG]); ax.invert_yaxis()
    ax.set_xlabel("initial novelty amplitude (z)\n>0 habituation, <0 sensitisation")
    ax.legend(frameon=False, fontsize=5, loc="lower right")
    letter(ax, "d", "Misses: exposure effect (filled = CI excl. 0)")

    # e misses model-free per region
    ax = fig.add_subplot(gs[1, 1])
    cols = {"Somatosensory-whisker": "#E08214", "Retrosplenial areas": "#8C510A", "Motor areas": "#01665E", "Thalamus": "#5E3C99"}
    Rg = R[(R.level == "area_group") & (R.n_units >= 5)].merge(C, on=["session_id", "trial"])
    for reg, col in cols.items():
        g = Rg[(Rg.region == reg) & (Rg.ttype == "whisker") & (Rg.lick == 0)].copy()
        g["zn"] = g.z_early if reg in ("Somatosensory-whisker", "Retrosplenial areas") else g.z_late
        x, mu, se = binned(g, "n_whisker_before", "zn", maxb=10)
        ax.plot(x, mu, color=col, lw=1, marker="o", ms=2,
                label=f"{RLAB[reg]} ({'early' if reg in ('Somatosensory-whisker', 'Retrosplenial areas') else 'late'})")
        ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color="0.6", lw=0.4); ax.set_xlabel("whisker trial"); ax.set_ylabel("miss response (z), both cohorts")
    ax.legend(frameon=False, fontsize=5)
    letter(ax, "e", "Misses, model-free")

    # f/g hits value weights per region
    for j, win in enumerate(["early", "late"]):
        ax = fig.add_subplot(gs[1 + j, 2] if j == 0 else gs[2, 0])
        S = T[(T.subset == "all") & (T.outcome == "hit") & (T.window == win)].set_index("region").reindex(REG)
        for k, (coh, col) in enumerate([("Rplus", COH["R+"]), ("Rminus", COH["R-"])]):
            v, lo, hi = S[f"bR_{coh}"], S[f"bR_{coh}_lo"], S[f"bR_{coh}_hi"]
            ax.errorbar(v, y + (k - 0.5) * 0.3, xerr=[v - lo, hi - v], fmt="o", ms=3, color=col, elinewidth=0.8,
                        label="R+" if k == 0 else "R−")
        for yv, p in zip(y, S.p_perm_bR_diff):
            ax.text(ax.get_xlim()[1] if False else 3.3, yv, f"{pstar(p)}", va="center", fontsize=7)
        ax.set_xlim(-1.2, 3.6); ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(y, [RLAB[r] for r in REG]); ax.invert_yaxis()
        ax.set_xlabel("value weight bR on hit response")
        ax.legend(frameon=False, fontsize=5, loc="lower right")
        letter(ax, "fg"[j], f"Hits: value weight, {win} window\n(* R+ vs R− permutation p)")

    # h held-out gains heatmap
    ax = fig.add_subplot(gs[2, 1:])
    rows = [("miss", "gain_freq_shared_vs_nuis", "pW_freq_shared_vs_nuis", "pT_freq_shared_vs_nuis", "Miss: novelty"),
            ("hit", "gain_freq_shared+reward_vs_freq_shared", "pW_freq_shared+reward_vs_freq_shared",
             "pT_freq_shared+reward_vs_freq_shared", "Hit: +value")]
    M = np.full((4, len(REG)), np.nan); ann = [[""] * len(REG) for _ in range(4)]; ylab = []
    for i, (outc, g, pw, pt, lab) in enumerate(rows):
        for w_i, win in enumerate(["early", "late"]):
            S = T[(T.subset == "all") & (T.outcome == outc) & (T.window == win)].set_index("region").reindex(REG)
            M[2 * i + w_i] = S[g] * 100
            ann[2 * i + w_i] = [f"{v * 100:+.1f}{'*' if (a < .05 and b < .05) else ''}" for v, a, b in zip(S[g], S[pw], S[pt])]
            ylab.append(f"{lab} ({win})")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for r_ in range(4):
        for c_ in range(len(REG)):
            ax.text(c_, r_, ann[r_][c_], ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(len(REG)), [RLAB[r] for r in REG], rotation=20, ha="right"); ax.set_yticks(range(4), ylab)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("held-out R² gain (%)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    letter(ax, "h", "Does the term predict held-out mice? (* both Wilcoxon and t p < .05)")

    # i subsets
    ax = fig.add_subplot(gs[3, 0:2])
    subs = [("all", "all sessions"), ("pre", "passive-pre"), ("none", "no passive")]
    items = [("hit", "all", "late"), ("hit", "Motor areas", "late"), ("hit", "Somatosensory-whisker", "late"),
             ("miss", "Somatosensory-whisker", "early"), ("miss", "all", "early")]
    xl = []
    for k, (outc, reg, win) in enumerate(items):
        for s_i, (sub, slab) in enumerate(subs):
            r = T[(T.subset == sub) & (T.outcome == outc) & (T.region == reg) & (T.window == win)]
            if not len(r):
                continue
            r = r.iloc[0]
            if outc == "hit":
                v = r.bR_Rplus - r.bR_Rminus; lo = hi = None
            else:
                v, lo, hi = r.initial_amplitude, r.amp_lo, r.amp_hi
            x = k + (s_i - 1) * 0.25
            ax.bar(x, v, 0.23, color=["0.3", "0.6", "0.85"][s_i], label=slab if k == 0 else None)
            if lo is not None:
                ax.errorbar(x, v, [[v - lo], [hi - v]], color="k", elinewidth=0.7, capsize=1.5)
            ax.text(x, 0, f"{int(r.n_sess_Rplus)}/{int(r.n_sess_Rminus)}", rotation=90, fontsize=4.2, ha="center",
                    va="top" if v >= 0 else "bottom")
            if outc == "hit":
                ax.text(x, v + (0.05 if v >= 0 else -0.12), pstar(r.p_perm_bR_diff), ha="center", fontsize=6)
        xl.append(("Hits: ΔbR (R+−R−)" if outc == "hit" else "Misses: novelty amp.") + f"\n{RLAB[reg]}, {win}")
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(items)), xl, fontsize=5.0)
    ax.set_ylabel("effect"); ax.legend(frameon=False, fontsize=5.5, ncol=3)
    letter(ax, "i", "Session subsets (numbers: R+/R− mice; no-passive subset is underpowered)")

    # j summary
    ax = fig.add_subplot(gs[3, 2]); ax.axis("off")
    ax.text(-0.1, 1.08, "j", transform=ax.transAxes, fontsize=9, weight="bold")
    ax.text(-0.12, 1.0,
            "Summary\n\n"
            "Exposure (misses, no lick, no reward)\n"
            "- small habituation in whisker S1,\n"
            "  retrosplenial (early) and motor (late);\n"
            "  sensitisation in thalamus\n"
            "- same in both cohorts; does not improve\n"
            "  held-out prediction beyond nuisance\n\n"
            "Reward (hits)\n"
            "- response scales with learned value in\n"
            "  R+ but not R- (whole brain late, motor,\n"
            "  retrosplenial, thalamus, whisker S1)\n"
            "- value improves held-out prediction only\n"
            "  for the whole-brain early window\n\n"
            "Caveats\n"
            "- uncorrected across 56 fits (strongest\n"
            "  p = 0.002 survive)\n"
            "- late window may contain lick preparation\n"
            "  (R+ reacts faster): add RT as nuisance\n"
            "- misses: value weights flip sign -> value\n"
            "  also tracks engagement state",
            transform=ax.transAxes, va="top", fontsize=5.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F7F7F7", ec="0.7", lw=0.6))
    fig.suptitle("Exposure and reward effects on whisker-trial responses, single-session learning\n"
                 "(90 mice, good units; 95% bootstrap CIs over mice)", fontsize=7.5, y=0.997)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"novelty_results_figure.{ext}", dpi=250)
    print("saved", OUT / "novelty_results_figure.png")


if __name__ == "__main__":
    main()
