"""017 -- figures of the article report (haas), from the result tables only (pilot: 002 / 008 / 009 / 010 tables;
full run: full_* / full_009_* / 016 tables). Output: figures/017_fig*.png|pdf|svg.

Fig 1  pilot: geometry removal and cross-mouse validity (session specificity raw / centre / rest; syllables used by >= 6
       of 8 mice; session-identity decoding; lightning-action leave-one-mouse-out F1).
Fig 2  meta-states: fingerprints in physical units (pooled-z colour, physical value in the cell) and how the states of
       each method / view map onto them (full run).
Fig 3  full run: usage of actions (lightning-action) and meta-states by trial epoch, R+ vs R-, all frames and no-lick.
Fig 4  learning analyses (016): copied from figures/016_learning_summary.
Fig S1 pilot DLC dropout per bodypart and around licks.
Fig S2 pilot kappa x number-of-states sweeps (median duration, session specificity).
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402

plt.rcParams.update({"font.size": 6, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
T = L.TABLES
KP = L.RES / "kuma"


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"017_{name}.{ext}", dpi=300)
    plt.close(fig)


def letter(ax, s):
    ax.text(-0.12, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def fig1():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6))
    ax = axs[0, 0]
    rows = []
    for v in L.VIEWS:
        for n in ("raw", "centre", "rest"):
            f = KP / "kpms" / v / n / "sweep_ns50_k1e+08.json"
            if f.exists():
                rows.append({**json.load(open(f)), "view": v, "norm": n})
    N = pd.DataFrame(rows)
    for i, v in enumerate(L.VIEWS):
        d = N[N.view == v].set_index("norm").reindex(["raw", "centre", "rest"])
        ax.bar(np.arange(3) + (i - 0.5) * 0.38, d.frac_frames_session_specific, 0.36, color=["#444444", "#1f77b4"][i], lw=0, label=f"{v} view")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["aligned (mm)", "median centring", "rest subtraction"])
    ax.set_ylabel("frames in session-specific syllables")
    ax.set_ylim(0, 1.02 * max(0.4, N.frac_frames_session_specific.max() if len(N) else 0.4))
    ax.legend(frameon=False)
    ax.set_title("keypoint-MoSeq AR stage (kappa 1e8, 50 states)", fontsize=6)
    letter(ax, "A")
    ax = axs[0, 1]
    if (T / "010_shared_syllables.csv").exists():
        S = pd.read_csv(T / "010_shared_syllables.csv")
        lab = [f"{m} {n} {v}".replace("  ", " ") for m, n, v in zip(S.method, S.norm.fillna(""), S.view)]
        ax.barh(range(len(S)), S.frac_frames_in_ge6_mice_syllables, color="#777777", lw=0, label="frames")
        ax.scatter(S.frac_syllables_ge6_mice, range(len(S)), color="k", s=8, zorder=3, label="syllables / states")
        ax.set_yticks(range(len(S)))
        ax.set_yticklabels(lab, fontsize=5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("fraction used by >= 6 of 8 mice")
        ax.legend(frameon=False, fontsize=5)
    letter(ax, "B")
    ax = axs[1, 0]
    if (T / "010_identity_decoding.csv").exists():
        D = pd.read_csv(T / "010_identity_decoding.csv")
        ax.barh(range(len(D)), D.accuracy, color="#777777", lw=0)
        ax.axvline(1 / 8, color="k", ls=":", lw=0.8)
        ax.set_yticks(range(len(D)))
        ax.set_yticklabels([f"{t}: {m}" for t, m in zip(D.test, D.model)], fontsize=4.5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("session-identity decoding accuracy (chance 0.125, dotted)")
    letter(ax, "C")
    ax = axs[1, 1]
    f1 = pd.concat([pd.read_csv(f) for f in (KP / "lightning_action").glob("*/f1_heldout_fold*.csv")]) \
        if list((KP / "lightning_action").glob("*/f1_heldout_fold*.csv")) else pd.DataFrame()
    if len(f1):
        sel = pd.read_csv(T / "001_selected_sessions.csv")
        coh = dict(zip(sel.session_id, sel.reward_group))
        cls = [c for c in ["still", "whisking", "nose", "jaw", "lick"] if c in set(f1.cls)]
        for i, v in enumerate(L.VIEWS):
            for j, c in enumerate(cls):
                q = f1[(f1.view == v) & (f1.cls == c)]
                x = j + (i - 0.5) * 0.38
                ax.bar(x, q.f1.mean(), 0.36, color=["#444444", "#1f77b4"][i], alpha=0.5, lw=0, label=f"{v} view" if j == 0 else None)
                ax.scatter(np.full(len(q), x), q.f1, s=5, color=[L.COH_COL[coh[s]] for s in q.session_id], lw=0, zorder=3)
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels(cls)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("F1 vs heuristic labels, held-out mouse")
        ax.legend(frameon=False, fontsize=5)
    letter(ax, "D")
    fig.tight_layout()
    save(fig, "fig1_validation")


def fig2(prefix="full_009_"):
    M = pd.read_csv(T / f"{prefix}metastates.csv")
    P = pd.read_csv(T / f"{prefix}fingerprints_pooled.csv")
    feats = [c for c in M.columns if f"phys_{c}" in M.columns]
    fig = plt.figure(figsize=(7.4, 6.0))
    ax = fig.add_axes([0.25, 0.55, 0.62, 0.38])
    Z = M[feats].to_numpy(float)
    v = np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(Z.shape[0]):
        for j, f in enumerate(feats):
            ax.text(j, i, f"{M[f'phys_{f}'].iloc[i]:.2g}", ha="center", va="center", fontsize=4.2)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=50, ha="right", fontsize=5)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{n} ({100 * f:.0f}%)" for n, f in zip(M.name, M.frac_frames)], fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("z (pooled statistics)")
    ax.set_title("Meta-state fingerprints: colour = pooled z, text = value in physical units (deg, mm, mm/s, P)", fontsize=6, loc="left")
    letter(ax, "A")
    ax = fig.add_axes([0.16, 0.12, 0.80, 0.30])
    rows = [(m, v) for m in ("kpms", "arhmm", "lightning_action") for v in L.VIEWS if (m, v) != ("lightning_action", "top")]
    K = len(M)
    U = np.zeros((len(rows), K))
    for i, (m, v) in enumerate(rows):
        g = P[(P.method == m) & (P.view == v) & P.shared]
        for k in range(K):
            U[i, k] = g[g.meta == k].usage.sum()
    ax.imshow(U, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(U.shape[0]):
        for k in range(K):
            ax.text(k, i, f"{U[i, k]:.2f}", ha="center", va="center", fontsize=5, color="w" if U[i, k] > 0.5 else "k")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m} {v}" for m, v in rows], fontsize=5)
    ax.set_xticks(range(K))
    ax.set_xticklabels(M.name, rotation=20, ha="right", fontsize=5)
    ax.set_title("Fraction of frames of each method / view in each meta-state (shared states, >= 3 mice)", fontsize=6, loc="left")
    letter(ax, "B")
    save(fig, "fig2_metastates")


def fig3():
    D = pd.read_parquet(T / "full_trial_measures.parquet")
    S = pd.read_csv(T / "full_sessions.csv")
    coh = dict(zip(S.session_id, S.reward_group))
    acts = [c for c in D.columns if c.startswith("act_") and not c.endswith("_nl")]
    metas = [c for c in D.columns if c.startswith("meta_side_M") and not c.endswith("_nl")]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4))
    for r, (cols, title) in enumerate(((acts, "actions (lightning-action, side)"), (metas, "meta-states (kpms side)"))):
        for c, suf in enumerate(("", "_nl")):
            ax = axs[r, c]
            w = D[D.stim == "whisker"]
            m = w.groupby(["session_id", "epoch"])[[x + suf for x in cols if x + suf in w]].mean().reset_index()
            m["cohort"] = m.session_id.map(coh)
            for j, x in enumerate(cols):
                for k, ep in enumerate(("pre", "resp", "post", "iti")):
                    for h, cg in enumerate(("R+", "R-")):
                        q = m[(m.epoch == ep) & (m.cohort == cg)][x + suf].dropna()
                        pos = j * 5 + k + (h - 0.5) * 0.35
                        ax.bar(pos, q.mean(), 0.33, yerr=q.sem(), color=L.COH_COL[cg], lw=0, alpha=0.8,
                               label=cg if (j == 0 and k == 0) else None, error_kw=dict(lw=0.5))
            ax.set_xticks([j * 5 + 1.5 for j in range(len(cols))])
            ax.set_xticklabels([x.replace("act_", "").replace("meta_side_", "") for x in cols], fontsize=5)
            ax.set_ylabel("fraction of frames (mean +- s.e.m. over mice)")
            ax.set_title(f"{title}; bars per item: pre / resp / post (0-5 s) / ITI" + ("; frames > 0.5 s from licks" if suf else ""), fontsize=5.5)
            ax.legend(frameon=False, fontsize=5)
            letter(ax, "ABCD"[2 * r + c])
    fig.tight_layout()
    save(fig, "fig3_usage_by_epoch")


def figS():
    D = pd.read_csv(T / "002_dropout.csv")
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    g = D.groupby(["view", "bodypart"])[["frac_low", "frac_low_peri_lick_0_200ms"]].mean().reset_index()
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.frac_low, 0.4, color="#777777", lw=0, label="whole session")
    ax.bar(x + 0.2, g.frac_low_peri_lick_0_200ms, 0.4, color="#d62728", lw=0, label="0-200 ms after piezo licks")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v} {b}" for v, b in zip(g.view, g.bodypart)], rotation=45, ha="right")
    ax.set_ylabel("fraction of frames below the likelihood cutoff (mean of 8)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "figS1_dropout")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for v, c in zip(L.VIEWS, ("#444444", "#1f77b4")):
        for path, mk, lab in ((KP / "kpms" / v / "rest", "o-", "kpms"), (KP / "arhmm" / v, "s--", "AR-HMM")):
            fs = list(path.glob("sweep_ns*_k*.json"))
            if not fs:
                continue
            d = pd.DataFrame([json.load(open(f)) for f in fs])
            for ns, gg in d.groupby("ns"):
                gg = gg.sort_values("kappa")
                axs[0].plot(gg.kappa, gg.median_ms, mk, color=c, alpha=0.4 + 0.6 * (ns == d.ns.min()), label=f"{lab} {v} {ns} states", ms=3)
                axs[1].plot(gg.kappa, gg.frac_frames_session_specific, mk, color=c, alpha=0.4 + 0.6 * (ns == d.ns.min()), ms=3)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("kappa (stickiness)")
    axs[0].set_yscale("log")
    axs[0].axhspan(150, 500, color="#dddddd", lw=0, edgecolor="none")
    axs[0].set_ylabel("median state duration (ms)")
    axs[1].set_ylabel("frames in session-specific states")
    axs[1].set_ylim(-0.02, 1.02)
    axs[0].legend(frameon=False, fontsize=4.5)
    fig.tight_layout()
    save(fig, "figS2_sweeps")


def main():
    L.FIGS.mkdir(exist_ok=True)
    fig1()
    if (T / "full_009_metastates.csv").exists():
        fig2()
    if (T / "full_trial_measures.parquet").exists():
        fig3()
    figS()
    if (L.FIGS / "016_learning_summary.png").exists():
        for ext in ("png", "pdf", "svg"):
            shutil.copyfile(L.FIGS / f"016_learning_summary.{ext}", L.FIGS / f"017_fig4_learning.{ext}")


if __name__ == "__main__":
    main()
