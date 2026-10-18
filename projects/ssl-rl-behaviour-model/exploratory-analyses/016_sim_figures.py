"""016 -- Figures of the simulation-based fits of the whole training process (015; model_sim_design.md).
  1 approach    schematic: agents simulated on each mouse's own trial sequence, summaries, matching, held-out comparison
  2 comparison  held-out LL relative to the core agent (all sessions; day-0 fold); best agent per mouse overall, by cohort and by the
                reference sheet's labels (learning_category, gradual, expert_start); share of mice whose best agent has each component
  3 trajectories group data vs best-agent simulations: lick rate per training day (auditory, whisker, no stimulus) and day-0 whisker
                blocks, per cohort x learning_category; fit quality per mouse = deviance explained relative to a no-learning null (one rate
                per trial type for the whole training) and the saturated model (the data rates)
  4 examples    one mouse per cohort x learning_category (median deviance explained): data vs best agent vs core agent
  5 recovery    generating vs selected agent (pseudo-mice simulated from fitted parameters), parameter recovery
  6 parameters  full-agent (GNZ) parameters by learning_category and cohort
Statistics: tables/016_best_by_category.csv, 016_fit_quality.csv
Run (haas): python 016_sim_figures.py [--only 1 2 ...]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402
import rl4_agents as A4  # noqa: E402

SRC, OUT = "015", ""                                       # set from --src (015: mean-rate binomial score; 017: synthetic likelihood)

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 6.5, "legend.fontsize": 5.2})
COH = {"R+": "#00B400", "R-": "#C800C8"}
TC = {"auditory": "#1f78b4", "whisker": "#d95f02", "no_stim": "0.35"}
MODELS = [A4.lab(m) for m in A4.MODELS]
MCOL = dict(zip(MODELS, plt.cm.tab10(np.arange(8))))
CATS = ["good", "moderate", "bad"]          # every mouse of the trial table has a learning_category


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"016{OUT}_{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load():
    F = pd.read_csv(C.FITS / f"{SRC}_fits.csv"); B = pd.read_parquet(C.TAB / f"{SRC}_bins.parquet").reset_index(drop=True)
    ref = pd.read_excel(C.reference_sheet())[["mouse_id", "learning_category", "gradual", "expert_start"]].drop_duplicates("mouse_id")
    ref["learning_category"] = ref.learning_category.fillna("unlabelled")
    H = F.pivot(index="mouse_id", columns="model", values="ll_held")[MODELS]
    H0 = F.pivot(index="mouse_id", columns="model", values="ll_held_day0")[MODELS]
    M = F.drop_duplicates("mouse_id").set_index("mouse_id")[["cohort"]].join(ref.set_index("mouse_id"))
    M["learning_category"] = M.learning_category.fillna("unlabelled")
    M["best"] = H.idxmax(1)
    return F, B, H, H0, M


def quality(B, M):
    """deviance explained per mouse for each agent: (LL_agent - LL_null) / (LL_sat - LL_null) on all bins (in-sample, full fits)"""
    rows = []
    for mid, b in B.groupby("mouse_id"):
        n, k = b.n.to_numpy(), b.k.to_numpy()
        null = b.groupby("trial_type").apply(lambda z: z.k.sum() / z.n.sum()).reindex(b.trial_type).to_numpy()
        ll0 = A4.binom_ll(null, n, k).sum(); lls = A4.binom_ll(k / n, n, k).sum()
        r = dict(mouse_id=mid)
        for m in MODELS:
            r[f"dev_{m}"] = (A4.binom_ll(b[f"sim_{m}"].to_numpy(), n, k).sum() - ll0) / (lls - ll0)
        rows.append(r)
    Q = pd.DataFrame(rows).set_index("mouse_id").join(M)
    Q["dev_best"] = [Q.loc[i, f"dev_{Q.loc[i, 'best']}"] for i in Q.index]
    Q.to_csv(C.TAB / f"016{OUT}_fit_quality.csv")
    return Q


def fig_approach():
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, t, b, fc):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01", fc=fc, ec="k", lw=0.6))
        ax.text(x, y + h / 2 - 0.04, t, ha="center", va="top", weight="bold", fontsize=6.2)
        ax.text(x, y + h / 2 - 0.13, b, ha="center", va="top", fontsize=5.1, linespacing=1.3)

    def ar(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.25"))
    box(0.12, 0.72, 0.22, 0.46, "mouse", "its trial sequence:\npretraining → day 0 →\nwhisker days\n(stimuli, sessions,\ncohort outcomes)", "#f4f4f4")
    box(0.40, 0.72, 0.26, 0.46, "8 agents", "core (specific learning\n+ disengagement)\n+ any of G generalisation,\nN novelty, Z context\nswitch; 2000 parameter\nsets × 10 runs each,\nrun on the same sequence",
        "#cfe2f3")
    box(0.70, 0.72, 0.24, 0.46, "summaries", "per session: lick rate of\nA, W, N trials in the 1st\nand 2nd half; day 0:\nwhisker trials in blocks\nof 10 (first 100)", "#fff2cc")
    box(0.40, 0.18, 0.62, 0.3, "matching and comparison", "score = binomial log-likelihood of the mouse's bin counts given the simulated rates; bank →\n"
        "refinement; leave-one-session-out: choose parameters on the other sessions, score the left-out one;\n"
        "best agent per mouse = highest held-out score → compared with learning_category, gradual, expert_start", "#d9ead3")
    box(0.89, 0.18, 0.2, 0.3, "recovery", "pseudo-mice from\nfitted agents, matched\nthe same way:\nis the generating\nagent selected?", "#e6d7f0")
    ar((0.23, 0.72), (0.27, 0.72)); ar((0.53, 0.72), (0.58, 0.72)); ar((0.12, 0.49), (0.22, 0.33)); ar((0.70, 0.49), (0.60, 0.33))
    ar((0.71, 0.18), (0.79, 0.18))
    save(fig, "1_approach")


def fig_comparison(H, H0, M):
    D = H.sub(H["core"], axis=0); D0 = H0.sub(H0["core"], axis=0)
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6), gridspec_kw=dict(hspace=0.75, wspace=0.55))
    ax = axs[0, 0]; y = np.arange(len(MODELS))
    for k_, c in enumerate(("R+", "R-")):
        ax.barh(y + (k_ - 0.5) * 0.38, D[M.cohort.reindex(D.index) == c].sum(), height=0.36, color=COH[c], label=c.replace("-", "−"))
    ax.set_yticks(y, MODELS); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.5); ax.legend(frameon=False)
    ax.set_xlabel("Σ held-out ΔLL vs core"); ax.set_title("all sessions")
    ax = axs[0, 1]
    for k_, c in enumerate(("R+", "R-")):
        ax.barh(y + (k_ - 0.5) * 0.38, D0[M.cohort.reindex(D0.index) == c].sum(), height=0.36, color=COH[c])
    ax.set_yticks(y, MODELS); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("Σ held-out ΔLL vs core"); ax.set_title("day-0 session")
    ax = axs[0, 2]
    comp = pd.DataFrame({c: M.best.str.contains(c) for c in "GNZ"})
    groups = [("all", M.index)] + [(c.replace("-", "−"), M.index[M.cohort == c]) for c in ("R+", "R-")]
    for i, (gname, idx) in enumerate(groups):
        ax.bar(np.arange(3) + (i - 1) * 0.27, comp.loc[idx].mean(), width=0.25, color=["0.4", COH["R+"], COH["R-"]][i], label=f"{gname} (n={len(idx)})")
    ax.set_xticks(range(3), [f"{c}\n{A4.CNAME[c]}" for c in "GNZ"], fontsize=5.2); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=4.6)
    ax.set_ylabel("fraction of mice whose best\nagent has the component")
    rows = []
    for j, (col, levels) in enumerate((("learning_category", CATS), ("gradual", None), ("expert_start", None))):
        ax = axs[1, j]
        lv = levels or sorted(M[col].dropna().unique(), key=str)
        tab = pd.crosstab(M[col].fillna("NA"), M.best).reindex(index=[str(x) if levels is None else x for x in lv] if levels else None).fillna(0)
        tab = pd.crosstab(M[col].astype(str), M.best).reindex(columns=MODELS).fillna(0)
        if levels:
            tab = tab.reindex([x for x in levels if x in tab.index])
        frac = tab.div(tab.sum(1), axis=0)
        left = np.zeros(len(frac))
        for m in MODELS:
            ax.barh(np.arange(len(frac)), frac[m], left=left, color=MCOL[m], label=m if j == 0 else None, height=0.6)
            left += frac[m].to_numpy()
        ax.set_yticks(range(len(frac)), [f"{i} (n={int(tab.loc[i].sum())})" for i in frac.index], fontsize=5.2); ax.invert_yaxis()
        ax.set_xlabel("fraction of mice"); ax.set_title(f"best agent by {col}")
        for i in frac.index:
            rows.append(dict(label=col, level=i, n=int(tab.loc[i].sum()), **{m: int(tab.loc[i, m]) for m in MODELS}))
    axs[1, 0].legend(frameon=False, fontsize=4.6, ncol=4, loc="upper center", bbox_to_anchor=(1.6, -0.35))
    pd.DataFrame(rows).to_csv(C.TAB / f"016{OUT}_best_by_category.csv", index=False)
    fig.suptitle("Which agent reproduces each mouse's training best? Held-out (leave-one-session-out) binomial log-likelihood of the "
                 "session summaries", fontsize=6.6)
    save(fig, "2_comparison")


def day_curves(B, col):
    """per mouse: rate per day and trial type (halves pooled; day-0 whisker blocks pooled) for data ('rate') or a simulation column"""
    b = B.assign(sk=B[col] * B.n if col != "rate" else B.k)
    return b.groupby(["mouse_id", "day", "trial_type"]).apply(lambda z: z.sk.sum() / z.n.sum()).rename("v").reset_index()


def fig_trajectories(B, M, Q):
    best_col = B.apply(lambda r: r[f"sim_{M.loc[r.mouse_id, 'best']}"], axis=1)
    Bb = B.assign(sim_best=best_col.to_numpy())
    data, sim = day_curves(Bb, "rate"), day_curves(Bb, "sim_best")
    fig = plt.figure(figsize=(7.4, 7.0)); gs = fig.add_gridspec(4, 4, hspace=0.8, wspace=0.45)
    for i, c in enumerate(("R+", "R-")):
        for j, cat in enumerate(CATS):
            ids = M.index[(M.cohort == c) & (M.learning_category == cat)]
            ax = fig.add_subplot(gs[i * 2, j]) if j < 4 else None
            if not len(ids):
                ax.axis("off"); continue
            for tt, col in TC.items():
                for frame, lw, a_, ls in ((data, 1.5, 0.45, "-"), (sim, 0.9, 1.0, "--")):
                    e = frame[frame.mouse_id.isin(ids) & (frame.trial_type == tt)].groupby("day").v.agg(["mean", "sem", "count"])
                    e = e[e["count"] >= min(3, len(ids))]
                    ax.plot(e.index, e["mean"], color=col, lw=lw, alpha=a_, ls=ls, marker="o", ms=1.5)
            ax.axvline(-0.5, color="r", ls="--", lw=0.5); ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{c.replace('-', '−')} {cat} (n={len(ids)})", color=COH[c], fontsize=6)
            if j == 0:
                ax.set_ylabel("lick rate per day")
            ax.set_xlabel("day")
            ax = fig.add_subplot(gs[i * 2 + 1, j])
            e = Bb[Bb.mouse_id.isin(ids) & (Bb.day == 0) & (Bb.trial_type == "whisker")]
            e = e.assign(blk=e.bin.str.replace("block", "").astype(int) * 10 + 5)
            for col_, lw, a_, ls in (("rate", 1.5, 0.45, "-"), ("sim_best", 0.9, 1.0, "--")):
                f = e.groupby("blk")[col_].agg(["mean", "count"]); f = f[f["count"] >= min(3, len(ids))]
                ax.plot(f.index, f["mean"], color=TC["whisker"], lw=lw, alpha=a_, ls=ls, marker="o", ms=1.5)
            ax.set_ylim(-0.02, 1.02); ax.set_xlabel("day-0 whisker trial"); ax.set_title("day-0 whisker learning", fontsize=5.6)
    ax = fig.add_subplot(gs[:, 3])
    for k_, c in enumerate(("R+", "R-")):
        for j, cat in enumerate(CATS):
            v = Q[(Q.cohort == c) & (Q.learning_category == cat)].dev_best
            yy = j + (k_ - 0.5) * 0.35
            ax.scatter(v, np.full(len(v), yy) + np.random.default_rng(j).uniform(-0.08, 0.08, len(v)), s=3, color=COH[c], alpha=0.5)
            if len(v):
                ax.plot([v.median()] * 2, [yy - 0.13, yy + 0.13], color="k", lw=0.9)
    ax.set_yticks(range(len(CATS)), CATS); ax.invert_yaxis(); ax.set_xlim(-0.2, 1.05); ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlabel("deviance explained by the best agent\n(0 = no learning, 1 = saturated)"); ax.set_title("fit quality per mouse")
    fig.suptitle("Data (thick, solid) vs best-agent simulations (thin, dashed), per cohort and learning_category: blue auditory, orange whisker, "
                 "grey no stimulus", fontsize=6.5)
    save(fig, "3_trajectories")


def fig_examples(B, M, Q):
    ex = []
    for c in ("R+", "R-"):
        for cat in ("good", "moderate", "bad"):
            q = Q[(Q.cohort == c) & (Q.learning_category == cat)].sort_values("dev_best")
            if len(q):
                ex.append(q.index[len(q) // 2])
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2), gridspec_kw=dict(hspace=0.6, wspace=0.3))
    for ax, mid in zip(axs.flat, ex):
        b = B[B.mouse_id == mid].reset_index(drop=True); best = M.loc[mid, "best"]
        # x position: training day, the two session halves at day -/+ 0.2, day-0 whisker blocks spread over day -0.4 .. +0.4
        blk = b.bin.str.startswith("block")
        b["x"] = b.day + np.where(blk, -0.4 + 0.08 * b.bin.str.replace("block", "").where(blk, "0").astype(int),
                                  np.where(b.bin == "half0", -0.2, 0.2))
        for tt, col in TC.items():
            e = b[b.trial_type == tt].groupby("x")[list(dict.fromkeys(["rate", f"sim_{best}", "sim_core", "n"]))].apply(
                lambda z: pd.Series({"rate": (z.rate * z.n).sum() / z.n.sum(), "best": (z[f"sim_{best}"] * z.n).sum() / z.n.sum(),
                                     "core": (z.sim_core * z.n).sum() / z.n.sum()}))
            ax.plot(e.index, e.rate, "o", color=col, ms=2.2, alpha=0.6)
            ax.plot(e.index, e.best, "-", color=col, lw=0.9)
            ax.plot(e.index, e.core, ":", color=col, lw=0.7)
        ax.axvspan(-0.5, 0.5, color="#fde0c5", alpha=0.35, lw=0); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{mid} ({M.loc[mid, 'cohort'].replace('-', '−')}, {M.loc[mid, 'learning_category']}): best {best}, "
                     f"dev. expl. {Q.loc[mid, 'dev_best']:.2f}", fontsize=5.6, color=COH[M.loc[mid, "cohort"]])
        ax.set_xlabel("day (halves; shaded: day-0 blocks)")
    fig.suptitle("Example mice (median fit per cohort x category): dots = data, line = best agent, dotted = core agent "
                 "(blue auditory, orange whisker, grey no stimulus)", fontsize=6.4)
    save(fig, "4_examples")


def fig_recovery():
    p = C.FITS / f"{SRC}_recovery.csv"
    if not p.exists():
        print("no recovery yet"); return
    R = pd.read_csv(p)
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1], wspace=0.5))
    cm = pd.crosstab(R.generating, R.selected, normalize="index").reindex(index=MODELS, columns=MODELS).fillna(0)
    ax = axs[0]; ax.imshow(cm.to_numpy(), cmap="Greys", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm.to_numpy()):
        if v > 0:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4, color="w" if v > 0.5 else "k")
    ax.set_xticks(range(8), MODELS, rotation=45, fontsize=5); ax.set_yticks(range(8), MODELS, fontsize=5)
    ax.set_xlabel("selected"); ax.set_ylabel("generating"); ax.set_title(f"model recovery (n={R.mouse_id.nunique()} mice)")
    for ax, q in zip(axs[1:], ("gamma", "nu", "zeta")):
        d = R[R.generating == "GNZ"]
        if f"true_{q}" in d:
            ax.scatter(d[f"true_{q}"], d[f"fit_{q}"], s=4, color="k")
            lo, hi = A4.BOUNDS[q]; ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--")
            r = stats.spearmanr(d[f"true_{q}"], d[f"fit_{q}"])[0]
            ax.set_title(f"{q} ({A4.CNAME[{'gamma': 'G', 'nu': 'N', 'zeta': 'Z'}[q]]})\nρ = {r:.2f}", fontsize=5.8)
            ax.set_xlabel("true"); ax.set_ylabel("recovered")
    fig.suptitle("Recovery: pseudo-mice simulated from fitted agents (own trial sequences), matched exactly like the data", fontsize=6.4)
    save(fig, "5_recovery")


def fig_parameters(F, M):
    G = F[F.model == "GNZ"].set_index("mouse_id").join(M[["learning_category"]])
    P = A4.free("GNZ")
    fig, axs = plt.subplots(2, 6, figsize=(7.4, 3.6), gridspec_kw=dict(hspace=0.7, wspace=0.6))
    for ax, q in zip(axs.flat, P):
        for j, cat in enumerate(CATS):
            for k_, c in enumerate(("R+", "R-")):
                v = G[(G.cohort == c) & (G.learning_category == cat)][q]; x = j + (k_ - 0.5) * 0.35
                ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-0.06, 0.06, len(v)), v, s=2.5, color=COH[c], alpha=0.5, lw=0)
                if len(v):
                    ax.plot([x - 0.12, x + 0.12], [v.median()] * 2, color="k", lw=0.8)
        if q in A4.LOG:
            ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(4), ["good", "mod.", "bad", "unl."], fontsize=4.8); ax.set_title(q, fontsize=6)
    axs.flat[-1].axis("off")
    fig.suptitle("Full-agent (GNZ) parameters per mouse by learning_category (green R+, magenta R−); values at a bound may be unidentified",
                 fontsize=6.4)
    save(fig, "6_parameters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--src", default="015", help="015 (mean-rate binomial score) or 017 (synthetic likelihood)")
    a = ap.parse_args()
    SRC = a.src; OUT = "" if a.src == "015" else f"_{a.src}"
    if 1 in a.only:
        fig_approach()
    F, B, H, H0, M = load()
    Q = quality(B, M)
    print(M.best.value_counts()); print(Q.groupby("learning_category").dev_best.median().round(3))
    if 2 in a.only:
        fig_comparison(H, H0, M)
    if 3 in a.only:
        fig_trajectories(B, M, Q)
    if 4 in a.only:
        fig_examples(B, M, Q)
    if 5 in a.only:
        fig_recovery()
    if 6 in a.only:
        fig_parameters(F, M)
