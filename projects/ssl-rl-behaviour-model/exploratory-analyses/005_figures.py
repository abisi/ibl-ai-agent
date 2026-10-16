"""005 -- Figures describing the model, the data it is fitted to, the joint fits, model comparison, parameter recovery, model latents,
cohort comparisons and per-mouse fits (user 2026-10-16: "make several figures showing and describing the models, fitting, performance,
evaluation, examples, stats, comparisons across cohorts").
Inputs: tables/001_trials.csv, targets/, fits/003_*.json (joint mean fits), tables/002_recovery_*.csv, fits/004_*_individual_*.csv.
Figures (figures/005_<n>_<name>.{png,pdf,svg}); every figure is skipped with a message if its inputs are missing:
  1 model     the three architectures, learning rule and the simulated task phases
  2 data      targets: per-mouse and mean P(lick) curves per trial type and cohort; trials per mouse in each phase
  3 fits      joint mean fits (common parameters, and cohort-specific b, a, v in transfer) vs data, per model and cohort
  4 compare   loss per component, nominal AIC / BIC per model and variant
  5 recovery  parameter recovery (true vs fitted, log scale)
  6 latents   whisker drive decomposed into the shared-node and direct parts, weights over trials, R+ vs R-
  7 cohorts   cohort-specific transfer parameters (free b, a, v) and the loss they buy
  8 mice      per-mouse fits: examples, loss per cohort, fitted initial weights R+ vs R- (Mann-Whitney + Welch, BH)
Statistics tables: tables/005_model_comparison.csv, 005_individual_stats.csv.
AIC / BIC are nominal: the targets are smoothed (100-trial windows) and interpolated to 1000 points per phase, so points are strongly
autocorrelated; n is the number of target points and RSS = loss x n. Use them for ranking, not as absolute evidence.
Run (haas): python 005_figures.py [--only 3 4]
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402
import rl_joint as J  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
TT = [("auditory", "auditory", "Auditory", "#1f78b4"), ("whisker", "whisker", "Whisker", "#d95f02"), ("nostim", "no_stim", "No stimulus (FA)", "0.35")]
MODELS = ["independent", "shared", "mixed"]
VARIANTS = {"": "common", "_free-b-a-v": "cohort b, a, v", "_free-tr": "cohort tr"}


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"005_{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_fits():
    out = {}
    for m in MODELS:
        for suf, lab in VARIANTS.items():
            p = C.FITS / f"003_{m}_mean{suf}.json"
            if p.exists():
                out[(m, lab)] = json.loads(p.read_text())
    return out


def targets_mean():
    return J.load_mean_targets(C.TARGETS / "final_mean_withoutfree")


def targets_ind():
    T = [np.load(p, allow_pickle=True).item() for p in sorted((C.TARGETS / "final_individual_withoutfree").glob("target_*.npy"))]
    return {t["mouse_id"]: t for t in T}


def sim(res):
    mod, Model = J.load_model(res["model"])
    return mod, J.simulate_joint(mod, Model, res["common"], res["transfer_overrides"])


# ── 1 model schematic ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_model():
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.35, wspace=0.25)

    def box(ax, xy, txt, fc):
        ax.add_patch(FancyBboxPatch((xy[0] - 0.16, xy[1] - 0.065), 0.32, 0.13, boxstyle="round,pad=0.01", fc=fc, ec="k", lw=0.6, clip_on=False))
        ax.text(*xy, txt, ha="center", va="center", fontsize=5.3)

    def arrow(ax, a, b, lab="", col="k"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, shrinkA=14, shrinkB=14))
        if lab:
            ax.text((a[0] + b[0]) / 2 + 0.02, (a[1] + b[1]) / 2, lab, fontsize=5, color=col)

    ins = {"A": (0.15, 0.85), "W": (0.15, 0.62), "NS": (0.15, 0.39), "C": (0.15, 0.16)}
    cols = {"A": "#cfe2f3", "W": "#fde0c5", "NS": "#e0e0e0", "C": "#f4f4f4"}
    names = {"A": "auditory", "W": "whisker", "NS": "no stim", "C": "context (bias)"}
    titles = {"independent": "Independent: one weight pair per input", "shared": "Shared: A and W through one node",
              "mixed": "Mixed: direct weights + shared node"}
    for j, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, j]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title(titles[m], fontsize=6.5)
        for k, xy in ins.items():
            box(ax, xy, names[k], cols[k])
        box(ax, (0.85, 0.5), "ΔQ → P(lick)\n= Φ(ΔQ)", "#fff2cc")
        direct = {"independent": ["A", "W", "NS", "C"], "shared": ["NS", "C"], "mixed": ["A", "W", "NS", "C"]}[m]
        for k in direct:
            arrow(ax, ins[k], (0.85, 0.5), "", "0.3")
        if m != "independent":
            box(ax, (0.52, 0.92), "shared node S", "#d9ead3")
            arrow(ax, ins["A"], (0.52, 0.92), "", "#38761d"); arrow(ax, ins["W"], (0.52, 0.92), "", "#38761d")
            arrow(ax, (0.52, 0.92), (0.85, 0.5), "", "#38761d")
            ax.text(0.80, 0.80, "Q_out", fontsize=5, color="#38761d", ha="left")
            ax.text(0.34, 0.93, "Q_in", fontsize=5, color="#38761d", ha="right")
        ax.text(0.5, 0.0, {"independent": "ΔQ = (Q_exc − Q_inh)·x", "shared": "ΔQ = (Q_exc − Q_inh)·[NS, C]\n+ (Q_out,exc − Q_out,inh)·S,  S = Q_in·[A, W]",
                           "mixed": "ΔQ = (Q_exc − Q_inh)·x\n+ (Q_out,exc − Q_out,inh)·S,  S = Q_in·[A, W]"}[m], ha="center", va="top", fontsize=5.2)
    ax = fig.add_subplot(gs[1, :2]); ax.axis("off")
    ax.text(0, 1, "Learning rule (every trial, expected update weighted by P(lick))", fontsize=6.8, weight="bold", va="top")
    ax.text(0, 0.84, "δ = R − a·ΔQ(x);  R = +1 rewarded lick, −1 unrewarded lick (FA; whisker in R−)\n"
                     "g(δ): auditory v·δ if δ>0 | no stim v·δ if δ<0 | whisker R+ v·tr·δ if δ>0\n"
                     "      | whisker R− v·δ if δ<0 | otherwise δ   (tr unused in R−)\n"
                     "Q_exc ← Q_exc·(1 + b·g·P·x),  Q_inh ← Q_inh·(1 − b·g·P·x),  P = P(lick)\n"
                     "node (A, W trials): Q_out scaled by S, Q_in ← Q_in·(1 + b·g·P·[A, W])",
            fontsize=5.2, va="top", family="monospace")
    ax.text(0, 0.2, "Free parameters: b (learning rate), a (prediction scaling), v (gain on informative errors),\n"
                    "tr (whisker gain, R+ only), initial weights. Joint fit: ONE pretraining for R+ and R− (same\n"
                    "b, a, v, initial weights); both cohorts start the transfer from the same state and, in the\n"
                    "common variant, differ only through the whisker reward.",
            fontsize=5.4, va="top")
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("Simulated task", fontsize=6.5)
    ax.barh([1, 1], [1000, 1000], left=[-1000, 0], color=["#cfe2f3", "#fde0c5"], height=0.5)
    ax.text(-500, 1, "pretraining\nA 50 % / NS 50 %", ha="center", va="center", fontsize=5.3)
    ax.text(500, 1, "transfer\nW 50 / NS 40 / A 10 %", ha="center", va="center", fontsize=5.3)
    ax.axvline(0, color="r", ls="--", lw=0.8); ax.set_yticks([]); ax.set_xlabel("trials (time-normalised: 1000 per phase)")
    ax.text(0, 0.45, "whisker onset (day 0)", color="r", fontsize=5, ha="center")
    ax.set_ylim(0.3, 1.5)
    save(fig, "1_model")


# ── 2 data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_data():
    TI, TM = targets_ind(), targets_mean()
    x = np.linspace(-1, 1, 2000)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
    for i, g in enumerate(J.GROUPS):
        for j, (_, tk, lab, _) in enumerate(TT):
            ax = axs[i, j]
            for t in TI.values():
                if t["group"] == g:
                    ax.plot(x, t[tk][:2000], color=COH[g], lw=0.3, alpha=0.18)
            ax.plot(x, TM[g][tk][:2000], color="k", lw=1.2)
            ax.axvline(0, color="r", ls="--", lw=0.6); ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{g.replace('-', '−')} — {lab} (n = {sum(t['group'] == g for t in TI.values())} mice)")
            if j == 0:
                ax.set_ylabel("P(lick)")
            if i == 1:
                ax.set_xlabel("normalised time (0 = whisker onset)")
    D = C.TRIALS
    if D.exists():
        d = pd.read_csv(D); d = d[d.bhv_type != "free_licking"]
        n = d.assign(phase=np.where(d.day < 0, "pretraining", "transfer")).groupby(["mouse_id", "reward_group", "phase"]).size().unstack()
        for i, ph in enumerate(("pretraining", "transfer")):
            ax = axs[i, 3]
            for g in J.GROUPS:
                v = n.xs(g, level="reward_group")[ph].dropna()
                ax.hist(v, bins=20, histtype="step", color=COH[g], lw=1, label=f"{g.replace('-', '−')} median {int(v.median())}")
            ax.set_title(f"trials per mouse, {ph}"); ax.legend(frameon=False)
            ax.set_xlabel("trials (all days)")
    fig.suptitle("Targets: P(lick) per trial type (causal 100-trial window, time-normalised per phase); thin = mice, black = mean. "
                 "Each mouse's phases are stretched to 1000 points regardless of its number of trials (right).", fontsize=6.3)
    fig.tight_layout()
    save(fig, "2_data")


# ── 3 fits ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_fits(F):
    TM = targets_mean()
    ms = [m for m in MODELS if (m, "common") in F]
    if not ms:
        print("no common fits yet"); return
    fig, axs = plt.subplots(len(ms), 3, figsize=(7.2, 1.9 * len(ms) + 0.4), squeeze=False, sharey=True)
    for i, m in enumerate(ms):
        for var, ls in (("common", "-"), ("cohort b, a, v", ":")):
            if (m, var) not in F:
                continue
            mod, S = sim(F[(m, var)])
            xp, xt = np.arange(mod.N_PRE) - mod.N_PRE, np.arange(mod.N_TRANS)
            for j, (sk, tk, lab, _) in enumerate(TT):
                ax = axs[i, j]
                for g in J.GROUPS:
                    if var == "common":
                        ax.plot(np.r_[xp, xt], TM[g][tk][:2000], color=COH[g], lw=2.2, alpha=0.3, label=f"data {g.replace('-', '−')}" if j == 0 else None)
                    sp, st = S[g]
                    if sk in sp:
                        ax.plot(xp, sp[sk], color="k", lw=0.9, ls=ls, label=("model pretraining (shared)" if var == "common" else None) if (j == 0 and g == "R+") else None)
                    ax.plot(xt, st[sk], color=COH[g], lw=0.9, ls=ls, label=f"model {g.replace('-', '−')} ({var})" if j == 0 else None)
                ax.axvline(0, color="r", ls="--", lw=0.6); ax.set_ylim(-0.03, 1.03)
                ax.set_title(f"{m} — {lab}")
                if i == len(ms) - 1:
                    ax.set_xlabel("normalised trials (0 = whisker onset)")
        axs[i, 0].set_ylabel("P(lick)")
        r = F[(m, "common")]
        axs[i, 2].text(0.98, 0.62, f"loss {r['loss']:.4f}\nb {r['common']['b']:.2g}, a {r['common']['a']:.2f}\nv {r['common']['v']:.2f}, tr {r['common']['tr']:.2f}",
                       transform=axs[i, 2].transAxes, ha="right", fontsize=5.2, bbox=dict(fc="white", ec="0.8", lw=0.4))
    axs[0, 0].legend(frameon=False, fontsize=4.8, loc="lower right")
    fig.suptitle("Joint fits to the group means: thick = data, solid = common parameters (one pretraining; cohorts differ only in the whisker "
                 "reward), dotted = cohort-specific b, a, v in transfer", fontsize=6.3)
    fig.tight_layout()
    save(fig, "3_fits")


# ── 4 model comparison ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def comparison_table(F):
    TM = targets_mean()
    n = sum(int(np.isfinite(TM[g][tk][:1000]).sum()) for g in J.GROUPS for tk in ("auditory", "no_stim")) + \
        sum(int(np.isfinite(TM[g][tk][1000:2000]).sum()) for g in J.GROUPS for tk in ("auditory", "whisker", "no_stim"))
    rows = []
    for (m, var), r in F.items():
        mod, _ = J.load_model(m)
        k = len(J.DYN_KEYS) + len(J.weight_keys(mod)) + len(r["free_transfer"])          # each freed parameter adds one (two cohort values)
        if "tr" in r["free_transfer"]:
            k -= 1                                                                        # tr is not used by R-: no extra parameter
        rss = r["loss"] * n
        row = dict(model=m, variant=var, loss=r["loss"], k=k, n_points=n, aic=n * np.log(rss / n) + 2 * k, bic=n * np.log(rss / n) + k * np.log(n),
                   converged=r["converged"], n_eval=r["n_eval"])
        for g in J.GROUPS:
            for c in ("mse_pre_aud", "mse_pre_ns", "mse_trans_aud", "mse_trans_whisk", "mse_trans_ns"):
                row[f"{c}|{g}"] = r["mse"][g][c]
        rows.append(row)
    T = pd.DataFrame(rows)
    T["d_aic"] = T.aic - T.aic.min(); T["d_bic"] = T.bic - T.bic.min()
    T.to_csv(C.TAB / "005_model_comparison.csv", index=False)
    return T


def fig_compare(F):
    if not F:
        return
    T = comparison_table(F).sort_values(["model", "variant"])
    comps = ["mse_pre_aud", "mse_pre_ns", "mse_trans_aud", "mse_trans_whisk", "mse_trans_ns"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1.8, 1]))
    lab = [f"{a}\n{b}" for a, b in zip(T.model, T.variant)]
    ax = axs[0]; ax.barh(range(len(T)), T.loss, color=["0.4" if v == "common" else "0.75" for v in T.variant])
    ax.set_yticks(range(len(T)), lab, fontsize=5); ax.set_xlabel("loss (mean MSE, both cohorts)"); ax.invert_yaxis()
    ax = axs[1]
    M = np.array([[r[f"{c}|{g}"] for g in J.GROUPS for c in comps] for _, r in T.iterrows()])
    im = ax.imshow(M, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(comps) * 2), [f"{c.replace('mse_', '').replace('_', ' ')}\n{g.replace('-', '−')}" for g in J.GROUPS for c in comps], fontsize=4.6)
    ax.set_yticks(range(len(T)), lab, fontsize=5); fig.colorbar(im, ax=ax, label="MSE", shrink=0.8)
    for (y, x_), v in np.ndenumerate(M):
        ax.text(x_, y, f"{v:.3f}", ha="center", va="center", fontsize=3.8, color="w" if v > M.max() * 0.6 else "k")
    ax.set_title("error per curve and cohort")
    ax = axs[2]
    ax.barh(np.arange(len(T)) - 0.18, T.d_aic, height=0.36, color="#6a51a3", label="Δ AIC")
    ax.barh(np.arange(len(T)) + 0.18, T.d_bic, height=0.36, color="#9e9ac8", label="Δ BIC")
    ax.set_yticks(range(len(T)), [f"{a} / {b} (k={k})" for a, b, k in zip(T.model, T.variant, T.k)], fontsize=5); ax.invert_yaxis()
    ax.legend(frameon=False); ax.set_xlabel("relative to best (nominal)")
    fig.suptitle("Model comparison of the joint mean fits. AIC / BIC are nominal (autocorrelated, interpolated points): ranking only.", fontsize=6.3)
    fig.tight_layout()
    save(fig, "4_compare")


# ── 5 recovery ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_recovery():
    R = [pd.read_csv(p) for p in sorted(C.TAB.glob("002_recovery_*.csv"))]
    if not R:
        print("no recovery tables"); return
    R = pd.concat(R)
    ms = [m for m in MODELS if m in set(R.model)]
    P = J.DYN_KEYS + ["b·v"]
    fig, axs = plt.subplots(len(ms), len(P) + 1, figsize=(7.2, 1.45 * len(ms) + 0.3), squeeze=False)
    for i, m in enumerate(ms):
        d = R[R.model == m]
        for j, p in enumerate(P):
            ax = axs[i, j]
            if p == "b·v":
                w = d[d.param.isin(["b", "v"])].pivot(index="truth_id", columns="param", values=["true", "fitted"])
                t, f = w["true"]["b"] * w["true"]["v"], w["fitted"]["b"] * w["fitted"]["v"]
            else:
                q = d[d.param == p]; t, f = q.true.to_numpy(), q.fitted.to_numpy()
            ax.loglog(t, f, "o", ms=2.5, color="k", alpha=0.7)
            lo, hi = min(np.min(t), np.min(f)) / 1.3, max(np.max(t), np.max(f)) * 1.3
            ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            r = np.corrcoef(np.log10(t), np.log10(f))[0, 1]
            ax.set_title(f"{m}: {p}  r = {r:.2f}", fontsize=5.8)
            if i == len(ms) - 1:
                ax.set_xlabel("true")
            if j == 0:
                ax.set_ylabel("fitted")
        ax = axs[i, -1]
        L = d.drop_duplicates("truth_id")
        ax.hist(np.log10(L.loss + 1e-12), bins=12, color="0.5"); ax.set_title(f"{m}: log10 loss at fit", fontsize=5.8)
        ax.set_xlabel("log10 MSE (0 = perfect)")
    fig.suptitle("Parameter recovery: 12 parameter sets drawn ±0.3 decades around the defaults, targets simulated with one common pretraining, "
                 "refitted from the default start", fontsize=6.3)
    fig.tight_layout()
    save(fig, "5_recovery")


# ── 6 latents ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def trace(mod, Model, common, over, seed=0):
    """per trial: P(lick) per stimulus and the whisker drive split into shared-node and direct parts (common pretraining, then each cohort)"""
    rng = np.random.RandomState(seed)
    p = mod.Params(**{k: common[k] for k in mod.Params.__dataclass_fields__ if k in common})
    m = Model(p=p)

    def rec(mm):
        dq = mm.dQ(mod.STIM["whisker"])
        node = float((mm.Q_exc_out - mm.Q_inh_out) * mm.Q_in[1]) if hasattr(mm, "Q_in") else 0.0
        return dict(p_aud=mm.p_lick(mod.STIM["auditory"]), p_whisk=mm.p_lick(mod.STIM["whisker"]), p_ns=mm.p_lick(mod.STIM["nostim"]),
                    dq_whisk=dq, node_whisk=node, direct_whisk=dq - node)

    def run(mm, trials):
        out = []
        for t in trials:
            mm._update(t["x"], t["R_if_lick"], t["whisker_reward"], t["stim"]); out.append(rec(mm))
        return pd.DataFrame(out)
    pre = run(m, mod.make_pretrain_trials(mod.N_PRE, rng=rng))
    import copy
    res = {}
    for i, g in enumerate(J.GROUPS):
        mg = copy.deepcopy(m)
        for k, v in over.get(g, {}).items():
            setattr(mg.p, k, v)
        res[g] = run(mg, mod.make_transfer_trials(g, mod.N_TRANS, rng=np.random.RandomState(seed + 1 + i)))
    return pre, res


def fig_latents(F):
    ms = [m for m in MODELS if (m, "common") in F]
    if not ms:
        return
    fig, axs = plt.subplots(2, len(ms), figsize=(2.4 * len(ms), 3.6), squeeze=False)
    for j, m in enumerate(ms):
        mod, Model = J.load_model(m); r = F[(m, "common")]
        pre, res = trace(mod, Model, r["common"], r["transfer_overrides"])
        xp, xt = np.arange(len(pre)) - len(pre), np.arange(mod.N_TRANS)
        ax = axs[0, j]
        ax.plot(xp, pre.dq_whisk, color="k", lw=1, label="ΔQ whisker (total)")
        ax.plot(xp, pre.node_whisk, color="#38761d", lw=0.8, ls="--", label="via shared node")
        ax.plot(xp, pre.direct_whisk, color="0.5", lw=0.8, ls=":", label="direct + context")
        for g in J.GROUPS:
            ax.plot(xt, res[g].dq_whisk, color=COH[g], lw=1)
            ax.plot(xt, res[g].node_whisk, color=COH[g], lw=0.8, ls="--")
            ax.plot(xt, res[g].direct_whisk, color=COH[g], lw=0.8, ls=":")
        ax.axhline(0, color="0.7", lw=0.4); ax.axvline(0, color="r", ls="--", lw=0.6)
        ax.set_title(f"{m}: whisker drive ΔQ(W)"); ax.set_xlabel("trials")
        if j == 0:
            ax.legend(frameon=False, fontsize=4.8); ax.set_ylabel("ΔQ (Φ⁻¹ of P(lick))")
        ax = axs[1, j]
        for g in J.GROUPS:
            ax.plot(xt, res[g].p_whisk, color=COH[g], lw=1, label=f"whisker {g.replace('-', '−')}")
            ax.plot(xt, res[g].p_aud, color=COH[g], lw=0.7, ls="--", label=f"auditory {g.replace('-', '−')}")
            ax.plot(xt, res[g].p_ns, color=COH[g], lw=0.7, ls=":", label=f"no stim {g.replace('-', '−')}")
        ax.plot(xp, pre.p_whisk, color="k", lw=1, label="whisker (untested in pretraining)")
        ax.axvline(0, color="r", ls="--", lw=0.6); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("trials")
        ax.set_title(f"{m}: model P(lick)")
        if j == 0:
            ax.legend(frameon=False, fontsize=4.3); ax.set_ylabel("P(lick)")
    fig.suptitle("Model latents (common-parameter fits): whisker drive ΔQ(W) split into the shared-node part (generalisation from "
                 "auditory) and the direct + context-bias part; the pretraining state is common, transfer splits it by cohort", fontsize=6.3)
    fig.tight_layout()
    save(fig, "6_latents")


# ── 7 cohorts ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_cohorts(F):
    ms = [m for m in MODELS if (m, "cohort b, a, v") in F and (m, "common") in F]
    if not ms:
        print("no free b a v fits yet"); return
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for j, p in enumerate(("b", "a", "v")):
        ax = axs[j]
        for i, m in enumerate(ms):
            r = F[(m, "cohort b, a, v")]
            for k, g in enumerate(J.GROUPS):
                ax.bar(i + (k - 0.5) * 0.35, r["transfer_overrides"][g][p], width=0.33, color=COH[g], label=g.replace("-", "−") if i == 0 else None)
            ax.plot([i - 0.35, i + 0.35], [r["common"][p]] * 2, color="k", lw=0.8, label="pretraining (shared)" if i == 0 else None)
            ax.plot([i - 0.35, i + 0.35], [F[(m, "common")]["common"][p]] * 2, color="0.5", lw=0.8, ls="--", label="common fit" if i == 0 else None)
        ax.set_xticks(range(len(ms)), ms); ax.set_title(f"transfer {p}")
        if p == "b":
            ax.set_yscale("log")
    axs[0].legend(frameon=False, fontsize=4.8)
    ax = axs[3]
    for i, m in enumerate(ms):
        a_, b_ = F[(m, "common")], F[(m, "cohort b, a, v")]
        for k, g in enumerate(J.GROUPS):
            ax.bar(i + (k - 0.5) * 0.35, 100 * (1 - b_["mse"][g]["mse_trans"] / a_["mse"][g]["mse_trans"]), width=0.33, color=COH[g])
    ax.axhline(0, color="0.6", lw=0.5); ax.set_xticks(range(len(ms)), ms); ax.set_title("transfer error reduction\nwith cohort b, a, v (%)")
    fig.suptitle("Cohort comparison: does the transfer need cohort-specific learning parameters? (pretraining shared in both variants)", fontsize=6.3)
    fig.tight_layout()
    save(fig, "7_cohorts")


# ── 8 mice ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_mice():
    P = sorted(C.FITS.glob("004_*_individual_common_dynamics.csv"))
    if not P:
        print("no per-mouse fits yet"); return
    TI = targets_ind()
    for p in P:
        m = p.name.split("_")[1]
        mod, Model = J.load_model(m)
        D = pd.read_csv(p)
        wk = J.weight_keys(mod)
        rows = []
        for q in wk + ["mse_global", "mse_pre", "mse_trans", "mse_trans_whisk"]:
            a, b = D[D.group == "R+"][q].dropna(), D[D.group == "R-"][q].dropna()
            x, y = (np.log10(a), np.log10(b)) if q in wk else (a, b)
            rows.append(dict(model=m, measure=q, n_rp=len(a), n_rm=len(b), median_rp=a.median(), median_rm=b.median(),
                             p_mwu=stats.mannwhitneyu(x, y).pvalue, p_welch=stats.ttest_ind(x, y, equal_var=False).pvalue))
        S = pd.DataFrame(rows)
        for c in ("p_mwu", "p_welch"):
            ok = S.measure.isin(wk)
            S.loc[ok, f"q_{c}"] = stats.false_discovery_control(S.loc[ok, c]) if ok.sum() else np.nan
        S.to_csv(C.TAB / f"005_individual_stats_{m}.csv", index=False)
        fig = plt.figure(figsize=(7.2, 6.0))
        gs = fig.add_gridspec(3, 6, hspace=0.6, wspace=0.5)
        ex = []
        for g in J.GROUPS:
            d = D[D.group == g].sort_values("mse_global")
            ex += [(g, d.mouse_id.iloc[k]) for k in (0, len(d) // 2, len(d) - 1)]
        for e, (g, mid) in enumerate(ex):
            ax = fig.add_subplot(gs[e // 3, (e % 3) * 2:(e % 3) * 2 + 2])
            r = D[D.mouse_id == mid].iloc[0]
            par = {k: r[k] for k in J.DYN_KEYS + wk}
            S_ = J.simulate_joint(mod, Model, par, {})
            sp, st = S_[g]; t = TI[mid]
            xp, xt = np.arange(mod.N_PRE) - mod.N_PRE, np.arange(mod.N_TRANS)
            for sk, tk, lab, col in TT:
                ax.plot(np.r_[xp, xt], t[tk][:2000], color=col, lw=1.6, alpha=0.35)
                if sk in sp:
                    ax.plot(xp, sp[sk], color=col, lw=0.7)
                ax.plot(xt, st[sk], color=col, lw=0.7)
            ax.axvline(0, color="r", ls="--", lw=0.5); ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{g.replace('-', '−')} {mid} ({['best', 'median', 'worst'][e % 3]}), MSE {r.mse_global:.3f}", fontsize=5.8, color=COH[g])
        ax = fig.add_subplot(gs[2, 0:2])
        for k, g in enumerate(J.GROUPS):
            v = D[D.group == g].mse_global
            ax.scatter(np.full(len(v), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=4, color=COH[g], alpha=0.5)
            ax.errorbar(k, v.median(), fmt="o", color="k", ms=3)
        s = S[S.measure == "mse_global"].iloc[0]
        ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_ylabel("per-mouse loss"); ax.set_title(f"fit quality: MWU p = {s.p_mwu:.2g}, Welch p = {s.p_welch:.2g}", fontsize=5.6)
        ax = fig.add_subplot(gs[2, 2:6])
        for i, q in enumerate(wk):
            for k, g in enumerate(J.GROUPS):
                v = np.log10(D[D.group == g][q])
                ax.scatter(np.full(len(v), i + (k - 0.5) * 0.35) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), v, s=2.5, color=COH[g], alpha=0.45)
                ax.plot([i + (k - 0.5) * 0.35 - 0.12, i + (k - 0.5) * 0.35 + 0.12], [v.median()] * 2, color="k", lw=0.8)
            s = S[S.measure == q].iloc[0]
            if s.p_mwu < 0.05 or s.p_welch < 0.05:
                ax.text(i, 0.97, "*" if min(s.q_p_mwu, s.q_p_welch) >= 0.05 else "**", ha="center", va="top", fontsize=7,
                        transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(wk)), [q.replace("Q_", "") for q in wk], rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("log10 initial weight"); ax.set_title("fitted initial weights R+ vs R− (set before pretraining: expected equal; "
                                                            "* p < .05 MWU or Welch, ** BH q < .05)", fontsize=5.6)
        fig.suptitle(f"Per-mouse fits ({m}; b, a, v, tr fixed to the joint mean fit for every mouse, initial weights per mouse): "
                     "thick = data, thin = model", fontsize=6.3)
        save(fig, f"8_mice_{m}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=list(range(1, 9)))
    a = ap.parse_args()
    F = load_fits()
    fns = {1: fig_model, 2: fig_data, 3: lambda: fig_fits(F), 4: lambda: fig_compare(F), 5: fig_recovery, 6: lambda: fig_latents(F),
           7: lambda: fig_cohorts(F), 8: fig_mice}
    for k in a.only:
        try:
            fns[k]()
        except Exception as e:  # noqa: BLE001
            print(f"figure {k} failed: {e!r}")
            raise
