"""008 -- Figures of the redesigned model (v2: rl2_model, fits 006, evaluation 007).
  1 model        features, learning rule, outcomes per cohort (the only cohort difference), variants
  2 comparison   per-mouse BIC of the variants (summed, and best variant per mouse by cohort), pooled common vs cohort-specific LRT,
                 model recovery confusion matrix
  3 parameters   "full" fits per mouse, R+ vs R- (Mann-Whitney + Welch, BH over parameters): if the contingency explains the cohort
                 difference, no parameter should differ
  4 predictive   data vs one-step model predictions vs closed-loop simulations, per cohort and trial type (time-normalised as the targets)
  5 swapped      each mouse simulated with its own parameters under its own vs the other cohort's whisker outcome
  6 latents      weights at whisker onset (how much of the day-0 whisker drive is generalisation vs general lick tendency); example
                 trajectories over trials
  7 recovery     parameter recovery (true vs refitted, per mouse)
  8 examples     best / median / worst mice per cohort (NLL per trial): data, one-step and generative curves
Statistics: tables/008_model_comparison.csv, 008_param_stats.csv.
Run (haas): python 008_figures_v2.py [--only 1 2 ...]
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402
import rl2_model as M  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
TTL = [("auditory", "Auditory", "#1f78b4"), ("whisker", "Whisker", "#d95f02"), ("no_stim", "No stimulus (FA)", "0.35")]
STRAT = ["full", "specific", "general", "no_bias", "symmetric", "ns_weight"]
SHOW = {"full": "full", "specific": "specific (γ=0)", "general": "general (γ=1)", "no_bias": "no bias learning (κ=0)",
        "symmetric": "symmetric α", "ns_weight": "+ no-stim weight"}
PLAB = {"alpha_pos": "α+", "alpha_neg": "α−", "beta": "β", "gamma": "γ generalisation", "kappa": "κ bias learning", "w0": "w0 initial bias",
        "eps": "ε lapse", "beta_x_alpha_pos": "β·α+ (speed)", "alpha_ratio": "α− / α+"}


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"008_{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fits():
    F = pd.read_csv(C.FITS / "006_fits.csv")
    F["beta_x_alpha_pos"] = F.beta * F.alpha_pos
    F["alpha_ratio"] = F.alpha_neg / F.alpha_pos
    return F


def both_tests(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


# ── 1 model ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_model():
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_axes([0.0, 0.42, 0.42, 0.56]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    feats = [("bias (every trial)", 0.85, "#f4f4f4", "w_b  · κ"), ("stimulus present S\n(A or W)", 0.62, "#d9ead3", "w_S  · γ"),
             ("auditory A", 0.39, "#cfe2f3", "w_A  · (1−γ)"), ("whisker W", 0.16, "#fde0c5", "w_W  · (1−γ)")]
    for txt, y, fc, lab in feats:
        ax.add_patch(plt.Rectangle((0.02, y - 0.08), 0.33, 0.16, fc=fc, ec="k", lw=0.6))
        ax.text(0.185, y, txt, ha="center", va="center", fontsize=5.8)
        ax.annotate("", xy=(0.72, 0.5), xytext=(0.36, y), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3"))
        ax.text(0.47, y + (0.5 - y) * 0.3 + 0.03, lab, fontsize=5.2, color="0.25")
    ax.add_patch(plt.Rectangle((0.72, 0.38), 0.27, 0.24, fc="#fff2cc", ec="k", lw=0.6))
    ax.text(0.855, 0.5, "V → P(lick)\n= ε/2 + (1−ε)·σ(β·V)", ha="center", va="center", fontsize=5.6)
    ax = fig.add_axes([0.44, 0.42, 0.56, 0.56]); ax.axis("off")
    ax.text(0, 0.98, "One learning rule for every mouse", weight="bold", fontsize=7, va="top")
    ax.text(0, 0.86, "V = w_b + γ·w_S + (1−γ)·w_X        X = A or W;  no-stimulus trial: V = w_b\n"
                     "after a lick: δ = r − V,   α = α+ if δ > 0 else α−\n"
                     "  w_b += α·κ·δ     w_S += α·γ·δ     w_X += α·(1−γ)·δ\n"
                     "no lick: no outcome, no update\n"
                     "start of pretraining: w_b = w0, w_S = w_A = w_W = 0", fontsize=5.6, va="top", family="monospace")
    ax.text(0, 0.42, "Outcome r of a lick (the only difference between cohorts):", fontsize=6, va="top", weight="bold")
    tab = [["", "auditory", "whisker", "no stimulus"], ["R+", "+1", "+1", "−1"], ["R−", "+1", "−1", "−1"]]   # display minus
    t = ax.table(cellText=tab, cellLoc="center", bbox=[0, 0.05, 0.62, 0.32])
    t.auto_set_font_size(False); t.set_fontsize(5.8)
    for (i, j), c in t.get_celld().items():
        c.set_linewidth(0.4)
        if i > 0 and j == 0:
            c.get_text().set_color(COH[tab[i][0].replace("−", "-")])
    ax = fig.add_axes([0.02, 0.0, 0.96, 0.38]); ax.axis("off")
    ax.text(0, 0.95, "Strategy parameters and variants (each mouse fitted by maximum likelihood of its own trial-by-trial licks)", weight="bold",
            fontsize=6.5, va="top")
    ax.text(0, 0.78, "γ  generalisation: share of stimulus learning credited to 'a stimulus was presented' (γ = 1: all auditory learning transfers to\n"
                     "    the whisker stimulus at day 0; γ = 0: stimulus-specific, the whisker is learnt from scratch)\n"
                     "κ  bias learning: how much each outcome changes the general lick tendency (no-stimulus licking, and the floor of all trial types)\n"
                     "α+, α−  learning from better / worse than expected outcomes;  β, ε  choice determinism and lapses;  w0  initial lick tendency\n"
                     "Variants: full | specific (γ = 0) | general (γ = 1) | no bias learning (κ = 0) | symmetric (α− = α+) | + no-stimulus weight (own\n"
                     "rate; tests whether a separate no-stimulus node adds anything beyond the bias, the question behind 'context vs no-stim')",
            fontsize=5.6, va="top")
    save(fig, "1_model")


# ── 2 comparison ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_comparison():
    F = fits()
    B = F.pivot(index="mouse_id", columns="variant", values="bic"); coh = F.drop_duplicates("mouse_id").set_index("mouse_id").cohort
    D = B.sub(B["full"], axis=0)
    rows = []
    for v in STRAT:
        for c in ("all", "R+", "R-"):
            x = D[v] if c == "all" else D[v][coh == c]
            rows.append(dict(variant=v, cohort=c, n=len(x), sum_dbic=x.sum(), median_dbic=x.median(),
                             n_better_than_full=int((x < 0).sum()), p_wilcoxon=stats.wilcoxon(x).pvalue if (x != 0).sum() > 5 else np.nan))
    T = pd.DataFrame(rows); T.to_csv(C.TAB / "008_model_comparison.csv", index=False)
    best = B.idxmin(axis=1)
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.4), gridspec_kw=dict(width_ratios=[1.3, 1.1, 0.95, 1.1], wspace=0.9))
    ax = axs[0]
    for k, c in enumerate(("R+", "R-")):
        y = [T[(T.variant == v) & (T.cohort == c)].sum_dbic.iloc[0] for v in STRAT[1:]]
        ax.barh(np.arange(len(y)) + (k - 0.5) * 0.38, y, height=0.36, color=COH[c], label=c.replace("-", "−"))
    ax.axvline(0, color="k", lw=0.5); ax.set_yticks(range(len(STRAT) - 1), [SHOW[v] for v in STRAT[1:]], fontsize=5.3); ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("Σ ΔBIC vs full (> 0: full better)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax = axs[1]
    ct = pd.crosstab(best, coh).reindex(STRAT).fillna(0)
    for k, c in enumerate(("R+", "R-")):
        ax.barh(np.arange(len(STRAT)) + (k - 0.5) * 0.38, ct[c] / ct[c].sum(), height=0.36, color=COH[c])
    ax.set_yticks(range(len(STRAT)), [SHOW[v] for v in STRAT], fontsize=5.3); ax.invert_yaxis(); ax.set_xlabel("fraction of mice best (BIC)")
    ax = axs[2]; ax.axis("off")
    P = json.loads((C.FITS / "006_pooled.json").read_text())
    ax.text(0, 0.95, "Pooled fits (all mice\none parameter set)", weight="bold", fontsize=6, va="top")
    ax.text(0, 0.72, f"common: NLL {P['all']['nll']:.0f}\nR+ only: {P['R+']['nll']:.0f}\nR− only: {P['R-']['nll']:.0f}\n"
                    f"LRT cohort-specific vs\ncommon: χ²({P['lrt']['df']}) = {P['lrt']['stat']:.1f}\np = {P['lrt']['p']:.2g}\n"
                    "(anti-conservative: ignores\nbetween-mouse variability)", fontsize=5.4, va="top")
    ax = axs[3]
    p = C.TAB / "007_model_recovery.csv"
    if p.exists():
        R = pd.read_csv(p); S4 = ["full", "specific", "general", "no_bias"]
        cm = pd.crosstab(R.generating, R.recovered, normalize="index").reindex(index=S4, columns=S4).fillna(0)
        ax.imshow(cm.to_numpy(), cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="w" if v > 0.5 else "k")
        ax.set_xticks(range(4), [SHOW[s].split(" (")[0] for s in S4], rotation=40, ha="right", fontsize=5)
        ax.set_yticks(range(4), [SHOW[s].split(" (")[0] for s in S4], fontsize=5)
        ax.set_xlabel("recovered (BIC)"); ax.set_title("model recovery\n(rows: generating model)")
    fig.suptitle("Model comparison (per-mouse maximum likelihood; BIC = 2·NLL + k·ln n_trials)", fontsize=6.5)
    save(fig, "2_comparison")


# ── 3 parameters ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_parameters():
    F = fits(); F = F[F.variant == "full"]
    P = ["gamma", "kappa", "alpha_ratio", "beta_x_alpha_pos", "alpha_pos", "alpha_neg", "beta", "w0", "eps"]
    logp = {"alpha_ratio", "beta_x_alpha_pos", "alpha_pos", "alpha_neg", "beta"}
    rows = []
    for q in P:
        a, b = F[F.cohort == "R+"][q], F[F.cohort == "R-"][q]
        x, y = (np.log10(a), np.log10(b)) if q in logp else (a, b)
        pm, pw = both_tests(x, y)
        rows.append(dict(param=q, n_rp=len(a), n_rm=len(b), median_rp=a.median(), median_rm=b.median(), p_mwu=pm, p_welch=pw))
    S = pd.DataFrame(rows)
    S["q_mwu"] = stats.false_discovery_control(S.p_mwu); S["q_welch"] = stats.false_discovery_control(S.p_welch)
    S.to_csv(C.TAB / "008_param_stats.csv", index=False)
    fig, axs = plt.subplots(1, len(P), figsize=(7.2, 2.1))
    for ax, q in zip(axs, P):
        for k, c in enumerate(("R+", "R-")):
            v = F[F.cohort == c][q]
            ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=3, color=COH[c], alpha=0.5, lw=0)
            ax.plot([k - 0.22, k + 0.22], [v.median()] * 2, color="k", lw=0.9)
        if q in logp:
            ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
        s = S[S.param == q].iloc[0]
        ax.set_title(f"{PLAB[q]}\nMWU p={s.p_mwu:.2g}\nWelch p={s.p_welch:.2g}", fontsize=5.2)
        ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_xlim(-0.6, 1.6)
    fig.suptitle("Per-mouse parameters of the full model (one learning rule; the cohorts differ only in the whisker outcome). "
                 "Black = median. Log-scaled parameters tested on log10. BH q in 008_param_stats.csv", fontsize=6.2)
    fig.tight_layout()
    save(fig, "3_parameters")


# ── 4 / 5 / 8 curves ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_curves():
    z = np.load(C.TAB / "007_curves.npz", allow_pickle=True)
    return z["mouse_id"], z["cohort"], list(z["kinds"]), z["curves"]          # mice x kinds x 3 x 2000


def band(ax, x, Y, col, lab, lw=1.0, ls="-", alpha=0.15):
    m, s = np.nanmean(Y, 0), np.nanstd(Y, 0) / np.sqrt(np.sum(np.isfinite(Y), 0))
    ax.plot(x, m, color=col, lw=lw, ls=ls, label=lab); ax.fill_between(x, m - s, m + s, color=col, alpha=alpha, lw=0)


def fig_predictive():
    mid, coh, kinds, Cv = load_curves(); x = np.linspace(-1, 1, 2000)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.6), sharey=True)
    for i, c in enumerate(("R+", "R-")):
        for j, (tk, lab, _) in enumerate(TTL):
            ax = axs[i, j]; sel = coh == c
            band(ax, x, Cv[sel, kinds.index("data"), j], "k", "data", lw=1.6, alpha=0.12)
            band(ax, x, Cv[sel, kinds.index("onestep"), j], COH[c], "model, one-step", lw=0.9)
            band(ax, x, Cv[sel, kinds.index("generative"), j], COH[c], "model, closed-loop simulation", lw=0.9, ls="--", alpha=0.08)
            ax.axvline(0, color="r", ls="--", lw=0.6); ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{c.replace('-', '−')} — {lab} (n = {sel.sum()})")
            if i == 1:
                ax.set_xlabel("normalised time (0 = whisker onset)")
        axs[i, 0].set_ylabel("P(lick)")
    axs[0, 0].legend(frameon=False, fontsize=5)
    fig.suptitle("Predictive check: one-step = model P(lick) given each mouse's actual history; closed-loop = the model run on the mouse's "
                 "trial sequence with its own simulated licks (20 runs). Mean ± s.e.m. over mice", fontsize=6.2)
    fig.tight_layout()
    save(fig, "4_predictive")


def fig_swapped():
    mid, coh, kinds, Cv = load_curves(); x = np.linspace(-1, 1, 2000)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.6), sharey=True)
    for i, c in enumerate(("R+", "R-")):
        other = "R-" if c == "R+" else "R+"
        for j, (tk, lab, _) in enumerate(TTL):
            ax = axs[i, j]; sel = coh == c
            band(ax, x, Cv[sel, kinds.index("data"), j], "k", "data", lw=1.4, alpha=0.1)
            band(ax, x, Cv[sel, kinds.index("generative"), j], COH[c], f"simulated, own contingency ({c.replace('-', '−')})", lw=1)
            band(ax, x, Cv[sel, kinds.index("swapped"), j], COH[other], f"simulated, swapped ({other.replace('-', '−')} outcome)", lw=1, ls="--")
            ax.axvline(0, color="r", ls="--", lw=0.6); ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{c.replace('-', '−')} mice — {lab}")
            if i == 1:
                ax.set_xlabel("normalised time (0 = whisker onset)")
        axs[i, 0].set_ylabel("P(lick)"); axs[i, 0].legend(frameon=False, fontsize=4.8)
    fig.suptitle("Is the cohort difference produced by the contingency alone? Each mouse simulated with its own parameters under its own "
                 "and under the other cohort's whisker outcome", fontsize=6.2)
    fig.tight_layout()
    save(fig, "5_swapped")


def fig_examples():
    mid, coh, kinds, Cv = load_curves(); x = np.linspace(-1, 1, 2000)
    F = fits(); F = F[F.variant == "full"].set_index("mouse_id"); F["nll_per_trial"] = F.nll / F.n
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.8), sharey=True)
    for i, c in enumerate(("R+", "R-")):
        f = F[F.cohort == c].sort_values("nll_per_trial")
        for j, (k, lab) in enumerate(((0, "best"), (len(f) // 2, "median"), (len(f) - 1, "worst"))):
            m = f.index[k]; ax = axs[i, j]; w = np.where(mid == m)[0]
            if not len(w):
                continue
            for t, (_, tl, col) in enumerate(TTL):
                ax.plot(x, Cv[w[0], kinds.index("data"), t], color=col, lw=1.6, alpha=0.35)
                ax.plot(x, Cv[w[0], kinds.index("onestep"), t], color=col, lw=0.8)
                ax.plot(x, Cv[w[0], kinds.index("generative"), t], color=col, lw=0.7, ls=":")
            r = f.loc[m]
            ax.set_title(f"{c.replace('-', '−')} {m} ({lab}); NLL/trial {r.nll_per_trial:.3f}\nγ {r.gamma:.2f}, κ {r.kappa:.2f}, α−/α+ {r.alpha_ratio:.2g}",
                         fontsize=5.5, color=COH[c])
            ax.axvline(0, color="r", ls="--", lw=0.6); ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Example mice: thick = data, thin = one-step model, dotted = closed-loop simulation (blue auditory, orange whisker, grey no stimulus)",
                 fontsize=6.2)
    fig.tight_layout()
    save(fig, "8_examples")


# ── 6 latents ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_latents():
    L = pd.read_parquet(C.TAB / "006_latents.parquet")
    F = fits(); F = F[F.variant == "full"].set_index("mouse_id")
    on = L[L.day == 0].groupby("mouse_id").head(1).set_index("mouse_id")         # state before the first trial of day 0
    on = on.join(F[["gamma", "beta"]])
    on["gen"] = on.gamma * on.w_S; on["bias"] = on.w_b
    on["p_w0"] = L[(L.day == 0) & (L.trial_type == "whisker")].groupby("mouse_id").p_lick.first()     # model P(lick) on the first whisker trial
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    for k, c in enumerate(("R+", "R-")):
        d = on[on.cohort == c]
        for j, (q, lab) in enumerate((("bias", "w_b"), ("gen", "γ·w_S"))):
            v = d[q]; xx = j * 2.4 + k
            ax.scatter(np.full(len(v), xx) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=3, color=COH[c], alpha=0.5, lw=0)
            ax.plot([xx - 0.22, xx + 0.22], [v.median()] * 2, color="k", lw=0.9)
    ax.set_xticks([0.5, 2.9], ["bias w_b", "generalised\nγ·w_S"]); ax.axhline(0, color="0.6", lw=0.5)
    ax.set_title("whisker value at day-0 onset\n(V_W = w_b + γ·w_S)", fontsize=6)
    ax = fig.add_subplot(gs[0, 1])
    for c in ("R+", "R-"):
        d = on[on.cohort == c]
        ax.scatter(d.gamma, d.p_w0, s=4, color=COH[c], alpha=0.6, label=c.replace("-", "−"))
    ax.set_xlabel("γ"); ax.set_ylabel("model P(lick | whisker) at onset"); ax.legend(frameon=False)
    ax.set_title("generalisation vs initial whisker licking", fontsize=6)
    ax = fig.add_subplot(gs[0, 2:])
    first = L[(L.day == 0) & (L.trial_type == "whisker")].groupby("mouse_id").head(10)                 # first 10 whisker trials of day 0
    obs = first.groupby("mouse_id").lick.mean()
    pred = first.groupby("mouse_id").p_lick.mean()
    fa = L[(L.day == -1) & (L.trial_type == "no_stim")].groupby("mouse_id").lick.mean()
    J = pd.DataFrame(dict(obs=obs, pred=pred, fa=fa)).join(on.cohort).dropna()
    for c in ("R+", "R-"):
        d = J[J.cohort == c]
        ax.scatter(d.fa, d.obs, s=5, color=COH[c], alpha=0.6, label=f"{c.replace('-', '−')}")
    lim = [0, 1]; ax.plot(lim, lim, color="0.6", lw=0.5, ls="--")
    r, p = stats.pearsonr(J.fa, J.obs)
    ax.set_xlabel("FA rate, last pretraining day"); ax.set_ylabel("whisker lick rate,\nfirst 10 whisker trials")
    ax.set_title(f"is early whisker licking above the general lick tendency? r = {r:.2f}, p = {p:.2g}\n"
                 f"whisker − FA: median {np.median(J.obs - J.fa):.2f}, Wilcoxon p = {stats.wilcoxon(J.obs - J.fa).pvalue:.2g}", fontsize=5.6)
    ax.legend(frameon=False)
    ex = []
    for c in ("R+", "R-"):
        f = F[F.cohort == c].sort_values("gamma")
        ex += [f.index[len(f) // 4], f.index[3 * len(f) // 4]]
    for e, m in enumerate(ex):
        ax = fig.add_subplot(gs[1, e]); d = L[L.mouse_id == m].reset_index(drop=True)
        for q, col in (("w_b", "0.3"), ("w_S", "#38761d"), ("w_A", "#1f78b4"), ("w_W", "#d95f02")):
            ax.plot(d.index, d[q], color=col, lw=0.8, label=q)
        on0 = d.index[d.day == 0][0]; ax.axvline(on0, color="r", ls="--", lw=0.6)
        for b in d.index[d.day.diff().fillna(0) != 0]:
            ax.axvline(b, color="0.85", lw=0.4)
        ax.set_title(f"{m} ({F.loc[m, 'cohort'].replace('-', '−')}), γ = {F.loc[m, 'gamma']:.2f}, κ = {F.loc[m, 'kappa']:.2f}", fontsize=5.6,
                     color=COH[F.loc[m, "cohort"]])
        ax.set_xlabel("trial")
        if e == 0:
            ax.legend(frameon=False, fontsize=4.8); ax.set_ylabel("weight")
    fig.suptitle("Latents of the full model: weights before the first whisker trial, and example weight trajectories "
                 "(bottom: grey = day boundaries, red = whisker onset)", fontsize=6.5)
    save(fig, "6_latents")


# ── 7 recovery ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_recovery():
    R = pd.read_csv(C.TAB / "007_param_recovery.csv")
    P = ["gamma", "kappa", "alpha_ratio", "beta_x_alpha_pos", "alpha_pos", "alpha_neg", "beta", "w0", "eps"]
    logp = {"alpha_ratio", "beta_x_alpha_pos", "alpha_pos", "alpha_neg", "beta", "eps"}
    fig, axs = plt.subplots(1, len(P), figsize=(7.2, 1.6))
    for ax, q in zip(axs, P):
        d = R[R.param == q]
        for c in ("R+", "R-"):
            e = d[d.cohort == c]; ax.scatter(e.true, e.fitted, s=3, color=COH[c], alpha=0.6, lw=0)
        t, f = d.true.to_numpy(float), d.fitted.to_numpy(float)
        if q in logp:
            ax.set_xscale("log"); ax.set_yscale("log"); ok = (t > 0) & (f > 0); r = stats.spearmanr(t[ok], f[ok])[0]
            for a_ in (ax.xaxis, ax.yaxis):
                a_.set_minor_formatter(NullFormatter())
        else:
            r = stats.spearmanr(t, f)[0]
        lo, hi = np.nanmin([t.min(), f.min()]), np.nanmax([t.max(), f.max()])
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--")
        ax.set_title(f"{PLAB[q]}\nρ = {r:.2f}", fontsize=5.4); ax.tick_params(labelsize=4.5)
    axs[0].set_ylabel("refitted"); fig.text(0.5, -0.02, "true (each mouse's fitted parameters; licks simulated on its own trial sequence)",
                                            ha="center", fontsize=5.6)
    fig.tight_layout()
    save(fig, "7_recovery")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=list(range(1, 9)))
    a = ap.parse_args()
    fns = {1: fig_model, 2: fig_comparison, 3: fig_parameters, 4: fig_predictive, 5: fig_swapped, 6: fig_latents, 7: fig_recovery, 8: fig_examples}
    for k in a.only:
        fns[k]()
