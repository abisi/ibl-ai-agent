"""012 -- Model comparison and fit performance of model v3, first iteration (user 2026-10-18: "review performance fits and compare
models"). Inputs: fits/011_full.csv, fits/011_loso.csv, tables/011_sequences.parquet.
Model comparison (held-out = leave-one-session-out log-likelihood, LL):
  1 models       held-out LL of each model relative to the core model, summed over mice (all sessions, and the day-0 session alone);
                 best model per mouse; random-effects Bayesian model selection (Stephan et al. 2009; held-out LL as log evidence):
                 expected model frequencies and exceedance probabilities
  2 components   per component: mean over the 32 model pairs with vs without it of the held-out LL difference, per mouse; Wilcoxon
                 across mice, fraction of mice improved; family-level BMS (probability that a mouse's model contains the component);
                 the same on the day-0 session; by cohort
Fit performance:
  3 performance  per mouse, held-out: LL per trial, geometric-mean predicted probability of the observed choice, AUC of P(lick) for the
                 observed licks; core vs winning vs full model; held-out LL per trial by training day
  4 curves       one-step P(lick) (full-data fit) vs data: time-normalised learning curves per cohort and trial type, and within-session
                 curves (lick rate vs trial position within the session, per stage: pretraining, day 0, later whisker days)
  5 parameters   full-data parameters of the winning model, R+ vs R- (Mann-Whitney + Welch, BH)
Output: figures/012_*.{png,pdf,svg}; tables/012_models.csv, 012_components.csv, 012_performance.csv, 012_params.csv
Run (haas): python 012_compare_v3.py [--tag _test3] [--only 1 2 ...]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import digamma  # noqa: E402

import common as C  # noqa: E402
import rl3_model as M  # noqa: E402
import rl_data_utils as du  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 6.8, "axes.labelsize": 6.5, "legend.fontsize": 5.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
TTL = [("auditory", "Auditory", "#1f78b4"), ("whisker", "Whisker", "#d95f02"), ("no_stim", "No stimulus (FA)", "0.35")]
FULL = "GMNZRP"


def lab(m):
    return m if m else "core"


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"012_{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load(tag):
    F = pd.read_csv(C.FITS / f"011_full{tag}.csv", keep_default_na=False, na_values=[""]).assign(model=lambda d: d.model.astype(str))
    L = pd.read_csv(C.FITS / f"011_loso{tag}.csv", keep_default_na=False, na_values=[""]).assign(model=lambda d: d.model.astype(str))
    F["model"] = F.model.replace("nan", ""); L["model"] = L.model.replace("nan", "")
    H = L.groupby(["mouse_id", "model"]).ll_test.sum().unstack()
    H0 = L[L.is_day0.astype(str) == "True"].groupby(["mouse_id", "model"]).ll_test.sum().unstack()
    coh = F.drop_duplicates("mouse_id").set_index("mouse_id").cohort
    return F, L, H, H0, coh


def bms(logev, n_iter=500, n_samp=20000, seed=0):
    """random-effects Bayesian model selection (Stephan et al. 2009): logev subjects x models -> Dirichlet alpha, expected frequencies,
    exceedance probabilities"""
    X = np.asarray(logev, float); n, k = X.shape
    a0 = np.ones(k); a = a0.copy()
    for _ in range(n_iter):
        u = X + digamma(a) - digamma(a.sum())
        u -= u.max(1, keepdims=True); g = np.exp(u); g /= g.sum(1, keepdims=True)
        a_new = a0 + g.sum(0)
        if np.max(np.abs(a_new - a)) < 1e-8:
            a = a_new; break
        a = a_new
    r = np.random.default_rng(seed).dirichlet(a, n_samp)
    xp = np.bincount(r.argmax(1), minlength=k) / n_samp
    return a, a / a.sum(), xp, r


# ── 1 models ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_models(F, L, H, H0, coh):
    D = H.sub(H[""], axis=0); D0 = H0.sub(H0[""], axis=0)
    a, freq, xp, _ = bms(H.to_numpy())
    T = pd.DataFrame(dict(model=H.columns, n_comp=[len(m) for m in H.columns], dLL_sum=D.sum().values, dLL_median=D.median().values,
                          dLL_day0_sum=D0.sum().reindex(H.columns).values, n_best=(H.idxmax(1).value_counts().reindex(H.columns).fillna(0)).values,
                          bms_freq=freq, bms_xp=xp))
    for c in ("R+", "R-"):
        T[f"dLL_sum_{c}"] = D[coh.reindex(D.index) == c].sum().values
    T = T.sort_values("dLL_sum", ascending=False); T.to_csv(C.TAB / "012_models.csv", index=False)
    top = T.head(16)
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.2, 1.0, 0.8, 0.9], wspace=0.6))
    ax = axs[0]; y = np.arange(len(top))
    for k_, c in enumerate(("R+", "R-")):
        ax.barh(y + (k_ - 0.5) * 0.38, top[f"dLL_sum_{c}"], height=0.36, color=COH[c], label=c.replace("-", "−"))
    ax.set_yticks(y, [lab(m) for m in top.model], fontsize=5.2); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Σ held-out ΔLL vs core\n(all sessions)"); ax.legend(frameon=False, fontsize=5); ax.set_title("16 best models")
    ax = axs[1]
    ax.barh(y, top.dLL_day0_sum, height=0.6, color="0.4"); ax.set_yticks(y, [lab(m) for m in top.model], fontsize=5.2); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("Σ held-out ΔLL vs core\n(day-0 session)"); ax.set_title("day 0 only")
    ax = axs[2]
    ax.barh(y, top.bms_freq, height=0.6, color="#6a51a3"); ax.set_yticks(y, [lab(m) for m in top.model], fontsize=5.2); ax.invert_yaxis()
    for yy, (f_, x_) in enumerate(zip(top.bms_freq, top.bms_xp)):
        ax.text(f_, yy, f" xp {x_:.2f}", va="center", fontsize=4.6)
    ax.set_xlabel("expected frequency (BMS)"); ax.set_title("random-effects BMS")
    ax = axs[3]
    nb = T.set_index("model").n_comp
    best = H.idxmax(1)
    for k_, c in enumerate(("R+", "R-")):
        b = best[coh.reindex(best.index) == c].map(lambda m: len(m))
        cnt = np.bincount(b, minlength=7) / len(b)
        ax.bar(np.arange(7) + (k_ - 0.5) * 0.38, cnt, width=0.36, color=COH[c])
    ax.set_xlabel("components in the best model"); ax.set_ylabel("fraction of mice"); ax.set_title("best model per mouse\n(held-out LL)")
    fig.suptitle("Model comparison, v3 first iteration: leave-one-session-out held-out log-likelihood (higher = better), relative to the core "
                 "model (learning + decision only)", fontsize=6.6)
    save(fig, "1_models")
    return T


# ── 2 components ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def component_effects(H):
    out = {}
    for comp in "GMNZRP":
        with_ = [m for m in H.columns if comp in m]
        out[comp] = np.mean([H[m] - H[m.replace(comp, "")] for m in with_], axis=0)
    return pd.DataFrame(out, index=H.index)


def fig_components(F, L, H, H0, coh):
    E, E0 = component_effects(H), component_effects(H0)
    _, _, _, r = bms(H.to_numpy())
    rows = []
    for comp in "GMNZRP":
        fam = np.array([comp in m for m in H.columns])
        pfam = r[:, fam].sum(1)
        for c in ("all", "R+", "R-"):
            sel = np.ones(len(E), bool) if c == "all" else (coh.reindex(E.index) == c).to_numpy()
            e, e0 = E[comp][sel], E0[comp].reindex(E.index)[sel].dropna()
            rows.append(dict(component=comp, name=M.CNAME[comp], cohort=c, n=sel.sum(), median_dLL=e.median(), sum_dLL=e.sum(),
                             frac_improved=(e > 0).mean(), p_wilcoxon=stats.wilcoxon(e).pvalue,
                             median_dLL_day0=e0.median(), p_wilcoxon_day0=stats.wilcoxon(e0).pvalue if len(e0) > 5 else np.nan,
                             family_freq=pfam.mean() if c == "all" else np.nan, family_xp=(pfam > 0.5).mean() if c == "all" else np.nan))
    T = pd.DataFrame(rows); T.to_csv(C.TAB / "012_components.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.45))
    for ax, (Ex, ttl) in zip(axs[:2], ((E, "all sessions"), (E0, "day-0 session"))):
        for i, comp in enumerate("GMNZRP"):
            for k_, c in enumerate(("R+", "R-")):
                v = Ex[comp][coh.reindex(Ex.index) == c].dropna()
                x = i + (k_ - 0.5) * 0.36
                ax.scatter(np.full(len(v), x) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), v, s=2.5, color=COH[c], alpha=0.45, lw=0)
                ax.plot([x - 0.13, x + 0.13], [v.median()] * 2, color="k", lw=0.9)
            t_ = T[(T.component == comp) & (T.cohort == "all")].iloc[0]
            p_ = t_.p_wilcoxon if ttl == "all sessions" else t_.p_wilcoxon_day0
            ax.text(i, 1.0, f"p={p_:.1g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=4.6)
        ax.axhline(0, color="0.5", lw=0.5); ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(6), [f"{c}\n{M.CNAME[c]}" for c in "GMNZRP"], fontsize=4.8)
        ax.set_ylabel("held-out ΔLL per mouse\n(with − without, mean of 32 pairs)"); ax.set_title(ttl, pad=10)
    ax = axs[2]; t = T[T.cohort == "all"]
    ax.bar(range(6), t.family_freq, color="#6a51a3")
    for i, (f_, x_) in enumerate(zip(t.family_freq, t.family_xp)):
        ax.text(i, f_ + 0.02, f"{x_:.2f}", ha="center", fontsize=4.6)
    ax.axhline(0.5, color="0.5", lw=0.5, ls="--"); ax.set_ylim(0, 1.1)
    ax.set_xticks(range(6), list("GMNZRP")); ax.set_ylabel("P(component in a mouse's model)\n(family BMS; label: exceedance)")
    ax.set_title("family-level BMS")
    fig.suptitle("Which components does the data need? Per-mouse held-out gain of each component (Wilcoxon across mice), and family-level "
                 "model selection", fontsize=6.5)
    save(fig, "2_components")
    return T


# ── 3 performance ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def heldout_predictions(F, d, models):
    """held-out per-trial predictions are not stored; approximate per-trial probabilities with the full-data fit (for AUC and curves) --
    the LL-based measures use the LOSO scores"""
    out = {}
    for m in models:
        Fm = F[F.model == m].set_index("mouse_id")
        P = []
        for mid, g in d.groupby("mouse_id"):
            if mid not in Fm.index:
                continue
            pv = np.array([Fm.loc[mid, k] for k in M.PARAMS], float)
            P.append(pd.Series(M.trace(pv, M.sequence_arrays(g))[:, 0], index=g.index))
        out[m] = pd.concat(P)
    return out


def fig_performance(F, L, H, H0, coh, best_model, d):
    n = L.groupby(["mouse_id", "model"]).n_test.sum().unstack()
    Lr = H / n                                       # held-out LL per trial
    gm = np.exp(Lr)                                  # geometric-mean probability of the observed choice
    base = d.groupby("mouse_id").lick_flag.mean()
    llb = (base * np.log(base) + (1 - base) * np.log(1 - base)).reindex(gm.index)       # constant P(lick) per mouse (no learning)
    Pr = heldout_predictions(F, d, ["", best_model, FULL])
    from sklearn.metrics import roc_auc_score
    rows = []
    for mid, g in d.groupby("mouse_id"):
        r = dict(mouse_id=mid, cohort=coh.get(mid), ll_const=llb.get(mid))
        for m in ("", best_model, FULL):
            r[f"ll_{lab(m)}"] = Lr.loc[mid, m]; r[f"gm_{lab(m)}"] = gm.loc[mid, m]
            r[f"auc_{lab(m)}"] = roc_auc_score(g.lick_flag, Pr[m].loc[g.index]) if g.lick_flag.nunique() > 1 else np.nan
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(C.TAB / "012_performance.csv", index=False)
    Ld = L.copy(); Ld["llt"] = Ld.ll_test / Ld.n_test
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.4), gridspec_kw=dict(wspace=0.55))
    ms = [("", "core"), (best_model, f"best: {lab(best_model)}"), (FULL, "full")]
    for ax, (col, ttl) in zip(axs[:2], (("gm", "geometric-mean P(observed choice), held-out"), ("auc", "AUC of P(lick), full-data fit"))):
        for k_, c in enumerate(("R+", "R-")):
            t = T[T.cohort == c]
            for j, (m, ml) in enumerate(ms):
                v = t[f"{col}_{lab(m)}"].dropna(); x = j + (k_ - 0.5) * 0.36
                ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-0.07, 0.07, len(v)), v, s=2.5, color=COH[c], alpha=0.45, lw=0)
                ax.plot([x - 0.13, x + 0.13], [v.median()] * 2, color="k", lw=0.9)
        if col == "gm":
            ax.axhline(np.exp(T.ll_const.median()), color="0.5", ls="--", lw=0.6)
            ax.text(2.4, np.exp(T.ll_const.median()), "constant\nP(lick)", fontsize=4.5, va="center")
        ax.set_xticks(range(3), [ml for _, ml in ms], fontsize=5, rotation=20); ax.set_title(ttl, fontsize=5.8)
    ax = axs[2]
    for m, ls, ml in ((("", ":", "core")), (best_model, "-", f"best {lab(best_model)}"), (FULL, "--", "full")):
        s = Ld[Ld.model == m].groupby("day").llt.agg(["mean", "sem"])
        ax.errorbar(s.index, np.exp(s["mean"]), yerr=np.exp(s["mean"]) * s["sem"], ls=ls, marker="o", ms=2, lw=0.8, label=ml, color="k")
    ax.axvline(-0.5, color="r", ls="--", lw=0.5); ax.set_xlabel("training day (0 = whisker onset)"); ax.set_ylabel("held-out geom. mean P")
    ax.legend(frameon=False, fontsize=4.8); ax.set_title("held-out performance by day", fontsize=5.8)
    ax = axs[3]
    gain = (T[f"ll_{lab(best_model)}"] - T["ll_core"])
    for c in ("R+", "R-"):
        v = gain[T.cohort == c]
        ax.hist(v, bins=20, histtype="step", color=COH[c], lw=1, label=f"{c.replace('-', '−')}: median {v.median():.3f}")
    ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("held-out LL per trial, best − core"); ax.legend(frameon=False, fontsize=4.8)
    ax.set_title("gain over the core model", fontsize=5.8)
    fig.suptitle("Fit performance (per mouse). Held-out = leave-one-session-out; AUC from the full-data fit (in-sample).", fontsize=6.5)
    save(fig, "3_performance")
    return T


# ── 4 curves ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_curves(F, d, best_model):
    Pr = heldout_predictions(F, d, ["", best_model])
    x = np.linspace(-1, 1, 2000)
    stage = np.where(d.day < 0, "pretraining", np.where(d.day == 0, "day 0", "whisker days ≥ 1"))
    d = d.assign(stage=stage, pos=d.groupby("session_id").cumcount())
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(4, 3, hspace=0.75, wspace=0.3)
    for i, c in enumerate(("R+", "R-")):
        cv = {"data": [], "core": [], "best": []}
        for mid, g in d[d.reward_group == c].groupby("mouse_id"):
            base = g[["mouse_id", "day", "trial_idx", "trial_type"]].copy()
            for key, val in (("data", g.lick_flag.astype(float)), ("core", Pr[""].loc[g.index]), ("best", Pr[best_model].loc[g.index])):
                cc = du.get_mouse_curves(base.assign(lick_flag=val.to_numpy()), trim_last=0)
                if cc is not None:
                    cv[key].append(np.stack([cc[k] for k, _, _ in TTL]))
        for j, (tk, tl, _) in enumerate(TTL):
            ax = fig.add_subplot(gs[i, j])
            for key, col, lw, ls in (("data", "k", 1.5, "-"), ("core", "0.55", 0.8, ":"), ("best", COH[c], 0.9, "-")):
                Y = np.array(cv[key])[:, j]; m_ = np.nanmean(Y, 0); s_ = np.nanstd(Y, 0) / np.sqrt(np.sum(np.isfinite(Y), 0))
                ax.plot(x, m_, color=col, lw=lw, ls=ls, label={"data": "data", "core": "core model", "best": f"best ({lab(best_model)})"}[key])
                ax.fill_between(x, m_ - s_, m_ + s_, color=col, alpha=0.1, lw=0)
            ax.axvline(0, color="r", ls="--", lw=0.5); ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{c.replace('-', '−')} — {tl}", fontsize=6)
            if j == 0:
                ax.set_ylabel("P(lick)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=4.8)
        for j, st in enumerate(("pretraining", "day 0", "whisker days ≥ 1")):
            ax = fig.add_subplot(gs[2 + i, j]); e = d[(d.reward_group == c) & (d.stage == st)].assign(b=lambda z: (z.pos // 20) * 20 + 10)
            e = e.assign(core=Pr[""].loc[e.index].to_numpy(), best=Pr[best_model].loc[e.index].to_numpy())
            for (tk, tl, col) in TTL:
                f = e[e.trial_type == tk].groupby(["mouse_id", "b"])[["lick_flag", "best", "core"]].mean().groupby("b").mean()
                f = f[f.index < 400]
                ax.plot(f.index, f.lick_flag, color=col, lw=1.4, alpha=0.45)
                ax.plot(f.index, f.best, color=col, lw=0.8)
                ax.plot(f.index, f.core, color=col, lw=0.6, ls=":")
            ax.set_ylim(-0.02, 1.02); ax.set_title(f"{c.replace('-', '−')} — within session, {st}", fontsize=6)
            ax.set_xlabel("trial in session (bins of 20)")
            if j == 0:
                ax.set_ylabel("lick rate")
    fig.suptitle(f"Data vs one-step model P(lick) (full-data fits). Top: time-normalised learning curves (thick black = data, colour = best "
                 f"model {lab(best_model)}, dotted = core). Bottom: within-session curves (thick = data, thin = best, dotted = core; "
                 "blue auditory, orange whisker, grey no stimulus)", fontsize=6.2)
    save(fig, "4_curves")


# ── 5 parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fig_params(F, best_model):
    Fb = F[F.model == FULL].copy()
    P = M.free_params(FULL)
    rows = []
    for q in P:
        a, b = Fb[Fb.cohort == "R+"][q].astype(float), Fb[Fb.cohort == "R-"][q].astype(float)
        rows.append(dict(param=q, median_rp=a.median(), median_rm=b.median(), p_mwu=stats.mannwhitneyu(a, b).pvalue,
                         p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    S = pd.DataFrame(rows)
    S["q_mwu"] = stats.false_discovery_control(S.p_mwu.fillna(1)); S["q_welch"] = stats.false_discovery_control(S.p_welch.fillna(1))
    S.to_csv(C.TAB / "012_params.csv", index=False)
    fig, axs = plt.subplots(3, 6, figsize=(7.4, 4.6))
    for ax, q in zip(axs.flat, P):
        for k_, c in enumerate(("R+", "R-")):
            v = Fb[Fb.cohort == c][q].astype(float)
            ax.scatter(np.full(len(v), k_) + np.random.default_rng(k_).uniform(-0.15, 0.15, len(v)), v, s=2.5, color=COH[c], alpha=0.5, lw=0)
            ax.plot([k_ - 0.22, k_ + 0.22], [v.median()] * 2, color="k", lw=0.9)
        if q in M.LOG:
            ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
        s = S[S.param == q].iloc[0]
        ax.set_title(f"{q}\nMWU {s.p_mwu:.2g}, Welch {s.p_welch:.2g}", fontsize=5)
        ax.set_xticks([0, 1], ["R+", "R−"], fontsize=5); ax.tick_params(labelsize=4.5)
    fig.suptitle("Full-model parameters per mouse (full-data fit), R+ vs R− (BH q in 012_params.csv). Parameters of switched-on components "
                 "only; values at a bound may be unidentified", fontsize=6.2)
    fig.tight_layout()
    save(fig, "5_parameters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    F, L, H, H0, coh = load(a.tag)
    d = pd.read_parquet(C.TAB / "011_sequences.parquet"); d = d[d.mouse_id.isin(H.index)]
    best_model = H.sub(H[""], axis=0).sum().idxmax()
    print("best model (summed held-out LL):", lab(best_model))
    if 1 in a.only:
        fig_models(F, L, H, H0, coh)
    if 2 in a.only:
        fig_components(F, L, H, H0, coh)
    if 3 in a.only:
        fig_performance(F, L, H, H0, coh, best_model, d)
    if 4 in a.only:
        fig_curves(F, d, best_model)
    if 5 in a.only:
        fig_params(F, best_model)
