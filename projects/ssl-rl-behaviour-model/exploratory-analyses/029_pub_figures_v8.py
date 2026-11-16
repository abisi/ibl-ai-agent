"""029 -- publication figures of the final (lean, stimulus-specific learning rates) model (user 2026-10-24: "wrap up, polish reports
and make predictions. Show example mice. Make publication quality summary figures").
Final model: sess-thr-abs-noeng+fa+dis+kap+off+ctx+stim (027 variant lean-abs-stim), mouse-level alpha_A, alpha_W, beta, mu_rho (025S).
Style and helpers from 026 (skills/ssl-figure-style; R+ #00B400, R- #C800C8; mouse = unit; MWU + Welch for R+ vs R-).
Inputs: tables/011_sequences.parquet; fits/023c_fits.csv, fits/027{,b,c,p,q}_fits.csv; tables/027_latents/lean-abs-stim.parquet;
fits/025S_hier_mice.csv, tables/025S_latents, fits/025S_recovery_mice.csv, tables/025S_recovery_summary.csv; tables/028_025S_*.
Output: figures/pub_v8/*.{png,pdf,svg}; tables/029_figure_numbers.csv
Run (haas): python 029_pub_figures_v8.py
"""
from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as C
import rl7_model as M7

F = importlib.import_module("026_pub_figures")
F019 = importlib.import_module("019_fit_v4")
F.OUT = C.FIG / "pub_v8"
F.OUT.mkdir(exist_ok=True)
COH, TC, W, num, band, msem, fmt_p, save, letter_row = F.COH, F.TC, F.W, F.num, F.band, F.msem, F.fmt_p, F.save, F.letter_row
FINAL = "lean-abs-stim"
TAG = "025S"
F.PLAB.update({"alpha_pos": "α_A (auditory learning rate)", "alpha_neg": "α_W (whisker learning rate)", "beta": "β (value gain)",
               "mu_rho": "μ_ρ (log thirst rate)"})
DAYS3 = [0, 1, 2]
NWIN = 20
FIT_TAGS = ("027", "027b", "027c", "027d", "027p", "027q")
VW_TITLE = "Learned whisker value: R+ climbs toward +1 slowly, R− moves toward −1 as slowly"
# fig6: profile (variant prefix, parameter) per cohort, optional R+ alpha+ profile, forced-rate variants shown against the data
PROF = {"R+": ("profS-R+", "alpha_neg"), "R-": ("prof-R--", "alpha_neg")}
PROF_AP = ("prof-R+-", "alpha_pos", "lean-abs")
FORCED = {"R+": "profS-R+-alpha_W-0.2", "R-": "prof-R--alpha_neg-0.2"}
FORCED_LAB = "forced α_W = 0.2"
NUM_FILE, MOUSE_FILE = "029_figure_numbers.csv", "029_mouse_table.csv"
EQ_TITLE = "Go/NoGo Q-learning with thirst-dependent value and go bias, and an engagement state"


def cs_lab(cs):
    return cs.replace("-", "−")


def fits():
    out = [pd.read_csv(C.FITS / "023c_fits.csv").assign(variant=lambda e: "023c-" + e.model.str.split("-").str[2])]
    for t in FIT_TAGS:
        f = C.FITS / f"{t}_fits.csv"
        if f.exists():
            out.append(pd.read_csv(f))
    return pd.concat(out, ignore_index=True).drop_duplicates(["variant", "cohort_set"], keep="last")


# ------------------------------------------------------------------ figure 2: model and model comparison
VLAB = [(FINAL, "final: stimulus-specific α (α_A, α_W)"),
        ("lean-abs", "valence-specific α (α+, α−)"),
        ("lean-inst-stim", "no water-absorption lag"),
        ("lean-abs-gw", "FA value follows whisker value (no step)"),
        ("lean-abs-hom", "reward scaled by thirst (homeostatic)"),
        ("lean-abs-stim-hom", "final + reward scaled by thirst"),
        ("lean-abs-stim-gw-hom", "final + thirst-scaled reward + FA follows whisker"),
        ("min-dis-stim", "minimal: go bias + linear satiety + HMM"),
        ("min-stim", "minimal without engagement HMM"),
        ("023c-abs", "previous fit (local optimum, α− = 0.23)")]

EQ = [r"$\mathbf{Learning}$ (licked stimulus trials):  $\delta = o - V_s$,   $V_s \leftarrow V_s + \alpha_s\,\delta$",
      r"$o = +1$ water, $-1$ no water;   $\alpha_A$ auditory, $\alpha_W$ whisker",
      r"no-stimulus value $V_N$: learned from false alarms ($\alpha_N$), step $d_{ctx}$ at whisker onset",
      r"$\mathbf{Thirst}$:  $T = \max(0,\,1 - \rho_s\,S)$,  $S$ = water absorbed (lag $\tau_a$)",
      r"$\mathbf{Choice}$ (engaged):  $P(\mathrm{lick}) = 0.01 + 0.98\,\sigma(z)$",
      r"$z = \beta\,T^{\kappa}\,V_s - c + s_B\,\mathrm{softplus}(T-\theta) + u_s$",
      r"$\mathbf{Engagement}$: hidden state, disengage $\sigma(h_0 + h_1(1-T))$, return $\sigma(h_b)$; $P(\mathrm{lick}\,|\,\mathrm{dis}) = 0.01$",
      r"per session: $\log\rho_s \sim N(\mu_\rho, \sigma_\rho^2)$, $u_s \sim N(0, \sigma_u^2)$ (integrated out)",
      r"per mouse: $\alpha_A, \alpha_W, \beta, \mu_\rho$ (hierarchical EM, Huys et al. 2011)"]


def fig2(Fa):
    fig = plt.figure(figsize=(W, 3.0))
    axe = fig.add_axes([0.01, 0.05, 0.55, 0.85]); axe.axis("off")
    axe.set_title(EQ_TITLE, loc="left", fontsize=6.5)
    for i, s in enumerate(EQ):
        axe.text(0.02, 0.93 - i * 0.105, s, fontsize=5.6, transform=axe.transAxes, va="top")
    ax = fig.add_axes([0.72, 0.17, 0.26, 0.7])
    ref = Fa[Fa.variant == FINAL].set_index("cohort_set").bic
    rows = [(v, lab) for v, lab in VLAB if v in set(Fa.variant)]
    for j, cs in enumerate(COH):
        d = [Fa[(Fa.variant == v) & (Fa.cohort_set == cs)].bic.iloc[0] - ref[cs] for v, _ in rows]
        ax.barh(np.arange(len(rows)) + (j - 0.5) * 0.38, d, height=0.36, color=COH[cs], lw=0, label=cs_lab(cs))
        for (v, _), x in zip(rows, d):
            num(f"fig2_dbic_{v}_{cs}", float(x))
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows], fontsize=5)
    ax.invert_yaxis(); ax.axvline(0, color="0.4", lw=0.5)
    ax.set_xscale("symlog", linthresh=100); ax.set_xlabel("ΔBIC vs final (cohort-level fit)")
    ax.legend(loc="lower right", fontsize=5); ax.set_title("Model comparison", fontsize=6)
    fig.text(0.005, 0.95, "a", fontsize=9, weight="bold"); fig.text(0.585, 0.95, "b", fontsize=9, weight="bold")
    save(fig, "fig2_model")


# ------------------------------------------------------------------ whisker-specific excess at session start / end
def excess(L, col):
    e = L[L.day.isin(DAYS3) & L.trial_type.isin(["whisker", "no_stim"])].copy()
    g = e.groupby(["session_id", "trial_type"])
    e["k"] = g.cumcount(); e["kend"] = g.cumcount(ascending=False)
    out = []
    for w, sel in (("start", e.k < NWIN), ("end", e.kend < NWIN)):
        m = e[sel].groupby(["reward_group", "day", "mouse_id", "trial_type"])[col].mean().unstack()
        out.append((m.whisker - m.no_stim).rename("x").reset_index().assign(win=w))
    return pd.concat(out, ignore_index=True)


def excess_panel(ax, data, models, cs):
    """data/model: whisker - FA lick rate at session start and end per day; x = day + 0 (start) / + 0.6 (end)"""
    off = {"start": 0.0, "end": 0.6}
    for d_ in DAYS3:
        e = data[(data.reward_group == cs) & (data.day == d_)]
        g = e.groupby("win").x.agg(["mean", "sem"]).reindex(["start", "end"])
        ax.errorbar([d_ + off[w] for w in g.index], g["mean"], g["sem"], fmt="o-", ms=3, lw=1.0, color=COH[cs], capsize=0,
                    label="data" if d_ == 0 else None)
        for lab, (Mx, ls, col) in models.items():
            m = Mx[(Mx.reward_group == cs) & (Mx.day == d_)].groupby("win").x.mean().reindex(["start", "end"])
            ax.plot([d_ + off[w] for w in m.index], m.values, ls=ls, color=col, lw=0.9, label=lab if d_ == 0 else None)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xticks([d + 0.3 for d in DAYS3], [f"day {d}" for d in DAYS3])
    ax.set_ylabel("whisker − false-alarm\nlick rate")
    ax.set_title(f"{cs_lab(cs)}: whisker-specific licking\n(first / last {NWIN} trials of each type)", fontsize=6, color=COH[cs])


# ------------------------------------------------------------------ figure 5: learning speed
def fig5(Lm, Mm, S, Rm):
    data = excess(Lm, "lick_flag"); model = excess(Lm, "p_lick")
    fig = plt.figure(figsize=(W, 4.3))
    gs = fig.add_gridspec(2, 4, left=0.08, right=0.98, top=0.9, bottom=0.1, hspace=0.75, wspace=0.6)
    axa = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, cs in zip(axa, COH):
        excess_panel(ax, data, {"model": (model, "--", "k")}, cs)
        for d_ in DAYS3:
            for w in ("start", "end"):
                e = data[(data.reward_group == cs) & (data.day == d_) & (data.win == w)].x
                num(f"fig5_excess_{cs}_day{d_}_{w}", float(e.mean()))
    axa[0].legend(fontsize=5, loc="upper left")
    e0 = Lm[(Lm.day == 0) & (Lm.trial_type == "whisker")].copy()   # first whisker trials of day 0, blocks of 10 (remaining misfit)
    e0["b"] = e0.groupby("mouse_id").cumcount() // 10
    for cs in COH:
        g = e0[(e0.reward_group == cs) & (e0.b < 6)].groupby(["b", "mouse_id"])[["lick_flag", "p_lick"]].mean().groupby("b").mean()
        for b_, r in g.iterrows():
            num(f"fig5_early_{cs}_data_b{b_}", float(r.lick_flag)); num(f"fig5_early_{cs}_model_b{b_}", float(r.p_lick))
    ax = axc = fig.add_subplot(gs[0, 2:4])
    w = Lm[(Lm.day >= 0) & (Lm.trial_type == "whisker")].copy()
    w["i"] = w.groupby("mouse_id").cumcount()
    w = w[w.i < 700]
    w["b"] = (w.i // 25) * 25 + 12
    for cs in COH:
        T = msem(w[w.reward_group == cs], ["b"], "V_W")
        T = T[T.n >= 20]                               # bins with at least 20 mice (fewer mice reach many whisker trials)
        band(ax, T.b, T.m, T.s, COH[cs], lw=1.3, label=cs_lab(cs))
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("whisker trials since whisker onset (bins with ≥ 20 mice)"); ax.set_ylabel("whisker value V_W\n(mean ± s.e.m. over mice)")
    ax.set_title(VW_TITLE, fontsize=6); ax.legend(fontsize=5)
    axb = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for ax, q, lab, lg in [(axb[0], "alpha_W", "α_W whisker learning rate", True), (axb[1], "alpha_A", "α_A auditory learning rate", True),
                           (axb[2], "tau_W", "τ_W (licked whisker trials\nto move 63% of the way)", True)]:
        col = {"alpha_W": "p_alpha_neg", "alpha_A": "p_alpha_pos"}.get(q, q)
        for j, cs in enumerate(COH):
            y = Mm[Mm.reward_group == cs][col]
            ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(y)), y, s=4, color=COH[cs], lw=0, alpha=0.6)
            ax.errorbar(j + 0.32, np.exp(np.log(y).mean()), fmt="o", ms=3, color=COH[cs])
        a_, b_ = Mm[Mm.reward_group == "R+"][col], Mm[Mm.reward_group == "R-"][col]
        pm, pw = stats.mannwhitneyu(a_, b_).pvalue, stats.ttest_ind(np.log(a_), np.log(b_), equal_var=False).pvalue
        num(f"fig5_{q}_median_R+", float(a_.median())); num(f"fig5_{q}_median_R-", float(b_.median()))
        num(f"fig5_{q}_p_mwu", pm); num(f"fig5_{q}_p_welch", pw)
        ax.set_yscale("log"); ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_xlim(-0.5, 1.6)
        ax.set_title(f"{lab}\nMWU {fmt_p(pm)}, Welch {fmt_p(pw)}", fontsize=5.5)
    axb[0].set_ylabel("per-mouse estimate")
    ax = axb[3]
    for j, cs in enumerate(COH):
        for k, d_ in enumerate(DAYS3):
            y = Mm[Mm.reward_group == cs][f"V_W_end_day{d_}"].dropna()
            ax.errorbar(d_ + (j - 0.5) * 0.2, y.mean(), y.sem(), fmt="o", ms=3, color=COH[cs], label=cs_lab(cs) if k == 0 else None)
    ax.axhline(0, color="0.7", lw=0.5); ax.set_xticks(DAYS3); ax.set_xlabel("end of day"); ax.set_ylabel("V_W (mean ± s.e.m.)")
    ax.set_title("Whisker value reached\nat the end of each day", fontsize=5.5); ax.legend(fontsize=5)
    letter_row(fig, axa + [axc], "abc"); letter_row(fig, axb, "defg")
    save(fig, "fig5_learning_speed")


# ------------------------------------------------------------------ figure 6: learning-rate profiles and the fast-learning prediction
def profile(Fa, prefix, cs, q):
    P = Fa[Fa.variant.str.startswith(prefix) & (Fa.cohort_set == cs)].copy()
    P["a"] = P[f"p_{q}"]
    return P.sort_values("a")


def forced_latents(Fa, D, rows):
    out = []
    for cs, var in rows.items():
        r = Fa[(Fa.variant == var) & (Fa.cohort_set == cs)].iloc[0]
        d = D[D.reward_group == cs].reset_index(drop=True)
        p = {k: r[f"p_{k}"] if pd.notna(r.get(f"p_{k}", np.nan)) else M7.FIXED[k] for k in M7.P_ALL}
        L = M7.latents(r.model, p, M7.arrays(d))
        out.append(d[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group"]].assign(p_lick=L["p_lick"]))
    return pd.concat(out, ignore_index=True)


def fig6(Fa, D, Lc):
    prof = {cs: profile(Fa, pre, cs, q) for cs, (pre, q) in PROF.items()}
    prof_ap = profile(Fa, PROF_AP[0], "R+", PROF_AP[1]) if PROF_AP else pd.DataFrame()
    fig = plt.figure(figsize=(W, 2.6))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.8, bottom=0.2, wspace=0.5)
    ax = fig.add_subplot(gs[0])
    best = {}
    for cs, P in prof.items():
        if P.empty:
            continue
        ref = min(P.nll.min(), Fa[(Fa.variant == FINAL) & (Fa.cohort_set == cs)].nll.min())
        ax.plot(P.a, P.nll - ref, "o-", ms=2.5, lw=1, color=COH[cs], label=f"{cs_lab(cs)}: α_W")
        best[cs] = ref
        for _, r in P.iterrows():
            num(f"fig6_profile_{cs}_{r.a:g}", float(r.nll - ref))
    if not prof_ap.empty:
        ref = min(prof_ap.nll.min(), Fa[(Fa.variant == PROF_AP[2]) & (Fa.cohort_set == "R+")].nll.min())
        ax.plot(prof_ap.a, prof_ap.nll - ref, "o:", ms=2, lw=0.8, color=COH["R+"], alpha=0.6, label="R+: shared α+ (valence model)")
    ax.axhline(1.92, color="0.5", lw=0.5, ls="--")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("whisker learning rate (fixed)"); ax.set_ylabel("Δ negative log-likelihood\n(all else refitted)")
    ax.set_title("Likelihood profile of the\nwhisker learning rate (cohort level)", fontsize=6); ax.legend(fontsize=4.8)
    rows = {}
    for cs, pre in FORCED.items():
        if pre in set(Fa.variant):
            rows[cs] = pre
    axs = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    if rows:
        Lf = forced_latents(Fa, D, rows)
        data, fin, fast = excess(Lc, "lick_flag"), excess(Lc, "p_lick"), excess(Lf, "p_lick")
        for ax2, cs in zip(axs, COH):
            flab = FORCED_LAB[cs] if isinstance(FORCED_LAB, dict) else FORCED_LAB
            excess_panel(ax2, data, {"final model": (fin, "--", "k"), flab: (fast, ":", "#d62728")}, cs)
            for d_ in DAYS3:
                for w in ("start", "end"):
                    num(f"fig6_fast_excess_{cs}_day{d_}_{w}", float(fast[(fast.reward_group == cs) & (fast.day == d_) & (fast.win == w)].x.mean()))
                    num(f"fig6_final_excess_{cs}_day{d_}_{w}", float(fin[(fin.reward_group == cs) & (fin.day == d_) & (fin.win == w)].x.mean()))
                    num(f"fig6_data_excess_{cs}_day{d_}_{w}", float(data[(data.reward_group == cs) & (data.day == d_) & (data.win == w)].x.mean()))
        axs[0].legend(fontsize=4.8, loc="upper left")
    letter_row(fig, [ax] + axs, "abc")
    save(fig, "fig6_profiles")


# ------------------------------------------------------------------ figure 7: per-mouse parameters and model-free correlates
def corr_panel(ax, x, y, color, xl, yl, key):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    ax.scatter(x, y, s=5, color=color, lw=0, alpha=0.7)
    r, p = stats.pearsonr(x, y); rs, ps = stats.spearmanr(x, y)
    b, a = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    n = len(x); res = y - (a + b * x); s = np.sqrt(np.sum(res ** 2) / (n - 2))
    se = s * np.sqrt(1 / n + (xx - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t = stats.t.ppf(0.975, n - 2)
    ax.fill_between(xx, a + b * xx - t * se, a + b * xx + t * se, color=color, alpha=0.18, lw=0)
    ax.plot(xx, a + b * xx, color=color, lw=1, ls="-" if p < 0.05 else "--")
    ax.set_xlabel(xl, fontsize=5.5); ax.set_ylabel(yl, fontsize=5.5)
    ax.set_title(f"r = {r:.2f} ({fmt_p(p)})\nSpearman ρ = {rs:.2f} ({fmt_p(ps)}), n = {n}", fontsize=5)
    num(f"fig7_{key}_r", r); num(f"fig7_{key}_p", p); num(f"fig7_{key}_spearman", rs); num(f"fig7_{key}_spearman_p", ps)


def fig7(Mm, Lm, d):
    ex = excess(Lm, "lick_flag")
    st = ex[ex.win == "start"].pivot_table(index="mouse_id", columns="day", values="x")
    Mm = Mm.set_index("mouse_id")
    Mm["mf_rplus"] = st.get(2) - st.get(0)       # R+: growth of whisker-specific licking, day-0 start -> day-2 start
    Mm["mf_rminus"] = st.get(0) - st.get(1)      # R-: loss of whisker-specific licking, day-0 start -> day-1 start
    sess = d[d.day.isin(DAYS3)].groupby(["mouse_id", "session_id"]).water.sum().groupby("mouse_id").median()
    Mm["rewards_per_session"] = sess
    fig = plt.figure(figsize=(W, 4.2))
    gs = fig.add_gridspec(2, 4, left=0.08, right=0.98, top=0.9, bottom=0.1, hspace=0.8, wspace=0.6)
    axt = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (q, lab) in zip(axt, [("p_beta", "β (value gain)"), ("p_mu_rho", "μ_ρ (log thirst rate)"),
                                  ("rewards_to_half_thirst", "rewards to halve thirst"), ("ysd_alpha_neg", "posterior s.d. of α_W\n(logit scale)")]):
        for j, cs in enumerate(COH):
            y = Mm[Mm.reward_group == cs][q]
            ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(y)), y, s=4, color=COH[cs], lw=0, alpha=0.6)
            ax.errorbar(j + 0.32, y.mean(), y.sem(), fmt="o", ms=3, color=COH[cs])
        a_, b_ = Mm[Mm.reward_group == "R+"][q], Mm[Mm.reward_group == "R-"][q]
        lg = q in ("p_beta", "rewards_to_half_thirst")
        pm = stats.mannwhitneyu(a_, b_).pvalue
        pw = stats.ttest_ind(np.log(a_) if lg else a_, np.log(b_) if lg else b_, equal_var=False).pvalue
        num(f"fig7_{q}_p_mwu", pm); num(f"fig7_{q}_p_welch", pw)
        num(f"fig7_{q}_median_R+", float(a_.median())); num(f"fig7_{q}_median_R-", float(b_.median()))
        if lg:
            ax.set_yscale("log")
        ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_xlim(-0.5, 1.6)
        ax.set_title(f"{lab}\nMWU {fmt_p(pm)}, Welch {fmt_p(pw)}", fontsize=5.5)
    axb = [fig.add_subplot(gs[1, i]) for i in range(4)]
    rp, rm = Mm[Mm.reward_group == "R+"], Mm[Mm.reward_group == "R-"]
    corr_panel(axb[0], np.log10(rp.p_alpha_neg), rp.mf_rplus, COH["R+"], "log10 α_W", "Δ whisker − FA, day 0 → day 2 start", "rplus_alphaW_mf")
    corr_panel(axb[1], np.log10(rm.p_alpha_neg), rm.mf_rminus, COH["R-"], "log10 α_W", "loss of whisker − FA,\nday 0 → day 1 start", "rminus_alphaW_mf")
    corr_panel(axb[2], np.log10(Mm.rewards_to_half_thirst), Mm.rewards_per_session, "0.3", "log10 rewards to halve thirst",
               "median rewards per session", "thirst_rewards")
    corr_panel(axb[3], np.log10(Mm.p_alpha_pos), np.log10(Mm.p_alpha_neg), "0.3", "log10 α_A", "log10 α_W", "alphaA_alphaW")
    for ax in axb[2:]:
        for cs in COH:
            e = Mm[Mm.reward_group == cs]
            x = np.log10(e.rewards_to_half_thirst) if ax is axb[2] else np.log10(e.p_alpha_pos)
            y = e.rewards_per_session if ax is axb[2] else np.log10(e.p_alpha_neg)
            ax.scatter(x, y, s=5, color=COH[cs], lw=0, alpha=0.8)
    letter_row(fig, axt, "abcd"); letter_row(fig, axb, "efgh")
    save(fig, "fig7_mouse_parameters")
    return Mm.reset_index()


def main():
    Fa = fits()
    D = F019.load()
    d = F.load_data()
    F.fig1(d)
    fig2(Fa)
    model = Fa[Fa.variant == FINAL].model.iloc[0]
    num("final_model", model)
    Lc = F.lat(C.TAB / "027_latents" / f"{FINAL}.parquet")
    Lm = F.lat(C.TAB / f"{TAG}_latents" / f"{model}.parquet")
    lats = [("cohort-level fit", Lc), ("mouse-level fit", Lm)]
    F.fig3(lats)
    F.fig3b(lats)
    ex = F.fig4(Lm, "mouse-level final model")
    F.fig4b(Lm, ex, "mouse-level final model")
    num("example_mice", "; ".join(f"{k}: {v}" for k, v in ex.items()))
    Mm = pd.read_csv(C.TAB / f"028_{TAG}_mouse_metrics.csv")
    fig5(Lm, Mm, pd.read_csv(C.TAB / f"028_{TAG}_stats.csv"), pd.read_csv(C.FITS / f"{TAG}_hier_mice.csv"))
    fig6(Fa, D, Lc)
    M2 = fig7(Mm, Lm, d)
    M2.to_csv(C.TAB / MOUSE_FILE, index=False)
    if (C.TAB / f"{TAG}_recovery_summary.csv").exists():
        F.figS2(pd.read_csv(C.TAB / f"{TAG}_recovery_summary.csv"), pd.read_csv(C.FITS / f"{TAG}_recovery_mice.csv"))
    pd.DataFrame(F.NUM).to_csv(C.TAB / NUM_FILE, index=False)


if __name__ == "__main__":
    main()
