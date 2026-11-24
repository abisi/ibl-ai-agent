"""009 -- Does the learner definition (border / extreme mice) change the neural result of Figure 4 K? (user 2026-11-23: "AB086
could actually be moderate and AB117 too. It's hard to decide as I am not sure it greatly affects neural results ... let's do
analysis and see whether these border mice / extreme mice influence the results ... correlate the effect with performance";
"redo this analysis in this project, and add more figures, more controls").

Neural effect per day-0 session (whole brain, tracked stable units; values of Figure 4 K = ssl-hitmiss-axis-passive 165
SSL_165_ANGLE, read from that project's tables, nothing recomputed):
  q_W, q_A        change of the passive whisker / auditory projection on the session's hit - miss axis (5-35 ms; post - pre, per
                  neuron), real axis - mean of the linear-shift axes (135 shift_excess_dWP / dAP, area 'All units'; session set:
                  whisker shift null defined, as 165)
  q_th_W, q_th_A  change of the angle between the passive response and the axis (deg), real - shifted (169, as 165 load_angle)
All day-0 sessions with these values (one per mouse); Figure 4 keeps sheet learners only (27 R+ / 24 R-).
Behaviour per mouse (this project): sheet category; known bad / flexible (005); R+ S_mean_20 (006, best 20-trial window of the
whisker curve, last 15 % cut, threshold of 006); R- end_w (007, whisker licks in the last 20 whisker trials of the cut session,
rule <= 0.25); day-0 hit_w, d', peak d'; day +1 hit_w, d'.
Border = R+ S_mean_20 in [0.25, 0.55] (the 008 review) or R- end_w > 0.25.

Figures (figures/009_*.png|pdf):
  1 effect_vs_behaviour   effect (whisker; auditory = within-session control) vs the cohort's sustained metric, all sessions;
                          Spearman for all sessions and sheet learners only
  2 correlation_matrix    Spearman rho of every effect with every behaviour metric, per cohort (all / learners)
  3 populations           Figure 4 K statistics for six learner definitions: per modality R+ vs R- (Mann-Whitney and Welch),
                          cohort x modality interaction (W - A difference between cohorts, cohort labels permuted across mice,
                          10,000); z-scored (Figure 4) and raw-rate axes (135 / 169 _raw)
  4 influence             leave one mouse out from all day-0 mice: change of the R+ - R- difference and its p; bootstrap over
                          mice of the R+ - R- difference (sheet learners vs all mice)
  5 controls              confounds: Spearman of the effect with n_units, passive trial counts, pre response size (rho_pre) and
                          n whisker trials, per cohort; OLS effect ~ cohort + log n_units + rho_pre + n_whisker (all mice and
                          learners); does the cohort effect survive adding behaviour (effect ~ cohort + day-0 hit_w)?
Tables: tables/009_sessions.csv, 009_correlations.csv, 009_populations.csv, 009_leave_one_out.csv, 009_bootstrap.csv, 009_ols.csv.
Exploratory, uncorrected (MW + Welch; mouse = session = unit).
Run (haas, repo root): ~/code/ibl-ai-agent/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/009_neural_sensitivity_fig4K.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root() / "combined_results_ks4"
HM = ROOT / "ssl-hitmiss-axis-passive" / "tables"
RES = ROOT / "ssl-learner-classification"
LC = RES / "tables"
FIG = RES / "figures"
COL = {"R+": "#00B400", "R-": "#C800C8"}
GCOL = {"good": "#00B400", "moderate": "#7fd17f", "flexible": "tab:orange", "known bad": "k", "bad": "0.45"}
BORDER = (0.25, 0.55)
N_PERM, N_BOOT = 10000, 5000
EFF = {"q_W": "Δ projection, whisker", "q_A": "Δ projection, auditory", "q_th_W": "Δ angle, whisker (°)",
       "q_th_A": "Δ angle, auditory (°)"}
PERF = {"R+": ["S_mean_20", "hit_w", "dprime", "peak_dprime", "hit_w_day1", "dprime_day1"],
        "R-": ["end_w", "early_w", "hit_w", "dprime", "hit_w_day1", "dprime_day1"]}
CONF = ["n_units", "n_pre", "n_post", "rho_pre", "n_whisker"]
rng = np.random.default_rng(0)


def load_effects(rs):
    A = pd.read_parquet(HM / f"135_alignment_epochs_tracked_harm{rs}.parquet")
    A = A[A.skipped_reason.isna() & (A.area == "All units")].dropna(subset=["shift_null_post_W_mean"])
    A = A.assign(q_W=A.shift_excess_dWP, q_A=A.shift_excess_dAP)[["session_id", "reward_group", "n_units", "q_W", "q_A"]]
    P = pd.read_parquet(HM / f"169_along_orthogonal{rs}.parquet")
    th = lambda c: np.degrees(np.arccos(np.clip(c, -1, 1)))  # noqa: E731
    P["q_th"] = (th(P.cos_post) - th(P.cos_pre)) - (th(P.cos_post_null) - th(P.cos_pre_null))
    T = P.pivot_table(index="session_id", columns="stim", values=["q_th", "rho_pre", "n_pre", "n_post"])
    T.columns = [f"{a}_{b}" for a, b in T.columns]
    T = T.rename(columns={"rho_pre_W": "rho_pre", "n_pre_W": "n_pre", "n_post_W": "n_post"})
    return A.merge(T.reset_index()[["session_id", "q_th_W", "q_th_A", "rho_pre", "n_pre", "n_post"]], on="session_id", how="left")


def load(rs=""):
    d = load_effects(rs)
    d["mouse_id"] = d.session_id.str.split("_").str[0]
    pm = pd.read_csv(LC / "001_per_mouse.csv")[["mouse_id", "cohort", "learning_category", "learner", "fixed_bad", "flexible",
                                                 "n_whisker", "hit_w", "dprime", "peak_dprime", "hit_w_day1", "dprime_day1"]]
    s6 = pd.read_csv(LC / "006_sustained_per_mouse_keep85.csv")[["mouse_id", "S_mean_20"]]
    s7 = pd.read_csv(LC / "007_rminus_metrics_keep85.csv")[["mouse_id", "end_w", "early_w"]]
    d = d.merge(pm, on="mouse_id", how="left").merge(s6, on="mouse_id", how="left").merge(s7, on="mouse_id", how="left")
    assert (d.cohort == d.reward_group).all() and not d.mouse_id.duplicated().any()
    d["group"] = np.where(d.fixed_bad, "known bad", np.where(d.flexible, "flexible", d.learning_category))
    th6 = float(pd.read_csv(LC / "006_sustained_summary_keep85.csv").set_index("metric").loc["S_mean_20", "threshold"])
    rp = (d.cohort == "R+").to_numpy()
    d["border"] = np.where(rp, d.S_mean_20.between(*BORDER), d.end_w > 0.25)
    sheet = (d.learning_category != "bad").to_numpy()
    pops = {"sheet (Fig. 4)": sheet,
            "sheet + AB117": sheet | (d.mouse_id == "AB117").to_numpy(),
            "all day-0 mice": np.ones(len(d), bool),
            "rule 006 / 007": np.where(rp, d.S_mean_20 > th6, d.end_w <= 0.25),
            "consensus (001)": d.learner.astype(bool).to_numpy(),
            "clear learners only": sheet & ~d.border.to_numpy()}
    return d, pops, th6


def interaction(g, a, b):
    x = (g[a] - g[b]).to_numpy(float)
    c = (g.cohort == "R+").to_numpy()
    ok = np.isfinite(x)
    x, c = x[ok], c[ok]
    obs = x[c].mean() - x[~c].mean()
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        p = rng.permutation(c)
        null[k] = x[p].mean() - x[~p].mean()
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))


def two_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def spear(g, x, y):
    ok = g[[x, y]].dropna()
    return (*stats.spearmanr(ok[x], ok[y]), len(ok)) if len(ok) > 4 else (np.nan, np.nan, len(ok))


def population_stats(d, pops, variant):
    rows = []
    for name, keep in pops.items():
        g = d[keep]
        r = dict(variant=variant, population=name, n_Rp=int((g.cohort == "R+").sum()), n_Rm=int((g.cohort == "R-").sum()))
        for e in EFF:
            a, b = g[g.cohort == "R+"][e], g[g.cohort == "R-"][e]
            r[f"{e}_Rp"], r[f"{e}_Rm"], r[f"{e}_Rp_sem"], r[f"{e}_Rm_sem"] = a.mean(), b.mean(), a.sem(), b.sem()
            r[f"{e}_p_mw"], r[f"{e}_p_welch"] = two_tests(a, b)
        for a_, b_, k in (("q_W", "q_A", "proj"), ("q_th_W", "q_th_A", "angle")):
            r[f"int_{k}"], r[f"int_{k}_p"] = interaction(g, a_, b_)
        r["excluded"] = ",".join(sorted(d.mouse_id[~keep]))
        rows.append(r)
    return pd.DataFrame(rows)


def label_points(ax, g, x, y):
    for _, r_ in g[g.border | (g.learning_category == "bad")].iterrows():
        if np.isfinite(r_[x]) and np.isfinite(r_[y]):
            ax.annotate(r_.mouse_id, (r_[x], r_[y]), fontsize=3.6, xytext=(2, 1), textcoords="offset points")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"009_{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    d, pops, th6 = load("")
    draw, pops_raw, _ = load("_raw")
    d.to_csv(LC / "009_sessions.csv", index=False)

    # correlations
    rows = []
    for coh, ms in PERF.items():
        for scope in ("all", "learners"):
            g = d[(d.cohort == coh) & ((d.learning_category != "bad") | (scope == "all"))]
            for e in EFF:
                for m in ms + CONF:
                    r, p, n = spear(g, e, m)
                    rows.append(dict(cohort=coh, scope=scope, effect=e, metric=m, kind="confound" if m in CONF else "behaviour",
                                     n=n, rho=r, p=p))
    C = pd.DataFrame(rows)
    C.to_csv(LC / "009_correlations.csv", index=False)

    # populations (z-scored axes = Figure 4, raw-rate axes)
    S = pd.concat([population_stats(d, pops, "z-scored (Fig. 4)"), population_stats(draw, pops_raw, "raw rates")])
    S.to_csv(LC / "009_populations.csv", index=False)

    # leave one mouse out, bootstrap
    rows = []
    diff = lambda g, e: g[g.cohort == "R+"][e].mean() - g[g.cohort == "R-"][e].mean()  # noqa: E731
    base = {e: diff(d, e) for e in EFF}
    for i, r_ in d.iterrows():
        g = d.drop(index=i)
        row = dict(mouse_id=r_.mouse_id, cohort=r_.cohort, group=r_.group, border=bool(r_.border))
        for e in ("q_W", "q_th_W"):
            row[f"d_{e}"] = diff(g, e) - base[e]
            row[f"p_{e}"] = two_tests(g[g.cohort == "R+"][e], g[g.cohort == "R-"][e])[0]
        rows.append(row)
    L = pd.DataFrame(rows)
    L.to_csv(LC / "009_leave_one_out.csv", index=False)
    rows = []
    for name in ("sheet (Fig. 4)", "all day-0 mice", "clear learners only"):
        g = d[pops[name]]
        for e in ("q_W", "q_th_W"):
            a, b = g[g.cohort == "R+"][e].dropna().to_numpy(), g[g.cohort == "R-"][e].dropna().to_numpy()
            bs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)])
            rows.append(dict(population=name, effect=e, diff=a.mean() - b.mean(), ci_lo=np.percentile(bs, 2.5),
                             ci_hi=np.percentile(bs, 97.5), frac_same_sign=float(np.mean(np.sign(bs) == np.sign(a.mean() - b.mean()))),
                             boot=bs))
    B = pd.DataFrame(rows)
    B.drop(columns="boot").to_csv(LC / "009_bootstrap.csv", index=False)

    # OLS with confounds
    rows = []
    for scope in ("all", "learners"):
        g = d if scope == "all" else d[d.learning_category != "bad"]
        g = g.assign(log_units=np.log(g.n_units), rp=(g.cohort == "R+").astype(int))
        for e in ("q_W", "q_th_W", "q_A", "q_th_A"):
            for form in ("rp", "rp + log_units + rho_pre + n_whisker", "rp + hit_w", "rp * hit_w"):
                m = smf.ols(f"{e} ~ {form}", g.dropna(subset=[e])).fit()
                rows.append(dict(scope=scope, effect=e, model=form, n=int(m.nobs), b_cohort=m.params["rp"], p_cohort=m.pvalues["rp"],
                                 p_hit_w=m.pvalues.get("hit_w", np.nan), p_inter=m.pvalues.get("rp:hit_w", np.nan), r2=m.rsquared))
    O = pd.DataFrame(rows)
    O.to_csv(LC / "009_ols.csv", index=False)

    pd.set_option("display.width", 260)
    print(d[["mouse_id", "cohort", "group", "border", "S_mean_20", "end_w", "q_W", "q_th_W", "n_units"]]
          .sort_values(["cohort", "S_mean_20"]).round(3).to_string(index=False))
    print(S[["variant", "population", "n_Rp", "n_Rm", "q_W_Rp", "q_W_Rm", "q_W_p_mw", "q_W_p_welch", "int_proj_p", "q_th_W_Rp",
             "q_th_W_Rm", "q_th_W_p_mw", "q_th_W_p_welch", "int_angle_p"]].round(4).to_string(index=False))
    print(C[C.p < 0.1].round(3).to_string(index=False))
    print(L.sort_values("p_q_W", ascending=False).head(5).round(4).to_string(index=False))
    print(L.sort_values("p_q_th_W", ascending=False).head(5).round(4).to_string(index=False))
    print(B.drop(columns="boot").round(4).to_string(index=False))
    print(O.round(4).to_string(index=False))

    # ---- figure 1: effect vs behaviour
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.9), gridspec_kw=dict(hspace=0.8, wspace=0.55))
    for j, (coh, m, mlab, thr) in enumerate((("R+", "S_mean_20", "Sustained whisker P(lick)\n(best 20 trials)", th6),
                                             ("R-", "end_w", "Whisker licks,\nlast 20 trials (cut)", 0.25))):
        g = d[d.cohort == coh]
        for k, e in enumerate(EFF):
            ax = axs[j, k]
            for grp, gg in g.groupby("group"):
                ax.scatter(gg[m], gg[e], s=8, color=GCOL.get(grp, "0.5") if coh == "R+" else
                           ("k" if grp == "bad" else COL["R-"]), lw=0, label=grp)
            label_points(ax, g, m, e)
            ax.axvline(thr, color="0.5", lw=0.5, ls="--")
            ax.axhline(0, color="0.8", lw=0.5)
            ra, pa, na = spear(g, m, e)
            rl, pl, nl = spear(g[g.learning_category != "bad"], m, e)
            ax.set_title(f"{coh.replace('-', '−')} {EFF[e]}\nall ρ {ra:+.2f} p {pa:.2f} (n {na}); learners ρ {rl:+.2f} p {pl:.2f}",
                         fontsize=4.6, color=COL[coh])
            ax.set_xlabel(mlab, fontsize=4.8)
            ax.tick_params(labelsize=4.3)
        axs[j, 0].legend(fontsize=3.6, frameon=False, loc="lower left", handletextpad=0.1)
    fig.suptitle("Figure 4 K effect (post − pre on the hit − miss axis, real − shifted; whole brain, day 0) vs the cohort's "
                 "learner metric, all day-0 sessions. Dashed = learner threshold (006 / 007); labelled = border or sheet bad. "
                 "Auditory columns = within-session control. Exploratory.", fontsize=5.6, y=1.03, wrap=True)
    save(fig, "effect_vs_behaviour")

    # ---- figure 2: correlation matrix
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.9))
    for ax, (coh, scope) in zip(axs, [(c, s) for c in ("R+", "R-") for s in ("all", "learners")]):
        ms = PERF[coh] + CONF
        M = C[(C.cohort == coh) & (C.scope == scope)].pivot(index="metric", columns="effect", values="rho").reindex(ms)[list(EFF)]
        Pm = C[(C.cohort == coh) & (C.scope == scope)].pivot(index="metric", columns="effect", values="p").reindex(ms)[list(EFF)]
        ax.imshow(M.to_numpy(float), cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
        for i in range(len(ms)):
            for j in range(len(EFF)):
                v, p = M.iat[i, j], Pm.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}{'*' if p < 0.05 else ''}", ha="center", va="center", fontsize=3.6)
        ax.set_xticks(range(len(EFF)), ["proj W", "proj A", "angle W", "angle A"], fontsize=4.3, rotation=45, ha="right")
        ax.set_yticks(range(len(ms)), ms, fontsize=4.3)
        ax.axhline(len(PERF[coh]) - 0.5, color="k", lw=0.6)
        n = int(C[(C.cohort == coh) & (C.scope == scope)].n.max())
        ax.set_title(f"{coh.replace('-', '−')}, {scope} (n ≤ {n})", fontsize=5.5, color=COL[coh])
    fig.suptitle("Spearman ρ of the K effects with behaviour (above line) and confounds (below; n units, passive trials, pre "
                 "response size, n whisker trials). * p < 0.05 uncorrected. Exploratory.", fontsize=5.8, y=1.04)
    save(fig, "correlation_matrix")

    # ---- figure 3: populations
    names = list(pops)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 4.6), gridspec_kw=dict(hspace=1.0, wspace=0.3))
    for r, variant in enumerate(("z-scored (Fig. 4)", "raw rates")):
        Sv = S[S.variant == variant].reset_index(drop=True)
        for c, (e, ea, ik) in enumerate((("q_W", "q_A", "proj"), ("q_th_W", "q_th_A", "angle"))):
            ax = axs[r, c]
            for i, nm in enumerate(names):
                rr = Sv[Sv.population == nm].iloc[0]
                for o, coh, t in ((-0.15, "R+", "Rp"), (0.15, "R-", "Rm")):
                    ax.errorbar(i + o, rr[f"{e}_{t}"], rr[f"{e}_{t}_sem"], fmt="o", ms=3, color=COL[coh], lw=0.9, capsize=0)
                    ax.errorbar(i + o, rr[f"{ea}_{t}"], rr[f"{ea}_{t}_sem"], fmt="s", ms=2.2, mfc="white", color=COL[coh],
                                lw=0.5, capsize=0, alpha=0.7)
                ax.text(i, 1.01, f"W: MW {rr[f'{e}_p_mw']:.3f}\nWelch {rr[f'{e}_p_welch']:.3f}\nW×A int. {rr[f'int_{ik}_p']:.3f}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=3.6)
            ax.axhline(0, color="0.8", lw=0.5)
            ax.set_xticks(range(len(names)), [f"{n}\n({Sv.n_Rp[i]}/{Sv.n_Rm[i]})" for i, n in enumerate(names)], fontsize=3.9,
                          rotation=25, ha="right")
            ax.set_ylabel(f"{EFF[e].replace(', whisker', '')}\n({variant})", fontsize=5)
            ax.tick_params(labelsize=4.3)
    fig.suptitle("Figure 4 K statistics per learner definition (n R+/R−). Filled = whisker, open squares = auditory; mean ± s.e.m. "
                 "Top: z-scored axes (Figure 4); bottom: raw-rate axes. W×A int. = cohort × modality, cohort labels permuted "
                 "across mice. Exploratory, uncorrected.", fontsize=5.8, y=1.02, wrap=True)
    save(fig, "populations")

    # ---- figure 4: influence
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.9, wspace=0.45, width_ratios=[2.4, 2.4, 1])
    for r, e in enumerate(("q_W", "q_th_W")):
        ax = fig.add_subplot(gs[r, :2])
        l_ = L.sort_values(f"d_{e}").reset_index(drop=True)
        cols = [GCOL.get(g, "0.5") if c == "R+" else ("k" if g == "bad" else COL["R-"]) for g, c in zip(l_.group, l_.cohort)]
        ax.bar(range(len(l_)), l_[f"d_{e}"], color=cols, width=0.8, edgecolor=["tab:red" if b else "none" for b in l_.border], lw=0.6)
        ax.set_xticks(range(len(l_)), l_.mouse_id, rotation=90, fontsize=3.3)
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_ylabel("Δ (R+ − R−) when dropped", fontsize=5)
        ax.set_title(f"{EFF[e]}: leave one mouse out of all day-0 mice (R+ − R− = {base[e]:+.3g}); MW p over drops "
                     f"{L[f'p_{e}'].min():.3f}–{L[f'p_{e}'].max():.3f}; red outline = border", fontsize=5)
        ax.tick_params(labelsize=4.3)
        ax = fig.add_subplot(gs[r, 2])
        for i, nm in enumerate(("sheet (Fig. 4)", "all day-0 mice", "clear learners only")):
            b = B[(B.population == nm) & (B.effect == e)].iloc[0]
            ax.violinplot(b.boot, [i], widths=0.7, showextrema=False)
            ax.errorbar(i, b["diff"], [[b["diff"] - b.ci_lo], [b.ci_hi - b["diff"]]], fmt="o", ms=3, color="k", lw=0.8)
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_xticks(range(3), ["sheet", "all", "clear"], fontsize=4.5)
        ax.set_ylabel("R+ − R− (bootstrap over mice)", fontsize=4.8)
        ax.tick_params(labelsize=4.3)
    fig.suptitle("Influence of single mice on the cohort difference of Figure 4 K (whisker). Exploratory.", fontsize=6, y=0.99)
    save(fig, "influence")

    # ---- figure 5: controls
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), gridspec_kw=dict(hspace=0.75, wspace=0.55))
    for k, (x, xl) in enumerate((("n_units", "Tracked units"), ("rho_pre", "Pre whisker response size (ρ)"),
                                 ("n_whisker", "Whisker trials (day 0)"), ("hit_w", "Day-0 whisker lick rate"))):
        for r, e in enumerate(("q_W", "q_th_W")):
            ax = axs[r, k]
            for coh in ("R+", "R-"):
                g = d[d.cohort == coh]
                ax.scatter(g[x], g[e], s=6, color=COL[coh], lw=0, alpha=0.8)
                rr, pp, _ = spear(g, x, e)
                ax.text(0.02, 0.97 - 0.1 * (coh == "R-"), f"{coh.replace('-', '−')} ρ {rr:+.2f} p {pp:.2f}", transform=ax.transAxes,
                        fontsize=3.9, color=COL[coh], va="top")
            if x == "n_units":
                ax.set_xscale("log")
            ax.axhline(0, color="0.8", lw=0.5)
            ax.set_xlabel(xl, fontsize=4.8)
            ax.set_ylabel(EFF[e], fontsize=4.8)
            ax.tick_params(labelsize=4.3)
    o = O[(O.effect.isin(["q_W", "q_th_W"]))]
    txt = "; ".join(f"{r_.effect} {r_.scope} [{r_.model}]: cohort p {r_.p_cohort:.3f}" for _, r_ in o.iterrows()
                    if r_.model != "rp * hit_w")
    fig.suptitle("Controls: K effect vs session size, pre response size and behaviour, all day-0 sessions. OLS (009_ols.csv): "
                 + txt, fontsize=4.6, y=1.06, wrap=True)
    save(fig, "controls")


if __name__ == "__main__":
    main()
