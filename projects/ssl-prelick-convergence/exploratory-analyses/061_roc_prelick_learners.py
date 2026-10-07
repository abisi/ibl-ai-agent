"""Convergence analyses restricted to learners + lambda_LDA projections for example areas.

Learners: day-0 sessions of mice with learning_category in {good, moderate}; non-learner mice (bad or missing) lose
only their day-0 session, their expert (day >= 1) sessions are kept (user rule 2026-10-02).
Re-tested without recomputation, from the per-session tables of: 057 lambda (lambda, dd), 059 single neurons
(likeness, frac_ah_like, frac_transfer) and decoder (wh_as_AH, transfer_n, bacc), 060 lambda_LDA (lam_lda, d_prime).
Statistics as 057 / 059 (unit = session; MWU + Welch; interaction mouse-level cohort permutation; family-wise max-|z|).
Projections: for the example regions, held-out shrinkage-LDA scores (060 procedure) of AH, FA and WH trials, normalised
per session so that the held-out FA mean = 0 and the held-out AH mean = 1 (per fold, then pooled); shown pooled over
learner sessions per cohort x stage (trial distributions + session means) and for one example session per group
(the session with the median lambda_LDA).
Output (new folder, nothing overwritten): combined_results_ks4/ssl-prelick-convergence/across_days/fa/learners/
"""
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m59 = importlib.import_module("059_roc_prelick_transfer")
m60 = importlib.import_module("060_roc_prelick_lambda_lda")
BASE = m51.OUTROOT
OUT = BASE / "learners"
LEARNER_CATS = ["good", "moderate"]
EX_REGIONS = [("all", "all"), ("area_group", "Motor areas"), ("area_group", "Hippocampus"), ("area_group", "Thalamus"),
              ("area_group", "Somatosensory-whisker"), ("area_group", "Frontal areas")]
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
CCOL = {"WH": "#1f77b4", "AH": "#d62728", "FA": "#7f7f7f"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
SUMMARY = [("lambda", "lam", "λ"), ("lambda", "dd", "Δd"), ("neurons", "likeness", "AH-likeness"),
           ("neurons", "frac_ah_like", "frac. AH-like"), ("neurons", "frac_transfer", "ROC transfer"),
           ("decoder", "wh_as_AH", "WH decoded as AH"), ("decoder", "transfer_n", "norm. transfer"),
           ("lda", "lam_lda", "λ_LDA")]


def learner_mice():
    U = pd.read_parquet(m51.RES / "_roc_stage_analysis" / "units.parquet", columns=["mouse_id", "learning_category"])
    U = U.drop_duplicates("mouse_id")
    return set(U[U.learning_category.isin(LEARNER_CATS)].mouse_id), U


def learner_filter(df, learners=None):
    """learners population (user rule 2026-10-02): the learning category describes day 0, so a non-learner mouse
    (category bad or missing) loses only its learning-day (day 0) sessions; its expert (day >= 1) sessions are kept.
    df needs mouse_id and stage."""
    if learners is None:
        learners, _ = learner_mice()
    return df[df.mouse_id.isin(learners) | (df.stage == "expert")]


def projections(sessions, rng):
    """normalised held-out LDA scores per trial for the example regions (variant all)"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    W = pd.read_parquet(BASE / "prelick_units.parquet")
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(sessions)]
    rows = []
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "area_group", "cohort", "stage", "quality_label"]],
                    on=["electrode_group", "cluster_id"], how="left")
        coh, stg = K.cohort.dropna().iloc[0], K.stage.dropna().iloc[0]
        X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
        ok = (raw.mean(1) >= m51.MIN_FR) & K.quality_label.isin(m60.UNIT_SET).to_numpy()
        for level, reg in EX_REGIONS:
            u = ok & (np.ones(len(K), bool) if level == "all" else (K.area_group == reg).to_numpy())
            if u.sum() < m60.MIN_UNITS:
                continue
            Z = X[u].T
            m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"; y = (lab[m] == "AH").astype(int)
            k = min(5, int(y.sum()), int((1 - y).sum()))
            if k < 3 or w.sum() < 3:
                continue
            Xa, Xw = Z[m], Z[w]
            wh_scores = np.zeros((k, w.sum())); held = []
            for f, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1e9))).split(Xa, y)):
                mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit((Xa[tr] - mu) / sd, y[tr])
                s_te = lda.decision_function((Xa[te] - mu) / sd); s_w = lda.decision_function((Xw - mu) / sd)
                fa_m, ah_m = s_te[y[te] == 0].mean(), s_te[y[te] == 1].mean()
                if ah_m - fa_m <= 0:
                    wh_scores[f] = np.nan; continue
                held += [(c, (s - fa_m) / (ah_m - fa_m)) for c, s in zip(np.where(y[te] == 1, "AH", "FA"), s_te)]
                wh_scores[f] = (s_w - fa_m) / (ah_m - fa_m)
            whm = np.nanmean(wh_scores, 0)
            for c, s in held + [("WH", s) for s in whm]:
                rows.append(dict(session_id=r.session_id, cohort=coh, stage=stg, region=reg, cls=c, score=s))
    return pd.DataFrame(rows)


def proj_figures(P, lda_sess, plt):
    keep = lda_sess.dropna(subset=["lam_lda"])[["session_id", "region"]].drop_duplicates()
    P = P.merge(keep, on=["session_id", "region"])                  # sessions kept by lambda_LDA (held-out d' >= 0.5)
    XL = (-2.0, 3.0)
    bins = np.linspace(XL[0], XL[1], 41)
    regs = [r for _, r in EX_REGIONS if r in set(P.region)]
    fig, axs = plt.subplots(len(regs), 4, figsize=(7.4, 1.35 * len(regs) + 0.7), sharex=True,
                            gridspec_kw=dict(hspace=0.75, wspace=0.3))
    for i, reg in enumerate(regs):
        for j, (c, s_) in enumerate(GROUPS):
            ax = axs[i, j]; d = P[(P.region == reg) & (P.cohort == c) & (P.stage == s_)]
            for cl in ["FA", "AH", "WH"]:
                v = d[d.cls == cl].score.dropna(); v = v[(v > XL[0]) & (v < XL[1])]
                if len(v):
                    ax.hist(v, bins, density=True, histtype="stepfilled" if cl != "WH" else "step", alpha=0.35 if cl != "WH" else 1,
                            color=CCOL[cl], lw=1.1 if cl == "WH" else 0, label=cl)
            sm = d[d.cls == "WH"].groupby("session_id").score.mean().clip(*XL)
            ax.plot(sm, np.full(len(sm), ax.get_ylim()[1] * 0.92), "|", color=CCOL["WH"], ms=5, mew=0.8)
            ax.set_xlim(*XL)
            lam = lda_sess[(lda_sess.region == reg) & (lda_sess.cohort == c) & (lda_sess.stage == s_)].lam_lda
            ax.axvline(0, color=CCOL["FA"], lw=0.6, ls="--"); ax.axvline(1, color=CCOL["AH"], lw=0.6, ls="--")
            ax.set_title(f"{reg.replace(' areas', '')}: λ_LDA {lam.mean():.2f} ({lam.notna().sum()} sess.)", fontsize=5.2)
            ax.tick_params(labelsize=4.5)
            if i == 0:
                ax.text(0.5, 1.45, f"{c.replace('-', '−')} {s_}", transform=ax.transAxes, ha="center", color=COH[c],
                        fontsize=6.5, weight="bold")
            if i == len(regs) - 1:
                ax.set_xlabel("LDA score (FA = 0, AH = 1)", fontsize=5)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=4.5, loc="upper right")
    fig.suptitle("λ_LDA projections (learners): held-out AH / FA and all WH trials on the AH-vs-FA LDA axis "
                 "(sessions with held-out d' >= 0.5; ticks = session WH means)", fontsize=6.5, y=0.998)
    fig.subplots_adjust(left=0.06, right=0.98, top=1 - 0.8 / (1.35 * len(regs) + 0.7), bottom=0.05)
    importlib.import_module("062_pub_convergence_figures").relabel_reference(fig) if m51.REF == "sl" else None
    fig.savefig(OUT / "lambda_lda_projections_pooled.png", dpi=230); fig.savefig(OUT / "lambda_lda_projections_pooled.pdf")
    plt.close(fig)
    # single example sessions (median lambda_LDA per group), strip plots
    fig, axs = plt.subplots(len(regs), 4, figsize=(7.4, 1.35 * len(regs) + 0.7), sharey=False,
                            gridspec_kw=dict(hspace=0.8, wspace=0.35))
    rng = np.random.default_rng(0)
    for i, reg in enumerate(regs):
        for j, (c, s_) in enumerate(GROUPS):
            ax = axs[i, j]
            cand = lda_sess[(lda_sess.region == reg) & (lda_sess.cohort == c) & (lda_sess.stage == s_)].dropna(subset=["lam_lda"])
            cand = cand[cand.session_id.isin(P[(P.region == reg)].session_id)]
            if not len(cand):
                ax.axis("off"); continue
            sid = cand.iloc[(cand.lam_lda - cand.lam_lda.median()).abs().argsort().iloc[0]].session_id
            d = P[(P.region == reg) & (P.session_id == sid)]
            for k, cl in enumerate(["FA", "AH", "WH"]):
                v = d[d.cls == cl].score.dropna().to_numpy()
                ax.scatter(rng.uniform(-0.25, 0.25, len(v)) + k, v, s=3, color=CCOL[cl], alpha=0.6, lw=0)
                if len(v):
                    ax.plot([k - 0.3, k + 0.3], [v.mean()] * 2, color="k", lw=1)
            ax.axhline(0, color=CCOL["FA"], lw=0.5, ls="--"); ax.axhline(1, color=CCOL["AH"], lw=0.5, ls="--")
            ax.set_xticks(range(3), ["FA", "AH", "WH"]); ax.tick_params(labelsize=4.5)
            lam = cand[cand.session_id == sid].lam_lda.iloc[0]
            ax.set_title(f"{reg.replace(' areas', '')} {sid[:5]} {sid[6:14]}: λ_LDA {lam:.2f}", fontsize=4.8)
            if j == 0:
                ax.set_ylabel("LDA score", fontsize=5)
            if i == 0:
                ax.text(0.5, 1.45, f"{c.replace('-', '−')} {s_}", transform=ax.transAxes, ha="center", color=COH[c],
                        fontsize=6.5, weight="bold")
    fig.suptitle("λ_LDA example sessions (learners; session with the median λ_LDA of each group): single trials on the "
                 "AH-vs-FA LDA axis (FA = 0, AH = 1)", fontsize=6.5, y=0.998)
    fig.subplots_adjust(left=0.07, right=0.98, top=1 - 0.8 / (1.35 * len(regs) + 0.7), bottom=0.04)
    importlib.import_module("062_pub_convergence_figures").relabel_reference(fig) if m51.REF == "sl" else None
    fig.savefig(OUT / "lambda_lda_projections_examples.png", dpi=230); fig.savefig(OUT / "lambda_lda_projections_examples.pdf")
    plt.close(fig)


def contrast_figure(Tall, Tlearn, plt):
    """whole-brain contrast, all mice vs learners: per measure, R+ and R- change, expert R+-R-, interaction"""
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    labels = [s[2] for s in SUMMARY]
    y = np.arange(len(SUMMARY))
    for ax, (col, ttl) in zip(axs, [("stage:R+", "R+: expert − learning"), ("stage:R-", "R−: expert − learning"),
                                    ("cohort:expert", "expert: R+ − R−"), ("interaction", "interaction")]):
        for k, (T, name, mk) in enumerate([(Tall, "all mice", "o"), (Tlearn, "learners", "s")]):
            ps = []
            for i, (src, met, _) in enumerate(SUMMARY):
                r = T[(T.src == src) & (T.metric == met) & (T.level == "all") & (T.variant == "all")]
                if not len(r):
                    ps.append(np.nan); continue
                r = r.iloc[0]
                p = r["interaction_p"] if col == "interaction" else r.get(f"{col}_p_mwu", np.nan)
                ps.append(p)
            ps = np.array(ps, float)
            ax.scatter(-np.log10(ps), y + (k - 0.5) * 0.3, marker=mk, s=16, color="k" if k == 0 else "#e6550d",
                       label=name, zorder=3)
        ax.axvline(-np.log10(0.05), color="0.5", lw=0.6, ls="--")
        ax.set_title(ttl, fontsize=6); ax.set_xlabel("−log10 p (MWU; interaction: perm.)")
    axs[0].set_yticks(y, labels, fontsize=5.5); axs[0].invert_yaxis(); axs[0].legend(frameon=False, fontsize=5)
    fig.suptitle("Whole-brain convergence tests: all mice vs learners only (dashed: p = .05)", fontsize=7)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.85, bottom=0.17)
    importlib.import_module("062_pub_convergence_figures").relabel_reference(fig) if m51.REF == "sl" else None
    fig.savefig(OUT / "contrast_all_vs_learners.png", dpi=230); fig.savefig(OUT / "contrast_all_vs_learners.pdf"); plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5.2, "ytick.labelsize": 5.2, "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    learners, U = learner_mice()
    src = {"lambda": (BASE / "lambda" / "lambda_sessions.csv", ["lam", "dd"]),
           "neurons": (BASE / "transfer" / "transfer_sessions.csv", ["likeness", "frac_ah_like", "frac_transfer"]),
           "decoder": (BASE / "transfer" / "transfer_sessions.csv", ["wh_as_AH", "transfer_n", "bacc"]),
           "lda": (BASE / "lambda_lda" / "lambda_lda_sessions.csv", ["lam_lda", "d_prime"])}
    tabs_all, tabs_learn, counts = [], [], {}
    m59.VARIANTS = ["all", "rt_matched"]
    for name, (f, mets) in src.items():
        L = pd.read_csv(f)
        Ll = learner_filter(L, learners)
        counts[name] = {f"{a}_{b}": int(n) for (a, b), n in Ll[Ll.level == "all"].drop_duplicates("session_id").groupby(["cohort", "stage"]).size().items()}
        m59.METRICS = mets
        for L_, store in [(L, tabs_all), (Ll, tabs_learn)]:
            t = m59.run_tests(L_); t["src"] = name; store.append(t)
        Ll.to_csv(OUT / f"{name}_sessions_learners.csv", index=False)
    Tall, Tl = pd.concat(tabs_all), pd.concat(tabs_learn)
    Tl.to_csv(OUT / "tests_learners.csv", index=False); Tall.to_csv(OUT / "tests_all_mice.csv", index=False)
    contrast_figure(Tall, Tl, plt)
    # full figures of 059 / 060 for learners, written into the learners folder
    m59.OUT = OUT; m60.OUT = OUT
    Lt = learner_filter(pd.read_csv(BASE / "transfer" / "transfer_sessions.csv"), learners)
    m59.METRICS = ["likeness", "frac_ah_like", "frac_transfer", "wh_as_AH", "transfer_n", "bacc"]
    m59.figure(Lt, Tl[Tl.src.isin(["neurons", "decoder"])])
    import shutil
    for ext in ["png", "pdf"]:
        shutil.move(OUT / f"transfer_summary.{ext}", OUT / f"transfer_summary_learners.{ext}")
    Ld = learner_filter(pd.read_csv(BASE / "lambda_lda" / "lambda_lda_sessions.csv"), learners)
    m60.figure(Ld, Tl[Tl.src == "lda"])
    for ext in ["png", "pdf"]:
        shutil.move(OUT / f"lambda_lda_summary.{ext}", OUT / f"lambda_lda_summary_learners.{ext}")
    # projections
    fp = OUT / "lambda_lda_projection_scores_learners.csv"
    P = pd.read_csv(fp) if fp.exists() else projections(set(Ld.session_id), np.random.default_rng(0))
    P.to_csv(fp, index=False)
    proj_figures(P, Ld[(Ld.variant == "all") & Ld.level.isin(["all", "area_group"])], plt)
    json.dump(dict(script="061_roc_prelick_learners.py", learner_categories=LEARNER_CATS, n_learner_mice=len(learners),
                   sessions_per_group=counts, example_regions=EX_REGIONS), open(OUT / "provenance.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = ["src", "metric", "variant", "mean_R+_learning", "mean_R+_expert", "mean_R-_learning", "mean_R-_expert",
            "stage:R+_p_mwu", "stage:R-_p_mwu", "cohort:expert_p_mwu", "interaction", "interaction_p", "n_R+_expert", "n_R-_expert"]
    print(counts)
    print(Tl[(Tl.level == "all")][cols].round(3).to_string())
    a = Tl[(Tl.level == "area_group") & (Tl.interaction_p < 0.05)]
    print(a[cols[:3] + ["region", "stage:R+_p_mwu", "interaction", "interaction_p", "interaction_p_maxT", "n_R+_expert",
                        "n_R-_expert"]].round(3).to_string())


if __name__ == "__main__":
    main()
