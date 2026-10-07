"""lambda_LDA: position of whisker hits on a cross-validated shrinkage-LDA axis separating false alarms (0) from
auditory hits (1), pre-lick 100 ms window before the corrected first lick.

Per session x area (units with mean raw pre-lick rate >= 0.1 Hz on the used trials, quality good or mua, >= MIN_UNITS = 5 units), per trial variant
(all, rt_matched): stratified k-fold over AH and FA trials (k = min(5, smallest class)); in each fold, units are z-scored
on the training trials and an LDA (solver lsqr, Ledoit-Wolf shrinkage) is fit on training AH vs FA; held-out AH, held-out
FA and ALL WH trials (never used for fitting) are projected on the discriminant axis w.
  lambda_LDA = sum_folds (mean s_WH - mean s_FA,test) / sum_folds (mean s_AH,test - mean s_FA,test)
  d_prime    = held-out separation of AH and FA scores (pooled over folds; pooled SD)
Kept if d_prime >= DPRIME_MIN (a reliable AH-FA axis); clipped to [-1, 2].
Difference from lambda (057): the axis is Sigma^-1 (mu_AH - mu_FA) (noise-whitened, shrinkage-regularised) instead of
the plain mean difference, so noisy / correlated directions are down-weighted.
Statistics as 057 / 059 (unit = session; MWU + Welch; interaction by mouse-level cohort permutation; family-wise max-|z|
across areas). Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/lambda_lda/
"""
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m59 = importlib.import_module("059_roc_prelick_transfer")
OUT = m51.OUTROOT / "lambda_lda"
MIN_UNITS = 5
UNIT_SET = ("good", "mua")      # quality_label kept (non-soma and unlabelled excluded)
DPRIME_MIN = 0.3             # held-out d' of the AH-FA axis (user 2026-10-02: smaller than the first 0.5)
CLIP = (-1.0, 2.0)
VARIANTS = ["all", "rt_matched"]


def lambda_lda(Z, lab, rng):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"
    y = (lab[m] == "AH").astype(int)
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 3 or w.sum() < 3:
        return None
    Xa, Xw = Z[m], Z[w]
    num = den = 0.0
    sA, sF = [], []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1e9))).split(Xa, y):
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit((Xa[tr] - mu) / sd, y[tr])
        s_te = lda.decision_function((Xa[te] - mu) / sd); s_w = lda.decision_function((Xw - mu) / sd)
        a, f = s_te[y[te] == 1], s_te[y[te] == 0]
        num += s_w.mean() - f.mean(); den += a.mean() - f.mean()
        sA.append(a - a.mean() + (a.mean() - f.mean())); sF.append(f - f.mean())   # centred per fold, separation kept
    A, F = np.concatenate(sA), np.concatenate(sF)
    pooled = np.sqrt((A.var(ddof=1) * (len(A) - 1) + F.var(ddof=1) * (len(F) - 1)) / (len(A) + len(F) - 2))
    dprime = (A.mean() - F.mean()) / pooled if pooled > 0 else np.nan
    lam = num / den if den > 0 else np.nan
    return dict(lam_lda=float(np.clip(lam, *CLIP)) if np.isfinite(lam) and dprime >= DPRIME_MIN else np.nan,
                lam_lda_raw=lam, d_prime=dprime, n_units=Z.shape[1])


def compute():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    rng = np.random.default_rng(0)
    rows = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "area_group", "area_acronym_custom",
                                                      "cohort", "stage", "mouse_id", "quality_label"]], on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        meta = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                    stage=K.stage.dropna().iloc[0])
        X, raw, lab, rt = z["rates"].astype(float), z["raw"].astype(float), z["cls"], z["rt"]
        for v in VARIANTS:
            idx = m51.trial_set(lab, rt, m51.CLASSES, v, np.random.default_rng(0))
            l = lab[idx]
            if min((l == c).sum() for c in m51.CLASSES) < 3:
                continue
            ok = (raw[:, idx].mean(1) >= m51.MIN_FR) & K.quality_label.isin(UNIT_SET).to_numpy()
            Zall = X[:, idx].T
            for level in ["all", "area_group", "area_acronym_custom"]:
                regs = {"all": np.ones(len(K), bool)} if level == "all" else \
                    {a: (K[level] == a).to_numpy() for a in K[level].dropna().unique()}
                for reg, mreg in regs.items():
                    u = ok & mreg
                    if u.sum() < MIN_UNITS:
                        continue
                    res = lambda_lda(Zall[:, u], l, rng)
                    if res:
                        rows.append(dict(meta, variant=v, level=level, region=reg, **res))
        if i % 20 == 0:
            print(f"[060] {i + 1}/{len(ss)} sessions", flush=True)
    return pd.DataFrame(rows)


def figure(L, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m58 = importlib.import_module("058_roc_prelick_convergence_summary")
    import ephys_utilities.allen_utils.allen_utils as au
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5.2, "ytick.labelsize": 5.2, "pdf.fonttype": 42})
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(3, 4, hspace=1.05, wspace=0.6, left=0.08, right=0.98, top=0.88, bottom=0.06)
    wb = {v: L[(L.level == "all") & (L.variant == v)] for v in VARIANTS}
    for spec, d, col, yl, title in [(gs[0, 0], wb["all"], "lam_lda", "λ_LDA", "a  λ_LDA, whole brain"),
                                    (gs[0, 1], wb["rt_matched"], "lam_lda", "λ_LDA", "b  λ_LDA, RT-matched"),
                                    (gs[0, 2], wb["all"], "d_prime", "held-out d′ (AH vs FA)", "c  axis quality (check)")]:
        ax = fig.add_subplot(spec)
        m58.dots(ax, d, col, yl, m58.test_text(d, col, rng), rng, title)
    ax = fig.add_subplot(gs[0, 3])
    lam = pd.read_csv(m51.OUTROOT / "lambda" / "lambda_sessions.csv")
    M = L.merge(lam[["session_id", "variant", "level", "region", "lam"]], on=["session_id", "variant", "level", "region"])
    M = M[(M.variant == "all") & (M.level == "all")].dropna(subset=["lam", "lam_lda"])
    for c in ["R+", "R-"]:
        mm = M[M.cohort == c]
        ax.scatter(mm.lam, mm.lam_lda, s=6, color=m58.COH[c], alpha=0.7, lw=0, label=c.replace("-", "−"))
    r, p = stats.spearmanr(M.lam, M.lam_lda)
    ax.plot([-1, 2], [-1, 2], color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("λ (mean-difference axis)"); ax.set_ylabel("λ_LDA")
    ax.set_title(f"d  λ vs λ_LDA (whole brain)\nSpearman r = {r:.2f}, {m58.fmt(p)}", fontsize=5.5)
    ax.legend(frameon=False, fontsize=5)
    for row, v in [(1, "all"), (2, "rt_matched")]:
        ax = fig.add_subplot(gs[row, :])
        A = T[(T.metric == "lam_lda") & (T.variant == v) & (T.level == "area_group")].set_index("region")
        order = [g for g in au.get_area_group_custom_order() if g in A.index and
                 np.isfinite(A.loc[g, ["stage:R+_diff", "stage:R-_diff"]].astype(float)).any()]
        x = np.arange(len(order))
        for k, c in enumerate(["R+", "R-"]):
            d_ = A.loc[order, f"stage:{c}_diff"].astype(float); p_ = A.loc[order, f"stage:{c}_p_mwu"].astype(float)
            ax.scatter(x + (k - 0.5) * 0.25, d_, s=14, color=[m58.COH[c] if pp < 0.05 else "white" for pp in p_.fillna(1)],
                       edgecolors=m58.COH[c], lw=0.8, zorder=3, label=f"{c.replace('-', '−')}: expert − learning")
        top = ax.get_ylim()[1]
        for xi, g in zip(x, order):
            rr = A.loc[g]
            if np.isfinite(rr.get("interaction_p", np.nan)):
                ax.text(xi, top, f"int {m58.fmt(rr.interaction_p)}\nFW {m58.fmt(rr.interaction_p_maxT)}\n"
                                 f"nE {int(rr['n_R+_expert'])}/{int(rr['n_R-_expert'])}", ha="center", va="bottom", fontsize=3.9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x, [g.replace(" areas", "").replace("Somatosensory", "SS") for g in order], rotation=30, ha="right")
        ax.set_ylabel("Δ λ_LDA (E − L)"); ax.legend(frameon=False, fontsize=5, loc="lower left")
        ax.set_title(f"{'e' if row == 1 else 'f'}  area groups, {'all trials' if v == 'all' else 'RT-matched'} "
                     "(filled = MWU p < .05; int = learning x cohort; FW = family-wise max-T)", loc="left", fontsize=6, pad=18)
    fig.suptitle("λ_LDA: whisker hits on the cross-validated shrinkage-LDA axis from false alarms (0) to auditory hits (1); "
                 "unit = session", fontsize=7, y=0.995)
    importlib.import_module("062_pub_convergence_figures").relabel_reference(fig) if m51.REF == "sl" else None
    fig.savefig(OUT / "lambda_lda_summary.png", dpi=230); fig.savefig(OUT / "lambda_lda_summary.pdf"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "lambda_lda_sessions.csv"
    L = pd.read_csv(f) if f.exists() else compute()
    # lambda_LDA from the raw ratio with the current threshold (raw is NaN when the AH-FA separation is <= 0)
    L["lam_lda"] = np.where(np.isfinite(L.lam_lda_raw) & (L.d_prime >= DPRIME_MIN), L.lam_lda_raw.clip(*CLIP), np.nan)
    L.to_csv(f, index=False)
    m59.METRICS = ["lam_lda", "d_prime"]; m59.VARIANTS = VARIANTS
    T = m59.run_tests(L); T.to_csv(OUT / "lambda_lda_tests.csv", index=False)
    json.dump(dict(script="060_roc_prelick_lambda_lda.py", min_units=MIN_UNITS, dprime_min=DPRIME_MIN, clip=CLIP,
                   lda="sklearn LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), units z-scored on training folds",
                   cv="stratified k-fold over AH/FA (k = min(5, smallest class)); WH never used for fitting",
                   unit_of_analysis="session"), open(OUT / "provenance.json", "w"), indent=1)
    figure(L, T)
    pd.set_option("display.width", 250)
    cols = ["metric", "variant", "region", "mean_R+_learning", "mean_R+_expert", "mean_R-_learning", "mean_R-_expert",
            "stage:R+_p_mwu", "stage:R-_p_mwu", "cohort:expert_p_mwu", "interaction", "interaction_p", "interaction_p_maxT",
            "n_R+_expert", "n_R-_expert"]
    print(T[T.level == "all"][cols].round(3).to_string())
    a = T[(T.level == "area_group") & (T.metric == "lam_lda") & ((T.interaction_p < 0.05) | (T["stage:R+_p_mwu"] < 0.05))]
    print(a[cols].round(3).to_string())
    print("kept fraction (d' >= min):", L.groupby(["variant", "level"]).lam_lda.apply(lambda x: x.notna().mean()).round(2).to_dict())


if __name__ == "__main__":
    main()
