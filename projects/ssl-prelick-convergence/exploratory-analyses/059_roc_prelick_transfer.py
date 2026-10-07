"""Do whisker hits look like auditory hits before the lick (more in R+)? Two simple tests.

Data: pre-lick 100 ms window before the corrected first lick (051: baseline-corrected per-trial rates, units with mean
raw rate >= 0.1 Hz on the used trials, quality good or mua; decoder >= 5 units, balanced accuracy >= 0.55); trial variants: all trials, RT-matched (051 trial_set, 3 classes matched).

1. Single neurons ("reward-lick neurons"). Neurons with significant AH vs FA pre-lick ROC (051 result of the same
   variant; selection does not involve whisker trials). Per neuron, class means m_WH, m_AH, m_FA:
     AH-likeness  c = (|m_WH - m_FA| - |m_WH - m_AH|) / |m_AH - m_FA|   in [-1, 1] (triangle inequality):
                  +1 = WH equals AH, -1 = WH equals FA
     AH-like      c > 0 (WH closer to AH than to FA)
     transfer     WH vs FA ROC also significant with the same sign as AH vs FA
   Session (or session x area) values = mean c, fraction AH-like, fraction transfer, over >= MIN_NEUR neurons.
2. Population decoder transfer. L2 logistic regression (C = 0.05, balanced classes, units z-scored on training
   trials) trained on AH vs FA trials with stratified k-fold CV (k = min(5, smallest class)); applied to all WH
   trials (never used for training; averaged over folds).
     wh_as_AH     fraction of WH trials decoded as AH
     hit / fa     held-out AH decoded as AH (TPR), held-out FA decoded as AH (FPR); bacc = (TPR + 1 - FPR) / 2
     transfer_n   (wh_as_AH - FPR) / (TPR - FPR) (0 = like FA, 1 = like AH), only if TPR - FPR >= 0.2, clipped [-1, 2]
   Session x area values kept if bacc >= BACC_MIN (the AH-FA decoder works there).
Statistics as 057 (unit = session; MWU + Welch; interaction by mouse-level cohort permutation; family-wise max-|z|
permutation across areas within metric x variant x level).
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/transfer/
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
m57 = importlib.import_module("057_roc_prelick_lambda")
OUT = m51.OUTROOT / "transfer"
MIN_NEUR_WB, MIN_NEUR_AREA = 10, 5
MIN_UNITS_DEC = 5
BACC_MIN = 0.55
UNIT_SET = ("good", "mua")      # quality_label kept (non-soma and unlabelled excluded)
C_REG = 0.05
VARIANTS = ["all", "rt_matched"]
METRICS = ["likeness", "frac_ah_like", "frac_transfer", "wh_as_AH", "transfer_n", "bacc"]


def neuron_metrics(Xm, sel_af, sig_af, sel_wf, sig_wf, mask):
    """Xm: (units, 3) class means (WH, AH, FA); masks over units"""
    m = mask & sig_af
    if m.sum() == 0:
        return dict(n_neur=0)
    wh, ah, fa = Xm[m, 0], Xm[m, 1], Xm[m, 2]
    den = np.abs(ah - fa)
    c = np.where(den > 0, (np.abs(wh - fa) - np.abs(wh - ah)) / np.where(den > 0, den, 1), np.nan)
    tr = sig_wf[m] & (np.sign(sel_wf[m]) == np.sign(sel_af[m]))
    return dict(n_neur=int(m.sum()), likeness=np.nanmean(c), frac_ah_like=np.nanmean(c > 0), frac_transfer=tr.mean())


def decode(Z, lab, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"
    y = (lab[m] == "AH").astype(int)
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 3 or w.sum() < 3:
        return None
    Xa, Xw = Z[m], Z[w]
    pred = np.zeros(len(y)); whp = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1e9))).split(Xa, y):
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
        clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000)
        clf.fit((Xa[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((Xa[te] - mu) / sd)
        whp.append(clf.predict((Xw - mu) / sd).mean())
    tpr, fpr = pred[y == 1].mean(), pred[y == 0].mean()
    wh = float(np.mean(whp))
    tn = (wh - fpr) / (tpr - fpr) if tpr - fpr >= 0.2 else np.nan
    return dict(wh_as_AH=wh, tpr=tpr, fpr=fpr, bacc=(tpr + 1 - fpr) / 2,
                transfer_n=float(np.clip(tn, -1, 2)) if np.isfinite(tn) else np.nan, n_units_dec=Z.shape[1])


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
        K = K.merge(W[W.session_id == r.session_id], on=["electrode_group", "cluster_id"], how="left")
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
            Xm = np.column_stack([X[:, idx][:, l == c].mean(1) for c in m51.CLASSES])
            def col(name):
                return K[f"{name}@{v}"].to_numpy()
            sig_af = col("sig:auditory_hit_vs_fa_prelick").astype(float) == 1
            sig_wf = col("sig:whisker_hit_vs_fa_prelick").astype(float) == 1
            sel_af, sel_wf = col("sel:auditory_hit_vs_fa_prelick").astype(float), col("sel:whisker_hit_vs_fa_prelick").astype(float)
            qok = K.quality_label.isin(UNIT_SET).to_numpy()
            sig_af &= qok
            ok = (raw[:, idx].mean(1) >= m51.MIN_FR) & qok
            Zall = X[:, idx].T
            for level in ["all", "area_group", "area_acronym_custom"]:
                regs = {"all": np.ones(len(K), bool)} if level == "all" else \
                    {a: (K[level] == a).to_numpy() for a in K[level].dropna().unique()}
                for reg, mreg in regs.items():
                    row = dict(meta, variant=v, level=level, region=reg)
                    nm = neuron_metrics(Xm, sel_af, sig_af, sel_wf, sig_wf, mreg)
                    if nm["n_neur"] >= (MIN_NEUR_WB if level == "all" else MIN_NEUR_AREA):
                        row.update(nm)
                    else:
                        row["n_neur"] = nm["n_neur"]
                    u = ok & mreg
                    if u.sum() >= MIN_UNITS_DEC:
                        d = decode(Zall[:, u], l, rng)
                        if d:
                            row.update(d)
                            if d["bacc"] < BACC_MIN:
                                row["wh_as_AH"] = np.nan; row["transfer_n"] = np.nan
                    rows.append(row)
        if i % 20 == 0:
            print(f"[059] {i + 1}/{len(ss)} sessions", flush=True)
    return pd.DataFrame(rows)


def run_tests(L):
    rng = np.random.default_rng(1)
    mice = L.mouse_id.unique(); mc = L.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    perm_coh = [rng.permutation(mc) for _ in range(m57.N_PERM)]
    rows = []
    for metric in METRICS:
        for v in VARIANTS:
            for level in ["all", "area_group", "area_acronym_custom"]:
                D = L[(L.variant == v) & (L.level == level)]
                if metric not in D or D[metric].notna().sum() == 0:
                    continue
                obs, null = m57.interaction_perm(D, D.region.unique(), perm_coh, mice, metric)
                zs = {r: obs[r] / np.nanstd(null[r]) for r in obs if np.nanstd(null[r]) > 0}
                Z0 = np.column_stack([null[r] / np.nanstd(null[r]) for r in zs]) if zs else None
                maxz = np.nanmax(np.abs(Z0), 1) if Z0 is not None else None
                for reg in D.region.unique():
                    t = m57.group_tests(D[D.region == reg], metric)
                    row = dict(metric=metric, variant=v, level=level, region=reg,
                               **{f"mean_{k}": x for k, x in t["means"].items()}, **{f"n_{k}": x for k, x in t["n"].items()})
                    for c in ["stage:R+", "stage:R-", "cohort:expert", "cohort:learning"]:
                        if c in t:
                            for k in ["diff", "p_mwu", "p_welch"]:
                                row[f"{c}_{k}"] = t[c][k]
                    if reg in zs:
                        nl = null[reg][np.isfinite(null[reg])]
                        row["interaction"] = obs[reg]
                        row["interaction_p"] = (1 + np.sum(np.abs(nl) >= abs(obs[reg]))) / (1 + len(nl))
                        row["interaction_p_maxT"] = (1 + np.sum(maxz >= abs(zs[reg]))) / (1 + len(maxz))
                        row["n_regions_family"] = len(zs)
                    rows.append(row)
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
    fig = plt.figure(figsize=(7.4, 9.6))
    gs = fig.add_gridspec(4, 4, hspace=1.0, wspace=0.6, left=0.08, right=0.98, top=0.9, bottom=0.05)
    wb = {v: L[(L.level == "all") & (L.variant == v)] for v in VARIANTS}
    # decoder panels: chance-corrected single-session decoders from 069 (linear shift for spontaneous licks, label
    # permutation for false alarms), restricted to the sessions of L (population filter applied by the caller)
    nl = "shift"
    fdec = m51.OUTROOT / f"decoder_transfer_{nl}" / "sessions.csv"
    DC = pd.read_csv(fdec) if fdec.exists() else pd.DataFrame(columns=["session_id", "level"])
    DC = DC[(DC.level == "all") & DC.session_id.isin(L.session_id)]
    panels = [
        (gs[0, 0], wb["all"], "likeness", "AH-likeness c (mean)", "a  reward-lick neurons: AH-likeness"),
        (gs[0, 1], wb["all"], "frac_ah_like", "fraction WH closer to AH", "b  fraction AH-like neurons"),
        (gs[0, 2], wb["all"], "frac_transfer", "fraction WH vs FA same sign", "c  ROC transfer"),
        (gs[0, 3], wb["rt_matched"], "likeness", "AH-likeness c", "d  AH-likeness (RT-matched)"),
        (gs[2, 0], DC, "transfer_bin_corrected", "transfer − chance (yes/no)", "f  WH decoded as AH (chance-corr.)"),
        (gs[2, 1], DC, "transfer_prob_corrected", "transfer − chance (probability)", "g  probability transfer (chance-corr.)"),
        (gs[2, 2], DC, "bacc_corrected", "balanced accuracy − chance", "h  decoder above chance"),
        (gs[2, 3], DC, "num_prob_corrected", "P(AH|WH) − P(AH|ref) − chance", "i  WH more AH-like (chance-corr.)")]
    if m51.REF == "sl":                                     # no reaction time for spontaneous licks
        panels[3] = (gs[0, 3], wb["all"], "likeness", "AH-likeness c", "d  AH-likeness (RT matching n/a)")
    for spec, d, col, yl, title in panels:
        ax = fig.add_subplot(spec)
        m58.dots(ax, d, col, yl, m58.test_text(d, col, rng), rng, title)
    for spec, metric, title in [(gs[1, :], "likeness", "e  AH-likeness of reward-lick neurons by area group"),
                                (gs[3, :], "wh_as_AH", "j  WH decoded as AH by area group (raw decoder, not chance-corrected)")]:
        ax = fig.add_subplot(spec)
        A = T[(T.metric == metric) & (T.variant == "all") & (T.level == "area_group")].set_index("region")
        order = [g for g in au.get_area_group_custom_order() if g in A.index and
                 np.isfinite(A.loc[g, ["stage:R+_diff", "stage:R-_diff"]].astype(float)).any()]
        x = np.arange(len(order))
        for k, c in enumerate(["R+", "R-"]):
            d_ = A.loc[order, f"stage:{c}_diff"].astype(float); p_ = A.loc[order, f"stage:{c}_p_mwu"].astype(float)
            ax.scatter(x + (k - 0.5) * 0.25, d_, s=14, color=[m58.COH[c] if pp < 0.05 else "white" for pp in p_.fillna(1)],
                       edgecolors=m58.COH[c], lw=0.8, zorder=3, label=f"{c.replace('-', '−')}: expert − learning")
        top = ax.get_ylim()[1]
        for xi, g in zip(x, order):
            r = A.loc[g]
            if np.isfinite(r.get("interaction_p", np.nan)):
                ax.text(xi, top, f"int {m58.fmt(r.interaction_p)}\nFW {m58.fmt(r.interaction_p_maxT)}\n"
                                 f"nE {int(r['n_R+_expert'])}/{int(r['n_R-_expert'])}", ha="center", va="bottom", fontsize=3.9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x, [g.replace(" areas", "").replace("Somatosensory", "SS") for g in order], rotation=30, ha="right")
        ax.set_ylabel(f"Δ {metric} (E − L)"); ax.legend(frameon=False, fontsize=5, loc="lower left")
        ax.set_title(title + " (filled = MWU p < .05; int = learning x cohort, FW = family-wise max-T)",
                     loc="left", fontsize=6, pad=18)
    fig.suptitle("Do whisker hits look like auditory hits before the lick? Single neurons (a-e) and population decoder (f-j); "
                 "unit = session", fontsize=7, y=0.995)
    importlib.import_module("062_pub_convergence_figures").relabel_reference(fig) if m51.REF == "sl" else None
    fig.savefig(OUT / "transfer_summary.png", dpi=230); fig.savefig(OUT / "transfer_summary.pdf"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "transfer_sessions.csv"
    L = pd.read_csv(f) if f.exists() else compute()
    L.to_csv(f, index=False)
    T = run_tests(L); T.to_csv(OUT / "transfer_tests.csv", index=False)
    json.dump(dict(script="059_roc_prelick_transfer.py", min_neurons_wholebrain=MIN_NEUR_WB, min_neurons_area=MIN_NEUR_AREA,
                   min_units_decoder=MIN_UNITS_DEC, bacc_min=BACC_MIN, C=C_REG, variants=VARIANTS,
                   selection="AH vs FA pre-lick ROC significant (051, same variant)", unit_of_analysis="session",
                   tests="MWU + Welch; interaction mouse-level cohort permutation; family-wise max-|z| across areas"),
              open(OUT / "provenance.json", "w"), indent=1)
    figure(L, T)
    pd.set_option("display.width", 250)
    cols = ["metric", "variant", "region", "mean_R+_learning", "mean_R+_expert", "mean_R-_learning", "mean_R-_expert",
            "stage:R+_p_mwu", "stage:R-_p_mwu", "cohort:expert_p_mwu", "interaction", "interaction_p", "interaction_p_maxT",
            "n_R+_expert", "n_R-_expert"]
    print(T[T.level == "all"][cols].round(3).to_string())
    a = T[(T.level == "area_group") & ((T.interaction_p < 0.05) | (T["stage:R+_p_mwu"] < 0.05))]
    print(a[cols].round(3).to_string())


if __name__ == "__main__":
    main()
