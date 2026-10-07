"""079 -- Supplementary analyses (user review 2026-10-07).

FigS_converging_sign: converging neurons defined by the sign of the reward-lick response. Reward-lick neurons: AH vs ref
  ROC significant (051; 1,000 permutations, p < 0.05), split into AH > ref (s_AH > 0) and AH < ref (s_AH < 0). Per
  session (>= 3 reward-lick neurons of that sign, good + mua, mean pre-lick rate >= 0.1 Hz):
    converging (AH > ref)  fraction of AH > ref neurons whose WH vs ref ROC is significant with WH > ref
    converging (AH < ref)  fraction of AH < ref neurons with WH vs ref significant, WH < ref
    converging (both)      current definition (either sign, same sign for WH)
    AH-likeness            c = (|m_WH - m_ref| - |m_WH - m_AH|) / |m_AH - m_ref| per neuron (class means of the 051
                           baseline-corrected pre-lick rates), mean over the neurons of each sign
  Statistics: 062 dots_panel (MWU learning vs expert, expert cohort contrast, mouse-level interaction).
FigS_decoder_quality: does the quality of the AH vs ref decoder limit the readouts? Per session (069, whole brain):
  raw and chance-corrected balanced accuracy against the chance-corrected readouts (transfer yes/no, transfer
  probability, P(AH|WH) - P(AH|ref)) and against lambda (057), Spearman rho per cohort; and the readouts within tertiles
  of decoder accuracy, per cohort x stage.
Output: across_days/<ref>/publication/<pop>/FigS_converging_sign.*, FigS_decoder_quality.*, supp_079_<pop>.csv
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
AF, WF = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
MIN_NEUR = 3


def converging_sessions(W):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    rows = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        g = W[W.session_id == r.session_id]
        K = K.merge(g[["electrode_group", "cluster_id", f"sel:{AF}", f"sig:{AF}", f"sel:{WF}", f"sig:{WF}", "cohort", "stage",
                       "mouse_id"]], on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
        ok = K[f"sig:{AF}"].notna().to_numpy() & K[f"sig:{WF}"].notna().to_numpy() & (raw.mean(1) >= m51.MIN_FR)
        m = {c: X[:, lab == c].mean(1) for c in m51.CLASSES}
        sa, ga = K[f"sel:{AF}"].to_numpy(float), K[f"sig:{AF}"].to_numpy(float) == 1
        sw, gw = K[f"sel:{WF}"].to_numpy(float), K[f"sig:{WF}"].to_numpy(float) == 1
        den = np.abs(m["AH"] - m["FA"])
        c = np.where(den > 0, (np.abs(m["WH"] - m["FA"]) - np.abs(m["WH"] - m["AH"])) / np.where(den > 0, den, 1), np.nan)
        row = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                   stage=K.stage.dropna().iloc[0])
        for key, sel in (("pos", ok & ga & (sa > 0)), ("neg", ok & ga & (sa < 0)), ("both", ok & ga)):
            n = int(sel.sum()); row[f"n_{key}"] = n
            if n < MIN_NEUR:
                continue
            same = gw & (np.sign(sw) == np.sign(sa))
            row[f"conv_{key}"] = float(same[sel].mean())
            row[f"like_{key}"] = float(np.nanmean(c[sel]))
        rows.append(row)
    return pd.DataFrame(rows)


def main(a):
    plt = m62.setup()
    RA = "FA" if m51.REF == "fa" else "SL"
    out = m51.OUTROOT / "publication" / a.population
    keep = m61.learner_filter if a.population == "learners" else (lambda df: df)
    rng = np.random.default_rng(0)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(m62.UNIT_SET)]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    C = keep(converging_sessions(W))
    # --- converging by sign
    fig = plt.figure(figsize=(m62.W_IN, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.6, left=0.08, right=0.98, top=0.88, bottom=0.07)
    spec = [("conv_pos", f"Converging, AH > {RA}\n(WH > {RA} significant)"), ("conv_neg", f"Converging, AH < {RA}\n(WH < {RA} significant)"),
            ("conv_both", "Converging, both signs\n(current definition)"), ("like_pos", f"AH-likeness, AH > {RA} neurons"),
            ("like_neg", f"AH-likeness, AH < {RA} neurons"), ("like_both", "AH-likeness, both signs")]
    axs = []
    for k, (col, ttl) in enumerate(spec):
        ax = fig.add_subplot(gs[k // 3, k % 3]); axs.append(ax)
        yl = "Fraction of reward-lick neurons" if col.startswith("conv") else f"AH-likeness (−1 {RA}, +1 AH)"
        m62.dots_panel(ax, C, col, yl if k % 3 == 0 else "", rng, f"S-conv {col}", ttl, ref=[(0, "0.6")] if "like" in col else None)
        ax.set_title(ax.get_title(), fontsize=5.4)
    m62.letter_row(fig, axs[:3], "abc"); m62.letter_row(fig, axs[3:], "def")
    fig.suptitle(f"Figure 2—supplement | Converging neurons by sign of the reward-lick response ({RA}, {a.population}; sessions with "
                 f"≥ {MIN_NEUR} neurons of that sign)", x=0.02, y=0.99, ha="left", va="top", fontsize=6.8, weight="bold")
    m62.save(fig, out, "FigS_converging_sign"); plt.close(fig)
    # --- decoder quality
    D = keep(pd.read_csv(m51.OUTROOT / "decoder_transfer_shift" / "sessions.csv"))
    D = D[D.level == "all"]
    L = pd.read_csv(m51.OUTROOT / "lambda" / "lambda_sessions.csv")
    L = L[(L.level == "all") & (L.variant == "all")][["session_id", "lam"]]
    D = D.merge(L, on="session_id", how="left")
    Y = [("transfer_bin_corrected", "Transfer (yes/no) − chance"), ("transfer_prob_corrected", "Transfer (probability) − chance"),
         ("num_prob_corrected", f"P(AH|WH) − P(AH|{RA}) − chance"), ("lam", "λ (population geometry)")]
    fig = plt.figure(figsize=(m62.W_IN, 5.6))
    gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.88, bottom=0.08)
    axs = []
    for j, (yc, yl) in enumerate(Y):
        ax = fig.add_subplot(gs[0, j]); axs.append(ax)
        txt = []
        for coh in ("R+", "R-"):
            q = D[D.cohort == coh][["bacc", yc, "stage"]].dropna()
            for st_, fc in (("learning", "white"), ("expert", m62.COH[coh])):
                g = q[q.stage == st_]
                ax.scatter(g.bacc, g[yc], s=6, facecolor=fc, edgecolor=m62.COH[coh], lw=0.5)
            if len(q) >= 5:
                rho, p = stats.spearmanr(q.bacc, q[yc])
                txt.append(f"{coh.replace('-', '−')}: ρ = {rho:.2f}, {m62.fmt_p(p)}")
                m62.STATS.append(dict(panel="S-decq", measure=f"{yc} vs bacc", cohort=coh, rho=rho, p=p, n=len(q)))
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlabel("Raw balanced accuracy (AH vs SL)", fontsize=5); ax.set_ylabel(yl, fontsize=5)
        ax.set_title("\n".join(txt), loc="left", fontsize=4.4)
        # tertiles of decoder quality
        ax = fig.add_subplot(gs[1, j]); axs.append(ax)
        D["tert"] = pd.qcut(D.bacc, 3, labels=["low", "mid", "high"])
        for k_, gk in enumerate(m62.GROUPS):
            q = D[(D.cohort == gk[0]) & (D.stage == gk[1])]
            mu = q.groupby("tert", observed=False)[yc].mean(); se = q.groupby("tert", observed=False)[yc].sem()
            x = np.arange(3) + (k_ - 1.5) * 0.12
            ax.errorbar(x, mu.values, se.values, fmt="o-" if gk[1] == "expert" else "o--", ms=3, lw=0.8, color=m62.COH[gk[0]],
                        mfc="white" if gk[1] == "learning" else m62.COH[gk[0]], label=m62.GLAB[gk], capsize=0)
        ax.set_xticks(range(3), ["low", "mid", "high"]); ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlabel("Decoder accuracy tertile", fontsize=5); ax.set_ylabel(yl, fontsize=5)
        if j == 0:
            ax.legend(frameon=False, fontsize=4.2, loc="upper left")
    m62.letter_row(fig, axs[0::2], "abcd"); m62.letter_row(fig, axs[1::2], "efgh")
    q_ = D.bacc.quantile([1 / 3, 2 / 3]).round(3).tolist()
    fig.suptitle(f"Figure 5—supplement | Decoder quality vs readouts ({RA}, {a.population}; whole brain; tertile bounds {q_[0]}, {q_[1]})",
                 x=0.02, y=0.99, ha="left", va="top", fontsize=6.8, weight="bold")
    m62.save(fig, out, "FigS_decoder_quality"); plt.close(fig)
    C.merge(D[["session_id", "bacc", "bacc_corrected", "transfer_bin_corrected", "transfer_prob_corrected", "num_prob_corrected",
               "lam"]], on="session_id", how="outer").to_csv(out / f"supp_079_{a.population}.csv", index=False)
    pd.DataFrame(m62.STATS).to_csv(out / f"stats_supp_079_{a.population}.csv", index=False)
    print(pd.DataFrame(m62.STATS).reindex(columns=["panel", "measure", "cohort", "rho", "p", "interaction_p_perm",
                                                   "R+ L vs E p_MWU", "R- L vs E p_MWU", "mean_R+ learning", "mean_R+ expert",
                                                   "mean_R− learning", "mean_R− expert"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
