"""078 -- Supplementary figures (user review 2026-10-07): linear-shift geometry, coding-direction estimators, quiet windows,
session quantities vs performance. Population all mice or learners (061 rule).

FigS_shift_geometry: delta d, along (cross-validated dot product per unit) and lambda per session, observed (top) and
  linear-shift corrected (bottom; 076: observed - median over 40 shifted relabellings; lambda numerator and denominator
  corrected separately).
FigS_cd_estimators: (a) lambda vs the mean WH score of the unified split-half trial scores (coding_direction; equal by
  construction), (b) lambda vs the Part II 5-fold coding direction (mean WH score; sessions with held-out d' >= 0.3),
  (c) the 5-fold coding direction per cohort x stage.
FigS_quiet_windows: quiet windows [-200, -100] ms before every trial start of the analysed epoch (077), raw rates,
  lick-free windows only (no lick in the preceding 1 s), placed on the same readouts as the events: units scaled with the
  event parameters (057 pooled SD over the events); (a) cross-validated d(Q, ref), d(Q, AH), d(Q, WH) per unit (split
  halves of quiet windows and events, 50 splits); (b) lambda_Q: position of the quiet mean on the ref -> AH axis
  (cross-validated, as lambda with Q in place of WH), with lambda_WH of the same computation; (c) lambda_LDA,Q:
  shrinkage-LDA axis (AH vs ref, 5-fold, units z-scored on the training folds), quiet windows projected and normalised
  (held-out ref = 0, AH = 1). Events are baseline-subtracted pre-lick rates, quiet windows raw rates (user choice).
FigS_performance: per-session quantities vs whisker hit rate, false-alarm rate and d' (077), one fit per cohort (OLS +
  95 % CI band, solid only if Spearman p < 0.05), learning (open) and expert (filled) sessions.
Output: across_days/<ref>/publication/<pop>/FigS_*.{png,pdf,svg}, supp_078_<pop>.csv
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
m57 = importlib.import_module("057_roc_prelick_lambda")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
EX = m51.OUTROOT / "extras"


# ------------------------------------------------------------------ quiet windows
def cv_dot(Z, ia, ib, ic, id_, rng, n=50):
    """cross-validated (mean_a - mean_b) . (mean_c - mean_d) per unit over random split halves of each index set"""
    out = []
    for _ in range(n):
        h = [rng.permutation(v) for v in (ia, ib, ic, id_)]
        s = [(x[: len(x) // 2], x[len(x) // 2:]) for x in h]
        m = lambda k, j: Z[:, s[k][j]].mean(1)
        out.append(0.5 * ((m(0, 0) - m(1, 0)) @ (m(2, 1) - m(3, 1)) + (m(0, 1) - m(1, 1)) @ (m(2, 0) - m(3, 0))))
    return float(np.mean(out)) / Z.shape[0]


def quiet_session(r, W):
    f_ev = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
    f_q = r.file.parent / f"{r.mouse}_quiet_windows.npz"
    if not (f_ev.exists() and f_q.exists()):
        return None
    z, q = np.load(f_ev, allow_pickle=True), np.load(f_q, allow_pickle=True)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "quality_label", "cohort", "stage", "mouse_id"]],
                on=["electrode_group", "cluster_id"], how="left")
    if K.cohort.isna().all():
        return None
    Kq = pd.DataFrame(dict(electrode_group=q["electrode_group"].astype(str), cluster_id=q["cluster_id"].astype(str)))
    Kq["qi"] = np.arange(len(Kq))
    K = K.merge(Kq, on=["electrode_group", "cluster_id"], how="left")
    X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
    Q = q["rates"].astype(float)[:, q["lick_free"].astype(bool)]
    if min((lab == c).sum() for c in m51.CLASSES) < 4 or Q.shape[1] < 8:
        return None
    ok = K.quality_label.isin(m57.UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR) & K.qi.notna().to_numpy()
    mu, sd = X.mean(1), X.std(1)
    ok &= sd > 0
    if ok.sum() < m57.MIN_UNITS:
        return None
    Ze = (X[ok] - mu[ok, None]) / sd[ok, None]
    Zq = (Q[K.qi[ok].astype(int).to_numpy()] - mu[ok, None]) / sd[ok, None]
    Z = np.c_[Ze, Zq]
    ne = Ze.shape[1]
    I = {c: np.where(lab == c)[0] for c in m51.CLASSES}; I["Q"] = ne + np.arange(Zq.shape[1])
    rng = np.random.default_rng(0)
    row = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
               stage=K.stage.dropna().iloc[0], n_units=int(ok.sum()), n_quiet=int(Zq.shape[1]))
    for c in ("FA", "AH", "WH"):
        row[f"d_Q_{c}"] = cv_dot(Z, I["Q"], I[c], I["Q"], I[c], rng)
    den = cv_dot(Z, I["AH"], I["FA"], I["AH"], I["FA"], rng)
    row["d_AH_FA"] = den
    row["lam_Q"] = cv_dot(Z, I["Q"], I["FA"], I["AH"], I["FA"], rng) / den if den > 0 else np.nan
    row["lam_WH"] = cv_dot(Z, I["WH"], I["FA"], I["AH"], I["FA"], rng) / den if den > 0 else np.nan
    # LDA axis
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    ia = np.r_[I["AH"], I["FA"]]; y = (np.isin(ia, I["AH"])).astype(int)
    E = np.c_[X[ok], Q[K.qi[ok].astype(int).to_numpy()]].T            # events x units, raw scale (z-scored per fold)
    pa, pf, pq, pw = [], [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(ia, y):
        A = E[ia[tr]]; m_, s_ = A.mean(0), A.std(0); s_[s_ == 0] = 1
        zf = lambda v: (v - m_) / s_
        w = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(zf(A), y[tr]).coef_[0]
        p = lambda idx: zf(E[idx]) @ w
        pa.append(p(ia[te][y[te] == 1])); pf.append(p(ia[te][y[te] == 0])); pq.append(p(I["Q"]).mean()); pw.append(p(I["WH"]).mean())
    a_, f_ = np.concatenate(pa).mean(), np.concatenate(pf).mean()
    row["lam_lda_Q"] = (np.mean(pq) - f_) / (a_ - f_) if a_ != f_ else np.nan
    row["lam_lda_WH"] = (np.mean(pw) - f_) / (a_ - f_) if a_ != f_ else np.nan
    return row


# ------------------------------------------------------------------ correlation panel (scatter + OLS + 95 % CI)
def corr_panel(ax, d, x, y, rng):
    from scipy.stats import t as tdist
    txt = []
    for coh in ("R+", "R-"):
        q = d[(d.cohort == coh)][[x, y, "stage"]].dropna()
        for st_, fc in (("learning", "white"), ("expert", m62.COH[coh])):
            g = q[q.stage == st_]
            ax.scatter(g[x], g[y], s=6, facecolor=fc, edgecolor=m62.COH[coh], lw=0.5, zorder=3)
        if len(q) < 5:
            continue
        rho, p = stats.spearmanr(q[x], q[y])
        b = np.polyfit(q[x], q[y], 1); xs = np.linspace(q[x].min(), q[x].max(), 50)
        res = q[y] - np.polyval(b, q[x]); s2 = (res ** 2).sum() / (len(q) - 2)
        se = np.sqrt(s2 * (1 / len(q) + (xs - q[x].mean()) ** 2 / ((q[x] - q[x].mean()) ** 2).sum()))
        tq = tdist.ppf(0.975, len(q) - 2)
        ax.fill_between(xs, np.polyval(b, xs) - tq * se, np.polyval(b, xs) + tq * se, color=m62.COH[coh], alpha=0.12, lw=0)
        ax.plot(xs, np.polyval(b, xs), color=m62.COH[coh], lw=0.9, ls="-" if p < 0.05 else (0, (2, 2)))
        txt.append(f"{coh.replace('-', '−')}: ρ = {rho:.2f}, {m62.fmt_p(p)}, n = {len(q)}")
        m62.STATS.append(dict(panel="S-perf", measure=f"{y} vs {x}", cohort=coh, rho=rho, p=p, n=len(q)))
    ax.set_title("\n".join(txt), loc="left", fontsize=4.0, pad=2)          # stats above the axes, never over data


def main(a):
    plt = m62.setup()
    RA = "FA" if m51.REF == "fa" else "SL"
    out = m51.OUTROOT / "publication" / a.population
    keep = m61.learner_filter if a.population == "learners" else (lambda df: df)
    rng = np.random.default_rng(0)
    X = keep(pd.read_csv(EX / "extras_sessions_pooled.csv"))
    # --- shift geometry
    fig = plt.figure(figsize=(m62.W_IN, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.6, left=0.08, right=0.98, top=0.88, bottom=0.07)
    spec = [("dd", f"d(WH,{RA}) − d(WH,AH)", "Distance difference Δd"), ("along", "Cross-validated dot product\nper unit", f"Along {RA} → AH"),
            ("lam", "λ", "λ")]
    axs = []
    for j, (col, yl, ttl) in enumerate(spec):
        for i, (suf, lab) in enumerate((("", "observed"), ("_corrected", "linear-shift corrected"))):
            ax = fig.add_subplot(gs[i, j]); axs.append(ax)
            m62.dots_panel(ax, X, col + suf, yl, rng, f"S-shift {col}{suf}", f"{ttl}, {lab}", ref=[(0, "0.6")])
            ax.set_title(ax.get_title(), fontsize=5.4)
    m62.letter_row(fig, axs[0::2], "abc"); m62.letter_row(fig, axs[1::2], "def")
    fig.suptitle(f"Figure 3—supplement | Population geometry, observed vs linear-shift corrected ({RA}, {a.population})",
                 x=0.02, y=0.99, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "FigS_shift_geometry"); plt.close(fig)
    # --- coding-direction estimators
    fig = plt.figure(figsize=(m62.W_IN, 2.8))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.08, right=0.98, top=0.78, bottom=0.18)
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax, ycol, yl in ((axs[0], "lam_split_trials", "Unified split-half: mean WH trial score"),
                         (axs[1], "cd_kfold_wh", "Part II 5-fold CD: mean WH score")):
        for k in m62.GROUPS:
            q = X[(X.cohort == k[0]) & (X.stage == k[1])]
            ax.scatter(q.lam, q[ycol], s=7, facecolor="white" if k[1] == "learning" else m62.COH[k[0]],
                       edgecolor=m62.COH[k[0]], lw=0.5, label=m62.GLAB[k])
        qq = X[["lam", ycol]].dropna(); rho = stats.spearmanr(qq.lam, qq[ycol])[0]
        lim = (-1, 2); ax.plot(lim, lim, color="0.6", lw=0.5, ls=(0, (2, 2))); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("λ (split-half, Part I)"); ax.set_ylabel(yl, fontsize=5)
        ax.set_title(f"ρ = {rho:.3f}, n = {len(qq)} sessions", fontsize=5.6)
    axs[0].legend(frameon=False, fontsize=4.2, loc="upper left")
    m62.dots_panel(axs[2], X, "cd_kfold_wh", "Mean WH score (ref = 0, AH = 1)", rng, "S-cd kfold", "Part II 5-fold CD,\nPart I sessions",
                   ref=[(0, "0.6"), (1, "0.8")])
    axs[2].set_title(axs[2].get_title(), fontsize=5.4)
    m62.letter_row(fig, axs, "abc")
    fig.suptitle(f"Figure 3—supplement | Coding-direction estimators ({RA}, {a.population}; whole brain)", x=0.02, y=0.99,
                 ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "FigS_cd_estimators"); plt.close(fig)
    # --- quiet windows
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    fq = EX / "quiet_sessions.csv"
    if fq.exists() and not a.recompute:
        Qs = pd.read_csv(fq)
    else:
        Qs = pd.DataFrame([r for r in (quiet_session(r, W) for r in ss.itertuples()) if r]); Qs.to_csv(fq, index=False)
    Qs = keep(Qs)
    fig = plt.figure(figsize=(m62.W_IN, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.6, left=0.08, right=0.98, top=0.88, bottom=0.07)
    spec = [("d_Q_FA", "Squared distance per unit", f"d(quiet, {RA})"), ("d_Q_AH", "Squared distance per unit", "d(quiet, AH)"),
            ("d_Q_WH", "Squared distance per unit", "d(quiet, WH)"), ("lam_Q", "λ (ref = 0, AH = 1)", "Quiet windows on the\nmean-difference axis"),
            ("lam_WH", "λ (ref = 0, AH = 1)", "Whisker hits, same computation"), ("lam_lda_Q", "λ_LDA (ref = 0, AH = 1)", "Quiet windows on the\nshrinkage-LDA axis")]
    axs = []
    for k_, (col, yl, ttl) in enumerate(spec):
        ax = fig.add_subplot(gs[k_ // 3, k_ % 3]); axs.append(ax)
        m62.dots_panel(ax, Qs, col, yl, rng, f"S-quiet {col}", ttl, ref=[(0, "0.6")] + ([(1, "0.8")] if "lam" in col else []))
        ax.set_title(ax.get_title(), fontsize=5.4)
    m62.letter_row(fig, axs[:3], "abc"); m62.letter_row(fig, axs[3:], "def")
    fig.suptitle(f"Figure 3—supplement | Quiet windows (−200 to −100 ms before trial start, lick-free, raw rates) on the "
                 f"population readouts ({RA}, {a.population})", x=0.02, y=0.99, ha="left", va="top", fontsize=6.6, weight="bold")
    m62.save(fig, out, "FigS_quiet_windows"); plt.close(fig)
    # --- performance
    P = pd.read_csv(EX / "performance_sessions.csv")[["session_id", "rate_wh", "rate_fa", "dprime_wh_fa"]]
    D = m62.load_population(a.population)
    S = m62.session_unit_metrics(D["W"])[["session_id", "frac_sig_WHvsFA", "r_shared"]]
    T = D["tr"]; T = T[(T.level == "all") & (T.variant == "all")][["session_id", "frac_transfer", "likeness"]]
    DC = D["dec"]; DC = DC[DC.level == "all"][["session_id", "bacc_corrected", "transfer_prob_corrected", "num_prob_corrected"]]
    M = X[["session_id", "mouse_id", "cohort", "stage", "dd", "lam"]].merge(S, on="session_id", how="outer") \
        .merge(T, on="session_id", how="outer").merge(DC, on="session_id", how="outer").merge(P, on="session_id", how="left")
    meta = pd.concat([X[["session_id", "cohort", "stage"]], D["W"][["session_id", "cohort", "stage"]]]).drop_duplicates("session_id")
    M = M.drop(columns=["cohort", "stage"]).merge(meta, on="session_id", how="left")
    M = keep(M.assign(mouse_id=M.mouse_id.fillna(M.session_id.str.split("_").str[0])))
    Y = [("frac_sig_WHvsFA", f"WH vs {RA} selective"), ("frac_transfer", "Converging neurons"), ("likeness", "AH-likeness"),
         ("r_shared", "Shared hit code r"), ("dd", "Δd"), ("lam", "λ"), ("bacc_corrected", "Decoder accuracy − chance"),
         ("num_prob_corrected", f"P(AH|WH) − P(AH|{RA}) − chance")]
    Xp = [("rate_wh", "Whisker hit rate"), ("rate_fa", "False-alarm rate (no-stim trials)"),
          ("dprime_wh_fa", "d′: whisker hit vs no-stim lick rate")]
    fig = plt.figure(figsize=(m62.W_IN, 1.45 * len(Y) + 0.6))
    gs = fig.add_gridspec(len(Y), len(Xp), hspace=1.0, wspace=0.45, left=0.12, right=0.98, top=1 - 0.5 / (1.45 * len(Y) + 0.6), bottom=0.04)
    for i, (yc, yl) in enumerate(Y):
        for j, (xc, xl) in enumerate(Xp):
            ax = fig.add_subplot(gs[i, j]); corr_panel(ax, M, xc, yc, rng)
            if j == 0:
                ax.set_ylabel(yl, fontsize=5)
            if i == len(Y) - 1:
                ax.set_xlabel(xl, fontsize=5.4)
            ax.tick_params(labelsize=4.5)
    fig.suptitle(f"Supplement | Session quantities vs behavioural performance ({RA}, {a.population}; open: learning, filled: "
                 "expert; one fit per cohort, solid if p < 0.05)", x=0.02, y=0.995, ha="left", va="top", fontsize=6.5, weight="bold")
    m62.save(fig, out, "FigS_performance"); plt.close(fig)
    M.to_csv(out / f"supp_078_{a.population}.csv", index=False)
    pd.DataFrame(m62.STATS).to_csv(out / f"stats_supp_078_{a.population}.csv", index=False)
    print("ALL DONE", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    ap.add_argument("--recompute", action="store_true")
    main(ap.parse_args())
