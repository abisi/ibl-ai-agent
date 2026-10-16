"""005 -- Relative position of whisker hits between spontaneous licks and auditory hits over the session (lambda), and the auditory-hit drift,
day 0 vs expert (user 2026-10-16: improve the day 0 vs expert comparison; compare with auditory hits).
Data: ssl-prelick-convergence within_day 002 per-event scores on a fixed session CD (5-fold CV over AH + SL events, unit-norm AH - SL axis;
held-out AH / SL and fold-averaged WH projections; one session normalisation: held-out SL mean = 0, AH mean = 1; sessions with held-out
AH-vs-SL d' >= 0.3). Normalised session time tau (0 = first, 1 = last kept event).
Per session and class (WH, AH, SL; >= MIN_EV events): OLS fit score = a + b tau. Quantities:
  b_WH, b_AH, b_SL                      within-session slopes (score units per session; 1 = the full SL -> AH distance)
  b_WH-SL, b_WH-AH, b_AH-SL             slope contrasts: drift of one class relative to another measured at the same times (shared drift cancels)
  lambda(tau) = (f_WH - f_SL) / (f_AH - f_SL) at tau = 0 and 1 from the fits (lam_start, lam_end), dlam = lam_end - lam_start
                (0 = like spontaneous licks, 1 = like auditory hits; invariant to drift shared by all classes and to changes of the AH - SL
                separation); sessions with fitted AH - SL separation >= SEP_MIN at both ends
  lam_mean                              lambda from the session means (one value per session; the level reached)
Display: per-third class means and lambda (thirds with >= MIN_BIN AH and SL events), mean +- s.e.m. over sessions.
Day-0 start / end vs the expert level: lam_start(day 0) vs lam_mean(expert) and lam_end(day 0) vs lam_mean(expert) (Mann-Whitney).
Statistics: common.measure_stats (Wilcoxon vs 0 or vs the null value, day 0 vs expert, R+ vs R-, cohort x stage permutation).
Output: tables/005_sessions.csv, 005_stats.csv, 005_thirds.csv; figures/005_lambda_auditory.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

MIN_EV, MIN_BIN, SEP_MIN = 4, 2, 0.3
MEAS = [("b_WH", "WH slope"), ("b_AH", "AH slope"), ("b_SL", "SL slope"), ("b_WH-SL", "WH − SL slope"), ("b_WH-AH", "WH − AH slope"),
        ("b_AH-SL", "AH − SL slope"), ("lam_start", "λ at session start"), ("lam_end", "λ at session end"), ("dlam", "Δλ (end − start)"),
        ("lam_mean", "λ (session mean)")]


def fit(d):
    if len(d) < MIN_EV:
        return np.nan, np.nan
    b, a = np.polyfit(d.tau.to_numpy(), d.cd.to_numpy(), 1)
    return a, b


def session(sid, E):
    out = dict(session_id=sid)
    F = {}
    for c in ("WH", "AH", "SL"):
        a, b = fit(E[E.cls == c]); F[c] = (a, b); out[f"b_{c}"] = b; out[f"n_{c}"] = int((E.cls == c).sum())
    for x, y in (("WH", "SL"), ("WH", "AH"), ("AH", "SL")):
        out[f"b_{x}-{y}"] = F[x][1] - F[y][1]
    lam = {}
    for t in (0.0, 1.0):
        f = {c: F[c][0] + F[c][1] * t for c in F}
        sep = f["AH"] - f["SL"]
        lam[t] = (f["WH"] - f["SL"]) / sep if np.isfinite(sep) and sep >= SEP_MIN else np.nan
    out.update(lam_start=lam[0.0], lam_end=lam[1.0], dlam=lam[1.0] - lam[0.0])
    m = E.groupby("cls").cd.mean()
    if all(c in m for c in ("WH", "AH", "SL")) and m["AH"] - m["SL"] >= SEP_MIN:
        out["lam_mean"] = (m["WH"] - m["SL"]) / (m["AH"] - m["SL"])
    rows = []
    E = E.assign(third=np.minimum((E.tau * 3).astype(int), 2))
    for k, g in E.groupby("third"):
        mm = g.groupby("cls").cd.agg(["mean", "size"])
        r = dict(session_id=sid, third=int(k), **{f"m_{c}": mm.loc[c, "mean"] if c in mm.index else np.nan for c in ("WH", "AH", "SL")})
        ok = all(c in mm.index and mm.loc[c, "size"] >= MIN_BIN for c in ("AH", "SL")) and "WH" in mm.index
        sep = r["m_AH"] - r["m_SL"]
        r["lam"] = (r["m_WH"] - r["m_SL"]) / sep if ok and sep >= SEP_MIN else np.nan
        rows.append(r)
    return out, rows


def figure(S, Th, R, X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = fig.add_gridspec(3, 4, wspace=0.6, hspace=1.2)
    CL = {"WH": "#f7b519", "AH": "#2c2cdb", "SL": "#262626"}
    # a: class means per third (relative to SL of the first third? absolute score), per cohort x stage
    for i, c in enumerate(("R+", "R-")):
        for j, s_ in enumerate(("learning", "expert")):
            ax = fig.add_subplot(gs[0, 2 * i + j])
            d = Th[(Th.cohort == c) & (Th.stage == s_)]
            for cl in ("SL", "WH", "AH"):
                piv = d.pivot_table(index="session_id", columns="third", values=f"m_{cl}")
                ax.errorbar(np.arange(1, 4), piv.mean(), piv.sem(), color=CL[cl], marker="o", ms=2.5, lw=0.9, capsize=0, label=cl)
            ax.set_xticks([1, 2, 3]); ax.set_xlabel("third of the session"); ax.axhline(0, color="0.8", lw=0.4); ax.axhline(1, color="0.8", lw=0.4)
            ax.set_title(f"{C.GLAB[(c, s_)]} ({d.session_id.nunique()})", color=C.COH[c], fontsize=6)
            if i == 0 and j == 0:
                ax.set_ylabel("CD score\n(held-out SL = 0, AH = 1)"); ax.legend(frameon=False, fontsize=4.6, loc="upper left")
            if i == 0 and j == 0:
                ax.text(-0.45, 1.25, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    # b: lambda per third, day 0 vs expert, per cohort
    for i, c in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[1, i])
        for s_, ls, mfc in (("learning", (0, (3, 1.5)), "white"), ("expert", "-", None)):
            piv = Th[(Th.cohort == c) & (Th.stage == s_)].pivot_table(index="session_id", columns="third", values="lam")
            ax.errorbar(np.arange(1, 4) + (0.05 if s_ == "expert" else -0.05), piv.mean(), piv.sem(), color=C.COH[c], ls=ls, marker="o", ms=2.8,
                        mfc=mfc or C.COH[c], lw=0.9, capsize=0, label=f"{'day 0' if s_ == 'learning' else 'expert'} ({piv.notna().any(axis=1).sum()})")
        ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color="0.7", lw=0.5, ls=":"); ax.set_xticks([1, 2, 3])
        ax.set_xlabel("third of the session"); ax.set_ylabel("λ (0 = SL, 1 = AH)"); ax.legend(frameon=False, fontsize=4.6)
        ax.set_title(f"{c.replace('-', '−')}: λ per third", color=C.COH[c], fontsize=6)
        if i == 0:
            ax.text(-0.45, 1.2, "b", transform=ax.transAxes, fontsize=8, weight="bold")
    # c: start / end vs expert level
    ax = fig.add_subplot(gs[1, 2:])
    for i, c in enumerate(("R+", "R-")):
        d0 = S[(S.cohort == c) & (S.stage == "learning")]; e = S[(S.cohort == c) & (S.stage == "expert")]
        for k, (v, lab, mfc) in enumerate(((d0.lam_start, "D0 start", "white"), (d0.lam_end, "D0 end", "white"),
                                            (e.lam_start, "E start", None), (e.lam_end, "E end", None))):
            v = v.dropna(); x = i * 5 + k
            ax.scatter(x + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=3, color=C.COH[c], alpha=0.3, lw=0)
            ax.errorbar(x, v.mean(), v.sem(), fmt="o", ms=3.5, color=C.COH[c], mfc=mfc or C.COH[c], capsize=0)
        xx = X[X.cohort == c].set_index("test").p_mwu
        ax.text(i * 5 + 1.5, 1.0, f"D0 start vs E: {C.fp(xx['day-0 start vs expert mean'])}\nD0 end vs E: {C.fp(xx['day-0 end vs expert mean'])}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.4, color=C.COH[c])
    ax.set_xticks([0, 1, 2, 3, 5, 6, 7, 8], ["D0\nstart", "D0\nend", "E\nstart", "E\nend"] * 2, fontsize=4.8)
    ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color="0.7", lw=0.5, ls=":"); ax.set_ylabel("λ from the linear fits")
    ax.set_title("λ at session start and end (R+ left, R− right)", fontsize=6, pad=20)
    ax.text(-0.18, 1.25, "c", transform=ax.transAxes, fontsize=8, weight="bold")
    # d: slopes and contrasts, day 0 vs expert
    for k, (m, l) in enumerate((("b_AH", "d"), ("b_WH-SL", "e"), ("b_WH-AH", "f"), ("dlam", "g"))):
        ax = fig.add_subplot(gs[2, k]); Rm = R[R.measure == m]
        C.dot_panel(ax, S, m, Rm, dict(MEAS)[m] + (" (score / session)" if m.startswith("b_") else ""), l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle("Reward-lick CD: relative position λ of whisker hits between spontaneous licks and auditory hits, and auditory-hit drift "
                 "(day 0 vs expert; green R+, magenta R−)", fontsize=6.4, x=0.01, y=0.995, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"005_lambda_auditory.{ext}", dpi=300, bbox_inches="tight")


def main():
    T = pd.read_parquet(C.m51.WITHIN / "slopes" / "trial_scores.parquet")
    T["cls"] = T.cls.replace({"FA": "SL"})
    res = [session(sid, E) for sid, E in T.groupby("session_id")]
    meta = T.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    S = C.population(pd.DataFrame([r for r, _ in res]).merge(meta, on="session_id"))
    Th = pd.DataFrame([x for _, rr in res for x in rr]).merge(meta, on="session_id"); Th = Th[Th.session_id.isin(S.session_id)]
    S.to_csv(C.TAB / "005_sessions.csv", index=False); Th.to_csv(C.TAB / "005_thirds.csv", index=False)
    R = pd.DataFrame([r for m, _ in MEAS for r in C.measure_stats(S, m, "whole brain", m)])
    # lambda levels are tested against 0 by measure_stats; for level measures the relevant contrasts are the stage comparisons
    X = []
    for c in ("R+", "R-"):
        e = S[(S.cohort == c) & (S.stage == "expert")].lam_mean.dropna()
        for q, lab in (("lam_start", "day-0 start vs expert mean"), ("lam_end", "day-0 end vs expert mean")):
            d0 = S[(S.cohort == c) & (S.stage == "learning")][q].dropna()
            X.append(dict(cohort=c, test=lab, n_d0=len(d0), n_e=len(e), mean_d0=d0.mean(), mean_e=e.mean(),
                          p_mwu=stats.mannwhitneyu(d0, e).pvalue, p_welch=stats.ttest_ind(d0, e, equal_var=False).pvalue))
    X = pd.DataFrame(X)
    R.to_csv(C.TAB / "005_stats.csv", index=False); X.to_csv(C.TAB / "005_levels.csv", index=False)
    figure(S, Th, R, X)
    pd.set_option("display.width", 250)
    print(R[["measure", "cohort", "n_d0", "n_e", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(4).to_string())
    print(X.round(4).to_string())


if __name__ == "__main__":
    main()
