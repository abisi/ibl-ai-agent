"""016 -- Single-event reward-lick CD scores vs the whisker learning curve directly (user 2026-10-16: "what about correlating with whisker
curve directly"). Day-0 sessions with learning curves; events of 008 (both variants); curve = whisker P(lick) (024 HMM, sigma 1) at the last
whisker trial before each event (whisker - FA curve also reported).
Per session and class (WH, SL):
  r          Pearson r(score, curve) over the class's events (plain correlation; includes everything that co-varies with time)
  r_partial  partial r(score, curve | tau): residuals of score and curve after regressing each on session time
  null       the same two statistics with the curve linearly shifted against the time-ordered events (k = 10-50 % of the events, both
             directions, non-wrapping, N_SHIFT shifts); excess = real - mean null
Contrast: r(WH) - r(SL) (whisker hits relative to spontaneous licks measured against the same curve).
Statistics: Wilcoxon vs 0 per cohort (real and excess); R+ vs R- (MWU, Welch).
Output: tables/016_sessions.csv, 016_stats.csv; figures/016_whisker_curve_corr.{png,pdf,svg}
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C

N_SHIFT, LO, HI, MIN_EV = 100, 0.1, 0.5, 8


def resid(y, x):
    X = np.column_stack([np.ones(len(x)), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def two(y, cu, tau):
    if np.std(cu) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r = np.corrcoef(y, cu)[0, 1]
    ry, rc = resid(y, tau), resid(cu, tau)
    rp = np.corrcoef(ry, rc)[0, 1] if np.std(rc) > 0 else np.nan
    return r, rp


def null(y, cu, tau, rng):
    n = len(y); ks = np.arange(max(1, int(LO * n)), max(2, int(HI * n)) + 1); out = []
    for _ in range(N_SHIFT):
        k = int(rng.choice(ks)); s = rng.choice([-1, 1])
        yy, tt, cc = (y[k:], tau[k:], cu[:n - k]) if s > 0 else (y[:n - k], tau[:n - k], cu[k:])
        if len(yy) >= MIN_EV:
            out.append(two(yy, cc, tt))
    return np.array(out)


def session(E, curve):
    rng = np.random.default_rng(zlib.crc32((E.session_id.iloc[0] + curve).encode()))
    out = {}
    for c in ("WH", "SL"):
        d = E[E.cls == c].dropna(subset=["cd", curve]).sort_values("t")
        if len(d) < MIN_EV:
            return None
        y, cu, tau = d.cd.to_numpy(float), d[curve].to_numpy(float), d.tau.to_numpy(float)
        r, rp = two(y, cu, tau); nl = null(y, cu, tau, rng)
        out.update({f"r_{c}": r, f"rp_{c}": rp, f"r_{c}_null": np.nanmean(nl[:, 0]) if len(nl) else np.nan,
                    f"rp_{c}_null": np.nanmean(nl[:, 1]) if len(nl) else np.nan})
        out[f"r_{c}_ex"], out[f"rp_{c}_ex"] = r - out[f"r_{c}_null"], rp - out[f"rp_{c}_null"]
    for k in ("r", "rp", "r_ex", "rp_ex"):
        a, b = k.split("_")[0], ("_ex" if k.endswith("_ex") else "")
        out[f"{a}_rel{b}"] = out[f"{a}_WH{b}"] - out[f"{a}_SL{b}"]
    return out


MEAS = [("r_WH", "r(WH score, curve)"), ("r_WH_ex", "r(WH, curve) − shift null"), ("rp_WH", "partial r | time"),
        ("rp_WH_ex", "partial r | time − null"), ("r_rel", "r(WH) − r(SL)"), ("rp_rel_ex", "partial r(WH) − r(SL), − null")]


def main():
    E0 = pd.read_parquet(C.TAB / "008_events.parquet")
    E0 = C.population(E0[(E0.stage == "learning") & E0.curve_whisker.notna()])
    rows = []
    for v in ("corr", "raw"):
        E = E0.rename(columns={f"cd_{v}": "cd"})
        for curve in ("curve_whisker", "curve_wfa"):
            for sid, g in E.groupby("session_id"):
                o = session(g, curve)
                if o is not None:
                    rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], variant=v, curve=curve, **o))
    S = pd.DataFrame(rows); S.to_csv(C.TAB / "016_sessions.csv", index=False)
    St = []
    for (v, cu), d in S.groupby(["variant", "curve"]):
        for m, _ in MEAS + [("r_SL", ""), ("rp_SL_ex", "")]:
            for c in ("R+", "R-"):
                x = d[d.cohort == c][m].dropna()
                St.append(dict(variant=v, curve=cu, measure=m, cohort=c, n=len(x), mean=x.mean(), sem=x.sem(), p=stats.wilcoxon(x).pvalue))
            a, b = (d[d.cohort == c][m].dropna() for c in ("R+", "R-"))
            St.append(dict(variant=v, curve=cu, measure=m, cohort="R+ vs R-", p=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    St = pd.DataFrame(St); St.to_csv(C.TAB / "016_stats.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 6, figsize=(8.0, 3.8), gridspec_kw=dict(wspace=0.75, hspace=1.0))
    for i, v in enumerate(("corr", "raw")):
        d = S[(S.variant == v) & (S.curve == "curve_whisker")]
        for j, (m, lab) in enumerate(MEAS):
            ax = axs[i, j]
            for k, c in enumerate(("R+", "R-")):
                x = d[d.cohort == c][m].dropna()
                ax.scatter(k + np.random.default_rng(0).uniform(-0.12, 0.12, len(x)), x, s=3, color=C.COH[c], alpha=0.35, lw=0)
                ax.errorbar(k, x.mean(), x.sem(), fmt="o", ms=3.4, color=C.COH[c], mfc="white", capsize=0)
                r = St[(St.variant == v) & (St.curve == "curve_whisker") & (St.measure == m) & (St.cohort == c)].iloc[0]
                ax.text(k, 1.0, C.fp(r.p), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.4, color=C.COH[c])
            rc = St[(St.variant == v) & (St.curve == "curve_whisker") & (St.measure == m) & (St.cohort == "R+ vs R-")].iloc[0]
            ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_xlim(-0.6, 1.6)
            ax.set_title(f"{lab}\nR+ vs R− {C.fp(rc.p)}", fontsize=5.2, pad=10)
            if j == 0:
                ax.set_ylabel(f"{'baseline-corrected' if v == 'corr' else 'raw'}\nper-session r", fontsize=5.4)
            if i == 0:
                ax.text(-0.45, 1.3, "abcdef"[j], transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Day 0: reward-lick CD score of each event vs the whisker learning curve P(lick | whisker) at that trial (dots: sessions; "
                 "p: Wilcoxon vs 0)", fontsize=6.4, x=0.01, y=1.04, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"016_whisker_curve_corr.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(St[St.curve == "curve_whisker"].round(4).to_string())


if __name__ == "__main__":
    main()
