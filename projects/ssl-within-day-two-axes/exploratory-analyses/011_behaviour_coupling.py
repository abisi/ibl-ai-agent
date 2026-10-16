"""011 -- Test 2: does the whisker-hit position on the reward-lick CD follow the behavioural learning curve beyond session time on day 0?
(user 2026-10-16). Events: 008 (both variants), day-0 sessions with learning curves (024, sigma 1; whisker - FA curve at the last whisker
trial before each event).
Per session and class (WH, SL): OLS score = a + b_tau tau_z + b_curve curve_z (predictors z-scored within the session, so b are per SD);
b_curve is the coupling to behaviour with session time held fixed; relative coupling b_curve(WH) - b_curve(SL). Collinearity: r(tau, curve)
over the session's events. Null: the curve values linearly shifted against the time-ordered events (k = 10-50 % of the events, both
directions, non-wrapping; N_SHIFT shifts), b_curve refitted; excess = real - mean null.
Statistics: Wilcoxon of the excess vs 0 per cohort; R+ vs R- (MWU, Welch); by variant.
Output: tables/011_sessions.csv, 011_stats.csv; figures/011_behaviour_coupling_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C

N_SHIFT, LO, HI, MIN_EV = 100, 0.1, 0.5, 8


def coef(y, tau, cu):
    X = np.column_stack([np.ones(len(y)), (tau - tau.mean()) / (tau.std() or 1), (cu - cu.mean()) / (cu.std() or 1)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1], b[2]


def shifted(y, tau, cu, rng):
    n = len(y); out = []
    ks = np.arange(max(1, int(LO * n)), max(2, int(HI * n)) + 1)
    for _ in range(N_SHIFT):
        k = int(rng.choice(ks)); s = rng.choice([-1, 1])
        if s > 0:
            yy, tt, cc = y[k:], tau[k:], cu[:n - k]
        else:
            yy, tt, cc = y[:n - k], tau[:n - k], cu[k:]
        if len(yy) >= MIN_EV and np.std(cc) > 0:
            out.append(coef(yy, tt, cc)[1])
    return np.array(out)


def session(E):
    rng = np.random.default_rng(zlib.crc32(E.session_id.iloc[0].encode()))
    out = {}
    for c in ("WH", "SL"):
        d = E[(E.cls == c)].dropna(subset=["cd", "curve_wfa"]).sort_values("t")
        if len(d) < MIN_EV or d.curve_wfa.std() == 0:
            return None
        y, tau, cu = d.cd.to_numpy(float), d.tau.to_numpy(float), d.curve_wfa.to_numpy(float)
        bt, bc = coef(y, tau, cu); nl = shifted(y, tau, cu, rng)
        out.update({f"btau_{c}": bt, f"bcurve_{c}": bc, f"null_{c}": nl.mean() if len(nl) else np.nan,
                    f"excess_{c}": bc - nl.mean() if len(nl) else np.nan, f"r_tau_curve_{c}": np.corrcoef(tau, cu)[0, 1]})
    out["bcurve_rel"] = out["bcurve_WH"] - out["bcurve_SL"]; out["excess_rel"] = out["excess_WH"] - out["excess_SL"]
    return out


def run(variant):
    E = pd.read_parquet(C.TAB / "008_events.parquet").rename(columns={f"cd_{variant}": "cd"})
    E = C.population(E[(E.stage == "learning") & E.curve_wfa.notna()])
    rows = []
    for sid, g in E.groupby("session_id"):
        o = session(g)
        if o is not None:
            rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], variant=variant, **o))
    S = pd.DataFrame(rows); St = []
    for m in ("excess_WH", "excess_SL", "excess_rel", "btau_WH", "bcurve_WH", "r_tau_curve_WH"):
        for c in ("R+", "R-"):
            v = S[S.cohort == c][m].dropna()
            St.append(dict(variant=variant, measure=m, cohort=c, n=len(v), mean=v.mean(), sem=v.sem(), p=stats.wilcoxon(v).pvalue))
        a, b = (S[S.cohort == c][m].dropna() for c in ("R+", "R-"))
        St.append(dict(variant=variant, measure=m, cohort="R+ vs R-", p=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return S, pd.DataFrame(St), E


def figure(S, St, E, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8.0, 4.8)); gs = fig.add_gridspec(2, 4, wspace=0.95, hspace=0.95, width_ratios=[1.25, 1, 1, 1])
    # a: example sessions (closest to the cohort median relative excess): WH scores and curve over whisker trials
    for i, c in enumerate(("R+", "R-")):
        d = S[S.cohort == c]; sid = d.iloc[(d.excess_rel - d.excess_rel.median()).abs().argsort().iloc[0]].session_id
        g = E[(E.session_id == sid)].sort_values("t")
        ax = fig.add_subplot(gs[i, 0]); w = g[g.cls == "WH"]; s = g[g.cls == "SL"]
        ax.scatter(s.wtrial, s.cd, s=2, color="0.5", alpha=0.4, lw=0, label="SL"); ax.scatter(w.wtrial, w.cd, s=4, color="#f7b519", lw=0, label="WH")
        ax2 = ax.twinx(); ax2.plot(g.wtrial, g.curve_wfa, color=C.COH[c], lw=1); ax2.set_ylabel("whisker − FA curve", color=C.COH[c], fontsize=5, labelpad=1); ax2.tick_params(labelsize=4.4)
        ax2.spines["top"].set_visible(False)
        ax.set_xlabel("whisker trial"); ax.set_ylabel("CD score"); ax.set_title(f"{c.replace('-', '−')} example {sid[:5]}", color=C.COH[c], fontsize=6)
        if i == 0:
            ax.legend(frameon=False, fontsize=4.6, loc="upper left"); ax.text(-0.45, 1.15, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    # b: tau-curve collinearity; c: b_curve real vs null (WH); d: excess WH / SL / rel per cohort; e: b_tau with curve in the model
    ax = fig.add_subplot(gs[0, 1])
    for c in ("R+", "R-"):
        ax.hist(S[S.cohort == c].r_tau_curve_WH.dropna(), bins=np.linspace(-1, 1, 21), color=C.COH[c], alpha=0.5, label=c.replace("-", "−"))
    ax.set_xlabel("r(session time, learning curve)"); ax.set_title("collinearity of time and behaviour", fontsize=5.6); ax.legend(frameon=False, fontsize=4.6)
    ax.text(-0.35, 1.15, "b", transform=ax.transAxes, fontsize=8, weight="bold")
    ax = fig.add_subplot(gs[0, 2])
    for j, c in enumerate(("R+", "R-")):
        d = S[S.cohort == c]
        for _, r in d.iterrows():
            ax.plot([j * 2.4, j * 2.4 + 1], [r.null_WH, r.bcurve_WH], color=C.COH[c], lw=0.4, alpha=0.4)
        ax.errorbar([j * 2.4, j * 2.4 + 1], [d.null_WH.mean(), d.bcurve_WH.mean()], [d.null_WH.sem(), d.bcurve_WH.sem()], color=C.COH[c], marker="o", ms=3, capsize=0)
    ax.set_xticks([0, 1, 2.4, 3.4], ["null", "real", "null", "real"]); ax.axhline(0, color="0.8", lw=0.4)
    ax.set_ylabel("WH coupling to the curve (score / SD)"); ax.set_title("coupling beyond time, real vs shifted curve", fontsize=5.6)
    ax.text(-0.4, 1.15, "c", transform=ax.transAxes, fontsize=8, weight="bold")
    for k, (m, lab, l) in enumerate((("excess_WH", "WH", "d"), ("excess_rel", "WH − SL", "e"), ("btau_WH", "time slope with curve", "f"))):
        ax = fig.add_subplot(gs[1, 1 + k] if k < 3 else gs[0, 3])
        for j, c in enumerate(("R+", "R-")):
            v = S[S.cohort == c][m].dropna()
            ax.scatter(j + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=3, color=C.COH[c], alpha=0.35, lw=0)
            ax.errorbar(j, v.mean(), v.sem(), fmt="o", ms=3.5, color=C.COH[c], mfc="white", capsize=0)
            r = St[(St.measure == m) & (St.cohort == c)].iloc[0]
            ax.text(j, 1.0, C.fp(r.p), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.6, color=C.COH[c])
        rc = St[(St.measure == m) & (St.cohort == "R+ vs R-")].iloc[0]
        ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{lab}\nR+ vs R− {C.fp(rc.p)}", fontsize=5.6, pad=10); ax.text(-0.4, 1.2, l, transform=ax.transAxes, fontsize=8, weight="bold")
        ax.set_ylabel("excess coupling" if k < 2 else "b_tau (score / SD)")
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    ax.text(0, 0.9, "score = a + b_τ τ + b_curve curve\n(per session, predictors z-scored)\n\nnull: curve linearly shifted\nagainst the events (100 shifts)\n\n"
            "excess = b_curve − mean null", fontsize=5, va="top")
    fig.suptitle(f"Test 2: day-0 coupling of the reward-lick CD score to the learning curve beyond session time "
                 f"({'baseline-corrected' if variant == 'corr' else 'raw'} rates)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"011_behaviour_coupling_{variant}.{ext}", dpi=300, bbox_inches="tight")


def main():
    out = {v: run(v) for v in ("corr", "raw")}
    pd.concat([o[0] for o in out.values()]).to_csv(C.TAB / "011_sessions.csv", index=False)
    St = pd.concat([o[1] for o in out.values()]); St.to_csv(C.TAB / "011_stats.csv", index=False)
    for v, (S, st, E) in out.items():
        figure(S, st, E, v)
    pd.set_option("display.width", 220)
    print(St.round(4).to_string())


if __name__ == "__main__":
    main()
