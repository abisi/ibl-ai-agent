"""012 -- Test 3: are day-0 whisker hits a mixture of auditory-hit-like and spontaneous-lick-like events whose auditory-like share rises
within the session, while expert whisker hits form one population? (user 2026-10-16).
Events: 008 (both variants). Per session, the reference distributions are the session's held-out SL and AH scores, each a Gaussian with its
own mean and SD (SL ~ 0, AH ~ 1 by construction). Whisker hits are modelled as a mixture with these two FIXED components; the mixing weight
pi (share of AH-like whisker hits) is fitted by EM separately in the early and late half of the session (tau median split) and over the whole
session; dpi = pi_late - pi_early. Per event: posterior P(AH-like). Goodness of the two-component description: the WH score variance relative
to the mixture's predicted variance (var ratio ~ 1 if a mixture of the two references describes the whisker hits).
Statistics: common.measure_stats on pi, pi_early, pi_late, dpi (Wilcoxon vs 0, day 0 vs expert, R+ vs R-, cohort x stage).
Output: tables/012_sessions.csv, 012_stats.csv; figures/012_mixture_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

MIN_EV = 4
MEAS = [("pi", "share AH-like (session)"), ("pi_early", "share AH-like, early half"), ("pi_late", "share AH-like, late half"),
        ("dpi", "Δ share AH-like (late − early)"), ("var_ratio", "WH variance / mixture variance")]


def em(x, m0, s0, m1, s1, n_iter=200):
    pi = 0.5
    l0, l1 = stats.norm.pdf(x, m0, s0), stats.norm.pdf(x, m1, s1)
    for _ in range(n_iter):
        r = pi * l1 / np.maximum(pi * l1 + (1 - pi) * l0, 1e-300)
        new = r.mean()
        if abs(new - pi) < 1e-6:
            break
        pi = new
    return pi, r


def session(E):
    E = E.dropna(subset=["cd"])
    w, a, s = E[E.cls == "WH"], E[E.cls == "AH"], E[E.cls == "SL"]
    if len(w) < 2 * MIN_EV or len(a) < 4 or len(s) < 4:
        return None, None
    m0, s0, m1, s1 = s.cd.mean(), s.cd.std(), a.cd.mean(), a.cd.std()
    pi, r = em(w.cd.to_numpy(), m0, s0, m1, s1)
    med = E.tau.median(); e_, l_ = w[w.tau <= med], w[w.tau > med]
    pe = em(e_.cd.to_numpy(), m0, s0, m1, s1)[0] if len(e_) >= MIN_EV else np.nan
    pl = em(l_.cd.to_numpy(), m0, s0, m1, s1)[0] if len(l_) >= MIN_EV else np.nan
    mix_mean = pi * m1 + (1 - pi) * m0
    mix_var = pi * (s1 ** 2 + m1 ** 2) + (1 - pi) * (s0 ** 2 + m0 ** 2) - mix_mean ** 2
    ev = pd.DataFrame(dict(session_id=E.session_id.iloc[0], tau=w.tau.to_numpy(), cd=w.cd.to_numpy(), p_ah=r))
    return dict(pi=pi, pi_early=pe, pi_late=pl, dpi=pl - pe, var_ratio=w.cd.var() / mix_var if mix_var > 0 else np.nan), ev


def run(variant):
    E = pd.read_parquet(C.TAB / "008_events.parquet").rename(columns={f"cd_{variant}": "cd"})
    rows, evs = [], []
    for sid, g in E.groupby("session_id"):
        o, ev = session(g)
        if o is not None:
            rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0], variant=variant, **o))
            evs.append(ev)
    S = C.population(pd.DataFrame(rows)); P = pd.concat(evs).merge(S[["session_id", "cohort", "stage"]], on="session_id")
    R = pd.DataFrame([r for m, _ in MEAS for r in C.measure_stats(S, m, "whole brain", m)]).assign(variant=variant)
    return S, R, P, E[E.session_id.isin(S.session_id)]


def figure(S, R, P, E, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.6, 5.4)); gs = fig.add_gridspec(2, 4, wspace=0.55, hspace=1.1)
    bins = np.linspace(-2, 3, 41)
    for k, (c, s_) in enumerate(C.GROUPS):
        ax = fig.add_subplot(gs[0, k]); d = E[(E.cohort == c) & (E.stage == s_)]
        med = d.groupby("session_id").tau.transform("median")
        for cl, col, ls in (("SL", "0.3", "-"), ("AH", "#2c2cdb", "-")):
            ax.hist(d[d.cls == cl].cd.dropna(), bins=bins, density=True, histtype="step", color=col, lw=0.8, label=cl)
        for half, col, ls in (("early", "#fdd49e", "-"), ("late", "#f7b519", "-")):
            m_ = (d.tau <= med) if half == "early" else (d.tau > med)
            ax.hist(d[(d.cls == "WH") & m_].cd.dropna(), bins=bins, density=True, histtype="stepfilled", alpha=0.5, color=col, label=f"WH {half}")
        ax.set_xlabel("CD score (SL = 0, AH = 1)"); ax.set_title(f"{C.GLAB[(c, s_)]} ({d.session_id.nunique()})", color=C.COH[c], fontsize=6)
        if k == 0:
            ax.set_ylabel("density (pooled events)"); ax.legend(frameon=False, fontsize=4.4); ax.text(-0.4, 1.15, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    for k, (m, l) in enumerate((("pi_early", "b"), ("pi_late", "c"), ("dpi", "d"), ("var_ratio", "e"))):
        ax = fig.add_subplot(gs[1, k]); Rm = R[R.measure == m]
        C.dot_panel(ax, S, m, Rm, dict(MEAS)[m], l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
        if m == "var_ratio":
            ax.axhline(1, color="0.4", lw=0.6, ls="--")
    fig.suptitle(f"Test 3: whisker hits as a mixture of spontaneous-lick-like and auditory-hit-like events "
                 f"({'baseline-corrected' if variant == 'corr' else 'raw'} rates; a: pooled distributions; b-e: per session)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"012_mixture_{variant}.{ext}", dpi=300, bbox_inches="tight")


def main():
    out = {v: run(v) for v in ("corr", "raw")}
    pd.concat([o[0] for o in out.values()]).to_csv(C.TAB / "012_sessions.csv", index=False)
    R = pd.concat([o[1] for o in out.values()]); R.to_csv(C.TAB / "012_stats.csv", index=False)
    for v, o in out.items():
        figure(*o, v)
    pd.set_option("display.width", 220)
    print(R[R.cohort.isin(["R+", "R-"])][["variant", "measure", "cohort", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "cls"]].round(3).to_string())


if __name__ == "__main__":
    main()
