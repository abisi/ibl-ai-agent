"""010 -- Test 1: is there a step in the whisker-hit position on the reward-lick CD at each mouse's behavioural learning trial, beyond the
smooth within-session ramp? (user 2026-10-16). Placebo splits within the session; expert sessions as control with pseudo-learning trials.
Events: 008 (both variants). x = whisker-trial index (hits and misses; 105), LT = 017 lt_final (sigma1) on day-0 sessions with a learning
trial. Per session and class (WH, SL): OLS score = a + b x + c [x >= k] (ramp + step at split k); step_WH(k), step_SL(k), relative step
step_rel(k) = step_WH - step_SL (shared changes cancel). Real split k = LT; placebo splits = every whisker-trial position with >= MIN_SIDE
WH and SL events on each side and |k - LT| >= MIN_DIST; excess = step(LT) - mean placebo step; percentile of step(LT) among placebos.
Expert control: pseudo-LT = the median relative LT position of the day-0 learners of the same cohort (fraction of whisker trials), same
estimator and placebos.
LT-aligned display: WH and SL scores in 5-whisker-trial bins from -30 to +30 around the LT (day 0), minus the session's mean in the
pre-LT bins, mean +- s.e.m. over sessions.
Statistics: Wilcoxon of the excess vs 0 and of the percentile vs 0.5 per group; R+ vs R- (MWU, Welch); day 0 vs expert per cohort (MWU).
Output: tables/010_sessions.csv, 010_stats.csv, 010_aligned.csv; figures/010_lt_aligned_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

MIN_SIDE, MIN_DIST, BIN, SPAN = 4, 10, 5, 30


def steps(E, k):
    out = {}
    for c in ("WH", "SL"):
        d = E[E.cls == c]
        s_ = (d.wtrial >= k).to_numpy()
        if min(s_.sum(), (~s_).sum()) < MIN_SIDE:
            return None
        X = np.column_stack([np.ones(len(d)), d.wtrial.to_numpy(float), s_.astype(float)])
        b, *_ = np.linalg.lstsq(X, d.cd.to_numpy(float), rcond=None); out[c] = b[2]
    return out["WH"], out["SL"], out["WH"] - out["SL"]


def session(E, k_real):
    E = E.dropna(subset=["cd", "wtrial"])
    r = steps(E, k_real)
    if r is None:
        return None
    pl = []
    for k in range(int(E.wtrial.min()) + 1, int(E.wtrial.max())):
        if abs(k - k_real) < MIN_DIST:
            continue
        q = steps(E, k)
        if q is not None:
            pl.append(q)
    if len(pl) < 5:
        return None
    pl = np.array(pl)
    out = dict(n_placebo=len(pl))
    for i, n in enumerate(("WH", "SL", "rel")):
        out[f"step_{n}"] = r[i]; out[f"placebo_{n}"] = pl[:, i].mean(); out[f"excess_{n}"] = r[i] - pl[:, i].mean()
        out[f"pct_{n}"] = (np.sum(pl[:, i] < r[i]) + 0.5 * np.sum(pl[:, i] == r[i])) / len(pl)
    return out


def aligned(E, k):
    E = E.dropna(subset=["cd", "wtrial"]).assign(rel=lambda d: d.wtrial - k)
    E = E[(E.rel >= -SPAN) & (E.rel < SPAN)]; E["bin"] = np.floor(E.rel / BIN).astype(int) * BIN
    rows = []
    for c in ("WH", "SL"):
        d = E[E.cls == c]; ref = d[d.rel < 0].cd.mean()
        for b, g in d.groupby("bin"):
            rows.append(dict(cls=c, bin=b, score=g.cd.mean() - ref))
    return rows


def run(variant):
    E = pd.read_parquet(C.TAB / "008_events.parquet").rename(columns={f"cd_{variant}": "cd"})
    S0 = pd.read_csv(C.TAB / "008_sessions.csv")
    S0 = C.population(S0)
    rows, al = [], []
    d0 = S0[(S0.stage == "learning") & S0.lt_wtrial.notna()]
    relpos = (d0.lt_wtrial / d0.n_wtrials).groupby(d0.cohort).median()
    for r in S0.itertuples():
        if r.stage == "learning":
            if not np.isfinite(r.lt_wtrial):
                continue
            k = int(r.lt_wtrial)
        else:
            if not np.isfinite(getattr(r, "n_wtrials", np.nan)):
                continue
            k = int(round(relpos[r.cohort] * r.n_wtrials))
        g = E[E.session_id == r.session_id]
        o = session(g, k)
        if o is None:
            continue
        rows.append(dict(session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, k=k, n_wtrials=r.n_wtrials, **o))
        if r.stage == "learning":
            al += [dict(session_id=r.session_id, cohort=r.cohort, **x) for x in aligned(g, k)]
    S = pd.DataFrame(rows).assign(variant=variant); A = pd.DataFrame(al).assign(variant=variant)
    St = []
    for n in ("WH", "SL", "rel"):
        for c in ("R+", "R-"):
            for s_ in ("learning", "expert"):
                v = S[(S.cohort == c) & (S.stage == s_)]
                if len(v) < 3:
                    continue
                St.append(dict(variant=variant, measure=n, cohort=c, stage=s_, n=len(v), excess=v[f"excess_{n}"].mean(),
                               p_excess=stats.wilcoxon(v[f"excess_{n}"]).pvalue, pct=v[f"pct_{n}"].mean(),
                               p_pct=stats.wilcoxon(v[f"pct_{n}"] - 0.5).pvalue))
            a, b = (S[(S.cohort == c) & (S.stage == s_)][f"excess_{n}"] for s_ in ("learning", "expert"))
            if len(a) >= 3 and len(b) >= 3:
                St.append(dict(variant=variant, measure=n, cohort=c, stage="day 0 vs expert", p_excess=stats.mannwhitneyu(a, b).pvalue))
        a, b = (S[(S.cohort == c) & (S.stage == "learning")][f"excess_{n}"] for c in ("R+", "R-"))
        St.append(dict(variant=variant, measure=n, cohort="R+ vs R-", stage="learning", p_excess=stats.mannwhitneyu(a, b).pvalue,
                       p_pct=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return S, A, pd.DataFrame(St)


def figure(S, A, St, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.6, 4.8)); gs = fig.add_gridspec(2, 4, wspace=0.6, hspace=0.9)
    CL = {"WH": "#f7b519", "SL": "#262626"}
    for i, c in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[i, 0])
        for cl in ("SL", "WH"):
            d = A[(A.cohort == c) & (A.cls == cl)].pivot_table(index="session_id", columns="bin", values="score")
            ax.errorbar(d.columns + BIN / 2, d.mean(), d.sem(), color=CL[cl], marker="o", ms=2.4, lw=0.9, capsize=0, label=cl)
        ax.axvline(0, color="0.4", lw=0.6, ls="--"); ax.axhline(0, color="0.8", lw=0.4)
        ax.set_xlabel("whisker trials from the learning trial"); ax.set_ylabel("score re. pre-LT mean")
        ax.set_title(f"{c.replace('-', '−')} day 0 ({A[A.cohort == c].session_id.nunique()})", color=C.COH[c], fontsize=6)
        if i == 0:
            ax.legend(frameon=False, fontsize=4.8); ax.text(-0.4, 1.15, "a", transform=ax.transAxes, fontsize=8, weight="bold")
        # b: step at LT vs placebo mean, per session
        ax = fig.add_subplot(gs[i, 1]); d = S[(S.cohort == c) & (S.stage == "learning")]
        for _, r in d.iterrows():
            ax.plot([0, 1], [r.placebo_rel, r.step_rel], color=C.COH[c], lw=0.4, alpha=0.4)
        ax.errorbar([0, 1], [d.placebo_rel.mean(), d.step_rel.mean()], [d.placebo_rel.sem(), d.step_rel.sem()], color=C.COH[c], marker="o", ms=3, capsize=0, lw=1.2)
        st = St[(St.measure == "rel") & (St.cohort == c) & (St.stage == "learning")]
        ax.set_xticks([0, 1], ["placebo\nsplits", "learning\ntrial"]); ax.set_xlim(-0.4, 1.4); ax.axhline(0, color="0.8", lw=0.4)
        ax.set_ylabel("relative step WH − SL"); ax.set_title(f"step at LT vs placebo\nexcess {C.fp(st.p_excess.iloc[0]) if len(st) else 'n/a'}", fontsize=5.6)
        if i == 0:
            ax.text(-0.45, 1.15, "b", transform=ax.transAxes, fontsize=8, weight="bold")
        # c: percentile histogram
        ax = fig.add_subplot(gs[i, 2])
        ax.hist(d.pct_rel.dropna(), bins=np.linspace(0, 1, 11), color=C.COH[c], alpha=0.7)
        ax.axvline(0.5, color="0.4", lw=0.6, ls="--"); ax.set_xlabel("percentile of the LT step among placebos")
        ax.set_title(f"mean {d.pct_rel.mean():.2f} ({C.fp(st.p_pct.iloc[0]) if len(st) else 'n/a'} vs 0.5)", fontsize=5.6)
        if i == 0:
            ax.text(-0.35, 1.15, "c", transform=ax.transAxes, fontsize=8, weight="bold")
        # d: excess day 0 (LT) vs expert (pseudo-LT), WH / SL / rel
        ax = fig.add_subplot(gs[i, 3])
        for k, n in enumerate(("WH", "SL", "rel")):
            for s_, xo in (("learning", -0.15), ("expert", 0.15)):
                v = S[(S.cohort == c) & (S.stage == s_)][f"excess_{n}"].dropna()
                ax.errorbar(k + xo, v.mean(), v.sem(), fmt="o", ms=3, color=C.COH[c], mfc="white" if s_ == "learning" else C.COH[c], capsize=0)
            pde = St[(St.measure == n) & (St.cohort == c) & (St.stage == "day 0 vs expert")]
            ax.text(k, 1.0, f"D0 vs E\n{C.fp(pde.p_excess.iloc[0]) if len(pde) else 'n/a'}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.2)
        ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_xticks([0, 1, 2], ["WH", "SL", "WH − SL"])
        ax.set_ylabel("step excess over placebo"); ax.set_title("day 0 (open, LT) vs expert\n(filled, pseudo-LT)", fontsize=5.6, pad=16)
        if i == 0:
            ax.text(-0.45, 1.25, "d", transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle(f"Test 1: step of the reward-lick CD score at the behavioural learning trial beyond the within-session ramp "
                 f"({'baseline-corrected' if variant == 'corr' else 'raw'} rates; green R+, magenta R−)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"010_lt_aligned_{variant}.{ext}", dpi=300, bbox_inches="tight")


def main():
    out = [run(v) for v in ("corr", "raw")]
    S, A, St = (pd.concat([o[i] for o in out]) for i in range(3))
    S.to_csv(C.TAB / "010_sessions.csv", index=False); A.to_csv(C.TAB / "010_aligned.csv", index=False); St.to_csv(C.TAB / "010_stats.csv", index=False)
    for v in ("corr", "raw"):
        figure(S[S.variant == v], A[A.variant == v], St[St.variant == v], v)
    pd.set_option("display.width", 220)
    print(St.round(4).to_string())


if __name__ == "__main__":
    main()
