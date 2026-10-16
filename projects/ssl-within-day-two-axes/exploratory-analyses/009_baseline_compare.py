"""009 -- With vs without per-event baseline correction (user 2026-10-16: "try without baseline correction to see, not sure we should go down
that road unless results differ strongly across conditions").
Events and scores of 008 (variants corr / raw; identical estimator). Per session and variant, the 005 quantities: class slopes and contrasts,
lambda at start / end / change and the session-mean lambda; and the 006 per-event mixed model (WH vs SL; expert - day 0 start offset W:E and
relative slope tau:W:E per cohort). Comparison: per-session scatter raw vs corr, group means and classes per variant.
Output: tables/009_sessions.csv, 009_stats.csv, 009_mixed.csv; figures/009_baseline_compare.{png,pdf,svg}
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd

import common as C

warnings.filterwarnings("ignore")
m005 = importlib.import_module("005_lambda_auditory")
m006 = importlib.import_module("006_mixed_model")
SHOW = [("b_WH-SL", "WH − SL slope"), ("b_AH", "AH slope"), ("lam_start", "λ at start"), ("dlam", "Δλ (end − start)"), ("lam_mean", "λ (session mean)")]


def per_variant(E, v):
    D = E.rename(columns={f"cd_{v}": "cd"})[["session_id", "cls", "tau", "cd"]]
    res = [m005.session(sid, g) for sid, g in D.groupby("session_id")]
    return pd.DataFrame([r for r, _ in res]).assign(variant=v)


def main():
    E = pd.read_parquet(C.TAB / "008_events.parquet")
    meta = E.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    S = C.population(pd.concat([per_variant(E, v) for v in ("corr", "raw")]).merge(meta, on="session_id"))
    S.to_csv(C.TAB / "009_sessions.csv", index=False)
    R = pd.concat([pd.DataFrame(C.measure_stats(S[S.variant == v], m, "whole brain", m)).assign(variant=v)
                   for v in ("corr", "raw") for m, _ in SHOW])
    R.to_csv(C.TAB / "009_stats.csv", index=False)
    rows = []
    for v in ("corr", "raw"):
        T = C.population(E.rename(columns={f"cd_{v}": "cd"}).dropna(subset=["cd", "tau"]))
        T = T[T.cls.isin(["WH", "SL"])].copy(); T["W"] = (T.cls == "WH").astype(int); T["E"] = (T.stage == "expert").astype(int)
        for c in ("R+", "R-"):
            r = m006.fit(T[T.cohort == c], "cd ~ tau * W * E")
            for t in ("tau:W", "tau:W:E", "W", "W:E"):
                rows.append(dict(variant=v, cohort=c, term=t, coef=r.params[t], se=r.bse[t], p=r.pvalues[t]))
    M = pd.DataFrame(rows); M.to_csv(C.TAB / "009_mixed.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.6, 6.4)); gs = fig.add_gridspec(3, 5, wspace=0.6, hspace=1.3)
    P = S.pivot_table(index=["session_id", "cohort", "stage"], columns="variant", values=[m for m, _ in SHOW]).reset_index()
    for j, (m, lab) in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, j])
        for c in ("R+", "R-"):
            for s_, mk in (("learning", "o"), ("expert", "s")):
                d = P[(P.cohort == c) & (P.stage == s_)]
                ax.scatter(d[(m, "corr")], d[(m, "raw")], s=5, marker=mk, color=C.COH[c], facecolor="none" if s_ == "learning" else C.COH[c], lw=0.5)
        x = P[(m, "corr")].to_numpy(); y = P[(m, "raw")].to_numpy(); ok = np.isfinite(x) & np.isfinite(y)
        lim = np.nanpercentile(np.r_[x[ok], y[ok]], [1, 99]); ax.plot(lim, lim, color="0.6", lw=0.5, ls=":")
        ax.set_xlim(lim); ax.set_ylim(lim)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        ax.set_title(f"{lab}\nr = {r:.2f}", fontsize=5.6); ax.set_xlabel("baseline-corrected", fontsize=5)
        if j == 0:
            ax.set_ylabel("raw (no baseline)", fontsize=5); ax.text(-0.5, 1.25, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    for i, v in enumerate(("corr", "raw")):
        for j, (m, lab) in enumerate([x for x in SHOW if x[0] != "lam_mean"] + [("lam_mean", "λ (session mean)")]):
            ax = fig.add_subplot(gs[1 + i, j]); Rm = R[(R.variant == v) & (R.measure == m)]
            C.dot_panel(ax, S[S.variant == v], m, Rm, f"{lab} ({'corrected' if v == 'corr' else 'raw'})",
                        ("b" if i == 0 else "c") if j == 0 else None, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle("Reward-lick CD with vs without per-event baseline correction (a: per session, circles day 0, squares expert; b: corrected; "
                 "c: raw)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"009_baseline_compare.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 230)
    print(R[R.cohort.isin(["R+", "R-"])][["variant", "measure", "cohort", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "cls"]].round(3).to_string())
    print(M.round(4).to_string())


if __name__ == "__main__":
    main()
