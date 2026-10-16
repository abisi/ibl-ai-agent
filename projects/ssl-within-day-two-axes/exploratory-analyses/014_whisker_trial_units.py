"""014 -- Test 5: the within-session comparison in units of whisker trials (stimulus-outcome pairings, hits and misses), day 0 vs expert
(user 2026-10-16: "maybe time units is better").
Events: 008 (both variants), x = whisker-trial index (105). Per session:
  slopes per 10 whisker trials of WH, SL and AH scores and the contrasts WH - SL, WH - AH, over the whole session and over the first N_FIRST
  whisker trials (same exposure range at both stages)
  lambda from per-class linear fits against x, evaluated at whisker trial 0 and N_FIRST (lam_0, lam_N; time-matched in pairings), dlam_N
Display: lambda in bins of BIN whisker trials (bins with >= 2 AH and SL events), day 0 vs expert, per cohort; WH score per bin.
Statistics: common.measure_stats.
Output: tables/014_sessions.csv, 014_stats.csv, 014_bins.csv; figures/014_whisker_trial_units_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

N_FIRST, BIN, MIN_EV, SEP_MIN = 60, 20, 4, 0.3
MEAS = [("s_WH-SL", "WH − SL slope / 10 whisker trials"), ("s_WH-SL_first", f"WH − SL slope / 10 WT, first {N_FIRST}"),
        ("s_WH-AH_first", f"WH − AH slope / 10 WT, first {N_FIRST}"), ("lam_0", "λ at whisker trial 0"), ("lam_N", f"λ at whisker trial {N_FIRST}"),
        ("dlam_N", f"Δλ over the first {N_FIRST} whisker trials")]


def fit(d):
    return np.polyfit(d.wtrial.to_numpy(float), d.cd.to_numpy(float), 1) if len(d) >= MIN_EV and d.wtrial.nunique() > 1 else (np.nan, np.nan)


def session(E):
    E = E.dropna(subset=["cd", "wtrial"])
    out = {}
    for tag, d in (("", E), ("_first", E[E.wtrial < N_FIRST])):
        F = {c: fit(d[d.cls == c]) for c in ("WH", "SL", "AH")}
        for c in F:
            out[f"s_{c}{tag}"] = F[c][0] * 10
        out[f"s_WH-SL{tag}"] = out[f"s_WH{tag}"] - out[f"s_SL{tag}"]; out[f"s_WH-AH{tag}"] = out[f"s_WH{tag}"] - out[f"s_AH{tag}"]
        if tag == "_first":
            for x, k in ((0, "lam_0"), (N_FIRST, "lam_N")):
                f = {c: F[c][0] * x + F[c][1] for c in F}; sep = f["AH"] - f["SL"]
                out[k] = (f["WH"] - f["SL"]) / sep if np.isfinite(sep) and sep >= SEP_MIN else np.nan
            out["dlam_N"] = out["lam_N"] - out["lam_0"]
    bins = []
    E = E.assign(b=(E.wtrial // BIN) * BIN)
    for b, g in E.groupby("b"):
        m = g.groupby("cls").cd.agg(["mean", "size"])
        if all(c in m.index for c in ("WH", "SL", "AH")) and m.loc["AH", "size"] >= 2 and m.loc["SL", "size"] >= 2:
            sep = m.loc["AH", "mean"] - m.loc["SL", "mean"]
            bins.append(dict(bin=int(b), wh=m.loc["WH", "mean"], lam=(m.loc["WH", "mean"] - m.loc["SL", "mean"]) / sep if sep >= SEP_MIN else np.nan))
    return out, bins


def run(variant):
    E = pd.read_parquet(C.TAB / "008_events.parquet").rename(columns={f"cd_{variant}": "cd"})
    rows, br = [], []
    for sid, g in E.groupby("session_id"):
        o, b = session(g)
        meta = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0], variant=variant)
        rows.append(dict(meta, **o)); br += [dict(meta, **x) for x in b]
    S = C.population(pd.DataFrame(rows)); B = pd.DataFrame(br); B = B[B.session_id.isin(S.session_id)]
    R = pd.DataFrame([r for m, _ in MEAS for r in C.measure_stats(S, m, "whole brain", m)]).assign(variant=variant)
    return S, R, B


def figure(S, R, B, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.8, 5.4)); gs = fig.add_gridspec(2, 4, wspace=0.6, hspace=1.3)
    for i, c in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[0, i])
        for s_, ls, mfc in (("learning", (0, (3, 1.5)), "white"), ("expert", "-", None)):
            p = B[(B.cohort == c) & (B.stage == s_)].pivot_table(index="session_id", columns="bin", values="lam")
            n = p.notna().sum(); keep = n >= 3
            ax.errorbar(p.columns[keep] + BIN / 2, p.mean()[keep], p.sem()[keep], color=C.COH[c], ls=ls, marker="o", ms=2.6, mfc=mfc or C.COH[c],
                        lw=0.9, capsize=0, label=f"{'day 0' if s_ == 'learning' else 'expert'} ({p.notna().any(axis=1).sum()})")
        ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("whisker trial"); ax.set_ylabel("λ (0 = SL, 1 = AH)"); ax.legend(frameon=False, fontsize=4.6)
        ax.set_title(f"{c.replace('-', '−')}: λ per {BIN} whisker trials", color=C.COH[c], fontsize=6)
        if i == 0:
            ax.text(-0.4, 1.15, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    for k, (m, l) in enumerate(zip([m for m, _ in MEAS], "bcdefg")):
        ax = fig.add_subplot(gs[0, 2 + k] if k < 2 else gs[1, k - 2]); Rm = R[R.measure == m]
        C.dot_panel(ax, S, m, Rm, dict(MEAS)[m], l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle(f"Test 5: day 0 vs expert in units of whisker trials (stimulus-outcome pairings) "
                 f"({'baseline-corrected' if variant == 'corr' else 'raw'} rates)", fontsize=6.4, x=0.01, y=1.05, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"014_whisker_trial_units_{variant}.{ext}", dpi=300, bbox_inches="tight")


def main():
    out = {v: run(v) for v in ("corr", "raw")}
    pd.concat([o[0] for o in out.values()]).to_csv(C.TAB / "014_sessions.csv", index=False)
    R = pd.concat([o[1] for o in out.values()]); R.to_csv(C.TAB / "014_stats.csv", index=False)
    pd.concat([o[2] for o in out.values()]).to_csv(C.TAB / "014_bins.csv", index=False)
    for v, o in out.items():
        figure(*o, v)
    pd.set_option("display.width", 220)
    print(R[R.cohort.isin(["R+", "R-"])][["variant", "measure", "cohort", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "cls"]].round(3).to_string())


if __name__ == "__main__":
    main()
