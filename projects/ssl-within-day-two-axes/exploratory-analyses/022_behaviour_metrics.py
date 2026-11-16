"""022 -- 021 with behavioural-gain metrics that leave out the end-of-session disengagement (user 2026-11-11: "the last third contains
the end disengagement for whisker; remove the last 10% of whisker trials").

Exact curves of whisker day 0 (Figure 1 exploration, curves_day0.parquet); per session, whisker trials after 90% of the session's whisker
trials are dropped (t_norm > 0.9), then on gap = p_whisker - p_chance:
  gap_change90   mean gap over the last third minus the first third of the remaining trials (t_norm 0.6-0.9 vs 0-0.3)
  gap_slope90    least-squares slope of the gap on t_norm over the remaining trials (per session)
  gap_peak90     max gap minus the mean gap over the first 10% of whisker trials
Neural measures as 021 (whole brain, all day-0 sessions, session = unit; not recomputed without the session end). Spearman rho and partial
Spearman controlling for the number of whisker hits, per cohort.
Also: R- hit - miss sessions ranked by the passive whisker change, with their licking (which mice drive the R- correlation).
Output: tables/022_correlations.csv, 022_rminus_hitmiss_sessions.csv; figures/022_behaviour_metrics.{png,pdf}
"""
from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402

m020 = importlib.import_module("020_learner_sensitivity")
m021 = importlib.import_module("021_dprime_correlation")
BEHAV = [("gap_change90", "Δ gap, last − first third\n(first 90% of W trials)"), ("gap_slope90", "gap slope\n(first 90%)"),
         ("gap_peak90", "peak gap − start\n(first 90%)")]


def gain_metrics(src):
    cur = pd.read_parquet(src / "curves_day0.parquet")
    cur["gap"] = cur.p_mean - cur.p_chance
    rows = []
    for m, c in cur.groupby("mouse_id"):
        c = c[c.t_norm <= 0.9]
        t, g = c.t_norm.to_numpy(), c.gap.to_numpy()
        rows.append(dict(mouse_id=m,
                         gap_change90=g[t >= 0.6].mean() - g[t < 0.3].mean(),
                         gap_slope90=np.polyfit(t, g, 1)[0] if len(t) > 2 else np.nan,
                         gap_peak90=g.max() - g[t < 0.1].mean()))
    return pd.DataFrame(rows)


def main():
    rl, H = m020.load_all()
    pm, src = m021.behaviour()
    pm = pm.merge(gain_metrics(src), on="mouse_id", how="left")
    full = pd.read_csv(src / "per_mouse.csv")[["mouse_id", "hit_whisker", "fa", "hit_auditory", "n_no_stim"]]
    pm = pm.merge(full, on="mouse_id", how="left")
    rows = []
    for D, ms in ((rl, m021.MEAS[:3]), (H, m021.MEAS[3:])):
        d = D[(D.area == "whole_brain") & (D.stage == "learning")]
        for m, _ in ms:
            rows.append(d[["session_id", "mouse_id", "cohort", m]].dropna().rename(columns={m: "value"}).assign(measure=m))
    S = pd.concat(rows).merge(pm.drop(columns="mouse_id"), on="session_id", how="left")

    res = []
    for (m, coh), d in S.groupby(["measure", "cohort"]):
        for b, _ in BEHAV + [("dprime", "")]:
            e = d.dropna(subset=[b, "n_hits"])
            r, p = stats.spearmanr(e[b], e.value)
            rp, pp = m021.partial_spearman(e[b].to_numpy(), e.value.to_numpy(), e.n_hits.to_numpy())
            res.append(dict(measure=m, cohort=coh, behaviour=b, n=len(e), rho=r, p=p, rho_partial_hits=rp, p_partial_hits=pp))
    R = pd.DataFrame(res)
    R.to_csv(C.TAB / "022_correlations.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.pivot_table(index=["measure", "cohort"], columns="behaviour", values=["rho", "p", "p_partial_hits"]).round(3).to_string())

    # R- hit - miss: which sessions carry the correlation
    r_ = S[(S.cohort == "R-") & (S.measure == "hm_W")].sort_values("value")
    cols = ["mouse_id", "learning_category", "value", "hit_whisker", "fa", "n_hits", "n_whisker", "dprime", "gap_change90"]
    r_[cols].to_csv(C.TAB / "022_rminus_hitmiss_sessions.csv", index=False)
    print(r_[cols].round(3).to_string())
    for b in ("hit_whisker", "fa", "n_hits", "n_whisker"):
        rr, pp = stats.spearmanr(r_[b], r_.value)
        print(f"R- hm_W vs {b}: rho={rr:.2f} p={pp:.4f}")

    # figure: rows = neural measures, columns = cohort x behaviour metric
    fig, axes = plt.subplots(len(m021.MEAS), 6, figsize=(19, 2.9 * len(m021.MEAS)))
    for i, (m, title) in enumerate(m021.MEAS):
        for j, (coh, (b, blab)) in enumerate([(c, bb) for c in ("R+", "R-") for bb in BEHAV]):
            a = axes[i, j]
            d = S[(S.measure == m) & (S.cohort == coh)].dropna(subset=[b])
            for cat in ("good", "moderate", "bad"):
                e = d[d.learning_category == cat]
                a.scatter(e[b], e.value, s=12, color=m021.shade(C.COH[coh], m021.CAT_SHADE[cat]),
                          edgecolor="k" if cat == "bad" else "none", lw=0.5, label=cat)
            for r in d[d.learning_category == "bad"].itertuples():
                a.annotate(r.mouse_id, (getattr(r, b), r.value), fontsize=5, xytext=(2, 2), textcoords="offset points")
            k = np.polyfit(d[b], d.value, 1)
            xs = np.linspace(d[b].min(), d[b].max(), 2)
            a.plot(xs, np.polyval(k, xs), color="0.4", lw=0.8)
            a.axhline(0, color="0.7", lw=0.5, ls=":")
            r = R[(R.measure == m) & (R.cohort == coh) & (R.behaviour == b)].iloc[0]
            a.text(0.02, 0.98, f"ρ={r.rho:.2f} p={r.p:.3f}\npartial(hits) ρ={r.rho_partial_hits:.2f} p={r.p_partial_hits:.3f}",
                   transform=a.transAxes, fontsize=6, va="top")
            a.set_title(f"{coh}: {title}", fontsize=7)
            a.set_xlabel(blab, fontsize=6.5)
            a.tick_params(labelsize=6)
            if i == 0 and j == 0:
                a.legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.suptitle("Day-0 effect vs behavioural gain without the last 10% of whisker trials (whole brain, session = unit)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG / f"022_behaviour_metrics.{ext}", dpi=170)

    # R- hit - miss vs licking
    d = S[(S.cohort == "R-") & (S.measure.isin(["hm_W", "hm_WA"]))]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for i, m in enumerate(["hm_W", "hm_WA"]):
        e = d[d.measure == m]
        for j, (b, lab) in enumerate([("hit_whisker", "whisker lick probability"), ("fa", "false-alarm rate"),
                                      ("n_hits", "whisker hits (n)"), ("dprime", "session d′")]):
            a = axes[i, j]
            a.scatter(e[b], e.value, s=14, color=C.COH["R-"])
            for r in e.itertuples():
                a.annotate(r.mouse_id, (getattr(r, b), r.value), fontsize=5, xytext=(2, 2), textcoords="offset points")
            rr, pp = stats.spearmanr(e[b], e.value)
            a.set_title(f"R-: {m} vs {lab}\nρ={rr:.2f} p={pp:.4f}", fontsize=7)
            a.axhline(0, color="0.7", lw=0.5, ls=":")
            a.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG / f"022_rminus_hitmiss_licking.{ext}", dpi=170)
    print(C.FIG / "022_behaviour_metrics.png")


if __name__ == "__main__":
    main()
