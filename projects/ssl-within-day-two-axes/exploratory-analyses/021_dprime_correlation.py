"""021 -- Do the day-0 two-axis effects scale with behavioural discrimination? (user 2026-11-11: instead of bad / moderate / good,
correlate the effect with d'.)

Per day-0 session (all mice, no learner filter; whole brain; session = unit), per cohort:
  neural measures as 003 (rl_WH_raw, rl_WH, rl_WHSL; hm_W, hm_A, hm_WA);
  behaviour (Figure 1 exploration, Figures/_explore/learner_criterion_<date>/; whisker day 0, early licks dropped):
    dprime      session d' (log-linear correction), whisker vs no-stimulus licks
    gap_change  within-session change of the exact-curve gap: mean (p_whisker - p_chance) over the last third of whisker trials
                minus the first third
  Spearman rho (effect vs behaviour), and partial Spearman controlling for the number of whisker hits (ranks residualised on
  ranked hits; hits set the precision of the drift and of the shift null). Hit - miss measures also without AB119_20240731_102619
  (outlier: 192 units, 146 in wS1, largest shift-null s.d.).
Output: tables/021_sessions.csv, 021_correlations.csv; figures/021_dprime_correlation.{png,pdf}
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
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m020 = importlib.import_module("020_learner_sensitivity")
OUTLIER = "AB119_20240731_102619"
MEAS = [("rl_WH_raw", "Reward-lick CD: WH drift (raw)"), ("rl_WH", "Reward-lick CD: WH drift beyond null"),
        ("rl_WHSL", "Reward-lick CD: WH − SL drift beyond null"), ("hm_W", "Hit − miss: passive W post − pre"),
        ("hm_A", "Hit − miss: passive A post − pre"), ("hm_WA", "Hit − miss: W − A post − pre")]
BEHAV = [("dprime", "session d′"), ("gap_change", "Δ gap (last − first third)")]
CAT_SHADE = {"good": 1.0, "moderate": 0.6, "bad": 0.25}


def behaviour():
    src = sorted((axel_bisi_root() / "manuscripts" / "Bisi_Hamon" / "Figures" / "_explore").glob("learner_criterion_*"))[-1]
    pm = pd.read_csv(src / "per_mouse.csv")
    pm["n_hits"] = (pm.hit_whisker * pm.n_whisker).round()
    cur = pd.read_parquet(src / "curves_day0.parquet")
    cur["gap"] = cur.p_mean - cur.p_chance
    third = lambda t: np.where(t < 1 / 3, "first", np.where(t >= 2 / 3, "last", "mid"))
    g = cur.assign(part=third(cur.t_norm)).groupby(["mouse_id", "part"]).gap.mean().unstack()
    pm["gap_change"] = pm.mouse_id.map(g["last"] - g["first"])
    return pm[["mouse_id", "session_id", "learning_category", "dprime", "gap_change", "n_hits", "n_whisker"]], src


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return stats.pearsonr(res(rx), res(ry))


def shade(color, f):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - f * (1 - rgb))


def main():
    rl, H = m020.load_all()
    pm, src = behaviour()
    rows = []
    for D, ms in ((rl, MEAS[:3]), (H, MEAS[3:])):
        d = D[(D.area == "whole_brain") & (D.stage == "learning")]
        for m, _ in ms:
            x = d[["session_id", "mouse_id", "cohort", m]].dropna().rename(columns={m: "value"}).assign(measure=m)
            rows.append(x)
    S = pd.concat(rows).merge(pm.drop(columns="mouse_id"), on="session_id", how="left")
    S.to_csv(C.TAB / "021_sessions.csv", index=False)
    print("sessions without behaviour:", sorted(S[S.dprime.isna()].session_id.unique()))

    res = []
    for (m, coh), d in S.groupby(["measure", "cohort"]):
        for excl in ([False, True] if m.startswith("hm") else [False]):
            dd = d[d.session_id != OUTLIER] if excl else d
            for b, _ in BEHAV:
                e = dd.dropna(subset=[b, "n_hits"])
                if len(e) < 6:
                    continue
                r, p = stats.spearmanr(e[b], e.value)
                rp, pp = partial_spearman(e[b].to_numpy(), e.value.to_numpy(), e.n_hits.to_numpy())
                rh, ph = stats.spearmanr(e.n_hits, e.value)
                res.append(dict(measure=m, cohort=coh, behaviour=b, without_outlier=excl, n=len(e), rho=r, p=p,
                                rho_partial_hits=rp, p_partial_hits=pp, rho_vs_hits=rh, p_vs_hits=ph))
    R = pd.DataFrame(res)
    R.to_csv(C.TAB / "021_correlations.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string())

    fig, axes = plt.subplots(len(MEAS), 4, figsize=(14, 2.8 * len(MEAS)))
    for i, (m, title) in enumerate(MEAS):
        for j, (coh, (b, blab)) in enumerate([(c, bb) for c in ("R+", "R-") for bb in BEHAV]):
            a = axes[i, j]
            d = S[(S.measure == m) & (S.cohort == coh)].dropna(subset=[b])
            for cat in ("good", "moderate", "bad"):
                e = d[d.learning_category == cat]
                a.scatter(e[b], e.value, s=12, color=shade(C.COH[coh], CAT_SHADE[cat]),
                          edgecolor="k" if cat == "bad" else "none", lw=0.5, label=cat)
            for r in d[(d.learning_category == "bad") | (d.session_id == OUTLIER)].itertuples():
                a.annotate(r.session_id.split("_")[0], (getattr(r, b), r.value), fontsize=5, xytext=(2, 2), textcoords="offset points")
            if len(d) > 2:
                k = np.polyfit(d[b], d.value, 1)
                xs = np.linspace(d[b].min(), d[b].max(), 2)
                a.plot(xs, np.polyval(k, xs), color="0.4", lw=0.8)
            a.axhline(0, color="0.7", lw=0.5, ls=":")
            r = R[(R.measure == m) & (R.cohort == coh) & (R.behaviour == b) & ~R.without_outlier]
            txt = "" if r.empty else (f"ρ={r.rho.iloc[0]:.2f} p={r.p.iloc[0]:.3f}\n"
                                      f"partial(hits) ρ={r.rho_partial_hits.iloc[0]:.2f} p={r.p_partial_hits.iloc[0]:.3f}")
            ro = R[(R.measure == m) & (R.cohort == coh) & (R.behaviour == b) & R.without_outlier]
            if not ro.empty:
                txt += f"\nw/o AB119 ρ={ro.rho.iloc[0]:.2f} p={ro.p.iloc[0]:.3f}"
            a.text(0.02, 0.98, txt, transform=a.transAxes, fontsize=5.5, va="top")
            a.set_title(f"{coh}: {title}", fontsize=7)
            a.set_xlabel(blab, fontsize=7)
            a.tick_params(labelsize=6)
            if i == 0 and j == 0:
                a.legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.suptitle(f"Day-0 effect vs behaviour per session (whole brain; behaviour from {src.name})", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG / f"021_dprime_correlation.{ext}", dpi=200)
    print(C.FIG / "021_dprime_correlation.png")


if __name__ == "__main__":
    main()
