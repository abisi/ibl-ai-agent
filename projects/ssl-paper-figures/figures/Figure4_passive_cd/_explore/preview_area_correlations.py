"""Figure 4G candidates, part 4 (user 2026-11-26: "show more correlations, at the single area group and single fine area level
between effect and sensory responsiveness (whisker, auditory, both, whisker choice coding (whisker hit/miss))"; "keep at least
if 3 mice per cohort per area").

Selection: 179 units (learners, whisker day 0, stable good + MUA); per session x area, the mean over the area's neurons of the
passive evoked change post - pre (5-35 ms, Hz), whisker W and auditory A; an area is kept if at least 3 mice per cohort each
have >= 10 neurons (area groups) or >= 5 neurons (fine areas; user 2026-11-26) in it. Effect per area: delta = mean_R+(W - A) - mean_R-(W - A) (sessions; one session per mouse on day 0).
Encoding per area (ROC table, merged roc_core, both cohorts pooled, same units, P < 0.05), fraction of the area's neurons with:
  whisker      whisker_active   (whisker stimulus vs baseline, 5-35 ms, active trials)
  auditory     auditory_active
  both         whisker_active AND auditory_active
  W vs A       wh_vs_aud_active (whisker vs auditory response, active trials)
  W choice     whisker_choice   (whisker hit vs whisker miss)
Rows: area groups, fine areas (area_acronym_custom). Scatter + OLS line + analytic 95 % band (solid if p < 0.05); Pearson r and
p, Spearman rho and p; colour = area group.
Output: Figures/_explore/Figure4_passive_cd_<date>/G_area_alternatives/area_effect_correlations.{png,pdf,svg}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_area_correlations.py
"""
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT, axel_bisi_root  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

FS = 5
K = ["session_id", "electrode_group", "cid"]
ROC = ["whisker_active", "auditory_active", "wh_vs_aud_active", "whisker_choice"]
MEAS = [("frac_w", "Whisker-responsive"), ("frac_a", "Auditory-responsive"), ("frac_both", "Whisker and auditory"),
        ("frac_wa", "Whisker vs auditory selective"), ("frac_choice", "Whisker choice (hit vs miss)")]
MIN_MICE = 3
MIN_N = {"area_group": 10, "area_acronym_custom": 5}          # neurons per session x area (fine: 5, user 2026-11-26)


def units():
    u = S._ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant"])
            r = r[r.analysis_type.isin(ROC)]
            rows.append(r.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type",
                                      values="significant", aggfunc="first").reset_index())
    r = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    r["cid"] = r.cid.astype(int)
    u = u.merge(r, on=K, how="left")
    for c in ROC:
        u[c] = u[c].astype(float)
    u["both"] = u.whisker_active * u.auditory_active
    return u


def per_area(u, col):
    d = u.groupby(["session_id", "mouse_id", "cohort", col]).agg(
        n=("cid", "size"), W=("ev_post_W", "mean"), W0=("ev_pre_W", "mean"), A=("ev_post_A", "mean"),
        A0=("ev_pre_A", "mean")).reset_index()
    d = d[d.n >= MIN_N[col]]
    d["WA"] = (d.W - d.W0) - (d.A - d.A0)
    k = d.groupby([col, "cohort"]).mouse_id.nunique().unstack().fillna(0)
    keep = k[(k.get("R+", 0) >= MIN_MICE) & (k.get("R-", 0) >= MIN_MICE)].index
    d = d[d[col].isin(keep)]
    m = d.groupby([col, "cohort"]).WA.mean().unstack()
    out = pd.DataFrame({"delta": m["R+"] - m["R-"]})
    out["n_mice_Rp"] = k.loc[keep, "R+"]
    out["n_mice_Rm"] = k.loc[keep, "R-"]
    # encoding fractions over the same neurons (sessions where the area passed the neuron criterion)
    uu = u.merge(d[["session_id", col]], on=["session_id", col])
    enc = uu.groupby(col).agg(frac_w=("whisker_active", "mean"), frac_a=("auditory_active", "mean"),
                              frac_both=("both", "mean"), frac_wa=("wh_vs_aud_active", "mean"),
                              frac_choice=("whisker_choice", "mean"), n_units=("cid", "size"))
    out = out.join(enc)
    out.index.name = "area"
    return out


def regress(ax, x, y, colors, size):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, colors = x[ok], y[ok], np.asarray(colors, dtype=object)[ok]
    ax.scatter(x, y, s=size, c=list(colors), lw=0.2, edgecolors="k", zorder=3)
    n = len(x)
    sl, ic, r, p, _ = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    yhat = ic + sl * x
    s_err = np.sqrt(np.sum((y - yhat) ** 2) / (n - 2))
    band = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xs, ic + sl * xs - band, ic + sl * xs + band, color="0.9", lw=0, edgecolor="none", zorder=1)
    ax.plot(xs, ic + sl * xs, color="0.2", lw=0.7, ls="-" if p < 0.05 else (0, (3, 2)), zorder=2)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (1, 1.5)), zorder=0)
    ax.text(0.03, 0.97, f"n = {n}\nr = {r:.2f}, p = {p:.3f}\nρ = {rho:.2f}, p = {prho:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS - 0.5)
    ax.tick_params(labelsize=FS, length=1.5)
    return dict(n=n, r=r, p=p, rho=rho, p_rho=prho)


def main():
    set_nature_rc()
    u = units()
    col, _, _ = S.area_palette()
    parent = u.groupby("area_acronym_custom").area_group.agg(lambda x: x.mode().iat[0])
    levels = [("area_group", "Area groups"), ("area_acronym_custom", "Fine areas")]
    tabs = {lv: per_area(u, lv) for lv, _ in levels}
    W, H = 183, 85
    fig, axes = plt.subplots(2, len(MEAS), figsize=(W * MM, H * MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.90, wspace=0.35, hspace=0.55)
    res = []
    for i, (lv, lab) in enumerate(levels):
        t = tabs[lv]
        cols = [col.get(a_ if lv == "area_group" else parent.get(a_), "0.5") for a_ in t.index]
        for j, (m, mlab) in enumerate(MEAS):
            a = axes[i, j]
            st = regress(a, t[m].to_numpy(float), t.delta.to_numpy(float), cols, 9 if lv == "area_group" else 5)
            res.append(dict(level=lab, measure=mlab, **st))
            a.set_xlabel(f"Fraction {mlab[0].lower() + mlab[1:]}", fontsize=FS)
            if j == 0:
                a.set_ylabel(f"{lab}\nΔ (R+ − R−) whisker − auditory\nevoked change (spks/s)", fontsize=FS)
            if i == 0:
                a.set_title(mlab, fontsize=FS + 0.5, pad=3)
    fig.suptitle("Area effect vs encoding (ROC, active trials, both cohorts pooled); areas with ≥ 3 mice per cohort, "
                 "≥ 10 neurons each (fine areas ≥ 5)", fontsize=FS + 1, y=0.985)
    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}" / "G_area_alternatives"
    save_figure(fig, out, "area_effect_correlations")
    r = pd.DataFrame(res)
    r.to_csv(out / "area_effect_correlations_stats.csv", index=False)
    for lv, t in tabs.items():
        t.to_csv(out / f"area_effect_correlations_{lv}.csv")
    print(r.round(3).to_string())
    for lv, t in tabs.items():
        print(lv, len(t), "areas; delta > 0:", int((t.delta > 0).sum()))
    print("saved", out / "area_effect_correlations.png")


if __name__ == "__main__":
    main()
