"""Figure 4H candidates (user 2026-11-26: "Would it be better to correlate with absolute selectivity of the hit/miss ROC? Show
alternatives in _explore. Data inclusion for H?").

Inclusion (as panel H): 179 stable good + MUA units of learners, whisker day 0 (no responsiveness or rate criterion); per
session x area, the mean over the area's neurons; an area is kept if >= 3 mice per cohort each have >= 10 neurons in it (area
groups) or >= 5 (fine areas). One session per mouse on day 0.
Effect (y), per area:
  spks/s      mean over R+ sessions of (dW - dA) minus the same for R- (dW, dA: post - pre evoked change, 5-35 ms)
  normalised  same with the per-neuron response-normalised interaction index (dW - dA) / (|pre W| + |post W| + |pre A| +
              |post A|), mean per session x area
Choice coding (x), per area, ROC whisker_choice (whisker hit vs whisker miss, active trials, 5-35 ms; merged roc_core), both
cohorts pooled, same neurons:
  fraction significant (P < 0.05)            (current panel H)
  mean |selectivity| (all neurons)           threshold-free; selectivity = ROC selectivity column (2 AUC - 1)
  mean |selectivity| (significant neurons)   strength among choice-coding neurons
  mean signed selectivity (all neurons)      direction (hit > miss positive)
Per panel: least-squares line (solid if p < 0.05), Pearson r and p, Spearman rho and p.
Output: Figures/_explore/Figure4_passive_cd_<date>/G_area_alternatives/choice_corr_variants.{png,pdf,svg} + csv
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_choice_corr_variants.py
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
MIN_N = {"area_group": 10, "area_acronym_custom": 5}
XS = [("frac_sig", "Fraction choice-coding\n(P < 0.05)"), ("abs_sel", "Mean |selectivity|\n(all neurons)"),
      ("abs_sel_sig", "Mean |selectivity|\n(choice-coding neurons)"), ("sel", "Mean signed selectivity\n(hit > miss > 0)")]
YS = [("delta", "R+ − R−: whisker − auditory\nchange (spks/s)"), ("delta_norm", "R+ − R−: response-normalised\ninteraction index")]


def units():
    u = S._ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant",
                                        "selectivity"])
            rows.append(r[r.analysis_type == "whisker_choice"].drop(columns="analysis_type"))
    r = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    r["cid"] = r.cid.astype(int)
    u = u.merge(r.drop_duplicates(K), on=K, how="inner")
    u["significant"] = u.significant.astype(float)
    with np.errstate(all="ignore"):
        u["dWA"] = (u.ev_post_W - u.ev_pre_W) - (u.ev_post_A - u.ev_pre_A)
        u["mi_norm"] = u.dWA / (u.ev_pre_W.abs() + u.ev_post_W.abs() + u.ev_pre_A.abs() + u.ev_post_A.abs())
    u["mi_norm"] = u.mi_norm.replace([np.inf, -np.inf], np.nan)
    return u


def per_area(u, col):
    d = u.groupby(["session_id", "mouse_id", "cohort", col]).agg(n=("cid", "size"), dWA=("dWA", "mean"),
                                                                  mi=("mi_norm", "mean")).reset_index()
    d = d[d.n >= MIN_N[col]]
    k = d.groupby([col, "cohort"]).mouse_id.nunique().unstack().fillna(0)
    keep = k[(k.get("R+", 0) >= 3) & (k.get("R-", 0) >= 3)].index
    d = d[d[col].isin(keep)]
    m = d.groupby([col, "cohort"])[["dWA", "mi"]].mean().unstack()
    out = pd.DataFrame({"delta": m[("dWA", "R+")] - m[("dWA", "R-")], "delta_norm": m[("mi", "R+")] - m[("mi", "R-")]})
    uu = u.merge(d[["session_id", col]], on=["session_id", col])
    uu["abs_sel"] = uu.selectivity.abs()
    out = out.join(uu.groupby(col).agg(frac_sig=("significant", "mean"), abs_sel=("abs_sel", "mean"),
                                       sel=("selectivity", "mean"), n_units=("cid", "size")))
    out = out.join(uu[uu.significant == 1].groupby(col).abs_sel.mean().rename("abs_sel_sig"))
    out["n_mice_Rp"], out["n_mice_Rm"] = k.loc[keep, "R+"], k.loc[keep, "R-"]
    return out


def main():
    set_nature_rc()
    u = units()
    col, _, _ = S.area_palette()
    parent = u.groupby("area_acronym_custom").area_group.agg(lambda x: x.mode().iat[0])
    levels = [("area_group", "Area groups"), ("area_acronym_custom", "Fine areas")]
    tabs = {lv: per_area(u, lv) for lv, _ in levels}
    fig, axes = plt.subplots(4, 4, figsize=(183 * MM, 170 * MM))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.07, top=0.93, wspace=0.5, hspace=0.75)
    res = []
    for li, (lv, llab) in enumerate(levels):
        t = tabs[lv]
        cs = [col.get(a_ if lv == "area_group" else parent.get(a_), "0.5") for a_ in t.index]
        for yi, (yk, ylab) in enumerate(YS):
            row = 2 * li + yi
            for j, (xk, xlab) in enumerate(XS):
                a = axes[row, j]
                ok = t[[xk, yk]].notna().all(1).to_numpy()
                x, y = t[xk].to_numpy(float)[ok], t[yk].to_numpy(float)[ok]
                a.scatter(x, y, s=8 if lv == "area_group" else 5, c=list(np.asarray(cs, dtype=object)[ok]), lw=0.2,
                          edgecolors="k", zorder=3)
                lr = stats.linregress(x, y)
                rho, prho = stats.spearmanr(x, y)
                xs = np.linspace(x.min(), x.max(), 50)
                a.plot(xs, lr.intercept + lr.slope * xs, color="0.25", lw=0.7, ls="-" if lr.pvalue < 0.05 else (0, (3, 2)))
                a.axhline(0, color="0.6", lw=0.4, ls=(0, (1, 1.5)))
                a.text(0.03, 0.97, f"n = {len(x)}\nr = {lr.rvalue:.2f}, {S.p_text(lr.pvalue)}\nρ = {rho:.2f}, {S.p_text(prho)}",
                       transform=a.transAxes, ha="left", va="top", fontsize=FS - 1)
                a.set_xlabel(xlab, fontsize=FS)
                a.set_ylabel(f"{llab}\n{ylab}" if j == 0 else ylab, fontsize=FS - 0.5)
                a.tick_params(labelsize=FS, length=1.5)
                res.append(dict(level=llab, effect=yk, choice=xk, n=len(x), r=lr.rvalue, p=lr.pvalue, rho=rho, p_rho=prho))
    fig.suptitle("Area effect vs whisker choice coding (ROC whisker hit vs miss): measures of choice coding (columns) × "
                 "effect measures (rows)", fontsize=FS + 1.5)
    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}" / "G_area_alternatives"
    out.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, "choice_corr_variants")
    r = pd.DataFrame(res)
    r.to_csv(out / "choice_corr_variants_stats.csv", index=False)
    for lv, t in tabs.items():
        t.to_csv(out / f"choice_corr_variants_{lv}.csv")
    pd.set_option("display.width", 200)
    print(r.round(3).to_string())
    print("saved", out)


if __name__ == "__main__":
    main()
