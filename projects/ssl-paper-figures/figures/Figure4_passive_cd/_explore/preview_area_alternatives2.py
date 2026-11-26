"""Figure 4G candidates, part 2 (user 2026-11-25: "What about the fine area label? Heatmaps? Correlation with stimulus
encoding?").

Data: 179 units (learners, whisker day 0, stable), per session x area the mean over the area's neurons of the passive evoked
change post - pre (5-35 ms, Hz), whisker W and auditory A; areas kept with >= 10 neurons in a session and >= 3 such sessions
per cohort. Two area levels: area_group (16) and area_acronym_custom (fine). Effect per area: delta = mean_R+(W - A) -
mean_R-(W - A).
Stimulus encoding per area (both cohorts pooled; ROC table, merged roc_core, 5-35 ms, P < 0.05): fraction of neurons with a
significant whisker response in active trials (whisker_active), and fraction whisker-vs-auditory selective in active trials
(wh_vs_aud_active).
Panels:
  a  fine areas: forest plot (delta, bootstrap 95 % CI over sessions), sorted, colour = parent area group
  b  heatmap, area groups: rows = areas (sorted by delta), columns = R+ W, R+ A, R- W, R- A change and delta
  c  same heatmap, fine areas
  d  delta vs fraction whisker-responsive neurons (fine areas; area groups as large dots), Spearman
  e  delta vs fraction whisker-vs-auditory selective neurons, Spearman
Output: Figures/_explore/Figure4_passive_cd_<date>/area_effect_alternatives2.{png,pdf,svg}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_area_alternatives2.py
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
rng = np.random.default_rng(0)
K = ["session_id", "electrode_group", "cid"]


def units():
    u = S._ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant"])
            r = r[r.analysis_type.isin(["whisker_active", "wh_vs_aud_active"])]
            rows.append(r.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type",
                                      values="significant", aggfunc="first").reset_index())
    r = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    r["cid"] = r.cid.astype(int)
    return u.merge(r, on=K, how="left")


def per_area(u, col):
    d = u.groupby(["session_id", "cohort", col]).agg(n=("cid", "size"), W=("ev_post_W", "mean"), W0=("ev_pre_W", "mean"),
                                                     A=("ev_post_A", "mean"), A0=("ev_pre_A", "mean")).reset_index()
    d = d[d.n >= 10]
    d["W"], d["A"] = d.W - d.W0, d.A - d.A0
    d["WA"] = d.W - d.A
    k = d.groupby([col, "cohort"]).session_id.nunique().unstack().fillna(0)
    keep = k[(k.get("R+", 0) >= 3) & (k.get("R-", 0) >= 3)].index
    d = d[d[col].isin(keep)].rename(columns={col: "area"})
    enc = u[u[col].isin(keep)].groupby(col).agg(frac_w=("whisker_active", "mean"), frac_wa=("wh_vs_aud_active", "mean"),
                                                 n_units=("cid", "size"))
    enc.index.name = "area"
    m = d.groupby(["area", "cohort"])[["W", "A", "WA"]].mean().unstack()
    m.columns = [f"{c}_{v}" for v, c in m.columns]
    m["delta"] = m["R+_WA"] - m["R-_WA"]
    ci = {}
    for a_, g in d.groupby("area"):
        p, q = g[g.cohort == "R+"].WA.to_numpy(), g[g.cohort == "R-"].WA.to_numpy()
        b = [rng.choice(p, len(p)).mean() - rng.choice(q, len(q)).mean() for _ in range(2000)]
        ci[a_] = np.percentile(b, [2.5, 97.5])
    m[["lo", "hi"]] = pd.DataFrame(ci, index=["lo", "hi"]).T
    return m.join(enc)


def heat(ax, m, title, labels_fs):
    m = m.sort_values("delta")
    cols = ["R+_W", "R+_A", "R-_W", "R-_A"]
    Z = m[cols].to_numpy()
    v = np.nanpercentile(np.abs(Z), 95)
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["R+ W", "R+ A", "R− W", "R− A"], fontsize=FS, rotation=90)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=labels_fs)
    ax.tick_params(length=0, pad=1)
    ax.set_title(title, fontsize=FS)
    ax2 = ax.inset_axes([1.06, 0, 0.25, 1])
    vd = np.nanmax(np.abs(m.delta))
    ax2.imshow(m[["delta"]].to_numpy(), aspect="auto", cmap="PuOr_r", vmin=-vd, vmax=vd, interpolation="nearest")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Δ"], fontsize=FS)
    ax2.set_yticks([])
    ax2.tick_params(length=0)
    return im


def scatter(ax, mf, mg, x, xlabel):
    col, _, _ = S.area_palette()
    ax.scatter(mf[x], mf.delta, s=4, color="0.6", lw=0, label="fine areas")
    for a_, r in mg.iterrows():
        ax.plot(r[x], r.delta, "o", ms=3.5, color=col.get(a_, "0.3"), mec="k", mew=0.3)
    ok = mf[[x, "delta"]].dropna()
    rf, pf = stats.spearmanr(ok[x], ok.delta)
    rg, pg = stats.spearmanr(mg[x], mg.delta)
    ax.axhline(0, color="0.75", lw=0.4)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel("Δ (R+ − R−) of whisker − auditory\nchange (Hz)", fontsize=FS)
    ax.set_title(f"fine: r={rf:.2f}, p={pf:.3f} (n={len(ok)})\ngroups: r={rg:.2f}, p={pg:.3f} (n={len(mg)})", fontsize=FS - 0.5)
    ax.tick_params(labelsize=FS, length=1.5)
    return rf, pf, rg, pg


def main():
    set_nature_rc()
    u = units()
    mg = per_area(u, "area_group")
    mf = per_area(u, "area_acronym_custom")
    parent = u.groupby("area_acronym_custom").area_group.agg(lambda x: x.mode().iat[0])
    col, _, _ = S.area_palette()
    W, H = 183, 175
    fig = plt.figure(figsize=(W * MM, H * MM))

    def ax_at(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def letter(x, y, s, t):
        fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top")
        fig.text((x + 4) / W, 1 - y / H, t, fontsize=6, va="top")

    # a: fine-area forest
    letter(0, 2, "a", f"Fine areas: Δ with bootstrap 95% CI ({len(mf)} areas)")
    a = ax_at(14, 10, 26, 160)
    s = mf.sort_values("delta")
    for k, (a_, r) in enumerate(s.iterrows()):
        c = col.get(parent.get(a_), "0.4")
        sig = r.lo > 0 or r.hi < 0
        a.plot([r.lo, r.hi], [k, k], color=c, lw=0.6)
        a.plot(r.delta, k, "o", ms=2.0, color=c, mfc=c if sig else "white", mew=0.5)
    a.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_yticks(range(len(s)))
    a.set_yticklabels(s.index, fontsize=FS - 1.5)
    for tl, a_ in zip(a.get_yticklabels(), s.index):
        tl.set_color(col.get(parent.get(a_), "0.3"))
    a.set_ylim(len(s) - 0.5, -0.5)
    lim = np.nanpercentile(np.r_[s.lo, s.hi], [2, 98])
    a.set_xlim(lim[0] - 1, lim[1] + 1)
    a.set_xlabel("Δ (Hz)", fontsize=FS)
    a.tick_params(labelsize=FS, length=1.5)

    # b, c: heatmaps
    letter(52, 2, "b", "Heatmap, area groups")
    b = ax_at(80, 10, 14, 40)
    im = heat(b, mg, "post − pre (Hz)", FS - 1)
    cb = fig.colorbar(im, cax=ax_at(80, 55, 14, 1.2), orientation="horizontal")
    cb.ax.tick_params(labelsize=FS - 1, length=1)
    cb.set_label("Evoked change (Hz)", fontsize=FS - 0.5)
    letter(52, 68, "c", "Heatmap, fine areas")
    c = ax_at(80, 76, 14, 94)
    heat(c, mf, "post − pre (Hz)", FS - 2)

    # d, e: correlation with stimulus encoding
    letter(115, 2, "d", "Δ vs whisker encoding (ROC, active)")
    d_ = ax_at(127, 12, 48, 48)
    r1 = scatter(d_, mf, mg, "frac_w", "Fraction whisker-responsive neurons")
    letter(115, 78, "e", "Δ vs whisker-vs-auditory selectivity")
    e = ax_at(127, 88, 48, 48)
    r2 = scatter(e, mf, mg, "frac_wa", "Fraction whisker-vs-auditory selective neurons")

    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}"
    save_figure(fig, out, "area_effect_alternatives2")
    print("fine areas", len(mf), "significant (CI excludes 0):", int(((mf.lo > 0) | (mf.hi < 0)).sum()),
          "| delta > 0:", int((mf.delta > 0).sum()))
    print("whisker encoding: fine r=%.2f p=%.3f, groups r=%.2f p=%.3f" % r1)
    print("wh-vs-aud: fine r=%.2f p=%.3f, groups r=%.2f p=%.3f" % r2)
    print(mf.sort_values("delta")[["delta", "lo", "hi", "frac_w", "frac_wa", "n_units"]].round(2).tail(8).to_string())
    print("saved", out / "area_effect_alternatives2.png")


if __name__ == "__main__":
    main()
