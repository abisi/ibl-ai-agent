"""180 -- Overview of the clean CD pipeline (179) for the Figure 4 review (user 2026-11-23: "keep going" after step 1). One page,
every candidate panel of the main figure + supp; exploratory, uncorrected. Not a paper figure.
Session set: learning stage = sheet learners (mouse_reference learning_category good / moderate), one day-0 session per mouse;
expert stage (if 179_cd_clean_expert.parquet exists) = all mice (learner rule: only day-0 sessions of bad mice dropped).
Excess = real - shift-null mean (179: labels shifted 10-50 %, non-wrapping). Tests: R+ vs R- Mann-Whitney AND Welch (session =
mouse at day 0); W - A interaction: cohort labels permuted across sessions (10000).
  a / b  time course on the axis (All units), R+ / R-: passive pre / post W, held-out active hits / misses (LOO)
  c      d_along excess, W and A, per cohort (+ interaction)          d  angle change excess (deg), W and A
  e      Delta a decomposed (W): a = rho cos -> Delta a = Delta rho cos_pre (gain) + rho_post Delta cos (rotation)
  f      stimulus axis s (active W - A): Delta projection of passive W on s, and cos(u, s)
  g      area_group: d_along excess W, R+ vs R-                       h  areas: d_along excess W vs Delta evoked rate W
  i      fine areas (>= 4 sessions per cohort): d_along excess W       j  units: Spearman(u weight, Delta evoked W) per session
  k      across days: d_along excess W, learning vs expert            l  behaviour: d_along excess W vs P(lick | W) in the session
Output: figures/180_cd_clean_overview.png|pdf, tables/180_cd_clean_stats.csv. Run (haas, repo root): python .../180_cd_clean_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

EA = Path(__file__).resolve().parent
REPO = EA.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
COL = {"R+": "#00B400", "R-": "#C800C8"}
LAB = {"R+": "R+", "R-": "R−"}
RNG = np.random.default_rng(0)


def learners():
    r = pd.read_parquet(REPO / "reports" / "ssl_analysis" / "derived" / "mouse_reference.parquet")
    return set(r[r.learning_category.isin(["good", "moderate"])].subject_id)


def tests(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if min(len(x), len(y)) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue


def _d(v, g):
    return v[g == "R+"].mean() - v[g == "R-"].mean()


def perm_diff(v, g, n=10000):
    v, g = np.asarray(v, float), np.asarray(g)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    obs = v[g == "R+"].mean() - v[g == "R-"].mean()
    null = np.array([_d(v, RNG.permutation(g)) for _ in range(n)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)


def pstr(pm, pw):
    f = lambda p: "n.a." if not np.isfinite(p) else (f"{p:.3f}" if p >= 0.001 else "<.001")
    return f"MW {f(pm)}, W {f(pw)}"


def strip(ax, d, col, keys, labels=None, ylab="", show_p=True):
    """cohort strips per key; returns stats rows"""
    out = []
    for i, k in enumerate(keys):
        sub = d[d.key == k] if "key" in d else d
        for j, rg in enumerate(("R+", "R-")):
            v = sub[sub.reward_group == rg][col].dropna().to_numpy()
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + RNG.uniform(-0.07, 0.07, len(v)), v, s=4, color=COL[rg], alpha=0.45, lw=0)
            if len(v):
                ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=COL[rg], marker="o", ms=3.5, capsize=0, lw=1.3)
        a, b = (sub[sub.reward_group == rg][col] for rg in ("R+", "R-"))
        pm, pw = tests(a, b)
        if show_p:
            ax.text(i, 1.0, pstr(pm, pw), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.3)
        out.append(dict(key=k, measure=col, n_Rp=a.notna().sum(), n_Rm=b.notna().sum(), mean_Rp=a.mean(), mean_Rm=b.mean(),
                        p_MW=pm, p_Welch=pw))
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(range(len(keys)), labels or keys, fontsize=5.5)
    ax.set_ylabel(ylab, fontsize=5.5)
    return out


def main():
    tab = EA / "179_cd_clean.parquet"
    D = pd.read_parquet(tab)
    D = D[D.skipped_reason.isna()]
    L = learners()
    D = D[D.mouse_id.isin(L)].copy()
    D["q"] = D.d_along - D.d_along_null
    D["qth"] = D.dtheta - D.dtheta_null
    Tm = pd.read_parquet(EA / "179_cd_time.parquet")
    Tm = Tm[Tm.mouse_id.isin(L)]
    Un = pd.read_parquet(EA / "179_cd_units.parquet")
    Un = Un[Un.mouse_id.isin(L)]
    W = D[(D.area == "All units") & (D.stim == "W")].set_index("session_id")
    A = D[(D.area == "All units") & (D.stim == "A")].set_index("session_id")
    S = []

    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axs = plt.subplots(3, 4, figsize=(8.27, 7.6), gridspec_kw=dict(hspace=0.75, wspace=0.5))
    ax = axs.flat

    # a / b time course
    for k, rg in enumerate(("R+", "R-")):
        a_ = ax[k]
        t = Tm[(Tm.area == "All units") & (Tm.reward_group == rg)]
        for cls, c, ls, lab in (("hit", "k", "-", "active hit (held out)"), ("miss", "0.55", "-", "active miss (held out)"),
                                ("pre_W", COL[rg], "-", "passive pre W"), ("post_W", COL[rg], "--", "passive post W"),
                                ("post_A", "tab:blue", "--", "passive post A")):
            g = t[t.cls == cls].pivot_table(index="session_id", columns="t", values="value")
            m, se = g.mean(), g.sem()
            a_.plot(m.index * 1000, m, color=c, ls=ls, lw=0.9, label=lab)
            a_.fill_between(m.index * 1000, m - se, m + se, color=c, alpha=0.15, lw=0)
        a_.axvspan(5, 35, color="0.9", lw=0, zorder=0)
        a_.axhline(0, color="0.6", lw=0.4)
        a_.set(xlim=(-100, 200), xlabel="Time from stimulus (ms)", ylabel="Projection on hit − miss axis")
        a_.set_title(f"{'ab'[k]}  {LAB[rg]}: time course (n {t.session_id.nunique()})", fontsize=6.5, loc="left")
        if k == 0:
            a_.legend(fontsize=4.3, frameon=False)

    # c projection excess
    d = pd.concat([W.assign(key="W"), A.assign(key="A")]).reset_index()
    S += strip(ax[2], d, "q", ["W", "A"], ["Whisker", "Auditory"], "Δ projection, post − pre (excess over null)")
    obs, p = perm_diff((W.q - A.q.reindex(W.index)).to_numpy(), W.reward_group.to_numpy())
    S.append(dict(key="W-A", measure="q interaction", mean_Rp=obs, p_perm=p))
    ax[2].set_title(f"c  Δ projection; W−A interaction p {p:.3f}", fontsize=6.5, loc="left")

    # d angle
    S += strip(ax[3], d, "qth", ["W", "A"], ["Whisker", "Auditory"], "Δ angle to axis, post − pre (deg, excess)")
    obs, p = perm_diff((W.qth - A.qth.reindex(W.index)).to_numpy(), W.reward_group.to_numpy())
    S.append(dict(key="W-A", measure="qth interaction", mean_Rp=obs, p_perm=p))
    ax[3].set_title(f"d  Δ angle; W−A interaction p {p:.3f}", fontsize=6.5, loc="left")

    # e gain / rotation decomposition (whisker)
    w = W.copy()
    w["gain"] = (w.rho_post - w.rho_pre) * w.cos_pre
    w["rot"] = w.rho_post * (w.cos_post - w.cos_pre)
    e = pd.concat([w.assign(key="gain", v=w.gain), w.assign(key="rot", v=w.rot), w.assign(key="Δa", v=w.d_along)]).reset_index()
    S += strip(ax[4], e, "v", ["Δa", "gain", "rot"], ["Δa (W)", "Δρ·cos_pre\n(gain)", "ρ_post·Δcos\n(rotation)"], "Δ projection (W)")
    ax[4].set_title("e  gain vs rotation (W)", fontsize=6.5, loc="left")

    # f stimulus axis
    w["ds"] = w.s_post - w.s_pre
    f_ = pd.concat([w.assign(key="ds", v=w.ds), w.assign(key="cos", v=w.cos_u_s)]).reset_index()
    S += strip(ax[5], f_, "v", ["ds", "cos"], ["Δ proj. W on\nstimulus axis", "cos(hit−miss,\nstimulus axis)"], "")
    ax[5].set_title("f  stimulus axis (active W − A)", fontsize=6.5, loc="left")

    # g area_group
    G = D[(D.area_col == "area_group") & (D.stim == "W")]
    cnt = G.groupby(["area", "reward_group"]).session_id.nunique().unstack()
    keep = cnt[(cnt.fillna(0).min(1) >= 3)].index.tolist()
    order = G[G.area.isin(keep)].groupby("area").q.mean().sort_values().index.tolist()
    S += strip(ax[6], G.assign(key=G.area), "q", order, [a.replace(" areas", "").replace("Somatosensory-", "SS-")[:12] for a in order],
               "Δ projection W (excess)", show_p=False)
    ax[6].tick_params(axis="x", labelrotation=90, labelsize=4.5)
    ax[6].set_title("g  area groups (p in table)", fontsize=6.5, loc="left")

    # h area level: excess vs evoked change
    H = D[(D.area_col == "area_group") & (D.stim == "W") & D.area.isin(keep)].assign(dev=lambda x: x.ev_post - x.ev_pre)
    hm = H.groupby(["area", "reward_group"])[["q", "dev"]].mean().reset_index()
    for rg in ("R+", "R-"):
        h = hm[hm.reward_group == rg]
        ax[7].scatter(h.dev, h.q, s=8, color=COL[rg], lw=0)
        r, p = stats.spearmanr(h.dev, h.q)
        ax[7].text(0.02, 0.95 - 0.08 * (rg == "R-"), f"{LAB[rg]} ρ {r:+.2f}, p {p:.2f} (areas)", transform=ax[7].transAxes,
                   color=COL[rg], fontsize=4.8)
        S.append(dict(key="areas", measure=f"spearman q vs dev {rg}", mean_Rp=r, p_MW=p))
        rs, ps = stats.spearmanr(H[H.reward_group == rg].dev, H[H.reward_group == rg].q, nan_policy="omit")
        S.append(dict(key="session x area", measure=f"spearman q vs dev {rg}", mean_Rp=rs, p_MW=ps))
    for a_, g in hm.groupby("area"):
        ax[7].plot(g.dev, g.q, color="0.8", lw=0.4, zorder=0)
    ax[7].axhline(0, color="0.6", lw=0.4, ls=":"); ax[7].axvline(0, color="0.6", lw=0.4, ls=":")
    ax[7].set(xlabel="Δ evoked rate W, post − pre (sp/s)", ylabel="Δ projection W (excess)")
    ax[7].set_title("h  areas: Δ axis vs Δ response", fontsize=6.5, loc="left")

    # i fine areas
    F = D[(D.area_col == "area_acronym_custom") & (D.stim == "W")]
    cf = F.groupby(["area", "reward_group"]).session_id.nunique().unstack()
    kf = cf[cf.fillna(0).min(1) >= 4].index.tolist()
    fm = F[F.area.isin(kf)].groupby(["area", "reward_group"]).q.agg(["mean", "sem", "count"]).reset_index()
    diff = fm.pivot(index="area", columns="reward_group", values="mean")
    ordf = (diff["R+"] - diff["R-"]).sort_values().index.tolist()
    for j, rg in enumerate(("R+", "R-")):
        f2 = fm[fm.reward_group == rg].set_index("area").reindex(ordf)
        ax[8].errorbar(np.arange(len(ordf)) + (j - 0.5) * 0.3, f2["mean"], f2["sem"], fmt="o", ms=2, color=COL[rg], lw=0.7)
    for i, a_ in enumerate(ordf):
        pm, pw = tests(F[(F.area == a_) & (F.reward_group == "R+")].q, F[(F.area == a_) & (F.reward_group == "R-")].q)
        S.append(dict(key=a_, measure="q fine", mean_Rp=diff.loc[a_, "R+"], mean_Rm=diff.loc[a_, "R-"], p_MW=pm, p_Welch=pw))
        if pm < 0.05 and pw < 0.05:
            ax[8].text(i, 1.0, "*", transform=ax[8].get_xaxis_transform(), ha="center", fontsize=7)
    ax[8].axhline(0, color="0.6", lw=0.4, ls=":")
    ax[8].set_xticks(range(len(ordf)), ordf, rotation=90, fontsize=4)
    ax[8].set_ylabel("Δ projection W (excess)", fontsize=5.5)
    ax[8].set_title("i  fine areas (* MW & Welch < .05)", fontsize=6.5, loc="left")

    # j unit level
    Un = Un.assign(dW=Un.ev_post_W - Un.ev_pre_W)
    uj = Un.groupby(["session_id", "reward_group"]).apply(lambda g: stats.spearmanr(g.u_weight, g.dW)[0]).rename("r").reset_index()
    S += strip(ax[9], uj.assign(key="r"), "r", ["r"], ["units: u weight\nvs Δ evoked W"], "Spearman ρ per session")
    for rg in ("R+", "R-"):
        v = uj[uj.reward_group == rg].r
        S.append(dict(key="units", measure=f"rho vs 0 {rg}", mean_Rp=v.mean(), p_MW=stats.wilcoxon(v).pvalue,
                      p_Welch=stats.ttest_1samp(v, 0).pvalue))
    ax[9].set_title("j  units: u weight vs Δ evoked", fontsize=6.5, loc="left")

    # k across days
    ex = EA / "179_cd_clean_expert.parquet"
    if ex.exists():
        X = pd.read_parquet(ex)
        X = X[X.skipped_reason.isna() & (X.area == "All units") & (X.stim == "W")].copy()
        X["q"] = X.d_along - X.d_along_null
        k_ = pd.concat([W.reset_index().assign(key="learning"), X.assign(key="expert")])
        S += strip(ax[10], k_, "q", ["learning", "expert"], ["day 0\n(learners)", "expert\n(all mice)"], "Δ projection W (excess)")
        ax[10].set_title("k  across days", fontsize=6.5, loc="left")
    else:
        ax[10].text(0.5, 0.5, "expert run pending", ha="center", transform=ax[10].transAxes)

    # l behaviour
    for rg in ("R+", "R-"):
        b = W[W.reward_group == rg]
        ax[11].scatter(b.plick_W, b.q, s=6, color=COL[rg], lw=0)
        r, p = stats.spearmanr(b.plick_W, b.q)
        ax[11].text(0.02, 0.95 - 0.08 * (rg == "R-"), f"{LAB[rg]} ρ {r:+.2f}, p {p:.3f}", transform=ax[11].transAxes,
                    color=COL[rg], fontsize=4.8)
        S.append(dict(key="behaviour", measure=f"spearman q vs plick_W {rg}", mean_Rp=r, p_MW=p))
    ax[11].axhline(0, color="0.6", lw=0.4, ls=":")
    ax[11].set(xlabel="P(lick | whisker), active session", ylabel="Δ projection W (excess)")
    ax[11].set_title("l  behaviour", fontsize=6.5, loc="left")

    for a_ in axs.flat:
        a_.tick_params(labelsize=5)
    fig.suptitle(f"Clean CD pipeline (179): hit − miss axis from all active whisker trials (≥ 3 hits and misses), tracked stable "
                 f"units, 5–35 ms; sheet learners day 0 (R+ n {int((W.reward_group == 'R+').sum())}, R− n "
                 f"{int((W.reward_group == 'R-').sum())}). Excess = real − linear-shift null. Exploratory, uncorrected.",
                 fontsize=6.5, y=0.995)
    out = EA / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"180_cd_clean_overview.{ext}", dpi=250, bbox_inches="tight")
    St = pd.DataFrame(S)
    St.to_csv(EA / "180_cd_clean_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(St.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
