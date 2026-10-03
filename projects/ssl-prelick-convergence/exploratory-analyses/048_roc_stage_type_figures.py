"""Per-ROC-type (and per tuning category) publication summary of learning-stage effects (good units).

Inputs: 045 (units / roc_long), 047 (stats/*.csv), 046 (psth/*.npz).
One page per measure -> combined_results_ks4/_roc_stage_analysis/figures/types/<measure>.{png,pdf}
  a  pooled fraction significant, stacked positive (solid) + negative (hatched) = total, per cohort x stage; session
     dots; permutation p: stage within cohort (stage labels across sessions), cohort within stage (labels across
     mice); stars = total, +/- = per-sign component
  b  pooled mean |selectivity| = mean positive part + mean negative part (all tested units), same layout
  c  ECDF of signed selectivity, learning vs expert, per cohort (Wasserstein + mean |sel| permutation p)
  d  area groups, % significant per cohort (stacked by sign; * total, +/- per-sign stage perm. p < .05)
  e  area groups, mean |selectivity| (stacked by sign), same annotation
  f  fine areas (area_acronym_custom): expert - learning in % significant, 95% bootstrap CI, filled = p < .05
  g  same for mean |selectivity|
  h  Lorenz curves of % significant across fine areas (focality): Gini / entropy focality, p of the change
  i  divergence: per fine area, change in R+ vs change in R- (% significant, then |selectivity|; OLS + 95% CI band,
     solid if p < .05); black edge = learning x cohort interaction p < .05 (cohort labels permuted across mice)
  j  ANOVA (metric ~ group x area group, weighted by units; permutation p of group for % sig, % +, % -, |sel|,
     sel+, sel-; group x area permutation p) and matched mice (paired sign-flip)
  k  PSTHs (z vs pre-stimulus rate) of significant units for the conditions this measure compares,
     learning (light) vs expert (dark), per cohort and direction; mean +- SEM across units
Regions: >= 10 tested units in total and >= 3 sessions per group. No multiple-comparison correction.
"""
import argparse
import importlib
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
st47 = importlib.import_module("047_roc_stage_stats")
BASE = st47.BASE
FIG = BASE / "figures" / "types"
COH = {"R+": "#00B400", "R-": "#C800C8"}
STAGE_SHADE = {"learning": 0.45, "expert": 1.0}
CONDS_STIM = ["W_pre", "W_post", "A_pre", "A_post", "W_hit", "W_miss", "A_hit", "A_miss", "CR", "FA"]
CLABEL = {"W_pre": "whisker passive (pre)", "W_post": "whisker passive (post)", "A_pre": "auditory passive (pre)",
          "A_post": "auditory passive (post)", "W_hit": "whisker hit", "W_miss": "whisker miss", "A_hit": "auditory hit",
          "A_miss": "auditory miss", "CR": "correct rejection", "FA": "false alarm", "SPONT": "spontaneous lick",
          "W_act": "whisker (active)", "A_act": "auditory (active)", "LICK": "lick trials", "NOLICK": "no-lick trials"}
PSTH_OF = {
    "whisker_passive_pre": ["W_pre"], "whisker_passive_post": ["W_post"], "whisker_active": ["W_act"],
    "auditory_passive_pre": ["A_pre"], "auditory_passive_post": ["A_post"], "auditory_active": ["A_act"],
    "whisker_pre_vs_post_learning": ["W_pre", "W_post"], "auditory_pre_vs_post_learning": ["A_pre", "A_post"],
    "wh_vs_aud_passive_pre": ["W_pre", "A_pre"], "wh_vs_aud_passive_post": ["W_post", "A_post"],
    "wh_vs_aud_active": ["W_act", "A_act"], "wh_vs_aud_pre_vs_post_learning": ["W_pre", "A_post"],
    "spontaneous_licks": ["SPONT"], "spontaneous_licks_vs_cr": ["CR", "SPONT"], "choice": ["LICK", "NOLICK"],
    "whisker_choice": ["W_hit", "W_miss"], "auditory_choice": ["A_hit", "A_miss"], "baseline_choice": ["LICK", "NOLICK"],
    "baseline_whisker_choice": ["W_hit", "W_miss"], "baseline_auditory_choice": ["A_hit", "A_miss"],
    "baseline_pre_vs_post_learning": ["W_pre", "W_post"], "whisker_sensory": ["CR", "W_miss"],
    "auditory_sensory": ["CR", "A_miss"], "whisker_hit_vs_cr": ["CR", "W_hit"], "auditory_hit_vs_cr": ["CR", "A_hit"],
    "whisker_hit_vs_spontaneous": ["W_hit", "SPONT"], "auditory_hit_vs_spontaneous": ["A_hit", "SPONT"],
    "cat:resp_whisker": ["W_act"], "cat:resp_auditory": ["A_act"], "cat:bimodal": ["W_act", "A_act"],
    "cat:whisker_only": ["W_act", "A_act"], "cat:auditory_only": ["W_act", "A_act"], "cat:pref_whisker": ["W_act", "A_act"],
    "cat:pref_auditory": ["W_act", "A_act"], "cat:whisker_decision": ["W_hit", "W_miss"], "cat:whisker_gated": ["W_hit", "W_miss"],
    "cat:auditory_decision": ["A_hit", "A_miss"], "cat:auditory_gated": ["A_hit", "A_miss"], "cat:lick_resp": ["SPONT"],
    "cat:motor": ["CR", "SPONT"], "cat:choice_in_whisker_resp": ["W_hit", "W_miss"],
    "cat:decision_in_whisker_resp": ["W_hit", "W_miss"], "cat:choice_in_auditory_resp": ["A_hit", "A_miss"],
    "cat:mixed_n": ["W_act", "A_act"]}


def pstar(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def shade(col, f):
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(col))
    return tuple(1 - f * (1 - c))


# ------------------------------------------------------------------ data
def load_all():
    U = pd.read_parquet(BASE / "units.parquet")
    U = U[(U.quality_label == "good") & U.cohort.isin(["R+", "R-"])].reset_index(drop=True)
    U["uid"] = np.arange(len(U))
    L = pd.read_parquet(BASE / "roc_long.parquet"); L["cluster_id"] = L.cluster_id.astype(str)
    L = L.merge(U[st47.KEYS + ["uid"]], on=st47.KEYS, how="inner")
    U = U.set_index("uid")
    M = st47.build_measures(U, L)
    S = {k: pd.read_csv(BASE / "stats" / f"{k}.csv") for k in
         ["region_stats", "focality", "anova", "selectivity_distribution", "matched_mice"]}
    return U, M, S


def load_psth(U):
    """dict condition -> (N_units, bins) z-scored PSTH aligned to U index (NaN if missing); plus derived conditions"""
    P = {}
    bins = {}
    pdir = BASE / "psth"
    idx = {(s, e, c): i for i, (s, e, c) in enumerate(zip(U.session_id, U.electrode_group, U.cluster_id.astype(str)))}
    for f in sorted(pdir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        sid = f.stem
        rows = [idx.get((sid, e, c), -1) for e, c in zip(d["electrode_group"], d["cluster_id"])]
        rows = np.array(rows); ok = rows >= 0
        if not ok.any():
            continue
        sd = np.maximum(d["base_sd"], 1.0); mu = d["base_mu"]
        for k in CONDS_STIM + ["SPONT"]:
            arr = d[k]
            if arr.ndim != 2 or arr.shape[0] != len(rows):
                continue
            if k not in P:
                P[k] = np.full((len(U), arr.shape[1]), np.nan, np.float32)
                bins[k] = d["lick_bins"] if k == "SPONT" else d["stim_bins"]
            P[k][rows[ok]] = (arr[ok] - mu[ok, None]) / sd[ok, None]
            P[f"n_{k}"] = P.get(f"n_{k}", np.zeros(len(U)))
            P[f"n_{k}"][rows[ok]] = int(d[f"n_{k}"])

    def wmean(a, b):
        na, nb = P[f"n_{a}"][:, None], P[f"n_{b}"][:, None]
        with np.errstate(invalid="ignore"):
            return (np.nan_to_num(P[a]) * na + np.nan_to_num(P[b]) * nb) / np.where(na + nb > 0, na + nb, np.nan)
    if "W_hit" in P:
        P["W_act"] = wmean("W_hit", "W_miss"); P["A_act"] = wmean("A_hit", "A_miss")
        tmp = {"_l1": wmean("W_hit", "A_hit"), "_n1": P["n_W_hit"] + P["n_A_hit"], "_l2": wmean("W_miss", "A_miss"),
               "_n2": P["n_W_miss"] + P["n_A_miss"]}
        P["LICK"] = tmp["_l1"]; P["NOLICK"] = tmp["_l2"]
        for k in ["W_act", "A_act", "LICK", "NOLICK"]:
            bins[k] = bins["W_hit"]
    return P, bins


# ------------------------------------------------------------------ panels
def session_points(U, meas, value="frac"):
    d = pd.DataFrame(dict(s=U.session_id, c=U.cohort, g=U.stage, v=meas.valid.to_numpy(), f=meas.flag.to_numpy(float),
                          a=meas.abs_sel.to_numpy(float)))
    d = d[d.v]
    return d.groupby(["s", "c", "g"]).agg(frac=("f", "mean"), abs_sel=("a", "mean"), n=("f", "size")).reset_index()


METRIC_KEYS = {"frac": ("frac_sig", "frac_pos", "frac_neg", 100.0), "abs": ("mean_abs_sel", "mean_sel_pos", "mean_sel_neg", 1.0)}


def sign_stars(r, metric):
    """compact per-sign significance: '+*' / '−**' for the positive / negative component"""
    _, kp, kn, _ = METRIC_KEYS[metric]
    out = []
    for k, s in [(kp, "+"), (kn, "−")]:
        p = r.get(f"{k}_p", np.nan)
        if np.isfinite(p) and p < 0.05:
            out.append(s + pstar(p))
    return " ".join(out)


def panel_pooled(ax, U, meas, R, metric, label, cat):
    """stacked positive (solid) + negative (hatched) component = total (fraction significant or mean |sel|);
    stars: total; second line per sign (+ / −) where p < .05"""
    pts = session_points(U, meas)
    allr = R[(R.level == "all") & (R.region == "all")].set_index("comparison")
    kt, kp, kn, sc = METRIC_KEYS[metric]
    xs = {("R+", "learning"): 0, ("R+", "expert"): 1, ("R-", "learning"): 2.5, ("R-", "expert"): 3.5}
    rng = np.random.default_rng(0)
    for (c, g), x in xs.items():
        comp = f"stage:{c}"
        if comp not in allr.index:
            continue
        side = "B" if g == "expert" else "A"
        r = allr.loc[comp]
        col = shade(COH[c], STAGE_SHADE[g])
        if not cat:
            fp, fn = r[f"{kp}_{side}"] * sc, r[f"{kn}_{side}"] * sc
            ax.bar(x, fp, 0.75, color=col, edgecolor="k", lw=0.4)
            ax.bar(x, fn, 0.75, bottom=fp, color="white", edgecolor=col, hatch="////", lw=0.6)
        else:
            ax.bar(x, r[f"{kt}_{side}"] * sc, 0.75, color=col, edgecolor="k", lw=0.4)
        p = pts[(pts.c == c) & (pts.g == g)]
        y = p.frac * 100 if metric == "frac" else p.abs_sel
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(p)), y, s=4, color="0.25", alpha=0.5, lw=0, zorder=3)
    top = ax.get_ylim()[1]
    for c, (x0, x1) in [("R+", (0, 1)), ("R-", (2.5, 3.5))]:
        comp = f"stage:{c}"
        if comp in allr.index:
            r = allr.loc[comp]
            ax.plot([x0, x0, x1, x1], [top * 1.02, top * 1.05, top * 1.05, top * 1.02], color="k", lw=0.6)
            ss = "" if cat else sign_stars(r, metric)
            ax.text((x0 + x1) / 2, top * 1.06, pstar(r[f"{kt}_p"]) + (f"\n{ss}" if ss else ""), ha="center",
                    fontsize=5.5, linespacing=0.95)
    for j, (s_, (x0, x1)) in enumerate([("learning", (0, 2.5)), ("expert", (1, 3.5))]):
        comp = f"cohort:{s_}"
        if comp in allr.index:
            r = allr.loc[comp]
            ss = "" if cat else sign_stars(r, metric)
            yy = top * (1.34 + 0.13 * j)
            ax.plot([x0, x0, x1, x1], [yy - top * 0.02, yy, yy, yy - top * 0.02], color="0.45", lw=0.5)
            ax.text((x0 + x1) / 2, yy + top * 0.005, f"{s_[:5]}. R+ vs R− {pstar(r[f'{kt}_p'])}" + (f" ({ss})" if ss else ""),
                    ha="center", fontsize=4.8, color="0.35")
    ax.set_ylim(0, top * 1.62)
    ax.set_xticks(list(xs.values()), ["learn.", "expert", "learn.", "expert"], fontsize=6)
    ax.text(0.5, -0.24, "R+", transform=ax.get_xaxis_transform(), ha="center", color=COH["R+"], fontsize=7, weight="bold")
    ax.text(3.0, -0.24, "R−", transform=ax.get_xaxis_transform(), ha="center", color=COH["R-"], fontsize=7, weight="bold")
    ax.set_ylabel(label)


def panel_ecdf(ax, U, meas, D, cohort, mname):
    v = meas.valid.to_numpy() & np.isfinite(meas.sel.to_numpy(float))
    for g, ls in [("learning", "-"), ("expert", "-")]:
        m = v & (U.cohort == cohort).to_numpy() & (U.stage == g).to_numpy()
        x = np.sort(meas.sel.to_numpy(float)[m])
        if len(x):
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=shade(COH[cohort], STAGE_SHADE[g]), lw=1.2,
                    label=f"{g} (n={len(x)})")
    r = D[(D.measure == mname) & (D.comparison == f"stage:{cohort}")]
    if len(r):
        r = r.iloc[0]
        ax.set_title(f"{cohort.replace('-', chr(8722))}: W={r.wasserstein:.3f} {fmt_p(r.p_wasserstein)}\n"
                     f"Δ|sel|={r.d_mean_abs_sel:+.3f} {fmt_p(r.p_mean_abs_sel)}", fontsize=5.5)
    ax.axvline(0, color="0.6", lw=0.4); ax.set_xlabel("selectivity"); ax.legend(frameon=False, fontsize=4.8, loc="lower right")


def panel_area_group(ax, R, cohort, cat, metric="frac"):
    import ephys_utilities.allen_utils.allen_utils as au
    kt, kp, kn, sc = METRIC_KEYS[metric]
    order = au.get_area_group_custom_order()
    r = R[(R.level == "area_group") & (R.comparison == f"stage:{cohort}") & R.included].set_index("region")
    regs = [g for g in order if g in r.index] + [g for g in r.index if g not in order]
    x = np.arange(len(regs))
    for j, (g, side) in enumerate([("learning", "A"), ("expert", "B")]):
        col = shade(COH[cohort], STAGE_SHADE[g]); xx = x + (j - 0.5) * 0.38
        if cat:
            ax.bar(xx, r.loc[regs, f"{kt}_{side}"] * sc, 0.36, color=col, edgecolor="k", lw=0.3, label=g)
        else:
            fp, fn = r.loc[regs, f"{kp}_{side}"] * sc, r.loc[regs, f"{kn}_{side}"] * sc
            ax.bar(xx, fp, 0.36, color=col, edgecolor="k", lw=0.3, label=g)
            ax.bar(xx, fn, 0.36, bottom=fp, color="white", edgecolor=col, hatch="////", lw=0.4)
    top = ax.get_ylim()[1]
    for xi, reg in zip(x, regs):
        p = r.loc[reg, f"{kt}_p"]
        lab = pstar(p) if p < 0.05 else ""
        ss = "" if cat else sign_stars(r.loc[reg], metric).replace("*", "")
        lab = "\n".join([t for t in [lab, ss.replace(" ", "")] if t])
        if lab:
            ax.text(xi, top * 0.97, lab, ha="center", va="top", fontsize=5, linespacing=0.9)
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(x, [g.replace(" areas", "").replace("Somatosensory", "SS") for g in regs], rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("% significant" if metric == "frac" else "mean |selectivity|")
    what = "% significant" if metric == "frac" else "|selectivity|"
    ax.set_title(f"{cohort.replace('-', chr(8722))}: area groups, {what} (* total, +/− per-sign stage perm. p<.05)", fontsize=6)
    ax.legend(frameon=False, fontsize=5, ncol=2, loc="upper left")


def lorenz(ax, R, F, mname):
    for c in ["R+", "R-"]:
        r = R[(R.level == "area_acronym_custom") & (R.comparison == f"stage:{c}") & R.included]
        for g, side in [("learning", "A"), ("expert", "B")]:
            v = np.sort(np.clip(r[f"frac_sig_{side}"].dropna().to_numpy(), 0, None))
            if len(v) < 3 or v.sum() == 0:
                continue
            cum = np.r_[0, np.cumsum(v) / v.sum()]
            ax.plot(np.linspace(0, 1, len(cum)), cum, color=shade(COH[c], STAGE_SHADE[g]), lw=1, ls="-" if c == "R+" else "--")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5)
    txt = []
    for c in ["R+", "R-"]:
        f = F[(F.measure == mname) & (F.level == "area_acronym_custom") & (F.comparison == f"stage:{c}")]
        if "metric" in f:
            f = f[f.metric == "frac_sig"]
        if len(f):
            f = f.iloc[0]
            txt.append(f"{c.replace('-', chr(8722))}: Gini {f.gini_A:.2f}→{f.gini_B:.2f} {fmt_p(f.gini_p)}; "
                       f"H-foc. {f.entropy_focality_A:.2f}→{f.entropy_focality_B:.2f} {fmt_p(f.entropy_focality_p)}")
    ax.text(0.02, 0.98, "\n".join(t.replace("; ", "\n   ") for t in txt), transform=ax.transAxes, va="top", fontsize=4.2)
    ax.set_title("Focality across fine areas", fontsize=6)
    ax.set_xlabel("fraction of areas (sorted)"); ax.set_ylabel("cum. share of % sig.")


def forest(ax, R, cat, metric="frac"):
    import ephys_utilities.allen_utils.allen_utils as au
    a2g = {a: g for g, acs in au.get_custom_area_groups().items() for a in acs}
    order = au.get_area_group_custom_order()
    regs = sorted(set(R[(R.level == "area_acronym_custom") & R.comparison.isin(["stage:R+", "stage:R-"]) & R.included].region),
                  key=lambda r: (order.index(a2g[r]) if a2g.get(r) in order else 99, r))
    y = np.arange(len(regs))
    for k, c in enumerate(["R+", "R-"]):
        r = R[(R.level == "area_acronym_custom") & (R.comparison == f"stage:{c}") & R.included].set_index("region").reindex(regs)
        kt, _, _, sc = METRIC_KEYS[metric]
        d, lo, hi, p = r[f"{kt}_diff"] * sc, r[f"{kt}_lo"] * sc, r[f"{kt}_hi"] * sc, r[f"{kt}_p"]
        yy = y + (k - 0.5) * 0.32
        ax.errorbar(d, yy, xerr=[d - lo, hi - d], fmt="none", ecolor=COH[c], elinewidth=0.6, alpha=0.8)
        ax.scatter(d, yy, s=9, marker="o", facecolors=[COH[c] if (pp < 0.05) else "white" for pp in p.fillna(1)],
                   edgecolors=COH[c], linewidths=0.6, zorder=3, label=c.replace("-", "−"))
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(y, regs, fontsize=4.6); ax.invert_yaxis(); ax.set_ylim(len(regs) - 0.4, -0.6)
    ax.set_xlabel(("Δ % significant" if metric == "frac" else "Δ mean |selectivity|") + " (expert − learning)")
    ax.set_title(f"Fine areas, {'% significant' if metric == 'frac' else '|selectivity|'} (filled = p < .05; 95% bootstrap CI)",
                 fontsize=6)
    ax.legend(frameon=False, fontsize=5, loc="lower right")


def divergence(ax, R, mname, metric="frac"):
    import ephys_utilities.allen_utils.allen_utils as au
    a2g = {a: g for g, acs in au.get_custom_area_groups().items() for a in acs}
    gcol = au.get_custom_area_groups_colors()
    A = R[(R.level == "area_acronym_custom")]
    p_ = A[A.comparison == "stage:R+"].set_index("region"); m_ = A[A.comparison == "stage:R-"].set_index("region")
    kt, _, _, sc = METRIC_KEYS[metric]
    it = A[(A.comparison == f"interaction:{kt}")].set_index("region")
    regs = [r for r in it.index if it.loc[r, "included"]]
    if len(regs) < 4:
        ax.axis("off"); return
    x = m_.loc[regs, f"{kt}_diff"].to_numpy() * sc; y = p_.loc[regs, f"{kt}_diff"].to_numpy() * sc
    sig = it.loc[regs, f"{kt}_p"].to_numpy() < 0.05
    ax.scatter(x, y, s=12, c=[gcol.get(a2g.get(r), "0.6") for r in regs], edgecolors=np.where(sig, "k", "none"),
               linewidths=0.8, zorder=3)
    for xi, yi, r, s_ in zip(x, y, regs, sig):
        if s_:
            ax.text(xi, yi, f" {r}", fontsize=4.5, va="center")
    res = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    X = np.column_stack([np.ones_like(x), x]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta; s2 = resid @ resid / (len(x) - 2); cov = s2 * np.linalg.inv(X.T @ X)
    Xs = np.column_stack([np.ones_like(xs), xs]); yh = Xs @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs)); tq = stats.t.ppf(0.975, len(x) - 2)
    ax.fill_between(xs, yh - tq * se, yh + tq * se, color="0.6", alpha=0.25, lw=0)
    ax.plot(xs, yh, color="k", lw=1, ls="-" if res.pvalue < 0.05 else "--")
    lim = np.nanmax(np.abs(np.r_[x, y])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.4, ls=":")
    ax.axhline(0, color="0.7", lw=0.4); ax.axvline(0, color="0.7", lw=0.4)
    allit = R[(R.level == "all") & (R.comparison == f"interaction:{kt}")]
    pi = allit[f"{kt}_p"].iloc[0] if len(allit) else np.nan
    w = "% sig." if metric == "frac" else "|sel|"
    ax.set_title(f"Divergence ({w}): r={res.rvalue:.2f} {fmt_p(res.pvalue)}\npooled interaction {fmt_p(pi)}", fontsize=5.5)
    ax.set_xlabel(f"Δ {w} R− (expert − learning)"); ax.set_ylabel(f"Δ {w} R+")


def psth_panels(fig, sub, U, meas, P, bins, mname, cat):
    conds = [c for c in PSTH_OF.get(mname, []) if c in P]
    if not conds:
        return
    dirs = [("pos", "positive")] if cat else [("pos", "positive"), ("neg", "negative")]
    gs = sub.subgridspec(len(dirs), 2 * len(conds), wspace=0.35, hspace=0.7)
    for i, (dkey, dlab) in enumerate(dirs):
        flag = meas[dkey].to_numpy() & meas.valid.to_numpy()
        for j, c_ in enumerate(["R+", "R-"]):
            for k, cond in enumerate(conds):
                ax = fig.add_subplot(gs[i, j * len(conds) + k])
                b = bins[cond]; tc = (b[:-1] + np.diff(b) / 2) * 1e3
                for g in ["learning", "expert"]:
                    m = flag & (U.cohort == c_).to_numpy() & (U.stage == g).to_numpy()
                    A = P[cond][m]; A = A[np.isfinite(A).all(1)]
                    if len(A) < 3:
                        continue
                    A = np.apply_along_axis(lambda x: np.convolve(x, np.ones(4) / 4, "same"), 1, A)
                    mu, se = A.mean(0), A.std(0) / np.sqrt(len(A))
                    col = shade(COH[c_], STAGE_SHADE[g])
                    ax.fill_between(tc, mu - se, mu + se, color=col, alpha=0.25, lw=0)
                    ax.plot(tc, mu, color=col, lw=0.9, label=f"{g} n={len(A)}")
                ax.axvline(0, color="k", lw=0.4); ax.axhline(0, color="0.7", lw=0.3)
                if cond != "SPONT":
                    ax.axvspan(5, 35, color="#FDD49E", alpha=0.5, lw=0)
                ax.set_title(f"{c_.replace('-', chr(8722))} {dlab}: {CLABEL.get(cond, cond)}", fontsize=4.8)
                ax.tick_params(labelsize=4.5); ax.legend(frameon=False, fontsize=3.8, loc="upper right")
                if i == len(dirs) - 1:
                    ax.set_xlabel("ms from lick" if cond == "SPONT" else "ms from stimulus", fontsize=5)
                if j == 0 and k == 0:
                    ax.set_ylabel("z", fontsize=5)


ANOVA_SHORT = [("frac_sig", "%sig"), ("frac_pos", "%+"), ("frac_neg", "%−"), ("mean_abs_sel", "|s|"),
               ("mean_sel_pos", "s+"), ("mean_sel_neg", "s−")]


def text_panel(ax, A, Mm, mname):
    ax.axis("off")
    a = A[A.measure == mname]
    if "metric" not in a:
        a = a.assign(metric="frac_sig")
    mets = [(k, s) for k, s in ANOVA_SHORT if k in set(a.metric)]
    lines = ["ANOVA metric ~ group x area group", "perm. p of group (stage / cohort):",
             "        " + "".join(f"{s:>5}" for _, s in mets)]
    for comp in a.comparison.drop_duplicates():
        row = f"{comp.replace('cohort:', 'c:').replace('stage:', 's:').replace('learning', 'learn').replace('-', '−'):<8}"
        for k, _ in mets:
            r = a[(a.comparison == comp) & (a.metric == k)]
            row += f"{r.p_perm_group.iloc[0]:>5.2g}" if len(r) else f"{'':>5}"
        lines.append(row)
    lines.append("x area perm. p (%sig / |s|):")
    for comp in a.comparison.drop_duplicates():
        vals = [a[(a.comparison == comp) & (a.metric == k)].p_perm_interaction for k in ["frac_sig", "mean_abs_sel"]]
        lines.append(f"{comp.replace('cohort:', 'c:').replace('stage:', 's:').replace('learning', 'learn').replace('-', '−'):<8}" + "".join(
            f"{v.iloc[0]:>5.2g}" if len(v) else f"{'':>5}" for v in vals))
    lines.append("")
    lines.append("Matched mice, paired sign-flip")
    for _, r in Mm[Mm.measure == mname].iterrows():
        lines.append(f"{r.cohort} {r.metric.replace('mean_abs_sel', '|s|')} n={int(r.n_mice)}: Δ={r.mean_diff:+.3f} p={r.p_signflip:.2g}")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=3.9, transform=ax.transAxes)


def make_figure(args):
    mname, U, meas, S, P, bins = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.5, "pdf.fonttype": 42, "hatch.linewidth": 0.4})
    R = S["region_stats"][S["region_stats"].measure == mname]
    cat = mname.startswith("cat:")
    fig = plt.figure(figsize=(7.4, 13.8 if not cat else 11.6))
    hr = [1, 1, 1, 1.45, 1.05, 1.35] if not cat else [1, 1, 0.001, 1.45, 1.05, 1.35]
    gs = fig.add_gridspec(6, 4, height_ratios=hr, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.95, bottom=0.035)

    def lab(ax, t, x=-0.25, y=1.12):
        ax.text(x, y, t, transform=ax.transAxes, fontsize=8, weight="bold")
    ax = fig.add_subplot(gs[0, 0]); panel_pooled(ax, U, meas, R, "frac", "% significant" if not cat else "% of units", cat)
    ax.set_title("pooled % sig. (hatched = neg.)" if not cat else "pooled fraction", fontsize=5.5); lab(ax, "a", -0.3, 1.3)
    if mname == "cat:mixed_n" or not cat:
        ax = fig.add_subplot(gs[0, 1]); panel_pooled(ax, U, meas, R, "abs", "mean |selectivity|" if not cat else "mean # sig. types", cat)
        ax.set_title("pooled |sel| = sel+ + |sel−|" if not cat else "pooled", fontsize=5.5); lab(ax, "b", -0.3, 1.3)
    if not cat:
        for j, c in enumerate(["R+", "R-"]):
            ax = fig.add_subplot(gs[0, 2 + j]); panel_ecdf(ax, U, meas, S["selectivity_distribution"], c, mname)
            if j == 0:
                lab(ax, "c", -0.25, 1.22)
    for j, c in enumerate(["R+", "R-"]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); panel_area_group(ax, R, c, cat, "frac")
        if j == 0:
            lab(ax, "d", -0.1, 1.08)
        if not cat:
            ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2]); panel_area_group(ax, R, c, cat, "abs")
            if j == 0:
                lab(ax, "e", -0.1, 1.08)
    ax = fig.add_subplot(gs[3, 0:2]); forest(ax, R, cat, "frac"); lab(ax, "f", -0.15, 1.04)
    if not cat:
        ax = fig.add_subplot(gs[3, 2:4]); forest(ax, R, cat, "abs"); lab(ax, "g", -0.15, 1.04)
    ax = fig.add_subplot(gs[4, 0]); lorenz(ax, R, S["focality"], mname); lab(ax, "h", -0.3, 1.2)
    ax = fig.add_subplot(gs[4, 1]); divergence(ax, R, mname, "frac"); lab(ax, "i", -0.3, 1.15)
    if not cat:
        ax = fig.add_subplot(gs[4, 2]); divergence(ax, R, mname, "abs")
    ax = fig.add_subplot(gs[4, 3]); text_panel(ax, S["anova"], S["matched_mice"], mname); lab(ax, "j", -0.08, 1.05)
    psth_panels(fig, gs[5, 0:4], U, meas, P, bins, mname, cat)
    bb = gs[5, 0].get_position(fig)
    fig.text(0.015, bb.y1 + 0.008, "k", fontsize=8, weight="bold")
    pos = importlib.import_module("045_roc_stage_table").POS_MEANING.get(mname, "")
    fig.suptitle(f"{mname}  —  learning (day 0) vs expert (day ≥ 1), good units"
                 + (f"; positive = {pos}" if pos else ""), fontsize=8, y=0.99)
    safe = mname.replace(":", "_")
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{safe}.{ext}", dpi=220)
    plt.close(fig)
    return mname


def main(a):
    FIG.mkdir(parents=True, exist_ok=True)
    U, M, S = load_all()
    P, bins = load_psth(U)
    names = a.measures or list(M)
    for n in names:                                    # sequential: shared big arrays (PSTHs) stay in one process
        make_figure((n, U, M[n], S, P, bins))
        print("saved", n, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measures", nargs="*", default=None)
    main(ap.parse_args())
