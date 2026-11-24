"""Figure 4 (rows 3+) and its supplement: passive responses on the hit - miss coding direction (CD). Shared compute_<key>()
(returns the panel CSV, fills META) and plot_<key>(ax, df) for figures/Figure4_passive_cd and figures/SuppFigure4_passive_cd.
User 2026-11-23: one clean pipeline; v2 the same day in the style of the COSYNE 2027 figures (Posters/cosyne_2027,
rows O-S): mean +- s.e.m. without session dots, pre -> post with the linear-shift null as open markers, post - pre excess per
modality with the cohort x modality interaction, areas as a horizontal dot plot in the area palette; skills/ssl-figure-style.

Input: ssl-hitmiss-axis-passive 179_cd_clean.py tables (combined_results_ks4/ssl-hitmiss-axis-passive/tables).
Sessions: day 0 = learners (sheet CATEGORY_COLUMN good / moderate), one session per mouse; expert = every recorded expert
session. Excess = real - linear-shift null. R+ vs R-: Mann-Whitney U and Welch t; pre vs post within a cohort: Wilcoxon
signed-rank and paired t; panels mark the larger P of the two (both in the meta / caption); uncorrected. Cohort x modality:
cohort labels permuted across sessions (N_PERM).
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import REFERENCE_SHEET, axel_bisi_root, two_group_tests
from paperstyle import COLORS, light

TAB = axel_bisi_root() / "combined_results_ks4" / "ssl-hitmiss-axis-passive" / "tables"
CATEGORY_COLUMN = "learning_category"   # as panels A-G; learning_category_new (2026-11-19) adds AB141, MH009, MH026 (R+)
N_PERM = 10000
META = {}
COH = ("R+", "R-")
LAB = {"R+": "R+", "R-": "R−"}
W_COL, A_COL, MISS_COL = COLORS["whisker"], COLORS["auditory"], "#8c5a1e"
FS = 5
_c = {}


# ------------------------------------------------------------------ data

def learners():
    ref = pd.read_excel(REFERENCE_SHEET)
    ref = ref[(ref.exclude == 0) & ref.reward_group.isin(COH)]
    return set(ref[ref[CATEGORY_COLUMN].isin(["good", "moderate"])].mouse_id)


def area_palette():
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
    from ephys_utilities.allen_utils import allen_utils as AU
    return AU.get_custom_area_groups_colors(), AU.get_area_group_custom_order(), AU.get_custom_area_groups_from_name()


def clean(stage="learning"):
    key = f"clean_{stage}"
    if key not in _c:
        d = pd.read_parquet(TAB / f"179_cd_clean{'' if stage == 'learning' else '_' + stage}.parquet")
        d = d[d.skipped_reason.isna()].copy()
        if stage == "learning":
            d = d[d.mouse_id.isin(learners())]
        d["q"] = d.d_along - d.d_along_null
        d["qth"] = d.dtheta - d.dtheta_null
        d["cohort"] = d.reward_group
        _c[key] = d
    return _c[key]


def whole(stim="W", stage="learning"):
    d = clean(stage)
    return d[(d.area == "All units") & (d.stim == stim)].copy()


def tests(df, col, by="cohort"):
    a, b = (df[df[by] == c][col] for c in COH)
    t = two_group_tests(a, b)
    return dict(mean_Rp=float(np.nanmean(a)), sem_Rp=float(stats.sem(a, nan_policy="omit")),
                mean_Rm=float(np.nanmean(b)), sem_Rm=float(stats.sem(b, nan_policy="omit")), **t)


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return dict(n=int(ok.sum()), wilcoxon_p=float(stats.wilcoxon(a[ok], b[ok]).pvalue),
                paired_t_p=float(stats.ttest_rel(a[ok], b[ok]).pvalue))


def perm_interaction(v, g, n=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    v, g = np.asarray(v, float), np.asarray(g)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    d = lambda gg: v[gg == "R+"].mean() - v[gg == "R-"].mean()
    obs = d(g)
    null = np.array([d(rng.permutation(g)) for _ in range(n)])
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def wa(col):
    w, a = whole("W").set_index("session_id"), whole("A").set_index("session_id")
    out = w[["mouse_id", "cohort"]].copy()
    out["W"], out["A"] = w[col], a[col].reindex(w.index)
    out["WA"] = out.W - out.A
    return out.reset_index()


def theta(c):
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ------------------------------------------------------------------ compute (main)

def compute_time():
    t = pd.read_parquet(TAB / "179_cd_time.parquet")
    t = t[(t.area == "All units") & t.mouse_id.isin(learners())]
    g = t.groupby(["reward_group", "cls", "t"]).value.agg(["mean", "sem", "count"]).reset_index().rename(columns={"reward_group": "cohort"})
    META["time"] = {c: int(t[t.reward_group == c].session_id.nunique()) for c in COH}
    return g


def _prepost(pre, post, pre0, post0, name, stim="W"):
    """long table session x epoch x kind (data / null) + within-cohort pre vs post and R+ vs R- excess tests"""
    w = whole(stim)
    rows = []
    for kind, (a, b) in (("data", (pre, post)), ("shifted", (pre0, post0))):
        for ep, f in (("pre", a), ("post", b)):
            rows.append(pd.DataFrame(dict(session_id=w.session_id, mouse_id=w.mouse_id, cohort=w.cohort, kind=kind, epoch=ep,
                                          value=f(w))))
    d = pd.concat(rows, ignore_index=True)
    meta = {}
    for c in COH:
        x = w[w.cohort == c]
        meta[c] = dict(data=paired(pre(x), post(x)), shifted=paired(pre0(x), post0(x)))
    ex = w.assign(ex=(post(w) - pre(w)) - (post0(w) - pre0(w)))
    meta["excess"] = tests(ex, "ex")
    meta["pre_data"] = tests(w.assign(v=pre(w)), "v")
    META[name] = meta
    return d


def compute_projection_prepost(stim="W", name="projection_prepost"):
    return _prepost(lambda x: x.a_pre, lambda x: x.a_post, lambda x: x.a_pre_null, lambda x: x.a_post_null, name, stim=stim)


def compute_angle_prepost(stim="W", name="angle_prepost"):
    return _prepost(lambda x: theta(x.cos_pre), lambda x: theta(x.cos_post), lambda x: theta(x.cos_pre_null),
                    lambda x: theta(x.cos_post_null), name, stim=stim)


def compute_delta():
    d = wa("q").melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=["W", "A"], var_name="stim")
    obs, p = perm_interaction(wa("q").WA, wa("q").cohort)
    META["delta"] = dict(W=tests(d[d.stim == "W"], "value"), A=tests(d[d.stim == "A"], "value"),
                         interaction=dict(diff=obs, p=p))
    return d


def compute_gain_rotation(stim="W", name="gain_rotation"):
    """excess decomposition: a = rho cos; real and null share rho -> excess = d_rho (cos_pre - cos_pre_null)
    [gain] + rho_post (d_cos - d_cos_null) [rotation]; the two add up to the excess projection change"""
    w = whole(stim)
    w["gain"] = (w.rho_post - w.rho_pre) * (w.cos_pre - w.cos_pre_null)
    w["rotation"] = w.rho_post * ((w.cos_post - w.cos_pre) - (w.cos_post_null - w.cos_pre_null))
    d = w.melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=["gain", "rotation"], var_name="term")
    META[name] = {k: tests(d[d.term == k], "value") for k in ("gain", "rotation")}
    META[name]["check_max_abs_residual"] = float(np.nanmax(np.abs(w.gain + w.rotation - w.q)))
    obs, p = perm_interaction((w.rotation - w.gain).to_numpy(), w.cohort.to_numpy())
    META[name]["interaction"] = dict(diff=obs, p=p)
    return d


def compute_areas(stim="W", name="areas"):
    col, order, _ = area_palette()
    G = clean()
    G = G[(G.area_col == "area_group") & (G.stim == stim)]
    n = G.groupby(["area", "cohort"]).session_id.nunique().unstack().fillna(0)
    keep = [a for a in order if a in n.index and n.loc[a].min() >= 3]
    rows = []
    for a in keep:
        t = tests(G[G.area == a], "q")
        rows.append(dict(area=a, color=col[a], **{k: t[k] for k in ("mean_Rp", "sem_Rp", "mean_Rm", "sem_Rm", "n_R+", "n_R-",
                                                                    "mannwhitney_p", "welch_p")}))
    d = pd.DataFrame(rows)
    g = G[G.area.isin(keep)]
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    fit = smf.ols("q ~ C(cohort) + C(area)", data=g).fit()
    META[name] = dict(n_areas=len(d), anova_cohort_p=float(anova_lm(fit, typ=2).loc["C(cohort)", "PR(>F)"]),
                         n_both=int(((d.mannwhitney_p < 0.05) & (d.welch_p < 0.05)).sum()))
    return d


def compute_area_link(stim="W", name="area_link"):
    col, order, _ = area_palette()
    G = clean()
    G = G[(G.area_col == "area_group") & (G.stim == stim)].assign(dev=lambda x: x.ev_post - x.ev_pre)
    n = G.groupby(["area", "cohort"]).session_id.nunique().unstack().fillna(0)
    keep = n[n.min(1) >= 3].index
    G = G[G.area.isin(keep)]
    m = G.groupby(["area", "cohort"]).agg(dev=("dev", "mean"), dev_sem=("dev", "sem"), q=("q", "mean"), q_sem=("q", "sem"),
                                         n=("session_id", "nunique")).reset_index()
    m["color"] = m.area.map(col)
    meta = dict(n_areas=int(len(keep)))
    for c in COH:
        g = m[m.cohort == c]
        lr = stats.linregress(g.dev, g.q)
        sp = stats.spearmanr(g.dev, g.q)
        meta[c] = dict(r=float(lr.rvalue), p=float(lr.pvalue), slope=float(lr.slope), rho=float(sp[0]), p_spearman=float(sp[1]))
    META[name] = meta
    return m


def compute_across_days(stim="W", name="across_days"):
    a = whole(stim).assign(stage="day 0")
    b = whole(stim, "expert").assign(stage="expert")
    d = pd.concat([a, b])[["session_id", "mouse_id", "cohort", "stage", "q", "qth"]]
    META[name] = {s: dict(q=tests(d[d.stage == s], "q"),
                                   n_mice_rp=int(d[(d.stage == s) & (d.cohort == "R+")].mouse_id.nunique()),
                                   n_mice_rm=int(d[(d.stage == s) & (d.cohort == "R-")].mouse_id.nunique()))
                           for s in ("day 0", "expert")}
    return d


# ------------------------------------------------------------------ compute (supp)

def compute_data_vs_shifted(stim="W", name="data_vs_shifted"):
    """user 2026-11-24: the whisker change (post - pre) on the data axis, on the shifted-label axes, and their difference
    (data - shifted, the main-figure measure), for the projection and the angle"""
    w = whole(stim)
    parts = dict(projection=(w.a_post - w.a_pre, w.a_post_null - w.a_pre_null),
                 angle=(theta(w.cos_post) - theta(w.cos_pre), theta(w.cos_post_null) - theta(w.cos_pre_null)))
    rows, meta = [], {}
    for m, (dat, sh) in parts.items():
        for comp, v in (("data", dat), ("shifted", sh), ("excess", dat - sh)):
            d = pd.DataFrame(dict(session_id=w.session_id, mouse_id=w.mouse_id, cohort=w.cohort, measure=m, component=comp,
                                  value=v))
            rows.append(d)
            meta.setdefault(m, {})[comp] = tests(d, "value")
    META[name] = meta
    return pd.concat(rows, ignore_index=True)


# ---- whisker | auditory side by side (user 2026-11-24: "side by side main whisker figures and auditory figures, matching
# axes everywhere"): one CSV with a stim column, META[name] = {"W": ..., "A": ...}

def _wa_compute(fn, name):
    frames, meta = [], {}
    for s in ("W", "A"):
        frames.append(fn(s, f"_{name}_{s}").assign(stim=s))
        meta[s] = META.pop(f"_{name}_{s}")
    META[name] = meta
    return pd.concat(frames, ignore_index=True)


def compute_wa_time():
    d = compute_time()
    META["wa_time"] = META["time"]
    return d


def compute_wa_projection_prepost():
    return _wa_compute(compute_projection_prepost, "wa_projection_prepost")


def compute_wa_angle_prepost():
    return _wa_compute(compute_angle_prepost, "wa_angle_prepost")


def compute_wa_gain_rotation():
    return _wa_compute(compute_gain_rotation, "wa_gain_rotation")


def compute_wa_areas():
    return _wa_compute(compute_areas, "wa_areas")


def compute_wa_area_link():
    return _wa_compute(compute_area_link, "wa_area_link")


def compute_wa_across_days():
    return _wa_compute(compute_across_days, "wa_across_days")


def compute_wa_data_vs_shifted():
    return _wa_compute(compute_data_vs_shifted, "wa_data_vs_shifted")


def compute_wa_area_types():
    """user 2026-11-24 ("next E, add within-cohort comparison of trial types, then within-cohort difference between trial
    types"): per area group and cohort, passive whisker vs auditory change (post - pre; data - shifted) in the same sessions
    (paired: Wilcoxon + paired t), and the whisker - auditory difference (R+ vs R-: Mann-Whitney + Welch = cohort x modality
    per area). Areas: >= 3 sessions per cohort with both modalities (same unit threshold as Fig. 4N)."""
    col, order, _ = area_palette()
    G = clean()
    G = G[G.area_col == "area_group"].pivot_table(index=["session_id", "cohort", "area"], columns="stim", values="q").dropna()
    G = G.reset_index().assign(D=lambda x: x.W - x.A)
    n = G.groupby(["area", "cohort"]).session_id.nunique().unstack().fillna(0)
    keep = [a for a in order if a in n.index and n.loc[a].min() >= 3]
    rows = []
    for a in keep:
        g = G[G.area == a]
        r = dict(area=a, color=col[a])
        for c, k in (("R+", "Rp"), ("R-", "Rm")):
            x = g[g.cohort == c]
            for v in ("W", "A", "D"):
                r[f"mean_{v}_{k}"], r[f"sem_{v}_{k}"] = mean_sem(x[v])
            pr = paired(x.W, x.A)
            r[f"n_{k}"], r[f"p_WA_{k}"] = pr["n"], pmax(pr["wilcoxon_p"], pr["paired_t_p"])
        t = two_group_tests(g[g.cohort == "R+"].D, g[g.cohort == "R-"].D)
        r["p_D_cohort"] = pmax(t["mannwhitney_p"], t["welch_p"])
        rows.append(r)
    d = pd.DataFrame(rows)
    META["wa_area_types"] = dict(n_areas=len(d), **{f"n_WA_{k}": int((d[f"p_WA_{k}"] < 0.05).sum()) for k in ("Rp", "Rm")},
                                 n_D_cohort=int((d.p_D_cohort < 0.05).sum()))
    return d


def compute_wa_area_diff():
    d = compute_wa_area_types()
    META["wa_area_diff"] = META.pop("wa_area_types")
    return d


def compute_wa_decoder():
    """user 2026-11-24 ("plot the current decoder results in the supp figure so that I can check"): ssl-hitmiss-axis-passive
    146 as it stands (NOT rerun on the 179 sessions): L2 logistic hit vs miss decoder on active whisker trials (z-scored
    responses, balanced subsamples, held-out scoring), passive trials scored; standardised readout = (score - hit / miss
    midpoint) / SD of the held-out active whisker scores (> 0 hit side); linear-shift null stored only for the post - pre
    change (shift_excess_d{W,A} = real change - mean of shifted-label decoders). Whole brain, tracked stable units,
    epoch baseline, learners."""
    d = pd.read_parquet(TAB / "146_choice_axis_readout.parquet")
    d = d[d.skipped_reason.isna() & (d.unit_set == "stable") & (d.area == "whole_brain") & (d.response == "epochbase")
          & d.mouse_id.isin(learners())].assign(cohort=lambda x: x.reward_group)
    rows, meta = [], {"n": {c: int((d.cohort == c).sum()) for c in COH},
                      "n_in_179": int(d.session_id.isin(whole().session_id).sum()), "bal_acc": tests(d, "decoder_bal_acc")}
    for s in ("W", "A"):
        for ep in ("pre", "post"):
            rows.append(pd.DataFrame(dict(session_id=d.session_id, mouse_id=d.mouse_id, cohort=d.cohort, stim=s, kind="data",
                                          epoch=ep, value=d[f"ro_std_passive_{ep}_{s}"])))
        rows.append(pd.DataFrame(dict(session_id=d.session_id, mouse_id=d.mouse_id, cohort=d.cohort, stim=s, kind="excess",
                                      epoch="change", value=d[f"shift_excess_d{s}"])))
        meta[s] = {c: dict(data=paired(d[d.cohort == c][f"ro_std_passive_pre_{s}"], d[d.cohort == c][f"ro_std_passive_post_{s}"]))
                   for c in COH}
        meta[s]["excess"] = tests(d, f"shift_excess_d{s}")
    META["wa_decoder"] = meta
    return pd.concat(rows, ignore_index=True)


def compute_axis_quality():
    w = whole("W")[["session_id", "mouse_id", "cohort", "n_hit", "n_miss", "n_units", "dprime_loo", "n_shifts"]]
    META["axis_quality"] = dict(dprime=tests(w, "dprime_loo"), min_hit=int(w.n_hit.min()), min_miss=int(w.n_miss.min()),
                                units_median=float(w.n_units.median()), units_min=int(w.n_units.min()))
    return w


def compute_stimulus_axis():
    w = whole("W")
    w["ds"] = w.s_post - w.s_pre
    d = w.melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=["ds", "cos_u_s"], var_name="measure")
    META["stimulus_axis"] = {k: tests(d[d.measure == k], "value") for k in ("ds", "cos_u_s")}
    return d


def compute_units():
    u = pd.read_parquet(TAB / "179_cd_units.parquet")
    u = u[u.mouse_id.isin(learners())]
    u["dW"] = u.ev_post_W - u.ev_pre_W
    r = (u.groupby(["session_id", "mouse_id", "reward_group"])
         .apply(lambda g: stats.spearmanr(g.u_weight, g.dW)[0]).rename("rho").reset_index().rename(columns={"reward_group": "cohort"}))
    meta = dict(cohorts=tests(r, "rho"), n_units=int(len(u)))
    for c in COH:
        v = r[r.cohort == c].rho
        meta[c] = dict(mean=float(v.mean()), p_wilcoxon=float(stats.wilcoxon(v).pvalue), p_t=float(stats.ttest_1samp(v, 0).pvalue))
    META["units"] = meta
    return r


def compute_fine_areas():
    col, order, _ = area_palette()
    u = pd.read_parquet(TAB / "179_cd_units.parquet", columns=["area_group", "area_acronym_custom"]).dropna()
    grp = u.groupby("area_acronym_custom").area_group.agg(lambda x: x.mode().iloc[0]).to_dict()   # labels table membership
    F = clean()
    F = F[(F.area_col == "area_acronym_custom") & (F.stim == "W")]
    n = F.groupby(["area", "cohort"]).session_id.nunique().unstack().fillna(0)
    keep = n[n.min(1) >= 4].index
    rows = []
    for a in keep:
        t = tests(F[F.area == a], "q")
        g = grp.get(a)
        if g is None or g not in col:
            raise KeyError(f"fine area {a} has no area group colour")
        rows.append(dict(area=a, group=g, group_rank=order.index(g), color=col[g],
                         **{k: t[k] for k in ("mean_Rp", "sem_Rp", "mean_Rm", "sem_Rm", "n_R+", "n_R-", "mannwhitney_p", "welch_p")}))
    d = pd.DataFrame(rows).sort_values(["group_rank", "area"]).reset_index(drop=True)
    META["fine_areas"] = dict(n_areas=int(len(d)), n_both=int(((d.mannwhitney_p < 0.05) & (d.welch_p < 0.05)).sum()),
                              n_one=int(((d.mannwhitney_p < 0.05) | (d.welch_p < 0.05)).sum()))
    return d


def compute_behaviour():
    w = whole("W")[["session_id", "mouse_id", "cohort", "plick_W", "q", "qth"]]
    meta = {}
    for c in COH:
        g = w[w.cohort == c]
        lr = stats.linregress(g.plick_W, g.q)
        r2, p2 = stats.spearmanr(g.plick_W, g.qth, nan_policy="omit")
        meta[c] = dict(r=float(lr.rvalue), p=float(lr.pvalue), rho_angle=float(r2), p_angle=float(p2), n=int(len(g)))
    META["behaviour"] = meta
    return w


# ------------------------------------------------------------------ plot helpers (skills/ssl-figure-style)

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}" if p < 0.01 else f"p = {p:.2f}"


def sig(ax, x, y, p, color="k", rotate=False, ha="center"):
    """stars with the p-value right above (one significance style, skills/ssl-figure-style rule 10)"""
    if p is None or not np.isfinite(p):
        return
    kw = dict(fontsize=FS - 0.5, color=color, annotation_clip=False, textcoords="offset points")
    if rotate:
        ax.annotate(f"{stars(p)}  {p_text(p)}", (x, y), xytext=(0, 1), ha="center", va="bottom", rotation=90, **kw)
        return
    s = stars(p)
    ax.annotate(s, (x, y), xytext=(0, 0), ha=ha, va="bottom", **kw)
    ax.annotate(p_text(p), (x, y), xytext=(0, 5.0 if s == "n.s." else 4.0), ha=ha, va="bottom", **kw)


def bracket(ax, x0, x1, y, p, color="k"):
    ax.plot([x0, x1], [y, y], color=color, lw=0.4, clip_on=False)
    sig(ax, (x0 + x1) / 2, y, p, color)


def vbracket(ax, x, y0, y1, p, color="k"):
    ax.plot([x, x], [y0, y1], color=color, lw=0.4, clip_on=False)
    ax.annotate(f"{p_text(p)}\n{stars(p)}", (x, (y0 + y1) / 2), xytext=(2, 0), textcoords="offset points", ha="left",
                va="center", fontsize=FS - 0.5, color=color, annotation_clip=False, linespacing=1.0)


def pmax(*ps):
    return max(ps)


def mean_sem(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (v.mean(), stats.sem(v)) if len(v) > 1 else (np.nan, np.nan)


def ols_band(ax, x, y, color):
    """rule 5: OLS line (solid if p < 0.05, dashed otherwise) + analytic 95 % band"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    lr = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    n = len(x)
    yh = lr.intercept + lr.slope * xs
    s_err = np.sqrt(np.sum((y - (lr.intercept + lr.slope * x)) ** 2) / (n - 2))
    band = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xs, yh - band, yh + band, color=light(color, 0.85), lw=0, edgecolor="none", zorder=0)
    ax.plot(xs, yh, color=color, lw=0.7, ls="-" if lr.pvalue < 0.05 else (0, (3, 2)), zorder=1)
    return lr


def cohort_ticks(ax):
    ax.set_xticks([0, 1], ["R+", "R−"])
    for t_, c in zip(ax.get_xticklabels(), COH):
        t_.set_color(COLORS[c])


# ------------------------------------------------------------------ plot (main)

MOD = {"W": ("Whisker", "whisker"), "A": ("Auditory", "auditory")}


def _pair(ax, gap=0.12, n=2):
    """n side-by-side insets filling ax (ax itself hidden)"""
    ax.axis("off")
    w = (1 - gap * (n - 1)) / n
    return [ax.inset_axes([j * (w + gap), 0, w, 1]) for j in range(n)]


def _mod_title(a, s, pad=2):
    """pad (points): ~16 above panels whose p-value brackets sit above the axes"""
    a.set_title(MOD[s][0], color=W_COL if s == "W" else A_COL, fontsize=FS + 0.5, pad=pad)


def _ms_range(df, keys, val="value"):
    g = df.groupby(keys)[val].agg(["mean", "sem"])
    return float((g["mean"] - g["sem"].fillna(0)).min()), float((g["mean"] + g["sem"].fillna(0)).max())


def _padded(lo, hi, f=0.06, zero=True):
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    s = (hi - lo) or 1.0
    return lo - f * s, hi + f * s


def plot_time(ax, df, stims=("W",)):
    """R+ (top) / R- (bottom); one column per passive modality (active whisker hit / miss in every column, as reference);
    shared y axis within a row"""
    ax.axis("off")
    ncol, gap = len(stims), 0.1
    w = (1 - gap * (ncol - 1)) / ncol
    for k, c in enumerate(COH):
        g0 = df[df.cohort == c]
        row = []
        for j, s in enumerate(stims):
            a = ax.inset_axes([j * (w + gap), 0.58 - 0.58 * k, w, 0.42])
            row.append(a)
            for cls, col in (("hit", W_COL), ("miss", MISS_COL), (f"pre_{s}", "0.55"), (f"post_{s}", COLORS[c])):
                g = g0[g0.cls == cls].sort_values("t")
                x = g.t.to_numpy()
                a.fill_between(x, g["mean"] - g["sem"], g["mean"] + g["sem"], color=light(col, 0.7), lw=0, edgecolor="none")
                a.plot(x, g["mean"], color=col, lw=0.7)
            a.axvspan(0.005, 0.035, facecolor="0.92", lw=0, edgecolor="none", zorder=0)
            a.axvline(0, color="0.4", lw=0.4, ls=(0, (2, 2)))
            a.axhline(0, color="0.6", lw=0.4)
            a.set_xlim(-0.1, 0.2)
            a.set_xticks([-0.1, 0, 0.1, 0.2])
            if j == 0:
                n = int(g0[g0.cls == f"pre_{s}"]["count"].max())
                a.text(0.02, 0.98, f"{LAB[c]} ({n})", transform=a.transAxes, ha="left", va="top", color=COLORS[c],
                       fontsize=FS + 0.5)
            if k == 0 and ncol > 1:
                _mod_title(a, s)
            y = 0.98
            act = (("Active whisker hit" if ncol > 1 else "Active hit", W_COL),
                   ("Active whisker miss" if ncol > 1 else "Active miss", MISS_COL))
            labs = (act if k == 0 else ()) + (("Passive pre", "0.55"), ("Passive post", COLORS[c]))
            for lab, col in labs:
                a.text(1.0, y, lab, transform=a.transAxes, ha="right", va="top", fontsize=FS - 0.8, color=col)
                y -= 0.085
            if k == 0:
                a.set_xticklabels([])
            else:
                a.set_xlabel("Time from stimulus (s)")
            a.tick_params(labelsize=FS)
        lo = min(a.get_ylim()[0] for a in row)
        hi = max(a.get_ylim()[1] for a in row)
        if k == 0:
            hi = hi + 0.45 * (hi - lo)                 # room for the labels above the traces
        for a in row:
            a.set_ylim(lo, hi)
    ax.annotate("Proj. on hit − miss axis", xy=(0, 0.5), xycoords="axes fraction", xytext=(-19, 0),
                textcoords="offset points", rotation=90, ha="center", va="center", fontsize=FS + 0.5)   # one label, both rows


def _prepost_plot(ax, df, ylabel, ylim=None, legend=True):
    for c in COH:
        off = -0.06 if c == "R+" else 0.06
        x = np.array([0, 1]) + off
        for kind in ("shifted", "data"):
            g = df[(df.cohort == c) & (df.kind == kind)]
            m = [mean_sem(g[g.epoch == e].value) for e in ("pre", "post")]
            ys, es = [v[0] for v in m], [v[1] for v in m]
            if kind == "data":
                ax.errorbar(x, ys, es, color=COLORS[c], marker="o", ms=3, lw=1.0, capsize=0, zorder=3)
            else:
                ax.plot(x, ys, color="0.65", lw=0.7, zorder=1)
                ax.errorbar(x, ys, es, color=COLORS[c], marker="o", ms=3, mfc="white", mew=0.7, lw=0, elinewidth=0.6,
                            capsize=0, zorder=2)
    if ylim is not None:
        ax.set_ylim(ylim)
    yl = ax.get_ylim()
    span = yl[1] - yl[0]
    for k, c in enumerate(COH):
        g = df[(df.cohort == c) & (df.kind == "data")].pivot(index="session_id", columns="epoch", values="value").dropna()
        p = pmax(stats.wilcoxon(g.pre, g.post).pvalue, stats.ttest_rel(g.pre, g.post).pvalue)
        bracket(ax, 0, 1, yl[1] + span * (0.04 + 0.24 * k), p, COLORS[c])
    if (df.kind == "excess").any():          # null stored only as the change (146 decoder): data - shifted change given
        e = df[df.kind == "excess"]
        t = two_group_tests(e[e.cohort == "R+"].value, e[e.cohort == "R-"].value)
    else:
        d = df.pivot_table(index=["session_id", "cohort"], columns=["kind", "epoch"], values="value").reset_index()
        ex = (d[("data", "post")] - d[("data", "pre")]) - (d[("shifted", "post")] - d[("shifted", "pre")])
        t = two_group_tests(ex[d.cohort == "R+"], ex[d.cohort == "R-"])
    post = [mean_sem(df[(df.cohort == c) & (df.kind == "data") & (df.epoch == "post")].value)[0] for c in COH]
    vbracket(ax, 1.2, min(post), max(post), pmax(t["mannwhitney_p"], t["welch_p"]))
    ax.set_ylim(yl[0], yl[1] + span * 0.3)
    ax.set_xlim(-0.3, 1.75)
    ax.set_xticks([0, 1], ["Pre", "Post"])
    ax.set_ylabel(ylabel)
    if not legend:
        return
    ax.plot([], [], color="0.2", marker="o", ms=2.5, lw=1, label="Data")
    ax.plot([], [], color="0.65", marker="o", ms=2.5, mfc="white", lw=0.7, label="Linear shifts")
    ax.legend(fontsize=FS - 0.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, handlelength=1.4,
              columnspacing=0.8, borderaxespad=0)


def plot_projection_prepost(ax, df):
    _prepost_plot(ax, df, "Passive whisker,\nproj. on hit − miss axis")


def plot_angle_prepost(ax, df):
    _prepost_plot(ax, df, "Passive whisker,\nangle to hit − miss axis (°)")


def _cohort_lines(ax, df, col, groups, colors, labels, ylabel, gcol, ylim=None, legend=True):
    for g, cl, lab in zip(groups, colors, labels):
        m = [mean_sem(df[(df[gcol] == g) & (df.cohort == c)][col]) for c in COH]
        ax.errorbar([0, 1], [v[0] for v in m], [v[1] for v in m], color=cl, marker="o", ms=3, lw=1.0, capsize=0, label=lab)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    if ylim is not None:
        ax.set_ylim(ylim)
    yl = ax.get_ylim()
    span = yl[1] - yl[0]
    for k, (g, cl) in enumerate(zip(groups, colors)):
        t = two_group_tests(df[(df[gcol] == g) & (df.cohort == "R+")][col], df[(df[gcol] == g) & (df.cohort == "R-")][col])
        bracket(ax, 0, 1, yl[1] + span * (0.04 + 0.24 * k), pmax(t["mannwhitney_p"], t["welch_p"]), cl)
    ax.set_ylim(yl[0], yl[1] + span * 0.3)
    ax.set_xlim(-0.4, 1.4)
    cohort_ticks(ax)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=FS - 0.5, loc="lower center", handlelength=1.2, borderaxespad=0.1)


def plot_delta(ax, df):
    _cohort_lines(ax, df, "value", ["W", "A"], [W_COL, A_COL], ["Whisker", "Auditory"],
                  "Δ proj. on hit − miss axis\n(post − pre; data − shifted)", "stim")


def plot_gain_rotation(ax, df, modality="whisker", ylim=None, legend=True, ylabel=True):
    _cohort_lines(ax, df, "value", ["gain", "rotation"], ["0.62", "0.1"], ["Gain", "Rotation"],
                  f"Δ {modality} proj. on hit − miss axis\n(post − pre; data − shifted)" if ylabel else "", "term",
                  ylim=ylim, legend=legend)


def area_label(a):
    return (a.replace(" areas", "").replace("Somatosensory-", "Ss-").replace("Lateral septal complex", "Lat. septum")
            .replace("Posterior parietal", "Post. parietal").replace("Amygdala and hypothalamus", "Amygdala / hypoth."))


def plot_areas(ax, df, modality="whisker", xlim=None, ylabels=True, legend=True):
    y = np.arange(len(df))[::-1]
    for c, m, s, off in (("R+", "mean_Rp", "sem_Rp", 0.14), ("R-", "mean_Rm", "sem_Rm", -0.14)):
        ax.errorbar(df[m], y + off, xerr=df[s], fmt="o", ms=2.4, color=COLORS[c], lw=0.7, capsize=0)
    ax.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_yticks(y, [area_label(a) for a in df.area] if ylabels else [""] * len(df))
    for t_, c in zip(ax.get_yticklabels(), df.color):
        t_.set_color(c)
    ax.tick_params(axis="y", length=0, labelsize=FS)
    ax.set_ylim(-0.7, len(df) - 0.3)
    if xlim is not None:
        ax.set_xlim(xlim)
    for yy, r in zip(y, df.itertuples()):
        p = pmax(r.mannwhitney_p, r.welch_p)
        if p < 0.05:
            ax.annotate(stars(p), (1.0, yy), xycoords=("axes fraction", "data"), ha="left", va="center", fontsize=FS)
    ax.set_xlabel(f"Δ {modality} proj. on hit − miss axis\n(post − pre; data − shifted)")
    if legend:
        ax.text(0.02, 1.0, "R+", transform=ax.transAxes, ha="left", va="top", color=COLORS["R+"], fontsize=FS + 0.5)
        ax.text(0.02, 0.94, "R−", transform=ax.transAxes, ha="left", va="top", color=COLORS["R-"], fontsize=FS + 0.5)


def plot_area_link(ax, df, modality="whisker", xlim=None, ylim=None, ylabel=True, note=True):
    for k, c in enumerate(COH):
        g = df[df.cohort == c]
        lr = ols_band(ax, g.dev, g.q, COLORS[c])
        ax.scatter(g.dev, g.q, s=10, c=list(g.color), edgecolors=COLORS[c], linewidths=0.7, zorder=3)
        ax.text(0.03, 0.97 - 0.09 * k, f"{LAB[c]}: r = {lr.rvalue:.2f}, {p_text(lr.pvalue)}", transform=ax.transAxes,
                color=COLORS[c], fontsize=FS, va="top")
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(f"Δ {modality} evoked rate, post − pre (spikes/s)" if modality else "Δ evoked rate, post − pre (spikes/s)")
    if ylabel:
        ax.set_ylabel(f"Δ {modality or 'passive'} proj. on hit − miss axis\n(post − pre; data − shifted)")
    if note:
        ax.text(0.98, 0.03, "One point per area and cohort\n(fill = area, edge = cohort)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FS - 0.8, color="0.4")


def plot_across_days(ax, df, modality="whisker", ylim=None, ylabel=True, legend=True):
    for c in COH:
        m = [mean_sem(df[(df.cohort == c) & (df.stage == s)].q) for s in ("day 0", "expert")]
        off = -0.05 if c == "R+" else 0.05
        ax.plot([0 + off, 1 + off], [v[0] for v in m], color=COLORS[c], lw=1.0, zorder=2)
        ax.errorbar(0 + off, m[0][0], m[0][1], color=COLORS[c], marker="o", ms=3, mfc="white", mew=0.8, capsize=0, lw=1.0, zorder=3)
        ax.errorbar(1 + off, m[1][0], m[1][1], color=COLORS[c], marker="o", ms=3, capsize=0, lw=1.0, zorder=3)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    if ylim is not None:
        ax.set_ylim(ylim)
    yl = ax.get_ylim()
    span = yl[1] - yl[0]
    for i, s in enumerate(("day 0", "expert")):
        g = df[df.stage == s]
        t = two_group_tests(g[g.cohort == "R+"].q, g[g.cohort == "R-"].q)
        sig(ax, i, yl[1] + span * 0.04, pmax(t["mannwhitney_p"], t["welch_p"]))
    ax.set_ylim(yl[0], yl[1] + span * 0.3)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1], ["Learning\n(day 0)", "Expert"])
    if ylabel:
        ax.set_ylabel(f"Δ {modality} proj. on hit − miss axis\n(post − pre; data − shifted)")
    for k, c in enumerate(COH if legend else ()):
        ax.text(0.97, 0.03 + 0.09 * (1 - k), LAB[c], transform=ax.transAxes, ha="right", va="bottom", color=COLORS[c],
                fontsize=FS + 0.5)


# ------------------------------------------------------------------ plot (supp)

def _two_bars(ax, df, col, ylabel):
    for i, c in enumerate(COH):
        m, s = mean_sem(df[df.cohort == c][col])
        ax.bar(i, m, 0.6, color=COLORS[c], lw=0)
        ax.errorbar(i, m, s, color="0.2", capsize=0, lw=0.7)
    t = two_group_tests(df[df.cohort == "R+"][col], df[df.cohort == "R-"][col])
    yl = ax.get_ylim()
    bracket(ax, 0, 1, yl[1] + 0.05 * (yl[1] - yl[0]), pmax(t["mannwhitney_p"], t["welch_p"]))
    ax.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))
    ax.axhline(0, color="0.4", lw=0.4)
    cohort_ticks(ax)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel(ylabel)


DVS_COMPS = (("data", "Data"), ("shifted", "Shifted"), ("excess", "Data −\nshifted"))
DVS_LAB = {"projection": "Δ {} proj. on\nhit − miss axis (post − pre)", "angle": "Δ {} angle to\nhit − miss axis (°, post − pre)"}


def _dvs_inset(a, g0, ylabel, ylim=None, legend=False):
    """data, shifted, data - shifted change per cohort (one measure); R+ vs R- bracket per component"""
    for c in COH:
        off = -0.12 if c == "R+" else 0.12
        for i, (comp, _) in enumerate(DVS_COMPS):
            mm, s = mean_sem(g0[(g0.cohort == c) & (g0.component == comp)].value)
            a.errorbar(i + off, mm, s, color=COLORS[c], marker="o", ms=2.6, capsize=0, lw=0.8,
                       mfc="white" if comp == "shifted" else COLORS[c], mew=0.7)
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    if ylim is not None:
        a.set_ylim(ylim)
    yl = a.get_ylim()
    span = yl[1] - yl[0]
    for i, (comp, _) in enumerate(DVS_COMPS):
        g = g0[g0.component == comp]
        t = two_group_tests(g[g.cohort == "R+"].value, g[g.cohort == "R-"].value)
        bracket(a, i - 0.15, i + 0.15, yl[1] + 0.04 * span, pmax(t["mannwhitney_p"], t["welch_p"]))
    a.set_ylim(yl[0], yl[1] + 0.3 * span)
    a.set_xlim(-0.5, 2.5)
    a.set_xticks(range(3), [l for _, l in DVS_COMPS])
    a.tick_params(labelsize=FS)
    a.set_ylabel(ylabel, fontsize=FS)
    if legend:
        for j, c in enumerate(COH):
            a.text(0.335, 0.03 + 0.09 * (1 - j), LAB[c], transform=a.transAxes, color=COLORS[c], fontsize=FS + 0.5,
                   ha="center")   # between the Data and Shifted columns (no data there)


def plot_data_vs_shifted(ax, df, modality="whisker"):
    """two insets (projection, angle) for one modality"""
    for k, (a, m) in enumerate(zip(_pair(ax, 0.2), ("projection", "angle"))):
        _dvs_inset(a, df[df.measure == m], DVS_LAB[m].format(modality), legend=k == 0)


# ---- whisker | auditory pairs, shared axes (SuppFigure4)

def plot_wa_time(ax, df):
    plot_time(ax, df, stims=("W", "A"))


def plot_wa_projection_prepost(ax, df):
    lim = _padded(*_ms_range(df, ["stim", "cohort", "kind", "epoch"]))
    for a, s in zip(_pair(ax, 0.1), "WA"):
        _prepost_plot(a, df[df.stim == s], "Passive response,\nproj. on hit − miss axis" if s == "W" else "", ylim=lim,
                      legend=s == "W")
        _mod_title(a, s, pad=16)


def plot_wa_angle_prepost(ax, df):
    lim = _padded(*_ms_range(df, ["stim", "cohort", "kind", "epoch"]), zero=False)
    for a, s in zip(_pair(ax, 0.1), "WA"):
        _prepost_plot(a, df[df.stim == s], "Passive response,\nangle to hit − miss axis (°)" if s == "W" else "", ylim=lim,
                      legend=s == "W")
        _mod_title(a, s, pad=16)


def plot_wa_gain_rotation(ax, df):
    lim = _padded(*_ms_range(df, ["stim", "term", "cohort"]))
    for a, s in zip(_pair(ax, 0.1), "WA"):
        plot_gain_rotation(a, df[df.stim == s], "passive", ylim=lim, legend=s == "W", ylabel=s == "W")
        _mod_title(a, s, pad=16)


def plot_wa_areas(ax, df):
    order = list(dict.fromkeys(df[df.stim == "W"].area.tolist() + df[df.stim == "A"].area.tolist()))
    col = df.drop_duplicates("area").set_index("area").color
    lo = min((df.mean_Rp - df.sem_Rp).min(), (df.mean_Rm - df.sem_Rm).min())
    hi = max((df.mean_Rp + df.sem_Rp).max(), (df.mean_Rm + df.sem_Rm).max())
    lim = _padded(lo, hi)
    for a, s in zip(_pair(ax, 0.06), "WA"):
        g = df[df.stim == s].set_index("area").reindex(order).reset_index()
        g["color"] = g.area.map(col)
        plot_areas(a, g, MOD[s][1], xlim=lim, ylabels=s == "W", legend=s == "W")
        _mod_title(a, s)


def plot_wa_area_link(ax, df):
    xl = _padded(df.dev.min(), df.dev.max(), 0.08)
    yl = _padded(df.q.min(), df.q.max(), 0.08)
    yl = (yl[0], yl[1] + 0.25 * (yl[1] - yl[0]))          # room for the r / p text
    for a, s in zip(_pair(ax, 0.1), "WA"):
        plot_area_link(a, df[df.stim == s], "", xlim=xl, ylim=yl, ylabel=s == "W", note=False)
        _mod_title(a, s)


def plot_wa_across_days(ax, df):
    lim = _padded(*_ms_range(df, ["stim", "stage", "cohort"], "q"))
    for a, s in zip(_pair(ax, 0.1), "WA"):
        plot_across_days(a, df[df.stim == s], "passive", ylim=lim, ylabel=s == "W", legend=s == "A")
        _mod_title(a, s, pad=16)


def _area_rows(ax, df, series, xlim, ylabels, stars_col=None):
    """horizontal per-area dot plot; series = [(mean col, sem col, colour, y offset)]; stars at the right edge"""
    y = np.arange(len(df))[::-1]
    for m, s, cl, off in series:
        ax.errorbar(df[m], y + off, xerr=df[s], fmt="o", ms=2.4, color=cl, lw=0.7, capsize=0)
    ax.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_yticks(y, [area_label(a) for a in df.area] if ylabels else [""] * len(df))
    for t_, c in zip(ax.get_yticklabels(), df.color):
        t_.set_color(c)
    ax.tick_params(axis="y", length=0, labelsize=FS)
    ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_xlim(xlim)
    if stars_col:
        for yy, p in zip(y, df[stars_col]):
            if p < 0.05:
                ax.annotate(stars(p), (1.0, yy), xycoords=("axes fraction", "data"), ha="left", va="center", fontsize=FS)


def plot_wa_area_types(ax, df):
    """R+ | R-: whisker (yellow) vs auditory (blue) change per area; stars, whisker vs auditory (paired) within the cohort"""
    cols = [f"{v}_{k}" for v in ("W", "A") for k in ("Rp", "Rm")]
    lim = _padded(min((df[f"mean_{c}"] - df[f"sem_{c}"]).min() for c in cols),
                  max((df[f"mean_{c}"] + df[f"sem_{c}"]).max() for c in cols))
    for j, (a, c, k) in enumerate(zip(_pair(ax, 0.08), COH, ("Rp", "Rm"))):
        _area_rows(a, df, [(f"mean_W_{k}", f"sem_W_{k}", W_COL, 0.14), (f"mean_A_{k}", f"sem_A_{k}", A_COL, -0.14)], lim,
                   ylabels=j == 0, stars_col=f"p_WA_{k}")
        a.set_title(LAB[c], color=COLORS[c], fontsize=FS + 0.5, pad=2)
        a.set_xlabel("Δ passive proj. on hit − miss axis\n(post − pre; data − shifted)")
        if j == 0:
            a.text(0.02, 1.0, "Whisker", transform=a.transAxes, ha="left", va="top", color=W_COL, fontsize=FS)
            a.text(0.02, 0.94, "Auditory", transform=a.transAxes, ha="left", va="top", color=A_COL, fontsize=FS)


def plot_wa_area_diff(ax, df):
    """whisker - auditory change per area, R+ vs R-; stars, R+ vs R- (cohort x modality per area)"""
    lim = _padded(min((df[f"mean_D_{k}"] - df[f"sem_D_{k}"]).min() for k in ("Rp", "Rm")),
                  max((df[f"mean_D_{k}"] + df[f"sem_D_{k}"]).max() for k in ("Rp", "Rm")))
    _area_rows(ax, df, [("mean_D_Rp", "sem_D_Rp", COLORS["R+"], 0.14), ("mean_D_Rm", "sem_D_Rm", COLORS["R-"], -0.14)], lim,
               ylabels=True, stars_col="p_D_cohort")
    ax.set_xlabel("Whisker − auditory change\n(post − pre; data − shifted)")
    ax.text(0.02, 1.0, "R+", transform=ax.transAxes, ha="left", va="top", color=COLORS["R+"], fontsize=FS + 0.5)
    ax.text(0.02, 0.94, "R−", transform=ax.transAxes, ha="left", va="top", color=COLORS["R-"], fontsize=FS + 0.5)


def plot_wa_decoder(ax, df):
    g = df[df.kind == "data"]
    lim = _padded(*_ms_range(g, ["stim", "cohort", "epoch"]))
    for a, s in zip(_pair(ax, 0.1), "WA"):
        _prepost_plot(a, df[df.stim == s], "Passive response, hit / miss\ndecoder readout (s.d.)" if s == "W" else "",
                      ylim=lim, legend=False)
        _mod_title(a, s, pad=16)


def plot_wa_data_vs_shifted(ax, df):
    """projection W | A, angle W | A; shared y within a measure"""
    axs = _pair(ax, 0.1, n=4)
    for k, m in enumerate(("projection", "angle")):
        g = df[df.measure == m]
        lim = _padded(*_ms_range(g, ["stim", "component", "cohort"]))
        for j, s in enumerate("WA"):
            a = axs[2 * k + j]
            _dvs_inset(a, g[g.stim == s], DVS_LAB[m].format("passive") if j == 0 else "", ylim=lim, legend=k == 0 and j == 0)
            _mod_title(a, s, pad=16)


def plot_axis_quality(ax, df):
    _two_bars(ax, df, "dprime_loo", "Held-out whisker hits vs misses,\nd′ on hit − miss axis")


def plot_stimulus_axis(ax, df):
    _two_bars(ax, df[df.measure == "cos_u_s"], "value", "Cosine, hit − miss axis vs\nwhisker − auditory axis")


def plot_units(ax, df):
    _two_bars(ax, df, "rho", "Axis weight vs Δ whisker rate\n(Spearman ρ over neurons)")
    for i, c in enumerate(COH):
        v = df[df.cohort == c].rho
        p = pmax(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue)
        m, s = mean_sem(v)
        ax.annotate(f"{stars(p)}\n{p_text(p)}", (i, min(m - s, 0)), xytext=(0, -2), textcoords="offset points", ha="center",
                    va="top", fontsize=FS - 0.5, color=COLORS[c], linespacing=1.0)
    yl = ax.get_ylim()
    ax.set_ylim(yl[0] - 0.25 * (yl[1] - yl[0]), yl[1])


def plot_fine_areas(ax, df):
    x = np.arange(len(df))
    for c, m, s, off in (("R+", "mean_Rp", "sem_Rp", -0.15), ("R-", "mean_Rm", "sem_Rm", 0.15)):
        ax.errorbar(x + off, df[m], df[s], fmt="o", ms=2, color=COLORS[c], lw=0.6, capsize=0)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks(x, df.area, rotation=90)
    for t_, c in zip(ax.get_xticklabels(), df.color):
        t_.set_color(c)
    ax.tick_params(axis="x", labelsize=FS - 0.5, length=0)
    ax.set_xlim(-0.7, len(df) - 0.3)
    yl = ax.get_ylim()
    for i, r in df.iterrows():
        p = pmax(r.mannwhitney_p, r.welch_p)
        if p < 0.05:
            sig(ax, i, yl[1], p, rotate=True)
    ax.set_ylabel("Δ whisker proj. on hit − miss axis\n(post − pre; data − shifted)")
    ax.text(0.01, 0.10, "R+", transform=ax.transAxes, color=COLORS["R+"], fontsize=FS + 0.5)
    ax.text(0.01, 0.03, "R−", transform=ax.transAxes, color=COLORS["R-"], fontsize=FS + 0.5)


def plot_behaviour(ax, df):
    for k, c in enumerate(COH):
        g = df[df.cohort == c]
        lr = ols_band(ax, g.plick_W, g.q, COLORS[c])
        ax.scatter(g.plick_W, g.q, s=4, color=COLORS[c], lw=0, zorder=3)
        ax.text(0.97, 0.97 - 0.09 * k, f"{LAB[c]}: n = {len(g)}, r = {lr.rvalue:.2f}, {p_text(lr.pvalue)}",
                transform=ax.transAxes, color=COLORS[c], fontsize=FS, va="top", ha="right")
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))      # stats text above the points
    ax.set_xlabel("Whisker P(lick), active block")
    ax.set_ylabel("Δ whisker proj. on hit − miss axis\n(post − pre; data − shifted)")
