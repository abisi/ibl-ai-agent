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
import os  # noqa: E402
# learner population (user 2026-11-24: compare learning_category, learning_category_new and all mice):
# SSL_CD_LEARNERS = sheet (default, learning_category good / moderate, as panels A-G) | new (learning_category_new) | all
LEARNERS = os.environ.get("SSL_CD_LEARNERS", "sheet")
CATEGORY_COLUMN = {"sheet": "learning_category", "new": "learning_category_new", "all": None}[LEARNERS]
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
    if CATEGORY_COLUMN is None:                                        # all mice
        return set(ref.mouse_id)
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


def perm_interaction(v, g, n=N_PERM, seed=0, return_null=False):
    rng = np.random.default_rng(seed)
    v, g = np.asarray(v, float), np.asarray(g)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    d = lambda gg: v[gg == "R+"].mean() - v[gg == "R-"].mean()
    obs = d(g)
    null = np.array([d(rng.permutation(g)) for _ in range(n)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))
    return (float(obs), p, null) if return_null else (float(obs), p)


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
    # user 2026-11-25: the cohort x modality interaction drawn explicitly as a third series (whisker - auditory)
    d = wa("q").melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=["W", "A", "WA"], var_name="stim")
    obs, p = perm_interaction(wa("q").WA, wa("q").cohort)
    META["delta"] = dict(W=tests(d[d.stim == "W"], "value"), A=tests(d[d.stim == "A"], "value"),
                         interaction=dict(diff=obs, p=p), WA=tests(wa("q"), "WA"))   # MW + Welch on W - A (user 2026-11-24)
    return d


def compute_strength_alignment(stim="W", name="strength_alignment"):
    """user 2026-11-24 (Method 7): response strength and alignment changes shown separately, no weighted split.
    strength = rho_post - rho_pre (cross-validated norm; independent of the axis, so no shifted value);
    alignment = (cos_post - cos_pre) - (cos_post_null - cos_pre_null) (data - shifted).
    The exact split of the projection change (a = rho cos) under the three orderings is kept as extra rows (term
    gain_* / rotation_*: o1 = gain at cos_pre, rotation at rho_post; o2 = gain at cos_post, rotation at rho_pre; sym = average)
    and in META[name]['orderings'] (supp table)."""
    w = whole(stim)
    dr = w.rho_post - w.rho_pre
    dc = (w.cos_post - w.cos_pre) - (w.cos_post_null - w.cos_pre_null)
    w["strength"], w["alignment"] = dr, dc
    w["gain_o1"], w["rotation_o1"] = dr * (w.cos_pre - w.cos_pre_null), w.rho_post * dc
    w["gain_o2"], w["rotation_o2"] = dr * (w.cos_post - w.cos_post_null), w.rho_pre * dc
    w["gain_sym"] = dr * 0.5 * ((w.cos_pre - w.cos_pre_null) + (w.cos_post - w.cos_post_null))
    w["rotation_sym"] = 0.5 * (w.rho_pre + w.rho_post) * dc
    terms = ["strength", "alignment"] + [f"{k}_{o}" for o in ("o1", "o2", "sym") for k in ("gain", "rotation")]
    d = w.melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=terms, var_name="term")
    META[name] = {k: tests(d[d.term == k], "value") for k in ("strength", "alignment")}
    for c in COH:
        x = w[w.cohort == c]
        META[name][c] = dict(strength_vs0=float(stats.wilcoxon(x.strength.dropna()).pvalue),
                             alignment_vs0=float(stats.wilcoxon(x.alignment.dropna()).pvalue))
    META[name]["orderings"] = {k: tests(d[d.term == k], "value") for k in terms[2:]}
    META[name]["check_max_abs_residual"] = float(np.nanmax(np.abs(w.gain_sym + w.rotation_sym - w.q)))
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
    # user 2026-11-24 (Method 9b): y = angle change (data - shifted), not the projection change (partly built in)
    m = G.groupby(["area", "cohort"]).agg(dev=("dev", "mean"), dev_sem=("dev", "sem"), q=("qth", "mean"), q_sem=("qth", "sem"),
                                         q_proj=("q", "mean"), n=("session_id", "nunique")).reset_index()
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


SS_CONDS = ("pre_W", "post_W", "act_hit", "act_miss")


def _geometry(stage):
    d = pd.read_parquet(TAB / f"193_session_geometry{'' if stage == 'learning' else '_' + stage}.parquet")
    d = d[d.skipped_reason.isna()].rename(columns={"reward_group": "cohort"})
    if stage == "learning":
        return d[d.mouse_id.isin(learners())].assign(unit_id=lambda x: x.session_id)
    return d.groupby(["mouse_id", "cohort"]).mean(numeric_only=True).reset_index().assign(unit_id=lambda x: x.mouse_id)


def compute_state_space():
    """user 2026-11-25: Figure 4 state space = passive-pre plane. x = hit - miss axis u; y = passive-pre whisker direction
    orthogonal to u, cross-validated (193 ss2cv_*: direction from one half of the pre trials; passive-pre whisker projected
    with the other half; every other condition on both half-directions, averaged). Learning day: sessions; expert: mice."""
    rows, meta = [], {}
    for stage in ("learning", "expert"):
        d = _geometry(stage)
        for cond in SS_CONDS:
            rows.append(pd.DataFrame(dict(stage=stage, unit_id=d.unit_id, cohort=d.cohort, cond=cond,
                                          x=d[f"ss_{cond}_u"], y=d[f"ss2cv_{cond}_v"])))
        dd = d.assign(dx=d.ss_post_W_u - d.ss_pre_W_u, dy=d.ss2cv_post_W_v - d.ss2cv_pre_W_v)
        meta[stage] = dict(dx=tests(dd, "dx"), dy=tests(dd, "dy"),
                           **{c: dict(n=int((d.cohort == c).sum()), dx=paired(d[d.cohort == c].ss_pre_W_u, d[d.cohort == c].ss_post_W_u),
                                      dy=paired(d[d.cohort == c].ss2cv_pre_W_v, d[d.cohort == c].ss2cv_post_W_v)) for c in COH})
    META["state_space"] = meta
    return pd.concat(rows, ignore_index=True)


LC_MEAS = "hit_p_spearman_excess"
LC_MIN_SESS = 3


def compute_learning_curve():
    """user 2026-11-25: active whisker hits projected on the hit - miss axis track the exact whisker learning curve (195;
    Spearman, data - shifted); whole brain, then the distribution of that correlation over area groups (not SSp-bfd / L2/3)."""
    s = pd.read_parquet(TAB / "195_hit_trajectory_sessions.parquet")
    s = s[s.skipped_reason.isna() & s.mouse_id.isin(learners())].rename(columns={"reward_group": "cohort"})
    s = s[(s.group == "whole_brain") | s.group.str.startswith("area: ")]
    d = pd.DataFrame(dict(session_id=s.session_id, mouse_id=s.mouse_id, cohort=s.cohort,
                          group=s.group.str.replace("area: ", "", regex=False), value=s[LC_MEAS])).dropna(subset=["value"])
    wb = d[d.group == "whole_brain"]
    ar = d[d.group != "whole_brain"]
    am = ar.groupby(["group", "cohort"]).value.agg(["mean", "count"]).reset_index()
    am = am[am["count"] >= LC_MIN_SESS]
    both = am.pivot(index="group", columns="cohort", values="mean").dropna()
    meta = dict(whole=tests(wb, "value"), n_areas={c: int((am.cohort == c).sum()) for c in COH}, n_both=len(both),
                areas_rp_vs_rm=dict(wilcoxon_p=float(stats.wilcoxon(both["R+"], both["R-"]).pvalue),
                                    paired_t_p=float(stats.ttest_rel(both["R+"], both["R-"]).pvalue)))
    for c in COH:
        v = wb[wb.cohort == c].value
        meta[c] = dict(n=int(len(v)), wilcoxon_p=float(stats.wilcoxon(v).pvalue), t_p=float(stats.ttest_1samp(v, 0).pvalue))
        a = am[am.cohort == c]["mean"]
        meta[f"areas_{c}"] = dict(wilcoxon_p=float(stats.wilcoxon(a).pvalue), t_p=float(stats.ttest_1samp(a, 0).pvalue),
                                  frac_pos=float((a > 0).mean()))
    META["learning_curve"] = meta
    return d


def compute_orthogonalisation():
    """185 (user 2026-11-24/25): angle of the pre-task passive whisker response to the hit - miss axis, learning day vs expert
    days (data, filled; shifted axes, open); expert sessions averaged per mouse, day 0 = one session per mouse."""
    rows = []
    for stage, lab in (("learning", "day 0"), ("expert", "expert")):
        w = whole("W", stage)
        rows.append(pd.DataFrame(dict(mouse_id=w.mouse_id, cohort=w.cohort, stage=lab, data=theta(w.cos_pre),
                                      shifted=theta(w.cos_pre_null))))
    d = pd.concat(rows).groupby(["mouse_id", "cohort", "stage"]).mean(numeric_only=True).reset_index()
    import statsmodels.formula.api as smf
    meta = {c: dict(two_group_tests(d[(d.cohort == c) & (d.stage == "day 0")].data, d[(d.cohort == c) & (d.stage == "expert")].data))
            for c in COH}
    for s in ("day 0", "expert"):
        meta[s] = tests(d[d.stage == s], "data")
    meta["interaction_p"] = float(smf.ols("data ~ C(cohort) * C(stage)", data=d).fit().pvalues.filter(like=":").iloc[0])
    META["orthogonalisation"] = meta
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


def compute_wa_strength_alignment():
    return _wa_compute(compute_strength_alignment, "wa_strength_alignment")


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


def plot_time(ax, df, stims=("W",), xlim=(-0.05, 0.1)):
    """R+ (top) / R- (bottom); one column per passive modality (active whisker hit / miss in every column, as reference);
    shared y axis within a row. Main figure zoomed to -50..100 ms (user 2026-11-25); supp keeps -100..200 ms."""
    df = df[(df.t >= xlim[0] - 1e-9) & (df.t <= xlim[1] + 1e-9)]
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
            a.set_xlim(*xlim)
            a.set_xticks([-0.05, 0, 0.05, 0.1] if xlim[1] <= 0.1 else [-0.1, 0, 0.1, 0.2])
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
                if ncol == 1:   # main (square axes): labels outside, right of the traces (2026-11-25)
                    a.text(1.04, y, lab, transform=a.transAxes, ha="left", va="top", fontsize=FS - 0.8, color=col, clip_on=False)
                else:
                    a.text(1.0, y, lab, transform=a.transAxes, ha="right", va="top", fontsize=FS - 0.8, color=col)
                y -= 0.085 if ncol > 1 else 0.2
            if k == 0:
                a.set_xticklabels([])
            else:
                a.set_xlabel("Time from stimulus (s)")
            a.tick_params(labelsize=FS)
        lo = min(a.get_ylim()[0] for a in row)
        hi = max(a.get_ylim()[1] for a in row)
        if k == 0:
            hi = hi + (0.45 if ncol > 1 else 0.1) * (hi - lo)                 # room for the labels above the traces
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
        step = 0.24 if len(groups) <= 2 else 0.36
        bracket(ax, 0, 1, yl[1] + span * (0.04 + step * k), pmax(t["mannwhitney_p"], t["welch_p"]), cl)
    ax.set_ylim(yl[0], yl[1] + span * (0.3 if len(groups) <= 2 else 0.1 + 0.36 * (len(groups) - 1)))
    ax.set_xlim(-0.4, 1.4)
    cohort_ticks(ax)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=FS - 0.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.2, borderaxespad=0,
                  frameon=False)


def plot_delta(ax, df):
    groups = [g for g in ("W", "A", "WA") if (df.stim == g).any()]
    _cohort_lines(ax, df, "value", groups, [W_COL, A_COL, "0.25"][:len(groups)],
                  ["Whisker", "Auditory", "Whisker − auditory"][:len(groups)],
                  "Δ proj. on hit − miss axis\n(post − pre; data − shifted)", "stim")


def _ss_inset(a, df, title, xlim=None, ylim=None, ylabel=True, legend=False):
    for c in COH:
        g = df[df.cohort == c]
        m = {k: (mean_sem(g[g.cond == k].x), mean_sem(g[g.cond == k].y)) for k in SS_CONDS}
        for k, fc in (("act_hit", W_COL), ("act_miss", MISS_COL)):          # user 2026-11-26: upward triangles, hit / miss colours
            (x, ex), (y, ey) = m[k]
            a.errorbar(x, y, xerr=ex, yerr=ey, color=COLORS[c], marker="^", ms=3.4, mfc=fc, mec=COLORS[c], mew=0.6, lw=0.4,
                       capsize=0)
        (x0, ex0), (y0, ey0) = m["pre_W"]
        (x1, ex1), (y1, ey1) = m["post_W"]
        a.errorbar(x0, y0, xerr=ex0, yerr=ey0, color=COLORS[c], marker="o", ms=3.5, mfc="white", mew=0.8, lw=0.5, capsize=0)
        a.errorbar(x1, y1, xerr=ex1, yerr=ey1, color=COLORS[c], marker="o", ms=3.5, mfc=COLORS[c], mew=0.8, lw=0.5, capsize=0)
        a.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=COLORS[c], lw=1.2, mutation_scale=7,
                                                             shrinkA=2.5, shrinkB=2.5))
    a.axhline(0, color="0.75", lw=0.4)
    a.axvline(0, color="0.75", lw=0.4)
    if xlim is not None:
        a.set_xlim(xlim)
    if ylim is not None:
        a.set_ylim(ylim)
    a.set_title(title, fontsize=FS + 0.5, pad=2)
    a.set_xlabel("Projection on the hit − miss axis")
    if ylabel:
        a.set_ylabel("Projection on the passive whisker\ndirection (orthogonal)")
    if legend:
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color="0.3", marker="o", mfc="white", ms=3, ls="", label="Passive pre"),
             Line2D([], [], color="0.3", marker="o", ms=3, ls="", label="Passive post"),
             Line2D([], [], color="0.3", marker="^", mfc=W_COL, ms=3.4, ls="", label="Active whisker hit"),
             Line2D([], [], color="0.3", marker="^", mfc=MISS_COL, ms=3.4, ls="", label="Active whisker miss")]
        a.legend(handles=h, fontsize=FS - 0.8, frameon=False, loc="lower right", handletextpad=0.2,
                 borderaxespad=0.1, labelspacing=0.25)


def _state_space_stages(ax, df, stages):
    """passive whisker pre (open) -> post (filled) per cohort; one inset per stage, shared axes"""
    df = df[df.stage.isin(stages)]
    lo_x, hi_x = df.groupby(["stage", "cohort", "cond"]).x.mean().agg(["min", "max"])
    lo_y, hi_y = df.groupby(["stage", "cohort", "cond"]).y.mean().agg(["min", "max"])
    xl, yl = _padded(lo_x, hi_x, 0.15), _padded(lo_y, hi_y, 0.15)
    titles = {"learning": "Learning day", "expert": "Expert days"}
    insets = _pair(ax, 0.1, n=len(stages)) if len(stages) > 1 else [ax]
    for j, (a, stage) in enumerate(zip(insets, stages)):
        g = df[df.stage == stage]
        n = "/".join(str(g[(g.cohort == c) & (g.cond == "pre_W")].unit_id.nunique()) for c in COH)
        _ss_inset(a, g, f"{titles[stage]} ({n} {'sessions' if stage == 'learning' else 'mice'})", xl, yl, ylabel=j == 0,
                  legend=j == len(stages) - 1)
        if j:
            a.set_yticklabels([])


def plot_state_space(ax, df):
    """main (user 2026-11-25: expert results in the supplement): learning day only; annotated as the bridge to the passive
    response analysis (panels A-G): y = the passive-pre response itself (drop = weaker response, both cohorts), x = the
    hit - miss axis (shift toward the miss point, R- only)"""
    a = _mm_insets(ax, [min(_ax_width_mm(ax), AX_MM * 1.25)], [])[0]   # square (user 2026-11-26); summary text removed
    _state_space_stages(a, df, ("learning",))


def plot_state_space_expert(ax, df):
    """supp: learning day | expert days side by side (same axes)"""
    _state_space_stages(ax, df, ("learning", "expert"))


def compute_state_space_expert():
    return compute_state_space()


def plot_learning_curve(ax, df):
    """left: whole brain, Spearman r (hit projections vs P(lick | whisker); data - shifted) per cohort; right: one dot per area
    group and cohort (area mean, >= LC_MIN_SESS sessions), lines join an area's R+ and R- values"""
    a, b = _pair(ax, 0.3)
    wb = df[df.group == "whole_brain"]
    for i, c in enumerate(COH):
        v = wb[wb.cohort == c].value
        m, e = mean_sem(v)
        a.errorbar(i, m, e, color=COLORS[c], marker="o", ms=3, capsize=0, lw=1.0)
        sig(a, i, m + e, pmax(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue), COLORS[c])
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    yl = a.get_ylim()
    span = yl[1] - yl[0]
    t = two_group_tests(wb[wb.cohort == "R+"].value, wb[wb.cohort == "R-"].value)
    bracket(a, 0, 1, yl[1] + span * 0.25, pmax(t["mannwhitney_p"], t["welch_p"]))
    a.set_ylim(yl[0], yl[1] + span * 0.55)
    a.set_xlim(-0.5, 1.5)
    cohort_ticks(a)
    a.set_title("Whole brain", fontsize=FS + 0.5, pad=2)
    a.set_ylabel("Hit projection vs P(lick | whisker)\nSpearman r (data − shifted)")
    col, _, _ = area_palette()
    ar = df[df.group != "whole_brain"]
    am = ar.groupby(["group", "cohort"]).value.agg(["mean", "count"]).reset_index()
    am = am[am["count"] >= LC_MIN_SESS]
    w = am.pivot(index="group", columns="cohort", values="mean")
    for gname, r in w.iterrows():
        cl = col[gname]
        if r.notna().all():
            b.plot([0, 1], [r["R+"], r["R-"]], color=light(cl, 0.4), lw=0.5, zorder=1)
        for i, c in enumerate(COH):
            if np.isfinite(r.get(c, np.nan)):
                b.scatter(i, r[c], s=9, color=cl, edgecolors=COLORS[c], linewidths=0.6, zorder=2)
    for i, c in enumerate(COH):
        m, e = mean_sem(w[c])
        b.errorbar(i + (-0.22 if i == 0 else 0.22), m, e, color=COLORS[c], marker="s", ms=3, capsize=0, lw=1.0, zorder=3)
    b.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    both = w.dropna()
    yl = b.get_ylim()
    bracket(b, 0, 1, yl[1] + 0.05 * (yl[1] - yl[0]),
            pmax(stats.wilcoxon(both["R+"], both["R-"]).pvalue, stats.ttest_rel(both["R+"], both["R-"]).pvalue))
    b.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))
    b.set_xlim(-0.5, 1.5)
    cohort_ticks(b)
    b.set_title(f"Area groups ({len(both)} in both cohorts)", fontsize=FS + 0.5, pad=2)
    b.set_ylabel("Spearman r (data − shifted),\narea mean")


def plot_orthogonalisation(ax, df):
    for c in COH:
        off = -0.05 if c == "R+" else 0.05
        for kind, filled, z in (("shifted", False, 1), ("data", True, 3)):
            m = [mean_sem(df[(df.cohort == c) & (df.stage == s)][kind]) for s in ("day 0", "expert")]
            ax.plot([0 + off, 1 + off], [v[0] for v in m], color=COLORS[c] if filled else "0.65", lw=1.0 if filled else 0.6, zorder=z)
            ax.errorbar([0 + off, 1 + off], [v[0] for v in m], [v[1] for v in m], color=COLORS[c], marker="o", ms=3,
                        mfc=COLORS[c] if filled else "white", mew=0.7, lw=0, elinewidth=0.8, capsize=0, zorder=z + 1)
    yl = ax.get_ylim()
    span = yl[1] - yl[0]
    for k, c in enumerate(COH):
        x0, x1 = (df[(df.cohort == c) & (df.stage == s)].data for s in ("day 0", "expert"))
        t = two_group_tests(x0, x1)
        bracket(ax, 0, 1, yl[1] + span * (0.04 + 0.24 * k), pmax(t["mannwhitney_p"], t["welch_p"]), COLORS[c])
    ax.set_ylim(yl[0], yl[1] + span * 0.55)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1], ["Learning\n(day 0)", "Expert"])
    ax.set_ylabel("Passive-pre whisker response,\nangle to hit − miss axis (°)")


SA_LAB = {"strength": "Δ response strength ρ\n(post − pre)", "alignment": "Δ alignment, cos θ\n(post − pre; data − shifted)"}


def _sa_inset(a, df, term, ylim=None, ylabel=True):
    """one term (strength or alignment): R+ and R− mean ± s.e.m. with the R+ vs R− bracket"""
    g = df[df.term == term]
    for i, c in enumerate(COH):
        m, e = mean_sem(g[g.cohort == c].value)
        a.errorbar(i, m, e, color=COLORS[c], marker="o", ms=3, capsize=0, lw=1.0)
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    if ylim is not None:
        a.set_ylim(ylim)
    yl = a.get_ylim()
    t = two_group_tests(g[g.cohort == "R+"].value, g[g.cohort == "R-"].value)
    bracket(a, 0, 1, yl[1] + 0.05 * (yl[1] - yl[0]), pmax(t["mannwhitney_p"], t["welch_p"]))
    a.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))
    a.set_xlim(-0.5, 1.5)
    cohort_ticks(a)
    a.tick_params(labelsize=FS)
    if ylabel:
        a.set_ylabel(SA_LAB[term], fontsize=FS)


def plot_strength_alignment(ax, df, lims=None):
    """user 2026-11-24 (Method 7): response strength and alignment changes side by side (own y axes)"""
    for a, term in zip(_pair(ax, 0.6), ("strength", "alignment")):
        _sa_inset(a, df, term, None if lims is None else lims[term])


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
        ax.set_ylabel(f"Δ {modality or 'passive'} angle to hit − miss axis\n(°, post − pre; data − shifted)")
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
    plot_time(ax, df, stims=("W", "A"), xlim=(-0.1, 0.2))


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


def plot_wa_strength_alignment(ax, df):
    """whisker | auditory, each with strength and alignment insets; y axes shared per term across modalities"""
    lims = {k: _padded(*_ms_range(df[df.term == k], ["stim", "cohort"])) for k in ("strength", "alignment")}
    axs = _pair(ax, 0.12, n=4)
    for j, s in enumerate("WA"):
        for k, term in enumerate(("strength", "alignment")):
            a = axs[2 * j + k]
            _sa_inset(a, df[df.stim == s], term, lims[term], ylabel=(j == 0))
            if k == 0:
                a.set_title(MOD[s][0], color=W_COL if s == "W" else A_COL, fontsize=FS + 0.5, pad=16, loc="left")


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


# ================================================================== Figure 4 v3 (user 2026-11-25): three sections
# whole brain (state space, projection W | A with the interaction, geometric decomposition), areas on the whole-brain axis
# (202), learning curves (examples, trial types x curves, areas). Everything data - shifted unless stated.

def compute_projection_wa():
    d = compute_wa_projection_prepost()
    meta = META.pop("wa_projection_prepost")
    q = wa("q")
    obs, p = perm_interaction(q.WA, q.cohort)
    meta["interaction"], meta["WA"] = dict(diff=obs, p=p), tests(q, "WA")
    META["projection_wa"] = meta
    return d


def plot_projection_wa(ax, df):
    lim = _padded(*_ms_range(df, ["stim", "cohort", "kind", "epoch"]))
    for a, s_ in zip(_pair(ax, 0.22), "WA"):
        _prepost_plot(a, df[df.stim == s_], "Passive response,\nproj. on hit − miss axis" if s_ == "W" else "", ylim=lim,
                      legend=s_ == "W")
        _mod_title(a, s_, pad=16)
        if s_ == "A":
            a.set_yticklabels([])
    d = df.pivot_table(index=["session_id", "cohort", "stim"], columns=["kind", "epoch"], values="value")
    ex = ((d[("data", "post")] - d[("data", "pre")]) - (d[("shifted", "post")] - d[("shifted", "pre")])).rename("ex").reset_index()
    w = ex.pivot_table(index=["session_id", "cohort"], columns="stim", values="ex").reset_index()
    w["WA"] = w.W - w.A
    t = two_group_tests(w[w.cohort == "R+"].WA, w[w.cohort == "R-"].WA)
    p = pmax(t["mannwhitney_p"], t["welch_p"])
    ax.text(0.5, -0.33, f"Cohort × modality (Δ whisker − Δ auditory): {p_text(p)} {stars(p)}", transform=ax.transAxes,
            ha="center", va="top", fontsize=FS)


def compute_decomposition():
    """a = rho cos(theta): strength change (rho post - pre, cross-validated norm, axis-free) and angle change (data -
    shifted), whisker and auditory"""
    rows, meta = [], {}
    for s in ("W", "A"):
        w = whole(s)
        ang = (theta(w.cos_post) - theta(w.cos_pre)) - (theta(w.cos_post_null) - theta(w.cos_pre_null))
        for term, v in (("strength", w.rho_post - w.rho_pre), ("angle", ang)):
            g = pd.DataFrame(dict(session_id=w.session_id, mouse_id=w.mouse_id, cohort=w.cohort, stim=s, term=term, value=v))
            rows.append(g)
            meta[f"{term}_{s}"] = tests(g, "value")
    META["decomposition"] = meta
    return pd.concat(rows, ignore_index=True)


def _decomp_schematic(a):
    a.set_xlim(-0.15, 1.15)
    a.set_ylim(-0.45, 1.05)
    a.axis("off")
    a.annotate("", (1.12, 0), (-0.05, 0), arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.8, mutation_scale=6))
    a.text(0.55, -0.05, "hit − miss axis", ha="center", va="top", fontsize=FS - 0.5)
    for th, r, col, lab in ((35, 0.95, "0.55", "pre"), (62, 0.8, COLORS["R-"], "post")):
        x, y = r * np.cos(np.radians(th)), r * np.sin(np.radians(th))
        a.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=6))
        a.plot([x, x], [0, y], color=col, lw=0.5, ls=(0, (1.5, 1.5)))
        a.text(x + 0.03, y, lab, color=col, fontsize=FS - 0.5, va="bottom")
    a.text(0.22, 0.06, "θ", fontsize=FS, color="0.2")
    a.text(0.3, 0.55, "ρ", fontsize=FS, color="0.2")
    a.text(0.5, -0.27, "a = ρ cos θ", ha="center", va="top", fontsize=FS)


def _two_cohorts(a, g, ylim=None, ylabel=None, dots=False):
    if dots:   # user 2026-11-25 ("show without clutter"): sessions as small light jittered dots behind the mean
        rng = np.random.default_rng(0)
        for i_, c_ in enumerate(COH):
            v_ = g[g.cohort == c_].value.dropna().to_numpy()
            if ylim is not None:        # 2026-11-25: axes zoomed on the means; sessions outside drawn at the border
                v_ = np.clip(v_, *ylim)
            a.scatter(i_ + rng.uniform(-0.13, 0.13, len(v_)), v_, s=1.8, color="0.8", lw=0, zorder=0, clip_on=False)
    for i, c in enumerate(COH):
        m, e = mean_sem(g[g.cohort == c].value)
        a.errorbar(i, m, e, color=COLORS[c], marker="o", ms=3, capsize=0, lw=1.0)
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    if ylim is not None:
        a.set_ylim(ylim)
    yl = a.get_ylim()
    t = two_group_tests(g[g.cohort == "R+"].value, g[g.cohort == "R-"].value)
    bracket(a, 0, 1, yl[1] + 0.05 * (yl[1] - yl[0]), pmax(t["mannwhitney_p"], t["welch_p"]))
    a.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))
    a.set_xlim(-0.5, 1.5)
    cohort_ticks(a)
    if ylabel:
        a.set_ylabel(ylabel)


def plot_decomposition(ax, df):
    """strength W, A | angle W, A (shared y within a term); the a = rho cos theta schematic is in the explainer panel"""
    ax.axis("off")
    widths = [0.2, 0.2, 0.2, 0.2]
    gaps = [0.03, 0.14, 0.03]
    x, insets = 0.0, []
    for k, w in enumerate(widths):
        insets.append(ax.inset_axes([x, 0, w, 1]))
        x += w + (gaps[k] if k < len(gaps) else 0)
    for j, term in enumerate(("strength", "angle")):
        g0 = df[df.term == term]
        lim = _padded(*_ms_range(g0, ["stim", "cohort"]))
        for k, s in enumerate("WA"):
            a = insets[2 * j + k]
            lab = ("Δ response strength ρ\n(post − pre)" if term == "strength"
                   else "Δ angle θ to axis (°)\n(post − pre; data − shifted)")
            _two_cohorts(a, g0[g0.stim == s], lim, lab if k == 0 else None)
            if k:
                a.set_yticklabels([])
            _mod_title(a, s, pad=16)


AC_MIN_SESS = 3


def compute_area_contrib():
    """202: contribution of each area group to the whole-brain whisker change on the whole-brain axis (data - shifted)"""
    d = pd.read_parquet(TAB / "202_area_contributions_sessions.parquet")
    d = d[(d.stim == "W") & (d.kind == "excess") & d.session_id.isin(whole().session_id)]
    keep = d.groupby(["area", "cohort"]).session_id.nunique().unstack().fillna(0)
    keep = keep[(keep["R+"] >= AC_MIN_SESS) & (keep["R-"] >= AC_MIN_SESS)].index
    tot = d.drop_duplicates("session_id")
    d = d[d.area.isin(keep)][["session_id", "mouse_id", "cohort", "area", "n_area", "contrib", "total", "per_neuron", "z"]]
    # 2026-11-25: angle change of each area read out on its part of the whole-brain axis (179 wb_area_group; data - shifted)
    c = clean()
    c = c[(c.area_col == "wb_area_group") & (c.stim == "W")]
    if len(c):
        c = c.assign(dth=(theta(c.cos_post) - theta(c.cos_pre)) - (theta(c.cos_post_null) - theta(c.cos_pre_null)))
        d = d.merge(c[["session_id", "area", "dth"]], on=["session_id", "area"], how="left")
    meta = dict(total=tests(tot, "total"), n_areas=len(keep))
    if "dth" in d:
        meta["angle"] = {a: tests(g, "dth") for a, g in d.groupby("area") if g.dth.notna().sum() >= 6}
    META["area_contrib"] = meta
    return d


def plot_area_contrib(ax, df):
    """left: each area's R- minus R+ contribution to the whole-brain change (sessions without the area count 0);
    middle: angle change of the area read out on its part of the whole-brain axis (data - shifted; 2026-11-25, replaces the
    per-neuron contribution); right: enrichment z (vs random groups of the same size in the session)"""
    col, _, _ = area_palette()
    ses = df.drop_duplicates("session_id")[["session_id", "cohort", "total"]]
    full = df.pivot_table(index="session_id", columns="area", values="contrib", fill_value=0.0).reindex(ses.session_id).fillna(0)
    full["cohort"] = ses.set_index("session_id").cohort
    diff = {a: full[full.cohort == "R-"][a].mean() - full[full.cohort == "R+"][a].mean() for a in df.area.unique()}
    order = sorted(diff, key=diff.get)
    y = np.arange(len(order))[::-1]
    ax.axis("off")
    insets = [ax.inset_axes([x0, 0, w, 1]) for x0, w in ((0.0, 0.3), (0.36, 0.3), (0.72, 0.28))]
    a = insets[0]
    a.barh(y, [diff[k] for k in order], color=[col[k] for k in order], height=0.7, lw=0)
    for yy, k in zip(y, order):
        t = two_group_tests(full[full.cohort == "R+"][k], full[full.cohort == "R-"][k])
        p = pmax(t["mannwhitney_p"], t["welch_p"])
        if p < 0.05:
            a.annotate(stars(p), (diff[k], yy), xytext=(-2, -1.5), textcoords="offset points", ha="right", va="center", fontsize=FS)
    tt = two_group_tests(ses[ses.cohort == "R+"].total, ses[ses.cohort == "R-"].total)
    dtot = ses[ses.cohort == "R-"].total.mean() - ses[ses.cohort == "R+"].total.mean()
    a.set_title(f"Share of the whole-brain change\n(R− − R+ = {dtot:.3f}, {p_text(pmax(tt['mannwhitney_p'], tt['welch_p']))})",
                fontsize=FS + 0.5, pad=2)
    a.set_xlabel("R− minus R+ contribution\n(data − shifted)")
    mid = (("dth", "Δ angle to the whole-brain axis (°)\n(area's part; data − shifted)", "Angle change")
           if "dth" in df and df.dth.notna().any() else
           ("per_neuron", "Per-neuron contribution (× √N)\n(data − shifted)", "Per neuron"))
    for j, (meas, lab, ttl) in enumerate((mid, ("z", "Enrichment z (vs random groups\nof the same size)", "Enrichment"))):
        a = insets[1 + j]
        for c, off in (("R+", 0.15), ("R-", -0.15)):
            m = [mean_sem(df[(df.area == k) & (df.cohort == c)][meas]) for k in order]
            a.errorbar([v[0] for v in m], y + off, xerr=[v[1] for v in m], fmt="o", ms=2.4, color=COLORS[c], lw=0.7, capsize=0)
        for yy, k in zip(y, order):
            g = df[df.area == k]
            t = two_group_tests(g[g.cohort == "R+"][meas], g[g.cohort == "R-"][meas])
            p = pmax(t["mannwhitney_p"], t["welch_p"])
            if np.isfinite(p) and p < 0.05:
                a.annotate(stars(p), (1.0, yy), xycoords=("axes fraction", "data"), ha="left", va="center", fontsize=FS)
        if meas == "z":
            for v in (-1.96, 1.96):
                a.axvline(v, color="0.85", lw=0.4)
        if meas == "dth":                       # user 2026-11-25: expected changes are small; show -10..20 deg, arrows if outside
            lo, hi = -10.0, 20.0
            a.set_xlim(lo, hi)
            for c, off in (("R+", 0.15), ("R-", -0.15)):
                for yy, k in zip(y, order):
                    m_ = mean_sem(df[(df.area == k) & (df.cohort == c)][meas])[0]
                    if np.isfinite(m_) and not lo <= m_ <= hi:
                        a.plot(hi if m_ > hi else lo, yy + off, marker=">" if m_ > hi else "<", ms=2.6, color=COLORS[c],
                               lw=0, clip_on=False, zorder=4)
        a.set_xlabel(lab)
        a.set_title(ttl, fontsize=FS + 0.5, pad=2)
    for j, a in enumerate(insets):
        a.set_ylim(-0.7, len(order) - 0.3)
        a.axvline(0, color="0.5", lw=0.4)
        a.tick_params(axis="y", length=0, labelsize=FS)
        a.set_yticks(y, [area_label(k) for k in order] if j == 0 else [""] * len(order))
        if j == 0:
            for t_, k in zip(a.get_yticklabels(), order):
                t_.set_color(col[k])
    insets[1].text(0.02, 1.0, "R+", transform=insets[1].transAxes, ha="left", va="top", color=COLORS["R+"], fontsize=FS + 0.5)
    insets[1].text(0.02, 0.94, "R−", transform=insets[1].transAxes, ha="left", va="top", color=COLORS["R-"], fontsize=FS + 0.5)


# ---- learning curves (195): examples, trial types x curves, areas

LC_CATS = (("hit", "W hits"), ("miss", "W misses"), ("aud_hit", "A hits"))   # all misses of the trimmed session (user 2026-11-25)
LC_CURVES = (("p", "P(lick | whisker)"), ("wfa", "P(lick | W) − P(lick | no stim.)"))


def _lc_sessions():
    s = pd.read_parquet(TAB / "195_hit_trajectory_sessions.parquet")
    return s[s.skipped_reason.isna() & s.mouse_id.isin(learners())].rename(columns={"reward_group": "cohort"})


LC_EXAMPLE_MOUSE = {"R+": "AB119", "R-": "AB158"}   # user 2026-11-25 (AB119; AB095 has no passive epochs -> not analysed)


def _lc_example(cohort, cat, outcome, engaged_only, nmin_col):
    """example session: LC_EXAMPLE_MOUSE[cohort] (day 0); without one, the session whose Spearman r (data - shifted) is
    closest to its cohort mean, >= 25 trials of the type"""
    s = _lc_sessions()
    m = f"{cat}_p_spearman_excess"
    s = s[(s.group == "whole_brain") & (s.cohort == cohort)]
    if LC_EXAMPLE_MOUSE.get(cohort) in set(s.mouse_id):
        row = s[s.mouse_id == LC_EXAMPLE_MOUSE[cohort]].iloc[0]
    else:
        s = s[s[nmin_col] >= 25].dropna(subset=[m])
        row = s.iloc[(s[m] - s[m].mean()).abs().argsort()].iloc[0]
    t = pd.read_parquet(TAB / "195_hit_trajectory_trials.parquet")
    t = t[(t.session_id == row.session_id) & (t.group == "whole_brain")].sort_values("start_time").copy()
    t["ex"] = t.proj - t.proj_null
    t["t_min"] = (t.start_time - t.start_time.min()) / 60
    meta = dict(session_id=row.session_id, mouse_id=row.mouse_id)
    for k in ("hit", "miss", "aud_hit"):
        meta[k] = dict(r=float(row[f"{k}_p_spearman_r"]), r_null=float(row[f"{k}_p_spearman_null"]),
                       excess=float(row[f"{k}_p_spearman_excess"]), n=int(row[f"n_{k}"]))
    META[f"lc_example_{'rp' if cohort == 'R+' else 'rm'}"] = meta
    return t[["session_id", "outcome", "t_min", "ex", "p_curve"]].assign(cohort=cohort, mouse_id=row.mouse_id)


def compute_lc_example_rp():
    return _lc_example("R+", "hit", "hit", False, "n_hit")


def compute_lc_example_rm():
    return _lc_example("R-", "miss", "miss", False, "n_miss")


LC_EX_TYPES = (("hit", "W hits", W_COL), ("miss", "W misses", MISS_COL), ("aud_hit", "A hits", A_COL))


def _plot_lc_example(ax, df):
    """every active trial type of the session against time (user 2026-11-25): dots = trials, lines = running mean of 10
    trials of the type (data - shifted); black = exact whisker learning curve at the whisker trials (right axis)"""
    c = df.cohort.iloc[0]
    for k, lab, cl in LC_EX_TYPES:
        g = df[df.outcome == k].sort_values("t_min")
        if len(g) < 3:
            continue
        ax.scatter(g.t_min, g.ex, s=1.6, color=light(cl, 0.45), lw=0, zorder=1, rasterized=False)
        ax.plot(g.t_min, g.ex.rolling(10, center=True, min_periods=5).mean(), color=cl, lw=0.9, zorder=2, label=lab)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel("Time in session (min)")
    if c == "R+":                                  # one left label (R+) and one right label (R-) for the pair
        ax.set_ylabel("Proj. on hit − miss axis\n(data − shifted)")
        ax.legend(fontsize=FS - 0.8, loc="upper left", handlelength=1.0, borderaxespad=0.1, labelspacing=0.2)
    a2 = ax.twinx()
    ax.set_zorder(a2.get_zorder() + 1)                 # trials in front of the learning-curve twin axis
    ax.patch.set_visible(False)
    w = df[df.outcome.isin(["hit", "miss"])].sort_values("t_min")
    a2.plot(w.t_min, w.p_curve, color="0.1", lw=0.8)
    a2.set_ylim(0, 1)
    if c == "R-":
        a2.set_ylabel("P(lick | whisker)", fontsize=FS)
    a2.tick_params(labelsize=FS)
    a2.spines["top"].set_visible(False)
    ax.set_title(f"{LAB[c]} example ({df.mouse_id.iloc[0]})", fontsize=FS + 0.5, color=COLORS[c], pad=2)


def plot_lc_example_rp(ax, df):
    _plot_lc_example(ax, df)


def plot_lc_example_rm(ax, df):
    _plot_lc_example(ax, df)


def compute_lc_types():
    s = _lc_sessions()
    s = s[s.group == "whole_brain"]
    rows, meta = [], {}
    for cat, _ in LC_CATS:
        for cv, _ in LC_CURVES:
            col = f"{cat}_{cv}_spearman_excess"
            if col not in s:
                continue
            g = pd.DataFrame(dict(session_id=s.session_id, cohort=s.cohort, cat=cat, curve=cv, value=s[col])).dropna()
            rows.append(g)
            vs0 = {}
            for c in COH:
                v = g[g.cohort == c].value
                vs0[f"vs0_{c}"] = float(pmax(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue)) if len(v) >= 5 else np.nan
            meta[f"{cat}_{cv}"] = dict(tests(g, "value"), **vs0)
    META["lc_types"] = meta
    return pd.concat(rows, ignore_index=True)


def plot_lc_types(ax, df):
    """two curves side by side; x = trial type, R+ / R- offset; stars vs 0 in cohort colour, R+ vs R- above each type"""
    insets = _mm_insets(ax, [AX_MM, AX_MM], [2.5])
    ms_ = df.groupby(["curve", "cat", "cohort"]).value.agg(["mean", "sem"])     # no dots (user 2026-11-26): zoom on means
    lim = _padded(float((ms_["mean"] - ms_["sem"]).min()), float((ms_["mean"] + ms_["sem"]).max()))
    lim = (min(lim[0], 0.0), max(lim[1], 0.0))
    for j, (a, (cv, lab)) in enumerate(zip(insets, LC_CURVES)):
        g0 = df[df.curve == cv]
        for i, (cat, _) in enumerate(LC_CATS):
            for c, off in (("R+", -0.15), ("R-", 0.15)):
                v = g0[(g0.cat == cat) & (g0.cohort == c)].value
                if len(v) < 3:
                    continue
                m, e = mean_sem(v)
                a.errorbar(i + off, m, e, color=COLORS[c], marker="o", ms=2.6, capsize=0, lw=0.9)
                p = pmax(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue)
                if p < 0.05:
                    a.annotate(stars(p), (i + off, m + e), xytext=(0, 0.5), textcoords="offset points", ha="center",
                               va="bottom", fontsize=FS - 0.5, color=COLORS[c])
        a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
        a.set_ylim(lim[0], lim[1] + 0.35 * (lim[1] - lim[0]))
        for i, (cat, _) in enumerate(LC_CATS):
            g = g0[g0.cat == cat]
            t = two_group_tests(g[g.cohort == "R+"].value, g[g.cohort == "R-"].value)
            p = pmax(t["mannwhitney_p"], t["welch_p"])
            yb = lim[1] + 0.15 * (lim[1] - lim[0])
            a.plot([i - 0.15, i + 0.15], [yb, yb], color="k", lw=0.4)
            a.annotate(stars(p) if p < 0.05 else "n.s.", (i, yb), xytext=(0, 0.5), textcoords="offset points", ha="center",
                       va="bottom", fontsize=FS - 0.5)
        a.set_xticks(range(len(LC_CATS)), [l.replace(" ", "\n") for _, l in LC_CATS], fontsize=FS)
        a.set_xlim(-0.5, len(LC_CATS) - 0.5)
        a.set_title(lab, fontsize=FS + 0.3, pad=2)
        if j == 0:
            a.set_ylabel("Projection vs curve, Spearman r\n(data − shifted)")
        else:
            a.set_yticklabels([])


def compute_lc_areas():
    d = compute_learning_curve()
    META["lc_areas"] = META.pop("learning_curve")
    return d[d.group != "whole_brain"]


def plot_lc_areas(ax, df):
    """user 2026-11-25 ("yes replace P"): named area groups as rows (palette order, area colour), whisker hits vs
    P(lick | W) (data - shifted), R+ / R- mean +- s.e.m. over sessions; filled = different from 0 (larger P of Wilcoxon / t);
    stars = R+ vs R- (larger P of Mann-Whitney / Welch); same style as the area panel of the second section"""
    col, order, _ = area_palette()
    n = df.groupby(["group", "cohort"]).session_id.nunique().unstack().fillna(0)
    rows = [a for a in order if a in n.index and n.loc[a].get("R+", 0) >= LC_MIN_SESS and n.loc[a].get("R-", 0) >= LC_MIN_SESS]
    y = np.arange(len(rows))
    for c, off in (("R+", -0.15), ("R-", 0.15)):
        for i, a in enumerate(rows):
            v = df[(df.group == a) & (df.cohort == c)].value.dropna()
            m, e = mean_sem(v)
            sig0 = len(v) >= 4 and pmax(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue) < 0.05
            ax.errorbar(m, i + off, xerr=e, fmt="o", ms=2.4, color=COLORS[c], mfc=COLORS[c] if sig0 else "white", mew=0.6,
                        lw=0.7, capsize=0, zorder=3)
    for i, a in enumerate(rows):
        g = df[df.group == a]
        t = two_group_tests(g[g.cohort == "R+"].value, g[g.cohort == "R-"].value)
        p = pmax(t["mannwhitney_p"], t["welch_p"])
        if p < 0.05:
            ax.annotate(stars(p), (1.0, i), xycoords=("axes fraction", "data"), ha="left", va="center", fontsize=FS)
    ax.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(y, [area_label(a) for a in rows])
    ax.tick_params(axis="y", length=0, labelsize=FS)
    for t_, a in zip(ax.get_yticklabels(), rows):
        t_.set_color(col[a])
    ax.xaxis.set_major_locator(__import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(3))
    ax.set_xlabel("W hits vs P(lick | W),\nSpearman r (data − shifted)")
    ax.text(1.0, 1.02, "filled: ≠ 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS - 0.8, color="0.4")
    return


def _plot_lc_areas_old(ax, df):
    """(previous version, kept) one dot per area group and cohort, lines join an area's R+ and R- values"""
    col, _, _ = area_palette()
    am = df.groupby(["group", "cohort"]).value.agg(["mean", "count"]).reset_index()
    am = am[am["count"] >= LC_MIN_SESS]
    w = am.pivot(index="group", columns="cohort", values="mean")
    for gname, r in w.iterrows():
        if r.notna().all():
            ax.plot([0, 1], [r["R+"], r["R-"]], color="0.82", lw=0.5, zorder=1)
        for i, c in enumerate(COH):
            if np.isfinite(r.get(c, np.nan)):
                ax.scatter(i, r[c], s=8, color=col[gname], lw=0, zorder=2)
    for i, c in enumerate(COH):
        m, e = mean_sem(w[c])
        ax.errorbar(i + (-0.25 if i == 0 else 0.25), m, e, color=COLORS[c], marker="s", ms=3, capsize=0, lw=1.0, zorder=3)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    both = w.dropna()
    yl = ax.get_ylim()
    bracket(ax, 0, 1, yl[1] + 0.05 * (yl[1] - yl[0]),
            pmax(stats.wilcoxon(both["R+"], both["R-"]).pvalue, stats.ttest_rel(both["R+"], both["R-"]).pvalue))
    ax.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))
    ax.set_xlim(-0.6, 1.6)
    cohort_ticks(ax)
    ax.set_ylabel("W hits vs P(lick | W), Spearman r\n(data − shifted), area mean")


# ---- explainer (user 2026-11-25: "I miss a figure that explains what we are looking at"): draft for a Figma redraw, as H

def compute_explainer():
    """drawing only (no data); one row so build.py calls plot_explainer"""
    return pd.DataFrame(dict(drawing=["explainer"]))


def plot_explainer(ax, df):
    """top: what is measured on the hit - miss axis (projection a, angle theta, strength rho; R+ post stays aligned, R- post
    turns away); bottom: the control (axes rebuilt with hit / miss labels slid in time; reported = real - mean of shifted)"""
    ax.axis("off")
    top = ax.inset_axes([0, 0.45, 1, 0.5])
    bot = ax.inset_axes([0, 0, 1, 0.33])
    for a in (top, bot):
        a.axis("off")
    # ---- top: vectors drawn in a box of fixed data range (not equal aspect: the panel box sets the shape)
    a = top
    a.set_xlim(-0.08, 1.3)
    a.set_ylim(-0.32, 0.95)
    a.annotate("", (1.25, 0), (-0.02, 0), arrowprops=dict(arrowstyle="-|>", color="0.1", lw=0.9, mutation_scale=6))
    a.text(1.25, -0.06, "hit − miss axis", ha="right", va="top", fontsize=FS - 0.5)
    vecs = (("pre", 22, 0.95, "0.55", (0.03, -0.02), "left"), ("R+ post", 26, 0.88, COLORS["R+"], (0.03, 0.0), "left"),
            ("R− post", 58, 0.82, COLORS["R-"], (-0.03, 0.02), "right"))
    for lab, th, r, col, off, ha in vecs:
        x, y = r * np.cos(np.radians(th)), r * np.sin(np.radians(th))
        a.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=6))
        a.plot([x, x], [0, y], color=col, lw=0.5, ls=(0, (1.5, 1.5)))
        a.plot([x], [0], marker="D", ms=2, color=col)
        a.text(x + off[0], y + off[1], lab, color=col, fontsize=FS - 0.5, va="bottom", ha=ha)
    t = np.radians(np.linspace(0, 58, 30))
    a.plot(0.2 * np.cos(t), 0.2 * np.sin(t), color=COLORS["R-"], lw=0.5)
    a.text(0.22, 0.07, "θ", fontsize=FS, color=COLORS["R-"])
    a.text(0.13, 0.4, "ρ", fontsize=FS, color=COLORS["R-"])
    a.text(0.62, -0.2, "projection a = ρ cos θ (diamonds)", ha="center", va="top", fontsize=FS - 0.5, color="0.3")
    a.set_title("Passive responses on the axis", fontsize=FS + 0.3, pad=2)
    # ---- bottom: control
    a = bot
    a.set_xlim(-0.08, 1.3)
    a.set_ylim(-0.3, 0.8)
    rng = np.random.default_rng(3)
    for th in rng.uniform(20, 80, 12):
        a.annotate("", (0.62 * np.cos(np.radians(th)), 0.62 * np.sin(np.radians(th))), (0, 0),
                   arrowprops=dict(arrowstyle="-|>", color="0.75", lw=0.6, mutation_scale=4))
    a.annotate("", (0.9, 0.0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="0.1", lw=0.9, mutation_scale=6))
    a.text(0.55, 0.5, "50 axes, hit / miss\nlabels slid in time", fontsize=FS - 1, color="0.45", ha="left", va="center")
    a.text(0.93, 0.0, "real axis", fontsize=FS - 1, color="0.1", ha="left", va="center")
    a.text(0.62, -0.12, "reported: real − mean of shifted", ha="center", va="top", fontsize=FS - 0.5)
    a.set_title("Control: linear shifts", fontsize=FS + 0.3, pad=2)


# ---- control: pre-task whisker response strength vs the passive change (user 2026-11-25, "show that in figures")

def compute_strength_control():
    """per session (whole brain, whisker, learners): pre-task response strength rho_pre (cross-validated norm) against the
    projection change and the angle change (both post - pre, data - shifted); Spearman per cohort"""
    w = whole("W").copy()
    w["dth"] = (theta(w.cos_post) - theta(w.cos_pre)) - (theta(w.cos_post_null) - theta(w.cos_pre_null))
    d = w[["session_id", "mouse_id", "cohort", "rho_pre", "q", "dth"]].rename(columns={"q": "dq"})
    meta = {}
    for y in ("dq", "dth"):
        for c in COH:
            g = d[d.cohort == c][["rho_pre", y]].dropna()
            r, p = stats.spearmanr(g.rho_pre, g[y])
            meta[f"{y}_{c}"] = dict(rho=float(r), p=float(p), n=int(len(g)))
    meta["rho_pre"] = tests(d, "rho_pre")
    META["strength_control"] = meta
    return d


def plot_strength_control(ax, df):
    """left: projection change vs pre-task strength (scales); right: angle change vs pre-task strength (flat)"""
    for a, (y, lab) in zip(_pair(ax, 0.3), (("dq", "Δ whisker proj. on axis\n(post − pre; data − shifted)"),
                                            ("dth", "Δ whisker angle to axis (°)\n(post − pre; data − shifted)"))):
        for k, c in enumerate(COH):
            g = df[df.cohort == c][["rho_pre", y]].dropna()
            ols_band(a, g.rho_pre, g[y], COLORS[c])
            a.scatter(g.rho_pre, g[y], s=5, color=light(COLORS[c], 0.35), lw=0, zorder=2)
            r, p = stats.spearmanr(g.rho_pre, g[y])
            a.text(0.03, 0.97 - 0.09 * k, f"{LAB[c]}: ρ = {r:.2f}, {p_text(p)}", transform=a.transAxes, color=COLORS[c],
                   fontsize=FS, va="top")
        a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
        a.set_xlabel("Pre-task whisker response\nstrength ρ (passive pre)")
        a.set_ylabel(lab)


# ---- harmonised changes panel (user 2026-11-25: "harmonize J and K in all ways and place this on the side of the
# time-resolved projections"): projection, strength and angle changes, whisker | auditory, same layout and tests

CH_TERMS = (("angle", "Δ angle θ (°)"), ("strength", "Δ strength ρ"), ("projection", "Δ projection a"))   # angle first (user 2026-11-25)
CH_SHOW = ("angle", "strength")   # main: angle and strength; projection (their combination) in supp (user 2026-11-25)


def compute_changes():
    rows, meta = [], {}
    for s in ("W", "A"):
        w = whole(s)
        ang = (theta(w.cos_post) - theta(w.cos_pre)) - (theta(w.cos_post_null) - theta(w.cos_pre_null))
        for term, v in (("projection", w.q), ("strength", w.rho_post - w.rho_pre), ("angle", ang)):
            if term not in CH_SHOW:
                continue
            rows.append(pd.DataFrame(dict(session_id=w.session_id, mouse_id=w.mouse_id, cohort=w.cohort, stim=s, term=term,
                                          value=v.to_numpy())))
    d = pd.concat(rows, ignore_index=True)
    for term, _ in CH_TERMS:
        if term not in CH_SHOW:
            continue
        g = d[d.term == term]
        for s in ("W", "A"):
            meta[f"{term}_{s}"] = tests(g[g.stim == s], "value")
        p = g.pivot_table(index=["session_id", "cohort"], columns="stim", values="value").reset_index()
        p["WA"] = p.W - p.A
        meta[f"{term}_WA"] = tests(p, "WA")
    META["changes"] = meta
    return d


def plot_changes(ax, df):
    """three pairs (projection, strength, angle) x (whisker, auditory); R+ / R- mean +- s.e.m.; bracket R+ vs R- per modality;
    under each pair the cohort x modality test on whisker - auditory"""
    ax.axis("off")
    wmm = _ax_width_mm(ax)
    gin, gout, w_in = 2.5, 11.0, AX_MM * 0.6           # compact (user 2026-11-26: "a lot more compact", no dots)
    pw, gp, gi, iw = (2 * w_in + gin) / wmm, gout / wmm, gin / wmm, w_in / wmm
    terms = [t for t in CH_TERMS if t[0] in set(df.term)]
    for j, (term, lab) in enumerate(terms):
        g0 = df[df.term == term]
        lo_, hi_ = _ms_range(g0, ["stim", "cohort"])   # zoom on the means (user 2026-11-25)
        sp = hi_ - lo_
        lim = (min(lo_, 0.0) - 0.25 * sp, max(hi_, 0.0) + 0.25 * sp)
        x0 = j * (pw + gp)
        for k, s in enumerate("WA"):
            a = ax.inset_axes([x0 + k * (iw + gi), 0, iw, 1])
            _two_cohorts(a, g0[g0.stim == s], lim, lab if k == 0 else None, dots=False)
            if k:
                a.set_yticklabels([])
            _mod_title(a, s, pad=2)
            a.tick_params(labelsize=FS)
            a.yaxis.set_major_locator(__import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(3))
        p = g0.pivot_table(index=["session_id", "cohort"], columns="stim", values="value").reset_index()
        t = two_group_tests(p[p.cohort == "R+"].W - p[p.cohort == "R+"].A, p[p.cohort == "R-"].W - p[p.cohort == "R-"].A)
        pv = pmax(t["mannwhitney_p"], t["welch_p"])
        ax.text(x0 + pw / 2, -0.2, f"W − A: {p_text(pv)} {stars(pv) if pv < 0.05 else ''}".strip(), transform=ax.transAxes,
                ha="center", va="top", fontsize=FS - 0.3)



# ---- uniform panel sizes (user 2026-11-25: "too many panels of different sizes"): every data axis AX_MM x AX_MM

AX_MM = 22.0


def _ax_width_mm(ax):
    fw = ax.figure.get_size_inches()[0] * 25.4
    return ax.get_position().width * fw


def _mm_insets(ax, widths_mm, gaps_mm):
    """insets of exact widths (mm) left to right inside ax (ax hidden), full height"""
    ax.axis("off")
    W = _ax_width_mm(ax)
    out, x = [], 0.0
    for k, w in enumerate(widths_mm):
        out.append(ax.inset_axes([x / W, 0, w / W, 1]))
        x += w + (gaps_mm[k] if k < len(gaps_mm) else 0)
    return out


def compute_angle_wa():
    d = compute_wa_angle_prepost()
    META["angle_wa"] = META.pop("wa_angle_prepost")
    return d


def plot_angle_wa(ax, df):
    """angle pre / post, whisker | auditory, each inset AX_MM wide, shared y"""
    lim = _padded(*_ms_range(df, ["stim", "cohort", "kind", "epoch"]), zero=False)
    for a, s_ in zip(_mm_insets(ax, [AX_MM, AX_MM], [2.5]), "WA"):
        _prepost_plot(a, df[df.stim == s_], "Angle to hit − miss axis (°)" if s_ == "W" else "", ylim=lim, legend=s_ == "W")
        _mod_title(a, s_, pad=12)
        if s_ == "A":
            a.set_yticklabels([])



# ---- angle change vs pre-task strength (single AX_MM panel; main row 2)

def compute_strength_angle():
    d = compute_strength_control()
    META["strength_angle"] = META.pop("strength_control")
    return d


def plot_strength_angle(ax, df):
    a = _mm_insets(ax, [AX_MM], [])[0]
    for k, c in enumerate(COH):
        g = df[df.cohort == c][["rho_pre", "dth"]].dropna()
        a.scatter(g.rho_pre, g.dth, s=2.5, color=light(COLORS[c], 0.35), lw=0, zorder=2)
        ols_band(a, g.rho_pre, g.dth, COLORS[c])
        r, p = stats.spearmanr(g.rho_pre, g.dth)
        a.text(0.02, 1.02 + 0.1 * (1 - k), f"{LAB[c]}: ρ = {r:.2f}, {p_text(p)}", transform=a.transAxes, color=COLORS[c],
               fontsize=FS - 0.5, va="bottom")
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_xlabel("Pre-task response\nstrength ρ")
    a.set_ylabel("Δ whisker angle (°)\n(data − shifted)")


# ---- expert section (user 2026-11-25): rate, projection and angle change, learning day vs expert days, W | A

ST_TERMS = (("rate", "Δ evoked rate (spks/s)"), ("projection", "Δ projection a"), ("angle", "Δ angle θ (°)"))


def compute_stages():
    rows, meta = [], {}
    for stage, lab in (("learning", "day 0"), ("expert", "expert")):
        for s in ("W", "A"):
            w = whole(s, stage)
            ang = (theta(w.cos_post) - theta(w.cos_pre)) - (theta(w.cos_post_null) - theta(w.cos_pre_null))
            d = pd.DataFrame(dict(mouse_id=w.mouse_id, cohort=w.cohort, rate=(w.ev_post - w.ev_pre).to_numpy(),
                                  projection=w.q.to_numpy(), angle=ang.to_numpy()))
            d = d.groupby(["mouse_id", "cohort"]).mean(numeric_only=True).reset_index()     # mouse = unit (expert averaged)
            for term, _ in ST_TERMS:
                rows.append(d[["mouse_id", "cohort"]].assign(stage=lab, stim=s, term=term, value=d[term].to_numpy()))
    out = pd.concat(rows, ignore_index=True)
    for (st, s, term), g in out.groupby(["stage", "stim", "term"]):
        meta[f"{term}_{s}_{st.replace(' ', '')}"] = tests(g, "value")
    META["stages"] = meta
    return out


def plot_stages(ax, df):
    """three pairs (rate, projection, angle) x (whisker, auditory); x = learning day, expert; R+ / R- mean +- s.e.m. over mice;
    stars above each stage = R+ vs R- (larger P of Mann-Whitney / Welch)"""
    gin, gout = 2.5, 14.0
    widths, gaps = [], []
    for j in range(len(ST_TERMS)):
        widths += [AX_MM, AX_MM]
        gaps += [gin] + ([gout] if j < len(ST_TERMS) - 1 else [])
    insets = _mm_insets(ax, widths, gaps)
    for j, (term, lab) in enumerate(ST_TERMS):
        g0 = df[df.term == term]
        lo, hi = _ms_range(g0, ["stage", "stim", "cohort"])
        lim = _padded(lo, hi, 0.1)
        for k, s in enumerate("WA"):
            a = insets[2 * j + k]
            g = g0[g0.stim == s]
            for c, off in (("R+", -0.06), ("R-", 0.06)):
                m = [mean_sem(g[(g.cohort == c) & (g.stage == st)].value) for st in ("day 0", "expert")]
                a.plot([off, 1 + off], [v[0] for v in m], color=COLORS[c], lw=0.9)
                a.errorbar([off, 1 + off], [v[0] for v in m], [v[1] for v in m], color=COLORS[c], marker="o", ms=2.6,
                           lw=0, elinewidth=0.8, capsize=0)
            a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
            a.set_ylim(lim[0], lim[1] + 0.25 * (lim[1] - lim[0]))
            for i, st in enumerate(("day 0", "expert")):
                gg = g[g.stage == st]
                t = two_group_tests(gg[gg.cohort == "R+"].value, gg[gg.cohort == "R-"].value)
                p = pmax(t["mannwhitney_p"], t["welch_p"])
                a.annotate(stars(p) if p < 0.05 else "n.s.", (i, 1.0), xycoords=("data", "axes fraction"), ha="center",
                           va="top", fontsize=FS - 0.5)
            a.set_xlim(-0.4, 1.4)
            a.set_xticks([0, 1], ["Day 0", "Expert"], fontsize=FS)
            _mod_title(a, s, pad=2)
            a.yaxis.set_major_locator(__import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(3))
            if k == 0:
                a.set_ylabel(lab)
            else:
                a.set_yticklabels([])


# ---- behavioural state in the passive epochs (205; user 2026-11-25: "compare pupil area as well between cohorts - they
# have reached the same state"): pupil area before the stimulus, passive pre vs post, whisker trials, learning day

def compute_pupil():
    d = pd.read_parquet(TAB / "205_passive_state_dlc.parquet")
    d = d[d.skipped_reason.isna() & (d.trial_type == "whisker_trial") & d.mouse_id.isin(learners())]
    w = d.pivot_table(index=["session_id", "mouse_id", "cohort"], columns="epoch", values="pupil_base").dropna().reset_index()
    w["delta"] = w["post"] - w["pre"]
    META["pupil"] = dict(delta=tests(w, "delta"), **{c: paired(w[w.cohort == c].pre, w[w.cohort == c].post) for c in COH})
    return w[["session_id", "mouse_id", "cohort", "pre", "post", "delta"]]


def plot_pupil(ax, df):
    """left: pupil area before the stimulus, passive pre and post (mean +- s.e.m., sessions grey); right: change post - pre,
    R+ vs R- (sessions grey)"""
    a, b = _mm_insets(ax, [AX_MM, AX_MM], [10.0])
    for c, off in (("R+", -0.06), ("R-", 0.06)):
        g = df[df.cohort == c]
        for _, r in g.iterrows():
            a.plot([off, 1 + off], [r.pre, r.post], color="0.85", lw=0.3, zorder=0)
        m = [mean_sem(g[e]) for e in ("pre", "post")]
        a.errorbar([off, 1 + off], [v[0] for v in m], [v[1] for v in m], color=COLORS[c], marker="o", ms=2.6, lw=0.9,
                   capsize=0, zorder=3)
    a.set_xticks([0, 1], ["Pre", "Post"])
    a.set_xlim(-0.4, 1.4)
    a.set_ylabel("Pupil area before stimulus (mm²)")
    g = df.rename(columns={"delta": "value"})
    lo, hi = _ms_range(g, ["cohort"])
    sp = hi - lo
    _two_cohorts(b, g, (min(lo, 0) - 0.9 * sp, max(hi, 0) + 0.9 * sp), "Δ pupil area (mm²)\n(post − pre)", dots=True)


# ---- pupil, raw values (user 2026-11-25: "show both baseline pupil area (1 s prestim) and after (1 s); raw values as barplots")

PUPIL_WINDOWS = (("before", (-1.0, 0.0), "1 s before stimulus"), ("after", (0.0, 1.0), "1 s after stimulus"))


def compute_pupil_bars():
    """205 per-session mean pupil traces (whisker trials, passive pre / post; blink / squint trials NaN), raw mm^2 averaged
    over 1 s before and 1 s after the stimulus"""
    t = pd.read_parquet(TAB / "205_passive_state_traces.parquet")
    t = t[(t.signal == "pupil_area") & (t.trial_type == "whisker_trial")]
    s = pd.read_parquet(TAB / "205_passive_state_dlc.parquet")[["session_id", "mouse_id"]].drop_duplicates()
    t = t.merge(s, on="session_id")
    t = t[t.mouse_id.isin(learners())]
    rows = []
    for name, (lo, hi), _ in PUPIL_WINDOWS:
        w = t[(t.time_s >= lo) & (t.time_s < hi)].groupby(["session_id", "mouse_id", "cohort", "epoch"]).value.mean()
        rows.append(w.rename("value").reset_index().assign(window=name))
    d = pd.concat(rows, ignore_index=True)
    meta = {}
    for name, _, _ in PUPIL_WINDOWS:
        p = d[d.window == name].pivot_table(index=["session_id", "cohort"], columns="epoch", values="value").dropna().reset_index()
        p["delta"] = p["post"] - p["pre"]
        meta[name] = dict(delta=tests(p, "delta"), pre=tests(p, "pre"), post=tests(p, "post"),
                          **{c: paired(p[p.cohort == c].pre, p[p.cohort == c].post) for c in COH})
    META["pupil_bars"] = meta
    return d


def plot_pupil_bars(ax, df):
    """one inset per window (before / after the stimulus); x = passive pre, post; bars = cohort mean (light), s.e.m.;
    sessions as grey dots; brackets: pre vs post within cohort (Wilcoxon / paired t), and R+ vs R- on the change"""
    insets = _mm_insets(ax, [AX_MM, AX_MM], [6.0])
    hi_all = float(np.nanpercentile(df.value, 98)) * 1.35
    for a, (name, _, lab) in zip(insets, PUPIL_WINDOWS):
        g0 = df[df.window == name]
        for e_i, ep in enumerate(("pre", "post")):
            for c, off in (("R+", -0.18), ("R-", 0.18)):
                v = g0[(g0.epoch == ep) & (g0.cohort == c)].value.dropna()
                m, se = mean_sem(v)
                x = e_i + off
                a.bar(x, m, 0.32, color=light(COLORS[c], 0.45), lw=0, zorder=1)
                a.errorbar(x, m, se, color=COLORS[c], capsize=0, lw=0.9, zorder=3)
                a.scatter(x + np.random.default_rng(e_i).uniform(-0.08, 0.08, len(v)), v, s=1.4, color="0.75", lw=0, zorder=2)
        p = g0.pivot_table(index=["session_id", "cohort"], columns="epoch", values="value").dropna().reset_index()
        for k, (c, off) in enumerate((("R+", -0.18), ("R-", 0.18))):
            q = p[p.cohort == c]
            pv = pmax(stats.wilcoxon(q.pre, q.post).pvalue, stats.ttest_rel(q.pre, q.post).pvalue)
            yb = hi_all * (0.86 + 0.07 * k)
            a.plot([off, 1 + off], [yb, yb], color=COLORS[c], lw=0.4)
            a.text(0.5 + off * 0.2, yb, stars(pv) if pv < 0.05 else "n.s.", ha="center", va="bottom", fontsize=FS - 0.5,
                   color=COLORS[c])
        t = two_group_tests(p[p.cohort == "R+"].post - p[p.cohort == "R+"].pre, p[p.cohort == "R-"].post - p[p.cohort == "R-"].pre)
        a.set_title(f"{lab}\nΔ R+ vs R−: {p_text(pmax(t['mannwhitney_p'], t['welch_p']))}", fontsize=FS, pad=2)
        a.set_ylim(0, hi_all)
        a.set_xticks([0, 1], ["Pre", "Post"])
        a.set_xlim(-0.55, 1.55)
    insets[0].set_ylabel("Pupil area (mm²)")
    insets[1].set_yticklabels([])


def compute_pupil_traces():
    """205 pupil traces (whisker trials), single-trial baseline: each trial minus its own mean over the 1 s before the
    stimulus, then averaged per session and epoch; learners"""
    t = pd.read_parquet(TAB / "205_passive_state_traces.parquet")
    t = t[(t.signal == "pupil_area") & (t.trial_type == "whisker_trial")]
    s = pd.read_parquet(TAB / "205_passive_state_dlc.parquet")[["session_id", "mouse_id"]].drop_duplicates()
    t = t.merge(s, on="session_id")
    t = t[t.mouse_id.isin(learners())].copy()
    if "value_bc" not in t:     # older table: session-level baseline (2026-11-25: 205 now stores the single-trial one)
        base = t[(t.time_s >= -1.0) & (t.time_s < 0)].groupby(["session_id", "epoch"]).value.mean().rename("base")
        t = t.join(base, on=["session_id", "epoch"])
        t["value_bc"] = t.value - t.base
    META["pupil_traces"] = {c: int(t[t.cohort == c].session_id.nunique()) for c in COH}
    return t[["session_id", "mouse_id", "cohort", "epoch", "time_s", "value_bc"]]


def plot_pupil_traces(ax, df):
    """one panel per passive epoch present in df (pre | post); R+ and R- overlaid (mean +- s.e.m. over sessions),
    baseline-corrected; eye ROI inset lower right of the first panel, below the traces"""
    # user 2026-11-25: -0.5..2 s around the stimulus; every axis labelled; eye ROI inset (eye_roi_inset.py -> _cache);
    # Figure 4F shows passive post only (pupil_post)
    eps = [e for e in ("pre", "post") if e in set(df.epoch)]
    insets = _mm_insets(ax, [AX_MM] * len(eps), [11.0] * (len(eps) - 1))
    df = df[(df.time_s >= -0.5) & (df.time_s <= 2.0)]
    g = df.groupby(["epoch", "cohort", "time_s"]).value_bc.agg(["mean", "sem"]).reset_index()
    lo = float((g["mean"] - g["sem"]).min())
    hi = float((g["mean"] + g["sem"]).max())
    lo, hi = _padded(lo, hi, 0.1)
    lim = (lo, hi + 0.9 * (hi - lo))                                   # room above the traces for the eye inset (user 2026-11-25)
    for j, (a, ep) in enumerate(zip(insets, eps)):
        for c in COH:
            h = g[(g.epoch == ep) & (g.cohort == c)].sort_values("time_s")
            a.fill_between(h.time_s, h["mean"] - h["sem"], h["mean"] + h["sem"], color=light(COLORS[c], 0.7), lw=0,
                           edgecolor="none")
            a.plot(h.time_s, h["mean"], color=COLORS[c], lw=0.9, label=f"{LAB[c]} ({df[df.cohort == c].session_id.nunique()})")
        a.axvline(0, color="0.4", lw=0.4, ls=(0, (2, 2)))
        a.axhline(0, color="0.6", lw=0.4)
        a.set_ylim(lim)
        a.set_xlim(-0.5, 2.0)
        a.set_xticks([-0.5, 0, 1, 2])
        if len(eps) > 1:
            a.set_title(f"Passive {ep}", fontsize=FS + 0.5, pad=2)
        a.set_xlabel("Time from stimulus (s)")
        a.set_ylabel("Δ pupil area (mm²)")
        if j == 0:
            a.legend(fontsize=FS - 0.5, frameon=False, loc="upper left",
                     handlelength=1.0)
    from common import CACHE_DIR
    eye = CACHE_DIR / "eye_roi_inset.png"
    if eye.exists():
        import matplotlib.image as mpimg
        e = insets[0].inset_axes([0.62, 0.60, 0.38, 0.38])          # upper right (user 2026-11-25), smaller (2026-11-26)
        e.imshow(mpimg.imread(eye), interpolation="lanczos")
        e.set_axis_off()


def compute_pupil_post():
    """Figure 4F (user 2026-11-25: "show only passive post with the inset")"""
    d = compute_pupil_traces()
    META["pupil_post"] = META["pupil_traces"]
    return d[d.epoch == "post"]


def plot_pupil_post(ax, df):
    plot_pupil_traces(ax, df)


# ================================================================== panels A-G rebuilt on this pipeline (user 2026-11-25:
# "can you reproduce the analyses for the A-G"; original: brain_wide_analysis/passive_analysis/psth_per_mouse_with_barplots.py).
# Same sessions, passive trials (ephys_utilities passive blocks, licked passive trials dropped) and neurons (137 stable good +
# MUA) as H-S; rates in spikes/s from 179 (5-35 ms evoked = rate minus the trial's own baseline -1 s..-15 ms; time courses from
# artefact-corrected spikes, 10-ms windows); DLC via ephys_utilities dlc_helpers (205). Differences to the original script:
# passive pre / post from the passive blocks (not a split at the middle trial index), stable units (not its own quality
# filter), per-trial baseline (not the -50..0 ms PSTH mean), expert results at mouse level, no BH-FDR (uncorrected, rule).

AG_T = np.round(np.arange(-0.045, 0.1001, 0.005), 3)          # 10-ms window centres stored by 179 (psth_*)


def _ag_units(stage="learning"):
    u = pd.read_parquet(TAB / f"179_cd_units{'' if stage == 'learning' else '_' + stage}.parquet")
    if stage == "learning":
        u = u[u.mouse_id.isin(learners())]
    return u.rename(columns={"reward_group": "cohort"})


def compute_ag_psth():
    """B: per session, mean over neurons of the baseline-subtracted time course (Hz), passive pre / post, whisker / auditory"""
    u = _ag_units()
    rows = []
    for (sid, coh), g in u.groupby(["session_id", "cohort"]):
        for e in ("pre", "post"):
            for s in ("W", "A"):
                col = f"psth_{e}_{s}"
                if col not in g:
                    continue
                M = np.stack([np.asarray(v, float) for v in g[col].to_numpy()])
                m = np.nanmean(M, 0)
                rows.append(pd.DataFrame(dict(session_id=sid, cohort=coh, epoch=e, stim=s, t=AG_T[:len(m)], value=m)))
    d = pd.concat(rows, ignore_index=True)
    META["ag_psth"] = {c: int(d[d.cohort == c].session_id.nunique()) for c in COH}
    return d


def plot_ag_psth(ax, df):
    """2 x 2: rows R+ / R-, columns whisker / auditory; pre grey, post cohort colour (mean +- s.e.m. over sessions)"""
    cols = _mm_insets(ax, [AX_MM, AX_MM], [5.0])
    g = df.groupby(["cohort", "stim", "epoch", "t"]).value.agg(["mean", "sem"]).reset_index()
    for j, (colax, s) in enumerate(zip(cols, "WA")):
        colax.axis("off")
        lim = _padded(float((g[g.stim == s]["mean"] - g[g.stim == s]["sem"]).min()),
                      float((g[g.stim == s]["mean"] + g[g.stim == s]["sem"]).max()), 0.08)
        for i, c in enumerate(COH):
            a = colax.inset_axes([0, 0.55 - 0.55 * i, 1, 0.45])
            for e, cl in (("pre", "0.55"), ("post", COLORS[c])):
                h = g[(g.cohort == c) & (g.stim == s) & (g.epoch == e)].sort_values("t")
                a.fill_between(h.t * 1000, h["mean"] - h["sem"], h["mean"] + h["sem"], color=light(cl, 0.7), lw=0,
                               edgecolor="none")
                a.plot(h.t * 1000, h["mean"], color=cl, lw=0.8, label="Pre" if e == "pre" else "Post")
            a.axvspan(5, 35, facecolor="0.93", lw=0, edgecolor="none", zorder=0)
            a.axvline(0, color=W_COL if s == "W" else A_COL, lw=0.6)
            a.set_ylim(lim)
            a.set_xlim(-45, 100)
            a.tick_params(labelsize=FS)
            a.text(0.03, 0.97, f"{LAB[c]} ({df[df.cohort == c].session_id.nunique()})", transform=a.transAxes, ha="left",
                   va="top", color=COLORS[c], fontsize=FS)
            if i == 0:
                _mod_title(a, s, pad=2)
                a.set_xticklabels([])
            else:
                a.set_xlabel("Time from stimulus (ms)")
            if j == 0:
                a.set_ylabel("Evoked rate (spks/s)")
            if i == 0 and j == 1:
                a.legend(fontsize=FS - 0.8, frameon=False, loc="upper right", handlelength=1.0)


def compute_ag_baseline():
    """C: change of the baseline rate (post - pre; Hz; mean over neurons and over whisker and auditory trials) per session"""
    u = _ag_units()
    u = u.assign(b_pre=u[["base_pre_W", "base_pre_A"]].mean(1), b_post=u[["base_post_W", "base_post_A"]].mean(1))
    d = u.groupby(["session_id", "mouse_id", "cohort"])[["b_pre", "b_post"]].mean().reset_index()
    d["value"] = d.b_post - d.b_pre
    META["ag_baseline"] = tests(d, "value")
    return d


def plot_ag_baseline(ax, df):
    a = _mm_insets(ax, [AX_MM], [])[0]
    lo, hi = _ms_range(df, ["cohort"])
    sp = hi - lo
    _two_cohorts(a, df, (min(lo, 0) - 0.9 * sp, max(hi, 0) + 0.9 * sp), "Baseline change (spks/s)\n(post − pre)", dots=True)


def compute_ag_interaction():
    """E: per session, evoked change (post - pre; 5-35 ms minus each trial's baseline; Hz; mean over neurons), whisker and
    auditory; cohort x modality: permutation of cohort labels (as the original) and Mann-Whitney / Welch on whisker - auditory"""
    u = _ag_units()
    d = u.groupby(["session_id", "mouse_id", "cohort"])[["ev_pre_W", "ev_post_W", "ev_pre_A", "ev_post_A"]].mean().reset_index()
    d["W"], d["A"] = d.ev_post_W - d.ev_pre_W, d.ev_post_A - d.ev_pre_A
    d["WA"] = d.W - d.A
    obs, p = perm_interaction(d.WA, d.cohort)
    META["ag_interaction"] = dict(W=tests(d, "W"), A=tests(d, "A"), WA=tests(d, "WA"), perm=dict(diff=obs, p=p),
                                  **{f"vs0_{s}_{c}": float(pmax(stats.wilcoxon(d[d.cohort == c][s]).pvalue,
                                                                stats.ttest_1samp(d[d.cohort == c][s], 0).pvalue))
                                     for s in "WA" for c in COH})
    return d.melt(id_vars=["session_id", "mouse_id", "cohort"], value_vars=["W", "A"], var_name="stim")


def plot_ag_interaction(ax, df):
    """x = cohort; one line per modality (mean +- s.e.m.; open = not different from 0); sessions grey; permutation p"""
    a = _mm_insets(ax, [AX_MM * 0.75], [])[0]                         # tighter (user 2026-11-25, 11-26)
    for s, cl, off in (("W", W_COL, -0.06), ("A", A_COL, 0.06)):
        m = []
        for i, c in enumerate(COH):
            v = df[(df.stim == s) & (df.cohort == c)].value                # session dots removed (user 2026-11-26)
            mm, se = mean_sem(v)
            p0 = pmax(stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue)
            # all circles filled; changes different from 0 annotated with stars and p (user 2026-11-26)
            a.errorbar(i + off, mm, se, color=cl, marker="o", ms=3, mfc=cl, mew=0.8, capsize=0, lw=0.9, zorder=3)
            if p0 < 0.05:
                a.text(i + off + (-0.1 if i == 0 else 0.1), mm, f"{stars(p0)}\n{p_text(p0).replace('p = ', 'p=')}",
                       ha="right" if i == 0 else "left", va="center", fontsize=FS - 1.5, color=cl, linespacing=0.9)
            m.append(mm)
        a.plot([off, 1 + off], m, color=cl, lw=0.9, zorder=2, label="Whisker" if s == "W" else "Auditory")
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    w = df.pivot_table(index=["session_id", "cohort"], columns="stim", values="value").reset_index()
    obs, p, null = perm_interaction(w.W - w.A, w.cohort, return_null=True)
    # inset (as the original panel; user 2026-11-25): permutation null of the interaction vs the observed value
    ins = a.inset_axes([0.31, 0.80, 0.38, 0.18])                       # centred (user 2026-11-25)
    ins.hist(null, bins=40, color="0.75", lw=0)
    ins.axvline(obs, color="#B5524A", lw=0.9)                          # matte red (user 2026-11-25)
    ins.set_yticks([])
    ins.tick_params(labelsize=FS - 1.5, length=1, pad=0.5)
    ins.set_xticks([])
    for sp_ in ("top", "right", "left"):
        ins.spines[sp_].set_visible(False)
    ins.set_xlabel("Null", fontsize=FS - 1.5, labelpad=0.5)
    # y range zoomed on the means +- s.e.m. (user 2026-11-26: "remove single dots, adjust scale"); room above for the inset
    ms = [mean_sem(df[(df.stim == s) & (df.cohort == c)].value) for s in "WA" for c in COH]
    lo, hi = min(min(m_ - s_ for m_, s_ in ms), 0.0), max(max(m_ + s_ for m_, s_ in ms), 0.0)
    sp = hi - lo
    a.set_ylim(lo - 0.45 * sp, hi + 0.6 * sp)
    a.set_title(f"Interaction {p_text(p)}", fontsize=FS, pad=2)
    a.set_xlim(-0.35, 1.35)
    cohort_ticks(a)
    a.set_ylabel("Evoked change (spks/s)\n(post − pre)")
    a.legend(fontsize=FS - 1, frameon=False, loc="lower center", ncol=2, handlelength=0.8, columnspacing=0.6,
             borderaxespad=0.1)


def compute_ag_areas():
    """G: per area group (>= 10 neurons, >= 3 sessions per cohort), per session the mean over the area's neurons of the evoked
    change (whisker W, auditory A) and of the baseline change (B; mean over whisker and auditory trials)"""
    u = _ag_units()
    u = u.assign(b_pre=u[["base_pre_W", "base_pre_A"]].mean(1), b_post=u[["base_post_W", "base_post_A"]].mean(1))
    d = u.groupby(["session_id", "mouse_id", "cohort", "area_group"]).agg(n=("cluster_id", "size"), ev_pre_W=("ev_pre_W", "mean"),
                                                              ev_post_W=("ev_post_W", "mean"), ev_pre_A=("ev_pre_A", "mean"),
                                                              ev_post_A=("ev_post_A", "mean"), b_pre=("b_pre", "mean"),
                                                              b_post=("b_post", "mean")).reset_index()
    d = d[d.n >= 10]
    d["W"], d["A"], d["B"] = d.ev_post_W - d.ev_pre_W, d.ev_post_A - d.ev_pre_A, d.b_post - d.b_pre
    # area kept with >= 3 mice per cohort, each with >= 10 neurons in that area (user 2026-11-26; was >= 3 sessions)
    keep = d.groupby(["area_group", "cohort"]).mouse_id.nunique().unstack().fillna(0)
    keep = keep[(keep["R+"] >= 3) & (keep["R-"] >= 3)].index
    d = d[d.area_group.isin(keep)]
    META["ag_areas"] = dict(n_areas=len(keep), n_sessions=int(d.session_id.nunique()), n_mice=int(d.mouse_id.nunique()),
                            sessions_per_mouse=float(d.groupby("mouse_id").session_id.nunique().max()))
    return d[["session_id", "mouse_id", "cohort", "area_group", "n", "W", "A", "B"]]


def plot_ag_areas(ax, df):
    """Whisker | auditory: per area group (rows, ephys_utilities custom order), evoked change post - pre (mean +- s.e.m. over
    sessions), R+ and R- offset; filled = R+ vs R- differ for that modality in that area (larger P of Mann-Whitney / Welch,
    uncorrected); x axis zoomed to the bulk of the means, out-of-range means drawn as triangles at the border."""
    col, order, _ = area_palette()
    areas = [a_ for a_ in order if a_ in set(df.area_group)] + sorted(set(df.area_group) - set(order))
    cols_ = [c_ for c_ in "WAB" if c_ in df]                          # B = baseline change (user 2026-11-25)
    insets = _mm_insets(ax, [AX_MM] * len(cols_), [4.0] * (len(cols_) - 1))
    y = {a_: i for i, a_ in enumerate(areas)}
    for j, (a, s) in enumerate(zip(insets, cols_)):
        m = df.groupby(["cohort", "area_group"])[s].agg(["mean", "sem"]).reset_index()
        lo, hi = np.nanpercentile(m["mean"], [5, 95])
        sp = hi - lo
        lim = (min(lo, 0) - 0.35 * sp, max(hi, 0) + 0.35 * sp)
        for a_ in areas:
            g = df[df.area_group == a_]
            t = two_group_tests(g[g.cohort == "R+"][s], g[g.cohort == "R-"][s])
            sig = pmax(t["mannwhitney_p"], t["welch_p"]) < 0.05
            for c, off in (("R+", -0.17), ("R-", 0.17)):
                r = m[(m.cohort == c) & (m.area_group == a_)]
                if r.empty:
                    continue
                mu, se = float(r["mean"].iloc[0]), float(r["sem"].iloc[0])
                yy = y[a_] + off
                if lim[0] <= mu <= lim[1]:
                    a.errorbar(mu, yy, xerr=se, fmt="o", ms=2.2, color=COLORS[c], mfc=COLORS[c] if sig else "white",
                               mew=0.6, lw=0.5, capsize=0, zorder=3)
                else:                                                   # out of frame: triangle at the border
                    a.plot(lim[1] if mu > lim[1] else lim[0], yy, ">" if mu > lim[1] else "<", ms=2.4, color=COLORS[c],
                           mfc=COLORS[c] if sig else "white", mew=0.6, clip_on=False, zorder=3)
        a.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
        a.set_xlim(lim)
        a.set_ylim(len(areas) - 0.5, -0.5)
        a.set_yticks(range(len(areas)))
        if j == 0:
            a.set_yticklabels(areas, fontsize=FS - 0.5)
            for tl, a_ in zip(a.get_yticklabels(), areas):
                tl.set_color(col.get(a_, "0.2"))
        else:
            a.set_yticklabels([])
        a.tick_params(axis="y", length=0)
        if s == "B":
            a.set_title("Baseline", fontsize=FS + 0.5, pad=2)
            a.set_xlabel("Baseline rate\nchange (spks/s, post − pre)")
        else:
            _mod_title(a, s, pad=2)
            a.set_xlabel(f"{'Whisker' if s == 'W' else 'Auditory'} response\nchange (spks/s, post − pre)")
    insets[-1].text(1.04, 1.0, "R+", color=COLORS["R+"], transform=insets[-1].transAxes, ha="left", va="top", fontsize=FS)
    insets[-1].text(1.04, 0.93, "R−", color=COLORS["R-"], transform=insets[-1].transAxes, ha="left", va="top", fontsize=FS)


def compute_ag_whisking():
    """E: whisker speed (dlc_helpers, deg/s; each trial minus its pre-stimulus mean), passive whisker trials, pre / post;
    quantified 5-35 ms (the evoked-response window; user 2026-11-25)"""
    t = pd.read_parquet(TAB / "205_passive_state_traces.parquet")
    t = t[(t.signal == "whisker_speed") & (t.trial_type == "whisker_trial")]
    s = pd.read_parquet(TAB / "205_passive_state_dlc.parquet")[["session_id", "mouse_id"]].drop_duplicates()
    t = t.merge(s, on="session_id")
    t = t[t.mouse_id.isin(learners()) & (t.time_s >= -0.05) & (t.time_s <= 0.1)]
    ev = t[(t.time_s >= 0.005) & (t.time_s <= 0.035)].groupby(["session_id", "cohort", "epoch"]).value.mean().unstack()
    dd = (ev["post"] - ev["pre"]).rename("d").reset_index()
    META["ag_whisking"] = dict(delta=tests(dd, "d"), **{c: int((dd.cohort == c).sum()) for c in COH})
    return t[["session_id", "cohort", "epoch", "time_s", "value"]]


def plot_ag_whisking(ax, df):
    """R+ | R-: whisker speed change from baseline, pre grey, post cohort colour (mean +- s.e.m. over sessions)"""
    insets = _mm_insets(ax, [AX_MM, AX_MM], [8.5])
    g = df.groupby(["cohort", "epoch", "time_s"]).value.agg(["mean", "sem"]).reset_index()
    lim = _padded(float((g["mean"] - g["sem"]).min()), float((g["mean"] + g["sem"]).max()), 0.08)
    for j, (a, c) in enumerate(zip(insets, COH)):
        for e, cl in (("pre", "0.55"), ("post", COLORS[c])):
            h = g[(g.cohort == c) & (g.epoch == e)].sort_values("time_s")
            a.fill_between(h.time_s * 1000, h["mean"] - h["sem"], h["mean"] + h["sem"], color=light(cl, 0.7), lw=0,
                           edgecolor="none")
            a.plot(h.time_s * 1000, h["mean"], color=cl, lw=0.8)
        a.axvspan(5, 35, facecolor="0.93", lw=0, edgecolor="none", zorder=0)        # same window as the evoked response
        a.axvline(0, color=W_COL, lw=0.6)
        a.set_ylim(lim)
        a.set_xlim(-50, 100)
        a.set_title(f"{LAB[c]} ({df[df.cohort == c].session_id.nunique()})", color=COLORS[c], fontsize=FS + 0.5, pad=2)
        a.set_xlabel("Time from stimulus (ms)")
        a.set_ylabel("Δ whisker speed (°/s)")


AG_T1 = np.round(np.arange(-0.030, 0.0601, 0.001), 3)          # 1-ms bin starts stored by 179 (psth1_*)
AG_SMOOTH_MS = 5                                                 # causal boxcar: rate at t = mean of the bins [t - 4 ms, t]
AG_BLOCK = 20                                                    # neurons per displayed row
Z_LIM = 2.0                                                      # raster colour range (z)


def _roc_sensory(u):
    """per unit: sensory = whisker_active or auditory_active ROC significant; firing_rate (spikes/s) from the same table.
    ROC: combined_results_ks4/<mouse>/whisker_0/roc_analysis/<mouse>_roc_stimulus.csv (merged roc_core, as 182)"""
    root = axel_bisi_root() / "combined_results_ks4"
    out = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant",
                                    "firing_rate"])
        r = r[r.analysis_type.isin(["whisker_active", "auditory_active"])]
        g = r.groupby(["session_id", "electrode_group", "cluster_id"]).agg(sensory=("significant", "any"),
                                                                          firing_rate=("firing_rate", "first"))
        out.append(g.reset_index().rename(columns={"cluster_id": "cid"}))
    r = pd.concat(out, ignore_index=True)
    r["cid"] = r.cid.astype(int)
    return r


def compute_ag_raster():
    """A (user 2026-11-25): every neuron of both cohorts and all areas in one raster; four conditions (whisker pre, post,
    auditory pre, post); baseline-subtracted rate in 1-ms bins (179 psth1_*), causal 5-ms boxcar (stride 1 ms), -20..60 ms;
    z-scored per neuron across the four conditions (user 2026-11-26, was peak / soft-peak normalisation): s.d. over all
    bins of the four smoothed time courses (-20..60 ms), zero = the neuron's pre-stimulus mean (-20..0 ms, four conditions),
    so that white stays the baseline (a plain mean over the window would make the baseline negative);
    neurons ordered by the pre-task whisker response 5-35 ms (the evoked-response window); averaged in blocks of 20 neurons
    for display. User 2026-11-25 (visualisation only): stable units that are sensory-responsive by ROC (whisker_active or
    auditory_active significant: stimulus vs baseline, 5-35 ms, active trials, so independent of the passive epochs) with a
    baseline rate > 0.5 spikes/s in both passive epochs (present before and after the task)"""
    u = _ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)                 # 179 stores imec * 1e6 + id (as 182)
    n_all = len(u)
    u = u.merge(_roc_sensory(u), on=["session_id", "electrode_group", "cid"], how="inner")
    # user 2026-11-25 ("odd interruptions ... are you sure these are stable units only?"): 179 stable units can still be
    # silent in one passive epoch (8,025 of 81,878 have 0 Hz baseline in passive pre); require > 0.5 spikes/s baseline in
    # BOTH passive epochs (mean over whisker and auditory trials) instead of the session-wide ROC firing rate
    bpre = u[["base_pre_W", "base_pre_A"]].mean(1)
    bpost = u[["base_post_W", "base_post_A"]].mean(1)
    u = u[u.sensory & (bpre > 0.5) & (bpost > 0.5)]
    keys = ["psth1_pre_W", "psth1_post_W", "psth1_pre_A", "psth1_post_A"]
    M = np.stack([np.stack([np.asarray(v, float) for v in u[k].to_numpy()]) for k in keys], 1)        # n x 4 x T
    ker = np.ones(AG_SMOOTH_MS) / AG_SMOOTH_MS
    S = np.apply_along_axis(lambda x: np.convolve(x, ker, mode="full")[:len(x)], 2, M)               # causal
    t_end = AG_T1 + 0.001                                                                             # bin end = time
    keep = (t_end >= -0.020 - 1e-9) & (t_end <= 0.060 + 1e-9)
    S, tt = S[:, :, keep], t_end[keep]
    mu = np.nanmean(S[:, :, tt <= 0], axis=(1, 2))                                     # pre-stimulus mean, 4 conditions
    sd = np.nanstd(S, axis=(1, 2))                                                     # s.d. over all bins, 4 conditions
    S = (S - mu[:, None, None]) / np.where(sd > 0, sd, np.nan)[:, None, None]             # z across the 4 conditions
    resp = np.nanmean(M[:, 0, (AG_T1 >= 0.005) & (AG_T1 < 0.035)], 1)                                # raw 5-35 ms (Hz)
    o = np.argsort(-np.nan_to_num(resp, nan=-np.inf))
    S = S[o]
    nb = int(np.ceil(len(S) / AG_BLOCK))
    B = np.stack([np.nanmean(S[i * AG_BLOCK:(i + 1) * AG_BLOCK], 0) for i in range(nb)])             # blocks x 4 x T
    d = pd.DataFrame({f"{k.replace('psth1_', '')}_{int(round(t * 1000))}": B[:, j, q]
                      for j, k in enumerate(keys) for q, t in enumerate(tt)})
    d.insert(0, "n", [min(AG_BLOCK, len(S) - i * AG_BLOCK) for i in range(nb)])   # cohort strip removed (user 2026-11-25)
    META["ag_raster"] = dict(n=int(len(u)), n_all=int(n_all), **{c: int((u.cohort == c).sum()) for c in COH},
                             n_sessions=int(u.session_id.nunique()))
    return d


def plot_ag_raster(ax, df):
    """one raster: columns whisker pre, post, auditory pre, post; rows = neurons (blocks of 20) by pre-task whisker response"""
    keys = ["pre_W", "post_W", "pre_A", "post_A"]
    heads = ["Pre", "Post", "Pre", "Post"]
    tt = np.array(sorted({int(c.rsplit("_", 1)[1]) for c in df.columns if c.startswith("pre_W_")}))
    ax.axis("off")
    fig = ax.figure
    bb = ax.get_position()
    W_mm, H_mm = bb.width * fig.get_figwidth() * 25.4, bb.height * fig.get_figheight() * 25.4
    gap, mgap = 0.6, 1.6                                              # denser (user 2026-11-26)
    cw = (W_mm - 3 * gap - (mgap - gap)) / 4
    n = int(df.n.sum()) if "n" in df else len(df) * AG_BLOCK
    edges = np.r_[0, np.cumsum(df.n.to_numpy())] if "n" in df else np.arange(len(df) + 1) * AG_BLOCK
    x, im = 0.0, None
    for j, k in enumerate(keys):
        a = ax.inset_axes([x / W_mm, 0, cw / W_mm, 1])
        Z = df[[f"{k}_{t}" for t in tt]].to_numpy()
        im = a.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-Z_LIM, vmax=Z_LIM, interpolation="nearest",
                      extent=[tt[0] - 0.5, tt[-1] + 0.5, n, 0])
        a.axvline(0, color="0.3", lw=0.3, ls=(0, (2, 2)))
        for sp_ in a.spines.values():
            sp_.set_visible(False)
        a.set_xticks([0, 50])
        a.tick_params(labelsize=FS - 0.5, length=1.5, pad=1)
        a.set_title(heads[j], fontsize=FS, pad=1.5)
        if j == 0:
            # n in the label instead of tick labels: panel closer to the left edge (user 2026-11-26)
            a.set_ylabel(f"{n:,} neurons (by pre-task whisker response)", fontsize=FS, labelpad=1)
            a.set_yticks([])
            a.set_xlabel("Time from stimulus (ms)", fontsize=FS, labelpad=1, loc="left")
        else:
            a.set_yticks([])
        x += cw + (mgap if j == 1 else gap)
    for j, s in enumerate("WA"):
        xc = (j * (2 * cw + gap + mgap) + cw + gap / 2) / W_mm
        ax.text(xc, 1.0 + 4.0 / H_mm, MOD[s][0], color=W_COL if s == "W" else A_COL, fontsize=FS + 0.5, ha="center",
                va="bottom", transform=ax.transAxes)
    cax = ax.inset_axes([1 + 0.8 / W_mm, 0.0, 0.9 / W_mm, 0.30])        # vertical, right of the raster, bottom
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[-Z_LIM, 0, Z_LIM])
    cb.ax.tick_params(labelsize=FS - 1, length=1, pad=0.5)
    cb.outline.set_linewidth(0.3)
    # units on every colour bar (user 2026-11-26): z-score of the rate, per neuron, across the four conditions
    cb.set_label("Rate (z, across\n4 conditions)", fontsize=FS - 0.5, labelpad=0.5, linespacing=1.0)


def compute_ag_whisking_post():
    """E, one-row layout (user 2026-11-25: first section on one row): whisker speed, passive post, R+ and R- overlaid"""
    d = compute_ag_whisking()
    META["ag_whisking_post"] = META["ag_whisking"]
    return d[d.epoch == "post"]


def plot_ag_whisking_post(ax, df):
    a = _mm_insets(ax, [AX_MM], [])[0]
    g = df.groupby(["cohort", "time_s"]).value.agg(["mean", "sem"]).reset_index()
    for c in COH:
        h = g[g.cohort == c].sort_values("time_s")
        a.fill_between(h.time_s * 1000, h["mean"] - h["sem"], h["mean"] + h["sem"], color=light(COLORS[c], 0.7), lw=0,
                       edgecolor="none")
        a.plot(h.time_s * 1000, h["mean"], color=COLORS[c], lw=0.8,
               label=f"{LAB[c]} ({df[df.cohort == c].session_id.nunique()})")
    a.axvspan(5, 35, facecolor="0.93", lw=0, edgecolor="none", zorder=0)
    a.axvline(0, color=W_COL, lw=0.6)
    a.set_ylim(_padded(float((g["mean"] - g["sem"]).min()), float((g["mean"] + g["sem"]).max()), 0.08))
    a.set_xlim(-50, 100)
    a.set_title("Passive post", fontsize=FS + 0.5, pad=2)
    a.set_xlabel("Time from stimulus (ms)")
    a.set_ylabel("Δ whisker speed (°/s)")
    a.legend(fontsize=FS - 0.5, frameon=False, loc="upper left", handlelength=1.0)


# ------------------------------------------------------------------ first-section layout v3 (user 2026-11-25): B single column
# aligned to A, C baseline as narrow bars, D pupil (passive post), E evoked response decomposed (epoch x cohort x stimulus),
# F interaction (with the permutation-null inset)

def compute_ag_psth_col():
    d = compute_ag_psth()
    META["ag_psth_col"] = META["ag_psth"]
    return d


def plot_ag_psth_col(ax, df):
    """one column, top to bottom: whisker R+, whisker R-, auditory R+, auditory R-; pre grey, post cohort colour"""
    ax.axis("off")
    g = df.groupby(["cohort", "stim", "epoch", "t"]).value.agg(["mean", "sem"]).reset_index()
    rows = [("W", "R+"), ("W", "R-"), ("A", "R+"), ("A", "R-")]
    gap = 0.045
    h = (1 - 3 * gap - 0.03) / 4                                        # extra 0.03 between modalities
    y = 1.0
    for i, (s, c) in enumerate(rows):
        y -= h
        a = ax.inset_axes([0, y, 1, h])
        y -= gap + (0.03 if i == 1 else 0)
        sub = g[g.stim == s]
        lim = _padded(float((sub["mean"] - sub["sem"]).min()), float((sub["mean"] + sub["sem"]).max()), 0.08)
        for e, cl in (("pre", "0.55"), ("post", COLORS[c])):
            hh = sub[(sub.cohort == c) & (sub.epoch == e)].sort_values("t")
            a.fill_between(hh.t * 1000, hh["mean"] - hh["sem"], hh["mean"] + hh["sem"], color=light(cl, 0.7), lw=0,
                           edgecolor="none")
            a.plot(hh.t * 1000, hh["mean"], color=cl, lw=0.7, label="Pre" if e == "pre" else "Post")
        a.axvspan(5, 35, facecolor="0.93", lw=0, edgecolor="none", zorder=0)
        a.axvline(0, color=W_COL if s == "W" else A_COL, lw=0.5)
        a.set_ylim(lim)
        a.set_xlim(-20, 60)
        a.set_xticks([0, 50])
        a.tick_params(labelsize=FS - 0.5, length=1.5, pad=1)
        n_ses = df[(df.cohort == c) & (df.stim == s)].session_id.nunique()            # n shown (user 2026-11-26)
        a.text(0.98, 0.97, f"{MOD[s][0]} {LAB[c]}\nn = {n_ses}", transform=a.transAxes, ha="right", va="top",
               fontsize=FS - 0.5, color=COLORS[c], linespacing=1.0)
        if i < 3:
            a.set_xticklabels([])
        else:
            a.set_xlabel("Time (ms)", fontsize=FS, labelpad=1)
        if i == 0:
            # pre / post named at the curves' right ends (a legend overlapped the label or the traces in this narrow panel)
            yr = lim[1] - lim[0]
            for e, cl, off, va in (("pre", "0.45", 0.06, "bottom"), ("post", COLORS[c], -0.06, "top")):
                hh = sub[(sub.cohort == c) & (sub.epoch == e) & (sub.t * 1000 <= 60)].sort_values("t")
                a.text(59, float(hh["mean"].iloc[-1]) + off * yr, "Pre" if e == "pre" else "Post", ha="right", va=va,
                       fontsize=FS - 1, color=cl)
            # analysis window (user 2026-11-26: "annotate window of analysis, 5-35 ms")
            a.plot([5, 35], [1.06, 1.06], color="0.3", lw=0.6, transform=a.get_xaxis_transform(), clip_on=False)
            a.text(20, 1.10, "5–35 ms", ha="center", va="bottom", fontsize=FS - 0.5, color="0.3",
                   transform=a.get_xaxis_transform())
    w_mm = ax.get_position().width * ax.figure.get_figwidth() * 25.4
    ax.text(-5.5 / w_mm, 0.5, "Evoked rate (spks/s)", rotation=90, ha="center", va="center", fontsize=FS,
            transform=ax.transAxes)


def compute_ag_baseline_bars():
    d = compute_ag_baseline()
    META["ag_baseline_bars"] = META["ag_baseline"]
    return d


def plot_ag_baseline_bars(ax, df):
    """narrow: bars (mean +- s.e.m.), R+ / R- (no session dots, user 2026-11-25)"""
    a = _mm_insets(ax, [AX_MM * 0.55], [])[0]
    for q, c in enumerate(COH):
        v = df[df.cohort == c].value
        mm, se = mean_sem(v)
        a.bar(q, mm, width=0.62, color=light(COLORS[c], 0.55), edgecolor=COLORS[c], lw=0.6, zorder=1)
        a.errorbar(q, mm, se, color="0.25", lw=0.8, capsize=0, zorder=3)        # dark grey s.e.m. (user 2026-11-25)
    t = tests(df, "value")
    p = pmax(t["mannwhitney_p"], t["welch_p"])
    # top of the y scale from the spread of sessions (90th percentile; user 2026-11-25); bottom just below the lowest
    # mean - s.e.m. or 0 (user 2026-11-26: "show less of negative range")
    hi = float(np.nanpercentile(df.value, 90))
    lo = min(0.0, min(m - s_ for m, s_ in (mean_sem(df[df.cohort == c].value) for c in COH)))
    a.plot([0, 1], [hi + 0.08 * (hi - lo)] * 2, color="k", lw=0.5)
    a.text(0.5, hi + 0.1 * (hi - lo), p_text(p), ha="center", va="bottom", fontsize=FS - 0.5)
    a.axhline(0, color="0.6", lw=0.4)
    a.set_xlim(-0.6, 1.6)
    a.set_ylim(lo - 0.08 * (hi - lo), hi + 0.3 * (hi - lo))
    cohort_ticks(a)
    a.set_ylabel("Baseline change (spks/s)\n(post − pre)")


def compute_ag_evoked():
    """E: per session, evoked response 5-35 ms (Hz; each trial minus its own baseline; mean over neurons), passive pre and
    post, whisker and auditory; pre vs post per cohort (Wilcoxon + paired t, larger P)"""
    u = _ag_units()
    d = u.groupby(["session_id", "mouse_id", "cohort"])[["ev_pre_W", "ev_post_W", "ev_pre_A", "ev_post_A"]].mean().reset_index()
    meta = {}
    for s in "WA":
        for c in COH:
            x = d[d.cohort == c]
            meta[f"{s}_{c}"] = dict(pre=float(x[f"ev_pre_{s}"].mean()), post=float(x[f"ev_post_{s}"].mean()),
                                    p=float(pmax(stats.wilcoxon(x[f"ev_post_{s}"], x[f"ev_pre_{s}"]).pvalue,
                                                 stats.ttest_rel(x[f"ev_post_{s}"], x[f"ev_pre_{s}"]).pvalue)))
    META["ag_evoked"] = meta
    return d


def _axis_break(a, where="top"):
    """two short diagonal strokes across the left spine near its top / bottom end: the y axis is cut (values beyond it)"""
    y = 0.97 if where == "top" else 0.03
    for dy in (-0.018, 0.018):
        a.plot([-0.035, 0.035], [y + dy - 0.012, y + dy + 0.012], transform=a.transAxes, color="k", lw=0.5,
               clip_on=False, zorder=5)
    a.plot([0, 0], [y - 0.018, y + 0.018], transform=a.transAxes, color="white", lw=1.2, clip_on=False, zorder=4)


def plot_ag_evoked(ax, df):
    """whisker | auditory; x = R+ pre, R+ post, R- pre, R- post; mean +- s.e.m. over sessions (pre open, post filled);
    pre vs post p per cohort; y axis zoomed on the means, own range per modality"""
    insets = _mm_insets(ax, [AX_MM * 0.8, AX_MM * 0.8], [6.0])                 # condensed (user 2026-11-26)
    for j, (a, s) in enumerate(zip(insets, "WA")):
        # y range per modality (user 2026-11-26: "adjust y-scale to emphasize group effects"; "add lines back but cut axes
        # for outliers"): up to the 90th percentile of the sessions (and every mean +- s.e.m.); sessions above are cut at the
        # axis, marked by a break on the y axis; nothing hidden at the bottom (user 2026-11-26: "do not hide outliers at
        # bottom, only at top")
        ms = [mean_sem(df[df.cohort == c][f"ev_{e}_{s}"]) for c in COH for e in ("pre", "post")]
        v = df[[f"ev_pre_{s}", f"ev_post_{s}"]].to_numpy()
        lo = min(float(np.nanmin(v)), min(m - se for m, se in ms))
        hi = max(float(np.nanpercentile(v, 90)), max(m + se for m, se in ms))
        sp = hi - lo
        lo, hi = lo - 0.05 * sp, hi + 0.05 * sp
        top = hi + 0.2 * (hi - lo)
        for q, c in enumerate(COH):
            x = df[df.cohort == c]
            xp, xq = 2 * q, 2 * q + 1
            for _, r in x.iterrows():
                a.plot([xp, xq], [r[f"ev_pre_{s}"], r[f"ev_post_{s}"]], color="0.86", lw=0.3, zorder=1)
            for xx, e in ((xp, "pre"), (xq, "post")):
                mm, se = mean_sem(x[f"ev_{e}_{s}"])
                a.errorbar(xx, mm, se, color=COLORS[c], marker="o", ms=2.8, mfc="white" if e == "pre" else COLORS[c],
                           mew=0.8, lw=0.9, capsize=0, zorder=3)
            a.plot([xp, xq], [x[f"ev_pre_{s}"].mean(), x[f"ev_post_{s}"].mean()], color=COLORS[c], lw=0.9, zorder=2)
            p = pmax(stats.wilcoxon(x[f"ev_post_{s}"], x[f"ev_pre_{s}"]).pvalue,
                     stats.ttest_rel(x[f"ev_post_{s}"], x[f"ev_pre_{s}"]).pvalue)
            a.text(2 * q + 0.5, hi + 0.04 * (hi - lo), p_text(p).replace("p = ", "p="), ha="center", va="bottom",
                   fontsize=FS - 1, color=COLORS[c])
        a.set_xticks([0, 1, 2, 3])
        a.set_xticklabels(["Pre", "Post", "Pre", "Post"], fontsize=FS - 0.5)
        for q, c in enumerate(COH):
            a.text(2 * q + 0.5, -0.17, LAB[c], color=COLORS[c], ha="center", va="top", fontsize=FS,
                   transform=a.get_xaxis_transform())
        a.set_xlim(-0.6, 3.6)
        a.set_ylim(lo, top)
        if np.nanmax(v) > top:
            _axis_break(a, "top")
        if np.nanmin(v) < lo:
            _axis_break(a, "bottom")
        _mod_title(a, s, pad=2)
        a.set_ylabel("Evoked response (spks/s)\n5–35 ms" if j == 0 else "")



def compute_ag_areas_wa():
    """G (user 2026-11-25: "below that let's find a way to show area by area"): per area group, the interaction measure
    whisker - auditory evoked change (post - pre, Hz) per session, R+ and R-"""
    d = compute_ag_areas()
    d["WA"] = d.W - d.A
    META["ag_areas_wa"] = META["ag_areas"]
    return d[["session_id", "mouse_id", "cohort", "area_group", "n", "WA"]]


def plot_ag_areas_wa(ax, df):
    """one column: area groups (rows), whisker - auditory change, R+ and R- (mean +- s.e.m. over sessions); filled = R+ vs R-
    differ (larger P of Mann-Whitney / Welch, uncorrected); triangles at the border = beyond the axis range"""
    col, order, _ = area_palette()
    areas = [a_ for a_ in order if a_ in set(df.area_group)] + sorted(set(df.area_group) - set(order))
    a = _mm_insets(ax, [AX_MM * 0.75], [])[0]
    m = df.groupby(["cohort", "area_group"]).WA.agg(["mean", "sem"]).reset_index()
    lo, hi = np.nanpercentile(m["mean"], [5, 95])
    sp = hi - lo
    lim = (min(lo, 0) - 0.3 * sp, max(hi, 0) + 0.3 * sp)
    for k, a_ in enumerate(areas):
        g = df[df.area_group == a_]
        t = two_group_tests(g[g.cohort == "R+"].WA, g[g.cohort == "R-"].WA)
        sig = pmax(t["mannwhitney_p"], t["welch_p"]) < 0.05
        for c, off in (("R+", -0.17), ("R-", 0.17)):
            r = m[(m.cohort == c) & (m.area_group == a_)]
            if r.empty:
                continue
            mu, se = float(r["mean"].iloc[0]), float(r["sem"].iloc[0])
            if lim[0] <= mu <= lim[1]:
                a.errorbar(mu, k + off, xerr=se, fmt="o", ms=1.8, color=COLORS[c], mfc=COLORS[c] if sig else "white",
                           mew=0.5, lw=0.4, capsize=0, zorder=3)
            else:
                a.plot(lim[1] if mu > lim[1] else lim[0], k + off, ">" if mu > lim[1] else "<", ms=2.0, color=COLORS[c],
                       mfc=COLORS[c] if sig else "white", mew=0.5, clip_on=False, zorder=3)
    a.axvline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_xlim(lim)
    a.set_ylim(len(areas) - 0.5, -0.5)
    a.set_yticks(range(len(areas)))
    a.set_yticklabels(areas, fontsize=FS - 1)
    for tl, a_ in zip(a.get_yticklabels(), areas):
        tl.set_color(col.get(a_, "0.2"))
    a.tick_params(axis="y", length=0, pad=1)
    a.set_xlabel("Whisker − auditory\nchange (spks/s)")


# ------------------------------------------------------------------ D and G, 2026-11-26 (user: "For D: show change in pupil
# area for both cohorts, and show evoked movement in that window (time-course) using PC1 from DLC keypoints. replace G by
# correlation, at area group level, of effects with hit/miss choice coding fraction")



def _pupil_session_stats():
    """per session: mean and s.d. of the resting pupil over all passive trials (205 --pupil-trials, pupil_base, mm^2)"""
    t = pd.read_parquet(TAB / "205_passive_pupil_trials.parquet")
    t = t[t.mouse_id.isin(learners())].dropna(subset=["pupil_base"])
    return t, t.groupby("session_id").pupil_base.agg(mu="mean", sd="std")


def _pupil_z_change():
    """resting pupil z-scored within session (= per mouse; one whisker-day-0 session per mouse; user 2026-11-26): every
    passive trial's pupil_base (mm^2, -1..0 s; blink / squint trials NaN), z over all passive trials of the session, then mean
    z post - mean z pre (s.d. units). Removes eye-size / camera-scale differences between mice."""
    t, ms = _pupil_session_stats()
    t = t.join(ms, on="session_id")
    t["z"] = (t.pupil_base - t.mu) / t.sd
    w = t.groupby(["session_id", "mouse_id", "cohort", "epoch"]).z.mean().unstack().dropna().reset_index()
    return w.assign(value=w["post"] - w["pre"])[["session_id", "mouse_id", "cohort", "value"]]


MOVE_SIGS = (("whisker_speed", "Whisker speed", "°/s"),)          # jaw removed (user 2026-11-26)
PUPIL_T = (-1.0, 3.0)


def compute_ag_pupil():
    """D (user 2026-11-26: "make D time-resolved for pupil area"): pupil area time course (205 per-session mean traces, whisker
    and auditory trials averaged, blink / squint trials removed, not baseline-subtracted), z-scored with the session's resting
    pupil mean and s.d. (all passive trials, = per mouse), change post - pre; t < 0 = resting state, t > 0 = evoked. Test:
    R+ vs R- on the resting change (_pupil_z_change)."""
    t = pd.read_parquet(TAB / "205_passive_state_traces.parquet")
    t = t[t.signal == "pupil_area"]
    s = pd.read_parquet(TAB / "205_passive_state_dlc.parquet")[["session_id", "mouse_id"]].drop_duplicates()
    t = t.merge(s, on="session_id")
    t = t[t.mouse_id.isin(learners())]
    _, ms = _pupil_session_stats()
    t = t.join(ms, on="session_id", how="inner")
    t["z"] = (t.value - t.mu) / t.sd
    w = t.groupby(["session_id", "mouse_id", "cohort", "epoch", "time_s"]).z.mean().unstack("epoch").dropna().reset_index()
    w["value"] = w["post"] - w["pre"]
    w = w[(w.time_s >= PUPIL_T[0] - 1e-9) & (w.time_s <= PUPIL_T[1] + 1e-9)]
    META["ag_pupil"] = dict(rest=tests(_pupil_z_change(), "value"),
                            evoked=tests(w[(w.time_s >= 0.5) & (w.time_s <= 2.0)].groupby(["session_id", "cohort"])
                                         .value.mean().reset_index(), "value"))
    return w[["session_id", "mouse_id", "cohort", "time_s", "value"]]


def plot_ag_pupil(ax, df):
    """change post - pre of the pupil (z) over time, R+ and R- (mean +- s.e.m. over sessions)"""
    a = _mm_insets(ax, [_ax_width_mm(ax)], [])[0]
    g = df.groupby(["cohort", "time_s"]).value.agg(["mean", "sem"]).reset_index()
    for c in COH:
        h = g[g.cohort == c].sort_values("time_s")
        a.fill_between(h.time_s, h["mean"] - h["sem"], h["mean"] + h["sem"], color=light(COLORS[c], 0.7), lw=0,
                       edgecolor="none")
        a.plot(h.time_s, h["mean"], color=COLORS[c], lw=0.8, label=LAB[c])
    a.axvline(0, color="0.4", lw=0.4, ls=(0, (2, 2)))
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    a.set_xlim(*PUPIL_T)
    a.set_xticks([-1, 0, 1, 2, 3])
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + 0.3 * (hi - lo))
    m = META.get("ag_pupil")
    a.set_xlabel("Time from stimulus (s)")
    a.set_ylabel("Δ pupil area\n(z, post − pre)")
    a.legend(fontsize=FS - 0.5, frameon=False, loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8)


def compute_ag_state():
    """E (user 2026-11-26: "E quantifications as barplots instead"; jaw distance instead of jaw speed): change post - pre of
    the evoked whisker speed (top view, deg/s) and jaw opening (jaw distance, side view, mm) in the 5-35 ms window, whisker and
    auditory passive trials pooled; 207 (DLC via ephys_utilities dlc_helpers; each trial minus its -0.2..0 s mean)"""
    t = pd.read_parquet(TAB / "207_passive_whisker_jaw_traces.parquet")
    t = t[t.mouse_id.isin(learners()) & (t.stim == "all") & t.signal.isin([s for s, _, _ in MOVE_SIGS])]
    t = t[(t.time_s >= 0.005 - 1e-9) & (t.time_s <= 0.035 + 1e-9)]
    w = t.groupby(["session_id", "mouse_id", "cohort", "signal", "epoch"]).raw_bc.mean().unstack("epoch").dropna()
    w = (w["post"] - w["pre"]).rename("value").reset_index()
    META["ag_state"] = {s: tests(w[w.signal == s], "value") for s, _, _ in MOVE_SIGS}
    return w


def plot_ag_state(ax, df):
    """two narrow bar plots: whisker speed | jaw opening change (5-35 ms, post - pre), R+ / R- (mean +- s.e.m. over
    sessions; Mann-Whitney / Welch, larger P)"""
    insets = _mm_insets(ax, [9.0] * len(MOVE_SIGS), [14.0] * (len(MOVE_SIGS) - 1))
    for a, (s, lab, unit) in zip(insets, MOVE_SIGS):
        g = df[df.signal == s]
        ms = []
        for q, c in enumerate(COH):
            mm, se = mean_sem(g[g.cohort == c].value)
            ms.append((mm, se))
            a.bar(q, mm, width=0.62, color=light(COLORS[c], 0.55), edgecolor=COLORS[c], lw=0.6, zorder=1)
            a.errorbar(q, mm, se, color="0.25", lw=0.8, capsize=0, zorder=3)
        t = tests(g, "value")
        lo = min(0.0, min(m_ - s_ for m_, s_ in ms))
        hi = max(0.0, max(m_ + s_ for m_, s_ in ms))
        sp = hi - lo
        a.plot([0, 1], [hi + 0.12 * sp] * 2, color="k", lw=0.5)
        p = pmax(t["mannwhitney_p"], t["welch_p"])
        a.text(0.5, hi + 0.15 * sp, f"{stars(p)}\n{p_text(p)}" if p < 0.05 else p_text(p), ha="center", va="bottom",
               fontsize=FS - 1, linespacing=0.9)
        a.axhline(0, color="0.6", lw=0.4)
        a.set_xlim(-0.6, 1.6)
        a.set_ylim(lo - 0.1 * sp, hi + 0.75 * sp)
        cohort_ticks(a)
        a.set_title(lab, fontsize=FS, pad=2)
        a.set_ylabel(f"Δ {lab.lower()}, 5–35 ms\n({unit}, post − pre)", labelpad=1)


def _roc_flag(u, analysis_type):
    """per unit: ROC significant and selectivity for one analysis type (roc_stimulus table, merged roc_core)"""
    root = axel_bisi_root() / "combined_results_ks4"
    out = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant",
                                        "selectivity"])
            r = r[r.analysis_type == analysis_type]
            out.append(r.drop(columns="analysis_type").rename(columns={"cluster_id": "cid", "significant": analysis_type}))
    r = pd.concat(out, ignore_index=True)
    r["cid"] = r.cid.astype(int)
    return r.drop_duplicates(["session_id", "electrode_group", "cid"])


def compute_ag_choice_corr():
    """H. Per area group (selection as compute_ag_areas: >= 3 mice per cohort, each >= 10 neurons in the area): the effect
    delta = mean over R+ sessions of (whisker - auditory evoked change) - same for R- (spikes/s), against the mean absolute
    whisker choice selectivity of the area's neurons (ROC whisker_choice, whisker hit vs miss, active trials, 5-35 ms;
    |selectivity| = |2 AUC - 1|, all neurons, threshold-free; user 2026-11-26, was the fraction significant; both cohorts
    pooled; neurons of the sessions that pass the selection)"""
    d = compute_ag_areas()
    d["WA"] = d.W - d.A
    m = d.groupby(["area_group", "cohort"]).WA.mean().unstack()
    out = pd.DataFrame({"delta": m["R+"] - m["R-"]})
    u = _ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    u = u.merge(_roc_flag(u, "whisker_choice"), on=["session_id", "electrode_group", "cid"], how="inner")
    u = u.merge(d[["session_id", "area_group"]].drop_duplicates(), on=["session_id", "area_group"])
    u["whisker_choice"] = u.whisker_choice.astype(float)
    u["abs_sel"] = u.selectivity.abs()
    out = out.join(u.groupby("area_group").agg(abs_sel=("abs_sel", "mean"), frac_choice=("whisker_choice", "mean"),
                                                n_units=("cid", "size")))
    out.index.name = "area_group"
    out = out.reset_index()
    r, p = stats.pearsonr(out.abs_sel, out.delta)
    rho, prho = stats.spearmanr(out.abs_sel, out.delta)
    META["ag_choice_corr"] = dict(n=len(out), r=float(r), p=float(p), rho=float(rho), p_rho=float(prho),
                                  n_delta_pos=int((out.delta > 0).sum()), n_units=int(out.n_units.sum()))
    return out


def plot_ag_choice_corr(ax, df):
    """scatter (area-group colours, no outline) + least-squares line (solid if p < 0.05); area names"""
    col, _, _ = area_palette()
    a = ax
    x, y = df.abs_sel.to_numpy(float), df.delta.to_numpy(float)
    n = len(x)
    sl, ic, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    a.plot(xs, ic + sl * xs, color="0.25", lw=0.7, ls="-" if p < 0.05 else (0, (3, 2)), zorder=1)
    yr, xr = y.max() - y.min(), x.max() - x.min()
    ly = y.copy()
    order = np.argsort(y)
    for _ in range(50):                                               # labels pushed apart vertically (greedy repel)
        moved = False
        for i_ in range(len(order)):
            for j_ in range(i_ + 1, len(order)):
                p_, q_ = order[i_], order[j_]
                if abs(x[p_] - x[q_]) < 0.35 * xr and abs(ly[p_] - ly[q_]) < 0.095 * yr:
                    sh = (0.095 * yr - abs(ly[p_] - ly[q_])) / 2 + 1e-9
                    lo_, hi_ = (p_, q_) if ly[p_] <= ly[q_] else (q_, p_)
                    ly[lo_] -= sh
                    ly[hi_] += sh
                    moved = True
        if not moved:
            break
    for k_, (_, rr) in enumerate(df.iterrows()):
        c = col.get(rr.area_group, "0.4")
        a.scatter(rr.abs_sel, rr.delta, s=7, color=c, lw=0, edgecolors="none", zorder=3)   # no outline (user 2026-11-26)
        a.text(rr.abs_sel + 0.025 * xr, ly[k_], area_label(rr.area_group), fontsize=FS - 2.2, color=c, ha="left",
               va="center", zorder=4)
        if abs(ly[k_] - rr.delta) > 0.03 * yr:
            a.plot([rr.abs_sel, rr.abs_sel + 0.022 * xr], [rr.delta, ly[k_]], color=c, lw=0.25, zorder=2)
    a.axhline(0, color="0.6", lw=0.4, ls=(0, (1, 1.5)), zorder=0)
    a.text(0.03, 0.97, f"n = {n}\nr = {r:.2f}\n{p_text(p)}", transform=a.transAxes, ha="left", va="top", fontsize=FS - 0.5)
    a.set_xlim(x.min() - 0.08 * xr, x.max() + 0.35 * xr)
    a.set_xlabel("Mean |whisker choice selectivity|\n(hit vs miss ROC)")
    a.set_ylabel("R+ − R−: whisker − auditory\nevoked change (spks/s)")


# ------------------------------------------------------------------ second / learning sections, 2026-11-26 (user: "L: show
# directly angle relative to shift null"; "O left only next to M"; "single panel for the two examples, side by side")

def compute_angle_excess():
    d = compute_wa_angle_prepost()
    META["angle_excess"] = META.pop("wa_angle_prepost")
    return d


def plot_angle_excess(ax, df):
    """angle to the hit - miss axis minus its linear-shift null (data - shifted), pre and post, whisker | auditory; R+ / R-
    mean +- s.e.m. over sessions; post vs pre per cohort (Wilcoxon / paired t), R+ vs R- on the change (MW / Welch)"""
    w = df.pivot_table(index=["session_id", "cohort", "stim", "epoch"], columns="kind", values="value").reset_index()
    w["value"] = w["data"] - w["shifted"]
    ms = w.groupby(["stim", "cohort", "epoch"]).value.agg(["mean", "sem"])
    lo, hi = float((ms["mean"] - ms["sem"]).min()), float((ms["mean"] + ms["sem"]).max())
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    sp = hi - lo
    for a, s_ in zip(_mm_insets(ax, [AX_MM, AX_MM], [11.0]), "WA"):
        g = w[w.stim == s_]
        for c, off in (("R+", -0.06), ("R-", 0.06)):
            m = [mean_sem(g[(g.cohort == c) & (g.epoch == e)].value) for e in ("pre", "post")]
            a.errorbar(np.array([0, 1]) + off, [v[0] for v in m], [v[1] for v in m], color=COLORS[c], marker="o", ms=3,
                       lw=1.0, capsize=0)
        a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
        a.set_ylim(lo - 0.1 * sp, hi + 0.1 * sp)
        yl = a.get_ylim()
        span = yl[1] - yl[0]
        for k, c in enumerate(COH):
            q = g[g.cohort == c].pivot(index="session_id", columns="epoch", values="value").dropna()
            p = pmax(stats.wilcoxon(q.pre, q.post).pvalue, stats.ttest_rel(q.pre, q.post).pvalue)
            bracket(a, 0, 1, yl[1] + span * (0.04 + 0.24 * k), p, COLORS[c])
        q = g.pivot_table(index=["session_id", "cohort"], columns="epoch", values="value").reset_index()
        q["d"] = q["post"] - q["pre"]
        t = two_group_tests(q[q.cohort == "R+"].d, q[q.cohort == "R-"].d)
        post = [mean_sem(g[(g.cohort == c) & (g.epoch == "post")].value)[0] for c in COH]
        vbracket(a, 1.2, min(post), max(post), pmax(t["mannwhitney_p"], t["welch_p"]))
        a.set_ylim(yl[0], yl[1] + span * 0.55)
        a.set_xlim(-0.3, 1.75)
        a.set_xticks([0, 1], ["Pre", "Post"])
        _mod_title(a, s_, pad=2)
        a.set_ylabel("Angle to hit − miss axis (°)\n(data − shifted)" if s_ == "W" else "")


def compute_area_share():
    d = compute_area_contrib()
    META["area_share"] = META.pop("area_contrib")
    return d


def plot_area_share(ax, df):
    """left part of the former area-contribution panel only (user 2026-11-26): each area's R- minus R+ contribution to the
    whole-brain whisker change (data - shifted; sessions without the area count 0), stars = R+ vs R- (MW / Welch)"""
    col, _, _ = area_palette()
    ses = df.drop_duplicates("session_id")[["session_id", "cohort", "total"]]
    full = df.pivot_table(index="session_id", columns="area", values="contrib", fill_value=0.0).reindex(ses.session_id).fillna(0)
    full["cohort"] = ses.set_index("session_id").cohort
    diff = {a_: full[full.cohort == "R-"][a_].mean() - full[full.cohort == "R+"][a_].mean() for a_ in df.area.unique()}
    order = sorted(diff, key=diff.get)
    y = np.arange(len(order))[::-1]
    a = ax
    a.barh(y, [diff[k] for k in order], color=[col[k] for k in order], height=0.7, lw=0)
    for yy, k in zip(y, order):
        t = two_group_tests(full[full.cohort == "R+"][k], full[full.cohort == "R-"][k])
        p = pmax(t["mannwhitney_p"], t["welch_p"])
        if p < 0.05:
            a.annotate(stars(p), (diff[k], yy), xytext=(-2, -1.5), textcoords="offset points", ha="right", va="center",
                       fontsize=FS)
    tt = two_group_tests(ses[ses.cohort == "R+"].total, ses[ses.cohort == "R-"].total)
    dtot = ses[ses.cohort == "R-"].total.mean() - ses[ses.cohort == "R+"].total.mean()
    a.text(0.02, 0.02, f"Whole brain, R− − R+:\n{dtot:.3f}, {p_text(pmax(tt['mannwhitney_p'], tt['welch_p']))}",
           transform=a.transAxes, ha="left", va="bottom", fontsize=FS - 0.5)
    a.axvline(0, color="0.5", lw=0.4)
    a.set_ylim(-0.7, len(order) - 0.3)
    a.set_yticks(y, [area_label(k) for k in order])
    a.tick_params(axis="y", length=0, labelsize=FS - 0.5)
    for t_, k in zip(a.get_yticklabels(), order):
        t_.set_color(col[k])
    a.set_xlabel("R− minus R+ share of the whole-brain\nwhisker change (data − shifted)")


def compute_lc_examples():
    d = pd.concat([compute_lc_example_rp(), compute_lc_example_rm()], ignore_index=True)
    META["lc_examples"] = {"R+": META.pop("lc_example_rp"), "R-": META.pop("lc_example_rm")}
    return d


def plot_lc_examples(ax, df):
    """the R+ and R- example sessions side by side in one panel (user 2026-11-26)"""
    # figure-level axes (not insets): the learning-curve twin axis of an inset is drawn above it and hid the trials
    W = _ax_width_mm(ax)
    gap = 12.0
    pos = ax.get_position()
    ax.axis("off")
    w = (W - gap) / 2 / W * pos.width
    for k, c in enumerate(COH):
        a = ax.figure.add_axes([pos.x0 + k * (w + gap / W * pos.width), pos.y0, w, pos.height])
        _plot_lc_example(a, df[df.cohort == c])
