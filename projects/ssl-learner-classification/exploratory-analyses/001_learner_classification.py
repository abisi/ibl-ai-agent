"""001 -- Who is a learner on whisker day 0? A label built from many criteria, without the reference sheet (user
2026-11-13: "using my learning curves (exact), learning trial definitions and the behaviour clustering, delineate learners
vs non-learners agnostic to my own learning_category; compare with my categories and my clustering").

Data: whisker day-0 session of every mouse in the shared behaviour table (projects/ssl-paper-figures shared_behaviour:
NWB_ks4, perf != 6, not passive, day-0 auditory warm-up cut; early-lick trials dropped here); cohort from the reference
sheet; exact learning curves (grid HMM, sigma 1, 80 % band; Figure 1 compute.session_curves), whisker curve on whisker
trials and chance curve = no-stim curve interpolated in time onto the whisker trials.

Per mouse, features (curves on the whisker-trial axis): P(lick) per trial type, d' (whisker vs no stimulus, log-linear
correction), gap = whisker - chance curve (mean, max, last 20 trials), fraction and longest run of whisker trials with the
whisker curve credibly above chance (p_low > p_chance) or not above it (p_low <= p_chance, after the first hit), second-half
and last-30-trial whisker and false-alarm rates.

Nine binary criteria per cohort (learner = True). R+ learns to lick the rewarded whisker stimulus above the no-stim rate;
R- learns to stop licking the non-rewarded whisker stimulus (whisker back to the no-stim rate):
  exact_rule    learning_curve_exact rule (behaviour_analysis learning_utils on the exact curves): category learner/expert
  high_state    a high-performance state (assign_expertise_blocks, 5 consecutive trials; R+ p_low > p_chance, R- p_low < p_chance)
  half2         second half of the whisker trials: Fisher exact one-sided whisker > no-stim P(lick), R+ p < 0.05, R- p >= 0.05
  late30        same on the last 30 whisker trials and the no-stim trials of that period
  sustained     R+ longest run credibly above chance >= 10 trials; R- longest run not above chance after the first hit
                >= 10 trials (never-hit R- mice: the whole session)
  L6            joint whisker + no-stim Bayesian change point (ssl-learning-trial-identification 028/017): R+ learner or
                high_from_start; R- learner or never_licked
  cascade       lenient cascade learning trial of the same project (L6 lenient -> L5w -> L7), or the same learned-from-start
                categories
  gmm           2-component Gaussian mixture on the standardised features (per cohort); learner component = higher (R+)
                or lower (R-) mean last-30 whisker - no-stim gap
  dtw           TimeSeriesKMeans k = 2, DTW, whisker + chance curves resampled to 100 points (behaviour_analysis
                behavior_clustering.py settings, here on the exact curves); learner cluster as for gmm
Consensus = fraction of criteria met. Label = learner if consensus >= 0.5. Constraint (user): R+ mice MH020, MH013, AB142,
MH068, AB151, AB117, AB107 stay non-learners; the flexible R+ bad mice (AB141, MH007, AB104, MH026, MH009) take the consensus.
Comparisons: sheet learning_category; the earlier DTW clusters (behaviour_analysis behavior_clustering.py, 3 clusters, old
PyMC curves; combined_results/behavior_clustering/3_both_dtw_interpolated_input); k = 3 DTW on the exact curves.
Validation: whisker P(lick) on day +1 (not used by any criterion), AUC of every labelling.
Performance: learners vs non-learners per cohort (Mann-Whitney and Welch, mice).

Outputs: combined_results_ks4/ssl-learner-classification/{tables,cache}/001_*.
Run on haas (repo root): PYTHONPATH=~/pylibs/tslearn nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python \
    projects/ssl-learner-classification/exploratory-analyses/001_learner_classification.py
"""

import importlib.util
import json
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PF = REPO / "projects" / "ssl-paper-figures"
sys.path.insert(0, str(PF))
sys.path.insert(0, str(REPO / "scripts"))
import shared_behaviour  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig1 = _load("fig1_compute", PF / "figures" / "Figure1_behaviour" / "compute.py")
ROOT = axel_bisi_root()
OUT = ROOT / "combined_results_ks4" / "ssl-learner-classification"
LTI = REPO / "projects" / "ssl-learning-trial-identification" / "artifacts"
OLD_DTW = ROOT / "combined_results" / "behavior_clustering" / "3_both_dtw_interpolated_input" / "mouse_cluster_assignments.csv"
COHORTS = ["R+", "R-"]
FIXED_BAD = ["MH020", "MH013", "AB142", "MH068", "AB151", "AB117", "AB107"]  # user: stay non-learners
FLEX = ["AB141", "MH007", "AB104", "MH026", "MH009"]  # user: R+ 'bad' mice that may move
CRITERIA = ["exact_rule", "high_state", "half2", "late30", "sustained", "L6", "cascade", "gmm", "dtw"]
CRITERIA_LABELS = {"exact_rule": "Exact-curve rule", "high_state": "High-performance state", "half2": "Second half W vs FA",
                   "late30": "Last 30 W trials vs FA", "sustained": "Sustained run (≥ 10 trials)",
                   "L6": "Joint change point (L6)", "cascade": "Lenient cascade", "gmm": "Gaussian mixture",
                   "dtw": "DTW clustering (k = 2)"}
GMM_FEATURES = ["hit_w", "fa", "dprime", "gap_mean", "gap_last20", "frac_above", "frac_not_above", "half2_gap", "late30_gap"]
RUN_MIN, LATE_N, N_POINTS, CURVE_BINS = 10, 30, 100, 50


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def dprime(h, n_s, f, n_n):
    """Log-linear corrected d' from hit and false-alarm counts."""
    return stats.norm.ppf((h + 0.5) / (n_s + 1)) - stats.norm.ppf((f + 0.5) / (n_n + 1))


def fisher_greater(w, fa):
    """One-sided Fisher exact: whisker P(lick) > no-stim P(lick)."""
    w, fa = np.asarray(w), np.asarray(fa)
    if len(w) == 0 or len(fa) == 0:
        return np.nan
    return stats.fisher_exact([[w.sum(), len(w) - w.sum()], [fa.sum(), len(fa) - fa.sum()]], alternative="greater")[1]


def _rt(d):
    v = d[(d.lick_flag == 1) & d.computed_reaction_time.between(0, 1)].computed_reaction_time
    return v.median() if len(v) else np.nan


def session_features(st_raw):
    curves, lt, cat = fig1.session_curves(st_raw)
    st = st_raw[st_raw.early_lick == 0].sort_values("start_time")
    w = st[st.trial_type == "whisker_trial"].reset_index(drop=True)
    ns, au = st[st.trial_type == "no_stim_trial"], st[st.trial_type == "auditory_trial"]
    n = len(w)
    above = (curves.p_low > curves.p_chance).to_numpy()
    gap = (curves.p_mean - curves.p_chance).to_numpy()
    hits = np.flatnonzero(w.lick_flag.to_numpy() == 1)
    first_hit = int(hits[0]) if len(hits) else n
    not_above = ~above[first_hit:]
    t_half, t_late = w.start_time.iloc[n // 2], w.start_time.iloc[max(n - LATE_N, 0)]
    w2, ns2 = w.iloc[n // 2:].lick_flag, ns[ns.start_time >= t_half].lick_flag
    wl, nsl = w.iloc[max(n - LATE_N, 0):].lick_flag, ns[ns.start_time >= t_late].lick_flag
    r = dict(mouse_id=st.mouse_id.iloc[0], session_id=st.session_id.iloc[0], cohort=st.cohort.iloc[0],
             learning_category=st.learning_category.iloc[0], n_whisker=n, n_no_stim=len(ns), n_auditory=len(au),
             hit_w=w.lick_flag.mean(), fa=ns.lick_flag.mean(), hit_a=au.lick_flag.mean(),
             dprime=dprime(w.lick_flag.sum(), n, ns.lick_flag.sum(), len(ns)),
             rt_w=_rt(w), rt_a=_rt(au), first_hit=first_hit if len(hits) else np.nan,
             gap_mean=gap.mean(), gap_max=gap.max(), gap_last20=gap[-20:].mean(), p_whisker_max=curves.p_mean.max(),
             frac_above=above.mean(), run_above=longest_run(above), frac_not_above=(~above).mean(),
             run_not_above=longest_run(not_above) if len(hits) else n,
             half2_gap=w2.mean() - ns2.mean(), half2_p=fisher_greater(w2, ns2),
             late30_gap=wl.mean() - nsl.mean(), late30_p=fisher_greater(wl, nsl),
             exact_cat=cat, exact_lt=lt, high_any=bool((curves.state == "high").any()),
             frac_high=(curves.state == "high").mean())
    # curves for the figure: session progress (0-1) and whisker-trial axis
    prog = np.linspace(0, 1, CURVE_BINS)
    x = np.linspace(0, 1, n)
    c = dict(mouse_id=r["mouse_id"], whisker_prog=np.interp(prog, x, curves.p_mean), chance_prog=np.interp(prog, x, curves.p_chance),
             auditory_prog=np.interp(prog, x, curves.p_auditory) if np.isfinite(curves.p_auditory).all() else np.full(CURVE_BINS, np.nan),
             whisker=curves.p_mean.to_numpy(), chance=curves.p_chance.to_numpy(),
             whisker_100=np.interp(np.linspace(0, 1, N_POINTS), x, curves.p_mean),
             chance_100=np.interp(np.linspace(0, 1, N_POINTS), x, curves.p_chance))
    return r, c


def other_days(t):
    """Per mouse, P(lick) per day and trial type (session mean, then mouse mean), d' on day +1, pretraining."""
    t = t[t.early_lick == 0]
    g = t.groupby(["mouse_id", "day", "session_id", "trial_type"]).lick_flag
    p = g.mean().groupby(["mouse_id", "day", "trial_type"]).mean().unstack(["day", "trial_type"])
    out = pd.DataFrame({k: p.get((d, tt)) for k, d, tt in [
        ("hit_w_day1", 1, "whisker_trial"), ("fa_day1", 1, "no_stim_trial"), ("hit_w_day2", 2, "whisker_trial"),
        ("hit_a_daym1", -1, "auditory_trial"), ("fa_daym1", -1, "no_stim_trial")]})
    d1 = t[t.day == 1].groupby(["mouse_id", "trial_type"]).lick_flag.agg(["sum", "size"]).unstack("trial_type")
    out["dprime_day1"] = pd.Series(dprime(d1[("sum", "whisker_trial")], d1[("size", "whisker_trial")],
                                          d1[("sum", "no_stim_trial")], d1[("size", "no_stim_trial")]), index=d1.index)
    pre, _ = fig1.compute_pretraining({"table": t})
    out = out.reset_index().merge(pre[["mouse_id", "n_pretraining_trials", "n_auditory_days"]].rename(
        columns={"n_auditory_days": "n_pretraining_days"}), on="mouse_id", how="left")
    return out


def lt_methods():
    """Learning-trial methods of ssl-learning-trial-identification (028: all 100 day-0 mice)."""
    a = pd.read_csv(LTI / "028_learning_trials_all_methods_all_mice.csv")
    return a[["session_id", "L6 category", "L6 joint CP", "lenient cascade", "L7 half-way", "L5 whisker CP"]].rename(
        columns={"L6 category": "L6_category", "L6 joint CP": "L6_lt", "lenient cascade": "cascade_lt",
                 "L7 half-way": "L7_lt", "L5 whisker CP": "L5_lt"})


def gmm_labels(df):
    X = StandardScaler().fit_transform(df[GMM_FEATURES].fillna(df[GMM_FEATURES].median()))
    gm = GaussianMixture(2, covariance_type="diag", n_init=50, random_state=0).fit(X)
    lab, post = gm.predict(X), gm.predict_proba(X)
    return lab, post


def dtw_labels(curves, k, seed=42):
    from tslearn.clustering import TimeSeriesKMeans
    X = np.stack([np.column_stack([c["whisker_100"], c["chance_100"]]) for c in curves])
    m = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_init=5, max_iter=50, random_state=seed)
    return m.fit_predict(X), m.cluster_centers_


def learner_side(df, lab, cohort):
    """Which cluster is the learner one: higher (R+) / lower (R-) mean last-30 whisker - no-stim gap."""
    means = df.groupby(lab).late30_gap.mean()
    return lab == (means.idxmax() if cohort == "R+" else means.idxmin())


def criteria(df, curves_by_mouse):
    rows = []
    for coh in COHORTS:
        d = df[df.cohort == coh].copy()
        rp = coh == "R+"
        d["c_exact_rule"] = d.exact_cat.isin(["learner", "expert"])
        d["c_high_state"] = d.high_any
        d["c_half2"] = (d.half2_p < 0.05) if rp else (d.half2_p >= 0.05)
        d["c_late30"] = (d.late30_p < 0.05) if rp else (d.late30_p >= 0.05)
        d["c_sustained"] = (d.run_above >= RUN_MIN) if rp else (d.run_not_above >= RUN_MIN)
        start = ["high_from_start"] if rp else ["never_licked"]
        has_l6 = d.L6_category.notna()
        d["c_L6"] = np.where(has_l6, d.L6_category.isin(["learner"] + start), np.nan)
        d["c_cascade"] = np.where(has_l6 | d.cascade_lt.notna(), d.cascade_lt.notna() | d.L6_category.isin(start), np.nan)
        lab, post = gmm_labels(d)
        d["c_gmm"] = learner_side(d, lab, coh)
        d["gmm_p_learner"] = post[:, lab[d.c_gmm.to_numpy()][0]] if d.c_gmm.any() else 0.0
        cv = [curves_by_mouse[m] for m in d.mouse_id]
        lab2, _ = dtw_labels(cv, 2)
        d["dtw2_cluster"] = lab2
        d["c_dtw"] = learner_side(d, lab2, coh)
        lab3, _ = dtw_labels(cv, 3)
        d["dtw3_cluster"] = lab3
        rows.append(d)
    d = pd.concat(rows)
    cc = [f"c_{c}" for c in CRITERIA]
    d[cc] = d[cc].astype(float)
    d["n_criteria"] = d[cc].notna().sum(axis=1)
    d["consensus"] = d[cc].mean(axis=1)
    d["learner_consensus"] = d.consensus >= 0.5
    d["fixed_bad"] = d.mouse_id.isin(FIXED_BAD)
    d["flexible"] = d.mouse_id.isin(FLEX)
    d["learner"] = d.learner_consensus & ~d.fixed_bad
    d["sheet_learner"] = d.learning_category.map({"good": True, "moderate": True, "bad": False})
    return d


def agreement(d, old):
    """Cohen's kappa of every binary labelling with the final label and with the sheet; ARI/NMI of clusterings."""
    rows = []
    for coh in COHORTS + ["both"]:
        g = d if coh == "both" else d[d.cohort == coh]
        labs = {**{c: g[f"c_{c}"] for c in CRITERIA}, "consensus (before constraint)": g.learner_consensus.astype(float),
                "final label": g.learner.astype(float), "sheet": g.sheet_learner.astype(float)}
        for name, v in labs.items():
            for ref in ("final label", "sheet"):
                ok = v.notna() & labs[ref].notna()
                a, b = v[ok].astype(bool), labs[ref][ok].astype(bool)
                rows.append(dict(cohort=coh, labelling=name, reference=ref, n=int(ok.sum()),
                                 agreement=float((a == b).mean()),
                                 kappa=float(cohen_kappa_score(a, b)) if a.nunique() > 1 or b.nunique() > 1 else np.nan,
                                 n_learner=int(a.sum())))
    ag = pd.DataFrame(rows)
    crows = []
    for coh in COHORTS:
        g = d[d.cohort == coh]  # old clusters already merged
        for name, col in (("DTW k=3, old curves (behavior_clustering.py)", "old_cluster"),
                          ("DTW k=3, exact curves", "dtw3_cluster"), ("DTW k=2, exact curves", "dtw2_cluster")):
            for ref, rc in (("sheet category", "learning_category"), ("final label", "learner")):
                ok = g[col].notna() & g[rc].notna()
                crows.append(dict(cohort=coh, clustering=name, reference=ref, n=int(ok.sum()),
                                  ari=adjusted_rand_score(g[rc][ok].astype(str), g[col][ok].astype(int)),
                                  nmi=normalized_mutual_info_score(g[rc][ok].astype(str), g[col][ok].astype(int))))
    return ag, pd.DataFrame(crows)


VALIDATION = {"hit_w_day1": "W P(lick) day +1", "dprime_day1": "d' day +1"}


def validation(d):
    """Day +1 (not used by any criterion): AUC of every labelling for whisker P(lick) and d' (R+ learners higher, R-
    learners lower), Mann-Whitney p; Spearman rho of the consensus score."""
    rows = []
    for coh in COHORTS:
        sign = 1 if coh == "R+" else -1
        for out, out_lab in VALIDATION.items():
            g = d[(d.cohort == coh) & d[out].notna()]
            labs = {**{c: g[f"c_{c}"] for c in CRITERIA}, "final label": g.learner.astype(float),
                    "consensus (before constraint)": g.learner_consensus.astype(float), "sheet": g.sheet_learner.astype(float)}
            for name, v in labs.items():
                ok = v.notna()
                y, s = v[ok].astype(bool), sign * g[out][ok]
                r = dict(cohort=coh, outcome=out, outcome_label=out_lab, labelling=name, n=int(ok.sum()),
                         n_learner=int(y.sum()), auc=np.nan, p=np.nan)
                if y.nunique() == 2:
                    r.update(auc=roc_auc_score(y, s), p=stats.mannwhitneyu(s[y], s[~y], alternative="two-sided").pvalue)
                rows.append(r)
            rho, p = stats.spearmanr(g.consensus, g[out])
            rows.append(dict(cohort=coh, outcome=out, outcome_label=out_lab, labelling="consensus score (Spearman ρ)",
                             n=len(g), auc=rho, p=p))
    return pd.DataFrame(rows)


def stability(d):
    """Leave one criterion out: fraction of mice whose label (consensus >= 0.5, constraint applied) is unchanged; and
    threshold 0.4 / 0.6 instead of 0.5."""
    cc = [f"c_{c}" for c in CRITERIA]
    rows = []
    for coh in COHORTS:
        g = d[d.cohort == coh]
        for drop in [None] + CRITERIA:
            use = [c for c in cc if c != f"c_{drop}"]
            for th in (0.4, 0.5, 0.6):
                lab = (g[use].mean(axis=1) >= th) & ~g.fixed_bad
                rows.append(dict(cohort=coh, dropped=drop or "none", threshold=th, n_learner=int(lab.sum()),
                                 same_as_final=float((lab == g.learner).mean()),
                                 changed=",".join(g.mouse_id[lab != g.learner])))
    return pd.DataFrame(rows)


RULE_FEATURES = {"dprime": "d' day 0", "late30_gap": "W − FA, last 30 W trials", "half2_gap": "W − FA, 2nd half",
                 "gap_mean": "W − chance curve, mean", "frac_above": "Fraction W trials above chance", "hit_w": "W P(lick) day 0",
                 "fa": "FA P(lick) day 0"}


def simple_rule(d):
    """What a learner is, in one line: best single-feature threshold for the final label per cohort (balanced
    accuracy), with leave-one-out accuracy of re-choosing the threshold."""
    from sklearn.metrics import balanced_accuracy_score
    rows = []
    for coh in COHORTS:
        g = d[d.cohort == coh]
        y = g.learner.to_numpy()
        for f, lab in RULE_FEATURES.items():
            x = g[f].to_numpy()
            ok = np.isfinite(x)
            best = (-1, None, None)
            for th in np.unique(x[ok]):
                for side in (">=", "<="):
                    pred = x >= th if side == ">=" else x <= th
                    ba = balanced_accuracy_score(y[ok], pred[ok]) if len(np.unique(y[ok])) > 1 else np.nan
                    if ba > best[0]:
                        best = (ba, th, side)
            loo = []
            for i in np.flatnonzero(ok):
                m = ok.copy(); m[i] = False
                bb = (-1, None, None)
                for th in np.unique(x[m]):
                    for side in (">=", "<="):
                        pred = x[m] >= th if side == ">=" else x[m] <= th
                        ba = balanced_accuracy_score(y[m], pred)
                        if ba > bb[0]:
                            bb = (ba, th, side)
                loo.append((x[i] >= bb[1] if bb[2] == ">=" else x[i] <= bb[1]) == y[i])
            rows.append(dict(cohort=coh, feature=f, label=lab, threshold=best[1], side=best[2], balanced_acc=best[0],
                             loo_accuracy=float(np.mean(loo)), auc=roc_auc_score(y[ok], x[ok]) if len(np.unique(y[ok])) > 1 else np.nan))
    return pd.DataFrame(rows)


COMPARE = {"hit_w": "W P(lick) day 0", "fa": "FA P(lick) day 0", "hit_a": "A P(lick) day 0", "dprime": "d' day 0",
           "gap_mean": "W − chance curve, mean", "late30_gap": "W − FA, last 30 W trials", "rt_w": "W RT day 0 (s)",
           "rt_a": "A RT day 0 (s)", "n_whisker": "W trials day 0", "first_hit": "W trials before 1st hit",
           "hit_w_day1": "W P(lick) day +1", "fa_day1": "FA P(lick) day +1", "dprime_day1": "d' day +1",
           "hit_w_day2": "W P(lick) day +2", "hit_a_daym1": "A P(lick) day −1", "n_pretraining_trials": "Pretraining trials"}


def comparisons(d, label="learner"):
    rows = []
    for coh in COHORTS:
        g = d[d.cohort == coh]
        for v, lab in COMPARE.items():
            a, b = g[g[label]][v].dropna(), g[~g[label]][v].dropna()
            r = dict(cohort=coh, measure=v, label=lab, labelling=label, n_learner=len(a), n_non=len(b),
                     learner_mean=a.mean(), learner_sem=a.sem(), non_mean=b.mean(), non_sem=b.sem(),
                     learner_median=a.median(), non_median=b.median())
            if len(a) >= 2 and len(b) >= 2:
                r.update(mannwhitney_p=stats.mannwhitneyu(a, b).pvalue, welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue)
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)
    t = shared_behaviour.load()
    d0 = fig1.day0_table(t)
    # one day-0 session per mouse (the longest, if several)
    nw = d0[d0.trial_type == "whisker_trial"].groupby(["mouse_id", "session_id"]).size().reset_index(name="n")
    keep = nw.sort_values("n").groupby("mouse_id").tail(1).session_id
    cache = OUT / "cache" / "001_session_features.pkl"  # exact fits are the slow step; delete to refit
    if cache.exists():
        res = pd.read_pickle(cache)
    else:
        sessions = [st for sid, st in d0[d0.session_id.isin(keep)].groupby("session_id")]
        with Pool(min(30, len(sessions))) as pool:
            res = pool.map(session_features, sessions)
        pd.to_pickle(res, cache)
    df = pd.DataFrame([r for r, _ in res])
    curves_by_mouse = {c["mouse_id"]: c for _, c in res}
    df = df.merge(other_days(t), on="mouse_id", how="left").merge(lt_methods(), on="session_id", how="left")
    d = criteria(df, curves_by_mouse)
    old = pd.read_csv(OLD_DTW)[["mouse_id", "cluster"]].rename(columns={"cluster": "old_cluster"})
    d = d.merge(old, on="mouse_id", how="left")
    d.to_csv(OUT / "tables" / "001_per_mouse.csv", index=False)
    ag, cl = agreement(d, old)
    ag.to_csv(OUT / "tables" / "001_agreement.csv", index=False)
    cl.to_csv(OUT / "tables" / "001_clustering_agreement.csv", index=False)
    val = validation(d)
    val.to_csv(OUT / "tables" / "001_validation_day1.csv", index=False)
    stab = stability(d)
    stab.to_csv(OUT / "tables" / "001_stability.csv", index=False)
    rule = simple_rule(d)
    rule.to_csv(OUT / "tables" / "001_simple_rule.csv", index=False)
    comp = pd.concat([comparisons(d, "learner"), comparisons(d.assign(sheet_learner=d.sheet_learner.fillna(False).astype(bool)),
                                                             "sheet_learner")])
    comp.to_csv(OUT / "tables" / "001_comparisons.csv", index=False)
    pd.crosstab([d.cohort, d.learning_category.fillna("missing")], d.learner).to_csv(OUT / "tables" / "001_crosstab_sheet.csv")
    crv = pd.DataFrame([{**{k: c[k] for k in ("mouse_id",)},
                         **{f"{k}_{i}": v for k in ("whisker_prog", "chance_prog", "auditory_prog") for i, v in enumerate(c[k])}}
                        for c in curves_by_mouse.values()])
    crv.to_parquet(OUT / "cache" / "001_curves_progress.parquet")
    pd.to_pickle({m: {"whisker": c["whisker"], "chance": c["chance"]} for m, c in curves_by_mouse.items()},
                 OUT / "cache" / "001_curves_trials.pkl")
    prov = dict(script=str(Path(__file__).relative_to(REPO)), run=datetime.now().isoformat(timespec="seconds"),
                n_mice=int(len(d)), criteria=CRITERIA, run_min=RUN_MIN, late_n=LATE_N, gmm_features=GMM_FEATURES,
                fixed_bad=FIXED_BAD, flexible=FLEX, lt_source=str(LTI / "028_learning_trials_all_methods_all_mice.csv"),
                old_dtw=str(OLD_DTW), curve_params=fig1.CURVE_PARAMS, state_params=fig1.STATE_PARAMS)
    (OUT / "provenance_001.json").write_text(json.dumps(prov, indent=1, default=str))
    # console summary
    print(d.groupby("cohort").agg(mice=("mouse_id", "size"), learners=("learner", "sum"),
                                  sheet_learners=("sheet_learner", "sum"), lt_methods=("L6_category", "count")))
    print(pd.crosstab([d.cohort, d.learning_category.fillna("missing")], d.learner))
    dis = d[(d.learner != d.sheet_learner.fillna(False).astype(bool))]
    print("disagreements with sheet:\n", dis[["mouse_id", "cohort", "learning_category", "consensus", "learner", "fixed_bad",
                                             "flexible"] + [f"c_{c}" for c in CRITERIA]].to_string(index=False))
    print(ag[ag.reference == "final label"].pivot(index="labelling", columns="cohort", values="kappa").round(2))
    print(cl.round(3).to_string(index=False))
    print(val.round(3).to_string(index=False))
    print(stab[stab.threshold == 0.5].round(3).to_string(index=False))
    print(stab[(stab.dropped == "none")].round(3).to_string(index=False))
    print(rule.round(3).to_string(index=False))
    print("fixed bad with consensus >= 0.5:", d[d.fixed_bad & d.learner_consensus].mouse_id.tolist())
    print(d[d.flexible][["mouse_id", "consensus", "learner"]].to_string(index=False))


if __name__ == "__main__":
    main()
