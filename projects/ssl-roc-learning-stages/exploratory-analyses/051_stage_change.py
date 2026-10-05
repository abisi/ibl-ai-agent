"""051 -- Change of whisker / auditory responsiveness (active task) from learning to expert, per area group and cohort
(COSYNE figures; user 2026-10-05).

Units: Kilosort 4 good + mua (045 tables: units.parquet, roc_long.parquet; rate-based ROC, roc_utils_new).
Responsiveness (per modality m in {whisker, auditory}; user 2026-10-05): stimulus vs baseline ROC in the active task
(`whisker_active`, `auditory_active`; 5-35 ms after stimulus onset vs pre-trial baseline, 1000 label permutations).
A unit is responsive if significant (one-sided p on the side of the selectivity < 0.05). |selectivity| = |2 AUC - 1|, all
tested units. (An earlier version used the passive pre AND post blocks with Bonferroni; outputs in passive_stage_change/.)
Cohort: per mouse from the mouse reference sheet (R+ / R- only); stage: learning = day 0, expert = day >= 1.
Populations: all mice; learners (day-0 sessions of mice whose learning_category is not good / moderate are dropped, their
expert sessions kept).
Session x area group: fraction responsive and mean |sel| over the tested units (>= MIN_UNITS); unit of analysis = session.
Inclusion: within each cohort, an area group is kept if it has >= MIN_SESS sessions at both stages (not required across
cohorts).
Change per area group = mean over expert sessions - mean over learning sessions; 95 % CI: bootstrap of sessions within each
stage (N_BOOT); tests: Mann-Whitney U and Welch t (learning vs expert sessions).
ANOVA per cohort x modality x metric: value ~ stage * area (type II OLS on session x area rows of the included area
groups); p from the F test and from N_PERM permutations of the stage label across the cohort's sessions (a session keeps
all its area rows).
Focality (whisker, fraction responsive; user 2026-10-05): concentration index C = sum(x_i^2) / (sum x_i)^2 of the
profile x of mean fractions across the K included area groups (1/K: evenly spread; 1: in a single area group), learning vs
expert; bootstrap CI (sessions within stage) and permutation p of
the change (stage labels across sessions).
Output: combined_results_ks4/ssl-roc-learning-stages/stage_change/<population>/ (session_area.csv, change.csv,
anova.csv, focality.csv, provenance.json, captions.md, figures/*.{png,pdf,svg}).
"""
import argparse
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
BASE = RES / "ssl-roc-learning-stages"
MOUSE_REF = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MODS = ["whisker", "auditory"]
ALPHA = 0.05                         # single active-task test per modality
MIN_UNITS, MIN_SESS = 5, 3
N_BOOT, N_PERM = 2000, 1000
COL = {"whisker": "#f7b519", "auditory": "#2c2cdb", "R+": "#00B400", "R-": "#C800C8"}
METRICS = {"frac": "fraction of responsive units", "abs_sel": "mean |selectivity|"}
COHORT_TXT = {"R+": "R+ (whisker licks rewarded)", "R-": "R− (whisker licks not rewarded)"}
W_IN = 7.4


# ------------------------------------------------------------------------------------------------ data
def mouse_table():
    M = pd.read_excel(MOUSE_REF).rename(columns={"mouse_name": "mouse_id"})
    M = M[M.reward_group.isin(["R+", "R-"])]
    return M.drop_duplicates("mouse_id").set_index("mouse_id")[["reward_group", "learning_category"]]


def unit_metrics():
    U = pd.read_parquet(BASE / "units.parquet")
    U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    M = mouse_table()
    U = U[U.mouse_id.isin(M.index)].copy()
    U["cohort"] = U.mouse_id.map(M.reward_group)                       # per mouse, from the reference sheet
    U["learning_category"] = U.mouse_id.map(M.learning_category)
    L = pd.read_parquet(BASE / "roc_long.parquet", columns=KEYS + ["analysis_type", "sel", "p_value_to_show"])
    for c in KEYS:
        U[c], L[c] = U[c].astype(str), L[c].astype(str)
    out = []
    for m in MODS:
        X = L[L.analysis_type == f"{m}_active"].drop(columns="analysis_type").dropna(subset=["sel", "p_value_to_show"])
        X["resp"] = X.p_value_to_show < ALPHA
        X["abs_sel"] = X.sel.abs()
        X["modality"] = m
        out.append(X)
    X = pd.concat(out).merge(U[KEYS + ["cohort", "stage", "day", "area_group", "learning_category"]], on=KEYS)
    return X


def session_area(X, population):
    if population == "learners":
        bad = (X.stage == "learning") & ~X.learning_category.isin(["good", "moderate"])
        X = X[~bad]
    S = X.groupby(["modality", "cohort", "stage", "mouse_id", "session_id", "day", "area_group"]).agg(
        n_units=("resp", "size"), n_resp=("resp", "sum"), frac=("resp", "mean"), abs_sel=("abs_sel", "mean")).reset_index()
    return S[S.n_units >= MIN_UNITS].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ statistics
def boot_diff(a, b, rng):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    d = rng.choice(b, (N_BOOT, len(b))).mean(1) - rng.choice(a, (N_BOOT, len(a))).mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def change_table(S, rng):
    rows = []
    for (mod, coh, area), g in S.groupby(["modality", "cohort", "area_group"]):
        for met in METRICS:
            a = g[g.stage == "learning"][met].to_numpy()
            b = g[g.stage == "expert"][met].to_numpy()
            inc = len(a) >= MIN_SESS and len(b) >= MIN_SESS
            r = dict(modality=mod, cohort=coh, area_group=area, metric=met, included=inc, n_learning=len(a), n_expert=len(b),
                     n_mice_learning=g[g.stage == "learning"].mouse_id.nunique(), n_mice_expert=g[g.stage == "expert"].mouse_id.nunique(),
                     mean_learning=a.mean() if len(a) else np.nan, mean_expert=b.mean() if len(b) else np.nan,
                     sem_learning=stats.sem(a) if len(a) > 1 else np.nan, sem_expert=stats.sem(b) if len(b) > 1 else np.nan)
            r["change"] = r["mean_expert"] - r["mean_learning"]
            if inc:
                r["ci_lo"], r["ci_hi"] = boot_diff(a, b, rng)
                r["p_mwu"] = stats.mannwhitneyu(a, b).pvalue
                r["p_welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
            rows.append(r)
    return pd.DataFrame(rows)


def anova(S, C, rng):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    for (mod, coh), g in S.groupby(["modality", "cohort"]):
        for met in METRICS:
            inc = C[(C.modality == mod) & (C.cohort == coh) & (C.metric == met) & C.included].area_group
            d = g[g.area_group.isin(inc)].copy()
            if d.area_group.nunique() < 2:
                continue
            d["y"] = d[met]

            def fit(df):
                t = anova_lm(smf.ols("y ~ C(stage) * C(area_group)", df).fit(), typ=2)
                return t.loc["C(stage)", "F"], t.loc["C(stage):C(area_group)", "F"], t
            Fs, Fi, t = fit(d)
            sess = d.drop_duplicates("session_id")[["session_id", "stage"]]
            ns, ni = [], []
            for _ in range(N_PERM):
                lab = dict(zip(sess.session_id, rng.permutation(sess.stage.to_numpy())))
                dp = d.assign(stage=d.session_id.map(lab))
                fs, fi, _ = fit(dp)
                ns.append(fs); ni.append(fi)
            ns, ni = np.array(ns), np.array(ni)
            rows.append(dict(modality=mod, cohort=coh, metric=met, n_areas=d.area_group.nunique(), n_sessions=d.session_id.nunique(),
                             n_rows=len(d), F_stage=Fs, p_stage=t.loc["C(stage)", "PR(>F)"], p_stage_perm=(1 + (ns >= Fs).sum()) / (1 + N_PERM),
                             F_stage_x_area=Fi, p_stage_x_area=t.loc["C(stage):C(area_group)", "PR(>F)"],
                             p_stage_x_area_perm=(1 + (ni >= Fi).sum()) / (1 + N_PERM),
                             F_area=t.loc["C(area_group)", "F"], p_area=t.loc["C(area_group)", "PR(>F)"]))
    return pd.DataFrame(rows)


def concentration(x):
    """sum of squared fractions over the squared sum of fractions: 1/K (evenly spread) .. 1 (one area group)"""
    x = np.clip(np.asarray(x, float), 0, None)
    return np.nan if len(x) == 0 or x.sum() == 0 else np.sum(x ** 2) / x.sum() ** 2


def focality(S, C, rng):
    rows, prof = [], {}
    for coh in ["R+", "R-"]:
        inc = list(C[(C.modality == "whisker") & (C.cohort == coh) & (C.metric == "frac") & C.included].area_group)
        g = S[(S.modality == "whisker") & (S.cohort == coh) & S.area_group.isin(inc)]
        if len(inc) < 3:
            continue
        P = g.pivot_table(index="session_id", columns="area_group", values="frac")      # sessions x areas (NaN: not recorded)
        stage = g.drop_duplicates("session_id").set_index("session_id").stage.reindex(P.index)

        def prof_of(st):
            return P[st == "learning"].mean().reindex(inc).to_numpy(), P[st == "expert"].mean().reindex(inc).to_numpy()
        pa, pb = prof_of(stage)
        prof[coh] = (inc, pa, pb)
        for name, fn in (("concentration", concentration),):
            fa, fb = fn(pa), fn(pb)
            bl, be = [], []
            for _ in range(N_BOOT):
                ia = rng.choice(np.where(stage == "learning")[0], (stage == "learning").sum())
                ie = rng.choice(np.where(stage == "expert")[0], (stage == "expert").sum())
                bl.append(fn(P.iloc[ia].mean().reindex(inc).to_numpy())); be.append(fn(P.iloc[ie].mean().reindex(inc).to_numpy()))
            dp = []
            for _ in range(N_PERM):
                st = pd.Series(rng.permutation(stage.to_numpy()), index=stage.index)
                a_, b_ = prof_of(st)
                dp.append(fn(b_) - fn(a_))
            dp = np.array(dp)
            rows.append(dict(cohort=coh, measure=name, n_areas=len(inc), learning=fa, expert=fb, change=fb - fa,
                             learning_lo=np.nanpercentile(bl, 2.5), learning_hi=np.nanpercentile(bl, 97.5),
                             expert_lo=np.nanpercentile(be, 2.5), expert_hi=np.nanpercentile(be, 97.5),
                             p_perm=(1 + np.sum(np.abs(dp) >= abs(fb - fa) - 1e-12)) / (1 + np.isfinite(dp).sum()),
                             n_learning=int((stage == "learning").sum()), n_expert=int((stage == "expert").sum())))
    return pd.DataFrame(rows), prof


# ------------------------------------------------------------------------------------------------ figures
def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 6,
                         "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    return plt


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)


def area_order_and_colours():
    sys.path.insert(0, str(pathlib.Path.home() / "code" / "ephys_utilities"))         # ephys_utilities package (haas)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "ssl-stimulus-arrival-decoding" / "exploratory-analyses"))
    import ephys_utilities.allen_utils.allen_utils as au
    AR = importlib.import_module("_areas")
    return list(au.get_area_group_custom_order()), AR.colors()


def p_txt(p):
    return "n/a" if p is None or not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def fig_change(plt, C, A, met, coh, order, acol, out, population):
    D = C[(C.metric == met) & (C.cohort == coh)]
    areas = [a for a in order if a in set(D[D.included].area_group)]
    if not areas:
        return
    inc = C[(C.metric == met) & C.included]                  # x range shared by both cohorts (same scale in R+ and R- figures)
    hi = np.nanmax(inc.ci_hi) if len(inc) else 0.1
    lo = np.nanmin(inc.ci_lo) if len(inc) else 0
    span_pos, span_neg = max(hi, 0.01) * 1.12, max(-lo, 0.0) * 1.12 + 0.12 * max(hi, 0.01)
    H = 0.85 + 0.17 * len(areas)
    fig = plt.figure(figsize=(W_IN / 2, H))
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.3, right=0.97, top=1 - 0.5 / H, bottom=0.55 / H)
    axA, axW = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    y = np.arange(len(areas))
    for ax, mod in ((axA, "auditory"), (axW, "whisker")):
        q = D[D.modality == mod].set_index("area_group")
        for i, a in enumerate(areas):
            if a not in q.index or not q.loc[a, "included"]:
                ax.text(0, i, "–", ha="center", va="center", color="0.6", fontsize=5)
                continue
            r = q.loc[a]
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=COL[mod], lw=0.9, solid_capstyle="butt", zorder=2)
            sig = r.p_mwu < 0.05
            ax.plot(r.change, i, "o", ms=3.6, mfc=COL[mod] if sig else "white", mec=COL[mod], mew=0.8, zorder=3)
        ax.axvline(0, color="0.3", lw=0.5, zorder=1)
        ax.set_ylim(len(areas) - 0.5, -0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axA.set_xlim(span_pos, -span_neg)                       # mirrored: auditory increase to the left
    axW.set_xlim(-span_neg, span_pos)
    axA.set_yticks(y, [a.replace(" areas", "").replace("Somatosensory-", "SS-") for a in areas])
    for t, a in zip(axA.get_yticklabels(), areas):
        t.set_color(acol.get(a, "k"))
    axW.set_yticks([])
    unit = "fraction" if met == "frac" else "|sel|"
    axA.set_xlabel(f"Δ {unit} (auditory)")
    axW.set_xlabel(f"Δ {unit} (whisker)")
    axA.set_title("← auditory increase", fontsize=6, color=COL["auditory"], loc="right")
    axW.set_title("whisker increase →", fontsize=6, color=COL["whisker"], loc="left")
    for ax, mod in ((axA, "auditory"), (axW, "whisker")):
        a = A[(A.metric == met) & (A.cohort == coh) & (A.modality == mod)]
        if len(a):
            a = a.iloc[0]
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1) / 2, 0.02 / H, f"ANOVA: stage {p_txt(a.p_stage_perm)}\nstage × area {p_txt(a.p_stage_x_area_perm)}",
                     ha="center", va="bottom", fontsize=4.6, color="0.25")
    fig.suptitle(f"Change in {METRICS[met]} from learning to expert, {COHORT_TXT[coh]}",
                 x=0.02, y=1 - 0.05 / H, ha="left", va="top", fontsize=6.6, weight="bold", color=COL[coh])
    save(fig, out, f"change_{met}_{coh.replace('+', 'plus').replace('-', 'minus')}")
    plt.close(fig)


def fig_focality(plt, F, prof, acol, out):
    cohs = [c for c in ("R+", "R-") if c in prof]
    if not cohs:
        return
    fig, axs = plt.subplots(1, 2 * len(cohs), figsize=(W_IN, 2.0),
                            gridspec_kw=dict(wspace=0.75, width_ratios=[1.3, 0.7] * len(cohs)))
    for k, coh in enumerate(cohs):
        inc, pa, pb = prof[coh]
        ax = axs[2 * k]                                      # profile: share of the whisker-responsive fraction per area group
        o = np.argsort(-pb)
        xx = np.arange(len(inc))
        for p, mfc, lab in ((pa, "white", "learning"), (pb, COL[coh], "expert")):
            sh = np.clip(p, 0, None) / np.clip(p, 0, None).sum()
            ax.plot(xx, sh[o], "o-", color=COL[coh], mfc=mfc, mec=COL[coh], ms=3, lw=0.7, mew=0.7, label=lab)
        ax.axhline(1 / len(inc), color="0.6", lw=0.5, ls=":")
        ax.set_xticks(xx, [inc[i].replace(" areas", "").replace("Somatosensory-", "SS-") for i in o], rotation=60, ha="right",
                      fontsize=4.8)
        for t, i in zip(ax.get_xticklabels(), o):
            t.set_color(acol.get(inc[i], "k"))
        ax.set_ylabel("Share of the whisker-\nresponsive fraction (x / Σx)")
        ax.set_title(f"{COHORT_TXT[coh]} ({len(inc)} area groups)", fontsize=6, color=COL[coh], loc="left", pad=14)
        ax.legend(frameon=False, fontsize=5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0, handlelength=1.2)
        ax2 = axs[2 * k + 1]
        g = F[(F.cohort == coh) & (F.measure == "concentration")]
        if len(g):
            g = g.iloc[0]
            for x, v, lo_, hi_, mfc in ((0, g.learning, g.learning_lo, g.learning_hi, "white"), (1, g.expert, g.expert_lo, g.expert_hi, COL[coh])):
                ax2.plot([x, x], [lo_, hi_], color=COL[coh], lw=0.8)
                ax2.plot(x, v, "o", ms=4, mfc=mfc, mec=COL[coh], mew=0.8)
            ax2.plot([0, 1], [g.learning, g.expert], color=COL[coh], lw=0.6)
            top = max(g.learning_hi, g.expert_hi)
            ax2.plot([0, 0, 1, 1], [top * 1.04, top * 1.08, top * 1.08, top * 1.04], color="k", lw=0.5)
            ax2.text(0.5, top * 1.1, p_txt(g.p_perm), ha="center", va="bottom", fontsize=5)
            ax2.set_ylim(0, top * 1.3)
        ax2.set_xticks([0, 1], ["learning", "expert"]); ax2.set_xlim(-0.5, 1.5)
        ax2.axhline(1 / len(inc), color="0.6", lw=0.5, ls=":")
        ax2.text(-0.45, 1 / len(inc), "even", ha="left", va="bottom", fontsize=4.6, color="0.5")
        ax2.set_ylabel("Concentration Σx² / (Σx)²")
    fig.suptitle("Focality of whisker responsiveness across area groups", x=0.02, y=1.13, ha="left",
                 fontsize=6.6, weight="bold")
    save(fig, out, "focality_whisker")
    plt.close(fig)


def captions(out, population, S, C, A, F):
    n = S.drop_duplicates("session_id").groupby(["cohort", "stage"]).size().to_dict()
    gen = (f"Passive whisker and auditory responses of Kilosort 4 good and multi-unit clusters, {population} "
           f"({'all mice' if population == 'all' else 'day-0 sessions of non-learning mice removed, expert sessions kept'}). "
           "A unit is responsive to a modality if its stimulus-vs-baseline ROC in the active task (rate-based, 5-35 ms after stimulus "
           "onset vs the pre-trial baseline, 1000 label permutations, one-sided p on the side of the selectivity) is significant "
           "(p < 0.05); |selectivity| = |2 AUC − 1| over all tested units. Per session and area "
           f"group (≥ {MIN_UNITS} tested units): fraction responsive and mean |selectivity|; mean over sessions per stage (learning = day 0, "
           f"expert = later days). Area groups are shown for a cohort if they have ≥ {MIN_SESS} sessions at both stages. Dot: change "
           "expert − learning; bar: 95 % bootstrap CI (sessions resampled within stage, 2000); filled: Mann-Whitney U p < 0.05 (Welch "
           "t in the tables); –: area group not included for that modality. ANOVA: value ~ stage × area (type II OLS on session × area "
           f"values), p from {N_PERM} permutations of the stage label across sessions. Sessions per cohort and stage: "
           + ", ".join(f"{c} {s} {v}" for (c, s), v in sorted(n.items())) + ". No correction across area groups.")
    lines = ["# Captions (051, stage change, active-task ROC)", "", "**General.** " + gen,
             "**Focality.** Concentration index C = Σx² / (Σx)² of the mean fraction of whisker-responsive units across the included "
             "area groups (1/K: evenly spread over K groups; 1: a single group); 95 % bootstrap CI (sessions within stage), "
             f"permutation p of the change ({N_PERM} permutations of the stage label across sessions).", ""]
    for _, a in A.iterrows():
        lines.append(f"- ANOVA {a.cohort} {a.modality} {a.metric}: stage F = {a.F_stage:.2f}, {p_txt(a.p_stage_perm)} (permutation), "
                     f"stage × area F = {a.F_stage_x_area:.2f}, {p_txt(a.p_stage_x_area_perm)}; {a.n_areas} area groups, {a.n_sessions} sessions.")
    for _, f in F.iterrows():
        lines.append(f"- Focality {f.cohort} ({f.measure}): learning {f.learning:.3f} [{f.learning_lo:.3f}, {f.learning_hi:.3f}], expert "
                     f"{f.expert:.3f} [{f.expert_lo:.3f}, {f.expert_hi:.3f}], change {f.change:+.3f}, permutation {p_txt(f.p_perm)} "
                     f"({f.n_areas} area groups).")
    (out / "captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="all", choices=["all", "learners"])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = BASE / "stage_change" / a.population
    out.mkdir(parents=True, exist_ok=True)
    X = unit_metrics()
    S = session_area(X, a.population)
    C = change_table(S, rng)
    A = anova(S, C, rng)
    F, prof = focality(S, C, rng)
    S.to_csv(out / "session_area.csv", index=False)
    C.to_csv(out / "change.csv", index=False)
    A.to_csv(out / "anova.csv", index=False)
    F.to_csv(out / "focality.csv", index=False)
    plt = setup()
    order, acol = area_order_and_colours()
    for met in METRICS:
        for coh in ("R+", "R-"):
            fig_change(plt, C, A, met, coh, order, acol, out / "figures", a.population)
    fig_focality(plt, F, prof, acol, out / "figures")
    captions(out, a.population, S, C, A, F)
    json.dump(dict(script="051_stage_change.py", population=a.population, units="good + mua", alpha=ALPHA,
                   rule="whisker_active / auditory_active ROC significant (p < 0.05)", focality="sum(x^2) / (sum x)^2", min_units=MIN_UNITS, min_sessions=MIN_SESS,
                   n_boot=N_BOOT, n_perm=N_PERM, cohort_source=str(MOUSE_REF), tables=str(BASE),
                   n_sessions=int(S.session_id.nunique()), n_mice=int(S.mouse_id.nunique())), open(out / "provenance.json", "w"), indent=1)
    print(C[C.included].groupby(["metric", "cohort", "modality"]).size().to_string())
    print(A.round(3).to_string())
    print(F.round(3).to_string())
    print("ALL DONE", out)


if __name__ == "__main__":
    main()
