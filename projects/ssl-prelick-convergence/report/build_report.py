"""Merged article-style report of ssl-prelick-convergence (skills/project-report, report_lib): Part I across days,
Part II within days (learning day and expert sessions). Run on haas after 062 (full, both populations), 074-079 and the
within_day scripts 001-021; render locally with render.sh.

Main population: learners (061 rule); reference: spontaneous licks (SL). Every quoted number is read from the result
tables (062 stats, 074-079 supplements, within_day tables) through report_lib. Every figure written by the current scripts
is included (main text or supplement), with its producing script in the figure-provenance table. Unit set: PRELICK_UNITS
(all = good + mua, report in report/; stable = stable good + mua, report in report_stable/; user 2026-10-07: both kept, not
formally compared). Display equations are numbered (report_lib eq) and cited.
Run (haas):  cd ~/code/unit_spikes_analysis && PRELICK_REF=sl [PRELICK_UNITS=stable] PYTHONPATH=~/code/NWB_reader:. \
             ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-prelick-convergence/report/build_report.py
"""
from __future__ import annotations

import importlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
sys.path.insert(0, str(EA))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
from report_lib import Report  # noqa: E402

m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
POP = "learners"
PUB = m51.OUTROOT / "publication"
P1, PA = PUB / POP, PUB / "all"
WD = m51.WITHIN
NORM = m51.OUTROOT / "normalisation"
STABLE = m51.UNITS == "stable"
OUT = m51.HOME / ("report" if not STABLE else f"report_{m51.UNITS}")
UNIT_TXT = ("stable good + mua units (coverage ≥ 0.9, presence ≥ 0.5 and no dependence on probe drift; 137_stable_units)"
            if STABLE else "good + mua units")

R = Report(banned=())
num, pv, ref, figure, table, eq = R.num, R.pv, R.ref, R.figure, R.table, R.eq
G4 = ["R+ learning", "R+ expert", "R− learning", "R− expert"]


# ---------------------------------------------------------------------------------------------------------- tables
def rd(f):
    return pd.read_csv(f) if Path(f).exists() else pd.DataFrame()


ST = rd(P1 / f"stats_{POP}.csv")
SS = rd(P1 / f"stats_supp_selectivity_angles_{POP}.csv")
S78 = rd(P1 / f"stats_supp_078_{POP}.csv")
S79 = rd(P1 / f"stats_supp_079_{POP}.csv")
NS = rd(NORM / "stats.csv")
HV = rd(WD / "halves" / POP / "within_session_tests.csv")
SL = rd(WD / "slopes" / POP / "trial_slopes_tests.csv")
EP = rd(WD / "epochs" / POP / "epoch_contrasts.csv")
MM = rd(WD / "mixed_model" / "mixed_model_terms.csv")
C11 = rd(WD / "cd_trajectories" / POP / "tests.csv")
SPEC = rd(WD / "cohort_specificity" / POP / "cohort_specificity.csv")
D16 = rd(WD / "day0_predicts_expert" / POP / "correlations.csv")
M16 = rd(WD / "day0_predicts_expert" / POP / "mice.csv")
A17 = rd(WD / "area_cd_drift" / POP / "area_stage_tests_strict.csv")
AN = rd(WD / "area_cd_drift" / POP / "area_anova.csv")
CH = rd(WD / "within_change" / POP / "within_session_change_stats.csv")
DE = rd(WD / "cosyne" / POP / "drift_estimators.csv")
AC = rd(WD / "area_cd_drift" / POP / "per_area" / "area_stage_contrasts.csv")


def row(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    r = df[m]
    if len(r) == 0:
        raise KeyError(f"no row for {kw}")
    return r.iloc[0]


def P(v):
    v = float(v)
    return "n/a" if not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"= {v:.3f}" if v < 0.01 else f"= {v:.2f}")


def pp(v):
    s = P(v)
    return s[2:] if s.startswith("=") else s


def grp(panel, key, df=None, fmt="{:.3f}"):
    """R+ / R- learning -> expert means, MWU per cohort, expert cohort contrast, interaction (dots_panel stats row)"""
    r = row(ST if df is None else df, panel=panel)
    v = [num(f"{key}|mean {g}", r[f"mean_{g}"], fmt) for g in G4]
    n = [int(r[f"n_{g}"]) for g in G4]
    for f_ in ("R+ L vs E p_MWU", "R- L vs E p_MWU", "expert R+ vs R- p_MWU", "interaction_p_perm"):
        num(f"{key}|{f_}", r[f_], "{:.4g}")
    return (f"R+ {v[0]} → {v[1]} (MWU p {P(r['R+ L vs E p_MWU'])}, n = {n[0]} / {n[1]} sessions), "
            f"R− {v[2]} → {v[3]} (p {P(r['R- L vs E p_MWU'])}, n = {n[2]} / {n[3]}); experts R+ vs R− "
            f"p {P(r['expert R+ vs R- p_MWU'])}; learning × cohort interaction p {P(r['interaction_p_perm'])}")


def gi(panel, df=None):
    return f"p {P(row(ST if df is None else df, panel=panel)['interaction_p_perm'])}"


def sizes():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(["good", "mua"])]
    fr = W["fr_window:whisker_hit_vs_fa_prelick@all"]
    W = m61.learner_filter(W[W["sig:whisker_hit_vs_fa_prelick@all"].notna() & (fr >= m51.MIN_FR)])
    T = W.groupby(["cohort", "stage"]).agg(Sessions=("session_id", "nunique"), Mice=("mouse_id", "nunique"),
                                           Units=("session_id", "size")).reset_index()
    T["cohort"] = T.cohort.str.replace("-", "−")
    A = W.groupby(["area_group", "cohort", "stage"]).agg(s=("session_id", "nunique"), u=("session_id", "size")).reset_index()
    A["cell"] = A.s.astype(str) + " / " + A.u.astype(str)
    A["col"] = A.cohort.str.replace("-", "−") + " " + A.stage
    TA = A.pivot(index="area_group", columns="col", values="cell").fillna("–").reset_index()
    TA = TA.rename(columns={"area_group": "Area group"})[["Area group"] + [c for c in G4 if c in TA]]
    return T.rename(columns={"cohort": "Cohort", "stage": "Stage"}), TA




# ---------------------------------------------------------------------------------------------------------- figure inventory
# (anchor, file, producing script). Main figures in reading order (Part I, then Part II one analysis per section, summary
# figures last), then supplements.
PER_AREA = sorted(f for f in (WD / "area_cd_drift" / POP / "per_area").glob("*.png") if not f.stem.endswith("_by_stage"))
area_of = lambda f: f.stem.replace("_by_stage", "").split("_", 1)[1].replace("_", " ").replace("Somatosensory ", "Somatosensory-")
SUC = m51.OUTROOT / "single_unit_classes" / POP
SUG = m51.OUTROOT / "single_unit_geometry" / POP
# redundant figures removed (user 2026-10-08): all-mice duplicates (all-mice versions stay in the <pop> = all result folders),
# superseded COSYNE versions (Part I v2 / v3, day-0 v1 x4) and timeline summaries (006, 008, 005), non-strict and
# both-stage area versions, per-area cohort-within-stage duplicates, unit-sampling and >= 8 WH variants
MAIN = [
    ("fig-task", P1 / "Fig1_task_data.png", "062 fig1"),
    ("fig-single", P1 / "Fig2_single_neurons.png", "062 fig2"),
    ("fig-classes", SUC / "neuron_class_psth.png", "081 + 082"),
    ("fig-distance", P1 / "Fig3_population_distance.png", "062 fig3"),
    ("fig-areas", P1 / "Fig4_areas.png", "062 fig4"),
    ("fig-decoders", P1 / "Fig5_decoders.png", "062 fig5"),
    ("fig-cosyne-across", P1 / "COSYNE_across_days.png", "083"),
    ("fig-cd", WD / "cd_trajectories" / POP / "cd_trajectories.png", "within_day 011"),
    ("fig-halves", WD / "halves" / POP / "within_session_time.png", "within_day 001"),
    ("fig-psth-halves", WD / "psth_halves" / POP / "psth_halves_groups.png", "within_day 004 + 012"),
    ("fig-slopes", WD / "slopes" / POP / "trial_slopes.png", "within_day 002"),
    ("fig-change", WD / "within_change" / POP / "within_session_change.png", "within_day 013"),
    ("fig-epochs", WD / "epochs" / POP / "epoch_comparison.png", "within_day 003"),
    ("fig-area-day0", WD / "area_cd_drift" / POP / "area_cd_drift_day0_strict.png", "within_day 015 + 017"),
    ("fig-area-expert", WD / "area_cd_drift" / POP / "area_cd_drift_expert_strict.png", "within_day 015 + 017"),
    ("fig-area-matrix", WD / "area_cd_drift" / POP / "area_cd_drift_matrix.png", "within_day 015 + 017"),
    ("fig-specificity", WD / "cohort_specificity" / POP / "cohort_specificity.png", "within_day 014"),
    ("fig-day0-expert", WD / "day0_predicts_expert" / POP / "day0_predicts_expert.png", "within_day 016"),
    ("fig-cosyne-day0", WD / "cosyne" / POP / "COSYNE_day0_v2.png", "within_day 019")]
SUPP1 = [  # Part I supplements, learners
    ("fig-s-converging", P1 / "Fig2S_converging_neurons.png", "062 fig2_supplement"),
    ("fig-s-conv-sign", P1 / "FigS_converging_sign.png", "079"),
    ("fig-s-abssel", P1 / "FigS_abs_selectivity.png", "075"),
    ("fig-s-classes", SUC / "single_unit_classes.png", "081"),
    ("fig-s-sug", SUG / "single_unit_geometry.png", "080"),
    ("fig-s-sug-d", SUG / "single_unit_geometry_effectsize.png", "080"),
    ("fig-s-lambda", P1 / "Fig3S_lambda_lambdaLDA.png", "062 fig3s_lambda"),
    ("fig-s-angles", P1 / "FigS_angles.png", "075"),
    ("fig-s-norm", NORM / f"geometry_normalisation_{POP}.png", "074"),
    ("fig-s-shift", P1 / "FigS_shift_geometry.png", "076 + 078"),
    ("fig-s-cd", P1 / "FigS_cd_estimators.png", "076 + 078 (coding_direction.py)"),
    ("fig-s-quiet", P1 / "FigS_quiet_windows.png", "077 + 078"),
    ("fig-s-decq", P1 / "FigS_decoder_quality.png", "079"),
    ("fig-s-perf", P1 / "FigS_performance.png", "077 + 078")]
SUPP2 = [  # Part II supplements, learners
    ("fig-s-cd-by-stage", WD / "cd_trajectories" / POP / "cd_trajectories_by_stage.png", "within_day 023"),
    ("fig-s-drift-compact", WD / "cosyne" / POP / "drift_compact.png", "within_day 024")] + [
    (f"fig-s-pa-{i + 1:02d}", f, "within_day 020") for i, f in enumerate(PER_AREA)] + [
    ("fig-s-wh10", WD / "wh_trials" / POP / "wh_trials_N10.png", "within_day 022 (N = 10)"),
    ("fig-s-wh20", WD / "wh_trials" / POP / "wh_trials_N20.png", "within_day 022 (N = 20)"),
    ("fig-s-oddeven", WD / "halves" / POP / "within_session_oddeven.png", "within_day 001"),
    ("fig-s-controls", WD / "halves" / POP / "within_session_controls.png", "within_day 001"),
    ("fig-s-decschemes", WD / "cosyne" / "decoder_schemes.png", "within_day 010"),
    ("fig-s-epochs-n4", WD / "epochs_n4" / POP / "epoch_comparison.png", "within_day 003 (4 events per class)")]
ALLM = []                                   # all-mice duplicates removed (2026-10-08)
FILE = {a_: f_ for a_, f_, *_ in MAIN + SUPP1 + SUPP2}

# numbered display equations, in reading order (Methods); each is cited where the quantity is used
EQ = {
    "eq-lick": r"t_{\mathrm{lick}} = t_{\mathrm{start}} + \left(t_{\mathrm{lick,NWB}} - t_{\mathrm{rw,start}}\right)",
    "eq-auc": r"\mathrm{AUC} = P(r_B > r_A) + \tfrac12\,P(r_B = r_A), \qquad s = 2\,\mathrm{AUC} - 1 \in [-1, 1]",
    "eq-likeness": r"c = \frac{|m_W - m_S| - |m_W - m_A|}{|m_A - m_S|} \in [-1, 1]",
    "eq-sigma": r"\sigma_u = \sqrt{\tfrac13 \textstyle\sum_{c \in \{W,A,S\}} \mathrm{Var}_c(r_u)}",
    "eq-cvd": r"d(X, Y) = \frac{1}{N} \,\big\langle (\bar x_a - \bar y_a) \cdot (\bar x_b - \bar y_b) \big\rangle_{\mathrm{splits}}",
    "eq-bias": r"E[d] = \frac{\|\mu_X - \mu_Y\|^2}{N}, \qquad E\!\left[\frac{\|\bar x - \bar y\|^2}{N}\right] = "
               r"\frac{\|\mu_X - \mu_Y\|^2 + \mathrm{tr}(\Sigma)\,(1/n_X + 1/n_Y)}{N}",
    "eq-along": r"a = \frac{(\mu_W - \mu_S)\cdot(\mu_A - \mu_S)}{N} = \tfrac12\left[d(W,S) + d(A,S) - d(W,A)\right], \qquad "
                r"o^2 = d(W,S) - \frac{a^2}{d(A,S)}",
    "eq-triangle": r"x = \frac{d(W,S) - d(W,A) + d(A,S)}{2\sqrt{d(A,S)}}, \qquad y = \sqrt{d(W,S) - x^2}",
    "eq-lambda": r"\lambda = \frac{(\mu_W - \mu_S)\cdot(\mu_A - \mu_S)}{\|\mu_A - \mu_S\|^2} = \frac{a}{d(A,S)}",
    "eq-angle": r"\lambda = \rho\,\cos\theta, \qquad \cos\theta = \frac{a}{\sqrt{d(W,S)\,d(A,S)}}, \qquad "
                r"\rho = \sqrt{\frac{d(W,S)}{d(A,S)}} = \frac{\|\mu_W - \mu_S\|}{\|\mu_A - \mu_S\|}",
    "eq-dprime": r"d' = \frac{\bar p_{AH} - \bar p_{SL}}{\sqrt{\tfrac12\left(\mathrm{Var}(p_{AH}) + \mathrm{Var}(p_{SL})\right)}}",
    "eq-cd": r"c_i = \frac{x_i \cdot w - \overline{x\cdot w}_{S}}{\overline{x\cdot w}_{A} - \overline{x\cdot w}_{S}} "
             r"\qquad (\mathrm{SL} = 0,\ \mathrm{AH} = 1)",
    "eq-shift": r"q_{\mathrm{corrected}} = q_{\mathrm{observed}} - \operatorname{median}_k\, q_k",
    "eq-transfer": r"\mathrm{transfer}_{\mathrm{bin}} = \frac{F_{WH} - \mathrm{FPR}}{\mathrm{TPR} - \mathrm{FPR}}, \qquad "
                   r"\mathrm{transfer}_{\mathrm{prob}} = \frac{\bar P(AH|WH) - \bar P(AH|SL)}{\bar P(AH|AH) - \bar P(AH|SL)}",
    "eq-interaction": r"I = \left[\bar x_{E} - \bar x_{L}\right]_{R+} - \left[\bar x_{E} - \bar x_{L}\right]_{R-}",
    "eq-ols": r"c_i = \alpha_k + \beta_k\,\tau_i, \qquad \tau_i = \frac{t_i - t_{\mathrm{first}}}{t_{\mathrm{last}} - t_{\mathrm{first}}}, "
              r"\qquad k \in \{WH, AH, SL\}",
    "eq-drift": r"D_{\mathrm{ref}} = \beta_{WH} - \beta_{\mathrm{ref}}, \qquad \mathrm{ref} \in \{SL, AH\}",
    "eq-epochs": r"\text{within} = D_{L,\mathrm{late}} - D_{L,\mathrm{early}}, \qquad \text{across} = D_{E,\mathrm{early}} - "
                 r"D_{L,\mathrm{early}}, \qquad \text{carry-over} = D_{E,\mathrm{early}} - D_{L,\mathrm{late}}",
    "eq-mixed": r"c_i \sim \mathrm{WH}_i \times \tau_i \times \mathrm{R+} + (1 + \tau_i + \mathrm{WH}_i \mid \mathrm{session})",
}
E = lambda k, indent="": eq(k, EQ[k], indent)

# anchors: main figures, Part I / Part II / all-mice supplements, tables, equations
lab = []
for i, (a, *_r) in enumerate(MAIN):
    lab.append((a, f"Figure {i + 1}"))
for i, (a, *_r) in enumerate(SUPP1 + SUPP2):
    lab.append((a, f"Figure S{i + 1}"))
lab += [("tbl-estimators", "Table 1"), ("tbl-mixed", "Table 2"), ("tbl-halves", "Table 3"), ("tbl-epochs", "Table 4"),
        ("tbl-area-anova", "Table 5"),
        ("tbl-s-sizes", "Table S1"), ("tbl-s-sizes-areas", "Table S2"), ("tbl-s-slopes", "Table S3"),
        ("tbl-s-area-contrasts", "Table S4"), ("tbl-a-params", "Table A1"), ("tbl-a-files", "Table A2"),
        ("tbl-provenance", "Table A3")]
lab += [(k, f"Eq. ({i + 1})") for i, k in enumerate(EQ)]
R.register(lab)
L = lambda a_: R.anchors.get(a_, a_)


# ---------------------------------------------------------------------------------------------------------- captions
def captions():
    C = {
        "fig-task": f"Task and data. (a) Task and reward contingencies: auditory licks rewarded in both cohorts, whisker licks only in R+; spontaneous licks never rewarded. (b) Alignment: the 100 ms before the corrected first lick (shaded); hit baseline 1 s before trial start. (c) Reaction-time distributions of whisker and auditory hits (pooled trials; legend: medians). (d) Median reaction time per session, mean ± s.e.m. over sessions, by cohort and stage. (e) Sessions per cohort and stage, with mice and tested units. (f) Recorded units in Allen CCF coordinates (sagittal and coronal projections; random subset; R+ green, R− magenta). Population: {POP}.",
        "fig-single": f"Single neurons. (a) Example neurons (R+ expert top, R− expert bottom): converging (AH vs SL and WH vs SL significant, same sign), reward-lick-only (AH vs SL significant, |WH vs SL| < 0.1) and whisker-hit-only; PSTH (10-ms bins, Gaussian σ = 20 ms, mean ± s.e.m. over events) above rasters (≤ 25 events per type); titles: area and selectivity s = 2·AUC − 1. (b–e) Schematics (top) and per-session values (bottom; open: learning, filled: expert; large symbol: mean ± s.e.m.): (b) fraction of tested units with significant WH vs SL ROC (1,000 label permutations, p < 0.05); (c) fraction with significant WH vs AH ROC; (d) converging neurons as a fraction of reward-lick neurons (either sign); (e) AH-likeness c of reward-lick neurons. (f) Schematic of the shared hit code; (g) Spearman r across units per session; (h, i) selectivities of all tested units (log density) with OLS fits per stage (learning dashed, expert solid), Pearson r and the session-bootstrap Δr. (j) Converging neurons per area, expert sessions, mean ± s.e.m. over sessions; ANOVA with cohort labels permuted across mice. Brackets: MWU p; titles: learning × cohort interaction, mouse-level permutation. Sessions with ≥ 10 tested units; {POP}.",
        "fig-distance": f"Population distance. (a) Schematic: each event type is a mean pre-lick population vector; d = cross-validated squared Euclidean distance per unit (50 random split halves); Δd = d(WH, SL) − d(WH, AH). (b) Group-mean geometry, real data: SL at the origin, AH on the x-axis at √d(AH, SL), WH placed from its two distances; axes in √(squared distance per unit); learning dashed / open, expert solid / filled. (c–f) Per session: Δd, d(WH, SL), d(WH, AH), d(AH, SL). (g) RT matching is not defined for spontaneous licks. (h) Cumulative distributions of Δd. (i) Cross-validated dot product (WH − SL)·(AH − SL) per unit = λ·d(AH, SL). (j) Orthogonal squared distance per unit. Statistics as {L('fig-single')}; sessions with ≥ 5 units and ≥ 4 events per class; {POP}.",
        "fig-areas": "Brain areas. (a) Change of Δd from learning to expert per area group (mean ± s.e.m.; filled: MWU p < 0.05), area groups with ≥ 3 sessions at both stages in a cohort; right: interaction p (mouse permutation) and family-wise p (max-|z|). Inset: ANOVA Δd ~ cohort × stage × area, F test and cohort-label permutation across mice. (b) Mean Δd per area group, cohort and stage. (c) Δd per session for four area groups. (d) Lick-aligned PSTHs, rate − baseline (spikes/s), mean ± s.e.m. over units (expert sessions). (e) The same, averaged over units within session, then mean ± s.e.m. over sessions. Baselines: hits 1 s before trial start, SL [−1, −0.5] s before the lick; shaded: pre-lick window.",
        "fig-decoders": "Decoders. (a) Raw balanced accuracy of the single-session AH vs SL decoder (dots: sessions) with the median (line) and 2.5–97.5 % range (band) of each group's linear-shift chance level. (b) Accuracy above chance; (c) WH decoded as AH (yes/no) and (d) (probability), as transfer (0 = like SL, 1 = like AH); (e) P(AH|WH) − P(AH|SL); all chance-corrected by 40 linear shifts. (f–i) Hierarchical pseudo-populations (mice → sessions → neurons → events) at M = 2,000 neurons: bootstrap distributions and 95 % CI. (j–l) Readouts vs M. (m) Learning → expert change at M = 2,000 with bootstrap CI; top: interaction p.",
        "fig-timeline": "Convergence timeline (all mice). (a) First-lick-aligned population PSTHs of R+ (top) and R− (bottom) mice: day-0 early half, day-0 late half and expert sessions; rate − baseline, mean over units within session, then mean ± s.e.m. over sessions (10-ms bins, 30-ms boxcar); shaded: pre-lick window. (b) Day 0: mean WH − SL score on the session's reward-lick coding direction (5-fold cross-validated SL → AH mean-difference axis; SL = 0, AH = 1) in five bins of normalised session time, mean ± s.e.m. over sessions; right: expert session mean. (c) Per-session WH − SL slope on day 0 (OLS), Mann-Whitney and mouse-permutation R+ vs R−; title: mixed-model R+ vs R−. (d) Common footing (epochs_n4_u150x10: 4 events per class, 10 draws of 150 units): decoder readout P(AH|WH) − P(AH|SL) − chance per half, mean over mice and 95 % hierarchical-bootstrap CI. (e) Within day 0, across days and carry-over changes of (d), 95 % CI (filled: p < 0.05); top: R+ vs R− by cohort-label permutation across mice.",
        "fig-timeline-exp": "Convergence timeline, expanded (all mice): methods with real examples (row 1), results (row 2) and controls (row 3). Row 1: (a) one R+ learning-day session, every analysed event on the session time line, split into halves at the midpoint between the two middle auditory hits; (b) schematic of the cross-validated SL → AH axis and the trial score (SL = 0, AH = 1); (c, d) trial scores of an R+ and an R− example against normalised time with OLS lines for WH and SL (the WH − SL slope is the tested quantity). Row 2: PSTHs per half, WH − SL score across the day-0 session, day-0 slopes, decoder per half on the common footing, changes within / across days (as Figure 6). Row 3: whisker-hit reaction time per half, WH relative to AH (slope of WH − AH), and the same within-day change with an odd/even split (no temporal order).",
        "fig-wd-rminus": "R− mice change more within the learning day than within expert sessions (all mice). (a) First-lick-aligned PSTHs of R− sessions (004): day-0 early / late halves and expert early / late halves; WH, AH, SL; rate − baseline, mean over units per session, mean ± s.e.m. over sessions. (b) Whisker-hit rate across the session (002 trajectories), R− day 0 vs expert (R+ day 0 for reference). (c) WH − SL score on the session's SL → AH axis across the session (5 bins), mean ± s.e.m. over sessions. (d) Epoch means on a common footing (003, 4 events per class): Δd normalised by the session's d(AH, SL), hierarchical-bootstrap 95 % CI. (e) Size of the within-session change |late − early| per session, day 0 vs expert, for the position and the normalised distance; p: hierarchical bootstrap of the day 0 − expert difference and Mann-Whitney U.",
        "fig-halves": f"Session halves, learning day and expert sessions ({POP}). Each session split at the midpoint between its two middle auditory hits; events count-matched per half (20 subsamples). (a–f) Early (open) and late (filled) half per group, lines join the halves of a session: Δd, λ, decoder P(AH|WH) − P(AH|SL) − chance (whole-session decoder; cross-half decoder: trained on one half, read out on the other), accuracy − chance. (g–l) Late − early per session; titles: day-0 R+ vs R− (mouse permutation) and cohort × stage × half interaction.",
        "fig-slopes": f"Trial-level trajectories ({POP}). (a–d) Scores on the session's SL → AH axis (5-fold coding direction; SL = 0, AH = 1) in five bins of normalised session time per class, mean ± s.e.m. over sessions; dashed band: expert WH level. (e–h) Per-session OLS slopes against normalised time: WH, WH − SL, decoder P(AH|WH) and decoder WH − SL. (i, j) Day-0 fitted WH score at the start and end of the session vs the expert session mean, per cohort. (k, l) Whisker hit and spontaneous lick rates (events / min) over the session.",
        "fig-epochs": f"Within-day vs across-day change on a common footing ({POP}): phase-matched halves, 6 events per class, 150 units. (a–c) Epoch means with 95 % hierarchical-bootstrap CI (mice, then sessions): WH − SL position on the session's coding direction, decoder P(AH|WH) − P(AH|SL) − chance, and Δd / d(AH, SL). (d–f) Contrasts (filled: bootstrap p < 0.05): within day 0, across days (early and late phase) and carry-over; top: R+ vs R− by mouse-level permutation.",
        "fig-s-converging": "Converging neurons. (a) Example reward-lick neurons (PSTH only) that do and do not also separate WH from SL. (b) Functional types of converging neurons, reward-lick-only neurons and all tested units in the stimulus-aligned ROC analyses; right: enrichment of converging neurons. (c, d) Density of converging neurons on coronal and sagittal slabs (expert sessions).",
        "fig-s-conv-types": "Converging-neuron functional types (standalone version of panel b of the previous figure): fraction of converging, reward-lick-only and all tested units significant in each stimulus-aligned ROC analysis, per cohort and stage, and the enrichment of converging neurons.",
        "fig-s-examples-psth": "Example reward-lick neurons, PSTH only (good units, mean pre-lick rate ≥ 8 Hz, AH vs SL selectivity ≥ 0.6): two R+ expert neurons that also separate WH from SL with the same sign and two R− expert neurons that do not (|WH vs SL| < 0.1); 10-ms bins, 50-ms boxcar, rate − the unit's mean in [−600, −400] ms.",
        "fig-s-conv-sign": f"Converging neurons by the sign of the reward-lick response ({POP}). Reward-lick neurons (AH vs SL significant) split into AH > SL and AH < SL; sessions with ≥ 3 neurons of that sign. (a) Fraction of AH > SL neurons with WH > SL significant; (b) fraction of AH < SL neurons with WH < SL significant; (c) both signs (definition of {L('fig-single')}d). (d–f) Mean AH-likeness c of the same neurons. Statistics as {L('fig-single')}.",
        "fig-s-abssel": f"Absolute selectivity and its sign ({POP}; one dot per session; sessions with ≥ 10 tested units). Columns: WH vs SL (+: WH > SL), WH vs AH (+: AH > WH), AH vs SL (+: AH > SL). Rows: (a–c) mean |s| over all tested units; (d–f) fraction significant with s > 0; (g–i) fraction significant with s < 0; (j–l) mean |s| over significant units. Statistics as {L('fig-single')}.",
        "fig-s-lambda": "λ and λ_LDA. (a) Schematic of λ. (b) Single-event projections on the cross-validated SL → AH axis for an example R+ and R− expert session (SL = 0, AH = 1; bars: means). (c) Pooled single-event projections per group. (d) λ per session. (e) λ vs Δd. (f) RT matching not defined for SL. (g) Cumulative distributions of λ. (h) Example sessions in the plane of the shrinkage-LDA axis (x; SL = 0, AH = 1) and the WH off-axis direction (y; defined on half of the WH events, the other half shown); arrow: mean-difference axis. (i) λ vs λ_LDA. (j) λ_LDA per session. (k) Held-out d′ of the LDA axis (AH vs SL). (m) Cumulative distributions of λ_LDA.",
        "fig-s-angles": "Angle and length behind λ (cross-validated distances; sessions with a valid axis). (a) Angle θ (degrees) between WH − SL and AH − SL; (b) cos θ; (c) length ratio ρ = |WH − SL| / |AH − SL|; (d) λ = ρ cos θ; (e) sessions in the (cos θ, ρ) plane with iso-λ lines.",
        "fig-s-norm": "Unit scaling. Δd, λ and the three distances under raw spikes/s (left), pooled-SD z-scoring (current) and within-class SD (diagonal noise normalisation), same events and splits; right: session values against the current scaling (z-scored across sessions), Spearman ρ.",
        "fig-s-shift": "Linear-shift control. Δd, along component a and λ: (a–c) observed values; (d–f) the same values corrected by the linear-shift null, i.e. observed minus the median of the value recomputed on 40 linear shifts of the time-ordered event labels against the activity (|k| = 5 … n/3 events, no wrap-around); for λ, numerator a and denominator d(AH, SL) are corrected separately before the ratio. A value near the observed one means the effect is not produced by slow drifts or event-order structure.",
        "fig-s-cd": "Coding-direction estimators. (a) λ vs the mean WH score of the unified split-half trial scores (equal by construction). (b) λ vs the mean WH score of the Part II 5-fold coding direction (sessions with held-out d′ ≥ 0.3). (c) 5-fold CD per cohort and stage.",
        "fig-s-quiet": "Quiet windows. Raw rates in [−200, −100] ms before every trial start (lick-free: no lick in the preceding 1 s) placed on the readouts: (a–c) cross-validated squared distance per unit to SL, AH and WH; (d) position on the SL → AH mean-difference axis (λ_Q); (e) WH computed the same way; (f) position on the shrinkage-LDA axis. Events are baseline-subtracted pre-lick rates, quiet windows raw rates.",
        "fig-s-decq": f"Decoder quality vs readouts ({POP}; whole brain). (a–d) Per session, raw balanced accuracy of the AH vs SL decoder against the chance-corrected readouts (transfer yes/no, transfer probability, P(AH|WH) − P(AH|SL)) and against λ; open: learning, filled: expert; titles: Spearman ρ per cohort. (e–h) The same readouts within tertiles of decoder accuracy (bounds in the title), mean ± s.e.m. per cohort × stage.",
        "fig-s-perf": "Session quantities vs behaviour: rows, single-neuron, geometry and decoder measures; columns, whisker hit rate, false-alarm rate (no-stim trials) and d′ (whisker hit vs no-stim lick rate) on the analysed trials. Open: learning, filled: expert; one OLS fit per cohort with 95 % CI band (solid if Spearman p < 0.05); titles: Spearman ρ, p, n.",
        "fig-s-cosyne-v2": "COSYNE summary figure, Part I (condensed, learning → expert): population PSTHs (pre-trial baseline), converging neurons, population geometry, distance difference and decoder readout, assembled from the panels of Figures 2–5 with the same data and statistics.",
        "fig-s-cosyne-v3": "COSYNE summary figure, Part I, v3 (6 panels): (a) population PSTHs; (b) converging neurons; (c) population geometry (triangles); (d) distance difference; (e) decoder P(AH|WH) − P(AH|SL) (accuracy in the title); (f) area heatmap of mean distance difference (area groups with ≥ 3 sessions in all four cohort × stage groups). Same data and statistics as Figures 2–5.",
        "fig-s-oddeven": f"Session halves, odd/even null split ({POP}): events alternately assigned to the two halves, which removes time; changes should vanish. Layout as {L('fig-halves')}.",
        "fig-s-controls": f"Session-half controls ({POP}): d(AH, SL), event rates, reaction times, firing rates and half durations per half.",
        "fig-s-decschemes": "Decoder readouts across unit-sampling schemes (all mice; 003 with 4 events per class): within-day, across-day and carry-over changes for one draw of 150 units, 10 draws of 150, 10 of 50 and 100, 5 of 400, and all units of each session.",
        "fig-s-epochs-n4": f"As {L('fig-epochs')}, 4 events per class (one draw of 150 units).",
    }
    for v in ("u50x10", "u100x10", "u150x10", "u400x5", "uall"):
        C[f"fig-s-epochs-{v}"] = (f"As {L('fig-epochs')}, 4 events per class, unit sampling {v} (units × draws; uall: all units); "
                                  + (f"the version used by {L('fig-timeline')} and {L('fig-timeline-exp')}." if v == "u150x10" else "see the decoder-scheme comparison."))
    for a, f, s, base in ALLM:
        C[a] = f"As {L(base)}, all mice."
    return C



def captions_part1_new(C):
    C.update({
        "fig-classes": f"Converging neurons by stimulus class across learning ({POP}). Classes from the stimulus-aligned ROC analyses: sensory = significant whisker_active or auditory_active; lick-gated = not sensory, whisker_choice significant; non-sensory = neither. Rows 1–3: first-lick-aligned PSTHs of converging neurons (pre-lick AH > SL and WH > SL, significant) of each class; row 4: reward-lick-only neurons (AH > SL, WH vs SL not significant); columns: R+ day 0, R+ expert, R− day 0, R− expert; rate − pre-trial baseline, mean over units within session, mean ± s.e.m. over sessions; shaded: pre-lick window. Right column: converging fraction among reward-lick neurons per class (mean ± s.e.m. over sessions) and the class composition of converging neurons.",
        "fig-cosyne-across": f"COSYNE figure, across days ({POP}). (a) Population PSTHs (all tested neurons) per cohort and stage; (b) converging non-sensory neurons; (c) converging fraction among reward-lick neurons (squares: the three stimulus classes); (d) AH-likeness ({ref('eq-likeness')}); (e) group geometry from cross-validated distances ({ref('eq-cvd')}); (f) Δd per session; (g) angle θ and length ratio ρ ({ref('eq-angle')}), mean and 95 % hierarchical-bootstrap CI, with the cohort × stage interaction of the change and its CI; (h) change of Δd per area group, R+ and R−, with the FDR q of the per-area interaction; title: ANOVA Δd ~ cohort × stage × area. Statistics as {L('fig-single')}.",
        "fig-s-classes": f"Single neurons ({POP}): (a–d) cross-validated squared ROC selectivity per unit for WH vs SL, AH vs SL and WH vs AH (the single-neuron counterparts of the three distances, {ref('eq-cvd')}) and their Δd counterpart; (e–h) fraction selective with event counts matched (8 events per class; close to the 5 % false-positive rate, not used for inference); (i–p) converging fraction, AH-likeness and per-unit cross product within the sensory, lick-gated and non-sensory classes.",
        "fig-s-sug": f"Single-neuron (ROC) counterpart of the population geometry ({POP}). (a–d) ROC selectivity WH vs SL against AH vs SL for every unit per group (density) with the mean gain (slope through the origin); (e–h) per session, cross-validated across event halves: alignment (cos θ analogue), gain (λ analogue, {ref('eq-lambda')}), spread ratio (ρ analogue) and the WH-specific residual (o² analogue); (i, j) per-unit contribution to the cross product of converging and WH-only neurons; (k, l) gain and alignment per area group (day 0 → expert change; interaction p and FDR q).",
        "fig-s-sug-d": f"As {L('fig-s-sug')}, with per-unit effect sizes (mean difference / within-class SD, {ref('eq-sigma')}) instead of ROC selectivity; this version corresponds exactly to the population geometry.",
    })
    return C


def captions_part2(C):
    """captions of the Part II figures (one per analysis) and their supplements"""
    C.update({
        "fig-cd": f"Reward-lick coding direction within sessions, learning day and expert sessions ({POP}). Every event's score on the session's 5-fold CD ({ref('eq-cd')}; SL = 0, AH = 1). (a–d) Score of WH, AH and SL in five bins of normalised session time τ, per cohort and stage; mean ± s.e.m. over sessions (n sessions in the titles). (e–h) Score differences across the session, day 0 (dashed, open) vs expert (solid): WH − SL (e, f) and AH − WH, the distance of WH to AH (g, h). (i–l) Per-session drifts ({ref('eq-drift')}) β_WH − β_SL and β_WH − β_AH and the class slopes β_WH and β_SL ({ref('eq-ols')}); open: day 0, filled: expert; large symbol: mean ± s.e.m.; brackets: Mann-Whitney U (day 0 vs expert per cohort; expert R+ vs R−); titles: cohort × stage interaction ({ref('eq-interaction')}), cohort labels permuted across mice. (m) Slope of the AH score. (n) Mixed-model drift estimates ({ref('eq-mixed')}; all mice) ± 95 % CI, both references and stages.",
        "fig-psth-halves": f"First-lick-aligned activity per session half, all four groups ({POP}). Rows: R+ and R−; columns: day-0 early half, day-0 late half, expert early half, expert late half; WH, AH, SL; rate − baseline, mean over units per session, mean ± s.e.m. over sessions (sessions with ≥ 3 events of the class in the half; 10-ms bins, 30-ms boxcar); shaded: pre-lick window. Right column: WH − SL difference trace, early (dashed) vs late (solid) half, day 0 (dark) and expert (light).",
        "fig-change": f"Within-session change on the learning day and in expert sessions, R+ and R− at equal weight ({POP}). (a–c) Whisker-hit, auditory-hit and spontaneous-lick rates (events / min) in five bins of session time, mean ± s.e.m. over sessions; day 0 dashed / open, expert solid. (d, e) Common footing (003, 4 events per class, 150 units): Δd / d(AH, SL) and position on the SL → AH axis per half, mean over mice with 95 % hierarchical-bootstrap CI. (f–h) Size of the within-session change |late − early| per session for the normalised distance, the position and the decoder transfer, day 0 (D0) vs expert (E) per cohort; dots: sessions; p: hierarchical bootstrap of the day 0 − expert difference.",
        "fig-area-day0": f"Day-0 drift per area group ({POP}): the CD built from the units of one area group (≥ 5 units, held-out d′ ≥ 0.3) and the drifts of {ref('eq-drift')}; area groups with ≥ 3 sessions in every cohort × stage condition. R+ and R− mean ± s.e.m. over sessions; filled: drift ≠ 0 (Wilcoxon p < 0.05). Columns: R+ vs R− mouse-level permutation p, its Benjamini-Hochberg q over the area groups shown (bold: q < 0.05), Mann-Whitney U and Welch p. Titles: two-way ANOVA drift ~ cohort × area (cohort terms by cohort-label permutation across mice; area: F test).",
        "fig-area-expert": f"As {L('fig-area-day0')}, expert sessions.",
        "fig-area-matrix": f"Area group × group matrix of the per-area drifts ({POP}): mean drift per area group for R+ day 0 (D0), R+ expert (E), R− D0 and R− E (colour; '*': drift ≠ 0, Wilcoxon p < 0.05); right: R+ vs R− mouse-permutation p (FDR q) on day 0, in experts and for the cohort × stage interaction ({ref('eq-interaction')}). Titles: ANOVA drift ~ cohort × stage × area (cohort-label permutation across mice). Area groups with ≥ 3 sessions at both stages in both cohorts.",
        "fig-specificity": f"Which within-day cohort differences are specific to the learning day ({POP})? For 30 session-level within-day measures (reward-lick CD drifts and slopes, half-to-half changes of distances, λ, decoders, rates and reaction times, trial slopes, common-footing epochs): Hedges' g of R+ − R− with 95 % CI on day 0 (open) and in expert sessions (filled); red star: mouse-level permutation p < 0.05; ◆: learning-day specific (day-0 p < 0.05, expert p ≥ 0.05 and cohort × stage interaction p < 0.05). Grey lines separate the measure families.",
        "fig-day0-expert": f"Day-0 effects per mouse vs the mouse's expert outcome ({POP}; mice with a day-0 and an expert session; expert values averaged over expert sessions). Rows: day-0 drift of WH relative to SL and to AH, day-0 WH score at session end, day-0 whisker hit rate, day-0 d′ and day-0 change of the whisker hit rate; columns: expert whisker hit rate, expert d′ and expert WH CD score. One OLS fit per cohort with 95 % CI band (solid if Spearman p < 0.05); titles: Spearman ρ and Pearson r per cohort.",
        "fig-cosyne-day0": f"COSYNE figure, learning day ({POP}). Row 1: (a, b) score of every event type across the day-0 session ({ref('eq-cd')}), R+ and R−; (c, d) one example session per cohort (≥ 20 WH, |β_SL| and |β_AH| < 0.35, largest WH slope in the cohort's direction): every event's score, OLS line per class and both drifts. Row 2, relative to spontaneous licks: (e) WH − SL across the session, both cohorts; (f) drift β_WH − β_SL, mean ± s.e.m. over sessions, mouse-permutation p; (g) the same per area group (filled ≠ 0; p and FDR q; title: cohort × area ANOVA). Row 3, relative to auditory hits: (h) AH − WH; (i) β_WH − β_AH; (j) per area group.",
        "fig-s-area-all": f"Per-area drifts, both stages together ({POP}; the figure of 015): R+ and R− day 0 (open) and expert (filled), mean ± s.e.m. over sessions; faded: n.s. vs 0; right: R+ vs R− p on day 0, in experts and the cohort × stage interaction.",
        "fig-s-area-day0-all": f"As {L('fig-area-day0')}, every area group with ≥ 3 day-0 sessions in at least one cohort (R+ vs R− only where both cohorts have ≥ 3).",
        "fig-s-area-expert-all": f"As {L('fig-area-expert')}, every area group with ≥ 3 expert sessions in at least one cohort.",
        "fig-s-cosyne-v1": f"COSYNE figure, learning day, first version ({POP}): (a, b) class scores across the day-0 session; (c) WH − SL and (d) AH − WH, both cohorts; (e) drifts per session (dots: sessions); (f) per-area matrix of day-0 drifts with R+ vs R− p (FDR q); areas with ≥ 3 day-0 sessions in both cohorts.",
        "fig-s-cosyne-v1-dots": f"As {L('fig-s-cosyne-v1')}, panel f as a dot plot.",
        "fig-s-cosyne-v1-strict": f"As {L('fig-s-cosyne-v1')}, panel f restricted to area groups with ≥ 3 sessions in every cohort × stage condition.",
        "fig-s-cosyne-v1-strict-dots": f"As {L('fig-s-cosyne-v1-strict')}, panel f as a dot plot.",
        "fig-s-slopes-wh8": f"As {L('fig-slopes')}, sessions with ≥ 8 whisker hits.",
    })
    C.update({
        "fig-s-cd-by-stage": f"As {L('fig-cd')}, with the cohorts contrasted within each stage ({POP}): (a–d) WH − SL and AH − WH across the session, R+ and R− overlaid, day 0 and expert; (e–h) class scores per cohort × stage; (i–l) per-session drifts ({ref('eq-drift')}), R+ vs R− within each stage (mouse-level permutation p).",
        "fig-s-drift-compact": f"Within-session drifts ({ref('eq-drift')}) in the compact format of the COSYNE figure ({POP}): mean ± s.e.m. over sessions, R+ and R−, day 0 (open) and expert (filled), whole brain and every area group with ≥ 3 sessions in every cohort × stage condition; above each stage: R+ vs R− mouse-level permutation p; titles: cohort × stage interaction.",
        "fig-s-wh10": f"The reward-lick CD along the first 10 whisker hits ({POP}; sessions with ≥ 10 whisker hits). (a, b) Whisker-hit projections, every trial of every session (dots) and the mean ± s.e.m. over sessions per trial number, R+ and R− overlaid, day 0 and expert; (c, d) per-session change of the projection over the 10 trials (slope against the trial number scaled to 0–1), R+ vs R− within stage; (e–h) the same relative to time-matched references: WH − SL and AH − WH, with SL and AH taken from the session's OLS lines ({ref('eq-ols')}) at the time of each whisker hit; (i–l) per-session change of WH − SL and WH − AH; (m–p) as (i–l) restricted to the sessions that also enter the 20-trial version.",
        "fig-s-wh20": f"As {L('fig-s-wh10')}, first 20 whisker hits (sessions with ≥ 20 whisker hits).",
    })
    for i, f in enumerate(PER_AREA):
        C[f"fig-s-pa-{i + 1:02d}"] = (f"{area_of(f)}: the analyses of {L('fig-cd')} with the CD built from this area group's "
                                       f"units ({POP}). (a–d) class scores per cohort × stage; (e–h) WH − SL and AH − WH, day 0 "
                                       f"vs expert; (i–l) drifts β_WH − β_SL and β_WH − β_AH and slopes β_WH, β_SL per session "
                                       f"with day 0 vs expert per cohort, expert R+ vs R− and the cohort × stage interaction.")
    return C


# ---------------------------------------------------------------------------------------------------------- sections
def front(T):
    tot = T[["Sessions", "Mice", "Units"]].sum()
    a = MM.set_index(["stage", "reference"]).loc[("learning", "spontaneous licks")]
    return f"""---
title: "Pre-lick convergence: whisker-triggered licks come to resemble rewarded auditory licks, across and within days{' (stable units)' if STABLE else ''}"
author: "Axel Bisi (analysis with Claude Code)"
date: "{date.today().isoformat()}"
---

**Data version.** Kilosort 4 (NWB_ks4), v2 unit table, {UNIT_TXT} with mean pre-lick rate ≥ {m51.MIN_FR} Hz;
SSL whisker-training sessions, R+ and R− cohorts (cohort per mouse from the reference sheet), learning day (day 0) and
expert days; learners population (non-learner mice lose only their day-0 session). Part I: {num('n_sessions', int(tot.Sessions))}
sessions, {num('n_units', int(tot.Units))} tested units ({ref('tbl-s-sizes')}, {ref('tbl-s-sizes-areas')}). Reference for
unrewarded licks: spontaneous licks (SL). Results produced with the current analysis settings; pending changes are listed in
the caveats. Every figure and its producing script: {ref('tbl-provenance')}.

# Abstract

In the SSL task, licks after a whisker stimulus are rewarded in R+ mice and unrewarded in R− mice, while licks after an
auditory tone are always rewarded. We asked whether the neural activity that precedes a whisker-triggered lick (whisker
hit, WH) comes to resemble the activity preceding a rewarded auditory lick (auditory hit, AH) rather than an unrewarded
spontaneous lick (SL), and whether this depends on the reward contingency.

Across days (Part I), in R+ but not R− mice, more neurons distinguished whisker hits from spontaneous licks, neurons
that fire more before rewarded auditory licks came to fire more before whisker hits too, and the population vector of
whisker hits moved away from spontaneous licks toward auditory hits. Decoders trained on auditory hits vs spontaneous licks
read whisker hits as more auditory-like in R+ experts.

Within days (Part II), the change was already visible during the learning session: on a single-trial coding direction,
R+ whisker hits drifted toward auditory hits by {num('mm_rplus_d0', a.drift_Rplus, '{:+.2f}')} of the SL → AH distance over
one session, R− whisker hits did not ({num('mm_rminus_d0', a.drift_Rminus, '{:+.2f}')}), and the drift continued within
expert sessions.

# Key results

1. **Single neurons.** WH vs SL selective units: {grp('2b', 'k1')} ({ref('fig-single')}b). Converging neurons among
   AH > SL neurons: {grp('S-conv conv_pos', 'k1b', S79)} ({ref('fig-s-conv-sign')}a).
2. **Population distance.** Δd = d(WH, SL) − d(WH, AH): {grp('3c', 'k2')} ({ref('fig-distance')}c).
3. **Where the change lies.** The whisker-hit vector grew rather than rotated: length ratio interaction
   {gi('S-ang ratio', SS)}, angle interaction {gi('S-ang theta', SS)} ({ref('fig-s-angles')}).
4. **Decoders.** P(AH | WH) − P(AH | SL), chance-corrected: {grp('5e', 'k4')} ({ref('fig-decoders')}e).
5. **Within days.** Mixed model of single-trial coding-direction scores ({ref('tbl-mixed')}, {ref('fig-cd')}):
   day-0 drift R+ {a.drift_Rplus:+.2f} ± {a.se_Rplus:.2f} (p {pv('mm_p_rplus_d0', a.p_Rplus)}), R− {a.drift_Rminus:+.2f} ± {a.se_Rminus:.2f}
   (p {pv('mm_p_rminus_d0', a.p_Rminus)}); R+ − R− mouse-permutation p {pv('mm_pperm_d0', a.p_perm_diff)}.
"""


def intro():
    return """
# Introduction

Mice in the SSL task lick to report a brief whisker deflection or an auditory tone. The two cohorts differ only in the
outcome of whisker-triggered licks: rewarded in R+ mice, unrewarded in R− mice. Auditory licks are rewarded in both and
spontaneous licks between trials are never rewarded. Because the motor act is the same lick, comparing the activity in the
100 ms before the first lick across these event types isolates what the brain represents about the upcoming lick beyond
the movement itself: the sensory trigger, the expected outcome, or both.

The hypothesis is that reward expectation is part of the pre-lick representation. If so, whisker-triggered licks of R+
mice, which become rewarded, should come to resemble auditory licks (rewarded), while those of R− mice should stay like
spontaneous licks (unrewarded). The report tests this at the level of single neurons, population geometry and decoders
(Part I, learning day vs expert sessions), and then asks when the change happens: already within the learning session or
only between days, and whether it continues within expert sessions (Part II).
"""



def methods():
    return f"""
# Methods

## Data, events and units

**Sessions and cohorts.** All whisker-training sessions with KS4 NWB files; learning = day 0, expert = later days. Main
population: learners (a mouse whose learning-day category is not "learner" contributes only its expert sessions); every
population-specific figure is repeated for all mice in the supplement. Sample sizes per cohort, stage and area group:
{ref('tbl-s-sizes')}, {ref('tbl-s-sizes-areas')}. Every figure with its producing script: {ref('tbl-provenance')}.

**Trials.** Active context (per-trial rule: a trial in a fixed ~3 s inter-trial sequence is passive), perf ≠ 6, auditory
warm-up cut (one trial before the first whisker trial kept), disengagement tail trimmed (rule A1: trials after the last lick
dropped when that tail holds ≥ 5 whisker and ≥ 1 auditory trials).

**Events.** Whisker hit (WH) and auditory hit (AH): first lick of a licked whisker / auditory trial. The NWB lick time is
late by the artefact window, so the corrected first lick is
{E('eq-lick')}
where $t_{{\\mathrm{{start}}}}$ is the trial start and $t_{{\\mathrm{{rw,start}}}}$ the response-window start. Spontaneous lick
(SL): a piezo lick ≥ 1 s after the previous piezo lick (bout onset), outside every trial window [start − 0.2 s,
response-window end + 0.5 s], within the analysed epoch.

**Rates.** Pre-lick rate of unit $u$ for event $i$: spikes in $[t_i - 100\\,\\mathrm{{ms}}, t_i)$ / 0.1 s, with $t_i$ from
{ref('eq-lick')} for hits, minus a baseline (hits: [−1.0, −0.015] s before trial start; SL: [−1.0, −0.5] s before the lick).
Spike trains are whisker-artefact corrected. Units: {UNIT_TXT}, mean raw pre-lick rate ≥ {m51.MIN_FR} Hz over the
analysed events.

## Single-neuron selectivity

For two event classes A and B, the ROC area under the curve of a unit's pre-lick rates and its selectivity are
{E('eq-auc')}
(positive: B higher). Significance: 1,000 label permutations, one-tailed on the side of the observed $s$, p < 0.05.
Comparisons: WH vs SL, AH vs SL, WH vs AH.

- **Fraction selective**: units with significant $s$ ({ref('eq-auc')}) / tested units per session (sessions with ≥ 10
  tested units).
- **Reward-lick neurons**: AH vs SL significant; split by sign into AH > SL and AH < SL ({ref('fig-s-conv-sign')}).
- **Converging neurons**: among reward-lick neurons, the fraction whose WH vs SL selectivity is also significant with the
  same sign ({ref('fig-single')}d: both signs pooled; {ref('fig-s-conv-sign')}a: AH > SL only).
- **AH-likeness** of a reward-lick neuron with class means $m_W, m_A, m_S$ is
{E('eq-likeness', '  ')}
  bounded by the triangle inequality ($-1$: WH like SL, $+1$: WH like AH).
- **Shared hit code**: Spearman correlation across a session's units of $s_{{AH\\,vs\\,SL}}$ and $s_{{WH\\,vs\\,SL}}$; pooled
  over neurons, Pearson $r$ per stage with a session-bootstrap test of $\\Delta r$ ({ref('fig-single')}h–i).
- **Absolute selectivity** ({ref('fig-s-abssel')}): mean $|s|$ over all tested units, the fraction significant with $s > 0$
  and with $s < 0$, and mean $|s|$ over significant units.

## Population geometry

**Unit scaling.** Per session, each unit's event rates are z-scored over the analysed events (mean and SD over all events
pooled). {ref('fig-s-norm')} compares this with raw spikes/s and with the within-class SD
{E('eq-sigma')}
(diagonal noise normalisation), the scaling decided for the next iteration.

**Cross-validated squared distance per unit.** For classes X and Y, the events of each class are split at random into
halves a and b ($n_s = 50$ splits), and
{E('eq-cvd')}
with $N$ the number of units and $\\bar x_a$ the mean of class X in half a. The noise of halves a and b is independent, so
the expectation of {ref('eq-cvd')} and that of the plug-in squared distance are
{E('eq-bias')}
where $\\Sigma$ is the noise covariance and $n_X, n_Y$ the event counts: the cross-validated estimate removes the positive
noise bias of the plug-in one, which would otherwise depend on the number of events. Single estimates can be negative.
With z-scored units, $d$ is the mean squared effect size per unit. Distance difference: $\\Delta d = d(WH, SL) - d(WH, AH)$
(> 0: WH nearer AH).

**Along and orthogonal components.** With the reference $S$ at the origin, the polarisation identity gives the
cross-validated dot product ({ref('fig-distance')}i) and the squared orthogonal part ({ref('fig-distance')}j):
{E('eq-along')}
both from the distances of {ref('eq-cvd')}. The group triangles ({ref('fig-distance')}b, real group-mean distances) place SL
at (0, 0), AH at $(\\sqrt{{d(A,S)}}, 0)$ and WH at
{E('eq-triangle')}

**λ.** The position of the whisker-hit mean on the SL → AH axis is
{E('eq-lambda')}
estimated with the same split halves (numerator and denominator each cross-validated, $a$ from {ref('eq-along')}):
$\\lambda = 0$, WH like SL; $\\lambda = 1$, like AH; along the axis WH ≈ $(1-\\lambda)\\,S + \\lambda\\,A$. Sessions need
$d(A,S) \\ge 0.01$; λ clipped to [−1, 2].

**Angle and length** ({ref('fig-s-angles')}). λ ({ref('eq-lambda')}) factorises into an angle and a length ratio:
{E('eq-angle')}
with θ the angle between WH − SL and AH − SL. The same λ can arise from a long vector at a wide angle or a short one at a
narrow angle, and λ > 1 is possible when $\\rho\\cos\\theta > 1$; the factors are reported separately.

**λ_LDA and axis quality** ({ref('fig-s-lambda')}). A shrinkage LDA (Ledoit–Wolf) separates AH from SL on training folds
(5-fold, units z-scored on training folds); held-out AH and SL and all WH events are projected and normalised so that the
training SL mean is 0 and the AH mean 1; λ_LDA is the mean WH projection. LDA weights each unit by the inverse within-class
covariance, so it is the noise-whitened counterpart of λ. Axis quality is the held-out
{E('eq-dprime')}
of the held-out projections $p$; sessions with $d' \\ge 0.3$. The second axis of the example planes is the WH off-axis
direction, defined on half the WH events and shown for the other half.

**Reward-lick coding direction (CD) and trial scores** (Part II; {ref('fig-s-cd')}). (i) 5-fold CD, used by all Part II
figures: on the training folds of AH and SL events, $w = (\\bar x_{{A}} - \\bar x_{{S}}) / \\|\\bar x_{{A}} - \\bar x_{{S}}\\|$ (units
z-scored on the training folds); held-out AH / SL events and every WH event (fold-averaged) are projected, and one
normalisation per session gives the trial score
{E('eq-cd')}
with sessions kept if the held-out $d'$ of {ref('eq-dprime')} is ≥ 0.3. (ii) Unified split-half score: in each of the 50 λ
splits, the axis $w = \\bar a_A - \\bar a_S$ comes from one half and every event of the other half is scored
$(x_i - \\bar b_S)\\cdot w$; scores are averaged over the splits in which the event was held out and divided by
$D = \\langle (\\bar b_A - \\bar b_S)\\cdot w \\rangle$. The mean WH score of (ii) equals λ (same numerator and denominator).
Per area group (Part II), the same CD is built from the units of one area group (≥ 5 units).

**Linear-shift chance** ({ref('fig-s-shift')}). Event labels in time order are shifted against the activity by $k$ events
($|k| = 5 \\dots n/3$, 40 shifts, no wrap-around), which keeps the temporal autocorrelation of event types and of neural
drift but breaks the event–activity pairing. For a quantity $q$,
{E('eq-shift')}
and for ratios (λ, transfers) numerator and denominator are corrected separately before the ratio.

**Quiet windows** ({ref('fig-s-quiet')}). Raw rates in [−200, −100] ms before every trial start of the analysed epoch,
lick-free (no lick in the preceding 1 s), scaled with the event parameters and placed on the same readouts.

## Decoders

L2 logistic regression ($C = 0.05$, balanced classes, units z-scored on training folds), trained on AH vs SL with stratified
$k$-fold cross-validation ($k = \\min(5, n_{{\\min}})$); WH events are never used for training and are scored by every fold's
decoder. With TPR and FPR the fractions of held-out AH and SL decoded as AH and $F_{{WH}}$ the fraction of WH decoded as AH,
the transfers are
{E('eq-transfer')}
together with $\\bar P(AH|WH) - \\bar P(AH|SL)$ and the held-out balanced accuracy. The transfers of {ref('eq-transfer')} are 0
when WH are decoded like SL and 1 when like AH; their denominators are the decoder's own separation, so they become unstable
when the decoder is weak, whereas $\\bar P(AH|WH) - \\bar P(AH|SL)$ (the numerator alone) does not divide by decoder quality
({ref('fig-s-decq')}). Chance: linear shifts ({ref('eq-shift')}). Pseudo-populations: hierarchical bootstrap (mice →
sessions → neurons → events) of $M$ neurons, same decoder and readouts ({ref('fig-decoders')}f–m).

## Statistics

Unit of analysis: session. Learning vs expert per cohort and R+ vs R− per stage: Mann-Whitney U (Welch t-test in the stats
tables). The learning × cohort interaction is
{E('eq-interaction')}
with $\\bar x$ the mean over sessions of a group, tested by 10,000 permutations of cohort labels across mice (sessions of a
mouse move together); in Part II the same statistic is the cohort × stage interaction. Correlations with behaviour:
Spearman, OLS with 95 % CI band ({ref('fig-s-perf')}). Per-area tests: Benjamini-Hochberg FDR over the area groups shown;
two-way ANOVA drift ~ cohort × area per stage and drift ~ cohort × stage × area, cohort terms tested by cohort-label
permutation across mice (a session contributes several area groups).

## Part II: within-session analyses

**Drift on the reward-lick CD (002, 011).** Every event gets its 5-fold CD score $c_i$ ({ref('eq-cd')}). Per session and
class $k$, OLS of the scores on normalised time,
{E('eq-ols')}
where $\\beta_k$ is the change of the class's mean score over the whole session in units of the SL → AH distance. All
classes drift together with slow state changes (arousal, satiety, recording drift), so the tested quantity is the drift of
whisker hits relative to a reference class,
{E('eq-drift')}
(> 0: whisker hits move toward AH relative to the reference). Relative to SL it removes drift shared by all licks; relative
to AH, drift shared by rewarded, stimulus-triggered licks. Alternative estimators (Theil-Sen slopes, slope of the binned
difference, late − early half) are compared in {ref('tbl-estimators')}. Trial by trial along the first 10 and 20 whisker hits: {ref('fig-s-wh10')}, {ref('fig-s-wh20')}.

**Halves (001, 004, 012).** Each session is split at the midpoint between its two middle auditory hits (equal AH per half);
per half the events are count-matched by subsampling (20 subsamples) and Δd, λ and decoder readouts recomputed; change =
late − early. Null split: odd vs even events ({ref('fig-s-oddeven')}); controls per half in {ref('fig-s-controls')}.

**Epochs on a common footing (003, 013).** Day-0 and expert halves each estimated from a fixed number of events per class (6;
variants with 4: {ref('fig-s-epochs-n4')}) drawn 20 times, and a fixed number of units (150; other unit samplings are
compared in the decoder-scheme figure); one decoder per session read out per half; hierarchical bootstrap over mice then
sessions; with $D$ a readout of a half, the contrasts are
{E('eq-epochs')}
($L$: learning day, $E$: expert). The decoder schemes are compared in {ref('fig-s-decschemes')}.

**Mixed model (009).** On single events,
{E('eq-mixed')}
[+ (1 | mouse) for experts]; the WH × τ coefficient is the drift of {ref('eq-drift')} estimated on single trials (events
weighted, sessions shrunk), in units of the SL → AH distance; R+ vs R− also by 500 cohort-label permutations across mice.

**Cohort specificity (014) and day-0 effects vs expert outcome (016).** For 30 session-level within-day measures, R+ vs R− on
day 0 and in experts (MWU, Welch, mouse permutation) and the cohort × stage interaction ({ref('eq-interaction')}); a measure is
called learning-day specific when the day-0 p < 0.05, the expert p ≥ 0.05 and the interaction p < 0.05. Per mouse (mice with
both stages), day-0 drifts and behaviour against expert behaviour and expert WH score (Spearman, Pearson, partial Spearman
given the day-0 whisker hit rate).
"""


def part1():
    r3S = row(ST, panel="3S i")
    nsn = NS[(NS.population == POP) & (NS.metric == "dd")].set_index("norm")
    sp = lambda k: num(f"norm_rho_{k}", nsn.loc[f"spearman pooled vs {k}", "interaction"], "{:.2f}")
    th = row(SS, panel="S-ang theta")
    dq = lambda m, c: row(S79, panel="S-decq", measure=f"{m} vs bacc", cohort=c)
    return f"""
# Part I: across days

The task, the alignment and the data are summarised in {ref('fig-task')}.

{figure('fig-task', FILE['fig-task'], CAP['fig-task'])}

## Single neurons

More neurons separated whisker hits from spontaneous licks in R+ experts than on the learning day, and not in R−:
{grp('2b', 'fracWH')} ({ref('fig-single')}b). The fraction separating whisker hits from auditory hits changed little:
{grp('2c', 'fracWHAH')} ({ref('fig-single')}c).

Among reward-lick neurons of either sign, the fraction that also separated whisker hits from spontaneous licks with the
same sign (converging neurons): {grp('2d', 'conv')} ({ref('fig-single')}d); their AH-likeness ({ref('eq-likeness')}): {grp('2e', 'like')}
({ref('fig-single')}e).

Restricting reward-lick neurons to those firing more before auditory hits than before spontaneous licks (AH > SL) makes
the effect clearer: converging fraction {grp('S-conv conv_pos', 'convpos', S79)}; AH-likeness {grp('S-conv like_pos', 'likepos', S79)}
({ref('fig-s-conv-sign')}a, d). Neurons with AH < SL showed no cohort-specific change (converging interaction
{gi('S-conv conv_neg', S79)}, AH-likeness {gi('S-conv like_neg', S79)}; {ref('fig-s-conv-sign')}b, e): the convergence is
carried by neurons that are excited before rewarded licks.

The shared hit code, the correlation across units of AH vs SL and WH vs SL selectivities, rose in R+:
{grp('2g', 'shared')} ({ref('fig-single')}g); pooled over neurons the correlation increased in R+ and not in R−
({ref('fig-single')}h–i). Converging neurons were found across areas ({ref('fig-single')}j); examples, functional types and
anatomical distribution: {ref('fig-s-converging')}.

The increase in WH vs SL selective units in R+ is carried by positively selective units (WH > SL; interaction
{gi('S-sel pos whisker_hit_vs_fa_prelick', SS)}); mean absolute selectivity over all units changed less
({ref('fig-s-abssel')}).

{figure('fig-single', FILE['fig-single'], CAP['fig-single'])}

The convergence is not a property of stimulus-responsive neurons only ({ref('fig-classes')}). Classifying neurons by the
stimulus-aligned ROC analyses (sensory: responsive to the whisker or auditory stimulus in the active context; lick-gated:
whisker hit vs miss; non-sensory: neither), the converging fraction among reward-lick neurons roughly doubled in R+ in
every class and did not change in R−, including neurons that respond to neither stimulus ({ref('fig-s-classes')}). In R+
experts the whisker-hit trace of converging neurons reaches the auditory-hit trace before the lick in all three classes.

Single neurons also mirror the population geometry ({ref('fig-s-sug')}, {ref('fig-s-sug-d')}): across units, the
selectivity for WH vs SL against AH vs SL becomes more aligned in both cohorts, while its spread (the counterpart of the
length ratio ρ) grows in R+ only; converging neurons carry most of the cross product and their number, not their
individual contribution, increases.

{figure('fig-classes', FILE['fig-classes'], CAP['fig-classes'])}

## Population geometry

In R+, whisker hits moved away from spontaneous licks and toward auditory hits: Δd, from the cross-validated distances of
{ref('eq-cvd')} (free of the noise bias of {ref('eq-bias')}), {grp('3c', 'dd')} ({ref('fig-distance')}c). The group-mean
geometry places whisker hits with {ref('eq-triangle')} ({ref('fig-distance')}b). The change came from d(WH, SL), which increased in R+ ({grp('3d', 'dws')}), while d(WH, AH)
({gi('3e')}) and the axis length d(AH, SL) ({gi('3f')}) did not change differently between cohorts ({ref('fig-distance')}d–f).

The whisker-hit displacement grew both along the SL → AH direction ({grp('3i', 'along')}) and orthogonal to it
({grp('3j', 'ortho')}) ({ref('fig-distance')}i–j). The along component divided by the axis length is λ, which increased in
R+ only ({grp('3S d', 'lam')}; {ref('fig-s-lambda')}); λ and λ_LDA agreed (Spearman r = {num('lam_vs_lda', r3S.r, '{:.2f}')},
n = {int(r3S.n)} sessions).

Decomposing λ = ρ cos θ ({ref('eq-angle')}, {ref('fig-s-angles')}): the angle θ between the whisker-hit and auditory-hit displacements was
{num('theta_rp_l', th['mean_R+ learning'], '{:.0f}')}° → {num('theta_rp_e', th['mean_R+ expert'], '{:.0f}')}° in R+ and
{num('theta_rm_l', th['mean_R− learning'], '{:.0f}')}° → {num('theta_rm_e', th['mean_R− expert'], '{:.0f}')}° in R−
(interaction {gi('S-ang theta', SS)}), while the length ratio changed in R+ only ({grp('S-ang ratio', 'ratio', SS)}): in R+ the
whisker-hit vector grew, at a roughly constant angle.

These results did not depend on the unit scaling (session values of Δd vs the current scaling: within-class SD of {ref('eq-sigma')} ρ =
{sp('within')}, raw rates ρ = {sp('raw')}; {ref('fig-s-norm')}). The linear-shift correction left them unchanged: Δd
observed {grp('S-shift dd', 'shdd', S78)}; corrected {grp('S-shift dd_corrected', 'shddc', S78)}; λ corrected
{grp('S-shift lam_corrected', 'shlamc', S78)} ({ref('fig-s-shift')}). The two coding-direction estimators agreed with λ
({ref('fig-s-cd')}). Lick-free quiet windows sat near the spontaneous-lick end of the axis and did not show the whisker-hit
change ({ref('fig-s-quiet')}).

{figure('fig-distance', FILE['fig-distance'], CAP['fig-distance'])}

## Brain areas

Δd computed per session × area group changed in R+ mainly in motor, frontal, retrosplenial, insular and hippocampal groups,
but no area group survived family-wise correction ({ref('fig-areas')}a–c); lick-aligned PSTHs of two groups illustrate the
three event types per neuron and per session ({ref('fig-areas')}d–e).

{figure('fig-areas', FILE['fig-areas'], CAP['fig-areas'])}

## Decoders

Decoders separated auditory hits from spontaneous licks above chance in every group ({ref('fig-decoders')}a–b; raw accuracy
{grp('5a', 'bacc_raw')}). Applied to whisker hits, they read R+ expert whisker hits as more auditory-like: yes/no transfer
{grp('5c', 'tb')}; P(AH|WH) − P(AH|SL) {grp('5e', 'num')} ({ref('fig-decoders')}c–e). Pseudo-populations of up to 2,000
neurons gave the same direction ({ref('fig-decoders')}f–m).

Decoder quality did not drive the result ({ref('fig-s-decq')}). Across sessions, P(AH|WH) − P(AH|SL) was not correlated with
raw accuracy (R+ ρ = {num('dq_num_rp', dq('num_prob_corrected', 'R+').rho, '{:.2f}')}, p {pv('dq_num_rp_p', dq('num_prob_corrected', 'R+').p)};
R− ρ = {num('dq_num_rm', dq('num_prob_corrected', 'R-').rho, '{:.2f}')}, p {pv('dq_num_rm_p', dq('num_prob_corrected', 'R-').p)}), and
R+ experts exceeded the other groups within every accuracy tertile. The transfers, which divide by the decoder's own
separation, were largest in low-accuracy sessions; λ was weakly anti-correlated with accuracy in R+
(ρ = {num('dq_lam_rp', dq('lam', 'R+').rho, '{:.2f}')}, p {pv('dq_lam_rp_p', dq('lam', 'R+').p)}), as expected when a short
AH − SL axis inflates a ratio.

## Behaviour

Across sessions, most single-neuron and geometry measures tracked whisker hit rate and d′, strongly in R+ ({ref('fig-s-perf')}).
The across-days COSYNE figure ({ref('fig-cosyne-across')}) assembles the single-neuron and geometry results on one page:
converging neurons and AH-likeness, group geometry and Δd, the length / angle decomposition with bootstrap CIs (the
length grows in R+ only, no detectable rotation) and Δd per area group (FDR q per area; the cohort × stage × area term of
the ANOVA tests whether the effect differs between areas).

{figure('fig-decoders', FILE['fig-decoders'], CAP['fig-decoders'])}

{figure('fig-cosyne-across', FILE['fig-cosyne-across'], CAP['fig-cosyne-across'])}
"""


def mm_table():
    T = MM
    return pd.DataFrame({
        "Stage": T.stage, "Reference": T.reference, "Sessions / mice": T.n_sessions.astype(str) + " / " + T.n_mice.astype(str),
        "WH events": T.n_wh, "R+ drift": [f"{a:+.2f} ± {b:.2f}" for a, b in zip(T.drift_Rplus, T.se_Rplus)], "p R+": T.p_Rplus.map(pp),
        "R− drift": [f"{a:+.2f} ± {b:.2f}" for a, b in zip(T.drift_Rminus, T.se_Rminus)], "p R−": T.p_Rminus.map(pp),
        "R+ − R−": [f"{a:+.2f} ± {b:.2f}" for a, b in zip(T.diff_Rplus_minus_Rminus, T.se_diff)], "p perm": T.p_perm_diff.map(pp)})


def halves_table():
    lab_ = {"dd": "Δd", "lam": "λ", "ws_num_c": "Decoder P(AH|WH) − P(AH|SL), whole session", "xh_num_c": "Decoder, cross-half",
            "ws_bacc_c": "Accuracy − chance, whole session"}
    rows = []
    for m, l in lab_.items():
        r = row(HV, split="time", measure=m)
        d = {"Measure": l}
        for g, gl in (("R+ learning", "R+ day 0"), ("R+ expert", "R+ expert"), ("R- learning", "R− day 0"), ("R- expert", "R− expert")):
            d[gl] = f"{r[f'mean Δ {g}']:+.3f} (p {pp(r[f'p_wilcoxon {g}'])})"
        d["Day 0 R+ vs R− (perm)"] = pp(r["day0 p_perm"]); d["Cohort × stage (perm)"] = pp(r["cxsxh p_perm"])
        rows.append(d)
    return pd.DataFrame(rows)


def slopes_table():
    lab_ = {"md_WH_slope": "WH score slope", "md_WH-FA_slope": "WH − SL slope", "md_WH-AH_slope": "WH − AH slope",
            "dec_WH_slope": "Decoder P(AH|WH) slope", "dec_WH-FA_slope": "Decoder WH − SL slope"}
    rows = []
    for m, l in lab_.items():
        r = row(SL, measure=m)
        d = {"Measure": l}
        for g, gl in (("R+ learning", "R+ day 0"), ("R+ expert", "R+ expert"), ("R- learning", "R− day 0"), ("R- expert", "R− expert")):
            d[gl] = f"{r[f'mean {g}']:+.3f} (p {pp(r[f'p_wilcoxon {g}'])})"
        d["Day 0 R+ vs R− (perm)"] = pp(r["day0 p_perm"]); d["Cohort × stage (perm)"] = pp(r["cxs p_perm"])
        rows.append(d)
    return pd.DataFrame(rows)


def epochs_table():
    lab_ = {"pos": "WH − SL position on the CD", "dec": "Decoder P(AH|WH) − P(AH|SL) − chance", "ddn": "Δd / d(AH, SL)"}
    rows = []
    for m, l in lab_.items():
        for c in ("within-day", "across-day (early)", "carry-over"):
            d = {"Measure": l, "Contrast": c}
            for coh in ("R+", "R-"):
                r = row(EP, measure=m, cohort=coh, kind="contrast", name=c)
                d[coh.replace("-", "−")] = f"{r.value:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
            d["R+ vs R− (perm)"] = pp(row(EP, measure=m, cohort="R+ - R-", kind="contrast", name=c).p_perm)
            rows.append(d)
    return pd.DataFrame(rows)



def estimators_table():
    lab_ = {"ols": "OLS slopes (main)", "theil": "Theil-Sen slopes", "binned": "Slope of the binned difference",
            "half": "Late − early half"}
    rows = []
    for _, r in DE.iterrows():
        rows.append({"Reference": r.reference, "Estimator": lab_.get(r.estimator, r.estimator),
                     "R+ mean": f"{r.mean_Rplus:+.2f} (n = {int(r.n_Rplus)})", "R− mean": f"{r.mean_Rminus:+.2f} (n = {int(r.n_Rminus)})",
                     "Hedges g": f"{r.hedges_g:.2f}", "p perm": pp(r.p_perm), "p MWU": pp(r.p_MWU), "p Welch": pp(r.p_Welch),
                     "ρ with OLS": f"{r.rho_with_ols:.2f}"})
    return pd.DataFrame(rows)


def area_anova_table():
    rows = []
    for _, r in AN.iterrows():
        if str(r.get("model", "")) == "cohort x stage x area":
            rows.append({"Drift": r.measure.replace("drift_WH_", "WH − "), "Model": "cohort × stage × area (both stages)",
                         "Cohort": pp(r["cohort p_perm"]), "Cohort × area": pp(r["cohort x area p_perm"]),
                         "Cohort × stage": pp(r["cohort x stage p_perm"]), "Cohort × stage × area": pp(r["cohort x stage x area p_perm"])})
        elif str(r.strict) == "True":
            rows.append({"Drift": r.measure.replace("drift_WH_", "WH − "),
                         "Model": f"cohort × area ({'day 0' if r.stage == 'learning' else 'expert'})",
                         "Cohort": pp(r["cohort p_perm"]), "Cohort × area": pp(r["cohort x area p_perm"]),
                         "Cohort × stage": "–", "Cohort × stage × area": "–"})
    return pd.DataFrame(rows)


def area_contrasts_table():
    rows = []
    for _, r in AC[AC.measure.isin(["drift_WH_SL", "drift_WH_AH"])].iterrows():
        rows.append({"Area group": r.area_group, "Drift": r.measure.replace("drift_WH_", "WH − "),
                     "R+ D0 / E": f"{r['mean R+ learning']:+.2f} / {r['mean R+ expert']:+.2f}",
                     "R− D0 / E": f"{r['mean R- learning']:+.2f} / {r['mean R- expert']:+.2f}",
                     "D0 R+ vs R−": pp(r["learning p_perm"]), "E R+ vs R−": pp(r["expert p_perm"]),
                     "Cohort × stage": pp(r["cxs p_perm"])})
    return pd.DataFrame(rows)


def part2():
    mm = MM.set_index(["stage", "reference"])
    a = mm.loc[("learning", "spontaneous licks")]
    hv = row(HV, split="time", measure="dd")
    s1 = row(SL, measure="md_WH-FA_slope")
    w = lambda coh, name: row(EP, measure="pos", cohort=coh, kind="contrast", name=name)
    dsl, dah = row(C11, measure="drift_WH_SL"), row(C11, measure="drift_WH_AH")
    bwh, bsl, bah = row(C11, measure="beta_WH"), row(C11, measure="beta_FA"), row(C11, measure="beta_AH")
    c = lambda r, g, k: num(f"c11|{r.measure}|{g}", r[f"mean {g}"], "{:+.2f}")
    chp = row(CH, panel="abs_change", measure="pos")
    an = lambda st, m: row(AN, stage=st, measure=m, strict="True")
    an3 = lambda m: row(AN, stage="both", measure=m)
    sig_areas = lambda m: ", ".join(A17[(A17.stage == "learning") & (A17.measure == m) & (A17.q_BH < 0.05)].area_group)
    n_meas = len(SPEC)
    d0 = SPEC["learning p_perm"] < 0.05
    ex = SPEC["expert p_perm"] < 0.05
    spec_list = ", ".join(SPEC[SPEC.call == "learning-day specific"].label) or "none"
    nm = M16.groupby("cohort").size() if len(M16) else pd.Series(dtype=int)
    best = D16.loc[D16.spearman_p.idxmin()] if len(D16) else None
    de_th = row(DE, reference="SL", estimator="theil")
    de_h = row(DE, reference="SL", estimator="half")
    return f"""
# Part II: within days

The across-day change raises the question of when it happens. Each session was analysed along its own time axis, on the
learning day and in expert sessions. All Part II analyses use the reward-lick coding direction (CD): the cross-validated
direction separating rewarded auditory licks from unrewarded spontaneous licks, on which every event gets a score
({ref('eq-cd')}; SL = 0, AH = 1). Each section below is one analysis with its own figure; the summary figures (COSYNE
versions) are collected at the end of Part II.

## Reward-lick coding direction within sessions

On the learning day, whisker hits drifted toward auditory hits relative to spontaneous licks in R+ mice and away from
them in R− mice ({ref('fig-cd')}e, f, i): $D_{{SL}}$ ({ref('eq-drift')}) was {c(dsl, 'R+ learning', 0)} in R+ (Wilcoxon p
{pv('c11_dsl_rp_p', dsl['p_wilcoxon R+ learning'])}, n = {int(dsl['n R+ learning'])} sessions) and {c(dsl, 'R- learning', 0)} in R−
(p {pv('c11_dsl_rm_p', dsl['p_wilcoxon R- learning'])}, n = {int(dsl['n R- learning'])}); R+ vs R− mouse permutation p
{pv('c11_dsl_d0', dsl['learning p_perm'])} (MWU p {P(dsl['learning R+ vs R- p_MWU'])}, Welch p {P(dsl['learning R+ vs R- p_Welch'])}).
Relative to auditory hits ({ref('fig-cd')}g, h, j), $D_{{AH}}$ was {c(dah, 'R+ learning', 0)} in R+ and {c(dah, 'R- learning', 0)}
in R− (R+ vs R− p {pv('c11_dah_d0', dah['learning p_perm'])}).

In expert sessions the same cohort difference was present (D_SL: R+ {c(dsl, 'R+ expert', 0)}, R− {c(dsl, 'R- expert', 0)},
p {pv('c11_dsl_e', dsl['expert p_perm'])}; D_AH: R+ {c(dah, 'R+ expert', 0)}, R− {c(dah, 'R- expert', 0)}, p
{pv('c11_dah_e', dah['expert p_perm'])}), and the cohort × stage interaction ({ref('eq-interaction')}) was not significant
(D_SL p {pv('c11_dsl_x', dsl['cxs p_perm'])}, D_AH p {pv('c11_dah_x', dah['cxs p_perm'])}): the drift is a property of the
contingency that is already present on the learning day, not specific to it.

The class slopes show what moves ({ref('fig-cd')}a–d, k–m). Spontaneous-lick scores rose within sessions in every group
(β_SL day 0: R+ {c(bsl, 'R+ learning', 0)}, R− {c(bsl, 'R- learning', 0)}), which is why the drift is taken relative to a
reference class. In R+ the whisker-hit score itself rose (β_WH {c(bwh, 'R+ learning', 0)}); in R− it stayed flat
(β_WH {c(bwh, 'R- learning', 0)}, p {pv('c11_bwh_rm_p', bwh['p_wilcoxon R- learning'])}) while the auditory-hit score rose
(β_AH {c(bah, 'R- learning', 0)}): R− whisker hits did not move away from auditory hits; auditory hits moved away from them.

The estimators of the drift agree ({ref('tbl-estimators')}): Theil-Sen slopes, robust to outlying trials, gave the largest
effect (relative to SL, Hedges g = {num('de_th_g', de_th.hedges_g, '{:.2f}')}), and the late − early half split, which assumes no
linear trend, the smallest (g = {num('de_h_g', de_h.hedges_g, '{:.2f}')}). The single-trial mixed model ({ref('eq-mixed')},
{ref('tbl-mixed')}; all mice) weighs trials rather than sessions: relative to auditory hits it agrees, relative to spontaneous
licks the R− drift is close to zero on the learning day ({a.drift_Rminus:+.2f} ± {a.se_Rminus:.2f}; R+ − R− permutation p
{P(a.p_perm_diff)}), so the R− drift relative to SL is partly carried by sessions with few events.

{figure('fig-cd', FILE['fig-cd'], CAP['fig-cd'])}

{table('tbl-estimators', estimators_table(), f"Drift of whisker hits relative to the reference ({ref('eq-drift')}) on the learning day ({POP}) with four estimators: OLS slopes of {ref('eq-ols')} (main), Theil-Sen slopes, the OLS slope of the per-bin difference (five bins of session time, each part of the session weighted equally) and the late − early half difference. R+ vs R−: Hedges g, mouse-level permutation, Mann-Whitney U and Welch p; ρ: Spearman correlation with the OLS estimate across sessions.")}

{table('tbl-mixed', mm_table(), "Single-trial mixed model of reward-lick CD scores (Eq. as cited in the text): WH × time coefficient = drift of whisker hits relative to the reference over one session (normalised time 0 → 1), estimate ± s.e. and Wald p per cohort; R+ − R− with cohort-label permutation across mice (500 permutations). All mice; learning day and expert sessions; reference: spontaneous licks or auditory hits.")}

## Session halves

Splitting each session at its middle auditory hit, Δd increased from the early to the late half on the learning day in R+
(mean Δ {num('hv_dd_rplus', hv['mean Δ R+ learning'], '{:+.3f}')}, Wilcoxon p {pv('hv_dd_p_rplus', hv['p_wilcoxon R+ learning'])})
and not in R− (day-0 R+ vs R− permutation p {pv('hv_dd_perm', hv['day0 p_perm'])}); the same held for λ and the
whole-session decoder readout ({ref('tbl-halves')}, {ref('fig-halves')}). In expert sessions the halves changed less. The
odd/even split, which removes time, shows no change ({ref('fig-s-oddeven')}); event rates, reaction times and firing rates
per half are in {ref('fig-s-controls')}. The first-lick-aligned activity per half, for all four groups, is in
{ref('fig-psth-halves')}: in R+ the whisker-hit trace separates from the spontaneous-lick trace before the lick at both
stages, in R− it overlaps it.

{figure('fig-halves', FILE['fig-halves'], CAP['fig-halves'])}

{table('tbl-halves', halves_table(), f"Late − early change per group (mean; Wilcoxon p vs 0), time split, {POP}; day 0 R+ vs R− and cohort × stage by mouse-level permutation.")}

{figure('fig-psth-halves', FILE['fig-psth-halves'], CAP['fig-psth-halves'])}

## Trial-level trajectories

On the reward-lick CD, the WH − SL slope was positive on day 0 in R+ and differed from R− (day-0 permutation p
{pv('sl_perm', s1['day0 p_perm'])}; {ref('tbl-s-slopes')}, {ref('fig-slopes')}e–h). Fitted at the start of the learning
session, R+ whisker hits were far from their expert position; by the end they approached it ({ref('fig-slopes')}i). Spontaneous
lick rates fell during sessions while whisker hit rates rose in R+ ({ref('fig-slopes')}k–l), which is why the slopes are taken
relative to spontaneous licks measured at the same times.

{figure('fig-slopes', FILE['fig-slopes'], CAP['fig-slopes'])}

## Size of the within-session change

With R+ and R− shown at equal weight ({ref('fig-change')}), the within-session change was not larger on the learning day than
in expert sessions: position |late − early|, R− day 0 {num('ch_pos_rm_d0', chp['R- day0 mean'], '{:.2f}')} vs expert
{num('ch_pos_rm_e', chp['R- expert mean'], '{:.2f}')} (hierarchical bootstrap p {pv('ch_pos_rm_p', chp['R- day0 vs expert p_boot'])}),
R+ {num('ch_pos_rp_d0', chp['R+ day0 mean'], '{:.2f}')} vs {num('ch_pos_rp_e', chp['R+ expert mean'], '{:.2f}')}
(p {pv('ch_pos_rp_p', chp['R+ day0 vs expert p_boot'])}). Whisker-hit rates rose across R+ sessions and fell across R−
sessions at both stages ({ref('fig-change')}a).

{figure('fig-change', FILE['fig-change'], CAP['fig-change'])}

## Within days vs across days

On a common footing (same events per class and units in every half), the across-day change in R+ was larger than the
within-day change (WH − SL position, contrasts of {ref('eq-epochs')}: within day 0 {num('ep_w_rp', w('R+', 'within-day').value, '{:+.2f}')},
across days {num('ep_a_rp', w('R+', 'across-day (early)').value, '{:+.2f}')}), and part of it was already present at the start
of expert sessions (carry-over {num('ep_c_rp', w('R+', 'carry-over').value, '{:+.2f}')}) ({ref('tbl-epochs')}, {ref('fig-epochs')}).

{figure('fig-epochs', FILE['fig-epochs'], CAP['fig-epochs'])}

{table('tbl-epochs', epochs_table(), f"Epoch contrasts on a common footing ({POP}; 6 events per class, 150 units): estimate and 95 % hierarchical-bootstrap CI per cohort; R+ vs R− by cohort-label permutation across mice.")}

## Brain areas

With the CD built from the units of each area group, the day-0 cohort difference was found across the brain
({ref('fig-area-day0')}): ANOVA drift ~ cohort × area, cohort p {pv('an_d0_sl_c', an('learning', 'drift_WH_SL')['cohort p_perm'])}
(D_SL) and {pv('an_d0_ah_c', an('learning', 'drift_WH_AH')['cohort p_perm'])} (D_AH), cohort × area p
{pv('an_d0_sl_x', an('learning', 'drift_WH_SL')['cohort x area p_perm'])} and {pv('an_d0_ah_x', an('learning', 'drift_WH_AH')['cohort x area p_perm'])}
({ref('tbl-area-anova')}). The cohort difference did not depend detectably on the area group; the area groups in which it was
significant after FDR correction were {sig_areas('drift_WH_SL') or 'none'} (D_SL) and {sig_areas('drift_WH_AH') or 'none'} (D_AH),
those with enough sessions. In expert sessions ({ref('fig-area-expert')}) the cohort effect across areas was weaker (cohort p
{pv('an_e_sl_c', an('expert', 'drift_WH_SL')['cohort p_perm'])} and {pv('an_e_ah_c', an('expert', 'drift_WH_AH')['cohort p_perm'])}).
Over both stages ({ref('fig-area-matrix')}), cohort × stage p {pv('an3_sl_cs', an3('drift_WH_SL')['cohort x stage p_perm'])} (D_SL)
and {pv('an3_ah_cs', an3('drift_WH_AH')['cohort x stage p_perm'])} (D_AH), cohort × stage × area p
{pv('an3_sl_csa', an3('drift_WH_SL')['cohort x stage x area p_perm'])} and {pv('an3_ah_csa', an3('drift_WH_AH')['cohort x stage x area p_perm'])}:
area-specific learning-day effects (single area groups with a significant cohort × stage interaction, {ref('tbl-s-area-contrasts')})
are not supported by the full model. One figure per area group, day 0 vs expert: {", ".join(f"{ref(f'fig-s-pa-{i + 1:02d}')} ({area_of(f)})" for i, f in enumerate(PER_AREA))};
R+ and R− contrasted within each stage (whole brain): {ref('fig-s-cd-by-stage')}; all drifts, whole brain and per area
group, in the compact format: {ref('fig-s-drift-compact')}. Along the first 10 and 20 whisker hits instead of session time: {ref('fig-s-wh10')},
{ref('fig-s-wh20')}.

{figure('fig-area-day0', FILE['fig-area-day0'], CAP['fig-area-day0'])}

{figure('fig-area-expert', FILE['fig-area-expert'], CAP['fig-area-expert'])}

{figure('fig-area-matrix', FILE['fig-area-matrix'], CAP['fig-area-matrix'])}

{table('tbl-area-anova', area_anova_table(), "ANOVA of the per-area drifts (session × area group values; area groups with ≥ 3 sessions in every cohort × stage condition): p of the cohort terms by cohort-label permutation across mice (1,000), per stage (cohort × area) and over both stages (cohort × stage × area).")}

## Which cohort differences are specific to the learning day?

Of {num('spec_n', n_meas)} within-day measures ({ref('fig-specificity')}), {num('spec_d0', int(d0.sum()))} differed between R+ and R−
on the learning day (mouse permutation p < 0.05), {num('spec_both', int((d0 & ex).sum()))} of them also in expert sessions.
Measures meeting the learning-day-specific criterion (day-0 p < 0.05, expert p ≥ 0.05, interaction p < 0.05): {spec_list}.
Expert sessions are fewer than day-0 sessions, so "not significant in experts" is weak evidence; the interaction is the
relevant test.

{figure('fig-specificity', FILE['fig-specificity'], CAP['fig-specificity'])}

## Day-0 effects and expert outcome

Per mouse ({num('m16_rp', int(nm.get('R+', 0)))} R+ and {num('m16_rm', int(nm.get('R-', 0)))} R− mice with both stages;
{ref('fig-day0-expert')}), day-0 drifts did not reliably predict expert performance; the smallest p was
{best.x} vs {best.y} in {best.cohort} (Spearman ρ = {num('m16_best_rho', best.spearman_rho, '{:.2f}')}, p
{pv('m16_best_p', best.spearman_p)}, uncorrected over {len(D16)} tests). The samples are too small to conclude.

{figure('fig-day0-expert', FILE['fig-day0-expert'], CAP['fig-day0-expert'])}

## Summary figures

The learning-day COSYNE figure ({ref('fig-cosyne-day0')}) combines the class scores, example sessions, the drift
relative to each reference and its area breakdown.

{figure('fig-cosyne-day0', FILE['fig-cosyne-day0'], CAP['fig-cosyne-day0'])}

"""


def discussion():
    return """
# Discussion

Across days, the pre-lick activity of whisker-triggered licks in R+ mice moved away from that of unrewarded spontaneous
licks, partly toward that of rewarded auditory licks, at the level of single neurons, population distances and decoders. R−
mice, whose whisker licks are not rewarded, did not show this change. The shared lick movement does not explain the
difference: all three event types are first licks, compared in the same 100-ms window.

At the single-neuron level, the convergence is carried by neurons excited before rewarded licks (AH > SL): these neurons
come to fire before whisker hits in R+ mice, whereas neurons suppressed before rewarded licks do not change. This is what a
reward-expectation signal recruited by the whisker stimulus would produce.

The geometry says more about how the change happens. The whisker-hit displacement from the spontaneous-lick state grew,
whereas no rotation toward the auditory-hit direction was detected (the confidence interval of the angle change is wide,
so a moderate rotation cannot be excluded). In R+ mice, learning amplifies an existing direction of whisker-hit activity,
which is partly shared with auditory hits. Whisker hits do not become auditory hits: they become more distinct from
spontaneous licks, partly along the reward-lick axis.

Within days, the change starts during the learning session, in the direction of the contingency, and continues in expert
sessions; a large part of the across-day change is already present at the start of expert sessions, consistent with
consolidation between sessions.

The measures track behaviour: sessions with higher whisker hit rates show stronger convergence. This is expected if the
representation underlies the behaviour, but it also means that the difference between cohorts and stages is partly a
difference in performance.
"""


def literature():
    return """
# Relation to the literature

**Outcome-defined convergence.** In R+ mice, a stimulus that has become reward-predictive acquires part of the pre-action
state of an already rewarded action. This is the neural counterpart of stimulus substitution and acquired equivalence in
learning theory: stimuli that predict the same outcome come to share a representation. In the rodent hippocampus,
outcome-related reorganisation of activity has been described as remapping toward goal locations (Dupret et al., 2010) and
as a dedicated population of reward cells reused across environments (Gauthier and Tank, 2018). The present result extends
this to the moment just before the action, and across most of the recorded brain.

**A distributed expected-value signal.** The spontaneous-lick to auditory-hit axis separates licks with and without an
expected reward, so AH-likeness behaves like a reward-expectation or value code rather than a motor code. A value code for
reward-predicting cues that is evenly distributed across frontal cortex, appears with learning and then stays stable was
reported by Ottenheimer et al. (2023), and brain-wide recordings show that choice, action, engagement and prior signals
are spread across many areas (Steinmetz et al., 2019; International Brain Laboratory, 2025; Findling et al., 2025). Global,
cortex-wide representations of a goal-directed lick decision were also found with imaging (Allen et al., 2017), and thirst
changes brain-wide population dynamics (Allen et al., 2019). Here, the effect has the same sign in most area groups, no
single group survives correction, and the cohort × stage × area term is not significant: the convergence looks broadcast
rather than local. The convergence is not concentrated where whisker and auditory cortical projections overlap
(ssl-sensory-spatial-maps 008, at the 90 % and the stricter 50 % contour), which argues against a feed-forward anatomical
convergence of the two sensory streams and favours a shared state signal; axonal projection density, however, is a coarse
proxy for synaptic input.

**Whisker detection learning.** In the same family of whisker-to-lick tasks, learning creates fast whisker-evoked signals in
medial prefrontal cortex and dorsal hippocampus (Le Merre et al., 2018), recruits secondary whisker motor cortex and
suppresses orofacial cortex (Esmaeili et al., 2021), and produces decision signals in sensory, prefrontal and motor cortex,
with prefrontal decision neurons responding little on miss trials and to spontaneous licks (Oryshchuk et al., 2024). In
dorsolateral striatum, direct-pathway neurons carry a whisker-evoked go signal (Sippy et al., 2015), and corticostriatal
synapses from auditory cortex are selectively potentiated during auditory discrimination learning (Xiong et al., 2015).
These studies describe stimulus-evoked activity; the present analysis adds a pre-action, outcome-referenced measure that
contrasts the same movement with and without expected reward, and a cohort in which the whisker lick is not rewarded.

**Fast learning within the first session.** The within-day drift is in the direction of the contingency in both cohorts
already on the learning day. This fits evidence that animals acquire task contingencies faster than their performance
shows (Kuchibhotla et al., 2019) and that mice can learn abruptly (Rosenberg et al., 2021), and it agrees with the
across-days change being partly present at the start of expert sessions.

**Geometry: reuse of an existing direction.** Growth of the whisker-hit displacement without detected rotation resembles
learning that first reuses existing population activity patterns before creating new ones, as in brain-computer
interface experiments (Sadtler et al., 2014; Golub et al., 2018). In low-rank recurrent network models, a small,
structured change of connectivity aligned with an output direction moves a new input onto an existing readout
(Mastrogiuseppe and Ostojic, 2018; Dubreuil et al., 2022). Such models predict that learning acts mainly along the
reward-lick axis, while the orthogonal components change little; the length / angle / orthogonal decomposition used here
can test this directly.

**Pre-lick motor preparation and movement confounds.** Anterior lateral motor cortex carries preparatory activity along a
coding direction before licking (Li et al., 2016; Inagaki et al., 2019; Svoboda and Li, 2018), and a midbrain-thalamus-cortex
pathway switches cortex from preparation to movement (Inagaki et al., 2022). Comparing three first licks in the same window
removes the shared lick command. Reward expectation, however, also changes anticipatory orofacial movements and arousal,
which dominate activity across the brain (Musall et al., 2019; Stringer et al., 2019). Part of AH-likeness may therefore
reflect anticipatory movement or arousal rather than an internal value state; lick-matched comparisons and regressing out
face motion or pupil would separate the two.

# References

- Allen WE, Kauvar IV, Chen MZ, Richman EB, Yang SJ, Chan K, Gradinaru V, Deverman BE, Luo L, Deisseroth K (2017). Global
  representations of goal-directed behavior in distinct cell types of mouse neocortex. *Neuron* 94:891–907.
- Allen WE, Chen MZ, Pichamoorthy N, Tien RH, Pachitariu M, Luo L, Deisseroth K (2019). Thirst regulates motivated
  behavior through modulation of brainwide neural population dynamics. *Science* 364:eaav3932.
- Dubreuil A, Valente A, Beiran M, Mastrogiuseppe F, Ostojic S (2022). The role of population structure in computations
  through neural dynamics. *Nat Neurosci* 25:783–794.
- Dupret D, O'Neill J, Pleydell-Bouverie B, Csicsvari J (2010). The reorganization and reactivation of hippocampal maps
  predict spatial memory performance. *Nat Neurosci* 13:995–1002.
- Esmaeili V, Tamura K, Muscinelli SP, Modirshanechi A, Boscaglia M, Lee AB, Oryshchuk A, Foustoukos G, Liu Y, Crochet S,
  Gerstner W, Petersen CCH (2021). Rapid suppression and sustained activation of distinct cortical regions for a delayed
  sensory-triggered motor response. *Neuron* 109:2183–2201.
- Findling C, et al., International Brain Laboratory (2025). Brain-wide representations of prior information in mouse
  decision-making. *Nature* 645.
- Gauthier JL, Tank DW (2018). A dedicated population for reward coding in the hippocampus. *Neuron* 99:179–193.
- Golub MD, Sadtler PT, Oby ER, Quick KM, Ryu SI, Tyler-Kabara EC, Batista AP, Chase SM, Yu BM (2018). Learning by neural
  reassociation. *Nat Neurosci* 21:607–616.
- Inagaki HK, Fontolan L, Romani S, Svoboda K (2019). Discrete attractor dynamics underlies persistent activity in the
  frontal cortex. *Nature* 566:212–217.
- Inagaki HK, Chen S, Ridder MC, Sah P, Li N, Yang Z, Hasanbegovic H, Gao Z, Gerfen CR, Svoboda K (2022). A
  midbrain-thalamus-cortex circuit reorganizes cortical dynamics to initiate movement. *Cell* 185:1065–1081.
- International Brain Laboratory (2025). A brain-wide map of neural activity during complex behaviour. *Nature* 645.
- Kuchibhotla KV, Hindmarsh Sten T, Papadoyannis ES, Elnozahy S, Fogelson KA, Kumar R, Boubenec Y, Holland PC, Ostojic S,
  Froemke RC (2019). Dissociating task acquisition from expression during learning reveals latent knowledge. *Nat
  Commun* 10:2151.
- Le Merre P, Esmaeili V, Charrière E, Galan K, Salin PA, Petersen CCH, Crochet S (2018). Reward-based learning drives rapid
  sensory signals in medial prefrontal cortex and dorsal hippocampus necessary for goal-directed behavior. *Neuron*
  97:83–91.
- Li N, Daie K, Svoboda K, Druckmann S (2016). Robust neuronal dynamics in premotor cortex during motor planning. *Nature*
  532:459–464.
- Mastrogiuseppe F, Ostojic S (2018). Linking connectivity, dynamics, and computations in low-rank recurrent neural
  networks. *Neuron* 99:609–623.
- Musall S, Kaufman MT, Juavinett AL, Gluf S, Churchland AK (2019). Single-trial neural dynamics are dominated by richly
  varied movements. *Nat Neurosci* 22:1677–1686.
- Oryshchuk A, Sourmpis C, Weverbergh J, Asri R, Esmaeili V, Modirshanechi A, Gerstner W, Petersen CCH, Crochet S (2024).
  Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during
  goal-directed behavior. *Cell Rep* 43:113618.
- Ottenheimer DJ, Hjort MM, Bowen AJ, Steinmetz NA, Stuber GD (2023). A stable, distributed code for cue value in mouse
  cortex during reward learning. *eLife* 12:RP84604.
- Rosenberg M, Zhang T, Perona P, Meister M (2021). Mice in a labyrinth show rapid learning, sudden insight, and efficient
  exploration. *eLife* 10:e66175.
- Sadtler PT, Quick KM, Golub MD, Chase SM, Ryu SI, Tyler-Kabara EC, Yu BM, Batista AP (2014). Neural constraints on
  learning. *Nature* 512:423–426.
- Sippy T, Lapray D, Crochet S, Petersen CCH (2015). Cell-type-specific sensorimotor processing in striatal projection
  neurons during goal-directed behavior. *Neuron* 88:298–305.
- Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD (2019). Distributed coding of choice, action and engagement across the
  mouse brain. *Nature* 576:266–273.
- Stringer C, Pachitariu M, Steinmetz N, Reddy CB, Carandini M, Harris KD (2019). Spontaneous behaviors drive
  multidimensional, brainwide activity. *Science* 364:eaav7893.
- Svoboda K, Li N (2018). Neural mechanisms of movement planning: motor cortex and beyond. *Curr Opin Neurobiol* 49:33–41.
- Xiong Q, Znamenskiy P, Zador AM (2015). Selective corticostriatal plasticity during acquisition of an auditory
  discrimination task. *Nature* 521:348–351.
"""


def caveats():
    unit_line = ("- **Unit set.** This report uses stable units only; the good + mua version is kept as a record (report/), "
                 "not formally compared.\n" if STABLE else
                 "- **Unit set.** good + mua units; the stable-unit version (coverage, presence, drift tests) is in report_stable/; "
                 "the two are kept side by side, not formally compared.\n")
    return f"""
# Caveats and limitations

{unit_line}- **Spontaneous-lick reference.** The current definition excludes licks inside trial windows (no-stim trials included) but
  keeps licks 1.5–8 s after stimuli (reward collection); the decided definition (C8: false-alarm licks counted, exclusion
  [stim − 0.5 s, stim + 8 s] after every stimulus trial) is pending a rerun. In two example sessions it changed neither the
  pre-lick activity nor the per-neuron selectivity.
- **Pending settings.** Within-class SD unit scaling ({ref('eq-sigma')}) and a minimum of 4 events per class everywhere are
  pending; the false-alarm reference variants are not shown.
- **Part II settings.** Learners; common footing with 6 events per class (variant with 4 in the supplement).
- **Drift relative to SL.** In the single-trial mixed model the R− drift relative to spontaneous licks is near zero on the
  learning day; the per-session estimate is partly carried by sessions with few events. The drift relative to AH is robust
  to the estimator.
- **Areas.** Per-area p values are corrected over the area groups shown; the cohort × area and cohort × stage × area terms are
  not significant, so single-area statements are descriptive. Per-area drifts have not been tested against a linear-shift
  null yet.
- **Performance.** Convergence measures correlate with whisker hit rate; cohort and stage differences cannot be fully
  separated from performance differences.
- **Decoder transfers** ({ref('eq-transfer')}) divide by the decoder's separation and are unstable when the decoder is weak;
  P(AH|WH) − P(AH|SL) is the robust readout.
- **Time within a session** is confounded with satiety and engagement; the reference events measured at the same times
  control for shared drifts, not for event-specific ones.
- **Multiple measures.** No correction across panels; the measures are strongly correlated (e.g. λ and Δd).

# Open questions and next steps

1. Rerun with the pending settings (C8 reference, within-class SD, 4 events per class), then the false-alarm reference as a
   control.
2. Linear-shift null for the per-area drifts.
3. Separate convergence from performance (e.g. performance-matched sessions or performance as a covariate).
4. More expert sessions per area group, or pooling of related area groups, to test learning-day specificity per area.
"""


def supp(T, TA):
    o = ["\n# Supplementary figures\n", "\n## Part I\n"]
    for a, f, s in SUPP1:
        o.append(figure(a, f, CAP[a] + f" Script: {s}."))
    o.append("\n## Part II\n")
    for a, f, s in SUPP2:
        o.append(figure(a, f, CAP[a] + f" Script: {s}."))
    o.append("\nAll-mice versions of every population-specific figure are in the result folders (<pop> = all); the "
             "report shows learners only.\n")
    o.append("\n# Supplementary tables\n")
    o.append(table("tbl-s-sizes", T, f"Sample sizes, Part I ({POP}): sessions, mice and tested units ({UNIT_TXT}, mean pre-lick rate ≥ {m51.MIN_FR} Hz, WH vs SL ROC tested) per cohort and stage."))
    o.append(table("tbl-s-sizes-areas", TA, f"Sessions / tested units per area group, cohort and stage ({POP})."))
    o.append(table("tbl-s-slopes", slopes_table(), f"Trial-level slopes against normalised session time (mean per group; Wilcoxon p vs 0), {POP}; day 0 R+ vs R− and cohort × stage by mouse-level permutation."))
    o.append(table("tbl-s-area-contrasts", area_contrasts_table(), f"Per-area drifts, day 0 (D0) and expert (E) means per cohort, R+ vs R− per stage and the cohort × stage interaction (mouse-level permutation, uncorrected), {POP}; area groups with ≥ 3 sessions in every condition (within_day 020)."))
    return "\n".join(o)


def provenance_table():
    rows = []
    for a, f, s in MAIN + SUPP1 + SUPP2:
        rows.append({"Figure": L(a), "Script": s, "File (results folder)": str(f).split("ssl-prelick-convergence/")[-1]})

    return pd.DataFrame(rows)


def appendix():
    par = pd.DataFrame([
        ("Pre-lick window", "[−100, 0) ms before the corrected first lick (Eq. (1))"), ("Hit baseline", "[−1.0, −0.015] s before trial start"),
        ("SL baseline", "[−1.0, −0.5] s before the lick"), ("SL definition", "piezo bout onset (≥ 1 s gap), outside [start − 0.2, rw stop + 0.5] s"),
        ("Units", UNIT_TXT), ("Min firing rate", f"{m51.MIN_FR} Hz"), ("ROC permutations", "1,000"), ("Split halves (d, λ)", "50"),
        ("Min units / events per class", "5 / 4 (area CD: 5 units per area group)"), ("Decoder", "L2 logistic, C = 0.05, balanced, k-fold"),
        ("CD axis criterion", "held-out d′ ≥ 0.3; ≥ 4 WH, ≥ 8 AH, ≥ 8 SL per session"),
        ("Linear shifts", "40, |k| = 5 … n/3"), ("Interaction permutations", "10,000 (mouse level); 5,000 per area"),
        ("Halves subsamples", "20"), ("Epoch footing", "6 events per class, 150 units (variants: 4 events; 50–400 units, all)")],
        columns=["Parameter", "Value"])
    sfx = m51.OUTROOT.name
    files = pd.DataFrame([
        (f"across_days/{sfx}/publication/<pop>/", "Part I figures and stats (062, 075, 078, 079)"),
        (f"across_days/{sfx}/{{lambda,lambda_lda,transfer,decoder_*,pseudopop}}/", "Part I session tables"),
        (f"across_days/{sfx}/normalisation/, extras/", "074 scaling comparison; 076 / 077 per-session extras"),
        (f"within_day/{sfx}/{{halves,slopes*,epochs*,mixed_model,psth_halves}}/", "Part II 001–004, 009"),
        (f"within_day/{sfx}/{{cd_trajectories,within_change,cohort_specificity,day0_predicts_expert}}/<pop>/", "Part II 011, 013, 014, 016"),
        (f"within_day/{sfx}/area_cd_drift/ (+ <pop>/per_area/)", "Part II 015, 017, 020 (per-area CD)"),
        (f"within_day/{sfx}/cosyne/ (+ <pop>/)", "Part II summary figures 005, 006, 008, 010, 018, 019"),
        (f"within_day/{sfx}/FIGURES.md", "index of every Part II figure by question (021)")],
        columns=["Results (combined_results_ks4/ssl-prelick-convergence/)", "Content"])
    return f"""
# Appendix

{table('tbl-a-params', par, "Parameters.")}

{table('tbl-a-files', files, "Result files.")}

{table('tbl-provenance', provenance_table(), "Every figure of the report with the script that writes it (exploratory-analyses/; within_day/ for Part II) and its file in combined_results_ks4/ssl-prelick-convergence/. The COSYNE summary figures reuse the data and statistics of the scripts they summarise.")}

**Code.** `projects/ssl-prelick-convergence/exploratory-analyses/` (Part I: 051–079, `coding_direction.py`; Part II:
`within_day/` 001–021); report: `report/build_report.py`. Parameters and files: {ref('tbl-a-params')}, {ref('tbl-a-files')}.
**History.** 2026-10-07: projects ssl-prelick-convergence and ssl-within-day-remapping merged into this report (Parts I
and II); Part II reorganised as one section per analysis (reward-lick CD drift, halves, PSTHs per half, trial slopes,
change size, epochs, areas, cohort specificity, day-0 vs expert outcome, summary figures); numbered equations; stable-unit
version of the report (report_stable/).
"""


CAP = captions_part2(captions_part1_new(captions()))


def main():
    missing = [str(f) for a, f, *r in MAIN + SUPP1 + SUPP2 + ALLM if not Path(f).exists()]
    if missing:
        raise SystemExit("missing figures:\n  " + "\n  ".join(missing))
    T, TA = sizes()
    md = front(T) + intro() + methods() + part1() + part2() + discussion() + literature() + caveats() + supp(T, TA) + appendix()
    R.write(OUT, md)


if __name__ == "__main__":
    main()
