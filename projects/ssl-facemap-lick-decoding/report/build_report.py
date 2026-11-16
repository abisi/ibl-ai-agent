"""Article-style report of ssl-facemap-lick-decoding (skills/project-report): does pre-stimulus face video (Facemap)
predict the upcoming whisker-trial lick and track the learning curve, and does it beat DLC keypoints?

Reads the result tables of combined_results_ks4/ssl-facemap-lick-decoding/ (written by 001, 003, 006, 007; frozen in
report/_snapshots/<date>/), the reference sheet (learner category) and the figures of 008_report_figures.py; computes
every quoted statistic here (numbers.json); writes report.md / report_html.md / summary.md / registry.csv and copies
figures + build.sh into report/. No number in the text is typed by hand.
Run (haas):   nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python \
                  ~/code/ibl-ai-agent/projects/ssl-facemap-lick-decoding/report/build_report.py
Render (PC):  bash projects/ssl-facemap-lick-decoding/report/render.sh
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path.insert(0, str(REPO / "skills" / "project-report"))
from report_lib import Report  # noqa: E402

BASE = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-facemap-lick-decoding")
OUT = BASE / "report"
FIG = BASE / "figures"
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
SNAP = os.environ.get("REPORT_SNAPSHOT")
CODE = "projects/ssl-facemap-lick-decoding"
COH = ("R+", "R-")
STG = {"day0": "day 0", "expert": "expert"}

R = Report()
num, pv, ref, figure, table, eq = R.num, R.pv, R.ref, R.figure, R.table, R.eq
R.register([("fig-rois", "Figure 1"), ("fig-auc", "Figure 2"), ("fig-null", "Figure 3"), ("fig-curve", "Figure 4"),
            ("fig-roi-dlc", "Figure 5"), ("fig-features", "Figure 6"),
            ("tbl-sessions", "Table 1"), ("tbl-auc", "Table 2"), ("tbl-null", "Table 3"), ("tbl-curve", "Table 4"),
            ("tbl-roi-dlc", "Table 5"), ("tbl-sizes", "Table S1"), ("tbl-learners", "Table S2"),
            ("tbl-scripts", "Table S3"), ("tbl-registry", "Table S4"),
            ("eq-logit", "Eq. (1)"), ("eq-gain", "Eq. (2)"), ("eq-z", "Eq. (3)"), ("eq-delta", "Eq. (4)")])


def rd(path, **kw):
    p = R.snapshot([path], OUT, tag=SNAP)[0]
    return pd.read_csv(p, **kw)


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return stats.wilcoxon(v).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan


def two(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue


def p3(v):
    return "n/a" if not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")


def f3(v):
    return "n/a" if not np.isfinite(v) else f"{v:.3f}"


# ---------------------------------------------------------------------------------------------------------- data
sheet = pd.read_excel(SHEET).drop_duplicates("mouse_id").set_index("mouse_id")
sessions_all = [s.strip() for s in open(BASE / "sessions.txt") if s.strip()]
clip_sum = pd.concat([rd(f) for f in sorted(glob.glob(str(BASE / "clips" / "summary_*.csv")))])
clip_sum = clip_sum.drop_duplicates("session_id", keep="last").set_index("session_id").reindex(sessions_all)
D = rd(BASE / "analysis" / "decoding_per_session.csv").copy()
C = rd(BASE / "analysis" / "learning_curve_correlations.csv")
Dd = rd(BASE / "roi" / "compare" / "per_session_deltas.csv")
P = rd(BASE / "roi" / "compare" / "roi_vs_dlc_paired.csv")
S = rd(BASE / "roi" / "compare" / "summary_vs_null.csv")
Fr = rd(BASE / "roi" / "compare" / "features_ranking.csv")
rois = rd(BASE / "roi" / "rois.csv")

D["stage"] = np.where(D.day == 0, "day0", "expert")
C["stage"] = np.where(C.day == 0, "day0", "expert")
Dd["stage"] = np.where(Dd.day == 0, "day0", "expert")
D["gain_fm"] = D["auc_base+fm_mot_1s"] - D.auc_base
D["gain_kpca"] = D["auc_base+kpca_mean_1s"] - D.auc_base
D["learning_category"] = D.mouse_id.map(sheet.learning_category)
D["learner_scope"] = ~((D.stage == "day0") & (D.learning_category == "bad"))

# ---------------------------------------------------------------------------------------------------------- sessions
status = clip_sum.status.fillna("not run").replace({"exists": "ok"})
status = status.where(~status.str.startswith("short read"), "short read")
n_list, n_ok = len(sessions_all), int((status == "ok").sum())
n_novid, n_noclip, n_short = (int((status == s).sum()) for s in ("no video", "no clips", "short read"))
mice_list = sorted({s.split("_")[0] for s in sessions_all})
miss = status[status != "ok"]
miss_mice = sorted({s.split("_")[0] for s in miss.index})
novid_mh06 = int(sum(s.startswith(("MH06", "MH07")) for s in miss[miss == "no video"].index))
n_dec = len(D)
assert n_dec == n_ok, (n_dec, n_ok)
n_fallback = int(rois.whisker_fallback.sum()) if "whisker_fallback" in rois else 0

tbl_sessions = pd.DataFrame([
    ("whisker-day sessions listed (sessions.txt)", num("n_listed", n_list)),
    ("side video not found by the ssl-pupil-qc video matcher", num("n_novideo", n_novid)),
    ("no trial window inside the video", num("n_noclips", n_noclip)),
    ("video shorter than expected (short read)", num("n_short", n_short)),
    ("clips + Facemap + decoding done", num("n_ok", n_ok)),
], columns=["step", "sessions"])

sizes = []
for st in STG:
    for c in COH:
        g = D[(D.stage == st) & (D.cohort == c)]
        sizes.append(dict(stage=STG[st], cohort=c, sessions=len(g), mice=g.mouse_id.nunique(),
                          **{"decodable (AUC base)": int(g.auc_base.notna().sum()),
                             "with DLC kpca": int(g["auc_base+kpca_mean_1s"].notna().sum()),
                             "ROI comparison": int(((Dd.stage == st) & (Dd.cohort == c)).sum()),
                             "median whisker trials": f"{g.n_trials.median():.0f}"}))
sizes = pd.DataFrame(sizes)
n_mice = D.mouse_id.nunique()

# ---------------------------------------------------------------------------------------------------------- decoding
models = [("base", "base"), ("kpca_mean_1s", "DLC kpca"), ("fm_mot_1s", "Facemap"),
          ("base+kpca_mean_1s", "base + kpca"), ("base+fm_mot_1s", "base + Facemap")]
med = {}
rows = []
for st in STG:
    for c in COH:
        g = D[(D.stage == st) & (D.cohort == c)]
        r = dict(stage=STG[st], cohort=c)
        for m, lab in models:
            med[(st, c, m)] = g[f"auc_{m}"].median()
            r[lab] = f3(med[(st, c, m)])
        for k, lab in (("fm", "Facemap"), ("kpca", "kpca")):
            v = g[f"gain_{k}"]
            med[(st, c, f"gain_{k}")] = v.median()
            med[(st, c, f"pgain_{k}")] = wil(v)
            r[f"added AUC {lab} (Wilcoxon p)"] = f"{v.median():+.3f} ({p3(wil(v))})"
        h = g[["gain_fm", "gain_kpca"]].dropna()
        med[(st, c, "p_fm_vs_kpca")] = wil(h.gain_fm - h.gain_kpca)
        med[(st, c, "fm_minus_kpca")] = (h.gain_fm - h.gain_kpca).median()
        r["Facemap - kpca gain (paired p)"] = f"{(h.gain_fm - h.gain_kpca).median():+.3f} ({p3(med[(st, c, 'p_fm_vs_kpca')])})"
        rows.append(r)
tbl_auc = pd.DataFrame(rows)
coh_p = {}
for st in STG:
    g = D[D.stage == st]
    for k in ("fm", "kpca"):
        coh_p[(st, k)] = two(g[g.cohort == "R+"][f"gain_{k}"], g[g.cohort == "R-"][f"gain_{k}"])
    coh_p[(st, "auc_fm")] = two(g[g.cohort == "R+"]["auc_base+fm_mot_1s"], g[g.cohort == "R-"]["auc_base+fm_mot_1s"])
gall = {st: D[D.stage == st].gain_fm for st in STG}

# learners-only scope (day-0 sessions of non-learners dropped; expert sessions unchanged)
lrows = []
for c in COH:
    g = D[(D.stage == "day0") & (D.cohort == c) & D.learner_scope]
    lrows.append(dict(cohort=c, sessions=len(g), mice=g.mouse_id.nunique(),
                      **{"added AUC Facemap (p)": f"{g.gain_fm.median():+.3f} ({p3(wil(g.gain_fm))})",
                         "added AUC kpca (p)": f"{g.gain_kpca.median():+.3f} ({p3(wil(g.gain_kpca))})"}))
    med[("learn", c, "gain_fm")] = g.gain_fm.median()
    med[("learn", c, "pgain_fm")] = wil(g.gain_fm)
tbl_learn = pd.DataFrame(lrows)
n_bad_day0 = int(((D.stage == "day0") & ~D.learner_scope).sum())

# ---------------------------------------------------------------------------------------------------------- shift null
nk = [("time", "time"), ("base", "base"), ("kpca_mean_1s", "DLC kpca"), ("fm_mot_1s", "Facemap"),
      ("base+kpca_mean_1s", "base + kpca"), ("base+fm_mot_1s", "base + Facemap")]
nrows, nz = [], {}
for st in STG:
    for c in COH:
        g = D[(D.stage == st) & (D.cohort == c)]
        r = dict(stage=STG[st], cohort=c)
        for k, lab in nk:
            z, p = g[f"r_null_z_{k}"], g[f"r_null_p_{k}"]
            nz[(st, c, k)] = (z.median(), (p.dropna() < 0.05).mean(), wil(z), int(z.notna().sum()))
            r[lab] = f"{z.median():.2f} / {100 * (p.dropna() < 0.05).mean():.0f}%"
        nrows.append(r)
tbl_null = pd.DataFrame(nrows)

# ---------------------------------------------------------------------------------------------------------- learning curve
crows, cz = [], {}
for m, lab in (("motion_energy_1s", "motion energy"), ("lick_proba_fm_mot_1s", "Facemap P(lick)")):
    for cur in ("p_mean", "p_mean_minus_chance"):
        for st in STG:
            for c in COH:
                g = C[(C.measure == m) & (C.curve == cur) & (C.stage == st) & (C.cohort == c)]
                z = g.z_shift.dropna()
                fr = (g.p_shift.dropna() < 0.05).mean()
                cz[(m, cur, st, c)] = (z.median(), fr, wil(z), len(z), g.r.median())
                crows.append({"measure": lab, "curve": cur.replace("_minus_chance", " − chance"), "stage": STG[st], "cohort": c, "n": len(z),
                              "median r": f3(g.r.median()), "median z": f"{z.median():.2f}",
                              "sessions p < 0.05": f"{100 * fr:.0f}%", "Wilcoxon z vs 0": p3(wil(z))})
tbl_curve = pd.DataFrame(crows)
max_frac = max(v[1] for v in cz.values())

# ---------------------------------------------------------------------------------------------------------- ROI vs DLC
def pr(st, met, a, b):
    g = P[(P.stage == st) & (P.cohort == "all") & (P.metric == met) & (P.roi == a) & (P.dlc_or_ref == b)]
    return g.iloc[0] if len(g) else None


comps = [("roi_energy_1s", "dlc_bp4_1s", "ROI energy vs DLC 4 parts"),
         ("roi_svd_1s", "dlc_kpca_1s", "ROI motion SVD vs DLC kpca"),
         ("roi_svd_1s", "full_svd_1s", "ROI motion SVD vs full-frame SVD")]
prow = []
for st in STG:
    for met, ml in (("lick_bacc", "lick bal. acc."), ("curve_pmean", "curve p_mean R2")):
        for a, b, lab in comps:
            x = pr(st, met, a, b)
            if x is None:
                continue
            prow.append({"stage": STG[st], "metric": ml, "comparison": lab, "n": int(x.n),
                         "median ROI": f3(x.median_roi), "median reference": f3(x.median_ref),
                         "median paired diff": f"{x.median_diff:+.3f}", "Wilcoxon p": p3(x.p_wilcoxon_paired)})
tbl_roi = pd.DataFrame(prow)
pmin_roi = min(pr(st, met, a, b).p_wilcoxon_paired for st in STG for met in ("lick_bacc", "curve_pmean")
               for a, b, _ in comps[:2] if pr(st, met, a, b) is not None)
n_roi = Dd.session_id.nunique()
roi_better = sum(int(x.p_wilcoxon_paired < 0.05 and x.median_diff > 0) for st in STG for met in ("lick_bacc", "curve_pmean")
                 for a, b, _ in comps for x in [pr(st, met, a, b)] if x is not None)
ref_better = sum(int(x.p_wilcoxon_paired < 0.05 and x.median_diff < 0) for st in STG for met in ("lick_bacc", "curve_pmean")
                 for a, b, _ in comps for x in [pr(st, met, a, b)] if x is not None)
n_comp = len(tbl_roi)
ge = D[(D.stage == "expert")][["gain_fm", "gain_kpca"]].dropna()
rho_gain, p_rho_gain = stats.spearmanr(ge.gain_fm, ge.gain_kpca)
lick_ex = {c: D[(D.stage == "expert") & (D.cohort == c)].lick_rate.median() for c in COH}

# single features: strongest expert R+ lick features
fx = Fr[(Fr.stage == "expert") & (Fr.target == "lick") & (Fr.cohort == "R+")].sort_values("median_delta")
top_neg = fx.iloc[0]
fxm = Fr[(Fr.stage == "expert") & (Fr.target == "lick") & (Fr.cohort == "R-")]
max_rminus = fxm.median_delta.abs().max()
jaw = Fr[(Fr.target == "pmean") & (Fr.stage == "expert") & (Fr.cohort == "R+") & (Fr.feature == "dlc_jaw_posture_1s")]
jaw_pm = jaw.median_delta.iloc[0] if len(jaw) else np.nan
roi_wh = Fr[(Fr.stage == "expert") & (Fr.target == "lick") & (Fr.cohort == "R+") & (Fr.feature == "roi_whisker_energy_1s")]
dlc_wh = Fr[(Fr.stage == "expert") & (Fr.target == "lick") & (Fr.cohort == "R+") & (Fr.feature == "dlc_whisker_movement_1s")]

# ---------------------------------------------------------------------------------------------------------- text
e = lambda st, c, k: med[(st, c, k)]  # noqa: E731
md = f"""---
title: "Pre-stimulus face video and the upcoming whisker lick (ssl-facemap-lick-decoding)"
date: "{date.today().isoformat()}"
---

**Data.** SSL dataset, NWB_ks4 trials and DLC (old single-project models), side-camera videos; {num('n_ok_t', n_ok)} whisker-day
sessions from {num('n_mice', n_mice)} mice (day 0 and expert days 1–3); cohort (R+ / R−) per mouse from the reference
sheet. Unit of analysis: the session.

# Abstract

We asked whether the mouse's face before stimulus onset carries information about the upcoming choice on whisker trials,
beyond what the trial position and recent history predict, and whether a pixel-based description of the face (Facemap
motion SVD) captures more of it than DeepLabCut (DLC) keypoints. Peri-trial clips of the side video were processed with
Facemap for {num('n_ok_t2', n_ok)} of {num('n_listed_t', n_list)} sessions, and the upcoming lick was decoded from the 1 s
before stimulus onset with a blocked cross-validated logistic regression. On day 0 the face adds little: the median added
AUC over the time + history model is {num('g_d0_rp', e('day0', 'R+', 'gain_fm'), '{:+.3f}')} (R+) and
{num('g_d0_rm', e('day0', 'R-', 'gain_fm'), '{:+.3f}')} (R−). In expert R+ sessions the face adds
{num('g_ex_rp', e('expert', 'R+', 'gain_fm'), '{:+.3f}')} AUC, as much as DLC keypoint PCA
({num('g_ex_rp_k', e('expert', 'R+', 'gain_kpca'), '{:+.3f}')}); expert R− sessions gain nothing. Per-session
correlations of face measures with the learning curve rarely exceed their linear-shift null (at most
{num('maxfrac', 100 * max_frac, '{:.0f}')} % of sessions per group), though the group-level shift z is shifted
consistently. Automatic face ROIs (eye, snout, whisker pad, jaw) do not improve on DLC features for either the lick or the
learning curve. Facemap therefore adds no information beyond the old DLC keypoints for this question.

**Key results**

1. Day 0: face features add little to time + history (median added AUC Facemap
   {e('day0', 'R+', 'gain_fm'):+.3f} R+, p {pv('p_g_d0_rp', e('day0', 'R+', 'pgain_fm'))}; {e('day0', 'R-', 'gain_fm'):+.3f} R−,
   p {pv('p_g_d0_rm', e('day0', 'R-', 'pgain_fm'))}; Wilcoxon vs 0; {ref('fig-auc')}c, {ref('tbl-auc')}).
2. Expert R+: Facemap adds {e('expert', 'R+', 'gain_fm'):+.3f} AUC (p {pv('p_g_ex_rp', e('expert', 'R+', 'pgain_fm'))}),
   DLC kpca {e('expert', 'R+', 'gain_kpca'):+.3f} (p {pv('p_gk_ex_rp', e('expert', 'R+', 'pgain_kpca'))}); paired difference
   Facemap − kpca p {pv('p_fk_ex_rp', e('expert', 'R+', 'p_fm_vs_kpca'))} ({ref('fig-auc')}d).
3. Expert R−: Facemap adds {e('expert', 'R-', 'gain_fm'):+.3f} AUC (p {pv('p_g_ex_rm', e('expert', 'R-', 'pgain_fm'))});
   R+ vs R− added AUC, Mann–Whitney p {pv('p_coh_ex_mw', coh_p[('expert', 'fm')][0])}, Welch p
   {pv('p_coh_ex_w', coh_p[('expert', 'fm')][1])}.
4. Decoding exceeds the linear-shift null only modestly: median z of the base + Facemap model
   {nz[('expert', 'R+', 'base+fm_mot_1s')][0]:.2f} in expert R+ ({ref('fig-null')}, {ref('tbl-null')}).
5. Learning curve: per session, face measures rarely correlate with the curve beyond the shift null; Facemap lick
   probability has median z {cz[('lick_proba_fm_mot_1s', 'p_mean', 'expert', 'R+')][0]:.2f} in expert R+
   (Wilcoxon p {pv('p_cz_ex_rp', cz[('lick_proba_fm_mot_1s', 'p_mean', 'expert', 'R+')][2])}; {ref('fig-curve')}).
6. ROI Facemap vs DLC: {num('roi_better', roi_better)} of {num('n_comp', n_comp)} paired comparisons favour the ROIs at
   p < 0.05 (uncorrected), {num('ref_better', ref_better)} favour the reference ({ref('fig-roi-dlc')}, {ref('tbl-roi-dlc')}).
7. In expert R+ sessions, more pre-stimulus movement of the face predicts no lick (strongest single feature
   {top_neg.feature}, median r − null {num('top_neg', top_neg.median_delta, '{:+.2f}')}); in R− the same features are
   much weaker (largest |median| {num('max_rm', max_rminus, '{:.2f}')}; {ref('fig-features')}).

# Introduction

On whisker trials the mouse either licks or withholds, and over the first whisker session (day 0) R+ mice learn to lick
for the whisker stimulus while R− mice learn to withhold. Part of the trial-to-trial variability of the choice may be
visible on the face before the stimulus: licking preparation (jaw), whisking, sniffing and arousal (pupil). If so, these
pre-stimulus states could confound neural analyses of choice and learning, or could serve as behavioural readouts of
learning in their own right.

An earlier pilot (DLC keypoints, day 0) tested whether keypoints add to a model with only trial position and lick /
reward history. DLC tracks a fixed set of points; Facemap's motion SVD describes all pixel motion and might
capture facial movements that the keypoints miss. This project asks three questions: (i) does the pre-stimulus face
predict the upcoming whisker lick beyond time and history, on day 0 and in expert sessions, in each cohort; (ii) does
Facemap do better than DLC keypoints, on the full frame and on anatomically defined face ROIs; (iii) do pre-stimulus face
measures follow the learning curve within a session.

# Methods

## Task, sessions and trials

{R.methods('task')}
Sessions are all whisker-day sessions with a side video and an exact learning curve (day 0 and expert days 1–3) of
mice with `exclude == 0` in the reference sheet ({num('n_mice_list', len(mice_list))} mice, {n_list} sessions;
{ref('tbl-sessions')}). The analysed trials are the active whisker trials used by the exact learning curves:
`context != passive`, `perf != 6`, `early_lick == 0` (trials with a lick in the pre-stimulus quiet window are absent by
construction). The target is `lick_flag` (lick in the response window). Day 0 and expert (day > 0) sessions are analysed
separately and per cohort; sample sizes are in {ref('tbl-sizes')}. The main analyses use all mice; the learners-only
scope (day-0 sessions of mice with `learning_category == bad` dropped, {num('n_bad_day0', n_bad_day0)} sessions) is in
{ref('tbl-learners')}.

{table('tbl-sessions', tbl_sessions, 'Session accounting. "Video not found": the ssl-pupil-qc matcher (pupil_qc.find_video) did not return a side video aligned to the NWB camera timestamps; most of these are the 2026 MH062–MH070 sessions. Script 001_extract_clips.py; clips/summary_*.csv.')}

## Video clips and Facemap

The raw side videos are uncompressed 640 × 480 AVIs (about 65 TB for all sessions); only peri-trial clips were read:
−2.5 to +0.5 s around the start of every active whisker trial, frames binned 2 × 2 (320 × 240) and encoded as H.264
(CRF 18). The video frame of each NWB side-camera timestamp comes from the frame offset of the ssl-pupil-qc matcher. Each
clip starts with a duplicated first frame so that Facemap's frame difference is zero at clip boundaries; these lead
frames are dropped before analysis (001_extract_clips.py).

Facemap (motion SVD and movie SVD, spatial binning 2 → 160 × 120, no ROI, 500 components computed, first 100 kept) was
run on each session's clips video on the Kuma GPU cluster (002_kuma_facemap.py). The spatial basis is therefore fit on
peri-trial frames of that session only. Components were z-scored over the session's frames. Per trial and window, the
features are the window means of the first 50 motion-SVD components (`fm_mot`), movie-SVD components (`fm_mov`) or the
summed motion energy (`fm_energy`), taken from the trial's own clip. The main window is −1 to 0 s before stimulus onset;
−0.2 and −2 s windows and 250 ms bins were also computed (003_analysis.py).

## Face ROIs

Four rectangular ROIs were placed automatically from the session's median DLC keypoints of the side view
({ref('fig-rois')}): eye (centroid of the four pupil points), snout (nose tip), whisker pad (the `particle` keypoint,
box moved up by about 1 mm) and jaw (cut on the side of the spout when the spout lies inside). ROI sizes scale with the
eye – nose-tip distance (fractions calibrated on hand-drawn ROIs of the pilot). When the whisker-pad keypoint was
mistracked (below the jaw, near the spout, or a > 3 MAD outlier in face coordinates), the whisker box was centred
between the jaw and snout ROIs ({num('n_fallback', n_fallback)} sessions). Each ROI was cropped from the clips video
and processed with Facemap at full resolution (004_roi_auto.py, 005_kuma_roi_facemap.py).

{figure('fig-rois', FIG / 'fig1_rois.png', f'Automatic face ROIs on the mean clip frame of three sessions per cohort (evenly spaced in the sorted session list): eye (blue), whisker pad (orange), snout (green), jaw (red); yellow crosses: DLC anchor keypoints (median position), red cross: spout. Whisker-pad boxes of mistracked sessions are centred between jaw and snout (fallback; {n_fallback} sessions). Script 004_roi_auto.py (roi/previews/); assembled by 008_report_figures.py.', script=f'{CODE}/008_report_figures.py')}

## DLC features

DLC features come from the NWB files (old per-project DLC models). DLC keypoint PCA (`kpca`): 11 keypoints (side and top
view) x / y, points with likelihood < 0.5 masked, resampled to 50 Hz and robustly z-scored over the session; 10 posture
PCs (positions) and 10 movement PCs (|velocity|), averaged in the window. DLC four body parts (`bp4`): jaw distance,
top-view nose distance, whisker angle and pupil area, each as posture (window mean) and movement (mean |difference|);
pupil uses −2..0 s for the 0.2 s window. Sessions where the DLC data and timestamps differ by more than 200 samples are
excluded from the DLC sets (dlc_kpca.py, from the ssl-face-lick-decoding pilot).

## Decoding model

For each session and feature set $X$, the probability of a lick on whisker trial $i$ is modelled by an L2-regularised
logistic regression with balanced class weights:
{eq('eq-logit', r"P(\mathrm{lick}_i = 1 \mid x_i) = \sigma\!\left(\beta_0 + \beta^\top x_i\right), \qquad \sigma(u) = \frac{1}{1 + e^{-u}}")}
where $x_i$ are the standardised features of trial $i$ and the regularisation strength is chosen by an inner 3-fold CV
(9 values, $10^{{-3}}$ to $10$). Trials are split into 5 contiguous blocks (blocked CV, no shuffling, so slow drifts
cannot leak between training and test); each block is predicted from the other four. The held-out predicted
probabilities give the AUC, the balanced accuracy (threshold 0.5) and $r$, the Pearson correlation between predicted
probability and lick. Sessions with fewer than 30 trials or fewer than 5 licks or no-licks are not decoded.

The feature sets of [Eq. (1)](#eq-logit) are: `time` (trial index and its square), `history` (previous trial lick,
previous trial rewarded, previous whisker-trial lick, log time since the last lick; over all active trials), `base` =
time + history, each face set alone, and `base + set`. The added value of a face set is
{eq('eq-gain', r"\Delta\mathrm{AUC}_{\mathrm{set}} = \mathrm{AUC}(\mathrm{base} + \mathrm{set}) - \mathrm{AUC}(\mathrm{base})")}
computed per session ({ref('eq-gain')} is the quantity in {ref('fig-auc')}c,d).

## Linear-shift null

{R.methods('shift_null')}
Here the licks are shifted against the features by $s$ trials ($10 < |s| \le n/3$, 100 shifts) and the whole CV
pipeline is rerun. For a statistic $T$ (held-out $r$ or balanced accuracy) the session's z-score is
{eq('eq-z', r"z = \frac{T_{\mathrm{obs}} - \overline{T_{\mathrm{shift}}}}{\mathrm{sd}(T_{\mathrm{shift}})}")}
where $T_{{\mathrm{{shift}}}}$ are the values over shifts, and the one-sided p is the fraction of shifts with
$T_{{\mathrm{{shift}}}} \ge T_{{\mathrm{{obs}}}}$.

## Learning-curve correlation

Per session, two trial-wise measures were correlated (Pearson, lag 0) with the exact learning curve `p_mean` and with
`p_mean − p_chance`: the pre-stimulus motion energy (−1..0 s) and the cross-validated lick probability of the
Facemap motion-SVD model. Significance is again a linear shift (500 shifts, $|s| > 10$ trials), summarised by the z of
[Eq. (3)](#eq-z) and the two-sided shift p.

## ROI versus DLC comparison

On the ROI sessions, each feature set (DLC bp4, DLC kpca, ROI motion energy of the four ROIs, ROI motion SVD with 20
components per ROI, full-frame SVD) was scored without the time / history model, for the upcoming lick (held-out $r$ and
balanced accuracy, blocked CV as above) and for the learning curve (in-sample $R^2$ of a ridge regression of the curve on
the features). Each score is reported relative to its own shift null:
{eq('eq-delta', r"\delta = T_{\mathrm{obs}} - \mathrm{median}(T_{\mathrm{shift}})")}
which also removes the inflation of the in-sample $R^2$ by the number of features ({ref('eq-delta')} is the quantity of
{ref('fig-roi-dlc')}). Single features (each DLC posture / movement and each ROI energy) were compared by their
Pearson $r$ with the lick or the curve minus the shift-null median (200 shifts; 006_compare_roi_dlc.py,
007_feature_by_feature.py).

## Statistics

The session is the unit. Added AUC and z against 0: Wilcoxon signed-rank test per stage and cohort. Facemap vs kpca
and ROI vs DLC: paired Wilcoxon test on per-session differences. R+ vs R−: Mann–Whitney U and Welch's t test. No
correction for multiple comparisons; the tests are descriptive (many feature sets and windows were examined).

# Results

## Pre-stimulus face predicts the upcoming lick in expert R+ sessions, not on day 0

Every model decodes the upcoming lick above chance on average, mostly through trial position and history
({ref('fig-auc')}a,b; {ref('tbl-auc')}). On day 0, adding the face to the time + history model changes the AUC
little: median added AUC of Facemap {e('day0', 'R+', 'gain_fm'):+.3f} (R+) and {e('day0', 'R-', 'gain_fm'):+.3f}
(R−), of DLC kpca {num('gk_d0_rp', e('day0', 'R+', 'gain_kpca'), '{:+.3f}')} and
{num('gk_d0_rm', e('day0', 'R-', 'gain_kpca'), '{:+.3f}')} ({ref('fig-auc')}c). The learners-only scope gives the same
picture ({ref('tbl-learners')}).

In expert R+ sessions the face adds {e('expert', 'R+', 'gain_fm'):+.3f} AUC with Facemap and
{e('expert', 'R+', 'gain_kpca'):+.3f} with DLC kpca; in expert R− sessions {e('expert', 'R-', 'gain_fm'):+.3f} and
{num('gk_ex_rm', e('expert', 'R-', 'gain_kpca'), '{:+.3f}')}. The cohort difference of the Facemap gain in expert
sessions has Mann–Whitney p {pv('p_coh_ex_mw2', coh_p[('expert', 'fm')][0])} and Welch p
{pv('p_coh_ex_w2', coh_p[('expert', 'fm')][1])}. Per session, the Facemap and kpca gains are similar and correlated
(expert sessions, Spearman $\\rho$ = {num('rho_gain', rho_gain, '{:.2f}')}, p {pv('p_rho_gain', p_rho_gain)};
{ref('fig-auc')}d); their paired difference in expert R+ is {num('fk_ex_rp', e('expert', 'R+', 'fm_minus_kpca'), '{:+.3f}')}
(p {pv('p_fk_ex_rp2', e('expert', 'R+', 'p_fm_vs_kpca'))}).

{figure('fig-auc', FIG / 'fig2_decoding_auc.png', f'Decoding the upcoming whisker lick from the pre-stimulus face (−1..0 s). (a, b) Held-out AUC (blocked 5-fold CV logistic regression, [Eq. (1)](#eq-logit)) per session for the time + history model (base), DLC keypoint PCA, Facemap full-frame motion SVD (50 components), and each face set added to base; day 0 (a) and expert, day > 0 (b). Dots: sessions; bars: medians; green R+, magenta R−; dotted line: chance. (c) Added AUC of Facemap and DLC kpca over base ([Eq. (2)](#eq-gain)) per stage and cohort. (d) Expert sessions: Facemap vs kpca added AUC per session; line: identity. n per group in ' + ref('tbl-sizes') + '; statistics in ' + ref('tbl-auc') + '. Script 008_report_figures.py from analysis/decoding_per_session.csv (003_analysis.py).', script=f'{CODE}/008_report_figures.py')}

{table('tbl-auc', tbl_auc, 'Median held-out AUC per model, median added AUC over base ([Eq. (2)](#eq-gain)) with the Wilcoxon signed-rank p against 0, and the median paired difference Facemap − kpca gain with its paired Wilcoxon p (sessions with both). Uncorrected.')}

## The face signal is partly slow drift

Because licking and face movements both drift within a session, part of the decodability may come from slow trends
shared by features and licks rather than trial-by-trial information. Against the linear-shift null, held-out $r$ of the
face models is above the null on average but modestly ({ref('fig-null')}, {ref('tbl-null')}): median z of base +
Facemap {num('nz_ex_rp', nz[('expert', 'R+', 'base+fm_mot_1s')][0], '{:.2f}')} in expert R+ and
{num('nz_d0_rp', nz[('day0', 'R+', 'base+fm_mot_1s')][0], '{:.2f}')} on day 0 R+, with
{num('nzf_ex_rp', 100 * nz[('expert', 'R+', 'base+fm_mot_1s')][1], '{:.0f}')} % of expert R+ sessions individually
significant (p < 0.05). The time-only model is close to its null (median z
{num('nz_t_ex_rp', nz[('expert', 'R+', 'time')][0], '{:.2f}')} in expert R+), as expected since it only contains drift.

{figure('fig-null', FIG / 'fig3_shift_null.png', 'Decoding relative to the linear-shift null. Per session, z of the held-out r between predicted P(lick) and lick against 100 shifts of the licks relative to the features ([Eq. (3)](#eq-z)); day 0 (left) and expert (right); green R+, magenta R−; bars: medians; dotted line: z = 1.645. Script 008_report_figures.py from analysis/decoding_per_session.csv.', script=f'{CODE}/008_report_figures.py')}

{table('tbl-null', tbl_null, 'Median z against the linear-shift null ([Eq. (3)](#eq-z)) / percentage of sessions with shift p < 0.05, per feature set, stage and cohort (held-out r).')}

## Face measures follow the learning curve only weakly

Per session, neither pre-stimulus motion energy nor the Facemap lick probability correlates with the exact learning
curve beyond the shift null in more than a minority of sessions (at most {max_frac * 100:.0f} % per group;
{ref('fig-curve')}, {ref('tbl-curve')}). At the group level the z are shifted consistently: the Facemap lick
probability is positively related to `p_mean` (median z {cz[('lick_proba_fm_mot_1s', 'p_mean', 'expert', 'R+')][0]:.2f}
expert R+, {num('cz_d0_rp', cz[('lick_proba_fm_mot_1s', 'p_mean', 'day0', 'R+')][0], '{:.2f}')} day 0 R+), and motion
energy negatively in expert R+ (median z {num('cze_ex_rp', cz[('motion_energy_1s', 'p_mean', 'expert', 'R+')][0], '{:.2f}')},
Wilcoxon p {pv('p_cze_ex_rp', cz[('motion_energy_1s', 'p_mean', 'expert', 'R+')][2])}). On day 0, where the curve
changes most, the motion-energy z is near 0 in both cohorts.

{figure('fig-curve', FIG / 'fig4_learning_curve.png', 'Face measures vs the exact learning curve p_mean. Per session, Pearson r (lag 0, across whisker trials) between the measure and the curve, expressed as z against 500 linear shifts ([Eq. (3)](#eq-z)). Left: pre-stimulus motion energy (−1..0 s). Right: cross-validated lick probability of the Facemap motion-SVD model. Green R+, magenta R−; bars: medians. Statistics in {}. Script 008_report_figures.py from analysis/learning_curve_correlations.csv (003_analysis.py).'.format(ref('tbl-curve')), script=f'{CODE}/008_report_figures.py')}

{table('tbl-curve', tbl_curve, 'Learning-curve correlations: median r, median shift z ([Eq. (3)](#eq-z)), percentage of sessions with shift p < 0.05 and Wilcoxon p of z against 0, per measure, curve (p_mean or p_mean − p_chance), stage and cohort.')}

## Face ROIs do not beat DLC

With ROIs on the eye, snout, whisker pad and jaw ({ref('fig-rois')}), Facemap features score like the DLC features for
both the lick and the learning curve ({roi_better} of {n_comp} paired comparisons favour the ROIs, {ref_better} the
reference, p < 0.05 uncorrected), once each is referred to its own shift null ({ref('fig-roi-dlc')},
{ref('tbl-roi-dlc')}; {num('n_roi', n_roi)} sessions). The ROI motion SVD neither beats DLC kpca nor the full-frame
SVD, and ROI motion energy is not better than the DLC four-part posture / movement features. All sets carry the same
cohort pattern: lick information in expert R+ sessions, little in R−.


{figure('fig-roi-dlc', FIG / 'fig5_roi_vs_dlc.png', 'ROI Facemap vs DLC feature sets (−1..0 s), each relative to its linear-shift null ([Eq. (4)](#eq-delta)). (a, b) Upcoming lick: held-out balanced accuracy minus the null median (100 shifts). (c, d) Learning curve p_mean: in-sample ridge R² minus the null median. Grey band: DLC sets (four body parts; keypoint PCA); then ROI motion energy (4 ROIs), ROI motion SVD (20 components per ROI) and full-frame SVD. No time / history model. Dots: sessions; bars: medians; green R+, magenta R−. Script 008_report_figures.py from roi/compare/per_session_deltas.csv (006_compare_roi_dlc.py).', script=f'{CODE}/008_report_figures.py')}

{table('tbl-roi-dlc', tbl_roi, 'Paired comparisons (both cohorts pooled) of the null-referenced scores ([Eq. (4)](#eq-delta)): medians, median paired difference (ROI − reference) and paired Wilcoxon p, uncorrected. Full table (per cohort, 0.2 s window, all metrics): roi/compare/roi_vs_dlc_paired.csv.')}

## Which face signals carry the information

Single features show the same picture ({ref('fig-features')}). In expert R+ sessions, movement of the whiskers, nose and
the whole face before the stimulus predicts that the mouse will not lick: {top_neg.feature} has median r − null
{top_neg.median_delta:+.2f}, and the ROI and DLC versions of the same body part agree (whisker-pad ROI energy
{num('roi_wh', roi_wh.median_delta.iloc[0], '{:+.2f}')}, DLC whisker movement
{num('dlc_wh', dlc_wh.median_delta.iloc[0], '{:+.2f}')}). DLC jaw posture is the one feature positively related to the
lick and to the learning curve (expert R+ p_mean {num('jaw_pm', jaw_pm, '{:+.2f}')}). In R− sessions the features are
much closer to their null.

{figure('fig-features', FIG / 'fig6_single_features.png', 'Single pre-stimulus features. Median over sessions of the Pearson r between the feature and the target minus the median of its linear-shift null (200 shifts), per target (upcoming lick, learning curve p_mean), stage and cohort. Rows: DLC posture / movement of jaw, nose (top view), whisker and pupil, and ROI motion energy (yellow labels), for the −1..0 s (top) and −0.2..0 s (bottom) windows. Script 008_report_figures.py from roi/compare/features_ranking.csv (007_feature_by_feature.py).', script=f'{CODE}/008_report_figures.py')}

# Discussion

The pre-stimulus face contains information about the upcoming whisker choice mainly once R+ mice are experts: a still
face precedes a lick, a moving face (whisking, sniffing, general movement) precedes a withheld response. This fits a
state in which the expert R+ mouse waits for the stimulus with the jaw in position, while movement signals disengagement
or another ongoing behaviour. R− experts withhold licking on most whisker trials (median lick rate
{num('lick_ex_rm', lick_ex['R-'], '{:.2f}')}, vs {num('lick_ex_rp', lick_ex['R+'], '{:.2f}')} in R+), and their remaining licks are not
announced by the face.

On day 0, during learning, the face adds little to trial position and history, and the face measures follow the learning
curve only weakly and, per session, mostly not beyond slow drift. The pre-stimulus face is therefore not a useful
trial-by-trial readout of day-0 learning in this dataset, and it is unlikely to explain choice-related neural signals on
day 0 beyond what the history regressors already capture.

Facemap and DLC carry the same information: full-frame or ROI motion SVD and DLC keypoint PCA give the same added AUC,
the same cohort pattern and correlated per-session gains, and the single-feature analysis shows the same body parts in
both. For this question the DLC features, which are already in the NWB files, are sufficient.

# Caveats and limitations

- **Missing sessions.** {n_list - n_ok} of {n_list} listed sessions are missing: {n_novid} because the side video was
  not found / matched by `pupil_qc.find_video` ({novid_mh06} of them MH062–MH070, recorded 2026), {n_noclip} with no
  trial window inside the video and {n_short} with a short video read. Affected mice: {', '.join(miss_mice)}. Fixing the
  video finder for the 2026 sessions would add an MH-only block of sessions.
- **Old DLC models.** The kpca and bp4 features (and the ROI placement) use the DLC tracks in the current NWB files,
  from the old per-project models. A new merged DLC model is being evaluated and may re-run inference; some old top-view
  tracks failed (e.g. MH009 0207 / 0208, MH025 0421 / 0422, MH027 0422), and sessions with large DLC data / timestamp
  mismatches are excluded from the DLC sets (which is why kpca n is smaller than Facemap n in {ref('tbl-sizes')}). The
  Facemap vs DLC comparison should be repeated with the new tracks.
- **Clip-based Facemap.** The Facemap spatial basis is fit on peri-trial clips only (−2.5..+0.5 s around whisker
  trials), not on the whole session; features outside these windows (e.g. long ITI states) were not examined.
- **Slow drift.** Decoding is cross-validated in contiguous blocks, but licks and face measures drift together; the
  shift null shows that only part of the decodability is trial-by-trial information.
- **No multiple-comparison correction.** Many feature sets, windows and targets were examined; p values are descriptive.
- **Expert stage pooled.** Expert days 1–3 are pooled (day 3 is rare); the history model differs from the RL / GLM
  behaviour models of other projects.

# Appendix

## Sample sizes

{table('tbl-sizes', sizes, 'Sessions and mice per stage and cohort (all sessions with Facemap); sessions decodable by the base model (enough licks and no-licks), with the DLC kpca set, and in the ROI comparison; median number of analysed whisker trials.')}

{table('tbl-learners', tbl_learn, 'Learners-only scope, day 0: day-0 sessions of mice with learning_category == bad (reference sheet) removed; median added AUC over base and Wilcoxon p against 0. Expert sessions are unchanged by the learner rule.')}

## Code, results and methods history

Every script, where it runs and what it writes is listed in {ref('tbl-scripts')}; every claim of the report, with its
figure, statistic and status, is in {ref('tbl-registry')}.

{table('tbl-scripts', pd.DataFrame([
    ('001_extract_clips.py', 'haas', 'clips/SESSION_clips.{mp4,npz}, clips/summary_*.csv'),
    ('002_kuma_facemap.py (+ job / submit .sh)', 'Kuma', 'facemap/SESSION_facemap_small.npz'),
    ('003_analysis.py', 'haas', 'analysis/decoding_per_session.csv, decoding_summary.csv, learning_curve_correlations.csv, per_trial_measures.csv, facemap_lick_decoding_summary.png'),
    ('004_roi_auto.py', 'haas', 'roi/rois.csv, roi/previews/, roi/videos/'),
    ('005_kuma_roi_facemap.py (+ job / submit .sh)', 'Kuma', 'roi/facemap/SESSION_ROI_facemap_small.npz'),
    ('006_compare_roi_dlc.py', 'haas', 'roi/compare/per_session*.csv, summary_vs_null.csv, roi_vs_dlc_paired.csv, roi_vs_dlc.png'),
    ('007_feature_by_feature.py', 'haas', 'roi/compare/features_*.csv, sets_/features_/beh_correlates_real_vs_null.png'),
    ('008_report_figures.py', 'haas', 'figures/fig1-6'),
    ('dlc_kpca.py', 'module', 'DLC loading, trial table, kpca / bp4 features'),
    ('report/build_report.py, render.sh', 'haas / PC', 'report/ (SESSION = mouse_date_time, ROI = eye, jaw, whisker, snout)'),
], columns=['script (' + CODE + ')', 'where', 'outputs (combined_results_ks4/ssl-facemap-lick-decoding/)']), 'Scripts, where they run and what they write.')}

Other result folders: `analysis/facemap_lick_decoding_summary.png` (r-based overview of all feature sets, including the
0.2 / 2 s windows and binned features), `roi/compare/sets_real_vs_null.png`, `features_real_vs_null.png` (real scores and
null medians side by side) and `beh_correlates_real_vs_null.png` (comparison with the earlier beh_correlates analysis).
Superseded runs are kept: `analysis/auc_*.csv` (first AUC-only run of 003), `analysis_test/`, `facemap_test/`
(single-session test) and `roi/compare_v1_heldout_r/` (curve scored by held-out r of blocked CV, biased negative for a
smooth curve; replaced by the shift-referenced in-sample R²).

Methods history: a 6-session pilot (projects/ssl-facemap-pilot) compared full-frame Facemap with hand-drawn ROIs and set
the Facemap parameters used here; the DLC keypoint-PCA pilot (projects/ssl-face-lick-decoding, day 0 only) defined the
trial table, history model and DLC features, now in `dlc_kpca.py`.

"""

R.claim("C1", "Day 0: face adds little AUC over time + history", "Figure 2c", f"Facemap {e('day0', 'R+', 'gain_fm'):+.3f} / {e('day0', 'R-', 'gain_fm'):+.3f}",
        n=f"{int(D[(D.stage == 'day0')].gain_fm.notna().sum())} sessions", status="holds")
R.claim("C2", "Expert R+: Facemap adds AUC, as much as DLC kpca", "Figure 2c,d",
        f"{e('expert', 'R+', 'gain_fm'):+.3f} vs {e('expert', 'R+', 'gain_kpca'):+.3f}; paired p {p3(e('expert', 'R+', 'p_fm_vs_kpca'))}",
        n=f"{int(D[(D.stage == 'expert') & (D.cohort == 'R+')].gain_fm.notna().sum())} sessions", status="holds")
R.claim("C3", "Expert R-: no added value of the face", "Figure 2c", f"{e('expert', 'R-', 'gain_fm'):+.3f}, p {p3(e('expert', 'R-', 'pgain_fm'))}",
        status="holds")
R.claim("C4", "Face decoding exceeds the shift null only modestly", "Figure 3", f"median z {nz[('expert', 'R+', 'base+fm_mot_1s')][0]:.2f} (expert R+)",
        status="fragile", note="few sessions individually significant")
R.claim("C5", "Face measures rarely track the learning curve per session", "Figure 4", f"<= {max_frac * 100:.0f} % sessions p < 0.05",
        status="holds", note="group-level z consistently shifted")
R.claim("C6", "ROI Facemap does not beat DLC", "Figure 5", f"{roi_better}/{n_comp} paired comparisons favour ROIs (p < 0.05, uncorrected)", status="holds",
        note="old DLC tracks")
R.claim("C7", "Expert R+: pre-stimulus face movement predicts no lick", "Figure 6", f"{top_neg.feature} {top_neg.median_delta:+.2f}",
        status="holds")
md += R.registry_table("tbl-registry")

summary = f"""# ssl-facemap-lick-decoding: summary ({date.today().isoformat()})

{n_ok} whisker-day sessions ({n_mice} mice), pre-stimulus (−1..0 s) face features, blocked-CV logistic regression of the
upcoming whisker lick, added to a time + history model.

- Day 0: Facemap adds {e('day0', 'R+', 'gain_fm'):+.3f} (R+) / {e('day0', 'R-', 'gain_fm'):+.3f} (R−) AUC.
- Expert R+: Facemap {e('expert', 'R+', 'gain_fm'):+.3f}, DLC kpca {e('expert', 'R+', 'gain_kpca'):+.3f}; expert R− {e('expert', 'R-', 'gain_fm'):+.3f}.
- Learning-curve correlations: at most {max_frac * 100:.0f} % of sessions per group beat the shift null.
- Face ROIs (eye, snout, whisker pad, jaw) do not beat DLC features; Facemap adds nothing beyond DLC keypoints.
- Caveats: {n_list - n_ok} sessions missing (video not found, mostly MH062–MH070); DLC comparison uses the old DLC models.

![Decoding](figures/fig-auc.png)
"""
R.write(OUT, md, summary=summary)
