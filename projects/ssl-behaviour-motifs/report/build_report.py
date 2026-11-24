"""Article report of ssl-behaviour-motifs (skills/project-report): report.md (+ numbers.json, figures/) written into
combined_results_ks4/ssl-behaviour-motifs/report/ by report_lib; report.pdf / report.html built there with build.sh.
Every number comes from the result tables (frozen with R.snapshot)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
import bm_lib as L  # noqa: E402
from report_lib import Report  # noqa: E402

OUT = L.RES / "report"
T, F, K = L.TABLES, L.FIGS, L.RES / "kuma"
R = Report(banned=())
R.register([("fig-validation", "Figure 1"), ("fig-meta", "Figure 2"), ("fig-atlas", "Figure 3"), ("fig-ex-sessions", "Figure 4"),
            ("fig-ex-trials", "Figure 5"), ("fig-usage", "Figure 6"), ("fig-tc-states", "Figure 7"), ("fig-tc-feat", "Figure 8"),
            ("fig-learning", "Figure 9"), ("fig-ex-corr", "Figure 10"),
            ("fig-s-dropout", "Figure S1"), ("fig-s-sweeps", "Figure S2"), ("fig-s-atlas-kpms-side", "Figure S3"),
            ("fig-s-atlas-kpms-top", "Figure S4"), ("fig-s-atlas-arhmm-side", "Figure S5"), ("fig-s-atlas-arhmm-top", "Figure S6"),
            ("tbl-sizes", "Table 1"), ("tbl-thr", "Table 2"), ("tbl-models", "Table 3"), ("tbl-learning", "Table 4"),
            ("tbl-s-excl", "Table S1"), ("tbl-s-registry", "Table S2"),
            ("eq-mm", "Eq. (1)"), ("eq-rest", "Eq. (2)"), ("eq-spec", "Eq. (3)"), ("eq-excess", "Eq. (4)"), ("eq-lt", "Eq. (5)")])
num, pv, ref, figure, table, eq = R.num, R.pv, R.ref, R.figure, R.table, R.eq


def load(name, **kw):
    p = T / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, **kw)


def main():
    snap = [p for p in [T / n for n in ("001_selected_sessions.csv", "full_sessions.csv", "full_exclusions.csv", "label_thresholds.json",
                                         "full_label_thresholds.json", "010_shared_syllables.csv", "010_identity_decoding.csv",
                                         "010_heldout_stability.csv", "010_normalisation_comparison.csv", "010_power.csv",
                                         "009_fingerprint_consistency.csv", "full_009_metastates.csv", "full_009_fingerprints_pooled.csv",
                                         "016_A_curve_correlation_group.csv", "016_B_learning_trial_group.csv",
                                         "016_C_cohort_trial_group.csv", "016_D_prestim_auc_group.csv", "016_E_rplus_rminus_group.csv",
                                         "008_lightning_action_f1_per_mouse.csv", "full_label_fractions.csv",
                                         "018_timecourse_epoch_means.csv", "018_example_sessions.csv")] if p.exists()]
    R.snapshot(snap, OUT)
    pil = load("001_selected_sessions.csv")
    full = load("full_sessions.csv")
    exc = load("full_exclusions.csv")
    thr = json.load(open(T / "full_label_thresholds.json"))
    thr_p = json.load(open(T / "label_thresholds.json"))
    sh = load("010_shared_syllables.csv")
    idd = load("010_identity_decoding.csv")
    ho = load("010_heldout_stability.csv")
    pw = load("010_power.csv")
    cons = load("009_fingerprint_consistency.csv")
    M = load("full_009_metastates.csv")
    Pp = load("full_009_fingerprints_pooled.csv")
    f1 = load("008_lightning_action_f1_per_mouse.csv")
    GA, GB, GC, GD, GE = (load(f"016_{k}_group.csv") for k in ("A_curve_correlation", "B_learning_trial", "C_cohort_trial",
                                                                  "D_prestim_auc", "E_rplus_rminus"))
    nsw = lambda d: {v: json.load(open(K / "kpms" / v / n / "sweep_ns50_k1e+08.json")) for v in L.VIEWS for n in [d]  # noqa: E731
                     if (K / "kpms" / v / n / "sweep_ns50_k1e+08.json").exists()}
    spec = {n: nsw(n) for n in ("raw", "centre", "rest")}
    sel = json.load(open(T / "selected_models.json"))
    md = []
    md.append('---\ntitle: "Behavioural motifs from side and top keypoints during whisker learning (day 0)"\n'
              'subtitle: "ssl-behaviour-motifs: keypoint-MoSeq, AR-HMM and lightning-action, pilot validation and full run"\n'
              f'date: "{pd.Timestamp.today().date()}"\n---\n')
    # ---------------------------------------------------------------- abstract
    nf = len(full)
    md.append("# Abstract\n")
    md.append(f"We segmented the face movements of head-fixed mice during their first whisker-training session (day 0) into discrete "
              f"states with three complementary methods, applied to side- and top-camera DeepLabCut keypoints at 200 Hz: "
              f"keypoint-MoSeq (unsupervised syllables of keypoint movement), a sticky AR-HMM on interpretable features (whisker "
              f"angle, whisking amplitude and phase, jaw, tongue, nose) and lightning-action (supervised classification of still, "
              f"whisking, jaw and lick from heuristic labels). Because states must mean the same thing in every mouse, geometry was "
              f"normalised in millimetres to a common spout-tip reference, every threshold and scaling constant is a statistic of "
              f"the pooled dataset, and keypoint-MoSeq models deviations from each session's resting pose only. A pilot of 8 "
              f"sessions (4 R+, 4 R−) validated the pipeline (cross-mouse sharing of states, session-identity decoding, "
              f"leave-mouse-out recurrence, physical fingerprints), after which all {num('n_full', nf)} eligible day-0 sessions "
              f"were processed and per-trial action and state usage was related to learning. Apart from licking itself, "
              f"learning came with few changes in the action distribution: in R+ mice the inter-trial interval became less "
              f"still and more whisking as the whisker lick probability rose (most likely reward consumption, absent in R−), "
              f"pre-stimulus behaviour hardly changed, and pre-stimulus stillness predicted whisker hits equally in both "
              f"cohorts and in both halves of the session.\n")
    # ---------------------------------------------------------------- intro
    md.append("# Introduction\n")
    md.append("Day-0 whisker learning in this task is measured by licking, but licking is only the final step of a sequence of "
              "orofacial actions (whisking, nose and jaw movements). Whether these actions change as mice learn the whisker "
              "contingency, and whether they differ between the rewarded (R+) and unrewarded (R−) cohorts beyond licking, "
              "requires a segmentation of the video that is comparable across mice.\n")
    md.append("Three methods were used because they answer different questions. keypoint-MoSeq finds recurring movement "
              "motifs without labels and can reveal unexpected structure, but its syllables are hard to interpret and can "
              "capture animal identity instead of behaviour. The AR-HMM on hand-picked features is interpretable by construction "
              "and directly comparable across animals once the features are in physical units. lightning-action classifies the "
              "actions we care about (still, whisking, jaw, lick); its labels come from heuristics, so its accuracy against them "
              "is circular, but it gives a dense, smooth per-frame classification usable for learning analyses.\n")
    # ---------------------------------------------------------------- methods
    md.append("# Methods\n")
    md.append("## Data and inclusion\n")
    md.append(R.methods("task", "lick_alignment"))
    md.append(f"Sessions: every whisker day-0 session of NWB_ks4 with both DLC cameras in the NWB, a known behaviour setup (pixel "
              f"calibration), no DLC timing flags, no video exclusion and at least 30 kept whisker trials, from mice with "
              f"exclude = 0 and cohort R+ or R− in the reference sheet ({ref('tbl-sizes')}; exclusions in {ref('tbl-s-excl')}). "
              f"The pilot used 8 of these sessions (4 R+, 4 R−; 2 learners and 2 non-learners per cohort where available: only "
              f"one dual-view R− non-learner exists). Trials: active context, performance code 6 removed, auditory warm-up block "
              f"cut; no disengagement trimming (behavioural analyses).\n")
    sz = pd.DataFrame([dict(set="pilot", cohort=c, learners=int(((pil.reward_group == c) & (pil.learner == "learner")).sum()),
                            non_learners=int(((pil.reward_group == c) & (pil.learner == "non-learner")).sum()),
                            sessions=int((pil.reward_group == c).sum())) for c in ("R+", "R-")] +
                      [dict(set="full", cohort=c, learners=int(((full.reward_group == c) & (full.learner == "learner")).sum()),
                            non_learners=int(((full.reward_group == c) & (full.learner == "non-learner")).sum()),
                            sessions=int((full.reward_group == c).sum())) for c in ("R+", "R-")])
    md.append(table("tbl-sizes", sz, "Sessions (one day-0 session per mouse) per set and cohort; learner = sheet learning_category "
                    "good / moderate, non-learner = bad."))
    md.append("## Keypoints and geometry\n")
    md.append("Keypoint positions, per-keypoint likelihoods and camera timestamps were read from the NWB_ks4 DLC series "
              "(ephys_utilities dlc_helpers), which carry the alignment corrections; the merged-model h5 files gave identical "
              "frames (lag 0, correlation ≥ 0.98 in every pilot session and view; table 002_nwb_vs_h5). Frames were kept at "
              "the native 200 Hz. Likelihood gating: the jaw is never gated, the tongue below 0.6 is placed on the jaw (tongue in "
              "the mouth), other keypoints below 0.1 are interpolated. Pixels were converted to millimetres and each session was "
              "translated so that its spout tip sits at a common position:" +
              eq("eq-mm", r"\mathbf{x}^{\mathrm{mm}}_{s,k}(t) = c_s\,\mathbf{x}^{\mathrm{px}}_{s,k}(t) + \left(\bar{\mathbf{p}} - c_s\,\mathbf{p}_s\right)") +
              f"where $c_s$ is the mm per pixel of the session's setup and camera, $\\mathbf{{p}}_s$ the median spout-tip position "
              f"(likelihood > 0.5) and $\\bar{{\\mathbf{{p}}}}$ its mean over sessions ({ref('eq-mm')}). The top-camera spout is not "
              f"in the NWB and was taken from the h5; where it is not tracked (top view of two pilot mice) the nose base plus the "
              f"mean spout − nose-base vector replaced it. The particle and jaw_ref points move with the whiskers and the jaw "
              f"(they define the whisker and jaw angles) and were modelled, never used as references.\n")
    md.append("Each session's resting pose is the median position of every keypoint over rest frames (pooled thresholds, below); "
              "keypoint-MoSeq models the deviation from it:" +
              eq("eq-rest", r"\tilde{\mathbf{x}}_{s,k}(t) = \mathbf{x}^{\mathrm{mm}}_{s,k}(t) - \mathbf{r}_{s,k} + \bar{\mathbf{r}}_{k}") +
              f"with $\\mathbf{{r}}_{{s,k}}$ the resting position of keypoint $k$ in session $s$ and $\\bar{{\\mathbf{{r}}}}_k$ its "
              f"mean over the training sessions ({ref('eq-rest')}), so the model sees movement only, not face shape or camera "
              f"framing.\n")
    md.append("## Features and heuristic labels (pooled thresholds)\n")
    tb = pd.DataFrame([dict(quantity=k, unit=thr["units"].get(k.split("_")[-1] if k not in thr["units"] else k, ""),
                            percentile=str(thr["percentiles"].get(k, "")), pilot=f"{thr_p[k]:.3g}", full=f"{thr[k]:.3g}")
                       for k in ("rest_motion", "rest_amp", "whisk", "nose_speed", "nose_amp", "still_motion", "still_amp", "jaw")])
    md.append(f"Features (physical units): whisker angle (deg, NWB), its set-point (< 3 Hz), whisking amplitude and phase (Hilbert "
              f"transform of the 5–15 Hz band), jaw opening (mm from rest), jaw angle (deg), tongue out (likelihood > 0.6), nose "
              f"displacement from rest (mm) and speed (mm/s) in both views, and the summed speed of the modelled keypoints. "
              f"Labels: lick = −40 to +60 ms around piezo licks; jaw = opening above half the median opening at licks without a "
              f"lick within 150 ms; whisking = amplitude above the pooled 75th percentile; nose = nose speed above the pooled "
              f"85th percentile with low whisking; still = low motion and amplitude, no lick within 0.5 s; everything else is "
              f"unlabelled. All thresholds are percentiles of the pooled sessions and are recomputed on the full dataset with the "
              f"same definitions ({ref('tbl-thr')}).\n")
    md.append(table("tbl-thr", tb, "Pooled thresholds (tables/label_thresholds.json, full_label_thresholds.json): pilot (8 "
                    "sessions) and full run (every 10th frame of all sessions)."))
    md.append("## Models\n")
    mt = pd.DataFrame([
        dict(method="keypoint-MoSeq side", input="nose_tip, jaw, jaw_ref (deviation from rest, mm)", settings=f"25 states, kappa AR {sel['kpms']['side']['kappa_ar']:.0e} / full {sel['kpms']['side']['kappa_full']:.0e}, 10 lags, float64"),
        dict(method="keypoint-MoSeq top", input="nose_tip, whisker_base, whisker_tip, particle", settings=f"25 states, kappa {sel['kpms']['top']['kappa_ar']:.0e} / {sel['kpms']['top']['kappa_full']:.0e}, 10 lags, float64"),
        dict(method="AR-HMM side", input="jaw opening, jaw angle, tongue out, nose displacement / speed", settings="15 states, kappa 1e10, 10 lags, pooled scaling"),
        dict(method="AR-HMM top", input="set-point, amplitude, phase (sin, cos), nose displacement / speed", settings="15 states, kappa 1e10, 10 lags, pooled scaling"),
        dict(method="lightning-action", input="keypoint deviations from rest + velocities", settings="dilated TCN, classes still / whisking / jaw / lick (+ nose in the pilot)")])
    md.append(f"Models were fitted on 60 s chunks (pilot: every 240 s of the 8 sessions; full: every 360 s of 16 training mice, "
              f"8 R+ and 8 R−) and applied to every frame of every session ({ref('tbl-models')}). The side tongue was left out of "
              f"keypoint-MoSeq: its rare, multi-millimetre excursions carry almost all of the pose variance and made the "
              f"whitened model degenerate; licking is represented by the other two methods. At 200 Hz the ten autoregressive lags "
              f"are nearly collinear, which produced NaN parameters in single precision, so both autoregressive models run in "
              f"double precision. Stickiness and the number of states were chosen from sweeps ({ref('fig-s-sweeps')}).\n")
    md.append(table("tbl-models", mt, "Models: inputs and settings (chosen from the pilot sweeps)."))
    md.append("## Validation measures\n")
    md.append("A state is session specific when most of its usage, normalised per session, comes from one session:" +
              eq("eq-spec", r"\mathrm{spec}(j) = \frac{\max_s u_{s,j}}{\sum_s u_{s,j}} > 0.8") +
              f"where $u_{{s,j}}$ is the fraction of frames of session $s$ in state $j$; the share of frames in such states "
              f"summarises a model ({ref('eq-spec')}). States used (≥ 0.5 % of a session) by fewer than three mice are flagged and "
              f"excluded from cross-animal comparisons. Session identity was decoded (multinomial logistic regression, five "
              f"contiguous time blocks per session as cross-validation folds, chance 1/8) from the pose in whisking frames and from "
              f"state usage in 30 s windows counted within one heuristic behaviour (matched behaviour). Leave-mouse-out recurrence: "
              f"models fitted without two mice were applied to them. Fingerprints are the mean features of a state in physical "
              f"units, compared across mice.\n")
    md.append("## Learning analyses\n")
    md.append(R.methods("shift_null"))
    md.append("Per trial and epoch (pre-stimulus 2 s, response window, inter-trial interval up to 5 s) we computed the fraction of "
              "frames in each action and meta-state, keypoint-MoSeq transition rates and response latencies, on all frames and "
              "on frames more than 0.5 s from any lick. Correlation with learning is the excess correlation over the "
              "linear-shift null:" +
              eq("eq-excess", r"\Delta r_s = r\left(m_s, p_s\right) - \frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}} r\left(m_s^{(k)}, p_s\right)") +
              f"where $m_s$ is the measure over whisker trials, $p_s$ the whisker P(lick) curve (both Gaussian-smoothed over 5 "
              f"trials), $m_s^{{(k)}}$ the measure shifted by $k$ trials and $\\mathcal{{K}}$ the set of shifts ({ref('eq-excess')}); "
              f"groups were tested by sign-flipping $\\Delta r_s$ across mice (10,000 flips). The change at the learning trial is" +
              eq("eq-lt", r"\Delta_s = \overline{m}_{[\mathrm{LT}, \mathrm{LT}+20)} - \overline{m}_{[\mathrm{LT}-20, \mathrm{LT})}") +
              f"compared with the same difference at pseudo-learning trials ({ref('eq-lt')}). Cohort × trial interactions use per-"
              f"session slopes over the session for whisker, auditory and no-stimulus trials (R− as time control, the other trial "
              f"types as within-session controls); pre-stimulus actions were related to whisker hits by AUC in the early and late "
              f"halves of the session.\n")
    md.append("## Examples and trial-epoch time courses\n")
    md.append("To interpret the states, every state of every method was described in all 76 analysed sessions: the keypoint "
              "positions in the state (deviation from the session's resting pose, drawn on the mean resting pose), the average "
              "of the features around bout onset (up to 200 random bouts of ≥ 25 ms per session; mean ± s.e.m. over mice), the "
              "distribution of bout durations and the fraction of each trial epoch spent in the state. Trial-epoch time courses "
              "are the fraction of frames in each action or meta-state (and the mean of each feature) in 25 ms bins from 2 s "
              "before to 6.5 s after the stimulus, averaged over the kept trials of a condition in each session (whisker, "
              "auditory or no-stimulus trials with or without a lick; whisker hits of the first and last third of whisker "
              "trials), then over sessions with ≥ 3 such trials. Example sessions were chosen by rule, one per cohort × learner "
              "group: among sessions with at least the group's median number of whisker trials (learners: with a learning "
              "trial), the one whose whisker P(lick) is closest to the group median (tables/018_example_sessions.csv). "
              "Script: 018_report_examples.py.\n")
    md.append(R.methods("statistics").split("Cohort ×")[0] + "Tests between R+ and R− are exploratory and uncorrected.\n")
    # ---------------------------------------------------------------- results
    md.append("# Results\n")
    md.append("## Pilot: geometry removal and cross-mouse validity\n")
    s_side = {n: spec[n].get("side", {}).get("frac_frames_session_specific", np.nan) for n in spec}
    s_top = {n: spec[n].get("top", {}).get("frac_frames_session_specific", np.nan) for n in spec}
    txt = (f"On the side view, the share of frames in session-specific syllables was small whatever the normalisation "
           f"(aligned {num('spec_side_raw', s_side['raw'], '{:.2f}')}, centred {num('spec_side_centre', s_side['centre'], '{:.2f}')}, "
           f"rest-subtracted {num('spec_side_rest', s_side['rest'], '{:.2f}')}); on the top view it stayed high "
           f"({num('spec_top_raw', s_top['raw'], '{:.2f}')}, {num('spec_top_centre', s_top['centre'], '{:.2f}')}, "
           f"{num('spec_top_rest', s_top['rest'], '{:.2f}')}; {ref('fig-validation')}A), so on the top view the identity "
           f"signal is not geometric (it remains after the resting pose is removed).")
    md.append(R.claim("geom", txt, "Figure 1A", "share of frames in session-specific syllables", "8 mice",
                      "holds" if np.isfinite(s_top["rest"]) else "fragile") + "\n")
    if len(sh):
        r_ = sh.assign(norm=sh.norm.fillna("")).set_index(["method", "norm", "view"])
        def g(m, n, v, c):
            try:
                return float(r_.loc[(m, n, v), c])
            except KeyError:
                return np.nan
        md.append(f"Syllables used by at least 6 of the 8 mice covered {num('ge6_kpms_side', g('kpms', 'rest', 'side', 'frac_frames_in_ge6_mice_syllables'), '{:.2f}')} "
                  f"of side and {num('ge6_kpms_top', g('kpms', 'rest', 'top', 'frac_frames_in_ge6_mice_syllables'), '{:.2f}')} of top keypoint-MoSeq "
                  f"frames, and {num('ge6_arhmm_side', g('arhmm', '', 'side', 'frac_frames_in_ge6_mice_syllables'), '{:.2f}')} / "
                  f"{num('ge6_arhmm_top', g('arhmm', '', 'top', 'frac_frames_in_ge6_mice_syllables'), '{:.2f}')} of AR-HMM frames ({ref('fig-validation')}B).\n")
    if len(idd):
        pose = idd[idd.test.str.startswith("pose")].set_index("model").accuracy
        md.append(f"Session identity was decoded from the pose in whisking frames with accuracy "
                  f"{num('id_pose_raw', pose.get('keypoints (raw)', np.nan), '{:.2f}')} from aligned keypoints and "
                  f"{num('id_pose_rest', pose.get('keypoints (rest)', np.nan), '{:.2f}')} after rest subtraction (chance 0.125), "
                  f"and from state usage within matched behaviour as shown in {ref('fig-validation')}C.\n")
    if len(f1):
        g1 = f1.groupby(["view", "cls"]).f1.agg(["mean", "min"])
        md.append(f"lightning-action, trained without the tested mouse, reproduced the heuristic labels with mean F1 still "
                  f"{num('f1_still', g1.loc[('side', 'still'), 'mean'], '{:.2f}')}, whisking {num('f1_whisk', g1.loc[('side', 'whisking'), 'mean'], '{:.2f}')}, "
                  f"jaw {num('f1_jaw', g1.loc[('side', 'jaw'), 'mean'], '{:.2f}')} and lick {num('f1_lick', g1.loc[('side', 'lick'), 'mean'], '{:.2f}')} "
                  f"(side view), and nose only {num('f1_nose', g1.loc[('side', 'nose'), 'mean'], '{:.2f}')}; the nose class was "
                  f"therefore dropped from the full run ({ref('fig-validation')}D). These F1 values are against heuristic labels "
                  f"derived from the same keypoints and are circular: they show that the classifier reproduces the heuristic in "
                  f"unseen mice, not that the heuristic is behaviourally correct.\n")
    md.append(figure("fig-validation", F / "017_fig1_validation.png",
                     "Pilot validation (8 mice). A, share of frames in session-specific keypoint-MoSeq syllables (AR stage, "
                     "kappa 1e8, 50 states) for aligned, median-centred and rest-subtracted keypoints, side and top view. B, "
                     "fraction of syllables / states (dots) and of frames (bars) used by at least 6 of 8 mice. C, session-"
                     "identity decoding accuracy (chance 0.125, dotted). D, lightning-action F1 per held-out mouse (dots, R+ "
                     "green, R− magenta) against heuristic labels; bars = mean.", script="017_report_figures.py fig1"))
    if len(ho):
        md.append(f"Leave-mouse-out recurrence: in the two held-out mice, {num('ho_frac_min', ho.frac_frames_in_training_states.min(), '{:.2f}')}–"
                  f"{num('ho_frac_max', ho.frac_frames_in_training_states.max(), '{:.2f}')} of frames fell in states used by the "
                  f"training mice, and the state usage profile correlated with the training mice at r = "
                  f"{num('ho_r_min', ho.usage_corr_with_training.min(), '{:.2f}')}–{num('ho_r_max', ho.usage_corr_with_training.max(), '{:.2f}')} "
                  f"(table 010_heldout_stability).\n")
    if len(cons):
        md.append(f"Fingerprints of shared states agreed across mice (correlation of each mouse's fingerprint with the other mice, "
                  f"median {num('cons_med', cons.r_vs_other_mice.median(), '{:.2f}')}).\n")
    md.append("Decision: the pilot passed for the side view of keypoint-MoSeq, the AR-HMM (states shared by all mice once "
              "features are scaled with pooled statistics) and lightning-action, so the full run was started; top-view keypoint-"
              "MoSeq syllables are reported with the caveat that part of them remain mouse specific.\n")
    md.append("## States and actions across mice (full run)\n")
    if len(M):
        md.append(f"The shared states of all methods and views clustered into {num('K_meta', len(M))} meta-states "
                  f"({', '.join(M.name)}; {ref('fig-meta')}), which link the unsupervised syllables to interpretable movements in "
                  f"physical units.\n")
        md.append(figure("fig-meta", F / "017_fig2_metastates.png", "Meta-states of the full run. A, fingerprint of each "
                         "meta-state (colour: z with pooled statistics; text: value in physical units). B, fraction of frames of "
                         "each method and view in each meta-state (states used by ≥ 3 mice).", script="017_report_figures.py fig2"))
    md.append("### What the actions and states look like\n")
    md.append(f"{ref('fig-atlas')} describes each action and meta-state by its keypoints, its time course around onset, its "
              f"duration and when it occurs in the trial. Still is the absence of whisking, jaw and nose movement and fills the "
              f"pre-stimulus window; whisking is a whisker protraction of several degrees with little jaw movement; lick bouts "
              f"show the ~7 Hz rhythm of jaw opening and tongue protrusion and occur in the response window and, in R+, the ITI. "
              f"Meta-state M2 (jaw angle high + jaw open) is the whole licking episode: whisker protraction, nose movement and jaw "
              f"opening rise together at its onset and it lasts about half a second, while M0 (still) bouts last seconds and "
              f"begin as a movement ends. M1 (nose moving + whisk) and M3 (strong whisking, set-point high) are rarer and less "
              f"tied to the trial. The same description for the individual keypoint-MoSeq syllables and AR-HMM states, with the "
              f"meta-state each maps to, is in {ref('fig-s-atlas-kpms-side')} and {ref('fig-s-atlas-kpms-top')} (keypoint-MoSeq) and {ref('fig-s-atlas-arhmm-side')} and {ref('fig-s-atlas-arhmm-top')} (AR-HMM): side-view "
              f"keypoint-MoSeq has only four syllables shared by ≥ 3 mice with ≥ 0.5 % usage, two of them still, one the "
              f"licking episode and one nose + whisking.\n")
    md.append(figure("fig-atlas", F / "018_fig_atlas_main.png", "Atlas of the lightning-action classes (rows 1–4) and kpms "
                     "side meta-states (rows 5–8), 76 sessions. Columns: side and top keypoints in frames of the state (coloured "
                     "dots = deviation from each session's resting pose drawn on the mean resting pose, × = rest; scale bar 1 mm); "
                     "features around bout onset (whisker angle from the session median, whisking amplitude, jaw opening, tongue "
                     "out, side nose speed; mean ± s.e.m. over mice with ≥ 10 bouts); pooled bout durations (dashed = median); "
                     "fraction of each trial epoch spent in the state (other = outside trial epochs; R+ green, R− magenta, mean ± "
                     "s.e.m. over mice).", script="018_report_examples.py figures"))
    md.append("### Example sessions and trials\n")
    md.append(f"{ref('fig-ex-sessions')} shows one example session per cohort × learner group: its lick curves and every "
              f"whisker trial as a row of actions and meta-states aligned to the stimulus. In the R+ learner, licking (red) and "
              f"the M2 licking episode fill the response window and extend for seconds into the ITI after hits from the "
              f"learning trial on; in R− sessions, hits are followed by a short lick burst only. {ref('fig-ex-trials')} zooms on "
              f"one hit and one miss whisker trial per session, with the raw features and the state sequence of every method: "
              f"the methods agree on licking episodes, whereas the boundaries of still and whisking differ between methods.\n")
    md.append(figure("fig-ex-sessions", F / "018_fig_example_sessions.png", "Example sessions (A, R+ learner; B, R+ non-learner; "
                     "C, R− learner; D, R− non-learner). Top: P(lick) per trial type over the session (Gaussian smoothing, σ = 3 "
                     "trials; whisker orange, auditory blue, no-stimulus black; red dashed = learning trial). Middle and bottom: "
                     "whisker trials (first at the top) aligned to the stimulus, coloured by the dominant lightning-action class "
                     "and kpms side meta-state per 25 ms bin; white lines = stimulus, end of the response window and end of the "
                     "5 s ITI; right strip, hit (black) or miss (grey).", script="018_report_examples.py figures"))
    md.append(figure("fig-ex-trials", F / "018_fig_example_trials.png", "Example whisker trials (left, a hit from the second "
                     "half of the session; right, a miss) of the four example sessions: whisker angle, whisking amplitude, jaw "
                     "opening and tongue (red ticks = piezo licks; shading = response window) and the state sequences of "
                     "lightning-action, kpms meta-states (side, top) and kpms syllables (side, top; arbitrary colours).",
                     script="018_report_examples.py figures"))
    if (F / "017_fig3_usage_by_epoch.png").exists():
        md.append(f"Action and meta-state usage by trial epoch is shown for both cohorts in {ref('fig-usage')}.\n")
        md.append(figure("fig-usage", F / "017_fig3_usage_by_epoch.png", "Usage by trial epoch, whisker trials, full run: actions "
                         "(lightning-action) and keypoint-MoSeq side meta-states, all frames (A, C) and frames > 0.5 s from licks "
                         "(B, D); mean ± s.e.m. over mice, R+ green, R− magenta.", script="017_report_figures.py fig3"))
    tc = load("018_timecourse_epoch_means.csv")
    if len(tc):
        def tcv(ch, cond, cg, epn):
            q = tc[(tc.channel == ch) & (tc.condition == cond) & (tc.cohort == cg) & (tc.epoch == epn)]
            return float(q["mean"].iloc[0]) if len(q) else np.nan
        md.append("### Trial epochs\n")
        txt = (f"The time courses separate the three windows ({ref('fig-tc-states')}, {ref('fig-tc-feat')}). Before the "
               f"stimulus mice were mostly still, more so before whisker hits than before misses (P(still) R+ "
               f"{num('tc_pre_still_wh_rp', tcv('la:1', 'WH', 'R+', 'pre'), '{:.2f}')} vs {num('tc_pre_still_wm_rp', tcv('la:1', 'WM', 'R+', 'pre'), '{:.2f}')}, "
               f"R− {num('tc_pre_still_wh_rm', tcv('la:1', 'WH', 'R-', 'pre'), '{:.2f}')} vs {num('tc_pre_still_wm_rm', tcv('la:1', 'WM', 'R-', 'pre'), '{:.2f}')}), "
               f"with whisking amplitude dropping in the last second before hits. In the response window every lick trial "
               f"started with a whisking burst; R− whisker hits had shorter licking (P(lick) {num('tc_resp_lick_wh_rm', tcv('la:4', 'WH', 'R-', 'resp'), '{:.2f}')} "
               f"vs {num('tc_resp_lick_wh_rp', tcv('la:4', 'WH', 'R+', 'resp'), '{:.2f}')} in R+) followed by whisking. "
               f"In the ITI the cohorts differed only after rewarded licks: after whisker hits R+ mice were rarely still "
               f"(P(still) {num('tc_iti_still_wh_rp', tcv('la:1', 'WH', 'R+', 'iti'), '{:.2f}')} vs {num('tc_iti_still_wh_rm', tcv('la:1', 'WH', 'R-', 'iti'), '{:.2f}')} in R−) "
               f"and kept licking and whisking for 3–5 s, whereas after auditory hits, rewarded in both cohorts, both cohorts "
               f"were rarely still (R+ {num('tc_iti_still_ah_rp', tcv('la:1', 'AH', 'R+', 'iti'), '{:.2f}')}, R− {num('tc_iti_still_ah_rm', tcv('la:1', 'AH', 'R-', 'iti'), '{:.2f}')}); "
               f"after misses, false alarms and correct rejections the ITI was the same in both cohorts. The ITI differences "
               f"therefore follow reward delivery, not cohort.")
        nwh = {c: int(tc[(tc.condition == "WH") & (tc.cohort == c)].n_sessions.max()) for c in ("R+", "R-")}
        md.append(R.claim("iti_reward", txt, "Figure 7", "epoch means of the time courses (descriptive)",
                          f"R+ {nwh['R+']}, R− {nwh['R-']} mice", "holds") + "\n")
        md.append(f"Within the session, R+ hits of the last third of whisker trials were followed by slightly less ITI licking and "
                  f"whisking than hits of the first third (right columns of {ref('fig-tc-states')} and {ref('fig-tc-feat')}).\n")
        md.append(figure("fig-tc-states", F / "018_fig_timecourse_states.png", "Trial-epoch time courses of the lightning-action "
                         "classes and kpms side meta-states (fraction of frames per 25 ms bin, mean ± s.e.m. over mice; R+ green, "
                         "R− magenta). Columns: whisker hit vs miss; auditory hit vs miss; no-stimulus false alarm vs correct "
                         "rejection; whisker hits of the last vs first third of whisker trials. Shading: grey = pre-stimulus "
                         "(2 s), yellow = response window, blue = ITI (5 s).", script="018_report_examples.py figures"))
        md.append(figure("fig-tc-feat", F / "018_fig_timecourse_features.png", "Trial-epoch time courses of the features in "
                         "physical units, same layout as Figure 7: whisker angle (from the session median), whisking amplitude, "
                         "jaw opening, tongue out and nose speed (side, top).", script="018_report_examples.py figures"))
    md.append("## Learning\n")
    lines = []
    for nm, G in (("A curve correlation", GA), ("B learning trial", GB)):
        if len(G):
            g = G[G.scope == "all"].assign(p=lambda d: d.p_signflip)
            lines.append(dict(analysis=nm, tests=len(g), p_below_005=int((g.p < 0.05).sum()),
                              strongest=f"{g.sort_values('p').iloc[0].measure} ({g.sort_values('p').iloc[0].epoch}, {g.sort_values('p').iloc[0].cohort})"))
    if len(GC):
        g = GC[GC.scope == "all"].assign(p=lambda d: d[["p_mannwhitney", "p_welch"]].min(1))
        lines.append(dict(analysis="C cohort x trial", tests=len(g), p_below_005=int((g.p < 0.05).sum()),
                          strongest=f"{g.sort_values('p').iloc[0].measure} ({g.sort_values('p').iloc[0].epoch}, {g.sort_values('p').iloc[0].contrast})"))
    if len(GD):
        g = GD[GD.scope == "all"]
        pl = g.filter(like="p_late_signflip").iloc[:, 0] if "p_late_signflip" in g else pd.Series(dtype=float)
        lines.append(dict(analysis="D pre-stim AUC (late)", tests=int(pl.notna().sum()), p_below_005=int((pl < 0.05).sum()), strongest=""))
    if len(GE):
        g = GE[GE.scope == "all"].assign(p=lambda d: d[["p_mannwhitney", "p_welch"]].min(1))
        lines.append(dict(analysis="E R+ vs R−", tests=len(g), p_below_005=int((g.p < 0.05).sum()),
                          strongest=f"{g.sort_values('p').iloc[0].measure} ({g.sort_values('p').iloc[0].epoch})"))
    LT = pd.DataFrame(lines)
    md.append(f"{ref('fig-learning')} summarises the learning analyses and {ref('tbl-learning')} counts the tests below 0.05 "
              f"(uncorrected; expect about 5 % by chance). Measures that include licking track the learning curve by "
              f"construction (licks define the curve); the informative rows are the no-lick versions and the non-lick actions.\n")
    md.append(table("tbl-learning", LT, "Learning analyses, all mice: number of group tests, number with p < 0.05 "
                    "(sign-flip for A, B and D; min of Mann–Whitney and Welch for C and E; uncorrected) and the measure "
                    "with the smallest p. Full tables: tables/016_*_group.csv."))
    if len(GA) and len(GD) and len(GE):
        def ga(ep, m, c, col="excess_mean"):
            q = GA[(GA.scope == "all") & (GA.epoch == ep) & (GA.measure == m) & (GA.cohort == c)]
            return float(q[col].iloc[0]) if len(q) else np.nan
        def gd(m, c, col):
            q = GD[(GD.scope == "all") & (GD.measure == m) & (GD.cohort == c)]
            return float(q[col].iloc[0]) if len(q) and col in q else np.nan
        def ge(ep, m, col):
            q = GE[(GE.scope == "all") & (GE.epoch == ep) & (GE.measure == m)]
            return float(q[col].iloc[0]) if len(q) else np.nan
        n_an = int(GA[GA.scope == "all"].groupby("cohort").n.max().sum())
        gcp = GC[(GC.scope == "all") & (GC.epoch == "pre")]
        n_cpre = int((gcp[["p_mannwhitney", "p_welch"]].max(1) < 0.05).sum())
        md.append(f"Analysed sessions: {num('n_analysed', n_an)} of the {nf} eligible (the three sessions filmed at 100 Hz were "
                  f"left out because every model was fitted at 200 Hz).\n")
        txt = (f"Licking-related measures track the learning curve by construction (response-window lick fraction, excess r "
               f"{num('A_resp_lick_rp', ga('resp', 'act_lick', 'R+'), '{:.2f}')} in R+ and {num('A_resp_lick_rm', ga('resp', 'act_lick', 'R-'), '{:.2f}')} in R−). "
               f"Away from licks (> 0.5 s), the inter-trial interval of R+ mice became less still and more whisking as their "
               f"whisker P(lick) rose (still excess r {num('A_iti_still_nl_rp', ga('iti', 'act_still_nl', 'R+'), '{:.2f}')}, p {pv('p_A_iti_still_nl_rp', ga('iti', 'act_still_nl', 'R+', 'p_signflip'))}; "
               f"whisking {num('A_iti_whisk_nl_rp', ga('iti', 'act_whisking_nl', 'R+'), '{:.2f}')}, p {pv('p_A_iti_whisk_nl_rp', ga('iti', 'act_whisking_nl', 'R+', 'p_signflip'))}), "
               f"with no such relation in R− (still {num('A_iti_still_nl_rm', ga('iti', 'act_still_nl', 'R-'), '{:.2f}')}, p {pv('p_A_iti_still_nl_rm', ga('iti', 'act_still_nl', 'R-', 'p_signflip'))}; "
               f"whisking {num('A_iti_whisk_nl_rm', ga('iti', 'act_whisking_nl', 'R-'), '{:.2f}')}, p {pv('p_A_iti_whisk_nl_rm', ga('iti', 'act_whisking_nl', 'R-', 'p_signflip'))}). "
               f"The ITI follows the outcome, so this is most likely reward consumption and its after-effects (R+ hits are "
               f"rewarded, R− hits are not), not a learning-related change of spontaneous behaviour: the trial-epoch time courses show "
               f"the same long ITI licking and whisking after auditory hits, which are rewarded in both cohorts ({ref('fig-tc-states')}).")
        md.append(R.claim("iti_rplus", txt, "Figure 9A", "excess r vs whisker P(lick) curve, sign-flip", f"R+ {int(ga('iti', 'act_still_nl', 'R+', 'n'))}, R− {int(ga('iti', 'act_still_nl', 'R-', 'n'))} mice",
                          "holds") + "\n")
        txt = (f"Before the stimulus, behaviour changed little with learning: no pre-stimulus measure had a cohort × trial "
               f"interaction with both tests below 0.05 ({num('n_C_pre_sig', n_cpre)} of {len(gcp)}), and the pre-stimulus "
               f"lightning-action whisking fraction did not follow the curve in either cohort (R+ excess r "
               f"{num('A_pre_whisk_rp', ga('pre', 'act_whisking', 'R+'), '{:.2f}')}, p {pv('p_A_pre_whisk_rp', ga('pre', 'act_whisking', 'R+', 'p_signflip'))}). "
               f"Only R+ stillness rose slightly with the curve (excess r {num('A_pre_still_nl_rp', ga('pre', 'act_still_nl', 'R+'), '{:.2f}')}, "
               f"p {pv('p_A_pre_still_nl_rp', ga('pre', 'act_still_nl', 'R+', 'p_signflip'))}, uncorrected).")
        md.append(R.claim("pre_flat", txt, "Figure 9A, C", "excess r; slopes MW + Welch", f"{n_an} mice", "fragile") + "\n")
        txt = (f"Pre-stimulus state predicted whisker hits in both cohorts and throughout the session: stillness before the "
               f"stimulus raised the hit probability (AUC R+ early {num('D_still_rp_e', gd('act_still_nl', 'R+', 'auc_early'), '{:.2f}')} / late "
               f"{num('D_still_rp_l', gd('act_still_nl', 'R+', 'auc_late'), '{:.2f}')}, R− {num('D_still_rm_e', gd('act_still_nl', 'R-', 'auc_early'), '{:.2f}')} / "
               f"{num('D_still_rm_l', gd('act_still_nl', 'R-', 'auc_late'), '{:.2f}')}; largest sign-flip p {pv('p_D_still_max', max(gd('act_still_nl', c, f'p_{h}_signflip') for c in ('R+', 'R-') for h in ('early', 'late')))}), "
               f"ongoing whisking lowered it (R+ {num('D_whisk_rp_e', gd('act_whisking_nl', 'R+', 'auc_early'), '{:.2f}')} / {num('D_whisk_rp_l', gd('act_whisking_nl', 'R+', 'auc_late'), '{:.2f}')}, "
               f"R− {num('D_whisk_rm_e', gd('act_whisking_nl', 'R-', 'auc_early'), '{:.2f}')} / {num('D_whisk_rm_l', gd('act_whisking_nl', 'R-', 'auc_late'), '{:.2f}')}), "
               f"and the effect did not change from the early to the late half (Wilcoxon, R+ p {pv('p_D_still_rp_le', gd('act_still_nl', 'R+', 'p_late_vs_early_wilcoxon'))}, "
               f"R− p {pv('p_D_still_rm_le', gd('act_still_nl', 'R-', 'p_late_vs_early_wilcoxon'))}). Pre-stimulus quiescence is thus a "
               f"cohort-independent, stable correlate of detecting the whisker stimulus, not a product of learning.")
        md.append(R.claim("pre_still_auc", txt, "Figure 9D", "AUC, sign-flip; Wilcoxon late vs early", f"{n_an} mice", "holds") + "\n")
        txt = (f"Between cohorts, R+ mice were somewhat more still before the stimulus (still fraction "
               f"{num('E_pre_still_rp', ge('pre', 'act_still_nl', 'mean_Rplus'), '{:.2f}')} vs {num('E_pre_still_rm', ge('pre', 'act_still_nl', 'mean_Rminus'), '{:.2f}')}; "
               f"Mann–Whitney p {pv('p_E_pre_still_mw', ge('pre', 'act_still_nl', 'p_mannwhitney'))}, Welch p {pv('p_E_pre_still_w', ge('pre', 'act_still_nl', 'p_welch'))}, exploratory); "
               f"the large cohort differences are in the response window and ITI and follow licking and reward ({ref('tbl-learning')}).\n")
        md.append(txt)
    if (F / "017_fig4_learning.png").exists():
        md.append(figure("fig-learning", F / "017_fig4_learning.png", "Learning analyses, full run. A, excess correlation "
                         "(Eq. 4) of each measure with the whisker P(lick) curve per epoch and cohort; B, change at the learning "
                         "trial minus pseudo-learning trials (Eq. 5); * p < 0.05 sign-flip across mice, uncorrected. C, cohort × "
                         "trial interaction (difference of whisker − auditory slopes, R+ − R−), 12 smallest p, exploratory. D, "
                         "AUC of pre-stimulus measures for whisker hits, early (open) and late (filled) halves.",
                         script="016_learning_stats.py"))
    md.append(f"{ref('fig-ex-corr')} shows the learning analyses on the example sessions: the per-trial measure, its smoothed "
              f"version, the whisker P(lick) curve, the correlation, the mean of the linear-shift null and the excess correlation, "
              f"and the pre-stimulus stillness of hits and misses with its AUC; the last row places the examples among all mice. "
              f"Single sessions vary widely (the example R+ learner's ITI whisking runs against the group trend), which is why "
              f"the group tests are run on per-mouse excess correlations.\n")
    md.append(figure("fig-ex-corr", F / "018_fig_example_correlations.png", "Worked examples of the learning analyses (rows: "
                     "example sessions of Figure 4). Columns 1–3: ITI whisking and ITI stillness (frames > 0.5 s from licks) and "
                     "pre-stimulus stillness per whisker trial (grey dots), smoothed (black, σ = 5 trials) with the smoothed "
                     "whisker P(lick) (orange, right axis); text: r, mean r of the linear-shift null and excess r (Eq. 4); red "
                     "dashed = learning trial. Column 4: pre-stimulus stillness of whisker misses and hits in the early and late "
                     "halves, orange = median, AUC as in Figure 9D. Last row: every mouse (excess r per cohort, black bar = mean; "
                     "AUC early vs late), examples circled.", script="018_report_examples.py figures"))
    if len(pw):
        w = pw[pw.contrast == "within-session change (all 8)"]
        md.append(f"Pilot power: the median absolute effect size of the within-session change was "
                  f"{num('pw_d_within', np.nanmedian(np.abs(w.cohen_d)), '{:.2f}')}; between learners and non-learners "
                  f"{num('pw_d_lrn', np.nanmedian(np.abs(pw[pw.contrast == 'learner vs non-learner'].cohen_d)), '{:.2f}')} "
                  f"(table 010_power).\n")
    # ---------------------------------------------------------------- discussion / caveats
    md.append("# Discussion\n")
    md.append("Making states comparable across mice required removing everything that is not movement: millimetre geometry "
              "anchored on the spout, pooled thresholds and scaling, and the resting pose. With these, AR-HMM states and "
              "lightning-action classes are shared by all mice by construction of their inputs, and side-view keypoint-MoSeq "
              "syllables are mostly shared. Top-view syllables remain partly mouse specific even after rest subtraction, which "
              "points to differences in whisker tracking (which whisker, its length) rather than camera geometry.\n")
    md.append("# Caveats and limitations\n")
    md.append("- Heuristic labels define lightning-action's targets; its F1 is circular and the classes inherit the heuristics' "
              "blind spots (nose movements without whisking are rare and were not separable). The example trials show two of "
              "them: the jaw class can cover seconds of a slightly open but immobile jaw, and the whisking class covers small "
              "(< 5 deg) whisker movements that the other methods call still.\n"
              "- AR-HMM states are short (50–90 ms) at every stickiness tested, so they describe sub-movements rather than "
              "actions; their meta-state mapping carries their interpretation.\n"
              "- keypoint-MoSeq side excludes the tongue; top-view syllables are partly mouse specific; models were fitted on 16 "
              "mice and applied to the others.\n"
              "- Only one dual-view R− non-learner exists, so learner / non-learner contrasts are R+ driven; R+ vs R− tests are "
              "exploratory, uncorrected.\n"
              "- The learning trial exists only for mice classified as learners by ssl-learning-trial-identification 017.\n")
    md.append("# Appendix\n")
    if len(exc):
        ex = exc.groupby(exc.reason.str.split(":").str[0]).size().rename("sessions").reset_index()
        md.append(table("tbl-s-excl", ex, "Excluded day-0 sessions by reason (full list: tables/full_exclusions.csv)."))
    md.append(figure("fig-s-dropout", F / "017_figS1_dropout.png", "DLC dropout in the pilot: fraction of frames below the "
                     "likelihood cutoff per view and keypoint, whole session and 0–200 ms after piezo licks (mean of 8 sessions; "
                     "the tongue below 0.6 means tongue in the mouth).", kind="supplementary", script="017_report_figures.py figS"))
    md.append(f"DLC dropout is summarised in {ref('fig-s-dropout')}, the pilot sweeps in {ref('fig-s-sweeps')}, and every claim "
              f"in {ref('tbl-s-registry')}. Scripts: projects/ssl-behaviour-motifs (001–017, kuma/003–012); results: "
              f"combined_results_ks4/ssl-behaviour-motifs (tables/, figures/, reports/, kuma/).\n")
    md.append(figure("fig-s-sweeps", F / "017_figS2_sweeps.png", "Pilot sweeps of stickiness (kappa) and number of states: median "
                     "state duration (left; grey = 150–500 ms target) and share of frames in session-specific states (right), "
                     "keypoint-MoSeq AR stage (circles) and AR-HMM (squares), side (grey) and top (blue).",
                     kind="supplementary", script="017_report_figures.py figS"))
    for k, a, title in (("kpms_side", "fig-s-atlas-kpms-side", "keypoint-MoSeq side syllables"),
                        ("kpms_top", "fig-s-atlas-kpms-top", "keypoint-MoSeq top syllables"),
                        ("arhmm_side", "fig-s-atlas-arhmm-side", "AR-HMM side states"),
                        ("arhmm_top", "fig-s-atlas-arhmm-top", "AR-HMM top states")):
        md.append(figure(a, F / f"018_fig_atlas_{k}.png", f"Atlas of the {title}: the most used states shared by ≥ 3 mice "
                         f"(at most 12; row label = state → meta-state, usage, number of mice), panels as in Figure 3.",
                         kind="supplementary", script="018_report_examples.py figures"))
    md.append(R.registry_table("tbl-s-registry"))
    text = "\n".join(md)
    summary = "\n".join(md[1:4])
    R.write(OUT, text, summary=summary)


if __name__ == "__main__":
    main()
