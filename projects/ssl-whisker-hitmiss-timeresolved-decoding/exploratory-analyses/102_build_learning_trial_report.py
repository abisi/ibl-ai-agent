"""102 -- Self-contained HTML report recapping the learning-trial work of
2026-09-24/25 (user request: "recap everything in detail: make an html
report with figures"). Numbers in tables are read from the stats CSVs, not
typed; figures are embedded (downscaled JPEG, base64) so the file is
standalone. Sections whose runs are still pending are marked as such and
filled on re-run. SSL data are private: the report is a LOCAL file.

Output: projects/ssl-whisker-hitmiss-timeresolved-decoding/report_learning_trial_split.html
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EA = Path(__file__).resolve().parent
PROJ = EA.parent
FIG = EA / "figures"
LT = PROJ.parent / "ssl-learning-trial-identification"
LTE, LTA = LT / "exploratory-analyses", LT / "artifacts"
OUT = PROJ / "report_learning_trial_split.html"
MAX_W = 1700

_fig_counter = [0]


def img(path: Path, caption: str, width: str = "100%") -> str:
    if not path.exists():
        return f'<p class="pending">[figure pending: {html.escape(path.name)}]</p>'
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, int(im.height * MAX_W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    _fig_counter[0] += 1
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<figure><img src="data:image/jpeg;base64,{b64}" style="width:{width}" alt="{html.escape(path.name)}">'
            f'<figcaption><b>Figure {_fig_counter[0]}.</b> {caption} <span class="src">[{html.escape(path.name)}]</span>'
            f'</figcaption></figure>')


def table(df: pd.DataFrame, caption: str = "", digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return '<p class="pending">[table pending]</p>'
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else (f"{v:.2g}" if c.startswith("p") and abs(v) < 0.01 else f"{v:.{digits}f}"))
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tw"><table>{cap}' + d.to_html(index=False, escape=True, border=0).split("<table", 1)[1].split(">", 1)[1]


def csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def sec(title: str, anchor: str, body: str) -> str:
    return f'<section id="{anchor}"><h2>{title}</h2>{body}</section>'


def p(t: str) -> str:
    return f"<p>{t}</p>"


def ul(items) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ----------------------------------------------------------------------------------------------------------------------
def s_recap() -> str:
    rows = [
        ("Hit/miss decoding split at the learning trial (not session halves), sensory window, generous imbalance handling",
         "Done: 090 (first version, label-shuffle null), superseded by 095 (shift null)", "#decoding"),
        ("Sanity-check the learning trial in the H5", "Done: reproduced 89/89, outcomes match 88/89 (089)", "#lt-check"),
        ("Figures like the session-halves figures (027/029/034 style)", "Done: 091 / 096 whole brain, area_group, area_acronym_custom", "#decoding"),
        ("Explain the figures and the method", "Done (chat) + this report", "#methods"),
        ("Decoding aligned to the learning trial; example sessions with test-trial predictions", "094 done; 092 trial-aligned decoding run, 093 figure not produced on full data (label-shuffle null, superseded)", "#examples"),
        ("List ways to improve the decoding", "Done (chat); items 1-3 implemented (shift null, windows, disengagement)", "#improvements"),
        ("Redo with linear shift null; baseline, sensory, baseline-corrected windows; drop clear disengagement", "Done: 095/096", "#decoding"),
        ("Placebo-split test of the learning trial", "Done with stored LT (097/098)", "#placebo"),
        ("Show disengaged sessions; motivate size matching; are shift null and placebo redundant; compare with halves",
         "Done: 095b, size-matching numbers, 099", "#disengagement"),
        ("Review all learning curves; improve learning-trial identification; alternative curves; later LT (overnight side project)",
         "Done: ssl-learning-trial-identification 000-010", "#lt-review"),
        ("Explain how the learning-trial identification was improved relative to the stored one", "Section 12", "#lt-how"),
        ("Show the re-estimation process in detail vs previous curves", "Done: 008, 009", "#lt-process"),
        ("Q1 should FA be used (R+ FA higher)? Q2 how to get more learners, keeping clean pre/post separation",
         "Done: 007 (lenient LTs, clean-separation gate)", "#lt-lenient"),
        ("Compare the effect in non-learners (vs mid-session splits)", "Done: 101 (learners vs non-learners)", "#nonlearners"),
        ("Performance curves aligned to the new learning trials", "Done: 009_aligned_performance", "#aligned"),
        ("Keep disengaged trials from now on", "Done for all new runs (SSL_NO_DISENGAGE=1)", "#disengagement"),
        ("Re-run the placebo test with the new learning trials", "Done: 097/098 with lt_lenient_clean, disengaged kept", "#placebo-new"),
        ("Is the FA rate estimated at whisker times? How to correct my code? What to change in PyMC? Is the grid still an HMM? "
         "Grid or MCMC?", "Answered: FA fix, PyMC fix, model formulation, comparison", "#codefix"),
    ]
    body = "<table class='recap'><tr><th>#</th><th>You asked</th><th>Status / where</th></tr>" + "".join(
        f"<tr><td>{i + 1}</td><td>{html.escape(a)}</td><td><a href='{h}'>{html.escape(b)}</a></td></tr>"
        for i, (a, b, h) in enumerate(rows)) + "</table>"
    return sec("1. What you asked, and where it is", "recap", body)


def s_summary(lv_stats, new_nodrop, plc_new) -> str:
    items = [
        "<b>Stored learning trial (LT) is often misplaced.</b> Its curve comes from an unconverged sampler (10 tune / 10 draws, "
        "R-hat 1.4-2.0), a prior that forces jumpy curves, and a false-alarm (FA) line placed on the wrong time grid; 36/91 LTs "
        "sit on transient excursions, 24 are hard-coded 10s.",
        "<b>Re-estimated curves and LTs</b> (exact grid inference of the same model, data-chosen smoothness, FA at real trial "
        "times, joint whisker+FA Bayesian change point, clean-separation gate): LTs are later (median +4 R+, +11.5 R-) and "
        "behavioral transitions ~2x sharper. Lenient + clean-gated: 25/50 R+ and 14/38 R- sessions have an LT.",
        "<b>Decoding, stored LT, drift-controlled (shift null)</b>: R+ no change; R- evoked hit/miss information drops after the "
        "LT (sensory minus baseline W p = 0.012), broadly across areas; baseline window unchanged.",
        "<b>With the new (strict) LT</b>: R+ sensory decoding rises clearly (0.065 -> 0.177 above null, p = 0.003, n = 14) and the "
        "R- drop remains; the stored LT masked the R+ effect.",
        "<b>Is the LT special?</b> Placebo splits: the R- drop is deepest at the LT; cohorts differ (p = 0.002); halves split "
        "shows much smaller changes (R- LT vs halves t p = 0.033).",
        "<b>Lenient clean-gated LT</b> (25 R+ / 14 R- learners; 19 / 13 decodable; disengaged trials kept): R+ sensory rise "
        "+0.060 (p = 0.10) -- weaker than strict, the lenient additions dilute it; R+ BASELINE decoding rises (+0.050, p = 0.011); "
        "R- sensory-minus-baseline drop -0.078 (t p = 0.045).",
        "<b>Placebo with the new LT</b>: the R+ sensory change is specific to the LT (percentile 0.68, p = 0.017; learners-only "
        "0.70, p = 0.007); the R- change is not special at the new LT (0.47).",
        "<b>Learners vs non-learners</b>: in R+ learners the sensory change is larger at the new LT than at the session midpoint "
        "(+0.060 vs -0.003, paired p = 0.023); non-learners split at the midpoint show no change (+0.007).",
        "<b>Disengagement</b> barely matters (mean |change| 0.04 on 5 decodable flagged sessions; stored-LT results with and "
        "without the drop are nearly identical) -> kept from now on.",
    ]
    return sec("2. Summary", "summary", ul(items) + p("<i>All results are exploratory: no confirmatory split, many "
                                                    "windows / scopes / variants, uncorrected across them.</i>"))


def s_methods() -> str:
    body = ul([
        "<b>Data</b>: SSL KS4 ephys dataset (<code>ssl_ephys</code>), learning-stage (training day 0) whisker trials in the active "
        "epoch; mouse filters exclude==0, exclude_ephys==0, R+/R- (R+proba dropped); 89 sessions (one per mouse). Hit = lick on "
        "a whisker trial (cohort-independent).",
        "<b>Learning trial (stored)</b>: whisker-trial index from <code>combined_results_ks4/&lt;mouse&gt;/whisker_0/learning_curve/"
        "..._whisker_trial_learning_curve_interp.h5</code>; rule in <code>behaviour_analysis/learning_utils.py</code> (R+: first "
        "5-run with p_low &gt; p_chance; R-: last trial of first above-chance 5-run followed by 5 not-above; &le;10 clamped to 10; "
        "R- fallback 10). Decoded trials split by the LT's start time; LT itself is 'post'.",
        "<b>Features</b>: mean firing rate per unit in a window -- sensory 5-50 ms, baseline -200..-10 ms, sensory minus "
        "baseline; whole brain (all good+MUA units) or areas (area_group, area_acronym_custom, &ge;5 units).",
        "<b>Decoder</b>: StandardScaler + L2 logistic regression, C chosen once per session/area/window, pooled stratified CV, "
        "balanced accuracy.",
        "<b>Size matching</b>: both epochs subsampled (100x) to the per-class minimum counts across epochs (in practice post cut "
        "to pre's hits and misses), same folds; min 2 trials per class per epoch.",
        "<b>Linear shift null</b>: the label sequence shifted against the neural trials by 10-50% of the session (non-wrapping), "
        "50 valid shifts, whole pre/post matched pipeline re-run; reported value = accuracy minus the shift-null mean. Controls "
        "slow drift in firing and behavior.",
        "<b>Placebo split</b>: same pipeline at every 3rd whisker trial (&le;60 splits); per-session percentile of the real LT's "
        "change among placebo changes (&ge;5 trials away); group Wilcoxon/t vs 0.5.",
        "<b>Statistics</b>: within cohort paired Wilcoxon + paired t; R+ vs R- Mann-Whitney + Welch (mandatory test pair); areas: "
        "epoch x area ANOVA + per-area paired Wilcoxon (BH-FDR). Two scopes: all sessions, learners-only (good/moderate).",
    ])
    return sec("3. Methods (current pipeline)", "methods", body)


def s_ltcheck() -> str:
    body = p("The stored <code>learning_trial</code> was reproduced exactly (89/89) by porting the lab rule, and the curve file's "
             "outcomes equal the session's whisker <code>lick_flag</code> in 88/89 sessions (MH038: length mismatch, dropped).")
    body += img(EA / "089_learning_trial_curves_grid.png", "Stored whisker learning curves for all learning sessions, red = "
                "stored learning trial.")
    body += img(EA / "089_learning_trial_summary.png", "Position of the stored learning trial, and hit/miss counts per epoch "
                "(the pre epoch is often tiny).")
    return sec("4. Learning-trial sanity check (stored)", "lt-check", body)


def s_decoding() -> str:
    body = p("<b>First version (090/091)</b> used a per-epoch label-shuffle null. It showed an apparent R+ rise after the LT; "
             "this turned out to be slow drift (the project's convention for hit/miss is the linear shift null, which 090 should "
             "have used). <b>The redo (095/096)</b> uses the shift null, three windows, and (then) dropped the terminal disengaged "
             "block.")
    rows = []
    for w in ("sensory", "baseline", "sensory_minus_base"):
        d = csv(EA / f"096_learning_trial_split_shiftnull_stats_{w}.csv")
        if d is None:
            continue
        d = d[(d.scheme == "whole_brain") & d.metric.isin(["matched_abovenull", "matched_abovenull_delta"])]
        for r in d.itertuples():
            rows.append(dict(window=w, scope=r.scope, group=r.reward_group, n=r.n_sessions, pre=r.mean_pre, post=r.mean_post,
                             p_wilcoxon=r.p_wilcoxon, p_t=r.p_paired_t, p_MW=getattr(r, "p_mannwhitney", np.nan),
                             p_Welch=getattr(r, "p_welch", np.nan)))
    body += table(pd.DataFrame(rows), "Whole brain, stored LT, matched accuracy minus shift null (pre / post means); last row per "
                  "block: R+ vs R- on the change.")
    body += img(FIG / "whole_brain/learning/096_ltsplit_shiftnull_whole_brain_sensory_overlaid_all.png",
                "Stored LT, sensory window, cohorts overlaid (top: pre vs post; bottom: per-session change, R+ vs R-).")
    body += img(FIG / "whole_brain/learning/096_ltsplit_shiftnull_whole_brain_sensory_minus_base_paired_grid_all.png",
                "Stored LT, sensory minus baseline: per-session pre/post lines (markers = how the LT was reached; triangles = "
                "clamped to 10).")
    body += img(FIG / "area_group/learning/096_ltsplit_shiftnull_area_group_sensory_minus_base_grid_condition_cohorts_all.png",
                "Area groups, sensory minus baseline: pre (pale) vs post (full colour). R- drops broadly (epoch x area ANOVA "
                "p = 4e-6); R+ does not.")
    body += img(FIG / "whole_brain/learning/091_hitmiss_ltsplit_whole_brain_sensory_overlaid_all.png",
                "Superseded first version (label-shuffle null): the apparent R+ rise is drift.", width="70%")
    return sec("5. Pre/post learning-trial decoding (stored LT)", "decoding", body)


def s_examples() -> str:
    body = p("One class-balanced decoder per session (all trials), held-out P(hit) per trial. Hits/misses by epoch; running "
             "separation. Selection: per cohort, 2 strongest in the group direction, median, strongest counter-direction.")
    body += img(FIG / "whole_brain/learning/094_hitmiss_lt_example_sessions.png",
                "Example sessions: learning curve (top), held-out decoder P(hit) per trial (filled = hit, open = miss), running "
                "hit-miss separation and pre/post means (bottom). Late-session miss clusters in R+ reflect disengagement; R- "
                "sessions have few hits.")
    return sec("6. Example sessions (test-trial predictions)", "examples", body)


def s_improvements() -> str:
    body = ul([
        "Shift null for drift (<b>implemented</b>, 095).",
        "Engagement/arousal control: baseline and baseline-corrected windows (<b>implemented</b>); disengagement drop "
        "(<b>implemented, then reverted</b> -- negligible effect).",
        "Minimum trials per class, AUC/d' readouts, pooled/mixed-effects models, cross-epoch generalisation, PCA/shrinkage, "
        "motor-contamination and unit-stability checks -- <b>not yet done</b>.",
        "Placebo split (<b>implemented</b>, 097/098).",
    ])
    return sec("7. Improvements to the decoding", "improvements", body)


def s_disengagement() -> str:
    body = p("Rule: trailing block after the last lick on any trial type with &ge;5 whisker AND &ge;3 auditory unlicked trials "
             "(auditory is always rewarded, so unlicked auditory trials mark disengagement). 8/89 sessions flagged; a whisker-only "
             "terminal miss run was not used (it is the learned behavior in R-).")
    body += img(FIG / "whole_brain/learning/095b_disengaged_sessions_detail.png",
                "The 8 flagged sessions: running lick rate per trial type, lick raster, learning curve; grey = dropped block. "
                "AB086 also disengages mid-session (not caught by an end-only rule).")
    sens = csv(EA / "099_disengagement_sensitivity.csv")
    if sens is not None:
        s = sens.assign(abs_diff=(sens.d_drop - sens.d_nodrop).abs())[["session_id", "window", "reward_group", "d_drop", "d_nodrop",
                                                                          "abs_diff"]]
        body += table(s, f"Per-session change (post - pre) with vs without the drop; mean |difference| = {s.abs_diff.mean():.3f}. "
                         "Decision: keep disengaged trials.")
    body += p("<b>Size matching.</b> Pre epochs are small (median 21 whisker trials vs 94.5 post). Using all post trials raises "
              "post accuracy by +0.038 on average (W p = 6e-6), growing with the post/pre trial ratio (Spearman r = 0.54). "
              "Unmatched, the R+ raw change is +0.081; matched, +0.030: two thirds of the unmatched R+ 'rise' is post's extra "
              "training data. R- (-0.048) is unaffected.")
    body += p("<b>Shift null vs placebo are not redundant.</b> The shift null asks whether hit/miss is decodable trial-by-trial "
              "beyond slow drift <i>within</i> an epoch (level); the placebo asks whether the change at the LT exceeds the change "
              "at arbitrary split points (location). They overlap only partly.")
    return sec("8. Disengagement, size matching, nulls", "disengagement", body)


def s_halves() -> str:
    d = csv(EA / "099_lt_vs_halves_stats.csv")
    body = p("Identical pipeline (095; then with the disengaged block dropped), split at the session midpoint instead of the LT.")
    if d is not None:
        body += table(d[["scope", "window", "reward_group", "n", "mean_d_lt", "mean_d_half", "p_wilcoxon_lt_vs_half", "p_t_lt_vs_half",
                         "p_wilcoxon_prepost_lt", "p_wilcoxon_prepost_half"]],
                      "Change at the LT split vs at the halves split (matched acc minus shift null).")
    body += img(FIG / "whole_brain/learning/099_hitmiss_lt_vs_halves_whole_brain_all.png",
                "Top: per-session change at halves (x) vs at the LT (y). Bottom: pre/post means for both splits (absolute levels "
                "differ because matched trial counts differ; compare changes).")
    return sec("9. Learning-trial split vs session halves", "halves", body)


def s_placebo() -> str:
    d = csv(EA / "098_learning_trial_placebo_stats.csv")
    body = p("Percentile of the real LT change among placebo splits of the same session; &lt; 0.5 = bigger drop at the LT than "
             "elsewhere. 'size-matched' = only placebos with a similar matched trial count.")
    if d is not None:
        body += table(d[d.scope == "all"].drop(columns=["scope"]), "Placebo test, stored LT, all sessions.")
    body += img(FIG / "whole_brain/learning/098_hitmiss_lt_placebo_whole_brain_summary_all.png",
                "Left/middle: change vs split position relative to the LT (star = real LT); right: per-session percentile. The R- "
                "change is deepest at the LT; the R+ percentile is driven by splits 20-40 trials before the LT (composition "
                "caveat).")
    return sec("10. Placebo-split test (stored LT)", "placebo", body)


def s_lt_review() -> str:
    inv = csv(LTA / "000_inventory.csv")
    body = p("Side project <code>projects/ssl-learning-trial-identification</code> (overnight). Code reviewed: "
             "<code>beh_plotting_functions.plot_single_mouse_session_learning_curve</code> (PyMC random-walk-on-logit model) and "
             "<code>learning_utils.identify_learning_trial_*</code>.")
    if inv is not None:
        s = inv[inv.sustain_20.notna()]
        body += p(f"In {(s.sustain_20 < 0.5).sum()}/{len(s)} stored curves the LT criterion holds on fewer than half of the 20 "
                  f"trials after the LT (R+ mean {s[s.reward_group == 1].sustain_20.mean():.2f}, R- "
                  f"{s[s.reward_group == 0].sustain_20.mean():.2f}): the 5-trial run is often a transient excursion.")
    body += img(LTE / "000_all_curves.png", "All 98 stored whisker learning curves (40 posterior samples), red = stored LT, blue "
                "dashed = model-free change point; 'sus' = fraction of the next 20 trials where the LT criterion still holds.")
    val = csv(LTE / "001c_validation.csv")
    body += h3("Causes")
    body += ul([
        "<b>Unconverged sampler</b>: <code>pm.sample(10, tune=10, chains=4)</code> = 40 samples.",
        "<b>Prior forces jumpy curves</b>: tau ~ Gamma(10,10), sigma ~ 1 logit/trial.",
        "<b>FA interpolation</b>: <code>interp_fa_curve</code> places the no-stim curve on an evenly spaced time grid.",
        "<b>Whisker-only logic</b> misses R+ mice that learn by dropping FA; <b>no 'no learning' outcome</b>: flat / never-licked "
        "sessions still get an LT.",
    ])
    if val is not None:
        body += table(val, "Refitting the ORIGINAL model with PyMC: stored setting vs converged, compared to the exact grid posterior.")
    body += img(LTE / "001c_validation.png", "Exact grid posterior (black) vs PyMC 10x10 (red) and 1000x1000 (blue).", width="80%")
    return sec("11. Learning-curve review: is the stored LT well placed?", "lt-review", body)


def h3(t):
    return f"<h3>{t}</h3>"



def s_lt_how() -> str:
    body = p("This section explains, step by step, how the learning-trial (LT) identification was changed relative to the "
             "stored one, why, and what each change bought. The learning-curve MODEL itself is kept as in the original "
             "implementation (random walk on the logit of lick probability, Bernoulli licks); what changes is how it is "
             "computed, how smooth it is allowed to be, how the false-alarm (FA) chance line is placed, and the rule that "
             "turns the curves into a learning trial.")
    body += h3("The stored procedure (starting point)")
    body += ul([
        "Fit the whisker and the no-stim (FA) learning curves with PyMC, <code>pm.sample(10, tune=10, chains=4)</code> "
        "(40 samples), prior 1/&sigma;&sup2; ~ Gamma(10,10) (&sigma; &asymp; 1 logit per trial).",
        "Place the FA curve on an evenly spaced time grid and read it at whisker times = 'chance' line.",
        "Mark whisker trials where the lower edge of the 80% band exceeds chance (R+) and apply a run rule: R+ = start of "
        "the first run of 5 such trials (or 'expert' if the first 20 are above with mean p &ge; 0.8); R- = the last trial "
        "of the first above-chance 5-run followed by 5 not-above trials.",
        "Clamp any LT &le; 10 to 10; if no R- criterion is met, set LT = 10. Every session therefore gets an LT except R+ "
        "non-learners.",
    ])
    body += h3("What went wrong (diagnosis)")
    body += ul([
        "<b>Transient placements</b>: in 36/91 sessions the LT criterion holds on fewer than half of the 20 trials after "
        "the LT -- the 5-trial run is often a brief excursion (R+ mean 0.71, R- 0.43).",
        "<b>Arbitrary values</b>: 18/47 R+ and 6/34 R- LTs are exactly 10 (clamp or fallback); several R- LTs land on "
        "isolated late blips in mice that never licked the whisker above FA (AB093 118, AB139 117, AB140 173).",
        "<b>Noisy band</b>: the 40-sample sampler is unconverged (R-hat 1.4-2.0, ESS 12-17), so the band that decides "
        "'above chance' flickers.",
        "<b>Over-jumpy curves</b>: the prior forces &sigma; &asymp; 1 whereas the data prefer 0.49 (R+) / 0.14 (R-).",
        "<b>Mis-timed chance line</b>: even-grid FA placement shifts the chance line by 0.06 on average in R+ (up to 0.29).",
        "<b>Blind spots of the rule</b>: R+ mice that lick the whisker from the start and learn by DROPPING FA are missed "
        "or clamped to 10 (MH028: stored 10, behavior changes at ~27); a whisker AND FA decline (general licking drop) can "
        "be read as learning; there is no 'no learning event' outcome.",
    ])
    body += h3("Changes, in order")
    steps = pd.DataFrame([
        ("1", "Exact posterior of the same model",
         "Grid forward-backward computation replaces the 10/10 sampler (same model and prior).",
         "Removes sampler noise; mean curve moves only ~0.015 but the band is now stable. Validated vs converged PyMC "
         "(&le; 0.005). Alone it does not fix the rule problems (stored rule on exact curves = L1: still 18 R+ LTs at 10)."),
        ("2", "Data-chosen smoothness",
         "Step size &sigma; integrated over a broad range (0.02-1.5) weighted by the data, instead of pinned at ~1.",
         "Smoother, interpretable curves (&sigma; 0.49 R+ / 0.14 R-); evidence &gt; 20:1 for the smoother model in 30/88 "
         "sessions, never the reverse. Short excursions no longer look like learning."),
        ("3", "FA curve at real trial times",
         "Each no-stim estimate placed at its own trial time, then interpolated at whisker times.",
         "Correct chance line (fixes up to 0.29 shifts); matters most where FA changes during the session (MH028)."),
        ("4", "Discrimination as a probability",
         "Per trial P(whisker lick prob &gt; FA lick prob) from the posteriors, instead of 'lower band edge above the chance "
         "line'.",
         "A graded, stable measure instead of an on/off test on a noisy band (process figure, panel D)."),
        ("5", "Learning trial = change point with uncertainty",
         "A left-to-right segment model on raw whisker AND no-stim outcomes: R+ naive -&gt; learned (higher whisker-minus-FA "
         "discrimination) -&gt; optional end-of-session decline; R- generalising -&gt; learned (discrimination at least "
         "halves). Lick rates per segment integrated out; posterior over the LT (median, 90% CI) and a Bayes factor for "
         "'any transition' (strict: log10 BF &gt; 0.5).",
         "Placed where behavior actually changes: later (median +4 R+, +11.5 R-), no clamp at 10, sharper transitions "
         "(whisker-FA rate 20 after minus 20 before: R+ +0.40 vs +0.21 stored; R- -0.27 vs -0.14). Using FA jointly catches "
         "FA-drop learners (MH028 -&gt; 27) and rejects general licking declines (MH018, MH034)."),
        ("6", "Explicit 'no learning event' outcomes",
         "Categories instead of a forced LT: learner / already discriminating from the start (R+) / never licked above FA "
         "(R-) / no clear transition.",
         "Flat, never-licked and high-from-start sessions no longer receive an arbitrary LT (e.g. AB139, AB094, AB107)."),
        ("7", "Half-way rule for gradual learners",
         "On the smooth curves: first trial where whisker-minus-FA crosses half-way between its start and plateau, "
         "sustained 16/20 trials.",
         "Covers gradual changes a step model handles poorly; agrees with the change point where both exist (median 1 trial "
         "R+, 3 trials R-)."),
        ("8", "Less conservative cascade + clean-separation gate",
         "Take the first available of: strict change point -&gt; lenient change point (log10 BF &gt; 0) -&gt; R+ whisker-only "
         "rise with FA as a floor -&gt; half-way rule. Keep only LTs where the 20 whisker trials after vs before differ in "
         "the learned direction by &gt; 0.05 in whisker lick rate OR whisker-FA discrimination (posterior P &gt; 0.9).",
         "More learners (25 R+ / 14 R- vs 17 / 14 strict) with a guaranteed clean behavioral separation. The same gate "
         "passes only 20/47 R+ and 13/34 R- stored LTs."),
    ], columns=["#", "Change", "What", "Why / effect"])
    body += table(steps, "Changes relative to the stored procedure.")
    body += h3("Tried and dropped")
    body += ul([
        "<b>Stored rule on the smoother curves</b> (L2): R- almost never satisfies the 'first run followed by 5 not-above' "
        "pattern on smooth curves (13/38 defined, mostly clamped).",
        "<b>Sustained-probability rule</b> (L3: P(whisker &gt; FA) &ge; 0.9 on 16/20 trials): clean but far too strict for "
        "R- (8/38 defined).",
        "<b>Whisker-only change point</b> (L4 max-likelihood, L5 Bayesian): misses R+ mice that learn by dropping FA "
        "(whisker already high); kept only as a lenient R+ fallback with FA as a floor.",
        "<b>Whisker-only clean gate</b> (first version, 30 trials, 0.1 margin): kept only 14/34 R+ -- rejected FA-drop "
        "learners; replaced by the either-criterion gate.",
    ])
    rs = csv(LTE / "005_rule_summary.csv")
    rs7 = csv(LTE / "006_rule_summary.csv")
    if rs is not None:
        if rs7 is not None:
            rs = pd.concat([rs, rs7[rs7.rule == "L7"]], ignore_index=True)
        names = {"L0_stored": "stored", "L1_exact": "L1 stored rule, exact curve", "L3_sustain": "L3 sustained prob.",
                 "L4_cp": "L4 whisker CP (ML)", "L5": "L5 whisker CP (Bayes)", "L6": "L6 joint CP, strict",
                 "L6_lenient": "L6 joint CP, lenient", "L7": "L7 half-way"}
        rs = rs.assign(rule=rs.rule.map(names).fillna(rs.rule))
        cols = [c for c in ["reward_group", "rule", "n_defined", "median_lt", "n_at_10", "median_shift_vs_stored", "contrast20",
                            "held_post"] if c in rs.columns]
        body += table(rs[cols].sort_values(["reward_group"], kind="stable"),
                      "All rules compared (ephys learning sessions): sessions with an LT, median LT, LTs clamped at 10, shift vs "
                      "stored, contrast20 = whisker-minus-FA lick rate 20 trials after minus 20 before (R+ should be &gt; 0, "
                      "R- &lt; 0), held_post = fraction of later windows on the learned side.")
    body += img(LTE / "003_rule_summary.png", "First comparison of rules on model-free behavior (L0 stored, L1 stored rule on exact "
                "curves, L2 stored rule on smooth curves, L3 sustained probability, L4 whisker change point).")
    body += p("<b>Net effect.</b> The new LT sits where the raw behavior changes (see the aligned-performance figure: R+ "
              "0.37 -&gt; 0.89, R- 0.66 -&gt; 0.11 whisker lick rate around the new LT vs 0.45 -&gt; 0.74 and 0.38 -&gt; 0.13 "
              "around the stored LT), comes with an uncertainty, and is withheld when there is no learning event -- at the cost "
              "of fewer sessions with an LT.")
    return sec("12. How the learning-trial identification was improved", "lt-how", body)


def s_lt_process() -> str:
    body = p("New curves: exact forward-backward (hidden-Markov) inference of the same state-space model, with the random-walk "
             "step chosen by the data, FA placed at the actual no-stim trial times. New LTs: joint whisker+FA Bayesian change "
             "point (posterior over the LT, credible interval, Bayes factor for 'any transition', categories) and a half-way rule "
             "for gradual learners.")
    body += img(LTE / "008_process_examples.png", "Step by step on 6 example sessions: A stored curve; B same model computed "
                "exactly; C data-chosen smoothness + FA at real times; D per-trial P(whisker &gt; FA); E learning-trial posteriors "
                "and candidates.")
    body += img(LTE / "009_population_comparison.png", "Population: (a) step size, (b) evidence for smoother curves, (c) FA "
                "interpolation error, (d) stored vs new LT, (e) sessions per rule and passing the clean gate, (f) FA early vs mid "
                "(Q1).")
    rs = csv(LTE / "006_rule_summary.csv")
    if rs is not None:
        body += table(rs, "Rules compared on model-free behavior: contrast20 = (whisker - FA) rate 20 trials after minus before; "
                          "held_post = fraction of later windows on the learned side.")
    body += img(LTE / "006_final_review_Rplus.png", "R+: every session, re-estimated curve, stored LT (red), strict change point "
                "(blue, CI shaded), half-way rule (orange).")
    body += img(LTE / "006_final_review_Rminus.png", "R-: same.")
    return sec("13. Re-estimating curves and learning trials", "lt-process", body)


def s_lt_lenient() -> str:
    fa = csv(LTE / "007_fa_behavior.csv")
    cnt = csv(LTE / "007_counts.csv")
    body = h3("Q1: should FA be used, given R+ FA is higher?")
    if fa is not None:
        g = fa.groupby("reward_group")[["fa_early", "fa_mid", "wh_early", "wh_mid"]].mean()
        body += table(g.reset_index(), "Lick rates, first 20% vs middle 20-80% of whisker trials (session means).")
    body += p("R+ FA stays high while R- FA falls. Requiring the <i>discrimination</i> (whisker - FA) to rise under-detects R+ "
              "learners; for R-, FA is essential (without it a general drop in licking -- whisker and FA falling together -- would "
              "count as whisker learning). <b>Implemented</b>: R+ may use a whisker-only rise with FA only as a floor; R- keeps the "
              "joint criterion.")
    body += h3("Q2: more learners, and the clean-separation note")
    body += p("Less conservative cascade: strict joint change point -> lenient joint (log10 BF &gt; 0) -> R+ whisker-only rise "
              "(FA floor) -> half-way rule. <b>Clean-learning note</b>: every LT must pass a gate -- in the 20 whisker trials "
              "before vs after, whisker lick rate OR whisker-FA discrimination changes in the learned direction by &gt; 0.05 with "
              "posterior P &gt; 0.9. The same gate passes only 20/47 R+ and 13/34 R- stored LTs.")
    if cnt is not None:
        body += table(cnt, "Sessions with an LT per rule, and passing the clean-separation gate.")
    body += p("Gate trade-off (R+/R- passing, of 34/17 lenient candidates): margin 0 -> 31/17; margin 0.05 -> 25/14 (P&gt;0.9, "
              "chosen) or 32/17 (P&gt;0.8); margin 0.10 -> 22/12. Some gated-in R+ sessions look marginal on inspection (AB127, "
              "AB104, MH025, AB145, MH068).")
    body += img(LTE / "010_review_grid_lenient_Rplus.png", "R+: lenient LTs (orange; solid = passes the clean gate, dashed = "
                "fails), stored LT (red), gate windows shaded.")
    body += img(LTE / "010_review_grid_lenient_Rminus.png", "R-: same.")
    return sec("14. Less conservative learning trials (Q1, Q2, clean separation)", "lt-lenient", body)


def s_aligned() -> str:
    body = img(LTE / "009_aligned_performance.png", "Whisker hit rate (cohort colour), FA (grey) and auditory hit rate (blue) "
               "aligned to the stored (left) vs new clean-gated LT (right). New LTs give sharp steps (R+ 0.37 -> 0.89, R- 0.66 -> "
               "0.11), partly by construction (a change point sits on the step). The R+ high state decays over ~40 trials.")
    return sec("15. Performance aligned to the new learning trials", "aligned", body)


def s_newlt() -> str:
    d = csv(EA / "100_new_learning_trial_stats.csv")
    body = p("Same pipeline (095, disengaged block dropped), learning trial replaced by the strict change point (lt_recommended) "
             "or strict-else-half-way (lt_recommended_broad).")
    if d is not None:
        d = d[(d.scope == "all")]
        body += table(d[["version", "window", "reward_group", "n", "mean_pre", "mean_post", "p_wilcoxon", "p_t", "p_mannwhitney",
                         "p_welch"]], "Matched accuracy minus shift null, pre vs post; R+ vs R- rows test the change.")
    body += img(FIG / "whole_brain/learning/100_hitmiss_newlt_whole_brain_all.png", "Stored vs strict vs broad new LT (whole brain).")
    body += h3("Lenient clean-gated LT, disengaged trials kept")
    s2 = csv(EA / "101_stored_vs_newlt_nodrop_stats.csv")
    if s2 is not None:
        body += table(s2[s2.scope == "all"].drop(columns=["scope"]), "Stored vs new lenient clean LT, disengaged trials kept.")
    body += img(FIG / "whole_brain/learning/101_hitmiss_stored_vs_newlt_nodrop_whole_brain_all.png",
                "Stored vs new lenient clean LT, disengaged trials kept.")
    return sec("16. Decoding with the new learning trials", "newlt", body)


def s_nonlearners() -> str:
    d = csv(EA / "101_learners_vs_nonlearners_stats.csv")
    body = p("Learners = sessions with a clean new LT; non-learners = all other decodable sessions (no identifiable clean learning "
             "event). Non-learners have no LT, so they are split at the session midpoint; learners are shown at both their LT and "
             "the midpoint. Disengaged trials kept.")
    if d is not None:
        body += table(d[d.scope == "all"][["window", "reward_group", "n_L", "n_NL", "mean_d_L_lt", "mean_d_L_half", "mean_d_NL_half",
                                           "p_prepost_L_lt_w", "p_prepost_L_half_w", "p_prepost_NL_half_w", "p_Llt_vs_Lhalf_w",
                                           "p_Llt_vs_NLhalf_mw", "p_Lhalf_vs_NLhalf_mw"]],
                      "Change (post - pre) by group; p-values: pre vs post within group, and between groups.")
    body += img(FIG / "whole_brain/learning/101_hitmiss_learners_vs_nonlearners_whole_brain_all.png",
                "Learners at the new LT and at the midpoint, non-learners at the midpoint.")
    return sec("17. Learners vs non-learners", "nonlearners", body)


def s_placebo_new() -> str:
    d = csv(EA / "098_learning_trial_placebo_lt_lenient_clean_nodrop_stats.csv")
    body = p("Placebo test re-run with the new lenient clean-gated LT, disengaged trials kept.")
    if d is not None:
        body += table(d[d.scope == "all"].drop(columns=["scope"]), "Placebo test, new LT, all sessions.")
    body += img(FIG / "whole_brain/learning/098_hitmiss_lt_placebo_lt_lenient_clean_nodrop_whole_brain_summary_all.png",
                "Placebo test with the new learning trials.")
    return sec("18. Placebo-split test with the new learning trials", "placebo-new", body)



FA_CODE = """def interp_fa_curve(session_table, far_df, return_samples=False):
    \"\"\"False-alarm learning curve evaluated at whisker-trial times. Each no-stim estimate
    is placed at the start time of ITS OWN no-stim trial, then linearly interpolated at the
    whisker-trial start times (edge values held, no NaNs).\"\"\"
    p_mean_no_stim = np.asarray(far_df['p_mean'].values[0])
    t_nostim = session_table.loc[session_table.trial_type == 'no_stim_trial', 'start_time'].to_numpy()
    t_whisker = session_table.loc[session_table.trial_type == 'whisker_trial', 'start_time'].to_numpy()
    if len(t_nostim) != len(p_mean_no_stim):
        raise ValueError(f"{len(p_mean_no_stim)} FA estimates but {len(t_nostim)} no-stim trials")
    order = np.argsort(t_nostim)
    t_nostim = t_nostim[order]
    interp_p_far = np.interp(t_whisker, t_nostim, p_mean_no_stim[order])
    if not return_samples:
        return interp_p_far
    fa_samples = np.asarray(far_df['p_samples'].values[0])[:, order]
    return interp_p_far, np.stack([np.interp(t_whisker, t_nostim, s_) for s_ in fa_samples])"""

PYMC_CODE = """import numpy as np, pymc as pm, pytensor.tensor as pt, arviz as az

def fit_learning_curve(outcomes, sigma_scale=0.5, draws=1000, tune=1000, chains=4, seed=0, conf_int=80):
    y = np.asarray(outcomes, dtype=int); n = len(y)
    with pm.Model():
        sigma = pm.HalfNormal("sigma", sigma=sigma_scale)       # step size learned from the data
        x0 = pm.Normal("x0", mu=0.0, sigma=1.5)                  # sensible initial logit
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=n - 1)       # non-centred random walk
        x = pt.concatenate([pt.stack([x0]), x0 + pt.cumsum(sigma * z)])
        pm.Deterministic("p", pm.math.invlogit(x))
        pm.Bernoulli("obs", logit_p=x, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.95, random_seed=seed)
    summ = az.summary(idata, var_names=["sigma", "p"], kind="diagnostics")
    n_div = int(idata.sample_stats["diverging"].sum())
    if summ["r_hat"].max() > 1.01 or summ["ess_bulk"].min() < 400 or n_div > 0:
        print("WARNING: not converged")
    p_samples = idata.posterior["p"].values.reshape(-1, n)
    lo, hi = (100 - conf_int) / 2, 100 - (100 - conf_int) / 2
    p_low, p_high = np.percentile(p_samples, [lo, hi], axis=0)
    return dict(p_samples=p_samples, p_mean=p_samples.mean(0), p_low=p_low, p_high=p_high)"""


def s_codefix() -> str:
    body = h3("False-alarm rate at whisker-trial times")
    body += p("Both the stored pipeline and the re-estimation evaluate the FA curve at whisker-trial times. The difference is "
              "where each no-stim estimate is placed in time first: <code>learning_utils.interp_fa_curve</code> puts the k-th "
              "estimate at the k-th point of an <i>evenly spaced</i> grid (<code>np.linspace</code>) instead of at the k-th "
              "no-stim trial's real time. Across the 88 sessions (~1 h each) this misplaces the estimates by a median 1.1 min "
              "per session (worst trial: median 3.0 min, max 36 min); MH028: median 5.7 min. The resulting FA curve at whisker "
              "times differs by up to 0.29 (median of per-session maxima, R+). The cubic spline with "
              "<code>extrapolate=False</code> also returns NaN for whisker trials outside the no-stim range. Corrected version "
              "(same call site; fitted FA curves need not be refitted):")
    body += f"<pre><code>{html.escape(FA_CODE)}</code></pre>"
    body += h3("Fitting the curve with PyMC: what was wrong, and a corrected version")
    body += ul([
        "<b>Unconverged sampler</b>: <code>pm.sample(10, tune=10, chains=4)</code> -- NUTS needs ~1000 tuning steps to adapt; "
        "40 autocorrelated samples gave R-hat 1.4-2.0 and ESS 12-17. The mean curve was roughly right (MAE ~0.015) but the 80% "
        "band, which the LT rule uses, was noisy.",
        "<b>Prior forcing jumpy curves</b>: tau ~ Gamma(10,10) pins sigma at ~1 logit/trial; the data prefer 0.49 (R+) / 0.14 "
        "(R-) (Bayes factor &gt; 20:1 in 30/88 sessions, never the reverse).",
        "<b>Geometry and defaults</b>: centred random walk with a learned sigma (funnel -&gt; divergences); initial state "
        "Normal(0, 100) by PyMC default.",
    ])
    body += f"<pre><code>{html.escape(PYMC_CODE)}</code></pre>"
    body += h3("Model formulation, and the grid (exact) alternative")
    body += p("The model is unchanged -- a hidden Markov model with a <b>continuous</b> hidden state (state-space model): "
              "x<sub>1</sub> ~ N(0, 100&sup2;); x<sub>t</sub> | x<sub>t-1</sub> ~ N(x<sub>t-1</sub>, &sigma;&sup2;); "
              "y<sub>t</sub> | x<sub>t</sub> ~ Bernoulli(logistic(x<sub>t</sub>)); &sigma; with a prior (stored: 1/&sigma;&sup2; "
              "~ Gamma(10,10); re-estimation: log-uniform on [0.02, 1.5]). MCMC and the grid are two ways of computing the "
              "same posterior p(x<sub>t</sub> | all trials).")
    body += ul([
        "<b>Grid</b> (<code>lt_lib.fit_curve</code>): discretise x on 181 points in [-9, 9] (0.1 logit), Gaussian transition "
        "matrix, Bernoulli emissions, forward-backward recursions -&gt; exact smoothed posterior of every trial; the product of "
        "forward normalisers is p(y | &sigma;), used to weight 36 &sigma; values (integration over &sigma;, and Bayes "
        "factors). Agrees with converged PyMC within 0.005.",
        "<b>Not to be confused with</b> the learning-trial change-point model: a <i>discrete</i> left-to-right HMM with 2-3 "
        "states (R+: naive -&gt; learned -&gt; optional decline; R-: generalising -&gt; learned) with constant whisker and FA "
        "rates per state; the LT is the entry into the learned state.",
    ])
    body += h3("Which to use: MCMC or grid?")
    body += table(pd.DataFrame([
        ("Exactness", "exact up to grid spacing", "approximate; only valid once converged"),
        ("Reproducibility", "deterministic", "seed-dependent"),
        ("Speed (per curve)", "~1 s", "tens of seconds (converged NUTS)"),
        ("Model evidence / Bayes factors", "free (forward normalisers)", "hard / expensive"),
        ("Hidden dimensions", "1-2 only (cost grows as grid^dims)", "many"),
        ("Whole-trajectory samples", "via forward-filter backward-sample (extra step)", "native"),
        ("Joint whisker+FA, hierarchical across mice, covariates", "impractical", "the right tool"),
    ], columns=["", "Grid", "MCMC (PyMC, configured as above)"]),
        "Recommendation: grid for the per-session curves and learning trials now; PyMC if you move to joint / hierarchical "
        "models (then use the grid results to check it on the per-session model).")
    return sec("19. Fixing the learning-curve code: FA interpolation, PyMC, grid vs MCMC", "codefix", body)


def s_caveats() -> str:
    body = ul([
        "Exploratory throughout: no confirmatory split; many windows x scopes x LT versions; p-values uncorrected across them.",
        "New-LT learners are selected as clear transitions (small n: 14/13 strict, 25/14 lenient); selection could correlate with "
        "decodability.",
        "R+ and R- learning trials mean different things (licking onset vs offset); 'post' is not the same event across cohorts.",
        "Gradual R- learners are ambiguous for step models (e.g. AB085: change point 141 vs half-way 77).",
        "Pre epochs are small (min 2 trials per class); R- sessions have few hits.",
    ])
    body += h3("Decisions for you")
    body += ul([
        "Adopt the new LT (which version: strict / lenient-clean; gate thresholds) and the exact curves as the default?",
        "Port the curve/LT fixes into <code>behaviour_analysis</code> (sampler, prior, FA interpolation, change-point LT)?",
        "Extend to area-level decoding and expert sessions with the chosen LT?",
    ])
    return sec("20. Caveats and open decisions", "caveats", body)


def s_files() -> str:
    body = ul([
        "Decoding project: <code>projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/</code> 089-102; "
        "figures under <code>figures/&lt;scheme&gt;/learning/</code>.",
        "Learning-trial project: <code>projects/ssl-learning-trial-identification/</code> (<code>SUMMARY.md</code>, "
        "<code>exploratory-analyses/000-010</code>, <code>lt_lib.py</code>, table <code>artifacts/007_learning_trials_v2.csv</code>).",
        "Shared library additions: <code>scripts/ssl_timeresolved_decoding.py</code> (reconstruct_learning_trial, "
        "prep_hitmiss_trials_learning_split, detect_terminal_disengagement).",
    ])
    return sec("21. Files", "files", body)


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#5f6368;--line:#e2e2e2;--acc:#2c5f5b;--pend:#b3261e;--card:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#a0a0a0;--line:#333;--acc:#7fc4bd;--pend:#ff8a80;--card:#1f1f1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:1.7em;margin:.2em 0}h2{font-size:1.3em;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2.2em;color:var(--acc)}
h3{font-size:1.05em;margin-top:1.4em}
nav{columns:2;font-size:.9em;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px}
nav a{display:block;color:var(--acc);text-decoration:none;padding:1px 0}
figure{margin:18px 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px}
figure img{display:block;margin:0 auto;max-width:100%;height:auto}
figcaption{color:#333;font-size:.88em;margin-top:6px}.src{color:#888;font-size:.85em}
.tw{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;font-size:.82em;min-width:50%}
th,td{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}th{background:var(--card)}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:.95em;padding:4px 0}
table.recap td{white-space:normal}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto;font-size:.8em;line-height:1.4}.pending{color:var(--pend);font-style:italic}code{font-size:.9em}
.meta{color:var(--muted);font-size:.9em}
"""


def main():
    sections = [s_recap(), s_summary(None, None, None), s_methods(), s_ltcheck(), s_decoding(), s_examples(), s_improvements(),
                s_disengagement(), s_halves(), s_placebo(), s_lt_review(), s_lt_how(), s_lt_process(), s_lt_lenient(), s_aligned(),
                s_newlt(), s_nonlearners(), s_placebo_new(), s_codefix(), s_caveats(), s_files()]
    nav = "".join(f'<a href="#{s.split(chr(34))[1]}">{s.split("<h2>")[1].split("</h2>")[0]}</a>' for s in sections)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
           f"initial-scale=1'><title>Learning-trial decoding report</title><style>{CSS}</style></head><body><main>"
           f"<h1>Hit/miss decoding around the learning trial, and learning-trial re-identification</h1>"
           f"<p class='meta'>SSL whisker/auditory Go/NoGo, KS4, learning stage, whole brain unless noted. Generated "
           f"{datetime.now():%Y-%m-%d %H:%M} by <code>102_build_learning_trial_report.py</code>. Private, unpublished data -- "
           f"local file.</p><nav>{nav}</nav>{''.join(sections)}</main></body></html>")
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), {_fig_counter[0]} figures")


if __name__ == "__main__":
    main()
