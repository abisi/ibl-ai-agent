"""014 -- Self-contained HTML report of the learning-trial identification project: every
method tried so far, schematised and exemplified (user request 2026-09-25). Reuses the
figure/table helpers of the decoding report builder (102). Tables are read from the CSVs.
Output: projects/ssl-learning-trial-identification/report.html (local; SSL data are private).
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
ART = PROJ / "artifacts"
DEC = PROJ.parent / "ssl-whisker-hitmiss-timeresolved-decoding"
_s = importlib.util.spec_from_file_location("r102", DEC / "exploratory-analyses" / "102_build_learning_trial_report.py")
R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)
img, table, csv, sec, p, ul, h3 = R.img, R.table, R.csv, R.sec, R.p, R.ul, R.h3
OUT = PROJ / "report.html"
F = HERE


def s_question():
    return sec("1. Question and scope", "question", ul([
        "Is the stored learning trial (LT) well placed on the whisker learning curves? If not, how can its identification "
        "be improved, keeping the original learning-curve model (random walk on the logit of lick probability)?",
        "Data: the 88 ephys learning-stage (day 0) sessions with stored curves (50 R+, 38 R-; MH038 excluded, trial mismatch); "
        "all 98 stored curves for the inventory. Whisker and no-stim (false-alarm, FA) licks of the active epoch.",
        "Everything here is exploratory. Figure/script numbers refer to <code>exploratory-analyses/NNN_*</code>.",
    ]))


def s_overview():
    b = img(F / "013_pipeline_schematic.png", "Pipeline, stored vs re-estimated. The curve model is unchanged; the "
            "computation, smoothness, FA timing, rule and gate change.")
    b += p("<b>One-line summary.</b> The stored LT often sits on a transient excursion, a hard-coded 10, or a late blip. Causes: "
           "an unconverged sampler, a prior that forces jumpy curves, a mistimed FA line, and a rule that cannot say 'no "
           "learning event'. The re-estimated curves fix the first three; the LT is then read either from the smoothed "
           "whisker-minus-FA curve (half-way rule) or from a change point on raw licks, with a behavioral gate.")
    return sec("2. Overview", "overview", b)


def s_stored():
    inv = csv(ART / "000_inventory.csv")
    b = h3("What the stored pipeline does")
    b += ul([
        "Whisker and no-stim curves: PyMC random walk on logit p, <code>pm.sample(10, tune=10, chains=4)</code>, prior "
        "1/&sigma;&sup2; ~ Gamma(10,10).",
        "FA curve read at whisker times after being placed on an evenly spaced time grid (<code>interp_fa_curve</code>).",
        "R+: LT = start of the first run of 5 whisker trials whose 80% band lower edge exceeds the FA line ('expert' if the "
        "first 20 all qualify with mean p &ge; 0.8). R-: LT = last trial of the first such 5-run followed by 5 not-above "
        "trials. Only trials after the first whisker hit count. LT &le; 10 -&gt; 10; R- without criterion -&gt; 10.",
    ])
    b += h3("Diagnosis")
    if inv is not None:
        s = inv[inv.sustain_20.notna()]
        b += p(f"The LT criterion holds on fewer than half of the 20 trials after the LT in {(s.sustain_20 < 0.5).sum()}/"
               f"{len(s)} curves (R+ mean {s[s.reward_group == 1].sustain_20.mean():.2f}, R- "
               f"{s[s.reward_group == 0].sustain_20.mean():.2f}). 18/47 R+ and 6/34 R- ephys LTs are exactly 10. Several R- "
               "LTs fall on isolated late blips in mice that never licked the whisker above FA (AB093 118, AB139 117, AB140 173).")
    b += img(F / "000_all_curves.png", "All 98 stored curves; red = stored LT, blue dashed = model-free change point; 'sus' = "
             "fraction of the next 20 trials where the stored criterion still holds.")
    return sec("3. The stored learning trial and what goes wrong", "stored", b)


def s_curves():
    b = h3("Model (unchanged) and how it is computed")
    b += p("Hidden state x<sub>t</sub> = logit of lick probability; x<sub>t</sub> = x<sub>t-1</sub> + N(0, &sigma;&sup2;); lick "
           "y<sub>t</sub> ~ Bernoulli(logistic(x<sub>t</sub>)); fitted separately for whisker and for no-stim trials. This is a "
           "hidden Markov model with a continuous state. The stored fit samples it with MCMC (10 tune / 10 draws: R-hat 1.4-2.0, "
           "ESS 12-17). The re-estimation computes the same posterior exactly: x discretised on 181 values (&plusmn;9 logit), "
           "Gaussian transition matrix, forward-backward recursions, &sigma; integrated over 36 values weighted by the evidence "
           "p(licks | &sigma;) (<code>lt_lib.fit_curve</code>, ~1 s per curve, matches converged PyMC within 0.005).")
    val = csv(F / "001c_validation.csv")
    if val is not None:
        b += table(val, "Stored sampler setting vs converged PyMC, both against the exact grid posterior.")
    b += img(F / "001c_validation.png", "Exact grid (black) vs PyMC 10x10 (red) and 1000x1000 (blue).", width="80%")
    b += h3("Smoothness (&sigma;): per session, shared, or a readout?")
    ev = csv(ART / "012_sigma_evidence.csv")
    b += p("Per session, the data choose &sigma; (evidence-weighted): median 0.49 R+ / 0.14 R- (whisker), 0.23 / 0.15 (FA), "
           "against ~1 imposed by the stored prior; the smoother model is preferred &gt; 20:1 in 30/88 sessions. A single shared "
           "&sigma; costs fit (table). R+ whisker curves are ~3.5x jumpier than R- (p = 3e-7), but &sigma; also rises with lick "
           "rate (r = 0.36): on the logit scale, licks near p = 0 carry little information about change. Treat the cohort "
           "difference as descriptive unless rate-controlled. Recommended: &sigma; per session (or per cohort if one value is "
           "wanted). The change-point LT does not depend on &sigma; (it uses raw licks).")
    if ev is not None:
        b += table(ev, "Best shared &sigma; (maximising summed evidence) and the log-evidence lost vs per-session &sigma;.")
    b += img(F / "012_sigma.png", "Per-session &sigma; by cohort (whisker, FA), &sigma; vs lick rate, and summed evidence for a "
             "shared &sigma;.")
    b += h3("False-alarm curve at whisker times")
    b += p("Both pipelines evaluate the FA curve at whisker-trial times; the stored one first places the k-th no-stim estimate "
           "at the k-th point of an evenly spaced grid instead of at the k-th no-stim trial's real time (median misplacement "
           "1.1 min per session, up to 36 min; FA line off by up to 0.29). Re-estimation places each estimate at its own trial "
           "time (linear interpolation, edges held). Code fix in the decoding report, section 19.")
    b += img(F / "009_population_comparison.png", "Population: (a) &sigma; stored prior vs data, (b) evidence for smoother "
             "curves, (c) FA timing error, (d) stored vs new LT, (e) LTs per rule and passing the gate, (f) FA early vs mid.")
    return sec("4. Re-estimating the learning curves", "curves", b)


def s_process():
    b = img(F / "008_process_examples.png", "Step by step on 6 sessions: A stored curve and 'above chance' trials; B same model "
            "computed exactly; C data-chosen smoothness + FA at real times; D per-trial P(whisker &gt; FA); E learning-trial "
            "posteriors and candidates.")
    return sec("5. The re-estimation, step by step", "process", b)


METHODS = [
    ("L0", "Stored", "Stored H5 value (see section 3).", "baseline", "transient / clamped / arbitrary placements"),
    ("L1", "Stored rule, exact curve", "Stored run rule on the exact posterior of the same model (stored prior), FA at real "
     "times.", "separates sampler noise from rule problems", "still 18 R+ LTs at 10: the rule, not the sampler, is the main problem"),
    ("L2", "Stored rule, smooth curve", "Stored run rule on the data-chosen-smoothness curve.", "smoother input",
     "R- almost never matches the 'first run then 5 not-above' pattern (13/38, mostly clamped)"),
    ("L3", "Sustained probability", "P(whisker &gt; FA) &ge; 0.9 on 16 of 20 trials (R+); R- after that, &le; 0.5 on 16/20.",
     "uses the full posterior, sustained", "absolute criterion: early generalised separation counts as learned; R- 8/38"),
    ("L4", "Whisker change point (ML)", "Best 2-3-segment split of whisker outcomes only.", "model-free", "ignores FA"),
    ("L5", "Whisker change point (Bayes)", "Posterior over the split, whisker outcomes only, Bayes factor vs no change.",
     "uncertainty + 'no change'", "misses R+ mice whose whisker licking is already high and whose FA drops"),
    ("L6", "Joint change point", "Segments with constant whisker AND FA rates; R+ naive -&gt; learned (higher whisker-FA) "
     "-&gt; optional decline; R- generalising -&gt; learned (discrimination at least halves). Posterior median, 90% CI, "
     "log10 BF &gt; 0.5; categories.", "uses FA; CI; 'no learning event'", "assumes a step; gradual learners get the "
     "sharpest drop (AB085 141 vs half-way 77)"),
    ("L6 lenient", "Joint change point, lenient", "Same, log10 BF &gt; 0, relaxed checks.", "more learners", "weaker evidence"),
    ("L5w lenient", "R+ whisker rise, FA as floor", "Whisker-only change point with the learned whisker rate above FA.",
     "R+ FA stays high: do not require FA to change", "R+ only; mixes definitions"),
    ("L7", "Half-way on the smoothed curves", "D = smoothed whisker - smoothed FA. R+: first trial where D crosses half-way "
     "between its start (first 10 trials) and plateau (90th pct), held 16/20; R-: half-way between the early peak and the "
     "floor, held 16/20. No LT if the change is &lt; 0.2.", "simple, one rule mirrored by cohort, relative to each session's "
     "own start/plateau (t50-like)", "no CI/BF; depends on smoothing"),
    ("L8", "Fixed margin on the smoothed curves", "R+: first rise of D above 0.2 held 16/20 (after having been below); R-: "
     "first fall below 0.1 held 16/20 (after having been above 0.2).", "absolute separation", "R+ early (early modest "
     "separation crosses first: AB119 12 vs 47); R- late (end of a gradual decline: +26 trials median)"),
    ("cascade", "Lenient cascade + clean gate", "First of L6, L6 lenient, L5w lenient, L7; kept if 20 whisker trials after "
     "vs before differ in the learned direction by &gt; 0.05 in whisker rate OR whisker-FA (P &gt; 0.9).", "most learners with "
     "a clean behavioral split", "mixes definitions across sessions"),
]


def s_catalogue():
    b = p("All methods tried, what each does, why it was tried, and its main weakness. The clean-separation gate can be applied "
          "on top of any of them.")
    b += table(pd.DataFrame(METHODS, columns=["id", "method", "definition", "motivation", "weakness"]), "Catalogue of learning-trial methods.")
    b += h3("Schematic: every rule on the same two synthetic sessions")
    b += p("Synthetic R+: early modest bump (trials 15-35), true learning at trial 60, end-of-session decline. Synthetic R-: "
           "generalising until 40, gradual decline 40-120 (half-way ~80). Dotted black = the true probabilities.")
    b += img(F / "013_rules_schematic.png", "A stored rule (yellow = band edge above FA); B sustained probability; C fixed margin "
             "on D; D half-way on D; E joint change point (cumulative licks, posterior). R+: stored rule 22 (early bump), "
             "sustained 49, fixed margin 47, half-way 57, change point 59 [53, 62] (truth 60). R-: stored 54, sustained none, fixed "
             "margin 116 (end of decline), half-way 71, change point 71 [51, 76] (truth: decline 40-120).")
    b += h3("Real examples: every rule on 8 sessions")
    b += img(F / "013_rules_examples.png", "Smoothed whisker (80% band) and FA curves, D = whisker - FA (dash-dot), stored curve "
             "(thin); vertical lines = every rule's LT (values listed right; 'none' = no LT).")
    b += h3("Every method on every session")
    b += p("One panel per session: smoothed whisker curve (80% band; thin = stored curve), FA at real trial times (dashed), "
           "D = whisker - FA (dash-dot), whisker licks (ticks), one vertical line per method. The strip under each panel puts "
           "each method on its own row (L0 top ... gate bottom) so coinciding learning trials stay visible. Sessions are "
           "sorted by disagreement between methods (largest spread first); sessions where at most one method finds a "
           "learning event come last -- mostly the stored rule forcing an LT on flat or never-licked sessions.")
    b += img(F / "015_all_methods_Rplus.png", "R+: all 50 sessions, all methods.")
    b += img(F / "015_all_methods_Rminus.png", "R-: all 38 sessions, all methods.")
    sm = csv(ART / "013_methods_summary.csv")
    if sm is not None:
        b += table(sm, "All methods, all sessions: sessions with an LT, median LT, LTs at 10, LTs passing the clean gate, "
                       "contrast20 = whisker-FA lick rate 20 trials after minus before (R+ &gt; 0, R- &lt; 0 expected), held_post = "
                       "fraction of later windows on the learned side.")
    b += p("Per-session values of every method: <code>artifacts/013_learning_trials_all_methods.csv</code>.")
    b += img(F / "003_rule_summary.png", "First rule comparison (L0-L4) on model-free behavior.")
    return sec("6. Learning-trial methods: catalogue, schematic, examples", "catalogue", b)


def s_cp():
    b = p("<b>Change point (L6).</b> Behavior goes through a few states in order, never back, each with a constant whisker lick "
          "rate and FA lick rate. Every whisker trial k is a candidate start of the learned state (segments &ge; 5 whisker "
          "trials; no-stim trials assigned by time). For each candidate, the unknown rates of each segment are integrated out "
          "(Beta(1,1) priors: a product of Beta functions of lick / no-lick counts), giving a score; with a uniform prior over "
          "candidates, scores become a posterior over k. LT = posterior median, 90% CI = 5th-95th percentiles; Bayes factor = "
          "average score of 'change' vs 'no change' models. Checks assign the category (R+: learned whisker &gt; FA; R-: initial "
          "whisker &gt; FA, discrimination at least halves).")
    b += img(F / "012_changepoint_explained.png", "1 raw licks and the best segmentation (segment rates); 2 cumulative lick "
             "counts: the LT is where the whisker slope bends relative to the FA slope; 3 score of every candidate LT and the "
             "posterior; 4 the half-way rule on the smoothed curves, same session.")
    b += h3("Half-way vs fixed margin on the separation curve")
    b += p("Both use the separation D = whisker - FA. A fixed margin times the LT by when D passes an arbitrary level, which "
           "depends on each mouse's baseline and ceiling: R+ mice with some early generalised separation cross it too early "
           "(AB119: 12 vs 47; MH070: 13 vs 39); R- mice cross 'below 0.1' only at the end of a gradual decline (+26 trials "
           "median vs half-way; AB085 181 vs 77). Half-way measures each session against its own start and plateau (the "
           "'trial to 50% of learning'), while the size of the change (&ge; 0.2) decides whether there is a learning event.")
    b += img(F / "012_cp_vs_curve.png", "Change point (with 90% CI) vs half-way curve rule, all sessions: median difference 1 "
             "trial (R+), 3 trials (R-); the curve rule defines more sessions.")
    return sec("7. Change point and curve rules explained", "cp", b)


def s_lenient():
    cnt = csv(F / "007_counts.csv")
    fa = csv(F / "007_fa_behavior.csv")
    b = h3("Should FA be used (R+ FA higher)?")
    if fa is not None:
        b += table(fa.groupby("reward_group")[["fa_early", "fa_mid", "wh_early", "wh_mid"]].mean().reset_index(),
                   "Lick rates, first 20% vs middle 20-80% of whisker trials.")
    b += p("R+ FA stays high while R- FA falls. For R-, FA is essential (else a general licking decline would count as "
           "learning); for R+, requiring FA to change under-detects learners, so FA was used as a floor there (L5w lenient).")
    b += h3("More learners, with a clean behavioral separation")
    b += p("Cascade L6 -&gt; L6 lenient -&gt; L5w lenient -&gt; L7, then the clean-separation gate (20 trials either side, change "
           "&gt; 0.05 in whisker rate OR whisker-FA, P &gt; 0.9). The same gate passes only 20/47 R+ and 13/34 R- stored LTs. "
           "Note: the cascade mixes definitions; a single rule (L6 lenient or L7) with the gate costs few sessions.")
    if cnt is not None:
        b += table(cnt, "Sessions with an LT per rule and passing the gate.")
    b += img(F / "010_review_grid_lenient_Rplus.png", "R+: lenient LTs (orange: solid passes the gate, dashed fails), stored LT "
             "(red), gate windows shaded.")
    b += img(F / "010_review_grid_lenient_Rminus.png", "R-: same.")
    return sec("8. Less conservative learning trials and the clean-separation gate", "lenient", b)


def s_review():
    b = img(F / "006_final_review_Rplus.png", "R+: every session, re-estimated curve, stored LT (red), change point (blue, CI), "
            "half-way (orange).")
    b += img(F / "006_final_review_Rminus.png", "R-: same.")
    b += img(F / "009_aligned_performance.png", "Whisker, FA and auditory licking aligned to the stored vs new (lenient, gated) "
             "LT. New LTs give sharp steps (R+ 0.37 -&gt; 0.89, R- 0.66 -&gt; 0.11), partly by construction.")
    return sec("9. All sessions and aligned behavior", "review", b)


def s_impact():
    d = csv(DEC / "exploratory-analyses" / "100_new_learning_trial_stats.csv")
    b = p("Hit/miss decoding (whole brain, sensory window, size-matched, shift null) pre vs post LT; full details in the "
          "decoding report (<code>ssl-whisker-hitmiss-timeresolved-decoding/report_learning_trial_split.html</code>).")
    if d is not None:
        d = d[(d.scope == "all") & (d.window.isin(["sensory", "sensory_minus_base"]))]
        b += table(d[["version", "window", "reward_group", "n", "mean_pre", "mean_post", "p_wilcoxon", "p_t", "p_mannwhitney",
                      "p_welch"]], "Stored vs re-identified LT (strict L6; broad = L6 else L7).")
    b += p("With the strict change-point LT, R+ sensory decoding rises (0.065 -&gt; 0.177, p = 0.003, n = 14), masked with the "
           "stored LT; the R- drop is kept. The placebo test with the new LT shows the R+ change is specific to the LT.")
    return sec("10. Impact on the hit/miss decoding", "impact", b)


def s_appendix():
    s = csv(F / "011_joint_model_summary.csv")
    b = p("<i>Explored, not adopted (user decision: keep the original model).</i> Joint model over all trials: logit P(FA lick) = "
          "g (general lick propensity), logit P(whisker lick) = g + d (whisker-specific drive), g and d random walks, exact 2-D "
          "grid; and a switching version where d jumps once. Learning in logit d treats co-moving whisker and FA near 0/1 as "
          "general licking (MH028, AB085); on the probability scale it agrees with L6 where learning is clear. The single-jump "
          "version was weak in 4/6 sessions.")
    if s is not None:
        b += table(s[["session_id", "reward_group", "stored_lt", "L6", "lt_cont", "lt_cont_prob", "lt_switch", "switch_log10_bf"]],
                   "Joint-model LTs on 6 sessions.")
    b += img(F / "011_joint_model_test.png", "Joint-model test (panels A-E per session).")
    b += img(F / "011b_joint_pymc_check.png", "Grid vs PyMC for the joint continuous model.", width="80%")
    return sec("A. Appendix: joint whisker + FA state-space model (not adopted)", "appendix", b)


def s_decisions():
    b = ul([
        "<b>Curves</b>: exact computation of the original model, &sigma; per session (or per cohort), FA at real trial times.",
        "<b>Learning trial, one rule mirrored by cohort</b>: either the half-way rule on the smoothed whisker-FA curve (L7; "
        "simple, curve-based) or the joint change point (L6; uncertainty and a 'no change' test). They agree within 1-3 trials "
        "where both apply.",
        "<b>Gate</b>: clean-separation gate on top; sessions without a clean event are 'no learning event' (not assigned 10).",
        "<b>Open</b>: which rule; gate thresholds; port to <code>behaviour_analysis</code>; expert sessions.",
    ])
    return sec("11. Recommendations and open decisions", "decisions", b)


def s_files():
    return sec("12. Files", "files", ul([
        "<code>exploratory-analyses/lt_lib.py</code> exact curve fits; <code>lt_joint.py</code> joint model (appendix).",
        "000 inventory; 001a inputs, 001c PyMC validation; 002 exact curves; 003 rules L0-L4; 004 whisker CP; 005 joint CP; "
        "006 half-way + table; 007 lenient + gate + FA; 008 process; 009 population + aligned; 010 lenient grids; 011 joint "
        "model; 012 smoothness + CP explained; 013 schematics + all methods; 014 this report; 015 all methods x all sessions.",
        "Tables: <code>artifacts/013_learning_trials_all_methods.csv</code> (every method, every session), "
        "<code>007_learning_trials_v2.csv</code>, <code>006_learning_trials_final.csv</code>.",
    ]))


def main():
    sections = [s_question(), s_overview(), s_stored(), s_curves(), s_process(), s_catalogue(), s_cp(), s_lenient(),
                s_review(), s_impact(), s_decisions(), s_files(), s_appendix()]
    nav = "".join(f'<a href="#{s.split(chr(34))[1]}">{s.split("<h2>")[1].split("</h2>")[0]}</a>' for s in sections)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
           f"initial-scale=1'><title>Learning-trial identification</title><style>{R.CSS}</style></head><body><main>"
           f"<h1>Learning-trial identification: stored vs re-estimated, all methods</h1>"
           f"<p class='meta'>SSL whisker/auditory Go/NoGo, learning stage (day 0). Generated {datetime.now():%Y-%m-%d %H:%M} by "
           f"<code>014_build_report.py</code>. Private, unpublished data -- local file.</p><nav>{nav}</nav>{''.join(sections)}"
           f"</main></body></html>")
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), {R._fig_counter[0]} figures; pending: "
          f"{doc.count('class=' + chr(34) + 'pending')}")


if __name__ == "__main__":
    main()
