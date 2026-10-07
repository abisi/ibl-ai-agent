"""021 -- Index of every within-day (Part II) figure, grouped by question (user 2026-10-07: "organize all figures of today in
a better way"). Writes within_day/<ref>[_<units>]/FIGURES.md: one section per analysis (question, producing script,
figures per population with relative links), in reading order: summary figures, whole-brain analyses, per-area analyses,
controls and variants. Every PNG under the folder that is not listed in GROUPS is reported under "Other" so nothing is
missed. Result-bearing: written to the NAS only (not to the public code fork).
"""
import importlib
import pathlib
import sys
from datetime import datetime

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
m51 = importlib.import_module("051_roc_prelick")
W = m51.WITHIN

# (section, question, script, glob patterns relative to WITHIN)
GROUPS = [
    ("A. COSYNE figures (summaries)", None, None, []),
    ("A1. Learning day, v2 (scores + examples / relative to SL / relative to AH)", "Does whisker-hit pre-lick activity drift "
     "toward the rewarded-lick pattern within the learning day, in R+ but not R-?", "019", ["cosyne/*/COSYNE_day0_v2.png"]),
    ("A2. Learning day, v1 (matrix or dot-plot area panel; all areas or strict set)", "Same question, earlier layout.", "018",
     ["cosyne/*/COSYNE_day0*.png"]),
    ("A3. Convergence timeline (day 0 halves, expert, decoders)", "When does the across-day change appear?", "006 / 008",
     ["cosyne/COSYNE_convergence_timeline*.png"]),
    ("A4. R- within-day vs expert", "Do R- mice change more within day 0 than within expert sessions?", "005",
     ["cosyne/COSYNE_within_day.png"]),
    ("B. Whole-brain analyses", None, None, []),
    ("B1. Reward-lick CD within sessions (class trajectories, WH-SL, AH-WH, drifts, mixed model)", "How do WH, AH and SL "
     "scores move on the reward-lick axis across the session, day 0 vs expert?", "011", ["cd_trajectories/*/cd_trajectories.png"]),
    ("B2. Session halves (distances, lambda, decoders per half)", "Early vs late half of each session.", "001",
     ["halves/*/within_session_time.png"]),
    ("B3. PSTHs per session half, all four groups", "What does the activity look like per half?", "004 / 012",
     ["psth_halves/*/psth_halves_groups.png"]),
    ("B4. Trial-level trajectories and slopes", "Single-trial scores and slopes against session time.", "002",
     ["slopes/*/trial_slopes.png"]),
    ("B5. Within-session change, both cohorts (rates, per-half distance and position, change size)", "Is the within-session "
     "change larger on day 0 than in expert sessions?", "013", ["within_change/*/within_session_change.png"]),
    ("B6. Within vs across days on a common footing", "Within-day, across-day and carry-over changes.", "003",
     ["epochs/*/epoch_comparison.png"]),
    ("B7. Cohort specificity of every within-day measure", "Which cohort differences are specific to the learning day?",
     "014", ["cohort_specificity/*/cohort_specificity.png"]),
    ("B8. Day-0 effects per mouse vs expert outcome", "Do day-0 effects predict expert performance?", "016",
     ["day0_predicts_expert/*/day0_predicts_expert.png"]),
    ("B9. Cohorts contrasted within each stage (time-normalised)", "R+ vs R- on day 0 and in experts.", "023",
     ["cd_trajectories/*/cd_trajectories_by_stage.png"]),
    ("B10. Along the first 10 / 20 whisker hits, trial by trial", "Does the drift follow the number of whisker-hit "
     "outcomes?", "022", ["wh_trials/*/wh_trials_N*.png"]),
    ("B11. All drifts, compact (mean +- s.e.m.), whole brain and per area", "Summary of the drift quantifications.", "024",
     ["cosyne/*/drift_compact.png"]),
    ("C. Per-area analyses (CD built from each area group's units)", None, None, []),
    ("C1. Day 0 and expert, all area groups and strict set", "Which area groups carry the drift, per stage?", "017",
     ["area_cd_drift/*/area_cd_drift_day0*.png", "area_cd_drift/*/area_cd_drift_expert*.png"]),
    ("C2. Area x group matrices (with cohort x area and cohort x stage x area ANOVA)", "Overview across areas and stages.",
     "017", ["area_cd_drift/*/area_cd_drift_matrix.png"]),
    ("C3. Both stages per area (forest)", "Drift per area group, day 0 and expert together.", "015",
     ["area_cd_drift/*/area_cd_drift.png"]),
    ("C4. One figure per area group: trajectories and drifts, day 0 vs expert", "Learning-day vs expert contrast within "
     "each area group.", "020 / 023 (_by_stage: cohorts within stage)", ["area_cd_drift/*/per_area/*.png"]),
    ("D. Controls and variants", None, None, []),
    ("D1. Odd/even split and per-half controls", "Null split without time; rates, RTs, firing rates per half.", "001",
     ["halves/*/within_session_oddeven.png", "halves/*/within_session_controls.png"]),
    ("D2. Slopes with >= 4 / >= 8 whisker hits", "Robustness to the whisker-hit minimum.", "002",
     ["slopes_wh*/*/trial_slopes.png"]),
    ("D3. Epochs: 4 events per class and unit-sampling schemes", "Robustness of the common footing.", "003 / 010",
     ["epochs_n4*/*/epoch_comparison.png", "cosyne/decoder_schemes.png"]),
]


def main():
    seen, out = set(), []
    out.append(f"# Within-day figures ({W.name})\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} by within_day/021_figure_index.py. "
               "Populations: learners (main) and all mice. Code: projects/ssl-prelick-convergence/exploratory-analyses/within_day/.\n")
    for title, q, script, pats in GROUPS:
        if not pats:
            out.append(f"\n## {title}\n")
            continue
        files = sorted({f for p in pats for f in W.glob(p)} - seen)
        if not files:
            continue
        seen |= set(files)
        out.append(f"\n### {title}\n\n*{q}* Script {script}.\n")
        for f in files:
            rel = f.relative_to(W).as_posix()
            out.append(f"- [{rel}]({rel})")
    rest = sorted(set(W.rglob("*.png")) - seen)
    if rest:
        out.append("\n## Other figures\n")
        out += [f"- [{f.relative_to(W).as_posix()}]({f.relative_to(W).as_posix()})" for f in rest]
    (W / "FIGURES.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(seen)} figures indexed, {len(rest)} other -> {W / 'FIGURES.md'}")


if __name__ == "__main__":
    main()
