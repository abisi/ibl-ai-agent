"""Article-style report of ssl-prelick-convergence (skills/project-report): run on haas.

Main text: build_article.py in Markdown mode on the headline variant of this report (spontaneous-lick reference,
learners -- the variant re-run on 2026-10-05 with the per-trial context rule). Added here: the analysis variants
(locked set, archived iterations v1-v3, FA reference and all-mice population as controls, with a comparison table),
the projection-zone co-location null (ssl-sensory-spatial-maps 008), supplementary figures and an appendix.
Output: combined_results_ks4/ssl-prelick-convergence/report/ (report.md, numbers.json, figures/, build.sh).
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
SL, FA = RES / "ssl-prelick-convergence" / "across_days" / "sl", RES / "ssl-prelick-convergence" / "across_days" / "fa"
REP = RES / "ssl-prelick-convergence" / "report"
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN = ("sl", "learners")
VARIANTS = [("sl", "learners", "SL, learners (main)"), ("sl", "all", "SL, all mice"), ("fa", "all", "FA, all mice"),
            ("fa", "learners", "FA, learners")]
PANELS = [("2b", "WH-vs-reference selective units"), ("2c", "Converging neurons"), ("3c", "Distance difference Δd"),
          ("3d", "d(WH, reference)"), ("5a", "Decoder accuracy (AH vs reference)"), ("5d", "Decoder probability numerator")]
FIGS = []


def P(p):
    return "n/a" if p is None or not np.isfinite(p) else "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def pub(ref, pop):
    return (SL if ref == "sl" else FA) / "publication" / pop


def fig(src, name, caption, width="100%"):
    name = name.replace(".png", ".jpg")
    FIGS.append((pathlib.Path(src), name))
    return f"![{caption}](figures/{name}){{width={width}}}\n"


def variant_table():
    rows = []
    for panel, label in PANELS:
        r = {"Measure": label}
        for ref, pop, lab in VARIANTS:
            f = pub(ref, pop) / f"stats_{pop}.csv"
            if not f.exists():
                r[lab] = "n/a"; continue
            S = pd.read_csv(f)
            q = S[S.panel == panel]
            if not len(q):
                r[lab] = "n/a"; continue
            q = q.iloc[0]
            r[lab] = f"{P(q.get('R+ L vs E p_MWU'))} / {P(q.get('interaction_p_perm'))}"
        rows.append(r)
    T = pd.DataFrame(rows)
    cols = list(T.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in T.itertuples(index=False)]
    return "\n".join(out) + "\n", T


def main():
    REP.mkdir(parents=True, exist_ok=True)
    tmp = REP / "_main.md"
    env = dict(os.environ, ARTICLE_FMT="md", ARTICLE_OUT=str(tmp), ARTICLE_FIGPREFIX="")
    subprocess.run([sys.executable, str(HERE / "build_article.py"), str(pub(*MAIN)), MAIN[0], MAIN[1]], env=env, check=True)
    md = tmp.read_text(encoding="utf-8")
    tmp.unlink()
    for f in ["Fig1_task_data.png", "Fig2_single_neurons.png", "Fig2S_converging_neurons.png", "Fig3_population_distance.png",
              "Fig3S_lambda_lambdaLDA.png", "Fig4_areas.png", "Fig5_decoders.png"]:
        FIGS.append((pub(*MAIN) / f, f.replace(".png", ".jpg")))
    md = md.replace("(working draft)", "-- report generated " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    vt, VT = variant_table()
    colo = RES / "ssl-prelick-convergence" / "across_days" / "sl" / "projection_colocation"
    PT = pd.read_csv(colo / "prelick_colocation_tests_zone90.csv").set_index("group")
    PT7 = pd.read_csv(colo / "prelick_colocation_tests.csv").set_index("group")
    rec = (FA / "recap" / "recap.md").read_text(encoding="utf-8") if (FA / "recap" / "recap.md").exists() else ""
    rec_rows = [l for l in rec.split("\n") if l.startswith("| iteration") or l.startswith("|---") or "lambda | " in l and "| v" in l]
    md += f"""

# Analysis variants and controls

**Variant reported here.** The main text uses the spontaneous-lick (SL) reference and the *learners* population
(day-0 sessions of non-learning mice removed, their expert sessions kept), the variant re-run on 2026-10-05 after the
per-trial context rule (a trial inside a fixed ~3 s ITI sequence is passive, otherwise the context column) changed the
trial selection of three sessions (MH062, MH064, AB128). The locked analysis set (2026-10-03) names SL with all mice as
headline; the all-mice SL figures predate the context fix and are reported here as a control, together with the
false-alarm (FA) reference.

**Locked analysis set (v4).** Pre-lick window: 100 ms before the corrected first lick (start time + lick time − response-
window start). Trials: active, `perf` ≠ 6, warm-up cut (one trial before the first whisker trial kept), A1 disengagement
trim. Units: Kilosort 4 good + multi-unit, mean raw pre-lick rate ≥ 0.1 Hz. Area analyses: ≥ 5 units per session × area;
area groups kept per cohort if ≥ 3 sessions at both stages. Reference: SL (headline) or FA (control). Nulls: label
permutation for single-neuron ROC; linear shift of the neural activity against the time-ordered labels for single-session
decoders and pseudo-populations. Statistics: Mann-Whitney U and Welch per contrast, learning × cohort interaction by
mouse-level cohort permutation (10,000), area ANOVA with mouse-level permutation, no correction across panels.

**Archived iterations (not used; `combined_results_ks4/ssl-prelick-convergence/across_days/fa/_archive_*`).**

| Iteration | Unit selection and thresholds | Folder |
|---|---|---|
| v1 | min raw pre-lick rate 1 Hz, all units (no quality filter); λ ≥ 10 units, AH-FA axis ≥ 0.02; λ_LDA ≥ 10 units, d' ≥ 0.5; decoders ≥ 10 units, accuracy ≥ 0.6 (uncorrected) | `_archive_v1_minfr1Hz_allunits` |
| v2 | 0.1 Hz, good + mua; λ ≥ 10 units, axis ≥ 0.02; λ_LDA ≥ 5 units, d' ≥ 0.5; decoders ≥ 5 units, accuracy ≥ 0.55 (uncorrected) | `_archive_v2_minunits10_axisthr` |
| v3 | as v2, λ ≥ 5 units, no axis-reliability threshold | `_archive_v3_minunits5_noaxisthr` |
| (decoders) | label-permutation null for decoders, replaced by the linear-shift null | `_superseded_perm_null_decoders` |
| v4 FA / v4 SL (locked) | as v3, axis ≥ 0.01 (λ), d' ≥ 0.3 (λ_LDA), decoders chance-corrected by linear shift; FA or SL reference | `_roc_prelick/`, `_roc_prelick_sl/` |

**Table A.** Iterations of the analysis. The recap (070) recomputes the same statistics for every iteration:

{chr(10).join(rec_rows)}

**Comparison of the current variants.** For key measures: p of the R+ learning-vs-expert change (Mann-Whitney U) / p of the
learning × cohort interaction (mouse-level permutation).

{vt}
**Table B.** Key statistics across reference (SL, FA) and population (learners, all mice). Measures as in the main text
(panels 2b, 2c, 3c, 3d, 5a, 5d).

# Projection-zone co-location (null result)

**Motivation.** If modality-convergent pre-lick activity were inherited from converging cortical inputs, converging
neurons should be enriched where the projections of whisker cortex (SSp-bfd, SSs) and auditory cortex (AUDp, AUDd/v)
overlap (ssl-sensory-spatial-maps). **Method.** Converging neurons as above (SL reference, all mice; WH vs SL and AH vs SL
both significant, same sign); projection zones from Allen anterograde tracing (all projection-neuron lines averaged per
line); pooled fraction inside the overlap vs among all tested neurons, hierarchical bootstrap (sessions, then neurons;
2000 resamples, one-sided) and Fisher's exact test, per cohort × stage with Holm correction. **Results.** Converging neurons
were not enriched in the overlap: 90 % zones {100 * PT.loc['All sessions', 'P_in']:.1f} % vs {100 * PT.loc['All sessions', 'P_ref']:.1f} %
(p = {P(PT.loc['All sessions', 'p_boot'])}); 70 % zones {100 * PT7.loc['All sessions', 'P_in']:.1f} % vs
{100 * PT7.loc['All sessions', 'P_ref']:.1f} % (p = {P(PT7.loc['All sessions', 'p_boot'])}); no cohort × stage group was enriched
(Figure S1). The convergence is therefore not tied to anatomical convergence zones.

"""
    md += fig(colo / "figures_zone90" / "prelick_convergence_colocation.png", "figS1_projection_colocation.png",
              "**Figure S1. Pre-lick converging neurons and the convergence of whisker- and auditory-cortex projections (90 % zones).** "
              "a, projection zones on coronal 500-µm slabs (yellow: whisker cortex, blue: auditory cortex, purple: overlap); b, tested "
              "(grey) and converging (orange) neurons; c, converging fraction (3-D Gaussian, sigma 150 µm, normalised by tested-neuron "
              "density); d, inside the overlap vs all tested neurons; e, per cohort and stage (Holm-corrected bootstrap p); f, by "
              "projection-zone category.")
    md += "\n# Supplementary figures\n\n"
    k = 2
    for ref, pop, lab in VARIANTS[1:]:
        f = pub(ref, pop) / "Fig3_population_distance.png"
        if f.exists():
            md += fig(f, f"figS{k}_{ref}_{pop}_Fig3.png",
                      f"**Figure S{k}. Population distances, control variant: {lab}** (as Figure 3).")
            k += 1
    for ref, pop, lab in VARIANTS[1:2]:
        f = pub(ref, pop) / "Fig2_single_neurons.png"
        if f.exists():
            md += fig(f, f"figS{k}_{ref}_{pop}_Fig2.png", f"**Figure S{k}. Single neurons, control variant: {lab}** (as Figure 2).")
            k += 1
    if (FA / "recap" / "recap.png").exists():
        md += fig(FA / "recap" / "recap.png", f"figS{k}_iterations_recap.png",
                  f"**Figure S{k}. Iterations of the analysis (Table A)**: same statistics recomputed for v1-v4, FA and SL reference, "
                  "all mice and learners.")
    md += f"""
# Caveats and limitations

1. **Reaction time.** Whisker reaction times shorten to auditory-like values in R+ mice only; reaction-time matching is
   possible only with the FA reference, where it removes the cohort difference. This is the main open confound.
2. **Reference choice** changes the strength of the population interaction (SL stronger than FA); the SL timing control
   (071) did not change Δd.
3. **Few expert sessions**; interactions are tested at the mouse level.
4. **Stage** (day 0 vs later days) is confounded with time on task, recording day and probe placement.
5. **Toward, not onto.** Whisker hits move away from the reference with a component toward auditory hits; d(WH, AH) does
   not shrink.
6. **Areas are under-powered**; absence of area specificity is not evidence of uniformity.
7. **Converging neurons are mostly lick-responsive**: reward expectation or vigour vs a modality-general code are not
   separable.
8. **No correction across panels**; conclusions rest on agreement across measures and variants (Table B).

# Appendix

## Files and code

- Code: `projects/ssl-prelick-convergence/exploratory-analyses/` (051-072, `build_article.py`, `build_report_article.py`;
  ibl-ai-agent fork); within-session follow-up: Part II of ssl-prelick-convergence.
- Results: `combined_results_ks4/ssl-prelick-convergence/across_days/sl/` (SL) and `_roc_prelick/` (FA, archives, recap); to be moved to
  `combined_results_ks4/ssl-prelick-convergence/ref_sl/` and `ref_fa/`.
- Locked analysis set: `LOCKED.md` (project folder) and `_roc_prelick_sl/prelick_convergence_LOCKED.md`.
"""
    if (REP / "figures").exists():
        shutil.rmtree(REP / "figures")
    (REP / "figures").mkdir()
    for src, name in FIGS:
        if not src.exists():
            print("MISSING", src); continue
        im = Image.open(src)
        if im.width > 2000:
            im = im.resize((2000, round(im.height * 2000 / im.width)), Image.LANCZOS)
        im.convert("RGB").save(REP / "figures" / name, quality=90, optimize=True)
    (REP / "report.md").write_text(md, encoding="utf-8")
    (REP / "numbers.json").write_text(json.dumps(dict(
        variant_main=list(MAIN), variant_table=VT.to_dict(orient="records"),
        projection_colocation_90=PT.reset_index().to_dict(orient="records"),
        generated=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"), indent=1, default=float))
    shutil.copyfile(REPO / "skills" / "project-report" / "build.sh", REP / "build.sh")
    print("wrote", REP, len(FIGS), "figures")


if __name__ == "__main__":
    main()
