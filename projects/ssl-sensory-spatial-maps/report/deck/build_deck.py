"""Sensory maps + stimulus-arrival slide deck (16:9 PowerPoint) from the panels in img/ (made on haas by
../build_deck_assets.py). Each slide: claim title, figure(s) fitted to the free area, optional bullets, footnote with
n / test / effect size, speaker notes. Run: python build_deck.py (needs python-pptx and Pillow)."""
import pathlib

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = pathlib.Path(__file__).parent
IMG = HERE / "img"
W, H = 13.333, 7.5
M = 0.4                                   # side margin (in)
TOP, FOOT = 1.15, 0.75                    # title band, footnote band
GREY = RGBColor(0x55, 0x55, 0x55)

S = [  # (title, images, bullets, footnote, notes); images = list of (file, width fraction of the figure area)
    ("Conclusions", [], [
        "About 15 % of neurons respond to each stimulus; modality preference is organised across and within areas",
        "Each stimulus reaches its own sensory system first: whisker → barrel cortex (~12 ms), sound → midbrain (~9 ms)",
        "Population decoding detects stimulus modality within 8–18 ms: first midbrain, whisker cortex and thalamus; last motor-frontal cortex",
        "Faster areas also carry more early information (ρ ≈ −0.9)",
        "Whisker and auditory cortex project together to caudal striatum, posterior parietal cortex and SC; bimodal neurons are enriched there (+6 points)"],
     "Decoding: 6 area groups and 8 fine areas so far; full sweep (18 groups, 40 areas) and passive trials are running.",
     "Same order as the report's key conclusions."),
    ("179,064 neurons from 122 sessions, all registered to the Allen CCF", [("flat_recorded_zones.png", 0.38)], [
        "92 mice, R+ and R− cohorts, learning day and expert days (pooled)",
        "Good + multi-unit clusters (Kilosort 4)",
        "Probe-track reconstruction, folded onto the left hemisphere",
        "111 recorded structures (≥ 10 neurons)",
        "Whisker stimulus artefact (−10 to +5 ms) corrected on all spike trains"],
     "Top: recorded isocortex neurons on the Allen butterfly flatmap (anterior up). Bottom: 70 % projection zones of whisker (yellow) and auditory (blue) cortex.",
     "Active trials: perf ≠ 6, disengaged tail trimmed. Passive: fixed ~3 s ITI blocks."),
    ("Whisker and auditory cortex project to partly overlapping targets", [("projection_zones_coronal.png", 1)], [],
     "Allen anterograde tracing (SSp-bfd 8, SSs 3, AUDp 6, AUDd/v 3 injections), 70 % projection zones. "
     "Overlap 3.9 mm³: caudal / tail striatum, VISa / VISrl / VISal, SCm, MRN, TEa.", ""),
    ("About 15 % of neurons respond to each stimulus, in different places", [("flat_whisker_auditory.png", 1)], [],
     "Whisker: 15.3 % significant (12.1 % excited); auditory: 15.2 % (13.6 % excited). ROC 5–35 ms vs pre-trial baseline, 1000 permutations, p < 0.05. "
     "Bottom: selectivity density normalised by recorded-neuron density (3D Gaussian, σ = 150 µm).",
     "Selectivity = 2·AUC − 1. Flatmap: left hemisphere, anterior up."),
    ("Modality preference is spatially organised", [("flat_modality.png", 0.27), ("modality_targets.png", 0.73)], [],
     "27.8 % of neurons prefer one modality (17.6 % auditory, 10.2 % whisker). Yellow: whisker-preferring; blue: auditory-preferring. "
     "Right: 500 µm coronal slabs centred on the projection zones.", ""),
    ("Within areas, whisker- and auditory-preferring neurons are spatially offset", [("modality_contours.png", 1)], [],
     "9 of 13 target slabs: centroids 70–350 µm apart (within-session label permutation, Holm p < 0.05). Auditory-preferring neurons "
     "deeper in SSp-bfd, SCm and striatum tail; more superficial in auditory cortex and anterior DMS.",
     "80 % highest-density contours (Gaussian KDE). Permuting within sessions removes offsets caused by which sessions recorded which preference."),
    ("Bimodal neurons are enriched where the projections converge", [("colocation_figure.png", 0.58)], [
        "Bimodal = responsive to both stimuli (active or passive tests, Bonferroni per session)",
        "24 % of responsive neurons are bimodal",
        "Inside the overlap: 29.5 % vs 23.8 % overall",
        "+5.7 points (95 % CI +1.1 to +10.3), p = 0.008",
        "90 % zones: +6.3 points, p < 0.001"],
     "Hierarchical bootstrap (sessions, then neurons) of P(bimodal | inside overlap) − P(bimodal | all responsive). Co-location, not causation.", ""),
    ("Each modality reaches its own sensory system first", [("flat_latency.png", 0.55)], "TABLE",
     "Median half-time to peak of responsive neurons (light = fast). Whisker latency searched after +5 ms (artefact window).",
     "Latency = last upward crossing of half the peak before the peak; 1 ms PSTH, σ = 2 ms."),
    ("Stimulus modality is decodable within 8–18 ms, first in the midbrain", [("arr_groups_heatmap_ranking.png", 1)], [],
     "Pseudo-populations of 200 neurons (20 sessions × 10), whisker vs auditory task trials, L2 logistic regression per 20 ms bin. "
     "Corrected accuracy = real − label-shuffled. Grey: not above chance. Onset: ≥ 80 % of the next 25 ms above chance. Preliminary: 6 area groups.",
     "Onsets: midbrain 8, SS-whisker 10, thalamus 10, auditory 12, striatum 12, motor 16 ms. 100 iterations × 10 shuffles (pilot)."),
    ("Fine areas: SCm first, then whisker cortex, striatum and motor-frontal cortex", [("arr_areas_heatmap_ranking.png", 1)], [],
     "SCm 8 ms; SSp-bfd and SSs 10 ms; DMS and DLS 12 ms; wM1 14, wM2 16, ALM 18 ms. Preliminary: 8 areas (the 40 best-sampled are running).", ""),
    ("Single-neuron latency and population onset agree", [("link_latency_onset.png", 1)], [],
     "Population decoding onset (N = 200) vs median single-neuron latency per area group; OLS line with 95 % CI. "
     "Preliminary: n = 6 groups, faster modality ρ = 0.77, p = 0.08.", ""),
    ("Faster areas carry more early information", [("arr_groups_controls.png", 1)], [],
     "Early accuracy (5–50 ms) grows with neuron count; onset vs early accuracy: Spearman ρ = −0.9. "
     "Matching whisker cortex at 100 neurons needs ~40 midbrain, ~160 striatum, > 1,000 motor neurons.",
     "Matched-accuracy control: neuron count at which each group reaches the early accuracy of SS-whisker at 100 neurons."),
    ("Passive trials (no task, no licks): pending", [("pas_groups_heatmap_ranking.png", 1)], [
        "Passive whisker vs auditory decoding runs overnight (110 sessions with ≥ 3 passive trials per stimulus)",
        "Tests whether the onset order is sensory rather than task- or lick-related"],
     "Shown here once the passive sweep finishes.", ""),
    ("Summary: separate entry routes, shared targets", [("summary_schematic.png", 1)], [],
     "Draft. a) Whisker route via thalamus to barrel cortex, sound via midbrain and thalamus to auditory cortex; both cortices project to tail striatum, "
     "posterior parietal cortex, SCm / MRN and TEa, where bimodal neurons are enriched. Times: population decoding onset (N = 200). b) Onsets with 95 % bootstrap CI.",
     "Routes are textbook anatomy plus the Allen projection zones; times update automatically from onset_bootstrap_N200.csv."),
    ("Caveats and next steps", [], [
        "Co-location ≠ causation: tracing from other mice, axons of passage, ~100–200 µm CCF uncertainty",
        "Cohorts and stages pooled → split R+ / R− and learning / expert",
        "Task-trial decoding after ~100 ms can use licks → passive trials (tonight), licked-only control",
        "Whisker information cannot appear before +5 ms (artefact window)",
        "Pilot sampling (100 iterations × 10 shuffles) → increase for final figures"], "", ""),
    ("Backup: decoding method", [("arr_groups_method.png", 0.4)], [
        "20 sessions with replacement, N/20 neurons per session",
        "Pseudo-trials per class by balanced reuse of each session's trials",
        "L2 logistic regression per bin, 3-fold CV, inner 2-fold for C",
        "d(t) = BA_real − mean of 10 within-session label shuffles",
        "Above chance: 5th percentile of d over 100 iterations > 0",
        "Onset: first bin above chance with ≥ 80 % of the next 25 ms above chance"], "", ""),
    ("Backup: trial-shuffle control", [("arr_groups_shuffle.png", 0.5), ("arr_groups_timecourse.png", 0.5)], [], "", ""),
    ("Backup: bimodal neurons on the flatmap", [("flat_bimodal.png", 1)], [], "", ""),
    ("Backup: all flatmaps", [("flat_all.png", 1)], [], "", ""),
    ("Backup: whisker responsiveness in target slabs", [("whisker_targets.png", 1)], [], "", ""),
]

SECTION = [("179,064", "I · Anatomy"), ("Whisker and auditory cortex project", "I · Anatomy"),
           ("About 15 %", "II · Where neurons respond"), ("Modality preference", "II · Where neurons respond"),
           ("Within areas", "II · Where neurons respond"), ("Bimodal", "II · Where neurons respond"),
           ("Each modality", "III · When neurons respond"), ("Stimulus modality", "IV · Population coding"),
           ("Fine areas", "IV · Population coding"), ("Single-neuron", "IV · Population coding"),
           ("Faster areas", "IV · Population coding"), ("Passive", "IV · Population coding"), ("Summary", "Summary")]


TABLE = [("Area group", "Whisker (ms)", "Auditory (ms)"), ("SS-whisker", "11.6", "18.0"), ("Posterior parietal", "12.6", "16.7"),
         ("Auditory", "15.1", "15.4"), ("Midbrain", "16.9", "8.5"), ("Thalamus", "17.8", "15.8"), ("Motor", "20.8", "15.8"),
         ("Striatum", "21.1", "16.5")]


def text(slide, x, y, w, h, s, size, bold=False, color=None, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = s
    p.font.size, p.font.bold, p.font.italic = Pt(size), bold, italic
    if color is not None:
        p.font.color.rgb = color
    return tf


def bullets(slide, x, y, w, h, items, size):
    tf = text(slide, x, y, w, h, "• " + items[0], size)
    for it in items[1:]:
        p = tf.add_paragraph()
        p.text, p.font.size = "• " + it, Pt(size)
        p.space_before = Pt(8)


def picture(slide, f, x, y, w, h):
    """fit the image in the box, keep aspect, centre"""
    iw, ih = Image.open(f).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(f), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, x, y, w):
    t = slide.shapes.add_table(len(TABLE), 3, Inches(x), Inches(y), Inches(w), Inches(0.38 * len(TABLE))).table
    for i, row in enumerate(TABLE):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = v
            c.text_frame.paragraphs[0].font.size = Pt(14)
            c.text_frame.paragraphs[0].font.bold = (i == 0) or (i, j) in {(1, 1), (2, 1), (4, 2)}


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    text(s, 1, 1.9, W - 2, 1.6, "Whisker and auditory signals across the mouse brain", 40, bold=True)
    text(s, 1, 3.9, W - 2, 1, "Where, when and through which pathways: sensory maps and stimulus-arrival decoding", 22, color=GREY)
    text(s, 1, 4.9, W - 2, 0.6, "SSL Neuropixels dataset, working version 2026-10-04 (unpublished)", 16, color=GREY)

    missing = []
    for title, imgs, bl, foot, notes in S:
        s = prs.slides.add_slide(blank)
        text(s, M, 0.3, W - 2 * M, 0.8, title, 28, bold=True)
        kick = next((k for t, k in SECTION if title.startswith(t)), "")
        if kick:
            text(s, M, 0.02, 4, 0.3, kick.upper(), 10, bold=True, color=GREY)
        y0, y1 = TOP, H - (FOOT if foot else 0.3)
        imgs = [(f, fr) for f, fr in imgs if (IMG / f).exists() or missing.append(f)]
        fig_frac = sum(fr for _, fr in imgs)
        x = M
        for f, fr in imgs:
            w = (W - 2 * M) * fr
            picture(s, IMG / f, x, y0, w, y1 - y0)
            x += w
        rest = W - M - x
        if bl == "TABLE":
            table(s, x + 0.3, y0 + 0.6, rest - 0.6)
        elif bl:
            if fig_frac == 0:
                bullets(s, 1.0, y0 + 0.2, W - 2.0, y1 - y0, bl, 22)
            else:
                bullets(s, x + 0.2, y0 + 0.4, max(rest - 0.4, 3), y1 - y0, bl, 16 if len(bl) > 4 else 18)
        if foot:
            text(s, M, H - FOOT, W - 2 * M, FOOT - 0.05, foot, 11, italic=True, color=GREY)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
    out = HERE / "sensory_deck.pptx"
    prs.save(out)
    print(out, len(prs.slides), "slides; missing:", missing)


if __name__ == "__main__":
    main()
