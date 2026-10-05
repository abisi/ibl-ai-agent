"""Article-style report of ssl-within-day-remapping (skills/project-report): run on haas.

Builds on ssl-prelick-convergence (same events, units and spontaneous-lick reference; imports its 051 / 057 / 061 / 062
and reads its pre-lick unit table). Methods are taken from the docstrings of the analysis scripts (001-004, 009, 010),
figure captions and results from the captions written by the figure scripts (COSYNE_captions.md, mixed-model summary),
tables from the result CSVs. Output: combined_results_ks4/ssl-within-day-remapping/report/ (report.md, numbers.json,
figures/, build.sh).
"""
import ast
import json
import pathlib
import re
import shutil

import numpy as np
import pandas as pd
from PIL import Image

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
WD = RES / "_within_day_sl"
REP = RES / "ssl-within-day-remapping" / "report"
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPS = ["R+ learning", "R+ expert", "R- learning", "R- expert"]
FIGS = []


def P(p):
    return "n/a" if p is None or not np.isfinite(p) else "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def fig(src, name, caption, width="100%"):
    name = name.replace(".png", ".jpg")
    FIGS.append((pathlib.Path(src), name))
    return f"![](figures/{name}){{width={width}}}\n\n{caption}\n"


def docstring_md(script):
    """docstring of an analysis script as Markdown: paragraphs joined, indented lines as list items"""
    d = ast.get_docstring(ast.parse((HERE / script).read_text(encoding="utf-8"))) or ""
    out = []
    for para in re.split(r"\n\s*\n", d):
        lines = para.split("\n")
        if any(l.startswith("  ") for l in lines[1:]) or lines[0].startswith("  "):
            head = [l for l in lines if not l.startswith("  ")]
            items, cur = [], None
            for l in lines:
                if l.startswith("    ") and cur is not None:
                    cur += " " + l.strip()
                elif l.startswith("  "):
                    if cur:
                        items.append(cur)
                    cur = l.strip()
                elif cur is not None:
                    cur += " " + l.strip()
            if cur:
                items.append(cur)
            if head and not lines[0].startswith("  "):
                out.append(" ".join(h.strip() for h in head[:1]))
            out.append("\n".join(f"- {i}" for i in items))
        else:
            out.append(" ".join(l.strip() for l in lines))
    return "\n\n".join(out).replace("<", "&lt;") + "\n"


def caption_sections():
    t = (WD / "cosyne" / "COSYNE_captions.md").read_text(encoding="utf-8")
    gen = re.search(r"General conventions \(all figures\):(.*?)\n", t)
    secs = {}
    for b in re.split(r"\n## ", t)[1:]:
        head, body = b.split("\n", 1)
        secs[head.split(" ")[0]] = body.strip()
    return (gen.group(1).strip() if gen else ""), secs


def halves_table(pop):
    f = WD / "halves" / pop / "within_session_tests.csv"
    if not f.exists():
        return ""
    T = pd.read_csv(f)
    T = T[T.split == "time"]
    rows = ["| Measure | " + " | ".join(f"{g}: Δ (p)" for g in GROUPS) + " |", "|" + "---|" * (len(GROUPS) + 1)]
    for r in T.itertuples(index=False):
        d = r._asdict()
        cells = []
        for g in GROUPS:
            m = {k: v for k, v in zip(T.columns, r)}
            mu, pw = m.get(f"mean Δ {g}"), m.get(f"p_wilcoxon {g}")
            cells.append("n/a" if mu is None or not np.isfinite(mu) else f"{mu:+.3f} ({P(pw)})")
        rows.append(f"| {d['measure']} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main():
    gen, S = caption_sections()
    mm = (WD / "mixed_model" / "mixed_model_summary.md").read_text(encoding="utf-8") if (WD / "mixed_model" / "mixed_model_summary.md").exists() else ""
    mm_body = "\n".join(l for l in mm.split("\n") if not l.startswith("# "))
    md = f"""---
title: "Whisker-hit activity remaps within the learning day: R+ mice converge toward auditory hits, R− mice diverge"
subtitle: "ssl-within-day-remapping -- SSL dataset (Neuropixels, Kilosort 4), report generated {pd.Timestamp.now():%Y-%m-%d %H:%M}"
author: "Axel Bisi (data); analysis with the IBL AI agent"
date: "{pd.Timestamp.now():%Y-%m-%d}"
---

# Summary

ssl-prelick-convergence showed that, across days, the pre-lick activity of whisker hits (WH) moves toward that of
auditory hits (AH) in R+ mice (rewarded whisker licks) but not in R− mice. This project asks *when* that change happens:
already within the first whisker-training session (learning day), or only between days. Each session is analysed along
its own time axis -- early vs late halves, trial-by-trial trajectories, and a single-trial mixed model of the projection
on the reward-lick coding direction (CD, spontaneous licks = 0, auditory hits = 1).

# Introduction

In the SSL task, licks after an auditory tone are always rewarded, licks after a whisker stimulus only in R+ mice. If the
reward contingency remaps whisker-hit activity onto the reward-lick representation, the remapping should start during
the learning day, as R+ mice collect whisker rewards and R− mice do not, and it should be smaller within expert sessions,
where the contingency has already been learnt. Hypotheses: on day 0, R+ whisker hits move toward auditory hits and R−
whisker hits move away from them (toward spontaneous licks); within expert sessions, both change less.

# Methods

**Dependency.** All events, trial selection, units and the spontaneous-lick reference are those of ssl-prelick-convergence
(051, locked analysis set v4 SL): {gen}

## Session halves (001)

{docstring_md('001_within_session_halves.py')}
## Trial-level trajectories on a fixed session axis (002)

{docstring_md('002_trial_slopes.py')}
## Within-day and across-day epochs on a common footing (003)

{docstring_md('003_epoch_comparison.py')}
## PSTHs per session half (004)

{docstring_md('004_halves_psth.py')}
## Single-trial mixed model (009)

{docstring_md('009_mixed_model.py')}
## Unit-sampling schemes of the decoders (010)

{docstring_md('010_decoder_schemes.py')}
# Results

## Convergence timeline

{S.get('COSYNE_convergence_timeline', '').split(chr(10) + '**Single-trial', 1)[0]}

"""
    md += fig(WD / "cosyne" / "COSYNE_convergence_timeline.png", "fig1_convergence_timeline.png",
              "**Figure 1. Convergence timeline.** " + " ".join(S.get("COSYNE_convergence_timeline", "").split("\n")))
    md += f"""
## Single-trial mixed model

{mm_body}

## Methods with examples, results and controls

"""
    md += fig(WD / "cosyne" / "COSYNE_convergence_timeline_expanded.png", "fig2_timeline_expanded.png",
              "**Figure 2. Convergence timeline, expanded: methods with real examples, results and controls.** " +
              " ".join(S.get("COSYNE_convergence_timeline_expanded", "").split("\n")))
    md += f"""
## Session halves

Change between the early and late half of each session (late − early; split at the midpoint between the two middle
auditory hits, events count-matched), all mice (Table 1; Figure 3) and learners (Figure S1). Δd = d(WH, SL) − d(WH, AH);
λ = projection of whisker hits on the SL → AH axis; decoder readouts chance-corrected by linear shift.

{halves_table('all')}
**Table 1.** Change late − early per group (mean Δ; Wilcoxon signed-rank p vs 0), time split, all mice.

"""
    md += fig(WD / "halves" / "all" / "within_session_time.png", "fig3_halves_time.png",
              "**Figure 3. Session halves, time split (all mice).** Early vs late half per session and group for the distance "
              "difference, λ and the decoder readouts; lines join the halves of a session; statistics as in Table 1.")
    md += "\n## Decoder unit-sampling schemes\n\n"
    md += fig(WD / "cosyne" / "decoder_schemes.png", "fig4_decoder_schemes.png",
              "**Figure 4. Decoder results across unit-sampling schemes.** " + " ".join(S.get("decoder_schemes", "").split("\n")))
    md += f"""
# Discussion

On the learning day, whisker-hit activity in R+ mice already moves along the reward-lick coding direction toward
auditory hits, while in R− mice it moves away, so the cohort difference seen across days in ssl-prelick-convergence
begins within the first session in which the contingency is experienced. Within expert sessions the drift is smaller and
less consistent across measures. The single-trial mixed model, which uses every lick event and separates session
baselines and time trends from the whisker-hit offset, gives the most direct estimate of the within-session drift.

# Caveats and limitations

- **Time within a session** is confounded with satiety, engagement and fatigue; the reference lick types (spontaneous
  licks, auditory hits) are measured at the same times to control for global drifts, but event-specific confounds remain.
- **Few events per half** in some sessions; count matching and the odd/even null split guard against small-sample effects.
- **Reaction times** change with learning; within-session RT changes are not modelled here.
- **Population choice.** Main figures use all mice; learners give the same direction (Figures S1-S3).
- **No correction across panels.**

# Supplementary figures

"""
    k = 1
    for f, cap in [(WD / "halves" / "learners" / "within_session_time.png", "Session halves, time split, learners."),
                   (WD / "halves" / "all" / "within_session_oddeven.png", "Session halves, odd/even null split (all mice): no temporal structure; changes should vanish."),
                   (WD / "halves" / "all" / "within_session_controls.png", "Session halves, controls (all mice): d(AH, SL), event rates, reaction times, firing rates, half durations."),
                   (WD / "slopes" / "all" / "trial_slopes.png", "Trial-level trajectories on a fixed session axis (all mice)."),
                   (WD / "slopes" / "learners" / "trial_slopes.png", "Trial-level trajectories, learners."),
                   (WD / "epochs_n4" / "all" / "epoch_comparison.png", "Within-day and across-day epochs on a common footing (4 events per class; all mice)."),
                   (WD / "epochs_n4" / "learners" / "epoch_comparison.png", "Epoch comparison, learners.")]:
        if f.exists():
            md += fig(f, f"figS{k}_{f.parent.parent.name}_{f.parent.name}_{f.stem}.png", f"**Figure S{k}.** {cap}")
            k += 1
    for v in ["u50x10", "u100x10", "u150x10", "u400x5", "uall"]:
        f = WD / f"epochs_n4_{v}" / "all" / "epoch_comparison.png"
        if f.exists():
            md += fig(f, f"figS{k}_epochs_{v}.png", f"**Figure S{k}.** Epoch comparison with unit sampling {v} (all mice; see 010).")
            k += 1
    md += """
# Appendix

## Files and code

- Code: `projects/ssl-within-day-remapping/exploratory-analyses/` (001-010, `build_report_article.py`; ibl-ai-agent fork).
  Depends on `projects/ssl-prelick-convergence/exploratory-analyses/` (051, 057, 061, 062) and its pre-lick unit table.
- Results: `combined_results_ks4/_within_day_sl/` (halves, slopes, epochs*, mixed_model, psth_halves, cosyne); to be
  moved to `combined_results_ks4/ssl-within-day-remapping/ref_sl/`.
"""
    REP.mkdir(parents=True, exist_ok=True)
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
    (REP / "numbers.json").write_text(json.dumps(dict(mixed_model=mm_body, generated=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"), indent=1))
    shutil.copyfile(REPO / "skills" / "project-report" / "build.sh", REP / "build.sh")
    print("wrote", REP, len(FIGS), "figures")


if __name__ == "__main__":
    main()
