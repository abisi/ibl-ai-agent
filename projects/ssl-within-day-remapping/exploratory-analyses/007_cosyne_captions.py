"""Collect the captions of every COSYNE figure of the pre-lick convergence analyses into one file.

  COSYNE_figure_v3, COSYNE_figure_v2   ssl-prelick-convergence 062 (captions_<pop>.md, per reference x population)
  COSYNE_convergence_timeline          006 (caption written by 006)
  COSYNE_within_day                    005 (superseded by 006; caption built here from COSYNE_within_day_stats.csv)
Output: combined_results_ks4/_within_day<TAG>/cosyne/COSYNE_captions.md (headline: spontaneous-lick reference, all mice;
the 062 figures of the other variants have the same structure, their captions are in each publication folder).
"""
import importlib
import pathlib
import re
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
BASE = m51.RES / f"_within_day{m51.TAG}"
OUT = BASE / "cosyne"
PUB = m51.OUTROOT / "publication"


def section(md, header):
    for b in re.split(r"\n## ", md):
        if b.startswith(header):
            return b.split("\n", 1)[1].strip()
    return "(caption not found)"


def within_day_caption():
    S = pd.read_csv(OUT / "COSYNE_within_day_stats.csv")
    e = S[S.panel == "e"].set_index(["measure", "cohort"])
    f3 = lambda v: f"{v:.2f}"
    row = lambda m, c: e.loc[(m, c)]
    P = m62.fmt_p
    return f"""**R− whisker-hit activity changes more within the learning day than within expert sessions** (superseded by
COSYNE_convergence_timeline: the effect does not survive the odd/even-split noise control; kept for reference).
**a**, First-lick-aligned PSTHs of R− sessions, day-0 and expert sessions split into early and late halves at the
midpoint between the two middle auditory hits (WH, AH, SL; rate minus baseline, mean over units per session, mean ±
s.e.m. over sessions). **b**, Whisker-hit rate across the session (5 bins of normalised time) for R− day 0, R− expert and
R+ day 0. **c**, Trial-level WH − SL score on each session's SL → AH axis across the session (as in the timeline figure,
panel b), R− day 0 and expert, R+ day 0 for reference. **d**, Population distance per half on a common footing (4 events
per class, 150 units): cross-validated distance difference d(WH, SL) − d(WH, AH) normalised by the session's d(AH, SL);
mean over mice, 95% hierarchical-bootstrap CI. **e**, Size of the within-session change |late − early| per session,
day 0 (D0) vs expert (E), normalised by the median across sessions; p: hierarchical bootstrap of the D0 − E difference.
Position: R− {f3(row('pos', 'R-').abs_change_day0)} vs {f3(row('pos', 'R-').abs_change_expert)} ({P(row('pos', 'R-').p_boot)}),
R+ {f3(row('pos', 'R+').abs_change_day0)} vs {f3(row('pos', 'R+').abs_change_expert)} ({P(row('pos', 'R+').p_boot)});
normalised distance: R− {f3(row('ddn', 'R-').abs_change_day0)} vs {f3(row('ddn', 'R-').abs_change_expert)}
({P(row('ddn', 'R-').p_boot)}), R+ {f3(row('ddn', 'R+').abs_change_day0)} vs {f3(row('ddn', 'R+').abs_change_expert)}
({P(row('ddn', 'R+').p_boot)}). R− expert sessions: {int(row('pos', 'R-').n_expert)} ({int(row('pos', 'R-').n_mice_expert)} mice)."""


def main():
    cap = (PUB / "all" / "captions_all.md").read_text(encoding="utf-8")
    general = re.search(r"\*\*General\.\*\*(.*?)\n", cap)
    parts = ["# COSYNE figure captions — pre-lick convergence (spontaneous-lick reference, all mice)\n",
             "General conventions (all figures): " + (general.group(1).strip() if general else "") + "\n",
             "## COSYNE_convergence_timeline (`_within_day_sl/cosyne/`; recommended)\n",
             (OUT / "COSYNE_convergence_timeline_caption.md").read_text(encoding="utf-8").split("\n", 2)[2].strip() + "\n",
             "## COSYNE_figure_v3 (`_roc_prelick_sl/publication/all/`)\n", section(cap, "COSYNE abstract figure (COSYNE_figure_v3)") + "\n",
             "## COSYNE_figure_v2 (`_roc_prelick_sl/publication/all/`)\n", section(cap, "COSYNE abstract figure (COSYNE_figure_v2)") + "\n",
             "## COSYNE_within_day (`_within_day_sl/cosyne/`; superseded)\n", within_day_caption() + "\n"]
    (OUT / "COSYNE_captions.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT / "COSYNE_captions.md")


if __name__ == "__main__":
    main()
