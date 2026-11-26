"""Preview: baseline-corrected pupil time courses, passive pre | post, cohorts overlaid (205). Output in the manuscript
Figures/_explore (user rule). Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python
projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_pupil_traces.py"""
import datetime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}"
set_nature_rc()
df = S.compute_pupil_traces()
fig = plt.figure(figsize=(98 * MM, 45 * MM))
ax = fig.add_axes([14 / 98, 10 / 45, 82 / 98, 27 / 45])
S.plot_pupil_traces(ax, df)
save_figure(fig, OUT, "pupil_timecourse_bc")
print(S.META["pupil_traces"])
