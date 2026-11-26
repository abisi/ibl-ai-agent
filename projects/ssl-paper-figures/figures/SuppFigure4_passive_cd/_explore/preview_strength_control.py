"""Preview of the supp panel 'strength_control' (pre-task whisker response strength vs the passive projection / angle change)
before it enters the supp spec. Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python
projects/ssl-paper-figures/figures/SuppFigure4_passive_cd/_explore/preview_strength_control.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shared_passive_cd as S  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402
import datetime  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
OUT = FIGURES_ROOT / "_explore" / f"SuppFigure4_passive_cd_{datetime.date.today().isoformat()}"   # manuscript _explore (user rule)

set_nature_rc()
df = S.compute_strength_control()
fig = plt.figure(figsize=(120 * MM, 52 * MM))
ax = fig.add_axes([0.1, 0.24, 0.86, 0.7])
S.plot_strength_control(ax, df)
save_figure(fig, OUT, "strength_control")
print(S.META["strength_control"])
