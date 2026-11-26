"""Preview of the passive-state pupil panel (205) before it enters the figure. Run (haas, repo root): .venv/bin/python
projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_pupil.py"""
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
OUT = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}"   # manuscript _explore (user rule)

set_nature_rc()
df = S.compute_pupil_bars()
fig = plt.figure(figsize=(75 * MM, 45 * MM))
ax = fig.add_axes([14 / 75, 8 / 45, 60 / 75, 26 / 45])
S.plot_pupil_bars(ax, df)
save_figure(fig, OUT, "pupil_pre_post_bars")
print(S.META["pupil_bars"])
