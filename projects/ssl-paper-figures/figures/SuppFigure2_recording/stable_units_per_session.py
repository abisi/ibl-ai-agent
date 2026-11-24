"""Supp Fig 2: histogram of the number of stable units per day-0 session (user request 2026-11-18).

Stable units = skill ssl-valid-data 'Unit sets': good + mua units with coverage >= 0.9, presence >= 0.5 and a passed
drift joint test (no drift result = not stable); table 137_stable_units.parquet (column `stable`). Day per session from
the v2 unit table (unit_labels_all_days_v2.parquet). One count per session, all probes pooled.
Output (manuscript Figures/SuppFigure2_recording_<date>/): SuppFigure2_stable_units_per_session.{pdf,svg,eps,ai,png,jpg}
+ _counts.csv (source data).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python stable_units_per_session.py
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import paperstyle as ps  # noqa: E402

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
STABLE = ROOT / "ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units.parquet"
DAYS = ROOT / "ssl-ssp-un-review/unit_labels_all_days_v2.parquet"
OUT = ROOT.parent / "manuscripts/Bisi_Hamon/Figures" / f"SuppFigure2_recording_{datetime.date.today():%Y-%m-%d}"
NAME = "SuppFigure2_stable_units_per_session"
BIN = 250


def main():
    t = pd.read_parquet(STABLE, columns=["mouse_id", "session_id", "electrode_group", "unit_set_all", "stable"])
    day = pd.read_parquet(DAYS, columns=["session_id", "day"]).drop_duplicates("session_id")
    t = t.merge(day, on="session_id", how="inner")
    d0 = t[t.day == 0]
    c = (d0.groupby(["mouse_id", "session_id"])
         .agg(n_stable=("stable", "sum"), n_good_mua=("unit_set_all", "sum"),
              n_probes=("electrode_group", lambda x: x.str.split("_").str[0].nunique()))
         .reset_index().sort_values("session_id"))
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT / f"{NAME}_counts.csv", index=False)
    n, med = len(c), float(np.median(c.n_stable))
    print(f"{n} day-0 sessions, {c.mouse_id.nunique()} mice; stable units median {med:.0f}, "
          f"range {c.n_stable.min()}-{c.n_stable.max()}, total {c.n_stable.sum()}")

    ps.set_nature_rc()
    W, H = 40.0, 40.0
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
    ax = fig.add_axes([9 / W, 8 / H, 28 / W, 28 / H])  # square axes
    edges = np.arange(0, np.ceil(c.n_stable.max() / BIN) * BIN + BIN, BIN)
    ax.hist(c.n_stable, bins=edges, color="0.55", edgecolor="white", linewidth=0.2)
    top = int(np.ceil(np.histogram(c.n_stable, edges)[0].max() * 1.3))
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.axvline(med, ymax=0.86, color="k", lw=0.6, ls=(0, (2, 1.5)))
    ax.text(med, top * 0.9, f"median {med:,.0f}", fontsize=5, ha="center", va="bottom")
    one = c.mouse_id.nunique() == n
    ax.text(0.0, 1.0, f"n = {n} sessions" + (" (1 per mouse)" if one else f", {c.mouse_id.nunique()} mice"),
            transform=ax.transAxes, fontsize=5, ha="left", va="bottom")
    ax.set_xlim(-0.02 * edges[-1], edges[-1] * 1.02)
    ax.set_xlabel("Stable units per session (day 0)", fontsize=6)
    ax.set_ylabel("Sessions", fontsize=6)
    ax.tick_params(labelsize=5, length=2, width=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ps.save_figure(fig, OUT, NAME)
    plt.close(fig)
    print("saved", OUT / NAME)


if __name__ == "__main__":
    main()
