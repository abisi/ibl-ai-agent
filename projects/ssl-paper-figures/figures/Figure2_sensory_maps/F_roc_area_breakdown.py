"""Figure 2F (middle column): fraction of ROC-significant units per fine area, up-modulated above zero, down-modulated
below (user layout 2026-11-18), one panel per row: whisker, auditory, lick.

Analyses (roc_long.parquet, _roc_stage_analysis): whisker = whisker_active, auditory = auditory_active (task trials;
user 2026-11-18), lick = spontaneous_licks. Day 0, cohorts pooled. Units: stable set (skill ssl-valid-data,
137_stable_units.parquet `stable`). Fine areas = area_acronym_custom (units.parquet), kept if >= MIN_UNITS units and
>= MIN_SESS sessions; ordered by area group (get_area_group_custom_order) then by name; bars in the group colour,
down-modulated in a light shade. Fraction = units pooled over sessions (source data also has the per-session values).
Output: manuscript Figures/Figure2_sensory_maps_explore_2026-11-17/F_roc/F_<row>_area_breakdown.* + _data.csv
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python F_roc_area_breakdown.py
"""
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
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
ROC = ROOT / "_roc_stage_analysis"
STABLE = ROOT / "ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units.parquet"
OUT = ROOT.parent / "manuscripts/Bisi_Hamon/Figures/Figure2_sensory_maps_explore_2026-11-17/F_roc"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
ROWS = {"whisker": "whisker_active", "auditory": "auditory_active", "lick": "spontaneous_licks"}
TITLES = {"whisker": "Whisker trials", "auditory": "Auditory trials", "lick": "Spontaneous licks"}
MIN_UNITS, MIN_SESS = 30, 3
W, H = 90.0, 32.0  # mm; 47 fine areas need >= ~1.8 mm per bar for 5-pt labels


def load():
    u = pd.read_parquet(ROC / "units.parquet", columns=KEYS + ["day", "area_acronym_custom", "area_group"])
    u = u[u.day == 0].drop_duplicates(KEYS)
    st = pd.read_parquet(STABLE, columns=KEYS[:3] + ["nwb_cluster_id", "stable"])
    st["cluster_id"] = st.nwb_cluster_id.astype(str)  # ROC tables key units by the NWB cluster id
    u["cluster_id"] = u.cluster_id.astype(str)
    u = u.merge(st[st.stable][KEYS], on=KEYS)
    r = pd.read_parquet(ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sig", "pos", "neg"],
                        filters=[("analysis_type", "in", list(ROWS.values()))])
    r["cluster_id"] = r.cluster_id.astype(str)
    return u.merge(r.drop_duplicates(KEYS + ["analysis_type"]), on=KEYS)


def main():
    d = load()
    order = au.get_area_group_custom_order()
    cols = au.get_custom_area_groups_colors()
    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_nature_rc()
    for row, at in ROWS.items():
        x = d[(d.analysis_type == at) & d.area_group.isin(order)]
        x = x.assign(up=x.sig & x.pos, down=x.sig & x.neg)
        a = (x.groupby(["area_group", "area_acronym_custom"])
             .agg(n_units=("up", "size"), n_sessions=("session_id", "nunique"), up=("up", "mean"),
                  down=("down", "mean")).reset_index())
        a = a[(a.n_units >= MIN_UNITS) & (a.n_sessions >= MIN_SESS)]
        a["g"] = a.area_group.map({g: i for i, g in enumerate(order)})
        a = a.sort_values(["g", "area_acronym_custom"]).reset_index(drop=True)
        ses = (x.groupby(["area_acronym_custom", "session_id"])[["up", "down"]].mean().reset_index()
               .rename(columns={"up": "up_session", "down": "down_session"}))
        a.drop(columns="g").to_csv(OUT / f"F_{row}_area_breakdown_data.csv", index=False)
        ses.to_csv(OUT / f"F_{row}_area_breakdown_sessions.csv", index=False)

        fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
        ax = fig.add_axes([8 / W, 10 / H, (W - 20) / W, (H - 13) / H])
        xs = np.arange(len(a))
        c = [cols[g] for g in a.area_group]
        ax.bar(xs, 100 * a.up, width=0.8, color=c, lw=0)
        ax.bar(xs, -100 * a.down, width=0.8, color=[ps.light(k, 0.5) for k in c], lw=0)
        ax.axhline(0, color="k", lw=0.4)
        hi, lo = 100 * a.up.max() * 1.08, 100 * a.down.max() * 1.08
        lo = max(lo, 0.25 * hi)  # keep the down half readable when it is small
        ax.set_ylim(-lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.set_xlim(-0.7, len(a) - 0.3)
        ax.set_xticks(xs)
        ax.set_xticklabels(a.area_acronym_custom, rotation=90, fontsize=5)
        for t, k in zip(ax.get_xticklabels(), c):
            t.set_color(k)
        ax.tick_params(axis="x", length=0, pad=1)
        ax.tick_params(axis="y", labelsize=5, length=1.5, width=0.4)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f"{abs(v):.0f}" for v in ax.get_yticks()])
        ax.set_ylim(-lo, hi)
        ax.set_ylabel("Units (%)", fontsize=5.5, labelpad=1)
        z = lo / (hi + lo)  # zero line in axes fraction
        ax.text(1.01, z + (1 - z) / 2, "Up-\nmodulated", transform=ax.transAxes, fontsize=5, va="center")
        ax.text(1.01, z / 2, "Down-\nmodulated", transform=ax.transAxes, fontsize=5, va="center", color="0.4")
        ax.set_title(TITLES[row], fontsize=6, pad=2)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.spines["left"].set_linewidth(0.4)
        ps.save_figure(fig, OUT, f"F_{row}_area_breakdown")
        plt.close(fig)
        print(row, at, f"{len(a)} areas, {int(a.n_units.sum())} units, {x.session_id.nunique()} sessions;",
              f"up {100 * x.up.mean():.1f} %, down {100 * x.down.mean():.1f} %")


if __name__ == "__main__":
    main()
