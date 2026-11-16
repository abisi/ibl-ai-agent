"""Explore (Supplementary Figure 2 candidate): four types of unit (in)stability, one example unit each, as the last three
panels of the drift shift-test process figure (unit_spikes_analysis single_neuron_shift_test/
unit_fr_motion_shift_test_harris.py: FR and motion over time, FR against motion, |r| across the shift sweep).

Types (stable rule, ssl-valid-data: coverage >= 0.9, presence >= 0.5, drift fail = |r| > 0.5 and p_conservative < 0.01):
  stable              coverage >= 0.98, presence >= 0.95, no drift dependence (|r| < 0.1, p >= 0.5)
  stable, sparse      coverage >= 0.95, presence 0.5-0.8 (passes, low), no drift dependence, evenly sparse
  present, drifting   coverage >= 0.98, presence >= 0.95, drift fail (|r| > 0.6, p < 0.01)
  unstable            coverage < 0.6, presence < 0.45, drift fail
Candidates: unit_set_all (Bombcell good / mua) of 137_stable_units.parquet, firing rate 1-30 Hz; ranked by how clearly
they meet their type (unstable: closest to coverage 0.4, presence 0.3); the first whose DREDge motion and NWB spikes load
and whose recomputed shift test (Harris 2021, 1-s bins, N = 500, Pearson) reproduces the table (|r| within 0.02, p within
0.02; excludes the constant-motion case r = 0, p = 1) and whose probe moved >= 15 um over the session is used. 2 x 2 quadrants x 3 panels, 4 grid cells wide (138.6 mm).

Output: <manuscript>/Figures/_explore/supp2_unit_instability_types_<date>.{pdf,svg,ai,eps,png,jpg} and _units.csv
Run (haas): PYTHONPATH=~/code/unit_spikes_analysis ~/code/unit_spikes_analysis/.venv/bin/python explore_unit_instability.py
"""
import datetime
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import FIGURES_ROOT, axel_bisi_root  # noqa: E402
import paperstyle as ps  # noqa: E402
from single_neuron_shift_test import unit_fr_motion_shift_test_harris as mst  # noqa: E402

UNITS = axel_bisi_root() / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet"
NWB = axel_bisi_root() / "NWB_ks4"
DATA_ROOT = {"AB": axel_bisi_root() / "data", "MH": axel_bisi_root().parent / "Myriam_Hamon" / "data"}
BIN_S, N_SHIFT = 1.0, 500
MIN_MOTION_UM = 15.0  # motion range over the session, so every example shows real probe drift
MIN_THIRD_RATIO = {"Stable": 0.5, "Stable, sparse": 0.35, "Present, drifting": 0.25, "Unstable": 0.0}  # firing in every
# third of the session: lowest / highest third mean rate (sparse must be evenly sparse, drifting must stay present)
FR_COLOR, MOTION_COLOR = mst.FR_COLOR, mst.MOTION_COLOR
TYPES = {
    "Stable": lambda t: (t.coverage_ratio >= 0.98) & (t.presence_ratio >= 0.95) & (t.drift_abs_r < 0.1)
                        & (t.drift_shift_test_pval >= 0.5),
    "Stable, sparse": lambda t: (t.coverage_ratio >= 0.95) & t.presence_ratio.between(0.5, 0.8) & (t.drift_abs_r < 0.1)
                                & (t.drift_shift_test_pval >= 0.5),
    "Present, drifting": lambda t: (t.coverage_ratio >= 0.98) & (t.presence_ratio >= 0.95) & (t.drift_abs_r > 0.6)
                                   & (t.drift_shift_test_pval < 0.01),
    "Unstable": lambda t: (t.coverage_ratio < 0.6) & (t.presence_ratio < 0.45) & (t.drift_abs_r > 0.6)
                          & (t.drift_shift_test_pval < 0.01),
}
# Examples chosen by the user (2026-11-11: "the original 4 examples of the first iteration"): session, probe, cluster.
# The automatic search below (TYPES, RANK, checks) is kept to document how they were found.
PINNED = {"Stable": ("MH021_20250309_122552", "imec0_shank0", 609),
          "Stable, sparse": ("AB123_20240806_110231", "imec1_shank0", 1000290),
          "Present, drifting": ("AB144_20241205_111557", "imec4_shank0", 2000171),
          "Unstable": ("AB130_20240902_123634", "imec0_shank0", 530)}
RANK = {  # most typical first
    "Stable": lambda t: -(t.coverage_ratio + t.presence_ratio) + (t.drift_abs_r - 0.03).abs(),
    "Stable, sparse": lambda t: t.presence_ratio - t.coverage_ratio + (t.drift_abs_r - 0.03).abs(),
    "Present, drifting": lambda t: -t.drift_abs_r - t.presence_ratio,
    "Unstable": lambda t: (t.coverage_ratio - 0.4).abs() + (t.presence_ratio - 0.3).abs() - t.drift_abs_r,
}


def unit_series(row):
    """1-s binned firing rate and DREDge motion at the unit depth over the session (all trials), as in the test."""
    with h5py.File(NWB / f"{row.session_id}.nwb", "r") as f:
        u = f["units"]
        grp = [f[r].name.split("/")[-1] for r in u["electrode_group"][:]]
        cid = u["cluster_id"][:].astype(str)
        i = next(k for k, (g, c) in enumerate(zip(grp, cid)) if g == row.electrode_group and c == str(int(row.nwb_cluster_id)))
        idx = u["spike_times_index"][:]
        spikes = u["spike_times"][(idx[i - 1] if i else 0):idx[i]]
        depth = float(u["depth"][i].decode() if isinstance(u["depth"][i], bytes) else u["depth"][i])
        start = f["intervals/trials/start_time"][:]
    mouse = row.mouse_id
    cfg = {**mst.DEFAULT_CONFIG, "data_root": str(DATA_ROOT[mouse[:2]])}
    motion = mst.load_motion(mst.build_motion_path(cfg, mouse, row.session_id, mst._get_imec_id(row.electrode_group)))
    tx, fr = mst.compute_binned_firing_rate(spikes, start.min(), start.max(), BIN_S)
    mo = np.asarray(mst.get_motion_at_times_and_depth(motion, tx, depth), float)
    ok = np.isfinite(fr) & np.isfinite(mo)
    return tx[ok], fr[ok], mo[ok]


def pick_examples(table):
    t = table[table.unit_set_all & table.firing_rate.between(0.3, 30) & table.has_drift_test]
    chosen, used_sessions = {}, set()
    for name, sel in TYPES.items():
        cand = t[sel(t)].assign(_rank=lambda x: RANK[name](x)).sort_values("_rank")
        for row in cand.itertuples():
            if row.session_id in used_sessions:
                continue
            try:
                series = unit_series(row)
            except Exception as err:  # motion or spikes missing for this session: next candidate
                print(name, row.session_id, row.electrode_group, row.cluster_id, "skipped:", type(err).__name__, err)
                continue
            if len(series[0]) <= 2 * N_SHIFT + 10 or np.ptp(series[2]) < MIN_MOTION_UM:
                continue  # too short, or a probe that hardly moved (the example must show drift)
            thirds = np.array([x.mean() for x in np.array_split(series[1], 3)])
            if thirds.min() < MIN_THIRD_RATIO[name] * thirds.max():
                continue
            r0, p0 = mst.shift_test_exhaustive(series[1], series[2], N_SHIFT, method="pearson")[:2]
            if abs(abs(r0) - row.drift_abs_r) <= 0.02 and abs(p0 - row.drift_shift_test_pval) <= 0.02:
                chosen[name] = (row, series)
                used_sessions.add(row.session_id)
                break
            print(name, row.session_id, row.cluster_id, f"skipped: recomputed |r| {abs(r0):.3f} p {p0:.3f} vs table "
                  f"{row.drift_abs_r:.3f} {row.drift_shift_test_pval:.3f}")
        if name not in chosen:
            raise RuntimeError(f"{name}: none of {len(cand)} candidates passes the checks")
        print(name, f"{len(cand)} candidates; using", chosen[name][0].session_id, chosen[name][0].electrode_group,
              chosen[name][0].cluster_id, flush=True)
    return chosen


PANEL_MM = 12.5                       # square panels
GAPS_MM = (6.0, 14.5, 8.0, 2.0)       # left margin, after the twin-axis panel, between the last two, right margin
ROW_MM = (9.0, PANEL_MM, 7.0)         # title band, panel, x-label band


def quadrant(fig, x0_mm, y0_mm, name, row, series):
    """Three square panels for one unit; (x0_mm, y0_mm) = top-left corner of the quadrant (mm from the figure top-left)."""
    tx, fr, mo = series
    r0, p_cons, _, m, D, scores = mst.shift_test_exhaustive(fr, mo, N_SHIFT, method="pearson", return_scores=True)
    W, H = fig.get_size_inches() / ps.MM
    left, g1, g2, _ = GAPS_MM
    y = y0_mm + ROW_MM[0] + PANEL_MM
    xs = [x0_mm + left, x0_mm + left + PANEL_MM + g1, x0_mm + left + 2 * PANEL_MM + g1 + g2]
    ax_t, ax_s, ax_h = (fig.add_axes([x / W, 1 - y / H, PANEL_MM / W, PANEL_MM / H]) for x in xs)

    tmin = (tx - tx[0]) / 60
    ax_t.plot(tmin, fr, color=FR_COLOR, lw=0.3)
    ax_m = ax_t.twinx()
    ax_m.plot(tmin, mo, color=MOTION_COLOR, lw=0.6)
    ax_m.spines["right"].set_visible(True)
    ax_t.set_xlabel("Time (min)", labelpad=1)
    ax_t.set_ylabel("Firing rate (Hz)", color=FR_COLOR, labelpad=1)
    ax_m.set_ylabel("Motion (µm)", color=MOTION_COLOR, labelpad=1)
    ax_t.tick_params(axis="y", colors=FR_COLOR)
    ax_m.tick_params(axis="y", colors=MOTION_COLOR)
    ax_t.set_xlim(0, tmin[-1])

    ax_s.scatter(mo, fr, s=0.5, lw=0, color=ps.light(FR_COLOR, 0.3), rasterized=True)
    sl, ic, *_ = stats.linregress(mo, fr)
    xr = np.array([mo.min(), mo.max()])
    ax_s.plot(xr, sl * xr + ic, color="k", lw=0.6, ls="--")
    ax_s.set_xlabel("Motion (µm)", labelpad=1)
    ax_s.set_ylabel("Firing rate (Hz)", labelpad=1)

    sc = scores[np.isfinite(scores)]
    ax_h.hist(sc, bins=30, color="0.75", lw=0, edgecolor="none")
    ax_h.axvline(scores[N_SHIFT], color=FR_COLOR, lw=0.9)
    ax_h.set_xlabel("|r|", labelpad=1)
    ax_h.set_ylabel("Shifts", labelpad=1)
    for ax in (ax_t, ax_s, ax_h):
        ax.tick_params(labelsize=5, pad=1)
    ax_m.tick_params(labelsize=5, pad=1)

    p_txt = f"P = {p_cons:.2f}" if p_cons >= 0.01 else f"P = {p_cons:.3f}"
    fig.text((x0_mm + left) / W, 1 - (y0_mm + 1.0) / H, name, fontsize=6.5, fontweight="bold", ha="left", va="top")
    fig.text((x0_mm + left) / W, 1 - (y0_mm + 4.2) / H,
             f"coverage {row.coverage_ratio:.2f}  ·  presence {row.presence_ratio:.2f}  ·  |r| {abs(r0):.2f} ({p_txt})",
             fontsize=5.5, ha="left", va="top")
    return dict(type=name, mouse_id=row.mouse_id, session_id=row.session_id, electrode_group=row.electrode_group,
                cluster_id=row.cluster_id, area_group=row.area_group, firing_rate=row.firing_rate,
                coverage_ratio=row.coverage_ratio, presence_ratio=row.presence_ratio, drift_abs_r_table=row.drift_abs_r,
                drift_p_table=row.drift_shift_test_pval, stable_table=row.stable, r_recomputed=r0,
                p_conservative_recomputed=p_cons, n_bins=len(fr), D=D)


def legend(fig, y_mm):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=FR_COLOR, lw=0.8, label="Firing rate (1-s bins)"),
               Line2D([], [], color=MOTION_COLOR, lw=0.8, label="Probe motion at the unit depth (DREDge)"),
               Line2D([], [], color="k", lw=0.6, ls="--", label="Linear fit"),
               Line2D([], [], color=FR_COLOR, lw=0.9, label="|r| at zero shift"),
               Patch(facecolor="0.75", edgecolor="none", label=f"|r| at shifts of -{N_SHIFT} to {N_SHIFT} s")]
    H = fig.get_size_inches()[1] / ps.MM
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - y_mm / H), ncol=5, fontsize=5.5,
               frameon=False, handlelength=1.6, columnspacing=1.4, borderaxespad=0)


def pinned_examples(table):
    chosen = {}
    for name, (sid, grp, cid) in PINNED.items():
        row = next(table[(table.session_id == sid) & (table.electrode_group == grp) & (table.cluster_id == cid)].itertuples())
        chosen[name] = (row, unit_series(row))
    return chosen


def main():
    ps.set_nature_rc()
    table = pd.read_parquet(UNITS)
    chosen = pinned_examples(table) if "--search" not in sys.argv else pick_examples(table)
    w_mm = 4 * 29.4 + 3 * 7  # four Supp-figure grid cells (183-mm grid, 5 columns, 7-mm gaps)
    q_w, q_h, legend_mm = (w_mm - 2) / 2, sum(ROW_MM), 5.0
    fig = plt.figure(figsize=(w_mm * ps.MM, (legend_mm + 2 * q_h + 1) * ps.MM))
    legend(fig, 0.5)
    rows = [quadrant(fig, (k % 2) * (q_w + 2), legend_mm + (k // 2) * q_h, name, *chosen[name])
            for k, name in enumerate(TYPES)]
    out = FIGURES_ROOT / "_explore"
    name = f"supp2_unit_instability_types_{datetime.date.today().isoformat()}"
    ps.save_figure(fig, out, name)
    pd.DataFrame(rows).to_csv(out / f"{name}_units.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
