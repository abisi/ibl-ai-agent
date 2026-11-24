"""Figure 2F as one grid (layout of Figure2_test.pdf, user 2026-11-19): rows whisker / auditory / sensory / lick
(bimodal row = units responsive to both whisker and auditory trials, either sign, stacked: no preference,
whisker-preferring, auditory-preferring by wh_vs_aud_active); columns
ROC schematic (aligned event + analysis windows) | fraction of units per fine area | PSTHs of an example area group.

ROC (roc_utils_new, rate-based since 2026-09-28; _roc_stage_analysis/roc_long.parquet):
  whisker_active / auditory_active: per-trial rate in pre (-50..-15 ms) vs post (+5..+35 ms) of the stimulus, active
    trials (hit + miss), each minus the trial's baseline rate (-1 s..-15 ms); permutation test (1000), alpha 0.05.
  spontaneous_licks: pre (-400..-200 ms) vs post (0..+200 ms) of piezo licks > 1 s from any stimulus.
  wh_vs_aud_active: post window of whisker vs auditory trials (positive = auditory > whisker).
Classes (partition, sums to 100 % per area; checked: sig never NaN, sig implies pos xor neg): up = sig & sel > 0,
  down = sig & sel < 0, non = not sig.
Bimodal row: responsive to both = sig in whisker_active AND auditory_active (either sign); split by wh_vs_aud_active:
  whisker-preferring (sig, whisker > auditory), auditory-preferring (sig, auditory > whisker), non-preferring (not sig).
Units, areas: as F_roc_area_breakdown (day 0, stable set, firing rate >= 2 Hz (MIN_FR, user 2026-11-19), fine areas with >= 30 units and >= 3 sessions); Pons and
  medulla and Amygdala and hypothalamus groups removed (user 2026-11-19).
PSTHs: F_roc_example_psths (cache F_example_psths.parquet); bimodal row: whisker and auditory PSTHs of all bimodal (bar-plot)
  bimodal units of EX_BIMODAL (cache F_bimodal_psths.parquet, same method).
Output: manuscript Figures/Figure2_sensory_maps_explore_2026-11-17/F_roc/Figure2F_grid.* + Figure2F_bars_data.csv
Run (haas): PYTHONPATH=<NAS ephys_utilities>:~/code/NWB_reader ~/code/unit_spikes_analysis/.venv/bin/python F_roc_grid.py
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import paperstyle as ps  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402
import F_roc_area_breakdown as fb  # noqa: E402
import F_roc_example_psths as fp  # noqa: E402

KEYS = fb.KEYS
HIDE = ["Pons and medulla", "Amygdala and hypothalamus"]
ATYPES = ["whisker_active", "auditory_active", "spontaneous_licks", "wh_vs_aud_active"]
C_W, C_A = ps.COLORS["whisker"], ps.COLORS["auditory"]
EX_BIMODAL = "Midbrain"
BIM_CACHE = fp.CACHE.with_name("F_bimodal_all_psths.parquet")  # all bimodal units of EX_BIMODAL (user 2026-11-19)
LAST_ROW = "bimodal"  # area labels under the bottom row
MIN_FR = 2.0       # Hz, session mean firing rate (units.parquet firing_rate); all F panels
W = 130.0                        # F sits beside panel E (user 2026-11-19)
ROW_H, TOP, BOT = 18.0, 3.0, 10.0
COL_X = (0.0, 20.0, 103.0)       # left edges (mm): schematic, bars, PSTH
COL_W = (19.0, 82.0, 27.0)


def load():
    u = pd.read_parquet(fb.ROC / "units.parquet",
                        columns=KEYS + ["day", "area_acronym_custom", "area_group", "firing_rate", "target_region"])
    u = u[u.day == 0].drop_duplicates(KEYS)
    # allen_utils cc65eac (pushed): MOs/MOp units of OFC-target probes -> MO-FC. The ROC tables predate it; applied here
    # until the next table rebuild (same rule as create_areas_subdivisions' target_region_map)
    u.loc[u.area_acronym_custom.isin(["MOs", "MOp"]) & (u.target_region.astype(str) == "OFC"),
          "area_acronym_custom"] = "MO-FC"
    u = u[u.firing_rate >= MIN_FR]  # user 2026-11-19: decreases are undetectable in the 30-ms window below ~2 Hz
    st = pd.read_parquet(fb.STABLE, columns=KEYS[:3] + ["nwb_cluster_id", "stable"])
    st["cluster_id"] = st.nwb_cluster_id.astype(str)
    u["cluster_id"] = u.cluster_id.astype(str)
    u = u.merge(st[st.stable][KEYS], on=KEYS)
    u = u[u.area_group.isin(au.get_area_group_custom_order()) & ~u.area_group.isin(HIDE)]
    r = pd.read_parquet(fb.ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sel", "sig"],
                        filters=[("analysis_type", "in", ATYPES)])
    r["cluster_id"] = r.cluster_id.astype(str)
    r = r.drop_duplicates(KEYS + ["analysis_type"])
    for at in ATYPES:
        x = r[r.analysis_type == at][KEYS + ["sel", "sig"]].rename(columns={"sel": f"sel_{at}", "sig": f"sig_{at}"})
        u = u.merge(x, on=KEYS, how="inner")
        u[f"sig_{at}"] = u[f"sig_{at}"] == 1
    u["bimodal"] = u.sig_whisker_active & u.sig_auditory_active
    pref = u.sig_wh_vs_aud_active
    u["bim_w"] = u.bimodal & pref & (u.sel_wh_vs_aud_active < 0)
    u["bim_a"] = u.bimodal & pref & (u.sel_wh_vs_aud_active > 0)
    u["bim_n"] = u.bimodal & ~pref
    u["bim_p"] = u.bim_w | u.bim_a
    u["w_only"] = u.sig_whisker_active & ~u.sig_auditory_active
    u["a_only"] = u.sig_auditory_active & ~u.sig_whisker_active
    u["sensory"] = u.sig_whisker_active | u.sig_auditory_active
    return u


def area_table(u):
    order = au.get_area_group_custom_order()
    a = u.groupby(["area_group", "area_acronym_custom"]).agg(n_units=("session_id", "size"),
                                                             n_sessions=("session_id", "nunique")).reset_index()
    for row, at in fb.ROWS.items():
        s, v = u[f"sig_{at}"], u[f"sel_{at}"]
        u[f"{row}_up"], u[f"{row}_down"] = s & (v > 0), s & (v < 0)
        u[f"{row}_non"] = ~s
    cols = [f"{r}_{c}" for r in fb.ROWS for c in ("up", "down", "non")] + ["bimodal", "bim_w", "bim_a", "bim_n", "bim_p",
                                                                             "w_only", "a_only", "sensory"]
    m = u.groupby(["area_group", "area_acronym_custom"])[cols].mean().reset_index()
    a = a.merge(m, on=["area_group", "area_acronym_custom"])
    a = a[(a.n_units >= fb.MIN_UNITS) & (a.n_sessions >= fb.MIN_SESS)]
    a["g"] = a.area_group.map({g: i for i, g in enumerate(order)})
    fine = {x: i for i, x in enumerate(au.get_area_acronym_custom_order())}       # allen_utils fine-area order
    a["f"] = a.area_acronym_custom.map(fine).fillna(len(fine))
    a = a.sort_values(["g", "f", "area_acronym_custom"]).drop(columns=["g", "f"]).reset_index(drop=True)
    for r in fb.ROWS:   # partition check
        assert np.allclose(a[[f"{r}_up", f"{r}_down", f"{r}_non"]].sum(1), 1.0)
    return a


def bimodal_psths(u, n_proc=40):
    if BIM_CACHE.exists():
        return pd.read_parquet(BIM_CACHE)
    b = u[u.bimodal & (u.area_group == EX_BIMODAL)]
    U = pd.concat([b.assign(row=r, cls="nonpref") for r in ("whisker", "auditory")])[KEYS + ["row", "cls"]]
    with mp.get_context("fork").Pool(n_proc) as pool:
        P = pd.concat(pool.map(fp.run_session, list(U.groupby("session_id"))), ignore_index=True)
    P.to_parquet(BIM_CACHE)
    return P


# ---------------------------------------------------------------- drawing
def mm_ax(fig, H, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def schematic(ax, row):
    ax.set_axis_off()
    if row == "lick":
        t0, t1, pre, post, unit, ticks = -1.05, 0.3, (-0.4, -0.2), (0.0, 0.2), "s", [-1, -0.4, 0, 0.2]
        ev, col = "Spont. lick", "0.2"
    else:
        t0, t1, pre, post, unit, ticks = -0.065, 0.045, (-0.05, -0.015), (0.005, 0.035), "ms", [-50, -15, 5, 35]
        ev, col = ("Whisker stim." if row == "whisker" else "Tone") if row != "bimodal" else "Whisker or tone", None
        col = {"whisker": C_W, "auditory": C_A, "bimodal": "0.2"}[row]
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.45, 1.75)
    ax.plot([t0, t1], [-0.6, -0.6], color="k", lw=0.4)
    for t in ticks:
        tt = t / 1000 if unit == "ms" else t
        if t0 <= tt <= t1:
            ax.plot([tt, tt], [-0.6, -0.7], color="k", lw=0.4)
            ax.text(tt, -0.75, f"{t:g}", ha="center", va="top", fontsize=5)
    ax.text(t1, -1.45, unit, ha="right", va="bottom", fontsize=5)
    for (a, b), lab, fc in ((pre, "Pre", "0.8"), (post, "Post", ps.light(col, 0.45) if col != "0.2" else "0.55")):
        ax.add_patch(Rectangle((a, -0.5), b - a, 0.8, fc=fc, ec="none"))
        ax.text((a + b) / 2, -0.1, lab, ha="center", va="center", fontsize=5)
    # baseline (-1 s .. -15 ms): to scale for licks, arrow off the left edge for stimuli
    bl0 = max(-1.0, t0)
    ax.annotate("", xy=(bl0, 0.55), xytext=(-0.015, 0.55),
                arrowprops=dict(arrowstyle="-|>" if row != "lick" else "-", lw=0.4, color="0.4",
                                shrinkA=0, shrinkB=0, mutation_scale=4))
    ax.text(-0.015, 0.62, "Baseline", ha="right", va="bottom", fontsize=5, color="0.4")
    ax.plot([0, 0], [-0.6, 1.25], color=col, lw=0.6)
    ax.set_clip_on(False)
    ax.text(0, 1.3, ev, ha="center" if row != "lick" else "right", va="bottom", fontsize=5, color=col)


def bars(ax, a, row, cols):
    xs = np.arange(len(a))
    c = [cols[g] for g in a.area_group]
    if row == "bimodal":
        b0 = np.zeros(len(a))
        # bimodal = responsive to both whisker and auditory trials (user 2026-11-19), split by wh_vs_aud_active
        for k, fc, lab in (("bim_n", "0.45", "No preference"), ("bim_w", C_W, "Whisker-preferring"),
                           ("bim_a", C_A, "Auditory-preferring")):
            ax.bar(xs, 100 * a[k], bottom=b0, width=0.8, color=fc, lw=0, label=lab)
            b0 += 100 * a[k].to_numpy()
        ax.set_ylim(0, b0.max() * 1.1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 1.05),
                  handlelength=0.8, handletextpad=0.4, borderaxespad=0, ncol=2, columnspacing=0.8)
        ax.set_ylabel("Both (%)", fontsize=5, labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        up, dn = 100 * a[f"{row}_up"], 100 * a[f"{row}_down"]
        ax.bar(xs, up, width=0.8, color=c, lw=0)
        ax.bar(xs, -dn, width=0.8, color=[ps.light(k, 0.5) for k in c], lw=0)
        ax.axhline(0, color="k", lw=0.4)
        hi, lo = up.max() * 1.08, dn.max() * 1.08
        lo = max(lo, 0.25 * hi)
        ax.set_ylim(-lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f"{abs(v):.0f}" for v in ax.get_yticks()])
        ax.set_ylim(-lo, hi)
        ax.set_ylabel("Units (%)", fontsize=5, labelpad=1)
        ax.text(1.0, 1.0, "Up", transform=ax.transAxes, fontsize=5, ha="right", va="top")
        ax.text(1.0, 0.0, "Down", transform=ax.transAxes, fontsize=5, ha="right", va="bottom", color="0.4")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_xlim(-0.7, len(a) - 0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(a.area_acronym_custom if row == LAST_ROW else [], rotation=90, fontsize=5)
    for t, k in zip(ax.get_xticklabels(), c):
        t.set_color(k)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.tick_params(axis="y", labelsize=5, length=1.5, width=0.4, pad=1)
    ax.spines["left"].set_linewidth(0.4)


def psth(ax, P, row, cols, out):
    if row == "bimodal":
        x, g = P, EX_BIMODAL
        series = [("whisker", "nonpref", C_W, "Whisker"), ("auditory", "nonpref", C_A, "Tone")]
    else:
        x, g = P[P.row == row], fp.EXAMPLE[row]
        k = cols[g]
        series = [(row, "up", k, "Up"), (row, "down", ps.light(k, 0.5), "Down"), (row, "non", "0.6", "Non")]
    t0 = fp.WIN["lick" if row == "lick" else "whisker"][0]
    for i, (r, cl, colr, lab) in enumerate(series):
        y = x[(x.row == r) & (x.cls == cl)]
        if not len(y):
            continue
        M = np.stack(y.groupby("session_id").psth.apply(lambda s: np.mean(np.stack(s.to_numpy()), 0)).to_numpy())
        tt = (t0 + 5e-4 + 1e-3 * np.arange(M.shape[1])) * (1 if row == "lick" else 1000)
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        ax.fill_between(tt, mu - se, mu + se, color=ps.light(colr, 0.6), lw=0)
        ax.plot(tt, mu, color=colr, lw=0.6)
        ax.text(0.03, 1.0 - 0.17 * i, lab, transform=ax.transAxes, fontsize=5, ha="left", va="top", color=colr)
        out.append(pd.DataFrame(dict(row=row, area_group=g, series=f"{r}_{cl}", t=tt, mean=mu, sem=se,
                                     n_sessions=len(M), n_units=len(y))))
    ax.axvline(0, color="k", lw=0.4, ls=(0, (2, 1.5)))
    ax.axhline(0, color="0.3", lw=0.3)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_xlabel("Time (s)" if row == "lick" else "Time (ms)", fontsize=5, labelpad=0.5)
    ax.set_ylabel("Δ Hz", fontsize=5, labelpad=0.5)
    lbl = {"Somatosensory-whisker": "Ss-whisker", "Auditory areas": "Auditory", "Motor areas": "Motor areas",
           "Midbrain": "Midbrain, bimodal"}[g]
    ax.set_title(lbl, fontsize=5, pad=1, color=cols[g] if row != "bimodal" else cols[g])
    ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    u = load()
    a = area_table(u)
    a.to_csv(fb.OUT / "Figure2F_bars_data.csv", index=False)
    print(f"{len(a)} areas, {int(a.n_units.sum())} units; bimodal {100 * u.bimodal.mean():.1f} % "
          f"(w-pref {100 * u.bim_w.mean():.1f}, a-pref {100 * u.bim_a.mean():.1f}, non-pref {100 * u.bim_n.mean():.1f})")
    print(u.groupby("area_group")[["bimodal", "bim_n"]].mean().mul(100).round(1).sort_values("bim_n").tail(6))
    keep = u[KEYS].astype(str)
    P = pd.read_parquet(fp.CACHE).astype({"cluster_id": str}).merge(keep, on=KEYS)       # PSTH caches: >= MIN_FR units
    PB = bimodal_psths(u).astype({"cluster_id": str}).merge(keep, on=KEYS)
    cols = au.get_custom_area_groups_colors()
    ps.set_nature_rc()
    rows = ["whisker", "auditory", "bimodal"]       # user 2026-11-19: spontaneous-lick row removed
    titles = {"whisker": "Whisker", "auditory": "Auditory", "lick": "Lick", "bimodal": "Bimodal"}
    print(f"sensory {100 * u.sensory.mean():.1f} %: whisker only {100 * u.w_only.mean():.1f}, auditory only "
          f"{100 * u.a_only.mean():.1f}, both {100 * u.bimodal.mean():.1f}")
    H = TOP + ROW_H * len(rows) + BOT
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
    out = []
    for i, row in enumerate(rows):
        y = TOP + i * ROW_H
        schematic(mm_ax(fig, H, COL_X[0] + 3, y + 0.5, COL_W[0] - 3, ROW_H - 2), row)
        fig.text(COL_X[0] / W, 1 - (y + ROW_H / 2) / H, titles[row], rotation=90, ha="left", va="center",
                 fontsize=6)
        bars(mm_ax(fig, H, COL_X[1] + 5, y + 1, COL_W[1] - 5, ROW_H - 3), a, row, cols)
        psth(mm_ax(fig, H, COL_X[2] + 5, y + 2.5, COL_W[2] - 5, ROW_H - 7.5), PB if row == "bimodal" else P, row,
             cols, out)
    pd.concat(out).to_csv(fb.OUT / "Figure2F_psth_data.csv", index=False)
    ps.save_figure(fig, fb.OUT, "Figure2F_grid")
    print("saved", fb.OUT / "Figure2F_grid")


if __name__ == "__main__":
    main()
