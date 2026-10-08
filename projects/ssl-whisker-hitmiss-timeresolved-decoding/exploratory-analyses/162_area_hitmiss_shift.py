"""162 -- Per-area breakdown of the passive whisker response shift along the active hit / miss axis (user 2026-10-08: "a similar
breakdown per area of that shift along the hit / miss", as the per-area breakdown of the pre-lick drift).
Data: 135 (shared tracked stable units; evoked 5-35 ms, epoch baseline), one lick axis per area group built from that group's units
(active whisker hits - misses, held-out halves); learners; area groups with >= 3 sessions in each cohort (>= 20 units for the
shift null, as 135).
Columns (native scale: projection on the unit hit - miss axis / sqrt(n units)):
  a  post - pre change of the passive whisker projection
  b  the same beyond session time (excess over the linear-shift null, shift_excess_dWP)
  c  whisker - auditory, beyond session time (shift_excess_dWAP)
Per area group: R+ and R- mean +- s.e.m. over sessions (filled: Wilcoxon vs 0 p < 0.05); right of each panel: R+ vs R- Mann-Whitney p
and its Benjamini-Hochberg q over the area groups shown. Whole brain on top. Colours of the area labels: custom area-group palette.
Outputs: figures/publication/162_area_hitmiss_shift_learners.{png,pdf,svg}; 162_stats_learners.csv
Run (haas, repo root): python .../162_area_hitmiss_shift.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
MEAS = [("dW", "Whisker: post − pre\n(projection, per unit)"), ("shift_excess_dWP", "Whisker: beyond\nsession time"),
        ("shift_excess_dWAP", "Whisker − auditory:\nbeyond session time")]


def area_colors():
    try:
        from ephys_utilities import allen_utils
        return allen_utils.get_custom_area_groups_colors()
    except Exception:  # noqa: BLE001
        return {}


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def main():
    H.setup()
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    A = A[A.skipped_reason.isna()]; A = A[H.mouse_of(A).isin(H.learners())].copy()
    A["dW"] = A.evokedW_proj_passive_post - A.evokedW_proj_passive_pre
    n = A.groupby(["area", "reward_group"]).shift_excess_dWP.apply(lambda x: x.notna().sum()).unstack().fillna(0)
    areas = [a for a in n.index if a != "All units" and min(n.loc[a, "R+"], n.loc[a, "R-"]) >= 3]
    order = ["All units"] + sorted(areas, key=lambda a: -(n.loc[a].sum()))
    rows = []
    for m, _ in MEAS:
        for a in order:
            v = {c: A[(A.area == a) & (A.reward_group == c)][m].dropna().to_numpy(float) for c in COH}
            r = dict(measure=m, area=a, n_Rp=len(v["R+"]), n_Rm=len(v["R-"]), mean_Rp=v["R+"].mean(), sem_Rp=H.sem(v["R+"]),
                     mean_Rm=v["R-"].mean(), sem_Rm=H.sem(v["R-"]))
            for c, k in (("R+", "Rp"), ("R-", "Rm")):
                r[f"p0_{k}"] = stats.wilcoxon(v[c]).pvalue if len(v[c]) > 5 else np.nan
            r["p_MWU"] = stats.mannwhitneyu(v["R+"], v["R-"]).pvalue if min(len(v["R+"]), len(v["R-"])) >= 3 else np.nan
            rows.append(r)
    S = pd.DataFrame(rows)
    for m, _ in MEAS:
        k = (S.measure == m) & (S.area != "All units")
        S.loc[k, "q_BH"] = bh(S.loc[k, "p_MWU"])
    S.to_csv(EA / "162_stats_learners.csv", index=False)
    AC = area_colors()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 0.32 * len(order) + 1.2), sharey=True)
    y = np.arange(len(order))
    for ax, (m, title) in zip(axs, MEAS):
        s = S[S.measure == m].set_index("area").reindex(order)
        for k, (c, kk, dy) in enumerate((("R+", "Rp", -0.15), ("R-", "Rm", 0.15))):
            for i, a in enumerate(order):
                p0 = s.loc[a, f"p0_{kk}"]
                ax.errorbar(s.loc[a, f"mean_{kk}"], i + dy, xerr=s.loc[a, f"sem_{kk}"], fmt="o", ms=3.2, color=COL[c],
                            mfc=COL[c] if np.isfinite(p0) and p0 < 0.05 else "white", lw=0.9, capsize=0)
        for i, a in enumerate(order):
            p, q = s.loc[a, "p_MWU"], s.loc[a, "q_BH"]
            t = H.pnum(p) + ("" if a == "All units" or not np.isfinite(q) else f" (q {H.pnum(q)})")
            ax.text(1.02, i, t, transform=ax.get_yaxis_transform(), va="center", fontsize=4.6,
                    weight="bold" if (a != "All units" and np.isfinite(q) and q < 0.05) or (a == "All units" and p < 0.05) else "normal")
        ax.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 2))); ax.axhline(0.5, color="0.85", lw=0.5)
        ax.set_title(title, pad=6); ax.set_xlabel("change (SD units per neuron)")
    axs[0].set_yticks(y); axs[0].set_yticklabels([f"{'Whole brain' if a == 'All units' else a} ({int(n.loc[a, 'R+'])}/{int(n.loc[a, 'R-'])})" for a in order])
    for t, a in zip(axs[0].get_yticklabels(), order):
        t.set_color(AC.get(a, "black"))
    axs[0].set_ylim(len(order) - 0.5, -0.5)
    fig.suptitle("Passive whisker response along the active hit − miss axis, per area group (learners; R+ green, R− magenta; "
                 "filled: ≠ 0; right: R+ vs R− Mann-Whitney p, BH q over areas)", fontsize=6)
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"162_area_hitmiss_shift_learners.{ext}", dpi=300)
    print(S.round(4).to_string())
    print("ALL DONE", FIGDIR / "162_area_hitmiss_shift_learners.png")


if __name__ == "__main__":
    main()
