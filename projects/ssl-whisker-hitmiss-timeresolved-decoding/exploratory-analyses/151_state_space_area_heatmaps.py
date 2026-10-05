"""151 -- State-space variants and heatmap summaries of the passive pre -> post changes, whole brain and per area group (user
2026-10-05: "make also these state spaces figures with y axis as whisker axis (passive pre) and auditory axis (passive pre) ...
show the changes in heatmap matrix format for conciseness ... repeat analysis per area_group, on the areas that are sampled in
both cohorts with at least 3 sessions").
Inputs: 146_choice_axis_readout.parquet (stable units, epoch baseline; state-space coordinates ss_ / ss2_ / ss3_ and the
linear-shift excess per area group), 135_alignment_epochs_tracked.parquet (lick-axis shift excess per area group).
State space (146): x = unit hit - miss coding direction from all active whisker trials; y = one of three passive-pre patterns,
orthogonalised to x: whisker-evoked pattern (ss_), whisker - auditory difference (ss2_), auditory-evoked pattern (ss3_). Each
condition mean (evoked 5-35 ms, z units) is projected on x and y; displacement = passive post - passive pre per session.
Figures (figures/publication/):
  151_state_space_variants   3 rows (y = whisker, whisker - auditory, auditory) x 2 cohorts, mean over sessions
  151_state_space_heatmap    whole brain: rows = whisker, auditory, whisker - auditory displacement; columns = x and the three y
                             axes; panels R+ mean, R- mean, R- minus R+ (stars: * one, ** both tests p < 0.05)
  151_area_heatmap           rows = whole brain + area groups with >= 3 sessions in each cohort (>= 20 tracked units); columns =
                             state-space displacement along x (W, A, W - A) and the excess over the linear-shift null (146 decoder
                             readout W, W - A; 135 lick axis raw cos W, W - A projection); colour scaled per column
Stats: 151_stats.csv (area, measure, cohort, n, mean, p_nonparam, p_param; within cohort vs 0 Wilcoxon | t, R+ vs R-
Mann-Whitney | Welch). Session = unit; uncorrected.
Run (haas, repo root): python .../151_state_space_area_heatmaps.py
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

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
WC, AC = "#f7b519", "#2c2cdb"
MIN_SESS = 3
YVAR = [("ss", "whisker pattern"), ("ss2", "whisker - auditory"), ("ss3", "auditory pattern")]
CONDS = ["pre_W", "pre_A", "post_W", "post_A", "act1_hit", "act1_miss", "act2_hit", "act2_miss", "act1_A", "act2_A"]


def stars(pn, pp):
    k = int(np.isfinite(pn) and pn < 0.05) + int(np.isfinite(pp) and pp < 0.05)
    return "**" if k == 2 else "*" if k == 1 else ""


def test_col(d, col, area, rows):
    """within-cohort vs 0 and R+ vs R-; returns means and stars per panel"""
    v = {c: d[d.reward_group == c][col].dropna().to_numpy(float) for c in COH}
    out = {}
    for c in COH:
        pw, pt, n = H.one_sample(v[c])
        rows.append(dict(area=area, measure=col, cohort=c, n=n, mean=np.mean(v[c]) if len(v[c]) else np.nan, p_nonparam=pw, p_param=pt))
        out[c] = (np.mean(v[c]) if len(v[c]) else np.nan, stars(pw, pt))
    mw, we = H.unpaired(v["R+"], v["R-"])
    diff = (np.mean(v["R-"]) - np.mean(v["R+"])) if len(v["R+"]) and len(v["R-"]) else np.nan
    rows.append(dict(area=area, measure=col, cohort="R- minus R+", n=np.nan, mean=diff, p_nonparam=mw, p_param=we))
    out["diff"] = (diff, stars(mw, we))
    return out


def add_disp(d):
    d = d.copy()
    for s in ("W", "A"):
        d[f"dx_{s}"] = d[f"ss_post_{s}_x"] - d[f"ss_pre_{s}_x"]
        for pre, _ in YVAR:
            if f"{pre}_post_{s}_y" in d:
                d[f"d{pre}y_{s}"] = d[f"{pre}_post_{s}_y"] - d[f"{pre}_pre_{s}_y"]
    d["dx_WA"] = d.dx_W - d.dx_A
    for pre, _ in YVAR:
        if f"d{pre}y_W" in d:
            d[f"d{pre}y_WA"] = d[f"d{pre}y_W"] - d[f"d{pre}y_A"]
    return d


def heat(ax, M, S, rlab, clab, title, scale_cols=False, cmap="RdBu_r", lim=None):
    """M values (rows x cols), S star strings; colour symmetric, optionally scaled per column"""
    C = M.copy()
    if scale_cols:
        mx = np.nanmax(np.abs(M), axis=0); mx[~np.isfinite(mx) | (mx == 0)] = 1
        C = M / mx
        lim = 1.0
    elif lim is None:
        lim = np.nanmax(np.abs(M)) or 1.0
    ax.imshow(C, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}\n{S[i][j]}" if S[i][j] else f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=4.6,
                        color="k")
    ax.set_xticks(range(len(clab))); ax.set_xticklabels(clab, rotation=40, ha="right", fontsize=5)
    ax.set_yticks(range(len(rlab))); ax.set_yticklabels(rlab, fontsize=5.5)
    ax.set_title(title, fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_variants(d, scope):
    fig, axes = plt.subplots(3, 2, figsize=(5.2, 7.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.07, hspace=0.55, wspace=0.35)
    for r, (pre, lab) in enumerate(YVAR):
        for k, c in enumerate(COH):
            ax = axes[r, k]; g = d[d.reward_group == c]
            P = {cn: (g[f"ss_{cn}_x"].mean(), g[f"{pre}_{cn}_y"].mean()) for cn in CONDS}
            for cn, (x, y) in P.items():
                col = WC if cn.endswith("W") else AC if cn.endswith("A") else ("0.15" if "hit" in cn else "0.6")
                mk = "^" if "hit" in cn else "v" if "miss" in cn else "o" if cn.startswith("pre") else "s" if cn.startswith("post") else "D"
                ax.plot(x, y, mk, color=col, ms=4.2 if cn[:3] in ("pre", "pos") else 3.4, mfc="white" if cn.startswith("pre") else col, mew=0.9)
            for s, col in (("W", WC), ("A", AC)):
                ax.annotate("", xy=P[f"post_{s}"], xytext=P[f"pre_{s}"], arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
            ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
            ax.set_xlabel("choice axis (hit - miss)", fontsize=6)
            if k == 0:
                ax.set_ylabel(f"y = passive-pre {lab}\n(orthogonalised to x)", fontsize=6)
            ax.set_title(f"{c} (n = {len(g)})", color=COL[c], fontsize=6.5)
    fig.suptitle(f"State space, three y axes (whole brain, stable units, {scope}); o pre, s post (whisker yellow, auditory blue),\n"
                 "^ v active hits / misses (1st, 2nd half), D active auditory; mean over sessions, z units", fontsize=6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"151_state_space_variants_{scope}.{ext}", dpi=300)
    plt.close(fig)


def fig_wb_heat(d, scope, rows):
    rl = ["whisker", "auditory", "whisker - auditory"]
    cl = ["x: choice axis", "y: whisker pattern", "y: whisker - auditory", "y: auditory pattern"]
    cols = [["dx_W", "dssy_W", "dss2y_W", "dss3y_W"], ["dx_A", "dssy_A", "dss2y_A", "dss3y_A"], ["dx_WA", "dssy_WA", "dss2y_WA", "dss3y_WA"]]
    res = {key: (np.full((3, 4), np.nan), [[""] * 4 for _ in range(3)]) for key in ("R+", "R-", "diff")}
    for i, rr in enumerate(cols):
        for j, c in enumerate(rr):
            o = test_col(d, c, "whole_brain", rows)
            for key in res:
                res[key][0][i, j], res[key][1][i][j] = o[key]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.8, bottom=0.32, wspace=0.45)
    lim = np.nanmax(np.abs(np.r_[res["R+"][0].ravel(), res["R-"][0].ravel()]))
    for ax, key, t in zip(axes, ("R+", "R-", "diff"), ("R+: post - pre", "R-: post - pre", "R- minus R+")):
        M, S = res[key]
        heat(ax, M, S, rl if key == "R+" else [""] * len(rl), cl, t, lim=None if key == "diff" else lim)
    fig.suptitle(f"Passive pre -> post displacement in the state space (whole brain, stable units, {scope}; z units; stars: * one, "
                 "** both tests p < 0.05; within cohort vs 0, R- minus R+: Mann-Whitney | Welch)", fontsize=5.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"151_state_space_heatmap_{scope}.{ext}", dpi=300)
    plt.close(fig)


AREA_COLS = [("dx_W", "W along x\n(state space)"), ("dx_A", "A along x\n(state space)"), ("dx_WA", "W - A along x\n(state space)"),
             ("shift_excess_dW", "decoder readout W\n(excess, 146)"), ("shift_excess_dWA", "decoder W - A\n(excess, 146)"),
             ("lick_dWR", "lick axis W raw cos\n(excess, 135)"), ("lick_dWAP", "lick axis W - A proj.\n(excess, 135)")]


def fig_area(d146, d135, scope, rows):
    d146 = add_disp(d146)
    a = d135.rename(columns={"shift_excess_dWR": "lick_dWR", "shift_excess_dWAP": "lick_dWAP"})
    a["area"] = a.area.replace({"All units": "whole_brain"})
    a = a[(a.area == "whole_brain") | (a.n_units >= 20)]
    m = d146.merge(a[["session_id", "area", "lick_dWR", "lick_dWAP"]], on=["session_id", "area"], how="left")
    cnt = m.dropna(subset=["dx_W"]).groupby(["area", "reward_group"]).session_id.nunique().unstack(fill_value=0)
    keep = [x for x in cnt.index if min(cnt.loc[x].get("R+", 0), cnt.loc[x].get("R-", 0)) >= MIN_SESS]
    keep = ["whole_brain"] + sorted(k for k in keep if k != "whole_brain")
    res = {key: (np.full((len(keep), len(AREA_COLS)), np.nan), [[""] * len(AREA_COLS) for _ in keep]) for key in ("R+", "R-", "diff")}
    for i, ar in enumerate(keep):
        g = m[m.area == ar]
        for j, (c, _) in enumerate(AREA_COLS):
            if c in g and g[c].notna().sum() >= 2 * MIN_SESS:
                o = test_col(g, c, ar, rows)
                for key in res:
                    res[key][0][i, j], res[key][1][i][j] = o[key]
    rl = [f"{ar} ({int(cnt.loc[ar].get('R+', 0))} | {int(cnt.loc[ar].get('R-', 0))})" for ar in keep]
    cl = [lab for _, lab in AREA_COLS]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.9 + 0.32 * len(keep)))
    fig.subplots_adjust(left=0.12, right=0.995, top=0.86, bottom=0.25, wspace=0.55)
    for ax, key, t in zip(axes, ("R+", "R-", "diff"), ("R+: post - pre", "R-: post - pre", "R- minus R+")):
        heat(ax, res[key][0], res[key][1], rl if key == "R+" else [""] * len(rl), cl, t, scale_cols=True)
    fig.suptitle(f"Passive pre -> post changes per area group (stable units, {scope}; rows: area (sessions R+ | R-), >= {MIN_SESS} "
                 "sessions per cohort and >= 20 units; colour scaled per column; stars: * one, ** both tests p < 0.05)", fontsize=6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"151_area_heatmap_{scope}.{ext}", dpi=300)
    plt.close(fig)


def main():
    H.setup()
    D = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    D = D[D.skipped_reason.isna() & (D.unit_set == "stable") & (D.response == "epochbase")]
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    A = A[A.skipped_reason.isna()] if "skipped_reason" in A else A
    rows = []
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for scope in ("all", "learners"):
        d, a = D, A
        if scope == "learners":
            d = D[H.mouse_of(D).isin(H.learners())]; a = A[H.mouse_of(A).isin(H.learners())]
        wb = add_disp(d[d.area == "whole_brain"])
        fig_variants(wb, scope)
        r0 = len(rows)
        fig_wb_heat(wb, scope, rows)
        for r in rows[r0:]:
            r["panel"] = "whole-brain state space"
        r1 = len(rows)
        fig_area(d, a, scope, rows)
        for r in rows[r1:]:
            r["panel"] = "area groups"
        for r in rows[r0:]:
            r["scope"] = scope
    R = pd.DataFrame(rows)
    R.to_csv(EA / "151_stats.csv", index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
