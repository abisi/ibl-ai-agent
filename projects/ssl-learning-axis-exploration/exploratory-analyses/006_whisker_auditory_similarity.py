"""006 -- Does the whisker-evoked population response become auditory-like across the session? (user 2026-10-16, "Do 3": learning
read as the whisker stimulus acquiring the auditory stimulus's meaning; R- is the control.)
Rates: 005 per-trial matrices. Per unit, evoked = post-stimulus rate - the trial's own pre-trial baseline, scaled by the unit's SD of the
baseline across trials (floor 0.5 spikes/s); units with mean baseline >= 0.1 spikes/s. Bins that end after the trial's corrected first
lick are masked, so licking trials contribute only pre-lick activity; misses (no lick) are uncontaminated.
Windows: early [0, 100) ms, late [100, 200) ms (mean of the valid 25-ms bins; trial dropped from a window without any valid bin).
Auditory template A = mean evoked vector of auditory hits (>= 5 valid). Per whisker trial (hits and misses):
  pA      (W . A) / |A|^2  -- the whisker response along the auditory direction, in units of the auditory response (1 = as large as
          the mean auditory response along its own direction)
  cos     cosine(W, A) (single-trial, noise-limited; used only as a secondary measure)
  pA_loc  pA with a time-local template (the 20 auditory hits nearest in time), against auditory drift
  ref     leave-one-out pA of auditory hits themselves (self-reference, ~1 minus template noise)
Levels: whole brain and area groups (>= 10 units). Per session, subset (all whisker trials / hits / hits with RT >= 250 ms, so the window
  ends >= 150 ms before the lick, against lick-preparation and RT differences between stages / misses) and window:
  lvl20   mean pA over the first 20 whisker trials; lvl mean pA over the session
  s60     slope of pA per 10 whisker trials over the first 60 whisker trials (same exposure range at both stages); s_all whole session
  hm      hits - misses mean pA
Statistics: common.measure_stats (day 0 / expert vs 0, day 0 vs expert MWU + Welch, R+ vs R-, cohort x stage mouse permutation), learners
rule on day 0, session = unit; area level: area groups with >= 3 sessions at both stages within a cohort, BH over areas per measure.
Output: tables/006_trials.parquet, 006_sessions.csv, 006_stats.csv, 006_area_stats.csv; figures/006_*.{png,pdf,svg}
Run (haas, prelick environment): python 006_whisker_auditory_similarity.py [--n-proc 25] [--plot-only]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C

SRC = C.TAB / "005_trial_windows"
WIN = {"early": (0.0, 0.100), "late": (0.100, 0.200)}
MIN_FR, SD_FLOOR, MIN_UNITS, N_LOC, MIN_A = 0.1, 0.5, 10, 20, 5
N_FIRST, N_LVL, MIN_T = 60, 20, 8
SUBSETS = ("all", "hit", "hitslow", "miss")
RT_SLOW = 0.25


def window_matrix(z, keep):
    """units x trials x windows evoked (z-scored by baseline SD), NaN where no valid pre-lick bin"""
    base, post = z["base"][keep].astype(float), z["post"][keep].astype(float)
    sd = np.maximum(base.std(1), SD_FLOOR)
    ev = (post - base[:, :, None]) / sd[:, None, None]
    e, bw, rt = z["post_e"], float(z["bw"]), z["rt"]
    valid = ~((e[None, :] + bw) > np.where(np.isfinite(rt), rt, np.inf)[:, None])           # trials x bins
    out = {}
    for w, (lo, hi) in WIN.items():
        bins = (e >= lo - 1e-9) & (e + bw <= hi + 1e-9)
        m = valid[:, bins]
        num = np.where(m[None], ev[:, :, bins], 0).sum(2); n = m.sum(1)
        out[w] = np.where(n[None] > 0, num / np.maximum(n, 1)[None], np.nan)
    return out


def session(sid):
    z = np.load(SRC / f"{sid}.npz", allow_pickle=True)
    keep = z["base"].mean(1) >= MIN_FR
    ag = z["area_group"][keep].astype(str)
    X = window_matrix(z, keep)
    tt, lick, st = z["trial_type"], z["lick"], z["start_time"]
    iW, iA = np.where(tt == "W")[0], np.where((tt == "A") & lick)[0]
    levels = [("whole brain", np.ones(len(ag), bool))] + [(a, ag == a) for a in np.unique(ag) if (ag == a).sum() >= MIN_UNITS]
    rows = []
    for lev, um in levels:
        for w, M in X.items():
            M = M[um]
            a_ok = iA[np.isfinite(M[0, iA])]
            if len(a_ok) < MIN_A:
                continue
            A = M[:, a_ok].mean(1); nA = A @ A
            if nA <= 0:
                continue
            ref = np.mean([(M[:, i] @ (A * len(a_ok) - M[:, i]) / (len(a_ok) - 1)) / nA for i in a_ok])
            for i in iW:
                x = M[:, i]
                if not np.isfinite(x[0]):
                    continue
                loc = a_ok[np.argsort(np.abs(st[a_ok] - st[i]))[:N_LOC]]; Al = M[:, loc].mean(1)
                nx = np.sqrt(x @ x)
                rows.append(dict(session_id=sid, level=lev, n_units=int(um.sum()), window=w, trial_index=int(i), wtrial=int(z["wtrial"][i]),
                                 tau=float(z["tau"][i]), hit=bool(lick[i]), rt=float(z["rt"][i]), pA=float(x @ A / nA),
                                 cos=float(x @ A / (nx * np.sqrt(nA))) if nx > 0 else np.nan,
                                 pA_loc=float(x @ Al / (Al @ Al)) if Al @ Al > 0 else np.nan, ref=float(ref)))
    return rows


def slope(d, col):
    return np.polyfit(d.wtrial, d[col], 1)[0] * 10 if len(d) >= MIN_T and d.wtrial.nunique() > 2 else np.nan


def session_measures(T):
    rows = []
    for (sid, lev, w), g in T.groupby(["session_id", "level", "window"]):
        o = dict(session_id=sid, level=lev, window=w, ref=g.ref.iloc[0], n_units=g.n_units.iloc[0])
        for sub in SUBSETS:
            d = g if sub == "all" else (g[g.hit & (g.rt >= RT_SLOW)] if sub == "hitslow" else g[g.hit == (sub == "hit")])
            for col in ("pA", "pA_loc", "cos"):
                k = f"{col}_{sub}"
                first = d[d.wtrial < N_FIRST]
                o[f"lvl20_{k}"] = d[d.wtrial < N_LVL][col].mean() if (d.wtrial < N_LVL).sum() >= MIN_T // 2 else np.nan
                o[f"lvl_{k}"] = d[col].mean() if len(d) >= MIN_T else np.nan
                o[f"s60_{k}"] = slope(first, col); o[f"sall_{k}"] = slope(d, col)
        o["hm_pA"] = o["lvl_pA_hit"] - o["lvl_pA_miss"]
        rows.append(o)
    return pd.DataFrame(rows)


MEAS = [("lvl_pA_all", "all WT: level"), ("lvl_pA_hit", "hits: level"), ("lvl_pA_hitslow", "hits RT ≥ 250 ms: level"),
        ("lvl_pA_miss", "misses: level"), ("hm_pA", "hits − misses"), ("s60_pA_all", "all WT: slope / 10 WT\n(first 60)"),
        ("s60_pA_hit", "hits: slope / 10 WT\n(first 60)"), ("s60_pA_miss", "misses: slope / 10 WT\n(first 60)")]
EXTRA = [("lvl20_pA_all", ""), ("lvl20_pA_miss", ""), ("s60_pA_loc_miss", ""), ("sall_pA_miss", ""), ("sall_pA_all", "")]


def main(a):
    meta = pd.read_csv(SRC / "sessions.csv")
    meta = meta[meta.get("error").isna()] if "error" in meta else meta
    if not a.plot_only:
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            res = pool.map(session, meta.session_id.tolist())
        T = pd.DataFrame([r for rr in res for r in rr]).merge(meta[["session_id", "mouse_id", "cohort", "stage"]], on="session_id")
        T.to_parquet(C.TAB / "006_trials.parquet")
        S = session_measures(T).merge(meta[["session_id", "mouse_id", "cohort", "stage"]], on="session_id")
        S = C.population(S); S.to_csv(C.TAB / "006_sessions.csv", index=False)
        R = pd.DataFrame([dict(r, window=w) for w in WIN for m, _ in MEAS + EXTRA for r in C.measure_stats(S[(S.level == "whole brain") & (S.window == w)], m, "whole brain", m)])
        R.to_csv(C.TAB / "006_stats.csv", index=False)
        AR = []
        for w in WIN:
            for lev, g in S[(S.level != "whole brain") & (S.window == w)].groupby("level"):
                for m in ("lvl_pA_all", "lvl_pA_hit", "lvl_pA_hitslow", "lvl_pA_miss", "s60_pA_miss", "s60_pA_all"):
                    for r in C.measure_stats(g, m, lev, m):
                        if r["cohort"] != "R+ vs R-":
                            n = g[g.cohort == r["cohort"]].groupby("stage").size()
                            if min(n.get("learning", 0), n.get("expert", 0)) < 3:
                                continue
                        AR.append(dict(r, window=w))
        AR = pd.DataFrame(AR)
        for (w, m, c), g in AR.groupby(["window", "measure", "cohort"]):
            for p in ("p_d0", "p_e", "p_stage_mwu"):
                if p in g:
                    AR.loc[g.index, f"q_{p}"] = C.bh(g[p].to_numpy())
        AR.to_csv(C.TAB / "006_area_stats.csv", index=False)
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    T = pd.read_parquet(C.TAB / "006_trials.parquet"); S = pd.read_csv(C.TAB / "006_sessions.csv"); R = pd.read_csv(C.TAB / "006_stats.csv")
    T = T[T.session_id.isin(S.session_id) & (T.level == "whole brain")]
    BIN = 10
    # figure 1: time course (bins of 10 whisker trials, first 100), all / hits / misses x windows; self-reference line
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharey="row")
    for i, w in enumerate(WIN):
        for j, sub in enumerate(("all", "hit", "miss")):
            ax = axs[i, j]
            d = T[(T.window == w) & (T.wtrial < 100)]
            d = d if sub == "all" else d[d.hit == (sub == "hit")]
            d = d.assign(b=(d.wtrial // BIN) * BIN + BIN / 2)
            for coh, stg in C.GROUPS:
                s = d[(d.cohort == coh) & (d.stage == stg)].groupby(["session_id", "b"]).pA.mean().reset_index()
                m = s.groupby("b").pA.agg(["mean", "sem", "size"]); m = m[m["size"] >= 3]
                ax.plot(m.index, m["mean"], color=C.COH[coh], ls="-" if stg == "expert" else "--", lw=1, label=C.GLAB[(coh, stg)])
                ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"], color=C.COH[coh], alpha=0.12 if stg == "expert" else 0.07, lw=0)
            ax.axhline(0, color="0.6", lw=0.5, ls=":")
            ax.set_title(f"{ {'all': 'all whisker trials', 'hit': 'whisker hits (pre-lick bins)', 'miss': 'whisker misses'}[sub]}, {w} {int(WIN[w][0]*1e3)}–{int(WIN[w][1]*1e3)} ms", fontsize=6)
            if j == 0:
                ax.set_ylabel("whisker response along auditory direction\n(pA, 1 = mean auditory response)")
            if i == 1:
                ax.set_xlabel("whisker trial")
    axs[0, 0].legend(frameon=False, fontsize=5)
    fig.suptitle("Whisker-evoked population response projected on the auditory-hit response (whole brain; mean ± s.e.m. over sessions)", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"006_whisker_auditory_timecourse.{ext}", dpi=300)
    plt.close(fig)
    # figure 2: session measures, day 0 vs expert per cohort, both windows
    fig, axs = plt.subplots(2, len(MEAS), figsize=(1.75 * len(MEAS), 5.0))
    for i, w in enumerate(WIN):
        Sw = S[(S.level == "whole brain") & (S.window == w)]
        for j, (m, lab) in enumerate(MEAS):
            Rm = R[(R.window == w) & (R.measure == m)]
            C.dot_panel(axs[i, j], Sw, m, Rm, f"{lab}\n({w})", rc=Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"006_whisker_auditory_sessions.{ext}", dpi=300)
    plt.close(fig)
    # figure 3: area groups, misses slope (first 60 WT) and level, day 0 vs expert per cohort, both windows
    AS = S[S.level != "whole brain"]
    try:
        from ephys_utilities.allen_utils import get_custom_area_groups_colors
        acol = get_custom_area_groups_colors()
    except Exception:  # noqa: BLE001
        acol = {}
    AR = pd.read_csv(C.TAB / "006_area_stats.csv")
    areas = sorted(AR.level.unique())
    AM = [("lvl_pA_all", "all whisker trials"), ("lvl_pA_hit", "hits"), ("lvl_pA_hitslow", "hits, RT ≥ 250 ms"), ("lvl_pA_miss", "misses")]
    fig, axs = plt.subplots(2, len(AM), figsize=(9.0, 2 * (0.2 * len(areas) + 0.9)), sharey=True, squeeze=False)
    for i, w in enumerate(WIN):
        for j, (m, mlab) in enumerate(AM):
            ax = axs[i, j]
            for k, (coh, stg) in enumerate(C.GROUPS):
                g = AS[(AS.window == w) & (AS.cohort == coh) & (AS.stage == stg)].groupby("level")[m].agg(["mean", "sem", "count"]).reindex(areas)
                y = np.arange(len(areas)) + (k - 1.5) * 0.18
                ok = g["count"] >= 3
                ax.errorbar(g["mean"][ok], y[ok], xerr=g["sem"][ok], fmt="o", ms=2.5, lw=0.6, color=C.COH[coh],
                            mfc="white" if stg == "learning" else C.COH[coh], label=C.GLAB[(coh, stg)])
            for yy, ar in enumerate(areas):
                for coh, dx in (("R+", 0), ("R-", 1)):
                    r = AR[(AR.window == w) & (AR.measure == m) & (AR.level == ar) & (AR.cohort == coh)]
                    if len(r) and np.isfinite(r.p_stage_mwu.iloc[0]) and r.p_stage_mwu.iloc[0] < 0.05:
                        q = r.q_p_stage_mwu.iloc[0]
                        ax.text(0.92 + 0.05 * dx, yy, "**" if q < 0.05 else "*", transform=ax.get_yaxis_transform(), color=C.COH[coh], va="center", fontsize=6)
            ax.axvline(0, color="0.6", lw=0.5, ls=":")
            ax.set_yticks(range(len(areas)), areas)
            for t_ in ax.get_yticklabels():
                t_.set_color(acol.get(t_.get_text(), "k"))
            ax.set_title(f"{mlab}, {w} {int(WIN[w][0]*1e3)}–{int(WIN[w][1]*1e3)} ms", fontsize=6)
            if i == 1:
                ax.set_xlabel("pA (mean over session)")
    axs[0, 0].legend(frameon=False, fontsize=4.5, loc="lower right")
    fig.text(0.5, 0.005, "* day 0 vs expert p < 0.05 (MWU), ** BH q < 0.05 over areas; colour = cohort", ha="center", fontsize=5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"006_whisker_auditory_areas.{ext}", dpi=300)
    plt.close(fig)
    print("figures done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot()
    else:
        main(a)
