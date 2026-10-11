"""099 -- Per-area PSTHs including whisker misses (user 2026-10-11: "whisker misses should be in the picture"). Exploratory.
Trials: 051 active selection (perf != 6, warm-up / A1 rules) before the lick-only classification: whisker hits (WH, lick), whisker
misses (WM, no lick), auditory hits (AH). First lick = start + (lick_time - response_window_start). Misses have no lick: pseudo-lick =
stimulus + the session's median WH reaction time.
Units: tested units of 081 (stable, mean pre-lick rate >= 0.1 Hz) by area group (>= 5 per session). Rates in 10-ms bins, 3-bin
boxcar (edge bins dropped), minus the unit's pre-stimulus baseline per class (mean rate in [-1, -0.015] s before trial start; same
window for every class).
Columns (R+ block top, R- block bottom; day 0 dashed, expert solid; mean +- s.e.m. over sessions)
  1  stimulus-aligned WH (yellow), WM (grey), AH (blue)
  2  stimulus-aligned WH - WM (same stimulus, lick vs no lick)
  3  lick-aligned WH, AH and WM at the pseudo-lick
  4  lick-aligned WH - WM
Window tests: stimulus 50-200 ms; lick [-100, 0) ms; day 0 vs expert within cohort (MWU, titles); cohort x stage (mouse permutation)
in the R- titles; BH over areas in the stats file. Learners rule.
Output: across_days/<ref>/area_psth_misses/{psth.parquet, area_psth_misses.{png,pdf}, stats.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../099_area_psth_misses.py [--n-proc 25] [--replot]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m81 = importlib.import_module("081_single_unit_classes")
OUT = m51.OUTROOT / "area_psth_misses"
BIN, MIN_UNITS, BASE = 0.010, 5, (-1.0, -0.015)
WINS = {"stim": (-0.2, 0.5), "lick": (-0.5, 0.3)}
TWIN = {"stim": (50, 200), "lick": (-100, 0)}


def session(args):
    sid, units = args
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        U, _ = m51.ru.process_nwb_tables(nwb)
        t, _ = m51.active_trials(nwb.trials.to_dataframe(), sid)
    t = t.copy(); rt = (t.lick_time - t.response_window_start_time).to_numpy()
    lick = (t.lick_flag == 1).to_numpy() & np.isfinite(rt)
    isw, isa = (t.trial_type == "whisker_trial").to_numpy(), (t.trial_type == "auditory_trial").to_numpy()
    st = t.start_time.to_numpy()
    cls = {"WH": isw & lick, "WM": isw & ~lick, "AH": isa & lick}
    if min(c.sum() for c in cls.values()) < 3:
        return []
    med = np.nanmedian(rt[cls["WH"]])
    lk = np.where(lick, st + np.nan_to_num(rt), st + med)                 # pseudo-lick for misses
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str)).merge(units, on=["electrode_group", "cluster_id"], how="inner")
    rows = []
    for ag, g in U.groupby("area_group"):
        if len(g) < MIN_UNITS:
            continue
        acc = {(w, c): [] for w in WINS for c in cls}
        for r in g.itertuples():
            s = np.sort(np.asarray(r.spike_times))
            for c, m in cls.items():
                b = ((np.searchsorted(s, st[m] + BASE[1]) - np.searchsorted(s, st[m] + BASE[0])) / (BASE[1] - BASE[0])).mean()
                for w, (lo, hi) in WINS.items():
                    ev = (st if w == "stim" else lk)[m]; edges = np.arange(lo, hi + BIN / 2, BIN)
                    H = np.array([np.histogram(s[np.searchsorted(s, e + lo):np.searchsorted(s, e + hi)] - e, edges)[0] for e in ev]) / BIN
                    acc[(w, c)].append(np.convolve(H.mean(0), np.ones(3) / 3, "same")[1:-1] - b)
        for (w, c), v in acc.items():
            rows.append(dict(session_id=sid, area_group=ag, align=w, cls=c, n_units=len(g), n_trials=int(cls[c].sum()), psth=np.mean(v, 0).tolist()))
    return rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    jobs = [(sid, g[["electrode_group", "cluster_id", "area_group"]]) for sid, g in U0.groupby("session_id")]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    D = pd.DataFrame(res).merge(U0.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]], on="session_id", how="left")
    D.to_parquet(OUT / "psth.parquet", index=False); print("computed", len(D), flush=True)


def figure():
    m88 = importlib.import_module("088_cosyne_across_days_v2")
    m62, m61 = m88.m62, m88.m61
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup(); rng = np.random.default_rng(0); gcol = au.get_custom_area_groups_colors(); COH, CL = m88.COH, m88.CL
    D = m61.learner_filter(pd.read_parquet(OUT / "psth.parquet")); D["psth"] = D.psth.apply(np.asarray)
    W = D.pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "area_group", "align"], columns="cls", values="psth", aggfunc="first").reset_index()
    W = W.dropna(subset=["WH", "WM", "AH"])
    W["D"] = [a - b for a, b in zip(W.WH, W.WM)]
    tt = {w: ((np.arange(lo, hi + BIN / 2, BIN)[:-1] + BIN / 2) * 1e3)[1:-1] for w, (lo, hi) in WINS.items()}
    n = W[W["align"] == "stim"].groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    areas = [a for a in au.get_area_group_custom_order() if a in n.index and all(n.loc[a].get((c, s), 0) >= 3 for c in ("R+", "R-") for s in ("learning", "expert"))]
    NA = len(areas); ST = []
    CC = {"WH": CL["WH"], "AH": CL["AH"], "WM": "0.55"}
    COLS = (("stim", "direct"), ("stim", "D"), ("lick", "direct"), ("lick", "D"))
    fig, axs = plt.subplots(2 * NA, 4, figsize=(4.4, 1.1 * 2 * NA + 0.6), squeeze=False)
    for bi, coh in enumerate(("R+", "R-")):
        for i, ag in enumerate(areas):
            rr = bi * NA + i
            for j, (w, kind) in enumerate(COLS):
                ax = axs[rr, j]; x = tt[w]; d = W[(W.area_group == ag) & (W["align"] == w) & (W.cohort == coh)]
                series = [("WH", CC["WH"]), ("WM", CC["WM"]), ("AH", CC["AH"])] if kind == "direct" else [("D", COH[coh])]
                for k, c in series:
                    for s_, ls, a_ in (("learning", (0, (2.5, 1.5)), 0.6), ("expert", "-", 1.0)):
                        g = d[d.stage == s_]
                        if not len(g):
                            continue
                        A = np.stack(g[k].to_numpy()); mu = A.mean(0); se = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * mu
                        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.12, lw=0); ax.plot(x, mu, color=c, lw=0.8, ls=ls, alpha=a_)
                lo, hi = TWIN[w]; win = (x >= lo) & (x < hi)
                if kind == "direct":
                    ttl = f"D0 {len(d[d.stage == 'learning'])}, E {len(d[d.stage == 'expert'])}"
                else:
                    wd = W[(W.area_group == ag) & (W["align"] == w)].assign(v=lambda q: [a[win].mean() for a in q.D])
                    L_, E_ = (wd[(wd.cohort == coh) & (wd.stage == s_)].v for s_ in ("learning", "expert"))
                    p = stats.mannwhitneyu(L_, E_).pvalue if min(len(L_), len(E_)) >= 3 else np.nan
                    ttl = f"D0 vs E {m62.fmt_p(p)}"
                    ST.append(dict(area=ag, cohort=coh, align=w, mean_D0=L_.mean(), mean_E=E_.mean(), n_D0=len(L_), n_E=len(E_), p=p))
                    if coh == "R-":
                        S = m62.group_stats(wd, "v", f"{ag} {w}", rng, "099"); ip = S["interaction"][1]
                        ttl += f"; c×s {m62.fmt_p(ip)}"; ST.append(dict(area=ag, cohort="interaction", align=w, p=ip))
                ax.set_title(ttl, fontsize=3.8, pad=1.5)
                ax.axvspan(lo, hi, color="#FDD49E" if w == "lick" else "#DDDDDD", alpha=0.4, lw=0, zorder=0)
                ax.axvline(0, color="0.4", lw=0.4, ls=":"); ax.axhline(0, color="0.85", lw=0.4)
                ax.set_xlim(x[0], x[-1]); ax.set_box_aspect(1); ax.tick_params(labelsize=3.9)
                if rr == 0:
                    ax.annotate(["stim: WH, WM, AH", "stim: WH − WM", "lick: WH, AH, WM*", "lick: WH − WM*"][j], xy=(0.5, 1.28),
                                xycoords="axes fraction", ha="center", fontsize=4.8, weight="bold")
                if rr != 2 * NA - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("from stimulus (ms)" if w == "stim" else "from (pseudo-)lick (ms)", fontsize=4.2)
                if j == 0:
                    ax.set_ylabel(f"{m88.m83.m62_short(ag)}\nΔ rate (spk/s)", fontsize=4.1, color=gcol.get(ag, "k"))
    fig.text(0.005, 0.75, "R+", color=COH["R+"], fontsize=9, weight="bold", rotation=90, va="center")
    fig.text(0.005, 0.27, "R−", color=COH["R-"], fontsize=9, weight="bold", rotation=90, va="center")
    fig.suptitle("Per-area PSTHs with whisker misses (* misses at a pseudo-lick: stimulus + median WH RT); dashed day 0, solid expert",
                 fontsize=5.2, y=0.999)
    fig.tight_layout(rect=(0.03, 0, 1, 0.99), h_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"area_psth_misses.{ext}", dpi=300)
    S = pd.DataFrame(ST)
    for w in WINS:
        for c in ("R+", "R-", "interaction"):
            m = (S["align"] == w) & (S.cohort == c); v = S.loc[m, "p"].to_numpy(float); ok = np.isfinite(v); q = np.full(len(v), np.nan)
            if ok.sum():
                o = np.argsort(v[ok]); r_ = v[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1); r_ = np.minimum.accumulate(r_[::-1])[::-1]
                qq = np.empty_like(r_); qq[o] = np.minimum(r_, 1); q[ok] = qq
            S.loc[m, "q_BH"] = q
    S.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(); print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
