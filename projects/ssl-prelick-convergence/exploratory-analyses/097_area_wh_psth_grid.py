"""097 -- Whisker-hit PSTHs per area group, stimulus- and lick-aligned, day 0 vs expert, R+ and R- (user 2026-10-11). Exploratory.
Events: active whisker hits of the 051 trial selection (perf != 6, warm-up / A1 rules), stimulus = trial start, first lick = corrected
first lick (start + RT). Units: tested units of 081 (stable, mean pre-lick rate >= 0.1 Hz), grouped by area_group.
Rates in 10-ms bins (stimulus-aligned -0.2 .. +0.5 s; lick-aligned -0.5 .. +0.3 s), 3-bin boxcar.
  Hz   rate minus the unit's baseline, the mean rate in [-1, -0.015] s before trial start over its whisker-hit trials (pre-stimulus
       baseline, used for both alignments, as 062)
  z    the same divided by the across-trial SD of that baseline rate (floor SD_MIN = 0.5 Hz, so silent units do not blow up)
Per session x area group: mean over units (>= MIN_UNITS); figure: mean +- s.e.m. over sessions per cohort x stage (day 0 dashed /
open, expert solid); stage test per time bin: Mann-Whitney day 0 vs expert within cohort, Benjamini-Hochberg over the bins of the
panel; bars under the curves where q < 0.05 (cohort colour). Area groups with >= 3 sessions per cohort x stage. Learners rule.
Output: across_days/<ref>/area_wh_psth/{psth.parquet, area_wh_psth_hz.{png,pdf}, area_wh_psth_z.{png,pdf}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../097_area_wh_psth_grid.py [--n-proc 25] [--replot]
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
OUT = m51.OUTROOT / "area_wh_psth"
BIN, SD_MIN, MIN_UNITS = 0.010, 0.5, 5
WINS = {"stim": (-0.2, 0.5), "lick": (-0.5, 0.3)}
BASE = (-1.0, -0.015)


def session(args):
    sid, units = args
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        U, _ = m51.ru.process_nwb_tables(nwb)
        t, log = m51.select_trials(nwb.trials.to_dataframe(), sid)
    t = t[t.cls == "WH"]
    if len(t) < 4:
        return []
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str))
    U = U.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    st, fl = t.start_time.to_numpy(), t.first_lick_time.to_numpy()
    rows = []
    for ag, g in U.groupby("area_group"):
        if len(g) < MIN_UNITS:
            continue
        acc = {(w, k): [] for w in WINS for k in ("hz", "z")}
        for r in g.itertuples():
            s = np.sort(np.asarray(r.spike_times))
            b = (np.searchsorted(s, st + BASE[1]) - np.searchsorted(s, st + BASE[0])) / (BASE[1] - BASE[0])
            mu, sd = b.mean(), max(b.std(), SD_MIN)
            for w, (lo, hi) in WINS.items():
                ev = st if w == "stim" else fl
                edges = np.arange(lo, hi + BIN / 2, BIN)
                H = np.array([np.histogram(s[np.searchsorted(s, e + lo):np.searchsorted(s, e + hi)] - e, edges)[0] for e in ev]) / BIN
                m = np.convolve(H.mean(0), np.ones(3) / 3, "same") - mu
                acc[(w, "hz")].append(m); acc[(w, "z")].append(m / sd)
        for (w, k), v in acc.items():
            rows.append(dict(session_id=sid, area_group=ag, align=w, norm=k, n_units=len(g), n_wh=len(t), psth=np.mean(v, 0).astype(np.float32)))
    return rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    jobs = [(sid, g[["electrode_group", "cluster_id", "area_group"]]) for sid, g in U0.groupby("session_id")]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    D = pd.DataFrame(res)
    meta = U0.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    D = D.merge(meta, on="session_id", how="left")
    D["psth"] = D.psth.apply(lambda a: a.tolist())
    D.to_parquet(OUT / "psth.parquet", index=False)
    print("computed", len(D), flush=True)


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def figure(D, norm):
    m62 = importlib.import_module("062_pub_convergence_figures")
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup(); gcol = au.get_custom_area_groups_colors()
    D = D[D.norm == norm]
    n = D[D["align"] == "stim"].groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    areas = [a for a in au.get_area_group_custom_order() if a in n.index and all(n.loc[a].get((c, s), 0) >= 3 for c in ("R+", "R-") for s in ("learning", "expert"))]
    fig, axs = plt.subplots(len(areas), 2, figsize=(3.2, 1.45 * len(areas) + 0.4), squeeze=False)
    COH = m62.COH
    for i, ag in enumerate(areas):
        for j, w in enumerate(("stim", "lick")):
            ax = axs[i, j]; lo, hi = WINS[w]; tt = (np.arange(lo, hi + BIN / 2, BIN)[:-1] + BIN / 2) * 1e3
            tt = tt[1:-1]                                                  # edge bins of the 3-bin boxcar dropped
            d = D[(D.area_group == ag) & (D["align"] == w)]
            yb = []
            for c in ("R+", "R-"):
                X = {}
                for s_, ls, a_ in (("learning", (0, (2.5, 1.5)), 0.55), ("expert", "-", 1.0)):
                    g = d[(d.cohort == c) & (d.stage == s_)]
                    A = np.stack(g.psth.to_numpy()) if len(g) else np.zeros((0, len(tt))); X[s_] = A
                    if len(A):
                        mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * A.mean(0)
                        ax.fill_between(tt, mu - se, mu + se, color=COH[c], alpha=0.12 * a_ + 0.05, lw=0)
                        ax.plot(tt, mu, color=COH[c], lw=0.8, ls=ls, alpha=a_, label=f"{c.replace('-', '−')} {'D0' if s_ == 'learning' else 'E'} ({len(A)})")
                if len(X["learning"]) >= 3 and len(X["expert"]) >= 3:
                    p = [stats.mannwhitneyu(X["learning"][:, k], X["expert"][:, k]).pvalue for k in range(len(tt))]
                    yb.append((c, bh(p) < 0.05))
            yl = ax.get_ylim(); r_ = yl[1] - yl[0]
            for k_, (c, sig) in enumerate(yb):
                y0 = yl[0] - (0.06 + 0.06 * k_) * r_
                ax.plot(np.where(sig, tt, np.nan), np.full(len(tt), y0), color=COH[c], lw=2, solid_capstyle="butt")
            ax.set_ylim(yl[0] - 0.2 * r_, yl[1])
            ax.axvline(0, color="0.4", lw=0.5, ls=":"); ax.axhline(0, color="0.85", lw=0.4)
            if w == "lick":
                ax.axvspan(-100, 0, color="#FDD49E", alpha=0.4, lw=0, zorder=0)
            ax.set_box_aspect(1); ax.tick_params(labelsize=4.4)
            if i == 0:
                ax.set_title("whisker hit, stimulus-aligned" if w == "stim" else "whisker hit, lick-aligned", fontsize=5.2)
            if i == len(areas) - 1:
                ax.set_xlabel("time from stimulus (ms)" if w == "stim" else "time from first lick (ms)", fontsize=4.8)
            if j == 0:
                ax.set_ylabel(f"{ag}\n{'Δ rate (spk/s)' if norm == 'hz' else 'Δ rate (baseline SD)'}", fontsize=4.6, color=gcol.get(ag, "k"))
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=3.8, loc="upper left", handlelength=1.6)
    fig.suptitle(f"Whisker-hit PSTHs per area group ({'spk/s, baseline-corrected' if norm == 'hz' else 'standardised by baseline SD'}); "
                 "bars: day 0 vs expert, q < 0.05", fontsize=5.4, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"area_wh_psth_{norm}.{ext}", dpi=300)
    print("saved", norm, len(areas), "areas")


def main(a):
    if not a.replot:
        compute(a.n_proc)
    m61 = importlib.import_module("061_roc_prelick_learners")
    D = m61.learner_filter(pd.read_parquet(OUT / "psth.parquet"))
    D["psth"] = D.psth.apply(lambda v: np.asarray(v)[1:-1])          # drop the boxcar edge bins
    for norm in ("hz", "z"):
        figure(D, norm)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
