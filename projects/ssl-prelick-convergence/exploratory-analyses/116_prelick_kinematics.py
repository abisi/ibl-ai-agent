"""116 -- Pre-lick kinematics of the events behind the reward-lick CD (user 2026-10-12: "do pre-lick kinematics figures"; movement
control for 087-114). Exploratory.
Events: exactly those of 105 (lick_t, lick_lab): WH and AH = corrected first lick, start_time + (lick_time - response_window_start_time)
(skill ssl-lick-alignment); SL = 051 spontaneous lick-bout onsets (piezo; uncued, unrewarded). SL times are on the piezo clock, which
differs from the corrected trial lick by a session-specific ~+-25 ms: small offsets between SL and WH / AH traces near 0 can be clock.
DLC series and checks: 113 load_series (per-series timestamps and length checks, likelihood >= 0.5; skill ssl-dlc-units):
  jaw opening      jaw_distance in mm (session setup calibration) minus the session median (resting jaw), not trial-baselined
                   (a trial baseline before the lick would contain the stimulus-evoked movement of cued trials)
  P(tongue out)    tongue_likelihood >= 0.5
  whisker speed    |d whisker angle / dt| (deg/s, top camera), absolute
  nose speed       |d nose angle / dt| (deg/s, side camera), absolute
Each event: interpolated on a 5-ms grid, -500 .. +200 ms from the lick (gaps > 50 ms NaN); per session: mean over events of each class
(>= 3 events with > 50 % valid samples). Window of the CD: [-100, 0) ms.
Population: learners rule (m61) for day 0, all expert sessions. Session = unit.
Figure: rows = measures; columns 1-4 = R+ day 0, R+ expert, R- day 0, R- expert (WH / AH / SL traces, mean +- SEM over sessions,
25-ms boxcar for display); column 5 = WH - AH in [-100, 0) ms (088 mean_panel2: day 0 vs expert per cohort, R+ vs R- per stage,
cohort x stage by mouse permutation); WH - SL in the csv.
Output: across_days/<ref>/prelick_kinematics/{traces.parquet, windows.csv, stats_<pop>.csv, prelick_kinematics_<pop>.{png,pdf,svg}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../116_prelick_kinematics.py [--n-proc 20] [--replot]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m113 = importlib.import_module("113_dlc_passive_pre")
m105, m51 = m113.m105, m113.m51
OUT = m51.OUTROOT / "prelick_kinematics"
GRID = np.round(np.arange(-0.5, 0.2 + 1e-9, 0.005), 4)
WIN = (-0.1, 0.0)
MEAS = [("jaw opening", "jaw opening (mm)"), ("tongue out", "P(tongue out)"), ("|whisker velocity|", "whisker speed (°/s)"),
        ("nose movement", "nose speed (°/s)")]
CLS = ("WH", "AH", "SL")


def session(args):
    sid, = args
    f = m105.OUT / f"{sid}.npz"
    if not f.exists():
        return [], []
    z = np.load(f, allow_pickle=True); lab, lt = z["lick_lab"].astype(str), z["lick_t"].astype(float)
    info, tr, series = m113.load_series(sid)
    rows, wins = [], []
    for k, _ in MEAS:
        if k not in series:
            continue
        t, v = series[k]
        if k == "jaw opening":
            v = v - np.nanmedian(v)                                       # opening relative to the resting jaw (mm)
        for c in CLS:
            ev = lt[lab == c]
            if len(ev) < m113.MIN_TR:
                continue
            G = m113.to_grid(t, v, ev, None, grid=GRID); ok = np.isfinite(G).mean(1) > 0.5
            if ok.sum() < m113.MIN_TR:
                continue
            tr_ = np.nanmean(G[ok], 0); w = (GRID >= WIN[0]) & (GRID < WIN[1])
            rows.append(dict(session_id=sid, measure=k, cls=c, n_events=int(ok.sum()), trace=tr_.tolist()))
            wins.append(dict(session_id=sid, measure=k, cls=c, n_events=int(ok.sum()), win=float(np.nanmean(tr_[w])), setup=info.get("setup")))
    return rows, wins


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(s,) for s in meta.session_id])
    pd.DataFrame([x for r, _ in res for x in r]).merge(meta, on="session_id").to_parquet(OUT / "traces.parquet", index=False)
    pd.DataFrame([x for _, w in res for x in w]).merge(meta, on="session_id").to_csv(OUT / "windows.csv", index=False)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH, CL = m88.COH, m88.CL
    plt = m62.setup(); rng = np.random.default_rng(0)
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    T = flt(pd.read_parquet(OUT / "traces.parquet")); W = flt(pd.read_csv(OUT / "windows.csv"))
    P = W.pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "measure"], columns="cls", values="win").reset_index()
    P["WH_AH"], P["WH_SL"] = P.WH - P.AH, P.WH - P.SL
    tms = GRID * 1e3; k5 = np.ones(5) / 5; sm = lambda x: np.convolve(np.nan_to_num(x), k5, "same")
    lab = {"WH": "whisker hit", "AH": "auditory hit", "SL": "spontaneous lick"}
    with plt.rc_context({"font.size": 5.2, "axes.labelsize": 5.2, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(7.2, 6.4))
        gs = fig.add_gridspec(len(MEAS), 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.35, hspace=0.55, left=0.08, right=0.99, top=0.9, bottom=0.07)
        for i, (mz, yl) in enumerate(MEAS):
            axs = []; lo, hi = np.inf, -np.inf
            for j, (coh, st) in enumerate(m88.GROUPS):
                ax = fig.add_subplot(gs[i, j]); axs.append(ax)
                for c in CLS:
                    d = T[(T.measure == mz) & (T.cls == c) & (T.cohort == coh) & (T.stage == st)]
                    if len(d) < 2:
                        continue
                    A = np.stack([sm(np.asarray(x)) for x in d.trace]); mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(tms, mu - se, mu + se, color=CL["FA" if c == "SL" else c], alpha=0.2, lw=0)
                    ax.plot(tms, mu, color=CL["FA" if c == "SL" else c], lw=0.8, label=f"{lab[c]} ({len(d)})")
                    lo, hi = min(lo, np.nanmin(mu - se)), max(hi, np.nanmax(mu + se))
                ax.axvspan(WIN[0] * 1e3, WIN[1] * 1e3, color="#FDD49E", alpha=0.45, lw=0, zorder=0); ax.axvline(0, color="0.4", lw=0.4, ls=":")
                ax.set_xlim(tms[0], tms[-1]); ax.set_xticks([-400, -200, 0, 200])
                if i == 0:
                    ax.set_title(f"{coh.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'}", color=COH[coh], fontsize=5.6)
                if j == 0:
                    ax.set_ylabel(yl)
                else:
                    ax.tick_params(labelleft=False)
                if i == len(MEAS) - 1:
                    ax.set_xlabel("time from lick (ms)")
                else:
                    ax.tick_params(labelbottom=False)
                ax.legend(frameon=False, fontsize=3.4, loc="upper left", handlelength=1.0, borderaxespad=0.1, labelspacing=0.15)
            pad = 0.04 * (hi - lo)
            for ax in axs:
                ax.set_ylim(lo - pad, hi + pad)
            ax = fig.add_subplot(gs[i, 4]); ax.set_box_aspect(0.9)
            m88.mean_panel2(ax, P[P.measure == mz], "WH_AH", "WH − AH, −100–0 ms", rng, f"116 {mz} WH-AH", yl.split(" (")[0], ref=0)
        fig.suptitle(f"Kinematics before the lick for the events of the reward-lick CD (WH, AH: corrected first lick; SL: piezo bout onset; {pop})",
                     x=0.01, y=0.995, ha="left", fontsize=6.4, weight="bold")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"prelick_kinematics_{pop}.{ext}", dpi=300)
    # stats WH - AH and WH - SL
    for mz, _ in MEAS:
        m62.group_stats(P[P.measure == mz], "WH_SL", f"{mz} WH-SL", rng, f"116 {mz} WH-SL")
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("116")]); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning",
              "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    # vs zero: is WH matched to AH in the CD window?
    from scipy import stats
    for mz, _ in MEAS:
        for (coh, st) in m88.GROUPS:
            v = P[(P.measure == mz) & (P.cohort == coh) & (P.stage == st)].WH_AH.dropna()
            if len(v) >= 5:
                print(f"{mz:20s} {coh} {st:8s} WH-AH mean {v.mean():8.4f}  Wilcoxon vs 0 p = {stats.wilcoxon(v).pvalue:.3g}  (n = {len(v)})")


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=20)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
