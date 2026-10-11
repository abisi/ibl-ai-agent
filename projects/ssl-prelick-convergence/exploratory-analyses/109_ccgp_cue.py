"""109 -- Cross-modal CCGP of the cued-lick code (user 2026-10-11: "what about auditory hits?"). Exploratory.
Auditory misses are almost absent, so auditory hits cannot enter a stimulus x lick 2 x 2 (108). Here the lick is held constant: all three
conditions are licks, lick-aligned (105 lickbins: 50-ms windows, baseline-corrected spikes/s, stable tested units):
  variable  cued (WH or AH) vs spontaneous (SL) lick;  condition  modality of the cue (whisker / auditory)
  SL events split at random into halves A / B per session; training always uses SL-A, testing SL-B (no event shared)
  CCGP_cue  = mean of [train WH vs SL-A -> test AH vs SL-B] and [train AH vs SL-A -> test WH vs SL-B], balanced accuracy
  within    5-fold CV balanced accuracy of WH vs SL and of AH vs SL (all SL)
  PS_cue    cosine between WH - SL-A and AH - SL-B (z-scored units; different SL halves, so shared noise does not inflate it)
Decoder: logistic regression (L2, C = 0.1, balanced class weights), units z-scored on the training events.
Window: mean of the bins centred in [-100, 0) ms; chance = the same with the training labels shuffled (N_SHUF = 20). Time course: CCGP per
50-ms bin, -400 .. +100 ms (chance 0.5 for balanced accuracy; no shuffles per bin). Sessions with >= MIN_EV WH, AH and SL per half.
Output: across_days/<ref>/ccgp/{cue_sessions.parquet, cue_timecourse.parquet, ccgp_cue_<pop>.{png,pdf}, cue_stats_<pop>.csv}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../109_ccgp_cue.py [--n-proc 25] [--replot] [--pop learners]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m108 = importlib.import_module("108_ccgp")
m105, m51 = m108.m105, m108.m51
OUT = m108.OUT
MIN_EV, N_SHUF = 6, 20


def ccgp(X, wh, ah, sa, sb, rng=None):
    if rng is None:
        return 0.5 * (m108.fit_test(X, wh, sa, ah, sb) + m108.fit_test(X, ah, sa, wh, sb))
    y1 = rng.permutation(np.r_[np.ones(len(wh)), np.zeros(len(sa))]); y2 = rng.permutation(np.r_[np.ones(len(ah)), np.zeros(len(sa))])
    return 0.5 * (m108.fit_test(X, wh, sa, ah, sb, y1) + m108.fit_test(X, ah, sa, wh, sb, y2))


def session(args):
    f, = args
    sid = f.stem; z = np.load(f, allow_pickle=True)
    lab, L, c = z["lick_lab"], z["lickbins"].astype(float), z["lick_c"]
    wh, ah, sl = (np.where(lab == k)[0] for k in ("WH", "AH", "SL"))
    r = dict(session_id=sid, n_WH=len(wh), n_AH=len(ah), n_SL=len(sl), n_units=L.shape[0])
    if min(len(wh), len(ah), len(sl) // 2) < MIN_EV:
        return r, []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|109".encode())); seed = int(rng.integers(1e6))
    p = rng.permutation(sl); sa, sb = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
    Xw = L[:, :, (c >= -0.1) & (c < 0.0)].mean(2)
    r["CCGP_cue"] = ccgp(Xw, wh, ah, sa, sb); r["CCGP_cue_null"] = float(np.mean([ccgp(Xw, wh, ah, sa, sb, rng) for _ in range(N_SHUF)]))
    for k, a in (("WH", wh), ("AH", ah)):
        r[f"within_{k}"] = m108.cv_within(Xw, a, sl, seed)
        r[f"within_{k}_null"] = float(np.mean([m108.cv_within(Xw, a, sl, seed, rng.permutation(np.r_[np.ones(len(a)), np.zeros(len(sl))])) for _ in range(N_SHUF)]))
    mu, sd = m108.zfit(Xw[:, np.r_[wh, ah, sl]].T); Z = (Xw.T - mu) / sd
    v1, v2 = Z[wh].mean(0) - Z[sa].mean(0), Z[ah].mean(0) - Z[sb].mean(0)
    r["PS_cue"] = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
    tc = [dict(session_id=sid, t=float(cc * 1e3), ccgp=ccgp(L[:, :, k], wh, ah, sa, sb)) for k, cc in enumerate(c)]
    return r, tc


def compute(n_proc):
    files = sorted(m105.OUT.glob("*.npz"))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(f,) for f in files])
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    pd.DataFrame([r for r, _ in res]).merge(meta, on="session_id").to_parquet(OUT / "cue_sessions.parquet", index=False)
    pd.DataFrame([x for _, t in res for x in t]).merge(meta, on="session_id").to_parquet(OUT / "cue_timecourse.parquet", index=False)
    print("computed", sum(1 for r, _ in res if "CCGP_cue" in r), "sessions", flush=True)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61; COH = m88.COH
    plt = m62.setup(); rng = np.random.default_rng(0)
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    D = flt(pd.read_parquet(OUT / "cue_sessions.parquet")).dropna(subset=["CCGP_cue"]); T = flt(pd.read_parquet(OUT / "cue_timecourse.parquet"))
    for k in ("CCGP_cue", "within_WH", "within_AH"):
        D[f"{k}_ex"] = D[k] - D[f"{k}_null"]
    fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.3))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.3, 1.3, 1, 1, 1, 1], wspace=0.6, left=0.05, right=0.98, top=0.78, bottom=0.2)
    for j, coh in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[0, j]); ax.set_box_aspect(1)
        for st, ls in (("learning", (0, (2.5, 1.5))), ("expert", "-")):
            piv = T[(T.cohort == coh) & (T.stage == st)].pivot_table(index="session_id", columns="t", values="ccgp") - 0.5
            mu, se = piv.mean(0), piv.sem(0)
            ax.fill_between(piv.columns, mu - se, mu + se, color=COH[coh], alpha=0.15, lw=0)
            ax.plot(piv.columns, mu, color=COH[coh], ls=ls, lw=1.0, label=f"{'day 0' if st == 'learning' else 'expert'} ({piv.shape[0]})")
        ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.4, ls=":"); ax.axvspan(-100, 0, color="#FDD49E", alpha=0.4, lw=0, zorder=0)
        ax.legend(frameon=False, fontsize=4.2, loc="upper left"); ax.tick_params(labelsize=4.6)
        ax.set_title(f"{coh.replace('-', '−')}: cue CCGP over time", color=COH[coh], fontsize=5.4); ax.set_xlabel("from lick (ms)", fontsize=4.8)
        ax.set_ylabel("CCGP − 0.5", fontsize=4.8)
    for j, (col, yl, ttl) in enumerate((("CCGP_cue_ex", "CCGP − chance", "Cue CCGP, −100–0 ms\n(WH/SL ⇆ AH/SL)".replace("⇆", "<->")),
                                         ("PS_cue", "cosine", "Parallelism\n(WH − SL vs AH − SL)"),
                                         ("within_WH_ex", "CV acc − chance", "Within: WH vs SL"), ("within_AH_ex", "CV acc − chance", "Within: AH vs SL"))):
        ax = fig.add_subplot(gs[0, 2 + j]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, D, col, yl, rng, f"109 {col}", ttl, ref=0)
    fig.suptitle(f"Cross-modal generalisation of the cued-lick code: train on one modality vs spontaneous licks, test on the other ({pop})",
                 x=0.02, y=0.995, ha="left", fontsize=6.4, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ccgp_cue_{pop}.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("109")]); St.to_csv(OUT / f"cue_stats_{pop}.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning", "n_R− expert",
              "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
