"""117 -- Alignment of the stimulus-onset hit - miss axis and the pre-lick reward-lick CD in the same neurons (user 2026-10-12: "run the
axis alignment"). Exploratory.
Units and events: 105 (stable tested units; active trials of 051: perf != 6, warm-up cut, rule A1).
  u_HM  hit - miss axis: active whisker trials, rate in [5, 35) ms after stimulus onset minus [-55, -20) ms (as 135), trials with a lick
        before 35 ms dropped; units z-scored over these trials; u_HM = unit(mean hit - mean miss)
  u_CD  reward-lick CD: AH and SL events (105 lickbins, mean of the bins centred in [-100, 0) ms before the lick, baseline-corrected);
        units z-scored over AH + SL; u_CD = unit(mean AH - mean SL)
  The two axes come from disjoint events (whisker trials at stimulus onset vs auditory hits / spontaneous licks before the lick).
Quantities per session x unit group (whole brain; area groups with >= MIN_UNITS units):
  cos       cos(u_HM, u_CD) (sign meaningful: > 0 = the hit pattern at stimulus onset resembles the pre-lick reward-lick pattern)
  rel_HM, rel_CD  split-half reliability of each axis (cos of the axes from random halves of each class, mean over N_SPLIT, floor 0.05)
  cos_norm  cos / sqrt(rel_HM rel_CD)  (noise-corrected, as 135)
  null      the same with both axes rebuilt from linear-shifted labels (time-ordered events, |k| in [10 %, 50 %] of n, random sign, no
            wrap; N_SHIFT shifts; mean); reported as real - null
  dp_HM_on_CD  functional: d' between hits and misses (stimulus window, own z-scoring) along u_CD; real - mean over shifted u_CD
  dp_CD_on_HM  d' between AH and SL (pre-lick window, own z-scoring) along u_HM; real - mean over shifted u_HM
Sessions need >= MIN_CLASS hits and misses and >= 2 MIN_CLASS AH and SL. Learners rule; session = unit; 088 panels (day 0 vs expert per
cohort, R+ vs R- per stage, cohort x stage by mouse permutation). Random unit vectors: |cos| ~ 1 / sqrt(N) (N printed).
Output: across_days/<ref>/axis_alignment/{alignment.parquet, stats_<pop>.csv, axis_alignment_<pop>.{png,pdf,svg}}
Run (haas): OMP_NUM_THREADS=1 PRELICK_REF=sl PRELICK_UNITS=stable python .../117_axis_alignment.py [--n-proc 25] [--replot]
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
m105 = importlib.import_module("105_trial_rates")
m51 = m105.m51
OUT = m51.OUTROOT / "axis_alignment"
WIN, BASE = (0.005, 0.035), (-0.055, -0.020)
MIN_CLASS, MIN_UNITS, N_SPLIT, N_SHIFT = 4, 10, 50, 20


def zaxis(X, pos, neg):
    """X units x events; z-score over pos + neg events; returns unit axis, mean, sd"""
    idx = np.r_[pos, neg]; mu = X[:, idx].mean(1); sd = X[:, idx].std(1); sd[sd == 0] = 1
    Z = (X - mu[:, None]) / sd[:, None]; v = Z[:, pos].mean(1) - Z[:, neg].mean(1)
    return v / max(np.linalg.norm(v), 1e-12), mu, sd


def cosv(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def rel(X, pos, neg, rng):
    out = []
    for _ in range(N_SPLIT):
        p, n = rng.permutation(pos), rng.permutation(neg)
        a = zaxis(X, p[: len(p) // 2], n[: len(n) // 2])[0]; b = zaxis(X, p[len(p) // 2:], n[len(n) // 2:])[0]
        out.append(cosv(a, b))
    return max(float(np.mean(out)), 0.05)


def dprime(X, pos, neg, u, mu, sd):
    Z = (X - mu[:, None]) / sd[:, None]; p, n = u @ Z[:, pos], u @ Z[:, neg]
    return float((p.mean() - n.mean()) / np.sqrt(0.5 * (p.var() + n.var()) + 1e-12))


def shift_labels(lab_by_time, rng):
    """lab_by_time: labels (bool / str) in time order -> (shifted labels for the kept events, kept event indices)"""
    n = len(lab_by_time); lo, hi = max(1, int(0.1 * n)), max(2, int(0.5 * n))
    k = int(rng.integers(lo, hi)) * (1 if rng.random() < 0.5 else -1)
    idx = np.arange(n); src = idx + k; ok = (src >= 0) & (src < n)
    return lab_by_time[src[ok]], idx[ok]          # event idx[ok] takes the label of event idx[ok] + k


def session(args):
    sid, = args
    f = m105.OUT / f"{sid}.npz"
    if not f.exists():
        return []
    z = np.load(f, allow_pickle=True)
    eg, cid, ag = z["electrode_group"].astype(str), z["cluster_id"].astype(str), z["area_group"].astype(str)
    sl, st, rt, hit = z["stim_lab"].astype(str), z["stim_t"].astype(float), z["stim_rt"].astype(float), z["is_hit"].astype(bool)
    w = np.where((sl == "W") & (rt >= WIN[1]))[0]; w = w[np.argsort(st[w])]
    h = hit[w]
    if min(h.sum(), (~h).sum()) < MIN_CLASS:
        return []
    ll, lt = z["lick_lab"].astype(str), z["lick_t"].astype(float); c = z["lick_c"]
    Xcd_all = z["lickbins"].astype(float)[:, :, (c >= -0.1) & (c < 0.0)].mean(2)
    e = np.where(np.isin(ll, ["AH", "SL"]))[0]; e = e[np.argsort(lt[e])]
    lab_cd = ll[e]
    if min((lab_cd == "AH").sum(), (lab_cd == "SL").sum()) < 2 * MIN_CLASS:
        return []
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        U, _ = m51.ru.process_nwb_tables(io.read())
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str)).set_index(["electrode_group", "cluster_id"])
    spk = [np.sort(np.asarray(U.loc[(a, b)].spike_times)) for a, b in zip(eg, cid)]
    s0 = st[w]
    cnt = lambda s, lo, hi: np.searchsorted(s, s0 + hi) - np.searchsorted(s, s0 + lo)
    Xhm_all = np.array([cnt(s, *WIN) / (WIN[1] - WIN[0]) - cnt(s, *BASE) / (BASE[1] - BASE[0]) for s in spk], float)
    Xcd_all = Xcd_all[:, e]
    rng = np.random.default_rng(zlib.crc32(f"{sid}|117".encode()))
    shifts_hm = [shift_labels(h, rng) for _ in range(N_SHIFT)]; shifts_cd = [shift_labels(lab_cd, rng) for _ in range(N_SHIFT)]
    rows = []
    groups = [("Whole brain", np.ones(len(ag), bool))] + [(g, ag == g) for g in sorted(set(ag)) if g not in ("nan", "None")]
    for g, m in groups:
        if m.sum() < MIN_UNITS:
            continue
        Xh, Xc = Xhm_all[m], Xcd_all[m]
        keep = (Xh.std(1) > 0) & (Xc.std(1) > 0); Xh, Xc = Xh[keep], Xc[keep]
        if len(Xh) < MIN_UNITS:
            continue
        hp, hn = np.where(h)[0], np.where(~h)[0]; cp, cn = np.where(lab_cd == "AH")[0], np.where(lab_cd == "SL")[0]
        uh, muh, sdh = zaxis(Xh, hp, hn); uc, muc, sdc = zaxis(Xc, cp, cn)
        rh, rc = rel(Xh, hp, hn, rng), rel(Xc, cp, cn, rng)
        cs = cosv(uh, uc)
        dp_h = dprime(Xh, hp, hn, uc, muh, sdh); dp_c = dprime(Xc, cp, cn, uh, muc, sdc)
        nc, nd_h, nd_c = [], [], []
        for (lh, ih), (lc, ic) in zip(shifts_hm, shifts_cd):
            if min(lh.sum(), (~lh).sum()) < 2 or min((lc == "AH").sum(), (lc == "SL").sum()) < 2:
                continue
            us = zaxis(Xh[:, ih], np.where(lh)[0], np.where(~lh)[0])[0]; vs = zaxis(Xc[:, ic], np.where(lc == "AH")[0], np.where(lc == "SL")[0])[0]
            nc.append(cosv(us, vs)); nd_h.append(dprime(Xh, hp, hn, vs, muh, sdh)); nd_c.append(dprime(Xc, cp, cn, us, muc, sdc))
        nul = np.nanmean(nc) if nc else np.nan
        rows.append(dict(session_id=sid, area=g, n_units=len(Xh), n_hit=int(h.sum()), n_miss=int((~h).sum()), n_AH=len(cp), n_SL=len(cn),
                         cos=cs, cos_null=nul, cos_ex=cs - nul, rel_HM=rh, rel_CD=rc, cos_norm=cs / np.sqrt(rh * rc),
                         cos_norm_ex=(cs - nul) / np.sqrt(rh * rc), dp_HM_on_CD=dp_h, dp_HM_on_CD_ex=dp_h - np.nanmean(nd_h) if nd_h else np.nan,
                         dp_CD_on_HM=dp_c, dp_CD_on_HM_ex=dp_c - np.nanmean(nd_c) if nd_c else np.nan, n_shift=len(nc)))
    return rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, [(s,) for s in meta.session_id]) for x in r]
    D = pd.DataFrame(res).merge(meta, on="session_id"); D.to_parquet(OUT / "alignment.parquet", index=False)
    print("computed", len(D), "rows;", D[D.area == "Whole brain"].session_id.nunique(), "sessions", flush=True)


def figure(pop):
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62, m61 = m88.m62, m88.m61
    plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(OUT / "alignment.parquet"); D = m61.learner_filter(D) if pop == "learners" else D
    W = D[D.area == "Whole brain"]
    with plt.rc_context({"font.size": 5.4, "axes.labelsize": 5.4, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(7.2, 2.1))
        gs = fig.add_gridspec(1, 4, wspace=0.7, left=0.07, right=0.99, top=0.72, bottom=0.2)
        L = []
        for j, (col, yl, tt) in enumerate((("cos_ex", "cos − shift null", "cos(hit − miss axis, CD)"),
                                           ("cos_norm_ex", "noise-corrected cos − null", "Noise-corrected alignment"),
                                           ("dp_HM_on_CD_ex", "d′ − shift null", "Hits vs misses (5–35 ms)\non the reward-lick CD"),
                                           ("dp_CD_on_HM_ex", "d′ − shift null", "AH vs SL (pre-lick)\non the hit − miss axis"))):
            ax = fig.add_subplot(gs[0, j]); ax.set_box_aspect(1); L.append(ax)
            m88.mean_panel2(ax, W, col, yl, rng, f"117 {col}", tt, ref=0)
        m62.letter_row(fig, L, "abcd", dx_in=0.26, dy_in=0.14)
        fig.suptitle(f"Stimulus-onset hit − miss axis vs pre-lick reward-lick CD, same neurons, whole brain ({pop}; "
                     f"median N = {int(W.n_units.median())} units, 1/√N = {1 / np.sqrt(W.n_units.median()):.2f})", x=0.01, y=0.995, ha="left",
                     fontsize=6.2, weight="bold")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"axis_alignment_{pop}.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("117")]); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning",
              "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    from scipy import stats
    for col in ("cos", "cos_ex", "cos_norm_ex", "dp_HM_on_CD_ex"):
        for c, s in m88.GROUPS:
            v = W[(W.cohort == c) & (W.stage == s)][col].dropna()
            print(f"{col:15s} {c} {s:8s} mean {v.mean():7.3f}  vs 0 Wilcoxon p = {stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan:.3g} (n = {len(v)})")
    print(W[["rel_HM", "rel_CD"]].describe().round(3))


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
