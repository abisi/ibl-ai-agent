"""122 -- Reaction-time controls for the 104 ROC similarity index (user 2026-10-14). Whisker-hit RTs shorten to auditory-like values in
R+ only (090), so the [-100, 0) ms pre-lick window of expert R+ whisker hits sits at the same post-stimulus latency as that of auditory
hits: stimulus-locked, modality-unspecific activity alone would make WH look AH-like. The sensory exclusion (5-50 ms stimulus response)
does not remove this.
Neurons and measure as 104 (095 reward-lick neurons, SL-B reference, cross-validated |s|, SI = (|s_WS| - |s_WA|) / (|s_WS| + |s_WA|));
summaries restricted to non-sensory neurons (081 class not "sensory"), >= 3 per session, learners rule (as 119 h / 121).
Variants (spontaneous licks never subsampled; >= 4 WH and >= 4 AH after the adjustment):
  all          no adjustment (= 104)
  long150/200  WH and AH with RT >= 150 / 200 ms (the 5-50 ms stimulus response falls before the window)
  match_WA     WH and AH matched on RT within the session (20-ms bins, per bin min(n_WH, n_AH) of each; N_DRAW draws)
  match_stage  WH resampled to the pooled expert-WH RT distribution of the same cohort (20-ms bins, acceptance probability
               p_target / p_session scaled to max 1; N_DRAW draws), AH all: day 0 vs expert at equal whisker-hit RT
  rtbin        WH split into fixed RT bins (RT_BINS), AH all; SI per bin (>= 4 WH in the bin) -> SI as a function of WH RT
Per neuron |s| averaged over draws, SI from the averaged |s|. Session measures: mean SI, fraction SI > 0, mean |s_WS|, mean |s_WA|.
Statistics: 062 group_stats (MWU, Welch, mouse-level cohort x stage permutation), unit = session.
Output: across_days/<ref>/similarity_index/rt_control/{units.parquet, sessions.csv, stats.csv, si_rt_control.{png,pdf,svg},
        si_rt_bins.{png,pdf,svg}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../122_si_rt_control.py [--n-proc 30] [--replot] [--pop learners|all]
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
m104 = importlib.import_module("104_similarity_index")
m51, m61, m62, m80, m81, m96 = m104.m51, m104.m61, m104.m62, m104.m80, m104.m81, m104.m96
OUT = m104.OUT / "rt_control"
BIN, N_DRAW, N_SPLIT, MIN_EV = 0.020, 10, 10, 4
RT_BINS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 2.0)]
VARIANTS = ["all", "long150", "long200", "match_WA", "match_stage"]
VLAB = {"all": "All events", "long150": "RT ≥ 150 ms", "long200": "RT ≥ 200 ms", "match_WA": "WH / AH RT-matched",
        "match_stage": "WH RT matched\nto expert"}
COH = {"R+": "#00B400", "R-": "#C800C8"}


def cv_abs(X, ia, ib, rng, n_split=N_SPLIT):
    acc = np.zeros(len(X))
    for _ in range(n_split):
        pa, pb = rng.permutation(ia), rng.permutation(ib)
        a1, a2, b1, b2 = pa[: len(pa) // 2], pa[len(pa) // 2:], pb[: len(pb) // 2], pb[len(pb) // 2:]
        acc += m80.auc_sel(X, a1, b1) * m80.auc_sel(X, a2, b2)
    return np.sqrt(np.clip(acc / n_split, 0, None))


def match_pair(iw, ia, rt, rng):
    bw, ba = np.floor(rt[iw] / BIN), np.floor(rt[ia] / BIN)
    sw, sa = [], []
    for b in np.intersect1d(bw, ba):
        w, a = iw[bw == b], ia[ba == b]; k = min(len(w), len(a))
        sw.append(rng.choice(w, k, replace=False)); sa.append(rng.choice(a, k, replace=False))
    return (np.concatenate(sw), np.concatenate(sa)) if sw else (iw[:0], ia[:0])


def match_target(iw, rt, target, rng):
    """accept WH events with probability p_target(bin) / p_session(bin), scaled so that the max is 1"""
    b = np.floor(rt[iw] / BIN).astype(int)
    ub, cnt = np.unique(b, return_counts=True)
    ps = dict(zip(ub, cnt / cnt.sum()))
    ratio = np.array([target.get(x, 0.0) / ps[x] for x in b])
    if ratio.max() <= 0:
        return iw[:0]
    return iw[rng.random(len(iw)) < ratio / ratio.max()]


def measure(Xs, sets, slB, rng):
    """sets: list of (iw, ia) draws -> per-unit mean |s_WS|, |s_WA| over draws"""
    sws, swa = np.zeros(len(Xs)), np.zeros(len(Xs))
    for iw, ia in sets:
        sws += cv_abs(Xs, iw, slB, rng); swa += cv_abs(Xs, iw, ia, rng)
    return sws / len(sets), swa / len(sets)


def session(args):
    sid, f, units, target = args
    z = np.load(f, allow_pickle=True)
    lab = z["cls"]; X = z["rates"].astype(float); rt = z["rt"].astype(float)
    iw, ia, isl = (np.where(lab == c)[0] for c in ("WH", "AH", "FA"))
    iw, ia = iw[np.isfinite(rt[iw])], ia[np.isfinite(rt[ia])]
    if min(len(iw), len(ia)) < MIN_EV or len(isl) < 16:
        return []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|095".encode()))
    p = rng.permutation(isl); slB = np.sort(p[len(p) // 2:])                  # same split as 095 / 104
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    if K.empty:
        return []
    Xs = X[K.row.to_numpy()]
    r2 = np.random.default_rng(zlib.crc32(f"{sid}|122".encode()))
    out = []

    def add(variant, sets, rtbin="", n_wh=np.nan):
        sets = [(w, a) for w, a in sets if min(len(w), len(a)) >= MIN_EV]
        if not sets:
            return
        sws, swa = measure(Xs, sets, slB, r2)
        den = sws + swa
        si = np.where(den > 0, (sws - swa) / np.where(den > 0, den, 1), np.nan)
        nw = np.mean([len(w) for w, _ in sets])
        out.extend(dict(session_id=sid, electrode_group=e, cluster_id=c, variant=variant, rtbin=rtbin, n_wh=nw, abs_sWS=a, abs_sWA=b, SI=s)
                   for e, c, a, b, s in zip(K.electrode_group, K.cluster_id, sws, swa, si))

    add("all", [(iw, ia)])
    for thr, v in [(0.150, "long150"), (0.200, "long200")]:
        add(v, [(iw[rt[iw] >= thr], ia[rt[ia] >= thr])])
    add("match_WA", [match_pair(iw, ia, rt, r2) for _ in range(N_DRAW)])
    if target:
        add("match_stage", [(match_target(iw, rt, target, r2), ia) for _ in range(N_DRAW)])
    for lo, hi in RT_BINS:
        add("rtbin", [(iw[(rt[iw] >= lo) & (rt[iw] < hi)], ia)], rtbin=f"{int(lo * 1e3)}-{int(hi * 1e3)}")
    return out


def compute(n_proc):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(m96.SPLIT / "selectivity_index.parquet")
    Un = D.drop_duplicates(["session_id", "electrode_group", "cluster_id"])[["session_id", "electrode_group", "cluster_id", "mouse_id", "cohort", "stage", "nclass"]]
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(Un.session_id.unique())]
    files = {r.session_id: r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz" for r in ss.itertuples()}
    files = {k: v for k, v in files.items() if v.exists()}
    meta = Un.drop_duplicates("session_id").set_index("session_id")
    # pooled expert whisker-hit RT distribution per cohort (20-ms bins)
    rts = {"R+": [], "R-": []}
    for sid, f in files.items():
        if meta.loc[sid, "stage"] == "expert" and meta.loc[sid, "cohort"] in rts:
            z = np.load(f, allow_pickle=True); r = z["rt"].astype(float)[z["cls"] == "WH"]
            rts[meta.loc[sid, "cohort"]].extend(r[np.isfinite(r)])
    target = {}
    for c, r in rts.items():
        ub, cnt = np.unique(np.floor(np.array(r) / BIN).astype(int), return_counts=True)
        target[c] = dict(zip(ub, cnt / cnt.sum()))
        print(c, "expert WH RT target: n", len(r), "median", round(float(np.median(r)), 3), flush=True)
    jobs = [(sid, f, Un[Un.session_id == sid][["electrode_group", "cluster_id"]], target.get(meta.loc[sid, "cohort"], {})) for sid, f in files.items()]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    U = pd.DataFrame(res).merge(Un, on=["session_id", "electrode_group", "cluster_id"], how="left")
    U.to_parquet(OUT / "units.parquet", index=False)
    print("computed", len(U), "rows", U.session_id.nunique(), "sessions", flush=True)


def sessions(pop):
    U = pd.read_parquet(OUT / "units.parquet")
    if pop == "learners":
        U = m61.learner_filter(U)
    U = U[U.nclass.isin(m81.CLASSES) & (U.nclass != "sensory") & U.SI.notna()]
    S = U.groupby(["variant", "rtbin", "session_id", "mouse_id", "cohort", "stage"]).agg(
        SI=("SI", "mean"), frac_pos=("SI", lambda x: (x > 0).mean()), abs_sWS=("abs_sWS", "mean"), abs_sWA=("abs_sWA", "mean"),
        n_wh=("n_wh", "first"), n=("SI", "size")).reset_index()
    return S[S.n >= m81.MIN_CLASS]


def figure(pop):
    plt = m62.setup()
    S = sessions(pop); S.to_csv(OUT / f"sessions_{pop}.csv", index=False)
    rng = np.random.default_rng(1)
    meas = [("SI", "Session mean SI", 0), ("frac_pos", "Fraction SI > 0", 0.5), ("abs_sWS", "|s_WS| (WH vs SL)", None),
            ("abs_sWA", "|s_WA| (WH vs AH)", None)]
    fig, axs = plt.subplots(len(meas), len(VARIANTS), figsize=(7.0, 6.4), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.05, hspace=0.75, wspace=0.55)
    m62.STATS.clear()
    for j, v in enumerate(VARIANTS):
        d = S[(S.variant == v)]
        for i, (col, lab, ref) in enumerate(meas):
            ax = axs[i, j]
            m62.dots_panel(ax, d, col, lab if j == 0 else "", rng, f"122 {v} {col}", f"{VLAB[v]}" if i == 0 else "",
                           ref=[(ref, "0.6")] if ref is not None else None)
    fig.suptitle("ROC similarity index, non-sensory reward-lick neurons: reaction-time controls (○ day 0, ● expert; unit = session)",
                 fontsize=6.5, y=0.985)
    St = pd.DataFrame(m62.STATS); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"si_rt_control_{pop}.{ext}", dpi=300)
    plt.close(fig)
    # SI as a function of whisker-hit RT
    B = S[S.variant == "rtbin"]
    order = [f"{int(lo * 1e3)}-{int(hi * 1e3)}" for lo, hi in RT_BINS]
    fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.33, wspace=0.45)
    for ax, (col, lab, ref) in zip(axs, meas[:1] + meas[2:]):
        for (c, st), g in B.groupby(["cohort", "stage"]):
            m = g.groupby("rtbin")[col].agg(["mean", "sem", "size"]).reindex(order)
            x = np.arange(len(order)) + (0.06 if st == "expert" else -0.06)
            ok = m["size"] >= 3
            ax.errorbar(x[ok], m["mean"][ok], m["sem"][ok], color=COH[c], ls="-" if st == "expert" else (0, (2, 1.5)), marker="o", ms=2.8,
                        mfc=COH[c] if st == "expert" else "white", lw=0.9, label=f"{c} {'day 0' if st == 'learning' else st}")
        if ref is not None:
            ax.axhline(ref, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("-2000", "+") for o in order], rotation=30)
        ax.set_xlabel("Whisker-hit RT (ms)"); ax.set_ylabel(lab)
    axs[0].legend(frameon=False, fontsize=4.8, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=4)
    fig.text(0.99, 0.005, "AH: all events; session means where >= 4 WH in the bin; points with >= 3 sessions", ha="right", fontsize=4.6, color="0.4")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"si_rt_bins_{pop}.{ext}", dpi=300)
    plt.close(fig)
    cols = ["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert",
            "n_R− learning", "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R+ L vs E p_Welch", "expert R+ vs R- p_MWU",
            "expert R+ vs R- p_Welch"]
    print(St[[c for c in cols if c in St]].round(4).to_string())
    print(B.groupby(["cohort", "stage", "rtbin"]).agg(SI=("SI", "mean"), n=("session_id", "nunique")).round(3).to_string())


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--n-proc", type=int, default=30); p.add_argument("--replot", action="store_true")
    p.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(p.parse_args())
