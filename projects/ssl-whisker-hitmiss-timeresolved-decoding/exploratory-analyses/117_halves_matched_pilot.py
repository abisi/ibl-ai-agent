"""117 -- PILOT (user 2026-09-30): count-matched session halves for hit vs miss decoding. The chronological halves are
class-unbalanced (median hit rate R+ 0.54 -> 0.32, R- 0.25 -> 0.11 from 1st to 2nd half), so the 2nd-half decoder is
trained on fewer hits and is noisier. Here each half is decoded
  unmatched : all trials of the half (as the 024 "half" condition);
  matched   : K random subsamples; in every subsample each half contributes the SAME number of hits (min over the two
              halves) and the SAME number of misses (min over the two halves); trials drawn without replacement.
Same pipeline as 024 halves: separate decoder per half, whole brain, stimulus-aligned causal 50-ms bins (20-ms stride,
-200..600 ms), dead zone -10..+5 ms, one C per decode (select_fixed_c_pooled on the wide window), pooled stratified CV
(n_folds = min(5, minority)), N_REP repeats; chance = linear-shift null within the (sub)sampled half (N_SHIFT shifts of
10-50% of its trials, pooled scoring). Value = accuracy - mean null. Window 5-100 ms = mean over its bins.
Sessions: a few R+ and R- learning sessions where matching leaves >= MIN_MATCH trials of each class per half.
Outputs: 117_halves_matched_pilot.parquet (session x half x mode [x subsample]), figures/117_halves_matched_pilot.pdf/png
Run (haas, repo root): python .../117_halves_matched_pilot.py [n_per_cohort=4] [K=10]
FULL RUN (user 2026-10-01: "do the count-matched split half full run"): n_per_cohort = 0 -> every eligible learning
session; terminal disengagement dropped with rule A1 (library default) BEFORE the halves are defined (halves recomputed
as the median split of the retained trials); outputs 117_halves_matched_full.parquet / _summary.csv /
figures/117_halves_matched_full.*; N_WORKERS parallel sessions (env SSL_DECODE_N_WORKERS, default 40).
"""

from __future__ import annotations

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
WINDOW, BIN_W, STRIDE = (-0.2, 0.6), 0.05, 0.02
DZ = (-0.010, 0.005)
N_REP, N_SHIFT, MIN_MATCH = 5, 10, 5
WIN = (5, 100)


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def decode_set(T, M, y, rng):
    """Real curve and mean linear-shift null curve on the given trials (matrices M: list of bins x (trials x units))."""
    C = T.select_fixed_c_pooled(T.wide_window_matrix_from_bins(M), y, rng)
    real = T.decode_curve_pooled(M, y, C, rng, n_repeats=N_REP)
    null = T.linear_shift_null_curves_pooled(M, y, C, rng, n_shuf=N_SHIFT, n_repeats=2)
    return real, np.nanmean(null, 0), C


FULL = False
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))


def prep(T, root, sid, st, tt, full):
    """prep_hitmiss_trials; in the full run the A1-disengaged tail is dropped and the halves recomputed."""
    tr = T.prep_hitmiss_trials(root, sid, st, tt)
    if tr is None or not full:
        return tr
    dis = T.detect_terminal_disengagement(sid, tt)
    if dis["disengaged"]:
        tr = tr[tr["start_time"] < dis["t_cut"]].reset_index(drop=True)
        m = len(tr) // 2
        tr["half"] = ["first"] * m + ["second"] * (len(tr) - m)
    return tr


def process(args):
    sid, subject, rg, K, full = args
    sys.path.insert(0, SCRIPTS)
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    tr = prep(T, root, sid, st, tt, full)
    y = (tr.lick_flag == 1).to_numpy()
    half2 = (tr.half == "second").to_numpy()
    units = T.area_units(sid, "whole_brain", "All units", labels)
    spikes = T.load_session_unit_spikes(root, sid)
    edges = T.causal_bin_edges(WINDOW, bin_width=BIN_W, stride=STRIDE)
    M = T.sliding_bin_population_matrices(spikes, units, tr.start_time.to_numpy(), np.ones(len(y), bool), edges, dead_zone=DZ)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    idx = {h: np.where(half2 == (h == "second"))[0] for h in ("first", "second")}
    nh = min(int(y[idx[h]].sum()) for h in idx)
    nm = min(int((~y[idx[h]]).sum()) for h in idx)
    rows = []
    base = dict(session_id=sid, subject_id=subject, reward_group=rg, n_units=len(units), matched_hits=nh, matched_misses=nm)
    for h, ii in idx.items():
        real, null, C = decode_set(T, [m[ii] for m in M], y[ii], rng)
        rows.append(dict(base, half=h, mode="unmatched", subsample=-1, n_hits=int(y[ii].sum()), n_misses=int((~y[ii]).sum()),
                         C=C, real_curve=real.tolist(), null_curve=null.tolist()))
        hi, mi = ii[y[ii]], ii[~y[ii]]
        for k in range(K):
            sel = np.sort(np.r_[rng.choice(hi, nh, replace=False), rng.choice(mi, nm, replace=False)])  # keep time order
            real, null, C = decode_set(T, [m[sel] for m in M], y[sel], rng)
            rows.append(dict(base, half=h, mode="matched", subsample=k, n_hits=nh, n_misses=nm, C=C,
                             real_curve=real.tolist(), null_curve=null.tolist()))
    return rows


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    os.chdir(OUT.parents[2])
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])]
    elig = []
    for r in sess.itertuples():
        x = prep(T, root, r.session_id, st, tt, n_per == 0)
        if x is None:
            continue
        c = x.groupby("half").lick_flag.agg(["sum", "size"])
        if len(c) < 2:
            continue
        nh, nm = int(c["sum"].min()), int((c["size"] - c["sum"]).min())
        if min(nh, nm) >= MIN_MATCH:
            elig.append((r.session_id, r.subject_id, r.reward_group, min(nh, nm)))
    E = pd.DataFrame(elig, columns=["session_id", "subject_id", "reward_group", "min_matched"])
    print(E.groupby("reward_group").size().to_dict(), "eligible sessions (>= %d per class per half after matching)" % MIN_MATCH)
    rng = np.random.default_rng(0)
    if n_per == 0:
        pick = E
    else:
        pick = pd.concat([g.sample(min(n_per, len(g)), random_state=1) for _, g in E.groupby("reward_group")])
    print(pick.to_string(index=False), flush=True)
    rows = []
    global FULL
    FULL = n_per == 0
    with ProcessPoolExecutor(min(len(pick), N_WORKERS), initializer=_init) as ex:
        futs = [ex.submit(process, (r.session_id, r.subject_id, r.reward_group, K, FULL)) for r in pick.itertuples()]
        for f in as_completed(futs):
            rows += f.result()
            print("session done", flush=True)
    d = pd.DataFrame(rows)
    d["disengagement_rule"] = "A1" if FULL else "none"
    d.to_parquet(OUT / f"117_halves_matched_{'full' if FULL else 'pilot'}.parquet", index=False)
    summarize(d)


def summarize(d):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, SCRIPTS)
    import ssl_timeresolved_decoding as T
    t = np.array([e[1] for e in T.causal_bin_edges(WINDOW, bin_width=BIN_W, stride=STRIDE)]) * 1000
    wm = (t >= WIN[0]) & (t <= WIN[1])
    d = d.copy()
    d["dc"] = [np.asarray(r, float) - np.asarray(n, float) for r, n in zip(d.real_curve, d.null_curve)]
    d["w"] = d.dc.map(lambda c: float(np.nanmean(c[wm])))
    g = d.groupby(["session_id", "reward_group", "half", "mode"]).agg(w=("w", "mean"), w_sd=("w", "std"),
                                                                     n_hits=("n_hits", "first"), n_misses=("n_misses", "first"))
    S = g.reset_index().pivot_table(index=["session_id", "reward_group"], columns=["mode", "half"], values=["w", "n_hits", "n_misses"])
    S.columns = [f"{a}_{b}_{c}" for a, b, c in S.columns]
    S["change_unmatched"] = S.w_unmatched_second - S.w_unmatched_first
    S["change_matched"] = S.w_matched_second - S.w_matched_first
    sd = g.reset_index()
    sd = sd[sd["mode"] == "matched"].groupby("session_id").w_sd.mean()
    S["matched_subsample_sd"] = S.index.get_level_values(0).map(sd)
    pd.set_option("display.width", 250)
    cols = ["n_hits_unmatched_first", "n_misses_unmatched_first", "n_hits_unmatched_second", "n_misses_unmatched_second",
            "n_hits_matched_first", "n_misses_matched_first", "w_unmatched_first", "w_unmatched_second", "change_unmatched",
            "w_matched_first", "w_matched_second", "change_matched", "matched_subsample_sd"]
    print(S[cols].round(3).to_string())
    tag = "full" if (d.get("disengagement_rule", pd.Series(["none"])).iloc[0] == "A1") else "pilot"
    S.to_csv(OUT / f"117_halves_matched_{tag}_summary.csv")
    sids = S.index.get_level_values(0).tolist()
    if len(sids) > 16:                    # full run: group-level figure in 117b instead of one panel per session
        return
    fig, axes = plt.subplots(2, len(sids) // 2 + len(sids) % 2, figsize=(1.9 * (len(sids) // 2 + 1), 4.0), squeeze=False)
    COL = {"R+": "#00B400", "R-": "#C800C8"}
    for ax, (sid, rg) in zip(axes.flat, S.index):
        for mode, ls in (("unmatched", "--"), ("matched", "-")):
            x = d[(d.session_id == sid) & (d["mode"] == mode)]
            ch = np.nanmean(np.stack(x[x.half == "second"].dc)) if False else None
            f = np.nanmean(np.stack(x[x.half == "first"].dc.to_numpy()), 0)
            s = np.nanmean(np.stack(x[x.half == "second"].dc.to_numpy()), 0)
            ax.plot(t, s - f, color=COL[rg], ls=ls, lw=0.9, label=f"{mode}")
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5)
        ax.axvspan(*WIN, color="0.92", lw=0, zorder=0)
        ax.set_title(f"{sid[:5]} {rg}", fontsize=6, color=COL[rg])
        ax.tick_params(labelsize=5)
    for ax in axes.flat[len(sids):]:
        ax.set_axis_off()
    axes.flat[0].legend(fontsize=5, frameon=False)
    axes.flat[0].set_ylabel("2nd - 1st half (acc. - null)", fontsize=6)
    fig.suptitle(f"{tag.upper()}: hit vs miss, whole brain, change between session halves -- dashed: all trials per half; solid: "
                 "count-matched halves (mean of subsamples)", fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"117_halves_matched_{tag}.{ext}", dpi=250)


if __name__ == "__main__":
    main()
