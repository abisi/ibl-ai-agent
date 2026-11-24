"""012 -- jaw onsets recomputed on the NWB_ks4 files reconverted 2026-11-20 WITHOUT dlc2kinematics smoothing, written to a
SEPARATE folder and compared with the current per-mouse tables (user 2026-11-20: "compute them and compare, check before
replacing any").

compute: the exact 001 builder (behaviour_analysis plot_behavior -> dlc_helpers.jaw_onset_times, same parameters), with
         the output root redirected to combined_results_ks4/ssl-dlc-jaw-onset/unsmoothed_20261120/<mouse>/; the current
         combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl are NOT touched.
compare: per trial (session_id + start_time), old (current table, smoothed NWB) vs new (unsmoothed): onset found in
         both / old only / new only; onset shift (new - old, ms); per session baseline threshold (old vs new); onset vs
         piezo first-lick latency; split by context (active / passive), trial type and lick. AB116_20240725_094541 and
         AB155_20250208_175103 kept their pre-reconversion NWB (video-link check) -> expected identical (sanity check).
Outputs: <out>/compare_trials.parquet, compare_sessions.csv, compare_summary.csv, compare_figure.png|pdf.
Run (haas, repo root): nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-dlc-jaw-onset/012_unsmoothed_compare.py
                       compute [--n-proc 12] [--mice ...]  |  compare
"""
import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("j001", HERE / "001_behaviour_analysis_jaw_onsets.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)
OLD_ROOT = J.OUT_ROOT
NEW_ROOT = J.OUT_ROOT / "ssl-dlc-jaw-onset" / "unsmoothed_20261120"
J.OUT_ROOT = NEW_ROOT  # redirect the 001 builder; run_mouse reads the module global
HELD = {"AB116_20240725_094541", "AB155_20250208_175103"}


def run_one(mouse_id):
    J.OUT_ROOT = NEW_ROOT  # also in the worker (spawn / forkserver re-import)
    return J.run_mouse(mouse_id)


def compute(args):
    ref = pd.read_excel(J.REFERENCE_SHEET)
    mice = sorted(ref[(ref.with_dual_view == 1) & (ref.exclude == 0)].mouse_id.unique())
    if args.mice:
        mice = [m for m in mice if m in args.mice]
    mice = [m for m in mice if J.session_nwbs(m) and (OLD_ROOT / m / "dlc_jaw_onset_times.pkl").exists()]
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    print(len(mice), "mice", flush=True)
    with Pool(args.n_proc) as pool:
        for m, msg, tb in pool.imap_unordered(run_one, mice, chunksize=1):
            print(m, msg, tb, flush=True)


def compare(args):
    rows = []
    for newp in sorted(NEW_ROOT.glob("*/dlc_jaw_onset_times.pkl")):
        m = newp.parent.name
        oldp = OLD_ROOT / m / "dlc_jaw_onset_times.pkl"
        if not oldp.exists():
            continue
        o, n = pd.read_pickle(oldp), pd.read_pickle(newp)
        for d in (o, n):
            d["key"] = d.session_id.astype(str) + "|" + d.start_time.round(3).astype(str)
        cols = ["key", "session_id", "start_time", "trial_type", "context", "perf", "jaw_dlc_onset", "piezo_lick_time", "threshold"]
        cols = [c for c in cols if c in o.columns and c in n.columns]
        mm = o[cols].merge(n[["key", "jaw_dlc_onset", "threshold"]], on="key", how="outer", suffixes=("_old", "_new"))
        mm["mouse_id"] = m
        rows.append(mm)
    D = pd.concat(rows, ignore_index=True)
    D["context"] = D.context.astype(str)
    D["active"] = ~D.context.str.contains("passive")
    D["has_old"], D["has_new"] = np.isfinite(D.jaw_dlc_onset_old), np.isfinite(D.jaw_dlc_onset_new)
    D["shift_ms"] = 1000 * (D.jaw_dlc_onset_new - D.jaw_dlc_onset_old)
    D["held_session"] = D.session_id.isin(HELD)
    lick_lat = D.piezo_lick_time - D.start_time if "piezo_lick_time" in D and D.piezo_lick_time.max() > 100 else D.get("piezo_lick_time")
    D["lick_latency"] = lick_lat
    D["lick"] = np.isfinite(D.lick_latency) if "lick_latency" in D else False
    D.to_parquet(NEW_ROOT / "compare_trials.parquet")

    def summ(g):
        both = g.has_old & g.has_new
        s = g.shift_ms[both]
        return pd.Series(dict(trials=len(g), onset_old=g.has_old.mean(), onset_new=g.has_new.mean(), both=both.mean(),
                              old_only=(g.has_old & ~g.has_new).mean(), new_only=(~g.has_old & g.has_new).mean(),
                              shift_median_ms=s.median(), shift_iqr_ms=s.quantile(0.75) - s.quantile(0.25),
                              within_5ms=(s.abs() <= 5.01).mean(), beyond_20ms=(s.abs() > 20).mean(),
                              beyond_100ms=(s.abs() > 100).mean(), identical=(s.abs() < 1e-6).mean()))
    S = []
    for name, g in [("all trials", D[~D.held_session]), ("held sessions (old NWB, sanity)", D[D.held_session])] + \
                   [(f"active {tt}{' lick' if lk else ' no lick'}", D[~D.held_session & D.active & (D.trial_type.astype(str) == tt) & (D.lick == lk)])
                    for tt in sorted(D.trial_type.astype(str).unique()) for lk in (True, False)] + \
                   [("passive", D[~D.held_session & ~D.active])]:
        if len(g):
            S.append(summ(g).rename(name))
    S = pd.DataFrame(S)
    S.to_csv(NEW_ROOT / "compare_summary.csv")
    sess = D[~D.held_session].groupby(["mouse_id", "session_id"]).apply(
        lambda g: pd.Series(dict(threshold_old=g.threshold_old.median(), threshold_new=g.threshold_new.median(),
                                 shift_median_ms=g.shift_ms.median(), onset_old=g.has_old.mean(), onset_new=g.has_new.mean(),
                                 beyond_20ms=(g.shift_ms.abs() > 20).mean()))).reset_index()
    sess.to_csv(NEW_ROOT / "compare_sessions.csv", index=False)
    # onset vs lick latency (active lick trials)
    L = D[~D.held_session & D.active & D.lick & D.has_old & D.has_new]
    lat = pd.DataFrame(dict(version=["old", "new"],
                            jaw_before_lick=[np.mean(L.jaw_dlc_onset_old < L.lick_latency), np.mean(L.jaw_dlc_onset_new < L.lick_latency)],
                            median_lick_minus_jaw_ms=[1000 * np.median(L.lick_latency - L.jaw_dlc_onset_old),
                                                      1000 * np.median(L.lick_latency - L.jaw_dlc_onset_new)],
                            corr_with_lick=[np.corrcoef(L.jaw_dlc_onset_old, L.lick_latency)[0, 1],
                                            np.corrcoef(L.jaw_dlc_onset_new, L.lick_latency)[0, 1]]))
    lat.to_csv(NEW_ROOT / "compare_lick_latency.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    print(lat.round(3).to_string())
    print(sess[["threshold_old", "threshold_new", "shift_median_ms", "onset_old", "onset_new", "beyond_20ms"]].describe().round(3).to_string())
    # figure
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6))
    g = D[~D.held_session & D.has_old & D.has_new]
    ax = axs[0, 0]
    ax.hist(g.shift_ms.clip(-100, 100), bins=np.arange(-100, 101, 2.5), color="#555555")
    ax.set_yscale("log")
    ax.set_xlabel("onset shift new − old (ms, clipped ±100)")
    ax.set_ylabel("trials")
    ax = axs[0, 1]
    ax.scatter(g.jaw_dlc_onset_old, g.jaw_dlc_onset_new, s=0.3, alpha=0.2, color="k", rasterized=True)
    ax.plot([-1, 2], [-1, 2], "r-", lw=0.5)
    ax.set_xlabel("old onset (s from trial start)")
    ax.set_ylabel("new onset (s)")
    ax = axs[0, 2]
    ax.scatter(sess.threshold_old, sess.threshold_new, s=4, color="k")
    lim = [0, np.nanmax(sess[["threshold_old", "threshold_new"]].to_numpy()) * 1.05]
    ax.plot(lim, lim, "r-", lw=0.5)
    ax.set_xlabel("baseline threshold old (per session)")
    ax.set_ylabel("threshold new")
    ax = axs[1, 0]
    x = np.arange(len(S))
    ax.bar(x, S.both, color="#999999", label="both")
    ax.bar(x, S.old_only, bottom=S.both, color="#1f77b4", label="old only")
    ax.bar(x, S.new_only, bottom=S.both + S.old_only, color="#d62728", label="new only")
    ax.set_xticks(x)
    ax.set_xticklabels(S.index, rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("fraction of trials with an onset")
    ax.legend(frameon=False, fontsize=5)
    ax = axs[1, 1]
    for v, col in (("old", "#1f77b4"), ("new", "#d62728")):
        ax.hist(1000 * (L.lick_latency - L[f"jaw_dlc_onset_{v}"]).clip(-200, 400), bins=np.arange(-200, 401, 10), histtype="step",
                color=col, label=v)
    ax.set_xlabel("first lick − jaw onset (ms), active lick trials")
    ax.legend(frameon=False, fontsize=5)
    ax = axs[1, 2]
    ax.hist(sess.shift_median_ms.clip(-30, 30), bins=np.arange(-30, 31, 1), color="#555555")
    ax.set_xlabel("median shift per session (ms)")
    ax.set_ylabel("sessions")
    for a in axs.ravel():
        a.tick_params(labelsize=5)
    fig.suptitle("Jaw onsets: unsmoothed NWB (new) vs current tables (smoothed NWB, old)", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(NEW_ROOT / f"compare_figure.{ext}", dpi=250)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["compute", "compare"])
    p.add_argument("--n-proc", type=int, default=12)
    p.add_argument("--mice", nargs="*")
    a = p.parse_args()
    {"compute": compute, "compare": compare}[a.step](a)
