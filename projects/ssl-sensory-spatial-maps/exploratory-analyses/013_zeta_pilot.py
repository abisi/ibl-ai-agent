"""013 -- ZETA-test pilot on one session, lick trials left out (user 2026-11-18).

Question: does a binning-free test (ZETA, Montijn et al. 2021, zetapy) find sensory responses -- including late-only
ones that the 5-35 ms ROC window misses -- on trials without licks, and where do their response peaks fall?
Trials (no lick in any set):
  W_miss    active whisker trials, lick_flag 0 (001 selection: perf != 6, warm-up cut, rule A1)
  W_passive passive whisker trials (pre + post; 001 resolve_context)
  A_passive passive auditory trials (active auditory trials are almost all hits)
Units: all good + mua units of the session (not only ROC-responsive). Spikes artefact-corrected as in 001/011
(Poisson replacement -10..+5 ms around whisker onsets, seeded per session).
ZETA: window 0..MAX_DUR after onset, 100 jitter resamplings; significant p < 0.01. Peak time = argmax of the zetapy
instantaneous firing rate (IFR) within 5..MAX_DUR ms (zetapy 4.2 returns none); sign from the mean rate
(window vs. pre-stimulus [-MAX_DUR, 0]).
Also: 1-ms PSTH (sigma 2 ms) per unit and set, the 001 ROC responsive flag and t_peak (010 units.parquet) for comparison.
Outputs: ssl-sensory-spatial-maps/zeta_pilot/<session>/{zeta.parquet, psths.npz, zeta_pilot.{png,pdf,svg}}
Run (haas): PYTHONPATH=~/code/NWB_reader:~/zeta_lib ~/code/unit_spikes_analysis/.venv/bin/python 013_zeta_pilot.py \
            [--session MH069_20260122_111455] [--n-proc 60]
"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import pathlib
import time
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("lat", HERE / "001_unit_latency.py")
lat = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lat)

MAX_DUR = 0.2
P_SIG = 0.01
SETS = ("W_miss", "W_passive", "A_passive")
COL = {"W_miss": "#f7b519", "W_passive": "#b37f00", "A_passive": "#2c2cdb"}
EDGES = np.round(np.arange(-0.1, 0.2 + 0.0005, 0.001), 6)
TMS = 1000 * (EDGES[:-1] + 0.0005)

_G = {}


def trial_sets(trials):
    t = trials.sort_values("start_time").reset_index(drop=True)
    ctx = lat.resolve_context(t)
    act = lat.stim_trials(trials)
    pas = t[ctx == "passive"]
    return {"W_miss": act.start_time[(act.trial_type == "whisker_trial") & (act.lick_flag == 0)].to_numpy(),
            "W_passive": pas.start_time[pas.trial_type == "whisker_trial"].to_numpy(),
            "A_passive": pas.start_time[pas.trial_type == "auditory_trial"].to_numpy()}


def one_unit(i):
    import warnings
    from scipy.ndimage import gaussian_filter1d
    from zetapy import zetatest
    st, key = _G["spikes"][i], _G["keys"][i]
    rows, P = [], []
    for s, ev in _G["sets"].items():
        if len(ev) < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = zetatest(st, ev, max_duration=MAX_DUR, resampling_number=100, return_rate=True,
                           restrict_range=(0.005, MAX_DUR), random_seed=zlib.crc32(f"{key}{s}".encode()))
        p, zd, rd = out[:3]               # zetapy returns a 4th value when there are < 3 spikes (p = 1)
        stim = np.array([np.sum((st >= e) & (st < e + MAX_DUR)) for e in ev]) / MAX_DUR
        pre = np.array([np.sum((st >= e - MAX_DUR) & (st < e)) for e in ev]) / MAX_DUR
        cnt = np.zeros(len(EDGES) - 1)
        for e in ev:
            a, b = np.searchsorted(st, [e + EDGES[0], e + EDGES[-1]])
            cnt += np.histogram(st[a:b] - e, EDGES)[0]
        psth = gaussian_filter1d(cnt / len(ev) / 0.001, 2.0)
        pk = np.nan                       # zetapy 4.2 returns no peak latency: argmax of its IFR within 5..MAX_DUR ms
        if isinstance(rd, dict) and rd.get("rate_vector") is not None and len(rd["rate_vector"]):
            tv, rv = np.asarray(rd["timestamps"]), np.asarray(rd["rate_vector"])
            k = (tv >= 0.005) & (tv <= MAX_DUR)
            if k.any():
                pk = tv[k][np.argmax(rv[k])]
        lz = zd.get("latency_zeta")
        rows.append(dict(**key, trial_set=s, n_trials=len(ev), zeta_p=p, zeta_z=zd.get("zeta_score", np.nan),
                         zeta_time_ms=1000 * lz if lz is not None else np.nan,
                         peak_time_ms=1000 * pk,
                         rate_stim=stim.mean(), rate_pre=pre.mean(), sign=np.sign(stim.mean() - pre.mean())))
        P.append(psth.astype(np.float32))
    return rows, P


def compute(sid, n_proc):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from pynwb import NWBHDF5IO
    from roc_analysis import roc_utils_new as ru
    with NWBHDF5IO(str(lat.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
        trials = nwb.trials.to_dataframe()
    units = units[units.quality_label.isin(["good", "mua"])] if "quality_label" in units else units
    units["cluster_id"] = units.cluster_id.astype(str)
    onsets = trials.loc[trials.whisker_stim == 1, "start_time"].to_numpy()
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    _G["sets"] = trial_sets(trials)
    print({k: len(v) for k, v in _G["sets"].items()}, "units", len(units), flush=True)
    _G["spikes"] = [ru.correct_neuron_spike_train(np.asarray(u.spike_times), onsets, rng) for u in units.itertuples()]
    ac = "area_acronym_custom" if "area_acronym_custom" in units else "ccf_acronym"
    _G["keys"] = [dict(mouse_id=sid.split("_")[0], session_id=sid, electrode_group=u.electrode_group,
                       cluster_id=u.cluster_id, area=getattr(u, ac)) for u in units.itertuples()]
    t0, R, P = time.time(), [], []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for k, (r, p) in enumerate(pool.imap(one_unit, range(len(units)), chunksize=4)):
            R += r
            P += p
            if (k + 1) % 100 == 0:
                print(f"{time.time() - t0:6.0f}s {k + 1}/{len(units)}", flush=True)
    return pd.DataFrame(R), np.stack(P), {k: len(v) for k, v in _G["sets"].items()}


def add_reference(R):
    """area_group, the 001 ROC responsive flag and t_peak (all trials, lick or not) from 010 units.parquet"""
    K = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
    U = pd.read_parquet(lat.OUT.parent / "_roc_stage_analysis" / "units.parquet",
                        columns=K + ["area_group"], filters=[("session_id", "==", R.session_id.iloc[0])])
    U["cluster_id"] = U.cluster_id.astype(str)
    R = R.merge(U.drop_duplicates(K), on=K, how="left")
    F = pd.read_parquet(lat.OUT / "fast_slow_peaks" / "units.parquet")
    F = F[F.session_id == R.session_id.iloc[0]]
    F["cluster_id"] = F.cluster_id.astype(str)
    for m, s in (("whisker", "W_"), ("auditory", "A_")):
        f = F[F.modality == m][K + ["t_peak_ms", "z_peak"]].rename(
            columns={"t_peak_ms": f"roc_t_peak_{m}_ms", "z_peak": f"roc_z_peak_{m}"})
        R = R.merge(f, on=K, how="left")
    R["roc_excited_whisker"] = R.roc_t_peak_whisker_ms.notna()
    R["roc_excited_auditory"] = R.roc_t_peak_auditory_ms.notna()
    return R


def plot(R, P, out, sid, ntr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ephys_utilities.allen_utils import allen_utils as au
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    pal = au.get_custom_area_groups_colors()
    order = [g for g in au.get_area_group_custom_order() if g in set(R.area_group.dropna())]
    base = P[:, (TMS >= -100) & (TMS < -5)].mean(1, keepdims=True)
    Y = P - base
    sets = [s for s in SETS if (R.trial_set == s).any()]
    fig, ax = plt.subplots(3, len(sets), figsize=(3.0 * len(sets), 6.6), squeeze=False,
                           gridspec_kw=dict(height_ratios=[1, 1, 1.5], hspace=0.6, wspace=0.95))
    for j, s in enumerate(sets):
        x = R[R.trial_set == s]
        sig = x[(x.zeta_p < P_SIG) & (x.sign > 0)]
        roc = "roc_excited_whisker" if s.startswith("W") else "roc_excited_auditory"
        # row 1: peak-time histogram of ZETA-significant excited units, split by ROC (5-35 ms) responsive or not
        a = ax[0, j]
        b = np.arange(0, 201, 5)
        a.hist([sig.peak_time_ms[sig[roc]], sig.peak_time_ms[~sig[roc]]], b, stacked=True, lw=0,
               color=[COL[s], "0.7"], label=["also ROC-responsive (5-35 ms, all trials)", "ZETA only"])
        a.axvline(20, color="k", lw=0.5, ls="--")
        a.axvline(50, color="0.4", lw=0.5, ls=":")
        a.set_title(f"{s} ({ntr[s]} trials): ZETA p < {P_SIG:g}, excited\n{len(sig)} of {len(x)} units; "
                    f"{(~sig[roc]).sum()} not ROC-responsive", fontsize=6.5)
        a.set_xlabel("IFR peak time (ms)")
        a.set_ylabel("Units per 5-ms bin")
        a.legend(frameon=False, fontsize=5)
        # row 2: fraction responsive per area group
        a = ax[1, j]
        g = x.groupby("area_group").apply(lambda y: pd.Series(dict(
            n=len(y), f=((y.zeta_p < P_SIG) & (y.sign > 0)).mean(),
            late=((y.zeta_p < P_SIG) & (y.sign > 0) & (y.peak_time_ms >= 50)).mean())), include_groups=False)
        g = g.reindex([o for o in order if o in g.index])
        yy = np.arange(len(g))[::-1]
        a.barh(yy, g.f, color=matplotlib.colors.to_rgba(COL[s], 0.4), lw=0, label="excited, any peak time")
        a.barh(yy, g.late, color=COL[s], lw=0, label="excited, peak >= 50 ms")
        a.set_yticks(yy)
        a.set_yticklabels([f"{k} ({int(n)})" for k, n in zip(g.index, g.n)], fontsize=5)
        for lab, k in zip(a.get_yticklabels(), g.index):
            lab.set_color(pal.get(k, "k"))
        a.set_xlabel("Fraction of units (ZETA p < 0.01)")
        a.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
        # row 3: heatmap of peak-normalised PSTHs of ZETA-excited units, sorted by IFR peak time
        a = ax[2, j]
        o = sig.sort_values("peak_time_ms").index.to_numpy()
        Z = Y[o]
        show = TMS >= -50
        pk = Z[:, (TMS >= 5) & (TMS <= 200)].max(1)
        Z = Z / np.where(pk > 0, pk, np.nan)[:, None]
        a.imshow(Z[:, show], aspect="auto", cmap="magma", vmin=-0.2, vmax=1, interpolation="none", rasterized=True,
                 extent=(TMS[show][0], TMS[show][-1], len(Z), 0))
        a.plot(sig.loc[o].peak_time_ms, np.arange(len(o)) + 0.5, ",", color="c")
        a.axvline(0, color="w", lw=0.5, ls="--")
        a.set_xlabel("Time from stimulus onset (ms)")
        a.set_ylabel("ZETA-excited units, sorted by IFR peak")
    fig.suptitle(f"ZETA pilot, {sid}: no-lick trials only (window 0-{1000 * MAX_DUR:.0f} ms, 100 jitters); "
                 "PSTH 1 ms, sigma 2 ms, baseline -100..-5 ms; cyan = ZETA IFR peak", fontsize=7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"zeta_pilot.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="MH069_20260122_111455")
    ap.add_argument("--n-proc", type=int, default=60)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    out = lat.OUT / "zeta_pilot" / a.session
    out.mkdir(parents=True, exist_ok=True)
    if not a.plot_only:
        R, P, ntr = compute(a.session, a.n_proc)
        R = add_reference(R)
        R.to_parquet(out / "zeta.parquet")
        np.savez_compressed(out / "psths.npz", psth=P, t_ms=TMS, n_trials=np.array([ntr[s] for s in SETS]))
    R = pd.read_parquet(out / "zeta.parquet")
    z = np.load(out / "psths.npz")
    ntr = dict(zip(SETS, z["n_trials"]))
    exc = (R.zeta_p < P_SIG) & (R.sign > 0)
    for s, x in R.assign(exc=exc).groupby("trial_set"):
        roc = x["roc_excited_whisker" if s.startswith("W") else "roc_excited_auditory"]
        e = x[x.exc]
        print(f"{s}: n={len(x)} zeta-excited {len(e)} ({len(e) / len(x):.3f}); ROC-excited {int(roc.sum())}, of which "
              f"zeta-sig {int((x.exc & roc).sum())}; zeta-only {int((x.exc & ~roc).sum())}; "
              f"peak<20 {int((e.peak_time_ms < 20).sum())}, 20-50 {int(e.peak_time_ms.between(20, 50).sum())}, "
              f">=50 {int((e.peak_time_ms >= 50).sum())}; inhibited {int(((x.zeta_p < P_SIG) & (x.sign < 0)).sum())}")
    plot(R, z["psth"], out, a.session, ntr)
    print("written", out)


if __name__ == "__main__":
    main()
