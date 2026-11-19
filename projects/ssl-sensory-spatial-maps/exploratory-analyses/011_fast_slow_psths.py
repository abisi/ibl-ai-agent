"""011 -- PSTHs of fast (peak < 20 ms) and slow (peak 20-95 ms) excited sensory responses per area group (user 2026-11-18).

Classes and units: 010 (fast_slow_peaks/units.parquet), excited, peak z >= 5, edge class (peak >= 95 ms) left out.
PSTH: recomputed exactly as in 001 (artefact-corrected KS4 spikes seeded per session, same trial selection, 1-ms bins,
-100..+200 ms), Gaussian sigma 2 ms. Baseline: the pre-trial rate [-1, -0.015] s (ssl-figure-style rule 2); the 001
local baseline (-100..-10 ms) is stored too. Stored per unit: raw smoothed PSTH (Hz), pre-trial and local baselines.
Averaging: mean over units of a class within each session, then mean +- s.e.m. over sessions (sessions with >= 1 unit).
Outputs: ssl-sensory-spatial-maps/fast_slow_peaks/{psths.npz, psth_units.parquet,
         psths_<modality>_{hz,norm}.{png,pdf,svg}}  (norm = each unit divided by its peak, baseline-subtracted)
Run (haas): PYTHONPATH=~/code/NWB_reader ~/code/unit_spikes_analysis/.venv/bin/python 011_fast_slow_psths.py [--n-proc 40] [--plot-only]
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

OUT = lat.OUT / "fast_slow_peaks"
KEYS = lat.KEYS
Z_MIN = 5.0
COL = {"whisker": "#f7b519", "auditory": "#2c2cdb"}
CLASSES = {"fast": "peak < 20 ms", "slow": "peak 20-95 ms"}
MIN_SESS = 3


def wanted():
    u = pd.read_parquet(OUT / "units.parquet")
    u = u[(u.z_peak >= Z_MIN) & u.cls.isin(list(CLASSES))]
    u["cluster_id"] = u.cluster_id.astype(str)
    return u


def run_session(args):
    sid, want = args            # want: DataFrame rows (electrode_group, cluster_id, modality) for this session
    from pynwb import NWBHDF5IO
    from scipy.ndimage import gaussian_filter1d
    from roc_analysis import roc_utils_new as ru
    with NWBHDF5IO(str(lat.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
        trials = nwb.trials.to_dataframe()
    units["cluster_id"] = units["cluster_id"].astype(str)
    onsets = trials.loc[trials["whisker_stim"] == 1, "start_time"].to_numpy()
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    t = lat.stim_trials(trials)
    edges = np.round(np.arange(lat.T0, lat.T1 + lat.BIN / 2, lat.BIN), 6)
    tms = 1000 * (edges[:-1] + lat.BIN / 2)
    keyset = set(zip(want.electrode_group, want.cluster_id))
    rows, P = [], []
    for u in units.itertuples():
        if (u.electrode_group, u.cluster_id) not in keyset:
            continue
        st = ru.correct_neuron_spike_train(np.asarray(u.spike_times), onsets, rng)
        for m in want[(want.electrode_group == u.electrode_group) & (want.cluster_id == u.cluster_id)].modality:
            s0 = t.start_time[t.trial_type == lat.MEASURES[m][1]].to_numpy()
            if len(s0) < 3:
                continue
            base = ((np.searchsorted(st, s0 + lat.BASE[1]) - np.searchsorted(st, s0 + lat.BASE[0]))
                    / (lat.BASE[1] - lat.BASE[0])).mean()
            cnt = np.zeros(len(edges) - 1)
            for s in s0:
                a, b = np.searchsorted(st, [s + lat.T0, s + lat.T1])
                cnt += np.histogram(st[a:b] - s, edges)[0]
            psth = gaussian_filter1d(cnt / len(s0) / lat.BIN, lat.SIGMA_MS)
            local = psth[(tms >= lat.LOCAL_BASE[0]) & (tms < lat.LOCAL_BASE[1])].mean()
            rows.append(dict(session_id=sid, electrode_group=u.electrode_group, cluster_id=u.cluster_id, modality=m,
                             base_pretrial_hz=base, base_local_hz=local))
            P.append(psth.astype(np.float32))
    return pd.DataFrame(rows), (np.stack(P) if P else np.zeros((0, len(tms)), np.float32))


def compute(n_proc):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    w = wanted()
    jobs = [(s, g[["electrode_group", "cluster_id", "modality"]]) for s, g in w.groupby("session_id")]
    t0, R, P = time.time(), [], []
    with mp.get_context("fork").Pool(n_proc) as pool:
        for k, (r, p) in enumerate(pool.imap_unordered(run_session, jobs)):
            R.append(r)
            P.append(p)
            print(f"{time.time() - t0:6.0f}s {k + 1}/{len(jobs)}", flush=True)
    R = pd.concat(R, ignore_index=True)
    P = np.concatenate(P)
    R["row"] = np.arange(len(R))
    R = R.merge(w[KEYS + ["modality", "area_group", "cls", "t_peak_ms", "latency_ms", "z_peak"]],
                on=["session_id", "electrode_group", "cluster_id", "modality"], how="left")
    tms = 1000 * (np.round(np.arange(lat.T0, lat.T1 + lat.BIN / 2, lat.BIN), 6)[:-1] + lat.BIN / 2)
    np.savez_compressed(OUT / "psths.npz", psth=P, t_ms=tms)
    R.to_parquet(OUT / "psth_units.parquet")
    print("units with PSTH", len(R), "of", len(w))


def session_mean(Y, R):
    """mean over units per session, then mean and s.e.m. over sessions"""
    S = np.stack([np.nanmean(Y[g.index.to_numpy()], 0) for _, g in R.groupby("session_id")])
    S = S[np.isfinite(S).all(1)]
    se = np.std(S, 0, ddof=1) / np.sqrt(len(S)) if len(S) > 1 else np.zeros(S.shape[1])
    return S.mean(0), se, len(S)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ephys_utilities.allen_utils import allen_utils as au
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    z = np.load(OUT / "psths.npz")
    P, tms = z["psth"], z["t_ms"]
    R = pd.read_parquet(OUT / "psth_units.parquet").sort_values("row").reset_index(drop=True)
    Y = P - R.base_pretrial_hz.to_numpy()[:, None]
    pk = np.array([y[(tms >= 5) & (tms <= 100)].max() for y in Y])
    ok = pk > 0                     # a few units: baseline-subtracted peak <= 0 against the pre-trial baseline
    Yn = Y / np.where(ok, pk, np.nan)[:, None]
    print("units with peak <= 0 vs pre-trial baseline (left out of norm):", int((~ok).sum()))
    pal = au.get_custom_area_groups_colors()
    order = au.get_area_group_custom_order()
    show = (tms >= -50) & (tms <= 200)
    for m in ("whisker", "auditory"):
        Rm = R[R.modality == m]
        ns = Rm.groupby(["area_group", "cls"]).session_id.nunique().unstack()
        groups = [g for g in order if g in ns.index and (ns.loc[g].reindex(list(CLASSES)).fillna(0) >= MIN_SESS).all()]
        for kind, Z, ylab in (("hz", Y, "Rate − pre-trial baseline (Hz)"), ("norm", Yn, "Rate / unit's peak (5-100 ms)")):
            nc = 4
            nr = int(np.ceil((len(groups) + 1) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(7.4, 1.55 * nr), squeeze=False)
            for i, g in enumerate(["All area groups"] + groups):
                a = axs.flat[i]
                sub = Rm if i == 0 else Rm[Rm.area_group == g]
                txt = []
                for c in ("fast", "slow"):
                    s = sub[sub.cls == c]
                    if s.session_id.nunique() < 2:
                        continue
                    mu, se, n = session_mean(Z, s)
                    col = COL[m] if c == "fast" else "0.35"
                    a.fill_between(tms[show], (mu - se)[show], (mu + se)[show], color=col, alpha=0.3, lw=0,
                                   edgecolor="none")
                    a.plot(tms[show], mu[show], color=col, lw=0.8, label=f"{c} ({CLASSES[c]})")
                    txt.append(f"{c} {len(s)} u / {n} s")
                a.axvline(0, color="0.3", lw=0.5, ls="--")
                a.axvspan(-10, 5, color="0.9", lw=0, edgecolor="none") if m == "whisker" else None
                a.axhline(0, color="0.6", lw=0.4, ls=":")
                a.set_title(g, fontsize=6.5, color="k" if i == 0 else pal[g])
                a.text(0.98, 0.95, "\n".join(txt), transform=a.transAxes, ha="right", va="top", fontsize=4.8)
                a.set_xlim(-50, 200)
                a.set_xlabel("Time from stimulus onset (ms)", fontsize=5.5)
                a.set_ylabel(ylab, fontsize=5.5)
                if i == 0:
                    a.legend(frameon=False, fontsize=5, loc="center right")
            for a in axs.flat[len(groups) + 1:]:
                a.axis("off")
            fig.suptitle(f"{m.capitalize()} stimulus: PSTHs of excited neurons with a fast vs slow response peak "
                         f"(peak z ≥ {Z_MIN:g})\nmean ± s.e.m. over sessions (u = neurons, s = sessions); "
                         f"area groups with ≥ {MIN_SESS} sessions per class"
                         + ("; grey band = artefact-replaced window" if m == "whisker" else ""), fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            for ext in ("png", "pdf", "svg"):
                fig.savefig(OUT / f"psths_{m}_{kind}.{ext}", dpi=300)
            plt.close(fig)
    print("figures written", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        compute(a.n_proc)
    plot()
