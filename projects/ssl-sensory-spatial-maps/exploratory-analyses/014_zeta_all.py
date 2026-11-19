"""014 -- ZETA test on no-lick trials, all sessions (user 2026-11-18; expands the 013 one-session pilot).

Units: every unit of the 045 ROC table (roc_long; good + mua as in 001), all sessions with an NWB.
Spikes: KS4, whisker artefact window (-10..+5 ms) replaced by Poisson spikes as in 001/011 (seeded per session).
Trial sets (no licks in any):
  W_miss          active whisker trials, lick_flag 0 (001 selection: perf != 6, auditory warm-up cut, rule A1)
  W_pas_pre/post  passive whisker trials before / after the active block -- the ROC rule (ru.process_nwb_tables:
                  passive trial in the first / second half of the trial table)
  W_pas           pre + post pooled (only when both have >= MIN_TR trials; ssl passive rule = pre AND post)
  A_pas_pre/post/A_pas  the same for auditory trials
  AB155 imec4 passive rows dropped (data_utils.keep_passive_units).
ZETA (Montijn et al. 2021 eLife, zetapy 4.2): spikes in 0..200 ms after each onset; the cumulative fraction of spikes
  over time is compared with a linear (constant-rate) cumulative; its maximal absolute deviation is compared with
  100 jittered repeats (onsets shifted uniformly by +- 2 x window) -> Gumbel p / zeta z. No bins, no baseline window.
  IFR (instantaneous firing rate, multi-scale derivative of the deviation curve): peak time = argmax in 5..200 ms.
Excited: zeta p < 0.01 AND the 1-ms PSTH (Gaussian sigma 5 ms, 5..200 ms) deviates more upward than downward from the
  local baseline (-100..-5 ms) AND the upward peak has z >= 3 (Poisson noise of the smoothed PSTH at the baseline rate).
Categories (IFR peak): fast < 20 ms, slow 20-50 ms, late 50-200 ms.
ROC comparison: the 045 ROC (5-35 ms vs -50..-15 ms, permutation p < .05) on the matched trial set: whisker_passive_pre/
  post <-> W_pas_pre/post, auditory_passive_pre/post <-> A_pas_pre/post; W_miss is compared with whisker_active (all
  active whisker trials, licks included -- not matched).
Outputs: ssl-sensory-spatial-maps/zeta_all/{sessions/<sid>.parquet + <sid>_psth.npz, zeta_units.parquet,
  figures/*.{png,pdf,svg}, provenance.json}
Run (haas): PYTHONPATH=~/code/NWB_reader:~/zeta_lib ~/code/unit_spikes_analysis/.venv/bin/python 014_zeta_all.py
            [--n-proc 60] [--sessions a,b] [--plot-only]
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import pathlib
import time
import warnings
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("lat", HERE / "001_unit_latency.py")
lat = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lat)

OUT = lat.OUT / "zeta_all"
KEYS = lat.KEYS
MAX_DUR, N_RES, P_SIG, Z_MIN, MIN_TR = 0.2, 100, 0.01, 3.0, 10
SIG_SMOOTH = 5.0
EDGES = np.round(np.arange(-0.1, 0.25 + 0.0005, 0.001), 6)
TMS = 1000 * (EDGES[:-1] + 0.0005)
SETS = ["W_miss", "W_pas_pre", "W_pas_post", "W_pas", "A_pas_pre", "A_pas_post", "A_pas"]
ROC_MATCH = {"W_miss": "whisker_active", "W_pas_pre": "whisker_passive_pre", "W_pas_post": "whisker_passive_post",
             "A_pas_pre": "auditory_passive_pre", "A_pas_post": "auditory_passive_post"}
CATS = {"fast": (5, 20), "slow": (20, 50), "late": (50, 200.01)}
MOD_COL = {"W": "#f7b519", "A": "#2c2cdb"}


def trial_sets(raw, proc):
    act = lat.stim_trials(raw)
    d = {"W_miss": act.start_time[(act.trial_type == "whisker_trial") & (act.lick_flag == 0)].to_numpy()}
    for m, tt in (("W", "whisker_trial"), ("A", "auditory_trial")):
        for c in ("pre", "post"):
            d[f"{m}_pas_{c}"] = proc.start_time[(proc.context == f"passive_{c}") & (proc.trial_type == tt)].to_numpy()
        if len(d[f"{m}_pas_pre"]) >= MIN_TR and len(d[f"{m}_pas_post"]) >= MIN_TR:
            d[f"{m}_pas"] = np.sort(np.r_[d[f"{m}_pas_pre"], d[f"{m}_pas_post"]])
    return {k: np.sort(v) for k, v in d.items() if len(v) >= MIN_TR}


def psth(st, ev):
    cnt = np.zeros(len(EDGES) - 1)
    for e in ev:
        a, b = np.searchsorted(st, [e + EDGES[0], e + EDGES[-1]])
        cnt += np.histogram(st[a:b] - e, EDGES)[0]
    return cnt / len(ev) / 0.001


def unit_set(st, ev, seed):
    from scipy.ndimage import gaussian_filter1d
    from zetapy import zetatest
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = zetatest(st, ev, max_duration=MAX_DUR, resampling_number=N_RES, return_rate=True, random_seed=seed)
    p, zd, rd = out[:3]               # zetapy returns a 4th value when there are < 3 spikes (p = 1)
    pk_ifr = np.nan
    if isinstance(rd, dict) and rd.get("rate_vector") is not None and len(rd["rate_vector"]):
        tv, rv = np.asarray(rd["timestamps"]), np.asarray(rd["rate_vector"])
        k = (tv >= 0.005) & (tv <= MAX_DUR)
        if k.any():
            pk_ifr = 1000 * tv[k][np.argmax(rv[k])]
    raw = psth(st, ev)
    base = raw[(TMS >= -100) & (TMS < -5)].mean()
    sm = gaussian_filter1d(raw, SIG_SMOOTH) - base
    w = (TMS >= 5) & (TMS <= 1000 * MAX_DUR)
    up, dn = sm[w].max(), -sm[w].min()
    sd = np.sqrt(max(base, 0.5) / (len(ev) * 0.001)) / np.sqrt(2 * np.sqrt(np.pi) * SIG_SMOOTH)
    return dict(n_trials=len(ev), zeta_p=p, zeta_z=zd.get("zeta_score", np.nan), ifr_peak_ms=pk_ifr,
                psth_peak_ms=TMS[w][np.argmax(sm[w])], base_hz=base, up_hz=up, down_hz=dn, up_z=up / sd), raw


def run_session(sid):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    f = OUT / "sessions" / f"{sid}.parquet"
    if f.exists():
        return sid, "cached"
    t0 = time.time()
    from pynwb import NWBHDF5IO
    from roc_analysis import roc_utils_new as ru
    want = pd.read_parquet(lat.ROC / "roc_long.parquet", columns=KEYS, filters=[("session_id", "==", sid),
                           ("analysis_type", "==", "whisker_active")])
    want = set(zip(want.electrode_group, want.cluster_id.astype(str)))
    with NWBHDF5IO(str(lat.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        raw = nwb.trials.to_dataframe()
        units, proc = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
    units["cluster_id"] = units.cluster_id.astype(str)
    units = units[[(g, c) in want for g, c in zip(units.electrode_group, units.cluster_id)]]
    sets = trial_sets(raw, proc)
    onsets = raw.loc[raw.whisker_stim == 1, "start_time"].to_numpy()
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    rows, P = [], []
    for u in units.itertuples():
        st = ru.correct_neuron_spike_train(np.asarray(u.spike_times), onsets, rng)
        for s, ev in sets.items():
            r, ps = unit_set(st, ev, zlib.crc32(f"{sid}{u.electrode_group}{u.cluster_id}{s}".encode()))
            rows.append(dict(mouse_id=sid.split("_")[0], session_id=sid, electrode_group=u.electrode_group,
                             cluster_id=u.cluster_id, trial_set=s, **r))
            P.append(ps.astype(np.float32))
    R = pd.DataFrame(rows)
    np.savez_compressed(OUT / "sessions" / f"{sid}_psth.npz", psth=np.stack(P) if P else np.zeros((0, len(TMS))))
    R.to_parquet(f)
    return sid, f"{len(units)} units, sets {', '.join(f'{k}:{len(v)}' for k, v in sets.items())}, {time.time() - t0:.0f}s"


def compute(n_proc, sessions):
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    if not sessions:
        s = pd.read_parquet(lat.ROC / "roc_long.parquet", columns=["session_id"]).session_id.unique()
        sessions = sorted(x for x in s if (lat.NWB / f"{x}.nwb").exists())
    t0 = time.time()
    with mp.get_context("fork").Pool(n_proc, maxtasksperchild=1) as pool:
        for k, (sid, msg) in enumerate(pool.imap_unordered(run_session, sessions)):
            print(f"{time.time() - t0:6.0f}s {k + 1}/{len(sessions)} {sid}: {msg}", flush=True)


def assemble():
    """all sessions -> zeta_units.parquet (+ area_group, quality, ROC on the matched set) and psths.npz (row-aligned)"""
    from ephys_utilities.helpers import data_utils
    R, P = [], []
    for f in sorted((OUT / "sessions").glob("*.parquet")):
        r = pd.read_parquet(f)
        R.append(r)
        P.append(np.load(f.with_name(f.stem + "_psth.npz"))["psth"])
    R = pd.concat(R, ignore_index=True)
    R["row"] = np.arange(len(R))
    P = np.concatenate(P)
    U = pd.read_parquet(lat.ROC / "units.parquet", columns=KEYS + ["area_group", "area_acronym_custom", "cohort",
                                                                   "day", "quality_label"])
    U["cluster_id"] = U.cluster_id.astype(str)
    R = R.merge(U.drop_duplicates(KEYS), on=KEYS, how="left")
    R = R[R.quality_label.isin(["good", "mua"])]
    pas = R.trial_set.str.contains("pas")
    R = pd.concat([R[~pas], data_utils.keep_passive_units(R[pas])])
    roc = pd.read_parquet(lat.ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sel", "sig"],
                          filters=[("analysis_type", "in", list(ROC_MATCH.values()))])
    roc["cluster_id"] = roc.cluster_id.astype(str)
    roc = roc.drop_duplicates(KEYS + ["analysis_type"])
    inv = {v: k for k, v in ROC_MATCH.items()}
    roc["trial_set"] = roc.analysis_type.map(inv)
    roc = roc.rename(columns={"sel": "roc_sel", "sig": "roc_sig"}).drop(columns="analysis_type")
    R = R.merge(roc, on=KEYS + ["trial_set"], how="left")
    R["zeta_sig"] = R.zeta_p < P_SIG
    R["excited"] = R.zeta_sig & (R.up_hz > R.down_hz) & (R.up_z >= Z_MIN)
    R["category"] = pd.cut(R.ifr_peak_ms, [CATS["fast"][0], 20, 50, 200.01], labels=list(CATS), right=False)
    R["roc_exc"] = (R.roc_sig == True) & (R.roc_sel > 0)  # noqa: E712
    R = R.sort_values("row").reset_index(drop=True)
    P = P[R.row.to_numpy()]
    R["row"] = np.arange(len(R))
    R.to_parquet(OUT / "zeta_units.parquet")
    np.savez_compressed(OUT / "psths.npz", psth=P, t_ms=TMS)
    json.dump(dict(script="014_zeta_all.py", zetapy="4.2", max_dur_s=MAX_DUR, resamplings=N_RES, p_sig=P_SIG,
                   z_min=Z_MIN, sigma_ms=SIG_SMOOTH, min_trials=MIN_TR, categories_ms=CATS, roc_match=ROC_MATCH,
                   n_sessions=int(R.session_id.nunique()), date=time.strftime("%Y-%m-%d %H:%M")),
              open(OUT / "provenance.json", "w"), indent=1)
    return R, P


def summary(R):
    pd.set_option("display.width", 220)
    rows = []
    for s, x in R.groupby("trial_set"):
        e = x[x.excited]
        r = dict(trial_set=s, sessions=x.session_id.nunique(), units=len(x), zeta_sig=x.zeta_sig.mean(),
                 excited=x.excited.mean(), **{f"frac_{c}": (e.category == c).mean() for c in CATS})
        if x.roc_sig.notna().any():
            y = x[x.roc_sig.notna()]
            r.update(roc_exc=y.roc_exc.mean(), both=(y.excited & y.roc_exc).mean(),
                     zeta_only=(y.excited & ~y.roc_exc).mean(), roc_only=(~y.excited & y.roc_exc).mean())
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary_sets.csv", index=False)
    print(S.round(3).to_string())
    return S


# ---------------------------------------------------------------- figures
def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    return plt


def save(fig, name):
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def session_curve(Y, x):
    S = np.stack([np.nanmean(Y[g.row.to_numpy()], 0) for _, g in x.groupby("session_id")])
    S = S[np.isfinite(S).all(1)]
    se = S.std(0, ddof=1) / np.sqrt(len(S)) if len(S) > 1 else np.zeros(S.shape[1])
    return S.mean(0), se, len(S)


def fig_method(R):
    """worked example: one fast, one late, one ROC-missed neuron (W_pas_pre, ROC-matched), the ZETA steps for each"""
    from pynwb import NWBHDF5IO
    from roc_analysis import roc_utils_new as ru
    from zetapy import zetatest
    plt = mpl()
    x = R[(R.trial_set == "W_pas_pre") & R.excited & R.roc_sig.notna()]
    sid = x.groupby("session_id").size().idxmax()
    x = x[x.session_id == sid]
    pick = [("fast, ROC-responsive", x[(x.category == "fast") & x.roc_exc].sort_values("zeta_z").iloc[-1]),
            ("late, ROC-responsive", x[(x.category == "late") & x.roc_exc].sort_values("zeta_z").iloc[-1]
             if ((x.category == "late") & x.roc_exc).any() else None),
            ("late, missed by the ROC", x[(x.category == "late") & ~x.roc_exc].sort_values("zeta_z").iloc[-1])]
    pick = [(a, b) for a, b in pick if b is not None]
    with NWBHDF5IO(str(lat.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        raw = nwb.trials.to_dataframe()
        units, proc = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
    units["cluster_id"] = units.cluster_id.astype(str)
    ev = trial_sets(raw, proc)["W_pas_pre"]
    onsets = raw.loc[raw.whisker_stim == 1, "start_time"].to_numpy()
    fig, ax = plt.subplots(len(pick), 4, figsize=(7.4, 1.7 * len(pick)), squeeze=False,
                           gridspec_kw=dict(wspace=0.45, hspace=0.75))
    for i, (lab, r) in enumerate(pick):
        u = units[(units.electrode_group == r.electrode_group) & (units.cluster_id == r.cluster_id)].iloc[0]
        st = ru.correct_neuron_spike_train(np.asarray(u.spike_times), onsets, np.random.default_rng(0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, zd, rd = zetatest(st, ev, max_duration=MAX_DUR, resampling_number=N_RES, return_rate=True)[:3]
        a = ax[i, 0]
        for k, e in enumerate(ev):
            s = st[(st >= e - 0.05) & (st < e + MAX_DUR)] - e
            a.plot(1000 * s, np.full(len(s), k), "|", ms=1.2, mew=0.4, color="k")
        a.axvspan(0, 1000 * MAX_DUR, facecolor="0.93", lw=0, zorder=0)
        a.set_xlim(-50, 1000 * MAX_DUR)
        a.set_ylabel("Trial")
        a.set_title(f"{lab}\n{r.area_acronym_custom} ({r.area_group})", fontsize=6)
        a = ax[i, 1]
        tv = 1000 * zd["spike_time_vector"]
        for tr, dr in zip(zd["random_times"][:40], zd["random_deviations"][:40]):
            a.plot(1000 * np.asarray(tr), dr, color="0.75", lw=0.3)
        a.plot(tv, zd["real_deviation"], color=MOD_COL["W"], lw=0.9)
        j = zd["zeta_index"]
        a.plot(tv[j], zd["real_deviation"][j], "o", ms=2.5, color="k")
        a.axhline(0, color="0.5", lw=0.4, ls=":")
        a.set_title("2. Deviation of cumulative spikes\nfrom constant rate (grey: jittered)", fontsize=6)
        a.set_ylabel("Deviation (fraction of spikes)")
        a = ax[i, 2]
        nul = np.array([np.max(np.abs(d)) for d in zd["random_deviations"]])
        a.hist(nul, 20, color="0.7", lw=0)
        a.axvline(np.abs(zd["real_deviation"][j]), color=MOD_COL["W"], lw=1)
        a.set_title(f"3. Max |deviation| vs jitter null\nZETA z = {zd['zeta_score']:.1f}, p = {p:.1e}", fontsize=6)
        a.set_xlabel("Max |deviation|")
        a = ax[i, 3]
        a.plot(1000 * np.asarray(rd["timestamps"]), rd["rate_vector"], color=MOD_COL["W"], lw=0.7, label="IFR")
        a.plot(TMS, psth(st, ev), color="0.4", lw=0.4, label="1-ms PSTH")
        a.axvline(r.ifr_peak_ms, color="k", lw=0.5, ls="--")
        a.set_xlim(-50, 1000 * MAX_DUR)
        a.set_title(f"4. Instantaneous rate; peak {r.ifr_peak_ms:.0f} ms\n"
                    f"ROC (5-35 ms) {'significant' if r.roc_exc else 'not significant'}", fontsize=6)
        a.set_ylabel("Rate (Hz)")
        a.legend(frameon=False, fontsize=4.5)
        for a in ax[i, [0, 1, 3]]:
            a.set_xlabel("Time from whisker stimulus (ms)")
    ax[0, 0].set_title("1. Spikes per trial (0-200 ms used)\n" + ax[0, 0].get_title(), fontsize=6)
    fig.suptitle(f"How ZETA works: three example neurons, passive whisker trials before training ({sid}, {len(ev)} trials)", fontsize=7)
    save(fig, "zeta_method")
    plt.close(fig)


def fig_psths(R, P):
    """mean +- s.e.m. over sessions of the PSTHs of ZETA-excited units per category and area group"""
    from ephys_utilities.allen_utils import allen_utils as au
    from scipy.ndimage import gaussian_filter1d
    plt = mpl()
    pal = au.get_custom_area_groups_colors()
    order = au.get_area_group_custom_order()
    Ps = gaussian_filter1d(P, 2.0, axis=1)
    Y = Ps - Ps[:, (TMS >= -100) & (TMS < -5)].mean(1, keepdims=True)
    pk = Y[:, (TMS >= 5) & (TMS <= 200)].max(1)
    Yn = Y / np.where(pk > 0, pk, np.nan)[:, None]
    for s in ("W_miss", "W_pas", "A_pas"):
        x = R[(R.trial_set == s) & R.excited & R.category.notna()]
        col = {"fast": MOD_COL[s[0]], "slow": "0.45", "late": "k"}
        ns = x.groupby(["area_group", "category"], observed=True).session_id.nunique().unstack()
        groups = [g for g in order if g in ns.index and (ns.loc[g].fillna(0) >= 3).sum() >= 2]
        for kind, Z, ylab in (("norm", Yn, "Rate / unit's peak"), ("hz", Y, "Rate − baseline (Hz)")):
            nc = 4
            nr = int(np.ceil((len(groups) + 1) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(7.4, 1.55 * nr), squeeze=False)
            for i, g in enumerate(["All area groups"] + groups):
                a = axs.flat[i]
                sub = x if i == 0 else x[x.area_group == g]
                txt = []
                for c in CATS:
                    y = sub[sub.category == c]
                    if y.session_id.nunique() < 3:
                        continue
                    mu, se, n = session_curve(Z, y)
                    a.fill_between(TMS, mu - se, mu + se, color=col[c], alpha=0.25, lw=0, edgecolor="none")
                    a.plot(TMS, mu, color=col[c], lw=0.8, label=f"{c} (IFR peak {CATS[c][0]:.0f}-{CATS[c][1]:.0f} ms)")
                    txt.append(f"{c} {len(y)} u / {n} s")
                a.axvline(0, color="0.3", lw=0.5, ls="--")
                if s[0] == "W":
                    a.axvspan(-10, 5, facecolor="0.9", lw=0, edgecolor="none")
                a.axhline(0, color="0.6", lw=0.4, ls=":")
                a.set_title(g, fontsize=6.5, color="k" if i == 0 else pal[g])
                a.text(0.98, 0.95, "\n".join(txt), transform=a.transAxes, ha="right", va="top", fontsize=4.6)
                a.set_xlim(-50, 250)
                a.set_xlabel("Time from stimulus onset (ms)", fontsize=5.5)
                a.set_ylabel(ylab, fontsize=5.5)
                if i == 0:
                    a.legend(frameon=False, fontsize=4.6, loc="center right")
            for a in axs.flat[len(groups) + 1:]:
                a.axis("off")
            fig.suptitle(f"{s}: PSTHs of ZETA-excited neurons by IFR peak time (no-lick trials); mean ± s.e.m. over "
                         f"sessions (u = neurons, s = sessions); categories with ≥ 3 sessions shown", fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            save(fig, f"zeta_psths_{s}_{kind}")
            plt.close(fig)


def fig_roc(R):
    """ZETA vs ROC responsive fractions per area group (session mean +- s.e.m.), and overlap, on matched trial sets"""
    from ephys_utilities.allen_utils import allen_utils as au
    from scipy import stats
    plt = mpl()
    pal = au.get_custom_area_groups_colors()
    order = au.get_area_group_custom_order()
    sets = ["W_pas_pre", "W_pas_post", "A_pas_pre", "A_pas_post", "W_miss"]
    fig, ax = plt.subplots(2, len(sets), figsize=(11, 6.6), squeeze=False,
                           gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.55, wspace=1.1))
    T = []
    for j, s in enumerate(sets):
        x = R[(R.trial_set == s) & R.roc_sig.notna()]
        F = x.groupby(["session_id", "area_group"]).agg(n=("row", "size"), z=("excited", "mean"),
                                                         r=("roc_exc", "mean")).reset_index()
        F = F[F.n >= 10]
        G = F.groupby("area_group").agg(s=("session_id", "size"), z=("z", "mean"), zs=("z", "sem"), r=("r", "mean"),
                                        rs=("r", "sem"))
        G = G[G.s >= 3].reindex([g for g in order if g in G.index[G.s >= 3]])
        for g in G.index:
            f = F[F.area_group == g]
            T.append(dict(trial_set=s, area_group=g, sessions=len(f), zeta=f.z.mean(), roc=f.r.mean(),
                          p_wilcoxon=stats.wilcoxon(f.z, f.r).pvalue if len(f) >= 5 and (f.z != f.r).any() else np.nan,
                          p_ttest=stats.ttest_rel(f.z, f.r).pvalue if len(f) >= 3 else np.nan))
        a = ax[0, j]
        yy = np.arange(len(G))[::-1]
        a.barh(yy + 0.2, G.z, 0.4, xerr=G.zs, color=MOD_COL[s[0]], lw=0, error_kw=dict(lw=0.5), label="ZETA excited")
        a.barh(yy - 0.2, G.r, 0.4, xerr=G.rs, color="0.6", lw=0, error_kw=dict(lw=0.5), label="ROC excited")
        a.set_yticks(yy)
        a.set_yticklabels([f"{g} ({int(n)})" for g, n in zip(G.index, G.s)], fontsize=5)
        for lab, g in zip(a.get_yticklabels(), G.index):
            lab.set_color(pal[g])
        a.set_xlabel("Fraction of neurons (session mean ± s.e.m.)")
        roc_name = ROC_MATCH[s] + ("\n(all active trials, not matched)" if s == "W_miss" else "")
        a.set_title(f"ZETA {s}\nvs ROC {roc_name}", fontsize=6)
        a.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        a = ax[1, j]
        e = x[x.excited | x.roc_exc]
        grp = {"both": e[e.excited & e.roc_exc], "ZETA only": e[e.excited & ~e.roc_exc],
               "ROC only": e[~e.excited & e.roc_exc]}
        b = np.arange(0, 201, 5)
        for (k, y), c in zip(grp.items(), [MOD_COL[s[0]], "0.6", "k"]):
            if k == "ROC only":
                continue
            a.hist(y.ifr_peak_ms, b, histtype="step", color=c, lw=0.8, label=f"{k} ({len(y)})")
        a.set_xlabel("ZETA IFR peak time (ms)")
        a.set_ylabel("Neurons per 5 ms")
        a.set_title(f"Peak times; ROC only (no ZETA peak):\n{len(grp['ROC only'])} neurons", fontsize=6)
        a.legend(frameon=False, fontsize=5)
    pd.DataFrame(T).to_csv(OUT / "zeta_vs_roc_areas.csv", index=False)
    fig.suptitle("ZETA (0-200 ms, no baseline window) vs ROC (5-35 ms vs −50..−15 ms) on the same no-lick trials; "
                 "area groups with ≥ 3 sessions of ≥ 10 neurons", fontsize=7)
    save(fig, "zeta_vs_roc")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=60)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        compute(a.n_proc, [s for s in a.sessions.split(",") if s])
        R, P = assemble()
    else:
        R = pd.read_parquet(OUT / "zeta_units.parquet")
        P = np.load(OUT / "psths.npz")["psth"]
    summary(R)
    fig_method(R)
    fig_psths(R, P)
    fig_roc(R)
    print("written", OUT)


if __name__ == "__main__":
    main()
