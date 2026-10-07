"""072 -- Does the spontaneous-lick (SL) definition matter for the pre-lick convergence analysis? (user 2026-10-07)
One example session at a time: the SL reference of 051 vs the definition of behaviour_analysis/spontaneous_licks_utils.py
(Axel Bisi), compared on lick trains, statistics, trial licks in the piezo trace, lick-aligned population responses and
the per-unit pre-lick selectivities that the convergence analysis uses.

Definitions (all restricted to the analysed task epoch of 051: active context, perf != 6, warm-up cut, A1 trim):
  A  current (051.spontaneous_licks): piezo licks; onset = lick >= 1.0 s after the previous piezo lick; dropped if inside
     any trial window [start_time - 0.2 s, response_window_stop_time + 0.5 s].
  B  utils (spontaneous_licks_utils): piezo licks merged with the trials-table lick_time (uncorrected), debounced (50 ms),
     clusters split at gaps >= 1.0 s, typed bout (>= 5 licks within 1.5 s) / single / short_cluster; onsets within
     +-1.0 s of a hit-trial stimulus onset dropped (auditory hits; whisker hits too in R+).
     B_all: every cluster onset; B_bout: bouts only; B_nonbout: singles + short clusters.
Trial licks: for hit and false-alarm trials, the distance from the trials-table lick_time (raw) and from the corrected
  first lick (start_time + lick_time - response_window_start_time) to the nearest piezo lick.
Neural (good + mua units of the v2 unit table, artefact-corrected spikes as in 051): lick-aligned PSTH (10-ms bins,
  -0.6..0.4 s, baseline = rate in [-1.0, -0.5] s before each event, as 051 for SL; first lick of hits with the pre-trial
  baseline); pre-lick rate [-100, 0) ms minus baseline; per unit AUC selectivity (2 AUC - 1) for AH vs SL and WH vs SL
  for every definition, 1000 label permutations (051.auc_two).
Usage (haas): python 072_sl_definitions_example.py --session AB080_20230622_152205
Output: combined_results_ks4/ssl-prelick-convergence/sl_definitions/<session>/: sl_events.csv, trial_licks.csv,
  unit_selectivity.csv, summary.json, sl_definitions_<session>.{png,pdf,svg}
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code" / "NWB_reader"))
m51 = importlib.import_module("051_roc_prelick")
ru = m51.ru
AXEL = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi")
UTILS = AXEL / "behaviour_analysis" / "spontaneous_licks_utils.py"
UNITS = AXEL / "combined_results_ks4" / "_roc_stage_analysis" / "units.parquet"
OUT = AXEL / "combined_results_ks4" / "ssl-prelick-convergence" / "sl_definitions"
EXCL_S = 1.0                 # utils exclusion window around hit stimulus onsets (its filter default)
TOL = (0.010, 0.025, 0.050)  # s, trial-lick vs piezo matching tolerances
PSTH_WIN, BIN = (-0.6, 0.4), 0.010
DEFS = ["A", "B_all", "B_bout", "B_nonbout"]
DEF_LABEL = {"A": "A: current (051)", "B_all": "B: utils, all events", "B_bout": "B: utils, bouts",
             "B_nonbout": "B: utils, single + short"}
DEF_C = {"A": "#000000", "B_all": "#20118f", "B_bout": "#4158d9", "B_nonbout": "#a6a4a1", "AH": "#2c2cdb", "WH": "#f7b519"}


def load_utils():
    spec = importlib.util.spec_from_file_location("spontaneous_licks_utils", UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nearest(a, b):
    """distance from each a to the nearest b (both sorted not required)"""
    b = np.sort(b)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return np.minimum(np.abs(a - b[i - 1]), np.abs(a - b[i]))


def in_windows(x, lo, hi):
    m = np.zeros(len(x), bool)
    for a, b in zip(lo, hi):
        m |= (x >= a) & (x <= b)
    return m


def psth(spikes, ev, base):
    edges = np.arange(PSTH_WIN[0], PSTH_WIN[1] + BIN / 2, BIN)
    out = np.zeros((len(spikes), len(edges) - 1))
    for u, s in enumerate(spikes):
        rel = np.concatenate([s[np.searchsorted(s, e + PSTH_WIN[0]):np.searchsorted(s, e + PSTH_WIN[1])] - e for e in ev]) \
            if len(ev) else np.array([])
        b = np.mean([(np.searchsorted(s, hi) - np.searchsorted(s, lo)) / (hi - lo) for lo, hi in base]) if len(base) else 0
        out[u] = np.histogram(rel, edges)[0] / max(len(ev), 1) / BIN - b
    return out, edges


def prelick_rates(spikes, ev, base):
    """(U, n) pre-lick rate [-100, 0) ms minus per-event baseline"""
    W = np.zeros((len(spikes), len(ev)))
    for u, s in enumerate(spikes):
        w = (np.searchsorted(s, ev) - np.searchsorted(s, ev - 0.1)) / 0.1
        b = (np.searchsorted(s, base[:, 1]) - np.searchsorted(s, base[:, 0])) / (base[:, 1] - base[:, 0])
        W[u] = w - b
    return W


def main(a):
    from pynwb import NWBHDF5IO
    sid = a.session
    nwb_path = m51.NWB / f"{sid}.nwb"
    U = load_utils()
    out = OUT / sid
    out.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb)
        trials_raw = nwb.trials.to_dataframe()
        t, log = m51.select_trials(trials_raw, sid)
        piezo = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                                   .time_series["piezo_lick_times"].data[:], float))
        ev = {"A": m51.spontaneous_licks(nwb, trials_raw, log["epoch"])}
    ep = log["epoch"]
    # ---- definition B (utils functions, unchanged)
    meta = U.load_session_lick_data(str(nwb_path))
    bev, deb = U.detect_lick_bouts(meta["piezo_times"], bout_gap_s=1.0, bout_min_licks=5, bout_max_window_s=1.5,
                                   min_isi_debounce=0.05, return_debounced=True)
    bev = U.exclude_events_near_task_trials(bev, meta, window_s=EXCL_S)
    bev = bev[(bev.lick_time >= ep[0]) & (bev.lick_time <= ep[1])].reset_index(drop=True)
    ev["B_all"] = bev.lick_time.to_numpy()
    ev["B_bout"] = bev.lick_time[bev.event_type == "bout"].to_numpy()
    ev["B_nonbout"] = bev.lick_time[bev.event_type != "bout"].to_numpy()
    cohort = "R+" if meta["reward_group"] == 1 else "R-"

    # ---- trial windows (all trials) and where B events fall
    tr = trials_raw.sort_values("start_time").reset_index(drop=True)
    lo, hi = tr.start_time.to_numpy() - 0.2, tr.response_window_stop_time.to_numpy() + 0.5
    rows = []
    for d in DEFS:
        for x in ev[d]:
            j = np.searchsorted(tr.start_time.to_numpy(), x) - 1
            inside = bool(in_windows(np.array([x]), lo, hi)[0])
            rows.append(dict(definition=d, time=x, inside_trial_window=inside,
                             trial_type_of_window=(tr.trial_type.iloc[j] if (inside and j >= 0) else ""),
                             since_prev_trial_start=(x - tr.start_time.iloc[j]) if j >= 0 else np.nan,
                             in_A=bool(len(ev["A"]) and nearest(np.array([x]), ev["A"])[0] < 0.025)))
    E = pd.DataFrame(rows)
    bmerge = bev.assign(definition="B_all").rename(columns={"lick_time": "time"})[["definition", "time", "event_type", "n_licks",
                                                                                     "cluster_dur"]]
    E = E.merge(bmerge, on=["definition", "time"], how="left")
    E.to_csv(out / "sl_events.csv", index=False)

    # ---- trial licks vs piezo
    trl = tr[(tr.lick_flag == 1) & tr.lick_time.notna()].copy()
    trl["corrected_lick"] = trl.start_time + (trl.lick_time - trl.response_window_start_time)
    trl["kind"] = np.select([trl.trial_type == "no_stim_trial", trl.trial_type == "whisker_trial", trl.trial_type == "auditory_trial"],
                            ["FA", "WH", "AH"], "other")
    trl["context2"] = m51.resolve_context(tr).reindex(trl.index).to_numpy()
    trl["d_raw_ms"] = 1000 * nearest(trl.lick_time.to_numpy(), piezo)
    trl["d_corr_ms"] = 1000 * nearest(trl.corrected_lick.to_numpy(), piezo)
    trl["piezo_n_in_rw"] = [int(((piezo >= r.response_window_start_time) & (piezo <= r.response_window_stop_time)).sum())
                            for r in trl.itertuples()]
    trl[["trial_id", "kind", "context2", "start_time", "lick_time", "corrected_lick", "d_raw_ms", "d_corr_ms", "piezo_n_in_rw"]] \
        .to_csv(out / "trial_licks.csv", index=False)

    # ---- neural
    U2 = pd.read_parquet(UNITS, columns=["session_id", "electrode_group", "cluster_id", "quality_label"])
    q = U2[U2.session_id == sid].drop_duplicates(["electrode_group", "cluster_id"])
    units = units.reset_index(drop=True)
    key = units.electrode_group.astype(str) + "|" + units.cluster_id.astype(int).astype(str)
    good = set(q[q.quality_label.isin(["good", "mua"])].apply(lambda r: f"{r.electrode_group}|{int(r.cluster_id) % 1000000}", axis=1))
    sel = key.isin(good).to_numpy() if len(good) else np.ones(len(units), bool)
    spikes = [np.sort(np.asarray(s)) for s, k in zip(units.spike_times, sel) if k]
    hits = {c: t[t.cls == c] for c in ("AH", "WH")}
    hit_ev = {c: h.first_lick_time.to_numpy() for c, h in hits.items()}
    hit_base = {c: np.c_[h.base_lo.to_numpy(), h.base_hi.to_numpy()] for c, h in hits.items()}
    sl_base = {d: np.c_[ev[d] + m51.SL_BASE[0], ev[d] + m51.SL_BASE[1]] for d in DEFS}
    P = {d: psth(spikes, ev[d], sl_base[d])[0] for d in DEFS}
    for c in ("AH", "WH"):
        P[c], edges = psth(spikes, hit_ev[c], hit_base[c])
    tc = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(0)
    R = {c: prelick_rates(spikes, hit_ev[c], hit_base[c]) for c in ("AH", "WH")}
    sel_rows = {}
    for d in DEFS:
        if len(ev[d]) < m51.N_MIN:
            continue
        Rs = prelick_rates(spikes, ev[d], sl_base[d])
        for c in ("AH", "WH"):
            if R[c].shape[1] < m51.N_MIN:
                continue
            X = np.c_[Rs, R[c]]
            y = np.r_[np.zeros(Rs.shape[1]), np.ones(R[c].shape[1])].astype(int)
            auc, null = m51.auc_two(X, y, rng)
            s = 2 * auc - 1
            p = np.where(s >= 0, (null >= auc[:, None]).mean(1), (null <= auc[:, None]).mean(1))
            sel_rows[f"{c}_vs_{d}"] = s
            sel_rows[f"{c}_vs_{d}_sig"] = p < m51.ALPHA
    S = pd.DataFrame(sel_rows)
    S.to_csv(out / "unit_selectivity.csv", index=False)

    # ---- summary
    def ioi(x):
        return np.diff(np.sort(x)) if len(x) > 1 else np.array([])
    summ = dict(session=sid, cohort=cohort, epoch=ep, n_units=int(sel.sum()), n_piezo=int(len(piezo)),
                n_piezo_in_epoch=int(((piezo >= ep[0]) & (piezo <= ep[1])).sum()),
                n_merged=int(len(meta["piezo_times"])), n_debounced=int(len(deb)),
                n_events={d: int(len(ev[d])) for d in DEFS},
                B_types=bev.event_type.value_counts().to_dict(),
                overlap_A_in_Ball=float(np.mean(nearest(ev["A"], ev["B_all"]) < 0.025)) if len(ev["A"]) and len(ev["B_all"]) else None,
                overlap_Ball_in_A=float(np.mean(nearest(ev["B_all"], ev["A"]) < 0.025)) if len(ev["A"]) and len(ev["B_all"]) else None,
                B_inside_trial_windows=E[(E.definition == "B_all")].inside_trial_window.mean(),
                B_inside_by_trial_type=E[(E.definition == "B_all") & E.inside_trial_window].trial_type_of_window.value_counts().to_dict(),
                ioi_median_s={d: float(np.median(ioi(ev[d]))) if len(ev[d]) > 1 else None for d in DEFS},
                trial_licks={k: {f"frac_raw_within_{int(1000 * tl)}ms": float((g.d_raw_ms < 1000 * tl).mean()) for tl in TOL}
                             | {f"frac_corrected_within_{int(1000 * tl)}ms": float((g.d_corr_ms < 1000 * tl).mean()) for tl in TOL}
                             | {"n": int(len(g)), "median_d_raw_ms": float(g.d_raw_ms.median()),
                                "median_d_corrected_ms": float(g.d_corr_ms.median()),
                                "frac_no_piezo_in_response_window": float((g.piezo_n_in_rw == 0).mean())}
                             for k, g in trl[trl.kind != "other"].groupby("kind")},
                n_trials_AH=int(len(hit_ev["AH"])), n_trials_WH=int(len(hit_ev["WH"])),
                selectivity_corr={f"{c}: A vs {d}": float(S[f"{c}_vs_A"].corr(S[f"{c}_vs_{d}"]))
                                  for c in ("AH", "WH") for d in DEFS[1:] if f"{c}_vs_A" in S and f"{c}_vs_{d}" in S},
                frac_sig={k: float(S[k].mean()) for k in S if k.endswith("_sig")},
                mean_abs_sel={k: float(S[k].abs().mean()) for k in S if not k.endswith("_sig")})
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    figure(sid, cohort, piezo, deb, ev, bev, tr, trl, P, tc, S, E, out, ep)


def figure(sid, cohort, piezo, deb, ev, bev, tr, trl, P, tc, S, E, out, ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 6,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(7.4, 8.2))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.1, 1.1], hspace=0.6, wspace=0.38, left=0.08, right=0.98, top=0.94, bottom=0.06)
    # a: 60-s raster segment with the most A events
    ax = fig.add_subplot(gs[0, :])
    t0 = ep[0] + 0.25 * (ep[1] - ep[0]); t1 = t0 + 60
    for s, e in zip(tr.start_time, tr.response_window_stop_time):
        if e > t0 and s < t1:
            ax.axvspan(s - 0.2, e + 0.5, color="0.9", lw=0)
    ax.eventplot([piezo[(piezo >= t0) & (piezo <= t1)]], lineoffsets=5, linelengths=0.8, colors="0.3", lw=0.5)
    dl = deb[(deb >= t0) & (deb <= t1)]
    ax.eventplot([dl], lineoffsets=4, linelengths=0.8, colors="0.55", lw=0.5)
    for k, d in enumerate(DEFS):
        x = ev[d][(ev[d] >= t0) & (ev[d] <= t1)]
        ax.plot(x, np.full(len(x), 3 - k), "v", ms=3, color=DEF_C[d])
    tl = trl[(trl.corrected_lick >= t0) & (trl.corrected_lick <= t1)]
    ax.plot(tl.lick_time, np.full(len(tl), 5.6), "|", ms=5, color="r", mew=0.8)
    ax.plot(tl.corrected_lick, np.full(len(tl), 5.9), "|", ms=5, color="g", mew=0.8)
    ax.set_yticks([5.9, 5.6, 5, 4, 3, 2, 1, 0], ["trial lick, corrected", "trial lick_time (raw)", "piezo licks", "merged + debounced",
                                                 *[DEF_LABEL[d] for d in DEFS]], fontsize=5)
    ax.set_xlim(t0, t1); ax.set_xlabel("Session time (s)")
    ax.set_title("60-s segment; grey bands: trial windows [start - 0.2 s, response window end + 0.5 s]", loc="left", fontsize=6)
    # b: counts
    ax = fig.add_subplot(gs[1, 0])
    n = [len(ev[d]) for d in DEFS]
    ax.bar(range(len(DEFS)), n, color=[DEF_C[d] for d in DEFS])
    ax.set_xticks(range(len(DEFS)), ["A", "B all", "B bout", "B other"]); ax.set_ylabel("Events in task epoch")
    for i, v in enumerate(n):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=5)
    inside = E[E.definition == "B_all"].inside_trial_window.mean()
    ax.set_title(f"Counts (B inside trial windows: {100 * inside:.0f} %)", loc="left", fontsize=6)
    # c: inter-event intervals
    ax = fig.add_subplot(gs[1, 1])
    bins = np.logspace(0, 3, 30)
    for d in DEFS:
        x = np.diff(np.sort(ev[d]))
        if len(x):
            ax.hist(x, bins, histtype="step", color=DEF_C[d], lw=0.9, density=True, label=DEF_LABEL[d])
    ax.set_xscale("log"); ax.set_xlabel("Inter-event interval (s)"); ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=4.5); ax.set_title("Inter-event intervals", loc="left", fontsize=6)
    # d: licks per cluster (B)
    ax = fig.add_subplot(gs[1, 2])
    for ty, c in (("bout", DEF_C["B_bout"]), ("short_cluster", DEF_C["B_nonbout"]), ("single", "0.75")):
        x = bev.n_licks[bev.event_type == ty]
        if len(x):
            ax.hist(x, np.arange(0.5, 30.5, 1), color=c, alpha=0.8, label=f"{ty} ({len(x)})")
    ax.set_xlabel("Licks per cluster (B)"); ax.set_ylabel("Clusters"); ax.legend(frameon=False, fontsize=4.5)
    ax.set_title("Cluster size", loc="left", fontsize=6)
    # e: trial licks vs piezo
    ax = fig.add_subplot(gs[2, 0])
    bins = np.r_[0, np.logspace(-1, 3, 30)]
    for k, c in (("FA", "k"), ("AH", DEF_C["AH"]), ("WH", DEF_C["WH"])):
        g = trl[trl.kind == k]
        if len(g):
            ax.hist(g.d_corr_ms.clip(upper=999), bins, histtype="step", color=c, lw=0.9, label=f"{k} corrected (n = {len(g)})")
            ax.hist(g.d_raw_ms.clip(upper=999), bins, histtype="step", color=c, lw=0.6, ls=":", label=f"{k} raw lick_time")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("Trial lick to nearest piezo lick (ms)"); ax.set_ylabel("Trials")
    ax.legend(frameon=False, fontsize=4.2); ax.set_title("Are trial licks in the piezo trace?", loc="left", fontsize=6)
    # f: population PSTH
    ax = fig.add_subplot(gs[2, 1:])
    for d in DEFS + ["AH", "WH"]:
        m = P[d].mean(0); se = P[d].std(0) / np.sqrt(len(P[d]))
        ls = "-" if d in DEFS else "--"
        lab = DEF_LABEL.get(d, {"AH": "auditory hit, first lick", "WH": "whisker hit, first lick"}.get(d))
        n_ev = len(ev[d]) if d in ev else None
        ax.plot(1000 * tc, m, color=DEF_C[d], lw=0.9, ls=ls, label=lab + (f" (n = {n_ev})" if n_ev is not None else ""))
        ax.fill_between(1000 * tc, m - se, m + se, color=DEF_C[d], alpha=0.12, lw=0)
    ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0); ax.axvline(0, color="0.4", lw=0.5, ls=":")
    ax.set_xlabel("Time from lick (ms)"); ax.set_ylabel("Rate - baseline (spikes/s), mean ± s.e.m. over units")
    ax.legend(frameon=False, fontsize=4.5, ncol=2); ax.set_title(f"Lick-aligned population response ({P['A'].shape[0]} good + mua units; "
                                                                 "grey: pre-lick window)", loc="left", fontsize=6)
    # g, h: selectivity A vs B
    for j, c in enumerate(("AH", "WH")):
        ax = fig.add_subplot(gs[3, j])
        if f"{c}_vs_A" not in S:
            ax.set_axis_off(); continue
        for d in ("B_all", "B_bout"):
            if f"{c}_vs_{d}" in S:
                r = S[f"{c}_vs_A"].corr(S[f"{c}_vs_{d}"])
                ax.scatter(S[f"{c}_vs_A"], S[f"{c}_vs_{d}"], s=3, color=DEF_C[d], alpha=0.5, lw=0, label=f"{DEF_LABEL[d]}, r = {r:.2f}")
        ax.plot([-1, 1], [-1, 1], color="0.6", lw=0.5, ls="--")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xlabel(f"{c} vs SL selectivity, A (current)"); ax.set_ylabel(f"{c} vs SL selectivity, B")
        ax.legend(frameon=False, fontsize=4.2, loc="upper left"); ax.set_title(f"Pre-lick selectivity per unit, {c} vs SL", loc="left", fontsize=6)
    ax = fig.add_subplot(gs[3, 2])
    labs, vals = [], []
    for c in ("AH", "WH"):
        for d in DEFS:
            k = f"{c}_vs_{d}_sig"
            if k in S:
                labs.append(f"{c}-{d.replace('B_', 'B ')}"); vals.append(100 * S[k].mean())
    ax.barh(range(len(vals)), vals, color=[DEF_C[l.split("-", 1)[1].replace("B ", "B_")] for l in labs])
    ax.set_yticks(range(len(vals)), labs, fontsize=4.8); ax.invert_yaxis(); ax.set_xlabel("Units significant (%)")
    ax.set_title("Selective units (p < 0.05, 1000 permutations)", loc="left", fontsize=6)
    fig.suptitle(f"Spontaneous-lick definitions, {sid} ({cohort}): current (051) vs spontaneous_licks_utils", x=0.02, y=0.99,
                 ha="left", fontsize=7, weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"sl_definitions_{sid}.{ext}", dpi=250)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="AB080_20230622_152205")
    main(ap.parse_args())
