"""073 -- Spontaneous licks with a long exclusion window after every stimulus trial (user 2026-10-07).

User definition: licks in the response window of no-stim trials (false alarms) ARE spontaneous licks (same behaviour,
measured in a specific window) and are counted as such; what must be excluded is any lick during stimulus processing or
reward collection, i.e. in a window after every whisker / auditory trial, hit or miss (e.g. 8 s).

Definitions (all within the analysed task epoch of 051: active context, perf != 6, warm-up cut, A1 trim):
  A      current 051 reference: piezo licks; onset = lick >= 1.0 s after the previous piezo lick; dropped inside any trial
         window [start_time - 0.2 s, response_window_stop_time + 0.5 s] (no-stim trials included -> false alarms out).
  C(W)   lick train = piezo licks + the corrected first lick of every trial (start_time + lick_time -
         response_window_start_time; piezo detection misses many false-alarm licks, 072), licks < 50 ms apart collapsed
         (earliest kept); onset = lick >= 1.0 s after the previous lick of that train; dropped if in
         [stim - 0.5 s, stim + W] of any whisker or auditory trial (hit or miss); no-stim trials not excluded.
         W = 1, 2, 4, 8, 12 s; C8 is the proposed reference. C8 events are split into false-alarm-window licks (onset in a
         no-stim response window) and other spontaneous licks.
Neural (good + mua units, artefact-corrected spikes as 051): lick-aligned PSTH (10-ms bins, -0.6..0.4 s; baseline = rate
  in [-1.0, -0.5] s before each spontaneous lick; hits: first lick with the pre-trial baseline), pre-lick rate
  [-100, 0) ms minus baseline, per-unit AUC selectivity (2 AUC - 1) AH vs SL and WH vs SL (1000 label permutations).
Usage (haas): python 073_sl_stim_exclusion.py --session AB080_20230622_152205
Output: combined_results_ks4/ssl-prelick-convergence/sl_definitions/<session>/: c_events.csv, c_selectivity.csv,
  c_summary.json, sl_stim_exclusion_<session>.{png,pdf,svg}
"""
import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m72 = importlib.import_module("072_sl_definitions_example")
ru = m51.ru
OUT = m72.OUT
GAP, DEBOUNCE, PRE_GUARD = 1.0, 0.05, 0.5
WS = [1, 2, 4, 8, 12]
W_MAIN = 8
C = {"A": "#000000", "C1": "#c6dbef", "C2": "#9ecae1", "C4": "#4292c6", "C8": "#08519c", "C12": "#08306b",
     "C8_FA": "#e6550d", "C8_other": "#31a354", "AH": "#2c2cdb", "WH": "#f7b519"}
LAB = {"A": "A: current (051)", **{f"C{w}": f"C, W = {w} s" for w in WS},
       "C8_FA": "C8: in no-stim response window (FA)", "C8_other": "C8: other spontaneous",
       "AH": "auditory hit, first lick", "WH": "whisker hit, first lick"}


def debounce(x, d=DEBOUNCE):
    x = np.sort(x)
    keep = np.ones(len(x), bool)
    last = -np.inf
    for i, v in enumerate(x):
        if v - last < d:
            keep[i] = False
        else:
            last = v
    return x[keep]


def onsets(x, gap=GAP):
    return x[np.r_[True, np.diff(x) >= gap]] if len(x) else x


def main(a):
    from pynwb import NWBHDF5IO
    sid = a.session
    out = OUT / sid
    out.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb)
        trials_raw = nwb.trials.to_dataframe()
        t, log = m51.select_trials(trials_raw, sid)
        piezo = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                                   .time_series["piezo_lick_times"].data[:], float))
        evA = m51.spontaneous_licks(nwb, trials_raw, log["epoch"])
    ep = log["epoch"]
    tr = trials_raw.sort_values("start_time").reset_index(drop=True)
    tl = tr[(tr.lick_flag == 1) & tr.lick_time.notna()]
    corr = (tl.start_time + tl.lick_time - tl.response_window_start_time).to_numpy()
    train = debounce(np.r_[piezo, corr])
    ons = onsets(train)
    ons = ons[(ons >= ep[0]) & (ons <= ep[1])]
    stim = tr[tr.trial_type.isin(["whisker_trial", "auditory_trial"])].start_time.to_numpy()
    nostim = tr[tr.trial_type == "no_stim_trial"]
    ev = {"A": evA}
    for w in WS:
        ev[f"C{w}"] = ons[~m72.in_windows(ons, stim - PRE_GUARD, stim + w)]
    fa_win = m72.in_windows(ev["C8"], nostim.response_window_start_time.to_numpy(), nostim.response_window_stop_time.to_numpy())
    ev["C8_FA"], ev["C8_other"] = ev["C8"][fa_win], ev["C8"][~fa_win]

    def since_stim(x):
        j = np.searchsorted(stim, x) - 1
        return np.where(j >= 0, x - stim[np.clip(j, 0, None)], np.nan)
    rows = [dict(definition=d, time=x, since_prev_stim_s=s, in_C8=bool(len(ev["C8"]) and m72.nearest(np.array([x]), ev["C8"])[0] < 0.025),
                 in_A=bool(len(evA) and m72.nearest(np.array([x]), evA)[0] < 0.025))
            for d in ["A", "C8"] for x, s in zip(ev[d], since_stim(ev[d]))]
    pd.DataFrame(rows).to_csv(out / "c_events.csv", index=False)

    # ---- neural
    U2 = pd.read_parquet(m72.UNITS, columns=["session_id", "electrode_group", "cluster_id", "quality_label"])
    q = U2[U2.session_id == sid].drop_duplicates(["electrode_group", "cluster_id"])
    units = units.reset_index(drop=True)
    key = units.electrode_group.astype(str) + "|" + units.cluster_id.astype(int).astype(str)
    good = set(q[q.quality_label.isin(["good", "mua"])].apply(lambda r: f"{r.electrode_group}|{int(r.cluster_id) % 1000000}", axis=1))
    sel = key.isin(good).to_numpy()
    spikes = [np.sort(np.asarray(s)) for s, k in zip(units.spike_times, sel) if k]
    hits = {c: t[t.cls == c] for c in ("AH", "WH")}
    hit_ev = {c: h.first_lick_time.to_numpy() for c, h in hits.items()}
    hit_base = {c: np.c_[h.base_lo.to_numpy(), h.base_hi.to_numpy()] for c, h in hits.items()}
    base = {d: np.c_[ev[d] + m51.SL_BASE[0], ev[d] + m51.SL_BASE[1]] for d in ev}
    P = {}
    for d in ["A", "C1", "C8", "C8_FA", "C8_other"]:
        if len(ev[d]):
            P[d], edges = m72.psth(spikes, ev[d], base[d])
    for c in ("AH", "WH"):
        P[c], edges = m72.psth(spikes, hit_ev[c], hit_base[c])
    tc = (edges[:-1] + edges[1:]) / 2
    R = {c: m72.prelick_rates(spikes, hit_ev[c], hit_base[c]) for c in ("AH", "WH")}
    rng = np.random.default_rng(0)
    S, pre = {}, {}
    for d in ev:
        if len(ev[d]) < m51.N_MIN:
            continue
        Rs = m72.prelick_rates(spikes, ev[d], base[d])
        pre[d] = Rs.mean(1)
        for c in ("AH", "WH"):
            X = np.c_[Rs, R[c]]
            y = np.r_[np.zeros(Rs.shape[1]), np.ones(R[c].shape[1])].astype(int)
            auc, null = m51.auc_two(X, y, rng)
            s = 2 * auc - 1
            p = np.where(s >= 0, (null >= auc[:, None]).mean(1), (null <= auc[:, None]).mean(1))
            S[f"{c}_vs_{d}"], S[f"{c}_vs_{d}_sig"] = s, p < m51.ALPHA
    S = pd.DataFrame(S)
    S.to_csv(out / "c_selectivity.csv", index=False)
    sA = since_stim(evA)
    summ = dict(session=sid, epoch=ep, n_units=int(sel.sum()), n_piezo=int(len(piezo)), n_trial_licks_added=int(len(corr)),
                n_train=int(len(train)), n_onsets_epoch=int(len(ons)), n_events={d: int(len(v)) for d, v in ev.items()},
                A_within_8s_after_stim=float(np.mean(sA <= 8)) if len(sA) else None,
                A_in_C8=float(np.mean(m72.nearest(evA, ev["C8"]) < 0.025)) if len(evA) and len(ev["C8"]) else None,
                C8_in_A=float(np.mean(m72.nearest(ev["C8"], evA) < 0.025)) if len(evA) and len(ev["C8"]) else None,
                prelick_rate_mean={d: float(v.mean()) for d, v in pre.items()},
                selectivity_corr_vs_A={f"{c}: {d}": float(S[f"{c}_vs_A"].corr(S[f"{c}_vs_{d}"]))
                                       for c in ("AH", "WH") for d in ev if d != "A" and f"{c}_vs_{d}" in S},
                frac_sig={k: float(S[k].mean()) for k in S if k.endswith("_sig")},
                mean_sel={k: float(S[k].mean()) for k in S if not k.endswith("_sig")},
                fa_vs_other_psth_corr=float(np.corrcoef(P["C8_FA"].mean(0), P["C8_other"].mean(0))[0, 1])
                if "C8_FA" in P and "C8_other" in P else None)
    (out / "c_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    figure(sid, piezo, corr, train, ev, stim, nostim, tr, ep, P, tc, S, pre, sA, out, summ)


def figure(sid, piezo, corr, train, ev, stim, nostim, tr, ep, P, tc, S, pre, sA, out, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 6,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(7.4, 9.0))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.25, 1.0, 1.15, 1.0], hspace=0.62, wspace=0.4, left=0.1, right=0.98, top=0.94,
                          bottom=0.05)
    # a: method on a 60-s segment with stimulus trials
    ax = fig.add_subplot(gs[0, :])
    starts = np.arange(ep[0], ep[1] - 60, 5)              # segment: most A + C1 + C8 events, >= 2 stimuli, >= 1 no-stim trial
    cnt = lambda x, s0: int(((x >= s0) & (x < s0 + 60)).sum())
    ns_t = nostim.start_time.to_numpy()
    score = [cnt(ev["A"], s0) + cnt(ev["C1"], s0) + cnt(ev["C8"], s0) + 3 * cnt(ev["C8_FA"], s0)
             if cnt(stim, s0) >= 2 and cnt(ns_t, s0) >= 1 else -1 for s0 in starts]
    t0 = starts[int(np.argmax(score))]; t1 = t0 + 60
    for s in stim[(stim > t0 - 9) & (stim < t1)]:
        ax.add_patch(Rectangle((s - PRE_GUARD, -0.5), PRE_GUARD + W_MAIN, 7, color="#fde0dd", lw=0, zorder=0))
        ax.add_patch(Rectangle((s - PRE_GUARD, -0.5), PRE_GUARD + 1, 7, color="#fa9fb5", lw=0, zorder=0, alpha=0.6))
        ax.axvline(s, color="#c51b8a", lw=0.8)
    for r in nostim[(nostim.start_time > t0) & (nostim.start_time < t1)].itertuples():
        ax.add_patch(Rectangle((r.response_window_start_time, -0.5), r.response_window_stop_time - r.response_window_start_time, 7,
                               color="#fee6ce", lw=0, zorder=0))
        ax.axvline(r.start_time, color="#e6550d", lw=0.6, ls=":")
    sl = lambda x: x[(x >= t0) & (x <= t1)]
    ax.eventplot([sl(piezo)], lineoffsets=5.5, linelengths=0.7, colors="0.35", lw=0.5)
    ax.plot(sl(corr), np.full(len(sl(corr)), 6.2), "|", color="#e6550d", ms=6, mew=1)
    ax.eventplot([sl(train)], lineoffsets=4.5, linelengths=0.7, colors="0.6", lw=0.5)
    for k, (d, lab) in enumerate([("A", LAB["A"]), ("C1", LAB["C1"]), ("C8", LAB["C8"])]):
        x = sl(ev[d]); ax.plot(x, np.full(len(x), 3 - k), "v", ms=3.5, color=C[d])
    ax.set_yticks([6.2, 5.5, 4.5, 3, 2, 1], ["trial first lick (corrected)", "piezo licks", "lick train (merged, 50-ms collapse)",
                                             LAB["A"], LAB["C1"], LAB["C8"] + " (proposed)"], fontsize=5)
    ax.set_ylim(0.3, 6.7); ax.set_xlim(t0, t1); ax.set_xlabel("Session time (s)")
    ax.set_title("Method, 60-s example. Magenta: whisker / auditory stimulus; pink: excluded [stim - 0.5 s, stim + 8 s] (dark: up to "
                 "+ 1 s);\ndotted orange: no-stim trial, orange band: its response window (licks kept: false alarms are spontaneous licks)",
                 loc="left", fontsize=5.5)
    # b: counts vs W
    ax = fig.add_subplot(gs[1, 0])
    n = [len(ev[f"C{w}"]) for w in WS]
    ax.plot(WS, n, "o-", color=C["C8"], ms=3); ax.axhline(len(ev["A"]), color="k", lw=0.8, ls="--", label=f"A (current): {len(ev['A'])}")
    for w, v in zip(WS, n):
        ax.text(w, v, f" {v}", fontsize=5, va="bottom")
    ax.set_xlabel("Exclusion after stimulus, W (s)"); ax.set_ylabel("Spontaneous licks (bout onsets)")
    ax.legend(frameon=False, fontsize=5); ax.set_title(f"Events vs W (C8: {len(ev['C8_FA'])} in FA windows)", loc="left", fontsize=6)
    # c: time since previous stimulus of A events
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(sA[np.isfinite(sA)], np.arange(0, 40.5, 1), color="0.4")
    ax.axvspan(0, W_MAIN, color="#fde0dd", lw=0, zorder=0)
    ax.set_xlabel("A events: time since previous stimulus (s)"); ax.set_ylabel("Events")
    ax.set_title(f"A within 8 s of a stimulus: {100 * summ['A_within_8s_after_stim']:.0f} %", loc="left", fontsize=6)
    # d: pre-lick rate vs W
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(WS, [pre[f"C{w}"].mean() for w in WS if f"C{w}" in pre], "o-", color=C["C8"], ms=3, label="C(W)")
    ax.axhline(pre["A"].mean(), color="k", ls="--", lw=0.8, label="A")
    ax.set_xlabel("W (s)"); ax.set_ylabel("Pre-lick rate - baseline (spikes/s),\nmean over units")
    ax.legend(frameon=False, fontsize=5); ax.set_title("Pre-lick activity vs W", loc="left", fontsize=6)
    # e: PSTHs
    ax = fig.add_subplot(gs[2, :2])
    for d, ls in (("A", "-"), ("C1", "-"), ("C8", "-"), ("C8_FA", "-"), ("C8_other", "-"), ("AH", "--"), ("WH", "--")):
        if d not in P:
            continue
        m = P[d].mean(0); se = P[d].std(0) / np.sqrt(len(P[d]))
        nn = f" (n = {len(ev[d])})" if d in ev else ""
        ax.plot(1000 * tc, m, color=C[d], ls=ls, lw=1.0 if d in ("C8", "A") else 0.7, label=LAB[d] + nn)
        ax.fill_between(1000 * tc, m - se, m + se, color=C[d], alpha=0.12, lw=0)
    ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0); ax.axvline(0, color="0.4", lw=0.5, ls=":")
    ax.set_xlabel("Time from lick (ms)"); ax.set_ylabel("Rate - baseline (spikes/s), mean ± s.e.m.")
    ax.legend(frameon=False, fontsize=4.6, ncol=2)
    ax.set_title(f"Lick-aligned population response ({P['A'].shape[0]} good + mua units; grey: pre-lick window)", loc="left", fontsize=6)
    # f: selective units
    ax = fig.add_subplot(gs[2, 2])
    ds = [d for d in ["A", "C1", "C2", "C4", "C8", "C12", "C8_FA", "C8_other"] if f"AH_vs_{d}_sig" in S]
    y = np.arange(len(ds))
    ax.barh(y - 0.18, [100 * S[f"AH_vs_{d}_sig"].mean() for d in ds], 0.36, color=[C[d] for d in ds], label="AH vs SL")
    ax.barh(y + 0.18, [100 * S[f"WH_vs_{d}_sig"].mean() for d in ds], 0.36, color=[C[d] for d in ds], alpha=0.5, label="WH vs SL")
    ax.set_yticks(y, [LAB[d].replace("C, ", "").replace("C8: ", "C8 ") for d in ds], fontsize=4.6); ax.invert_yaxis()
    ax.set_xlabel("Units significant (%)"); ax.legend(frameon=False, fontsize=4.6, loc="lower right")
    ax.set_title("Selective units (dark: AH, light: WH)", loc="left", fontsize=6)
    # g, h: selectivity scatter A vs C8; i: C8_FA vs C8_other
    for j, (c, d1, d2) in enumerate((("AH", "A", "C8"), ("WH", "A", "C8"), ("AH", "C8_other", "C8_FA"))):
        ax = fig.add_subplot(gs[3, j])
        k1, k2 = f"{c}_vs_{d1}", f"{c}_vs_{d2}"
        if k1 in S and k2 in S:
            r = S[k1].corr(S[k2])
            ax.scatter(S[k1], S[k2], s=3, color=C[d2], alpha=0.45, lw=0)
            ax.plot([-1, 1], [-1, 1], color="0.6", lw=0.5, ls="--")
            ax.text(0.03, 0.97, f"r = {r:.2f}, n = {len(S)} units", transform=ax.transAxes, va="top", fontsize=5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xlabel(f"{c} vs SL selectivity, {LAB[d1].split(':')[0] if d1 == 'A' else 'C8 other'}")
        ax.set_ylabel(f"{c} vs SL selectivity, {'C8' if d2 == 'C8' else 'C8 FA-window'}")
        ax.set_title(f"Pre-lick selectivity, {c} vs SL" + ("" if d1 == "A" else " (FA vs other SL)"), loc="left", fontsize=6)
    fig.suptitle(f"Spontaneous licks: exclusion after every stimulus, false-alarm licks included ({sid})", x=0.02, y=0.99, ha="left",
                 fontsize=7, weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"sl_stim_exclusion_{sid}.{ext}", dpi=250)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="AB080_20230622_152205")
    main(ap.parse_args())
