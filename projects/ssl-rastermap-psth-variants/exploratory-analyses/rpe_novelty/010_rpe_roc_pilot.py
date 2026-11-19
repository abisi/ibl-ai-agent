"""Pilot: single-neuron ROC tests for reward-prediction-error (RPE)-like signals, a few day-0 sessions.

Timing (Axel Bisi): the FIRST lick is the action that triggers reward delivery; the reward is contacted at
~the SECOND lick. Windows relative to the first lick L1 (ILI = session median inter-lick interval, piezo):
    pre   [-100, 0] ms                 decision / anticipation
    act   [0, ILI]                     action + delivery (first -> ~second lick)
    rec   [ILI, ILI + 150 ms]          reward receipt (or its omission); no second lick is required
Events (active context):
    WH / AH  whisker / auditory hits, L1 = corrected first lick = start_time + (lick_time - response_window_start)
             (NWB lick_time is late by the artifact window, computed per trial; see 013/014)
    SP       spontaneous lick onsets from <mouse>/whisker_0/spontaneous_licks/<session>_spontaneous_licks.csv
             (single / short_cluster / bout), excluding events that are trial licks or fall inside a trial
             ([start_time - 0.2 s, response_window_stop + 0.5 s])
Rates are baseline-subtracted per event (trials: [-200, 0] ms before stimulus; SP: [-500, -300] ms before L1).
Units: bc_label good + mua. Spike trains get the whisker-artifact correction (roc_utils_new.process_nwb_tables).

ROC per neuron x window: selectivity = 2(AUC - 0.5), + = class 2 higher; two-sided permutation p (1000 shuffles);
labels use uncorrected p < 0.05 (one label per neuron, no multiple-comparison correction).
    WH_vs_SP, AH_vs_SP, WH_vs_AH (pre / act / rec);  WH_early_late, AH_early_late (rec; + = early > late)
Labels (receipt window):
    whisker_pos = WH > SP & WH > AH;  whisker_neg = WH < SP & WH < AH
    R+ (whisker reward initially unexpected): RPE_pos = whisker_pos (burst), RPE_neg = whisker_neg (dip)
    R- (whisker hit = omission):              RPE_pos = whisker_neg (dip),   RPE_neg = whisker_pos (burst)
    *_learning = RPE label whose early-vs-late change matches the prediction (shrinks with learning)
    auditory_pos / auditory_neg = AH vs SP (reward is predicted by the tone in both cohorts)
Outputs -> combined_results_ks4/rpe_roc_pilot/ (previous runs archived in rpe_roc_pilot_v*/).
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru                     # noqa: E402
import rastermap_psth.population_matrix_summary as pms           # noqa: E402

ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi")
NWB_DIR = ROOT / "NWB_ks4"
RES = ROOT / "combined_results_ks4"
OUT = RES / "rpe_roc_pilot"
VARIANT = RES / "rastermap_variants" / "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes"
SESSIONS = ["AB130_20240902_123634", "AB127_20240821_103757", "AB125_20240817_123403",   # R+
            "AB154_20250205_172319", "AB122_20240804_134554", "AB128_20240829_112813"]   # R-
QC = ["good", "mua"]
N_PERM, ALPHA, SEED = 1000, 0.05, 0
MIN_EVENTS = 8
W_PRE, W_REC_LEN = (-0.10, 0.0), 0.15
BASE_TRIAL, BASE_SPONT = (-0.20, 0.0), (-0.50, -0.30)
SP_TRIAL_PAD = (0.2, 0.5)
WINDOWS = ["pre", "act", "rec"]
TESTS = {"WH_vs_SP": ("SP", "WH"), "AH_vs_SP": ("SP", "AH"), "WH_vs_AH": ("AH", "WH")}
COND_COLORS = {"WH": "#d4a017", "AH": "#1f4fbf", "SP": "0.45"}
COND_NAMES = {"WH": "whisker hit", "AH": "auditory hit", "SP": "spontaneous lick"}
LABELS = ["RPE_pos", "RPE_neg", "RPE_pos_learning", "RPE_neg_learning", "auditory_pos", "auditory_neg"]
LABEL_KEY = {"RPE_pos": "WH_vs_SP__rec__sel", "RPE_neg": "WH_vs_SP__rec__sel",
             "auditory_pos": "AH_vs_SP__rec__sel", "auditory_neg": "AH_vs_SP__rec__sel"}


# ── events ─────────────────────────────────────────────────────────────────
def session_events(nwb, trials, session_id, mouse_id):
    ev_ts = nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
    piezo = np.sort(np.asarray(ev_ts.time_series["piezo_lick_times"].data[:]))
    d = np.diff(piezo)
    ili = float(np.median(d[(d > 0.02) & (d < 0.3)]))              # interval only: clock offset irrelevant
    act = trials[trials["context"] == "active"]
    out = {}
    for key, col in (("WH", "whisker_stim"), ("AH", "auditory_stim")):
        t = act[(act[col] == 1) & (act["lick_flag"] == 1) & act["lick_time"].notna()]
        l1 = (t["start_time"] + t["lick_time"] - t["response_window_start_time"]).to_numpy(float)
        out[key] = dict(l1=l1, base_anchor=t["start_time"].to_numpy(float), base=BASE_TRIAL)
    csv = RES / mouse_id / "whisker_0" / "spontaneous_licks" / f"{session_id}_spontaneous_licks.csv"
    sp = pd.read_csv(csv)
    sp = sp[sp["event_type"].isin(["single", "short_cluster", "bout"])]
    l1 = sp["lick_time"].to_numpy(float)
    is_trial_lick = np.isin(np.round(l1, 6), np.round(trials["lick_time"].dropna().to_numpy(float), 6))
    t0 = trials["start_time"].to_numpy(float) - SP_TRIAL_PAD[0]
    t1 = trials["response_window_stop_time"].to_numpy(float) + SP_TRIAL_PAD[1]
    in_trial = np.array([np.any((x >= t0) & (x <= t1)) for x in l1])
    keep = ~is_trial_lick & ~in_trial
    order = np.argsort(l1[keep])
    out["SP"] = dict(l1=l1[keep][order], base_anchor=l1[keep][order], base=BASE_SPONT,
                     sp_type=sp["event_type"].to_numpy()[keep][order])
    excl = dict(n_csv=len(l1), n_trial_lick=int(is_trial_lick.sum()), n_in_trial=int((in_trial & ~is_trial_lick).sum()))
    return out, ili, excl


def rates(st, starts, stops):
    n = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    return n / np.maximum(stops - starts, 1e-3)


def window_rates(st, e, ili):
    b = rates(st, e["base_anchor"] + e["base"][0], e["base_anchor"] + e["base"][1])
    l1 = e["l1"]
    return {"pre": rates(st, l1 + W_PRE[0], l1 + W_PRE[1]) - b,
            "act": rates(st, l1, l1 + ili) - b,
            "rec": rates(st, l1 + ili, l1 + ili + W_REC_LEN) - b}


# ── ROC with a vectorised permutation null ────────────────────────────────
def roc_perm(x1, x2, rng):
    """x1: (n_neurons, n1), x2: (n_neurons, n2). Returns selectivity (2(AUC-.5), + = class 2 higher), p."""
    n1, n2 = x1.shape[1], x2.shape[1]
    r = rankdata(np.concatenate([x1, x2], 1), axis=1)
    auc = (r[:, n1:].sum(1) - n2 * (n2 + 1) / 2) / (n1 * n2)
    n = n1 + n2
    member = np.zeros((n, N_PERM))
    for b in range(N_PERM):
        member[rng.permutation(n)[:n2], b] = 1
    auc_null = (r @ member - n2 * (n2 + 1) / 2) / (n1 * n2)
    dev, dev_null = np.abs(auc - 0.5), np.abs(auc_null - 0.5)
    p = ((dev_null >= dev[:, None] - 1e-12).sum(1) + 1) / (N_PERM + 1)
    return 2 * (auc - 0.5), p


# ── rastermap membership (cluster / family / area; no post-hoc selection) ─
def rastermap_membership():
    md = sorted(VARIANT.glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]
    R = pms.load_rastermap(md)
    meta = pms.metaclustering(md)
    meta_csv = sorted(VARIANT.glob("rastermap_clustering/*/*/*/neuron_metadata.csv"))[0]
    m = pd.read_csv(meta_csv)[["unit_id", "mouse_id", "session_id", "electrode_group", "cluster_id",
                               "area_acronym_custom", "area_group"]]
    df = pd.DataFrame(dict(unit_id=R["unit_ids"], rm_cluster=R["labels"].astype(int)))
    df = df.merge(m, on="unit_id", how="left", validate="one_to_one")
    df["rm_family"] = df["rm_cluster"].map(meta["families"])
    names, _ = pms.family_names_psth(R, meta["families"])
    df["rm_family_name"] = df["rm_family"].map(names)
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["electrode_group"] = df["electrode_group"].astype(str)
    return df.drop(columns="unit_id")


# ── per session ───────────────────────────────────────────────────────────
def run_session(session_id, rm, rng):
    mouse_id = session_id.split("_")[0]
    nwb = NWBHDF5IO(str(NWB_DIR / f"{session_id}.nwb"), "r").read()
    units, trials = ru.process_nwb_tables(nwb)
    bc = nwb.units.to_dataframe()["bc_label"]
    units = units[bc.loc[units["neuron_id"]].isin(QC).to_numpy()].reset_index(drop=True)
    wh_rew = trials.loc[(trials["context"] == "active") & (trials["whisker_stim"] == 1), "reward_available"]
    cohort = "R+" if wh_rew.mean() > 0.5 else "R-"
    ev, ili, excl = session_events(nwb, trials, session_id, mouse_id)
    n_ev = {k: len(v["l1"]) for k, v in ev.items()}
    print(f"{session_id} {cohort}: {len(units)} units, events {n_ev}, ILI {ili * 1000:.0f} ms, "
          f"SP excluded: {excl}", flush=True)
    spikes = [np.sort(np.asarray(s)) for s in units["spike_times"]]
    W = {k: {w: np.stack([window_rates(st, e, ili)[w] for st in spikes]) for w in WINDOWS} for k, e in ev.items()}
    res = units[["electrode_group", "cluster_id", "neuron_id", "firing_rate"]].copy()
    res.insert(0, "session_id", session_id); res.insert(0, "mouse_id", mouse_id)
    res["cohort"] = cohort
    res["bc_label"] = bc.loc[units["neuron_id"]].to_numpy()
    res["ili_ms"] = ili * 1000
    for k in ev:
        res[f"n_{k}"] = n_ev[k]
    for test, (c1, c2) in TESTS.items():
        if min(n_ev[c1], n_ev[c2]) < MIN_EVENTS:
            continue
        for w in WINDOWS:
            sel, p = roc_perm(W[c1][w], W[c2][w], rng)
            res[f"{test}__{w}__sel"], res[f"{test}__{w}__p"] = sel, p
    for k in ("WH", "AH"):                                        # early vs late (+ = early > late)
        if n_ev[k] >= 2 * MIN_EVENTS:
            h = n_ev[k] // 2
            sel, p = roc_perm(W[k]["rec"][:, h:], W[k]["rec"][:, :h], rng)
            res[f"{k}_early_late__rec__sel"], res[f"{k}_early_late__rec__p"] = sel, p
    res["cluster_id"] = res["cluster_id"].astype(int)
    res["electrode_group"] = res["electrode_group"].astype(str)
    res = res.merge(rm, on=["mouse_id", "session_id", "electrode_group", "cluster_id"], how="left",
                    validate="one_to_one")
    res["in_rastermap"] = res["rm_cluster"].notna()
    return res, ev, spikes, ili, excl


def label_neurons(df):
    def sig(c, sign):
        p = df.get(f"{c}__p", pd.Series(np.nan, df.index)); s = df.get(f"{c}__sel", pd.Series(np.nan, df.index))
        return (p < ALPHA) & (np.sign(s) == sign)
    rp, rm_ = df["cohort"] == "R+", df["cohort"] == "R-"
    w_pos = sig("WH_vs_SP__rec", 1) & sig("WH_vs_AH__rec", 1)
    w_neg = sig("WH_vs_SP__rec", -1) & sig("WH_vs_AH__rec", -1)
    df["whisker_pos"], df["whisker_neg"] = w_pos, w_neg
    df["RPE_pos"] = (rp & w_pos) | (rm_ & w_neg)
    df["RPE_neg"] = (rp & w_neg) | (rm_ & w_pos)
    # learning: the whisker response shrinks toward baseline -> early > late for bursts, early < late for dips
    df["RPE_pos_learning"] = df["RPE_pos"] & ((rp & sig("WH_early_late__rec", 1)) | (rm_ & sig("WH_early_late__rec", -1)))
    df["RPE_neg_learning"] = df["RPE_neg"] & ((rp & sig("WH_early_late__rec", -1)) | (rm_ & sig("WH_early_late__rec", 1)))
    df["auditory_pos"], df["auditory_neg"] = sig("AH_vs_SP__rec", 1), sig("AH_vs_SP__rec", -1)
    return df


# ── figures ───────────────────────────────────────────────────────────────
def fig_fractions(df, out):
    tests = [c[:-5] for c in df.columns if c.endswith("__sel")]
    sess = list(dict.fromkeys(df["session_id"]))
    fig, axes = plt.subplots(len(sess), 1, figsize=(11, 2.2 * len(sess)), sharex=True, dpi=200)
    for ax, s in zip(np.atleast_1d(axes), sess):
        d = df[df["session_id"] == s]
        pos = [((d[f"{t}__p"] < ALPHA) & (d[f"{t}__sel"] > 0)).mean() for t in tests]
        neg = [((d[f"{t}__p"] < ALPHA) & (d[f"{t}__sel"] < 0)).mean() for t in tests]
        x = np.arange(len(tests))
        ax.bar(x - 0.2, pos, 0.4, color="#c0392b", label="class 2 > class 1")
        ax.bar(x + 0.2, neg, 0.4, color="#2471a3", label="class 2 < class 1")
        ax.axhline(ALPHA / 2, color="k", ls=":", lw=0.8, label="chance per direction")
        ax.set_ylabel("fraction"); ax.set_title(f"{s} ({d['cohort'].iat[0]}, {len(d)} units)", fontsize=9)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    ax.set_xticks(np.arange(len(tests))); ax.set_xticklabels(tests, rotation=40, ha="right", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(out / "rpe_roc_fractions.png"); fig.savefig(out / "rpe_roc_fractions.pdf")
    plt.close(fig)


def psth(st, t0, win=(-0.4, 0.6), bw=0.005, sm=2):
    edges = np.arange(win[0], win[1] + bw / 2, bw)
    h = np.stack([np.histogram(st[np.searchsorted(st, t + win[0]):np.searchsorted(st, t + win[1])] - t, edges)[0]
                  for t in t0]) / bw if len(t0) else np.full((1, len(edges) - 1), np.nan)
    return edges[:-1] + bw / 2, gaussian_filter1d(h.mean(0), sm), gaussian_filter1d(h.std(0) / np.sqrt(len(h)), sm)


def fig_examples(df_s, ev, spikes, ili, out, label, n_max=8):
    d = df_s[df_s[label]].copy()
    if d.empty:
        return 0
    key = LABEL_KEY.get(label.replace("_learning", ""), "WH_vs_SP__rec__sel")
    d = d.reindex(d[key].abs().sort_values(ascending=False).index).head(n_max)
    ncol = 2
    nrow = int(np.ceil(len(d) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 1.9 * nrow), dpi=200, squeeze=False)
    for n, (i, row) in enumerate(d.iterrows()):
        ax = axes[n // ncol, n % ncol]
        for k in ("WH", "AH", "SP"):
            t, mu, se = psth(spikes[i], ev[k]["l1"])
            ax.fill_between(t, mu - se, mu + se, color=COND_COLORS[k], alpha=0.2, lw=0)
            ax.plot(t, mu, color=COND_COLORS[k], lw=1, label=f"{COND_NAMES[k]} (n={len(ev[k]['l1'])})")
        ax.axvspan(*W_PRE, color="0.92", zorder=0, lw=0)
        ax.axvspan(ili, ili + W_REC_LEN, color="#fdebd0", zorder=0, lw=0)
        ax.axvline(0, color="k", lw=0.5, ls=":"); ax.axvline(ili, color="0.5", lw=0.5, ls="--")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(labelsize=6)
        rmtxt = (f"RM cl{int(row['rm_cluster'])} MC{int(row['rm_family'])} {row['rm_family_name']}"
                 if row["in_rastermap"] else "not in rastermap")
        ax.set_title(f"{row['electrode_group']} c{int(row['cluster_id'])} {row.get('area_acronym_custom', '')}  "
                     f"WH-SP {row.get('WH_vs_SP__rec__sel', np.nan):+.2f} WH-AH {row.get('WH_vs_AH__rec__sel', np.nan):+.2f} "
                     f"AH-SP {row.get('AH_vs_SP__rec__sel', np.nan):+.2f}\n{rmtxt}", fontsize=5.5, loc="left")
    for n in range(len(d), nrow * ncol):
        axes[n // ncol, n % ncol].axis("off")
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="upper left")
    for a in axes[-1]:
        a.set_xlabel("time from corrected first lick (s)", fontsize=6.5)
    s = df_s["session_id"].iat[0]
    fig.suptitle(f"{s} ({df_s['cohort'].iat[0]}): '{label}' neurons (top {len(d)} by |{key.split('__')[0]}|, receipt "
                 f"window shaded; dashed = 1 inter-lick interval)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"examples_{label}_{s}.png"); fig.savefig(out / f"examples_{label}_{s}.pdf")
    plt.close(fig)
    return len(d)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rm = rastermap_membership()
    all_res, cache, excl_all = [], {}, {}
    for s in SESSIONS:
        res, ev, spikes, ili, excl = run_session(s, rm, rng)
        all_res.append(res); cache[s] = (ev, spikes, ili); excl_all[s] = excl
    df = label_neurons(pd.concat(all_res, ignore_index=True))
    df.to_csv(OUT / "rpe_roc_units.csv", index=False)
    fig_fractions(df, OUT)
    for s, (ev, spikes, ili) in cache.items():
        d = df[df["session_id"] == s].reset_index(drop=True)
        for lab in ("RPE_pos", "RPE_neg", "auditory_pos"):
            fig_examples(d, ev, spikes, ili, OUT, lab)
    summ = df.groupby(["session_id", "cohort"]).agg(n_units=("neuron_id", "size"),
                                                   **{c: (c, "sum") for c in LABELS}).reset_index()
    frac = df.groupby(["session_id", "cohort"])[LABELS].mean().round(4).reset_index()
    summ.to_csv(OUT / "rpe_roc_summary_counts.csv", index=False)
    frac.to_csv(OUT / "rpe_roc_summary_fractions.csv", index=False)
    by_area = df[df["in_rastermap"]].groupby(["cohort", "area_group"]).agg(
        n_units=("neuron_id", "size"), **{c: (c, "mean") for c in LABELS}).reset_index()
    by_area.to_csv(OUT / "rpe_roc_by_area_group.csv", index=False)
    json.dump(dict(sessions=SESSIONS, qc=QC, n_perm=N_PERM, alpha=ALPHA, alpha_correction="none", seed=SEED,
                   first_lick="start_time + (lick_time - response_window_start_time)",
                   spontaneous="CSV single/short_cluster/bout, excluding trial licks and events within "
                               f"[start_time - {SP_TRIAL_PAD[0]}, response_window_stop + {SP_TRIAL_PAD[1]}]",
                   windows=dict(pre=W_PRE, act="[0, ILI]", rec=f"[ILI, ILI + {W_REC_LEN}]", base_trial=BASE_TRIAL,
                                base_spont=BASE_SPONT), tests=TESTS, sp_exclusions=excl_all,
                   rastermap_variant=str(VARIANT), min_events=MIN_EVENTS), open(OUT / "config.json", "w"), indent=2)
    print(summ.to_string(index=False))
    print(frac.to_string(index=False))
    print(by_area[by_area["n_units"] >= 30].round(3).to_string(index=False))
    print("ALL DONE", flush=True)
