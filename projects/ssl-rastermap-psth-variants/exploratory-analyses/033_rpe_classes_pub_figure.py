"""Publication figure: fraction and mean activity of the RPE-v2 neuron classes, R+ vs R- (learning day, good units).

Classes are defined with the SAME test in both cohorts (uncorrected p < .05, 029_roc_rpe_v2 results):
  novelty            first 5 whisker trials: cue (50-150 ms) > same-trial baseline               (C1 up)
  outcome response   first third of whisker licks: outcome (30-300 ms after first lick) > baseline (O1 R+ / O4 R- up)
  up & decreasing    outcome response AND whisker-lick T1 > T3        (positive-RPE-like; O1&O2 R+ / O4&O5 R-)
  down & recovering  outcome < baseline AND T1 < T3                    (negative-RPE-like)
  cue increase       whisker cue T3 > T1                               (C2 R+ / C3 R-)
  cue decrease       whisker cue T3 < T1
Fraction panel: one point per session (= mouse on day 0), mean +- SEM; dashed = chance (identical pipeline on
shuffled labels); R+ vs R- on (fraction - chance): Mann-Whitney U AND Welch t (both reported).
Activity panels: firing rate z-scored per neuron by its pre-trial rate ([-310, -10] ms, all whisker trials; SD floor
1 Hz), averaged per session, then mean +- SEM across sessions.
Steps: `compute` (loads NWB, caches per-neuron PSTHs) then `plot` (fast).
Output: <summary>/pub/rpe_classes_pub.{png,pdf,svg}, rpe_classes_pub_stats.csv, rpe_classes_pub_legend.md
"""
import argparse
import importlib
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m = importlib.import_module("029_roc_rpe_v2")
COH = {"R+": "#00B400", "R-": "#C800C8"}
BW = 0.01
STIM_BINS = np.arange(-0.3, 0.6001, BW)
LICK_BINS = np.arange(-0.5, 0.8001, BW)
STIM_SETS = {"First 5": "W_first", "Next": "W_next", "T1": "W_all_T1", "T2": "W_all_T2", "T3": "W_all_T3"}
LICK_SETS = {"Whisker T1": "W_lick_T1", "Whisker T2": "W_lick_T2", "Whisker T3": "W_lick_T3",
             "Auditory H1": "A_hit_H1", "Auditory H2": "A_hit_H2"}
CLASSES = [  # key, label, alignment, curves to plot, shaded window
    ("novelty", "Novelty response", "stim", ["First 5", "Next", "T3"], (0.05, 0.15)),
    ("outcome", "Outcome response", "lick", ["Whisker T1", "Auditory H1"], (0.03, 0.30)),
    ("up_dec", "Up & decreasing\n(positive-RPE-like)", "lick", ["Whisker T1", "Whisker T2", "Whisker T3", "Auditory H1", "Auditory H2"], (0.03, 0.30)),
    ("down_rec", "Down & recovering\n(negative-RPE-like)", "lick", ["Whisker T1", "Whisker T2", "Whisker T3", "Auditory H1", "Auditory H2"], (0.03, 0.30)),
    ("cue_up", "Cue increase\n(T3 > T1)", "stim", ["T1", "T2", "T3"], (0.05, 0.15)),
    ("cue_down", "Cue decrease\n(T3 < T1)", "stim", ["T1", "T2", "T3"], (0.05, 0.15)),
]
CURVE_COL = {"First 5": "#E08214", "Next": "#8C8C8C", "T1": "#A6DBA0", "T2": "#5AAE61", "T3": "#1B7837",
             "Whisker T1": "#A6DBA0", "Whisker T2": "#5AAE61", "Whisker T3": "#1B7837",
             "Auditory H1": "#92C5DE", "Auditory H2": "#2166AC"}
CURVE_LS = {"T3": "-", "Whisker T3": "-"}


def class_masks(d, suf=""):
    rp = (d.cohort == "R+").to_numpy()

    def sig(a, b, sign):
        p = np.where(rp, d[f"p{suf}_{a}"], d[f"p{suf}_{b}"]); auc = np.where(rp, d[f"auc{suf}_{a}"], d[f"auc{suf}_{b}"])
        return (p < .05) & (np.sign(auc - .5) == sign)
    out = {"novelty": sig("C1", "C1", 1), "outcome": sig("O1", "O4", 1),
           "cue_up": sig("C2", "C3", 1), "cue_down": sig("C2", "C3", -1)}
    # T1 vs T3 with trials matched on lick count in the outcome window (034_rpe_lick_control), same test in both cohorts
    out["up_dec"] = out["outcome"] & sig("T1vsT3_matched", "T1vsT3_matched", 1)
    out["down_rec"] = sig("O1", "O4", -1) & sig("T1vsT3_matched", "T1vsT3_matched", -1)
    return pd.DataFrame(out, index=d.index)


def load_units(summ):
    d = pd.read_parquet(summ / "rpe_v2_neurons.parquet")
    d = d[(d.stage == "learning") & (d.quality_label == "good")].reset_index(drop=True)
    assert len(d) and (d.quality_label == "good").all()
    lc = pd.read_parquet(summ / "lick_control" / "lick_control_neurons.parquet")
    keep = ["session_id", "neuron_id"] + [c for c in lc.columns if c.endswith("T1vsT3_matched")]
    d = d.merge(lc[keep], on=["session_id", "neuron_id"], how="left", validate="one_to_one")
    return d


def psth(st, t0, bins):
    if len(t0) == 0:
        return np.full(len(bins) - 1, np.nan)
    rel = np.concatenate([st[(st >= t + bins[0]) & (st < t + bins[-1])] - t for t in t0])
    return np.histogram(rel, bins)[0] / len(t0) / BW


def worker(args):
    sid, nids = args
    nwb, units, act = m.load_session(sid)
    _, S = m.sets_for(act)
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    lick = dict(session_id=sid, **{f"lick|{k}": psth(pz, S[v].L1.to_numpy(float), LICK_BINS).tolist()
                                   for k, v in LICK_SETS.items()})
    units = units[units.neuron_id.isin(nids)]
    t_all = S["W_all"].start_time.to_numpy(float)
    rows = []
    for nid, s in zip(units.neuron_id, units.spike_times):
        st = np.sort(np.asarray(s))
        b = np.array([np.sum((st >= t - 0.31) & (st < t - 0.01)) / 0.30 for t in t_all])
        mu, sd = b.mean(), max(b.std(), 1.0)
        r = dict(session_id=sid, neuron_id=nid)
        for k, v in STIM_SETS.items():
            r[f"stim|{k}"] = (psth(st, S[v].start_time.to_numpy(float), STIM_BINS) - mu) / sd
        for k, v in LICK_SETS.items():
            r[f"lick|{k}"] = (psth(st, S[v].L1.to_numpy(float), LICK_BINS) - mu) / sd
        rows.append(r)
    return rows, lick


def cmd_compute(a):
    summ = m.SUMMARY / a.tag; (summ / "pub").mkdir(exist_ok=True)
    d = load_units(summ)
    cm = class_masks(d)
    sel = d[cm.any(axis=1)]
    jobs = [(sid, set(g.neuron_id)) for sid, g in sel.groupby("session_id")]
    rows, licks = [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (r, lk) in enumerate(pool.map(worker, jobs)):
            rows += r; licks.append(lk)
            print(f"done {i + 1}/{len(jobs)}", flush=True)
    P = pd.DataFrame(rows)
    for c in P.columns[2:]:
        P[c] = P[c].apply(lambda x: np.asarray(x, float).tolist())
    P.to_parquet(summ / "pub" / "class_psth_cache.parquet", index=False)
    pd.DataFrame(licks).to_parquet(summ / "pub" / "class_lick_cache.parquet", index=False)
    print("cached", len(P), "neurons", flush=True)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "font.family": "sans-serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    summ = m.SUMMARY / a.tag; out = summ / "pub"
    d = load_units(summ)
    cm = class_masks(d); cn = class_masks(d, "_null")
    P = pd.read_parquet(out / "class_psth_cache.parquet").merge(d[["session_id", "neuron_id", "cohort"]], on=["session_id", "neuron_id"])
    cmP = cm.copy(); cmP[["session_id", "neuron_id"]] = d[["session_id", "neuron_id"]]
    P = P.merge(cmP, on=["session_id", "neuron_id"])
    Lk = pd.read_parquet(out / "class_lick_cache.parquet")
    n_rows = len(CLASSES) + 1                                             # + lick-controlled outcome row
    hr = [1.35 if c[2] == "lick" else 1.0 for c in CLASSES] + [1.35]
    fig = plt.figure(figsize=(7.2, 11.6))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[0.8, 1.25, 1.25], height_ratios=hr, hspace=0.62, wspace=0.42,
                          left=0.135, right=0.845, top=0.965, bottom=0.04)
    stat_rows, ylims = [], {}
    rng = np.random.default_rng(1)
    for i, (key, label, align, curves, win) in enumerate(CLASSES):
        # ---- fraction panel
        ax = fig.add_subplot(gs[i, 0])
        sess = pd.DataFrame({"session_id": d.session_id, "cohort": d.cohort, "f": cm[key], "c": cn[key]}) \
            .groupby(["session_id", "cohort"]).mean().reset_index()
        vals = {}
        for k, coh in enumerate(COH):
            s = sess[sess.cohort == coh]
            v = s.f.to_numpy() * 100; ch = s.c.mean() * 100; vals[coh] = (s.f - s.c).to_numpy()
            ax.scatter(k + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=COH[coh], alpha=0.45, lw=0, zorder=2)
            ax.errorbar(k, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=4, color="k", mfc=COH[coh],
                        mew=0.6, capsize=2.5, elinewidth=0.8, zorder=3)
            ax.plot([k - 0.3, k + 0.3], [ch, ch], ls="--", color="0.35", lw=0.8, zorder=1)
        pm = stats.mannwhitneyu(vals["R+"], vals["R-"], alternative="two-sided").pvalue
        pw = stats.ttest_ind(vals["R+"], vals["R-"], equal_var=False).pvalue
        top = max(sess.f.max() * 100, 1e-3)
        ax.set_ylim(-0.03 * top, top * 1.28)
        ax.plot([0, 0, 1, 1], [top * 1.08, top * 1.12, top * 1.12, top * 1.08], color="k", lw=0.6)
        ax.text(0.5, top * 1.14, f"MWU {fmt_p(pm)}\nWelch {fmt_p(pw)}", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks([0, 1], [f"R+\n(n={int((sess.cohort == 'R+').sum())})", f"R−\n(n={int((sess.cohort == 'R-').sum())})"])
        ax.set_xlim(-0.55, 1.55); ax.set_ylabel("% of neurons")
        pos = ax.get_position()
        fig.text(0.004, pos.y1 + 0.012, "abcdefgh"[i], ha="left", va="bottom", fontsize=9, weight="bold")
        fig.text(0.03, (pos.y0 + pos.y1) / 2, label, rotation=90, ha="center", va="center", fontsize=7,
                 weight="bold", linespacing=1.1, multialignment="center")
        for coh in COH:
            s = sess[sess.cohort == coh]
            stat_rows.append(dict(cls=key, cohort=coh, n_sessions=len(s), n_neurons=int(cm[key][d.cohort == coh].sum()),
                                  n_units=int((d.cohort == coh).sum()), pct_pooled=100 * cm[key][d.cohort == coh].mean(),
                                  pct_session_mean=100 * s.f.mean(), pct_session_sem=100 * s.f.std(ddof=1) / np.sqrt(len(s)),
                                  pct_chance=100 * s.c.mean(),
                                  p_vs_chance_wilcoxon=stats.wilcoxon(s.f - s.c).pvalue,
                                  p_vs_chance_ttest=stats.ttest_1samp(s.f - s.c, 0).pvalue,
                                  p_Rplus_vs_Rminus_MWU=pm, p_Rplus_vs_Rminus_Welch=pw))
        # ---- activity panels
        bins = STIM_BINS if align == "stim" else LICK_BINS
        tc = (bins[:-1] + BW / 2) * 1e3
        axes = []
        for j, coh in enumerate(COH):
            if align == "lick":
                sub = gs[i, 1 + j].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
                ax, axl = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            else:
                ax, axl = fig.add_subplot(gs[i, 1 + j]), None
            axes.append(ax)
            g = P[(P.cohort == coh) & P[key]]
            if axl is not None:                                           # lick rate of the same sessions / trial sets
                lk = Lk[Lk.session_id.isin(g.session_id.unique())]
                for cname in curves:
                    arr = np.stack(lk[f"lick|{cname}"].map(np.asarray).to_list()) if len(lk) else None
                    if arr is not None:
                        axl.plot(tc, np.convolve(np.nanmean(arr, 0), np.ones(3) / 3, "same"), color=CURVE_COL[cname], lw=0.7)
                axl.axvspan(win[0] * 1e3, win[1] * 1e3, color="#FDD49E", alpha=0.45, lw=0); axl.axvline(0, color="k", lw=0.5)
                axl.set_xlim(tc[0], tc[-1]); axl.set_xticks([-400, 0, 400]); axl.set_ylim(0, 12)
                axl.set_ylabel("Licks/s", fontsize=6); axl.tick_params(labelsize=5.5)
                ax.tick_params(labelbottom=False)
                axl.set_xlabel("Time from first lick (ms)", fontsize=6.5)
            ax.axvspan(win[0] * 1e3, win[1] * 1e3, color="#FDD49E", alpha=0.45, lw=0, zorder=0)
            ax.axvline(0, color="k", lw=0.5, zorder=1); ax.axhline(0, color="0.6", lw=0.4, zorder=1)
            for cname in curves:
                col = f"{align}|{cname}"
                sm = g.groupby("session_id")[col].apply(lambda s: np.nanmean(np.stack(s.map(np.asarray).to_list()), 0))
                if not len(sm):
                    continue
                arr = np.stack(sm.to_list())
                arr = np.apply_along_axis(lambda x: np.convolve(np.nan_to_num(x), np.ones(3) / 3, "same"), 1, arr)
                mu = np.nanmean(arr, 0); se = np.nanstd(arr, 0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * mu
                ax.fill_between(tc, mu - se, mu + se, color=CURVE_COL[cname], alpha=0.25, lw=0)
                ax.plot(tc, mu, color=CURVE_COL[cname], lw=1.0, label=cname)
            ax.set_xlim(tc[0], tc[-1])
            ax.set_xticks([-200, 0, 200, 400] if align == "stim" else [-400, 0, 400])
            ax.set_title(f"{coh}  ·  {len(g)} neurons, {g.session_id.nunique()} sessions", color=COH[coh], loc="left", pad=2)
            if align == "stim":
                ax.set_xlabel("Time from whisker onset (ms)")
            if j == 0:
                ax.set_ylabel("Firing rate (z)")
            if j == 1:
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.4, borderaxespad=0.0)
        lo = min(a_.get_ylim()[0] for a_ in axes); hi = max(a_.get_ylim()[1] for a_ in axes)
        for a_ in axes:
            a_.set_ylim(lo, hi)
    # ---- row g: outcome response at equal licking (034_rpe_lick_control)
    i = len(CLASSES)
    Rn = pd.read_parquet(summ / "lick_control" / "lick_control_neurons.parquet")
    Rn = Rn.merge(d[["session_id", "neuron_id"]], on=["session_id", "neuron_id"])       # good units only
    Pp = pd.read_parquet(summ / "lick_control" / "lick_control_psth.parquet")
    ax = fig.add_subplot(gs[i, 0])

    def frac(g, name, null=False):
        s = "_null" if null else ""
        v = g.dropna(subset=[f"p{s}_{name}"])
        return ((v[f"p{s}_{name}"] < .05) & (v[f"auc{s}_{name}"] > .5)).mean() if len(v) else np.nan
    for k in range(3):
        name = f"out_W_lick_T1_k{k}"
        per = Rn.groupby(["session_id", "cohort"]).apply(lambda g: pd.Series(dict(f=frac(g, name), c=frac(g, name, True)))) \
            .reset_index().dropna(subset=["f"])
        vals = {}
        for j, coh in enumerate(COH):
            v = per[per.cohort == coh]; x = k + (j - 0.5) * 0.36; vals[coh] = (v.f - v.c).to_numpy()
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), 100 * v.f, s=4, color=COH[coh], alpha=0.4, lw=0)
            ax.errorbar(x, 100 * v.f.mean(), 100 * v.f.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=3.5, mfc=COH[coh],
                        color="k", mew=0.5, capsize=2, elinewidth=0.8)
            ax.plot([x - 0.14, x + 0.14], [100 * v.c.mean()] * 2, "--", color="0.35", lw=0.7)
        pm = stats.mannwhitneyu(vals["R+"], vals["R-"]).pvalue; pw = stats.ttest_ind(vals["R+"], vals["R-"], equal_var=False).pvalue
        sp_ = lambda q: "<0.001" if q < 0.001 else f"{q:.3f}" if q < 0.01 else f"{q:.2f}"      # noqa: E731
        ax.text(k, 47, f"{sp_(pm)}\n{sp_(pw)}", ha="center", fontsize=5.5)
        for coh in COH:
            v = per[per.cohort == coh]
            stat_rows.append(dict(cls=f"outcome_at_{k}_licks", cohort=coh, n_sessions=len(v),
                                  pct_session_mean=100 * v.f.mean(), pct_session_sem=100 * v.f.std(ddof=1) / np.sqrt(len(v)),
                                  pct_chance=100 * v.c.mean(), p_Rplus_vs_Rminus_MWU=pm, p_Rplus_vs_Rminus_Welch=pw))
    ax.set_ylim(-2, 58); ax.set_xticks(range(3), ["0", "1", "2"]); ax.set_xlim(-0.55, 2.55)
    ax.set_xlabel("Licks 30-300 ms after first lick"); ax.set_ylabel("% of neurons")
    ax.text(-0.5, 56, "p: MWU (top) / Welch (bottom)", fontsize=5, color="0.3")
    pos = ax.get_position()
    fig.text(0.004, pos.y1 + 0.012, "g", ha="left", va="bottom", fontsize=9, weight="bold")
    fig.text(0.03, (pos.y0 + pos.y1) / 2, "Outcome response\nat equal licking", rotation=90, ha="center", va="center",
             fontsize=7, weight="bold", linespacing=1.1, multialignment="center")
    tc = (LICK_BINS[:-1] + BW / 2) * 1e3
    axes = []
    for j, k in enumerate([1, 2]):
        sub = gs[i, 1 + j].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax, axl = fig.add_subplot(sub[0]), fig.add_subplot(sub[1]); axes.append(ax)
        for coh in COH:
            for src, ls in [("whisker", "-"), ("auditory", ":")]:
                g = Pp[(Pp.cohort == coh) & (Pp.src == src) & (Pp.k == k)]
                if not len(g):
                    continue
                z = np.stack(g.z.map(np.asarray).to_list()); lk = np.stack(g.lick.map(np.asarray).to_list())
                z = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, "same"), 1, z)
                mu, se = z.mean(0), z.std(0, ddof=1) / np.sqrt(len(z))
                ax.fill_between(tc, mu - se, mu + se, color=COH[coh], alpha=0.18, lw=0)
                rew = "rewarded" if (src == "auditory" or coh == "R+") else "unrewarded"
                ax.plot(tc, mu, color=COH[coh], ls=ls, lw=1.0,
                        label=f"{coh.replace('-', chr(8722))} {src}\n({rew})")
                axl.plot(tc, np.convolve(lk.mean(0), np.ones(3) / 3, "same"), color=COH[coh], ls=ls, lw=0.7)
        for a_ in (ax, axl):
            a_.axvspan(30, 300, color="#FDD49E", alpha=0.45, lw=0); a_.axvline(0, color="k", lw=0.5)
            a_.set_xlim(tc[0], tc[-1]); a_.set_xticks([-400, 0, 400])
        ax.axhline(0, color="0.6", lw=0.4); ax.tick_params(labelbottom=False)
        axl.set_ylim(0, 12); axl.set_ylabel("Licks/s", fontsize=6); axl.tick_params(labelsize=5.5)
        axl.set_xlabel("Time from first lick (ms)", fontsize=6.5)
        ax.set_title(f"All good units · {k} lick{'s' if k > 1 else ''} in window", loc="left", pad=2)
        if j == 0:
            ax.set_ylabel("Firing rate (z)")
        if j == 1:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.6, borderaxespad=0.0)
    lo = min(a_.get_ylim()[0] for a_ in axes); hi = max(a_.get_ylim()[1] for a_ in axes)
    for a_ in axes:
        a_.set_ylim(lo, hi)
    fig.text(0.135, 0.982, "Fraction of neurons", fontsize=8, weight="bold")
    fig.text(0.40, 0.982, "Mean activity (z vs pre-trial rate; mean ± SEM across sessions)",
             fontsize=8, weight="bold")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"rpe_classes_pub.{ext}", dpi=300)
    S = pd.DataFrame(stat_rows); S.to_csv(out / "rpe_classes_pub_stats.csv", index=False)
    legend = f"""# Figure: neuron classes during whisker learning (R+ vs R−)

Learning day (day 0), {d.session_id.nunique()} sessions (one per mouse; R+ {int((sess.cohort == 'R+').sum())},
R− {int((sess.cohort == 'R-').sum())}), {len(d)} units with quality_label = good
(R+ {int((d.cohort == 'R+').sum())}, R− {int((d.cohort == 'R-').sum())}).

**Classes** (same ROC test in both cohorts; label-permutation p < 0.05, uncorrected, predicted direction):
novelty = cue response (50-150 ms after whisker onset) to the first 5 whisker trials > same-trial baseline;
outcome response = response 30-300 ms after the first lick on the first third of whisker-lick trials > baseline;
up & decreasing = outcome response and first third > last third of whisker-lick trials;
down & recovering = outcome < baseline and first third < last third;
for both, first vs last third is compared on trials matched exactly on the number of licks in the outcome window
(0, 1, 2, 3+; random subsampling within each count);
cue increase / decrease = whisker cue response in the last vs first third of whisker trials.
Baselines are pre-trial windows from the same trial, matched in length to the tested window.
Lick-aligned panels show the lick rate (piezo) of the same sessions and trial sets below the activity.

**Row g, licking control:** left, percentage of neurons with an outcome response (> baseline) computed on first-third
whisker-lick trials with exactly 0, 1 or 2 licks in the outcome window (sessions with >= 5 such trials); R+ vs R−
on (fraction − chance), Mann-Whitney U / Welch t. Middle/right, mean activity of ALL good units (no selection) on
trials with exactly 1 or 2 licks in the window: whisker licks (solid; rewarded in R+, unrewarded in R−) and
auditory hits (dotted; rewarded in both cohorts); lick rate of the same trials below.

**Left:** percentage of neurons per session (dots), mean ± SEM (black); dashed line, chance (same pipeline with
shuffled trial labels). R+ vs R− on (fraction − chance): two-sided Mann-Whitney U and Welch t-test.
**Middle/right:** mean firing rate of the class, z-scored to each neuron's pre-trial rate (SD floor 1 Hz), averaged
per session then mean ± SEM across sessions; shading, tested window (cue 50-150 ms after whisker onset; outcome
30-300 ms after the first lick). Whisker T1-T3, thirds of the session's whisker trials (lick-aligned panels:
thirds of whisker-lick trials; R+ rewarded, R− unrewarded); auditory H1/H2, halves of auditory hits (rewarded
in both cohorts); First 5 / Next, first 5 whisker trials / following whisker trials of the first third.
Stats per class: rpe_classes_pub_stats.csv.
"""
    (out / "rpe_classes_pub_legend.md").write_text(legend, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(S.round(4).to_string()); print("saved", out / "rpe_classes_pub.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compute"); c.add_argument("--tag", default="learning"); c.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("plot"); p.add_argument("--tag", default="learning")
    a = ap.parse_args()
    {"compute": cmd_compute, "plot": cmd_plot}[a.cmd](a)
