"""Debug + example figures for the RPE ROC (027_roc_rpe.py).

Per debug session (default: one R+ learner, one R-):
  fig A  timing:   piezo-lick raster aligned to the corrected first lick L1 per trial type (stim onset + baseline window
                   drawn per trial), lick-rate PSTH, L1 -> nearest piezo lick offset (alignment check), 2nd-lick latency
                   vs the ILI-based outcome window, licks inside the outcome window per type (motor confound),
                   whisker learning curve with learning trial and tercile boundaries.
  fig B  implementation: independent re-implementation (explicit loop rates + sklearn AUC) vs the stored CSV,
                   permutation null of one neuron, p-value histograms per comparison, NULL CALIBRATION
                   (pre-trial [-0.2, 0] vs [-0.4, -0.2] on the same trials: must give ~5% per direction).
Across all R+ learning sessions: example neurons (raster + licks + PSTH aligned to L1, grouped by auditory hits,
whisker hits in time terciles, false alarms; tercile means of baseline-subtracted outcome rate) for
  RPE candidates (whisker reward up, unexpected > expected, early > late, all significant), lick-like neurons
  (reward up AND FA up), dip neurons (FA down).
Output: combined_results_ks4/_roc_rpe_summary/debug/
"""
import argparse
import importlib
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from pynwb import NWBHDF5IO                                              # noqa: E402
from sklearn.metrics import roc_auc_score                                # noqa: E402

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
r27 = importlib.import_module("027_roc_rpe")
ru, RES, NWB = r27.ru, r27.RES, r27.NWB
OUT = RES / "_roc_rpe_summary" / "debug"
COL = {"auditory hit": "#1f77b4", "whisker hit": "#2ca02c", "false alarm": "#d62728", "R- whisker hit": "#9467bd",
       "W T1": "#98df8a", "W T2": "#2ca02c", "W T3": "#0b4d0b"}


def load(sid, day=0):
    mouse = sid.split("_")[0]
    nwb = NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r").read()
    units, _ = ru.process_nwb_tables(nwb)
    t = nwb.trials.to_dataframe()
    ctx = t["context"].astype(str).str.strip().str.lower()
    t["context"] = "active" if ctx.isin(["nan", "none", ""]).all() else ctx
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    d = np.diff(pz); ili = float(np.median(d[(d > 0.02) & (d < 0.3)]))
    act = t[t["context"] == "active"].copy()
    act["L1"] = act["start_time"] + act["lick_time"] - act["response_window_start_time"]
    wh_all = act[act.whisker_stim == 1]
    cohort = "R+" if wh_all.reward_available.mean() > 0.5 else "R-"
    lick = act.lick_flag == 1
    groups = {"auditory hit": act[(act.auditory_stim == 1) & lick],
              ("whisker hit" if cohort == "R+" else "R- whisker hit"): act[(act.whisker_stim == 1) & lick],
              "false alarm": act[(act.no_stim == 1) & lick]}
    lt_time, how = r27.learning_split_time(mouse, day, np.sort(wh_all.start_time.to_numpy()))
    acr = units["ccf_acronym"].astype(str).to_numpy() if "ccf_acronym" in units else np.array(["?"] * len(units))
    return dict(sid=sid, mouse=mouse, units=units.reset_index(drop=True), acr=acr, act=act, wh_all=wh_all, pz=pz,
                ili=ili, cohort=cohort, groups=groups, split_time=lt_time, split_how=how,
                spikes=[np.sort(np.asarray(s)) for s in units["spike_times"]])


def rel_events(ev, t0s, w=(-0.6, 1.2)):
    out = []
    for t0 in t0s:
        a, b = np.searchsorted(ev, [t0 + w[0], t0 + w[1]])
        out.append(ev[a:b] - t0)
    return out


def loop_rate(st, a, b):
    return np.array([np.sum((st >= x) & (st < y)) / (y - x) for x, y in zip(a, b)])


# ------------------------------------------------------------------ fig A: timing
def fig_timing(S):
    g = S["groups"]; ili = S["ili"]
    fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[:2, :2]); y = 0
    for name, tr in g.items():
        rel = rel_events(S["pz"], tr.L1.to_numpy())
        stim = (tr.start_time - tr.L1).to_numpy()
        for k, r in enumerate(rel):
            ax.plot(r, np.full(len(r), y + k), "|", ms=3, color=COL[name])
            ax.plot([stim[k] - 0.2, stim[k]], [y + k, y + k], color="0.75", lw=1)       # baseline window
            ax.plot(stim[k], y + k, "k.", ms=2)                                        # stim onset
        ax.text(1.22, y + len(rel) / 2, f"{name}\n(n={len(rel)})", color=COL[name], va="center", fontsize=8)
        y += len(rel) + 5
    ax.axvspan(ili, ili + r27.OUT_W, color="orange", alpha=0.25, label="outcome window [ILI, ILI+150 ms]")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.6, 1.2); ax.set_ylim(y, -2); ax.set_xlabel("time from corrected first lick L1 (s)")
    ax.set_ylabel("trials (chronological within type)")
    ax.set_title(f"Piezo licks aligned to L1 (black dot = stim onset, grey = baseline window)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax = fig.add_subplot(gs[0, 2]); bins = np.arange(-0.6, 1.2, 0.02)
    for name, tr in g.items():
        rel = rel_events(S["pz"], tr.L1.to_numpy())
        if len(rel):
            h = np.histogram(np.concatenate(rel), bins)[0] / len(rel) / 0.02
            ax.plot(bins[:-1] + 0.01, h, color=COL[name], label=name)
    ax.axvspan(ili, ili + r27.OUT_W, color="orange", alpha=0.25); ax.axvline(0, color="k", lw=0.8)
    ax.set_title("lick rate (Hz)", fontsize=9); ax.legend(fontsize=7); ax.set_xlabel("time from L1 (s)")
    ax = fig.add_subplot(gs[0, 3])
    for name, tr in g.items():
        l1 = tr.L1.to_numpy(); j = np.clip(np.searchsorted(S["pz"], l1), 1, len(S["pz"]) - 1)
        near = np.where(np.abs(S["pz"][j] - l1) < np.abs(S["pz"][j - 1] - l1), S["pz"][j], S["pz"][j - 1]) - l1
        ax.hist(near * 1e3, np.arange(-100, 101, 5), histtype="step", color=COL[name], label=name)
    ax.set_title("nearest piezo lick - L1 (ms): alignment check", fontsize=9); ax.set_xlabel("ms")
    ax = fig.add_subplot(gs[1, 2])
    for name, tr in g.items():
        l1 = tr.L1.to_numpy(); j = np.searchsorted(S["pz"], l1 + 0.02)
        second = np.where(j < len(S["pz"]), S["pz"][np.minimum(j, len(S["pz"]) - 1)] - l1, np.nan)
        second[second > 1.0] = np.nan
        ax.hist(second * 1e3, np.arange(0, 1001, 20), histtype="step", color=COL[name],
                label=f"{name}: {np.mean(second <= ili + r27.OUT_W):.0%} <= window end")
    ax.axvspan(ili * 1e3, (ili + r27.OUT_W) * 1e3, color="orange", alpha=0.25)
    ax.set_title(f"2nd lick latency after L1 (ms); ILI = {ili * 1e3:.0f} ms", fontsize=9); ax.legend(fontsize=6)
    ax = fig.add_subplot(gs[1, 3])
    vals = []
    for name, tr in g.items():
        l1 = tr.L1.to_numpy()
        n = np.searchsorted(S["pz"], l1 + ili + r27.OUT_W) - np.searchsorted(S["pz"], l1 + ili)
        vals.append(n)
    ax.boxplot(vals, tick_labels=[k.replace(" ", "\n") for k in g], showfliers=False)
    for k, v in enumerate(vals):
        ax.plot(np.full(len(v), k + 1) + np.random.uniform(-0.15, 0.15, len(v)), v, ".", ms=3, color=list(COL.values())[0], alpha=0.3)
    ax.set_title("# licks inside outcome window (motor confound)", fontsize=9)
    ax = fig.add_subplot(gs[2, :])
    w = S["wh_all"].sort_values("start_time"); hit = (w.lick_flag == 1).astype(float).to_numpy()
    ax.plot(np.arange(len(hit)), pd.Series(hit).rolling(10, min_periods=1, center=True).mean(), color="#2ca02c", label="whisker hit rate (10-trial)")
    a = S["act"][S["act"].auditory_stim == 1].sort_values("start_time")
    aidx = np.searchsorted(w.start_time.to_numpy(), a.start_time.to_numpy())
    ax.plot(aidx, pd.Series((a.lick_flag == 1).astype(float).to_numpy()).rolling(10, min_periods=1, center=True).mean(),
            color="#1f77b4", label="auditory hit rate")
    wh = S["groups"].get("whisker hit", S["groups"].get("R- whisker hit"))
    if S["split_time"] is not None:
        ax.axvline(np.searchsorted(w.start_time.to_numpy(), S["split_time"]), color="k", ls="--", label=S["split_how"])
    for q in np.quantile(np.arange(len(wh)), [1 / 3, 2 / 3]) if len(wh) else []:
        tq = wh.sort_values("start_time").start_time.iloc[int(q)]
        ax.axvline(np.searchsorted(w.start_time.to_numpy(), tq), color="0.5", ls=":")
    ax.set_xlabel("whisker trial index"); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7, ncol=4)
    ax.set_title("learning curve; dotted = whisker-hit terciles", fontsize=9)
    fig.suptitle(f"{S['sid']} ({S['cohort']}) timing check", fontsize=11)
    fig.savefig(OUT / f"A_timing_{S['sid']}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ fig B: implementation
def fig_impl(S, rng):
    stored = pd.read_csv(RES / S["mouse"] / "whisker_0" / "roc_analysis_rpe" / f"{S['mouse']}_roc_rpe.csv")
    g = S["groups"]; ili = S["ili"]
    rew = pd.concat([g["auditory hit"]] + ([g["whisker hit"]] if S["cohort"] == "R+" else []))
    l1, st0 = rew.L1.to_numpy(), rew.start_time.to_numpy()
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    ax = axs[0, 0]; mine = []
    for s in S["spikes"]:
        o = loop_rate(s, l1 + ili, l1 + ili + r27.OUT_W); b = loop_rate(s, st0 - 0.2, st0)
        y = np.r_[np.zeros(len(b)), np.ones(len(o))]; x = np.r_[b, o]
        mine.append(roc_auc_score(y, x) if np.ptp(x) > 0 else 0.5)
    ref = stored[stored.comparison == "reward_vs_baseline"].set_index("neuron_id").auc.reindex(S["units"].neuron_id).to_numpy()
    ax.plot(ref, mine, ".", ms=2); ax.plot([0, 1], [0, 1], "k-", lw=0.5)
    ax.set_title(f"reward_vs_baseline AUC: stored vs loop+sklearn\nmax |diff| = {np.nanmax(np.abs(ref - np.array(mine))):.2e}", fontsize=9)
    ax.set_xlabel("stored (027)"); ax.set_ylabel("independent re-implementation")
    ax = axs[0, 1]
    k = int(np.nanargmax(np.abs(ref - 0.5) * (ref < 0.75)))
    b = r27.rates(S["spikes"][k], st0 - 0.2, st0)[None]; o = r27.rates(S["spikes"][k], l1 + ili, l1 + ili + r27.OUT_W)[None]
    x = np.concatenate([b, o], 1)
    null = [roc_auc_score(np.r_[np.zeros(b.shape[1]), np.ones(o.shape[1])], rng.permutation(x[0])) for _ in range(1000)]
    auc, sel, p = r27.roc_perm(b, o, rng)
    ax.hist(null, 40, color="0.7"); ax.axvline(auc[0], color="r")
    ax.set_title(f"neuron {S['units'].neuron_id[k]}: permutation null, AUC={auc[0]:.3f}, p={p[0]:.3f}", fontsize=9)
    ax = axs[0, 2]
    for comp, dd in stored.dropna(subset=["p_value"]).groupby("comparison"):
        ax.hist(dd.p_value, np.linspace(0, 1, 21), histtype="step", density=True, label=comp)
    ax.set_title("p-value histograms (uniform = null)", fontsize=9); ax.legend(fontsize=6)
    # null calibration: [-0.2, 0] vs [-0.4, -0.2] on the same rewarded trials
    X1 = np.stack([r27.rates(s, st0 - 0.4, st0 - 0.2) for s in S["spikes"]])
    X2 = np.stack([r27.rates(s, st0 - 0.2, st0) for s in S["spikes"]])
    auc, sel, p = r27.roc_perm(X1, X2, rng)
    sig = p < r27.ALPHA
    ax = axs[1, 0]
    ax.bar(["up", "down"], [np.mean(sig & (sel > 0)), np.mean(sig & (sel < 0))], color=["k", "0.5"])
    ax.axhline(0.05, color="r", ls="--"); ax.set_ylim(0, 0.2)
    ax.set_title("NULL calibration: pre-trial [-0.2,0] vs [-0.4,-0.2]\n(expect ~5% each, red)", fontsize=9)
    ax = axs[1, 1]
    comp = stored[stored.comparison.isin(["reward_vs_baseline", "fa_vs_baseline", "unrewarded_lick_vs_baseline",
                                           "rminus_whisker_hit_vs_baseline"])]
    for c, dd in comp.groupby("comparison"):
        ax.hist(dd.selectivity.dropna(), np.linspace(-1, 1, 41), histtype="step", label=c)
    ax.set_title("selectivity (vs-baseline comparisons)", fontsize=9); ax.legend(fontsize=6)
    ax = axs[1, 2]; ax.axis("off")
    txt = [f"cohort {S['cohort']}, ILI {ili * 1e3:.0f} ms, split: {S['split_how']}"]
    for kk, tr in g.items():
        txt.append(f"{kk}: n={len(tr)}, RT(L1-stim) median {np.median(tr.L1 - tr.start_time) * 1e3:.0f} ms")
    ax.text(0, 1, "\n".join(txt), va="top", fontsize=9, family="monospace")
    fig.suptitle(f"{S['sid']} implementation checks", fontsize=11); fig.tight_layout()
    fig.savefig(OUT / f"B_impl_{S['sid']}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ fig C: examples
def terciles(tr):
    tr = tr.sort_values("start_time")
    if len(tr) < 6:
        return []
    return [tr.iloc[ix] for ix in np.array_split(np.arange(len(tr)), 3)]


def example_panel(fig, sub, S, k, title):
    gs = sub.subgridspec(3, 2, height_ratios=[2.2, 1, 1], width_ratios=[3, 1.2], hspace=0.15, wspace=0.35)
    st = S["spikes"][k]; ili = S["ili"]
    wh = S["groups"].get("whisker hit", pd.DataFrame())
    blocks = [("auditory hit", S["groups"]["auditory hit"])]
    blocks += [(f"W T{i + 1}", t) for i, t in enumerate(terciles(wh))]
    blocks += [("false alarm", S["groups"]["false alarm"])]
    ax = fig.add_subplot(gs[0, 0]); axp = fig.add_subplot(gs[1, 0], sharex=ax); y = 0
    bins = np.arange(-0.6, 1.2, 0.02)
    for name, tr in blocks:
        l1 = tr.L1.to_numpy()
        sp = rel_events(st, l1); lk = rel_events(S["pz"], l1)
        for j in range(len(l1)):
            ax.plot(sp[j], np.full(len(sp[j]), y + j), "|", ms=1.5, color=COL[name], mew=0.6)
            ax.plot(lk[j], np.full(len(lk[j]), y + j), ".", ms=1.2, color="0.6")
        y += len(l1) + 3
        if len(l1):
            h = np.histogram(np.concatenate(sp), bins)[0] / len(l1) / 0.02
            axp.plot(bins[:-1] + 0.01, np.convolve(h, np.ones(3) / 3, "same"), color=COL[name], lw=1, label=name)
    for a in (ax, axp):
        a.axvspan(ili, ili + r27.OUT_W, color="orange", alpha=0.2); a.axvline(0, color="k", lw=0.5)
    ax.set_ylim(y, -1); ax.set_xlim(-0.6, 1.2); ax.set_yticks([]); plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(title, fontsize=7.5)
    axp.set_xlabel("time from first lick (s)", fontsize=7); axp.set_ylabel("Hz", fontsize=7)
    axp.tick_params(labelsize=6); axp.legend(fontsize=5, ncol=3, loc="upper left")
    # tercile panel: baseline-subtracted outcome rate
    ax = fig.add_subplot(gs[0:2, 1])
    for grp, c in [(wh, "#2ca02c"), (S["groups"]["auditory hit"], "#1f77b4")]:
        m, e = [], []
        for t in terciles(grp):
            l1, s0 = t.L1.to_numpy(), t.start_time.to_numpy()
            v = r27.rates(st, l1 + ili, l1 + ili + r27.OUT_W) - r27.rates(st, s0 - 0.2, s0)
            m.append(v.mean()); e.append(v.std() / np.sqrt(len(v)))
        if m:
            ax.errorbar([1, 2, 3], m, e, color=c, marker="o", ms=3, capsize=2)
    fa = S["groups"]["false alarm"]
    if len(fa):
        v = r27.rates(st, fa.L1.to_numpy() + ili, fa.L1.to_numpy() + ili + r27.OUT_W) - r27.rates(st, fa.start_time.to_numpy() - 0.2, fa.start_time.to_numpy())
        ax.errorbar([3.6], [v.mean()], [v.std() / np.sqrt(len(v))], color="#d62728", marker="s", ms=3)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks([1, 2, 3, 3.6], ["T1", "T2", "T3", "FA"], fontsize=6)
    ax.tick_params(labelsize=6); ax.set_title("outcome - baseline (Hz)\ngreen whisker, blue auditory", fontsize=6.5)


def examples(rng, n_rpe=6, n_other=3):
    import glob
    fs = glob.glob(str(RES / "*" / "whisker_0" / "roc_analysis_rpe" / "*_roc_rpe.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d.cohort == "R+"]
    w = d.pivot_table(index=["session_id", "neuron_id"], columns="comparison", values="selectivity")
    s = d.pivot_table(index=["session_id", "neuron_id"], columns="comparison", values="significant").fillna(0).astype(bool)
    up = lambda c: s[c] & (w[c] > 0)                                               # noqa: E731
    dn = lambda c: s[c] & (w[c] < 0)                                               # noqa: E731
    sets = {
        "RPE candidate (whisker reward up, unexpected>expected, early>late, FA not up)":
            (up("reward_vs_baseline_whisker") & up("unexpected_vs_expected") & up("early_vs_late_whisker") & ~up("fa_vs_baseline"),
             w["reward_vs_baseline_whisker"] + w["unexpected_vs_expected"] + w["early_vs_late_whisker"] - w["fa_vs_baseline"], n_rpe),
        "lick-like (reward up AND FA up)":
            (up("reward_vs_baseline") & up("fa_vs_baseline"), w["reward_vs_baseline"] + w["fa_vs_baseline"], n_other),
        "negative dip (FA down, reward up)":
            (dn("fa_vs_baseline") & up("reward_vs_baseline"), w["reward_vs_baseline"] - w["fa_vs_baseline"], n_other),
    }
    picks = []
    for lab, (mask, score, n) in sets.items():
        sc = score[mask].sort_values(ascending=False)
        used = set()
        for (sid, nid), v in sc.items():
            if sid in used:
                continue
            picks.append((lab, sid, nid, v)); used.add(sid)
            if len(used) >= n:
                break
        print(f"{lab}: {mask.sum()} / {len(mask)} R+ day-0 units", flush=True)
    cache = {}
    fig = plt.figure(figsize=(22, 5.2 * int(np.ceil(len(picks) / 3))))
    outer = fig.add_gridspec(int(np.ceil(len(picks) / 3)), 3, hspace=0.35, wspace=0.18)
    for i, (lab, sid, nid, v) in enumerate(picks):
        if sid not in cache:
            cache[sid] = load(sid)
        S = cache[sid]
        k = int(np.where(S["units"].neuron_id.to_numpy() == nid)[0][0])
        row = w.loc[(sid, nid)]
        title = (f"[{lab.split(' (')[0]}] {sid} n{nid} {S['acr'][k]}\n"
                 f"sel: rew_wh {row['reward_vs_baseline_whisker']:+.2f} unexp {row['unexpected_vs_expected']:+.2f} "
                 f"early {row['early_vs_late_whisker']:+.2f} FA {row['fa_vs_baseline']:+.2f}")
        example_panel(fig, outer[i // 3, i % 3], S, k, title)
    fig.savefig(OUT / "C_examples_R+.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(picks, columns=["set", "session_id", "neuron_id", "score"]).to_csv(OUT / "C_examples_R+.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=["MH069_20260122_111455", "AB093_20231207_111207"])
    ap.add_argument("--skip-examples", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for sid in a.sessions:
        S = load(sid)
        fig_timing(S); fig_impl(S, rng)
        print("done", sid, flush=True)
    if not a.skip_examples:
        examples(rng)
    print("ALL DONE", flush=True)
