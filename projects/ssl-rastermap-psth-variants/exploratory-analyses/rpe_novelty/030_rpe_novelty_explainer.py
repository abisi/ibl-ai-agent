"""Explainer figure for the novelty result of the RPE ROC v2 analysis (029_roc_rpe_v2.py).

Novelty = the cue response (50-150 ms after whisker onset, vs the same trial's [-110, -10] ms baseline) to the first
N_FIRST whisker presentations of the session, and whether it is larger than on the following whisker trials.
Panels
  A  definitions: trial sets in session order and the within-trial windows; what C1 and C1b test
  B  behaviour: P(lick) and reaction time on whisker trials 1..20 (licking can enter the cue window if RT < 150 ms)
  C  trial-by-trial cue response (z, per neuron) for whisker trials 1..20 vs auditory trials 1..20 (auditory is already
     familiar = no-novelty control); neurons selected as cue-responsive on whisker T3 only (independent of trials 1..20,
     so the early-vs-later comparison is not circular)
  D  population PSTH (same neurons) aligned to whisker onset: First 5 vs Next vs T3, per cohort
  E  fraction of neurons with C1 / C1b significant (uncorrected p < .05) per direction vs shuffled-label chance
  F  example neurons (raster + PSTH)
Input: 029 pilot (or full) summary parquet (quality_label == 'good' units) + NWB spikes.
Output: <summary>/novelty_explainer.png (+ novelty_explainer_data.parquet)
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m = importlib.import_module("029_roc_rpe_v2")
N_TR = 20
BINS = np.arange(-0.2, 0.401, 0.01)
COH = {"R+": "#00B400", "R-": "#C800C8"}
SET_COL = {"First 5": "#ff7f0e", "Next": "#1f77b4", "T3": "0.35"}
EXAMPLES = [("AB159_20250409_135813", 3165), ("AB087_20231017_141901", 532)]


def per_session(sid, nids):
    _, units, act = m.load_session(sid)
    cohort, S = m.sets_for(act)
    units = units[units.neuron_id.isin(nids)].reset_index(drop=True)
    sp = [np.sort(np.asarray(s)) for s in units.spike_times]
    R = m.Rates(sp)
    out = dict(cohort=cohort, neuron_id=units.neuron_id.to_numpy())
    for key, tr in [("W", S["W_all"]), ("A", S["A_all"])]:
        bs = R.bs(tr, "cue")                                             # (N, n_trials) cue - same-trial baseline
        sd = bs.std(1); sd[sd == 0] = np.nan
        z = bs[:, :N_TR] / sd[:, None]
        out[f"z_{key}"] = np.pad(z, ((0, 0), (0, N_TR - z.shape[1])), constant_values=np.nan)
    base = R.win(S["W_all"], "baseline_cue")
    mu, sd = base.mean(1), np.maximum(base.std(1), 1.0)
    for name, tr in [("First 5", S["W_first"]), ("Next", S["W_next"]), ("T3", S["W_all_T3"])]:
        t0 = tr.start_time.to_numpy(float)
        h = np.stack([np.histogram(np.concatenate([s[(s >= t + BINS[0]) & (s < t + BINS[-1])] - t for t in t0]) if len(t0) else [],
                                   BINS)[0] / max(len(t0), 1) / 0.01 for s in sp])
        out[f"psth_{name}"] = (h - mu[:, None]) / sd[:, None]
    w = S["W_all"].iloc[:N_TR]
    out["p_lick"] = np.pad(w.lick_flag.to_numpy(float), (0, N_TR - len(w)), constant_values=np.nan)
    rt = (w.L1 - w.start_time).to_numpy(float)
    out["rt"] = np.pad(np.where(w.lick_flag == 1, rt, np.nan), (0, N_TR - len(w)), constant_values=np.nan)
    return out, (units, S)


def _worker(args):
    sid, nids = args
    return sid, per_session(sid, nids)[0]


def draw_schema(ax):
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 9.6, "A  What is tested", fontsize=11, weight="bold")
    ax.text(0, 8.7, "Whisker trials in session order:", fontsize=8.5)
    segs = [("First 5", 0, 1.2, SET_COL["First 5"]), ("Next\n(to end of T1)", 1.2, 3.3, SET_COL["Next"]),
            ("T2", 3.3, 6.6, "0.75"), ("T3", 6.6, 9.9, SET_COL["T3"])]
    for lab, a, b, c in segs:
        ax.add_patch(plt.Rectangle((a, 7.0), b - a, 1.0, color=c, alpha=0.8))
        ax.text((a + b) / 2, 7.5, lab, ha="center", va="center", fontsize=7.5, color="w" if c != "0.75" else "k")
    ax.text(0, 6.3, "Within each trial (time from whisker onset):", fontsize=8.5)
    x0, sc = 3.5, 10.0
    ax.plot([x0 - 0.25 * sc, x0 + 0.3 * sc], [5.0, 5.0], "k-", lw=0.8)
    ax.add_patch(plt.Rectangle((x0 - 0.11 * sc, 4.6), 0.1 * sc, 0.8, color="0.6"))
    ax.add_patch(plt.Rectangle((x0 + 0.05 * sc, 4.6), 0.1 * sc, 0.8, color="orange"))
    ax.plot([x0, x0], [4.3, 5.7], "k-", lw=1.5)
    ax.text(x0 - 0.06 * sc, 4.0, "baseline\n-110..-10 ms", ha="center", va="top", fontsize=7)
    ax.text(x0 + 0.10 * sc, 4.0, "cue\n50..150 ms", ha="center", va="top", fontsize=7)
    ax.text(x0, 5.9, "whisker onset", ha="center", fontsize=7)
    ax.text(0, 2.2, "C1  novelty response: First-5 cue rate > same trials' baseline (ROC)\n"
                    "C1b novelty decrement: First-5 (cue - baseline) > Next (cue - baseline)\n"
                    "Hypothesis: both present and similar in R+ and R- (no outcome learned yet)",
            fontsize=8, va="top", family="monospace")


def main(a):
    summ = m.SUMMARY / a.tag if a.tag else m.SUMMARY
    d = pd.read_parquet(summ / "rpe_v2_neurons.parquet")
    d = d[(d.stage == "learning") & (d.quality_label == "good")]
    assert len(d) and (d.quality_label == "good").all(), "good units only"
    rows, ex = [], {}
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(sid, set(g.neuron_id)) for sid, g in d.groupby("session_id")]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(_worker, jobs))
    for sid, o in results:
        for i, nid in enumerate(o["neuron_id"]):
            r = dict(session_id=sid, neuron_id=nid, cohort=o["cohort"])
            for t in range(N_TR):
                r[f"zW{t + 1}"], r[f"zA{t + 1}"] = o["z_W"][i, t], o["z_A"][i, t]
            for name in SET_COL:
                r[f"psth_{name}"] = o[f"psth_{name}"][i]
            rows.append(r)
        rows.append(dict(session_id=sid, neuron_id=-1, cohort=o["cohort"], p_lick=o["p_lick"], rt=o["rt"]))
        print("done", sid, flush=True)
    for es, en in EXAMPLES:                                              # reload only the example sessions
        if es in set(d.session_id):
            _, units, act = m.load_session(es)
            ex[(es, en)] = (units, m.sets_for(act)[1])
    D = pd.DataFrame(rows)
    beh = D[D.neuron_id == -1]; N = D[D.neuron_id >= 0].merge(d, on=["session_id", "neuron_id", "cohort"])
    N.drop(columns=[c for c in N.columns if c.startswith("psth_")]).to_parquet(summ / "novelty_explainer_data.parquet")
    resp = N[N["popauc_cue_W_all_T3"] > 0.6]                              # independent selection (late trials)

    fig = plt.figure(figsize=(17, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)
    draw_schema(fig.add_subplot(gs[0, 0]))
    # B behaviour
    ax = fig.add_subplot(gs[0, 1]); ax2 = ax.twinx(); x = np.arange(1, N_TR + 1)
    for coh, c in COH.items():
        b = beh[beh.cohort == coh]
        if not len(b):
            continue
        pl = np.stack(b.p_lick.to_list()); rt = np.stack(b.rt.to_list())
        ax.errorbar(x, np.nanmean(pl, 0), np.nanstd(pl, 0) / np.sqrt(len(b)), color=c, marker="o", ms=3,
                    label=f"{coh} P(lick) (n={len(b)} sessions)")
        fast = np.nanmean(rt[:, :5] < 0.15)
        ax.text(0.98, 0.05 + 0.08 * list(COH).index(coh), f"{coh}: {fast:.0%} of first-5 licks have RT < 150 ms",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=c)
    ax.axvspan(0.5, 5.5, color=SET_COL["First 5"], alpha=0.12)
    ax.set_ylim(0, 1.05); ax.set_xlabel("whisker trial index"); ax.set_ylabel("P(lick)")
    ax2.set_yticks([])
    ax.set_title("B  Behaviour: mice already lick on the first whisker trials", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper right")
    # E fractions
    ax = fig.add_subplot(gs[0, 2]); w = 0.18
    for j, c in enumerate(["C1", "C1b"]):
        for k, coh in enumerate(COH):
            g = N[(N.cohort == coh) & ~N[f"skip_{c}"].astype(bool)]
            up = ((g[f"p_{c}"] < .05) & (g[f"auc_{c}"] > .5)).mean(); dn = ((g[f"p_{c}"] < .05) & (g[f"auc_{c}"] < .5)).mean()
            ch = ((g[f"p_null_{c}"] < .05) & (g[f"auc_null_{c}"] > .5)).mean()
            xx = j + (k - 0.5) * 2 * w
            ax.bar(xx - w / 2, up, w, color=COH[coh], label=f"{coh} up" if j == 0 else None)
            ax.bar(xx + w / 2, dn, w, color=COH[coh], alpha=0.35, label=f"{coh} down" if j == 0 else None)
            ax.plot([xx - w, xx], [ch, ch], "k-", lw=2, label="chance (shuffled labels)" if (j == 0 and k == 0) else None)
            ax.text(xx, up + 0.003, f"n={len(g)}", ha="center", fontsize=6.5)
    ax.set_xticks([0, 1], ["C1 novelty response\n(First 5 vs baseline)", "C1b novelty decrement\n(First 5 vs Next)"], fontsize=8)
    ax.set_ylabel("fraction of good units (p < .05)"); ax.legend(fontsize=7)
    ax.set_title("E  How many neurons show it", fontsize=10, loc="left")
    # C novelty curve
    for col, coh in enumerate(COH):
        ax = fig.add_subplot(gs[1, col])
        g = resp[resp.cohort == coh]
        for key, ls, lab in [("W", "-", "whisker (new stimulus)"), ("A", "--", "auditory (familiar, control)")]:
            ps = g.groupby("session_id")[[f"z{key}{t}" for t in x]].mean()
            ax.errorbar(x, ps.mean(), ps.std() / np.sqrt(len(ps)), color=COH[coh], ls=ls, marker="o" if key == "W" else "s",
                        ms=3, capsize=2, label=f"{lab}")
        ax.axvspan(0.5, 5.5, color=SET_COL["First 5"], alpha=0.12); ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("trial index of that modality (session order)"); ax.set_ylabel("cue - baseline (z per neuron)")
        ax.set_title(f"C  {coh}: cue response per trial\n{len(g)} neurons cue-responsive on whisker T3, "
                     f"{g.session_id.nunique()} sessions (+-SEM)", fontsize=9, loc="left")
        ax.legend(fontsize=7)
    # D PSTHs
    ax = fig.add_subplot(gs[1, 2]); tc = (BINS[:-1] + 0.005) * 1e3
    for coh, lw_ in [("R+", 1.0), ("R-", 2.2)]:
        g = resp[resp.cohort == coh]
        for name, c in SET_COL.items():
            ps = g.groupby("session_id")[f"psth_{name}"].apply(lambda s: np.nanmean(np.stack(s.to_list()), 0))
            mu = np.nanmean(np.stack(ps.to_list()), 0)
            ax.plot(tc, np.convolve(mu, np.ones(3) / 3, "same"), color=c, lw=lw_, ls="-" if coh == "R-" else "--",
                    label=f"{coh} {name}")
    ax.axvspan(50, 150, color="orange", alpha=0.12); ax.axvspan(-110, -10, color="0.8", alpha=0.4); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("time from whisker onset (ms)"); ax.set_ylabel("firing rate (z vs baseline)")
    ax.set_title("D  Population PSTH, same neurons\n(solid R-, dashed R+)", fontsize=9, loc="left"); ax.legend(fontsize=6.5, ncol=2)
    # F examples
    for col, (key, (units, S)) in enumerate(ex.items()):
        sub = gs[2, col].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
        ax, axp = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        st = np.sort(np.asarray(units.spike_times[units.neuron_id == key[1]].iloc[0])); y = 0
        row = d[(d.session_id == key[0]) & (d.neuron_id == key[1])].iloc[0]
        for name, tr in [("First 5", S["W_first"]), ("Next", S["W_next"]), ("T3", S["W_all_T3"])]:
            t0 = tr.start_time.to_numpy(float); rel = [st[(st >= t - 0.2) & (st < t + 0.4)] - t for t in t0]
            for r_ in rel:
                ax.plot(r_ * 1e3, np.full(len(r_), y), "|", ms=2, color=SET_COL[name]); y += 1
            y += 3
            h = np.histogram(np.concatenate(rel) if rel else [], BINS)[0] / max(len(t0), 1) / 0.01
            axp.plot(tc, np.convolve(h, np.ones(3) / 3, "same"), color=SET_COL[name], label=f"{name} (n={len(t0)})")
        for aa in (ax, axp):
            aa.axvspan(50, 150, color="orange", alpha=0.12); aa.axvspan(-110, -10, color="0.8", alpha=0.4); aa.axvline(0, color="k", lw=0.5)
            aa.set_xlim(-200, 400)
        ax.set_ylim(y, -1); ax.set_yticks([]); ax.set_xticks([])
        ax.set_title(f"F  example: {key[0]} n{key[1]} {row.area_acronym_custom} ({row.cohort})\n"
                     f"C1 AUC={row.auc_C1:.2f} (p={row.p_C1:.3f}), C1b AUC={row.auc_C1b:.2f} (p={row.p_C1b:.3f})",
                     fontsize=9, loc="left")
        axp.set_xlabel("time from whisker onset (ms)"); axp.set_ylabel("Hz"); axp.legend(fontsize=7)
    # G regions of C1b-up neurons
    ax = fig.add_subplot(gs[2, 2])
    nov = N[(N.p_C1b < .05) & (N.auc_C1b > .5)]
    tab = pd.crosstab(nov.area_group, nov.cohort).reindex(columns=list(COH), fill_value=0)
    tot = pd.crosstab(N.area_group, N.cohort).reindex(columns=list(COH), fill_value=0)
    frac = (tab / tot.reindex(tab.index)).fillna(0)
    frac = frac.loc[frac.max(1).sort_values().index]
    yy = np.arange(len(frac))
    for k, coh in enumerate(COH):
        ax.barh(yy + (k - 0.5) * 0.4, frac[coh], 0.4, color=COH[coh], label=coh)
        for yv, (n1, n0) in zip(yy, zip(tab.reindex(frac.index)[coh], tot.reindex(frac.index)[coh])):
            ax.text(frac[coh].iloc[int(yv)] + 0.002, yv + (k - 0.5) * 0.4, f"{n1}/{n0}", va="center", fontsize=6)
    ax.set_yticks(yy, frac.index, fontsize=7); ax.set_xlabel("fraction of good units with C1b up (p < .05)")
    ax.set_title("G  Where (area group; counts = C1b-up / all good units)", fontsize=9, loc="left"); ax.legend(fontsize=7)
    fig.suptitle(f"Novelty response to the first whisker presentations (learning day, {d.session_id.nunique()} sessions, "
                 f"quality_label good, uncorrected p)", fontsize=13, y=0.995)
    fig.savefig(summ / "novelty_explainer.png", dpi=130, bbox_inches="tight")
    print("saved", summ / "novelty_explainer.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
