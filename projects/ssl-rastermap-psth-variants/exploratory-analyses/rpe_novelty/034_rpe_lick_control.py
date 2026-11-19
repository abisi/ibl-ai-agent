"""Licking control for the RPE-v2 classes (learning day, quality_label == 'good' units).

1. Describe licking: piezo licks in the outcome window (30-300 ms after the corrected first lick L1) and first-lick
   latency (L1 - whisker/auditory onset) per trial, for whisker-lick thirds (T1-T3) and auditory-hit halves (H1/H2),
   per cohort (per session, then R+ vs R-: Mann-Whitney AND Welch).
2. Lick-matched reruns of the 029 comparisons (same ROC + label-permutation p + shuffled-label chance run):
     O2/O5 matched  whisker-lick T1 vs T3, trials matched EXACTLY on lick count in the window (0, 1, 2, 3+)
     O3/O6 matched  whisker-lick T1 vs auditory-hit H1 / auditory H1 vs H2, same matching
     O1/O4 by lick count  outcome vs same-trial baseline, whisker-lick T1 trials restricted to ONE lick-count stratum
                    (0, 1, 2, 3+) -> the R+ vs R- difference in responsive fraction at equal licking
                    (also on all whisker-lick trials of the session, more trials per stratum)
   Matching: within each count, min(n_a, n_b) trials drawn at random from each set (fixed seed).
3. Population activity of ALL good units (no selection -> no circularity), z-scored to the pre-trial rate, for
   whisker-lick trials of one lick-count stratum, R+ vs R-; with the lick rate of the same trials.
Output: <summary>/lick_control/: lick_trials.parquet, lick_control_neurons.parquet, lick_control_psth.parquet,
        lick_control_summary.csv, lick_control.png/.pdf
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
KMAX = 3                                   # lick counts 0, 1, 2, 3+
N_MIN_STRAT = 5
BW = 0.01
LBINS = np.arange(-0.5, 0.8001, BW)
SETS = ["W_lick_T1", "W_lick_T2", "W_lick_T3", "A_hit_H1", "A_hit_H2"]


def lick_counts(pz, tr):
    l1 = tr.L1.to_numpy(float)
    return np.minimum(np.searchsorted(pz, l1 + m.OUT_W[1]) - np.searchsorted(pz, l1 + m.OUT_W[0]), KMAX)


def match_idx(ca, cb, rng):
    ia, ib = [], []
    for k in range(KMAX + 1):
        a, b = np.where(ca == k)[0], np.where(cb == k)[0]
        n = min(len(a), len(b))
        if n:
            ia += list(rng.choice(a, n, replace=False)); ib += list(rng.choice(b, n, replace=False))
    return np.sort(ia).astype(int), np.sort(ib).astype(int)


def roc_with_null(x1, x2, rng):
    auc, p, _ = m.roc_test(x1, x2, rng)
    pool = np.concatenate([x1, x2], 1)[:, rng.permutation(x1.shape[1] + x2.shape[1])]
    auc0, p0, _ = m.roc_test(pool[:, :x1.shape[1]], pool[:, x1.shape[1]:], rng)
    return auc, p, auc0, p0


def worker(args):
    sid, nids = args
    rng = np.random.default_rng(abs(hash(sid)) % 2 ** 32)
    nwb, units, act = m.load_session(sid)
    cohort, S = m.sets_for(act)
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    units = units[units.neuron_id.isin(nids)].reset_index(drop=True)
    sp = [np.sort(np.asarray(s)) for s in units.spike_times]
    R = m.Rates(sp)
    C = {k: lick_counts(pz, S[k]) for k in SETS + ["W_lick"]}
    trials = pd.concat([pd.DataFrame(dict(session_id=sid, cohort=cohort, set=k, n_licks=C[k],
                                          rt=(S[k].L1 - S[k].start_time).to_numpy(float))) for k in SETS])
    res = pd.DataFrame(dict(session_id=sid, neuron_id=units.neuron_id, cohort=cohort))
    N = len(res)

    def put(name, fn):
        out = fn()
        for lab, v in zip(["auc", "p", "auc_null", "p_null"], out if out is not None else [np.full(N, np.nan)] * 4):
            res[f"{lab}_{name}"] = v

    # matched set-vs-set comparisons (baseline-subtracted outcome rates, as in 029)
    for name, a, b in [("T1vsT3_matched", "W_lick_T1", "W_lick_T3"), ("WvsA_matched", "W_lick_T1", "A_hit_H1"),
                       ("H1vsH2_matched", "A_hit_H1", "A_hit_H2")]:
        ia, ib = match_idx(C[a], C[b], rng)
        res[f"n_{name}"] = len(ia)
        put(name, lambda: roc_with_null(R.bs(S[a].iloc[ia], "outcome"), R.bs(S[b].iloc[ib], "outcome"), rng)
            if len(ia) >= N_MIN_STRAT else None)
    # outcome vs baseline within one lick-count stratum
    for src in ["W_lick_T1", "W_lick"]:
        for k in range(KMAX + 1):
            idx = np.where(C[src] == k)[0]
            name = f"out_{src}_k{k}"
            res[f"n_{name}"] = len(idx)
            tr = S[src].iloc[idx]
            put(name, lambda: roc_with_null(R.win(tr, "outcome"), R.win(tr, "baseline_outcome"), rng)
                if len(idx) >= N_MIN_STRAT else None)
    # population activity of all good units per lick-count stratum (whisker-lick trials, all thirds; auditory H1+H2)
    t_all = S["W_all"].start_time.to_numpy(float)
    base = np.stack([np.array([np.sum((s >= t - 0.31) & (s < t - 0.01)) / 0.30 for t in t_all]) for s in sp])
    mu, sd = base.mean(1), np.maximum(base.std(1), 1.0)
    prow = []
    for src, tr_all, cc in [("whisker", S["W_lick"], C["W_lick"]),
                            ("auditory", pd.concat([S["A_hit_H1"], S["A_hit_H2"]]), np.r_[C["A_hit_H1"], C["A_hit_H2"]])]:
        for k in range(KMAX + 1):
            tr = tr_all.iloc[np.where(cc == k)[0]]
            if len(tr) < N_MIN_STRAT:
                continue
            l1 = tr.L1.to_numpy(float)
            h = np.stack([np.histogram(np.concatenate([s[(s >= t + LBINS[0]) & (s < t + LBINS[-1])] - t for t in l1]),
                                       LBINS)[0] / len(l1) / BW for s in sp])
            z = (h - mu[:, None]) / sd[:, None]
            lk = np.histogram(np.concatenate([pz[(pz >= t + LBINS[0]) & (pz < t + LBINS[-1])] - t for t in l1]),
                              LBINS)[0] / len(l1) / BW
            prow.append(dict(session_id=sid, cohort=cohort, src=src, k=k, n_trials=len(tr), n_units=len(sp),
                             z=np.nanmean(z, 0).tolist(), lick=lk.tolist()))
    return trials, res, pd.DataFrame(prow)


def cmd_compute(a):
    summ = m.SUMMARY / a.tag; out = summ / "lick_control"; out.mkdir(exist_ok=True)
    d = pd.read_parquet(summ / "rpe_v2_neurons.parquet")
    d = d[(d.stage == "learning") & (d.quality_label == "good")]
    assert len(d) and (d.quality_label == "good").all()
    jobs = [(sid, set(g.neuron_id)) for sid, g in d.groupby("session_id")]
    T, Rs, Ps = [], [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (t, r, p) in enumerate(pool.map(worker, jobs)):
            T.append(t); Rs.append(r); Ps.append(p)
            print(f"done {i + 1}/{len(jobs)}", flush=True)
    pd.concat(T).to_parquet(out / "lick_trials.parquet", index=False)
    pd.concat(Rs).to_parquet(out / "lick_control_neurons.parquet", index=False)
    pd.concat(Ps).to_parquet(out / "lick_control_psth.parquet", index=False)
    print("cached", flush=True)


def test2(x, y):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    summ = m.SUMMARY / a.tag; out = summ / "lick_control"
    T = pd.read_parquet(out / "lick_trials.parquet")
    Rn = pd.read_parquet(out / "lick_control_neurons.parquet")
    Pp = pd.read_parquet(out / "lick_control_psth.parquet")
    d = pd.read_parquet(summ / "rpe_v2_neurons.parquet")
    d = d[(d.stage == "learning") & (d.quality_label == "good")]
    rows = []

    def frac(g, name, sign, null=False):
        s = "_null" if null else ""
        v = g.dropna(subset=[f"p{s}_{name}"])
        return ((v[f"p{s}_{name}"] < .05) & (np.sign(v[f"auc{s}_{name}"] - .5) == sign)).mean() if len(v) else np.nan

    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(3, 4, hspace=0.7, wspace=0.6, left=0.08, right=0.975, top=0.955, bottom=0.05,
                          height_ratios=[1, 1, 1.35])
    # A: lick counts per set
    ax = fig.add_subplot(gs[0, :2])
    ps = T.groupby(["session_id", "cohort", "set"]).n_licks.mean().reset_index()
    for k, s_ in enumerate(SETS):
        for j, coh in enumerate(COH):
            v = ps[(ps.set == s_) & (ps.cohort == coh)].n_licks
            x = k + (j - 0.5) * 0.35
            ax.scatter(x + np.random.default_rng(k).uniform(-0.08, 0.08, len(v)), v, s=4, color=COH[coh], alpha=0.4, lw=0)
            ax.errorbar(x, v.mean(), v.std() / np.sqrt(len(v)), fmt="o", ms=3.5, mfc=COH[coh], color="k", mew=0.5, capsize=2)
        pm, pw = test2(ps[(ps.set == s_) & (ps.cohort == "R+")].n_licks, ps[(ps.set == s_) & (ps.cohort == "R-")].n_licks)
        ax.text(k, 3.25, f"MWU {fmt_p(pm)}\nWelch {fmt_p(pw)}", ha="center", fontsize=5.5)
        rows.append(dict(analysis="licks_in_window", set=s_, mean_Rplus=ps[(ps.set == s_) & (ps.cohort == "R+")].n_licks.mean(),
                         mean_Rminus=ps[(ps.set == s_) & (ps.cohort == "R-")].n_licks.mean(), p_MWU=pm, p_Welch=pw))
    ax.set_xticks(range(len(SETS)), ["Whisker T1", "Whisker T2", "Whisker T3", "Auditory H1", "Auditory H2"])
    ax.set_ylim(0, 3.7); ax.set_ylabel("Licks 30-300 ms after first lick\n(per trial, session mean; 3 = 3+)")
    ax.set_title("a  Licking in the outcome window (green R+, magenta R−)", loc="left")
    # B: first-lick latency
    ax = fig.add_subplot(gs[0, 2:])
    pr = T.groupby(["session_id", "cohort", "set"]).rt.median().reset_index()
    for k, s_ in enumerate(SETS):
        for j, coh in enumerate(COH):
            v = pr[(pr.set == s_) & (pr.cohort == coh)].rt * 1e3
            ax.errorbar(k + (j - 0.5) * 0.35, v.mean(), v.std() / np.sqrt(len(v)), fmt="o", ms=3.5, mfc=COH[coh], color="k",
                        mew=0.5, capsize=2)
    ax.set_xticks(range(len(SETS)), ["W T1", "W T2", "W T3", "A H1", "A H2"]); ax.set_ylabel("First-lick latency (ms)")
    ax.set_title("b  Reaction time (session median, mean ± SEM)", loc="left")
    # C: outcome response at equal licking, R+ vs R-
    for jj, src in enumerate(["W_lick_T1", "W_lick"]):
        ax = fig.add_subplot(gs[1, 2 * jj:2 * jj + 2])
        for k in range(KMAX + 1):
            name = f"out_{src}_k{k}"
            per = Rn.groupby(["session_id", "cohort"]).apply(lambda g: pd.Series(dict(
                f=frac(g, name, 1), c=frac(g, name, 1, True)))).reset_index().dropna(subset=["f"])
            for j, coh in enumerate(COH):
                v = per[per.cohort == coh]
                x = k + (j - 0.5) * 0.35
                ax.errorbar(x, 100 * v.f.mean(), 100 * v.f.std() / np.sqrt(max(len(v), 1)), fmt="o", ms=3.5, mfc=COH[coh],
                            color="k", mew=0.5, capsize=2)
                ax.text(x, -3.5, f"{len(v)}", ha="center", fontsize=5, color=COH[coh])
                ax.plot([x - 0.14, x + 0.14], [100 * v.c.mean()] * 2, "--", color="0.4", lw=0.7)
            pm, pw = test2((per[per.cohort == "R+"].f - per[per.cohort == "R+"].c), (per[per.cohort == "R-"].f - per[per.cohort == "R-"].c))
            ax.text(k, ax.get_ylim()[1] if False else 38, f"MWU {fmt_p(pm)}\nWelch {fmt_p(pw)}", ha="center", fontsize=5.5)
            rows.append(dict(analysis=f"outcome_vs_baseline_{src}", set=f"{k}{'+' if k == KMAX else ''} licks",
                             mean_Rplus=100 * per[per.cohort == "R+"].f.mean(), mean_Rminus=100 * per[per.cohort == "R-"].f.mean(),
                             n_sess_Rplus=int((per.cohort == "R+").sum()), n_sess_Rminus=int((per.cohort == "R-").sum()),
                             p_MWU=pm, p_Welch=pw))
        ax.set_ylim(-5, 44); ax.set_xticks(range(KMAX + 1), ["0", "1", "2", "3+"])
        ax.set_xlabel("Licks in window (stratum)"); ax.set_ylabel("% outcome-responsive\n(outcome > baseline)")
        ax.set_title(f"{'c' if jj == 0 else 'd'}  Outcome response at equal licking\n    "
                     f"({'first third of whisker licks' if src == 'W_lick_T1' else 'all whisker licks'}; n = sessions)", loc="left")
    # E: matched set-vs-set comparisons
    ax = fig.add_subplot(gs[2, :2])
    comps = [("T1vsT3_matched", "O2/O5", "Whisker\nT1 vs T3"), ("WvsA_matched", "O3", "Whisker T1 vs\nauditory H1"),
             ("H1vsH2_matched", "O6", "Auditory\nH1 vs H2")]
    labels = []
    for k, (name, orig, lab) in enumerate(comps):
        for j, coh in enumerate(COH):
            g = Rn[Rn.cohort == coh]
            up, dn, ch = frac(g, name, 1), frac(g, name, -1), frac(g, name, 1, True)
            if orig == "O2/O5":
                dd = d[d.cohort == coh]; oc = "O2" if coh == "R+" else "O5"
                up0 = ((dd[f"p_{oc}"] < .05) & (dd[f"auc_{oc}"] > .5)).mean()
            elif orig == "O3":
                dd = d[d.cohort == coh]
                up0 = ((dd["p_O3"] < .05) & (dd["auc_O3"] > .5)).mean() if coh == "R+" else np.nan
            else:
                dd = d[d.cohort == coh]; up0 = ((dd["p_O6"] < .05) & (dd["auc_O6"] > .5)).mean()
            x = k + (j - 0.5) * 0.4
            ax.bar(x - 0.09, 100 * up0, 0.17, color=COH[coh], alpha=0.35)
            ax.bar(x + 0.09, 100 * up, 0.17, color=COH[coh])
            ax.plot([x + 0.0, x + 0.18], [100 * ch] * 2, "k-", lw=1)
            rows.append(dict(analysis=f"matched_{name}", set=coh, pct_up_unmatched=100 * up0, pct_up_matched=100 * up,
                             pct_down_matched=100 * dn, pct_chance=100 * ch,
                             median_matched_trials=float(g[f"n_{name}"].median())))
        labels.append(lab)
    ax.set_xticks(range(len(comps)), labels, fontsize=6.5); ax.set_ylabel("% neurons, first set > second")
    ax.set_title("e  Set-vs-set tests before (light) / after (dark)\n    lick-count matching; black = chance", loc="left")
    # F: population activity at equal licking (all good units)
    for jj, k in enumerate([1, 2]):
        sub = gs[2, 2 + jj].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
        ax, axl = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        tc = (LBINS[:-1] + BW / 2) * 1e3
        for coh in COH:
            for src, ls in [("whisker", "-"), ("auditory", ":")]:
                g = Pp[(Pp.cohort == coh) & (Pp.src == src) & (Pp.k == k)]
                if not len(g):
                    continue
                z = np.stack(g.z.map(np.asarray).to_list()); lk = np.stack(g.lick.map(np.asarray).to_list())
                sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")        # noqa: E731
                mu, se = sm(z.mean(0)), sm(z.std(0) / np.sqrt(len(z)))
                ax.fill_between(tc, mu - se, mu + se, color=COH[coh], alpha=0.2, lw=0)
                ax.plot(tc, mu, color=COH[coh], ls=ls, lw=1, label=f"{coh} {src} (n={len(g)})")
                axl.plot(tc, sm(lk.mean(0)), color=COH[coh], ls=ls, lw=0.8)
        for a_ in (ax, axl):
            a_.axvspan(30, 300, color="#FDD49E", alpha=0.45, lw=0); a_.axvline(0, color="k", lw=0.5)
            a_.set_xlim(-500, 800); a_.spines[["top", "right"]].set_visible(False)
        ax.set_xticklabels([]); axl.set_xlabel("Time from first lick (ms)"); axl.set_ylabel("Licks/s", fontsize=6)
        ax.set_ylabel("All good units (z)"); ax.legend(frameon=False, fontsize=5, loc="upper right")
        ax.set_title(f"{'f' if jj == 0 else 'g'}  {k} lick{'s' if k > 1 else ''} in window", loc="left")
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"lick_control.{ext}", dpi=300)
    S = pd.DataFrame(rows); S.to_csv(out / "lick_control_summary.csv", index=False)
    pd.set_option("display.width", 220); print(S.round(4).to_string()); print("saved", out / "lick_control.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compute"); c.add_argument("--tag", default="learning"); c.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("plot"); p.add_argument("--tag", default="learning")
    a = ap.parse_args()
    {"compute": cmd_compute, "plot": cmd_plot}[a.cmd](a)
