"""124 -- Time course of the signed single-neuron selectivities around the first lick (user 2026-10-14, review point 7): when, relative
to the lick, does whisker-hit activity of reward-lick neurons separate from spontaneous licks (s_WS) and approach auditory hits (s_WA)?
Context: 123 finds the convergence lick-aligned ([-100, 0) ms) but not jaw-aligned (100 ms before the DLC jaw-opening onset, which
precedes the lick by ~90-190 ms), i.e. it is carried by the movement period, not by pre-movement activity.
Data: 095 (lick-aligned, 50-ms windows, 5-ms stride, -500 .. +200 ms; signed s = 2 AUC - 1 per reward-lick neuron for AH vs SL-B,
WH vs SL-B, WH vs AH; neurons selected on AH vs SL-A in [-100, 0) ms, so s_AS / s_WA near that window carry the AH-reuse bias of the
selection). Non-sensory neurons (081 class not "sensory"), learners rule; per session mean over >= 3 neurons; mean +- SEM over sessions.
Per time bin: R+ day 0 vs expert MWU, and the cohort x stage interaction (difference of stage differences) by mouse-level cohort
permutation (N_PERM); bins with p < 0.05 marked.
Also: the median jaw-onset lead per cohort x stage (118 events) drawn as the expected jaw-opening time.
Output: <OUTROOT>/single_neuron_signed/timecourse_<pop>.{png,pdf,svg}, timecourse_<pop>.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../124_signed_selectivity_timecourse.py [--pop learners|all]
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m95 = importlib.import_module("095_timeresolved_split_sl")
m51, m81 = m95.m51, m95.m81
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
OUT = m51.OUTROOT / "single_neuron_signed"
N_PERM = 2000
COH = {"R+": "#00B400", "R-": "#C800C8"}
PAIRS = [("WS", "s_WS (WH vs SL)"), ("WA", "s_WA (WH vs AH)"), ("AS", "s_AS (AH vs SL)")]
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]


def interaction_p(Y, coh, st, mouse, rng):
    """Y sessions x time; mouse-level cohort permutation of (E - L)_R+ - (E - L)_R-; two-sided p per time bin"""
    def stat(c):
        g = {k: Y[(c == k[0]) & (st == k[1])].mean(0) for k in GROUPS}
        return (g[GROUPS[1]] - g[GROUPS[0]]) - (g[GROUPS[3]] - g[GROUPS[2]])
    obs = stat(coh)
    mice = pd.unique(mouse); mc = pd.Series(coh, index=mouse).groupby(level=0).first().reindex(mice).to_numpy()
    idx = pd.Index(mice).get_indexer(mouse)
    cnt = np.zeros(Y.shape[1])
    for _ in range(N_PERM):
        cnt += np.abs(stat(rng.permutation(mc)[idx])) >= np.abs(obs) - 1e-12
    return obs, (1 + cnt) / (1 + N_PERM)


def main(a):
    plt = m62.setup()
    D = pd.read_parquet(m95.OUT / "selectivity_index.parquet"); S = np.load(m95.OUT / "selectivity.npy"); t = np.load(m95.OUT / "t.npy") * 1e3
    D = D.assign(row=np.arange(len(D)))
    D = D[D.nclass.isin(m81.CLASSES) & (D.nclass != "sensory") & D.cohort.isin(["R+", "R-"])]
    if a.pop == "learners":
        D = m61.learner_filter(D)
    E = pd.read_csv(m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}_jaw" / "jaw_onsets" / "events.csv").dropna(subset=["lead"])
    meta = D.drop_duplicates("session_id")[["session_id", "cohort", "stage"]]
    lead = E.merge(meta, on="session_id").groupby(["cohort", "stage", "session_id"]).lead.median().groupby(["cohort", "stage"]).median() * 1e3
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(len(PAIRS), 2, figsize=(4.6, 4.6), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.09, hspace=0.45, wspace=0.35)
    rows = []
    for i, (pair, lab) in enumerate(PAIRS):
        q = D[D.pair == pair]
        ses = q.groupby(["session_id", "mouse_id", "cohort", "stage"]).row.apply(list).reset_index()
        ses = ses[ses.row.apply(len) >= m81.MIN_CLASS]
        Y = np.stack([S[r].mean(0) for r in ses.row])
        coh, st, mouse = ses.cohort.to_numpy(), ses.stage.to_numpy(), ses.mouse_id.to_numpy()
        _, p_int = interaction_p(Y, coh, st, mouse, rng)
        for j, c in enumerate(["R+", "R-"]):
            ax = axs[i, j]
            for s_, ls in [("learning", (0, (2, 1.5))), ("expert", "-")]:
                m = (coh == c) & (st == s_)
                mu, se = Y[m].mean(0), Y[m].std(0, ddof=1) / np.sqrt(m.sum())
                ax.plot(t, mu, color=COH[c], ls=ls, lw=1.0, label=f"{'day 0' if s_ == 'learning' else 'expert'} (n={m.sum()})")
                ax.fill_between(t, mu - se, mu + se, color=COH[c], alpha=0.15 if s_ == "expert" else 0.08, lw=0)
                if (c, s_) in lead.index:
                    ax.axvline(-lead[(c, s_)], color=COH[c], lw=0.6, ls=ls, alpha=0.6)
            pL = np.array([mannwhitneyu(Y[(coh == c) & (st == "learning"), k], Y[(coh == c) & (st == "expert"), k]).pvalue for k in range(len(t))])
            lo = ax.get_ylim()[0]
            ax.scatter(t[pL < 0.05], np.full((pL < 0.05).sum(), lo), s=1.2, color=COH[c], marker="s", lw=0)
            if j == 0:
                ax.scatter(t[p_int < 0.05], np.full((p_int < 0.05).sum(), lo + 0.04 * (ax.get_ylim()[1] - lo)), s=1.2, color="0.2", marker="s", lw=0)
            ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.4", lw=0.5, ls=(0, (2, 2)))
            ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0)
            ax.set_title(f"{c.replace('-', '−')}: {lab}", fontsize=6, color=COH[c])
            if i == 0:
                ax.legend(frameon=False, fontsize=4.6, loc="upper left", handlelength=1.4)
            if i == len(PAIRS) - 1:
                ax.set_xlabel("Time from first lick (ms; 50-ms windows)")
            if j == 0:
                ax.set_ylabel("Mean signed s")
            for k in range(len(t)):
                rows.append(dict(pair=pair, cohort=c, t_ms=t[k], p_LvE=pL[k], p_interaction=p_int[k],
                                 mean_learning=Y[(coh == c) & (st == "learning"), k].mean(), mean_expert=Y[(coh == c) & (st == "expert"), k].mean()))
    fig.suptitle("Signed selectivity of non-sensory reward-lick neurons around the first lick\n"
                 "coloured squares: day 0 vs expert p < 0.05 (MWU); black: cohort × stage p < 0.05; vertical lines: median jaw-opening onset",
                 fontsize=5.6, y=0.995)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"timecourse_{a.pop}.{ext}", dpi=300)
    R = pd.DataFrame(rows); R.to_csv(OUT / f"timecourse_{a.pop}.csv", index=False)
    print("median jaw lead (ms)\n", lead.round(0).to_string())
    for pair in ["WS", "WA"]:
        for c in ["R+", "R-"]:
            r = R[(R.pair == pair) & (R.cohort == c)]
            sig = r[r.p_LvE < 0.05].t_ms
            print(pair, c, "LvE p<0.05 bins:", f"{sig.min():.0f}..{sig.max():.0f} ms ({len(sig)} bins)" if len(sig) else "none")
        r = R[(R.pair == pair) & (R.cohort == "R+")]; sig = r[r.p_interaction < 0.05].t_ms
        print(pair, "interaction p<0.05 bins:", list(sig.round(0).astype(int)) if len(sig) < 40 else f"{sig.min():.0f}..{sig.max():.0f} ({len(sig)})")
    print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(p.parse_args())
