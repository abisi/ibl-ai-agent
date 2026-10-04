"""012 -- (A) smoothness (random-walk step size sigma): per-session vs a single shared value, and
sigma distributions across cohorts; (B) the change-point learning trial explained visually on
example sessions, next to a learning trial read off the smoothed curves; (C) change point vs
curve-based rule across all sessions (user request 2026-09-25).

(A) For every session and trial type (whisker, no-stim), the log evidence log p(licks | sigma) of
    the original learning-curve model (random walk on logit lick probability) on a grid of 36 sigma
    values (lt_lib, exact). Per-session best sigma = evidence-weighted mean (broad prior); shared
    sigma = the value maximising the SUMMED log evidence across sessions (per cohort, and overall).
    sigma vs mean lick rate shows the logit-scale confound (rates near 0/1 carry little information
    about logit changes).
(B) Change point (joint whisker + FA, 005): raw licks with the MAP segment rates; cumulative lick
    counts (a change point = a bend in slope); per-candidate score (log posterior over the LT) and
    posterior; next to the smoothed discrimination curve D(t) = p_whisker(t) - p_FA(t) with the
    half-way rule (L7): R+ first trial where D crosses half-way between its start (first 10 trials)
    and plateau (90th percentile) and stays above on 16/20 trials; R- mirrored (half-way between the
    early peak and the floor, stays below 16/20); no LT if the change is < 0.2.
(C) L6 (change point) vs L7 (curve rule) for all sessions.
Outputs: exploratory-analyses/012_sigma.png, 012_changepoint_explained.png, 012_cp_vs_curve.png,
         artifacts/012_sigma_evidence.csv
Run on haas.
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lt_lib as L  # noqa: E402

ART = HERE.parent / "artifacts"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
SIG = np.exp(np.linspace(np.log(0.02), np.log(1.5), 36))
EXAMPLES = [("AB119", "R+ learner"), ("MH028", "R+ learns by FA dropping"), ("AB085", "R- gradual learner")]


def evidence_profile(item):
    sid, d = item
    out = {}
    for name, y in (("whisker", d["w_outcomes"]), ("fa", d["n_outcomes"])):
        out[name] = np.array([L.forward_backward(np.asarray(y).astype(int), s)[1] for s in SIG])
    return sid, out


def part_a(inputs, prof):
    rows = []
    for sid, d in inputs.items():
        for tt in ("whisker", "fa"):
            lz = prof[sid][tt]
            w = np.exp(lz - logsumexp(lz))
            y = d["w_outcomes"] if tt == "whisker" else d["n_outcomes"]
            rows.append(dict(session_id=sid, reward_group=d["reward_group"], trial_type=tt, sigma=float(w @ SIG),
                             sigma_map=float(SIG[np.argmax(lz)]), mean_rate=float(np.mean(y)), n=len(y)))
    df = pd.DataFrame(rows)
    ev = []
    for tt in ("whisker", "fa"):
        for grp in ("R+", "R-", "all"):
            sids = [s for s in inputs if grp == "all" or inputs[s]["reward_group"] == grp]
            tot = np.sum([prof[s][tt] for s in sids], axis=0)
            ev.append(dict(trial_type=tt, group=grp, shared_sigma=float(SIG[np.argmax(tot)]), n_sessions=len(sids),
                           log_evidence_loss_vs_per_session=float(np.sum([prof[s][tt].max() for s in sids]) - tot.max())))
    evd = pd.DataFrame(ev)
    evd.to_csv(ART / "012_sigma_evidence.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    for k, tt in enumerate(("whisker", "fa")):
        ax = axes[k]
        g = df[df.trial_type == tt]
        a, b = g[g.reward_group == "R+"].sigma, g[g.reward_group == "R-"].sigma
        for j, rg in enumerate(("R+", "R-")):
            v = g[g.reward_group == rg].sigma
            ax.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=16, color=COHORT_COLOR[rg], alpha=0.75)
            ax.plot([j - 0.3, j + 0.3], [v.median()] * 2, color="k")
            ax.text(j, v.median() * 1.3, f"median {v.median():.2f}", ha="center", fontsize=8)
        sh = evd[(evd.trial_type == tt)].set_index("group").shared_sigma
        for grp, ls in (("R+", "--"), ("R-", ":"), ("all", "-")):
            ax.axhline(sh[grp], color="#888888", ls=ls, lw=1, label=f"best shared sigma, {grp}: {sh[grp]:.2f}")
        pm_, pw_ = mannwhitneyu(a, b).pvalue, ttest_ind(np.log(a), np.log(b), equal_var=False).pvalue
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["R+", "R-"])
        ax.set_ylabel("sigma (logit / trial), per session")
        ax.set_title(f"{'Whisker' if tt == 'whisker' else 'False-alarm (no-stim)'} curve: smoothness chosen by the data\n"
                     f"R+ vs R-: MW p={pm_:.2g}, Welch(log) p={pw_:.2g}", fontsize=9.5)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[2]
    for tt, mk in (("whisker", "o"), ("fa", "^")):
        g = df[df.trial_type == tt]
        for rg in ("R+", "R-"):
            gg = g[g.reward_group == rg]
            ax.scatter(gg.mean_rate, gg.sigma, s=18, marker=mk, color=COHORT_COLOR[rg], alpha=0.7,
                       label=f"{rg} {tt}")
    r_ = spearmanr(df.mean_rate, df.sigma)
    ax.set_yscale("log")
    ax.set_xlabel("session mean lick rate (that trial type)")
    ax.set_ylabel("sigma")
    ax.set_title(f"Confound: sigma vs lick rate (all curves, Spearman r = {r_.statistic:.2f}, p = {r_.pvalue:.1g})\n"
                 "rates near 0 carry little information about logit changes -> small sigma", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[3]
    for tt, ls in (("whisker", "-"), ("fa", "--")):
        for grp in ("R+", "R-"):
            sids = [s for s in inputs if inputs[s]["reward_group"] == grp]
            tot = np.sum([prof[s][tt] for s in sids], axis=0)
            ax.plot(SIG, tot - tot.max(), color=COHORT_COLOR[grp], ls=ls, label=f"{grp} {tt}")
    ax.set_xscale("log")
    ax.set_ylim(-60, 2)
    ax.set_xlabel("shared sigma")
    ax.set_ylabel("summed log evidence (0 = best)")
    ax.set_title("A single sigma for all sessions: summed evidence across sessions\n(peak = best shared value)", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Smoothness of the learning curves (random-walk step size): per-session values chosen by the data vs a "
                 "single shared value", fontsize=12)
    fig.savefig(HERE / "012_sigma.png", dpi=160)
    return df, evd


def part_b(inputs, curves, cps, table):
    sel = [(next(k for k in inputs if k.startswith(p)), lab) for p, lab in EXAMPLES]
    fig, axes = plt.subplots(len(sel), 4, figsize=(26, 4.4 * len(sel)), constrained_layout=True, squeeze=False)
    for r, (sid, lab) in enumerate(sel):
        d, r_cp, row = inputs[sid], cps[sid], table.loc[sid]
        rg = d["reward_group"]
        col = COHORT_COLOR[rg]
        o, wt, no, nt = d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"]
        n = len(o)
        x = np.arange(n)
        nx = np.clip(np.searchsorted(wt, nt), 0, n - 1) - 0.5    # no-stim trials placed on the whisker axis
        k, j = int(r_cp["map_k"]), int(r_cp["map_j"])
        bounds = [0, k, j, n] if (rg == "R+" and j < n) else [0, k, n]
        names = ["naive", "learned", "decline"] if rg == "R+" else ["generalising", "learned"]

        ax = axes[r, 0]
        ax.scatter(x[o == 1], np.full((o == 1).sum(), 1.06), marker="|", s=40, color=col)
        ax.scatter(x[o == 0], np.full((o == 0).sum(), -0.06), marker="|", s=40, color="#cccccc")
        ax.scatter(nx[no == 1], np.full((no == 1).sum(), 1.14), marker="|", s=40, color="#555555")
        for a0, a1, nm in zip(bounds[:-1], bounds[1:], names):
            sel_n = (nx >= a0 - 0.5) & (nx < a1 - 0.5)
            wr, fr = o[a0:a1].mean(), no[sel_n].mean() if sel_n.any() else np.nan
            ax.hlines(wr, a0, a1, color=col, lw=3)
            ax.hlines(fr, a0, a1, color="#555555", lw=3, ls="--")
            ax.text((a0 + a1) / 2, min(wr, 0.9) + 0.05, f"{nm}\nwhisker {wr:.2f}\nFA {fr:.2f}", ha="center", fontsize=7.5)
        for b in bounds[1:-1]:
            ax.axvline(b, color="k", lw=1)
        ax.set_ylim(-0.15, 1.25)
        ax.set_ylabel(f"{sid[:14]}\n{rg} -- {lab}", fontsize=8.5)
        ax.set_title("1. RAW LICKS + best segmentation: ticks = whisker lick (colour),\nno lick (grey), FA lick (dark); "
                     "lines = whisker (solid) / FA (dashed) rate", fontsize=8.5, loc="left")

        ax = axes[r, 1]
        cw = np.cumsum(o)
        order = np.argsort(nt)
        cf = np.cumsum(no[order])
        ax.plot(x, cw, color=col, lw=1.8, label="cumulative whisker licks")
        ax.plot(nx[order], cf, color="#555555", lw=1.5, ls="--", label="cumulative FA licks")
        for b in bounds[1:-1]:
            ax.axvline(b, color="k", lw=1)
        for a0, a1 in zip(bounds[:-1], bounds[1:]):
            ax.plot([a0, a1], [cw[a0] if a0 else 0, cw[a1 - 1]], color=col, lw=0.8, ls=":")
        ax.set_title("2. CUMULATIVE LICK COUNTS (slope = lick rate): change point =\nwhere the whisker slope bends vs the "
                     "FA slope", fontsize=8.5, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left")

        ax = axes[r, 2]
        pk = r_cp["pk"]
        with np.errstate(divide="ignore"):
            lp = np.log10(np.where(pk > 0, pk / pk.max(), np.nan))
        ax.plot(x, lp, color="#1f77b4", lw=1.2)
        ax.set_ylim(-6, 0.3)
        ax.set_ylabel("log10 relative score of LT = k", fontsize=8)
        ax2 = ax.twinx()
        ax2.fill_between(x, 0, pk, color="#1f77b4", alpha=0.35, step="mid")
        ax2.set_ylabel("posterior P(LT = k)", fontsize=8, color="#1f77b4")
        ax.axvspan(r_cp["ci05"], r_cp["ci95"], color="#1f77b4", alpha=0.08)
        ax.axvline(r_cp["median"], color="#1f77b4", lw=1.6)
        cat = row.L6_category
        ax.set_title(f"3. SCORE OF EVERY CANDIDATE LT -> POSTERIOR\nLT = median {r_cp['median']:.0f}, 90% CI [{r_cp['ci05']:.0f}, "
                     f"{r_cp['ci95']:.0f}]; log10 BF vs no change = {r_cp['log10_bf']:+.1f}; category: {cat}", fontsize=8.5,
                     loc="left")

        ax = axes[r, 3]
        e = curves[sid]["eb"]
        D = e["p_mean"] - e["fa_time"]
        ax.plot(x, e["p_mean"], color=col, lw=1.2, alpha=0.6, label="whisker (smoothed)")
        ax.plot(x, e["fa_time"], color="#555555", lw=1.1, ls="--", alpha=0.7, label="FA (smoothed)")
        ax.plot(x, D, color="#8c564b", lw=2, label="D = whisker - FA")
        if rg == "R+":
            d0, dhi = D[:10].mean(), np.percentile(D, 90)
            thr = d0 + 0.5 * (dhi - d0)
            ax.axhline(d0, color="#8c564b", ls=":", lw=0.8)
            ax.axhline(dhi, color="#8c564b", ls=":", lw=0.8)
        else:
            i_max = int(np.argmax(D[:30]))
            dhi, dlo = D[i_max], np.percentile(D[i_max:], 10)
            thr = dlo + 0.5 * (dhi - dlo)
            ax.axhline(dhi, color="#8c564b", ls=":", lw=0.8)
            ax.axhline(dlo, color="#8c564b", ls=":", lw=0.8)
        ax.axhline(thr, color="#ff7f0e", ls="--", lw=1)
        for val, c_, ls, name in ((row.L0_stored, "#d62728", "-", "stored"), (row.L6, "#1f77b4", "-", "change point (L6)"),
                                  (row.L7, "#ff7f0e", "-", "curve rule (L7)")):
            if not pd.isna(val):
                ax.axvline(val, color=c_, ls=ls, lw=1.6, label=f"{name} = {val:.0f}")
        ax.set_ylim(-0.4, 1.05)
        ax.set_title("4. CURVE RULE on smoothed D = whisker - FA: dotted = start/plateau,\norange dashed = half-way; "
                     "LT = first crossing held 16/20", fontsize=8.5, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        for a in axes[r]:
            a.set_xlim(-1, n)
            a.tick_params(labelsize=7)
    for a in axes[-1]:
        a.set_xlabel("whisker trial")
    fig.suptitle("How the change-point learning trial works (panels 1-3), next to a learning trial read off the smoothed "
                 "curves (panel 4)", fontsize=12)
    fig.savefig(HERE / "012_changepoint_explained.png", dpi=160)


def part_c(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for k, rg in enumerate(("R+", "R-")):
        ax = axes[k]
        g = table[table.reward_group == rg]
        both = g.L6.notna() & g.L7.notna()
        ax.scatter(g.L7[both], g.L6[both], s=24, color=COHORT_COLOR[rg], label=f"both defined (n={both.sum()})")
        only6, only7 = g.L6.notna() & g.L7.isna(), g.L7.notna() & g.L6.isna()
        ax.scatter(np.full(only6.sum(), -8), g.L6[only6], s=18, marker="<", color="#1f77b4",
                   label=f"change point only (n={only6.sum()})")
        ax.scatter(g.L7[only7], np.full(only7.sum(), -8), s=18, marker="v", color="#ff7f0e",
                   label=f"curve rule only (n={only7.sum()})")
        if both.any():
            ax.errorbar(g.L7[both], g.L6[both], yerr=[g.L6[both] - g.L6_ci05[both], g.L6_ci95[both] - g.L6[both]], fmt="none",
                        ecolor=COHORT_COLOR[rg], alpha=0.35)
        lim = np.nanmax([g.L6.max(), g.L7.max()]) + 10
        ax.plot([0, lim], [0, lim], color="#888888", ls="--")
        med = np.median(np.abs(g.L6[both] - g.L7[both])) if both.any() else np.nan
        ax.set_xlabel("curve rule (half-way on smoothed D), L7")
        ax.set_ylabel("change point (L6), with 90% CI")
        ax.set_title(f"{rg}: change point vs curve rule; median |difference| = {med:.0f} trials\nneither: "
                     f"{(g.L6.isna() & g.L7.isna()).sum()} sessions", fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Learning trial from the change point vs from the smoothed curves, all sessions", fontsize=11)
    fig.savefig(HERE / "012_cp_vs_curve.png", dpi=160)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    cps = pickle.load(open(ART / "005_cp_posteriors.pkl", "rb"))
    table = pd.read_csv(ART / "007_learning_trials_v2.csv").set_index("session_id")
    with ProcessPoolExecutor(max_workers=24) as ex:
        prof = dict(ex.map(evidence_profile, inputs.items()))
    df, evd = part_a(inputs, prof)
    print(df.groupby(["trial_type", "reward_group"]).sigma.describe().round(3).to_string())
    print(evd.round(3).to_string())
    part_b(inputs, curves, cps, table)
    part_c(table.reset_index())
    print("done")


if __name__ == "__main__":
    main()
