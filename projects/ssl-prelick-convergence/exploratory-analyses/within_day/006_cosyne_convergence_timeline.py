"""COSYNE summary, compact single-row layout (5 panels): in R+ mice, convergence of whisker hits (WH) toward auditory hits
(AH) begins within the learning day and continues across days; R- diverge on day 0. Reference: spontaneous licks (SL).

  a  First-lick-aligned PSTHs (004): rows R+ / R-, columns day 0 early half, day 0 late half, expert.
  b  Trial level (002): WH - SL score on each session's SL -> AH axis across the learning-day session; expert level.
  c  Day-0 slope of the WH - SL score per session (002), R+ vs R-.
  d  Common footing (003, 4 events per class): whole-session decoder readout per session half.
  e  Contrasts of d: within day 0, across days (early phase), carry-over.
Writes the figure and a detailed caption (COSYNE_convergence_timeline_caption.md) with the statistics of this run.
Output: combined_results_ks4/ssl-prelick-convergence/within_day/<ref>/cosyne/
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent                                         # across-day scripts (projects merged 2026-10-07)
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
BASE = m51.WITHIN
OUT = BASE / "cosyne"
COH, CL, CLAB = m62.COH, m62.CL, m62.CLAB
EPOCH_DIR = "epochs_n4_u150x10" if (BASE / "epochs_n4_u150x10" / "all" / "epoch_contrasts.csv").exists() else "epochs_n4"
rng = np.random.default_rng(0)
P_ = lambda p: m62.fmt_p(p)



def mixed_text():
    """plain-language sentence on the single-trial mixed model (009), day 0 and expert, reference = spontaneous licks"""
    f = BASE / "mixed_model" / "mixed_model_terms.csv"
    if not f.exists():
        return ""
    r = pd.read_csv(f).set_index(["stage", "reference"])
    out = []
    for st, lab in [("learning", "On the learning day"), ("expert", "within expert sessions")]:
        if (st, "spontaneous licks") not in r.index:
            continue
        x = r.loc[(st, "spontaneous licks")]
        out.append(f"{lab}, whisker hits moved toward auditory hits by {x.drift_Rplus:+.2f} (R+) and {x.drift_Rminus:+.2f} (R−) "
                   f"of the spontaneous-lick-to-auditory-hit distance over one session (R+ vs R−: shuffling cohort labels across "
                   f"mice, {P_(x.p_perm_diff)})")
    return ("**Single-trial mixed model** (every lick event of every session; model: CD projection ~ whisker hit × time in "
            "session × cohort, with a separate baseline, time trend and whisker-hit offset per session). " + "; ".join(out) + ".")


def mixed():
    """R+ vs R- difference of the day-0 whisker-hit drift (reference: spontaneous licks) from the mixed model (009)"""
    f = BASE / "mixed_model" / "mixed_model_terms.csv"
    if not f.exists():
        return ""
    r = pd.read_csv(f)
    r = r[(r.stage == "learning") & (r.reference == "spontaneous licks")]
    return "" if r.empty else f"R+ vs R−: {P_(r.p_perm_diff.iloc[0])}"


def main():
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    MM = mixed()
    with plt.rc_context({"font.size": 5.2, "axes.titlesize": 5.3, "axes.labelsize": 5.1, "xtick.labelsize": 4.7,
                         "ytick.labelsize": 4.7, "legend.fontsize": 4.5, "axes.titlepad": 2.5}):
        fig = plt.figure(figsize=(m62.W_IN, 2.5))
        gs = fig.add_gridspec(1, 5, width_ratios=[2.15, 1.35, 0.72, 1.3, 1.0], wspace=0.55, left=0.055, right=0.99,
                              top=0.8, bottom=0.2)
        # a: PSTH grid (R+ / R- x day 0 early, day 0 late, expert)
        ga = gs[0].subgridspec(2, 3, hspace=0.42, wspace=0.08)
        z = np.load(BASE / "psth_halves" / "psth_halves.npz", allow_pickle=True)
        S = pd.read_csv(BASE / "psth_halves" / "sessions.csv")
        P, Nev, edges = z["psth"], z["n"], z["edges"]
        tc = (edges[:-1] + np.diff(edges) / 2) * 1e3
        sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")
        axs_a = np.empty((2, 3), object)
        for i, c in enumerate(["R+", "R-"]):
            for j, (stg, halves, ttl) in enumerate([("learning", [0], "D0 early"), ("learning", [1], "D0 late"),
                                                    ("expert", [0, 1], "Expert")]):
                ax = fig.add_subplot(ga[i, j]); axs_a[i, j] = ax
                msk = ((S.cohort == c) & (S.stage == stg)).to_numpy()
                ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.4, ls=(0, (2, 2)))
                for k, cl in enumerate(m51.CLASSES):
                    ok = msk & (Nev[:, k, halves].min(1) >= 3)
                    A = np.array([sm(np.nanmean(p[halves], 0)) for p in P[ok, k]])
                    if len(A) < 2:
                        continue
                    mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(tc, mu - se, mu + se, color=CL[cl], alpha=0.25, lw=0)
                    ax.plot(tc, mu, color=CL[cl], lw=0.75, label={"WH": "WH", "AH": "AH", "FA": "SL"}[cl])
                R[f"a_n_{c}_{ttl}"] = int(msk.sum())
                ax.set_xlim(-400, 200); ax.set_xticks([-300, 0])
                ax.set_xticklabels(["−300", "0"] if i == 1 else [])
                ax.set_title(f"{c.replace('-', '−')} {ttl}", color=COH[c], fontsize=4.9)
                if j == 0:
                    ax.set_ylabel("Δ rate (Hz)", fontsize=4.9)
        axs_a[0, 0].legend(frameon=False, loc="upper left", handlelength=0.8, borderaxespad=0.0, fontsize=4.2, labelspacing=0.15)
        yl = (min(a.get_ylim()[0] for a in axs_a.ravel()), max(a.get_ylim()[1] for a in axs_a.ravel()))
        for i in range(2):
            for j in range(3):
                axs_a[i, j].set_ylim(*yl)
                if j:
                    axs_a[i, j].set_yticklabels([])
        axs_a[1, 1].set_xlabel("Time from first lick (ms)", fontsize=4.9)
        # b: trial-level WH - SL across the learning-day session
        TR = pd.read_csv(BASE / "slopes" / "trial_slopes_trajectories.csv")
        D2 = pd.read_csv(BASE / "slopes" / "trial_slopes_sessions.csv")
        md = TR[TR.axis == "md"].pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "bin"], columns="cls",
                                             values="score").reset_index()
        md["rel"] = md["WH"] - md["FA"]
        xb = (np.arange(5) + 0.5) / 5
        ax_b = fig.add_subplot(gs[1])
        for c in ["R+", "R-"]:
            g = md[(md.cohort == c) & (md.stage == "learning")]
            q = g.groupby("bin").rel.agg(["mean", "sem"]).reindex(range(5))
            ax_b.fill_between(xb, q["mean"] - q["sem"], q["mean"] + q["sem"], color=COH[c], alpha=0.15, lw=0)
            ax_b.plot(xb, q["mean"], color=COH[c], lw=1.1, marker="o", ms=2.2, mfc="white",
                      label=f"{c.replace('-', '−')} day 0 ({g.session_id.nunique()})")
            e = md[(md.cohort == c) & (md.stage == "expert")].groupby("session_id").rel.mean()
            ax_b.errorbar(1.13, e.mean(), e.sem(), fmt="o", ms=3.4, color=COH[c], capsize=0, lw=0.9, clip_on=False)
            R[f"b_{c}"] = dict(first=q["mean"].iloc[0], last=q["mean"].iloc[-1], expert=e.mean(), n_day0=g.session_id.nunique(),
                               n_expert=len(e))
        ax_b.text(1.13, 1.0, "expert", transform=ax_b.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.4)
        ax_b.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_b.set_xlim(0, 1.2); ax_b.set_xticks([0, 0.5, 1])
        ax_b.set_xlabel("Time in day-0 session"); ax_b.set_ylabel("CD projection, WH − SL\n(SL = 0, AH = 1)")
        ax_b.set_title("Reward-lick coding direction, day 0")
        ax_b.legend(frameon=False, loc="lower left", borderaxespad=0.1)
        # c: day-0 slopes
        ax_c = fig.add_subplot(gs[2])
        col = "md_WH-FA_slope"
        d0 = D2[D2.stage == "learning"].dropna(subset=[col])
        for k_, c in enumerate(["R+", "R-"]):
            v = d0[d0.cohort == c][col]
            ax_c.scatter(k_ + rng.uniform(-0.13, 0.13, len(v)), v, s=3, color=COH[c], alpha=0.45, lw=0)
            ax_c.errorbar(k_ + 0.3, v.mean(), v.sem(), fmt="o", ms=3.4, color=COH[c], mfc="white", capsize=0, lw=0.9)
            R[f"c_{c}"] = dict(mean=v.mean(), sem=v.sem(), n=len(v), p=stats.wilcoxon(v).pvalue)
        a_, b_ = d0[d0.cohort == "R+"][col], d0[d0.cohort == "R-"][col]
        _, pp = m001.perm_interaction(d0.assign(stage="learning"), col, kind="learning")
        R["c_test"] = dict(p_perm=pp, p_MWU=stats.mannwhitneyu(a_, b_).pvalue, p_Welch=stats.ttest_ind(a_, b_, equal_var=False).pvalue)
        lo, hi = np.nanpercentile(d0[col], [1, 99])
        ax_c.set_ylim(lo - 0.15 * (hi - lo), hi + 0.3 * (hi - lo))
        yb = hi + 0.1 * (hi - lo)
        ax_c.plot([0, 0, 1, 1], [yb, yb + 0.04 * (hi - lo), yb + 0.04 * (hi - lo), yb], color="0.2", lw=0.5)
        ax_c.text(0.5, yb + 0.06 * (hi - lo), P_(pp), ha="center", fontsize=4.5)
        ax_c.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_c.set_xticks([0, 1], ["R+", "R−"]); ax_c.set_xlim(-0.45, 1.55)
        for t, c in zip(ax_c.get_xticklabels(), ["R+", "R-"]):
            t.set_color(COH[c])
        ax_c.set_ylabel("WH drift along CD, β"); ax_c.set_title("Day-0 drift" + (f"\nmixed model {MM}" if MM else ""), fontsize=5.0)
        # d: decoder readout per session half (common footing)
        E = pd.read_csv(BASE / EPOCH_DIR / "all" / "epoch_contrasts.csv")
        meas = "dec"
        ax_d = fig.add_subplot(gs[3])
        xe = {"L-early": 0, "L-late": 1, "E-early": 2.3, "E-late": 3.3}
        for k_, c in enumerate(["R+", "R-"]):
            q = E[(E.measure == meas) & (E.cohort == c) & (E.kind == "epoch")].set_index("name").reindex(list(xe))
            dx = (k_ - 0.5) * 0.16
            for seg in (("L-early", "L-late"), ("E-early", "E-late")):
                ax_d.plot([xe[s_] + dx for s_ in seg], q.loc[list(seg), "value"], color=COH[c], lw=1.0)
            ax_d.plot([xe["L-late"] + dx, xe["E-early"] + dx], q.loc[["L-late", "E-early"], "value"], color=COH[c], lw=0.6, ls=(0, (2, 2)))
            for ep, x in xe.items():
                ax_d.errorbar(x + dx, q.loc[ep, "value"], [[q.loc[ep, "value"] - q.loc[ep, "lo"]], [q.loc[ep, "hi"] - q.loc[ep, "value"]]],
                              fmt="o", ms=3.0, color=COH[c], mfc="white" if ep.startswith("L") else COH[c], lw=0.8, capsize=0,
                              label=c.replace("-", "−") if ep == "L-early" else None)
            R[f"d_{c}"] = {ep: (q.loc[ep, "value"], q.loc[ep, "lo"], q.loc[ep, "hi"]) for ep in xe}
            R[f"d_n_{c}"] = {ep: int(E[(E.measure == meas) & (E.cohort == c) & (E.kind == "epoch") & (E.name == ep)].n_sessions.iloc[0])
                             for ep in xe}
        ax_d.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_d.set_xticks(list(xe.values()), ["early", "late", "early", "late"])
        for x_, s_ in ((0.5, "Day 0"), (2.8, "Expert")):
            ax_d.annotate(s_, xy=(x_, 0), xycoords=ax_d.get_xaxis_transform(), xytext=(0, -12), textcoords="offset points",
                          ha="center", va="top", fontsize=4.8)
        ax_d.set_ylabel("P(AH|WH) − P(AH|SL) − chance"); ax_d.set_title("Decoder, per half")
        ax_d.legend(frameon=False, loc="upper left", borderaxespad=0.1)
        # e: contrasts
        ax_e = fig.add_subplot(gs[4])
        names = ["within-day", "across-day (early)", "carry-over"]
        for k_, c in enumerate(["R+", "R-"]):
            q = E[(E.measure == meas) & (E.cohort == c) & (E.kind == "contrast")].set_index("name").reindex(names)
            x = np.arange(len(names)) + (k_ - 0.5) * 0.3
            for xi, (_, r) in zip(x, q.iterrows()):
                ax_e.errorbar(xi, r.value, [[r.value - r.lo], [r.hi - r.value]], fmt="o", ms=3.0, color=COH[c], lw=0.8, capsize=0,
                              mfc=COH[c] if (r.p_boot if np.isfinite(r.p_boot) else 1) < 0.05 else "white")
            R[f"e_{c}"] = {n_: (r.value, r.lo, r.hi, r.p_boot) for n_, (_, r) in zip(names, q.iterrows())}
        pp_ = E[(E.measure == meas) & (E.cohort == "R+ - R-")].set_index("name").reindex(names)
        R["e_perm"] = {n_: pp_.loc[n_, "p_perm"] for n_ in names}
        for i, n_ in enumerate(names):
            ax_e.text(i, 1.01, P_(pp_.loc[n_, "p_perm"]).replace("p = ", "").replace("p < ", "<"),
                      transform=ax_e.get_xaxis_transform(), ha="center", fontsize=4.3)
        ax_e.axhline(0, color="0.3", lw=0.4)
        ax_e.set_xticks(range(3), ["within\nday 0", "across\ndays", "carry-\nover"], fontsize=4.5)
        ax_e.set_ylabel("Change in decoder readout"); ax_e.set_title("Changes", pad=7)
        m62.letter_row(fig, [axs_a[0, 0], ax_b, ax_c, ax_d, ax_e], "abcde", dx_in=0.28, dy_in=0.13)
        fig.suptitle("R+ whisker hits converge toward auditory hits within the learning day and across days; R− diverge on day 0",
                     x=0.055, y=0.995, ha="left", fontsize=6.4, weight="bold")
        m62.save(fig, OUT, "COSYNE_convergence_timeline"); plt.close(fig)
    caption(R)
    print("ALL DONE", OUT)


def caption(R):
    f3 = lambda v: f"{v:+.3f}".replace("-", "−")
    ci = lambda t: f"{f3(t[0])} [{f3(t[1])}, {f3(t[2])}]"
    d, e = R["d_R+"], R["d_R-"]
    txt = f"""# COSYNE_convergence_timeline

**R+ whisker hits converge toward auditory hits within the learning day and across days; R− diverge on day 0.**
**In short.** For every session we find the population direction that separates rewarded licks after the auditory tone from unrewarded spontaneous licks (the reward-lick coding direction, CD), and ask where whisker-triggered licks fall on it. In R+ mice, whose whisker licks are rewarded, whisker licks move toward the rewarded-lick end of the CD during the learning session and further across days; in R− mice they move back toward the unrewarded end.

{mixed_text()}

Pre-lick window: 100 ms before the corrected first lick. Event classes: whisker hit (WH), auditory hit (AH) and the
unrewarded-lick reference, spontaneous licks (SL; licks outside trials). Equations below. Trials: active, perf ≠ 6, auditory warm-up
removed, end-of-session disengagement trimmed (rule A1). Units: Kilosort 4, quality good or mua, mean pre-lick rate
≥ 0.1 Hz. Cohort: per mouse from the reference sheet. Learning = day 0 (D0), expert = later days. Session halves are
split at the midpoint between the two middle auditory hits. Unit of analysis: session; cohort comparisons by permuting
cohort labels across mice.

**Definitions and equations.** For a session with units z-scored on training folds, let x_i be the pre-lick population
vector of event i (100 ms before the first lick).
- Reward-lick coding direction (CD): CD = (mean_{{AH, train}} x − mean_{{SL, train}} x) / ‖·‖ (5-fold cross-validation over
  the session's AH and SL; WH never used to build it).
- CD projection of event i, normalised so that the session's SL = 0 and AH = 1:
  c_i = (x_i·CD − mean_{{SL}} x·CD) / (mean_{{AH}} x·CD − mean_{{SL}} x·CD)  (held-out AH / SL; fold-averaged WH; d′ ≥ 0.3).
- Within-session drift of WH along CD (relative to SL): fit c_i = a_k + b_k τ_i separately for k ∈ {{WH, SL}} (τ_i = normalised
  time of event i in the session, 0 → 1); drift β = b_WH − b_SL (one value per session). β > 0: during the session WH
  move toward AH along CD faster than the shared drift of all events; β_AH = b_WH − b_AH uses AH as the reference.
- Decoder readout of an epoch e: D_e = mean_{{i ∈ WH_e}} p_i − mean_{{i ∈ SL_e}} p_i − median_null(D_e), where p_i = P(AH | x_i)
  from an L2 logistic regression trained on the session's AH vs SL (cross-validated), and the null re-fits the decoder
  after linearly shifting the activity against the time-ordered labels (40 shifts).
- Distance difference: Δd = d(WH, SL) − d(WH, AH), with the cross-validated squared distance per unit
  d(X, Y) = (X̄_a − Ȳ_a)·(X̄_b − Ȳ_b) / n_units over random trial halves a, b (unbiased; 0 for identical means).
- Contrasts: within day 0 = D(D0 late) − D(D0 early); across days = D(E early) − D(D0 early); carry-over = D(E early) − D(D0 late).

**a**, First-lick-aligned population PSTHs of R+ (top) and R− (bottom) mice: day-0 early half, day-0 late half and
expert sessions (both halves). Per unit, rate minus the event's baseline (1 s before trial start for hits,
[−1, −0.5] s before the lick for SL), averaged over units within each session, then mean ± s.e.m. over sessions
(10 ms bins, 30 ms boxcar); shaded: pre-lick window. Sessions: R+ D0 {R['a_n_R+_D0 early']}, expert {R['a_n_R+_Expert']};
R− D0 {R['a_n_R-_D0 early']}, expert {R['a_n_R-_Expert']}. In R+, pre-lick WH activity moves from between SL and AH
toward AH across day 0 and overlaps AH in experts; in R− it stays near SL.
**b**, Single trials on the learning day. Every event is scored on the session's own cross-validated SL → AH
mean-difference axis (5-fold over AH and SL; SL = 0, AH = 1; held-out d′ ≥ 0.3); plotted: mean WH score minus mean SL
score in five equal bins of normalised session time (mean ± s.e.m. over sessions), which removes the session-wide drift
of all event types along the axis. Right: expert sessions (session mean). R+: {f3(R['b_R+']['first'])} → {f3(R['b_R+']['last'])}
(expert {f3(R['b_R+']['expert'])}); R−: {f3(R['b_R-']['first'])} → {f3(R['b_R-']['last'])} (expert {f3(R['b_R-']['expert'])}).
**c**, Per-session slope of the WH − SL score against normalised time on day 0 (OLS of WH scores minus OLS of SL scores).
R+ {f3(R['c_R+']['mean'])} ± {R['c_R+']['sem']:.3f} (n = {R['c_R+']['n']}, Wilcoxon {P_(R['c_R+']['p'])}); R−
{f3(R['c_R-']['mean'])} ± {R['c_R-']['sem']:.3f} (n = {R['c_R-']['n']}, {P_(R['c_R-']['p'])}); R+ vs R−: mouse permutation
{P_(R['c_test']['p_perm'])}, Mann-Whitney {P_(R['c_test']['p_MWU'])}, Welch {P_(R['c_test']['p_Welch'])}.
**d**, Common footing across days: in every session (day 0 and expert) each half is estimated from exactly 4 WH, 4 AH
and 4 SL events (20 random subsamples) and 150 random units. One decoder per session (L2 logistic regression, C = 0.05,
5-fold cross-validation over the session's AH and SL; WH never in training) is read out per half as
P(AH | WH) − P(AH | SL), minus the median of the same readout under a whole-session linear-shift null (40 shifts).
Mean over mice and 95% hierarchical-bootstrap CI (mice, then sessions). R+: D0 early {ci(d['L-early'])}, D0 late
{ci(d['L-late'])}, expert early {ci(d['E-early'])}, expert late {ci(d['E-late'])}; R−: D0 early {ci(e['L-early'])},
D0 late {ci(e['L-late'])}, expert early {ci(e['E-early'])}, expert late {ci(e['E-late'])}. Sessions: R+ D0
{R['d_n_R+']['L-early']}, expert {R['d_n_R+']['E-early']}; R− D0 {R['d_n_R-']['L-early']}, expert {R['d_n_R-']['E-early']}.
**e**, Changes of the readout in d (95% CI; filled: bootstrap p < 0.05): within day 0 (late − early), across days in
the same session phase (expert early − D0 early) and carry-over (expert early − D0 late). R+: within
{ci(R['e_R+']['within-day'][:3])}, across {ci(R['e_R+']['across-day (early)'][:3])}, carry-over
{ci(R['e_R+']['carry-over'][:3])}; R−: within {ci(R['e_R-']['within-day'][:3])}, across
{ci(R['e_R-']['across-day (early)'][:3])}, carry-over {ci(R['e_R-']['carry-over'][:3])}. Top: R+ vs R− (cohort labels
permuted across mice): within {P_(R['e_perm']['within-day'])}, across {P_(R['e_perm']['across-day (early)'])},
carry-over {P_(R['e_perm']['carry-over'])}.
"""
    (OUT / "COSYNE_convergence_timeline_caption.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
