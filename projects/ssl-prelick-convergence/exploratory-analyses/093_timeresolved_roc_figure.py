"""093 -- Figure of the time-resolved pre-lick ROC (092; 50-ms windows, 5-ms stride, centres -500 .. +200 ms from the first lick).
Exploratory (user 2026-10-10).
  a  converging non-sensory neurons (081 class non-sensory; pre-lick AH > SL and WH > SL significant): mean selectivity over time for
     AH vs SL, WH vs SL and WH vs AH; mean over neurons within session, mean +- s.e.m. over sessions; R+ / R- x day 0 / expert
  b  control: tested neurons that are not reward-lick neurons (random sample per session), same layout
  c  selectivity onset per neuron: first window (from -500 ms) from which s > THR for >= N_CONSEC consecutive windows (15 ms);
     per session the median onset of AH vs SL and of WH vs SL among converging non-sensory neurons; difference WH - AH onset
     (> 0: whisker-hit selectivity appears later); statistics as 088 (MWU brackets, cohort x stage permutation)
Learners rule. Output: across_days/<ref>/timeresolved_roc/timeresolved_roc.{png,pdf}, onsets.csv, stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../093_timeresolved_roc_figure.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m88 = importlib.import_module("088_cosyne_across_days_v2")
m51, m61, m62 = m88.m51, m88.m61, m88.m62
COH, GROUPS, GLAB = m88.COH, m88.GROUPS, m88.GLAB
OUT = m51.OUTROOT / "timeresolved_roc"
PCOL = {"AS": "#2c2cdb", "WS": "#f7b519", "WA": "0.45"}
PLAB = {"AS": "AH vs SL", "WS": "WH vs SL", "WA": "WH vs AH"}
THR, N_CONSEC = 0.2, 3


def onset(s, t):
    ok = (s > THR) & (t <= 0)
    run = np.convolve(ok.astype(int), np.ones(N_CONSEC, int), "valid") == N_CONSEC
    i = np.flatnonzero(run)
    return t[i[0]] if len(i) else np.nan


def insets(fig, spec, D, S, t, mask, title):
    sub = spec.subgridspec(2, 2, hspace=0.4, wspace=0.12); axs = []
    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j // 2, j % 2]); axs.append(ax)
        m = mask & (D.cohort == g[0]).to_numpy() & (D.stage == g[1]).to_numpy()
        n_s = 0
        for pr in ("AS", "WS", "WA"):
            mm = m & (D.pair == pr).to_numpy()
            if not mm.any():
                continue
            per = pd.DataFrame(S[mm]).groupby(D.session_id.to_numpy()[mm]).mean()
            n_s = len(per); mu, se = per.mean(0).to_numpy(), per.sem(0).to_numpy()
            ax.fill_between(t * 1e3, mu - se, mu + se, color=PCOL[pr], alpha=0.25, lw=0); ax.plot(t * 1e3, mu, color=PCOL[pr], lw=0.9, label=PLAB[pr])
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.5, lw=0, zorder=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        ax.axhline(0, color="0.85", lw=0.4); ax.set_xlim(-500, 200)
        ax.set_title(f"{GLAB[g]} ({n_s})", color=COH[g[0]], fontsize=4.8, pad=1.5)
        ax.set_xticks([-400, -200, 0]); ax.set_xticklabels(["−400", "−200", "0"] if j >= 2 else [])
    yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
    for j, a in enumerate(axs):
        a.set_ylim(*yl); a.tick_params(labelsize=4.4)
        if j % 2:
            a.set_yticklabels([])
    axs[0].set_ylabel("selectivity (2 AUC − 1)", fontsize=4.8); axs[2].set_ylabel("selectivity (2 AUC − 1)", fontsize=4.8)
    axs[2].set_xlabel("time from first lick (ms)", fontsize=4.8)
    axs[0].legend(frameon=False, fontsize=4.0, loc="upper left", handlelength=1.0)
    return axs


def main():
    plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(OUT / "selectivity_index.parquet"); S = np.load(OUT / "selectivity.npy"); t = np.load(OUT / "t.npy")
    keep = D.index.isin(m61.learner_filter(D).index)
    D, S = D[keep].reset_index(drop=True), S[keep]
    conv_ns = (D.conv.fillna(False) & (D.nclass == "non-sensory")).to_numpy()
    ctrl = (~D.rl.fillna(False)).to_numpy()
    # onsets
    O = []
    for pr in ("AS", "WS"):
        mm = conv_ns & (D.pair == pr).to_numpy()
        on = np.array([onset(s, t) for s in S[mm]])
        O.append(D[mm][["session_id", "mouse_id", "cohort", "stage", "electrode_group", "cluster_id"]].assign(pair=pr, onset=on * 1e3))
    O = pd.concat(O); O.to_csv(OUT / "onsets.csv", index=False)
    P = O.pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "electrode_group", "cluster_id"], columns="pair", values="onset").reset_index()
    P["d"] = P.WS - P.AS
    Ses = P.groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(AS=("AS", "median"), WS=("WS", "median"), d=("d", "median"),
                                                                       n=("d", lambda x: x.notna().sum())).reset_index()
    Ses = Ses[Ses.n >= 3]
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.36))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.3, 1, 1], wspace=0.55, left=0.06, right=0.99, top=0.8, bottom=0.17)
    insets(fig, gs[0, 0], D, S, t, conv_ns, "a"); fig.text(0.06, 0.92, "a  Converging, non-sensory neurons", fontsize=5.6)
    insets(fig, gs[0, 1], D, S, t, ctrl, "b"); fig.text(0.36, 0.92, "b  Control: not reward-lick neurons", fontsize=5.6)
    ax = fig.add_subplot(gs[0, 2]); ax.set_box_aspect(1)
    for (c, s_), x in zip(GROUPS, (0, 1, 2.4, 3.4)):
        d = Ses[(Ses.cohort == c) & (Ses.stage == s_)]
        for pr, dx in (("AS", -0.12), ("WS", 0.12)):
            ax.errorbar(x + dx, d[pr].mean(), d[pr].sem(), fmt="o", ms=3, color=PCOL[pr], mfc="white" if s_ == "learning" else PCOL[pr], capsize=0, lw=0.9)
    ax.set_xticks([0, 1, 2.4, 3.4], [f"{l}\n({((Ses.cohort == g[0]) & (Ses.stage == g[1])).sum()})" for l, g in zip(["D0", "E", "D0", "E"], GROUPS)], fontsize=4.6)
    ax.set_ylabel("median selectivity onset (ms from lick)"); ax.set_title(f"Onset (s > {THR}, {N_CONSEC * 5} ms)\nAH vs SL (blue), WH vs SL (yellow)", fontsize=5.2)
    for x, c in [(0.5, "R+"), (2.9, "R-")]:
        ax.annotate(c.replace("-", "−"), xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -17), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color=COH[c], weight="bold")
    ax = fig.add_subplot(gs[0, 3]); ax.set_box_aspect(1)
    m88.mean_panel2(ax, Ses, "d", "onset WH − AH (ms)", rng, "093 d", "Whisker-hit onset relative\nto auditory-hit onset", ref=0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"timeresolved_roc.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("093")]); St.to_csv(OUT / "stats.csv", index=False)
    print(Ses.groupby(["cohort", "stage"])[["AS", "WS", "d"]].agg(["mean", "count"]).round(1))
    print(St[["mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "interaction_p_perm", "R+ L vs E p_MWU"]].round(3))
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
