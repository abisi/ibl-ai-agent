"""119 -- COSYNE figure in three rows (user 2026-10-12 layout; 115 kept). Plot only; learners rule (learners on day 0, all mice at expert);
session = unit; 088 mean_panel2 statistics (day 0 vs expert per cohort, R+ vs R- per stage, Mann-Whitney; Welch in the csv;
cohort x stage by mouse-level permutation; n under the ticks).
Row 1, stimulus side: each session's hit - miss axis (5-35 ms after stimulus onset, active whisker trials; hitmiss 135 with >= 4 hits
       and misses, 175), passive responses projected on it (per neuron), real and linear-shifted axes
  a  schematic slot (blank)
  b  passive whisker pre / post x day 0 / expert, real vs shifted
  c  the same for passive auditory responses
  d  within-session change post - pre of the passive whisker projection, real - shifted, day 0 vs expert
  e  the same for auditory responses
Row 2, single neurons (lick-aligned, corrected first lick; SL = spontaneous lick-bout onsets)
  f  population PSTHs of all tested neurons per cohort x stage (088 / 083 b)
  g  example non-sensory reward-lick neuron (088 pick: converging, non-sensory, R+ expert): stimulus-aligned (whisker misses, whisker
     hits, auditory hits) and lick-aligned (WH, AH, SL) PSTHs: no stimulus response without a lick, a pre-lick ramp for rewarded licks
  h  whisker-hit modulation index MI of reward-lick neurons, session means (088 e): MI = (|m_WH - m_SL| - |m_WH - m_AH|) /
     (|m_WH - m_SL| + |m_WH - m_AH|), +1 AH-like, -1 SL-like
  i  per area group: change of the session-mean MI (expert - day 0) per cohort; sessions with >= 3 reward-lick neurons in the area;
     * BH q < 0.05 of the per-area cohort x stage interaction, (*) p < 0.05
Row 3, population on the reward-lick CD (087: AH - SL, 5-fold CV, fold-wise z-scoring, per neuron; [-100, 0) ms before the lick)
  j  schematic
  k  session-mean projections of WH, AH and SL on the CD (088 f), day 0 open, expert filled
  l  angle theta between WH - SL and the CD
  m  per area: change of d_WS = p_WH - p_SL (expert - day 0)
  n  per area: change of theta
Output: across_days/<ref>/publication/learners/COSYNE_three_rows.{png,pdf,svg} + _stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../119_cosyne_three_rows.py [--n-proc 30]
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m88 = importlib.import_module("088_cosyne_across_days_v2")
m115 = importlib.import_module("115_cosyne_combined")
m51, m61, m62, m81, m83 = m88.m51, m88.m61, m88.m62, m88.m81, m88.m83
COH, CL, GROUPS = m88.COH, m88.CL, m88.GROUPS
GREY = "0.55"


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def four(ax, D, k, title):
    xs = [0, 1, 2.4, 3.4]
    for j, c in enumerate(("R+", "R-")):
        for i, s_ in enumerate(("learning", "expert")):
            d = D[(D.cohort == c) & (D.stage == s_)]; x = np.array(xs[2 * i: 2 * i + 2]) + (j - 0.5) * 0.14
            for kind, col, ls in (("null", GREY, (0, (2, 1.5))), ("real", COH[c], "-")):
                v = d[[f"pre_{k}_{kind}", f"post_{k}_{kind}"]]
                ax.errorbar(x, v.mean(), v.sem(), fmt="o", ls=ls, ms=2.6, color=col, lw=1.0, capsize=0, mfc=col if s_ == "expert" else "white", mew=0.7)
            ax.text(xs[2 * i] + 0.5, 0.94 - 0.08 * j, f"n = {len(d)}", color=COH[c], fontsize=4.2, ha="center", transform=ax.get_xaxis_transform())
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(xs, ["pre", "post", "pre", "post"], fontsize=4.6)
    for x0, s_ in ((0.5, "day 0"), (2.9, "expert")):
        ax.text(x0, -0.27, s_, ha="center", fontsize=4.8, transform=ax.get_xaxis_transform())
    ax.set_ylabel("projection on hit − miss axis\n(per neuron)")
    ax.set_title(f"{title}\n(grey: shifted axes)", fontsize=5.2)


def example_neuron(fig, spec, r, rng):
    """stimulus-aligned (W miss, W hit, A hit) and lick-aligned (WH, AH, SL) PSTHs of one unit"""
    from pynwb import NWBHDF5IO
    from scipy.ndimage import gaussian_filter1d
    with NWBHDF5IO(str(m51.NWB / f"{r.session_id}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = m51.ru.process_nwb_tables(nwb)
        tr = nwb.trials.to_dataframe()
        t_all, log = m51.active_trials(tr, r.session_id)
        t_l, _ = m51.select_trials(tr, r.session_id)
        sl = m51.spontaneous_licks(nwb, tr, log["epoch"])
    u = units[(units.electrode_group.astype(str) == str(r.electrode_group)) & (units.cluster_id.astype(str) == str(r.cluster_id))]
    s = np.sort(np.asarray(u.spike_times.iloc[0]))
    lick = t_all.lick_flag == 1
    stim = {"whisker miss": t_all.start_time[(t_all.trial_type == "whisker_trial") & ~lick].to_numpy(),
            "whisker hit": t_all.start_time[(t_all.trial_type == "whisker_trial") & lick].to_numpy(),
            "auditory hit": t_all.start_time[(t_all.trial_type == "auditory_trial") & lick].to_numpy()}
    lk = {"WH": t_l.first_lick_time[t_l.cls == "WH"].to_numpy(), "AH": t_l.first_lick_time[t_l.cls == "AH"].to_numpy(), "SL": sl}
    colors = {"whisker miss": "0.6", "whisker hit": CL["WH"], "auditory hit": CL["AH"], "WH": CL["WH"], "AH": CL["AH"], "SL": CL["FA"]}
    sub = spec.subgridspec(1, 2, wspace=0.15)
    axs = [fig.add_subplot(sub[0]), fig.add_subplot(sub[1])]
    top = 0
    for ax, ev, win, xl in ((axs[0], stim, (-0.1, 0.3), "from stimulus (ms)"), (axs[1], lk, (-0.4, 0.2), "from lick (ms)")):
        edges = np.arange(win[0], win[1] + 0.005, 0.01); tc = (edges[:-1] + 0.005) * 1e3
        for k, e in ev.items():
            if len(e) < 3:
                continue
            H = np.array([np.histogram(s[np.searchsorted(s, x + win[0]):np.searchsorted(s, x + win[1])] - x, edges)[0] for x in e]) / 0.01
            H = gaussian_filter1d(H.astype(float), 2, axis=1); mu, se = H.mean(0), H.std(0) / np.sqrt(len(H))
            ax.fill_between(tc, mu - se, mu + se, color=colors[k], alpha=0.25, lw=0); ax.plot(tc, mu, color=colors[k], lw=0.9, label=f"{k} ({len(e)})")
            top = max(top, np.nanmax(mu + se))
        ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2))); ax.set_xlabel(xl, fontsize=4.6); ax.tick_params(labelsize=4.4)
        ax.legend(frameon=False, fontsize=3.4, loc="upper left", handlelength=1.0, borderaxespad=0.05, labelspacing=0.15)
    axs[1].axvspan(-100, 0, color="#FDD49E", alpha=0.5, lw=0, zorder=0)
    for ax in axs:
        ax.set_ylim(0, top * 1.35)
    axs[1].tick_params(labelleft=False); axs[0].set_ylabel("spikes/s", fontsize=4.8)
    return axs


def pick_clean(Wc, n_cand=8):
    """example non-sensory reward-lick neuron: among the top n_cand converging non-sensory R+ expert good units (088 score, >= 5 Hz),
    the one with the smallest whisker-miss response (0-150 ms after the stimulus minus [-100, 0) ms) relative to its pre-lick AH rate"""
    from pynwb import NWBHDF5IO
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    G = Wc[(Wc.quality_label == "good")].copy()
    G["fr"] = G[f"fr_window:{af}"]; G["score"] = (G[f"sel:{af}"].abs() + G[f"sel:{wf}"].abs()) * np.log1p(G.fr)
    C = G[G.conv.fillna(False) & (G.nclass == "non-sensory") & (G.cohort == "R+") & (G.stage == "expert") & (G.fr >= 5)].nlargest(n_cand, "score")
    best, val = None, np.inf
    for r in C.itertuples():
        with NWBHDF5IO(str(m51.NWB / f"{r.session_id}.nwb"), "r", load_namespaces=True) as io:
            nwb = io.read(); units, _ = m51.ru.process_nwb_tables(nwb); t, _ = m51.active_trials(nwb.trials.to_dataframe(), r.session_id)
        u = units[(units.electrode_group.astype(str) == str(r.electrode_group)) & (units.cluster_id.astype(str) == str(r.cluster_id))]
        s = np.sort(np.asarray(u.spike_times.iloc[0])); e = t.start_time[(t.trial_type == "whisker_trial") & (t.lick_flag != 1)].to_numpy()
        if len(e) < 5:
            continue
        cnt = lambda lo, hi: (np.searchsorted(s, e + hi) - np.searchsorted(s, e + lo)).mean() / (hi - lo)
        v = abs(cnt(0.0, 0.15) - cnt(-0.1, 0.0)) / max(r.fr, 1e-9)
        if v < val:
            best, val = Wc.loc[r.Index], v
    return best


def sc_area_change(ax, U, col, xlab, title, rng, rows, panel):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    S = U[U.rl & U[col].notna()].groupby(["session_id", "mouse_id", "cohort", "stage", "area_group"]).agg(v=(col, "mean"), n=(col, "size")).reset_index()
    S = S[S.n >= 3]
    n = S.groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    order = [g for g in au.get_area_group_custom_order() if g in n.index and all(n.loc[g].get(k, 0) >= 3 for k in GROUPS)]
    area_forest(ax, S.rename(columns={"area_group": "area"}), "v", order, xlab, title, rng, rows, panel, gcol)


def area_forest(ax, S, col, order, xlab, title, rng, rows, panel, gcol):
    y = np.arange(len(order))[::-1]; pv = []
    for i, g in enumerate(order):
        d = S[S.area == g].dropna(subset=[col])
        for k, c in enumerate(("R+", "R-")):
            a, b = (d[(d.cohort == c) & (d.stage == s_)][col] for s_ in ("learning", "expert"))
            ax.errorbar(b.mean() - a.mean(), y[i] + 0.16 - 0.32 * k, xerr=np.sqrt(a.sem() ** 2 + b.sem() ** 2), fmt="o", ms=2.6,
                        color=COH[c], lw=0.8, capsize=0, mfc="white")
            rows.append(dict(panel=panel, area=g, cohort=c, change=b.mean() - a.mean(), n_day0=len(a), n_expert=len(b)))
        pv.append(m62.group_stats(d, col, f"{panel} area {g}", rng, f"119 {panel} area")["interaction"][1])
    qv = bh(pv)
    for i, (g, p_, q_) in enumerate(zip(order, pv, qv)):
        rows.append(dict(panel=f"{panel} q", area=g, p_perm=p_, q_BH=q_))
        mk = "*" if q_ < 0.05 else ("(*)" if p_ < 0.05 else "")
        if mk:
            ax.text(0.99, y[i], mk, transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", ha="right", weight="bold")
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [m83.m62_short(g) for g in order], fontsize=4.4)
    for tl, g in zip(ax.get_yticklabels(), order):
        tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel(xlab, fontsize=4.8)
    ax.set_title(f"{title}\ncohort × stage: * q < 0.05, (*) p < 0.05", fontsize=4.8, pad=2)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    plt = m62.setup(); rng = np.random.default_rng(0); rows = []
    HMD = pd.read_csv(m115.HM / "175_across_days_passive_projection_sessions.csv")
    # single neurons (as 088 main)
    MI = m88.all_mi(a.n_proc)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(U0, f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(U0, f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    for k in ("electrode_group", "cluster_id"):
        U0[k] = U0[k].astype(str)
    U0 = U0.merge(MI, on=["session_id", "electrode_group", "cluster_id"], how="left")
    D = m62.load_population("learners"); W = D["W"].copy()
    for k in ("electrode_group", "cluster_id"):
        W[k] = W[k].astype(str)
    PB, KB, tc = m62.load_psth_prestart(W)
    KB = KB.assign(electrode_group=KB.electrode_group.astype(str), cluster_id=KB.cluster_id.astype(str))
    W = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    Wc = W.merge(U0[["session_id", "electrode_group", "cluster_id", "nclass", "rl", "conv", "mi"]], on=["session_id", "electrode_group", "cluster_id"], how="left")
    U = m61.learner_filter(U0)
    if "area_group" not in U:
        U = U.merge(W[["session_id", "electrode_group", "cluster_id", "area_group"]].drop_duplicates(), on=["session_id", "electrode_group", "cluster_id"], how="left")
    S = U[U.rl & U.nclass.isin(m81.CLASSES)].groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(mi=("mi", "mean"), n=("mi", "size")).reset_index()
    S = S[S.n >= m81.MIN_CLASS]
    PA = m61.learner_filter(pd.read_csv(m88.H087)); P = PA[PA.area == "whole_brain"]
    with plt.rc_context({"font.size": 5.2, "axes.titlesize": 5.2, "axes.labelsize": 5.0, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6,
                         "legend.fontsize": 4.4, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.66))
        G = fig.add_gridspec(3, 1, hspace=0.75, left=0.06, right=0.985, top=0.92, bottom=0.07)
        r1 = G[0].subgridspec(1, 5, wspace=0.75); r2 = G[1].subgridspec(1, 4, wspace=0.55, width_ratios=[1.1, 1.3, 1, 1]); r3 = G[2].subgridspec(1, 5, wspace=0.75)
        L1, L2, L3 = [], [], []
        # row 1
        ax = fig.add_subplot(r1[0]); ax.axis("off"); ax.set_box_aspect(1); L1.append(ax)
        ax = fig.add_subplot(r1[1]); ax.set_box_aspect(1); four(ax, HMD, "W", "Passive whisker"); L1.append(ax)
        ax = fig.add_subplot(r1[2]); ax.set_box_aspect(1); four(ax, HMD, "A", "Passive auditory"); L1.append(ax)
        for j, (k, tt) in enumerate((("dW_ex", "Within session, whisker"), ("dA_ex", "Within session, auditory"))):
            ax = fig.add_subplot(r1[3 + j]); ax.set_box_aspect(1); L1.append(ax)
            m88.mean_panel2(ax, HMD, k, "post − pre, real − shifted", rng, f"119 {k}", tt, ref=0)
        # row 2
        tested = (Wc["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wc["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
        ib = m83.psth_insets(fig, r2[0], Wc, PB, tc, tested, "f", rows); L2.append(ib[0])
        ex = pick_clean(Wc)
        ia = example_neuron(fig, r2[1], ex, rng); L2.append(ia[0])
        ia[0].set_title(f"Non-sensory reward-lick neuron (R+ expert, {ex.get('area_acronym_custom', ex.get('area_group', ''))}, MI = {ex.mi:+.2f})",
                        fontsize=5.0, loc="left", pad=3)
        rows.append(dict(panel="g", session_id=ex.session_id, electrode_group=ex.electrode_group, cluster_id=ex.cluster_id, mi=ex.mi))
        ax = fig.add_subplot(r2[2]); ax.set_box_aspect(1); L2.append(ax)
        m88.mean_panel2(ax, S, "mi", "MI (−1 SL-like … +1 AH-like)", rng, "119 mi", "Whisker-hit modulation index\n(reward-lick neurons)", ref=0)
        ax = fig.add_subplot(r2[3]); L2.append(ax)
        sc_area_change(ax, U, "mi", "Δ MI (expert − day 0)", "Per area: modulation index", rng, rows, "i")
        # row 3
        ax = fig.add_subplot(r3[0]); ax.set_box_aspect(1); m115.panel_cd(ax); L3.append(ax)
        ax = fig.add_subplot(r3[1]); ax.set_box_aspect(1); m88.proj_levels(ax, P); L3.append(ax)
        ax = fig.add_subplot(r3[2]); ax.set_box_aspect(1); L3.append(ax)
        m88.mean_panel2(ax, P, "theta", "θ (°)", rng, "119 theta", "Angle θ of WH − SL to the CD", ref=90)
        LA = PA[PA.area != "whole_brain"]
        Q = m62.area_qualify(LA.rename(columns={"area": "region"}), col="d_WS")
        order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] and g in Q["R-"]]
        for j, (col, xl, tt, pn) in enumerate((("d_WS", "Δ WH − SL projection (per neuron)", "Per area: WH position on the CD", "m"),
                                              ("theta", "Δ θ (°)", "Per area: angle θ", "n"))):
            ax = fig.add_subplot(r3[3 + j]); L3.append(ax)
            area_forest(ax, LA, col, order, xl, tt, rng, rows, pn, gcol)
        from matplotlib.lines import Line2D
        L3[1].legend([Line2D([], [], color=CL[k_], lw=1.2, marker="o", ms=2.4) for k_ in ("WH", "AH", "FA")],
                     ["WH", "AH", "SL"], frameon=False, fontsize=4.2, ncol=1, loc="center right", handlelength=1.2)
        m62.letter_row(fig, L1, "abcde", dx_in=0.28, dy_in=0.16); m62.letter_row(fig, L2, "fghi", dx_in=0.28, dy_in=0.16)
        m62.letter_row(fig, L3, "jklmn", dx_in=0.28, dy_in=0.16)
        fig.suptitle("Reward learning routes the whisker stimulus to a go state and recruits whisker licks into the reward-lick code",
                     x=0.01, y=0.998, ha="left", va="top", fontsize=6.6, weight="bold")
        out = m51.OUTROOT / "publication" / "learners"; out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"COSYNE_three_rows.{ext}", dpi=300)
    R = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith("119")]); R.to_csv(out / "COSYNE_three_rows_stats.csv", index=False)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("119") and "area" not in str(r.get("panel", ""))])
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning", "n_R− expert",
              "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print(pd.DataFrame([r for r in rows if str(r.get("panel", "")).endswith(" q")]).round(4).to_string())
    print("ALL DONE", out / "COSYNE_three_rows.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    main(ap.parse_args())
