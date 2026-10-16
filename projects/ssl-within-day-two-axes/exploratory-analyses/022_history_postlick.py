"""022 -- Trial history and post-lick licking vs the reward-lick CD score, day 0 vs expert (user 2026-10-16, "explore 1-4": items 1 and 2).
Events: 008 (both variants; held-out SL = 0, AH = 1). Behaviour from the NWB file: trials table (active context, perf != 6) and piezo lick times.
History of each event (trials started before the event): prev_rew = the previous trial was rewarded (lick_flag & reward_available);
log_t_rew = log10 of the time since the last reward (s); n_rew5 = rewards in the last 5 trials; prev_wh_lick = the mouse licked on the previous
whisker trial.
Post-lick behaviour of each event (from its first lick: trial start + RT for WH / AH; the bout onset for SL): n_licks1 = licks in [0, 1] s;
bout_dur = duration of the lick bout (licks separated by < 0.5 s).
Per session and class (WH; AH and SL for reference), OLS on z-scored predictors (>= MIN_EV events):
  history   score ~ tau + prev_rew + log_t_rew + prev_wh_lick       (coefficients = score units per SD / per level)
  post-lick n_licks1 ~ score + tau  and  bout_dur ~ score + tau     (coupling of the pre-lick score to the licking that follows)
Statistics: common.measure_stats on the per-session coefficients (Wilcoxon vs 0, day 0 vs expert, R+ vs R-, cohort x stage).
Output: tables/022_events.parquet, 022_sessions.csv, 022_stats.csv; figures/022_history_postlick_<variant>.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C

MIN_EV, BOUT_GAP = 8, 0.5
MEAS = [("h_prev_rew", "prev. trial rewarded → score"), ("h_log_t_rew", "log time since reward → score"), ("h_prev_wh_lick", "licked prev. whisker trial → score"),
        ("h_tau", "session time → score (history model)"), ("p_n_licks1", "score → licks in 1 s"), ("p_bout_dur", "score → bout duration")]


def behaviour(sid):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(C.m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        n = io.read(); tr = n.trials.to_dataframe()
        lk = np.sort(np.asarray(n.processing["behavior"].data_interfaces["BehavioralEvents"].time_series["piezo_lick_times"].data[:], float))
    act = tr[tr.context == "active"] if "context" in tr else tr.iloc[:0]
    tr = act if len(act) else tr                                   # older sessions have no context labels: all trials
    return tr[tr.perf != 6].sort_values("start_time"), lk


def session(args):
    sid, E = args
    try:
        tr, lk = behaviour(sid)
    except Exception:  # noqa: BLE001
        return None
    if len(tr) < 10:
        return None
    st = tr.start_time.to_numpy(); rew = ((tr.lick_flag == 1) & (tr.reward_available == 1)).to_numpy()
    iswh = (tr.trial_type == "whisker_trial").to_numpy(); lic = (tr.lick_flag == 1).to_numpy()
    E = E.copy()
    k = np.searchsorted(st, E.t.to_numpy() - 1e-3)                  # trials started before the event (the event's own trial excluded)
    E["prev_rew"] = np.where(k > 0, rew[np.clip(k - 1, 0, None)], np.nan)
    rt_ = st[rew]
    if len(rt_) == 0:
        return None
    j = np.searchsorted(rt_, E.t.to_numpy() - 1e-3)
    E["log_t_rew"] = np.where(j > 0, np.log10(np.maximum(E.t.to_numpy() - rt_[np.clip(j - 1, 0, None)], 0.1)), np.nan)
    E["n_rew5"] = [rew[max(0, i - 5):i].sum() if i > 0 else np.nan for i in k]
    wpos = np.where(iswh)[0]
    pw = [wpos[wpos < i] for i in k]
    E["prev_wh_lick"] = [float(lic[p[-1]]) if len(p) else np.nan for p in pw]
    first = np.where(E.cls.isin(["WH", "AH"]), E.t + E.rt, E.t)
    nl, bd = [], []
    for f in first:
        if not np.isfinite(f):
            nl.append(np.nan); bd.append(np.nan); continue
        a = lk[(lk >= f - 0.02) & (lk <= f + 3.0)]
        nl.append(float(np.sum(a <= f + 1.0)))
        if len(a) == 0:
            bd.append(0.0); continue
        gaps = np.where(np.diff(a) >= BOUT_GAP)[0]
        bd.append(float(a[gaps[0]] - a[0]) if len(gaps) else float(a[-1] - a[0]))
    E["n_licks1"], E["bout_dur"] = nl, bd
    return E


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [(x - x.mean()) / (x.std() or 1) for x in X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1:]


def coefs(E, v):
    out = {}
    w = E[E.cls == "WH"].dropna(subset=[f"cd_{v}", "prev_rew", "log_t_rew", "prev_wh_lick", "tau"])
    if len(w) >= MIN_EV and w.prev_rew.nunique() > 1:
        b = ols(w[f"cd_{v}"].to_numpy(), [w.tau.to_numpy(), w.prev_rew.to_numpy(float), w.log_t_rew.to_numpy(), w.prev_wh_lick.to_numpy(float)])
        out.update(h_tau=b[0], h_prev_rew=b[1], h_log_t_rew=b[2], h_prev_wh_lick=b[3])
    w = E[E.cls == "WH"].dropna(subset=[f"cd_{v}", "n_licks1", "bout_dur", "tau"])
    if len(w) >= MIN_EV:
        out["p_n_licks1"] = ols(w.n_licks1.to_numpy(), [w[f"cd_{v}"].to_numpy(), w.tau.to_numpy()])[0]
        out["p_bout_dur"] = ols(w.bout_dur.to_numpy(), [w[f"cd_{v}"].to_numpy(), w.tau.to_numpy()])[0]
    for c in ("AH", "SL"):
        w = E[E.cls == c].dropna(subset=[f"cd_{v}", "n_licks1", "tau"])
        if len(w) >= MIN_EV:
            out[f"p_n_licks1_{c}"] = ols(w.n_licks1.to_numpy(), [w[f"cd_{v}"].to_numpy(), w.tau.to_numpy()])[0]
    return out


def main(a):
    E0 = pd.read_parquet(C.TAB / "008_events.parquet")
    if a.replot:
        E = pd.read_parquet(C.TAB / "022_events.parquet")
    else:
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            res = pool.map(session, list(E0.groupby("session_id")))
        E = pd.concat([r for r in res if r is not None], ignore_index=True); E.to_parquet(C.TAB / "022_events.parquet", index=False)
    rows = []
    for v in ("corr", "raw"):
        for sid, g in E.groupby("session_id"):
            rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0], variant=v, **coefs(g, v)))
    S = C.population(pd.DataFrame(rows)); S.to_csv(C.TAB / "022_sessions.csv", index=False)
    R = pd.concat([pd.DataFrame(C.measure_stats(S[S.variant == v], m, "whole brain", m)).assign(variant=v) for v in ("corr", "raw") for m, _ in MEAS if m in S])
    R.to_csv(C.TAB / "022_stats.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    for v in ("corr", "raw"):
        fig = plt.figure(figsize=(7.6, 5.0)); gs = fig.add_gridspec(2, 4, wspace=0.6, hspace=1.3)
        for k, ((m, lab), l) in enumerate(zip(MEAS, "abcdef")):
            ax = fig.add_subplot(gs[k // 3, k % 3]); Rm = R[(R.variant == v) & (R.measure == m)]
            C.dot_panel(ax, S[S.variant == v], m, Rm, lab, l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
        # g: licks in 1 s vs pre-lick score, binned, per group (pooled events, score deciles)
        ax = fig.add_subplot(gs[0, 3])
        for (c, s_), ls in zip(C.GROUPS, ("--", "-", "--", "-")):
            d = E[(E.cohort == c) & (E.stage == s_) & (E.cls == "WH")].dropna(subset=[f"cd_{v}", "n_licks1"])
            if len(d) < 50:
                continue
            q = pd.qcut(d[f"cd_{v}"], 8, duplicates="drop"); g_ = d.groupby(q, observed=True)
            ax.plot(g_[f"cd_{v}"].mean(), g_.n_licks1.mean(), ls=ls, color=C.COH[c], marker="o", ms=2, lw=0.9, label=C.GLAB[(c, s_)])
        ax.set_xlabel("WH pre-lick CD score (octiles)"); ax.set_ylabel("licks in 1 s after the first lick"); ax.legend(frameon=False, fontsize=4.4)
        ax.set_title("pooled whisker hits", fontsize=5.6); ax.text(-0.4, 1.15, "g", transform=ax.transAxes, fontsize=8, weight="bold")
        ax = fig.add_subplot(gs[1, 3])
        for (c, s_), ls in zip(C.GROUPS, ("--", "-", "--", "-")):
            d = E[(E.cohort == c) & (E.stage == s_) & (E.cls == "WH")].dropna(subset=[f"cd_{v}", "prev_rew"])
            mm = d.groupby("prev_rew")[f"cd_{v}"].agg(["mean", "sem"])
            if len(mm) == 2:
                ax.errorbar([0, 1], mm["mean"], mm["sem"], color=C.COH[c], ls=ls, marker="o", ms=2.5, capsize=0, lw=0.9, label=C.GLAB[(c, s_)])
        ax.set_xticks([0, 1], ["prev. not\nrewarded", "prev.\nrewarded"]); ax.set_ylabel("WH CD score (pooled events)"); ax.set_xlim(-0.4, 1.4)
        ax.set_title("score by previous outcome", fontsize=5.6); ax.text(-0.4, 1.15, "h", transform=ax.transAxes, fontsize=8, weight="bold")
        fig.suptitle(f"Trial history and post-lick licking vs the whisker-hit reward-lick CD score ({'baseline-corrected' if v == 'corr' else 'raw'} rates; "
                     "a-f: per-session OLS coefficients, z-scored predictors)", fontsize=6.4, x=0.01, y=1.0, ha="left")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(C.FIG / f"022_history_postlick_{v}.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(R[(R.variant == "corr") & R.cohort.isin(["R+", "R-", "R+ vs R-"])][["measure", "cohort", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=16); ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
