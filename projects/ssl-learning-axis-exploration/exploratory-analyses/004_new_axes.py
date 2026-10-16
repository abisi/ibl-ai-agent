"""004 -- New coding-direction definitions to investigate learning (user 2026-10-16: "do A and C" of the brainstorm).
Rates: ssl-prelick-convergence 105 per-event matrices (baseline-corrected spikes/s): stimulus-aligned 50-ms windows (centres 25..175 ms, NaN
after the trial's first lick) for whisker trials W (hits and misses), auditory hits AH, correct rejections CR, false alarms FA; lick-aligned
50-ms windows (centres -400..+100 ms) for WH, AH, SL. Units: stable good / mua of the session (prelick_units), whole brain. Every axis is
unit-norm, built on units scaled by the training events' SD, with 5-fold cross-validation over the events that define it: those events are
projected held-out, all other events by the fold-averaged projection; projections per neuron (/ sqrt N).
A1 reward expectation   AH vs CR in [50, 150) ms after the stimulus, orthogonalised to the AH - CR difference in [0, 50) ms (fast sensory
                        response). Whisker hits and misses projected in [50, 150) ms; normalised so that held-out CR = 0, AH = 1:
                        e_WH, e_WM (does the whisker cue acquire the auditory cue's reward-predicting state?), e_WH - e_WM.
A2 outcome              AH vs SL in [+25, +125) ms after the lick (bins 50, 75, 100), orthogonalised to the pre-lick AH - SL axis
                        ([-100, 0) ms); WH projected; normalised SL = 0, AH = 1: o_WH (R+ whisker hits are rewarded, R- are not);
                        o_sep = held-out AH - SL separation on the axis (per neuron; size of the outcome response).
A3 alignment            hit - miss axis of whisker trials and whisker - auditory identity axis in [0, 50) ms; hit - miss axis in [50, 150) ms
                        vs the A1 axis; cosines from disjoint random halves of the events, normalised by split-half reliabilities
                        (floor 0.05, 50 splits, clipped +-1.5): cos_HI_early, cos_HE.
C7 orthogonalised CD    the pre-lick reward-lick CD (AH - SL, [-100, 0) ms) with the motor axis M (mean SL pre-lick activity, baseline-
                        corrected and scaled, i.e. the direction of preparing any lick) and the sensory axis S (W - AH in [0, 50) ms)
                        projected out. Lick-aligned time course of lambda(t) = (WH - SL) / (AH - SL) on the original and the orthogonalised CD
                        (denominator: window separation), as 021; lam_orth_win = lambda in [-100, 0) ms; var_kept = |CD_orth|^2.
C8 motor axis           WH and AH relative to SL on M in [-100, 0) ms (m_WH, m_AH; per neuron): cue-specific movement preparation.
Statistics: common.measure_stats per session quantity (Wilcoxon vs 0, day 0 vs expert, R+ vs R-, cohort x stage); time courses: Mann-Whitney
day 0 vs expert per window, BH over windows.
Output: tables/004_sessions.csv, 004_stats.csv, 004_timecourse.csv, 004_tc_stats.csv; figures/004_A_axes.{png,pdf,svg}, 004_C_orth.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C

m002 = importlib.import_module("002_trial_slopes")
m105 = importlib.import_module("105_trial_rates")
MIN_UNITS, MIN_EV, N_SPLIT, FLOOR, SEP_MIN = 10, 6, 50, 0.05, 1e-3
MEAS = [("e_WH", "A1: whisker hits on reward-expectation axis (CR = 0, AH = 1)"), ("e_WM", "A1: whisker misses on reward-expectation axis"),
        ("e_WH_WM", "A1: hits − misses on reward-expectation axis"), ("o_WH", "A2: whisker hits on outcome axis (SL = 0, AH = 1)"),
        ("o_sep", "A2: AH − SL separation on outcome axis"), ("cos_HI_early", "A3: cos(hit − miss, whisker − auditory), 0–50 ms"),
        ("cos_HE", "A3: cos(hit − miss, reward expectation), 50–150 ms"), ("lam_win", "C7: λ on original CD, [−100, 0) ms"),
        ("lam_orth_win", "C7: λ on orthogonalised CD"), ("var_kept", "C7: fraction of the CD kept after orthogonalisation"),
        ("m_WH", "C8: WH − SL on motor axis"), ("m_AH", "C8: AH − SL on motor axis")]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def orth(u, others):
    for o in others:
        o = unit(o)
        if np.all(np.isfinite(o)):
            u = u - (u @ o) * o
    return unit(u)


def cv_axis(F, ia, ib, rest, seed, make=None):
    """F: events x units (window means). Axis = mean(a) - mean(b) on training events (units scaled by the training SD), optionally passed
    through make(u, sd, train) (orthogonalisation); held-out projections for a / b, fold average for rest. Returns proj (per neuron), axes."""
    from sklearn.model_selection import StratifiedKFold
    ev = np.r_[ia, ib]; y = np.r_[np.ones(len(ia)), np.zeros(len(ib))]
    p = np.full(F.shape[0], np.nan); pr = []; axes = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(ev, y):
        A = F[ev[tr]]; sd = np.nanstd(A, 0); sd[~(sd > 0)] = 1
        u = unit(np.nanmean(A[y[tr] == 1] / sd, 0) - np.nanmean(A[y[tr] == 0] / sd, 0))
        if make is not None:
            u = make(u, sd, ev[tr])
        axes.append(u / sd)                                         # axis in raw units (applied to unscaled rates)
        p[ev[te]] = (F[ev[te]] / sd) @ u
        if len(rest):
            pr.append((F[rest] / sd) @ u)
    if len(rest):
        p[rest] = np.nanmean(pr, 0)
    return p / np.sqrt(F.shape[1]), axes


def norm01(p, i0, i1):
    a, b = np.nanmean(p[i0]), np.nanmean(p[i1])
    return (p - a) / (b - a) if abs(b - a) > SEP_MIN else p * np.nan


def cos_norm(Xa, ia, ib, Xb, ja, jb, rng):
    """reliability-normalised cosine between axis a (Xa: a vs b events) and axis b; disjoint random halves"""
    def ax(X, i, j):
        sd = np.nanstd(X[np.r_[i, j]], 0); sd[~(sd > 0)] = 1
        return unit(np.nanmean(X[i] / sd, 0) - np.nanmean(X[j] / sd, 0))
    cn, r1, r2 = [], [], []
    for _ in range(N_SPLIT):
        h = lambda idx: (lambda q: (q[: len(q) // 2], q[len(q) // 2:]))(rng.permutation(idx))
        (a1, a2), (b1, b2), (c1, c2), (d1, d2) = h(ia), h(ib), h(ja), h(jb)
        if min(len(a1), len(b1), len(c1), len(d1), len(a2), len(b2), len(c2), len(d2)) < 2:
            continue
        u1, u2, v1, v2 = ax(Xa, a1, b1), ax(Xa, a2, b2), ax(Xb, c1, d1), ax(Xb, c2, d2)
        cn.append(u1 @ v2); r1.append(u1 @ u2); r2.append(v1 @ v2)
    if not cn:
        return np.nan
    return float(np.clip(np.nanmean(cn) / np.sqrt(max(np.nanmean(r1), FLOOR) * max(np.nanmean(r2), FLOOR)), -1.5, 1.5))


def session(args):
    sid, meta, W_s, seed = args
    f = m105.OUT / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True); rng = np.random.default_rng(seed)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))).merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy()
    if ok.sum() < MIN_UNITS:
        return None
    SB = np.transpose(z["stimbins"][ok].astype(float), (1, 0, 2)); sc = z["stim_c"].astype(float); sc = sc * (1e3 if sc.max() < 5 else 1)
    LB = np.transpose(z["lickbins"][ok].astype(float), (1, 0, 2)); lc = z["lick_c"].astype(float); lc = lc * (1e3 if lc.max() < 5 else 1)
    sl_, ll = z["stim_lab"].astype(str), z["lick_lab"].astype(str); hit = z["is_hit"].astype(bool)
    early = np.nanmean(SB[:, :, np.isclose(sc, 25)], 2)                          # [0, 50) ms
    late = np.nanmean(SB[:, :, (sc >= 75) & (sc <= 125)], 2)                    # [50, 150) ms (truncated at the lick)
    pre = np.nanmean(LB[:, :, (lc > -100) & (lc < 0)], 2)                       # [-100, 0) ms before the lick
    post = np.nanmean(LB[:, :, (lc >= 50) & (lc <= 100)], 2)                    # [+25, +125) ms after the lick
    iW, iAH, iCR = np.where(sl_ == "W")[0], np.where(sl_ == "AH")[0], np.where(sl_ == "CR")[0]
    iWH, iWM = iW[hit[iW]], iW[~hit[iW]]
    jWH, jAH, jSL = np.where(ll == "WH")[0], np.where(ll == "AH")[0], np.where(ll == "SL")[0]
    out = dict(meta, n_units=int(ok.sum()), n_WH=len(iWH), n_WM=len(iWM), n_CR=len(iCR))
    # A1 reward expectation: AH vs CR late, orthogonalised to AH - CR early
    if min(len(iAH), len(iCR)) >= MIN_EV:
        okl = np.isfinite(late).all(1)
        Fl = np.nan_to_num(late); Fe = np.nan_to_num(early)
        def mk(u, sd, tr):
            a, b = [t for t in tr if t in set(iAH)], [t for t in tr if t in set(iCR)]
            s = unit(np.nanmean(Fe[a] / sd, 0) - np.nanmean(Fe[b] / sd, 0))
            return orth(u, [s])
        rest = np.r_[iWH[okl[iWH]], iWM[okl[iWM]]]
        p, _ = cv_axis(Fl, iAH[okl[iAH]], iCR[okl[iCR]], rest, seed, mk)
        q = norm01(p, iCR[okl[iCR]], iAH[okl[iAH]])
        out["e_WH"] = np.nanmean(q[iWH[okl[iWH]]]) if okl[iWH].sum() >= 3 else np.nan
        out["e_WM"] = np.nanmean(q[iWM[okl[iWM]]]) if okl[iWM].sum() >= 3 else np.nan
        out["e_WH_WM"] = out["e_WH"] - out["e_WM"]
        # A3: hit - miss vs identity (early), hit - miss (late) vs reward expectation
        if min(len(iWH), len(iWM)) >= MIN_EV:
            oke = np.isfinite(early).all(1)
            out["cos_HI_early"] = cos_norm(Fe, iWH[oke[iWH]], iWM[oke[iWM]], Fe, iW[oke[iW]], iAH[oke[iAH]], rng)
            out["cos_HE"] = cos_norm(Fl, iWH[okl[iWH]], iWM[okl[iWM]], Fl, iAH[okl[iAH]], iCR[okl[iCR]], rng)
    # lick-aligned: A2 outcome, C7, C8
    if min(len(jAH), len(jSL)) >= MIN_EV and len(jWH) >= 4:
        Fp, Fo = np.nan_to_num(pre), np.nan_to_num(post)
        sdp = np.nanstd(Fp[np.r_[jAH, jSL]], 0); sdp[~(sdp > 0)] = 1
        cd_pre = unit(Fp[jAH].mean(0) / sdp - Fp[jSL].mean(0) / sdp)
        def mko(u, sd, tr):
            return orth(u, [cd_pre * sdp / sd])
        p, _ = cv_axis(Fo, jAH, jSL, jWH, seed, mko)
        q = norm01(p, jSL, jAH)
        out["o_WH"] = np.nanmean(q[jWH]); out["o_sep"] = np.nanmean(p[jAH]) - np.nanmean(p[jSL])
        # C8 motor axis: mean SL pre-lick activity (baseline-corrected, scaled), held-out for SL via folds is not needed (not a contrast)
        M = unit(Fp[jSL].mean(0) / sdp)
        S_ = None
        if len(iW) and len(iAH):
            oke = np.isfinite(early).all(1)
            S_ = unit(np.nanmean(np.nan_to_num(early)[iW[oke[iW]]] / sdp, 0) - np.nanmean(np.nan_to_num(early)[iAH[oke[iAH]]] / sdp, 0))
        pm = (Fp / sdp) @ M / np.sqrt(Fp.shape[1])
        out["m_WH"], out["m_AH"] = pm[jWH].mean() - pm[jSL].mean(), pm[jAH].mean() - pm[jSL].mean()
        # C7: original vs orthogonalised CD, cross-validated, time course
        def mkc(u, sd, tr):
            others = [M * sdp / sd] + ([S_ * sdp / sd] if S_ is not None and np.all(np.isfinite(S_)) else [])
            return orth(u, others)
        tc = []
        for name, make in (("orig", None), ("orth", mkc)):
            _, axes = cv_axis(Fp, jAH, jSL, jWH, seed, make)
            # project every lick-aligned window with the fold axes (held-out for AH / SL by fold assignment is approximated by the fold average
            # here; the window lambda below uses the held-out cv projections)
            pw, _ = cv_axis(Fp, jAH, jSL, jWH, seed, make)
            sep = np.nanmean(pw[jAH]) - np.nanmean(pw[jSL])
            out[f"lam_{'win' if name == 'orig' else 'orth_win'}"] = (np.nanmean(pw[jWH]) - np.nanmean(pw[jSL])) / sep if sep > SEP_MIN else np.nan
            A = np.mean(axes, 0)
            if name == "orth":
                A0 = np.mean(cv_axis(Fp, jAH, jSL, jWH, seed, None)[1], 0)
                out["var_kept"] = float((A @ A0) ** 2 / ((A @ A) * (A0 @ A0))) if np.all(np.isfinite(A)) else np.nan
            P = np.einsum("etu,u->et", np.transpose(np.nan_to_num(LB), (0, 2, 1)), A) / np.sqrt(Fp.shape[1])
            m = {c: np.nanmean(P[j], 0) for c, j in (("WH", jWH), ("AH", jAH), ("SL", jSL))}
            sw = np.nanmean(m["AH"][(lc > -100) & (lc < 0)]) - np.nanmean(m["SL"][(lc > -100) & (lc < 0)])
            for k, t in enumerate(lc):
                tc.append(dict(meta, axis=name, t=t, lam=(m["WH"][k] - m["SL"][k]) / sw if sw > SEP_MIN else np.nan))
        return out, tc
    return out, []


def main(a):
    W = pd.read_parquet(C.m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy(); W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    jobs = [(sid, dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0]),
             g[["electrode_group", "cluster_id", "quality_label"]], zlib.crc32(sid.encode())) for sid, g in W.groupby("session_id")]
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = [r for r in pool.map(session, jobs) if r is not None]
    S = C.population(pd.DataFrame([o for o, _ in res])); T = C.population(pd.DataFrame([x for _, tc in res for x in tc]))
    S.to_csv(C.TAB / "004_sessions.csv", index=False); T.to_csv(C.TAB / "004_timecourse.csv", index=False)
    R = pd.DataFrame([r for m, _ in MEAS if m in S for r in C.measure_stats(S, m, "whole brain", m)]); R.to_csv(C.TAB / "004_stats.csv", index=False)
    Tst = []
    for (c, ax_), g in T.groupby(["cohort", "axis"]):
        for t, h in g.groupby("t"):
            a0, b0 = h[h.stage == "learning"].lam.dropna(), h[h.stage == "expert"].lam.dropna()
            Tst.append(dict(cohort=c, axis=ax_, t=t, d0=a0.mean(), e=b0.mean(), p=stats.mannwhitneyu(a0, b0).pvalue if len(a0) > 2 and len(b0) > 2 else np.nan))
    Tst = pd.DataFrame(Tst)
    for k, g in Tst.groupby(["cohort", "axis"]):
        Tst.loc[g.index, "q"] = C.bh(g.p)
    Tst.to_csv(C.TAB / "004_tc_stats.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    # figure A
    fig, axs = plt.subplots(2, 4, figsize=(8.2, 5.2), gridspec_kw=dict(wspace=0.65, hspace=1.45))
    for ax, (m, lab), l in zip(axs.ravel(), [x for x in MEAS if x[0] in ("e_WH", "e_WM", "e_WH_WM", "o_WH", "o_sep", "cos_HI_early", "cos_HE")], "abcdefg"):
        Rm = R[R.measure == m]
        C.dot_panel(ax, S, m, Rm, lab.split(": ")[1].replace(" on ", "
on ").replace(", ", ",
"), l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
        if m in ("e_WH", "e_WM", "o_WH"):
            ax.axhline(1, color="0.4", lw=0.6, ls="--")
    axs[1, 3].axis("off")
    axs[1, 3].text(0, 0.95, "A1: AH vs CR, 50–150 ms after the stimulus,\northogonal to the 0–50 ms AH − CR response\n\n"
                   "A2: AH vs SL, +25 to +125 ms after the lick,\northogonal to the pre-lick CD\n\nA3: reliability-normalised cosines\n(1 = same direction)",
                   fontsize=5, va="top")
    fig.suptitle("A: reward-expectation, outcome and alignment axes, day 0 vs expert (green R+, magenta R−; open day 0, filled expert)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"004_A_axes.{ext}", dpi=300, bbox_inches="tight")
    # figure C
    fig, axs = plt.subplots(2, 4, figsize=(7.8, 4.6), gridspec_kw=dict(wspace=0.6, hspace=1.2))
    for i, c in enumerate(("R+", "R-")):
        for j, ax_ in enumerate(("orig", "orth")):
            ax = axs[i, j]
            for s_, ls in (("learning", (0, (3, 1.5))), ("expert", "-")):
                p = T[(T.cohort == c) & (T.stage == s_) & (T.axis == ax_)].pivot_table(index="session_id", columns="t", values="lam")
                ax.plot(p.columns, p.mean(), color=C.COH[c], ls=ls, lw=1, label=f"{'day 0' if s_ == 'learning' else 'expert'} ({len(p)})")
                ax.fill_between(p.columns, p.mean() - p.sem(), p.mean() + p.sem(), color=C.COH[c], alpha=0.15, lw=0)
            sg = Tst[(Tst.cohort == c) & (Tst.axis == ax_) & (Tst.q < 0.05)]; ytop = ax.get_ylim()[1]
            for t in sg.t:
                ax.plot([t - 12.5, t + 12.5], [ytop, ytop], color="k", lw=2)
            ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color="0.7", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.5, ls="--")
            ax.set_title(f"{c.replace('-', '−')}: λ(t) on {'original' if ax_ == 'orig' else 'orthogonalised'} CD\n(bars: D0 ≠ E, BH q < 0.05)", fontsize=5.6, color=C.COH[c])
            ax.set_xlabel("time from lick (ms)"); ax.legend(frameon=False, fontsize=4.6)
            if j == 0:
                ax.set_ylabel("λ(t)")
            if i == 0 and j == 0:
                ax.text(-0.4, 1.2, "a", transform=ax.transAxes, fontsize=8, weight="bold")
    for ax, (m, l) in zip([axs[0, 2], axs[0, 3], axs[1, 2], axs[1, 3]], (("lam_orth_win", "b"), ("var_kept", "c"), ("m_WH", "d"), ("m_AH", "e"))):
        Rm = R[R.measure == m]
        C.dot_panel(ax, S, m, Rm, dict(MEAS)[m].split(": ")[1], l, Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle("C: reward-lick CD with the motor (any-lick preparation) and sensory (whisker − auditory) axes projected out; motor-axis projections",
                 fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"004_C_orth.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(R[R.cohort.isin(["R+", "R-", "R+ vs R-"])][["measure", "cohort", "n_d0", "n_e", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(3).to_string(index=False))
    print(Tst[Tst.q < 0.05].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20)
    main(ap.parse_args())
