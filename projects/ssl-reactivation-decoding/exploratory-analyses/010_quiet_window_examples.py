"""010 -- Day-0 still events (no movement), R+ vs R- side by side (user 2026-10-30, after 009; v4: still time instead of the
task quiet window).

Data: 009 saved files (stable units), whole-brain decoder ("All") of one target, threshold = 009 THR_MAIN (v6: median over 100 rate-matched weight shuffles of the 99.9th percentile),
sessions with passive pre and post and balanced accuracy >= GOOD_BACC, day 0.
Still time (009 v4): valid time (outside every trial onset - 10 ms .. + 5 s, passive blocks and recording borders) with no
whisking > 200 deg/s or jaw movement within 250 ms and no lick within 1 s. Events: 009 events whose every window is still.
Figure fig21_still_examples_<target>:
  top: still-event rate per session, R+ vs R- (uncorrected Mann-Whitney / Welch p), example sessions ringed;
  below: example sessions at the 25th / 50th / 75th percentile of their cohort's still-event rate (R+ left, R- right):
  N_WIN still stretches (>= MIN_RUN s) spread evenly over the session, concatenated (grey lines: stretch borders); readout in
  MAD units, threshold (red), still events (triangles); whisker speed underneath (empty when not tracked).
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m8 = importlib.import_module("008_sensitivity_threshold_cohorts")
m9 = importlib.import_module("009_full_dataset_events")
N_WIN = 30
MIN_RUN = 1.0                                    # s: shortest still stretch shown
PCTL = (25, 50, 75)
COH = {"R+": "#00B400", "R-": "#C800C8"}


def quiet_windows(sid, target, group=m9.m8.ALL, unit_set="stable"):
    """readout, threshold, quiet-window runs and quiet-window event starts (grid indices) of one session"""
    out = m9.ROOT / unit_set / sid
    S = np.load(out / "session.npz")
    C = np.load(out / f"continuous_{target}_{group.replace(' ', '_')}.npz")
    t, z, valid, thr = C["t"], C["z"], C["valid"], float(C["thr"])
    q = S["still"] & valid
    d = np.diff(np.r_[0, q.astype(int), 0])
    runs = np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]               # [start, stop) grid indices
    runs = runs[(runs[:, 1] - runs[:, 0]) * m1.STEP >= MIN_RUN]
    E = pd.read_parquet(out / f"events_{target}_{group.replace(' ', '_')}.parquet")
    ev = np.searchsorted(t, E.t_peak[E.still].to_numpy() - 1e-6)
    return dict(t=t, z=z, thr=thr, runs=runs, quiet_s=q.sum() * m1.STEP, ev=ev, B=dict(np.load(out / "behaviour.npz", allow_pickle=True)))


def pick(D, coh):
    """sessions at the PCTL percentiles of the cohort's quiet-window rate"""
    d = D[D.cohort == coh].sort_values("rate_still").reset_index(drop=True)
    return [d.iloc[int(round(p / 100 * (len(d) - 1)))] for p in PCTL]


def draw(ax_r, ax_w, Q, coh, ylim):
    t, z, runs = Q["t"], Q["z"], Q["runs"]
    sel = runs[np.unique(np.linspace(0, len(runs) - 1, min(N_WIN, len(runs))).round().astype(int))]
    B = Q["B"]
    has_w = "whisker_speed_t" in B and np.isfinite(B["whisker_speed_v"]).any()
    x0 = 0.0
    for a, b in sel:
        x = x0 + (t[a:b] - t[a])
        ax_r.plot(x, np.clip(z[a:b], *ylim), color="k", lw=0.35)
        e = Q["ev"][(Q["ev"] >= a) & (Q["ev"] < b)]
        ax_r.scatter(x0 + t[e] - t[a], np.full(len(e), ylim[1] * 0.93), marker="v", s=7, color=COH[coh], lw=0, zorder=4)
        if has_w:
            m = (B["whisker_speed_t"] >= t[a]) & (B["whisker_speed_t"] < t[b - 1])
            ax_w.plot(x0 + B["whisker_speed_t"][m] - t[a], np.clip(B["whisker_speed_v"][m], 0, 2000), color="0.35", lw=0.35)
        x0 = x[-1] + m1.STEP
        for ax in (ax_r, ax_w):
            ax.axvline(x0, color="0.8", lw=0.3, zorder=0)
    ax_r.axhline(Q["thr"], color="r", lw=0.5)
    ax_r.set_ylim(*ylim)
    for ax in (ax_r, ax_w):
        ax.set_xlim(0, x0)
    if not has_w:
        ax_w.axis("off")
    else:
        ax_w.set_ylim(0, 2000)
    return len(sel), x0


def main(target, unit_set):
    plt = m2.setup()
    Dd = pd.read_csv(m1.HOME / f"full_decoders_{unit_set}.csv")
    D = Dd[(Dd.group == m9.m8.ALL) & (Dd.target == target) & (Dd.thr_source == m9.THR_MAIN) & (Dd.bacc >= m9.GOOD_BACC)
           & Dd.stage.str.startswith("day 0")].dropna(subset=["rate_still"])
    ex = {c: pick(D, c) for c in COH}
    fig = plt.figure(figsize=(7.4, 9.2), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.1, 6])
    ax = top.subplots(1, 3, width_ratios=[1, 1, 2.2])
    for j, (mt, lab) in enumerate((("rate_still", "events / s, still"), ("excess_still", "excess vs shuffled,\nstill (/ s)"))):
        a_, b_ = (D[D.cohort == c][mt].dropna() for c in COH)
        for x_, (c, v) in enumerate(zip(COH, (a_, b_))):
            jit = np.random.default_rng(x_).uniform(-0.15, 0.15, len(v))
            ax[j].scatter(x_ + jit, v, s=5, color=COH[c], lw=0, alpha=0.8, zorder=3)
            ax[j].plot([x_ - 0.28, x_ + 0.28], [v.median()] * 2, color="k", lw=0.9, zorder=4)
            ids = {r.session_id for r in ex[c]}
            hit = v.index.isin(D.index[D.session_id.isin(ids)])
            ax[j].scatter(x_ + jit[hit], v[hit], s=22, facecolors="none", edgecolors="k", lw=0.5, zorder=5)
        p_mw, p_t = m8.two_tests(a_, b_)
        ax[j].set_title(f"{lab}\np {p_mw:.2g} (MW) / {p_t:.2g} (Welch)", fontsize=5.5)
        ax[j].set_xticks([0, 1], [f"{c}\nn {len(v)}" for c, v in zip(COH, (a_, b_))], fontsize=5)
        ax[j].set_yscale("symlog", linthresh=m9.LINTHRESH)
        if mt.startswith("excess"):
            ax[j].axhline(0, color="0.6", lw=0.4, ls=":")
        ax[j].set_xlim(-0.6, 1.6)
    ax[2].axis("off")
    ax[2].text(0, 1, f"day 0, whole-brain {target} decoders ({unit_set} units, balanced accuracy >= {m9.GOOD_BACC}), "
               f"{m9.THR_TEXT}\n\nstill = outside trials (onset - 10 ms .. + 5 s), no whisking > 200 deg/s\n"
               "or jaw movement within 250 ms, no lick within 1 s (every window of the event)\n\nbelow: sessions at the 25th / 50th / "
               f"75th percentile of their cohort's\nstill-event rate (ringed above); {N_WIN} still stretches (>= {MIN_RUN:.0f} s) spread "
               f"over the session,\nconcatenated (grey: stretch borders); readout in MAD units, threshold red,\ntriangles: still events; "
               "grey trace below: whisker speed (deg/s)",
               transform=ax[2].transAxes, va="top", fontsize=5.2)
    cols = bot.subfigures(1, 2)
    for c_i, coh in enumerate(COH):
        axs = cols[c_i].subplots(2 * len(PCTL), 1, gridspec_kw=dict(height_ratios=[2.2, 0.7] * len(PCTL)))
        cols[c_i].suptitle(coh, color=COH[coh], fontsize=7, weight="bold")
        for k, (p, r) in enumerate(zip(PCTL, ex[coh])):
            Q = quiet_windows(r.session_id, target, unit_set=unit_set)
            ylim = (-5, max(12.0, 1.6 * Q["thr"]))
            n, dur = draw(axs[2 * k], axs[2 * k + 1], Q, coh, ylim)
            axs[2 * k].set_title(f"{p}th pct: {r.session_id}  pAUC {r.pauc:.2f}, thr {r.thr_norm:.1f} MAD; passive AUC vs sham pre {r.auc_sham_pre:.2f} / post "
                                 f"{r.auc_sham_post:.2f}, vs other modality {r.auc_other_pre:.2f} / {r.auc_other_post:.2f}\nstill-event rate "
                                 f"{r.rate_still:.4f}/s (shuffled templates {r.rate_shuf_still:.4f}, p {r.p_shuf_still:.2f}) over {Q['quiet_s']:.0f} s "
                                 f"of still time; shown: {n} stretches, {dur:.0f} s", fontsize=4.8, loc="left")
            axs[2 * k].set_ylabel("readout\n(MAD)", fontsize=5)
            axs[2 * k + 1].set_ylabel("whisk", fontsize=5) if axs[2 * k + 1].axison else None
            for a in axs[2 * k: 2 * k + 2]:
                a.tick_params(labelsize=4.5)
                a.set_xticklabels([]) if a is not axs[-1] else None
        axs[-1].set_xlabel("concatenated still time (s)", fontsize=5)
    m2.save(fig, m1.HOME / "figures" / f"full_{unit_set}", f"fig21_still_examples_{target}")
    plt.close(fig)
    print(pd.DataFrame([dict(cohort=c, pct=p, session_id=r.session_id, rate_still=r.rate_still, pauc=r.pauc)
                        for c in COH for p, r in zip(PCTL, ex[c])]).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="*", default=["whisker", "auditory"])
    ap.add_argument("--unit-set", default="stable")
    a = ap.parse_args()
    for tg in a.target:
        main(tg, a.unit_set)
