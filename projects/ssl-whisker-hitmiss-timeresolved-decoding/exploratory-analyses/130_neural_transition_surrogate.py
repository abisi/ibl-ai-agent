"""130 -- Neural transition trial per session, from the single-trial decoder margin, tested against PHASE-RANDOMISED surrogates
(user 2026-10-01: "identify the trial where decodability changes most" -> do it with a null that keeps autocorrelation, then
summarise).
Sources (single-trial held-out margins, A1-trimmed, perf != 6): hit vs miss (123 v2), whisker vs no-stim and whisker vs auditory
at stimulus onset (128); windows 5-50 and 5-100 ms.
Per session x decoding x window:
  z        class-residualised signed margin per decoded trial (minus the mean of its class);
  CP       c* = argmax over splits c (BORDER <= c <= n - BORDER and 10-90% of the trials) of |Welch t| of mean(z[c:]) -
           mean(z[:c]); sign = direction (+ = margin higher after = more decodable);
  null     N_SURR IAAFT surrogates of z (iterative amplitude-adjusted Fourier transform: same values, same power spectrum
           i.e. autocorrelation, phases randomised -> no specific step); the same max |t| search on each;
           p = (#surrogates with max|t| >= observed + 1) / (N_SURR + 1);
  position c* as a fraction of the decoded trials and as a whisker-trial index of the curve-aligned list (whisker trials
           starting before decoded trial c*).
Comparison with behaviour (positions as FRACTIONS of the session -- raw trial counts scale with session length for both):
behavioural CP = L6x (036: joint whisker + FA CP with lapse; forced = every session; "learner" =
log10 BF > 0.3 and P(whisker > FA) > 0.9). Spearman neural vs behavioural CP and median |difference| vs behavioural CPs
permuted across sessions within cohort (2000 permutations), for all sessions and for neurally significant sessions.
Group level per cohort: fraction of sessions with a significant neural transition (p < 0.05), fraction going up among them,
position; R+ vs R- (Fisher exact on fractions; Mann-Whitney AND Welch on positions and t). Uncorrected.
Outputs: figures/130_neural_transition.{pdf,png,svg}, 130_per_session.csv, 130_stats.csv
Run (haas): python 130_neural_transition_surrogate.py
"""

from __future__ import annotations

import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}
BORDER, N_SURR, N_ITER, N_PERM = 10, 500, 30, 2000
FS = 6.5


def sources():
    out = {}
    for f in ("123_singletrial_scores_all_v2.pkl", "123_singletrial_scores_all.pkl"):
        if (OUT / f).exists():
            out["hitmiss"] = OUT / f
            break
    for dec in ("whisker_nostim", "whisker_auditory"):
        if (OUT / f"128_singletrial_{dec}.pkl").exists():
            out[dec] = OUT / f"128_singletrial_{dec}.pkl"
    return out


def max_t(z, lo, hi):
    n = len(z)
    cs, cs2 = np.cumsum(z), np.cumsum(z * z)
    c = np.arange(lo, hi + 1)
    m1 = cs[c - 1] / c
    m2 = (cs[-1] - cs[c - 1]) / (n - c)
    v1 = (cs2[c - 1] - c * m1 ** 2) / np.maximum(c - 1, 1)
    v2 = ((cs2[-1] - cs2[c - 1]) - (n - c) * m2 ** 2) / np.maximum(n - c - 1, 1)
    se = np.sqrt(np.maximum(v1, 1e-12) / c + np.maximum(v2, 1e-12) / (n - c))
    t = (m2 - m1) / se
    i = int(np.argmax(np.abs(t)))
    return int(c[i]), float(t[i])


def iaaft(z, rng):
    sorted_z = np.sort(z)
    amp = np.abs(np.fft.rfft(z))
    s = rng.permutation(z)
    for _ in range(N_ITER):
        ph = np.angle(np.fft.rfft(s))
        s = np.fft.irfft(amp * np.exp(1j * ph), n=len(z))
        s = sorted_z[np.argsort(np.argsort(s))]
    return s


def one(args):
    key, dec, z, rg, cw_t, t = args
    n = len(z)
    lo, hi = max(BORDER, int(0.1 * n)), min(n - BORDER, int(0.9 * n))
    if hi - lo < 5:
        return None
    c, tv = max_t(z, lo, hi)
    rng = np.random.default_rng(zlib.crc32(f"{key}|{dec}".encode()))
    null = np.array([abs(max_t(iaaft(z, rng), lo, hi)[1]) for _ in range(N_SURR)])
    return dict(session_id=key[0], window=key[1], decoding=dec, reward_group=rg, n=n, cp=c, cp_frac=c / n, t=tv,
                p=float((np.sum(null >= abs(tv)) + 1) / (N_SURR + 1)), cp_whisker=int(np.sum(cw_t < t[c])),
                n_whisker=len(cw_t))


def main():
    jobs = []
    for dec, f in sources().items():
        D = pickle.load(open(f, "rb"))
        for (sid, w), out in D["res"].items():
            m = D["meta"][sid]
            y = np.asarray(m["y"], bool)
            z = np.asarray(out["real"]["margin"], float).copy()
            ok = np.isfinite(z)
            for cls in (True, False):
                sel = ok & (y == cls)
                if sel.any():
                    z[sel] -= z[sel].mean()
            if ok.sum() < 3 * BORDER:
                continue
            jobs.append(((sid, w), dec, z[ok], m["reward_group"], np.asarray(m["cw_t"], float), np.asarray(m["t"], float)[ok]))
    with ProcessPoolExecutor(40) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=4) if r is not None]
    P = pd.DataFrame(rows)
    b = pd.read_csv(LTP / "036_joint_cp_relaxed.csv")
    b = b[(b.version == "L6x") & (b.bf_th == 0.3) & (b.p_th == 0.9)].set_index("session_id")
    P["beh_cp"] = P.session_id.map(b.LT)
    P["beh_learner"] = P.session_id.map(b.learner)
    # positions as FRACTIONS of the session (whisker-trial counts scale with session length for both measures, which alone
    # would make them correlate)
    P["cp_wfrac"] = P.cp_whisker / (P.n_whisker - 1).clip(lower=1)
    P["beh_frac"] = P.beh_cp / (P.n_whisker - 1).clip(lower=1)
    P.to_csv(OUT / "130_per_session.csv", index=False)
    rng = np.random.default_rng(0)
    srows = []
    for (dec, w), g in P.groupby(["decoding", "window"]):
        sig = {}
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            s = x[x.p < 0.05]
            sig[rg] = (len(s), len(x))
            r = dict(decoding=dec, window=w, cohort=rg, n=len(x), n_sig=len(s), frac_sig=len(s) / max(len(x), 1),
                     frac_up_among_sig=(s.t > 0).mean() if len(s) else np.nan, median_pos_all=x.cp_frac.median(),
                     median_pos_sig=s.cp_frac.median() if len(s) else np.nan)
            for lab, xx in (("all", x), ("sig", s), ("learners", x[x.beh_learner == True])):  # noqa: E712
                c = xx.dropna(subset=["beh_frac"])
                if len(c) >= 5:
                    r[f"spearman_{lab}"], r[f"p_spearman_{lab}"] = stats.spearmanr(c.cp_wfrac, c.beh_frac)
                    obs = np.median(np.abs(c.cp_wfrac - c.beh_frac))
                    pool = g[(g.reward_group == rg)].dropna(subset=["beh_frac"]).beh_frac.to_numpy()
                    perm = [np.median(np.abs(c.cp_wfrac.to_numpy() - rng.choice(pool, len(c), replace=False)))
                            for _ in range(N_PERM)]
                    r.update(**{f"medabs_{lab}": obs, f"medabs_null_{lab}": float(np.median(perm)),
                                f"p_perm_{lab}": (np.sum(np.array(perm) <= obs) + 1) / (N_PERM + 1), f"n_{lab}": len(c)})
            srows.append(r)
        a, bb = g[g.reward_group == "R+"], g[g.reward_group == "R-"]
        fisher = stats.fisher_exact([[sig["R+"][0], sig["R+"][1] - sig["R+"][0]], [sig["R-"][0], sig["R-"][1] - sig["R-"][0]]])[1]
        pm_pos, pw_pos = stats.mannwhitneyu(a.cp_frac, bb.cp_frac).pvalue, stats.ttest_ind(a.cp_frac, bb.cp_frac, equal_var=False).pvalue
        pm_t, pw_t = stats.mannwhitneyu(a.t, bb.t).pvalue, stats.ttest_ind(a.t, bb.t, equal_var=False).pvalue
        for r in srows[-2:]:
            r.update(p_fisher_frac_sig=fisher, p_pos_mw=pm_pos, p_pos_welch=pw_pos, p_t_mw=pm_t, p_t_welch=pw_t)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "130_stats.csv", index=False)
    figure(P, S)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def figure(P, S):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    combos = [(d, w) for d in ("hitmiss", "whisker_nostim", "whisker_auditory") for w in ("5-50ms", "5-100ms")
              if ((P.decoding == d) & (P.window == w)).any()]
    fig, axes = plt.subplots(len(combos), 4, figsize=(8.27, 1.75 * len(combos) + 0.8), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=1 - 0.6 / (1.75 * len(combos) + 0.8), bottom=0.05, wspace=0.45, hspace=0.8)
    for r, (dec, w) in enumerate(combos):
        g = P[(P.decoding == dec) & (P.window == w)]
        s = S[(S.decoding == dec) & (S.window == w)].set_index("cohort")
        lab = f"{dec.replace('_', ' vs ').replace('nostim', 'no-stim')} {w}"
        ax = axes[r, 0]
        for j, rg in enumerate(("R+", "R-")):
            if rg not in s.index:
                continue
            fs, fu = s.loc[rg, "frac_sig"], s.loc[rg, "frac_up_among_sig"]
            ax.bar(j, fs, color=COL[rg], width=0.6)
            ax.bar(j, fs * (fu if np.isfinite(fu) else 0), color="k", alpha=0.25, width=0.6)
            ax.text(j, fs + 0.01, f"{int(s.loc[rg, 'n_sig'])}/{int(s.loc[rg, 'n'])}", ha="center", fontsize=FS - 1)
        ax.axhline(0.05, color="0.5", lw=0.6, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R+", "R−"])
        ax.set_ylabel(f"{lab}\nfraction with neural\ntransition (p<.05)", fontsize=FS - 0.5)
        ax.set_title(f"Fisher p {pf(s.iloc[0].p_fisher_frac_sig)}; dark = going up", fontsize=FS - 1)
        ax = axes[r, 1]
        bins = np.linspace(0, 1, 11)
        for rg in ("R+", "R-"):
            ax.hist(g[(g.reward_group == rg) & (g.p < 0.05)].cp_frac, bins=bins, histtype="step", color=COL[rg], lw=1.2)
            ax.hist(g[(g.reward_group == rg)].cp_frac, bins=bins, histtype="step", color=COL[rg], lw=0.6, ls=":")
        ax.set_xlabel("neural transition position\n(solid = significant, dotted = all)", fontsize=FS - 1)
        ax.set_title(f"position R+ vs R−: MW {pf(s.iloc[0].p_pos_mw)}, Welch {pf(s.iloc[0].p_pos_welch)}", fontsize=FS - 1)
        ax = axes[r, 2]
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg].dropna(subset=["beh_frac"])
            sg = x.p < 0.05
            ax.scatter(x.beh_frac[sg], x.cp_wfrac[sg], s=8, color=COL[rg], lw=0)
            ax.scatter(x.beh_frac[~sg], x.cp_wfrac[~sg], s=8, facecolor="none", edgecolor=COL[rg], lw=0.5)
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("behavioural CP (L6x, fraction of session)", fontsize=FS - 1)
        ax.set_ylabel("neural transition (fraction)", fontsize=FS - 1)
        ax = axes[r, 3]
        ax.set_axis_off()
        lines = []
        for rg in ("R+", "R-"):
            if rg not in s.index:
                continue
            q = s.loc[rg]
            for labx in ("all", "sig", "learners"):
                if np.isfinite(q.get(f"medabs_{labx}", np.nan)):
                    lines.append(f"{rg} {labx} (n={int(q[f'n_{labx}'])}): ρ {q[f'spearman_{labx}']:.2f} (p {pf(q[f'p_spearman_{labx}'])}); "
                                 f"|Δ| {q[f'medabs_{labx}']:.2f} vs {q[f'medabs_null_{labx}']:.2f} (p {pf(q[f'p_perm_{labx}'])})")
        ax.text(0, 0.95, "neural vs behavioural CP\n" + "\n".join(lines), fontsize=FS - 1.5, va="top", transform=ax.transAxes)
    fig.suptitle("Neural transition trial per session (max step of the class-residualised single-trial margin), tested against "
                 f"{N_SURR} IAAFT surrogates (same autocorrelation); compared with the behavioural change point (L6x). "
                 "Uncorrected.", fontsize=FS + 0.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"130_neural_transition.{ext}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
