"""026 -- does the whisker-stimulus-evoked movement become more frequent / larger when it is rewarded?
(haas; reads the 024 cache: 5 ms grid, NWB DLC without smoothing)

Per trial evoked measures (025-150 ms after the stimulus minus the -200..0 ms baseline for angles; mean for speeds):
whisker angle, whisker speed, top nose speed, side nose speed, whisker particle y. Hits only when the first jaw onset /
lick / tongue event is > 150 ms after the stimulus (else NaN); trials with a lick in -1..0 s dropped.
Evoked movement present = measure above the session's 95th percentile on correct-rejection (no-stim, no-lick) trials.
Analyses (unit = session = mouse; R+ vs R- Mann-Whitney + Welch; within cohort sign-flip; uncorrected, exploratory):
 A. session trend on whisker trials (all / misses only): late - early third of the whisker trials (fraction present,
    mean magnitude), slope over the normalised whisker-trial index;
 B. learning trial: after - before the LT (learners with an LT, whisker trials);
 C. trial-to-trial: P(present on whisker trial n+1 | whisker trial n was a hit) - P(... | miss), and the same for
    magnitude; R+ (hit rewarded) vs R- (hit unrewarded); also on auditory trials (rewarded in both cohorts) as control.
Output: tables/026_*.csv, figures/026_evoked_reinforcement.png|pdf|svg.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bm_lib as L  # noqa: E402

_sp = importlib.util.spec_from_file_location("m24", HERE / "024_stimulus_micromovements.py")
M24 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(M24)

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
WIN = (0.025, 0.150)
MEAS = ["whisker_angle", "whisker_speed", "top_nose_speed", "nose_speed", "top_particle_y"]
rng = np.random.default_rng(0)


def signflip(x, n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    s = rng.choice([-1, 1], size=(n, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= abs(x.mean())) + 1) / (n + 1))


def two(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def main():
    S = M24.M18.sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    lt = pd.read_csv(M24.LT_TABLE)
    ltd = dict(zip(lt.session_id, lt.lt_final))
    w = (M24.TGRID >= WIN[0]) & (M24.TGRID < WIN[1])
    si = {s: M24.SIGN.index(s) for s in MEAS}
    trial_rows, A, B, C = [], [], [], []
    for f in sorted(M24.OUT.glob("*.npz")):
        sid = f.stem
        if sid not in coh:
            continue
        z = np.load(f)
        X, stim, lick, first, lb, wti = z["X"].astype(float), z["stim"], z["lick"], z["first_event"], z["lick_before"], z["whisker_trial_index"]
        n = len(stim)
        with np.errstate(all="ignore"):
            V = {m: np.nanmean(X[:, si[m]][:, w], 1) for m in MEAS}
        valid = ~lb & ((lick == 0) | (first > WIN[1]))
        cr = valid & (stim == "no_stim") & (lick == 0)
        if cr.sum() < 10:
            continue
        thr = {m: np.nanpercentile(V[m][cr], 95) for m in MEAS}
        P = {m: np.where(np.isfinite(V[m]), (V[m] > thr[m]).astype(float), np.nan) for m in MEAS}
        for i in range(n):
            if valid[i] and stim[i] in ("whisker", "auditory"):
                trial_rows.append(dict(session_id=sid, cohort=coh[sid], trial=i, stim=stim[i], lick=int(lick[i]), wti=int(wti[i]),
                                       **{f"mag_{m}": V[m][i] for m in MEAS}, **{f"pres_{m}": P[m][i] for m in MEAS}))
        # A. session trend (whisker trials)
        wt = np.flatnonzero((stim == "whisker"))
        if len(wt) < 30:
            continue
        third = len(wt) // 3
        e_idx, l_idx = wt[:third], wt[-third:]
        pos = np.full(n, np.nan)
        pos[wt] = np.arange(len(wt)) / (len(wt) - 1)
        for subset in ("all whisker", "whisker misses"):
            keep = valid & ((lick == 0) if subset == "whisker misses" else np.ones(n, bool))
            for m in MEAS:
                for kind, D in (("present", P[m]), ("magnitude", V[m])):
                    e, l_ = D[np.intersect1d(e_idx, np.flatnonzero(keep))], D[np.intersect1d(l_idx, np.flatnonzero(keep))]
                    sel = keep & (stim == "whisker") & np.isfinite(D)
                    slope = np.polyfit(pos[sel], D[sel], 1)[0] if sel.sum() >= 15 else np.nan
                    A.append(dict(session_id=sid, cohort=coh[sid], subset=subset, measure=m, kind=kind,
                                  early=np.nanmean(e) if np.isfinite(e).sum() >= 5 else np.nan,
                                  late=np.nanmean(l_) if np.isfinite(l_).sum() >= 5 else np.nan, slope=slope))
                    # B. learning trial
                    if np.isfinite(ltd.get(sid, np.nan)):
                        wpos = np.where(stim == "whisker", wti, -1)
                        bef = keep & (stim == "whisker") & (wpos >= 0) & (wpos < ltd[sid]) & np.isfinite(D)
                        aft = keep & (stim == "whisker") & (wpos >= ltd[sid]) & np.isfinite(D)
                        if bef.sum() >= 5 and aft.sum() >= 5:
                            B.append(dict(session_id=sid, cohort=coh[sid], subset=subset, measure=m, kind=kind,
                                          before=np.nanmean(D[bef]), after=np.nanmean(D[aft])))
        # C. trial-to-trial: next trial of the same stimulus type after a hit vs after a miss
        for st in ("whisker", "auditory"):
            idx = np.flatnonzero(stim == st)
            if len(idx) < 20:
                continue
            prev, nxt = idx[:-1], idx[1:]
            for m in MEAS:
                for kind, D in (("present", P[m]), ("magnitude", V[m])):
                    for subset in ("next: all", "next: misses"):
                        okn = valid[nxt] & np.isfinite(D[nxt]) & ((lick[nxt] == 0) if subset == "next: misses" else True)
                        ah = okn & (lick[prev] == 1)
                        am = okn & (lick[prev] == 0)
                        if ah.sum() >= 5 and am.sum() >= 5:
                            C.append(dict(session_id=sid, cohort=coh[sid], stim=st, subset=subset, measure=m, kind=kind,
                                          after_hit=np.nanmean(D[nxt][ah]), after_miss=np.nanmean(D[nxt][am]),
                                          n_after_hit=int(ah.sum()), n_after_miss=int(am.sum())))
    TR = pd.DataFrame(trial_rows)
    TR.to_csv(T / "026_trials.csv", index=False)
    A, B, C = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(C)
    A["delta"] = A.late - A.early
    B["delta"] = B.after - B.before
    C["delta"] = C.after_hit - C.after_miss
    for df, name in ((A, "trend"), (B, "learning_trial"), (C, "trial_to_trial")):
        df.to_csv(T / f"026_{name}_per_session.csv", index=False)
    res = []
    for name, df, keys in (("A trend late-early", A, ["subset", "measure", "kind"]), ("A slope", A.assign(delta=A.slope), ["subset", "measure", "kind"]),
                           ("B after-before LT", B, ["subset", "measure", "kind"]),
                           ("C after hit - after miss", C, ["stim", "subset", "measure", "kind"])):
        for k, g in df.groupby(keys):
            a, b = g[g.cohort == "R+"].delta, g[g.cohort == "R-"].delta
            pm, pw = two(a, b)
            res.append(dict(analysis=name, **dict(zip(keys, k if isinstance(k, tuple) else (k,))), n_Rplus=a.notna().sum(),
                            n_Rminus=b.notna().sum(), delta_Rplus=a.mean(), p_Rplus=signflip(a), delta_Rminus=b.mean(),
                            p_Rminus=signflip(b), p_cohort_mw=pm, p_cohort_welch=pw))
    R = pd.DataFrame(res)
    R.to_csv(T / "026_group.csv", index=False)
    # baseline: how often is there an evoked movement at all (whisker vs auditory, hits vs misses)
    base = TR.groupby(["session_id", "cohort", "stim", "lick"])[[f"pres_{m}" for m in MEAS] + [f"mag_{m}" for m in MEAS]].mean().reset_index()
    base.to_csv(T / "026_evoked_rates_per_session.csv", index=False)
    figure(TR, A, B, C, R, base)
    json.dump(dict(script="026_evoked_reinforcement.py", window=WIN, measures=MEAS, versions=L.versions()),
              open(L.RES / "provenance" / "026.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(base.groupby(["cohort", "stim", "lick"])[[f"pres_{m}" for m in MEAS]].mean().round(3).to_string())
    print(R.round(4).to_string())


def figure(TR, A, B, C, R, base):
    fig = plt.figure(figsize=(7.4, 7.2))
    gs = fig.add_gridspec(3, len(MEAS), hspace=0.8, wspace=0.6, left=0.08, right=0.98, top=0.92, bottom=0.08)
    for j, m in enumerate(MEAS):
        # row 1: fraction present over the session (whisker misses / hits, deciles of the whisker trials), per cohort
        ax = fig.add_subplot(gs[0, j])
        g = TR[TR.stim == "whisker"].copy()
        g["dec"] = g.groupby("session_id").wti.transform(lambda x: np.minimum((x.rank(pct=True) * 5).astype(int), 4))
        for cg in ("R+", "R-"):
            for lk, ls in ((0, "--"), (1, "-")):
                q = g[(g.cohort == cg) & (g.lick == lk)].groupby(["session_id", "dec"])[f"pres_{m}"].mean().unstack()
                mm, ee = q.mean(), q.sem()
                ax.errorbar(mm.index + (0.05 if cg == "R-" else 0), mm, ee, color=L.COH_COL[cg], ls=ls, lw=0.7, capsize=0,
                            label=f"{cg} {'hits' if lk else 'misses'}")
        ax.axhline(0.05, color="k", lw=0.4, ls=":")
        ax.set_xticks(range(5))
        ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=4.5)
        ax.set_xlabel("fifth of whisker trials", fontsize=5)
        ax.set_title(f"P(evoked) {m.replace('_', ' ')}", fontsize=5.3)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        if j == 0:
            ax.set_ylabel("fraction of whisker trials\n(dotted: CR false-positive rate)", fontsize=5)
            ax.legend(frameon=False, fontsize=3.8)
        # row 2: trend late - early (all whisker trials, present) per cohort
        ax = fig.add_subplot(gs[1, j])
        for h, (an, df, extra) in enumerate((("late − early", A, {"subset": "all whisker"}), ("after − before LT", B, {"subset": "all whisker"}))):
            q = df[(df.measure == m) & (df.kind == "present") & (df.subset == extra["subset"])]
            for c_, cg in enumerate(("R+", "R-")):
                v = q[q.cohort == cg].delta.dropna()
                x = h * 2.4 + c_
                ax.scatter(x + np.random.default_rng(3).uniform(-0.15, 0.15, len(v)), v, s=2, color=L.COH_COL[cg], lw=0, alpha=0.6)
                ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color="k", lw=0.9)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xticks([0.5, 2.9])
        ax.set_xticklabels(["late − early", "after − before LT"], fontsize=4.5)
        ax.set_title("Δ P(evoked), whisker trials", fontsize=5.3)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        # row 3: trial-to-trial after hit - after miss (whisker and auditory)
        ax = fig.add_subplot(gs[2, j])
        for h, st in enumerate(("whisker", "auditory")):
            q = C[(C.measure == m) & (C.kind == "present") & (C.stim == st) & (C.subset == "next: all")]
            for c_, cg in enumerate(("R+", "R-")):
                v = q[q.cohort == cg].delta.dropna()
                x = h * 2.4 + c_
                ax.scatter(x + np.random.default_rng(4).uniform(-0.15, 0.15, len(v)), v, s=2, color=L.COH_COL[cg], lw=0, alpha=0.6)
                ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color="k", lw=0.9)
                r = R[(R.analysis == "C after hit - after miss") & (R.stim == st) & (R.subset == "next: all") & (R.measure == m) & (R.kind == "present")]
                if len(r):
                    p = r[f"p_{'Rplus' if cg == 'R+' else 'Rminus'}"].iloc[0]
                    if p < 0.05:
                        ax.text(x, 0.95, "*", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=L.COH_COL[cg])
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xticks([0.5, 2.9])
        ax.set_xticklabels(["whisker", "auditory"], fontsize=4.5)
        ax.set_title("next trial: after hit − after miss", fontsize=5.3)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    fig.suptitle("Whisker-stimulus-evoked movement (25-150 ms, 200 Hz, pre-lick) and reward: frequency over the session, at the learning "
                 "trial and after rewarded vs unrewarded hits (R+ green: whisker hits rewarded; R− magenta: not)", fontsize=5.8, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"026_evoked_reinforcement.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
