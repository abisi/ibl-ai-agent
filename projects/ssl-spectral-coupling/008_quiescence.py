"""Amount of quiescence (no licking) per session and cohort, whisker day 0 (user request 2026-11-18).

Quiet stretches from quiet_lib (reconstructed from piezo licks; CR trials add their response window). Per session:
  quiet_beh_median     s from max(previous trial stop, last lick) to the stimulus (the realised quiet window)
  since_lick_median    s from the last lick to the stimulus
  frac_iti_licks       fraction of inter-trial intervals with >= 1 lick (quiet-window restarts)
  iti_lick_rate        licks per s of inter-trial interval
  early_lick_frac      fraction of active trials aborted by an early lick (perf == 6)
  quiet_min            usable quiet time (stretches, 0.5 s after the last lick dropped), minutes
  quiet_frac           quiet_min / active session duration
  n_seg / n_seg_cr     1-s segments available for the spectral baseline (all / CR trials)
  quiet_after_lick / quiet_after_nolick   quiet_beh_median split by the previous trial's lick
R+ vs R-: Mann-Whitney + Welch (session = mouse at day 0), learners and all mice, uncorrected. Also the time course
over the session (5 normalised-time bins).
Outputs: OUT/v3_baseline/quiescence/{trials.parquet, sessions.csv, stats.csv, quiescence.{png,pdf}}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-spectral-coupling/008_quiescence.py
"""

import importlib.util
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quiet_lib as ql  # noqa: E402
import spec_lib as sl  # noqa: E402

OUT = sl.OUT / "v3_baseline" / "quiescence"
COL = {"R+": "#00B400", "R-": "#C800C8"}
SKIP = 0.5
METRICS = [("quiet_beh_median", "realised quiet window (s)"), ("since_lick_median", "time since last lick (s)"),
           ("frac_iti_licks", "ITIs with licks (frac.)"), ("iti_lick_rate", "ITI lick rate (Hz)"),
           ("early_lick_frac", "early-lick trials (frac.)"), ("quiet_min", "usable quiet time (min)"),
           ("quiet_frac", "quiet / session time"), ("n_seg", "1-s quiet segments"),
           ("n_seg_cr", "1-s segments (CR trials)"), ("quiet_after_lick", "quiet window after lick trial (s)"),
           ("quiet_after_nolick", "quiet window after no-lick trial (s)")]


def session_table():
    spec = importlib.util.spec_from_file_location("run002", HERE / "002_run_spikes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.session_table()[0]


def one(sid):
    try:
        t, licks = ql.read_behaviour(sid)
        st = ql.stretches(t, licks, skip=SKIP)
        seg = ql.segments(st)
        st.insert(0, "session_id", sid)
        st["n_seg"] = st.index.map(seg.trial.value_counts()).fillna(0).astype(int)
        act = t[(t.context != "passive") & (t.start_time >= st.start_time.min()) & (t.start_time <= st.start_time.max())]
        st["perf6_frac"] = float((act.perf == 6).mean())  # early-lick (aborted) trials in the same span
        return st
    except Exception as err:  # noqa: BLE001
        print(sid, "ERROR", err, flush=True)
        return None


def summarise(st):
    ok = st[st.early_lick == 0]
    dur = st.start_time.max() - st.start_time.min()
    usable = ok[(ok.q_len >= 0) & (ok.n_licks_inside == 0)]
    return dict(n_trials=len(st), quiet_beh_median=ok.quiet_beh.median(), since_lick_median=ok.since_last_lick.median(),
                frac_iti_licks=(st.n_iti_licks > 0).mean(), iti_lick_rate=st.n_iti_licks.sum() / st.iti.sum(),
                early_lick_frac=st.perf6_frac.iloc[0], quiet_min=usable.q_len.clip(lower=0).sum() / 60,
                quiet_frac=usable.q_len.clip(lower=0).sum() / dur, n_seg=int(st.n_seg.sum()),
                n_seg_cr=int(st.loc[st.cr, "n_seg"].sum()),
                quiet_after_lick=ok.loc[ok.prev_lick == 1, "quiet_beh"].median(),
                quiet_after_nolick=ok.loc[ok.prev_lick == 0, "quiet_beh"].median(),
                n_licks_inside_max=int(usable.n_licks_inside.max()) if len(usable) else 0)


def two_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(n_Rplus=len(a), n_Rminus=len(b), median_Rplus=np.median(a), median_Rminus=np.median(b),
                    mean_Rplus=a.mean(), mean_Rminus=b.mean(),
                    p_mw=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
                    p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = session_table()
    with Pool(30) as pool:
        res = pool.map(one, s.session_id.tolist())
    tr = pd.concat([r for r in res if r is not None], ignore_index=True)
    tr.to_parquet(OUT / "trials.parquet")
    rows = []
    for sid, st in tr.groupby("session_id"):
        rows.append(dict(session_id=sid, **summarise(st)))
    ss = s.merge(pd.DataFrame(rows), on="session_id")
    ss.to_csv(OUT / "sessions.csv", index=False)
    srows = []
    for pop, d in (("learners", ss[ss.learner]), ("all", ss)):
        for m, _ in METRICS:
            srows.append(dict(population=pop, metric=m, **two_tests(d.loc[d.cohort == "R+", m], d.loc[d.cohort == "R-", m])))
    sd = pd.DataFrame(srows)
    sd.to_csv(OUT / "stats.csv", index=False)
    print(sd.round(4).to_string())

    # time course: realised quiet window and ITI lick rate over 5 normalised-time bins
    tr = tr.merge(s[["session_id", "cohort", "learner"]], on="session_id")
    tr["tnorm"] = tr.groupby("session_id").start_time.transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    tr["tbin"] = np.minimum((tr.tnorm * 5).astype(int), 4)
    tc = tr[tr.early_lick == 0].groupby(["session_id", "cohort", "learner", "tbin"]).agg(
        quiet=("quiet_beh", "median"), lick_rate=("n_iti_licks", "sum"), iti=("iti", "sum")).reset_index()
    tc["lick_rate"] = tc.lick_rate / tc.iti
    tc.to_csv(OUT / "timecourse.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    nm = len(METRICS)
    fig, axs = plt.subplots(2, (nm + 2 + 1) // 2, figsize=(16, 6.5))
    axs = axs.ravel()
    d = ss[ss.learner]
    for ax, (m, lab) in zip(axs, METRICS):
        for k, c in enumerate(("R+", "R-")):
            v = d.loc[d.cohort == c, m].dropna()
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=COL[c], alpha=0.7, lw=0)
            ax.plot([k - 0.25, k + 0.25], [v.median()] * 2, color="k", lw=1.5)
        r = sd[(sd.population == "learners") & (sd.metric == m)].iloc[0]
        ra = sd[(sd.population == "all") & (sd.metric == m)].iloc[0]
        ax.set_title(f"{lab}\nlearners MW p={r.p_mw:.3f}, W p={r.p_welch:.3f}\nall MW p={ra.p_mw:.3f}", fontsize=7.5)
        ax.set_xticks([0, 1], ["R+", "R−"])
        ax.spines[["top", "right"]].set_visible(False)
    for ax, (m, lab) in zip(axs[nm:nm + 2], [("quiet", "realised quiet window (s)"), ("lick_rate", "ITI lick rate (Hz)")]):
        for c in ("R+", "R-"):
            g = tc[(tc.cohort == c) & tc.learner].groupby("tbin")[m]
            mu, se = g.mean(), g.sem()
            ax.errorbar(mu.index * 20 + 10, mu, se, color=COL[c], marker="o", ms=3, lw=1, label=c)
        ax.set_xlabel("session time (%)")
        ax.set_title(f"{lab} over the session (learners)", fontsize=7.5)
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[nm + 2:]:
        ax.axis("off")
    fig.suptitle(f"Quiescence, whisker day 0 (learners: R+ {int((d.cohort == 'R+').sum())}, "
                 f"R− {int((d.cohort == 'R-').sum())}; uncorrected)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"quiescence.{ext}", dpi=200)
    print("max licks inside a usable stretch:", ss.n_licks_inside_max.max())


if __name__ == "__main__":
    main()
