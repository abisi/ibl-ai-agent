"""Exploration (user 2026-11-09): lick probability over time after the stimulus, the corrected version of behaviour_analysis
beh_plotting_functions.plot_multi_mouse_piezo_lick_probability, in variants that show learning. Not part of Figure 1.

Fixes relative to the original (the original file is not edited):
  - lick times: NWB BehavioralEvents piezo_lick_times (+ the trial's registered lick, de-duplicated within 20 ms, as in
    spontaneous_licks_utils); the current standard trial table has no piezo_lick_times column, so the original cannot
    run on it
  - time zero = stimulus onset (trial start_time) instead of response_window_start_time (which sits an artifact
    window after the stimulus); window -0.5 to 1.5 s, 50-ms bins, P(at least one lick in the bin)
  - all trials and lick trials only (the original: lick trials only)
  - unit = mouse: per-session probability, then mean over sessions per mouse; mean +- s.e.m. over mice (the original:
    seaborn 95% bootstrap CI over session rows)
  - no mouse-ID filter (the original skipped every mouse numbered below 68, i.e. all MH mice up to MH067)
  - trials of the shared behaviour table: perf != 6, not passive, warm-up block trimmed, early licks dropped
  - performance states per whisker trial from the exact learning curves (session_curves 'state'); every trial takes
    the state of the most recent whisker trial

Variants (learners, whisker day 0): A, cohorts per trial type, all trials; B, the same on lick trials; C, high vs low
performance state per cohort, whisker trials, all trials; D, the same on lick trials.
Output: Figures/_explore/lick_proba_variants_<date>.{pdf,png} and the CSV.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_lick_proba.py [--check]
"""

import datetime
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, NWB_DIR, import_external_repos  # noqa: E402
from paperstyle import MM, set_nature_rc  # noqa: E402

WINDOW, BIN, TOL = (-0.5, 1.5), 0.05, 0.02
EDGES = np.round(np.arange(WINDOW[0], WINDOW[1] + BIN / 2, BIN), 3)
OUT = FIGURES_ROOT / "_explore"


def session_licks(session_id):
    """Sorted lick times of the session (piezo events + registered trial licks not within TOL of a piezo lick)."""
    import_external_repos()
    import NWB_reader_functions as nwbr
    nwb = NWB_DIR / f"{session_id}.nwb"
    ev = nwbr.get_behavioral_events(str(nwb))
    piezo = np.sort(np.asarray(ev.get("piezo_lick_times", []), float))
    tt = nwbr.get_trial_table(str(nwb))
    reg = tt["lick_time"].to_numpy(float)
    reg = reg[np.isfinite(reg)]
    if len(piezo):
        idx = np.clip(np.searchsorted(piezo, reg), 1, len(piezo) - 1)
        near = np.minimum(np.abs(piezo[idx] - reg), np.abs(piezo[idx - 1] - reg))
        reg_extra = reg[near > TOL]
    else:
        reg_extra = reg
    return session_id, piezo, np.sort(np.r_[piezo, reg_extra])


def trial_matrix(trials, licks):
    """P(lick in bin) per trial: trials x bins (0/1)."""
    m = np.zeros((len(trials), len(EDGES) - 1))
    for i, t0 in enumerate(trials.start_time.to_numpy()):
        lo, hi = np.searchsorted(licks, [t0 + WINDOW[0], t0 + WINDOW[1]])
        if hi > lo:
            m[i] = np.histogram(licks[lo:hi] - t0, EDGES)[0] > 0
    return m


def check(table):
    """Timing check: registered lick_time vs the nearest piezo lick, and vs start_time + computed_reaction_time."""
    d0 = table[(table.behavior == "whisker") & (table.day == 0)]
    for sid in d0.session_id.drop_duplicates().head(4):
        _, piezo, _ = session_licks(sid)
        s = d0[(d0.session_id == sid) & (d0.lick_flag == 1)]
        if not len(piezo):
            print(sid, "no piezo_lick_times")
            continue
        reg, corr = s.lick_time.to_numpy(float), (s.start_time + s.computed_reaction_time).to_numpy(float)
        for name, v in (("registered lick_time", reg), ("start + computed RT", corr)):
            i = np.clip(np.searchsorted(piezo, v), 1, len(piezo) - 1)
            nearest = np.where(np.abs(piezo[i] - v) < np.abs(piezo[i - 1] - v), piezo[i], piezo[i - 1])
            print(sid, name, "- nearest piezo lick (ms): median", round(np.median(v - nearest) * 1000, 1),
                  "IQR", np.round(np.percentile(v - nearest, [25, 75]) * 1000, 1))


def main():
    table = shared_behaviour.load()
    if "--check" in sys.argv:
        check(table)
        return
    from compute import day0_table, session_curves  # Figure 1 helpers (learners, exact curves, states)
    d0 = day0_table(table, learners_only=True)
    d0 = d0[d0.early_lick == 0]
    sids = list(d0.session_id.unique())
    with Pool(16) as pool:
        licks = {s: l for s, _, l in pool.map(session_licks, sids)}
    rows, ctx = [], {}
    for sid, st in d0.groupby("session_id"):
        if not len(licks[sid]):
            continue
        st = st.sort_values("start_time")
        curves, _, _ = session_curves(st, ctx)
        is_w = (st.trial_type == "whisker_trial").to_numpy()
        w_rank = np.cumsum(is_w) - 1
        state = np.where(w_rank >= 0, curves.state.to_numpy()[np.clip(w_rank, 0, len(curves) - 1)], "before")
        m = trial_matrix(st, licks[sid])
        t = st.start_time.to_numpy()
        tn = (t - t.min()) / max(t.max() - t.min(), 1e-9)  # normalised session time, first to last trial
        half = np.where(tn < 0.5, "first", "second")
        third = np.array(["early", "middle", "late"])[np.minimum((tn * 3).astype(int), 2)]
        combos = [(tt, lo, s, h) for tt in ("whisker_trial", "auditory_trial", "no_stim_trial") for lo in (False, True)
                  for s, h in [(None, None), ("high", None), ("low", None), (None, "first"), (None, "second"),
                               (None, "early"), (None, "middle"), (None, "late")]]
        period = {"first": half, "second": half, "early": third, "middle": third, "late": third}
        for tt, lick_only, st_state, h in combos:
            sel = ((st.trial_type == tt).to_numpy() & ((st.lick_flag.to_numpy() == 1) if lick_only else True)
                   & ((state == st_state) if st_state else True) & ((period[h] == h) if h else True))
            if sel.sum() == 0:
                continue
            rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                      "trial_type": tt, "lick_trials_only": lick_only, "state": st_state or "all",
                                      "half": h or "all",
                                      "time_s": np.round(EDGES[:-1] + BIN / 2, 3), "p_lick": m[sel].mean(0),
                                      "n_trials": int(sel.sum())}))
    df = pd.concat(rows, ignore_index=True)
    mouse = df.groupby(["mouse_id", "cohort", "trial_type", "lick_trials_only", "state", "half", "time_s"]
                       ).p_lick.mean().reset_index()
    plot(mouse[mouse.half == "all"])
    plot_halves(mouse[(mouse.state == "all") & mouse.half.isin(["first", "second"])],
                (("first", "-"), ("second", "--")), "halves")
    plot_halves(mouse[(mouse.state == "all") & mouse.half.isin(["early", "middle", "late"])],
                (("early", "-"), ("middle", "--"), ("late", ":")), "thirds")
    stem = OUT / f"lick_proba_variants_{datetime.date.today()}"
    mouse.to_csv(f"{stem}.csv", index=False)
    print(stem, df.groupby("state").session_id.nunique().to_dict(), "states seen:",
          sorted(set(np.concatenate([session_curves(s, ctx)[0].state.unique() for _, s in d0.groupby('session_id')]))))


def band(ax, d, color, ls="-", lw=0.8, label=None):
    g = d.groupby("time_s").p_lick
    m, s = g.mean(), g.sem()
    ax.plot(m.index, m, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.25, lw=0)


def plot(mouse):
    from plot import COHORTS, group_color
    set_nature_rc()
    TT = [("whisker_trial", "Whisker"), ("auditory_trial", "Auditory"), ("no_stim_trial", "No stimulus")]
    fig, axs = plt.subplots(4, 3, figsize=(183 * MM, 150 * MM), sharex=True)
    for r, lick_only in enumerate((False, True)):
        for c, (tt, name) in enumerate(TT):
            a = axs[r, c]
            for cohort in COHORTS:
                d = mouse[(mouse.trial_type == tt) & (mouse.lick_trials_only == lick_only) & (mouse.state == "all")
                          & (mouse.cohort == cohort)]
                band(a, d, group_color(tt, cohort), label=f"{cohort} (n={d.mouse_id.nunique()})")
            a.set_title(f"{name}, {'lick trials' if lick_only else 'all trials'}", fontsize=6, pad=2)
            a.legend(fontsize=4.5, frameon=False, loc="upper right")
    for r, lick_only in enumerate((False, True)):
        for c, cohort in enumerate(COHORTS):
            a = axs[2 + r, c]
            for st, ls in (("high", "-"), ("low", "--")):
                d = mouse[(mouse.trial_type == "whisker_trial") & (mouse.lick_trials_only == lick_only)
                          & (mouse.state == st) & (mouse.cohort == cohort)]
                if len(d):
                    band(a, d, group_color("whisker_trial", cohort), ls=ls, label=f"{st} (n={d.mouse_id.nunique()})")
            a.set_title(f"Whisker {cohort}, high vs low state, {'lick trials' if lick_only else 'all trials'}", fontsize=6, pad=2)
            a.legend(fontsize=4.5, frameon=False, loc="upper right")
        a = axs[2 + r, 2]  # R+ - R- per state
        for st, ls in (("high", "-"), ("low", "--")):
            d = mouse[(mouse.trial_type == "whisker_trial") & (mouse.lick_trials_only == lick_only) & (mouse.state == st)]
            for cohort in COHORTS:
                band(a, d[d.cohort == cohort], group_color("whisker_trial", cohort), ls=ls, lw=0.6)
        a.set_title(f"Whisker, both cohorts (solid high, dashed low)", fontsize=6, pad=2)
    for a in axs.flat:
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.set_ylim(-0.02, 1.02)
        a.set_xlim(*WINDOW)
    for a in axs[:, 0]:
        a.set_ylabel("P(lick in 50-ms bin)", labelpad=1)
    for a in axs[-1]:
        a.set_xlabel("Time from stimulus (s)", labelpad=1)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    stem = OUT / f"lick_proba_variants_{datetime.date.today()}"
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


def plot_halves(mouse, periods, tag):
    """Rows: all trials, lick trials; columns: whisker, auditory, no stimulus; one line style per session period
    (halves: first solid, second dashed; thirds of normalised session time: early solid, middle dashed, late dotted)."""
    from plot import COHORTS, group_color
    set_nature_rc()
    TT = [("whisker_trial", "Whisker"), ("auditory_trial", "Auditory"), ("no_stim_trial", "No stimulus")]
    fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 80 * MM), sharex=True)
    for r, lick_only in enumerate((False, True)):
        for c, (tt, name) in enumerate(TT):
            a = axs[r, c]
            for cohort in COHORTS:
                for h, ls in periods:
                    d = mouse[(mouse.trial_type == tt) & (mouse.lick_trials_only == lick_only) & (mouse.half == h)
                              & (mouse.cohort == cohort)]
                    if len(d):
                        band(a, d, group_color(tt, cohort), ls=ls, lw=0.7,
                             label=f"{cohort} {h} (n={d.mouse_id.nunique()})")
            a.set_title(f"{name}, {'lick trials' if lick_only else 'all trials'}", fontsize=6, pad=2)
            a.legend(fontsize=4, frameon=False, loc="upper right")
            a.axvline(0, color="0.6", lw=0.4, ls=":")
            a.set_ylim(-0.02, 0.62)
            a.set_xlim(*WINDOW)
    for a in axs[:, 0]:
        a.set_ylabel("P(lick in 50-ms bin)", labelpad=1)
    for a in axs[-1]:
        a.set_xlabel("Time from stimulus (s)", labelpad=1)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    stem = OUT / f"lick_proba_{tag}_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
