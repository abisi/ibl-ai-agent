"""Which areas carry the pre-movement R- > R+ between-area coupling on whisker licks (jaw-aligned)? Exploratory.

Window "prep" = -0.7 to -0.2 s before the jaw onset (post hoc: chosen from the v2 jaw time course, where R- > R+ in
corr50 from -0.6 to -0.25 s on W_lick). Lick conditions W_lick, A_hit, NS_fa; area groups and acronyms.
Inputs:  OUT/v2_jaw_prep/sessions/*_bands.parquet (002_run_spikes.py --align jaw --windows prep)
Unit:    mouse; excess = value - shuffle-null mean; Mann-Whitney + Welch uncorrected (both p < 0.05 flagged)
Tables:  OUT/v2_jaw_prep/stats/<pop>/{omnibus,pairs,areas}.csv
         omnibus = per mouse mean over its between-area pairs; areas = per mouse and area, mean over the pairs that
         include the area ("participation"); pairs = per pair
Figure:  OUT/v2_jaw_prep/figures/<pop>/e5_prep_breakdown.{png,pdf,svg}
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-spectral-coupling/008_prep_breakdown.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_lib as sl  # noqa: E402

_s = importlib.util.spec_from_file_location("ex", HERE / "007_exploratory_stats.py")
ex = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ex)

ROOT = sl.OUT / "v2_jaw_prep"
CONDS = ["W_lick", "A_hit", "NS_fa"]
FOCUS = [("corr50", "all"), ("msc", "beta"), ("msc", "theta")]
N_TOP = 12


def load():
    d = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "sessions").glob("*_bands.parquet"))], ignore_index=True)
    d = d[d.measure.isin(["corr50", "msc"]) & (d.window == "prep")].copy()
    d["excess"] = d.value - d.null_mean
    d = d[~d.source.astype(str).str.endswith(ex.WITHIN)]
    d["pair"] = d.source.astype(str) + " | " + d.target.astype(str)
    return d


def tests(d):
    keys = ["level", "condition", "measure", "band"]
    pm = d.groupby(["mouse_id", "cohort"] + keys + ["pair"], observed=True).excess.mean().reset_index()
    om = pm.groupby(["mouse_id", "cohort"] + keys, observed=True).excess.mean().reset_index()
    long = pd.concat([pm.assign(area=pm.pair.str.split(" | ", regex=False).str[0]),
                      pm.assign(area=pm.pair.str.split(" | ", regex=False).str[1])])
    am = long.groupby(["mouse_id", "cohort"] + keys + ["area"], observed=True).excess.mean().reset_index()
    return ex.test_by(om, keys), ex.test_by(pm, keys + ["pair"]), ex.test_by(am, keys + ["area"]), om


def bars(ax, t, label, title):
    t = t[np.isfinite(t.t_welch)].sort_values("t_welch")
    t = pd.concat([t.head(N_TOP // 2), t.tail(N_TOP // 2)]).drop_duplicates(label) if len(t) > N_TOP else t
    y = np.arange(len(t))
    from matplotlib.colors import to_rgba
    ax.barh(y, t.t_welch, color=[to_rgba(ex.COL["R+"] if v > 0 else ex.COL["R-"], 0.9 if b else 0.35)
                                 for v, b in zip(t.t_welch, t.both_p05)])
    for yi, r in zip(y, t.itertuples()):
        if r.both_p05:
            ax.text(r.t_welch, yi, " *" if r.t_welch > 0 else "* ", va="center", ha="left" if r.t_welch > 0 else "right")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v} ({int(a)}/{int(b)})" for v, a, b in zip(t[label], t.n_Rplus, t.n_Rminus)], fontsize=5.5)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("Welch t (R+ − R−)")
    ax.set_title(title, fontsize=7.5)


def latency():
    """Median jaw-onset latency (s from stimulus, onsets 0-1 s) per mouse and lick trial type; R+ vs R- tests."""
    s = pd.read_csv(sl.OUT / "v2_jaw" / "sessions_included.csv")
    rows = []
    for r in s.itertuples():
        f = sl.ROOT / "combined_results_ks4" / r.mouse_id / "dlc_jaw_onset_times.pkl"
        if not f.exists():
            continue
        j = pd.read_pickle(f)
        j = j[(j.session_id == r.session_id) & (j.lick_flag == 1) & (j.early_lick == 0) & (j.context != "passive")]
        for tt in ("whisker_trial", "auditory_trial"):
            on = j[j.trial_type == tt].jaw_dlc_onset
            on = on[(on > 0) & (on < 1)]
            rows.append(dict(mouse_id=r.mouse_id, cohort=r.cohort, learner=bool(r.learner), trial_type=tt,
                             median_latency=on.median(), n_trials=len(on)))
    x = pd.DataFrame(rows)
    out = []
    for pop in ("all", "learners"):
        y = x if pop == "all" else x[x.learner]
        m = y.groupby(["mouse_id", "cohort", "trial_type"]).median_latency.mean().reset_index()
        t = ex.test_by(m.rename(columns={"median_latency": "excess"}), ["trial_type"])
        out.append(t.assign(population=pop))
    t = pd.concat(out, ignore_index=True)
    (ROOT / "stats").mkdir(parents=True, exist_ok=True)
    t.to_csv(ROOT / "stats" / "jaw_latency_tests.csv", index=False)
    print(t.round(4).to_string(), flush=True)


def main():
    latency()
    d = load()
    plt = ex.setup_mpl()
    for pop in ("all", "learners"):
        x = d if pop == "all" else d[d.learner]
        ot, pt, at, om = tests(x)
        sd = ROOT / "stats" / pop
        sd.mkdir(parents=True, exist_ok=True)
        ot.to_csv(sd / "omnibus.csv", index=False)
        pt.to_csv(sd / "pairs.csv", index=False)
        at.to_csv(sd / "areas.csv", index=False)
        print(pop, "\n", ot[ot.measure.isin(["corr50", "msc"])].round(4).to_string(), flush=True)
        for lv in ("group", "acronym"):
            a = at[(at.level == lv) & (at.condition == "W_lick") & (at.measure == "corr50")]
            print(lv, "areas W_lick corr50 both p<.05:", a[a.both_p05][["area", "n_Rplus", "n_Rminus", "diff", "p_mw",
                                                                         "p_welch"]].round(4).to_string(), flush=True)
        # figure: row 1 omnibus per condition (strip) for corr50 and msc beta; rows 2-3 areas / pairs, W_lick corr50
        fig = plt.figure(figsize=(10, 11))
        gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.4, 1.4], hspace=0.45, wspace=0.9)
        for j, (m, b) in enumerate(FOCUS[:2]):
            for k, lv in enumerate(("group", "acronym")):
                ax = fig.add_subplot(gs[0, 2 * j + k])
                for i, c in enumerate(CONDS):
                    for s, coh in ((-0.17, "R+"), (0.17, "R-")):
                        v = om[(om.level == lv) & (om.condition == c) & (om.measure == m) & (om.band == b) &
                               (om.cohort == coh)].excess
                        ax.scatter(i + s + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)), v, s=4,
                                   color=ex.COL[coh], alpha=0.6, lw=0)
                        if len(v):
                            ax.plot([i + s - 0.1, i + s + 0.1], [v.mean()] * 2, color="k", lw=1.2)
                labs = []
                for c in CONDS:
                    r = ot[(ot.level == lv) & (ot.condition == c) & (ot.measure == m) & (ot.band == b)]
                    labs.append(f"{c}\n{r.p_mw.iloc[0]:.3f}\n{r.p_welch.iloc[0]:.3f}" if len(r) else c)
                ax.set_xticks(range(len(CONDS)))
                ax.set_xticklabels(labs, fontsize=6)
                ax.axhline(0, color="0.6", lw=0.5, ls=":")
                ax.set_title(f"{m}{'' if b == 'all' else ' ' + b}, {lv}s\n(mean over pairs per mouse)", fontsize=7.5)
                if j == 0 and k == 0:
                    ax.set_ylabel("excess, −0.7 to −0.2 s\nbefore jaw onset")
        for k, lv in enumerate(("group", "acronym")):
            ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
            bars(ax, at[(at.level == lv) & (at.condition == "W_lick") & (at.measure == "corr50")], "area",
                 f"W_lick corr50, per area {lv} (mean over its pairs); n R+/R−")
            ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
            bars(ax, pt[(pt.level == lv) & (pt.condition == "W_lick") & (pt.measure == "corr50")], "pair",
                 f"W_lick corr50, area-{lv} pairs (most extreme {N_TOP}); n R+/R−")
        fig.suptitle(f"Pre-movement between-area coupling on lick trials, jaw-aligned, mouse level ({pop}); "
                     "top row: under each condition p Mann-Whitney, p Welch; * both p < 0.05 (uncorrected)", fontsize=9)
        ex.save(fig, ROOT / "figures" / pop / "e5_prep_breakdown")
        plt.close(fig)


if __name__ == "__main__":
    main()
