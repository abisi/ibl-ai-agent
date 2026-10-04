"""Are there RPE-like neurons in R+ and R-? Explainer figure for the RPE ROC v2 classes (learning day).

Strict classes (029): positive_RPE (R+) = O1 up & O2 up & O3 up; negative_RPE (R-) = O4 down & O5 down.
Cross-cohort control: O1 (R+) and O4 (R-) are the SAME computation (whisker-lick T1 outcome vs baseline), as are
O2 and O5 (T1 vs T3). So two symmetric patterns can be scored in both cohorts:
  up&decreasing   outcome > baseline in T1 and T1 > T3      (positive-RPE-like; predicted enriched in R+)
  down&recovering outcome < baseline in T1 and T1 < T3      (negative-RPE-like; predicted enriched in R-)
If these reflect reward prediction errors, up&decreasing must be more frequent in R+ (rewarded whisker licks) than in
R- (never rewarded), and down&recovering more frequent in R- (omission) than in R+.
Panels: A definitions; B fractions vs chance; C per-session excess, R+ vs R- (Mann-Whitney AND Welch); D lick rate
aligned to the first lick (reward drives a lick bout -> motor confound); E/F population PSTH of the pattern neurons in
each cohort (aligned to the first lick; whisker-lick thirds, auditory-hit halves); G examples.
Output: <summary>/rpe_explainer.png, rpe_pattern_sessions.csv
"""
import argparse
import importlib
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from scipy import stats                                                  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
m = importlib.import_module("029_roc_rpe_v2")
BINS = np.arange(-0.5, 0.801, 0.02)
TC = BINS[:-1] + 0.01
COH = {"R+": "#00B400", "R-": "#C800C8"}
SETS = {"W lick T1": "#98df8a", "W lick T2": "#2ca02c", "W lick T3": "#0b4d0b", "A hit H1": "#6baed6", "A hit H2": "#08306b"}
SETKEY = {"W lick T1": "W_lick_T1", "W lick T2": "W_lick_T2", "W lick T3": "W_lick_T3", "A hit H1": "A_hit_H1",
          "A hit H2": "A_hit_H2"}


def patterns(d, suf=""):
    rp = (d.cohort == "R+").to_numpy()
    p1 = np.where(rp, d[f"p{suf}_O1"], d[f"p{suf}_O4"]); a1 = np.where(rp, d[f"auc{suf}_O1"], d[f"auc{suf}_O4"])
    p2 = np.where(rp, d[f"p{suf}_O2"], d[f"p{suf}_O5"]); a2 = np.where(rp, d[f"auc{suf}_O2"], d[f"auc{suf}_O5"])
    up = (p1 < .05) & (a1 > .5) & (p2 < .05) & (a2 > .5)
    dn = (p1 < .05) & (a1 < .5) & (p2 < .05) & (a2 < .5)
    return up, dn


def psth(st, t0):
    if len(t0) == 0:
        return np.full(len(TC), np.nan)
    rel = np.concatenate([st[(st >= t + BINS[0]) & (st < t + BINS[-1])] - t for t in t0])
    return np.histogram(rel, BINS)[0] / len(t0) / 0.02


def worker(args):
    sid, nids = args
    nwb, units, act = m.load_session(sid)
    _, S = m.sets_for(act)
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    lick = {k: psth(pz, S[v].L1.to_numpy(float)) for k, v in SETKEY.items()}
    units = units[units.neuron_id.isin(nids)]
    ref = S["W_lick"] if len(S["W_lick"]) >= 2 else S["W_all"]
    out = {}
    for nid, s in zip(units.neuron_id, units.spike_times):
        st = np.sort(np.asarray(s))
        t0 = ref.start_time.to_numpy(float)
        b = np.array([np.sum((st >= t - 0.28) & (st < t - 0.01)) / 0.27 for t in t0])
        mu, sd = b.mean(), max(b.std(), 1.0)
        out[nid] = {k: (psth(st, S[v].L1.to_numpy(float)) - mu) / sd for k, v in SETKEY.items()}
    return sid, lick, out


def main(a):
    summ = m.SUMMARY / a.tag
    d = pd.read_parquet(summ / "rpe_v2_neurons.parquet")
    d = d[(d.stage == "learning") & (d.quality_label == "good")].reset_index(drop=True)
    assert len(d) and (d.quality_label == "good").all(), "good units only"
    d["up_dec"], d["down_rec"] = patterns(d)
    d["up_dec_null"], d["down_rec_null"] = patterns(d, "_null")
    ps = d.groupby(["session_id", "cohort"]).agg(n_units=("neuron_id", "size"), up_dec=("up_dec", "mean"),
                                                   up_dec_null=("up_dec_null", "mean"), down_rec=("down_rec", "mean"),
                                                   down_rec_null=("down_rec_null", "mean"),
                                                   pos_RPE=("class_positive_RPE", "mean"),
                                                   neg_RPE=("class_negative_RPE", "mean")).reset_index()
    for k in ["up_dec", "down_rec"]:
        ps[f"{k}_excess"] = ps[k] - ps[f"{k}_null"]
    ps.to_csv(summ / "rpe_pattern_sessions.csv", index=False)
    sel = d[d.up_dec | d.down_rec]
    jobs = [(sid, set(g.neuron_id)) for sid, g in sel.groupby("session_id")]
    lick_rows, psth_rows = [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for sid, lick, out in pool.map(worker, jobs):
            coh = sel.loc[sel.session_id == sid, "cohort"].iloc[0]
            lick_rows.append(dict(session_id=sid, cohort=coh, **lick))
            for nid, pp in out.items():
                psth_rows.append(dict(session_id=sid, neuron_id=nid, **pp))
            print("done", sid, flush=True)
    L = pd.DataFrame(lick_rows)
    P = pd.DataFrame(psth_rows).merge(sel[["session_id", "neuron_id", "cohort", "up_dec", "down_rec",
                                           "class_positive_RPE", "class_negative_RPE", "area_acronym_custom",
                                           "auc_O1", "auc_O2", "auc_O3", "auc_O4", "auc_O5", "auc_O6"]],
                                      on=["session_id", "neuron_id"])

    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.text(0, 1, "A  Definitions (whisker licks, outcome\n    window 30-300 ms after first lick)", fontsize=11,
            weight="bold", va="top", transform=ax.transAxes)
    ax.text(0, 0.78,
            "Strict classes\n"
            " positive RPE (R+): T1 > baseline, T1 > T3,\n"
            "                    whisker T1 > auditory H1\n"
            " negative RPE (R-): T1 < baseline, T1 < T3\n\n"
            "Symmetric patterns (same test, both cohorts)\n"
            " up & decreasing  : T1 > baseline and T1 > T3\n"
            " down & recovering: T1 < baseline and T1 < T3\n\n"
            "RPE prediction\n"
            " up & decreasing   more frequent in R+\n"
            " down & recovering more frequent in R-\n\n"
            "T1/T3 = first/last third of the session's\n"
            "whisker licks (R+ rewarded, R- unrewarded);\n"
            "set vs set on baseline-subtracted rates;\n"
            "uncorrected p < .05; good units only",
            fontsize=8.5, va="top", family="monospace", transform=ax.transAxes)
    # B fractions
    ax = fig.add_subplot(gs[0, 1]); w = 0.35
    groups = [("positive RPE\n(strict, R+ only)", "class_positive_RPE", None),
              ("negative RPE\n(strict, R- only)", "class_negative_RPE", None),
              ("up & decreasing", "up_dec", "up_dec_null"), ("down & recovering", "down_rec", "down_rec_null")]
    for j, (lab, col, ncol) in enumerate(groups):
        for k, coh in enumerate(COH):
            g = d[d.cohort == coh]
            if col == "class_positive_RPE" and coh == "R-" or col == "class_negative_RPE" and coh == "R+":
                continue
            v = g[col].mean(); x = j + (k - 0.5) * w
            ax.bar(x, v, w, color=COH[coh], label=coh if j == 2 else None)
            ax.text(x, v + 0.0008, f"{int(g[col].sum())}", ha="center", fontsize=6.5)
            if ncol:
                ax.plot([x - w / 2, x + w / 2], [g[ncol].mean()] * 2, "k-", lw=2, label="chance" if (j == 2 and k == 0) else None)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("fraction of good units"); ax.legend(fontsize=7)
    ax.set_title(f"B  Fraction of neurons (R+ {int((d.cohort == 'R+').sum())} units / "
                 f"{d[d.cohort == 'R+'].session_id.nunique()} sessions, R- {int((d.cohort == 'R-').sum())} / "
                 f"{d[d.cohort == 'R-'].session_id.nunique()})", fontsize=8.5, loc="left")
    # C per-session excess
    ax = fig.add_subplot(gs[0, 2]); rng = np.random.default_rng(0)
    for j, k_ in enumerate(["up_dec_excess", "down_rec_excess"]):
        x_, y_ = ps.loc[ps.cohort == "R+", k_], ps.loc[ps.cohort == "R-", k_]
        for k, (coh, v) in enumerate([("R+", x_), ("R-", y_)]):
            xx = j + (k - 0.5) * 0.4
            ax.scatter(xx + rng.uniform(-0.07, 0.07, len(v)), v, s=9, color=COH[coh], alpha=0.5)
            ax.errorbar(xx + 0.13, v.mean(), v.std() / np.sqrt(len(v)), color="k", marker="_", ms=12, capsize=3)
        pm, pw = stats.mannwhitneyu(x_, y_).pvalue, stats.ttest_ind(x_, y_, equal_var=False).pvalue
        ax.text(j, ax.get_ylim()[1] if False else max(x_.max(), y_.max()) * 1.02, f"MWU p={pm:.2g}\nWelch p={pw:.2g}",
                ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0, 1], ["up & decreasing", "down & recovering"]); ax.set_ylabel("fraction - chance, per session")
    ax.set_title("C  Per session (one point = one mouse), R+ vs R-", fontsize=9, loc="left")
    # D licking
    ax = fig.add_subplot(gs[0, 3])
    for coh, ls in [("R+", "-"), ("R-", "--")]:
        g = L[L.cohort == coh]
        for k in ["W lick T1", "W lick T3", "A hit H1"]:
            mu = np.nanmean(np.stack(g[k].to_list()), 0)
            ax.plot(TC * 1e3, np.convolve(mu, np.ones(3) / 3, "same"), color=SETS[k], ls=ls, lw=1.4, label=f"{coh} {k}")
    ax.axvspan(30, 300, color="orange", alpha=0.15); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("time from first lick (ms)"); ax.set_ylabel("lick rate (Hz)"); ax.legend(fontsize=6.5, ncol=2)
    ax.set_title("D  Licking (solid R+, dashed R-): reward\n    triggers a lick bout, omission does not", fontsize=9, loc="left")
    # E/F population PSTH
    for r_, (pat, lab) in enumerate([("up_dec", "up & decreasing"), ("down_rec", "down & recovering")]):
        for c_, coh in enumerate(COH):
            ax = fig.add_subplot(gs[1, 2 * r_ + c_])
            g = P[(P.cohort == coh) & P[pat]]
            for k, col in SETS.items():
                sm = g.groupby("session_id")[k].apply(lambda s: np.nanmean(np.stack(s.to_list()), 0))
                if not len(sm):
                    continue
                arr = np.stack(sm.to_list()); mu = np.nanmean(arr, 0); se = np.nanstd(arr, 0) / np.sqrt(len(arr))
                ax.plot(TC * 1e3, mu, color=col, lw=1.4, label=k); ax.fill_between(TC * 1e3, mu - se, mu + se, color=col, alpha=0.15)
            ax.axvspan(30, 300, color="orange", alpha=0.12); ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.4)
            ax.set_xlabel("time from first lick (ms)"); ax.set_ylabel("firing rate (z vs pre-trial)")
            ax.set_title(f"{'E' if r_ == 0 else 'F'}  {lab}, {coh}: {len(g)} neurons, {g.session_id.nunique()} sessions\n"
                         f"    (whisker licks {'rewarded' if coh == 'R+' else 'NOT rewarded'})", fontsize=9, loc="left")
            ax.legend(fontsize=6.5)
    # G examples (strict classes)
    ex = pd.concat([P[P.class_positive_RPE].assign(sc=lambda x: x.auc_O1 + x.auc_O2 + x.auc_O3).sort_values("sc", ascending=False)
                    .drop_duplicates("session_id").head(2),
                    P[P.class_negative_RPE].assign(sc=lambda x: -(x.auc_O4 + x.auc_O5)).sort_values("sc", ascending=False)
                    .drop_duplicates("session_id").head(2)])
    for i, (_, row) in enumerate(ex.iterrows()):
        ax = fig.add_subplot(gs[2, i])
        for k, col in SETS.items():
            ax.plot(TC * 1e3, np.convolve(row[k], np.ones(3) / 3, "same"), color=col, lw=1.3, label=k)
        ax.axvspan(30, 300, color="orange", alpha=0.12); ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.4)
        kind = "positive RPE" if row.class_positive_RPE else "negative RPE"
        ax.set_title(f"G  {kind} example: {row.session_id}\n    n{row.neuron_id} {row.area_acronym_custom} ({row.cohort})",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("time from first lick (ms)"); ax.set_ylabel("z vs pre-trial"); ax.legend(fontsize=6)
    fig.suptitle("RPE-like neurons, learning day (quality_label good; outcome 30-300 ms after first lick; uncorrected p)",
                 fontsize=13, y=0.995)
    fig.savefig(summ / "rpe_explainer.png", dpi=125, bbox_inches="tight")
    print("saved", summ / "rpe_explainer.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="learning")
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
