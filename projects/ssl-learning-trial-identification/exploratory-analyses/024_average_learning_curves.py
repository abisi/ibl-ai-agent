"""024 -- Cohort-averaged learning curves (user 2026-09-30: "average these curves together. Whisker only, auditory only,
false alarm only, then all together ... one per cohort, cohorts overlaid, then differences whisker - false alarm, etc.").
Per session: the same exact HMM curves as 021 (sigma = 1, posterior mean P(lick)); whisker on its own trials, auditory
and no-stim (false alarm) fitted on their own trials and time-interpolated onto the whisker-trial axis. One session per
mouse (day 0), so mouse = unit.
Scopes (ssl-analyze rule: report both): all mice | learners only (learning_category in {good, moderate}).
x axes (one figure each):
  trial     whisker-trial index; a cohort's mean is drawn while >= 50% of its mice still have trials (R- sessions are
            longer), so the tail is not a few mice;
  progress  normalised session progress (first -> last whisker trial = 0 -> 1, 101 points), every mouse contributes
            everywhere.
Layout (A4 wide, 2 x 4 square panels):
  a whisker  b auditory  c false alarm (cohorts overlaid, mean +- SEM across mice)
  d whisker - false alarm (cohorts overlaid)
  e all trial types, R+   f all trial types, R-   (auditory blue, whisker cohort colour, false alarm grey)
  g auditory - whisker    h auditory - false alarm (cohorts overlaid)
No header text; one-line colour legend at the bottom.
Stats: per x point R+ vs R- Mann-Whitney AND Welch (both, uncorrected); bars at the top of a-d, g, h mark p < 0.05
(dark = Mann-Whitney, light = Welch). All per-mouse curves + stats saved with provenance.
Two versions (user 2026-09-30: "replot ... with trimming, but keep the version without trimming"):
  untrimmed  full sessions (as before);
  trimA1     end-of-session disengagement removed with rule A1 (the rule chosen for neural analyses, 026): the tail
             after the session's last lick on any trial type (from the first whisker trial on) is dropped when it holds
             >= 5 whisker AND >= 1 auditory trial; the HMM curves are REFIT on the remaining trials (all trial types
             with start_time < the first tail trial).
Outputs: 024_avg_curves_<scope>_<axis>{,_trimA1}.{pdf,png,svg}; artifacts/024_avg_curves_per_mouse{,_trimA1}.pkl,
  024_avg_curves_stats.csv (column trim)
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/024_average_learning_curves.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import lt_lib as L  # noqa: E402

SIGMA = 1.0
GROUP_COLORS = {"rplus": "#00B400", "rminus": "#C800C8"}   # ephys_utilities GROUP_COLORS (not importable: cmasher)
COL = {"R+": GROUP_COLORS["rplus"], "R-": GROUP_COLORS["rminus"]}
AUD, FA = "#1f5fbf", "0.35"
FS_L, FS_M, FS_S = 9, 8, 7
NPROG = 101
MIN_FRAC = 0.5


def post_mean(y):
    return L.forward_backward(np.asarray(y, int), SIGMA)[0]


def session_trials(inp, tt, untrimmed):
    yw, tw = np.asarray(inp["w_outcomes"], int), np.asarray(inp["w_start"], float)
    yn, tn = np.asarray(inp["n_outcomes"], int), np.asarray(inp["n_start"], float)
    if "a_outcomes" in inp:
        ya, ta = np.asarray(inp["a_outcomes"], int), np.asarray(inp["a_start"], float)
    else:
        u = untrimmed(inp["session_id"], tt)
        a = u[u.trial_type == "auditory_trial"]
        ya, ta = a.lick_flag.to_numpy().astype(int), a.start_time.to_numpy().astype(float)
    return yw, tw, yn, tn, ya, ta


MIN_TAIL_WHISKER, MIN_TAIL_AUDITORY = 5, 1         # rule A1 (user 2026-09-30)


def a1_cut(yw, tw, yn, tn, ya, ta):
    """Start time of the disengaged tail under rule A1 (None if the session is not flagged). Trials of all types
    from the first whisker trial on; tail = trials after the last lick; flagged if it holds >= 5 whisker and >= 1
    auditory trials."""
    t = np.concatenate([tw, tn, ta])
    y = np.concatenate([yw, yn, ya])
    typ = np.concatenate([np.full(len(tw), "w"), np.full(len(tn), "n"), np.full(len(ta), "a")])
    keep = t >= tw[0] - 1e-9
    o = np.argsort(t[keep], kind="stable")
    t, y, typ = t[keep][o], y[keep][o], typ[keep][o]
    lick = np.where(y == 1)[0]
    i0 = lick[-1] + 1 if len(lick) else 0
    if i0 >= len(t):
        return None
    tail = typ[i0:]
    if (tail == "w").sum() >= MIN_TAIL_WHISKER and (tail == "a").sum() >= MIN_TAIL_AUDITORY:
        return float(t[i0])
    return None


def session_curves(yw, tw, yn, tn, ya, ta):
    c = {"whisker": post_mean(yw) @ L.P_GRID, "fa": L.interp_marginals(post_mean(yn), tn, tw) @ L.P_GRID}
    c["auditory"] = L.interp_marginals(post_mean(ya), ta, tw) @ L.P_GRID if len(ya) >= 3 else np.full(len(yw), np.nan)
    return c, dict(n_whisker=len(yw), n_nostim=len(yn), n_auditory=len(ya))


def trim(arrs, t_cut):
    yw, tw, yn, tn, ya, ta = arrs
    return yw[tw < t_cut], tw[tw < t_cut], yn[tn < t_cut], tn[tn < t_cut], ya[ta < t_cut], ta[ta < t_cut]


def derived(c):
    return {**c, "w-fa": c["whisker"] - c["fa"], "a-w": c["auditory"] - c["whisker"], "a-fa": c["auditory"] - c["fa"]}


def on_axis(c, axis, nmax):
    n = len(c["whisker"])
    if axis == "trial":
        return {k: np.pad(v, (0, nmax - n), constant_values=np.nan) for k, v in c.items()}
    x = np.linspace(0, 1, n)
    return {k: np.interp(np.linspace(0, 1, NPROG), x, v) for k, v in c.items()}


def mean_sem(M):
    ok = np.isfinite(M)
    n = ok.sum(0)
    with np.errstate(all="ignore"):
        m = np.nanmean(M, 0)
        s = np.nanstd(M, 0, ddof=1) / np.sqrt(n)
    return m, s, n


def cohort_tests(A, B, valid):
    pm, pw = np.full(A.shape[1], np.nan), np.full(A.shape[1], np.nan)
    for j in np.where(valid)[0]:
        a, b = A[:, j][np.isfinite(A[:, j])], B[:, j][np.isfinite(B[:, j])]
        if len(a) >= 3 and len(b) >= 3:
            pm[j] = stats.mannwhitneyu(a, b).pvalue
            pw[j] = stats.ttest_ind(a, b, equal_var=False).pvalue
    return pm, pw


def sig_bars(ax, x, pm, pw, y0):
    for p, yy, c in ((pm, y0, "0.15"), (pw, y0 - 0.035, "0.6")):
        s = np.isfinite(p) & (p < 0.05)
        for i0, i1 in L.runs(s):
            ax.plot([x[i0] - 0.5 * (x[1] - x[0]), x[i1 - 1] + 0.5 * (x[1] - x[0])], [yy, yy], color=c, lw=2.2,
                    solid_capstyle="butt", transform=ax.get_xaxis_transform(), clip_on=False)


def figure(D, scope, axis, rows, tag=""):
    plt.rcParams.update({"font.family": "Arial", "font.size": FS_M, "axes.labelsize": FS_M, "xtick.labelsize": FS_S,
                         "ytick.labelsize": FS_S, "axes.linewidth": 0.9, "xtick.major.width": 0.9,
                         "ytick.major.width": 0.9, "xtick.major.size": 3, "ytick.major.size": 3,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 4.5))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.14, wspace=0.42, hspace=0.62)
    npts = next(iter(D.values()))["whisker"].shape[1]
    x = np.arange(1, npts + 1) if axis == "trial" else np.linspace(0, 1, NPROG)
    xlab = "whisker trial" if axis == "trial" else "session progress"
    valid = {}
    for rg in ("R+", "R-"):
        n = np.isfinite(D[rg]["whisker"]).sum(0)
        valid[rg] = n >= MIN_FRAC * D[rg]["whisker"].shape[0]
    xmax = x[valid["R+"] | valid["R-"]].max()

    def draw(ax, key, rg, color, lw=1.6):
        m, s, _ = mean_sem(D[rg][key])
        v = valid[rg] & np.isfinite(m)
        ax.fill_between(x[v], (m - s)[v], (m + s)[v], color=color, alpha=0.22, lw=0)
        ax.plot(x[v], m[v], color=color, lw=lw)

    panels = [(axes[0, 0], "whisker", "Whisker", "P(lick)"), (axes[0, 1], "auditory", "Auditory", "P(lick)"),
              (axes[0, 2], "fa", "False alarm (no stim)", "P(lick)"),
              (axes[0, 3], "w-fa", "Whisker − false alarm", "Δ P(lick)"),
              (axes[1, 2], "a-w", "Auditory − whisker", "Δ P(lick)"),
              (axes[1, 3], "a-fa", "Auditory − false alarm", "Δ P(lick)")]
    for ax, key, title, ylab in panels:
        for rg in ("R+", "R-"):
            draw(ax, key, rg, COL[rg])
        both = valid["R+"] & valid["R-"]
        pm, pw = cohort_tests(D["R+"][key], D["R-"][key], both)
        sig_bars(ax, x, pm, pw, 1.02)
        for j in range(len(x)):
            rows.append(dict(trim=tag.strip("_") or "none", scope=scope, axis=axis, curve=key, x=x[j], n_rplus=int(np.isfinite(D["R+"][key][:, j]).sum()),
                             n_rminus=int(np.isfinite(D["R-"][key][:, j]).sum()),
                             mean_rplus=np.nanmean(D["R+"][key][:, j]) if valid["R+"][j] else np.nan,
                             mean_rminus=np.nanmean(D["R-"][key][:, j]) if valid["R-"][j] else np.nan,
                             p_mannwhitney=pm[j], p_welch=pw[j]))
        if key.count("-"):
            ax.axhline(0, color="0.6", lw=0.6, ls=":")
            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        else:
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.5, 1])
        ax.set_title(title, fontsize=FS_M, pad=9)
        ax.set_ylabel(ylab)
    for ax, rg in ((axes[1, 0], "R+"), (axes[1, 1], "R-")):
        draw(ax, "auditory", rg, AUD)
        draw(ax, "whisker", rg, COL[rg], lw=1.9)
        draw(ax, "fa", rg, FA)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("P(lick)")
        nm = D[rg]["whisker"].shape[0]
        ax.set_title(f"{'R+' if rg == 'R+' else 'R−'} (n = {nm} mice)", fontsize=FS_M, pad=9, color=COL[rg])
    for ax, lett in zip(axes.flat, "abcdefgh"):
        ax.set_xlim(x[0] - (0.5 if axis == "trial" else 0), xmax + (0.5 if axis == "trial" else 0))
        if axis == "progress":
            ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(xlab)
        ax.set_box_aspect(1)
        ax.text(-0.36, 1.13, lett, transform=ax.transAxes, fontsize=FS_L, fontweight="bold")
    n = {rg: D[rg]["whisker"].shape[0] for rg in D}
    # compact one-line legend at the bottom
    fig.text(0.5, 0.015, "", ha="center")
    items = [(COL["R+"], f"R+ (n = {n['R+']})", 0.105), (COL["R-"], f"R− (n = {n['R-']})", 0.105),
             (AUD, "auditory (e, f)", 0.11), (FA, "false alarm (e, f)", 0.125)]
    xs = 0.04
    for c, lab, w in items:
        fig.lines.append(plt.Line2D([xs, xs + 0.025], [0.025, 0.025], color=c, lw=2, transform=fig.transFigure))
        fig.text(xs + 0.03, 0.025, lab, va="center", fontsize=FS_S, color=c)
        xs += w
    fig.lines.append(plt.Line2D([xs, xs + 0.025], [0.025, 0.025], color="0.15", lw=2.2, transform=fig.transFigure))
    fig.text(xs + 0.03, 0.025, "R+ vs R− Mann-Whitney p < 0.05", va="center", fontsize=FS_S)
    xs += 0.21
    fig.lines.append(plt.Line2D([xs, xs + 0.025], [0.025, 0.025], color="0.6", lw=2.2, transform=fig.transFigure))
    fig.text(xs + 0.03, 0.025, "Welch p < 0.05 (uncorrected)", va="center", fontsize=FS_S)
    stem = HERE / f"024_avg_curves_{scope}_{axis}{tag}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(stem)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed
    tt = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "trials.parquet")
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    inputs.update(pickle.load(open(ART / "022_inputs_behaviour_only.pkl", "rb")))
    pers = {"": [], "_trimA1": []}
    for sid, inp in inputs.items():
        arrs = session_trials(inp, tt, _active_trials_for_curve_untrimmed)
        t_cut = a1_cut(*arrs)
        for tag, a in (("", arrs), ("_trimA1", trim(arrs, t_cut) if t_cut is not None else arrs)):
            c, meta = session_curves(*a)
            pers[tag].append(dict(session_id=sid, mouse_id=inp["mouse_id"], reward_group=inp["reward_group"],
                                  learning_category=inp.get("learning_category"),
                                  source="001 ephys (ssl_ephys trials)" if "a_outcomes" not in inp else "022 NWB_ks4",
                                  **meta, trimmed=bool(tag and t_cut is not None), t_cut=t_cut if tag else None,
                                  n_whisker_full=len(arrs[0]), curves=derived(c)))
    rows = []
    for tag, per in pers.items():
        cfg = dict(sigma=SIGMA, n_progress=NPROG, min_frac=MIN_FRAC, learners=["good", "moderate"],
                   script=Path(__file__).name,
                   trim="rule A1 (tail after last lick, >= 5 whisker and >= 1 auditory)" if tag else "none")
        pickle.dump(dict(sessions=per, config=cfg), open(ART / f"024_avg_curves_per_mouse{tag}.pkl", "wb"))
        print(tag or "untrimmed", "sessions trimmed:", sum(p["trimmed"] for p in per),
              "whisker trials removed:", sum(p["n_whisker_full"] - p["n_whisker"] for p in per))
        for scope in ("all", "learners"):
            sel = [p for p in per if scope == "all" or p["learning_category"] in ("good", "moderate")]
            for axis in ("trial", "progress"):
                nmax = max(p["n_whisker"] for p in sel)
                D = {rg: {k: np.vstack([on_axis(p["curves"], axis, nmax)[k] for p in sel if p["reward_group"] == rg])
                          for k in sel[0]["curves"]} for rg in ("R+", "R-")}
                figure(D, scope, axis, rows, tag)
    pd.DataFrame(rows).to_csv(ART / "024_avg_curves_stats.csv", index=False)


if __name__ == "__main__":
    main()
