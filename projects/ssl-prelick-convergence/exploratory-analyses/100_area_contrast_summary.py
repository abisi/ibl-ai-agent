"""100 -- Summary of the per-area PSTH analyses (098, 099) as three within-session contrasts (user 2026-10-11: "too many dimensions;
example areas, then a forest plot across conditions and areas"). Exploratory.
Contrasts (per session x area group, same neurons, same pre-stimulus baseline logic):
  WH - WM   stimulus-aligned, 50-200 ms (099; lick vs no lick on the same stimulus)
  WH - AH   lick-aligned, [-100, 0) ms (098; whisker-cued vs auditory-cued rewarded lick)
  WH - SL   lick-aligned, [-100, 0) ms (098; cued vs spontaneous lick; SL with its own pre-lick baseline)
Whole brain: per session, unit-weighted mean over the area groups.
Left: example areas (rows) x contrasts (columns): contrast PSTHs, R+ green / R- magenta, day 0 dashed, expert solid, mean +- s.e.m.
over sessions; shaded: the test window.
Right: forest plot: per area (rows; whole brain first) and contrast (columns), the change expert - day 0 of the window value, R+ and
R- (mean +- combined s.e.m.); filled marker: day 0 vs expert within cohort p < 0.05 (Mann-Whitney); * after the area name: cohort x
stage interaction (mouse-level permutation) BH q < 0.05 within contrast, (*) p < 0.05 uncorrected. Areas with >= 3 sessions per
cohort x stage in the contrast.
Population: --pop learners (learners rule: day-0 sessions of non-learner mice dropped) or all.
Output: across_days/<ref>/area_contrast_summary/area_contrast_summary_<pop>.{png,pdf}, stats_<pop>.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../100_area_contrast_summary.py --pop learners
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m98 = importlib.import_module("098_area_psth_contrasts")
m99 = importlib.import_module("099_area_psth_misses")
m88, m51, m61, m62 = m98.m88, m98.m51, m98.m61, m98.m62
COH = m88.COH
OUT = m51.OUTROOT / "area_contrast_summary"
CONTR = [("WH_WM", "WH − WM\nstim, 50–200 ms", "stim", (50, 200)), ("WH_AH", "WH − AH\nlick, −100–0 ms", "lick", (-100, 0)),
         ("WH_SL", "WH − SL\nlick, −100–0 ms", "lick", (-100, 0))]
EXAMPLES = ["Thalamus", "Motor areas", "Hippocampus"]


def build(learners):
    D98, tc98 = m98.load(learners=learners)
    rows = []
    for _, r in D98.iterrows():
        for k, a, b in (("WH_AH", "WH", "AH"), ("WH_SL", "WH", "SL")):
            rows.append(dict(session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, area_group=r.area_group,
                             n_units=r.n_units, contrast=k, t=tc98, psth=r[a] - r[b]))
    P = pd.read_parquet(m99.OUT / "psth.parquet")
    P = m61.learner_filter(P) if learners else P
    P = P[P["align"] == "stim"]; P["psth"] = P.psth.apply(np.asarray)
    W = P.pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "area_group", "n_units"], columns="cls", values="psth", aggfunc="first").reset_index()
    lo, hi = m99.WINS["stim"]; tc99 = ((np.arange(lo, hi + m99.BIN / 2, m99.BIN)[:-1] + m99.BIN / 2) * 1e3)[1:-1]
    for _, r in W.dropna(subset=["WH", "WM"]).iterrows():
        rows.append(dict(session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, area_group=r.area_group,
                         n_units=r.n_units, contrast="WH_WM", t=tc99, psth=r.WH - r.WM))
    D = pd.DataFrame(rows)
    win = {c: w for c, _, _, w in CONTR}
    D["v"] = [p[(t >= win[c][0]) & (t < win[c][1])].mean() for p, t, c in zip(D.psth, D.t, D.contrast)]
    WB = []                                                                      # whole brain: unit-weighted mean over areas
    for (sid, c), g in D.groupby(["session_id", "contrast"]):
        w = g.n_units.to_numpy(float) / g.n_units.sum()
        WB.append(dict(g.iloc[0][["session_id", "mouse_id", "cohort", "stage"]].to_dict(), area_group="Whole brain", n_units=int(g.n_units.sum()),
                       contrast=c, t=g.t.iloc[0], psth=np.sum([wi * p for wi, p in zip(w, g.psth)], 0), v=float(np.sum(w * g.v.to_numpy()))))
    return pd.concat([pd.DataFrame(WB), D], ignore_index=True)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    OUT.mkdir(parents=True, exist_ok=True)
    plt = m62.setup(); rng = np.random.default_rng(0); gcol = au.get_custom_area_groups_colors()
    D = build(a.pop == "learners")
    n = D.groupby(["area_group", "contrast", "cohort", "stage"]).session_id.nunique()
    ok = lambda ag, c: all(n.get((ag, c, co, s), 0) >= 3 for co in ("R+", "R-") for s in ("learning", "expert"))
    areas = ["Whole brain"] + [g for g in au.get_area_group_custom_order() if g in D.area_group.unique() and any(ok(g, c) for c, *_ in CONTR)]
    ST = []
    for c, *_ in CONTR:
        for ag in areas:
            if not ok(ag, c):
                continue
            d = D[(D.area_group == ag) & (D.contrast == c)]
            r = dict(area=ag, contrast=c)
            for co in ("R+", "R-"):
                L_, E_ = (d[(d.cohort == co) & (d.stage == s_)].v for s_ in ("learning", "expert"))
                r[f"chg_{co}"] = E_.mean() - L_.mean(); r[f"sem_{co}"] = np.sqrt(L_.sem() ** 2 + E_.sem() ** 2)
                r[f"p_{co}"] = stats.mannwhitneyu(L_, E_).pvalue; r[f"n_{co}"] = f"{len(L_)}/{len(E_)}"
            S = m62.group_stats(d, "v", f"{ag} {c}", rng, "100"); r["p_int"] = S["interaction"][1]
            ST.append(r)
    ST = pd.DataFrame(ST)
    for c, *_ in CONTR:
        m = ST.contrast == c; ST.loc[m, "q_int"] = m98.bh(ST.loc[m, "p_int"])
    ST.to_csv(OUT / f"stats_{a.pop}.csv", index=False)
    # ---------------- figure
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.62))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.25, left=0.06, right=0.99, top=0.86, bottom=0.1)
    ge = gs[0, 0].subgridspec(len(EXAMPLES), 3, hspace=0.45, wspace=0.45)
    for i, ag in enumerate(EXAMPLES):
        for j, (c, lab, al, w) in enumerate(CONTR):
            ax = fig.add_subplot(ge[i, j]); d = D[(D.area_group == ag) & (D.contrast == c)]
            for co in ("R+", "R-"):
                for s_, ls, al_ in (("learning", (0, (2.5, 1.5)), 0.6), ("expert", "-", 1.0)):
                    g = d[(d.cohort == co) & (d.stage == s_)]
                    if not len(g):
                        continue
                    A = np.stack(g.psth.to_numpy()); x = g.t.iloc[0]; mu = A.mean(0); se = A.std(0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(x, mu - se, mu + se, color=COH[co], alpha=0.1, lw=0); ax.plot(x, mu, color=COH[co], ls=ls, lw=0.8, alpha=al_)
            ax.axvspan(*w, color="#FDD49E" if al == "lick" else "#DDDDDD", alpha=0.45, lw=0, zorder=0)
            ax.axvline(0, color="0.4", lw=0.4, ls=":"); ax.axhline(0, color="0.85", lw=0.4); ax.set_box_aspect(1); ax.tick_params(labelsize=4.2)
            if i == 0:
                ax.set_title(lab, fontsize=5)
            if j == 0:
                ax.set_ylabel(f"{m88.m83.m62_short(ag)}\nΔ rate (spk/s)", fontsize=4.6, color=gcol.get(ag, "k"))
            if i == len(EXAMPLES) - 1:
                ax.set_xlabel("from stimulus (ms)" if al == "stim" else "from first lick (ms)", fontsize=4.4)
    fig.text(0.06, 0.93, "a  Example areas (R+ green, R− magenta; day 0 dashed, expert solid)", fontsize=5.6)
    gf = gs[0, 1].subgridspec(1, 3, wspace=0.12)
    y = np.arange(len(areas))[::-1]
    for j, (c, lab, *_) in enumerate(CONTR):
        ax = fig.add_subplot(gf[j]); s = ST[ST.contrast == c].set_index("area")
        for i, ag in enumerate(areas):
            if ag not in s.index:
                continue
            r = s.loc[ag]
            for k, co in enumerate(("R+", "R-")):
                ax.errorbar(r[f"chg_{co}"], y[i] + 0.15 - 0.3 * k, xerr=r[f"sem_{co}"], fmt="o", ms=2.6, color=COH[co], lw=0.8, capsize=0,
                            mfc=COH[co] if r[f"p_{co}"] < 0.05 else "white")
            mk = " *" if r.q_int < 0.05 else (" (*)" if r.p_int < 0.05 else "")
            if mk:
                ax.text(0.99, y[i], mk.strip(), transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", ha="right", weight="bold")
        ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(-0.6, len(areas) - 0.4)
        ax.set_yticks(y); ax.set_yticklabels([m88.m83.m62_short(g) for g in areas] if j == 0 else [], fontsize=4.6)
        if j == 0:
            for tl, g in zip(ax.get_yticklabels(), areas):
                tl.set_color(gcol.get(g, "k")); tl.set_weight("bold" if g == "Whole brain" else "normal")
        ax.set_title(lab, fontsize=5); ax.set_xlabel("expert − day 0 (spk/s)", fontsize=4.6); ax.tick_params(labelsize=4.3)
        ax.axhline(len(areas) - 1.5, color="0.85", lw=0.5)
    fig.text(0.55, 0.93, "b  Change across learning per area; filled: within-cohort p < 0.05; * cohort × stage q < 0.05, (*) p < 0.05",
             fontsize=5.4)
    fig.suptitle(f"Within-session contrasts per area ({a.pop})", x=0.02, y=0.995, ha="left", fontsize=6.5, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"area_contrast_summary_{a.pop}.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    print(ST[["area", "contrast", "chg_R+", "p_R+", "chg_R-", "p_R-", "n_R+", "n_R-", "p_int", "q_int"]].round(3).to_string(index=False))
    print("ALL DONE", a.pop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
