"""024 -- Drift quantifications in the compact COSYNE format (user 2026-10-07: "the drift plots do not show sessions, only
the mean as a dot and the s.e.m."): one figure with the whole-brain drifts and one panel pair per area group.

Drifts D_SL = beta_WH - beta_SL and D_AH = beta_WH - beta_AH per session (Eq. 17): whole brain from 011
(cd_trajectories/<pop>/sessions.csv), per area group from 015 (area_sessions.csv; area groups with >= 3 sessions in every
cohort x stage condition). Each panel: R+ and R- mean +- s.e.m. over sessions, day 0 (open) and expert (filled), cohorts
side by side within each stage; above each stage: R+ vs R- mouse-level permutation p; panel title: cohort x stage
interaction (mouse-level permutation). n sessions under the points.
Output: within_day/<ref>/cosyne/<pop>/drift_compact.{png,pdf,svg} + drift_compact_stats.csv.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
m015 = importlib.import_module("015_area_cd_drift")
m017 = importlib.import_module("017_area_cd_drift_views")
COH = m62.COH
POS = {("R+", "learning"): 0.0, ("R-", "learning"): 0.45, ("R+", "expert"): 1.25, ("R-", "expert"): 1.7}


def panel(ax, S, col, title, rows, unit):
    S = S.dropna(subset=[col])
    hi = -np.inf
    for (c, st), x in POS.items():
        v = S[(S.cohort == c) & (S.stage == st)][col]
        if len(v) < 2:
            continue
        m, se = v.mean(), v.sem()
        ax.errorbar(x, m, se, fmt="o", ms=4.2, color=COH[c], lw=1.2, capsize=2, capthick=1.0,
                    mfc="white" if st == "learning" else COH[c])
        hi = max(hi, m + se)
        rows.append(dict(unit=unit, measure=col, cohort=c, stage=st, n=len(v), mean=m, sem=se))
    r = dict(unit=unit, measure=col)
    for st, x0 in (("learning", 0.0), ("expert", 1.25)):
        d = S[S.stage == st]
        if min((d.cohort == c).sum() for c in ("R+", "R-")) >= 3:
            _, p = m001.perm_interaction(d.assign(stage="learning"), col, kind="learning")
            r[f"{st} p_perm"] = p
            ax.plot([x0, x0 + 0.45], [hi * 1.12 + 0.05] * 2, color="0.3", lw=0.6)
            ax.text(x0 + 0.225, hi * 1.12 + 0.07, m62.fmt_p(p).replace("p = ", "").replace("p ", ""), ha="center",
                    va="bottom", fontsize=4.6)
    _, r["cxs p_perm"] = m001.perm_interaction(S, col, kind="stage")
    rows.append(r)
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0.225, 1.475], ["Day 0", "Expert"]); ax.set_xlim(-0.35, 2.05)
    ax.set_title(f"{title}\ncohort × stage {m62.fmt_p(r['cxs p_perm'])}", fontsize=5.2)
    ax.margins(y=0.25)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup()
    gcol = au.get_custom_area_groups_colors()
    A0 = pd.read_csv(m015.OUT / "area_sessions.csv")
    for pop in ("learners", "all"):
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        S = pd.read_csv(m51.WITHIN / "cd_trajectories" / pop / "sessions.csv")
        A = flt(A0)
        areas = m017.order_of(m017.strict_areas(A, "drift_WH_SL") & m017.strict_areas(A, "drift_WH_AH"))
        units = [("Whole brain", S)] + [(g, A[A.area_group == g]) for g in areas]
        ncol = 6
        nrow = int(np.ceil(len(units) * 2 / ncol))
        H = 1.32 * nrow + 0.75
        fig = plt.figure(figsize=(m62.W_IN, H))
        gs = fig.add_gridspec(nrow, ncol, hspace=0.75, wspace=0.6, left=0.06, right=0.99, top=1 - 0.7 / H, bottom=0.25 / H)
        rows = []
        for i, (u, d) in enumerate(units):
            for j, (col, ref) in enumerate((("drift_WH_SL", "SL"), ("drift_WH_AH", "AH"))):
                k = 2 * i + j
                ax = fig.add_subplot(gs[k // ncol, k % ncol]); ax.set_box_aspect(1)
                short = m017.SHORT(u) if u != "Whole brain" else u
                panel(ax, d, col, f"{short}: β_WH − β_{ref}", rows, u)
                ax.title.set_color(gcol.get(u, "k") if u != "Whole brain" else "k")
                if k % ncol == 0:
                    ax.set_ylabel("Drift (change over one session)", fontsize=5)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="", color=COH[c], mfc=("white" if st == "learning" else COH[c]), ms=4,
                    label=f"{c.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'}") for c in ("R+", "R-")
             for st in ("learning", "expert")]
        fig.legend(handles=h, loc="upper left", ncol=4, frameon=False, fontsize=5, bbox_to_anchor=(0.02, 1 - 0.22 / H))
        fig.suptitle(f"Within-session drift, mean ± s.e.m. over sessions ({pop}); p above: R+ vs R− within stage "
                     "(mouse-level permutation)", x=0.02, y=0.998, ha="left", va="top", fontsize=6.4, weight="bold")
        out = m51.WITHIN / "cosyne" / pop
        out.mkdir(parents=True, exist_ok=True)
        m62.save(fig, out, "drift_compact"); plt.close(fig)
        pd.DataFrame(rows).to_csv(out / "drift_compact_stats.csv", index=False)
        print(pop, len(units), "units")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
