"""178 -- Does early hit / miss decodability of an area relate to how much the passive whisker response changes along that area's
hit - miss axis? (user 2026-11-11: "Does area specific hit/miss decoding (e.g. in the first 50 ms) from my previous project correlate
with how much it changes from passive post minus pre (CD hit miss projections)?")

Per session x area group (and whole brain), same stage:
  decoding  ssl-whisker-hitmiss-timeresolved-decoding 024 (hitmiss_stim, condition 'whole'): balanced accuracy of the time-resolved
            hit vs miss decoder minus the mean of its linear-shift null curves, in
              dec_0_50   the 50-ms window [0, 50] ms after the whisker stimulus (one window, fully inside the first 50 ms),
              dec_sens   the project's sensory value (034 SENSORY_WINDOW: mean over windows whose right edge is in [5, 50] ms);
              dec_pre    pre-stimulus (quiet window) value (034 BASELINE_WINDOW: windows whose right edge is in [-200, -10] ms,
                         i.e. entirely before the stimulus; user 2026-11-11);
  SSL_178_AREA=area_acronym_custom: fine custom acronyms (024 *_area_acronym_custom, 135 *_fine); outputs *_fine (user 2026-11-11).
  passive   135 harmonised tracked (this project): change of the passive whisker projection on the session's hit - miss axis
            (5-35 ms), post - pre, beyond its linear-shift null (hm_W = shift_excess_dWP), whisker - auditory (hm_WA =
            shift_excess_dWAP), and the raw change (hm_W_raw = shift_real_dWP).
Population: day 0 learners (sheet rule, learner_filter of ssl-prelick-convergence 061) and all day-0 sessions; expert sessions as a
second stage. Statistics per cohort and stage (session = unit within an area):
  whole brain   Spearman rho across sessions;
  areas         Spearman rho per area group (areas with >= MIN_SESS sessions), BH over areas; pooled within-area relation:
                both variables ranked within area (percentile ranks), Spearman over all session x area points, p from 2,000
                permutations of the decoding values within area (keeps area structure; sessions contribute several areas);
                mean of per-area Fisher z, Wilcoxon over areas.
Output: combined_results_ks4/ssl-hitmiss-axis-passive/tables/178_points.csv, 178_correlations.csv; figures/178_decoding_vs_passive.{png,pdf}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/178_decoding_vs_passive_change.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m61 = importlib.import_module("061_roc_prelick_learners")
RES = axel_bisi_root() / "combined_results_ks4"
DEC = RES / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables"
HM = RES / "ssl-hitmiss-axis-passive" / "tables"
FIG = RES / "ssl-hitmiss-axis-passive" / "figures"
COH = {"R+": "#00B400", "R-": "#C800C8"}
MIN_SESS, N_PERM = 6, 2000
AREA = os.environ.get("SSL_178_AREA", "area_group")
SFX = "" if AREA == "area_group" else "_fine"
DEC_MEAS = {"dec_0_50": "Hit/miss decoding 0-50 ms\n(bal. acc. − shift null)", "dec_sens": "Hit/miss decoding, sensory window\n(− shift null)",
            "dec_pre": "Hit/miss decoding, pre-stimulus\n(−200 to −10 ms, − shift null)"}
HM_MEAS = {"hm_W": "Passive W post − pre on hit−miss axis\n(− shift null)", "hm_WA": "Passive W − A post − pre\n(− shift null)",
           "hm_W_raw": "Passive W post − pre (raw)"}


def decoding(stage):
    sfx = "" if stage == "learning" else "_expert"
    rows = []
    for tag, level in ((f"hitmiss_stim{sfx}_{AREA}", "area"), (f"hitmiss_stim{sfx}_whole_brain", "whole_brain")):
        if not (DEC / f"024_master_results_{tag}.parquet").exists():
            continue
        d = pd.read_parquet(DEC / f"024_master_results_{tag}.parquet")
        d = d[d.skipped_reason.isna() & (d.condition_value == "whole")]
        edges = np.array(json.load(open(DEC / f"024_bin_edges_{tag}.json")))
        right = edges[:, 1]
        i050 = int(np.argmin(np.abs(edges[:, 0]) + np.abs(edges[:, 1] - 0.05)))  # window [0, 0.05]
        sens = (right >= 0.005 - 1e-9) & (right <= 0.05 + 1e-9)                     # 034 SENSORY_WINDOW convention
        pre = (right >= -0.2 - 1e-9) & (right <= -0.01 + 1e-9)                      # 034 BASELINE_WINDOW convention
        for r in d.itertuples():
            real = np.asarray(r.real_curve, float)
            null = np.nanmean(np.vstack([np.asarray(c, float) for c in r.shift_null_curves]), 0)
            rows.append(dict(session_id=r.session_id, mouse_id=r.subject_id, cohort=r.reward_group,
                             area=r.area_value if level == "area" else "whole_brain", stage=stage, n_units_dec=r.n_units,
                             dec_0_50=real[i050] - null[i050], dec_sens=np.nanmean(real[sens] - null[sens]),
                             dec_pre=np.nanmean(real[pre] - null[pre]),
                             acc_0_50=real[i050]))
    return pd.DataFrame(rows)


def passive(stage):
    sfx = "" if stage == "learning" else "_expert"
    f = HM / f"135_alignment_epochs_min2_tracked_harm{SFX}{sfx}.parquet"
    if not f.exists():
        return pd.DataFrame(columns=["session_id", "area", "hm_W", "hm_WA", "hm_W_raw", "n_units_hm"])
    A = pd.read_parquet(f)
    A = A[A.skipped_reason.isna() & A.area.notna()].copy()
    A["area"] = A.area.replace({"All units": "whole_brain"})
    return A.rename(columns={"shift_excess_dWP": "hm_W", "shift_excess_dWAP": "hm_WA", "shift_real_dWP": "hm_W_raw",
                             "n_units": "n_units_hm"})[["session_id", "area", "hm_W", "hm_WA", "hm_W_raw", "n_units_hm"]]


def within_area_pooled(d, x, y, rng):
    d = d.dropna(subset=[x, y])
    d = d[d.groupby("area")[x].transform("size") >= MIN_SESS]
    if d.area.nunique() < 2:
        return np.nan, np.nan, 0
    rx = d.groupby("area")[x].rank(pct=True).to_numpy()
    ry = d.groupby("area")[y].rank(pct=True).to_numpy()
    obs = stats.spearmanr(rx, ry)[0]
    groups = [np.flatnonzero((d.area == a).to_numpy()) for a in d.area.unique()]
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        p = rx.copy()
        for g in groups:
            p[g] = rx[rng.permutation(g)]
        null[k] = stats.spearmanr(p, ry)[0]
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)), len(d)


def main():
    learners, _ = m61.learner_mice()
    P = []
    for stage in ("learning", "expert"):
        P.append(decoding(stage).merge(passive(stage), on=["session_id", "area"], how="inner"))
    P = pd.concat(P, ignore_index=True)
    P["learner"] = P.mouse_id.isin(learners)
    P.to_csv(HM / f"178_points{SFX}.csv", index=False)
    rng = np.random.default_rng(0)
    rows = []
    pops = {("learning", "learners"): P[(P.stage == "learning") & P.learner], ("learning", "all"): P[P.stage == "learning"],
            ("expert", "all"): P[P.stage == "expert"]}
    for (stage, pop), D in pops.items():
        for coh in ("R+", "R-"):
            d = D[D.cohort == coh]
            for x in DEC_MEAS:
                for y in HM_MEAS:
                    base = dict(stage=stage, pop=pop, cohort=coh, decoding=x, passive=y)
                    wb = d[d.area == "whole_brain"].dropna(subset=[x, y])
                    if len(wb) >= 5:
                        r, p = stats.spearmanr(wb[x], wb[y])
                        rows.append(dict(base, level="whole_brain", n=len(wb), rho=r, p=p))
                    ar = d[d.area != "whole_brain"]
                    per = []
                    for a, g in ar.dropna(subset=[x, y]).groupby("area"):
                        if len(g) >= MIN_SESS:
                            r, p = stats.spearmanr(g[x], g[y])
                            per.append(dict(base, level=a, n=len(g), rho=r, p=p))
                    if per:
                        q = multipletests([r["p"] for r in per], method="fdr_bh")[1]
                        for r, qq in zip(per, q):
                            r["q_bh"] = qq
                        rows += per
                        z = np.arctanh(np.clip([r["rho"] for r in per], -0.999, 0.999))
                        wz = stats.wilcoxon(z).pvalue if len(z) >= 5 else np.nan
                        rows.append(dict(base, level="areas: mean Fisher z", n=len(z), rho=float(np.tanh(z.mean())), p=wz))
                    obs, pp, n = within_area_pooled(ar, x, y, rng)
                    rows.append(dict(base, level="areas: pooled within-area ranks", n=n, rho=obs, p=pp))
    R = pd.DataFrame(rows)
    R.to_csv(HM / f"178_correlations{SFX}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 400)
    show = R[R.level.isin(["whole_brain", "areas: mean Fisher z", "areas: pooled within-area ranks"])]
    print(show.pivot_table(index=["stage", "pop", "cohort", "decoding", "passive"], columns="level", values=["rho", "p", "n"])
          .round(3).to_string())
    sig = R[(~R.level.str.startswith(("whole", "areas:"))) & (R.p < 0.05)]
    print("\nper-area p < 0.05:\n", sig[["stage", "pop", "cohort", "decoding", "passive", "level", "n", "rho", "p", "q_bh"]].round(3).to_string())

    # figure: day-0 learners, dec_0_50; rows = passive measure (hm_W, hm_WA); columns = whole brain | pooled within-area | per area
    D = pops[("learning", "learners")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
    x = "dec_0_50"
    for i, y in enumerate(["hm_W", "hm_WA"]):
        a = axes[i, 0]
        for coh in ("R+", "R-"):
            g = D[(D.cohort == coh) & (D.area == "whole_brain")].dropna(subset=[x, y])
            a.scatter(g[x], g[y], s=14, color=COH[coh], label=coh)
            r = R[(R.stage == "learning") & (R["pop"] == "learners") & (R.cohort == coh) & (R.decoding == x) & (R.passive == y)
                  & (R.level == "whole_brain")]
            if len(r):
                a.text(0.02, 0.98 - 0.08 * (coh == "R-"), f"{coh}: ρ={r.rho.iloc[0]:.2f}, p={r.p.iloc[0]:.3f}, n={int(r.n.iloc[0])}",
                       transform=a.transAxes, va="top", fontsize=7, color=COH[coh])
        a.axhline(0, color="0.7", lw=0.5, ls=":"); a.axvline(0, color="0.7", lw=0.5, ls=":")
        a.set_title("Whole brain", fontsize=9); a.set_xlabel(DEC_MEAS[x], fontsize=7); a.set_ylabel(HM_MEAS[y], fontsize=7)
        a = axes[i, 1]
        for coh in ("R+", "R-"):
            g = D[(D.cohort == coh) & (D.area != "whole_brain")].dropna(subset=[x, y])
            g = g[g.groupby("area")[x].transform("size") >= MIN_SESS]
            a.scatter(g.groupby("area")[x].rank(pct=True), g.groupby("area")[y].rank(pct=True), s=6, alpha=0.6, color=COH[coh])
            r = R[(R.stage == "learning") & (R["pop"] == "learners") & (R.cohort == coh) & (R.decoding == x) & (R.passive == y)
                  & (R.level == "areas: pooled within-area ranks")]
            if len(r):
                a.text(0.02, 0.98 - 0.08 * (coh == "R-"), f"{coh}: ρ={r.rho.iloc[0]:.2f}, perm p={r.p.iloc[0]:.3f}, n={int(r.n.iloc[0])}",
                       transform=a.transAxes, va="top", fontsize=7, color=COH[coh])
        a.set_title("Areas, ranked within area", fontsize=9); a.set_xlabel("decoding (percentile within area)", fontsize=7)
        a.set_ylabel("passive change (percentile within area)", fontsize=7)
        a = axes[i, 2]
        per = R[(R.stage == "learning") & (R["pop"] == "learners") & (R.decoding == x) & (R.passive == y)
                & ~R.level.str.startswith(("whole", "areas:"))]
        areas = sorted(per.level.unique())
        for k, coh in enumerate(("R+", "R-")):
            g = per[per.cohort == coh].set_index("level").reindex(areas)
            yy = np.arange(len(areas)) + (k - 0.5) * 0.35
            a.barh(yy, g.rho, height=0.33, color=COH[coh])
            for yi, (rr, pv, nn) in zip(yy, zip(g.rho, g.p, g.n)):
                if np.isfinite(pv):
                    a.text(rr + (0.02 if rr >= 0 else -0.02), yi, f"{'*' if pv < 0.05 else ''}n={int(nn)}", va="center",
                           ha="left" if rr >= 0 else "right", fontsize=5)
        a.set_yticks(range(len(areas)), areas, fontsize=6); a.axvline(0, color="0.5", lw=0.5)
        a.set_xlabel("Spearman ρ per area (sessions)", fontsize=7); a.set_title("Per area group (* p < 0.05, uncorrected)", fontsize=9)
    fig.suptitle("Day-0 learners: early hit/miss decoding (0-50 ms) vs passive post − pre change on the hit−miss axis", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"178_decoding_vs_passive{SFX}.{ext}", dpi=170)
    print(FIG / "178_decoding_vs_passive.png")


if __name__ == "__main__":
    main()
