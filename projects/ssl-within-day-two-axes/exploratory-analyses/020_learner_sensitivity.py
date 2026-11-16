"""020 -- Sensitivity of the two-axis day-0 results to the learner labels (user 2026-11-11: learning_category is not principled;
"see if the effects are stronger or cleaner with different good/moderate vs bad splits").

R+ 'bad' mice split in two (user): kept bad whatever the variant, MH020, MH013, AB142, MH068, AB151, AB117, AB107; flexible, the other
R+ bad mice (AB141, MH007, AB104, MH026, MH009), moved into the learners in the variants below. R- labels unchanged.
Variants (day-0 population = sheet learners + added mice; expert sessions always all):
  sheet            sheet labels (as 003)
  +<mouse>         one flexible mouse added
  all-but-<mouse>  every flexible mouse added except one
  +all flexible    all five added
  all mice         no learner filter on day 0 (reference)
Measures and statistics as 003 (common.measure_stats; session = unit): reward-lick CD whisker-hit drift (rl_WH_raw, rl_WH, rl_WHSL) and
hit - miss axis passive change (hm_W, hm_A, hm_WA); whole brain, and per area group (BH over areas of the day-0 p within measure x cohort,
areas with >= 3 sessions per cohort at both stages).
Output: tables/020_wholebrain.csv, 020_areas.csv, 020_flexible_sessions.csv; figures/020_sensitivity.{png,pdf}
"""
from __future__ import annotations

import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

m003 = importlib.import_module("003_classification")
FIXED_BAD = ["MH020", "MH013", "AB142", "MH068", "AB151", "AB117", "AB107"]
FLEX = ["AB141", "MH007", "AB104", "MH026", "MH009"]
SHOW = [("rl_WH", "Reward-lick CD: WH drift beyond null"), ("rl_WHSL", "Reward-lick CD: WH − SL drift beyond null"),
        ("hm_W", "Hit − miss axis: passive W post − pre"), ("hm_WA", "Hit − miss axis: W − A post − pre")]


def load_all():
    """003.load without the learner filter (every day-0 session)."""
    rl = pd.read_csv(C.m51.WITHIN / "cd_shift_null" / "sessions_areas.csv")
    rl = rl.rename(columns={"beta_pu": "rl_WH_raw", "excess_pu": "rl_WH", "excessC_pu": "rl_WHSL"})
    rl = rl[["session_id", "mouse_id", "cohort", "stage", "area", "rl_WH_raw", "rl_WH", "rl_WHSL"]]
    H = []
    for st_, sfx in (("learning", ""), ("expert", "_expert")):
        A = pd.read_parquet(C.HM / f"135_alignment_epochs_min2_tracked_harm{sfx}.parquet")
        A = A[A.skipped_reason.isna() & A.area.notna()].copy(); A["stage"] = st_; H.append(A)
    H = pd.concat(H).rename(columns={"reward_group": "cohort", "shift_excess_dWP": "hm_W", "shift_excess_dAP": "hm_A",
                                     "shift_excess_dWAP": "hm_WA"})
    H["area"] = H.area.replace({"All units": "whole_brain"})
    H = H[["session_id", "mouse_id", "cohort", "stage", "area", "hm_W", "hm_A", "hm_WA"]]
    return rl, H


def variants(sheet_learners):
    out = {"sheet": set(sheet_learners)}
    for m in FLEX:
        out[f"+{m}"] = set(sheet_learners) | {m}
    for m in FLEX:
        out[f"all-but-{m}"] = set(sheet_learners) | (set(FLEX) - {m})
    out["+all flexible"] = set(sheet_learners) | set(FLEX)
    out["all mice"] = None
    return out


def main(n_perm):
    rl, H = load_all()
    sheet, _ = C.m61.learner_mice()
    rows = []
    for name, L in variants(sheet).items():
        r_, h_ = (rl, H) if L is None else (C.m61.learner_filter(rl, L), C.m61.learner_filter(H, L))
        R = m003.run(r_, h_, n_perm)
        R["variant"] = name
        R["added"] = "" if L is None or name == "sheet" else ",".join(sorted(L - set(sheet)))
        rows.append(R)
        print(name, "done", flush=True)
    R = pd.concat(rows, ignore_index=True)
    wb = R[R.level == "whole_brain"]
    wb.to_csv(C.TAB / "020_wholebrain.csv", index=False)
    R[R.level != "whole_brain"].to_csv(C.TAB / "020_areas.csv", index=False)

    # where the flexible mice sit, per session, against the sheet learners (whole brain, day 0)
    sess = []
    for D, ms in ((rl, ["rl_WH_raw", "rl_WH", "rl_WHSL"]), (H, ["hm_W", "hm_A", "hm_WA"])):
        d = D[(D.area == "whole_brain") & (D.stage == "learning")]
        for m in ms:
            for r in d.dropna(subset=[m]).itertuples():
                grp = ("learner" if r.mouse_id in sheet else "flexible" if r.mouse_id in FLEX
                       else "fixed bad" if r.mouse_id in FIXED_BAD else "other bad")
                sess.append(dict(measure=m, session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort, group=grp,
                                 value=getattr(r, m)))
    S = pd.DataFrame(sess)
    S.to_csv(C.TAB / "020_flexible_sessions.csv", index=False)

    # summary print: whole brain R+ day 0 and R+ vs R- day 0, per variant
    pd.set_option("display.width", 250)
    for m, _ in SHOW + [("rl_WH_raw", ""), ("hm_A", "")]:
        a = wb[(wb.measure == m) & (wb.cohort == "R+")].set_index("variant")[["n_d0", "mean_d0", "sem_d0", "p_d0", "p_stage_mwu", "cls"]]
        b = wb[(wb.measure == m) & (wb.cohort == "R+ vs R-")].set_index("variant")[["p_d0", "p_d0_welch", "p_interaction", "cls"]]
        b.columns = ["coh_p_mwu", "coh_p_welch", "p_interaction", "coh_cls"]
        print(f"===== {m}\n", a.join(b).round(4).to_string())
    ar = R[(R.level != "whole_brain") & (R.cohort == "R+")]
    n_sig = ar.assign(sig=(ar.p_d0 < 0.05), sigq=(ar.q_d0 < 0.05)).groupby(["measure", "variant"])[["sig", "sigq"]].sum().unstack(0)
    print("===== R+ day-0 areas with p < 0.05 (sig) and BH q < 0.05 (sigq)\n", n_sig.to_string())

    # figure: per measure, left R+ day-0 mean +- s.e.m. per variant (stars p_d0, text R+ vs R- p); right session values by group
    order = list(variants(sheet))
    fig, axes = plt.subplots(len(SHOW), 2, figsize=(12, 3.0 * len(SHOW)), gridspec_kw=dict(width_ratios=[3, 1.3]))
    for i, (m, title) in enumerate(SHOW):
        a = axes[i, 0]
        d = wb[(wb.measure == m) & (wb.cohort == "R+")].set_index("variant").loc[order]
        coh = wb[(wb.measure == m) & (wb.cohort == "R+ vs R-")].set_index("variant").loc[order]
        rm = wb[(wb.measure == m) & (wb.cohort == "R-") & (wb.variant == "sheet")].iloc[0]
        x = np.arange(len(order))
        a.errorbar(x, d.mean_d0, d.sem_d0, fmt="o", color=C.COH["R+"], ms=4, capsize=0)
        a.axhspan(rm.mean_d0 - rm.sem_d0, rm.mean_d0 + rm.sem_d0, color=C.COH["R-"], alpha=0.15, lw=0)
        a.axhline(rm.mean_d0, color=C.COH["R-"], lw=0.8)
        a.axhline(0, color="0.6", lw=0.5, ls=":")
        top = (d.mean_d0 + d.sem_d0).max()
        for xi, v in zip(x, order):
            a.text(xi, top * 1.05 if top > 0 else 0, f"n={int(d.loc[v, 'n_d0'])}\np={d.loc[v, 'p_d0']:.3f}\nR+vR- {coh.loc[v, 'p_d0']:.3f}",
                   ha="center", va="bottom", fontsize=5.5)
        a.set_xticks(x, order, rotation=40, ha="right", fontsize=7)
        a.set_title(f"{title}: R+ day 0 per learner set (R- day 0 sheet, magenta band)", fontsize=8)
        a.set_ylim(top=top * 1.6 if top > 0 else None)
        b = axes[i, 1]
        s = S[(S.measure == m) & (S.cohort == "R+")]
        rng = np.random.default_rng(0)
        for k, grp in enumerate(["learner", "flexible", "fixed bad"]):
            v = s[s.group == grp]
            b.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v.value, s=10, color=C.COH["R+"] if grp == "learner" else "white",
                      edgecolor=C.COH["R+"] if grp != "fixed bad" else "0.4", lw=0.7)
            if grp == "flexible":
                for r in v.itertuples():
                    b.annotate(r.mouse_id, (k, r.value), fontsize=5.5, xytext=(5, 0), textcoords="offset points", va="center")
        b.axhline(0, color="0.6", lw=0.5, ls=":")
        b.set_xticks(range(3), ["sheet learners", "flexible", "fixed bad"], fontsize=7)
        b.set_title("R+ day-0 sessions", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG / f"020_sensitivity.{ext}", dpi=200)
    print(C.FIG / "020_sensitivity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    main(ap.parse_args().n_perm)
