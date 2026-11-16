"""179 -- Area-level link between brain-wide hit / miss decoding, the hit - miss coding direction and the passive responses
(user 2026-11-11: "correlate, at the level of areas, whether their decoding correlates with their change in passive projections,
not at the mouse level ... link brain-wide hit miss decoding with the hit miss coding direction and the passive. Does decoding
passive trials look more like hit or miss? That should confirm the CD analysis.").

Per session x area group (day-0 learners, sheet rule; stable tracked units):
  dec_0_50  hit / miss decoding 0-50 ms after the whisker stimulus, balanced accuracy minus the linear-shift null
            (ssl-whisker-hitmiss-timeresolved-decoding 024; via 178_points.csv)
  cd_dW     passive whisker post - pre change of the projection on the hit - miss coding direction (5-35 ms), beyond its
            linear-shift null (135 harmonised tracked; hm_W in 178); cd_dWA the whisker - auditory version
  ro_*      146 choice decoder (hit vs miss, active whisker trials, 5-35 ms, epochbase responses) read out on passive trials:
            anchored readout, +1 = like an active hit, -1 = like an active miss (held-out hit / miss means of the session);
            ro_pre_W / ro_post_W (passive whisker before / after the task), ro_dW = post - pre, same for auditory (A);
            ro_dW_excess = 146 shift_excess_dW (post - pre change beyond the linear-shift null)
Readout: 146 STANDARDISED readout ((score - hit/miss midpoint) / SD of held-out active whisker scores; > 0 hit side, < 0 miss
side; active hits / misses for reference); the anchored one (+-1 at the hit / miss means) explodes where decoding is weak.
Area level: per cohort, median over sessions of each measure per area group (areas with >= MIN_SESS sessions); Spearman rho across
areas (n = areas) for dec_0_50 vs cd_dW, dec_0_50 vs ro_dW, cd_dW vs ro_dW (does the decoder readout confirm the CD), and
dec_0_50 vs ro_pre_W / ro_post_W. Session level (support): cd_dW vs ro_dW within areas (ranks within area, pooled).
Hit-like or miss-like: per cohort, whole brain and per area: ro_pre_W, ro_post_W (Wilcoxon vs 0), post vs pre (Wilcoxon
signed-rank), R+ vs R- change (Mann-Whitney, Welch); auditory as the control stimulus.
Output: combined_results_ks4/ssl-hitmiss-axis-passive/tables/179_session_area.csv, 179_area_means.csv, 179_area_correlations.csv,
179_readout_tests.csv; figures/179_area_level.{png,pdf}, 179_passive_readout.{png,pdf}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/179_area_level_decoding_cd_readout.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
sys.path[:0] = [str(PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m61 = importlib.import_module("061_roc_prelick_learners")
RES = axel_bisi_root() / "combined_results_ks4" / "ssl-hitmiss-axis-passive"
TAB, FIG = RES / "tables", RES / "figures"
COH = {"R+": "#00B400", "R-": "#C800C8"}
MIN_SESS = 5
AREA = os.environ.get("SSL_179_AREA", "area_group")   # area_acronym_custom: fine labels (user 2026-11-11)
SFX = "" if AREA == "area_group" else "_fine"
PAIRS = [("dec_0_50", "cd_dW"), ("dec_0_50", "ro_dW"), ("cd_dW", "ro_dW"), ("dec_0_50", "ro_pre_W"), ("dec_0_50", "ro_post_W"),
         ("dec_0_50", "cd_dWA"), ("cd_dW", "ro_dW_excess"),
         # pre-stimulus (quiet window) hit / miss decoding (user 2026-11-11: attentional / state decision variable)
         ("dec_pre", "cd_dW"), ("dec_pre", "ro_dW"), ("dec_pre", "ro_post_W"), ("dec_pre", "cd_dWA"), ("dec_pre", "dec_0_50")]
LABEL = {"dec_pre": "hit/miss decoding pre-stim (-200 to -10 ms, - null)", "dec_0_50": "hit/miss decoding 0-50 ms (− shift null)", "cd_dW": "CD: passive W post − pre (− shift null)",
         "cd_dWA": "CD: passive W − A post − pre (− null)", "ro_dW": "decoder readout of passive W, post − pre",
         "ro_dW_excess": "decoder readout change (− shift null)", "ro_pre_W": "readout passive W, pre (> 0 hit side)",
         "ro_post_W": "readout passive W, post (> 0 hit side)"}


def area_colors():
    """Area-group palette and order (allen_utils); with fine labels (user 2026-11-11) each custom acronym takes the colour of
    its parent area group (the most common area_group among its units in the label table) and is ordered by group."""
    try:
        import ephys_utilities.allen_utils.allen_utils as au
        colors, order = au.get_custom_area_groups_colors(), list(au.get_area_group_custom_order())
    except Exception:  # noqa: BLE001
        colors, order = {}, []
    if AREA == "area_group":
        return colors, order
    L = pd.read_parquet(PROJECTS.parent / "reports" / "ssl_analysis" / "derived" / "unit_area_labels.parquet",
                        columns=[AREA, "area_group"]).dropna()
    parent = L.groupby(AREA).area_group.agg(lambda v: v.value_counts().index[0])
    rank = {g: i for i, g in enumerate(order)}
    fine_order = sorted(parent.index, key=lambda a: (rank.get(parent[a], len(rank)), a))
    return {a: colors.get(parent[a], "0.4") for a in parent.index}, fine_order


def load():
    learners, _ = m61.learner_mice()
    P = pd.read_csv(TAB / f"178_points{SFX}.csv")
    P = P[(P.stage == "learning") & P.mouse_id.isin(learners)]
    P = P.rename(columns={"hm_W": "cd_dW", "hm_WA": "cd_dWA"})[["session_id", "mouse_id", "cohort", "area", "dec_0_50", "dec_sens", "dec_pre",
                                                                "cd_dW", "cd_dWA"]]
    R = pd.read_parquet(TAB / f"146_choice_axis_readout{SFX}.parquet")
    R = R[R.skipped_reason.isna() & (R.unit_set == "stable") & (R.response == "epochbase") & R.mouse_id.isin(learners)].copy()
    # standardised readout (score - hit/miss midpoint) / SD of held-out active whisker scores: the anchored scale divides by the
    # hit - miss distance, which is tiny in areas that barely decode, and explodes there (values of +-20 seen 2026-11-11)
    R["ro_pre_W"], R["ro_post_W"] = R.ro_std_passive_pre_W, R.ro_std_passive_post_W
    R["ro_pre_A"], R["ro_post_A"] = R.ro_std_passive_pre_A, R.ro_std_passive_post_A
    R["ro_hit"] = R[["ro_std_active_1_hit", "ro_std_active_2_hit"]].mean(1)
    R["ro_miss"] = R[["ro_std_active_1_miss", "ro_std_active_2_miss"]].mean(1)
    R["ro_dW"], R["ro_dA"] = R.ro_post_W - R.ro_pre_W, R.ro_post_A - R.ro_pre_A
    R["ro_dW_excess"] = R.shift_excess_dW
    R = R.rename(columns={"reward_group": "cohort"})[["session_id", "mouse_id", "cohort", "area", "n_units", "decoder_bal_acc",
                                                      "ro_pre_W", "ro_post_W", "ro_dW", "ro_pre_A", "ro_post_A", "ro_dA",
                                                      "ro_dW_excess", "ro_hit", "ro_miss"]]
    return P.merge(R, on=["session_id", "mouse_id", "cohort", "area"], how="outer")


def wil(v):
    v = pd.Series(v).dropna()
    return stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan


def main():
    S = load()
    S.to_csv(TAB / f"179_session_area{SFX}.csv", index=False)
    colors, order = area_colors()
    meas = ["dec_pre", "dec_0_50", "cd_dW", "cd_dWA", "ro_dW", "ro_dW_excess", "ro_pre_W", "ro_post_W", "ro_dA", "decoder_bal_acc"]
    A = S[S.area != "whole_brain"].groupby(["cohort", "area"])[meas].agg(["median", "sem", "count"])
    A = A.rename(columns={"median": "mean"}, level=1)  # areas summarised by the median over sessions (robust); s.e.m. for bars
    A.columns = [f"{m}_{s}" for m, s in A.columns]
    A = A.reset_index()
    A.to_csv(TAB / f"179_area_means{SFX}.csv", index=False)

    rows = []
    for coh in ("R+", "R-"):
        a = A[A.cohort == coh]
        for x, y in PAIRS:
            ok = a[(a[f"{x}_count"] >= MIN_SESS) & (a[f"{y}_count"] >= MIN_SESS)]
            if len(ok) >= 4:
                r, p = stats.spearmanr(ok[f"{x}_mean"], ok[f"{y}_mean"])
                rp, pp = stats.pearsonr(ok[f"{x}_mean"], ok[f"{y}_mean"])
                rows.append(dict(cohort=coh, level="areas (means)", x=x, y=y, n=len(ok), rho=r, p=p, pearson_r=rp, pearson_p=pp,
                                 areas=";".join(ok.area)))
        # session level, within area (ranks within area, pooled): does the decoder readout confirm the CD?
        s = S[(S.cohort == coh) & (S.area != "whole_brain")].dropna(subset=["cd_dW", "ro_dW"])
        s = s[s.groupby("area").cd_dW.transform("size") >= MIN_SESS]
        if len(s) > 10:
            r, p = stats.spearmanr(s.groupby("area").cd_dW.rank(pct=True), s.groupby("area").ro_dW.rank(pct=True))
            rows.append(dict(cohort=coh, level="sessions, within-area ranks", x="cd_dW", y="ro_dW", n=len(s), rho=r, p=p))
        w = S[(S.cohort == coh) & (S.area == "whole_brain")].dropna(subset=["cd_dW", "ro_dW"])
        if len(w) >= 5:
            r, p = stats.spearmanr(w.cd_dW, w.ro_dW)
            rows.append(dict(cohort=coh, level="sessions, whole brain", x="cd_dW", y="ro_dW", n=len(w), rho=r, p=p))
    C = pd.DataFrame(rows)
    C.to_csv(TAB / f"179_area_correlations{SFX}.csv", index=False)

    # hit-like or miss-like: passive readouts
    T = []
    for area, g in S.groupby("area"):
        for coh in ("R+", "R-"):
            d = g[g.cohort == coh]
            if d.ro_pre_W.notna().sum() < MIN_SESS:
                continue
            T.append(dict(area=area, cohort=coh, n=int(d.ro_pre_W.notna().sum()),
                          pre_W=d.ro_pre_W.mean(), post_W=d.ro_post_W.mean(), p_pre_W=wil(d.ro_pre_W), p_post_W=wil(d.ro_post_W),
                          dW=d.ro_dW.mean(), p_dW=wil(d.ro_dW), dW_excess=d.ro_dW_excess.mean(), p_dW_excess=wil(d.ro_dW_excess),
                          pre_A=d.ro_pre_A.mean(), post_A=d.ro_post_A.mean(), dA=d.ro_dA.mean(), p_dA=wil(d.ro_dA),
                          hit=d.ro_hit.mean(), miss=d.ro_miss.mean()))
        a, b = g[g.cohort == "R+"].ro_dW.dropna(), g[g.cohort == "R-"].ro_dW.dropna()
        if len(a) >= 3 and len(b) >= 3:
            T.append(dict(area=area, cohort="R+ vs R-", n=len(a) + len(b), dW=a.mean() - b.mean(),
                          p_dW=stats.mannwhitneyu(a, b).pvalue, p_dW_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    T = pd.DataFrame(T)
    T.to_csv(TAB / f"179_readout_tests{SFX}.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
    print(C.drop(columns=["areas"]).round(3).to_string())
    print(T.round(3).to_string())

    # ---------------- figure 1: area-level scatter
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    for j, coh in enumerate(("R+", "R-")):
        a = A[A.cohort == coh]
        for i, (x, y) in enumerate([("dec_0_50", "cd_dW"), ("dec_0_50", "ro_dW"), ("cd_dW", "ro_dW"), ("dec_pre", "cd_dW"),
                                    ("dec_pre", "ro_dW")]):
            ax = axes[i, j]
            ok = a[(a[f"{x}_count"] >= MIN_SESS) & (a[f"{y}_count"] >= MIN_SESS)]
            for r in ok.itertuples():
                c = colors.get(r.area, "0.4")
                ax.errorbar(getattr(r, f"{x}_mean"), getattr(r, f"{y}_mean"), xerr=getattr(r, f"{x}_sem"), yerr=getattr(r, f"{y}_sem"),
                            fmt="o", color=c, ms=4 + getattr(r, f"{y}_count") / 4, elinewidth=0.6, alpha=0.9)
                ax.annotate(r.area.replace("Somatosensory", "SS"), (getattr(r, f"{x}_mean"), getattr(r, f"{y}_mean")), fontsize=5.5,
                            xytext=(3, 3), textcoords="offset points", color=c)
            wb = S[(S.cohort == coh) & (S.area == "whole_brain")]
            if wb[x].notna().any() and wb[y].notna().any():
                ax.plot(wb[x].mean(), wb[y].mean(), marker="*", color="k", ms=10, ls="none", label="whole brain")
            r = C[(C.cohort == coh) & (C.level == "areas (means)") & (C.x == x) & (C.y == y)]
            if len(r):
                ax.set_title(f"{coh}: ρ = {r.rho.iloc[0]:.2f}, p = {r.p.iloc[0]:.3f} (n = {int(r.n.iloc[0])} areas)", fontsize=8,
                             color=COH[coh])
            ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axvline(0, color="0.7", lw=0.5, ls=":")
            ax.set_xlabel(LABEL[x], fontsize=7); ax.set_ylabel(LABEL[y], fontsize=7); ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Area level (day-0 learners): median ± s.e.m. over sessions per area (marker size ~ sessions; colour = area group)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"179_area_level{SFX}.{ext}", dpi=170)

    # ---------------- figure 2: passive readouts, hit-like or miss-like
    have = set(T[T.cohort.isin(["R+", "R-"])].area)  # areas with per-cohort values (not only the R+ vs R- row)
    areas = ["whole_brain"] + [a for a in (order or sorted(S.area.unique())) if a in have and a != "whole_brain"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.35 * len(areas) + 1.5), sharey=True)
    for ax, stim in zip(axes, ("W", "A")):
        for k, coh in enumerate(("R+", "R-")):
            t = T[T.cohort == coh].set_index("area").reindex(areas)
            yy = np.arange(len(areas)) + (k - 0.5) * 0.3
            ax.scatter(t[f"pre_{stim}"], yy, facecolor="white", edgecolor=COH[coh], s=18, label=f"{coh} pre", zorder=3)
            ax.scatter(t[f"post_{stim}"], yy, color=COH[coh], s=18, label=f"{coh} post", zorder=3)
            for y0, a0, b0 in zip(yy, t[f"pre_{stim}"], t[f"post_{stim}"]):
                ax.plot([a0, b0], [y0, y0], color=COH[coh], lw=0.6)
            for y0, p, b0 in zip(yy, t[f"p_d{stim}"], t[f"post_{stim}"]):
                if np.isfinite(p) and p < 0.05:
                    ax.text(b0, y0 + 0.12, "*" if p >= 0.01 else "**" if p >= 0.001 else "***", fontsize=6, color=COH[coh], ha="center")
        ax.axvline(0, color="0.6", lw=0.5)
        for k, coh in enumerate(("R+", "R-")):  # active hit / miss readouts for reference (small ticks)
            t = T[T.cohort == coh].set_index("area").reindex(areas)
            yy = np.arange(len(areas)) + (k - 0.5) * 0.3
            ax.scatter(t.hit, yy, marker="|", color=COH[coh], s=30, alpha=0.5)
            ax.scatter(t.miss, yy, marker="|", color=COH[coh], s=30, alpha=0.5)
        ax.set_yticks(range(len(areas)), areas, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel(f"choice-decoder readout of passive {'whisker' if stim == 'W' else 'auditory'} trials\n(standardised; > 0 hit side, < 0 miss side; ticks = active hits / misses)",
                      fontsize=7)
        ax.set_title(f"Passive {'whisker' if stim == 'W' else 'auditory'}: pre (open) → post (filled); * post vs pre", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"179_passive_readout{SFX}.{ext}", dpi=170)
    print(FIG / "179_area_level.png")


if __name__ == "__main__":
    main()
