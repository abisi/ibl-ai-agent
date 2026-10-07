"""Where are the modality-generalizing ("transfer") neurons?

Transfer neuron: significant pre-lick ROC for auditory hit vs false alarm AND for whisker hit vs false alarm, with the
same sign (051, all trials; tested units = both ROCs tested: mean raw pre-lick rate >= 0.1 Hz, >= 3 trials per class).
Units: quality good or mua. Unit of analysis: session.
Outputs -> combined_results_ks4/ssl-prelick-convergence/across_days/fa/generalizing_units/
  areas.png / area_group_stats.csv / fine_area_stats.csv: fraction of transfer neurons among tested units per
     session x area (>= MIN_UNITS tested units), mean +- s.e.m. over sessions per cohort x stage; Mann-Whitney U for
     R+ learning vs expert and expert R+ vs R- (Welch in csv); areas need >= 3 sessions per group compared.
  ccf/frac_sig/transfer_units.png: CCF density maps (049 make: Gaussian-smoothed fraction of transfer neurons among
     tested units, 6 coronal + 4 sagittal slabs, white background, grey Allen contours, cohort single-hue maps, expert -
     learning diverging maps).
  ccf_units/transfer_units.png: every tested unit at its CCF position, transfer neurons coloured by WH vs FA selectivity.
"""
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
OUT = m51.OUTROOT / "generalizing_units"
UNIT_SET = ("good", "mua")
MIN_UNITS = 5
MIN_SESS = 3
AF, WF = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"


def unit_table():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(UNIT_SET)].reset_index(drop=True)
    tested = W[f"sig:{AF}"].notna() & W[f"sig:{WF}"].notna()
    saf, swf = np.sign(W[f"sel:{AF}"]), np.sign(W[f"sel:{WF}"])
    W["tested"] = tested
    W["transfer"] = tested & (W[f"sig:{AF}"] == 1) & (W[f"sig:{WF}"] == 1) & (saf == swf)
    W["transfer_pos"] = W.transfer & (saf > 0)
    return W


def area_stats(W, level):
    T = W[W.tested].groupby(["session_id", "mouse_id", "cohort", "stage", level]).agg(
        n=("transfer", "size"), k=("transfer", "sum")).reset_index()
    T = T[T.n >= MIN_UNITS]; T["frac"] = T.k / T.n
    rows = []
    for reg, d in T.groupby(level):
        g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])].frac.to_numpy() for k in m62.GROUPS}
        row = {level: reg, **{f"mean {m62.GLAB[k]}": g[k].mean() if len(g[k]) else np.nan for k in m62.GROUPS},
               **{f"sem {m62.GLAB[k]}": g[k].std(ddof=1) / np.sqrt(len(g[k])) if len(g[k]) > 1 else np.nan for k in m62.GROUPS},
               **{f"n {m62.GLAB[k]}": len(g[k]) for k in m62.GROUPS}, "n_units": int(d.n.sum())}
        for name, (a, b) in {"R+ L vs E": (m62.GROUPS[0], m62.GROUPS[1]), "R- L vs E": (m62.GROUPS[2], m62.GROUPS[3]),
                             "expert R+ vs R-": (m62.GROUPS[3], m62.GROUPS[1])}.items():
            if len(g[a]) >= MIN_SESS and len(g[b]) >= MIN_SESS:
                row[f"{name} diff"] = g[b].mean() - g[a].mean()
                row[f"{name} p_MWU"] = stats.mannwhitneyu(g[a], g[b]).pvalue
                row[f"{name} p_Welch"] = stats.ttest_ind(g[a], g[b], equal_var=False).pvalue
        rows.append(row)
    return pd.DataFrame(rows), T


def area_figure(A, plt, out, ANOVA=None):
    import ephys_utilities.allen_utils.allen_utils as au
    order = [g for g in au.get_area_group_custom_order() if g in set(A.area_group)]
    A = A.set_index("area_group").reindex(order)
    ok = lambda g, c: all(A.loc[g, f"n {m62.GLAB[(c, s)]}"] >= MIN_SESS for s in ("learning", "expert"))
    keep = [g for g in order if ok(g, "R+") or ok(g, "R-")]          # within-cohort inclusion (both stages)
    A = A.loc[keep]
    fig, ax = plt.subplots(figsize=(m62.W_IN, 2.6))
    x = np.arange(len(keep)); w = 0.2
    for j, k in enumerate(m62.GROUPS):
        n = np.array([A.loc[g, f"n {m62.GLAB[k]}"] if ok(g, k[0]) else 0 for g in keep])
        m = np.where(n >= MIN_SESS, A[f"mean {m62.GLAB[k]}"], np.nan); e = np.where(n >= MIN_SESS, A[f"sem {m62.GLAB[k]}"], np.nan)
        fc = "white" if k[1] == "learning" else m62.COH[k[0]]
        ax.bar(x + (j - 1.5) * w, m, w * 0.92, yerr=e, color=fc, edgecolor=m62.COH[k[0]], lw=0.8,
               error_kw=dict(lw=0.7, capsize=0, ecolor=m62.COH[k[0]]), label=m62.GLAB[k])
    top = np.nanmax(A[[f"mean {m62.GLAB[k]}" for k in m62.GROUPS]].to_numpy() +
                    np.nan_to_num(A[[f"sem {m62.GLAB[k]}" for k in m62.GROUPS]].to_numpy()))
    for i, g in enumerate(keep):
        for name, dx, c in [("R+ L vs E", -0.2, m62.COH["R+"]), ("expert R+ vs R-", 0.1, "0.15")]:
            p = A.loc[g].get(f"{name} p_MWU", np.nan)
            if np.isfinite(p) and p < 0.05:
                ax.text(i + dx, top * 1.04, "*" if p >= 0.01 else "**" if p >= 0.001 else "***", ha="center", fontsize=7,
                        color=c)
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(x, [g.replace(" areas", "").replace("Somatosensory-", "SS-").replace("Lateral septal complex", "LSX")
                      for g in keep], rotation=35, ha="right")
    ax.set_ylabel("Fraction of tested units that\ngeneralize (AH ≠ FA and WH ≠ FA,\nsame sign); mean ± s.e.m. over sessions")
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=5, borderaxespad=0.2)
    ax.set_title("Modality-generalizing neurons per area group (≥ 3 sessions at both stages of a cohort; green *: R+ "
                 "learning vs expert; black *: expert R+ vs R−; Mann-Whitney over sessions)", fontsize=5.8, loc="left", pad=16)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.84, bottom=0.25)
    ax.text(0.99, 0.97, m62.anova_text(ANOVA), transform=ax.transAxes, fontsize=4.4, va="top", ha="right", color="0.25",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    m62.save(fig, out, "areas"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = unit_table()
    plt = m62.setup()
    A, T = area_stats(W, "area_group"); A.to_csv(OUT / "area_group_stats.csv", index=False)
    ok = lambda r, c: all(r[f"n {m62.GLAB[(c, st)]}"] >= MIN_SESS for st in ("learning", "expert"))
    both = [x["area_group"] for x in A.to_dict("records") if ok(x, "R+") and ok(x, "R-")]
    ANOVA = m62.area_anova(T.rename(columns={"area_group": "region"}), both, col="frac")
    if ANOVA:
        pd.DataFrame([dict(term=k, **v) for k, v in ANOVA.items()]).to_csv(OUT / "area_anova.csv", index=False)
    F, _ = area_stats(W, "area_acronym_custom"); F.to_csv(OUT / "fine_area_stats.csv", index=False)
    T.to_csv(OUT / "session_area_fractions.csv", index=False)
    area_figure(A, plt, OUT, ANOVA)
    # CCF maps via 049
    m49 = importlib.import_module("049_roc_stage_ccf")
    U = W.copy()
    uxyz = U[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].to_numpy(float)
    xyz = uxyz[np.isfinite(uxyz).all(1)]
    sl = m49.Slabs(xyz, uxyz)
    meas = pd.DataFrame(dict(valid=U.tested.to_numpy(), flag=U.transfer.to_numpy(), pos=U.transfer_pos.to_numpy(),
                             neg=(U.transfer & ~U.transfer_pos).to_numpy(), abs_sel=U[f"sel:{WF}"].abs().to_numpy(),
                             sel=np.where(U.transfer, U[f"sel:{WF}"], np.nan)), index=U.index)
    m49.make("transfer_units", U, meas, sl, OUT / "ccf", "frac_sig")
    m49.make_units("transfer_units", U, meas, sl, OUT / "ccf")
    json.dump(dict(script="065_generalizing_units_location.py", definition="sig AH vs FA and sig WH vs FA, same sign "
                   "(pre-lick ROC, all trials)", unit_set=UNIT_SET, min_units_per_session_area=MIN_UNITS,
                   min_sessions=MIN_SESS, unit_of_analysis="session", ccf="049 make (frac_sig) / make_units"),
              open(OUT / "provenance.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    cols = ["area_group"] + [f"mean {m62.GLAB[k]}" for k in m62.GROUPS] + [f"n {m62.GLAB[k]}" for k in m62.GROUPS] + \
        ["R+ L vs E p_MWU", "expert R+ vs R- p_MWU"]
    print(A[[c for c in cols if c in A]].round(3).to_string(index=False))
    f = F[(F["n R+ expert"] >= MIN_SESS)].sort_values("mean R+ expert", ascending=False).head(15)
    print(f[["area_acronym_custom", "mean R+ learning", "mean R+ expert", "mean R− expert", "n R+ expert", "n R− expert",
             "R+ L vs E p_MWU", "expert R+ vs R- p_MWU"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
