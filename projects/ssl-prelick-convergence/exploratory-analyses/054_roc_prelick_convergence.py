"""Hypothesis test: convergence of whisker and auditory (modality) representations before the lick in R+ mice.

Rationale: in R+ licks after whisker AND auditory stimuli are rewarded; in R- only auditory hits are. If learning makes
the whisker hit a rewarded "go" equivalent to the auditory hit, pre-lick activity on whisker hits should converge onto
auditory hits in R+ (expert vs learning, and more than in R-), while both stay distinct from unrewarded false alarms.

Session-level metrics (pre-lick 100 ms window, corrected first lick; 051 units tested: min FR 0.1 Hz, >= 3 trials/class),
per trial variant (all, long_rt, rt_matched, short_rt):
  M1 frac_sig_WHvsAH     fraction of tested units with significant WH vs AH selectivity (modality selectivity)
  M2 abs_sel_WHvsAH      mean |selectivity| WH vs AH
  M3 frac_sig_WHvsFA / frac_sig_AHvsFA
  M4 r_shared            Spearman r across units between sel(WH vs FA) and sel(AH vs FA) (both + = hit > FA):
                         a shared hit-vs-FA (rewarded-lick preparation) code -> higher r
  M5 d_WH_AH             population geometry: cross-validated squared Euclidean distance between the WH and AH
                         class means (units z-scored over the used trials; 20 random split-halves; / n units;
                         unbiased, can be < 0) -> lower = whisker hits closer to auditory hits. (A ratio
                         d(WH,AH)/[d(WH,AH)+d(AH,FA)] was unstable because cv distances near 0 and dropped.)
  M6 pref_hit_share      among three-class significant units, fraction preferring a hit class (WH or AH) vs FA
Convergence predicts in R+: M1, M2, M5 decrease and M4 increases from learning to expert, with a learning x cohort
interaction vs R-.
Statistics (unit = session; skill memory: session is the unit for the expert stage): stage within cohort and cohort
within stage: Mann-Whitney U and Welch t (both reported); interaction [E-L](R+) - [E-L](R-): cohort labels permuted
across mice (10000). No multiple-comparison correction. Units: all units (min FR) and, as a check, good units.
Areas: M1 and M4 per session x area group (>= 10 tested units), stage change per cohort (MWU, Welch) and interaction.
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/convergence/
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
OUT = m51.OUTROOT / "convergence"
VARIANTS = ["all", "long_rt", "rt_matched", "short_rt"]
COH = {"R+": "#00B400", "R-": "#C800C8"}
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
N_SPLIT = 20
MIN_AREA_UNITS = 10


def cv_dist(Z, lab, a, b, rng):
    ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
    if len(ia) < 4 or len(ib) < 4:
        return np.nan
    out = []
    for _ in range(N_SPLIT):
        pa, pb = rng.permutation(ia), rng.permutation(ib)
        ha, hb = len(pa) // 2, len(pb) // 2
        d1 = Z[:, pa[:ha]].mean(1) - Z[:, pb[:hb]].mean(1)
        d2 = Z[:, pa[ha:]].mean(1) - Z[:, pb[hb:]].mean(1)
        out.append(d1 @ d2 / len(Z))
    return float(np.mean(out))


def geometry(W, ss, quality, rng):
    """M5 per session x variant from the 051 per-trial npz files"""
    rows = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        X, raw, lab, rt = z["rates"].astype(float), z["raw"].astype(float), z["cls"], z["rt"]
        keys = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        q = W[W.session_id == r.session_id][["electrode_group", "cluster_id", "quality_label"]]
        keys = keys.merge(q, on=["electrode_group", "cluster_id"], how="left")
        for v in VARIANTS:
            idx = m51.trial_set(lab, rt, m51.CLASSES, v, np.random.default_rng(0))
            if len(idx) < 12:
                continue
            ok = raw[:, idx].mean(1) >= m51.MIN_FR
            if quality == "good":
                ok &= (keys.quality_label == "good").to_numpy()
            Z = X[ok][:, idx]
            sd = Z.std(1); Z = (Z[sd > 0] - Z[sd > 0].mean(1, keepdims=True)) / sd[sd > 0, None]
            if len(Z) < 10:
                continue
            l = lab[idx]
            dWA, dAF, dWF = (cv_dist(Z, l, "WH", "AH", rng), cv_dist(Z, l, "AH", "FA", rng), cv_dist(Z, l, "WH", "FA", rng))
            rows.append(dict(session_id=r.session_id, variant=v, d_WH_AH=dWA, d_AH_FA=dAF, d_WH_FA=dWF,
                             dist_ratio=dWA / (dWA + dAF) if np.isfinite(dWA + dAF) and (dWA + dAF) > 0 else np.nan,
                             n_units_geom=len(Z)))
    return pd.DataFrame(rows)


def session_metrics(W, quality):
    rows = []
    D = W if quality == "all" else W[W.quality_label == "good"]
    for v in VARIANTS:
        a, b, c = f"wh_vs_aud_hit_prelick@{v}", f"whisker_hit_vs_fa_prelick@{v}", f"auditory_hit_vs_fa_prelick@{v}"
        for sid, g in D.groupby("session_id"):
            row = dict(session_id=sid, variant=v, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0])
            for name, m in [("WHvsAH", a), ("WHvsFA", b), ("AHvsFA", c)]:
                s = g[f"sig:{m}"].dropna()
                row[f"n_tested_{name}"] = len(s)
                row[f"frac_sig_{name}"] = s.astype(float).mean() if len(s) >= 10 else np.nan
                row[f"abs_sel_{name}"] = g[f"sel:{m}"].abs().mean() if len(s) >= 10 else np.nan
            both = g[[f"sel:{b}", f"sel:{c}"]].dropna()
            row["r_shared"] = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0] if len(both) >= 10 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def tests(S, metric, rng, n_perm=10000):
    """stage within cohort, cohort within stage (MWU + Welch), interaction (mouse-level cohort permutation)"""
    out = {}
    d = S.dropna(subset=[metric])
    g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])][metric].to_numpy() for k in GROUPS}
    for name, (A, B) in {"stage:R+": (("R+", "learning"), ("R+", "expert")), "stage:R-": (("R-", "learning"), ("R-", "expert")),
                         "cohort:learning": (("R-", "learning"), ("R+", "learning")),
                         "cohort:expert": (("R-", "expert"), ("R+", "expert"))}.items():
        x, y = g[A], g[B]
        if len(x) >= 3 and len(y) >= 3:
            out[name] = dict(diff=y.mean() - x.mean(), p_mwu=stats.mannwhitneyu(x, y).pvalue,
                             p_welch=stats.ttest_ind(x, y, equal_var=False).pvalue, n_A=len(x), n_B=len(y))
    if all(len(g[k]) >= 3 for k in GROUPS):
        def stat(coh):
            m = {k: d[metric][(coh == k[0]) & (d.stage == k[1]).to_numpy()].mean() for k in GROUPS}
            return (m[("R+", "expert")] - m[("R+", "learning")]) - (m[("R-", "expert")] - m[("R-", "learning")])
        mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
        midx = pd.Index(mice).get_indexer(d.mouse_id)
        obs = stat(d.cohort.to_numpy())
        null = np.array([stat(rng.permutation(mc)[midx]) for _ in range(n_perm)])
        out["interaction"] = dict(diff=obs, p_perm=(1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm))
    return out


def fmt(p):
    return "p<.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def panel(ax, S, metric, T, ylabel, rng):
    d = S.dropna(subset=[metric])
    xs = {("R+", "learning"): 0, ("R+", "expert"): 1, ("R-", "learning"): 2.4, ("R-", "expert"): 3.4}
    for k, x in xs.items():
        v = d[(d.cohort == k[0]) & (d.stage == k[1])][metric].to_numpy()
        col = COH[k[0]]
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=6, color=col, alpha=0.35 if k[1] == "learning" else 0.8, lw=0)
        if len(v):
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar(x + 0.27, m, se, fmt="o", color=col, ms=3.5, mfc="white" if k[1] == "learning" else col, lw=1)
    ax.set_xticks(list(xs.values()), ["learn.", "expert", "learn.", "expert"])
    lines = []
    for c in ["R+", "R-"]:
        t = T.get(f"stage:{c}")
        if t:
            lines.append(f"{c.replace('-', '−')} E−L {t['diff']:+.3f}: MWU {fmt(t['p_mwu'])}, Welch {fmt(t['p_welch'])}")
    t = T.get("cohort:expert")
    if t:
        lines.append(f"expert R+−R− {t['diff']:+.3f}: MWU {fmt(t['p_mwu'])}, Welch {fmt(t['p_welch'])}")
    t = T.get("interaction")
    if t:
        lines.append(f"interaction {t['diff']:+.3f}: perm {fmt(t['p_perm'])}")
    ax.set_title("\n".join(lines), fontsize=4.4)
    ax.set_ylabel(ylabel, fontsize=5.5)


METRICS = [("frac_sig_WHvsAH", "M1 % sig. WH vs AH"), ("abs_sel_WHvsAH", "M2 |sel| WH vs AH"),
           ("r_shared", "M4 r(sel WH−FA, sel AH−FA)"), ("d_WH_AH", "M5 cv distance² WH–AH (z, per unit)"),
           ("frac_sig_WHvsFA", "M3 % sig. WH vs FA"), ("frac_sig_AHvsFA", "M3 % sig. AH vs FA")]


def area_metrics(W, quality):
    rows = []
    D = W if quality == "all" else W[W.quality_label == "good"]
    for v in ["all", "rt_matched"]:
        a, b, c = f"wh_vs_aud_hit_prelick@{v}", f"whisker_hit_vs_fa_prelick@{v}", f"auditory_hit_vs_fa_prelick@{v}"
        for (sid, ag), g in D.groupby(["session_id", "area_group"]):
            s = g[f"sig:{a}"].dropna()
            both = g[[f"sel:{b}", f"sel:{c}"]].dropna()
            rows.append(dict(session_id=sid, area_group=ag, variant=v, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0],
                             stage=g.stage.iloc[0],
                             frac_sig_WHvsAH=s.astype(float).mean() if len(s) >= MIN_AREA_UNITS else np.nan,
                             r_shared=stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0] if len(both) >= MIN_AREA_UNITS else np.nan))
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5.2, "ytick.labelsize": 5.2, "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    all_tests = []
    for quality in ["all", "good"]:
        S = session_metrics(W, quality)
        G = geometry(W, ss, quality, rng)
        S = S.merge(G, on=["session_id", "variant"], how="left")
        S.to_csv(OUT / f"session_metrics_{quality}.csv", index=False)
        fig, axs = plt.subplots(len(METRICS), len(VARIANTS), figsize=(7.4, 1.55 * len(METRICS) + 0.6),
                                gridspec_kw=dict(hspace=1.05, wspace=0.45))
        for i, (m, lab) in enumerate(METRICS):
            for j, v in enumerate(VARIANTS):
                Sv = S[S.variant == v]
                T = tests(Sv, m, rng)
                for k, t in T.items():
                    all_tests.append(dict(quality=quality, variant=v, metric=m, comparison=k, **t))
                panel(axs[i, j], Sv, m, T, lab if j == 0 else "", rng)
                if i == 0:
                    axs[i, j].text(0.5, 1.75, {"all": "all trials", "long_rt": "long RT (>= 250 ms)",
                                               "rt_matched": "RT-matched", "short_rt": "short RT (< 250 ms)"}[v],
                                   transform=axs[i, j].transAxes, ha="center", fontsize=6.5, weight="bold")
        fig.suptitle(f"Modality convergence before the lick ({quality} units, min FR {m51.MIN_FR:g} Hz; unit = session; "
                     "open = learning, filled = expert)", fontsize=7, y=0.995)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.03)
        fig.savefig(OUT / f"convergence_{quality}.png", dpi=220); fig.savefig(OUT / f"convergence_{quality}.pdf"); plt.close(fig)
    TT = pd.DataFrame(all_tests); TT.to_csv(OUT / "convergence_tests.csv", index=False)
    # areas
    A = area_metrics(W, "all"); A.to_csv(OUT / "area_metrics_all.csv", index=False)
    import ephys_utilities.allen_utils.allen_utils as au
    order = [g for g in au.get_area_group_custom_order() if g in set(A.area_group)]
    arows = []
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw=dict(hspace=0.75, wspace=0.3))
    for i, m in enumerate(["frac_sig_WHvsAH", "r_shared"]):
        for j, v in enumerate(["all", "rt_matched"]):
            ax = axs[i, j]
            for k, c in enumerate(["R+", "R-"]):
                ys, ds, ps = [], [], []
                for y, ag in enumerate(order):
                    T = tests(A[(A.variant == v) & (A.area_group == ag)], m, rng, n_perm=2000)
                    t = T.get(f"stage:{c}")
                    if t:
                        ys.append(y); ds.append(t["diff"]); ps.append(t["p_mwu"])
                        arows.append(dict(metric=m, variant=v, area_group=ag, comparison=f"stage:{c}", **t))
                    if k == 0 and "interaction" in T:
                        arows.append(dict(metric=m, variant=v, area_group=ag, comparison="interaction", **T["interaction"]))
                ys = np.array(ys) + (k - 0.5) * 0.3
                ax.scatter(ds, ys, s=14, color=[COH[c] if p < 0.05 else "white" for p in ps], edgecolors=COH[c], lw=0.8,
                           label=c.replace("-", "−"), zorder=3)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_yticks(range(len(order)), [g.replace(" areas", "") for g in order], fontsize=5)
            ax.invert_yaxis(); ax.legend(frameon=False, fontsize=5)
            ax.set_xlabel(f"Δ {m} (expert − learning), session means")
            ax.set_title(f"{m} ({v}); filled = MWU p < .05", fontsize=6)
    fig.suptitle("Modality convergence by area group (all units, >= 10 tested units per session x area)", fontsize=7)
    fig.savefig(OUT / "convergence_areas.png", dpi=220); fig.savefig(OUT / "convergence_areas.pdf"); plt.close(fig)
    pd.DataFrame(arows).to_csv(OUT / "convergence_area_tests.csv", index=False)
    json.dump(dict(script="054_roc_prelick_convergence.py", min_fr_hz=m51.MIN_FR, n_split=N_SPLIT, variants=VARIANTS,
                   unit_of_analysis="session", tests="MWU + Welch; interaction: mouse-level cohort permutation (10000)",
                   correction="none", min_area_units=MIN_AREA_UNITS), open(OUT / "provenance.json", "w"), indent=1)
    print(TT[(TT.variant.isin(["all", "rt_matched", "long_rt"]))].round(4).to_string())


if __name__ == "__main__":
    main()
