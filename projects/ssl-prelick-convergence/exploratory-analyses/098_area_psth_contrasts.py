"""098 -- Lick-aligned PSTHs per area group: direct conditions and within-session contrasts, R+ (top block) and R- (bottom block)
(user 2026-10-11: "how to compare cohorts with different mice / recordings / overall activity; do all comparisons in a grid"). Exploratory.
Data: 051 first-lick PSTHs (10-ms bins, -0.6 .. +0.4 s, 3-bin boxcar; edge bins dropped) of the tested units of 081 (stable, mean
pre-lick rate >= 0.1 Hz), minus each unit's pre-trial baseline for that class (mean rate in [-1, -0.015] s before trial start; for
spontaneous licks the 051 baseline [-1, -0.5] s before the lick), as 062 load_psth_prestart. Per session x area group: mean over
units (>= 5); figure: mean +- s.e.m. over sessions; day 0 dashed, expert solid.
Columns
  1  direct: WH (yellow), AH (blue), SL (black) responses (spk/s)
  2  WH - AH within session (same neurons, same session: overall activity cancels; AH rewarded in both cohorts)
  3  WH - SL within session (same lick, no stimulus)
Statistics on the [-100, 0) ms window mean, per area and column: day 0 vs expert within cohort (Mann-Whitney, in the panel title);
for the contrasts also the cohort x stage interaction (mouse-level permutation, 062 group_stats) in the R- panel title; BH q over
areas within column in the stats file.
Reference check (printed and saved): per area and cohort x stage, mean pre-trial baseline rate and mean AH window response, to see
whether overall activity differs between cohorts.
Output: across_days/<ref>/area_psth_contrasts/{area_psth_contrasts.{png,pdf}, stats.csv, reference.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../098_area_psth_contrasts.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m88 = importlib.import_module("088_cosyne_across_days_v2")
m51, m61, m62, m81 = m88.m51, m88.m61, m88.m62, m88.m81
COH, CL = m88.COH, m88.CL
OUT = m51.OUTROOT / "area_psth_contrasts"
MIN_UNITS = 5


def load():
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    U = pd.read_parquet(m81.OUT / "units.parquet")
    U["electrode_group"] = U.electrode_group.astype(str); U["cluster_id"] = U.cluster_id.astype(str)
    U = m61.learner_filter(U)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(U.session_id.unique())]
    rows, edges = [], None
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        base = z["raw"].astype(float) - z["rates"].astype(float); lab = z["cls"]
        if any((lab == c).sum() < 3 for c in m51.CLASSES):
            continue
        b = np.column_stack([base[:, lab == c].mean(1) for c in m51.CLASSES])
        P = z["psth"].astype(float) - b[:, :, None]
        P = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, "same"), 2, P)[:, :, 1:-1]
        edges = z["psth_edges"]
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        K["row"] = np.arange(len(K))
        K = K.merge(U[U.session_id == r.session_id][["electrode_group", "cluster_id", "area_group", "mouse_id", "cohort", "stage"]],
                    on=["electrode_group", "cluster_id"], how="inner")
        for ag, g in K.groupby("area_group"):
            if len(g) < MIN_UNITS:
                continue
            m = P[g.row.to_numpy()].mean(0)                                   # class x time
            rows.append(dict(session_id=r.session_id, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0],
                             area_group=ag, n_units=len(g), WH=m[0], AH=m[1], SL=m[2], base=float(b[g.row.to_numpy()].mean())))
    tc = ((edges[:-1] + np.diff(edges) / 2) * 1e3)[1:-1]
    return pd.DataFrame(rows), tc


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def main():
    import ephys_utilities.allen_utils.allen_utils as au
    OUT.mkdir(parents=True, exist_ok=True)
    plt = m62.setup(); rng = np.random.default_rng(0); gcol = au.get_custom_area_groups_colors()
    D, tc = load()
    win = (tc >= -100) & (tc < 0)
    D["WH_AH"] = [a - b for a, b in zip(D.WH, D.AH)]; D["WH_SL"] = [a - b for a, b in zip(D.WH, D.SL)]
    for k in ("WH", "AH", "SL", "WH_AH", "WH_SL"):
        D[f"{k}_w"] = [v[win].mean() for v in D[k]]
    n = D.groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    areas = [a for a in au.get_area_group_custom_order() if a in n.index and all(n.loc[a].get((c, s), 0) >= 3 for c in ("R+", "R-") for s in ("learning", "expert"))]
    # reference check
    Ref = D.groupby(["area_group", "cohort", "stage"]).agg(base=("base", "mean"), AH_w=("AH_w", "mean"), n=("session_id", "nunique")).reset_index()
    Ref.to_csv(OUT / "reference.csv", index=False)
    ST = []
    COLS = (("direct", "response (spk/s)"), ("WH_AH", "WH − AH (spk/s)"), ("WH_SL", "WH − SL (spk/s)"))
    NA = len(areas)
    fig, axs = plt.subplots(2 * NA, 3, figsize=(3.4, 1.15 * 2 * NA + 0.6), squeeze=False)
    for bi, coh in enumerate(("R+", "R-")):
        for i, ag in enumerate(areas):
            row = bi * NA + i
            d = D[(D.area_group == ag) & (D.cohort == coh)]
            for j, (col, ylab) in enumerate(COLS):
                ax = axs[row, j]
                series = [("WH", CL["WH"]), ("AH", CL["AH"]), ("SL", CL["FA"])] if col == "direct" else [(col, COH[coh])]
                for k, c in series:
                    for s_, ls, a_ in (("learning", (0, (2.5, 1.5)), 0.6), ("expert", "-", 1.0)):
                        g = d[d.stage == s_]
                        if not len(g):
                            continue
                        A = np.stack(g[k].to_numpy()); mu = A.mean(0); se = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * mu
                        ax.fill_between(tc, mu - se, mu + se, color=c, alpha=0.12, lw=0); ax.plot(tc, mu, color=c, lw=0.8, ls=ls, alpha=a_)
                # stats on window means
                if col == "direct":
                    ttl = f"D0 {len(d[d.stage == 'learning'])}, E {len(d[d.stage == 'expert'])} sessions"
                else:
                    L_, E_ = (d[d.stage == s_][f"{col}_w"] for s_ in ("learning", "expert"))
                    p = stats.mannwhitneyu(L_, E_).pvalue if min(len(L_), len(E_)) >= 3 else np.nan
                    ttl = f"D0 vs E {m62.fmt_p(p)}"
                    ST.append(dict(area=ag, cohort=coh, measure=col, mean_D0=L_.mean(), mean_E=E_.mean(), n_D0=len(L_), n_E=len(E_), p_D0_vs_E=p))
                    if coh == "R-":
                        S = m62.group_stats(D[D.area_group == ag], f"{col}_w", f"{ag} {col}", rng, "098")
                        ip = S["interaction"][1]; ttl += f"; c×s {m62.fmt_p(ip)}"
                        ST.append(dict(area=ag, cohort="interaction", measure=col, p_perm=ip))
                ax.set_title(ttl, fontsize=3.9, pad=1.5)
                ax.axvspan(-100, 0, color="#FDD49E", alpha=0.4, lw=0, zorder=0); ax.axvline(0, color="0.4", lw=0.4, ls=":")
                ax.axhline(0, color="0.85", lw=0.4); ax.set_xlim(tc[0], tc[-1]); ax.set_box_aspect(1); ax.tick_params(labelsize=4.0)
                if row == 0:
                    ax.annotate(["WH, AH, SL", "WH − AH", "WH − SL"][j], xy=(0.5, 1.28), xycoords="axes fraction", ha="center", fontsize=5.2, weight="bold")
                if row != 2 * NA - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("time from first lick (ms)", fontsize=4.4)
                if j == 0:
                    ax.set_ylabel(f"{m88.m83.m62_short(ag)}\n{ylab}", fontsize=4.2, color=gcol.get(ag, "k"))
        y = 1 - (bi * NA + 0.0) / (2 * NA)
    fig.text(0.005, 0.75, "R+", color=COH["R+"], fontsize=9, weight="bold", rotation=90, va="center")
    fig.text(0.005, 0.27, "R−", color=COH["R-"], fontsize=9, weight="bold", rotation=90, va="center")
    fig.suptitle("Lick-aligned PSTHs per area: direct and within-session contrasts (dashed day 0, solid expert)", fontsize=5.6, y=0.999)
    fig.tight_layout(rect=(0.03, 0, 1, 0.99), h_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"area_psth_contrasts.{ext}", dpi=300)
    S = pd.DataFrame(ST)
    for mcol in ("WH_AH", "WH_SL"):
        for c in ("R+", "R-"):
            m = (S.measure == mcol) & (S.cohort == c)
            S.loc[m, "q_BH"] = bh(S.loc[m, "p_D0_vs_E"])
        m = (S.measure == mcol) & (S.cohort == "interaction")
        S.loc[m, "q_BH"] = bh(S.loc[m, "p_perm"])
    S.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(Ref.pivot_table(index="area_group", columns=["cohort", "stage"], values="base").round(2))
    print(S.round(3).to_string(index=False))
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
