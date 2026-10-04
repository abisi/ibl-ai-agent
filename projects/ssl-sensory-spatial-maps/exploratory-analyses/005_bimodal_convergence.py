"""005 -- Are bimodal (whisker + auditory responsive) neurons enriched where whisker and auditory cortex both project?

Units: good + mua, all sessions pooled; categories from 003.bimodal_classes (responsive to a modality = any of its ROC
stimulus tests significant after Bonferroni over the modality's number of tests in the session; all epochs).
Quantity: fraction of sensory-responsive units (responsive to >= 1 modality) that are bimodal.
Locations (unit positions, ML folded onto the right hemisphere, merged atlas labels as in 003):
  top-8 converging areas = the 8 structures holding the largest volume of the overlap of the merged whisker and auditory
  projection zones (002 projection_overlap.csv; generic labels such as "MB" -- unassigned midbrain -- skipped);
  overlap zone = inside zone70_whisker & zone70_auditory (002, 50-um grid).
Tests (session = unit of analysis):
  per session, fraction bimodal among its responsive units in the location vs elsewhere (sessions with >= MIN_RESP
  responsive units in both): mean +- s.e.m. over sessions; paired Wilcoxon signed-rank and paired t-test; within-session
  permutation (location labels shuffled among a session's responsive units, N_PERM) of the mean session difference
  (one-sided, location > elsewhere). Per area: area vs every area outside the top 8; Holm across the 8 areas.
  Control for area identity: within each top-8 structure, units inside vs outside the overlap zone (Holm across areas).
Output: combined_results_ks4/_sensory_spatial_maps/bimodal_convergence.csv (+ _sessions.csv) and
figures/bimodal_convergence.{png,pdf,svg}
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m3 = importlib.import_module("003_spatial_maps")
S, OUT = m3.S, m3.OUT
FIG, ZTAG = m3.FIG, m3.ZTAG
N_TOP, MIN_RESP, N_PERM = 8, 5, 5000
GENERIC = {"MB", "TH", "HY", "CTX", "grey", "root", "STR", "PAL", "CB", "P", "MY", "fiber tracts"}


def top_areas():
    T = pd.read_csv(OUT / f"projection_overlap{ZTAG}.csv")
    T = T[~T.structure.isin(GENERIC)].sort_values("overlap_mm3", ascending=False)
    rec = m3.recorded_structures()
    if rec is not None:
        T = T[T.structure.isin(rec)]                       # recorded structures only (>= 10 neurons; user)
    return T.head(N_TOP).structure.tolist(), T


def in_overlap(U):
    Z = np.load(m3.ZONES_NPZ)
    ov = Z["zone70_whisker"] & Z["zone70_auditory"]
    r = float(Z["res_um"])
    ijk = np.round(np.c_[U.ccf_atlas_ap, U.ccf_atlas_dv, U.ml_f].astype(float) / r - 0.5)
    ok = np.isfinite(ijk).all(1)
    ijk = np.where(ok[:, None], ijk, 0).astype(int)
    ok &= (ijk >= 0).all(1) & (ijk < np.array(ov.shape)).all(1)
    out = np.zeros(len(U), bool)
    out[ok] = ov[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out


def session_fracs(D, loc, ref):
    """per session: fraction bimodal among responsive units in `loc` and in `ref` (boolean masks over D)"""
    rows = []
    for sid, g in D.groupby("session_id"):
        a, b = g[loc[g.index]], g[ref[g.index]]
        if len(a) >= MIN_RESP and len(b) >= MIN_RESP:
            rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], n_loc=len(a), n_ref=len(b), frac_loc=a.bimodal.mean(),
                             frac_ref=b.bimodal.mean()))
    return pd.DataFrame(rows, columns=["session_id", "mouse_id", "n_loc", "n_ref", "frac_loc", "frac_ref"])


def perm_p(D, loc, ref, rng):
    """within-session permutation of the location label among the session's responsive units in loc | ref"""
    keep = loc | ref
    E = D[keep].copy()
    E["is_loc"] = loc[keep]
    groups = []
    for sid, g in E.groupby("session_id"):
        if g.is_loc.sum() >= MIN_RESP and (~g.is_loc).sum() >= MIN_RESP:
            groups.append((g.bimodal.to_numpy(float), g.is_loc.to_numpy()))
    if not groups:
        return np.nan, np.nan

    def stat(gs):
        return np.mean([y[l].mean() - y[~l].mean() for y, l in gs])
    obs = stat(groups)
    null = np.array([stat([(y, rng.permutation(l)) for y, l in groups]) for _ in range(N_PERM)])
    return obs, (1 + np.sum(null >= obs)) / (1 + N_PERM)


def compare(D, loc, ref, name, rng):
    F = session_fracs(D, loc, ref)
    d = F.frac_loc - F.frac_ref
    obs, pp = perm_p(D, loc, ref, rng)
    w = stats.wilcoxon(d) if len(d) >= 5 and np.any(d != 0) else None
    t = stats.ttest_rel(F.frac_loc, F.frac_ref) if len(d) >= 3 else None
    return dict(comparison=name, n_sessions=len(F), n_mice=F.mouse_id.nunique() if len(F) else 0,
                n_units_loc=int(loc.sum()), n_units_ref=int(ref.sum()),
                frac_loc_mean=F.frac_loc.mean(), frac_loc_sem=F.frac_loc.sem(), frac_ref_mean=F.frac_ref.mean(),
                frac_ref_sem=F.frac_ref.sem(), diff_mean=d.mean(), diff_sem=d.sem(),
                p_wilcoxon=w.pvalue if w else np.nan, p_ttest=t.pvalue if t else np.nan, p_perm_one_sided=pp,
                pooled_frac_loc=D.bimodal[loc].mean(), pooled_frac_ref=D.bimodal[ref].mean()), F.assign(comparison=name)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


def main():
    rng = np.random.default_rng(0)
    plt = S.setup()
    U = m3.load_units()
    A = m3.Atlas()
    U["atlas_id"] = A.unit_ids(U)
    U["structure"] = U.atlas_id.map(lambda k: A.acr.get(int(k), ""))
    U["in_overlap"] = in_overlap(U)
    D = U[U.bimodal_cat.isin([1, 2, 3])].reset_index(drop=True)          # sensory-responsive units
    D["bimodal"] = D.bimodal_cat == 3
    areas, T = top_areas()
    print("top areas:", areas)
    in_top = D.structure.isin(areas).to_numpy()
    rows, sess = [], []
    r, F = compare(D, in_top, ~in_top, "top-8 converging areas vs elsewhere", rng); rows.append(r); sess.append(F)
    ov = D.in_overlap.to_numpy()
    r, F = compare(D, ov, ~ov, "inside vs outside the whisker-auditory overlap zone", rng); rows.append(r); sess.append(F)
    per = []
    for a in areas:
        la = (D.structure == a).to_numpy()
        r, F = compare(D, la, ~in_top, f"{a} vs outside the top 8", rng)
        r["area"] = a
        per.append(r); sess.append(F)
    P = pd.DataFrame(per)
    P["p_perm_holm"] = holm(P.p_perm_one_sided.fillna(1))
    # control for area identity: inside vs outside the overlap zone WITHIN the same structure
    win = []
    for a in areas:
        sa = (D.structure == a).to_numpy()
        r, F = compare(D, sa & ov, sa & ~ov, f"{a}: inside vs outside the overlap zone", rng)
        r["area"] = a
        win.append(r); sess.append(F)
    Wt = pd.DataFrame(win)
    Wt["p_perm_holm"] = holm(Wt.p_perm_one_sided.fillna(1))
    Wt["control"] = "within structure"
    R = pd.concat([pd.DataFrame(rows), P, Wt], ignore_index=True)
    R.to_csv(OUT / f"bimodal_convergence{ZTAG}.csv", index=False)
    pd.concat(sess, ignore_index=True).to_csv(OUT / f"bimodal_convergence_sessions{ZTAG}.csv", index=False)
    # pooled counts per area (descriptive)
    cnt = D.assign(area=np.where(in_top, D.structure, "elsewhere")).groupby("area").agg(
        n_responsive=("bimodal", "size"), n_bimodal=("bimodal", "sum"), frac_bimodal=("bimodal", "mean"),
        n_sessions=("session_id", "nunique"))
    cnt.to_csv(OUT / f"bimodal_convergence_counts{ZTAG}.csv")
    print(R.drop(columns=[c for c in R.columns if c.startswith("pooled")]).round(4).to_string())
    print(cnt.round(3).to_string())
    figure(plt, R, P, Wt, cnt, areas, D)
    print("ALL DONE", FIG / "bimodal_convergence.png")


def figure(plt, R, P, Wt, cnt, areas, D):
    fig, axs = plt.subplots(1, 4, figsize=(S.W_IN, 2.5), gridspec_kw=dict(width_ratios=[2.4, 1, 1, 1.9], wspace=0.6))
    PURPLE, GREY = "#7b3294", "0.55"

    def paired_bars(ax, Q, labels):
        for i, (_, q) in enumerate(Q.iterrows()):
            if not q.n_sessions:
                ax.text(i, 1, "n/a", ha="center", fontsize=4.5, color="0.5")
                continue
            ax.bar(i - 0.18, 100 * q.frac_loc_mean, 0.34, yerr=100 * np.nan_to_num(q.frac_loc_sem), color=PURPLE, lw=0,
                   error_kw=dict(lw=0.6))
            ax.bar(i + 0.18, 100 * q.frac_ref_mean, 0.34, yerr=100 * np.nan_to_num(q.frac_ref_sem), color=GREY, lw=0,
                   error_kw=dict(lw=0.6))
            top = 100 * max(q.frac_loc_mean + np.nan_to_num(q.frac_loc_sem), q.frac_ref_mean + np.nan_to_num(q.frac_ref_sem))
            if q.n_sessions >= 3:
                ax.text(i, top + 1.5, S.fmt_p(q.p_perm_holm).replace("p = ", "").replace("p < ", "<"), ha="center",
                        fontsize=4.2)
            ax.text(i, -4.5, f"{int(q.n_sessions)}", ha="center", fontsize=4.3, color="0.35")
        ax.set_xticks(range(len(Q)), labels, fontsize=5.2, rotation=0)
        ax.set_xlim(-0.6, len(Q) - 0.4)
        ax.set_ylim(-7, None)
        ax.axhline(0, color="k", lw=0.4)

    paired_bars(axs[0], P.set_index("area").reindex(areas).reset_index(), areas)
    axs[0].set_ylabel("Bimodal neurons (% of\nsensory-responsive neurons)")
    axs[0].set_title("Each top-8 area (purple) vs areas\noutside the top 8 (grey), same sessions", loc="left", fontsize=5.6)
    axs[0].text(-0.6, -6.5, "n sessions:", fontsize=4.3, color="0.35", ha="right")
    for ax, k, lab, ttl in ((axs[1], 0, ("top-8\nareas", "elsewhere"), "Top-8 areas pooled"),
                            (axs[2], 1, ("inside\noverlap", "outside"), "Overlap zone")):
        q = R.iloc[k]
        ax.bar([0, 1], [100 * q.frac_loc_mean, 100 * q.frac_ref_mean], 0.6, yerr=[100 * q.frac_loc_sem, 100 * q.frac_ref_sem],
               color=[PURPLE, GREY], lw=0, error_kw=dict(lw=0.6))
        ax.set_xticks([0, 1], lab, fontsize=5.2)
        top = 100 * max(q.frac_loc_mean + q.frac_loc_sem, q.frac_ref_mean + q.frac_ref_sem)
        ax.plot([0, 0, 1, 1], [top + 2, top + 3, top + 3, top + 2], color="k", lw=0.5)
        ax.text(0.5, top + 3.8, f"perm. {S.fmt_p(q.p_perm_one_sided)}\nWilcoxon {S.fmt_p(q.p_wilcoxon)}\nt-test {S.fmt_p(q.p_ttest)}",
                ha="center", va="bottom", fontsize=4.3, linespacing=1.1)
        ax.set_ylim(0, top + 17)
        ax.set_title(f"{ttl}\n{q.n_sessions} sessions, {q.n_mice} mice", fontsize=5.4, loc="left")
        ax.set_ylabel("Bimodal (% of responsive)")
    Q = Wt[Wt.n_sessions >= 3].reset_index(drop=True)
    paired_bars(axs[3], Q, Q.area.tolist())
    axs[3].set_title("Same structure: inside (purple) vs\noutside (grey) the overlap zone", loc="left", fontsize=5.6)
    S.letter_row(fig, axs, "abcd")
    fig.suptitle("Bimodal neurons where whisker and auditory cortex both project", x=0.02, ha="left", y=1.13, fontsize=7,
                 weight="bold")
    fig.text(0.02, 1.06, "Mean ± s.e.m. over sessions; p: within-session permutation (one-sided; Holm across areas in a and d); "
             "grey numbers: sessions", ha="left", fontsize=5.2, color="0.3")
    S.save(fig, FIG, "bimodal_convergence")
    plt.close(fig)


if __name__ == "__main__":
    main()
