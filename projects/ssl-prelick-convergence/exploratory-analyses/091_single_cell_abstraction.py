"""091 -- Single-cell modality invariance of the pre-lick code across days (user 2026-10-10). Exploratory.
Units: tested units of 081 (stable, mean pre-lick rate >= 0.1 Hz), learners rule; unit = session for the statistics.
Pre-lick ROC (051, [-100, 0) ms before the first lick; SL reference): AS = AH vs SL, WS = WH vs SL, WA = WH vs AH.
Stimulus ROC (045 roc_long): whisker_active, auditory_active (stimulus response, active task), whisker_passive_pre,
auditory_passive_pre (passive, before the task), whisker_choice (whisker hit vs miss).
Definitions
  reward-lick      AS significant, AH > SL
  converging       reward-lick and WS significant, WH > SL
  modality-invariant  converging and WA not significant (whisker and auditory hits indistinguishable before the lick)
  sensory          significant in whisker_active or auditory_active (081)
  strictly non-sensory  not sensory and not significant in whisker_passive_pre nor auditory_passive_pre (passive data required)
  lick-gated       not sensory, whisker_choice significant (081)
  bimodal sensory  significant in both whisker_active and auditory_active
Panels
  a  area-proportional three-circle Euler diagrams (Chow & Rodgers 2005 construction: radii from set sizes, pairwise centre distances
     solved to match the pairwise overlaps; the triple overlap is approximate) of converging, lick-gated and sensory neurons among
     the tested units, pooled over sessions, per cohort x stage; exact region counts in the stats file
  b  fraction of reward-lick neurons that are modality-invariant, per session
  c  same within strictly non-sensory neurons
  d  fraction of converging neurons that are lick-gated (overlap with hit / miss decision coding)
  e  fraction of tested neurons that are bimodal sensory (control: multisensory stimulus responses)
Brackets: MWU (day 0 vs expert per cohort; R+ vs R- at day 0 and expert); titles: cohort x stage interaction (mouse permutation).
Output: across_days/<ref>/single_cell_abstraction/{units.parquet, euler_counts.csv, single_cell_abstraction.{png,pdf}, stats.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../091_single_cell_abstraction.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m88 = importlib.import_module("088_cosyne_across_days_v2")
m51, m61, m62, m81 = m88.m51, m88.m61, m88.m62, m88.m81
COH, GROUPS, GLAB = m88.COH, m88.GROUPS, m88.GLAB
OUT = m51.OUTROOT / "single_cell_abstraction"
SETS = {"converging": "#d94801", "lick-gated": "#DD8452", "sensory": "#4C72B0"}
SETS = {"converging": "#d94801", "lick-gated": "#8c6bb1", "sensory": "#4C72B0"}


def lens(r1, r2, d):
    """intersection area of two circles"""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)) + r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    return a - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))


def dist_for(r1, r2, ov):
    if ov <= 0:
        return r1 + r2
    if ov >= np.pi * min(r1, r2) ** 2:
        return abs(r1 - r2)
    return brentq(lambda d: lens(r1, r2, d) - ov, abs(r1 - r2) + 1e-9, r1 + r2 - 1e-9)


def euler3(ax, A, B, C, AB, AC, BC, names, cols):
    r = [np.sqrt(x / np.pi) for x in (A, B, C)]
    d12, d13, d23 = dist_for(r[0], r[1], AB), dist_for(r[0], r[2], AC), dist_for(r[1], r[2], BC)
    x3 = (d13 ** 2 - d23 ** 2 + d12 ** 2) / (2 * d12) if d12 > 0 else 0
    y3 = np.sqrt(max(d13 ** 2 - x3 ** 2, 0))
    cen = [(0, 0), (d12, 0), (x3, -y3)]
    from matplotlib.patches import Circle
    for (cx, cy), rr, c in zip(cen, r, cols):
        ax.add_patch(Circle((cx, cy), rr, fc=c, alpha=0.3, ec=c, lw=0.8))
    xs = [c[0] for c in cen]; ys = [c[1] for c in cen]; R = max(r)
    ax.set_xlim(min(xs) - R * 1.05, max(xs) + R * 1.05); ax.set_ylim(min(ys) - R * 1.05, max(ys) + R * 1.05)
    ax.set_aspect("equal"); ax.axis("off")


def load_units():
    U = pd.read_parquet(m81.OUT / "units.parquet")
    for k in ("electrode_group", "cluster_id"):
        U[k] = U[k].astype(str)
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U["rl"] = sig(U, f"sig:{m81.SEL_AS}") & (U[f"sel:{m81.SEL_AS}"] > 0)
    U["conv"] = U.rl & sig(U, f"sig:{m81.SEL_WS}") & (U[f"sel:{m81.SEL_WS}"] > 0)
    LP = pd.read_parquet(m51.OUTROOT / "roc_long_prelick.parquet")
    LP = LP[LP.analysis_type == "wh_vs_aud_hit_prelick@all"][["session_id", "electrode_group", "cluster_id", "sig", "sel"]]
    LP = LP.rename(columns={"sig": "sig_WA", "sel": "sel_WA"})
    L = pd.read_parquet(m81.ROC_LONG, columns=["session_id", "electrode_group", "cluster_id", "analysis_type", "sig"])
    L = L[L.analysis_type.isin(["whisker_active", "auditory_active", "whisker_passive_pre", "auditory_passive_pre", "whisker_choice"])]
    L = L.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type", values="sig", aggfunc="first").reset_index()
    for D_ in (LP, L):
        D_["electrode_group"] = D_.electrode_group.astype(str); D_["cluster_id"] = D_.cluster_id.astype(str)
    U = U.merge(LP, on=["session_id", "electrode_group", "cluster_id"], how="left").merge(L, on=["session_id", "electrode_group", "cluster_id"], how="left")
    b = lambda c: U[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U["inv"] = U.conv & ~b("sig_WA")
    U["sensory"] = U.nclass == "sensory"
    U["gated"] = U.nclass == "lick-gated"
    has_p = U.whisker_passive_pre.notna() & U.auditory_passive_pre.notna()
    U["strict_ns"] = (U.nclass == "non-sensory") & has_p & ~b("whisker_passive_pre") & ~b("auditory_passive_pre")
    U["bimodal"] = b("whisker_active") & b("auditory_active")
    return m61.learner_filter(U)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt = m62.setup(); rng = np.random.default_rng(0)
    U = load_units(); U.to_parquet(OUT / "units.parquet", index=False)
    keys = ["session_id", "mouse_id", "cohort", "stage"]
    rl = U[U.rl]; conv = U[U.conv]
    S_inv = rl.groupby(keys).agg(v=("inv", "mean"), n=("inv", "size")).reset_index(); S_inv = S_inv[S_inv.n >= 3]
    rls = U[U.rl & U.strict_ns]
    S_ns = rls.groupby(keys).agg(v=("inv", "mean"), n=("inv", "size")).reset_index(); S_ns = S_ns[S_ns.n >= 3]
    S_gc = conv.groupby(keys).agg(v=("gated", "mean"), n=("gated", "size")).reset_index(); S_gc = S_gc[S_gc.n >= 3]
    S_bi = U.groupby(keys).agg(v=("bimodal", "mean"), n=("bimodal", "size")).reset_index(); S_bi = S_bi[S_bi.n >= 10]
    rows = []
    with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9, "ytick.labelsize": 4.9}):
        fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.32))
        gs = fig.add_gridspec(1, 5, width_ratios=[1.6, 1, 1, 1, 1], wspace=0.6, left=0.02, right=0.99, top=0.78, bottom=0.18)
        sub = gs[0, 0].subgridspec(2, 2, hspace=0.6, wspace=0.15)
        for j, g in enumerate(GROUPS):
            d = U[(U.cohort == g[0]) & (U.stage == g[1])]
            A_, B_, C_ = d.conv, d.gated, d.sensory
            cnt = dict(group=GLAB[g], n_tested=len(d), conv=int(A_.sum()), gated=int(B_.sum()), sensory=int(C_.sum()),
                       conv_gated=int((A_ & B_).sum()), conv_sensory=int((A_ & C_).sum()), gated_sensory=int((B_ & C_).sum()),
                       all_three=int((A_ & B_ & C_).sum()), conv_only=int((A_ & ~B_ & ~C_).sum()))
            rows.append(cnt)
            ax = fig.add_subplot(sub[j // 2, j % 2])
            euler3(ax, cnt["conv"], cnt["gated"], cnt["sensory"], cnt["conv_gated"], cnt["conv_sensory"], cnt["gated_sensory"],
                   ["conv.", "gated", "sensory"], list(SETS.values()))
            ax.set_title(f"{GLAB[g]} ({len(d)} neurons)\nconv. {cnt['conv']}, gated {cnt['gated']}, sens. {cnt['sensory']}",
                         color=COH[g[0]], fontsize=4.3, pad=1)
        fig.text(0.02, 0.92, "a  Converging, lick-gated and sensory neurons (area-proportional)", fontsize=5.6)
        from matplotlib.patches import Patch
        fig.legend([Patch(fc=c, alpha=0.4, ec=c) for c in SETS.values()], ["converging", "lick-gated (hit/miss)", "sensory (active)"],
                   frameon=False, fontsize=4.6, ncol=3, loc="upper left", bbox_to_anchor=(0.02, 0.9), handlelength=1.0)
        for k, (S, ylab, ttl) in enumerate(((S_inv, "fraction of reward-lick neurons", "b  Modality-invariant\n(converging, WH ≈ AH)"),
                                            (S_ns, "fraction of reward-lick neurons", "c  Modality-invariant,\nstrictly non-sensory"),
                                            (S_gc, "fraction of converging neurons", "d  Converging that are\nlick-gated (hit/miss)"),
                                            (S_bi, "fraction of tested neurons", "e  Bimodal sensory\n(control)"))):
            ax = fig.add_subplot(gs[0, 1 + k]); ax.set_box_aspect(1)
            m88.mean_panel2(ax, S, "v", ylab, rng, f"091 {ttl.split()[0]}", ttl[3:])
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"single_cell_abstraction.{ext}", dpi=300)
    pd.DataFrame(rows).to_csv(OUT / "euler_counts.csv", index=False)
    S = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("091")]); S.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print(S[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning",
             "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
