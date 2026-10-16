"""128 -- Single-neuron figure from the unified analysis of 127 (user 2026-10-14): every panel uses the same events, the same non-sensory
neurons and the same count-matched reward-lick selection; baseline variant selectable (main: shared lick-relative baseline).
  a  two real example neurons (R+ Day 0, R+ Expert; most frequently selected units, ranked as in 125), lick-aligned PSTHs of AH / WH / SL
     with the variant's baseline, events valid for the variant
  b  time courses of s_WS and s_WA (rows R+ / R-), mean +- SEM over sessions; no cluster tests (user 2026-10-14)
  c  whole-brain 2D summary s_WS vs s_WA in [-100, 0) ms
  d  per area unpaired Delta = Expert session - cohort Day 0 mean (--dmode cohorts: both cohorts; diff: cohort difference of Delta,
     bootstrap CI); two-way ANOVA Delta ~ cohort x area with mouse-level cohort permutations (Day 0 means recomputed), per-area
     cohort difference with the same permutations, BH-FDR over areas within each index
Output: <OUTROOT>/unified_single_neuron_all/fig_unified_<baseline>_<measure>_<pop>[_ddiff].{png,pdf,svg}, _stats.csv, _areas.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../128_unified_figure.py [--baseline shared|orig|raw] [--pop learners]
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m125 = importlib.import_module("125_single_neuron_figure")
m127 = importlib.import_module("127_unified_single_neuron")
m51, m61, m62, m81 = m125.m51, m125.m61, m125.m62, m125.m81
OUT = m127.OUT
COH, CL, CLAB, STL, gcolor, cname, fmtp = m125.COH, m125.CL, m125.CLAB, m125.STL, m125.gcolor, m125.cname, m125.fmtp
PSTH_T, BIN, SMOOTH_MS, MIN_HZ = (-0.5, 0.2), 0.010, 30, 5.0


# ------------------------------------------------------------------ example PSTH with the variant's baseline
def psth(ex, variant):
    from pynwb import NWBHDF5IO
    from scipy.ndimage import gaussian_filter1d
    ss = m125.st26.all_sessions(); r = ss[ss.session_id == ex.session_id].iloc[0]
    z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
    K = np.where((z["electrode_group"].astype(str) == ex.electrode_group) & (z["cluster_id"].astype(str) == ex.cluster_id))[0][0]
    lab = z["cls"].astype(str); start = z["trial_start"].astype(float); lick = start + np.nan_to_num(z["rt"].astype(float))
    with NWBHDF5IO(str(m51.NWB / f"{ex.session_id}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read(); Uq, _ = m51.ru.process_nwb_tables(nwb)
        lk = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"].time_series["piezo_lick_times"].data[:], float))
    Uq = Uq.assign(electrode_group=Uq.electrode_group.astype(str), cluster_id=Uq.cluster_id.astype(str))
    st = np.sort(np.asarray(Uq[(Uq.electrode_group == ex.electrode_group) & (Uq.cluster_id == ex.cluster_id)].spike_times.iloc[0]))
    s0, s1 = m127.SHARED
    if variant == "shared":
        base = (np.searchsorted(st, lick + s1) - np.searchsorted(st, lick + s0)) / (s1 - s0)
        valid = (np.searchsorted(lk, lick + s1) == np.searchsorted(lk, lick + s0)) & ((lab == "FA") | (lick + s1 <= start))
    elif variant == "orig":
        base = (z["raw"].astype(float) - z["rates"].astype(float))[K]; valid = np.ones(len(lab), bool)
    else:
        base = np.zeros(len(lab)); valid = np.ones(len(lab), bool)
    edges = np.arange(PSTH_T[0], PSTH_T[1] + BIN / 2, BIN); t = (edges[:-1] + edges[1:]) / 2
    res = {}
    for c in ["AH", "WH", "FA"]:
        idx = np.where((lab == c) & valid)[0]
        M = np.stack([np.histogram(st[(st >= lick[i] + PSTH_T[0]) & (st < lick[i] + PSTH_T[1])] - lick[i], edges)[0] / BIN - base[i] for i in idx])
        M = gaussian_filter1d(M, SMOOTH_MS / 1e3 / BIN, axis=1)
        res[c] = (M.mean(0), M.std(0, ddof=1) / np.sqrt(len(idx)), len(idx))
    return t * 1e3, res


def panel_example(ax, ex, variant, title, show_x):
    t, res = psth(ex, variant)
    ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0)
    for c in ["FA", "WH", "AH"]:
        m, se, n = res[c]
        ax.plot(t, m, color=CL[c], lw=0.9, zorder=3 if c == "WH" else 2); ax.fill_between(t, m - se, m + se, color=CL[c], alpha=0.18, lw=0)
    ax.axvline(0, color="0.4", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(PSTH_T[0] * 1e3, PSTH_T[1] * 1e3); ax.set_xticks([-400, 0, 200]); ax.set_ylabel("Δ rate (Hz)" if variant != "raw" else "Rate (Hz)", labelpad=1)
    ax.set_box_aspect(1); ax.set_title(title, loc="left", fontsize=5.4, color=COH["R+"], pad=2)
    ax.text(0.03, 0.97, f"s_WS = {ex.sWS:+.2f}\ns_WA = {ex.sWA:+.2f}", transform=ax.transAxes, fontsize=4.6, va="top", color="0.2")
    if not show_x:
        ax.set_xticklabels([])



def delta_table(A, col, cohort_of=None):
    """unpaired (user 2026-10-14): per Expert session x area, Delta = Expert value - the mean Day 0 value of its cohort in that area"""
    A = A if cohort_of is None else A.assign(cohort=A.mouse_id.map(cohort_of))
    d0 = A[A.stage == "learning"].groupby(["cohort", "area"])[col].mean()
    E = A[A.stage == "expert"].copy()
    E["delta"] = E[col].to_numpy() - d0.reindex(pd.MultiIndex.from_frame(E[["cohort", "area"]])).to_numpy()
    return E.dropna(subset=["delta"])[["area", "session_id", "mouse_id", "cohort", "delta"]]


def anova2(A, col, order, rng, n_perm=10000):
    """two-way ANOVA delta ~ cohort * area (OLS, type II) on the unpaired Deltas; p by permuting cohort labels across mice, Day 0 means
    recomputed in every permutation; also the per-area cohort difference of Delta (Delta_R+ - Delta_R-) with the same permutations"""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    terms = ["C(cohort)", "C(area)", "C(cohort):C(area)"]
    A = A[A.area.isin(order)]
    def stats_(cmap):
        D = delta_table(A, col, cmap)
        F = anova_lm(smf.ols("delta ~ C(cohort) * C(area)", data=D).fit(), typ=2).reindex(terms)["F"].to_numpy()
        g = D.groupby(["area", "cohort"]).delta.mean().unstack()
        diff = (g.get("R+") - g.get("R-")).reindex(order).to_numpy()
        return F, diff
    mice = A.mouse_id.unique(); mc = A.groupby("mouse_id").cohort.first().reindex(mice)
    F0, d0 = stats_(dict(mc))
    cF, cd, done = np.zeros(3), np.zeros(len(order)), 0
    for _ in range(n_perm):
        F, d = stats_(dict(zip(mice, rng.permutation(mc.to_numpy()))))
        if np.isfinite(F).all():
            cF += F >= F0 - 1e-12; cd += np.abs(d) >= np.abs(d0) - 1e-12; done += 1
    return dict(zip(["cohort", "area", "cohort x area"], (1 + cF) / (1 + done))), dict(zip(["cohort", "area", "cohort x area"], F0)), d0, (1 + cd) / (1 + done)


def panel_area_delta(ax, A, col, order, rng, rows, gcol, m119, label, show_labels):
    """per area: Delta = Expert - Day 0 (cohort means), mean +- SEM (sqrt(se_E^2 + se_D0^2)) per cohort, filled; two-way ANOVA on the unpaired
    Deltas in the title; post hoc per area: cohort difference of Delta (mouse-level permutation, BH-FDR over areas within this measure,
    * q < 0.05), MWU and Welch of the per-session Deltas in the csv"""
    from scipy.stats import mannwhitneyu, ttest_ind
    D = delta_table(A[A.area.isin(order)], col)
    P, Fv, dd, pd_ = anova2(A, col, order, rng)
    q = m119.bh(list(pd_))
    y = np.arange(len(order))[::-1]
    for i, g in enumerate(order):
        a0 = A[(A.area == g)]
        for k, c in enumerate(("R+", "R-")):
            e_, l_ = (a0[(a0.cohort == c) & (a0.stage == s_)][col] for s_ in ("expert", "learning"))
            ax.errorbar(e_.mean() - l_.mean(), y[i] + 0.16 - 0.32 * k, xerr=np.hypot(e_.sem(), l_.sem()), fmt="o", ms=2.8, color=COH[c], lw=0.8, capsize=0)
        dv = {c: D[(D.area == g) & (D.cohort == c)].delta.to_numpy() for c in ("R+", "R-")}
        rows.append(dict(panel=f"128 d {col}", area=g, n_Rp=len(dv["R+"]), n_Rm=len(dv["R-"]), delta_Rp=dv["R+"].mean(), delta_Rm=dv["R-"].mean(),
                         cohort_diff=dd[i], p_perm=pd_[i], q_perm=q[i], p_MWU=mannwhitneyu(dv["R+"], dv["R-"]).pvalue,
                         p_Welch=ttest_ind(dv["R+"], dv["R-"], equal_var=False).pvalue))
        ax.text(1.02, y[i], fmtp(pd_[i]).replace("p = ", "").replace("p ", "") + ("*" if q[i] < 0.05 else ""), transform=ax.get_yaxis_transform(), fontsize=3.9,
                va="center", ha="left", color="0.25", clip_on=False)
    ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(y, [m119.m83.m62_short(g) for g in order] if show_labels else [], fontsize=4.6)
    if show_labels:
        for tl, g in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel(f"Δ {label}\n(Expert − Day 0)")
    rows.append(dict(panel=f"128 d {col} ANOVA", **{f"{k} p_perm": v for k, v in P.items()}, **{f"{k} F": v for k, v in Fv.items()},
                     n_expert_sessions=D.session_id.nunique()))
    ax.set_title(f"{label}\nΔ R+ vs R−: {fmtp(P['cohort'])}\nvaries by area: {fmtp(P['cohort x area'])}", fontsize=4.4, pad=2)


def panel_area_diff(ax, A, cols, labels, order, rng, rows, gcol, m119, n_boot=10000):
    """d (alternative): per area the tested quantity itself, the cohort difference of the change, (Expert - Day 0)_R+ - (Expert - Day 0)_R-,
    for s_WS and s_WA side by side; 95 % bootstrap CI (sessions resampled within cohort x stage x area); p of the per-area permutation
    (mouse-level cohort shuffle, Day 0 means recomputed), BH over areas within each measure; two-way ANOVA (cohort, cohort x area) in title"""
    y = np.arange(len(order))[::-1]; tl = []
    MC = {cols[0]: "0.35", cols[1]: "#2c2cdb"}
    for j, (col, lab) in enumerate(zip(cols, labels)):
        P, Fv, dd, pd_ = anova2(A, col, order, rng); q = m119.bh(list(pd_))
        for i, g in enumerate(order):
            a0 = A[A.area == g]
            grp = {k: a0[(a0.cohort == k[0]) & (a0.stage == k[1])][col].to_numpy() for k in
                   [("R+", "expert"), ("R+", "learning"), ("R-", "expert"), ("R-", "learning")]}
            bs = np.array([(rng.choice(grp[("R+", "expert")], len(grp[("R+", "expert")])).mean() - rng.choice(grp[("R+", "learning")], len(grp[("R+", "learning")])).mean())
                           - (rng.choice(grp[("R-", "expert")], len(grp[("R-", "expert")])).mean() - rng.choice(grp[("R-", "learning")], len(grp[("R-", "learning")])).mean())
                           for _ in range(n_boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5]); yy = y[i] + 0.15 - 0.3 * j
            ax.errorbar(dd[i], yy, xerr=[[dd[i] - lo], [hi - dd[i]]], fmt="o", ms=2.8, color=MC[col], lw=0.8, capsize=0)
            ax.text(1.02, yy, fmtp(pd_[i]).replace("p = ", "").replace("p ", "") + ("*" if q[i] < 0.05 else ""), transform=ax.get_yaxis_transform(),
                    fontsize=3.7, va="center", ha="left", color=MC[col], clip_on=False)
            rows.append(dict(panel=f"128 d diff {col}", area=g, cohort_diff=dd[i], ci_lo=lo, ci_hi=hi, p_perm=pd_[i], q_perm=q[i]))
        rows.append(dict(panel=f"128 d diff {col} ANOVA", **{f"{k} p_perm": v for k, v in P.items()}))
        tl.append(f"{lab}: R+ vs R− {fmtp(P['cohort'])}, by area {fmtp(P['cohort x area'])}")
    ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(y, [m119.m83.m62_short(g) for g in order], fontsize=4.6)
    for t_, g in zip(ax.get_yticklabels(), order):
        t_.set_color(gcol.get(g, "k"))
    ax.set_xlabel("Change (Expert − Day 0): R+ minus R−\n(> 0: larger increase in R+)")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", ms=3, color=MC[c]) for c in cols], labels, frameon=False, fontsize=4.4, loc="lower right",
              handletextpad=0.2, borderaxespad=0.2)
    ax.set_title("Per area: cohort difference of the change\n" + "\n".join(tl), fontsize=4.4, pad=2)


def main(a):
    plt = m62.setup(); rng = np.random.default_rng(1); v = a.baseline
    S = pd.read_csv(OUT / "sessions.csv"); S = S[(S.variant == v) & (S.status == "ok") & S.cohort.isin(["R+", "R-"])]
    A = pd.read_csv(OUT / "areas.csv"); A = A[(A.variant == v) & A.cohort.isin(["R+", "R-"])]
    Un = pd.read_parquet(OUT / "units.parquet"); Un = Un[(Un.variant == v) & Un.cohort.isin(["R+", "R-"])]
    if a.pop == "learners":
        S, A, Un = m61.learner_filter(S), m61.learner_filter(A), m61.learner_filter(Un)
    S = S.copy(); S["session_id"] = S.session_id.astype(str)
    Z = np.load(OUT / "tc.npz"); t = Z["t"] * 1e3; pairs = list(Z["pairs"])
    ses = S[S.session_id.apply(lambda s_: f"{v}|{s_}" in Z.files)].reset_index(drop=True)
    Y = np.stack([Z[f"{v}|{s_}"] for s_ in ses.session_id])                 # sessions x pairs x time
    coh, st, mouse = ses.cohort.to_numpy(), ses.stage.to_numpy(), ses.mouse_id.to_numpy()
    # examples
    R = Un[(Un.cohort == "R+") & (Un.hz >= MIN_HZ) & (Un.f_sel >= 0.5)]
    # examples (clear cases): Day 0 SL-like |s_WS| < 0.15 and s_WA < -0.5; Expert AH-like s_WS > 0.4 and |s_WA| < 0.15; ranked by f_sel x s_AS
    # typical examples (user: less extreme): moderate reward-lick modulation (0.3 <= s_AS <= 0.6), closest to a target point:
    # Day 0 (s_WS, s_WA) = (0, -0.35), Expert (0.25, -0.05)
    # 2026-10-14 (user: "looks odd"): well-firing units (>= 20 Hz in the window) within 0.12 of the target point, the highest-rate one
    # (smoothest PSTH)
    R = R[(R.sAS >= 0.3) & (R.sAS <= 0.7) & (R.hz >= 20)]
    def near(d, x, y):
        d = d.assign(dist=np.hypot(d.sWS - x, d.sWA - y)); c_ = d[d.dist <= 0.12]
        return (c_.nlargest(1, "hz") if len(c_) else d.nsmallest(1, "dist")).iloc[0]
    e0 = near(R[R.stage == "learning"], 0.0, -0.35); e1 = near(R[R.stage == "expert"], 0.25, -0.05)
    # areas
    m119 = importlib.import_module("119_cosyne_three_rows")
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    MS = {"signed": ("WS", "WA"), "abs": ("aWS", "aWA")}[a.measure]
    LAB = {"WS": "s_WS: WH vs SL", "WA": "s_WA: WH vs AH", "aWS": "|s_WS| − null: WH vs SL", "aWA": "|s_WA| − null: WH vs AH"}
    AQ = m119.qualify(A.rename(columns={f"s{MS[1]}": "v"}))           # >= 3 sessions per cohort at both stages (unpaired)
    ORD = [g for g in au.get_area_group_custom_order() if g in AQ["R+"] and g in AQ["R-"]]
    m62.STATS.clear(); rows = []

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(2, 6, width_ratios=[0.95, 0.75, 0.75, 1.2, 0.62, 0.62], hspace=0.5, wspace=0.75,
                          left=0.055, right=0.965, top=0.74, bottom=0.2)
    axa = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    panel_example(axa[0], e0, v, "R+ Day 0: WH like SL", show_x=False)
    panel_example(axa[1], e1, v, "R+ Expert: WH like AH", show_x=True)
    axa[1].set_xlabel("Time from first lick (ms)")
    # b
    axb = {}
    gb = gs[:, 1:3].subgridspec(2, 2, wspace=0.3, hspace=0.42)              # compact square subpanels (user 2026-10-14)
    for j, pair in enumerate(["WS", "WA"]):
        k = pairs.index({"WS": MS[0], "WA": MS[1]}[pair])
        for i, c in enumerate(["R+", "R-"]):
            ax = fig.add_subplot(gb[i, j]); axb[(c, pair)] = ax; ax.set_box_aspect(1)
            for s_ in ["learning", "expert"]:
                q = Y[(coh == c) & (st == s_), k]
                mu, se = q.mean(0), q.std(0, ddof=1) / np.sqrt(len(q))
                ax.plot(t, mu, color=gcolor(c, s_), lw=0.9); ax.fill_between(t, mu - se, mu + se, color=gcolor(c, s_), alpha=0.18, lw=0)
            ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.4", lw=0.5, ls=(0, (2, 2))); ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0)
            ax.set_xlim(t[0], t[-1]); ax.set_xticks([-400, 0]); ax.tick_params(labelsize=4.6)
            if i == 0:
                ax.set_title(LAB[MS[0]] if pair == "WS" else LAB[MS[1]], fontsize=5.0, pad=2); ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{cname(c)}: {'signed s' if a.measure == 'signed' else '|s| − null'}", color=COH[c], labelpad=1)
        lo = min(axb[(c, pair)].get_ylim()[0] for c in ["R+", "R-"]); hi = max(axb[(c, pair)].get_ylim()[1] for c in ["R+", "R-"])
        for c in ["R+", "R-"]:                                   # no cluster tests (user 2026-10-14)
            axb[(c, pair)].set_ylim(lo, hi)
    xm = np.mean([axb[("R-", "WS")].get_position().x0, axb[("R-", "WA")].get_position().x1])
    fig.text(xm, 0.085, "Time from first lick (ms)", ha="center", fontsize=6)
    # c, d, e
    axc = fig.add_subplot(gs[:, 3]); m125.panel_2d(axc, S.assign(sWS=S[f"s{MS[0]}"], sWA=S[f"s{MS[1]}"]), rng)
    if a.measure == "abs":
        axc.invert_yaxis()                                   # up = smaller |s_WA| = more AH-like
        axc.set_xlabel("|s_WS| − null (→ less SL-like)"); axc.set_ylabel("|s_WA| − null (↑ more AH-like; axis inverted)", labelpad=1)
    if a.dmode == "cohorts":
        axd = [fig.add_subplot(gs[:, 4 + j]) for j in range(2)]
        for j, col in enumerate(["sWS", "sWA"]):
            pp = MS[j]; panel_area_delta(axd[j], A, f"s{pp}", ORD, rng, rows, gcol, m119, LAB[pp].split(":")[0], show_labels=(j == 0))
        xl = (min(x.get_xlim()[0] for x in axd), max(x.get_xlim()[1] for x in axd))
        for x in axd:
            x.set_xlim(xl)
    else:
        axd = [fig.add_subplot(gs[:, 4:6])]
        panel_area_diff(axd[0], A, [f"s{MS[0]}", f"s{MS[1]}"], [LAB[MS[0]].split(":")[0], LAB[MS[1]].split(":")[0]], ORD, rng, rows, gcol, m119)
    h = [plt.Line2D([], [], color=CL[c], lw=1.1) for c in ["AH", "WH", "FA"]] + \
        [plt.Line2D([], [], color=gcolor(c, s_), lw=0, marker="o", ms=3.2) for c in ["R+", "R-"] for s_ in ["learning", "expert"]]
    fig.legend(h, [CLAB[c] for c in ["AH", "WH", "FA"]] + [f"{cname(c)} {STL[s_]}" for c in ["R+", "R-"] for s_ in ["learning", "expert"]],
               frameon=False, fontsize=4.8, ncol=7, loc="upper left", bbox_to_anchor=(0.045, 1.0), columnspacing=1.0, handlelength=1.2)
    bl = {"shared": "shared baseline [−1.5, −1.0) s before the lick", "orig": "event-specific baseline (051)", "raw": "no baseline"}[v]
    fig.text(0.055, 0.93, (f"All non-sensory neurons (no selection); count-matched draws (" if m127.SELECT == "all" else f"Non-sensory neurons; reward-lick selection on count-matched draws (") + f"{m127.N_WH} WH / {m127.N_AH} AH / {m127.N_SL} SL × "
             f"{m127.N_DRAW}); {bl}; s = 2 AUC − 1; grey [−100, 0) ms; mean ± SEM over sessions", fontsize=4.4, color="0.35", va="top")
    for ax, l in [(axa[0], "a"), (axb[("R+", "WS")], "b"), (axc, "c"), (axd[0], "d")]:
        fig.text(ax.get_position().x0 - (0.075 if l == "d" else 0.035), 0.87, l, fontsize=9, weight="bold", va="top")
    name = f"fig_unified_{v}_{a.measure}_{a.pop}" + ("" if a.dmode == "cohorts" else "_ddiff")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    pd.DataFrame(m62.STATS).to_csv(OUT / f"{name}_stats.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / f"{name}_areas.csv", index=False)
    St = pd.DataFrame(m62.STATS)
    cols = ["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert",
            "n_R− learning", "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]
    pd.set_option("display.width", 300)
    print(St[~St.panel.str.contains("area")][[c for c in cols if c in St]].round(4).to_string())
    print("examples", e0.session_id, e0.electrode_group, e0.cluster_id, "|", e1.session_id, e1.electrode_group, e1.cluster_id)
    print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--baseline", default="shared", choices=["shared", "orig", "raw"])
    p.add_argument("--measure", default="signed", choices=["signed", "abs"])
    p.add_argument("--dmode", default="cohorts", choices=["cohorts", "diff"])
    p.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(p.parse_args())
