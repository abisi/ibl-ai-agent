"""125 -- One-row summary figure of the single-neuron result (user 2026-10-14): in R+ mice only, whisker-hit (WH) activity of
non-sensory reward-lick neurons around the first lick becomes less like spontaneous licks (SL) and more like rewarded auditory hits (AH).
  a  two real example neurons (R+ Day 0, WH SL-like; R+ Expert, WH AH-like): lick-aligned PSTHs of AH, WH, SL (baseline-corrected with
     each event's 051 baseline, 10-ms bins, Gaussian smoothing SMOOTH_MS, mean +- SEM over events), all-event s_WS / s_WA in text
  b  time courses of the signed s_WS (WH vs SL) and s_WA (WH vs AH) (095 data, 50-ms windows; non-sensory reward-lick neurons, session
     means +- SEM; rows R+ / R-), Day 0 vs Expert MWU per bin (p < 0.05 marked)
  c  jaw opening (116, mm re. resting jaw) per cohort x stage for AH / WH / SL; 126 trial-level jaw statistics in the title
     (jaw s_WA per trial: cohort x stage; neural s_WA after jaw matching: cohort x stage)
  d  2D summary: session-mean s_WS (x) vs s_WA (y) in [-100, 0) ms, 123 B1_matched; mean +- SEM per group, Day 0 -> Expert arrows
  e  fractions of AH-like and SL-like reward-lick neurons (123 B1_matched categories), mean +- SEM with Day 0 vs Expert per cohort and
     R+ vs R- per stage (MWU; Welch in the stats csv), cohort x stage (mouse permutation)
  f  per area: Delta s_WA (Expert - Day 0) per cohort (123 all-event selected neurons, >= 3 per session x area; areas with >= 3 sessions
     per cohort at both stages), per-area cohort x stage (mouse permutation) with BH-FDR over areas (* q < 0.05), omnibus ANOVA
Colours: Expert = cohort colour, Day 0 = the same hue desaturated (DAY0).
Examples: 123 units (all events), non-sensory, reward-lick (s_AS > 0, p < 0.05), mean raw pre-lick rate >= MIN_HZ, learners; Day 0 R+
"SL-like" ranked by s_AS - s_WA - |s_WS|, Expert R+ "AH-like" by s_AS + s_WS - |s_WA|; override with --ex0 / --ex1
(session_id,electrode_group,cluster_id).
Output: <OUTROOT>/single_neuron_signed/fig_single_neuron_<pop>.{png,pdf,svg}, _examples.csv, _stats.csv, _areas.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../125_single_neuron_figure.py [--pop learners]
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m95 = importlib.import_module("095_timeresolved_split_sl")
m51, m81 = m95.m51, m95.m81
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
st26 = importlib.import_module("026_roc_rates_all_sessions")
OUT = m51.OUTROOT / "single_neuron_signed"
COH = {"R+": "#00B400", "R-": "#C800C8"}
DAY0 = {"R+": "#6BB36B", "R-": "#BB6BBB"}            # same hue, desaturated (40 % colour + 60 % grey)
CL = {"AH": "#2c2cdb", "WH": "#f7b519", "FA": "#262626"}
CLAB = {"AH": "Auditory hit", "WH": "Whisker hit", "FA": "Spontaneous lick"}
STL = {"learning": "Day 0", "expert": "Expert"}
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
XS = {("R+", "learning"): 0, ("R+", "expert"): 1, ("R-", "learning"): 2.4, ("R-", "expert"): 3.4}
PSTH_T, BIN, SMOOTH_MS, MIN_HZ = (-0.5, 0.2), 0.010, 20, 5.0


def gcolor(c, st):
    return COH[c] if st == "expert" else DAY0[c]


def cname(c):
    return c.replace("-", "−")


def fmtp(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ------------------------------------------------------------------ a: example neurons
def pick_examples(U, a):
    R = U[(U.pAS < 0.05) & (U.sAS > 0) & (U.cohort == "R+") & (U.hz >= MIN_HZ)]
    out = []
    for st, cat, score, ov in [("learning", "SL-like", R.sAS - R.sWA - R.sWS.abs(), a.ex0), ("expert", "AH-like", R.sAS + R.sWS - R.sWA.abs(), a.ex1)]:
        if ov:
            sid, eg, cid = ov.split(","); out.append(R[(R.session_id == sid) & (R.electrode_group == eg) & (R.cluster_id == cid)].iloc[0]); continue
        q = R[(R.stage == st) & (R.category == cat)].assign(score=score)
        out.append(q.sort_values("score", ascending=False).iloc[0])
    return out


def psth(ex):
    from pynwb import NWBHDF5IO
    ss = st26.all_sessions(); r = ss[ss.session_id == ex.session_id].iloc[0]
    z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
    K = np.where((z["electrode_group"].astype(str) == ex.electrode_group) & (z["cluster_id"].astype(str) == ex.cluster_id))[0][0]
    ev = z["trial_start"].astype(float) + np.nan_to_num(z["rt"].astype(float)); lab = z["cls"]
    base = (z["raw"].astype(float) - z["rates"].astype(float))[K]
    with NWBHDF5IO(str(m51.NWB / f"{ex.session_id}.nwb"), "r", load_namespaces=True) as io:
        Uq, _ = m51.ru.process_nwb_tables(io.read())
    Uq = Uq.assign(electrode_group=Uq.electrode_group.astype(str), cluster_id=Uq.cluster_id.astype(str))
    st = np.sort(np.asarray(Uq[(Uq.electrode_group == ex.electrode_group) & (Uq.cluster_id == ex.cluster_id)].spike_times.iloc[0]))
    edges = np.arange(PSTH_T[0], PSTH_T[1] + BIN / 2, BIN); t = (edges[:-1] + edges[1:]) / 2
    res = {}
    for c in ["AH", "WH", "FA"]:
        idx = np.where(lab == c)[0]
        M = np.stack([np.histogram(st[(st >= ev[i] + PSTH_T[0]) & (st < ev[i] + PSTH_T[1])] - ev[i], edges)[0] / BIN - base[i] for i in idx])
        M = gaussian_filter1d(M, SMOOTH_MS / 1e3 / BIN, axis=1)
        res[c] = (M.mean(0), M.std(0, ddof=1) / np.sqrt(len(idx)), len(idx))
    return t * 1e3, res


def panel_example(ax, ex, title, show_x):
    t, res = psth(ex)
    ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0)
    for c in ["FA", "WH", "AH"]:
        m, se, n = res[c]
        ax.plot(t, m, color=CL[c], lw=0.9, zorder=3 if c == "WH" else 2)
        ax.fill_between(t, m - se, m + se, color=CL[c], alpha=0.18, lw=0)
    ax.axvline(0, color="0.4", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(PSTH_T[0] * 1e3, PSTH_T[1] * 1e3); ax.set_xticks([-400, 0, 200]); ax.set_ylabel("Δ rate (Hz)", labelpad=1)
    ax.set_box_aspect(1)
    ax.set_title(title, loc="left", fontsize=5.4, color=COH["R+"], pad=2)
    ax.text(0.03, 0.97, f"s_WS = {ex.sWS:+.2f}\ns_WA = {ex.sWA:+.2f}", transform=ax.transAxes, fontsize=4.6, va="top", color="0.2")
    if not show_x:
        ax.set_xticklabels([])


# ------------------------------------------------------------------ b: time courses
def timecourse(D, S, t, pair):
    out = {}
    for c in ["R+", "R-"]:
        q = D[(D.pair == pair) & (D.cohort == c)]
        ses = q.groupby(["session_id", "stage"]).row.apply(list).reset_index()
        ses = ses[ses.row.apply(len) >= m81.MIN_CLASS]
        Y = {st: np.stack([S[r].mean(0) for r in ses[ses.stage == st].row]) for st in ["learning", "expert"]}
        p = np.array([mannwhitneyu(Y["learning"][:, k], Y["expert"][:, k]).pvalue for k in range(len(t))])
        out[c] = (Y, p)
    return out


def panel_tc(ax, Y, p, t, c):
    for st in ["learning", "expert"]:
        mu, se = Y[st].mean(0), Y[st].std(0, ddof=1) / np.sqrt(len(Y[st]))
        ax.plot(t, mu, color=gcolor(c, st), lw=0.9, label=f"{STL[st]} ({len(Y[st])})")
        ax.fill_between(t, mu - se, mu + se, color=gcolor(c, st), alpha=0.18, lw=0)
    ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.4", lw=0.5, ls=(0, (2, 2))); ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0)
    ax.set_xlim(t[0], t[-1]); ax.set_xticks([-400, 0, 200])
    return p


def mark_sig(ax, t, p, c):
    lo = ax.get_ylim()[0]
    ax.scatter(t[p < 0.05], np.full((p < 0.05).sum(), lo), s=1.0, color=COH[c], marker="s", lw=0, clip_on=False)


# ------------------------------------------------------------------ c: jaw
def panel_jaw(fig, spec, T):
    m116 = importlib.import_module("116_prelick_kinematics")
    sub = spec.subgridspec(2, 2, hspace=0.45, wspace=0.12)
    tms = m116.GRID * 1e3; k5 = np.ones(5) / 5; sm = lambda x: np.convolve(np.nan_to_num(x), k5, "same")
    axs, lo, hi = [], np.inf, -np.inf
    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j // 2, j % 2]); axs.append(ax)
        n = 0
        for c in ["SL", "WH", "AH"]:
            d = T[(T.measure == "jaw opening") & (T.cls == c) & (T.cohort == g[0]) & (T.stage == g[1])]
            if len(d) < 2:
                continue
            n = d.session_id.nunique()
            A = np.stack([sm(np.asarray(x)) for x in d.trace]); mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
            col = CL["FA" if c == "SL" else c]
            ax.fill_between(tms, mu - se, mu + se, color=col, alpha=0.2, lw=0); ax.plot(tms, mu, color=col, lw=0.8)
            lo, hi = min(lo, np.nanmin(mu - se)), max(hi, np.nanmax(mu + se))
        ax.axvspan(-100, 0, color="0.9", lw=0, zorder=0); ax.axvline(0, color="0.4", lw=0.4, ls=(0, (2, 2)))
        ax.set_xlim(tms[0], tms[-1]); ax.set_xticks([-400, 0])
        ax.set_title(f"{STL[g[1]]} ({n})", fontsize=4.6, color=gcolor(*g), pad=1.5)
        ax.tick_params(labelsize=4.4)
        if j % 2:
            ax.tick_params(labelleft=False)
        if j < 2:
            ax.tick_params(labelbottom=False)
    for ax in axs:
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    axs[0].set_ylabel("R+\njaw opening (mm)", fontsize=5, color=COH["R+"]); axs[2].set_ylabel("R−\njaw opening (mm)", fontsize=5, color=COH["R-"])
    return axs


# ------------------------------------------------------------------ d: 2D summary, e: fractions
def panel_2d(ax, S, rng):
    gx = m62.group_stats(S, "sWS", "s_WS", rng, "125 d sWS"); gy = m62.group_stats(S, "sWA", "s_WA", rng, "125 d sWA")
    for c in ["R+", "R-"]:
        p0 = (gx["means"][(c, "learning")], gy["means"][(c, "learning")]); p1 = (gx["means"][(c, "expert")], gy["means"][(c, "expert")])
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=COH[c], lw=0.9, mutation_scale=6, shrinkA=3, shrinkB=3))
        for st in ["learning", "expert"]:
            k = (c, st)
            ax.errorbar(gx["means"][k], gy["means"][k], xerr=gx["sems"][k], yerr=gy["sems"][k], fmt="o", ms=3.4, color=gcolor(c, st),
                        lw=0.8, capsize=0, zorder=3)
    ax.set_xlabel("s_WS: WH vs SL (→ less SL-like)"); ax.set_ylabel("s_WA: WH vs AH (↑ more AH-like)", labelpad=1)
    ax.set_box_aspect(1)
    txt = ("(s_WS | s_WA)\n"
           f"R+ Day 0→Expert: {fmtp(gx['R+ L vs E'][0])} | {fmtp(gy['R+ L vs E'][0])}\n"
           f"R− Day 0→Expert: {fmtp(gx['R- L vs E'][0])} | {fmtp(gy['R- L vs E'][0])}\n"
           f"Day 0 R+ vs R−: {fmtp(gx['learning R+ vs R-'][0])} | {fmtp(gy['learning R+ vs R-'][0])}\n"
           f"Cohort × stage: {fmtp(gx['interaction'][1])} | {fmtp(gy['interaction'][1])}")
    ax.set_title(txt, fontsize=4.1, pad=3, color="0.25")


def means_panel(ax, S, col, ylab, rng, panel, show_x, title=""):
    """group means +- SEM over sessions; brackets: Day 0 vs Expert per cohort, R+ vs R- at Day 0 and at Expert (MWU; Welch in the stats
    csv); cohort x stage interaction (mouse permutation) in the title"""
    g = m62.group_stats(S, col, ylab, rng, panel)
    for k, x in XS.items():
        ax.errorbar(x, g["means"][k], g["sems"][k], fmt="o", ms=3.4, color=gcolor(*k), mew=0.9, lw=0.9, capsize=0)
    lo, hi = min(g["means"][k] - g["sems"][k] for k in XS), max(g["means"][k] + g["sems"][k] for k in XS)
    r = hi - lo; levels = [hi + 0.12 * r, hi + 0.12 * r, hi + 0.42 * r, hi + 0.72 * r]
    for (name, a_, b_, col_), y in zip([("R+ L vs E", ("R+", "learning"), ("R+", "expert"), COH["R+"]),
                                        ("R- L vs E", ("R-", "learning"), ("R-", "expert"), COH["R-"]),
                                        ("learning R+ vs R-", ("R+", "learning"), ("R-", "learning"), "0.55"),
                                        ("expert R+ vs R-", ("R+", "expert"), ("R-", "expert"), "0.15")], levels):
        if name not in g:
            continue
        x0, x1 = XS[a_], XS[b_]
        ax.plot([x0, x0, x1, x1], [y - 0.04 * r, y, y, y - 0.04 * r], color=col_, lw=0.5)
        ax.text((x0 + x1) / 2, y + 0.01 * r, fmtp(g[name][0]), ha="center", va="bottom", fontsize=4.0, color=col_)
    ax.set_ylim(lo - 0.15 * r, hi + 0.95 * r)
    ax.set_xlim(-0.5, 3.9); ax.set_xticks(list(XS.values()))
    ax.set_xticklabels(["Day 0", "Expert", "Day 0", "Expert"] if show_x else [], fontsize=4.3, rotation=35)
    ax.set_ylabel(ylab, labelpad=1)
    ax.set_title((title + "\n" if title else "") + f"Cohort × stage {fmtp(g['interaction'][1])}", fontsize=4.8, pad=2)
    if show_x:
        for x, c in [(0.5, "R+"), (2.9, "R-")]:
            ax.annotate(cname(c), xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -27), textcoords="offset points", color=COH[c],
                        ha="center", va="top", fontsize=5.5, weight="bold", annotation_clip=False)
    return g


# ------------------------------------------------------------------ f: areas
def area_data(U, col="sWA"):
    R = U[(U.pAS < 0.05) & (U.sAS > 0) & U.area_group.notna()]
    A = R.groupby(["session_id", "mouse_id", "cohort", "stage", "area_group"]).agg(v=(col, "mean"), n=(col, "size")).reset_index()
    return A[A.n >= m81.MIN_CLASS].rename(columns={"area_group": "area"})[["area", "session_id", "mouse_id", "cohort", "stage", "v"]]


def area_xy(U, order):
    """per area (and 'all' = all selected neurons of the session) x cohort x stage: mean +- SEM over sessions of the session x area means
    of s_WS and s_WA (selected non-sensory reward-lick neurons, all events; >= 3 neurons per session x area)"""
    R = U[(U.pAS < 0.05) & (U.sAS > 0)]
    parts = [R[R.area_group.isin(order)].assign(area=lambda d: d.area_group), R.assign(area="all")]
    A = pd.concat(parts).groupby(["area", "session_id", "cohort", "stage"]).agg(x=("sWS", "mean"), y=("sWA", "mean"), n=("sWA", "size")).reset_index()
    A = A[A.n >= m81.MIN_CLASS]
    return A.groupby(["area", "cohort", "stage"]).agg(x=("x", "mean"), xs=("x", "sem"), y=("y", "mean"), ys=("y", "sem"), ns=("x", "size")).reset_index()


def lighten(hexc, f=0.55):
    import matplotlib.colors as mc
    r = np.array(mc.to_rgb(hexc)); return tuple(r + (1 - r) * f)


def panel_2d_areas(fig, spec, M, order, gcol):
    """d (trajectories): rows R+ / R-; per area a thin arrow Day 0 (light area colour) -> Expert (area colour); all neurons: bold cohort
    arrow with SEM crosses; shared axes"""
    sub = spec.subgridspec(2, 1, hspace=0.45)
    axs = [fig.add_subplot(sub[i]) for i in range(2)]
    for ax, c in zip(axs, ["R+", "R-"]):
        ax.axhline(0, color="0.85", lw=0.5, zorder=0); ax.axvline(0, color="0.85", lw=0.5, zorder=0)
        for g in order + ["all"]:
            q = M[(M.area == g) & (M.cohort == c)].set_index("stage")
            if not {"learning", "expert"} <= set(q.index):
                continue
            p0, p1 = (q.loc["learning", "x"], q.loc["learning", "y"]), (q.loc["expert", "x"], q.loc["expert", "y"])
            if g == "all":
                for st, pp in [("learning", p0), ("expert", p1)]:
                    ax.errorbar(*pp, xerr=q.loc[st, "xs"], yerr=q.loc[st, "ys"], fmt="o", ms=5.0, color=gcolor(c, st), lw=0.9, capsize=0, zorder=6,
                                mec="k", mew=0.6)
                ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4, mutation_scale=8, shrinkA=4.5, shrinkB=4.5),
                            zorder=5)
            else:
                col = gcol.get(g, "0.4")
                ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=5.5, shrinkA=2.6,
                                                                  shrinkB=2.6, alpha=0.95), zorder=2)
                ax.scatter(*p0, s=10, color=lighten(col), lw=0, zorder=3); ax.scatter(*p1, s=14, color=col, edgecolor="white", lw=0.3, zorder=3)
        ax.set_title(f"{cname(c)}", color=COH[c], fontsize=5.6, pad=1.5, weight="bold")
        ax.set_ylabel("s_WA: WH vs AH", fontsize=5, labelpad=1)
    xl = (min(a.get_xlim()[0] for a in axs), max(a.get_xlim()[1] for a in axs)); yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
    for ax in axs:
        ax.set_xlim(xl); ax.set_ylim(yl)
    axs[0].tick_params(labelbottom=False)
    axs[1].set_xlabel("s_WS: WH vs SL", fontsize=5)
    axs[0].text(0.98, 0.04, "→ less SL-like\n↑ more AH-like", transform=axs[0].transAxes, ha="right", va="bottom", fontsize=4.2, color="0.4")
    return axs


def panel_delta_areas(ax, M, order, gcol, gx, gy):
    """d (alternative): change Expert - Day 0 on both axes; per area dots in the area colour (R+ circles, R- diamonds, cohort-coloured edge);
    all neurons: large cohort-coloured markers with SEM crosses"""
    ax.axhline(0, color="0.8", lw=0.5, zorder=0); ax.axvline(0, color="0.8", lw=0.5, zorder=0)
    for c, mk in [("R+", "o"), ("R-", "D")]:
        for g in order + ["all"]:
            q = M[(M.area == g) & (M.cohort == c)].set_index("stage")
            if not {"learning", "expert"} <= set(q.index):
                continue
            dx, dy = q.loc["expert", "x"] - q.loc["learning", "x"], q.loc["expert", "y"] - q.loc["learning", "y"]
            if g == "all":
                ex_ = np.hypot(q.loc["expert", "xs"], q.loc["learning", "xs"]); ey_ = np.hypot(q.loc["expert", "ys"], q.loc["learning", "ys"])
                ax.errorbar(dx, dy, xerr=ex_, yerr=ey_, fmt=mk, ms=6.5, color=COH[c], mec="k", mew=0.8, ecolor=COH[c], lw=1.1, capsize=0, zorder=5)
            else:
                ax.scatter(dx, dy, s=16 if mk == "o" else 12, marker=mk, color=gcol.get(g, "0.4"), edgecolor=COH[c], lw=0.7, zorder=3)
    lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_box_aspect(1)
    ax.set_xlabel("Δ s_WS (Expert − Day 0)\n→ less SL-like", fontsize=5); ax.set_ylabel("Δ s_WA (Expert − Day 0)\n↑ more AH-like", fontsize=5, labelpad=1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=COH["R+"], mec="k"), Line2D([], [], marker="D", ls="", ms=4, color=COH["R-"], mec="k")]
    ax.legend(h, ["R+ (all neurons)", "R− (all neurons)"], frameon=False, fontsize=4.3, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    ax.set_title(f"Cohort × stage: Δs_WS {fmtp(gx['interaction'][1])}\nΔs_WA {fmtp(gy['interaction'][1])}; small: areas", fontsize=4.6, pad=2)


def panel_area(ax, S, order, rng, rows, gcol, m119, col, show_labels):
    """per-area Delta (Expert - Day 0) per cohort, filled markers; per-area cohort x stage (mouse permutation) with BH-FDR over areas
    (* q < 0.05); omnibus ANOVA (cohort x stage, x area) in the title"""
    y = np.arange(len(order))[::-1]; pv = []
    for i, g in enumerate(order):
        d = S[S.area == g]
        for k, c in enumerate(("R+", "R-")):
            a, b = (d[(d.cohort == c) & (d.stage == s_)].v for s_ in ("learning", "expert"))
            ax.errorbar(b.mean() - a.mean(), y[i] + 0.16 - 0.32 * k, xerr=np.sqrt(a.sem() ** 2 + b.sem() ** 2), fmt="o", ms=2.8,
                        color=COH[c], lw=0.8, capsize=0)
            rows.append(dict(panel=f"125 d {col}", area=g, cohort=c, change=b.mean() - a.mean(), n_day0=len(a), n_expert=len(b)))
        pv.append(m62.group_stats(d, "v", f"125 d {col} area {g}", rng, f"125 d {col} area")["interaction"][1])
    qv = m119.bh(pv)
    for i, (p_, q_) in enumerate(zip(pv, qv)):
        rows.append(dict(panel=f"125 d {col} q", area=order[i], p_perm=p_, q_BH=q_))
        if q_ < 0.05:
            ax.text(0.99, y[i], "*", transform=ax.get_yaxis_transform(), fontsize=6, va="center", ha="right", weight="bold")
    ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(y, [m119.m83.m62_short(g) for g in order] if show_labels else [], fontsize=4.6)
    if show_labels:
        for tl, g in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel(f"Δ {'s_WS' if col == 'sWS' else 's_WA'}\n(Expert − Day 0)")
    AN = m62.area_anova(S[S.area.isin(order)].rename(columns={"area": "region"}), order, col="v", n_perm=1000) or {}
    rows.append(dict(panel=f"125 d {col} ANOVA", **{f"{k} p_perm": v["p_perm"] for k, v in AN.items()}))
    pa = lambda k: fmtp(AN[k]["p_perm"]) if k in AN else "n/a"
    ax.set_title(f"{'s_WS: WH vs SL' if col == 'sWS' else 's_WA: WH vs AH'}\nCohort × stage {pa('cohort x stage')}\n× area {pa('cohort x stage x area')}",
                 fontsize=4.6, pad=2)
    return pv, qv


def jaw_figure(plt, T, J, rng, pop):
    """separate figure: jaw-opening traces (116) per cohort x stage + 126 trial-level jaw statistics and jaw-matched neural s_WA"""
    m62.STATS.clear()
    fig = plt.figure(figsize=(7.2, 2.3))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 0.75, 0.75, 0.75, 0.75], hspace=0.55, wspace=0.8, left=0.07, right=0.99, top=0.78, bottom=0.26)
    axc = panel_jaw(fig, gs[:, 0], T)
    xm = np.mean([axc[2].get_position().x0, axc[3].get_position().x1])
    fig.text(xm, 0.1, "Time from first lick (ms)", ha="center", fontsize=6)
    panels = [("sWA_jaw", "Jaw s_WA per trial\n(WH vs AH)"), ("sd_ratio_WA", "Jaw trial SD\nWH / AH"),
              ("sWA_all", "Neural s_WA\n(jaw-valid events)"), ("sWA_jm", "Neural s_WA\n(jaw-matched)")]
    axs = []
    for j, (col, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[:, 1 + j]); axs.append(ax)
        means_panel(ax, J.dropna(subset=[col]), col, lab, rng, f"125 jaw {col}", show_x=True)
        if col == "sd_ratio_WA":
            ax.axhline(1, color="0.7", lw=0.5, ls=(0, (2, 2)))
    h = [plt.Line2D([], [], color=CL[c], lw=1.1) for c in ["AH", "WH", "FA"]] + \
        [plt.Line2D([], [], color=gcolor(c, st), lw=0, marker="o", ms=3.2) for c in ["R+", "R-"] for st in ["learning", "expert"]]
    fig.legend(h, [CLAB[c] for c in ["AH", "WH", "FA"]] + [f"{cname(c)} {STL[st]}" for c in ["R+", "R-"] for st in ["learning", "expert"]],
               frameon=False, fontsize=4.8, ncol=7, loc="upper left", bbox_to_anchor=(0.05, 1.0), columnspacing=1.0, handlelength=1.2)
    fig.text(0.07, 0.92, f"Jaw opening (DLC, mm re. resting jaw), [−100, 0) ms grey; trial-level statistics on the same events (sessions with jaw "
             f"tracking); jaw-matched: WH / AH matched on per-trial jaw opening (matched jaw s_WA = {J.sWA_jaw_jm.mean():+.3f})",
             fontsize=4.4, color="0.35", va="top")
    for ax, l in [(axc[0], "a"), (axs[0], "b"), (axs[2], "c")]:
        fig.text(ax.get_position().x0 - 0.04, 0.86, l, fontsize=9, weight="bold", va="top")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"fig_jaw_control_{pop}.{ext}", dpi=300)
    pd.DataFrame(m62.STATS).to_csv(OUT / f"fig_jaw_control_{pop}_stats.csv", index=False)
    plt.close(fig)


def main(a):
    plt = m62.setup()
    U = pd.read_parquet(OUT / "units.parquet")
    U = U[U.cohort.isin(["R+", "R-"])]
    if a.pop == "learners":
        U = m61.learner_filter(U)
    hz = []
    ss = st26.all_sessions()
    for r in ss[ss.session_id.isin(U[(U.cohort == "R+")].session_id.unique())].itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            hz.append(pd.DataFrame(dict(session_id=r.session_id, electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str),
                                        hz=z["raw"].astype(float).mean(1))))
    U = U.merge(pd.concat(hz), on=["session_id", "electrode_group", "cluster_id"], how="left")
    ex = pick_examples(U, a)
    pd.DataFrame(ex).to_csv(OUT / f"fig_single_neuron_{a.pop}_examples.csv", index=False)
    S = pd.read_csv(OUT / f"sessions_{a.pop}.csv"); S1 = S[S.variant == "B1_matched"]
    D = pd.read_parquet(m95.OUT / "selectivity_index.parquet").assign(row=lambda d: np.arange(len(d)))
    D = D[D.nclass.isin(m81.CLASSES) & (D.nclass != "sensory") & D.cohort.isin(["R+", "R-"])]
    if a.pop == "learners":
        D = m61.learner_filter(D)
    Sel = np.load(m95.OUT / "selectivity.npy"); t = np.load(m95.OUT / "t.npy") * 1e3
    TC = {pair: timecourse(D, Sel, t, pair) for pair in ["WS", "WA"]}
    m116 = importlib.import_module("116_prelick_kinematics")
    T = pd.read_parquet(m116.OUT / "traces.parquet"); T = T[T.cohort.isin(["R+", "R-"])]
    J = pd.read_csv(m51.OUTROOT / "jaw_control" / "sessions.csv"); J = J[J.cohort.isin(["R+", "R-"])]
    if a.pop == "learners":
        T, J = m61.learner_filter(T), m61.learner_filter(J)
    A0 = pd.read_parquet(m81.OUT / "units.parquet")[["session_id", "electrode_group", "cluster_id", "area_group"]].drop_duplicates()
    A0 = A0.assign(electrode_group=A0.electrode_group.astype(str), cluster_id=A0.cluster_id.astype(str))
    UA = U.merge(A0, on=["session_id", "electrode_group", "cluster_id"], how="left")
    AD = {col: area_data(UA, col) for col in ["sWS", "sWA"]}
    m119 = importlib.import_module("119_cosyne_three_rows")
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    AQ = {k: m119.qualify(v) for k, v in AD.items()}
    ORD = [g for g in au.get_area_group_custom_order() if all(g in AQ[k]["R+"] and g in AQ[k]["R-"] for k in AQ)]
    rng = np.random.default_rng(1); rows = []
    jaw_figure(plt, T, J, rng, a.pop)
    m62.STATS.clear()

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(2, 6, width_ratios=[0.95, 0.72, 0.72, 1.3, 0.62, 0.62], hspace=0.5, wspace=0.55,
                          left=0.055, right=0.985, top=0.76, bottom=0.17)
    axa = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    panel_example(axa[0], ex[0], "R+ Day 0: WH like SL", show_x=False)
    panel_example(axa[1], ex[1], "R+ Expert: WH like AH", show_x=True)
    axa[1].set_xlabel("Time from first lick (ms)")
    # b: compact 2 x 2 (rows R+ / R-, columns s_WS / s_WA), y shared within a column, shared x label
    axb = {}
    for j, pair in enumerate(["WS", "WA"]):
        for i, c in enumerate(["R+", "R-"]):
            ax = fig.add_subplot(gs[i, 1 + j]); axb[(c, pair)] = ax
            Y, p = TC[pair][c]
            panel_tc(ax, Y, p, t, c)
            ax.set_xticks([-400, 0]); ax.tick_params(labelsize=4.6)
            if i == 0:
                ax.set_title("s_WS: WH vs SL" if pair == "WS" else "s_WA: WH vs AH", fontsize=5.2, pad=2)
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{cname(c)}: signed s", color=COH[c], labelpad=1)
        lo = min(axb[(c, pair)].get_ylim()[0] for c in ["R+", "R-"]); hi = max(axb[(c, pair)].get_ylim()[1] for c in ["R+", "R-"])
        for c in ["R+", "R-"]:
            axb[(c, pair)].set_ylim(lo, hi); mark_sig(axb[(c, pair)], t, TC[pair][c][1], c)
    xm = np.mean([axb[("R-", "WS")].get_position().x0, axb[("R-", "WA")].get_position().x1])
    fig.text(xm, 0.075, "Time from first lick (ms)", ha="center", fontsize=6)
    # c: whole-brain 2D summary (count-matched), both cohorts
    axc = fig.add_subplot(gs[:, 3]); panel_2d(axc, S1, rng)
    # d: per area, Delta s_WS and Delta s_WA
    axd = [fig.add_subplot(gs[:, 4 + j]) for j in range(2)]
    pq = {}
    for j, col in enumerate(["sWS", "sWA"]):
        pq[col] = panel_area(axd[j], AD[col], ORD, rng, rows, gcol, m119, col, show_labels=(j == 0))
    xl = (min(ax.get_xlim()[0] for ax in axd), max(ax.get_xlim()[1] for ax in axd))
    for ax in axd:
        ax.set_xlim(xl)
    h = [plt.Line2D([], [], color=CL[c], lw=1.1) for c in ["AH", "WH", "FA"]] + \
        [plt.Line2D([], [], color=gcolor(c, st), lw=0, marker="o", ms=3.2) for c in ["R+", "R-"] for st in ["learning", "expert"]]
    fig.legend(h, [CLAB[c] for c in ["AH", "WH", "FA"]] + [f"{cname(c)} {STL[st]}" for c in ["R+", "R-"] for st in ["learning", "expert"]],
               frameon=False, fontsize=4.8, ncol=7, loc="upper left", bbox_to_anchor=(0.045, 1.0), columnspacing=1.0, handlelength=1.2)
    fig.text(0.055, 0.925, "Non-sensory reward-lick neurons; signed ROC selectivity s = 2 AUC − 1, grey: [−100, 0) ms before the first lick; "
             "mean ± SEM over sessions; c: count-matched selection; d: all-event selection, ≥ 3 sessions per cohort and stage", fontsize=4.4,
             color="0.35", va="top")
    for ax, l in [(axa[0], "a"), (axb[("R+", "WS")], "b"), (axc, "c"), (axd[0], "d")]:
        fig.text(ax.get_position().x0 - (0.075 if l == "d" else 0.035), 0.87, l, fontsize=9, weight="bold", va="top")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"fig_single_neuron_{a.pop}.{ext}", dpi=300)
    pd.DataFrame(m62.STATS).to_csv(OUT / f"fig_single_neuron_{a.pop}_stats.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / f"fig_single_neuron_{a.pop}_areas.csv", index=False)
    print("areas", ORD)
    for col, (pv, qv) in pq.items():
        print(col, "per-area cohort x stage p", np.round(pv, 4), "BH q", np.round(qv, 4))
    print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--pop", default="learners", choices=["learners", "all"])
    p.add_argument("--ex0", default=""); p.add_argument("--ex1", default="")
    main(p.parse_args())
