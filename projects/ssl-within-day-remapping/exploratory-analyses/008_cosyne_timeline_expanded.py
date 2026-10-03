"""Expanded version of the COSYNE convergence-timeline figure: methods with real examples, results, controls.

Row 1, methods (examples from the data):
  a  One R+ learning-day session: every analysed event (WH, AH, SL) on the session time line; the session is split into
     halves at the midpoint between the two middle auditory hits.
  b  The session axis: schematic of the cross-validated SL -> AH axis and of the trial score (SL = 0, AH = 1).
  c  The slope, R+ example: trial scores of that session against normalised time, OLS lines for WH and SL; the slope
     of WH relative to SL is the quantity tested.
  d  The slope, R- example.
Row 2, results: PSTHs per half, WH - SL score across the day-0 session, day-0 slopes, decoder per half (common footing),
  changes within / across days.
Row 3, controls: whisker-hit reaction time per half (vigour), WH relative to AH (slope of WH - AH), and the same
  within-day change with an interleaved odd/even split (no temporal order: should be ~0).
Writes COSYNE_convergence_timeline_expanded.{png,pdf,svg} and a caption md.
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
m002 = importlib.import_module("002_trial_slopes")
BASE = m51.RES / f"_within_day{m51.TAG}"
OUT = BASE / "cosyne"
COH, CL, CLAB = m62.COH, m62.CL, m62.CLAB
rng = np.random.default_rng(0)
P_ = m62.fmt_p


def example_session(D2, cohort):
    """day-0 session whose WH - SL slope is closest to its cohort median, among sessions with >= 20 WH"""
    d = D2[(D2.cohort == cohort) & (D2.stage == "learning") & (D2.n_WH >= 20)].dropna(subset=["md_WH-FA_slope"])
    med = D2[(D2.cohort == cohort) & (D2.stage == "learning")]["md_WH-FA_slope"].median()
    return d.iloc[(d["md_WH-FA_slope"] - med).abs().argsort().iloc[0]].session_id


def session_scores(sid):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); r = ss[ss.session_id == sid].iloc[0]
    z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["trial_start"].astype(float)[o]
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet", columns=["session_id", "electrode_group", "cluster_id", "quality_label"])
    W = W[W.session_id == sid].astype({"electrode_group": str, "cluster_id": str})
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))).merge(
        W, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(["good", "mua"]).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    S = m002.trial_scores(X[ok].T, lab, 0)
    tau = (t - t.min()) / (t.max() - t.min())
    return lab, t, tau, S["md"], int(ok.sum())


def slope_panel(ax, lab, tau, s, c, sid, nu, R):
    out = {}
    for cl in ["FA", "AH", "WH"]:
        k = (lab == cl) & np.isfinite(s)
        ax.scatter(tau[k], s[k], s=3 if cl == "FA" else 6, color=CL[cl], alpha=0.35 if cl == "FA" else 0.8, lw=0,
                   zorder=2 if cl == "FA" else 3, label={"WH": "WH", "AH": "AH", "FA": "SL"}[cl])
        if cl in ("WH", "FA"):
            b, a = np.polyfit(tau[k], s[k], 1)
            ax.plot([0, 1], [a, a + b], color=CL[cl] if cl == "FA" else "#b07e00", lw=1.3, zorder=4)
            out[cl] = b
    ax.axhline(0, color=CL["FA"], lw=0.4, ls=(0, (2, 2))); ax.axhline(1, color=CL["AH"], lw=0.4, ls=(0, (2, 2)))
    lo, hi = np.nanpercentile(s[np.isin(lab, ["WH", "AH", "FA"])], [1, 99.5])
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.35 * (hi - lo))
    ax.set_xlabel("Time in session τ (normalised)"); ax.set_ylabel("CD projection c (SL = 0, AH = 1)")
    ax.text(0.02, 0.98, f"b_WH {out['WH']:+.2f}, b_SL {out['FA']:+.2f}\nβ = b_WH − b_SL = {out['WH'] - out['FA']:+.2f}".replace("-", "−"),
            transform=ax.transAxes, va="top", fontsize=4.6, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.set_title(f"{c.replace('-', '−')} day-0 example ({nu} units)", color=COH[c], fontsize=5.2)
    R[f"ex_{c}"] = dict(session=sid, units=nu, slope_WH=out["WH"], slope_SL=out["FA"])



def mixed_text():
    """plain-language sentence on the single-trial mixed model (009), day 0 and expert, reference = spontaneous licks"""
    f = BASE / "mixed_model" / "mixed_model_terms.csv"
    if not f.exists():
        return ""
    r = pd.read_csv(f).set_index(["stage", "reference"])
    out = []
    for st, lab in [("learning", "On the learning day"), ("expert", "within expert sessions")]:
        if (st, "spontaneous licks") not in r.index:
            continue
        x = r.loc[(st, "spontaneous licks")]
        out.append(f"{lab}, whisker hits moved toward auditory hits by {x.drift_Rplus:+.2f} (R+) and {x.drift_Rminus:+.2f} (R−) "
                   f"of the spontaneous-lick-to-auditory-hit distance over one session (R+ vs R−: shuffling cohort labels across "
                   f"mice, {P_(x.p_perm_diff)})")
    return ("**Single-trial mixed model** (every lick event of every session; model: CD projection ~ whisker hit × time in "
            "session × cohort, with a separate baseline, time trend and whisker-hit offset per session). " + "; ".join(out) + ".")


def mixed():
    """R+ vs R- difference of the day-0 whisker-hit drift (reference: spontaneous licks) from the mixed model (009)"""
    f = BASE / "mixed_model" / "mixed_model_terms.csv"
    if not f.exists():
        return ""
    r = pd.read_csv(f)
    r = r[(r.stage == "learning") & (r.reference == "spontaneous licks")]
    return "" if r.empty else f"R+ vs R−: {P_(r.p_perm_diff.iloc[0])}"


def main():
    plt = m62.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}
    D2 = pd.read_csv(BASE / "slopes" / "trial_slopes_sessions.csv")
    H = pd.read_csv(BASE / "halves" / "within_session_sessions.csv")
    ED = "epochs_n4_u150x10" if (BASE / "epochs_n4_u150x10" / "all" / "epoch_contrasts.csv").exists() else "epochs_n4"
    E = pd.read_csv(BASE / ED / "all" / "epoch_contrasts.csv")
    MM = mixed()
    TR = pd.read_csv(BASE / "slopes" / "trial_slopes_trajectories.csv")
    with plt.rc_context({"font.size": 5.2, "axes.titlesize": 5.3, "axes.labelsize": 5.1, "xtick.labelsize": 4.7,
                         "ytick.labelsize": 4.7, "legend.fontsize": 4.5, "axes.titlepad": 2.5}):
        fig = plt.figure(figsize=(m62.W_IN, 7.2))
        outer = fig.add_gridspec(3, 1, height_ratios=[1, 1.1, 0.9], hspace=0.62, left=0.065, right=0.99, top=0.93, bottom=0.06)
        # ---------------- row 1: methods
        g1 = outer[0].subgridspec(1, 4, width_ratios=[1.35, 0.9, 1.1, 1.1], wspace=0.5)
        sid_p, sid_m = example_session(D2, "R+"), example_session(D2, "R-")
        lab, t, tau, s, nu = session_scores(sid_p)
        ax_a = fig.add_subplot(g1[0])
        tm = (t - t.min()) / 60
        for k_, cl in enumerate(["WH", "AH", "FA"]):
            m = lab == cl
            ax_a.vlines(tm[m], k_ + 0.65, k_ + 1.35, color=CL[cl], lw=0.5 if cl != "FA" else 0.15, alpha=1 if cl != "FA" else 0.5)
        ta = np.sort(t[lab == "AH"]); h = len(ta) // 2; cut = (0.5 * (ta[h - 1] + ta[h]) - t.min()) / 60
        ax_a.axvline(cut, color="0.2", lw=0.8, ls=(0, (3, 2)))
        ax_a.text(cut / 2, 0.2, f"early half\n{int(((lab == 'AH') & (tm < cut)).sum())} AH, {int(((lab == 'WH') & (tm < cut)).sum())} WH",
                  ha="center", fontsize=4.5, va="center")
        ax_a.text((cut + tm.max()) / 2, 0.2, f"late half\n{int(((lab == 'AH') & (tm >= cut)).sum())} AH, {int(((lab == 'WH') & (tm >= cut)).sum())} WH",
                  ha="center", fontsize=4.5, va="center")
        ax_a.set_yticks([1, 2, 3], ["WH", "AH", "SL"]); ax_a.set_ylim(3.5, -0.4)
        for tk, cl in zip(ax_a.get_yticklabels(), ["WH", "AH", "FA"]):
            tk.set_color(CL[cl])
        ax_a.set_xlabel("Session time (min)")
        ax_a.set_title(f"Events of one R+ learning-day session; split at the median AH", fontsize=5.2)
        R["ex_events"] = dict(session=sid_p, n={c: int((lab == c).sum()) for c in ["WH", "AH", "FA"]})
        # b: axis schematic
        ax_b = fig.add_subplot(g1[1]); ax_b.axis("off"); ax_b.set_xlim(-1.3, 2.3); ax_b.set_ylim(-1.3, 1.6)
        r_ = np.random.default_rng(3)
        for cx, cy, cl in [(0, 0, "FA"), (1, 0.15, "AH")]:
            ax_b.scatter(cx + r_.normal(0, 0.22, 25), cy + r_.normal(0, 0.3, 25), s=3, color=CL[cl], alpha=0.6, lw=0)
        ax_b.annotate("", xy=(1.25, 0.19), xytext=(-0.25, -0.04), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9))
        wx, wy = 0.55, 0.9
        ax_b.scatter(wx, wy, s=14, color=CL["WH"], zorder=3)
        ax_b.plot([wx, wx + 0.02], [wy, 0.08], color=CL["WH"], lw=0.7, ls=(0, (2, 1.5)))
        ax_b.text(0, -0.75, "SL = 0", ha="center", color=CL["FA"], fontsize=4.8)
        ax_b.text(1, -0.75, "AH = 1", ha="center", color=CL["AH"], fontsize=4.8)
        ax_b.text(wx, wy + 0.18, "WH trial", ha="center", color="#b07e00", fontsize=4.8)
        ax_b.text(0.5, 1.45, "CD = mean AH − mean SL\n(cross-validated, units z-scored)", ha="center", va="top", fontsize=4.5)
        ax_b.text(0.5, -1.05, "CD projection c of each event,\nSL mean → 0, AH mean → 1", ha="center", va="top", fontsize=4.5)
        ax_b.set_title("Reward-lick coding direction (CD)", fontsize=5.2)
        # c, d: slope examples
        ax_c = fig.add_subplot(g1[2]); slope_panel(ax_c, lab, tau, s, "R+", sid_p, nu, R)
        lab2, _, tau2, s2, nu2 = session_scores(sid_m)
        ax_d = fig.add_subplot(g1[3]); slope_panel(ax_d, lab2, tau2, s2, "R-", sid_m, nu2, R)
        ax_c.legend(frameon=False, loc="lower right", markerscale=1.5, handletextpad=0.1, borderaxespad=0.1, ncol=3, columnspacing=0.6)
        # ---------------- row 2: results (as the compact figure)
        g2 = outer[1].subgridspec(1, 5, width_ratios=[2.0, 1.35, 0.75, 1.3, 1.0], wspace=0.55)
        ga = g2[0].subgridspec(2, 3, hspace=0.42, wspace=0.08)
        z = np.load(BASE / "psth_halves" / "psth_halves.npz", allow_pickle=True)
        S = pd.read_csv(BASE / "psth_halves" / "sessions.csv")
        P, Nev, edges = z["psth"], z["n"], z["edges"]
        tc = (edges[:-1] + np.diff(edges) / 2) * 1e3
        sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")
        axs_e = np.empty((2, 3), object)
        for i, c in enumerate(["R+", "R-"]):
            for j, (stg, halves, ttl) in enumerate([("learning", [0], "D0 early"), ("learning", [1], "D0 late"),
                                                    ("expert", [0, 1], "Expert")]):
                ax = fig.add_subplot(ga[i, j]); axs_e[i, j] = ax
                msk = ((S.cohort == c) & (S.stage == stg)).to_numpy()
                ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.4, ls=(0, (2, 2)))
                for k, cl in enumerate(m51.CLASSES):
                    ok = msk & (Nev[:, k, halves].min(1) >= 3)
                    A = np.array([sm(np.nanmean(p[halves], 0)) for p in P[ok, k]])
                    if len(A) < 2:
                        continue
                    mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                    ax.fill_between(tc, mu - se, mu + se, color=CL[cl], alpha=0.25, lw=0)
                    ax.plot(tc, mu, color=CL[cl], lw=0.75, label={"WH": "WH", "AH": "AH", "FA": "SL"}[cl])
                ax.set_xlim(-400, 200); ax.set_xticks([-300, 0]); ax.set_xticklabels(["−300", "0"] if i == 1 else [])
                ax.set_title(f"{c.replace('-', '−')} {ttl} ({int(msk.sum())})", color=COH[c], fontsize=4.8)
                if j == 0:
                    ax.set_ylabel("Δ rate (Hz)", fontsize=4.9)
        axs_e[0, 0].legend(frameon=False, loc="upper left", handlelength=0.8, borderaxespad=0.0, fontsize=4.2, labelspacing=0.15)
        yl = (min(a.get_ylim()[0] for a in axs_e.ravel()), max(a.get_ylim()[1] for a in axs_e.ravel()))
        for i in range(2):
            for j in range(3):
                axs_e[i, j].set_ylim(*yl)
                if j:
                    axs_e[i, j].set_yticklabels([])
        axs_e[1, 1].set_xlabel("Time from first lick (ms)", fontsize=4.9)
        md = TR[TR.axis == "md"].pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "bin"], columns="cls",
                                             values="score").reset_index()
        md["rel"] = md["WH"] - md["FA"]
        xb = (np.arange(5) + 0.5) / 5
        ax_f = fig.add_subplot(g2[1])
        for c in ["R+", "R-"]:
            g = md[(md.cohort == c) & (md.stage == "learning")]
            q = g.groupby("bin").rel.agg(["mean", "sem"]).reindex(range(5))
            ax_f.fill_between(xb, q["mean"] - q["sem"], q["mean"] + q["sem"], color=COH[c], alpha=0.15, lw=0)
            ax_f.plot(xb, q["mean"], color=COH[c], lw=1.1, marker="o", ms=2.2, mfc="white",
                      label=f"{c.replace('-', '−')} day 0 ({g.session_id.nunique()})")
            e_ = md[(md.cohort == c) & (md.stage == "expert")].groupby("session_id").rel.mean()
            ax_f.errorbar(1.13, e_.mean(), e_.sem(), fmt="o", ms=3.4, color=COH[c], capsize=0, lw=0.9, clip_on=False)
        ax_f.text(1.13, 1.0, "expert", transform=ax_f.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.4)
        ax_f.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2))); ax_f.set_xlim(0, 1.2); ax_f.set_xticks([0, 0.5, 1])
        ax_f.set_xlabel("Time in day-0 session"); ax_f.set_ylabel("CD projection, WH − SL"); ax_f.set_title("All day-0 sessions", loc="left")
        ax_f.legend(frameon=False, loc="lower left", borderaxespad=0.1)
        ax_g = fig.add_subplot(g2[2])
        col = "md_WH-FA_slope"
        d0 = D2[D2.stage == "learning"].dropna(subset=[col])
        for k_, c in enumerate(["R+", "R-"]):
            v = d0[d0.cohort == c][col]
            ax_g.scatter(k_ + rng.uniform(-0.13, 0.13, len(v)), v, s=3, color=COH[c], alpha=0.45, lw=0)
            ax_g.errorbar(k_ + 0.3, v.mean(), v.sem(), fmt="o", ms=3.4, color=COH[c], mfc="white", capsize=0, lw=0.9)
            R[f"slope_{c}"] = dict(mean=v.mean(), sem=v.sem(), n=len(v), p=stats.wilcoxon(v).pvalue)
        _, pp = m001.perm_interaction(d0.assign(stage="learning"), col, kind="learning")
        R["slope_perm"] = pp
        lo, hi = np.nanpercentile(d0[col], [1, 99]); yb = hi + 0.1 * (hi - lo)
        ax_g.set_ylim(lo - 0.15 * (hi - lo), hi + 0.3 * (hi - lo))
        ax_g.plot([0, 0, 1, 1], [yb, yb + 0.04 * (hi - lo), yb + 0.04 * (hi - lo), yb], color="0.2", lw=0.5)
        ax_g.text(0.5, yb + 0.06 * (hi - lo), P_(pp), ha="center", fontsize=4.5)
        ax_g.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_g.set_xticks([0, 1], ["R+", "R−"]); ax_g.set_xlim(-0.45, 1.55); ax_g.set_ylabel("Day-0 drift β (WH − SL)")
        ax_g.set_title("Drift β, one per session" + (f"\nmixed model {MM}" if MM else ""), fontsize=5.0)
        ax_h = fig.add_subplot(g2[3])
        xe = {"L-early": 0, "L-late": 1, "E-early": 2.3, "E-late": 3.3}
        for k_, c in enumerate(["R+", "R-"]):
            q = E[(E.measure == "dec") & (E.cohort == c) & (E.kind == "epoch")].set_index("name").reindex(list(xe))
            dx = (k_ - 0.5) * 0.16
            for seg in (("L-early", "L-late"), ("E-early", "E-late")):
                ax_h.plot([xe[s_] + dx for s_ in seg], q.loc[list(seg), "value"], color=COH[c], lw=1.0)
            ax_h.plot([xe["L-late"] + dx, xe["E-early"] + dx], q.loc[["L-late", "E-early"], "value"], color=COH[c], lw=0.6, ls=(0, (2, 2)))
            for ep, x in xe.items():
                ax_h.errorbar(x + dx, q.loc[ep, "value"], [[q.loc[ep, "value"] - q.loc[ep, "lo"]], [q.loc[ep, "hi"] - q.loc[ep, "value"]]],
                              fmt="o", ms=3.0, color=COH[c], mfc="white" if ep.startswith("L") else COH[c], lw=0.8, capsize=0,
                              label=c.replace("-", "−") if ep == "L-early" else None)
        ax_h.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_h.set_xticks(list(xe.values()), ["early", "late", "early", "late"])
        for x_, s_ in ((0.5, "Day 0"), (2.8, "Expert")):
            ax_h.annotate(s_, xy=(x_, 0), xycoords=ax_h.get_xaxis_transform(), xytext=(0, -12), textcoords="offset points",
                          ha="center", va="top", fontsize=4.8)
        ax_h.set_ylabel("P(AH|WH) − P(AH|SL) − chance"); ax_h.set_title("Decoder per half" + (" (150 units × 10)" if ED.endswith("x10") else " (150 units)"))
        ax_h.legend(frameon=False, loc="upper left", borderaxespad=0.1)
        ax_i = fig.add_subplot(g2[4])
        names = ["within-day", "across-day (early)", "carry-over"]
        for k_, c in enumerate(["R+", "R-"]):
            q = E[(E.measure == "dec") & (E.cohort == c) & (E.kind == "contrast")].set_index("name").reindex(names)
            x = np.arange(len(names)) + (k_ - 0.5) * 0.3
            for xi, (_, r) in zip(x, q.iterrows()):
                ax_i.errorbar(xi, r.value, [[r.value - r.lo], [r.hi - r.value]], fmt="o", ms=3.0, color=COH[c], lw=0.8, capsize=0,
                              mfc=COH[c] if (r.p_boot if np.isfinite(r.p_boot) else 1) < 0.05 else "white")
            R[f"contr_{c}"] = {n_: (r.value, r.lo, r.hi, r.p_boot) for n_, (_, r) in zip(names, q.iterrows())}
        pp_ = E[(E.measure == "dec") & (E.cohort == "R+ - R-")].set_index("name").reindex(names)
        R["contr_perm"] = {n_: pp_.loc[n_, "p_perm"] for n_ in names}
        for i, n_ in enumerate(names):
            ax_i.text(i, 1.01, P_(pp_.loc[n_, "p_perm"]).replace("p = ", "").replace("p < ", "<"), transform=ax_i.get_xaxis_transform(),
                      ha="center", fontsize=4.3)
        ax_i.axhline(0, color="0.3", lw=0.4)
        ax_i.set_xticks(range(3), ["within\nday 0", "across\ndays", "carry-\nover"], fontsize=4.5)
        ax_i.set_ylabel("Change in decoder readout"); ax_i.set_title("Changes", pad=7)
        # ---------------- row 3: controls
        g3 = outer[2].subgridspec(1, 3, wspace=0.5)
        ax_j = fig.add_subplot(g3[0])
        Ht = H[H.split == "time"]
        xs = {("R+", "learning"): 0, ("R+", "expert"): 1, ("R-", "learning"): 2.4, ("R-", "expert"): 3.4}
        for (c, stg), x in xs.items():
            g = Ht[(Ht.cohort == c) & (Ht.stage == stg)].dropna(subset=["rt_WH_early", "rt_WH_late"])
            for dx, h in ((-0.15, "early"), (0.15, "late")):
                v = g[f"rt_WH_{h}"]
                ax_j.errorbar(x + dx, v.mean(), v.sem(), fmt="o", ms=3.0, color=COH[c], mfc="white" if h == "early" else COH[c], lw=0.8, capsize=0)
            ax_j.plot([x - 0.15, x + 0.15], [g.rt_WH_early.mean(), g.rt_WH_late.mean()], color=COH[c], lw=0.7)
            dd_ = (g.rt_WH_late - g.rt_WH_early)
            R[f"rt_{c}_{stg}"] = dict(early=g.rt_WH_early.mean(), late=g.rt_WH_late.mean(), p=stats.wilcoxon(dd_).pvalue if len(dd_) > 2 else np.nan, n=len(g))
        ax_j.set_xticks(list(xs.values()), ["D0", "E", "D0", "E"])
        ax_j.set_ylabel("Median WH reaction time (ms)")
        ax_j.set_title("Control: WH RT, early (open) vs late half")
        ax_k = fig.add_subplot(g3[1])
        col2 = "md_WH-AH_slope"
        d0b = D2[D2.stage == "learning"].dropna(subset=[col2])
        for k_, c in enumerate(["R+", "R-"]):
            v = d0b[d0b.cohort == c][col2]
            ax_k.scatter(k_ + rng.uniform(-0.13, 0.13, len(v)), v, s=3, color=COH[c], alpha=0.45, lw=0)
            ax_k.errorbar(k_ + 0.3, v.mean(), v.sem(), fmt="o", ms=3.4, color=COH[c], mfc="white", capsize=0, lw=0.9)
            R[f"slopeAH_{c}"] = dict(mean=v.mean(), n=len(v), p=stats.wilcoxon(v).pvalue)
        _, pp2 = m001.perm_interaction(d0b.assign(stage="learning"), col2, kind="learning")
        R["slopeAH_perm"] = pp2
        ax_k.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_k.set_xticks([0, 1], ["R+", "R−"]); ax_k.set_xlim(-0.45, 1.55)
        lo, hi = np.nanpercentile(d0b[col2], [1, 99]); ax_k.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        ax_k.set_ylabel("Day-0 drift β_AH (WH − AH)"); ax_k.set_title(f"Control: WH relative to AH\nR+ vs R− {P_(pp2)}")
        ax_l = fig.add_subplot(g3[2])
        for k_, split in enumerate(["time", "oddeven"]):
            for kk, c in enumerate(["R+", "R-"]):
                g = H[(H.split == split) & (H.cohort == c) & (H.stage == "learning")]
                v = (g.dd_late - g.dd_early).dropna()
                x = k_ * 1.3 + kk * 0.45
                ax_l.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=3, color=COH[c], alpha=0.4, lw=0)
                ax_l.errorbar(x + 0.15, v.mean(), v.sem(), fmt="o", ms=3.2, color=COH[c], mfc="white", capsize=0, lw=0.9)
                R[f"split_{split}_{c}"] = dict(mean=v.mean(), n=len(v), p=stats.wilcoxon(v).pvalue)
        ax_l.axhline(0, color="0.5", lw=0.4, ls=(0, (2, 2)))
        ax_l.set_xticks([0.22, 1.52], ["time halves", "odd / even"])
        lo, hi = np.nanpercentile((H.dd_late - H.dd_early).dropna(), [2, 98]); ax_l.set_ylim(lo, hi)
        ax_l.set_ylabel("Day-0 Δd, late − early"); ax_l.set_title("Control: temporal split vs interleaved split")
        m62.letter_row(fig, [ax_a, ax_b, ax_c, ax_d], "abcd", dx_in=0.3, dy_in=0.13)
        m62.letter_row(fig, [axs_e[0, 0], ax_f, ax_g, ax_h, ax_i], "efghi", dx_in=0.28, dy_in=0.13)
        m62.letter_row(fig, [ax_j, ax_k, ax_l], "jkl", dx_in=0.3, dy_in=0.13)
        fig.text(0.065, 0.955, "Methods (examples)", fontsize=6, weight="bold", color="0.3")
        fig.text(0.065, 0.645, "Results", fontsize=6, weight="bold", color="0.3")
        fig.text(0.065, 0.305, "Controls", fontsize=6, weight="bold", color="0.3")
        fig.suptitle("R+ whisker hits converge toward auditory hits within the learning day and across days; R− diverge on day 0",
                     x=0.065, y=0.995, ha="left", fontsize=6.6, weight="bold")
        m62.save(fig, OUT, "COSYNE_convergence_timeline_expanded"); plt.close(fig)
    pd.Series({k: str(v) for k, v in R.items()}).to_csv(OUT / "COSYNE_convergence_timeline_expanded_values.csv")
    caption(R)
    print("ALL DONE", OUT)


def caption(R):
    f = lambda v: f"{v:+.2f}".replace("-", "−")
    c3 = lambda t: f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"
    txt = f"""# COSYNE_convergence_timeline_expanded

**R+ whisker hits converge toward auditory hits within the learning day and across days; R− diverge on day 0.**
**In short.** For every session we find the population direction that separates rewarded licks after the auditory tone from unrewarded spontaneous licks (the reward-lick coding direction, CD), and ask where whisker-triggered licks fall on it. In R+ mice, whose whisker licks are rewarded, whisker licks move toward the rewarded-lick end of the CD during the learning session and further across days; in R− mice they move back toward the unrewarded end.

{mixed_text()}

Conventions: pre-lick window = 100 ms before the corrected first lick; whisker hits (WH), auditory hits (AH) and
spontaneous licks (SL, unrewarded licks outside trials; reference); active trials, perf ≠ 6, auditory warm-up removed,
end-of-session disengagement trimmed; Kilosort 4 good + mua units (pre-lick rate ≥ 0.1 Hz); cohort per mouse from the
reference sheet; day 0 (D0) = learning day, expert (E) = later days; session = unit of analysis; cohort comparisons by
permuting cohort labels across mice.

**Definitions and equations.** For a session with units z-scored on training folds, let x_i be the pre-lick population
vector of event i (100 ms before the first lick).
- Reward-lick coding direction (CD): CD = (mean_{{AH, train}} x − mean_{{SL, train}} x) / ‖·‖ (5-fold cross-validation over
  the session's AH and SL; WH never used to build it).
- CD projection of event i, normalised so that the session's SL = 0 and AH = 1:
  c_i = (x_i·CD − mean_{{SL}} x·CD) / (mean_{{AH}} x·CD − mean_{{SL}} x·CD)  (held-out AH / SL; fold-averaged WH; d′ ≥ 0.3).
- Within-session drift of WH along CD (relative to SL): fit c_i = a_k + b_k τ_i separately for k ∈ {{WH, SL}} (τ_i = normalised
  time of event i in the session, 0 → 1); drift β = b_WH − b_SL (one value per session). β > 0: during the session WH
  move toward AH along CD faster than the shared drift of all events; β_AH = b_WH − b_AH uses AH as the reference.
- Decoder readout of an epoch e: D_e = mean_{{i ∈ WH_e}} p_i − mean_{{i ∈ SL_e}} p_i − median_null(D_e), where p_i = P(AH | x_i)
  from an L2 logistic regression trained on the session's AH vs SL (cross-validated), and the null re-fits the decoder
  after linearly shifting the activity against the time-ordered labels (40 shifts).
- Distance difference: Δd = d(WH, SL) − d(WH, AH), with the cross-validated squared distance per unit
  d(X, Y) = (X̄_a − Ȳ_a)·(X̄_b − Ȳ_b) / n_units over random trial halves a, b (unbiased; 0 for identical means).
- Contrasts: within day 0 = D(D0 late) − D(D0 early); across days = D(E early) − D(D0 early); carry-over = D(E early) − D(D0 late).

**Methods (examples).**
**a**, Analysed events of one R+ learning-day session ({R['ex_events']['session']}; {R['ex_events']['n']['WH']} WH,
{R['ex_events']['n']['AH']} AH, {R['ex_events']['n']['FA']} SL) on the session time line. Halves (used in e, h, i, j, l)
are split at the midpoint between the two middle auditory hits, so both halves contain the same number of AH.
**b**, Session axis: units are z-scored and the direction from the mean SL to the mean AH population vector is
computed on training folds (5-fold over AH and SL); every event is projected on it, held-out AH / SL events and all WH
events (never used for the axis). Scores are scaled so that the session's mean SL = 0 and mean AH = 1; sessions need a
reliable axis (held-out d′ ≥ 0.3). A WH score of 0.5 means half-way from SL to AH along this direction.
**c**, **d**, The slope: trial scores of one R+ and one R− day-0 session (the session closest to its cohort's median,
among sessions with ≥ 20 WH) against normalised session time (0 = first, 1 = last analysed event), with least-squares
lines for WH (gold) and SL (black). The tested quantity is slope(WH) − slope(SL): how much WH move toward AH across the
session beyond the drift that all event types share (here SL also drift: R− example SL slope {f(R['ex_R-']['slope_SL'])}).
R+ example: WH {f(R['ex_R+']['slope_WH'])}, SL {f(R['ex_R+']['slope_SL'])}; R− example: WH {f(R['ex_R-']['slope_WH'])},
SL {f(R['ex_R-']['slope_SL'])}.

**Results.**
**e**, First-lick-aligned population PSTHs (rate minus each event's baseline; mean over units per session, mean ± s.e.m.
over sessions; shaded: pre-lick window) for D0 early half, D0 late half and expert sessions; numbers: sessions.
**f**, WH − SL score in five bins of normalised day-0 session time (mean ± s.e.m. over sessions); right: expert sessions.
**g**, One slope per day-0 session (as c, d). R+ {f(R['slope_R+']['mean'])} ± {R['slope_R+']['sem']:.2f} (n =
{R['slope_R+']['n']}, Wilcoxon {P_(R['slope_R+']['p'])}); R− {f(R['slope_R-']['mean'])} ± {R['slope_R-']['sem']:.2f} (n =
{R['slope_R-']['n']}, {P_(R['slope_R-']['p'])}); R+ vs R− {P_(R['slope_perm'])}.
**h**, Common footing across days: each half of every session is estimated from exactly 4 WH, 4 AH and 4 SL events (20
subsamples) and 150 units; one cross-validated decoder per session (L2 logistic regression AH vs SL; WH never in
training) is read out per half as P(AH | WH) − P(AH | SL) minus its linear-shift chance level; mean over mice, 95%
hierarchical-bootstrap CI.
**i**, Changes of h: within day 0 (late − early), across days at the same session phase (expert early − D0 early),
carry-over (expert early − D0 late). R+: {c3(R['contr_R+']['within-day'])}, {c3(R['contr_R+']['across-day (early)'])},
{c3(R['contr_R+']['carry-over'])}; R−: {c3(R['contr_R-']['within-day'])}, {c3(R['contr_R-']['across-day (early)'])},
{c3(R['contr_R-']['carry-over'])}; R+ vs R− (top): {P_(R['contr_perm']['within-day'])}, {P_(R['contr_perm']['across-day (early)'])},
{P_(R['contr_perm']['carry-over'])}.

**Controls.**
**j**, Median WH reaction time per half: R+ D0 {R['rt_R+_learning']['early']:.0f} → {R['rt_R+_learning']['late']:.0f} ms
({P_(R['rt_R+_learning']['p'])}); R+ E {R['rt_R+_expert']['early']:.0f} → {R['rt_R+_expert']['late']:.0f} ms
({P_(R['rt_R+_expert']['p'])}); R− D0 {R['rt_R-_learning']['early']:.0f} → {R['rt_R-_learning']['late']:.0f} ms; R− E
{R['rt_R-_expert']['early']:.0f} → {R['rt_R-_expert']['late']:.0f} ms. WH do not get faster late in the session, so a gain
in lick vigour does not explain the R+ increase.
**k**, Same slope with AH as the reference, slope(WH) − slope(AH): R+ {f(R['slopeAH_R+']['mean'])} ({P_(R['slopeAH_R+']['p'])}),
R− {f(R['slopeAH_R-']['mean'])} ({P_(R['slopeAH_R-']['p'])}); R+ vs R− {P_(R['slopeAH_perm'])}.
**l**, Within-day change of the distance difference Δd = d(WH, SL) − d(WH, AH) (count-matched halves) with the
temporal split (R+ {f(R['split_time_R+']['mean'])}, {P_(R['split_time_R+']['p'])}; R− {f(R['split_time_R-']['mean'])},
{P_(R['split_time_R-']['p'])}) and with an interleaved odd / even split of the same events (R+ {f(R['split_oddeven_R+']['mean'])},
{P_(R['split_oddeven_R+']['p'])}; R− {f(R['split_oddeven_R-']['mean'])}, {P_(R['split_oddeven_R-']['p'])}): the change is
temporal, not an estimation artefact.
"""
    (OUT / "COSYNE_convergence_timeline_expanded_caption.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
