"""115 -- COSYNE combined figure: stimulus-side hit - miss axis, reward-lick CD across learning stages, decoding (CCGP) and convergence
(user 2026-10-12: max 15 panels, three per row, simple -> complex; lick-aligned panels, jaw-onset alignment as control). Plot only.
Rows (sources; learners rule: learners on day 0, all mice at expert; session = unit; 088 mean_panel2 statistics: day 0 vs expert per
cohort and R+ vs R- per stage, Mann-Whitney (Welch in the csv), cohort x stage by mouse-level permutation; n under the ticks):
  1  a task; b hit - miss axis (definition); c reward-lick CD (definition)                                     schematics
  2  d passive whisker projection on the hit - miss axis, pre / post x stage, real vs shifted; e pre; f post     hitmiss 175 (min2 axes)
  3  g single-neuron similarity index SI (104); h whisker-hit position on the CD d_WS (087); i d_WS before jaw-opening onset (087, 118)
  4  j angle theta between WH - SL and the CD (087); k cue CCGP - shift null over time (112); l train AH -> test WH (112)
  5  m CCGP vs parallelism on independent trial halves (112); n CCGP per area, 40 units x 50 draws (114); o jaw opening WH - AH before
     the lick (116)
Output: across_days/<ref>/publication/learners/COSYNE_combined.{png,pdf,svg} + COSYNE_combined_stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../115_cosyne_combined.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m88 = importlib.import_module("088_cosyne_across_days_v2")
m62, m61, m51 = m88.m62, m88.m61, m88.m51
m114 = importlib.import_module("114_ccgp_cue_sessions_equalised")
COH, CL, GROUPS = m88.COH, m88.CL, m88.GROUPS
HM = pathlib.Path.home() / "code/ibl-ai-agent/projects/ssl-hitmiss-axis-passive/exploratory-analyses"
JAW = m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}_jaw"
OUT = m51.OUTROOT / "publication" / "learners"
GREY = "0.55"


def load():
    d = {}
    d["hm"] = pd.read_csv(HM / "175_across_days_passive_projection_sessions.csv")
    d["si"] = m61.learner_filter(pd.read_csv(m51.OUTROOT / "similarity_index" / "sessions.csv"))
    h = lambda root: m61.learner_filter(pd.read_csv(root / "harmonised" / "sessions_areas.csv").query("area == 'whole_brain'"))
    d["cd"], d["cdj"] = h(m51.OUTROOT), h(JAW)
    A = m61.learner_filter(pd.read_parquet(m51.OUTROOT / "ccgp_pub2" / "sessions.parquet"))
    for k in ("CCGP", "dir_AtoW"):
        A[f"{k}_ex"] = A[k] - A[f"{k}_null"]
    d["cc"] = A[A.area == "Whole brain"]
    T = m61.learner_filter(pd.read_parquet(m51.OUTROOT / "ccgp_pub2" / "timecourse.parquet")); T["ex"] = T.CCGP - T.CCGP_null; d["tc"] = T
    d["eq"] = m61.learner_filter(pd.read_parquet(m114.OUT / f"sessions{m114.SFX}.parquet"))
    K = m61.learner_filter(pd.read_csv(m51.OUTROOT / "prelick_kinematics" / "windows.csv"))
    K = K.pivot_table(index=["session_id", "mouse_id", "cohort", "stage", "measure"], columns="cls", values="win").reset_index()
    d["kin"] = K[K.measure == "jaw opening"].assign(WH_AH=lambda q: q.WH - q.AH)
    return d


# ------------------------------------------------------------------ schematics
def box(ax, x, y, w, h, txt, fc, ec="none", fs=4.6, color="k"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, x0, y0, x1, y1, color="k", lw=0.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=6))


def panel_task(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = [("whisker", CL["WH"], "lick (WH)", "R+: reward\nR−: none"), ("tone", CL["AH"], "lick (AH)", "reward\n(both)"),
            ("none", CL["FA"], "lick (SL)", "none")]
    for i, (s, c, l, r) in enumerate(rows):
        y = 0.74 - 0.25 * i
        box(ax, 0.0, y, 0.24, 0.16, s, c, fs=4.6, color="white")
        arrow(ax, 0.25, y + 0.08, 0.32, y + 0.08); box(ax, 0.33, y, 0.3, 0.16, l, "0.9", fs=4.6)
        arrow(ax, 0.64, y + 0.08, 0.7, y + 0.08); ax.text(0.72, y + 0.08, r, fontsize=4.4, va="center")
    ax.text(0.0, 0.0, "stimulus → lick → outcome\nday 0 (first whisker day) vs expert", fontsize=4.1, color="0.3")
    ax.set_title("Task: two cohorts, same stimuli", fontsize=5.4)


def clouds(ax, rng, centers, cols, labels, n=18, s=0.11):
    for (cx, cy), c, l in zip(centers, cols, labels):
        p = rng.normal([cx, cy], s, (n, 2)); ax.plot(p[:, 0], p[:, 1], "o", ms=1.5, color=c, alpha=0.8)
        ax.text(cx, cy + 0.27, l, color=c, fontsize=4.6, ha="center", weight="bold")


def panel_hm_axis(ax):
    rng = np.random.default_rng(3); ax.set_xlim(-0.2, 1.3); ax.set_ylim(-0.25, 1.15); ax.set_xticks([]); ax.set_yticks([])
    clouds(ax, rng, [(0.15, 0.25), (0.95, 0.35)], ["0.45", CL["WH"]], ["miss", "hit"])
    arrow(ax, 0.15, 0.25, 0.95, 0.35, "#d94801", 1.1); ax.text(0.55, -0.15, "hit − miss axis", fontsize=4.5, color="#d94801", ha="center")
    for x, y, l in ((0.25, 0.85, "pre"), (0.85, 0.9, "post")):
        ax.plot(x, y, "s", ms=3.5, color="k", mfc="white" if l == "pre" else "k"); ax.text(x, y + 0.1, f"passive {l}", fontsize=4.2, ha="center")
        ax.plot([x, x], [y, 0.25 + (x - 0.15) * 0.125], ":", color="0.5", lw=0.6)
    ax.set_xlabel("neuron 1", fontsize=4.6); ax.set_ylabel("neuron 2", fontsize=4.6)
    ax.set_title("Stimulus side: hit − miss axis\n(active whisker trials, 5–35 ms)", fontsize=5.2)


def panel_cd(ax):
    rng = np.random.default_rng(5); ax.set_xlim(-0.2, 1.3); ax.set_ylim(-0.25, 1.15); ax.set_xticks([]); ax.set_yticks([])
    clouds(ax, rng, [(0.1, 0.15), (1.0, 0.15), (0.6, 0.7)], [CL["FA"], CL["AH"], CL["WH"]], ["SL", "AH", "WH"])
    arrow(ax, 0.1, 0.15, 1.0, 0.15, "#d94801", 1.1); ax.text(0.55, -0.15, "reward-lick CD (AH − SL)", fontsize=4.5, color="#d94801", ha="center")
    arrow(ax, 0.1, 0.15, 0.6, 0.7, CL["WH"], 0.8); ax.plot([0.6, 0.6], [0.7, 0.15], ":", color="0.4", lw=0.6)
    ax.text(0.62, 0.36, "d_WS", fontsize=4.3, color="0.3"); ax.text(0.24, 0.22, "θ", fontsize=5, color="0.2")
    ax.set_xlabel("neuron 1", fontsize=4.6); ax.set_ylabel("neuron 2", fontsize=4.6)
    ax.set_title("Action side: reward-lick CD\n(100 ms before the lick, cross-validated)", fontsize=5.2)


# ------------------------------------------------------------------ data panels
def panel_four(ax, D):
    xs = [0, 1, 2.4, 3.4]
    for j, c in enumerate(("R+", "R-")):
        for k, s_ in enumerate(("learning", "expert")):
            d = D[(D.cohort == c) & (D.stage == s_)]; x = np.array(xs[2 * k: 2 * k + 2]) + (j - 0.5) * 0.14
            for kind, col, ls in (("null", GREY, (0, (2, 1.5))), ("real", COH[c], "-")):
                v = d[[f"pre_W_{kind}", f"post_W_{kind}"]]
                ax.errorbar(x, v.mean(), v.sem(), fmt="o", ls=ls, ms=2.8, color=col, lw=1.0, capsize=0, mfc=col if s_ == "expert" else "white", mew=0.7)
            ax.text(xs[2 * k] + 0.5, 0.93 - 0.08 * j, f"n = {len(d)}", color=COH[c], fontsize=4.3, ha="center", transform=ax.get_xaxis_transform())
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(xs, ["pre", "post", "pre", "post"], fontsize=4.8)
    for x0, s_ in ((0.5, "day 0"), (2.9, "expert")):
        ax.text(x0, -0.2, s_, ha="center", fontsize=5, transform=ax.get_xaxis_transform())
    ax.set_ylabel("passive whisker projection\n(per neuron)"); ax.set_title("Passive whisker response on the\nhit − miss axis (grey: shifted axes)", fontsize=5.2)


def panel_tc(ax, T):
    for c in ("R+", "R-"):
        for st, ls in (("learning", (0, (2.5, 1.5))), ("expert", "-")):
            piv = T[(T.cohort == c) & (T.stage == st)].pivot_table(index="session_id", columns="t", values="ex"); mu, se = piv.mean(0), piv.sem(0)
            ax.fill_between(piv.columns, mu - se, mu + se, color=COH[c], alpha=0.13, lw=0)
            ax.plot(piv.columns, mu, color=COH[c], ls=ls, lw=0.9, label=f"{c.replace('-', '−')} {'day 0' if st == 'learning' else 'expert'} ({piv.shape[0]})")
    ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.axvline(0, color="0.4", lw=0.4, ls=":"); ax.axvspan(-100, 0, color="#FDD49E", alpha=0.4, lw=0, zorder=0)
    ax.set_xlabel("time from lick (ms)"); ax.set_ylabel("CCGP − shift null")
    ax.legend(frameon=False, fontsize=3.8, loc="upper left", handlelength=1.4)
    ax.set_title("Decoding: train AH vs SL-A, test WH vs SL-B\n(and reverse); real − linear-shift null", fontsize=5.2)


def panel_halves(ax, W):
    import statsmodels.formula.api as smf
    from scipy import stats
    d = W.dropna(subset=["PS_h1", "CCGP_h2"])
    for (c, s) in GROUPS:
        q = d[(d.cohort == c) & (d.stage == s)]
        ax.plot(q.PS_h1, q.CCGP_h2, "o", ms=2.3, color=COH[c], mfc=COH[c] if s == "expert" else "white", mew=0.6, ls="")
    mdl = smf.ols("CCGP_h2 ~ PS_h1", d).fit(); xx = np.linspace(d.PS_h1.min(), d.PS_h1.max(), 50)
    pr = mdl.get_prediction(pd.DataFrame({"PS_h1": xx})).summary_frame(alpha=0.05); rs, ps_ = stats.spearmanr(d.PS_h1, d.CCGP_h2)
    ax.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color="0.5", alpha=0.2, lw=0)
    ax.plot(xx, pr["mean"], color="0.2", lw=0.9, ls="-" if mdl.pvalues["PS_h1"] < 0.05 else "--")
    ax.set_xlabel("parallelism cos(WH − SL, AH − SL)\n(trial half 1)"); ax.set_ylabel("CCGP (trial half 2)")
    ax.set_title(f"Decoding tracks CD alignment\nSpearman r = {rs:.2f}, {m62.fmt_p(ps_)} (n = {len(d)})", fontsize=5.2)


def panel_areas(ax, Q, rng, gcol):
    S = m114.stats_table(Q, "ex", m62, rng).set_index("area")
    order = ["Whole brain"] + S.drop("Whole brain").sort_values("R+ expert", ascending=False).index.tolist()
    y = np.arange(len(order))[::-1]; off = {GROUPS[0]: 0.27, GROUPS[1]: 0.09, GROUPS[2]: -0.09, GROUPS[3]: -0.27}
    for i, a in enumerate(order):
        r = S.loc[a]
        for k in GROUPS:
            key = f"{k[0]} {k[1]}"
            if r[f"n {key}"] >= m114.MIN_SESS:
                ax.errorbar(r[key], y[i] + off[k], xerr=r[f"sem {key}"], fmt="o", ms=2.2, color=COH[k[0]], mfc=COH[k[0]] if k[1] == "expert" else "white",
                            lw=0.7, capsize=0, mew=0.5)
    ax.axvline(0, color="0.6", lw=0.5); ax.axhline(len(order) - 1.5, color="0.85", lw=0.5)
    ax.set_yticks(y, [m88.m83.m62_short(a) for a in order], fontsize=4.3)
    for tl, a in zip(ax.get_yticklabels(), order):
        tl.set_color(gcol.get(a, "k")); tl.set_weight("bold" if a == "Whole brain" else "normal")
    ax.set_xlabel(f"CCGP − shift null ({m114.N_SUB} units × {m114.R_SUB} draws)")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", color=COH[c], mfc=COH[c] if s_ == "expert" else "white", ms=2.5) for c, s_ in GROUPS],
              ["R+ D0", "R+ E", "R− D0", "R− E"], frameon=False, fontsize=3.6, loc="lower right", handletextpad=0.1, ncol=2, columnspacing=0.6)
    ax.set_title("Areas, ranked by R+ expert\n(no area survives BH correction)", fontsize=5.2)
    return S


def main():
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors(); plt = m62.setup(); rng = np.random.default_rng(0)
    d = load()
    with plt.rc_context({"font.size": 5.2, "axes.labelsize": 5.0, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(6.3, 10.6))
        gs = fig.add_gridspec(5, 3, hspace=0.95, wspace=0.75, left=0.1, right=0.97, top=0.95, bottom=0.04)
        A = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(5)]
        for row in A:
            for ax in row:
                ax.set_box_aspect(1)
        panel_task(A[0][0]); panel_hm_axis(A[0][1]); panel_cd(A[0][2])
        panel_four(A[1][0], d["hm"])
        m88.mean_panel2(A[1][1], d["hm"], "pre_W_ex", "projection, real − shifted", rng, "115 hm pre", "Passive pre (whisker)", ref=0)
        m88.mean_panel2(A[1][2], d["hm"], "post_W_ex", "projection, real − shifted", rng, "115 hm post", "Passive post (whisker)", ref=0)
        m88.mean_panel2(A[2][0], d["si"], "SI", "SI (−1 SL-like … +1 AH-like)", rng, "115 SI", "Single reward-lick neurons:\nWH response AH-like vs SL-like", ref=0)
        m88.mean_panel2(A[2][1], d["cd"], "d_WS", "d_WS = p_WH − p_SL (per neuron)", rng, "115 dWS", "WH position on the CD\n(100 ms before the lick)", ref=0)
        m88.mean_panel2(A[2][2], d["cdj"], "d_WS", "d_WS (per neuron)", rng, "115 dWS jaw", "Control: 100 ms before\njaw-opening onset", ref=0)
        m88.mean_panel2(A[3][0], d["cd"], "theta", "θ (°)", rng, "115 theta", "Angle θ of WH − SL to the CD", ref=90)
        panel_tc(A[3][1], d["tc"])
        m88.mean_panel2(A[3][2], d["cc"], "dir_AtoW_ex", "bal. acc. − shift null", rng, "115 AtoW", "Train AH vs SL → test WH vs SL\n(−100–0 ms)", ref=0)
        panel_halves(A[4][0], d["cc"])
        A[4][1].set_box_aspect(None); S = panel_areas(A[4][1], d["eq"], rng, gcol)
        m88.mean_panel2(A[4][2], d["kin"], "WH_AH", "jaw opening WH − AH (mm)", rng, "115 jaw", "Control: jaw opening WH − AH\n(−100–0 ms before the lick)", ref=0)
        for r, row in enumerate(A):
            m62.letter_row(fig, row, "abcdefghijklmno"[3 * r: 3 * r + 3], dx_in=0.3, dy_in=0.16)
        fig.suptitle("Learning a whisker → reward contingency routes the stimulus to a go state and recruits whisker licks into the auditory reward-lick code",
                     x=0.01, y=0.995, ha="left", fontsize=6.2, weight="bold")
        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"COSYNE_combined.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("115")]); St.to_csv(OUT / "COSYNE_combined_stats.csv", index=False)
    S.to_csv(OUT / "COSYNE_combined_areas.csv")
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning", "n_R− expert",
              "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU", "expert R+ vs R- p_Welch"]].round(4).to_string())
    print("ALL DONE", OUT / "COSYNE_combined.png")


if __name__ == "__main__":
    main()
