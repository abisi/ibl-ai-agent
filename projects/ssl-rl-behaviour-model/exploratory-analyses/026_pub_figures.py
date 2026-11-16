"""026 -- publication figures of ssl-rl-behaviour-model (user 2026-10-23: "update the figures, show more example predictions,
mouse-level, more figures that help understand, publication quality, then make the report").
Style: skills/ssl-figure-style (Arial 5-7.5 pt, 7.4 in width, aligned bold panel letters, edge-free bands, mean +- s.e.m. over mice,
auditory #2c2cdb, whisker #f7b519, no stimulus #262626, R+ #00B400, R- #C800C8). Mouse = unit for behaviour.
Inputs (each figure is skipped with a message if its inputs are missing):
  tables/011_sequences.parquet; tables/021_model_comparison.csv, fits/021_fits.csv, tables/021_latents (v5); fits/023*_fits.csv,
  tables/023*_latents (v6, v6b); tables/021b_g_profile.csv, tables/022_param_recovery.csv, tables/022_recovery_confusion.csv;
  tables/024_mouse_best.csv; fits/025_hier_mice.csv, tables/025_latents, tables/025_stats.csv, tables/025_recovery_summary.csv,
  fits/025_recovery_mice.csv; tables/018_boundaries.csv, 018_stats.csv
Output: figures/pub/fig{1..7}_*.{png,pdf,svg}, figS{1..4}_*; tables/026_figure_numbers.csv (numbers quoted by the report)
Run (haas): python 026_pub_figures.py [--model <v6 model shown as final>]
"""
from __future__ import annotations

import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

import common as C
import rl7_model as M7

OUT = C.FIG / "pub"
OUT.mkdir(exist_ok=True)
COH = {"R+": "#00B400", "R-": "#C800C8"}
TC = {"auditory": "#2c2cdb", "whisker": "#f7b519", "no_stim": "#262626"}
TLAB = {"auditory": "auditory", "whisker": "whisker", "no_stim": "no stimulus (false alarm)"}
DAYS = list(range(-4, 3))
V5 = "overlap-E3-eng-asym"
W = 7.4
NUM = []
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 6,
                     "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.titlepad": 4, "legend.frameon": False})


def num(key, value, note=""):
    NUM.append(dict(key=key, value=value, note=note))
    return value


def letter_row(fig, axes, letters, dx_in=0.32, dy_in=0.12):
    Wf, Hf = fig.get_size_inches()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, l in zip(axes, letters):
        fig.text(ax.get_position().x0 - dx_in / Wf, top + dy_in / Hf, l, fontsize=9, weight="bold", ha="left", va="bottom")


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("saved", name, flush=True)


def band(ax, x, m, s, color, lw=1.0, ls="-", label=None, alpha=0.25):
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, lw=0, edgecolor="none")
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)


def msem(df, by, col, unit="mouse_id"):
    """mean +- s.e.m. over units of the per-unit means"""
    u = df.groupby(list(by) + [unit])[col].mean().reset_index()
    g = u.groupby(list(by))[col]
    return pd.DataFrame({"m": g.mean(), "s": g.std(ddof=1) / np.sqrt(g.size()), "n": g.size()}).reset_index()


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def load_data():
    d = pd.read_parquet(C.TAB / "011_sequences.parquet")
    g = d.groupby("session_id", sort=False)
    d["pos"] = g.cumcount(); d["n_sess"] = g.session_id.transform("size"); d["toend"] = g.cumcount(ascending=False)
    d["water"] = ((d.trial_type == "auditory") | ((d.trial_type == "whisker") & (d.reward_group == "R+"))) & (d.lick_flag == 1)
    d["cumR"] = g.water.cumsum() - d.water; d["totR"] = g.water.transform("sum")
    d["fifth"] = np.floor(5 * d.pos / d.n_sess) / 5
    return d


def lat(path):
    L = pd.read_parquet(path)
    g = L.groupby("session_id", sort=False)
    return L.assign(pos=g.cumcount(), n_sess=g.session_id.transform("size"), toend=g.cumcount(ascending=False),
                    fifth=lambda e: np.floor(5 * e.pos / e.n_sess) / 5)


def fits_all():
    return pd.concat([pd.read_csv(f) for f in sorted(C.FITS.glob("023*_fits.csv"))], ignore_index=True).drop_duplicates(
        ["model", "cohort_set"], keep="last")


# ------------------------------------------------------------------ figure 1: behaviour
def fig1(d):
    fig = plt.figure(figsize=(W, 4.3))
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.98, top=0.92, bottom=0.1, hspace=0.6, wspace=0.45)
    axa = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
    e = d[d.day.isin(DAYS)]
    for ax, cs in zip(axa, COH):
        T = msem(e[e.reward_group == cs], ["trial_type", "day", "fifth"], "lick_flag")
        for t in TC:
            for dd, e2 in T[T.trial_type == t].groupby("day"):
                band(ax, dd + e2.fifth + 0.1, e2.m, e2.s, TC[t], label=TLAB[t] if dd == 0 else None)
        ax.axvline(0, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(np.array(DAYS) + 0.5, DAYS); ax.set_ylim(0, 1.02)
        ax.set_xlabel("training day (each day split in session fifths; 0 = first whisker day)")
        ax.set_title(f"{cs.replace('-', '−')} mice (n = {e[e.reward_group == cs].mouse_id.nunique()})", color=COH[cs])
    axa[0].set_ylabel("lick probability"); axa[0].legend(loc="lower left", fontsize=5)
    axb = [fig.add_subplot(gs[1, i]) for i in range(4)]
    w = d[d.day.isin(DAYS)]
    sess = w.groupby(["session_id", "reward_group"]).totR.first().reset_index()
    bins = np.arange(0, 260, 15)
    for cs in COH:
        axb[0].hist(sess[sess.reward_group == cs].totR, bins=bins, color=COH[cs], alpha=0.5, lw=0, edgecolor="none",
                    label=cs.replace("-", "−"))
    q = sess.totR.quantile([0.1, 0.5, 0.9])
    num("rewards_per_session_p10", q[0.1]); num("rewards_per_session_median", q[0.5]); num("rewards_per_session_p90", q[0.9])
    axb[0].set_xlabel("rewards per session"); axb[0].set_ylabel("sessions"); axb[0].legend()
    axb[0].set_title("Sessions end after very\ndifferent numbers of rewards")
    end_bins = [0, 10, 20, 40, 80, 160]
    mids = [5, 15, 30, 60, 120]
    w2 = w.assign(eb=pd.cut(w.toend, end_bins, right=False, labels=mids))
    w3 = w.assign(cb=pd.cut(w.cumR, [0, 20, 40, 60, 80, 100, 120, 150], right=False, labels=[10, 30, 50, 70, 90, 110, 135]))
    w4 = w.assign(fb=(np.floor(10 * w.cumR / w.totR.clip(lower=1)).clip(upper=9) + 0.5) / 10)
    for ax, col, xl, ttl, xs in [(axb[1], "eb", "trials before session end", "Aligned to session end", "log"),
                                  (axb[2], "cb", "rewards received so far", "Aligned to rewards received", "lin"),
                                  (axb[3], "fb", "fraction of the session's rewards", "Aligned to fraction of rewards", "lin")]:
        src = {"eb": w2, "cb": w3, "fb": w4}[col]
        for t in ("auditory", "no_stim"):
            for cs in COH:
                T = msem(src[(src.trial_type == t) & (src.reward_group == cs)].dropna(subset=[col]), [col], "lick_flag")
                band(ax, T[col].astype(float), T.m, T.s, TC[t], ls="-" if cs == "R+" else "--",
                     label=f"{TLAB[t].split(' (')[0]} {cs.replace('-', '−')}")
        if xs == "log":
            ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel(xl); ax.set_title(ttl); ax.set_ylim(0, 1)
    axb[1].set_ylabel("lick probability"); axb[1].legend(fontsize=4.8, loc="center left")
    fa = msem(w2[(w2.trial_type == "no_stim")].dropna(subset=["eb"]), ["eb"], "lick_flag").set_index("eb").m
    au = msem(w2[(w2.trial_type == "auditory")].dropna(subset=["eb"]), ["eb"], "lick_flag").set_index("eb").m
    num("fa_last10", fa.iloc[0]); num("fa_80_160_before_end", fa.iloc[-1]); num("aud_last10", au.iloc[0]); num("aud_80_160_before_end", au.iloc[-1])
    letter_row(fig, axa, "ab"); letter_row(fig, axb, "cdef")
    save(fig, "fig1_behaviour")


# ------------------------------------------------------------------ figure 2: model schematic and model ladder
def node(ax, xy, text, color, r=0.07):
    ax.add_patch(Circle(xy, r, facecolor=color, edgecolor="none", alpha=0.9))
    ax.text(*xy, text, ha="center", va="center", color="white", fontsize=6, weight="bold")


def arrow(ax, a, b, text="", color="0.2", ls="-", rad=0.0, fs=5.5, off=(0, 0.03)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=9, shrinkB=9))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=fs, color=color)


def ladder(F5, F6, final):
    """per cohort, full-data negative log-likelihood and BIC of the model sequence"""
    steps = [("v5 best (E3 exploration, engagement)", "v5", V5), ("v6: one thirst rate per cohort", "v6", "pop-lin-inst-noeng"),
             ("+ thirst capacity per session", "v6", "sess-lin-inst-noeng"), ("+ thirst threshold on lick bias", "v6", "sess-thr-inst-noeng"),
             ("+ learned false-alarm value", "v6", "sess-thr-inst-noeng+fa"), ("+ disengaged state", "v6", "sess-thr-inst-noeng+fa+dis"),
             ("+ thirst sensitivity of stimulus licking κ", "v6", "sess-thr-inst-noeng+fa+dis+kap"),
             ("+ session lick-propensity offset u_s", "v6", "sess-thr-inst-noeng+fa+dis+kap+off"),
             ("+ context step at whisker onset", "v6", "sess-thr-inst-noeng+fa+dis+kap+off+ctx"),
             ("+ water absorption lag", "v6", "sess-thr-abs-noeng+fa+dis+kap+off+ctx"),
             ("final", "v6", final)]
    rows = []
    for lab, src, m in steps:
        for cs in COH:
            if src == "v5":
                r = F5[(F5.model == m) & (F5.cohort_set == cs) & (F5.fold.astype(str) == "full")]
                if len(r):
                    rows.append(dict(step=lab, model=m, cohort=cs, nll=r.nll_train.iloc[0], k=r.k.iloc[0]))
            else:
                r = F6[(F6.model == m) & (F6.cohort_set == cs)]
                if len(r):
                    rows.append(dict(step=lab if lab != "final" else f"final: {m}", model=m, cohort=cs, nll=r.nll.iloc[0], k=r.k.iloc[0]))
    L = pd.DataFrame(rows).drop_duplicates(["model", "cohort"])
    n = pd.read_parquet(C.TAB / "011_sequences.parquet").reward_group.value_counts()
    L["bic"] = 2 * L.nll + L.k * np.log(L.cohort.map(n))
    return L


def fig2(F5, F6, final):
    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(2, 2, left=0.27, right=0.98, top=0.95, bottom=0.1, hspace=0.35, wspace=0.12, height_ratios=[1.15, 1])
    ax = fig.add_axes([0.03, 0.55, 0.5, 0.4]); ax.set_xlim(0, 1.35); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    axe = fig.add_axes([0.56, 0.55, 0.42, 0.4]); axe.axis("off")
    eqs = [r"value         $V = w \cdot x_s$,  $x_A=(1,g)$, $x_W=(g,1)$;  $V_N = w_N + g_N w_A$ ($w_N$ += $d_{ctx}$ at whisker onset)",
           r"learning     $\delta = o - V$,  $w \leftarrow w + \alpha_\pm\, \delta\, x_s$;  false alarm: $w_N \leftarrow w_N + \alpha_N(-1 - V_N)$",
           r"thirst         $T = \max(0,\, 1-\rho_s S)$, $S$: water absorbed ($\tau_a$);  $\log\rho_s \sim \mathcal{N}(\mu_\rho, \sigma_\rho^2)$",
           r"lick bias    $B(T) = s_B\, \mathrm{softplus}(20(T-\theta))/20$",
           r"engaged     $P(\mathrm{lick}) = 0.01 + 0.98\, \sigma(\beta T^\kappa V - c + B(T) + u_s)$,  $u_s \sim \mathcal{N}(0, \sigma_u^2)$",
           r"disengaged $P(\mathrm{lick}) = 0.01$;  leave $\sigma(h_0 + h_1 (1-T))$, return $\sigma(h_{back})$",
           r"(no stimulus: $V = V_N$; $\rho_s$, $u_s$ integrated out, engaged state filtered per trial)"]
    for i, s in enumerate(eqs):
        axe.text(0, 0.92 - 0.135 * i, s, fontsize=5.3, va="top")
    node(ax, (0.12, 0.8), "A", TC["auditory"]); node(ax, (0.12, 0.5), "W", "#c48a00"); node(ax, (0.12, 0.2), "N", TC["no_stim"])
    node(ax, (0.62, 0.5), "lick", "0.45", r=0.085)
    arrow(ax, (0.12, 0.8), (0.62, 0.5), "w_A", TC["auditory"], off=(0, 0.04))
    arrow(ax, (0.12, 0.5), (0.62, 0.5), "w_W", "#c48a00", off=(-0.05, 0.01))
    arrow(ax, (0.12, 0.2), (0.62, 0.5), "w_N", TC["no_stim"], off=(0, -0.11))
    arrow(ax, (0.12, 0.8), (0.12, 0.5), "", "0.5", ls=":", rad=0.5)
    ax.text(0.015, 0.65, "g", fontsize=6, style="italic", color="0.4")
    ax.add_patch(FancyBboxPatch((0.86, 0.66), 0.42, 0.26, boxstyle="round,pad=0.01", facecolor="#e8f1fb", edgecolor="none"))
    ax.text(1.07, 0.79, "thirst T = 1 − ρ_s · water absorbed\nρ_s and offset u_s per session\nlick bias B(T): threshold θ", ha="center", va="center", fontsize=5.2)
    ax.add_patch(FancyBboxPatch((0.86, 0.1), 0.42, 0.26, boxstyle="round,pad=0.01", facecolor="#f3ece0", edgecolor="none"))
    ax.text(1.07, 0.23, "engaged ↔ disengaged\nleaving rises with satiety,\nre-engagement possible\n(disengaged: no licking)", ha="center", va="center", fontsize=5.2)
    arrow(ax, (0.9, 0.66), (0.68, 0.55), "", "#3b78b5")
    arrow(ax, (0.9, 0.36), (0.68, 0.45), "", "#a5743a")
    ax.text(0.37, 0.95, "δ = outcome − V; w += α± · δ · x", ha="center", fontsize=5.5)
    ax.set_title("Model: stimulus → lick weights, thirst and engagement", loc="left")
    L = ladder(F5, F6, final)
    L.to_csv(C.TAB / "026_model_ladder.csv", index=False)
    axb = fig.add_subplot(gs[1, 0]); axc = fig.add_subplot(gs[1, 1])
    order = list(dict.fromkeys(L.step))
    for ax_, q, lab in [(axb, "nll", "−log-likelihood relative to v5 best\n(nats; lower = better)"),
                        (axc, "bic", "ΔBIC relative to v5 best (lower = better)")]:
        for j, cs in enumerate(COH):
            e = L[L.cohort == cs].set_index("step").reindex(order)
            ref = e[q].iloc[0]
            y = np.arange(len(order))
            ax_.barh(y + (j - 0.5) * 0.38, e[q] - ref, height=0.36, color=COH[cs], lw=0, label=cs.replace("-", "−"))
        ax_.axvline(0, color="0.5", lw=0.5)
        ax_.set_yticks(np.arange(len(order)), order if ax_ is axb else [""] * len(order), fontsize=5.2); ax_.invert_yaxis()
        ax_.set_xlabel(lab)
    axc.legend(loc="lower right")
    axb.set_title("Adding one component at a time (cohort-level fits on all trials)", loc="left")
    fig.text(0.01, 0.97, "a", fontsize=9, weight="bold"); fig.text(0.54, 0.97, "b", fontsize=9, weight="bold")
    letter_row(fig, [axb, axc], "cd", dx_in=0.15)
    save(fig, "fig2_model")
    return L


# ------------------------------------------------------------------ figure 3: fit vs data (cohort level)
def curve_panel(ax, L, col, cs, lw_model=0.9, ls="-", data=True, types=TC):
    e = L[(L.reward_group == cs) & L.day.isin(DAYS)]
    for t in types:
        for dd, e2 in e[e.trial_type == t].groupby("day"):
            if data:
                T = msem(e2, ["fifth"], "lick_flag")
                ax.fill_between(dd + T.fifth + 0.1, T.m - T.s, T.m + T.s, color=TC[t], alpha=0.22, lw=0, edgecolor="none")
            T2 = e2.groupby(["mouse_id", "fifth"])[col].mean().groupby("fifth").mean()
            ax.plot(dd + T2.index + 0.1, T2.values, color=TC[t], lw=lw_model, ls=ls)


def end_panel(ax, L, cs, col, ls="-", data=True, types=("auditory", "no_stim", "whisker")):
    e = L[(L.reward_group == cs) & L.day.isin(DAYS) & (L.toend < 160)]
    e = e.assign(eb=pd.cut(e.toend, [0, 10, 20, 40, 80, 160], right=False, labels=[5, 15, 30, 60, 120]).astype(float))
    for t in types:
        x = e[e.trial_type == t]
        if data:
            T = msem(x, ["eb"], "lick_flag")
            ax.fill_between(T.eb, T.m - T.s, T.m + T.s, color=TC[t], alpha=0.22, lw=0, edgecolor="none")
        T2 = x.groupby(["mouse_id", "eb"])[col].mean().groupby("eb").mean()
        ax.plot(T2.index, T2.values, color=TC[t], lw=0.9, ls=ls)
    ax.set_xscale("log"); ax.invert_xaxis()


def fig3(models):
    """models: list of (label, latents DataFrame)"""
    fig = plt.figure(figsize=(W, 1.45 * len(models) + 0.5))
    gs = fig.add_gridspec(len(models), 3, left=0.07, right=0.98, top=0.93, bottom=0.08, hspace=0.75, wspace=0.28,
                          width_ratios=[2.4, 2.4, 1.1])
    rows = []
    for i, (lab, L) in enumerate(models):
        axs = [fig.add_subplot(gs[i, j]) for j in range(3)]
        for ax, cs in zip(axs[:2], COH):
            curve_panel(ax, L, "p_lick", cs)
            ax.axvline(0, color="0.6", lw=0.5, ls="--"); ax.set_ylim(0, 1.02)
            ax.set_xticks(np.array(DAYS) + 0.5, DAYS if i == len(models) - 1 else [])
            ax.set_title(f"{cs.replace('-', '−')} — {lab}", loc="left", fontsize=6)
        for cs, ls in zip(COH, ("-", "--")):
            end_panel(axs[2], L, cs, "p_lick", ls=ls)
        axs[2].set_ylim(0, 1); axs[2].set_title("session end (R+ solid, R− dashed)", fontsize=5.5)
        axs[0].set_ylabel("lick probability")
        if i == len(models) - 1:
            axs[0].set_xlabel("training day (session fifths)"); axs[1].set_xlabel("training day (session fifths)")
            axs[2].set_xlabel("trials before session end")
        rows.append(axs)
        # binned error: mean |data - model| over mouse x day x fifth x type cells
        e = L[L.day.isin(DAYS)].groupby(["reward_group", "mouse_id", "day", "fifth", "trial_type"])[["lick_flag", "p_lick"]].mean()
        num(f"fig3_mae_{lab}", float((e.lick_flag - e.p_lick).abs().mean()), "mean abs error, mouse x day x fifth x trial type")
    letter_row(fig, [r[0] for r in rows[:1]] + rows[0][1:], "abc")
    fig.suptitle("Data (bands: mean ± s.e.m. over mice) and model one-step-ahead predictions (lines); blue auditory, yellow whisker, "
                 "black no stimulus", y=0.995, fontsize=6)
    save(fig, "fig3_fit_vs_data")


def start_errors(L):
    """session start (first fifth): per mouse x day x trial type mean data and model, then mean over mice; per session FA"""
    s = L[L.day.isin(DAYS) & (L.fifth == 0)]
    m = s.groupby(["reward_group", "day", "trial_type", "mouse_id"])[["lick_flag", "p_lick"]].mean()
    m = m.assign(err=m.p_lick - m.lick_flag).groupby(["reward_group", "day", "trial_type"]).err.agg(["mean", "std", "size"])
    fa = s[s.trial_type == "no_stim"].groupby(["session_id", "reward_group"])[["lick_flag", "p_lick"]].mean().reset_index()
    return m.reset_index(), fa


def fig3b(models):
    """errors the user flagged (2026-10-23): auditory under-prediction on whisker days, session-start false alarms"""
    fig = plt.figure(figsize=(W, 4.0))
    gs = fig.add_gridspec(2, 4, left=0.08, right=0.98, top=0.9, bottom=0.11, hspace=0.6, wspace=0.45)
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(models)))
    axs = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)]
    for k, (lab, L) in enumerate(models):
        m, fa = start_errors(L)
        for i, t in enumerate(("auditory", "no_stim")):
            for j, cs in enumerate(COH):
                ax = axs[i][j]
                e = m[(m.trial_type == t) & (m.reward_group == cs)]
                ax.errorbar(e.day + (k - len(models) / 2) * 0.08, e["mean"], e["std"] / np.sqrt(e["size"]), fmt="o-", ms=2.5, lw=0.8,
                            color=cols[k], label=lab if (i, j) == (0, 0) else None)
                ax.axhline(0, color="0.6", lw=0.5); ax.axvline(-0.5, color="0.6", lw=0.5, ls="--")
                ax.set_title(f"{cs.replace('-', '−')} {TLAB[t].split(' (')[0]}: session start", fontsize=6)
                if j == 0:
                    ax.set_ylabel("model − data lick prob.\n(first fifth; mean ± s.e.m. over mice)")
                ax.set_xlabel("training day")
                for d_ in (1, 2):
                    r = e[e.day == d_]
                    if len(r):
                        num(f"fig3b_start_err_{t}_{cs}_day{d_}_{lab}", float(r["mean"].iloc[0]))
        for j, cs in enumerate(COH):
            ax = axs[j][2]
            e = fa[fa.reward_group == cs]
            r = stats.pearsonr(e.lick_flag, e.p_lick)[0]
            ax.scatter(e.lick_flag, e.p_lick, s=2, color=cols[k], lw=0, alpha=0.35)
            ax.text(0.02, 0.95 - 0.08 * k, f"r = {r:.2f}", color=cols[k], transform=ax.transAxes, fontsize=5, va="top")
            num(f"fig3b_fa_start_r_{cs}_{lab}", r)
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("data: session-start FA rate"); ax.set_ylabel("model"); ax.set_title(f"{cs.replace('-', '−')}: per session", fontsize=6)
            sd = axs[j][3]
            sd.bar(k, e.p_lick.std(), color=cols[k], lw=0)
            sd.axhline(e.lick_flag.std(), color="k", lw=0.8, ls="--")
            sd.set_title(f"{cs.replace('-', '−')}: spread across sessions\n(dashed: data)", fontsize=6); sd.set_ylabel("s.d. of start FA rate")
            sd.set_xticks([])
    axs[0][0].legend(fontsize=4.5, loc="lower left")
    letter_row(fig, axs[0], "abcd"); letter_row(fig, axs[1], "efgh")
    save(fig, "fig3b_start_errors")


# ------------------------------------------------------------------ figure 4: example mice
def pick_examples(L, best=None):
    d0 = L[L.day == 0].copy(); d0["i"] = d0.groupby(["mouse_id", "trial_type"]).cumcount()
    e = d0[d0.i < 30].groupby(["mouse_id", "trial_type"]).lick_flag.mean().unstack()
    late = L[L.day >= 1].groupby(["mouse_id", "trial_type"]).lick_flag.mean().unstack()
    coh = L.groupby("mouse_id").reward_group.first()
    nd = L.groupby("mouse_id").day.nunique()
    ok = nd[nd >= 6].index
    ex = {}
    rp = late.loc[late.index.isin(ok) & (coh.reindex(late.index) == "R+")].whisker
    ex["R+ strong whisker learner"] = rp.idxmax() if len(rp) else None
    ex["R+ weak whisker learner"] = rp.idxmin() if len(rp) else None
    rm = (e.whisker - e.no_stim)[e.index.isin(ok) & (coh.reindex(e.index) == "R-")]
    ex["R− day-0 whisker excess"] = rm.idxmax() if len(rm) else None
    ex["R− no day-0 excess"] = (rm - rm.median()).abs().idxmin() if len(rm) else None
    return {k: v for k, v in ex.items() if v is not None}


def roll(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=max(2, w // 3)).mean().to_numpy()


def fig4(L, title_model, w=15):
    ex = pick_examples(L)
    fig = plt.figure(figsize=(W, 1.95 * len(ex) + 0.4))
    outer = fig.add_gridspec(len(ex), 1, left=0.07, right=0.985, top=0.95, bottom=0.05, hspace=0.42)
    first = []
    for i, (lab, m) in enumerate(ex.items()):
        e = L[(L.mouse_id == m) & (L.day >= -3)].reset_index(drop=True)
        inner = outer[i].subgridspec(2, 1, hspace=0.08, height_ratios=[2.2, 1])
        ax = fig.add_subplot(inner[0]); axw = fig.add_subplot(inner[1], sharex=ax)
        x = np.arange(len(e))
        for t in TC:
            idx = np.flatnonzero(e.trial_type == t)
            ax.plot(x[idx], roll(e.lick_flag.to_numpy()[idx], w), color=TC[t], lw=1.6, alpha=0.35)
            ax.plot(x[idx], roll(e.p_lick.to_numpy()[idx], w), color=TC[t], lw=0.8)
        if "p_engaged" in e:
            ax.plot(x, e.p_engaged, color="#a5743a", lw=0.5, ls=":")
        for b in np.flatnonzero(np.r_[1, np.diff(e.day.to_numpy())]):
            for a_ in (ax, axw):
                a_.axvline(b, color="0.75", lw=0.4)
            ax.text(b + 5, 1.04, f"day {e.day.iloc[b]}", fontsize=5, color="0.4")
        ax.set_ylim(0, 1.12); ax.set_ylabel("lick prob."); ax.tick_params(labelbottom=False)
        ax.set_title(f"{lab}: {m} ({e.reward_group.iloc[0].replace('-', '−')})", loc="left", fontsize=6, color=COH[e.reward_group.iloc[0]])
        for q, col, lb in [("V_A", TC["auditory"], "V(auditory)"), ("V_W", "#c48a00", "V(whisker)"), ("V_N", TC["no_stim"], "V(no stim.)")]:
            if q in e:
                axw.plot(x, e[q], color=col, lw=0.9, label=lb)
        axw.axhline(0, color="0.8", lw=0.4); axw.set_ylabel("value"); axw.set_ylim(-1.1, 1.1)
        if i == 0:
            axw.legend(ncol=3, loc="lower right", fontsize=5)
        first.append(ax)
    axw.set_xlabel("trial (all trials of the mouse from day −3; thin vertical lines = session starts)")
    fig.suptitle(f"Example mice: data (thick, running mean of {w} trials of each type) and model one-step-ahead prediction (thin), "
                 f"{title_model}; dotted brown = P(engaged)", y=0.995, fontsize=6)
    letter_row(fig, first[:1], "a")
    for ax, l in zip(first[1:], "bcd"):
        fig.text(ax.get_position().x0 - 0.32 / W, ax.get_position().y1 + 0.06 / fig.get_size_inches()[1], l, fontsize=9, weight="bold")
    save(fig, "fig4_example_mice")
    return ex


def fig4b(L, ex, title_model):
    """single sessions zoom: day-0 session of each example mouse with thirst and engagement"""
    fig, axs = plt.subplots(2, len(ex), figsize=(W, 2.9), sharey="row", gridspec_kw=dict(hspace=0.35, wspace=0.12, height_ratios=[2, 1]))
    for j, (lab, m) in enumerate(ex.items()):
        e = L[(L.mouse_id == m) & (L.day == 0)].reset_index(drop=True)
        ax = axs[0, j]; x = np.arange(len(e))
        for k, t in enumerate(TC):
            idx = np.flatnonzero(e.trial_type == t)
            lk = idx[e.lick_flag.to_numpy()[idx] == 1]
            ax.scatter(lk, np.full(len(lk), 1.06 + 0.05 * k), marker="|", s=6, color=TC[t], lw=0.6)
            ax.plot(x[idx], e.p_lick.to_numpy()[idx], color=TC[t], lw=0.8)
        ax.set_ylim(0, 1.22); ax.set_title(f"{m} day 0", fontsize=6, color=COH[e.reward_group.iloc[0]])
        a2 = axs[1, j]
        T = np.clip(1 - np.exp(e.lrho_smooth) * e.S, 0, 1)
        a2.plot(x, T, color="#3b78b5", lw=0.9, label="thirst T")
        if "p_engaged" in e:
            a2.plot(x, e.p_engaged, color="#a5743a", lw=0.9, label="P(engaged)")
        a2.set_ylim(0, 1.05); a2.set_xlabel("trial in session")
    axs[0, 0].set_ylabel("lick prob. (ticks: licks)"); axs[1, 0].set_ylabel("state"); axs[1, 0].legend(fontsize=5)
    fig.suptitle(f"Day-0 sessions of the example mice ({title_model}): licks (ticks) and one-step-ahead lick probability per trial type; "
                 "thirst at the session's posterior thirst rate", y=1.0, fontsize=6)
    letter_row(fig, list(axs[0]), "abcd")
    save(fig, "fig4b_example_sessions")


# ------------------------------------------------------------------ figure 5: per-mouse model preference (v5) and behaviour
def fig5(B, d):
    fig = plt.figure(figsize=(W, 2.3))
    gs = fig.add_gridspec(1, 4, left=0.07, right=0.98, top=0.85, bottom=0.25, wspace=0.55)
    facs = [("FA", "false-alarm term"), ("engagement", "engagement"), ("learning rates", "α+ vs α−")]
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax, (fac, lab) in zip(axs, facs):
        lev = sorted(B[f"best_{fac}"].unique())
        for i, cs in enumerate(COH):
            e = B[B.reward_group == cs]
            for j, l in enumerate(lev):
                y = e[e[f"best_{fac}"] == l][f"margin_{fac}"].clip(lower=0.1)
                x = j + (i - 0.5) * 0.32 + np.random.default_rng(j + 10 * i).uniform(-0.08, 0.08, len(y))
                ax.scatter(x, y, s=5, color=COH[cs], lw=0, alpha=0.85, label=cs.replace("-", "−") if j == 0 else None)
        ax.axhline(10, color="0.6", lw=0.5, ls=":")
        ax.set_yscale("log"); ax.set_xticks(range(len(lev)), lev, rotation=30); ax.set_title(f"best {lab} per mouse", fontsize=6)
    axs[0].set_ylabel("held-out margin over the\nnext level (nats, log scale)"); axs[0].legend(loc="upper left")
    d0 = d[d.day == 0].copy(); d0["i"] = d0.groupby(["mouse_id", "trial_type"]).cumcount()
    ex = d0[d0.i < 30].groupby(["mouse_id", "trial_type"]).lick_flag.mean().unstack()
    r = B[B.reward_group == "R-"].merge((ex.whisker - ex.no_stim).rename("excess"), left_on="mouse_id", right_index=True)
    r["grp"] = np.where(r.best_FA == "E3", "E3 (reward-rate)", "E0–E2 (allow fast\nwhisker extinction)")
    ax = axs[3]
    for j, (g, e) in enumerate(r.groupby("grp")):
        ax.scatter(j + np.random.default_rng(j).uniform(-0.12, 0.12, len(e)), e.excess, s=6, color=COH["R-"], lw=0, alpha=0.8)
        ax.errorbar(j + 0.28, e.excess.mean(), e.excess.std(ddof=1) / np.sqrt(len(e)), fmt="o", ms=3, color="k", lw=0.8)
        num(f"fig5_rminus_excess_{'E3' if g.startswith('E3') else 'E0-2'}_mean", e.excess.mean(), f"n = {len(e)}")
    a, b = [e.excess for _, e in r.groupby("grp")]
    p1, p2 = stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue
    num("fig5_excess_p_mwu", p1); num("fig5_excess_p_welch", p2)
    ax.set_xticks([0, 1], sorted(r.grp.unique()), fontsize=5); ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel("day-0 whisker − no-stim. lick\nprob. (first 30 trials each)")
    ax.set_title(f"R− mice by preferred FA term\nMWU {fmt_p(p1)}, Welch {fmt_p(p2)}", fontsize=6)
    letter_row(fig, axs, "abcd")
    save(fig, "fig5_per_mouse_models")


# ------------------------------------------------------------------ figure 6: auditory -> whisker generalisation
def fig6(G, P):
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw=dict(wspace=0.45, left=0.08, right=0.98, top=0.82, bottom=0.22))
    for ax, var in zip(axs[:2], ("init", "overlap")):
        for cs in COH:
            e = G[(G.cohort == cs) & G.model.str.startswith(var)].sort_values("g")
            ax.plot(e.g, e.dnll, "o-", color=COH[cs], ms=2.5, lw=0.9, label=cs.replace("-", "−"))
        ax.set_xlabel("transfer g (fixed)"); ax.set_title({"init": "copy at whisker onset: w_W ← g · w_A",
                                                           "overlap": "overlapping inputs: A = (1, g), W = (g, 1)"}[var], fontsize=5.8)
        ax.set_yscale("symlog", linthresh=10); ax.axhline(0, color="0.6", lw=0.5)
    axs[0].set_ylabel("Δ −log-likelihood vs best g\n(nats, refit all other parameters)"); axs[0].legend()
    e = P[(P.param == "g")]
    for cs in COH:
        x = e[e.cohort == cs]
        axs[2].scatter(x.true, x.refit, s=12, color=COH[cs], lw=0, alpha=0.7,
                       label=cs.replace("-", "−"))
    axs[2].plot([0, 0.6], [0, 0.6], color="0.6", lw=0.5, ls="--")
    axs[2].set_xlabel("g used to simulate"); axs[2].set_ylabel("g recovered"); axs[2].set_title("Recovery of g (simulated data)", fontsize=6)
    for _, r in G[G.g == 0.05].iterrows():
        num(f"fig6_dnll_g005_{r.model.split('-')[0]}_{r.cohort}", r.dnll)
    letter_row(fig, axs, "abc")
    save(fig, "fig6_generalisation")


# ------------------------------------------------------------------ figure 7: hierarchical mouse-level parameters and weights
PLAB = {"alpha_pos": "α+ (learning from reward)", "alpha_neg": "α− (learning from no reward)", "beta": "β (value gain)",
        "c": "c (lick threshold)", "s_B": "s_B (thirst lick bias)", "mu_rho": "μ_ρ (log thirst rate)", "theta": "θ (bias threshold)",
        "log_beta_alpha_pos": "log(β·α+) (scale-free\nlearning speed)", "w_N0": "w_N0 (initial FA value)", "alpha_N": "α_N (FA learning)",
        "g_N": "g_N (auditory → no stim.)", "h0": "h0", "h1": "h1"}


def fig7(R, S, L):
    params = [p for p in ["log_beta_alpha_pos", "alpha_pos", "alpha_neg", "beta", "c", "s_B", "mu_rho", "theta", "alpha_N"] if p in set(S.param)]
    fig = plt.figure(figsize=(W, 4.4))
    gs = fig.add_gridspec(2, 1, left=0.07, right=0.98, top=0.86, bottom=0.09, hspace=0.55, height_ratios=[1, 1.15])
    gt = gs[0].subgridspec(1, len(params), wspace=0.9)
    axt = [fig.add_subplot(gt[i]) for i in range(len(params))]
    R = R.assign(y_log_beta_alpha_pos=np.log(R.p_beta * R.p_alpha_pos))
    for ax, p in zip(axt, params):
        for j, cs in enumerate(COH):
            y = R[R.cohort_set == cs][f"y_{p}"]
            ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(y)), y, s=4, color=COH[cs], lw=0, alpha=0.6)
            ax.errorbar(j + 0.32, y.mean(), y.std(ddof=1) / np.sqrt(len(y)), fmt="o", ms=3, color=COH[cs], lw=0.9)
        s = S[S.param == p].iloc[0]
        ax.set_title(f"{PLAB.get(p, p)}\nMWU {fmt_p(s.p_mwu)}\nWelch {fmt_p(s.p_welch)}", fontsize=5)
        ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_xlim(-0.5, 1.6)
        num(f"fig7_{p}_p_mwu", s.p_mwu); num(f"fig7_{p}_p_welch", s.p_welch)
    axt[0].set_ylabel("per-mouse estimate\n(unbounded scale)")
    gb = gs[1].subgridspec(1, 3, wspace=0.35)
    axb = [fig.add_subplot(gb[i]) for i in range(3)]
    e = L[L.day.isin(DAYS)]
    for ax, q, lab in zip(axb, ["V_A", "V_W", "V_N"], ["auditory value V(A)", "whisker value V(W)", "no-stimulus value V(N)"]):
        if q not in e:
            ax.axis("off"); continue
        for cs in COH:
            per = e[e.reward_group == cs].groupby(["mouse_id", "day"])[q].mean().reset_index()
            for _, pm in per.groupby("mouse_id"):
                ax.plot(pm.day, pm[q], color=COH[cs], lw=0.3, alpha=0.25)
            T = per.groupby("day")[q].agg(["mean", "std", "size"])
            band(ax, T.index, T["mean"], T["std"] / np.sqrt(T["size"]), COH[cs], lw=1.4, label=cs.replace("-", "−"))
        ax.axvline(-0.5, color="0.6", lw=0.5, ls="--"); ax.axhline(0, color="0.8", lw=0.4)
        ax.set_xlabel("training day"); ax.set_title(lab, fontsize=6)
    axb[0].set_ylabel("learned value (per mouse: thin;\nmean ± s.e.m.: thick)"); axb[0].legend()
    letter_row(fig, axt[:1], "a"); letter_row(fig, axb, "bcd")
    save(fig, "fig7_mouse_level")


# ------------------------------------------------------------------ supplementary
def figS1(S5):
    import rl6_model as M6
    ref = S5[S5.model == "init-E0-noeng-sym"].iloc[0]
    S5 = S5.assign(d_ll=S5.ll_test_per_trial - ref.ll_test_per_trial)
    fig, ax = plt.subplots(figsize=(3.6, 2.3), gridspec_kw=dict(left=0.17, right=0.97, top=0.85, bottom=0.2))
    for i, fa in enumerate(M6.FA_MODES):
        for j, (tr, en, lr) in enumerate([(tr, en, lr) for tr in M6.TRANSFER for en in ("noeng", "eng") for lr in ("sym", "asym")]):
            r = S5[S5.model == f"{tr}-{fa}-{en}-{lr}"]
            ax.scatter(i + (j - 3.5) * 0.08, r.d_ll.iloc[0], s=9, marker="o" if tr == "init" else "^", lw=0,
                       color="#a5743a" if en == "eng" else "0.55", alpha=1 if lr == "asym" else 0.5,
                       label=f"{tr}, {'engagement' if en == 'eng' else 'no engagement'}, {lr}" if i == 0 else None)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xticks(range(4), ["E0 constant", "E1 decay", "E2 FA-driven", "E3 reward rate"])
    ax.set_ylabel("held-out log-likelihood per trial\nvs init-E0-noeng-sym (5-fold over mice)")
    ax.legend(fontsize=4.3, ncol=2, loc="lower right")
    ax.set_title("v5 model comparison (cohort-level parameters)", fontsize=6)
    save(fig, "figS1_v5_comparison")


def figS2(Rsum, Rm):
    a = Rm[Rm.scenario == "map"]
    params = [c[2:-5] for c in a if c.startswith("y_") and c.endswith("_true")]
    fig, axs = plt.subplots(1, len(params) + 1, figsize=(W, 1.7), gridspec_kw=dict(wspace=0.55, left=0.06, right=0.98, top=0.78, bottom=0.25))
    for ax, p in zip(axs, params):
        for cs in COH:
            e = a[a.cohort_set == cs]
            ax.scatter(e[f"y_{p}_true"], e[f"y_{p}_refit"], s=3, color=COH[cs], lw=0, alpha=0.7)
        lo, hi = a[[f"y_{p}_true", f"y_{p}_refit"]].min().min(), a[[f"y_{p}_true", f"y_{p}_refit"]].max().max()
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--")
        r = stats.spearmanr(a[f"y_{p}_true"], a[f"y_{p}_refit"])[0]
        ax.set_title(f"{PLAB.get(p, p).split(' (')[0]}\nρ = {r:.2f}", fontsize=5.5); ax.set_xlabel("true", fontsize=5)
        num(f"figS2_spearman_{p}", r)
    axs[0].set_ylabel("recovered")
    b = Rm[Rm.scenario == "mixed_rminus"]
    ax = axs[-1]
    for j, (f, e) in enumerate(b.groupby("fast")):
        ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(e)), e.p_alpha_neg, s=5, color=COH["R-"], lw=0)
    ax.set_yscale("log"); ax.set_xticks([0, 1], ["slow\n(α− = 0.001)", "fast\n(α− = 0.5)"], fontsize=5)
    auc = Rsum[Rsum.scenario == "mixed_rminus"].auc_fast_vs_slow.iloc[0]
    num("figS2_mixed_auc", auc)
    ax.set_title(f"Mixed R− cohort:\nrecovered α−, AUC = {auc:.2f}", fontsize=5.5)
    save(fig, "figS2_hier_recovery")


def figS3(conf, P5):
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw=dict(wspace=0.4, left=0.3, right=0.98, top=0.85, bottom=0.25))
    c = conf.copy()
    axs[0].barh(range(len(c)), c.dbic_runner_up, color=["0.3" if a == b.split(" (")[0] else "#d62728" for a, b in zip(c.bic_winner, c.generator)],
                lw=0)
    axs[0].set_yticks(range(len(c)), [f"{g} → {w}" for g, w in zip(c.generator, c.bic_winner)], fontsize=4.6)
    axs[0].set_xscale("symlog", linthresh=1); axs[0].set_xlabel("ΔBIC of runner-up (higher = clearer)")
    axs[0].set_title("v5 model recovery: generator → BIC winner (red: other model)", fontsize=5.5)
    e = P5[P5.param.isin(["alpha_pos", "c", "rho", "eps_inf", "eta", "e_min", "r_e", "g"])]
    for cs in COH:
        x = e[e.cohort == cs]
        axs[1].scatter(x.true, x.refit, s=5, color=COH[cs], lw=0, alpha=0.8, label=cs.replace("-", "−"))
    axs[1].set_xscale("symlog", linthresh=0.01); axs[1].set_yscale("symlog", linthresh=0.01)
    axs[1].plot([0, 1], [0, 1], color="0.6", lw=0.5, ls="--"); axs[1].set_xlabel("true"); axs[1].set_ylabel("recovered")
    axs[1].set_title("v5 parameter recovery (all generators)", fontsize=5.5); axs[1].legend()
    save(fig, "figS3_v5_recovery")


def figS4(Bd, Bs):
    q = [("within_whisker", "whisker, within session"), ("offline_whisker", "whisker, overnight"), ("across_whisker", "whisker, start to start"),
         ("offline_no_stim", "no stimulus, overnight")]
    q = [x for x in q if x[0] in set(Bs.quantity)]
    fig, ax = plt.subplots(figsize=(3.6, 2.2), gridspec_kw=dict(left=0.15, right=0.97, top=0.85, bottom=0.3))
    for i, (k, lab) in enumerate(q):
        for j, cs in enumerate(COH):
            r = Bs[(Bs.quantity == k) & (Bs.cohort == cs)]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(i + (j - 0.5) * 0.25, r["mean"], yerr=np.nan, fmt="o", ms=3.5, color=COH[cs], label=cs.replace("-", "−") if i == 0 else None)
    ax.axhline(0, color="0.6", lw=0.5); ax.set_xticks(range(len(q)), [x[1] for x in q], rotation=25, fontsize=5)
    ax.set_ylabel("change in lick probability\n(mean over mice)"); ax.legend()
    ax.set_title("Session boundaries (whisker days; 018)", fontsize=6)
    save(fig, "figS4_session_boundaries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="v6 model shown as final (default: lowest BIC summed over cohorts among 023* fits)")
    a = ap.parse_args()
    d = load_data()
    fig1(d)
    F5 = pd.read_csv(C.FITS / "021_fits.csv")
    F6 = fits_all()
    tot = F6.groupby("model").bic.sum()
    final = a.model or tot.idxmin()
    num("final_model", final)
    L = fig2(F5, F6, final)
    print(L.to_string(), flush=True)
    lats = [(f"v5 best ({V5})", lat(C.TAB / "021_latents" / f"{V5}.parquet"))]
    for m in ["sess-thr-inst-noeng", final]:
        for p in sorted(C.TAB.glob(f"023*_latents/{m}.parquet")):
            lats.append((f"v6 {m}", lat(p)))
    hier = C.TAB / "025_latents" / f"{final}.parquet"
    if hier.exists():
        lats.append((f"v7 mouse-level {final}", lat(hier)))
    lats = list(dict(lats).items())
    fig3(lats)
    fig3b(lats)
    Lex = lats[-1][1]
    ex = fig4(Lex, lats[-1][0])
    fig4b(Lex, ex, lats[-1][0])
    num("example_mice", "; ".join(f"{k}: {v}" for k, v in ex.items()))
    fig5(pd.read_csv(C.TAB / "024_mouse_best.csv"), d)
    fig6(pd.read_csv(C.TAB / "021b_g_profile.csv"), pd.read_csv(C.TAB / "022_param_recovery.csv"))
    if (C.FITS / "025_hier_mice.csv").exists():
        R = pd.read_csv(C.FITS / "025_hier_mice.csv")
        if final in set(R.model) and hier.exists():
            fig7(R[R.model == final], pd.read_csv(C.TAB / "025_stats.csv").query("model == @final"), lat(hier))
    figS1(pd.read_csv(C.TAB / "021_model_comparison.csv"))
    if (C.TAB / "025_recovery_summary.csv").exists():
        figS2(pd.read_csv(C.TAB / "025_recovery_summary.csv"), pd.read_csv(C.FITS / "025_recovery_mice.csv"))
    figS3(pd.read_csv(C.TAB / "022_recovery_confusion.csv"), pd.read_csv(C.TAB / "022_param_recovery.csv"))
    figS4(pd.read_csv(C.TAB / "018_boundaries.csv"), pd.read_csv(C.TAB / "018_stats.csv"))
    pd.DataFrame(NUM).to_csv(C.TAB / "026_figure_numbers.csv", index=False)


if __name__ == "__main__":
    main()
