"""031 -- single-mouse, single-trial data vs model (user 2026-10-29: "inspect more single-mouse single-trial data vs model predictions";
"greatly improve the single-mouse figures to be publication ready like the other figures").
Style from 026 (skills/ssl-figure-style: Arial 5-7.5 pt, 7.4 in width, auditory #2c2cdb, whisker #f7b519, no stimulus #262626,
R+ #00B400, R- #C800C8, aligned bold panel letters).

Per session: raster of every trial (lick = full tick, no lick = faint tick; one row per trial type), the data's centred running mean over
W_SMOOTH trials of the same type (thick, translucent), the model's one-step-ahead P(lick) on every trial (thin, unsmoothed); below,
the latents V_A, V_W, V_N, thirst T (at the session's posterior thirst rate) and P(engaged) if the model has the HMM.
Fit per mouse: log-likelihood per trial vs a baseline lick rate per trial type and day fitted on that mouse (in-sample, 3 numbers per day).
Outputs: figures/<pub dir>/fig4c_single_trials_<tag>.{png,pdf,svg} (4 example mice x day 0, day 1 + per-mouse fit summary);
figures/031_single_mouse/031_<tag>_single_trials_R{plus,minus}.pdf (one page per mouse, same style); tables/031_<tag>_mouse_fit.csv
Usage (haas): 031_single_mouse_trials.py [--latents 028_025S_trial_latents.parquet --tag 025S --label "final model"]
"""
from __future__ import annotations

import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

import common as C

F = importlib.import_module("026_pub_figures")
COH, TC, W, fmt_p, letter_row = F.COH, F.TC, F.W, F.fmt_p, F.letter_row
ROWS = ["auditory", "whisker", "no_stim"]          # raster rows, top to bottom
RLAB = {"auditory": "aud.", "whisker": "whi.", "no_stim": "no st."}
W_SMOOTH = 7
ENG = "#a5743a"
THIRST = "#6f8fb0"


def cs_lab(cs):
    return cs.replace("-", "−")


def roll(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def scores(e):
    """log-likelihood per trial of the model and of the per-day per-trial-type lick rate of the same mouse"""
    p = e.p_lick.clip(1e-6, 1 - 1e-6).to_numpy(); y = e.lick_flag.to_numpy()
    ll = np.where(y == 1, np.log(p), np.log1p(-p))
    q = e.groupby(["day", "trial_type"]).lick_flag.transform("mean").clip(1e-3, 1 - 1e-3).to_numpy()
    llb = np.where(y == 1, np.log(q), np.log1p(-q))
    return ll.mean(), llb.mean(), 1 - ll.sum() / llb.sum()


def session(axr, ax, a2, s, xmax, show_x=True, ylab=True, info=True):
    """one session: raster (axr), data running mean + model P(lick) (ax), latents (a2)"""
    x = np.arange(len(s))
    for r, t in enumerate(ROWS):
        idx = np.flatnonzero(s.trial_type == t)
        yr = len(ROWS) - 1 - r
        if len(idx):
            lk = s.lick_flag.to_numpy()[idx] == 1
            axr.vlines(idx[lk], yr - 0.42, yr + 0.42, color=TC[t], lw=0.45)
            axr.vlines(idx[~lk], yr - 0.15, yr + 0.15, color=TC[t], lw=0.3, alpha=0.3)
            ax.plot(idx, roll(s.lick_flag.to_numpy()[idx].astype(float), W_SMOOTH), color=TC[t], lw=2.0, alpha=0.28,
                    solid_capstyle="round")
            ax.plot(idx, s.p_lick.to_numpy()[idx], color=TC[t], lw=0.6)
    axr.set_ylim(-0.6, len(ROWS) - 0.4); axr.set_xlim(-2, xmax + 2); axr.axis("off")
    if ylab:
        for r, t in enumerate(ROWS):
            axr.text(-0.01, (len(ROWS) - 1 - r + 0.6) / len(ROWS), RLAB[t], transform=axr.transAxes, fontsize=4.5, color=TC[t],
                     ha="right", va="center")
    if info:
        ll, llb, _ = scores(s)
        axr.set_title(f"day {s.day.iloc[0]}, {len(s)} trials", loc="left", fontsize=5.5, pad=1.5, color="0.2")
        axr.set_title(f"LL/trial {ll:.3f} (baseline {llb:.3f})", loc="right", fontsize=5, pad=1.5, color="0.4")
    ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1], ["0", "", "1"]); ax.tick_params(labelbottom=False)
    if ylab:
        ax.set_ylabel("P(lick)")
    T = np.clip(1 - np.exp(s.lrho_smooth) * s.S, 0, 1)
    a2.axhline(0, color="0.85", lw=0.4)
    a2.plot(x, T, color=THIRST, lw=0.7, ls=(0, (3, 1.5)))
    if "p_engaged" in s and s.p_engaged.notna().any() and s.p_engaged.min() < 0.999:
        a2.plot(x, s.p_engaged, color=ENG, lw=0.6, ls=":")
    for q, t in (("V_A", "auditory"), ("V_W", "whisker"), ("V_N", "no_stim")):
        a2.plot(x, s[q], color=TC[t], lw=0.8)
    a2.set_ylim(-0.6, 1.08); a2.set_yticks([-0.5, 0, 0.5, 1], ["", "0", "", "1"])
    if ylab:
        a2.set_ylabel("value / state")
    a2.tick_params(labelbottom=show_x)
    if show_x:
        a2.set_xlabel("trial in session")


RATIOS = [0.55, 2.0, 1.0]


def legend_handles(hmm):
    h = [Line2D([], [], color="0.3", lw=0, marker="|", ms=5, mew=0.6, label="lick (short faint tick: no lick)"),
         Line2D([], [], color="0.3", lw=2.0, alpha=0.3, label=f"data, running mean of {W_SMOOTH} same-type trials"),
         Line2D([], [], color="0.3", lw=0.6, label="model P(lick) on every trial (one step ahead)"),
         Line2D([], [], color=TC["auditory"], lw=0.8, label="V_A"), Line2D([], [], color=TC["whisker"], lw=0.8, label="V_W"),
         Line2D([], [], color=TC["no_stim"], lw=0.8, label="V_N"), Line2D([], [], color=THIRST, lw=0.7, ls=(0, (3, 1.5)), label="thirst T")]
    if hmm:
        h.append(Line2D([], [], color=ENG, lw=0.6, ls=":", label="P(engaged)"))
    return h


def param_text(m, M):
    if M is None or m not in M.index:
        return ""
    r = M.loc[m]
    return (f"α_A {r.p_alpha_pos:.4f},  α_W {r.alpha_W:.4f} (τ_W {r.tau_W:.0f} whisker trials),  β {r.p_beta:.1f},  "
            f"{r.rewards_to_half_thirst:.0f} rewards halve thirst")


def mouse_page(e, label, M, days, hmm):
    m = e.mouse_id.iloc[0]; cs = e.reward_group.iloc[0]
    sess = [(d, sid) for (d, sid) in e.groupby(["day", "session_id"], sort=True).groups if d in days]
    n = len(sess)
    xmax = max(int((e.session_id == sid).sum()) for _, sid in sess)
    H = 0.95 + 1.6 * n
    fig = plt.figure(figsize=(W, H))
    outer = fig.add_gridspec(n, 1, hspace=0.32, left=0.085, right=0.985, top=1 - 0.8 / H, bottom=0.36 / H)
    for i, (d, sid) in enumerate(sess):
        s = e[e.session_id == sid].reset_index(drop=True)
        g = outer[i].subgridspec(3, 1, height_ratios=RATIOS, hspace=0.2)
        ax = fig.add_subplot(g[1]); axr = fig.add_subplot(g[0], sharex=ax); a2 = fig.add_subplot(g[2], sharex=ax)
        session(axr, ax, a2, s, xmax, show_x=i == n - 1)
    ll, llb, r2 = scores(e[e.day.isin(days)])
    fig.text(0.012, 1 - 0.08 / H, f"{m} ({cs_lab(cs)})", fontsize=7.5, weight="bold", color=COH[cs], va="top")
    fig.text(0.16, 1 - 0.09 / H, f"{label}:  {param_text(m, M)}", fontsize=5.5, va="top")
    fig.text(0.16, 1 - 0.23 / H, f"all sessions shown: LL/trial {ll:.3f} vs per-day trial-type lick rate {llb:.3f} "
             f"(pseudo-R² {r2:.3f})", fontsize=5.5, va="top", color="0.35")
    fig.legend(handles=legend_handles(hmm), ncol=4, loc="upper left", bbox_to_anchor=(0.012, 1 - 0.36 / H), fontsize=5,
               handlelength=1.8, columnspacing=1.2)
    return fig


def composite(L, ex, S, label, tag, hmm):
    """4 example mice x (day 0, day 1) + per-mouse fit summary"""
    days = [0, 1]
    n = len(ex)
    H = 1.75 * n + 2.1
    fig = plt.figure(figsize=(W, H))
    outer = fig.add_gridspec(n + 1, 1, height_ratios=[1] * n + [1.0], hspace=0.5, left=0.08, right=0.985, top=1 - 0.5 / H,
                             bottom=0.5 / H)
    firsts = []
    for i, (lab, m) in enumerate(ex.items()):
        e = L[L.mouse_id == m]
        cs = e.reward_group.iloc[0]
        row = outer[i].subgridspec(3, 2, height_ratios=RATIOS, hspace=0.2, wspace=0.07)
        ss = [e[e.day == d] for d in days]
        ss = [s[s.session_id == s.session_id.iloc[0]].reset_index(drop=True) if len(s) else s for s in ss]
        xmax = max(len(s) for s in ss)
        for j, s in enumerate(ss):
            ax = fig.add_subplot(row[1, j]); axr = fig.add_subplot(row[0, j], sharex=ax); a2 = fig.add_subplot(row[2, j], sharex=ax)
            if not len(s):
                for a_ in (ax, axr, a2):
                    a_.axis("off")
                continue
            session(axr, ax, a2, s, xmax, show_x=i == n - 1, ylab=j == 0)
            if j == 1:
                ax.tick_params(labelleft=False); a2.tick_params(labelleft=False)
            if j == 0:
                firsts.append(axr)
        Hf = fig.get_size_inches()[1]
        y1 = firsts[-1].get_position().y1
        fig.text(firsts[-1].get_position().x0, y1 + 0.17 / Hf, f"{lab}: {m} ({cs_lab(cs)})", color=COH[cs], fontsize=6.5, va="bottom")
        fig.text(firsts[-1].get_position().x0 - 0.42 / W, y1 + 0.15 / Hf, "abcd"[i], fontsize=9, weight="bold", va="bottom")
    # summary: per-mouse LL per trial vs baseline, pseudo-R^2 by cohort
    row = outer[n].subgridspec(1, 3, wspace=0.6, width_ratios=[1, 1, 1.25])
    ax = fig.add_subplot(row[0])
    for cs in COH:
        g = S[S.reward_group == cs]
        ax.scatter(g.ll_base, g.ll_per_trial, s=5, color=COH[cs], lw=0, alpha=0.8, label=cs_lab(cs))
    lo, hi = S[["ll_base", "ll_per_trial"]].min().min() - 0.02, S[["ll_base", "ll_per_trial"]].max().max() + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("baseline LL/trial (per-day\ntrial-type lick rate)"); ax.set_ylabel("model LL/trial"); ax.legend(loc="upper left")
    ax.set_title("per mouse, days −1 to 2", fontsize=6)
    ax2 = fig.add_subplot(row[1])
    for k, cs in enumerate(COH):
        y = S[S.reward_group == cs].pseudo_r2
        ax2.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(y)), y, s=5, color=COH[cs], lw=0, alpha=0.75)
        ax2.errorbar(k + 0.3, y.median(), yerr=[[y.median() - y.quantile(0.25)], [y.quantile(0.75) - y.median()]], fmt="o", ms=2.5,
                     color="k", lw=0.7)
    a, b = S[S.reward_group == "R+"].pseudo_r2, S[S.reward_group == "R-"].pseudo_r2
    p1, p2 = stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue
    ax2.axhline(0, color="0.6", lw=0.5, ls="--"); ax2.set_xticks([0, 1], [cs_lab(c) for c in COH]); ax2.set_xlim(-0.5, 1.6)
    ax2.set_ylabel("pseudo-R² vs baseline"); ax2.set_title(f"median, IQR; MWU {fmt_p(p1)}, Welch {fmt_p(p2)}", fontsize=6)
    ax3 = fig.add_subplot(row[2])
    ax3.axis("off")
    ax3.legend(handles=legend_handles(hmm), loc="center left", fontsize=5.5, handlelength=2.0)
    letter_row(fig, [ax, ax2], "ef")
    fig.suptitle(f"Single trials, data and model ({label}): example mice on day 0 (left) and day 1 (right)", y=1 - 0.1 / H, fontsize=7)
    F.OUT = C.FIG / "pub_v8"; F.OUT.mkdir(exist_ok=True)
    F.save(fig, f"fig4c_single_trials_{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="028_025S_trial_latents.parquet")
    ap.add_argument("--label", default="final model, mouse-level")
    ap.add_argument("--tag", default="025S")
    ap.add_argument("--days", type=int, nargs="+", default=[-1, 0, 1, 2])
    ap.add_argument("--png", nargs="*", default=None, help="mice to also save as png (default: the example mice)")
    a = ap.parse_args()
    L = pd.read_parquet(C.TAB / a.latents)
    if "trial_idx" in L:
        L = L.sort_values(["mouse_id", "day", "session_id", "trial_idx"], kind="stable")
    mf = C.TAB / f"028_{a.tag}_mouse_metrics.csv"
    M = pd.read_csv(mf).set_index("mouse_id") if mf.exists() else None
    hmm = "p_engaged" in L and (L.p_engaged < 0.999).any()
    ex = F.pick_examples(L)
    out = C.FIG / "031_single_mouse"; out.mkdir(parents=True, exist_ok=True)
    pngs = set(ex.values()) if a.png is None else set(a.png)
    rows = []
    for cs in COH:
        with PdfPages(out / f"031_{a.tag}_single_trials_{cs.replace('-', 'minus').replace('+', 'plus')}.pdf") as pdf:
            for m, e in L[L.reward_group == cs].groupby("mouse_id", sort=True):
                e = e.reset_index(drop=True)
                ll, llb, r2 = scores(e[e.day.isin(a.days)])
                rows.append(dict(mouse_id=m, reward_group=cs, n=int(e.day.isin(a.days).sum()), ll_per_trial=ll, ll_base=llb, pseudo_r2=r2))
                fig = mouse_page(e, a.label, M, a.days, hmm)
                pdf.savefig(fig, dpi=200)
                if m in pngs:
                    fig.savefig(out / f"031_{a.tag}_{m}.png", dpi=200)
                plt.close(fig)
    S = pd.DataFrame(rows)
    S.to_csv(C.TAB / f"031_{a.tag}_mouse_fit.csv", index=False)
    composite(L, ex, S, a.label, a.tag, hmm)
    print(S.groupby("reward_group")[["ll_per_trial", "ll_base", "pseudo_r2"]].median().round(3).to_string())
    print("examples:", ex)
