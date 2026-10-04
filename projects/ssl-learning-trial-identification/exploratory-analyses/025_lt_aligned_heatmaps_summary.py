"""025 -- Learning-trial-aligned curves and raw data for every LT definition, per-mouse heatmaps, and per-mouse summary
points (user 2026-09-30: "align curves / raw data to each mouse's learning trial, for all learning trial definitions.
Try the heatmap. Also do point 4 [per-mouse early vs late / pre vs post summary points]").
Inputs: 024 per-mouse HMM curves (sigma = 1; whisker, auditory and FA on the whisker-trial axis, 0-based index),
001/022 raw outcomes, and the 12 LT definitions of artifacts/013_learning_trials_all_methods.csv (whisker-trial index,
0-based; only the 88 ephys sessions have LTs -- the 12 behaviour-only mice enter the heatmaps and the first/last
summary but not the LT alignment).
Terminal disengagement (for the "last trials" summary and the heatmap end marker): same rule as
ssl_timeresolved_decoding.detect_terminal_disengagement -- trials after the session's last lick on any trial type
(from the first whisker trial on), counted only if >= 5 whisker AND >= 3 auditory trials (auditory is always
rewarded, so unlicked auditory trials mark disengagement). Alternative criterion reported in the csv for comparison:
auditory HMM curve < 0.5 held from some trial to the end of the session.
Figures (exploratory-analyses/):
  025_lt_aligned_grid.{pdf,png,svg}   rows = LT definitions, columns = whisker / FA / whisker - FA / auditory curves
                                      (mean +- SEM, drawn where >= 50% of the aligned mice have data) and raw whisker and
                                      FA lick rates in 5-trial bins; R+ and R- overlaid, n mice per cohort;
  025_heatmaps/<definition>.{pdf,png} rows = mice sorted by that definition's LT (undefined last, by length), columns =
                                      whisker trial; colour = whisker / FA / whisker - FA / auditory curve; white tick =
                                      LT, black tick = disengagement cut; one row of panels per cohort;
  025_heatmaps/aligned_<definition>   same, x = trial - LT (defined mice only);
  025_summary_first_last.{pdf,png,svg} per mouse: raw whisker / FA / whisker - FA / auditory lick rates in the first 20
                                      vs last 20 engaged whisker trials; all mice | learners; paired lines;
  025_summary_pre_post_lt.{pdf,png,svg} per definition: same rates in the 20 whisker trials before vs from the LT.
Stats (both tests, uncorrected): within cohort paired Wilcoxon + paired t; R+ vs R- on the change Mann-Whitney + Welch.
Tables (artifacts/): 025_disengagement.csv, 025_summary_first_last.csv, 025_summary_pre_post_lt.csv,
  025_lt_aligned_means.csv.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/025_lt_aligned_heatmaps_summary.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

COL = {"R+": "#00B400", "R-": "#C800C8"}          # GROUP_COLORS rplus / rminus
AUD, FA = "#1f5fbf", "0.35"
FS_L, FS_M, FS_S = 8, 7, 6
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate"]
PRE, POST = 30, 50                                 # aligned window: trials LT-30 .. LT+49
BIN = 5
NSUM = 20                                          # trials per summary epoch
MIN_FRAC = 0.5
METRICS = [("whisker", "Whisker"), ("fa", "False alarm"), ("w-fa", "Whisker − FA"), ("auditory", "Auditory")]


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": FS_M, "axes.labelsize": FS_M, "xtick.labelsize": FS_S,
                         "ytick.labelsize": FS_S, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})


def save(fig, stem, svg=True):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png") + (("svg",) if svg else ()):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def pf(p):
    return "n.a." if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}")


# ------------------------------------------------------------------------------------------------ per-session data
def session_table(inputs, tt, untrimmed):
    """Raw trials per session (all three types from the first whisker trial on) + disengagement cut."""
    out = {}
    for sid, inp in inputs.items():
        tw = np.asarray(inp["w_start"], float)
        parts = [pd.DataFrame(dict(t=tw, y=np.asarray(inp["w_outcomes"], int), tt="w")),
                 pd.DataFrame(dict(t=np.asarray(inp["n_start"], float), y=np.asarray(inp["n_outcomes"], int), tt="n"))]
        if "a_outcomes" in inp:
            parts.append(pd.DataFrame(dict(t=np.asarray(inp["a_start"], float), y=np.asarray(inp["a_outcomes"], int), tt="a")))
        else:
            u = untrimmed(sid, tt)
            a = u[u.trial_type == "auditory_trial"]
            parts.append(pd.DataFrame(dict(t=a.start_time.to_numpy(float), y=a.lick_flag.to_numpy(int), tt="a")))
        d = pd.concat(parts).sort_values("t").reset_index(drop=True)
        d["pos"] = np.interp(d.t, tw, np.arange(len(tw)))            # position on the 0-based whisker-trial axis
        d.loc[d.t < tw[0] - 1e-9, "pos"] = np.nan                    # trials before the first whisker trial: no position
        dd = d[d.t >= tw[0] - 1e-9]
        lick = np.where(dd.y.to_numpy() == 1)[0]
        trail = dd.iloc[lick[-1] + 1:] if len(lick) else dd
        n_w, n_a = int((trail.tt == "w").sum()), int((trail.tt == "a").sum())
        dis = n_w >= 5 and n_a >= 3
        n_eng = int((dd[dd.t < trail.t.iloc[0]].tt == "w").sum()) if dis else len(tw)
        out[sid] = dict(trials=d, disengaged=dis, n_engaged=n_eng, n_whisker=len(tw), trail_whisker=n_w, trail_auditory=n_a)
    return out


def aud_curve_cut(a):
    """First whisker-trial index from which the auditory curve stays < 0.5 to the end (NaN if none)."""
    if not np.isfinite(a).all():
        return np.nan
    below = a < 0.5
    if not below[-1]:
        return np.nan
    i = len(a) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    return i


def raw_rate(d, typ, lo, hi):
    s = d[(d.tt == typ) & (d.pos >= lo - 0.5) & (d.pos < hi - 0.5)]
    return s.y.mean() if len(s) else np.nan


# ------------------------------------------------------------------------------------------------ LT-aligned grid
def aligned_grid(S, lt, rows_out):
    rel = np.arange(-PRE, POST)
    edges = np.arange(-PRE, POST + 1, BIN)
    fig, axes = plt.subplots(len(DEFS), 6, figsize=(8.27, 1.25 * len(DEFS) + 0.6))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.965, bottom=0.04, wspace=0.45, hspace=0.55)
    cols = [("whisker", "Whisker curve"), ("fa", "FA curve"), ("w-fa", "Whisker − FA"), ("auditory", "Auditory curve"),
            ("raw_w", f"Whisker licks ({BIN}-trial bins)"), ("raw_n", f"FA licks ({BIN}-trial bins)")]
    for r, df_ in enumerate(DEFS):
        nlab = []
        for rg in ("R+", "R-"):
            sids = [s for s in S if S[s]["rg"] == rg and np.isfinite(lt.get(s, {}).get(df_, np.nan))]
            nlab.append(f"{'R+' if rg == 'R+' else 'R−'} {len(sids)}")
            if not sids:
                continue
            for c, (key, _) in enumerate(cols):
                ax = axes[r, c]
                if key.startswith("raw"):
                    M = np.full((len(sids), len(edges) - 1), np.nan)
                    for i, s in enumerate(sids):
                        d, L0 = S[s]["trials"], lt[s][df_]
                        for j in range(len(edges) - 1):
                            M[i, j] = raw_rate(d, "w" if key == "raw_w" else "n", L0 + edges[j], L0 + edges[j + 1])
                    x = edges[:-1] + BIN / 2
                else:
                    M = np.full((len(sids), len(rel)), np.nan)
                    for i, s in enumerate(sids):
                        v, L0 = S[s]["curves"][key], int(lt[s][df_])
                        idx = rel + L0
                        ok = (idx >= 0) & (idx < len(v))
                        M[i, ok] = v[idx[ok]]
                    x = rel
                n = np.isfinite(M).sum(0)
                with np.errstate(all="ignore"):
                    m = np.nanmean(M, 0)
                    se = np.nanstd(M, 0, ddof=1) / np.sqrt(n)
                v_ = n >= max(2, MIN_FRAC * len(sids))
                if key.startswith("raw"):
                    ax.errorbar(x[v_], m[v_], yerr=se[v_], color=COL[rg], lw=1, ms=2, marker="o", elinewidth=0.6, capsize=0)
                else:
                    ax.fill_between(x[v_], (m - se)[v_], (m + se)[v_], color=COL[rg], alpha=0.22, lw=0)
                    ax.plot(x[v_], m[v_], color=COL[rg], lw=1.1)
                for j in np.where(v_)[0]:
                    rows_out.append(dict(definition=df_, cohort=rg, measure=key, rel_trial=x[j], n_mice=int(n[j]),
                                         mean=m[j], sem=se[j]))
        for c, (key, title) in enumerate(cols):
            ax = axes[r, c]
            ax.axvline(0, color="0.5", lw=0.6, ls="--")
            if key == "w-fa":
                ax.axhline(0, color="0.6", lw=0.5, ls=":")
                ax.set_ylim(-0.4, 0.9)
            else:
                ax.set_ylim(-0.03, 1.03)
            ax.set_xlim(-PRE, POST)
            if r == 0:
                ax.set_title(title, fontsize=FS_M)
            if r == len(DEFS) - 1:
                ax.set_xlabel("whisker trial − LT")
            else:
                ax.set_xticklabels([])
        axes[r, 0].set_ylabel(f"{df_}\n({', '.join(nlab)})", fontsize=FS_S)
    save(fig, HERE / "025_lt_aligned_grid")


# ------------------------------------------------------------------------------------------------ heatmaps
def heatmaps(S, lt):
    cmap_p = "viridis"
    for df_ in DEFS:
        for aligned in (False, True):
            fig, axes = plt.subplots(2, 4, figsize=(8.27, 7.0), gridspec_kw=dict(height_ratios=[1, 1]))
            fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.07, wspace=0.25, hspace=0.3)
            for r, rg in enumerate(("R+", "R-")):
                sids = [s for s in S if S[s]["rg"] == rg]
                ltv = {s: lt.get(s, {}).get(df_, np.nan) for s in sids}
                if aligned:
                    sids = [s for s in sids if np.isfinite(ltv[s])]
                sids = sorted(sids, key=lambda s: (not np.isfinite(ltv[s]), ltv[s] if np.isfinite(ltv[s]) else -S[s]["n_whisker"]))
                for c, (key, title) in enumerate(METRICS):
                    ax = axes[r, c]
                    if not sids:
                        ax.set_axis_off()
                        continue
                    if aligned:
                        x0, x1 = -PRE, POST
                        M = np.full((len(sids), x1 - x0), np.nan)
                        for i, s in enumerate(sids):
                            v, L0 = S[s]["curves"][key], int(ltv[s])
                            idx = np.arange(x0, x1) + L0
                            ok = (idx >= 0) & (idx < len(v))
                            M[i, ok] = v[idx[ok]]
                    else:
                        x0, x1 = 0, max(S[s]["n_whisker"] for s in sids)
                        M = np.full((len(sids), x1), np.nan)
                        for i, s in enumerate(sids):
                            v = S[s]["curves"][key]
                            M[i, :len(v)] = v
                    vmin, vmax, cm = (-1, 1, "RdBu_r") if key == "w-fa" else (0, 1, cmap_p)
                    cmo = plt.get_cmap(cm).copy()
                    cmo.set_bad("white")
                    im = ax.imshow(M, aspect="auto", cmap=cmo, vmin=vmin, vmax=vmax, interpolation="nearest",
                                   extent=(x0 - 0.5, x1 - 0.5, len(sids) - 0.5, -0.5))
                    for i, s in enumerate(sids):
                        if aligned:
                            ax.plot([0, 0], [i - 0.45, i + 0.45], color="w", lw=0.8)
                            if S[s]["disengaged"]:
                                xc = S[s]["n_engaged"] - ltv[s]
                                if x0 <= xc < x1:
                                    ax.plot([xc, xc], [i - 0.45, i + 0.45], color="k", lw=0.8)
                        else:
                            if np.isfinite(ltv[s]):
                                ax.plot([ltv[s], ltv[s]], [i - 0.45, i + 0.45], color="w", lw=0.9)
                            if S[s]["disengaged"]:
                                ax.plot([S[s]["n_engaged"]] * 2, [i - 0.45, i + 0.45], color="k", lw=0.9)
                    n_def = sum(np.isfinite(ltv[s]) for s in sids)
                    if not aligned and n_def < len(sids):
                        ax.axhline(n_def - 0.5, color="k", lw=0.6)
                    ax.set_title(title, fontsize=FS_M, color="k")
                    ax.set_xlabel("whisker trial − LT" if aligned else "whisker trial")
                    ax.set_yticks([])
                    if c == 0:
                        ax.set_ylabel(f"{'R+' if rg == 'R+' else 'R−'} mice (n = {len(sids)}; LT defined {n_def})",
                                      color=COL[rg])
                    if c == 3 or key == "w-fa":
                        cax = ax.inset_axes([1.03, 0.0, 0.04, 0.35])
                        cb = fig.colorbar(im, cax=cax)
                        cb.ax.tick_params(labelsize=FS_S, length=2)
                        cb.set_label("Δ P(lick)" if key == "w-fa" else "P(lick)", fontsize=FS_S)
            fig.text(0.07, 0.975, f"{df_} · rows sorted by LT (undefined below the line, by session length) · "
                     "white tick = LT, black tick = disengagement cut", fontsize=FS_M, va="top")
            name = ("aligned_" if aligned else "") + df_.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("+", "plus")
            save(fig, HERE / "025_heatmaps" / name, svg=False)


# ------------------------------------------------------------------------------------------------ summary points
def paired_panel(ax, vals, title, rows_out, ctx):
    """vals: {rg: (a, b)} per-mouse arrays; paired lines, mean +- SEM, stats text."""
    chg = {}
    for k, rg in enumerate(("R+", "R-")):
        a, b = vals[rg]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        xs = np.array([0, 1]) + 2.6 * k
        for ai, bi in zip(a, b):
            ax.plot(xs, [ai, bi], color=COL[rg], alpha=0.25, lw=0.5)
        for xx, v, face in ((xs[0], a, "white"), (xs[1], b, COL[rg])):
            if len(v):
                ax.errorbar(xx, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0, fmt="o",
                            mfc=face, mec=COL[rg], ecolor=COL[rg], ms=3.5, elinewidth=0.9, capsize=0, zorder=4)
        pw = stats.wilcoxon(b, a).pvalue if len(a) >= 5 and np.any(b != a) else np.nan
        pt = stats.ttest_rel(b, a).pvalue if len(a) >= 3 else np.nan
        chg[rg] = b - a
        ax.text(xs.mean(), 1.02, f"n={len(a)}\nW {pf(pw)}\nt {pf(pt)}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=FS_S - 0.5, color=COL[rg])
        rows_out.append(dict(**ctx, measure=title, cohort=rg, n=len(a), first=np.mean(a) if len(a) else np.nan,
                             second=np.mean(b) if len(b) else np.nan, change=np.mean(b - a) if len(a) else np.nan,
                             p_wilcoxon=pw, p_paired_t=pt))
    if min(len(chg["R+"]), len(chg["R-"])) >= 3:
        pm = stats.mannwhitneyu(chg["R+"], chg["R-"]).pvalue
        pwl = stats.ttest_ind(chg["R+"], chg["R-"], equal_var=False).pvalue
    else:
        pm = pwl = np.nan
    for r in rows_out[-2:]:
        r.update(p_change_mannwhitney=pm, p_change_welch=pwl)
    ax.set_title(title, fontsize=FS_M, pad=26)
    ax.text(0.5, -0.3, f"Δ R+ vs R−: MW {pf(pm)}, Welch {pf(pwl)}", transform=ax.transAxes, ha="center",
            fontsize=FS_S - 0.5)
    ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.set_xlim(-0.5, 4.1)


def rates(S, s, lo, hi):
    d = S[s]["trials"]
    w, n, a = raw_rate(d, "w", lo, hi), raw_rate(d, "n", lo, hi), raw_rate(d, "a", lo, hi)
    return dict(whisker=w, fa=n, **{"w-fa": w - n}, auditory=a)


def summary_first_last(S, rows_out):
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 5.4))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12, wspace=0.45, hspace=0.95)
    for r, scope in enumerate(("all", "learners")):
        sids = [s for s in S if scope == "all" or S[s]["lc"] in ("good", "moderate")]
        for c, (key, title) in enumerate(METRICS):
            vals = {}
            for rg in ("R+", "R-"):
                ss = [s for s in sids if S[s]["rg"] == rg and S[s]["n_engaged"] >= 2 * NSUM]
                vals[rg] = (np.array([rates(S, s, 0, NSUM)[key] for s in ss]),
                            np.array([rates(S, s, S[s]["n_engaged"] - NSUM, S[s]["n_engaged"])[key] for s in ss]))
            paired_panel(axes[r, c], vals, title, rows_out, dict(scope=scope, comparison="first vs last engaged"))
            ax = axes[r, c]
            ax.set_xticks([0, 1, 2.6, 3.6])
            ax.set_xticklabels(["first", "last", "first", "last"])
            ax.set_ylim(-0.6 if key == "w-fa" else -0.03, 1.03)
            if c == 0:
                ax.set_ylabel(f"{'All mice' if scope == 'all' else 'Learners'}\nlick rate")
    fig.text(0.5, 0.975, f"First vs last {NSUM} engaged whisker trials (terminal disengaged block removed); open = first, "
             "filled = last; W = paired Wilcoxon, t = paired t (uncorrected)", ha="center", va="top", fontsize=FS_M)
    save(fig, HERE / "025_summary_first_last")


def summary_pre_post(S, lt, rows_out):
    fig, axes = plt.subplots(len(DEFS), 4, figsize=(8.27, 1.55 * len(DEFS) + 0.8))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.975, bottom=0.03, wspace=0.45, hspace=1.35)
    for r, df_ in enumerate(DEFS):
        for c, (key, title) in enumerate(METRICS):
            vals = {}
            for rg in ("R+", "R-"):
                ss = [s for s in S if S[s]["rg"] == rg and np.isfinite(lt.get(s, {}).get(df_, np.nan))]
                vals[rg] = (np.array([rates(S, s, lt[s][df_] - NSUM, lt[s][df_])[key] for s in ss]),
                            np.array([rates(S, s, lt[s][df_], lt[s][df_] + NSUM)[key] for s in ss]))
            ax = axes[r, c]
            paired_panel(ax, vals, title, rows_out, dict(scope="LT defined", comparison=df_))
            if r:
                ax.set_title("")
            ax.set_xticks([0, 1, 2.6, 3.6])
            ax.set_xticklabels(["pre", "post", "pre", "post"])
            ax.set_ylim(-0.6 if key == "w-fa" else -0.03, 1.03)
            if c == 0:
                ax.set_ylabel(df_, fontsize=FS_S)
    save(fig, HERE / "025_summary_pre_post_lt")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed, detect_terminal_disengagement
    style()
    tt = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "trials.parquet")
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    inputs.update(pickle.load(open(ART / "022_inputs_behaviour_only.pkl", "rb")))
    per = {p["session_id"]: p for p in pickle.load(open(ART / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    S = session_table(inputs, tt, _active_trials_for_curve_untrimmed)
    dis_rows = []
    for sid in S:
        S[sid].update(rg=per[sid]["reward_group"], lc=per[sid]["learning_category"], curves=per[sid]["curves"],
                      mouse_id=per[sid]["mouse_id"])
        ref = detect_terminal_disengagement(sid, tt) if "a_outcomes" not in inputs[sid] else None
        dis_rows.append(dict(session_id=sid, mouse_id=S[sid]["mouse_id"], reward_group=S[sid]["rg"],
                             learning_category=S[sid]["lc"], n_whisker=S[sid]["n_whisker"],
                             disengaged=S[sid]["disengaged"], n_engaged_whisker=S[sid]["n_engaged"],
                             trail_whisker=S[sid]["trail_whisker"], trail_auditory=S[sid]["trail_auditory"],
                             library_disengaged=None if ref is None else ref["disengaged"],
                             library_n_whisker_dropped=None if ref is None else ref["n_whisker_dropped"],
                             aud_curve_below_half_from=aud_curve_cut(S[sid]["curves"]["auditory"])))
    dis = pd.DataFrame(dis_rows)
    dis.to_csv(ART / "025_disengagement.csv", index=False)
    chk = dis.dropna(subset=["library_disengaged"])
    print("disengagement rule vs library:", int((chk.disengaged == chk.library_disengaged.astype(bool)).sum()), "/", len(chk),
          "agree; disengaged:", int(dis.disengaged.sum()), "/", len(dis))
    tab = pd.read_csv(ART / "013_learning_trials_all_methods.csv").set_index("session_id")
    lt = {s: {d: float(tab.loc[s, d]) for d in DEFS} for s in tab.index if s in S}
    rows = []
    aligned_grid(S, lt, rows)
    pd.DataFrame(rows).to_csv(ART / "025_lt_aligned_means.csv", index=False)
    heatmaps(S, lt)
    r1, r2 = [], []
    summary_first_last(S, r1)
    pd.DataFrame(r1).to_csv(ART / "025_summary_first_last.csv", index=False)
    summary_pre_post(S, lt, r2)
    pd.DataFrame(r2).to_csv(ART / "025_summary_pre_post_lt.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
