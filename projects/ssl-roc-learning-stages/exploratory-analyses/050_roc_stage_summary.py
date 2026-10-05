"""Cross-type summaries of the multi-ROC learning-stage analysis (inputs: 045 / 047; good units; no correction).

Figures -> combined_results_ks4/ssl-roc-learning-stages/figures/summary/
  S1_overview_heatmap      measures x comparisons: change in % significant and in mean |sel| (pooled), stars = perm. p
  S1b / S1c                same per sign: % positive / % negative units; mean positive / |negative| selectivity part
  S2*_area_group_heatmaps  measures x area groups: expert - learning per cohort for % sig. (S2), |sel| (S2b) and per
                           sign (S2c-f); grey = not included
  S3_selectivity           forest of the change in pooled mean |sel| (95% bootstrap CI) per cohort + split violins of
                           signed selectivity (learning | expert) for the key types
  S4_focality              change in Gini / entropy focality across fine areas per cohort (95% bootstrap CI), of
                           % significant and of mean |sel|
  S5_tuning                single-cell tuning: modality composition (bimodal / whisker-only / auditory-only),
                           preference, decision / gated, conditional tuning, mixed selectivity
  S6_divergence            learning x cohort interaction [delta(R+) - delta(R-)] per measure (95% CI, perm. p);
                           S6b per sign
Also: summary_table.csv (pooled effects of every measure and comparison).
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
st47 = importlib.import_module("047_roc_stage_stats")
f48 = importlib.import_module("048_roc_stage_type_figures")
BASE = st47.BASE
FIG = BASE / "figures" / "summary"
COH = f48.COH
GROUPS = [
    ("Passive sensory", ["whisker_passive_pre", "whisker_passive_post", "auditory_passive_pre", "auditory_passive_post"]),
    ("Passive pre→post session", ["whisker_pre_vs_post_learning", "auditory_pre_vs_post_learning",
                                       "wh_vs_aud_pre_vs_post_learning", "baseline_pre_vs_post_learning"]),
    ("Active sensory", ["whisker_active", "auditory_active", "whisker_sensory", "auditory_sensory", "whisker_hit_vs_cr",
                        "auditory_hit_vs_cr"]),
    ("Modality", ["wh_vs_aud_passive_pre", "wh_vs_aud_passive_post", "wh_vs_aud_active"]),
    ("Choice / decision", ["choice", "whisker_choice", "auditory_choice", "whisker_hit_vs_spontaneous",
                           "auditory_hit_vs_spontaneous"]),
    ("Pre-stimulus", ["baseline_choice", "baseline_whisker_choice", "baseline_auditory_choice"]),
    ("Lick / motor", ["spontaneous_licks", "spontaneous_licks_vs_cr"]),
    ("Tuning categories", ["cat:resp_whisker", "cat:resp_auditory", "cat:bimodal", "cat:whisker_only", "cat:auditory_only",
                           "cat:pref_whisker", "cat:pref_auditory", "cat:whisker_decision", "cat:whisker_gated",
                           "cat:auditory_decision", "cat:auditory_gated", "cat:lick_resp", "cat:motor",
                           "cat:choice_in_whisker_resp", "cat:decision_in_whisker_resp", "cat:choice_in_auditory_resp",
                           "cat:mixed_n"])]
COMPS = [("stage:R+", "R+: expert−learning"), ("stage:R-", "R−: expert−learning"),
         ("interaction:frac_sig", "interaction"), ("cohort:learning", "learning: R+−R−"),
         ("cohort:expert", "expert: R+−R−")]
KEY_TYPES = ["whisker_passive_post", "whisker_active", "auditory_active", "whisker_sensory", "wh_vs_aud_active",
             "whisker_choice", "choice", "whisker_hit_vs_spontaneous", "spontaneous_licks", "spontaneous_licks_vs_cr",
             "whisker_pre_vs_post_learning", "baseline_choice"]


def pstar(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.5, "pdf.fonttype": 42})
    return plt


def ordered(meas_avail, headers=False):
    """rows in group order; headers=True inserts a '##<group>' header row before each group"""
    rows, sep = [], []
    for g, ms in GROUPS:
        ms = [m for m in ms if m in meas_avail]
        if ms:
            if headers:
                rows.append(f"##{g}")
            sep.append((len(rows), g)); rows += ms
    return rows, sep


def style_rows(ax, rows, fontsize=4.8):
    ax.set_yticks(range(len(rows)), [r[2:] if r.startswith("##") else r.replace("cat:", "• ") for r in rows],
                  fontsize=fontsize)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r.startswith("##"):
            t.set_fontweight("bold"); t.set_fontsize(fontsize + 0.4)


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=230)


MLAB = {"frac_sig": ("Δ % significant", 100), "frac_pos": ("Δ % significant, positive", 100),
        "frac_neg": ("Δ % significant, negative", 100), "mean_abs_sel": ("Δ mean |selectivity|", 1),
        "mean_sel_pos": ("Δ mean positive selectivity", 1), "mean_sel_neg": ("Δ mean |negative selectivity|", 1)}


def s1_overview(plt, R, metrics=("frac_sig", "mean_abs_sel"), name="S1_overview_heatmap", title=None):
    A = R[(R.level == "all") & (R.region == "all")]
    rows_all, sep_all = ordered(set(A.measure), headers=True)
    rows_sel, sep_sel = ordered({m for m in A.measure if not m.startswith("cat:") or m == "cat:mixed_n"}, headers=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 9.4), gridspec_kw=dict(wspace=0.95))
    out = []
    for ax, metric in zip(axs, metrics):
        lab, scale = MLAB[metric]
        rows, sep = (rows_all, sep_all) if metric == "frac_sig" else (rows_sel, sep_sel)
        if metric not in ("frac_sig", "mean_abs_sel"):                  # per-sign: no sign for the mixed-selectivity count
            rows = [r for r in rows if r not in ("cat:mixed_n", "##Tuning categories")]
        Mx = np.full((len(rows), len(COMPS)), np.nan); ann = [[""] * len(COMPS) for _ in rows]
        for j, (c, _) in enumerate(COMPS):
            cc = c if not c.startswith("interaction") else f"interaction:{metric}"
            sub = A[A.comparison == cc].set_index("measure")
            if f"{metric}_diff" not in sub:
                continue
            for i, m in enumerate(rows):
                if m in sub.index and np.isfinite(sub.loc[m, f"{metric}_diff"]):
                    d, p = sub.loc[m, f"{metric}_diff"] * scale, sub.loc[m, f"{metric}_p"]
                    Mx[i, j] = d; ann[i][j] = f"{d:+.1f}{pstar(p)}" if scale == 100 else f"{d:+.3f}{pstar(p)}"
                    out.append(dict(measure=m, comparison=cc, metric=metric, diff=sub.loc[m, f"{metric}_diff"], p=p,
                                    lo=sub.loc[m, f"{metric}_lo"], hi=sub.loc[m, f"{metric}_hi"]))
        v = np.nanpercentile(np.abs(Mx), 95) if np.isfinite(Mx).any() else 1
        im = ax.imshow(Mx, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(COMPS)):
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=4.2)
        style_rows(ax, rows)
        ax.set_xticks(range(len(COMPS)), [c[1] for c in COMPS], rotation=40, ha="right", fontsize=5)
        for y0, g in sep:
            ax.axhline(y0 - 1.5, color="k", lw=0.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label(lab, fontsize=5.5); cb.ax.tick_params(labelsize=5)
        ax.set_title(lab + "\n(* p<.05, ** p<.01, *** p<.001;\npermutation, uncorrected)", fontsize=5.8)
    fig.suptitle(title or "Learning-stage and cohort effects across all ROC types (pooled over areas; good units)",
                 fontsize=7.5, y=0.995)
    fig.subplots_adjust(left=0.2, right=0.93, top=0.92, bottom=0.08)
    save(fig, name); plt.close(fig)
    return pd.DataFrame(out)


def s2_area(plt, R, metric="frac_sig", name="S2_area_group_heatmaps"):
    import ephys_utilities.allen_utils.allen_utils as au
    order = au.get_area_group_custom_order()
    G = R[R.level == "area_group"]
    lab, scale = MLAB[metric]
    rows, sep = ordered(set(G.measure) if metric == "frac_sig" else {m for m in G.measure if not m.startswith("cat:")})
    regs = [g for g in order if g in set(G.region)] + sorted(set(G.region) - set(order))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 9.0), gridspec_kw=dict(wspace=0.12))
    for ax, c in zip(axs, ["R+", "R-"]):
        sub = G[G.comparison == f"stage:{c}"].set_index(["measure", "region"])
        Mx = np.full((len(rows), len(regs)), np.nan); ann = [[""] * len(regs) for _ in rows]
        inc = np.zeros_like(Mx, bool)
        for i, m in enumerate(rows):
            for j, r in enumerate(regs):
                if (m, r) in sub.index:
                    x = sub.loc[(m, r)]
                    inc[i, j] = bool(x.included)
                    if x.included:
                        Mx[i, j] = x[f"{metric}_diff"] * scale; ann[i][j] = pstar(x[f"{metric}_p"])
        vm = 20 if scale == 100 else 0.06
        im = ax.imshow(Mx, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        ax.imshow(np.where(inc, np.nan, 1.0), cmap="Greys", vmin=0, vmax=4, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(regs)):
                if ann[i][j]:
                    ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=4.5)
        ax.set_xticks(range(len(regs)), [r.replace(" areas", "").replace("Somatosensory", "SS") for r in regs],
                      rotation=60, ha="right", fontsize=5)
        ax.set_yticks(range(len(rows)), [m.replace("cat:", "• ") for m in rows] if c == "R+" else [], fontsize=4.8)
        for y0, _ in sep:
            ax.axhline(y0 - 0.5, color="k", lw=0.6)
        ax.set_title(f"{c.replace('-', chr(8722))}: expert − learning", color=COH[c], fontsize=6.5)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label(lab, fontsize=5.5)
    fig.suptitle(f"Area-group changes with learning stage: {lab}\n(* perm. p < .05; grey = < 10 units or < 3 sessions per stage)",
                 fontsize=7, y=0.995)
    fig.subplots_adjust(left=0.22, right=0.92, top=0.94, bottom=0.12)
    save(fig, name); plt.close(fig)


def s3_selectivity(plt, R, U, M):
    A = R[(R.level == "all") & (R.region == "all")]
    types = [m for g, ms in GROUPS[:-1] for m in ms if m in set(A.measure)]
    fig = plt.figure(figsize=(7.4, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.3, top=0.95, bottom=0.05, left=0.2, right=0.97)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(types))
    for k, c in enumerate(["R+", "R-"]):
        sub = A[A.comparison == f"stage:{c}"].set_index("measure").reindex(types)
        d, lo, hi, p = sub.mean_abs_sel_diff, sub.mean_abs_sel_lo, sub.mean_abs_sel_hi, sub.mean_abs_sel_p
        yy = y + (k - 0.5) * 0.3
        ax.errorbar(d, yy, xerr=[d - lo, hi - d], fmt="none", ecolor=COH[c], elinewidth=0.7)
        ax.scatter(d, yy, s=12, facecolors=[COH[c] if pp < 0.05 else "white" for pp in p.fillna(1)], edgecolors=COH[c],
                   linewidths=0.7, zorder=3, label=c.replace("-", "−"))
    ax.axvline(0, color="k", lw=0.5); ax.set_yticks(y, types, fontsize=5); ax.invert_yaxis()
    ax.set_xlabel("Δ mean |selectivity| (expert − learning), pooled; 95% bootstrap CI; filled = perm. p < .05")
    ax.legend(frameon=False, fontsize=6); ax.set_title("a  Change of selectivity magnitude with learning stage", loc="left")
    sub = gs[1].subgridspec(3, 4, hspace=0.6, wspace=0.35)
    for i, t in enumerate([k for k in KEY_TYPES if k in M][:12]):
        ax = fig.add_subplot(sub[i // 4, i % 4])
        meas = M[t]; v = meas.valid.to_numpy() & np.isfinite(meas.sel.to_numpy(float))
        for k, c in enumerate(["R+", "R-"]):
            for side, g in [(-1, "learning"), (1, "expert")]:
                x = meas.sel.to_numpy(float)[v & (U.cohort == c).to_numpy() & (U.stage == g).to_numpy()]
                if len(x) < 5:
                    continue
                vp = ax.violinplot(x, positions=[k], widths=0.8, showextrema=False, points=80)
                for b in vp["bodies"]:
                    verts = b.get_paths()[0].vertices
                    verts[:, 0] = np.clip(verts[:, 0], k, np.inf) if side > 0 else np.clip(verts[:, 0], -np.inf, k)
                    b.set_facecolor(f48.shade(COH[c], f48.STAGE_SHADE[g])); b.set_edgecolor("none"); b.set_alpha(0.9)
                ax.plot([k + side * 0.25], [np.median(x)], "_", color="k", ms=5)
        ax.axhline(0, color="0.6", lw=0.4); ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_title(t, fontsize=5.5)
        ax.set_ylim(-1, 1)
        if i % 4 == 0:
            ax.set_ylabel("selectivity")
    fig.text(0.02, 0.44, "b  Selectivity distributions (left half learning, right half expert; bar = median)", fontsize=6.5)
    save(fig, "S3_selectivity"); plt.close(fig)


def s4_focality(plt, F):
    if "metric" not in F:
        F = F.assign(metric="frac_sig")
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 7.0), gridspec_kw=dict(wspace=0.12))
    for ax, (fm, ind) in zip(axs, [(fm, ind) for fm in ["frac_sig", "mean_abs_sel"] for ind in ["gini", "entropy_focality"]]):
        sub = F[(F.level == "area_acronym_custom") & F.comparison.str.startswith("stage:") & (F.metric == fm)]
        rows, sep = ordered(set(sub.measure))
        y = np.arange(len(rows))
        for k, c in enumerate(["R+", "R-"]):
            s = sub[sub.comparison == f"stage:{c}"].set_index("measure").reindex(rows)
            d, lo, hi, p = s[f"{ind}_diff"], s[f"{ind}_lo"], s[f"{ind}_hi"], s[f"{ind}_p"]
            yy = y + (k - 0.5) * 0.3
            ax.errorbar(d, yy, xerr=[d - lo, hi - d], fmt="none", ecolor=COH[c], elinewidth=0.6)
            ax.scatter(d, yy, s=10, facecolors=[COH[c] if pp < 0.05 else "white" for pp in p.fillna(1)],
                       edgecolors=COH[c], linewidths=0.6, zorder=3, label=c.replace("-", "−"))
        ax.axvline(0, color="k", lw=0.5); ax.invert_yaxis()
        ax.set_yticks(y, [m.replace("cat:", "• ") for m in rows] if ax is axs[0] else [], fontsize=4.8)
        ax.set_title(f"{'% significant' if fm == 'frac_sig' else 'mean |selectivity|'}", fontsize=6)
        for y0, _ in sep:
            ax.axhline(y0 - 0.5, color="0.6", lw=0.4)
        ax.set_xlabel(f"Δ {'Gini' if ind == 'gini' else 'entropy foc.'}\n(expert − learning)", fontsize=5.5)
        ax.legend(frameon=False, fontsize=5.5)
    fig.suptitle("Focality of significant units across fine areas: >0 = more concentrated in fewer areas after learning\n"
                 "(95% bootstrap CI over sessions; filled = stage-label permutation p < .05)", fontsize=6.8)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.07)
    save(fig, "S4_focality"); plt.close(fig)


def s5_tuning(plt, R, U, M):
    A = R[(R.level == "all") & (R.region == "all")]
    fig = plt.figure(figsize=(7.4, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45, left=0.08, right=0.98, top=0.92, bottom=0.08)
    xs = {("R+", "learning"): 0, ("R+", "expert"): 1, ("R-", "learning"): 2.5, ("R-", "expert"): 3.5}

    def stacked(ax, cats, labels, colors, title):
        for (c, g), x in xs.items():
            bottom = 0
            for m, col in zip(cats, colors):
                r = A[(A.measure == m) & (A.comparison == f"stage:{c}")]
                if not len(r):
                    continue
                v = r.iloc[0][f"frac_sig_{'B' if g == 'expert' else 'A'}"] * 100
                ax.bar(x, v, 0.75, bottom=bottom, color=col, edgecolor="k", lw=0.3,
                       label=labels[cats.index(m)] if (c, g) == ("R+", "learning") else None)
                bottom += v
        txt = []
        for m, lab in zip(cats, labels):
            for c in ["R+", "R-"]:
                r = A[(A.measure == m) & (A.comparison == f"stage:{c}")]
                if len(r) and r.iloc[0].frac_sig_p < 0.05:
                    txt.append(f"{lab} {c}: {r.iloc[0].frac_sig_diff * 100:+.1f} pp {pstar(r.iloc[0].frac_sig_p)}")
        ax.set_xticks(list(xs.values()), ["learn.", "expert"] * 2, fontsize=5.5)
        ax.text(0.5, -0.13, "R+", transform=ax.get_xaxis_transform(), ha="center", color=COH["R+"], weight="bold")
        ax.text(3.0, -0.13, "R−", transform=ax.get_xaxis_transform(), ha="center", color=COH["R-"], weight="bold")
        ax.set_ylabel("% of units"); ax.legend(frameon=False, fontsize=5, loc="upper left")
        ax.set_title(title + ("\n" + "; ".join(txt) if txt else ""), fontsize=5.5)

    stacked(fig.add_subplot(gs[0, 0]), ["cat:whisker_only", "cat:auditory_only", "cat:bimodal"],
            ["whisker only", "auditory only", "bimodal"], ["#E08214", "#2166AC", "#7B3294"], "a  Stimulus responsiveness (active)")
    stacked(fig.add_subplot(gs[0, 1]), ["cat:pref_whisker", "cat:pref_auditory"], ["whisker > auditory", "auditory > whisker"],
            ["#E08214", "#2166AC"], "b  Modality preference (wh_vs_aud_active)")
    stacked(fig.add_subplot(gs[0, 2]), ["cat:whisker_gated", "cat:whisker_decision"], ["whisker gated", "whisker decision"],
            ["#1B7837", "#A6DBA0"], "c  Decision coding (whisker)")
    ax = fig.add_subplot(gs[1, 0:2])
    conds = ["cat:choice_in_whisker_resp", "cat:decision_in_whisker_resp", "cat:choice_in_auditory_resp"]
    labs = ["choice | whisker-resp.", "decision | whisker-resp.", "choice | auditory-resp."]
    for i, m in enumerate(conds):
        for (c, g), x in xs.items():
            r = A[(A.measure == m) & (A.comparison == f"stage:{c}")]
            if not len(r):
                continue
            r = r.iloc[0]; side = "B" if g == "expert" else "A"
            xx = i * 5 + x
            ax.bar(xx, r[f"frac_sig_{side}"] * 100, 0.75, color=f48.shade(COH[c], f48.STAGE_SHADE[g]), edgecolor="k", lw=0.3)
        for c, (x0, x1) in [("R+", (0, 1)), ("R-", (2.5, 3.5))]:
            r = A[(A.measure == m) & (A.comparison == f"stage:{c}")]
            if len(r):
                top = ax.get_ylim()[1]
                ax.text(i * 5 + (x0 + x1) / 2, top * 0.95, pstar(r.iloc[0].frac_sig_p) or "n.s.", ha="center", fontsize=5.5)
    ax.set_xticks([i * 5 + 1.75 for i in range(3)], labs)
    ax.set_ylabel("% of responsive units"); ax.set_title("d  Conditional tuning: sensory-responsive units that also code choice / decision\n"
                                                      "(bars: R+ learn., R+ expert, R− learn., R− expert)", fontsize=6)
    ax = fig.add_subplot(gs[1, 2])
    mx = M["cat:mixed_n"]; v = mx.valid.to_numpy()
    for (c, g), x in xs.items():
        n = mx.sel.to_numpy(float)[v & (U.cohort == c).to_numpy() & (U.stage == g).to_numpy()]
        h = np.bincount(n.astype(int), minlength=len(st47.CORE) + 1) / max(len(n), 1)
        ax.plot(np.arange(len(h)), h * 100, color=f48.shade(COH[c], f48.STAGE_SHADE[g]), marker="o", ms=2, lw=0.9,
                ls="-" if c == "R+" else "--", label=f"{c} {g}")
    r = A[(A.measure == "cat:mixed_n")]
    txt = "; ".join(f"{c}: Δmean {rr.mean_abs_sel_diff:+.2f} {pstar(rr.mean_abs_sel_p) or 'n.s.'}"
                    for c, rr in [(x.split(':')[1], x2) for x, x2 in r.set_index('comparison').iterrows() if x.startswith('stage:')])
    ax.set_xlabel("# significant core ROC types per unit"); ax.set_ylabel("% of units")
    ax.set_title("e  Mixed selectivity\n" + txt, fontsize=5.5); ax.legend(frameon=False, fontsize=4.8)
    fig.suptitle("Single-cell tuning across learning stages (pooled over areas; permutation p, uncorrected)", fontsize=7.5)
    save(fig, "S5_tuning"); plt.close(fig)


def s6_divergence(plt, R, metrics=("frac_sig", "mean_abs_sel"), name="S6_divergence"):
    A = R[(R.level == "all") & (R.region == "all")]
    fig, axs = plt.subplots(1, len(metrics), figsize=(7.4, 8.0), gridspec_kw=dict(wspace=0.12))
    for ax, metric in zip(axs, metrics):
        lab, scale = MLAB[metric]; lab = lab[2:]
        sub = A[A.comparison == f"interaction:{metric}"].set_index("measure")
        if metric != "frac_sig":
            sub = sub[[not m.startswith("cat:") or m == "cat:mixed_n" for m in sub.index]]
        rows, sep = ordered(set(sub.index))
        y = np.arange(len(rows)); s = sub.reindex(rows)
        d, lo, hi, p = s[f"{metric}_diff"] * scale, s[f"{metric}_lo"] * scale, s[f"{metric}_hi"] * scale, s[f"{metric}_p"]
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="none", ecolor="0.3", elinewidth=0.6)
        ax.scatter(d, y, s=12, facecolors=["k" if pp < 0.05 else "white" for pp in p.fillna(1)], edgecolors="k", lw=0.6, zorder=3)
        ax.axvline(0, color="k", lw=0.5); ax.invert_yaxis()
        show = len(metrics) <= 2 or ax is axs[0]
        ax.set_yticks(y, [m.replace("cat:", "• ") for m in rows] if show else [], fontsize=4.8)
        for y0, _ in sep:
            ax.axhline(y0 - 0.5, color="0.6", lw=0.4)
        ax.set_title(lab, fontsize=6)
        ax.set_xlabel("Δ(R+) − Δ(R−), Δ = expert − learning", fontsize=5)
    fig.suptitle("Learning x cohort divergence (pooled; 95% bootstrap CI; filled = cohort-label permutation p < .05)", fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.93, bottom=0.07, wspace=0.15 if len(metrics) > 2 else 0.7)
    save(fig, name); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt = setup()
    U, M, S = f48.load_all()
    R = S["region_stats"]
    T = pd.concat([
        s1_overview(plt, R),
        s1_overview(plt, R, ("frac_pos", "frac_neg"), "S1b_overview_heatmap_sign_fraction",
                    "Per-sign % significant: positive vs negative units (pooled; good units)"),
        s1_overview(plt, R, ("mean_sel_pos", "mean_sel_neg"), "S1c_overview_heatmap_sign_selectivity",
                    "Per-sign selectivity: mean positive part and mean |negative part| (sum = mean |sel|; pooled)")])
    T.to_csv(BASE / "figures" / "summary_table.csv", index=False)
    for m, suf in [("frac_sig", ""), ("mean_abs_sel", "b_abs_sel"), ("frac_pos", "c_frac_pos"),
                   ("frac_neg", "d_frac_neg"), ("mean_sel_pos", "e_sel_pos"), ("mean_sel_neg", "f_sel_neg")]:
        s2_area(plt, R, m, f"S2{suf}_area_group_heatmaps")
    s3_selectivity(plt, R, U, M); s4_focality(plt, S["focality"]); s5_tuning(plt, R, U, M)
    s6_divergence(plt, R)
    s6_divergence(plt, R, ("frac_pos", "frac_neg", "mean_sel_pos", "mean_sel_neg"), "S6b_divergence_per_sign")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
