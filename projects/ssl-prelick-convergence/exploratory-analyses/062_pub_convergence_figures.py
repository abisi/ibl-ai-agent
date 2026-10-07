"""Publication figures: do whisker hits become more like auditory hits before the lick, more in R+ than in R-?

Population: --population all (all mice, headline) or learners (learning_category good / moderate).
Data (v2 thresholds): pre-lick 100 ms window before the corrected first lick; units quality good or mua with mean raw
pre-lick rate >= 0.1 Hz; trials active, perf != 6, warm-up cut, A1 trim; classes WH (whisker hit), AH (auditory hit),
FA (false alarm = no-stim lick). Inputs: 051 per-session npz, 053 prelick_units, 057 lambda, 059 transfer/decoder,
060 lambda_LDA. Unit of analysis = session. Tests: Mann-Whitney U (shown) and Welch t (table) for learning vs expert
within cohort and R+ vs R- at expert; learning x cohort interaction [E-L](R+) - [E-L](R-) by permuting cohort labels
across mice (10000). Area family-wise p: max-|z| permutation across area groups (057 / 059 / 060). No other correction.
Figures -> combined_results_ks4/ssl-prelick-convergence/across_days/fa/publication/<population>/Fig1..Fig5.{png,pdf} + stats_<population>.csv
  Fig1 task, alignment, reaction times, dataset, recording coverage, whole-brain population PSTHs
  Fig2 single neurons: examples, WH vs FA selectivity, reward-lick neuron transfer, selectivity scatter, shared code
  Fig3 population distance (main): schematic, class-mean triangles, distance difference dd = d(WH,ref) - d(WH,AH) and its
       components per session, RT-matched dd, cumulative dd (cv squared distances per unit from 057; no ratio)
  Fig3S lambda (first half: schematic, projections, lambda, lambda vs dd, RT-matched, cumulative) and lambda_LDA (second
       half: agreement, lambda_LDA, d', RT-matched lambda_LDA)
  Fig4 areas: change of dd per area group (+ ANOVA), dd map, area dd, area PSTHs (units / sessions)
  Fig5 single-session decoders, chance-corrected by linear shifts (069)
  COSYNE_figure_v2 (condensed, learning -> expert, distance-based) and captions_<population>.md
  (the lambda-based COSYNE_figure v1 is no longer built; fig_cosyne kept for reference)
Area inclusion (Fig4, COSYNE): per cohort, >= 3 sessions with a valid dd at both stages; areas need not be shared
across cohorts; the interaction (and family-wise p) is reported only for areas included in both cohorts.
"""
import argparse
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m56 = importlib.import_module("056_roc_prelick_psth")
m61 = importlib.import_module("061_roc_prelick_learners")
BASE = m51.OUTROOT
CL = {"WH": "#f7b519", "AH": "#2c2cdb", "FA": "#262626"}
CLAB = {"WH": "Whisker hit", "AH": "Auditory hit", "FA": "False alarm" if m51.REF == "fa" else "Spontaneous lick"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
GLAB = {("R+", "learning"): "R+ learning", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− learning",
        ("R-", "expert"): "R− expert"}
XS = {GROUPS[0]: 0.0, GROUPS[1]: 1.0, GROUPS[2]: 2.4, GROUPS[3]: 3.4}
UNIT_SET = ("good", "mua")
N_PERM = 10000
W_IN = 7.4
STATS = []


# ------------------------------------------------------------------ style helpers
def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titlepad": 4})
    return plt


def letter_row(fig, axes, letters, dx_in=0.32, dy_in=0.2):
    """panel letters aligned on one row: same y (above the highest axis of the row), x = left of each axis"""
    W, H = fig.get_size_inches()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, l in zip(axes, letters):
        fig.text(ax.get_position().x0 - dx_in / W, top + dy_in / H, l, fontsize=9, weight="bold", ha="left", va="bottom")


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


REF_WORDS = [("false-alarm", "spontaneous-lick"), ("False alarm", "Spontaneous lick"), ("false alarm", "spontaneous lick"),
             ("FA", "SL")]


def relabel_reference(fig):
    """spontaneous-lick reference: every text element names the reference correctly (FA -> SL)"""
    import re
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s:
            continue
        s2 = s
        for a, b in REF_WORDS[:3]:
            s2 = s2.replace(a, b)
        s2 = re.sub(r"\bFA\b", "SL", s2)
        if s2 != s:
            t.set_text(s2)


def save(fig, out, name):
    if m51.REF == "sl":
        relabel_reference(fig)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{ext}", dpi=300)


# ------------------------------------------------------------------ statistics
def group_stats(d, col, label, rng, fig_panel):
    d = d.dropna(subset=[col])
    g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])][col].to_numpy() for k in GROUPS}
    out = {"means": {k: g[k].mean() if len(g[k]) else np.nan for k in GROUPS},
           "sems": {k: g[k].std(ddof=1) / np.sqrt(len(g[k])) if len(g[k]) > 1 else np.nan for k in GROUPS},
           "n": {k: len(g[k]) for k in GROUPS}}
    for name, (a, b) in {"R+ L vs E": (GROUPS[0], GROUPS[1]), "R- L vs E": (GROUPS[2], GROUPS[3]),
                         "expert R+ vs R-": (GROUPS[3], GROUPS[1]), "learning R+ vs R-": (GROUPS[2], GROUPS[0])}.items():
        x, y = g[a], g[b]
        if len(x) >= 3 and len(y) >= 3:
            out[name] = (stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue, y.mean() - x.mean())
    out["interaction"] = (np.nan, np.nan)
    if all(len(v) >= 3 for v in g.values()):
        mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
        midx = pd.Index(mice).get_indexer(d.mouse_id); st = d.stage.to_numpy(); y = d[col].to_numpy()

        def stat(coh):
            mm = {k: y[(coh == k[0]) & (st == k[1])] for k in GROUPS}
            if min(len(v) for v in mm.values()) == 0:
                return np.nan
            return (mm[GROUPS[1]].mean() - mm[GROUPS[0]].mean()) - (mm[GROUPS[3]].mean() - mm[GROUPS[2]].mean())
        obs = stat(d.cohort.to_numpy())
        null = np.array([stat(rng.permutation(mc)[midx]) for _ in range(N_PERM)]); null = null[np.isfinite(null)]
        out["interaction"] = (obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))
    row = dict(panel=fig_panel, measure=label, **{f"mean_{GLAB[k]}": out["means"][k] for k in GROUPS},
               **{f"n_{GLAB[k]}": out["n"][k] for k in GROUPS},
               interaction=out["interaction"][0], interaction_p_perm=out["interaction"][1])
    for name in ["R+ L vs E", "R- L vs E", "expert R+ vs R-", "learning R+ vs R-"]:
        if name in out:
            row[f"{name} diff"], row[f"{name} p_MWU"], row[f"{name} p_Welch"] = out[name][2], out[name][0], out[name][1]
    STATS.append(row)
    return out


def dots_panel(ax, d, col, ylabel, rng, panel, title, ref=None):
    """session dots (open = learning, filled = expert), mean +- SEM, brackets with MWU p, interaction in the title"""
    S = group_stats(d, col, title, rng, panel)
    d = d.dropna(subset=[col])
    for k, x in XS.items():
        v = d[(d.cohort == k[0]) & (d.stage == k[1])][col].to_numpy()
        fc = "white" if k[1] == "learning" else COH[k[0]]
        ax.scatter(x - 0.12 + rng.uniform(-0.1, 0.1, len(v)), v, s=7, facecolor=fc, edgecolor=COH[k[0]], lw=0.5,
                   alpha=0.8, zorder=2)
        if len(v):
            ax.errorbar(x + 0.2, S["means"][k], S["sems"][k], fmt="o", ms=3.8, color=COH[k[0]], mfc=fc, mew=0.9,
                        lw=1.0, capsize=0, zorder=3)
    if ref is not None:
        for yv, c in ref:
            ax.axhline(yv, color=c, lw=0.6, ls=(0, (2, 2)), zorder=1)
    lo, hi = ax.get_ylim(); rngy = hi - lo
    levels = [hi + 0.06 * rngy, hi + 0.06 * rngy, hi + 0.22 * rngy]
    for (name, (x0, x1), c, yb) in [("R+ L vs E", (0, 1), COH["R+"], levels[0]), ("R- L vs E", (2.4, 3.4), COH["R-"], levels[1]),
                                    ("expert R+ vs R-", (1, 3.4), "0.15", levels[2])]:
        if name in S:
            p = S[name][0]
            ax.plot([x0, x0, x1, x1], [yb - 0.02 * rngy, yb, yb, yb - 0.02 * rngy], color=c, lw=0.6)
            ax.text((x0 + x1) / 2, yb + 0.01 * rngy, fmt_p(p), ha="center", va="bottom", fontsize=5, color=c)
    ax.set_ylim(lo, hi + 0.36 * rngy)
    ax.set_xticks([0, 1, 2.4, 3.4], ["Learn.", "Expert", "Learn.", "Expert"])
    ax.set_xlim(-0.55, 3.95)
    for x, c in [(0.5, "R+"), (2.9, "R-")]:
        ax.text(x, -0.2, c.replace("-", "−"), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5,
                color=COH[c], weight="bold")
    ax.set_ylabel(ylabel)
    ip = S["interaction"][1]
    ax.set_title(f"{title}\nInteraction {fmt_p(ip)}", fontsize=6)
    return S


# ------------------------------------------------------------------ data
def load_population(pop):
    keep = m61.learner_filter if pop == "learners" else (lambda df: df)     # day-0 sessions of non-learners dropped
    D = {}
    D["lam"] = keep(pd.read_csv(BASE / "lambda" / "lambda_sessions.csv"))
    D["tr"] = keep(pd.read_csv(BASE / "transfer" / "transfer_sessions.csv"))
    D["lda"] = keep(pd.read_csv(BASE / "lambda_lda" / "lambda_lda_sessions.csv"))
    nl = "shift"                   # 069 chance-corrected single-session decoders
    D["dec"] = keep(pd.read_csv(BASE / f"decoder_transfer_{nl}" / "sessions.csv"))
    fpp = BASE / "pseudopop" / pop / "pseudopop_stats.csv"          # 064 hierarchical bootstrap (already population-specific)
    D["pp"] = pd.read_csv(fpp) if fpp.exists() else None
    fpb = BASE / "pseudopop" / pop / "bootstrap_samples.json"
    D["pp_boot"] = {k: v for k, v in json.load(open(fpb)).items() if k.startswith("all|")} if fpb.exists() else None
    D["lam_tests"] = pd.read_csv(BASE / "lambda" / "lambda_tests.csv") if pop == "all" else \
        pd.read_csv(BASE / "learners" / "tests_learners.csv")
    W, P, tc = m56.load()
    W = keep(W[W.quality_label.isin(UNIT_SET)])
    D["W"], D["P"], D["tc"] = W, P, tc
    return D


def session_unit_metrics(W):
    rows = []
    for sid, g in W.groupby("session_id"):
        a = g["sig:whisker_hit_vs_fa_prelick@all"].dropna().astype(float)
        wa = g["sig:wh_vs_aud_hit_prelick@all"].dropna().astype(float)            # WH vs AH (user 2026-10-07, Fig 2c)
        both = g[["sel:whisker_hit_vs_fa_prelick@all", "sel:auditory_hit_vs_fa_prelick@all"]].dropna()
        rows.append(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0],
                         frac_sig_WHvsFA=a.mean() if len(a) >= 10 else np.nan,
                         frac_sig_WHvsAH=wa.mean() if len(wa) >= 10 else np.nan,
                         r_shared=stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0] if len(both) >= 10 else np.nan))
    return pd.DataFrame(rows)


def axis_projections(sessions, regions, W, rng):
    """cross-validated projections on the mean-difference (lambda) axis: k-fold over AH / FA trials; axis = mean(AH) -
    mean(FA) of the training trials (units z-scored on training trials); held-out AH, FA and all WH projected; per fold
    normalised so that held-out FA mean = 0 and AH mean = 1."""
    from sklearn.model_selection import StratifiedKFold
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(sessions)]
    rows = []
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "area_group", "cohort", "stage"]],
                    on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        coh, stg = K.cohort.dropna().iloc[0], K.stage.dropna().iloc[0]
        X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
        ok = (raw.mean(1) >= m51.MIN_FR) & K.cohort.notna().to_numpy()
        for reg in regions:
            u = ok & (np.ones(len(K), bool) if reg == "all" else (K.area_group == reg).to_numpy())
            if u.sum() < 5:
                continue
            Z = X[u].T
            m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"; y = (lab[m] == "AH").astype(int)
            k = min(5, int(y.sum()), int((1 - y).sum()))
            if k < 3 or w.sum() < 3:
                continue
            Xa, Xw = Z[m], Z[w]
            whs = np.full((k, w.sum()), np.nan)
            for f, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1e9))).split(Xa, y)):
                mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
                A, B = (Xa[tr] - mu) / sd, (Xa[te] - mu) / sd
                ax_ = A[y[tr] == 1].mean(0) - A[y[tr] == 0].mean(0)
                s_te, s_w = B @ ax_, ((Xw - mu) / sd) @ ax_
                fa_m, ah_m = s_te[y[te] == 0].mean(), s_te[y[te] == 1].mean()
                if ah_m - fa_m <= 0:
                    continue
                for c, s in zip(np.where(y[te] == 1, "AH", "FA"), s_te):
                    rows.append(dict(session_id=r.session_id, cohort=coh, stage=stg, region=reg, cls=c, score=(s - fa_m) / (ah_m - fa_m)))
                whs[f] = (s_w - fa_m) / (ah_m - fa_m)
            for s in np.nanmean(whs, 0):
                rows.append(dict(session_id=r.session_id, cohort=coh, stage=stg, region=reg, cls="WH", score=s))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Figure 1
def fig1(plt, D, out, pop, rng):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    W, P, tc = D["W"], D["P"], D["tc"]
    fig = plt.figure(figsize=(W_IN, 8.0))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 0.95, 0.95], hspace=0.95, wspace=0.55,
                          left=0.08, right=0.98, top=0.91, bottom=0.06)
    # a task schematic
    ax_a = fig.add_subplot(gs[0, 0:2]); ax_a.axis("off"); ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 4.2)
    ax_a.set_title("Task: lick after the stimulus; reward depends on cohort", loc="left")
    ax_a.text(3.3, 3.75, "Stimulus", ha="center", fontsize=6, weight="bold")
    ax_a.text(5.3, 3.75, "Lick", ha="center", fontsize=6, weight="bold")
    ax_a.text(7.2, 3.75, "Reward R+", ha="center", fontsize=6, weight="bold", color=COH["R+"])
    ax_a.text(9.0, 3.75, "Reward R−", ha="center", fontsize=6, weight="bold", color=COH["R-"])
    for i, (c, stim, rp, rm) in enumerate([("WH", "Whisker", "yes", "no"), ("AH", "Auditory", "yes", "yes"),
                                            ("FA", "None", "no", "no")]):
        yy = 2.9 - i * 1.1
        ax_a.add_patch(FancyBboxPatch((0.05, yy - 0.32), 2.0, 0.64, boxstyle="round,pad=0.02,rounding_size=0.15",
                                      fc=CL[c], ec="none", alpha=0.9))
        ax_a.text(1.05, yy, CLAB[c], ha="center", va="center", fontsize=6 if len(CLAB[c]) < 14 else 5.2, color="white" if c != "WH" else "k", weight="bold")
        ax_a.text(3.3, yy, stim, ha="center", va="center", fontsize=6)
        ax_a.text(5.3, yy, "yes", ha="center", va="center", fontsize=6)
        for x, v, coh in [(7.2, rp, "R+"), (9.0, rm, "R-")]:
            ax_a.text(x, yy, "Yes" if v == "yes" else "No", ha="center", va="center", fontsize=6,
                      color=COH[coh] if v == "yes" else "0.55", weight="bold" if v == "yes" else "normal")
    # b alignment schematic
    ax_b = fig.add_subplot(gs[0, 2:4]); ax_b.axis("off"); ax_b.set_xlim(-0.6, 0.8); ax_b.set_ylim(0, 1)
    ax_b.set_title("Alignment: 100 ms before the first lick", loc="left")
    ax_b.plot([-0.55, 0.75], [0.45, 0.45], color="0.2", lw=0.8)
    ax_b.add_patch(Rectangle((0.25, 0.35), 0.1, 0.2, fc="#FDD49E", ec="none"))
    for x, t in [(-0.45, "Trial\nstart"), (0.0, "Stimulus\n(WH, AH)"), (0.35, "First lick\n(corrected)")]:
        ax_b.plot([x, x], [0.38, 0.52], color="0.2", lw=0.8)
        ax_b.text(x, 0.62, t, ha="center", va="bottom", fontsize=5.5)
    ax_b.text(0.3, 0.27, "Pre-lick window\n[−100, 0] ms", ha="center", va="top", fontsize=5.5)
    ax_b.text(-0.45, 0.05, "Baseline: [−1, −0.015] s before trial start; first lick = start + (lick − response window start)",
              fontsize=4.8, color="0.3")
    # c reaction times (pooled) and c' median RT per session by cohort x stage
    row1 = gs[1, :].subgridspec(1, 5, wspace=0.55, width_ratios=[1, 1, 1, 0.75, 0.75])
    ax_c = fig.add_subplot(row1[0])
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    sess_info = W.drop_duplicates("session_id").set_index("session_id")[["cohort", "stage", "mouse_id"]]
    rts = {c: [] for c in m51.CLASSES}; rt_rows = []
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        for c in m51.CLASSES:
            v = z["rt"][z["cls"] == c].astype(float); v = v[np.isfinite(v)]
            rts[c].append(v)
            if len(v) >= 3 and r.session_id in sess_info.index:
                rt_rows.append(dict(session_id=r.session_id, cls=c, rt=np.median(v) * 1e3, **sess_info.loc[r.session_id].to_dict()))
    RT = pd.DataFrame(rt_rows)
    rt_classes = [c for c in m51.CLASSES if len(np.concatenate(rts[c]))]
    bins = np.arange(0, 1.0, 0.025) * 1e3
    for c in rt_classes:
        v = np.concatenate(rts[c]) * 1e3
        ax_c.hist(v, bins, density=True, histtype="step", color=CL[c], lw=1.1, label=f"{CLAB[c]} ({np.median(v):.0f} ms)")
    ax_c.set_xlabel("Reaction time (ms)"); ax_c.set_ylabel("Density"); ax_c.set_yticks([])
    ax_c.set_ylim(0, ax_c.get_ylim()[1] * 1.75); ax_c.set_xlim(0, 1000)
    ax_c.legend(frameon=False, loc="upper right", handlelength=1.0, fontsize=4.6, title="Median RT", title_fontsize=4.6)
    ax_c.set_title("Reaction times (all trials)")
    ax_c2 = fig.add_subplot(row1[1])
    xs_ = {GROUPS[0]: 0, GROUPS[1]: 1, GROUPS[2]: 2.4, GROUPS[3]: 3.4}
    for kk, c in enumerate(rt_classes):
        dx = (kk - (len(rt_classes) - 1) / 2) * 0.22
        for k, x in xs_.items():
            v = RT[(RT.cls == c) & (RT.cohort == k[0]) & (RT.stage == k[1])].rt
            if len(v) > 1:
                ax_c2.errorbar(x + dx, v.mean(), v.sem(), fmt="o", ms=3, color=CL[c], mfc="white" if k[1] == "learning" else CL[c],
                               mew=0.8, lw=0.8, capsize=0)
        for c0, (a_, b_) in [("R+", (GROUPS[0], GROUPS[1])), ("R-", (GROUPS[2], GROUPS[3]))]:
            m_ = [RT[(RT.cls == c) & (RT.cohort == k[0]) & (RT.stage == k[1])].rt.mean() for k in (a_, b_)]
            ax_c2.plot([xs_[a_] + dx, xs_[b_] + dx], m_, color=CL[c], lw=0.6)
    ax_c2.set_xticks([0, 1, 2.4, 3.4], ["L", "E", "L", "E"]); ax_c2.set_xlim(-0.5, 3.9)
    for x, c0 in [(0.5, "R+"), (2.9, "R-")]:
        ax_c2.annotate(c0.replace("-", "−"), xy=(x, 0), xycoords=ax_c2.get_xaxis_transform(), xytext=(0, -11),
                       textcoords="offset points", ha="center", va="top", fontsize=6, color=COH[c0], weight="bold")
    ax_c2.set_ylabel("Median RT per session (ms)"); ax_c2.set_title("RT by cohort and stage\n(mean ± s.e.m. sessions)", fontsize=5.6)
    # d dataset
    ax_d = fig.add_subplot(row1[2])
    cnt = W.drop_duplicates("session_id").groupby(["cohort", "stage"]).agg(sess=("session_id", "size"), mice=("mouse_id", "nunique"))
    nu = W[(W["sig:auditory_hit_vs_fa_prelick@all"].notna() & W["sig:whisker_hit_vs_fa_prelick@all"].notna())].groupby(["cohort", "stage"]).size()
    for k, x in XS.items():
        if k in cnt.index:
            fc = "white" if k[1] == "learning" else COH[k[0]]
            ax_d.bar(x, cnt.loc[k, "sess"], 0.75, color=fc, edgecolor=COH[k[0]], lw=0.9)
            ax_d.text(x, cnt.loc[k, "sess"] + 1, f"{cnt.loc[k, 'mice']} mice\n{nu.get(k, 0) / 1000:.1f}k units", ha="center",
                      va="bottom", fontsize=4.3)
    ax_d.set_xticks([0, 1, 2.4, 3.4], ["L", "E", "L", "E"])
    for x, c in [(0.5, "R+"), (2.9, "R-")]:
        ax_d.annotate(c.replace("-", "−"), xy=(x, 0), xycoords=ax_d.get_xaxis_transform(), xytext=(0, -11),
                      textcoords="offset points", ha="center", va="top", fontsize=6, color=COH[c], weight="bold")
    ax_d.set_ylabel("Sessions"); ax_d.set_ylim(0, cnt.sess.max() * 1.45); ax_d.set_xlim(-0.7, 4.1); ax_d.set_title("Dataset")
    # e coverage (sagittal + coronal unit positions)
    outl = coverage_outlines(out.parent)
    xyz = W[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].dropna()
    idx = rng.choice(len(xyz), min(len(xyz), 30000), replace=False); xyz = xyz.iloc[idx]
    cohs = W.loc[xyz.index, "cohort"].to_numpy()
    axs_e = []
    for j, (kind, xc, mask, xl) in enumerate([("sag", "ccf_atlas_ap", outl["sag"], "AP (mm)"),
                                              ("cor", "ccf_atlas_ml", outl["cor"], "ML (mm)")]):
        ax = fig.add_subplot(row1[3 + j]); axs_e.append(ax)
        ax.contour(np.arange(mask.shape[1]) * 0.01 * outl["step"], np.arange(mask.shape[0]) * 0.01 * outl["step"],
                   mask.astype(float), levels=[0.5], colors="0.45", linewidths=0.6)
        for c in ["R-", "R+"]:
            mm = cohs == c
            ax.scatter(xyz[xc].to_numpy()[mm] / 1000, xyz.ccf_atlas_dv.to_numpy()[mm] / 1000, s=0.2, color=COH[c],
                       alpha=0.25, lw=0, rasterized=True)
        ax.set_ylim(8, 0); ax.set_aspect("equal"); ax.set_xlabel(xl)
        if kind == "cor":
            ax.set_xlim(5.2, 11.4); ax.set_yticks([])
        else:
            ax.set_ylabel("DV (mm)")
        ax.spines["left"].set_visible(kind == "sag")
    axs_e[0].set_title("Recorded units (CCF)", loc="left")
    # f population PSTHs: mean over units within session, mean +- s.e.m. over sessions (pre-trial baseline)
    PB, KB, tcb = load_psth_prestart(W)
    Wb = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    sub = gs[2, :].subgridspec(1, 4, wspace=0.3)
    tested = (Wb["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wb["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
    axs_f = []
    for j, k in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j]); axs_f.append(ax)
        ns, nu_ = mouse_psth(ax, Wb, PB, tcb, tested & ((Wb.cohort == k[0]) & (Wb.stage == k[1])).to_numpy(), legend=(j == 0))
        ax.set_title(f"{GLAB[k]} ({ns} sessions)", color=COH[k[0]])
        ax.set_xlabel("Time from first lick (ms)"); ax.set_xlim(-600, 400); ax.set_xticks([-400, -200, 0, 200, 400])
        if j == 0:
            ax.set_ylabel("Firing rate − pre-trial\nbaseline (Hz)")
    yl = (min(a.get_ylim()[0] for a in axs_f), max(a.get_ylim()[1] for a in axs_f))
    for a in axs_f:
        a.set_ylim(*yl)
    # h: pre-lick-window rate change per trial type (session means of the g traces in [-100, 0) ms) and ratios to ref
    RA = "FA" if m51.REF == "fa" else "SL"
    win = (tcb >= -100) & (tcb < 0)
    v = PB[Wb.row_b.to_numpy()][:, :, win].mean(2)                   # units x class
    U = Wb[["session_id", "mouse_id", "cohort", "stage"]].copy()
    for kk, c in enumerate(m51.CLASSES):
        U[c] = v[:, kk]
    Sx = U[tested].groupby(["session_id", "mouse_id", "cohort", "stage"], as_index=False)[list(m51.CLASSES)].mean()
    den = Sx["FA"].where(Sx["FA"] >= 0.5)                             # ratio only where the reference change is >= 0.5 Hz
    Sx["r_WH"], Sx["r_AH"] = Sx["WH"] / den, Sx["AH"] / den
    Sx.to_csv(out / "fig1h_prelick_rates.csv", index=False)
    sub = gs[3, :].subgridspec(1, 5, wspace=0.75)
    axs_h = []
    for j, (col, yl, pid, ttl, ref_) in enumerate([
            ("WH", "Δ rate (Hz)", "1h WH", "Whisker hits", None), ("AH", "Δ rate (Hz)", "1h AH", "Auditory hits", None),
            ("FA", "Δ rate (Hz)", "1h ref", CLAB["FA"] + "s", None),
            ("r_WH", f"WH / {RA}", "1h WH/ref", f"Whisker hits / {RA}", [(1, "0.6")]),
            ("r_AH", f"AH / {RA}", "1h AH/ref", f"Auditory hits / {RA}", [(1, "0.6")])]):
        ax = fig.add_subplot(sub[j]); axs_h.append(ax)
        dots_panel(ax, Sx, col, yl, rng, pid, ttl, ref=ref_)
        ax.set_title(ax.get_title(), fontsize=5.3); ax.set_xticklabels(["L", "E", "L", "E"])
    fig.align_ylabels([ax_c] + axs_f[:1] + axs_h[:1])
    letter_row(fig, [ax_a, ax_b], "ab"); letter_row(fig, [ax_c, ax_c2, ax_d, axs_e[0]], "cdef")
    letter_row(fig, [axs_f[0]], "g"); letter_row(fig, [axs_h[0]], "h")
    fig.suptitle("Figure 1 | Pre-lick activity on whisker hits, auditory hits and false alarms", x=0.08, y=0.985, ha="left",
                 fontsize=7.5, weight="bold")
    save(fig, out, "Fig1_task_data"); plt.close(fig)


def coverage_outlines(cache_dir):
    f = cache_dir / "ccf_outlines.npz"
    if f.exists():
        z = np.load(f); return dict(sag=z["sag"], cor=z["cor"], step=int(z["step"]))
    import tifffile
    ann = tifffile.imread(m51.RES.parent / "Anatomy" / "allen_mouse_bluebrain_barrels_10um_v1.0" / "annotation.tiff")
    step = 4
    a = ann[::step, ::step, ::step] > 0                            # (AP, DV, ML)
    sag = a.any(axis=2).T                                           # (DV, AP)
    cor = a.any(axis=0)                                             # (DV, ML)
    np.savez_compressed(f, sag=sag, cor=cor, step=step)
    return dict(sag=sag, cor=cor, step=step)


# ------------------------------------------------------------------ Figure 2
EX_TYPES = [("conv", "Converging neuron\n(AH ≠ {r} and WH ≠ {r}, same sign)"),
            ("rlonly", "Reward-lick-only neuron\n(AH ≠ {r}, WH ≈ {r})"),
            ("whonly", "Whisker-hit-only neuron\n(WH ≠ {r}, AH ≈ {r})")]
RASTER_WIN = (-0.5, 0.3)
RASTER_MAX = 25


def pick_examples(W):
    """one good, high-rate, strongly selective example per type for R+ expert and R- expert (positive selectivity)"""
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    G = W[(W.quality_label == "good") & W[f"sig:{af}"].notna() & W[f"sig:{wf}"].notna()].copy()
    G["fr"] = G[f"fr_window:{af}"]
    sa, sw = G[f"sel:{af}"], G[f"sel:{wf}"]
    gA, gW = G[f"sig:{af}"] == 1, G[f"sig:{wf}"] == 1
    masks = {"conv": gA & gW & (sa > 0.3) & (sw > 0.3),
             "rlonly": gA & ~gW & (sa > 0.4) & (sw.abs() < 0.1),
             "whonly": gW & ~gA & (sw > 0.4) & (sa.abs() < 0.1)}
    out = {}
    for k in [GROUPS[1], GROUPS[3]]:
        for t, m in masks.items():
            g = G[m & (G.cohort == k[0]) & (G.stage == k[1]) & (G.fr >= 5)]
            if len(g):
                score = (sa.loc[g.index].abs() + sw.loc[g.index].abs()) * np.log1p(g.fr)
                out[(k, t)] = g.loc[score.idxmax()]
    return out


def unit_events(row):
    """spike times of one unit (artefact-corrected, as 051) and corrected first-lick times per class (ref = FA or SL)"""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{row.session_id}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = m51.ru.process_nwb_tables(nwb)
        trials_raw = nwb.trials.to_dataframe()
        t, log = m51.select_trials(trials_raw)
        ev = {c: t.first_lick_time[t.cls == c].to_numpy() for c in ["WH", "AH"]}
        ev["FA"] = (m51.spontaneous_licks(nwb, trials_raw, log["epoch"]) if m51.REF == "sl"
                    else t.first_lick_time[t.cls == "FA"].to_numpy())
    u = units[(units.electrode_group.astype(str) == str(row.electrode_group)) &
              (units.cluster_id.astype(str) == str(row.cluster_id))]
    return np.sort(np.asarray(u.spike_times.iloc[0])), ev


def raster_psth(fig, spec, row, rng, title, color):
    s, ev = unit_events(row)
    sub = spec.subgridspec(2, 1, height_ratios=[1, 1.1], hspace=0.05)
    ax_p, ax_r = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    edges = np.arange(RASTER_WIN[0], RASTER_WIN[1] + 0.005, 0.01); tc = (edges[:-1] + 0.005) * 1e3
    from scipy.ndimage import gaussian_filter1d
    y0 = 0
    for c in ["FA", "AH", "WH"]:
        e = ev[c]
        if len(e) == 0:
            continue
        rel = [s[np.searchsorted(s, x + RASTER_WIN[0]):np.searchsorted(s, x + RASTER_WIN[1])] - x for x in e]
        H = np.array([np.histogram(r, edges)[0] for r in rel]) / 0.01
        H = gaussian_filter1d(H, 2, axis=1)
        mu, se = H.mean(0), H.std(0) / np.sqrt(len(H))
        ax_p.fill_between(tc, mu - se, mu + se, color=CL[c], alpha=0.25, lw=0, edgecolor="none")
        ax_p.plot(tc, mu, color=CL[c], lw=0.9)
        pick = np.sort(rng.choice(len(rel), min(len(rel), RASTER_MAX), replace=False))
        for j, i in enumerate(pick):
            ax_r.vlines(rel[i] * 1e3, y0 + j, y0 + j + 0.9, color=CL[c], lw=0.35)
        y0 += len(pick) + 2
    for ax in (ax_p, ax_r):
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.5, lw=0, edgecolor="none", zorder=0)
        ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(RASTER_WIN[0] * 1e3, RASTER_WIN[1] * 1e3)
    ax_p.set_xticklabels([]); ax_r.set_yticks([]); ax_r.set_ylim(y0, -1)
    ax_r.spines["left"].set_visible(False)
    ax_r.set_xticks([-400, -200, 0, 200]); ax_r.tick_params(labelsize=4.8); ax_p.tick_params(labelsize=4.8)
    ax_p.set_title(title, fontsize=5.0, color=color, pad=2)
    return ax_p, ax_r


def schematic(ax, kind, RA):
    """schematic definition of one single-neuron quantification (drawn in axes coordinates)"""
    from matplotlib.patches import Circle
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    x = np.linspace(0.05, 0.95, 200)
    g = lambda m, s_: np.exp(-0.5 * ((x - m) / s_) ** 2)
    if kind in ("frac_wh", "frac_whah"):                # per-unit ROC between two event types
        c0, l0 = ("FA", RA) if kind == "frac_wh" else ("AH", "AH")
        ax.fill_between(x, 0.22, 0.22 + 0.34 * g(0.38, 0.09), color=CL[c0], alpha=0.35, lw=0, edgecolor="none")
        ax.fill_between(x, 0.22, 0.22 + 0.34 * g(0.62, 0.09), color=CL["WH"], alpha=0.55, lw=0, edgecolor="none")
        ax.plot([0.05, 0.95], [0.22, 0.22], color="0.3", lw=0.6)
        ax.text(0.30, 0.6, l0, ha="center", fontsize=5, color=CL[c0]); ax.text(0.72, 0.6, "WH", ha="center", fontsize=5, color="#b07e00")
        ax.text(0.5, 0.08, "pre-lick rate of one unit", ha="center", fontsize=4.6, color="0.3")
        ax.text(0.5, 1.0, f"Unit counted if ROC WH vs {l0}\nsignificant (label permutation)", ha="center", va="top", fontsize=4.6)
    elif kind == "conv":                                # converging = overlap of reward-lick and WH-selective
        ax.add_patch(Circle((0.4, 0.45), 0.24, fc=CL["AH"], alpha=0.25, ec="none"))
        ax.add_patch(Circle((0.62, 0.45), 0.24, fc=CL["WH"], alpha=0.35, ec="none"))
        ax.text(0.27, 0.45, f"AH ≠ {RA}", ha="center", va="center", fontsize=4.6, color=CL["AH"])
        ax.text(0.76, 0.45, f"WH ≠ {RA}", ha="center", va="center", fontsize=4.6, color="#b07e00")
        ax.text(0.51, 0.45, "same\nsign", ha="center", va="center", fontsize=4.2)
        ax.text(0.5, 0.92, "Converging = overlap / reward-lick\nneurons (AH ≠ ref)", ha="center", va="top", fontsize=4.6)
    elif kind == "like":                                # AH-likeness on the reference -> AH line
        ax.plot([0.12, 0.88], [0.45, 0.45], color="0.3", lw=0.8)
        for xx, c, t in [(0.12, "FA", RA), (0.88, "AH", "AH")]:
            ax.scatter(xx, 0.45, s=28, color=CL[c], zorder=3); ax.text(xx, 0.3, t, ha="center", fontsize=5, color=CL[c])
        ax.scatter(0.66, 0.45, s=28, color=CL["WH"], zorder=4); ax.text(0.66, 0.56, "WH", ha="center", fontsize=5, color="#b07e00")
        ax.text(0.5, 0.92, "c = (|WH−ref| − |WH−AH|) / |AH−ref|\n−1: like ref, +1: like AH (per neuron)", ha="center",
                va="top", fontsize=4.4)
        ax.text(0.12, 0.12, "−1", ha="center", fontsize=4.6); ax.text(0.88, 0.12, "+1", ha="center", fontsize=4.6)
    elif kind == "shared":                              # correlation of selectivities across units
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 60); b = 0.7 * a + rng.normal(0, 0.6, 60)
        ax.scatter(0.5 + a * 0.12, 0.42 + b * 0.12, s=2, color="0.3", lw=0)
        ax.plot([0.2, 0.8], [0.12, 0.72], color="0.5", lw=0.6, ls=(0, (2, 2)))
        ax.text(0.5, 0.92, f"Spearman r across units of\nsel(AH vs {RA}) and sel(WH vs {RA})", ha="center", va="top", fontsize=4.6)
        ax.text(0.5, 0.03, f"x: AH vs {RA}, y: WH vs {RA}", ha="center", fontsize=4.2, color="0.3")
    return ax


def schematics(fig, specs, RA, kinds=("frac_wh", "conv", "like", "shared")):
    """row of schematic definitions of the single-neuron quantifications"""
    return [schematic(fig.add_subplot(sp), k, RA) for sp, k in zip(specs, kinds)]


# ------------------------------------------------------------------ CCF slab maps (Fig 2g, Fig 2S)
CCF_SLAB = 500                      # slab thickness (um): a neuron belongs to a slab if within 250 um of its centre
CCF_BIN, CCF_SIGMA = 50, 150        # density grid (um) and Gaussian smoothing (um)
CCF_MID = 5700                      # CCF midline (ML, um); hemispheres folded onto the right side
CCF_COR = list(range(2500, 11001, 500))     # candidate coronal slab centres (AP, um)
CCF_SAG = list(range(6000, 11001, 500))     # candidate sagittal slab centres (folded ML, um)


def ccf_slices(cache_dir, specs):
    """atlas region boundaries + brain mask for coronal ('cor', AP) / sagittal ('sag', ML) slices; cached (npz)"""
    from scipy import ndimage
    f = cache_dir / "ccf_slab_slices.npz"
    cache = dict(np.load(f)) if f.exists() else {}
    need = [sp for sp in specs if f"{sp[0]}_{sp[1]}_b" not in cache]
    if need:
        import tifffile
        m49 = importlib.import_module("049_roc_stage_ccf")
        ann = tifffile.imread(m49.ATLAS / "annotation.tiff")          # (AP, DV, ML), 10 um
        pm = m49.structure_parent_map()
        for kind, c in need:
            i = int(round(c / 10))
            a = ann[min(i, ann.shape[0] - 1)] if kind == "cor" else ann[:, :, min(i, ann.shape[2] - 1)].T   # (DV, ML|AP)
            u, inv = np.unique(a, return_inverse=True)
            a = np.array([pm.get(int(k), int(k)) for k in u])[inv].reshape(a.shape)
            bnd = (np.diff(a, axis=0, prepend=a[:1]) != 0) | (np.diff(a, axis=1, prepend=a[:, :1]) != 0)
            cache[f"{kind}_{c}_b"] = ndimage.binary_dilation(bnd & (a != 0))
            cache[f"{kind}_{c}_i"] = ndimage.binary_fill_holes(a != 0)
        np.savez_compressed(f, **cache)
    return {sp: (cache[f"{sp[0]}_{sp[1]}_b"], cache[f"{sp[0]}_{sp[1]}_i"]) for sp in specs}


def ccf_coords(W):
    """folded ML, AP, DV (um) of units"""
    ml = W.ccf_atlas_ml.to_numpy(float)
    return CCF_MID + np.abs(ml - CCF_MID), W.ccf_atlas_ap.to_numpy(float), W.ccf_atlas_dv.to_numpy(float)


def ccf_in_slab(W, sp):
    ml, ap, dv = ccf_coords(W)
    kind, c = sp
    m = np.abs((ap if kind == "cor" else ml) - c) <= CCF_SLAB / 2
    x = ml if kind == "cor" else ap
    return m & np.isfinite(x) & np.isfinite(dv), x, dv


def converging_mask(W):
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    tested = (W[f"sig:{af}"].notna() & W[f"sig:{wf}"].notna()).to_numpy()
    conv = tested & (W[f"sig:{af}"] == 1).to_numpy() & (W[f"sig:{wf}"] == 1).to_numpy() & \
        (np.sign(W[f"sel:{af}"]) == np.sign(W[f"sel:{wf}"])).to_numpy()
    return tested, conv


def ccf_pick_slabs(W, n_cor, n_sag, min_units=200, min_gap=1000):
    """slabs with the largest cohort difference in the fraction of converging neurons (expert sessions), weighted by
    sampling: score = |f(R+) - f(R-)| * sqrt(min(n_R+, n_R-)), both cohorts >= min_units tested; greedy, >= min_gap apart"""
    tested, conv = converging_mask(W)
    ex = (W.stage == "expert").to_numpy()
    rows = []
    for kind, cands in [("cor", CCF_COR), ("sag", CCF_SAG)]:
        for c in cands:
            m, _, _ = ccf_in_slab(W, (kind, c))
            r = dict(kind=kind, c=c)
            for coh in ["R+", "R-"]:
                g = m & tested & ex & (W.cohort == coh).to_numpy()
                r[f"n {coh}"], r[f"f {coh}"] = int(g.sum()), (conv[g].mean() if g.sum() else np.nan)
            r["score"] = abs(r["f R+"] - r["f R-"]) * np.sqrt(min(r["n R+"], r["n R-"])) \
                if min(r["n R+"], r["n R-"]) >= min_units else -1
            rows.append(r)
    R = pd.DataFrame(rows)
    pick = []
    for kind, n in [("cor", n_cor), ("sag", n_sag)]:
        chosen = []
        for r in R[R.kind == kind].sort_values("score", ascending=False).itertuples():
            if r.score > 0 and all(abs(r.c - c) >= min_gap for c in chosen) and len(chosen) < n:
                chosen.append(r.c)
        pick += [(kind, c) for c in sorted(chosen)]
    return pick, R


def ccf_grid(fig, spec, W, slabs, slices, rng, label_rows=True, cbar=True, title_size=4.6):
    """4 rows x len(slabs): sampled (tested) neurons R+, converging-neuron density R+, sampled R-, density R- (expert
    sessions). Density = converging neurons per mm2 of the projected slab (Gaussian-smoothed 2D histogram), one shared
    discrete colour scale for both cohorts."""
    from scipy import ndimage
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    tested, conv = converging_mask(W)
    ex = (W.stage == "expert").to_numpy()
    shape_ml, shape_ap, shape_dv = 11400, 13200, 8000
    H, ext = {}, {}
    for sp in slabs:
        m, x, y = ccf_in_slab(W, sp)
        xmax = shape_ml if sp[0] == "cor" else shape_ap
        bx, by = np.arange(0, xmax + CCF_BIN, CCF_BIN), np.arange(0, shape_dv + CCF_BIN, CCF_BIN)
        for coh in ["R+", "R-"]:
            g = m & conv & ex & (W.cohort == coh).to_numpy()
            h, _, _ = np.histogram2d(y[g], x[g], bins=[by, bx])
            H[(sp, coh)] = ndimage.gaussian_filter(h, CCF_SIGMA / CCF_BIN) / (CCF_BIN / 1000) ** 2
        ext[sp] = (bx[0], bx[-1], by[-1], by[0])
    vmax = np.nanpercentile(np.concatenate([h.ravel() for h in H.values()]), 99.8)
    step = max(1, int(np.ceil(vmax / 6 / 5)) * 5) if vmax > 10 else max(1, int(np.ceil(vmax / 6)))
    levels = np.arange(step, step * 7 + 1, step)
    grid = spec.subgridspec(4, len(slabs), wspace=0.03, hspace=0.04)
    axs = np.empty((4, len(slabs)), object)
    for j, sp in enumerate(slabs):
        bnd, inb = slices[sp]
        m, x, y = ccf_in_slab(W, sp)
        for i, (coh, kind_) in enumerate([("R+", "pts"), ("R+", "den"), ("R-", "pts"), ("R-", "den")]):
            ax = fig.add_subplot(grid[i, j]); axs[i, j] = ax
            cm = LinearSegmentedColormap.from_list(coh, ["#FFFFFF", COH[coh], {"R+": "#005A00", "R-": "#640064"}[coh]])
            if kind_ == "pts":
                g = m & tested & ex & (W.cohort == coh).to_numpy()
                ax.scatter(x[g], y[g], s=0.3, color=COH[coh], alpha=0.3, lw=0, rasterized=True, zorder=4)
            else:
                cmap = cm.resampled(len(levels) + 1)
                Hm = np.ma.masked_less(H[(sp, coh)], levels[0])
                ax.imshow(Hm, cmap=cmap, norm=BoundaryNorm(np.r_[levels, levels[-1] * 10], cmap.N, extend="neither"),
                          extent=ext[sp], interpolation="nearest", zorder=1)
                ax.contour(np.arange(0, ext[sp][1], CCF_BIN) + CCF_BIN / 2, np.arange(0, ext[sp][2], CCF_BIN) + CCF_BIN / 2,
                           H[(sp, coh)], levels=levels, colors=[COH[coh]], linewidths=0.25, zorder=1.5)
            rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.6; rgba[..., 3] = bnd * 0.5
            ax.imshow(rgba, extent=(0, bnd.shape[1] * 10, bnd.shape[0] * 10, 0), interpolation="antialiased", zorder=2)
            ax.contour((np.arange(inb.shape[1]) + 0.5) * 10, (np.arange(inb.shape[0]) + 0.5) * 10, inb.astype(float),
                       levels=[0.5], colors="0.3", linewidths=0.4, zorder=3)
            if sp[0] == "cor":
                ax.set_xlim(CCF_MID - 200, shape_ml)
            else:
                ax.set_xlim(500, shape_ap - 700)
            ax.set_ylim(shape_dv - 1000, 0); ax.set_aspect("equal"); ax.axis("off")
            if i == 0:
                ax.set_title(("AP" if sp[0] == "cor" else "ML") + f" {(sp[1] - (0 if sp[0] == 'cor' else CCF_MID)) / 1000:.1f} mm",
                             fontsize=title_size, pad=1.5)
            if j == 0 and label_rows:
                n = int((tested & ex & (W.cohort == coh).to_numpy()).sum())
                ax.text(-0.04, 0.5, {"pts": f"{coh.replace('-', '−')} expert\nsampled", "den": f"{coh.replace('-', '−')} expert\nconverging / mm²"}[kind_],
                        transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=4.6, color=COH[coh])
    if cbar:
        for k_, coh in enumerate(["R+", "R-"]):
            cm = LinearSegmentedColormap.from_list(coh, ["#FFFFFF", COH[coh], {"R+": "#005A00", "R-": "#640064"}[coh]])
            cmap = cm.resampled(len(levels) + 1)
            cax = axs[1 + 2 * k_, -1].inset_axes([1.04, 0.1, 0.05, 0.8])
            cb = fig.colorbar(plt_cm_mappable(cmap, levels), cax=cax, ticks=levels[::2])
            cb.ax.tick_params(labelsize=4); cb.outline.set_linewidth(0.3)
            if k_ == 1:
                cb.set_label("Converging neurons\nper mm²", fontsize=4.4)
    STATS.append(dict(panel="2g CCF", measure="converging neurons per mm2 levels", levels=",".join(str(v) for v in levels),
                      slabs=";".join(f"{a}{b}" for a, b in slabs)))
    return axs


def plt_cm_mappable(cmap, levels):
    import matplotlib.cm as mcm
    from matplotlib.colors import BoundaryNorm
    return mcm.ScalarMappable(norm=BoundaryNorm(np.r_[levels, levels[-1] * 10], cmap.N), cmap=cmap)


def selectivity_density(ax, W, coh, RA, rng, B=2000):
    """units of one cohort (learning + expert merged) as a log-density; separate OLS fits per stage; Pearson r per stage
    and the difference r(expert) - r(learning) with a session-level bootstrap (sessions resampled within stage)"""
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    g = W[W.cohort == coh][["session_id", "stage", f"sel:{af}", f"sel:{wf}"]].dropna()
    ax.hexbin(g[f"sel:{af}"], g[f"sel:{wf}"], gridsize=40, extent=(-1, 1, -1, 1), bins="log", cmap="Greys",
              mincnt=1, linewidths=0, rasterized=True)
    res = {}
    for st, ls in [("learning", (0, (3, 1.5))), ("expert", "-")]:
        d = g[g.stage == st]
        x, y = d[f"sel:{af}"].to_numpy(), d[f"sel:{wf}"].to_numpy()
        sl, ic = np.polyfit(x, y, 1)
        xx = np.array([-1, 1])
        ax.plot(xx, ic + sl * xx, color=COH[coh], lw=1.2 if st == "expert" else 1.0, ls=ls, alpha=1 if st == "expert" else 0.75)
        # sufficient statistics per session -> fast session bootstrap of the pooled Pearson r
        d = d.assign(x=x, y=y, xx=x * x, yy=y * y, xy=x * y, n=1)
        T = d.groupby("session_id")[["n", "x", "y", "xx", "yy", "xy"]].sum().to_numpy()
        res[st] = dict(T=T, r=stats.pearsonr(x, y)[0], slope=sl, n=len(x), ns=len(T))
    def r_of(T):
        n, sx, sy, sxx, syy, sxy = T.sum(0)
        return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
    bl = np.array([r_of(res["learning"]["T"][rng.integers(0, res["learning"]["ns"], res["learning"]["ns"])]) for _ in range(B)])
    be = np.array([r_of(res["expert"]["T"][rng.integers(0, res["expert"]["ns"], res["expert"]["ns"])]) for _ in range(B)])
    dr = be - bl; obs = res["expert"]["r"] - res["learning"]["r"]
    p = min(1.0, 2 * min((dr <= 0).mean(), (dr >= 0).mean()))
    p = max(p, 1 / B)
    ax.plot([-1, 1], [-1, 1], color="0.55", lw=0.5, ls=(0, (1, 1.5)))
    ax.axhline(0, color="0.8", lw=0.4); ax.axvline(0, color="0.8", lw=0.4)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.text(0.03, 0.97, f"learning r = {res['learning']['r']:.2f} (slope {res['learning']['slope']:.2f})\n"
                        f"expert r = {res['expert']['r']:.2f} (slope {res['expert']['slope']:.2f})\n"
                        f"Δr = {obs:+.2f}, {fmt_p(p)}".replace("-", "−"), transform=ax.transAxes, va="top", fontsize=4.5,
            color=COH[coh], bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
    ax.set_xlabel(f"Selectivity AH vs {RA}"); ax.set_ylabel(f"Selectivity WH vs {RA}")
    ax.set_title(f"{coh.replace('-', '−')}: learning (dashed) and expert (solid) fits", color=COH[coh], fontsize=5.4)
    STATS.append(dict(panel=f"2f {coh}", measure="Pearson r WH vs ref ~ AH vs ref selectivity",
                      r_learning=res["learning"]["r"], r_expert=res["expert"]["r"], slope_learning=res["learning"]["slope"],
                      slope_expert=res["expert"]["slope"], dr=obs, dr_lo=np.percentile(dr, 2.5), dr_hi=np.percentile(dr, 97.5),
                      p_boot=p, n_units_learning=res["learning"]["n"], n_units_expert=res["expert"]["n"],
                      n_sessions_learning=res["learning"]["ns"], n_sessions_expert=res["expert"]["ns"]))
    return ax


def converging_area_table(W):
    """expert sessions: per session x area (area_acronym_custom) the fraction of tested units that are converging
    neurons; areas kept if present in expert sessions of both cohorts (no minimum count)"""
    tested, conv = converging_mask(W)
    E = W.assign(tested=tested, conv=conv)
    E = E[(E.stage == "expert") & E.tested & E.area_acronym_custom.notna()]
    T = E.groupby(["mouse_id", "session_id", "cohort", "area_group", "area_acronym_custom"], as_index=False).agg(
        n=("conv", "size"), k=("conv", "sum"))
    T["frac"] = T.k / T.n
    shared = set(T[T.cohort == "R+"].area_acronym_custom) & set(T[T.cohort == "R-"].area_acronym_custom)
    return T[T.area_acronym_custom.isin(shared)].rename(columns={"area_acronym_custom": "area"})


def anova_cohort_area(T, n_perm=1000, seed=0):
    """frac ~ C(cohort) * C(area), type II OLS on session x area values; p also from permuting cohort labels across
    mice (sessions contribute several areas)"""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    terms = {"cohort": "C(cohort)", "area": "C(area)", "cohort x area": "C(cohort):C(area)"}
    def fit(d):
        tab = anova_lm(smf.ols("frac ~ C(cohort) * C(area)", data=d).fit(), typ=2)
        return {k: (tab.loc[v, "F"], tab.loc[v, "PR(>F)"]) for k, v in terms.items()}
    obs = fit(T)
    rng = np.random.default_rng(seed)
    mice = T.mouse_id.unique(); mc = T.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    midx = pd.Index(mice).get_indexer(T.mouse_id)
    null = {k: [] for k in terms}
    for _ in range(n_perm):
        d = T.copy(); d["cohort"] = rng.permutation(mc)[midx]
        try:
            r = fit(d)
        except Exception:
            continue
        for k in terms:
            null[k].append(r[k][0])
    return {k: dict(F=obs[k][0], p=obs[k][1], p_perm=(1 + np.sum(np.array(null[k]) >= obs[k][0])) / (1 + len(null[k])))
            for k in terms}


def converging_area_bars(ax, W, rng, out):
    import ephys_utilities.allen_utils.allen_utils as au
    T = converging_area_table(W)
    T.to_csv(out / "fig2g_converging_by_area.csv", index=False)
    gord = {g: i for i, g in enumerate(au.get_area_group_custom_order())}
    areas = (T.drop_duplicates("area").assign(o=lambda d: d.area_group.map(gord).fillna(99))
             .sort_values(["o", "area"]).area.tolist())
    for k_, c in enumerate(["R+", "R-"]):
        xs = np.arange(len(areas)) + (k_ - 0.5) * 0.38
        g = T[T.cohort == c].groupby("area").frac
        mu, se = g.mean().reindex(areas), g.sem().reindex(areas)
        ax.bar(xs, mu * 100, 0.36, color=COH[c], alpha=0.85 if c == "R+" else 0.75, lw=0, label=c.replace("-", "−") + " expert")
        ax.errorbar(xs, mu * 100, se * 100, fmt="none", ecolor="0.25", elinewidth=0.5, capsize=0)
        for i, a in enumerate(areas):
            vv = T[(T.cohort == c) & (T.area == a)].frac.to_numpy() * 100
            ax.scatter(xs[i] + rng.uniform(-0.08, 0.08, len(vv)), vv, s=1.5, color="0.2", lw=0, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(areas)), areas, rotation=90, fontsize=4.2)
    ax.set_xlim(-0.7, len(areas) - 0.3)
    ax.set_ylabel("Converging neurons\n(% of tested units)")
    # area-group separators and labels
    grp = T.drop_duplicates("area").set_index("area").area_group.reindex(areas).tolist()
    for i in range(1, len(areas)):
        if grp[i] != grp[i - 1]:
            ax.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)
    AN = anova_cohort_area(T)
    for k, v_ in AN.items():
        STATS.append(dict(panel="2g ANOVA", measure=k, F=v_["F"], p=v_["p"], p_perm=v_["p_perm"], n_areas=len(areas),
                          n_sessions=T.session_id.nunique()))
    ax.set_title(f"Converging neurons per area (expert sessions; {len(areas)} areas recorded in both cohorts; "
                 f"mean ± s.e.m. over sessions)\nANOVA (session × area values; p by permuting cohort labels across mice): "
                 f"cohort F = {AN['cohort']['F']:.1f}, {fmt_p(AN['cohort']['p_perm'])}; "
                 f"cohort × area F = {AN['cohort x area']['F']:.2f}, {fmt_p(AN['cohort x area']['p_perm'])}",
                 fontsize=5.4, loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    return ax


def fig2(plt, D, out, pop, rng):
    """Figure 2 (layout 2026-10-07): a square example panels (PSTH + raster); b-e schematics above the session
    quantifications (WH vs ref selective units, WH vs AH selective units, converging neurons, AH-likeness); f-i shared
    hit code on one row (schematic, Spearman r per session, selectivity scatter R+ and R-); j converging neurons per area"""
    W = D["W"]
    S = session_unit_metrics(W)
    RA = "FA" if m51.REF == "fa" else "SL"
    cell = (W_IN * 0.89 - 2 * 0.3) / 3                 # width of one example panel (in); PSTH + raster block is square
    H = 2 * cell * 1.12 + 0.7 + 1.0 + 1.55 + 1.75 + 1.6 + 1.2
    fig = plt.figure(figsize=(W_IN, H))
    gs = fig.add_gridspec(5, 1, height_ratios=[2 * cell * 1.12 + 0.7, 1.0, 1.55, 1.75, 1.6], hspace=0.42, left=0.08, right=0.97,
                          top=1 - 0.55 / H, bottom=0.45 / H)
    ex = pick_examples(W)
    ga = gs[0].subgridspec(2, 3, hspace=0.32, wspace=0.28)
    axs_a = []
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    for i, k in enumerate([GROUPS[1], GROUPS[3]]):
        for j, (t, lab) in enumerate(EX_TYPES):
            if (k, t) not in ex:
                ax = fig.add_subplot(ga[i, j]); ax.axis("off")
                ax.text(0.5, 0.5, f"{GLAB[k]}: no example\nmeeting the criteria", ha="center", va="center", fontsize=5,
                        color="0.4", transform=ax.transAxes)
                continue
            r = ex[(k, t)]
            ttl = (f"{GLAB[k]} · {lab.format(r=RA).splitlines()[0]}\n{r.area_acronym_custom}, "
                   f"AH vs {RA} {r[f'sel:{af}']:+.2f}, WH vs {RA} {r[f'sel:{wf}']:+.2f}")
            ap, ar = raster_psth(fig, ga[i, j], r, rng, ttl, COH[k[0]])
            axs_a.append(ap)
            if j == 0:
                ap.set_ylabel("Rate (Hz)", fontsize=5.2); ar.set_ylabel("Events", fontsize=5.2)
            ar.set_xlabel("Time from first lick (ms)", fontsize=5.0)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=CL[c], lw=1.2) for c in ["WH", "AH", "FA"]],
               [CLAB[c] for c in ["WH", "AH", "FA"]], loc="upper right", bbox_to_anchor=(0.98, 1 - 0.18 / H), ncol=3,
               frameon=False, fontsize=5.2)
    # b-e: schematics (row 2) above the session values (row 3)
    gsch = gs[1].subgridspec(1, 4, wspace=0.35)
    axs_s = schematics(fig, [gsch[i] for i in range(4)], RA, kinds=("frac_wh", "frac_whah", "conv", "like"))
    gq = gs[2].subgridspec(1, 4, wspace=0.6)
    ax_b = fig.add_subplot(gq[0])
    dots_panel(ax_b, S, "frac_sig_WHvsFA", "Fraction of units", rng, "2b", f"WH vs {RA} selective units")
    ax_c = fig.add_subplot(gq[1])
    dots_panel(ax_c, S, "frac_sig_WHvsAH", "Fraction of units", rng, "2c", "WH vs AH selective units")
    T = D["tr"]; T = T[(T.level == "all") & (T.variant == "all")]
    ax_d = fig.add_subplot(gq[2])
    dots_panel(ax_d, T, "frac_transfer", f"Fraction of AH-vs-{RA} neurons", rng, "2d", "Converging neurons")
    ax_e = fig.add_subplot(gq[3])
    dots_panel(ax_e, T, "likeness", f"AH-likeness (−1 {RA}, +1 AH)", rng, "2e", "AH-likeness of\nreward-lick neurons",
               ref=[(0, "0.6")])
    for a_ in (ax_b, ax_c, ax_d, ax_e):
        a_.set_title(a_.get_title(), fontsize=5.5)
    # f-i: shared hit code on one row
    gf = gs[3].subgridspec(1, 4, wspace=0.55, width_ratios=[0.9, 1, 1, 1])
    ax_fs = schematic(fig.add_subplot(gf[0]), "shared", RA)
    ax_g = fig.add_subplot(gf[1])
    dots_panel(ax_g, S, "r_shared", "Spearman r", rng, "2g", "Shared hit code")
    ax_g.set_title(ax_g.get_title(), fontsize=5.5)
    axs_hi = [selectivity_density(fig.add_subplot(gf[2 + k_]), W, coh, RA, rng) for k_, coh in enumerate(["R+", "R-"])]
    ax_j = converging_area_bars(fig.add_subplot(gs[4]), W, rng, out)
    letter_row(fig, axs_a[:1], "a", dy_in=0.12)
    letter_row(fig, axs_s, "bcde", dy_in=0.05)
    letter_row(fig, [ax_fs, ax_g] + list(axs_hi), "fghi", dy_in=0.12)
    letter_row(fig, [ax_j], "j", dy_in=0.3)
    fig.suptitle("Figure 2 | Single neurons: in R+ mice, reward-lick neurons come to treat whisker hits like auditory hits",
                 x=0.08, y=1 - 0.02 / H, ha="left", va="top", fontsize=7.5, weight="bold")
    save(fig, out, "Fig2_single_neurons"); plt.close(fig)
    pd.DataFrame([dict(group=GLAB[k], type=t, session_id=r.session_id, electrode_group=r.electrode_group,
                       cluster_id=r.cluster_id, area=r.area_acronym_custom) for (k, t), r in ex.items()]).to_csv(
        out / "fig2_examples.csv", index=False)


def fig2_old_examples(plt, D, out, pop, sf=None):
    """previous Fig 2a (kept as supplementary): PSTH-only examples of reward-lick neurons (good units, mean pre-lick rate
    >= 8 Hz, AH vs ref selectivity >= 0.6): two R+ expert neurons that also separate WH from the reference (same sign),
    two R- expert neurons that do not (|WH vs ref| < 0.1); 10 ms bins, 50 ms boxcar, rate minus the unit's mean in
    [-600, -400] ms"""
    W, P, tc = D["W"], D["P"], D["tc"]
    RA = "FA" if m51.REF == "fa" else "SL"
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    cand = W[(W.quality_label == "good") & (W[f"sig:{af}"] == 1) & (W[f"sel:{af}"] > 0) & (W[f"fr_window:{af}"] >= 8)]
    picks = []
    for (c, s_), want in [(("R+", "expert"), True), (("R-", "expert"), False)]:
        g = cand[(cand.cohort == c) & (cand.stage == s_)]
        g = g[(g[f"sig:{wf}"] == 1) & (g[f"sel:{wf}"] > 0)] if want else g[(g[f"sig:{wf}"] != 1) & (g[f"sel:{wf}"].abs() < 0.1)]
        g = g[g[f"sel:{af}"] >= 0.6].sort_values(f"fr_window:{af}", ascending=False)
        picks += [r for _, r in g.drop_duplicates("session_id").head(2).iterrows()]
    fig = sf if sf is not None else plt.figure(figsize=(W_IN, 1.9))
    axs = fig.subplots(1, 4, gridspec_kw=dict(wspace=0.45))
    for j, (ax, r) in enumerate(zip(axs, picks[:4])):
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0, edgecolor="none"); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        for kk, c in enumerate(m51.CLASSES):
            ax.plot(tc, np.convolve(P[int(r.row), kk], np.ones(2) / 2, "same"), color=CL[c], lw=0.9, label=CLAB[c])
        ax.set_xlim(-550, 350); ax.set_xticks([-400, -200, 0, 200]); ax.set_xlabel("Time from first lick (ms)")
        ax.set_title(f"{GLAB[(r.cohort, r.stage)]}, {r.area_acronym_custom}\nAH vs {RA} {r[f'sel:{af}']:+.2f}, "
                     f"WH vs {RA} {r[f'sel:{wf}']:+.2f}", color=COH[r.cohort], fontsize=5.6)
        if j == 0:
            ax.set_ylabel("Δ firing rate (Hz)"); ax.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=5)
    fig.suptitle("a  Example reward-lick neurons (PSTH only; 10 ms bins, 20 ms boxcar)", x=0.01, y=0.99, ha="left",
                 fontsize=6.8, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.22)
    if sf is None:
        save(fig, out, "Fig2S_examples_psth_only"); plt.close(fig)


# main-ROC types used to characterise converging neurons (045 sign convention: choice types flipped so positive =
# hit > miss; wh_vs_aud positive = auditory > whisker; response types positive = excited)
TYPE_ROWS = [("whisker_active", "pos", "Whisker-responsive (active), excited"),
             ("whisker_active", "neg", "Whisker-responsive (active), suppressed"),
             ("auditory_active", "pos", "Auditory-responsive (active), excited"),
             ("auditory_active", "neg", "Auditory-responsive (active), suppressed"),
             ("whisker_passive_pre", "sig", "Whisker-responsive (passive, pre)"),
             ("auditory_passive_pre", "sig", "Auditory-responsive (passive, pre)"),
             ("wh_vs_aud_active", "neg", "Prefers whisker (active stimulus)"),
             ("wh_vs_aud_active", "pos", "Prefers auditory (active stimulus)"),
             ("whisker_choice", "pos", "Whisker hit > miss"),
             ("whisker_choice", "neg", "Whisker miss > hit"),
             ("auditory_choice", "pos", "Auditory hit > miss"),
             ("auditory_choice", "neg", "Auditory miss > hit"),
             ("spontaneous_licks", "sig", "Spontaneous-lick responsive"),
             ("spontaneous_licks_vs_cr", "pos", "Lick > correct rejection (motor)"),
             ("baseline_choice", "sig", "Pre-stimulus lick vs no-lick")]


def fig2_types(plt, D, out, pop, sf=None):
    """what kind of neurons are the converging neurons? fraction significant in the main stimulus-aligned ROC types"""
    W = D["W"]
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    tested = W[f"sig:{af}"].notna() & W[f"sig:{wf}"].notna()
    gA, gW = W[f"sig:{af}"] == 1, W[f"sig:{wf}"] == 1
    same = np.sign(W[f"sel:{af}"]) == np.sign(W[f"sel:{wf}"])
    subsets = {"Converging neurons": tested & gA & gW & same, "Reward-lick-only neurons": tested & gA & ~gW,
               "All tested units": tested}
    KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
    L = pd.read_parquet(m51.RES / "_roc_stage_analysis" / "roc_long.parquet")
    L = L[L.analysis_type.isin({t for t, _, _ in TYPE_ROWS})]
    L["cluster_id"] = L.cluster_id.astype(str); L["electrode_group"] = L.electrode_group.astype(str)
    Wk = W[KEYS].astype(str).reset_index()
    M = {}
    for col in ["sig", "pos", "neg"]:
        pv = L.pivot_table(index=KEYS, columns="analysis_type", values=col, aggfunc="first")
        M[col] = Wk.merge(pv.reset_index(), on=KEYS, how="left").set_index("index").reindex(W.index)
    rows = []
    for sname, sm in subsets.items():
        for k in GROUPS:
            gm = (sm & (W.cohort == k[0]) & (W.stage == k[1])).to_numpy()
            for t, sgn, lab in TYPE_ROWS:
                v = M["sig"][t].to_numpy()[gm]; ok = ~pd.isna(v)
                f = M[sgn][t].to_numpy()[gm][ok].astype(float)
                rows.append(dict(subset=sname, group=GLAB[k], type=t, sign=sgn, label=lab, n_tested=int(ok.sum()),
                                 n_subset=int(gm.sum()), n_sessions=int(W.session_id[gm].nunique()),
                                 frac=f.mean() if ok.sum() >= 10 else np.nan))
    R = pd.DataFrame(rows); R.to_csv(out / "fig2_types_breakdown.csv", index=False)
    cols = [GLAB[k] for k in GROUPS]; labs = [lab for _, _, lab in TYPE_ROWS]
    fig = sf if sf is not None else plt.figure(figsize=(W_IN, 4.5))
    axs = fig.subplots(1, 4, gridspec_kw=dict(wspace=0.1, width_ratios=[1, 1, 1, 1]))
    fig.subplots_adjust(left=0.22, right=0.83, top=0.87, bottom=0.25)
    vmax = np.nanpercentile(R.frac * 100, 98)
    for ax, sname in zip(axs[:3], subsets):
        Mx = R[R.subset == sname].pivot_table(index="label", columns="group", values="frac").reindex(index=labs, columns=cols) * 100
        im = ax.imshow(Mx.to_numpy(), cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                v = Mx.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.6, color="white" if v > vmax * 0.55 else "k")
        n = R[(R.subset == sname)].drop_duplicates("group").set_index("group").reindex(cols)
        ax.set_xticks(range(4), [f"{c}\n(n = {int(x)})" for c, x in zip(cols, n.n_subset)], rotation=45, ha="right", fontsize=4.8)
        for tk, k in zip(ax.get_xticklabels(), GROUPS):
            tk.set_color(COH[k[0]])
        ax.set_yticks(range(len(labs)), labs if ax is axs[0] else [], fontsize=5)
        ax.set_title(sname, fontsize=6)
    cax = axs[3].inset_axes([1.62, 0.0, 0.07, 1.0])
    cb = fig.colorbar(im, cax=cax); cb.set_label("% significant (main ROC)", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    C = R[R.subset == "Converging neurons"].pivot_table(index="label", columns="group", values="frac").reindex(index=labs, columns=cols)
    B = R[R.subset == "All tested units"].pivot_table(index="label", columns="group", values="frac").reindex(index=labs, columns=cols)
    E = np.log2((C + 1e-3) / (B + 1e-3))
    ax = axs[3]
    im2 = ax.imshow(E.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            v = E.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{2 ** v:.1f}×", ha="center", va="center", fontsize=4.4)
    ax.set_xticks(range(4), cols, rotation=45, ha="right", fontsize=4.8)
    for tk, k in zip(ax.get_xticklabels(), GROUPS):
        tk.set_color(COH[k[0]])
    ax.set_yticks(range(len(labs)), [])
    ax.set_title("Enrichment: converging\nvs all tested units", fontsize=6)
    cax2 = ax.inset_axes([1.06, 0.0, 0.07, 1.0])
    cb2 = fig.colorbar(im2, cax=cax2); cb2.set_label("log2 ratio", fontsize=5); cb2.ax.tick_params(labelsize=4.5)
    RA = "false alarms" if m51.REF == "fa" else "spontaneous licks"
    fig.suptitle(f"{'b  ' if sf is not None else ''}What are the converging neurons? Fraction significant in stimulus-aligned "
                 f"ROC types (whole brain; reference: {RA}; pooled over neurons)", x=0.01, y=0.99, ha="left", fontsize=6.8,
                 weight="bold")
    if sf is None:
        save(fig, out, "Fig2S_converging_neuron_types"); plt.close(fig)


def fig2_supplement(plt, D, out, pop, rng):
    """Fig 2S: (a) PSTH-only example reward-lick neurons, (b) converging-neuron type breakdown, (c, d) converging-neuron
    density on more coronal and sagittal slabs (expert sessions)"""
    W = D["W"]
    fig = plt.figure(figsize=(W_IN, 12.5))
    _, sfa, sfb, sfc, sfd = fig.subfigures(5, 1, height_ratios=[0.25, 1.9, 3.9, 3.4, 2.8], hspace=0.0)
    fig2_old_examples(plt, D, out, pop, sf=sfa)
    fig2_types(plt, D, out, pop, sf=sfb)
    _, SR = ccf_pick_slabs(W, 1, 1)
    ok = SR[(SR["n R+"] >= 100) & (SR["n R-"] >= 100)]
    cor = [("cor", int(c)) for c in ok[ok.kind == "cor"].c if (c - 3000) % 1000 == 0][:8]
    sag = [("sag", int(c)) for c in ok[ok.kind == "sag"].c][:6]
    slices = ccf_slices(BASE / "publication", cor + sag)
    for sf_, sl_, lab in [(sfc, cor, "c  Converging neurons, coronal slabs (500 µm; expert sessions; every 1 mm)"),
                          (sfd, sag, "d  Converging neurons, sagittal slabs (500 µm; expert sessions; hemispheres folded)")]:
        gsx = sf_.add_gridspec(1, 1, left=0.08, right=0.92, top=0.9, bottom=0.02)
        ccf_grid(sf_, gsx[0], W, sl_, slices, rng, title_size=4.4)
        sf_.suptitle(lab, x=0.01, y=0.99, ha="left", fontsize=6.8, weight="bold")
    fig.suptitle("Figure 2—supplement | Converging neurons: examples, functional types and anatomical distribution",
                 x=0.01, y=0.998, ha="left", fontsize=7.5, weight="bold")
    save(fig, out, "Fig2S_converging_neurons"); plt.close(fig)


# ------------------------------------------------------------------ Figure 3
DIST_LAB = {"dd": "Distance difference d(WH,ref) − d(WH,AH)", "d_WH_FA": "d(WH, ref)", "d_WH_AH": "d(WH, AH)",
            "d_AH_FA": "d(AH, ref)"}


def triangle(dWF, dWA, dAF):
    """2D positions (ref, AH, WH) whose pairwise distances are sqrt of the cross-validated squared distances"""
    rAF, rWF, rWA = (np.sqrt(max(v, 0)) for v in (dAF, dWF, dWA))
    if rAF <= 0:
        return None
    x = (rWF ** 2 - rWA ** 2 + rAF ** 2) / (2 * rAF)
    y = np.sqrt(max(rWF ** 2 - x ** 2, 0))
    return np.array([[0, 0], [rAF, 0], [x, y]])


def draw_triangles(ax, Lwb, cohort, RA):
    """group-mean geometry of the three class means for one cohort: learning (dashed) and expert (solid)"""
    for s_, ls, a_ in [("learning", (0, (2, 1.5)), 0.6), ("expert", "-", 1.0)]:
        d = Lwb[(Lwb.cohort == cohort) & (Lwb.stage == s_)]
        T = triangle(d.d_WH_FA.mean(), d.d_WH_AH.mean(), d.d_AH_FA.mean())
        if T is None:
            continue
        ax.plot(*np.r_[T, T[:1]].T, color=COH[cohort], lw=0.9, ls=ls, alpha=a_)
        for (x, y), c in zip(T, ["FA", "AH", "WH"]):
            ax.scatter(x, y, s=22 if s_ == "expert" else 12, color=CL[c], zorder=3,
                       edgecolor="none" if s_ == "expert" else CL[c], facecolor=CL[c] if s_ == "expert" else "white", lw=0.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{cohort.replace('-', '−')}: learning (dashed) → expert", color=COH[cohort], fontsize=5.6)


def share_triangle_limits(axs):
    """same data limits on all triangle axes so that R+ and R- geometries are drawn on one scale"""
    xl = (min(a.get_xlim()[0] for a in axs), max(a.get_xlim()[1] for a in axs))
    yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
    for a in axs:
        a.set_xlim(*xl); a.set_ylim(*yl)


def fig3(plt, D, out, pop, rng, PR):
    """main population figure: cross-validated distances between class-mean pre-lick population vectors"""
    L = D["lam"]; Lwb = L[(L.level == "all") & (L.variant == "all")]
    RA = "FA" if m51.REF == "fa" else "SL"
    RN = "false alarms" if m51.REF == "fa" else "spontaneous licks"
    fig = plt.figure(figsize=(W_IN, 6.3))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 1.0], hspace=0.95, wspace=0.6,
                          left=0.08, right=0.98, top=0.88, bottom=0.07)
    # a schematic
    ax_a = fig.add_subplot(gs[0, 0:2]); ax_a.axis("off"); ax_a.set_xlim(-0.2, 2.8); ax_a.set_ylim(-0.35, 1.05)
    P = {"FA": (0.0, 0.0), "AH": (1.0, 0.0), "WH": (0.72, 0.55)}
    for a, b, col, lab, off in [("WH", "FA", CL["FA"], f"d(WH, {RA})", (-0.16, 0.06)), ("WH", "AH", CL["AH"], "d(WH, AH)", (0.24, 0.0))]:
        ax_a.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=col, lw=1.0)
        ax_a.text((P[a][0] + P[b][0]) / 2 + off[0], (P[a][1] + P[b][1]) / 2 + off[1], lab, fontsize=5.2, color=col, ha="center")
    ax_a.plot([0, 1], [0, 0], color="0.6", lw=0.6, ls=(0, (2, 2)))
    for c, (x, y) in P.items():
        ax_a.scatter(x, y, s=60, color=CL[c], zorder=3)
        ax_a.text(x, y - 0.15 if c != "WH" else y + 0.1, CLAB[c] if c != "FA" else ("Ref. (" + RN + ")"), ha="center",
                  fontsize=5.2, color=CL[c] if c != "WH" else "#b07e00")
    ax_a.text(1.5, 0.95, "Each point: mean pre-lick population vector\n(z-scored good + mua units).\n"
                          "d = cross-validated squared Euclidean\ndistance per unit (trial halves).\n\n"
                          f"Distance difference = d(WH, {RA}) − d(WH, AH)\n"
                          "> 0: whisker hits closer to auditory hits\n< 0: closer to the reference", fontsize=4.9,
              va="top", color="0.2")
    ax_a.set_title("Population distance between trial types", loc="left")
    # b triangles (group geometry)
    sub = gs[0, 2:4].subgridspec(1, 2, wspace=0.15)
    axs_b = []
    for j, c in enumerate(["R+", "R-"]):
        ax = fig.add_subplot(sub[j]); axs_b.append(ax); draw_triangles(ax, Lwb, c, RA)
    from matplotlib.lines import Line2D
    share_triangle_limits(axs_b)
    axs_b[0].legend([Line2D([], [], marker="o", ls="", color=CL[c], ms=3.5) for c in ["FA", "AH", "WH"]],
                    [RA, "AH", "WH"], frameon=False, fontsize=4.8, loc="upper center", handletextpad=0.1, ncol=3,
                    bbox_to_anchor=(1.07, -0.02))
    # c-f per session
    row = [("dd", "3c", "Distance difference\n(> 0: WH nearer AH)", [(0, "0.6")]),
           ("d_WH_FA", "3d", f"WH to {RA}", None), ("d_WH_AH", "3e", "WH to AH", None),
           ("d_AH_FA", "3f", f"AH to {RA} (axis length)", None)]
    axs_c = []
    for j, (col, pid, ttl, ref) in enumerate(row):
        ax = fig.add_subplot(gs[1, j]); axs_c.append(ax)
        dots_panel(ax, Lwb, col, "Squared distance per unit" if j else "d(WH,ref) − d(WH,AH)", rng, pid, ttl, ref=ref)
        ax.set_title(ax.get_title(), fontsize=5.5)
    # g RT-matched distance difference, h cumulative distributions
    ax_g = fig.add_subplot(gs[2, 0])
    Lrt = L[(L.level == "all") & (L.variant == "rt_matched")]
    if m51.REF == "sl":
        ax_g.axis("off"); ax_g.text(0.5, 0.5, "RT-matched: not defined for\nspontaneous licks (no reaction time)",
                                    ha="center", va="center", fontsize=5.5, color="0.4", transform=ax_g.transAxes)
    else:
        dots_panel(ax_g, Lrt, "dd", "d(WH,ref) − d(WH,AH)", rng, "3g", "Distance difference,\nRT-matched trials",
                   ref=[(0, "0.6")])
        ax_g.set_title(ax_g.get_title(), fontsize=5.5)
    ax_h = fig.add_subplot(gs[2, 1])
    for k in GROUPS:
        v = np.sort(Lwb[(Lwb.cohort == k[0]) & (Lwb.stage == k[1])].dd.dropna().to_numpy())
        if len(v):
            ax_h.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COH[k[0]], lw=1.1,
                      ls="-" if k[1] == "expert" else (0, (2, 1.5)), label=GLAB[k])
    ax_h.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax_h.set_xlabel("Distance difference (session)"); ax_h.set_ylabel("Cumulative fraction")
    ax_h.set_title("Distributions over sessions", fontsize=5.5)
    ax_h.legend(frameon=False, fontsize=4.6, loc="upper left", handlelength=1.6)
    # i/j: decompose the WH displacement from the reference into a component along ref -> AH and an orthogonal one
    # (from the same cv distances: (WH - ref).(AH - ref) = (d_WR + d_AR - d_WA) / 2; orthogonal^2 = d_WR - along^2 / d_AR)
    Ld = Lwb.copy()
    Ld["along"] = (Ld.d_WH_FA + Ld.d_AH_FA - Ld.d_WH_AH) / 2
    ok = Ld.d_AH_FA >= 0.01
    Ld["ortho"] = np.where(ok, Ld.d_WH_FA - Ld.along ** 2 / Ld.d_AH_FA.where(ok), np.nan)
    ax_i = fig.add_subplot(gs[2, 2])
    dots_panel(ax_i, Ld, "along", f"(WH − {RA})·(AH − {RA}) per unit", rng, "3i",
               f"WH displacement along\nthe {RA} → AH direction", ref=[(0, "0.6")])
    ax_i.set_title(ax_i.get_title(), fontsize=5.5)
    ax_j = fig.add_subplot(gs[2, 3])
    dots_panel(ax_j, Ld, "ortho", "Squared distance per unit", rng, "3j",
               f"WH displacement orthogonal\nto the {RA} → AH direction", ref=[(0, "0.6")])
    ax_j.set_title(ax_j.get_title(), fontsize=5.5)
    letter_row(fig, [ax_a, axs_b[0]], "ab"); letter_row(fig, axs_c, "cdef"); letter_row(fig, [ax_g, ax_h, ax_i, ax_j], "ghij")
    fig.suptitle(f"Figure 3 | Population distance: in R+ but not R− mice, whisker hits leave the {RN[:-1]} state, partly toward auditory hits",
                 x=0.08, y=0.985, ha="left", fontsize=7.5, weight="bold")
    save(fig, out, "Fig3_population_distance"); plt.close(fig)


def lda_plane(sid, W, rng):
    """held-out trials of one session in the plane spanned by the shrinkage-LDA axis (AH vs ref) and the first principal
    component of the training AH + ref trials orthogonal to it: 5-fold; per fold units are z-scored and the LDA fitted on
    the training trials, held-out AH / ref trials and all WH trials (averaged over folds) are projected; LDA coordinate
    normalised so that the training ref mean = 0 and AH mean = 1. Also returns the direction of the (training)
    mean-difference axis in this plane."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); r = ss[ss.session_id == sid].iloc[0]
    z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W[W.session_id == sid][["electrode_group", "cluster_id", "cohort"]], on=["electrode_group", "cluster_id"], how="left")
    X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
    u = (raw.mean(1) >= m51.MIN_FR) & K.cohort.notna().to_numpy()
    Z = X[u].T
    m = np.isin(lab, ["AH", "FA"])
    from sklearn.model_selection import StratifiedKFold
    ia = np.where(m)[0]; iw = np.where(lab == "WH")[0]
    y = (lab[ia] == "AH").astype(int)
    P = np.full((len(lab), 2), np.nan); Pw = []; mds = []; v0 = None
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(ia, y):
        A = Z[ia[tr]]; mu_, sd_ = A.mean(0), A.std(0); sd_[sd_ == 0] = 1
        zf = lambda q: (q - mu_) / sd_
        w = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(zf(A), y[tr]).coef_[0]
        w = w / np.linalg.norm(w)
        Rr = zf(A) - np.outer(zf(A) @ w, w)
        v = np.linalg.svd(Rr - Rr.mean(0), full_matrices=False)[2][0]
        if v0 is not None and v @ v0 < 0:
            v = -v
        v0 = v if v0 is None else v0
        f0 = (zf(A[y[tr] == 0]) @ w).mean(); f1 = (zf(A[y[tr] == 1]) @ w).mean(); sc = f1 - f0
        o0 = (zf(A) @ v).mean()
        prj = lambda q: np.c_[(zf(q) @ w - f0) / sc, (zf(q) @ v - o0) / sc]
        P[ia[te]] = prj(Z[ia[te]]); Pw.append(prj(Z[iw]))
        md = zf(A[y[tr] == 1]).mean(0) - zf(A[y[tr] == 0]).mean(0)
        mds.append(np.array([md @ w, md @ v]) / np.linalg.norm([md @ w, md @ v]))
    P[iw] = np.mean(Pw, 0)
    mdv = np.mean(mds, 0); mdv = mdv / np.linalg.norm(mdv)
    return P, lab, mdv, int(u.sum())


def lda_plane_panel(ax, sid, W, rng, k, lam_lda):
    P, lab, mdv, nu = lda_plane(sid, W, rng)
    for c in ["FA", "AH", "WH"]:
        q = P[lab == c]
        ax.scatter(q[:, 0], q[:, 1], s=3, color=CL[c], alpha=0.5, lw=0, label=CLAB[c], zorder=2 if c != "WH" else 3)
        ax.scatter(*q.mean(0), s=28, color=CL[c], edgecolor="white", lw=0.6, zorder=5)
    ax.annotate("", xy=(0.5 + 0.6 * mdv[0], 0.6 * mdv[1]), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.8, ls=(0, (2, 1))))
    ax.text(0.5 + 0.65 * mdv[0], 0.65 * mdv[1], "mean-\ndifference\naxis", fontsize=4.2, color="0.35", ha="left", va="center")
    ax.axvline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax.axvline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
    ax.axvline(0.5, color="0.6", lw=0.4)
    q = np.nanpercentile(np.abs(P), 98, axis=0)
    ax.set_xlim(-max(1.2, q[0] * 0.8), max(2.2, q[0] * 0.9)); ax.set_ylim(-q[1], q[1])
    ax.set_xlabel("Shrinkage-LDA axis (ref = 0, AH = 1)"); ax.set_ylabel("Orthogonal PC 1")
    ax.set_title(f"{GLAB[k]} example ({nu} units)\nλ_LDA = {lam_lda:.2f}", color=COH[k[0]], fontsize=5.4)


def fig3s_lambda(plt, D, out, pop, rng, PR):
    L = D["lam"]; Lwb = L[(L.level == "all") & (L.variant == "all")]
    fig = plt.figure(figsize=(W_IN, 9.8))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.0, 1.0, 1.15, 1.25, 1.15], hspace=0.95, wspace=0.55,
                          left=0.08, right=0.98, top=0.925, bottom=0.05)
    # a schematic
    ax_a = fig.add_subplot(gs[0, 0:2]); ax_a.axis("off"); ax_a.set_xlim(-0.35, 2.35); ax_a.set_ylim(-0.75, 0.8)
    ax_a.set_title("λ: position of whisker hits on the FA → AH axis", loc="left")
    ax_a.annotate("", xy=(1.0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.3"))
    for x, y, c in [(0, 0, "FA"), (1, 0, "AH")]:
        ax_a.scatter(x, y, s=70, color=CL[c], zorder=3); ax_a.text(x, -0.18, CLAB[c], ha="center", va="top", fontsize=6, color=CL[c])
    wx, wy = 0.55, 0.5
    ax_a.scatter(wx, wy, s=70, color=CL["WH"], zorder=3); ax_a.text(wx + 0.06, wy + 0.05, CLAB["WH"], fontsize=6, color="#b07e00")
    ax_a.plot([wx, wx], [wy, 0], color=CL["WH"], lw=0.8, ls=(0, (2, 2)))
    ax_a.text(wx, -0.42, "λ = 0.55", ha="center", fontsize=6, color="#b07e00")
    ax_a.text(1.2, 0.55, "Points: population mean pre-lick vectors\n(z-scored good + mua units)\n"
                         "λ = 0: like false alarms (unrewarded lick)\nλ = 1: like auditory hits (rewarded lick)\n"
                         "Cross-validated over trial halves", fontsize=5, va="top", color="0.25")
    # b example sessions: single-trial projections
    ex = PR[PR.region == "all"]
    sub = gs[0, 2:4].subgridspec(1, 2, wspace=0.35)
    axs_b = []
    for j, k in enumerate([GROUPS[1], GROUPS[3]]):
        ax = fig.add_subplot(sub[j]); axs_b.append(ax)
        cand = Lwb[(Lwb.cohort == k[0]) & (Lwb.stage == k[1])].dropna(subset=["lam"])
        cand = cand[cand.session_id.isin(ex.session_id)]
        if not len(cand):
            continue
        sid = cand.iloc[(cand.lam - cand.lam.median()).abs().argsort().iloc[0]].session_id
        d = ex[ex.session_id == sid]
        for kk, c in enumerate(["FA", "AH", "WH"]):
            v = d[d.cls == c].score.to_numpy()
            ax.scatter(kk + rng.uniform(-0.22, 0.22, len(v)), v, s=3, color=CL[c], alpha=0.6, lw=0)
            ax.plot([kk - 0.3, kk + 0.3], [np.mean(v)] * 2, color="k", lw=1.0)
        ax.axhline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax.axhline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(range(3), ["FA", "AH", "WH"]); ax.set_ylim(-2, 3)
        ax.set_title(f"{GLAB[k]}: example session\nλ = {cand[cand.session_id == sid].lam.iloc[0]:.2f}", color=COH[k[0]], fontsize=5.8)
        if j == 0:
            ax.set_ylabel("Projection (FA = 0, AH = 1)")
    # c pooled distributions
    sub = gs[1, :].subgridspec(1, 4, wspace=0.3)
    axs_c = []
    bins = np.linspace(-2, 3, 41)
    for j, k in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j]); axs_c.append(ax)
        d = ex[(ex.cohort == k[0]) & (ex.stage == k[1])]
        for c in ["FA", "AH", "WH"]:
            v = d[d.cls == c].score.dropna(); v = v[(v > -2) & (v < 3)]
            ax.hist(v, bins, density=True, histtype="stepfilled" if c != "WH" else "step", color=CL[c],
                    alpha=0.3 if c != "WH" else 1, lw=1.2 if c == "WH" else 0, label=CLAB[c])
        ax.axvline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax.axvline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
        lam_k = Lwb[(Lwb.cohort == k[0]) & (Lwb.stage == k[1]) & Lwb.session_id.isin(d.session_id)].lam
        ax.set_title(f"{GLAB[k]} ({d.session_id.nunique()} sessions)\nλ = {lam_k.mean():.2f} ± {lam_k.sem():.2f}", color=COH[k[0]], fontsize=5.8)
        ax.set_xlabel("Projection (FA = 0, AH = 1)"); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Trial density"); ax.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=5)
    # d lambda per session, e dd, f lambda learners/all comparison: lambda RT-matched
    ax_d = fig.add_subplot(gs[2, 0])
    dots_panel(ax_d, Lwb, "lam", "λ", rng, "3S d", "λ, whole brain", ref=[(0, CL["FA"]), (1, CL["AH"])])
    ax_e = fig.add_subplot(gs[2, 1])
    M_ = Lwb.dropna(subset=["lam", "dd"])
    for c in ["R+", "R-"]:
        mm = M_[M_.cohort == c]; ax_e.scatter(mm.dd, mm.lam, s=6, color=COH[c], alpha=0.7, lw=0, label=c.replace("-", "−"))
    rr_, pp_ = stats.spearmanr(M_.dd, M_.lam)
    ax_e.set_xlabel("Distance difference"); ax_e.set_ylabel("λ")
    ax_e.set_title(f"λ vs distance difference\nr = {rr_:.2f} ({fmt_p(pp_)})", fontsize=5.6)
    ax_e.legend(frameon=False, fontsize=4.6, loc="upper left")
    ax_f = fig.add_subplot(gs[2, 2])
    Lrt = L[(L.level == "all") & (L.variant == "rt_matched")]
    if m51.REF == "sl":
        ax_f.axis("off"); ax_f.text(0.5, 0.5, "RT-matched λ: not defined for\nspontaneous licks (no reaction time)", ha="center",
                 va="center", fontsize=5.5, color="0.4", transform=ax_f.transAxes)
    else:
        dots_panel(ax_f, Lrt, "lam", "λ", rng, "3S f", "λ, RT-matched trials", ref=[(0, CL["FA"]), (1, CL["AH"])])
    # g: cumulative distribution of session lambda per group
    ax_g = fig.add_subplot(gs[2, 3])
    for k in GROUPS:
        v = np.sort(Lwb[(Lwb.cohort == k[0]) & (Lwb.stage == k[1])].lam.dropna().to_numpy())
        if len(v):
            ax_g.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COH[k[0]], lw=1.1,
                      ls="-" if k[1] == "expert" else (0, (2, 1.5)), label=GLAB[k])
    ax_g.axvline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax_g.axvline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
    ax_g.set_xlabel("λ (session)"); ax_g.set_ylabel("Cumulative fraction"); ax_g.set_title("λ distributions\n ")
    ax_g.legend(frameon=False, fontsize=4.8, loc="upper left", handlelength=1.6)
    # LDA half (formerly Fig. 5a-d)
    LD = D["lda"]
    wb = lambda t, v="all": t[(t.level == "all") & (t.variant == v)]
    RA = "FA" if m51.REF == "fa" else "SL"
    # h: two example sessions in the plane (LDA axis, orthogonal PC); the session with the median lambda_LDA of its group
    axs_h = []
    for jj, k in enumerate([GROUPS[1], GROUPS[3]]):
        ax = fig.add_subplot(gs[3, jj]); axs_h.append(ax)
        cand = wb(LD)[(wb(LD).cohort == k[0]) & (wb(LD).stage == k[1])].dropna(subset=["lam_lda"])
        if not len(cand):
            ax.axis("off"); continue
        rr = cand.iloc[(cand.lam_lda - cand.lam_lda.median()).abs().argsort().iloc[0]]
        lda_plane_panel(ax, rr.session_id, D["W"], rng, k, rr.lam_lda)
        if jj == 0:
            ax.legend(frameon=False, fontsize=4.3, loc="upper left", markerscale=2, handletextpad=0.1, borderaxespad=0.1)
    ax_h = fig.add_subplot(gs[3, 2])
    M = wb(L).merge(wb(LD)[["session_id", "lam_lda", "cohort"]], on=["session_id", "cohort"]).dropna(subset=["lam", "lam_lda"])
    for c in ["R+", "R-"]:
        mm = M[M.cohort == c]; ax_h.scatter(mm.lam, mm.lam_lda, s=7, color=COH[c], alpha=0.7, lw=0, label=c.replace("-", "−"))
    r, p = stats.spearmanr(M.lam, M.lam_lda)
    ax_h.plot([-1, 2], [-1, 2], color="0.6", lw=0.5, ls=(0, (2, 2))); ax_h.set_xlim(-1, 2); ax_h.set_ylim(-1, 2)
    ax_h.set_xlabel("λ (mean-difference axis)"); ax_h.set_ylabel("λ_LDA (shrinkage LDA axis)")
    ax_h.set_title(f"λ vs λ_LDA, r = {r:.2f}\n({fmt_p(p)}, {len(M)} sessions)", fontsize=5.6)
    STATS.append(dict(panel="3S i", measure="Spearman lambda vs lambda_LDA", r=r, p=p, n=len(M)))
    ax_i = fig.add_subplot(gs[3, 3])
    dots_panel(ax_i, wb(LD), "lam_lda", "λ_LDA", rng, "3S j", "λ_LDA, whole brain", ref=[(0, CL["FA"]), (1, CL["AH"])])
    ax_j = fig.add_subplot(gs[4, 0])
    dots_panel(ax_j, wb(LD), "d_prime", f"Held-out d′ (AH vs {RA})", rng, "3S k", "LDA axis quality")
    ax_k = fig.add_subplot(gs[4, 1])
    if m51.REF == "sl":
        ax_k.axis("off"); ax_k.text(0.5, 0.5, "RT-matched λ_LDA: not defined for\nspontaneous licks (no reaction time)",
                                    ha="center", va="center", fontsize=5.5, color="0.4", transform=ax_k.transAxes)
    else:
        dots_panel(ax_k, wb(LD, "rt_matched"), "lam_lda", "λ_LDA", rng, "3S l", "λ_LDA, RT-matched",
                   ref=[(0, CL["FA"]), (1, CL["AH"])])
    ax_m = fig.add_subplot(gs[4, 2])
    for k in GROUPS:
        v = np.sort(wb(LD)[(wb(LD).cohort == k[0]) & (wb(LD).stage == k[1])].lam_lda.dropna().to_numpy())
        if len(v):
            ax_m.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COH[k[0]], lw=1.1,
                      ls="-" if k[1] == "expert" else (0, (2, 1.5)), label=GLAB[k])
    ax_m.axvline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax_m.axvline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
    ax_m.set_xlabel("λ_LDA (session)"); ax_m.set_ylabel("Cumulative fraction"); ax_m.set_title("λ_LDA distributions", fontsize=5.6)
    ax_m.legend(frameon=False, fontsize=4.4, loc="upper left", handlelength=1.6)
    fig.align_ylabels([axs_c[0], ax_d, axs_h[0], ax_j])
    letter_row(fig, [ax_a, axs_b[0]], "ab"); letter_row(fig, axs_c[:1], "c"); letter_row(fig, [ax_d, ax_e, ax_f, ax_g], "defg")
    letter_row(fig, axs_h[:1] + [ax_h, ax_i], "hij"); letter_row(fig, [ax_j, ax_k, ax_m], "klm")
    fig.suptitle("Figure 3—supplement | λ (projection on the reference → AH axis; a–g) and λ_LDA (shrinkage-LDA axis; h–m)",
                 x=0.08, y=0.99, ha="left", fontsize=7.5, weight="bold")
    save(fig, out, "Fig3S_lambda_lambdaLDA"); plt.close(fig)


# ------------------------------------------------------------------ Figure 4 / COSYNE
def area_qualify(LA, min_sess=3, col="dd"):
    """area groups included per cohort: >= min_sess sessions with a valid value of col in BOTH stages of that cohort"""
    ok = {}
    for c in ["R+", "R-"]:
        n = LA[LA.cohort == c].dropna(subset=[col]).groupby(["region", "stage"]).agg(
            s=("session_id", "nunique"), m=("mouse_id", "nunique")).unstack("stage")
        q = set()
        for g, r in n.iterrows():
            if r.get(("s", "learning"), 0) >= min_sess and r.get(("s", "expert"), 0) >= min_sess:
                q.add(g)
        ok[c] = q
    return ok


def area_anova(LA, areas, col="lam", n_perm=1000, seed=0):
    """two-way-plus ANOVA on session x area values: col ~ C(cohort) * C(stage) * C(area) (type II, OLS), restricted to
    areas included for both cohorts. Reported terms: cohort, cohort x area, cohort x stage (cohort difference in the
    learning change) and cohort x stage x area. Sessions contribute several areas (not independent), so p values are
    also computed by permuting cohort labels across mice (n_perm, F of each term re-estimated)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = LA[LA.region.isin(areas)].dropna(subset=[col]).rename(columns={col: "y", "region": "area"})
    if d.area.nunique() < 2 or d.cohort.nunique() < 2:
        return None
    terms = {"cohort": "C(cohort)", "cohort x area": "C(cohort):C(area)", "cohort x stage": "C(cohort):C(stage)",
             "cohort x stage x area": "C(cohort):C(stage):C(area)"}
    def fit(dd):
        tab = anova_lm(smf.ols("y ~ C(cohort) * C(stage) * C(area)", data=dd).fit(), typ=2)
        return {k: (tab.loc[v, "F"], tab.loc[v, "PR(>F)"]) if v in tab.index else (np.nan, np.nan) for k, v in terms.items()}
    obs = fit(d)
    rng = np.random.default_rng(seed)
    mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    midx = pd.Index(mice).get_indexer(d.mouse_id)
    null = {k: [] for k in terms}
    for _ in range(n_perm):
        dd = d.copy(); dd["cohort"] = rng.permutation(mc)[midx]
        try:
            r = fit(dd)
        except Exception:
            continue
        for k in terms:
            null[k].append(r[k][0])
    out = {}
    for k in terms:
        nl = np.array(null[k]); nl = nl[np.isfinite(nl)]
        F, p = obs[k]
        out[k] = dict(F=F, p=p, p_perm=(1 + np.sum(nl >= F)) / (1 + len(nl)) if len(nl) else np.nan)
    return out


def anova_text(A):
    if A is None:
        return "ANOVA: not enough areas"
    lab = {"cohort": "cohort", "cohort x area": "cohort × area", "cohort x stage": "cohort × stage",
           "cohort x stage x area": "cohort × stage × area"}
    return "ANOVA (area-level values; perm. over mice)\n" + "\n".join(
        f"{lab[k]}: F = {v['F']:.2f}, p = {fmt_p(v['p']).replace('p = ', '').replace('p < ', '<')}, "
        f"perm {fmt_p(v['p_perm']).replace('p = ', '').replace('p < ', '<')}" for k, v in A.items())


def fig4(plt, D, out, pop, rng, PR):
    import ephys_utilities.allen_utils.allen_utils as au
    L = D["lam"]; LA = L[(L.level == "area_group") & (L.variant == "all")]
    TT = D["lam_tests"]; TT = TT[(TT.metric == "dd") & (TT.variant == "all") & (TT.level == "area_group")].set_index("region")
    Q = area_qualify(LA, col="dd")
    order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] | Q["R-"]]
    both = [g for g in order if g in Q["R+"] and g in Q["R-"]]
    short = {g: g.replace(" areas", "").replace("Somatosensory", "SS").replace("Lateral septal complex", "LSX")
             .replace("Amygdala and hypothalamus", "Amyg./Hypoth.") for g in order}
    fig = plt.figure(figsize=(W_IN, 8.4))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.45, 1.0, 1.0, 1.0], hspace=0.95, wspace=0.6,
                          left=0.12, right=0.93, top=0.9, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0:2])
    y = np.arange(len(order))
    for k, c in enumerate(["R+", "R-"]):
        xs_, ys_, es_, ps_ = [], [], [], []
        for i, g in enumerate(order):
            if g not in Q[c]:
                continue
            d = LA[(LA.region == g) & (LA.cohort == c)].dropna(subset=["dd"])
            a, b = d[d.stage == "learning"], d[d.stage == "expert"]
            diff = b.dd.mean() - a.dd.mean()
            se = np.sqrt(a.dd.var(ddof=1) / len(a) + b.dd.var(ddof=1) / len(b))
            p = stats.mannwhitneyu(a.dd, b.dd).pvalue
            pw = stats.ttest_ind(a.dd, b.dd, equal_var=False).pvalue
            xs_.append(diff); ys_.append(i + (k - 0.5) * 0.3); es_.append(se); ps_.append(p)
            STATS.append(dict(panel="4a", measure=f"Δ(distance difference) {g} {c}", diff=diff, sem=se, p_MWU=p, p_Welch=pw,
                              n_sessions_learning=len(a), n_sessions_expert=len(b),
                              n_mice_learning=a.mouse_id.nunique(), n_mice_expert=b.mouse_id.nunique()))
        ax_a.errorbar(xs_, ys_, xerr=es_, fmt="none", ecolor=COH[c], elinewidth=0.8)
        ax_a.scatter(xs_, ys_, s=16, color=[COH[c] if p < 0.05 else "white" for p in ps_], edgecolor=COH[c], lw=0.8,
                     zorder=3, label=f"{c.replace('-', '−')}: expert − learning")
    for i, g in enumerate(order):
        if g in both and g in TT.index and np.isfinite(TT.loc[g, "interaction_p"]):
            txt = f"{fmt_p(TT.loc[g, 'interaction_p'])} ({fmt_p(TT.loc[g, 'interaction_p_maxT'])})"
        elif g in both:
            txt = "n/a"
        else:
            txt = f"{'R+' if g in Q['R+'] else 'R−'} only"
        ax_a.text(1.02, i, txt, transform=ax_a.get_yaxis_transform(), fontsize=4.8, va="center",
                  color="0.15" if g in both else "0.5")
    ax_a.text(1.02, -0.85, "Interaction p\n(family-wise p)", transform=ax_a.get_yaxis_transform(), fontsize=5,
              va="bottom", weight="bold")
    ax_a.axvline(0, color="0.3", lw=0.6)
    ax_a.set_yticks(y, [short[g] for g in order]); ax_a.invert_yaxis(); ax_a.set_ylim(len(order) - 0.4, -0.7)
    ax_a.set_xlabel("Change in distance difference, expert − learning\n(mean ± SEM; > 0: WH moved toward AH)"); ax_a.legend(frameon=False, loc="lower right", fontsize=5)
    ax_a.set_title("Change of distance difference per area group (filled: MWU p < 0.05)", loc="left")
    AN = area_anova(LA, both, col="dd")
    if AN:
        for k, v in AN.items():
            STATS.append(dict(panel="4a ANOVA", measure=k, F=v["F"], p=v["p"], p_perm=v["p_perm"], n_areas=len(both)))
    ax_a.text(1.02, -0.08, anova_text(AN), transform=ax_a.transAxes, fontsize=4.2, va="top", ha="left", color="0.25")
    ax_b = fig.add_subplot(gs[0, 3])
    M = np.full((len(order), 4), np.nan)
    for i, g in enumerate(order):
        for j, k in enumerate(GROUPS):
            if g in Q[k[0]]:
                M[i, j] = LA[(LA.region == g) & (LA.cohort == k[0]) & (LA.stage == k[1])].dd.mean()
    vlim = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.1
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.92")
    im = ax_b.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax_b.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=4.5)
    ax_b.set_xticks(range(4), ["R+ L", "R+ E", "R− L", "R− E"], rotation=45, ha="right")
    for t, c in zip(ax_b.get_xticklabels(), ["R+", "R+", "R-", "R-"]):
        t.set_color(COH[c])
    ax_b.set_yticks(range(len(order)), [short[g] for g in order])
    cb = fig.colorbar(im, ax=ax_b, fraction=0.08, pad=0.04); cb.set_label("Mean d(WH,ref) − d(WH,AH)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax_b.set_title("Distance difference per area\n(grey: < 3 sessions per stage)", fontsize=5.8)
    focus = [g for g in ["Motor areas", "Hippocampus", "Thalamus", "Somatosensory-whisker", "Striatum", "Midbrain"]
             if g in both][:4]
    axs_c = []
    for j, g in enumerate(focus):
        ax = fig.add_subplot(gs[1, j]); axs_c.append(ax)
        dots_panel(ax, LA[LA.region == g], "dd", "d(WH,ref) − d(WH,AH)" if j == 0 else "", rng, f"4c {g}", short[g],
                   ref=[(0, "0.6")])
    W, P, tc = D["W"], D["P"], D["tc"]
    tested = (W["sig:auditory_hit_vs_fa_prelick@all"].notna() & W["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
    sub = gs[2, :].subgridspec(1, 4, wspace=0.35)
    axs_d = []
    for j, (g, k) in enumerate([(focus[0], GROUPS[1]), (focus[0], GROUPS[3]), (focus[1], GROUPS[1]), (focus[1], GROUPS[3])]):
        ax = fig.add_subplot(sub[j]); axs_d.append(ax)
        idx = W.row.to_numpy()[tested & ((W.cohort == k[0]) & (W.stage == k[1]) & (W.area_group == g)).to_numpy()]
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        for kk, c in enumerate(m51.CLASSES):
            mu, se = P[idx, kk].mean(0), P[idx, kk].std(0) / np.sqrt(max(len(idx), 1))
            ax.fill_between(tc, mu - se, mu + se, color=CL[c], alpha=0.25, lw=0); ax.plot(tc, mu, color=CL[c], lw=1.0, label=CLAB[c])
        ax.set_xlim(-600, 400); ax.set_xticks([-400, 0, 400]); ax.set_xlabel("Time from first lick (ms)")
        ax.set_title(f"{short[g]}, {GLAB[k]}\n({len(idx)} units; mean ± s.e.m. over units)", color=COH[k[0]], fontsize=5.4)
        if j == 0:
            ax.set_ylabel("Δ firing rate (Hz)"); ax.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=5)
    # e: same areas, mean over units within session then mean +- s.e.m. over sessions (pre-trial baseline)
    PB, KB, tcb = load_psth_prestart(W)
    Wb = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    tb = (Wb["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wb["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
    sub = gs[3, :].subgridspec(1, 4, wspace=0.35)
    axs_e = []
    for j, (g, k) in enumerate([(focus[0], GROUPS[1]), (focus[0], GROUPS[3]), (focus[1], GROUPS[1]), (focus[1], GROUPS[3])]):
        ax = fig.add_subplot(sub[j]); axs_e.append(ax)
        ns, nu = mouse_psth(ax, Wb, PB, tcb, tb & ((Wb.cohort == k[0]) & (Wb.stage == k[1]) & (Wb.area_group == g)).to_numpy(),
                            legend=(j == 0), min_units=1)
        ax.set_xlim(-600, 400); ax.set_xticks([-400, 0, 400]); ax.set_xlabel("Time from first lick (ms)")
        ax.set_title(f"{short[g]}, {GLAB[k]}\n({ns} sessions, {nu} units)", color=COH[k[0]], fontsize=5.8)
        if j == 0:
            ax.set_ylabel("Rate − pre-trial\nbaseline (Hz)")
    fig.align_ylabels([axs_c[0], axs_d[0], axs_e[0]])
    letter_row(fig, [ax_a, ax_b], "ab", dx_in=0.75); letter_row(fig, axs_c[:1], "c"); letter_row(fig, axs_d[:1], "d")
    letter_row(fig, axs_e[:1], "e")
    fig.suptitle("Figure 4 | Convergence across brain areas (no area survives family-wise correction)",
                 x=0.08, y=0.985, ha="left", fontsize=7.5, weight="bold")
    save(fig, out, "Fig4_areas"); plt.close(fig)
    D["fig4_areas"] = dict(order=order, both=both, Q=Q, focus=focus)


def fig_cosyne(plt, D, out, pop, rng, PR):
    """COSYNE abstract figure: population PSTHs, projections, lambda, single-neuron transfer, areas (no schematic)"""
    W, P, tc = D["W"], D["P"], D["tc"]
    L = D["lam"]; Lwb = L[(L.level == "all") & (L.variant == "all")]
    LA = L[(L.level == "area_group") & (L.variant == "all")]
    fig = plt.figure(figsize=(W_IN, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=1.0, wspace=0.55,
                          left=0.08, right=0.985, top=0.86, bottom=0.13)
    tested = (W["sig:auditory_hit_vs_fa_prelick@all"].notna() & W["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
    axs_a = []
    for j, k in enumerate([GROUPS[1], GROUPS[3]]):
        ax = fig.add_subplot(gs[0, j]); axs_a.append(ax)
        idx = W.row.to_numpy()[tested & ((W.cohort == k[0]) & (W.stage == k[1])).to_numpy()]
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        for kk, c in enumerate(m51.CLASSES):
            mu, se = P[idx, kk].mean(0), P[idx, kk].std(0) / np.sqrt(len(idx))
            ax.fill_between(tc, mu - se, mu + se, color=CL[c], alpha=0.25, lw=0); ax.plot(tc, mu, color=CL[c], lw=1.0, label=CLAB[c])
        ax.set_xlim(-500, 300); ax.set_xticks([-400, 0, 200]); ax.set_xlabel("Time from first lick (ms)")
        ax.set_title(f"{GLAB[k]}\n({len(idx) / 1000:.1f}k units)", color=COH[k[0]], fontsize=5.8)
        if j == 0:
            ax.set_ylabel("Δ firing rate (Hz)"); ax.legend(frameon=False, loc="upper left", handlelength=1.0, fontsize=4.6)
    ymax = max(a.get_ylim()[1] for a in axs_a)
    for a in axs_a:
        a.set_ylim(-0.4, ymax)
    ex = PR[PR.region == "all"]; bins = np.linspace(-2, 3, 41)
    axs_b = []
    for j, k in enumerate([GROUPS[1], GROUPS[3]]):
        ax = fig.add_subplot(gs[0, 2 + j]); axs_b.append(ax)
        d = ex[(ex.cohort == k[0]) & (ex.stage == k[1])]
        for c in ["FA", "AH", "WH"]:
            v = d[d.cls == c].score.dropna(); v = v[(v > -2) & (v < 3)]
            ax.hist(v, bins, density=True, histtype="stepfilled" if c != "WH" else "step", color=CL[c],
                    alpha=0.3 if c != "WH" else 1, lw=1.2 if c == "WH" else 0, label=CLAB[c])
        ax.axvline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax.axvline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
        lam_k = Lwb[(Lwb.cohort == k[0]) & (Lwb.stage == k[1])].lam.dropna()
        ax.set_title(f"{GLAB[k]}\nλ = {lam_k.mean():.2f} ± {lam_k.sem():.2f}", color=COH[k[0]], fontsize=5.8)
        ax.set_xlabel("Projection (FA = 0, AH = 1)"); ax.set_yticks([]); ax.set_xticks([-1, 0, 1, 2])
        if j == 0:
            ax.set_ylabel("Trial density")
    ax_c = fig.add_subplot(gs[1, 0])
    dots_panel(ax_c, Lwb, "lam", "λ (WH on FA → AH axis)", rng, "C-c", "Population: λ",
               ref=[(0, CL["FA"]), (1, CL["AH"])])
    T = D["tr"]; T = T[(T.level == "all") & (T.variant == "all")]
    ax_d = fig.add_subplot(gs[1, 1])
    dots_panel(ax_d, T, "frac_transfer", "Fraction of reward-lick neurons", rng, "C-d", "Single neurons: also\nWH vs FA selective")
    ax_d.set_title(ax_d.get_title(), fontsize=5.6)
    # areas: delta lambda per area group (within-cohort inclusion)
    import ephys_utilities.allen_utils.allen_utils as au
    Q = area_qualify(LA)
    order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] | Q["R-"]]
    short = {g: g.replace(" areas", "").replace("Somatosensory", "SS").replace("Lateral septal complex", "LSX")
             .replace("Amygdala and hypothalamus", "Amyg./Hypoth.") for g in order}
    ax_e = fig.add_subplot(gs[1, 2:4])
    x = np.arange(len(order))
    for k, c in enumerate(["R+", "R-"]):
        xs_, ds_, es_, ps_ = [], [], [], []
        for i, g in enumerate(order):
            if g not in Q[c]:
                continue
            d = LA[(LA.region == g) & (LA.cohort == c)].dropna(subset=["lam"])
            a, b = d[d.stage == "learning"].lam, d[d.stage == "expert"].lam
            xs_.append(i + (k - 0.5) * 0.3); ds_.append(b.mean() - a.mean())
            es_.append(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))); ps_.append(stats.mannwhitneyu(a, b).pvalue)
        ax_e.errorbar(xs_, ds_, yerr=es_, fmt="none", ecolor=COH[c], elinewidth=0.8)
        ax_e.scatter(xs_, ds_, s=16, color=[COH[c] if p < 0.05 else "white" for p in ps_], edgecolor=COH[c], lw=0.8,
                     zorder=3, label=f"{c.replace('-', '−')}")
    ax_e.axhline(0, color="0.3", lw=0.6)
    ax_e.set_xticks(x, [short[g] for g in order], rotation=40, ha="right")
    ax_e.set_ylabel("Δλ, expert − learning")
    ax_e.legend(frameon=False, fontsize=5, loc="upper right", ncol=2)
    ax_e.set_title("Areas: change of λ (filled: MWU p < 0.05)")
    letter_row(fig, [axs_a[0], axs_b[0]], "ab"); letter_row(fig, [ax_c, ax_d, ax_e], "cde")
    fig.suptitle("Whisker hits become more like auditory hits before the lick in R+, not R−, mice", x=0.08, y=0.985,
                 ha="left", fontsize=7.5, weight="bold")
    save(fig, out, "COSYNE_figure"); plt.close(fig)


# ------------------------------------------------------------------ Figure 5
def fig5(plt, D, out, pop, rng):
    """row 1: single-session decoders, chance-corrected by linear shifts (069). row 2: pseudo-population decoders (064;
    hierarchical bootstrap mice -> sessions -> neurons -> trials, within-session shift null) as a function of the number
    of pooled neurons M, and the learning -> expert change at the largest M with the cohort-permutation interaction."""
    DC = D["dec"]; DC = DC[DC.level == "all"]
    RA = "FA" if m51.REF == "fa" else "SL"
    fig = plt.figure(figsize=(W_IN, 7.4))
    gs = fig.add_gridspec(3, 4, hspace=0.95, wspace=0.7, left=0.08, right=0.98, top=0.9, bottom=0.07)
    panels = [("bacc_corrected", "Balanced accuracy − chance", "5a", f"Decoder AH vs {RA}:\naccuracy above chance", [(0, "0.6")]),
              ("transfer_bin_corrected", f"Transfer − chance\n(0 = like {RA}, 1 = like AH)", "5b", "WH decoded as AH\n(yes/no), chance-corr.", [(0, CL["FA"]), (1, CL["AH"])]),
              ("transfer_prob_corrected", f"Transfer − chance\n(0 = like {RA}, 1 = like AH)", "5c", "WH decoded as AH\n(probability), chance-corr.", [(0, CL["FA"]), (1, CL["AH"])]),
              ("num_prob_corrected", f"P(AH|WH) − P(AH|{RA}) − chance", "5d", f"WH more AH-like than\n{RA}, chance-corrected", [(0, "0.6")])]
    axs = [fig.add_subplot(gs[0, k]) for k in range(4)]
    for ax, (col, yl, pid, ttl, ref) in zip(axs, panels):
        dots_panel(ax, DC, col, yl, rng, pid, ttl, ref=ref)
        ax.set_title(ax.get_title(), fontsize=5.4)
    PP = D.get("pp"); PB_ = D.get("pp_boot")
    axs2 = [fig.add_subplot(gs[1, k]) for k in range(4)]
    axs3 = [fig.add_subplot(gs[2, k]) for k in range(4)]
    have = PP is not None and "num_prob" in set(PP.readout)
    if not have:
        for ax in axs2 + axs3:
            ax.axis("off")
        axs2[0].text(0, 0.5, "pseudo-population readouts not available (rerun 064)", transform=axs2[0].transAxes, fontsize=6)
    else:
        W_ = PP[PP.region == "all"]
        Ms = sorted(W_.M.unique()); Mmax = Ms[-1]
        RO = [("bacc", "Balanced accuracy − chance", f"AH vs {RA} accuracy", None),
              ("transfer_bin", f"Transfer − chance\n(0 = like {RA}, 1 = like AH)", "WH decoded as AH (yes/no)", (0, 1)),
              ("transfer", f"Transfer − chance\n(0 = like {RA}, 1 = like AH)", "WH decoded as AH (probability)", (0, 1)),
              ("num_prob", f"P(AH|WH) − P(AH|{RA}) − chance", f"WH more AH-like than {RA}", None)]
        # e-h: bootstrap distributions at the largest M (equivalents of a-d)
        boot = PB_.get(f"all|{Mmax}") if PB_ else None
        names = ["transfer", "lambda", "bacc", "transfer_raw", "lambda_raw", "bacc_raw", "transfer_bin", "num_prob",
                 "transfer_bin_raw", "num_prob_raw"]
        for ax, (ro, yl, ttl, refs) in zip(axs2, RO):
            r = W_[(W_.readout == ro) & (W_.M == Mmax)].iloc[0]
            for k in GROUPS:
                x = XS[k]
                if boot is not None:
                    v = np.asarray(boot[f"{k[0]} {k[1]}"], float)[:, names.index(ro)]
                    v = v[np.isfinite(v)]
                    vp = ax.violinplot(v, positions=[x], widths=0.7, showextrema=False)
                    for b in vp["bodies"]:
                        b.set_facecolor(COH[k[0]]); b.set_alpha(0.18 if k[1] == "learning" else 0.35); b.set_edgecolor("none")
                g = f"{k[0]} {k[1]}"
                ax.errorbar(x, r[f"mean {g}"], yerr=[[r[f"mean {g}"] - r[f"lo {g}"]], [r[f"hi {g}"] - r[f"mean {g}"]]],
                            fmt="o", ms=3.5, color=COH[k[0]], mfc="white" if k[1] == "learning" else COH[k[0]], mew=0.9,
                            lw=0.9, capsize=0, zorder=3)
            for c, (a_, b_) in [("R+", (GROUPS[0], GROUPS[1])), ("R-", (GROUPS[2], GROUPS[3]))]:
                ax.plot([XS[a_], XS[b_]], [r[f"mean {a_[0]} {a_[1]}"], r[f"mean {b_[0]} {b_[1]}"]], color=COH[c], lw=0.8)
            if refs:
                ax.axhline(0, color=CL["FA"], lw=0.5, ls=(0, (2, 2))); ax.axhline(1, color=CL["AH"], lw=0.5, ls=(0, (2, 2)))
            else:
                ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
            ax.set_xticks([XS[k] for k in GROUPS], ["Learn.", "Expert", "Learn.", "Expert"], fontsize=4.8)
            for x_, c in [(0.5, "R+"), (2.9, "R-")]:
                ax.annotate(c.replace("-", "−"), xy=(x_, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -11),
                            textcoords="offset points", ha="center", va="top", fontsize=6, color=COH[c], weight="bold")
            ax.set_ylabel(yl)
            ax.set_title((f"{ttl} (M = {Mmax})\nchange R+ {fmt_p(r['dR+ p_boot'])}, R− {fmt_p(r['dR- p_boot'])}\ninteraction {fmt_p(r.perm_p_interaction)}"), fontsize=4.9)
            STATS.append(dict(panel=f"5 pp {ro}", measure=f"pseudo-population {ro}", M=Mmax,
                              **{f"mean_{GLAB[k]}": r[f"mean {k[0]} {k[1]}"] for k in GROUPS},
                              **{f"lo_{GLAB[k]}": r[f"lo {k[0]} {k[1]}"] for k in GROUPS},
                              **{f"hi_{GLAB[k]}": r[f"hi {k[0]} {k[1]}"] for k in GROUPS},
                              dRp=r["dR+"], dRp_p_boot=r["dR+ p_boot"], dRm=r["dR-"], dRm_p_boot=r["dR- p_boot"],
                              interaction=r["interaction"], interaction_p_boot=r["interaction p_boot"],
                              interaction_p_perm=r["perm_p_interaction"]))
        # i-k: dependence on the number of pooled neurons; l: learning -> expert change at the largest M
        for ax, (ro, yl, ttl, refs) in zip(axs3[:3], [RO[0], RO[2], RO[3]]):
            d = W_[W_.readout == ro].sort_values("M")
            for k in GROUPS:
                g = f"{k[0]} {k[1]}"
                ax.fill_between(d.M, d[f"lo {g}"], d[f"hi {g}"], color=COH[k[0]], alpha=0.12 if k[1] == "learning" else 0.22, lw=0)
                ax.plot(d.M, d[f"mean {g}"], color=COH[k[0]], lw=1.0, ls="-" if k[1] == "expert" else (0, (2, 1.5)),
                        marker="o", ms=2.2, mfc=COH[k[0]] if k[1] == "expert" else "white", label=GLAB[k])
            ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
            ax.set_xscale("log"); ax.set_xticks(Ms, [str(m_) for m_ in Ms], fontsize=4.6); ax.minorticks_off()
            ax.set_xlabel("Neurons per pseudo-population"); ax.set_ylabel(yl)
            ax.set_title(f"{ttl}\nvs population size", fontsize=5.2)
        axs3[0].legend(frameon=False, fontsize=4.3, loc="lower left", handlelength=1.6)
        ax = axs3[3]
        labs = {"bacc": "acc.", "transfer_bin": "yes/no", "transfer": "prob.", "num_prob": "num."}
        for i_, ro in enumerate(labs):
            r = W_[(W_.readout == ro) & (W_.M == Mmax)].iloc[0]
            for k_, (c, key) in enumerate([("R+", "dR+"), ("R-", "dR-")]):
                x = i_ + (k_ - 0.5) * 0.3
                ax.errorbar(x, r[key], yerr=[[r[key] - r[f"{key} lo"]], [r[f"{key} hi"] - r[key]]], fmt="o", ms=3,
                            color=COH[c], mfc=COH[c] if r[f"{key} p_boot"] < 0.05 else "white", elinewidth=0.8, capsize=0,
                            label=c.replace("-", "−") if i_ == 0 else None)
            ax.text(i_, 1.02, fmt_p(r.perm_p_interaction).replace("p = ", "").replace("p < ", "<"),
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=4.3)
        ax.axhline(0, color="0.3", lw=0.5)
        ax.set_xticks(range(4), list(labs.values()), fontsize=4.8); ax.set_xlim(-0.6, 3.6)
        ax.set_ylabel("Expert − learning\n(bootstrap 95% CI)")
        ax.set_title(f"Change at M = {Mmax} (filled: p < 0.05)\ntop: interaction p (permutation)", fontsize=5.0, pad=9)
        ax.legend(frameon=False, fontsize=4.6, loc="lower left")
    letter_row(fig, axs, "abcd"); letter_row(fig, axs2, "efgh"); letter_row(fig, axs3, "ijkl")
    fig.suptitle("Figure 5 | Decoders: single sessions (a–d) and hierarchical pseudo-populations (e–l), chance-corrected",
                 x=0.08, y=0.985, ha="left", fontsize=7.5, weight="bold")
    save(fig, out, "Fig5_decoders"); plt.close(fig)


def load_psth_prestart(W):
    """first-lick PSTHs (units x WH/AH/FA x bins) minus each unit's pre-trial baseline for that class: baseline = mean
    over the class's trials of the rate in [-1, -0.015] s before trial start (= raw pre-lick rate - baseline-corrected
    rate stored by 051); 3-bin boxcar. Returns (PB, keys with row_b)."""
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    P, K, edges = [], [], None
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        base = z["raw"].astype(float) - z["rates"].astype(float)            # per-trial pre-trial baseline (Hz)
        lab = z["cls"]
        b = np.column_stack([base[:, lab == c].mean(1) if (lab == c).any() else np.full(len(base), np.nan)
                             for c in m51.CLASSES])
        P.append((z["psth"].astype(np.float32) - b[:, :, None]).astype(np.float32))
        K.append(pd.DataFrame(dict(session_id=r.session_id, electrode_group=z["electrode_group"].astype(str),
                                   cluster_id=z["cluster_id"].astype(str))))
        edges = z["psth_edges"]
    P = np.concatenate(P)
    P = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, "same"), 2, P)
    K = pd.concat(K, ignore_index=True); K["row_b"] = np.arange(len(K))
    tc = (edges[:-1] + np.diff(edges) / 2) * 1e3
    return P, K, tc


def mouse_psth(ax, W, PB, tc, mask, sign=None, min_units=2, legend=False, lw=0.9, by="session_id"):
    """rule: mean over units within each session (by=session_id, the unit of analysis), then mean +- s.e.m. over
    sessions (by=mouse_id averages over mice instead); shaded areas without edges"""
    sub = W[mask]
    rows, mice = sub.row_b.to_numpy(), sub[by].to_numpy()
    sg = np.ones(len(rows)) if sign is None else sign[mask]
    per = []
    for m in np.unique(mice):
        k = mice == m
        if k.sum() >= min_units:
            per.append((PB[rows[k]] * sg[k][:, None, None]).mean(0))
    ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0, edgecolor="none")
    ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
    if not per:
        return 0, 0
    A = np.stack(per)
    for kk, c in enumerate(m51.CLASSES):
        mu = A[:, kk].mean(0); se = A[:, kk].std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * mu
        ax.fill_between(tc, mu - se, mu + se, color=CL[c], alpha=0.25, lw=0, edgecolor="none")
        ax.plot(tc, mu, color=CL[c], lw=lw, label=CLAB[c])
    if legend:
        ax.legend(frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.1, labelspacing=0.2)
    return len(A), len(sub)


def mean_panel(ax, d, col, ylabel, rng, panel, title, ref=None):
    """rule: group mean +- s.e.m. (sessions), learning -> expert lines per cohort, brackets with MWU p"""
    S = group_stats(d, col, title, rng, panel)
    xs = {"R+": (0.0, 1.0), "R-": (2.4, 3.4)}
    for c, (x0, x1) in xs.items():
        m = [S["means"][(c, s)] for s in ("learning", "expert")]
        e = [S["sems"][(c, s)] for s in ("learning", "expert")]
        ax.plot([x0, x1], m, color=COH[c], lw=1.1, zorder=2)
        for x, mm, ee, s in zip((x0, x1), m, e, ("learning", "expert")):
            ax.errorbar(x, mm, ee, fmt="o", ms=4.5, color=COH[c], mfc="white" if s == "learning" else COH[c],
                        mew=1.0, lw=1.1, capsize=0, zorder=3)
    if ref is not None:
        for yv, c in ref:
            ax.axhline(yv, color=c, lw=0.6, ls=(0, (2, 2)), zorder=1)
    lo, hi = ax.get_ylim(); r_ = hi - lo
    for name, (x0, x1), c, yb in [("R+ L vs E", (0, 1), COH["R+"], hi + 0.08 * r_), ("R- L vs E", (2.4, 3.4), COH["R-"], hi + 0.08 * r_),
                                  ("expert R+ vs R-", (1, 3.4), "0.15", hi + 0.3 * r_)]:
        if name in S:
            ax.plot([x0, x0, x1, x1], [yb - 0.03 * r_, yb, yb, yb - 0.03 * r_], color=c, lw=0.6)
            ax.text((x0 + x1) / 2, yb + 0.01 * r_, fmt_p(S[name][0]), ha="center", va="bottom", fontsize=4.8, color=c)
    ax.set_ylim(lo, hi + 0.48 * r_)
    ax.set_xlim(-0.5, 3.9)
    ax.set_xticks([0, 1, 2.4, 3.4], ["Learn.", "Expert", "Learn.", "Expert"])
    for x, c in [(0.5, "R+"), (2.9, "R-")]:                           # fixed offset in points: independent of axis height
        ax.annotate(c.replace("-", "−"), xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -13),
                    textcoords="offset points", ha="center", va="top", fontsize=6, color=COH[c], weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nLearning × cohort interaction {fmt_p(S['interaction'][1])}", fontsize=5.6, pad=3)
    return S


def fig_cosyne3(plt, D, out, pop, rng, PR):
    """COSYNE v3 (6 panels): a population PSTHs; b converging neurons; c population geometry (triangles);
    d distance difference; e decoder probability numerator (accuracy in the title); f area heatmap of mean distance
    difference (area groups with >= 3 sessions in all four cohort x stage groups)"""
    RN = "false alarm" if m51.REF == "fa" else "spontaneous lick"
    RA = "FA" if m51.REF == "fa" else "SL"
    import ephys_utilities.allen_utils.allen_utils as au
    W = D["W"].copy()
    PB, KB, tcb = load_psth_prestart(W)
    W = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    L = D["lam"]; Lwb = L[(L.level == "all") & (L.variant == "all")]
    LA = L[(L.level == "area_group") & (L.variant == "all")]
    T = D["tr"]; T = T[(T.level == "all") & (T.variant == "all")]
    DC = D["dec"]; DC = DC[DC.level == "all"]
    with plt.rc_context({"font.size": 5.5, "axes.titlesize": 5.8, "axes.labelsize": 5.5, "xtick.labelsize": 5,
                         "ytick.labelsize": 5, "legend.fontsize": 5}):
        fig = plt.figure(figsize=(W_IN, 4.9))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.55, left=0.075, right=0.945, top=0.9,
                                 bottom=0.09)
        gs = outer[0].subgridspec(1, 6, wspace=0.8, width_ratios=[1, 1, 1, 1, 1.25, 1.05])
        gl = outer[1].subgridspec(1, 3, wspace=0.55, width_ratios=[1, 1, 1.35])
        # a: PSTHs
        sub = gs[0, 0:4].subgridspec(1, 4, wspace=0.18)
        tested = (W["sig:auditory_hit_vs_fa_prelick@all"].notna() & W["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
        axs_a = []
        for j, k in enumerate(GROUPS):
            ax = fig.add_subplot(sub[j]); axs_a.append(ax)
            nm, nu = mouse_psth(ax, W, PB, tcb, tested & ((W.cohort == k[0]) & (W.stage == k[1])).to_numpy(), legend=(j == 0))
            ax.set_xlim(-500, 250); ax.set_xticks([-400, -200, 0, 200]); ax.set_xticklabels(["−400", "", "0", ""])
            ax.set_title(f"{GLAB[k]} ({nm} sess.)", color=COH[k[0]], pad=2)
        yl = (min(a.get_ylim()[0] for a in axs_a), max(a.get_ylim()[1] for a in axs_a))
        for j, ax in enumerate(axs_a):
            ax.set_ylim(*yl)
            if j:
                ax.set_yticklabels([])
        axs_a[0].set_ylabel("Rate − pre-trial\nbaseline (Hz)")
        x0, x1 = axs_a[0].get_position().x0, axs_a[-1].get_position().x1
        fig.text((x0 + x1) / 2, axs_a[0].get_position().y0 - 0.075, "Time from first lick (ms); mean ± s.e.m. over sessions",
                 ha="center", fontsize=5.3)
        # b: converging neurons
        ax_b = fig.add_subplot(gs[0, 4])
        mean_panel(ax_b, T, "frac_transfer", "Fraction of reward-lick neurons", rng, "C4-b",
                   f"Converging neurons\n(AH ≠ {RA} and WH ≠ {RA}, same sign)")
        ax_b.set_title(ax_b.get_title(), fontsize=5.2); ax_b.set_xticklabels(["L", "E", "L", "E"])
        # c: geometry (triangles), R+ above R-
        gc = gs[0, 5].subgridspec(2, 1, hspace=0.25)
        axs_c = [fig.add_subplot(gc[i]) for i in range(2)]
        for ax, c in zip(axs_c, ["R+", "R-"]):
            draw_triangles(ax, Lwb, c, RA)
            ax.set_title(f"{c.replace('-', '−')}: L (dashed) → E", color=COH[c], fontsize=4.8, pad=1)
        share_triangle_limits(axs_c)
        from matplotlib.lines import Line2D
        axs_c[1].legend([Line2D([], [], marker="o", ls="", color=CL[c], ms=2.8) for c in ["FA", "AH", "WH"]],
                        [RA, "AH", "WH"], frameon=False, fontsize=4.4, loc="upper center", ncol=3, handletextpad=0.1,
                        columnspacing=0.6, bbox_to_anchor=(0.5, 0.02))
        # d: distance difference
        ax_d = fig.add_subplot(gl[0])
        mean_panel(ax_d, Lwb, "dd", f"Δd = d(WH,{RA}) − d(WH,AH)\n(> 0: WH closer to AH)", rng, "C4-d",
                   f"Population distance:\nare whisker hits closer to AH than to {RA}?", ref=[(0, "0.6")])
        ax_d.set_title(ax_d.get_title(), fontsize=5.2)
        # e: decoder numerator; accuracy reported in the title
        ax_e = fig.add_subplot(gl[1])
        Se = group_stats(DC, "bacc_corrected", "acc", rng, "C4-e acc")
        mean_panel(ax_e, DC, "num_prob_corrected", f"P(AH|WH) − P(AH|{RA}) − chance", rng, "C4-e",
                   f"Decoder AH vs {RA} applied to WH\n(accuracy unchanged: interaction {fmt_p(Se['interaction'][1])})",
                   ref=[(0, "0.6")])
        ax_e.set_title(ax_e.get_title(), fontsize=5.2)
        # f: area heatmap
        Q = area_qualify(LA, col="dd")
        order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] and g in Q["R-"]]
        short = {g: g.replace(" areas", "").replace("Somatosensory-", "SS-").replace("Lateral septal complex", "LSX")
                 .replace("Posterior parietal", "PPC").replace("Retrosplenial", "RSP").replace("Hippocampus", "HPF")
                 .replace("Amygdala and hypothalamus", "Amyg.") for g in order}
        ax_f = fig.add_subplot(gl[2])
        M = np.full((len(order), 4), np.nan); star = np.zeros((len(order), 2), bool)
        for i_, g in enumerate(order):
            for j_, k in enumerate(GROUPS):
                M[i_, j_] = LA[(LA.region == g) & (LA.cohort == k[0]) & (LA.stage == k[1])].dd.mean()
            for c_, c in enumerate(["R+", "R-"]):
                d = LA[(LA.region == g) & (LA.cohort == c)].dropna(subset=["dd"])
                star[i_, c_] = stats.mannwhitneyu(d[d.stage == "learning"].dd, d[d.stage == "expert"].dd).pvalue < 0.05
        vlim = float(np.nanmax(np.abs(M)))
        im = ax_f.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        for i_ in range(M.shape[0]):
            for j_ in range(4):
                t = f"{M[i_, j_]:+.2f}".replace("-", "−") + ("*" if j_ in (1, 3) and star[i_, j_ // 2] else "")
                ax_f.text(j_, i_, t, ha="center", va="center", fontsize=4.2, color="white" if abs(M[i_, j_]) > 0.6 * vlim else "k")
        ax_f.set_xticks(range(4), ["R+ L", "R+ E", "R− L", "R− E"]); ax_f.xaxis.tick_top()
        for t, c in zip(ax_f.get_xticklabels(), ["R+", "R+", "R-", "R-"]):
            t.set_color(COH[c])
        ax_f.set_yticks(range(len(order)), [short[g] for g in order])
        cb = fig.colorbar(im, ax=ax_f, fraction=0.05, pad=0.03)
        cb.set_label("Mean Δd (> 0: WH closer to AH)", fontsize=4.8); cb.ax.tick_params(labelsize=4.5)
        ANc = area_anova(LA, order, col="dd")
        for k_, v_ in (ANc or {}).items():
            STATS.append(dict(panel="C4-f ANOVA", measure=k_, F=v_["F"], p=v_["p"], p_perm=v_["p_perm"], n_areas=len(order)))
        ttl = "Δd per area group (*: learning → expert p < 0.05)"
        if ANc:
            ttl += "\n" + f"ANOVA: cohort × stage {fmt_p(ANc['cohort x stage']['p_perm'])}, × area {fmt_p(ANc['cohort x stage x area']['p_perm'])}"
        ax_f.set_title(ttl, pad=12, fontsize=5.0)
        letter_row(fig, [axs_a[0], ax_b, axs_c[0]], "abc", dx_in=0.33, dy_in=0.12)
        letter_row(fig, [ax_d, ax_e, ax_f], "def", dx_in=0.42, dy_in=0.3)
        fig.suptitle(f"Before the lick, whisker hits move toward auditory hits when whisker licks are rewarded (R+), not otherwise (R−)",
                     x=0.075, y=0.995, ha="left", fontsize=7, weight="bold")
        save(fig, out, "COSYNE_figure_v3"); plt.close(fig)


def fig_cosyne2(plt, D, out, pop, rng, PR):
    """condensed COSYNE figure on learning -> expert transitions; style per skills/ssl-figure-style"""
    RN = "false alarm" if m51.REF == "fa" else "spontaneous lick"     # unrewarded-lick reference
    RA = "FA" if m51.REF == "fa" else "SL"
    import ephys_utilities.allen_utils.allen_utils as au
    W = D["W"].copy()
    PB, KB, tcb = load_psth_prestart(W)
    W = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    L = D["lam"]; Lwb = L[(L.level == "all") & (L.variant == "all")]
    LA = L[(L.level == "area_group") & (L.variant == "all")]
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    with plt.rc_context({"font.size": 5.5, "axes.titlesize": 5.8, "axes.labelsize": 5.5, "xtick.labelsize": 5,
                         "ytick.labelsize": 5, "legend.fontsize": 5}):
        fig = plt.figure(figsize=(W_IN, 5.2))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.85], hspace=0.42, left=0.075, right=0.99, top=0.91,
                                 bottom=0.075)
        gs = outer[0].subgridspec(1, 6, wspace=0.75)
        gl = outer[1].subgridspec(2, 4, height_ratios=[1.15, 0.62], width_ratios=[1.05, 1.25, 1.25, 1.45],
                                  hspace=0.9, wspace=0.62)
        # a: population PSTHs (mouse-averaged, pre-trial baseline)
        sub = gs[0, 0:4].subgridspec(1, 4, wspace=0.18)
        tested = (W["sig:auditory_hit_vs_fa_prelick@all"].notna() & W["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
        axs_a = []
        for j, k in enumerate(GROUPS):
            ax = fig.add_subplot(sub[j]); axs_a.append(ax)
            nm, nu = mouse_psth(ax, W, PB, tcb, tested & ((W.cohort == k[0]) & (W.stage == k[1])).to_numpy(), legend=(j == 0))
            ax.set_xlim(-500, 250); ax.set_xticks([-400, -200, 0, 200]); ax.set_xticklabels(["−400", "", "0", ""])
            ax.set_title(f"{GLAB[k]} ({nm} sessions)", color=COH[k[0]], pad=2)
        yl = (min(a.get_ylim()[0] for a in axs_a), max(a.get_ylim()[1] for a in axs_a))
        for j, ax in enumerate(axs_a):
            ax.set_ylim(*yl)
            if j:
                ax.set_yticklabels([])
        axs_a[0].set_ylabel("Firing rate − pre-trial\nbaseline (Hz)")
        x0, x1 = axs_a[0].get_position().x0, axs_a[-1].get_position().x1
        fig.text((x0 + x1) / 2, axs_a[0].get_position().y0 - 0.055, "Time from first lick (ms); mean ± s.e.m. over sessions",
                 ha="center", fontsize=5.5)
        # b: group geometry of the three class means (triangles), learning dashed -> expert solid
        sub = gs[0, 4:6].subgridspec(1, 2, wspace=0.12)
        axs_b = []
        for j, c in enumerate(["R+", "R-"]):
            ax = fig.add_subplot(sub[j]); axs_b.append(ax); draw_triangles(ax, Lwb, c, RA)
            dd_ = {s_: Lwb[(Lwb.cohort == c) & (Lwb.stage == s_)].dd.mean() for s_ in ["learning", "expert"]}
            ax.set_title(f"{c.replace('-', '−')}: Δd {dd_['learning']:+.3f} → {dd_['expert']:+.3f}".replace("-", "−"),
                         color=COH[c], pad=2)
        share_triangle_limits(axs_b)
        from matplotlib.lines import Line2D
        axs_b[0].legend([Line2D([], [], marker="o", ls="", color=CL[c], ms=3) for c in ["FA", "AH", "WH"]],
                        [RA, "AH", "WH"], frameon=False, fontsize=4.6, loc="upper center", handletextpad=0.1, ncol=3,
                        bbox_to_anchor=(1.06, 0.02), columnspacing=0.8)
        fig.text((axs_b[0].get_position().x0 + axs_b[1].get_position().x1) / 2, axs_b[0].get_position().y0 - 0.085,
                 "Mean population geometry (cross-validated distances);\ndashed: learning, solid: expert", ha="center",
                 fontsize=5.3)
        # c: population similarity (mean +- sem over sessions)
        T = D["tr"]; T = T[(T.level == "all") & (T.variant == "all")]
        S = session_unit_metrics(W)
        ax_c = fig.add_subplot(gl[0:2, 0])
        mean_panel(ax_c, Lwb, "dd", f"Distance difference d(WH,{RA}) − d(WH,AH)\n(> 0: whisker hits closer to auditory hits)",
                   rng, "C3-c", f"Population: are whisker hits closer\nto AH than to {RA}?", ref=[(0, "0.6")])
        # d / e: single-cell fractions + PSTHs of the quantified units
        ax_d = fig.add_subplot(gl[0, 1])
        mean_panel(ax_d, T, "frac_transfer", "Fraction of reward-lick neurons", rng, "C3-d",
                   f"Reward-lick neurons (AH ≠ {RA}) that\nalso separate whisker hits from {RA}")
        ax_e = fig.add_subplot(gl[0, 2])
        mean_panel(ax_e, S, "frac_sig_WHvsFA", "Fraction of all units", rng, "C3-e",
                   f"Units separating whisker\nhits from {RN}s")
        sig_af = (W[f"sig:{af}"] == 1).to_numpy(); sig_wf = (W[f"sig:{wf}"] == 1).to_numpy()
        s_af, s_wf = np.sign(W[f"sel:{af}"].fillna(0).to_numpy()), np.sign(W[f"sel:{wf}"].fillna(0).to_numpy())
        transfer_units = sig_af & sig_wf & (s_af == s_wf)
        sub_d = gl[1, 1].subgridspec(1, 2, wspace=0.12)
        sub_e = gl[1, 2].subgridspec(1, 2, wspace=0.12)
        minis = []
        for spec, units, sg, lab in [(sub_d, transfer_units, s_af, "d"), (sub_e, sig_wf, s_wf, "e")]:
            for j, k in enumerate([GROUPS[1], GROUPS[3]]):
                ax = fig.add_subplot(spec[j]); minis.append(ax)
                nm, nu = mouse_psth(ax, W, PB, tcb, units & ((W.cohort == k[0]) & (W.stage == k[1])).to_numpy(), sign=sg, lw=0.8)
                ax.set_xlim(-400, 200); ax.set_xticks([-400, 0, 200])
                ax.set_xticklabels(["−400", "0", "200"] if j == 0 else ["", "0", "200"], fontsize=4.4)
                ax.tick_params(labelsize=4.4)
                ax.set_title(f"{GLAB[k]}\n{nm} sess., {nu} units", color=COH[k[0]], fontsize=4.4, pad=1.5)
                if j == 0:
                    ax.set_ylabel("Sign-aligned\nΔ rate (Hz)", fontsize=4.6)
            pair = minis[-2:]
            yl2 = (min(a.get_ylim()[0] for a in pair), max(a.get_ylim()[1] for a in pair))
            for a in pair:
                a.set_ylim(*yl2)
            pair[1].set_yticklabels([])
        fig.text((minis[0].get_position().x0 + minis[3].get_position().x1) / 2, minis[0].get_position().y0 - 0.06,
                 "Average of the counted units: time from first lick (ms), mean ± s.e.m. over sessions", ha="center", fontsize=5)
        # f: areas, heatmap of mean distance difference (areas included in both cohorts only)
        Q = area_qualify(LA, col="dd")
        order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] and g in Q["R-"]]
        short = {g: g.replace(" areas", "").replace("Somatosensory-", "SS-").replace("Lateral septal complex", "LSX")
                 .replace("Posterior parietal", "PPC").replace("Retrosplenial", "RSP").replace("Hippocampus", "HPF")
                 .replace("Amygdala and hypothalamus", "Amyg.") for g in order}
        ax_f = fig.add_subplot(gl[0:2, 3])
        M = np.full((len(order), 4), np.nan); star = np.zeros((len(order), 2), bool)
        for i_, g in enumerate(order):
            for j_, k in enumerate(GROUPS):
                M[i_, j_] = LA[(LA.region == g) & (LA.cohort == k[0]) & (LA.stage == k[1])].dd.mean()
            for c_, c in enumerate(["R+", "R-"]):
                d = LA[(LA.region == g) & (LA.cohort == c)].dropna(subset=["dd"])
                star[i_, c_] = stats.mannwhitneyu(d[d.stage == "learning"].dd, d[d.stage == "expert"].dd).pvalue < 0.05
        vlim = float(np.nanmax(np.abs(M)))
        im = ax_f.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        for i_ in range(M.shape[0]):
            for j_ in range(4):
                t = f"{M[i_, j_]:+.2f}".replace("-", "−")
                if j_ in (1, 3) and star[i_, j_ // 2]:
                    t += "*"
                ax_f.text(j_, i_, t, ha="center", va="center", fontsize=4.3,
                          color="white" if abs(M[i_, j_]) > 0.6 * vlim else "k")
        ax_f.set_xticks(range(4), ["R+ L", "R+ E", "R− L", "R− E"]); ax_f.xaxis.tick_top()
        for t, c in zip(ax_f.get_xticklabels(), ["R+", "R+", "R-", "R-"]):
            t.set_color(COH[c])
        ax_f.set_yticks(range(len(order)), [short[g] for g in order])
        cb = fig.colorbar(im, ax=ax_f, fraction=0.06, pad=0.03, location="bottom")
        cb.set_label("Mean Δd (> 0: WH closer to AH)", fontsize=5); cb.ax.tick_params(labelsize=4.5)
        ANc = area_anova(LA, order, col="dd")
        ttl = "Mean Δd, areas in both cohorts (*: L → E p < 0.05)"
        if ANc:
            ttl += chr(10) + f"ANOVA: cohort × stage {fmt_p(ANc['cohort x stage']['p_perm'])}, × area {fmt_p(ANc['cohort x stage x area']['p_perm'])}"
        ax_f.set_title(ttl, pad=12, fontsize=4.8)
        letter_row(fig, [axs_a[0], axs_b[0]], "ab", dx_in=0.33, dy_in=0.12)
        letter_row(fig, [ax_c, ax_d, ax_e], "cde", dx_in=0.38, dy_in=0.3)
        letter_row(fig, [ax_f], "f", dx_in=0.6, dy_in=0.3)
        fig.suptitle("Learning to expert: before the lick, whisker hits move closer to auditory hits in R+, not R−, mice",
                     x=0.075, y=0.992, ha="left", fontsize=7, weight="bold")
        save(fig, out, "COSYNE_figure_v2"); plt.close(fig)


# ------------------------------------------------------------------ captions
def _st(panel):
    rows = [r for r in STATS if r.get("panel") == panel]
    if not rows:
        return ""
    r = rows[0]
    n = ", ".join(f"{GLAB[k]} {r.get(f'n_{GLAB[k]}', 0)}" for k in GROUPS)
    parts = [f"n sessions: {n}."]
    for name, lab in [("R+ L vs E", "R+ learning vs expert"), ("R- L vs E", "R− learning vs expert"),
                      ("expert R+ vs R-", "expert R+ vs R−")]:
        if f"{name} p_MWU" in r and np.isfinite(r.get(f"{name} p_MWU", np.nan)):
            parts.append(f"{lab}: Δ = {r[f'{name} diff']:+.3f}, MWU {fmt_p(r[f'{name} p_MWU'])}, "
                         f"Welch {fmt_p(r[f'{name} p_Welch'])}.")
    if np.isfinite(r.get("interaction_p_perm", np.nan)):
        parts.append(f"Learning × cohort interaction [Δ(R+) − Δ(R−)] = {r['interaction']:+.3f}, "
                     f"permutation {fmt_p(r['interaction_p_perm'])}.")
    return " ".join(parts)


def _row(panel):
    rows = [r for r in STATS if r.get("panel") == panel]
    return rows[0] if rows else None


def _st2f(coh):
    r = _row(f"2f {coh}")
    if r is None:
        return ""
    return (f"{coh.replace('-', '−')}: learning r = {r['r_learning']:.2f} (slope {r['slope_learning']:.2f}; "
            f"{r['n_units_learning']} units, {r['n_sessions_learning']} sessions), expert r = {r['r_expert']:.2f} "
            f"(slope {r['slope_expert']:.2f}; {r['n_units_expert']} units, {r['n_sessions_expert']} sessions); "
            f"Δr = {r['dr']:+.2f} [95% CI {r['dr_lo']:+.2f}, {r['dr_hi']:+.2f}], session-bootstrap {fmt_p(r['p_boot'])}.")


def _stpp(ro):
    r = _row(f"5 pp {ro}")
    if r is None:
        return ""
    m = ", ".join(f"{GLAB[k]} {r[f'mean_{GLAB[k]}']:.2f} [{r[f'lo_{GLAB[k]}']:.2f}, {r[f'hi_{GLAB[k]}']:.2f}]"
                  for k in GROUPS if f"lo_{GLAB[k]}" in r)
    return (f"Means [95% bootstrap interval]: {m}. Change R+ {r['dRp']:+.2f} (bootstrap {fmt_p(r['dRp_p_boot'])}), "
            f"R− {r['dRm']:+.2f} ({fmt_p(r['dRm_p_boot'])}); interaction {r['interaction']:+.2f}, permutation "
            f"{fmt_p(r['interaction_p_perm'])}.")


def _anova(prefix):
    rows = [r for r in STATS if r.get("panel") == prefix]
    if not rows:
        return ""
    lab = {"cohort": "cohort", "area": "area", "cohort x area": "cohort × area", "cohort x stage": "cohort × stage",
           "cohort x stage x area": "cohort × stage × area"}
    return "; ".join(f"{lab[r['measure']]}: F = {r['F']:.2f}, F-test {fmt_p(r['p'])}, permutation {fmt_p(r['p_perm'])}"
                     for r in rows) + "."


def write_captions(D, out, pop):
    RA = "FA" if m51.REF == "fa" else "SL"
    RN = "false alarm" if m51.REF == "fa" else "spontaneous lick"
    RNP = RN + "s"
    if m51.REF == "fa":
        refdef = ("false alarms (FA): first lick on a no-stimulus (catch) trial, aligned like hits to the corrected first "
                  "lick; baseline = 1 s before trial start")
    else:
        refdef = ("spontaneous licks (SL): licks emitted outside stimulus trials (no-stimulus periods), taken in time "
                  "order with the trials; baseline = [−1, −0.5] s before the lick")
    NULLTXT = ("linear shifts of the neural activity relative to the time-ordered event labels (|shift| 5 to N/3 events, "
               "no wrap-around; keeps the temporal autocorrelation of event types and of neural activity, e.g. engagement "
               "drifts; the same null for both references)")
    popt = ("all mice" if pop == "all" else
            "learners (day-0 sessions of mice with learning category good or moderate; expert sessions of all mice)")
    common = (f"**General.** Pre-lick window: the 100 ms before the first lick of each event, the first lick corrected for "
              f"the stimulus-artefact window (trial start + lick time − response-window start). Event classes: whisker hit "
              f"(WH, lick after a whisker stimulus), auditory hit (AH, lick after an auditory stimulus) and the unrewarded-lick "
              f"reference, {refdef}. Trials: active, perf ≠ 6, auditory warm-up block removed (one trial before the first "
              f"whisker trial kept), end-of-session disengagement trimmed (rule A1). Units: Kilosort 4, quality good or "
              f"mua, mean raw pre-lick rate ≥ 0.1 Hz. Hit rates are baseline-corrected per trial (rate in [−1, −0.015] s "
              f"before trial start subtracted). Learning = first whisker-training day (day 0); expert = later days. "
              f"Unit of analysis: the session. Tests: two-sided Mann-Whitney U (brackets) and Welch t (captions); "
              f"learning × cohort interaction [Δ(R+) − Δ(R−)] by permuting cohort labels across mice (10,000 permutations). "
              f"No correction across panels. In dot panels: small symbols = sessions (open: learning, filled: expert), "
              f"large symbols = mean ± s.e.m. over sessions. Population: {popt}.")
    A = D.get("fig4_areas", {})
    C = ["# Figure captions (" + popt + ")\n\n" + common + "\n"]
    C.append(f"""
## Figure 1 | Pre-lick activity on whisker hits, auditory hits and {RNP}
**a**, Task and reward contingencies. Mice lick after a whisker or an auditory stimulus; auditory licks are rewarded in
both cohorts, whisker licks only in R+ mice; {RNP} are never rewarded. The definition of a whisker hit is identical in
the two cohorts; only its outcome differs.
**b**, Alignment. All analyses use the 100 ms preceding the corrected first lick (shaded), so that the three event types
are compared at the same motor event; the baseline is the 1 s before trial start.
**c**, Reaction-time distributions of whisker and auditory hits (corrected first lick relative to stimulus onset),
pooled over trials of all sessions; legend: medians.{'' if m51.REF == 'fa' else ' Spontaneous licks have no stimulus and therefore no reaction time.'}
**d**, Median reaction time per session for each hit type, by cohort and stage (mean ± s.e.m. over sessions). Whisker
reaction times shorten to auditory-like values in R+ but not in R− mice.
**e**, Dataset: number of sessions per cohort and stage (bars), with the numbers of mice and of tested units.
**f**, Positions of all recorded units in the Allen CCF (sagittal and coronal projections; random subset of 30,000 units;
R+ green, R− magenta).
**g**, First-lick-aligned population PSTHs of all tested units for WH, AH and {RNP} per cohort and stage: per unit, firing
rate minus its pre-trial baseline for that event type; mean over units within each session, then mean ± s.e.m. over
sessions (10 ms bins, 30 ms boxcar). Shaded: pre-lick window.
**h**, Quantification of g: mean rate change in the pre-lick window per session for each event type (first three panels),
and the ratio of the whisker-hit and auditory-hit changes to the {RN} change (last two panels; sessions whose {RN} change
is < 0.5 Hz are excluded from the ratios). Whisker hits: {_st('1h WH')} Auditory hits: {_st('1h AH')}
{RNP.capitalize()}: {_st('1h ref')} WH / {RA}: {_st('1h WH/ref')} AH / {RA}: {_st('1h AH/ref')}
""")
    C.append(f"""
## Figure 2 | Single neurons
**a**, Example neurons of each type in R+ (top) and R− (bottom) experts (good units; the unit with the largest
selectivity × log firing rate among those meeting the type criteria): converging (AH vs {RA} and WH vs {RA} both
significant, same sign), reward-lick-only (AH vs {RA} significant, |WH vs {RA}| < 0.1) and whisker-hit-only (WH vs {RA}
significant, |AH vs {RA}| < 0.1). Top: PSTH (10 ms bins, Gaussian σ = 20 ms; mean ± s.e.m. over events); bottom: raster
of up to 25 events per type. Titles: area and ROC selectivity (2·AUC − 1).
**b–e, top row**, Schematic definitions of the four single-neuron quantifications.
**b**, Fraction of tested units per session with a significant WH vs {RA} pre-lick ROC (rate-based ROC; significance by
1,000 label permutations, one-tailed p < 0.05 on the side of the observed selectivity; sessions with ≥ 10 tested units).
{_st('2b')}
**c**, Converging neurons: among reward-lick neurons (significant AH vs {RA}; ≥ 10 per session), the fraction also
significant for WH vs {RA} with the same sign. {_st('2c')}
**d**, AH-likeness of reward-lick neurons, c = (|WH − {RA}| − |WH − AH|) / |AH − {RA}| from trial-mean pre-lick rates
(−1: WH equals {RA}; +1: WH equals AH), averaged per session. {_st('2d')}
**e**, Shared hit code: Spearman correlation across units of the WH vs {RA} and AH vs {RA} selectivities, per session.
{_st('2e')}
**f**, Unit-level selectivities (WH vs {RA} against AH vs {RA}) for each cohort, learning and expert units merged (grey:
log density of units), with separate least-squares fits for learning (dashed) and expert (solid) units; dotted: identity.
Pearson r per stage and the difference r(expert) − r(learning), tested by resampling sessions within each stage (2,000
session bootstraps of the pooled r). {_st2f('R+')} {_st2f('R-')}
**g**, Converging neurons per brain area (Allen-based custom acronyms; expert sessions): percentage of tested units that
are converging neurons, per session × area, mean ± s.e.m. over sessions (dots: sessions). All areas recorded in expert
sessions of both cohorts are shown (no minimum count), ordered by area group (grey separators). ANOVA on session × area
values, fraction ~ cohort × area (type II OLS); p values from 1,000 permutations of cohort labels across mice.
{_anova('2g ANOVA')}
""")
    C.append(f"""
## Figure 2—supplement | Converging neurons: examples, functional types and anatomy
**a**, Example reward-lick neurons (PSTH only; good units, mean pre-lick rate ≥ 8 Hz, AH vs {RA} selectivity ≥ 0.6): two
R+ expert neurons that also separate WH from {RA} with the same sign and two R− expert neurons that do not
(|WH vs {RA}| < 0.1); 10 ms bins, 20 ms boxcar; rate minus the unit's mean in [−600, −400] ms.
**b**, Functional types of converging neurons, reward-lick-only neurons and all tested units: percentage of each subset
significant in the stimulus-aligned ROC analyses of the main pipeline (whole brain, per cohort and stage; pooled over
neurons; n = neurons per subset). Right: enrichment of converging neurons relative to all tested units (ratio; colour:
log2 ratio).
**c**, **d**, Where are the converging neurons? Expert sessions; slabs 500 µm thick, coronal every 1 mm (c; AP centre) and
sagittal every 0.5 mm (d; ML distance from the midline, hemispheres folded). Rows 1 and 3: all tested neurons of the slab
(sampling; one dot per neuron) for R+ and R−. Rows 2 and 4: density of converging neurons (neurons per mm² of the
projected slab; 50 µm bins, Gaussian σ = 150 µm), on one discrete colour scale per panel shared by both cohorts. Grey:
Allen CCF region boundaries (layers and barrels merged).
""")
    C.append(f"""
## Figure 3 | Population distance
**a**, Schematic. Each event type is summarised by its mean pre-lick population vector (units z-scored over the used
trials). Distances are cross-validated squared Euclidean distances per unit: the trials of each class are split into two
random halves a and b, d(X, Y) = (X_a − Y_a)·(X_b − Y_b) / n_units, averaged over 50 splits. Because noise is
independent between halves, d is unbiased (expectation 0 for identical means; single estimates can be slightly
negative). Distance difference Δd = d(WH, {RA}) − d(WH, AH): > 0 when whisker hits are closer to auditory hits than to
{RNP}. Sessions need ≥ 5 units and ≥ 4 trials per class; no axis threshold is needed because Δd is not a ratio.
**b**, Group-mean geometry of the three class means per cohort: triangle whose side lengths are the square roots of the
group-mean distances (negative values set to 0); learning dashed, open symbols; expert solid, filled symbols; same scale
for both cohorts.
**c**, Δd per session (whole brain). {_st('3c')}
**d**, d(WH, {RA}). {_st('3d')}
**e**, d(WH, AH). {_st('3e')}
**f**, d(AH, {RA}): separation of rewarded and unrewarded licks (scale of the comparison). {_st('3f')}
**g**, Δd with reaction-time-matched trials (the three classes subsampled to identical RT histograms in 50 ms bins).
{_st('3g') if m51.REF == 'fa' else 'Not defined for spontaneous licks, which have no reaction time.'}
**h**, Cumulative distributions of session Δd per group.
**i**, Displacement of whisker hits from the reference along the reference → AH direction, (WH − ref)·(AH − ref) per
unit, computed from the same cross-validated distances as [d(WH, ref) + d(AH, ref) − d(WH, AH)] / 2 (> 0: WH displaced
toward AH). {_st('3i')}
**j**, Part of the displacement orthogonal to that direction, d(WH, ref) − along² / d(AH, ref) (sessions with
d(AH, ref) ≥ 0.01). {_st('3j')}
""")
    C.append(f"""
## Figure 3—supplement | λ and λ_LDA
**a**, λ: the cross-validated projection of the whisker-hit population vector onto the axis from {RNP} to auditory hits,
λ = mean[(WH_a − R_a)·(AH_b − R_b)] / mean[(AH_a − R_a)·(AH_b − R_b)] over 50 splits (R: reference), i.e. the mixing
weight in WH ≈ (1 − λ)·R + λ·AH. λ = 0: whisker hits like {RNP}; λ = 1: like auditory hits. Cases need ≥ 5 units, ≥ 4
trials per class and a reliable axis (d(AH, R) ≥ 0.01 per unit); λ is clipped to [−1, 2]. λ is the along-axis component
of Fig. 3i divided by the axis length d(AH, R); the ratio makes it noisy when the axis is short.
**b**, Single-trial projections for one example R+ and R− expert session (the session with the median λ of its group):
axis = AH − {RA} mean difference from training folds (5-fold, stratified); held-out {RA} and AH trials and all WH trials
projected and normalised so that the held-out {RA} mean is 0 and the AH mean 1. Bars: means.
**c**, Pooled single-trial projections over the sessions of each group. Titles: mean ± s.e.m. of session λ.
**d**, λ per session (whole brain). {_st('3S d')}
**e**, λ against Δd across sessions (Spearman).
**f**, λ with RT-matched trials. {_st('3S f') if m51.REF == 'fa' else 'Not defined for spontaneous licks.'}
**g**, Cumulative distributions of session λ.
**h**, Shrinkage LDA, illustrated in one R+ and one R− expert session (median λ_LDA of its group): held-out trials in the
plane spanned by the LDA axis (horizontal; normalised so that the training {RA} mean = 0 and AH mean = 1) and the first
principal component of the training trials orthogonal to it (vertical, same scale). 5-fold: units z-scored and the LDA
(Ledoit-Wolf shrinkage, lsqr solver) fitted on training {RA} and AH trials; held-out {RA} and AH trials and all WH trials
(averaged over folds) projected. Large dots: class means. Arrow: direction of the mean-difference axis (used for λ) in
this plane; the LDA axis weights units by the inverse (shrunk) noise covariance. Grey line: decision boundary (0.5).
**i**, Agreement of λ with λ_LDA across sessions (Spearman). {'' if _row('3S i') is None else f"r = {_row('3S i')['r']:.2f}, {fmt_p(_row('3S i')['p'])}, n = {_row('3S i')['n']}."}
**j**, λ_LDA per session (held-out d′ ≥ 0.3; clipped to [−1, 2]). {_st('3S j')}
**k**, Held-out d′ of the AH vs {RA} LDA axis (axis quality). {_st('3S k')}
**l**, λ_LDA with RT-matched trials. {_st('3S l') if m51.REF == 'fa' else 'Not defined for spontaneous licks.'}
**m**, Cumulative distributions of session λ_LDA.
""")
    C.append(f"""
## Figure 4 | Brain areas
**a**, Change of Δd from learning to expert per area group (mean ± s.e.m. of the difference; filled: Mann-Whitney
p < 0.05). Δd is computed per session × area group (≥ 5 units). An area group is shown for a cohort if that cohort has
≥ 3 sessions with a valid Δd at both stages; area groups need not be shared across cohorts. Right: learning × cohort
interaction p (permutation) and, in brackets, the family-wise p across area groups (max-|z| permutation), for area groups
included in both cohorts. Included: R+ {len(A.get('Q', {}).get('R+', []))}, R− {len(A.get('Q', {}).get('R-', []))},
both {len(A.get('both', []))} area groups. Inset: ANOVA Δd ~ cohort × stage × area (type II OLS on session × area
values, area groups included in both cohorts); p from the F test and from 1,000 permutations of cohort labels across mice
(sessions contribute several area groups). {_anova('4a ANOVA')}
**b**, Mean Δd per area group, cohort and stage (grey: area not included for that cohort).
**c**, Δd per session for four area groups included in both cohorts ({', '.join(A.get('focus', []))}).
{' '.join(f'{g}: ' + _st(f'4c {g}') for g in A.get('focus', []))}
**d**, First-lick PSTHs of expert sessions of R+ and R− mice in {', '.join(A.get('focus', [])[:2])}: mean ± s.e.m. over
units (rate minus the unit's mean in [−600, −400] ms).
**e**, Same areas: rate minus the pre-trial baseline, mean over units within each session, then mean ± s.e.m. over
sessions.
""")
    C.append(f"""
## Figure 5 | Decoders
**a–d**, Single-session decoders. L2 logistic regression (C = 0.05, balanced classes) trained on AH vs {RNP} with
stratified k-fold cross-validation (units z-scored on training folds; ≥ 5 good / mua units), applied unchanged to all
whisker-hit trials, which are never used for training. Chance is estimated by refitting and re-applying the decoder on 40
{NULLTXT}; every readout is reported as observed minus the null median (ratios: numerator and denominator corrected
separately).
**a**, Balanced accuracy above chance (held-out AH vs {RNP}). {_st('5a')}
**b**, Whisker hits decoded as AH (yes/no): [fraction of WH decoded as AH − false-positive rate] / [true-positive rate −
false-positive rate] (0: like {RNP}, 1: like auditory hits). {_st('5b')}
**c**, Same with decoder probabilities: [P(AH | WH) − P(AH | {RA})] / [P(AH | AH) − P(AH | {RA})]. {_st('5c')}
**d**, Numerator of c alone: how much more auditory-hit-like whisker hits are than {RNP}, in probability units. No
denominator (stable when the decoder is weak) and graded probabilities rather than thresholded decisions. {_st('5d')}
**e–l**, Pseudo-population decoders (hierarchical bootstrap per group: mice → sessions (experts) → M neurons pooled from
the sampled sessions → trials; 1,000 bootstraps; per iteration the trials of each sampled session are split into training
and test halves and 100 pseudo-trials per class drawn from each, with the same trial indices for all neurons of a session;
units z-scored on training pseudo-trials). Chance: the same mice, sessions and neurons with linearly shifted event labels
in every sampled session; readouts are observed minus chance. **e–h**, Equivalents of a–d at the largest population size
(violins: bootstrap distributions; dots and bars: mean and 95% bootstrap interval; lines join learning and expert).
Accuracy: {_stpp('bacc')} Yes/no transfer: {_stpp('transfer_bin')} Probability transfer: {_stpp('transfer')}
Probability numerator: {_stpp('num_prob')} **i–k**, Accuracy, probability transfer and probability numerator as a
function of the number of pooled neurons M (mean and 95% bootstrap interval). **l**, Learning → expert change at the
largest M per cohort and readout (95% bootstrap interval; filled: bootstrap p < 0.05); top: learning × cohort interaction
p (cohort labels permuted across mice, bootstrap re-run per permutation).
""")
    C.append(f"""
## COSYNE abstract figure (COSYNE_figure_v3)
**a**, First-lick-aligned PSTHs of all tested units (rate minus pre-trial baseline; mean over units within each session,
then mean ± s.e.m. over sessions) for whisker hits, auditory hits and {RNP}, per cohort and stage. Shaded: 100 ms
pre-lick window.
**b**, Converging neurons: fraction of reward-lick neurons (AH vs {RA} significant) that also separate whisker hits from
{RNP} with the same sign (mean ± s.e.m. over sessions). {_st('C4-b')}
**c**, Mean population geometry (triangle from the group-mean cross-validated distances; learning dashed, expert solid).
**d**, Distance difference Δd = d(WH, {RA}) − d(WH, AH) per session (cross-validated squared distances per unit).
{_st('C4-d')}
**e**, Single-session decoder trained on AH vs {RNP} and applied to whisker hits: P(AH | WH) − P(AH | {RA}), minus its
linear-shift chance level. {_st('C4-e')} Accuracy above chance on AH vs {RNP}: {_st('C4-e acc')}
**f**, Mean Δd per area group, cohort and stage for area groups with ≥ 3 sessions in all four groups; *: learning vs
expert Mann-Whitney p < 0.05; title: ANOVA Δd ~ cohort × stage × area (cohort labels permuted across mice).
{_anova('C4-f ANOVA')}

## COSYNE abstract figure (COSYNE_figure_v2)
**a**, First-lick-aligned PSTHs of all tested units (rate minus pre-trial baseline; mean over units within each session,
then mean ± s.e.m. over sessions) for whisker hits, auditory hits and {RNP}, per cohort and stage. Shaded: 100 ms
pre-lick window.
**b**, Mean population geometry (triangle from the group-mean cross-validated distances; learning dashed, expert solid).
Titles: mean Δd.
**c**, Δd = d(WH, {RA}) − d(WH, AH) per session, mean ± s.e.m. over sessions. {_st('C3-c')}
**d**, Fraction of reward-lick neurons that also separate whisker hits from {RNP} with the same sign; below: average
sign-aligned response of these neurons in R+ and R− experts (mean ± s.e.m. over sessions). {_st('C3-d')}
**e**, Fraction of all tested units separating whisker hits from {RNP}, with their average response below. {_st('C3-e')}
**f**, Mean Δd per area group, cohort and stage for area groups included in both cohorts; *: learning vs expert
Mann-Whitney p < 0.05; title: ANOVA (cohort labels permuted across mice).
""")
    txt = "\n".join(C)
    if m51.REF == "sl":
        txt = txt.replace("false alarms (FA)", "spontaneous licks (SL)")
    (out / f"captions_{pop}.md").write_text(txt, encoding="utf-8")


def main(a):
    plt = setup()
    out = BASE / "publication" / a.population
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    D = load_population(a.population)
    f = out / "lambda_axis_projections.csv"
    if f.exists():
        PR = pd.read_csv(f)
    else:
        sess = D["lam"][(D["lam"].level == "all") & (D["lam"].variant == "all")].dropna(subset=["lam"]).session_id.unique()
        PR = axis_projections(set(sess), ["all"], D["W"], rng); PR.to_csv(f, index=False)
    sel = set(a.figs.split(","))                        # --figs all | comma list of 1,2,2S,3,3S,4,5,cosyne
    run = lambda k: "all" in sel or k in sel
    if run("1"): fig1(plt, D, out, a.population, rng)
    if run("2"): fig2(plt, D, out, a.population, rng)
    if run("2S"): fig2_supplement(plt, D, out, a.population, rng)
    if run("3"): fig3(plt, D, out, a.population, rng, PR)
    if run("3S"): fig3s_lambda(plt, D, out, a.population, rng, PR)
    if run("4"): fig4(plt, D, out, a.population, rng, PR)
    if run("5"): fig5(plt, D, out, a.population, rng)
    if run("cosyne"):
        fig_cosyne2(plt, D, out, a.population, rng, PR)
        fig_cosyne3(plt, D, out, a.population, rng, PR)
    if "all" in sel:                                    # captions and stats only for complete runs
        write_captions(D, out, a.population)
        pd.DataFrame(STATS).to_csv(out / f"stats_{a.population}.csv", index=False)
    else:
        pd.DataFrame(STATS).to_csv(out / f"stats_{a.population}_partial_{'_'.join(sorted(sel))}.csv", index=False)
    json.dump(dict(script="062_pub_convergence_figures.py", population=a.population, unit_set=UNIT_SET,
                   min_fr_hz=m51.MIN_FR, colors=CL, cohort_colors=COH, n_perm_interaction=N_PERM,
                   tests="MWU shown (Welch in stats csv); interaction: mouse-level cohort permutation; area family-wise "
                         "max-|z| from 057 / 061", inputs=["051", "053", "056", "057", "059", "060", "061"]),
              open(out / "provenance.json", "w"), indent=1)
    print("ALL DONE", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="all", choices=["all", "learners"])
    ap.add_argument("--figs", default="all", help="all, or a comma list of 1,2,2S,3,3S,4,5,cosyne")
    main(ap.parse_args())
