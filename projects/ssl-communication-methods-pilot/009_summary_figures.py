"""Summary figures for the full-dataset communication study (user request 2026-11-09: "Make figures that summarize this
analysis, from dataset size, selection, processing, processes...").

Reads only saved outputs (no recomputation of any method):
  006 coverage/session_coverage.csv            day-0 sessions, unit counts (unit table v3), trial cells, jaw onsets
  007 hypothesis_v2/sessions/*_meta.json        per analysed session: units per area set (007 rate filter), trials per
                                                condition, pairs and conditions actually run
  008 hypothesis_v2/session_excess_{tag}.parquet, tests_{tag}.csv, positive_control_{tag}.csv (tag learners / all_mice)
  005 benchmark/benchmark_summary.csv, benchmark_raw.parquet
Every threshold and parameter printed on the figures is read from the code (007 CONFIG2, comm_lib CONFIG, 008 module
constants, 005 module constants), not retyped.
Figures (OUT/summary_figures/, each .png + .pdf, and all of them in summary_figures.pdf):
  01_dataset            sessions / mice per cohort and learner status, units per area set and per pair, trials per
                        condition, jaw-onset latencies
  02_selection_funnel   from day-0 sessions to the sessions entering each prediction; sessions per pair x condition
  03_pipeline           processing schematic with the parameter values
  04_benchmark          005 detection power and direction calls per method and scenario; power vs trials
  05_results_forest     R+ minus R- per prediction x pair x window (lagRRR_dir excess), learners and all mice
  06_positive_control   VPM-PO -> SSp-w direction index per cohort, condition and window
Also: funnel_counts.csv, forest_effects.csv (same folder; provenance of the plotted numbers).
Run (haas): python projects/ssl-communication-methods-pilot/009_summary_figures.py
"""

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comm_lib as cl  # noqa: E402


def _module(fname, name):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cov = _module("006_coverage_table.py", "cov006")
hyp = _module("007_run_hypothesis.py", "hyp007")
st = _module("008_hypothesis_stats.py", "stats008")
bm = _module("005_method_benchmark.py", "bench005")

OUT = cl.OUT / "summary_figures"
H2 = cl.OUT / "hypothesis_v2"
C2 = hyp.CONFIG2
COL = st.COL  # R+ #00B400, R- #C800C8
COHORTS = ["R+", "R-"]
TAGS = {"learners": "learners (primary)", "all_mice": "all mice (sensitivity)"}
PAIRS = [tuple(p) for p in C2["pairs"]]
PAIR_LAB = {p: f"{p[0]} → {p[1]}" for p in PAIRS}
CONDS = C2["conditions"]
PAIR_AREAS = list(dict.fromkeys(a for p in PAIRS for a in p))
PRIMARY = st.PRIMARY
ROBUST = "lagRRR|rest_dir"

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- helpers

def area_colors():
    """Area set -> colour: the area group's colour from ephys_utilities allen_utils get_custom_area_groups_colors()
    (group of the set's first acronym); sets sharing a group get darker / lighter shades. Grey if unavailable."""
    try:
        from ephys_utilities.allen_utils import allen_utils as au
        pal, grp = au.get_custom_area_groups_colors(), au.get_custom_area_groups_from_name()
    except Exception as err:  # palette not reachable: neutral grey
        print("area palette unavailable:", err)
        return {a: "0.5" for a in cov.AREA_SETS}
    by_group, out = {}, {}
    for a, acr in cov.AREA_SETS.items():
        by_group.setdefault(grp.get(acr[0]), []).append(a)
    shade = [0.0, -0.35, 0.35, -0.6]  # <0 towards black, >0 towards white
    for g, sets in by_group.items():
        base = np.array(to_rgb(pal.get(g, "0.5")))
        for k, a in enumerate(sets):
            f = shade[k % len(shade)]
            out[a] = tuple(base * (1 + f) if f < 0 else base + (1 - base) * f)
    return out


ACOL = area_colors()


def save(fig, name, pdf):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    pdf.savefig(fig)
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def jitter(n, rng, w=0.07):
    return rng.uniform(-w, w, n)


def welch_ci(a, b, level=0.95):
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    h = stats.t.ppf(0.5 + level / 2, df) * se
    d = a.mean() - b.mean()
    return d - h, d + h


def fmt_p(p):
    if not np.isfinite(p):
        return "–"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.0e}"


# ---------------------------------------------------------------- data

def load_all():
    covdf = pd.read_csv(cl.OUT / "coverage" / "session_coverage.csv")
    metas = [json.load(open(f)) for f in sorted((H2 / "sessions").glob("*_meta.json")) if "_quick" not in f.name]
    rows = []
    for m in metas:
        r = dict(session_id=m["session_id"], mouse_id=m["mouse_id"], cohort=m["cohort"], learner=bool(m["learner"]),
                 pairs=[tuple(p) for p in m["pairs"]], conditions=list(m["conditions"]))
        r.update({f"u_{k}": v for k, v in m["n_units"].items()})
        r.update({f"t_{k}": v for k, v in m["n_trials"].items()})
        rows.append(r)
    meta = pd.DataFrame(rows).merge(covdf[["session_id", "learning_category", "median_jaw_onset_W_lick_ms",
                                           "median_jaw_onset_A_hit_ms"]], on="session_id", how="left")
    sx, wm, tests, pc = {}, {}, {}, {}
    for tag in TAGS:
        sx[tag] = pd.read_parquet(H2 / f"session_excess_{tag}.parquet")
        wm[tag] = st.window_means(sx[tag])
        tests[tag] = pd.read_csv(H2 / f"tests_{tag}.csv")
        pc[tag] = pd.read_csv(H2 / f"positive_control_{tag}.csv")
    bsum = pd.read_csv(cl.OUT / "benchmark" / "benchmark_summary.csv")
    braw = pd.read_parquet(cl.OUT / "benchmark" / "benchmark_raw.parquet")
    return covdf, meta, sx, wm, tests, pc, bsum, braw


# ---------------------------------------------------------------- 1. dataset

def fig_dataset(covdf, meta, pdf):
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    # a: sessions per cohort by learning category (analysed sessions, 007)
    ax = axs[0, 0]
    cats = [("good", 1.0), ("moderate", 0.6), ("bad", 0.25), ("n/a", 0.08)]
    lc = meta.learning_category.fillna("n/a")
    for i, coh in enumerate(COHORTS):
        bottom = 0
        for cat, alpha in cats:
            n = int(((meta.cohort == coh) & (lc == cat)).sum())
            if n:
                ax.bar(i, n, 0.6, bottom=bottom, color=COL[coh], alpha=alpha, ec=COL[coh], lw=0.6)
                ax.text(i + 0.33, bottom + n / 2, f"{cat}: {n}", va="center", fontsize=6.5)
            bottom += n
        g = meta[meta.cohort == coh]
        ax.text(i, bottom + 0.8, f"{len(g)} sessions\n{g.mouse_id.nunique()} mice\n{int(g.learner.sum())} learners",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([0, 1], COHORTS)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(0, meta.groupby("cohort").size().max() * 1.35)
    ax.set_ylabel("sessions (whisker day 0, one per mouse)")
    ax.set_title(f"a  Sessions analysed (007), n = {len(meta)}\nlearner = sheet learning_category good / moderate",
                 loc="left")
    # b: units per area set per session
    ax = axs[0, 1]
    for i, a in enumerate(PAIR_AREAS):
        for k, coh in enumerate(COHORTS):
            v = meta.loc[meta.cohort == coh, f"u_{a}"].to_numpy()
            v = v[v > 0]
            x = i + (k - 0.5) * 0.36
            ax.scatter(x + jitter(len(v), rng), v, s=5, color=COL[coh], alpha=0.7, lw=0)
            if len(v):
                ax.hlines(np.median(v), x - 0.13, x + 0.13, color="k", lw=1)
            ok = int((meta.loc[meta.cohort == coh, f"u_{a}"] >= C2["n_units"]).sum())
            ax.text(x, 1.0, f"{ok}", ha="center", va="bottom", fontsize=6, color=COL[coh],
                    transform=ax.get_xaxis_transform())
    ax.axhline(C2["n_units"], color="0.4", lw=0.6, ls="--")
    ax.text(len(PAIR_AREAS) - 0.5, C2["n_units"], f" {C2['n_units']} units drawn", va="bottom", ha="right",
            fontsize=6, color="0.3")
    tot = {a: int(meta[f"u_{a}"].sum()) for a in PAIR_AREAS}
    ax.set_xticks(range(len(PAIR_AREAS)), [f"{a}\n({tot[a]} units)" for a in PAIR_AREAS])
    for t, a in zip(ax.get_xticklabels(), PAIR_AREAS):
        t.set_color(ACOL[a])
        t.set_fontweight("bold")
    ax.set_ylabel(f"good units ≥ {C2['min_rate_hz']:g} Hz per session")
    ymax = max(meta[f"u_{a}"].max() for a in PAIR_AREAS)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_title("b  Units per area set (sessions with > 0 units; bar = median; total units under labels)\n"
                 f"numbers on top: sessions with ≥ {C2['n_units']} units (R+ / R-)", loc="left", pad=12)
    # c: sessions per pair (pairs actually run by 007) per cohort, learners solid
    ax = axs[0, 2]
    for i, p in enumerate(PAIRS):
        for k, coh in enumerate(COHORTS):
            g = meta[(meta.cohort == coh) & meta.pairs.apply(lambda ps: p in ps)]
            nl, na = int(g.learner.sum()), len(g)
            y = i + (k - 0.5) * 0.38
            ax.barh(y, nl, 0.36, color=COL[coh], lw=0)
            ax.barh(y, na - nl, 0.36, left=nl, color=COL[coh], alpha=0.3, lw=0)
            ax.text(na + 0.3, y, f"{nl} ({na})", va="center", fontsize=6)
    ax.set_yticks(range(len(PAIRS)), [PAIR_LAB[p] for p in PAIRS])
    ax.invert_yaxis()
    ax.axvline(st.MIN_SESSIONS, color="0.4", lw=0.6, ls="--")
    ax.set_xlabel("sessions with ≥ 20 units in both areas")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    ax.set_title(f"c  Sessions per pair: learners (solid) and all mice (light; in brackets)\n"
                 f"dashed: {st.MIN_SESSIONS} sessions per cohort needed for a test", loc="left")
    # d: units per pair: smaller of the two areas
    ax = axs[1, 0]
    for i, p in enumerate(PAIRS):
        for k, coh in enumerate(COHORTS):
            g = meta[(meta.cohort == coh) & meta.pairs.apply(lambda ps: p in ps)]
            v = np.minimum(g[f"u_{p[0]}"], g[f"u_{p[1]}"]).to_numpy()
            x = i + (k - 0.5) * 0.36
            ax.scatter(x + jitter(len(v), rng), v, s=5, color=COL[coh], alpha=0.7, lw=0)
            if len(v):
                ax.hlines(np.median(v), x - 0.13, x + 0.13, color="k", lw=1)
    ax.axhline(C2["n_units"], color="0.4", lw=0.6, ls="--")
    ax.set_xticks(range(len(PAIRS)), [PAIR_LAB[p] for p in PAIRS], rotation=40, ha="right")
    ax.set_ylabel("units in the smaller area of the pair")
    ax.set_ylim(0, None)
    ax.set_title(f"d  Units per pair and session (min of the two areas; {C2['n_units']} drawn per area and draw)",
                 loc="left")
    # e: trials per condition
    ax = axs[1, 1]
    for i, c in enumerate(CONDS):
        for k, coh in enumerate(COHORTS):
            v = meta.loc[meta.cohort == coh, f"t_{c}"].to_numpy()
            x = i + (k - 0.5) * 0.36
            ax.scatter(x + jitter(len(v), rng), v, s=5, color=COL[coh], alpha=0.7, lw=0)
            ax.hlines(np.median(v), x - 0.13, x + 0.13, color="k", lw=1)
            ax.text(x, 1.0, f"{int((v >= C2['n_trials']).sum())}", ha="center", va="bottom", fontsize=6,
                    color=COL[coh], transform=ax.get_xaxis_transform())
    ax.axhline(C2["n_trials"], color="0.4", lw=0.6, ls="--")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, meta[[f"t_{c}" for c in CONDS]].max().max() * 1.6)
    ax.set_xticks(range(len(CONDS)), CONDS, rotation=30, ha="right")
    ax.set_ylabel("trials per session (active, perf != 6, no early lick, A1 trim)")
    ax.set_title(f"e  Trials per condition; dashed: {C2['n_trials']} trials drawn per draw\n"
                 "numbers on top: sessions with enough trials (R+ / R-)", loc="left", pad=12)
    # f: jaw-onset latency (behaviour_analysis) vs the analysis windows
    ax = axs[1, 2]
    for i, (col, lab) in enumerate([("median_jaw_onset_W_lick_ms", "W_lick"), ("median_jaw_onset_A_hit_ms", "A_hit")]):
        for k, coh in enumerate(COHORTS):
            v = meta.loc[meta.cohort == coh, col].dropna().to_numpy()
            x = i + (k - 0.5) * 0.36
            ax.scatter(x + jitter(len(v), rng), v, s=6, color=COL[coh], alpha=0.7, lw=0)
            ax.hlines(np.median(v), x - 0.13, x + 0.13, color="k", lw=1)
            ax.text(x, 1.0, f"n={len(v)}", ha="center", va="bottom", fontsize=6, color=COL[coh],
                    transform=ax.get_xaxis_transform())
    pm = C2["premovement_conditions"]["W_lick_pm150"] * 1000
    early_end = (st.WINDOWS["early"][1] + C2["window_bins"] * C2["bin_s"] / 2) * 1000
    ax.axhline(pm, color="0.4", lw=0.6, ls="--")
    ax.text(1.55, pm, f"{pm:.0f} ms: pm150 cut", va="bottom", ha="right", fontsize=6, color="0.3")
    ax.axhline(early_end, color="0.4", lw=0.6, ls=":")
    ax.text(1.55, early_end, f"{early_end:.0f} ms: end of last 'early' window", va="bottom", ha="right", fontsize=6,
            color="0.3")
    ax.set_xticks([0, 1], ["whisker lick", "auditory hit"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("median jaw onset per session (ms from stimulus)")
    ax.set_title("f  Movement onset (DLC jaw) vs analysis windows", loc="left", pad=12)
    for coh in COHORTS:
        axs[0, 0].plot([], [], "s", color=COL[coh], label=coh)
    axs[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Dataset: whisker day-0 sessions entering the hypothesis-v2 run (007); units and trials as counted "
                 "by 007 (meta json)", fontsize=9)
    fig.tight_layout(h_pad=2.5)
    save(fig, "01_dataset", pdf)


# ---------------------------------------------------------------- 2. funnel

def n_by_cohort(sids, meta_or_cov):
    d = meta_or_cov[meta_or_cov.session_id.isin(sids)]
    return {c: int((d.cohort == c).sum()) for c in COHORTS}


def funnel_counts(covdf, meta, sx):
    u = C2["n_units"]
    steps = []
    m = pd.Series(True, index=covdf.index)
    steps.append(("all", "Day-0 sessions with ≥ 1 good unit ≥ 1 Hz\n(unit table v3, 006)", m.copy()))
    m &= covdf.has_jaw_table
    steps.append(("all", "+ jaw-onset table\n(behaviour_analysis dlc_jaw_onset_times.pkl)", m.copy()))
    m &= covdf.exclude_ephys != 1
    steps.append(("all", "+ not exclude_ephys (sheet)", m.copy()))
    m &= covdf.cohort.isin(COHORTS)
    steps.append(("all", "+ cohort R+ / R- (sheet)", m.copy()))
    ok = np.zeros(len(covdf), bool)
    for a, b in PAIRS:
        ok |= (covdf[f"units_{a}"] >= u).to_numpy() & (covdf[f"units_{b}"] >= u).to_numpy()
    m &= ok
    steps.append(("all", f"+ ≥ 1 pre-registered pair with ≥ {u} units\nin both areas (006 counts) = 007 session_list",
                  m.copy()))
    rows = []
    for pop, lab, mk in steps:
        rows.append(dict(population=pop, step=lab, **n_by_cohort(covdf.session_id[mk], covdf)))
    sl = set(hyp.session_list().session_id)
    assert sl == set(covdf.session_id[m]), "funnel step differs from 007 session_list()"
    ran = meta[meta.pairs.apply(len) > 0]
    rows.append(dict(population="all", step=f"+ ≥ 1 pair with ≥ {u} units after the\n007 rate filter "
                     f"(≥ {C2['min_rate_hz']:g} Hz over analysed trials)", **n_by_cohort(ran.session_id, meta)))
    rows.append(dict(population="learners", step="+ learner (good / moderate) [primary set]",
                     **n_by_cohort(ran.session_id[ran.learner], meta)))
    # per prediction: sessions with data (available) and in a pair testable in both cohorts (>= MIN_SESSIONS)
    preds = [("P1", "W_lick", st.P1_PAIRS), ("P2", "W_nolick", st.P2_PAIRS), ("P3", ("W_lick", "A_hit"), st.P1_PAIRS),
             ("PM", "W_lick_pm150", st.P1_PAIRS), ("PC", None, [st.PC_PAIR])]
    for tag in TAGS:
        s = sx[tag][sx[tag].quantity == PRIMARY]
        for pred, cond, pairs in preds:
            avail, test = set(), set()
            for a, b in pairs:
                x = s[(s.area_a == a) & (s.area_b == b)]
                if cond is None:
                    have = x.groupby("session_id").cohort.first()
                elif isinstance(cond, tuple):
                    sets = [set(x[x.condition == c].session_id) for c in cond]
                    have = x[x.session_id.isin(set.intersection(*sets))].groupby("session_id").cohort.first()
                else:
                    have = x[x.condition == cond].groupby("session_id").cohort.first()
                avail |= set(have.index)
                cnt = have.value_counts()
                if pred == "PC" or all(cnt.get(c, 0) >= st.MIN_SESSIONS for c in COHORTS):
                    test |= set(have.index)
            if pred == "PC":
                lab = f"PC: VPM-PO & SSp-w ≥ {u} units (any condition)"
            elif isinstance(cond, tuple):
                lab = f"{pred}: {' & '.join(cond)} ≥ {C2['n_trials']} trials, {pairs[0][0]} → {'/'.join(b for _, b in pairs)}"
            else:
                lab = f"{pred}: {cond} ≥ {C2['n_trials']} trials, {pairs[0][0]} → {'/'.join(b for _, b in pairs)}"
            rows.append(dict(population=tag, step=lab, kind="available", **n_by_cohort(avail, meta)))
            rows.append(dict(population=tag, step=lab, kind="testable", **n_by_cohort(test, meta)))
    return pd.DataFrame(rows)


def fig_funnel(covdf, meta, sx, pdf):
    fc = funnel_counts(covdf, meta, sx)
    fc.to_csv(OUT / "funnel_counts.csv", index=False)
    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=0.45, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    core = fc[fc.kind.isna()]
    pred_l = fc[(fc.population == "learners") & fc.kind.notna()]
    pred_a = fc[(fc.population == "all_mice") & fc.kind.notna()]
    labels, y = [], 0
    ys = []
    for _, r in core.iterrows():
        for k, coh in enumerate(COHORTS):
            yy = y + (k - 0.5) * 0.38
            ax.barh(yy, r[coh], 0.36, color=COL[coh], lw=0)
            ax.text(r[coh] + 0.6, yy, str(r[coh]), va="center", fontsize=6.5)
        labels.append(r["step"])
        ys.append(y)
        y += 1
    y += 0.4
    ax.axhline(y - 0.7, color="0.6", lw=0.5)
    for (lab, g) in pred_l.groupby("step", sort=False):
        ga = pred_a[pred_a.step == lab]
        av, te = g[g.kind == "available"].iloc[0], g[g.kind == "testable"].iloc[0]
        tea = ga[ga.kind == "testable"].iloc[0]
        for k, coh in enumerate(COHORTS):
            yy = y + (k - 0.5) * 0.38
            ax.barh(yy, av[coh], 0.36, color="none", ec=COL[coh], lw=0.7)
            ax.barh(yy, te[coh], 0.36, color=COL[coh], lw=0)
            ax.text(av[coh] + 0.6, yy, f"{te[coh]} / {av[coh]}  (all mice {tea[coh]})", va="center", fontsize=6.5)
        labels.append(lab)
        ys.append(y)
        y += 1
    ax.set_yticks(ys, labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, core[COHORTS].max().max() * 1.25)
    ax.set_xlabel("sessions (= mice)")
    for coh in COHORTS:
        pass
    ax.legend(handles=[Patch(color=COL[c], label=c) for c in COHORTS], frameon=False, loc="lower right")
    ax.text(0.99, 0.30, "below the line, learners:\nsolid = in a pair testable in both cohorts\n"
            f"(≥ {st.MIN_SESSIONS} sessions per cohort; PC: within cohort)\noutline = sessions with data\n"
            "text: testable / with data (all-mice testable)", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.3, color="0.25")
    ax.set_title("a  Selection funnel per cohort (thresholds from 006 / 007 / 008 code)", loc="left")
    # b, c: sessions per pair x condition
    for row, tag in enumerate(TAGS):
        ax = fig.add_subplot(gs[row, 1])
        s = sx[tag][sx[tag].quantity == PRIMARY]
        cnt = s.groupby(["area_a", "area_b", "condition", "cohort"]).session_id.nunique()
        M = np.zeros((len(PAIRS), len(CONDS)))
        for i, (a, b) in enumerate(PAIRS):
            for j, c in enumerate(CONDS):
                n = [int(cnt.get((a, b, c, coh), 0)) for coh in COHORTS]
                M[i, j] = min(n)
                ok = min(n) >= st.MIN_SESSIONS
                ax.text(j, i, f"{n[0]} / {n[1]}", ha="center", va="center", fontsize=6.3,
                        color="k" if ok else "0.55", fontweight="bold" if ok else "normal")
                if not ok:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="0.93", ec="none", zorder=0))
        Mm = np.ma.masked_less(M, st.MIN_SESSIONS)
        ax.imshow(Mm, cmap="Blues", vmin=0, vmax=max(M.max() * 1.6, 1), aspect="auto", zorder=1, alpha=0.9)
        ax.set_xticks(range(len(CONDS)), CONDS, rotation=30, ha="right")
        ax.set_yticks(range(len(PAIRS)), [PAIR_LAB[p] for p in PAIRS])
        ax.set_xlim(-0.5, len(CONDS) - 0.5)
        ax.set_ylim(len(PAIRS) - 0.5, -0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        nses = s.groupby("cohort").session_id.nunique()
        ax.set_title(f"{'bc'[row]}  Sessions per pair × condition, R+ / R-: {TAGS[tag]} "
                     f"(R+ {nses.get('R+', 0)}, R- {nses.get('R-', 0)})\n"
                     f"bold / blue = testable (≥ {st.MIN_SESSIONS} per cohort); grey = not tested", loc="left")
    fig.suptitle(f"Selection: units ≥ {C2['n_units']} per area, ≥ {C2['min_rate_hz']:g} Hz; trials ≥ "
                 f"{C2['n_trials']} per condition; ≥ {st.MIN_SESSIONS} sessions per cohort per pair", fontsize=9)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.08)
    save(fig, "02_selection_funnel", pdf)
    return fc


# ---------------------------------------------------------------- 3. pipeline

def fig_pipeline(pdf):
    C, C1 = C2, cl.CONFIG
    centers, widx = hyp.windows()
    bin_ms = C["bin_s"] * 1000
    L = C["lags_bins"]
    wins = ", ".join(f"{k} {lo:g}..{hi:g}" for k, (lo, hi) in st.WINDOWS.items())
    areas = "; ".join(f"{a} = {'+'.join(v)}" for a, v in C["area_sets"].items() if a in PAIR_AREAS)
    pm = ", ".join(f"{k} (jaw ≥ {v * 1000:.0f} ms)" for k, v in C["premovement_conditions"].items())
    rows = [
        ("Data", [
            ("Sessions", "whisker day 0, one per mouse; jaw-onset table; not exclude_ephys; cohort R+ / R- and "
                         "learner (learning_category good / moderate) from the reference sheet; primary set = learners, "
                         "sensitivity = all mice"),
            ("Units", f"good units (ssl-cross-area-communication quality thresholds), rate ≥ {C['min_rate_hz']:g} Hz "
                      f"over the analysed trials; area sets: {areas}"),
            ("Trials", C["trials"]),
            ("Conditions", f"{', '.join(C['conditions'][:4])}; pre-movement subsets {pm}, windows ending ≤ "
                           f"{max(C['premovement_conditions'].values()):g} s only; a condition is used with ≥ "
                           f"{C['n_trials']} trials"),
        ]),
        ("Processing", [
            ("Binning", f"{bin_ms:.0f}-ms spike-count bins, {C['edges_s'][0]:g} to {C['edges_s'][1]:g} s around "
                        f"trial start ({len(hyp.edges()) - 1} bins)"),
            ("Draws", f"{C['n_draws']} draws per session (seed crc32(session) + draw); per draw and condition "
                      f"{C['n_trials']} trials, per pair {C['n_units']} units per area, random without replacement"),
            ("Residuals", "per draw: condition mean subtracted per unit and bin (trial-to-trial co-fluctuations); "
                          "gain-corrected residuals benchmarked and dropped"),
            ("Windows", f"{C['window_bins']} bins = {C['window_bins'] * bin_ms:.0f} ms, centres {C['centers_s'][0]:g} "
                        f"to {C['centers_s'][1]:g} s every {C['step_s'] * 1000:.0f} ms ({len(centers)} windows); "
                        f"source lags 1..{L} bins ({L * bin_ms:.0f} ms)"),
        ]),
        ("Methods\n(per window)", [
            ("RRR", f"same-bin reduced-rank ridge A→B; CV {C1['cv_folds']} folds by trial, ridge α ∈ "
                    f"{C1['ridge_alphas']} × trace, rank ≤ {C1['max_rank']} (1-s.e.m. rule); score 1 − NSE"),
            ("lagged RRR", f"A(t−1..t−{L}) → B(t) and B → A; direction index lagRRR_dir = A→B − B→A "
                           "[PRIMARY]"),
            ("lagRRR | rest", f"target's own past (lags 1..{L}) and rest PCs (lags 0..{L}) ridge-regressed out "
                              "(λ = 0.01 × trace), then lagged RRR; same index [required robustness]"),
            ("Granger / TRGC", f"{C['granger_pcs']} PCs per area, order {C['granger_order']}, {C['granger_bins']} "
                               f"bins ({C['granger_bins'] * bin_ms:.0f} ms) in the window; net TRGC = (A→B − B→A) "
                               "− time-reversed"),
            ("Granger | rest", f"conditional Granger, rest's {C['rest_pcs']} PCs in both models; net TRGC"),
            ("CCA", f"PCA ({C1['cca_pcs']} PCs per area, training folds) → regularised CCA ({C1['cca_reg']:g}); "
                    "held-out cc1"),
            ("pCCA | rest", f"partial_CCA 0.1.0 with Z = rest PCs (lags 0..{L}); CV and held-out Z removal in "
                            "comm_lib.pcca_cv; cc1"),
        ]),
        ("Null and\naggregation", [
            ("Shuffle null", f"{C['n_shuffles']} permutations of A's trials (same for all methods; keeps PSTHs, "
                             "breaks trial-matched coupling)"),
            ("Excess", "value − null mean, per draw, condition, pair, window and quantity"),
            ("Session value", f"excess averaged over the {C['n_draws']} draws: one value per session (unit of "
                              "analysis = session = mouse)"),
            ("Window means", f"mean over window centres (s): {wins}; PM: early only"),
        ]),
        ("Statistics", [
            ("Predictions", "P1 W_lick, SSp-w → wMO / ALM-tjM1 / Striatum, R+ > R-; P2 W_nolick, Frontal → "
                            "SSp-w / Striatum / wMO / ALM-tjM1, R- > R+; P3 (W_lick − A_hit) for P1 pairs; P3b A_hit "
                            "alone; PM W_lick_pm150 / A_hit_pm150"),
            ("R+ vs R-", f"Mann-Whitney and Welch, both reported; a pair is tested only with ≥ {st.MIN_SESSIONS} "
                         "sessions in each cohort"),
            ("FDR", "Benjamini-Hochberg across pairs, within prediction × condition × quantity × window"),
            ("Positive control", f"{st.PC_PAIR[0]} → {st.PC_PAIR[1]} direction index vs 0 per cohort "
                                 "(Wilcoxon signed-rank), every condition and window"),
        ]),
    ]
    W, Hf = 17, 9.5
    fig = plt.figure(figsize=(W, Hf))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    x0, x1, gap = 0.085, 0.99, 0.008
    tops = [0.945, 0.755, 0.565, 0.305, 0.145]
    heights = [0.155, 0.155, 0.225, 0.125, 0.13]
    fills = ["#eef3f8", "#f3f3f3", "#fff6e5", "#f0f7ee", "#f7eef6"]
    fs = 7.6
    centers_xy = []
    for (rname, boxes), top, h, fcol in zip(rows, tops, heights, fills):
        n = len(boxes)
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        bw = (x1 - x0 - gap * (ncol - 1)) / ncol
        bh = (h - gap * (nrow - 1)) / nrow
        ax.text(0.01, top - h / 2, rname, fontsize=9, fontweight="bold", va="center", ha="left")
        row_xy = []
        for k, (title, body) in enumerate(boxes):
            r, c = divmod(k, ncol)
            bx, by = x0 + c * (bw + gap), top - (r + 1) * bh - r * gap
            ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.006", fc=fcol,
                                        ec="0.45", lw=0.6))
            ax.text(bx + 0.005, by + bh - 0.008, title, fontsize=8.5, fontweight="bold", va="top")
            wrap = int(bw * W * 72 / (fs * 0.62))
            ax.text(bx + 0.005, by + bh - 0.036, textwrap.fill(body, wrap), fontsize=fs, va="top", linespacing=1.25)
            row_xy.append((bx, by, bw, bh))
            if c > 0:
                ax.annotate("", xy=(bx, by + bh / 2), xytext=(bx - gap, by + bh / 2),
                            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35", mutation_scale=7))
        centers_xy.append((top, h))
    for (t0, h0), (t1, _) in zip(centers_xy[:-1], centers_xy[1:]):
        ax.annotate("", xy=(0.5, t1 + 0.002), xytext=(0.5, t0 - h0 - 0.002),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.3", mutation_scale=9))
    ax.text(0.5, 0.995, "Processing pipeline of the hypothesis-v2 run (007) and statistics (008); all values read from "
            "007 CONFIG2, comm_lib CONFIG and 008", fontsize=10, ha="center", va="top", fontweight="bold")
    ax.text(0.085, 0.004, f"'rest' = top {C['rest_pcs']} PCs of every other good ≥ {C['min_rate_hz']:g} Hz unit of the "
            f"session (not in A or B; needs ≥ {C['rest_pcs']} units). Directed quantities: positive = A leads B. "
            "Benchmark 005 chose lagRRR_dir as primary (best power and direction) and lagRRR | rest as the robustness "
            "check.", fontsize=6.5, va="bottom", color="0.25")
    save(fig, "03_pipeline", pdf)


# ---------------------------------------------------------------- 4. benchmark

def fig_benchmark(bsum, braw, pdf):
    BIN = 20  # 007 uses 20-ms bins
    s = bsum[(bsum.bin_ms == BIN) & (bsum.prep == "standard")]
    meths = ["RRR", "lagRRR", "lagRRR|C", "CCA", "pCCA|C", "lagCCA", "Granger", "Granger|C", "pCCA|oracle",
             "Granger|oracle"]
    used = {"RRR": "RRR", "lagRRR": "lagRRR (primary)", "lagRRR|C": "lagRRR|C (= |rest)", "CCA": "CCA",
            "pCCA|C": "pCCA|C (= |rest)", "lagCCA": "lagCCA (not in 007)", "Granger": "Granger",
            "Granger|C": "Granger|C (= |rest)", "pCCA|oracle": "pCCA|true driver*", "Granger|oracle":
            "Granger|true driver*"}
    scen = bm.SCENARIOS
    truth = {"none": "none", "ff": "A→B", "fb": "B→A", "bidir": "both", "common_rec": "none (C rec.)",
             "common_hidden": "none (H hidden)", "gain": "none (gain)"}
    cmax = max(bm.COUPLINGS)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.55)
    # a: detection at the strongest coupling
    ax = fig.add_subplot(gs[0, :2])
    M = np.full((len(meths), len(scen)), np.nan)
    for i, m in enumerate(meths):
        for j, sc in enumerate(scen):
            x = s[(s.method == m) & (s.scenario == sc) & (s.coupling == (0.0 if sc == "none" else cmax))]
            if len(x):
                M[i, j] = x.detect.iloc[0]
    ax.imshow(M, cmap="Greys", vmin=0, vmax=2.4, aspect="auto")
    for i in range(len(meths)):
        for j, sc in enumerate(scen):
            good = (sc in ("ff", "fb", "bidir")) == (M[i, j] >= 0.5) if sc in ("ff", "fb", "bidir") else M[i, j] <= 0.1
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="k" if good else "#c0392b", fontweight="normal" if good else "bold")
    ax.set_xticks(range(len(scen)), [f"{sc}\ntruth: {truth[sc]}" for sc in scen], fontsize=6.3)
    ax.set_yticks(range(len(meths)), [used[m] for m in meths])
    ax.axhline(7.5, color="0.3", lw=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"a  Detection rate of an A-B interaction (p < {bm.SIG} vs {bm.N_SH} shuffles; coupling {cmax:g}, "
                 "none: 0)\nred = wrong call (true interaction missed in ≥ 50%, or false detection > 10%)", loc="left")
    # b: direction calls at the strongest coupling
    ax = fig.add_subplot(gs[0, 2:])
    dm = [m for m in meths if m in bm.DIRECTION]
    cols = [(sc, d) for sc in scen for d in ("A→B", "B→A")]
    D = np.full((len(dm), len(cols)), np.nan)
    for i, m in enumerate(dm):
        for j, (sc, d) in enumerate(cols):
            x = s[(s.method == m) & (s.scenario == sc) & (s.coupling == (0.0 if sc == "none" else cmax))]
            if len(x):
                D[i, j] = x["lead_AB" if d == "A→B" else "lead_BA"].iloc[0]
    ax.imshow(D, cmap="Greys", vmin=0, vmax=2.4, aspect="auto")
    true_cells = {("ff", "A→B"), ("fb", "B→A"), ("bidir", "A→B"), ("bidir", "B→A")}
    for i in range(len(dm)):
        for j, (sc, d) in enumerate(cols):
            ok = D[i, j] >= 0.5 if (sc, d) in true_cells else D[i, j] <= 0.1
            ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="k" if ok else "#c0392b", fontweight="normal" if ok else "bold")
            if (sc, d) in true_cells:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#00468b", lw=1.0))
    ax.set_xticks(range(len(cols)), [d for _, d in cols], fontsize=6)
    for k, sc in enumerate(scen):
        ax.text(2 * k + 0.5, -0.9, sc, ha="center", va="bottom", fontsize=6.5)
        if k:
            ax.axvline(2 * k - 0.5, color="0.3", lw=0.6)
    ax.set_ylim(len(dm) - 0.5, -1.4)
    ax.set_yticks(range(len(dm)), [used[m] for m in dm])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("b  Direction: fraction of sims with a significant lead in each direction\n"
                 "blue box = true direction; red = wrong call (true lead < 50%, spurious lead > 10%)", loc="left",
                 pad=14)
    # c: detection vs coupling (ff and common_hidden)
    show = ["RRR", "lagRRR", "lagRRR|C", "CCA", "pCCA|C", "Granger", "Granger|C"]
    none = s[s.scenario == "none"].set_index("method")
    for k, (sc, ttl) in enumerate([("ff", "c  power: ff (A→B)"), ("common_hidden", "d  false detection: hidden "
                                                                                   "common input")]):
        ax = fig.add_subplot(gs[1, k])
        for m in show:
            x = s[(s.scenario == sc) & (s.method == m)].sort_values("coupling")
            ax.plot([0] + list(x.coupling), [none.loc[m, "detect"]] + list(x.detect), "-o", ms=2.5, lw=0.9,
                    color=bm.COLORS[m], label=used[m])
        ax.axhline(bm.SIG, color="0.5", lw=0.6, ls=":")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.05, cmax + 0.05)
        ax.set_xlabel("coupling")
        ax.set_ylabel("detection rate")
        ax.set_title(f"{ttl}\n30 trials, 20 units, {BIN}-ms bins, n = {int(s.n.iloc[0])} sims/point", loc="left")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    # e, f: power / direction / false positives vs trials (sweep, 20 units)
    sw = braw[(braw.prep == "standard") & (braw.bin_ms == BIN) & (braw.n_units == 20) &
              braw.coupling.isin([0.0, 0.5]) & braw.scenario.isin(["ff", "none"])]
    sw = sw[(sw.grid == "sweep") | (sw.n_trials == 30)]
    ax = fig.add_subplot(gs[1, 2])
    ax2 = fig.add_subplot(gs[1, 3])
    for m in show:
        for sc, ls in (("ff", "-"), ("none", "--")):
            p = bm.detection_p(sw[(sw.scenario == sc) & (sw.coupling == (0.5 if sc == "ff" else 0.0))], m)
            r = (p < bm.SIG).groupby(level="n_trials").mean()
            ax.plot(r.index, r.values, ls, marker="o", ms=2.5, lw=0.9, color=bm.COLORS[m])
        if m in bm.DIRECTION:
            y = sw[(sw.scenario == "ff") & (sw.coupling == 0.5) & (sw.quantity == bm.DIRECTION[m])]
            r = (y.p_upper < bm.SIG).groupby(y.n_trials).mean()
            ax2.plot(r.index, r.values, "-o", ms=2.5, lw=0.9, color=bm.COLORS[m], label=used[m])
    ns = int(sw.groupby(["scenario", "n_trials", "quantity"]).size().min())
    for a_ in (ax, ax2):
        a_.axvline(C2["n_trials"], color="0.6", lw=0.6)
        a_.text(C2["n_trials"], 1.0, f" 007: {C2['n_trials']} trials/draw", fontsize=6, va="top", color="0.35")
        a_.axhline(bm.SIG, color="0.5", lw=0.6, ls=":")
        a_.set_xticks([15, 30, 60])
        a_.set_ylim(-0.03, 1.03)
        a_.set_xlabel("trials per condition")
    ax.set_ylabel("detection rate")
    ax.plot([], [], "k-", label="ff, coupling 0.5 (power)")
    ax.plot([], [], "k--", label="none (false positives)")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.02, 0.93))
    ax.set_title(f"e  Detection vs trials (20 units, {BIN}-ms bins)\nn = {ns} sims/point", loc="left")
    ax2.set_ylabel("fraction with correct A→B lead")
    ax2.set_title("f  Correct direction vs trials\n(ff, coupling 0.5)", loc="left")
    fig.suptitle(f"Method benchmark (005) on synthetic spikes with known ground truth: {BIN}-ms bins, condition-mean "
                 f"residuals, {bm.N_REP} simulated sessions per cell, {bm.N_SH} trial shuffles each; "
                 "* conditioning on the true (simulated) driver, not possible on data", fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.07)
    save(fig, "04_benchmark", pdf)


# ---------------------------------------------------------------- 5. forest

SPEC = [("P1", "W_lick", st.P1_PAIRS, list(st.WINDOWS), +1), ("P2", "W_nolick", st.P2_PAIRS, list(st.WINDOWS), -1),
        ("P3", "W_lick-A_hit", st.P1_PAIRS, list(st.WINDOWS), +1), ("P3b", "A_hit", st.P1_PAIRS, list(st.WINDOWS), 0),
        ("PM", "W_lick_pm150", st.P1_PAIRS, ["early"], +1), ("PM", "A_hit_pm150", st.P1_PAIRS, ["early"], 0)]


def effects(wm, quantity):
    k = ["session_id", "mouse_id", "cohort", "area_a", "area_b", "quantity", "window"]
    wl = wm[wm.condition == "W_lick"].set_index(k).excess
    ah = wm[wm.condition == "A_hit"].set_index(k).excess
    dd = (wl - ah).dropna().reset_index()
    dd["condition"] = "W_lick-A_hit"
    rows = []
    for pred, cond, pairs, wins, sign in SPEC:
        f = dd if pred == "P3" else wm
        for a, b in pairs:
            for w in wins:
                x = f[(f.condition == cond) & (f.area_a == a) & (f.area_b == b) & (f.quantity == quantity) &
                      (f.window == w)]
                rp = x[x.cohort == "R+"].excess.dropna().to_numpy()
                rm = x[x.cohort == "R-"].excess.dropna().to_numpy()
                r = dict(prediction=pred, condition=cond, pair=f"{a} -> {b}", window=w, quantity=quantity,
                         expected_sign=sign, n_Rp=len(rp), n_Rm=len(rm), diff=np.nan, ci_lo=np.nan, ci_hi=np.nan)
                if len(rp) >= st.MIN_SESSIONS and len(rm) >= st.MIN_SESSIONS:
                    r["diff"] = rp.mean() - rm.mean()
                    r["ci_lo"], r["ci_hi"] = welch_ci(rp, rm)
                rows.append(r)
    return pd.DataFrame(rows)


def fig_forest(wm, tests, pdf):
    keys = ["prediction", "condition", "pair", "window", "quantity"]
    eff = {}
    for tag in TAGS:
        e = effects(wm[tag], PRIMARY)
        t = tests[tag][tests[tag].quantity == PRIMARY][keys + ["diff", "p_mw", "p_welch", "p_mw_fdr", "p_welch_fdr"]]
        e = e.merge(t.rename(columns={"diff": "diff_008"}), on=keys, how="left")
        bad = (e["diff"] - e["diff_008"]).abs() > 1e-6
        assert not bad.any(), f"forest differences disagree with tests_{tag}.csv"
        rob = effects(wm[tag], ROBUST)[["prediction", "condition", "pair", "window", "diff"]]
        e = e.merge(rob.rename(columns={"diff": "diff_rest"}), on=["prediction", "condition", "pair", "window"])
        e["population"] = tag
        eff[tag] = e
    pd.concat(eff.values()).to_csv(OUT / "forest_effects.csv", index=False)
    n = len(eff["learners"])
    fig = plt.figure(figsize=(17, 0.205 * n + 2.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 1.55, 2.2, 1.55], wspace=0.04, left=0.17, right=0.995,
                          top=1 - 1.15 / (0.205 * n + 2.2), bottom=0.6 / (0.205 * n + 2.2))
    ci = np.abs(np.r_[eff["learners"][["ci_lo", "ci_hi"]].to_numpy().ravel(),
                      eff["all_mice"][["ci_lo", "ci_hi"]].to_numpy().ravel()])
    lim = np.nanpercentile(ci, 90) * 1.15  # a few n = 3 cells have very wide CIs: clipped, arrow at the limit
    n_fdr = sum(int(((e.p_mw_fdr < 0.05) | (e.p_welch_fdr < 0.05)).sum()) for e in eff.values())
    y = np.arange(n)
    for k, tag in enumerate(TAGS):
        e = eff[tag]
        ax = fig.add_subplot(gs[0, 2 * k])
        tx = fig.add_subplot(gs[0, 2 * k + 1], sharey=ax)
        tx.set_axis_off()
        prev = None
        for i, r in enumerate(e.itertuples()):
            grp = (r.prediction, r.condition)
            if grp != prev:
                if i:
                    ax.axhline(i - 0.5, color="0.6", lw=0.5)
                    tx.axhline(i - 0.5, color="0.6", lw=0.5, xmin=0, xmax=1)
                exp = {1: "expected R+ > R-", -1: "expected R- > R+", 0: "reference (Go in both)"}[r.expected_sign]
                ax.text(lim * 0.98, i - 0.42, f"{r.prediction} {r.condition}: {exp}", fontsize=6, ha="right",
                        va="top", color="0.25", style="italic")
                prev = grp
            if np.isfinite(r.diff):
                sig = (r.p_mw < 0.05) or (r.p_welch < 0.05)
                fdr = (r.p_mw_fdr < 0.05) or (r.p_welch_fdr < 0.05)
                ax.plot([max(r.ci_lo, -lim), min(r.ci_hi, lim)], [i, i], color="0.2", lw=0.8)
                for end, side in ((r.ci_lo, -1), (r.ci_hi, 1)):
                    if side * end > lim:
                        ax.plot(side * lim * 0.995, i, marker=">" if side > 0 else "<", ms=3.5, color="0.2",
                                ls="none")
                ax.plot(r.diff, i, ls="none", ms=4 if not fdr else 5.5, mfc="0.15" if sig else "white", mec="0.15",
                        mew=0.8, marker="D" if fdr else "o")
            else:
                ax.text(-lim * 0.97, i, f"not tested (n < {st.MIN_SESSIONS})", fontsize=5.8, ha="left", va="center",
                        color="0.5")
            same = "–" if not (np.isfinite(r.diff) and np.isfinite(r.diff_rest)) else \
                ("yes" if np.sign(r.diff) == np.sign(r.diff_rest) else "no")
            vals = [f"{r.n_Rp}/{r.n_Rm}", fmt_p(r.p_mw), fmt_p(r.p_welch), fmt_p(r.p_mw_fdr), fmt_p(r.p_welch_fdr),
                    same]
            for c, v in enumerate(vals):
                bold = (c in (1, 2) and v != "–" and float(v) < 0.05) or (c in (3, 4) and v != "–" and float(v) < 0.05)
                tx.text(0.02 + c * 0.165, i, v, fontsize=5.9, va="center", fontweight="bold" if bold else "normal")
        for c, h in enumerate(["n\nR+/R-", "p\nMW", "p\nWelch", "FDR\nMW", "FDR\nWelch", "|rest\nsame sign"]):
            tx.text(0.02 + c * 0.165, -1.0, h, fontsize=6.2, va="bottom", fontweight="bold")
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(n - 0.5, -2.4)
        tx.set_xlim(0, 1)
        if k == 0:
            ax.set_yticks(y, [f"{r.pair.replace(' -> ', ' → ')}  {r.window}" for r in e.itertuples()], fontsize=6)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel("R+ minus R- (lagRRR_dir excess over shuffle; mean, Welch 95% CI)")
        nses = wm[tag].groupby("cohort").session_id.nunique()
        ax.set_title(f"{TAGS[tag]}: R+ {nses.get('R+', 0)}, R- {nses.get('R-', 0)} sessions", loc="left",
                     fontsize=8.5, fontweight="bold")
        ax.spines["left"].set_visible(k == 0)
    fig.suptitle("Results: cohort difference of the primary direction index (A→B minus B→A) per prediction, pair and "
                 "window. Filled = p < 0.05 (MW or Welch, uncorrected); diamond = FDR < 0.05 "
                 f"({n_fdr} rows); wider CIs are clipped (arrow);\n"
                 "'|rest same sign' = lagRRR|rest gives the same sign (pre-registered requirement). Windows (centres, "
                 "s): " + ", ".join(f"{k} {lo:g}..{hi:g}" for k, (lo, hi) in st.WINDOWS.items())
                 + "; PM early = centre 0.05 only", fontsize=8.5)
    save(fig, "05_results_forest", pdf)
    return eff


# ---------------------------------------------------------------- 6. positive control

def fig_positive_control(wm, pc, pdf):
    rng = np.random.default_rng(1)
    conds = ["W_lick", "W_nolick", "A_hit", "NS_cr"]
    wins = list(st.WINDOWS)
    a, b = st.PC_PAIR
    fig, axs = plt.subplots(2, len(conds), figsize=(15, 7.2), sharey=True)
    vals_all = []
    for tag in TAGS:
        x = wm[tag]
        vals_all.append(x[(x.area_a == a) & (x.area_b == b) & (x.quantity == PRIMARY)].excess.to_numpy())
    v = np.concatenate(vals_all)
    lo, hi = np.nanmin(v), np.nanmax(v)
    pad = (hi - lo) * 0.12
    for row, tag in enumerate(TAGS):
        x = wm[tag]
        x = x[(x.area_a == a) & (x.area_b == b) & (x.quantity == PRIMARY)]
        p = pc[tag][(pc[tag].quantity == PRIMARY)]
        for j, c in enumerate(conds):
            ax = axs[row, j]
            for i, w in enumerate(wins):
                for k, coh in enumerate(COHORTS):
                    vv = x[(x.condition == c) & (x.window == w) & (x.cohort == coh)].excess.dropna().to_numpy()
                    xp = i + (k - 0.5) * 0.36
                    ax.scatter(xp + jitter(len(vv), rng), vv, s=7, color=COL[coh], alpha=0.65, lw=0)
                    if len(vv):
                        se = vv.std(ddof=1) / np.sqrt(len(vv)) if len(vv) > 1 else 0
                        ax.errorbar(xp, vv.mean(), yerr=se, fmt="_", color="k", ms=9, lw=0.9, capsize=0)
                    q = p[(p.condition == c) & (p.window == w) & (p.cohort == coh)]
                    pv = q.p_wilcoxon.iloc[0] if len(q) else np.nan
                    ax.text(xp, hi + pad * 0.55, f"n={len(vv)}\np={fmt_p(pv)}", ha="center", va="center",
                            fontsize=5.6, color=COL[coh], fontweight="bold" if pv < 0.05 else "normal")
            ax.axhline(0, color="0.4", lw=0.6)
            ax.set_xticks(range(len(wins)), wins)
            ax.set_ylim(lo - pad * 0.3, hi + pad * 1.15)
            ax.set_title(f"{c} — {TAGS[tag]}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{a} → {b} lagRRR_dir excess\n(+ = thalamus leads)")
    for coh in COHORTS:
        axs[0, 0].scatter([], [], color=COL[coh], s=10, label=coh)
    axs[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Positive control: {a} → {b} direction index per session (dots), mean ± s.e.m. (black); "
                 "Wilcoxon signed-rank vs 0 per cohort (008). Expected: positive in both cohorts.", fontsize=9)
    fig.tight_layout()
    save(fig, "06_positive_control", pdf)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    covdf, meta, sx, wm, tests, pc, bsum, braw = load_all()
    with PdfPages(OUT / "summary_figures.pdf") as pdf:
        fig_dataset(covdf, meta, pdf)
        fc = fig_funnel(covdf, meta, sx, pdf)
        fig_pipeline(pdf)
        fig_benchmark(bsum, braw, pdf)
        eff = fig_forest(wm, tests, pdf)
        fig_positive_control(wm, pc, pdf)
    print(fc.to_string())
    for tag, e in eff.items():
        ok = e[np.isfinite(e["diff"])]
        print(tag, "tests", len(ok), "uncorrected MW<0.05", int((ok.p_mw < 0.05).sum()), "Welch<0.05",
              int((ok.p_welch < 0.05).sum()), "FDR<0.05", int(((ok.p_mw_fdr < 0.05) | (ok.p_welch_fdr < 0.05)).sum()),
              "rest same sign", int((np.sign(ok["diff"]) == np.sign(ok.diff_rest)).sum()))
    print("units per area (analysed sessions):", {a: int(meta[f"u_{a}"].sum()) for a in PAIR_AREAS})


if __name__ == "__main__":
    main()
