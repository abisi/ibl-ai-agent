"""Article-style report: whisker and auditory stimulus-presence decoding across brain regions (skills/project-report).
Reads the result tables of the project results folder (<target>_<epoch>/, <target>_task_vs_passive/, anatomy/) and the
N = 200 onsets of the stimulus-identity project (whisker vs auditory) for the comparison, freezes them in
report/_snapshots/<date>/, writes numbers.json and report.md (Pandoc Markdown with LaTeX math), copies every figure used
into report/figures/ and copies build.sh there. No number in the text is typed by hand.
Structure (as the identity report, user 2026-10-06): passive trials, then task trials, then their comparison; whisker and
auditory targets are dimensions inside each section; then onset vs information, the identity comparison and anatomy.
Repetitions: 100 iterations x 10 linear shifts everywhere are final for this analysis (user 2026-10-23).
The matched-accuracy control (003) enters when its runs are done (matched_accuracy_check.csv present); until then the text
says it is pending.
Run (haas):  cd ~/code/unit_spikes_analysis && PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities ./.venv/bin/python \
             ~/code/ibl-ai-agent/projects/ssl-stimulus-detection-decoding/report/build_report.py
Render (locally, Quarto + TinyTeX):  bash projects/ssl-stimulus-detection-decoding/report/render.sh
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
sys.path.insert(0, str(EA))
AR = importlib.import_module("_areas")
m1 = importlib.import_module("001_detection_pseudopop")
m4 = importlib.import_module("004_main_figures")

HOME = AR.HOME
OUT = HOME / "report"
IDENT = AR.RES / "ssl-stimulus-arrival-decoding"          # stimulus-identity project (whisker vs auditory)
sys.path.insert(0, str(REPO / "skills" / "project-report"))
from report_lib import Report  # noqa: E402

R_ = Report(banned=("arrival",))     # user 2026-10-06 (identity project): never "stimulus arrival"
num, pv, ref, figure, mdtable, eq = R_.num, R_.pv, R_.ref, R_.figure, R_.table, R_.eq
ANCHORS = R_.anchors
LEVELS = ("area_group", "area_acronym_custom")
TARGETS = ("whisker", "auditory")
EPOCHS = ("passive", "active")
EW = {"passive": "passive", "active": "task"}
LV = {"area_group": "area groups", "area_acronym_custom": "areas"}
NOSTIM = {"active": "correct rejections", "passive": "pre-stimulus windows"}
N_MAIN = m4.N_MAIN
SNAP = os.environ.get("REPORT_SNAPSHOT")   # snapshot tag (default: today)
READ: list = []                              # every statistics file read (one manifest at the end)


# ---------------------------------------------------------------------------------------------------------- helpers
def rd(path, **kw):
    """read a statistics table through the frozen snapshot (skills/project-report: frozen statistics)"""
    READ.append(Path(path))
    p = R_.snapshot([path], OUT, tag=SNAP)[0]
    return pd.read_parquet(p, **kw) if p.suffix == ".parquet" else pd.read_csv(p, **kw)


def ms(v):
    return "n.s." if not np.isfinite(v) else f"{v:.0f}"


def rng_ms(r):
    return f"{ms(r.onset_ms)} [{ms(r.lo)}, {ms(r.hi)}]" + (" †" if r.flag else "")


def short(a):
    return m4.label(a)


def p3(v):
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def lst(q, k=None, unit=" ms"):
    q = q if k is None else q.head(k)
    return ", ".join(f"{short(a)} ({ms(r.onset_ms)}{unit})" for a, r in q.iterrows())


# ---------------------------------------------------------------------------------------------------------- data
def cond_data(t, e):
    f = HOME / f"{t}_{e}"
    d = dict(folder=f, prov=json.loads((f / "provenance_001.json").read_text()), ob={}, w200={},
             w=rd(f / "window_accuracy.csv"), ov=rd(f / "onset_vs_window_stats.csv"))
    ob = rd(f / "onset_bootstrap_N200.csv")
    for level in LEVELS:
        q = ob[ob.level == level].copy()
        q["flag"] = m4.unreliable(q)
        d["ob"][level] = q.set_index("area")
        d["w200"][level] = d["w"][(d["w"].level == level) & (d["w"].N == N_MAIN)].set_index("area")
    d["n_iter"] = int(ob.n_iter.iloc[0])
    mc = f / "matched_accuracy_check.csv"
    d["matched"] = rd(mc) if mc.exists() and (f / "figures" / "matched_accuracy.png").exists() else None
    d["matched_n"] = rd(f / "matched_n.csv") if (f / "matched_n.csv").exists() else None
    d["onsets"] = rd(f / "onsets.csv")
    return d


def onset_at(t, area, n, level="area_group"):
    """first above-chance onset (2-ms steps, 20-ms bins) of the averaged time course at N neurons; NaN if none"""
    o = C[(t, "active")]["onsets"]
    o = o[(o.level == level) & (o.area == area) & (o.N == n) & (o.resolution == "zoom")]
    return float(o.onset_ms.iloc[0]) if len(o) else np.nan


C = {(t, e): cond_data(t, e) for t in TARGETS for e in EPOCHS}
X = {t: dict(tests=rd(HOME / f"{t}_task_vs_passive" / "tables" / "active_vs_passive_tests.csv"),
             rank=rd(HOME / f"{t}_task_vs_passive" / "tables" / "area_ranking.csv"),
             ss=rd(HOME / f"{t}_task_vs_passive" / "tables" / "sample_sizes.csv"),
             seq=rd(HOME / f"{t}_task_vs_passive" / "tables" / "trial_sequence_check.csv"),
             fig=HOME / f"{t}_task_vs_passive" / "figures") for t in TARGETS}
ANA = rd(HOME / "anatomy" / "anatomy_tests.csv")
ANA_BY = rd(HOME / "anatomy" / "anatomy_by_area.csv")
ANAFIG = HOME / "anatomy" / "figures"
IDOB = {e: rd(IDENT / f"{e}_final_n200" / "onset_bootstrap_N200.csv") for e in EPOCHS}
for e in EPOCHS:
    IDOB[e]["flag"] = m4.unreliable(IDOB[e])
INJ = rd(AR.RES / "_sensory_spatial_maps" / "projection_experiments.csv")
MATCHED = C[("whisker", "active")]["matched"] is not None and C[("auditory", "active")]["matched"] is not None
SESS = {(t, e): sorted(C[(t, e)]["prov"]["sessions"]) for t in TARGETS for e in EPOCHS}
ALL_SESS = sorted(set().union(*SESS.values()))
MICE = sorted({s.split("_")[0] for s in ALL_SESS})


def reliable_sorted(t, e, level="area_group"):
    ob = C[(t, e)]["ob"][level]
    return ob[~ob.flag].sort_values("onset_ms")


def unrel(t, e, level="area_group"):
    return [short(a) for a, r in C[(t, e)]["ob"][level].iterrows() if r.flag]


def tst(t, level, measure):
    T = X[t]["tests"]
    return T[(T.level == level) & (T.measure == measure)].iloc[0]


def ovw(t, e, level="area_group", set_="N200_all_areas"):
    ov = C[(t, e)]["ov"]
    return ov[(ov.level == level) & (ov["set"] == set_)].iloc[0]


def ana(t, e, level, source, measure="early"):
    q = ANA[(ANA.target == t) & (ANA.epoch == e) & (ANA.level == level) & (ANA.source == source) & (ANA.measure == measure)]
    return q.iloc[0] if len(q) else None


def ident_cmp(t, e, level="area_group"):
    """detection vs identity onsets per area, both reliable"""
    d = C[(t, e)]["ob"][level]
    i = IDOB[e][IDOB[e].level == level].set_index("area")
    m = pd.DataFrame(dict(det=d.onset_ms, ide=i.onset_ms, ok=~d.flag & ~i.flag)).dropna()
    m = m[m.ok]
    w = stats.wilcoxon(m.det, m.ide) if len(m) > 5 and (m.det != m.ide).any() else None
    return dict(n=len(m), med=float((m.det - m.ide).median()), p=w.pvalue if w is not None else np.nan,
                rho=stats.spearmanr(m.det, m.ide).statistic)


def ss_range(t, e, level, col="n_sessions"):
    q = X[t]["ss"]
    q = q[(q.epoch == e) & (q.level == level) & (q.n_sessions > 0)][col]
    return f"{int(q.min())}-{int(q.max())}"


def fig_main(t, e, level):
    return C[(t, e)]["folder"] / "figures" / f"detection_main_N200_{level}.png"


# anchors in reading order (labels fixed before any text cites them)
_figs = [("fig-w-passive", "Figure 1"), ("fig-a-passive", "Figure 2"), ("fig-w-task", "Figure 3"), ("fig-a-task", "Figure 4"),
         ("fig-onset-acc", "Figure 5")]
_figs += [("fig-matched", "Figure 6")] if MATCHED else []
_k = len(_figs)
_figs += [("fig-avp", f"Figure {_k + 1}"), ("fig-anatomy", f"Figure {_k + 2}")]
_supp = [("fig-s-w-passive-areas", "Figure S1"), ("fig-s-a-passive-areas", "Figure S2"), ("fig-s-w-task-areas", "Figure S3"),
         ("fig-s-a-task-areas", "Figure S4"), ("fig-s-onset-acc-aud", "Figure S5"), ("fig-s-n", "Figure S6"),
         ("fig-s-avp-aud", "Figure S7"), ("fig-s-anatomy-areas", "Figure S8"), ("fig-s-anatomy-liu", "Figure S9")]
_tbls = [("tbl-onsets", "Table 1"), ("tbl-ident", "Table 2"), ("tbl-tests", "Table 3"), ("tbl-anatomy", "Table 4")]
_tbls += [("tbl-matched", "Table 5")] if MATCHED else []
R_.register([("eq-corr", "Eq. (1)"), ("eq-early", "Eq. (2)"), ("eq-exp", "Eq. (3)")] + _figs + _supp + _tbls
            + [("tbl-s-sizes-groups", "Table S1"), ("tbl-s-sizes-areas", "Table S2"), ("tbl-s-onsets-areas", "Table S3"),
               ("tbl-s-seq", "Table S4"), ("tbl-a-params", "Table A1"), ("tbl-a-files", "Table A2"),
               ("tbl-a-registry", "Table A3")])


# ---------------------------------------------------------------------------------------------------------- tables
def onset_table(level):
    areas = AR.LEVELS[level]
    P = C[("whisker", "passive")]["ob"][level]
    key = lambda a: (P.loc[a, "onset_ms"] if a in P.index and np.isfinite(P.loc[a, "onset_ms"]) else 1e9)
    rows = []
    for a in sorted(areas, key=key):
        r = {"Area": short(a)}
        for t in TARGETS:
            for e in EPOCHS:
                ob, w = C[(t, e)]["ob"][level], C[(t, e)]["w200"][level]
                r[f"{t.capitalize()} {EW[e]}"] = (f"{rng_ms(ob.loc[a])}; {w.loc[a, 'mean']:.2f}" if a in ob.index and a in w.index else "")
        rows.append(r)
    return pd.DataFrame(rows)


def ident_table():
    rows = []
    for a in AR.COARSE:
        r = {"Area group": short(a)}
        for e in EPOCHS:
            i = IDOB[e][(IDOB[e].level == "area_group")].set_index("area")
            r[f"Identity {EW[e]}"] = (ms(i.loc[a, "onset_ms"]) + (" †" if i.loc[a, "flag"] else "")) if a in i.index else ""
            for t in TARGETS:
                o = C[(t, e)]["ob"]["area_group"]
                r[f"{t.capitalize()} {EW[e]}"] = (ms(o.loc[a, "onset_ms"]) + (" †" if o.loc[a, "flag"] else "")) if a in o.index else ""
        rows.append(r)
    return pd.DataFrame(rows)


def tests_table():
    rows = []
    for t in TARGETS:
        for _, r in X[t]["tests"].iterrows():
            rows.append({"Target": t, "Level": LV[r.level], "Measure": {"onset_ms": "onset (ms)", "accuracy_5_50ms": "accuracy 5-50 ms",
                                                                       "baseline_-150_0ms": "baseline -150..0 ms"}[r.measure],
                         "n": int(r.n), "Median passive - task": f"{r.median_diff:+.3f}", "Mean": f"{r.mean_diff:+.3f}",
                         "p Wilcoxon": p3(r.p_wilcoxon), "p paired t": p3(r.p_paired_t)})
    return pd.DataFrame(rows)


def anatomy_table():
    rows = []
    for level in LEVELS:
        for t in TARGETS:
            other = "auditory" if t == "whisker" else "whisker"
            for e in EPOCHS:
                own, oth, liu = ana(t, e, level, t), ana(t, e, level, other), ana(t, e, level, "liu")
                f = lambda r: "" if r is None else f"{r.pearson_r_logdens:+.2f} ({p3(r.pearson_p)}); ρ {r.spearman_rho:+.2f}"
                rows.append({"Level": LV[level], "Target": t, "Trials": EW[e], "n": int(own.n_areas),
                             "Own cortex (Allen)": f(own), "Other cortex (Allen)": f(oth), "wS1 + wS2 (Liu)": f(liu)})
    return pd.DataFrame(rows)


def fmt_ms(x):
    return "none" if np.isnan(x) else f"{x:.0f}"


def matched_spread():
    """per target and reference: onsets of the groups at matched accuracy (min, max, n) vs the same groups at N = 100"""
    out = []
    for t in TARGETS:
        m = C[(t, "active")]["matched"]
        m = m[m.level == "area_group"]
        for ref_, q in m.groupby("reference", sort=False):
            om = np.array([onset_at(t, a, int(n)) for a, n in zip(q.area, q.matched_N)])
            o1 = np.array([onset_at(t, a, 100) for a in q.area])
            ok = ~np.isnan(om)
            out.append(dict(t=t, ref=short(ref_), n=int(ok.sum()), lo=np.nanmin(om), hi=np.nanmax(om),
                            lo100=np.nanmin(o1[ok]) if (ok & ~np.isnan(o1)).any() else np.nan,
                            hi100=np.nanmax(o1[ok]) if (ok & ~np.isnan(o1)).any() else np.nan,
                            first=short(q.area.to_numpy()[ok][np.argmin(om[ok])]), last=short(q.area.to_numpy()[ok][np.argmax(om[ok])])))
    return out


def matched_verdict():
    d = matched_spread()
    nar = [x for x in d if (x["hi"] - x["lo"]) < (x["hi100"] - x["lo100"])]
    same = [x for x in d if x not in nar]
    mid = sum(x["first"] == "Midbrain" for x in d)
    txt = (f"The onset range at matched accuracy was narrower than at N = 100 in {len(nar)} of {len(d)} target × reference "
           f"comparisons (largest range at matched accuracy {max(x['hi'] - x['lo'] for x in d):.0f} ms, at N = 100 "
           f"{max(x['hi100'] - x['lo100'] for x in d):.0f} ms)")
    if same:
        txt += "; not narrower: " + ", ".join(f"{x['t']} with reference {x['ref']}" for x in same)
    return (txt + f". Midbrain was the earliest group in {mid} of {len(d)} comparisons. Part of the onset differences at a fixed N "
            f"therefore follow from different accuracies; differences of up to {max(x['hi'] - x['lo'] for x in d):.0f} ms remain at "
            "matched accuracy.")


def matched_table():
    rows = []
    for t in TARGETS:
        m = C[(t, "active")]["matched"]
        m = m[m.level == "area_group"]
        for _, r in m.iterrows():
            rows.append({"Target": t, "Reference (N = 100)": short(r.reference), "Area group": short(r.area),
                         "Matched N": int(r.matched_N), "Target acc.": f"{r.target:.3f}", "Achieved acc.": f"{r.achieved:.3f}",
                         "Onset (ms)": fmt_ms(onset_at(t, r.area, int(r.matched_N)))})
    return pd.DataFrame(rows)


def sizes_table(level):
    rows = []
    for a in AR.LEVELS[level]:
        r = {"Area": short(a)}
        for t in TARGETS:
            q = X[t]["ss"]
            for e in EPOCHS:
                x = q[(q.level == level) & (q.area == a) & (q.epoch == e)]
                lab = f"{t[0].upper()} {EW[e]}"
                if len(x) and x.n_sessions.iloc[0] > 0:
                    x = x.iloc[0]
                    r[f"{lab}: sess. / mice"] = f"{int(x.n_sessions)} / {int(x.n_mice)}"
                    r[f"{lab}: events"] = f"{int(x.n_stim_events)} / {int(x.n_nostim_events)}"
                else:
                    r[f"{lab}: sess. / mice"], r[f"{lab}: events"] = "0", ""
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- sections
def sec_front():
    lv = "area_group"
    rw, ra = reliable_sorted("whisker", "passive"), reliable_sorted("auditory", "passive")
    tw = tst("whisker", lv, "baseline_-150_0ms")
    ta = tst("auditory", lv, "baseline_-150_0ms")
    tw_acc = tst("whisker", lv, "accuracy_5_50ms")
    ow = ovw("whisker", "active", lv, "all_areas_all_N")
    aw_own, aw_oth = ana("whisker", "active", lv, "whisker"), ana("whisker", "active", lv, "auditory")
    iw = ident_cmp("whisker", "active")
    summary_claims = [
        R_.claim("c1", f"Whisker presence is decodable first in {', '.join(short(a) for a in rw.head(4).index)} (≤ {ms(rw.onset_ms.iloc[3])} ms, passive trials).",
                 "Figure 1", "onset, N = 200", f"{len(rw)} reliable groups"),
        R_.claim("c2", f"Auditory presence is decodable in most area groups within {ms(ra.onset_ms.quantile(.75))} ms (75th percentile of reliable passive onsets).",
                 "Figure 2", "onset, N = 200", f"{len(ra)} reliable groups"),
        R_.claim("c3", "Onset decreases with early accuracy along a decreasing exponential.", ANCHORS["fig-onset-acc"],
                 f"ρ = {ow.rho:.2f}, R² exp {ow.r2_exp:.2f} vs lin {ow.r2_ols:.2f}", f"{int(ow.n)} area × N"),
        R_.claim("c4", "Whisker-presence onsets are the same on task and passive trials; early accuracy is lower on passive trials.",
                 ANCHORS["fig-avp"], f"onset median {tst('whisker', lv, 'onset_ms').median_diff:+.0f} ms (p = {tst('whisker', lv, 'onset_ms').p_wilcoxon:.2f}); acc. {tw_acc.median_diff:+.3f} (p = {tw_acc.p_wilcoxon:.3f})",
                 f"{int(tw_acc.n)} groups"),
        R_.claim("c5", "Passive pre-stimulus decoding is above the task-trial value (both targets).", ANCHORS["fig-avp"],
                 f"whisker {tw.median_diff:+.3f} (p = {tw.p_wilcoxon:.3f}); auditory {ta.median_diff:+.3f} (p = {ta.p_wilcoxon:.4f})",
                 f"{int(tw.n)} groups", "fragile", "candidate cause: predictable passive stimulus timing"),
        R_.claim("c6", "Whisker-presence onsets are later than whisker vs auditory (identity) onsets.", ANCHORS["tbl-ident"],
                 f"median {iw['med']:+.0f} ms, Wilcoxon p = {iw['p']:.3f}", f"{iw['n']} groups", "fragile", "different nulls"),
        R_.claim("c7", "Early accuracy rises with projection density from the stimulus' cortex, but not specifically.",
                 ANCHORS["fig-anatomy"], f"whisker task: own r = {aw_own.pearson_r_logdens:+.2f}, other r = {aw_oth.pearson_r_logdens:+.2f}",
                 f"{int(aw_own.n_areas)} groups", "fragile", "auditory density predicts whisker decoding as well"),
    ]
    _ = summary_claims
    return f"""---
title: "Whisker and auditory stimulus-presence decoding across brain regions"
author: "Axel Bisi (analysis with Claude Code)"
date: "{date.today().isoformat()}"
---

**Data version.** KS4 spike sorting (NWB_ks4), v2 unit table (good + mua units), SSL whisker-training sessions of both
cohorts, learning day and expert days pooled: {num('n_sess', len(ALL_SESS))} sessions from {num('n_mice', len(MICE))} mice
(task trials: {num('n_sess_wt', len(SESS[('whisker', 'active')]))} sessions with whisker and
{num('n_sess_at', len(SESS[('auditory', 'active')]))} with auditory events; passive trials:
{num('n_sess_wp', len(SESS[('whisker', 'passive')]))} and {num('n_sess_ap', len(SESS[('auditory', 'passive')]))}).
Decoding of stimulus presence (whisker vs no stimulus, auditory vs no stimulus) in all {num('n_groups', len(AR.COARSE))}
area groups and the {num('n_fine', len(AR.FINE))} best-sampled areas; sample sizes in {ref('tbl-s-sizes-groups')} and
{ref('tbl-s-sizes-areas')}. Every condition: {C[('whisker', 'active')]['n_iter']} iterations × {m1.N_SHIFT} linear label
shifts.

# Abstract

When and where does a brief whisker deflection or an auditory tone become detectable in neural population activity? We
decoded stimulus presence (stimulus vs no stimulus) from pseudo-populations of N neurons pooled across sessions, in 20-ms
bins stepped by 2 ms, against a linear-shift null that keeps slow drifts, and took the first reliably above-chance bin as
the onset. No-stimulus events were correct rejections on task trials and pre-stimulus windows on passive trials.

With N = {N_MAIN} neurons and passive trials, whisker presence was decodable first in
{', '.join(short(a) for a in rw.head(4).index)} ({ms(rw.onset_ms.iloc[0])}-{ms(rw.onset_ms.iloc[3])} ms) and spread to
striatal, motor, hippocampal and frontal groups, the latest reliable group at {num('w_last', rw.onset_ms.max(), '{:.0f}')} ms. Auditory presence was decodable almost everywhere
within {num('aud_q75', ra.onset_ms.quantile(.75), '{:.0f}')} ms.

Onset tracked the amount of early information along a decreasing exponential (whisker, task trials, every area × N:
Spearman ρ = {num('rho_all_w_task', ow.rho, '{:.2f}')}). Task and passive trials gave the same whisker onsets; whisker
early accuracy was {num('w_acc_diff', abs(tw_acc.median_diff), '{:.3f}')} lower on passive trials, and passive pre-stimulus
decoding was slightly above the task-trial value for both stimuli.

Across area groups, early accuracy rose with the Allen projection density from the stimulus' own cortex, but whisker
decoding rose at least as much with the auditory-cortex density, so the anatomical link is not specific.

# Key results

1. **Whisker presence, passive trials.** First: {lst(rw, 6)}{'; unreliable: ' + ', '.join(unrel('whisker', 'passive')) if unrel('whisker', 'passive') else ''} ({ref('fig-w-passive')}, {ref('tbl-onsets')}).
2. **Auditory presence, passive trials.** First: {lst(ra, 6)}; {num('aud_n_by16', int((ra.onset_ms <= 16).sum()))} of {len(ra)} reliable groups by 16 ms ({ref('fig-a-passive')}).
3. **Task trials, same order** for both stimuli ({ref('fig-w-task')}, {ref('fig-a-task')}).
4. **Onset follows early information.** Whisker, task trials, every area × N: ρ = {ow.rho:.2f}, exponential R² = {ow.r2_exp:.2f} vs linear {ow.r2_ols:.2f}, n = {int(ow.n)} ({ref('fig-onset-acc')}).
5. **Task vs passive.** Whisker onset passive - task: median {tst('whisker', lv, 'onset_ms').median_diff:+.0f} ms (Wilcoxon p {pv('p_w_onset', tst('whisker', lv, 'onset_ms').p_wilcoxon)}, n = {int(tst('whisker', lv, 'onset_ms').n)});
   pre-stimulus accuracy passive - task: whisker {tw.median_diff:+.3f} (p {pv('p_w_base', tw.p_wilcoxon)}), auditory {ta.median_diff:+.3f} (p {pv('p_a_base', ta.p_wilcoxon)}) ({ref('fig-avp')}, {ref('tbl-tests')}).
6. **Presence vs identity.** Whisker-presence onsets were {num('ident_w_task_med', iw['med'], '{:+.0f}')} ms (median) relative to
   whisker vs auditory onsets on task trials (Wilcoxon p {pv('p_ident_w_task', iw['p'])}, n = {iw['n']} groups; {ref('tbl-ident')}).
7. **Anatomy, not specific.** Whisker task trials, area groups: early accuracy vs whisker-cortex density r = {aw_own.pearson_r_logdens:+.2f}
   (p {pv('p_ana_w_own', aw_own.pearson_p)}), vs auditory-cortex density r = {aw_oth.pearson_r_logdens:+.2f} (p {pv('p_ana_w_oth', aw_oth.pearson_p)}) ({ref('fig-anatomy')}, {ref('tbl-anatomy')}).
"""


def sec_intro():
    return f"""
# Introduction

{R_.methods('task')}
Outside the task, the same stimuli are delivered passively before and after the session. A companion project decoded
stimulus identity (whisker vs auditory) across the same regions. Identity can be read out wherever either stimulus evokes
a response, so it does not say which stimulus a region responds to. This project asks the presence question for each
stimulus separately: where and when can a whisker deflection, or a tone, be told apart from no stimulus at all?

Presence decoding attributes the information to one stimulus. A region whose onset for identity is early because it
responds to tones will show an early auditory-presence onset and a late or absent whisker-presence onset.

As in the identity project, passive trials come first (no licking, no reward), then task trials, then their comparison.
The whisker and auditory targets are shown side by side in each section. Two confounds are addressed explicitly:
decoding strength (a region with weaker information crosses any threshold later) and the precision of each onset. Last,
decoding per region is compared with the anatomical input from the stimulus' cortex as a sanity check.
"""


def sec_methods():
    p = C[("whisker", "active")]["prov"]
    pp = C[("whisker", "passive")]["prov"]
    inj = INJ.assign(mod=INJ.source.map({"SSp-bfd": "whisker", "SSs": "whisker", "AUDp": "auditory", "AUD-sec": "auditory"}))
    inj_w = inj[inj["mod"] == "whisker"]
    inj_a = inj[inj["mod"] == "auditory"]
    return f"""
# Methods

## Data and inclusion

All whisker-training sessions of the v2 unit table with KS4 NWB files (both cohorts, learning day and expert days pooled;
cohort and stage are not analysed separately). Units: quality label good or mua. A session contributes to an area when it
has ≥ {m1.MIN_UNITS} units of the area and ≥ {m1.MIN_TRIALS} events per class. Sample sizes per area and condition are in
{ref('tbl-s-sizes-groups')} (area groups: {ss_range('whisker', 'active', 'area_group')} sessions per group, whisker task
trials) and {ref('tbl-s-sizes-areas')}.

**Task trials.** {R_.methods('trials').strip()} Stimulus events: every whisker (or auditory) trial, licked or not. No-stimulus
events: correct rejections (no-stimulus trials without a lick).

**Passive trials.** Trials in passive context (fixed ~3 s inter-trial sequence, or labelled passive), pre- and post-task
blocks pooled. Passive blocks contain almost no no-stimulus trials (0-1 per session), so the no-stimulus events are sham
onsets {m1.SHAM:.1f} s before each stimulus of the other modality (whisker target: before auditory stimuli, and vice
versa). Each passive trial therefore gives one event, the labels follow the trial sequence, and the sham's whole window
(-200..+600 ms) ends 200 ms before a stimulus. Shams with a lick within {m1.LICK_WIN[0]:+.1f}..{m1.LICK_WIN[1]:+.1f} s are
dropped.

**Spikes.** Whisker stimulation produces an electrical artefact. Spikes from -10 to +5 ms around every whisker onset and
around every event onset of every class (correct rejections and shams included) are replaced by a Poisson train at the
unit's pre-onset rate (seeded per session), so that window cannot separate the classes.

## Pseudo-population decoding

One iteration draws {m1.N_SESS} eligible sessions with replacement, then N/{m1.N_SESS} units of the area within each drawn
session, and builds {p['t_train']} + {p['t_train']} pseudo-trials per class by balanced reuse of each session's real events,
separately for training and test folds; the unequal numbers of stimulus and no-stimulus events are absorbed by this balancing.

Per time bin, an L2-regularised logistic regression (liblinear; inverse regularisation C from {len(m1.C_GRID)} values by an
inner {p['n_inner']}-fold cross-validation) is trained and tested in a {p['n_outer']}-fold outer cross-validation built on
each session's real events; features are z-scored in the training fold. The score is the balanced accuracy $a_i(t)$ of
iteration $i$ in bin $t$.

**Null.** {R_.methods('shift_null').strip()} Here the same draw is decoded again with $S = {m1.N_SHIFT}$ shifts,
with C fixed from the real decode. A label shuffle is not used: stimulus trials and correct rejections are spread
differently through a session, and a shuffle would let slow drift count as information. The corrected accuracy is
{eq('eq-corr', r"d_i(t) = a_i(t) - \frac{1}{S}\sum_{s=1}^{S} a^{\mathrm{shift}}_{i,s}(t),")}
where $a^{{\\mathrm{{shift}}}}_{{i,s}}(t)$ is the balanced accuracy with the labels shifted by the $s$-th shift; $d_i(t)$ is 0
at chance and 0.5 for perfect decoding.

**Time resolutions.** Zoom: causal 20-ms bins in 2-ms steps from -20 to +100 ms (a bin labelled $t$ covers $[t-20, t)$ ms);
wide: causal 50-ms bins in 5-ms steps from -200 to +600 ms.

## Onset, unreliable onsets and accuracy measures

**Above chance.** Bin $t$ is above chance when the 5th percentile of $d_i(t)$ ({ref('eq-corr')}) over iterations is > 0.

**Onset.** The first post-stimulus zoom bin that is above chance and is followed by ≥ 80 % above-chance bins over the next
25 ms (11 of 13 bins). **Onset range**: 95 % range of the onset over {m4.N_BOOT} resamples (with replacement) of the iterations.

**Unreliable onset.** The 95 % range is wider than {m4.WIDE_RANGE_MS} ms or the onset is undefined in more than
{100 * (1 - m4.MIN_DEFINED):.0f} % of the resamples. Unreliable onsets are shown hatched with a dagger (†) and are excluded
from rankings and tests.

**Early accuracy** is the mean corrected accuracy over the $K$ zoom bins whose end lies 5-50 ms after the stimulus,
{eq('eq-early', r"\bar d_{5\text{-}50} = \frac{1}{I K}\sum_{i=1}^{I}\sum_{t \in [5, 50]} d_i(t),")}
with $I$ the number of iterations. **Peak accuracy** is the maximum over post-stimulus wide bins of the mean corrected
accuracy; **pre-stimulus accuracy** is the mean over the wide bins ending -150..0 ms (no post-stimulus spikes; expected 0).

## Population size, onset vs accuracy and matched accuracy

N = {', '.join(str(n) for n in m1.N_LIST)} neurons. **Onset vs early accuracy.** Spearman correlation over areas, and a
decreasing exponential fitted by least squares,
{eq('eq-exp', r"\mathrm{onset} = c + a\,e^{-x/\tau},")}
with $x$ the early accuracy ({ref('eq-early')}), compared with a straight line (R², AIC); 95 % band: 500 bootstrap
resamples of the points. **Matched accuracy** (task trials): for three reference area groups (Ss-whisker, Midbrain,
Auditory) at N = 100, each other group's N giving the same early accuracy is read from its accuracy-vs-N curve (log-N
interpolation, extrapolated beyond 500 up to 2000) and decoded again at that N.

## Task vs passive, identity comparison and anatomy

**Task vs passive.** Per area at N = {N_MAIN}: onset, early accuracy and pre-stimulus accuracy; paired over areas
(passive - task) with the Wilcoxon signed-rank test and the paired t-test; onsets only for areas reliable in both.

**Presence vs identity.** The N = {N_MAIN} onsets of the identity project (whisker vs auditory, 500 iterations × 20
within-session label shuffles) are compared per area group with the presence onsets of the same trial type (Wilcoxon
signed-rank over groups reliable in both). The nulls differ (shuffle vs linear shift), which matters for the comparison
(Caveats).

**Anatomy.** Allen Mouse Connectivity anterograde experiments (projection density, 50-µm grid) from the whisker cortices
(SSp-bfd + SSs: {num('inj_w_n', len(inj_w))} experiments, {num('inj_w_lines', inj_w.line.nunique())} lines, injection volume
median {num('inj_w_med', inj_w.injection_volume.median(), '{:.3f}')} mm³) and the auditory cortices (AUDp + AUDd/AUDv:
{num('inj_a_n', len(inj_a))} experiments, {inj_a.line.nunique()} lines, median {num('inj_a_med', inj_a.injection_volume.median(), '{:.3f}')} mm³).
Per experiment the density is normalised by its sum (shape of the projection), averaged within Cre line and then over lines,
smoothed (σ = 50 µm), with the injected area masked. The value of a unit is the density at its CCF position (folded onto
the right hemisphere); a unit inside a source area takes the projection of the other source of its modality (SSp-bfd ↔ SSs,
AUDp ↔ AUD-sec). Per area: mean over its units. Liu et al. (2024) per-structure projections of wS1 and wS2 (axon voxels
per labelled cell, ipsilateral, 6 Cre lines, 37 injections of 25 nl) are used as a second source; their injection site is
not masked, so for single areas SSp-bfd units take the wS2 value and SSs units the wS1 value. Across areas: Pearson r of
each decoding measure with log10 density (and Spearman ρ); the specificity control is the other modality's density.

**Colours.** Area groups: `ephys_utilities.allen_utils.get_custom_area_groups_colors()`; areas: shades of their group's
colour. **Code** (`exploratory-analyses/`): `001` decoding, `002` summaries and N-sweep figures, `003` matched accuracy,
`004` main figures, `006` task vs passive, `008` sample sizes and event-sequence check, `009` area rankings, `010` anatomy.
Passive provenance: {pp['events']}.
"""


def sec_results():
    lv = "area_group"
    o = ["\n# Results\n"]
    # ---- passive
    rw, ra = reliable_sorted("whisker", "passive"), reliable_sorted("auditory", "passive")
    o.append(f"""
## Passive trials: where each stimulus is first detected

**Whisker.** With N = {N_MAIN} neurons, whisker presence had an onset in every area group except
{', '.join(short(a) for a, r in C[('whisker', 'passive')]['ob'][lv].iterrows() if not np.isfinite(r.onset_ms)) or 'none'} ({ref('fig-w-passive')}). The earliest onsets were in {lst(rw, 6)}; the latest reliable ones in {lst(rw.tail(3))}
({ref('tbl-onsets')}). Unreliable: {', '.join(unrel('whisker', 'passive')) or 'none'}. Among the {len(AR.FINE)} areas
({ref('fig-s-w-passive-areas')}, {ref('tbl-s-onsets-areas')}) the earliest were {', '.join(f"{a} ({ms(r.onset_ms)} ms)" for a, r in reliable_sorted('whisker', 'passive', 'area_acronym_custom').head(8).iterrows())}.

**Auditory.** Auditory presence was decodable earlier and more uniformly ({ref('fig-a-passive')}): {lst(ra, 6)}, and
{int((ra.onset_ms <= 16).sum())} of {len(ra)} reliable groups by 16 ms; the latest were {lst(ra.tail(3))}. Unreliable:
{', '.join(unrel('auditory', 'passive')) or 'none'}. Areas ({ref('fig-s-a-passive-areas')}): {', '.join(f"{a} ({ms(r.onset_ms)} ms)" for a, r in reliable_sorted('auditory', 'passive', 'area_acronym_custom').head(8).iterrows())}.

The two stimuli differ less in where they are first detected (both: {', '.join(short(a) for a in rw.head(4).index if a in set(ra.head(6).index)) or 'different groups'})
than in how fast the information spreads: reliable whisker onsets span {num('w_span', rw.onset_ms.max() - rw.onset_ms.min(), '{:.0f}')} ms
({ms(rw.onset_ms.min())}-{ms(rw.onset_ms.max())} ms), auditory onsets {num('a_span', ra.onset_ms.max() - ra.onset_ms.min(), '{:.0f}')} ms
({ms(ra.onset_ms.min())}-{ms(ra.onset_ms.max())} ms).

{figure('fig-w-passive', fig_main('whisker', 'passive', lv), f"Whisker vs no stimulus decoding across area groups, passive trials (no stimulus: sham onsets {m1.SHAM:.1f} s before auditory stimuli), all sessions pooled, N = {N_MAIN} neurons per pseudo-population ({C[('whisker', 'passive')]['n_iter']} iterations × {m1.N_SHIFT} linear shifts; {ss_range('whisker', 'passive', lv)} sessions per group, {ref('tbl-s-sizes-groups')}). (a) Corrected balanced accuracy ({ref('eq-corr')}; 0 = chance, 0.5 = perfect) over time, causal 50-ms bins in 5-ms steps; magma, dark = low; grey: bins not above chance (5th percentile over iterations ≤ 0); tick: onset. Rows sorted by onset. (b) First 50 ms, causal 20-ms bins in 2-ms steps (bin labelled at its end). (c) Onset per group; error bars: 95 % range over {m4.N_BOOT} resamples of the iterations; † hatched: unreliable onset (95 % range > {m4.WIDE_RANGE_MS} ms), not ranked. (d, e) Time courses of the 8 best-sampled groups, mean ± s.d. over iterations, -200..600 ms and the first 50 ms; bars above: bins above chance; black tick: onset. Colours: area-group palette.")}

{figure('fig-a-passive', fig_main('auditory', 'passive', lv), f"As {ANCHORS['fig-w-passive']}, auditory vs no stimulus (no stimulus: sham onsets {m1.SHAM:.1f} s before whisker stimuli), passive trials ({ss_range('auditory', 'passive', lv)} sessions per group).")}

{mdtable('tbl-onsets', onset_table(lv), f"Onsets (ms, [95 % range over resamples of the iterations]; † unreliable) and early corrected accuracy ({ref('eq-early')}) at N = {N_MAIN}, area groups, per target and trial type; sorted by the whisker passive onset. {C[('whisker', 'active')]['n_iter']} iterations × {m1.N_SHIFT} shifts.")}
""")
    # ---- task
    rwt, rat = reliable_sorted("whisker", "active"), reliable_sorted("auditory", "active")
    pk = {}
    for t in TARGETS:
        q = X[t]["rank"]
        pk[t] = q[q.level == lv].pivot(index="area", columns="epoch", values="peak").corr(method="spearman").loc["passive", "active"]

    def top(t, e, k=4):
        q = X[t]["rank"]
        q = q[(q.epoch == e) & (q.level == lv)].sort_values("peak", ascending=False).head(k)
        return ", ".join(f"{short(r.area)} ({r.peak:.2f})" for _, r in q.iterrows())
    o.append(f"""
## Task trials: the same order

On task trials (no stimulus: correct rejections), whisker presence was decodable first in {lst(rwt, 6)}, last in
{lst(rwt.tail(3))} ({ref('fig-w-task')}); auditory presence first in {lst(rat, 6)}, last in {lst(rat.tail(3))}
({ref('fig-a-task')}). Unreliable: whisker {', '.join(unrel('whisker', 'active')) or 'none'}; auditory
{', '.join(unrel('auditory', 'active')) or 'none'}. Areas: {ref('fig-s-w-task-areas')} and {ref('fig-s-a-task-areas')}.

The best decoders by peak accuracy were {top('whisker', 'active')} for whisker and {top('auditory', 'active')} for auditory
presence on task trials; the group ranking of peaks was similar on passive trials (Spearman ρ between trial types: whisker
{num('rho_peak_w', pk['whisker'], '{:.2f}')}, auditory {num('rho_peak_a', pk['auditory'], '{:.2f}')}). Per-area rankings by
peak and by accuracy at onset are in the result folders (appendix).

{figure('fig-w-task', fig_main('whisker', 'active', lv), f"As {ANCHORS['fig-w-passive']}, whisker vs no stimulus on task trials (no stimulus: correct rejections; stimulus trials licked or not), {ss_range('whisker', 'active', lv)} sessions per group.")}

{figure('fig-a-task', fig_main('auditory', 'active', lv), f"As {ANCHORS['fig-w-passive']}, auditory vs no stimulus on task trials (no stimulus: correct rejections), {ss_range('auditory', 'active', lv)} sessions per group.")}
""")
    # ---- onset vs information
    ow, ob8 = ovw("whisker", "active"), ovw("whisker", "active", lv, "best8_all_N")
    oa = ovw("auditory", "active")
    owa = ovw("whisker", "active", lv, "all_areas_all_N")
    wa = C[("whisker", "active")]["w"]
    acc = lambda a, n: wa[(wa.level == lv) & (wa.area == a) & (wa.N == n)]["mean"].iloc[0]
    sw = "Somatosensory-whisker"
    o.append(f"""
## Onset follows the amount of early information

Early accuracy grew with the number of neurons (e.g. whisker presence in {short(sw)}: {num('acc_sw_20', acc(sw, 20))} at
N = 20 and {num('acc_sw_500', acc(sw, 500))} at N = 500, task trials; {ref('fig-s-n')}), and onsets became earlier as N grew.

Onset decreased with early accuracy ({ref('eq-early')}) along a decreasing exponential ({ref('eq-exp')}) rather than a line
({ref('fig-onset-acc')}). Whisker presence, task trials, every area group at N = {N_MAIN}: ρ = {num('rho_w_n200', ow.rho, '{:.2f}')},
exponential R² = {num('r2e_w_n200', ow.r2_exp, '{:.2f}')}, linear R² = {num('r2l_w_n200', ow.r2_ols, '{:.2f}')}; the 8
best-sampled groups across N: ρ = {num('rho_w_b8', ob8.rho, '{:.2f}')}, exponential R² = {num('r2e_w_b8', ob8.r2_exp, '{:.2f}')};
every group × N: exponential R² = {owa.r2_exp:.2f} vs linear {owa.r2_ols:.2f}. Auditory presence
({ref('fig-s-onset-acc-aud')}): ρ = {num('rho_a_n200', oa.rho, '{:.2f}')}, exponential R² = {num('r2e_a_n200', oa.r2_exp, '{:.2f}')}
vs linear {num('r2l_a_n200', oa.r2_ols, '{:.2f}')}.

The onset of a region is therefore partly a read-out of how much early information it carries: strongly decodable regions
saturate near the earliest onsets, weakly decodable ones have late and variable onsets.

{figure('fig-onset-acc', C[('whisker', 'active')]['folder'] / 'figures' / 'onset_vs_window.png', f"Onset vs early accuracy, whisker presence, task trials ({C[('whisker', 'active')]['n_iter']} iterations × {m1.N_SHIFT} shifts). (a) Every area group at N = {N_MAIN}; (b) the 8 best-sampled groups at every N (dot size: N = 20-500; lines join a group's values); (c, d) the same for areas. Black line: least-squares decreasing exponential ({ref('eq-exp')}), solid when Spearman p < 0.05; grey band: 95 % range of the fit over 500 bootstrap resamples of the points. Text: Spearman ρ, p, n points; R² of the exponential and of a straight line; ΔAIC (exponential - linear). Onset: 20-ms bins, 2-ms steps. Colours: area palette.")}
""")
    if MATCHED:
        o.append(sec_matched())
    else:
        o.append(f"""
**Matched accuracy (pending).** Decoding every group at the N that matches the early accuracy of a reference group
separates onset from information directly; its {sum(len((C[(t, 'active')]['folder'] / 'matched_runs.txt').read_text().splitlines()) for t in TARGETS)}
task-trial runs were still running when this report was built, and the section will be added when they finish.
""")
    # ---- task vs passive
    t_on, t_acc, t_b = (tst("whisker", lv, m) for m in ("onset_ms", "accuracy_5_50ms", "baseline_-150_0ms"))
    a_on, a_acc, a_b = (tst("auditory", lv, m) for m in ("onset_ms", "accuracy_5_50ms", "baseline_-150_0ms"))
    af_on = tst("auditory", "area_acronym_custom", "onset_ms")
    avp = rd(HOME / "whisker_task_vs_passive" / "tables" / "active_vs_passive_area_group.csv").set_index("area")
    base_top = avp.base_passive.sort_values(ascending=False).head(2)
    seqp = X["whisker"]["seq"]
    seqp = seqp[seqp.epoch == "passive"]
    o.append(f"""
## Task vs passive trials

**Onsets are the same.** Whisker presence, passive - task, area groups: median {t_on.median_diff:+.0f} ms (n = {int(t_on.n)},
Wilcoxon p {pv('p_w_on2', t_on.p_wilcoxon)}); auditory: median {a_on.median_diff:+.0f} ms (n = {int(a_on.n)}, p {pv('p_a_on', a_on.p_wilcoxon)})
({ref('fig-avp')}, {ref('fig-s-avp-aud')}, {ref('tbl-tests')}). For single areas, auditory onsets were marginally earlier
on passive trials (mean {af_on.mean_diff:+.1f} ms, median {af_on.median_diff:+.0f}, n = {int(af_on.n)}, p {pv('p_a_on_fine', af_on.p_wilcoxon)}),
a difference below the 2-ms bin step.

**Early accuracy.** Whisker early accuracy was lower on passive trials (median {t_acc.median_diff:+.3f}, p {pv('p_w_acc', t_acc.p_wilcoxon)});
auditory early accuracy did not differ ({a_acc.median_diff:+.3f}, p {pv('p_a_acc', a_acc.p_wilcoxon)}).

**Later in the trial**, task decoding stayed high while passive decoding decayed ({ref('fig-avp')}e). On task
trials the stimulus class includes licked trials and the no-stimulus class does not, so late decoding reflects licking and
reward as well as the stimulus.

**Pre-stimulus offset.** Passive pre-stimulus accuracy was above the task-trial value (whisker {t_b.median_diff:+.3f},
p {pv('p_w_b2', t_b.p_wilcoxon)}; auditory {a_b.median_diff:+.3f}, p {pv('p_a_b2', a_b.p_wilcoxon)}; n = {int(t_b.n)} groups),
largest for whisker presence in {', '.join(f"{short(a)} ({v:+.3f})" for a, v in base_top.items())}. A candidate cause is the fixed passive timing: a stimulus window starts at a
predictable time after the previous stimulus, a sham window 0.8 s earlier, so slow post-stimulus activity distinguishes
them. The event sequences themselves are close to independent (passive, whisker target: median P(repeat)
{num('seq_rep', seqp.p_repeat.median(), '{:.3f}')} vs {num('seq_rep_ind', seqp.p_repeat_independent.median(), '{:.3f}')} for an
independent sequence, n = {num('seq_n', len(seqp))} sessions; {ref('tbl-s-seq')}).

{figure('fig-avp', X['whisker']['fig'] / 'active_vs_passive_area_group.png', f"Task vs passive trials, whisker presence, area groups, N = {N_MAIN} ({C[('whisker', 'active')]['n_iter']} iterations × {m1.N_SHIFT} shifts each). (a) Onset per group (filled: task, open: passive), 95 % ranges over resamples of the iterations; grey: unreliable in either trial type, not tested. (b) Early corrected accuracy ({ref('eq-early')}), 2.5-97.5 percentiles over iterations. (c) Pre-stimulus corrected accuracy (wide bins ending -150..0 ms), mean ± 1.96 s.e.m. over iterations; expected 0. Rows sorted by the task onset. Under each axis: passive - task median, Wilcoxon signed-rank and paired t-test over groups. (d, e) First 50 ms and -200..600 ms time courses of the 6 best-sampled groups, task solid, passive dashed, mean ± s.d. over iterations.")}

{mdtable('tbl-tests', tests_table(), "Passive - task differences over regions (paired), per target and level: onset (reliable in both trial types only), early accuracy (Eq. (2)) and pre-stimulus corrected accuracy (-150..0 ms).")}
""")
    # ---- identity
    cmp = {(t, e): ident_cmp(t, e) for t in TARGETS for e in EPOCHS}
    o.append(f"""
## Presence vs identity onsets

Whisker-presence onsets were later than whisker vs auditory (identity) onsets in the same area groups: median
{cmp[('whisker', 'passive')]['med']:+.0f} ms on passive trials (Wilcoxon p {pv('p_id_wp', cmp[('whisker', 'passive')]['p'])},
n = {cmp[('whisker', 'passive')]['n']}) and {cmp[('whisker', 'active')]['med']:+.0f} ms on task trials (p {pv('p_id_wt', cmp[('whisker', 'active')]['p'])},
n = {cmp[('whisker', 'active')]['n']}). Auditory-presence onsets were close to the identity onsets ({cmp[('auditory', 'passive')]['med']:+.0f} ms,
p {pv('p_id_ap', cmp[('auditory', 'passive')]['p'])}; task {cmp[('auditory', 'active')]['med']:+.0f} ms, p {pv('p_id_at', cmp[('auditory', 'active')]['p'])})
({ref('tbl-ident')}). The group order was preserved (Spearman ρ, presence vs identity: whisker passive
{num('rho_id_wp', cmp[('whisker', 'passive')]['rho'], '{:.2f}')}, task {num('rho_id_wt', cmp[('whisker', 'active')]['rho'], '{:.2f}')}).

Taken at face value, the earliest identity information in most groups is carried by the auditory response. The two
analyses use different nulls, however (within-session shuffle for identity, linear shift for presence; the shift null is
the more conservative), so a few ms of the difference can come from the null alone.

{mdtable('tbl-ident', ident_table(), "Onsets (ms) at N = 200 per area group: identity (whisker vs auditory, 500 iterations × 20 shuffles, companion project) and presence (whisker or auditory vs no stimulus, 100 iterations × 10 linear shifts), passive and task trials. † unreliable.")}
""")
    # ---- anatomy
    rows = {(t, e, s): ana(t, e, lv, s) for t in TARGETS for e in EPOCHS for s in ("whisker", "auditory", "liu")}
    q = ANA_BY[(ANA_BY.level == lv) & (ANA_BY.dens_whisker_incl > 0) & (ANA_BY.dens_auditory_incl > 0)]
    dens_r = stats.pearsonr(np.log10(q.dens_whisker_incl), np.log10(q.dens_auditory_incl))
    o.append(f"""
## Anatomical input from the stimulus' cortex

Across area groups, early accuracy rose with the Allen projection density from the stimulus' own cortex at the recorded
units ({ref('fig-anatomy')}, {ref('tbl-anatomy')}): whisker presence, task trials r = {rows[('whisker', 'active', 'whisker')].pearson_r_logdens:+.2f}
(p {pv('p_ana_wt', rows[('whisker', 'active', 'whisker')].pearson_p)}), passive r = {rows[('whisker', 'passive', 'whisker')].pearson_r_logdens:+.2f};
auditory presence, task r = {rows[('auditory', 'active', 'auditory')].pearson_r_logdens:+.2f} (p {pv('p_ana_at', rows[('auditory', 'active', 'auditory')].pearson_p)}).
Onset and peak accuracy followed the same direction with weaker, mostly non-significant correlations ({ref('fig-anatomy')}).

The relation is not specific. Whisker early accuracy correlated at least as strongly with the auditory-cortex density
(task r = {rows[('whisker', 'active', 'auditory')].pearson_r_logdens:+.2f}, p {pv('p_ana_wt_oth', rows[('whisker', 'active', 'auditory')].pearson_p)}),
and the two densities are correlated across groups (r = {num('dens_r', dens_r.statistic, '{:.2f}')}, log scale): regions that
receive dense cortical sensory input receive it from both cortices. With the Liu et al. projections, the whisker relation
was weaker (task r = {rows[('whisker', 'active', 'liu')].pearson_r_logdens:+.2f}, p {pv('p_ana_liu', rows[('whisker', 'active', 'liu')].pearson_p)})
and absent for single areas ({ref('fig-s-anatomy-liu')}); the per-structure Liu values give one number to whole structures
such as the caudoputamen, which removes the gradients the Allen voxels keep. Single areas: {ref('fig-s-anatomy-areas')}.

As a sanity check the anatomy is consistent (regions with more cortical sensory input decode better), but it does not
single out the input from the stimulus' own cortex.

{figure('fig-anatomy', ANAFIG / 'anatomy_allen_vs_decoding_area_group.png', f"Stimulus-presence decoding vs Allen projection density from the stimulus' own cortex (whisker rows: SSp-bfd + SSs; auditory rows: AUDp + AUDd/AUDv), area groups, N = {N_MAIN}. Rows: target × trial type; columns: early accuracy ({ref('eq-early')}), peak accuracy, onset (axis inverted). x: mean density at the units' CCF positions (log scale); a unit inside a source area takes the other source's projection. Line: least-squares fit of the measure on log10 density with 95 % confidence band, solid when Pearson p < 0.05; title: Pearson r (log density), Spearman ρ, n groups. Open circles: unreliable onsets, not tested. Black ring: injection site (the source area group). Grouped injection sizes under the figure. Colours and labels: area-group palette.")}

{mdtable('tbl-anatomy', anatomy_table(), "Early accuracy (Eq. (2)) vs log10 projection density across regions: Pearson r (p); Spearman ρ. Own cortex: the decoded stimulus' cortex (Allen); other cortex: the other modality (specificity control); Liu: wS1 + wS2 ipsilateral per-structure projections (whisker cortex only; for single areas the injection site takes the other site's value). n = regions with an Allen value.")}
""")
    return "\n".join(o)


def sec_matched():
    lines = []
    for t in TARGETS:
        mn = C[(t, "active")]["matched_n"]
        mn = mn[mn.level == "area_group"]
        nr = sorted(set(mn[mn.how == "not reached"].area))
        lines.append(f"{t}: {', '.join(short(a) for a in nr) or 'none'} could not reach at least one reference within N ≤ 2000")
    return f"""
To separate the two, early accuracy was matched to a reference group at N = 100 ({ref('fig-matched')}, {ref('tbl-matched')};
references: Ss-whisker, Midbrain, Auditory). {'; '.join(lines)}. Onsets of the groups that reached the reference's accuracy
(onset of the averaged time course, 20-ms bins, 2-ms steps): {'; '.join(f"{d['t']}, reference {d['ref']}: {d['lo']:.0f}-{d['hi']:.0f} ms over {d['n']} groups (earliest {d['first']}, latest {d['last']}; the same groups at N = 100: {d['lo100']:.0f}-{d['hi100']:.0f} ms)" for d in matched_spread())}.
{matched_verdict()}

{figure('fig-matched', C[('whisker', 'active')]['folder'] / 'figures' / 'matched_accuracy.png', "Matched-accuracy control, whisker presence, task trials, area groups. Each row: one reference group decoded with 100 neurons; every other group decoded at the N that gives the same early corrected accuracy (Eq. (2)), read from its accuracy-vs-N curve (log-N interpolation; * extrapolated beyond 500; groups not reaching the target within 2000 neurons listed as > 2000). Left: time courses at the matched N, 50-ms bins, mean ± s.d. over iterations; middle: first 100 ms, 20-ms bins, shaded: 5-50 ms; right: matched N per group, log scale; dotted: 100 neurons. 100 iterations × 10 shifts.")}

{mdtable('tbl-matched', matched_table(), "Matched-accuracy control, task trials, area groups: target = the reference's early accuracy at N = 100; achieved = the group's accuracy at its matched N.")}
"""


def sec_discussion():
    rw, ra = reliable_sorted("whisker", "passive"), reliable_sorted("auditory", "passive")
    return f"""
# Discussion

Stimulus presence is detected first in {', '.join(short(a) for a in rw.head(4).index)} for whisker and in
{', '.join(short(a) for a in ra.head(4).index)} for auditory stimuli, within {ms(max(rw.onset_ms.iloc[3], ra.onset_ms.iloc[3]))} ms
(passive trials). The two stimuli differ in what follows: auditory presence becomes decodable in almost every recorded
group within a few ms, whereas whisker presence spreads over tens of ms from sensory to striatal, motor, hippocampal and
frontal groups. Fast, widespread auditory responses are consistent with tones driving arousal and
orienting circuits brain-wide, while the whisker signal follows the slower cortico-striatal and cortico-cortical routes.

As for identity decoding, onset and information are entangled: the exponential relation between onset and early accuracy
means that a late region is often a region with little early information. A population onset is the first time a linear
read-out of N neurons separates the classes reliably and depends on the bin width (20 ms, causal) and on the criterion.

Engagement does not change when the stimuli are detected: task and passive onsets agree. It changes what follows (sustained
decoding on task trials, where licking and reward follow) and, for whisker, the early accuracy.

The anatomy agrees with the decoding in the coarse sense (regions with denser cortical sensory input decode better) but not
in the specific sense (input from the stimulus' own cortex is not a better predictor than input from the other cortex).
Much of the early presence signal may come from subcortical routes (collicular and thalamic) that the cortical injections
do not capture.
"""


def sec_caveats():
    return f"""
# Caveats and limitations

- **Passive no-stimulus events are shams.** Passive blocks have almost no no-stimulus trials; shams 0.8 s before the other
  modality's stimulus sit at a predictable time in the fixed passive sequence, which can explain the small passive
  pre-stimulus offset. Task no-stimulus events are correct rejections, which differ from stimulus trials also in not being
  followed by a lick.
- **Late decoding on task trials** includes licking and reward, because licked stimulus trials are compared with unlicked
  correct rejections. Only the early window and the onset are stimulus-specific.
- **Whisker artefact.** Spikes from -10 to +5 ms around every event onset are replaced by Poisson spikes, so information can
  appear only from +5 ms; with causal 20-ms bins an onset at $t$ means information somewhere in $[t-20, t)$ ms.
- **Nulls differ from the identity project** (linear shift vs shuffle), which limits the presence-vs-identity comparison.
- **Pooling.** Cohorts (R+, R-) and stages (learning, expert) are pooled; differences between them are not tested.
- **Iterations.** 100 iterations × {m1.N_SHIFT} shifts for every analysis (final for this analysis); onset ranges are
  95 % ranges over resamples of these iterations.
- **Anatomy.** Cortical injections only; correlations across 18 groups (40 areas); the whisker and auditory densities are
  correlated with each other; the Liu per-structure values are coarse. A voxel-level re-analysis of the Liu data is a
  separate project (on hold).

# Open questions and next steps

1. Split passive trials into pre- and post-task blocks, and model the time since the previous stimulus, to locate the
   passive pre-stimulus offset.
2. Cohort and stage splits (R+ vs R-, learning vs expert) of presence onsets and early accuracy.
3. Decoders trained on stimulus presence applied to continuous data to detect reactivation events.
"""


def sec_supp():
    o = ["\n# Supplementary figures\n"]
    for (t, e), an in ((("whisker", "passive"), "fig-s-w-passive-areas"), (("auditory", "passive"), "fig-s-a-passive-areas"),
                       (("whisker", "active"), "fig-s-w-task-areas"), (("auditory", "active"), "fig-s-a-task-areas")):
        main = {"fig-s-w-passive-areas": "fig-w-passive", "fig-s-a-passive-areas": "fig-a-passive",
                "fig-s-w-task-areas": "fig-w-task", "fig-s-a-task-areas": "fig-a-task"}[an]
        o.append(figure(an, fig_main(t, e, "area_acronym_custom"),
                        f"As {ANCHORS[main]}, the {len(AR.FINE)} best-sampled areas ({ss_range(t, e, 'area_acronym_custom')} sessions per area, {ref('tbl-s-sizes-areas')}). Colours: shades of the area's group colour.",
                        kind="supplementary"))
    o.append(figure("fig-s-onset-acc-aud", C[("auditory", "active")]["folder"] / "figures" / "onset_vs_window.png",
                    f"As {ANCHORS['fig-onset-acc']}, auditory presence, task trials.", kind="supplementary"))
    o.append(figure("fig-s-n", C[("whisker", "active")]["folder"] / "figures" / "accuracy_onset_vs_N.png",
                    "Effect of the number of neurons, whisker presence, task trials. (a, c) Early corrected accuracy (Eq. (2)) vs N, area groups and areas, 95 % range over iterations (100 × 10); (b, d) onset vs N; missing points: no onset at that N. Colours: area palette. The other conditions are in the result folders.",
                    kind="supplementary"))
    o.append(figure("fig-s-avp-aud", X["auditory"]["fig"] / "active_vs_passive_area_group.png",
                    f"As {ANCHORS['fig-avp']}, auditory presence.", kind="supplementary"))
    o.append(figure("fig-s-anatomy-areas", ANAFIG / "anatomy_allen_vs_decoding_area_acronym_custom.png",
                    f"As {ANCHORS['fig-anatomy']}, the {len(AR.FINE)} areas; legend: area-group colours (areas in shades); rings: SSp-bfd and SSs (whisker rows), AUD (auditory rows).",
                    kind="supplementary"))
    o.append(figure("fig-s-anatomy-liu", ANAFIG / "anatomy_liu_vs_decoding_area_group.png",
                    f"As {ANCHORS['fig-anatomy']}, with the Liu et al. (2024) wS1 + wS2 ipsilateral per-structure projections (axon voxels per labelled cell; 37 injections of 25 nl, 6 Cre lines) for both targets (auditory rows: specificity control). Each unit takes the value of the deepest Liu target structure containing it; areas with < 50 % of units mapped are omitted.",
                    kind="supplementary"))
    return "\n".join(o)


def sec_supp_tables():
    rows = []
    for t in TARGETS:
        for e in EPOCHS:
            q = X[t]["seq"]
            q = q[q.epoch == e]
            d = (q.p_stim_first_half - q.p_stim_second_half).abs()
            rows.append({"Target": t, "Trials": EW[e], "Sessions": len(q), "Events (median)": f"{q.n_events.median():.0f}",
                         "P(stimulus)": f"{q.p_stim.median():.3f}", "P(repeat)": f"{q.p_repeat.median():.3f}",
                         "P(repeat), independent": f"{q.p_repeat_independent.median():.3f}",
                         "|Δ P(stimulus)|, halves": f"{d.median():.3f}"})
    return "\n# Supplementary tables\n" + "\n".join([
        mdtable("tbl-s-sizes-groups", sizes_table("area_group"), f"Sample sizes per area group, target (W whisker, A auditory) and trial type: eligible sessions (≥ {m1.MIN_UNITS} good + mua units of the group and ≥ {m1.MIN_TRIALS} events per class) / their mice, and stimulus / no-stimulus events (sum over eligible sessions)."),
        mdtable("tbl-s-sizes-areas", sizes_table("area_acronym_custom"), f"As {ANCHORS['tbl-s-sizes-groups']}, the {len(AR.FINE)} areas."),
        mdtable("tbl-s-onsets-areas", onset_table("area_acronym_custom"), f"As {ANCHORS['tbl-onsets']}, the {len(AR.FINE)} areas."),
        mdtable("tbl-s-seq", pd.DataFrame(rows), "Event sequences per target and trial type (medians over sessions): P(stimulus) = fraction of stimulus events; P(repeat) = probability that an event has the class of the previous one, and its value for an independent sequence with the session's P(stimulus); |Δ P(stimulus)|: difference of P(stimulus) between the first and second half of the session's events.")])


def sec_appendix():
    p = C[("whisker", "active")]["prov"]
    par = pd.DataFrame([
        ("Sessions per draw", m1.N_SESS), ("Min units / events per class", f"{m1.MIN_UNITS} / {m1.MIN_TRIALS}"),
        ("Pseudo-trials per class (train / test)", f"{p['t_train']} / {p['t_test']}"),
        ("Outer / inner CV folds", f"{p['n_outer']} / {p['n_inner']}"), ("C grid", ", ".join(f"{c:g}" for c in m1.C_GRID)),
        ("N sweep", ", ".join(map(str, m1.N_LIST))), ("Zoom bins", p["zoom"]), ("Wide bins", p["wide"]),
        ("Null", p["null"]), ("Iterations × shifts", f"{C[('whisker', 'active')]['n_iter']} × {m1.N_SHIFT} (all N)"),
        ("Early window", "5-50 ms (zoom bin ends)"), ("Baseline window", "wide bins ending -150..0 ms"),
        ("Above chance", "5th percentile over iterations > 0"), ("Onset", "≥ 80 % of the next 25 ms above chance (11 / 13)"),
        ("Onset resamples", m4.N_BOOT), ("Unreliable onset", f"95 % range > {m4.WIDE_RANGE_MS} ms or undefined in > 5 %"),
        ("Passive sham", f"{m1.SHAM:.1f} s before the other modality's stimulus; lick-free {m1.LICK_WIN}"),
        ("Artefact window", "-10..+5 ms at every event onset, Poisson replacement")], columns=["Parameter", "Value"])
    files = pd.DataFrame([
        ("<target>_<epoch>/", "per target (whisker, auditory) and trial type (active = task, passive): decoding (raw/), summaries, figures, onset bootstrap, matched accuracy"),
        ("<target>_task_vs_passive/", "task vs passive figures and tests, area rankings (peak, accuracy at onset), sample sizes, event-sequence check"),
        ("anatomy/", "anatomy per area, tests, Allen and Liu figures for both levels"),
        ("report/", "this report; _snapshots/ = frozen statistics tables")], columns=["Results folder", "Content"])
    return f"""
# Appendix

{mdtable('tbl-a-params', par, "Parameters.")}

{mdtable('tbl-a-files', files, "Result folders (combined_results_ks4/ssl-stimulus-detection-decoding/ on the NAS). Figures not shown in the report (summary figures, per-N curves of every area, area rankings, the other conditions' N effect) are in each folder's figures/.")}

**Version history.** 2026-10-16: project created (presence decoding, linear-shift null, passive shams); N = 200 runs and
N sweep for both targets and trial types (finished 2026-10-20). 2026-10-23: 100 iterations × 10 shifts declared final;
anatomy figures with regression bands, injection sites and grouped injection sizes; Liu own-site leave-out for single
areas; first report. Parameters and folders: {ref('tbl-a-params')}, {ref('tbl-a-files')}; claims: {ref('tbl-a-registry')}.

{R_.registry_table('tbl-a-registry')}
"""


def summary_md():
    rw, ra = reliable_sorted("whisker", "passive"), reliable_sorted("auditory", "passive")
    return f"""# Stimulus-presence decoding: summary ({date.today().isoformat()})

- Whisker presence (passive, N = {N_MAIN}): first {lst(rw, 4)}; spreads to striatal, motor, hippocampal and frontal groups over 20-30 ms.
- Auditory presence: almost every group by {ra.onset_ms.quantile(.75):.0f} ms; first {lst(ra, 4)}.
- Onset tracks early information (decreasing exponential); task and passive onsets agree; passive pre-stimulus offset is small but significant.
- Presence vs identity: whisker-presence onsets later than identity onsets (different nulls).
- Anatomy: early accuracy rises with cortical sensory input, but not specifically from the stimulus' own cortex.

![](figures/fig-w-passive.png)
"""


def main():
    md = (sec_front() + sec_intro() + sec_methods() + sec_results() + sec_discussion() + sec_caveats() + sec_supp()
          + sec_supp_tables() + sec_appendix())
    R_.write(OUT, md, summary=summary_md())
    R_.snapshot(sorted(set(READ)), OUT, tag=SNAP)          # one manifest listing every frozen table
    print("matched:", MATCHED)


if __name__ == "__main__":
    main()
