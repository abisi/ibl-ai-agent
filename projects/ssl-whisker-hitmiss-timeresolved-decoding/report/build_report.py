"""Article-style report of the within-session (split) decoding work (skills/project-report).
Reads the result tables, writes numbers.json and report.md (single source: Pandoc Markdown with LaTeX math), copies every
figure used into the self-contained report folder combined_results_ks4/<slug>/report/, and writes build.sh there.
No number in the text is typed by hand: every quoted value comes from a table through num().
Run (haas, repo root):  python projects/ssl-whisker-hitmiss-timeresolved-decoding/report/build_report.py
Render (where Quarto / TinyTeX are, e.g. locally):  bash render.sh   (or bash <report folder>/build.sh)
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

SLUG = PROJ.name
EA = PROJ / "exploratory-analyses"
FIG = EA / "figures"
PP = PROJ.parent / "ssl-pseudopopulation-area-decoding"
OUT = axel_bisi_root() / "combined_results_ks4" / SLUG / "report"
NUM: dict[str, float | int | str] = {}
FIGS: list[Path] = []


# ---------------------------------------------------------------------------------------------------------- helpers
def num(key: str, value, fmt: str = "{:.3f}") -> str:
    """record a quoted number in numbers.json and return it formatted"""
    if isinstance(value, (np.integer, int)):
        NUM[key] = int(value)
        return str(int(value))
    v = float(value)
    NUM[key] = None if np.isnan(v) else v
    return "n/a" if np.isnan(v) else fmt.format(v)


def pv(key: str, value) -> str:
    v = float(value)
    NUM[key] = v
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def row(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    r = df[m]
    if len(r) != 1:
        raise ValueError(f"{len(r)} rows for {kw}")
    return r.iloc[0]


def figure(src: Path, label: str, caption: str) -> str:
    FIGS.append(src)
    return f"![**{label}.** {caption}](figures/{src.name}){{width=100%}}\n"


def mdtable(df: pd.DataFrame, caption: str, pcols=(), signed=(), ints=()) -> str:
    d = df.copy()
    for c in d.columns:
        if c in pcols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else ("**<0.001**" if v < 0.001 else (f"**{v:.3f}**" if v < 0.05 else f"{v:.3f}")))
        elif c in ints:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
        elif c in signed:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:+.3f}")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    cols = [SHORT.get(c, c.replace("_", " ")) for c in d.columns]
    w = [max(len(h), *(len(str(v).replace("*", "")) for v in d[c])) for h, c in zip(cols, d.columns)]
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(("-" * (n + 1) + ":") if i else (":" + "-" * (n + 1)) for i, n in enumerate(w)) + "|\n"
    body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in d.itertuples(index=False))
    return f"\n```{{=latex}}\n\\begingroup\\footnotesize\n```\n\n{head}{body}\n: {caption}\n\n```{{=latex}}\n\\endgroup\n```\n"


SHORT = {"mean_delta": "delta", "mean_excess": "excess", "mean_percentile": "pct", "p_wilcoxon": "p W", "p_t": "p t",
         "p_paired_t": "p t", "p_mw_cohorts": "p MW", "p_welch_cohorts": "p Welch", "p_cohort_mannwhitney": "p MW",
         "p_cohort_welch": "p Welch", "diff_of_diff": "diff-diff", "p_perm": "p perm", "n_rplus": "n R+", "n_rminus": "n R-",
         "mean_rplus": "R+", "mean_rminus": "R-", "half_pct_mean": "half pct", "p_half_pct_wilcoxon": "p W",
         "p_half_pct_t": "p t", "best_pos_median": "best pos", "spearman_best_cp": "rho(best, CP)", "p_perm_best_cp": "p perm",
         "frac_sig": "frac sig", "n_sig": "n sig", "p_fisher_frac_sig": "p Fisher", "p_perm_all": "p perm all",
         "p_perm_learners": "p perm learners", "p_spearman": "p S", "p_pearson": "p P", "p_perm_spearman": "p perm",
         "definition": "split", "variant": "split"}


# ---------------------------------------------------------------------------------------------------------- tables
T117 = csv(EA / "117b_halves_matched_full_stats.csv")
T016 = csv(PP / "artifacts" / "016_pseudopop_halves_stats.csv")
T119 = csv(EA / "119_lt_definitions_stats_A1v2.csv")
T120 = csv(EA / "120_lt_defined_vs_undefined_stats_A1v2.csv")
T126 = csv(EA / "126_lt_variants_stats_step1_pl100.csv")
T131 = csv(EA / "131_cohort_permutation.csv")
T129 = csv(EA / "129_stats.csv")
T124 = csv(EA / "124_correlations.csv")
T127 = csv(EA / "127_stats.csv")
T130 = csv(EA / "130_stats.csv")
T135 = csv(EA / "135b_stats.csv")
T133 = csv(EA / "133_stats_good.csv")
VAR = ["half", "L0 stored", "L5 whisker CP", "L6 joint CP", "L6 lenient", "L6x forced", "L6x relaxed", "L8 fixed margin"]
# Part I (session-wide): 113 = publication sets (learning stage; hitmiss, modality_stim, perfstate); 110 = rerun of the
# pre-lick modality analysis with the -100..0 ms window (2026-10-02), which supersedes 113's modality_lick rows.
T113 = csv(EA / "113_stats.csv")
T110 = csv(EA / "110_publication_stats.csv")
WB = [("hitmiss", T113, "sensory"), ("hitmiss", T113, "baseline"), ("modality_stim", T113, "sensory"),
      ("modality_lick", T110, "pre-lick"), ("modality_lick", T110, "post-lick"), ("perfstate", T113, "sensory"),
      ("perfstate", T113, "baseline")]
LABEL = {"hitmiss": "hit vs miss", "modality_stim": "whisker vs auditory (stimulus)", "modality_lick": "whisker vs auditory (pre-lick)",
         "perfstate": "performance state"}


def pick(df: pd.DataFrame, **kw) -> pd.Series:
    """the single group-level row (session_id empty) matching kw, whole brain, learning stage unless given"""
    kw = {"level": "whole_brain", **kw}
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    if "session_id" in df:
        m &= df["session_id"].isna().to_numpy()
    r = df[m]
    if len(r) == 0:
        raise ValueError(f"no row for {kw}")
    return r.iloc[0]


def clusters(df: pd.DataFrame, analysis: str, cohort: str, test: str, condition: str = "whole") -> pd.DataFrame:
    m = ((df.level == "whole_brain") & (df.stage == "learning") & (df.window == "time-resolved") & (df.analysis == analysis)
         & (df.cohort == cohort) & (df.test == test) & (df.condition == condition) & (df.p < 0.05))
    if "session_id" in df:
        m &= df.session_id.isna()
    return df[m].sort_values("t_start_ms")


def wb_table() -> pd.DataFrame:
    rows = []
    for an, df, w in WB:
        for c in ("R+", "R-"):
            r = pick(df, analysis=an, stage="learning", window=w, condition="whole", cohort=c,
                     test="accuracy - null vs 0 (Wilcoxon | one-sample t)")
            rows.append(dict(decoding=LABEL[an], window=w, group=c, n=r.n, value=r["mean"], p_np=r.p_nonparam, p_par=r.p_param))
        r = pick(df, analysis=an, stage="learning", window=w, condition="whole", cohort="R+ vs R-", test="Mann-Whitney | Welch")
        rows.append(dict(decoding=LABEL[an], window=w, group="R+ vs R-", n=r.n, value=r.mean_a - r.mean_b, p_np=r.p_nonparam, p_par=r.p_param))
    t = pd.DataFrame(rows)
    t.columns = ["decoding", "window", "group", "n", "acc - null (R+ - R- for the test row)", "p W / MW", "p t / Welch"]
    return t


def sec_part1() -> str:
    P = ("p W / MW", "p t / Welch")
    hs = {c: pick(T113, analysis="hitmiss", window="sensory", condition="whole", cohort=c,
                  test="accuracy - null vs 0 (Wilcoxon | one-sample t)") for c in ("R+", "R-")}
    hb = {c: pick(T113, analysis="hitmiss", window="baseline", condition="whole", cohort=c,
                  test="accuracy - null vs 0 (Wilcoxon | one-sample t)") for c in ("R+", "R-")}
    hc = pick(T113, analysis="hitmiss", window="sensory", condition="whole", cohort="R+ vs R-", test="Mann-Whitney | Welch")
    hcl = clusters(T113, "hitmiss", "R+ - R-", "two-sided cluster vs mouse-level cohort-label permutation")
    hd = pick(T113, analysis="hitmiss", window="sensory", condition="whole", cohort="R-",
              test="correlation with behavioural d' (Pearson | Spearman)")
    hdp = pick(T113, analysis="hitmiss", window="sensory", condition="whole", cohort="R+",
               test="correlation with behavioural d' (Pearson | Spearman)")
    ms = {c: pick(T113, analysis="modality_stim", window="sensory", condition="whole", cohort=c,
                  test="accuracy - null vs 0 (Wilcoxon | one-sample t)") for c in ("R+", "R-")}
    mcl = clusters(T113, "modality_stim", "R+ - R-", "two-sided cluster vs mouse-level cohort-label permutation")
    mon = {c: clusters(T113, "modality_stim", c, "cluster vs group mouse-block null") for c in ("R+", "R-")}
    ml = {c: pick(T110, analysis="modality_lick", window="pre-lick", condition="whole", cohort=c,
                  test="accuracy - null vs 0 (Wilcoxon | one-sample t)") for c in ("R+", "R-")}
    mlc = pick(T110, analysis="modality_lick", window="pre-lick", condition="whole", cohort="R+ vs R-", test="Mann-Whitney | Welch")
    mpc = pick(T110, analysis="modality_lick", window="post-lick", condition="whole", cohort="R+ vs R-", test="Mann-Whitney | Welch")
    mlcl = clusters(T110, "modality_lick", "R+ - R-", "two-sided cluster vs mouse-level cohort-label permutation")
    mh = {c: pick(T110, analysis="modality_lick", window="pre-lick", condition="half first vs second", cohort=c,
                  test="paired Wilcoxon | paired t") for c in ("R+", "R-")}
    lat = T110[(T110.analysis == "modality_lick") & (T110.stage == "learning") & (T110.level == "area_group")
               & (T110.test == "50%-of-peak latency, mouse bootstrap 95% CI")].pivot_table(index="area", columns="cohort", values="latency_ms")
    lat_med = {c: float(lat[c].median()) for c in ("R+", "R-")}
    pd_ = pick(T113, analysis="perfstate", window="sensory", condition="whole", cohort="R-",
               test="correlation with behavioural d' (Pearson | Spearman)")
    ps = {c: pick(T113, analysis="perfstate", window="sensory", condition="whole", cohort=c,
                  test="accuracy - null vs 0 (Wilcoxon | one-sample t)") for c in ("R+", "R-")}
    lat_t = lat.reset_index()
    lat_t.columns = ["area group", "R+ latency (ms)", "R- latency (ms)"]
    return f"""
# Results

## Part I. Session-wide decoding

### I.1 Choice (hit vs miss) is decodable in both cohorts, before and after the whisker stimulus

Over the whole learning session, whole-brain activity 5-50 ms after the whisker stimulus predicted whether the mouse would
lick, above the session's linear-shift null, in both cohorts (R+ {num('p1_hs_rp', hs['R+']['mean'], '{:+.3f}')}, n = {int(hs['R+'].n)},
Wilcoxon p {pv('p1_hs_rp_p', hs['R+'].p_nonparam)}; R- {num('p1_hs_rm', hs['R-']['mean'], '{:+.3f}')}, n = {int(hs['R-'].n)},
p {pv('p1_hs_rm_p', hs['R-'].p_nonparam)}), with no cohort difference in that window (Mann-Whitney p = {pv('p1_hc_mw', hc.p_nonparam)},
Welch p = {pv('p1_hc_w', hc.p_param)}). Choice was also decodable before the stimulus (baseline -200 to -10 ms: R+
{num('p1_hb_rp', hb['R+']['mean'], '{:+.3f}')}, R- {num('p1_hb_rm', hb['R-']['mean'], '{:+.3f}')}, both p {pv('p1_hb_p', max(hb['R+'].p_nonparam, hb['R-'].p_nonparam))}),
so part of it reflects the pre-stimulus state (engagement) rather than the evoked response. The cohort curves diverged
late, higher in {'R+' if len(hcl) and hcl['sign'].iloc[0] > 0 else 'R-'}, from {num('p1_hcl_t0', hcl.t_start_ms.iloc[0] if len(hcl) else np.nan, '{:.0f}')} ms after the stimulus (two-sided
cluster, p = {pv('p1_hcl_p', hcl.p.iloc[0] if len(hcl) else np.nan)}), when licks are under way (Figure 1). Across R- mice,
hit/miss decoding increased with behavioural d' (Pearson r = {num('p1_hd_r', hd.r_pearson, '{:+.2f}')}, p = {pv('p1_hd_p', hd.p_nonparam)};
Spearman rho = {num('p1_hd_rho', hd.rho_spearman, '{:+.2f}')}, p = {pv('p1_hd_ps', hd.p_param)}), not across R+ mice (r =
{num('p1_hdp_r', hdp.r_pearson, '{:+.2f}')}, p = {pv('p1_hdp_p', hdp.p_nonparam)}).

{figure(FIG / 'publication' / '113_hitmiss_set1_session_wide.png', 'Figure 1', 'Hit vs miss decoding, session-wide, learning stage (113 set 1). a schematic; b, c example sessions per cohort (real curve vs its own null); d whole brain R+ / R- (mean +- SEM; bars = clusters vs group mouse-block null); e R+ minus R- (two-sided cluster, mouse-level cohort permutation); f window values (Mann-Whitney | Welch); g decoding vs behavioural d (scatter, OLS, 95 % CI; solid if p < 0.05); h, i area overlays; j onset latency per area; k, l area x time heatmaps; m per-area R+ vs R- with two-way ANOVA cohort x area. Values = accuracy minus each session null.')}

### I.2 Stimulus modality is decodable from 10 ms and more strongly in R- later on

Whisker vs auditory trials were separable from the first bins after onset in both cohorts (first significant cluster
from {num('p1_mon_rp', mon['R+'].t_start_ms.iloc[0], '{:.0f}')} ms in R+ and {num('p1_mon_rm', mon['R-'].t_start_ms.iloc[0], '{:.0f}')} ms in R-;
5-50 ms R+ {num('p1_ms_rp', ms['R+']['mean'], '{:+.3f}')}, R- {num('p1_ms_rm', ms['R-']['mean'], '{:+.3f}')}). The cohorts differed, higher in {'R+' if len(mcl) and mcl['sign'].iloc[0] > 0 else 'R-'},
from {num('p1_mcl_t0', mcl.t_start_ms.iloc[0] if len(mcl) else np.nan, '{:.0f}')} ms (p = {pv('p1_mcl_p', mcl.p.iloc[0] if len(mcl) else np.nan)}),
i.e. once the lick-related activity differs (Supplementary Figure S11).

### I.3 Pre-lick modality: equal in the two cohorts, earlier in R+ areas, declining over the R+ session

In the 100 ms before the first lick, licked whisker and auditory trials were decodable in both cohorts (R+
{num('p1_ml_rp', ml['R+']['mean'], '{:+.3f}')}, n = {int(ml['R+'].n)}; R- {num('p1_ml_rm', ml['R-']['mean'], '{:+.3f}')}, n = {int(ml['R-'].n)};
both p {pv('p1_ml_p', max(ml['R+'].p_nonparam, ml['R-'].p_nonparam))}) without a cohort difference (Mann-Whitney p =
{pv('p1_mlc_mw', mlc.p_nonparam)}, Welch p = {pv('p1_mlc_w', mlc.p_param)}). After the lick, R- decoding was higher (p
{pv('p1_mpc_mw', mpc.p_nonparam)} / {pv('p1_mpc_w', mpc.p_param)}; cluster from {num('p1_mlcl_t0', mlcl.t_start_ms.iloc[-1] if len(mlcl) else np.nan, '{:.0f}')} ms).
Across area groups, modality information reached half of its peak earlier before the lick in R+ (median latency
{num('p1_lat_rp', lat_med['R+'], '{:.0f}')} ms) than in R- ({num('p1_lat_rm', lat_med['R-'], '{:.0f}')} ms; Table 2). In R+ the
pre-lick decoding was lower in the second half of the session ({num('p1_mh_a', mh['R+'].mean_a, '{:.3f}')} -> {num('p1_mh_b', mh['R+'].mean_b, '{:.3f}')},
paired Wilcoxon p = {pv('p1_mh_pw', mh['R+'].p_nonparam)}, paired t p = {pv('p1_mh_pt', mh['R+'].p_param)}, n = {int(mh['R+'].n)}); not in R-
(p = {pv('p1_mh_rm', mh['R-'].p_nonparam)}). This half decline is not larger than at placebo splits (Part II, Table 7) (Figure 2).

{figure(FIG / 'publication' / '110_modality_lick.png', 'Figure 2', 'Whisker vs auditory decoding aligned to the corrected first lick, learning stage (110, pre-lick window -100 to 0 ms). Whole-brain curves per cohort with clusters vs the group mouse-block null, R+ minus R- cluster, window values, session halves, area overlays, onset latencies (50 % of peak, mouse-bootstrap 95 % CI) and area x time heatmaps.')}

### I.4 Performance state

The high vs low performance state of 5-trial blocks was decodable at a low level in both cohorts (5-50 ms: R+
{num('p1_ps_rp', ps['R+']['mean'], '{:+.3f}')}, R- {num('p1_ps_rm', ps['R-']['mean'], '{:+.3f}')}), and in R- it tracked behavioural
d' (Pearson r = {num('p1_pd_r', pd_.r_pearson, '{:+.2f}')}, p = {pv('p1_pd_p', pd_.p_nonparam)}; Spearman p = {pv('p1_pd_ps', pd_.p_param)})
(Supplementary Figure S12).

{mdtable(wb_table(), 'Table 1. Session-wide whole-brain decoding, learning stage: accuracy minus null per cohort (Wilcoxon | one-sample t) and R+ vs R- (Mann-Whitney | Welch). Windows: sensory 5-50 ms, baseline -200 to -10 ms, pre-lick -100 to 0 ms, post-lick 5-200 ms.', P, ("acc - null (R+ - R- for the test row)",), ("n",))}
{mdtable(lat_t, 'Table 2. Pre-lick modality onset latency per area group (50 % of the group-mean peak, ms before the first lick; 110).', (), (), ('R+ latency (ms)', 'R- latency (ms)'))}
"""


def sec_front() -> str:
    h = row(T117, cohort="R+", mode="matched"), row(T117, cohort="R-", mode="matched")
    l5 = row(T126, decoding="hitmiss", window="5-100ms", variant="L5 whisker CP", cohort="R+")
    l5m = row(T126, decoding="hitmiss", window="5-100ms", variant="L5 whisker CP", cohort="R-")
    ml5 = row(T126, decoding="modality_lick", window="-100-0ms", variant="L5 whisker CP", cohort="R-")
    pp = row(T016, decoding="hitmiss", window="5-100ms", area="All units")
    c131 = row(T131, stat="excess", decoding="hitmiss", window="5-100ms", variant="L5 whisker CP")
    m127a = row(T127, decoding="whisker_nostim", window="5-100ms", curve="whisker", cohort="R+")
    m127b = row(T127, decoding="whisker_auditory", window="5-100ms", curve="whisker", cohort="R-")
    a = row(T135, panel="b")
    hs_rp, hs_rm = (pick(T113, analysis="hitmiss", window="sensory", condition="whole", cohort=c,
                         test="accuracy - null vs 0 (Wilcoxon | one-sample t)")["mean"] for c in ("R+", "R-"))
    hcm = pick(T113, analysis="hitmiss", window="sensory", condition="whole", cohort="R+ vs R-", test="Mann-Whitney | Welch")
    n_rp, n_rm = int(T126[(T126.variant == "half") & (T126.decoding == "hitmiss") & (T126.window == "5-100ms") & (T126.cohort == "R+")].n.iloc[0]), \
        int(T126[(T126.variant == "half") & (T126.decoding == "hitmiss") & (T126.window == "5-100ms") & (T126.cohort == "R-")].n.iloc[0])
    return f"""---
title: "Choice and modality information in whole-brain activity during single-session whisker learning"
subtitle: "ssl-whisker-hitmiss-timeresolved-decoding: session-wide time-resolved decoding, within-session change, stimulus-onset geometry"
date: "{date.today():%Y-%m-%d}"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
header-includes:
  - \\usepackage{{caption}}
  - \\captionsetup{{labelformat=empty,font=small}}
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
---

**Data.** SSL dataset (Axel Bisi, private), Kilosort 4 spike sorting (`ssl_ephys` 1.0.0, built 2026-08-27 from `NWB_ks4`);
learning-stage sessions (whisker day 0), one per mouse; cohort (R+ / R-) per mouse from `joint_mouse_reference_weight.xlsx`
(R+proba excluded); up to {num('n_rplus_half', n_rp)} R+ and {num('n_rminus_half', n_rm)} R- sessions per analysis.
Population scope: entire dataset unless stated (learners-only scope reported where computed).

# Abstract

Mice of two cohorts learn the same whisker Go/NoGo task within a single session; in R+ licking after the whisker stimulus
is rewarded, in R- it is not. We decoded, from whole-brain and area-level Neuropixels populations, the upcoming choice
(lick vs no lick on whisker trials) after the whisker stimulus, the stimulus modality at onset and before the first lick,
and asked how this information differs between cohorts and whether it changes within the learning session. Over the whole
session, choice was decodable in both cohorts already before the stimulus and after it (5-50 ms: R+
{num('a_hs_rp', hs_rp, '{:+.3f}')}, R- {num('a_hs_rm', hs_rm, '{:+.3f}')} above the shift null), without a cohort difference
in the early window; pre-lick modality information was equal in the two cohorts but reached half of its peak earlier
before the lick in R+ areas. Splitting each session in halves showed no
change in either cohort, per session (count-matched decoders) or with pseudo-populations. Splitting at behavioural learning
trials instead, and comparing each split with placebo splits of the same session, showed one effect: in R+, hit/miss
decoding increased at the whisker-only behavioural change point (post - pre {num('l5_rp_delta', l5.mean_delta, '{:+.3f}')},
excess over placebo {num('l5_rp_excess', l5.mean_excess, '{:+.3f}')}, Wilcoxon p = {pv('l5_rp_pw', l5.p_wilcoxon)},
t p = {pv('l5_rp_pt', l5.p_t)}, n = {num('l5_rp_n', int(l5.n))}), more than in R- (Mann-Whitney p = {pv('l5_mw', l5.p_mw_cohorts)},
Welch p = {pv('l5_welch', l5.p_welch_cohorts)}). Single-trial decoder margins tracked the behavioural learning curve in
opposite directions in the two cohorts. At stimulus onset (5-35 ms), the whisker-evoked population pattern of R- mice
decoupled from the active lick axis in the second half of the task, while the auditory-evoked pattern did not. All
p-values are uncorrected and all analyses are exploratory.

# Key results

1. **Session-wide choice decoding is equal across cohorts early on.** Hit vs miss 5-50 ms above the shift null: R+
   {num('k0_rp', hs_rp, '{:+.3f}')}, R- {num('k0_rm', hs_rm, '{:+.3f}')}; R+ vs R- Mann-Whitney p = {pv('k0_mw', hcm.p_nonparam)},
   Welch p = {pv('k0_w', hcm.p_param)} (Part I).
1. **No change between session halves.** Count-matched per-session hit/miss decoding (5-50 ms): R+ change
   {num('h117_rp', h[0].change, '{:+.3f}')} (n = {num('h117_rp_n', int(h[0].n))}, Wilcoxon p = {pv('h117_rp_p', h[0].p_wilcoxon)}),
   R- {num('h117_rm', h[1].change, '{:+.3f}')} (n = {num('h117_rm_n', int(h[1].n))}, p = {pv('h117_rm_p', h[1].p_wilcoxon)});
   pseudo-population hit/miss 5-100 ms: R+ delta {num('pp_rp', pp['delta_R+_med'], '{:+.3f}')} (p = {pv('pp_rp_p', pp['p_delta_R+'])}),
   R- {num('pp_rm', pp['delta_R-_med'], '{:+.3f}')} (p = {pv('pp_rm_p', pp['p_delta_R-'])}).
2. **R+ hit/miss gain at the whisker change point (L5).** Excess over placebo {num('k_l5', l5.mean_excess, '{:+.3f}')}
   (n = {int(l5.n)}, Wilcoxon p = {pv('k_l5_pw', l5.p_wilcoxon)}, t p = {pv('k_l5_pt', l5.p_t)}); R-
   {num('k_l5m', l5m.mean_excess, '{:+.3f}')} (n = {num('k_l5m_n', int(l5m.n))}, p = {pv('k_l5m_p', l5m.p_wilcoxon)});
   cohort-label permutation of the excess p = {pv('k_c131', c131.p_perm)}.
3. **R- pre-lick modality decoding drops at L5** (excess {num('k_ml5', ml5.mean_excess, '{:+.3f}')}, n = {num('k_ml5_n', int(ml5.n))},
   Wilcoxon p = {pv('k_ml5_pw', ml5.p_wilcoxon)}, t p = {pv('k_ml5_pt', ml5.p_t)}).
4. **Single-trial margins vs learning curve.** R+ whisker vs no-stim margin rises with the whisker curve (excess r
   {num('k_127a', m127a.mean_excess, '{:+.3f}')}, n = {num('k_127a_n', int(m127a.n))}, p = {pv('k_127a_p', m127a.p_wilcoxon)});
   R- whisker vs auditory margin falls with it ({num('k_127b', m127b.mean_excess, '{:+.3f}')}, n = {num('k_127b_n', int(m127b.n))},
   p = {pv('k_127b_p', m127b.p_wilcoxon)}); cohort difference p = {pv('k_127b_mw', m127b.p_mw_cohorts)} (Mann-Whitney).
5. **Whisker-evoked pattern vs lick axis (5-35 ms).** R- falls from {num('k_135_pre', a['mean_R-_passive_pre'], '{:+.2f}')}
   (passive pre) to {num('k_135_a2', a['mean_R-_active_2'], '{:+.2f}')} (active, 2nd half), R+ {num('k_135r_pre', a['mean_R+_passive_pre'], '{:+.2f}')}
   to {num('k_135r_a2', a['mean_R+_active_2'], '{:+.2f}')}; cohort difference of the change p = {pv('k_135_mw', a.pMW_change_active_2)}
   (Mann-Whitney), {pv('k_135_welch', a.pWelch_change_active_2)} (Welch); n = {num('k_135_nrp', int(a['n_R+']))} R+ (bad learners
   removed), {num('k_135_nrm', int(a['n_R-']))} R-.
"""


def sec_intro() -> str:
    return """
# Introduction

In the single-session-learning (SSL) task, head-fixed mice that already lick for an auditory tone learn, within one session,
to respond to a whisker deflection. The two cohorts receive identical stimuli; they differ only in whether a lick after the
whisker stimulus is rewarded (R+) or not (R-). Behaviourally, R+ mice increase their whisker hit rate while R- mice reduce
licking to whisker stimuli. The same overt action, a lick after the whisker stimulus, is therefore the trained response in
R+ and an error in R-.

The project asks three questions. **Part I** (session-wide): how much information do whole-brain and area-level
populations carry about the upcoming choice after the whisker stimulus, about the stimulus modality at onset and before
the first lick, and about the performance state; when does it appear, in which areas, and does it differ between cohorts?
**Part II** (within-session change): does this information change within the learning session, at the time the
behaviour changes, and does the change differ between cohorts? **Part III** (geometry): is the early (5-35 ms) sensory
representation of the whisker stimulus re-mapped relative to the lick representation between passive and active epochs?
Hypotheses: (H1) choice information after the whisker stimulus
grows when R+ mice learn; (H2) in R-, where licking to the whisker is not reinforced, it does not grow, or declines;
(H3) the early sensory representation of the whisker stimulus is re-mapped relative to the motor (lick) representation
depending on the reward contingency.

Part II goes from the simplest split to finer ones: (i) session halves; (ii) splits at behavioural learning trials (LT)
of several definitions; (iii) the LT split compared with placebo splits at every other whisker trial of the same session,
since any split also contains drift, engagement and hit-rate changes; (iv) cohort-label permutation of the pre/post change;
(v) continuous single-trial decoder margins compared with the learning curve, and searched for a neural transition; and
(vi) the geometry of stimulus-onset responses across passive and active epochs.
"""


def sec_methods() -> str:
    return r"""
# Methods

## Data, inclusion and trial selection

Learning-stage (whisker day 0) sessions with ephys from `ssl_ephys` 1.0.0 (Kilosort 4). Mice: `exclude == 0` and
`exclude_ephys == 0` in `joint_mouse_reference_weight.xlsx`; cohort from the same sheet; R+proba excluded. Units: quality
labels good + mua unless stated; whole brain = all units of the session.

Trials (`scripts/ssl_bwm_trial_prep.py::prep_session`, skill `ssl-trial-exclusion`): active context; trials with
`perf == 6` (invalid trials) excluded; the auditory-only warm-up block removed except its last trial (the first whisker trial
is always kept); end-of-session disengagement trimmed with rule A1 (the tail after the last lick is dropped when it holds
$\geq 5$ whisker and $\geq 1$ auditory trial). Lick times are corrected for the artefact window
($t_\text{lick} = t_\text{start} + \text{lick\_time} - t_\text{rw,start}$).

## Decodings and windows

* **Hit vs miss**: whisker trials, $y$ = lick; mean rate per unit in 5-50 ms or 5-100 ms after the stimulus (dead zone
  -10 to +5 ms excised).
* **Pre-lick modality**: licked whisker and auditory trials, $y$ = whisker, mean rate in -100 to 0 ms before the corrected
  first lick.
* **Stimulus decodings** (single-trial analyses): whisker vs no-stim and whisker vs auditory, 5-50 / 5-100 ms.

Classifier: standardisation then L2 logistic regression (liblinear), one $C$ per session and decoding from the grid by
pooled cross-validation on all trials; stratified $K$-fold CV with $K = \min(5, n_\text{minority})$; balanced accuracy.

## Part I: session-wide time-resolved decoding (024 sweep)

One decoder per session (and area group) and time bin: causal 50-ms bins labelled at their end ($(t - 50, t]$ ms, 5-ms
stride; stimulus-aligned -200 to 600 ms, lick-aligned -600 to 200 ms), units = good + mua of the whole brain or of each
`area_group` ($\geq 5$ units), $\geq 3$ trials per class. Hit vs miss uses `lick_flag` on whisker trials (the same label in both
cohorts, so it is a lick vs no-lick decoding). Each session's curve is compared with its own null curves (linear shift for
hit vs miss and performance state, trial shuffle for modality); values are accuracy minus the mean null. Group
significance: cluster-mass test across time bins against a group null built by mouse-block sign flips between the real
curve and one per-session surrogate (1000 draws, $|z| > 1.96$). R+ vs R-: two-sided cluster test of the difference curve
with mouse-level cohort-label permutation. Window values (baseline -200 to -10 ms, sensory 5-50 ms, pre-lick -100 to 0 ms,
post-lick 5-200 ms): Wilcoxon and one-sample $t$ against 0 within cohort, Mann-Whitney and Welch between cohorts, paired
Wilcoxon and paired $t$ between session halves. Onset latency: first bin reaching 50 % of the group-mean peak of accuracy
minus null, 95 % CI from 500 mouse bootstraps. Behavioural $d' = z(\text{whisker lick rate}) - z(\text{no-stim lick rate})$
(log-linear corrected). Performance state: high vs low hit rate of 5-whisker-trial blocks, median split per session.
Area tests are BH-FDR corrected across areas.

## Session-wide chance: linear-shift null

Labels are shifted against the neural trials by $k$ trials (non-wrapping, $k$ = 10-50 % of the trials, both directions);
each shift keeps the autocorrelation of both series, so slow drift shared by labels and activity is in the null. The
reported value is accuracy minus the mean null accuracy.

## Splits

For a split at trial $s$, *pre* = trials starting before whisker trial $s$, *post* = the rest. Separate decoders are trained
and tested within each epoch. **Size matching**: each epoch is subsampled to $\min(n_\text{pre}, n_\text{post})$ trials per
class (50 subsamples), so a change in hit rate cannot change the training-set size. Change:
$$\Delta = (a_\text{post} - \bar a^\text{null}_\text{post}) - (a_\text{pre} - \bar a^\text{null}_\text{pre}).$$
Split definitions (whisker-trial index; `ssl-learning-trial-identification`): **half** (median split of the decoded trials);
**L0** stored LT; **L5** Bayesian change point of the whisker hit sequence; **L6** joint whisker + false-alarm change point
(strict, lenient); **L6x** joint change point with a lapse segment (*relaxed*: $\log_{10}\text{BF} > 0.3$ and
$P(p_w > p_\text{FA}) > 0.9$ over a 20-trial window; *forced*: the best change point in every session); **L8** fixed-margin
crossing; and cascades of these. Sessions without an LT under a definition drop out of that definition.
Session halves were also decoded with **count-matched halves** (117: each half subsampled to the same number of hits and
of misses) and with **pseudo-populations** (016: 20 sessions $\times$ 10 units drawn with replacement per iteration,
pseudo-trials built within each half, 3-fold CV, trial-shuffle null; 100 iterations, a pilot value; $p$ from the bootstrap
distribution of the change, $p = 2\min(P(\Delta>0), P(\Delta<0))$).

## Placebo splits

Every whisker trial $k$ that leaves $\geq 2$ trials of each class in both epochs is a candidate split, decoded with the same
estimator (raw $\Delta_k = a_\text{post} - a_\text{pre}$, size-matched). For a real split $s$,
$$\text{excess} = \Delta_s - \frac{1}{|P|}\sum_{k \in P} \Delta_k, \qquad P = \{k : |k - s| \geq 10\},$$
and the percentile of $\Delta_s$ among $\{\Delta_k\}_{k\in P}$ (0.5 = a typical split). The placebo splits share the
session's drift and class structure, so no per-split shift null is needed. **Cohort-label permutation**: statistic
$\overline{\text{excess}}_{R+} - \overline{\text{excess}}_{R-}$ (difference of differences), null from 20000 shuffles of
the cohort labels across mice (cohort sizes kept), two-sided. **Mid and best split** (129): percentile of the half split
among interior splits; best split = argmax of the smoothed $\Delta_k$ profile, compared with the behavioural change point
(L6x forced) by Spearman correlation and by the median $|k_\text{best} - \text{CP}|$ against change points permuted
across sessions within cohort (2000 permutations).

## Single-trial margins

Nested repeated CV (inner 3-fold CV on log-loss for $C$, balanced class weights) gives each trial a held-out signed margin,
scaled per session and **class-residualised** ($z_i$ minus the mean of its class), so a changing class composition cannot
create a trend. **Margin vs learning curve** (127): $r = \text{corr}(z, c)$ with $c$ the behavioural learning curve
(whisker $P(\text{lick})$ or whisker - FA, HMM, $\sigma = 1$) at the trial's time; excess $r$ = $r$ minus the mean $r$ of
linear-shift nulls (200 shifts). **Neural transition** (130): $c^* = \arg\max_c |t_\text{Welch}(z_{c:}, z_{:c})|$ over interior splits,
tested against 500 IAAFT surrogates of $z$ (same values and power spectrum, phases randomised).

## Stimulus-onset geometry across epochs (132-135b)

Sessions with passive trials before and after the active block. Units tracked across the session: quality label good and
$\geq 0.5$ Hz in every epoch. Response = rate (Hz) 5-35 ms after stimulus onset, minus the unit's mean -55 to -20 ms
baseline rate *within its epoch*, z-scored per unit over all trials. Whisker axis per epoch $D = \bar r_W - \bar r_A$;
lick axis (active epoch) $L = \bar r_\text{hit} - \bar r_\text{miss}$ of whisker trials without a lick before 35 ms;
evoked patterns $\bar r_W$, $\bar r_A$ alone. Cross-validated distance $d = \langle D^{(1)}, D^{(2)}\rangle / n_\text{units}$
over random half splits. Similarity of two axes estimated from disjoint trial halves, normalised by their split-half
reliabilities:
$$\cos_\text{norm}(u, v) = \frac{\cos(u^{(1)}, v^{(2)})}{\sqrt{\rho_u \rho_v}}, \qquad \rho_u = \cos(u^{(1)}, u^{(2)}),$$
with reliabilities floored at 0.05 and 50 splits; it can exceed $\pm 1$ when reliabilities are low, so it is not converted
to an angle. 135b splits the active epoch into halves and removes R+ mice with `learning_category == 'bad'`.

## Statistics

Unit of analysis: mouse (one learning session per mouse). Within cohort: Wilcoxon signed-rank and one-sample $t$ test
against 0 (or 0.5 for percentiles); across epochs: Friedman and repeated-measures ANOVA. Between cohorts: Mann-Whitney $U$
and Welch $t$. Non-parametric and parametric tests are always reported together. No correction for multiple comparisons.
Colours: R+ #00B400, R- #C800C8.
"""


def sec_results() -> str:
    t117 = T117[["cohort", "mode", "n", "first", "second", "change", "p_wilcoxon", "p_paired_t", "p_cohort_mannwhitney", "p_cohort_welch"]]
    t016 = T016[T016.area == "All units"][["decoding", "window", "n_pool_R+", "corr_h1_R+_med", "corr_h2_R+_med", "p_delta_R+",
                                           "n_pool_R-", "corr_h1_R-_med", "corr_h2_R-_med", "p_delta_R-", "p_diffdiff"]]
    t016.columns = ["decoding", "window", "n R+", "R+ h1", "R+ h2", "p R+", "n R-", "R- h1", "R- h2", "p R-", "p cohort"]
    hm = T119[(T119.decoding == "hitmiss") & (T119.window == "5-100ms") & T119.definition.isin(VAR)]
    t119 = hm[["definition", "cohort", "n", "pre", "post", "change", "p_wilcoxon", "p_mw_cohorts"]]
    t126h = T126[(T126.decoding == "hitmiss") & (T126.window == "5-100ms") & T126.variant.isin(VAR)][
        ["variant", "cohort", "n", "mean_delta", "mean_excess", "mean_percentile", "p_wilcoxon", "p_t", "p_mw_cohorts", "p_welch_cohorts"]]
    t126m = T126[(T126.decoding == "modality_lick") & T126.variant.isin(VAR)][
        ["variant", "cohort", "n", "mean_delta", "mean_excess", "mean_percentile", "p_wilcoxon", "p_t", "p_mw_cohorts", "p_welch_cohorts"]]
    t131 = T131[T131.variant.isin(VAR) & (T131.stat == "excess")][["decoding", "window", "variant", "n_rplus", "n_rminus", "mean_rplus", "mean_rminus", "diff_of_diff", "p_perm"]]
    t129 = T129[["decoding", "window", "cohort", "n", "half_pct_mean", "p_half_pct_wilcoxon", "p_half_pct_t", "best_pos_median", "spearman_best_cp", "p_perm_best_cp"]]
    t127 = T127[["decoding", "window", "curve", "cohort", "n", "mean_excess", "p_wilcoxon", "p_t", "p_mw_cohorts", "p_welch_cohorts"]]
    t130 = T130[["decoding", "window", "cohort", "n", "n_sig", "frac_sig", "p_fisher_frac_sig", "p_perm_all", "p_perm_learners"]]
    t135 = T135[["panel", "n_R+", "mean_R+_passive_pre", "mean_R+_active_1", "mean_R+_active_2", "mean_R+_passive_post",
                 "n_R-", "mean_R-_passive_pre", "mean_R-_active_1", "mean_R-_active_2", "mean_R-_passive_post",
                 "pMW_change_active_2", "pWelch_change_active_2"]].copy()
    t135.columns = ["panel", "n R+", "R+ pre", "R+ act1", "R+ act2", "R+ post", "n R-", "R- pre", "R- act1", "R- act2", "R- post", "p MW act2", "p Welch act2"]
    P = ("p_wilcoxon", "p_paired_t", "p_cohort_mannwhitney", "p_cohort_welch", "p R+", "p R-", "p cohort", "p_mw_cohorts", "p_t",
         "p_welch_cohorts", "p_perm", "p_half_pct_wilcoxon", "p_half_pct_t", "p_perm_best_cp", "p_fisher_frac_sig", "p_perm_all",
         "p_perm_learners", "p MW act2", "p Welch act2")
    S = ("change", "mean_delta", "mean_excess", "diff_of_diff", "mean_rplus", "mean_rminus", "first", "second", "pre", "post")
    I = ("n", "n R+", "n R-", "n_rplus", "n_rminus", "n_sig")

    hp, hm_ = row(T117, cohort="R+", mode="matched"), row(T117, cohort="R-", mode="matched")
    l5 = row(T126, decoding="hitmiss", window="5-100ms", variant="L5 whisker CP", cohort="R+")
    l5_50 = row(T126, decoding="hitmiss", window="5-50ms", variant="L5 whisker CP", cohort="R+")
    l6 = row(T126, decoding="hitmiss", window="5-100ms", variant="L6 joint CP", cohort="R+")
    l8 = row(T126, decoding="hitmiss", window="5-100ms", variant="L8 fixed margin", cohort="R-")
    half = row(T126, decoding="hitmiss", window="5-100ms", variant="half", cohort="R+")
    l0m = row(T119, decoding="modality_lick", window="-100-0ms", definition="L0 stored", cohort="R+")
    b129 = row(T129, decoding="hitmiss", window="5-100ms", cohort="R+")
    c1 = row(T131, stat="excess", decoding="hitmiss", window="5-100ms", variant="L5 whisker CP")
    c0 = row(T131, stat="delta", decoding="modality_lick", window="-100-0ms", variant="L0 stored")
    w127 = row(T127, decoding="whisker_nostim", window="5-100ms", curve="whisker", cohort="R+")
    h127 = row(T127, decoding="hitmiss", window="5-100ms", curve="whisker", cohort="R-")
    h127p = row(T127, decoding="hitmiss", window="5-100ms", curve="whisker", cohort="R+")
    n130 = row(T130, decoding="hitmiss", window="5-50ms", cohort="R-"), row(T130, decoding="hitmiss", window="5-50ms", cohort="R+")
    sig124 = T124[(T124.p_spearman < 0.05) & (T124.p_pearson < 0.05)]
    a, b, c = row(T135, panel="a"), row(T135, panel="b"), row(T135, panel="c")
    g133 = {m: row(T133, area="All units", metric=m) for m in ("within", "single", "distance", "angle")}

    return f"""
## Part II. Within-session change

### II.1 Decoding does not change between session halves

The chronological halves differ in hit rate, so the second-half decoder of an unmatched split is trained on fewer hits.
With halves matched in the number of hits and misses, hit/miss decoding (5-50 ms, above the shift null) was
{num('r1_rp_first', hp['first'])} vs {num('r1_rp_second', hp.second)} in R+ (change {num('r1_rp_ch', hp.change, '{:+.3f}')}, Wilcoxon p =
{pv('r1_rp_pw', hp.p_wilcoxon)}, paired t p = {pv('r1_rp_pt', hp.p_paired_t)}, n = {int(hp.n)}) and {num('r1_rm_first', hm_['first'])}
vs {num('r1_rm_second', hm_.second)} in R- (change {num('r1_rm_ch', hm_.change, '{:+.3f}')}, p = {pv('r1_rm_pw', hm_.p_wilcoxon)} /
{pv('r1_rm_pt', hm_.p_paired_t)}, n = {int(hm_.n)}); cohorts did not differ (Mann-Whitney p = {pv('r1_mw', hp.p_cohort_mannwhitney)},
Welch p = {pv('r1_welch', hp.p_cohort_welch)}; Figure 4A, Table 3). Pseudo-populations, which equalise neuron and trial numbers
across cohorts, gave the same answer for hit/miss and for pre-lick modality decoding (Figure 4B, Table 4). Note: 117 was run
before invalid (`perf == 6`) trials were excluded (Caveats).

{figure(FIG / '117b_halves_matched_full.png', 'Figure 4A', 'Session halves, per session. Hit/miss decoding (5-50 ms, accuracy minus the linear-shift null) in the first and second half, halves unmatched (left) or matched in the number of hits and misses (right); lines = mice, R+ green, R- magenta; Wilcoxon and paired t within cohort, Mann-Whitney and Welch between cohorts.')}
{mdtable(t117, 'Table 3. Count-matched session halves (117b), hit/miss 5-50 ms.', P, S, I)}
{figure(PP / 'exploratory-analyses' / 'figures' / '016_pseudopop_halves.png', 'Figure 4B', 'Pseudo-population halves (016). Trial-shuffle-corrected balanced accuracy in each half (median and 95 % interval over 100 iterations of 20 sessions x 10 units), whole brain and per area group, for hit/miss 5-50 / 5-100 ms and pre-lick modality (-100 to 0 ms).')}
{mdtable(t016, 'Table 4. Pseudo-population halves, whole brain (016; pilot, 100 iterations).', P, (), I)}

### II.2 Splits at learning-trial definitions

Splitting at a learning trial instead of the half (Figure 5, Table 5) gave larger R+ hit/miss gains for the change-point
definitions (L5, L6) than for the half or the stored LT. Because each definition selects its own sessions (those with an
LT), these changes are not comparable with each other without a common reference; section II.3 provides it. Pre-lick modality
decoding decreased after the stored LT in R+ ({num('r2_l0m', l0m.change, '{:+.3f}')}, p = {pv('r2_l0m_p', l0m.p_wilcoxon)},
n = {int(l0m.n)}).

{figure(FIG / '119_lt_definitions_comparison_A1v2.png', 'Figure 5', 'Pre vs post decoding at each learning-trial definition (118 -> 119). Accuracy above the linear-shift null before and after the split, size-matched decoders, per cohort; columns = hit/miss 5-50 ms, 5-100 ms, pre-lick modality; rows = definitions.')}
{mdtable(t119, 'Table 5. Pre / post hit/miss decoding (5-100 ms) per definition (119).', P, S, I)}

### II.3 Placebo splits: the whisker change point is special in R+

Against the placebo splits of the same session, the R+ hit/miss change at L5 was larger than at typical splits (excess
{num('r3_l5', l5.mean_excess, '{:+.3f}')}, percentile {num('r3_l5_pct', l5.mean_percentile, '{:.2f}')}, Wilcoxon p = {pv('r3_l5_pw', l5.p_wilcoxon)},
t p = {pv('r3_l5_pt', l5.p_t)}, n = {int(l5.n)}; 5-50 ms: p = {pv('r3_l5_50', l5_50.p_wilcoxon)}), and larger than in R-
(Mann-Whitney p = {pv('r3_l5_mw', l5.p_mw_cohorts)}, Welch p = {pv('r3_l5_w', l5.p_welch_cohorts)}). L6 went the same way
(excess {num('r3_l6', l6.mean_excess, '{:+.3f}')}, p = {pv('r3_l6_p', l6.p_wilcoxon)}). The half split was not special (R+ excess
{num('r3_half', half.mean_excess, '{:+.3f}')}, p = {pv('r3_half_p', half.p_wilcoxon)}). In R-, hit/miss decoding dropped after the
fixed-margin LT (L8: {num('r3_l8', l8.mean_excess, '{:+.3f}')}, p = {pv('r3_l8_p', l8.p_wilcoxon)}, n = {int(l8.n)}) (Figure 6, Tables 6-7).

The cohort-label permutation of the excess confirmed the cohort difference at L5 (difference of differences
{num('r3_c1', c1.diff_of_diff, '{:+.3f}')}, p = {pv('r3_c1_p', c1.p_perm)}; Figure 7, Table 8). For pre-lick modality decoding,
only the raw change at the stored LT differed between cohorts ({num('r3_c0', c0.diff_of_diff, '{:+.3f}')}, p = {pv('r3_c0_p', c0.p_perm)});
no placebo-corrected modality effect differed.

The split with the largest change (129) did not sit at the behavioural change point (R+ hit/miss 5-100 ms: Spearman
{num('r3_b_rho', b129.spearman_best_cp, '{:+.2f}')}, permutation p = {pv('r3_b_p', b129.p_perm_best_cp)}; Supplementary Figure S2), so
the L5 effect is a property of that split, not the peak of the session's change profile.

{figure(FIG / '126_lt_variants_modality_step1_pl100.png', 'Figure 6', 'Real split vs placebo splits (126). Excess of the post - pre change at each definition over the mean of placebo splits >= 10 whisker trials away, per mouse (dots) and cohort mean +- SEM; hit/miss 5-50 / 5-100 ms and pre-lick modality -100 to 0 ms.')}
{mdtable(t126h, 'Table 6. Real vs placebo splits, hit/miss 5-100 ms (126).', P, S, I)}
{mdtable(t126m, 'Table 7. Real vs placebo splits, pre-lick modality -100 to 0 ms (126).', P, S, I)}
{figure(FIG / '131_cohort_permutation_excess.png', 'Figure 7', 'Cohort-label permutation (131). Difference of differences mean(R+) - mean(R-) of the excess over placebo: real value (black) and null from 20000 shuffles of cohort labels across mice (grey); rows = decodings, columns = split definitions; red title = p < 0.05 (uncorrected).')}
{mdtable(t131, 'Table 8. Cohort-label permutation of the excess (131).', P, S, I)}

### II.4 Behaviour scores and split effects

Across sessions, behaviour scores were related to split effects only in R- and mainly for pre-lick modality decoding: the
stronger the change-point evidence, the lower the modality excess; {num('r4_nsig', len(sig124))} score x measure pairs had both Spearman
and Pearson p < 0.05 (Supplementary Figure S3, Table S2).

### II.5 Single-trial margins follow the learning curve with cohort-specific signs

Correlating class-residualised margins with the learning curve against a linear-shift null (Figure 8, Table 9): the R+
whisker vs no-stim margin rose with the whisker curve (excess r {num('r5_w', w127.mean_excess, '{:+.3f}')}, p = {pv('r5_w_p', w127.p_wilcoxon)},
t p = {pv('r5_w_pt', w127.p_t)}, n = {int(w127.n)}); the R- hit/miss margin fell with it ({num('r5_h', h127.mean_excess, '{:+.3f}')},
p = {pv('r5_h_p', h127.p_wilcoxon)}, n = {int(h127.n)}; R+ {num('r5_hp', h127p.mean_excess, '{:+.3f}')}, p = {pv('r5_hp_p', h127p.p_wilcoxon)};
cohort difference Mann-Whitney p = {pv('r5_h_mw', h127.p_mw_cohorts)}, Welch p = {pv('r5_h_w', h127.p_welch_cohorts)}). A step-like
neural transition beyond IAAFT surrogates was found in few sessions (hit/miss 5-50 ms: {int(n130[1].n_sig)}/{int(n130[1].n)} R+,
{int(n130[0].n_sig)}/{int(n130[0].n)} R-; Fisher p = {pv('r5_fisher', n130[0].p_fisher_frac_sig)}; Supplementary Figure S5, Table S3): the
changes are gradual rather than steps.

{figure(FIG / '127_margin_vs_learning_curve.png', 'Figure 8', 'Single-trial margin vs learning curve (127). Per session, excess correlation (r minus the mean r of linear-shift nulls) between the class-residualised held-out margin and the whisker or whisker - FA learning curve, for hit/miss, whisker vs no-stim and whisker vs auditory decoders; dots = mice, bars = cohort mean +- SEM.')}
{mdtable(t127, 'Table 9. Margin vs learning curve (127).', P, S, I)}

## Part III. Stimulus-onset geometry: R- whisker responses decouple from the lick axis

Whisker vs auditory decoding at stimulus onset was near ceiling in every epoch. Between cohorts, the change from passive pre
to active differed weakly for the within-epoch decoder (Mann-Whitney p = {pv('r6_within_mw', g133['within']['mw_active-pre'])},
Welch p = {pv('r6_within_w', g133['within']['welch_active-pre'])}) and not for the single decoder (p = {pv('r6_single_mw', g133['single']['mw_active-pre'])} /
{pv('r6_single_w', g133['single']['welch_active-pre'])}) or the cross-validated distance (p = {pv('r6_dist_mw', g133['distance']['mw_active-pre'])} /
{pv('r6_dist_w', g133['distance']['welch_active-pre'])}); the raw angle between the active and passive-post axes differed
(p = {pv('r6_angle_mw', g133['angle']['mw_post-active'])} / {pv('r6_angle_w', g133['angle']['welch_post-active'])})
(Supplementary Figures S7-S9). The cohorts differed most clearly in the
alignment of the whisker response with the active lick axis (Figure 9, Table 10). The whisker-evoked pattern alone went from
{num('r6_b_rm_pre', b['mean_R-_passive_pre'], '{:+.2f}')} (passive pre) through {num('r6_b_rm_a1', b['mean_R-_active_1'], '{:+.2f}')} and
{num('r6_b_rm_a2', b['mean_R-_active_2'], '{:+.2f}')} (active halves) to {num('r6_b_rm_post', b['mean_R-_passive_post'], '{:+.2f}')} (passive post)
in R- (Friedman p = {pv('r6_b_fr', b['friedman_R-'])}, RM-ANOVA p = {pv('r6_b_rm', b['rmanova_R-'])}), and from
{num('r6_b_rp_pre', b['mean_R+_passive_pre'], '{:+.2f}')} to {num('r6_b_rp_a2', b['mean_R+_active_2'], '{:+.2f}')} in R+; the change to the
second active half differed between cohorts (Mann-Whitney p = {pv('r6_b_mw', b.pMW_change_active_2)}, Welch p = {pv('r6_b_w', b.pWelch_change_active_2)}).
The whisker axis (whisker - auditory) behaved the same way (p = {pv('r6_a_mw', a.pMW_change_active_2)} / {pv('r6_a_w', a.pWelch_change_active_2)}),
whereas the auditory-evoked pattern did not differ between cohorts (p = {pv('r6_c_mw', c.pMW_change_active_2)} / {pv('r6_c_w', c.pWelch_change_active_2)}),
so the effect is not a general reward-expectation signal carried by the auditory response.

{figure(FIG / '135b_alignment_no_bad_rplus.png', 'Figure 9', 'Whisker axis and evoked patterns vs the active lick axis, 5-35 ms (135b). Reliability-normalised cosine in passive pre, active 1st and 2nd half and passive post; a whisker - auditory axis; b whisker-evoked pattern; c auditory-evoked pattern; d as a with >= 6 licked and unlicked trials; e-g area groups (midbrain, motor, striatum). Mean +- SEM per cohort, faint lines = mice; R+ bad learners removed. Stars: R+ vs R- on the change from passive pre (** both tests p < 0.05, * one).')}
{mdtable(t135, 'Table 10. Alignment with the lick axis per epoch (135b; panels as Figure 9).', P, (), I)}
"""


def sec_discussion() -> str:
    return """
# Discussion

Across the whole learning session, whole-brain activity carries as much early choice information in R- as in R+, and
equal pre-lick modality information; the cohorts differ late after the stimulus, when the licks themselves differ, and in
how early modality information builds up before the lick (earlier in R+ areas). A lick after the whisker stimulus is
thus predictable from early activity whether or not it is rewarded, consistent with the same sensorimotor pathway being
engaged in both cohorts, while the reward contingency shapes the timing and the later course of the signal.

Within the learning session, the choice information carried by early whisker responses does not change on average between
session halves in either cohort. It does change at the trial where R+ behaviour changes: hit/miss decoding is higher after
the whisker change point than before it, more than at other splits of the same session and more than in R-. This fits H1
(choice information grows with learning when licking is rewarded) and H2 (no growth in R-), but only at a split defined
from behaviour. Since the best split of the neural change profile does not coincide with the behavioural change point, the
neural change may be gradual and the change-point split may simply be the split that separates low- and high-performance
trials best.

The single-trial margins support a gradual co-variation with behaviour that differs by cohort: in R+ the stimulus
representation strengthens with the learning curve; in R- the whisker vs auditory and hit/miss margins weaken as the mice
learn to withhold licks. At stimulus onset, the R- whisker-evoked pattern stops pointing along the lick axis during the
task (H3), while the auditory-evoked pattern does not change, consistent with a contingency-specific re-mapping of the
whisker response relative to the motor output.

Alternative explanations: (i) the class balance changes most at the behavioural change point; decoders are size-matched,
but the remaining hits after an R+ change point may be more typical; (ii) engagement and arousal co-vary with learning;
the placebo comparison removes shared session-level drift but not changes locked to the change point; (iii) the change-point
definitions select subsets of mice, and different definitions give different answers.
"""


def sec_caveats() -> str:
    return """
# Caveats and limitations

* All analyses are exploratory; parameters (windows, definitions, the 5-35 ms window and the bad-learner removal in 135b)
  were chosen on these data. No multiple-comparison correction; many tests were run.
* The count-matched halves (117) and the single-trial margin vs LT comparison (123b) were run before invalid (`perf == 6`)
  trials were excluded (AB107, AB142, AB149 and three other day-0 sessions); they need a rerun.
* Pseudo-population results use 100 iterations (pilot); final figures need more.
* The session-wide sweep (Part I; 024 -> 110 / 113) was also run before the `perf == 6` exclusion (13 sessions: six at day 0,
  seven expert); expert-stage area-group results mix redone and older sessions; `area_acronym_custom` results are no longer
  maintained. Reruns are deferred.
* Hit vs miss is a lick vs no-lick decoding with the same label in both cohorts; a cohort difference can reflect the
  meaning of the action (trained response in R+, error in R-) or motor differences, not only coding.
* Choice is decodable before the stimulus (baseline window), so post-stimulus hit/miss decoding mixes evoked and state
  (engagement) information; the shift null removes slow drift but not trial-to-trial state.
* The learning-trial definition is not locked; the L5 effect is one of several definitions tested.
* Parts II and III are learning stage only; expert-stage results appear only as a learning vs expert comparison (S14), and expert-stage trial selection changed on 2026-10-04 for
  one session, MH062_20260113).

# Open questions and next steps

1. Rerun 117 and 123b with the current trial exclusions.
2. Test the R+ change-point effect against a placebo matched on the local change in hit rate, and in the learners-only
   scope.
3. Lock a learning-trial definition (L6x relaxed is the current candidate).
4. 135b robustness: other early windows, all units vs good units, both population scopes, per area group.
5. Raise pseudo-population iterations for final figures.
"""


def sec_supp() -> str:
    t124 = T124[(T124.p_spearman < 0.05) | (T124.p_pearson < 0.05)][["neural", "score", "cohort", "n", "spearman", "p_spearman", "pearson", "p_pearson", "p_perm_spearman"]]
    t130 = T130[["decoding", "window", "cohort", "n", "n_sig", "frac_sig", "p_fisher_frac_sig", "p_perm_all", "p_perm_learners"]]
    t129 = T129[["decoding", "window", "cohort", "n", "half_pct_mean", "p_half_pct_wilcoxon", "p_half_pct_t", "best_pos_median", "spearman_best_cp", "p_perm_best_cp"]]
    P = ("p_spearman", "p_pearson", "p_perm_spearman", "p_fisher_frac_sig", "p_perm_all", "p_perm_learners", "p_half_pct_wilcoxon", "p_half_pct_t", "p_perm_best_cp")
    I = ("n", "n_sig")
    return f"""
# Supplementary figures and tables

{figure(FIG / '122b_lt_placebo.png', 'Figure S1', 'Placebo percentile of the real split per definition (122b): rank of the real post - pre change among the session placebo splits; 0.5 = typical.')}
{figure(FIG / '129_mid_and_best_split.png', 'Figure S2', 'Mid-session and best split (129). Rows = hit/miss 5-50 ms, 5-100 ms, pre-lick modality. Columns: percentile of the half split among interior splits (dots = mice); histogram of the best-split position (fraction of the session); mean post - pre change vs split position (10 bins, mean +- SEM); best split vs behavioural change point (L6x forced).')}
{mdtable(t129, 'Table S1. Mid and best split (129).', P, (), I)}
{figure(FIG / '124_score_vs_excess.png', 'Figure S3', 'Behaviour scores vs excess over placebo (124): change-point evidence (L5, L6 log10 BF), net and linear slope of the W - FA curve, max slope; scatter with OLS line and 95 % CI (solid if p < 0.05).')}
{mdtable(t124, 'Table S2. Behaviour score correlations with Spearman or Pearson p < 0.05 (124).', P, (), I)}
{figure(FIG / '125_behaviour_groups_split.png', 'Figure S4', 'Split effects per behaviour group (125): step learners, gradual, no transition; forced and mid-session splits.')}
{figure(FIG / '130_neural_transition.png', 'Figure S5', 'Neural transitions in the single-trial margin vs IAAFT surrogates (130): fraction of sessions with a significant step, its direction and position relative to the behavioural change point.')}
{mdtable(t130, 'Table S3. Neural transitions (130).', P, (), I)}
{figure(FIG / '120_lt_defined_vs_undefined_A1v2.png', 'Figure S6', 'Sessions with an LT (split at the LT and at the half) vs sessions without an LT (split at the half) (120).')}
{figure(FIG / '133_whole_brain_good.png', 'Figure S7', 'Stimulus-onset whisker vs auditory across passive pre, active and passive post (133, tracked good units): within-epoch, single and cross-epoch decoders (minus shuffle), cross-validated distance and reliability-normalised cosine between epoch axes.')}
{figure(FIG / '133c_single_decoder_behaviour_good.png', 'Figure S8', 'Single decoder evaluated per epoch vs behaviour (133c).')}
{figure(FIG / '134_whole_brain.png', 'Figure S9', 'Whisker-specific gain change (auditory as control) and lick-axis cosine in sliding 30-ms windows 5-200 ms (134).')}
{figure(FIG / '134c_illustration.png', 'Figure S10', 'Illustration of the whisker axis, evoked patterns and lick axis (134c).')}
{figure(FIG / 'publication' / '113_modality_stim_set1_session_wide.png', 'Figure S11', 'Whisker vs auditory decoding at stimulus onset, session-wide, learning stage (113 set 1; layout as Figure 1).')}
{figure(FIG / 'publication' / '113_perfstate_set1_session_wide.png', 'Figure S12', 'Performance-state decoding (high vs low hit-rate state of 5-trial blocks), session-wide, whole brain, learning stage (113 set 1, panels a-g).')}
{figure(FIG / 'publication' / '113_hitmiss_set2_within_session.png', 'Figure S13', 'Hit vs miss decoding in the first vs second half of the session from the session-wide sweep (113 set 2): whole-brain curves per half with paired-difference clusters, window values, change over time, cross-half generalisation, area x time heatmaps of the change and per-area tests. Separate decoders per half, not count-matched (compare Figure 4A).')}
{figure(FIG / 'whole_brain' / 'learning_vs_expert' / '041_learning_vs_expert_whole_brain_comparison.png', 'Figure S14', 'Learning vs expert stage, whole brain, window values per decoding (041): rows hit/miss, performance state, modality at stimulus, modality pre-lick; columns R+, R-, pooled; Mann-Whitney, Welch and mouse-block permutation (mice contribute several expert sessions). Pre-dates the perf == 6 exclusion and the -100 ms pre-lick window.')}
{figure(FIG / 'publication' / '115_cosyne_halves_figure_024_5-100.png', 'Figure S15', 'Abstract figure: session-half hit/miss decoding 5-100 ms, R+ vs R- (115, from the 024 sweep halves).')}
"""


def sec_appendix() -> str:
    params = [
        ("Stimulus windows", "5-50, 5-100 ms (decoding); 5-35 ms (geometry); dead zone -10 to +5 ms"),
        ("Pre-lick window", "-100 to 0 ms before the corrected first lick (changed from -150 ms on 2026-10-02)"),
        ("Baseline (geometry)", "-55 to -20 ms, per epoch"),
        ("Classifier", "StandardScaler + L2 logistic regression, one C per session (pooled CV), K = min(5, minority)"),
        ("Size matching", "min(pre, post) per class, 50 subsamples (118), 10 (117 count-matched)"),
        ("Shift null", "linear, non-wrapping, 10-50 % of the trials; 25 shifts (118)"),
        ("Placebo", "every whisker trial with >= 2 trials per class per epoch; placebo set |k - s| >= 10"),
        ("Cohort permutation", "20000 shuffles of cohort labels across mice"),
        ("IAAFT", "500 surrogates (130)"),
        ("Pseudo-population", "20 sessions x 10 units, 100 iterations (pilot), 3-fold CV"),
        ("Geometry splits", "50 random half splits; reliabilities floored at 0.05; min 5 trials per class per half"),
        ("Disengagement", "rule A1 (>= 5 whisker and >= 1 auditory trial after the last lick)"),
    ]
    scripts = ["024_master_sweep.py (session-wide sweep), 110_publication_figures.py (pre-lick rerun), 113_publication_merged.py, 041, 115", "117_halves_matched_pilot.py / 117b_halves_matched_full_figure.py", "118_lt_definitions_split_decoding.py, 119, 120",
               "122_lt_definitions_placebo.py (SSL_PLACEBO_TAG=_step1_pl100), 122b, 126", "124, 125 (behaviour)", "129, 131",
               "123_singletrial_scores_all.py (v2), 128, 127, 130", "132, 133 (SSL_UNITS=good), 133c, 134, 134c, 135, 135b",
               "ssl-pseudopopulation-area-decoding/exploratory-analyses/016_pseudopop_halves.py"]
    return ("\n# Appendix\n\n## Parameters\n\n| parameter | value |\n|:---|:---|\n" + "".join(f"| {a} | {b} |\n" for a, b in params)
            + "\n: Table A1. Parameters.\n\n## Scripts and results\n\nCode: `projects/" + SLUG + "/exploratory-analyses/` (ibl-ai-agent "
            "fork); results: the same folder on haas (`~/code/ibl-ai-agent/projects/" + SLUG + "/exploratory-analyses/`, tables and "
            "figures); report: `combined_results_ks4/" + SLUG + "/report/`.\n\n" + "".join(f"* `{s}`\n" for s in scripts)
            + "\n## Version history\n\n* 2026-10-01: invalid (`perf == 6`) trials excluded from every active-trial analysis; "
            "disengagement rule A1 adopted.\n* 2026-10-02: pre-lick window -150 -> -100 ms; placebo, LT and cohort-permutation "
            "analyses rerun.\n* 2026-10-04: per-trial context rule (expert stage only); first article-style report.\n")


def build_sh() -> str:
    return """#!/usr/bin/env bash
# Rebuild report.tex / report.pdf / report.html from report.md (needs Quarto with TinyTeX).
set -euo pipefail
cd "$(dirname "$0")"
quarto pandoc report.md -s -o report.tex
latexmk -pdf -interaction=nonstopmode -quiet report.tex
latexmk -c report.tex >/dev/null 2>&1 || true
quarto pandoc report.md -s --embed-resources --mathjax -c report.css -o report.html --metadata pagetitle="Split decoding report"
echo "built: report.pdf report.html"
"""


CSS = """body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.45;color:#1a1a1a;background:#fff}
img{max-width:100%}table{border-collapse:collapse;font-size:.82em;display:block;overflow-x:auto}td,th{border:1px solid #ddd;padding:2px 6px}
th{background:#f4f4f4}figcaption,caption{font-size:.9em;text-align:left}"""


def main():
    md = sec_front() + sec_intro() + sec_methods() + sec_part1() + sec_results() + sec_discussion() + sec_caveats() + sec_supp() + sec_appendix()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for f in FIGS:
        shutil.copyfile(f, OUT / "figures" / f.name)
        pdf = f.with_suffix(".pdf")
        if pdf.exists():
            shutil.copyfile(pdf, OUT / "figures" / pdf.name)
    md = md.replace("= < 0.001", "< 0.001")
    (OUT / "report.md").write_text(md, encoding="utf-8")
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1), encoding="utf-8")
    (OUT / "build.sh").write_text(build_sh(), encoding="utf-8", newline="\n")
    (OUT / "report.css").write_text(CSS, encoding="utf-8")
    print(OUT, f"{len(FIGS)} figures, {len(NUM)} numbers")


if __name__ == "__main__":
    main()
