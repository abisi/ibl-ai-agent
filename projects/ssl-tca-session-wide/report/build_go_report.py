"""Stand-alone report on the go component (jaw_all main-fit cluster 69 / reference cluster 49) and its change with
learning (user 2026-11-23: report for that component, then the TCA analysis goes on hold). Written by report_lib into
combined_results_ks4/ssl-tca-session-wide/report/candidate_go_cl69/; PDF / HTML with build.sh there. Every number from
the result tables of 013 (_go), 022 and 023, frozen with R.snapshot. Exploratory, uncorrected.

Run: python report/build_go_report.py ; bash <OUT>/report/candidate_go_cl69/build.sh
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
import tca_lib as L  # noqa: E402
from report_lib import Report  # noqa: E402

OUT = L.OUT / "report" / "candidate_go_cl69"
T, F = L.OUT / "tables", L.OUT / "figures" / "aligned_gpu" / "go2"
R = Report(banned=(), spelling={r"\bSomatosensory-(whisker|orofacial|body)\b": r"Ss-\1"})
R.register([
    ("fig-comp", "Figure 1"), ("fig-learn", "Figure 2"),
    ("tbl-n", "Table 1"), ("tbl-comp", "Table 2"), ("tbl-learn", "Table 3"),
    ("eq-pre", "Eq. (1)"), ("eq-cp", "Eq. (2)"), ("eq-split", "Eq. (3)"), ("eq-sim", "Eq. (4)"), ("eq-z", "Eq. (5)"),
    ("eq-auc", "Eq. (6)"), ("eq-pre-auc", "Eq. (7)"), ("eq-d", "Eq. (8)"), ("eq-ols", "Eq. (9)"), ("eq-ratio", "Eq. (10)"),
    ("eq-proj", "Eq. (11)"),
    ("fig-s-ref", "Figure S1"), ("fig-s-reflearn", "Figure S2"), ("fig-s-overview", "Figure S3"),
    ("tbl-s-params", "Table S1"), ("tbl-s-registry", "Table S2")])
num, pv, ref, table, eq = R.num, R.pv, R.ref, R.table, R.eq
MAIN, REF = ("_splitnn", 69), ("_nn", 49)


def figure(*a, width=100, **k):
    return R.figure(*a, **k).replace("{width=100%}", f"{{width={width}%}}")


def crop(src, top, bottom, dst):
    im = Image.open(src)
    im.crop((0, int(im.height * top), im.width, int(im.height * bottom))).save(dst)
    return dst


def P(p):
    return "n/a" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"= {p:.3f}")


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(stats.wilcoxon(v).pvalue) if len(v) >= 5 and np.any(v != 0) else np.nan


def main():
    files = [T / n for n in ("clusters_go.csv", "go2_clusters.csv", "go_learning_stats.csv", "go_learning_sessions.parquet",
                             "sessions.csv", "rank_by_cohort_go.csv")]
    snap = dict(zip([f.name for f in files], R.snapshot(files, OUT)))
    cl = pd.read_csv(snap["clusters_go.csv"]).set_index(["method", "alignment", "cluster"])
    g2 = pd.read_csv(snap["go2_clusters.csv"]).set_index(["method", "alignment", "cluster"])
    st = pd.read_csv(snap["go_learning_stats.csv"])
    S = pd.read_parquet(snap["go_learning_sessions.parquet"])
    ses = pd.read_csv(snap["sessions.csv"])
    rk = pd.read_csv(snap["rank_by_cohort_go.csv"])
    c69, c49 = cl.loc[(MAIN[0], "jaw_all", MAIN[1])], cl.loc[(REF[0], "jaw_all", REF[1])]
    q69, q49 = g2.loc[(MAIN[0], "jaw_all", MAIN[1])], g2.loc[(REF[0], "jaw_all", REF[1])]
    g2r = g2.reset_index()
    n_cl = int(((g2r.method == "_splitnn") & (g2r.alignment == "jaw_all")).sum())

    def s_(meth, k, stage, var):
        return st[(st.method == meth) & (st.cluster == k) & (st.stage == stage) & (st["var"] == var)].iloc[0]

    def txt(meth, k, stage, var, key, fmt="{:+.2f}"):
        r = s_(meth, k, stage, var)
        return (f"R+ median {num(key + '_Rp', r['R+_median'], fmt)} (Wilcoxon p {pv(key + '_Rp_p', r['R+_p'])}, "
                f"n = {num(key + '_Rp_n', int(r['R+_n']))}), R− {num(key + '_Rm', r['R-_median'], fmt)} (p "
                f"{pv(key + '_Rm_p', r['R-_p'])}, n = {num(key + '_Rm_n', int(r['R-_n']))}); R+ vs R− Mann–Whitney p "
                f"{pv(key + '_mw', r['mw_p'])}, Welch p {pv(key + '_we', r['welch_p'])}")

    m0 = S[(S.method == MAIN[0]) & (S.cluster == MAIN[1])]
    d0, ex = m0[m0.stage == "learning"], m0[m0.stage == "expert"]
    rat = {}
    for c in ("R+", "R-"):
        h = d0[d0.cohort == c]
        pr = h[["ratio_first", "ratio_last"]].dropna()
        e = ex[ex.cohort == c]
        rat[c] = dict(first=num(f"ratio_first_{c}", h.ratio_first.median(), "{:.2f}"),
                      last=num(f"ratio_last_{c}", h.ratio_last.median(), "{:.2f}"),
                      p=pv(f"ratio_p_{c}", stats.wilcoxon(pr.ratio_first, pr.ratio_last).pvalue),
                      n=num(f"ratio_n_{c}", len(pr)),
                      exp=num(f"ratio_exp_{c}", (e.mean_W / e.mean_A).median(), "{:.2f}"),
                      nexp=num(f"ratio_nexp_{c}", int(len(e))))
    perf = {c: np.nanmean(np.array(d0[d0.cohort == c].perf_W.tolist(), float), 0) for c in ("R+", "R-")}
    nd0 = {c: num(f"n_d0_{c}", int(((ses.stage == "learning") & (ses.cohort == c)).sum())) for c in ("R+", "R-")}
    nex = {c: num(f"n_ex_{c}", int(((ses.stage == "expert") & (ses.cohort == c)).sum())) for c in ("R+", "R-")}
    rk0 = rk[(rk.stage == "learning") & (rk.alignment == "jaw_all")].set_index(["method", "cohort"]).proposed_rank

    top69 = crop(F / "go2_gallery_jaw_all_splitnn_p1.png", 0.0, 0.168, F / "go2_cl69_row.png")
    top49 = crop(F / "go2_gallery_jaw_all_nn_p1.png", 0.0, 0.168, F / "go2_cl49_row.png")

    md = ['---\ntitle: "A brain-wide lick go component in session-wide TCA and its change with whisker learning '
          '(jaw-aligned cluster 69)"\nsubtitle: "ssl-tca-session-wide, candidate report; exploratory, uncorrected"\n'
          f'date: "{pd.Timestamp.today().date()}"\n---\n']
    md.append("# Abstract\n")
    md.append(
        f"Tensor component analysis (TCA) was fitted to brain-wide Neuropixels activity aligned to jaw-movement onset "
        f"(10-ms bins, all trials of the session) in whisker-learning (day 0) sessions ({nd0['R+']} R+, {nd0['R-']} R−) "
        f"and expert sessions. One component recurs in nearly every mouse ({num('n_mice', int(c69.n_mice))} mice, main "
        f"fit; {num('n_mice_ref', int(c49.n_mice))} in the reference fit): it rises from about 100 ms before jaw onset, "
        f"peaks about {num('peak', c69.peak_ms_median, '{:.0f}')} ms after it, is high on every lick (whisker hits, "
        f"auditory hits and false alarms) and near zero on trials without a lick, including trials with a jaw "
        f"movement but no lick, and it is spread over all recorded area groups. Over the learning session, its weight "
        f"on whisker licks moves towards its weight on auditory licks in R+ mice but not in R− mice "
        f"(whisker-minus-auditory slope: {txt(*MAIN, 'learning', 'd', 'd')}). The whisker / auditory lick ratio goes "
        f"from {rat['R+']['first']} to {rat['R+']['last']} within the R+ session and is ≈ {rat['R+']['exp']} in R+ "
        f"expert sessions, while R− whisker licks stay at the false-alarm level ({rat['R-']['first']}, "
        f"{rat['R-']['last']}, expert {rat['R-']['exp']}). Part of the relative change comes from a decline of the "
        f"auditory-lick weight over the session, which false alarms do not follow either. The component is a "
        f"candidate for a brain-wide go / lick-preparation signal whose engagement by the whisker stimulus increases "
        f"with learning.\n")
    md.append("# Key results\n")
    md.append(
        f"1. **Component.** Jaw-aligned cluster 69 (main fit) / {REF[1]} (reference fit): in "
        f"{num('frac_Rp', 100 * c69.frac_Rp, '{:.0f}')} % of R+ and {num('frac_Rm', 100 * c69.frac_Rm, '{:.0f}')} % of R− "
        f"mice; area profile broad (top group striatum in only {num('top_share', 100 * c69.top_group_share, '{:.0f}')} % "
        f"of members) ({ref('fig-comp')}, {ref('tbl-comp')}).\n"
        f"2. **Lick-specific, stimulus-independent.** AUC lick vs jaw move without lick "
        f"{num('auc_lm', q69.auc_lick_move_median, '{:.2f}')}; false alarm vs correct rejection "
        f"{num('auc_fa', q69.auc_FA_CR_median, '{:.2f}')}; false-alarm weight {num('fa_ratio', q69.fa_ratio_median, '{:.2f}')} "
        f"of hits; passive jaw movements raise it a little (AUC {num('auc_pm', q69.auc_Pmove_median, '{:.2f}')}).\n"
        f"3. **Learning.** The whisker-lick weight gains on the auditory-lick weight over the R+ session and not in R− "
        f"({ref('fig-learn')}B–C, {ref('tbl-learn')}); latency-controlled regression term "
        f"{txt(*MAIN, 'learning', 'b_posW', 'bposW')}.\n"
        f"4. **Whisker / auditory ratio**: R+ {rat['R+']['first']} → {rat['R+']['last']} (p {rat['R+']['p']}) → "
        f"expert {rat['R+']['exp']}; R− {rat['R-']['first']} → {rat['R-']['last']} → expert {rat['R-']['exp']} "
        f"({ref('fig-learn')}D).\n"
        f"5. **Replication**: reference-fit cluster {REF[1]}: {txt(*REF, 'learning', 'd', 'd_ref')} "
        f"({ref('fig-s-reflearn')}).\n")

    md.append("# Introduction\n")
    md.append(
        "In the SSL task, mice that already lick for an auditory tone learn within one session (whisker day 0) to lick "
        "after a whisker deflection; whisker licks are rewarded in the R+ cohort and not in the R− cohort. Each session "
        "has passive blocks before and after the active block, and no-stimulus trials in the active block measure "
        "spontaneous licking (false alarms).\n")
    md.append(
        "A decision to lick (a go signal) should be present on every lick whatever the stimulus, absent when the mouse "
        "does not lick, distinct from a jaw movement that does not become a lick, and present before the movement. If "
        "the whisker stimulus comes to trigger this decision through learning, the go signal on whisker licks should "
        "become like that on auditory licks in R+ mice. The project ssl-tca-session-wide fitted TCA per session and "
        "matched components across mice; this report documents the go-like component found in a jaw-aligned fit and "
        "its change with learning. The analysis is exploratory: quantities are reported without pass / fail criteria.\n")

    md.append("# Methods\n")
    md.append("## Data and tensors\n")
    md.append(
        f"Neuropixels recordings (Kilosort 4, NWB_ks4), stable good + MUA units (137 stable-unit list), cohort from the "
        f"reference sheet; whisker day 0: {nd0['R+']} R+ and {nd0['R-']} R− sessions (one per mouse); expert (whisker "
        f"day 1–4): {nex['R+']} R+ and {nex['R-']} R−. Trials: perf == 6 dropped except passive, warm-up cut, per-trial "
        f"context rule, disengaged tail trimmed; all trial types (whisker, auditory, no-stimulus) of the passive and "
        f"active blocks. Jaw-movement onsets from DLC (per-mouse jaw-onset tables); three mice without them (AB080, "
        f"AB082, MH026) are excluded. Sample sizes in {ref('tbl-n')}.\n")
    md.append(
        "Each trial is aligned to an event $e_k$ and binned in 10-ms bins from −300 to +300 ms. Lick trials (active, "
        "first-lick latency > 0, jaw onset 15 ms to the first lick) and trials with a jaw movement but no lick (onset 15–600 "
        "ms) are aligned to their own jaw onset; trials without a jaw onset in 0–1 s ('still') are aligned to a pseudo "
        "onset, the session's median jaw onset of the lick trials of the same trial type. Spikes in the stimulation "
        "artefact window (−10 to +5 ms around the stimulus) are replaced, before binning, by Poisson spikes at the "
        "trial's baseline rate. The input is")
    md.append(eq("eq-pre", r"X_{u t k} = \frac{r_{u t k} - b_{u k}}{\sigma_u}"))
    md.append(
        "with $r_{utk}$ the rate of unit $u$ in bin $t$ of trial $k$, $b_{uk}$ the trial's baseline (−1.0 to −0.010 s "
        "before trial start) and $\\sigma_u$ the SD of $r - b$ over trials and bins; units with a mean rate < 0.5 Hz "
        "were dropped.\n")
    md.append("## TCA and recurring components\n")
    md.append("A rank-$R$ CP model approximates the tensor:")
    md.append(eq("eq-cp", r"X_{u t k} \approx \sum_{c=1}^{R} n_{u c}\, \tau_{t c}\, w_{k c}"))
    md.append(
        f"(fitted to $X$ of {ref('eq-pre')}). Main fit of {ref('eq-cp')}: each unit is split into increase and decrease channels and a "
        f"non-negative CP model is fitted (HALS), with net unit weight")
    md.append(eq("eq-split", r"n^{\mathrm{net}}_{u c} = n^{+}_{u c} - n^{-}_{u c}"))
    md.append(
        f"Reference fit: non-negative CP on the scaled rates without baseline subtraction. Ranks 1–12, 10 restarts, GPU; "
        f"rank per cohort by the Williams et al. (2018) trade-off (main fit R+ {int(rk0[('_splitnn', 'R+')])}, R− "
        f"{int(rk0[('_splitnn', 'R-')])}; reference R+ {int(rk0[('_nn', 'R+')])}, R− {int(rk0[('_nn', 'R-')])}). "
        f"Components of different mice were compared by")
    md.append(eq("eq-sim", r"s_{ij} = \tfrac{1}{3}\left[\rho(\tau_i, \tau_j) + \rho(a_i, a_j) + \rho(p_i, p_j)\right]"))
    md.append(
        f"($\\rho$ Pearson correlation; $a$ area profile, mean net weight per area group; $p$ trial-class profile) and "
        f"clustered (average linkage, cut at the 95th percentile of within-session similarity, threshold "
        f"{num('thr', c69.threshold, '{:.2f}')}); clusters with ≥ 10 mice kept ({num('n_cl', n_cl)} in the main jaw-aligned "
        f"fit).\n")
    md.append("## Trial classes and quantities\n")
    md.append(
        "Classes: lick W / A / FA (active whisker hit, auditory hit, false alarm on a no-stimulus trial); move (active, "
        "jaw movement without lick); still (active whisker miss, auditory miss, correct rejection without jaw onset); "
        "passive move / still. Trial weights of a member component are expressed as")
    md.append(eq("eq-z", r"z_k = \frac{w_k - \overline{w}_{\mathrm{still}}}{\mathrm{SD}_{\mathrm{active}}(w)}"))
    md.append(
        f"(0 = mean of the active still trials; members of a session averaged). Class separation is the area under the "
        f"ROC curve")
    md.append(eq("eq-auc", r"\mathrm{AUC}(a, b) = P(z_a > z_b)"))
    md.append(
        f"for licks vs jaw moves, false alarms vs correct rejections and passive moves vs passive stills (0.5 = none). "
        f"Before the movement, the data in bins −300 to −20 ms are projected on the neuron vector, $p_k = \\sum_u "
        f"n^{{net}}_u \\sum_{{t < -20}} \\tau_t X_{{utk}}$, and licks are compared with jaw moves (both aligned to a real "
        f"onset) against 50 unit-shuffled vectors:")
    md.append(eq("eq-pre-auc", r"\Delta\mathrm{AUC} = \mathrm{AUC}(p_{\mathrm{lick}}, p_{\mathrm{move}}) - "
                 r"\overline{\mathrm{AUC}(p^{\mathrm{shuf}}_{\mathrm{lick}}, p^{\mathrm{shuf}}_{\mathrm{move}})}"))
    md.append("## Learning measures\n")
    md.append(
        f"With $x_k$ the trial's position in the active block (0–1), the least-squares slopes of $z$ ({ref('eq-z')}) "
        f"over $x$ on whisker licks ($\\beta_W$), auditory licks ($\\beta_A$) and false alarms ($\\beta_{{FA}}$) give the "
        f"contrast")
    md.append(eq("eq-d", r"d = \beta_W - \beta_A"))
    md.append(
        "(auditory licks: same action, association already learned; a general change gives $d \\approx 0$). On lick "
        "trials, per mouse,")
    md.append(eq("eq-ols", r"z_k = \gamma_0 + \gamma_x x_k + \gamma_W W_k + \gamma_F F_k + \gamma_{xW} x_k W_k + "
                 r"\gamma_{xF} x_k F_k + \gamma_\ell \ell_k + \epsilon_k"))
    md.append(
        "with $W$, $F$ indicators of whisker licks and false alarms, $x$ and the lick latency $\\ell$ standardised; "
        "$\\gamma_{xW}$ is the change of whisker licks relative to auditory licks at equal latency. The whisker / "
        "auditory ratio in a third of the block is")
    md.append(eq("eq-ratio", r"q = \overline{z}_{W} \,/\, \overline{z}_{A}"))
    md.append(
        f"and the model-free time course of a trial class is the mean over its trials of the per-bin projection on the net "
        f"unit weights of {ref('eq-split')}")
    md.append(eq("eq-proj", r"P_{k t} = \sum_u n^{\mathrm{net}}_u X_{u t k}"))
    md.append(
        f"(normalised by the peak of the mean over licks; no time factor). Expert sessions: each expert component was "
        f"compared with the day-0 members ({ref('eq-sim')}), the best one per session accepted if it reached the 10th "
        f"percentile of the members' leave-one-out similarity. Statistics per mouse: Wilcoxon against 0 within cohort; "
        f"R+ vs R− Mann–Whitney and Welch; uncorrected. Parameters in {ref('tbl-s-params')}. Scripts: 005 (tensors), "
        f"006g (fits, Kuma), 011–013 (ranks, components, clusters; TCA_SUFFIX=_go), 022 (component quantities), 023 "
        f"(learning), report/build_go_report.py.\n")

    md.append("# Results\n")
    md.append("## A brain-wide component on every lick, before and after jaw onset\n")
    md.append(R.claim("comp",
        f"The component is found in {num('n_mice2', int(c69.n_mice))} mice and has no cohort bias (permutation p "
        f"{pv('coh_p', c69.cohort_bias_p)}). Its time factor rises from about −100 ms and peaks about "
        f"{num('peak2', c69.peak_ms_median, '{:.0f}')} ms after jaw onset; net unit weights are similar across area groups "
        f"({ref('fig-comp')}). Trial weights are high on auditory licks, whisker licks and false alarms and near zero on "
        f"still trials and on jaw movements without a lick (AUC lick vs move {num('auc_lm2', q69.auc_lick_move_median, '{:.2f}')}, "
        f"lick vs still {num('auc_ls', q69.auc_lick_still_median, '{:.2f}')}); false alarms reach "
        f"{num('fa_ratio2', q69.fa_ratio_median, '{:.2f}')} of the hit weight (AUC vs correct rejection "
        f"{num('auc_fa2', q69.auc_FA_CR_median, '{:.2f}')}). Before the movement, the projection separates licks from jaw "
        f"moves only slightly better than shuffled units ({ref('eq-pre-auc')}: "
        f"{num('pre_ex', q69.pre_auc_excess_median, '{:+.3f}')}, p {pv('pre_ex_p', q69.pre_auc_excess_p)}). Weight and "
        f"lick latency are negatively related (ρ {num('rho', q69.rho_lat_median, '{:+.2f}')}, p "
        f"{pv('rho_p', q69.rho_lat_p)}). The reference fit gives the same component (cluster {REF[1]}, "
        f"{ref('fig-s-ref')}).", figure="Fig. 1", stat="AUC, Wilcoxon", n=f"{int(c69.n_mice)} mice") + "\n")
    md.append(figure("fig-comp", top69,
        f"Jaw-aligned main-fit cluster {MAIN[1]} (whisker day 0). Left: time factor, mean over member mice, green R+, "
        f"magenta R−; 0 = jaw onset (still trials at a pseudo onset). Middle: area profile, mean net unit weight per area "
        f"group (area-group colours). Right: trial weight per class ({ref('eq-z')}; lick W / A / FA = whisker hit, "
        f"auditory hit, false alarm; move = jaw movement without lick; still W miss / A miss / CR; P = passive, shaded), "
        f"mean ± SEM over mice; the numbers are the cluster medians of the quantities of {ref('eq-auc')} and "
        f"{ref('eq-pre-auc')}. From the 022 gallery.", width=100, script="022_go_signal_v2.py"))
    rows = []
    for (meth, k), cc, qq in ((MAIN, c69, q69), (REF, c49, q49)):
        rows.append({"Fit": f"{'main' if meth == '_splitnn' else 'reference'} cl. {k}", "Mice": int(cc.n_mice),
                     "Peak (ms)": f"{cc.peak_ms_median:.0f}", "AUC lick/move": f"{qq.auc_lick_move_median:.2f}",
                     "AUC lick/still": f"{qq.auc_lick_still_median:.2f}", "AUC FA/CR": f"{qq.auc_FA_CR_median:.2f}",
                     "FA / hit": f"{qq.fa_ratio_median:.2f}", "AUC P move/still": f"{qq.auc_Pmove_median:.2f}",
                     "Pre-move ΔAUC": f"{qq.pre_auc_excess_median:+.3f} (p {P(qq.pre_auc_excess_p)})",
                     "ρ latency": f"{qq.rho_lat_median:+.2f}"})
    md.append(table("tbl-comp", pd.DataFrame(rows),
        f"Component quantities (medians over mice; {ref('eq-auc')}, {ref('eq-pre-auc')}), both fits, whisker day 0."))

    md.append("## The go component on whisker licks converges to auditory licks in R+ mice\n")
    md.append(R.claim("learn",
        f"Within the R+ active block, the weight on auditory licks declines (slope {txt(*MAIN, 'learning', 'slope_A', 'sA')}) "
        f"while the weight on whisker licks does not ({txt(*MAIN, 'learning', 'slope_W', 'sW')}), so that the contrast "
        f"$d$ ({ref('eq-d')}) is positive in R+ and not in R− ({txt(*MAIN, 'learning', 'd', 'd2')}; {ref('fig-learn')}A–B). "
        f"At equal lick latency, the regression term $\\gamma_{{xW}}$ ({ref('eq-ols')}) gives "
        f"{txt(*MAIN, 'learning', 'b_posW', 'bposW2')} ({ref('fig-learn')}C).", figure="Fig. 2B-C",
        stat="Wilcoxon; MW + Welch", n="R+ / R- mice") + "\n")
    md.append(R.claim("ratio",
        f"The whisker / auditory ratio ({ref('eq-ratio')}) rises in R+ from {rat['R+']['first']} (first third) to "
        f"{rat['R+']['last']} (last third; paired Wilcoxon p {rat['R+']['p']}, n = {rat['R+']['n']}) and is "
        f"{rat['R+']['exp']} in R+ expert sessions (n = {rat['R+']['nexp']}); in R− it stays low ({rat['R-']['first']}, "
        f"{rat['R-']['last']}, p {rat['R-']['p']}; expert {rat['R-']['exp']}, n = {rat['R-']['nexp']}), close to the "
        f"false-alarm level ({ref('fig-learn')}A, D). Binned by the whisker learning curve, the whisker-lick weight "
        f"increases with P(lick | whisker) in R+ (from {num('perf_lo', perf['R+'][0], '{:.2f}')} to "
        f"{num('perf_hi', perf['R+'][-1], '{:.2f}')} SD) and less in R− ({num('perf_lo_m', perf['R-'][0], '{:.2f}')} to "
        f"{num('perf_hi_m', perf['R-'][-1], '{:.2f}')}) ({ref('fig-learn')}E).", figure="Fig. 2D-E",
        stat="paired Wilcoxon", n="R+ / R- mice", status="fragile",
        note="expert values cross-sectional; part of the change is the auditory decline") + "\n")
    md.append(
        f"Model-free time courses ({ref('eq-proj')}, {ref('fig-learn')}F) show that auditory licks rise steeply from "
        f"about −100 ms, whereas whisker licks and false alarms ramp more slowly from about −250 ms; the onset of the "
        f"whisker-lick signal does not shift between the first and last third ({ref('fig-learn')}G). In R+ expert "
        f"sessions, whisker and auditory lick weights decline alike over the session "
        f"({txt(*MAIN, 'expert', 'd', 'd_exp')}; {ref('fig-learn')}H), and false alarms keep their weight relative to "
        f"auditory licks at day 0 as well ({txt(*MAIN, 'learning', 'd_FA', 'dFA')}). The reference fit replicates the "
        f"day-0 result ({txt(*REF, 'learning', 'd', 'd_ref2')}; {ref('fig-s-reflearn')}).\n")
    md.append(figure("fig-learn", F / f"go_learning{MAIN[0]}_cl{MAIN[1]}.png",
        f"Go component and learning (main fit cluster {MAIN[1]}; weights in SD units, {ref('eq-z')}). **A** Session "
        f"course of the trial weight in fifths of the active block for whisker licks (orange), auditory licks (blue), "
        f"false alarms (dark grey, dashed) and jaw moves without lick (light grey), mean ± SEM over mice, R+ and R− day "
        f"0. **B** $d$ per mouse ({ref('eq-d')}). **C** Latency-controlled regression term $\\gamma_{{xW}}$ "
        f"({ref('eq-ols')}). **D** Whisker / auditory lick ratio ({ref('eq-ratio')}), median ± SEM: first and last third "
        f"of day 0 and expert sessions (diamond); paired Wilcoxon first vs last. **E** Whisker-lick weight binned by "
        f"P(lick | whisker). **F** Model-free time course ({ref('eq-proj')}) of whisker licks of the first and last third, "
        f"auditory licks, false alarms and still trials, mean ± SEM over mice. **G** Change of the whisker-lick onset "
        f"(50 % of peak), last − first third. **H** $d$ in expert sessions with a matched component. B, C, G, H: dots = "
        f"mice, bar = median, Wilcoxon p (vs 0) on top, R+ vs R− Mann–Whitney and Welch p in the title.", width=95,
        script="023_go_learning.py"))
    lrows = []
    for stage in ("learning", "expert"):
        for var, lab in (("d", "d = β_W − β_A"), ("slope_W", "β_W (whisker licks)"), ("slope_A", "β_A (auditory licks)"),
                         ("d_FA", "β_FA − β_A"), ("b_posW", "γ_xW (latency controlled)")):
            r = s_(*MAIN, stage, var)
            lrows.append({"Stage": "day 0" if stage == "learning" else "expert", "Measure": lab,
                          "R+ median (p, n)": f"{r['R+_median']:+.2f} ({P(r['R+_p'])}, {int(r['R+_n'])})",
                          "R− median (p, n)": f"{r['R-_median']:+.2f} ({P(r['R-_p'])}, {int(r['R-_n'])})",
                          "MW p": P(r.mw_p), "Welch p": P(r.welch_p)})
    md.append(table("tbl-learn", pd.DataFrame(lrows),
        f"Learning measures of main-fit cluster {MAIN[1]} ({ref('eq-d')}, {ref('eq-ols')}): medians per cohort "
        f"(Wilcoxon p vs 0, n mice), R+ vs R− tests; uncorrected."))

    md.append("# Discussion\n")
    md.append(
        "**What the component is.** A brain-wide signal present on every lick, whatever the stimulus, and absent from "
        "jaw movements that do not become licks; it starts before jaw onset and continues through the lick. This fits a "
        "go / lick-preparation and execution signal shared across areas (or a lick-locked arousal signal). Its "
        "before-movement part separates licks from non-lick jaw movements only weakly beyond shuffled units, so the "
        "decision component cannot be separated from the execution component at this resolution.\n")
    md.append(
        "**What changes with learning.** On day 0, R− whisker licks carry a go signal like spontaneous licks (false "
        "alarms), weaker than auditory licks; in R+ the whisker-lick signal moves towards the auditory-lick level during "
        "the session and matches it in expert sessions. The whisker stimulus therefore comes to engage the same go "
        "signal as the already-learned auditory stimulus, only when whisker licks are rewarded. The steep "
        "stimulus-locked rise of auditory licks vs the slower ramp of whisker licks and false alarms suggests that "
        "learned licks are triggered by the stimulus while the others are self-initiated. This agrees with the "
        "single-neuron whisker-hit → auditory-hit convergence in the jaw-movement period found in ssl-prelick-convergence.\n")
    md.append(
        "**Alternatives.** (i) Satiety: the relative gain in R+ comes mostly from a decline of the auditory-lick signal "
        "over the session that whisker licks and false alarms do not follow; lick vigour (number of licks, bout length) "
        "after reward may drive the post-onset part of the component. (ii) Licks of different kinds (stimulus-triggered "
        "vs self-initiated) differ in their pre-movement ramp independent of learning. (iii) The expert comparison is "
        "cross-sectional.\n")
    md.append("# Caveats and limitations\n")
    md.append(
        "- Exploratory, uncorrected; one cluster examined in detail out of the recurring jaw-aligned clusters.\n"
        "- Pseudo-onsets for still trials; jaw onsets from DLC (thresholded), a fraction of 'still' trials may contain "
        "small movements.\n"
        "- 10-ms bins but only 30 bins before onset; the time factor is shared by all trials, so trial-specific timing "
        "changes appear only in the model-free time courses.\n"
        f"- Expert sessions few ({nex['R+']} R+, {nex['R-']} R−) and only partly the same mice.\n"
        "- Lick vigour and reward consumption not controlled.\n")
    md.append("# Next steps (analysis on hold)\n")
    md.append(
        "1. Control lick vigour: number of licks and bout duration per trial (lick sensor / DLC) as covariates in "
        f"{ref('eq-ols')}.\n"
        "2. Paired day 0 vs expert comparison within mice.\n"
        "3. Area breakdown of the convergence (area-restricted projection and single units, as for the sensory candidate).\n"
        "4. Separate the pre-movement part: TCA on −300 to 0 ms only, or demixed PCA with lick × stimulus × time "
        "marginals.\n"
        "5. Align the convergence to each mouse's learning trial.\n")

    nrows = []
    for c in ("R+", "R-"):
        h = d0[d0.cohort == c]
        nrows.append({"Cohort": c, "Day-0 sessions": int(((ses.stage == "learning") & (ses.cohort == c)).sum()),
                      "Mice with cl. 69": int(c69[f"n_{'Rp' if c == 'R+' else 'Rm'}"]),
                      "Analysed (jaw onsets)": int(len(h)), "Whisker licks (median)": int(h.n_W.median()),
                      "Auditory licks (median)": int(h.n_A.median()), "False alarms (median)": int(h.n_FA.median()),
                      "Expert matched": int(len(ex[ex.cohort == c]))})
    md.append("# Tables\n")
    md.append(table("tbl-n", pd.DataFrame(nrows),
        "Sample sizes: day-0 sessions, mice with the component, mice analysed (jaw onsets available), median trial "
        "counts per class per session, matched expert sessions."))
    md.append("# Supplementary figures\n")
    md.append(figure("fig-s-ref", top49, f"Reference fit (non-negative, rates), jaw-aligned cluster {REF[1]}; panels as "
                     f"{ref('fig-comp')}.", width=100, kind="supplementary", script="022_go_signal_v2.py"))
    md.append(figure("fig-s-reflearn", F / f"go_learning{REF[0]}_cl{REF[1]}.png",
                     f"Learning measures for reference-fit cluster {REF[1]}; panels as {ref('fig-learn')}.", width=95,
                     kind="supplementary", script="023_go_learning.py"))
    md.append(figure("fig-s-overview", F / "go2_overview.png",
                     "All recurring clusters of the go-signal fits (jaw-aligned and stimulus-aligned with no-stimulus "
                     "trials, both fits): lick vs jaw-move AUC against false-alarm vs correct-rejection AUC (left) and "
                     "onset against the pre-movement AUC excess (right); medians over mice, size = n mice, numbers = "
                     "cluster.", width=80, kind="supplementary", script="022_go_signal_v2.py"))
    md.append(f"Supplementary: reference-fit component ({ref('fig-s-ref')}) and learning measures "
              f"({ref('fig-s-reflearn')}); overview of all clusters ({ref('fig-s-overview')}).\n")
    md.append("# Appendix\n")
    md.append(table("tbl-s-params", pd.DataFrame([
        ("Alignment", "jaw onset, −300 to +300 ms, 10-ms bins; still trials at the type-matched median lick onset"),
        ("Jaw move without lick", "onset 15–600 ms, no lick"), ("Still", "no jaw onset in 0–1 s"),
        ("Baseline", "−1.0 to −0.010 s before trial start"), ("Artefact", "−10 to +5 ms around the stimulus, Poisson fill"),
        ("Min unit rate", "0.5 Hz"), ("TCA", "ranks 1–12, 10 restarts; split non-negative (main), non-negative rates (reference)"),
        ("Clusters", "95th pct within-session similarity, ≥ 10 mice"), ("Pre-movement window", "−300 to −20 ms, 50 unit shuffles"),
        ("Slopes", "≥ 8 trials per class"), ("Expert match", "10th pct of members' leave-one-out similarity"),
        ("Statistics", "per mouse; Wilcoxon; Mann–Whitney + Welch; uncorrected")], columns=["Parameter", "Value"]),
        "Parameters."))
    md.append(R.registry_table("tbl-s-registry"))
    md.append(f"Parameters in {ref('tbl-s-params')}; claims in {ref('tbl-s-registry')}. Results: "
              f"combined_results_ks4/ssl-tca-session-wide/ (tables/*_go*, go2_*, go_learning_*; figures/aligned_gpu/go2/). "
              f"Code: projects/ssl-tca-session-wide/ (005, 006g, 011–013, 022, 023, report/build_go_report.py).\n")
    text = "\n".join(md)
    summary = "# Go component cl. 69: summary\n\n" + text.split("# Key results\n")[1].split("# Introduction")[0]
    R.write(OUT, text, summary=summary)


if __name__ == "__main__":
    main()
