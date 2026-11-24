"""Article report of ssl-tca-session-wide (skills/project-report): report.md (+ numbers.json, figures/) written into
combined_results_ks4/ssl-tca-session-wide/report/ by report_lib; report.pdf / report.html built there with build.sh.
Every number comes from the result tables (frozen with R.snapshot). Exploratory, uncorrected (user 2026-11-22).

Run: python report/build_report.py ; bash <OUT>/report/build.sh
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
import tca_lib as L  # noqa: E402
from report_lib import Report  # noqa: E402

OUT = L.OUT / "report"
T, F = L.OUT / "tables", L.OUT / "figures" / "aligned_gpu"
EX = F / "AB123_20240806_110231"
R = Report(banned=(), spelling={r"\bSomatosensory-(whisker|orofacial|body)\b": r"Ss-\1"})
R.register([("fig-example", "Figure 1"), ("fig-rank", "Figure 2"), ("fig-clusters", "Figure 3"), ("fig-cand", "Figure 4"),
            ("tbl-align", "Table 1"), ("tbl-rank", "Table 2"), ("tbl-kinds", "Table 3"), ("tbl-behav", "Table 4"),
            ("tbl-agree", "Table 5"),
            ("eq-pre", "Eq. (1)"), ("eq-cp", "Eq. (2)"), ("eq-split", "Eq. (3)"), ("eq-net", "Eq. (4)"), ("eq-dir", "Eq. (5)"),
            ("eq-err", "Eq. (6)"), ("eq-gain", "Eq. (7)"), ("eq-ctrl", "Eq. (8)"), ("eq-excess", "Eq. (9)"),
            ("eq-sim", "Eq. (10)"),
            ("fig-s-example-ref", "Figure S1"), ("fig-s-variants", "Figure S2"), ("fig-s-clusters-ref", "Figure S3"),
            ("fig-s-sensory", "Figure S4"),
            ("tbl-s-sizes", "Table S1"), ("tbl-s-params", "Table S2"), ("tbl-s-registry", "Table S3")])
num, pv, ref, table, eq = R.num, R.pv, R.ref, R.table, R.eq


def figure(*a, width=100, **k):
    """R.figure with a narrower width for tall figures (stay within one page)"""
    return R.figure(*a, **k).replace("{width=100%}", f"{{width={width}%}}")
AL = ["stim_long", "stim_sensory", "jaw", "lick"]
METH = {"_splitnn": "main (split non-neg.)", "_nn": "reference (non-neg., rates)"}


def p_txt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def main():
    files = [T / n for n in ("rank_by_cohort.csv", "clusters.csv", "cluster_members.parquet", "components.parquet",
                             "method_agreement.csv", "aligned_build_log.csv")]
    snap = dict(zip([f.name for f in files], R.snapshot(files, OUT)))
    rk = pd.read_csv(snap["rank_by_cohort.csv"])
    cl = pd.read_csv(snap["clusters.csv"])
    comp = pd.read_parquet(snap["components.parquet"])
    ag = pd.read_csv(snap["method_agreement.csv"])
    blog = pd.read_csv(snap["aligned_build_log.csv"])
    C = cl.set_index(["method", "alignment", "cluster"])
    A = ag.set_index(["alignment", "split_cluster"])
    fits = sorted((L.OUT / "artifacts" / "tca_aligned_gpu").glob("*/*_zscore_*.npz"))
    n_fit = sum(f.stem.endswith(("_zscore_splitnn", "_zscore_nn")) for f in fits)
    ses = comp.drop_duplicates(["session_id", "method", "alignment"])
    s_main = ses[(ses.method == "_splitnn") & (ses.alignment == "stim_long")]
    n_ses = len(blog)
    n_d0 = {c: int(((s_main.stage == "learning") & (s_main.cohort == c)).sum()) for c in ("R+", "R-")}
    n_exp = int((s_main.stage == "expert").sum())
    n_comp = len(comp)
    n_nojaw = int((blog.jaw_trials == 0).sum())
    units_med = float(s_main.n_units.median())

    md = []
    md.append('---\ntitle: "Session-wide tensor component analysis of brain-wide Neuropixels activity during whisker learning"\n'
              'subtitle: "ssl-tca-session-wide: main fit (split non-negative, baseline-subtracted) and reference fit '
              '(non-negative, rates); exploratory, uncorrected"\n'
              f'date: "{pd.Timestamp.today().date()}"\n---\n')

    # ------------------------------------------------------------------ abstract + key results
    k67, k56, k68 = C.loc[("_splitnn", "stim_long", 67)], C.loc[("_splitnn", "stim_long", 56)], C.loc[("_splitnn", "stim_sensory", 68)]
    r44 = C.loc[("_nn", "stim_long", int(A.loc[("stim_long", 67), "ref_cluster"]))]
    r40 = C.loc[("_nn", "stim_long", int(A.loc[("stim_long", 56), "ref_cluster"]))]
    r39 = C.loc[("_nn", "stim_sensory", int(A.loc[("stim_sensory", 68), "ref_cluster"]))]
    n_rec = cl.groupby(["method", "alignment"]).size()
    md.append("# Abstract\n")
    md.append(f"We decomposed the activity of all stable units of each session (Kilosort 4, {num('units_med', units_med, '{:.0f}')} units "
              f"per session, median) into trial × unit × time components with tensor component analysis (TCA), in "
              f"{num('n_ses', n_ses)} sessions (whisker day 0: {num('n_d0_rp', n_d0['R+'])} R+ and {num('n_d0_rm', n_d0['R-'])} R− "
              f"mice; {num('n_exp', n_exp)} expert sessions as reference) and four event alignments (stimulus, 1.7 s and "
              f"0.16 s windows; jaw-movement onset; first lick). The main fit is a split non-negative TCA of single-trial "
              f"baseline-subtracted activity, which keeps increases and decreases from baseline in separate channels; a "
              f"reference fit, non-negative TCA of the firing rates without baseline subtraction, checks that conclusions do "
              f"not depend on these choices. Components recurring across mice were found by clustering "
              f"({num('n_comp', n_comp)} components). Most stimulus-aligned clusters separate hits from misses, as expected "
              f"from licking, and most of their correlation with the learning curve disappears once the outcome of each trial "
              f"is controlled. Two stimulus-aligned components, a broad response peaking at about "
              f"{num('k67_peak', k67.peak_ms_median, '{:.0f}')} ms and a late one at about {num('k56_peak', k56.peak_ms_median, '{:.0f}')} ms, "
              f"grow with learning beyond outcome in R+ but not in R− mice, in both fits; an early (≈ "
              f"{num('k68_peak', k68.peak_ms_median, '{:.0f}')} ms) sensory component weakens with learning in R+ in both fits, "
              f"but its cohort difference holds only in the main fit. All tests are exploratory and uncorrected.\n")
    md.append("**Key results**\n")
    md.append(
        f"1. The restart-similarity / error-decrease rule proposes ranks of 8–12 per cohort and alignment, nearly equal in R+ "
        f"and R−; for the split fit of the jaw and lick windows it reaches the maximum rank tested (12) ({ref('tbl-rank')}, {ref('fig-rank')}).\n"
        f"2. {num('n_rec_min', int(n_rec.min()))}–{num('n_rec_max', int(n_rec.max()))} clusters recur in ≥ 10 day-0 mice per fit and "
        f"alignment: global stimulus and movement components, a few area-specific ones (e.g. a fast Ss-whisker-led "
        f"transient), and, in the main fit, near-symmetric variability components ({ref('tbl-kinds')}, {ref('fig-clusters')}).\n"
        f"3. Learning effects are mostly outcome effects: in the main stim_long fit, clusters with a learning correlation "
        f"fall from {num('lr_rp_raw', int((cl[(cl.method == '_splitnn') & (cl.alignment == 'stim_long')]['all_learn_R+_wilcoxon_p'] < .05).sum()))} "
        f"to {num('lr_rp_ctrl', int((cl[(cl.method == '_splitnn') & (cl.alignment == 'stim_long')]['all_learnctrl_R+_wilcoxon_p'] < .05).sum()))} "
        f"in R+ when outcome is controlled ({ref('tbl-behav')}).\n"
        f"4. The ≈ {k67.peak_ms_median:.0f} ms component grows with learning beyond outcome in R+ (median excess "
        f"{num('k67_rp', k67['all_learnctrl_R+_median'], '{:+.2f}')}, Wilcoxon p {pv('p_k67_rp', k67['all_learnctrl_R+_wilcoxon_p'])}) "
        f"but not R− (p {pv('p_k67_rm', k67['all_learnctrl_R-_wilcoxon_p'])}); R+ vs R− Mann–Whitney p "
        f"{pv('p_k67_mw', k67['all_learnctrl_RpRm_mw_p'])}, Welch p {pv('p_k67_w', k67['all_learnctrl_RpRm_welch_p'])}; "
        f"reference fit R+ vs R− Mann–Whitney p {pv('p_r44_mw', r44['all_learnctrl_RpRm_mw_p'])} ({ref('fig-cand')}).\n"
        f"5. The ≈ {k56.peak_ms_median:.0f} ms component (lick / reward period) rises with learning in R+ and falls in R− "
        f"(R+ vs R− Mann–Whitney p {pv('p_k56_mw', k56['all_learnctrl_RpRm_mw_p'])}, Welch p "
        f"{pv('p_k56_w', k56['all_learnctrl_RpRm_welch_p'])}; reference p {pv('p_r40_mw', r40['all_learnctrl_RpRm_mw_p'])}) "
        f"({ref('fig-cand')}, {ref('tbl-agree')}).\n")

    # ------------------------------------------------------------------ introduction
    md.append("# Introduction\n")
    md.append("Learning the whisker contingency changes activity in many brain areas, but single-unit and population analyses "
              "usually average over trials or fix the time window in advance. Tensor component analysis (TCA; Williams et al. "
              "2018, Neuron) instead describes a session as a small number of components, each the product of a unit "
              "pattern, a within-trial time course and a trial weight; the trial weights then follow slow changes across the "
              "session, such as learning, without assuming when or where they happen.\n")
    md.append("The questions are: which components recur across mice (global, area-specific, or cohort-specific), and do "
              "any of them follow performance or learning, beyond the trivial fact that hit trials contain licking. "
              "Because a decomposition's components depend on preprocessing and model constraints, every candidate result is "
              "checked in a second, independent fit (the reference fit, see Methods).\n")

    # ------------------------------------------------------------------ methods
    md.append("# Methods\n")
    md.append("## Data, trials and units\n")
    md.append(R.methods("task", "trials", "units"))
    md.append(f"Additionally, active trials with an early lick were removed; passive trials (before and after the active "
              f"block) were kept. Sessions: all whisker day-0 sessions of included mice and expert sessions as reference "
              f"({num('n_ses2', n_ses)} sessions; sizes in {ref('tbl-s-sizes')}). Spikes within −10 to +5 ms of every trial "
              f"start (the stimulus artefact window) were removed.\n")
    md.append("## Event-aligned tensors\n")
    md.append(f"One builder (005_build_aligned.py) produced, for each session, firing-rate tensors in unsmoothed bins for four "
              f"alignments ({ref('tbl-align')}). Bins that overlap the artefact window take their rate over the valid part of "
              f"the bin only. The sensory window uses a 10-ms grid shifted by −2.5 ms so that no bin lies inside the artefact "
              f"window; the two bins touching it keep 2.5 ms of valid time each. Jaw onsets come from the per-mouse DLC table "
              f"(valid onset 15 ms after the stimulus or later, and before the first lick); {num('n_nojaw', n_nojaw)} sessions "
              f"have no jaw-onset data and were fitted on the stimulus alignments only. Units with a mean rate below 0.5 Hz "
              f"were dropped.\n")
    tal = pd.DataFrame([["stim_long", "stimulus onset", "−0.2 to +1.5 s", "50 ms (34)", "whisker + auditory, active + passive"],
                        ["stim_sensory", "stimulus onset", "−52.5 to +107.5 ms", "10 ms (16)", "whisker + auditory, active + passive"],
                        ["jaw", "jaw-movement onset (DLC)", "−0.3 to +0.3 s", "50 ms (12)", "active lick trials with valid jaw onset"],
                        ["lick", "corrected first lick", "−0.3 to +0.3 s", "50 ms (12)", "same trials as jaw"]],
                       columns=["alignment", "event", "window", "bin (n bins)", "trials"])
    md.append(table("tbl-align", tal, "Alignments. No-stimulus trials are excluded from the stimulus alignments. The corrected "
                    "first lick is start_time + (lick_time − response_window_start_time)."))
    md.append("## Preprocessing\n")
    md.append("For the main fit, each entry is the single-trial baseline-subtracted rate, scaled per unit:")
    md.append(eq("eq-pre", r"D_{ijk} = \frac{r_{ijk} - b_{ij}}{s_j}"))
    md.append(f"where $r_{{ijk}}$ is the rate of unit $j$ in trial $i$ and time bin $k$, $b_{{ij}}$ the rate of the same unit "
              f"from −1.0 to −0.010 s before the start of the same trial (jaw- and lick-aligned tensors borrow this "
              f"pre-trial baseline), and $s_j$ the standard deviation of $r_{{ijk}} - b_{{ij}}$ over trials and bins "
              f"(scaling only, no centring). The reference fit uses $r_{{ijk}} / s_j$ with $s_j$ the standard deviation of the "
              f"rates (no baseline subtraction, so all entries are ≥ 0; {ref('eq-pre')} without $b_{{ij}}$). No capping and no "
              f"outlier-trial removal were applied.\n")
    md.append("## Decomposition: main fit and reference fit\n")
    md.append("A rank-$R$ CP / TCA model approximates a trial × unit × time tensor $X$ by")
    md.append(eq("eq-cp", r"\hat X_{ijk} = \sum_{r=1}^{R} a_{ir}\, w_{jr}\, c_{kr}"))
    md.append(f"with trial weights $a_{{ir}}$, unit weights $w_{{jr}}$ and time weights $c_{{kr}}$ ({ref('eq-cp')}). An "
              f"unconstrained (signed) fit of baseline-subtracted data produced pairs of components with near-identical "
              f"time and trial profiles and opposite unit weights that cancel each other (CP degeneracy); ridge "
              f"regularisation and a semi-non-negative model (unit weights signed, trial and time weights ≥ 0) did not "
              f"remove them in the pilot mice ({ref('fig-s-variants')}).\n")
    md.append("**Main fit: split non-negative TCA.** Each unit is split into an increase and a decrease channel,")
    md.append(eq("eq-split", r"X = \big[\, \max(D, 0) \;\big|\; \max(-D, 0) \,\big] \in \mathbb{R}_{\ge 0}^{\,I \times 2J \times K}"))
    md.append(f"and a non-negative CP model ({ref('eq-cp')}, all factors ≥ 0) is fitted to $X$ ({ref('eq-split')}). A unit "
              f"then has two weights per component, $w^+_{{jr}}$ (increase) and $w^-_{{jr}}$ (decrease). Its reported weight is "
              f"the net weight")
    md.append(eq("eq-net", r"w^{\mathrm{net}}_{jr} = w^{+}_{jr} - w^{-}_{jr}"))
    md.append(f"positive when the unit rises above baseline with the component and negative when it falls below ({ref('eq-net')}). "
              f"The directionality of a component is the median, over the top half of units by total weight, of")
    md.append(eq("eq-dir", r"d_{jr} = \frac{w^{+}_{jr} - w^{-}_{jr}}{w^{+}_{jr} + w^{-}_{jr}}"))
    md.append(f"(±1: one-sided change; ≈ 0: the unit loads equally on both channels; {ref('eq-dir')}). Components with "
              f"|median $d$| < 0.5 are labelled *variability*: they capture rectified trial-to-trial fluctuations around "
              f"baseline, not a directed response.\n")
    md.append("**Reference fit: non-negative TCA on rates.** The reference fit is the standard non-negative TCA of Williams "
              "et al. (2018): the same tensors, alignments, trials, ranks, restarts and fitting algorithm, but the input is the "
              "scaled firing rate without baseline subtraction and without the channel split. It makes none of the main "
              "fit's two choices (single-trial baseline subtraction, increase / decrease split), so a result found in both "
              "fits does not depend on them. Its components are rate patterns, so slow baseline changes across the session "
              "appear as flat time courses (labelled *flat*) instead of being removed beforehand.\n")
    md.append("**Fitting.** Both fits use non-negative CP by hierarchical alternating least squares (tensorly "
              "non_negative_parafac_hals, PyTorch backend, float32) on Kuma GPUs (L40S and H100 slices), ranks 1–12, 10 random "
              "restarts per rank, tolerance $10^{-6}$, at most 500 iterations; the restart with the lowest error is kept. The "
              "normalised reconstruction error is")
    md.append(eq("eq-err", r"E_R = \frac{\lVert D - \hat D_R \rVert_F}{\lVert D \rVert_F}"))
    md.append(f"computed on the signed data $D$ for the main fit (increase minus decrease reconstruction) and on the rates for "
              f"the reference fit ({ref('eq-err')}). A GPU port of the original CPU pipeline (tensortools) was validated on "
              f"one session (006v_compare_gpu.py; same errors, 40–90× faster).\n")
    md.append("## Rank selection\n")
    md.append(f"Following Williams et al. (2018), the rank trades the error decrease against the reproducibility of the "
              f"components across restarts. Restart similarity is the tensortools kruskal_align score: the mean over the "
              f"three modes of the absolute cosine between matched factors (Hungarian matching), averaged over matched pairs "
              f"and over restarts relative to the best one. The relative error gain of adding the $R$-th component is")
    md.append(eq("eq-gain", r"g_R = \frac{E_{R-1} - E_R}{E_1 - E_{12}}"))
    md.append(f"({ref('eq-gain')}). Per cohort (whisker day 0) and per fit and alignment, the medians over sessions of $E_R$ "
              f"({ref('eq-err')}) and of the similarity were computed, and the proposed rank is the largest $R$ with median "
              f"similarity ≥ 0.85 and $g_R$ ≥ 0.05 (011_rank_by_cohort.py). Expert sessions are shown for reference only. "
              f"All later analyses use the cohort's rank in every session of that cohort.\n")
    md.append("## Component descriptors\n")
    md.append("For every component (012_component_table.py): time course peak latency, width at half maximum and flatness "
              "(coefficient of variation of the time weights; < 0.2 = flat); area profile (mean net unit weight, "
              f"{ref('eq-net')}, per area group) with the top group and its specificity (|top-group mean| / Σ |group means|, "
              "p from 300 permutations of area labels across units); mean trial weight per trial class (active / passive × "
              "whisker / auditory × lick / no lick).\n")
    md.append("## Relations to performance and learning\n")
    md.append("Three per-component measures use the trial weights $a_{ir}$ in trial order:\n")
    md.append("- *Hit / miss*: area under the ROC curve of the trial weights for whisker hits vs misses (stimulus alignments; "
              "≥ 5 of each).\n"
              "- *Learning*: Spearman correlation of the trial weights with the whisker learning curve (exact curve, "
              "$p^{\\mathrm{wh}}_i$, P(lick) on whisker trials), both residualised on trial index (active whisker trials for the "
              "stimulus alignments, whisker hits for jaw / lick).\n"
              "- *Learning beyond outcome* (stimulus alignments): the learning curve is a smoothed hit rate, so a component that "
              "is simply larger on hit trials correlates with it. The trial weights are therefore also residualised on the "
              "trial's own outcome $y_i$ (lick 0 / 1):\n")
    md.append(eq("eq-ctrl", r"\rho^{\mathrm{ctrl}} = \operatorname{corr}_S\!\big(\, a_i - \hat a_i(i, y_i),\; p^{\mathrm{wh}}_i - \hat p_i(i) \,\big)",
                 indent="  "))
    md.append(f"  where $\\hat a_i$ and $\\hat p_i$ are linear least-squares fits on trial index $i$ (and $y_i$ for the "
              f"weights), and corr$_S$ is the Spearman correlation ({ref('eq-ctrl')}).\n")
    md.append(R.methods("shift_null"))
    md.append("Up to 100 shifts were used. Each measure is reported as its excess over the null,")
    md.append(eq("eq-excess", r"\Delta = m_{\mathrm{real}} - \overline{m}_{\mathrm{shift}}"))
    md.append(f"for the hit / miss AUC, the learning correlation and $\\rho^{{\\mathrm{{ctrl}}}}$ ({ref('eq-ctrl')}, "
              f"{ref('eq-excess')}).\n")
    md.append("## Recurring components across mice\n")
    md.append("Per fit and alignment, the components of all whisker day-0 sessions (one per mouse) were clustered "
              "(013_recurring_components.py). The similarity of two components of different mice is")
    md.append(eq("eq-sim", r"S = \tfrac{1}{3}\big[\, r(c, c') + r(\bar w^{\mathrm{area}}, \bar w'^{\,\mathrm{area}}) + r(\bar a^{\mathrm{class}}, \bar a'^{\,\mathrm{class}}) \,\big]"))
    md.append(f"the mean of the Pearson correlations of the time courses, of the area profiles (area groups present in both; "
              f"≥ 3) and of the trial-class profiles ({ref('eq-sim')}); Pearson (centred) rather than cosine, because "
              f"non-negative profiles have high cosines for almost any pair. Average-linkage hierarchical clustering on "
              f"1 − $S$ (components of the same session never merged) was cut at the 95th percentile of the similarity "
              f"between distinct components of the same session. Clusters with members in ≥ 10 mice were kept and labelled "
              f"variability (main fit, |median directionality| < 0.5, {ref('eq-dir')}), flat (median flatness < 0.2), "
              f"area-specific (≥ 50 % of members share a top group with specificity p < 0.05) or global.\n")
    md.append("## Statistics\n")
    md.append(f"Per cluster, measures are averaged per mouse (members of the same mouse). Within cohort: one-sample Wilcoxon "
              f"signed-rank test of the per-mouse excess ({ref('eq-excess')}) against 0; R+ vs R−: Mann–Whitney U and Welch "
              f"t tests, both reported. Cohort bias of membership: difference in the fraction of R+ and R− mice with a member, "
              f"p from 2,000 permutations of cohort labels across mice; for area-led clusters the fraction of mice with units "
              f"in the top area group is reported per cohort. Populations: all mice, and learners (sheet learning category). "
              f"The mouse (one day-0 session) is the unit of analysis. All tests are exploratory and uncorrected (user "
              f"decision): about {num('n_clusters_total', len(cl))} clusters × 3 measures × several tests are reported, so "
              f"p < 0.05 results are expected by chance.\n")
    md.append("**Agreement between fits.** For each main-fit cluster, the best-matching reference cluster of the same "
              f"alignment was found by the similarity of the cluster means ({ref('eq-sim')}; 014_method_agreement.py). A "
              "main-fit result is said to hold in the reference fit when the matched cluster shows the same effect.\n")

    # ------------------------------------------------------------------ results
    md.append("# Results\n")
    md.append("## One session: what the components look like\n")
    md.append(f"In the example R+ day-0 session ({ref('fig-example')}), the stimulus-aligned main fit separates a fast "
              f"evoked transient, broader sustained responses that appear on active trials once the mouse licks, a late "
              f"component in the lick / reward period, a component active only in the first passive block, and a flat "
              f"variability component on passive trials. The reference fit of the same session ({ref('fig-s-example-ref')}) "
              f"recovers the evoked and response components and adds flat components that follow the slow rate changes "
              f"across the session.\n")
    md.append(figure("fig-example", EX / "factors_AB123_20240806_110231_stim_long_zscore_splitnn_r7.png",
                     "Main fit (split non-negative TCA, single-trial baseline-subtracted, Eq. 1–4) of one R+ whisker day-0 "
                     "session (AB123, 201 trials × 1,014 units), aligned to the stimulus (stim_long), rank 7 (pilot fixed "
                     "rank; the R+ cohort rank is 8). Rows: components C1–C7. Left: net unit weight (Eq. 4) per unit, up = "
                     "increase, down = decrease, coloured by area group (area palette, groups in default order). Middle: time "
                     "weight, 0 = stimulus onset. Right: trial weight per trial in session order; orange = whisker, blue = "
                     "auditory; filled = lick, open = no lick; light colours and grey background = passive blocks. Top right: "
                     "P(lick) per trial type.", script="007_aligned_figures.py"))
    md.append("## Rank per cohort\n")
    rt = rk[rk.stage == "learning"].pivot_table(index="alignment", columns=["method", "cohort"], values="proposed_rank")
    trows = []
    for a in AL:
        trows.append([a] + [f"{int(rt.loc[a, (m, 'R+')])} / {int(rt.loc[a, (m, 'R-')])}" for m in ("_splitnn", "_nn")])
    md.append(f"The proposed ranks ({ref('tbl-rank')}, {ref('fig-rank')}) are similar in R+ and R− and between fits (8–12). "
              f"The rule constrains the rank weakly: median restart similarity stays above 0.85 up to rank 12 in every "
              f"alignment, so the choice rests on the 5 % gain criterion ({ref('eq-gain')}), and for the main fit of the "
              f"jaw and lick windows it reaches 12, the highest rank fitted. Error curves of R+ and R− overlap; expert "
              f"sessions have slightly lower errors.\n")
    md.append(table("tbl-rank", pd.DataFrame(trows, columns=["alignment", "main fit, R+ / R−", "reference fit, R+ / R−"]),
                    "Proposed rank per cohort (whisker day 0; rule in Eq. 7: median restart similarity ≥ 0.85 and error gain "
                    "≥ 0.05). Source: tables/rank_by_cohort.csv."))
    md.append(figure("fig-rank", F / "rank_by_cohort.png",
                     "Rank selection per cohort. Columns: alignment × fit (left four: main fit; right four: reference fit). "
                     "Rows: median relative error (normalised to rank 1, Eq. 6), median restart similarity (dotted line = "
                     "0.85), and error decrease per added component as a fraction of the decrease from rank 1 to 12 (Eq. 7; "
                     "dotted line = 0.05). Green = R+, magenta = R− (whisker day 0, solid; expert sessions dotted, not used "
                     "for the choice). Vertical lines = proposed rank per cohort. n sessions in Table S1.",
                     script="011_rank_by_cohort.py"))
    md.append("## Recurring components across mice\n")
    kinds = cl.groupby(["method", "alignment"]).kind.value_counts().unstack(fill_value=0)
    krows = []
    for m in ("_splitnn", "_nn"):
        for a in AL:
            k = kinds.loc[(m, a)] if (m, a) in kinds.index else pd.Series(dtype=int)
            krows.append([METH[m], a, int(k.sum())] + [int(k.get(x, 0)) for x in ("global", "area-specific", "flat", "variability")]
                         + [int((cl[(cl.method == m) & (cl.alignment == a)].cohort_bias_p < 0.05).sum())])
    kt = pd.DataFrame(krows, columns=["fit", "alignment", "clusters", "global", "area-specific", "flat", "variability",
                                      "cohort-biased (p < 0.05)"])
    k45 = C.loc[("_splitnn", "stim_long", 45)]
    md.append(f"Clustering the components of the day-0 mice gave {int(n_rec.min())}–{int(n_rec.max())} clusters with members in ≥ 10 "
              f"mice per fit and alignment ({ref('tbl-kinds')}). Most are global: the same time course and trial-class profile "
              f"with weights spread over many area groups ({ref('fig-clusters')}). Area-specific clusters are rare; the clearest "
              f"is a fast whisker-evoked transient led by Ss-whisker (stim_long cl. 45, {num('k45_n', int(k45.n_mice))} mice, "
              f"peak {num('k45_peak', k45.peak_ms_median, '{:.0f}')} ms), present also on passive whisker trials. The main fit "
              f"has variability clusters (directionality ≈ 0, {ref('eq-dir')}) and almost no flat ones; the reference fit has "
              f"flat clusters (slow rate drifts) and, by construction, no variability ones.\n")
    md.append(table("tbl-kinds", kt, "Recurring clusters (members in ≥ 10 whisker day-0 mice) per fit and alignment, by kind, "
                    "and the number with a cohort bias in membership (permutation p < 0.05, uncorrected). Source: "
                    "tables/clusters.csv."))
    md.append(figure("fig-clusters", F / "clusters_splitnn_stim_long.png",
                     f"Recurring components of the main fit, stimulus-aligned long window (whisker day 0: "
                     f"{n_d0['R+']} R+, {n_d0['R-']} R− mice). Rows: clusters with members in ≥ 10 mice (label: cluster, "
                     f"kind, number of mice). Columns: mean time factor (green R+, magenta R−); mean net unit weight per "
                     f"area group (Eq. 4); mean trial weight per class (orange whisker, blue auditory; dark = lick, light = "
                     f"no lick, palest = passive); fraction of mice of each cohort with a member (p: cohort-label "
                     f"permutation, 2,000); per-mouse excess (Eq. 9) of the hit / miss AUC and of the outcome-controlled "
                     f"learning correlation (Eq. 8) over the linear-shift null, black bar = median. Uncorrected.",
                     script="013_recurring_components.py", width=72))
    md.append("## Performance and learning: mostly outcome\n")
    brows = []
    for m in ("_splitnn", "_nn"):
        for a in ("stim_long", "stim_sensory"):
            g = cl[(cl.method == m) & (cl.alignment == a)]
            brows.append([METH[m], a, len(g)] + [int((g[f"all_{x}_{c}_wilcoxon_p"] < 0.05).sum())
                                                 for x in ("perf", "learn", "learnctrl") for c in ("R+", "R-")])
    bt = pd.DataFrame(brows, columns=["fit", "alignment", "clusters", "hit/miss R+", "hit/miss R−", "learning R+",
                                      "learning R−", "beyond outcome R+", "beyond outcome R−"])
    md.append(f"In the stimulus alignments, most clusters separate hit from miss trials in both cohorts ({ref('tbl-behav')}). "
              f"This is expected: hit trials contain licking and, in R+, reward, which recruit activity in most areas. Raw "
              f"correlations with the learning curve are also frequent, but most vanish when the trial's outcome is "
              f"controlled ({ref('eq-ctrl')}): the learning curve is a smoothed hit rate, so these correlations re-express "
              f"the hit / miss difference. The jaw and lick alignments use lick trials only; their learning correlations "
              f"are rare (≤ 1 cluster per cohort in the main fit).\n")
    md.append(table("tbl-behav", bt, "Number of stimulus-aligned clusters with a per-mouse excess (Eq. 9) different from 0 "
                    "(Wilcoxon p < 0.05, uncorrected, all day-0 mice): hit / miss AUC, learning correlation residualised on "
                    "trial index, and learning beyond outcome (Eq. 8). Source: tables/clusters.csv."))
    md.append("## Candidate learning components, checked in the reference fit\n")
    md.append(R.claim("k67", f"A broad stimulus-aligned component (peak ≈ {k67.peak_ms_median:.0f} ms, increase, thalamus with "
                      f"the largest mean weight) grows with learning beyond outcome in R+ (median excess "
                      f"{k67['all_learnctrl_R+_median']:+.2f}, Wilcoxon p {pv('p_k67_rp2', k67['all_learnctrl_R+_wilcoxon_p'])}, "
                      f"{num('k67_nrp', int(k67['all_learnctrl_R+_n']))} mice; learners p "
                      f"{pv('p_k67_lrn', k67['learners_learnctrl_R+_wilcoxon_p'])}) but not in R− "
                      f"({num('k67_rm', k67['all_learnctrl_R-_median'], '{:+.2f}')}, p {p_txt(k67['all_learnctrl_R-_wilcoxon_p'])}, "
                      f"{num('k67_nrm', int(k67['all_learnctrl_R-_n']))} mice); R+ vs R− Mann–Whitney p "
                      f"{p_txt(k67['all_learnctrl_RpRm_mw_p'])}, Welch p {p_txt(k67['all_learnctrl_RpRm_welch_p'])}. The matched "
                      f"reference cluster (similarity {num('k67_sim', A.loc[('stim_long', 67), 'match_similarity'], '{:.2f}')}) "
                      f"shows the same (R+ {num('r44_rp', r44['all_learnctrl_R+_median'], '{:+.2f}')}, R+ vs R− Mann–Whitney p "
                      f"{p_txt(r44['all_learnctrl_RpRm_mw_p'])}, Welch p {pv('p_r44_w', r44['all_learnctrl_RpRm_welch_p'])}), "
                      f"with striatum rather than thalamus as top group ({ref('fig-cand')}).",
                      "Figure 4 row 1", "Wilcoxon; MW + Welch", f"{int(k67['all_learnctrl_R+_n'])} R+ / {int(k67['all_learnctrl_R-_n'])} R− mice",
                      "fragile", "uncorrected; present in a third of mice") + "\n")
    md.append(R.claim("k56", f"A late component (peak ≈ {k56.peak_ms_median:.0f} ms, lick / reward period) rises with learning "
                      f"beyond outcome in R+ ({num('k56_rp', k56['all_learnctrl_R+_median'], '{:+.2f}')}, p "
                      f"{pv('p_k56_rp', k56['all_learnctrl_R+_wilcoxon_p'])}) and falls in R− "
                      f"({num('k56_rm', k56['all_learnctrl_R-_median'], '{:+.2f}')}, p {pv('p_k56_rm', k56['all_learnctrl_R-_wilcoxon_p'])}); "
                      f"R+ vs R− Mann–Whitney p {p_txt(k56['all_learnctrl_RpRm_mw_p'])}, Welch p {p_txt(k56['all_learnctrl_RpRm_welch_p'])}. "
                      f"It is found in more R+ than R− mice ({num('k56_frp', 100 * k56.frac_Rp, '{:.0f}')} % vs "
                      f"{num('k56_frm', 100 * k56.frac_Rm, '{:.0f}')} %, permutation p {pv('p_k56_bias', k56.cohort_bias_p)}). "
                      f"The reference fit agrees (R+ vs R− Mann–Whitney p {p_txt(r40['all_learnctrl_RpRm_mw_p'])}, Welch p "
                      f"{pv('p_r40_w', r40['all_learnctrl_RpRm_welch_p'])}; cohort bias p {pv('p_r40_bias', r40.cohort_bias_p)}). "
                      f"Because it lies in the lick and reward period, it may reflect reward (R+ only) rather than a decision "
                      f"signal.", "Figure 4 row 2", "Wilcoxon; MW + Welch; permutation",
                      f"{int(k56['all_learnctrl_R+_n'])} R+ / {int(k56['all_learnctrl_R-_n'])} R− mice", "fragile",
                      "uncorrected; lick / reward confound") + "\n")
    md.append(R.claim("k68", f"An early sensory component (stim_sensory, peak ≈ {k68.peak_ms_median:.0f} ms, striatum top group) "
                      f"weakens with learning beyond outcome in R+ ({num('k68_rp', k68['all_learnctrl_R+_median'], '{:+.2f}')}, p "
                      f"{pv('p_k68_rp', k68['all_learnctrl_R+_wilcoxon_p'])}) more than in R− (R+ vs R− Mann–Whitney p "
                      f"{pv('p_k68_mw', k68['all_learnctrl_RpRm_mw_p'])}, Welch p {pv('p_k68_w', k68['all_learnctrl_RpRm_welch_p'])}). "
                      f"The reference fit keeps the R+ decrease ({num('r39_rp', r39['all_learnctrl_R+_median'], '{:+.2f}')}, p "
                      f"{pv('p_r39_rp', r39['all_learnctrl_R+_wilcoxon_p'])}) but not the cohort difference (Mann–Whitney p "
                      f"{pv('p_r39_mw', r39['all_learnctrl_RpRm_mw_p'])}): early evoked activity may adapt over the session in "
                      f"both cohorts.", "Figure 4 row 3; Figure S4", "Wilcoxon; MW + Welch",
                      f"{int(k68['all_learnctrl_R+_n'])} R+ / {int(k68['all_learnctrl_R-_n'])} R− mice", "fragile",
                      "cohort difference only in the main fit") + "\n")
    k61 = C.loc[("_splitnn", "stim_long", 61)]
    md.append(R.claim("k61", f"A broad late component (stim_long cl. 61, midbrain top group) found in "
                      f"{num('k61_frm', 100 * k61.frac_Rm, '{:.0f}')} % of R− but {num('k61_frp', 100 * k61.frac_Rp, '{:.0f}')} % of "
                      f"R+ mice (permutation p {pv('p_k61', k61.cohort_bias_p)}; midbrain units recorded in "
                      f"{num('k61_cov_rp', 100 * k61.top_group_coverage_Rp, '{:.0f}')} % vs "
                      f"{num('k61_cov_rm', 100 * k61.top_group_coverage_Rm, '{:.0f}')} % of mice) has no counterpart in the "
                      f"reference fit (best match similarity {num('k61_sim', A.loc[('stim_long', 61), 'match_similarity'], '{:.2f}')}); "
                      f"the cohort biases of the main jaw and lick clusters are not found in the reference fit either.",
                      "Table 5", "permutation", f"{n_d0['R+']} R+ / {n_d0['R-']} R− mice", "n.s.", "main fit only") + "\n")
    md.append(f"The candidates and their reference matches are summarised in {ref('tbl-agree')}; the sensory clusters of the "
              f"main fit are shown in {ref('fig-s-sensory')} and the stimulus-aligned clusters of the reference fit in "
              f"{ref('fig-s-clusters-ref')}.\n")
    md.append(figure("fig-cand", F / "candidates_main_vs_reference.png",
                     "Candidate learning components in the main fit and their best-matching cluster in the reference fit "
                     "(014). Rows: stim_long main cl. 67 (reference cl. 44), stim_long main cl. 56 (reference cl. 40), "
                     "stim_sensory main cl. 68 (reference cl. 39). Columns 1–2: mean time factor per cohort (green R+, magenta "
                     "R−; whisker day 0), with peak latency, top area group and number of mice with a member. Columns 3–4: "
                     "per-mouse excess (Eq. 9) of the learning correlation controlled for trial index and outcome (Eq. 8) over "
                     "its linear-shift null (≤ 100 shifts), black bar = median; p above each cohort: one-sample Wilcoxon; title: "
                     "R+ vs R− Mann–Whitney and Welch. Exploratory, uncorrected.", script="015_candidates_figure.py"))
    sel = [("stim_long", 67), ("stim_long", 56), ("stim_sensory", 68), ("stim_sensory", 67), ("stim_long", 61)]
    arows = []
    for a, k in sel:
        r = A.loc[(a, k)]
        arows.append([a, k, f"{r.split_peak_ms_median:.0f}", int(r.ref_cluster), f"{r.match_similarity:.2f}",
                      f"{r['split_all_learnctrl_R+_median']:+.2f} / {r['split_all_learnctrl_R-_median']:+.2f}",
                      f"{r['ref_all_learnctrl_R+_median']:+.2f} / {r['ref_all_learnctrl_R-_median']:+.2f}",
                      p_txt(r.split_all_learnctrl_RpRm_mw_p), p_txt(r.ref_all_learnctrl_RpRm_mw_p),
                      p_txt(r.split_cohort_bias_p), p_txt(r.ref_cohort_bias_p)])
    at = pd.DataFrame(arows, columns=["alignment", "main cl.", "peak (ms)", "ref. cl.", "match S", "main Δ R+ / R−",
                                      "ref. Δ R+ / R−", "main MW p", "ref. MW p", "main bias p", "ref. bias p"])
    md.append(table("tbl-agree", at, "Candidate clusters of the main fit and their best-matching reference cluster (match S: "
                    "similarity of cluster means, Eq. 10). Δ: median per-mouse excess of the outcome-controlled learning "
                    "correlation (Eq. 8, 9); MW p: R+ vs R− Mann–Whitney; bias p: cohort membership permutation. Uncorrected. "
                    "Source: tables/method_agreement.csv."))

    # ------------------------------------------------------------------ discussion, caveats
    md.append("# Discussion\n")
    md.append("Session-wide TCA finds a small set of components shared by most mice: stimulus-evoked transients, broader "
              "response components tied to licking, late lick / reward components, and slow components (variability around "
              "baseline in the main fit, rate drifts in the reference fit). Most are spread over many areas; area-specific "
              "components are few, the clearest being the fast Ss-whisker-led whisker response.\n")
    md.append("The comparison with behaviour shows how strongly trial outcome dominates: almost every stimulus-aligned "
              "component is larger on hit trials, and that alone makes it correlate with the learning curve. After controlling "
              "for outcome, a broad component around 300–500 ms after the stimulus still grows with learning in R+ mice, and "
              "does so in both fits; its timing, before most first licks, makes it a candidate for a learned, "
              "reward-contingency-dependent response rather than a movement signal. The late component's R+ increase is "
              "harder to interpret because it overlaps reward.\n")
    md.append("Two results did not survive the reference fit: the R−-specific cluster and the cohort biases of the jaw and "
              "lick clusters. Cohort membership differences in clustered components are sensitive to rank and to the "
              "clustering threshold, and these should be treated as fit-specific.\n")
    md.append("# Caveats and limitations\n")
    md.append("- Exploratory and uncorrected: about 130 clusters × 3 measures × several tests; p < 0.05 results are expected "
              "by chance. The candidates are fragile until corrected and checked across ranks.\n"
              "- The rank rule is weakly constraining (restart similarity stays high) and saturates at 12 for the main jaw / lick "
              "fits; results may change with the rank.\n"
              "- Components after about 300 ms overlap licking and, in R+, reward; the jaw- and lick-aligned fits (lick trials "
              "only) are needed to separate them.\n"
              "- The outcome control is linear in trial index plus outcome; slower nonlinear drifts are covered only by the "
              "shift null.\n"
              "- Cluster membership depends on the clustering threshold; candidate clusters contain a subset of mice (e.g. "
              f"{int(k67['all_learnctrl_R+_n'])} R+ and {int(k67['all_learnctrl_R-_n'])} R− mice for the ≈ "
              f"{k67.peak_ms_median:.0f} ms component).\n"
              f"- {n_nojaw} sessions have no jaw / lick fits (no jaw-onset data).\n")
    md.append("# Open questions and next steps\n")
    md.append("1. Rank rule: stricter similarity threshold or one common rank, and stability of the candidates across ranks.\n"
              "2. Lick / reward vs decision: jaw- and lick-aligned clusters and reaction time as covariate for the late "
              "candidates.\n"
              "3. Multiple-comparison correction (e.g. Benjamini–Hochberg per family) once the candidate list is fixed.\n"
              "4. Expert sessions: compare day 0 with expert per cohort for the candidate clusters.\n")

    # ------------------------------------------------------------------ appendix
    md.append("# Appendix\n")
    sz = []
    for a in AL:
        g = ses[(ses.method == "_splitnn") & (ses.alignment == a)]
        for (st, c), h in g.groupby(["stage", "cohort"]):
            sz.append([a, "day 0" if st == "learning" else "expert", c, len(h), h.mouse_id.nunique(),
                       int(h.n_trials.median()), int(h.n_units.median()), f"{int(h.n_units.min())}–{int(h.n_units.max())}"])
    md.append(f"Sample sizes are given in {ref('tbl-s-sizes')}, all parameters in {ref('tbl-s-params')}, and every claim with "
              f"its status in {ref('tbl-s-registry')}. The pilot comparison of model variants that led to the split fit is in "
              f"{ref('fig-s-variants')}.\n")
    md.append(table("tbl-s-sizes", pd.DataFrame(sz, columns=["alignment", "stage", "cohort", "sessions", "mice",
                                                             "trials (median)", "units (median)", "units (range)"]),
                    f"Sample sizes per alignment, stage and cohort (main fit; reference fit identical). {n_fit} fits in total "
                    f"(session × alignment × fit)."))
    prm = pd.DataFrame([["baseline", "−1.0 to −0.010 s before trial start, per trial and unit"],
                        ["artefact window", "−10 to +5 ms around trial start (spikes removed, valid-time rates)"],
                        ["minimum rate", "0.5 Hz (window + baseline mean)"],
                        ["scaling", "per-unit SD over trials and bins, no centring"],
                        ["ranks / restarts", "1–12 / 10; tolerance 1e-6; ≤ 500 iterations (HALS)"],
                        ["rank rule", "median restart similarity ≥ 0.85 and gain ≥ 0.05 (cohort medians, day 0)"],
                        ["shift null", "linear, 10–50 % lags, both directions, ≤ 100 shifts"],
                        ["area specificity null", "300 permutations of area labels"],
                        ["clustering", "Pearson similarity (Eq. 10), average linkage, cut at within-session 95th percentile; ≥ 10 mice"],
                        ["cohort bias", "2,000 cohort-label permutations across mice"],
                        ["labels", "variability |dir| < 0.5; flat CV < 0.2; area-specific ≥ 50 % same top group (p < 0.05)"]],
                       columns=["parameter", "value"])
    md.append(table("tbl-s-params", prm, "Parameters."))
    md.append(figure("fig-s-example-ref", EX / "factors_AB123_20240806_110231_stim_long_zscore_nn_r7.png",
                     "Reference fit (non-negative TCA on scaled rates, no baseline subtraction) of the session of Figure 1, "
                     "stim_long, rank 7. Panels and colours as in Figure 1; unit weights are all ≥ 0.", kind="supplementary",
                     script="007_aligned_figures.py"))
    md.append(figure("fig-s-variants", F / "variant_compare.png",
                     "Pilot comparison of model variants on 6 day-0 mice (stim_long left, jaw right; ranks 1–12): signed CP "
                     "(black), ridge l2 = 10 and 100 (light / dark blue), split non-negative (orange), semi-non-negative "
                     "(green). Rows: relative error on the signed data (thin = mice, thick = median); median restart "
                     "similarity (dotted = 0.85); most negative signed congruence between two components (median; < −0.5 = "
                     "cancelling pair, red dotted); number of mice with a cancelling pair. Only the split fit has no "
                     "cancelling pairs at any rank with high restart similarity in both windows.", kind="supplementary",
                     script="010_variant_compare.py"))
    md.append(figure("fig-s-clusters-ref", F / "clusters_nn_stim_long.png",
                     "Recurring components of the reference fit, stim_long; panels as in Figure 3.", kind="supplementary",
                     script="013_recurring_components.py", width=72))
    md.append(figure("fig-s-sensory", F / "clusters_splitnn_stim_sensory.png",
                     "Recurring components of the main fit, stim_sensory window (10-ms bins, −52.5 to +107.5 ms); panels as "
                     "in Figure 3.", kind="supplementary", script="013_recurring_components.py", width=72))
    md.append(R.registry_table("tbl-s-registry"))
    md.append("**Scripts** (projects/ssl-tca-session-wide): 005_build_aligned.py (tensors), 006g_tca_gpu.py + "
              "kuma_tca_array.sh (fits on Kuma), 007_aligned_figures.py (per-session pages), 010_variant_compare.py, "
              "011_rank_by_cohort.py, 012_component_table.py, 013_recurring_components.py, 014_method_agreement.py, "
              "015_candidates_figure.py. **Results** (combined_results_ks4/ssl-tca-session-wide): artifacts/aligned, "
              "artifacts/tca_aligned_gpu (fits), tables/ (components, clusters, rank, agreement), figures/aligned_gpu/ "
              "(per-session factor and reconstruction pages for the pilot mice, cluster pages for every fit and alignment).\n")
    text = "\n".join(md)
    summary = "\n".join(md[1:4])
    R.write(OUT, text, summary=summary)


if __name__ == "__main__":
    main()
