"""Stand-alone article report on the trend candidate (stim_sensory main-fit cl. 10 / reference cl. 40), to share with a
colleague (user 2026-11-23). Written by report_lib into combined_results_ks4/ssl-tca-session-wide/report/candidate_sensory_cl10/;
PDF / HTML with build.sh there. Every number from the result tables (017, 018, 013, 014), frozen with R.snapshot.
Exploratory, uncorrected.

Run: python report/build_candidate_report.py ; bash <OUT>/report/candidate_sensory_cl10/build.sh
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
import tca_lib as L  # noqa: E402
from report_lib import Report  # noqa: E402

D18 = importlib.import_module("018_candidate_deep")
OUT = L.OUT / "report" / "candidate_sensory_cl10"
T, F = L.OUT / "tables", L.OUT / "figures" / "aligned_gpu"
FC = F / "candidates"
R = Report(banned=(), spelling={r"\bSomatosensory-(whisker|orofacial|body)\b": r"Ss-\1"})
R.register([
    ("fig-comp", "Figure 1"), ("fig-trend", "Figure 2"), ("fig-factors", "Figure 3"), ("fig-area", "Figure 4"),
    ("fig-methods", "Figure 5"), ("fig-expert", "Figure 6"),
    ("tbl-n", "Table 1"), ("tbl-trend", "Table 2"), ("tbl-reg", "Table 3"), ("tbl-methods", "Table 4"),
    ("eq-pre", "Eq. (1)"), ("eq-cp", "Eq. (2)"), ("eq-split", "Eq. (3)"), ("eq-sim", "Eq. (4)"), ("eq-y", "Eq. (5)"),
    ("eq-d", "Eq. (6)"), ("eq-ols", "Eq. (7)"), ("eq-area", "Eq. (8)"), ("eq-ev", "Eq. (9)"),
    ("eq-tmpl", "Eq. (10)"), ("eq-stim", "Eq. (11)"),
    ("fig-s-ref", "Figure S1"), ("fig-s-reftrend", "Figure S2"), ("fig-s-exRp", "Figure S3"), ("fig-s-exRm", "Figure S4"),
    ("fig-s-refarea", "Figure S5"),
    ("tbl-s-params", "Table S1"), ("tbl-s-registry", "Table S2")])
num, pv, ref, table, eq = R.num, R.pv, R.ref, R.table, R.eq


def figure(*a, width=100, **k):
    return R.figure(*a, **k).replace("{width=100%}", f"{{width={width}%}}")


def crop_top(src, frac, dst):
    """top part of a 016 detail page (panels A-D), the lower panels use the superseded learning measure of 012"""
    im = Image.open(src)
    im.crop((0, 0, im.width, int(im.height * frac))).save(dst)
    return dst


def P(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def stats_block(key, a, b):
    """R+ / R- medians, Wilcoxon p, n; MW / Welch p (recorded)"""
    a, b = (np.asarray(x, float) for x in (a, b))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    mw, we = D18.two(a, b)
    return dict(
        mp=num(f"{key}_Rp_med", np.median(a), "{:+.2f}"), pp=pv(f"{key}_Rp_p", D18.wil(a)), np_=num(f"{key}_Rp_n", len(a)),
        mm=num(f"{key}_Rm_med", np.median(b), "{:+.2f}"), pm=pv(f"{key}_Rm_p", D18.wil(b)), nm=num(f"{key}_Rm_n", len(b)),
        mw=pv(f"{key}_mw", mw), we=pv(f"{key}_welch", we))


def txt(s):
    return (f"R+ median {s['mp']} (Wilcoxon p {s['pp']}, n = {s['np_']}), R− {s['mm']} (p {s['pm']}, n = {s['nm']}); "
            f"R+ vs R− Mann–Whitney p {s['mw']}, Welch p {s['we']}")


def load(stem):
    files = [T / f"{stem}_{k}" for k in ("trials.parquet", "units.parquet", "areafifths.parquet", "expert.parquet",
                                         "regression.csv", "area.csv", "methods.csv")]
    snap = dict(zip([f.name for f in files], R.snapshot(files, OUT)))
    rd = lambda n: pd.read_parquet(snap[f"{stem}_{n}"]) if n.endswith("parquet") else pd.read_csv(snap[f"{stem}_{n}"])
    return {k.split(".")[0]: rd(k) for k in ("trials.parquet", "units.parquet", "areafifths.parquet", "expert.parquet",
                                              "regression.csv", "area.csv", "methods.csv")}


def main():
    base = [T / n for n in ("trend_clusters.csv", "clusters.csv", "method_agreement.csv", "sessions.csv")]
    snap = dict(zip([f.name for f in base], R.snapshot(base, OUT)))
    tc = pd.read_csv(snap["trend_clusters.csv"]).set_index(["method", "alignment", "cluster"])
    cl = pd.read_csv(snap["clusters.csv"]).set_index(["method", "alignment", "cluster"])
    ag = pd.read_csv(snap["method_agreement.csv"]).set_index(["alignment", "split_cluster"])
    ses = pd.read_csv(snap["sessions.csv"])
    m, r = load("deep_splitnn_cl10"), load("deep_nn_cl40")
    kref = int(ag.loc[("stim_sensory", 10), "ref_cluster"])
    c10, c40 = cl.loc[("_splitnn", "stim_sensory", 10)], cl.loc[("_nn", "stim_sensory", kref)]
    t10, t40 = tc.loc[("_splitnn", "stim_sensory", 10)], tc.loc[("_nn", "stim_sensory", kref)]
    n_tested = tc.reset_index().groupby(["method", "alignment"]).size()

    # ------------------------------------------------------------ derived statistics (018 tables)
    def S_of(d, score="member"):
        S = D18.per_session_stats(d["trials"], score)
        return S
    Sm = S_of(m)
    S0 = Sm[(Sm.stage == "learning") & Sm.is_member]
    Se = Sm[(Sm.stage == "expert") & Sm.is_member]
    Sr = S_of(r)
    Sr0 = Sr[(Sr.stage == "learning") & Sr.is_member]
    coh = lambda S, c, col: S.loc[S.cohort == c, col]
    d_main = stats_block("d_sd_main", coh(S0, "R+", "d"), coh(S0, "R-", "d"))
    d_ref = stats_block("d_sd_ref", coh(Sr0, "R+", "d"), coh(Sr0, "R-", "d"))
    pp_main = stats_block("passive_main", coh(S0, "R+", "passive_post") - coh(S0, "R+", "passive_pre"),
                          coh(S0, "R-", "passive_post") - coh(S0, "R-", "passive_pre"))
    rg = m["regression"]
    rg0, rge = rg[rg.stage == "learning"], rg[rg.stage == "expert"]
    REGS = {k: stats_block(f"reg_{k}", coh(rg0, "R+", k), coh(rg0, "R-", k))
            for k in ["pos", "hit", "prev_rew", "base_pop", "log_iti", "pos_m2", "p_wh_i_m2", "p_au_i_m2"]}
    rgr = r["regression"]
    rgr0 = rgr[rgr.stage == "learning"]
    pos_ref = stats_block("reg_pos_ref", coh(rgr0, "R+", "pos"), coh(rgr0, "R-", "pos"))
    pos2_ref = stats_block("reg_pos_m2_ref", coh(rgr0, "R+", "pos_m2"), coh(rgr0, "R-", "pos_m2"))
    # area contributions
    A = m["area"]
    def area_stat(g, key):
        a = A[A.group == g].groupby(["mouse_id", "cohort"]).d.mean().reset_index()
        return stats_block(key, a.loc[a.cohort == "R+", "d"], a.loc[a.cohort == "R-", "d"])
    ar_ss, ar_mb = area_stat("Somatosensory-whisker", "area_ss"), area_stat("Midbrain", "area_mb")
    lev = A.groupby(["mouse_id", "group"])["level"].mean().unstack().fillna(0)
    share_ss = num("share_ss", lev["Somatosensory-whisker"].median(), "{:.2f}")
    share_mb = num("share_mb", lev["Midbrain"].median(), "{:.2f}")
    # single units
    U = m["units"]
    u0 = U[(U.stage == "learning") & U.responsive].dropna(subset=["d"])
    def unit_stat(col, val, key):
        s = u0[u0[col] == val].groupby(["mouse_id", "cohort"]).d.mean().reset_index()
        return stats_block(key, s.loc[s.cohort == "R+", "d"], s.loc[s.cohort == "R-", "d"])
    un_ss = unit_stat("area_group", "Somatosensory-whisker", "unit_ss")
    un_bfd = unit_stat("area", "SSp-bfd", "unit_bfd")
    un_sss = unit_stat("area", "SSs", "unit_sss")
    un_scm = unit_stat("area", "SCm", "unit_scm")
    un_th = unit_stat("area_group", "Thalamus", "unit_th")
    un_mb = unit_stat("area_group", "Midbrain", "unit_mb")
    n_resp = num("n_resp_units", int(len(u0)))
    n_units = num("n_units_day0", int(((U.stage == "learning")).sum()))
    # expert
    E = m["expert"]
    e = E[E.stage == "expert"]
    b0 = E[(E.stage == "learning") & ~E.is_member]
    fp = num("exp_fp", 100 * b0.passes.mean(), "{:.0f}")
    ex_n = {c: (num(f"exp_match_{c}", int(e[e.cohort == c].passes.sum())), num(f"exp_n_{c}", int((e.cohort == c).sum())))
            for c in ("R+", "R-")}
    ex_d = stats_block("exp_d", coh(Se, "R+", "d"), coh(Se, "R-", "d"))
    ex_hit = stats_block("exp_hit", coh(Se, "R+", "slope_wh_hit"), coh(Se, "R-", "slope_wh_hit"))
    ex_miss = stats_block("exp_miss", coh(Se, "R+", "slope_wh_miss"), coh(Se, "R-", "slope_wh_miss"))
    ex_pos = stats_block("exp_pos", coh(rge, "R+", "pos"), coh(rge, "R-", "pos"))
    uss = U[U.responsive & U.area_group.str.contains("whisker")]
    hz = {}
    for c in ("R+", "R-"):
        u_ = uss[(uss.cohort == c) & (uss.stage == "learning")].groupby("mouse_id")[["hz_hit_first", "hz_hit_last"]].mean().dropna()
        from scipy import stats as st
        hz[c] = dict(first=num(f"hz_first_{c}", u_.hz_hit_first.median(), "{:.1f}"),
                     last=num(f"hz_last_{c}", u_.hz_hit_last.median(), "{:.1f}"),
                     p=pv(f"hz_p_{c}", st.wilcoxon(u_.hz_hit_first, u_.hz_hit_last).pvalue),
                     n=num(f"hz_n_{c}", len(u_)),
                     exp=num(f"hz_exp_{c}", uss[(uss.cohort == c) & (uss.stage == "expert")].groupby("mouse_id")
                             .hz_hit.mean().median(), "{:.1f}"))
    # other methods
    MT = m["methods"]
    def meth(sc):
        S = D18.per_session_stats(m["trials"][m["trials"].stage == "learning"], sc)
        S = S[S.is_member] if sc == "member" else S
        return stats_block(f"meth_{sc}", coh(S, "R+", "d"), coh(S, "R-", "d")), S
    MS = {sc: meth(sc) for sc in D18.SCORES}
    memb = {sc: stats_block(f"meth_{sc}_mem", coh(S[S.is_member], "R+", "d"), coh(S[S.is_member], "R-", "d"))
            for sc, (_, S) in MS.items() if sc != "member"}
    tr = m["trials"]
    aw = tr[(tr.stage == "learning") & tr.member.notna() & (tr.trial_type == "whisker_trial") & (tr.epoch == "active")]
    from scipy import stats as st
    agree = {s: num(f"agree_{s}", aw.groupby("session_id").apply(lambda g: st.spearmanr(g.member, g[s],
                                                                                         nan_policy="omit")[0]).median(),
                    "{:.2f}") for s in ("template", "pca", "stimaxis")}

    nd0 = {c: num(f"n_d0_{c}", int(((ses.stage == "learning") & (ses.cohort == c)).sum())) for c in ("R+", "R-")}
    nex = {c: num(f"n_exp_{c}", int(((ses.stage == "expert") & (ses.cohort == c)).sum())) for c in ("R+", "R-")}

    # ------------------------------------------------------------ figures (copies)
    top10 = crop_top(FC / "stim_sensory_splitnn_cl10.png", 0.49, FC / "stim_sensory_splitnn_cl10_top.png")
    top40 = crop_top(FC / f"stim_sensory_nn_cl{kref}.png", 0.49, FC / f"stim_sensory_nn_cl{kref}_top.png")

    md = []
    md.append('---\ntitle: "A candidate learning-related component in session-wide TCA: the early Ss-whisker response '
              '(stim_sensory cluster 10)"\nsubtitle: "ssl-tca-session-wide, candidate report; exploratory, uncorrected"\n'
              f'date: "{pd.Timestamp.today().date()}"\n---\n')
    # ---------------------------------------------------------------- abstract
    md.append("# Abstract\n")
    md.append(
        f"Tensor component analysis (TCA) was fitted to every whisker-learning (day 0) Neuropixels session of the "
        f"SSL dataset (brain-wide recordings, {nd0['R+']} R+ and {nd0['R-']} R− sessions, one per mouse) and to "
        f"{num('n_exp', int((ses.stage == 'expert').sum()))} expert sessions; components were matched across mice by "
        f"clustering. Of the recurring components, one was selected as a learning candidate by a within-session trend "
        f"contrast: an early (peak ≈ {num('peak', c10.peak_ms_median, '{:.1f}')} ms) whisker-evoked component with its "
        f"largest weight in Ss-whisker and the midbrain (cluster 10 of the main fit, matched by cluster {kref} of the "
        f"reference fit). Its trial weight rises over the session on whisker trials in R+ mice and falls on whisker "
        f"hits in R− mice, while auditory hits stay flat. In standard-deviation units the whisker-minus-auditory slope "
        f"difference is {txt(d_main)}. It holds after controlling hit, previous reward, pre-stimulus population rate, "
        f"inter-trial interval and the performance curves, is carried by Ss-whisker units (also without TCA, in single "
        f"whisker-responsive units), and replicates in the reference fit. Two observations limit its reading as "
        f"learning: the same within-session rise is present in R+ expert sessions, and simpler population methods "
        f"(PCA, a whisker − auditory axis) do not recover the R+ rise; a fixed whisker template recovers only the R− "
        f"decrease. The component is best described as a reward-contingency-dependent, within-session change of the "
        f"early Ss-whisker response, not yet as a day-0 learning signal.\n")
    md.append("# Key results\n")
    md.append(
        f"1. **Component.** Cluster 10 (main fit) recurs in {num('n_mice', int(c10.n_mice))} of the day-0 mice "
        f"({num('n_Rp', int(c10.n_Rp))} R+, {num('n_Rm', int(c10.n_Rm))} R−); peak ≈ {num('peak2', c10.peak_ms_median, '{:.1f}')} ms; "
        f"top area group Ss-whisker in {num('topshare', 100 * c10.top_group_share, '{:.0f}')} % of its members; "
        f"nearly only whisker trials ({ref('fig-comp')}).\n"
        f"2. **Selection statistic** (017, weights / their active mean): d R+ {num('t_dRp', t10['d_R+_median'], '{:+.2f}')} "
        f"(p {pv('t_dRp_p', t10['d_R+_p'])}), R− {num('t_dRm', t10['d_R-_median'], '{:+.2f}')} (p {pv('t_dRm_p', t10['d_R-_p'])}); "
        f"R+ vs R− Mann–Whitney p {pv('t_mw', t10['d_RpRm_mw_p'])}, Welch p {pv('t_we', t10['d_RpRm_welch_p'])} "
        f"({ref('fig-trend')}); one of {num('n_tested', int(n_tested.loc[('_splitnn', 'stim_sensory')]))} sensory clusters tested.\n"
        f"3. **Robust units** (SD units): {txt(d_main)}; reference fit: {txt(d_ref)} ({ref('tbl-trend')}).\n"
        f"4. **Trial factors**: the position coefficient remains in R+ after hit, previous reward, pre-stimulus rate and "
        f"ITI ({txt(REGS['pos'])}) and after adding the performance curves ({txt(REGS['pos_m2'])}); a higher "
        f"pre-stimulus population rate lowers the component in both cohorts ({ref('fig-factors')}, {ref('tbl-reg')}).\n"
        f"5. **Area**: the cohort difference of the trend sits in Ss-whisker ({txt(ar_ss)}); without TCA, "
        f"whisker-responsive Ss-whisker units show the same difference ({txt(un_ss)}) ({ref('fig-area')}).\n"
        f"6. **Other methods**: a fixed whisker template finds the R− decrease but no R+ rise; PCA and a "
        f"whisker − auditory axis show no cohort difference over all mice ({ref('fig-methods')}, {ref('tbl-methods')}).\n"
        f"7. **Expert sessions**: the component is found in {ex_n['R+'][0]}/{ex_n['R+'][1]} R+ and {ex_n['R-'][0]}/{ex_n['R-'][1]} "
        f"R− expert sessions, and in R+ it still rises within the session on whisker hits and misses ({ref('fig-expert')}).\n")

    # ---------------------------------------------------------------- introduction
    md.append("# Introduction\n")
    md.append(
        "In the SSL task, head-fixed mice that already lick for an auditory tone learn, within one session (whisker "
        "day 0), to lick after a brief whisker deflection. In the R+ cohort the whisker lick is rewarded; in the R− cohort "
        "it is not, so the session structure, stimuli and auditory rewards are the same, but only R+ mice learn the "
        "whisker association. Each session has a passive block before and after the active block (same stimuli, no "
        "reward, no task context).\n")
    md.append(
        "The project ssl-tca-session-wide asks which low-dimensional neuron × time × trial structure brain-wide activity "
        "has within a learning session, and whether trial factors follow learning. TCA was fitted per session; "
        "components that recur across mice were found by clustering; a candidate learning component was required to "
        "(i) change over the session on whisker trials in R+ mice, (ii) not change alike on auditory hits (same lick, "
        "same reward, association already learned), and (iii) differ from R− mice. This report documents the only "
        "component that passed in both fits and the controls run on it. It is meant to be read on its own.\n")
    md.append(
        "Questions addressed: what the component is (time course, areas, trial classes); how its trial weight changes "
        "over the session per trial class and cohort; which trial factors explain it; in which area the change is; "
        "whether methods other than TCA find it; and whether it is still present, and still changing, in expert "
        "sessions.\n")

    # ---------------------------------------------------------------- methods
    md.append("# Methods\n")
    md.append("## Data\n")
    md.append(
        f"Neuropixels recordings (Kilosort 4, NWB_ks4), stable good + MUA units (unit table unit_info_all_days_v2, "
        f"137 stable-unit list), cohort and learner labels from the reference sheet. Whisker day 0: {nd0['R+']} R+ and "
        f"{nd0['R-']} R− sessions (one per mouse); expert (whisker day 1–4): {nex['R+']} R+ and {nex['R-']} R− sessions. "
        f"Trials: perf == 6 dropped except passive, warm-up cut, per-trial context rule, disengaged tail trimmed (rule A1); "
        f"whisker and auditory trials of the passive pre, active and passive post blocks (no-stimulus trials dropped). "
        f"Sample sizes in {ref('tbl-n')}.\n")
    md.append("## Tensors\n")
    md.append(
        "Spikes were binned in 10-ms bins from −52.5 to +107.5 ms around stimulus onset (16 bins, the grid shifted by "
        "−2.5 ms). The stimulation artefact window (−10 to +5 ms) had its spikes removed and replaced, before binning, "
        "by Poisson spikes at the trial's own baseline rate (uniform in the window, seeded per session), so that every "
        "bin is a full-width count with the noise of real data. For unit $u$, trial $k$ and bin $t$ the input is the "
        "baseline-subtracted, scaled rate")
    md.append(eq("eq-pre", r"X_{u t k} = \frac{r_{u t k} - b_{u k}}{\sigma_u}"))
    md.append(
        "where $r_{utk}$ is the rate (Hz), $b_{uk}$ the same trial's baseline rate in −1.0 to −0.010 s and $\\sigma_u$ "
        "the standard deviation of $r - b$ over trials and bins (not centred, so 0 stays the trial's baseline). Units "
        "with a mean rate < 0.5 Hz were dropped.\n")
    md.append("## TCA fits\n")
    md.append("A rank-$R$ CP model approximates the tensor as")
    md.append(eq("eq-cp", r"X_{u t k} \approx \sum_{c=1}^{R} n_{u c}\, \tau_{t c}\, w_{k c}"))
    md.append(
        "(fitted to $X$ of " + ref("eq-pre") + ") with neuron factors $n$, time factors $\\tau$ and trial factors "
        "(trial weights) $w$. **Main fit (split "
        "non-negative)**: each unit is split into an increase and a decrease channel and a non-negative CP model "
        "(HALS) is fitted to")
    md.append(eq("eq-split", r"\tilde X = \left[\max(X, 0);\ \max(-X, 0)\right], \qquad n^{\mathrm{net}}_{u c} = "
                 r"n^{+}_{u c} - n^{-}_{u c}"))
    md.append(
        f"where the net unit weight $n^{{net}}$ is the increase minus the decrease weight. **Reference fit**: "
        f"non-negative CP on the scaled rates without baseline subtraction. Ranks 1–12, 10 restarts, GPU (Kuma); the "
        f"rank per cohort follows the Williams et al. (2018) trade-off (median restart similarity ≥ 0.85, error "
        f"decrease per component). Trial weights of {ref('eq-cp')} are in arbitrary units per session.\n")
    md.append("## Recurring components\n")
    md.append("Two components $i, j$ of different mice were compared by")
    md.append(eq("eq-sim", r"s_{ij} = \tfrac{1}{3}\left[\rho(\tau_i, \tau_j) + \rho(a_i, a_j) + \rho(p_i, p_j)\right]"))
    md.append(
        f"where $\\rho$ is the Pearson correlation, $a$ the area profile (mean net unit weight per area group, groups "
        f"present in both) and $p$ the trial-class profile (mean trial weight of active whisker hit / miss, auditory "
        f"hit / miss, passive whisker / auditory). Average-linkage clustering on $1 - s$, cut at the 95th percentile of "
        f"the similarity between different components of the same session (threshold "
        f"{num('thr', c10.threshold, '{:.2f}')}); clusters with ≥ 10 mice kept. The main-fit cluster was matched to the "
        f"reference-fit cluster with the most similar centroid (similarity "
        f"{num('ag_sim', ag.loc[('stim_sensory', 10), 'match_similarity'], '{:.3f}')}).\n")
    md.append("## Trend measure\n")
    md.append(
        "For a member component the trial weight of each active trial is normalised by the session's active whisker "
        "trials (A = active whisker trials):")
    md.append(eq("eq-y", r"y_k = \frac{w_k}{\overline{w}_{A}} \ \ \text{(selection, 017)}, \qquad "
                 r"z_k = \frac{w_k - \overline{w}_{A}}{\mathrm{SD}_{A}(w)} \ \ \text{(robustness, 018)}"))
    md.append(
        f"The SD units $z$ are used for all analyses after the selection, because dividing by a mean near 0 (weak "
        f"members, PCA scores) makes $y$ unstable. Sessions with several member components average their $z$. With "
        f"$x_k$ the position of the trial in the active block (0 = first, 1 = last), the least-squares slope of $y$ "
        f"(or $z$) on $x$ is computed separately for whisker hits ($\\beta_{{WH}}$), whisker misses ($\\beta_{{WM}}$) "
        f"and auditory hits ($\\beta_{{AH}}$), ≥ 8 trials each, and the contrast is")
    md.append(eq("eq-d", r"d = \beta_{WH} - \beta_{AH}"))
    md.append(
        "(change per active block). Whisker and auditory hits share the lick and the reward (R+); only the whisker "
        "association is learned on day 0, so a lick, reward or drift component gives $d \\approx 0$. A candidate needs "
        "$d \\neq 0$ in R+ (Wilcoxon p < 0.05) and R+ ≠ R− (Mann–Whitney p < 0.05).\n")
    md.append(
        "Fifths: the active trials of each class are divided into five equal position bins and averaged, with the "
        "passive pre and post whisker trials as end points.\n")
    md.append("## Trial factors\n")
    md.append("Per session, on active whisker trials, the score $z$ (" + ref("eq-y") + ") was regressed on standardised "
              "predictors:")
    md.append(eq("eq-ols", r"z_k = \beta_0 + \beta_{pos} x_k + \beta_{hit} h_k + \beta_{rew} q_{k-1} + \beta_{pre} "
                 r"\bar b_k + \beta_{iti} \log \Delta_k \ \left[+ \beta_{W} P_W(k) + \beta_{A} P_A(k)\right] + \epsilon_k"))
    md.append(
        "where $h$ = lick (hit), $q_{k-1}$ = reward delivered on the previous trial, $\\bar b_k$ = pre-stimulus "
        "population rate (mean over units of the per-unit z-scored baseline), $\\Delta_k$ = time since the previous "
        "trial start, and, in model 2, $P_W$, $P_A$ = lick probabilities of the whisker and auditory learning curves "
        "(exact curves). The coefficients (per mouse) were tested against 0 (Wilcoxon) and between cohorts.\n")
    md.append("## Area attribution\n")
    md.append("The member projection is split into area groups $g$:")
    md.append(eq("eq-area", r"p_{g}(k) = \frac{1}{\mathrm{SD}_A(P)} \sum_{u \in g} n^{\mathrm{net}}_{u}\, "
                 r"\sum_t \tau_t X_{u t k}, \qquad P(k) = \sum_g p_g(k)"))
    md.append(
        f"with the net unit weights of {ref('eq-split')}, so the area contributions add up to the total projection and their slopes, and their $d$ ({ref('eq-d')}), "
        f"add up to the total. An area not recorded in a mouse contributes 0. The share of each area is "
        f"$\\overline{{p_g}}_A / \\overline{{P}}_A$.\n")
    md.append("Without TCA, each unit's evoked response on trial $k$ is")
    md.append(eq("eq-ev", r"e_{u k} = \frac{1}{4} \sum_{t \in \{12.5, \dots, 42.5\,\mathrm{ms}\}} X_{u t k}"))
    md.append(
        f"(and the same in Hz without $\\sigma_u$). Units are whisker-responsive if $e$ over all whisker trials is > 0 "
        f"with Wilcoxon p < 0.01. Per unit the slopes and $d$ ({ref('eq-d')}) of $e$ are computed as for the TCA score "
        f"and averaged per mouse within an area group or area.\n")
    md.append("## Other methods\n")
    md.append("All on bins ≥ +7.5 ms (the corrected bins excluded), flattened to units × time. Template:")
    md.append(eq("eq-tmpl", r"s^{\mathrm{tmpl}}_k = \frac{\langle X_k, \bar X_{\mathrm{pre}} \rangle}{\lVert \bar "
                 r"X_{\mathrm{pre}} \rVert^2}"))
    md.append(
        "where $\\bar X_{pre}$ is the mean response of the passive-pre whisker trials (independent of the active "
        "trials; pre trials scored with the template of the other half). PCA: uncentred scores on the principal "
        "component, among the first five of all stimulus trials, that best separates whisker from auditory trials "
        "(AUC), signed whisker > auditory. Stimulus axis (the stimulus marginal of demixed PCA without "
        "regularisation, cross-validated in two folds of alternating trials):")
    md.append(eq("eq-stim", r"s^{\mathrm{stim}}_k = \frac{\langle X_k,\ \bar X^{(-k)}_{W} - \bar X^{(-k)}_{A} "
                 r"\rangle}{\lVert \bar X^{(-k)}_{W} - \bar X^{(-k)}_{A} \rVert}"))
    md.append(
        f"where the means come from the fold not containing $k$. Each score is put in SD units ({ref('eq-y')}) and "
        f"analysed with {ref('eq-d')}, in all day-0 mice and in the mice with a member.\n")
    md.append("## Expert matching\n")
    md.append(
        f"Each expert-session component (main fit, same rank rule) was compared with the day-0 members by its mean "
        f"similarity ({ref('eq-sim')}); the best component of a session was accepted if it reached the 10th percentile "
        f"of the members' leave-one-out mean similarity to each other. False-positive rate: share of day-0 sessions "
        f"without a member whose best component passes ({fp} %).\n")
    md.append("## Statistics\n")
    md.append(
        "Unit of analysis: mouse (one day-0 session per mouse; components and units averaged per mouse). Within cohort: "
        "Wilcoxon signed-rank against 0. R+ vs R−: Mann–Whitney U and Welch t-test, both reported. No correction for "
        "multiple comparisons (exploratory). Parameters in " + ref("tbl-s-params") + ". Scripts: 005 (tensors), 006g "
        "(fits), 012–014 (components, clusters, fit matching), 016 (detail pages), 017 (trend selection), 018 (this "
        "report's analyses), report/build_candidate_report.py.\n")

    # ---------------------------------------------------------------- results
    md.append("# Results\n")
    md.append("## An early, whisker-specific component with most weight in Ss-whisker and the midbrain\n")
    md.append(R.claim("comp",
        f"The component peaks {num('peak3', c10.peak_ms_median, '{:.1f}')} ms after the whisker stimulus and is back to "
        f"baseline by ≈ 50 ms ({ref('fig-comp')}A). Its net unit weight is largest in Ss-whisker, followed by the midbrain, "
        f"pallidum and thalamus, alike in both cohorts ({ref('fig-comp')}B). It is almost absent on auditory trials "
        f"(trial weight ≈ 0) and present on whisker trials of all blocks, larger on whisker hits than misses "
        f"({ref('fig-comp')}C). It is found in {num('frac_Rp', 100 * c10.frac_Rp, '{:.0f}')} % of R+ and "
        f"{num('frac_Rm', 100 * c10.frac_Rm, '{:.0f}')} % of R− mice (cohort difference p "
        f"{pv('coh_bias_p', c10.cohort_bias_p)}, label permutation); Ss-whisker was recorded in about two thirds of the "
        f"mice of each cohort.", figure="Fig. 1", stat="descriptive", n=f"{int(c10.n_mice)} mice") + "\n")
    md.append(figure("fig-comp", top10,
        "Main-fit cluster 10 (stim_sensory, whisker day 0, split non-negative fit). **A** Time factor (unit norm), mean ± "
        "SEM over member mice; green R+, magenta R−; 10-ms bins, stimulus at 0. **B** Area profile: mean net unit weight "
        "per area group (left bar R+, right bar R−, edge colour = cohort, fill = area-group colour), mean ± SEM over "
        "mice, groups with ≥ 3 mice. **C** Trial-class profile: per mouse, the mean trial weight of a class divided by "
        "the mouse's mean over active whisker trials (dots = mice, bar = median; symmetric-log axis): active whisker "
        "hit / miss, auditory hit / miss, passive pre / post whisker. **D** Course over the session on active whisker "
        "trials: trials split into five equal position bins (fifths of the active block); mean normalised weight per "
        "bin, whisker hits (filled, solid) and misses (open, dashed) separately, mean ± SEM over mice. D separates a "
        "change of the response within an outcome from a change in the number of hits: learning beyond outcome would "
        "be a rise within hits and within misses. Script 016.", width=92, script="016_candidate_details.py"))
    md.append("## The component rises within the session in R+ and falls on whisker hits in R−\n")
    md.append(R.claim("trend",
        f"In R+ mice the normalised weight rises across the active block on whisker hits (slope "
        f"{num('s_wh_Rp', t10['slope_wh_hit_R+_median'], '{:+.2f}')} per block, p {pv('s_wh_Rp_p', t10['slope_wh_hit_R+_p'])}) "
        f"and on whisker misses ({num('s_wm_Rp', t10['slope_wh_miss_R+_median'], '{:+.2f}')}, p "
        f"{pv('s_wm_Rp_p', t10['slope_wh_miss_R+_p'])}), not on auditory hits "
        f"({num('s_ah_Rp', t10['slope_au_hit_R+_median'], '{:+.2f}')}, p {pv('s_ah_Rp_p', t10['slope_au_hit_R+_p'])}). "
        f"In R− mice it falls on whisker hits ({num('s_wh_Rm', t10['slope_wh_hit_R-_median'], '{:+.2f}')}, p "
        f"{pv('s_wh_Rm_p', t10['slope_wh_hit_R-_p'])}) and is flat on misses. The contrast $d$ ({ref('eq-d')}) differs "
        f"between cohorts (Mann–Whitney p {pv('t_mw2', t10['d_RpRm_mw_p'])}, Welch p {pv('t_we2', t10['d_RpRm_welch_p'])}; "
        f"{ref('fig-trend')}). In SD units ({ref('eq-y')}), which are not sensitive to a small mean, {txt(d_main)}; "
        f"the reference fit gives {txt(d_ref)} ({ref('tbl-trend')}, {ref('fig-s-ref')}, {ref('fig-s-reftrend')}).",
        figure="Fig. 2, Table 2", stat="Wilcoxon; MW + Welch", n=f"{d_main['np_']} / {d_main['nm']} mice",
        status="holds") + "\n")
    md.append(
        f"Individual sessions ({ref('fig-s-exRp')}, {ref('fig-s-exRm')}) show the spread: the rise is visible in about "
        f"half of the R+ mice, several mice have a component that is strong only for part of the session, and in many "
        f"mice the weight follows the bouts of high whisker lick probability. The passive blocks do not show a "
        f"lasting change: post − pre on passive whisker trials, {txt(pp_main)}.\n")
    md.append(figure("fig-trend", F / "trend_candidates_splitnn.png",
        "Trend selection (017), main fit. Left: time factor per cohort (mean over member mice). Middle: R+ and R− mice, "
        "normalised weight (weight / mean over active trials, $y$ in " + ref("eq-y") + ") in fifths of the active block "
        "for whisker hits (orange filled), whisker misses (orange open) and auditory hits (blue), mean ± SEM over mice; "
        "titles give the median slopes and the median $d$ with Wilcoxon p. Right: $d$ per mouse (" + ref("eq-d") +
        "), bar = median; R+ vs R− Mann–Whitney and Welch p. Members of the same mouse averaged.", width=100,
        script="017_trend_contrasts.py"))
    md.append(table("tbl-trend", pd.DataFrame([
        dict(Fit="main (cl. 10)", Units="mean-normalised (017)", **{"R+ d": f"{t10['d_R+_median']:+.2f} (p {P(t10['d_R+_p'])})",
             "R− d": f"{t10['d_R-_median']:+.2f} (p {P(t10['d_R-_p'])})", "MW p": P(t10['d_RpRm_mw_p']),
             "Welch p": P(t10['d_RpRm_welch_p'])}),
        dict(Fit=f"reference (cl. {kref})", Units="mean-normalised (017)", **{"R+ d": f"{t40['d_R+_median']:+.2f} (p {P(t40['d_R+_p'])})",
             "R− d": f"{t40['d_R-_median']:+.2f} (p {P(t40['d_R-_p'])})", "MW p": P(t40['d_RpRm_mw_p']),
             "Welch p": P(t40['d_RpRm_welch_p'])}),
        dict(Fit="main (cl. 10)", Units="SD (018)", **{"R+ d": f"{d_main['mp']} (p {d_main['pp']})",
             "R− d": f"{d_main['mm']} (p {d_main['pm']})", "MW p": d_main['mw'], "Welch p": d_main['we']}),
        dict(Fit=f"reference (cl. {kref})", Units="SD (018)", **{"R+ d": f"{d_ref['mp']} (p {d_ref['pp']})",
             "R− d": f"{d_ref['mm']} (p {d_ref['pm']})", "MW p": d_ref['mw'], "Welch p": d_ref['we']})]),
        "Trend contrast $d$ (" + ref("eq-d") + ") per fit and normalisation: median per cohort (Wilcoxon p vs 0) and R+ "
        "vs R− tests; mice with a member; uncorrected."))

    md.append("## The rise is not explained by hits, reward history, pre-stimulus state or performance\n")
    md.append(R.claim("factors",
        f"Per-mouse regressions ({ref('eq-ols')}) on active whisker trials ({ref('fig-factors')}C–K, {ref('tbl-reg')}): "
        f"the position coefficient is positive in R+ and ≈ 0 in R− ({txt(REGS['pos'])}); it is unchanged when the "
        f"whisker and auditory performance curves are added ({txt(REGS['pos_m2'])}), whose own coefficients are small. "
        f"Hits carry a larger weight than misses in both cohorts, more in R+ ({txt(REGS['hit'])}). A higher "
        f"pre-stimulus population rate lowers the component in both cohorts alike ({txt(REGS['base_pop'])}); a "
        f"reward on the previous trial lowers it slightly ({txt(REGS['prev_rew'])}). The reference fit gives a "
        f"position coefficient {txt(pos_ref)}; model 2: {txt(pos2_ref)}.",
        figure="Fig. 3", stat="OLS per mouse; Wilcoxon; MW + Welch", n=f"{REGS['pos']['np_']} / {REGS['pos']['nm']} mice",
        status="fragile" if float(R.numbers['reg_pos_welch']) >= 0.05 else "holds",
        note="cohort difference of the position coefficient borderline") + "\n")
    md.append(
        f"Panel A of {ref('fig-factors')} gives the session course for each trial class in SD units with the passive "
        f"blocks at either end; panel B bins whisker trials by the learning-curve level instead of position: in R+ "
        f"misses become weaker as performance rises (misses at high performance are lapses), while hits change little "
        f"— the weight tracks the response outcome more than the performance level.\n")
    md.append(figure("fig-factors", FC / "deep_splitnn_cl10_factors.png",
        "Trial factors of the member score $z$ (" + ref("eq-y") + "), main fit, whisker day 0, mice with a member. "
        "**A** Session course: fifths of the active block per class (whisker hit filled circles, whisker miss open "
        "squares, auditory hit triangles) and passive pre / post whisker trials (diamonds), mean ± SEM over mice; green "
        "R+, magenta R−. **B** Whisker hits (filled) and misses (open) binned by the whisker learning curve "
        "P(lick | whisker), mean ± SEM over mice. **C–G** Standardised coefficients of " + ref("eq-ols") + " (model 1) "
        "per mouse: position, hit, previous trial rewarded, pre-stimulus population rate, log ITI. **H** Passive post − "
        "pre whisker score. **I–K** Model 2 (+ performance curves): position, P(lick | whisker), P(lick | auditory). "
        "Dots = mice, bar = median; n and Wilcoxon p (vs 0) on top; title: R+ vs R− Mann–Whitney and Welch p.",
        width=95, script="018_candidate_deep.py"))
    reg_rows = []
    for k, lab in [("pos", "position"), ("hit", "hit"), ("prev_rew", "previous reward"), ("base_pop", "pre-stimulus rate"),
                   ("log_iti", "log ITI"), ("pos_m2", "position (model 2)"), ("p_wh_i_m2", "P(lick | whisker) (model 2)"),
                   ("p_au_i_m2", "P(lick | auditory) (model 2)")]:
        s = REGS[k]
        reg_rows.append({"Predictor": lab, "R+ median (p)": f"{s['mp']} ({s['pp']})", "R− median (p)": f"{s['mm']} ({s['pm']})",
                         "MW p": s["mw"], "Welch p": s["we"]})
    md.append(table("tbl-reg", pd.DataFrame(reg_rows),
        "Standardised regression coefficients of " + ref("eq-ols") + " on active whisker trials, main fit, day 0; "
        "median per cohort with Wilcoxon p (vs 0), R+ vs R− tests; n = " + REGS['pos']['np_'] + " R+ / " +
        REGS['pos']['nm'] + " R− mice; uncorrected."))

    md.append("## The change sits in Ss-whisker\n")
    md.append(R.claim("area",
        f"Ss-whisker carries a median {share_ss} of the member projection on active whisker trials and the midbrain "
        f"{share_mb} ({ref('fig-area')}A, {ref('eq-area')}). The cohort difference in the trend is in Ss-whisker "
        f"({txt(ar_ss)}) and weaker in the midbrain ({txt(ar_mb)}) ({ref('fig-area')}B). Without TCA, "
        f"{n_resp} whisker-responsive units of {n_units} day-0 units: Ss-whisker units show the same cohort difference "
        f"({txt(un_ss)}), thalamus ({txt(un_th)}) and midbrain ({txt(un_mb)}) do not ({ref('fig-area')}C, E–H). "
        f"Within Ss-whisker, the cohort difference is near p = 0.05 in both SSp-bfd ({txt(un_bfd)}) and SSs "
        f"({txt(un_sss)}), with few mice per area; "
        f"in the midbrain, SCm differs ({txt(un_scm)}) ({ref('fig-area')}D).",
        figure="Fig. 4", stat="MW + Welch", n="mice per area", status="holds",
        note="single-area results are small-n and uncorrected") + "\n")
    md.append(figure("fig-area", FC / "deep_splitnn_cl10_area.png",
        "Where is the trend? **A** Share of the member projection per area group (" + ref("eq-area") + "), active "
        "whisker trials, median and interquartile range over mice (left bar R+, right R−; an unrecorded area counts 0). "
        "**B** Contribution of each area group to the projection's $d$ (" + ref("eq-d") + ", SD units; contributions add "
        "up), mean ± SEM over mice; Mann–Whitney (MW) and Welch (W) p printed where MW < 0.05. **C** No TCA: "
        "whisker-responsive units (" + ref("eq-ev") + ", Wilcoxon p < 0.01), $d$ of the evoked response per mouse "
        "(mean over units), area groups with ≥ 5 mice per cohort, mean ± SEM; top: MW / Welch p. **D** Same for single "
        "areas (14 areas with most mice). **E–H** Session course of the evoked response of responsive units in "
        "Ss-whisker, thalamus, midbrain, striatum: fifths per class (whisker hit filled circles, miss open squares, "
        "auditory hit triangles), passive pre / post (diamonds), mean ± SEM over mice. Whisker day 0, all mice.",
        width=82, script="018_candidate_deep.py"))

    md.append("## Simpler population methods recover the R− decrease, not the R+ rise\n")
    s_t, s_p, s_s = MS["template"][0], MS["pca"][0], MS["stimaxis"][0]
    md.append(R.claim("methods",
        f"The template projection ({ref('eq-tmpl')}) correlates with the member score within sessions (median "
        f"Spearman r {agree['template']}; PCA {agree['pca']}, stimulus axis {agree['stimaxis']}). Over all day-0 mice "
        f"it shows the R− decrease but no R+ rise ({txt(s_t)}). PCA ({txt(s_p)}) and the stimulus axis "
        f"({ref('eq-stim')}; {txt(s_s)}) show no cohort difference; restricted to the mice with a TCA member, the "
        f"stimulus axis differs by Mann–Whitney only ({txt(memb['stimaxis'])}) ({ref('fig-methods')}, "
        f"{ref('tbl-methods')}).", figure="Fig. 5", stat="Wilcoxon; MW + Welch", n="all day-0 mice",
        status="fragile", note="R+ rise is specific to the TCA component") + "\n")
    md.append(
        "A whole-population axis mixes the component with other evoked activity (later responses, other areas, "
        "auditory-shared activity); the template and the stimulus axis are dominated by many units that do not change. "
        "The TCA component isolates a subset of units and a time window, which is where the R+ rise is; the "
        "single-unit Ss-whisker result above supports that the rise is real but local.\n")
    md.append(figure("fig-methods", FC / "deep_splitnn_cl10_methods.png",
        "Other methods, whisker day 0. Rows: TCA member score; template (" + ref("eq-tmpl") + "); PCA (best whisker "
        "PC of 5, uncentred); stimulus axis (" + ref("eq-stim") + "). Left: session course in SD units (" + ref("eq-y") +
        ") per class (as Figure 3A), mean ± SEM over mice (all day-0 mice for the non-TCA scores; auditory hits fall "
        "far below 0 for PCA and the stimulus axis, which separate whisker from auditory by construction). Middle: $d$ "
        "per mouse (" + ref("eq-d") + "). Right: top row, per-session Spearman correlation of each score with the member "
        "score on active whisker trials; other rows, $d$ restricted to the mice with a member. Bar = median; Wilcoxon "
        "p per cohort, MW and Welch p R+ vs R−.", width=90, script="018_candidate_deep.py"))
    mrows = []
    for sc, lab in D18.SCORES.items():
        s = MS[sc][0]
        mrows.append({"Method": lab, "Mice": "member" if sc == "member" else "all", "R+ d (p)": f"{s['mp']} ({s['pp']})",
                      "R− d (p)": f"{s['mm']} ({s['pm']})", "MW p": s["mw"], "Welch p": s["we"]})
        if sc != "member":
            s = memb[sc]
            mrows.append({"Method": lab, "Mice": "member", "R+ d (p)": f"{s['mp']} ({s['pp']})",
                          "R− d (p)": f"{s['mm']} ({s['pm']})", "MW p": s["mw"], "Welch p": s["we"]})
    md.append(table("tbl-methods", pd.DataFrame(mrows),
        "Trend contrast $d$ (" + ref("eq-d") + ", SD units) for TCA and the three other methods, main fit, whisker "
        "day 0; all mice or mice with a member; median (Wilcoxon p), R+ vs R− tests; uncorrected."))

    md.append("## The component and its R+ rise are still present in expert sessions\n")
    md.append(R.claim("expert",
        f"A component matching cluster 10 was found in {ex_n['R+'][0]} of {ex_n['R+'][1]} R+ and {ex_n['R-'][0]} of "
        f"{ex_n['R-'][1]} R− expert sessions (false-positive rate on day-0 non-member sessions {fp} %; "
        f"{ref('fig-expert')}A–B). In R+ expert sessions the weight still rises within the session on whisker hits "
        f"({txt(ex_hit)}) and misses ({txt(ex_miss)}); the position coefficient of {ref('eq-ols')} is "
        f"{txt(ex_pos)}; $d$: {txt(ex_d)} ({ref('fig-expert')}C–H). Ss-whisker responsive units: the evoked rate on "
        f"whisker hits goes from {hz['R+']['first']} Hz (first third) to {hz['R+']['last']} Hz (last third) in R+ "
        f"(p {hz['R+']['p']}, n = {hz['R+']['n']}) and from {hz['R-']['first']} to {hz['R-']['last']} Hz in R− "
        f"(p {hz['R-']['p']}, n = {hz['R-']['n']}); expert sessions: {hz['R+']['exp']} Hz (R+), {hz['R-']['exp']} Hz "
        f"(R−), different mice and units ({ref('fig-expert')}F).",
        figure="Fig. 6", stat="Wilcoxon; MW + Welch", n="expert sessions", status="holds",
        note="expert n small; within-session rise is not learning-day specific") + "\n")
    md.append(figure("fig-expert", FC / "deep_splitnn_cl10_expert.png",
        "Expert sessions (whisker day 1–4). **A** Mean similarity (" + ref("eq-sim") + ") of each session's best "
        "component to the day-0 members: day-0 members (leave-one-out, grey), best component of day-0 sessions without "
        "a member (black), expert sessions (red); dotted line = threshold. **B** Share of sessions with the component, "
        "day 0 (light) and expert (dark), with counts. **C** Session course of matched expert components (as Figure 3A). "
        "**D–E** $d$ per mouse at day 0 and in expert sessions. **F** No TCA: evoked rate (Hz) of whisker-responsive "
        "Ss-whisker units on whisker hits, first and last third of the day-0 active block and expert sessions (mouse "
        "means, bar = median); Wilcoxon first vs last. **G–H** Expert whisker-hit and whisker-miss slopes. **I** Hit − "
        "miss weight (SD) at day 0 and expert. Green R+, magenta R−.", width=95, script="018_candidate_deep.py"))

    # ---------------------------------------------------------------- discussion
    md.append("# Discussion\n")
    md.append(
        "**What the component is.** An early (≈ 15–40 ms) whisker-evoked response, largest in Ss-whisker with a "
        "midbrain (superior colliculus) part, specific to whisker trials, stronger on hits than misses, and weaker when "
        "the pre-stimulus population rate is high. These properties fit the primary sensory response and its known "
        "state dependence.\n")
    md.append(
        "**What changes.** Within the session, in R+ mice the response grows on whisker hits and misses; in R− mice "
        "it shrinks on whisker hits. Auditory hits, which share lick and reward, do not change, and the growth is not "
        "explained by hit / miss, previous reward, pre-stimulus rate, ITI or the performance curves. The difference is "
        "in Ss-whisker units and is seen without TCA in single units.\n")
    md.append(
        "**Is it learning?** Not established. The same within-session rise is present in R+ expert sessions, when the "
        "association is already learned, so the most parsimonious reading is a within-session effect of the reward "
        "contingency on the early whisker response (rewarded stimulus: response grows; unrewarded stimulus: response "
        "adapts), present each day, rather than a learning-day-specific change. The passive blocks do not show a "
        "lasting change, so the effect is context dependent (present in the active block only). Alternatives that are "
        "not excluded: whisker movement / arousal specific to R+ (whisking before the stimulus changes the evoked "
        "response), unit drift that correlates with time in a cohort-specific way, and the selection itself (one "
        "cluster out of " + num('n_tested2', int(n_tested.loc[('_splitnn', 'stim_sensory')])) + " tested, uncorrected).\n")
    md.append(
        "**How to say that the signal is in area X.** Four levels of evidence, used here in order of strength: (1) the "
        "component's weight is concentrated in X beyond what the unit count predicts (area specificity against a "
        "within-session permutation of area labels, 012; the share in Figure 4A); (2) the trend itself, not only the "
        "weight, is carried by X (area-restricted projection, Figure 4B, which separates where the component is from "
        "where it changes); (3) the change is present in X without the decomposition (single units, Figure 4C–H), "
        "with coverage controlled — compare within the mice that have both X and Y recorded, paired, so that recording "
        "coverage cannot create the difference; (4) timing — onset latency in X before Y (e.g. VPM/POm → SSp-bfd → SC) "
        "places the change in the pathway. Causality would need perturbation (not available here). By these criteria "
        "the change is in Ss-whisker (levels 1–3); the midbrain carries weight but its change is weak.\n")
    md.append("# Caveats and limitations\n")
    md.append(
        "- Exploratory selection among " + num('n_tested3', int(n_tested.loc[('_splitnn', 'stim_sensory')] +
                                                                  n_tested.loc[('_splitnn', 'stim_long')])) +
        " clusters of the main fit, uncorrected p values; the reference fit is not an independent sample (same data).\n"
        "- The Welch test of the selection statistic is borderline (p " + pv('t_we3', t10['d_RpRm_welch_p']) + "); in "
        "SD units it is significant. The R− distribution has outlier mice (" + ref('fig-s-exRm') + ").\n"
        "- The auditory control is weak for a whisker-specific component (auditory weights ≈ 0); R− is the stronger "
        "control.\n"
        "- Expert sessions are few (" + nex['R+'] + " R+, " + nex['R-'] + " R−), from partly different mice, and matched "
        "with an 8 % false-positive rate.\n"
        "- Trial weights are fitted with the whole session, so the trend is not cross-validated over time.\n"
        "- The tensors' two bins next to the stimulus contain Poisson-filled artefact windows (5–7 % of sensory "
        "components still peak there); the component peaks at 22.5 ms, outside them.\n")
    md.append("# Next steps and further controls\n")
    md.append(
        "1. **Movement and arousal**: pre-stimulus whisking (DLC whisker / top view), pupil (ssl-pupil-qc) and jaw "
        "movement as covariates in " + ref("eq-ols") + "; the pre-stimulus population rate is a partial proxy only.\n"
        "2. **Unit stability**: repeat with drift-stable units only and per-unit amplitude / firing-rate trends; a "
        "drift component would also change on auditory trials, which argues against it, but cohort-specific drift is "
        "not excluded.\n"
        "3. **Time-matched comparison**: R+ and R− sessions matched on block length and trial counts; trials matched "
        "on lick latency and on the previous trial's outcome.\n"
        "4. **Cross-validated trend**: fit TCA on the first half of the active block, project the second half (and "
        "vice versa) to test the rise out of sample.\n"
        "5. **Other decompositions**: sliceTCA (Pellegrino et al. 2024) with trial-slicing components, which allows a "
        "per-trial change of the time course and neuron loadings; demixed PCA with time × stimulus × outcome × "
        "block-position marginals; both on the same tensors (Kuma GPU).\n"
        "6. **Learning-specific test**: align to the identified learning trial (ssl-learning-trial-identification) and "
        "compare the slope before and after; compare day 0 with the expert sessions of the same mice, paired.\n"
        "7. **Latency**: onset and peak per area (VPM / POm, SSp-bfd layers, SC) on the evoked responses of the member "
        "units, to see whether the change starts in thalamus or cortex.\n")

    # ---------------------------------------------------------------- tables / supplement
    nrows = []
    for c in ("R+", "R-"):
        nrows.append({"Cohort": c, "Day-0 sessions": int(((ses.stage == "learning") & (ses.cohort == c)).sum()),
                      "Day-0 mice with member": int(c10[f"n_{'Rp' if c == 'R+' else 'Rm'}"]),
                      "Expert sessions": int(((ses.stage == "expert") & (ses.cohort == c)).sum()),
                      "Expert matched": int(e[e.cohort == c].passes.sum()),
                      "Responsive units (day 0)": int(((u0.cohort == c)).sum()),
                      "Ss-whisker responsive units": int(((u0.cohort == c) & (u0.area_group == "Somatosensory-whisker")).sum())})
    md.append("# Tables\n")
    md.append(table("tbl-n", pd.DataFrame(nrows),
        "Sample sizes: sessions, mice with a member of cluster 10, expert sessions and matches, and whisker-responsive "
        "units with a defined $d$ (day 0)."))
    md.append("# Supplementary figures\n")
    md.append(figure("fig-s-ref", top40,
        f"Reference fit (non-negative, rates), cluster {kref}, matched to main-fit cluster 10; panels as Figure 1.",
        width=92, kind="supplementary", script="016_candidate_details.py"))
    md.append(figure("fig-s-reftrend", F / "trend_candidates_nn.png",
        "Trend selection in the reference fit; panels as Figure 2.", kind="supplementary", script="017_trend_contrasts.py"))
    md.append(figure("fig-s-exRp", FC / "deep_splitnn_cl10_examples_Rp.png",
        "All R+ mice with a member, sorted by $d$ (lowest first). Trial weight in SD units of the active whisker trials "
        "(0 = their mean) for every trial in session order: whisker trials orange (filled hit, open miss), auditory "
        "blue; black = running median of whisker trials (15 trials); green line = P(lick | whisker) learning curve "
        "(right axis); grey = passive blocks.", width=90, kind="supplementary", script="018_candidate_deep.py"))
    md.append(figure("fig-s-exRm", FC / "deep_splitnn_cl10_examples_Rm.png",
        "All R− mice with a member, as Figure S3 (magenta line = P(lick | whisker)).", width=90, kind="supplementary",
        script="018_candidate_deep.py"))
    md.append(figure("fig-s-refarea", FC / f"deep_nn_cl{kref}_area.png",
        f"Area attribution in the reference fit (cluster {kref}); panels as Figure 4 (C–H do not depend on the fit).",
        width=82, kind="supplementary", script="018_candidate_deep.py"))
    md.append(
        f"Supplementary material: reference-fit detail ({ref('fig-s-ref')}) and trend ({ref('fig-s-reftrend')}); all "
        f"example mice ({ref('fig-s-exRp')}, {ref('fig-s-exRm')}); reference-fit area attribution "
        f"({ref('fig-s-refarea')}).\n")
    md.append("# Appendix\n")
    md.append(table("tbl-s-params", pd.DataFrame([
        ("Bins", "10 ms, −52.5 to +107.5 ms (16 bins)"), ("Artefact window", "−10 to +5 ms, Poisson fill at the trial baseline"),
        ("Baseline", "−1.0 to −0.010 s, same trial"), ("Min unit rate", "0.5 Hz"), ("TCA ranks", "1–12, 10 restarts, GPU"),
        ("Cluster threshold", "95th pct within-session similarity"), ("Min mice per cluster", "10"),
        ("Min trials per slope", f"{D18.MIN_N}"), ("Fifths", f"{D18.NB}"), ("Evoked window (units)", "bins 12.5–42.5 ms"),
        ("Responsive unit", "Wilcoxon p < 0.01, mean > 0, all whisker trials"), ("Other methods' bins", "≥ +7.5 ms"),
        ("Expert match threshold", "10th pct of members' LOO similarity"), ("Statistics", "Wilcoxon; MW + Welch; uncorrected")],
        columns=["Parameter", "Value"]), "Parameters."))
    md.append(R.registry_table("tbl-s-registry"))
    md.append(
        f"Parameters in {ref('tbl-s-params')}; claims in {ref('tbl-s-registry')}. Results: "
        f"combined_results_ks4/ssl-tca-session-wide/ (tables/deep_*_cl*, trend_*, clusters*; figures/aligned_gpu/"
        f"candidates/). Code: projects/ssl-tca-session-wide/ (005, 006g, 012–018, report/build_candidate_report.py).\n")
    text = "\n".join(md)
    summary = ("# Candidate cl. 10 (stim_sensory): summary\n\n" + text.split("# Key results\n")[1].split("# Introduction")[0])
    R.write(OUT, text, summary=summary)


if __name__ == "__main__":
    main()
