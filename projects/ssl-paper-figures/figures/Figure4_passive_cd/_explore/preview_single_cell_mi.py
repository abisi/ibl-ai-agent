"""Figure 4, single-cell view of the passive change (user 2026-11-26, draft: "show the effect as CDF for the cohorts and test
difference in distributions. Also derive a modulation index (or several types) that quantify single-cell effect and show the
CDF from -1 to 1; show several variants, and highlight what -1 means vs +1").

Neurons: as the passive analyses (179 stable good + MUA units of learners, whisker day 0; no responsiveness or rate
criterion, user 2026-11-26). Per neuron (179): evoked response ev = mean rate 5-35 ms
minus each trial's own baseline (spikes/s), passive pre / post, whisker W / auditory A; baseline rate b (-1 s..-15 ms).

Raw effects (spikes/s): dW = ev_post_W - ev_pre_W, dA, dW - dA.
Modulation indices, all in [-1, 1]:
  MI evoked (W, A)   (ev_post - ev_pre) / (|ev_post| + |ev_pre|)
                     -1 = evoked response only before the task (lost), +1 = only after (gained), 0 = unchanged
  MI rate (W, A)     (r_post - r_pre) / (r_post + r_pre), r = b + ev = total rate 5-35 ms (>= 0; includes baseline changes)
                     -1 = silent in the window after the task, +1 = silent before
  Preference shift   (SI_post - SI_pre) / 2, SI = (ev_W - ev_A) / (|ev_W| + |ev_A|) (whisker vs auditory preference)
                     -1 = from purely whisker- to purely auditory-preferring, +1 = from auditory- to whisker-preferring
  MI interaction     (dW - dA) / (|dW| + |dA|)
                     -1 = whisker response decreased relative to auditory, +1 = whisker increased relative to auditory
Variants without the +-1 pile-up (user 2026-11-26: "how to remove the pile up"; sign flips of baseline-subtracted responses
put a neuron at exactly -1 / +1):
  MI evoked, positive responders   as MI evoked, only neurons with a positive response to that stimulus in ACTIVE trials (ROC
                     significant, direction > 0): selection independent of the passive data (selecting on a positive
                     passive-pre response would regress to the mean and bias the MI negative)
  MI evoked, soft    (post - pre) / (|post| + |pre| + c), c = 5 spikes/s
  MI interaction, response-normalised   (dW - dA) / (|pre W| + |post W| + |pre A| + |post A|): the change normalised by the
                     neuron's response size, bounded in [-1, 1], no sign-flip saturation
Pile-up: fraction of neurons with |MI| > 0.99, written on each bounded panel.
Tests, R+ vs R-: neuron level, Kolmogorov-Smirnov and Mann-Whitney (neurons pooled across sessions: anti-conservative, the
neurons of a session are not independent); session level, per-session median, Mann-Whitney + Welch (the unit of analysis).
Output: Figures/_explore/Figure4_passive_cd_<date>/single_cell/single_cell_mi.{png,pdf,svg} + stats csv
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_single_cell_mi.py
"""
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

FS = 5
COH = ["R+", "R-"]
COL = {"R+": "#00B400", "R-": "#C800C8"}
LAB = {"R+": "R+", "R-": "R−"}


SOFT_C = 5.0


def roc_direction(u):
    """per unit: positive responder to whisker / auditory in active trials (ROC whisker_active / auditory_active significant
    with direction > 0)"""
    root = S.axel_bisi_root() / "combined_results_ks4"
    out = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "significant",
                                    "direction"])
        r = r[r.analysis_type.isin(["whisker_active", "auditory_active"])]
        d = pd.to_numeric(r.direction, errors="coerce")
        if d.isna().all():
            d = r.direction.astype(str).str.lower().map({"positive": 1, "up": 1, "excited": 1}).fillna(-1)
        r = r.assign(pos=r.significant.astype(bool) & (d > 0))
        w = r.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type", values="pos",
                          aggfunc="first").reset_index()
        out.append(w.rename(columns={"cluster_id": "cid", "whisker_active": "pos_W", "auditory_active": "pos_A"}))
    r = pd.concat(out, ignore_index=True)
    r["cid"] = r.cid.astype(int)
    r["pos_W"] = r.pos_W.fillna(False).astype(bool)
    r["pos_A"] = r.pos_A.fillna(False).astype(bool)
    return r


def units():
    """same neurons as the passive analyses (E-G, 179): all stable good + MUA units of learners, whisker day 0, no
    responsiveness or rate criterion (user 2026-11-26: the sensory-responsive and baseline > 0.5 spikes/s filters are for the
    panel A display only)"""
    u = S._ag_units()
    u["cid"] = (u.cluster_id % 1_000_000).astype(int)
    u = u.merge(roc_direction(u), on=["session_id", "electrode_group", "cid"], how="left")
    with np.errstate(all="ignore"):
        u["dW"] = u.ev_post_W - u.ev_pre_W
        u["dA"] = u.ev_post_A - u.ev_pre_A
        u["dWA"] = u.dW - u.dA
        for s in "WA":
            u[f"mi_ev_{s}"] = (u[f"ev_post_{s}"] - u[f"ev_pre_{s}"]) / (u[f"ev_post_{s}"].abs() + u[f"ev_pre_{s}"].abs())
            rpre = (u[f"base_pre_{s}"] + u[f"ev_pre_{s}"]).clip(lower=0)
            rpost = (u[f"base_post_{s}"] + u[f"ev_post_{s}"]).clip(lower=0)
            u[f"mi_rate_{s}"] = (rpost - rpre) / (rpost + rpre)
        si = {e: (u[f"ev_{e}_W"] - u[f"ev_{e}_A"]) / (u[f"ev_{e}_W"].abs() + u[f"ev_{e}_A"].abs()) for e in ("pre", "post")}
        u["si_shift"] = (si["post"] - si["pre"]) / 2
        u["mi_int"] = (u.dW - u.dA) / (u.dW.abs() + u.dA.abs())
        for s in "WA":
            u[f"mi_ev_{s}_pos"] = u[f"mi_ev_{s}"].where(u[f"pos_{s}"])
            u[f"mi_soft_{s}"] = (u[f"ev_post_{s}"] - u[f"ev_pre_{s}"]) / (u[f"ev_post_{s}"].abs() + u[f"ev_pre_{s}"].abs() + SOFT_C)
        u["mi_int_norm"] = (u.dW - u.dA) / (u.ev_pre_W.abs() + u.ev_post_W.abs() + u.ev_pre_A.abs() + u.ev_post_A.abs())
    return u


PANELS = [
    # key, title, x label, bounded, (-1 meaning, +1 meaning)
    ("dW", "Whisker, raw change", "Evoked change (spks/s, post − pre)", False, None),
    ("dA", "Auditory, raw change", "Evoked change (spks/s, post − pre)", False, None),
    ("dWA", "Whisker − auditory, raw change", "Whisker − auditory change (spks/s)", False, None),
    ("mi_ev_W", "MI evoked, whisker", "(post − pre) / (|post| + |pre|)", True, ("response lost\nafter task", "response gained\nafter task")),
    ("mi_ev_A", "MI evoked, auditory", "(post − pre) / (|post| + |pre|)", True, ("response lost\nafter task", "response gained\nafter task")),
    ("mi_rate_W", "MI total rate, whisker", "(r post − r pre) / (r post + r pre)", True, ("silent after\ntask", "silent before\ntask")),
    ("mi_rate_A", "MI total rate, auditory", "(r post − r pre) / (r post + r pre)", True, ("silent after\ntask", "silent before\ntask")),
    ("si_shift", "Preference shift", "(SI post − SI pre) / 2", True, ("whisker → auditory\npreferring", "auditory → whisker\npreferring")),
    ("mi_int", "MI interaction", "(ΔW − ΔA) / (|ΔW| + |ΔA|)", True, ("whisker decreased\nrelative to auditory", "whisker increased\nrelative to auditory")),
    ("mi_ev_W_pos", "MI evoked, whisker, + responders", "(post − pre) / (|post| + |pre|)", True, ("response lost\nafter task", "response gained\nafter task")),
    ("mi_ev_A_pos", "MI evoked, auditory, + responders", "(post − pre) / (|post| + |pre|)", True, ("response lost\nafter task", "response gained\nafter task")),
    ("mi_soft_W", "MI evoked soft, whisker", "(post − pre) / (|post| + |pre| + 5)", True, ("response lost\nafter task", "response gained\nafter task")),
    ("mi_soft_A", "MI evoked soft, auditory", "(post − pre) / (|post| + |pre| + 5)", True, ("response lost\nafter task", "response gained\nafter task")),
    ("mi_int_norm", "MI interaction, response-normalised", "(ΔW − ΔA) / Σ|responses|", True, ("whisker decreased\nrelative to auditory", "whisker increased\nrelative to auditory")),
]


def ptxt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def cdf_panel(ax, u, key, title, xlabel, bounded, ends):
    res = dict(measure=key)
    vals = {c: u[u.cohort == c][key].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for c in COH}
    for c in COH:
        v = np.sort(vals[c])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[c], lw=0.8, label=f"{LAB[c]} (n = {len(v):,})")
        ax.axvline(np.median(v), color=COL[c], lw=0.4, ls=(0, (2, 2)))
        res[f"median_{c}"] = float(np.median(v))
        res[f"n_{c}"] = len(v)
    ks = stats.ks_2samp(vals["R+"], vals["R-"])
    mw = stats.mannwhitneyu(vals["R+"], vals["R-"])
    sess = u[["session_id", "cohort", key]].replace([np.inf, -np.inf], np.nan).dropna()
    sm = sess.groupby(["session_id", "cohort"])[key].median().reset_index()
    a_, b_ = sm[sm.cohort == "R+"][key], sm[sm.cohort == "R-"][key]
    ps = max(stats.mannwhitneyu(a_, b_).pvalue, stats.ttest_ind(a_, b_, equal_var=False).pvalue)
    res.update(ks_D=float(ks.statistic), ks_p=float(ks.pvalue), mw_p=float(mw.pvalue), session_p=float(ps),
               n_sessions_Rp=len(a_), n_sessions_Rm=len(b_))
    allv = np.r_[vals["R+"], vals["R-"]]
    res["pileup"] = float(np.mean(np.abs(allv) > 0.99)) if bounded else np.nan
    pile = f"\n|MI| > 0.99: {res['pileup']:.0%}" if bounded else ""
    ax.text(0.03, 0.97, f"Neurons: KS D = {ks.statistic:.2f}, {ptxt(ks.pvalue)}\nMW {ptxt(mw.pvalue)}\n"
                        f"Sessions (medians): {ptxt(ps)}{pile}", transform=ax.transAxes, ha="left", va="top", fontsize=FS - 1)
    if bounded:
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.axvline(0, color="0.6", lw=0.4)
        lo, hi = ends
        ax.text(-1.0, -0.30, f"−1: {lo}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=FS - 1.3,
                color="0.3")
        ax.text(1.0, -0.30, f"+1: {hi}", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=FS - 1.3,
                color="0.3")
        # inset: R- minus R+ CDF difference
    else:
        allv = np.r_[vals["R+"], vals["R-"]]
        lo, hi = np.percentile(allv, [1, 99])
        ax.set_xlim(lo, hi)
        ax.axvline(0, color="0.6", lw=0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Cumulative fraction of neurons", fontsize=FS)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_title(title, fontsize=FS + 0.5, pad=2)
    ax.tick_params(labelsize=FS, length=1.5)
    return res


def main():
    set_nature_rc()
    u = units()
    fig = plt.figure(figsize=(183 * MM, 200 * MM))
    gs = fig.add_gridspec(4, 4, left=0.06, right=0.99, top=0.95, bottom=0.06, wspace=0.42, hspace=1.0)
    pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2)]
    res = []
    for (r, c), (key, title, xl, bounded, ends) in zip(pos, PANELS):
        ax = fig.add_subplot(gs[r, c])
        res.append(cdf_panel(ax, u, key, title, xl, bounded, ends))
        if (r, c) == (0, 0):
            ax.legend(fontsize=FS - 0.5, frameon=False, loc="lower right", handlelength=1.0)
    fig.text(0.01, 0.985, f"Single-neuron change of the passive evoked response (5–35 ms), R+ vs R−: all stable neurons "
                          f"({len(u):,}; {u.session_id.nunique()} sessions). Draft.", fontsize=FS + 1.5, va="top")
    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}" / "single_cell"
    out.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, "single_cell_mi")
    r = pd.DataFrame(res)
    r.to_csv(out / "single_cell_mi_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string())
    # focused figure (user 2026-11-26: "do the response-normalised interaction index then"): raw whisker - auditory change and
    # the response-normalised interaction index, with a session-level view (per-session median, mean +- s.e.m.)
    f2 = plt.figure(figsize=(120 * MM, 50 * MM))
    g2 = f2.add_gridspec(1, 3, left=0.08, right=0.98, top=0.86, bottom=0.30, wspace=0.55, width_ratios=[1, 1, 0.45])
    for k, key in enumerate(("dWA", "mi_int_norm")):
        pnl = next(x for x in PANELS if x[0] == key)
        cdf_panel(f2.add_subplot(g2[0, k]), u, *pnl)
    ax = f2.add_subplot(g2[0, 2])
    sm = u[["session_id", "cohort", "mi_int_norm"]].dropna().groupby(["session_id", "cohort"]).mi_int_norm.median().reset_index()
    for q, c in enumerate(COH):
        v = sm[sm.cohort == c].mi_int_norm
        ax.scatter(q + np.random.default_rng(q).uniform(-0.12, 0.12, len(v)), v, s=2, color="0.75", lw=0, zorder=1)
        ax.errorbar(q, v.mean(), v.sem(), color=COL[c], marker="o", ms=3, capsize=0, lw=1.0, zorder=3)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks([0, 1], [LAB[c] for c in COH])
    for tl, c in zip(ax.get_xticklabels(), COH):
        tl.set_color(COL[c])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Session median of the\nresponse-normalised MI", fontsize=FS)
    ax.tick_params(labelsize=FS, length=1.5)
    f2.text(0.01, 0.99, f"All stable neurons ({len(u):,}; {u.session_id.nunique()} sessions)", fontsize=FS + 1, va="top")
    save_figure(f2, out, "single_cell_interaction_index")
    print("saved", out)


if __name__ == "__main__":
    main()
