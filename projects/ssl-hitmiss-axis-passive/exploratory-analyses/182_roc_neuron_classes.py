"""182 -- Neuron classes from the single-neuron ROC (per-mouse result folders) and the passive whisker change on the hit - miss
axis (user 2026-11-24: "in my roc rates analyses I do have hit/miss coding neurons and stimulus coding neurons ... use the roc
rates in the per mouse result folders"; Kingsbury et al. 2025: context-specific decision coding in separate subpopulations).

Units: 179_cd_units (whole brain, tracked stable units, learners as the figure, whisker day 0). ROC: combined_results_ks4/
<mouse>/whisker_0/roc_analysis/<mouse>_roc_results_new.csv (rate-based ROC, post window 5-35 ms, 1000 permutations,
p < 0.05 uncorrected); join keys session_id, electrode_group, cluster_id (179 stores imec * 1e6 + id).
Classes: hit / miss = whisker_choice (whisker hit vs miss, active); stimulus = wh_vs_aud_active (whisker vs auditory, active);
  hit-miss only, stimulus only, both, neither.
Measures per session x class:
  frac                  fraction of the session's neurons
  overlap               P(both) / (P(hit-miss) P(stimulus)) (1 = independent; > 1 shared neurons, < 1 separate subpopulations)
  d_rate                mean change of the passive whisker evoked rate, post - pre (spikes/s)
  contrib               contribution to the change of the passive whisker projection on the hit - miss axis:
                        sum over the class of u_i (m_post,i - m_pre,i) / sd_i / sqrt(N); the four classes add up exactly to the
                        session's d_along (179, data axis; the shifted-axis null cannot be split by neuron)
Stats: R+ vs R- per class (Mann-Whitney + Welch); overlap vs 1 per cohort (Wilcoxon + t); uncorrected.
Outputs: 182_roc_neuron_classes_sessions.parquet, 182_roc_neuron_classes_stats.csv, figures/182_roc_neuron_classes.png.
Run (haas, repo root): .venv/bin/python projects/ssl-hitmiss-axis-passive/exploratory-analyses/182_roc_neuron_classes.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA.parents[1] / "ssl-paper-figures"))
import shared_passive_cd as S  # noqa: E402
from common import axel_bisi_root  # noqa: E402
from paperstyle import COLORS  # noqa: E402

ROOT = axel_bisi_root() / "combined_results_ks4"
TYPES = {"whisker_choice": "hm", "wh_vs_aud_active": "st"}
CLASSES = ("hit-miss only", "stimulus only", "both", "neither")
K = ["session_id", "electrode_group", "cluster_id"]


def load_roc(mouse):
    f = ROOT / mouse / "whisker_0" / "roc_analysis" / f"{mouse}_roc_results_new.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f, usecols=K + ["analysis_type", "significant", "selectivity"])
    r = r[r.analysis_type.isin(TYPES)]
    w = r.pivot_table(index=K, columns="analysis_type", values="significant", aggfunc="first").rename(columns=TYPES)
    return w.astype(bool).reset_index()


def main():
    u = pd.read_parquet(S.TAB / "179_cd_units.parquet")
    u = u[u.mouse_id.isin(S.learners())].copy()
    u["cluster_id"] = (u.cluster_id % 1_000_000).astype(int)
    with ProcessPoolExecutor(32) as ex:
        roc = [r for r in ex.map(load_roc, sorted(u.mouse_id.unique())) if r is not None]
    roc = pd.concat(roc, ignore_index=True)
    roc["cluster_id"] = roc.cluster_id.astype(int)
    m = u.merge(roc, on=K, how="left")
    print(f"units {len(u)}, with ROC {m.hm.notna().sum()}, sessions {m.session_id.nunique()}", flush=True)
    m = m.dropna(subset=["hm", "st"])
    m["hm"], m["st"] = m.hm.astype(bool), m.st.astype(bool)
    m["cls"] = np.select([m.hm & ~m.st, ~m.hm & m.st, m.hm & m.st], CLASSES[:3], CLASSES[3])
    m["d_rate"] = m.ev_post_W - m.ev_pre_W
    rows = []
    for (sid, coh), g in m.groupby(["session_id", "reward_group"]):
        sN = np.sqrt(len(g))
        c_all = g.u_weight * (g.ev_post_W - g.ev_pre_W) / g.sd_active / sN
        ov = g.hm.mean() * g.st.mean()
        base = dict(session_id=sid, cohort=coh, n=len(g), d_along=float(c_all.sum()),
                    overlap=float((g.hm & g.st).mean() / ov) if ov > 0 else np.nan)
        for c in CLASSES:
            k = (g.cls == c).to_numpy()
            rows.append(dict(base, cls=c, frac=k.mean(), d_rate=g.d_rate[k].mean() if k.any() else np.nan,
                             contrib=float(c_all[k].sum()), mean_weight=g.u_weight[k].mean() if k.any() else np.nan))
    s = pd.DataFrame(rows)
    s.to_parquet(EA / "182_roc_neuron_classes_sessions.parquet", index=False)

    st = []
    for meas in ("frac", "d_rate", "contrib", "mean_weight"):
        for c in CLASSES:
            g = s[s.cls == c]
            t = S.tests(g, meas)
            st.append(dict(measure=meas, cls=c, test="R+ vs R-", **{k: t[k] for k in ("mean_Rp", "mean_Rm", "mannwhitney_p", "welch_p")}))
    ov = s.drop_duplicates("session_id")
    for c in S.COH:
        v = ov[ov.cohort == c].overlap.dropna()
        st.append(dict(measure="overlap", cls="hit-miss x stimulus", test=f"{c} vs 1", mean_Rp=v.mean(),
                       mannwhitney_p=stats.wilcoxon(v - 1).pvalue, welch_p=stats.ttest_1samp(v, 1).pvalue))
    t = S.tests(ov, "overlap")
    st.append(dict(measure="overlap", cls="hit-miss x stimulus", test="R+ vs R-", **{k: t[k] for k in ("mean_Rp", "mean_Rm", "mannwhitney_p", "welch_p")}))
    # within R-: does the hit-miss-coding part carry more of the change than the stimulus-only part? (paired)
    for c in S.COH:
        p = s[s.cohort == c].pivot(index="session_id", columns="cls", values="contrib")
        hm = p["hit-miss only"] + p["both"]
        st.append(dict(measure="contrib", cls="hit-miss coding (only + both) vs stimulus only", test=f"{c} paired",
                       mean_Rp=hm.mean(), mean_Rm=p["stimulus only"].mean(),
                       mannwhitney_p=stats.wilcoxon(hm, p["stimulus only"]).pvalue,
                       welch_p=stats.ttest_rel(hm, p["stimulus only"]).pvalue))
    st = pd.DataFrame(st)
    st.to_csv(EA / "182_roc_neuron_classes_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(4).to_string(index=False))

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, (meas, lab) in zip(axs, (("frac", "Fraction of neurons"), ("d_rate", "Δ whisker rate, post − pre (Hz)"),
                                     ("contrib", "Contribution to Δ whisker proj.\non hit − miss axis"),
                                     ("mean_weight", "Mean weight on hit − miss axis"))):
        for c, off in (("R+", -0.12), ("R-", 0.12)):
            g = s[s.cohort == c].groupby("cls")[meas].agg(["mean", "sem"]).reindex(CLASSES)
            ax.errorbar(np.arange(4) + off, g["mean"], g["sem"], fmt="o", color=COLORS[c], ms=3, capsize=0, label=c)
        ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_xticks(range(4), [c.replace(" only", "\nonly") for c in CLASSES], fontsize=6)
        ax.set_ylabel(lab, fontsize=7)
        r = st[(st.measure == meas) & (st.test == "R+ vs R-")].set_index("cls")
        yl = ax.get_ylim()
        for i, c in enumerate(CLASSES):
            ax.text(i, yl[1], f"{max(r.loc[c, 'mannwhitney_p'], r.loc[c, 'welch_p']):.2g}", ha="center", va="bottom", fontsize=5)
    axs[0].legend(fontsize=6)
    o = st[st.measure == "overlap"]
    fig.suptitle("ROC neuron classes (whisker hit vs miss; whisker vs auditory), 5-35 ms. Overlap obs/expected: "
                 + "; ".join(f"{r.test} {r.mean_Rp:.2f} p {max(r.mannwhitney_p, r.welch_p):.2g}" for r in o.itertuples()), fontsize=7)
    fig.tight_layout()
    (EA / "figures").mkdir(exist_ok=True)
    fig.savefig(EA / "figures" / "182_roc_neuron_classes.png", dpi=200)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
