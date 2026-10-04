"""013 -- Schematics and examples of every learning-trial (LT) method tried so far, plus a full
comparison table (user request 2026-09-25: "Schematize and exemplify these methods and add onto
the learning trial project report. Be exhaustive and show all methods so far.").

Methods (whisker-trial index, as the stored LT):
  L0  stored               stored H5 value (40-sample PyMC curve, sigma~1 prior, even-grid FA, 5-run
                           rule on the 80% band, clamp <=10 -> 10, R- fallback 10)
  L1  stored rule, exact   same rule on the exact posterior of the same model (FA at real times)
  L2  stored rule, smooth  same rule on the data-chosen-smoothness curve
  L3  sustained prob.      P(whisker > FA) >= 0.9 on 16/20 trials (R+); R- mirrored (<= 0.5)
  L4  whisker CP (ML)      maximum-likelihood change point of whisker outcomes only
  L5  whisker CP (Bayes)   Bayesian change point, whisker outcomes only
  L6  joint CP             Bayesian change point on whisker AND no-stim outcomes (strict, BF>10^0.5)
  L6_len                   same, lenient (BF > 1)
  L5w_len                  R+ only: whisker-only CP, FA as a floor (lenient)
  L7  half-way curve rule  smoothed D = whisker - FA: first crossing of half-way between start and
                           plateau, held 16/20; change >= 0.2 required
  L8  fixed-margin rule    smoothed D: R+ first rise above 0.2 held 16/20 (after having been below);
                           R- first fall below 0.1 held 16/20 (after having been above 0.2)
  lenient cascade          first of L6, L6_len, L5w_len, L7; + clean gate -> lt_lenient_clean
Outputs (exploratory-analyses/): 013_pipeline_schematic.png, 013_rules_schematic.png,
  013_rules_examples.png; artifacts/013_learning_trials_all_methods.csv, 013_methods_summary.csv
Run on haas.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import lt_lib as L  # noqa: E402



def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m002 = _load("m002", "002_exact_curves.py")
m003 = _load("m003", "003_learning_trial_rules.py")
m004 = _load("m004", "004_bayesian_changepoint.py")
m005 = _load("m005", "005_joint_changepoint.py")
m006 = _load("m006", "006_halfway_rule_and_final_table.py")
m007 = _load("m007", "007_lenient_learning_trials.py")

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
MCOL = {"L0 stored": "#d62728", "L1 stored rule, exact": "#ff9896", "L2 stored rule, smooth": "#e377c2",
        "L3 sustained prob.": "#bcbd22", "L5 whisker CP": "#7f7f7f", "L6 joint CP": "#1f77b4",
        "L7 half-way": "#ff7f0e", "L8 fixed margin": "#17becf"}
W, K = 20, 16


def fixed_margin(D, rg, hi=0.2, lo=0.1):
    n = len(D)
    if rg == 1:
        below = np.where(D < hi)[0]
        if len(below) == 0:
            return np.nan
        for s in range(below[0], n - W + 1):
            if D[s] >= hi and (D[s:s + W] >= hi).sum() >= K:
                return float(s)
        return np.nan
    above = np.where(D >= hi)[0]
    if len(above) == 0:
        return np.nan
    for s in range(above[0], n - W + 1):
        if D[s] <= lo and (D[s:s + W] <= lo).sum() >= K:
            return float(s)
    return np.nan


def all_methods(d, c, cp_joint=None, cp_w=None):
    """Every method's LT for one session (inputs dict d, curves dict c from 002)."""
    rg = 1 if d["reward_group"] == "R+" else 0
    wo = d["w_outcomes"]
    o, e = c["orig"], c["eb"]
    out = {}
    out["L1 stored rule, exact"] = m003.stored_rule(wo, o["p_mean"], o["p_low80"], o["p_high80"], o["fa_time"], rg)[0]
    out["L2 stored rule, smooth"] = m003.stored_rule(wo, e["p_mean"], e["p_low80"], e["p_high80"], e["fa_time"], rg)[0]
    out["L3 sustained prob."] = m003.sustain_rule(e["p_above"], wo, rg)
    rw = cp_w if cp_w is not None else m004.bayes_cp(wo, rg)
    out["L5 whisker CP"] = rw["median"] if rw is not None and rw["log10_bf"] > 0.5 else np.nan
    S = m005.Streams(wo, d["w_start"], d["n_outcomes"], d["n_start"])
    rj = cp_joint if cp_joint is not None else m005.joint_cp(S, rg)
    out["_joint"] = rj
    D = e["p_mean"] - e["fa_time"]
    out["L7 half-way"] = m006.halfway(D, rg)
    out["L8 fixed margin"] = fixed_margin(D, rg)
    out["_D"] = D
    out["_S"] = S
    return out


# ------------------------------------------------------------------------------------------------ schematic 1
def pipeline_schematic():
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec="#444444", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="#444444", lw=1.2))

    ax.text(0.2, 7.15, "STORED", fontsize=12, weight="bold", color="#d62728")
    ax.text(0.2, 3.55, "RE-ESTIMATED", fontsize=12, weight="bold", color="#1f77b4")
    rows = [
        (4.3, "#fde2e2", ["Whisker & no-stim licks\n(0/1 per trial)",
                          "Curve model per trial type\nrandom walk on logit p\nPyMC 10 tune / 10 draws\n(40 samples, unconverged)",
                          "Smoothness fixed by prior\n1/sigma^2 ~ Gamma(10,10)\n-> sigma ~ 1 (jumpy)",
                          "FA curve at whisker times\nplaced on EVEN time grid\n(mis-timed chance line)",
                          "Rule: p_low > FA on 5\nconsecutive trials; clamp\n<=10 -> 10; R- fallback 10",
                          "LT for (almost)\nevery session"]),
        (0.7, "#e2ecfd", ["Whisker & no-stim licks\n(0/1 per trial)",
                          "SAME curve model\nexact grid forward-backward\n(deterministic, converged)",
                          "Smoothness chosen by data\nevidence-weighted sigma\n(R+ ~0.5, R- ~0.14)",
                          "FA curve at REAL no-stim\ntrial times, then at\nwhisker times",
                          "Rule on whisker vs FA:\nhalf-way (curve) or joint\nchange point (raw licks)",
                          "Clean-separation gate\n-> LT, or 'no learning\nevent' category"]),
    ]
    for y, fc, labels in rows:
        for i, t in enumerate(labels):
            x = 0.2 + i * 2.95
            box(x, y, 2.55, 2.2, t, fc)
            if i:
                arrow(x - 0.36, y + 1.1, x - 0.02, y + 1.1)
    ax.text(9.0, 3.3, "The model stays the original one (random walk on the logit of lick probability, Bernoulli licks); "
                      "what changes: computation, smoothness, FA timing, rule, gate.", ha="center", fontsize=9.5, style="italic")
    fig.savefig(HERE / "013_pipeline_schematic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ schematic 2
def synthetic_session(rg, seed):
    rng = np.random.default_rng(seed)
    if rg == "R+":
        nw = 130
        pw = np.r_[np.full(15, 0.40), np.full(20, 0.62), np.full(25, 0.45), np.full(45, 0.95), np.full(25, 0.20)]
        pf = np.r_[np.full(60, 0.30), np.full(45, 0.25), np.full(25, 0.10)]
    else:
        nw = 200
        pw = np.r_[np.full(40, 0.60), np.linspace(0.60, 0.08, 80), np.full(80, 0.06)]
        pf = np.r_[np.linspace(0.15, 0.05, 120), np.full(80, 0.04)]
    wo = (rng.random(nw) < pw).astype(int)
    no = (rng.random(nw) < pf).astype(int)
    w_start = np.arange(nw) * 10.0 + rng.uniform(0, 3, nw)
    n_start = np.arange(nw) * 10.0 + 5 + rng.uniform(0, 3, nw)
    d = dict(session_id=f"synthetic_{rg}", mouse_id="synthetic", learning_category="synthetic", reward_group=rg, w_outcomes=wo, n_outcomes=no, w_start=w_start, w_stop=w_start + 2, n_start=n_start)
    return d, pw, pf


def rules_schematic():
    rows = []
    for rg, seed in (("R+", 3), ("R-", 5)):
        d, pw, pf = synthetic_session(rg, seed)
        _, c = m002.process(("synthetic", d))
        rows.append((rg, d, c, pw, pf, all_methods(d, c)))
    cols = ["A. stored rule (L1/L2)", "B. sustained probability (L3)", "C. fixed margin on D (L8)",
            "D. half-way on D (L7)", "E. joint change point (L6)"]
    fig, axes = plt.subplots(2, 5, figsize=(29, 9.5), constrained_layout=True)
    for r, (rg, d, c, pw, pf, m) in enumerate(rows):
        col = COHORT_COLOR[rg]
        x = np.arange(len(d["w_outcomes"]))
        o_, e = c["orig"], c["eb"]
        D = m["_D"]
        rgi = 1 if rg == "R+" else 0
        # A stored rule on the exact stored-prior curve
        ax = axes[r, 0]
        ax.fill_between(x, o_["p_low80"], o_["p_high80"], color=col, alpha=0.2, lw=0)
        ax.plot(x, o_["p_mean"], color=col, lw=1)
        ax.plot(x, o_["fa_time"], color="#555555", ls="--", lw=1)
        above = o_["p_low80"] > o_["fa_time"]
        for a0, a1 in L.runs(above):
            ax.axvspan(a0, a1, color="#ffd27f", alpha=0.4, lw=0)
        ax.plot(x, pw, color="k", lw=0.8, ls=":")
        v = m["L1 stored rule, exact"]
        if not pd.isna(v):
            ax.axvline(v, color=MCOL["L1 stored rule, exact"], lw=2)
        ax.set_title(f"{cols[0]}\nyellow = band edge above FA; LT = start of first 5-run ({'R+' if rgi else 'R-: end of 1st run'}) "
                     f"= {v:.0f}".replace("nan", "none"), fontsize=8.5, loc="left")
        # B sustained probability
        ax = axes[r, 1]
        ax.plot(x, e["p_above"], color=col, lw=1.5, label="P(whisker > FA)")
        ax.axhline(0.9, color="#888888", ls="--", lw=0.8)
        ax.axhline(0.5, color="#bbbbbb", ls=":", lw=0.8)
        v = m["L3 sustained prob."]
        if not pd.isna(v):
            ax.axvline(v, color=MCOL["L3 sustained prob."], lw=2)
        ax.set_title(f"{cols[1]}\nR+: >= 0.9 on 16/20 trials; R-: <= 0.5 on 16/20 after being >= 0.9; LT = {v:.0f}"
                     .replace("nan", "none"), fontsize=8.5, loc="left")
        # C fixed margin
        ax = axes[r, 2]
        ax.plot(x, e["p_mean"], color=col, lw=1, alpha=0.5)
        ax.plot(x, e["fa_time"], color="#555555", lw=1, ls="--", alpha=0.6)
        ax.plot(x, D, color="#8c564b", lw=2, label="D = whisker - FA (smoothed)")
        ax.axhline(0.2, color=MCOL["L8 fixed margin"], ls="--")
        if not rgi:
            ax.axhline(0.1, color=MCOL["L8 fixed margin"], ls=":")
        v = m["L8 fixed margin"]
        if not pd.isna(v):
            ax.axvline(v, color=MCOL["L8 fixed margin"], lw=2)
        ax.set_title(f"{cols[2]}\nR+: first rise above 0.2 held 16/20; R-: first fall below 0.1 held 16/20; LT = {v:.0f}"
                     .replace("nan", "none"), fontsize=8.5, loc="left")
        # D half-way
        ax = axes[r, 3]
        ax.plot(x, D, color="#8c564b", lw=2)
        if rgi:
            lo_, hi_ = D[:10].mean(), np.percentile(D, 90)
            lab = "start (first 10 trials)", "plateau (90th pct)"
        else:
            i_max = int(np.argmax(D[:30]))
            hi_, lo_ = D[i_max], np.percentile(D[i_max:], 10)
            lab = "floor (10th pct)", "early peak"
        ax.axhline(lo_, color="#888888", ls=":")
        ax.axhline(hi_, color="#888888", ls=":")
        ax.text(len(x) * 0.99, lo_, lab[0], ha="right", va="bottom", fontsize=7.5)
        ax.text(len(x) * 0.99, hi_, lab[1], ha="right", va="bottom", fontsize=7.5)
        ax.axhline(lo_ + 0.5 * (hi_ - lo_), color=MCOL["L7 half-way"], ls="--")
        v = m["L7 half-way"]
        if not pd.isna(v):
            ax.axvline(v, color=MCOL["L7 half-way"], lw=2)
        ax.set_title(f"{cols[3]}\nhalf-way between the session's own start and plateau, held 16/20; LT = {v:.0f}"
                     .replace("nan", "none"), fontsize=8.5, loc="left")
        # E joint change point
        ax = axes[r, 4]
        rj = m["_joint"]
        cw = np.cumsum(d["w_outcomes"])
        nx = np.clip(np.searchsorted(d["w_start"], d["n_start"]), 0, len(x) - 1)
        ax.plot(x, cw, color=col, lw=1.8, label="cumulative whisker licks")
        ax.plot(nx, np.cumsum(d["n_outcomes"]), color="#555555", lw=1.4, ls="--", label="cumulative FA licks")
        ax2 = ax.twinx()
        if rj is not None:
            ax2.fill_between(x, 0, rj["pk"], color=MCOL["L6 joint CP"], alpha=0.35, step="mid")
            ax.axvline(rj["median"], color=MCOL["L6 joint CP"], lw=2)
            ax.axvspan(rj["ci05"], rj["ci95"], color=MCOL["L6 joint CP"], alpha=0.08)
            ttl = f"LT = {rj['median']:.0f} [{rj['ci05']:.0f}, {rj['ci95']:.0f}], log10 BF {rj['log10_bf']:+.1f}"
        else:
            ttl = "no change allowed"
        ax2.set_yticks([])
        ax.set_title(f"{cols[4]}\nslope bend of whisker vs FA cumulative licks; shaded = posterior; {ttl}", fontsize=8.5,
                     loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        for a in axes[r, :4]:
            a.plot(x, pw - (pf if a in (axes[r, 2], axes[r, 3]) else 0), color="k", lw=0.8, ls=":") if a is not axes[r, 0] and a is not axes[r, 1] else None
            a.set_ylim(-0.25, 1.08)
        axes[r, 0].set_ylabel(f"SYNTHETIC {rg}\n" + ("early modest bump, jump at 60,\nend decline" if rgi else
                                                   "generalising 0-40, gradual\ndecline 40-120"), fontsize=9)
        for a in axes[r]:
            a.set_xlim(-1, len(x))
            a.tick_params(labelsize=7)
    for a in axes[-1]:
        a.set_xlabel("whisker trial")
    fig.suptitle("Every rule on the same two synthetic sessions (dotted black = TRUE whisker probability (A) or true "
                 "whisker - FA (C, D)). R+: true learning at trial 60 (after an early modest bump). R-: generalising until "
                 "40, gradual decline 40-120 (half-way ~ 80)", fontsize=11)
    fig.savefig(HERE / "013_rules_schematic.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ examples + table
def examples_and_table():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    cpj = pickle.load(open(ART / "005_cp_posteriors.pkl", "rb"))
    cpw = pickle.load(open(ART / "004_cp_posteriors.pkl", "rb"))
    tab = pd.read_csv(ART / "007_learning_trials_v2.csv").set_index("session_id")
    rows = []
    cache = {}
    for sid, d in inputs.items():
        rg = 1 if d["reward_group"] == "R+" else 0
        m = all_methods(d, curves[sid], cp_joint=cpj.get(sid), cp_w=cpw.get(sid))
        cache[sid] = m
        row = dict(session_id=sid, reward_group=d["reward_group"], learning_category=d["learning_category"],
                   **{"L0 stored": tab.loc[sid, "L0_stored"]},
                   **{k: v for k, v in m.items() if not k.startswith("_")},
                   **{"L6 joint CP": tab.loc[sid, "L6"], "L6 lenient": tab.loc[sid, "L6_len"],
                      "L5w lenient (R+)": tab.loc[sid, "L5w_len"], "lenient cascade": tab.loc[sid, "lt_lenient"],
                      "lenient cascade + clean gate": tab.loc[sid, "lt_lenient_clean"], "L6 category": tab.loc[sid, "L6_category"]})
        for meth in [k for k in row if k.startswith(("L", "lenient")) and k != "L6 category" and k != "learning_category"]:
            ev = m003.evaluate(row[meth], d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"], rg)
            p_clean = m007.clean_sep(m["_S"], row[meth], rg)[0]
            row[f"{meth} | contrast20"] = ev["contrast20"]
            row[f"{meth} | held_post"] = ev["held_post"]
            row[f"{meth} | clean"] = (p_clean > m007.SEP_P) if not pd.isna(p_clean) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "013_learning_trials_all_methods.csv", index=False)
    methods = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
               "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "L7 half-way", "L8 fixed margin", "lenient cascade",
               "lenient cascade + clean gate"]
    summ = []
    for rg, g in df.groupby("reward_group"):
        for meth in methods:
            v = g[meth]
            summ.append(dict(reward_group=rg, method=meth, n_defined=int(v.notna().sum()), median_lt=v.median(),
                             n_at_10=int((v == 10).sum()), n_clean=int((g.loc[v.notna(), f"{meth} | clean"] == True).sum()),  # noqa: E712
                             contrast20=g[f"{meth} | contrast20"].mean(), held_post=g[f"{meth} | held_post"].mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(ART / "013_methods_summary.csv", index=False)
    print(summ.round(3).to_string())

    ex = ["AB119", "MH028", "AB082", "AB107", "AB085", "AB139", "MH070", "AB093"]
    sel = [next(k for k in inputs if k.startswith(p)) for p in ex]
    ncol = 2
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(24, 4.6 * nrow), constrained_layout=True, squeeze=False)
    show = ["L0 stored", "L1 stored rule, exact", "L3 sustained prob.", "L5 whisker CP", "L6 joint CP", "L7 half-way", "L8 fixed margin"]
    for ax, sid in zip(axes.flat, sel):
        d, e, m = inputs[sid], curves[sid]["eb"], cache[sid]
        row = df[df.session_id == sid].iloc[0]
        col = COHORT_COLOR[d["reward_group"]]
        x = np.arange(len(d["w_outcomes"]))
        ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.18, lw=0)
        ax.plot(x, e["p_mean"], color=col, lw=1.5, label="whisker (smoothed)")
        ax.plot(x, d["stored"]["p_mean"], color=col, lw=0.5, alpha=0.5, label="whisker (stored)")
        ax.plot(x, e["fa_time"], color="#555555", lw=1.1, ls="--", label="FA at real times")
        ax.plot(x, m["_D"], color="#8c564b", lw=1.3, ls="-.", label="D = whisker - FA")
        o = d["w_outcomes"]
        ax.scatter(x, np.where(o == 1, 1.07, -0.3), s=2, color="k", marker="|")
        rj = cpj.get(sid)
        if rj is not None and not pd.isna(row["L6 joint CP"]):
            ax.axvspan(rj["ci05"], rj["ci95"], color=MCOL["L6 joint CP"], alpha=0.08)
        txt = []
        for i, meth in enumerate(show):
            v = row[meth]
            txt.append(f"{meth}: {v:.0f}".replace("nan", "none"))
            if not pd.isna(v):
                ax.axvline(v + (i - 3) * 0.25, color=MCOL[meth], lw=1.6, ls="-" if i % 2 == 0 else "--")
        ax.text(1.005, 0.5, "\n".join(txt) + f"\nlenient+gate: {row['lenient cascade + clean gate']:.0f}".replace("nan", "none")
                + f"\nL6 category: {row['L6 category']}", transform=ax.transAxes, fontsize=7.5, va="center")
        ax.set_ylim(-0.35, 1.12)
        ax.set_xlim(-1, len(x))
        ax.set_title(f"{sid}  ({d['reward_group']}, {d['learning_category']})", fontsize=9.5, loc="left")
        ax.legend(fontsize=6.8, frameon=False, loc="upper right", ncol=2)
    handles = [plt.Line2D([], [], color=MCOL[mm], lw=2, ls="-" if i % 2 == 0 else "--", label=mm) for i, mm in enumerate(show)]
    fig.legend(handles=handles, loc="upper center", ncol=len(show), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.savefig(HERE / "013_rules_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    pipeline_schematic()
    rules_schematic()
    examples_and_table()
    print("done")


if __name__ == "__main__":
    main()
