"""019 -- All learning-trial methods recomputed on sigma = 1.0 curves (user decision 2026-09-25:
"keep sigma at 1, how do methods compare then?"). Curves: artifacts/018_curves_sigma1.pkl
("sigma1": exact posterior of the original model at fixed sigma = 1, FA at real trial times).

Curve-based methods (recomputed on sigma = 1 curves):
  L1  stored run rule (5 consecutive trials with 80% lower edge > FA; clamp 10; R- fallback)
  L3  sustained probability (P(whisker > FA) >= 0.9 on 16/20; R- <= 0.5 after being >= 0.9)
  L7  half-way on D = whisker - FA, PEAK-based levels (start = mean first 10, plateau = 90th pct)
  L8  fixed margin on D (0.2 / 0.1)
  L7avg  half-way on D, AVERAGE-based (robust to spikes): start = mean of D over the first 10 trials,
         plateau = max of the 20-trial running mean of D; LT = first t with D[t] >= half-way AND
         mean(D[t:t+20]) >= half-way; change >= 0.2.
  RmW    R- half-way decline of the WHISKER curve, peak-based (017: early = max first 20 trials)
  RmWavg R- half-way decline of the whisker curve, average-based: early = mean of first 20 trials,
         floor = min of the 20-trial running mean after trial 20; LT = first t with w[t] <= half-way
         AND mean(w[t:t+20]) <= half-way; drop >= 0.15; >= 3 whisker hits in first 20 trials.
Raw-lick methods (sigma-independent, from 004/005/007): L5 whisker CP, L6 joint CP, L6 lenient.
Combined "final" definitions (as frozen in 017):
  FINAL_peak  R+: L7 else L6 lenient; R-: RmW; light direction gate          (= 017 sigma1)
  FINAL_avg   R+: L7avg else L6 lenient; R-: RmWavg; light direction gate
Metrics per method: n with LT, median LT, n at 10, separation = whisker-FA lick rate 20 trials
after minus before (sign-corrected, larger = better), n with separation > 0.1, n passing the
clean gate (007 settings), n with separation <= 0 (wrong direction).
Outputs: artifacts/019_methods_sigma1.csv (per session), 019_methods_sigma1_summary.csv;
exploratory-analyses/019_methods_sigma1_examples.png, 019_methods_sigma1_Rplus.png / _Rminus.png
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

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m003 = _load("m003", "003_learning_trial_rules.py")
m006 = _load("m006", "006_halfway_rule_and_final_table.py")
m007 = _load("m007", "007_lenient_learning_trials.py")
m013 = _load("m013", "013_methods_schematic_and_examples.py")
m017 = _load("m017", "017_final_learning_trials.py")
m005 = m007.j005

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
W = 20
METHODS = ["L0 stored", "L1 stored rule", "L3 sustained", "L5 whisker CP", "L6 joint CP", "L6 lenient", "L7 half-way (peak)",
           "L7 half-way (avg)", "L8 fixed margin", "R- whisker half-way (peak)", "R- whisker half-way (avg)", "FINAL peak",
           "FINAL avg"]
MCOL = {"L0 stored": "#d62728", "L1 stored rule": "#ff9896", "L3 sustained": "#bcbd22", "L5 whisker CP": "#7f7f7f",
        "L6 joint CP": "#1f77b4", "L6 lenient": "#9ecae1", "L7 half-way (peak)": "#ff7f0e", "L7 half-way (avg)": "#8c564b",
        "L8 fixed margin": "#17becf", "R- whisker half-way (peak)": "#e377c2", "R- whisker half-way (avg)": "#9467bd",
        "FINAL peak": "#ffbb78", "FINAL avg": "#000000"}


def runmean(x, w=W):
    c = np.r_[0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    n = len(x) - w + 1
    if n > 0:
        out[:n] = (c[w:] - c[:-w]) / w
    return out


def halfway_avg_D(D):
    rm = runmean(D)
    d0 = D[:10].mean()
    dhi = np.nanmax(rm[10:]) if np.isfinite(rm[10:]).any() else np.nan
    if not np.isfinite(dhi) or dhi - d0 < 0.2:
        return np.nan
    thr = d0 + 0.5 * (dhi - d0)
    for t in range(len(D) - W + 1):
        if D[t] >= thr and rm[t] >= thr:
            return float(t)
    return np.nan


def halfway_avg_whisker(w, o):
    if o[:20].sum() < 3:
        return np.nan
    rm = runmean(w)
    hi = w[:20].mean()
    lo = np.nanmin(rm[20:]) if np.isfinite(rm[20:]).any() else np.nan
    if not np.isfinite(lo) or hi - lo < 0.15:
        return np.nan
    thr = lo + 0.5 * (hi - lo)
    for t in range(len(w) - W + 1):
        if w[t] <= thr and rm[t] <= thr:
            return float(t)
    return np.nan


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "018_curves_sigma1.pkl", "rb"))
    old = pd.read_csv(ART / "013_learning_trials_all_methods.csv").set_index("session_id")
    rows, cache = [], {}
    for sid, d in inputs.items():
        rgs = d["reward_group"]
        rg = 1 if rgs == "R+" else 0
        o, wt, no, nt = d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"]
        c = curves[sid]["sigma1"]
        D = c["p_mean"] - c["fa_time"]
        S = m005.Streams(o, wt, no, nt)
        r = {"L0 stored": old.loc[sid, "L0 stored"],
             "L1 stored rule": m003.stored_rule(o, c["p_mean"], c["p_low80"], c["p_high80"], c["fa_time"], rg)[0],
             "L3 sustained": m003.sustain_rule(c["p_above"], o, rg),
             "L5 whisker CP": old.loc[sid, "L5 whisker CP"], "L6 joint CP": old.loc[sid, "L6 joint CP"],
             "L6 lenient": old.loc[sid, "L6 lenient"],
             "L7 half-way (peak)": m006.halfway(D, rg), "L7 half-way (avg)": halfway_avg_D(D) if rg == 1 else m006.halfway(D, rg),
             "L8 fixed margin": m013.fixed_margin(D, rg),
             "R- whisker half-way (peak)": (m017.rminus_halfway(c["p_mean"]) if o[:20].sum() >= 3 else np.nan) if rg == 0 else np.nan,
             "R- whisker half-way (avg)": halfway_avg_whisker(c["p_mean"], o) if rg == 0 else np.nan}
        for name, a, b in (("FINAL peak", "L7 half-way (peak)", "R- whisker half-way (peak)"),
                           ("FINAL avg", "L7 half-way (avg)", "R- whisker half-way (avg)")):
            lt = (r[a] if not pd.isna(r[a]) else r["L6 lenient"]) if rg == 1 else r[b]
            dw, dd = m017.gate_values(lt, o, wt, no, nt)
            ok = (not pd.isna(dd) and dd > 0) if rg == 1 else (not pd.isna(dw) and dw < 0)
            r[name] = lt if ok else np.nan
        row = dict(session_id=sid, reward_group=rgs, learning_category=d["learning_category"], early_hits20=int(o[:20].sum()), **r)
        for m in METHODS:
            ev = m003.evaluate(r[m], o, wt, no, nt, rg)
            row[f"{m} | sep"] = (1 if rg == 1 else -1) * ev["contrast20"] if not pd.isna(ev["contrast20"]) else np.nan
            p = m007.clean_sep(S, r[m], rg)[0]
            row[f"{m} | clean"] = (p > m007.SEP_P) if not pd.isna(p) else np.nan
        rows.append(row)
        cache[sid] = D
    df = pd.DataFrame(rows)
    df.to_csv(ART / "019_methods_sigma1.csv", index=False)
    summ = []
    for rgs, g in df.groupby("reward_group"):
        for m in METHODS:
            v = g[m].notna()
            if v.sum() == 0:
                continue
            sep = g.loc[v, f"{m} | sep"]
            summ.append(dict(reward_group=rgs, method=m, n=int(v.sum()), pct=round(100 * v.mean()), median_lt=g.loc[v, m].median(),
                             n_at_10=int((g[m] == 10).sum()), sep_mean=sep.mean(), n_sep_gt_0p1=int((sep > 0.1).sum()),
                             n_wrong_dir=int((sep <= 0).sum()), n_clean=int((g.loc[v, f"{m} | clean"] == True).sum())))  # noqa: E712
    s = pd.DataFrame(summ)
    s.to_csv(ART / "019_methods_sigma1_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string())

    show = ["L0 stored", "L1 stored rule", "L3 sustained", "L6 joint CP", "L7 half-way (peak)", "L7 half-way (avg)",
            "R- whisker half-way (peak)", "R- whisker half-way (avg)", "FINAL avg"]

    def panel(ax, sid, small=False):
        d, c = inputs[sid], curves[sid]["sigma1"]
        row = df[df.session_id == sid].iloc[0]
        o = d["w_outcomes"]
        x = np.arange(len(o))
        col = COHORT_COLOR[d["reward_group"]]
        ax.fill_between(x, c["p_low80"], c["p_high80"], color=col, alpha=0.18, lw=0)
        ax.plot(x, c["p_mean"], color=col, lw=1.1)
        ax.plot(x, c["fa_time"], color="#555555", lw=0.9, ls="--")
        if d["reward_group"] == "R+":
            ax.plot(x, runmean(cache[sid]), color="#8c564b", lw=1.2, ls="-.")
        else:
            ax.plot(x, runmean(c["p_mean"]), color="#9467bd", lw=1.2, ls="-.")
        ax.scatter(x, np.where(o == 1, 1.07, -0.1), s=1.5, color="k", marker="|")
        txt = []
        for i, m in enumerate([m for m in show if not (d["reward_group"] == "R+" and m.startswith("R-"))]):
            v = row[m]
            txt.append(f"{m}: {v:.0f}".replace("nan", "-"))
            if not pd.isna(v):
                ax.axvline(v, color=MCOL[m], lw=2.2 if m == "FINAL avg" else 1.2, ls="-" if i % 2 == 0 else "--")
        ax.set_ylim(-0.15, 1.12)
        ax.set_xlim(-1, len(o))
        ax.tick_params(labelsize=6)
        ttl = f"{sid[:5]} {d['reward_group']} {d['learning_category']}"
        if small:
            ax.set_title(ttl + f" | final avg {row['FINAL avg']:.0f}, stored {row['L0 stored']:.0f}".replace("nan", "-"), fontsize=6.5)
        else:
            ax.set_title(ttl, fontsize=9, loc="left")
            ax.text(1.005, 0.5, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="center")

    ex = ["AB119", "MH028", "AB082", "AB107", "AB085", "AB139", "MH068", "AB128", "AB153", "AB122"]
    sel = [next(k for k in inputs if k.startswith(p)) for p in ex]
    fig, axes = plt.subplots(5, 2, figsize=(24, 19), constrained_layout=True)
    for ax, sid in zip(axes.flat, sel):
        panel(ax, sid)
    handles = [plt.Line2D([], [], color=MCOL[m], lw=2, label=m) for m in show]
    handles.append(plt.Line2D([], [], color="#8c564b", ls="-.", label="20-trial running mean: D (R+) / whisker (R-)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.suptitle("Methods on sigma = 1.0 curves (whisker 80% band, FA dashed), 10 example sessions", fontsize=12, y=1.06)
    fig.savefig(HERE / "019_methods_sigma1_examples.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    for rgs in ("R+", "R-"):
        g = df[df.reward_group == rgs].sort_values(["FINAL avg", "session_id"], na_position="last")
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), constrained_layout=True)
        for ax, sid in zip(axes.flat, g.session_id):
            panel(ax, sid, small=True)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        fig.legend(handles=[h for h in handles if not (rgs == "R+" and h.get_label().startswith("R-"))], loc="upper center",
                   ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle(f"{rgs}: every session, methods on sigma = 1.0 curves; black thick = FINAL (average-based half-way); "
                     f"dash-dot = 20-trial running mean", fontsize=10, y=1.05)
        fig.savefig(HERE / f"019_methods_sigma1_{rgs.replace('+', 'plus').replace('-', 'minus')}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
