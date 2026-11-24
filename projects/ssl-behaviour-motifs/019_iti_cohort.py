"""019 -- ITI: R+ vs R- per trial outcome (haas, full run).

Inputs: cache/full/018/<sid>.pkl (peri-stimulus 25 ms bins per kept trial; lightning-action classes, kpms side meta-states,
features) and tables/full_trial_measures.parquet (015; no-lick versions of the ITI action fractions).
Per session and condition (whisker / auditory / no-stim x lick or not; >= 3 trials): mean of each measure over
- pre-stim = -2-0 s (baseline control); post early / late = 0-2 / 2-5 s after the response window ends; ITI = response end
  + 5 s -> start of the next trial's quiet window (user 2026-11-20), per trial (bins of the 018 time courses);
- post / ITI no-lick (015 `_nl` measures: frames > 0.5 s from any lick).
Tests: R+ vs R- per condition x measure x window, Mann-Whitney + Welch (unit = session = mouse), uncorrected, exploratory.
Within-session reward contrast: WH - AH per session (R+: both rewarded; R-: only AH rewarded), R+ vs R-.
Output: tables/019_iti_cohort.csv, figures/019_iti_cohort.png|pdf|svg, provenance/019.json.
"""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
C18 = L.CACHE / "full" / "018"
PERI, BIN = (-2.0, 16.0), 0.025  # as 018
NB = int(round((PERI[1] - PERI[0]) / BIN))
CH = {"la:1": "P(still)", "la:2": "P(whisking)", "la:3": "P(jaw)", "la:4": "P(lick)", "meta_side:0": "P(M0 still)",
      "meta_side:2": "P(M2 lick episode)", "whisking_amp": "whisking amp. (deg)", "whisker_angle_dev": "whisker angle (deg)",
      "jaw_open": "jaw opening (mm)", "nose_speed": "nose speed (mm/s)"}
NL = {"act_still_nl": "P(still), no-lick", "act_whisking_nl": "P(whisking), no-lick", "act_jaw_nl": "P(jaw), no-lick"}
CONDS = ["WH", "WM", "AH", "AM", "FA", "CR"]


def cond_of(stim, lick):
    return {("whisker", 1): "WH", ("whisker", 0): "WM", ("auditory", 1): "AH", ("auditory", 0): "AM",
            ("no_stim", 1): "FA", ("no_stim", 0): "CR"}[(stim, int(lick))]


def two(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def main():
    S = pd.read_csv(T / "full_sessions.csv")
    coh = dict(zip(S.session_id, S.reward_group))
    tb = PERI[0] + BIN * (np.arange(NB) + 0.5)
    rows = []
    for f in sorted(C18.glob("*.pkl")):
        d = pickle.load(open(f, "rb"))
        sid = d["session_id"]
        if sid not in coh:
            continue
        tr = d["trials"]
        P = d["peri"].astype(np.float32)
        cond = np.array([cond_of(s, lk) for s, lk in zip(tr.stim, tr.lick_flag)])
        rs, ps = tr.resp_stop_rel.to_numpy(), tr.post_stop_rel.to_numpy()
        ia, ib = tr.iti_start_rel.to_numpy(), tr.iti_stop_rel.to_numpy()
        # per-trial bin masks (user 2026-11-20: ITI = response end + 5 s -> next quiet window; the 5 s after the
        # response window are the post-response epoch)
        wins = {"pre-stim (-2-0 s)": np.broadcast_to((tb >= -2) & (tb < 0), (len(tr), NB)),
                "post early (0-2 s)": (tb[None] >= rs[:, None]) & (tb[None] < np.fmin(rs + 2, ps)[:, None]),
                "post late (2-5 s)": (tb[None] >= (rs + 2)[:, None]) & (tb[None] < ps[:, None]),
                "ITI (+5 s -> next quiet window)": (tb[None] >= ia[:, None]) & (tb[None] < ib[:, None])}
        for c in CONDS:
            m = cond == c
            if m.sum() < 3:
                continue
            for ch, lab in CH.items():
                if ch not in d["peri_channels"]:
                    continue
                k = d["peri_channels"].index(ch)
                for w, wm in wins.items():
                    X = np.where(wm[m], P[m][:, k], np.nan)
                    with np.errstate(all="ignore"):
                        v = np.nanmean(X, 1)  # per trial
                    if np.isfinite(v).sum() < 3:
                        continue
                    rows.append(dict(session_id=sid, cohort=coh[sid], condition=c, measure=lab, window=w,
                                     value=float(np.nanmean(v)), n_trials=int(np.isfinite(v).sum())))
    D = pd.read_parquet(T / "full_trial_measures.parquet")
    D = D[D.epoch.isin(["post", "iti"])]
    D["condition"] = [cond_of(s, lk) for s, lk in zip(D.stim, D.lick_flag)]
    for (sid, c, ep), g in D.groupby(["session_id", "condition", "epoch"]):
        if sid not in coh:
            continue
        for m, lab in NL.items():
            if g[m].notna().sum() < 3:
                continue
            rows.append(dict(session_id=sid, cohort=coh[sid], condition=c, measure=lab,
                             window="post no-lick (0-5 s)" if ep == "post" else "ITI no-lick",
                             value=float(g[m].mean()), n_trials=int(g[m].notna().sum())))
    V = pd.DataFrame(rows)
    V.to_csv(T / "019_iti_per_session.csv", index=False)
    # tests
    out = []
    for (c, m, w), g in V.groupby(["condition", "measure", "window"]):
        a, b = g[g.cohort == "R+"].value, g[g.cohort == "R-"].value
        pm, pw = two(a, b)
        sd = np.sqrt((a.var() + b.var()) / 2)
        out.append(dict(condition=c, measure=m, window=w, n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=a.mean(),
                        mean_Rminus=b.mean(), diff=a.mean() - b.mean(), cohen_d=(a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                        p_mannwhitney=pm, p_welch=pw))
    # within-session reward contrast WH - AH
    for (m, w), g in V.groupby(["measure", "window"]):
        pv = g.pivot_table(index=["session_id", "cohort"], columns="condition", values="value").reset_index()
        if "WH" not in pv or "AH" not in pv:
            continue
        pv["x"] = pv.WH - pv.AH
        a, b = pv[pv.cohort == "R+"].x.dropna(), pv[pv.cohort == "R-"].x.dropna()
        pm, pw = two(a, b)
        sd = np.sqrt((a.var() + b.var()) / 2)
        out.append(dict(condition="WH-AH", measure=m, window=w, n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=a.mean(),
                        mean_Rminus=b.mean(), diff=a.mean() - b.mean(), cohen_d=(a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                        p_mannwhitney=pm, p_welch=pw))
    O = pd.DataFrame(out)
    O["both_p_lt_005"] = (O.p_mannwhitney < 0.05) & (O.p_welch < 0.05)
    O.to_csv(T / "019_iti_cohort.csv", index=False)
    # figure: Cohen d heatmaps (condition x measure) per window, * = both tests < 0.05
    wins = ["pre-stim (-2-0 s)", "post early (0-2 s)", "post late (2-5 s)", "ITI (+5 s -> next quiet window)", "ITI no-lick"]
    conds = CONDS + ["WH-AH"]
    fig, axs = plt.subplots(1, 5, figsize=(11.5, 3.0), gridspec_kw=dict(width_ratios=[len(CH)] * 4 + [len(NL) + 1.5]))
    for ax, w in zip(axs, wins):
        g = O[O.window == w]
        ms = [m for m in list(CH.values()) + list(NL.values()) if m in set(g.measure)]
        Z = np.full((len(conds), len(ms)), np.nan)
        for i, c in enumerate(conds):
            for j, m in enumerate(ms):
                q = g[(g.condition == c) & (g.measure == m)]
                if len(q):
                    Z[i, j] = q.cohen_d.iloc[0]
                    if q.both_p_lt_005.iloc[0]:
                        ax.text(j, i, "*", ha="center", va="center", fontsize=8, color="k")
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(ms, rotation=55, ha="right", fontsize=5)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds if ax is axs[0] else [])
        ax.set_title(w, fontsize=6)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
    cb.set_label("Cohen's d, R+ − R− (red = higher in R+)")
    fig.suptitle("ITI, R+ vs R− per trial outcome (session means; * = Mann–Whitney and Welch p < 0.05, uncorrected). "
                 "WH−AH = within-session whisker-hit minus auditory-hit ITI", fontsize=6, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.35, top=0.86, wspace=0.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"019_iti_cohort.{ext}", dpi=300)
    plt.close(fig)
    json.dump(dict(script="019_iti_cohort.py", n_sessions=int(V.session_id.nunique()), min_trials=3,
                   windows=wins, versions=L.versions()), open(L.RES / "provenance" / "019.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(O[O.both_p_lt_005].sort_values(["window", "condition"]).round(3).to_string())


if __name__ == "__main__":
    main()
