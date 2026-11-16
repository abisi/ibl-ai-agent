"""011 -- Decoders with vs without z-scoring the units (user 2026-10-30), whole brain, example day-0 sessions.

Variants (decoder fit; everything else as 009):
  zscore  units z-scored on the training windows (001 fit: mean / s.d., s.d. floor 0.2 spikes per window), L2 logistic
          regression; readout = counts . (w / sd) + c
  raw     centred native spike counts, no scaling; readout = counts . w + c
Shuffles (009 SHUF_STRATA): plain = decoder weights (in the fit space) permuted across all units; rate = within
firing-rate deciles. Each variant x shuffle reruns 009 run() for the whole-brain group into
<HOME>/scratch/standardization_test/<variant>_<shuffle>/stable/<session>/.
Sessions (--sessions): default = the 8 day-0 sessions of the shuffle-space test (whole-brain whisker pAUC >= 0.8, 4 R+ / 4 R-
at spread quiet-window rates).
--summary: table standardization_test.csv and fig22_standardization_test: per target, zscore vs raw per session (lines):
held-out pAUC, threshold-crossing rate (all valid time, quiet window), excess vs plain and rate-matched shuffles,
weight-rate relation (Spearman |native weight| vs mean count), readout kurtosis, and the overlap of the two variants' events.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m9 = importlib.import_module("009_full_dataset_events")
BASE = m1.HOME / "scratch" / "standardization_test"
SESSIONS = ["MH038_20250524_105329", "AB117_20240723_125437", "AB130_20240902_123634", "MH030_20250501_151231",
            "AB126_20240822_114405", "AB116_20240724_102941", "AB140_20241129_104154", "AB139_20241119_115957"]
VARIANTS = ("zscore", "raw")
SHUFFLES = ("plain", "rate")
COINC = 0.05                                     # s: events of the two variants coincide when their peaks are this close
COH = {"R+": "#00B400", "R-": "#C800C8"}
fit_z = m1.fit


def fit_raw(X, y, w, C, scale=None):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), np.ones(X.shape[1])
    m = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000).fit(X - mu, y, sample_weight=w)
    return dict(w=m.coef_[0], b=float(m.intercept_[0]), mu=mu, sd=sd, C=C)


def run(sid):
    for var, f in zip(VARIANTS, (fit_z, fit_raw)):
        for shuf in SHUFFLES:
            m1.fit, m9.SHUF_STRATA, m9.ROOT = f, 10 if shuf == "rate" else 1, BASE / f"{var}_{shuf}"
            m9.run(sid, [m9.m8.ALL], "stable", force=True)
    m1.fit = fit_z


def summary(sessions):
    from scipy.stats import kurtosis, spearmanr
    plt = m2.setup()
    coh = m9.sessions_table().set_index("session_id").cohort
    rows = []
    for sid in sessions:
        for tgt in m9.m6.TARGETS:
            ev = {}
            for var in VARIANTS:
                r = dict(session_id=sid, cohort=coh.get(sid), target=tgt, variant=var)
                for shuf in SHUFFLES:
                    m9.ROOT = BASE / f"{var}_{shuf}"
                    D = m9.decoders_table(sid, "stable")
                    d = D[(D.group == m9.m8.ALL) & (D.target == tgt) & (D.sens == m9.SENS_MAIN)].iloc[0]
                    r.update({f"excess_{shuf}": d.get("excess"), f"p_{shuf}": d.get("p_shuf"),
                              f"excess_prestart_{shuf}": d.get("excess_prestart")})
                    if shuf == "plain":
                        r.update(pauc=d.pauc, thr=d.thr_norm, rate=d.get("rate"), rate_prestart=d.get("rate_prestart"))
                out = BASE / f"{var}_plain" / "stable" / sid
                G = np.load(out / f"g_All_{tgt}.npz")
                vv = np.load(out / "session.npz")["valid"]
                r["rho_wnative_rate"] = spearmanr(np.abs(G["w"] / G["sd"]), G["mu"])[0]
                r["kurtosis"] = float(kurtosis(G["z"][vv]))
                ev[var] = pd.read_parquet(out / f"events_{tgt}_All.parquet").t_peak.to_numpy()
                rows.append(r)
            a, b = np.sort(ev["zscore"]), np.sort(ev["raw"])
            near = lambda x, y: np.mean(np.min(np.abs(x[:, None] - y[None, :]), 1) <= COINC) if len(x) and len(y) else np.nan
            rows[-2]["overlap"] = near(a, b)                     # fraction of z-score events also found by raw
            rows[-1]["overlap"] = near(b, a)
    R = pd.DataFrame(rows)
    R.to_csv(m1.HOME / "scratch" / "standardization_test.csv", index=False)
    cols = [("pauc", "held-out pAUC"), ("rate", "events / min"), ("rate_prestart", "events / min,\nquiet window"),
            ("excess_plain", "excess vs plain\nshuffle (/min)"), ("excess_rate", "excess vs rate-matched\nshuffle (/min)"),
            ("rho_wnative_rate", "Spearman |native weight|\nvs mean count"), ("kurtosis", "readout kurtosis"),
            ("overlap", "events shared with\nthe other variant")]
    fig, axs = plt.subplots(len(m9.m6.TARGETS), len(cols), figsize=(7.4, 3.6), squeeze=False)
    for i, tgt in enumerate(m9.m6.TARGETS):
        Q = R[R.target == tgt]
        for j, (c, lab) in enumerate(cols):
            ax = axs[i, j]
            for sid, q in Q.groupby("session_id"):
                q = q.set_index("variant").reindex(VARIANTS)
                ax.plot([0, 1], q[c], color=COH.get(q.cohort.iloc[0], "0.5"), lw=0.6, marker="o", ms=2, alpha=0.8)
            if c.startswith("excess"):
                ax.axhline(0, color="0.6", lw=0.4, ls=":")
                ax.set_yscale("symlog", linthresh=0.1)
            elif c.startswith("rate"):
                ax.set_yscale("symlog", linthresh=0.1)
            ax.set_xticks([0, 1], list(VARIANTS) if i == len(m9.m6.TARGETS) - 1 else [], fontsize=4.6)
            ax.set_xlim(-0.4, 1.4)
            ax.tick_params(labelsize=4.4)
            if i == 0:
                ax.set_title(lab, fontsize=4.8)
        axs[i, 0].set_ylabel(f"{tgt}, whole brain", fontsize=5.2)
    fig.suptitle(f"decoders with (zscore) vs without (raw) z-scoring the units; {len(sessions)} day-0 sessions (green R+, magenta "
                 f"R-), stable units; threshold = {m9.SENS_MAIN:.0%} held-out sensitivity per variant; quiet window = task no-lick "
                 f"window; events shared = peaks within {COINC * 1000:.0f} ms", x=0.01, ha="left", fontsize=5.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    m2.save(fig, m1.HOME / "figures" / "full_stable", "fig22_standardization_test")
    plt.close(fig)
    pd.set_option("display.width", 260)
    print(R.round(3).to_string(index=False))
    print(R.groupby(["target", "variant"])[[c for c, _ in cols] + ["thr", "p_plain", "p_rate"]].median().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.run:
        for s in a.sessions:
            run(s)
    if a.summary:
        summary(a.sessions)
