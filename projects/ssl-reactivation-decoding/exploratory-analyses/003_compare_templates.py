"""003 -- Clean vs spont template, one session and target (user 2026-10-23: fit both and compare).

clean: quiet target stimuli vs correct rejections + passive shams; spont: the same plus random spontaneous continuous windows as
negatives. Both read out with every unit of each key area group, one threshold per area group, the same
for both templates and every session (user 2026-10-24).
Figure (fig5_template_comparison): per area group, (a) held-out ROC on the event classes; (b) at the threshold: fraction of
held-out target stimuli detected, held-out no-stimulus events and other-modality stimuli crossing;
(c) continuous detections / min, all valid time and quiet time (> 1 s from licks, low whisking), data vs time-shift null;
(d) event rate vs threshold, data and time-shift null, both models. Table: template_comparison.csv.
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
MODELS = {"clean": "-", "spont": "--"}


def main(sid, target):
    base = m1.HOME / "pilot" / sid / target
    V = pd.concat([pd.read_csv(base / m / "validation.csv") for m in MODELS], ignore_index=True)
    V["excess_all"] = V.rate_real / V.rate_null_shift.clip(lower=1e-3)
    V["excess_quiet"] = V.rate_real_quiet / V.rate_null_shift_quiet.clip(lower=1e-3)
    V.to_csv(base / "template_comparison.csv", index=False)
    N = {m: pd.read_parquet(base / m / "null_curves.parquet") for m in MODELS}
    G = [g for g in m1.GROUPS if g in set(V.group)]
    plt = m2.setup()
    col = m2.colors()
    from sklearn.metrics import roc_curve
    fig, axs = plt.subplots(len(G), 4, figsize=(7.4, 1.6 * len(G)), squeeze=False, gridspec_kw=dict(width_ratios=[1, 1.4, 1.4, 1.3]))
    for i, g in enumerate(G):
        c = col.get(g, "k")
        ax = axs[i, 0]
        for m, ls in MODELS.items():
            d = np.load(base / m / f"continuous_{g.replace(' ', '_')}.npz")
            fpr, tpr, _ = roc_curve(d["y"], d["oof"])
            ax.plot(fpr, tpr, color=c, ls=ls, label=m)
        ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR (held-out events)"); ax.set_ylabel("hit rate")
        ax.set_title(m2.short(g), color=c, loc="left")
        if i == 0:
            ax.legend(frameon=False)
        ax = axs[i, 1]
        q = V[V.group == g].set_index("model")
        lab = ["target stim.", "no-stim. events", "other modality"]
        x = np.arange(3)
        for k, (m, ls) in enumerate(MODELS.items()):
            if m in q.index:
                r = q.loc[m]
                ax.bar(x + (k - 0.5) * 0.38, [r.sens, r.fp_events, r.fp_other_modality], 0.38, color=c, alpha=1 if m == "clean" else 0.45,
                       hatch=None if m == "clean" else "///", edgecolor="w", lw=0, label=m)
        ax.set_xticks(x, lab); ax.set_ylim(0, 1); ax.set_ylabel("fraction crossing")
        ax.set_title(f"at the group threshold ({q.thr.iloc[0]:.1f})", loc="left", fontsize=5.4)
        ax = axs[i, 2]
        lab = ["all", "quiet"]
        for k, (m, ls) in enumerate(MODELS.items()):
            if m in q.index:
                r = q.loc[m]
                xx = np.arange(2) * 2.4 + k * 1.0
                ax.bar(xx, [r.rate_real, r.rate_real_quiet], 0.45, color=c, alpha=1 if m == "clean" else 0.45, lw=0)
                ax.bar(xx + 0.45, [r.rate_null_shift, r.rate_null_shift_quiet], 0.45, color="0.6" if m == "clean" else "0.8", lw=0)
                for xi, a, b in zip(xx, (r.rate_real, r.rate_real_quiet), (r.rate_null_shift, r.rate_null_shift_quiet)):
                    ax.text(xi + 0.2, max(a, b) * 1.03, f"{m[0]} ×{a / max(b, 1e-3):.1f}", ha="center", va="bottom", fontsize=4.3)
        ax.set_xticks([0.95, 3.35], lab); ax.set_ylabel("detections / min")
        ax.set_title("data (colour) vs time-shift null (grey); c = clean, s = spont", loc="left", fontsize=5.4)
        ax = axs[i, 3]
        for m, ls in MODELS.items():
            n = N[m][N[m].group == g]
            ax.plot(n.thr, n.real, color=c, ls=ls, label=f"{m} data")
            ax.plot(n.thr, n.null_shift, color="0.4", ls=ls, lw=0.6, label=f"{m} time shift")
            if m in q.index:
                ax.axvline(q.loc[m].thr, color=c, ls=ls, lw=0.5)
        ax.set_yscale("log"); ax.set_ylim(0.01, None); ax.set_xlim(1, 15)
        ax.set_xlabel("threshold (normalised readout)"); ax.set_ylabel("events / min")
        if i == 0:
            ax.legend(frameon=False, fontsize=4.3)
    fig.suptitle(f"{sid}: {target} templates, clean (stimuli vs no-stimulus events) vs spont (+ spontaneous continuous windows as "
                 f"negatives); every unit of each group; one threshold per group ({V.thr_rule.iloc[0]})",
                 x=0.01, ha="left", fontsize=6.0, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    m2.save(fig, base / "figures", "fig5_template_comparison")
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(V[["group", "model", "n_units", "pauc", "thr", "sens", "fp_events", "fp_other_modality", "rate_real", "rate_null_shift",
             "excess_all", "rate_real_quiet", "rate_null_shift_quiet", "excess_quiet"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--target", default="whisker")
    a = ap.parse_args()
    main(a.session, a.target)
