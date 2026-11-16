"""004 -- One threshold per area group, the same for every session and both templates (user 2026-10-24).

Pools every pilot session run by 001 (<HOME>/pilot/<session>/<target>/<clean|spont>). For each area group and each threshold of
the 001 grid (normalised readout units): the time-shift null event rate per session and template (all valid time), the held-out
sensitivity to quiet target stimuli, false detections on held-out no-stimulus events, other-modality stimuli crossing (reported,
not penalised), and the data event rate.
Rule: the group threshold is the smallest grid value at which the time-shift null rate is <= NULL_MAX events / min in every
eligible session and template of the group (a fixed false-positive budget for every group; the sensitivity then follows per
group). Eligible: held-out pAUC >= PAUC_MIN (a decoder without a template does not set the threshold; its heavy-tailed
readout would) and not in ARTEFACT. Every decoder is still reported at its group threshold.
Writes <HOME>/thresholds_<target>.json, group_threshold_curves_<target>.csv, figures/fig6_group_thresholds_<target>; then, with
--apply, re-derives the threshold-dependent outputs of every session (001 apply_thresholds).
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
NULL_MAX = 0.01                                  # events / min: ~ one null event per 100 min in every session
MODELS = ("clean", "spont")
PAUC_MIN = 0.8
# (session, group): template driven by stimulus artefact. MH070 ALM: periodic artefact spikes -5..+8 ms after whisker onsets,
# not after auditory, extending past the -10..+5 ms replacement into the training window (1-ms PSTH check 2026-10-24)
ARTEFACT = {("MH070_20260121_140848", "Motor areas")}


def curves(target):
    rows = []
    for d in sorted((m1.HOME / "pilot").glob(f"*/{target}/*/fit_summary.csv")):
        out = d.parent
        sid, model = out.parts[-3], out.parts[-1]
        if model not in MODELS:
            continue
        N = pd.read_parquet(out / "null_curves.parquet")
        PE = pd.read_parquet(out / "peri_event.parquet")
        F = pd.read_csv(d).set_index("group")
        other = "A" if target == "whisker" else "W"
        for g, n in N.groupby("group"):
            q = PE[PE.group == g]
            pk = q.peak.to_numpy()
            stim, neg, oth = (q.train & (q.y == 1)).to_numpy(), (q.train & (q.y == 0)).to_numpy(), q.cls.eq(other).to_numpy()
            th = n.thr.to_numpy()
            rows.append(pd.DataFrame(dict(
                session_id=sid, target=target, model=model, group=g, pauc=F.loc[g, "pauc"], thr=th,
                null_shift=n.null_shift.to_numpy(), null_shift_quiet=n.null_shift_quiet.to_numpy(), rate=n.real.to_numpy(),
                rate_quiet=n.real_quiet.to_numpy(),
                sens=(pk[stim][None, :] > th[:, None]).mean(1),
                fp_events=(pk[neg][None, :] > th[:, None]).mean(1),
                fp_other_modality=(pk[oth][None, :] > th[:, None]).mean(1) if oth.any() else np.nan)))
    K = pd.concat(rows, ignore_index=True)
    K["artefact"] = [(s_, g) in ARTEFACT for s_, g in zip(K.session_id, K.group)]
    K["eligible"] = (K.pauc >= PAUC_MIN) & ~K.artefact
    return K


def choose(K):
    thr = {}
    for g, q in K.groupby("group"):
        e = q[q.eligible]
        worst = (e if len(e) else q).groupby("thr").null_shift.max()
        ok = worst.index[worst.to_numpy() <= NULL_MAX]
        thr[g] = float(ok.min()) if len(ok) else float(worst.index.max())
    return thr


def figure(K, thr, target):
    plt = m2.setup()
    col = m2.colors()
    G = [g for g in m1.GROUPS if g in thr]
    cols = [("null_shift", "time-shift null (events / min)", True), ("rate", "data detections (events / min)", True),
            ("sens", "held-out target stimuli detected", False), ("fp_events", "held-out no-stimulus events crossing", False),
            ("fp_other_modality", "other-modality stimuli crossing", False)]
    fig, axs = plt.subplots(len(G), len(cols), figsize=(7.4, 1.35 * len(G)), squeeze=False)
    for i, g in enumerate(G):
        c = col.get(g, "k")
        for j, (k, lab, lg) in enumerate(cols):
            ax = axs[i, j]
            for (sid, m), q in K[K.group == g].groupby(["session_id", "model"]):
                el = bool(q.eligible.iloc[0])
                ax.plot(q.thr, q[k], color=c if el else "0.7", lw=0.5, alpha=0.8, ls="-" if m == "clean" else "--")
            ax.axvline(thr[g], color="crimson", lw=0.6, ls="--")
            if lg:
                ax.set_yscale("log"); ax.set_ylim(1e-3, None)
                if k == "null_shift":
                    ax.axhline(NULL_MAX, color="0.5", lw=0.4)
            else:
                ax.set_ylim(0, 1.02)
            ax.set_xlim(1, 15)
            if i == 0:
                ax.set_title(lab, loc="left", fontsize=5.2)
            if i == len(G) - 1:
                ax.set_xlabel("threshold (normalised readout)")
            if j == 0:
                ax.set_ylabel(f"{m2.short(g)}\nthr {thr[g]:.2f}", color=c)
    n_s = K.session_id.nunique()
    fig.suptitle(f"{target}: one threshold per area group, the same for every session and template ({n_s} sessions; solid clean, "
                 f"dashed spont; grey: pAUC < {PAUC_MIN:g} or artefact, not used); red: smallest threshold with time-shift null <= {NULL_MAX:g} / min in every eligible decoder",
                 x=0.01, ha="left", fontsize=5.8, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    m2.save(fig, m1.HOME / "figures", f"fig6_group_thresholds_{target}")
    plt.close(fig)


def main(target, apply):
    K = curves(target)
    thr = choose(K)
    rule = f"004: smallest threshold with time-shift null <= {NULL_MAX:g}/min in every eligible decoder (pAUC >= {PAUC_MIN:g})"
    (m1.HOME / f"thresholds_{target}.json").write_text(json.dumps(dict(
        thr=thr, rule=rule, null_max=NULL_MAX, pauc_min=PAUC_MIN, artefact=sorted(map(list, ARTEFACT)),
        n_eligible=K.drop_duplicates(["session_id", "model", "group"]).groupby("group").eligible.sum().astype(int).to_dict(), sessions=sorted(K.session_id.unique()), models=sorted(K.model.unique())), indent=1))
    K.to_csv(m1.HOME / f"group_threshold_curves_{target}.csv", index=False)
    figure(K, thr, target)
    at = pd.concat([q[np.isclose(q.thr, thr[g])] for g, q in K.groupby("group")])
    pd.set_option("display.width", 250)
    print(at.drop(columns=["target", "thr"]).assign(thr=at.group.map(thr)).round(3).to_string(index=False))
    if apply:
        for sid, m in at[["session_id", "model"]].drop_duplicates().itertuples(index=False):
            print(sid, m, flush=True)
            m1.apply_thresholds(sid, target, m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="whisker")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    main(a.target, a.apply)
