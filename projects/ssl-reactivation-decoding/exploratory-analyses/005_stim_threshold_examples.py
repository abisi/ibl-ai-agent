"""005 -- Decoders trained on stimulus data only, threshold per decoder from stimulus data, applied to continuous data (user 2026-10-24).

"Rewind": no spontaneous windows in training (the clean template of 001: quiet target stimuli vs correct rejections + passive
shams), no pooled or session-normalised threshold. Each decoder (session x area group) gets its own threshold in its own
decision-value units (log-odds, w . z + b), set on its held-out stimulus data: the value that FPR_HELD of the held-out
no-stimulus events exceed (the low-FPR point the regularisation is chosen for, 001 MAX_FPR). The threshold is applied to the
decoder's raw continuous readout (30-ms windows, 5-ms steps, valid time = outside -100..600 ms of trial onsets and shams).
Reported per decoder: threshold, held-out sensitivity, events / min (all valid time and quiet time), events per epoch.
The time-shift null of 001 is in per-trace normalised units; the threshold is mapped with the data trace's median / MAD
(approximate: every null trace is normalised by its own).

Figures (per session, <pilot>/<session>/whisker/stim_threshold/):
  fig7_examples_<session>: per area group (rows), six continuous events chosen to span the range: the two strongest, two at the
    median peak, two just above threshold; drawn from different epochs when possible. Each: raster of the group's units in
    +-0.5 s (sorted by decoder weight; colour = positive weight), readout with threshold, licks and whisker speed.
Table: stim_threshold_summary.csv (all sessions) in <HOME>.
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
FPR_HELD = 0.05
SESSIONS = ["MH070_20260121_140848", "AB150_20241219_144646", "AB136_20241117_140311", "AB157_20250412_154744",
            "MH030_20250503_154256", "MH031_20250509_102210", "MH036_20250517_140456", "MH035_20250517_105351"]
STAGE = {"MH070": "day 0 R+", "AB150": "day 0 R+", "AB136": "day 0 R-", "AB157": "day 0 R-",
         "MH030": "expert R+", "MH031": "expert R+", "MH036": "expert R-", "MH035": "expert R-"}
HALF = 0.5                                       # s around an example event
N_EX = 6


def epoch_of(t, epochs, rec_end):
    for k in ("passive_pre", "active", "passive_post"):
        for a, b in epochs.get(k, []):
            if a <= t <= b:
                return {"active": "task"}.get(k, k)
    return "outside"


def decoder(out, g):
    c = dict(np.load(out / f"continuous_{g.replace(' ', '_')}.npz"))
    c["raw"] = c["trace"].astype(float) * float(c["norm_scale"]) + float(c["norm_median"])
    neg, pos = c["oof"][c["y"] == 0], c["oof"][c["y"] == 1]
    c["thr"] = float(np.quantile(neg, 1 - FPR_HELD))
    c["sens"] = float(np.mean(pos > c["thr"]))
    c["thr0"] = float(neg.max())                      # stricter alternative: no held-out no-stimulus event crosses
    c["sens0"] = float(np.mean(pos > c["thr0"]))
    return c


def session(sid, plt, col):
    out = m1.HOME / "pilot" / sid / "whisker" / "clean"
    fdir = out.parent / "stim_threshold"
    P = json.loads((out / "params.json").read_text())
    F = pd.read_csv(out / "fit_summary.csv")
    N = pd.read_parquet(out / "null_curves.parquet")
    B = dict(np.load(out / "behaviour.npz"))
    epochs, rec_end = P["epochs"], P["recording"][1]
    units, *_ = m1.load_session(sid)
    rows, ex = [], {}
    for g in F.group:
        c = decoder(out, g)
        D = m1.detect(c["raw"], c["valid"], c["thr"])
        t = c["t"]
        E = pd.DataFrame(dict(t_peak=t[D[:, 2]], peak=c["raw"][D[:, 2]], quiet=c["quiet"][D[:, 2]],
                              dur_ms=(D[:, 1] - D[:, 0] + 1) * m1.STEP * 1000)) if len(D) else pd.DataFrame(
            columns=["t_peak", "peak", "quiet", "dur_ms"])
        E["epoch"] = [epoch_of(x, epochs, rec_end) for x in E.t_peak]
        fdir.mkdir(parents=True, exist_ok=True)
        E.to_csv(fdir / f"events_{g.replace(' ', '_')}.csv", index=False)
        n = N[N.group == g]
        thr_n = (c["thr"] - float(c["norm_median"])) / float(c["norm_scale"])
        vm = c["valid"].sum() * m1.STEP / 60
        qm = (c["valid"] & c["quiet"]).sum() * m1.STEP / 60
        per_ep = {}
        for k in ("passive_pre", "task", "passive_post", "outside"):
            tm = np.array([epoch_of(x, epochs, rec_end) == k for x in t[::200]])      # epoch of the valid time, 1-s sampling
            mins = (tm & c["valid"][::200]).sum() / 60
            per_ep[f"rate_{k}"] = (E.epoch == k).sum() / mins if mins > 0.5 else np.nan
        rows.append(dict(session_id=sid, stage=STAGE[sid[:5]], group=g, n_units=int(F.set_index("group").loc[g, "n_units"]),
                         pauc=float(F.set_index("group").loc[g, "pauc"]), thr=c["thr"], thr_norm=thr_n, sens=c["sens"],
                         fpr_held=FPR_HELD, n_events=len(E), rate=len(E) / vm, rate_quiet=E.quiet.sum() / qm if qm else np.nan,
                         null_shift=float(np.interp(thr_n, n.thr, n.null_shift)), **per_ep,
                         thr0=c["thr0"], sens0=c["sens0"], rate0=len(m1.detect(c["raw"], c["valid"], c["thr0"])) / vm,
                         null_shift0=float(np.interp((c["thr0"] - float(c["norm_median"])) / float(c["norm_scale"]), n.thr, n.null_shift))))
        ex[g] = (c, E)
        print(f"{sid} {g}: thr {c['thr']:.2f} (norm {thr_n:.1f}), sens {c['sens']:.2f}, {len(E) / vm:.2f}/min "
              f"(null ~{rows[-1]['null_shift']:.2f}), quiet {rows[-1]['rate_quiet']:.2f}", flush=True)
    fig_examples(plt, col, sid, units, ex, B, fdir)
    return rows


def pick(E, rng):
    """six events spanning the range: 2 strongest, 2 at the median, 2 just above threshold; different epochs preferred"""
    if len(E) <= N_EX:
        return E.sort_values("peak", ascending=False)
    E = E.sort_values("peak", ascending=False).reset_index(drop=True)
    sel = []
    for centre in (0, len(E) // 2, len(E) - 1):
        lo, hi = max(0, centre - 15), min(len(E), centre + 16)
        cand = E.iloc[lo:hi]
        got = []
        for ep in rng.permutation(cand.epoch.unique()):
            i = cand.index[cand.epoch == ep]
            got.append(i[np.argmin(np.abs(i - centre))])
            if len(got) == 2:
                break
        while len(got) < 2:
            rest = [i for i in cand.index if i not in got]
            got.append(rest[np.argmin(np.abs(np.array(rest) - centre))])
        sel += got
    return E.loc[sel]


def fig_examples(plt, col, sid, units, ex, B, fdir):
    G = [g for g in m1.GROUPS if g in ex]
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.4, 2.1 * len(G)), layout="constrained")
    sub = fig.subfigures(len(G), 1)
    sub = np.atleast_1d(sub)
    for r, g in enumerate(G):
        c, E = ex[g]
        cc = col.get(g, "k")
        axs = sub[r].subplots(3, N_EX, sharex="col", gridspec_kw=dict(height_ratios=[2.2, 1, 0.5], hspace=0.05))
        vm = c["valid"].sum() * m1.STEP / 60
        sub[r].suptitle(f"{m2.short(g)}: {len(c['w'])} units, threshold {c['thr']:.2f} (held-out FPR {FPR_HELD:.0%}, "
                        f"sensitivity {c['sens']:.2f}), {len(E) / vm:.1f} events / min", x=0.01, ha="left", fontsize=6,
                        color=cc, weight="bold")
        P = pick(E, rng) if len(E) else E
        order = np.argsort(c["w"])[::-1]
        for j in range(N_EX):
            a0, a1, a2 = axs[:, j]
            if j >= len(P):
                for a in (a0, a1, a2):
                    a.axis("off")
                continue
            e = P.iloc[j]
            t0 = float(e.t_peak)
            for k, u in enumerate(order):
                st = np.asarray(units.spike_times.iloc[int(c["unit_index"][u])])
                st = st[(st >= t0 - HALF) & (st < t0 + HALF)] - t0
                a0.vlines(st, k, k + 0.9, color=cc if c["w"][u] > 0 else "0.6", lw=0.3)
            a0.set_ylim(0, len(order)); a0.set_yticks([])
            a0.axvspan(-m1.WIDTH, 0, color="crimson", alpha=0.08, lw=0)
            a0.set_title(f"{e.epoch}{', quiet' if e.quiet else ''}\npeak {e.peak:.1f}, t = {t0:.0f} s", fontsize=4.8)
            m = (c["t"] >= t0 - HALF) & (c["t"] <= t0 + HALF)
            a1.plot(c["t"][m] - t0, np.where(c["valid"][m], c["raw"][m], np.nan), color=cc, lw=0.6)
            a1.plot(c["t"][m] - t0, np.where(c["valid"][m], np.nan, c["raw"][m]), color="0.8", lw=0.6)
            a1.axhline(c["thr"], color="crimson", lw=0.5, ls="--")
            lk = B["licks"][(B["licks"] >= t0 - HALF) & (B["licks"] < t0 + HALF)] - t0
            a2.vlines(lk, 0, 1, color="k", lw=0.5)
            if "whisker_speed_t" in B:
                mm = (B["whisker_speed_t"] >= t0 - HALF) & (B["whisker_speed_t"] < t0 + HALF)
                v = B["whisker_speed_v"][mm]
                if len(v) and np.isfinite(v).any():
                    a2.plot(B["whisker_speed_t"][mm] - t0, 1 + np.clip(v, 0, 2000) / 2000, color="0.4", lw=0.4)
            a2.set_ylim(0, 2.05); a2.set_yticks([])
            a2.set_xlim(-HALF, HALF)
            if j == 0:
                a0.set_ylabel("units\n(by weight)", fontsize=5)
                a1.set_ylabel("readout\n(log-odds)", fontsize=5)
                a2.set_ylabel("licks /\nwhisk", fontsize=5)
            if r == len(G) - 1:
                a2.set_xlabel("time from event peak (s)")
    fig.suptitle(f"{sid} ({STAGE[sid[:5]]}): whisker decoders trained on stimulus data only, threshold per decoder\n"
                 "events: 2 strongest, 2 median, 2 near threshold; raster colour: positive weight; shade: 30-ms readout window",
                 x=0.01, ha="left", fontsize=6.5, weight="bold")
    m2.save(fig, fdir, f"fig7_examples_{sid}")
    m2.save(fig, m1.HOME / "figures" / "stim_threshold_examples", f"fig7_examples_{sid}")
    plt.close(fig)


def main(sessions):
    plt = m2.setup()
    col = m2.colors()
    rows = []
    for sid in sessions:
        rows += session(sid, plt, col)
    S = pd.DataFrame(rows)
    f = m1.HOME / "stim_threshold_summary.csv"
    if f.exists() and len(sessions) < len(SESSIONS):
        old = pd.read_csv(f)
        S = pd.concat([old[~old.session_id.isin(sessions)], S], ignore_index=True)
    S.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    print(S[["session_id", "stage", "group", "pauc", "thr", "sens", "rate", "rate_quiet", "null_shift", "rate_passive_pre",
             "rate_task", "rate_passive_post", "rate_outside", "thr0", "sens0", "rate0", "null_shift0"]].round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="*", default=SESSIONS)
    a = ap.parse_args()
    main(a.session)
