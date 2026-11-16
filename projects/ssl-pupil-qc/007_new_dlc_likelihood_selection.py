"""New DLC (2026-11-11): does selecting on likelihood change the trial-averaged pupil, in particular the small early dip
after the stimulus (006)? (user, 2026-11-12: "I don't want to drop these sessions but just select on likelihood")

Sessions: those where the new model is under-confident on the pupil (fraction of frames with all 4 points >= 0.5 fell to
3-68 %), plus two clean reference sessions. Per session, aligned to trial start (-1 to 2 s, 20 ms bins), whisker and
auditory trials (hits + misses), no-stim trials as reference; 4-point area change from the trial baseline (-1..0 s) in %
of the session median (as 006):
  col 1: old NWB (NWB_ks4_backup_20261111), frames with all 4 points >= 0.5 (converter cutoff, QC "valid")
  col 2: new NWB, frames with all 4 points >= 0.5
  col 3: new NWB, frames >= 0.5, high-likelihood TRIALS only (>= 90 % of the frames of the -1..2 s window with all 4
         points >= 0.5)
  col 4: new NWB, frames with all 4 points >= 0.1 (dlc_helpers cutoff), all trials
  col 5: % of frames with all 4 points >= 0.5 across the trial, old (dashed) vs new (solid)
Raw NWB keypoints (THREE_POINT correction applied by pupil_qc.load_session), no QC states, no smoothing.
Output: combined_results_ks4/ssl-pupil-qc/figures/007_new_dlc_likelihood_selection.png (+ .pdf),
        tables/007_new_dlc_likelihood_selection.csv (dip per panel, QC verdicts old / new)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 007_new_dlc_likelihood_selection.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import pupil_qc as pq  # noqa: E402

SESSIONS = ["AB154_20250207_131706", "AB161_20250417_173823", "AB161_20250418_093334", "MH038_20250525_130335",
            "MH028_20250502_123728", "MH028_20250503_112316", "MH064_20260115_104148", "MH019_20250304_115527",
            "AB105_20240314_115206", "AB164_20250422_115457", "AB131_20240905_123601"]
OLD_NWB = pq.NWB_DIR.parent / "NWB_ks4_backup_20261111"
OUT = pq.RESULTS / "ssl-pupil-qc"
EDGES = np.arange(-1.0, 2.0 + 1e-9, 0.02)
X = (EDGES[:-1] + EDGES[1:]) / 2
GROUPS = {"Auditory": ["A hit", "A miss"], "Whisker": ["W hit", "W miss"], "No stim.": ["FA", "CR"]}
COLORS = {"Auditory": "#2c2cdb", "Whisker": "#f7b519", "No stim.": "0.3"}
TRIAL_MIN_HIGH = 0.9
PANELS = ["old NWB, frames >= 0.5", "new NWB, frames >= 0.5", f"new, frames >= 0.5, trials >= {TRIAL_MIN_HIGH:.0%} high",
          "new NWB, frames >= 0.1"]


def locked(x, t, starts):
    nb = len(EDGES) - 1
    out = np.full((len(starts), nb), np.nan)
    for j, s in enumerate(starts):
        idx = np.searchsorted(t, s + EDGES)
        for k in range(nb):
            seg = x[idx[k]:idx[k + 1]]
            if len(seg) and np.isfinite(seg).any():
                out[j, k] = np.nanmean(seg)
    return out


def load(sid, nwb_dir):
    pq.NWB_DIR = nwb_dir
    df, trials, meta = pq.load_session(sid)
    L = np.column_stack([df[f"{q}_likelihood"] for q in pq.POINTS]).min(1)
    return df.t.to_numpy(), pq.polygon_area(df), L, trials, meta


def verdicts(meta):
    d = pq.RESULTS / meta["mouse_id"] / meta["folder"] / "pupil_qc"
    out = {}
    for tag, f in (("old", d / "old_dlc_20261111" / f"{meta['session_id']}_metrics.json"),
                   ("new", d / f"{meta['session_id']}_metrics.json")):
        if f.exists():
            m = json.load(open(f))
            out.update({f"verdict_{tag}": m["verdict"], f"pct_valid_{tag}": m["pct_valid"],
                        f"pct_trials_valid_{tag}": m["pct_trials_valid"]})
    return out


def one(sid):
    data = {"old": load(sid, OLD_NWB), "new": load(sid, pq.NWB_DIR.parent / "NWB_ks4")}
    meta = data["new"][4]
    res = dict(sid=sid, desc=meta["description"], traces={}, valid={}, rows=[], meta=verdicts(meta))
    early = (X >= 0) & (X < 0.5)
    for g, outcomes in GROUPS.items():
        for tag in ("old", "new"):
            t, area, L, trials, _ = data[tag]
            starts = trials.loc[trials.outcome.isin(outcomes), "start_time"].to_numpy()
            ok5 = L >= 0.5
            res["valid"][(g, tag)] = 100 * np.nanmean(locked(ok5.astype(float), t, starts), 0)
            if tag == "old":
                variants = {0: (ok5, np.ones(len(starts), bool))}
            else:
                frac = np.array([ok5[np.searchsorted(t, s - 1):np.searchsorted(t, s + 2)].mean() for s in starts])
                variants = {1: (ok5, np.ones(len(starts), bool)), 2: (ok5, frac >= TRIAL_MIN_HIGH),
                            3: (L >= 0.1, np.ones(len(starts), bool))}
            for col, (ok, keep) in variants.items():
                v = np.where(ok, area, np.nan)
                tr = locked(v, t, starts[keep])
                d = (tr - np.nanmean(tr[:, X < 0], 1, keepdims=True)) / np.nanmedian(v) * 100
                trace = np.nanmean(d, 0) if len(d) else np.full(len(X), np.nan)
                res["traces"][(g, col)] = (trace, int(keep.sum()), len(starts))
                dip = trace[early]
                res["rows"].append(dict(session_id=sid, day=meta["description"], group=g, panel=PANELS[col],
                                        n_trials=int(keep.sum()), n_trials_all=len(starts),
                                        dip_min_0_500ms=np.nanmin(dip) if np.isfinite(dip).any() else np.nan,
                                        t_dip_ms=1000 * X[early][np.nanargmin(dip)] if np.isfinite(dip).any()
                                        else np.nan, **res["meta"]))
    return res


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    out = Parallel(n_jobs=len(SESSIONS))(delayed(one)(s) for s in SESSIONS)
    fig, axs = plt.subplots(len(out), 5, figsize=(17, 2.3 * len(out)), sharex=True)
    rows = []
    for r, res in enumerate(out):
        rows += res["rows"]
        for col in range(4):
            ax = axs[r, col]
            ns = []
            for g in GROUPS:
                if (g, col) in res["traces"]:
                    trace, n, n_all = res["traces"][(g, col)]
                    ax.plot(X, trace, color=COLORS[g], lw=0.9)
                    ns.append(f"{g[0]} {n}/{n_all}")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.text(0.98, 0.04, "trials " + ", ".join(ns), transform=ax.transAxes, ha="right", fontsize=5.5)
            if r == 0:
                ax.set_title(PANELS[col], fontsize=8.5)
        for g in GROUPS:
            for tag, ls in (("old", "--"), ("new", "-")):
                axs[r, 4].plot(X, res["valid"][(g, tag)], color=COLORS[g], ls=ls, lw=0.9)
        axs[r, 4].set_ylim(0, 102)
        m = res["meta"]
        axs[r, 0].set_ylabel(f"{res['sid']}\n{res['desc']}\nQC {m.get('verdict_old', '?')} -> "
                             f"{m.get('verdict_new', '?')}\nΔ area (% median)", fontsize=6.5)
        axs[r, 4].set_ylabel("% frames all 4 >= 0.5", fontsize=6.5)
        lo = min(np.nanmin(axs[r, c].get_ylim()[0]) for c in range(4))
        hi = max(np.nanmax(axs[r, c].get_ylim()[1]) for c in range(4))
        for c in range(4):
            axs[r, c].set_ylim(lo, hi)
    axs[0, 4].set_title("Frames above 0.5 (dashed old, solid new)", fontsize=8.5)
    for c in range(5):
        axs[-1, c].set_xlabel("Time from trial start (s)")
    handles = [plt.Line2D([], [], color=col, label=g) for g, col in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"007_new_dlc_likelihood_selection.{ext}", dpi=140)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tables" / "007_new_dlc_likelihood_selection.csv", index=False)
    w = tab[tab.group != "No stim."].pivot_table(index=["session_id", "group"], columns="panel",
                                                 values="dip_min_0_500ms")[PANELS]
    print(w.round(1).to_string())
    print(tab.drop_duplicates("session_id")[["session_id", "verdict_old", "verdict_new", "pct_valid_old", "pct_valid_new",
                                             "pct_trials_valid_old", "pct_trials_valid_new"]].round(1).to_string())


if __name__ == "__main__":
    main()
