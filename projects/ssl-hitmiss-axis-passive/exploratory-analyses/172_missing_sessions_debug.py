"""172 -- Why some learner sessions drop out of panels of 165 (user 2026-10-08: "debug these sessions missing, plot them").
Diagnostic. One panel per cause:
  a  hit - miss axis, no shifted-label null (AB123 R+, AB139 R-): active whisker trial outcomes in time order (hit / miss); the
     shifted-label null rebuilds the axis on labels shifted by 10-50 % of the trials (non-wrapping), so each shifted subset drops the
     trials at one end and needs >= 10 hits and >= 10 misses (166 / 169; 135 similar); with exactly 10 trials in the minority class,
     no shift keeps 10 -> no null. Shown: the number of minority-class trials kept by every shift.
  b  auditory angle undefined (AB133, AB140, R-): cross-validated passive auditory response size rho (169) pre and post for every
     learner session; these two have rho_post = 0 (split-half product <= 0: no reproducible auditory response after the task).
  c  reward-lick axis, no 026 result (AB143, MH017): event timeline (WH, AH, SL) over session time; both have exactly 8 auditory
     hits (the minimum, MIN_EV = 8), so every shifted subset has < 8 and fewer than 5 shifts survive -> no null.
  d  reward-lick bins (AB161, R-): no whisker hit before tau = 0.44, so the first fifth is empty and the "relative to first fifth"
     curve is undefined.
Inputs: 135_alignment_epochs_tracked_harm.parquet (counts), 169_along_orthogonal.parquet, 172_prelick_timelines.csv (exported from
the 051 trial files), session trials (132 session_trials). Output: figures/172_missing_sessions_debug.png
Run (haas, repo root): python .../172_missing_sessions_debug.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
H = importlib.import_module("143_lt_split_windows_figures")
M135 = importlib.import_module("135_whisker_lick_alignment_epochs")
COL = H.COL
FRAC, N_SHIFT, MIN_CLASS = (0.10, 0.50), 50, 10


def outcomes(sid):
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    act = M132.session_trials(sid, st, tt, T)["active"].sort_values("start_time").reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    keep = ~licked | (rt > M135.WIN[1]); act, licked = act[keep].reset_index(drop=True), licked[keep]
    return licked[(act.trial_type == "whisker_trial").to_numpy()]


def kept_by_shift(y):
    n = len(y); ks = np.unique(np.linspace(FRAC[0] * n, FRAC[1] * n, N_SHIFT // 2).astype(int)); out = []
    for k in np.r_[ks, -ks]:
        i = np.arange(max(0, -k), min(n, n - k)); ys = y[i]
        out.append((k, int(ys.sum()), int((~ys).sum())))
    return np.array(out)


def main():
    H.setup()
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.3)
    # a: hit / miss sequences and minority class kept by every shift
    for j, (sid, c) in enumerate((("AB123_20240806_110231", "R+"), ("AB139_20241119_115957", "R-"))):
        y = outcomes(sid)
        ax = fig.add_subplot(gs[0, j])
        ax.vlines(np.where(y)[0], 0.55, 1.0, color="0.1", lw=0.8); ax.vlines(np.where(~y)[0], 0.0, 0.45, color="0.6", lw=0.8)
        K = kept_by_shift(y); mino = np.minimum(K[:, 1], K[:, 2])
        ax2 = ax.twinx(); ax2.plot(np.abs(K[:, 0]) * np.sign(K[:, 0]) + len(y) / 2, mino, "o", ms=2.5, color=COL[c])
        ax2.axhline(MIN_CLASS, color=COL[c], ls=":", lw=0.8); ax2.set_ylabel("minority class kept\nby each shift", fontsize=6, color=COL[c])
        ax.set_yticks([0.22, 0.78]); ax.set_yticklabels(["miss", "hit"]); ax.set_xlabel("active whisker trial (time order)", fontsize=6)
        ax.set_title(f"a  {sid} ({c}): {int(y.sum())} hits / {int((~y).sum())} misses; max minority kept by a shift = {mino.max()} "
                     f"(need {MIN_CLASS}) -> no shifted-label null", fontsize=6, loc="left")
    # b: auditory response size
    G = pd.read_parquet(EA / "169_along_orthogonal.parquet"); G = G[(G.stim == "A") & H.mouse_of(G).isin(H.learners())]
    ax = fig.add_subplot(gs[1, 0])
    for c in ("R+", "R-"):
        d = G[G.reward_group == c]; ax.plot(d.rho_pre, d.rho_post, "o", ms=3, color=COL[c], alpha=0.7, label=c)
    bad = G[G.rho_post <= 0]
    ax.plot(bad.rho_pre, bad.rho_post, "o", ms=7, mfc="none", mec="black")
    for _, r in bad.iterrows():
        ax.text(r.rho_pre, r.rho_post + 0.04, f" {r.session_id[:5]} (n pre {r.n_pre}, post {r.n_post})", fontsize=5)
    ax.plot([0, 1.5], [0, 1.5], color="0.8", lw=0.6); ax.legend(frameon=False, fontsize=5)
    ax.set_xlabel("auditory response size ρ, passive pre (cross-validated)", fontsize=6); ax.set_ylabel("ρ, passive post", fontsize=6)
    ax.set_title("b  auditory angle undefined: no reproducible auditory response after the task (ρ_post = 0)", fontsize=6, loc="left")
    # c, d: prelick timelines
    Tl = pd.read_csv(EA / "172_prelick_timelines.csv")
    cl = {"WH": ("#f7b519", 2), "AH": ("#2c2cdb", 1), "FA": ("0.5", 0)}
    for k, (sid, ttl) in enumerate((("AB143_20241126_115737", "c  AB143 (R+): 8 auditory hits (minimum 8) -> shifted subsets < 8, < 5 shifts -> no null"),
                                     ("MH017_20250227_113307", "c  MH017 (R+): 8 auditory hits (minimum 8) -> no null"),
                                     ("AB161_20250417_173823", "d  AB161 (R-): first whisker hit at τ = 0.44 -> first fifth empty"))):
        ax = fig.add_subplot(gs[1, 1]) if k == 0 else fig.add_subplot(gs[2, k - 1])
        d = Tl[Tl.session_id == sid]
        for c_, (col, yy) in cl.items():
            x = d[d.cls == c_].tau.to_numpy(); ax.vlines(x, yy, yy + 0.8, color=col, lw=0.7)
            ax.text(1.01, yy + 0.4, f"{'SL' if c_ == 'FA' else c_} ({len(x)})", fontsize=5.5, va="center", transform=ax.get_yaxis_transform())
        if k == 2:
            for b in np.linspace(0, 1, 6):
                ax.axvline(b, color="0.8", lw=0.5, ls=":")
        ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("normalised session time τ", fontsize=6); ax.set_title(ttl, fontsize=6, loc="left")
    (EA / "figures").mkdir(exist_ok=True)
    fig.savefig(EA / "figures" / "172_missing_sessions_debug.png", dpi=200, bbox_inches="tight")
    print("ALL DONE")


if __name__ == "__main__":
    main()
