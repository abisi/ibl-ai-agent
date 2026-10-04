"""026 -- Review of end-of-session disengagement trimming on every session (user 2026-09-30: "Show how disengagement is
trimmed then for all sessions. I am unsure whether I should remove it").
Two candidate criteria (from 025):
  A  rule used in the decoding pipelines (ssl_timeresolved_decoding.detect_terminal_disengagement): everything after the
     session's last lick on any trial type, only if that tail holds >= 5 whisker AND >= 3 auditory trials;
  A1 as A but >= 1 auditory trial in the tail (user test);
  B  auditory HMM curve (sigma = 1) < 0.5 held from some whisker trial to the end of the session;
  C  same as B with threshold 0.3 (user);
  D  overall lick rate < 0.10 held to the end (user): HMM curve (sigma = 1) fitted on ALL trials (whisker, auditory,
     no stim) in time order from the first whisker trial; cut = first trial from which it stays < 0.10, expressed in
     whisker trials (whisker trials from that point on are removed). Weighted by trial-type proportions, so auditory
     trials count little (user);
  E  balanced overall lick rate < 0.10 held to the end: mean of the whisker, FA and auditory curves (equal weight per
     trial type, on the whisker-trial axis).
Panel marks: red shading = A; bars at the top: orange = B, brown = C, black = D; dashed black line = overall lick-rate
curve (D) at the trials' whisker-axis positions; blue = E (dotted blue curve).
Per session, one panel: whisker (cohort colour), false alarm (grey) and auditory (blue) curves on the whisker-trial
axis; lick ticks above and no-lick ticks below (rows: auditory, whisker, no stim; auditory misses light blue, whisker
hits green / misses purple, FA black / CR grey, as 021); red shading = trimmed by A, orange bar at the top = trimmed by
B; title = mouse, whisker trials removed by A / B. Sessions flagged by A or B first, then the rest, per cohort.
Decision (user 2026-09-30): learning-trial methods and identification keep the full session (no trimming); the
trimming rule is for subsequent neural analyses.
Outputs: 026_disengagement_review_{Rplus,Rminus}.{pdf,png}; artifacts/026_disengagement_review.csv
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/026_disengagement_review.py
"""

from __future__ import annotations

import importlib
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
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
M025 = importlib.import_module("025_lt_aligned_heatmaps_summary")
import lt_lib as L  # noqa: E402

GROUP_COLORS = {"rplus": "#00B400", "rminus": "#C800C8"}
COL = {"R+": GROUP_COLORS["rplus"], "R-": GROUP_COLORS["rminus"]}
AUD, FA, AUD_MISS = "#1f5fbf", "0.35", "#8fd3f5"
TICKS = {"a": (AUD, AUD_MISS), "w": (GROUP_COLORS["rplus"], GROUP_COLORS["rminus"]), "n": ("k", "0.65")}
NCOL = 8


def tail_cut(v, thr):
    """First index from which v stays < thr to the end (NaN if the last value is >= thr)."""
    v = np.asarray(v, float)
    if not len(v) or not np.isfinite(v).all() or v[-1] >= thr:
        return np.nan
    i = len(v) - 1
    while i > 0 and v[i - 1] < thr:
        i -= 1
    return i


def overall_curve(d):
    """HMM posterior-mean lick probability over all trials (time order, from the first whisker trial)."""
    dd = d[d.pos.notna()].sort_values("t")
    return dd, L.forward_backward(dd.y.to_numpy().astype(int), 1.0)[0] @ L.P_GRID


def removed(n, cut):
    return n - int(np.ceil(cut)) if np.isfinite(cut) else 0


def panel(ax, s, S):
    d, c = S[s]["trials"], S[s]["curves"]
    n = S[s]["n_whisker"]
    x = np.arange(n)
    if S[s]["disengaged"]:
        ax.axvspan(S[s]["n_engaged"] - 0.5, n - 0.5, color="#e41a1c", alpha=0.15, lw=0)
    for key, yy, cc in (("cut_b", 1.48, "#ff7f00"), ("cut_c", 1.58, "#8c510a"), ("cut_d", 1.68, "k"), ("cut_e", 1.78, "#377eb8"), ("cut_a1", 1.88, "#e41a1c")):
        if np.isfinite(S[s][key]):
            ax.plot([S[s][key] - 0.5, n - 0.5], [yy, yy], color=cc, lw=2.0, solid_capstyle="butt")
    ax.plot(S[s]["overall_pos"], S[s]["overall"], color="k", lw=0.6, ls="--")
    ax.plot(x, S[s]["balanced"], color="#377eb8", lw=0.7, ls=":")
    ax.plot(x, c["auditory"], color=AUD, lw=0.9)
    ax.plot(x, c["fa"], color=FA, lw=0.9)
    ax.plot(x, c["whisker"], color=COL[S[s]["rg"]], lw=1.1)
    for k, typ in enumerate(("a", "w", "n")):
        t = d[(d.tt == typ) & d.pos.notna()]
        cl, cn = TICKS[typ]
        ax.vlines(t.pos[t.y == 1], 1.06 + 0.12 * k, 1.14 + 0.12 * k, color=cl, lw=0.4)
        ax.vlines(t.pos[t.y == 0], -0.14 - 0.12 * k, -0.06 - 0.12 * k, color=cn, lw=0.4)
    ax.set_ylim(-0.44, 1.94)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=5, length=2, pad=1)
    ra = n - S[s]["n_engaged"] if S[s]["disengaged"] else 0
    rb, rc, rd, re_, ra1 = (removed(n, S[s][k]) for k in ("cut_b", "cut_c", "cut_d", "cut_e", "cut_a1"))
    flag = ra or rb or rc or re_ or ra1
    ax.set_title(f"{S[s]['mouse_id']} ({S[s]['lc'] or '?'}) A{ra} A1:{ra1} B{rb} C{rc} D{rd} E{re_}", fontsize=5.2,
                 fontweight="bold" if flag else "normal", pad=2)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6})
    tt = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "trials.parquet")
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    inputs.update(pickle.load(open(ART / "022_inputs_behaviour_only.pkl", "rb")))
    per = {p["session_id"]: p for p in pickle.load(open(ART / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
    S = M025.session_table(inputs, tt, _active_trials_for_curve_untrimmed)
    rows = []
    for s in S:
        S[s].update(rg=per[s]["reward_group"], lc=per[s]["learning_category"], curves=per[s]["curves"],
                    mouse_id=per[s]["mouse_id"])
        S[s]["cut_b"] = tail_cut(S[s]["curves"]["auditory"], 0.5)
        S[s]["cut_c"] = tail_cut(S[s]["curves"]["auditory"], 0.3)
        dd, ov = overall_curve(S[s]["trials"])
        S[s]["overall"], S[s]["overall_pos"] = ov, dd.pos.to_numpy()
        i = tail_cut(ov, 0.10)
        S[s]["cut_d"] = dd.pos.to_numpy()[int(i)] if np.isfinite(i) else np.nan
        cv = S[s]["curves"]
        S[s]["balanced"] = np.nanmean(np.vstack([cv["whisker"], cv["fa"], cv["auditory"]]), 0)
        S[s]["cut_e"] = tail_cut(S[s]["balanced"], 0.10)
        a1 = S[s]["trail_whisker"] >= 5 and S[s]["trail_auditory"] >= 1
        S[s]["cut_a1"] = float(S[s]["n_whisker"] - S[s]["trail_whisker"]) if a1 else np.nan
        n = S[s]["n_whisker"]
        rows.append(dict(session_id=s, mouse_id=S[s]["mouse_id"], reward_group=S[s]["rg"], learning_category=S[s]["lc"],
                         n_whisker=n, a_flag=S[s]["disengaged"],
                         a_whisker_removed=n - S[s]["n_engaged"] if S[s]["disengaged"] else 0,
                         a_tail_whisker=S[s]["trail_whisker"], a_tail_auditory=S[s]["trail_auditory"],
                         b_flag=bool(np.isfinite(S[s]["cut_b"])),
                         b_whisker_removed=removed(n, S[s]["cut_b"]),
                         c_flag=bool(np.isfinite(S[s]["cut_c"])), c_whisker_removed=removed(n, S[s]["cut_c"]),
                         d_flag=bool(np.isfinite(S[s]["cut_d"])), d_whisker_removed=removed(n, S[s]["cut_d"]),
                         e_flag=bool(np.isfinite(S[s]["cut_e"])), e_whisker_removed=removed(n, S[s]["cut_e"]),
                         a1_flag=bool(np.isfinite(S[s]["cut_a1"])), a1_whisker_removed=removed(n, S[s]["cut_a1"])))
    tab = pd.DataFrame(rows)
    tab.to_csv(ART / "026_disengagement_review.csv", index=False)
    for rg, tag in (("R+", "Rplus"), ("R-", "Rminus")):
        g = tab[tab.reward_group == rg].copy()
        anyf = g.a_flag | g.a1_flag | g.b_flag | g.c_flag | g.e_flag
        g["order"] = -(g.a_whisker_removed + g.a1_whisker_removed + g.b_whisker_removed + g.c_whisker_removed
                       + g.e_whisker_removed + 1000 * anyf)
        sids = g.sort_values(["order", "mouse_id"]).session_id.tolist()
        nrow = int(np.ceil(len(sids) / NCOL))
        fig, axes = plt.subplots(nrow, NCOL, figsize=(11.7, 1.05 * nrow + 0.7), squeeze=False)
        fig.subplots_adjust(left=0.02, right=0.995, top=1 - 0.55 / (1.05 * nrow + 0.7), bottom=0.02, wspace=0.18,
                            hspace=0.55)
        for ax, s in zip(axes.flat, sids):
            panel(ax, s, S)
        for ax in axes.flat[len(sids):]:
            ax.set_axis_off()
        na, nb, nc, nd, ne = (int(g[f"{k}_flag"].sum()) for k in "abcde")
        na1 = int(g.a1_flag.sum())
        fig.suptitle(f"{'R+' if rg == 'R+' else 'R−'} (n = {len(g)}): x = whisker trial. Red shading = trimmed by A "
                     f"(tail after the last lick, >= 5 whisker and >= 3 auditory; {na} sessions). Bars: orange = B (auditory curve < 0.5 to the end; "
                     f"{nb}), brown = C (auditory < 0.3; {nc}), black = D (overall lick rate, all trials, < 0.10; {nd}; "
                     f"dashed curve), blue = E (equal-weight mean of the 3 curves < 0.10; {ne}; dotted), red = A1 (as A with >= 1 auditory; {na1}). Title: whisker trials removed by A-E. "
                     "Curves: whisker, FA grey, auditory blue; ticks above = licks, below = no licks (rows: auditory, "
                     "whisker, no stim).", fontsize=7, x=0.01, ha="left", y=0.998, va="top", wrap=True)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"026_disengagement_review_{tag}.{ext}", dpi=250)
        plt.close(fig)
    pd.set_option("display.width", 200)
    for k in ("a", "a1", "b", "c", "d", "e"):
        for rg, g in tab.groupby("reward_group"):
            f = g[g[f"{k}_flag"]]
            print(k.upper(), rg, len(f), "sessions, whisker trials removed median", f[f"{k}_whisker_removed"].median(),
                  "total", int(f[f"{k}_whisker_removed"].sum()))
    print(tab[tab.a_flag | tab.b_flag | tab.c_flag | tab.e_flag].sort_values(["reward_group", "b_whisker_removed"])
          [["session_id", "reward_group", "learning_category", "n_whisker", "a_whisker_removed", "b_whisker_removed",
            "c_whisker_removed", "d_whisker_removed", "e_whisker_removed",
            "a_tail_auditory"]].to_string(index=False))


if __name__ == "__main__":
    main()
