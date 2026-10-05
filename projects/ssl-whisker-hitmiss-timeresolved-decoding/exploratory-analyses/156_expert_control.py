"""156 -- Expert sessions as a control (user 2026-10-05: "Can you run control analyses on expert sessions?").
Same pipeline, same definitions, run with SSL_STAGE=expert (137b tracked units, 135 lick axis, 140 late coding direction, 146 decoder /
state space, 139 session halves; outputs *_expert.parquet). Logic: if the learning-session effects reflect the contingency being
learned (re-mapping during the session), the within-session passive pre -> post change should be smaller in expert sessions; if the
re-mapping persists across days, the R- whisker pattern should already be decoupled from the lick axis at expert passive pre.
Measures (per session; whole brain, shared tracked stable units):
  change     pre -> post excess over the linear-shift null: lick axis whisker raw cosine, auditory raw cosine, whisker - auditory
             projection (135); late coding direction passive-axis cosine and W - A projection (140, hit-median split); decoder W - A
             readout (146, per-trial baseline); state-space whisker displacement along the choice axis (146, epoch baseline);
             raw whisker cosine post - pre (135)
  level      passive pre whisker cosine with the lick axis (135)
  active     hit / miss decoding change between halves (139, 5-100 ms, midpoint and hit-median splits)
Tests (session = unit; expert mice can contribute several sessions, learning one): per group vs 0 (Wilcoxon | t); R+ vs R- within
stage (Mann-Whitney | Welch); learning vs expert within cohort (Mann-Whitney | Welch); cohort x stage interaction: OLS
M ~ R- * expert (parametric) and permutation of stage labels within cohort (non-parametric, difference of differences).
Figure: figures/publication/156_expert_control.{png,pdf,svg}; stats 156_stats.csv. Run (haas, repo root): python .../156_expert_control.py
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
STAGES = ("learning", "expert")
N_PERM = 10000


def load(stage):
    sfx = "" if stage == "learning" else "_expert"
    out = {}
    A = pd.read_parquet(EA / f"135_alignment_epochs_tracked{sfx}.parquet")
    A = A[(A.area == "All units") & A.skipped_reason.isna()].copy()
    A["rawW"] = A.evokedW_cos_passive_post - A.evokedW_cos_passive_pre
    out["135"] = A
    C = pd.read_parquet(EA / f"140_coding_direction_noise{sfx}.parquet")
    out["140"] = C[(C.split == "hitmedian") & C.skipped_reason.isna()]
    V = pd.read_parquet(EA / f"146_choice_axis_readout{sfx}.parquet")
    V = V[(V.area == "whole_brain") & (V.unit_set == "stable") & V.skipped_reason.isna()].copy()
    V["dx_W"] = V.ss_post_W_x - V.ss_pre_W_x
    out["146e"], out["146t"] = V[V.response == "epochbase"], V[V.response == "trialbase"]
    D = pd.read_parquet(EA / f"139_hitmedian_split_whole_brain{sfx}.parquet")
    D = D[D.skipped_reason.isna() & (D.decoding == "hitmiss") & (D.window == "5-100ms")].copy()
    D["dsep"] = D.sep_corr_2 - D.sep_corr_1
    out["139m"], out["139h"] = D[D.split == "mid"], D[D.split == "hitmedian"]
    return out


MEASURES = [  # (source, column, title, kind)
    ("135", "shift_excess_dWR", "lick axis: whisker cosine\n(excess)", "change"),
    ("135", "shift_excess_dAR", "lick axis: auditory cosine\n(excess)", "change"),
    ("135", "shift_excess_dWAP", "lick axis: W - A projection\n(excess)", "change"),
    ("135", "rawW", "lick axis: whisker cosine\n(raw post - pre)", "change"),
    ("135", "evokedW_cos_passive_pre", "lick axis: whisker cosine\nat passive pre (level)", "level"),
    ("140", "shift_excess_daxisR", "late CD: passive axis cosine\n(excess)", "change"),
    ("140", "shift_excess_dWAP", "late CD: W - A projection\n(excess)", "change"),
    ("146t", "shift_excess_dWA", "decoder: W - A readout\n(excess, per-trial baseline)", "change"),
    ("146e", "dx_W", "state space: whisker along\nthe choice axis (post - pre)", "change"),
    ("139m", "dsep", "active: hit / miss decoding,\nmidpoint halves (2nd - 1st)", "active"),
    ("139h", "dsep", "active: hit / miss decoding,\nhit-median halves (2nd - 1st)", "active"),
]


def interaction(v, rng):
    """v[(cohort, stage)] arrays -> (OLS interaction p, permutation p, difference of differences)"""
    import statsmodels.formula.api as smf
    d = pd.concat([pd.DataFrame(dict(y=v[(c, s)], Rm=int(c == "R-"), ex=int(s == "expert"))) for c in COH for s in STAGES])
    d = d[np.isfinite(d.y)]
    if d.groupby(["Rm", "ex"]).size().min() < 3:
        return np.nan, np.nan, np.nan
    p_ols = smf.ols("y ~ Rm * ex", d).fit().pvalues["Rm:ex"]
    dd = lambda g: (g[(g.Rm == 1) & (g.ex == 1)].y.mean() - g[(g.Rm == 1) & (g.ex == 0)].y.mean()) - \
        (g[(g.Rm == 0) & (g.ex == 1)].y.mean() - g[(g.Rm == 0) & (g.ex == 0)].y.mean())
    d0 = dd(d); null = np.empty(N_PERM)
    for i in range(N_PERM):
        g = d.copy()
        for r in (0, 1):
            m = g.Rm == r
            g.loc[m, "ex"] = rng.permutation(g.loc[m, "ex"].to_numpy())
        null[i] = dd(g)
    return p_ols, (np.sum(np.abs(null) >= abs(d0)) + 1) / (N_PERM + 1), d0


def panel(ax, v, title, rows, kind, col, rng):
    xs = {("R+", "learning"): 0, ("R-", "learning"): 1, ("R+", "expert"): 2.6, ("R-", "expert"): 3.6}
    for (c, s), x in xs.items():
        y = v[(c, s)]
        ax.plot(x + np.random.default_rng(int(x * 10)).uniform(-0.14, 0.14, len(y)), y, "o", ms=1.8, color=COL[c], alpha=0.4, mew=0)
        ax.errorbar(x, np.nanmean(y), H.sem(y), fmt="o", ms=3.6, color=COL[c], mfc=COL[c] if s == "learning" else "white", lw=1,
                    capsize=0, zorder=5)
        pw, pt, n = H.one_sample(y)
        rows.append(dict(measure=col, title=title.replace("\n", " "), kind=kind, test="vs 0 (Wilcoxon | t)", group=f"{c} {s}", n=n,
                         mean=np.nanmean(y), p_nonparam=pw, p_param=pt))
        if kind != "level":
            ax.text(x, 1.0, f"{H.pnum(pw)}\n{H.pnum(pt)}", color=COL[c], ha="center", va="bottom", fontsize=3.9,
                    transform=ax.get_xaxis_transform())
    for s, x0 in (("learning", 0.5), ("expert", 3.1)):
        mw, we = H.unpaired(v[("R+", s)], v[("R-", s)])
        rows.append(dict(measure=col, title=title.replace("\n", " "), kind=kind, test="R+ vs R- (Mann-Whitney | Welch)", group=s,
                         n=np.nan, mean=np.nanmean(v[("R-", s)]) - np.nanmean(v[("R+", s)]), p_nonparam=mw, p_param=we))
        ax.text(x0, 1.16, f"R+ vs R-\n{H.pnum(mw)} | {H.pnum(we)}", ha="center", va="bottom", fontsize=4.1, transform=ax.get_xaxis_transform())
    for c in COH:
        mw, we = H.unpaired(v[(c, "learning")], v[(c, "expert")])
        rows.append(dict(measure=col, title=title.replace("\n", " "), kind=kind, test="learning vs expert (Mann-Whitney | Welch)", group=c,
                         n=np.nan, mean=np.nanmean(v[(c, "expert")]) - np.nanmean(v[(c, "learning")]), p_nonparam=mw, p_param=we))
    po, pp_, d0 = interaction(v, rng)
    rows.append(dict(measure=col, title=title.replace("\n", " "), kind=kind, test="cohort x stage (permutation | OLS)", group="interaction",
                     n=np.nan, mean=d0, p_nonparam=pp_, p_param=po))
    ax.set_xlabel(f"cohort x stage {H.pnum(pp_)} | {H.pnum(po)}", fontsize=4.6, labelpad=9)
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(["R+", "R-", "R+", "R-"]); ax.set_xlim(-0.6, 4.2)
    for t, c in zip(ax.get_xticklabels(), ["R+", "R-", "R+", "R-"]):
        t.set_color(COL[c])
    ax.text(0.5, -0.17, "learning", ha="center", va="top", fontsize=4.6, transform=ax.get_xaxis_transform())
    ax.text(3.1, -0.17, "expert", ha="center", va="top", fontsize=4.6, transform=ax.get_xaxis_transform())
    ax.set_title(title, fontsize=5.4, pad=24)


def main():
    H.setup()
    rng = np.random.default_rng(0)
    data = {s: load(s) for s in STAGES}
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 7.0), gridspec_kw=dict(wspace=0.55, hspace=1.05))
    for ax, (src, col, title, kind) in zip(axes.flat, MEASURES):
        v = {(c, s): data[s][src][data[s][src].reward_group == c][col].to_numpy(float) for c in COH for s in STAGES}
        panel(ax, v, title, rows, kind, col, rng)
    axes.flat[-1].axis("off")
    n = {s: data[s]["135"].groupby("reward_group").session_id.nunique().to_dict() for s in STAGES}
    nm = {s: data[s]["135"].assign(m=H.mouse_of(data[s]["135"])).groupby("reward_group").m.nunique().to_dict() for s in STAGES}
    axes.flat[-1].text(0, 0.95, "Sessions (lick axis, 135):\n" + "\n".join(
        f"  {s}: R+ {n[s].get('R+', 0)} ({nm[s].get('R+', 0)} mice), R- {n[s].get('R-', 0)} ({nm[s].get('R-', 0)} mice)" for s in STAGES)
        + "\n\nfilled: learning, open: expert\nper group: p vs 0 (Wilcoxon / t)\nR+ vs R-: Mann-Whitney | Welch\n"
          "cohort x stage: permutation | OLS", va="top", fontsize=5, transform=axes.flat[-1].transAxes)
    fig.suptitle("Expert sessions as a control: the same within-session measures (whole brain, tracked stable units; dots = sessions)",
                 fontsize=7)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"156_expert_control.{ext}", dpi=300, bbox_inches="tight")
    R = pd.DataFrame(rows)
    R.to_csv(EA / "156_stats.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
    print(R[R.test != "vs 0 (Wilcoxon | t)"].round(4).to_string(index=False))
    print(R[R.test == "vs 0 (Wilcoxon | t)"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
