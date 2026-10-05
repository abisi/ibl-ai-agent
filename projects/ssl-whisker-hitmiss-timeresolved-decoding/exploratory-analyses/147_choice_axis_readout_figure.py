"""147 -- Figure for 146 (choice-axis readout of sensory responses across passive pre / active / passive post).
Per unit set (stable, good) one figure 147_choice_axis_readout_<unitset>[_pilot]:
  a  state space (R+, R-): condition means (evoked 5-35 ms patterns) on the plane of the choice axis (hit - miss, active) and the
     passive-pre whisker pattern orthogonalised to it; pre -> active -> post arrows for whisker trials, auditory dashed
  b  standardised choice readout per epoch (choice decoder trained on active whisker trials; passive / auditory trials scored by
     models that never saw them): whisker solid, auditory dashed; active hits (^) and misses (v) for reference
  c  whisker - auditory readout per epoch (removes shifts common to both stimuli, e.g. slow drift)
  d  drift control: readout from a decoder trained on the first active half only
  e-g  whisker response along the mean-difference choice axis (unit length): size (norm / sqrt(units)), cosine, projection /
       sqrt(units) (= size x cosine)
  h  null: readout from decoders trained on shuffled hit / miss labels
Mean +- s.e.m. over sessions per cohort; faint lines = sessions. Stats: within cohort post - pre (Wilcoxon | paired t), cohort
difference of post - pre (Mann-Whitney | Welch). Uncorrected. Style: skills/ssl-figure-style.
Run (haas): python .../147_choice_axis_readout_figure.py [pilot]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
EP = ["passive_pre", "active_1", "active_2", "passive_post"]
EPL = ["passive\npre", "active\n1st half", "active\n2nd half", "passive\npost"]


def epoch_lines(ax, d, cols, ls="-", marker="o", faint=True, rows=None, tag=""):
    x = np.arange(len(cols))
    for c in COH:
        g = d[d.reward_group == c]
        M = g[cols].to_numpy(float)
        if faint:
            for m in M:
                ax.plot(x, m, color=COL[c], lw=0.4, alpha=0.3, ls=ls)
        mu = np.nanmean(M, 0); se = np.array([H.sem(M[:, j]) for j in range(M.shape[1])])
        ax.errorbar(x + (0.05 if c == "R-" else -0.05), mu, se, color=COL[c], lw=1.1, ls=ls, marker=marker, ms=3.2, capsize=0,
                    mfc=COL[c] if ls == "-" else "white", mew=0.8)
        if rows is not None and len(cols) >= 2:
            pw, pt, n = H.paired(M[:, -1], M[:, 0])
            rows.append(dict(measure=tag, test="post vs pre (Wilcoxon | paired t)", cohort=c, n=n, mean_a=np.nanmean(M[:, 0]),
                             mean_b=np.nanmean(M[:, -1]), p_nonparam=pw, p_param=pt))
    if rows is not None:
        chg = {c: (d[d.reward_group == c][cols[-1]] - d[d.reward_group == c][cols[0]]).to_numpy(float) for c in COH}
        mw, we = H.unpaired(chg["R+"], chg["R-"])
        rows.append(dict(measure=tag, test="post - pre, R+ vs R- (Mann-Whitney | Welch)", cohort="R+ vs R-", n=np.nan,
                         mean_a=np.nanmean(chg["R+"]), mean_b=np.nanmean(chg["R-"]), p_nonparam=mw, p_param=we))
        return mw, we
    return None


def main():
    pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    H.setup()
    D = pd.read_parquet(EA / ("146_choice_axis_readout_pilot.parquet" if pilot else "146_choice_axis_readout.parquet"))
    D = D[D.skipped_reason.isna()]
    DA = D.copy()
    if "area" in D:
        D = D[D.area == "whole_brain"]
    allrows = []
    for uset in ("stable", "good"):
        d = D[D.unit_set == uset].copy()
        rows = []
        fig = plt.figure(figsize=(7.4, 6.4))
        W, Hh = fig.get_size_inches()
        # a state space
        for k, c in enumerate(COH):
            ax = fig.add_axes([0.07 + k * 0.25, 0.66, 0.19, 0.25])
            g = d[d.reward_group == c]
            pts = {}
            for cn in ["pre_W", "pre_A", "act1_hit", "act1_miss", "act1_A", "act2_hit", "act2_miss", "act2_A", "post_W", "post_A"]:
                pts[cn] = (g[f"ss_{cn}_x"].mean(), g[f"ss_{cn}_y"].mean())
            for cn, (x, y) in pts.items():
                col = "#f7b519" if cn.endswith("W") else "#2c2cdb" if cn.endswith("A") else ("0.15" if "hit" in cn else "0.6")
                mk = "^" if "hit" in cn else "v" if "miss" in cn else "o" if cn.startswith("pre") else "s" if cn.startswith("post") else "D"
                ax.plot(x, y, mk, color=col, ms=4.5 if cn[:3] in ("pre", "pos") else 3.5, mfc=col if not cn.startswith("pre") else "white", mew=1)
            ax.annotate("", xy=pts["post_W"], xytext=pts["pre_W"], arrowprops=dict(arrowstyle="->", color="#f7b519", lw=1))
            ax.annotate("", xy=pts["post_A"], xytext=pts["pre_A"], arrowprops=dict(arrowstyle="->", color="#2c2cdb", lw=0.8, ls="--"))
            ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
            ax.set_xlabel("choice axis (hit - miss)"); ax.set_ylabel("whisker response axis\n(orthogonal)" if k == 0 else "")
            ax.set_title(f"{c} (n = {len(g)})", color=COL[c])
        axl = fig.add_axes([0.58, 0.66, 0.4, 0.25]); axl.axis("off")
        leg = [("o", "#f7b519", "white", "passive pre, whisker"), ("s", "#f7b519", "#f7b519", "passive post, whisker"),
               ("o", "#2c2cdb", "white", "passive pre, auditory"), ("s", "#2c2cdb", "#2c2cdb", "passive post, auditory"),
               ("^", "0.15", "0.15", "active hit"), ("v", "0.6", "0.6", "active miss"), ("D", "#2c2cdb", "#2c2cdb", "active auditory")]
        for i, (mk, c_, f_, t) in enumerate(leg):
            axl.plot(0.02, 0.92 - i * 0.13, mk, color=c_, mfc=f_, ms=4.5, mew=1, transform=axl.transAxes)
            axl.text(0.07, 0.92 - i * 0.13, t, va="center", fontsize=5.5, transform=axl.transAxes)
        axl.text(0.55, 0.92, "mean over sessions of the\ncondition means (evoked\n5-35 ms, z units);\narrows: pre -> post",
                 va="top", fontsize=5.5, transform=axl.transAxes)
        # b readout W / A + active hit / miss
        ax = fig.add_axes([0.07, 0.36, 0.19, 0.2])
        wc = [f"ro_std_{e}_W" for e in EP]; ac = [f"ro_std_{e}_A" for e in EP]
        mwb = epoch_lines(ax, d, wc, "-", rows=rows, tag="readout whisker")
        epoch_lines(ax, d, ac, "--", faint=False, rows=rows, tag="readout auditory")
        for c in COH:
            g = d[d.reward_group == c]
            for j, h in ((1, 1), (2, 2)):
                ax.plot(j, g[f"ro_std_active_{h}_hit"].mean(), "^", color=COL[c], ms=3, alpha=0.6)
                ax.plot(j, g[f"ro_std_active_{h}_miss"].mean(), "v", color=COL[c], ms=3, alpha=0.6)
        ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(range(4)); ax.set_xticklabels(EPL, fontsize=5); ax.set_ylabel("choice readout\n(+ hit-like, - miss-like; SD units)")
        ax.set_title(f"b  readout (whisker solid, auditory dashed)\npost - pre, R+ vs R-: {H.pnum(mwb[0])} | {H.pnum(mwb[1])}", fontsize=5.5)
        # c W - A
        ax = fig.add_axes([0.32, 0.36, 0.19, 0.2])
        for e in EP:
            d[f"wa_{e}"] = d[f"ro_std_{e}_W"] - d[f"ro_std_{e}_A"]
        m2 = epoch_lines(ax, d, [f"wa_{e}" for e in EP], rows=rows, tag="readout whisker - auditory")
        ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        lo, hi = np.nanpercentile(d[[f"wa_{e}" for e in EP]].to_numpy(float), [2, 98])
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))          # axis range from the 2-98 % of session values
        ax.set_xticks(range(4)); ax.set_xticklabels(EPL, fontsize=5); ax.set_ylabel("whisker - auditory readout")
        ax.set_title(f"c  whisker - auditory\npost - pre, R+ vs R-: {H.pnum(m2[0])} | {H.pnum(m2[1])}", fontsize=5.5)
        # d first-half decoder
        ax = fig.add_axes([0.57, 0.36, 0.19, 0.2])
        cols_d = ["ro_h1dec_passive_pre_W", "ro_h1dec_active_2_W", "ro_h1dec_passive_post_W"]
        if all(c in d for c in cols_d):
            m3 = epoch_lines(ax, d.dropna(subset=cols_d), cols_d, rows=rows, tag="readout whisker, decoder from active half 1")
            epoch_lines(ax, d.dropna(subset=cols_d), ["ro_h1dec_passive_pre_A", "ro_h1dec_passive_post_A"], "--", faint=False)
            ax.set_title(f"d  decoder from active 1st half only\npost - pre, R+ vs R-: {H.pnum(m3[0])} | {H.pnum(m3[1])}", fontsize=5.5)
        ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(range(3)); ax.set_xticklabels(["passive\npre", "active\n2nd half", "passive\npost"], fontsize=5)
        ax.set_ylabel("choice readout (SD units)")
        # h null
        ax = fig.add_axes([0.81, 0.36, 0.17, 0.2])
        epoch_lines(ax, d, ["null_std_passive_pre_W", "null_std_passive_post_W"], rows=rows, tag="null readout whisker")
        ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["passive\npre", "passive\npost"], fontsize=5); ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("readout, shuffled labels"); ax.set_title("h  null (shuffled hit / miss)", fontsize=5.5)
        # e-g decomposition; the projection is divided by sqrt(n units) like the size, so projection = size x cosine
        for e in EP:
            for s_ in ("W", "A"):
                d[f"projn_{e}_{s_}"] = d[f"proj_{e}_{s_}"] / np.sqrt(d.n_units)
        for j, (name, lab) in enumerate((("size", "size of whisker response\n(norm / sqrt(units))"), ("cos", "cosine of whisker response\nwith choice axis"),
                                          ("projn", "projection on choice axis\n(per unit; = size x cosine)"))):
            ax = fig.add_axes([0.07 + j * 0.25, 0.07, 0.19, 0.2])
            m4 = epoch_lines(ax, d, [f"{name}_{e}_W" for e in EP], rows=rows, tag=f"{name} whisker")
            epoch_lines(ax, d, [f"{name}_{e}_A" for e in EP], "--", faint=False)
            ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
            ax.set_xticks(range(4)); ax.set_xticklabels(EPL, fontsize=5); ax.set_ylabel(lab)
            ax.set_title(f"{'efg'[j]}  post - pre, R+ vs R-: {H.pnum(m4[0])} | {H.pnum(m4[1])}", fontsize=5.5)
        ax = fig.add_axes([0.81, 0.07, 0.17, 0.2])
        for k, c in enumerate(COH):
            g = d[d.reward_group == c]
            for jj, col in enumerate(("decoder_bal_acc", "cd_reliability")):
                ax.errorbar(jj + (k - 0.5) * 0.3, g[col].mean(), H.sem(g[col]), fmt="o", color=COL[c], ms=3.2, capsize=0, lw=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["decoder\nbal. acc.", "axis\nreliability"], fontsize=5); ax.set_xlim(-0.5, 1.5)
        ax.axhline(0.5, color="0.7", lw=0.4); ax.set_title("decoder quality", fontsize=5.5)
        fig.text(0.07 - 0.4 / W, 0.95, "a", fontsize=9, weight="bold")
        fig.suptitle(f"Choice-axis readout of 5-35 ms sensory responses, {uset} units{' (PILOT, n small)' if pilot else ''}", fontsize=6.5, y=0.995)
        name = f"147_choice_axis_readout_{uset}{'_pilot' if pilot else ''}"
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        R = pd.DataFrame(rows); R.insert(0, "unit_set", uset); allrows.append(R)
    # area groups: post - pre change of the whisker readout and of whisker - auditory, per cohort (stable units)
    if "area" in DA and DA.area.nunique() > 1:
        a = DA[(DA.unit_set == "stable") & (DA.area != "whole_brain")].copy()
        a["dW"] = a.ro_std_passive_post_W - a.ro_std_passive_pre_W
        a["dWA"] = (a.ro_std_passive_post_W - a.ro_std_passive_post_A) - (a.ro_std_passive_pre_W - a.ro_std_passive_pre_A)
        areas = [x for x, g in a.groupby("area") if min((g.reward_group == "R+").sum(), (g.reward_group == "R-").sum()) >= 3]
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.25, hspace=0.35)
        arows = []
        for ax, (col, lab) in zip(axes, (("dW", "whisker readout\npost - pre (SD units)"), ("dWA", "whisker - auditory readout\npost - pre"))):
            for i, ar in enumerate(areas):
                v = {}
                for k, c in enumerate(COH):
                    v[c] = a[(a.area == ar) & (a.reward_group == c)][col].to_numpy(float)
                    ax.errorbar(i + (k - 0.5) * 0.3, np.nanmean(v[c]), H.sem(v[c]), fmt="o", ms=3.2, color=COL[c], lw=1, capsize=0)
                mw, we = H.unpaired(v["R+"], v["R-"])
                arows.append(dict(measure=col, area=ar, n_rplus=len(v["R+"]), n_rminus=len(v["R-"]), mean_rplus=np.nanmean(v["R+"]),
                                  mean_rminus=np.nanmean(v["R-"]), p_nonparam=mw, p_param=we))
                if np.isfinite(mw) and min(mw, we) < 0.05:
                    ax.text(i, 1.0, f"{H.pnum(mw)}|{H.pnum(we)}", ha="center", fontsize=4.5, transform=ax.get_xaxis_transform())
            ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2))); ax.set_ylabel(lab)
        axes[1].set_xticks(range(len(areas))); axes[1].set_xticklabels(areas, rotation=60, ha="right", fontsize=5)
        fig.suptitle("Choice readout of passive whisker responses, post - pre, per area group (stable units; mean +- s.e.m. over "
                     "sessions; p shown where Mann-Whitney or Welch < 0.05, uncorrected)", fontsize=5.8)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIGDIR / f"147_choice_axis_readout_area_groups.{ext}", dpi=300)
        plt.close(fig)
        pd.DataFrame(arows).to_csv(EA / "147_stats_area_groups.csv", index=False)
    # state space with the whisker - auditory (stimulus identity) axis as y, both unit sets, whole brain
    if "ss2_pre_W_y" in D:
        fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.6))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.1, hspace=0.45, wspace=0.3)
        for i, uset in enumerate(("stable", "good")):
            d = D[D.unit_set == uset]
            for k, c in enumerate(COH):
                ax = axes[i, k]; g = d[d.reward_group == c]
                pts = {cn: (g[f"ss_{cn}_x"].mean(), g[f"ss2_{cn}_y"].mean()) for cn in
                       ["pre_W", "pre_A", "act1_hit", "act1_miss", "act1_A", "act2_hit", "act2_miss", "act2_A", "post_W", "post_A"]}
                for cn, (x, y) in pts.items():
                    col = "#f7b519" if cn.endswith("W") else "#2c2cdb" if cn.endswith("A") else ("0.15" if "hit" in cn else "0.6")
                    mk = "^" if "hit" in cn else "v" if "miss" in cn else "o" if cn.startswith("pre") else "s" if cn.startswith("post") else "D"
                    ax.plot(x, y, mk, color=col, ms=4, mfc=col if not cn.startswith("pre") else "white", mew=1)
                ax.annotate("", xy=pts["post_W"], xytext=pts["pre_W"], arrowprops=dict(arrowstyle="->", color="#f7b519", lw=1))
                ax.annotate("", xy=pts["post_A"], xytext=pts["pre_A"], arrowprops=dict(arrowstyle="->", color="#2c2cdb", lw=0.8))
                ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
                ax.set_title(f"{c}, {uset} units (n = {len(g)})", color=COL[c], fontsize=6)
                ax.set_xlabel("choice axis (hit - miss)")
                if k == 0:
                    ax.set_ylabel("whisker - auditory axis\n(passive pre, orthogonalised)")
        fig.suptitle("State space with the stimulus-identity axis: o passive pre, s passive post (whisker yellow, auditory blue), "
                     "^ hit, v miss, D active auditory", fontsize=5.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIGDIR / f"147_state_space_whisker_auditory_axis{'_pilot' if pilot else ''}.{ext}", dpi=300)
        plt.close(fig)
    S = pd.concat(allrows)
    S.to_csv(EA / f"147_stats{'_pilot' if pilot else ''}.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
