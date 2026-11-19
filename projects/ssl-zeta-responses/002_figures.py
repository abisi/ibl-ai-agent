"""002 -- summary figures and tables of the ZETA / latenZy results (all sessions, pooled over days and cohorts).

Input:  combined_results_ks4/ssl-zeta-responses/zeta_long.parquet (001) + per-session <mouse>_zeta_ifr.npz
Unit of analysis: session (fractions = mean over units within a session, then mean +- s.e.m. over sessions);
area-group values only where the group has >= MIN_UNITS units in >= MIN_SESS sessions.
Positive / negative neuron = ZETA p < 0.05 and direction positive / negative (sign of the largest IFR deviation from
the pre-event rate). Early dip = neg_peak flag of the pipeline. Latency = latenZy onset (NaN when latenZy finds none).
Outputs: ssl-zeta-responses/figures/*.{png,pdf,svg}, tables/*.csv
Run (haas): PYTHONPATH=~/code/ephys_utilities ~/code/unit_spikes_analysis/.venv/bin/python 002_figures.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-zeta-responses")
FIG, TAB = OUT / "figures", OUT / "tables"
MIN_UNITS, MIN_SESS = 5, 3
WH, AU, SL = "#f7b519", "#2c2cdb", "#262626"
ONE = ["whisker_passive_pre", "whisker_passive_post", "whisker_active", "whisker_miss_active", "auditory_passive_pre",
       "auditory_passive_post", "auditory_active", "auditory_miss_active", "spontaneous_licks"]
STIM4 = [("whisker_passive_pre", WH), ("whisker_active", WH), ("auditory_active", AU), ("spontaneous_licks", SL)]
LABEL = {"whisker_passive_pre": "whisker, passive pre", "whisker_passive_post": "whisker, passive post",
         "whisker_active": "whisker, active", "whisker_miss_active": "whisker miss, active",
         "auditory_passive_pre": "auditory, passive pre", "auditory_passive_post": "auditory, passive post",
         "auditory_active": "auditory, active", "auditory_miss_active": "auditory miss, active",
         "spontaneous_licks": "spontaneous licks"}
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.titlesize": 7})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def letters(fig, axes, labels):
    for a, l in zip(axes, labels):   # axes of one row share y1, so letters of a row are aligned
        p = a.get_position()
        fig.text(p.x0 - 0.05, p.y1 + 0.02, l, fontsize=9, fontweight="bold")


def sess_mean(df, col, by=("session_id",)):
    return df.groupby(list(by))[col].mean()


def mse(s):
    return s.mean(), s.sem()


def load():
    Z = pd.read_parquet(OUT / "zeta_long.parquet")
    one = Z.analysis_type.isin(ONE)
    Z["pos"] = Z.significant & np.where(one, Z.direction == "positive", Z.mean_rate_2 > Z.mean_rate_1)
    Z["neg"] = Z.significant & ~Z.pos
    Z["roc_sig"] = Z.roc_sig.astype("boolean")
    Z["neg_peak"] = Z.neg_peak.fillna(False).astype(bool)
    return Z


def table_types(Z):
    """per analysis type: sessions, units, ZETA / ROC responsive fractions (session means), overlap, sign agreement,
    latency coverage; Wilcoxon signed-rank ZETA vs ROC fraction over sessions"""
    rows = []
    order = ONE + sorted(set(Z.analysis_type) - set(ONE))
    for at in order:
        x = Z[Z.analysis_type == at]
        y = x[x.roc_sig.notna()].copy()
        y["roc_sig"] = y.roc_sig.astype(bool)
        fz, fr = sess_mean(y, "significant"), sess_mean(y, "roc_sig")
        fz_all = sess_mean(x, "significant").mean()
        both = y.significant & y.roc_sig
        agree = ((y.roc_sel > 0) == y.pos)[both].mean()
        p = stats.wilcoxon(fz, fr).pvalue if len(fz) > 5 and (fz != fr).any() else np.nan
        s = x[x.significant]
        rows.append(dict(analysis_type=at, sessions=x.session_id.nunique(), units=len(x),
                         zeta_frac=fz_all, roc_frac=fr.mean() if len(y) else np.nan, zeta_vs_roc_p=p,
                         both_over_either=both.sum() / (y.significant | y.roc_sig).sum() if len(y) else np.nan,
                         sign_agreement=agree, latency_cov_sig=s.latency_s.notna().mean(),
                         latency_cov_nonsig=x[~x.significant].latency_s.notna().mean(),
                         pos_frac=sess_mean(x, "pos").mean(), neg_frac=sess_mean(x, "neg").mean()))
    T = pd.DataFrame(rows)
    TAB.mkdir(exist_ok=True)
    T.to_csv(TAB / "analysis_types.csv", index=False)
    return T


def table_sizes(Z):
    x = Z[Z.analysis_type == "whisker_active"]
    T = (x.groupby("area_group").agg(sessions=("session_id", "nunique"), mice=("mouse_id", "nunique"),
                                     units=("cluster_id", "size")).reset_index())
    T.loc[len(T)] = ["all", x.session_id.nunique(), x.mouse_id.nunique(), len(x)]
    T.to_csv(TAB / "sample_sizes.csv", index=False)
    return T


def area_order(Z):
    from ephys_utilities.allen_utils import allen_utils as au
    x = Z[Z.analysis_type == "whisker_active"]
    n = x.groupby(["area_group", "session_id"]).size()
    ok = (n >= MIN_UNITS).groupby("area_group").sum()
    ok = set(ok[ok >= MIN_SESS].index)
    return [g for g in au.get_area_group_custom_order() if g in ok], au.get_custom_area_groups_colors()


def area_stat(Z, at, col, groups, by_value=None):
    x = Z[(Z.analysis_type == at) & Z.area_group.isin(groups)]
    if by_value is not None:
        x = x[by_value(x)]
    n = x.groupby(["area_group", "session_id"]).size()
    v = x.groupby(["area_group", "session_id"])[col].agg("median" if col == "latency_s" else "mean")
    v = v[(n >= (1 if col == "latency_s" else MIN_UNITS)).reindex(v.index)]
    return v.groupby("area_group").agg(["mean", "sem", "size"]).reindex(groups)


def ifr_means(Z):
    """mean IFR minus its pre-event mean (-180..-10 ms), positive and negative neurons, averaged per session then
    across sessions"""
    out = {}
    for at, _ in STIM4:
        x = Z[(Z.analysis_type == at) & (Z.pos | Z.neg)]
        acc = {"pos": [], "neg": []}
        t = None
        for src, g in x.groupby("src"):
            f = pathlib.Path(src) / f"{g.mouse_id.iloc[0]}_zeta_ifr.npz"
            if not f.exists():
                continue
            d = np.load(f)
            if f"{at}__ifr" not in d:
                continue
            ids, ifr, t = d[f"{at}__neuron_id"], d[f"{at}__ifr"], d[f"{at}__t_s"]
            idx = {n: i for i, n in enumerate(ids)}
            b = (t >= -0.180) & (t <= -0.010)
            for k in ("pos", "neg"):
                r = [idx[n] for n in g[g[k]].neuron_id if n in idx]
                if r:
                    m = ifr[r] - ifr[r][:, b].mean(1, keepdims=True)
                    acc[k].append(m.mean(0))
        out[at] = (t, {k: np.array(v) for k, v in acc.items()})
    return out


def fig_overview(Z, T, ifr):
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1], hspace=1.05, wspace=0.5)
    a = fig.add_subplot(gs[0, :])
    x = np.arange(len(T))
    a.bar(x - 0.2, T.zeta_frac, 0.4, color="0.25", lw=0, label="ZETA (p < 0.05)")
    a.bar(x + 0.2, T.roc_frac, 0.4, color="0.7", lw=0, label="ROC (permutation p < 0.05)")
    a.set_xticks(x, [LABEL.get(t, t).replace("_", " ") for t in T.analysis_type], rotation=55, ha="right", fontsize=5.5)
    a.set_ylabel("Fraction of neurons responsive\n(mean over sessions)")
    a.set_ylim(0, 1.02)
    a.axvline(len(ONE) - 0.5, color="0.6", lw=0.5, ls=":")
    a.text(len(ONE) / 2 - 0.5, 0.97, "one condition (event vs pre-event)", ha="center", fontsize=6)
    a.text(len(ONE) + (len(T) - len(ONE)) / 2 - 0.5, 0.97, "two conditions (ROC contrasts)", ha="center", fontsize=6)
    a.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.92))
    axes = [a]
    for j, (at, c) in enumerate(STIM4):
        b = fig.add_subplot(gs[1, j])
        t, acc = ifr[at]
        for k, ls in (("pos", "-"), ("neg", "--")):
            v = acc[k]
            if len(v):
                m = np.nanmean(v, 0)
                s = np.nanstd(v, 0) / np.sqrt(np.isfinite(v).sum(0).clip(1))
                b.plot(1000 * t, m, color=c, ls=ls, lw=1, label=f"{'positive' if k == 'pos' else 'negative'} (n = {len(v)})")
                b.fill_between(1000 * t, m - s, m + s, color=c, alpha=0.25, lw=0, edgecolor="none")
        b.axvline(0, color="0.5", lw=0.5, ls="--")
        b.axhline(0, color="0.5", lw=0.4)
        b.set_title(LABEL[at])
        b.set_xlabel("Time from " + ("lick" if at == "spontaneous_licks" else "stimulus") + " (ms)")
        b.set_ylabel("IFR − pre-event (Hz)")
        b.legend(frameon=False, fontsize=5, loc="best")
        axes.append(b)
    letters(fig, [axes[0], axes[1]], "AB")
    save(fig, "fig1_overview")


def fig_areas(Z, groups):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"wspace": 0.5})
    for a, col, cmap, title in ((axs[0], "pos", "Reds", "Positive neurons"),
                                (axs[1], "neg", "Blues", "Negative neurons")):
        M = pd.DataFrame({at: area_stat(Z, at, col, groups)["mean"] for at in ONE})
        M.to_csv(TAB / f"area_frac_{col}.csv")
        im = a.imshow(M.to_numpy(float), cmap=cmap, vmin=0, vmax=np.nanpercentile(M.to_numpy(float), 98), aspect="auto")
        a.set_yticks(range(len(groups)), groups, fontsize=5.5)
        a.set_xticks(range(len(ONE)), [LABEL[t] for t in ONE], rotation=55, ha="right", fontsize=5.5)
        a.set_title(title)
        cb = fig.colorbar(im, ax=a, shrink=0.7)
        cb.set_label("Fraction of neurons (mean over sessions)")
    letters(fig, axs, "AB")
    save(fig, "fig2_areas")


def fig_latency(Z, groups, pal):
    fig = plt.figure(figsize=(7.4, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.55)
    a = fig.add_subplot(gs[0])
    rows = []
    for k, (at, c) in enumerate(STIM4[:3]):
        S = area_stat(Z, at, "latency_s", groups, by_value=lambda x: x.pos & x.latency_s.notna())
        y = np.arange(len(groups)) + (k - 1) * 0.25
        ok = S["size"] >= MIN_SESS
        mk = "o" if "passive" in at else ("s" if "whisker" in at else "D")
        a.errorbar(1000 * S["mean"][ok], y[ok], xerr=1000 * S["sem"][ok], fmt=mk, ms=2.5, color=c, lw=0.6,
                   mfc="white" if "passive" in at else c, label=LABEL[at])
        rows.append(S.assign(analysis_type=at))
    pd.concat(rows).to_csv(TAB / "area_latency.csv")
    a.set_yticks(range(len(groups)), groups, fontsize=5.5)
    for tl, g in zip(a.get_yticklabels(), groups):
        tl.set_color(pal[g])
    a.invert_yaxis()
    a.set_xlabel("latenZy onset of positive neurons (ms)\n(session median, mean ± s.e.m. over sessions)")
    a.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    b = fig.add_subplot(gs[1])
    bins = np.arange(0, 201, 5)
    for at, c in STIM4[:3]:
        v = 1000 * Z[(Z.analysis_type == at) & Z.pos].latency_s.dropna()
        b.hist(v, bins, density=True, histtype="step", color=c, lw=0.9, ls=":" if "passive" in at else "-",
               label=f"{LABEL[at]} (n = {len(v)})")
    b.set_xlabel("latenZy onset (ms from stimulus)")
    b.set_ylabel("Density (positive neurons)")
    b.set_ylim(top=b.get_ylim()[1] * 1.35)
    b.legend(frameon=False, fontsize=5, loc="upper right")
    c_ = fig.add_subplot(gs[2])
    for k, ls in (("pos", "-"), ("neg", "--")):
        v = 1000 * Z[(Z.analysis_type == "spontaneous_licks") & Z[k]].latency_s.dropna()
        c_.hist(v, np.arange(-200, 201, 10), density=True, histtype="step", color=SL, ls=ls, lw=0.9,
                label=f"{'positive' if k == 'pos' else 'negative'} (n = {len(v)})")
    c_.axvline(0, color="0.5", lw=0.5, ls="--")
    c_.set_xlabel("latenZy onset (ms from lick)")
    c_.set_ylabel("Density (spontaneous-lick neurons)")
    c_.legend(frameon=False, fontsize=5)
    letters(fig, [a, b, c_], "ABC")
    save(fig, "fig3_latency")


def fig_dips(Z, groups, pal):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.45})
    a = axs[0]
    rows = []
    for k, (at, c) in enumerate(STIM4[:3]):
        S = area_stat(Z, at, "neg_peak", groups)
        y = np.arange(len(groups)) + (k - 1) * 0.25
        a.errorbar(S["mean"], y, xerr=S["sem"], fmt="o" if "passive" in at else "s", ms=2.5, color=c, lw=0.6,
                   mfc="white" if "passive" in at else c, label=LABEL[at])
        rows.append(S.assign(analysis_type=at))
    pd.concat(rows).to_csv(TAB / "area_dips.csv")
    a.set_yticks(range(len(groups)), groups, fontsize=5.5)
    for tl, g in zip(a.get_yticklabels(), groups):
        tl.set_color(pal[g])
    a.invert_yaxis()
    a.set_xlim(left=-0.002)
    a.set_xlabel("Fraction of neurons with an early dip\n(mean ± s.e.m. over sessions)")
    a.legend(frameon=False, fontsize=5, loc="upper right")
    b = axs[1]
    for at, c in STIM4[:3]:
        v = 1000 * Z[(Z.analysis_type == at) & Z.neg_peak].neg_peak_s.dropna()
        b.hist(v, np.arange(0, 101, 5), density=True, histtype="step", color=c, lw=0.9,
               ls=":" if "passive" in at else "-", label=f"{LABEL[at]} (n = {len(v)})")
    b.set_xlabel("Dip time (ms from stimulus)")
    b.set_ylabel("Density (early-dip neurons)")
    b.legend(frameon=False, fontsize=5)
    letters(fig, axs, "AB")
    save(fig, "fig4_dips")


def main():
    Z = load()
    T = table_types(Z)
    table_sizes(Z)
    groups, pal = area_order(Z)
    missing = [g for g in groups if g not in pal]
    if missing:
        raise KeyError(f"area groups missing from palette: {missing}")
    fig_overview(Z, T, ifr_means(Z))
    fig_areas(Z, groups)
    fig_latency(Z, groups, pal)
    fig_dips(Z, groups, pal)
    # numbers quoted in the report
    x = Z[Z.analysis_type.isin([a for a, _ in STIM4[:3]])]
    d = x[x.neg_peak]
    pd.Series(dict(
        dip_any_dir_pos=float((d.direction == "positive").mean()),
        dip_median_ms=float(1000 * d.neg_peak_s.median()),
        lat_sig_stim=float(x[x.pos].latency_s.notna().mean()),
        n_sessions=int(Z.session_id.nunique()), n_mice=int(Z.mouse_id.nunique()),
        n_units=int(Z.drop_duplicates(["session_id", "electrode_group", "cluster_id"]).shape[0]),
        **{f"lat_median_ms__{at}": float(1000 * Z[(Z.analysis_type == at) & Z.pos].latency_s.median())
           for at, _ in STIM4},
    )).to_json(TAB / "numbers.json", indent=1)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
