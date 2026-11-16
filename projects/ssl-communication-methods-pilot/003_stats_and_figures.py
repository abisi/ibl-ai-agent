"""Aggregate the communication pilot (002), test R+ vs R- per method, area pair, direction and condition, and plot
time-resolved, trial-start-aligned communication.

Per session: mean over draws. Communication = value minus the mean of its trial-shuffle null ("excess"), except the
control variants (within-area RRR, movement-regressed), which are shown raw next to the raw value of the main method.
Granger directionality: net time-reversed Granger (Haufe 2013; Winkler 2016) = (GC A->B - GC B->A) minus the same on
time-reversed data.
Statistics (exploratory pilot, n = 8 + 8 mice): per window, R+ vs R- Mann-Whitney and Welch (both reported, uncorrected);
across time, cluster-based permutation on Welch t (1,000 permutations of cohort labels, cluster-forming p < 0.05);
within cohort, communication > 0 (Wilcoxon) per window.
Outputs (combined_results_ks4/ssl-communication-methods-pilot/): pilot_long.parquet, session_level.parquet,
stats_per_window.csv, clusters.csv, figures/*.png, summary.md.
Run (haas): python projects/ssl-communication-methods-pilot/003_stats_and_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402

OUT = cl.OUT
FIG = OUT / "figures"
COLORS = {"R+": "#00B400", "R-": "#C800C8"}  # project rule (ssl_cross_area_rplus_rminus_colors)
COND_LABEL = {"W_hit": "Whisker hit", "W_miss": "Whisker miss", "A_hit": "Auditory hit", "NS_cr": "No-stim CR"}
SHORT = {"Motor and frontal areas": "MO/FRP", "SSp-w": "SSp-w", "Striatum and pallidum": "STR"}
COHORTS = ["R+", "R-"]


def load():
    files = sorted((OUT / "sessions").glob("*.parquet"))
    files = [f for f in files if not f.stem.endswith("_quick")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["excess"] = df.value - df.null_mean
    return df, [json.load(open(f.with_name(f.stem + "_meta.json"))) for f in files]


def session_level(df):
    keys = ["session_id", "mouse_id", "cohort", "method", "condition", "pair", "area_a", "area_b", "source", "target",
            "window_center"]
    agg = {"value": "mean", "excess": "mean", "null_mean": "mean"}
    if "value_reversed" in df:
        agg["value_reversed"] = "mean"
    s = df.groupby(keys, dropna=False).agg(agg).reset_index()
    # Granger net time-reversed directionality per pair (A -> B positive)
    g = s[s.method == "Granger"]
    ab = g[g.source == g.area_a].set_index(["session_id", "condition", "pair", "window_center"])
    ba = g[g.source == g.area_b].set_index(["session_id", "condition", "pair", "window_center"])
    j = ab.join(ba[["value", "value_reversed"]], rsuffix="_ba", how="inner")
    j["excess"] = (j.value - j.value_ba) - (j.value_reversed - j.value_reversed_ba)
    j = j.reset_index()[["session_id", "mouse_id", "cohort", "condition", "pair", "area_a", "area_b", "window_center",
                         "excess"]]
    j["method"], j["source"], j["target"] = "Granger_net_TRGC", j.area_a, j.area_b
    return pd.concat([s, j], ignore_index=True)


def cluster_test(wide_rp, wide_rm, n_perm=1000, alpha=0.05, seed=0):
    """Rows = mice, columns = windows. Clusters of contiguous windows with Welch p < alpha (same sign); null = largest
    |sum t| after permuting cohort labels."""
    data = np.vstack([wide_rp, wide_rm])
    lab = np.r_[np.zeros(len(wide_rp)), np.ones(len(wide_rm))]

    def clusters(labels):
        with np.errstate(all="ignore"):
            r = stats.ttest_ind(data[labels == 0], data[labels == 1], axis=0, equal_var=False, nan_policy="omit")
        t, p = np.asarray(r.statistic, float), np.nan_to_num(np.asarray(r.pvalue, float), nan=1)
        out, i = [], 0
        while i < len(t):
            if p[i] >= alpha:
                i += 1
                continue
            j = i
            while j + 1 < len(t) and p[j + 1] < alpha and np.sign(t[j + 1]) == np.sign(t[i]):
                j += 1
            out.append((i, j, float(np.nansum(t[i:j + 1]))))
            i = j + 1
        return out

    obs = clusters(lab)
    rng = np.random.default_rng(seed)
    null = np.array([max([abs(c[2]) for c in clusters(rng.permutation(lab))], default=0) for _ in range(n_perm)])
    return [(i, j, m, float((np.sum(null >= abs(m)) + 1) / (n_perm + 1))) for i, j, m in obs]


def tests(s):
    rows, crows = [], []
    for key, d in s.groupby(["method", "pair", "source", "target", "condition"]):
        w = d.pivot_table(index=["session_id", "cohort"], columns="window_center", values="excess")
        rp = w[w.index.get_level_values("cohort") == "R+"].to_numpy(float)
        rm = w[w.index.get_level_values("cohort") == "R-"].to_numpy(float)
        for k, wc in enumerate(w.columns):
            a, b = rp[:, k][np.isfinite(rp[:, k])], rm[:, k][np.isfinite(rm[:, k])]
            if len(a) < 3 or len(b) < 3:
                continue
            rows.append(dict(zip(["method", "pair", "source", "target", "condition"], key), window_center=wc,
                             n_rplus=len(a), n_rminus=len(b), mean_rplus=a.mean(), mean_rminus=b.mean(),
                             mannwhitney_p=stats.mannwhitneyu(a, b).pvalue,
                             welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue,
                             rplus_gt0_wilcoxon_p=stats.wilcoxon(a).pvalue if len(a) > 5 else np.nan,
                             rminus_gt0_wilcoxon_p=stats.wilcoxon(b).pvalue if len(b) > 5 else np.nan))
        if len(rp) >= 3 and len(rm) >= 3:
            for i, j, m, p in cluster_test(rp, rm):
                crows.append(dict(zip(["method", "pair", "source", "target", "condition"], key),
                                  start=w.columns[i], end=w.columns[j], mass=m, p=p))
    return pd.DataFrame(rows), pd.DataFrame(crows)


def plot_method(s, clusters, method, ylabel, fname, directional=True):
    pairs = sorted(s.pair.unique())
    conds = [c for c in COND_LABEL if c in s.condition.unique()]
    fig, axs = plt.subplots(len(pairs), len(conds), figsize=(3.0 * len(conds), 2.2 * len(pairs)), sharex=True,
                            squeeze=False)
    d0 = s[s.method == method]
    for i, pair in enumerate(pairs):
        for j, cond in enumerate(conds):
            ax = axs[i, j]
            d = d0[(d0.pair == pair) & (d0.condition == cond)]
            dirs = sorted(d.source.unique())
            for k, src in enumerate(dirs):
                dd = d[d.source == src]
                tgt = dd.target.iloc[0]
                for cohort in COHORTS:
                    g = dd[dd.cohort == cohort].groupby("window_center").excess
                    m, e = g.mean(), g.sem()
                    ls = "-" if k == 0 else "--"
                    ax.plot(m.index, m, color=COLORS[cohort], ls=ls, lw=1,
                            label=f"{cohort} {SHORT[src]}->{SHORT[tgt]}" if directional else cohort)
                    ax.fill_between(m.index, m - e, m + e, color=COLORS[cohort], alpha=0.15, lw=0)
                cl_ = clusters[(clusters.method == method) & (clusters.pair == pair) & (clusters.condition == cond)
                               & (clusters.source == src) & (clusters.p < 0.05)] if len(clusters) else []
                for c in (cl_.itertuples() if len(cl_) else []):
                    ax.plot([c.start, c.end], [0.97 - 0.05 * k] * 2, transform=ax.get_xaxis_transform(), color="k",
                            lw=2, ls=ls, solid_capstyle="butt")
            ax.axhline(0, color="0.6", lw=0.5, ls=":")
            ax.axvline(0, color="0.6", lw=0.5, ls=":")
            if i == 0:
                ax.set_title(COND_LABEL[cond], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{SHORT[pair.split(' | ')[0]]} - {SHORT[pair.split(' | ')[1]]}\n{ylabel}", fontsize=8)
            if i == len(pairs) - 1:
                ax.set_xlabel("Time from trial start (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == len(conds) - 1:
                ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.suptitle(f"{method}: communication minus trial-shuffle null (mean +- s.e.m. over mice; bars: R+ vs R- clusters, "
                 "p < 0.05)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def plot_controls(s, metas):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    # 1 within-area ceiling vs between, post-stimulus window
    post = (s.window_center >= 0) & (s.window_center <= 0.3)
    for k, (meth, lab) in enumerate([("RRR", "between"), ("RRR_within", "within")]):
        v = s[(s.method == meth) & post].groupby(["session_id", "cohort"]).value.mean().reset_index()
        for c in COHORTS:
            x = v[v.cohort == c].value
            axs[0].scatter(np.full(len(x), k + (0.15 if c == "R-" else -0.15)), x, color=COLORS[c], s=10)
    axs[0].set_xticks([0, 1], ["Between areas", "Within area"])
    axs[0].set_ylabel("RRR CV performance (0-0.3 s)")
    axs[0].set_title("Within-area ceiling", fontsize=9)
    # 2 movement regression: RRR and CCA raw vs movement-regressed
    for meth, mv, mk in [("RRR", "RRR_mvt_regressed", "o"), ("PCA_CCA", "PCA_CCA_mvt_regressed", "s")]:
        a = s[(s.method == meth) & post].groupby(["session_id", "cohort", "condition"]).value.mean()
        b = s[(s.method == mv) & post].groupby(["session_id", "cohort", "condition"]).value.mean()
        j = pd.concat([a, b], axis=1, keys=["raw", "mvt"]).dropna().reset_index()
        for c in COHORTS:
            jj = j[j.cohort == c]
            axs[1].scatter(jj.raw, jj.mvt, color=COLORS[c], marker=mk, s=10, label=f"{c} {meth}")
    lim = axs[1].get_xlim()
    axs[1].plot(lim, lim, color="0.6", lw=0.5, ls=":")
    axs[1].set_xlabel("Raw (0-0.3 s)")
    axs[1].set_ylabel("Movement-regressed")
    axs[1].set_title("Jaw + whisker movement regressed out", fontsize=9)
    axs[1].legend(fontsize=6, frameon=False)
    # 3 firing rates per area and cohort (units used)
    for k, area in enumerate(cl.CONFIG["areas"]):
        for c in COHORTS:
            r = [m["mean_rate_hz_by_area"][area] for m in metas if m["cohort"] == c and m["mean_rate_hz_by_area"][area]]
            axs[2].scatter(np.full(len(r), k + (0.15 if c == "R-" else -0.15)), r, color=COLORS[c], s=10)
    axs[2].set_xticks(range(3), [SHORT[a] for a in cl.CONFIG["areas"]])
    axs[2].set_ylabel("Mean rate of units (Hz)")
    axs[2].set_title("Firing rates (confound check)", fontsize=9)
    # 4 trials available per condition and cohort
    for k, cond in enumerate(COND_LABEL):
        for c in COHORTS:
            n = [m["n_trials_by_condition"].get(cond, 0) for m in metas if m["cohort"] == c]
            axs[3].scatter(np.full(len(n), k + (0.15 if c == "R-" else -0.15)), n, color=COLORS[c], s=10)
    axs[3].axhline(cl.CONFIG["n_trials"], color="0.4", lw=0.6, ls="--")
    axs[3].set_xticks(range(4), [COND_LABEL[c] for c in COND_LABEL], rotation=20, fontsize=7)
    axs[3].set_ylabel("Trials available (dashed: used)")
    axs[3].set_title("Trial counts", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "controls.png", dpi=200)
    plt.close(fig)


def plot_agreement(s):
    """Per session, correlation across (pair, condition, window) between methods' communication."""
    m = s[s.method.isin(["RRR", "PCA_CCA", "Granger"])].copy()
    m = m.groupby(["session_id", "cohort", "method", "pair", "condition", "window_center"]).excess.mean().unstack("method")
    names = ["RRR", "PCA_CCA", "Granger"]
    per = []
    for sid, d in m.groupby(level="session_id"):
        c = d[names].corr(method="spearman")
        per.append(c.to_numpy())
    mean = np.nanmean(per, 0)
    fig, ax = plt.subplots(figsize=(3.4, 3))
    im = ax.imshow(mean, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r (mean over sessions)")
    ax.set_title("Method agreement\n(pairs x conditions x windows)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "method_agreement.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(mean, index=names, columns=names)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, metas = load()
    df.to_parquet(OUT / "pilot_long.parquet")
    s = session_level(df)
    s.to_parquet(OUT / "session_level.parquet")
    per_window, clusters = tests(s)
    per_window.to_csv(OUT / "stats_per_window.csv", index=False)
    clusters.to_csv(OUT / "clusters.csv", index=False)
    plot_method(s, clusters, "RRR", "RRR excess (CV 1-NSE)", "rrr_time_resolved.png")
    plot_method(s, clusters, "PCA_CCA", "CCA excess (CV cc1)", "pca_cca_time_resolved.png", directional=False)
    plot_method(s, clusters, "Granger", "GC excess (nats)", "granger_time_resolved.png")
    plot_method(s, clusters, "Granger_net_TRGC", "net TRGC (A->B +)", "granger_net_trgc_time_resolved.png",
                directional=False)
    plot_controls(s, metas)
    agree = plot_agreement(s)
    sig = clusters[clusters.p < 0.05] if len(clusters) else clusters
    fence = "`" * 3

    def block(text):
        return "\n".join([fence, text, fence])

    lines = ["# Inter-areal communication methods pilot (generated)", "",
             f"- Sessions: {s.session_id.nunique()} ({', '.join(f'{c}: {s[s.cohort == c].session_id.nunique()}' for c in COHORTS)}).",
             f"- Rows: {len(df)}; methods: {sorted(df.method.unique())}.",
             "", "## Cohort clusters (R+ vs R-, p < 0.05)", "",
             block(sig.round(4).to_string(index=False)) if len(sig) else "None.", "",
             "## Method agreement (Spearman, mean over sessions)", "", block(agree.round(2).to_string()), ""]
    post = s[(s.window_center >= 0) & (s.window_center <= 0.3)]
    post = post.assign(val=post.excess.fillna(post.value))  # control variants have no null: raw value
    per_mouse = post.groupby(["method", "condition", "cohort", "session_id"]).val.mean().reset_index()
    tab = (per_mouse.groupby(["method", "condition", "cohort"]).val.agg(["mean", "sem", "count"])
           .unstack("cohort").round(4))  # s.e.m. over mice
    lines += ["## Mean communication 0-0.3 s after trial start (excess over null; control variants raw), mean, s.e.m. and n over mice", "", block(tab.to_string()), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
