"""Decision "go" components, exploratory (user 2026-11-23: re-evaluate the criteria, then look at the components; jaw-aligned 10-ms tensor over
all trials and stimulus tensor with the no-stimulus trials). Exploratory, uncorrected.

Fits: jaw_all (jaw onset -0.3..+0.3 s, 10-ms bins, all trials; licks and jaw movements without a lick at their own
onset, still trials at the type-matched median lick onset) and stim_long_ns (stimulus -0.2..1.5 s, 50-ms bins, all
trials including no-stimulus trials), main and reference fits; recurring clusters from 013 run with TCA_SUFFIX=_go.
Three mice without a jaw-onset table (AB080, AB082, MH026) are excluded from jaw_all.

Trial classes (jaw_class from 005): lick W / A / FA (whisker hit, auditory hit, false alarm = lick on a no-stimulus
trial); move (active, jaw movement 15-600 ms without lick); P-move (passive, jaw movement); still (active, no jaw onset
0-1 s: whisker / auditory misses, correct rejections); P-still. Weights in SD units of the active trials, 0 = mean of
the active still trials.

Quantities per cluster (medians over mice, Wilcoxon over mice; no pass / fail, user 2026-11-23: exploratory, look at
the components; reference values in brackets are what a go component would show):
  AUC lick vs active jaw move without lick (> 0.5: lick-specific), AUC passive move vs passive still (~0.5: not motor),
  AUC false alarm vs correct rejection and FA / hit weight (> 0.5 / ~1: stimulus-independent), R- values (unrewarded
  licks), onset of the time factor (jaw_all: < 0 = before movement) and pre-movement AUC (projection on bins
  -300..-20 ms) lick vs move over a unit-shuffle null (> 0), weight vs lick latency (supporting only).
stim_long_ns: the same classes, with onset relative to the stimulus and the median jaw onset (A, B, D, E; C via the
pre-movement projection on bins ending 25 ms before each lick's jaw onset vs. moves).

Run (haas): python 022_go_signal_v2.py [--n-jobs 40] [--reuse]
     -> tables/go2_members.parquet, tables/go2_clusters.csv, figures/aligned_gpu/go2/go2_overview.png|pdf,
        go2/go2_gallery_<alignment>_<fit>_p<n>.png|pdf (one row per recurring cluster)
"""
import argparse
import importlib
import json
import os
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

C12 = importlib.import_module("012_component_table")
A5 = importlib.import_module("005_build_aligned")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
FITN = {"_splitnn": "main fit", "_nn": "reference fit"}
ALS = ["jaw_all", "stim_long_ns"]
NO_JAW = ("AB080", "AB082", "MH026")
CLS = ["lick_W", "lick_A", "lick_FA", "move", "still_W", "still_A", "still_CR", "P_move", "P_still"]
CLS_LAB = ["lick\nW", "lick\nA", "lick\nFA", "move", "still\nW miss", "still\nA miss", "still\nCR", "P\nmove", "P\nstill"]
PRE_END = -0.020


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def classes(t):
    ep, ty, jc = t.epoch.to_numpy(), t.trial_type.to_numpy(), t.jaw_class.to_numpy()
    act, pas = ep == "active", ep != "active"
    c = np.full(len(t), "", object)
    for k, m in (("W", "whisker_trial"), ("A", "auditory_trial"), ("FA", "no_stim_trial")):
        c[act & (jc == "lick") & (ty == m)] = f"lick_{k}"
    c[act & (jc == "move")] = "move"
    for k, m in (("W", "whisker_trial"), ("A", "auditory_trial"), ("CR", "no_stim_trial")):
        c[act & (jc == "still") & (ty == m)] = f"still_{k}"
    c[pas & (jc == "move")] = "P_move"
    c[pas & (jc == "still")] = "P_still"
    return c


def session_rows(sid, rows):
    out = []
    t0 = pd.read_parquet(A5.OUT_DIR / sid / "trials.parquet").set_index("trial_index")
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    cache = {}
    for r in rows:
        al, meth = r["alignment"], r["method"]
        if al == "jaw_all" and r["mouse_id"] in NO_JAW:
            continue
        if al not in cache:
            d = np.load(A5.OUT_DIR / sid / f"{al}.npz")
            R, base = d["rate"], d["baseline"]
            keep = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= 0.5
            X = R[:, keep] - base[:, keep, None]
            X = np.nan_to_num(X / (np.nanstd(X, (0, 2), keepdims=True) + 1e-9))
            cache[al] = (d, X, t0.loc[d["trial_index"]].reset_index())
        d, X, t = cache[al]
        f = np.load(C12.D / sid / f"{al}_zscore{meth}.npz")
        k, rk = int(r["comp"]) - 1, int(r["rank"])
        w = f[f"r{rk}_f0"][:, k].astype(float)
        T = f[f"r{rk}_f2"][:, k]
        J = len(f["units_kept"])
        Fu = f[f"r{rk}_f1"][:, k]
        net = Fu[:J] - Fu[J:] if Fu.shape[0] == 2 * J else Fu
        c = classes(t)
        act = (t.epoch == "active").to_numpy()
        still_a = np.isin(c, ["still_W", "still_A", "still_CR"])
        sd = np.std(w[act])
        if sd <= 0 or still_a.sum() < 3:
            continue
        z = (w - w[still_a].mean()) / sd
        res = {q: r[q] for q in ("session_id", "mouse_id", "cohort", "learner", "method", "alignment", "cluster", "comp")}
        for q in CLS:
            res[f"m_{q}"] = float(np.mean(z[c == q])) if (c == q).sum() >= 3 else np.nan
            res[f"n_{q}"] = int((c == q).sum())
        lick = np.isin(c, ["lick_W", "lick_A", "lick_FA"])
        hit = np.isin(c, ["lick_W", "lick_A"])
        res["auc_lick_still"] = auc(w[lick], w[still_a])
        res["auc_lick_move"] = auc(w[lick], w[c == "move"])
        res["auc_Pmove"] = auc(w[c == "P_move"], w[c == "P_still"])
        res["auc_FA_CR"] = auc(w[c == "lick_FA"], w[c == "still_CR"])
        res["auc_W"] = auc(w[c == "lick_W"], w[c == "still_W"])
        mh = np.nanmean([res["m_lick_W"], res["m_lick_A"]])
        res["fa_ratio"] = res["m_lick_FA"] / mh if np.isfinite(mh) and mh > 0 else np.nan
        lat = t.lick_latency.to_numpy(float)
        ok = lick & np.isfinite(lat)
        res["rho_lat"] = float(stats.spearmanr(w[ok], lat[ok])[0]) if ok.sum() >= 10 else np.nan
        tc = d["t_centres"]
        post = tc > (-0.3 if al == "jaw_all" else 0)
        pk = T[post].max()
        res["onset_ms"] = float(tc[post][np.argmax(T[post] >= 0.5 * pk)] * 1e3) if pk > 0 else np.nan
        res["peak_ms"] = float(tc[np.argmax(T)] * 1e3)
        rel = t.jaw_all_rel.to_numpy(float)
        res["jaw_med_ms"] = float(np.nanmedian(rel[hit]) * 1e3) if hit.sum() else np.nan
        # pre-movement projection: lick vs move (both aligned to a real jaw onset)
        if al == "jaw_all":
            bm = tc + (np.median(np.diff(tc)) / 2) <= PRE_END
            Y = X[:, :, bm] @ T[bm]
        else:   # stimulus-aligned: bins after the stimulus ending 25 ms before each trial's jaw onset
            half = np.median(np.diff(tc)) / 2
            Y = np.full((len(t), X.shape[1]), np.nan)
            for i in np.flatnonzero(lick | (c == "move")):
                b_ = (tc > 0) & (tc + half <= rel[i] - 0.025)
                if b_.any():
                    Y[i] = X[i][:, b_] @ T[b_]
        okY = np.isfinite(Y[:, 0])
        a_, b_ = lick & okY, (c == "move") & okY
        p = Y @ net
        res["pre_auc"] = auc(p[a_], p[b_])
        nl = []
        for _ in range(50):
            q = Y @ rng.permutation(net)
            nl.append(auc(q[a_], q[b_]))
        res["pre_auc_null"] = float(np.nanmean(nl)) if np.isfinite(nl).any() else np.nan
        res["pre_auc_excess"] = res["pre_auc"] - res["pre_auc_null"]
        out.append(res)
    return out


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(stats.wilcoxon(v).pvalue) if len(v) >= 5 and np.any(v != 0) else np.nan


def two(a, b):
    a, b = (np.asarray(x, float) for x in (a, b))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


V = ["auc_lick_still", "auc_lick_move", "auc_Pmove", "auc_FA_CR", "auc_W", "fa_ratio", "rho_lat", "onset_ms", "peak_ms",
     "jaw_med_ms", "pre_auc", "pre_auc_null", "pre_auc_excess"] + [f"m_{q}" for q in CLS]


def cluster_stats(gm, cl):
    """per cluster: medians over mice (Wilcoxon vs chance), per cohort, R+ vs R- (MW + Welch); no pass / fail
    (user 2026-11-23: exploratory, look at the components)"""
    rows = []
    for (meth, al, k), g in gm.groupby(["method", "alignment", "cluster"]):
        pm = g.groupby(["mouse_id", "cohort"], as_index=False)[V].mean()
        c = cl.loc[(meth, al, k)]
        row = dict(method=meth, alignment=al, cluster=k, n_mice=pm.mouse_id.nunique(), top_group=c.top_group,
                   kind=c.kind, n_Rp=int((pm.cohort == "R+").sum()), n_Rm=int((pm.cohort == "R-").sum()))
        for v in V:
            null = 0.5 if v.startswith("auc") or v in ("pre_auc", "pre_auc_null") else 0.0
            row[f"{v}_median"] = float(pm[v].median())
            row[f"{v}_p"] = wil(pm[v] - null) if not v.endswith("_ms") else np.nan
            for coh in ("R+", "R-"):
                x = pm.loc[pm.cohort == coh, v]
                row[f"{v}_{coh}"] = float(x.median())
                row[f"{v}_{coh}_p"] = wil(x - null) if not v.endswith("_ms") else np.nan
            row[f"{v}_mw_p"], row[f"{v}_welch_p"] = two(pm.loc[pm.cohort == "R+", v], pm.loc[pm.cohort == "R-", v])
        rows.append(row)
    return pd.DataFrame(rows)


def overview(cs, out):
    fig, axs = plt.subplots(4, 2, figsize=(6.4, 9.0), gridspec_kw=dict(hspace=0.75, wspace=0.4))
    for i, (meth, al) in enumerate([(m, a) for a in ALS for m in ("_splitnn", "_nn")]):
        g = cs[(cs.method == meth) & (cs.alignment == al)]
        for j, (x, y, xl, yl) in enumerate((
                ("auc_lick_move_median", "auc_FA_CR_median", "AUC lick vs jaw move without lick (active)",
                 "AUC false alarm vs correct rejection"),
                ("onset_ms_median", "pre_auc_excess_median", "Onset (ms; jaw_all: from jaw onset)",
                 "Pre-movement AUC excess (lick vs move)"))):
            ax = axs[i, j]
            ax.scatter(g[x], g[y], s=4 + g.n_mice / 3, color="0.35", lw=0)
            for _, r in g.iterrows():
                ax.annotate(str(r.cluster), (r[x], r[y]), fontsize=4, xytext=(1.5, 1.5), textcoords="offset points")
            if j == 0:
                ax.axvline(0.5, color="0.7", lw=0.4)
                ax.axhline(0.5, color="0.7", lw=0.4)
                ax.set_title(f"{al}, {FITN[meth]} ({len(g)} clusters)", fontsize=6)
            else:
                ax.axhline(0, color="0.7", lw=0.4)
                ax.axvline(0 if al == "jaw_all" else np.nanmedian(g.jaw_med_ms_median), color="tab:red", lw=0.6, ls="--")
            ax.set(xlabel=xl, ylabel=yl)
    fig.suptitle("Recurring clusters of the go-signal fits (whisker day 0): medians over mice, size = n mice, numbers = "
                 "cluster. Red dashed: jaw onset. Exploratory, uncorrected.", fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"go2_overview.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.std(x) / np.sqrt(max(len(x) - 1, 1)) if len(x) else np.nan


def gallery(cs, gm, mem, meth, al, out, per_page=7):
    """one row per recurring cluster (sorted by n mice): time factor, area profile, weight per trial class (passive
    included), numbers"""
    from ephys_utilities.allen_utils import allen_utils as AU_
    order, colours = AU_.get_area_group_custom_order(), AU_.get_custom_area_groups_colors()
    sel = cs[(cs.method == meth) & (cs.alignment == al)].sort_values("n_mice", ascending=False)
    pages = [sel.iloc[i:i + per_page] for i in range(0, len(sel), per_page)]
    for pg, part in enumerate(pages, 1):
        n = len(part)
        fig, axs = plt.subplots(n, 4, figsize=(7.6, 1.6 * n + 0.6), squeeze=False,
                                gridspec_kw=dict(hspace=1.2, wspace=0.5, width_ratios=[0.9, 1.3, 1.6, 0.75]))
        for i, (_, r) in enumerate(part.iterrows()):
            m = mem[(mem.method == meth) & (mem.alignment == al) & (mem.cluster == r.cluster)]
            g = gm[(gm.method == meth) & (gm.alignment == al) & (gm.cluster == r.cluster)]
            ax = axs[i, 0]
            tc = np.array(json.loads(m.t_centres_ms.iloc[0]))
            for coh, h in m.groupby("cohort"):
                TF = np.array([json.loads(x) for x in h.time_factor])
                ax.plot(tc, TF.mean(0), color=COH[coh], lw=0.9)
            ax.axvline(0, color="0.6", lw=0.4)
            if al != "jaw_all":
                ax.axvline(r.jaw_med_ms_median, color="tab:red", lw=0.6, ls="--")
            ax.set_title(f"cl. {r.cluster} ({r.kind})\nR+ {r.n_Rp} / R- {r.n_Rm} mice", fontsize=5)
            ax.set_xlabel("ms from jaw onset" if al == "jaw_all" else "ms from stimulus", fontsize=4.8)
            ax.tick_params(labelsize=4.5)
            ax = axs[i, 1]
            prof = pd.DataFrame([json.loads(x) for x in m.area_profile])
            gl = [q for q in order if q in prof and prof[q].notna().sum() >= 3]
            mu = prof.reindex(columns=gl).mean()
            ax.bar(range(len(gl)), mu, color=[colours[q] for q in gl], width=0.8)
            ax.axhline(0, color="0.5", lw=0.4)
            ax.set_xticks(range(len(gl)), [q.replace("Somatosensory-", "Ss-")[:9] for q in gl], rotation=70,
                          fontsize=3.6, ha="right")
            ax.tick_params(axis="y", labelsize=4.5)
            ax.set_title(f"area profile (top {r.top_group.replace('Somatosensory-', 'Ss-')})", fontsize=5)
            ax = axs[i, 2]
            pm = g.groupby(["mouse_id", "cohort"])[[f"m_{q}" for q in CLS]].mean().reset_index()
            x = np.arange(len(CLS))
            for q_, coh in enumerate(("R+", "R-")):
                h = pm[pm.cohort == coh]
                ax.errorbar(x + (q_ - 0.5) * 0.3, [h[f"m_{c}"].mean() for c in CLS], [sem(h[f"m_{c}"]) for c in CLS],
                            color=COH[coh], marker="o", ms=2, lw=0, elinewidth=0.6)
            ax.axhline(0, color="0.6", lw=0.4)
            ax.axvspan(len(CLS) - 2.5, len(CLS) - 0.5, color="0.93", lw=0, zorder=-1)
            ax.set_xticks(x, CLS_LAB, fontsize=3.8)
            ax.tick_params(axis="y", labelsize=4.5)
            ax.set_title("weight per class (SD, 0 = active still; grey = passive)", fontsize=5)
            ax = axs[i, 3]
            ax.axis("off")
            ax.text(0, 1.0, (f"AUC lick/still {r.auc_lick_still_median:.2f}\nAUC lick/move {r.auc_lick_move_median:.2f}\n"
                             f"AUC P move/still {r.auc_Pmove_median:.2f}\nAUC FA/CR {r.auc_FA_CR_median:.2f}\n"
                             f"FA / hit {r.fa_ratio_median:.2f}\nonset {r.onset_ms_median:.0f} ms\n"
                             f"pre-mov. excess {r.pre_auc_excess_median:+.3f} (p {r.pre_auc_excess_p:.3f})\n"
                             f"rho latency {r.rho_lat_median:+.2f} (p {r.rho_lat_p:.3f})"),
                    fontsize=4.4, va="top", transform=ax.transAxes)
        fig.subplots_adjust(top=1 - 0.7 / (1.6 * n + 0.6))
        fig.suptitle(f"{al}, {FITN[meth]}: recurring clusters, page {pg}/{len(pages)} (whisker day 0; green R+, magenta "
                     "R-; mean ± SEM over mice). Classes: lick W / A / FA = whisker hit, auditory hit, false alarm;\nmove "
                     "= jaw movement without lick; still = no jaw onset (W miss, A miss, correct rejection); P = passive. "
                     "Exploratory, uncorrected.", fontsize=6, y=1.0)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"go2_gallery_{al}{meth}_p{pg}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members_go.parquet")
    mem = mem[(mem.stage == "learning") & mem.alignment.isin(ALS)]
    cl = pd.read_csv(L.OUT / "tables" / "clusters_go.csv").set_index(["method", "alignment", "cluster"])
    f = L.OUT / "tables" / "go2_members.parquet"
    if a.reuse and f.exists():
        gm = pd.read_parquet(f)
    else:
        by = {}
        for r in mem.drop(columns=["time_factor", "area_profile", "t_centres_ms"]).to_dict("records"):
            by.setdefault(r["session_id"], []).append(r)
        res = Parallel(n_jobs=a.n_jobs, verbose=2)(delayed(session_rows)(s, rs) for s, rs in by.items())
        gm = pd.DataFrame([x for xs in res for x in xs])
        gm.to_parquet(f, index=False)
    cs = cluster_stats(gm, cl)
    cs.to_csv(L.OUT / "tables" / "go2_clusters.csv", index=False)
    out = L.OUT / "figures" / "aligned_gpu" / "go2"
    out.mkdir(parents=True, exist_ok=True)
    overview(cs, out)
    for meth in ("_splitnn", "_nn"):
        for al in ALS:
            gallery(cs, gm, mem, meth, al, out)
    pd.set_option("display.width", 260)
    cols = ["method", "alignment", "cluster", "n_mice", "top_group", "kind", "auc_lick_still_median",
            "auc_lick_move_median", "auc_Pmove_median", "auc_FA_CR_median", "fa_ratio_median", "onset_ms_median",
            "jaw_med_ms_median", "pre_auc_excess_median", "pre_auc_excess_p", "rho_lat_median"]
    print(cs.sort_values(["alignment", "method", "n_mice"], ascending=[True, True, False])[cols].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
