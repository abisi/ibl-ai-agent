"""Decision "go" components (user 2026-11-23: can TCA isolate a decision go signal?). Exploratory, uncorrected.

A go (decision-to-lick) component should, on the stimulus-aligned long window (-0.2..1.5 s, all stimulus trials):
 1 separate hits from misses for BOTH modalities (decision, not one stimulus): AUC(hit vs still miss) > 0.5 for whisker
   and for auditory trials;
 2 not be a jaw-movement signal: trials with a jaw movement but no lick (active misses with a DLC jaw onset 15-600 ms,
   passive trials with a jaw onset) should stay low: AUC(hit vs moving miss) > 0.5 and AUC(passive moving vs passive
   still) ~ 0.5 (movement index);
 3 be present in both cohorts (R- whisker licks are unrewarded: a decision to lick, not reward expectation);
 4 start before the movement: time-factor onset (first bin >= 50 % of the peak) earlier than the session's median jaw
   onset on hits, and a projection restricted to the bins before each trial's jaw onset (misses: a jaw onset drawn from
   the session's hits, seeded) still separates hits from still misses ("pre-movement AUC");
 5 predict the lick: on hits, weight vs lick latency (Spearman; a go signal: negative).
Jaw-aligned fits (active lick trials: whisker hits, auditory hits, no-stimulus false alarms) are summarised alongside:
pre-movement share of the time factor and weight on false alarms relative to hits (a go signal is also present on
spontaneous licks).
Per member component of every recurring cluster (whisker day 0); per mouse averages; Wilcoxon per cohort, MW + Welch.

Run (haas): python 021_go_signal.py [--n-jobs 40] [--reuse]
     -> tables/go_members.parquet, tables/go_clusters.csv, tables/go_jaw_clusters.csv,
        figures/aligned_gpu/go_overview.png|pdf, go_candidates_<fit>.png|pdf
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
WH, AU = "#f7b519", "#2c2cdb"
FITN = {"_splitnn": "main fit", "_nn": "reference fit"}
MOVE = (0.015, 0.6)
CLS = ["W_hit", "W_miss_move", "W_miss_still", "A_hit", "A_miss_move", "A_miss_still", "PW_move", "PW_still",
       "PA_move", "PA_still"]


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def classes(t):
    ep, ty = t.epoch.to_numpy(), t.trial_type.to_numpy()
    lk = t.lick_flag.fillna(0).to_numpy().astype(int)
    on = t.jaw_dlc_onset.to_numpy(float)
    mv = (on >= MOVE[0]) & (on <= MOVE[1])
    still = ~((on >= 0) & (on <= 1.0))
    act, pas = ep == "active", np.char.startswith(ep.astype(str), "passive")
    c = np.full(len(t), "", object)
    for m, k in (("whisker_trial", "W"), ("auditory_trial", "A")):
        c[act & (ty == m) & (lk == 1)] = f"{k}_hit"
        c[act & (ty == m) & (lk == 0) & mv] = f"{k}_miss_move"
        c[act & (ty == m) & (lk == 0) & still] = f"{k}_miss_still"
    for m, k in (("whisker_trial", "PW"), ("auditory_trial", "PA")):   # passive split by modality (auditory: more moves)
        c[pas & (ty == m) & mv] = f"{k}_move"
        c[pas & (ty == m) & still] = f"{k}_still"
    return c


def session_rows(sid, rows):
    out = []
    t0 = pd.read_parquet(A5.OUT_DIR / sid / "trials.parquet").set_index("trial_index")
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    cache = {}
    for r in rows:
        al, meth = r["alignment"], r["method"]
        if al not in cache:
            d = np.load(A5.OUT_DIR / sid / f"{al}.npz")
            R, base = d["rate"], d["baseline"]
            keep = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= 0.5
            X = R[:, keep] - base[:, keep, None]
            X = np.nan_to_num(X / (np.nanstd(X, (0, 2), keepdims=True) + 1e-9))
            cache[al] = (d, X, t0.loc[d["trial_index"]].reset_index())
        d, X, t = cache[al]
        f = np.load(C12.D / sid / f"{al}_zscore{meth}.npz")
        k = int(r["comp"]) - 1
        w = f[f"r{int(r['rank'])}_f0"][:, k].astype(float)
        res = {q: r[q] for q in ("session_id", "mouse_id", "cohort", "learner", "method", "alignment", "cluster", "comp")}
        if al == "stim_long":
            c = classes(t)
            act = (t.epoch == "active").to_numpy()
            sd = np.std(w[act])
            z = (w - w[c == "W_miss_still"].mean()) / sd if sd > 0 and (c == "W_miss_still").sum() else np.full(len(w), np.nan)
            for q in CLS:
                res[f"m_{q}"] = float(np.mean(z[c == q])) if (c == q).sum() >= 3 else np.nan
                res[f"n_{q}"] = int((c == q).sum())
            res["auc_W"] = auc(w[c == "W_hit"], w[c == "W_miss_still"])
            res["auc_A"] = auc(w[c == "A_hit"], w[c == "A_miss_still"])
            res["auc_W_move"] = auc(w[c == "W_hit"], w[c == "W_miss_move"])
            res["auc_A_move"] = auc(w[c == "A_hit"], w[c == "A_miss_move"])
            with np.errstate(all="ignore"):   # within modality, averaged
                res["auc_P_move"] = float(np.nanmean([auc(w[c == "PW_move"], w[c == "PW_still"]),
                                                      auc(w[c == "PA_move"], w[c == "PA_still"])]))
            hit = np.isin(c, ["W_hit", "A_hit"])
            lat = t.lick_latency.to_numpy(float)
            ok = hit & np.isfinite(lat)
            res["rho_lat"] = float(stats.spearmanr(w[ok], lat[ok])[0]) if ok.sum() >= 10 else np.nan
            # timing
            tc = d["t_centres"] * 1e3
            T = f[f"r{int(r['rank'])}_f2"][:, k]
            post = tc > 0
            pk = T[post].max()
            res["onset_ms"] = float(tc[post][np.argmax(T[post] >= 0.5 * pk)]) if pk > 0 else np.nan
            on = t.jaw_dlc_onset.to_numpy(float)
            jh = on[hit & (on >= MOVE[0]) & (on <= 1.2)]
            res["jaw_med_ms"] = float(np.median(jh) * 1e3) if len(jh) >= 5 else np.nan
            # pre-movement projection
            J = len(f["units_kept"])
            Fu = f[f"r{int(r['rank'])}_f1"][:, k]
            net = Fu[:J] - Fu[J:] if Fu.shape[0] == 2 * J else Fu
            if len(jh) >= 5:
                tstar = np.where(hit, on, np.nan)
                ms = np.flatnonzero(np.isin(c, ["W_miss_still", "A_miss_still"]))
                tstar[ms] = rng.choice(jh, len(ms))
                half = np.median(np.diff(tc)) / 2 / 1e3
                # pre-movement data per trial (units), projected on the component's neuron vector and on 50
                # unit-shuffled vectors (null: hits have larger evoked responses than misses before any movement, which
                # any neuron vector picks up)
                ii = np.flatnonzero(np.isfinite(tstar))
                Y = np.full((len(t), J), np.nan)
                for i in ii:
                    bm = post & (tc / 1e3 + half <= tstar[i] - 0.025)
                    if bm.any():
                        Y[i] = X[i][:, bm] @ T[bm]
                ok = np.isfinite(Y[:, 0])
                pw, mw_ = (c == "W_hit") & ok, (c == "W_miss_still") & ok
                p = Y @ net
                res["pre_auc_W"] = auc(p[pw], p[mw_])
                res["pre_auc_A"] = auc(p[(c == "A_hit") & ok], p[(c == "A_miss_still") & ok])
                nl = []
                for _ in range(50):
                    q = Y @ rng.permutation(net)
                    nl.append(auc(q[pw], q[mw_]))
                res["pre_auc_W_null"] = float(np.nanmean(nl)) if len(nl) else np.nan
                res["pre_auc_W_excess"] = res["pre_auc_W"] - res["pre_auc_W_null"]
        else:   # jaw alignment: whisker hits, auditory hits, false alarms (no-stimulus licks)
            ty = t.trial_type.to_numpy()
            m = {"W": ty == "whisker_trial", "A": ty == "auditory_trial", "FA": ty == "no_stim_trial"}
            mu = max(w[m["W"] | m["A"]].mean(), 1e-12)
            for q, mm in m.items():
                res[f"j_{q}"] = float(w[mm].mean() / mu) if mm.sum() >= 3 else np.nan
            tc = d["t_centres"] * 1e3
            T = f[f"r{int(r['rank'])}_f2"][:, k]
            res["pre_share"] = float(T[tc < 0].sum() / max(T.sum(), 1e-12))
            res["rho_lat"] = (float(stats.spearmanr(w, t.lick_latency.to_numpy(float), nan_policy="omit")[0])
                              if len(w) >= 10 else np.nan)
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


V_STIM = ["auc_W", "auc_A", "auc_W_move", "auc_A_move", "auc_P_move", "rho_lat", "pre_auc_W", "pre_auc_A",
          "pre_auc_W_null", "pre_auc_W_excess",
          "onset_ms", "jaw_med_ms"] + [f"m_{q}" for q in CLS]
V_JAW = ["j_W", "j_A", "j_FA", "pre_share", "rho_lat"]


def cluster_stats(gm, cl):
    rows = []
    for (meth, al, k), g in gm.groupby(["method", "alignment", "cluster"]):
        V = V_STIM if al == "stim_long" else V_JAW
        pm = g.groupby(["mouse_id", "cohort"], as_index=False)[[v for v in V if v in g]].mean()
        c = cl.loc[(meth, al, k)]
        row = dict(method=meth, alignment=al, cluster=k, n_mice=pm.mouse_id.nunique(), top_group=c.top_group,
                   peak_ms=c.peak_ms_median, kind=c.kind, n_Rp=int((pm.cohort == "R+").sum()),
                   n_Rm=int((pm.cohort == "R-").sum()))
        for v in pm.columns[2:]:
            row[f"{v}_median"] = float(pm[v].median())
            null = 0.5 if v.startswith(("auc", "pre_auc")) and not v.endswith("excess") else 0.0
            row[f"{v}_p"] = wil(pm[v] - null) if not v.endswith("_ms") else np.nan
            for coh in ("R+", "R-"):
                x = pm.loc[pm.cohort == coh, v]
                row[f"{v}_{coh}"] = float(x.median())
                row[f"{v}_{coh}_p"] = wil(x - null) if not v.endswith("_ms") else np.nan
            row[f"{v}_mw_p"], row[f"{v}_welch_p"] = two(pm.loc[pm.cohort == "R+", v], pm.loc[pm.cohort == "R-", v])
        rows.append(row)
    cs = pd.DataFrame(rows)
    s = cs.alignment == "stim_long"
    crit = {
        # auditory misses are rare: auditory AUC required > 0.6 in the mice that have >= 3 still auditory misses
        "c1_both_modalities": (cs["auc_W_R+"] > 0.55) & (cs["auc_W_R+_p"] < 0.05) & (cs["auc_A_median"] > 0.6),
        "c2_not_movement": (cs["auc_W_move_median"] > 0.55) & (cs["auc_W_move_p"] < 0.05)
        & (cs["auc_P_move_median"] < 0.6),
        "c3_both_cohorts": (cs["auc_W_R-"] > 0.55) & (cs["auc_W_R-_p"] < 0.05),
        "c4_before_movement": (cs["onset_ms_median"] < cs["jaw_med_ms_median"])
        & (cs["pre_auc_W_excess_median"] > 0.03) & (cs["pre_auc_W_excess_p"] < 0.05),
        "c5_latency": (cs["rho_lat_median"] < 0) & (cs["rho_lat_p"] < 0.05),
    }
    for k_, v in crit.items():
        cs[k_] = np.where(s, v.fillna(False), np.nan)
    cs["n_criteria"] = np.where(s, sum(cs[k_].fillna(0).astype(float) for k_ in crit), np.nan)
    return cs, list(crit)


def overview(cs, out):
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.2), gridspec_kw=dict(hspace=0.6, wspace=0.45))
    for i, meth in enumerate(("_splitnn", "_nn")):
        g = cs[(cs.method == meth) & (cs.alignment == "stim_long")]
        for j, (x, y, xl, yl) in enumerate((
                ("auc_W_median", "auc_A_median", "AUC whisker hit vs still miss", "AUC auditory hit vs still miss"),
                ("auc_W_move_median", "auc_P_move_median", "AUC whisker hit vs moving miss",
                 "AUC passive moving vs still"),
                ("onset_ms_median", "pre_auc_W_excess_median", "Time-factor onset (ms)", "Pre-movement AUC excess (whisker)"))):
            ax = axs[i, j]
            sc = ax.scatter(g[x], g[y], s=4 + g.n_mice / 3, c=g.n_criteria, cmap="viridis", vmin=0, vmax=5, lw=0)
            for _, r in g[g.n_criteria >= 3].iterrows():
                ax.annotate(f"{r.cluster}: {r.top_group.replace('Somatosensory-', 'Ss-')[:10]}, {r.peak_ms:.0f}",
                            (r[x], r[y]), fontsize=4, xytext=(2, 2), textcoords="offset points")
            if j < 2:
                ax.axhline(0.5, color="0.7", lw=0.4)
                ax.axvline(0.5, color="0.7", lw=0.4)
            else:
                ax.axhline(0, color="0.7", lw=0.4)
                ax.axvline(np.nanmedian(g.jaw_med_ms_median), color="tab:red", lw=0.6, ls="--")
            ax.set(xlabel=xl, ylabel=yl)
            if j == 0:
                ax.set_title(f"{FITN[meth]}, stim_long ({len(g)} clusters)", fontsize=6)
        fig.colorbar(sc, ax=axs[i, 2], label="criteria met (of 5)", shrink=0.8)
    fig.suptitle("Go-signal criteria per recurring cluster (stimulus-aligned long window, whisker day 0; medians over "
                 "mice, size = n mice, colour = criteria met). Red dashed: median jaw onset on hits. Uncorrected.",
                 fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"go_overview.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.std(x) / np.sqrt(max(len(x) - 1, 1)) if len(x) else np.nan


def candidates_page(cs, gm, mem, meth, out, nmax=6):
    sel = cs[(cs.method == meth) & (cs.alignment == "stim_long")].sort_values(
        ["n_criteria", "auc_A_median"], ascending=False).head(nmax)
    n = len(sel)
    fig, axs = plt.subplots(n, 4, figsize=(7.6, 1.7 * n + 0.6), squeeze=False,
                            gridspec_kw=dict(hspace=1.1, wspace=0.55, width_ratios=[1, 1.8, 0.9, 0.9]))
    for i, (_, r) in enumerate(sel.iterrows()):
        m = mem[(mem.method == meth) & (mem.alignment == "stim_long") & (mem.cluster == r.cluster)]
        g = gm[(gm.method == meth) & (gm.alignment == "stim_long") & (gm.cluster == r.cluster)]
        ax = axs[i, 0]
        tc = np.array(json.loads(m.t_centres_ms.iloc[0]))
        for coh, h in m.groupby("cohort"):
            ax.plot(tc, np.array([json.loads(x) for x in h.time_factor]).mean(0), color=COH[coh], lw=0.9)
        ax.axvline(0, color="0.6", lw=0.4)
        ax.axvline(r.jaw_med_ms_median, color="tab:red", lw=0.6, ls="--")
        ax.set_title(f"cl. {r.cluster}: {r.top_group.replace('Somatosensory-', 'Ss-')[:16]}\npeak {r.peak_ms:.0f} ms, "
                     f"{int(r.n_criteria)}/5 criteria\nR+ {r.n_Rp} / R- {r.n_Rm} mice", fontsize=5)
        ax.set_xlabel("Time from stimulus (ms)", fontsize=5)
        ax = axs[i, 1]
        pm = g.groupby(["mouse_id", "cohort"])[[f"m_{q}" for q in CLS]].mean().reset_index()
        x = np.arange(len(CLS))
        for q_, coh in enumerate(("R+", "R-")):
            h = pm[pm.cohort == coh]
            ax.errorbar(x + (q_ - 0.5) * 0.3, [h[f"m_{c}"].mean() for c in CLS], [sem(h[f"m_{c}"]) for c in CLS],
                        color=COH[coh], marker="o", ms=2.3, lw=0, elinewidth=0.6)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xticks(x, ["W hit", "W miss\nmove", "W miss\nstill", "A hit", "A miss\nmove", "A miss\nstill",
                          "PW\nmove", "PW\nstill", "PA\nmove", "PA\nstill"], fontsize=4.0)
        ax.set_ylabel("Weight (SD, 0 = W still miss)", fontsize=5)
        ax.set_title(f"AUC hit/still miss W {r.auc_W_median:.2f}, A {r.auc_A_median:.2f}; hit/moving miss W "
                     f"{r.auc_W_move_median:.2f}; passive move {r.auc_P_move_median:.2f}", fontsize=4.8)
        for j, (v, lab, null) in enumerate((("pre_auc_W_excess", "Pre-movement AUC excess\n(over shuffled units)", 0.0),
                                            ("rho_lat", "rho weight vs\nlick latency (hits)", 0.0))):
            ax = axs[i, 2 + j]
            pmv = g.groupby(["mouse_id", "cohort"])[v].mean().reset_index()
            for q_, coh in enumerate(("R+", "R-")):
                vv = pmv[pmv.cohort == coh][v].dropna().to_numpy()
                ax.scatter(q_ + (np.random.default_rng(q_).random(len(vv)) - 0.5) * 0.3, vv, s=2.5, color=COH[coh], lw=0)
                if len(vv):
                    ax.plot([q_ - 0.22, q_ + 0.22], [np.median(vv)] * 2, color="k", lw=0.9)
            ax.axhline(null, color="0.6", lw=0.4)
            ax.set_xticks([0, 1], ["R+", "R-"])
            ax.set_xlim(-0.6, 1.6)
            mw, we = r[f"{v}_mw_p"], r[f"{v}_welch_p"]
            ax.set_title(f"{lab}\np {r[f'{v}_p']:.3f}; MW {mw:.3f}, W {we:.3f}", fontsize=4.8)
    fig.subplots_adjust(top=1 - 0.75 / (1.7 * n + 0.6))
    fig.suptitle(f"Go-signal candidates, {FITN[meth]}, stimulus-aligned long window (whisker day 0). Left: time factor "
                 "(red dashed = median jaw onset on hits). Middle: weight per trial class, mean ± SEM over mice "
                 "(move = jaw onset 15-600 ms without lick;\nstill = no jaw onset in 0-1 s; P = passive). Right: "
                 "pre-movement AUC (bins before each trial's jaw onset) and latency correlation; dots = mice. "
                 "Uncorrected.", fontsize=6, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"go_candidates{meth}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members.parquet")
    mem = mem[(mem.stage == "learning") & mem.alignment.isin(["stim_long", "jaw"])]
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    f = L.OUT / "tables" / "go_members.parquet"
    if a.reuse and f.exists():
        gm = pd.read_parquet(f)
    else:
        by = {}
        for r in mem.drop(columns=["time_factor", "area_profile", "t_centres_ms"]).to_dict("records"):
            by.setdefault(r["session_id"], []).append(r)
        res = Parallel(n_jobs=a.n_jobs, verbose=2)(delayed(session_rows)(s, rs) for s, rs in by.items())
        gm = pd.DataFrame([x for xs in res for x in xs])
        gm.to_parquet(f, index=False)
    cs, crit = cluster_stats(gm, cl)
    cs.to_csv(L.OUT / "tables" / "go_clusters.csv", index=False)
    out = L.OUT / "figures" / "aligned_gpu"
    overview(cs, out)
    for meth in ("_splitnn", "_nn"):
        candidates_page(cs, gm, mem, meth, out)
    pd.set_option("display.width", 250)
    s = cs[cs.alignment == "stim_long"].sort_values(["method", "n_criteria"], ascending=[True, False])
    print(s[["method", "cluster", "n_mice", "top_group", "peak_ms", "kind", "n_criteria"] + crit +
            ["auc_W_R+", "auc_W_R-", "auc_A_median", "auc_W_move_median", "auc_P_move_median", "onset_ms_median",
             "jaw_med_ms_median", "pre_auc_W_median", "pre_auc_W_null_median", "pre_auc_W_excess_median",
             "pre_auc_W_excess_p", "rho_lat_median"]].round(3)
          .head(40).to_string(index=False))
    j = cs[cs.alignment == "jaw"].sort_values(["method", "pre_share_median"], ascending=[True, False])
    j.to_csv(L.OUT / "tables" / "go_jaw_clusters.csv", index=False)
    print(j[["method", "cluster", "n_mice", "top_group", "peak_ms", "kind", "pre_share_median", "j_W_median",
             "j_A_median", "j_FA_median", "j_FA_p", "rho_lat_median", "rho_lat_p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
