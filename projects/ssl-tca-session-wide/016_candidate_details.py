"""Detail pages for the candidate learning components (exploratory; user 2026-11-22: more figures, active trials).

One page per candidate cluster (013 numbering) and fit: the main-fit cluster and its best-matching reference cluster
(014). Whisker day-0 mice with a member (one component per mouse). Panels
  A  time course: mean +- SEM across mice per cohort
  B  area profile: mean net unit weight per area group (+- SEM across mice), R+ / R- side by side, area palette
  C  trial-class profile: per mouse, mean trial weight per class divided by the mouse's mean over active whisker trials
     (whisker hit / miss, auditory hit / miss: active; passive pre / post whisker)
  D  course over the session (active whisker trials only): trials split into 5 equal bins by position; mean
     normalised weight per bin, hits and misses separately (mean +- SEM across mice) -> learning beyond outcome = a
     rise within hits (and within misses), not only more hits
  E  per mouse: raw learning correlation excess vs outcome-controlled excess (what the outcome control removes)
  F  per mouse: Spearman correlation of trial weight with lick latency on whisker hits (movement / reward timing)
  G  examples (one R+, one R- mouse, members closest to the 75th / 50th percentile of their cohort's
     outcome-controlled excess): trial weight per trial in session order (orange whisker: filled hit, open miss; blue
     auditory, small; passive shaded), P(lick) whisker curve (right axis)
  H  example R+ mouse: outcome-controlled correlation, real value vs its linear-shift null

Run: python 016_candidate_details.py -> OUT/figures/aligned_gpu/candidates/<alignment>_<fit>_cl<k>.png|pdf
"""
import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

C12 = importlib.import_module("012_component_table")
A5 = importlib.import_module("005_build_aligned")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
WH, AU = "#f7b519", "#2c2cdb"
CAND = [("stim_sensory", 10)]   # 017 trend candidate after the artefact fill (was stim_long 67, 56, sensory 68 on valid-time tensors)
FITN = {"_splitnn": "main fit (split non-negative, baseline-subtracted)", "_nn": "reference fit (non-negative, rates)"}
NB = 5
CLS = [("W hit", "A", "whisker_trial", 1), ("W miss", "A", "whisker_trial", 0), ("A hit", "A", "auditory_trial", 1),
       ("A miss", "A", "auditory_trial", 0), ("W pass. pre", "passive_pre", "whisker_trial", None),
       ("W pass. post", "passive_post", "whisker_trial", None)]


def session_weights(row):
    """trial weights of one member component + its trials table (fit order)"""
    d = np.load(C12.D / row.session_id / f"{row.alignment}_zscore{row.method}.npz")
    w = d[f"r{int(row['rank'])}_f0"][:, int(row.comp) - 1]
    t = pd.read_parquet(A5.OUT_DIR / row.session_id / "trials.parquet").set_index("trial_index").loc[d["trial_index"]]
    return w, t.reset_index()


def per_mouse(row):
    w, t = session_weights(row)
    act = ~t.epoch.str.startswith("passive").to_numpy()
    tt, lk = t.trial_type.to_numpy(), t.lick_flag.fillna(0).to_numpy().astype(int)
    aw = np.flatnonzero(act & (tt == "whisker_trial"))
    scale = max(w[aw].mean(), 1e-12)
    out = dict(mouse_id=row.mouse_id, cohort=row.cohort, w=w, t=t, scale=scale, aw=aw)
    for name, ep, ty, l_ in CLS:
        m = ((act if ep == "A" else (t.epoch == ep).to_numpy()) & (tt == ty) & ((lk == l_) if l_ is not None else True))
        out[name] = w[m].mean() / scale if m.sum() >= 3 else np.nan
    pos = np.arange(len(aw))
    b = np.minimum((pos * NB) // max(len(aw), 1), NB - 1)
    for o, nm in ((1, "hit"), (0, "miss")):
        out[nm] = np.array([w[aw[(b == i) & (lk[aw] == o)]].mean() / scale if ((b == i) & (lk[aw] == o)).sum() >= 2
                            else np.nan for i in range(NB)])
    h = aw[lk[aw] == 1]
    lat = t.lick_latency.to_numpy(float)[h] if "lick_latency" in t else np.full(len(h), np.nan)
    ok = np.isfinite(lat)
    out["rho_lat"] = stats.spearmanr(w[h][ok], lat[ok])[0] if ok.sum() >= 10 else np.nan
    return out


def sem(x, axis=0):
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis)
    return np.nanstd(x, axis) / np.sqrt(np.maximum(n - 1, 1))


def example_null(row, rng):
    w, t = session_weights(row)
    act = ~t.epoch.str.startswith("passive").to_numpy()
    aw = np.flatnonzero(act & (t.trial_type == "whisker_trial").to_numpy())
    lk = t.lick_flag.fillna(0).to_numpy(float)[aw]
    p, idx = t.p_wh_i.to_numpy(float)[aw], t.trial_index.to_numpy()[aw]
    real = C12.partial_rho(w[aw], p, idx, lk)
    null = []
    for k, dr in C12.shifts(len(aw), rng):
        pb, pw = C12.cut(len(aw), k, dr)
        null.append(C12.partial_rho(w[aw][pw], p[pb], idx[pw], lk[pw]))
    return real, np.array([x for x in null if np.isfinite(x)])


def page(al, k, method, mem, cl, out):
    from ephys_utilities.allen_utils import allen_utils as AU_
    order, colours = AU_.get_area_group_custom_order(), AU_.get_custom_area_groups_colors()
    m = mem[(mem.method == method) & (mem.alignment == al) & (mem.cluster == k)].copy()
    c = cl.loc[(method, al, k)]
    pm = [per_mouse(r) for _, r in m.iterrows()]
    fig = plt.figure(figsize=(7.6, 9.0))
    gs = fig.add_gridspec(4, 4, hspace=1.1, wspace=0.75, height_ratios=[1, 1, 1, 1.15], top=0.94)
    nm = m.groupby("cohort").mouse_id.nunique()
    fig.suptitle(f"{al}, {FITN[method]}, cluster {k}: peak {c.peak_ms_median:.0f} ms, top group {c.top_group}; "
                 f"R+ {nm.get('R+', 0)} / R- {nm.get('R-', 0)} whisker day-0 mice (exploratory, uncorrected)", fontsize=7,
                 y=0.985)
    # A time course
    ax = fig.add_subplot(gs[0, 0])
    tc = np.array(json.loads(m.t_centres_ms.iloc[0]))
    for coh, g in m.groupby("cohort"):
        TF = np.array([json.loads(x) for x in g.time_factor])
        mu, se = TF.mean(0), sem(TF)
        ax.plot(tc, mu, color=COH[coh], lw=1)
        ax.fill_between(tc, mu - se, mu + se, color=COH[coh], alpha=0.2, lw=0)
    ax.axvline(0, color="0.6", lw=0.4)
    ax.set(xlabel="Time from stimulus (ms)", ylabel="Time factor (unit norm)", title="A  Time course (mean ± SEM)")
    # B area profile
    ax = fig.add_subplot(gs[0, 1:])
    prof = {coh: pd.DataFrame([json.loads(x) for x in g.area_profile]) for coh, g in m.groupby("cohort")}
    groups = [g for g in order if any(g in p.columns and p[g].notna().sum() >= 3 for p in prof.values())]
    x = np.arange(len(groups))
    for i, coh in enumerate(("R+", "R-")):
        if coh not in prof:
            continue
        p = prof[coh].reindex(columns=groups)
        ax.bar(x + (i - 0.5) * 0.38, p.mean(), 0.36, yerr=sem(p.to_numpy()), color=[colours[g] for g in groups],
               edgecolor=COH[coh], lw=0.8, error_kw=dict(lw=0.4, capsize=0), alpha=0.9 if coh == "R+" else 0.55)
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_xticks(x, [g.replace("Somatosensory-", "Ss-") for g in groups], rotation=50, ha="right", fontsize=5)
    ax.set(ylabel="Mean net unit weight", title="B  Area profile (left bar R+, right R-; mean ± SEM, groups with >= 3 mice)")
    # C class profile
    ax = fig.add_subplot(gs[1, 0:2])
    P = pd.DataFrame([{**{n: d[n] for n, *_ in CLS}, "cohort": d["cohort"]} for d in pm])
    for j, (n, *_r) in enumerate(CLS):
        for i, coh in enumerate(("R+", "R-")):
            v = P.loc[P.cohort == coh, n].dropna().to_numpy()
            xx = j + (i - 0.5) * 0.35
            ax.scatter(xx + (np.random.default_rng(j).random(len(v)) - 0.5) * 0.15, v, s=2.5, color=COH[coh], lw=0)
            if len(v):
                ax.plot([xx - 0.12, xx + 0.12], [np.median(v)] * 2, color="k", lw=0.9)
    ax.axhline(1, color="0.6", lw=0.4, ls=":")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(len(CLS)), [n for n, *_ in CLS], rotation=30, ha="right")
    ax.set(ylabel="Trial weight / mean on\nactive whisker trials", title="C  Trial-class profile (dots = mice, bar = median)")
    # D course over the session
    ax = fig.add_subplot(gs[1, 2:])
    xb = (np.arange(NB) + 0.5) / NB
    for coh in ("R+", "R-"):
        for nmh, ls, mk in (("hit", "-", "o"), ("miss", "--", "s")):
            Y = np.array([d[nmh] for d in pm if d["cohort"] == coh])
            if not len(Y):
                continue
            mu, se = np.nanmean(Y, 0), sem(Y)
            ax.errorbar(xb + (0.01 if coh == "R-" else 0), mu, se, color=COH[coh], ls=ls, marker=mk, ms=2.5, lw=0.8,
                        elinewidth=0.4, mfc=COH[coh] if nmh == "hit" else "white", label=f"{coh} whisker {nmh}")
    ax.legend(frameon=False, fontsize=5, ncol=2)
    ax.set_xticks(xb, [str(i + 1) for i in range(NB)])
    ax.set(xlabel="Fifth of the active whisker trials", ylabel="Trial weight / mean",
           title="D  Active whisker trials over the session (mean ± SEM)")
    # E raw vs controlled
    ax = fig.add_subplot(gs[2, 0])
    for coh, g in m.groupby("cohort"):
        ax.scatter(g.learn_rho_excess, g.learn_ctrl_excess, s=4, color=COH[coh], lw=0)
    lim = np.nanmax(np.abs(m[["learn_rho_excess", "learn_ctrl_excess"]].to_numpy())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.4, ls=":")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.axvline(0, color="0.6", lw=0.4)
    ax.set(xlabel="Learning rho excess (trial index only)", ylabel="Learning rho excess\n(trial index + outcome)",
           title="E  Effect of the outcome control")
    # F lick latency
    ax = fig.add_subplot(gs[2, 1])
    for i, coh in enumerate(("R+", "R-")):
        v = np.array([d["rho_lat"] for d in pm if d["cohort"] == coh], float)
        v = v[np.isfinite(v)]
        ax.scatter(i + (np.random.default_rng(i).random(len(v)) - 0.5) * 0.3, v, s=3, color=COH[coh], lw=0)
        if len(v) >= 5:
            ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="k", lw=1)
            ax.text(i, 1.02, f"p = {stats.wilcoxon(v).pvalue:.3f}", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5)
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_xlim(-0.6, 1.6)
    ax.set(ylabel="Spearman rho, weight vs\nlick latency (whisker hits)", title="F  Weight vs lick latency")
    # G / H examples
    exs = {}
    for coh, q in (("R+", 75), ("R-", 50)):
        g = m[(m.cohort == coh) & m.learn_ctrl_excess.notna()]
        if len(g):
            exs[coh] = g.iloc[int(np.argmin(np.abs(g.learn_ctrl_excess - np.percentile(g.learn_ctrl_excess, q))))]
    for j, coh in enumerate(("R+", "R-")):
        if coh not in exs:
            continue
        r = exs[coh]
        d = next(x for x in pm if x["mouse_id"] == r.mouse_id)
        w, t = d["w"], d["t"]
        ax = fig.add_subplot(gs[3, :].subgridspec(1, 2, wspace=0.5)[0, j])
        xt = np.arange(len(w))
        pas = t.epoch.str.startswith("passive").to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, pas.astype(int), 0]))
        for a0, a1 in zip(edges[::2], edges[1::2]):
            ax.axvspan(a0 - 0.5, a1 - 0.5, color="0.92", lw=0, zorder=-1)
        tt, lk = t.trial_type.to_numpy(), t.lick_flag.fillna(0).to_numpy().astype(int)
        for ty, col, s in (("whisker_trial", WH, 6), ("auditory_trial", AU, 2.5)):
            for l_ in (1, 0):
                mm = (tt == ty) & (lk == l_)
                ax.scatter(xt[mm], w[mm], s=s, facecolors=col if l_ else "none", edgecolors=col, lw=0.5,
                           alpha=0.45 if ty == "auditory_trial" else 0.9)
        ax.set(xlabel="Trial in session (passive blocks shaded)", ylabel="Trial weight",
               title=f"G  Example {coh}: {r.mouse_id} (excess {r.learn_ctrl_excess:+.2f})")
        ax2 = ax.twinx()
        ax2.plot(xt, t.p_wh_i.to_numpy(float), color="k", lw=0.8)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("P(lick), whisker", fontsize=5.5)
        ax2.spines["right"].set_visible(True)
    if "R+" in exs:
        ax = fig.add_subplot(gs[2, 2:])
        real, null = example_null(exs["R+"], np.random.default_rng(0))
        ax.hist(null, bins=25, color="0.7")
        ax.axvline(real, color=COH["R+"], lw=1.2)
        ax.axvline(null.mean(), color="k", lw=0.6, ls="--")
        ax.set(xlabel="Learning rho (trial index + outcome controlled)", ylabel="Shifts",
               title=f"H  Example R+ {exs['R+'].mouse_id}: real (green) vs shift null (grey)")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{al}_{method.strip('_')}_cl{k}"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / f"{stem}.png")


def main():
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members.parquet")
    mem = mem[mem.stage == "learning"]
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    ag = pd.read_csv(L.OUT / "tables" / "method_agreement.csv").set_index(["alignment", "split_cluster"])
    out = L.OUT / "figures" / "aligned_gpu" / "candidates"
    for al, k in CAND:
        page(al, k, "_splitnn", mem, cl, out)
        page(al, int(ag.loc[(al, k), "ref_cluster"]), "_nn", mem, cl, out)


if __name__ == "__main__":
    main()
