"""Figure 3 panel E proposal (user 2026-11-26): replace the ad-hoc z >= 3 cut by an exact permutation test with FDR.

Per decoder x fine cluster (v2 main, 100 clusters, metaclusters at cut 0.40):
  cluster weight(t) = log2 mean over the cluster's neurons of |w| / mean |w| (per bin; pseudo-population decoders:
    mean over all neurons; hit vs miss per-session decoders: mean over the session's neurons);
  statistic = peak over the window of the 25-ms (5-bin) smoothed cluster weight;
  null = the same statistic for 'clusters' made by shuffling cluster labels within each session (N_PERM = 2000; same
    neurons per session, random membership), so the max over time is part of the null;
  z = (obs - mean null) / sd null (dot size, as now); exact p = (1 + #null >= obs) / (1 + N_PERM);
  significant = Benjamini-Hochberg q < 0.05 across the 100 clusters of the decoder (old rule: z >= 3);
  time course z(t) per bin with the same shuffles; half-rise = first time the smoothed z(t) reaches half its peak,
    searching back from the peak (dot colour, relative to the decoder reference time).
Decoders: whisker / auditory detection, whisker vs auditory (N = 200 area-group pseudo-populations), hit vs miss
  (per-session, whole brain), movement = FA vs CR at jaw onset (detection/movement_jaw_active, N = 200).
Also saved for panel D (example = strongest whisker-detection MC 12 cluster): the wide-grid observed and 80 shuffled
  cluster-weight curves, the 2000 shuffled peaks, and the zoom-grid (20-ms bins, 2-ms steps) z(t).
Output: manuscript Figures/_explore/Figure3_E_proposal_<date>/{option_e_cluster_stats.csv, option_e_z_time.parquet,
        option_e_example_*.csv, E_proposal.{png,pdf}}
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python <this>
"""
import datetime
import glob
import importlib
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute as C                                       # noqa: E402  (metacluster cut 0.40, v2 root)
import plot as P                                          # noqa: E402  (colour map / size scale of panel E)
sys.path.insert(0, str(C.REPO / "projects" / "ssl-decoding-rerun-202611"))
p004 = importlib.import_module("004_main_figure_prototype")

OUT = C.FIGROOT / "_explore" / f"Figure3_E_proposal_{datetime.date.today().isoformat()}"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
N_PERM, Q, K = 2000, 0.05, 100
ZOOM_T = np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 3)
DECS = [("wdet", "Whisker detection", "detection/whisker_active", (-0.2, 0.6), (0.0, 0.3)),
        ("adet", "Auditory detection", "detection/auditory_active", (-0.2, 0.6), (0.0, 0.3)),
        ("wva", "Whisker vs auditory", "arrival/active_final_n200", (-0.2, 0.6), (0.0, 0.3)),
        ("hm", "Hit vs miss", "hitmiss_stim_whole_brain", None, (0.0, 0.3)),
        ("move", "Movement (FA vs CR, jaw)", "detection/movement_jaw_active", (-1.0, 0.5), (-0.2, 0.3))]
EXAMPLE = dict(decoder="wdet", metacluster=12)


def pseudo_weights(folder, wide):
    """per neuron mean weight (area-group level, N = 200), full vector = wide grid + zoom grid"""
    wa = C.RERUN / folder / "weights_all.parquet"
    if wa.exists():
        d = pd.read_parquet(wa)
        d = d[(d.level == "area_group") & (d.N == 200)]
        W = np.stack(d.w_mean.to_numpy()).astype(float) * d.n_samples.to_numpy()[:, None]
        n = d.n_samples.to_numpy().astype(float)
    else:
        fs = sorted(glob.glob(str(C.RERUN / folder / "weight_chunks" / "area_group__*__N200__*.parquet")))
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        W = np.stack(d.w_sum.to_numpy()).astype(float)
        n = d.n_samples.to_numpy().astype(float)
    d["cluster_id"] = d["cluster_id"].astype(str)
    g = d[KEYS].astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(g)
    ws = np.zeros((codes.max() + 1, W.shape[1])); np.add.at(ws, codes, W)
    nn = np.bincount(codes, weights=n)
    keys = d.groupby(codes)[KEYS].first().sort_index().reset_index(drop=True)
    wide_t = np.round(np.arange(wide[0], wide[1] + 1e-9, 0.005), 3)
    Wm = ws / nn[:, None]
    assert Wm.shape[1] == len(wide_t) + len(ZOOM_T), (folder, Wm.shape, len(wide_t))
    return keys, Wm[:, :len(wide_t)], wide_t, Wm[:, len(wide_t):]


def cluster_stats(E, t, lab0, sess, win, seed=0, example=None):
    sel = (t >= win[0] - 1e-9) & (t <= win[1] + 1e-9)
    t, E = t[sel], E[:, sel]
    n = len(lab0)
    cnt = np.bincount(lab0, minlength=K).astype(float)
    k5 = np.ones(5) / 5
    smooth = lambda M: np.apply_along_axis(lambda r: np.convolve(r, k5, mode="same"), -1, M)

    def means(lab):
        A = sp.csr_matrix((np.ones(n), (lab, np.arange(n))), shape=(K, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(np.asarray(A @ E) / cnt[:, None])

    obs = means(lab0)
    rng = np.random.default_rng(seed)
    groups = [np.where(sess == s)[0] for s in np.unique(sess)]
    s1 = np.zeros_like(obs); s2 = np.zeros_like(obs); pk = np.zeros((N_PERM, K)); ex_curves = []
    for i in range(N_PERM):
        lab = lab0.copy()
        for g in groups:
            lab[g] = lab0[rng.permutation(g)]
        m = means(lab)
        s1 += m; s2 += m ** 2
        pk[i] = np.nanmax(smooth(m), 1)
        if example is not None and i < 80:
            ex_curves.append(m[example])
    mu = s1 / N_PERM; sd = np.sqrt(np.maximum(s2 / N_PERM - mu ** 2, 1e-12))
    zt = (obs - mu) / sd
    peak_obs = np.nanmax(smooth(obs), 1)
    z = (peak_obs - pk.mean(0)) / pk.std(0)
    n_ge = (pk >= peak_obs[None, :]).sum(0)
    p = (1 + n_ge) / (1 + N_PERM)
    ok = np.isfinite(peak_obs)
    q = np.full(K, np.nan)
    q[ok] = multipletests(p[ok], alpha=Q, method="fdr_bh")[1]
    zs = smooth(np.nan_to_num(zt))
    rows = []
    for k in range(K):
        ip = int(np.argmax(zs[k])); pkz = zs[k, ip]
        half = np.nan
        if pkz > 0:
            j = ip
            while j > 0 and zs[k, j - 1] >= pkz / 2:
                j -= 1
            half = t[j]
        rows.append(dict(cluster=k, n=int(cnt[k]), peak_obs=float(peak_obs[k]), null_mean=float(pk[:, k].mean()),
                         null_sd=float(pk[:, k].std()), peak_z=float(z[k]), n_null_ge=int(n_ge[k]), p=float(p[k]),
                         q=float(q[k]), sig_fdr=bool(q[k] < Q) if np.isfinite(q[k]) else False,
                         sig_z3=bool(z[k] >= 3), onset_half_s=float(half), peak_t_s=float(t[ip])))
    ztime = pd.DataFrame([dict(cluster=k, t=float(tt), z=float(v), z_smooth=float(vs))
                          for k in range(K) for tt, v, vs in zip(t, zt[k], zs[k])])
    ex = None
    if example is not None:
        ex = dict(t=t, obs=obs[example], obs_s=smooth(obs[example]), null=np.array(ex_curves),
                  null_s=smooth(np.array(ex_curves)), pk=pk[:, example], peak_obs=peak_obs[example])
    return pd.DataFrame(rows), ztime, ex


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fam, size, md, pms = C.families()
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
    neu = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    neu["cluster_id"] = neu["cluster_id"].astype(str)
    sig = json.load(open(C.CACHE / "fig3_sig_onsets.json"))
    top = sorted({k.split("|")[1] for k in sig})
    ref = {}
    for key in ("wdet", "adet", "wva"):
        v = [sig[f"{key}|{a}"] for a in top if sig.get(f"{key}|{a}") is not None]
        ref[key] = min(v)
    ref["hm"] = 0.0          # pending the baseline-referenced onset (user 2026-11-26)
    ref["move"] = 0.0        # jaw run: onset not yet computed with the baseline-referenced rule
    old = pd.read_csv(C.C_SRC / "option_iv_cluster_dots.csv")
    ex_k = int(old[(old.decoder == "wdet") & (old.cluster.map(fam) == 12)].sort_values("peak_z").cluster.iloc[-1])

    allstats, ztimes = [], []
    for key, label, src, wide, win in DECS:
        if wide is None:
            (keys, W, t), _ = p004.session_weights(src)
            Wz = None
            E = p004.enrichment(keys, W, per_session=True)
        else:
            keys, W, t, Wz = pseudo_weights(src, wide)
            E = p004.enrichment(keys, W, per_session=False)
        m = keys.merge(neu[KEYS + ["cluster_label"]], on=KEYS, how="left")
        ok = m.cluster_label.notna().to_numpy()
        lab0 = m.cluster_label[ok].astype(int).to_numpy()
        sess = m.session_id[ok].astype(str).to_numpy()
        exk = ex_k if key == EXAMPLE["decoder"] else None
        st, zt_, ex = cluster_stats(E[ok], np.asarray(t), lab0, sess, win, example=exk)
        st = st.assign(decoder=key, metacluster=st.cluster.map(fam), ref_s=ref[key],
                       n_matched=int(ok.sum()), n_weights=int(len(keys)))
        allstats.append(st); ztimes.append(zt_.assign(decoder=key, grid="wide"))
        print(key, "matched", ok.sum(), "/", len(keys), "sig z>=3", st.sig_z3.sum(), "sig FDR", st.sig_fdr.sum(), flush=True)
        if ex is not None:
            pd.DataFrame(dict(t=ex["t"], obs=ex["obs"], obs_s=ex["obs_s"])).to_csv(OUT / "option_e_example_curve.csv", index=False)
            pd.DataFrame(ex["null_s"].T, columns=[f"s{i}" for i in range(len(ex["null_s"]))]).assign(t=ex["t"]).to_csv(
                OUT / "option_e_example_null_curves.csv", index=False)
            pd.DataFrame(dict(null_peak=ex["pk"])).assign(peak_obs=ex["peak_obs"], cluster=ex_k).to_csv(
                OUT / "option_e_example_null_peaks.csv", index=False)
            # zoom grid (20-ms bins, 2-ms steps) for the example (panel D, timescale-matched pair)
            Ez = p004.enrichment(keys, Wz, per_session=False)
            stz, ztz, _ = cluster_stats(Ez[ok], ZOOM_T, lab0, sess, (-0.02, 0.1))
            ztimes.append(ztz.assign(decoder=key, grid="zoom"))
            print("zoom example", ex_k, stz.loc[ex_k].to_dict(), flush=True)
    S = pd.concat(allstats, ignore_index=True)
    S["onset_rel_s"] = S.onset_half_s - S.ref_s
    S.to_csv(OUT / "option_e_cluster_stats.csv", index=False)
    pd.concat(ztimes, ignore_index=True).to_parquet(OUT / "option_e_z_time.parquet", index=False)

    # ===== figure
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none"})
    FW, FH = 183, 150
    fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
    mm = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])
    runs = {}
    for k in range(K):
        a_, b_ = runs.get(fam[k], (k, k)); runs[fam[k]] = (min(a_, k), max(b_, k))

    def dotcols(x0, y0, w, h, rule, title):
        fig.text(x0 / FW, 1 - (y0 - 9) / FH, title, fontsize=6.5, fontweight="bold", va="top")
        sa = mm(x0, y0, 2.2, h)
        for k in range(K):
            sa.barh(k, 1, height=1, color=pal[fam[k]], edgecolor="none")
        for f, (a_, b_) in runs.items():
            if b_ - a_ >= 2:
                sa.text(-0.4, (a_ + b_) / 2, str(int(f)), ha="right", va="center", fontsize=5)
        sa.set_ylim(K - 0.5, -0.5); sa.axis("off")
        colw = (w - 3.2) / len(DECS)
        for j, (key, label, *_) in enumerate(DECS):
            a = mm(x0 + 3.2 + j * colw, y0, colw, h)
            d = S[S.decoder == key].set_index("cluster").reindex(range(K))
            z = d.peak_z.fillna(0).to_numpy(); yy = np.arange(K)
            size = P._size(z)
            col = P.CMAP(P.REL(np.nan_to_num(d.onset_rel_s.to_numpy(), nan=P.REL.vmax)))
            if rule == "old":
                s_ = d.sig_z3.fillna(False).to_numpy().astype(bool)
                a.scatter(np.zeros(K)[s_], yy[s_], s=size[s_], c=col[s_], lw=0)
                a.scatter(np.zeros(K)[~s_], yy[~s_], s=np.maximum(size[~s_], 1.5), facecolor="none",
                          edgecolor=col[~s_], lw=0.35)
            else:
                s_ = d.sig_fdr.fillna(False).to_numpy().astype(bool)
                a.scatter(np.zeros(K)[s_], yy[s_], s=size[s_], c=col[s_], lw=0)
                a.scatter(np.zeros(K)[~s_], yy[~s_], s=0.8, color="0.75", lw=0)
            for f, (a_, b_) in runs.items():
                a.axhline(b_ + 0.5, color="0.9", lw=0.25, zorder=0)
            a.set_ylim(K - 0.5, -0.5); a.set_xlim(-1, 1); a.axis("off")
            a.text(0.5, 1.0, label.split(" (")[0].replace(" ", "\n", 1), transform=a.transAxes, ha="center",
                   va="bottom", fontsize=5, linespacing=0.95)
            a.text(0.5, -0.01, f"{s_.sum()}", transform=a.transAxes, ha="center", va="top", fontsize=5, color="0.3")

    y0, h = 16, 118
    dotcols(4, y0, 62, h, "old", "Current rule: filled if z ≥ 3 (same 2000\nshuffles), open otherwise, all coloured")
    dotcols(72, y0, 62, h, "new", "Proposed E: filled if FDR q < 0.05\n(2000 shuffles), grey dot otherwise")
    fig.text(4 / FW, 1 - (y0 + h + 4) / FH, "numbers under the columns = significant clusters (of 100)", fontsize=5,
             color="0.3", va="top")
    # p explained (example cluster)
    ex_pk = pd.read_csv(OUT / "option_e_example_null_peaks.csv")
    a = mm(146, 12, 34, 26)
    a.hist(ex_pk.null_peak, bins=30, color="0.7", lw=0)
    po = float(ex_pk.peak_obs.iloc[0])
    a.axvline(po, color=matplotlib.colors.to_hex(pal[12]), lw=1.2)
    nge = int((ex_pk.null_peak >= po).sum())
    a.set_title(f"How p is computed (MC 12 cluster {ex_k},\nwhisker detection)", fontsize=5.5, loc="left", pad=2)
    a.text(0.98, 0.95, f"shuffles ≥ observed: {nge} / {N_PERM}\np = (1 + {nge}) / (1 + {N_PERM})\n  = {(1 + nge) / (1 + N_PERM):.4f}",
           transform=a.transAxes, ha="right", va="top", fontsize=5)
    a.set_xlabel("peak cluster weight (log2)", fontsize=5, labelpad=1); a.set_ylabel("shuffles", fontsize=5, labelpad=1)
    P._style(a)
    tab = S.groupby("decoder").agg(z3=("sig_z3", "sum"), fdr=("sig_fdr", "sum"), min_p=("p", "min")).reindex(
        [d[0] for d in DECS])
    lines = ["Significant clusters (of 100)", f"{'decoder':<10}{'z≥3':>5}{'FDR':>5}"]
    lines += [f"{k:<10}{int(r.z3):>5}{int(r.fdr):>5}" for k, r in tab.iterrows()]
    fig.text(146 / FW, 1 - 50 / FH, "\n".join(lines), fontsize=5.5, family="DejaVu Sans Mono", va="top")
    fig.text(146 / FW, 1 - 82 / FH,
             "Test: each cluster's peak weight vs the\npeaks of 2000 same-size 'clusters' drawn\nby shuffling labels within "
             "session (the\nmax over time is in the null). Exact p,\nBenjamini-Hochberg across the 100\nclusters of each "
             "decoder.\n\nSize = z (as now). Colour = half-rise of\nthe weight z, from the decoder onset\n(hit vs miss, "
             "movement: from the\nevent, pending the baseline onset).\nNon-significant: no colour (a rise\ntime on noise "
             "is undefined).", fontsize=5, va="top", linespacing=1.15, color="0.2")
    fig.savefig(OUT / "E_proposal.png", dpi=300)
    fig.savefig(OUT / "E_proposal.pdf")
    print(tab.to_string())
    print("saved", OUT)


if __name__ == "__main__":
    main()
