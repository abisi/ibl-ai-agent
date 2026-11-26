"""Figure 3 panel D options (user 2026-11-26): explain the weight z better and preview three candidates for next to D.

Row 1  how the weight z of one cluster is built (whisker detection, the strongest MC 12 cluster = panel D's example):
       (1) per neuron, |decoder weight| relative to the average neuron (log2), a few member neurons + the cluster mean;
       (2) the same cluster mean against 200 'clusters' of the same size drawn by shuffling cluster labels within each
           session (same neurons per session, random membership);
       (3) peak of the (25-ms smoothed) cluster mean vs the peaks of the shuffled clusters -> z = (obs - mean) / sd;
       (4) the time-resolved z (per bin, same shuffles) with its half-rise time (dot colour in E).
Row 2  option 1: decoding onset of each area group (panel C, first significant bin) vs the fraction of its neurons in
       the sensory metaclusters (MC 5, 7, 12; movement: lick-related MC 2, 3, 4) -- links the embedding (A, B) to C.
Row 3  option 2: decoding onset vs peak accuracy - null per area group (as in the decoding reports).
Row 4  option 3: the example cluster's mean response (whisker hit / miss, passive whisker) above its weight z.
Output: manuscript Figures/_explore/Figure3_D_options_<date>/D_options.{png,pdf}
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python <this>
"""
import datetime
import importlib
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute as C                                       # noqa: E402  (sets the metacluster cut 0.40 / v2 root)
sys.path.insert(0, str(C.REPO / "projects" / "ssl-decoding-rerun-202611"))
p004 = importlib.import_module("004_main_figure_prototype")
import ephys_utilities.allen_utils.allen_utils as allen_utils   # noqa: E402

OUT = C.FIGROOT / "_explore" / f"Figure3_D_options_{datetime.date.today().isoformat()}"
DEC = "wdet"
SRC = "detection/whisker_active"
WIN = (0.0, 0.3)
N_PERM = 200
SENSORY, LICK = {5, 7, 12}, {2, 3, 4}
LAB = {"wdet": "Whisker detection", "adet": "Auditory detection", "wva": "Whisker vs auditory", "move": "Movement"}


def style(a, fs=5.5):
    a.tick_params(labelsize=fs, length=1.5, pad=1, width=0.4)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_linewidth(0.4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fam, size, md, pms = C.families()
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
    R = pms.load_rastermap(md)
    neu = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    neu["cluster_id"] = neu["cluster_id"].astype(str)
    neu["metacluster"] = neu.cluster_label.map(fam)
    ex = pd.read_csv(sorted((C.FIGROOT).glob("Figure3_dataset_clustering_*/data/Fig3d_weight_example.csv"))[-1])
    k_ex = int(pd.read_csv(C.C_SRC / "option_iv_cluster_dots.csv").query("decoder == @DEC").assign(
        mc=lambda d: d.cluster.map(fam)).query("mc == 12").sort_values("peak_z").cluster.iloc[-1])
    mc_ex = fam[k_ex]
    col_ex = matplotlib.colors.to_hex(pal[mc_ex])

    # ---- weights -> enrichment, as explore_c_option_iv / 005 load_decoders
    (keys, W, t), _ = p004.pseudo_weights(SRC)
    E = p004.enrichment(keys, W, per_session=False)
    m = keys.merge(neu[C_KEYS + ["cluster_label"]], on=C_KEYS, how="left")
    ok = m.cluster_label.notna().to_numpy()
    E, lab0, sess = E[ok], m.cluster_label[ok].astype(int).to_numpy(), m.session_id[ok].astype(str).to_numpy()
    sel = (t >= WIN[0]) & (t <= WIN[1])
    tt, E = t[sel] * 1000, E[:, sel]
    k5 = np.ones(5) / 5
    sm = lambda r: np.convolve(r, k5, mode="same")
    mem = lab0 == k_ex
    obs = np.log2(np.nanmean(E[mem], 0))
    rng = np.random.default_rng(0)
    groups = [np.where(sess == s)[0] for s in np.unique(sess)]
    null = []
    for _ in range(N_PERM):
        lab = lab0.copy()
        for g in groups:
            lab[g] = lab0[rng.permutation(g)]
        null.append(np.log2(np.nanmean(E[lab == k_ex], 0)))
    null = np.array(null)
    pk_obs, pk_null = np.nanmax(sm(obs)), np.nanmax(np.apply_along_axis(sm, 1, null), 1)
    z_peak = (pk_obs - pk_null.mean()) / pk_null.std()
    zt = (obs - null.mean(0)) / null.std(0)

    # ---- area-level quantities (option 1 / 2)
    sig = json.load(open(C.CACHE / "fig3_sig_onsets.json"))
    cur = pd.read_csv(C.C_SRC / "option_iv_area_curves.csv")
    n_area = pd.Series(R["area_group_arr"]).value_counts()
    top = [a for a in n_area.index if a in set(cur.area_group)][:C.N_AREAS]
    fam_n = np.array([fam[k] for k in R["labels"]])
    acol = allen_utils.get_custom_area_groups_colors()
    rows = []
    for a in top:
        inA = R["area_group_arr"] == a
        for key in LAB:
            c = cur[(cur.decoder == key) & (cur.area_group == a)]
            w = (-0.2, 0.3) if key == "move" else (0.0, 0.3)
            c = c[c.t.between(*w)]
            rows.append(dict(area_group=a, decoder=key, onset_ms=(sig.get(f"{key}|{a}") or np.nan) * 1000,
                             frac=float(np.isin(fam_n[inA], list(LICK if key == "move" else SENSORY)).mean()),
                             peak=float(c.acc_minus_null.max()) if len(c) else np.nan, n=int(inA.sum())))
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "D_options_area_table.csv", index=False)

    # ---- option 3: cluster mean response
    import rastermap_psth.rastermap_utils as rmu
    prov = json.load(open(C.V2 / C.VARIANT / "variant_provenance.json"))
    m045 = importlib.import_module("045_metacluster_description")
    cfg = rmu.load_cfg(m045.CONFIG, **prov["config_overrides"])
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
    labels_c = list(rmu.COND_LABELS)
    off = np.r_[0, np.cumsum(R["n_bins_list"])]
    rows_k = R["labels"] == k_ex

    def psth(name):
        i = labels_c.index(name)
        X = R["X"][rows_k, off[i]:off[i + 1]]
        return R["t_ctrs"][i] * 1000, X.mean(0), X.std(0) / np.sqrt(len(X))

    # ================= figure
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none"})
    FW, FH = 183, 205
    fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
    mm = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])
    T = lambda x, y, s, **k: fig.text(x / FW, 1 - y / FH, s, va="top", **k)

    # row 1: z explained
    T(2, 2, "Weight z, step by step (whisker detection, example MC %d cluster, %d neurons)" % (mc_ex, mem.sum()),
      fontsize=7, fontweight="bold")
    y1, h1, w1 = 12, 30, 36
    a = mm(12, y1, w1, h1)
    idx = rng.choice(np.where(mem)[0], size=min(25, mem.sum()), replace=False)
    with np.errstate(divide="ignore"):
        for i in idx:
            a.plot(tt, sm(np.log2(E[i])), color="0.75", lw=0.3)
    a.plot(tt, sm(obs), color=col_ex, lw=1.2)
    a.axhline(0, color="k", lw=0.4, ls=":")
    a.set_ylim(-3, 3)
    a.set_ylabel("log2 |w| / mean |w|", fontsize=5.5, labelpad=1); a.set_xlabel("ms from stimulus", fontsize=5.5, labelpad=1)
    a.set_title("1  Each neuron's weight vs the\naverage neuron (grey); cluster mean", fontsize=5.5, loc="left", pad=2)
    style(a)
    a = mm(12 + w1 + 10, y1, w1, h1)
    for r in null[:80]:
        a.plot(tt, sm(r), color="0.7", lw=0.25)
    a.plot(tt, sm(obs), color=col_ex, lw=1.2)
    a.axhline(0, color="k", lw=0.4, ls=":")
    a.set_xlabel("ms from stimulus", fontsize=5.5, labelpad=1)
    a.set_title("2  Same cluster vs random 'clusters':\nlabels shuffled within session (200x)", fontsize=5.5, loc="left", pad=2)
    style(a)
    a = mm(12 + 2 * (w1 + 10), y1, w1, h1)
    a.hist(pk_null, bins=20, color="0.7", lw=0)
    a.axvline(pk_obs, color=col_ex, lw=1.2)
    a.annotate(f"z = (obs − mean) / s.d.\n   = {z_peak:.1f}", (pk_obs, a.get_ylim()[1] * 0.85), xytext=(-4, 0),
               textcoords="offset points", ha="right", fontsize=5.5)
    a.set_xlabel("peak cluster weight (log2)", fontsize=5.5, labelpad=1); a.set_ylabel("shuffles", fontsize=5.5, labelpad=1)
    a.set_title("3  Peak vs the shuffled peaks\n→ dot size in E", fontsize=5.5, loc="left", pad=2)
    style(a)
    a = mm(12 + 3 * (w1 + 10), y1, w1 - 6, h1)
    zs = sm(np.nan_to_num(zt))
    a.plot(tt, zs, color=col_ex, lw=1.0)
    ip = int(np.argmax(zs)); j = ip
    while j > 0 and zs[j - 1] >= zs[ip] / 2:
        j -= 1
    a.scatter([tt[j]], [zs[ip] / 2], s=12, color="k", zorder=4)
    a.axhline(0, color="0.8", lw=0.4)
    a.annotate("half rise\n→ dot colour", (tt[j], zs[ip] / 2), xytext=(8, -10), textcoords="offset points", fontsize=5.5)
    a.set_xlabel("ms from stimulus", fontsize=5.5, labelpad=1); a.set_ylabel("z per time bin", fontsize=5.5, labelpad=1)
    a.set_title("4  Same shuffles, bin by bin\n→ when the weight rises", fontsize=5.5, loc="left", pad=2)
    style(a)

    # rows 2 and 3: options 1 and 2
    for r_, (title, xcol, xlab) in enumerate([
            ("Option 1 — areas with more sensory-type neurons decode earlier? (onset vs share of neurons in MC 5, 7, 12; "
             "movement: MC 2, 3, 4)", "frac", "fraction of the area's neurons\nin sensory types"),
            ("Option 2 — onset vs peak accuracy (as in the decoding reports)", "peak", "peak accuracy − chance")]):
        y = 58 + r_ * 50
        T(2, y - 6, title, fontsize=6.5, fontweight="bold")
        for j_, key in enumerate(LAB):
            a = mm(14 + j_ * 44, y, 32, 30)
            d = A[A.decoder == key].dropna(subset=["onset_ms", xcol])
            for _, q in d.iterrows():
                a.scatter(q[xcol], q.onset_ms, s=12, color=acol.get(q.area_group, "#aaa"), lw=0)
            if len(d) >= 4:
                rho, p = ss.spearmanr(d[xcol], d.onset_ms)
                a.text(0.98, 0.98, f"ρ = {rho:.2f}, p = {p:.2g}\nn = {len(d)} areas", transform=a.transAxes, ha="right",
                       va="top", fontsize=5)
            a.set_title(LAB[key], fontsize=6, pad=2, loc="left")
            xl = xlab.replace("sensory", "lick-related") if (key == "move" and xcol == "frac") else xlab
            a.set_xlabel(xl, fontsize=5.5, labelpad=1)
            if j_ == 0:
                a.set_ylabel("area onset (ms)", fontsize=5.5, labelpad=1)
            style(a)
    # area key
    for i, ar in enumerate(top):
        T(178 - 2, 60 + i * 3.2, ar, fontsize=4.8, ha="right", color=acol.get(ar, "#aaa"))

    # row 4: option 3
    y = 165
    T(2, y - 6, "Option 3 — the example cluster's response above its weight z (what the weight tracks)", fontsize=6.5,
      fontweight="bold")
    a = mm(14, y, 60, 14)
    for name, c_, lab in [("Whisker hit", "#f7b519", "W hit"), ("Whisker miss", "#c8a27a", "W miss"),
                          ("Auditory hit", "#2c2cdb", "A hit")]:
        x_, mu, se = psth(name)
        a.fill_between(x_, mu - se, mu + se, color=c_, alpha=0.25, lw=0); a.plot(x_, mu, color=c_, lw=0.8, label=lab)
    a.set_xlim(-50, 300); a.set_xticklabels([]); a.set_ylabel("Δ Hz", fontsize=5.5, labelpad=1)
    a.legend(frameon=False, fontsize=5, ncol=3, loc="upper right", handlelength=1)
    style(a)
    b = mm(14, y + 16, 60, 14)
    b.plot(tt, zs, color=col_ex, lw=1.0); b.axhline(0, color="0.8", lw=0.4)
    b.set_xlim(-50, 300); b.set_xlabel("ms from stimulus", fontsize=5.5, labelpad=1)
    b.set_ylabel("weight z", fontsize=5.5, labelpad=1)
    style(b)
    T(84, y + 2, "Reads: the cluster fires within ~20 ms of the whisker stimulus on hits and misses\n"
      "(not on auditory trials); its decoder weight rises with the response and peaks ~40 ms\n"
      "later, as expected from the decoder's 50-ms causal bins (labelled at their end).\n\n"
      "Panel D in the figure would then be: steps 2-3 (schematic of the shuffle and\n"
      "the z) + this response/weight pair; the dot key moves to one line above E.",
      fontsize=5.5, color="0.25", linespacing=1.2)
    fig.savefig(OUT / "D_options.png", dpi=300)
    fig.savefig(OUT / "D_options.pdf")
    print("saved", OUT / "D_options.png", "cluster", k_ex, "z_peak", round(z_peak, 1))
    print(A.round(3).to_string())


C_KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]

if __name__ == "__main__":
    main()
