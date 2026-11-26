"""Figure 3 panel C -- three candidate forms of the condensed decoding summary, for the user to choose (2026-11-25).
Output: manuscript Figures/_explore/Figure3_C_options_<date>/ (png + pdf + the tables plotted).

Input (ssl-decoding-rerun-202611/prototype_main_figure/, from 006 and 005):
  v3_metacluster_weight_z.parquet  per decoder x metacluster x time: z of the metacluster-mean decoder weight (|w| per
                                   neuron relative to the decoder's average neuron, log2) against metacluster labels
                                   permuted within session (200x)
  v2_triangle_scores.csv           per metacluster: peak z for stimulus (max of whisker / auditory detection, 0-150 ms),
                                   choice (hit vs miss, 0-300 ms), movement (-100-200 ms), clipped at 0
Options:
  (i)   table: metaclusters (rastermap order) x decoders; colour = peak z, text = onset (first time z >= 3 for 2 bins)
  (ii)  takeover curves: per decoder, the neuron-weighted mean z of each functional block over time
  (iii) triangle: stimulus / choice / movement, position = relative peak z
Movement here = the first movement run (earliest of whisker / nose / jaw onsets, partial iterations); the jaw-aligned run
replaces it when it finishes (Jed, 2026-11-26 morning).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python <this>
"""
import datetime
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-decoding-rerun-202611/prototype_main_figure")
NAMES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2/main/metacluster_description/"
                     "metacluster_names_all.csv")
OUT = (pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/_explore")
       / f"Figure3_C_options_{datetime.date.today().isoformat()}")
BLOCKS = [("Whisker sensory", [20, 21, 23, 25, 26, 27], "#D98C1F"), ("Multisensory", [12, 22, 24], "#8E5BB5"),
          ("Auditory transient", [13, 14, 15, 16, 17], "#2F5FA8"), ("Auditory sustained", [5, 6, 7, 8, 9, 10, 11], "#86A9D6"),
          ("Lick-activated", [2, 3, 4], "#C0392B"), ("Suppressed / weak", [1, 18, 19, 28, 29], "#9E9E9E")]
DEC = [("wdet", "Whisker\ndetection"), ("adet", "Auditory\ndetection"), ("wva", "Whisker vs\nauditory"),
       ("hm", "Hit vs miss"), ("move", "Movement")]
WIN = {"wdet": (0, 0.3), "adet": (0, 0.3), "wva": (0, 0.3), "hm": (0, 0.3), "move": (-0.2, 0.3)}
ZTHR, ZMAX = 3.0, 10.0


def block_of(mc):
    return next((n for n, m, _ in BLOCKS if mc in m), "Suppressed / weak")


def onset(t, z):
    ok = z >= ZTHR
    for i in range(len(t) - 1):
        if ok[i] and ok[i + 1]:
            return t[i]
    return np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Z = pd.read_parquet(SRC / "v3_metacluster_weight_z.parquet")
    names = pd.read_csv(NAMES).set_index("family")["name"]
    bcol = {n: c for n, _, c in BLOCKS}
    mcs = sorted(names.index.astype(int))
    plt.rcParams.update({"font.size": 6, "font.family": "Arial", "axes.linewidth": 0.5, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})

    # (i) table
    rows = []
    for key, _ in DEC:
        d = Z[Z.decoder == key]
        for mc, g in d.groupby("metacluster"):
            g = g.sort_values("t")
            sel = (g.t >= WIN[key][0]) & (g.t <= WIN[key][1])
            t, z = g.t.to_numpy()[sel], g.z.to_numpy()[sel]
            rows.append(dict(decoder=key, metacluster=int(mc), n=int(g.n.iloc[0]), peak_z=float(np.nanmax(z)),
                             onset_s=onset(t, z)))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "option_i_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(70 / 25.4, 95 / 25.4))
    P = tab.pivot(index="metacluster", columns="decoder", values="peak_z").reindex(index=mcs, columns=[k for k, _ in DEC])
    O = tab.pivot(index="metacluster", columns="decoder", values="onset_s").reindex(index=mcs, columns=[k for k, _ in DEC])
    im = ax.imshow(np.clip(P.values, 0, ZMAX), cmap="Reds", vmin=0, vmax=ZMAX, aspect="auto")
    for (i, j), v in np.ndenumerate(O.values):
        if np.isfinite(v) and P.values[i, j] >= ZTHR:
            ax.text(j, i, f"{v * 1000:.0f}", ha="center", va="center", fontsize=4.2,
                    color="w" if P.values[i, j] > 0.6 * ZMAX else "k")
    ax.set_xticks(range(len(DEC))); ax.set_xticklabels([l for _, l in DEC], fontsize=5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(mcs))); ax.set_yticklabels([f"{m} {names[m]}" for m in mcs], fontsize=4.5)
    for i, m in enumerate(mcs):
        ax.add_patch(plt.Rectangle((-0.5 - 0.35, i - 0.5), 0.3, 1, color=bcol[block_of(m)], clip_on=False))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend="max")
    cb.set_label(f"peak weight z (vs label permutation)\nnumbers: onset (ms), first z >= {ZTHR:g}", fontsize=5)
    ax.set_title("(i) Metacluster x decoder table", fontsize=6.5, loc="left", pad=18)
    fig.savefig(OUT / "option_i_table.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "option_i_table.pdf", bbox_inches="tight")
    plt.close(fig)

    # (ii) takeover curves (block = neuron-weighted mean z of its metaclusters)
    fig, axes = plt.subplots(1, len(DEC), figsize=(183 / 25.4, 45 / 25.4), sharey=True, gridspec_kw=dict(wspace=0.15))
    crows = []
    for ax, (key, lab) in zip(axes, DEC):
        d = Z[Z.decoder == key].assign(block=lambda x: x.metacluster.map(block_of))
        for name, _, col in BLOCKS:
            g = d[d.block == name]
            if not len(g):
                continue
            c = g.groupby("t").apply(lambda x: np.average(x.z, weights=x.n))
            ax.plot(c.index * 1000, c.values, color=col, lw=0.9, label=name)
            crows += [dict(decoder=key, block=name, t=float(t), z=float(v)) for t, v in c.items()]
        ax.axhline(0, color="0.6", lw=0.4); ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_title(lab.replace("\n", " "), fontsize=5.5)
        ax.set_xlabel("ms from movement onset" if key == "move" else "ms from stimulus", fontsize=5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("block weight z", fontsize=5)
    axes[-1].legend(frameon=False, fontsize=4.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("(ii) Takeover curves: which response-type block carries each decoder's weight over time", fontsize=6.5,
                 x=0.01, ha="left", y=1.05)
    pd.DataFrame(crows).to_csv(OUT / "option_ii_curves.csv", index=False)
    fig.savefig(OUT / "option_ii_curves.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "option_ii_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # (iii) triangle
    tri = pd.read_csv(SRC / "v2_triangle_scores.csv")
    fig, ax = plt.subplots(figsize=(60 / 25.4, 60 / 25.4))
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    ax.plot(*np.r_[V, V[:1]].T, color="0.5", lw=0.5)
    for nm, v, dy in zip(["Stimulus", "Choice", "Movement"], V, [0.06, -0.07, -0.07]):
        ax.text(v[0], v[1] + dy, nm, ha="center", va="center", fontsize=6)
    for _, r in tri.iterrows():
        w = np.array([r.z_Stimulus, r.z_Choice, r.z_Movement], float)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            continue
        xy = (w / w.sum()) @ V
        ax.scatter(*xy, s=6 + 50 * np.sqrt(r.n / 15000), color=bcol[block_of(int(r.metacluster))], edgecolor="w", lw=0.3)
        ax.text(xy[0] + 0.015, xy[1] + 0.012, str(int(r.metacluster)), fontsize=4.5)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.15, 0.95); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(iii) Stimulus / choice / movement", fontsize=6.5, loc="left")
    fig.savefig(OUT / "option_iii_triangle.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "option_iii_triangle.pdf", bbox_inches="tight")
    plt.close(fig)
    json.dump(dict(script=__file__, sources=[str(SRC / "v3_metacluster_weight_z.parquet"), str(SRC / "v2_triangle_scores.csv")],
                   z_threshold=ZTHR, windows=WIN, movement="first run (earliest onset, partial); jaw run pending"),
              open(OUT / "provenance.json", "w"), indent=1)
    print("ALL DONE", OUT)


if __name__ == "__main__":
    main()
