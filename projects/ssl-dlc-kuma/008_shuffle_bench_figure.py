"""Shuffle 3 / 5 / 6 benchmark, step 3 (haas): pixel error of each shuffle on held-out labelled frames (2026-11-04).

Error = Euclidean distance (px) between prediction and label, per labelled bodypart. Not a fair comparison: the shuffles
were trained on different frames (s3: AB only; s5/s6: mostly MH), so the main comparison uses the frames held out of all
three shuffles ("common held-out"); each shuffle's own held-out vs training frames are shown for reference.
Frame error = median over its labelled bodyparts. "Detected" = likelihood >= PCUTOFF.
Writes shuffle_bench/{shuffle_bench.png, errors_long.csv, summary.csv}.
Usage (haas): ~/code/ibl-ai-agent/.venv/bin/python 008_shuffle_bench_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/shuffle_bench")
SHUFFLES = (3, 5, 6)
COLORS = {3: "#2a6fbb", 5: "#e08a2c", 6: "#3a9e5c"}
PCUTOFF = 0.6
BPS = ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom", "nose_tip", "nose_base", "jaw", "jaw_ref", "spout",
       "particle", "tongue", "whisker_base", "whisker_tip"]


def long_errors():
    gt = pd.read_csv(OUT / "gt.csv")
    gt["common_heldout"] = ~gt[[f"train_s{s}" for s in SHUFFLES]].any(axis=1)
    rows = []
    for s in SHUFFLES:
        pr = pd.read_csv(OUT / f"pred_s{s}.csv").drop_duplicates(["folder", "image"])
        pr = gt[["folder", "image"]].merge(pr, on=["folder", "image"], how="left")  # unreadable images -> NaN
        for bp in BPS:
            ok = gt[f"{bp}_x"].notna() & gt[f"{bp}_y"].notna()
            e = np.hypot(pr[f"{bp}_x"] - gt[f"{bp}_x"], pr[f"{bp}_y"] - gt[f"{bp}_y"])[ok]
            g = gt[ok]
            rows.append(pd.DataFrame(dict(shuffle=s, folder=g.folder, image=g.image, mouse_id=g.mouse_id, view=g.view,
                                          cohort=g.cohort, bodypart=bp, error=e, likelihood=pr.loc[ok, f"{bp}_likelihood"],
                                          trained=g[f"train_s{s}"], common_heldout=g.common_heldout)))
    return pd.concat(rows, ignore_index=True).dropna(subset=["error"])  # frames whose image could not be read


def boxes(ax, groups, labels, title):
    pos = np.arange(len(labels))
    w = 0.25
    for k, s in enumerate(SHUFFLES):
        data = [groups.get((s, lab), np.array([])) for lab in labels]
        p = pos + (k - 1) * w
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], positions=p, widths=w * 0.85, showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set(facecolor=COLORS[s], alpha=0.75, edgecolor="none")
        for x, d in zip(p, data):
            if len(d):
                ax.text(x, -0.06, f"{np.median(d):.1f}", ha="center", va="top", fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_xticks(pos, labels, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("frame error (px, median over keypoints)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    df = long_errors()
    df.to_csv(OUT / "errors_long.csv", index=False)
    fr = df.groupby(["shuffle", "folder", "image", "cohort", "view", "trained", "common_heldout"]).error.median().reset_index()
    fr["cv"] = fr.cohort + " " + fr.view
    cvs = ["AB side", "AB top", "MH side", "MH top"]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw=dict(hspace=0.45, wspace=0.25))
    ch = fr[fr.common_heldout]
    n = ch[ch.shuffle == 3].groupby("cv").size()
    boxes(axs[0, 0], {(s, c): g.error.values for (s, c), g in ch.groupby(["shuffle", "cv"])},
          [c for c in cvs], "A. Common held-out frames (no shuffle trained on them)\n" +
          ", ".join(f"{c}: n={n.get(c, 0)}" for c in cvs))

    ho = fr[~fr.trained]
    n = ho.groupby(["shuffle", "cv"]).size()
    boxes(axs[0, 1], {(s, c): g.error.values for (s, c), g in ho.groupby(["shuffle", "cv"])}, cvs,
          "B. Each shuffle's own held-out frames (any frame it did not train on)\nn per shuffle: " +
          "; ".join(f"s{s} " + "/".join(str(n.get((s, c), 0)) for c in cvs) for s in SHUFFLES))

    ax = axs[1, 0]
    b = df[df.common_heldout].groupby(["shuffle", "bodypart"]).error.median().unstack(0).reindex(BPS)
    x = np.arange(len(BPS))
    for k, s in enumerate(SHUFFLES):
        ax.bar(x + (k - 1) * 0.27, b[s], 0.25, color=COLORS[s], label=f"shuffle {s}")
    ax.set_xticks(x, BPS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("median error (px)", fontsize=8)
    ax.set_title("C. Per keypoint, common held-out frames (both cohorts, both views)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1, 1]
    t = fr.groupby(["shuffle", "cohort", "trained"]).error.median().unstack("trained")
    labels = []
    for k, s in enumerate(SHUFFLES):
        for j, c in enumerate(("AB", "MH")):
            xx = k * 3 + j
            for tr, mk in ((True, "o"), (False, "s")):
                v = t.loc[(s, c)].get(tr, np.nan) if (s, c) in t.index else np.nan
                ax.scatter(xx, v, marker=mk, s=50, color=COLORS[s], edgecolor="k" if tr else "none",
                           facecolor="white" if tr else COLORS[s], zorder=3)
            labels.append((xx, f"s{s}\n{c}"))
    ax.set_xticks([l[0] for l in labels], [l[1] for l in labels], fontsize=8)
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="k", label="trained on")
    ax.scatter([], [], marker="s", color="grey", label="held out")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("median frame error (px)", fontsize=8)
    ax.set_title("D. Training vs held-out frames, per shuffle and cohort", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DLC shuffles 3 / 5 / 6 on the merged labelled set (not a fair comparison: different training frames)",
                 fontsize=11)
    fig.savefig(OUT / "shuffle_bench.png", dpi=150, bbox_inches="tight")

    summ = []
    for name, sub in (("common_heldout", df[df.common_heldout]), ("own_heldout", df[~df.trained]), ("trained", df[df.trained])):
        g = sub.groupby(["shuffle", "cohort", "view"])
        s = g.error.agg(n_keypoints="size", median_px="median", p90_px=lambda e: np.percentile(e, 90))
        s["frac_detected"] = g.likelihood.apply(lambda l: (l >= PCUTOFF).mean())
        s["median_px_detected"] = sub[sub.likelihood >= PCUTOFF].groupby(["shuffle", "cohort", "view"]).error.median()
        summ.append(s.reset_index().assign(frames=name))
    summ = pd.concat(summ)
    summ.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    print("\ncommon held-out frames per mouse/view:")
    print(fr[fr.common_heldout & (fr.shuffle == 3)].assign(m=lambda d: d.folder.str.split("_").str[0])
          .groupby(["m", "view"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
