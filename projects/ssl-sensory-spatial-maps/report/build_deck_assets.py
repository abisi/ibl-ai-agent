"""Slide images for the sensory-maps + stimulus-arrival deck (run on haas): panels cut from the publication figures at
the white gaps between them, plus the link panel (decoding onset vs single-neuron latency per area group).
Output: combined_results_ks4/_sensory_spatial_maps/deck/img/*.png + manifest.txt"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from PIL import Image

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
SM, AA, AP = RES / "_sensory_spatial_maps", RES / "_stimulus_arrival", RES / "_stimulus_arrival_passive"
OUT = SM / "deck" / "img"
REPO = pathlib.Path.home() / "code" / "ibl-ai-agent" / "projects"
sys.path.insert(0, str(REPO / "ssl-stimulus-arrival-decoding" / "exploratory-analyses"))
LOG = []


def gaps(mask_1d, min_gap):
    """segments of True (content) separated by >= min_gap False"""
    segs, s, run = [], None, 0
    for i, v in enumerate(mask_1d):
        if v:
            if s is None:
                s = i
            run = 0
        elif s is not None:
            run += 1
            if run >= min_gap:
                segs.append((s, i - run + 1)); s, run = None, 0
    if s is not None:
        segs.append((s, len(mask_1d)))
    return segs


def content(im, thr=245):
    a = np.asarray(im.convert("L"))
    return a < thr


def rows_of(im, min_gap=25):
    c = content(im)
    return gaps(c.mean(1) > 0.002, min_gap)


def cols_of(im, y0, y1, min_gap=12):
    c = content(im)[y0:y1]
    return gaps(c.mean(0) > 0.003, min_gap)


def save(im, box, name, pad=8):
    x0, y0, x1, y1 = box
    W, H = im.size
    crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))
    crop.save(OUT / name)
    LOG.append(f"{name}\t{crop.size[0]}x{crop.size[1]}")


def flatmap_columns():
    im = Image.open(SM / "figures" / "cortical_flatmaps.png").convert("RGB")
    W, H = im.size
    R = rows_of(im, 15)
    body = (R[1][0], R[-1][1]) if len(R) > 2 else (0, H)            # skip the title row
    C = cols_of(im, int(H * 0.15), int(H * 0.8), 10)
    C = [c for c in C if c[1] - c[0] > W * 0.05]
    LOG.append(f"flatmap columns {C}")
    if len(C) >= 7:
        y0 = body[0]
        save(im, (C[0][0], y0, C[0][1], H), "flat_recorded_zones.png")
        save(im, (C[1][0], y0, C[2][1], H), "flat_whisker_auditory.png")
        save(im, (C[3][0], y0, C[3][1], H), "flat_modality.png")
        save(im, (C[4][0], y0, C[5][1], H), "flat_latency.png")
        save(im, (C[6][0], y0, C[6][1], H), "flat_bimodal.png")
    save(im, (0, 0, W, H), "flat_all.png", pad=0)


def summary_parts(path, tag):
    if not path.exists():
        return
    im = Image.open(path).convert("RGB")
    W, H = im.size
    R = [r for r in rows_of(im, 30) if r[1] - r[0] > H * 0.12]
    LOG.append(f"{tag} rows {R}")
    if len(R) < 3:
        save(im, (0, 0, W, H), f"{tag}_full.png", pad=0)
        return
    top, mid, bot = (max(R[0][0], int(0.03 * H)), R[0][1]), R[1], R[2]      # skip the figure title
    ct = [c for c in cols_of(im, *top, 20) if c[1] - c[0] > W * 0.1]
    LOG.append(f"{tag} top cols {ct}")
    if len(ct) >= 3:
        save(im, (ct[0][0], top[0], ct[0][1], top[1]), f"{tag}_method.png")
        save(im, (ct[1][0], top[0], ct[-1][1], top[1]), f"{tag}_heatmap_ranking.png")
    else:                                   # schematic and heatmap too close to split at a gap: split at 1/3 of the width
        save(im, (0, top[0], int(0.33 * W), top[1]), f"{tag}_method.png")
        save(im, (int(0.33 * W), top[0], W, top[1]), f"{tag}_heatmap_ranking.png")
    save(im, (0, mid[0], W, mid[1]), f"{tag}_controls.png")
    cb = [c for c in cols_of(im, *bot, 20) if c[1] - c[0] > W * 0.1]
    if len(cb) >= 2:
        save(im, (cb[0][0], bot[0], cb[0][1], bot[1]), f"{tag}_shuffle.png")
        save(im, (cb[1][0], bot[0], cb[-1][1], bot[1]), f"{tag}_timecourse.png")
    save(im, (0, 0, W, H), f"{tag}_full.png", pad=0)


def link_figure():
    """decoding onset (N = 200) vs median single-neuron latency per area group"""
    S = importlib.import_module("_style")
    AR = importlib.import_module("_areas")
    plt = S.setup()
    col = AR.colors()
    sys.path.insert(0, str(REPO / "ssl-sensory-spatial-maps" / "exploratory-analyses"))
    m3 = importlib.import_module("003_spatial_maps")
    U = m3.load_units()
    LG = U.groupby("area_group")[["latency_whisker_ms", "latency_auditory_ms"]].median()
    OB = pd.read_csv(AA / "onset_bootstrap_N200.csv")
    OB = OB[OB.level == "area_group"].set_index("area")
    D = LG.join(OB[["onset_ms", "lo", "hi"]], how="inner").dropna(subset=["onset_ms"])
    D["first"] = D[["latency_whisker_ms", "latency_auditory_ms"]].min(1)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6), gridspec_kw=dict(wspace=0.4))
    for ax, c, ttl in ((axs[0], "first", "faster modality"), (axs[1], "latency_whisker_ms", "whisker")):
        st = S.corr_panel(ax, D[c].to_numpy(), D.onset_ms.to_numpy(), colors=[col.get(a, "0.5") for a in D.index], sizes=np.full(len(D), 22))
        for a, r in D.iterrows():
            ax.annotate(a.replace(" areas", "").replace("Somatosensory-", "SS-"), (r[c], r.onset_ms), fontsize=5, xytext=(3, 2),
                        textcoords="offset points", color=col.get(a, "0.3"))
        ax.text(0.02, 0.97, f"rho = {st['rho']:.2f}, {S.fmt_p(st['p_rho'])}\nn = {st['n']} area groups", transform=ax.transAxes,
                va="top", fontsize=5.5)
        ax.set_xlabel(f"Median single-neuron latency, {ttl} (ms)")
        ax.set_ylabel("Population decoding onset (ms)")
    fig.suptitle("Single-neuron latency and population decoding onset (area groups, N = 200)", x=0.02, ha="left", fontsize=7, weight="bold")
    fig.savefig(OUT / "link_latency_onset.png", dpi=300)
    D.round(2).to_csv(OUT.parent / "link_latency_onset.csv")
    LOG.append(f"link: {len(D)} area groups")


def copies():
    for src, name in [(SM / "figures" / "projection_zones_coronal.png", "projection_zones_coronal.png"),
                      (SM / "figures" / "colocation_figure.png", "colocation_figure.png"),
                      (SM / "figures" / "modality_contours.png", "modality_contours.png"),
                      (SM / "figures" / "modality" / "targets_p1.png", "modality_targets.png"),
                      (SM / "figures" / "whisker" / "targets_p1.png", "whisker_targets.png")]:
        if src.exists():
            Image.open(src).convert("RGB").save(OUT / name)
            LOG.append(f"{name} copied")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flatmap_columns()
    summary_parts(AA / "figures" / "arrival_summary_area_group.png", "arr_groups")
    summary_parts(AA / "figures" / "arrival_summary_area_acronym_custom.png", "arr_areas")
    summary_parts(AP / "figures" / "arrival_summary_area_group.png", "pas_groups")
    summary_parts(AP / "figures" / "arrival_summary_area_acronym_custom.png", "pas_areas")
    copies()
    link_figure()
    (OUT.parent / "manifest.txt").write_text("\n".join(LOG) + "\n")
    print("\n".join(LOG))


if __name__ == "__main__":
    main()
