"""Which PPC-targeted insertions would be retargeted to wS1 (proposal 2026-11-18, not applied to the sheet).

Rule proposed: a PPC-targeted probe becomes wS1 when its cortical track is in the barrel field in BOTH sample and atlas
space (anatomy decides, not responses, to avoid circularity with whisker-response analyses).
A: dorsal view (right hemisphere) with Allen areas filled by area group (lightened), atlas-space entry points
   (probes.csv ml_mm / ap_mm, median per probe) of every wS1- and PPC-targeted probe, the two target centroids and
   the equidistance line; the 'PPC near wS1' probes labelled per mouse.
B: per near-wS1 PPC probe: cortex along the track (sample / atlas), passive whisker and auditory responses
   (fraction of cortical units, ROC pre-passive), cortical units that would change area group.
Run (haas): PYTHONPATH=<NAS ephys_utilities> ~/code/unit_spikes_analysis/.venv/bin/python 002_retarget_figure.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[0] / "ssl-paper-figures"
sys.path.insert(0, str(FIG))
sys.path.insert(0, str(FIG / "figures/SuppFigure2_recording"))
import paperstyle as ps  # noqa: E402
import area_group_schematic as ags  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402

RES_DIR = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-ppc-target-review")
REC = {"MH038": "wS1", "MH034": "wS1", "MH037": "wS1", "MH021": "wS1?",
       "MH036": "PPC", "MH039": "PPC", "MH035": "PPC?"}
C_RET, C_KEEP = "#d62728", "k"
REC_TXT = {"wS1": "→ wS1", "wS1?": "→ wS1 (flag)", "PPC": "keep PPC", "PPC?": "keep PPC (flag)"}
CENT = {"wS1": (3.38, -1.45), "PPC": (2.02, -1.84)}  # ML, AP (mm), 001
LAYER = re.compile(r"(1|2/3|2|3|4|5|6a|6b)")


def allen_area(atlas):
    """Structure id -> Allen area acronym with cortical layers collapsed."""
    out = {}
    for sid, s in atlas.structures.items():
        p = s["structure_id_path"]
        par = atlas.structures[p[-2]]["acronym"] if len(p) > 1 else s["acronym"]
        a = s["acronym"]
        out[sid] = par if a.startswith(par) and LAYER.fullmatch(a[len(par):]) else a
    return out


def group_of(acr):
    if acr.startswith("SSp-bfd"):
        return "Somatosensory-whisker"
    if acr in ("VISa", "VISrl"):
        return "Posterior parietal areas"
    if acr in ("SSp-tr", "SSp-un", "SSp-ul", "SSp-ll"):
        return "Somatosensory-body"
    return None


def parse(seq):
    if not isinstance(seq, str):
        return []
    return [(m.group(1), float(m.group(2))) for m in re.finditer(r"([\w/-]+) (\d+)%", seq)]


def main():
    from brainglobe_atlasapi import BrainGlobeAtlas
    atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    ann = np.asarray(atlas.annotation)
    B = ags.BREGMA
    ap_i = lambda ap: (B["ap"] - ap * 1000) / ags.RES  # noqa: E731
    ml_i = lambda ml: (B["ml"] + ml * 1000) / ags.RES  # noqa: E731

    df = ags.structure_table(atlas)
    df["area_acronym_custom"] = np.where(df.area_group.notna(), df.id.map(allen_area(atlas)), None)
    order = au.get_area_group_custom_order()
    cols = au.get_custom_area_groups_colors()
    lut = ags.Lut(df, order)
    pal = np.array([(1.0, 1.0, 1.0)] + [to_rgb(cols[g]) for g in order]) * 0.45 + 0.55

    excl = df.loc[df.excluded, "id"].to_numpy()
    valid = (ann > 0) & ~np.isin(ann, excl)
    k = np.argmax(valid, axis=1)
    dors = np.take_along_axis(ann, k[:, None, :], axis=1)[:, 0, :] * valid.any(1)
    r0, r1 = int(ap_i(0.3)), int(ap_i(-3.3))
    c0, c1 = int(ml_i(0.8)), int(ml_i(4.8))
    sub = dors[r0:r1, c0:c1]

    pr = pd.read_csv(RES_DIR / "probes.csv")
    pr["ml"] = pr.ml_mm.abs()
    near = pr[pr.kind == "PPC near wS1"].copy()
    near["rec"] = near.mouse_id.map(REC)
    ref = pr.groupby("kind")[["whisker_passive_pre", "auditory_passive_pre"]].median()

    ps.set_nature_rc()
    W, H = 183.0, 104.0
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
    # ---------------- A: map
    aw = 80.0
    LM = 26.0  # mm of label space right of the map
    s = (aw - LM) / sub.shape[1]
    ah = sub.shape[0] * s
    ax = fig.add_axes([2 / W, 1 - (6 + ah) / H, aw / W, ah / H])
    a = ags.render(ax, sub, lut, pal, ags.MARG / s)
    ax.set_title("A  Dorsal view, atlas-space entry points", fontsize=7, loc="left", pad=3)
    X = lambda ml: ml_i(ml) - c0  # noqa: E731
    Y = lambda ap: ap_i(ap) - r0  # noqa: E731
    # Allen area names inside (small, black) for orientation
    for size, name, c, r in ags.anchors(a, lut.areas, None, min_px=150):
        ax.text(c, r, name, fontsize=4.5, ha="center", va="center", color="0.25")
    # equidistance line between centroids
    (x1, y1), (x2, y2) = [(X(m), Y(p)) for m, p in CENT.values()]
    mx, my, dx, dy = (x1 + x2) / 2, (y1 + y2) / 2, -(y2 - y1), (x2 - x1)
    t = np.array([-3, 3])
    ax.plot(mx + t * dx, my + t * dy, ls=(0, (2, 1.5)), color="0.3", lw=0.6)
    ax.set_xlim(-0.5, sub.shape[1] - 0.5 + LM / s)
    ax.set_ylim(sub.shape[0] - 0.5, -0.5)
    jit = np.random.default_rng(0)
    for kind, mk, fc, ec, ms in (("wS1", "o", "white", "0.35", 2.6), ("PPC", "^", "white", "0.35", 3.0)):
        d = pr[pr.kind == kind]
        ax.plot(X(d.ml) + jit.normal(0, 0.6, len(d)), Y(d.ap_mm) + jit.normal(0, 0.6, len(d)), mk, ms=ms,
                mfc=fc, mec=ec, mew=0.5, ls="none")
    for rec, d in near.groupby("rec"):
        c = C_RET if rec.startswith("wS1") else C_KEEP
        ax.plot(X(d.ml), Y(d.ap_mm), "^", ms=3.4, mfc="white" if rec.endswith("?") else c, mec=c, mew=0.8, ls="none")
    for name, (m, p) in CENT.items():
        ax.plot(X(m), Y(p), "*", ms=6, mfc="w", mec="k", mew=0.6)
        ax.annotate(f"{name}\ncentroid", (X(m), Y(p)), xytext=(4, -6), textcoords="offset points", fontsize=5,
                    ha="left", va="top")
    cand = []
    for m, d in near.groupby("mouse_id"):
        cand.append((0, f"{m} ×{len(d)} {REC_TXT[REC[m]]}", float(X(d.ml.iloc[0])), float(Y(d.ap_mm.iloc[0]))))
    ags.label_outside(ax, cand, sub.shape[1], sub.shape[0], s, fs=5.5, only="R")
    for t in ax.texts[-len(cand):]:
        t.set_color(C_RET if "wS1" in t.get_text() else C_KEEP)
    # scale + legend
    ax.plot([2, 2 + 1000 / ags.RES], [sub.shape[0] + 6] * 2, color="k", lw=1, clip_on=False)
    ax.text(2 + 1000 / ags.RES + 2, sub.shape[0] + 6, "1 mm", fontsize=5.5, va="center")
    hl = [Line2D([], [], marker="o", ls="none", mfc="w", mec="0.35", ms=2.6, label="wS1-targeted"),
          Line2D([], [], marker="^", ls="none", mfc="w", mec="0.35", ms=3, label="PPC-targeted"),
          Line2D([], [], marker="^", ls="none", mfc=C_RET, mec=C_RET, ms=3.4, label="PPC near wS1 → retarget wS1"),
          Line2D([], [], marker="^", ls="none", mfc=C_KEEP, mec=C_KEEP, ms=3.4, label="PPC near wS1, keep PPC"),
          Line2D([], [], marker="^", ls="none", mfc="w", mec=C_RET, mew=0.8, ms=3.4, label="open = flagged"),
          Line2D([], [], color="0.3", ls=(0, (2, 1.5)), lw=0.6, label="equidistant to both centroids")]
    fig.legend(handles=hl, loc="lower left", bbox_to_anchor=(2 / W, 0.0), ncol=2, fontsize=5.5, frameon=False,
               handletextpad=0.4, columnspacing=1.0)

    # ---------------- B: per probe
    rank = {"wS1": 0, "wS1?": 1, "PPC": 2, "PPC?": 3}
    near = near.assign(_r=near.rec.map(rank)).sort_values(["_r", "mouse_id", "session_id"]).reset_index(drop=True)
    n = len(near)
    x0, top, rowh = 84.0, 8.0, 6.0
    lab_w, bar_w, dot_w = 25.0, 30.0, 22.0
    bh = n * rowh
    axb = fig.add_axes([(x0 + lab_w) / W, 1 - (top + bh) / H, bar_w / W, bh / H])
    axd = fig.add_axes([(x0 + lab_w + bar_w + 3) / W, 1 - (top + bh) / H, dot_w / W, bh / H])
    axn = fig.add_axes([(x0 + lab_w + bar_w + dot_w + 5) / W, 1 - (top + bh) / H, 11 / W, bh / H])
    for axx in (axb, axd, axn):
        axx.set_ylim(n - 0.5, -0.5)
        axx.set_yticks([])
        for sp in ("top", "right", "left"):
            axx.spines[sp].set_visible(False)
    fig.text(x0 / W, 1 - 4.5 / H, "B  PPC-targeted probes nearer the wS1 centroid", fontsize=7, va="center")
    for i, rw in near.iterrows():
        c = C_RET if rw.rec.startswith("wS1") else C_KEEP
        day = f"d{int(rw.day)}" if pd.notna(rw.day) else ""
        sess = rw.session_id.split("_")[1][4:]
        axb.text(-0.04, i, f"{rw.mouse_id} {sess} {rw.probe} {day}", transform=axb.get_yaxis_transform(),
                 ha="right", va="center", fontsize=5.5, color=c)
        axb.text(-0.04, i + 0.33, REC_TXT[rw.rec], transform=axb.get_yaxis_transform(), ha="right", va="center",
                 fontsize=4.8, color=c)
        for off, seq in ((-0.18, rw.cortex_sample), (0.18, rw.cortex_atlas)):
            left = 0.0
            for acr, pct in parse(seq):
                g = group_of(acr)
                axb.barh(i + off, pct, left=left, height=0.3, color=cols.get(g, "0.8") if g else "0.8",
                         edgecolor="white", lw=0.3)
                if pct >= 18:
                    axb.text(left + pct / 2, i + off, acr, fontsize=4.2, ha="center", va="center",
                             color="k" if g != "Posterior parietal areas" else "w")
                left += pct
        if pd.isna(rw.n_cortex):
            axd.text(0.5, i, "no data", fontsize=5, ha="center", va="center", color="0.4",
                     transform=axd.get_yaxis_transform())
            continue
        axd.plot(100 * rw.whisker_passive_pre, i, "o", ms=3, color=cols["Somatosensory-whisker"], mec="none")
        axd.plot(100 * rw.auditory_passive_pre, i, "s", ms=2.6, color=cols["Auditory areas"], mec="none")
        axn.barh(i, rw.n_cortex, height=0.5, color=c)
        axn.text(rw.n_cortex + 8, i, f"{int(rw.n_cortex)}", fontsize=5, va="center")
    axb.set_xlim(0, 100)
    axb.set_xlabel("Cortex along track (%)\nupper bar sample, lower atlas", fontsize=6)
    for kind, ls in (("wS1", "-"), ("PPC", ":")):
        for col, cc in (("whisker_passive_pre", cols["Somatosensory-whisker"]),
                        ("auditory_passive_pre", cols["Auditory areas"])):
            axd.axvline(100 * ref.loc[kind, col], color=cc, ls=ls, lw=0.6)
    axd.set_xlim(-2, 52)
    axd.set_xlabel("Passive responsive (%)", fontsize=6)
    axd.legend(handles=[Line2D([], [], marker="o", ls="none", color=cols["Somatosensory-whisker"], ms=3,
                               label="whisker"),
                        Line2D([], [], marker="s", ls="none", color=cols["Auditory areas"], ms=2.6, label="auditory"),
                        Line2D([], [], color="0.3", lw=0.6, label="wS1 probes (median)"),
                        Line2D([], [], color="0.3", lw=0.6, ls=":", label="far PPC (median)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=5, frameon=False, ncol=2,
               handlelength=1.4, columnspacing=0.8)
    axn.set_xlim(0, 420)
    axn.set_xlabel("Cortical units", fontsize=6)
    tot = near.loc[near.rec.str.startswith("wS1"), "n_cortex"].sum()
    fig.text(x0 / W, 1 - (top + bh + 23) / H,
             f"Retargeting moves {int(tot)} cortical units (good + mua) from Posterior parietal to Ss-whisker;"
             "\nsubcortical units unchanged. Flag: MH021 responds PPC-like; MH035 is SSp-un/ul in sample space.", fontsize=5.5)
    for axx in (axb, axd, axn):
        axx.tick_params(labelsize=5, length=1.5, width=0.4)
    ps.save_figure(fig, RES_DIR, "retarget_proposal")
    print("saved", RES_DIR / "retarget_proposal")


if __name__ == "__main__":
    main()
