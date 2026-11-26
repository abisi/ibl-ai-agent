"""Figure 3 preview with the lick-aligned whisker-vs-auditory decoder (modality_lick) added (user 2026-11-26: "show me how
the current Figure 3 would look like for the modality_lick decoding"). Separate explore figure; Figure 3's compute / plot /
spec are not touched (plot.py functions reused read-only, its decoder dicts extended in memory).

Source: ssl-decoding-rerun-202611 hitmiss/rows/modality_lick_area_group (per-session 024 area-group decoders, learning
day, stable units, corrected first lick, whisker vs auditory hits, 50-ms causal bins, 5-ms steps, -600..+200 ms) and
hitmiss/weights/modality_lick_whole_brain (per-neuron weights of the whole-brain per-session decoders, as the hit/miss
column of E). NOT a pseudo-population: the only pseudo-population version is in the superseded
ssl-pseudopopulation-area-decoding (artifacts/002_pseudo_modality_lick_R+/R-.parquet, per cohort, Sept 2026), not used.
C row: real - mean linear-shift null, mean over sessions (area groups with >= 3 sessions), the 10 area groups of Fig 3 C;
  onset ticks = 042 cluster_latency (a) on real - null as reported, (b) baseline-referenced (explore_modality_lick_onsets.py).
E column: per fine cluster peak weight z vs 200 within-session label shuffles (explore_c_option_iv.cluster_peak_z),
  enrichment per session (004 enrichment(per_session=True)), window -600..+200 ms; dot colour = half-rise time relative
  to the earliest baseline-referenced area onset; the other five columns from the current Fig3e CSV.
Output: manuscript Figures/_explore/Figure3_modality_lick_<date>/
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python <this>
"""
import datetime
import glob
import importlib
import json
import os
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute as C                                         # noqa: E402  (metacluster cut 0.40, v2 root)
import plot as P                                            # noqa: E402
xiv = importlib.import_module("explore_c_option_iv")       # cluster_peak_z; imports 005 / 004 of the rerun project
p004 = xiv.f005.p004
import ephys_utilities.allen_utils.allen_utils as allen_utils   # noqa: E402

OUT = C.FIGROOT / "_explore" / f"Figure3_modality_lick_{datetime.date.today().isoformat()}"
ROWS = C.RERUN / "hitmiss" / "rows" / "modality_lick_area_group"
WTAG = "modality_lick_whole_brain"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MIN_SESS = 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig3 = sorted(C.FIGROOT.glob("Figure3_dataset_clustering_*/data"))[-1]
    c3 = pd.read_csv(fig3 / "Fig3c_area_decoding.csv")
    e3 = pd.read_csv(fig3 / "Fig3e_cluster_weights.csv")
    areas = list(c3.drop_duplicates("area_group").sort_values("area_rank").area_group)

    # ---- C: area curves
    edges = json.load(open(next(ROWS.glob("024_bin_edges_*.json"))))
    t = np.round(np.array([e[1] for e in edges]), 3)
    r = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROWS / "*.parquet")))], ignore_index=True)
    r = r[(r.condition_type == "whole") & r.real_curve.notna()]
    curves = {}
    for a, g in r.groupby("area_value"):
        if a not in areas or g.session_id.nunique() < MIN_SESS:
            continue
        L = [np.asarray(x.real_curve, float) - np.stack(list(x.shift_null_curves)).astype(float).mean(0)
             for x in g.itertuples()]
        curves[a] = (np.nanmean(L, 0), g.session_id.nunique())
    on_json = OUT / "onsets.json"
    if not on_json.exists():
        env = {**os.environ, "PYTHONPATH": f"{C.REPO}:{C.REPO / 'scripts'}"}
        subprocess.run([str(pathlib.Path.home() / "code/ibl-ai-agent/.venv/bin/python"),
                        str(HERE / "explore_modality_lick_onsets.py"), str(ROWS), str(on_json), *curves],
                       check=True, env=env, cwd=str(C.REPO))
    on = json.load(open(on_json))
    acol = allen_utils.get_custom_area_groups_colors()
    order = allen_utils.get_area_group_custom_order()
    rows = []
    for a, (cv, ns) in curves.items():
        for tt, v in zip(t, cv):
            rows.append(dict(decoder="ml", area_group=a, t=float(tt), acc_minus_null=float(v), n_sessions=ns,
                             area_color=acol.get(a, "#aaaaaa"), area_rank=order.index(a) if a in order else 99,
                             onset_raw_s=on.get(f"raw|{a}"), onset_baseline_s=on.get(f"baseline|{a}")))
    dc = pd.DataFrame(rows)
    dc.to_csv(OUT / "modality_lick_area_curves.csv", index=False)

    # ---- E: cluster weight z
    fam, size, md, pms = C.families()
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
    neu = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    neu["cluster_id"] = neu["cluster_id"].astype(str)
    (keys, W, tw), nf = p004.session_weights(WTAG)
    E = p004.enrichment(keys, W, per_session=True)
    m = keys.merge(neu[KEYS + ["cluster_label"]], on=KEYS, how="left")
    ok = m.cluster_label.notna().to_numpy()
    df = pd.DataFrame(E[ok], columns=tw)
    df["cluster"] = m.cluster_label[ok].astype(int).to_numpy()
    df["session_id"] = m.session_id[ok].astype(str).to_numpy()
    print(f"weights: {nf} session files, {len(keys)} neurons, {ok.sum()} matched to rastermap", flush=True)
    dots, zt = xiv.cluster_peak_z(dict(df=df, t=np.asarray(tw)), float(tw[0]), float(tw[-1]), K=len(fam))
    ref = np.nanmin([v for k, v in on.items() if k.startswith("baseline|") and v is not None] or [np.nan])
    dots = dots.assign(decoder="ml", metacluster=dots.cluster.map(fam), pop_onset_s=ref)
    dots["mc_color"] = dots.metacluster.map(lambda f: mc.to_hex(pal[f]))
    dots["onset_rel_s"] = dots.onset_half_s - ref
    dots["in_b"] = dots.metacluster.isin(set(e3[e3.in_b].metacluster))
    dots.to_csv(OUT / "modality_lick_cluster_dots.csv", index=False)
    zt.to_parquet(OUT / "modality_lick_cluster_z_time.parquet", index=False)
    de = pd.concat([e3, dots[[c for c in e3.columns if c in dots.columns]]], ignore_index=True)

    # ---- figure: plot.py functions, decoder dicts extended in memory only
    P.DEC = P.DEC + [("ml", "W vs A at lick")]
    P.EVENT["ml"] = "first lick"
    P.XWIN["ml"] = (-0.6, 0.2)
    P.XTICKS["ml"] = [-600, -300, 0, 200]
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none"})
    FW, FH = 183, 120
    fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
    box = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])
    fig.text(2 / FW, 1 - 2 / FH, "Figure 3 preview with the lick-aligned whisker-vs-auditory decoder (per-session "
             "area-group decoders, rerun; not a pseudo-population)", fontsize=6.5, fontweight="bold", va="top")
    for i, (col, lab) in enumerate([("onset_raw_s", "onset: 042 cluster test on accuracy − null (as reported)"),
                                    ("onset_baseline_s", "onset: same test, deviation from the early baseline "
                                     "(−600..−400 ms)")]):
        y = 9 + i * 55
        fig.text(2 / FW, 1 - y / FH, ("C (i) " if i == 0 else "C (ii) ") + lab, fontsize=5.5, va="top")
        P.plot_area_decoding(box(0, y + 4, 72, 46), dc.assign(sig_onset_s=dc[col]))
    fig.text(78 / FW, 1 - 9 / FH, "E with a sixth column (lick-aligned W vs A; whole-brain per-session weights)",
             fontsize=5.5, va="top")
    P.plot_cluster_weights(box(78, 14, 105, 100), de)
    fig.savefig(OUT / "Figure3_modality_lick_preview.png", dpi=300)
    fig.savefig(OUT / "Figure3_modality_lick_preview.pdf")
    print("saved", OUT / "Figure3_modality_lick_preview.png")
    s = dc.groupby("area_group").apply(lambda g: pd.Series(dict(
        n_sessions=g.n_sessions.iloc[0], peak=g.acc_minus_null.max(), t_peak_ms=1000 * g.t[g.acc_minus_null.idxmax()],
        at_lick=g.acc_minus_null[g.t.round(3) == 0.0].mean(), pre_400=g.acc_minus_null[g.t.between(-0.45, -0.35)].mean(),
        onset_raw_ms=1000 * (g.onset_raw_s.iloc[0] if pd.notna(g.onset_raw_s.iloc[0]) else np.nan),
        onset_bl_ms=1000 * (g.onset_baseline_s.iloc[0] if pd.notna(g.onset_baseline_s.iloc[0]) else np.nan))))
    print(s.round(3).to_string())
    top = dots.sort_values("peak_z", ascending=False).head(15)[["cluster", "metacluster", "n", "peak_z", "peak_t_s",
                                                                  "onset_half_s"]]
    print(top.round(3).to_string())
    print(dots.assign(sig=dots.peak_z >= 3).groupby("metacluster").agg(n_cl=("cluster", "size"), n_sig=("sig", "sum"),
          max_z=("peak_z", "max")).query("n_sig > 0").to_string())


if __name__ == "__main__":
    main()
