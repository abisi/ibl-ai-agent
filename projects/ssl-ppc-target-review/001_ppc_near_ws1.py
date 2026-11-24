"""Review of PPC-targeted probes whose entry point is closer to the wS1 target centroid than to the PPC one
(user request 2026-11-18, follow-up of the 2026-11-17 off-target review, anatomy/tracks/off_target).

Entry points: anatomy/tracks/entry_points_surface/4d_entry_points_atlas_surface.csv (bregma mm, atlas surface).
Target centroids: median entry point of all probes of that target (current target from the unit table, so MH022 0309
imec0 is wS1 now). Per PPC probe: distance to the PPC and wS1 centroids; near-wS1 = closer to the wS1 centroid.
Function of the probe's cortical units (isocortex, ROC units good + mua, PSTH good units; all days, cohorts pooled, day
and cohort reported): % significant whisker / auditory passive (pre) and active, whisker-auditory index
(W - A) / (W + A) on passive % sig, early passive whisker response (z, 5-30 ms), peak latency of whisker-responsive
units, mean whisker PSTH. References: wS1 probes (cortical units) and PPC probes nearer the PPC centroid.
Outputs: combined_results_ks4/ssl-ppc-target-review/{probes.csv, review.png/pdf}
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 001_ppc_near_ws1.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = R / "ssl-ppc-target-review"
ROC = R / "_roc_stage_analysis"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
TYPES = ["whisker_passive_pre", "auditory_passive_pre", "whisker_active", "auditory_active"]
CORTEX = ["Motor areas", "Frontal areas", "Somatosensory-orofacial", "Somatosensory-body", "Somatosensory-whisker",
          "Auditory areas", "Retrosplenial areas", "Posterior parietal areas", "Visual areas", "Insular areas"]
C_WS1, C_FAR, C_NEAR = "#f7b519", "#009fac", "#C0392B"


def psth_unit():
    rows, curves = [], {}
    for f in sorted((ROC / "psth").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "W_pre" not in z.files or int(z["n_W_pre"]) < 10:
            continue
        b = z["stim_bins"]
        c = (b[:-1] + b[1:]) / 2
        sd = np.maximum(z["base_sd"], 0.5)
        zz = (z["W_pre"] - z["base_mu"][:, None]) / sd[:, None]
        w = (c > 0) & (c < 0.1)
        df = pd.DataFrame(dict(session_id=f.stem, electrode_group=z["electrode_group"],
                               cluster_id=z["cluster_id"].astype(str),
                               early_z=zz[:, (c > 0.005) & (c < 0.03)].mean(1),
                               peak_lat_ms=c[w][np.argmax(zz[:, w], 1)] * 1000))
        df["row"] = np.arange(len(df))
        rows.append(df)
        curves[f.stem] = (c, zz)
    return pd.concat(rows, ignore_index=True), curves


CROSS = R / "anatomy/_viz_trials_20261116/4c_entry_crossref.csv"  # entry areas, sample vs atlas space (2026-11-16)
EXCL = None


def cortex_sequence(csv_path):
    """Cortical areas along a histology track, surface first: 'SSp-bfd 80 %, VISa 20 %' (layers and barrel names
    collapsed), from the first labelled point to the first fibre tract / non-isocortex point."""
    global EXCL
    if EXCL is None:
        import ephys_utilities.allen_utils.allen_utils as au
        EXCL = set(au.get_excluded_areas()) | {"root", "Not found in brain", "nan"}
    if not csv_path.exists():
        return None
    acr = pd.read_csv(csv_path)["Region acronym"].fillna("root").astype(str).tolist()
    import re
    seq, started = [], False
    for a in acr:
        base = re.sub(r"(SSp-bfd)-.*", r"\1", a)
        base = re.sub(r"(\d.*|-?L?\d.*)$", "", base) if not base.startswith("CA") else base
        isoc = bool(re.match(r"^(SSp|SSs|VIS|MO|RSP|AUD|TEa|PTLp|ACA|ORB|AI|GU|FRP|PL|ILA)", base))
        if a in EXCL or not isoc:
            if started:
                break
            continue
        started = True
        seq.append(base)
    if not seq:
        return ""
    s = pd.Series(seq).value_counts(normalize=True)
    return ", ".join(f"{k} {v:.0%}" for k, v in s.items())


def track_paths(mouse, session, probe):
    base = Path("/mnt/lsens-analysis") / ("Axel_Bisi" if mouse.startswith("AB") else "Myriam_Hamon") / "data" / mouse \
        / session / "Anatomy/fused/registered/segmentation"
    return base / "sample_space/tracks" / f"{probe}.csv", base / "atlas_space/tracks" / f"{probe}.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(R / "ssl-ssp-un-review/unit_labels_all_days_v2.parquet").drop_duplicates(KEYS)
    lab["probe"] = lab.electrode_group.str.split("_").str[0]
    tgt = lab.groupby(["session_id", "probe"]).agg(target=("target_region", "first"), day=("day", "first"),
                                                     reward_group=("reward_group", "first"),
                                                     mouse_id=("mouse_id", "first")).reset_index()
    ep = pd.read_csv(R / "anatomy/tracks/entry_points_surface/4d_entry_points_atlas_surface.csv")
    ep = ep.rename(columns={"session": "session_id"})[["session_id", "probe", "ml_mm", "ap_mm", "entry_area",
                                                      "target_region"]]
    ep = ep.merge(tgt, on=["session_id", "probe"], how="inner")
    changed = ep[ep.target_region != ep.target]
    if len(changed):
        print("targets changed since the entry-point table:\n", changed[["session_id", "probe", "target_region",
                                                                           "target"]].to_string())
    cen = ep.groupby("target")[["ml_mm", "ap_mm"]].median()
    print("centroids (bregma mm):\n", cen.loc[["wS1", "PPC"]].round(2))
    for t in ("wS1", "PPC"):
        ep[f"d_{t}"] = np.hypot(ep.ml_mm - cen.loc[t, "ml_mm"], ep.ap_mm - cen.loc[t, "ap_mm"])
    ep["kind"] = np.where(ep.target == "wS1", "wS1",
                          np.where(ep.target != "PPC", None, np.where(ep.d_wS1 < ep.d_PPC, "PPC near wS1", "PPC")))
    ep = ep[ep.kind.notna()]

    # functional measures of the cortical units
    q = lab[lab.quality_label.isin(["good", "mua"]) & lab.area_group.isin(CORTEX)]
    roc = pd.read_parquet(ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sig"])
    roc = roc[roc.analysis_type.isin(TYPES)]
    roc["cluster_id"] = roc.cluster_id.astype(str)
    wide = roc.pivot_table(index=KEYS, columns="analysis_type", values="sig", aggfunc="first").astype(float)
    u = q.merge(wide.reset_index(), on=KEYS, how="inner")
    ps, curves = psth_unit()
    u = u.merge(ps, on=["session_id", "electrode_group", "cluster_id"], how="left")
    u.loc[u.quality_label != "good", ["early_z", "peak_lat_ms", "row"]] = np.nan
    u.loc[u.whisker_passive_pre != 1, "peak_lat_ms"] = np.nan
    pr = u.groupby(["session_id", "probe"]).agg(
        n_cortex=("cluster_id", "size"), **{t: (t, "mean") for t in TYPES},
        early_z=("early_z", "mean"), peak_lat_ms=("peak_lat_ms", "median"),
        areas=("ccf_atlas_acronym", lambda s: ", ".join(f"{k} {v}" for k, v in
                                                     s.str.extract(r"^([A-Za-z\-]+?)(?:\d|$)")[0]
                                                     .value_counts().head(4).items()))).reset_index()
    pr["wa_index"] = (pr.whisker_passive_pre - pr.auditory_passive_pre) / (
        pr.whisker_passive_pre + pr.auditory_passive_pre)
    df = ep.merge(pr, on=["session_id", "probe"], how="left")
    # location: entry area in sample space (own brain) and atlas space, cortical areas along the track in both
    cr = pd.read_csv(CROSS).rename(columns={"session": "session_id"})
    df = df.merge(cr[["session_id", "probe", "entry_sample", "entry_atlas", "group_agreement"]],
                  on=["session_id", "probe"], how="left")
    seq_s, seq_a = [], []
    for r in df.itertuples():
        s_csv, a_csv = track_paths(r.mouse_id, r.session_id, r.probe)
        seq_s.append(cortex_sequence(s_csv) if r.kind != "wS1" else None)
        seq_a.append(cortex_sequence(a_csv) if r.kind != "wS1" else None)
    df["cortex_sample"], df["cortex_atlas"] = seq_s, seq_a
    df = df.sort_values(["kind", "d_wS1"])
    df.to_csv(OUT / "probes.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["kind", "session_id", "probe", "day", "reward_group", "entry_area", "d_wS1", "d_PPC", "n_cortex",
            "whisker_passive_pre", "auditory_passive_pre", "wa_index", "whisker_active", "auditory_active",
            "early_z", "peak_lat_ms", "areas"]
    print(df[df.kind != "wS1"][cols].round(2).to_string(index=False))
    print(df[df.kind != "wS1"][["kind", "session_id", "probe", "entry_area", "entry_sample", "entry_atlas",
                                "group_agreement", "cortex_sample", "cortex_atlas"]].to_string(index=False))
    ref = df.groupby("kind")[["whisker_passive_pre", "auditory_passive_pre", "wa_index", "whisker_active",
                              "auditory_active", "early_z", "peak_lat_ms"]].agg(["median", "count"]).round(2)
    print(ref.to_string())

    # ---------------- figure
    col = {"wS1": C_WS1, "PPC": C_FAR, "PPC near wS1": C_NEAR}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
    ax = axs[0]
    for k, g in df.groupby("kind"):
        ax.scatter(g.ml_mm, g.ap_mm, s=14, color=col[k], label=k, lw=0)
    for t in ("wS1", "PPC"):
        ax.scatter(*cen.loc[t, ["ml_mm", "ap_mm"]], marker="x", s=60, color=col[t], lw=2)
    for _, r in df[df.kind == "PPC near wS1"].iterrows():
        ax.text(r.ml_mm + 0.03, r.ap_mm, r.session_id[:5], fontsize=6)
    ax.set_xlabel("ML (mm)"); ax.set_ylabel("AP (mm)"); ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Entry points (x = target centroid)", fontsize=9)
    ax = axs[1]
    for k, g in df.groupby("kind"):
        ax.scatter(g.auditory_passive_pre, g.whisker_passive_pre, s=14, color=col[k], lw=0, label=k)
    for _, r in df[df.kind == "PPC near wS1"].iterrows():
        ax.text(r.auditory_passive_pre, r.whisker_passive_pre, r.session_id[:5], fontsize=6)
    lim = [0, np.nanmax(df[["auditory_passive_pre", "whisker_passive_pre"]].to_numpy()) * 1.05]
    ax.plot(lim, lim, ":", color="0.6", lw=0.8)
    ax.set_xlabel("auditory passive, frac. sig."); ax.set_ylabel("whisker passive, frac. sig.")
    ax.set_title("Cortical units per probe", fontsize=9)
    ax = axs[2]
    for k, g in df.groupby("kind"):
        ax.scatter(g.d_wS1, g.early_z, s=14, color=col[k], lw=0)
    for _, r in df[df.kind == "PPC near wS1"].iterrows():
        ax.text(r.d_wS1, r.early_z, r.session_id[:5], fontsize=6)
    ax.set_xlabel("entry distance to wS1 centroid (mm)"); ax.set_ylabel("whisker 5-30 ms (z)")
    ax.set_title("Early whisker response vs distance", fontsize=9)
    ax = axs[3]
    for k in ("wS1", "PPC", "PPC near wS1"):
        ids = df[df.kind == k]
        cc = []
        for _, r in ids.iterrows():
            uu = u[(u.session_id == r.session_id) & (u.probe == r.probe) & u.row.notna()]
            if r.session_id in curves and len(uu):
                c, zz = curves[r.session_id]
                cc.append(zz[uu.row.astype(int)].mean(0))
        if cc:
            m = np.nanmean(cc, 0)
            ax.plot(c * 1000, m, color=col[k], lw=1.2, label=f"{k} (n={len(cc)} probes)")
            if k == "PPC near wS1":
                for x in cc:
                    ax.plot(c * 1000, x, color=col[k], lw=0.4, alpha=0.4)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlim(-50, 200); ax.set_xlabel("time from whisker stim (ms)"); ax.set_ylabel("z (mean over good units)")
    ax.legend(frameon=False, fontsize=7); ax.set_title("Passive whisker PSTH, cortical units", fontsize=9)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"review.{ext}", dpi=200)


if __name__ == "__main__":
    main()
