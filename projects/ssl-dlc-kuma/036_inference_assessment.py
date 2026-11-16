"""Assessment of the full-dataset re-inference (033): new (CSPNeXt-M snapshot-330) vs old (filtered h5 in the data
folders) DLC tracking, per session, keypoint and kinematic quantity, and over the whole dataset (user 2026-11-10:
"provide an assessment of improvement of dlc tracking over sessions and the whole datasets and over keypoints and
kinematics", for review before the new h5s are copied into the data folders as *_new).

Inputs: staged outputs DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330/<session>/ (034), the old
*_filtered.h5 of each session (Axel_Bisi/data or Myriam_Hamon/data, no *_backup_*), and the session NWB (timestamps and
mm-per-px conversion only).
Per session x camera:
  keypoints   (024 keypoint_metrics, view keypoints, likelihood cutoff 0.5): % frames confident, frame-to-frame step
              (median) and % steps > 1 mm, old and new; old <-> new distance on frames confident in both (median, mm)
  kinematics  both old and new h5 rebuilt through the converter path of 025 (dlc2kinematics smoothing, top orientation
              check, compute_kinematics_alt, the NWB timestamps; the old rebuild reproduces the NWB exactly) and read
              with dlc_helpers (clean_series defaults); per quantity: % frames valid (gating likelihoods at the
              dlc_helpers cutoffs), r(old, new) and median |new - old| on frames valid in both, SD ratio new / old
              (scale change, e.g. jaw angle). Side: jaw_distance, jaw_angle, pupil_area, nose_distance, tongue presence
              (% frames tongue_likelihood > 0.6); top: whisker_angle, top_nose_distance. Frames compared by index
              (both h5 are video-frame indexed), so per-session alignment exceptions do not matter here.
Outputs: DLCProjects/merged_training_2026-11-09/inference_assessment/
  rows/<session>_<view>.json (one per camera-session, resume-safe), keypoints.csv, kinematics.csv, sessions.csv,
  figures/assessment_dataset.png (+ pdf), figures/assessment_sessions.png, summary.md
Run on haas: nice -n 10 ~/code/ibl-ai-agent/.venv/bin/python 036_inference_assessment.py [--jobs 16] [--sessions ...]
"""

import argparse
import json
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ba = _load("ba", "025_pilot_before_after.py")
pv, ts, dh = ba.pv, ba.ts_, ba.dh
STAGE = ts.SIDE / "inference_cspnext_m_330"
OUT = ts.SIDE / "inference_assessment"
QUANT = {"side": ["jaw_distance", "jaw_angle", "pupil_area", "nose_distance"],
         "top": ["whisker_angle", "top_nose_distance"]}
SHM = "/dev/shm" if os.path.isdir("/dev/shm") else None  # haas root disk is nearly full: rebuilt files in RAM


def valid_mask(v, name):
    gate = [l for l in dh.likelihood_series(name) if l in v.likelihood]
    if not gate:
        return np.ones(len(v.timestamps), bool)
    return np.all([v.likelihood[l].to_numpy() >= dh.cutoff_for(l) for l in gate], 0)


def assess(job):
    session, view, new_h5 = job
    dest = OUT / "rows" / f"{session}_{view}.json"
    if dest.exists():
        return dest
    row = {"session_id": session, "mouse_id": session[:5], "view": view, "new_h5": str(new_h5)}
    try:
        old_h5 = pv.old_file(session, view)
        row["old_h5"] = str(old_h5)
        m = __import__("re").search(r"shuffle(\d)", old_h5.name)
        row["old_shuffle"] = int(m.group(1)) if m else 6
    except AssertionError:
        old_h5 = None
        row["old_h5"] = None
    new, old = pv.wide(new_h5), (pv.wide(old_h5) if old_h5 else None)
    row["n_new"], row["n_old"] = len(new), (len(old) if old is not None else None)
    mmpx = pv.mm_per_px(session[:5], view)
    kp = {}
    for k in ts.VIEW_KEYPOINTS[view]:
        e = {}
        for name, d in (("new", new), ("old", old)):
            if d is None or (k, "x") not in d.columns:
                continue
            conf, met = pv.keypoint_metrics(d, k, mmpx)
            e[name] = {kk: (None if not np.isfinite(vv) else float(vv)) for kk, vv in met.items()}
            e[name + "_conf"] = conf
        if "new_conf" in e and "old_conf" in e:
            n = min(len(new), len(old))
            both = e["new_conf"][:n] & e["old_conf"][:n]
            dist = np.hypot(new[(k, "x")].to_numpy()[:n] - old[(k, "x")].to_numpy()[:n],
                            new[(k, "y")].to_numpy()[:n] - old[(k, "y")].to_numpy()[:n])[both] * mmpx
            e["dist_old_new_mm"] = float(np.median(dist)) if len(dist) else None
        kp[k] = {kk: vv for kk, vv in e.items() if not kk.endswith("_conf")}
    row["keypoints"] = kp
    kin = {}
    if (ba.NWB / f"{session}.nwb").exists():
        with tempfile.TemporaryDirectory(dir=SHM) as tmp:
            views = {}
            for name, h5 in (("new", new_h5), ("old", old_h5)):
                if h5 is None:
                    continue
                try:
                    p = Path(tmp) / f"{name}.h5"
                    row[f"refs_{name}"] = ba.build_nwb_like(h5, session, view, p)
                    views[name] = ba.load_view(p, view, session)
                except Exception as err:  # noqa: BLE001 (NWB without this camera, etc.)
                    row[f"kin_error_{name}"] = repr(err)
            for q in QUANT[view]:
                vals = {}
                for name, v in views.items():
                    if q not in v.data:
                        continue
                    x = ba.series(v, q) if q != "top_nose_distance" else dh.clean_series(v, q)[0]
                    vals[name] = (np.asarray(x, float), valid_mask(v, q))
                e = {f"pct_valid_{n}": float(100 * vm.mean()) for n, (_, vm) in vals.items()}
                if len(vals) == 2:
                    n = min(len(vals["new"][0]), len(vals["old"][0]))
                    xn, xo = vals["new"][0][:n], vals["old"][0][:n]
                    ok = vals["new"][1][:n] & vals["old"][1][:n] & np.isfinite(xn) & np.isfinite(xo)
                    if ok.sum() > 100:
                        e.update(r=float(np.corrcoef(xn[ok], xo[ok])[0, 1]),
                                 median_abs_diff=float(np.median(np.abs(xn[ok] - xo[ok]))),
                                 sd_ratio=float(np.std(xn[ok]) / np.std(xo[ok])) if np.std(xo[ok]) > 0 else None)
                kin[q] = e
            if view == "side":
                for name, v in views.items():
                    if "tongue_likelihood" in v.likelihood:
                        kin.setdefault("tongue_presence", {})[f"pct_out_{name}"] = float(
                            100 * (v.likelihood["tongue_likelihood"].to_numpy() > dh.TONGUE_PRESENCE_CUTOFF).mean())
    row["kinematics"] = kin
    dest.write_text(json.dumps(row, indent=1, default=lambda o: None))
    return dest


def jobs(sessions=None, dirs=(STAGE,)):
    """(session, view, new filtered h5) per camera-session found in dirs (first dir wins)."""
    out = {}
    for d in dirs:
        for h5 in sorted(Path(d).glob("*/*_filtered.h5")):
            s = h5.parent.name
            if sessions and s not in sessions:
                continue
            view = "side" if "_side" in h5.name else "top"
            out.setdefault((s, view), h5)
    return [(s, v, h) for (s, v), h in out.items()]


def tables():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "rows").glob("*.json"))]
    kp, kin, ses = [], [], []
    for r in rows:
        base = {k: r.get(k) for k in ("session_id", "mouse_id", "view", "old_shuffle", "n_new", "n_old")}
        ses.append({**base, "has_old": r.get("old_h5") is not None,
                    "kin_error": r.get("kin_error_new") or r.get("kin_error_old")})
        for k, e in r["keypoints"].items():
            d = dict(base, keypoint=k, dist_old_new_mm=e.get("dist_old_new_mm"))
            for name in ("new", "old"):
                for kk, vv in (e.get(name) or {}).items():
                    d[f"{kk}_{name}"] = vv
            kp.append(d)
        for q, e in r["kinematics"].items():
            kin.append(dict(base, quantity=q, **e))
    return pd.DataFrame(kp), pd.DataFrame(kin), pd.DataFrame(ses)


def figures(kp, kin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = ts.ps
    ps.set_nature_rc()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    col = {"old": "#d6604d", "new": ts.MODEL_COLORS[6]}
    fig, axs = plt.subplots(2, 3, figsize=(183 * ps.MM, 110 * ps.MM))
    for j, v in enumerate(ts.VIEWS):
        ax = axs[0, j]
        kps = ts.VIEW_KEYPOINTS[v]
        d = kp[kp.view == v]
        for i, name in enumerate(("old", "new")):
            data = [d[d.keypoint == k][f"pct_confident_{name}"].dropna() for k in kps]
            bp = ax.boxplot(data, positions=np.arange(len(kps)) + (i - 0.5) * 0.35, widths=0.3, patch_artist=True,
                            showfliers=False, medianprops=dict(color="k", lw=0.6))
            for b in bp["boxes"]:
                b.set(facecolor=col[name], lw=0.4)
        ax.set_xticks(range(len(kps)), [k.replace("_", " ") for k in kps], rotation=60, ha="right", fontsize=5)
        ax.set_ylabel("Frames confident (%)"); ax.set_title(f"{v.capitalize()} keypoints, per session", fontsize=6)
        ax.set_ylim(-2, 102)
    ax = axs[0, 2]
    qs = [q for q in ("jaw_distance", "jaw_angle", "pupil_area", "nose_distance", "whisker_angle", "top_nose_distance")
          if q in set(kin.quantity)]
    for i, name in enumerate(("old", "new")):
        data = [kin[kin.quantity == q][f"pct_valid_{name}"].dropna() for q in qs]
        bp = ax.boxplot(data, positions=np.arange(len(qs)) + (i - 0.5) * 0.35, widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k", lw=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=col[name], lw=0.4)
    ax.set_xticks(range(len(qs)), [q.replace("_", " ") for q in qs], rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("Frames valid (%)"); ax.set_title("Kinematics, per session", fontsize=6); ax.set_ylim(-2, 102)
    for j, (c, lab, lim) in enumerate((("r", "r(old, new) per session", (-0.2, 1.02)),
                                       ("sd_ratio", "SD ratio new / old", (0, 3)),
                                       ("median_abs_diff", "Median |new - old|", None))):
        ax = axs[1, j]
        data = [kin[kin.quantity == q][c].dropna() for q in qs]
        ax.boxplot(data, widths=0.5, showfliers=True, flierprops=dict(ms=1.5), medianprops=dict(color="k", lw=0.6))
        ax.set_xticks(range(1, len(qs) + 1), [q.replace("_", " ") for q in qs], rotation=60, ha="right", fontsize=5)
        ax.set_title(lab, fontsize=6)
        if lim:
            ax.set_ylim(*lim)
        if c == "sd_ratio":
            ax.axhline(1, color="0.5", lw=0.5, ls=":")
        if c == "median_abs_diff":
            ax.set_yscale("log")
    axs[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[n]) for n in ("old", "new")],
                     labels=["old (in NWB)", "new (CSPNeXt-M)"], fontsize=5, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"assessment_dataset.{ext}", dpi=200)
    plt.close(fig)
    # per session: gain in valid frames of the main quantity per camera, by old model
    fig, axs = plt.subplots(1, 2, figsize=(183 * ps.MM, 60 * ps.MM))
    for ax, (v, q) in zip(axs, (("side", "jaw_distance"), ("top", "whisker_angle"))):
        d = kin[(kin.view == v) & (kin.quantity == q)].dropna(subset=["pct_valid_old", "pct_valid_new"])
        for s, c in ((3, "#1f77b4"), (5, "#ff7f0e"), (6, "#2ca02c")):
            g = d[d.old_shuffle == s]
            ax.scatter(g.pct_valid_old, g.pct_valid_new, s=5, color=c, edgecolors="none", label=f"old s{s} (n={len(g)})")
        ax.plot([0, 100], [0, 100], color="0.5", lw=0.5, ls=":")
        ax.set_xlabel(f"Old: {q.replace('_', ' ')} valid (%)"); ax.set_ylabel("New: valid (%)")
        ax.set_title(f"{v} camera, one dot per session", fontsize=6); ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"assessment_sessions.{ext}", dpi=200)
    plt.close(fig)


def summary(kp, kin, ses):
    lines = ["# Re-inference assessment (CSPNeXt-M snapshot-330 vs old DLC)", "",
             f"{len(ses)} camera-sessions ({ses.session_id.nunique()} sessions); without old DLC: "
             f"{int((~ses.has_old).sum())}; kinematics errors: {int(ses.kin_error.notna().sum())}.", "",
             "## Keypoints (median over sessions)", "",
             kp.groupby(["view", "keypoint"])[["pct_confident_old", "pct_confident_new", "pct_jumps_old",
                                               "pct_jumps_new", "dist_old_new_mm"]].median().round(2).to_string(), "",
             "## Kinematics (median over sessions)", "",
             kin.groupby(["view", "quantity"])[[c for c in ("pct_valid_old", "pct_valid_new", "r", "sd_ratio",
                                                            "median_abs_diff", "pct_out_old", "pct_out_new")
                                                if c in kin]].median().round(3).to_string(), "",
             "## By old model (median over sessions)", "",
             kin.groupby(["quantity", "old_shuffle"])[["pct_valid_old", "pct_valid_new", "r"]].median().round(3)
             .to_string(), "",
             "## Sessions that change most (lowest r, main quantity per camera)", ""]
    main = kin[((kin.view == "side") & (kin.quantity == "jaw_distance")) | ((kin.view == "top") & (kin.quantity == "whisker_angle"))]
    lines.append(main.sort_values("r").head(25)[["session_id", "view", "quantity", "old_shuffle", "pct_valid_old",
                                                  "pct_valid_new", "r", "sd_ratio"]].round(3).to_string(index=False))
    (OUT / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--source-dirs", nargs="*", default=[str(STAGE)],
                    help="folders of <session>/*_filtered.h5 (e.g. the pilot outputs, same model and snapshot)")
    a = ap.parse_args()
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    js = jobs(a.sessions, a.source_dirs)
    print(len(js), "camera-sessions staged", flush=True)
    with Pool(a.jobs) as p:
        for i, d in enumerate(p.imap_unordered(assess, js, chunksize=1)):
            if i % 25 == 0:
                print(i, d, flush=True)
    kp, kin, ses = tables()
    kp.to_csv(OUT / "keypoints.csv", index=False)
    kin.to_csv(OUT / "kinematics.csv", index=False)
    ses.to_csv(OUT / "sessions.csv", index=False)
    figures(kp, kin)
    summary(kp, kin, ses)
    print((OUT / "summary.md").read_text()[:4000])


if __name__ == "__main__":
    main()
